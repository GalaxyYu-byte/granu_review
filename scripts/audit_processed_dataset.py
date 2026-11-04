"""独立复核清洗后的记录、文本编辑和隔离副本，不修改任何数据集。

先核对原始校验、复核决定与执行日志之间的凭据关系，再重新读取所有
原始及处理后记录逐项对账。凭据摘要相符仅表示文件对应，不能代替内容核验。
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import sys
import unicodedata

try:
    from .dataset_io import (file_sha256, read_jsonl, receipt_origin, receipt_path,
                             resolve_receipt_path, write_json_atomic)
    from .label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS
    from .review_evidence import validate_review_evidence
    from .review_validation import match_review_approvals, validation_options
    from .validate_dataset import validate_dataset
except ImportError:
    from dataset_io import (file_sha256, read_jsonl, receipt_origin, receipt_path,
                            resolve_receipt_path, write_json_atomic)
    from label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS
    from review_evidence import validate_review_evidence
    from review_validation import match_review_approvals, validation_options
    from validate_dataset import validate_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILENAMES = {"train": "sentiment_analysis_trainingset.csv",
             "validation": "sentiment_analysis_validationset.csv"}


def _text_sha(text: str) -> str:
    """对解析后的文本计算摘要，避免把 CSV 转义方式差异误认为正文变化。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _references(records: list[dict], message: str) -> dict:
    """以集合和原始记录序号建立身份索引，同时拒绝无效或重复引用。

    不直接用字典推导式折叠重复项，否则一份重复引用可能伪装成完整且唯一的计划。
    ID 单独保存用于交叉核对；不同集合允许使用相同 ID。
    """
    result = {}
    for record in records:
        key = record["split"], record["record_number"]
        _check(key[0] in FILENAMES and type(key[1]) is int and key[1] > 0
               and isinstance(record["id"], str) and key not in result, message)
        result[key] = record["id"]
    return result


def _record_signature(records: list[dict]) -> list[tuple]:
    """生成包含集合、ID、全文和全部标签的组签名，保留重复成员的数量。

    隔离会改变输出中的记录序号，因此组匹配不依赖序号；全文和标签必须逐项一致，
    不能仅凭问题编号相同或成员 ID 相同就继承先前的复核批准。
    """
    return sorted((record["split"], record["id"], record["text"],
                   tuple(sorted(record["labels"].items()))) for record in records)


def _resolve_fresh_reviews(validation: dict, fresh_issues: list[dict],
                           previous_issues: list[dict], previous_resolutions: list[dict]) -> None:
    """仅将已经通过来源审计的保留批准传递给证据完全相同的新扫描问题。

    此函数由命令行流程在完整审计之后调用。新出现或成员、文本、标签发生变化的
    待复核问题必须报错；批准保留也不能消除扫描错误或隐藏原始扫描状态。
    """
    previous_by_id = {issue["issue_id"]: issue for issue in previous_issues}
    approved = [(previous_by_id[item["issue_id"]], item["source_issue_ids"])
                for item in previous_resolutions]
    resolutions = []
    for issue in fresh_issues:
        if issue["severity"] != "REVIEW":
            continue
        source_ids = {source_id for original, ids in approved
                      if original["rule_id"] == issue["rule_id"]
                      and _record_signature(original["records"]) == _record_signature(issue["records"])
                      for source_id in ids}
        _check(bool(source_ids), "重新校验发现未获保留批准的 REVIEW。")
        resolutions.append({"issue_id": issue["issue_id"], "source_issue_ids": sorted(source_ids)})
    if resolutions:
        # 同时保存扫描原始状态和批准来源，让“已复核保留”与“没有发现问题”可区分。
        validation["scan_status"] = validation["status"]
        validation["review_resolutions"] = resolutions
        validation["pending_reviews"] = 0
        if not validation["issue_counts"]["ERROR"]:
            validation["status"] = "passed_with_warnings"
            validation["validation_gate_passed"] = True


@dataclass(slots=True)
class _AuditBatch:
    """所选批次的凭据、实际文件角色及审计期间受保护的文件摘要。"""

    root: Path
    processing_path: Path
    processing: dict
    protected: dict[Path, str]
    output_paths: dict[str, Path]
    report_path: Path
    source_report: dict
    summary_path: Path
    summary: dict
    raw_paths: dict[str, Path]
    issues_path: Path
    decisions_path: Path
    source_origin: str | None
    processed_paths: dict[str, Path]
    post_validation: dict
    post_origin: str | None


@dataclass(slots=True)
class _ReviewedPlan:
    """从原始问题与复核决定独立核验出的隔离、编辑计划及对应日志。"""

    issues: list[dict]
    decisions: list[dict]
    changes: list[dict]
    applications: list[dict]
    by_issue: dict[str, dict]
    applications_by_issue: dict[str, dict]
    quarantine: dict
    quarantine_sources: dict
    edits_plan: dict
    expected_records: dict
    edited_texts: dict


@dataclass(slots=True)
class _RecordAudit:
    """逐条读取 CSV 后复算的数量、精确编辑证据和验证标签争议证据。"""

    raw_counts: Counter
    retained_counts: Counter
    unchanged_counts: Counter
    removed_chars: Counter
    quarantined_rows: list[list[str]]
    edit_evidence: list[dict]
    dispute_evidence: list[dict]
    dispute_ids: set[str]
    label_cells: int


def _load_audit_batch(root: Path, processing_report: Path | None) -> _AuditBatch:
    """读取所选处理凭据，校验文件摘要并按交叉引用唯一绑定全部来源角色。"""
    processing_path = resolve_receipt_path(str(processing_report), root=root) if processing_report is not None else root / "data/reports/processed/processing_report.json"
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    # 通过已知报告的完整项目内后缀绑定旧项目根目录，避免仅凭 data 目录名猜测路径。
    # 副本迁移后，即使旧项目仍存在，也应读取当前 root 下的对应文件。
    validation_name = PureWindowsPath(processing["outputs"]["validation_report"]["path"]).name
    processing_origin = receipt_origin(processing["outputs"]["validation_report"]["path"],
                                       processing_path.parent / validation_name, root)
    protected = {processing_path: file_sha256(processing_path)}
    # 记录本次实际读取文件的摘要，既校验处理凭据，也用于结束时发现并发修改。
    input_paths = set()
    for name, expected in processing["inputs"].items():
        path = resolve_receipt_path(name, root=root, legacy_root=processing_origin)
        _check(path not in input_paths, "处理凭据的输入路径重复。")
        input_paths.add(path)
        protected[path] = file_sha256(path)
        _check(protected[path] == expected, f"输入摘要不匹配：{path}")
    output_paths = {}
    for name, artifact in processing["outputs"].items():
        path = resolve_receipt_path(artifact["path"], root=root, legacy_root=processing_origin)
        _check(path not in protected, "处理凭据的产物路径重复或覆盖输入。")
        output_paths[name] = path
        protected[path] = file_sha256(path)
        _check(protected[path] == artifact["sha256"], f"处理产物摘要不匹配：{path}")

    # 批次可以使用独立目录。来源角色从已绑定的文件内容和引用关系确定，而不是根据文件名
    # 猜测，更不能在找不到文件时回退默认批次；每种角色最终都必须占用一个独立输入。
    _check(len(input_paths) == 6 and set(output_paths) == {
        "train", "validation", "quarantine", "changes", "applications", "validation_report", "validation_issues"},
        "处理凭据必须绑定完整的六个输入和七个产物。")
    objects = {}
    for path in input_paths:
        with path.open(encoding="utf-8-sig") as stream:
            first = stream.read(1)
            while first and first.isspace():
                first = stream.read(1)
            if first != "{":
                continue
            stream.seek(0)
            try:
                value = json.load(stream)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                objects[path] = value
    reports = [(path, value) for path, value in objects.items()
               if {"stage", "files", "rules_version", "artifacts", "issue_counts"} <= value.keys()]
    summaries = [(path, value) for path, value in objects.items()
                 if {"source_report_sha256", "source_issues_sha256", "source_sha256", "dry_run", "artifacts"} <= value.keys()]
    _check(len(reports) == len(summaries) == 1, "处理凭据无法唯一定位原始校验报告与复核汇总。")
    report_path, source_report = reports[0]
    summary_path, summary = summaries[0]
    source_origin = receipt_origin(source_report["artifacts"]["report"], report_path, root)
    _check(resolve_receipt_path(source_report["artifacts"]["report"], root=root, legacy_root=source_origin) == report_path,
           "原始校验报告的自引用路径不符。")
    raw_paths = {split: resolve_receipt_path(info["path"], root=root, legacy_root=source_origin)
                 for split, info in source_report["files"].items()}
    issues_path = resolve_receipt_path(source_report["artifacts"]["issues"], root=root, legacy_root=source_origin)
    # 无问题批次的问题文件和决定文件都可能为空、摘要相同。先用完整路径排除已知角色，
    # 再核对剩余决定文件摘要，不能将摘要相同误当成文件身份相同。
    decision_candidates = list(input_paths - {report_path, summary_path, issues_path, *raw_paths.values()})
    _check(len(decision_candidates) == 1
           and protected[decision_candidates[0]] == summary["artifacts"]["decisions_sha256"],
           "复核汇总无法唯一绑定决定文件。")
    decisions_path = decision_candidates[0]
    required_inputs = {report_path, summary_path, issues_path, decisions_path, *raw_paths.values()}
    _check(len(required_inputs) == 6 and input_paths == required_inputs,
           "处理凭据未完整绑定原始数据与复核文件。")
    expected_roles = {"raw_report": report_path, "review_summary": summary_path,
                      "issues": issues_path, "decisions": decisions_path, **raw_paths}
    if "input_roles" in processing:
        declared_roles = {name: resolve_receipt_path(value, root=root, legacy_root=processing_origin)
                          for name, value in processing["input_roles"].items()}
        _check(declared_roles == expected_roles, "处理凭据的输入角色与原始交叉引用不一致。")
    processed_paths = {split: output_paths[split] for split in FILENAMES}
    post_validation = json.loads(output_paths["validation_report"].read_text(encoding="utf-8"))
    post_origin = receipt_origin(post_validation["artifacts"]["report"], output_paths["validation_report"], root)
    return _AuditBatch(
        root=root, processing_path=processing_path, processing=processing,
        protected=protected, output_paths=output_paths, report_path=report_path,
        source_report=source_report, summary_path=summary_path, summary=summary,
        raw_paths=raw_paths, issues_path=issues_path, decisions_path=decisions_path,
        source_origin=source_origin, processed_paths=processed_paths, post_validation=post_validation,
        post_origin=post_origin,
    )


def _validate_receipt_chain(batch: _AuditBatch) -> tuple[dict[str, str], dict]:
    """核对原始报告、复核汇总和二次校验之间的状态、摘要链与扫描口径。"""
    # 分别验证“汇总引用校验报告和决定”“校验报告引用问题明细”的摘要链。
    # 这样即便有人更新了最外层处理凭据中的摘要，内部互相矛盾的材料仍无法通过。
    _check(batch.summary.get("review_complete") is True, "复核尚未完成。")
    _check(batch.processing.get("stage") == "cleaned_original_label_space" and batch.processing.get("mapping_applied") is False
           and all(batch.processing.get(field) is True for field in (
               "raw_data_unchanged", "original_reports_and_decisions_unchanged", "labels_unchanged",
               "ids_preserved", "source_order_preserved", "all_retained_records_audited_against_source")),
           "处理报告的映射或来源对账状态不一致。")
    _check(batch.summary["source_report_sha256"] == batch.protected[batch.report_path]
           and batch.summary["source_issues_sha256"] == batch.source_report["artifacts"]["issues_sha256"] == batch.protected[batch.issues_path]
           and batch.summary["artifacts"]["decisions_sha256"] == batch.protected[batch.decisions_path],
           "复核汇总、原始问题与决定的摘要关系不一致。")
    _check(batch.source_report["stage"] == "raw" and batch.source_report["scan_complete"] is True
           and not batch.source_report["issue_counts"]["ERROR"] and set(batch.source_report["files"]) == set(FILENAMES),
           "需要完整、无 ERROR 的原始校验报告。")
    _check(resolve_receipt_path(batch.source_report["artifacts"]["issues"], root=batch.root, legacy_root=batch.source_origin) == batch.issues_path,
           "原始校验报告的明细路径不符。")
    if "decisions" in batch.summary["artifacts"]:
        _check(resolve_receipt_path(batch.summary["artifacts"]["decisions"], root=batch.root, legacy_root=batch.source_origin) == batch.decisions_path,
               "复核汇总的决定路径不符。")
    source_hashes = {split: batch.protected[path] for split, path in batch.raw_paths.items()}
    _check(batch.summary["source_sha256"] == source_hashes, "复核汇总的原始数据摘要不符。")
    for split, path in batch.raw_paths.items():
        _check(batch.source_report["files"][split]["sha256"] == batch.protected[path]
               and resolve_receipt_path(batch.source_report["files"][split]["path"], root=batch.root, legacy_root=batch.source_origin) == path,
               "原始校验报告的数据来源不符。")
    _check(batch.post_validation["scan_complete"] is True
           and batch.post_validation["stage"] == "raw"
           and batch.post_validation["status"] in {"passed", "passed_with_warnings"}
           and batch.post_validation["status"] == batch.processing["post_validation_status"]
           and batch.post_validation["issue_counts"] == batch.processing["post_validation_issue_counts"]
           and not batch.post_validation["issue_counts"]["ERROR"], "二次校验状态与处理凭据不一致。")
    _check(resolve_receipt_path(batch.post_validation["artifacts"]["issues"], root=batch.root, legacy_root=batch.post_origin) == batch.output_paths["validation_issues"]
           and batch.post_validation["artifacts"]["issues_sha256"] == batch.protected[batch.output_paths["validation_issues"]],
           "二次校验明细与处理凭据不一致。")
    for split, path in batch.processed_paths.items():
        _check(resolve_receipt_path(batch.post_validation["files"][split]["path"], root=batch.root, legacy_root=batch.post_origin) == path
               and batch.post_validation["files"][split]["sha256"] == batch.protected[path]
               and batch.post_validation["files"][split]["records_scanned"] == batch.processing["processed_counts"][split],
               "二次校验的数据来源或条数不一致。")
    scan_options = validation_options(batch.source_report)
    _check(validation_options(batch.post_validation) == scan_options, "二次校验配置未继承原始校验口径。")
    return source_hashes, scan_options


def _audit_review_plan(batch: _AuditBatch, source_hashes: dict[str, str]) -> _ReviewedPlan:
    """独立检查问题、决定与应用日志，并从原始证据重建完整处理计划。"""
    # JSONL 按文件实际换行读取；正文中的 Unicode 分隔字符仍属于同一条证据。
    issues = read_jsonl(batch.issues_path)
    decisions = read_jsonl(batch.decisions_path)
    changes = read_jsonl(batch.output_paths["changes"])
    applications = read_jsonl(batch.output_paths["applications"])
    _check(len({item["issue_id"] for item in issues}) == len(issues), "原始问题编号重复。")
    _check(len({item["issue_id"] for item in decisions}) == len(decisions), "复核决定编号重复。")
    _check({item["issue_id"] for item in issues} == {item["issue_id"] for item in decisions}
           == {item["issue_id"] for item in applications}, "问题、决定、应用日志未完整对应。")
    _check(len(applications) == len(decisions), "应用日志条数不符。")
    by_issue = {item["issue_id"]: item for item in issues}
    applications_by_issue = {item["issue_id"]: item for item in applications}
    issue_counts = {severity: sum(item["severity"] == severity for item in issues)
                    for severity in ("ERROR", "REVIEW", "WARN")}
    _check(issue_counts == batch.source_report["issue_counts"], "原始问题数量与校验报告不一致。")
    if "reviewed_issue_count" in batch.summary:
        _check(batch.summary["reviewed_issue_count"] == len(decisions), "复核汇总的问题数量不符。")
    if "reviewed_issue_counts" in batch.summary:
        _check(batch.summary["reviewed_issue_counts"] == {key: value for key, value in issue_counts.items() if value},
               "复核汇总的问题级别数量不符。")
    if "action_issue_counts" in batch.summary:
        _check(batch.summary["action_issue_counts"] == dict(Counter(item["action"]["type"] for item in decisions)),
               "复核汇总的动作数量不符。")
    quarantine = {}
    quarantine_sources = {}
    edits_plan = {}
    expected_records = {}
    for decision in decisions:
        issue = by_issue[decision["issue_id"]]
        references = _references(issue["records"], "原始问题包含无效或重复样本。")
        affected = _references(decision["affected_records"], "复核决定包含无效或重复样本。")
        _check(references and affected == references, "决定引用的样本与原始问题不一致。")
        _check(issue["stage"] == "raw" and issue["rules_version"] == batch.source_report["rules_version"]
               and issue["source_sha256"] == {split: source_hashes[split] for split, _ in references},
               "原始问题的规则版本或来源摘要不符。")
        _check(decision.get("status") == "reviewed_pending_application"
               and decision.get("action_applied") is False and decision.get("label_changes") == [],
               "复核状态不允许应用，或包含未授权改标。")
        _check(decision["source_sha256"] == issue["source_sha256"]
               and decision["rule_id"] == issue["rule_id"]
               and decision["original_severity"] == issue["severity"], "决定的来源或规则与原始问题不一致。")
        validate_review_evidence(decision, issue["records"])
        for record in issue["records"]:
            key = record["split"], record["record_number"]
            _check(key not in expected_records or expected_records[key] == record, "同一原始记录的证据不一致。")
            expected_records[key] = record
        action = decision["action"]
        # 冲突摘要也是待核验材料：必须从成员的全部标签重新计算，防止清空摘要后隐去验证争议，
        # 或将训练冲突伪装成可以原样保留的同标签重复。
        if issue["rule_id"] in {"within_split_exact_duplicate", "within_split_normalized_duplicate",
                                "cross_split_exact_overlap", "cross_split_normalized_overlap"}:
            actual_conflicts = {
                aspect: [{**{field: record[field] for field in ("split", "id", "record_number")},
                          "value": record["labels"][aspect]} for record in issue["records"]]
                for aspect in ASPECT_COLUMNS
                if len({record["labels"][aspect] for record in issue["records"]}) > 1
            }
            _check(issue.get("label_conflicts") == actual_conflicts, "重复组的标签冲突明细与记录不一致。")
        application = applications_by_issue[decision["issue_id"]]
        # 问题编号对应还不够：应用日志还必须证明同一来源、同一目标、同一动作确已应用。
        # “排除训练风险”与“修复底层标注或原始缺字”含义不同，后者不能被日志擅自宣称完成。
        _check(application["action_type"] == action["type"]
               and application["source_sha256"] == decision["source_sha256"]
               and _references(application["affected_records"], "应用日志包含无效或重复样本。") == affected
               and application.get("label_changes") == []
               and application["post_validation_status"] == batch.processing["post_validation_status"]
               and application.get("underlying_annotation_or_source_repair_completed") is False
               and application["applied_at"] == batch.processing["applied_at"],
               "应用日志的状态、动作、来源或目标与复核决定不一致。")
        if action["type"] in {"quarantine_training_conflict_cluster", "quarantine_training_overlap",
                              "quarantine_corrupted_training_record"}:
            # 隔离只允许覆盖该问题的训练成员，固定验证样本必须保留，不能按处理结果反推目标。
            quarantine_refs = _references(action["quarantine"], "隔离计划包含无效或重复样本。")
            expected = {key: value for key, value in references.items() if key[0] == "train"}
            _check(bool(expected) and quarantine_refs == expected, "隔离目标必须且只能覆盖对应问题的训练成员。")
            if action["type"] == "quarantine_corrupted_training_record":
                _check(issue["rule_id"] == "suspicious_characters" and len(references) == 1,
                       "损坏隔离动作与原始问题类型不符。")
            elif action["type"] == "quarantine_training_overlap":
                _check(issue["rule_id"] in {"cross_split_exact_overlap", "cross_split_normalized_overlap"}
                       and issue["severity"] == "REVIEW"
                       and {key[0] for key in references} == {"train", "validation"}, "重叠隔离动作与原始问题不符。")
            else:
                _check(issue["rule_id"] in {"within_split_exact_duplicate", "within_split_normalized_duplicate"}
                       and issue["severity"] == "REVIEW" and {key[0] for key in references} == {"train"}
                       and bool(issue.get("label_conflicts")), "冲突隔离动作与原始问题不符。")
            validation_refs = {key: value for key, value in references.items() if key[0] == "validation"}
            _check(_references(action.get("preserve_fixed_validation", []), "验证保留计划包含重复样本。") == validation_refs,
                   "固定验证集保留目标与原始问题不一致。")
            for record in action["quarantine"]:
                _check(record["split"] == "train", "固定验证集不能被隔离。")
                key = record["split"], record["record_number"]
                if key in quarantine:
                    _check(quarantine[key]["id"] == record["id"], "隔离记录身份冲突。")
                quarantine[key] = record
                quarantine_sources.setdefault(key, []).append(decision["issue_id"])
        elif action["type"] == "remove_reviewed_control_characters":
            # 删除集合必须来自对应字符告警；普通文字、表情及正常制表符或换行均不在授权范围。
            # 从原始证据自行生成候选正文，再核对数量、长度和摘要，不信任决定自报的候选结果。
            record = action["target"]
            key = record["split"], record["record_number"]
            _check(issue["rule_id"] == "suspicious_characters" and len(references) == 1
                   and references == _references([record], "清洗目标无效。")
                   and key not in edits_plan, "清洗目标与原始问题不符或存在多项编辑。")
            codes = action["remove_code_points"]
            _check(codes and len(codes) == len(set(codes)) and set(codes) == set(issue["characters"])
                   and all(re.fullmatch(r"U\+[0-9A-F]{4,6}", value) for value in codes),
                   "清洗字符与原始告警不一致。")
            _check(all(int(value[2:], 16) <= 0x10FFFF for value in codes), "控制字符编码不合法。")
            chars = {chr(int(value[2:], 16)) for value in codes}
            _check(bool(chars) and all(unicodedata.category(char) == "Cc" and char not in "\t\r\n"
                                      for char in chars) and action.get("replacement", "") == "",
                   "计划包含普通字符或正常换行/制表符。")
            original = issue["records"][0]["text"]
            candidate = "".join(char for char in original if char not in chars)
            _check({value: original.count(chr(int(value[2:], 16))) for value in codes} == issue["characters"]
                   and len(original) == action["original_length"] and len(candidate) == action["candidate_length"]
                   and _text_sha(candidate) == action["candidate_text_sha256"] and bool(candidate.strip()),
                   "字符数量或清洗后长度/摘要不符合复核决定。")
            edits_plan[key] = {"id": record["id"], "characters": chars, "issue_id": decision["issue_id"]}
        elif action["type"] == "retain_reviewed_records":
            # 同标签训练重复可以明确保留；验证集即使标签冲突也保留原记录，另行报告争议。
            _check(issue["rule_id"] in {"within_split_exact_duplicate", "within_split_normalized_duplicate"}
                   and len({key[0] for key in references}) == 1
                   and (all(key[0] == "validation" for key in references)
                        or (not issue.get("label_conflicts")
                            and all(len({record["labels"][aspect] for record in issue["records"]}) == 1 for aspect in ASPECT_COLUMNS)))
                   and not action.get("quarantine") and not action.get("target"), "保留动作与集合内重复问题不一致。")
        else:
            raise ValueError(f"未知处理动作：{action['type']}")
    _check(not quarantine.keys() & edits_plan.keys(), "隔离与编辑目标相交。")
    edited_texts = {key: "".join(char for char in expected_records[key]["text"] if char not in plan["characters"])
                    for key, plan in edits_plan.items()}
    return _ReviewedPlan(
        issues=issues, decisions=decisions, changes=changes,
        applications=applications, by_issue=by_issue, applications_by_issue=applications_by_issue,
        quarantine=quarantine, quarantine_sources=quarantine_sources, edits_plan=edits_plan,
        expected_records=expected_records, edited_texts=edited_texts,
    )


def _audit_application_outcomes(batch: _AuditBatch, review: _ReviewedPlan) -> None:
    """依据完整计划重算每项决定的生效范围及记录去向，核对新旧应用日志。"""
    # 从完整动作计划独立推导最终去向。组级保留建议可能被另一个问题的隔离或字符清洗覆盖，
    # 因而必须在收集完所有决定后才判断每项建议实际生效了哪些成员。
    log_version = batch.processing.get("application_log_version")
    _check(log_version in {None, "1.0", "2.0"}, "应用日志版本不受支持。")
    if log_version == "2.0":
        _check("input_roles" in batch.processing, "新版处理凭据缺少输入角色。")
    else:
        # 删除版本字段不能让新版逐条字段逃过校验。真正旧版没有这些字段，仍可正常复算。
        _check("record_outcomes" not in batch.processing
               and all("record_results" not in item and "application_status" not in item for item in review.applications),
               "应用日志版本与新版逐条字段混用。")
    effective = {key: set() for key in review.expected_records}
    for decision in review.decisions:
        action_type = decision["action"]["type"]
        record_results = []
        for record in decision["affected_records"]:
            key = record["split"], record["record_number"]
            overridden = (sorted(review.quarantine_sources[key]) if key in review.quarantine else [review.edits_plan[key]["issue_id"]]
                          if key in review.edits_plan else []) if action_type == "retain_reviewed_records" else []
            applied = not overridden
            if applied:
                effective[key].add(decision["issue_id"])
            record_results.append({
                **{field: record[field] for field in ("split", "id", "record_number")},
                "disposition": "quarantined" if key in review.quarantine else "retained",
                "text_modified": key in review.edits_plan, "decision_applied": applied,
                "overridden_by_issue_ids": overridden,
            })
        applied_count = sum(item["decision_applied"] for item in record_results)
        status = "applied" if applied_count == len(record_results) else "partially_overridden" if applied_count else "overridden"
        resolution = ("superseded_by_other_reviewed_actions" if status == "overridden"
                      else "partially_superseded_by_other_reviewed_actions" if status == "partially_overridden"
                      else "control_characters_removed_in_derived_copy" if action_type == "remove_reviewed_control_characters"
                      else "reviewed_records_retained_unchanged" if action_type == "retain_reviewed_records"
                      else "excluded_from_training_and_preserved_in_quarantine")
        application = review.applications_by_issue[decision["issue_id"]]
        _check(application.get("action_applied") is bool(applied_count)
               and application.get("resolution") == resolution, "应用日志未如实记录决定生效状态。")
        if log_version == "2.0":
            actual_results = application.get("record_results")
            _check(isinstance(actual_results, list) and actual_results == record_results
                   and all(type(item.get("text_modified")) is bool and type(item.get("decision_applied")) is bool
                           for item in actual_results)
                   and application.get("application_status") == status, "应用日志逐条生效范围与完整计划不一致。")
        else:
            # 老版日志没有逐条覆盖说明，只有全部成员实际生效时才能接受其组级结论。
            _check(status == "applied", "旧版应用日志不能证明被其他动作覆盖的保留建议已生效。")
    expected_outcomes = [{
        "split": split, "id": review.expected_records[(split, number)]["id"], "source_record_number": number,
        "disposition": "quarantined" if (split, number) in review.quarantine else "retained",
        "text_modified": (split, number) in review.edits_plan, "effective_issue_ids": sorted(effective[(split, number)]),
    } for split, number in sorted(review.expected_records)]
    if log_version == "2.0":
        actual_outcomes = batch.processing.get("record_outcomes")
        _check(isinstance(actual_outcomes, list) and actual_outcomes == expected_outcomes
               and all(type(item.get("text_modified")) is bool for item in actual_outcomes),
               "处理报告的逐条最终去向与完整计划不一致。")


def _audit_post_validation(batch: _AuditBatch, review: _ReviewedPlan) -> None:
    """逐项核对二次校验问题，确保每个放行的重复组均有精确匹配的复核批准。"""
    post_issues = read_jsonl(batch.output_paths["validation_issues"])
    _check(len({item["issue_id"] for item in post_issues}) == len(post_issues), "二次校验问题编号重复。")
    post_counts = {severity: sum(item["severity"] == severity for item in post_issues)
                   for severity in ("ERROR", "REVIEW", "WARN")}
    _check(post_counts == batch.post_validation["issue_counts"], "二次校验问题数量不符。")
    post_reviews = {item["issue_id"]: item for item in post_issues if item["severity"] == "REVIEW"}
    resolutions = batch.post_validation.get("review_resolutions", [])
    # 二次扫描留下的每个待复核组，都必须逐项追溯到原始“保留”决定；不能整体放宽校验门槛。
    _check(len(resolutions) == len(post_reviews)
           and {item["issue_id"] for item in resolutions} == set(post_reviews), "二次校验的复核决定未完整对应。")
    for resolution in resolutions:
        issue = post_reviews[resolution["issue_id"]]
        source_ids = resolution["source_issue_ids"]
        expected_ids = set(match_review_approvals(issue, review.issues, review.decisions, review.quarantine.keys(), review.edited_texts))
        _check(source_ids and len(source_ids) == len(set(source_ids)) and set(source_ids) == expected_ids,
               "二次校验放行的问题不符合已复核保留证据。")
    if post_reviews:
        _check(batch.post_validation.get("scan_status") == "needs_review"
               and batch.post_validation["status"] == "passed_with_warnings"
               and batch.post_validation.get("pending_reviews") == 0
               and batch.post_validation.get("validation_gate_passed") is True,
               "二次校验的已复核保留状态不符。")


def _audit_text_edit(batch: _AuditBatch, review: _ReviewedPlan, key: tuple[str, int],
                     original: list[str], candidate: list[str], change_logs: dict) -> dict:
    """从实际文本重算删除位置、字符数量、长度和摘要，返回单条精确编辑证据。"""
    split, number = key
    plan = review.edits_plan[key]
    _check(original[0] == plan["id"], "编辑计划 ID 不符。")
    evidence_record = next(record for record in review.by_issue[plan["issue_id"]]["records"]
                           if (record["split"], record["record_number"]) == key)
    _check(evidence_record["text"] == original[1], "原始文本与复核证据不一致。")
    positions = []
    counts = Counter()
    # 删除位置采用原始解析文本的 Unicode 字符序号，并保留前后文供人工复核。
    for position, char in enumerate(original[1]):
        if char in plan["characters"]:
            code = f"U+{ord(char):04X}"
            counts[code] += 1
            positions.append({"original_position": position, "code_point": code,
                              "python_isspace": char.isspace(),
                              "context_before": original[1][max(0, position - 32):position],
                              "context_after": original[1][position + 1:position + 33]})
    log = change_logs[key]
    # 从实际差异重算日志中的字符数、长度和摘要，防止日志只记对问题编号。
    _check(log["id"] == original[0] and log["issue_id"] == plan["issue_id"]
           and log["source_sha256"] == batch.protected[batch.raw_paths[split]]
           and log["original_text_sha256"] == _text_sha(original[1])
           and log["processed_text_sha256"] == _text_sha(candidate[1])
           and log["original_length"] == len(original[1])
           and log["processed_length"] == len(candidate[1])
           and log["removed_characters"] == dict(counts) and log["labels_changed"] is False,
           "实际编辑与修改日志不符。")
    return {"split": split, "id": original[0], "source_record_number": number,
            "original_text": original[1], "processed_text": candidate[1],
            "original_text_sha256": _text_sha(original[1]),
            "processed_text_sha256": _text_sha(candidate[1]),
            "removed_characters": dict(counts), "deletions": positions}


def _audit_csv_records(batch: _AuditBatch, review: _ReviewedPlan) -> _RecordAudit:
    """顺序读取全部原始与处理后 CSV，核对保留记录、编辑内容和完整隔离副本。"""
    log_by_key = {(item["split"], item["source_record_number"]): item for item in review.changes}
    _check(len(log_by_key) == len(review.changes) and log_by_key.keys() == review.edits_plan.keys(), "修改日志与计划不对应。")
    raw_counts, retained_counts, unchanged_counts = Counter(), Counter(), Counter()
    removed_chars = Counter()
    quarantined_rows = []
    edit_evidence, dispute_evidence = [], []
    dispute_ids = {item["id"] for item in batch.processing["known_validation_label_disputes"]}
    # 从原始冲突证据重新推导争议集合，避免处理报告漏记争议或给同标签重叠虚构争议。
    expected_disputes = {(record["split"], record["record_number"]): record["id"]
                         for issue in review.issues if issue.get("label_conflicts")
                         for record in issue["records"] if record["split"] == "validation"}
    _check(_references(batch.processing["known_validation_label_disputes"], "验证标签争议记录无效或重复。") == expected_disputes,
           "处理报告的验证标签争议与原始证据不一致。")
    seen_edits, seen_quarantine, seen_disputes, seen_evidence = set(), set(), set(), set()
    label_cells = 0
    for split in FILENAMES:
        # 使用 CSV 解析后的逻辑记录比较，评论中的换行不等于新增样本。
        # 原始序号只定位来源；跳过隔离记录后顺序消费输出，才能发现错行、漏行或多行。
        with batch.raw_paths[split].open(encoding="utf-8-sig", newline="") as source_stream, \
                batch.processed_paths[split].open(encoding="utf-8-sig", newline="") as result_stream:
            source_reader = csv.reader(source_stream, strict=True)
            result_reader = csv.reader(result_stream, strict=True)
            _check(next(source_reader) == next(result_reader) == list(EXPECTED_COLUMNS), "表头不符。")
            for number, original in enumerate(source_reader, start=1):
                raw_counts[split] += 1
                _check(len(original) == len(EXPECTED_COLUMNS), f"源记录列宽异常：{split}/{number}")
                key = split, number
                if key in review.expected_records:
                    evidence = review.expected_records[key]
                    _check(evidence["id"] == original[0] and evidence["text"] == original[1]
                           and evidence["labels"] == dict(zip(ASPECT_COLUMNS, map(int, original[2:]))),
                           "原始记录与问题证据不一致。")
                    seen_evidence.add(key)
                if key in review.quarantine:
                    _check(original[0] == review.quarantine[key]["id"], "隔离计划 ID 不符。")
                    quarantined_rows.append(original)
                    seen_quarantine.add(key)
                    continue
                candidate = next(result_reader, None)
                _check(candidate is not None and len(candidate) == len(EXPECTED_COLUMNS), f"缺少或损坏输出：{split}/{number}")
                _check(candidate[0] == original[0], f"保留记录 ID 或顺序改变：{split}/{number}")
                _check(candidate[2:] == original[2:], f"标签被修改：{split}/{number}")
                # 保留记录的全部标签字符串必须原样相同，合法标签域内的改动也不能放过。
                label_cells += len(ASPECT_COLUMNS)
                retained_counts[split] += 1
                plan = review.edits_plan.get(key)
                expected = "".join(char for char in original[1] if char not in plan["characters"]) if plan else original[1]
                _check(candidate[1] == expected, f"发现计划外文本修改：{split}/{number}")
                if plan:
                    evidence = _audit_text_edit(batch, review, key, original, candidate, log_by_key)
                    removed_chars.update(evidence["removed_characters"])
                    edit_evidence.append(evidence)
                    seen_edits.add(key)
                else:
                    unchanged_counts[split] += 1
                if split == "validation" and original[0] in dispute_ids:
                    groups = [issue for issue in review.issues if any(record["split"] == split and record["id"] == original[0]
                                                               for record in issue["records"])]
                    dispute_evidence.append({"split": split, "id": original[0], "text": candidate[1],
                                             "text_unchanged": candidate[1] == original[1],
                                             "text_sha256": _text_sha(candidate[1]),
                                             "labels": dict(zip(ASPECT_COLUMNS, map(int, candidate[2:]))),
                                             "original_issue_groups": groups})
                    seen_disputes.add(original[0])
            _check(next(result_reader, None) is None, f"输出存在多余记录：{split}")
    quarantine_path = batch.output_paths["quarantine"]
    # 隔离副本必须完整保留原始整行，不能因不进入训练就允许改字、改标或遗漏记录。
    with quarantine_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        _check(next(reader) == list(EXPECTED_COLUMNS) and list(reader) == quarantined_rows, "隔离副本未完整保留原文。")
    _check(seen_edits == review.edits_plan.keys() and seen_quarantine == review.quarantine.keys() and seen_evidence == review.expected_records.keys()
           and seen_disputes == dispute_ids, "部分计划目标未在原始数据中找到。")
    return _RecordAudit(
        raw_counts=raw_counts, retained_counts=retained_counts, unchanged_counts=unchanged_counts,
        removed_chars=removed_chars, quarantined_rows=quarantined_rows, edit_evidence=edit_evidence,
        dispute_evidence=dispute_evidence, dispute_ids=dispute_ids, label_cells=label_cells,
    )


def _audit_summary_counts(batch: _AuditBatch, review: _ReviewedPlan, records: _RecordAudit) -> None:
    """将逐条复算的数量和成员身份同时与处理报告、原始扫描及复核预演对账。"""
    _check(dict(records.retained_counts) == batch.processing["processed_counts"] and len(records.quarantined_rows) == batch.processing["quarantine_count"],
           "复算条数与处理报告不符。")
    _check(sum(records.removed_chars.values()) == batch.processing["removed_control_characters"], "复算删除字符总数不符。")
    before_counts = {split: batch.source_report["files"][split]["records_scanned"] for split in FILENAMES}
    # 将逐条复算得到的数量同时与处理报告、复核预演对账，防止各自内部自洽却相互矛盾。
    cleaned_counts = {split: sum(key[0] == split for key in review.edits_plan) for split in FILENAMES}
    _check(dict(records.raw_counts) == before_counts == batch.summary["dry_run"]["actual_before_counts"]
           and dict(records.retained_counts) == batch.summary["dry_run"]["expected_after_counts"]
           and batch.summary["quarantine_count"] == len(review.quarantine)
           and batch.processing["cleaned_record_counts"] == cleaned_counts
           and all(batch.summary["proposed_cleaned_record_counts"].get(split, 0) == count for split, count in cleaned_counts.items())
           and batch.summary["proposed_removed_control_characters"] == sum(records.removed_chars.values()),
           "复算处理数量与复核汇总不一致。")
    expected_quarantine_records = [{"split": split, "id": review.expected_records[(split, number)]["id"],
                                    "source_record_number": number, "issue_ids": issue_ids}
                                   for (split, number), issue_ids in sorted(review.quarantine_sources.items())]
    _check(batch.processing["quarantine_records"] == expected_quarantine_records, "处理报告的隔离来源与决定不一致。")
    if "quarantine_training_records" in batch.summary:
        _check(_references(batch.summary["quarantine_training_records"], "复核汇总的隔离记录重复。")
               == {key: value["id"] for key, value in review.quarantine.items()}, "复核汇总的隔离成员不符。")
    if "preserved_validation_records" in batch.summary:
        _check(_references(batch.summary["preserved_validation_records"], "复核汇总的验证保留记录重复。")
               == {key: value["id"] for key, value in review.expected_records.items() if key[0] == "validation"},
               "复核汇总的验证保留成员不符。")
    if "unique_source_records_verified" in batch.summary:
        _check(batch.summary["unique_source_records_verified"] == len(review.expected_records), "复核汇总的来源样本数量不符。")


def _build_audit_report(batch: _AuditBatch, review: _ReviewedPlan, records: _RecordAudit,
                        scan_options: dict) -> dict:
    """在来源和产物核验全部通过后组装报告，保持既有字段和证据结构。"""
    report = {
        "schema_version": "1.0", "audit_method_version": "1.1.0",
        "reviewed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "scope": "全量来源对账、逐字符编辑核对和隔离副本核对；语义结论另存。",
        "mechanical_checks_passed": True, "source_files_unchanged_during_audit": True,
        "raw_counts": dict(records.raw_counts), "retained_counts": dict(records.retained_counts),
        "unchanged_text_counts": dict(records.unchanged_counts), "edited_text_records": len(records.edit_evidence),
        "removed_characters": dict(records.removed_chars), "removed_character_count": sum(records.removed_chars.values()),
        "removed_python_whitespace_control_count": sum(item["python_isspace"] for record in records.edit_evidence for item in record["deletions"]),
        "label_cells_compared": records.label_cells, "unplanned_text_edits": 0, "changed_label_cells": 0,
        "quarantined_records_preserved_exactly": len(records.quarantined_rows),
        "application_issue_ids_reconciled": len(review.applications),
        "validation_disputed_ids": sorted(records.dispute_ids),
        "validation_options": scan_options,
        "batch": {
            "processing_report": receipt_path(batch.processing_path, batch.root),
            "inputs": {"raw_report": receipt_path(batch.report_path, batch.root),
                       "review_summary": receipt_path(batch.summary_path, batch.root),
                       "issues": receipt_path(batch.issues_path, batch.root), "decisions": receipt_path(batch.decisions_path, batch.root),
                       **{split: receipt_path(path, batch.root) for split, path in batch.raw_paths.items()}},
            "outputs": {name: receipt_path(path, batch.root) for name, path in batch.output_paths.items()},
        },
        "file_sha256": {receipt_path(path, batch.root): value for path, value in batch.protected.items()},
        "limitations": ["字符未误删不等于原始文本没有噪声或缺字。",
                        "控制符也可能充当 token 分隔，需结合逐条上下文复核。",
                        "仅对已记录验证争议提取证据，不进行全量20方面语义重标。",
                        "本次语义建议不修改固定验证标签，也不作为权威替代真值。"]}
    return report


def audit_processed_dataset(root: Path = PROJECT_ROOT, *,
                            processing_report: Path | None = None) -> tuple[dict, list[dict], list[dict]]:
    """重新读取全部来源和产物，独立验证执行结果符合原始复核计划。

    不调用处理函数，也不把历史校验的通过状态当作文本未误删的证明；结论必须来自
    本次重新读取后的精确比较。本函数只读，可供笔记本调用，返回对账报告、
    逐条文本编辑证据和验证标签争议证据，不进行语义重标。
    """
    root = Path(root).resolve()
    batch = _load_audit_batch(root, processing_report)
    source_hashes, scan_options = _validate_receipt_chain(batch)
    review = _audit_review_plan(batch, source_hashes)
    _audit_application_outcomes(batch, review)
    _audit_post_validation(batch, review)
    records = _audit_csv_records(batch, review)
    _audit_summary_counts(batch, review, records)
    # 只有全部来源和产物在审计期间保持不变，前面的跨文件比较才属于同一份稳定快照。
    _check(all(file_sha256(path) == value for path, value in batch.protected.items()), "复核期间输入发生变化。")
    report = _build_audit_report(batch, review, records, scan_options)
    return report, records.edit_evidence, records.dispute_evidence


def main(argv=None) -> int:
    """在所选项目上执行只读对账和独立扫描，将新证据写入独立且尚未使用的报告目录。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="独立只读复核处理后文本与来源，另存证据和二次校验。")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--processing-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output_dir = (args.output_dir or root / "data/reports/processed_review").resolve()
    targets = [output_dir / name for name in ("lineage_audit.json", "text_edit_evidence.json", "validation_dispute_evidence.json")]
    structural_dir = output_dir / "structural_validation"
    try:
        # 拒绝写入原始数据、处理结果或原凭据目录，也不覆盖任何已有复核产物。
        forbidden = [root / name for name in ("data/raw", "data/processed", "data/quarantine", "data/reports/raw", "data/reports/processed")]
        _check(not any(output_dir.is_relative_to(path) for path in forbidden), "复核输出不能写入现有数据或报告目录。")
        _check(not any(path.exists() or path.is_symlink() for path in targets)
               and not structural_dir.exists(), "复核产物已存在，请使用新的输出目录。")
        report, edits, disputes = audit_processed_dataset(root, processing_report=args.processing_report)
        batch_paths = {name: resolve_receipt_path(value, root=root) for name, value in report["batch"]["outputs"].items()}
        source_paths = [resolve_receipt_path(value, root=root) for value in report["batch"]["inputs"].values()]
        # 自定义批次的实际输入、输出与凭据目录同样受保护，不能借更换批次绕过写入边界。
        bound_parents = {path.parent for path in (*batch_paths.values(), *source_paths,
                         resolve_receipt_path(report["batch"]["processing_report"], root=root))}
        _check(not any(output_dir.is_relative_to(path) for path in bound_parents), "复核输出不能写入所选批次的数据或凭据目录。")
        output_dir.mkdir(parents=True, exist_ok=True)
        validation = validate_dataset({split: batch_paths[split] for split in FILENAMES},
                                      structural_dir, stage="raw", root=root, **report["validation_options"])
        previous_path = batch_paths["validation_report"]
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        # 先前批准已在上面的全量审计中核验，此处只继承证据完全相同的重复保留决定。
        _resolve_fresh_reviews(validation, read_jsonl(structural_dir / "validation_issues.jsonl"),
                               read_jsonl(batch_paths["validation_issues"]),
                               previous.get("review_resolutions", []))
        _check(all(file_sha256(resolve_receipt_path(path, root=root)) == value
                   for path, value in report["file_sha256"].items()), "二次校验期间输入发生变化。")
        report["fresh_validation"] = {"status": validation["status"], "scan_complete": validation["scan_complete"],
                                      "issue_counts": validation["issue_counts"]}
        for field in ("scan_status", "review_resolutions", "pending_reviews", "validation_gate_passed"):
            if field in validation:
                report["fresh_validation"][field] = validation[field]
        # 将来源总报告最后写入，避免失败时出现一份看似已经完成的对账凭据。
        write_json_atomic(structural_dir / "validation_report.json", validation)
        write_json_atomic(targets[1], edits)
        write_json_atomic(targets[2], disputes)
        write_json_atomic(targets[0], report)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in ("mechanical_checks_passed", "retained_counts", "edited_text_records",
                                                  "removed_character_count", "label_cells_compared", "fresh_validation")},
                     ensure_ascii=False, indent=2))
    return 0 if validation["scan_complete"] and validation["status"] in {"passed", "passed_with_warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
