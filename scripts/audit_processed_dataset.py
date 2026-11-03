"""独立复核清洗后的记录、文本编辑和隔离副本，不修改任何数据集。

先核对原始校验、复核决定与执行日志之间的凭据关系，再重新读取所有
原始及处理后记录逐项对账。凭据摘要相符仅表示文件对应，不能代替内容核验。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

try:
    from .dataset_io import read_jsonl, receipt_origin, receipt_path, resolve_receipt_path
    from .label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS
    from .validate_dataset import file_sha256, validate_dataset, _write_json_atomic
except ImportError:
    from dataset_io import read_jsonl, receipt_origin, receipt_path, resolve_receipt_path
    from label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS
    from validate_dataset import file_sha256, validate_dataset, _write_json_atomic


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


def audit_processed_dataset(root: Path = PROJECT_ROOT) -> tuple[dict, list[dict], list[dict]]:
    """重新读取全部来源和产物，独立验证执行结果符合原始复核计划。

    不调用处理函数，也不把历史校验的通过状态当作文本未误删的证明；结论必须来自
    本次重新读取后的精确比较。本函数只读，可供笔记本调用，返回对账报告、
    逐条文本编辑证据和验证标签争议证据，不进行语义重标。
    """
    root = Path(root).resolve()
    raw_reports = root / "data/reports/raw"
    processed_reports = root / "data/reports/processed"
    processing_path = processed_reports / "processing_report.json"
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    # 通过已知报告的完整项目内后缀绑定旧项目根目录，避免仅凭 data 目录名猜测路径。
    # 副本迁移后，即使旧项目仍存在，也应读取当前 root 下的对应文件。
    processing_origin = receipt_origin(processing["outputs"]["validation_report"]["path"],
                                       processed_reports / "validation_report.json", root)
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

    raw_paths = {split: root / "data/raw" / name for split, name in FILENAMES.items()}
    processed_paths = {split: root / "data/processed" / name for split, name in FILENAMES.items()}
    report_path = raw_reports / "validation_report.json"
    summary_path = raw_reports / "review_summary.json"
    issues_path = raw_reports / "validation_issues.jsonl"
    decisions_path = raw_reports / "review_decisions.jsonl"
    required_inputs = {report_path, summary_path, issues_path, decisions_path, *raw_paths.values()}
    # 摘要必须绑定正确角色的完整文件集合；另一份同样有摘要的文件不能顶替指定输入。
    _check(input_paths == required_inputs, "处理凭据未完整绑定原始数据与复核文件。")
    required_outputs = {**processed_paths,
                        "quarantine": root / "data/quarantine/sentiment_analysis_trainingset_quarantine.csv",
                        "changes": processed_reports / "processing_changes.jsonl",
                        "applications": processed_reports / "application_results.jsonl",
                        "validation_report": processed_reports / "validation_report.json",
                        "validation_issues": processed_reports / "validation_issues.jsonl"}
    _check(output_paths == required_outputs, "数据路径没有绑定处理凭据中的正确产物。")
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    post_validation = json.loads(output_paths["validation_report"].read_text(encoding="utf-8"))
    source_origin = receipt_origin(source_report["artifacts"]["report"], report_path, root)
    post_origin = receipt_origin(post_validation["artifacts"]["report"], output_paths["validation_report"], root)
    # 分别验证“汇总引用校验报告和决定”“校验报告引用问题明细”的摘要链。
    # 这样即便有人更新了最外层处理凭据中的摘要，内部互相矛盾的材料仍无法通过。
    _check(summary.get("review_complete") is True, "复核尚未完成。")
    _check(processing.get("stage") == "cleaned_original_label_space" and processing.get("mapping_applied") is False
           and all(processing.get(field) is True for field in (
               "raw_data_unchanged", "original_reports_and_decisions_unchanged", "labels_unchanged",
               "ids_preserved", "source_order_preserved", "all_retained_records_audited_against_source")),
           "处理报告的映射或来源对账状态不一致。")
    _check(summary["source_report_sha256"] == protected[report_path]
           and summary["source_issues_sha256"] == source_report["artifacts"]["issues_sha256"] == protected[issues_path]
           and summary["artifacts"]["decisions_sha256"] == protected[decisions_path],
           "复核汇总、原始问题与决定的摘要关系不一致。")
    _check(source_report["stage"] == "raw" and source_report["scan_complete"] is True
           and not source_report["issue_counts"]["ERROR"] and set(source_report["files"]) == set(FILENAMES),
           "需要完整、无 ERROR 的原始校验报告。")
    _check(resolve_receipt_path(source_report["artifacts"]["issues"], root=root, legacy_root=source_origin) == issues_path,
           "原始校验报告的明细路径不符。")
    if "decisions" in summary["artifacts"]:
        _check(resolve_receipt_path(summary["artifacts"]["decisions"], root=root, legacy_root=source_origin) == decisions_path,
               "复核汇总的决定路径不符。")
    source_hashes = {split: protected[path] for split, path in raw_paths.items()}
    _check(summary["source_sha256"] == source_hashes, "复核汇总的原始数据摘要不符。")
    for split, path in raw_paths.items():
        _check(source_report["files"][split]["sha256"] == protected[path]
               and resolve_receipt_path(source_report["files"][split]["path"], root=root, legacy_root=source_origin) == path,
               "原始校验报告的数据来源不符。")
    _check(post_validation["scan_complete"] is True
           and post_validation["stage"] == "raw"
           and post_validation["status"] in {"passed", "passed_with_warnings"}
           and post_validation["status"] == processing["post_validation_status"]
           and post_validation["issue_counts"] == processing["post_validation_issue_counts"]
           and not post_validation["issue_counts"]["ERROR"], "二次校验状态与处理凭据不一致。")
    _check(resolve_receipt_path(post_validation["artifacts"]["issues"], root=root, legacy_root=post_origin) == output_paths["validation_issues"]
           and post_validation["artifacts"]["issues_sha256"] == protected[output_paths["validation_issues"]],
           "二次校验明细与处理凭据不一致。")
    for split, path in processed_paths.items():
        _check(resolve_receipt_path(post_validation["files"][split]["path"], root=root, legacy_root=post_origin) == path
               and post_validation["files"][split]["sha256"] == protected[path]
               and post_validation["files"][split]["records_scanned"] == processing["processed_counts"][split],
               "二次校验的数据来源或条数不一致。")
    # JSONL 按文件实际换行读取；正文中的 Unicode 分隔字符仍属于同一条证据。
    issues = read_jsonl(issues_path)
    decisions = read_jsonl(decisions_path)
    changes = read_jsonl(output_paths["changes"])
    applications = read_jsonl(output_paths["applications"])
    _check(len({item["issue_id"] for item in issues}) == len(issues), "原始问题编号重复。")
    _check(len({item["issue_id"] for item in decisions}) == len(decisions), "复核决定编号重复。")
    _check({item["issue_id"] for item in issues} == {item["issue_id"] for item in decisions}
           == {item["issue_id"] for item in applications}, "问题、决定、应用日志未完整对应。")
    _check(len(applications) == len(decisions), "应用日志条数不符。")
    by_issue = {item["issue_id"]: item for item in issues}
    applications_by_issue = {item["issue_id"]: item for item in applications}
    issue_counts = {severity: sum(item["severity"] == severity for item in issues)
                    for severity in ("ERROR", "REVIEW", "WARN")}
    _check(issue_counts == source_report["issue_counts"], "原始问题数量与校验报告不一致。")
    if "reviewed_issue_count" in summary:
        _check(summary["reviewed_issue_count"] == len(decisions), "复核汇总的问题数量不符。")
    if "reviewed_issue_counts" in summary:
        _check(summary["reviewed_issue_counts"] == {key: value for key, value in issue_counts.items() if value},
               "复核汇总的问题级别数量不符。")
    if "action_issue_counts" in summary:
        _check(summary["action_issue_counts"] == dict(Counter(item["action"]["type"] for item in decisions)),
               "复核汇总的动作数量不符。")
    quarantine = {}
    quarantine_sources = {}
    edits_plan = {}
    expected_records = {}
    retained_decisions = {}
    for decision in decisions:
        issue = by_issue[decision["issue_id"]]
        references = _references(issue["records"], "原始问题包含无效或重复样本。")
        affected = _references(decision["affected_records"], "复核决定包含无效或重复样本。")
        _check(references and affected == references, "决定引用的样本与原始问题不一致。")
        _check(issue["stage"] == "raw" and issue["rules_version"] == source_report["rules_version"]
               and issue["source_sha256"] == {split: source_hashes[split] for split, _ in references},
               "原始问题的规则版本或来源摘要不符。")
        _check(decision.get("status") == "reviewed_pending_application"
               and decision.get("action_applied") is False and decision.get("label_changes") == [],
               "复核状态不允许应用，或包含未授权改标。")
        _check(decision["source_sha256"] == issue["source_sha256"]
               and decision["rule_id"] == issue["rule_id"]
               and decision["original_severity"] == issue["severity"], "决定的来源或规则与原始问题不一致。")
        for record in issue["records"]:
            key = record["split"], record["record_number"]
            _check(key not in expected_records or expected_records[key] == record, "同一原始记录的证据不一致。")
            expected_records[key] = record
        action = decision["action"]
        application = applications_by_issue[decision["issue_id"]]
        # 问题编号对应还不够：应用日志还必须证明同一来源、同一目标、同一动作确已应用。
        # “排除训练风险”与“修复底层标注或原始缺字”含义不同，后者不能被日志擅自宣称完成。
        resolution = ("control_characters_removed_in_derived_copy" if action["type"] == "remove_reviewed_control_characters"
                      else "reviewed_records_retained_unchanged" if action["type"] == "retain_reviewed_records"
                      else "excluded_from_training_and_preserved_in_quarantine")
        _check(application.get("action_applied") is True and application["action_type"] == action["type"]
               and application["source_sha256"] == decision["source_sha256"]
               and _references(application["affected_records"], "应用日志包含无效或重复样本。") == affected
               and application.get("label_changes") == []
               and application["post_validation_status"] == processing["post_validation_status"]
               and application["resolution"] == resolution
               and application.get("underlying_annotation_or_source_repair_completed") is False
               and application["applied_at"] == processing["applied_at"],
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
                _check(issue["rule_id"].startswith("cross_split_") and issue["severity"] == "REVIEW"
                       and any(key[0] == "validation" for key in references), "重叠隔离动作与原始问题不符。")
            else:
                _check(issue["rule_id"].startswith("within_split_") and issue["severity"] == "REVIEW"
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
            retained_decisions[decision["issue_id"]] = issue
        else:
            raise ValueError(f"未知处理动作：{action['type']}")
    _check(not quarantine.keys() & edits_plan.keys(), "隔离与编辑目标相交。")

    def signature(issue, *, apply_plan=False):
        """计算组证据；比对输出问题时先在原始组上模拟已批准的隔离和字符删除。"""
        result = []
        for record in issue["records"]:
            key = record["split"], record["record_number"]
            if apply_plan and key in quarantine:
                continue
            text = record["text"]
            if apply_plan and key in edits_plan:
                text = "".join(char for char in text if char not in edits_plan[key]["characters"])
            result.append((record["split"], record["id"], text, tuple(sorted(record["labels"].items()))))
        return sorted(result)

    post_issues = read_jsonl(output_paths["validation_issues"])
    _check(len({item["issue_id"] for item in post_issues}) == len(post_issues), "二次校验问题编号重复。")
    post_counts = {severity: sum(item["severity"] == severity for item in post_issues)
                   for severity in ("ERROR", "REVIEW", "WARN")}
    _check(post_counts == post_validation["issue_counts"], "二次校验问题数量不符。")
    post_reviews = {item["issue_id"]: item for item in post_issues if item["severity"] == "REVIEW"}
    resolutions = post_validation.get("review_resolutions", [])
    # 二次扫描留下的每个待复核组，都必须逐项追溯到原始“保留”决定；不能整体放宽校验门槛。
    _check(len(resolutions) == len(post_reviews)
           and {item["issue_id"] for item in resolutions} == set(post_reviews), "二次校验的复核决定未完整对应。")
    for resolution in resolutions:
        issue = post_reviews[resolution["issue_id"]]
        source_ids = resolution["source_issue_ids"]
        expected_ids = {issue_id for issue_id, original in retained_decisions.items()
                        if original["rule_id"] == issue["rule_id"]
                        and signature(original, apply_plan=True) == signature(issue)}
        _check(source_ids and len(source_ids) == len(set(source_ids)) and set(source_ids) == expected_ids,
               "二次校验放行的问题不符合已复核保留证据。")
    if post_reviews:
        _check(post_validation.get("scan_status") == "needs_review"
               and post_validation["status"] == "passed_with_warnings"
               and post_validation.get("pending_reviews") == 0
               and post_validation.get("validation_gate_passed") is True,
               "二次校验的已复核保留状态不符。")

    log_by_key = {(item["split"], item["source_record_number"]): item for item in changes}
    _check(len(log_by_key) == len(changes) and log_by_key.keys() == edits_plan.keys(), "修改日志与计划不对应。")
    raw_counts, retained_counts, unchanged_counts = Counter(), Counter(), Counter()
    removed_chars = Counter()
    quarantined_rows = []
    edit_evidence, dispute_evidence = [], []
    dispute_ids = {item["id"] for item in processing["known_validation_label_disputes"]}
    # 从原始冲突证据重新推导争议集合，避免处理报告漏记争议或给同标签重叠虚构争议。
    expected_disputes = {(record["split"], record["record_number"]): record["id"]
                         for issue in issues if issue.get("label_conflicts")
                         for record in issue["records"] if record["split"] == "validation"}
    _check(_references(processing["known_validation_label_disputes"], "验证标签争议记录无效或重复。") == expected_disputes,
           "处理报告的验证标签争议与原始证据不一致。")
    seen_edits, seen_quarantine, seen_disputes, seen_evidence = set(), set(), set(), set()
    label_cells = 0
    for split in FILENAMES:
        # 使用 CSV 解析后的逻辑记录比较，评论中的换行不等于新增样本。
        # 原始序号只定位来源；跳过隔离记录后顺序消费输出，才能发现错行、漏行或多行。
        with raw_paths[split].open(encoding="utf-8-sig", newline="") as source_stream, \
                processed_paths[split].open(encoding="utf-8-sig", newline="") as result_stream:
            source_reader = csv.reader(source_stream, strict=True)
            result_reader = csv.reader(result_stream, strict=True)
            _check(next(source_reader) == next(result_reader) == list(EXPECTED_COLUMNS), "表头不符。")
            for number, original in enumerate(source_reader, start=1):
                raw_counts[split] += 1
                _check(len(original) == len(EXPECTED_COLUMNS), f"源记录列宽异常：{split}/{number}")
                key = split, number
                if key in expected_records:
                    evidence = expected_records[key]
                    _check(evidence["id"] == original[0] and evidence["text"] == original[1]
                           and evidence["labels"] == dict(zip(ASPECT_COLUMNS, map(int, original[2:]))),
                           "原始记录与问题证据不一致。")
                    seen_evidence.add(key)
                if key in quarantine:
                    _check(original[0] == quarantine[key]["id"], "隔离计划 ID 不符。")
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
                plan = edits_plan.get(key)
                expected = "".join(char for char in original[1] if char not in plan["characters"]) if plan else original[1]
                _check(candidate[1] == expected, f"发现计划外文本修改：{split}/{number}")
                if plan:
                    _check(original[0] == plan["id"], "编辑计划 ID 不符。")
                    evidence_record = next(record for record in by_issue[plan["issue_id"]]["records"]
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
                    log = log_by_key[key]
                    # 从实际差异重算日志中的字符数、长度和摘要，防止日志只记对问题编号。
                    _check(log["id"] == original[0] and log["issue_id"] == plan["issue_id"]
                           and log["source_sha256"] == protected[raw_paths[split]]
                           and log["original_text_sha256"] == _text_sha(original[1])
                           and log["processed_text_sha256"] == _text_sha(candidate[1])
                           and log["original_length"] == len(original[1])
                           and log["processed_length"] == len(candidate[1])
                           and log["removed_characters"] == dict(counts) and log["labels_changed"] is False,
                           "实际编辑与修改日志不符。")
                    removed_chars.update(counts)
                    edit_evidence.append({"split": split, "id": original[0], "source_record_number": number,
                                          "original_text": original[1], "processed_text": candidate[1],
                                          "original_text_sha256": _text_sha(original[1]),
                                          "processed_text_sha256": _text_sha(candidate[1]),
                                          "removed_characters": dict(counts), "deletions": positions})
                    seen_edits.add(key)
                else:
                    unchanged_counts[split] += 1
                if split == "validation" and original[0] in dispute_ids:
                    groups = [issue for issue in issues if any(record["split"] == split and record["id"] == original[0]
                                                               for record in issue["records"])]
                    dispute_evidence.append({"split": split, "id": original[0], "text": candidate[1],
                                             "text_unchanged": candidate[1] == original[1],
                                             "text_sha256": _text_sha(candidate[1]),
                                             "labels": dict(zip(ASPECT_COLUMNS, map(int, candidate[2:]))),
                                             "original_issue_groups": groups})
                    seen_disputes.add(original[0])
            _check(next(result_reader, None) is None, f"输出存在多余记录：{split}")
    quarantine_path = output_paths["quarantine"]
    # 隔离副本必须完整保留原始整行，不能因不进入训练就允许改字、改标或遗漏记录。
    with quarantine_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        _check(next(reader) == list(EXPECTED_COLUMNS) and list(reader) == quarantined_rows, "隔离副本未完整保留原文。")
    _check(seen_edits == edits_plan.keys() and seen_quarantine == quarantine.keys() and seen_evidence == expected_records.keys()
           and seen_disputes == dispute_ids, "部分计划目标未在原始数据中找到。")
    _check(dict(retained_counts) == processing["processed_counts"] and len(quarantined_rows) == processing["quarantine_count"],
           "复算条数与处理报告不符。")
    _check(sum(removed_chars.values()) == processing["removed_control_characters"], "复算删除字符总数不符。")
    before_counts = {split: source_report["files"][split]["records_scanned"] for split in FILENAMES}
    # 将逐条复算得到的数量同时与处理报告、复核预演对账，防止各自内部自洽却相互矛盾。
    cleaned_counts = {split: sum(key[0] == split for key in edits_plan) for split in FILENAMES}
    _check(dict(raw_counts) == before_counts == summary["dry_run"]["actual_before_counts"]
           and dict(retained_counts) == summary["dry_run"]["expected_after_counts"]
           and summary["quarantine_count"] == len(quarantine)
           and processing["cleaned_record_counts"] == cleaned_counts
           and all(summary["proposed_cleaned_record_counts"].get(split, 0) == count for split, count in cleaned_counts.items())
           and summary["proposed_removed_control_characters"] == sum(removed_chars.values()),
           "复算处理数量与复核汇总不一致。")
    expected_quarantine_records = [{"split": split, "id": expected_records[(split, number)]["id"],
                                    "source_record_number": number, "issue_ids": issue_ids}
                                   for (split, number), issue_ids in sorted(quarantine_sources.items())]
    _check(processing["quarantine_records"] == expected_quarantine_records, "处理报告的隔离来源与决定不一致。")
    if "quarantine_training_records" in summary:
        _check(_references(summary["quarantine_training_records"], "复核汇总的隔离记录重复。")
               == {key: value["id"] for key, value in quarantine.items()}, "复核汇总的隔离成员不符。")
    if "preserved_validation_records" in summary:
        _check(_references(summary["preserved_validation_records"], "复核汇总的验证保留记录重复。")
               == {key: value["id"] for key, value in expected_records.items() if key[0] == "validation"},
               "复核汇总的验证保留成员不符。")
    if "unique_source_records_verified" in summary:
        _check(summary["unique_source_records_verified"] == len(expected_records), "复核汇总的来源样本数量不符。")
    # 只有全部来源和产物在审计期间保持不变，前面的跨文件比较才属于同一份稳定快照。
    _check(all(file_sha256(path) == value for path, value in protected.items()), "复核期间输入发生变化。")
    report = {
        "schema_version": "1.0", "audit_method_version": "1.1.0",
        "reviewed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "scope": "全量来源对账、逐字符编辑核对和隔离副本核对；语义结论另存。",
        "mechanical_checks_passed": True, "source_files_unchanged_during_audit": True,
        "raw_counts": dict(raw_counts), "retained_counts": dict(retained_counts),
        "unchanged_text_counts": dict(unchanged_counts), "edited_text_records": len(edit_evidence),
        "removed_characters": dict(removed_chars), "removed_character_count": sum(removed_chars.values()),
        "removed_python_whitespace_control_count": sum(item["python_isspace"] for record in edit_evidence for item in record["deletions"]),
        "label_cells_compared": label_cells, "unplanned_text_edits": 0, "changed_label_cells": 0,
        "quarantined_records_preserved_exactly": len(quarantined_rows),
        "application_issue_ids_reconciled": len(applications),
        "validation_disputed_ids": sorted(dispute_ids),
        "file_sha256": {receipt_path(path, root): value for path, value in protected.items()},
        "limitations": ["字符未误删不等于原始文本没有噪声或缺字。",
                        "控制符也可能充当 token 分隔，需结合逐条上下文复核。",
                        "仅对已记录验证争议提取证据，不进行全量20方面语义重标。",
                        "本次语义建议不修改固定验证标签，也不作为权威替代真值。"]}
    return report, edit_evidence, dispute_evidence


def main(argv=None) -> int:
    """在所选项目上执行只读对账和独立扫描，将新证据写入独立且尚未使用的报告目录。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="独立只读复核处理后文本与来源，另存证据和二次校验。")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
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
        report, edits, disputes = audit_processed_dataset(root)
        output_dir.mkdir(parents=True, exist_ok=True)
        validation = validate_dataset({split: root / "data/processed" / name for split, name in FILENAMES.items()},
                                      structural_dir, stage="raw", id_scope="split", root=root)
        previous_path = root / "data/reports/processed/validation_report.json"
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        # 先前批准已在上面的全量审计中核验，此处只继承证据完全相同的重复保留决定。
        _resolve_fresh_reviews(validation, read_jsonl(structural_dir / "validation_issues.jsonl"),
                               read_jsonl(root / "data/reports/processed/validation_issues.jsonl"),
                               previous.get("review_resolutions", []))
        _check(all(file_sha256(resolve_receipt_path(path, root=root)) == value
                   for path, value in report["file_sha256"].items()), "二次校验期间输入发生变化。")
        report["fresh_validation"] = {"status": validation["status"], "scan_complete": validation["scan_complete"],
                                      "issue_counts": validation["issue_counts"]}
        for field in ("scan_status", "review_resolutions", "pending_reviews", "validation_gate_passed"):
            if field in validation:
                report["fresh_validation"][field] = validation[field]
        # 将来源总报告最后写入，避免失败时出现一份看似已经完成的对账凭据。
        _write_json_atomic(structural_dir / "validation_report.json", validation)
        _write_json_atomic(targets[1], edits)
        _write_json_atomic(targets[2], disputes)
        _write_json_atomic(targets[0], report)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.error(str(error))
    print(json.dumps({key: report[key] for key in ("mechanical_checks_passed", "retained_counts", "edited_text_records",
                                                  "removed_character_count", "label_cells_compared", "fresh_validation")},
                     ensure_ascii=False, indent=2))
    return 0 if validation["scan_complete"] and validation["status"] in {"passed", "passed_with_warnings"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
