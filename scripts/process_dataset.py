"""将已经完成复核的隔离方案和精确字符删除应用到派生 CSV。

先核对原始报告、问题明细、复核决定和汇总的摘要及业务状态，再在临时文件中
执行方案。只有来源逐条对账、二次校验及输入摘要复查全部通过，才发布结果。
原始 CSV、原始问题和复核决定保持不变；输出保留 -2、-1、0、1 四类原标签，
不在本步骤执行标签映射或语义重标。
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import unicodedata

try:
    from .dataset_io import read_jsonl, receipt_origin, receipt_path, resolve_receipt_path
    from .label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS, parse_label
    from .validate_dataset import file_sha256, validate_dataset, _write_json_atomic
except ImportError:
    from dataset_io import read_jsonl, receipt_origin, receipt_path, resolve_receipt_path
    from label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS, parse_label
    from validate_dataset import file_sha256, validate_dataset, _write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUARANTINE_ACTIONS = {
    "quarantine_training_conflict_cluster", "quarantine_training_overlap",
    "quarantine_corrupted_training_record",
}


_jsonl = read_jsonl


def _key(record: dict) -> tuple[str, int]:
    """使用集合名和原始数据记录序号定位样本，避免跨集合相同 ID 混淆。

    record_number 从表头后的第一条 CSV 记录开始计数，不是物理文件行号；
    含换行的评论仍然只算一条记录。隔离后序号会移动，因此一直引用原始序号。
    """
    return record["split"], record["record_number"]


def _text_sha(text: str) -> str:
    """计算解析后评论文本的 UTF-8 摘要，用于核对精确编辑前后的内容。

    此摘要不包含 CSV 引号和换行编码等文件格式差异，与整份文件摘要用途不同。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_dataset(report_path: Path, decisions_path: Path, review_summary_path: Path,
                    output_dir: Path, quarantine_dir: Path, reports_dir: Path, *,
                    root: Path | None = None) -> dict:
    """应用与输入摘要绑定的复核方案，并返回最终发布的处理报告。

    输入包括原始校验报告、逐问题决定和复核汇总；三个输出目录分别用于
    保留样本、隔离副本和处理凭据，不能覆盖输入或相互嵌套。

    决定必须完整覆盖问题，且每项都已复核、尚未应用、没有改标计划。
    所有实际写入先落到临时文件；逐条核对 ID、顺序、全部标签及文本变化，
    再执行二次校验。只有明确批准保留的重复组可以继承原复核结论。
    发布失败时仅清理本次创建的文件，既有输入和产物不会被覆盖。
    """
    # 先规范本次实际传入的位置，再解析凭据中的旧路径。
    # 显式 root 优先；省略时从标准 data 目录布局推导，兼容原来的调用方式。
    report_path, decisions_path, review_summary_path = (
        Path(path).resolve() for path in (report_path, decisions_path, review_summary_path)
    )
    root = (Path(root).resolve() if root is not None else
            next((parent.parent for parent in report_path.parents if parent.name == "data"), PROJECT_ROOT))
    output_dir, quarantine_dir, reports_dir = (
        Path(path).resolve() for path in (output_dir, quarantine_dir, reports_dir)
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(review_summary_path.read_text(encoding="utf-8"))
    legacy_root = receipt_origin(report["artifacts"]["report"], report_path, root)
    issues_path = resolve_receipt_path(report["artifacts"]["issues"], root=root, legacy_root=legacy_root)
    source_paths = {split: resolve_receipt_path(info["path"], root=root, legacy_root=legacy_root)
                    for split, info in report["files"].items()}
    # 摘要一致只能证明文件相互对应，不能证明业务复核已经完成。
    # 必须使用真正的布尔 True，不能把字符串或整数等真值当作完成状态。
    if summary.get("review_complete") is not True:
        raise ValueError("复核尚未完成，不能应用处理决定。")
    if (report["stage"] != "raw" or report["scan_complete"] is not True
            or report["issue_counts"]["ERROR"] or set(source_paths) != {"train", "validation"}):
        raise ValueError("需要完整、无 ERROR 的 train/validation 原始校验报告。")
    # 输出目录彼此分开，并避开所有输入所在目录；同时拒绝覆盖已有目标。
    # 这些检查在创建目录和临时文件之前完成，失败时不留下处理结果。
    directories = (output_dir, quarantine_dir, reports_dir)
    for index, directory in enumerate(directories):
        if any(directory.is_relative_to(other) or other.is_relative_to(directory)
               for other in directories[index + 1:]):
            raise ValueError("processed、quarantine 和报告目录不能相同或互相包含。")
        if any(directory.is_relative_to(path.parent)
               for path in (*source_paths.values(), report_path, decisions_path, review_summary_path, issues_path)):
            raise ValueError("输出目录不能位于输入数据或原始复核报告目录中。")

    targets = {
        "train": output_dir / source_paths["train"].name,
        "validation": output_dir / source_paths["validation"].name,
        "quarantine": quarantine_dir / "sentiment_analysis_trainingset_quarantine.csv",
        "changes": reports_dir / "processing_changes.jsonl",
        "applications": reports_dir / "application_results.jsonl",
        "validation_report": reports_dir / "validation_report.json",
        "validation_issues": reports_dir / "validation_issues.jsonl",
        "processing_report": reports_dir / "processing_report.json",
    }
    if len(set(targets.values())) != len(targets):
        raise ValueError("输出文件路径冲突。")
    if any(path.exists() or path.is_symlink() for path in targets.values()):
        raise ValueError("已有处理结果，拒绝覆盖；请使用新的输出目录。")

    # 保存所有受保护输入的当前摘要，并沿报告→问题→决定→汇总逐层核对。
    # 发布之前还会复算同一组摘要，防止处理期间输入被其他程序改写。
    protected = {path: file_sha256(path) for path in
                 (report_path, decisions_path, review_summary_path, issues_path, *source_paths.values())}
    if (summary["source_report_sha256"] != protected[report_path]
            or summary["source_issues_sha256"] != protected[issues_path]
            or report["artifacts"]["issues_sha256"] != protected[issues_path]
            or summary["artifacts"]["decisions_sha256"] != protected[decisions_path]):
        raise ValueError("复核报告、决定或异常明细的摘要不匹配。")
    for split, path in source_paths.items():
        if report["files"][split]["sha256"] != protected[path] or summary["source_sha256"][split] != protected[path]:
            raise ValueError("原始数据摘要已变化，必须重新校验复核。")
    if "decisions" in summary["artifacts"] and resolve_receipt_path(
            summary["artifacts"]["decisions"], root=root, legacy_root=legacy_root) != decisions_path:
        raise ValueError("复核汇总的决定路径与实际输入不一致。")

    # 要求问题与决定一一对应，不能漏审、重复应用，也不能混入其他批次决定。
    decisions = _jsonl(decisions_path)
    issues = _jsonl(issues_path)
    issues_by_id = {issue["issue_id"]: issue for issue in issues}
    if (len(issues_by_id) != len(issues)
            or len({decision["issue_id"] for decision in decisions}) != len(decisions)
            or {decision["issue_id"] for decision in decisions} != set(issues_by_id)):
        raise ValueError("处理决定必须完整覆盖所有原始问题，且不能重复。")
    if report["issue_counts"] != {
            severity: sum(issue["severity"] == severity for issue in issues)
            for severity in ("ERROR", "REVIEW", "WARN")}:
        raise ValueError("原始问题数量与校验报告不一致。")
    if ("reviewed_issue_count" in summary and summary["reviewed_issue_count"] != len(decisions)
            or "action_issue_counts" in summary and summary["action_issue_counts"] !=
            dict(Counter(decision["action"]["type"] for decision in decisions))):
        raise ValueError("复核汇总的问题或动作数量不一致。")
    # 先构建完整内存计划，再读写 CSV。隔离按原始记录去重，但保留所有来源问题；
    # 清洗计划同时保存精确候选文本和字符计数，供执行和后续对账共同核验。
    quarantine = {}
    clean = {}
    retained_issues = {}
    expected_records = {}
    for decision in decisions:
        issue = issues_by_id[decision["issue_id"]]
        if (decision["source_sha256"] != issue["source_sha256"]
                or issue["stage"] != "raw" or issue["rules_version"] != report["rules_version"]
                or issue["source_sha256"] != {split: report["files"][split]["sha256"]
                                               for split in {record["split"] for record in issue["records"]}}
                or decision["rule_id"] != issue["rule_id"]
                or decision["original_severity"] != issue["severity"]
                or decision.get("label_changes") != [] or decision.get("action_applied") is not False
                or decision.get("status") != "reviewed_pending_application"):
            raise ValueError("决定的来源、规则或状态不匹配，或包含未授权改标。")
        # 用字典比较成员之前先检查数量，防止重复引用被字典合并而悄悄漏过。
        references = {_key(record): record for record in issue["records"]}
        if (len(references) != len(issue["records"])
                or len({_key(record) for record in decision["affected_records"]}) != len(decision["affected_records"])
                or {_key(record): record["id"] for record in decision["affected_records"]} != {
            key: record["id"] for key, record in references.items()
        }):
            raise ValueError("决定引用的样本与异常证据不一致。")
        for key, record in references.items():
            if key in expected_records and expected_records[key] != record:
                raise ValueError("同一原始记录的证据不一致。")
            expected_records[key] = record
        action = decision["action"]
        # 标签冲突从成员的 20 个实际标签重新计算，不能只相信冲突摘要字段。
        # 否则清空 label_conflicts 就可能让本应隔离的训练冲突组获得保留资格。
        if issue["rule_id"] in {"within_split_exact_duplicate", "within_split_normalized_duplicate",
                                "cross_split_exact_overlap", "cross_split_normalized_overlap"}:
            actual_conflicts = {
                aspect: [{**{field: record[field] for field in ("split", "id", "record_number")},
                          "value": record["labels"][aspect]} for record in issue["records"]]
                for aspect in ASPECT_COLUMNS
                if len({record["labels"][aspect] for record in issue["records"]}) > 1
            }
            if issue.get("label_conflicts") != actual_conflicts:
                raise ValueError("重复组的标签冲突明细与记录不一致。")
        # 三种隔离动作具有不同依据，但都只能移出该问题的训练成员。
        # 固定验证集的成员、文本和标签必须继续保留。
        if action["type"] in QUARANTINE_ACTIONS:
            affected = {_key(record): record["id"] for record in action["quarantine"]}
            expected = {key: record["id"] for key, record in references.items() if key[0] == "train"}
            if (not affected or len(affected) != len(action["quarantine"])
                    or affected != expected or any(key[0] != "train" for key in affected)):
                raise ValueError("隔离必须且只能覆盖该问题的训练成员，不能移除验证样本。")
            if action["type"] == "quarantine_corrupted_training_record":
                if issue["rule_id"] != "suspicious_characters" or len(references) != 1:
                    raise ValueError("损坏隔离动作与问题类型不符。")
            elif action["type"] == "quarantine_training_overlap":
                if (issue["rule_id"] not in {"cross_split_exact_overlap", "cross_split_normalized_overlap"}
                        or issue["severity"] != "REVIEW" or {key[0] for key in references} != {"train", "validation"}):
                    raise ValueError("跨集合隔离动作必须对应跨集合重复。")
            elif (issue["rule_id"] not in {"within_split_exact_duplicate", "within_split_normalized_duplicate"}
                  or not issue.get("label_conflicts") or {key[0] for key in references} != {"train"}):
                raise ValueError("冲突隔离动作必须对应训练集标签冲突。")
            for key in affected:
                quarantine.setdefault(key, []).append(decision["issue_id"])
        elif action["type"] == "remove_reviewed_control_characters":
            # 字符删除必须唯一绑定一条告警，并且只删除其中已经确认的控制符。
            # 正常制表符、换行、普通文字和表情不能因通用“清洗”规则被一并去掉。
            key = _key(action["target"])
            if (issue["rule_id"] != "suspicious_characters" or len(references) != 1
                    or key not in references or action["target"]["id"] != references[key]["id"] or key in clean):
                raise ValueError("字符清洗必须唯一对应已复核的原始记录。")
            code_points = action["remove_code_points"]
            if not code_points or set(code_points) != set(issue["characters"]) or len(set(code_points)) != len(code_points):
                raise ValueError("清洗字符与原始告警不一致。")
            if any(not re.fullmatch(r"U\+[0-9A-F]{4,6}", value)
                   or int(value[2:], 16) > 0x10FFFF for value in code_points):
                raise ValueError("控制字符编码不合法。")
            characters = {chr(int(value[2:], 16)) for value in code_points}
            if (any(unicodedata.category(char) != "Cc" or char in "\t\r\n" for char in characters)
                    or action.get("replacement", "") != ""):
                raise ValueError("仅允许删除已复核异常控制符，不能删除普通字母、表情、换行或替换缺字。")
            original = references[key]["text"]
            candidate = "".join(char for char in original if char not in characters)
            actual = {value: original.count(chr(int(value[2:], 16))) for value in code_points}
            if (actual != issue["characters"] or len(original) != action["original_length"]
                    or len(candidate) != action["candidate_length"]
                    or _text_sha(candidate) != action["candidate_text_sha256"] or not candidate.strip()):
                raise ValueError("原文、字符数量或清洗后长度/摘要不符合复核决定。")
            clean[key] = {"issue_id": decision["issue_id"], "characters": characters,
                          "text": candidate, "counts": actual}
        elif action["type"] == "retain_reviewed_records":
            # 同集合同标签重复可以原样保留；训练标签冲突仍须隔离。
            # 验证集内部冲突保留原标并继续记录争议，不把“保留”解释为标签已修正。
            if (issue["rule_id"] not in {"within_split_exact_duplicate", "within_split_normalized_duplicate"}
                    or len({key[0] for key in references}) != 1
                    or (issue.get("label_conflicts") and any(key[0] == "train" for key in references))):
                raise ValueError("仅允许保留已复核的集合内重复；训练标签冲突必须隔离。")
            if action.get("quarantine") or action.get("target") or decision.get("label_changes"):
                raise ValueError("保留动作不能隔离记录或修改标签。")
            retained_issues[decision["issue_id"]] = issue
        else:
            raise ValueError("存在未授权或未支持的处理动作。")
        if action["type"] in QUARANTINE_ACTIONS | {"retain_reviewed_records"}:
            preserved = action.get("preserve_fixed_validation", [])
            if (len({_key(record) for record in preserved}) != len(preserved)
                    or {_key(record): record["id"] for record in preserved} != {
                        key: record["id"] for key, record in references.items() if key[0] == "validation"}):
                raise ValueError("固定验证集的保留目标与原始问题不一致。")
    if quarantine.keys() & clean.keys():
        raise ValueError("同一记录不能同时隔离和清洗。")
    # 将决定重新汇总为样本数和删除字符数，与只读预演核对。
    # 问题组可能共享同一条训练记录，因此隔离数量按唯一记录而非问题条数计算。
    expected_before = {split: info["records_scanned"] for split, info in report["files"].items()}
    expected_after = {split: count - sum(key[0] == split for key in quarantine)
                      for split, count in expected_before.items()}
    cleaned_counts = {split: sum(key[0] == split for key in clean) for split in source_paths}
    removed_characters = sum(sum(item["counts"].values()) for item in clean.values())
    if (summary["dry_run"]["actual_before_counts"] != expected_before
            or summary["dry_run"]["expected_after_counts"] != expected_after
            or summary["quarantine_count"] != len(quarantine)
            or any(summary["proposed_cleaned_record_counts"].get(split, 0) != count for split, count in cleaned_counts.items())
            or summary["proposed_removed_control_characters"] != removed_characters):
        raise ValueError("实际处理计划与已复核的数量预演不符。")

    # 临时文件与各自目标位于同一目录，便于最终发布；处理报告最后发布作为完成凭据。
    staged = {}
    published = []
    validation_staging = None
    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        for name, target in targets.items():
            with tempfile.NamedTemporaryFile(prefix=".processing-", suffix=target.suffix + ".tmp",
                                             dir=target.parent, delete=False) as stream:
                staged[name] = Path(stream.name)
        seen = set()
        quarantine_rows = []
        changes = []
        before_counts = Counter()
        processed_counts = Counter()
        # CSV 必须交给解析器处理多行评论，不能依赖物理行号或直接拆分逗号。
        # 每条命中问题的来源记录先与证据核对，再按计划隔离或替换文本。
        with staged["quarantine"].open("w", encoding="utf-8", newline="") as quarantine_stream:
            quarantine_writer = csv.writer(quarantine_stream)
            quarantine_writer.writerow(EXPECTED_COLUMNS)
            for split, source in source_paths.items():
                with source.open(encoding="utf-8-sig", newline="") as source_stream, staged[split].open("w", encoding="utf-8", newline="") as destination:
                    reader = csv.reader(source_stream, strict=True)
                    if next(reader, None) != list(EXPECTED_COLUMNS):
                        raise ValueError("原始数据表头发生变化。")
                    writer = csv.writer(destination)
                    writer.writerow(EXPECTED_COLUMNS)
                    for number, row in enumerate(reader, start=1):
                        before_counts[split] += 1
                        if len(row) != len(EXPECTED_COLUMNS):
                            raise ValueError("原始记录列宽异常。")
                        key = split, number
                        if key in expected_records:
                            expected = expected_records[key]
                            if (row[0] != expected["id"] or row[1] != expected["text"]
                                    or {aspect: parse_label(value) for aspect, value in zip(ASPECT_COLUMNS, row[2:])} != expected["labels"]):
                                raise ValueError("原始记录与问题证据不匹配。")
                            seen.add(key)
                        if key in quarantine:
                            quarantine_writer.writerow(row)
                            quarantine_rows.append(row)
                            continue
                        if key in clean:
                            item = clean[key]
                            changes.append({"split": split, "id": row[0], "source_record_number": number,
                                            "source_sha256": protected[source], "issue_id": item["issue_id"],
                                            "original_text_sha256": _text_sha(row[1]), "processed_text_sha256": _text_sha(item["text"]),
                                            "original_length": len(row[1]), "processed_length": len(item["text"]),
                                            "removed_characters": item["counts"], "labels_changed": False})
                            row[1] = item["text"]
                        writer.writerow(row)
                        processed_counts[split] += 1
        if (seen != expected_records.keys() or dict(before_counts) != expected_before
                or dict(processed_counts) != expected_after or len(quarantine_rows) != len(quarantine)):
            raise ValueError("处理后的样本数量或问题覆盖对不上。")

        # 独立重读原始与候选 CSV，避免只凭写入过程或文件摘要认定处理正确。
        # 每条保留记录的 ID、顺序和标签原始字符串都必须一致；文本只能等于计划结果，
        # 隔离副本则必须逐字段保存完整原始行。
        for split, source in source_paths.items():
            with source.open(encoding="utf-8-sig", newline="") as original_stream, staged[split].open(encoding="utf-8", newline="") as candidate_stream:
                original_reader = csv.reader(original_stream, strict=True)
                candidate_reader = csv.reader(candidate_stream, strict=True)
                if next(original_reader) != next(candidate_reader):
                    raise ValueError("处理后的表头变化。")
                for number, original in enumerate(original_reader, start=1):
                    key = split, number
                    if key in quarantine:
                        continue
                    candidate = next(candidate_reader, None)
                    if candidate is None or candidate[0] != original[0] or candidate[2:] != original[2:]:
                        raise ValueError("样本 ID、顺序或标签发生未授权变化。")
                    expected_text = "".join(c for c in original[1] if c not in clean[key]["characters"]) if key in clean else original[1]
                    if candidate[1] != expected_text:
                        raise ValueError("存在未授权的文本编辑。")
                if next(candidate_reader, None) is not None:
                    raise ValueError("处理后的数据包含多余记录。")
        with staged["quarantine"].open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            if next(reader) != list(EXPECTED_COLUMNS) or list(reader) != quarantine_rows:
                raise ValueError("隔离副本未完整保留原始记录。")
        validation_staging = Path(tempfile.mkdtemp(prefix=".validation-", dir=reports_dir)).resolve()
        if validation_staging.parent != reports_dir:
            raise ValueError("校验暂存目录不在预期位置。")
        post_validation = validate_dataset({split: staged[split] for split in source_paths}, validation_staging, stage="raw", id_scope="split", root=root)
        # 只承接明确批准保留、且执行后证据仍完全一致的重复问题。
        # 隔离会改变记录序号，因此用集合、ID、实际文本和标签的多重集合匹配；
        # 原计划成员若被其他决定隔离或清洗，也必须先应用同一计划再比较。
        def evidence(records, *, apply_plan=False):
            """生成重复组的内容签名，保留成员出现次数，必要时投影到处理后状态。"""
            return Counter((record["split"], record["id"],
                            clean[_key(record)]["text"] if apply_plan and _key(record) in clean else record["text"],
                            tuple(record["labels"][aspect] for aspect in ASPECT_COLUMNS))
                           for record in records if not apply_plan or _key(record) not in quarantine)

        # 未匹配的 REVIEW 是新的未复核风险，不能因为其他组已批准就整体放行。
        # 原扫描计数保留在报告中，另列每个已承接问题对应的原决定编号。
        resolutions = []
        for issue in read_jsonl(validation_staging / "validation_issues.jsonl"):
            if issue["severity"] != "REVIEW":
                continue
            matched = [issue_id for issue_id, source_issue in retained_issues.items()
                       if issue["rule_id"] == source_issue["rule_id"]
                       and evidence(issue["records"]) == evidence(source_issue["records"], apply_plan=True)]
            if not matched:
                raise ValueError("处理后二次校验发现未获保留批准的 REVIEW，拒绝发布。")
            resolutions.append({"issue_id": issue["issue_id"], "source_issue_ids": sorted(matched)})
        if resolutions:
            post_validation["scan_status"] = post_validation["status"]
            post_validation["review_resolutions"] = resolutions
            post_validation["pending_reviews"] = 0
            if not post_validation["issue_counts"]["ERROR"]:
                post_validation["status"] = "passed_with_warnings"
                post_validation["validation_gate_passed"] = True
        if not post_validation["scan_complete"] or post_validation["status"] not in {"passed", "passed_with_warnings"}:
            raise ValueError(f"处理后二次校验未通过：{post_validation['status']}，拒绝发布。")
        for split in source_paths:
            post_validation["files"][split]["path"] = receipt_path(targets[split], root)
        post_validation["artifacts"]["report"] = receipt_path(targets["validation_report"], root)
        post_validation["artifacts"]["issues"] = receipt_path(targets["validation_issues"], root)
        post_validation["review_policy"]["decisions_file"] = receipt_path(reports_dir / "review_decisions.jsonl", root)
        staged["validation_issues"].write_bytes((validation_staging / "validation_issues.jsonl").read_bytes())
        _write_json_atomic(staged["validation_report"], post_validation)
        _write_json_atomic(staged["changes"], changes, jsonl=True)
        applied_at = datetime.now(timezone.utc).isoformat()
        # 原决定仍是处理前快照；实际执行状态另写应用日志，便于独立审计交叉核对。
        applications = [{
            "issue_id": decision["issue_id"], "action_type": decision["action"]["type"],
            "source_sha256": decision["source_sha256"], "action_applied": True,
            "applied_at": applied_at, "post_validation_status": post_validation["status"],
            "affected_records": decision["affected_records"], "label_changes": [],
            "resolution": ("control_characters_removed_in_derived_copy" if decision["action"]["type"] == "remove_reviewed_control_characters"
                           else "reviewed_records_retained_unchanged" if decision["action"]["type"] == "retain_reviewed_records"
                           else "excluded_from_training_and_preserved_in_quarantine"),
            "underlying_annotation_or_source_repair_completed": False,
        } for decision in decisions]
        _write_json_atomic(staged["applications"], applications, jsonl=True)
        if any(file_sha256(path) != fingerprint for path, fingerprint in protected.items()):
            raise ValueError("输入数据或原始报告在处理期间发生变化，拒绝发布。")
        result = {
            "schema_version": "1.0", "processing_policy": "reviewed_exclusion_and_control_character_removal_v1",
            "applied_at": applied_at, "stage": "cleaned_original_label_space", "mapping_applied": False,
            "processed_counts": dict(processed_counts), "quarantine_count": len(quarantine),
            "cleaned_record_counts": cleaned_counts, "removed_control_characters": removed_characters,
            "post_validation_status": post_validation["status"], "post_validation_issue_counts": post_validation["issue_counts"],
            "raw_data_unchanged": True, "original_reports_and_decisions_unchanged": True,
            "labels_unchanged": True, "ids_preserved": True, "source_order_preserved": True,
            "all_retained_records_audited_against_source": True,
            "inputs": {receipt_path(path, root): fingerprint for path, fingerprint in protected.items()},
            "outputs": {name: {"path": receipt_path(targets[name], root), "sha256": file_sha256(path)} for name, path in staged.items() if name != "processing_report"},
            "quarantine_records": [{"split": split, "id": expected_records[(split, number)]["id"],
                                    "source_record_number": number, "issue_ids": issue_ids}
                                   for (split, number), issue_ids in sorted(quarantine.items())],
            "known_validation_label_disputes": sorted(
                {(_key(record), record["id"]): {field: record[field] for field in ("split", "id", "record_number")}
                 for issue in issues if issue.get("label_conflicts")
                 for record in issue["records"] if record["split"] == "validation"}.values(),
                key=lambda record: record["record_number"]),
            "limitations": ["Validation label disputes are recorded separately; no semantic reannotation was performed.",
                            "Passing validation covers the stated structural, label-domain, text and duplicate rules, not all possible semantic noise."],
        }
        _write_json_atomic(staged["processing_report"], result)
        # Windows 重命名可原子发布单个文件，并拒绝覆盖已有目标。
        # POSIX 的重命名可能覆盖目标，因此改用拒绝覆盖的硬链接。
        # 多个文件并非一次整体提交，所以处理报告必须最后发布，作为整批完成凭据。
        for name, temporary in staged.items():
            if os.name == "nt":
                os.rename(temporary, targets[name])
            else:
                os.link(temporary, targets[name])
            published.append(targets[name])
        return result
    except Exception:
        # published 只记录本次成功新建的目标，回滚不得触及既有输入或历史产物。
        for path in reversed(published):
            path.unlink()
        raise
    finally:
        # 无论成功或失败都清理本次暂存文件；不递归删除目录，避免误删其他内容。
        for path in staged.values():
            path.unlink(missing_ok=True)
        if validation_staging is not None:
            for name in ("validation_report.json", "validation_issues.jsonl"):
                (validation_staging / name).unlink(missing_ok=True)
            validation_staging.rmdir()


def main(argv=None) -> int:
    """解析命令行路径并执行处理；未指定的输入、输出位置均随 --root 定位。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="落实已复核的隔离和控制字符清洗，保留原标签并再次校验。")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="项目根目录，凭据内相对路径以此定位。")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--decisions", type=Path)
    parser.add_argument("--review-summary", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quarantine-dir", type=Path)
    parser.add_argument("--reports-dir", type=Path)
    args = parser.parse_args(argv)
    review_dir = args.root / "data/reports/raw"
    try:
        result = process_dataset(args.report or review_dir / "validation_report.json",
                                 args.decisions or review_dir / "review_decisions.jsonl",
                                 args.review_summary or review_dir / "review_summary.json",
                                 args.output_dir or args.root / "data/processed",
                                 args.quarantine_dir or args.root / "data/quarantine",
                                 args.reports_dir or args.root / "data/reports/processed", root=args.root)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in ("processed_counts", "quarantine_count", "cleaned_record_counts", "removed_control_characters", "post_validation_status", "mapping_applied")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
