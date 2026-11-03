"""Apply the reviewed exclusions and exact character edits to derived CSVs.

Raw data, original findings and review decisions remain immutable. Label mapping
is intentionally a separate step: the output keeps the original four labels.
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
    from .label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS, parse_label
    from .validate_dataset import file_sha256, validate_dataset, _write_json_atomic
except ImportError:
    from label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS, parse_label
    from validate_dataset import file_sha256, validate_dataset, _write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_DIR = PROJECT_ROOT / "data" / "reports" / "raw"
QUARANTINE_ACTIONS = {
    "quarantine_training_conflict_cluster", "quarantine_training_overlap",
    "quarantine_corrupted_training_record",
}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _key(record: dict) -> tuple[str, int]:
    return record["split"], record["record_number"]


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def process_dataset(report_path: Path, decisions_path: Path, review_summary_path: Path,
                    output_dir: Path, quarantine_dir: Path, reports_dir: Path) -> dict:
    """Apply a hash-bound reviewed plan, verify lineage, validate, then publish."""
    report_path, decisions_path, review_summary_path = (
        Path(path).resolve() for path in (report_path, decisions_path, review_summary_path)
    )
    output_dir, quarantine_dir, reports_dir = (
        Path(path).resolve() for path in (output_dir, quarantine_dir, reports_dir)
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = json.loads(review_summary_path.read_text(encoding="utf-8"))
    issues_path = Path(report["artifacts"]["issues"]).resolve()
    source_paths = {split: Path(info["path"]).resolve() for split, info in report["files"].items()}
    if (report["stage"] != "raw" or not report["scan_complete"]
            or report["issue_counts"]["ERROR"] or set(source_paths) != {"train", "validation"}):
        raise ValueError("需要完整、无 ERROR 的 train/validation 原始校验报告。")
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

    decisions = _jsonl(decisions_path)
    issues = _jsonl(issues_path)
    issues_by_id = {issue["issue_id"]: issue for issue in issues}
    if (len(issues_by_id) != len(issues)
            or len({decision["issue_id"] for decision in decisions}) != len(decisions)
            or {decision["issue_id"] for decision in decisions} != set(issues_by_id)):
        raise ValueError("处理决定必须完整覆盖所有原始问题，且不能重复。")
    quarantine = {}
    clean = {}
    expected_records = {}
    for decision in decisions:
        issue = issues_by_id[decision["issue_id"]]
        if (decision["source_sha256"] != issue["source_sha256"]
                or decision["rule_id"] != issue["rule_id"]
                or decision["original_severity"] != issue["severity"]
                or decision.get("label_changes") or decision.get("action_applied")):
            raise ValueError("决定的来源、规则或状态不匹配，或包含未授权改标。")
        references = {_key(record): record for record in issue["records"]}
        if {_key(record): record["id"] for record in decision["affected_records"]} != {
            key: record["id"] for key, record in references.items()
        }:
            raise ValueError("决定引用的样本与异常证据不一致。")
        for key, record in references.items():
            if key in expected_records and expected_records[key] != record:
                raise ValueError("同一原始记录的证据不一致。")
            expected_records[key] = record
        action = decision["action"]
        if action["type"] in QUARANTINE_ACTIONS:
            affected = {_key(record): record["id"] for record in action["quarantine"]}
            expected = {key: record["id"] for key, record in references.items() if key[0] == "train"}
            if not affected or affected != expected or any(key[0] != "train" for key in affected):
                raise ValueError("隔离必须且只能覆盖该问题的训练成员，不能移除验证样本。")
            if action["type"] == "quarantine_corrupted_training_record":
                if issue["rule_id"] != "suspicious_characters" or len(references) != 1:
                    raise ValueError("损坏隔离动作与问题类型不符。")
            elif issue["severity"] != "REVIEW":
                raise ValueError("冲突隔离动作必须对应 REVIEW。")
            for key in affected:
                quarantine.setdefault(key, []).append(decision["issue_id"])
        elif action["type"] == "remove_reviewed_control_characters":
            key = _key(action["target"])
            if (issue["rule_id"] != "suspicious_characters" or len(references) != 1
                    or key not in references or action["target"]["id"] != references[key]["id"] or key in clean):
                raise ValueError("字符清洗必须唯一对应已复核的原始记录。")
            code_points = action["remove_code_points"]
            if set(code_points) != set(issue["characters"]) or len(set(code_points)) != len(code_points):
                raise ValueError("清洗字符与原始告警不一致。")
            if any(not re.fullmatch(r"U\+[0-9A-F]{4,6}", value) for value in code_points):
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
        else:
            raise ValueError("存在未授权或未支持的处理动作。")
    if quarantine.keys() & clean.keys():
        raise ValueError("同一记录不能同时隔离和清洗。")
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

        # A separate read verifies every retained ID, label string and order, as
        # well as the exact reviewed edit; the quarantine must retain raw rows.
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
        post_validation = validate_dataset({split: staged[split] for split in source_paths}, validation_staging, stage="raw", id_scope="split")
        if not post_validation["scan_complete"] or post_validation["status"] not in {"passed", "passed_with_warnings"}:
            raise ValueError(f"处理后二次校验未通过：{post_validation['status']}，拒绝发布。")
        for split in source_paths:
            post_validation["files"][split]["path"] = str(targets[split])
        post_validation["artifacts"]["report"] = str(targets["validation_report"])
        post_validation["artifacts"]["issues"] = str(targets["validation_issues"])
        post_validation["review_policy"]["decisions_file"] = str(reports_dir / "review_decisions.jsonl")
        staged["validation_issues"].write_bytes((validation_staging / "validation_issues.jsonl").read_bytes())
        _write_json_atomic(staged["validation_report"], post_validation)
        _write_json_atomic(staged["changes"], changes, jsonl=True)
        applied_at = datetime.now(timezone.utc).isoformat()
        applications = [{
            "issue_id": decision["issue_id"], "action_type": decision["action"]["type"],
            "source_sha256": decision["source_sha256"], "action_applied": True,
            "applied_at": applied_at, "post_validation_status": post_validation["status"],
            "affected_records": decision["affected_records"], "label_changes": [],
            "resolution": "control_characters_removed_in_derived_copy" if decision["action"]["type"] == "remove_reviewed_control_characters" else "excluded_from_training_and_preserved_in_quarantine",
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
            "inputs": {str(path): fingerprint for path, fingerprint in protected.items()},
            "outputs": {name: {"path": str(targets[name]), "sha256": file_sha256(path)} for name, path in staged.items() if name != "processing_report"},
            "quarantine_records": [{"split": split, "id": expected_records[(split, number)]["id"],
                                    "source_record_number": number, "issue_ids": issue_ids}
                                   for (split, number), issue_ids in sorted(quarantine.items())],
            "known_validation_label_disputes": [record for decision in decisions
                                                 for record in decision["action"].get("preserve_fixed_validation", [])],
            "limitations": ["Fixed validation label disputes remain; no semantic reannotation was performed.",
                            "Passing validation covers the stated structural, label-domain, text and duplicate rules, not all possible semantic noise."],
        }
        _write_json_atomic(staged["processing_report"], result)
        # Windows rename is atomic and refuses an existing destination. POSIX
        # rename would overwrite, so use an exclusive link there instead.
        # Publish the processing report last as the completion receipt.
        for name, temporary in staged.items():
            if os.name == "nt":
                os.rename(temporary, targets[name])
            else:
                os.link(temporary, targets[name])
            published.append(targets[name])
        return result
    except Exception:
        # These paths were exclusively created by this invocation, never inputs.
        for path in reversed(published):
            path.unlink()
        raise
    finally:
        for path in staged.values():
            path.unlink(missing_ok=True)
        if validation_staging is not None:
            for name in ("validation_report.json", "validation_issues.jsonl"):
                (validation_staging / name).unlink(missing_ok=True)
            validation_staging.rmdir()


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="落实已复核的隔离和控制字符清洗，保留原标签并再次校验。")
    parser.add_argument("--report", type=Path, default=DEFAULT_REVIEW_DIR / "validation_report.json")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_REVIEW_DIR / "review_decisions.jsonl")
    parser.add_argument("--review-summary", type=Path, default=DEFAULT_REVIEW_DIR / "review_summary.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--quarantine-dir", type=Path, default=PROJECT_ROOT / "data" / "quarantine")
    parser.add_argument("--reports-dir", type=Path, default=PROJECT_ROOT / "data" / "reports" / "processed")
    args = parser.parse_args(argv)
    try:
        result = process_dataset(args.report, args.decisions, args.review_summary, args.output_dir, args.quarantine_dir, args.reports_dir)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in ("processed_counts", "quarantine_count", "cleaned_record_counts", "removed_control_characters", "post_validation_status", "mapping_applied")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
