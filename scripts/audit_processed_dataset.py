"""Independently audit retained text and exclusions without changing datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

try:
    from .label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS
    from .validate_dataset import file_sha256, validate_dataset, _write_json_atomic
except ImportError:
    from label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS
    from validate_dataset import file_sha256, validate_dataset, _write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FILENAMES = {"train": "sentiment_analysis_trainingset.csv",
             "validation": "sentiment_analysis_validationset.csv"}


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_processed_dataset(root: Path = PROJECT_ROOT) -> tuple[dict, list[dict], list[dict]]:
    """Read every source/output pair and compare with the original review plan.

    No processing function or previous validation result is used as proof of text
    preservation. Only a fresh read and exact comparisons establish that result.
    This function performs no writes, so it is also safe to call from a notebook.
    """
    root = Path(root).resolve()
    raw_reports = root / "data/reports/raw"
    processed_reports = root / "data/reports/processed"
    processing_path = processed_reports / "processing_report.json"
    processing = json.loads(processing_path.read_text(encoding="utf-8"))
    protected = {processing_path: file_sha256(processing_path)}
    for name, expected in processing["inputs"].items():
        path = Path(name)
        protected[path] = file_sha256(path)
        _check(protected[path] == expected, f"输入摘要不匹配：{path}")
    for artifact in processing["outputs"].values():
        path = Path(artifact["path"])
        protected[path] = file_sha256(path)
        _check(protected[path] == artifact["sha256"], f"处理产物摘要不匹配：{path}")

    raw_paths = {split: root / "data/raw" / name for split, name in FILENAMES.items()}
    processed_paths = {split: root / "data/processed" / name for split, name in FILENAMES.items()}
    for path in (*raw_paths.values(), *processed_paths.values()):
        _check(path in protected, f"数据路径没有绑定处理凭据：{path}")
    issues = _jsonl(raw_reports / "validation_issues.jsonl")
    decisions = _jsonl(raw_reports / "review_decisions.jsonl")
    changes = _jsonl(processed_reports / "processing_changes.jsonl")
    applications = _jsonl(processed_reports / "application_results.jsonl")
    _check(len({item["issue_id"] for item in issues}) == len(issues), "原始问题编号重复。")
    _check(len({item["issue_id"] for item in decisions}) == len(decisions), "复核决定编号重复。")
    _check({item["issue_id"] for item in issues} == {item["issue_id"] for item in decisions}
           == {item["issue_id"] for item in applications}, "问题、决定、应用日志未完整对应。")
    _check(len(applications) == len(decisions), "应用日志条数不符。")
    by_issue = {item["issue_id"]: item for item in issues}
    quarantine = {}
    edits_plan = {}
    for decision in decisions:
        action = decision["action"]
        if action["type"] in {"quarantine_training_conflict_cluster", "quarantine_training_overlap",
                              "quarantine_corrupted_training_record"}:
            for record in action["quarantine"]:
                _check(record["split"] == "train", "固定验证集不能被隔离。")
                key = record["split"], record["record_number"]
                if key in quarantine:
                    _check(quarantine[key]["id"] == record["id"], "隔离记录身份冲突。")
                quarantine[key] = record
        elif action["type"] == "remove_reviewed_control_characters":
            record = action["target"]
            key = record["split"], record["record_number"]
            _check(key not in edits_plan, "同一记录存在多项编辑。")
            chars = {chr(int(value[2:], 16)) for value in action["remove_code_points"]}
            _check(bool(chars) and all(unicodedata.category(char) == "Cc" and char not in "\t\r\n"
                                      for char in chars), "计划包含普通字符或正常换行/制表符。")
            edits_plan[key] = {"id": record["id"], "characters": chars, "issue_id": decision["issue_id"]}
        else:
            raise ValueError(f"未知处理动作：{action['type']}")
    _check(not quarantine.keys() & edits_plan.keys(), "隔离与编辑目标相交。")

    log_by_key = {(item["split"], item["source_record_number"]): item for item in changes}
    _check(len(log_by_key) == len(changes) and log_by_key.keys() == edits_plan.keys(), "修改日志与计划不对应。")
    raw_counts, retained_counts, unchanged_counts = Counter(), Counter(), Counter()
    removed_chars = Counter()
    quarantined_rows = []
    edit_evidence, dispute_evidence = [], []
    dispute_ids = {item["id"] for item in processing["known_validation_label_disputes"]}
    seen_edits, seen_quarantine, seen_disputes = set(), set(), set()
    label_cells = 0
    for split in FILENAMES:
        with raw_paths[split].open(encoding="utf-8-sig", newline="") as source_stream, \
                processed_paths[split].open(encoding="utf-8-sig", newline="") as result_stream:
            source_reader = csv.reader(source_stream, strict=True)
            result_reader = csv.reader(result_stream, strict=True)
            _check(next(source_reader) == next(result_reader) == list(EXPECTED_COLUMNS), "表头不符。")
            for number, original in enumerate(source_reader, start=1):
                raw_counts[split] += 1
                _check(len(original) == len(EXPECTED_COLUMNS), f"源记录列宽异常：{split}/{number}")
                key = split, number
                if key in quarantine:
                    _check(original[0] == quarantine[key]["id"], "隔离计划 ID 不符。")
                    quarantined_rows.append(original)
                    seen_quarantine.add(key)
                    continue
                candidate = next(result_reader, None)
                _check(candidate is not None and len(candidate) == len(EXPECTED_COLUMNS), f"缺少或损坏输出：{split}/{number}")
                _check(candidate[0] == original[0], f"保留记录 ID 或顺序改变：{split}/{number}")
                _check(candidate[2:] == original[2:], f"标签被修改：{split}/{number}")
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
                    for position, char in enumerate(original[1]):
                        if char in plan["characters"]:
                            code = f"U+{ord(char):04X}"
                            counts[code] += 1
                            positions.append({"original_position": position, "code_point": code,
                                              "python_isspace": char.isspace(),
                                              "context_before": original[1][max(0, position - 32):position],
                                              "context_after": original[1][position + 1:position + 33]})
                    log = log_by_key[key]
                    _check(log["id"] == original[0] and log["issue_id"] == plan["issue_id"]
                           and log["source_sha256"] == protected[raw_paths[split]]
                           and log["original_text_sha256"] == _text_sha(original[1])
                           and log["processed_text_sha256"] == _text_sha(candidate[1])
                           and log["original_length"] == len(original[1])
                           and log["processed_length"] == len(candidate[1])
                           and log["removed_characters"] == dict(counts) and not log["labels_changed"],
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
    quarantine_path = Path(processing["outputs"]["quarantine"]["path"])
    with quarantine_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        _check(next(reader) == list(EXPECTED_COLUMNS) and list(reader) == quarantined_rows, "隔离副本未完整保留原文。")
    _check(seen_edits == edits_plan.keys() and seen_quarantine == quarantine.keys()
           and seen_disputes == dispute_ids, "部分计划目标未在原始数据中找到。")
    _check(dict(retained_counts) == processing["processed_counts"] and len(quarantined_rows) == processing["quarantine_count"],
           "复算条数与处理报告不符。")
    _check(sum(removed_chars.values()) == processing["removed_control_characters"], "复算删除字符总数不符。")
    _check(all(file_sha256(path) == value for path, value in protected.items()), "复核期间输入发生变化。")
    report = {
        "schema_version": "1.0", "audit_method_version": "1.0.0",
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
        "file_sha256": {str(path): value for path, value in protected.items()},
        "limitations": ["字符未误删不等于原始文本没有噪声或缺字。",
                        "控制符也可能充当 token 分隔，需结合逐条上下文复核。",
                        "除两条指定验证样本外，不进行全量20方面语义重标。",
                        "本次语义建议不修改固定验证标签，也不作为权威替代真值。"]}
    return report, edit_evidence, dispute_evidence


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="独立只读复核处理后文本与来源，另存证据和二次校验。")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data/reports/processed_review")
    args = parser.parse_args(argv)
    root, output_dir = args.root.resolve(), args.output_dir.resolve()
    targets = [output_dir / name for name in ("lineage_audit.json", "text_edit_evidence.json", "validation_dispute_evidence.json")]
    structural_dir = output_dir / "structural_validation"
    try:
        forbidden = [root / name for name in ("data/raw", "data/processed", "data/quarantine", "data/reports/raw", "data/reports/processed")]
        _check(not any(output_dir.is_relative_to(path) for path in forbidden), "复核输出不能写入现有数据或报告目录。")
        _check(not any(path.exists() or path.is_symlink() for path in targets)
               and not structural_dir.exists(), "复核产物已存在，请使用新的输出目录。")
        report, edits, disputes = audit_processed_dataset(root)
        output_dir.mkdir(parents=True, exist_ok=True)
        validation = validate_dataset({split: root / "data/processed" / name for split, name in FILENAMES.items()},
                                      structural_dir, stage="raw", id_scope="split")
        _check(all(file_sha256(Path(path)) == value for path, value in report["file_sha256"].items()), "二次校验期间输入发生变化。")
        report["fresh_validation"] = {"status": validation["status"], "scan_complete": validation["scan_complete"],
                                      "issue_counts": validation["issue_counts"]}
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
