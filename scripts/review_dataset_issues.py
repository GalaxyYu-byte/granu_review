"""Reproduce the reviewed recommendations and a read-only processing dry run.

The JSON notes are an assistant-reviewed snapshot, not an automatic relabeler.
This script never creates processed datasets or edits validation findings.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timedelta, timezone
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

try:
    from .label_config import ASPECT_COLUMNS, CLASS_VALUES, EXPECTED_COLUMNS, parse_label
    from .validate_dataset import file_sha256, normalize_for_comparison, _write_json_atomic
except ImportError:
    from label_config import ASPECT_COLUMNS, CLASS_VALUES, EXPECTED_COLUMNS, parse_label
    from validate_dataset import file_sha256, normalize_for_comparison, _write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reports" / "raw" / "validation_report.json"
NOTE_NAMES = ("review_notes_duplicates_a.json", "review_notes_duplicates_b.json", "review_notes_characters.json")


def _identity(record: dict) -> tuple[str, int]:
    return record["split"], record["record_number"]


def _reference(record: dict) -> dict:
    return {key: record[key] for key in ("split", "id", "record_number")}


def _suspicious(text: str) -> Counter:
    return Counter(c for c in text if c == "\ufffd"
                   or (unicodedata.category(c) == "Cc" and c not in "\t\r\n"))


def _character_spans(text: str, characters: set[str]) -> list[dict]:
    result = []
    position = 0
    while position < len(text):
        if text[position] not in characters:
            position += 1
            continue
        end = position + 1
        while end < len(text) and text[end] in characters:
            end += 1
        result.append({
            "start": position, "end_exclusive": end,
            "characters": {f"U+{ord(c):04X}": n for c, n in sorted(Counter(text[position:end]).items())},
            "context_before": text[max(0, position - 24):position],
            "original_span": text[position:end], "context_after": text[end:end + 24],
        })
        position = end
    return result


def build_review(report_path: Path = DEFAULT_REPORT) -> tuple[list[dict], dict]:
    """Verify the evidence and return recommendations plus a full-population dry run."""
    report_path = Path(report_path).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["stage"] != "raw" or not report["scan_complete"] or report["issue_counts"]["ERROR"]:
        raise ValueError("这份复核只适用于完成扫描、没有 ERROR 的原始数据校验报告。")
    issues_path = Path(report["artifacts"]["issues"])
    if file_sha256(issues_path) != report["artifacts"]["issues_sha256"]:
        raise ValueError("问题明细摘要与校验报告不一致。")
    issues = [json.loads(line) for line in issues_path.read_text(encoding="utf-8").splitlines() if line]
    notes = {}
    note_hashes = {}
    for name in NOTE_NAMES:
        path = report_path.parent / name
        note_hashes[name] = file_sha256(path)
        for note in json.loads(path.read_text(encoding="utf-8")):
            if note["issue_id"] in notes:
                raise ValueError("复核注释中有重复 issue_id。")
            notes[note["issue_id"]] = note
    if len({item["issue_id"] for item in issues}) != len(issues) or set(notes) != {item["issue_id"] for item in issues}:
        raise ValueError("复核注释必须与本次问题明细一一对应，不能自动套用到新问题。")
    for split, info in report["files"].items():
        if file_sha256(Path(info["path"])) != info["sha256"]:
            raise ValueError(f"{split} 源文件发生变化，需要重新校验和复核。")

    reviewed_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    decisions = []
    expected_records = {}
    quarantine = set()
    replacements = {}
    preserved_validation = set()
    for issue in issues:
        note = notes[issue["issue_id"]]
        records = issue["records"]
        for record in records:
            expected_hash = report["files"][record["split"]]["sha256"]
            if issue["source_sha256"].get(record["split"]) != expected_hash:
                raise ValueError("问题记录的源摘要与报告不一致。")
            key = _identity(record)
            if key in expected_records and expected_records[key] != record:
                raise ValueError("同一原始记录在不同问题中的证据不一致。")
            expected_records[key] = record
        if not note["evidence_quotes"] or any(
            not any(quote in record["text"] for record in records) for quote in note["evidence_quotes"]
        ):
            raise ValueError(f"{issue['issue_id']} 的证据引文不在原始文本中。")
        decision = {
            "schema_version": "1.0", "review_method_version": "1.0.0",
            "issue_id": issue["issue_id"], "rule_id": issue["rule_id"],
            "original_severity": issue["severity"],
            "source_sha256": issue["source_sha256"],
            "status": "reviewed_pending_application", "reviewed_by": "Codex assistant",
            "reviewed_at": reviewed_at, "action_applied": False,
            "conclusion": "", "reason": note["reason"], "action": {},
            "evidence_quotes": note["evidence_quotes"],
            "affected_records": [_reference(record) for record in records],
            "label_changes": [],
        }
        if issue["severity"] == "REVIEW":
            if issue["rule_id"] not in {"within_split_normalized_duplicate", "cross_split_normalized_overlap"}:
                raise ValueError("遇到未覆盖的复核规则，请新增具体复核结论。")
            base = records[0]["text"]
            if not all(re.sub(r"\s+", "", record["text"]) == re.sub(r"\s+", "", base) for record in records):
                raise ValueError("文本存在非空白差异，不能使用本次同内容排版变体结论。")
            differences = []
            for record in records[1:]:
                changes = [
                    {"operation": operation, "base_start": a, "base_end": b,
                     "other_start": c, "other_end": d,
                     "base_span": base[a:b], "other_span": record["text"][c:d]}
                    for operation, a, b, c, d in difflib.SequenceMatcher(None, base, record["text"], autojunk=False).get_opcodes()
                    if operation != "equal"
                ]
                if any(change["base_span"].strip() or change["other_span"].strip() for change in changes):
                    raise ValueError("差异块包含非空白字符。")
                differences.append({"record": _reference(record), "changes": changes})
            train_records = [record for record in records if record["split"] == "train"]
            validation_records = [record for record in records if record["split"] == "validation"]
            if len(train_records) + len(validation_records) != len(records):
                raise ValueError("这份处理建议仅定义了 train 与 validation 两个集合。")
            quarantine.update(_identity(record) for record in train_records)
            preserved_validation.update(_identity(record) for record in validation_records)
            decision.update({
                "conclusion": "same_content_layout_variants_with_label_conflicts",
                "confidence": {"text_equivalence": "high", "correct_label_vector": "undetermined"},
                "label_resolution": note["label_resolution"],
                "label_conflicts": issue["label_conflicts"], "conflicting_aspect_count": len(issue["label_conflicts"]),
                "comparison_evidence": {"non_whitespace_characters_equal": True,
                                        "only_whitespace_edits": True, "differences_from_first_record": differences},
                "action": {
                    "type": "quarantine_training_overlap" if validation_records else "quarantine_training_conflict_cluster",
                    "quarantine": [_reference(record) for record in train_records],
                    "preserve_fixed_validation": [_reference(record) for record in validation_records],
                    "reannotation_policy": "Use a consistent 20-aspect rubric before restoring at most one training representative; never overwrite benchmark labels from a training member.",
                },
            })
        elif issue["rule_id"] == "suspicious_characters" and len(records) == 1:
            record = records[0]
            text = record["text"]
            characters = {chr(int(value[2:], 16)) for value in issue["characters"]}
            actual = {f"U+{ord(char):04X}": count for char, count in sorted(_suspicious(text).items())}
            if actual != issue["characters"]:
                raise ValueError("异常字符计数与证据不一致。")
            decision.update({
                "conclusion": note["recommended_action"], "confidence": note["confidence"],
                "cause_hypothesis": note["cause_hypothesis"],
                "characters": actual, "character_count": sum(actual.values()),
                "position_convention": "0-based Python Unicode code-point positions in parsed content, including literal outer quotes",
                "character_spans": _character_spans(text, characters),
            })
            if note["recommended_action"] == "quarantine_corrupted_training_record":
                if record["split"] != "train":
                    raise ValueError("当前方案不移除固定验证样本。")
                quarantine.add(_identity(record))
                decision["action"] = {"type": note["recommended_action"], "quarantine": [_reference(record)],
                                      "restore_condition": "Recover a reliable original text and review it; do not guess missing characters or globally delete ordinary letters."}
            elif note["recommended_action"] == "remove_reviewed_control_characters":
                if any(unicodedata.category(c) != "Cc" or c in "\t\r\n" for c in characters):
                    raise ValueError("仅允许移除本条已复核的异常控制字符，不自动删除替换字符或正常空白。")
                cleaned = "".join(c for c in text if c not in characters)
                if not cleaned.strip():
                    raise ValueError("候选清洗产生空文本。")
                replacements[_identity(record)] = cleaned
                if record["split"] == "validation":
                    preserved_validation.add(_identity(record))
                decision["action"] = {
                    "type": note["recommended_action"], "target": _reference(record),
                    "remove_code_points": sorted(issue["characters"]), "replacement": "",
                    "original_length": len(text), "candidate_length": len(cleaned),
                    "candidate_text_sha256": hashlib.sha256(cleaned.encode("utf-8")).hexdigest(),
                    "preserve_labels": True, "preserve_normal_whitespace_emoji_and_emotion_tokens": True,
                    "destination": "future processed copy or identical versioned model preprocessing for train and validation; raw remains unchanged",
                }
            else:
                raise ValueError("未支持的字符处理建议。")
        else:
            raise ValueError("遇到尚未复核的问题类型。")
        decisions.append(decision)

    # Simulate the proposed record selection and the exact reviewed text edits.
    # Only keys, counts and affected evidence are retained; no processed CSV is written.
    if quarantine & replacements.keys():
        raise ValueError("同一记录被同时指定隔离与清洗。")
    before = Counter()
    after = Counter()
    source_verified = set()
    exact = defaultdict(list)
    normalized = defaultdict(list)
    candidate_suspicious = candidate_empty = 0
    cleaned_count = Counter()
    candidate_label_counts = {split: {aspect: Counter() for aspect in ASPECT_COLUMNS} for split in report["files"]}
    for split, info in report["files"].items():
        with Path(info["path"]).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(EXPECTED_COLUMNS):
                raise ValueError("源文件表头与校验规范不一致。")
            for number, row in enumerate(reader, start=1):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError("源文件出现列宽异常。")
                before[split] += 1
                key = (split, number)
                labels = {aspect: parse_label(row[aspect]) for aspect in ASPECT_COLUMNS}
                if key in expected_records:
                    expected = expected_records[key]
                    if expected["id"] != row["id"] or expected["text"] != row["content"] or expected["labels"] != labels:
                        raise ValueError("已记录的问题证据与原始 CSV 不符。")
                    source_verified.add(key)
                if key in quarantine:
                    continue
                after[split] += 1
                text = replacements.get(key, row["content"])
                cleaned_count[split] += key in replacements
                candidate_suspicious += bool(_suspicious(text))
                candidate_empty += not bool(text.strip())
                for aspect, value in labels.items():
                    candidate_label_counts[split][aspect][str(value)] += 1
                exact[hashlib.sha256(text.encode("utf-8")).digest()].append(key)
                normalized[hashlib.sha256(normalize_for_comparison(text).encode("utf-8")).digest()].append(key)
        if before[split] != info["records_scanned"] or file_sha256(Path(info["path"])) != info["sha256"]:
            raise ValueError("源文件条数或摘要变化，预演结果无效。")
    if source_verified != expected_records.keys():
        raise ValueError("未在原始 CSV 找全问题中的记录。")
    normalized_duplicates = [group for group in normalized.values() if len(group) > 1]
    clean_keys = set(replacements)
    cleanup_collisions = [group for group in normalized_duplicates if clean_keys.intersection(group)]
    quarantine_references = sorted((_reference(expected_records[key]) for key in quarantine), key=lambda r: (r["split"], r["record_number"]))
    summary = {
        "schema_version": "1.0", "reviewed_at": reviewed_at,
        "review_complete": True, "implementation_status": "not_applied",
        "raw_data_modified": False, "original_validation_report_modified": False,
        "source_report_sha256": file_sha256(report_path), "source_issues_sha256": file_sha256(issues_path),
        "review_note_sha256": note_hashes,
        "source_sha256": {split: info["sha256"] for split, info in report["files"].items()},
        "reviewed_issue_count": len(decisions), "reviewed_issue_counts": dict(Counter(item["original_severity"] for item in decisions)),
        "unique_source_records_verified": len(source_verified),
        "action_issue_counts": dict(Counter(item["action"]["type"] for item in decisions)),
        "quarantine_training_records": quarantine_references,
        "quarantine_count": len(quarantine),
        "quarantine_fraction_of_training": len(quarantine) / before["train"],
        "preserved_validation_records": [_reference(expected_records[key]) for key in sorted(preserved_validation)],
        "proposed_cleaned_record_counts": dict(cleaned_count),
        "proposed_removed_control_characters": sum(
            item.get("character_count", 0) for item in decisions if item["action"]["type"] == "remove_reviewed_control_characters"
        ),
        "dry_run": {
            "scope": "full input population, proposed exclusion and exact reviewed control-character removal; labels unchanged",
            "actual_before_counts": dict(before), "expected_after_counts": dict(after),
            "candidate_empty_texts": candidate_empty,
            "candidate_suspicious_character_records": candidate_suspicious,
            "candidate_exact_duplicate_groups": sum(len(group) > 1 for group in exact.values()),
            "candidate_normalized_duplicate_groups": len(normalized_duplicates),
            "candidate_cross_split_normalized_groups": sum(len({split for split, _ in group}) > 1 for group in normalized_duplicates),
            "cleanup_collision_groups": len(cleanup_collisions),
            "candidate_label_distribution": candidate_label_counts,
            "all_four_classes_remain_per_aspect_per_split": all(
                set(counter) == {str(value) for value in CLASS_VALUES}
                for aspects in candidate_label_counts.values() for counter in aspects.values()
            ),
            "processed_files_written": False,
        },
        "limitations": [
            "Assistant-reviewed recommendations are not authoritative replacement labels; no labels were changed.",
            "Whitespace-only content equivalence does not prove that upstream global review identifiers are identical.",
            "Fixed validation labels remain disputed for two retained records; isolation removes overlap, not reference-label noise.",
            "The full-population dry run covers the stated rules, not fuzzy semantic deduplication or a complete label-quality audit.",
            "No temporal trend assessment is possible: these CSVs have no review timestamps.",
            "Apply the chosen policy to a versioned processed copy and rerun validation before closing the original issues.",
        ],
    }
    return decisions, summary


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="生成逐问题复核建议并在内存预演，不清洗或改动数据。")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_dir = (args.output_dir or args.report.resolve().parent).resolve()
    targets = [output_dir / name for name in ("review_decisions.jsonl", "review_summary.json")]
    try:
        if any(path.exists() or path.is_symlink() for path in targets):
            raise ValueError("复核输出已存在，不能覆盖既有决定；重算可调用 build_review，或使用新的 --output-dir。")
        decisions, summary = build_review(args.report)
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if any(output_dir.is_relative_to(Path(info["path"]).resolve().parent) for info in report["files"].values()):
            raise ValueError("复核输出不能写入原始数据目录。")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(targets[0], decisions, jsonl=True)
        summary["artifacts"] = {"decisions": str(targets[0]), "decisions_sha256": file_sha256(targets[0])}
        _write_json_atomic(targets[1], summary)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps({"reviewed_issues": summary["reviewed_issue_count"], "proposed_quarantine": summary["quarantine_count"],
                      "proposed_cleaned_record_counts": summary["proposed_cleaned_record_counts"],
                      "expected_after_counts": summary["dry_run"]["expected_after_counts"],
                      "action_applied": False}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
