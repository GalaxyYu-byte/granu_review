"""依据已有证据复算逐问题复核建议，并对全量数据进行只读处理预演。

复核注释记录的是助手已经审阅过的一批问题，不是自动重标规则。运行时重新核对输入摘要、
问题身份、原始记录及证据引文，只有这些内容一致才复用注释。
本模块只返回建议和候选处理统计，不生成清洗后的数据，也不修改原始问题或验证集标签。
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
    from .dataset_io import read_jsonl, receipt_origin, receipt_path, resolve_receipt_path
    from .label_config import ASPECT_COLUMNS, CLASS_VALUES, EXPECTED_COLUMNS, parse_label
    from .validate_dataset import file_sha256, normalize_for_comparison, _write_json_atomic
except ImportError:
    from dataset_io import read_jsonl, receipt_origin, receipt_path, resolve_receipt_path
    from label_config import ASPECT_COLUMNS, CLASS_VALUES, EXPECTED_COLUMNS, parse_label
    from validate_dataset import file_sha256, normalize_for_comparison, _write_json_atomic


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = PROJECT_ROOT / "data" / "reports" / "raw" / "validation_report.json"
NOTE_NAMES = ("review_notes_duplicates_a.json", "review_notes_duplicates_b.json", "review_notes_characters.json")


def _identity(record: dict) -> tuple[str, int]:
    """以集合和原始记录序号定位样本，避免不同集合的同名编号发生冲突。"""
    return record["split"], record["record_number"]


def _reference(record: dict) -> dict:
    """提取集合、编号和记录序号，供决定中的目标引用与原始证据逐项核对。"""
    return {key: record[key] for key in ("split", "id", "record_number")}


def _suspicious(text: str) -> Counter:
    """统计替换字符和异常控制符；正常换行、回车、制表符不属于本次清洗范围。"""
    return Counter(c for c in text if c == "\ufffd"
                   or (unicodedata.category(c) == "Cc" and c not in "\t\r\n"))


def _character_spans(text: str, characters: set[str]) -> list[dict]:
    """将连续的待处理字符合并为区段，并保留原文两侧上下文供人工复核。

    位置按解析后的评论文本计算，从零开始，右端位置不包含在区段内；不能直接当作文件字节偏移。
    """
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


def build_review(report_path: Path = DEFAULT_REPORT, *, root: Path = PROJECT_ROOT) -> tuple[list[dict], dict]:
    """验证复核依据，返回逐问题决定及全量候选数据的只读预演汇总。

    先绑定报告、问题明细、注释与原始文件，再区分重复冲突、集合重叠和字符异常。
    随后重新读取全部原始记录，核对证据并统计候选结果；复核完成不等于动作已经应用。
    标注争议只进入汇总，不以训练标签替换固定验证集标签。
    """
    report_path = Path(report_path).resolve()
    root = Path(root).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    # 旧凭据只按已知报告的完整相对后缀推导来源根目录，避免把外部同名数据目录误认成副本。
    legacy_root = receipt_origin(report["artifacts"]["report"], report_path, root)
    if report["stage"] != "raw" or not report["scan_complete"] or report["issue_counts"]["ERROR"]:
        raise ValueError("这份复核只适用于完成扫描、没有 ERROR 的原始数据校验报告。")
    issues_path = resolve_receipt_path(report["artifacts"]["issues"], root=root, legacy_root=legacy_root)
    if file_sha256(issues_path) != report["artifacts"]["issues_sha256"]:
        raise ValueError("问题明细摘要与校验报告不一致。")
    issues = read_jsonl(issues_path)
    # 注释与本批问题必须一一对应；摘要或问题编号变化后不能把旧解释直接套给新数据。
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
        if file_sha256(resolve_receipt_path(info["path"], root=root, legacy_root=legacy_root)) != info["sha256"]:
            raise ValueError(f"{split} 源文件发生变化，需要重新校验和复核。")

    reviewed_at = datetime.now(timezone(timedelta(hours=8))).isoformat()
    decisions = []
    expected_records = {}
    quarantine = set()
    replacements = {}
    preserved_validation = set()
    disputed_validation = set()
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
        # 引文必须确实出现在本问题原文中；有复核理由但缺乏可定位证据时同样拒绝复用。
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
        if issue["rule_id"] in {
            "within_split_exact_duplicate", "within_split_normalized_duplicate",
            "cross_split_exact_overlap", "cross_split_normalized_overlap",
        }:
            # 仅允许本批已经复核的空白排版差异；相似语义或非空白差异需要另行给出结论。
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
            conflicts = issue["label_conflicts"]
            # 从各记录的二十方面标签重新计算冲突，防止“无冲突”的汇总与实际标签向量脱节。
            actual_conflicts = {
                aspect: [{**_reference(record), "value": record["labels"][aspect]} for record in records]
                for aspect in ASPECT_COLUMNS
                if len({record["labels"][aspect] for record in records}) > 1
            }
            if conflicts != actual_conflicts:
                raise ValueError("重复组的标签冲突明细与记录不一致。")
            cross_split = bool(train_records and validation_records)
            if cross_split or (conflicts and train_records):
                quarantine.update(_identity(record) for record in train_records)
            preserved_validation.update(_identity(record) for record in validation_records)
            if conflicts:
                disputed_validation.update(_identity(record) for record in validation_records)
            if cross_split:
                # 跨集合重叠即使标签相同也存在评估泄漏风险，仅隔离训练侧并完整保留固定验证集。
                action = {
                    "type": "quarantine_training_overlap",
                    "quarantine": [_reference(record) for record in train_records],
                    "preserve_fixed_validation": [_reference(record) for record in validation_records],
                    "reannotation_policy": "Review conflicting labels using a consistent 20-aspect rubric; never overwrite benchmark labels from a training member.",
                }
            elif conflicts and train_records:
                # 集合内的训练冲突暂不选择某条标签作为真值，先隔离整组等待统一标注口径。
                action = {
                    "type": "quarantine_training_conflict_cluster",
                    "quarantine": [_reference(record) for record in train_records],
                    "preserve_fixed_validation": [],
                    "reannotation_policy": "Use a consistent 20-aspect rubric before restoring at most one training representative.",
                }
            else:
                # 同标签重复按明确策略保留；验证集内部冲突也保留，但继续单独登记未解决争议。
                action = {
                    "type": "retain_reviewed_records",
                    "preserve_fixed_validation": [_reference(record) for record in validation_records],
                    "retention_policy": "Preserve the reviewed records and all original labels; no training deduplication or benchmark reannotation is applied.",
                }
            equivalence = "same_content" if issue["comparison"] == "exact" else "same_content_layout_variants"
            decision.update({
                "conclusion": equivalence + ("_with_label_conflicts" if conflicts else "_with_consistent_labels"),
                "confidence": {"text_equivalence": "high", "correct_label_vector": "undetermined" if conflicts else "not_independently_verified"},
                "label_resolution": note.get("label_resolution", "Original labels retained; matching labels are not independent proof of label correctness."),
                "label_conflicts": conflicts, "conflicting_aspect_count": len(conflicts),
                "comparison_evidence": {"non_whitespace_characters_equal": True,
                                        "only_whitespace_edits": True, "differences_from_first_record": differences},
                "action": action,
            })
        elif issue["rule_id"] == "suspicious_characters" and len(records) == 1:
            # 字符数量、原文区段和清洗后摘要都来自当前记录，不能把某条记录的清洗动作扩散到整列。
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
                # 只删除已定位的异常控制符，保留正常空白、表情和正文；替换字符的缺字不能靠猜测修复。
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

    # 重新读取全量输入，在内存中预演记录选择和已复核的逐字符编辑；此处不会写出处理后的数据。
    # 同一样本不能同时要求隔离和清洗，否则无法明确后续应用与日志各自应证明的结果。
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
        source_path = resolve_receipt_path(info["path"], root=root, legacy_root=legacy_root)
        with source_path.open(encoding="utf-8-sig", newline="") as stream:
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
                    # 按原始序号核对编号、完整文本及全部标签，防止仅凭文件摘要遗漏了错误的记录引用。
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
        # 扫描结束再次核对条数与摘要，避免读入期间文件变化造成前后混合的预演结果。
        if before[split] != info["records_scanned"] or file_sha256(source_path) != info["sha256"]:
            raise ValueError("源文件条数或摘要变化，预演结果无效。")
    if source_verified != expected_records.keys():
        raise ValueError("未在原始 CSV 找全问题中的记录。")
    normalized_duplicates = [group for group in normalized.values() if len(group) > 1]
    clean_keys = set(replacements)
    cleanup_collisions = [group for group in normalized_duplicates if clean_keys.intersection(group)]
    quarantine_references = sorted((_reference(expected_records[key]) for key in quarantine), key=lambda r: (r["split"], r["record_number"]))
    # 汇总绑定输入与注释摘要，给实际处理阶段提供数量预期；争议记录去重统计，不能写死为某一批数量。
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
        "known_validation_label_disputes": [_reference(expected_records[key]) for key in sorted(disputed_validation)],
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
            f"{len(disputed_validation)} retained validation records occur in groups with label conflicts; preserving them does not resolve reference-label noise.",
            "The full-population dry run covers the stated rules, not fuzzy semantic deduplication or a complete label-quality audit.",
            "No temporal trend assessment is possible: these CSVs have no review timestamps.",
            "Apply the chosen policy to a versioned processed copy and rerun validation before closing the original issues.",
        ],
    }
    return decisions, summary


def main(argv=None) -> int:
    """读取指定项目的原始报告，将复算决定和汇总发布到新的复核目录。

    默认报告随项目根目录定位；已存在的决定和汇总拒绝覆盖，输入数据目录也不能作为输出目录。
    保存的决定路径优先采用项目相对路径，便于复制目录后继续执行后续处理。
    """
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="生成逐问题复核建议并在内存预演，不清洗或改动数据。")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args(argv)
    report_path = args.report or args.root / "data/reports/raw/validation_report.json"
    output_dir = (args.output_dir or report_path.resolve().parent).resolve()
    targets = [output_dir / name for name in ("review_decisions.jsonl", "review_summary.json")]
    try:
        if any(path.exists() or path.is_symlink() for path in targets):
            raise ValueError("复核输出已存在，不能覆盖既有决定；重算可调用 build_review，或使用新的 --output-dir。")
        decisions, summary = build_review(report_path, root=args.root)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        legacy_root = receipt_origin(report["artifacts"]["report"], report_path, args.root)
        if any(output_dir.is_relative_to(resolve_receipt_path(info["path"], root=args.root, legacy_root=legacy_root).parent)
               for info in report["files"].values()):
            raise ValueError("复核输出不能写入原始数据目录。")
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(targets[0], decisions, jsonl=True)
        summary["artifacts"] = {"decisions": receipt_path(targets[0], root=args.root), "decisions_sha256": file_sha256(targets[0])}
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
