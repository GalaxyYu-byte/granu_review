"""Read-only validation for the 20-aspect sentiment classification CSVs.

Only the Python standard library is required. Run with --help for CLI options.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
import unicodedata


try:
    from .label_config import (
        ASPECT_COLUMNS, EXPECTED_COLUMNS, LABEL_MAPPING, RAW_LABEL_MEANINGS,
        InvalidLabelError, parse_label,
    )
except ImportError:
    from label_config import (
        ASPECT_COLUMNS, EXPECTED_COLUMNS, LABEL_MAPPING, RAW_LABEL_MEANINGS,
        InvalidLabelError, parse_label,
    )


RULES_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0"
NORMALIZATION_RULE = "unicode_nfc_collapse_whitespace_strip_v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_for_comparison(text: str) -> str:
    """Preserve word boundaries and original text; normalize only comparison keys."""
    return WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", text)).strip()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class Record:
    split: str
    sample_id: str
    record_number: int
    physical_line_end: int
    text: str
    labels: tuple[int | None, ...]

    def as_dict(self) -> dict:
        return {
            "split": self.split,
            "id": self.sample_id,
            "record_number": self.record_number,
            "physical_line_end": self.physical_line_end,
            "text": self.text,
            "labels": dict(zip(ASPECT_COLUMNS, self.labels)),
        }


class IssueCollector:
    def __init__(self, stage: str):
        self.stage = stage
        self.items: list[dict] = []
        self.sources: dict[str, str | None] = {}

    def add(self, rule_id: str, severity: str, message: str, *, records=(), **details):
        source_splits = {record.split for record in records}
        if "split" in details:
            source_splits.add(details["split"])
        item = {
            "schema_version": SCHEMA_VERSION,
            "rules_version": RULES_VERSION,
            "rule_id": rule_id,
            "severity": severity,
            "message": message,
            "stage": self.stage,
            "source_sha256": {split: self.sources.get(split) for split in sorted(source_splits)},
            "records": [record.as_dict() for record in records],
            **details,
        }
        identity = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        item["issue_id"] = "issue_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        item["review"] = {
            "status": "pending" if severity == "REVIEW" else "not_required",
            "conclusion": None,
            "reason": None,
            "action": None,
            "affected_records": None,
            "reviewed_at": None,
        }
        self.items.append(item)


class UnusableHeader(Exception):
    """The header cannot be interpreted safely; retain an incomplete scan."""


def _label_conflicts(records: list[Record]) -> dict:
    conflicts = {}
    for index, aspect in enumerate(ASPECT_COLUMNS):
        values = {record.labels[index] for record in records if record.labels[index] is not None}
        if len(values) > 1:
            conflicts[aspect] = [
                {"split": record.split, "id": record.sample_id,
                 "record_number": record.record_number, "value": record.labels[index]}
                for record in records
            ]
    return conflicts


def _length_summary(lengths: list[int]) -> dict:
    if not lengths:
        return {"count": 0, "min": None, "max": None, "mean": None,
                "p50": None, "p95": None, "p99": None}
    ordered = sorted(lengths)
    result = {"count": len(ordered), "min": ordered[0], "max": ordered[-1],
              "mean": round(sum(ordered) / len(ordered), 3)}
    for name, fraction in (("p50", .5), ("p95", .95), ("p99", .99)):
        result[name] = ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]
    return result


def _scan_file(split: str, path: Path, allowed_labels: tuple[int, ...],
               collector: IssueCollector, max_text_chars: int | None,
               min_class_fraction: float | None) -> tuple[dict, list[Record]]:
    info = {
        "path": str(path), "sha256": None, "sha256_after_scan": None,
        "size_bytes": None, "scan_complete": False, "records_scanned": 0,
        "records_with_valid_width": 0, "records_indexed": 0, "header": None,
        "observed_zero_based_record_ids": None,
    }
    records: list[Record] = []
    label_counts = {aspect: Counter() for aspect in ASPECT_COLUMNS}
    lengths: list[int] = []
    empty_texts = multiline_texts = outer_quoted_texts = 0
    sequential_ids = True
    reader = None
    try:
        info["sha256"] = file_sha256(path)
        collector.sources[split] = info["sha256"]
        info["size_bytes"] = path.stat().st_size
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            info["header"] = header
            if header is None:
                collector.add("empty_file", "ERROR", "文件为空，缺少表头。", split=split)
                raise UnusableHeader

            counts = Counter(header)
            missing = [name for name in EXPECTED_COLUMNS if name not in counts]
            extra = [name for name in header if name not in EXPECTED_COLUMNS]
            duplicate = sorted(name for name, count in counts.items() if count > 1)
            if missing:
                collector.add("missing_columns", "ERROR", "缺少必需列。", split=split, columns=missing)
            if extra:
                collector.add("unexpected_columns", "ERROR", "存在规范之外的列。", split=split, columns=extra)
            if duplicate:
                collector.add("duplicate_columns", "ERROR", "表头列名重复。", split=split, columns=duplicate)
            if not missing and not extra and not duplicate and header != list(EXPECTED_COLUMNS):
                collector.add("column_order", "ERROR", "列顺序与固定规范不一致。", split=split,
                              expected_columns=list(EXPECTED_COLUMNS), actual_columns=header)
            if missing or duplicate:
                # No unambiguous field interpretation is possible; do not claim a complete scan.
                raise UnusableHeader
            positions = {name: header.index(name) for name in EXPECTED_COLUMNS}

            for record_number, row in enumerate(reader, start=1):
                info["records_scanned"] = record_number
                if len(row) != len(header):
                    sequential_ids = False
                    collector.add("record_width", "ERROR", "CSV 记录的列数与表头不一致。",
                                  split=split, record_number=record_number,
                                  physical_line_end=reader.line_num,
                                  expected_width=len(header), actual_width=len(row))
                    continue
                info["records_with_valid_width"] += 1
                sample_id = row[positions["id"]]
                text = row[positions["content"]]
                sequential_ids = sequential_ids and sample_id == str(record_number - 1)
                labels: list[int | None] = []
                bad_labels = []
                for aspect in ASPECT_COLUMNS:
                    value = row[positions[aspect]]
                    try:
                        parsed = parse_label(value, allowed_labels)
                    except InvalidLabelError as error:
                        bad_labels.append({"aspect": aspect, "raw_value": value, "reason": error.reason})
                        parsed = None
                    else:
                        label_counts[aspect][parsed] += 1
                    labels.append(parsed)
                record = Record(split, sample_id, record_number, reader.line_num, text, tuple(labels))
                records.append(record)
                if bad_labels:
                    collector.add("invalid_labels", "ERROR", "存在缺失、非整数或越界标签。",
                                  records=[record], invalid_labels=bad_labels,
                                  allowed_labels=list(allowed_labels))
                if not sample_id.strip():
                    collector.add("empty_id", "ERROR", "样本 ID 为空或仅包含空白。", records=[record])
                if not text.strip():
                    empty_texts += 1
                    collector.add("empty_text", "ERROR", "文本为空或仅包含空白。", records=[record])
                lengths.append(len(text))
                multiline_texts += "\n" in text or "\r" in text
                outer_quoted_texts += len(text) >= 2 and text.startswith('"') and text.endswith('"')
                suspicious = Counter(
                    char for char in text if char == "\ufffd"
                    or (unicodedata.category(char) == "Cc" and char not in "\t\n\r")
                )
                if suspicious:
                    collector.add("suspicious_characters", "WARN", "文本含替换字符或异常控制字符，需按内容判断。",
                                  records=[record], characters={f"U+{ord(c):04X}": n for c, n in sorted(suspicious.items())})
                if max_text_chars is not None and len(text) > max_text_chars:
                    collector.add("text_length", "WARN", "文本字符数超过显式配置的阈值。",
                                  records=[record], actual_length=len(text), threshold=max_text_chars)
            info["scan_complete"] = True
    except UnusableHeader:
        pass
    except (OSError, UnicodeError, csv.Error) as error:
        collector.add("file_read_failure", "ERROR", "文件读取、UTF-8 解码或 CSV 解析失败，扫描不完整。",
                      split=split, error_type=type(error).__name__, detail=str(error),
                      records_scanned=info["records_scanned"],
                      physical_line_end=reader.line_num if reader is not None else None)

    # A changed source invalidates statistics even when CSV parsing reached EOF.
    if info["sha256"] is not None:
        try:
            info["sha256_after_scan"] = file_sha256(path)
            if info["sha256_after_scan"] != info["sha256"]:
                info["scan_complete"] = False
                collector.add("source_changed", "ERROR", "文件在校验期间发生变化，本次结果不可作为完整校验。", split=split)
        except OSError as error:
            info["scan_complete"] = False
            collector.add("source_recheck_failure", "ERROR", "无法复核源文件摘要。", split=split, detail=str(error))

    info["records_indexed"] = len(records)
    info["observed_zero_based_record_ids"] = sequential_ids if records and info["scan_complete"] else None
    info["text_statistics"] = {
        "character_length": _length_summary(lengths),
        "empty_texts": empty_texts, "multiline_texts": multiline_texts,
        "outer_quoted_texts": outer_quoted_texts,
        "percentile_method": "nearest_rank",
        "denominator": "records_with_valid_width",
    }
    info["label_distribution"] = {}
    for aspect, counter in label_counts.items():
        valid_count = sum(counter.values())
        info["label_distribution"][aspect] = {
            "valid_label_count": valid_count,
            "invalid_label_count": info["records_with_valid_width"] - valid_count,
            "counts": {str(value): counter[value] for value in allowed_labels},
            "fractions": {str(value): counter[value] / valid_count if valid_count else None
                          for value in allowed_labels},
            "fraction_denominator": "valid_label_count",
            "partial": not info["scan_complete"],
        }
        if valid_count and info["scan_complete"]:
            missing_classes = [value for value in allowed_labels if counter[value] == 0]
            if missing_classes:
                collector.add("missing_label_classes", "WARN", "该方面缺少某些类别，未进行自动补样或重采样。",
                              split=split, aspect=aspect, missing_classes=missing_classes,
                              valid_label_count=valid_count)
            if min_class_fraction is not None:
                low = {str(value): counter[value] / valid_count for value in allowed_labels
                       if 0 < counter[value] / valid_count < min_class_fraction}
                if low:
                    collector.add("class_fraction", "WARN", "部分类别占比低于显式配置的阈值。",
                                  split=split, aspect=aspect, fractions=low,
                                  threshold=min_class_fraction, valid_label_count=valid_count)
    if info["scan_complete"] and info["records_scanned"] == 0:
        collector.add("no_records", "ERROR", "文件只有表头，没有样本。", split=split)
    return info, records


def _check_groups(records: list[Record], collector: IssueCollector, id_scope: str) -> dict:
    by_id = defaultdict(list)
    exact = defaultdict(list)
    normalized = defaultdict(list)
    for record in records:
        if record.sample_id.strip():
            by_id[(record.split, record.sample_id)].append(record)
        if record.text.strip():
            exact[hashlib.sha256(record.text.encode("utf-8")).digest()].append(record)
            key = normalize_for_comparison(record.text)
            normalized[hashlib.sha256(key.encode("utf-8")).digest()].append(record)
    for group in by_id.values():
        if len(group) > 1:
            collector.add("duplicate_id", "ERROR", "集合内样本 ID 重复。", records=group)
    if id_scope == "global":
        global_ids = defaultdict(list)
        for record in records:
            if record.sample_id.strip():
                global_ids[record.sample_id].append(record)
        for group in global_ids.values():
            if len({record.split for record in group}) > 1:
                collector.add("cross_split_global_id", "REVIEW", "全局 ID 在不同集合中重复，需核实样本身份。", records=group)

    summary = {"exact_duplicate_groups": 0, "normalized_only_duplicate_groups": 0,
               "cross_split_exact_groups": 0, "cross_split_normalized_only_groups": 0,
               "exact_groups_with_label_conflicts": 0,
               "normalized_only_groups_with_label_conflicts": 0}
    affected_records = set()
    cross_split_records = set()
    for comparison, groups in (("exact", exact), ("normalized", normalized)):
        for group in groups.values():
            if len(group) < 2:
                continue
            if comparison == "normalized" and len({record.text for record in group}) == 1:
                # This group was already reported in the exact pass.
                continue
            cross_split = len({record.split for record in group}) > 1
            conflicts = _label_conflicts(group)
            summary["exact_duplicate_groups" if comparison == "exact" else "normalized_only_duplicate_groups"] += 1
            if cross_split:
                summary["cross_split_exact_groups" if comparison == "exact" else "cross_split_normalized_only_groups"] += 1
            summary["exact_groups_with_label_conflicts" if comparison == "exact"
                    else "normalized_only_groups_with_label_conflicts"] += bool(conflicts)
            identities = {(record.split, record.record_number) for record in group}
            affected_records.update(identities)
            if cross_split:
                cross_split_records.update(identities)
            severity = "REVIEW" if cross_split or conflicts or comparison == "normalized" else "WARN"
            rule_id = f"cross_split_{comparison}_overlap" if cross_split else f"within_split_{comparison}_duplicate"
            collector.add(rule_id, severity,
                          "存在跨集合文本重叠或疑似重叠。" if cross_split else "集合内存在重复或疑似重复文本。",
                          records=group, comparison=comparison,
                          normalization_rule=NORMALIZATION_RULE if comparison == "normalized" else None,
                          label_conflicts=conflicts, record_count=len(group),
                          extra_records=len(group) - 1,
                          note="归一化仅用于比较；保留原文，不自动删除、改标签或判定为同一原始评论。")
    summary["unique_records_in_duplicate_groups"] = len(affected_records)
    summary["unique_records_in_cross_split_groups"] = len(cross_split_records)
    return summary


def _write_json_atomic(path: Path, value, *, jsonl: bool = False):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="\n", suffix=".tmp",
                                         prefix=path.stem + "_", dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            if jsonl:
                for item in value:
                    stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            else:
                json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
                stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_dataset(split_paths: dict[str, Path], output_dir: Path, *, stage: str = "raw",
                     id_scope: str = "split", max_text_chars: int | None = None,
                     min_class_fraction: float | None = None) -> dict:
    """Validate input CSVs without editing them; return the persisted JSON report.

    `mapped` checks processed CSV structure and labels, not source-to-output lineage.
    No review decisions are automatically applied or overwritten by this function.
    """
    if not split_paths or any(not isinstance(name, str) or not name for name in split_paths):
        raise ValueError("至少提供一个非空集合名及其 CSV 路径。")
    if stage not in {"raw", "mapped"}:
        raise ValueError("stage 必须是 raw 或 mapped。")
    if id_scope not in {"split", "global"}:
        raise ValueError("id_scope 必须是 split 或 global。")
    if max_text_chars is not None and (isinstance(max_text_chars, bool)
                                      or not isinstance(max_text_chars, int) or max_text_chars <= 0):
        raise ValueError("max_text_chars 必须是正整数。")
    if min_class_fraction is not None and (not math.isfinite(min_class_fraction)
                                          or not 0 < min_class_fraction < 1):
        raise ValueError("min_class_fraction 必须位于 0 与 1 之间。")
    paths = {name: Path(path).resolve() for name, path in sorted(split_paths.items())}
    output_dir = Path(output_dir).resolve()
    if len(set(paths.values())) != len(paths):
        raise ValueError("不同集合不能指向同一个输入文件。")
    for path in paths.values():
        if output_dir.is_relative_to(path.parent):
            raise ValueError("报告目录不能位于输入数据所在目录及其子目录中。")
        for name in ("validation_report.json", "validation_issues.jsonl"):
            target = output_dir / name
            if target.is_symlink():
                raise ValueError("报告目标不能是符号链接。")

    started_at = datetime.now(timezone.utc).isoformat()
    allowed = tuple(LABEL_MAPPING) if stage == "raw" else tuple(LABEL_MAPPING.values())
    collector = IssueCollector(stage)
    files = {}
    records = []
    previous_limit = csv.field_size_limit()
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    try:
        for split, path in paths.items():
            files[split], samples = _scan_file(split, path, allowed, collector,
                                               max_text_chars, min_class_fraction)
            records.extend(samples)
    finally:
        csv.field_size_limit(previous_limit)
    duplicate_summary = _check_groups(records, collector, id_scope)
    issue_counts = {severity: sum(item["severity"] == severity for item in collector.items)
                    for severity in ("ERROR", "REVIEW", "WARN")}
    if issue_counts["ERROR"]:
        status = "failed"
    elif issue_counts["REVIEW"]:
        status = "needs_review"
    elif issue_counts["WARN"]:
        status = "passed_with_warnings"
    else:
        status = "passed"
    rules = Counter(item["rule_id"] for item in collector.items)
    report = {
        "schema_version": SCHEMA_VERSION,
        "rules_version": RULES_VERSION,
        "started_at": started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "scope": "csv_structure_labels_text_and_split_independence",
        "status": status,
        "scan_complete": all(info["scan_complete"] for info in files.values()),
        "validation_gate_passed": status in {"passed", "passed_with_warnings"},
        "issue_counts": issue_counts,
        "issue_count_unit": "issue_groups_not_affected_records",
        "rule_counts": dict(sorted(rules.items())),
        "pending_reviews": issue_counts["REVIEW"],
        "files": files,
        "total_records_scanned": sum(info["records_scanned"] for info in files.values()),
        "duplicate_summary": duplicate_summary,
        "configuration": {
            "expected_columns": list(EXPECTED_COLUMNS),
            "allowed_labels": list(allowed),
            "label_mapping": {str(key): value for key, value in LABEL_MAPPING.items()},
            "raw_label_meanings": {str(key): value for key, value in RAW_LABEL_MEANINGS.items()},
            "id_scope": id_scope,
            "id_scope_basis": "configuration; per-file numbering is observed, global source identity is not verified",
            "encoding": "utf-8-sig", "csv_strict": True,
            "record_number_definition": "1-based data record ordinal, excluding header; not a physical line number",
            "normalization_rule": NORMALIZATION_RULE,
            "normalization_steps": ["Unicode NFC", "collapse consecutive Unicode whitespace to one ASCII space", "strip boundary whitespace"],
            "normalization_is_comparison_only": True,
            "max_text_chars": max_text_chars,
            "min_class_fraction": min_class_fraction,
            "missing_classes_are_warnings": True,
        },
        "review_policy": {
            "all_review_items_initially_pending": True,
            "required_decision_fields": ["issue_id", "source_sha256", "status", "conclusion", "reason", "action", "affected_records", "reviewed_at"],
            "decisions_file": str(output_dir / "review_decisions.jsonl"),
            "decisions_file_written_or_applied": False,
            "closure_condition": "record a reasoned decision and revalidate any processed dataset; merely viewed is not resolved",
        },
        "limitations": [
            "Text equality and normalization matches do not prove source identity; review original records.",
            "No semantic relabeling, cleaning, deduplication, label mapping, or review decision is performed.",
            "Mapped mode checks the processed CSV itself; mapping correctness and sample-count reconciliation require source/processing provenance and are not asserted here.",
        ],
        "artifacts": {
            "issues": str(output_dir / "validation_issues.jsonl"),
            "report": str(output_dir / "validation_report.json"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output_dir / "validation_issues.jsonl", collector.items, jsonl=True)
    report["artifacts"]["issues_sha256"] = file_sha256(output_dir / "validation_issues.jsonl")
    _write_json_atomic(output_dir / "validation_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    # Keep redirected CLI output readable on Windows, independently of the code page.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="只读校验细粒度情感数据集，输出 JSON 报告和 JSONL 异常明细。")
    parser.add_argument("--raw-dir", "--input-dir", type=Path, default=PROJECT_ROOT / "data" / "raw",
                        help="输入 CSV 目录（默认项目 data/raw；mapped 模式也可传处理后目录）。")
    parser.add_argument("--train-csv", type=Path, help="覆盖训练集文件路径。")
    parser.add_argument("--validation-csv", type=Path, help="覆盖验证集文件路径。")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="报告目录（默认 data/reports/<stage>）；不得位于输入目录内部。")
    parser.add_argument("--stage", choices=("raw", "mapped"), default="raw")
    parser.add_argument("--id-scope", choices=("split", "global"), default="split")
    parser.add_argument("--max-text-chars", type=int, default=None)
    parser.add_argument("--min-class-fraction", type=float, default=None)
    args = parser.parse_args(argv)
    paths = {
        "train": args.train_csv if args.train_csv is not None else args.raw_dir / "sentiment_analysis_trainingset.csv",
        "validation": args.validation_csv if args.validation_csv is not None else args.raw_dir / "sentiment_analysis_validationset.csv",
    }
    try:
        output_dir = args.output_dir if args.output_dir is not None else PROJECT_ROOT / "data" / "reports" / args.stage
        report = validate_dataset(paths, output_dir, stage=args.stage, id_scope=args.id_scope,
                                  max_text_chars=args.max_text_chars, min_class_fraction=args.min_class_fraction)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"status": report["status"], "scan_complete": report["scan_complete"],
                      "records_scanned": report["total_records_scanned"],
                      "issue_counts": report["issue_counts"], "report": report["artifacts"]["report"],
                      "issues": report["artifacts"]["issues"]}, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "failed" else 2 if report["status"] == "needs_review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
