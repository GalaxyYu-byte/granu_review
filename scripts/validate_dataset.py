"""只读校验含二十个方面标签的情感分类数据集，输出报告和问题明细。

校验使用 Python 标准库，不修改输入文本、标签或既有复核决定。结构错误、
重复问题和统计告警分别记录，扫描是否完成与是否通过校验也分别表达。
命令行参数可通过 --help 查看。
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
    from .dataset_io import receipt_path
except ImportError:
    from dataset_io import receipt_path


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
    """仅生成比较用的归一化键，不改变原文，连续空白仍保留一个分词边界。"""
    return WHITESPACE_PATTERN.sub(" ", unicodedata.normalize("NFC", text)).strip()


def file_sha256(path: Path) -> str:
    """分块计算文件摘要，避免将整份数据一次读入内存。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class Record:
    """保存一条已解析记录及其来源位置，供重复检查和原始证据导出使用。

    记录序号按解析后的数据记录计数；物理结束行号仅用于排查多行文本，
    两者不能互换。无法解析的标签以空值保留，不猜测或修正原始标签。
    """
    split: str
    sample_id: str
    record_number: int
    physical_line_end: int
    text: str
    labels: tuple[int | None, ...]

    def as_dict(self) -> dict:
        """按固定方面顺序导出标签，保留原文和完整的记录定位信息。"""
        return {
            "split": self.split,
            "id": self.sample_id,
            "record_number": self.record_number,
            "physical_line_end": self.physical_line_end,
            "text": self.text,
            "labels": dict(zip(ASPECT_COLUMNS, self.labels)),
        }


class IssueCollector:
    """统一保存发现，并将每项发现绑定到规则版本和对应输入摘要。"""
    def __init__(self, stage: str):
        self.stage = stage
        self.items: list[dict] = []
        self.sources: dict[str, str | None] = {}

    def add(self, rule_id: str, severity: str, message: str, *, records=(), **details):
        """根据问题内容生成稳定编号，初始复核状态不代表问题已获批准。"""
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
        # 编号只依赖规则、来源和证据，不包含运行时间，便于重复扫描时关联旧决定。
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
    """表头无法可靠解释时中止该文件扫描，并保留扫描未完成的状态。"""


def _label_conflicts(records: list[Record]) -> dict:
    """逐方面列出存在不同有效标签的组成员，不将缺失标签当作新类别。"""
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
    """汇总原始字符长度，分位数采用最近秩法，不在此设置长度阈值。"""
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
    """扫描单个文件，保留可定位的问题证据，并标明所有统计是否完整。

    列宽异常只跳过该条记录，后续记录仍继续扫描；表头歧义、解码或解析
    失败则不能声称扫描完成。读取前后的文件摘要用于发现并发修改。
    """
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
                # 缺列或重名列会造成字段解释歧义，不能继续按固定列含义统计。
                raise UnusableHeader
            positions = {name: header.index(name) for name in EXPECTED_COLUMNS}

            for record_number, row in enumerate(reader, start=1):
                # 使用 CSV 解析器的记录边界，多行评论仍是一条样本，不能按物理行拆分。
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
                # 正常制表符和换行允许保留；异常控制符只发告警，由后续复核决定处理。
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

    # 即使解析器读到了文件末尾，读取期间源文件发生变化也会使本次统计失效。
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
    # 每个方面的比例只以该方面的有效标签为分母；不完整扫描不触发分布阈值判断。
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
                # 零占比已经由缺失类别告警覆盖，这里只报告非零但低于显式阈值的类别。
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
    """分别检查编号、完全重复和归一化重复，保留整组成员及逐方面冲突。

    集合间编号能否复用由调用方的编号作用域决定。文本相等仅证明比较键
    相同，不据此自动删除记录，也不将训练标签作为验证标签的替代真值。
    """
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
                # 整组原文完全相同时，完全重复检查已经报告，不再重复生成归一化问题。
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
    # 完全重复子组可能包含在更大的归一化组内，因此受影响记录数需要按来源去重。
    summary["unique_records_in_duplicate_groups"] = len(affected_records)
    summary["unique_records_in_cross_split_groups"] = len(cross_split_records)
    return summary


def _write_json_atomic(path: Path, value, *, jsonl: bool = False):
    """先在目标目录写临时文件，再原子替换，避免读者看到半份报告。

    每个 JSONL 对象只追加一个真正的换行；字符串内部字符由 JSON 编码器
    处理。调用方负责判断目标是否允许覆盖，本函数仅保证单文件发布完整。
    """
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
                     min_class_fraction: float | None = None, root: Path | None = None) -> dict:
    """只读检查输入 CSV，落盘问题明细及汇总报告，并返回同一份报告内容。

    原始阶段和映射阶段使用不同合法标签域。映射阶段只验证文件自身，
    不能证明标签映射及样本增减与来源一致；这些需要独立来源对账。
    本函数不应用或覆盖复核决定，报告中的待复核问题仍需明确处理。
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
    # 写报告之前限制输出位置，防止报告混入输入目录或经符号链接覆盖其他文件。
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
    # 长评论可能超过 CSV 库默认限制；兼容平台整数范围，并在扫描后恢复进程原设置。
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
    # 严重程度决定校验状态；是否完整扫描另外保留，不能以读完文件等同于通过。
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
    # 先发布明细并计算实际文件摘要，再把该摘要写入汇总，建立后续复核的来源绑定。
    _write_json_atomic(output_dir / "validation_issues.jsonl", collector.items, jsonl=True)
    report["artifacts"]["issues_sha256"] = file_sha256(output_dir / "validation_issues.jsonl")
    # 项目内路径按根目录保存为相对路径；外部输入保留绝对位置，便于迁移后明确定位。
    for info in report["files"].values():
        info["path"] = receipt_path(Path(info["path"]), root)
    for field in ("issues", "report"):
        report["artifacts"][field] = receipt_path(Path(report["artifacts"][field]), root)
    report["review_policy"]["decisions_file"] = receipt_path(output_dir / "review_decisions.jsonl", root)
    _write_json_atomic(output_dir / "validation_report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """解析命令行参数；结构失败、待复核和通过分别使用不同退出码。"""
    # 显式使用 UTF-8，避免 Windows 控制台代码页影响重定向后的中文报告。
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="只读校验细粒度情感数据集，输出 JSON 报告和 JSONL 异常明细。")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="项目根目录，凭据保存该目录内的相对路径。")
    parser.add_argument("--raw-dir", "--input-dir", type=Path, default=None,
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
    input_dir = args.raw_dir or args.root / "data/raw"
    paths = {
        "train": args.train_csv if args.train_csv is not None else input_dir / "sentiment_analysis_trainingset.csv",
        "validation": args.validation_csv if args.validation_csv is not None else input_dir / "sentiment_analysis_validationset.csv",
    }
    try:
        output_dir = args.output_dir if args.output_dir is not None else args.root / "data" / "reports" / args.stage
        report = validate_dataset(paths, output_dir, stage=args.stage, id_scope=args.id_scope,
                                  max_text_chars=args.max_text_chars, min_class_fraction=args.min_class_fraction, root=args.root)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"status": report["status"], "scan_complete": report["scan_complete"],
                      "records_scanned": report["total_records_scanned"],
                      "issue_counts": report["issue_counts"], "report": report["artifacts"]["report"],
                      "issues": report["artifacts"]["issues"]}, ensure_ascii=False, indent=2))
    return 1 if report["status"] == "failed" else 2 if report["status"] == "needs_review" else 0


if __name__ == "__main__":
    raise SystemExit(main())
