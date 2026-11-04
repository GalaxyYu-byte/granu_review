"""严格读取清洗后、尚未映射标签的评论数据，供训练与评估共用。

单文件入口只保证 CSV 和原始标签可用；正式训练通过处理凭据选择批次，
核对二次校验、路径、摘要和条数后才返回完整样本。读取不改变文本、ID、
顺序或磁盘标签，也不负责分词、张量转换以及完整来源审计。
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import sys
from threading import RLock

try:
    from .dataset_io import PROJECT_ROOT, file_sha256, receipt_origin, resolve_receipt_path
    from .label_config import CLASS_VALUES, EXPECTED_COLUMNS, NUM_ASPECTS, NUM_CLASSES, encode_labels, get_label_config
except ImportError:
    from dataset_io import PROJECT_ROOT, file_sha256, receipt_origin, resolve_receipt_path
    from label_config import CLASS_VALUES, EXPECTED_COLUMNS, NUM_ASPECTS, NUM_CLASSES, encode_labels, get_label_config


SPLITS = ("train", "validation")
# CSV 字段上限属于进程全局设置；串行管理本模块的读取，避免两个读取器相互恢复上限。
_CSV_FIELD_LOCK = RLock()


class DatasetReadError(ValueError):
    """CSV 不可用、标签阶段错误或所选批次凭据不一致。"""


class _HashingReader(io.RawIOBase):
    """计算实际送给 CSV 解析器的字节摘要，避免只核对路径的前后版本。"""

    def __init__(self, source):
        self.source = source
        self.digest = hashlib.sha256()

    def readable(self):
        """供 BufferedReader 按块读取，不需要随机定位。"""
        return True

    def readinto(self, buffer):
        """在文本解码之前记录原始字节，BOM 和行尾也参与摘要。"""
        count = self.source.readinto(buffer)
        if count:
            self.digest.update(memoryview(buffer)[:count])
        return count


@dataclass(frozen=True, slots=True)
class ReviewSample:
    """不可变样本；位置均属于当前清洗后 CSV，不能据此推算原始行号。"""

    split: str
    sample_id: str
    text: str
    labels: tuple[int, ...]
    processed_record_number: int
    physical_line_end: int


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """完整加载的两个集合、独立标签配置及本次实际读取文件的路径和摘要。"""

    train: list[ReviewSample]
    validation: list[ReviewSample]
    label_config: dict[str, list]
    sources: dict[str, dict[str, str]]


@contextmanager
def _large_csv_fields():
    """沿用校验器的长字段策略，成功、失败时都恢复调用前的全局上限。"""
    with _CSV_FIELD_LOCK:
        previous_limit = csv.field_size_limit()
        try:
            field_limit = sys.maxsize
            while True:
                try:
                    csv.field_size_limit(field_limit)
                    break
                except OverflowError:
                    field_limit //= 10
                    if not field_limit:
                        raise DatasetReadError("当前平台无法设置 CSV 字段上限。")
            yield
        finally:
            csv.field_size_limit(previous_limit)


def _fingerprint(path: Path) -> str:
    """统一为文件访问错误附加明确路径。"""
    try:
        return file_sha256(path)
    except OSError as error:
        raise DatasetReadError(f"无法读取文件 {path}：{error}") from error


def read_split(path: str | Path, *, split: str, expected_sha256: str | None = None) -> list[ReviewSample]:
    """完整读取一份原始标签域 CSV，仅在全部样本合法且文件未变化时返回。

    不推断标签是否已经编码；原始值 0/1 与模型编号重叠，正式训练应调用
    load_datasets()，由批次凭据确认标签阶段并传入 expected_sha256。不同集合
    可以具有相同 ID；可选摘要用于绑定实际读取内容，不代表标签阶段证明。
    """
    path = Path(path).resolve()
    if split not in SPLITS:
        raise DatasetReadError(f"{path}：split 必须为 train 或 validation，收到 {split!r}。")
    before = _fingerprint(path)
    if expected_sha256 is not None:
        _sha256(expected_sha256, f"{path}.expected_sha256")
        _require(before == expected_sha256, f"{path} [split={split}]：CSV SHA-256 摘要不匹配。")
    samples: list[ReviewSample] = []
    seen: dict[str, int] = {}
    number = 0
    sample_id = None
    reader = None

    def location() -> str:
        """错误位置按 CSV 记录计数；多行评论的物理行号另列。"""
        return (f"{path} [split={split}, processed_record_number={number}, "
                f"physical_line_end={reader.line_num if reader is not None else 0}, id={sample_id!r}]")

    try:
        with _large_csv_fields(), path.open("rb", buffering=0) as source:
            hashed = _HashingReader(source)
            with io.TextIOWrapper(io.BufferedReader(hashed), encoding="utf-8-sig", newline="") as stream:
                reader = csv.reader(stream, strict=True)
                header = next(reader, None)
                if header != list(EXPECTED_COLUMNS):
                    raise DatasetReadError(f"{location()}：表头缺失，或列名、顺序与 EXPECTED_COLUMNS 不一致。")
                while True:
                    number = len(samples) + 1
                    sample_id = None
                    row = next(reader, None)
                    if row is None:
                        break
                    sample_id = row[0] if row else None
                    if len(row) != len(header):
                        raise DatasetReadError(f"{location()}：列宽错误，期望 {len(header)} 列，实际 {len(row)} 列。")
                    text = row[1]
                    if not sample_id.strip():
                        raise DatasetReadError(f"{location()}：样本 ID 为空或仅包含空白。")
                    if sample_id in seen:
                        raise DatasetReadError(f"{location()}：集合内重复 ID，首次出现在记录 {seen[sample_id]}。")
                    if not text.strip():
                        raise DatasetReadError(f"{location()}：文本为空或仅包含空白。")
                    try:
                        labels = tuple(encode_labels(dict(zip(header, row))))
                    except ValueError as error:
                        raise DatasetReadError(f"{location()}：{error}") from error
                    samples.append(ReviewSample(split, sample_id, text, labels, number, reader.line_num))
                    seen[sample_id] = number
    except (OSError, UnicodeError, csv.Error) as error:
        raise DatasetReadError(f"{location()}：CSV 读取失败：{error}") from error
    if not samples:
        raise DatasetReadError(f"{location()}：数据集为空，表头之后没有数据记录。")
    if hashed.digest.hexdigest() != before or _fingerprint(path) != before:
        raise DatasetReadError(f"{path} [split={split}]：文件在读取期间发生变化。")
    return samples


def _require(condition: bool, message: str) -> None:
    """凭据不符时立即停止，不尝试寻找其他批次。"""
    if not condition:
        raise DatasetReadError(message)


def _object(value: object, context: str) -> dict:
    """报告的对象字段不允许以空值或列表替代。"""
    _require(isinstance(value, dict), f"{context} 必须为 JSON 对象。")
    return value


def _positive_count(value: object, context: str) -> int:
    """拒绝布尔值和浮点数被隐式解释为条数。"""
    _require(type(value) is int and value > 0, f"{context} 必须为正整数。")
    return value


def _receipt_string(value: object, context: str) -> str:
    """不为缺失路径或摘要提供默认值。"""
    _require(isinstance(value, str) and bool(value.strip()), f"{context} 必须为非空字符串。")
    return value


def _sha256(value: object, context: str) -> str:
    """使用与现有产物相同的小写十六进制 SHA-256 格式。"""
    _require(isinstance(value, str) and len(value) == 64
             and all(char in "0123456789abcdef" for char in value), f"{context} 不是有效 SHA-256。")
    return value


def _json_snapshot(path: Path, expected: str | None = None) -> tuple[dict, str]:
    """摘要直接对应实际解析的报告字节，结束时再检查磁盘版本未变化。"""
    try:
        data = path.read_bytes()
        fingerprint = hashlib.sha256(data).hexdigest()
        _require(expected is None or fingerprint == expected, f"{path}：报告 SHA-256 摘要不匹配。")
        return _object(json.loads(data), str(path)), fingerprint
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetReadError(f"无法解析报告 {path}：{error}") from error


def _issue_counts(value: object, context: str) -> dict:
    """计数描述扫描发现；已复核 REVIEW 不据此再次拦截。"""
    counts = _object(value, context)
    for severity in ("ERROR", "REVIEW", "WARN"):
        count = counts.get(severity)
        _require(type(count) is int and count >= 0, f"{context}.{severity} 必须为非负整数。")
    _require(counts["ERROR"] == 0, f"{context} 仍含 ERROR。")
    return counts


def load_datasets(processing_report: str | Path, *, root: str | Path | None = None) -> DatasetBundle:
    """加载处理凭据指定的批次，核对两份 CSV 和二次校验后返回完整结果。

    兼容历史 1.0 与当前 1.1 凭据及可迁移路径；只核对训练输入绑定，完整的
    原始问题、决定和应用日志审计仍由 audit_processed_dataset 独立执行。
    """
    root = Path(root or PROJECT_ROOT).resolve()
    processing_path = resolve_receipt_path(str(processing_report), root=root)
    processing, processing_sha = _json_snapshot(processing_path)
    context = str(processing_path)
    _require(processing.get("schema_version") in ("1.0", "1.1"), f"{context}：不支持的处理报告版本。")
    _require(processing.get("stage") == "cleaned_original_label_space"
             and processing.get("mapping_applied") is False, f"{context}：必须为清洗后、尚未映射的原始标签域。")
    outputs = _object(processing.get("outputs"), f"{context}.outputs")
    validation_artifact = _object(outputs.get("validation_report"), f"{context}.outputs.validation_report")
    recorded_validation = _receipt_string(validation_artifact.get("path"), f"{context}.outputs.validation_report.path")
    # 与独立审计入口相同：旧凭据按报告的完整项目内后缀绑定旧根目录，不能猜测 data 目录。
    origin = receipt_origin(recorded_validation, processing_path.parent / PureWindowsPath(recorded_validation).name, root)
    paths = {"processing_report": processing_path}
    fingerprints = {"processing_report": processing_sha}
    for role in (*SPLITS, "validation_report"):
        artifact = _object(outputs.get(role), f"{context}.outputs.{role}")
        recorded = _receipt_string(artifact.get("path"), f"{context}.outputs.{role}.path")
        paths[role] = resolve_receipt_path(recorded, root=root, legacy_root=origin)
        fingerprints[role] = _sha256(artifact.get("sha256"), f"{context}.outputs.{role}.sha256")
    _require(len(set(paths.values())) == len(paths), f"{context}：不同角色绑定了相同文件路径。")
    validation_path = paths["validation_report"]
    validation, _ = _json_snapshot(validation_path, fingerprints["validation_report"])
    validation_context = str(validation_path)
    _require(validation.get("stage") == "raw", f"{validation_context}：二次校验必须使用原始标签域 stage=raw。")
    _require(validation.get("scan_complete") is True and validation.get("validation_gate_passed") is True
             and validation.get("status") in ("passed", "passed_with_warnings"),
             f"{validation_context}：二次校验扫描不完整或最终状态未通过。")
    _require(validation["status"] == processing.get("post_validation_status"), f"{context}：二次校验最终状态与处理凭据不一致。")
    counts = _issue_counts(validation.get("issue_counts"), f"{validation_context}.issue_counts")
    if "post_validation_issue_counts" in processing:
        processing_counts = _issue_counts(processing["post_validation_issue_counts"], f"{context}.post_validation_issue_counts")
        _require(counts == processing_counts, f"{context}：二次校验问题计数与处理凭据不一致。")
    if "pending_reviews" in validation:
        pending = validation["pending_reviews"]
        _require(type(pending) is int and pending == 0, f"{validation_context}：仍有未完成的复核。")
    configuration = _object(validation.get("configuration"), f"{validation_context}.configuration")
    allowed = configuration.get("allowed_labels")
    _require(isinstance(allowed, list) and all(type(value) is int for value in allowed)
             and allowed == list(CLASS_VALUES), f"{validation_context}：合法标签配置与原始标签域不一致。")
    _require(configuration.get("expected_columns") == list(EXPECTED_COLUMNS), f"{validation_context}：列配置与当前规范不一致。")
    artifacts = _object(validation.get("artifacts"), f"{validation_context}.artifacts")
    recorded_self = _receipt_string(artifacts.get("report"), f"{validation_context}.artifacts.report")
    post_origin = receipt_origin(recorded_self, validation_path, root)
    _require(resolve_receipt_path(recorded_self, root=root, legacy_root=post_origin) == validation_path,
             f"{validation_context}：二次校验报告自引用路径不一致。")
    files = _object(validation.get("files"), f"{validation_context}.files")
    expected_counts = _object(processing.get("processed_counts"), f"{context}.processed_counts")
    _require(set(files) == set(SPLITS) and set(expected_counts) == set(SPLITS), f"{context}：必须完整绑定 train 和 validation 两个集合。")
    for split in SPLITS:
        info = _object(files[split], f"{validation_context}.files.{split}")
        recorded = _receipt_string(info.get("path"), f"{validation_context}.files.{split}.path")
        expected_count = _positive_count(expected_counts[split], f"{context}.processed_counts.{split}")
        scanned_count = _positive_count(info.get("records_scanned"), f"{validation_context}.files.{split}.records_scanned")
        _require(resolve_receipt_path(recorded, root=root, legacy_root=post_origin) == paths[split]
                 and info.get("sha256") == fingerprints[split] and scanned_count == expected_count,
                 f"{validation_context} [split={split}]：CSV 路径、摘要或记录数与处理凭据不一致。")
        _require(info.get("scan_complete") is True, f"{validation_context} [split={split}]：集合扫描不完整。")
        if "sha256_after_scan" in info:
            _require(info["sha256_after_scan"] == fingerprints[split], f"{validation_context} [split={split}]：校验期间 CSV 摘要变化。")
        _require(_fingerprint(paths[split]) == fingerprints[split], f"{paths[split]} [split={split}]：CSV SHA-256 摘要不匹配。")
    # 不逐条向外产出：验证集失败时，调用方也不会拿到一份看似可用的训练集。
    loaded = {split: read_split(paths[split], split=split, expected_sha256=fingerprints[split]) for split in SPLITS}
    for split in SPLITS:
        _require(len(loaded[split]) == expected_counts[split], f"{paths[split]} [split={split}]：实际记录数与 processed_counts 不一致。")
    for role, path in paths.items():
        _require(_fingerprint(path) == fingerprints[role], f"{path}：文件或报告在读取期间发生变化。")
    return DatasetBundle(loaded["train"], loaded["validation"], get_label_config(),
                         {role: {"path": str(path), "sha256": fingerprints[role]} for role, path in paths.items()})


def main(argv=None) -> int:
    """只读加载所选批次并打印摘要，便于命令行核验训练输入。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="严格读取清洗后的评论数据，核对批次凭据并打印汇总。")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--processing-report", type=Path, default=Path("data/reports/processed/processing_report.json"))
    args = parser.parse_args(argv)
    try:
        bundle = load_datasets(args.processing_report, root=args.root)
    except (DatasetReadError, OSError) as error:
        print(f"读取失败：{error}", file=sys.stderr)
        return 1
    datasets = {split: getattr(bundle, split) for split in SPLITS}
    class_counts = {split: [0] * NUM_CLASSES for split in SPLITS}
    for split, samples in datasets.items():
        for sample in samples:
            for label in sample.labels:
                class_counts[split][label] += 1
    observed = [label for counts in class_counts.values() for label, count in enumerate(counts) if count]
    print(json.dumps({
        "counts": {split: len(samples) for split, samples in datasets.items()},
        "label_shapes": {split: [len(samples), NUM_ASPECTS] for split, samples in datasets.items()},
        "encoded_label_range": [min(observed), max(observed)],
        "class_counts": class_counts,
        "label_config": bundle.label_config,
        "sources": bundle.sources,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
