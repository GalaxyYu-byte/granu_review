"""验证训练读取的标签契约、原文保留、批次绑定和完整加载边界。

所有 CSV 与报告均在临时目录构造，二次校验报告由现有校验器实际生成，
不依赖仓库中的真实数据或已经保存的校验结果。
"""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import csv
from dataclasses import FrozenInstanceError
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import dataset_reader
from scripts.dataset_io import file_sha256, receipt_path
from scripts.dataset_reader import DatasetReadError, load_datasets, read_split
from scripts.label_config import ASPECT_COLUMNS, CLASS_VALUES, EXPECTED_COLUMNS, RAW_LABEL_MEANINGS, get_label_config
from scripts.validate_dataset import validate_dataset


FILENAMES = {
    "train": "sentiment_analysis_trainingset.csv",
    "validation": "sentiment_analysis_validationset.csv",
}


class DatasetReaderTests(unittest.TestCase):
    """单份 CSV 的读取失败不得被悄悄修复，批次必须与凭据精确对应。"""

    def setUp(self):
        """为每个用例分配独立项目根目录，避免批次和全局 CSV 设置互相污染。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        self.root = self.directory / "project"
        self.root.mkdir()
        self.single = self.root / "single.csv"

    @staticmethod
    def row(sample_id="007", text="正常评论", shift=0):
        """交错安排四类标签，使每个方面的次序或映射错误都能被发现。"""
        return {
            "id": sample_id,
            "content": text,
            **{aspect: str(CLASS_VALUES[(index + shift) % 4])
               for index, aspect in enumerate(ASPECT_COLUMNS)},
        }

    @staticmethod
    def write_csv(path, rows, *, header=EXPECTED_COLUMNS, encoding="utf-8"):
        """通过标准写入器构造合法转义，读者必须自行恢复评论内部的原始行尾。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding=encoding, newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            for row in rows:
                writer.writerow([row[column] for column in header] if isinstance(row, dict) else row)

    @staticmethod
    def save_processing(batch):
        """保存测试的处理凭据，保留真实摘要绑定的二次校验产物。"""
        batch.processing_path.write_text(json.dumps(batch.processing, ensure_ascii=False), encoding="utf-8")

    def save_validation(self, batch):
        """刷新校验报告及其处理凭据摘要，使负例真正测试内容一致性。"""
        batch.validation_path.write_text(json.dumps(batch.validation, ensure_ascii=False), encoding="utf-8")
        batch.processing["outputs"]["validation_report"]["sha256"] = file_sha256(batch.validation_path)
        self.save_processing(batch)

    def make_batch(self, name="processed", *, root=None, absolute=False, schema_version="1.0"):
        """生成真实二次校验报告和兼容历史版本的最小处理完成凭据。"""
        root = Path(root or self.root).resolve()
        paths = {split: root / "data" / name / filename for split, filename in FILENAMES.items()}
        rows = {
            split: [self.row(str(index * 10 + 1), f"{name} {split} 评论 {index}", index)
                    for index in range(4)]
            for split in FILENAMES
        }
        for split, path in paths.items():
            self.write_csv(path, rows[split])
        reports_dir = root / "data/reports" / name
        validation = validate_dataset(paths, reports_dir, root=root)
        validation_path = reports_dir / "validation_report.json"
        processing_path = reports_dir / "processing_report.json"
        path_value = lambda path: str(path.resolve()) if absolute else receipt_path(path, root)
        if absolute:
            validation["artifacts"]["report"] = str(validation_path)
            for split, path in paths.items():
                validation["files"][split]["path"] = str(path)
            validation_path.write_text(json.dumps(validation, ensure_ascii=False), encoding="utf-8")
        processing = {
            "schema_version": schema_version,
            "stage": "cleaned_original_label_space",
            "mapping_applied": False,
            "post_validation_status": validation["status"],
            "post_validation_issue_counts": deepcopy(validation["issue_counts"]),
            "processed_counts": {split: len(values) for split, values in rows.items()},
            "outputs": {
                **{split: {"path": path_value(path), "sha256": file_sha256(path)}
                   for split, path in paths.items()},
                "validation_report": {"path": path_value(validation_path), "sha256": file_sha256(validation_path)},
            },
        }
        batch = SimpleNamespace(root=root, paths=paths, rows=rows, validation=validation,
                                validation_path=validation_path, processing=processing,
                                processing_path=processing_path)
        self.save_processing(batch)
        return batch

    def test_preserves_bom_quotes_whitespace_and_multiline_record_positions(self):
        """BOM、引号、逗号、CRLF、Unicode 分隔符和表情均不改变评论内容。"""
        text = '  "味道,不错"\r\n下一行\n/流泪🤗\u0085\u2028\u2029  '
        rows = [self.row("007", text), self.row("99", "第二条评论", 1)]
        self.write_csv(self.single, rows, encoding="utf-8-sig")
        original = self.single.read_bytes()

        samples = read_split(self.single, split="train")

        self.assertEqual(len(samples), 2)
        self.assertEqual(samples[0].sample_id, "007")
        self.assertEqual(samples[0].text, text)
        self.assertEqual(samples[0].labels, tuple(index % 4 for index in range(20)))
        self.assertIsInstance(samples[0].labels, tuple)
        self.assertEqual(samples[0].processed_record_number, 1)
        self.assertEqual(samples[0].physical_line_end, 4)
        self.assertEqual(samples[1].processed_record_number, 2)
        self.assertEqual(samples[1].physical_line_end, 5)
        self.assertEqual(self.single.read_bytes(), original)
        with self.assertRaises(FrozenInstanceError):
            samples[0].text = "不允许改写"

    def test_both_splits_encode_identically_and_allow_non_contiguous_string_ids(self):
        """读取顺序不由 ID 排序，同一记录在两份集合中使用相同的方面与类别语义。"""
        self.write_csv(self.single, [self.row("099"), self.row("2", "另一条评论", 2)])
        train = read_split(self.single, split="train")
        validation = read_split(self.single, split="validation")
        self.assertEqual([sample.sample_id for sample in train], ["099", "2"])
        self.assertEqual([sample.labels for sample in train], [sample.labels for sample in validation])
        self.assertEqual([sample.split for sample in train], ["train", "train"])
        self.assertEqual([sample.split for sample in validation], ["validation", "validation"])

    def test_rejects_empty_file_and_header_only_dataset(self):
        """没有数据样本的输入不能作为成功的训练或验证集合返回。"""
        for content in (b"", (",".join(EXPECTED_COLUMNS) + "\n").encode("utf-8")):
            with self.subTest(content=content[:25]):
                self.single.write_bytes(content)
                with self.assertRaises(DatasetReadError):
                    read_split(self.single, split="train")

    def test_rejects_missing_extra_duplicate_and_reordered_headers(self):
        """固定列契约在字典化之前检查，不能掩盖重名或改变字段含义。"""
        headers = [EXPECTED_COLUMNS[:-1], (*EXPECTED_COLUMNS, "extra"),
                   (*EXPECTED_COLUMNS[:-1], EXPECTED_COLUMNS[-2]),
                   (EXPECTED_COLUMNS[1], EXPECTED_COLUMNS[0], *EXPECTED_COLUMNS[2:])]
        for header in headers:
            with self.subTest(header=header):
                self.write_csv(self.single, [], header=header)
                with self.assertRaises(DatasetReadError):
                    read_split(self.single, split="train")

    def test_rejects_bad_record_width_without_skipping_to_later_valid_samples(self):
        """首条合法记录之后遇到短行或多余字段时，不能返回部分训练数据。"""
        row = list(self.row().values())
        for bad_row in (row[:-1], row + ["extra"], []):
            with self.subTest(width=len(bad_row)):
                self.write_csv(self.single, [row, bad_row, row])
                with self.assertRaises(DatasetReadError):
                    read_split(self.single, split="train")

    def test_rejects_empty_whitespace_ids_texts_and_duplicate_ids(self):
        """空白检查不等于清洗；同一集合重复 ID 始终阻止加载。"""
        cases = [[self.row(sample_id="")], [self.row(sample_id=" \t")],
                 [self.row(text="")], [self.row(text=" \r\n\t")],
                 [self.row(), self.row(text="另一个内容")]]
        for rows in cases:
            with self.subTest(rows=rows):
                self.write_csv(self.single, rows)
                with self.assertRaises(DatasetReadError):
                    read_split(self.single, split="train")

    def test_invalid_labels_report_file_split_record_id_and_aspect(self):
        """标签错误定位到具体方面，不填默认值、不截断数字或自动去空格。"""
        aspect = ASPECT_COLUMNS[15]
        for value in ("", "1.0", " 1", "1 ", "nan", "2", "3", "1e0", "１"):
            row = self.row("bad-id")
            row[aspect] = value
            self.write_csv(self.single, [row])
            with self.subTest(value=value):
                with self.assertRaises(DatasetReadError) as raised:
                    read_split(self.single, split="train")
                message = str(raised.exception)
                for evidence in (str(self.single), "train", "bad-id", aspect, "1"):
                    self.assertIn(evidence, message)

    def test_rejects_invalid_utf8_and_unterminated_csv_quote(self):
        """编码与解析失败同样转换为可定位的读取异常。"""
        header = (",".join(EXPECTED_COLUMNS) + "\n").encode("utf-8")
        for body in (b'bad,\xff\n', b'bad,"unterminated\n'):
            with self.subTest(body=body):
                self.single.write_bytes(header + body)
                with self.assertRaises(DatasetReadError):
                    read_split(self.single, split="validation")

    def test_invalid_utf8_reports_actual_record_and_physical_line(self):
        """后续记录的坏字节不能在表头预读时误报，跨行记录也按真实物理行定位。"""
        cases = [
            ("前条评论", "坏字节位置", 3),
            ("前条评论\r\n续行", "坏字节位置", 4),
            ("前条评论", "当前评论\r\n坏字节位置", 4),
        ]
        for newline in (b"\n", b"\r\n", b"\r"):
            for first_text, bad_text, physical_line in cases:
                with self.subTest(newline=newline, first_text=first_text, bad_text=bad_text):
                    self.write_csv(self.single, [self.row("previous-id", first_text),
                                                 self.row("bad-id", bad_text)])
                    content = self.single.read_bytes().replace(b"\r\n", newline)
                    content = content.replace("坏字节位置".encode("utf-8"), b"\xff")
                    self.single.write_bytes(content)

                    with self.assertRaises(DatasetReadError) as raised:
                        read_split(self.single, split="validation")

                    message = str(raised.exception)
                    self.assertIn(str(self.single), message)
                    self.assertIn("split=validation,", message)
                    self.assertIn("processed_record_number=2,", message)
                    self.assertIn(f"physical_line_end={physical_line},", message)
                    self.assertIn("id=None", message)
                    self.assertNotIn("previous-id", message)
                    self.assertIsInstance(raised.exception.__cause__, UnicodeDecodeError)
                    self.assertEqual(self.single.read_bytes(), content)

    def test_invalid_utf8_in_header_reports_first_physical_line(self):
        """表头没有样本记录号，但解码失败的位置仍是物理第 1 行。"""
        header = (",".join(EXPECTED_COLUMNS) + "\n").encode("utf-8")
        for bom in (b"", b"\xef\xbb\xbf"):
            with self.subTest(bom=bom):
                self.single.write_bytes(bom + header.replace(b"id", b"\xff", 1))
                with self.assertRaises(DatasetReadError) as raised:
                    read_split(self.single, split="train")

                message = str(raised.exception)
                self.assertIn("processed_record_number=0,", message)
                self.assertIn("physical_line_end=1,", message)
                self.assertIn("id=None", message)
                self.assertIsInstance(raised.exception.__cause__, UnicodeDecodeError)

    def test_truncated_utf8_at_eof_reports_current_record_and_line(self):
        """末尾缺失多字节字符的后续字节时，也必须保留当前记录和实际行号。"""
        for newline in (b"\n", b"\r\n", b"\r"):
            with self.subTest(newline=newline):
                self.write_csv(self.single, [self.row("previous-id", "前条评论")])
                content = self.single.read_bytes().replace(b"\r\n", newline)
                self.single.write_bytes(content + b'bad-id,"truncated \xe4\xb8')
                with self.assertRaises(DatasetReadError) as raised:
                    read_split(self.single, split="train")

                message = str(raised.exception)
                self.assertIn("processed_record_number=2,", message)
                self.assertIn("physical_line_end=3,", message)
                self.assertIn("id=None", message)
                self.assertNotIn("previous-id", message)
                self.assertIsInstance(raised.exception.__cause__, UnicodeDecodeError)

    def test_missing_file_and_unknown_split_are_explicit_errors(self):
        """不存在的文件或未知集合名称不能产生空数据。"""
        with self.assertRaises(DatasetReadError):
            read_split(self.single, split="train")
        self.write_csv(self.single, [self.row()])
        with self.assertRaises(DatasetReadError):
            read_split(self.single, split="test")

    def test_long_fields_restore_process_limit_on_success_and_label_failure(self):
        """长评论超过默认上限仍完整读取，成功和失败均恢复调用方设置。"""
        previous_limit = csv.field_size_limit()
        try:
            csv.field_size_limit(64)
            long_text = "长评论" * 70000
            self.write_csv(self.single, [self.row(text=long_text)])
            self.assertEqual(read_split(self.single, split="train")[0].text, long_text)
            self.assertEqual(csv.field_size_limit(), 64)
            row = self.row(text=long_text)
            row[ASPECT_COLUMNS[0]] = "1.0"
            self.write_csv(self.single, [row])
            with self.assertRaises(DatasetReadError):
                read_split(self.single, split="train")
            self.assertEqual(csv.field_size_limit(), 64)
        finally:
            csv.field_size_limit(previous_limit)

    def test_loads_historical_and_new_receipts_with_provenance_and_shared_ids(self):
        """现有 1.0 凭据和 1.1 凭据都可读取，不强制新增审计字段。"""
        for version in ("1.0", "1.1"):
            with self.subTest(version=version):
                batch = self.make_batch("version_" + version, schema_version=version)
                self.assertEqual(batch.validation["pending_reviews"], 0)
                self.assertIn("post_validation_issue_counts", batch.processing)
                for split in FILENAMES:
                    self.assertEqual(batch.validation["files"][split]["sha256_after_scan"],
                                     batch.processing["outputs"][split]["sha256"])
                before = {path: path.read_bytes()
                          for path in (*batch.paths.values(), batch.processing_path, batch.validation_path)}
                bundle = load_datasets(batch.processing_path, root=self.root)
                self.assertEqual(len(bundle.train), 4)
                self.assertEqual(len(bundle.validation), 4)
                self.assertEqual(bundle.label_config, get_label_config())
                self.assertEqual([s.sample_id for s in bundle.train], ["1", "11", "21", "31"])
                self.assertEqual(bundle.train[0].sample_id, bundle.validation[0].sample_id)
                self.assertEqual(set(bundle.sources), {"processing_report", "validation_report", "train", "validation"})
                expected_paths = {**batch.paths, "processing_report": batch.processing_path,
                                  "validation_report": batch.validation_path}
                for name, path in expected_paths.items():
                    self.assertEqual(bundle.sources[name]["path"], str(path))
                    self.assertEqual(bundle.sources[name]["sha256"], file_sha256(path))
                for path, original in before.items():
                    self.assertEqual(path.read_bytes(), original)

    def test_refuses_mapped_or_unspecified_label_spaces_even_with_ambiguous_values(self):
        """数值只含 0/1 也不能代替明确的原始标签阶段声明。"""
        batch = self.make_batch()
        for split, path in batch.paths.items():
            rows = deepcopy(batch.rows[split])
            for row in rows:
                for index, aspect in enumerate(ASPECT_COLUMNS):
                    row[aspect] = str(index % 2)
            self.write_csv(path, rows)
            digest = file_sha256(path)
            batch.processing["outputs"][split]["sha256"] = digest
            batch.validation["files"][split]["sha256"] = digest
            batch.validation["files"][split]["sha256_after_scan"] = digest
        self.save_validation(batch)
        # 先证明这份数值有歧义的 CSV 可以按原始标签契约加载，后面的拒绝必须来自阶段声明。
        self.assertEqual(len(load_datasets(batch.processing_path, root=self.root).train), 4)
        original = deepcopy(batch.processing)
        cases = [("stage", "mapped"), ("stage", None), ("mapping_applied", True),
                 ("mapping_applied", 0), ("mapping_applied", "false"), ("mapping_applied", None)]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                batch.processing = deepcopy(original)
                if value is None:
                    batch.processing.pop(field)
                else:
                    batch.processing[field] = value
                self.save_processing(batch)
                with self.assertRaises(DatasetReadError):
                    load_datasets(batch.processing_path, root=self.root)

    def test_content_changes_with_same_record_count_fail_digest_check(self):
        """行数不变不能掩盖 CSV 内容已经偏离处理完成凭据。"""
        batch = self.make_batch()
        changed = deepcopy(batch.rows["train"])
        changed[0]["content"] = "修改后的评论"
        self.write_csv(batch.paths["train"], changed)
        with self.assertRaises(DatasetReadError):
            load_datasets(batch.processing_path, root=self.root)

    def test_modified_validation_report_fails_its_receipt_digest(self):
        """二次校验报告本身同样必须与处理凭据绑定。"""
        batch = self.make_batch()
        batch.validation_path.write_text(batch.validation_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(DatasetReadError):
            load_datasets(batch.processing_path, root=self.root)

    def test_rejects_swapped_report_label_mapping_before_reading_samples(self):
        """刷新报告摘要也不能让报告类别顺序偏离训练编码规则。"""
        for version in ("1.0", "1.1"):
            with self.subTest(version=version):
                batch = self.make_batch("swapped_mapping_" + version, schema_version=version)
                mapping = batch.validation["configuration"]["label_mapping"]
                mapping["-2"], mapping["1"] = mapping["1"], mapping["-2"]
                self.save_validation(batch)
                with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                    with self.assertRaisesRegex(DatasetReadError, r"configuration\.label_mapping"):
                        load_datasets(batch.processing_path, root=self.root)
                    reader.assert_not_called()

    def test_rejects_invalid_report_label_mappings_before_reading_samples(self):
        """映射必须完整且严格使用整数编号，不能靠缺省值或隐式相等通过。"""
        expected = {str(raw): index for index, raw in enumerate(CLASS_VALUES)}
        cases = [
            ("missing_field", None),
            ("null", None),
            ("list", list(expected.items())),
            ("missing_key", {key: value for key, value in expected.items() if key != "-2"}),
            ("extra_key", {**expected, "2": 4}),
            ("wrong_number", {**expected, "-2": 4}),
            ("boolean", {**expected, "-2": False}),
            ("float", {**expected, "-2": 0.0}),
            ("string_number", {**expected, "-2": "0"}),
        ]
        for version in ("1.0", "1.1"):
            for name, mapping in cases:
                with self.subTest(version=version, case=name):
                    batch = self.make_batch("invalid_mapping_" + version + "_" + name,
                                            schema_version=version)
                    configuration = batch.validation["configuration"]
                    if name == "missing_field":
                        del configuration["label_mapping"]
                    else:
                        configuration["label_mapping"] = deepcopy(mapping)
                    self.save_validation(batch)
                    with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                        with self.assertRaisesRegex(DatasetReadError, r"configuration\.label_mapping"):
                            load_datasets(batch.processing_path, root=self.root)
                        reader.assert_not_called()

    def test_loads_report_label_mapping_with_reordered_json_keys(self):
        """映射对象的键顺序不改变语义，两个集合仍按类别定义顺序编码。"""
        for version in ("1.0", "1.1"):
            with self.subTest(version=version):
                batch = self.make_batch("reordered_mapping_" + version, schema_version=version)
                mapping = batch.validation["configuration"]["label_mapping"]
                batch.validation["configuration"]["label_mapping"] = dict(reversed(list(mapping.items())))
                self.save_validation(batch)

                bundle = load_datasets(batch.processing_path, root=self.root)

                self.assertEqual(bundle.label_config["class_values"], list(CLASS_VALUES))
                for split in FILENAMES:
                    for shift, sample in enumerate(getattr(bundle, split)):
                        with self.subTest(split=split, sample=sample.sample_id):
                            self.assertEqual(sample.labels,
                                             tuple((index + shift) % len(CLASS_VALUES)
                                                   for index in range(len(ASPECT_COLUMNS))))

    def test_rejects_incompatible_report_aspect_columns_before_reading_samples(self):
        """报告方面轴必须与编码器一致，更新摘要不能掩盖顺序或结构变化。"""
        expected = list(EXPECTED_COLUMNS)
        swapped = expected.copy()
        swapped[2], swapped[3] = swapped[3], swapped[2]
        duplicate = expected.copy()
        duplicate[3] = duplicate[2]
        cases = [
            ("missing_field", None),
            ("null", None),
            ("swapped_aspects", swapped),
            ("missing_aspect", expected[:-1]),
            ("duplicate_aspect", duplicate),
            ("extra_aspect", [*expected, "invented_aspect"]),
            ("wrong_type", {"columns": expected}),
            ("non_string_column", [*expected[:-1], False]),
        ]
        for version in ("1.0", "1.1"):
            for name, columns in cases:
                with self.subTest(version=version, case=name):
                    batch = self.make_batch("invalid_columns_" + version + "_" + name,
                                            schema_version=version)
                    configuration = batch.validation["configuration"]
                    if name == "missing_field":
                        del configuration["expected_columns"]
                    else:
                        configuration["expected_columns"] = deepcopy(columns)
                    self.save_validation(batch)
                    with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                        with self.assertRaises(DatasetReadError):
                            load_datasets(batch.processing_path, root=self.root)
                        reader.assert_not_called()

    def test_rejects_incompatible_report_allowed_labels_before_reading_samples(self):
        """类别域的顺序、完整性和严格整数类型均应在加载样本前核对。"""
        expected = list(CLASS_VALUES)
        swapped = expected.copy()
        swapped[0], swapped[-1] = swapped[-1], swapped[0]
        cases = [
            ("missing_field", None),
            ("null", None),
            ("swapped_classes", swapped),
            ("missing_class", expected[:-1]),
            ("duplicate_class", [*expected[:-1], expected[-2]]),
            ("extra_class", [*expected, 2]),
            ("wrong_type", {"labels": expected}),
            ("boolean", [*expected[:-1], True]),
            ("float", [*expected[:2], 0.0, *expected[3:]]),
            ("string_number", [str(value) for value in expected]),
        ]
        for version in ("1.0", "1.1"):
            for name, labels in cases:
                with self.subTest(version=version, case=name):
                    batch = self.make_batch("invalid_allowed_" + version + "_" + name,
                                            schema_version=version)
                    configuration = batch.validation["configuration"]
                    if name == "missing_field":
                        del configuration["allowed_labels"]
                    else:
                        configuration["allowed_labels"] = deepcopy(labels)
                    self.save_validation(batch)
                    with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                        with self.assertRaises(DatasetReadError):
                            load_datasets(batch.processing_path, root=self.root)
                        reader.assert_not_called()

    def test_rejects_incompatible_report_label_meanings_before_reading_samples(self):
        """映射编号相同也必须拒绝类别含义冲突、缺失及类型不合法的报告。"""
        expected = {str(raw): meaning for raw, meaning in RAW_LABEL_MEANINGS.items()}
        swapped = expected.copy()
        swapped["-2"], swapped["1"] = swapped["1"], swapped["-2"]
        cases = [
            ("missing_field", None),
            ("null", None),
            ("list", list(expected.items())),
            ("swapped_meanings", swapped),
            ("missing_key", {key: value for key, value in expected.items() if key != "-2"}),
            ("extra_key", {**expected, "2": "invented"}),
            ("wrong_meaning", {**expected, "-2": "invented"}),
            ("empty_meaning", {**expected, "-2": ""}),
            ("boolean_meaning", {**expected, "-2": False}),
            ("number_meaning", {**expected, "-2": 0}),
            ("null_meaning", {**expected, "-2": None}),
            ("list_meaning", {**expected, "-2": []}),
            ("object_meaning", {**expected, "-2": {}}),
        ]
        for version in ("1.0", "1.1"):
            for name, meanings in cases:
                with self.subTest(version=version, case=name):
                    batch = self.make_batch("invalid_meanings_" + version + "_" + name,
                                            schema_version=version)
                    configuration = batch.validation["configuration"]
                    if name == "missing_field":
                        del configuration["raw_label_meanings"]
                    else:
                        configuration["raw_label_meanings"] = deepcopy(meanings)
                    self.save_validation(batch)
                    with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                        with self.assertRaisesRegex(DatasetReadError, r"configuration\.raw_label_meanings"):
                            load_datasets(batch.processing_path, root=self.root)
                        reader.assert_not_called()

    def test_loads_report_label_meanings_with_reordered_json_keys(self):
        """含义对象的键顺序不影响类别语义或两集合的标签编码。"""
        for version in ("1.0", "1.1"):
            with self.subTest(version=version):
                batch = self.make_batch("reordered_meanings_" + version, schema_version=version)
                meanings = batch.validation["configuration"]["raw_label_meanings"]
                batch.validation["configuration"]["raw_label_meanings"] = dict(reversed(list(meanings.items())))
                self.save_validation(batch)

                bundle = load_datasets(batch.processing_path, root=self.root)

                self.assertEqual(bundle.label_config, get_label_config())
                for split in FILENAMES:
                    for shift, sample in enumerate(getattr(bundle, split)):
                        with self.subTest(split=split, sample=sample.sample_id):
                            self.assertEqual(sample.labels,
                                             tuple((index + shift) % len(CLASS_VALUES)
                                                   for index in range(len(ASPECT_COLUMNS))))

    def test_selects_only_requested_batch_and_never_falls_back_on_missing_files(self):
        """默认批次存在时，显式指定的 v2 仍独立加载，缺文件立即失败。"""
        self.make_batch()
        batch = self.make_batch("processed_v2")
        bundle = load_datasets(receipt_path(batch.processing_path, self.root), root=self.root)
        self.assertTrue(bundle.train[0].text.startswith("processed_v2 "))
        self.assertEqual(bundle.sources["train"]["path"], str(batch.paths["train"]))
        batch.paths["train"].unlink()
        with self.assertRaises(DatasetReadError):
            load_datasets(batch.processing_path, root=self.root)

    def test_relative_receipts_are_resolved_against_root_not_working_directory(self):
        """从项目外运行时，报告和 CSV 都使用传入的项目根目录。"""
        batch = self.make_batch()
        original_directory = Path.cwd()
        try:
            os.chdir(self.directory)
            bundle = load_datasets("data/reports/processed/processing_report.json", root=self.root)
        finally:
            os.chdir(original_directory)
        self.assertEqual(bundle.sources["train"]["path"], str(batch.paths["train"]))

    def test_migrated_absolute_receipts_prefer_copy_even_when_old_project_exists(self):
        """旧绝对路径必须绑定到新副本，旧目录存在且被破坏也不能误读它。"""
        batch = self.make_batch(absolute=True)
        copied_root = self.directory / "copied-project"
        shutil.copytree(self.root, copied_root)
        batch.paths["train"].write_bytes(b"old CSV is no longer usable")
        batch.validation_path.write_bytes(b"old report is no longer usable")
        copied_report = copied_root / batch.processing_path.relative_to(self.root)

        bundle = load_datasets(copied_report, root=copied_root)

        self.assertEqual(len(bundle.train), 4)
        self.assertEqual(bundle.sources["train"]["path"],
                         str(copied_root / batch.paths["train"].relative_to(self.root)))
        self.assertEqual(bundle.sources["processing_report"]["path"], str(copied_report))

    def test_rejects_incomplete_failed_or_wrong_label_domain_validation(self):
        """最终放行须同时满足完整扫描、布尔 gate、合法状态和原始标签域。"""
        batch = self.make_batch()
        original = deepcopy(batch.validation)
        cases = [("scan_complete", False), ("scan_complete", 1),
                 ("validation_gate_passed", False), ("validation_gate_passed", 1),
                 ("status", "failed"), ("status", "needs_review"), ("stage", "mapped")]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                batch.validation = deepcopy(original)
                batch.validation[field] = value
                self.save_validation(batch)
                with self.assertRaises(DatasetReadError):
                    load_datasets(batch.processing_path, root=self.root)

    def test_allows_reviewed_findings_using_final_gate_instead_of_raw_review_count(self):
        """经复核放行后的 REVIEW 计数保留，不能因此拒绝整个批次。"""
        batch = self.make_batch()
        batch.validation.update(status="passed_with_warnings", scan_status="needs_review", pending_reviews=0,
                                review_resolutions=[{"issue_id": "approved", "source_issue_ids": ["source"]}])
        batch.validation["issue_counts"]["REVIEW"] = 2
        batch.processing["post_validation_status"] = "passed_with_warnings"
        batch.processing["post_validation_issue_counts"] = deepcopy(batch.validation["issue_counts"])
        self.save_validation(batch)

        bundle = load_datasets(batch.processing_path, root=self.root)

        self.assertEqual(len(bundle.train), 4)
        self.assertEqual(len(bundle.validation), 4)

    def test_refuses_errors_or_contradictory_final_status_and_issue_counts(self):
        """一个报告中的放行字段不能覆盖另一个报告声明的失败或不同扫描结果。"""
        for change in ("errors", "status", "counts"):
            with self.subTest(change=change):
                batch = self.make_batch(change)
                if change == "errors":
                    batch.validation["issue_counts"]["ERROR"] = 1
                    batch.processing["post_validation_issue_counts"] = deepcopy(batch.validation["issue_counts"])
                elif change == "status":
                    batch.processing["post_validation_status"] = "passed_with_warnings"
                else:
                    batch.processing["post_validation_issue_counts"]["WARN"] += 1
                self.save_validation(batch)
                with self.assertRaises(DatasetReadError):
                    load_datasets(batch.processing_path, root=self.root)

    def test_rejects_missing_post_validation_issue_counts_before_reading_samples(self):
        """两个支持版本都必须提供处理完成时的问题计数，不能跳过报告间对账。"""
        for version in ("1.0", "1.1"):
            with self.subTest(version=version):
                batch = self.make_batch("missing_counts_" + version, schema_version=version)
                del batch.processing["post_validation_issue_counts"]
                self.save_processing(batch)
                with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                    with self.assertRaises(DatasetReadError):
                        load_datasets(batch.processing_path, root=self.root)
                    reader.assert_not_called()

    def test_rejects_missing_pending_reviews_before_reading_samples(self):
        """两个支持版本均须明确声明待复核数，删除字段不能被解释为零。"""
        for version in ("1.0", "1.1"):
            with self.subTest(version=version):
                batch = self.make_batch("missing_pending_" + version, schema_version=version)
                del batch.validation["pending_reviews"]
                self.save_validation(batch)
                with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                    with self.assertRaises(DatasetReadError):
                        load_datasets(batch.processing_path, root=self.root)
                    reader.assert_not_called()

    def test_rejects_missing_after_scan_hash_before_reading_samples(self):
        """任一集合缺少扫描结束摘要时，两个报告版本均不能声明输入完整。"""
        for version in ("1.0", "1.1"):
            for split in FILENAMES:
                with self.subTest(version=version, split=split):
                    batch = self.make_batch("missing_after_" + version + "_" + split, schema_version=version)
                    del batch.validation["files"][split]["sha256_after_scan"]
                    self.save_validation(batch)
                    with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                        with self.assertRaises(DatasetReadError):
                            load_datasets(batch.processing_path, root=self.root)
                        reader.assert_not_called()

    def test_rejects_invalid_processing_issue_counts_before_reading_samples(self):
        """必需问题计数不能以 null、布尔值、数组或缺少级别的对象代替。"""
        cases = [None, True, False, 0, 0.0, "0", [], {},
                 {"ERROR": 0, "REVIEW": True, "WARN": 0}]
        for version in ("1.0", "1.1"):
            for index, value in enumerate(cases):
                with self.subTest(version=version, value=value):
                    batch = self.make_batch(f"invalid_counts_{version}_{index}", schema_version=version)
                    batch.processing["post_validation_issue_counts"] = value
                    self.save_processing(batch)
                    with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                        with self.assertRaises(DatasetReadError):
                            load_datasets(batch.processing_path, root=self.root)
                        reader.assert_not_called()

    def test_rejects_invalid_pending_review_counts_before_reading_samples(self):
        """待复核字段必须是精确整数零，False、0.0 或字符串零均不能放行。"""
        cases = [None, True, False, 0.0, "0", [], {}, 1, -1]
        for version in ("1.0", "1.1"):
            for index, value in enumerate(cases):
                with self.subTest(version=version, value=value):
                    batch = self.make_batch(f"invalid_pending_{version}_{index}", schema_version=version)
                    batch.validation["pending_reviews"] = value
                    self.save_validation(batch)
                    with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                        with self.assertRaises(DatasetReadError):
                            load_datasets(batch.processing_path, root=self.root)
                        reader.assert_not_called()

    def test_rejects_invalid_after_scan_hashes_before_reading_samples(self):
        """每个集合的结束摘要必须是合法 SHA-256，且与凭据声明的输入相同。"""
        cases = [None, True, False, 0, 0.0, [], {}, "invalid", "A" * 64, "0" * 64]
        for version in ("1.0", "1.1"):
            for split in FILENAMES:
                for index, value in enumerate(cases):
                    with self.subTest(version=version, split=split, value=value):
                        batch = self.make_batch(f"invalid_after_{version}_{split}_{index}", schema_version=version)
                        batch.validation["files"][split]["sha256_after_scan"] = value
                        self.save_validation(batch)
                        with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
                            with self.assertRaises(DatasetReadError):
                                load_datasets(batch.processing_path, root=self.root)
                            reader.assert_not_called()

    def test_validation_file_bindings_must_match_paths_hashes_and_counts(self):
        """即使报告摘要同步更新，二次校验也必须绑定同一文件而非另一份副本。"""
        for change in ("path", "sha256", "records_scanned"):
            with self.subTest(change=change):
                batch = self.make_batch("binding_" + change)
                info = batch.validation["files"]["train"]
                if change == "path":
                    copied_csv = batch.paths["train"].with_name("same_bytes_other_file.csv")
                    copied_csv.write_bytes(batch.paths["train"].read_bytes())
                    info["path"] = receipt_path(copied_csv, self.root)
                elif change == "sha256":
                    info["sha256"] = "0" * 64
                else:
                    info["records_scanned"] = 5
                self.save_validation(batch)
                with self.assertRaises(DatasetReadError):
                    load_datasets(batch.processing_path, root=self.root)

    def test_rejects_non_integer_empty_or_incorrect_processed_counts(self):
        """数量不能用布尔值冒充，也不能仅凭两个报告一致而忽略实际读取数量。"""
        for count in (True, "4", 4.0, 0, -1, 5):
            with self.subTest(count=count):
                batch = self.make_batch("count_" + str(count))
                batch.processing["processed_counts"]["train"] = count
                batch.validation["files"]["train"]["records_scanned"] = count
                self.save_validation(batch)
                with self.assertRaises(DatasetReadError):
                    load_datasets(batch.processing_path, root=self.root)

    def test_no_bundle_is_returned_when_second_split_is_invalid(self):
        """训练集已读完后验证集失败，也不能把训练集作为成功结果交付。"""
        batch = self.make_batch()
        rows = deepcopy(batch.rows["validation"])
        rows[-1][ASPECT_COLUMNS[-1]] = "1.0"
        self.write_csv(batch.paths["validation"], rows)
        digest = file_sha256(batch.paths["validation"])
        batch.processing["outputs"]["validation"]["sha256"] = digest
        batch.validation["files"]["validation"]["sha256"] = digest
        batch.validation["files"]["validation"]["sha256_after_scan"] = digest
        self.save_validation(batch)
        with patch.object(dataset_reader, "read_split", wraps=read_split) as reader:
            with self.assertRaises(DatasetReadError) as raised:
                load_datasets(batch.processing_path, root=self.root)
        self.assertEqual([call.kwargs["split"] for call in reader.call_args_list], ["train", "validation"])
        self.assertIn(ASPECT_COLUMNS[-1], str(raised.exception))
        self.assertIn("1.0", str(raised.exception))

    def test_detects_csv_or_report_changes_during_loading(self):
        """已经读取的 CSV 或报告在另一集合读取期间变化，最终仍须拒绝返回。"""
        real_read_split = read_split
        for role in ("train", "validation_report", "processing_report"):
            with self.subTest(role=role):
                batch = self.make_batch("concurrent_" + role)
                targets = {**batch.paths, "validation_report": batch.validation_path,
                           "processing_report": batch.processing_path}

                def read_and_change(path, *, split, expected_sha256=None):
                    samples = real_read_split(path, split=split, expected_sha256=expected_sha256)
                    if split == "validation":
                        with targets[role].open("ab") as stream:
                            stream.write(b"\n")
                    return samples

                with patch.object(dataset_reader, "read_split", side_effect=read_and_change):
                    with self.assertRaises(DatasetReadError):
                        load_datasets(batch.processing_path, root=self.root)

    def test_rejects_csv_swapped_after_prevalidation_even_if_original_is_restored(self):
        """预检后的临时替换不能令返回样本来自 B 而来源摘要仍声称是 A。"""
        batch = self.make_batch()
        original_bytes = batch.paths["train"].read_bytes()
        changed_rows = deepcopy(batch.rows["train"])
        changed_rows[0]["content"] = "预检之后临时替换的评论"
        real_read_split = read_split
        calls = []

        def read_swapped_csv(path, *, split, expected_sha256=None):
            calls.append(split)
            if split != "train":
                return real_read_split(path, split=split, expected_sha256=expected_sha256)
            self.write_csv(batch.paths["train"], changed_rows)
            try:
                return real_read_split(path, split=split, expected_sha256=expected_sha256)
            finally:
                batch.paths["train"].write_bytes(original_bytes)

        with patch.object(dataset_reader, "read_split", side_effect=read_swapped_csv):
            with self.assertRaises(DatasetReadError):
                load_datasets(batch.processing_path, root=self.root)
        self.assertEqual(calls, ["train"])
        self.assertEqual(batch.paths["train"].read_bytes(), original_bytes)

    def test_platform_field_limit_overflow_falls_back_and_restores_previous_limit(self):
        """模拟平台可用整数范围较小，第一次设置溢出后仍可读取并恢复全局配置。"""
        self.write_csv(self.single, [self.row(text="长评论" * 70000)])
        real_limit = csv.field_size_limit
        previous_limit = real_limit()
        maximum = sys.maxsize // 10
        attempted = []

        def bounded_limit(*args):
            if args:
                attempted.append(args[0])
                if args[0] > maximum:
                    raise OverflowError("模拟当前平台的 CSV 整数范围")
            return real_limit(*args)

        try:
            with patch.object(dataset_reader.csv, "field_size_limit", side_effect=bounded_limit):
                samples = read_split(self.single, split="train")
            self.assertEqual(samples[0].text, "长评论" * 70000)
            self.assertGreater(attempted[0], maximum)
            self.assertIn(maximum, attempted)
            self.assertEqual(real_limit(), previous_limit)
        finally:
            real_limit(previous_limit)

    def test_hashes_the_bytes_actually_parsed_when_path_snapshots_are_unchanged(self):
        """读取途中实际消费过临时改动，即使原文件及时恢复也不能返回成功。"""
        self.write_csv(self.single, [self.row("first", "首条短评论"),
                                     self.row("second", "A" * 30000)])
        original_bytes = self.single.read_bytes()
        original_sha = file_sha256(self.single)
        offset = original_bytes.index(b"A" * 30000) + 25000
        self.assertGreater(offset, 20000)
        real_encoder = dataset_reader.encode_labels
        consumed_temporary_text = []
        calls = 0

        def edit_during_encoding(row):
            nonlocal calls
            calls += 1
            if calls == 1:
                # 远处字节尚未进入初始读取缓冲，第二条记录必须从实际文件消费它。
                with self.single.open("r+b") as stream:
                    stream.seek(offset)
                    stream.write(b"B")
            elif calls == 2:
                consumed_temporary_text.append("B" in row["content"])
                with self.single.open("r+b") as stream:
                    stream.seek(offset)
                    stream.write(b"A")
            return real_encoder(row)

        try:
            with patch.object(dataset_reader, "encode_labels", side_effect=edit_during_encoding):
                with self.assertRaises(DatasetReadError):
                    read_split(self.single, split="train", expected_sha256=original_sha)
            self.assertEqual(calls, 2)
            self.assertEqual(consumed_temporary_text, [True])
            self.assertEqual(self.single.read_bytes(), original_bytes)
        finally:
            self.single.write_bytes(original_bytes)
        self.assertEqual(file_sha256(self.single), original_sha)

    def test_cli_prints_readonly_summary_and_returns_failure_for_invalid_batch(self):
        """命令行成功输出可解析汇总，失败输出明确错误并返回非零码。"""
        batch = self.make_batch()
        stdout, stderr = io.StringIO(), io.StringIO()
        arguments = ["--root", str(self.root), "--processing-report", str(batch.processing_path)]
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = dataset_reader.main(arguments)
        self.assertEqual(status, 0)
        self.assertEqual(stderr.getvalue(), "")
        summary = json.loads(stdout.getvalue())
        self.assertEqual(summary["counts"], {"train": 4, "validation": 4})
        self.assertEqual(summary["label_shapes"], {"train": [4, 20], "validation": [4, 20]})
        self.assertEqual(summary["encoded_label_range"], [0, 3])
        self.assertEqual(summary["class_counts"], {"train": [20, 20, 20, 20], "validation": [20, 20, 20, 20]})
        self.assertEqual(summary["label_config"], get_label_config())
        batch.processing["mapping_applied"] = True
        self.save_processing(batch)
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = dataset_reader.main(arguments)
        self.assertEqual(status, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn(str(batch.processing_path), stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
