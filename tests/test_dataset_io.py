"""回归验证原子报告写入、可迁移凭据路径及 JSONL 的实际行边界。"""

import json
import os
from pathlib import Path, PureWindowsPath
import tempfile
import unittest
from unittest.mock import patch

from scripts.dataset_io import (
    file_sha256, read_jsonl, receipt_origin, receipt_path, resolve_receipt_path, write_json_atomic,
)
from scripts import validate_dataset


class DatasetIOTests(unittest.TestCase):
    def setUp(self):
        """使用独立临时目录，同时模拟项目上层与项目内部均含数据目录的布局。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name).resolve()
        # 上层目录也命名为 data，防止迁移时误把第一个同名目录认作项目数据根目录。
        self.root = self.directory / "data" / "current-project"
        self.root.mkdir(parents=True)
        self.report = self.root / "data/reports/raw/validation_report.json"

    def test_atomic_reports_preserve_bytes_and_legacy_imports(self):
        """公共工具迁移后保持旧导入、UTF-8、排序、缩进和实际行尾格式。"""
        self.assertIs(validate_dataset.file_sha256, file_sha256)
        self.assertIs(validate_dataset._write_json_atomic, write_json_atomic)
        path = self.directory / "report.json"
        value = {"z": "评论\n下一行\u0085\u2028\u2029", "a": 1}
        expected = '{\n  "a": 1,\n  "z": "评论\\n下一行\u0085\u2028\u2029"\n}\n'

        write_json_atomic(path, value)

        self.assertEqual(path.read_bytes(), expected.encode("utf-8"))
        write_json_atomic(path, [value, {"a": 2}], jsonl=True)
        expected = '{"a": 1, "z": "评论\\n下一行\u0085\u2028\u2029"}\n{"a": 2}\n'
        self.assertEqual(path.read_bytes(), expected.encode("utf-8"))
        self.assertEqual(read_jsonl(path), [value, {"a": 2}])

    def test_serialization_failure_preserves_existing_report_and_cleans_temporary_file(self):
        """序列化在写出部分内容后失败，既有报告也不能被截断或留下临时文件。"""
        path = self.directory / "report.json"
        original = b"existing report\n"
        path.write_bytes(original)
        before = set(self.directory.iterdir())
        for jsonl, value in ((False, {"a": 1, "z": object()}),
                             (True, [{"a": 1}, {"z": object()}])):
            with self.subTest(jsonl=jsonl):
                with self.assertRaises(TypeError):
                    write_json_atomic(path, value, jsonl=jsonl)
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(set(self.directory.iterdir()), before)

    def test_publish_failure_preserves_existing_report_and_cleans_temporary_file(self):
        """临时报告完整写入后若替换失败，继续保留旧报告并清理本次临时文件。"""
        path = self.directory / "report.json"
        original = b"existing report\n"
        path.write_bytes(original)
        before = set(self.directory.iterdir())

        with patch.object(Path, "replace", side_effect=OSError("cannot publish")):
            with self.assertRaisesRegex(OSError, "cannot publish"):
                write_json_atomic(path, {"a": "new report"})

        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(set(self.directory.iterdir()), before)

    def test_project_receipts_are_relative_and_external_receipts_remain_absolute(self):
        """项目内产物应随根目录迁移，外部数据则必须继续保留其明确的绝对位置。"""
        internal = self.root / "data/raw/train.csv"
        external = self.directory / "external/data/raw/train.csv"

        self.assertEqual(receipt_path(internal, self.root), "data/raw/train.csv")
        self.assertEqual(receipt_path(external, self.root), str(external.resolve()))

    def test_external_data_directory_is_not_rebased_without_a_legacy_origin(self):
        """仅因路径出现 data 不能推断其属于旧项目，也不能悄悄改读项目内同名文件。"""
        external = self.directory / "external/data/raw/train.csv"
        external.parent.mkdir(parents=True)
        external.write_text("external source", encoding="utf-8")

        actual = resolve_receipt_path(str(external), root=self.root)

        self.assertEqual(actual, external.resolve())
        self.assertEqual(actual.read_text(encoding="utf-8"), "external source")

    def test_external_data_directory_is_not_rebased_with_an_unrelated_legacy_origin(self):
        external = self.directory / "external/data/raw/train.csv"
        legacy_root = self.directory / "old-project"

        self.assertEqual(resolve_receipt_path(str(external), root=self.root, legacy_root=str(legacy_root)),
                         external.resolve())

    def test_relative_receipts_resolve_against_root_independently_of_working_directory(self):
        """改变进程工作目录后，凭据仍按显式项目根目录定位；最后恢复目录供其他测试使用。"""
        original_directory = Path.cwd()
        try:
            os.chdir(self.directory)
            actual = resolve_receipt_path("data/raw/train.csv", root=self.root)
        finally:
            os.chdir(original_directory)

        self.assertEqual(actual, (self.root / "data/raw/train.csv").resolve())

    def test_legacy_copy_is_used_even_while_the_old_file_still_exists(self):
        """新旧文件故意写不同内容，确保迁移后确实读取副本，避免旧目录掩盖定位错误。"""
        old_root = self.directory / "old-project"
        original = old_root / "data/raw/train.csv"
        copied = self.root / "data/raw/train.csv"
        original.parent.mkdir(parents=True)
        copied.parent.mkdir(parents=True)
        original.write_text("old content", encoding="utf-8")
        copied.write_text("copied content", encoding="utf-8")

        actual = resolve_receipt_path(str(original), root=self.root, legacy_root=str(old_root))

        self.assertEqual(actual, copied.resolve())
        self.assertEqual(actual.read_text(encoding="utf-8"), "copied content")

    def test_legacy_prefix_must_match_a_complete_path_component(self):
        """目录名前缀相同并不表示位于同一项目，判断边界必须落在完整路径组件上。"""
        old_root = self.directory / "old-project"
        external = self.directory / "old-project-copy/data/raw/train.csv"

        self.assertEqual(resolve_receipt_path(str(external), root=self.root, legacy_root=str(old_root)),
                         external.resolve())

    def test_windows_origin_is_inferred_from_the_complete_report_suffix(self):
        """根据完整报告后缀推导 Windows 旧根目录，正确处理上层重复出现的数据目录名。"""
        recorded = r"D:\archive\data\old-project\data\reports\raw\validation_report.json"

        origin = receipt_origin(recorded, self.report, self.root)

        self.assertIsNotNone(origin)
        self.assertEqual(PureWindowsPath(origin), PureWindowsPath(r"D:\archive\data\old-project"))
        self.assertEqual(resolve_receipt_path(r"D:\archive\data\old-project\data\raw\train.csv",
                                             root=self.root, legacy_root=origin),
                         (self.root / "data/raw/train.csv").resolve())

    def test_posix_origin_is_inferred_and_rebased_on_any_host(self):
        """旧凭据的路径风格独立于执行主机，Windows 主机也要正确识别 POSIX 旧根目录。"""
        recorded = "/srv/data/old-project/data/reports/raw/validation_report.json"

        origin = receipt_origin(recorded, self.report, self.root)

        self.assertIsNotNone(origin)
        self.assertEqual(str(origin).replace("\\", "/").rstrip("/"), "/srv/data/old-project")
        self.assertEqual(resolve_receipt_path("/srv/data/old-project/data/raw/train.csv",
                                             root=self.root, legacy_root=origin),
                         (self.root / "data/raw/train.csv").resolve())

    def test_origin_is_not_invented_for_relative_or_mismatched_report_paths(self):
        """缺乏一致的绝对报告路径时不猜测旧根目录，避免把无关数据绑定到当前项目。"""
        self.assertIsNone(receipt_origin("data/reports/raw/validation_report.json", self.report, self.root))
        self.assertIsNone(receipt_origin("/srv/old-project/elsewhere/validation_report.json",
                                         self.report, self.root))

    def test_unicode_separators_inside_json_strings_do_not_split_records(self):
        """字符串中的 Unicode 分隔符不是 JSONL 记录边界，普通空行和行尾换行仍可忽略。"""
        rows = [
            {"id": "a", "text": "价格\u0085\u2028\u2029合理，\n下次还来"},
            {"id": "b", "text": "正常评论"},
        ]
        path = self.directory / "issues.jsonl"
        encoded = "\r\n\r\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\r\n"
        path.write_bytes(encoded.encode("utf-8"))

        self.assertEqual(read_jsonl(path), rows)


if __name__ == "__main__":
    unittest.main()
