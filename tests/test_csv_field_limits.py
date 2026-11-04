"""验证同进程 CSV 读取共享字段上限，并在成功、异常和混合导入时正确恢复。

用事件阻挡实际 CSV 解析打开，不使用 sleep，也不要求等待者在锁释放前进入
解析正文。所有等待和线程清理均有界，临时数据不会触碰仓库的真实 CSV。
"""

import csv
import importlib.util
from pathlib import Path
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import patch

from scripts import dataset_io, dataset_reader, validate_dataset
from scripts.label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS


class CSVFieldLimitTests(unittest.TestCase):
    """读取器和校验器串行管理标准库的进程全局字段上限。"""

    def setUp(self):
        """先用较小上限模拟其他调用方，再要求所有工作线程结束后恢复它。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.inputs = self.root / "inputs"
        self.inputs.mkdir()
        self.original_limit = csv.field_size_limit()
        self.addCleanup(csv.field_size_limit, self.original_limit)
        self.small_limit = 64
        csv.field_size_limit(self.small_limit)

    @staticmethod
    def write_csv(path, text, *, invalid=False):
        """生成规范表头和单条评论，长文本超过 Python CSV 默认字段上限。"""
        row = {"id": "01", "content": text, **{aspect: "-2" for aspect in ASPECT_COLUMNS}}
        if invalid:
            row[ASPECT_COLUMNS[-1]] = "1.0"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EXPECTED_COLUMNS)
            writer.writeheader()
            writer.writerow(row)

    @staticmethod
    def wait_for(event, message):
        """让失败的同步点显式报错，避免用例永久阻塞。"""
        if not event.wait(5):
            raise TimeoutError(message)

    def assert_serialized_csv_parsing(self, first_role, *, first_invalid=False):
        """阻挡首解析，启动另一入口，随后要求后者读取完整长评论。"""
        first_path = self.inputs / "first.csv"
        second_path = self.inputs / "second.csv"
        long_text = "长评论" * 90000
        self.write_csv(first_path, "首个短评论", invalid=first_invalid)
        self.write_csv(second_path, long_text)
        second_role = "validator" if first_role == "reader" else "reader"
        first_entered, second_entered, second_started = Event(), Event(), Event()
        allow_first, allow_second = Event(), Event()
        first_done = Event()
        results, errors = {}, {}
        real_open = Path.open

        def observed_open(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            # SHA-256 也会以 rb 打开文件；只阻挡两个入口真正开始 CSV 解析的位置。
            parsing = kwargs.get("newline") == "" or (mode == "rb" and kwargs.get("buffering") == 0)
            if parsing and path == first_path:
                first_entered.set()
                self.wait_for(allow_first, "首个 CSV 解析未获释放")
            elif parsing and path == second_path:
                second_entered.set()
                self.wait_for(allow_second, "第二个 CSV 解析未获释放")
            return real_open(path, *args, **kwargs)

        def operation(role, path, report_name):
            if role == "reader":
                return dataset_reader.read_split(path, split="train")
            return validate_dataset.validate_dataset({"train": path}, self.root / "reports" / report_name,
                                                     root=self.root)

        def run_first():
            try:
                results["first"] = operation(first_role, first_path, "first")
            except BaseException as error:
                errors["first"] = error
            finally:
                first_done.set()

        def run_second():
            second_started.set()
            try:
                results["second"] = operation(second_role, second_path, "second")
            except BaseException as error:
                errors["second"] = error

        first = Thread(target=run_first, name="csv-field-first", daemon=True)
        second = Thread(target=run_second, name="csv-field-second", daemon=True)
        with patch.object(Path, "open", observed_open):
            try:
                first.start()
                self.wait_for(first_entered, "首个入口未开始解析")
                second.start()
                self.wait_for(second_started, "第二个入口未启动")
                # 等待者应被共享锁挡住；旧的独立管理会直接触发 second_entered。
                entered_before_release = second_entered.wait(1)
                allow_first.set()
                self.wait_for(first_done, "首个入口释放后仍未结束")
                self.wait_for(second_entered, "首个入口结束后等待者未开始解析")
                allow_second.set()
            finally:
                allow_first.set()
                allow_second.set()
                first.join(5)
                if second.ident is not None:
                    second.join(5)
        self.assertFalse(first.is_alive(), "首个工作线程未退出")
        self.assertFalse(second.is_alive(), "第二个工作线程未退出")
        self.assertFalse(entered_before_release, "两个入口同时进入了 CSV 解析，字段上限没有共享保护")
        if first_invalid:
            self.assertIsInstance(errors.get("first"), dataset_reader.DatasetReadError)
        else:
            self.assertNotIn("first", errors, str(errors.get("first")))
        self.assertNotIn("second", errors, str(errors.get("second")))
        if second_role == "reader":
            self.assertEqual(results["second"][0].text, long_text)
        else:
            self.assertTrue(results["second"]["scan_complete"])
            self.assertEqual(results["second"]["issue_counts"]["ERROR"], 0)
            self.assertEqual(results["second"]["files"]["train"]["records_scanned"], 1)
        self.assertEqual(csv.field_size_limit(), self.small_limit)

    def test_reader_then_validator_serializes_entire_parse_and_restores_limit(self):
        """读取器先进入时，校验器等它退出后仍能读取超默认上限的合法评论。"""
        self.assert_serialized_csv_parsing("reader")

    def test_validator_then_reader_serializes_entire_parse_and_restores_limit(self):
        """校验器先进入时，读取器等它退出后仍能读取完整长评论。"""
        self.assert_serialized_csv_parsing("validator")

    def test_reader_failure_releases_waiting_validator_and_restores_limit(self):
        """首个读取器因非法标签失败，等待的校验器仍可继续并恢复最初上限。"""
        self.assert_serialized_csv_parsing("reader", first_invalid=True)

    def test_separately_imported_helpers_share_lock_and_release_on_exception(self):
        """别名加载的模块实例也保护同一个 CSV 全局状态，异常不得遗留锁或上限。"""
        specification = importlib.util.spec_from_file_location("_granu_review_csv_io_alias", dataset_io.__file__)
        alias = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(alias)
        first_entered, second_entered, second_started = Event(), Event(), Event()
        allow_first, allow_second, first_done = Event(), Event(), Event()
        errors, observed = {}, []

        def run_first():
            try:
                with dataset_io.large_csv_fields():
                    first_entered.set()
                    self.wait_for(allow_first, "首个 helper 未获释放")
                    raise RuntimeError("首个 helper 正文异常")
            except BaseException as error:
                errors["first"] = error
            finally:
                first_done.set()

        def run_second():
            second_started.set()
            try:
                with alias.large_csv_fields():
                    observed.append(csv.field_size_limit())
                    second_entered.set()
                    self.wait_for(allow_second, "别名 helper 未获释放")
            except BaseException as error:
                errors["second"] = error

        first = Thread(target=run_first, name="csv-helper-first", daemon=True)
        second = Thread(target=run_second, name="csv-helper-alias", daemon=True)
        try:
            first.start()
            self.wait_for(first_entered, "首个 helper 未进入正文")
            second.start()
            self.wait_for(second_started, "别名 helper 未启动")
            entered_before_release = second_entered.wait(1)
            allow_first.set()
            self.wait_for(first_done, "首个 helper 异常后未退出")
            self.wait_for(second_entered, "别名 helper 未在异常后继续")
            allow_second.set()
        finally:
            allow_first.set()
            allow_second.set()
            first.join(5)
            if second.ident is not None:
                second.join(5)
        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertFalse(entered_before_release, "不同模块实例没有共用字段上限锁")
        self.assertIsInstance(errors.get("first"), RuntimeError)
        self.assertNotIn("second", errors, str(errors.get("second")))
        self.assertEqual(len(observed), 1)
        self.assertGreater(observed[0], self.small_limit)
        self.assertEqual(csv.field_size_limit(), self.small_limit)


if __name__ == "__main__":
    unittest.main()
