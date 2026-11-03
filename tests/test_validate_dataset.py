"""验证 CSV 校验流程的对外行为，所有样本均在临时目录构造。

测试同时关注错误发现、扫描完整性和原始字节保护，不依赖仓库实际数据，
也不把生成报告本身当作输入未被改动的证据。
"""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.validate_dataset import ASPECT_COLUMNS, validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def setUp(self):
        """隔离输入与报告目录，避免异常样本和重复运行影响真实数据。"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.raw_dir = self.root / "data" / "raw"
        self.raw_dir.mkdir(parents=True)
        self.output_dir = self.root / "data" / "reports"
        self.header = ["id", "content", *ASPECT_COLUMNS]

    def write_csv(self, split, rows, header=None):
        """通过 CSV 写入器保留引号和多行字段，也允许显式构造异常表头。"""
        path = self.raw_dir / f"{split}.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.header if header is None else header)
            writer.writerows(rows)
        return path

    @staticmethod
    def row(sample_id, text, label=-2, **overrides):
        labels = {aspect: label for aspect in ASPECT_COLUMNS}
        labels.update(overrides)
        return [sample_id, text, *(labels[aspect] for aspect in ASPECT_COLUMNS)]

    def baseline_rows(self, split, labels=(-2, -1, 0, 1)):
        """为每个方面准备四类标签，避免无关的缺失类别告警干扰目标断言。"""
        return [self.row(str(i), f"{split} 独立评论 {i}", label)
                for i, label in enumerate(labels)]

    def read_issues(self):
        path = self.output_dir / "validation_issues.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]

    def validate(self, files, **kwargs):
        """除检查返回值外，还确认落盘报告与调用方实际看到的结果一致。"""
        report = validate_dataset(files, self.output_dir, **kwargs)
        saved_report = json.loads(
            (self.output_dir / "validation_report.json").read_text(encoding="utf-8")
        )
        self.assertEqual(saved_report, report)
        return report

    def test_valid_csv_preserves_quoted_multiline_text_and_input_bytes(self):
        """一条评论内的逗号、引号和换行不能改变记录数，也不能触发原文重写。"""
        rows = self.baseline_rows("train")
        text = '好吃, 他说"下次再来"\n第二行仍属于同一条评论'
        rows[0] = self.row("0", text)
        path = self.write_csv("train", rows)
        original = path.read_bytes()

        report = self.validate({"train": path})

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["scan_complete"])
        self.assertEqual(report["files"]["train"]["records_scanned"], 4)
        self.assertEqual(report["files"]["train"]["sha256"],
                         hashlib.sha256(original).hexdigest())
        self.assertEqual(report["issue_counts"], {"ERROR": 0, "REVIEW": 0, "WARN": 0})
        self.assertEqual(self.read_issues(), [])
        self.assertEqual(path.read_bytes(), original)

    def test_schema_errors_cover_missing_duplicate_extra_and_reordered_columns(self):
        row = self.row("0", "结构检查")
        reordered = self.header.copy()
        reordered[2], reordered[3] = reordered[3], reordered[2]
        variants = [
            ("missing", self.header[:-1], row[:-1]),
            ("duplicate", self.header[:-1] + [ASPECT_COLUMNS[0]], row),
            ("extra", self.header + ["unexpected"], row + ["value"]),
            ("reordered", reordered, row),
        ]
        for name, header, values in variants:
            with self.subTest(schema=name):
                path = self.write_csv("train", [values], header)
                report = self.validate({"train": path})
                self.assertEqual(report["status"], "failed")
                self.assertGreater(report["issue_counts"]["ERROR"], 0)

    def test_wrong_record_width_is_an_error_without_truncating_the_scan(self):
        """列宽错误应单独报告，不能让后续有效记录从统计中消失。"""
        for width_change in (-1, 1):
            with self.subTest(width_change=width_change):
                row = self.row("bad", "列数不一致")
                malformed = row[:-1] if width_change == -1 else row + ["extra"]
                path = self.write_csv("train", [malformed, self.row("ok", "后面的记录")])

                report = self.validate({"train": path})

                self.assertEqual(report["status"], "failed")
                self.assertTrue(report["files"]["train"]["scan_complete"])
                self.assertEqual(report["files"]["train"]["records_scanned"], 2)

    def test_unterminated_quote_marks_the_scan_incomplete(self):
        """解析器无法确定记录边界时保留已读条数，并明确标记扫描未完成。"""
        path = self.write_csv("train", [self.row("ok", "完整记录")])
        with path.open("a", encoding="utf-8", newline="") as stream:
            stream.write('broken,"unterminated')

        report = self.validate({"train": path})

        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["scan_complete"])
        self.assertFalse(report["files"]["train"]["scan_complete"])
        self.assertEqual(report["files"]["train"]["records_scanned"], 1)
        self.assertGreater(report["issue_counts"]["ERROR"], 0)

    def test_invalid_utf8_is_reported_as_an_incomplete_scan(self):
        path = self.raw_dir / "train.csv"
        path.write_bytes((",".join(self.header) + "\n").encode("utf-8") + b"0,\xff\n")

        report = self.validate({"train": path})

        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["scan_complete"])
        self.assertFalse(report["files"]["train"]["scan_complete"])
        self.assertGreater(report["issue_counts"]["ERROR"], 0)

    def test_missing_file_is_a_reported_failure(self):
        report = self.validate({"train": self.raw_dir / "missing.csv"})

        self.assertEqual(report["status"], "failed")
        self.assertFalse(report["scan_complete"])
        self.assertGreater(report["issue_counts"]["ERROR"], 0)

    def test_empty_text_id_and_duplicate_local_id_are_errors(self):
        cases = [
            ("empty_text", [self.row("0", " \t\n ")]),
            ("empty_id", [self.row("", "有内容")]),
            ("duplicate_id", [self.row("same", "第一条"), self.row("same", "第二条")]),
        ]
        for name, rows in cases:
            with self.subTest(case=name):
                path = self.write_csv("train", rows)
                report = self.validate({"train": path})
                self.assertEqual(report["status"], "failed")
                self.assertGreater(report["issue_counts"]["ERROR"], 0)

    def test_empty_noninteger_and_out_of_range_labels_are_not_repaired(self):
        for value in ("", "1.0", "nan", "2", "-3"):
            with self.subTest(value=value):
                path = self.write_csv("train", [self.row("0", "标签校验", dish_taste=value)])
                original = path.read_bytes()

                report = self.validate({"train": path})

                self.assertEqual(report["status"], "failed")
                self.assertGreater(report["issue_counts"]["ERROR"], 0)
                self.assertEqual(path.read_bytes(), original)

    def test_same_id_across_splits_is_allowed_for_split_scoped_ids(self):
        files = {split: self.write_csv(split, self.baseline_rows(split))
                 for split in ("train", "validation")}

        report = self.validate(files, id_scope="split")

        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["issue_counts"]["ERROR"], 0)
        self.assertEqual(report["issue_counts"]["REVIEW"], 0)

    def test_same_global_id_across_splits_requires_attention(self):
        files = {split: self.write_csv(split, self.baseline_rows(split))
                 for split in ("train", "validation")}

        report = self.validate(files, id_scope="global")

        self.assertIn(report["status"], ("failed", "needs_review"))
        self.assertGreater(report["issue_counts"]["ERROR"] + report["issue_counts"]["REVIEW"], 0)

    def test_normalized_cross_split_overlap_has_original_text_and_aspect_conflicts(self):
        """归一化仅用于发现候选重叠，问题证据仍须保留原文和逐方面标签。"""
        train_text = "Cafe\u0301  很好\n吃"
        validation_text = "Caf\u00e9 很好 吃"
        train = self.write_csv("train", [self.row("t1", train_text, dish_taste=1)])
        validation = self.write_csv("validation", [
            self.row("v1", validation_text, dish_taste=-1)
        ])
        original_bytes = {path: path.read_bytes() for path in (train, validation)}

        report = self.validate({"train": train, "validation": validation})

        self.assertEqual(report["status"], "needs_review")
        overlap_issues = [issue for issue in self.read_issues()
                          if {record["split"] for record in issue.get("records", [])}
                          == {"train", "validation"}]
        self.assertTrue(overlap_issues)
        conflict_issues = [issue for issue in overlap_issues
                           if "dish_taste" in issue.get("label_conflicts", {})]
        self.assertTrue(conflict_issues)
        issue = conflict_issues[0]
        self.assertEqual(issue["severity"], "REVIEW")
        members = {record["split"]: record for record in issue["records"]}
        self.assertEqual(members["train"]["text"], train_text)
        self.assertEqual(members["validation"]["text"], validation_text)
        self.assertEqual(members["train"]["record_number"], 1)
        self.assertEqual(members["validation"]["record_number"], 1)
        values = {member["split"]: member["value"]
                  for member in issue["label_conflicts"]["dish_taste"]}
        self.assertEqual(values, {"train": 1, "validation": -1})
        self.assertNotIn("price_level", issue["label_conflicts"])
        for path, original in original_bytes.items():
            self.assertEqual(path.read_bytes(), original)

    def test_exact_same_text_with_conflicting_labels_requires_review(self):
        path = self.write_csv("train", [
            self.row("a", "完全相同文本", dish_taste=1),
            self.row("b", "完全相同文本", dish_taste=0),
        ])

        report = self.validate({"train": path})

        self.assertEqual(report["status"], "needs_review")
        self.assertTrue(any("dish_taste" in issue.get("label_conflicts", {})
                            for issue in self.read_issues()))

    def test_mapped_stage_accepts_zero_through_three_and_rejects_negative_two(self):
        """切换阶段只切换合法标签域，不据此宣称已验证来源到输出的映射关系。"""
        path = self.write_csv("train", self.baseline_rows("mapped", labels=(0, 1, 2, 3)))
        report = self.validate({"train": path}, stage="mapped")
        self.assertEqual(report["status"], "passed")

        path = self.write_csv("train", [self.row("0", "仍然含有原始标签", label=-2)])
        report = self.validate({"train": path}, stage="mapped")
        self.assertEqual(report["status"], "failed")

    def test_length_warning_uses_an_explicit_threshold(self):
        rows = self.baseline_rows("train")
        rows[0] = self.row("0", "长" * 10000)
        path = self.write_csv("train", rows)

        without_threshold = self.validate({"train": path})
        self.assertEqual(without_threshold["status"], "passed")

        with_threshold = self.validate({"train": path}, max_text_chars=100)
        self.assertEqual(with_threshold["status"], "passed_with_warnings")
        self.assertGreater(with_threshold["issue_counts"]["WARN"], 0)

    def test_class_fraction_warning_uses_an_explicit_threshold(self):
        path = self.write_csv("train", self.baseline_rows("train"))

        report = self.validate({"train": path}, min_class_fraction=0.30)

        self.assertEqual(report["status"], "passed_with_warnings")
        self.assertGreater(report["issue_counts"]["WARN"], 0)

    def test_issue_ids_and_source_hashes_are_stable_and_existing_decisions_survive(self):
        """相同输入的重复扫描应产生稳定编号，同时原封保留既有人工决定文件。"""
        files = {
            "train": self.write_csv("train", [self.row("t1", "同一  文本", dish_taste=1)]),
            "validation": self.write_csv("validation", [
                self.row("v1", "同一 文本", dish_taste=-1)
            ]),
        }
        expected_hashes = {split: hashlib.sha256(path.read_bytes()).hexdigest()
                           for split, path in files.items()}
        self.validate(files)
        first_issues = self.read_issues()
        first_ids = [issue["issue_id"] for issue in first_issues]
        self.assertTrue(first_ids)
        self.assertEqual(len(first_ids), len(set(first_ids)))
        for issue in first_issues:
            self.assertTrue(issue["source_sha256"])
            self.assertEqual(issue["source_sha256"], {
                split: expected_hashes[split] for split in issue["source_sha256"]
            })
        review_issue = next(issue for issue in first_issues if issue["severity"] == "REVIEW")
        self.assertEqual(review_issue["source_sha256"], expected_hashes)
        decisions = self.output_dir / "review_decisions.jsonl"
        decision = {
            "issue_id": review_issue["issue_id"],
            "source_sha256": expected_hashes,
            "status": "resolved",
            "conclusion": "confirmed_duplicate",
            "reason": "已人工核对原文",
            "action": "isolate_train_record",
            "affected_records": [{"split": "train", "id": "t1"}],
            "reviewed_at": "2026-11-03T01:00:00+08:00",
        }
        existing_decision_bytes = (json.dumps(decision, ensure_ascii=False) + "\n").encode("utf-8")
        decisions.write_bytes(existing_decision_bytes)

        self.validate(files)
        second_issues = self.read_issues()

        self.assertEqual([issue["issue_id"] for issue in second_issues], first_ids)
        self.assertEqual(decisions.read_bytes(), existing_decision_bytes)
        self.assertEqual([issue["source_sha256"] for issue in second_issues],
                         [issue["source_sha256"] for issue in first_issues])

    def test_exact_subgroup_and_normalized_parent_use_distinct_group_and_record_counts(self):
        """重复子组与较大归一化组可以并存，组数相加不能当作唯一受影响样本数。"""
        train = self.write_csv("train", [
            self.row("t1", "口味 好", dish_taste=1),
            self.row("t2", "口味 好", dish_taste=0),
            self.row("t3", "口味  好", dish_taste=1),
        ])
        validation = self.write_csv("validation", [
            self.row("v1", "口味\t好", dish_taste=-1)
        ])

        report = self.validate({"train": train, "validation": validation})

        summary = report["duplicate_summary"]
        self.assertEqual(summary["exact_duplicate_groups"], 1)
        self.assertEqual(summary["normalized_only_duplicate_groups"], 1)
        self.assertEqual(summary["cross_split_exact_groups"], 0)
        self.assertEqual(summary["cross_split_normalized_only_groups"], 1)
        self.assertEqual(summary["exact_groups_with_label_conflicts"], 1)
        self.assertEqual(summary["normalized_only_groups_with_label_conflicts"], 1)
        self.assertEqual(summary["unique_records_in_duplicate_groups"], 4)
        self.assertEqual(summary["unique_records_in_cross_split_groups"], 4)
        self.assertNotIn("groups_with_label_conflicts", summary)
        grouped_issues = [issue for issue in self.read_issues()
                          if "comparison" in issue and "dish_taste" in issue.get("label_conflicts", {})]
        self.assertEqual(len(grouped_issues), 2)
        self.assertEqual({issue["comparison"]: len(issue["records"]) for issue in grouped_issues},
                         {"exact": 2, "normalized": 4})

    def test_output_inside_raw_is_rejected_without_touching_input(self):
        """危险的输出位置必须在写入前拒绝，并以字节比较确认输入未受影响。"""
        path = self.write_csv("train", self.baseline_rows("train"))
        original = path.read_bytes()

        with self.assertRaises(ValueError):
            validate_dataset({"train": path}, self.raw_dir / "reports")

        self.assertEqual(path.read_bytes(), original)
        self.assertFalse((self.raw_dir / "reports").exists())


if __name__ == "__main__":
    unittest.main()
