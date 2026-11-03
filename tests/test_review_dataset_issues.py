"""验证逐条复核的证据绑定、重复处理策略及复制项目后的路径兼容性。"""

import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.dataset_io import read_jsonl, resolve_receipt_path
from scripts.label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS
from scripts.review_dataset_issues import NOTE_NAMES, build_review, main
from scripts.validate_dataset import validate_dataset


class ReviewDatasetIssuesTests(unittest.TestCase):
    """使用临时数据覆盖不同重复类型，确保策略随输入证据变化而不是依赖某批固定样本。"""

    def setUp(self):
        """为两个集合准备四类基线记录，减少缺失类别告警对目标复核规则的干扰。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        (self.root / "raw").mkdir()
        self.paths = {split: self.root / "raw" / f"{split}.csv" for split in ("train", "validation")}
        self.reports = self.root / "reports"
        self.rows = {
            split: [self.row(f"{split}-{index}", f"{split} 基线评论 {index}", label)
                    for index, label in enumerate((-2, -1, 0, 1))]
            for split in self.paths
        }

    @staticmethod
    def row(sample_id, text, label):
        """生成二十方面取相同标签的完整记录，让测试只改变所关注的文本或冲突条件。"""
        return {"id": sample_id, "content": text,
                **{aspect: str(label) for aspect in ASPECT_COLUMNS}}

    def review(self):
        """真实执行校验、生成对应注释，再调用复核函数，不以手写问题明细替代规则扫描。

        注释覆盖本次全部问题并引用实际原文；字符异常只提出明确的控制符删除建议。
        此辅助函数不保存应用结果，便于断言复核阶段仍然只做只读预演。
        """
        for split, path in self.paths.items():
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=EXPECTED_COLUMNS)
                writer.writeheader()
                writer.writerows(self.rows[split])
        validate_dataset(self.paths, self.reports, root=self.root)
        issues = read_jsonl(self.reports / "validation_issues.jsonl")
        notes = []
        for issue in issues:
            note = {
                "issue_id": issue["issue_id"], "reason": "已核对原文及标签，按明确的重复策略处理。",
                "evidence_quotes": [issue["records"][0]["text"]],
                "label_resolution": "保留验证标签，冲突训练记录进入隔离。",
            }
            if issue["rule_id"] == "suspicious_characters":
                note.update(recommended_action="remove_reviewed_control_characters",
                            confidence="high", cause_hypothesis="异常控制字符")
            notes.append(note)
        for index, name in enumerate(NOTE_NAMES):
            (self.reports / name).write_text(json.dumps(notes if index == 0 else [], ensure_ascii=False),
                                            encoding="utf-8")
        return build_review(self.reports / "validation_report.json", root=self.root)

    def test_same_label_normalized_duplicates_are_retained(self):
        """仅有空格数量差异且标签相同时应保留记录，不能误报冲突或隔离训练样本。"""
        self.rows["train"].extend([self.row("a", "价格  合理", 1), self.row("b", "价格 合理", 1)])
        decisions, summary = self.review()

        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertEqual(decision["action"]["type"], "retain_reviewed_records")
        self.assertEqual(decision["label_conflicts"], {})
        self.assertEqual(decision["conflicting_aspect_count"], 0)
        self.assertIn("consistent_labels", decision["conclusion"])
        self.assertEqual(summary["quarantine_count"], 0)
        self.assertEqual(summary["dry_run"]["expected_after_counts"]["train"], 6)
        self.assertEqual(summary["known_validation_label_disputes"], [])

    def test_exact_duplicates_use_same_policy_for_matching_and_conflicting_labels(self):
        """完全重复和归一化重复共用按标签冲突分流的策略，而不是由严重程度直接决定隔离。"""
        for conflicting in (False, True):
            with self.subTest(conflicting=conflicting):
                self.rows["train"] = self.rows["train"][:4] + [
                    self.row("a", "同一条评论", 1), self.row("b", "同一条评论", -1 if conflicting else 1),
                ]
                decisions, summary = self.review()
                self.assertEqual(len(decisions), 1)
                self.assertEqual(decisions[0]["action"]["type"],
                                 "quarantine_training_conflict_cluster" if conflicting else "retain_reviewed_records")
                self.assertEqual(summary["quarantine_count"], 2 if conflicting else 0)

    def test_cross_split_same_label_overlap_still_excludes_training_side(self):
        """跨集合重叠即使标签一致也隔离训练侧，同时不得凭重叠本身捏造验证标签争议。"""
        for validation_text in ("跨集合 评论", "跨集合  评论"):
            with self.subTest(validation_text=validation_text):
                self.rows["train"] = self.rows["train"][:4] + [self.row("overlap-train", "跨集合 评论", 1)]
                self.rows["validation"] = self.rows["validation"][:4] + [self.row("overlap-validation", validation_text, 1)]
                decisions, summary = self.review()
                self.assertEqual(decisions[0]["action"]["type"], "quarantine_training_overlap")
                self.assertEqual([record["id"] for record in decisions[0]["action"]["quarantine"]], ["overlap-train"])
                self.assertEqual(summary["quarantine_count"], 1)
                self.assertEqual(summary["dry_run"]["expected_after_counts"], {"train": 4, "validation": 5})
                self.assertEqual(summary["known_validation_label_disputes"], [])

    def test_validation_conflict_is_preserved_and_counted_as_disputed(self):
        """验证集内部冲突继续保留原样本及原标签，但所有冲突成员必须进入未决争议清单。"""
        self.rows["validation"].extend([self.row("a", "价格  合理", 1), self.row("b", "价格 合理", -1)])
        decisions, summary = self.review()
        self.assertEqual(decisions[0]["action"]["type"], "retain_reviewed_records")
        self.assertEqual(summary["quarantine_count"], 0)
        self.assertEqual(summary["dry_run"]["expected_after_counts"]["validation"], 6)
        self.assertEqual({record["id"] for record in summary["known_validation_label_disputes"]}, {"a", "b"})

    def test_unicode_jsonl_separators_survive_csv_issue_review_round_trip(self):
        """在原文内放入多种特殊分隔符，确认逐行读取问题明细不会把一个对象拆成多条。"""
        text = "价格\u0085\u2028\u2029合理"
        self.rows["train"].extend([self.row("a", text, 1), self.row("b", "价格 合理", 1)])
        decisions, summary = self.review()
        duplicate = next(item for item in decisions if "duplicate" in item["rule_id"])
        self.assertIn(text, duplicate["evidence_quotes"])
        self.assertTrue(summary["review_complete"])

    def test_empty_issue_set_does_not_invent_validation_disputes(self):
        """没有问题的批次必须得到空争议清单，不能沿用某次真实数据中固定两条争议的结论。"""
        decisions, summary = self.review()
        self.assertEqual(decisions, [])
        self.assertEqual(summary["known_validation_label_disputes"], [])
        self.assertTrue(any("0 retained validation records" in line for line in summary["limitations"]))

    def test_relative_report_paths_can_be_reviewed_after_copying_project(self):
        """复制原始数据及报告后，相对路径应基于新根目录重算出相同的决定和数量预演。"""
        self.rows["train"].extend([self.row("a", "价格  合理", 1), self.row("b", "价格 合理", 1)])
        expected_decisions, expected_summary = self.review()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        copy = Path(temporary.name) / "copied-project"
        shutil.copytree(self.root, copy)
        decisions, summary = build_review(copy / "reports/validation_report.json", root=copy)
        self.assertEqual(decisions[0]["issue_id"], expected_decisions[0]["issue_id"])
        self.assertEqual(summary["dry_run"], expected_summary["dry_run"])

    def test_cli_root_selects_copied_project_default_report_and_output_directory(self):
        """只指定项目根目录也应读取副本的默认报告，并将复核结果写到副本中的默认目录。"""
        raw_dir = self.root / "data/raw"
        raw_dir.mkdir(parents=True)
        self.paths = {split: raw_dir / path.name for split, path in self.paths.items()}
        self.reports = self.root / "data/reports/raw"
        self.rows["train"].extend([self.row("a", "价格  合理", 1), self.row("b", "价格 合理", 1)])
        expected, _ = self.review()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        copy = Path(temporary.name) / "copied-project"
        shutil.copytree(self.root, copy)
        # 保留旧目录但损坏其中源文件，使测试能识别错误地继续访问旧位置的实现。
        self.paths["train"].write_bytes(b"old source must never be read")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--root", str(copy)]), 0)
        summary_path = copy / "data/reports/raw/review_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        decisions_path = resolve_receipt_path(summary["artifacts"]["decisions"], root=copy)
        self.assertEqual(decisions_path, copy / "data/reports/raw/review_decisions.jsonl")
        decisions = read_jsonl(decisions_path)
        self.assertEqual(decisions[0]["issue_id"], expected[0]["issue_id"])
        self.assertFalse((self.reports / "review_summary.json").exists())

    def test_cli_rebases_legacy_absolute_report_paths_to_copy_even_when_old_source_exists(self):
        """旧报告中的绝对路径应按报告来源根目录精确迁移，旧副本仍存在也不能误读。"""
        raw_dir = self.root / "data/raw"
        raw_dir.mkdir(parents=True)
        self.paths = {split: raw_dir / path.name for split, path in self.paths.items()}
        self.reports = self.root / "data/reports/raw"
        self.rows["train"].extend([self.row("a", "价格  合理", 1), self.row("b", "价格 合理", 1)])
        expected, _ = self.review()
        report_path = self.reports / "validation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        # 模拟旧版本凭据全部保存绝对路径的形式；文件摘要和问题内容仍保持同一批证据。
        for name in ("report", "issues"):
            report["artifacts"][name] = str(resolve_receipt_path(report["artifacts"][name], root=self.root))
        for info in report["files"].values():
            info["path"] = str(resolve_receipt_path(info["path"], root=self.root))
        report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        copy = Path(temporary.name) / "copied-project"
        shutil.copytree(self.root, copy)
        # 同时破坏旧源和旧问题明细，确认两类依赖都从新副本读取，而非只迁移其中一项。
        self.paths["train"].write_bytes(b"legacy old source must never be read")
        (self.reports / "validation_issues.jsonl").write_bytes(b"legacy old issues must never be read")
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--root", str(copy)]), 0)
        summary = json.loads((copy / "data/reports/raw/review_summary.json").read_text(encoding="utf-8"))
        decisions = read_jsonl(resolve_receipt_path(summary["artifacts"]["decisions"], root=copy))
        self.assertEqual(decisions[0]["issue_id"], expected[0]["issue_id"])
        self.assertEqual(summary["dry_run"]["actual_before_counts"], {"train": 6, "validation": 4})


if __name__ == "__main__":
    unittest.main()
