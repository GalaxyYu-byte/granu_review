"""串联校验、逐条复核、处理和独立审计，验证各阶段凭据及保留策略可以衔接。

全部文件都位于临时项目中。产物路径一律通过对应项目根目录解析，防止
相对凭据误指向当前工作目录中的真实数据。
"""

import csv
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from scripts.audit_processed_dataset import FILENAMES, audit_processed_dataset
from scripts.dataset_io import read_jsonl, receipt_path, resolve_receipt_path
from scripts.label_config import ASPECT_COLUMNS, EXPECTED_COLUMNS
from scripts.process_dataset import process_dataset
from scripts.review_dataset_issues import NOTE_NAMES, build_review
from scripts.validate_dataset import file_sha256, validate_dataset


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self):
        """为两个集合各建四类基线样本，让新增问题成为每项测试的唯一变量。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = {
            split: [self.row(f"{split}-{number}", f"{split} 基线评论 {number}", label)
                    for number, label in enumerate((-2, -1, 0, 1))]
            for split in FILENAMES
        }

    @staticmethod
    def row(sample_id, text, label):
        return {"id": sample_id, "content": text,
                **{aspect: str(label) for aspect in ASPECT_COLUMNS}}

    def prepare_review(self):
        """生成完整的临时原始报告、证据注释和复核快照，不跳过阶段之间的摘要绑定。"""
        raw_dir = self.root / "data/raw"
        raw_dir.mkdir(parents=True)
        raw_reports = self.root / "data/reports/raw"
        paths = {split: raw_dir / filename for split, filename in FILENAMES.items()}
        for split, path in paths.items():
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=EXPECTED_COLUMNS)
                writer.writeheader()
                writer.writerows(self.rows[split])
        report = validate_dataset(paths, raw_reports, root=self.root)
        issues = read_jsonl(resolve_receipt_path(report["artifacts"]["issues"], root=self.root))
        notes = []
        for issue in issues:
            note = {"issue_id": issue["issue_id"], "reason": "逐条核对原文及标签后采用明确的重复或控制字符策略。",
                    "evidence_quotes": [issue["records"][0]["text"]],
                    "label_resolution": "保留原验证标签，不以训练标签替换基准。"}
            if issue["rule_id"] == "suspicious_characters":
                note.update(recommended_action="remove_reviewed_control_characters",
                            confidence="high", cause_hypothesis="可定位的异常控制字符")
            notes.append(note)
        # 测试仍提供生产流程要求的三份注释文件；集中放入第一份仅简化夹具组织。
        for index, name in enumerate(NOTE_NAMES):
            (raw_reports / name).write_text(json.dumps(notes if index == 0 else [], ensure_ascii=False),
                                           encoding="utf-8")
        decisions, summary = build_review(raw_reports / "validation_report.json", root=self.root)
        decisions_path = raw_reports / "review_decisions.jsonl"
        decisions_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in decisions),
                                  encoding="utf-8")
        # 汇总必须绑定实际写出的决定字节，后续处理才有可验证的输入凭据。
        summary["artifacts"] = {"decisions": receipt_path(decisions_path, root=self.root),
                                "decisions_sha256": file_sha256(decisions_path)}
        (raw_reports / "review_summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
        return decisions, summary

    def process_and_audit(self, root=None):
        """在同一项目执行真实处理与独立审计，并逐文件确认输入和复核快照未被改动。"""
        root = root or self.root
        raw_reports = root / "data/reports/raw"
        before = {path: path.read_bytes() for path in (root / "data/raw").iterdir()}
        before.update({path: path.read_bytes() for path in raw_reports.iterdir()})
        processing = process_dataset(raw_reports / "validation_report.json", raw_reports / "review_decisions.jsonl",
                                     raw_reports / "review_summary.json", root / "data/processed",
                                     root / "data/quarantine", root / "data/reports/processed", root=root)
        audit, edits, disputes = audit_processed_dataset(root)
        self.assertTrue(audit["mechanical_checks_passed"])
        self.assertEqual(audit["retained_counts"], processing["processed_counts"])
        for path, expected in before.items():
            self.assertEqual(path.read_bytes(), expected)
        return processing, audit, edits, disputes

    def artifact_json(self, processing, name, root=None):
        """按产物所属项目解析凭据路径，避免复制项目后仍读取原目录。"""
        path = resolve_receipt_path(processing["outputs"][name]["path"], root=root or self.root)
        return json.loads(path.read_text(encoding="utf-8"))

    def artifact_rows(self, processing, split, root=None):
        """重新解析实际 CSV 产物，供文本、标签和记录顺序的独立断言使用。"""
        path = resolve_receipt_path(processing["outputs"][split]["path"], root=root or self.root)
        with path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def assert_review_resolutions(self, processing, decisions, root=None):
        """扫描发现仍保留为事实，每项放行都必须完整关联已复核的保留决定。

        因已知重复而继续保留的记录不应再次阻塞发布，但也不能通过清零
        问题数量掩盖发现；因此同时检查扫描状态、待复核数和放行凭据。
        """
        post = self.artifact_json(processing, "validation_report", root=root)
        issues = read_jsonl(resolve_receipt_path(post["artifacts"]["issues"], root=root or self.root))
        review_ids = {item["issue_id"] for item in issues if item["severity"] == "REVIEW"}
        resolutions = post["review_resolutions"]
        self.assertEqual({item["issue_id"] for item in resolutions}, review_ids)
        self.assertEqual(len(resolutions), len(review_ids))
        allowed = {item["issue_id"] for item in decisions if item["action"]["type"] == "retain_reviewed_records"}
        self.assertEqual({issue_id for item in resolutions for issue_id in item["source_issue_ids"]}, allowed)
        self.assertEqual(post["status"], "passed_with_warnings")
        self.assertEqual(post["scan_status"], "needs_review")
        self.assertEqual(post["pending_reviews"], 0)
        self.assertTrue(post["validation_gate_passed"])

    def test_same_label_normalized_duplicates_are_retained_with_explicit_post_review_resolutions(self):
        """同标签重复经明确复核后保持全部原文和标签，二次校验记录对应的保留依据。"""
        self.rows["train"].extend([self.row("a", "价格  合理", 1), self.row("b", "价格 合理", 1)])
        decisions, _ = self.prepare_review()
        processing, audit, edits, disputes = self.process_and_audit()
        self.assertEqual(self.artifact_rows(processing, "train"), self.rows["train"])
        self.assertEqual(processing["quarantine_count"], 0)
        self.assertEqual(audit["retained_counts"], {"train": 6, "validation": 4})
        self.assertEqual(edits, [])
        self.assertEqual(disputes, [])
        self.assert_review_resolutions(processing, decisions)

    def test_unicode_jsonl_separators_survive_full_pipeline_and_only_reviewed_control_is_removed(self):
        """同一记录可以同时涉及重复和字符告警；只删除已批准控制符，保留其他分隔字符。"""
        original = "价格\u0085\u2028\u2029合理"
        self.rows["train"].extend([self.row("a", original, 1), self.row("b", "价格 合理", 1)])
        decisions, _ = self.prepare_review()
        self.assertEqual({item["action"]["type"] for item in decisions},
                         {"retain_reviewed_records", "remove_reviewed_control_characters"})
        processing, audit, edits, disputes = self.process_and_audit()
        result = next(row for row in self.artifact_rows(processing, "train") if row["id"] == "a")
        self.assertEqual(result["content"], original.replace("\u0085", ""))
        self.assertEqual(processing["removed_control_characters"], 1)
        self.assertEqual(audit["removed_character_count"], 1)
        self.assertEqual(edits[0]["removed_characters"], {"U+0085": 1})
        self.assertEqual(disputes, [])
        self.assert_review_resolutions(processing, decisions)

    def test_validation_conflicts_are_retained_with_original_labels_and_recorded_as_disputed(self):
        """验证集标签冲突应保留为显式争议，不能因机械校验通过而自动改标或关闭争议。"""
        self.rows["validation"].extend([self.row("a", "价格  合理", 1), self.row("b", "价格 合理", -1)])
        decisions, summary = self.prepare_review()
        processing, audit, _, disputes = self.process_and_audit()
        self.assertEqual(self.artifact_rows(processing, "validation"), self.rows["validation"])
        self.assertEqual(processing["known_validation_label_disputes"], summary["known_validation_label_disputes"])
        self.assertEqual({record["id"] for record in disputes}, {"a", "b"})
        self.assertEqual(audit["changed_label_cells"], 0)
        self.assert_review_resolutions(processing, decisions)

    def test_cross_split_same_label_overlap_is_excluded_without_inventing_a_label_dispute(self):
        """跨集合重叠仍需隔离训练侧以保护评估独立性；同标签不能被误报为标签争议。"""
        self.rows["train"].append(self.row("overlap-train", "跨集合  评论", 1))
        self.rows["validation"].append(self.row("overlap-validation", "跨集合 评论", 1))
        decisions, summary = self.prepare_review()
        processing, audit, _, disputes = self.process_and_audit()
        self.assertEqual(decisions[0]["action"]["type"], "quarantine_training_overlap")
        self.assertEqual(summary["known_validation_label_disputes"], [])
        self.assertEqual(processing["known_validation_label_disputes"], [])
        self.assertEqual(disputes, [])
        self.assertEqual(audit["retained_counts"], {"train": 4, "validation": 5})
        self.assertEqual(processing["quarantine_count"], 1)

    def test_raw_and_review_files_can_move_before_processing_and_all_new_receipts_are_relative(self):
        """复制原始数据和复核快照后直接继续处理，所有新凭据均绑定副本项目。"""
        self.rows["train"].extend([self.row("a", "价格  合理", 1), self.row("b", "价格 合理", 1)])
        decisions, _ = self.prepare_review()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        copied_root = Path(temporary.name) / "copied-project"
        shutil.copytree(self.root, copied_root)
        # 故意破坏旧位置的临时输入；如果迁移后还访问旧目录，本测试必须失败。
        (self.root / "data/raw" / FILENAMES["train"]).write_bytes(b"original is no longer usable")
        processing, _, _, _ = self.process_and_audit(copied_root)
        self.assert_review_resolutions(processing, decisions, root=copied_root)
        paths = list(processing["inputs"]) + [item["path"] for item in processing["outputs"].values()]
        post = self.artifact_json(processing, "validation_report", root=copied_root)
        paths += [item["path"] for item in post["files"].values()]
        paths += [post["artifacts"]["report"], post["artifacts"]["issues"]]
        for value in paths:
            with self.subTest(path=value):
                self.assertTrue(value.startswith("data/"))
                path = resolve_receipt_path(value, root=copied_root)
                self.assertTrue(path.is_relative_to(copied_root))
                self.assertTrue(path.is_file())


if __name__ == "__main__":
    unittest.main()
