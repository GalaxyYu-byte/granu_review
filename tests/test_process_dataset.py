"""验证复核方案落实时的数据完整性、状态拦截和输出保护。

全部输入、报告及产物都在每个测试独立的临时目录内构建，不依赖仓库真实数据。
测试既检查合法方案的完整结果，也验证错误状态或不一致凭据不会发布任何产物。
"""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.process_dataset import _resolve_record_outcomes, process_dataset
from scripts.dataset_io import read_jsonl
from scripts.validate_dataset import ASPECT_COLUMNS, EXPECTED_COLUMNS, file_sha256, validate_dataset


class ProcessDatasetTests(unittest.TestCase):
    """构造同时包含重复冲突、跨集合重叠、损坏文本和可删除控制符的最小批次。"""

    def setUp(self):
        """生成临时 CSV 与摘要绑定的复核凭据，为每项测试提供独立输入。"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.raw_dir = self.root / "raw"
        self.raw_dir.mkdir()
        self.raw_reports = self.root / "reports" / "raw"
        self.processed_dir = self.root / "processed"
        self.quarantine_dir = self.root / "quarantine"
        self.processed_reports = self.root / "reports" / "processed"
        self.report_path = self.raw_reports / "validation_report.json"
        self.decisions_path = self.raw_reports / "review_decisions.jsonl"
        self.summary_path = self.raw_reports / "review_summary.json"
        self.train_path = self.raw_dir / "sentiment_analysis_trainingset.csv"
        self.validation_path = self.raw_dir / "sentiment_analysis_validationset.csv"
        self.paths = {"train": self.train_path, "validation": self.validation_path}
        # 两个集合都覆盖全部合法标签，避免“缺失类别”告警干扰目标行为。
        # 下方再追加需要隔离或精确清洗的样本，同时保留正常换行、标点和表情。
        self.rows = {
            split: [self.row(str(i * 10 + 1), f"{split} 基线评论 {i}", label)
                    for i, label in enumerate((-2, -1, 0, 1))]
            for split in self.paths
        }
        self.rows["train"].extend([
            self.row("conflict-a", "重复  评论", 1),
            self.row("conflict-b", "重复 评论", -1),
            self.row("corrupt", "感觉不\ufffd\ufffd\ufffd实惠", -1),
            self.row("clean-train", '香辣b\x14🤗,\n/鼓掌再来"', 1),
            self.row("overlap-train", "跨集合  重叠评论", 0),
        ])
        self.rows["validation"].extend([
            self.row("overlap-validation", "跨集合 重叠评论", 1),
            self.row("clean-validation", "安静\x01/偷笑\n还来", 1),
        ])
        for split, path in self.paths.items():
            self.write_csv(path, self.rows[split])
        # 保存原始文件的完整字节，验证任何成功或失败路径都没有改写输入。
        self.raw_bytes = {path: path.read_bytes() for path in self.paths.values()}
        self.report = validate_dataset(self.paths, self.raw_reports)
        self.issues = read_jsonl(self.raw_reports / "validation_issues.jsonl")
        self.decisions = []
        quarantine = set()
        cleaned = Counter()
        removed = 0
        # 测试在这里显式构造已复核决定，不把扫描发现自动视为处理授权。
        # 每个动作只引用对应问题的成员，并记录精确候选文本摘要和预期数量。
        for issue in self.issues:
            references = [self.reference(record) for record in issue["records"]]
            decision = {
                "issue_id": issue["issue_id"],
                "rule_id": issue["rule_id"],
                "original_severity": issue["severity"],
                "source_sha256": issue["source_sha256"],
                "affected_records": references,
                "action_applied": False,
                "label_changes": [],
                "status": "reviewed_pending_application",
                "reason": "已核对该问题的完整原文和标签，按明确的隔离或控制字符删除方案执行。",
                "evidence_quotes": [issue["records"][0]["text"]],
            }
            if issue["severity"] == "REVIEW":
                train = [ref for ref in references if ref["split"] == "train"]
                validation = [ref for ref in references if ref["split"] == "validation"]
                decision["action"] = {
                    "type": "quarantine_training_overlap" if validation else "quarantine_training_conflict_cluster",
                    "quarantine": train,
                    "preserve_fixed_validation": validation,
                }
                quarantine.update((ref["split"], ref["record_number"]) for ref in train)
            elif issue["records"][0]["id"] == "corrupt":
                decision["action"] = {"type": "quarantine_corrupted_training_record", "quarantine": references}
                quarantine.update((ref["split"], ref["record_number"]) for ref in references)
            else:
                record = issue["records"][0]
                code_points = sorted(issue["characters"])
                characters = {chr(int(value[2:], 16)) for value in code_points}
                candidate = "".join(char for char in record["text"] if char not in characters)
                decision["action"] = {
                    "type": "remove_reviewed_control_characters",
                    "target": references[0], "remove_code_points": code_points,
                    "replacement": "", "original_length": len(record["text"]),
                    "candidate_length": len(candidate),
                    "candidate_text_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
                    "preserve_labels": True,
                }
                decision["characters"] = issue["characters"]
                decision["character_count"] = sum(issue["characters"].values())
                cleaned[record["split"]] += 1
                removed += decision["character_count"]
            self.decisions.append(decision)
        self.quarantined = quarantine
        self.summary = {
            "review_complete": True,
            "source_report_sha256": file_sha256(self.report_path),
            "source_issues_sha256": self.report["artifacts"]["issues_sha256"],
            "source_sha256": {split: info["sha256"] for split, info in self.report["files"].items()},
            "artifacts": {},
            "quarantine_count": len(quarantine),
            "proposed_cleaned_record_counts": dict(cleaned),
            "proposed_removed_control_characters": removed,
            "dry_run": {
                "actual_before_counts": {split: len(rows) for split, rows in self.rows.items()},
                "expected_after_counts": {
                    split: len(rows) - sum(key[0] == split for key in quarantine)
                    for split, rows in self.rows.items()
                },
            },
        }
        self.save_decisions_and_summary()

    @staticmethod
    def row(sample_id, text, label):
        """按规范列名构造一条测试评论，全部方面使用给定的原始标签。"""
        return {"id": sample_id, "content": text, **{aspect: str(label) for aspect in ASPECT_COLUMNS}}

    @staticmethod
    def reference(record):
        """只保留定位成员所需的集合、ID 和原始记录序号，模拟正式决定格式。"""
        return {key: record[key] for key in ("split", "id", "record_number")}

    @staticmethod
    def write_csv(path, rows):
        """使用标准 CSV 写入器生成临时样本，保留多行文本和引号的合法转义。"""
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EXPECTED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read_csv(path):
        """重新解析实际 CSV，返回表头和记录，避免按物理文件行比较多行评论。"""
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def save_decisions_and_summary(self):
        """保存当前测试决定并刷新其摘要，让负例真正验证业务约束而非旧摘要。"""
        self.decisions_path.write_text(
            "".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in self.decisions),
            encoding="utf-8",
        )
        self.summary["artifacts"]["decisions_sha256"] = file_sha256(self.decisions_path)
        self.summary_path.write_text(json.dumps(self.summary, ensure_ascii=False), encoding="utf-8")

    def process(self, **overrides):
        """默认只使用本测试的临时目录；个别用例可覆盖参数验证目录限制。"""
        parameters = {
            "report_path": self.report_path, "decisions_path": self.decisions_path,
            "review_summary_path": self.summary_path,
            "output_dir": self.processed_dir, "quarantine_dir": self.quarantine_dir,
            "reports_dir": self.processed_reports,
        }
        parameters.update(overrides)
        return process_dataset(**parameters)

    def assert_raw_unchanged(self):
        """将原始 CSV 与测试开始时的完整字节比较，证明输入没有被改写。"""
        for path, original in self.raw_bytes.items():
            self.assertEqual(path.read_bytes(), original)

    def test_applies_only_reviewed_edits_and_preserves_split_ids_labels_order_and_quarantine(self):
        """合法方案只执行批准的隔离和字符删除，记录身份、标签及历史凭据均保留。"""
        original_report = self.report_path.read_bytes()
        original_decisions = self.decisions_path.read_bytes()
        result = self.process()

        self.assertEqual(result["processed_counts"], {"train": 5, "validation": 6})
        self.assertEqual(result["quarantine_count"], 4)
        self.assertEqual(result["cleaned_record_counts"], {"train": 1, "validation": 1})
        self.assertEqual(result["removed_control_characters"], 2)
        self.assertEqual(result["post_validation_status"], "passed")
        self.assertTrue(result["raw_data_unchanged"])
        self.assertTrue(result["labels_unchanged"])
        self.assertTrue(result["ids_preserved"])
        self.assertFalse(result["mapping_applied"])
        for split, raw_path in self.paths.items():
            header, actual = self.read_csv(self.processed_dir / raw_path.name)
            self.assertEqual(header, list(EXPECTED_COLUMNS))
            expected = []
            for number, original in enumerate(self.rows[split], start=1):
                if (split, number) in self.quarantined:
                    continue
                row = original.copy()
                if row["id"] == "clean-train":
                    row["content"] = row["content"].replace("\x14", "")
                if row["id"] == "clean-validation":
                    row["content"] = row["content"].replace("\x01", "")
                expected.append(row)
            self.assertEqual(actual, expected)
        _, quarantine = self.read_csv(self.quarantine_dir / "sentiment_analysis_trainingset_quarantine.csv")
        self.assertEqual(quarantine, [row for number, row in enumerate(self.rows["train"], start=1)
                                     if ("train", number) in self.quarantined])
        changes = [json.loads(line) for line in
                   (self.processed_reports / "processing_changes.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(changes), 2)
        self.assertEqual(json.loads((self.processed_reports / "processing_report.json").read_text(encoding="utf-8")), result)
        self.assertEqual(self.report_path.read_bytes(), original_report)
        self.assertEqual(self.decisions_path.read_bytes(), original_decisions)
        self.assert_raw_unchanged()

    def test_changed_source_hash_is_rejected_before_publishing_outputs(self):
        """原始文件即使只增加一个换行，也必须重新校验，不能沿用旧复核决定。"""
        self.train_path.write_bytes(self.train_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.train_path.name).exists())

    def test_incomplete_review_never_creates_outputs(self):
        """只有布尔 True 表示复核完成，其他真值或缺失状态都不能创建输出目录。"""
        for state in (False, None, "true", 1):
            with self.subTest(review_complete=state):
                self.summary["review_complete"] = state
                self.save_decisions_and_summary()
                with self.assertRaisesRegex(ValueError, "复核尚未完成"):
                    self.process()
                for path in (self.processed_dir, self.quarantine_dir, self.processed_reports):
                    self.assertFalse(path.exists())
        self.assert_raw_unchanged()

    def test_empty_evidence_is_rejected_even_after_refreshing_decision_digest(self):
        """手改决定并同步刷新摘要，也不能把空理由或空引文变成有效复核授权。"""
        decision = self.decisions[0]
        original = {field: decision[field] for field in ("reason", "evidence_quotes")}
        for field, value in (("reason", ""), ("reason", " \n\t"), ("reason", None),
                             ("evidence_quotes", [""]), ("evidence_quotes", [" \t"]),
                             ("evidence_quotes", [None]), ("evidence_quotes", "任意字符串")):
            with self.subTest(field=field, value=value):
                decision.update(original)
                decision[field] = value
                self.save_decisions_and_summary()
                with self.assertRaises(ValueError):
                    self.process()
                self.assertFalse(self.processed_dir.exists())
        self.assert_raw_unchanged()

    def test_partial_retention_override_records_each_member_and_preserves_order_independence(self):
        """保留组仅一员被清洗时，按成员记录部分覆盖，并且决定排列不影响最终结论。"""
        first = {"split": "train", "id": "a", "record_number": 1}
        second = {"split": "train", "id": "b", "record_number": 2}
        decisions = [
            {"issue_id": "keep", "action": {"type": "retain_reviewed_records"},
             "affected_records": [first, second]},
            {"issue_id": "clean", "action": {"type": "remove_reviewed_control_characters"},
             "affected_records": [first]},
        ]
        records = {("train", 1): first, ("train", 2): second}
        clean = {("train", 1): {"issue_id": "clean"}}
        execution, outcomes = _resolve_record_outcomes(decisions, records, {}, clean)
        self.assertEqual(execution["keep"]["application_status"], "partially_overridden")
        self.assertTrue(execution["keep"]["action_applied"])
        self.assertEqual(execution["keep"]["resolution"], "partially_superseded_by_other_reviewed_actions")
        self.assertFalse(execution["keep"]["record_results"][0]["decision_applied"])
        self.assertEqual(execution["keep"]["record_results"][0]["overridden_by_issue_ids"], ["clean"])
        self.assertTrue(execution["keep"]["record_results"][1]["decision_applied"])
        self.assertEqual(outcomes[0]["effective_issue_ids"], ["clean"])
        self.assertEqual(outcomes[1]["effective_issue_ids"], ["keep"])
        reordered_execution, reordered_outcomes = _resolve_record_outcomes(decisions[::-1], records, {}, clean)
        self.assertEqual(reordered_execution, execution)
        self.assertEqual(reordered_outcomes, outcomes)

    def test_each_decision_must_have_reviewed_unapplied_status(self):
        """逐项检查待复核、否决、已应用及非布尔应用状态，均应在写入前拒绝。"""
        decision = self.decisions[0]
        cases = [
            ("status", "pending"), ("status", "rejected"), ("status", "applied"),
            ("status", None), ("action_applied", True), ("action_applied", None),
            ("action_applied", 0),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                original = decision[field]
                decision[field] = value
                self.save_decisions_and_summary()
                with self.assertRaises(ValueError):
                    self.process()
                for path in (self.processed_dir, self.quarantine_dir, self.processed_reports):
                    self.assertFalse(path.exists())
                decision[field] = original
        self.assert_raw_unchanged()

    def test_duplicate_affected_reference_is_rejected(self):
        """同一成员重复出现不能被字典去重掩盖，决定必须准确引用原问题成员。"""
        self.decisions[0]["affected_records"].append(self.decisions[0]["affected_records"][0].copy())
        self.save_decisions_and_summary()
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse(self.processed_dir.exists())

    def test_wrong_quarantine_action_type_is_rejected(self):
        """训练集内部冲突不能冒充跨集合重叠，动作类型也必须与问题规则相符。"""
        decision = next(item for item in self.decisions
                        if item["action"]["type"] == "quarantine_training_conflict_cluster")
        decision["action"]["type"] = "quarantine_training_overlap"
        self.save_decisions_and_summary()
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse(self.processed_dir.exists())

    def test_training_conflict_cannot_be_hidden_to_authorize_retention(self):
        """同步刷新凭据摘要后，清空冲突明细仍不能让冲突训练样本获得保留批准。"""
        decision = next(item for item in self.decisions
                        if item["action"]["type"] == "quarantine_training_conflict_cluster")
        issue = next(item for item in self.issues if item["issue_id"] == decision["issue_id"])
        issue["label_conflicts"] = {}
        decision["action"] = {"type": "retain_reviewed_records"}
        issues_path = self.raw_reports / "validation_issues.jsonl"
        issues_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.issues),
                               encoding="utf-8")
        self.report["artifacts"]["issues_sha256"] = file_sha256(issues_path)
        self.report_path.write_text(json.dumps(self.report, ensure_ascii=False), encoding="utf-8")
        self.summary["source_issues_sha256"] = file_sha256(issues_path)
        self.summary["source_report_sha256"] = file_sha256(self.report_path)
        self.save_decisions_and_summary()
        with self.assertRaisesRegex(ValueError, "标签冲突明细"):
            self.process()
        self.assertFalse(self.processed_dir.exists())

    def test_non_control_character_deletion_is_rejected_even_with_updated_decision_digest(self):
        """删除普通字母的候选即使摘要正确，也不属于已批准的控制字符删除范围。"""
        decision = next(item for item in self.decisions
                        if item["action"].get("target", {}).get("id") == "clean-train")
        original = next(row["content"] for row in self.rows["train"] if row["id"] == "clean-train")
        candidate = original.replace("b", "")
        decision["action"].update({
            "remove_code_points": ["U+0062"], "candidate_length": len(candidate),
            "candidate_text_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
        })
        self.save_decisions_and_summary()

        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.train_path.name).exists())
        self.assert_raw_unchanged()

    def test_validation_record_cannot_be_quarantined(self):
        """错误决定若要求隔离固定验证样本，应拒绝整批处理并保留原始输入。"""
        decision = next(item for item in self.decisions
                        if item["action"].get("target", {}).get("id") == "clean-validation")
        decision["action"] = {"type": "quarantine_corrupted_training_record",
                              "quarantine": decision["affected_records"]}
        self.save_decisions_and_summary()
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.validation_path.name).exists())
        self.assert_raw_unchanged()

    def test_candidate_text_digest_mismatch_is_rejected(self):
        """清洗候选文本必须与决定内的摘要精确对应，不能只核对字符类型。"""
        decision = next(item for item in self.decisions if item["action"]["type"] == "remove_reviewed_control_characters")
        decision["action"]["candidate_text_sha256"] = "0" * 64
        self.save_decisions_and_summary()
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.train_path.name).exists())
        self.assert_raw_unchanged()

    def test_incomplete_review_coverage_is_rejected(self):
        """少一条决定就意味着存在未复核问题，不能发布部分完成的处理批次。"""
        self.decisions.pop()
        self.save_decisions_and_summary()
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.train_path.name).exists())
        self.assert_raw_unchanged()

    def test_expected_record_counts_must_reconcile_before_publication(self):
        """已复核预演与实际计划数量不同，即使其他凭据有效也必须拒绝发布。"""
        self.summary["dry_run"]["expected_after_counts"]["train"] += 1
        self.save_decisions_and_summary()
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.train_path.name).exists())
        self.assert_raw_unchanged()

    def test_existing_processed_file_is_never_overwritten(self):
        """已有目标文件必须逐字节保留，处理程序不能覆盖旧产物以完成本次运行。"""
        self.processed_dir.mkdir()
        sentinel = self.processed_dir / self.train_path.name
        sentinel.write_bytes(b"existing dataset must survive")
        with self.assertRaises(ValueError):
            self.process()
        self.assertEqual(sentinel.read_bytes(), b"existing dataset must survive")
        self.assert_raw_unchanged()

    def test_outputs_inside_raw_or_overlapping_each_other_are_rejected(self):
        """输出不得位于原始数据目录，也不得相同或互相包含，防止误写和清理越界。"""
        cases = [
            {"output_dir": self.raw_dir / "processed"},
            {"quarantine_dir": self.raw_dir / "quarantine"},
            {"reports_dir": self.raw_dir / "reports"},
            {"quarantine_dir": self.processed_dir / "quarantine"},
            {"reports_dir": self.processed_dir},
        ]
        for override in cases:
            with self.subTest(paths=override):
                with self.assertRaises(ValueError):
                    self.process(**override)
        self.assert_raw_unchanged()


if __name__ == "__main__":
    unittest.main()
