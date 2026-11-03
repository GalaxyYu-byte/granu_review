"""Data-integrity tests for applying reviewed cleaning and quarantine decisions."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.process_dataset import process_dataset
from scripts.validate_dataset import ASPECT_COLUMNS, EXPECTED_COLUMNS, file_sha256, validate_dataset


class ProcessDatasetTests(unittest.TestCase):
    def setUp(self):
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
        self.raw_bytes = {path: path.read_bytes() for path in self.paths.values()}
        self.report = validate_dataset(self.paths, self.raw_reports)
        self.issues = [json.loads(line) for line in
                       (self.raw_reports / "validation_issues.jsonl").read_text(encoding="utf-8").splitlines()]
        self.decisions = []
        quarantine = set()
        cleaned = Counter()
        removed = 0
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
        return {"id": sample_id, "content": text, **{aspect: str(label) for aspect in ASPECT_COLUMNS}}

    @staticmethod
    def reference(record):
        return {key: record[key] for key in ("split", "id", "record_number")}

    @staticmethod
    def write_csv(path, rows):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=EXPECTED_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def read_csv(path):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            return reader.fieldnames, list(reader)

    def save_decisions_and_summary(self):
        self.decisions_path.write_text(
            "".join(json.dumps(decision, ensure_ascii=False) + "\n" for decision in self.decisions),
            encoding="utf-8",
        )
        self.summary["artifacts"]["decisions_sha256"] = file_sha256(self.decisions_path)
        self.summary_path.write_text(json.dumps(self.summary, ensure_ascii=False), encoding="utf-8")

    def process(self, **overrides):
        parameters = {
            "report_path": self.report_path, "decisions_path": self.decisions_path,
            "review_summary_path": self.summary_path,
            "output_dir": self.processed_dir, "quarantine_dir": self.quarantine_dir,
            "reports_dir": self.processed_reports,
        }
        parameters.update(overrides)
        return process_dataset(**parameters)

    def assert_raw_unchanged(self):
        for path, original in self.raw_bytes.items():
            self.assertEqual(path.read_bytes(), original)

    def test_applies_only_reviewed_edits_and_preserves_split_ids_labels_order_and_quarantine(self):
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
        self.train_path.write_bytes(self.train_path.read_bytes() + b"\n")
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.train_path.name).exists())

    def test_non_control_character_deletion_is_rejected_even_with_updated_decision_digest(self):
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
        decision = next(item for item in self.decisions if item["action"]["type"] == "remove_reviewed_control_characters")
        decision["action"]["candidate_text_sha256"] = "0" * 64
        self.save_decisions_and_summary()
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.train_path.name).exists())
        self.assert_raw_unchanged()

    def test_incomplete_review_coverage_is_rejected(self):
        self.decisions.pop()
        self.save_decisions_and_summary()
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.train_path.name).exists())
        self.assert_raw_unchanged()

    def test_expected_record_counts_must_reconcile_before_publication(self):
        self.summary["dry_run"]["expected_after_counts"]["train"] += 1
        self.save_decisions_and_summary()
        with self.assertRaises(ValueError):
            self.process()
        self.assertFalse((self.processed_dir / self.train_path.name).exists())
        self.assert_raw_unchanged()

    def test_existing_processed_file_is_never_overwritten(self):
        self.processed_dir.mkdir()
        sentinel = self.processed_dir / self.train_path.name
        sentinel.write_bytes(b"existing dataset must survive")
        with self.assertRaises(ValueError):
            self.process()
        self.assertEqual(sentinel.read_bytes(), b"existing dataset must survive")
        self.assert_raw_unchanged()

    def test_outputs_inside_raw_or_overlapping_each_other_are_rejected(self):
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
