"""Adversarial checks for an independent audit of processed CSVs."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.audit_processed_dataset import audit_processed_dataset
from scripts.label_config import ASPECT_COLUMNS
from scripts.validate_dataset import file_sha256
import test_process_dataset as processing_fixtures


class AuditProcessedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        data_root = self.root / "data"
        data_root.mkdir()

        # Reuse the existing reviewed-plan fixture in the production layout.
        # Its temporary directory cleanup belongs to the outer fixture here.
        self.fixture = processing_fixtures.ProcessDatasetTests(
            "test_applies_only_reviewed_edits_and_preserves_split_ids_labels_order_and_quarantine"
        )
        self.addCleanup(self.fixture.doCleanups)
        fixture_directory = SimpleNamespace(name=str(data_root), cleanup=lambda: None)
        with patch.object(processing_fixtures.tempfile, "TemporaryDirectory", return_value=fixture_directory):
            self.fixture.setUp()
        self.processing = self.fixture.process()
        self.processing_path = self.fixture.processed_reports / "processing_report.json"

    def rewrite_artifact_and_refresh_digest(self, artifact_name, edit_rows):
        artifact = self.processing["outputs"][artifact_name]
        path = Path(artifact["path"])
        _, rows = self.fixture.read_csv(path)
        edit_rows(rows)
        self.fixture.write_csv(path, rows)
        # Deliberately bypass the existing artifact-hash protection. The audit
        # must still reject unauthorized content using its independent read.
        artifact["sha256"] = file_sha256(path)
        self.processing_path.write_text(json.dumps(self.processing), encoding="utf-8")

    def test_real_processing_result_passes_without_writing_any_inputs(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}

        report, edits, disputes = audit_processed_dataset(self.root)

        self.assertTrue(report["mechanical_checks_passed"])
        self.assertEqual(report["retained_counts"], self.processing["processed_counts"])
        self.assertEqual(report["raw_counts"], {
            split: len(rows) for split, rows in self.fixture.rows.items()
        })
        self.assertEqual(report["quarantined_records_preserved_exactly"], len(self.fixture.quarantined))
        self.assertEqual(report["edited_text_records"], len(edits))
        self.assertEqual(report["removed_character_count"], self.processing["removed_control_characters"])
        self.assertEqual(report["label_cells_compared"],
                         sum(self.processing["processed_counts"].values()) * len(ASPECT_COLUMNS))
        self.assertTrue(all(record["text_unchanged"] for record in disputes))
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)

    def test_meaningful_character_deletion_is_rejected_after_refreshing_output_digest(self):
        def remove_meaningful_character(rows):
            self.assertIn("评", rows[0]["content"])
            rows[0]["content"] = rows[0]["content"].replace("评", "", 1)

        self.rewrite_artifact_and_refresh_digest("train", remove_meaningful_character)

        with self.assertRaisesRegex(ValueError, "计划外文本修改"):
            audit_processed_dataset(self.root)

    def test_valid_label_tampering_is_rejected_after_refreshing_output_digest(self):
        def change_label(rows):
            self.assertNotEqual(rows[0][ASPECT_COLUMNS[0]], "1")
            rows[0][ASPECT_COLUMNS[0]] = "1"

        self.rewrite_artifact_and_refresh_digest("validation", change_label)

        with self.assertRaisesRegex(ValueError, "标签被修改"):
            audit_processed_dataset(self.root)

    def test_quarantined_text_tampering_is_rejected_after_refreshing_output_digest(self):
        def change_quarantined_text(rows):
            rows[0]["content"] += "篡改"

        self.rewrite_artifact_and_refresh_digest("quarantine", change_quarantined_text)

        with self.assertRaisesRegex(ValueError, "隔离副本未完整保留原文"):
            audit_processed_dataset(self.root)

    def test_missing_quarantine_row_is_rejected_after_refreshing_output_digest(self):
        self.rewrite_artifact_and_refresh_digest("quarantine", lambda rows: rows.pop())

        with self.assertRaisesRegex(ValueError, "隔离副本未完整保留原文"):
            audit_processed_dataset(self.root)

    def test_missing_quarantine_file_is_rejected(self):
        Path(self.processing["outputs"]["quarantine"]["path"]).unlink()

        with self.assertRaises(FileNotFoundError):
            audit_processed_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
