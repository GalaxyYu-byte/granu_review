"""用定向篡改验证独立审计能发现真实内容、凭据关系和复核批准中的错误。

测试会有意更新受篡改文件的外层摘要，确保审计不能仅凭摘要匹配就误判通过。
所有数据和报告均放在测试临时项目中，路径解析始终绑定该项目根目录。
"""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.audit_processed_dataset import audit_processed_dataset, main as audit_main
from scripts.dataset_io import read_jsonl, resolve_receipt_path
from scripts.label_config import ASPECT_COLUMNS
from scripts.validate_dataset import file_sha256, validate_dataset
import test_process_dataset as processing_fixtures
import test_dataset_pipeline as pipeline_fixtures


class AuditProcessedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        data_root = self.root / "data"
        data_root.mkdir()

        # 复用处理流程的已复核计划，并放入与实际项目一致的 data 目录结构。
        # 内层测试不重复清理共享目录，统一由本测试的外层临时目录负责释放。
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
        """篡改临时 CSV 并同步更新相关凭据摘要，专门检验独立内容对账。"""
        artifact = self.processing["outputs"][artifact_name]
        path = resolve_receipt_path(artifact["path"], root=self.root)
        _, rows = self.fixture.read_csv(path)
        edit_rows(rows)
        self.fixture.write_csv(path, rows)
        # 主动绕过最外层文件摘要保护；审计仍应从原始记录中发现未授权编辑。
        artifact["sha256"] = file_sha256(path)
        if artifact_name in {"train", "validation"}:
            # 同步伪造处理报告与二次校验报告，避免测试只命中较浅的摘要不符检查。
            # 后续必须继续依靠逐条正文和标签比较识别错误。
            validation_path = resolve_receipt_path(self.processing["outputs"]["validation_report"]["path"], root=self.root)
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            validation["files"][artifact_name]["sha256"] = file_sha256(path)
            validation_path.write_text(json.dumps(validation), encoding="utf-8")
            self.processing["outputs"]["validation_report"]["sha256"] = file_sha256(validation_path)
        self.processing_path.write_text(json.dumps(self.processing), encoding="utf-8")

    def refresh_input_digest(self, path):
        """仅在当前临时项目中查找输入，并刷新处理报告绑定的摘要。"""
        for name in self.processing["inputs"]:
            if resolve_receipt_path(name, root=self.root) == path:
                self.processing["inputs"][name] = file_sha256(path)
                break
        else:
            self.fail(f"fixture input missing: {path}")
        self.processing_path.write_text(json.dumps(self.processing), encoding="utf-8")

    def rewrite_jsonl_artifact(self, name, rows):
        """重写临时执行日志并更新外层摘要，保留 Unicode 字符以覆盖读取边界。"""
        artifact = self.processing["outputs"][name]
        path = resolve_receipt_path(artifact["path"], root=self.root)
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows), encoding="utf-8")
        artifact["sha256"] = file_sha256(path)
        self.processing_path.write_text(json.dumps(self.processing), encoding="utf-8")

    def refresh_review_chain(self):
        """同步更新决定、汇总和处理凭据，使后续失败必须来自业务关系检查。"""
        self.fixture.save_decisions_and_summary()
        self.refresh_input_digest(self.fixture.decisions_path)
        self.refresh_input_digest(self.fixture.summary_path)

    def refresh_source_issue_chain(self):
        """重写原始问题及所有引用摘要，模拟最外层材料都已同步更新的篡改。"""
        issues_path = self.fixture.raw_reports / "validation_issues.jsonl"
        issues_path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in self.fixture.issues), encoding="utf-8")
        report = json.loads(self.fixture.report_path.read_text(encoding="utf-8"))
        report["artifacts"]["issues_sha256"] = file_sha256(issues_path)
        report["rule_counts"] = {rule: sum(item["rule_id"] == rule for item in self.fixture.issues)
                                 for rule in {item["rule_id"] for item in self.fixture.issues}}
        self.fixture.report_path.write_text(json.dumps(report), encoding="utf-8")
        self.fixture.summary["source_issues_sha256"] = file_sha256(issues_path)
        self.fixture.summary["source_report_sha256"] = file_sha256(self.fixture.report_path)
        self.refresh_input_digest(issues_path)
        self.refresh_input_digest(self.fixture.report_path)
        self.refresh_review_chain()

    def test_real_processing_result_passes_without_writing_any_inputs(self):
        # 保存全部输入与产物字节，证明审计成功也不会顺便改写被审计材料。
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
        # 改成标签域内的另一个合法值，避免仅靠结构或合法取值检查就发现问题。
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
        resolve_receipt_path(self.processing["outputs"]["quarantine"]["path"], root=self.root).unlink()

        with self.assertRaises(FileNotFoundError):
            audit_processed_dataset(self.root)

    def test_application_contract_is_checked_even_after_digest_refresh(self):
        path = resolve_receipt_path(self.processing["outputs"]["applications"]["path"], root=self.root)
        original = read_jsonl(path)
        mutations = {
            "action_applied": False, "action_type": "unknown", "post_validation_status": "failed",
            "source_sha256": {}, "affected_records": [], "label_changes": [{"id": "changed"}],
            "resolution": "unrelated", "underlying_annotation_or_source_repair_completed": True,
            "applied_at": "yesterday",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                # 每次从同一份合法日志恢复，只改变一个字段，分别验证动作、来源及状态约束。
                rows = json.loads(json.dumps(original))
                rows[0][field] = value
                self.rewrite_jsonl_artifact("applications", rows)
                with self.assertRaisesRegex(ValueError, "应用日志"):
                    audit_processed_dataset(self.root)

    def test_incomplete_summary_is_rejected_after_refreshing_digest(self):
        self.fixture.summary["review_complete"] = False
        self.refresh_review_chain()
        with self.assertRaisesRegex(ValueError, "复核尚未完成"):
            audit_processed_dataset(self.root)

    def test_unreviewed_decision_is_rejected_after_refreshing_all_digests(self):
        self.fixture.decisions[0]["status"] = "pending"
        self.refresh_review_chain()
        with self.assertRaisesRegex(ValueError, "复核状态"):
            audit_processed_dataset(self.root)

    def test_summary_cross_digest_is_checked_after_processing_digest_refresh(self):
        self.fixture.summary["source_report_sha256"] = "0" * 64
        self.fixture.summary_path.write_text(json.dumps(self.fixture.summary), encoding="utf-8")
        self.refresh_input_digest(self.fixture.summary_path)
        with self.assertRaisesRegex(ValueError, "摘要关系"):
            audit_processed_dataset(self.root)

    def test_decision_target_must_belong_to_original_issue(self):
        decision = next(item for item in self.fixture.decisions if "quarantine" in item["action"])
        targets = decision["action"]["quarantine"]
        # 重新构造动作列表，避免夹具中的共享引用连带修改 affected_records，掩盖目标检查。
        decision["action"]["quarantine"] = [{**targets[0], "id": "unrelated"}, *targets[1:]]
        self.refresh_review_chain()
        with self.assertRaisesRegex(ValueError, "隔离目标"):
            audit_processed_dataset(self.root)

    def test_decision_rule_must_match_original_issue(self):
        self.fixture.decisions[0]["rule_id"] = "unknown"
        self.refresh_review_chain()
        with self.assertRaisesRegex(ValueError, "来源或规则"):
            audit_processed_dataset(self.root)

    def test_cleaning_candidate_digest_is_checked_against_decision(self):
        decision = next(item for item in self.fixture.decisions
                        if item["action"]["type"] == "remove_reviewed_control_characters")
        decision["action"]["candidate_text_sha256"] = "0" * 64
        self.refresh_review_chain()
        with self.assertRaisesRegex(ValueError, "清洗后长度/摘要"):
            audit_processed_dataset(self.root)

    def test_copied_project_can_be_audited_without_rewriting_receipts(self):
        self.assertTrue(all(not Path(item["path"]).is_absolute() for item in self.processing["outputs"].values()))
        self.assert_copied_project_passes()

    def assert_copied_project_passes(self):
        """保留旧目录但破坏旧数据，验证审计真正读取迁移后的副本。"""
        with tempfile.TemporaryDirectory() as directory:
            copied_root = Path(directory) / "copy"
            shutil.copytree(self.root, copied_root)
            # 旧绝对路径仍然存在；将旧数据破坏后，误读原项目就无法侥幸通过测试。
            self.fixture.train_path.write_bytes(b"original is no longer usable")
            report, _, _ = audit_processed_dataset(copied_root)
            self.assertTrue(report["mechanical_checks_passed"])
        self.assertTrue(all(not Path(name).is_absolute() for name in report["file_sha256"]))
        self.assertTrue(all(resolve_receipt_path(name, root=copied_root).is_relative_to(copied_root)
                            for name in report["file_sha256"]))

    def test_legacy_absolute_receipts_resolve_against_copied_root(self):
        # 将新凭据改造成旧版绝对路径格式，再完整复制，不在副本中修改任何凭据内容。
        # 内部摘要关系也同步更新，迁移成功不能依赖原项目恰好仍然可用。
        report_path = self.fixture.report_path
        report = json.loads(report_path.read_text(encoding="utf-8"))
        validation_path = resolve_receipt_path(self.processing["outputs"]["validation_report"]["path"], root=self.root)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        for receipt in (report, validation):
            for info in receipt["files"].values():
                info["path"] = str(resolve_receipt_path(info["path"], root=self.root))
            for name in ("issues", "report"):
                receipt["artifacts"][name] = str(resolve_receipt_path(receipt["artifacts"][name], root=self.root))
        report_path.write_text(json.dumps(report), encoding="utf-8")
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
        self.fixture.summary["source_report_sha256"] = file_sha256(report_path)
        self.fixture.save_decisions_and_summary()
        self.processing["inputs"] = {
            str(resolve_receipt_path(name, root=self.root)): file_sha256(resolve_receipt_path(name, root=self.root))
            for name in self.processing["inputs"]
        }
        for artifact in self.processing["outputs"].values():
            path = resolve_receipt_path(artifact["path"], root=self.root)
            artifact.update(path=str(path), sha256=file_sha256(path))
        if "input_roles" in self.processing:
            self.processing["input_roles"] = {name: str(resolve_receipt_path(value, root=self.root))
                                               for name, value in self.processing["input_roles"].items()}
        self.processing_path.write_text(json.dumps(self.processing), encoding="utf-8")
        self.assert_copied_project_passes()

    def test_unicode_line_separator_in_application_jsonl_is_preserved(self):
        # 这些字符可以出现在 JSON 字符串内，不能被当成 JSONL 记录之间的换行。
        path = resolve_receipt_path(self.processing["outputs"]["applications"]["path"], root=self.root)
        applications = read_jsonl(path)
        applications[0]["note"] = "原文分隔符\u0085\u2028\u2029保持在同一条 JSON 中"
        self.rewrite_jsonl_artifact("applications", applications)
        report, _, _ = audit_processed_dataset(self.root)
        self.assertTrue(report["mechanical_checks_passed"])

    def test_missing_validation_dispute_is_rejected(self):
        # 即使 CSV 标签始终未变，处理报告也不能悄悄遗漏原始证据已经记录的标签争议。
        self.assertTrue(self.processing["known_validation_label_disputes"])
        self.processing["known_validation_label_disputes"] = []
        self.processing_path.write_text(json.dumps(self.processing), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "验证标签争议"):
            audit_processed_dataset(self.root)

    def test_cli_default_output_follows_selected_root(self):
        # 不指定输出目录，验证 --root 同时约束默认报告位置和报告内部路径。
        with redirect_stdout(io.StringIO()):
            result = audit_main(["--root", str(self.root)])
        self.assertEqual(result, 0)
        output = self.root / "data/reports/processed_review"
        self.assertTrue((output / "lineage_audit.json").is_file())
        report = json.loads((output / "structural_validation/validation_report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed")
        self.assertTrue(all(not Path(info["path"]).is_absolute() for info in report["files"].values()))

    def retained_pipeline(self):
        """建立同标签归一化重复的完整流水线，供批准继承相关用例复用。"""
        fixture = pipeline_fixtures.DatasetPipelineTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.rows["train"].extend([fixture.row("a", "价格  合理", 1), fixture.row("b", "价格 合理", 1)])
        fixture.prepare_review()
        processing, _, _, _ = fixture.process_and_audit()
        return fixture, processing

    def test_cli_fresh_validation_honors_only_audited_duplicate_retention(self):
        # 重新扫描仍会客观发现重复；报告应保留该扫描事实，同时明确对应的批准来源。
        fixture, _ = self.retained_pipeline()
        with redirect_stdout(io.StringIO()):
            result = audit_main(["--root", str(fixture.root)])
        self.assertEqual(result, 0)
        output = fixture.root / "data/reports/processed_review"
        report = json.loads((output / "structural_validation/validation_report.json").read_text(encoding="utf-8"))
        lineage = json.loads((output / "lineage_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "passed_with_warnings")
        self.assertEqual(report["scan_status"], "needs_review")
        self.assertEqual(report["pending_reviews"], 0)
        self.assertTrue(report["review_resolutions"])
        self.assertEqual(lineage["fresh_validation"]["review_resolutions"], report["review_resolutions"])

    def test_unknown_source_in_post_validation_resolution_is_rejected(self):
        # 只更新二次校验报告的摘要，无法使无关问题编号变成有效的重复保留批准。
        fixture, processing = self.retained_pipeline()
        artifact = processing["outputs"]["validation_report"]
        validation_path = resolve_receipt_path(artifact["path"], root=fixture.root)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["review_resolutions"][0]["source_issue_ids"] = ["unrelated"]
        validation_path.write_text(json.dumps(validation), encoding="utf-8")
        artifact["sha256"] = file_sha256(validation_path)
        processing_path = fixture.root / "data/reports/processed/processing_report.json"
        processing_path.write_text(json.dumps(processing), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "保留证据"):
            audit_processed_dataset(fixture.root)

    def test_cli_fresh_validation_rejects_a_changed_review_group(self):
        fixture, _ = self.retained_pipeline()

        def changed_review(*args, **kwargs):
            # 模拟独立扫描发现正文不同的新证据：即便沿用旧问题编号，也必须重新复核。
            report = validate_dataset(*args, **kwargs)
            path = Path(args[1]) / "validation_issues.jsonl"
            issues = read_jsonl(path)
            issue = next(item for item in issues if item["severity"] == "REVIEW")
            issue["records"][0]["text"] += "未经复核的新证据"
            path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in issues), encoding="utf-8")
            return report

        with patch("scripts.audit_processed_dataset.validate_dataset", side_effect=changed_review), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                audit_main(["--root", str(fixture.root)])
        self.assertEqual(raised.exception.code, 2)
        # 失败时不能发布代表审计已完成的来源总报告。
        self.assertFalse((fixture.root / "data/reports/processed_review/lineage_audit.json").exists())

    def alternative_batch(self):
        """在同一临时项目中创建第二批产物，输入仍由其自身处理凭据完整绑定。"""
        reports = self.root / "data/reports/candidate"
        result = self.fixture.process(output_dir=self.root / "data/candidate",
                                      quarantine_dir=self.root / "data/candidate_quarantine",
                                      reports_dir=reports)
        return reports / "processing_report.json", result

    def test_selected_batch_is_audited_without_reading_default_outputs(self):
        selected, processing = self.alternative_batch()
        # 默认批次仍然存在但已经损坏；显式选择的新批次必须完全独立核验。
        default_train = resolve_receipt_path(self.processing["outputs"]["train"]["path"], root=self.root)
        default_train.write_bytes(b"default batch is unusable")
        report, _, _ = audit_processed_dataset(self.root, processing_report=selected)
        self.assertTrue(report["mechanical_checks_passed"])
        self.assertEqual(resolve_receipt_path(report["batch"]["processing_report"], root=self.root), selected)
        self.assertEqual(report["retained_counts"], processing["processed_counts"])

    def test_selected_batch_cli_revalidates_selected_outputs(self):
        selected, processing = self.alternative_batch()
        output = self.root / "data/reports/candidate_review"
        default_train = resolve_receipt_path(self.processing["outputs"]["train"]["path"], root=self.root)
        default_train.write_bytes(b"default batch is unusable")
        with redirect_stdout(io.StringIO()):
            result = audit_main(["--root", str(self.root), "--processing-report", str(selected),
                                 "--output-dir", str(output)])
        self.assertEqual(result, 0)
        validation = json.loads((output / "structural_validation/validation_report.json").read_text(encoding="utf-8"))
        for split in ("train", "validation"):
            self.assertEqual(validation["files"][split]["path"], processing["outputs"][split]["path"])

    def test_missing_selected_receipt_never_falls_back_to_default(self):
        with self.assertRaises(FileNotFoundError):
            audit_processed_dataset(self.root, processing_report=self.root / "missing/processing_report.json")

    def test_missing_selected_artifact_never_falls_back_to_default(self):
        selected, processing = self.alternative_batch()
        resolve_receipt_path(processing["outputs"]["train"]["path"], root=self.root).unlink()
        with self.assertRaises(FileNotFoundError):
            audit_processed_dataset(self.root, processing_report=selected)

    def test_source_report_with_same_name_cannot_replace_bound_reference(self):
        selected, processing = self.alternative_batch()
        original = self.fixture.report_path
        shadow = self.root / "shadow/validation_report.json"
        shadow.parent.mkdir()
        shadow.write_bytes(original.read_bytes())
        original_key = next(name for name in processing["inputs"]
                            if resolve_receipt_path(name, root=self.root) == original)
        processing["inputs"][str(shadow)] = processing["inputs"].pop(original_key)
        processing["input_roles"]["raw_report"] = str(shadow)
        selected.write_text(json.dumps(processing), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "自引用路径"):
            audit_processed_dataset(self.root, processing_report=selected)

    def test_handwritten_empty_or_unmatched_review_evidence_is_rejected(self):
        original = json.loads(json.dumps(self.fixture.decisions))
        for field, value in (("reason", ""), ("reason", " \n\t"), ("evidence_quotes", []),
                             ("evidence_quotes", [""]), ("evidence_quotes", ["原文中不存在的引文"])):
            with self.subTest(field=field, value=value):
                self.fixture.decisions = json.loads(json.dumps(original))
                self.fixture.decisions[0][field] = value
                self.refresh_review_chain()
                with self.assertRaisesRegex(ValueError, "复核理由|证据引文"):
                    audit_processed_dataset(self.root)

    def test_selected_legacy_absolute_batch_migrates_without_rewriting_copy(self):
        selected, processing = self.alternative_batch()
        processing["inputs"] = {str(resolve_receipt_path(name, root=self.root)): digest
                                for name, digest in processing["inputs"].items()}
        processing["input_roles"] = {name: str(resolve_receipt_path(value, root=self.root))
                                     for name, value in processing["input_roles"].items()}
        for artifact in processing["outputs"].values():
            artifact["path"] = str(resolve_receipt_path(artifact["path"], root=self.root))
        selected.write_text(json.dumps(processing), encoding="utf-8")
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / "copy"
            shutil.copytree(self.root, copied)
            # 破坏旧位置的来源和新批产物，确保迁移逻辑不依赖仍然存在的旧目录。
            self.fixture.train_path.write_bytes(b"old source is unusable")
            Path(processing["outputs"]["train"]["path"]).write_bytes(b"old batch is unusable")
            copied_receipt = copied / selected.relative_to(self.root)
            report, _, _ = audit_processed_dataset(copied, processing_report=copied_receipt)
            self.assertTrue(report["mechanical_checks_passed"])
            self.assertTrue(all(resolve_receipt_path(value, root=copied).is_relative_to(copied)
                                for value in report["file_sha256"]))

    def overridden_pipeline(self):
        """构造训练同标签重复被跨集合隔离完整覆盖的真实多问题计划。"""
        fixture = pipeline_fixtures.DatasetPipelineTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        fixture.rows["train"].extend([fixture.row("a", "价格 合理", 1), fixture.row("b", "价格 合理", 1)])
        fixture.rows["validation"].append(fixture.row("validation-a", "价格  合理", 1))
        fixture.prepare_review()
        processing, _, _, _ = fixture.process_and_audit()
        return fixture, processing

    @staticmethod
    def save_pipeline_application_changes(fixture, processing, applications):
        """更新测试日志与外层摘要，确保审计必须检查逐条覆盖关系。"""
        artifact = processing["outputs"]["applications"]
        path = resolve_receipt_path(artifact["path"], root=fixture.root)
        path.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in applications), encoding="utf-8")
        artifact["sha256"] = file_sha256(path)
        (fixture.root / "data/reports/processed/processing_report.json").write_text(json.dumps(processing), encoding="utf-8")

    def test_overridden_retention_cannot_claim_to_have_been_applied(self):
        fixture, processing = self.overridden_pipeline()
        path = resolve_receipt_path(processing["outputs"]["applications"]["path"], root=fixture.root)
        applications = read_jsonl(path)
        retained = next(item for item in applications if item["action_type"] == "retain_reviewed_records")
        self.assertEqual(retained["application_status"], "overridden")
        retained.update(action_applied=True, application_status="applied", resolution="reviewed_records_retained_unchanged")
        for record in retained["record_results"]:
            record.update(decision_applied=True, overridden_by_issue_ids=[])
        self.save_pipeline_application_changes(fixture, processing, applications)
        with self.assertRaisesRegex(ValueError, "应用日志"):
            audit_processed_dataset(fixture.root)

    def test_v2_record_results_cannot_hide_the_action_that_overrode_retention(self):
        fixture, processing = self.overridden_pipeline()
        path = resolve_receipt_path(processing["outputs"]["applications"]["path"], root=fixture.root)
        applications = read_jsonl(path)
        retained = next(item for item in applications if item["action_type"] == "retain_reviewed_records")
        retained["record_results"][0]["overridden_by_issue_ids"] = []
        self.save_pipeline_application_changes(fixture, processing, applications)
        with self.assertRaisesRegex(ValueError, "逐条生效范围"):
            audit_processed_dataset(fixture.root)

    def test_v2_final_record_outcomes_are_recomputed_from_the_complete_plan(self):
        fixture, processing = self.overridden_pipeline()
        processing["record_outcomes"][0]["disposition"] = "retained"
        path = fixture.root / "data/reports/processed/processing_report.json"
        path.write_text(json.dumps(processing), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "最终去向"):
            audit_processed_dataset(fixture.root)

    def test_legacy_logs_remain_valid_when_no_decision_is_overridden(self):
        processing = self.processing
        processing.pop("application_log_version")
        processing.pop("input_roles")
        processing.pop("record_outcomes")
        applications = read_jsonl(resolve_receipt_path(processing["outputs"]["applications"]["path"], root=self.root))
        for application in applications:
            application.pop("record_results")
            application.pop("application_status")
        self.rewrite_jsonl_artifact("applications", applications)
        report, _, _ = audit_processed_dataset(self.root)
        self.assertTrue(report["mechanical_checks_passed"])

    def test_downgrading_overridden_plan_to_legacy_logs_is_rejected(self):
        fixture, processing = self.overridden_pipeline()
        processing.pop("application_log_version")
        processing.pop("input_roles")
        processing.pop("record_outcomes")
        applications = read_jsonl(resolve_receipt_path(processing["outputs"]["applications"]["path"], root=fixture.root))
        for application in applications:
            application.pop("record_results")
            application.pop("application_status")
        self.save_pipeline_application_changes(fixture, processing, applications)
        with self.assertRaisesRegex(ValueError, "旧版应用日志"):
            audit_processed_dataset(fixture.root)

    def test_changed_post_validation_configuration_is_rejected(self):
        artifact = self.processing["outputs"]["validation_report"]
        path = resolve_receipt_path(artifact["path"], root=self.root)
        validation = json.loads(path.read_text(encoding="utf-8"))
        validation["configuration"]["min_class_fraction"] = 0.2
        path.write_text(json.dumps(validation), encoding="utf-8")
        artifact["sha256"] = file_sha256(path)
        self.processing_path.write_text(json.dumps(self.processing), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "校验配置"):
            audit_processed_dataset(self.root)

    def test_removing_version_cannot_bypass_new_record_outcomes(self):
        self.processing.pop("application_log_version")
        self.processing_path.write_text(json.dumps(self.processing), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "字段混用"):
            audit_processed_dataset(self.root)

    def test_removing_version_and_outcomes_cannot_bypass_new_application_fields(self):
        self.processing.pop("application_log_version")
        self.processing.pop("record_outcomes")
        self.processing_path.write_text(json.dumps(self.processing), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "字段混用"):
            audit_processed_dataset(self.root)

    def test_no_issue_batch_binds_two_distinct_empty_jsonl_inputs(self):
        """问题与决定都为空时摘要相同，仍须按各自路径角色完整执行和审计。"""
        fixture = pipeline_fixtures.DatasetPipelineTests()
        self.addCleanup(fixture.doCleanups)
        fixture.setUp()
        decisions, _ = fixture.prepare_review()
        self.assertEqual(decisions, [])
        processing, report, edits, disputes = fixture.process_and_audit()
        self.assertTrue(report["mechanical_checks_passed"])
        self.assertEqual(processing["record_outcomes"], [])
        self.assertEqual(edits, [])
        self.assertEqual(disputes, [])

    def test_unsupported_source_rules_cannot_authorize_isolation_after_digest_refresh(self):
        original_issues = json.loads(json.dumps(self.fixture.issues))
        original_decisions = json.loads(json.dumps(self.fixture.decisions))
        cases = (("quarantine_training_conflict_cluster", "within_split_unsupported_rule", "冲突隔离动作"),
                 ("quarantine_training_overlap", "cross_split_unsupported_rule", "重叠隔离动作"))
        for action_type, rule, message in cases:
            with self.subTest(rule=rule):
                self.fixture.issues = json.loads(json.dumps(original_issues))
                self.fixture.decisions = json.loads(json.dumps(original_decisions))
                decision = next(item for item in self.fixture.decisions if item["action"]["type"] == action_type)
                issue = next(item for item in self.fixture.issues if item["issue_id"] == decision["issue_id"])
                issue["rule_id"] = decision["rule_id"] = rule
                self.refresh_source_issue_chain()
                with self.assertRaisesRegex(ValueError, message):
                    audit_processed_dataset(self.root)

    def test_conflict_isolation_cannot_be_used_for_a_group_containing_validation(self):
        decision = next(item for item in self.fixture.decisions if item["action"]["type"] == "quarantine_training_overlap")
        issue = next(item for item in self.fixture.issues if item["issue_id"] == decision["issue_id"])
        issue["rule_id"] = decision["rule_id"] = "within_split_normalized_duplicate"
        decision["action"]["type"] = "quarantine_training_conflict_cluster"
        applications_path = resolve_receipt_path(self.processing["outputs"]["applications"]["path"], root=self.root)
        applications = read_jsonl(applications_path)
        application = next(item for item in applications if item["issue_id"] == decision["issue_id"])
        application["action_type"] = decision["action"]["type"]
        self.rewrite_jsonl_artifact("applications", applications)
        self.refresh_source_issue_chain()
        with self.assertRaisesRegex(ValueError, "冲突隔离动作"):
            audit_processed_dataset(self.root)

    def test_clearing_original_conflict_summary_cannot_hide_validation_disputes(self):
        issue = next(item for item in self.fixture.issues if item["rule_id"].startswith("cross_split_"))
        self.assertTrue(issue["label_conflicts"])
        issue["label_conflicts"] = {}
        self.processing["known_validation_label_disputes"] = []
        self.refresh_source_issue_chain()
        with self.assertRaisesRegex(ValueError, "标签冲突明细"):
            audit_processed_dataset(self.root)


if __name__ == "__main__":
    unittest.main()
