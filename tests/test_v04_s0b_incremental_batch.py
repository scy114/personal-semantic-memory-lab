import json
import tempfile
import unittest
from pathlib import Path

from tools.maintenance.operation_log import read_jsonl as read_operation_log
from tools.maintenance.s0b_incremental_batch import (
    build_s0b_incremental_batch,
    register_s0b_incremental_batch,
    write_jsonl,
)


class V04S0BIncrementalBatchTests(unittest.TestCase):
    def test_builds_add_only_batch_manifest(self):
        batch = build_s0b_incremental_batch(
            workspace=Path("workspace-a"),
            batch_id="batch-1",
            evidence_rows=[
                {
                    "evidence_ref": "evidence:1",
                    "source_id": "source-a",
                    "text": "Mira changed the plan.",
                }
            ],
            text_unit_rows=[
                {
                    "text_unit_id": "s0b:1",
                    "raw_source_id": "source-a",
                    "evidence_ref": "evidence:1",
                    "text": "Mira changed the plan.",
                }
            ],
            section_rows=[
                {
                    "raw_span_id": "span:1",
                    "raw_source_id": "source-a",
                    "final_section_type": "body",
                }
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        manifest = batch["manifest"]
        self.assertEqual(manifest["schema_version"], "maintenance.s0b_incremental_batch_manifest.v0.4")
        self.assertEqual(manifest["row_kind_counts"], {"evidence": 1, "section": 1, "text_unit": 1})
        self.assertEqual(manifest["source_ids"], ["source-a"])
        self.assertEqual(manifest["evidence_refs"], ["evidence:1"])
        self.assertEqual(manifest["s0b_unit_ids"], ["s0b:1", "span:1"])
        self.assertTrue(manifest["add_only"])
        self.assertFalse(manifest["canonical_mutation"])
        self.assertFalse(manifest["write_permission"])
        self.assertTrue(all(row["read_only"] for row in batch["rows"]))

    def test_registers_batch_and_appends_operation_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace-a"
            input_dir = root / "input"
            output_dir = root / "maintenance"
            operation_log = output_dir / "operation_log.jsonl"
            evidence_path = input_dir / "evidence.jsonl"
            text_units_path = input_dir / "text_units.jsonl"
            write_jsonl(
                evidence_path,
                [{"evidence_ref": "evidence:1", "source_id": "source-a", "text": "New evidence."}],
            )
            write_jsonl(
                text_units_path,
                [{"text_unit_id": "s0b:1", "raw_source_id": "source-a", "evidence_ref": "evidence:1"}],
            )
            before_evidence_text = evidence_path.read_text(encoding="utf-8")

            result = register_s0b_incremental_batch(
                workspace=workspace,
                batch_id="batch-1",
                output_dir=output_dir,
                evidence_jsonl=evidence_path,
                text_units_jsonl=text_units_path,
                operation_log=operation_log,
            )

            self.assertTrue(Path(result["paths"]["rows_path"]).exists())
            self.assertTrue(Path(result["paths"]["manifest_path"]).exists())
            self.assertEqual(evidence_path.read_text(encoding="utf-8"), before_evidence_text)

            operations = read_operation_log(operation_log)
            self.assertEqual(len(operations), 1)
            operation = operations[0]
            self.assertEqual(operation["operation_type"], "add_s0b_batch")
            self.assertEqual(operation["status"], "completed")
            self.assertEqual(operation["scope"]["source_ids"], ["source-a"])
            self.assertEqual(operation["scope"]["evidence_refs"], ["evidence:1"])
            self.assertEqual(operation["scope"]["s0b_unit_ids"], ["s0b:1"])
            self.assertIn("canonical_s0b_files_not_mutated", operation["warnings"])
            self.assertFalse(operation["write_permission"])

    def test_rejects_duplicate_object_ids_inside_batch(self):
        with self.assertRaises(ValueError):
            build_s0b_incremental_batch(
                workspace=Path("workspace-a"),
                batch_id="batch-dup",
                text_unit_rows=[
                    {"text_unit_id": "s0b:1"},
                    {"text_unit_id": "s0b:1"},
                ],
            )


if __name__ == "__main__":
    unittest.main()
