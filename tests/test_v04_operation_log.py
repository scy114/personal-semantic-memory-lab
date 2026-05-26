import json
import tempfile
import unittest
from pathlib import Path

from tools.maintenance.operation_log import (
    SCHEMA_VERSION,
    append_operation,
    build_manifest,
    build_operation,
    latest_status_by_operation_id,
    read_jsonl,
    stable_operation_id,
)


class V04OperationLogTests(unittest.TestCase):
    def test_builds_stable_append_only_operation(self):
        operation = build_operation(
            workspace_id="workspace-a",
            modeled_user_id="Mira",
            operation_type="add_source",
            idempotency_key="source:file-a:v1",
            scope={
                "source_ids": ["source-a", "source-a"],
                "source_versions": ["v1"],
                "evidence_refs": ["evidence-a"],
                "s0b_unit_ids": ["s0b-a"],
            },
            warnings=["review before publish"],
        )

        self.assertEqual(operation["schema_version"], SCHEMA_VERSION)
        self.assertEqual(operation["operation_type"], "add_source")
        self.assertEqual(operation["status"], "planned")
        self.assertTrue(operation["append_only"])
        self.assertFalse(operation["write_permission"])
        self.assertEqual(operation["scope"]["source_ids"], ["source-a"])
        self.assertEqual(operation["scope"]["evidence_refs"], ["evidence-a"])
        self.assertEqual(
            operation["operation_id"],
            stable_operation_id(
                workspace_id="workspace-a",
                operation_type="add_source",
                idempotency_key="source:file-a:v1",
            ),
        )

    def test_rejects_unknown_operation_type_and_scope_key(self):
        with self.assertRaises(ValueError):
            build_operation(
                workspace_id="workspace-a",
                operation_type="rewrite_everything",
                idempotency_key="bad",
            )
        with self.assertRaises(ValueError):
            build_operation(
                workspace_id="workspace-a",
                operation_type="add_source",
                idempotency_key="bad-scope",
                scope={"unknown_ids": ["x"]},
            )

    def test_appends_reads_and_summarizes_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "maintenance" / "operation_log.jsonl"
            planned = build_operation(
                workspace_id="workspace-a",
                operation_type="add_source",
                idempotency_key="source:file-a:v1",
                status="planned",
                scope={"source_ids": ["source-a"]},
            )
            completed = build_operation(
                workspace_id="workspace-a",
                operation_type="add_source",
                idempotency_key="source:file-a:v1",
                status="completed",
                scope={"source_ids": ["source-a"]},
            )

            append_operation(log_path, planned)
            append_operation(log_path, completed)

            rows = read_jsonl(log_path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(latest_status_by_operation_id(rows)[planned["operation_id"]], "completed")
            manifest = build_manifest(log_path, rows)
            self.assertEqual(manifest["operation_count"], 2)
            self.assertEqual(manifest["operation_type_counts"], {"add_source": 2})
            self.assertEqual(manifest["status_counts"], {"completed": 1, "planned": 1})

            raw_lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(raw_lines), 2)
            self.assertTrue(all(json.loads(line)["append_only"] for line in raw_lines))


if __name__ == "__main__":
    unittest.main()
