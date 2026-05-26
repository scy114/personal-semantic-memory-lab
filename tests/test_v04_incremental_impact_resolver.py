import tempfile
import unittest
from pathlib import Path

from tools.maintenance.incremental_impact_resolver import (
    read_jsonl,
    resolve_operation_impact,
    write_jsonl,
)
from tools.maintenance.operation_log import build_operation


class V04IncrementalImpactResolverTests(unittest.TestCase):
    def test_resolves_source_to_s1_s2_indexes_and_graph(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            write_jsonl(
                workspace / "evidence" / "evidence.jsonl",
                [
                    {
                        "evidence_ref": "evidence:1",
                        "source_id": "source-a",
                        "source_refs": ["source-a"],
                        "text": "Mira uses a notebook.",
                    }
                ],
            )
            write_jsonl(
                workspace / "memory" / "memory_units.jsonl",
                [
                    {
                        "memory_id": "memory-1",
                        "source_refs": ["source-a"],
                        "evidence_refs": ["evidence:1"],
                        "status": "accepted_for_experiment",
                    }
                ],
            )
            write_jsonl(
                workspace / "portrait" / "reviewed_units.jsonl",
                [
                    {
                        "unit_id": "unit-1",
                        "evidence_refs": ["evidence:1"],
                        "status": "active",
                    }
                ],
            )
            write_jsonl(
                workspace / "indexes" / "step1_bm25_entries.jsonl",
                [
                    {
                        "index_entry_id": "s1idx-1",
                        "object_id": "memory-1",
                        "source_refs": ["source-a"],
                        "evidence_refs": ["evidence:1"],
                    }
                ],
            )
            write_jsonl(
                workspace / "indexes" / "step2_user_model_embedding_entries.jsonl",
                [
                    {
                        "vector_entry_id": "s2idx-1",
                        "object_id": "unit-1",
                        "evidence_refs": ["evidence:1"],
                    }
                ],
            )
            write_jsonl(
                workspace / "graph_v03_consolidation_test" / "graph_edges_table.jsonl",
                [
                    {
                        "edge_id": "edge-1",
                        "evidence_refs": ["evidence:1"],
                        "graph_is_not_proof": True,
                    }
                ],
            )

            operation = build_operation(
                workspace_id="workspace",
                operation_type="supersede_source",
                idempotency_key="source-a:v2",
                scope={"source_ids": ["source-a"]},
            )
            report = resolve_operation_impact(workspace, operation)

            self.assertEqual(report["impacted"]["evidence_refs"], ["evidence:1"])
            self.assertEqual(report["impacted"]["s1_unit_ids"], ["memory-1"])
            self.assertEqual(report["impacted"]["s2_unit_ids"], ["unit-1"])
            self.assertEqual(report["impacted"]["s1_index_entry_ids"], ["s1idx-1"])
            self.assertEqual(report["impacted"]["s2_index_entry_ids"], ["s2idx-1"])
            self.assertEqual(report["impacted"]["graph_candidate_ids"], ["edge-1"])
            self.assertIn("invalidate_indexes", report["recommended_actions"])
            self.assertTrue(report["read_only"])
            self.assertFalse(report["write_permission"])

    def test_reports_gap_when_scope_does_not_resolve(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            operation = build_operation(
                workspace_id="workspace",
                operation_type="deprecate_source",
                idempotency_key="missing-source",
                scope={"source_ids": ["missing"]},
            )

            report = resolve_operation_impact(workspace, operation)

            self.assertEqual(report["recommended_actions"], ["gap_review_no_impacted_artifacts_found"])
            self.assertIn("operation_scope_did_not_resolve_to_s0b_s1_or_s2_assets", report["gaps"])

    def test_read_jsonl_ignores_blank_lines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "rows.jsonl"
            path.write_text('{"id": "a"}\n\n{"id": "b"}\n', encoding="utf-8")
            self.assertEqual([row["id"] for row in read_jsonl(path)], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
