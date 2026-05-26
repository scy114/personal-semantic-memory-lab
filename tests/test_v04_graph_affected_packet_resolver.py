import tempfile
import unittest
from pathlib import Path

from tools.maintenance.graph_affected_packet_resolver import (
    build_graph_affected_packet_report,
    build_graph_affected_packet_report_from_files,
    read_jsonl,
    write_jsonl,
)


class V04GraphAffectedPacketResolverTests(unittest.TestCase):
    def test_resolves_packet_and_candidates_by_evidence(self):
        bundle = build_graph_affected_packet_report(
            s1_delta_decisions=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-1",
                    "delta_type": "needs_review",
                    "new_evidence_refs": ["evidence:1"],
                }
            ],
            s2_affected_rows=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-1",
                    "s1_delta_type": "needs_review",
                    "s2_delta_signal": "needs_review",
                    "affected_s2_unit_id": "s2-1",
                    "new_evidence_refs": ["evidence:1"],
                }
            ],
            graph_packet_rows=[
                {
                    "packet_id": "graph_packet:1",
                    "evidence_refs": ["evidence:1"],
                    "context_evidence_refs": ["evidence:2"],
                }
            ],
            graph_candidate_rows=[
                {
                    "edge_id": "edge-1",
                    "source_packet_ids": ["graph_packet:1"],
                    "evidence_refs": ["evidence:1"],
                    "graph_is_not_proof": True,
                },
                {
                    "node_id": "node-1",
                    "source_packet_ids": ["graph_packet:1"],
                    "evidence_refs": ["evidence:1"],
                    "graph_is_not_proof": True,
                },
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        row = bundle["affected_rows"][0]
        self.assertEqual(row["affected_graph_packet_ids"], ["graph_packet:1"])
        self.assertEqual(row["primary_graph_packet_ids"], ["graph_packet:1"])
        self.assertEqual(row["context_graph_packet_ids"], [])
        self.assertEqual(row["affected_graph_candidate_ids"], ["edge-1", "node-1"])
        self.assertEqual(row["candidate_kind_counts"], {"edge": 1, "node": 1})
        self.assertEqual(row["graph_delta_signal"], "needs_review")
        self.assertEqual(row["recommended_action"], "review_graph_candidates_after_s1_s2_review")
        self.assertEqual(row["query_invalidation_hint"], "invalidate_graph_query_assets")
        self.assertEqual(row["refresh_scope_hint"], "primary_packets_only")
        self.assertTrue(row["entity_merge_separate"])
        self.assertTrue(row["graph_is_not_proof"])
        self.assertFalse(row["write_permission"])
        self.assertEqual(bundle["report"]["affected_graph_packet_ids"], ["graph_packet:1"])
        self.assertEqual(bundle["report"]["affected_graph_candidate_ids"], ["edge-1", "node-1"])

    def test_no_material_change_does_not_request_graph_refresh(self):
        bundle = build_graph_affected_packet_report(
            s1_delta_decisions=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-1",
                    "delta_type": "no_material_change",
                    "new_evidence_refs": ["evidence:1"],
                }
            ],
            s2_affected_rows=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-1",
                    "s1_delta_type": "no_material_change",
                    "s2_delta_signal": "no_material_change",
                    "affected_s2_unit_id": "s2-1",
                    "new_evidence_refs": ["evidence:1"],
                }
            ],
            graph_packet_rows=[{"packet_id": "graph_packet:1", "evidence_refs": ["evidence:1"]}],
            graph_candidate_rows=[{"edge_id": "edge-1", "source_packet_ids": ["graph_packet:1"]}],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        row = bundle["affected_rows"][0]
        self.assertEqual(row["graph_delta_signal"], "no_graph_change")
        self.assertEqual(row["recommended_action"], "no_graph_refresh_expected")
        self.assertEqual(row["query_invalidation_hint"], "no_query_invalidation_expected")
        self.assertEqual(row["refresh_scope_hint"], "no_refresh")

    def test_new_unit_without_existing_packet_routes_graph_extraction(self):
        bundle = build_graph_affected_packet_report(
            s1_delta_decisions=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-new",
                    "delta_type": "new_unit",
                    "new_evidence_refs": ["evidence:new"],
                }
            ],
            s2_affected_rows=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-new",
                    "s1_delta_type": "new_unit",
                    "s2_delta_signal": "new_profile_unit",
                    "new_evidence_refs": ["evidence:new"],
                }
            ],
            graph_packet_rows=[],
            graph_candidate_rows=[],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        row = bundle["affected_rows"][0]
        self.assertEqual(row["match_type"], "no_existing_graph_packet_for_new_or_changed_s1")
        self.assertEqual(row["graph_delta_signal"], "new_edge_candidate")
        self.assertEqual(row["recommended_action"], "route_graph_extraction_for_new_packet")

    def test_does_not_join_consolidated_candidates_by_broad_evidence_refs(self):
        bundle = build_graph_affected_packet_report(
            s1_delta_decisions=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-1",
                    "delta_type": "needs_review",
                    "new_evidence_refs": ["evidence:1"],
                }
            ],
            s2_affected_rows=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-1",
                    "s1_delta_type": "needs_review",
                    "new_evidence_refs": ["evidence:1"],
                }
            ],
            graph_packet_rows=[{"packet_id": "graph_packet:1", "evidence_refs": ["evidence:1"]}],
            graph_candidate_rows=[
                {"node_id": "node-in-scope", "source_packet_ids": ["graph_packet:1"], "evidence_refs": ["evidence:1"]},
                {"node_id": "node-broad", "source_packet_ids": ["graph_packet:other"], "evidence_refs": ["evidence:1"]},
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        self.assertEqual(bundle["affected_rows"][0]["affected_graph_candidate_ids"], ["node-in-scope"])
        self.assertEqual(
            bundle["report"]["candidate_resolution_policy"],
            "Graph candidates are resolved through source_packet_ids; consolidated node evidence_refs are not used as a broad candidate join key.",
        )

    def test_writes_report_bundle_from_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            s1_path = root / "s1_delta.jsonl"
            s2_path = root / "s2_affected.jsonl"
            packets_path = root / "packets.jsonl"
            edges_path = root / "edges.jsonl"
            output_dir = root / "out"
            write_jsonl(
                s1_path,
                [{"operation_id": "op-file", "new_s1_unit_id": "memory-1", "delta_type": "needs_review"}],
            )
            write_jsonl(
                s2_path,
                [
                    {
                        "operation_id": "op-file",
                        "new_s1_unit_id": "memory-1",
                        "s1_delta_type": "needs_review",
                        "new_evidence_refs": ["evidence:1"],
                    }
                ],
            )
            write_jsonl(packets_path, [{"packet_id": "graph_packet:1", "evidence_refs": ["evidence:1"]}])
            write_jsonl(edges_path, [{"edge_id": "edge-1", "source_packet_ids": ["graph_packet:1"]}])

            bundle = build_graph_affected_packet_report_from_files(
                s1_delta_decisions_jsonl=s1_path,
                s2_affected_rows_jsonl=s2_path,
                graph_packets_jsonl=packets_path,
                graph_edges_jsonl=edges_path,
                output_dir=output_dir,
            )

            self.assertTrue((output_dir / "graph_affected_packets.jsonl").exists())
            self.assertTrue((output_dir / "graph_affected_packet_report.json").exists())
            written = read_jsonl(output_dir / "graph_affected_packets.jsonl")
            self.assertEqual(written[0]["operation_id"], "op-file")
            self.assertEqual(bundle["report"]["affected_row_count"], 1)


if __name__ == "__main__":
    unittest.main()
