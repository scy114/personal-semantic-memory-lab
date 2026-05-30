import json
import shutil
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_jsonl
from tools.maintenance.graph_candidate_latest_view_runner import run_graph_candidate_latest_view


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class V04GraphCandidateLatestViewRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_v04_graph_candidate_latest_view_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def seed_graph_dir(self, graph_dir: Path, *, edge_relation: str, edge_id: str = "edge:1", low_value: bool = False) -> None:
        write_jsonl(
            graph_dir / "graph_nodes_table.jsonl",
            [
                {"node_id": "node:jon", "label": "Jon", "entity_quality_hint": "stable", "graph_is_not_proof": True},
                {"node_id": "node:thing", "label": "thing", "entity_quality_hint": "review_required", "graph_is_not_proof": True},
            ],
        )
        write_jsonl(
            graph_dir / "graph_edges_table.jsonl",
            [
                {
                    "edge_id": edge_id,
                    "source_node_id": "node:jon",
                    "target_node_id": "node:thing",
                    "relation_type": edge_relation,
                    "generic_relation_review_hint": "low_graph_value" if low_value else "not_generic",
                    "graph_is_not_proof": True,
                }
            ],
        )
        write_jsonl(graph_dir / "graph_claims_table.jsonl", [{"claim_id": "claim:1", "subject_endpoint_status": "local_entity_id", "graph_is_not_proof": True}])
        write_jsonl(
            graph_dir / "evidence_links.jsonl",
            [
                {"link_id": "link:edge", "owner_id": edge_id, "owner_kind": "edge", "graph_is_not_proof": True},
                {"link_id": "link:missing", "owner_id": "edge:missing", "owner_kind": "edge", "graph_is_not_proof": True},
            ],
        )
        (graph_dir / "graph_consolidation_manifest.json").parent.mkdir(parents=True, exist_ok=True)
        (graph_dir / "graph_consolidation_manifest.json").write_text('{"schema_version":"test"}\n', encoding="utf-8")

    def test_incremental_rows_overlay_base_and_exclude_low_value_audit_rows(self):
        base_dir = self.workspace / "base_graph"
        incremental_dir = self.workspace / "incremental_graph"
        output_dir = self.workspace / "latest_graph"
        self.seed_graph_dir(base_dir, edge_relation="desires")
        self.seed_graph_dir(incremental_dir, edge_relation="related_to_generic", low_value=True)

        manifest = run_graph_candidate_latest_view(
            base_graph_dir=base_dir,
            incremental_graph_dir=incremental_dir,
            output_dir=output_dir,
        )

        latest_edges = read_jsonl(output_dir / "graph_edges_latest_view.jsonl")
        excluded_edges = read_jsonl(output_dir / "graph_edges_latest_view_excluded.jsonl")
        latest_nodes = read_jsonl(output_dir / "graph_nodes_latest_view.jsonl")
        excluded_nodes = read_jsonl(output_dir / "graph_nodes_latest_view_excluded.jsonl")
        latest_links = read_jsonl(output_dir / "evidence_links_latest_view.jsonl")
        excluded_links = read_jsonl(output_dir / "evidence_links_latest_view_excluded.jsonl")

        self.assertEqual(latest_edges, [])
        edge_reasons = [row["_maintenance_graph_latest_view"]["reason"] for row in excluded_edges]
        self.assertIn("overlaid_by_incremental_candidate", edge_reasons)
        self.assertIn("edge_generic_relation_review:low_graph_value", edge_reasons)
        self.assertEqual([row["node_id"] for row in latest_nodes], ["node:jon"])
        node_reasons = [row["_maintenance_graph_latest_view"]["reason"] for row in excluded_nodes]
        self.assertIn("overlaid_by_incremental_candidate", node_reasons)
        self.assertIn("entity_review_required", node_reasons)
        self.assertEqual(latest_links, [])
        self.assertTrue(any(row["_maintenance_graph_latest_view"]["reason"] == "evidence_owner_not_active" for row in excluded_links))
        self.assertFalse(manifest["graph_truth_written"])
        self.assertFalse(manifest["graph_query_indexes_refreshed"])
        self.assertTrue((output_dir / "latest_views" / "graph_edges_latest_view.jsonl").exists())
        self.assertFalse((self.workspace / "graph" / "edges.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
