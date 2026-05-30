import json
import shutil
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_jsonl
from tools.maintenance.graph_dependency_map_builder import run_graph_dependency_map_build
from tools.maintenance.graph_query_index_invalidation_runner import run_graph_query_index_invalidation


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def seed_graph_current(graph_dir: Path, *, edge_relation: str = "desires", include_claim: bool = True) -> None:
    write_jsonl(
        graph_dir / "graph_nodes_latest_view.jsonl",
        [
            {
                "node_id": "node:jon",
                "label": "Jon",
                "candidate_ids": ["candidate:jon"],
                "source_packet_ids": ["packet:1"],
                "evidence_refs": ["evidence:1"],
                "graph_is_not_proof": True,
            },
            {
                "node_id": "node:studio",
                "label": "studio",
                "candidate_ids": ["candidate:studio"],
                "source_packet_ids": ["packet:1"],
                "evidence_refs": ["evidence:1"],
                "graph_is_not_proof": True,
            },
        ],
    )
    write_jsonl(graph_dir / "graph_nodes_latest_view_excluded.jsonl", [])
    write_jsonl(
        graph_dir / "graph_edges_latest_view.jsonl",
        [
            {
                "edge_id": "edge:jon-studio",
                "source_node_id": "node:jon",
                "target_node_id": "node:studio",
                "relation_type": edge_relation,
                "candidate_ids": ["candidate:edge"],
                "source_packet_ids": ["packet:1"],
                "evidence_refs": ["evidence:1"],
                "graph_is_not_proof": True,
            }
        ],
    )
    write_jsonl(graph_dir / "graph_edges_latest_view_excluded.jsonl", [])
    claims = []
    if include_claim:
        claims.append(
            {
                "claim_id": "claim:1",
                "subject_node_id": "node:jon",
                "candidate_id": "candidate:claim",
                "source_packet_id": "packet:2",
                "evidence_refs": ["evidence:2"],
                "graph_is_not_proof": True,
            }
        )
    write_jsonl(graph_dir / "graph_claims_latest_view.jsonl", claims)
    write_jsonl(graph_dir / "graph_claims_latest_view_excluded.jsonl", [])
    write_jsonl(
        graph_dir / "evidence_links_latest_view.jsonl",
        [
            {
                "link_id": "link:edge",
                "owner_kind": "edge",
                "owner_id": "edge:jon-studio",
                "candidate_id": "candidate:edge-link",
                "source_packet_id": "packet:link",
                "evidence_ref": "evidence:linked",
                "graph_is_not_proof": True,
            }
        ],
    )
    write_jsonl(graph_dir / "evidence_links_latest_view_excluded.jsonl", [])


class V04GraphDependencyAndInvalidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_v04_graph_dependency_invalidation_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def test_dependency_map_records_lineage_and_endpoint_dependencies(self):
        graph_dir = self.workspace / "graph_current"
        seed_graph_current(graph_dir)

        manifest = run_graph_dependency_map_build(graph_dir=graph_dir)
        rows = read_jsonl(Path(manifest["outputs"]["dependency_map"]))
        edge_dep = next(row for row in rows if row["graph_object_id"] == "edge:jon-studio")

        self.assertIn("node:jon", edge_dep["depends_on_node_ids"])
        self.assertIn("node:studio", edge_dep["depends_on_node_ids"])
        self.assertIn("evidence:1", edge_dep["evidence_refs"])
        self.assertIn("evidence:linked", edge_dep["evidence_refs"])
        self.assertIn("packet:link", edge_dep["source_packet_ids"])
        self.assertTrue(edge_dep["graph_is_not_proof"])
        self.assertFalse(manifest["graph_truth_written"])
        self.assertFalse(manifest["query_indexes_refreshed"])

    def test_invalidation_marks_changed_units_and_scoped_query_assets(self):
        previous_dir = self.workspace / "previous_graph_current"
        current_dir = self.workspace / "graph_current"
        seed_graph_current(previous_dir, edge_relation="desires", include_claim=True)
        seed_graph_current(current_dir, edge_relation="uses", include_claim=False)

        run_graph_dependency_map_build(graph_dir=current_dir)
        manifest = run_graph_query_index_invalidation(
            current_graph_dir=current_dir,
            previous_graph_dir=previous_dir,
        )
        changed = read_jsonl(Path(manifest["outputs"]["changed_graph_units"]))
        stale_assets = read_jsonl(Path(manifest["outputs"]["stale_graph_query_assets"]))

        edge_change = next(row for row in changed if row["graph_object_id"] == "edge:jon-studio")
        removed_claim = next(row for row in changed if row["graph_object_id"] == "claim:1")
        self.assertEqual(edge_change["change_type"], "changed")
        self.assertEqual(removed_claim["change_type"], "removed_from_current_view")
        self.assertTrue(any(row["asset_kind"] == "networkx_projection" and row["graph_object_id"] == "edge:jon-studio" for row in stale_assets))
        self.assertTrue(any(row["asset_kind"] == "graph_contradiction_update_neighborhood_cache" and row["graph_object_id"] == "claim:1" for row in stale_assets))
        self.assertFalse(manifest["query_indexes_refreshed"])
        self.assertFalse(manifest["visual_review_refreshed"])


if __name__ == "__main__":
    unittest.main()
