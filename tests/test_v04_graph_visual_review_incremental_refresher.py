import json
import shutil
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_jsonl
from tools.maintenance.graph_visual_review_incremental_refresher import run_graph_visual_review_incremental_refresh


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def seed_graph_current(graph_dir: Path) -> None:
    write_jsonl(
        graph_dir / "graph_nodes_latest_view.jsonl",
        [
            {
                "node_id": "node:lucilla",
                "label": "Lucilla",
                "entity_quality_hint": "stable",
                "evidence_refs": ["evidence:1"],
                "graph_is_not_proof": True,
            },
            {
                "node_id": "node:library",
                "label": "the library",
                "entity_quality_hint": "stable",
                "evidence_refs": ["evidence:1"],
                "graph_is_not_proof": True,
            },
        ],
    )
    write_jsonl(
        graph_dir / "graph_edges_latest_view.jsonl",
        [
            {
                "edge_id": "edge:lucilla-library",
                "source_node_id": "node:lucilla",
                "target_node_id": "node:library",
                "source_label": "Lucilla",
                "target_label": "the library",
                "relation_type": "visits",
                "generic_relation_review_hint": "not_generic",
                "evidence_refs": ["evidence:1"],
                "graph_is_not_proof": True,
            }
        ],
    )
    write_jsonl(graph_dir / "graph_claims_latest_view.jsonl", [])
    write_jsonl(
        graph_dir / "evidence_links_latest_view.jsonl",
        [
            {
                "link_id": "link:lucilla-library",
                "owner_kind": "edge",
                "owner_id": "edge:lucilla-library",
                "evidence_ref": "evidence:1",
                "graph_is_not_proof": True,
            }
        ],
    )
    write_jsonl(graph_dir / "graph_nodes_latest_view_excluded.jsonl", [])
    write_jsonl(graph_dir / "graph_edges_latest_view_excluded.jsonl", [])
    write_jsonl(graph_dir / "graph_claims_latest_view_excluded.jsonl", [])
    write_jsonl(graph_dir / "evidence_links_latest_view_excluded.jsonl", [])
    write_jsonl(graph_dir / "dependency_map" / "graph_dependency_map.jsonl", [])


class V04GraphVisualReviewIncrementalRefresherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_v04_graph_visual_review_incremental_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def test_writes_no_change_manifest_and_html_without_slices(self):
        graph_dir = self.workspace / "graph_current"
        seed_graph_current(graph_dir)
        write_jsonl(graph_dir / "query_index_invalidation" / "changed_graph_units.jsonl", [])

        manifest = run_graph_visual_review_incremental_refresh(workspace=self.workspace, graph_current_dir=graph_dir)

        self.assertTrue(Path(manifest["outputs"]["manifest"]).exists())
        self.assertTrue(Path(manifest["outputs"]["html_index"]).exists())
        self.assertEqual(manifest["counts"]["changed_graph_units"], 0)
        self.assertEqual(manifest["counts"]["slices"], 0)
        self.assertTrue(manifest["counts"]["no_changes_detected"])
        self.assertFalse(manifest["policies"]["visual_review_refreshed"])
        self.assertFalse(manifest["graph_truth_written"])

    def test_changed_edge_generates_bounded_endpoint_slice(self):
        graph_dir = self.workspace / "graph_current"
        seed_graph_current(graph_dir)
        write_jsonl(
            graph_dir / "query_index_invalidation" / "changed_graph_units.jsonl",
            [
                {
                    "graph_object_kind": "edge",
                    "graph_object_id": "edge:lucilla-library",
                    "change_type": "changed",
                    "current_present": True,
                    "previous_present": True,
                    "graph_is_not_proof": True,
                }
            ],
        )

        manifest = run_graph_visual_review_incremental_refresh(workspace=self.workspace, graph_current_dir=graph_dir)
        slices = read_jsonl(Path(manifest["outputs"]["slices"]))

        self.assertEqual(manifest["counts"]["changed_graph_units"], 1)
        self.assertEqual(manifest["counts"]["slices"], 1)
        self.assertTrue(manifest["policies"]["visual_review_refreshed"])
        self.assertEqual(slices[0]["slice_kind"], "incremental_changed_edge")
        self.assertEqual(slices[0]["node_count"], 2)
        self.assertEqual(slices[0]["edge_count"], 1)
        self.assertIn("evidence:1", slices[0]["evidence_refs"])
        self.assertTrue(Path(slices[0]["json_path"]).exists())
        self.assertTrue(Path(slices[0]["graphml_path"]).exists())


if __name__ == "__main__":
    unittest.main()
