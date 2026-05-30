import json
import shutil
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_jsonl
from tools.graph.graph_query_retriever import discover_graph_dir
from tools.maintenance.graph_current_view_publisher import run_graph_current_view_publish


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def seed_latest_view(source_dir: Path, *, edge_id: str = "edge:1") -> None:
    write_jsonl(source_dir / "graph_nodes_latest_view.jsonl", [{"node_id": "node:jon", "label": "Jon", "graph_is_not_proof": True}])
    write_jsonl(
        source_dir / "graph_edges_latest_view.jsonl",
        [{"edge_id": edge_id, "source_node_id": "node:jon", "target_node_id": "node:studio", "graph_is_not_proof": True}],
    )
    write_jsonl(source_dir / "graph_claims_latest_view.jsonl", [])
    write_jsonl(source_dir / "evidence_links_latest_view.jsonl", [{"link_id": "link:1", "owner_id": edge_id}])
    write_jsonl(source_dir / "graph_nodes_latest_view_excluded.jsonl", [])
    write_jsonl(source_dir / "graph_edges_latest_view_excluded.jsonl", [])
    write_jsonl(source_dir / "graph_claims_latest_view_excluded.jsonl", [])
    write_jsonl(source_dir / "evidence_links_latest_view_excluded.jsonl", [])
    write_json(source_dir / "graph_candidate_latest_view_manifest.json", {"schema_version": "test.latest"})


class V04GraphCurrentViewPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_v04_graph_current_view_publish_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def test_publish_latest_view_to_stable_graph_current_without_truth_write(self):
        source_dir = self.workspace / "graph_candidate_latest_view_incremental"
        seed_latest_view(source_dir)

        manifest = run_graph_current_view_publish(workspace=self.workspace, source_latest_view_dir=source_dir)
        current_dir = self.workspace / "graph_current"

        self.assertEqual(Path(manifest["current_graph_dir"]), current_dir.resolve())
        self.assertTrue((current_dir / "graph_current_manifest.json").exists())
        self.assertTrue((current_dir / "graph_current_report.md").exists())
        self.assertTrue((current_dir / "latest_views" / "graph_edges_latest_view.jsonl").exists())
        self.assertTrue((current_dir / "dependency_map" / "graph_dependency_map.jsonl").exists())
        self.assertTrue((current_dir / "query_index_invalidation" / "changed_graph_units.jsonl").exists())
        self.assertTrue(Path(manifest["maintenance_artifacts"]["incremental_visual_review_manifest"]).exists())
        self.assertTrue(Path(manifest["maintenance_artifacts"]["incremental_visual_review_html"]).exists())
        self.assertIn("dependency_map", manifest["maintenance_artifacts"])
        self.assertIn("changed_graph_units", manifest["maintenance_artifacts"])
        self.assertEqual(len(read_jsonl(current_dir / "graph_nodes_latest_view.jsonl")), 1)
        self.assertEqual(manifest["counts"]["edges"]["active_count"], 1)
        self.assertEqual(manifest["maintenance_counts"]["changed_graph_units"], 3)
        self.assertGreaterEqual(manifest["maintenance_counts"]["visual_review_slices"], 1)
        self.assertTrue(manifest["publish_executed"])
        self.assertTrue(manifest["query_default_ready"])
        self.assertTrue(manifest["graph_is_not_proof"])
        self.assertFalse(manifest["graph_truth_written"])
        self.assertFalse(manifest["durable_writes_executed"])
        self.assertFalse((self.workspace / "graph" / "edges.jsonl").exists())
        self.assertFalse((current_dir / "graph_edges_table.jsonl").exists())

    def test_republish_archives_previous_current_view(self):
        first_source = self.workspace / "first_latest"
        second_source = self.workspace / "second_latest"
        seed_latest_view(first_source, edge_id="edge:first")
        seed_latest_view(second_source, edge_id="edge:second")

        run_graph_current_view_publish(workspace=self.workspace, source_latest_view_dir=first_source)
        second_manifest = run_graph_current_view_publish(workspace=self.workspace, source_latest_view_dir=second_source)

        self.assertTrue(second_manifest["previous_current_archive_dir"])
        self.assertTrue(Path(second_manifest["previous_current_archive_dir"]).exists())
        latest_edges = read_jsonl(self.workspace / "graph_current" / "graph_edges_latest_view.jsonl")
        self.assertEqual(latest_edges[0]["edge_id"], "edge:second")

    def test_query_discovery_prefers_published_graph_current(self):
        current_source = self.workspace / "latest_for_current"
        seed_latest_view(current_source, edge_id="edge:current")
        run_graph_current_view_publish(workspace=self.workspace, source_latest_view_dir=current_source)

        old_graph = self.workspace / "graph_v03_consolidation_provider_80"
        write_jsonl(old_graph / "graph_nodes_table.jsonl", [{"node_id": "node:old", "label": "Old"}])
        write_jsonl(old_graph / "graph_edges_table.jsonl", [{"edge_id": "edge:old"}])

        self.assertEqual(discover_graph_dir(self.workspace, None), (self.workspace / "graph_current").resolve())


if __name__ == "__main__":
    unittest.main()
