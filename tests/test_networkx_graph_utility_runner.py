import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_json, read_jsonl
from tools.graph.networkx_graph_utility_runner import run_networkx_utility


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def node(node_id: str, label: str, quality: str = "stable", entity_type: str = "person") -> dict:
    return {
        "node_id": node_id,
        "label": label,
        "entity_type": entity_type,
        "entity_quality_hint": quality,
        "entity_quality_reasons": ["test"],
        "evidence_refs": [f"evidence:{node_id}"],
        "raw_backpointer_refs": [f"raw:{node_id}"],
        "warnings": [],
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
    }


def edge(edge_id: str, source: str, target: str, relation_type: str = "supports", generic_hint: str = "not_generic") -> dict:
    return {
        "edge_id": edge_id,
        "source_node_id": source,
        "target_node_id": target,
        "source_label": source,
        "target_label": target,
        "relation_type": relation_type,
        "raw_relation_types": [relation_type],
        "generic_relation_review_hint": generic_hint,
        "generic_relation_review_reasons": ["test"] if generic_hint != "not_generic" else [],
        "weight": 1.0,
        "evidence_count": 1,
        "evidence_refs": [f"evidence:{edge_id}"],
        "raw_backpointer_refs": [f"raw:{edge_id}"],
        "warnings": [],
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
    }


class NetworkXGraphUtilityRunnerTests(unittest.TestCase):
    def test_runs_three_projections_and_keeps_claims_out_of_topology(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            graph_dir = workspace / "graph_v03_consolidation_provider_80"
            output_dir = workspace / "graph_v03_networkx_utility"
            write_jsonl(
                graph_dir / "graph_nodes_table.jsonl",
                [
                    node("n1", "Jon"),
                    node("n2", "Gina"),
                    node("n3", "dance studio", entity_type="project"),
                    node("n4", "dancing", quality="review_required", entity_type="concept"),
                    node("n5", "our goals", quality="generic_fragment", entity_type="concept"),
                ],
            )
            write_jsonl(
                graph_dir / "graph_edges_table.jsonl",
                [
                    edge("e1", "n1", "n3", "plans"),
                    edge("e2", "n2", "n1", "supports"),
                    edge("e3", "n4", "n3", "related_to_generic", "low_graph_value"),
                    edge("e4", "n1", "n5", "related_to_generic", "needs_attribution_review"),
                ],
            )
            write_jsonl(
                graph_dir / "graph_claims_table.jsonl",
                [{"claim_id": "c1", "subject_node_id": "n1", "candidate_text": "Jon has a plan."}],
            )
            write_jsonl(graph_dir / "evidence_links.jsonl", [])
            write_json(graph_dir / "graph_consolidation_manifest.json", {"schema_version": "test"})

            manifest = run_networkx_utility(workspace, graph_dir=graph_dir, output_dir=output_dir)

            projection_stats = read_jsonl(output_dir / "graph_projection_stats.jsonl")
            algorithm_rows = read_jsonl(output_dir / "graph_algorithm_rows.jsonl")
            saved_manifest = read_json(output_dir / "graph_projection_manifest.json")

            by_projection = {row["projection"]: row for row in projection_stats}
            self.assertEqual(set(by_projection), {"full_candidate_graph", "review_aware_graph", "stable_core_graph"})
            self.assertEqual(by_projection["full_candidate_graph"]["node_count"], 5)
            self.assertEqual(by_projection["review_aware_graph"]["node_count"], 4)
            self.assertEqual(by_projection["review_aware_graph"]["edge_count"], 3)
            self.assertEqual(by_projection["stable_core_graph"]["node_count"], 3)
            self.assertEqual(by_projection["stable_core_graph"]["edge_count"], 2)
            self.assertTrue(saved_manifest["policies"]["claims_loaded_as_side_table_not_topology"])
            self.assertEqual(saved_manifest["input_counts"]["claims"], 1)
            self.assertTrue((output_dir / "full_candidate_graph.graphml").exists())
            self.assertTrue((output_dir / "graph_algorithm_report.md").exists())
            self.assertTrue(all(row["graph_is_not_proof"] is True for row in algorithm_rows))
            self.assertTrue(any(row["algorithm"] == "pagerank" for row in algorithm_rows))


if __name__ == "__main__":
    unittest.main()
