import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_visual_review_bundle_builder import run_visual_review_bundle


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def node(node_id: str, label: str, quality: str = "stable") -> dict:
    return {
        "node_id": node_id,
        "label": label,
        "entity_type": "person" if label in {"Jon", "Gina"} else "project",
        "entity_quality_hint": quality,
        "evidence_refs": [f"evidence:{node_id}"],
        "raw_backpointer_refs": [f"raw:{node_id}"],
        "source_text_quotes": [f"quote for {node_id}"],
        "warnings": [],
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }


def edge(edge_id: str, source: str, target: str, relation_type: str = "supports", generic_hint: str = "not_generic") -> dict:
    return {
        "edge_id": edge_id,
        "source_node_id": source,
        "target_node_id": target,
        "source_label": source,
        "target_label": target,
        "relation_type": relation_type,
        "generic_relation_review_hint": generic_hint,
        "evidence_refs": [f"evidence:{edge_id}"],
        "raw_backpointer_refs": [f"raw:{edge_id}"],
        "source_text_quotes": [f"source quote for {edge_id}"],
        "source_refs": ["fixture-source"],
        "source_perspective": "fixture speaker",
        "attribution_status": "explicit",
        "confidence_hint": "medium",
        "evidence_count": 1,
        "weight": 1.0,
        "warnings": [],
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }


class GraphVisualReviewBundleBuilderTests(unittest.TestCase):
    def test_builder_exports_bounded_graphml_and_json_review_slices(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            graph_dir = workspace / "graph_v03_consolidation_provider_80"
            networkx_dir = workspace / "graph_v03_networkx_utility_provider_80"
            profile_dir = workspace / "graph_v03_profile_communities"
            output_dir = workspace / "graph_v031_visual_review"

            write_jsonl(
                graph_dir / "graph_nodes_table.jsonl",
                [
                    node("n1", "Jon"),
                    node("n2", "Gina"),
                    node("n3", "dance studio"),
                    node("n4", "our goals", quality="generic_fragment"),
                ],
            )
            write_jsonl(
                graph_dir / "graph_edges_table.jsonl",
                [
                    edge("e1", "n1", "n2", "supports"),
                    edge("e2", "n1", "n3", "plans"),
                    edge("e3", "n1", "n4", "related_to_generic", "needs_attribution_review"),
                ],
            )
            write_jsonl(
                networkx_dir / "graph_algorithm_rows.jsonl",
                [
                    {
                        "algorithm": "shortest_evidence_path",
                        "projection": "review_aware_graph",
                        "subject_id": "n1->n3",
                        "subject_label": "Jon -> dance studio",
                        "payload": {
                            "path_node_ids": ["n1", "n3"],
                            "evidence_path": [
                                {
                                    "source_node_id": "n1",
                                    "target_node_id": "n3",
                                    "relation_type": "plans",
                                    "evidence_refs": ["evidence:e2"],
                                }
                            ],
                        },
                        "graph_is_not_proof": True,
                        "support_status": "not_checked",
                        "write_permission": False,
                    }
                ],
            )
            write_jsonl(
                profile_dir / "graph_communities.jsonl",
                [
                    {
                        "community_id": "c1",
                        "title": "Jon studio",
                        "entity_ids": ["n1", "n2", "n3", "n4"],
                        "activation_score": 10.0,
                        "activation_quality": "review_heavy",
                        "generic_relation_ratio": 0.25,
                        "weak_node_ratio": 0.25,
                        "graph_is_not_proof": True,
                        "support_status": "not_checked",
                        "write_permission": False,
                    }
                ],
            )
            write_jsonl(profile_dir / "graph_community_reports.jsonl", [])

            manifest = run_visual_review_bundle(
                workspace,
                graph_dir=graph_dir,
                networkx_dir=networkx_dir,
                profile_dir=profile_dir,
                output_dir=output_dir,
                top_ego_count=1,
                top_community_count=1,
                top_evidence_path_count=1,
                max_nodes_per_slice=10,
                max_edges_per_slice=10,
            )

            self.assertEqual(manifest["counts"]["slices"], 5)
            self.assertTrue((output_dir / "graph_visual_review_manifest.json").exists())
            self.assertTrue((output_dir / "graph_visual_review_report.md").exists())
            self.assertTrue((output_dir / "graph_visual_review_index.html").exists())
            html_text = (output_dir / "graph_visual_review_index.html").read_text(encoding="utf-8")
            self.assertIn("Cytoscape.js", html_text)
            self.assertIn("图可视化审计", html_text)
            self.assertIn("用于人工检查图候选", html_text)
            self.assertIn("显示边标签", html_text)
            self.assertIn("选中对象", html_text)
            self.assertIn("data-edge-id", html_text)
            self.assertIn("renderEdgeDetails", html_text)
            self.assertIn("原文 quote", html_text)

            slices = [
                json.loads(line)
                for line in (output_dir / "graph_visual_review_slices.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            kinds = {row["slice_kind"] for row in slices}
            self.assertIn("ego_network", kinds)
            self.assertIn("community_cluster", kinds)
            self.assertIn("noisy_generic_edges", kinds)
            self.assertIn("weak_or_isolated_nodes", kinds)
            self.assertIn("evidence_path", kinds)
            self.assertTrue(all(row["graph_is_not_proof"] is True for row in slices))
            self.assertTrue(all(row["support_status"] == "not_checked" for row in slices))
            self.assertTrue(all(row["write_permission"] is False for row in slices))
            self.assertTrue(all(row["visualization_is_audit_support_only"] is True for row in slices))
            self.assertTrue(all(Path(row["graphml_path"]).exists() for row in slices))
            self.assertTrue(all(Path(row["json_path"]).exists() for row in slices))

            noisy = [row for row in slices if row["slice_kind"] == "noisy_generic_edges"][0]
            self.assertEqual(noisy["edge_count"], 1)
            self.assertIn("related_to_generic", noisy["relation_type_counts"])
            ego_json = json.loads(Path([row for row in slices if row["slice_kind"] == "ego_network"][0]["json_path"]).read_text(encoding="utf-8"))
            self.assertTrue(ego_json["elements"]["visual_edge_groups"])
            first_edge = ego_json["elements"]["edges"][0]["data"]
            self.assertIn("visual_edge_group_count", first_edge)
            self.assertIn("source_text_quotes", first_edge)
            self.assertIn("raw_backpointer_refs", first_edge)


if __name__ == "__main__":
    unittest.main()
