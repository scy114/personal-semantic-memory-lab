import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import stable_id
from tools.graph.graph_profile_community_builder import run_builder


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def node(node_id: str, label: str, entity_type: str = "person") -> dict:
    return {
        "node_id": node_id,
        "label": label,
        "description": label,
        "entity_type": entity_type,
        "entity_quality_hint": "stable",
        "evidence_refs": [f"evidence:{node_id}"],
        "raw_backpointer_refs": [f"raw:{node_id}"],
        "warnings": [],
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }


def edge(edge_id: str, source: str, target: str, relation_type: str) -> dict:
    return {
        "edge_id": edge_id,
        "source_node_id": source,
        "target_node_id": target,
        "source_label": "Jon" if source == "n1" else "dance studio",
        "target_label": "dance studio" if target == "n2" else "Marley flooring",
        "relation_type": relation_type,
        "description": f"{source} {relation_type} {target}",
        "evidence_count": 1,
        "evidence_refs": [f"evidence:{edge_id}"],
        "raw_backpointer_refs": [f"raw:{edge_id}"],
        "generic_relation_review_hint": "not_generic",
        "warnings": [],
        "weight": 1.0,
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }


class GraphProfileCommunityBuilderTests(unittest.TestCase):
    def test_builder_writes_profile_and_community_assets_without_truth_claims(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            graph_dir = workspace / "graph_v03_consolidation_provider_80"
            output_dir = workspace / "graph_v03_profile_communities"
            write_jsonl(
                graph_dir / "graph_nodes_table.jsonl",
                [
                    node("n1", "Jon"),
                    node("n2", "dance studio", "project"),
                    node("n3", "Marley flooring", "material"),
                ],
            )
            write_jsonl(
                graph_dir / "graph_edges_table.jsonl",
                [
                    edge("e1", "n1", "n2", "plans"),
                    edge("e2", "n2", "n3", "needs_material"),
                ],
            )
            write_jsonl(graph_dir / "graph_claims_table.jsonl", [])

            manifest = run_builder(workspace, graph_dir=graph_dir, output_dir=output_dir)

            self.assertEqual(manifest["counts"]["entity_profile_cards"], 3)
            self.assertEqual(manifest["counts"]["relation_profile_cards"], 2)
            self.assertGreaterEqual(manifest["counts"]["communities"], 1)
            self.assertTrue((output_dir / "graph_community_reports.jsonl").exists())

            reports = [
                json.loads(line)
                for line in (output_dir / "graph_community_reports.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(reports)
            self.assertTrue(all(row["graph_is_not_proof"] is True for row in reports))
            self.assertTrue(all(row["support_status"] == "not_checked" for row in reports))
            self.assertTrue(all(row["write_permission"] is False for row in reports))
            self.assertTrue(any("evidence:e1" in row["evidence_refs"] for row in reports))

    def test_external_jsonl_provider_can_enhance_community_report_with_validated_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            graph_dir = workspace / "graph_v03_consolidation_provider_80"
            output_dir = workspace / "graph_v03_profile_communities"
            write_jsonl(
                graph_dir / "graph_nodes_table.jsonl",
                [
                    node("n1", "Jon"),
                    node("n2", "dance studio", "project"),
                    node("n3", "Marley flooring", "material"),
                ],
            )
            write_jsonl(
                graph_dir / "graph_edges_table.jsonl",
                [
                    edge("e1", "n1", "n2", "plans"),
                    edge("e2", "n2", "n3", "needs_material"),
                ],
            )
            write_jsonl(graph_dir / "graph_claims_table.jsonl", [])

            community_id = stable_id("graph_community", json.dumps(["review_aware_graph", 1, ["n1", "n2", "n3"]], sort_keys=True))
            proposal_input_id = stable_id("graph_community_report_input", community_id)
            external_outputs = workspace / "community_outputs.jsonl"
            write_jsonl(
                external_outputs,
                [
                    {
                        "proposal_input_id": proposal_input_id,
                        "model_id": "replay-model",
                        "model_output": {
                            "output_kind": "community_report_candidate",
                            "title": "Jon studio materials",
                            "summary": "This community activates Jon's dance studio project and its flooring material needs.",
                            "findings": [
                                {
                                    "summary": "Studio material dependency",
                                    "explanation": "The project is connected to Marley flooring through a material need.",
                                    "evidence_refs": ["evidence:e1", "unknown:evidence"],
                                }
                            ],
                            "retrieval_guidance": ["Use for studio project and flooring queries."],
                            "query_expansion_terms": ["dance studio", "Marley flooring"],
                            "warnings": ["graph_is_not_proof"],
                        },
                    }
                ],
            )

            manifest = run_builder(
                workspace,
                graph_dir=graph_dir,
                output_dir=output_dir,
                community_report_provider="external_jsonl",
                external_model_outputs_path=external_outputs,
                max_provider_reports=1,
            )

            reports = [
                json.loads(line)
                for line in (output_dir / "graph_community_reports.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            enhanced = [row for row in reports if row.get("provider_report_status") == "community_report_candidate"]
            self.assertEqual(manifest["counts"]["provider_report_count"], 1)
            self.assertEqual(len(enhanced), 1)
            self.assertEqual(enhanced[0]["provider"], "external_jsonl")
            self.assertIn("extractive_summary", enhanced[0])
            self.assertIn("evidence:e1", enhanced[0]["provider_selected_evidence_refs"])
            self.assertNotIn("unknown:evidence", enhanced[0]["provider_selected_evidence_refs"])
            self.assertIn("provider_report_referenced_unknown_evidence_refs", enhanced[0]["warnings"])
            self.assertTrue(enhanced[0]["graph_is_not_proof"])
            self.assertFalse(enhanced[0]["write_permission"])


if __name__ == "__main__":
    unittest.main()
