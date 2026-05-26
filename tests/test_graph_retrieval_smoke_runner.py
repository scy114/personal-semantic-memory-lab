import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_json, read_jsonl
from tools.graph.graph_retrieval_smoke_runner import run_graph_retrieval_smoke


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
        "description": label,
        "entity_type": entity_type,
        "entity_quality_hint": quality,
        "entity_quality_reasons": ["test"],
        "evidence_refs": [f"evidence:{node_id}"],
        "raw_backpointer_refs": [f"raw:{node_id}"],
        "source_text_quotes": [label],
        "warnings": [],
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
    }


def edge(edge_id: str, source: str, target: str, relation_type: str = "plans") -> dict:
    return {
        "edge_id": edge_id,
        "source_node_id": source,
        "target_node_id": target,
        "source_label": "Jon" if source == "n1" else source,
        "target_label": "dance studio" if target == "n2" else target,
        "relation_type": relation_type,
        "raw_relation_types": [relation_type],
        "generic_relation_review_hint": "not_generic",
        "generic_relation_review_reasons": [],
        "description": "Jon plans the dance studio.",
        "weight": 1.0,
        "evidence_count": 1,
        "evidence_refs": [f"evidence:{edge_id}"],
        "raw_backpointer_refs": [f"raw:{edge_id}"],
        "source_text_quotes": ["Jon plans a dance studio."],
        "warnings": [],
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
    }


class GraphRetrievalSmokeRunnerTests(unittest.TestCase):
    def test_writes_retrieval_packages_with_lexical_and_graph_branches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            graph_dir = workspace / "graph_v03_consolidation_provider_80"
            output_dir = workspace / "graph_v03_retrieval_smoke"
            write_jsonl(
                graph_dir / "graph_nodes_table.jsonl",
                [
                    node("n1", "Jon"),
                    node("n2", "dance studio", entity_type="project"),
                    node("n3", "festival", entity_type="event"),
                    node("n4", "unclear fragment", quality="review_required", entity_type="unknown"),
                ],
            )
            write_jsonl(
                graph_dir / "graph_edges_table.jsonl",
                [
                    edge("e1", "n1", "n2", "plans"),
                    edge("e2", "n2", "n3", "prepares_for"),
                    edge("e3", "n4", "n2", "related_to_generic"),
                ],
            )
            write_jsonl(
                graph_dir / "graph_claims_table.jsonl",
                [
                    {
                        "claim_id": "c1",
                        "subject_node_id": "n1",
                        "subject_label": "Jon",
                        "candidate_text": "Jon plans a dance studio.",
                        "evidence_refs": ["evidence:c1"],
                        "graph_is_not_proof": True,
                        "write_permission": False,
                        "support_status": "not_checked",
                    }
                ],
            )
            write_jsonl(graph_dir / "evidence_links.jsonl", [])
            write_json(graph_dir / "graph_consolidation_manifest.json", {"schema_version": "test"})

            manifest = run_graph_retrieval_smoke(workspace, graph_dir=graph_dir, output_dir=output_dir)

            rows = read_jsonl(output_dir / "graph_retrieval_smoke.jsonl")
            saved_manifest = read_json(output_dir / "graph_retrieval_manifest.json")

            self.assertEqual(len(rows), 3)
            self.assertTrue(saved_manifest["policies"]["retrieval_smoke_not_final_answer"])
            self.assertTrue(all(row["graph_is_not_proof"] is True for row in rows))
            self.assertTrue(all("lexical_branch" in row and "graph_branch" in row for row in rows))
            first = rows[0]
            self.assertEqual(first["query_id"], "entity_jon_dance_studio")
            self.assertGreaterEqual(len(first["graph_branch"]["matched_seed_nodes"]), 2)
            self.assertTrue(first["graph_branch"]["expanded_neighbors"])
            self.assertTrue(first["graph_branch"]["evidence_paths"])
            self.assertTrue((output_dir / "graph_retrieval_smoke_report.md").exists())


if __name__ == "__main__":
    unittest.main()
