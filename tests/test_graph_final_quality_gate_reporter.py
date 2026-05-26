import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_final_quality_gate_reporter import compute_quality_gate, write_jsonl


def node(node_id: str, label: str, quality: str = "stable") -> dict:
    return {
        "node_id": node_id,
        "label": label,
        "entity_quality_hint": quality,
        "evidence_refs": ["evidence:1"],
        "raw_backpointer_refs": ["D1:1"],
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
        "warnings": [],
    }


def edge(edge_id: str, source: str, target: str, *, relation_type: str = "plans", evidence: bool = True) -> dict:
    return {
        "edge_id": edge_id,
        "source_node_id": source,
        "target_node_id": target,
        "source_label": "Jon",
        "target_label": "studio",
        "relation_type": relation_type,
        "raw_relation_types": ["associated-with"] if relation_type == "related_to_generic" else ["plans"],
        "generic_relation_review_hint": "needs_schema_extension" if relation_type == "related_to_generic" else "not_generic",
        "evidence_refs": ["evidence:1"] if evidence else [],
        "raw_backpointer_refs": ["D1:1"] if evidence else [],
        "source_text_quotes": ["context quote"],
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
        "warnings": ["context_dependency_warning"] if relation_type == "related_to_generic" else [],
    }


def claim(claim_id: str, subject: str) -> dict:
    return {
        "claim_id": claim_id,
        "subject_node_id": subject,
        "candidate_text": "Jon says he is starting a studio.",
        "evidence_refs": ["evidence:1"],
        "raw_backpointer_refs": ["D1:1"],
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
        "warnings": [],
    }


class GraphFinalQualityGateReporterTests(unittest.TestCase):
    def write_graph(self, graph_dir: Path, *, bad: bool = False) -> None:
        graph_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(graph_dir / "graph_nodes_table.jsonl", [node("n1", "Jon"), node("n2", "studio", "generic_fragment")])
        rows = [edge("e1", "n1", "n2"), edge("e2", "n1", "n2", relation_type="related_to_generic")]
        if bad:
            rows.append(edge("e3", "n1", "missing", evidence=False))
        write_jsonl(graph_dir / "graph_edges_table.jsonl", rows)
        write_jsonl(graph_dir / "graph_claims_table.jsonl", [claim("c1", "n1")])
        write_jsonl(graph_dir / "evidence_links.jsonl", [{"evidence_ref": "evidence:1"}])
        write_jsonl(graph_dir / "graph_merge_decisions.jsonl", [{"decision": "merge"}])
        (graph_dir / "graph_consolidation_manifest.json").write_text(
            json.dumps({"policies": {"graph_is_not_proof": True}, "source_asset_hashes": {"source": "hash"}}),
            encoding="utf-8",
        )

    def test_gate_passes_with_known_warnings_for_reviewable_candidate_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_dir = Path(tmp) / "graph"
            self.write_graph(graph_dir)

            summary, samples, report = compute_quality_gate(graph_dir)

            self.assertEqual(summary["status"], "pass_with_known_warnings")
            self.assertEqual(summary["counts"]["unresolved_edge_endpoint_count"], 0)
            self.assertEqual(summary["counts"]["missing_evidence_edge_count"], 0)
            self.assertTrue(samples)
            self.assertIn("graph utility workflow closure", report)

    def test_gate_blocks_missing_endpoint_or_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            graph_dir = Path(tmp) / "graph"
            self.write_graph(graph_dir, bad=True)

            summary, _samples, _report = compute_quality_gate(graph_dir)

            self.assertEqual(summary["status"], "blocked")
            self.assertGreater(summary["counts"]["unresolved_edge_endpoint_count"], 0)
            self.assertGreater(summary["counts"]["missing_evidence_edge_count"], 0)


if __name__ == "__main__":
    unittest.main()
