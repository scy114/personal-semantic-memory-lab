import argparse
import json
import tempfile
import unittest
from pathlib import Path

from tools.step2.query_runner import WorkspaceAssets, WorkspacePaths, build_graph_context


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def node(node_id: str, label: str, entity_type: str = "person") -> dict:
    return {
        "node_id": node_id,
        "label": label,
        "entity_type": entity_type,
        "entity_quality_hint": "stable",
        "evidence_refs": [f"evidence:{node_id}"],
        "warnings": [],
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
    }


def edge(edge_id: str, source: str, target: str) -> dict:
    return {
        "edge_id": edge_id,
        "source_node_id": source,
        "target_node_id": target,
        "source_label": "Jon",
        "target_label": "dance studio",
        "relation_type": "plans",
        "generic_relation_review_hint": "not_generic",
        "weight": 1.0,
        "evidence_count": 1,
        "evidence_refs": [f"evidence:{edge_id}"],
        "warnings": [],
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
    }


def assets_for(workspace: Path, *, legacy_nodes: list[dict] | None = None, legacy_edges: list[dict] | None = None) -> WorkspaceAssets:
    paths = WorkspacePaths(
        workspace=workspace,
        manifest=workspace / "manifest.yaml",
        reviewed_units=workspace / "portrait" / "reviewed_units.jsonl",
        portrait_json=workspace / "portrait" / "current_portrait.json",
        portrait_md=workspace / "portrait" / "current_portrait.md",
        graph_nodes=workspace / "graph" / "nodes.jsonl",
        graph_edges=workspace / "graph" / "edges.jsonl",
        base_packet=workspace / "packets" / "base_assistance_packet.json",
        evidence=None,
        index_manifest=None,
        index_entries=None,
        index_vectors=None,
    )
    return WorkspaceAssets(
        paths=paths,
        manifest={},
        reviewed_units=[],
        portrait_json={},
        graph_nodes=legacy_nodes or [],
        graph_edges=legacy_edges or [],
        base_packet={},
        evidence=[],
        index_manifest={},
        index_entries=[],
        vectors=None,
    )


class Step2QueryGraphContextTests(unittest.TestCase):
    def test_auto_prefers_v03_graph_retrieval_over_legacy_anchor(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            graph_dir = workspace / "graph_v03_consolidation_provider_80"
            write_jsonl(graph_dir / "graph_nodes_table.jsonl", [node("n1", "Jon"), node("n2", "dance studio", "project")])
            write_jsonl(graph_dir / "graph_edges_table.jsonl", [edge("e1", "n1", "n2")])
            write_jsonl(graph_dir / "graph_claims_table.jsonl", [])
            write_jsonl(graph_dir / "evidence_links.jsonl", [])

            args = argparse.Namespace(
                graph_context_mode="auto",
                v03_graph_dir=None,
                v03_graph_projection="review_aware_graph",
                graph_top_k=8,
            )
            question = {"query_id": "q1", "question": "Jon 和 dance studio 有什么关系？", "needs_graph": True}
            context, package, status = build_graph_context(question, [], assets_for(workspace), args)

            self.assertEqual(status["mode"], "v03_graph_query_retriever")
            self.assertIsNotNone(package)
            self.assertTrue(context)
            self.assertTrue(all(row["graph_is_not_proof"] is True for row in context))
            self.assertTrue(any(row["kind"] == "v03_ranked_relation" for row in context))

    def test_legacy_anchor_is_deprecated_fallback_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            legacy_node = {"node_id": "old-node", "label": "old graph node", "evidence_refs": ["evidence:old"]}
            args = argparse.Namespace(
                graph_context_mode="auto",
                v03_graph_dir=None,
                v03_graph_projection="review_aware_graph",
                graph_top_k=8,
            )
            question = {"query_id": "q1", "question": "old graph node", "needs_graph": True}
            selected = [{"object_type": "portrait_unit", "object_id": "u1", "graph_refs": ["old-node"]}]
            context, package, status = build_graph_context(question, selected, assets_for(workspace, legacy_nodes=[legacy_node]), args)

            self.assertEqual(status["mode"], "legacy_anchor_deprecated")
            self.assertIsNone(package)
            self.assertTrue(context)
            self.assertIn("deprecated_legacy_anchor_graph_context", context[0]["warnings"])


if __name__ == "__main__":
    unittest.main()
