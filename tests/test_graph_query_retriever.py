import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_query_retriever import rerank_graph_items, retrieve_graph_query_package


class FakeEmbedder:
    def encode(self, text: str):
        lower = text.lower()
        return [
            1.0 if "marley" in lower or "flooring" in lower or "floor" in lower else 0.0,
            1.0 if "dance studio" in lower else 0.0,
            1.0 if "festival" in lower else 0.0,
        ]


class FakeReranker:
    def predict(self, pairs, **_kwargs):
        scores = []
        for _query, doc in pairs:
            text = doc.lower()
            scores.append(2.0 if "marley flooring" in text else 0.1)
        return scores


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
        "description": label,
        "entity_type": entity_type,
        "entity_quality_hint": "stable",
        "evidence_refs": [f"evidence:{node_id}"],
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
        "source_label": "Jon" if source == "n1" else "dance studio",
        "target_label": "dance studio" if target == "n2" else "festival",
        "relation_type": relation_type,
        "raw_relation_types": [relation_type],
        "generic_relation_review_hint": "not_generic",
        "description": "Jon plans the dance studio.",
        "weight": 1.0,
        "evidence_count": 1,
        "evidence_refs": [f"evidence:{edge_id}"],
        "warnings": [],
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
    }


class GraphQueryRetrieverTests(unittest.TestCase):
    def test_rrf_retriever_combines_lexical_seed_and_neighborhood_branches(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_dir = Path(temp_dir) / "graph"
            write_jsonl(
                graph_dir / "graph_nodes_table.jsonl",
                [
                    node("n1", "Jon"),
                    node("n2", "dance studio", "project"),
                    node("n3", "festival", "event"),
                ],
            )
            write_jsonl(
                graph_dir / "graph_edges_table.jsonl",
                [
                    edge("e1", "n1", "n2", "plans"),
                    edge("e2", "n2", "n3", "prepares_for"),
                ],
            )
            write_jsonl(graph_dir / "graph_claims_table.jsonl", [])

            question = {
                "query_id": "q1",
                "question": "What is Jon's relationship to the dance studio?",
                "seed_terms": ["Jon", "dance studio"],
            }
            package, context, status = retrieve_graph_query_package(
                question,
                [],
                graph_dir=graph_dir,
                projection="review_aware_graph",
                top_k=8,
            )

            self.assertEqual(status["mode"], "v03_graph_query_retriever")
            self.assertEqual(package["retrieval_policy"]["fusion"], "reciprocal_rank_fusion")
            self.assertTrue(package["ranked_items"])
            self.assertTrue(any("seed_node_match" in row["branches"] for row in package["ranked_items"]))
            self.assertTrue(any("graph_neighborhood_bfs_1hop" in row["branches"] for row in package["ranked_items"]))
            self.assertTrue(any(row["kind"] == "v03_ranked_relation" for row in context))
            self.assertTrue(all(row["graph_is_not_proof"] is True for row in context))

    def test_optional_graph_unit_embedding_branch_participates_in_rrf(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_dir = Path(temp_dir) / "graph"
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
                    {
                        **edge("e1", "n3", "n2", "supports"),
                        "source_label": "Marley flooring",
                        "target_label": "dance studio",
                        "description": "Marley flooring supports the dance studio project.",
                    },
                    edge("e2", "n1", "n2", "plans"),
                ],
            )
            write_jsonl(graph_dir / "graph_claims_table.jsonl", [])

            package, _context, status = retrieve_graph_query_package(
                {
                    "query_id": "q_embed",
                    "question": "Which movement floor material is tied to the studio?",
                    "seed_terms": ["studio"],
                },
                [],
                graph_dir=graph_dir,
                projection="review_aware_graph",
                top_k=8,
                graph_unit_embedder=FakeEmbedder(),
            )

            self.assertEqual(status["status"], "available")
            self.assertIn("embedding_graph_units", package["branches"])
            self.assertEqual(package["retrieval_policy"]["embedding_branch"], "live_graph_unit_embeddings")
            self.assertTrue(any("embedding_graph_units" in row["branches"] for row in package["ranked_items"]))

    def test_community_report_branch_participates_when_profile_assets_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            graph_dir = workspace / "graph_v03_consolidation_provider_80"
            profile_dir = workspace / "provider_backed_profiles"
            write_jsonl(
                graph_dir / "graph_nodes_table.jsonl",
                [
                    node("n1", "Jon"),
                    node("n2", "dance studio", "project"),
                    node("n3", "Marley flooring", "material"),
                ],
            )
            write_jsonl(graph_dir / "graph_edges_table.jsonl", [edge("e1", "n1", "n2", "plans")])
            write_jsonl(graph_dir / "graph_claims_table.jsonl", [])
            write_jsonl(
                profile_dir / "graph_community_reports.jsonl",
                [
                    {
                        "report_id": "cr1",
                        "community_id": "c1",
                        "title": "Jon dance studio materials",
                        "summary": "Jon's dance studio community includes Marley flooring.",
                        "full_content": "Jon plans the dance studio and discusses Marley flooring.",
                        "activation_quality": "clean_candidate",
                        "source_node_refs": ["n1", "n2", "n3"],
                        "source_edge_refs": ["e1"],
                        "evidence_refs": ["evidence:e1"],
                        "warnings": [],
                        "graph_is_not_proof": True,
                        "support_status": "not_checked",
                        "write_permission": False,
                    }
                ],
            )

            package, context, _status = retrieve_graph_query_package(
                {
                    "query_id": "q_community",
                    "question": "What community connects Jon, studio, and Marley flooring?",
                    "seed_terms": ["Jon", "studio"],
                },
                [],
                graph_dir=graph_dir,
                profile_dir=profile_dir,
                projection="review_aware_graph",
                top_k=8,
            )

            self.assertIn("community_report_match", package["branches"])
            self.assertEqual(package["retrieval_policy"]["community_report_branch"], "enabled")
            self.assertEqual(package["retrieval_policy"]["profile_dir"], str(profile_dir.resolve()))
            self.assertTrue(any("community_report_match" in row["branches"] for row in package["ranked_items"]))
            self.assertTrue(any(row["kind"] == "v03_ranked_community_report" for row in context))

    def test_community_report_branch_can_be_disabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            graph_dir = workspace / "graph_v03_consolidation_provider_80"
            write_jsonl(graph_dir / "graph_nodes_table.jsonl", [node("n1", "Jon"), node("n2", "dance studio", "project")])
            write_jsonl(graph_dir / "graph_edges_table.jsonl", [edge("e1", "n1", "n2", "plans")])
            write_jsonl(graph_dir / "graph_claims_table.jsonl", [])
            write_jsonl(
                workspace / "graph_v03_profile_communities" / "graph_community_reports.jsonl",
                [
                    {
                        "report_id": "cr1",
                        "community_id": "c1",
                        "title": "Jon dance studio",
                        "summary": "Jon's dance studio community.",
                        "full_content": "Jon plans the dance studio.",
                        "activation_quality": "clean_candidate",
                        "source_node_refs": ["n1", "n2"],
                        "source_edge_refs": ["e1"],
                        "evidence_refs": ["evidence:e1"],
                        "warnings": [],
                        "graph_is_not_proof": True,
                        "support_status": "not_checked",
                        "write_permission": False,
                    }
                ],
            )

            package, context, _status = retrieve_graph_query_package(
                {"query_id": "q_no_community", "question": "What connects Jon and the dance studio?"},
                [],
                graph_dir=graph_dir,
                projection="review_aware_graph",
                top_k=8,
                graph_community_mode="disabled",
            )

            self.assertNotIn("community_report_match", package["branches"])
            self.assertEqual(package["retrieval_policy"]["community_report_branch"], "disabled")
            self.assertEqual(package["retrieval_policy"]["graph_community_mode"], "disabled")
            self.assertFalse(any(row["kind"] == "v03_ranked_community_report" for row in context))

    def test_community_report_assist_overlay_is_used_when_provided(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            graph_dir = workspace / "graph_v03_consolidation_provider_80"
            profile_dir = workspace / "graph_v03_profile_communities"
            assists_dir = workspace / "graph_v03_community_report_assists"
            write_jsonl(graph_dir / "graph_nodes_table.jsonl", [node("n1", "Jon"), node("n2", "dance studio", "project")])
            write_jsonl(graph_dir / "graph_edges_table.jsonl", [edge("e1", "n1", "n2", "plans")])
            write_jsonl(graph_dir / "graph_claims_table.jsonl", [])
            write_jsonl(
                profile_dir / "graph_community_reports.jsonl",
                [
                    {
                        "report_id": "cr1",
                        "community_id": "c1",
                        "title": "extractive studio",
                        "summary": "extractive studio summary",
                        "full_content": "Jon plans the dance studio.",
                        "activation_quality": "clean_candidate",
                        "source_node_refs": ["n1", "n2"],
                        "source_edge_refs": ["e1"],
                        "evidence_refs": ["evidence:e1"],
                        "warnings": [],
                        "graph_is_not_proof": True,
                        "support_status": "not_checked",
                        "write_permission": False,
                    }
                ],
            )
            write_jsonl(
                assists_dir / "graph_community_report_assists.jsonl",
                [
                    {
                        "report_id": "cr1",
                        "community_id": "c1",
                        "title": "assisted studio planning",
                        "summary": "assisted studio planning summary",
                        "full_content": "Assisted retrieval guidance for studio planning.",
                        "activation_quality": "clean_candidate",
                        "source_node_refs": ["n1", "n2"],
                        "source_edge_refs": ["e1"],
                        "evidence_refs": ["evidence:e1"],
                        "provider_report_status": "community_report_candidate",
                        "provider": "external_jsonl",
                        "warnings": ["provider_report_is_activation_context_not_proof"],
                        "graph_is_not_proof": True,
                        "support_status": "not_checked",
                        "write_permission": False,
                    }
                ],
            )

            package, context, _status = retrieve_graph_query_package(
                {"query_id": "q_assist", "question": "What studio planning community is active?"},
                [],
                graph_dir=graph_dir,
                profile_dir=profile_dir,
                community_assists_path=assists_dir,
                projection="review_aware_graph",
                top_k=8,
            )

            community_rows = [row for row in context if row["kind"] == "v03_ranked_community_report"]
            self.assertEqual(package["retrieval_policy"]["community_assists_path"], str(assists_dir.resolve()))
            self.assertTrue(community_rows)
            self.assertEqual(community_rows[0]["community_title"], "assisted studio planning")
            self.assertIn("assisted studio planning summary", community_rows[0]["summary"])

    def test_paths_through_weak_reference_nodes_are_quarantined_as_noisy_navigation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            graph_dir = Path(temp_dir) / "graph"
            weak_node = node("n5", "this", "unknown")
            weak_node["entity_quality_hint"] = "context_dependent"
            write_jsonl(
                graph_dir / "graph_nodes_table.jsonl",
                [
                    node("n1", "Jon"),
                    node("n4", "Paris", "place"),
                    weak_node,
                ],
            )
            write_jsonl(
                graph_dir / "graph_edges_table.jsonl",
                [
                    {
                        **edge("e3", "n1", "n5", "related_to_generic"),
                        "source_label": "Jon",
                        "target_label": "this",
                        "generic_relation_review_hint": "needs_entity_resolution",
                    },
                    {
                        **edge("e4", "n5", "n4", "related_to_generic"),
                        "source_label": "this",
                        "target_label": "Paris",
                        "generic_relation_review_hint": "needs_entity_resolution",
                    },
                ],
            )
            write_jsonl(graph_dir / "graph_claims_table.jsonl", [])

            package, context, _status = retrieve_graph_query_package(
                {
                    "query_id": "q2",
                    "question": "How is Jon connected to Paris?",
                    "seed_terms": ["Jon", "Paris"],
                },
                [],
                graph_dir=graph_dir,
                projection="review_aware_graph",
                top_k=8,
            )

            clean_paths = package["graph_branch"]["evidence_paths"]
            noisy_paths = package["graph_branch"]["noisy_navigation_paths"]
            self.assertFalse(any("this" in row.get("path_labels", []) for row in clean_paths))
            self.assertTrue(any("this" in row.get("path_labels", []) for row in noisy_paths))
            self.assertTrue(all(row["path_quality_status"] == "noisy_navigation" for row in noisy_paths))
            self.assertFalse(any(row["kind"] == "v03_evidence_path" and "this" in row["summary"] for row in context))

    def test_feature_rerank_prefers_evidence_backed_specific_relation_over_generic_node(self):
        items = [
            {
                "unit_id": "node:n1",
                "kind": "node",
                "object_id": "n1",
                "text": "this",
                "rrf_score": 0.10,
                "branch_count": 1,
                "branches": ["seed_node_match"],
                "evidence_refs": [],
                "warnings": ["graph_retrieval_item_has_no_evidence_refs"],
                "payload": {"node_id": "n1", "label": "this", "entity_quality_hint": "context_dependent"},
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            },
            {
                "unit_id": "edge:e1",
                "kind": "edge",
                "object_id": "e1",
                "text": "Jon plans dance studio",
                "rrf_score": 0.09,
                "branch_count": 2,
                "branches": ["bm25_graph_units", "graph_neighborhood_bfs_1hop"],
                "evidence_refs": ["evidence:e1"],
                "warnings": [],
                "payload": {"edge_id": "e1", "relation_type": "plans", "generic_relation_review_hint": "not_generic"},
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            },
        ]

        ranked, status = rerank_graph_items("How is Jon related to the studio?", items, mode="feature")

        self.assertTrue(status["used"])
        self.assertEqual(ranked[0]["unit_id"], "edge:e1")
        self.assertEqual(ranked[0]["graph_rerank_policy"], "feature_quality_blend_v0.1")
        self.assertIn("pre_rerank_rrf_score", ranked[0])

    def test_cross_encoder_rerank_blends_with_feature_signal(self):
        items = [
            {
                "unit_id": "edge:e1",
                "kind": "edge",
                "object_id": "e1",
                "text": "Jon plans dance studio",
                "rrf_score": 0.10,
                "branch_count": 2,
                "branches": ["bm25_graph_units", "graph_neighborhood_bfs_1hop"],
                "evidence_refs": ["evidence:e1"],
                "warnings": [],
                "payload": {"edge_id": "e1", "source_label": "Jon", "relation_type": "plans", "target_label": "dance studio", "generic_relation_review_hint": "not_generic"},
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            },
            {
                "unit_id": "edge:e2",
                "kind": "edge",
                "object_id": "e2",
                "text": "Marley flooring supports dance studio",
                "rrf_score": 0.09,
                "branch_count": 1,
                "branches": ["embedding_graph_units"],
                "evidence_refs": ["evidence:e2"],
                "warnings": [],
                "payload": {"edge_id": "e2", "source_label": "Marley flooring", "relation_type": "supports", "target_label": "dance studio", "generic_relation_review_hint": "not_generic"},
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            },
        ]

        ranked, status = rerank_graph_items(
            "Which floor material is tied to the studio?",
            items,
            mode="cross_encoder",
            reranker=FakeReranker(),
            alpha=0.7,
        )

        self.assertEqual(status["model_blend"], "cross_encoder_feature_blend_v0.1")
        self.assertEqual(ranked[0]["unit_id"], "edge:e2")
        self.assertIn("cross_encoder_rerank_score", ranked[0])
        self.assertEqual(ranked[0]["graph_rerank_policy"], "cross_encoder_feature_blend_v0.1")


if __name__ == "__main__":
    unittest.main()
