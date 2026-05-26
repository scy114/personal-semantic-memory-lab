import unittest

from tools.step2.s23_answer_context_runner import build_s23_answer_context, render_s23_prompt_context


class S23GraphAnswerContextTests(unittest.TestCase):
    def test_graph_context_is_bucketed_for_answer_use(self):
        graph_context = [
            {
                "kind": "v03_relation_neighbor",
                "summary": "Jon --plans--> dance studio",
                "source_node_refs": ["n1", "n2"],
                "source_edge_refs": ["e1"],
                "evidence_refs": ["evidence:e1"],
                "confidence": "not_generic",
                "warnings": [],
                "retrieval_source": "v03_graph_retrieval",
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            },
            {
                "kind": "v03_relation_neighbor",
                "summary": "Jon --related_to_generic--> studio",
                "source_node_refs": ["n1", "n3"],
                "source_edge_refs": ["e2"],
                "evidence_refs": ["evidence:e2"],
                "confidence": "needs_entity_resolution",
                "warnings": [],
                "retrieval_source": "v03_graph_retrieval",
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            },
            {
                "kind": "v03_evidence_path",
                "summary": "Jon -> dance studio",
                "source_node_refs": ["n1", "n2"],
                "source_edge_refs": ["e1"],
                "evidence_refs": ["evidence:e1"],
                "confidence": "candidate_path",
                "warnings": ["v03_evidence_path_is_navigation_not_support_proof"],
                "retrieval_source": "v03_graph_retrieval",
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            },
        ]
        graph_retrieval_package = {
            "graph_branch": {
                "query_expansion_terms": [{"term": "plans", "count": 1}, {"term": "dance studio", "count": 1}],
                "noisy_navigation_paths": [
                    {
                        "path_labels": ["Jon", "this", "Paris"],
                        "path_node_ids": ["n1", "n4", "n5"],
                        "segments": [{"edge_id": "e3", "evidence_refs": ["evidence:e3"]}],
                        "path_quality_status": "noisy_navigation",
                        "path_quality_reasons": ["unresolved_reference_label"],
                    }
                ],
            }
        }

        context = build_s23_answer_context(
            dual_package={"query": {"question": "What is Jon's relation to the dance studio?"}},
            graph_context=graph_context,
            graph_retrieval_package=graph_retrieval_package,
        )
        graph_answer = context["graph_answer_context"]

        self.assertEqual(len(graph_answer["supported_graph_relations"]), 1)
        self.assertEqual(graph_answer["supported_graph_relations"][0]["summary"], "Jon --plans--> dance studio")
        self.assertEqual(len(graph_answer["cautious_graph_relations"]), 1)
        self.assertEqual(len(graph_answer["evidence_paths"]), 1)
        self.assertEqual(len(graph_answer["noisy_navigation_paths"]), 1)
        self.assertTrue(graph_answer["graph_is_not_proof"])
        prompt = render_s23_prompt_context(context)
        self.assertIn("plans", prompt)
        self.assertIn("Graph audit details omitted", prompt)
        self.assertNotIn("Evidence paths for navigation only", prompt)
        self.assertNotIn("Noisy navigation paths for audit only", prompt)

        why_prompt = render_s23_prompt_context(context, prompt_view="why")
        self.assertIn("Evidence paths for navigation only", why_prompt)
        self.assertNotIn("Noisy navigation paths for audit only", why_prompt)

        audit_prompt = render_s23_prompt_context(context, prompt_view="audit")
        self.assertIn("Evidence paths for navigation only", audit_prompt)
        self.assertIn("Noisy navigation paths for audit only", audit_prompt)


if __name__ == "__main__":
    unittest.main()
