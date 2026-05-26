import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_community_report_llm_assist import run_assist
from tools.graph.graph_construction_packet_builder import stable_id


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class GraphCommunityReportLlmAssistTests(unittest.TestCase):
    def test_external_jsonl_assist_selects_query_activated_community_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            profile_dir = root / "graph_v03_profile_communities"
            query_dir = root / "queries" / "q1"
            community_id = "community:target"
            other_id = "community:other"
            write_jsonl(
                profile_dir / "graph_community_reports.jsonl",
                [
                    {
                        "report_id": "r1",
                        "community_id": community_id,
                        "title": "extractive target",
                        "summary": "extractive target summary",
                        "full_content": "extractive content",
                        "rank": 1.0,
                        "activation_quality": "clean_candidate",
                        "evidence_refs": ["evidence:1"],
                        "source_node_refs": ["n1"],
                        "source_edge_refs": ["e1"],
                        "warnings": [],
                        "graph_is_not_proof": True,
                        "support_status": "not_checked",
                        "write_permission": False,
                    },
                    {
                        "report_id": "r2",
                        "community_id": other_id,
                        "title": "extractive other",
                        "summary": "extractive other summary",
                        "full_content": "other content",
                        "rank": 99.0,
                        "activation_quality": "clean_candidate",
                        "evidence_refs": ["evidence:2"],
                        "source_node_refs": ["n2"],
                        "source_edge_refs": ["e2"],
                        "warnings": [],
                        "graph_is_not_proof": True,
                        "support_status": "not_checked",
                        "write_permission": False,
                    },
                ],
            )
            write_jsonl(
                query_dir / "graph_context.jsonl",
                [
                    {
                        "kind": "v03_ranked_community_report",
                        "community_id": community_id,
                        "summary": "activated target",
                    }
                ],
            )
            proposal_input_id = stable_id("graph_community_report_input", community_id)
            external_outputs = root / "outputs.jsonl"
            write_jsonl(
                external_outputs,
                [
                    {
                        "proposal_input_id": proposal_input_id,
                        "model_id": "replay-model",
                        "model_output": {
                            "output_kind": "community_report_candidate",
                            "title": "assisted target",
                            "summary": "assisted target summary",
                            "findings": [
                                {
                                    "summary": "finding",
                                    "explanation": "uses valid evidence",
                                    "evidence_refs": ["evidence:1"],
                                }
                            ],
                            "retrieval_guidance": ["use this for target queries"],
                            "query_expansion_terms": ["target"],
                            "warnings": ["graph_is_not_proof"],
                        },
                    }
                ],
            )

            manifest = run_assist(
                profile_dir=profile_dir,
                output_dir=root / "assists",
                from_query_run=[query_dir],
                provider="external_jsonl",
                external_model_outputs_path=external_outputs,
            )

            assists = [
                json.loads(line)
                for line in (root / "assists" / "graph_community_report_assists.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(manifest["counts"]["selected_community_count"], 1)
            self.assertEqual(manifest["selection"]["selected_ids"], [community_id])
            self.assertEqual(len(assists), 1)
            self.assertEqual(assists[0]["title"], "assisted target")
            self.assertEqual(assists[0]["community_id"], community_id)
            self.assertTrue(assists[0]["graph_is_not_proof"])
            self.assertFalse(assists[0]["write_permission"])


if __name__ == "__main__":
    unittest.main()
