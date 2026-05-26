import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import build_graph_construction_packets, read_jsonl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class GraphConstructionPacketBuilderTests(unittest.TestCase):
    def test_builds_packets_from_project_upstream_assets_without_adapter(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_packet_workspace"
            (workspace / "evidence").mkdir(parents=True)
            (workspace / "portrait").mkdir()
            (workspace / "memory").mkdir()
            (workspace / "proposals" / "prebuild_s2_test").mkdir(parents=True)
            (workspace / "graph").mkdir()
            (workspace / "manifest.yaml").write_text(
                "workspace_id: graph_packet_workspace\nmodeled_user_id: Mira Chen\n",
                encoding="utf-8",
            )
            write_jsonl(
                workspace / "evidence" / "evidence.jsonl",
                [
                    {
                        "evidence_ref": "evidence:test:001",
                        "source_id": "source:test",
                        "step1_evidence_ref": "s1:evidence:001",
                        "step1_source_ref": "s1:source:001",
                        "participant": "Mira Chen",
                        "target_participant": "Mira Chen",
                        "subject_role": "target",
                        "timestamp": "2026-05-23T00:00:00Z",
                        "text": "Mira uses a graph notebook to track project decisions.",
                        "privacy_class": "public_dataset",
                        "subject_contamination_risk": "none",
                    }
                ],
            )
            write_jsonl(
                workspace / "portrait" / "normalized_candidates.jsonl",
                [
                    {
                        "candidate_id": "npc-mira-001",
                        "user_id": "Mira Chen",
                        "candidate_text": "Mira uses a graph notebook for project decisions.",
                        "candidate_type": "project_context",
                        "source_refs": ["source:test"],
                        "evidence_refs": ["evidence:test:001"],
                        "backpointer_refs": ["s1:evidence:001"],
                        "input_layer": "s2_proposal_outcome",
                        "input_refs": ["s2p:mira-001"],
                        "proposal_origin": {"proposal_id": "s2p:mira-001"},
                        "confidence": "high",
                        "inference_level": "explicit",
                        "privacy_class": "public_dataset",
                    }
                ],
            )
            write_jsonl(
                workspace / "portrait" / "review_decisions.jsonl",
                [
                    {
                        "decision_id": "review-mira-001",
                        "candidate_id": "npc-mira-001",
                        "review_status": "accepted_for_experiment",
                        "review_action": "accept_for_experiment",
                    }
                ],
            )
            write_jsonl(
                workspace / "portrait" / "reviewed_units.jsonl",
                [
                    {
                        "unit_id": "unit-mira-001",
                        "user_id": "Mira Chen",
                        "type": "project_context",
                        "memory_class": "semantic",
                        "content": "Mira uses a graph notebook for project decisions.",
                        "scope": "project",
                        "source_refs": ["source:test"],
                        "evidence_refs": ["evidence:test:001"],
                        "backpointer_refs": ["s1:evidence:001"],
                        "evidence_summary": "Mira uses a graph notebook to track project decisions.",
                        "confidence": "high",
                        "inference_level": "explicit",
                        "status": "active",
                        "privacy_class": "public_dataset",
                        "review_metadata": {
                            "review_status": "accepted_for_experiment",
                            "review_action": "accept_for_experiment",
                        },
                        "proposal_origin": {"proposal_id": "s2p:mira-001"},
                        "step1_origin": {
                            "input_layer": "s2_proposal_outcome",
                            "input_refs": ["s2p:mira-001"],
                        },
                    }
                ],
            )
            write_jsonl(
                workspace / "memory" / "preprocessing_decisions.jsonl",
                [
                    {
                        "decision_id": "prep-mira-001",
                        "input_layer": "evidence_item",
                        "evidence_ref": "evidence:test:001",
                        "route": "graph_candidate_needed",
                        "warnings": [],
                    }
                ],
            )
            write_jsonl(
                workspace / "proposals" / "prebuild_s2_test" / "proposal_outcomes.ai.jsonl",
                [
                    {
                        "proposal_id": "s2p:mira-001",
                        "output_kind": "portrait_fact_candidate",
                        "candidate_type": "project_context",
                        "fact_candidate_text": "Mira uses a graph notebook for project decisions.",
                        "source_text": "Mira uses a graph notebook to track project decisions.",
                        "source_perspective": "Mira Chen",
                        "attribution_status": "strict",
                        "source_refs": ["source:test"],
                        "evidence_refs": ["evidence:test:001"],
                        "raw_backpointer_refs": ["s1:evidence:001"],
                        "proposal_confidence": "high",
                        "inference_level": "explicit",
                        "write_permission": False,
                    }
                ],
            )
            write_jsonl(workspace / "graph" / "nodes.jsonl", [{"node_id": "existing-node"}])
            write_jsonl(workspace / "graph" / "edges.jsonl", [{"edge_id": "existing-edge"}])

            graph_nodes_before = (workspace / "graph" / "nodes.jsonl").read_text(encoding="utf-8")
            manifest = build_graph_construction_packets(workspace)
            packets = read_jsonl(Path(manifest["outputs"]["graph_construction_packets"]))
            graph_text_units = read_jsonl(Path(manifest["outputs"]["graph_text_units"]))

            self.assertEqual(manifest["policies"]["adapter_layer_used"], False)
            self.assertEqual(manifest["policies"]["graphrag_usage"], "stage_design_reference")
            self.assertEqual(manifest["counts"]["packet_count"], 4)
            self.assertEqual(manifest["counts"]["graph_text_unit_count"], 1)
            self.assertEqual(manifest["counts"]["missing_evidence_packets"], 0)
            self.assertEqual(
                sorted(manifest["counts"]["input_kind_counts"]),
                ["evidence_item", "normalized_candidate", "proposal_outcome", "reviewed_portrait_unit"],
            )
            self.assertTrue(all(packet["graph_is_not_proof"] is True for packet in packets))
            self.assertTrue(all(packet["evidence_refs"] == ["evidence:test:001"] for packet in packets))
            self.assertTrue(any(packet["route_refs"] == ["prep-mira-001"] for packet in packets))
            self.assertEqual(graph_text_units[0]["primary_evidence_ref"], "evidence:test:001")
            self.assertEqual(graph_text_units[0]["context_requirement_hint"], "local_turn_enough")
            self.assertEqual((workspace / "graph" / "nodes.jsonl").read_text(encoding="utf-8"), graph_nodes_before)
            self.assertTrue((workspace / "graph_v03_construction" / "graph_construction_report.md").exists())

    def test_builds_graph_text_units_with_neighbor_context_without_replacing_primary_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_context_workspace"
            (workspace / "evidence").mkdir(parents=True)
            (workspace / "portrait").mkdir()
            (workspace / "memory").mkdir()
            (workspace / "manifest.yaml").write_text(
                "workspace_id: graph_context_workspace\nmodeled_user_id: Jon\n",
                encoding="utf-8",
            )
            write_jsonl(
                workspace / "evidence" / "evidence.jsonl",
                [
                    {
                        "evidence_ref": "evidence:test:D1:1",
                        "source_id": "source:test",
                        "raw_source_id": "raw:test:dialogue",
                        "source_type": "conversation",
                        "participant": "Gina",
                        "speaker": "Gina",
                        "target_participant": "Jon",
                        "subject_role": "other_participant",
                        "locator": {"day_index": 1, "session_index": 1, "turn_index": 1},
                        "text": "What got you into this biz?",
                    },
                    {
                        "evidence_ref": "evidence:test:D1:2",
                        "source_id": "source:test",
                        "raw_source_id": "raw:test:dialogue",
                        "source_type": "conversation",
                        "participant": "Jon",
                        "speaker": "Jon",
                        "target_participant": "Jon",
                        "subject_role": "target",
                        "locator": {"day_index": 1, "session_index": 1, "turn_index": 2},
                        "text": "I want to start a dance studio.",
                    },
                    {
                        "evidence_ref": "evidence:test:D1:3",
                        "source_id": "source:test",
                        "raw_source_id": "raw:test:dialogue",
                        "source_type": "conversation",
                        "participant": "Gina",
                        "speaker": "Gina",
                        "target_participant": "Jon",
                        "subject_role": "other_participant",
                        "locator": {"day_index": 1, "session_index": 1, "turn_index": 3},
                        "text": "That sounds exciting.",
                    },
                ],
            )
            write_jsonl(workspace / "portrait" / "normalized_candidates.jsonl", [])
            write_jsonl(workspace / "portrait" / "review_decisions.jsonl", [])
            write_jsonl(workspace / "portrait" / "reviewed_units.jsonl", [])
            write_jsonl(workspace / "memory" / "preprocessing_decisions.jsonl", [])

            manifest = build_graph_construction_packets(workspace)
            packets = read_jsonl(Path(manifest["outputs"]["graph_construction_packets"]))
            graph_text_units = read_jsonl(Path(manifest["outputs"]["graph_text_units"]))
            middle_packet = next(packet for packet in packets if packet["input_ref"] == "evidence:test:D1:2")
            middle_unit = next(unit for unit in graph_text_units if unit["primary_evidence_ref"] == "evidence:test:D1:2")

            self.assertEqual(manifest["counts"]["packet_count"], 3)
            self.assertEqual(manifest["counts"]["graph_text_unit_count"], 3)
            self.assertEqual(middle_packet["evidence_refs"], ["evidence:test:D1:2"])
            self.assertEqual(middle_packet["primary_evidence_refs"], ["evidence:test:D1:2"])
            self.assertEqual(
                middle_packet["context_evidence_refs"],
                ["evidence:test:D1:1", "evidence:test:D1:3"],
            )
            self.assertIn("Gina: What got you into this biz?", middle_packet["graph_route_text"])
            self.assertIn("Jon: I want to start a dance studio.", middle_packet["graph_route_text"])
            self.assertEqual(middle_packet["context_requirement_hint"], "neighbor_turn_needed")
            self.assertIn("context_used_for_routing_not_primary_evidence", middle_unit["warnings"])


if __name__ == "__main__":
    unittest.main()
