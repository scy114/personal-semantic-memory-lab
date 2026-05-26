import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_jsonl
from tools.graph.graph_package_router import build_graph_package_routes


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def packet(packet_id: str, text: str, **overrides: object) -> dict:
    row = {
        "schema_version": "graph_v03.construction_packet.v0.1",
        "packet_id": packet_id,
        "workspace_id": "graph_route_workspace",
        "modeled_user_id": "Mira Chen",
        "input_kind": "evidence_item",
        "input_ref": f"evidence:{packet_id}",
        "original_text": text,
        "processed_text": "",
        "evidence_refs": [f"evidence:{packet_id}"],
        "source_refs": ["source:test"],
        "raw_backpointer_refs": [f"raw:{packet_id}"],
        "source_perspective": "Mira Chen",
        "subject_role": "target",
        "attribution_status": "source_text_only",
        "temporal_scope": {},
        "confidence": "source",
        "inference_level": "source_text",
        "privacy_class": "public_dataset",
        "route_refs": [],
        "proposal_refs": [],
        "review_refs": [],
        "warnings": [],
        "graph_is_not_proof": True,
    }
    row.update(overrides)
    return row


class GraphPackageRouterTests(unittest.TestCase):
    def test_routes_graph_packets_by_graph_specific_matrix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_route_workspace"
            output_dir = workspace / "graph_v03_construction"
            packets = [
                packet("explicit", "Mira uses a graph notebook to track project decisions."),
                packet("entity", "Mira Chen."),
                packet("low", "Thanks."),
                packet(
                    "attributed",
                    "Mira said the graph notebook replaced the old tracker because project decisions changed.",
                    attribution_status="reported_speech",
                ),
                packet("repair", "Mira depends on the routing report.", evidence_refs=[], raw_backpointer_refs=[]),
            ]
            write_jsonl(output_dir / "graph_construction_packets.jsonl", packets)

            manifest = build_graph_package_routes(workspace, output_dir=output_dir, route_run_id="test-graph-route")
            features = read_jsonl(output_dir / "graph_route_feature_matrix.jsonl")
            decisions = read_jsonl(output_dir / "graph_route_decisions.jsonl")
            routes = {row["source_packet_id"]: row["recommended_route"] for row in decisions}

            self.assertEqual(manifest["target_task"], "graph_relation_candidate")
            self.assertTrue((output_dir / "graph_route_feature_matrix.csv").exists())
            self.assertTrue(all(row["graph_is_not_proof"] is True for row in features + decisions))
            for key in [
                "relation_surface_clarity",
                "relation_likelihood",
                "endpoint_quality",
                "directionality_certainty",
                "context_requirement",
                "attribution_risk",
                "evidence_quality",
                "graph_utility_hint",
                "merge_ambiguity",
                "low_graph_value",
            ]:
                self.assertIn(key, features[0])
            self.assertEqual(routes["explicit"], "nlp_openie_candidate")
            self.assertEqual(routes["entity"], "entity_candidate_only")
            self.assertEqual(routes["low"], "skip_or_background_only")
            self.assertEqual(routes["attributed"], "strong_llm_graph_extraction")
            self.assertEqual(routes["repair"], "repair_or_review")


if __name__ == "__main__":
    unittest.main()
