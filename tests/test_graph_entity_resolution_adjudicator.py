import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_candidate_consolidator import consolidate_graph_candidates
from tools.graph.graph_construction_packet_builder import read_jsonl
from tools.graph.graph_entity_resolution_adjudicator import run_entity_resolution_adjudicator


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def base_entity(candidate_id: str, packet_id: str, local_id: str, label: str, **overrides: object) -> dict:
    row = {
        "schema_version": "graph_v03.entity_candidate.v0.1",
        "candidate_id": candidate_id,
        "candidate_kind": "graph_entity_candidate",
        "candidate_text": label,
        "local_entity_id": local_id,
        "source_packet_id": packet_id,
        "source_node_hint": label,
        "entity_type_hint": "person",
        "evidence_refs": [f"evidence:{packet_id}"],
        "raw_backpointer_refs": [f"raw:{packet_id}"],
        "source_refs": ["source:test"],
        "source_text_quote": label,
        "source_text_excerpt": f"{label} appears in the packet.",
        "graph_is_not_proof": True,
        "write_permission": False,
        "warnings": [],
    }
    row.update(overrides)
    return row


class GraphEntityResolutionAdjudicatorTests(unittest.TestCase):
    def test_mock_adjudicator_outputs_candidate_only_merge_recommendations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            extraction_dir = workspace / "graph_v03_extraction_provider_80"
            consolidation_dir = workspace / "graph_v03_consolidation"
            output_dir = workspace / "graph_v03_entity_resolution_adjudication"
            write_jsonl(
                extraction_dir / "graph_entity_candidates.jsonl",
                [
                    base_entity(
                        "entity:1",
                        "packet:1",
                        "e1",
                        "Dr Marjoribanks",
                        evidence_refs=["evidence:shared"],
                        raw_backpointer_refs=["raw:shared"],
                    ),
                    base_entity(
                        "entity:2",
                        "packet:1",
                        "e2",
                        "Doctor Marjoribanks",
                        evidence_refs=["evidence:shared"],
                        raw_backpointer_refs=["raw:shared"],
                    ),
                ],
            )
            write_jsonl(extraction_dir / "graph_relation_candidates.jsonl", [])
            write_jsonl(extraction_dir / "graph_claim_candidates.jsonl", [])
            write_jsonl(extraction_dir / "graph_merge_candidates.jsonl", [])
            (extraction_dir / "graph_extraction_manifest.json").write_text('{"schema_version":"test"}\n', encoding="utf-8")
            consolidate_graph_candidates(workspace, extraction_dir=extraction_dir, output_dir=consolidation_dir)

            manifest = run_entity_resolution_adjudicator(
                workspace,
                consolidation_dir=consolidation_dir,
                output_dir=output_dir,
                provider="mock",
            )
            rows = read_jsonl(output_dir / "graph_entity_merge_adjudications.jsonl")
            failures = read_jsonl(output_dir / "graph_entity_merge_adjudication_failures.jsonl")

            self.assertEqual(manifest["counts"]["adjudication_count"], 1)
            self.assertEqual(failures, [])
            self.assertEqual(rows[0]["decision"], "merge")
            self.assertFalse(rows[0]["merge_applied"])
            self.assertTrue(rows[0]["graph_is_not_proof"])
            self.assertFalse(rows[0]["write_permission"])
            self.assertEqual(manifest["policies"]["llm_controls_merge_recommendation_only"], True)
            self.assertEqual(manifest["policies"]["merge_applied"], False)


if __name__ == "__main__":
    unittest.main()
