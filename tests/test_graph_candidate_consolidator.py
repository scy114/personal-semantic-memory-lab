import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_candidate_consolidator import consolidate_graph_candidates
from tools.graph.graph_construction_packet_builder import read_jsonl


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
        "entity_type_hint": "project",
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


def base_relation(candidate_id: str, packet_id: str, source_id: str, target_id: str, relation_type: str, **overrides: object) -> dict:
    row = {
        "schema_version": "graph_v03.relation_candidate.v0.1",
        "candidate_id": candidate_id,
        "candidate_kind": "graph_relation_candidate",
        "candidate_text": "Mira works on routing calibration.",
        "relation_description": "Mira works on routing calibration.",
        "relation_type_hint": relation_type,
        "source_packet_id": packet_id,
        "source_local_entity_id": source_id,
        "target_local_entity_id": target_id,
        "source_node_hint": "Mira",
        "target_node_hint": "routing calibration",
        "source_perspective": "Mira",
        "attribution_status": "source_text_only",
        "temporal_scope": {},
        "evidence_refs": [f"evidence:{packet_id}"],
        "raw_backpointer_refs": [f"raw:{packet_id}"],
        "source_refs": ["source:test"],
        "source_text_quote": "Mira works on routing calibration.",
        "source_text_excerpt": "Mira works on routing calibration.",
        "graph_is_not_proof": True,
        "write_permission": False,
        "warnings": [],
    }
    row.update(overrides)
    return row


def base_claim(candidate_id: str, packet_id: str, subject_id: str, text: str, **overrides: object) -> dict:
    row = {
        "schema_version": "graph_v03.claim_candidate.v0.1",
        "candidate_id": candidate_id,
        "candidate_kind": "graph_claim_candidate",
        "candidate_text": text,
        "claim_type_hint": "state",
        "claim_status": "asserted",
        "subject_local_entity_id": subject_id,
        "source_node_hint": "Mira",
        "entity_type_hint": "person",
        "source_packet_id": packet_id,
        "source_perspective": "Mira",
        "attribution_status": "source_text_only",
        "temporal_scope": {},
        "evidence_refs": [f"evidence:{packet_id}"],
        "raw_backpointer_refs": [f"raw:{packet_id}"],
        "source_refs": ["source:test"],
        "source_text_quote": text,
        "source_text_excerpt": text,
        "confidence_hint": "high",
        "graph_is_not_proof": True,
        "write_permission": False,
        "warnings": [],
    }
    row.update(overrides)
    return row


class GraphCandidateConsolidatorTests(unittest.TestCase):
    def test_consolidates_exact_entity_groups_and_relation_edges(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            extraction_dir = workspace / "graph_v03_extraction_provider_80"
            output_dir = workspace / "graph_v03_consolidation"
            write_jsonl(
                extraction_dir / "graph_entity_candidates.jsonl",
                [
                    base_entity("entity:1", "packet:1", "e1", "Mira", entity_type_hint="person"),
                    base_entity("entity:4", "packet:1", "", "Mira", entity_type_hint="person"),
                    base_entity("entity:2", "packet:1", "e2", "routing calibration", entity_type_hint="project"),
                    base_entity("entity:3", "packet:2", "e1", "Routing Calibration", entity_type_hint="project"),
                ],
            )
            write_jsonl(
                extraction_dir / "graph_relation_candidates.jsonl",
                [
                    base_relation("relation:1", "packet:1", "e1", "e2", "participates-in"),
                ],
            )
            write_jsonl(
                extraction_dir / "graph_claim_candidates.jsonl",
                [
                    base_claim("claim:1", "packet:1", "e1", "Mira cares about routing calibration."),
                    base_claim(
                        "claim:2",
                        "packet:1",
                        "",
                        "Mira also tracks duplicate hint resolution.",
                        subject_local_entity_id=None,
                    ),
                ],
            )
            write_jsonl(extraction_dir / "graph_merge_candidates.jsonl", [])
            (extraction_dir / "graph_extraction_manifest.json").write_text('{"schema_version":"test"}\n', encoding="utf-8")

            manifest = consolidate_graph_candidates(workspace, extraction_dir=extraction_dir, output_dir=output_dir)

            nodes = read_jsonl(output_dir / "graph_nodes_table.jsonl")
            edges = read_jsonl(output_dir / "graph_edges_table.jsonl")
            claims = read_jsonl(output_dir / "graph_claims_table.jsonl")
            relation_norm = read_jsonl(output_dir / "graph_relation_type_normalization.jsonl")
            decisions = read_jsonl(output_dir / "graph_merge_decisions.jsonl")
            evidence_links = read_jsonl(output_dir / "evidence_links.jsonl")

            self.assertEqual(manifest["counts"]["node_count"], 2)
            self.assertEqual(manifest["counts"]["edge_count"], 1)
            self.assertEqual(manifest["counts"]["claim_table_count"], 2)
            self.assertEqual(manifest["counts"]["entity_quality_counts"]["stable"], 2)
            self.assertTrue(all(row["graph_is_not_proof"] is True for row in nodes + edges + claims + evidence_links))
            self.assertTrue(all(row["write_permission"] is False for row in nodes + edges + claims + evidence_links))
            self.assertTrue(all(row["entity_quality_hint"] == "stable" for row in nodes))
            self.assertEqual(edges[0]["source_label"], "Mira")
            self.assertEqual(edges[0]["target_label"], "routing calibration")
            self.assertEqual(edges[0]["relation_type"], "participates_in")
            self.assertEqual(edges[0]["generic_relation_review_hint"], "not_generic")
            self.assertEqual(claims[0]["subject_label"], "Mira")
            self.assertEqual(claims[0]["subject_endpoint_status"], "local_entity_id")
            self.assertEqual(claims[1]["subject_label"], "Mira")
            self.assertEqual(claims[1]["subject_endpoint_status"], "unique_hint_in_packet")
            self.assertEqual(relation_norm[0]["normalized_relation_type"], "participates_in")
            self.assertEqual(relation_norm[0]["relation_category"], "external_arf_fiction")
            self.assertIn("arf_fiction_relation_ontology", relation_norm[0]["relation_schema_sources"])
            self.assertEqual(edges[0]["relation_category"], "external_arf_fiction")
            self.assertIn("arf_fiction_relation_ontology", edges[0]["relation_schema_sources"])
            self.assertTrue(any(row["decision_kind"] == "entity_exact_normalized_label" and row["decision"] == "merge" for row in decisions))
            self.assertTrue((output_dir / "graph_consolidation_manifest.json").exists())
            self.assertTrue((output_dir / "graph_consolidation_report.md").exists())

    def test_external_narrative_relation_schema_maps_literary_family_edges(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            extraction_dir = workspace / "graph_v03_extraction_provider_80"
            output_dir = workspace / "graph_v03_consolidation"
            write_jsonl(
                extraction_dir / "graph_entity_candidates.jsonl",
                [
                    base_entity("entity:1", "packet:1", "e1", "Lucilla", entity_type_hint="person"),
                    base_entity("entity:2", "packet:1", "e2", "Mrs Marjoribanks", entity_type_hint="person"),
                ],
            )
            write_jsonl(
                extraction_dir / "graph_relation_candidates.jsonl",
                [base_relation("relation:1", "packet:1", "e1", "e2", "child-of")],
            )
            write_jsonl(extraction_dir / "graph_claim_candidates.jsonl", [])
            write_jsonl(extraction_dir / "graph_merge_candidates.jsonl", [])
            (extraction_dir / "graph_extraction_manifest.json").write_text('{"schema_version":"test"}\n', encoding="utf-8")

            consolidate_graph_candidates(workspace, extraction_dir=extraction_dir, output_dir=output_dir)
            edges = read_jsonl(output_dir / "graph_edges_table.jsonl")
            relation_norm = read_jsonl(output_dir / "graph_relation_type_normalization.jsonl")

            self.assertEqual(edges[0]["relation_type"], "child_of")
            self.assertEqual(edges[0]["relation_category"], "external_arf_fiction")
            self.assertEqual(edges[0]["generic_relation_review_hint"], "not_generic")
            self.assertEqual(relation_norm[0]["normalization_method"], "alias_map")
            self.assertEqual(relation_norm[0]["normalized_relation_type"], "child_of")
            self.assertEqual(relation_norm[0]["relation_category"], "external_arf_fiction")
            self.assertIn("arf_fiction_relation_ontology", relation_norm[0]["relation_schema_sources"])

    def test_unresolved_endpoint_goes_to_repair_review_without_edge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            extraction_dir = workspace / "graph_v03_extraction_provider_80"
            output_dir = workspace / "graph_v03_consolidation"
            write_jsonl(
                extraction_dir / "graph_entity_candidates.jsonl",
                [base_entity("entity:1", "packet:1", "e1", "Mira", entity_type_hint="person")],
            )
            write_jsonl(
                extraction_dir / "graph_relation_candidates.jsonl",
                [
                    base_relation(
                        "relation:1",
                        "packet:1",
                        "e1",
                        "missing",
                        "mystery-link",
                        evidence_refs=[],
                        raw_backpointer_refs=[],
                    ),
                ],
            )
            write_jsonl(extraction_dir / "graph_claim_candidates.jsonl", [])
            write_jsonl(extraction_dir / "graph_merge_candidates.jsonl", [])
            (extraction_dir / "graph_extraction_manifest.json").write_text('{"schema_version":"test"}\n', encoding="utf-8")

            manifest = consolidate_graph_candidates(workspace, extraction_dir=extraction_dir, output_dir=output_dir)
            edges = read_jsonl(output_dir / "graph_edges_table.jsonl")
            relation_norm = read_jsonl(output_dir / "graph_relation_type_normalization.jsonl")
            decisions = read_jsonl(output_dir / "graph_merge_decisions.jsonl")

            self.assertEqual(edges, [])
            self.assertEqual(manifest["counts"]["unresolved_endpoint_relation_count"], 1)
            self.assertEqual(relation_norm[0]["normalized_relation_type"], "related_to_generic")
            self.assertIn("unmapped_relation_type", relation_norm[0]["warnings"])
            self.assertTrue(any(row["decision"] == "repair_or_review" for row in decisions))

    def test_review_entity_groups_are_not_materialized_as_one_node(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            extraction_dir = workspace / "graph_v03_extraction_provider_80"
            output_dir = workspace / "graph_v03_consolidation"
            write_jsonl(
                extraction_dir / "graph_entity_candidates.jsonl",
                [
                    base_entity("entity:1", "packet:1", "e1", "dancing", entity_type_hint="preference"),
                    base_entity("entity:2", "packet:2", "e1", "dancing", entity_type_hint="event"),
                ],
            )
            write_jsonl(extraction_dir / "graph_relation_candidates.jsonl", [])
            write_jsonl(extraction_dir / "graph_claim_candidates.jsonl", [])
            write_jsonl(extraction_dir / "graph_merge_candidates.jsonl", [])
            (extraction_dir / "graph_extraction_manifest.json").write_text('{"schema_version":"test"}\n', encoding="utf-8")

            manifest = consolidate_graph_candidates(workspace, extraction_dir=extraction_dir, output_dir=output_dir)
            nodes = read_jsonl(output_dir / "graph_nodes_table.jsonl")
            decisions = read_jsonl(output_dir / "graph_merge_decisions.jsonl")

            self.assertEqual(manifest["counts"]["node_count"], 2)
            self.assertEqual(len({row["node_id"] for row in nodes}), 2)
            self.assertTrue(all(row["entity_quality_hint"] == "review_required" for row in nodes))
            self.assertTrue(all("entity_merge_requires_review" in row["warnings"] for row in nodes))
            self.assertTrue(any(row["decision"] == "review" and row["output_node_id"] == "" for row in decisions))

    def test_alias_like_entities_emit_resolution_candidate_without_silent_merge(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            extraction_dir = workspace / "graph_v03_extraction_provider_80"
            output_dir = workspace / "graph_v03_consolidation"
            write_jsonl(
                extraction_dir / "graph_entity_candidates.jsonl",
                [
                    base_entity(
                        "entity:1",
                        "packet:1",
                        "e1",
                        "Dr Marjoribanks",
                        entity_type_hint="person",
                        evidence_refs=["evidence:shared"],
                        raw_backpointer_refs=["raw:shared"],
                    ),
                    base_entity(
                        "entity:2",
                        "packet:1",
                        "e2",
                        "Doctor Marjoribanks",
                        entity_type_hint="person",
                        evidence_refs=["evidence:shared"],
                        raw_backpointer_refs=["raw:shared"],
                    ),
                ],
            )
            write_jsonl(extraction_dir / "graph_relation_candidates.jsonl", [])
            write_jsonl(extraction_dir / "graph_claim_candidates.jsonl", [])
            write_jsonl(extraction_dir / "graph_merge_candidates.jsonl", [])
            (extraction_dir / "graph_extraction_manifest.json").write_text('{"schema_version":"test"}\n', encoding="utf-8")

            manifest = consolidate_graph_candidates(workspace, extraction_dir=extraction_dir, output_dir=output_dir)
            nodes = read_jsonl(output_dir / "graph_nodes_table.jsonl")
            resolution_candidates = read_jsonl(output_dir / "graph_entity_resolution_candidates.jsonl")

            self.assertEqual(manifest["counts"]["node_count"], 2)
            self.assertEqual(manifest["counts"]["entity_resolution_candidate_count"], 1)
            self.assertEqual(len(nodes), 2)
            self.assertEqual(resolution_candidates[0]["candidate_source"], "derived_node_pair")
            self.assertEqual(resolution_candidates[0]["resolution_status"], "candidate_only")
            self.assertEqual(resolution_candidates[0]["recommended_action"], "review_before_merge")
            self.assertIn("shared_evidence_refs", resolution_candidates[0]["signals"])
            self.assertTrue(resolution_candidates[0]["graph_is_not_proof"])
            self.assertFalse(resolution_candidates[0]["write_permission"])

    def test_generic_relation_review_hint_marks_out_of_schema_cases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            extraction_dir = workspace / "graph_v03_extraction_provider_80"
            output_dir = workspace / "graph_v03_consolidation"
            write_jsonl(
                extraction_dir / "graph_entity_candidates.jsonl",
                [
                    base_entity("entity:1", "packet:1", "e1", "Mira", entity_type_hint="person"),
                    base_entity("entity:2", "packet:1", "e2", "store", entity_type_hint="organization"),
                ],
            )
            write_jsonl(
                extraction_dir / "graph_relation_candidates.jsonl",
                [base_relation("relation:1", "packet:1", "e1", "e2", "entity-resolution-needed-link")],
            )
            write_jsonl(extraction_dir / "graph_claim_candidates.jsonl", [])
            write_jsonl(extraction_dir / "graph_merge_candidates.jsonl", [])
            (extraction_dir / "graph_extraction_manifest.json").write_text('{"schema_version":"test"}\n', encoding="utf-8")

            manifest = consolidate_graph_candidates(workspace, extraction_dir=extraction_dir, output_dir=output_dir)
            edges = read_jsonl(output_dir / "graph_edges_table.jsonl")
            relation_norm = read_jsonl(output_dir / "graph_relation_type_normalization.jsonl")

            self.assertEqual(edges[0]["relation_type"], "related_to_generic")
            self.assertEqual(edges[0]["generic_relation_review_hint"], "needs_schema_extension")
            self.assertEqual(relation_norm[0]["generic_relation_review_hint"], "needs_schema_extension")
            self.assertEqual(manifest["counts"]["generic_relation_review_counts"]["needs_schema_extension"], 1)


if __name__ == "__main__":
    unittest.main()
