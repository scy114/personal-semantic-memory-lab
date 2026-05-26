import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.proposals.proposal_runner import ProviderResult
from tools.graph.graph_relation_candidate_extractor import build_graph_relation_candidates, read_jsonl


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class GraphRelationCandidateExtractorTests(unittest.TestCase):
    def test_mock_extraction_preserves_evidence_and_writes_candidate_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_extraction_workspace"
            output_dir = workspace / "graph_v03_construction"
            workspace.mkdir(parents=True)
            (workspace / "graph").mkdir()
            write_jsonl(workspace / "graph" / "nodes.jsonl", [{"node_id": "existing-node"}])
            graph_nodes_before = (workspace / "graph" / "nodes.jsonl").read_text(encoding="utf-8")

            packets = [
                {
                    "schema_version": "graph_v03.construction_packet.v0.1",
                    "packet_id": "graph_packet:001",
                    "workspace_id": workspace.name,
                    "modeled_user_id": "Mira Chen",
                    "input_kind": "evidence_item",
                    "input_ref": "evidence:test:001",
                    "original_text": "Mira uses a graph notebook to track project decisions.",
                    "processed_text": "",
                    "evidence_refs": ["evidence:test:001"],
                    "source_refs": ["source:test"],
                    "raw_backpointer_refs": ["raw:test:001"],
                    "source_perspective": "Mira Chen",
                    "subject_role": "target",
                    "attribution_status": "source_text_only",
                    "temporal_scope": {},
                    "confidence": "source",
                    "inference_level": "source_text",
                    "privacy_class": "public_dataset",
                    "route_refs": ["route:test:001"],
                    "proposal_refs": [],
                    "review_refs": [],
                    "warnings": [],
                    "graph_is_not_proof": True,
                },
                {
                    "schema_version": "graph_v03.construction_packet.v0.1",
                    "packet_id": "graph_packet:002",
                    "workspace_id": workspace.name,
                    "modeled_user_id": "Mira Chen",
                    "input_kind": "proposal_outcome",
                    "input_ref": "s2p:test:002",
                    "original_text": "Mira uses a graph notebook to track project decisions.",
                    "processed_text": "Mira uses a graph notebook for project decisions.",
                    "evidence_refs": ["evidence:test:002"],
                    "source_refs": ["source:test"],
                    "raw_backpointer_refs": ["raw:test:002"],
                    "source_perspective": "Mira Chen",
                    "subject_role": "target",
                    "attribution_status": "strict",
                    "temporal_scope": {},
                    "confidence": "high",
                    "inference_level": "explicit",
                    "privacy_class": "public_dataset",
                    "route_refs": ["route:test:002"],
                    "proposal_refs": ["s2p:test:002"],
                    "review_refs": [],
                    "warnings": [],
                    "graph_is_not_proof": True,
                },
                {
                    "schema_version": "graph_v03.construction_packet.v0.1",
                    "packet_id": "graph_packet:003",
                    "workspace_id": workspace.name,
                    "modeled_user_id": "Mira Chen",
                    "input_kind": "evidence_item",
                    "input_ref": "evidence:test:003",
                    "original_text": "Thanks.",
                    "processed_text": "",
                    "evidence_refs": ["evidence:test:003"],
                    "source_refs": ["source:test"],
                    "raw_backpointer_refs": ["raw:test:003"],
                    "source_perspective": "Mira Chen",
                    "subject_role": "target",
                    "attribution_status": "source_text_only",
                    "temporal_scope": {},
                    "confidence": "source",
                    "inference_level": "source_text",
                    "privacy_class": "public_dataset",
                    "route_refs": ["route:test:003"],
                    "proposal_refs": [],
                    "review_refs": [],
                    "warnings": [],
                    "graph_is_not_proof": True,
                },
            ]
            write_jsonl(output_dir / "graph_construction_packets.jsonl", packets)

            manifest = build_graph_relation_candidates(
                workspace,
                output_dir=output_dir,
                provider="mock",
                max_gleanings=2,
                duplicate_policy="overwrite_generated",
            )

            relations = read_jsonl(output_dir / "graph_relation_candidates.jsonl")
            entities = read_jsonl(output_dir / "graph_entity_candidates.jsonl")
            failures = read_jsonl(output_dir / "graph_extraction_failures.jsonl")
            merge_candidates = read_jsonl(output_dir / "graph_merge_candidates.jsonl")

            self.assertEqual(manifest["schema_version"], "graph_v03.relation_extraction.v0.1")
            self.assertEqual(manifest["policies"]["merge_performed"], False)
            self.assertTrue(all(row["graph_is_not_proof"] is True for row in relations + entities + merge_candidates))
            self.assertTrue(any(row["relation_type_hint"] == "uses" for row in relations))
            self.assertTrue(any(row["target_node_hint"] == "graph notebook" for row in relations))
            self.assertTrue(all(row["evidence_refs"] for row in relations))
            self.assertTrue(any(row["failure_kind"] == "no_useful_modeling_value" for row in failures))
            self.assertTrue(any(row["merge_kind"] == "relation_exact_match" for row in merge_candidates))
            self.assertEqual((workspace / "graph" / "nodes.jsonl").read_text(encoding="utf-8"), graph_nodes_before)
            self.assertTrue((output_dir / "graph_extraction_manifest.json").exists())
            self.assertTrue((output_dir / "graph_extraction_report.md").exists())

    def test_external_jsonl_replay_keeps_packet_provenance_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_external_workspace"
            output_dir = workspace / "graph_v03_construction"
            workspace.mkdir(parents=True)
            packet = {
                "schema_version": "graph_v03.construction_packet.v0.1",
                "packet_id": "graph_packet:external",
                "workspace_id": workspace.name,
                "modeled_user_id": "Mira Chen",
                "input_kind": "evidence_item",
                "input_ref": "evidence:test:external",
                "original_text": "Mira works on routing calibration.",
                "processed_text": "",
                "evidence_refs": ["evidence:test:external"],
                "source_refs": ["source:test"],
                "raw_backpointer_refs": ["raw:test:external"],
                "source_perspective": "Mira Chen",
                "subject_role": "target",
                "attribution_status": "source_text_only",
                "temporal_scope": {},
                "confidence": "source",
                "inference_level": "source_text",
                "privacy_class": "public_dataset",
                "route_refs": ["route:test:external"],
                "proposal_refs": [],
                "review_refs": [],
                "warnings": [],
                "graph_is_not_proof": True,
            }
            write_jsonl(output_dir / "graph_construction_packets.jsonl", [packet])
            external_outputs = workspace / "external_graph_outputs.jsonl"
            write_jsonl(
                external_outputs,
                [
                    {
                        "source_packet_id": "graph_packet:external",
                        "relation_candidates": [
                            {
                                "candidate_id": "external-rel-001",
                                "source_node_hint": "Mira Chen",
                                "target_node_hint": "routing calibration",
                                "relation_type_hint": "works_on",
                                "directionality": "forward",
                                "confidence_hint": "high",
                                "inference_level_hint": "explicit",
                            }
                        ],
                    }
                ],
            )

            build_graph_relation_candidates(
                workspace,
                output_dir=output_dir,
                provider="external_jsonl",
                external_model_outputs_path=external_outputs,
                duplicate_policy="overwrite_generated",
            )

            relations = read_jsonl(output_dir / "graph_relation_candidates.jsonl")
            self.assertEqual(len(relations), 1)
            self.assertEqual(relations[0]["candidate_id"], "external-rel-001")
            self.assertEqual(relations[0]["source_packet_id"], "graph_packet:external")
            self.assertEqual(relations[0]["evidence_refs"], ["evidence:test:external"])
            self.assertTrue(relations[0]["graph_is_not_proof"])

    def test_extractor_consumes_graph_route_decisions_and_keeps_regex_as_baseline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_routed_workspace"
            output_dir = workspace / "graph_v03_construction"
            workspace.mkdir(parents=True)
            packets = [
                {
                    "schema_version": "graph_v03.construction_packet.v0.1",
                    "packet_id": "graph_packet:nlp",
                    "workspace_id": workspace.name,
                    "modeled_user_id": "Mira Chen",
                    "input_kind": "evidence_item",
                    "input_ref": "evidence:test:nlp",
                    "original_text": "Mira uses a graph notebook to track project decisions.",
                    "processed_text": "",
                    "evidence_refs": ["evidence:test:nlp"],
                    "source_refs": ["source:test"],
                    "raw_backpointer_refs": ["raw:test:nlp"],
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
                },
                {
                    "schema_version": "graph_v03.construction_packet.v0.1",
                    "packet_id": "graph_packet:llm",
                    "workspace_id": workspace.name,
                    "modeled_user_id": "Mira Chen",
                    "input_kind": "evidence_item",
                    "input_ref": "evidence:test:llm",
                    "original_text": "Mira said the notebook changed how project decisions are tracked.",
                    "processed_text": "",
                    "evidence_refs": ["evidence:test:llm"],
                    "source_refs": ["source:test"],
                    "raw_backpointer_refs": ["raw:test:llm"],
                    "source_perspective": "Mira Chen",
                    "subject_role": "target",
                    "attribution_status": "reported_speech",
                    "temporal_scope": {},
                    "confidence": "source",
                    "inference_level": "source_text",
                    "privacy_class": "public_dataset",
                    "route_refs": [],
                    "proposal_refs": [],
                    "review_refs": [],
                    "warnings": [],
                    "graph_is_not_proof": True,
                },
            ]
            write_jsonl(output_dir / "graph_construction_packets.jsonl", packets)
            route_decisions = [
                {
                    "route_decision_id": "graph_route:nlp",
                    "target_task": "graph_relation_candidate",
                    "source_packet_id": "graph_packet:nlp",
                    "recommended_route": "nlp_openie_candidate",
                    "routing_reasons": ["graph_route_surface_relation_clear"],
                    "warnings": [],
                    "graph_is_not_proof": True,
                },
                {
                    "route_decision_id": "graph_route:llm",
                    "target_task": "graph_relation_candidate",
                    "source_packet_id": "graph_packet:llm",
                    "recommended_route": "strong_llm_graph_extraction",
                    "routing_reasons": ["graph_route_high_utility_complex_or_attributed"],
                    "warnings": [],
                    "graph_is_not_proof": True,
                },
            ]
            route_path = output_dir / "graph_route_decisions.jsonl"
            write_jsonl(route_path, route_decisions)

            manifest = build_graph_relation_candidates(
                workspace,
                output_dir=output_dir,
                provider="mock_regex_baseline",
                route_decisions_path=route_path,
                duplicate_policy="overwrite_generated",
            )

            relations = read_jsonl(output_dir / "graph_relation_candidates.jsonl")
            failures = read_jsonl(output_dir / "graph_extraction_failures.jsonl")

            self.assertEqual(manifest["provider"], "mock_regex_baseline")
            self.assertEqual(manifest["policies"]["regex_extractor_role"], "mock_regex_baseline_only")
            self.assertTrue(manifest["policies"]["route_decisions_consumed"])
            self.assertTrue(any(row["source_packet_id"] == "graph_packet:nlp" for row in relations))
            self.assertFalse(any(row["source_packet_id"] == "graph_packet:llm" for row in relations))
            self.assertTrue(any(row["failure_kind"] == "llm_schema_extraction_required" for row in failures))

    def test_openai_provider_schema_guided_extraction_validates_endpoints_and_quotes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_openai_workspace"
            output_dir = workspace / "graph_v03_construction"
            workspace.mkdir(parents=True)
            packet = {
                "schema_version": "graph_v03.construction_packet.v0.1",
                "packet_id": "graph_packet:openai",
                "workspace_id": workspace.name,
                "modeled_user_id": "Jon",
                "input_kind": "evidence_item",
                "input_ref": "evidence:test:openai",
                "original_text": "Jon lost his banker job and is starting a dance studio.",
                "processed_text": "",
                "graph_extraction_text": "Gina: Anything new?\nJon lost his banker job and is starting a dance studio.",
                "evidence_refs": ["evidence:test:openai"],
                "primary_evidence_refs": ["evidence:test:openai"],
                "context_evidence_refs": ["evidence:test:ctx"],
                "source_refs": ["source:test"],
                "raw_backpointer_refs": ["raw:test:openai"],
                "source_perspective": "Jon",
                "subject_role": "target",
                "attribution_status": "source_text_only",
                "temporal_scope": {},
                "confidence": "source",
                "inference_level": "source_text",
                "privacy_class": "public_dataset",
                "route_refs": [],
                "proposal_refs": [],
                "review_refs": [],
                "warnings": ["context_used_for_routing_not_primary_evidence"],
                "graph_is_not_proof": True,
            }
            write_jsonl(output_dir / "graph_construction_packets.jsonl", [packet])
            route_path = output_dir / "graph_route_decisions.jsonl"
            write_jsonl(
                route_path,
                [
                    {
                        "route_decision_id": "graph_route:openai",
                        "target_task": "graph_relation_candidate",
                        "source_packet_id": "graph_packet:openai",
                        "recommended_route": "strong_llm_graph_extraction",
                        "routing_reasons": ["graph_route_high_utility_complex_or_attributed"],
                        "warnings": [],
                        "graph_is_not_proof": True,
                    }
                ],
            )
            relation_schema_path = output_dir / "relation_schema_candidates.jsonl"
            write_jsonl(
                relation_schema_path,
                [
                    {
                        "packet_id": "graph_packet:openai",
                        "relation_schema_candidates": [
                            {
                                "rank": 1,
                                "relation_type": "starts",
                                "score": 12.0,
                                "category": "external_arf_fiction",
                                "aliases": ["starts", "begins"],
                                "external_sources": ["arf_fiction_relation_ontology"],
                            }
                        ],
                    }
                ],
            )

            class GraphProvider:
                def generate(self, *, prompt, model_id, input_packet):
                    self.input_packet = input_packet
                    output = json.dumps(
                        {
                            "output_kind": "graph_bundle_candidate",
                            "entity_candidates": [
                                {
                                    "local_entity_id": "e1",
                                    "name": "Jon",
                                    "entity_type_hint": "person",
                                    "description": "speaker",
                                    "source_text_quote": "Jon",
                                    "attribution_status": "source_text_only",
                                    "confidence_hint": "high",
                                    "warnings": [],
                                },
                                {
                                    "local_entity_id": "e2",
                                    "name": "dance studio",
                                    "entity_type_hint": "project",
                                    "description": "business Jon is starting",
                                    "source_text_quote": "starting a dance studio",
                                    "attribution_status": "source_text_only",
                                    "confidence_hint": "high",
                                    "warnings": [],
                                },
                            ],
                            "relation_candidates": [
                                {
                                    "source_local_entity_id": "e1",
                                    "target_local_entity_id": "e2",
                                    "relation_type_hint": "starts",
                                    "relation_description": "Jon is starting a dance studio.",
                                    "why_related": "The source text directly states Jon is starting a dance studio.",
                                    "directionality_status": "directed",
                                    "source_text_quote": "Jon lost his banker job and is starting a dance studio.",
                                    "attribution_status": "source_text_only",
                                    "confidence_hint": "high",
                                    "warnings": [],
                                }
                            ],
                            "claim_candidates": [],
                            "warnings": [],
                            "graph_is_not_proof": True,
                        },
                        ensure_ascii=False,
                    )
                    return ProviderResult(output, model_id, "openai", 100, 80, None, 1)

            graph_provider = GraphProvider()
            with patch("tools.graph.graph_relation_candidate_extractor.resolve_live_api", return_value=(True, "test")), patch(
                "tools.graph.graph_relation_candidate_extractor.build_provider",
                return_value=graph_provider,
            ):
                manifest = build_graph_relation_candidates(
                    workspace,
                    project_root=Path(__file__).resolve().parents[1],
                    output_dir=output_dir,
                    provider="openai",
                    route_decisions_path=route_path,
                    relation_schema_candidates_path=relation_schema_path,
                    duplicate_policy="overwrite_generated",
                )

            relations = read_jsonl(output_dir / "graph_relation_candidates.jsonl")
            entities = read_jsonl(output_dir / "graph_entity_candidates.jsonl")
            failures = read_jsonl(output_dir / "graph_extraction_failures.jsonl")
            model_inputs = read_jsonl(output_dir / "graph_model_call_inputs.jsonl")

            self.assertEqual(manifest["provider"], "openai")
            self.assertEqual(manifest["policies"]["llm_primary_path"], "openai_schema_guided_extraction")
            self.assertEqual(manifest["policies"]["relation_schema_candidates_consumed"], True)
            self.assertEqual(len(model_inputs), 1)
            self.assertEqual(graph_provider.input_packet["allowed_relation_types"][0]["relation_type"], "starts")
            self.assertEqual(len(entities), 2)
            self.assertEqual(len(relations), 1)
            self.assertEqual(relations[0]["source_node_hint"], "Jon")
            self.assertEqual(relations[0]["target_node_hint"], "dance studio")
            self.assertEqual(relations[0]["relation_type_hint"], "starts")
            self.assertEqual(relations[0]["relation_schema_status"], "selected_from_retrieved_schema")
            self.assertTrue(relations[0]["graph_is_not_proof"])
            self.assertFalse(relations[0]["write_permission"])
            self.assertEqual(failures, [])

    def test_openai_provider_rejects_orphan_relation_endpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_orphan_workspace"
            output_dir = workspace / "graph_v03_construction"
            workspace.mkdir(parents=True)
            packet = {
                "schema_version": "graph_v03.construction_packet.v0.1",
                "packet_id": "graph_packet:orphan",
                "workspace_id": workspace.name,
                "modeled_user_id": "Mira",
                "input_kind": "evidence_item",
                "input_ref": "evidence:test:orphan",
                "original_text": "Mira works on routing calibration.",
                "processed_text": "",
                "graph_extraction_text": "Mira works on routing calibration.",
                "evidence_refs": ["evidence:test:orphan"],
                "source_refs": ["source:test"],
                "raw_backpointer_refs": ["raw:test:orphan"],
                "source_perspective": "Mira",
                "subject_role": "target",
                "attribution_status": "source_text_only",
                "temporal_scope": {},
                "warnings": [],
                "graph_is_not_proof": True,
            }
            write_jsonl(output_dir / "graph_construction_packets.jsonl", [packet])
            route_path = output_dir / "graph_route_decisions.jsonl"
            write_jsonl(
                route_path,
                [
                    {
                        "source_packet_id": "graph_packet:orphan",
                        "recommended_route": "weak_llm_graph_extraction",
                        "routing_reasons": [],
                        "warnings": [],
                    }
                ],
            )

            class OrphanProvider:
                def generate(self, *, prompt, model_id, input_packet):
                    output = json.dumps(
                        {
                            "output_kind": "graph_bundle_candidate",
                            "entity_candidates": [
                                {
                                    "local_entity_id": "e1",
                                    "name": "Mira",
                                    "entity_type_hint": "person",
                                    "source_text_quote": "Mira",
                                    "attribution_status": "source_text_only",
                                    "confidence_hint": "high",
                                    "warnings": [],
                                }
                            ],
                            "relation_candidates": [
                                {
                                    "source_local_entity_id": "e1",
                                    "target_local_entity_id": "missing",
                                    "relation_type_hint": "works_on",
                                    "relation_description": "bad endpoint",
                                    "why_related": "bad endpoint",
                                    "directionality_status": "directed",
                                    "source_text_quote": "Mira works on routing calibration.",
                                    "attribution_status": "source_text_only",
                                    "confidence_hint": "high",
                                    "warnings": [],
                                }
                            ],
                            "claim_candidates": [],
                            "warnings": [],
                            "graph_is_not_proof": True,
                        },
                        ensure_ascii=False,
                    )
                    return ProviderResult(output, model_id, "openai", 100, 80, None, 1)

            with patch("tools.graph.graph_relation_candidate_extractor.resolve_live_api", return_value=(True, "test")), patch(
                "tools.graph.graph_relation_candidate_extractor.build_provider",
                return_value=OrphanProvider(),
            ):
                build_graph_relation_candidates(
                    workspace,
                    project_root=Path(__file__).resolve().parents[1],
                    output_dir=output_dir,
                    provider="openai",
                    route_decisions_path=route_path,
                    duplicate_policy="overwrite_generated",
                )

            relations = read_jsonl(output_dir / "graph_relation_candidates.jsonl")
            failures = read_jsonl(output_dir / "graph_extraction_failures.jsonl")
            self.assertEqual(relations, [])
            self.assertTrue(any(row["failure_kind"] == "endpoint_validation_failed" for row in failures))

    def test_openai_provider_marks_context_quote_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_context_workspace"
            output_dir = workspace / "graph_v03_construction"
            workspace.mkdir(parents=True)
            packet = {
                "schema_version": "graph_v03.construction_packet.v0.1",
                "packet_id": "graph_packet:context",
                "workspace_id": workspace.name,
                "modeled_user_id": "Jon",
                "input_kind": "evidence_item",
                "input_ref": "evidence:test:context",
                "original_text": "What business are you thinking of?",
                "processed_text": "",
                "graph_extraction_text": "Gina: What business are you thinking of?\nJon: I'm starting a dance studio.",
                "evidence_refs": ["evidence:test:context"],
                "source_refs": ["source:test"],
                "raw_backpointer_refs": ["raw:test:context"],
                "source_perspective": "Gina",
                "subject_role": "other_participant",
                "attribution_status": "source_text_only",
                "temporal_scope": {},
                "warnings": ["context_used_for_routing_not_primary_evidence"],
                "graph_is_not_proof": True,
            }
            write_jsonl(output_dir / "graph_construction_packets.jsonl", [packet])
            route_path = output_dir / "graph_route_decisions.jsonl"
            write_jsonl(route_path, [{"source_packet_id": "graph_packet:context", "recommended_route": "strong_llm_graph_extraction"}])

            class ContextProvider:
                def generate(self, *, prompt, model_id, input_packet):
                    output = json.dumps(
                        {
                            "output_kind": "graph_bundle_candidate",
                            "entity_candidates": [
                                {
                                    "local_entity_id": "e1",
                                    "name": "Jon",
                                    "entity_type_hint": "person",
                                    "source_text_quote": "Jon",
                                    "attribution_status": "context_supported",
                                    "confidence_hint": "medium",
                                    "warnings": [],
                                },
                                {
                                    "local_entity_id": "e2",
                                    "name": "dance studio",
                                    "entity_type_hint": "project",
                                    "source_text_quote": "dance studio",
                                    "attribution_status": "context_supported",
                                    "confidence_hint": "medium",
                                    "warnings": [],
                                },
                            ],
                            "relation_candidates": [
                                {
                                    "source_local_entity_id": "e1",
                                    "target_local_entity_id": "e2",
                                    "relation_type_hint": "starts",
                                    "relation_description": "context relation",
                                    "why_related": "Supported only by the neighbor turn.",
                                    "directionality_status": "directed",
                                    "source_text_quote": "I'm starting a dance studio.",
                                    "attribution_status": "context_supported",
                                    "confidence_hint": "medium",
                                    "warnings": [],
                                }
                            ],
                            "claim_candidates": [],
                            "warnings": [],
                            "graph_is_not_proof": True,
                        },
                        ensure_ascii=False,
                    )
                    return ProviderResult(output, model_id, "openai", 100, 80, None, 1)

            with patch("tools.graph.graph_relation_candidate_extractor.resolve_live_api", return_value=(True, "test")), patch(
                "tools.graph.graph_relation_candidate_extractor.build_provider",
                return_value=ContextProvider(),
            ):
                build_graph_relation_candidates(
                    workspace,
                    project_root=Path(__file__).resolve().parents[1],
                    output_dir=output_dir,
                    provider="openai",
                    route_decisions_path=route_path,
                    duplicate_policy="overwrite_generated",
                )

            relations = read_jsonl(output_dir / "graph_relation_candidates.jsonl")
            self.assertIn("context_dependency_warning", relations[0]["warnings"])
            self.assertEqual(relations[0]["evidence_role"], "context")
            self.assertEqual(relations[0]["evidence_refs"], [])
            self.assertEqual(relations[0]["context_evidence_refs"], [])

    def test_openai_provider_accepts_approximate_context_quote_with_warning(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "graph_context_approx_workspace"
            output_dir = workspace / "graph_v03_construction"
            workspace.mkdir(parents=True)
            packet = {
                "schema_version": "graph_v03.construction_packet.v0.1",
                "packet_id": "graph_packet:context_approx",
                "workspace_id": workspace.name,
                "modeled_user_id": "Lucilla",
                "input_kind": "evidence_item",
                "input_ref": "evidence:test:context_approx:primary",
                "original_text": "Altogether the picture was a very pretty one.",
                "processed_text": "",
                "graph_extraction_text": (
                    "unknown: Lucilla must sacrifice her own feelings , and make a cheerful home for papa .\n"
                    "unknown: Altogether the picture was a very pretty one."
                ),
                "primary_evidence_refs": ["evidence:test:context_approx:primary"],
                "evidence_refs": ["evidence:test:context_approx:primary"],
                "context_evidence_refs": ["evidence:test:context_approx:ctx1"],
                "source_refs": ["source:test"],
                "raw_backpointer_refs": ["raw:test:context_approx"],
                "source_perspective": "unknown",
                "subject_role": "unknown",
                "attribution_status": "source_text_only",
                "temporal_scope": {},
                "warnings": ["context_used_for_routing_not_primary_evidence"],
                "graph_is_not_proof": True,
            }
            write_jsonl(output_dir / "graph_construction_packets.jsonl", [packet])
            route_path = output_dir / "graph_route_decisions.jsonl"
            write_jsonl(route_path, [{"source_packet_id": "graph_packet:context_approx", "recommended_route": "strong_llm_graph_extraction"}])

            class ApproxContextProvider:
                def generate(self, *, prompt, model_id, input_packet):
                    output = json.dumps(
                        {
                            "output_kind": "graph_bundle_candidate",
                            "entity_candidates": [
                                {
                                    "local_entity_id": "e1",
                                    "name": "Lucilla",
                                    "entity_type_hint": "person",
                                    "source_text_quote": "Lucilla",
                                    "attribution_status": "context_supported",
                                    "confidence_hint": "medium",
                                    "warnings": [],
                                },
                                {
                                    "local_entity_id": "e2",
                                    "name": "papa",
                                    "entity_type_hint": "person",
                                    "source_text_quote": "papa",
                                    "attribution_status": "context_supported",
                                    "confidence_hint": "medium",
                                    "warnings": [],
                                },
                            ],
                            "relation_candidates": [
                                {
                                    "source_local_entity_id": "e1",
                                    "target_local_entity_id": "e2",
                                    "relation_type_hint": "comforter_of",
                                    "relation_description": "Lucilla wants to make a cheerful home for papa.",
                                    "why_related": "Supported by the context sentence.",
                                    "directionality_status": "directed",
                                    "source_text_quote": "she must sacrifice her feelings and make a cheerful home for papa",
                                    "attribution_status": "context_supported",
                                    "confidence_hint": "medium",
                                    "warnings": [],
                                }
                            ],
                            "claim_candidates": [],
                            "warnings": [],
                            "graph_is_not_proof": True,
                        },
                        ensure_ascii=False,
                    )
                    return ProviderResult(output, model_id, "openai", 100, 80, None, 1)

            with patch("tools.graph.graph_relation_candidate_extractor.resolve_live_api", return_value=(True, "test")), patch(
                "tools.graph.graph_relation_candidate_extractor.build_provider",
                return_value=ApproxContextProvider(),
            ):
                build_graph_relation_candidates(
                    workspace,
                    project_root=Path(__file__).resolve().parents[1],
                    output_dir=output_dir,
                    provider="openai",
                    route_decisions_path=route_path,
                    duplicate_policy="overwrite_generated",
                )

            relations = read_jsonl(output_dir / "graph_relation_candidates.jsonl")
            failures = read_jsonl(output_dir / "graph_extraction_failures.jsonl")
            self.assertEqual(failures, [])
            self.assertEqual(relations[0]["evidence_role"], "context_approximate")
            self.assertEqual(relations[0]["evidence_refs"], ["evidence:test:context_approx:ctx1"])
            self.assertIn("relation_quote_from_context", relations[0]["warnings"])
            self.assertIn("relation_quote_approximate_match", relations[0]["warnings"])


if __name__ == "__main__":
    unittest.main()
