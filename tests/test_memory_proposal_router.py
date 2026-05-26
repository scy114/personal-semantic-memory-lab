import json
import shutil
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import yaml

from tools.memory_proposal_router import (
    RegexLexiconFeatureExtractor,
    build_feature_extractor,
    build_router_backend,
    input_from_text_unit,
    load_policy,
    run_router,
)


ROOT = Path(__file__).resolve().parents[1]


def make_router_workspace(parent: Path, name: str = "router_fixture") -> Path:
    workspace = parent / name
    org = workspace / "raw" / "organization"
    org.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "text_unit_id": "text-unit:public:001",
            "unit_type": "sentence",
            "text": "Mira decided to use the local notebook for project planning because privacy matters.",
            "section_type": "body",
            "perspective": "target",
            "segmentation_confidence": "high",
            "retrieval_policy": "default_retrieval",
            "s2_policy": "candidate_allowed",
            "subject_role": "target",
        },
        {
            "text_unit_id": "text-unit:public:002",
            "unit_type": "sentence",
            "text": "Header",
            "section_type": "metadata",
            "perspective": "unknown",
            "segmentation_confidence": "high",
            "retrieval_policy": "exclude_from_retrieval",
            "s2_policy": "exclude",
        },
    ]
    (org / "text_units.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return workspace


class MemoryProposalRouterTests(unittest.TestCase):
    def test_dry_run_router_writes_multi_target_logs_without_writes_or_llm(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            workspace = make_router_workspace(Path(temp_dir))
            try:
                result = run_router(
                    Namespace(
                        project_root=str(ROOT),
                        workspace=str(workspace),
                        output_dir=str(workspace / "routing" / "memory_proposal_router_test"),
                        run_id="memory-proposal-router:test:generic",
                        policy="configs/routing/memory_proposal_router/heuristic_v0.1.yaml",
                        target_tasks="s1_memory_candidate,s2_portrait_candidate,graph_relation_candidate,s0b_section_review,support_check_candidate",
                        duplicate_policy="fail",
                    )
                )
                decisions_path = Path(result["route_decisions"])
                manifest_path = Path(result["route_run_manifest"])
                summary_path = Path(result["route_summary"])
                self.assertTrue(decisions_path.exists())
                self.assertTrue(manifest_path.exists())
                self.assertTrue(summary_path.exists())

                decisions = [
                    json.loads(line)
                    for line in decisions_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

                self.assertGreater(len(decisions), 0)
                self.assertFalse(manifest["llm_calls_executed"])
                self.assertFalse(manifest["proposal_generation_executed"])
                self.assertFalse(manifest["durable_writes_executed"])
                self.assertFalse(manifest["automatic_cascade_executed"])
                self.assertFalse(manifest["s3_hypothesis_candidate_emitted"])
                self.assertFalse(manifest["support_check_candidate_sets_support_status"])
                self.assertEqual(manifest["route_confidence_type"], "heuristic_score_uncalibrated")
                self.assertTrue(Path(manifest["router_policy_path"]).exists())
                self.assertIn("router_policy_hash", manifest)
                self.assertEqual(len(manifest["router_policy_hash"]), 64)
                self.assertIn("lexicon_hash", manifest)
                self.assertEqual(len(manifest["lexicon_hash"]), 64)
                self.assertEqual(manifest["matcher_version"], "regex_word_boundary_zh_phrase.v0.1")

                task_routes = [route for decision in decisions for route in decision["task_routes"]]
                task_names = {route["target_task"] for route in task_routes}
                self.assertEqual(
                    task_names,
                    {
                        "s1_memory_candidate",
                        "s2_portrait_candidate",
                        "graph_relation_candidate",
                        "s0b_section_review",
                        "support_check_candidate",
                    },
                )
                self.assertNotIn("s3_hypothesis_candidate", task_names)
                self.assertTrue(all(route["write_permission"] is False for route in task_routes))
                self.assertTrue(all(route["route_confidence_type"] == "heuristic_score_uncalibrated" for route in task_routes))
                self.assertTrue(all(route["route_confidence"] is None for route in task_routes))
                self.assertTrue(all("route_score" in route and "route_score_max" in route for route in task_routes))
                self.assertTrue(all("support_status" not in route for route in task_routes))
                support_routes = [route for route in task_routes if route["target_task"] == "support_check_candidate"]
                self.assertTrue(
                    all("support_check_candidate_does_not_set_support_status" in route["warnings"] for route in support_routes)
                )
                self.assertTrue(any(route["recommended_route"] == "skip_or_background_only" for route in task_routes))

                summary = summary_path.read_text(encoding="utf-8")
                self.assertIn("No LLM calls executed.", summary)
                self.assertIn("No durable writes executed.", summary)
                self.assertIn("support_check_candidate", summary)
            finally:
                if workspace.exists():
                    shutil.rmtree(workspace)

    def test_v0_1_refuses_s3_hypothesis_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            workspace = make_router_workspace(Path(temp_dir))
            with self.assertRaises(ValueError):
                run_router(
                    Namespace(
                        project_root=str(ROOT),
                        workspace=str(workspace),
                        output_dir=str(Path(temp_dir) / "router_outputs"),
                        run_id="memory-proposal-router:test:s3-refusal",
                        policy="configs/routing/memory_proposal_router/heuristic_v0.1.yaml",
                        target_tasks="s1_memory_candidate,s3_hypothesis_candidate",
                        duplicate_policy="fail",
                    )
                )

    def test_lexicons_are_loaded_from_config_and_word_boundaries_are_used(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_v0.1.yaml")
        extractor = RegexLexiconFeatureExtractor(policy)

        self.assertEqual(extractor.term_count("gift", "logical_markers"), 0)
        self.assertEqual(extractor.term_count("if the plan changes", "logical_markers"), 1)
        self.assertEqual(extractor.term_count("因为计划变了，所以需要调整。", "logical_markers"), 2)

        source = (ROOT / "tools" / "memory_proposal_router.py").read_text(encoding="utf-8")
        self.assertNotIn("ROLE_TERMS =", source)
        self.assertNotIn("LOGICAL_MARKERS =", source)

    def test_unknown_missing_metadata_does_not_force_high_risk_route(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_v0.1.yaml")
        extractor = RegexLexiconFeatureExtractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "Mira wrote a short note.",
            {
                "section_type": "unknown",
                "perspective": "unknown",
                "confidence": "medium",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
            },
        )
        route = backend.route_for_task("route-run:test", "span:test", "s1_memory_candidate", features)

        self.assertIn("unknown_missing_metadata", features["unknown_metadata_reasons"])
        self.assertNotEqual(route["recommended_route"], "strong_llm_proposal")
        self.assertNotEqual(route["recommended_route"], "human_review")

    def test_path_traversal_in_raw_backpointer_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            org = workspace / "raw" / "organization"
            org.mkdir(parents=True)
            outside = Path(temp_dir) / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            section_row = {
                "raw_span_id": "raw:test:text_span:0001",
                "final_section_type": "body",
                "perspective": "author",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "raw_backpointer": {
                    "source_file": str(outside),
                    "locator": {"kind": "text_span", "char_start": 0, "char_end": 7},
                },
            }
            (org / "section_map.jsonl").write_text(json.dumps(section_row, ensure_ascii=False) + "\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                run_router(
                    Namespace(
                        project_root=str(ROOT),
                        workspace=str(workspace),
                        output_dir=str(Path(temp_dir) / "router_outputs"),
                        run_id="memory-proposal-router:test:path-safety",
                        policy="configs/routing/memory_proposal_router/heuristic_v0.1.yaml",
                        target_tasks="s1_memory_candidate",
                        duplicate_policy="fail",
                    )
                )

    def test_duplicate_outputs_obey_duplicate_policy(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            workspace = make_router_workspace(Path(temp_dir))
            output_dir = workspace / "routing" / "memory_proposal_router_test"
            args = Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                output_dir=str(output_dir),
                run_id="memory-proposal-router:test:duplicate-policy",
                policy="configs/routing/memory_proposal_router/heuristic_v0.1.yaml",
                target_tasks="s1_memory_candidate",
                duplicate_policy="fail",
            )
            run_router(args)
            with self.assertRaises(FileExistsError):
                run_router(args)

    def test_policy_schema_validation_rejects_missing_required_keys(self):
        policy_path = ROOT / "configs" / "routing" / "memory_proposal_router" / "heuristic_v0.1.yaml"
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        policy.pop("routes")
        with tempfile.TemporaryDirectory() as temp_dir:
            bad_policy = Path(temp_dir) / "bad_policy.yaml"
            bad_policy.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy(ROOT, str(bad_policy))

    def test_workspace_must_be_inside_project_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            outside_workspace = Path(temp_dir) / "outside_workspace"
            outside_workspace.mkdir()
            with self.assertRaises(ValueError):
                run_router(
                    Namespace(
                        project_root=str(ROOT),
                        workspace=str(outside_workspace),
                        output_dir=None,
                        run_id="memory-proposal-router:test:workspace-safety",
                        policy="configs/routing/memory_proposal_router/heuristic_v0.1.yaml",
                        target_tasks="s1_memory_candidate",
                        duplicate_policy="fail",
                    )
                )

    def test_output_dir_must_be_inside_project_or_workspace(self):
        with tempfile.TemporaryDirectory() as temp_dir, tempfile.TemporaryDirectory(dir=ROOT) as project_temp_dir:
            workspace = make_router_workspace(Path(project_temp_dir))
            with self.assertRaises(ValueError):
                run_router(
                    Namespace(
                        project_root=str(ROOT),
                        workspace=str(workspace),
                        output_dir=str(Path(temp_dir) / "outside_outputs"),
                        run_id="memory-proposal-router:test:output-safety",
                        policy="configs/routing/memory_proposal_router/heuristic_v0.1.yaml",
                        target_tasks="s1_memory_candidate",
                        duplicate_policy="fail",
                    )
                )

    def test_long_simple_raw_span_routes_to_segmentation_not_strong_llm(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_v0.1.yaml")
        extractor = RegexLexiconFeatureExtractor(policy)
        backend = build_router_backend(policy)
        text = "Mira wrote a project note. " * 80
        features = extractor.extract(
            text,
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "source_span_fallback",
            },
        )
        route = backend.route_for_task("route-run:test", "span:long-simple", "s1_memory_candidate", features)

        self.assertTrue(features["source_span_is_coarse"])
        self.assertIn("source_span_char_count", features)
        self.assertIn("proposal_unit_char_count", features)
        self.assertEqual(route["recommended_route"], "split_or_segment_first")
        self.assertIn("needs_proposal_unit_segmentation", route["warnings"])
        self.assertNotEqual(route["recommended_route"], "strong_llm_proposal")

    def test_short_dense_relationship_sentence_can_escalate(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_v0.1.yaml")
        extractor = RegexLexiconFeatureExtractor(policy)
        backend = build_router_backend(policy)
        text = "Mira's boss and teacher said the project is important because the client needs it urgently."
        features = extractor.extract(
            text,
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )
        route = backend.route_for_task("route-run:test", "span:short-dense", "graph_relation_candidate", features)

        self.assertFalse(features["source_span_is_coarse"])
        self.assertGreater(features["role_terms_per_100_tokens"], 0)
        self.assertGreater(features["logical_markers_per_100_tokens"], 0)
        self.assertIn(route["recommended_route"], {"strong_llm_proposal", "human_review"})

    def test_density_features_are_computed(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_v0.1.yaml")
        extractor = RegexLexiconFeatureExtractor(policy)
        features = extractor.extract(
            "Mira said because the project is important.",
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )

        self.assertGreater(features["event_terms_per_100_tokens"], 0)
        self.assertGreater(features["logical_markers_per_100_tokens"], 0)
        self.assertGreater(features["importance_terms_per_100_tokens"], 0)
        self.assertIn("max_sentence_char_count", features)
        self.assertIn("avg_sentence_char_count", features)

    def test_short_proposal_unit_with_long_parent_span_does_not_route_to_segmentation(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_v0.1.yaml")
        extractor = RegexLexiconFeatureExtractor(policy)
        backend = build_router_backend(policy)
        parent_span = "Mira wrote a project note. " * 80
        proposal_unit = "Mira wrote a project note."
        features = extractor.extract(
            proposal_unit,
            {
                "source_span_text": parent_span,
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )
        route = backend.route_for_task("route-run:test", "span:short-unit", "s1_memory_candidate", features)

        self.assertGreater(features["source_span_char_count"], features["proposal_unit_char_count"])
        self.assertFalse(features["source_span_is_coarse"])
        self.assertNotEqual(route["recommended_route"], "split_or_segment_first")

    def test_s2_portrait_skips_non_target_conversation_evidence(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_v0.1.yaml")
        extractor = RegexLexiconFeatureExtractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "Jon loves running his own dance studio.",
            {
                "section_type": "conversation",
                "perspective": "other_participant",
                "confidence": 1.0,
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "evidence_item",
                "subject_role": "other_participant",
                "subject_contamination_risk": "medium",
            },
        )
        route = backend.route_for_task("route-run:test", "evidence:test", "s2_portrait_candidate", features)

        self.assertTrue(features["is_non_target_subject"])
        self.assertEqual(route["recommended_route"], "skip_or_background_only")
        self.assertIn("non_target_subject", route["routing_reasons"])
        self.assertIn("non_target_subject_skipped_for_s2_portrait", route["warnings"])

    def test_text_unit_input_preserves_dialogue_metadata(self):
        row = {
            "text_unit_id": "evidence:locomo:test:D1:7:TURN.S0002",
            "raw_span_id": "evidence:locomo:test:D1:7",
            "unit_type": "sentence",
            "text": 'We just did a contemporary piece called "Finding Freedom."',
            "evidence_ref": "evidence:locomo:test:D1:7",
            "raw_backpointer": {
                "source_file": None,
                "locator": {
                    "kind": "conversation_turn_sentence",
                    "evidence_ref": "evidence:locomo:test:D1:7",
                    "char_start": 8,
                    "char_end": 64,
                },
            },
            "section_type": "conversation",
            "perspective": "target",
            "segmentation_confidence": "medium",
            "retrieval_policy": "default_retrieval",
            "s2_policy": "candidate_allowed",
            "speaker": "Gina",
            "subject_role": "target",
            "target_participant": "Gina",
            "target_subject_ids": ["Gina"],
            "subject_ids": ["Gina"],
            "subject_contamination_risk": "low",
            "parent_text_unit_id": "evidence:locomo:test:D1:7:TURN",
            "previous_text_unit_id": "evidence:locomo:test:D1:7:TURN.S0001",
            "next_text_unit_id": "evidence:locomo:test:D1:7:TURN.S0003",
            "warnings": [],
        }

        item = input_from_text_unit(row)

        self.assertEqual(item["source_layer"], "s0b_text_unit")
        self.assertEqual(item["text_unit_id"], row["text_unit_id"])
        self.assertEqual(item["evidence_ref"], "evidence:locomo:test:D1:7")
        self.assertEqual(item["speaker"], "Gina")
        self.assertEqual(item["subject_role"], "target")
        self.assertEqual(item["target_participant"], "Gina")
        self.assertEqual(item["target_subject_ids"], ["Gina"])
        self.assertEqual(item["subject_ids"], ["Gina"])
        self.assertEqual(item["subject_contamination_risk"], "low")
        self.assertEqual(item["unit_type"], "sentence")
        self.assertEqual(item["parent_text_unit_id"], row["parent_text_unit_id"])

    def test_salience_v0_2_policy_loads_shared_salience_features(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        features = extractor.extract(
            "我这个月失业了，正在准备重新找工作。",
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )

        self.assertEqual(features["salience_feature_version"], "salience_v0.2")
        self.assertTrue(features["salience_contains_chinese"])
        self.assertGreater(features["first_person_term_count"], 0)
        self.assertIn("salience_group_scores", features)
        self.assertGreater(features["salience_value_score"], 0)

    def test_salience_v0_2_routes_low_value_text_to_background_for_s1_and_s2(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "谢谢！",
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )

        s1_route = backend.route_for_task("route-run:test", "span:zh-thanks", "s1_memory_candidate", features)
        s2_route = backend.route_for_task("route-run:test", "span:zh-thanks", "s2_portrait_candidate", features)

        self.assertEqual(s1_route["recommended_route"], "skip_or_background_only")
        self.assertEqual(s2_route["recommended_route"], "skip_or_background_only")
        self.assertIn("low_salience_or_low_value", s1_route["warnings"])
        self.assertIn("low_salience_or_low_value", s2_route["warnings"])

    def test_salience_v0_2_s1_keeps_chinese_first_person_event(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "我这个月失业了，正在准备重新找工作。",
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )

        route = backend.route_for_task("route-run:test", "span:zh-job-loss", "s1_memory_candidate", features)

        self.assertIn(route["recommended_route"], {"script_only", "weak_llm_proposal"})
        self.assertNotEqual(route["recommended_route"], "skip_or_background_only")
        self.assertIn("salience_value_signal", route["routing_reasons"])

    def test_salience_v0_2_s1_low_value_fragments_skip_instead_of_script(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        cases = [
            ("2nd.", {}),
            ("Ob.", {}),
            ("1684.", {}),
            ("--WOOD'S ATHENAE.]", {"warnings": ["unbalanced_bracket_fragment"]}),
        ]

        for text, metadata in cases:
            with self.subTest(text=text):
                features = extractor.extract(
                    text,
                    {
                        "section_type": "body",
                        "perspective": "author",
                        "confidence": "high",
                        "retrieval_policy": "default_retrieval",
                        "s2_policy": "candidate_allowed",
                        "unit_type": "sentence",
                        **metadata,
                    },
                )
                route = backend.route_for_task("route-run:test", f"span:{text}", "s1_memory_candidate", features)

                self.assertEqual(route["recommended_route"], "skip_or_background_only")
                self.assertIn("s1_low_value_not_materialized", route["warnings"])
                self.assertIn("s1_low_modeling_value", route["routing_reasons"])

    def test_salience_v0_2_s1_high_value_low_risk_routes_to_script(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "I launched my clothing store this month.",
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )

        route = backend.route_for_task("route-run:test", "span:s1-low-risk-value", "s1_memory_candidate", features)

        self.assertEqual(route["recommended_route"], "script_only")
        self.assertIn("s1_high_value_low_risk", route["routing_reasons"])

    def test_salience_v0_2_s1_high_value_high_risk_routes_to_strong_llm(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "I reported a privacy risk in the project because the database may crash.",
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )

        route = backend.route_for_task("route-run:test", "span:s1-high-risk-value", "s1_memory_candidate", features)

        self.assertEqual(route["recommended_route"], "strong_llm_proposal")
        self.assertIn("s1_high_value_high_risk", route["routing_reasons"])
        self.assertIn("s1_high_value_high_risk_requires_strong_llm", route["warnings"])

    def test_salience_v0_2_s2_routes_other_directed_compliment_to_weak_review(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "Wow Jon, you're so talented! Keep it up.",
            {
                "section_type": "conversation",
                "perspective": "target",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
                "subject_role": "target",
                "subject_contamination_risk": "medium",
            },
        )

        route = backend.route_for_task("route-run:test", "span:jon-compliment", "s2_portrait_candidate", features)

        self.assertEqual(features["first_person_term_count"], 0)
        self.assertGreater(features["other_person_term_count"], 0)
        self.assertEqual(route["recommended_route"], "weak_llm_proposal")
        self.assertIn("other_directed_interaction_review_signal", route["routing_reasons"])

    def test_salience_v0_2_demonstrative_project_text_is_not_other_person_review(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "这个项目的隐私风险比较高，不能直接上线。",
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )

        route = backend.route_for_task("route-run:test", "span:zh-project-risk", "s2_portrait_candidate", features)

        self.assertEqual(features["first_person_term_count"], 0)
        self.assertEqual(features["other_person_term_count"], 0)
        self.assertNotIn("other_directed_interaction_review_signal", route["routing_reasons"])

    def test_salience_v0_2_plaintext_author_other_directed_content_gets_review_not_skip(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "Wow Jon, you're talented; keep it up.",
            {
                "section_type": "body",
                "perspective": "author",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "sentence",
            },
        )

        route = backend.route_for_task("route-run:test", "span:plaintext-jon-compliment", "s2_portrait_candidate", features)

        self.assertEqual(route["recommended_route"], "weak_llm_proposal")
        self.assertIn("other_directed_interaction_review_signal", route["routing_reasons"])

    def test_salience_v0_2_s2_low_value_memory_unit_skips(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "Thanks.",
            {
                "source_layer": "s1_memory_unit",
                "memory_class": "episodic",
                "section_type": "body",
                "perspective": "target",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "s1_memory_unit",
                "subject_role": "target",
                "evidence_refs": ["evidence:test"],
            },
        )

        route = backend.route_for_task("route-run:test", "memory:thanks", "s2_portrait_candidate", features)

        self.assertEqual(route["recommended_route"], "skip_or_background_only")
        self.assertIn("s2_low_portrait_value", route["routing_reasons"])
        self.assertIn("s2_low_value_not_modeled", route["warnings"])

    def test_salience_v0_2_s2_direct_preference_memory_can_use_script_lane(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "I prefer quiet morning writing sessions.",
            {
                "source_layer": "s1_memory_unit",
                "memory_class": "preference",
                "section_type": "body",
                "perspective": "target",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "s1_memory_unit",
                "subject_role": "target",
                "evidence_refs": ["evidence:test"],
            },
        )

        route = backend.route_for_task("route-run:test", "memory:preference", "s2_portrait_candidate", features)

        self.assertEqual(route["recommended_route"], "script_only")
        self.assertIn("s2_high_value_low_risk_script_eligible", route["routing_reasons"])
        self.assertIn("s2_direct_low_risk_script_lane", route["warnings"])

    def test_salience_v0_2_s2_s1_memory_author_he_is_target_source_not_other_directed(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "The author says he rose this morning and walked to the office.",
            {
                "source_layer": "s1_memory_unit",
                "memory_class": "episodic",
                "section_type": "body",
                "perspective": "target",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "s1_memory_unit",
                "subject_role": "target",
                "evidence_refs": ["evidence:test"],
            },
        )

        route = backend.route_for_task("route-run:test", "memory:author-he", "s2_portrait_candidate", features)

        self.assertEqual(route["recommended_route"], "weak_llm_proposal")
        self.assertIn("source_perspective_or_first_person_signal", route["routing_reasons"])
        self.assertNotIn("other_directed_interaction_review_signal", route["routing_reasons"])

    def test_salience_v0_2_s2_high_risk_memory_routes_strong(self):
        policy = load_policy(ROOT, "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml")
        extractor = build_feature_extractor(policy)
        backend = build_router_backend(policy)
        features = extractor.extract(
            "I am worried the project has privacy risk and may fail.",
            {
                "source_layer": "s1_memory_unit",
                "memory_class": "constraint",
                "section_type": "body",
                "perspective": "target",
                "confidence": "high",
                "retrieval_policy": "default_retrieval",
                "s2_policy": "candidate_allowed",
                "unit_type": "s1_memory_unit",
                "subject_role": "target",
                "evidence_refs": ["evidence:test"],
            },
        )

        route = backend.route_for_task("route-run:test", "memory:risk", "s2_portrait_candidate", features)

        self.assertEqual(route["recommended_route"], "strong_llm_proposal")
        self.assertIn("s2_high_value_high_promotion_risk", route["routing_reasons"])
        self.assertIn("s2_high_value_high_promotion_risk_requires_strong_llm", route["warnings"])

    def test_s2_route_prefers_s1_memory_units_when_available(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp_dir:
            workspace = Path(temp_dir) / "workspace"
            (workspace / "raw" / "organization").mkdir(parents=True)
            (workspace / "memory").mkdir(parents=True)
            (workspace / "raw" / "organization" / "text_units.jsonl").write_text(
                json.dumps(
                    {
                        "text_unit_id": "raw:text-unit:1",
                        "unit_type": "sentence",
                        "text": "This raw S0B sentence should not be the S2 route input.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (workspace / "memory" / "memory_units.jsonl").write_text(
                json.dumps(
                    {
                        "memory_id": "memory:test-preference",
                        "content": "I prefer quiet morning writing sessions.",
                        "processed_text": "I prefer quiet morning writing sessions.",
                        "original_text": "I prefer quiet morning writing sessions.",
                        "evidence_refs": ["evidence:test-preference"],
                        "raw_backpointer_refs": [],
                        "memory_class": "preference",
                        "section_type": "body",
                        "retrieval_policy": "default_retrieval",
                        "s2_policy": "candidate_allowed",
                        "subject_id": "subject:test",
                        "target_subject_id": "subject:test",
                        "subject_role": "target",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            result = run_router(
                Namespace(
                    project_root=str(ROOT),
                    workspace=str(workspace),
                    output_dir=str(workspace / "routing" / "s2_matrix_test"),
                    run_id="route-run:test:s2-memory-preferred",
                    policy="configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml",
                    target_tasks="s2_portrait_candidate",
                    duplicate_policy="fail",
                )
            )

            manifest = json.loads(Path(result["route_run_manifest"]).read_text(encoding="utf-8"))
            decisions = [
                json.loads(line)
                for line in Path(result["route_decisions"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(manifest["input_source"], "s1_memory_units_for_s2")
        self.assertEqual(manifest["counts"]["input_items"], 1)
        self.assertEqual(decisions[0]["memory_id"], "memory:test-preference")
        self.assertEqual(decisions[0]["source_layer"], "s1_memory_unit")
        self.assertEqual(decisions[0]["text"], "I prefer quiet morning writing sessions.")
        self.assertEqual(decisions[0]["task_routes"][0]["recommended_route"], "script_only")


if __name__ == "__main__":
    unittest.main()
