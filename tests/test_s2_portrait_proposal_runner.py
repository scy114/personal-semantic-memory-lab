import json
import shutil
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from tools.proposals.proposal_runner import build_input_packets, load_inputs, run_proposal_runner


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def route_decision(text_unit_id: str, route_id: str, route: str, text: str) -> dict:
    return {
        "schema_version": "memory_proposal_router_pilot.v0.1",
        "route_run_id": "memory-proposal-router:test:s2-portrait",
        "span_id": text_unit_id,
        "text_unit_id": text_unit_id,
        "raw_span_id": "raw:test:span:0001",
        "evidence_ref": None,
        "raw_backpointer": {
            "source_file": "raw/test.txt",
            "locator": {"kind": "text_span", "char_start": 0, "char_end": len(text)},
        },
        "source_layer": "s0b_text_unit",
        "section_type": "body",
        "perspective": "author",
        "retrieval_policy": "default_retrieval",
        "s2_policy": "candidate_allowed",
        "unit_type": "sentence",
        "parent_text_unit_id": "tu:p1",
        "previous_text_unit_id": None,
        "next_text_unit_id": None,
        "task_routes": [
            {
                "route_decision_id": route_id,
                "target_task": "s2_portrait_candidate",
                "recommended_route": route,
                "fallback_route": "human_review",
                "route_score": 5,
                "route_score_max": 12,
                "route_confidence": None,
                "route_confidence_type": "heuristic_score_uncalibrated",
                "cost_class": "medium",
                "routing_reasons": ["test_reason"],
                "write_permission": False,
                "warnings": [],
            }
        ],
        "warnings": [],
    }


def route_decision_with_span(span_id: str, route_id: str, route: str, text: str) -> dict:
    row = route_decision(span_id, route_id, route, text)
    row.pop("text_unit_id", None)
    row["span_id"] = span_id
    return row


def evidence_route_decision(evidence_ref: str, route_id: str, route: str) -> dict:
    row = route_decision(evidence_ref, route_id, route, "")
    row.pop("text_unit_id", None)
    row["span_id"] = evidence_ref
    row["raw_span_id"] = None
    row["evidence_ref"] = evidence_ref
    row["source_layer"] = "s1_raw_evidence"
    row["unit_type"] = None
    row["perspective"] = "target"
    return row


def memory_route_decision(memory_id: str, route_id: str, route: str, processed_text: str, original_text: str) -> dict:
    row = route_decision(memory_id, route_id, route, processed_text)
    row.pop("text_unit_id", None)
    row["span_id"] = memory_id
    row["memory_id"] = memory_id
    row["text"] = processed_text
    row["processed_text"] = processed_text
    row["original_text"] = original_text
    row["evidence_ref"] = "evidence:memory-original"
    row["evidence_refs"] = ["evidence:memory-original"]
    row["raw_backpointer_refs"] = [{"source_file": "raw/test.txt", "locator": {"kind": "text_span", "char_start": 0, "char_end": len(original_text)}}]
    row["source_layer"] = "s1_memory_unit"
    row["unit_type"] = "s1_memory_unit"
    row["perspective"] = "target"
    row["subject_role"] = "target"
    return row


def text_unit(text_unit_id: str, text: str, parent: str | None = "tu:p1", **overrides) -> dict:
    row = {
        "schema_version": "s0b.text_unit.v0.1",
        "text_unit_id": text_unit_id,
        "unit_type": "sentence" if parent else "paragraph",
        "text": text,
        "parent_text_unit_id": parent,
        "previous_text_unit_id": None,
        "next_text_unit_id": None,
        "raw_span_id": "raw:test:span:0001",
        "raw_backpointer": {
            "source_file": "raw/test.txt",
            "locator": {"kind": "text_span", "char_start": 0, "char_end": len(text)},
        },
        "section_type": "body",
        "perspective": "author",
        "s1_storage_policy": "ordinary_evidence",
        "retrieval_policy": "default_retrieval",
        "s2_policy": "candidate_allowed",
        "warnings": [],
    }
    row.update(overrides)
    return row


def evidence_item(evidence_ref: str, text: str, turn_index: int) -> dict:
    return {
        "schema_version": "step1.evidence_item.v1",
        "evidence_ref": evidence_ref,
        "canonical_evidence_ref": evidence_ref,
        "source_type": "conversation",
        "subject_role": "target",
        "text": text,
        "locator": {
            "kind": "conversation_turn",
            "record_id": "test-conv",
            "session": "session_1",
            "turn_index": turn_index,
            "speaker": "Gina",
        },
        "metadata": {"modeled_subject_id": "Gina"},
    }


class S2PortraitProposalRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_s2_portrait_proposal_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        (self.workspace / "raw").mkdir(parents=True)
        (self.workspace / "raw" / "test.txt").write_text("test source", encoding="utf-8")
        rows = [
            text_unit("tu:p1", "I prefer quiet focused work and enjoy careful planning.", None),
            text_unit("tu:s1", "I prefer quiet focused work and enjoy careful planning."),
            text_unit("tu:s2", "This sentence has no durable portrait candidate."),
            text_unit("tu:s3", "This important decision shows I resolved to keep better records."),
            text_unit("tu:s4", "Ambiguous high risk text that needs human review."),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        decisions = [
            route_decision("tu:s1", "route:script", "script_only", rows[1]["text"]),
            route_decision("tu:s2", "route:weak", "weak_llm_proposal", rows[2]["text"]),
            route_decision("tu:s3", "route:strong", "strong_llm_proposal", rows[3]["text"]),
            route_decision("tu:s4", "route:human", "human_review", rows[4]["text"]),
        ]
        write_jsonl(self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl", decisions)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def run_runner(
        self,
        provider: str = "mock",
        output_name: str = "s2_portrait_candidate",
        profile: str = "configs/proposals/s2_portrait_candidate.v0.1.json",
        external_model_outputs: str | None = None,
    ) -> dict:
        return run_proposal_runner(
            Namespace(
                project_root=str(ROOT),
                workspace=str(self.workspace),
                profile=profile,
                route_decisions=None,
                output_dir=str(self.workspace / "proposals" / output_name),
                run_id="s2-portrait-proposal:test",
                duplicate_policy="fail",
                provider=provider,
                api_mode="responses",
                allow_live_api=False,
                weak_model="mock-weak",
                strong_model="mock-strong",
                weak_prompt=None,
                strong_prompt=None,
                env_file=None,
                max_items=None,
                external_model_outputs=external_model_outputs,
            )
        )

    def test_mock_runner_writes_reviewable_outputs_without_canonical_writes(self):
        result = self.run_runner()
        output_dir = Path(result["output_dir"])
        proposals_path = output_dir / "portrait_candidate_proposals.ai.jsonl"
        human_queue_path = output_dir / "human_review_queue.jsonl"
        failures_path = output_dir / "model_output_failures.jsonl"
        manifest_path = output_dir / "proposal_run_manifest.json"
        review_log_path = output_dir / "proposal_review_log.jsonl"

        self.assertTrue(proposals_path.exists())
        self.assertTrue(human_queue_path.exists())
        self.assertTrue(failures_path.exists())
        self.assertTrue(manifest_path.exists())
        self.assertTrue(review_log_path.exists())

        proposals = [json.loads(line) for line in proposals_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        human_queue = [json.loads(line) for line in human_queue_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        failures = [json.loads(line) for line in failures_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(len(human_queue), 1)
        self.assertEqual(human_queue[0]["review_status"], "needs_review")
        self.assertFalse(failures)
        self.assertFalse(manifest["durable_writes_executed"])
        self.assertFalse(manifest["reviewed_units_written"])
        self.assertFalse(manifest["current_portrait_written"])
        self.assertFalse(manifest["graph_truth_written"])
        self.assertFalse(manifest["automatic_acceptance_executed"])
        self.assertFalse(manifest["script_only_generates_candidates"])
        self.assertFalse(manifest["human_review_calls_model_by_default"])
        self.assertTrue(manifest["strict_schema_validation"])
        self.assertFalse(manifest["structured_output_enforced"])
        self.assertFalse(manifest["live_api_enabled"])
        self.assertIsNone(manifest["live_api_unlock_source"])
        self.assertFalse(manifest["api_key_recorded"])
        self.assertEqual(manifest["proposal_profile_id"], "s2_portrait_candidate.v0.1")
        self.assertEqual(manifest["api_mode"], "responses")
        self.assertIn("proposal_profile_hash", manifest)
        self.assertIn("input_route_decisions_hash", manifest)
        self.assertEqual(len(manifest["input_route_decisions_hash"]), 64)
        self.assertIn("text_units_hash", manifest)
        self.assertEqual(len(manifest["text_units_hash"]), 64)
        self.assertIn("weak_prompt_hash", manifest)
        self.assertEqual(len(manifest["weak_prompt_hash"]), 64)
        self.assertIn("evidence_hash", manifest)

        self.assertTrue(all(row["write_permission"] is False for row in proposals))
        script_rows = [row for row in proposals if row["route_recommended"] == "script_only"]
        self.assertEqual(script_rows[0]["proposal_status"], "skipped")
        self.assertEqual(script_rows[0]["candidate_type"], "none")
        self.assertIn("script_skipped_or_no_candidate", script_rows[0]["warnings"])

        model_rows = [row for row in proposals if row["provider"] == "mock"]
        self.assertTrue(model_rows)
        self.assertTrue(all(row["route_recommended"] == row["route_used"] for row in model_rows))
        self.assertTrue(all(len(row["source_text_quote"]) <= 300 for row in model_rows))
        self.assertTrue(all(row["review_status"] == "needs_review" for row in proposals))

        self.assertFalse((self.workspace / "portrait" / "reviewed_units.jsonl").exists())
        self.assertFalse((self.workspace / "portrait" / "current_portrait.json").exists())

    def test_invalid_model_output_is_logged_as_failure_not_candidate(self):
        result = self.run_runner(provider="mock_invalid", output_name="invalid_provider")
        output_dir = Path(result["output_dir"])
        failures = [
            json.loads(line)
            for line in (output_dir / "model_output_failures.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreater(len(failures), 0)
        self.assertTrue(all(row["proposal_status"] == "model_failure" for row in failures))
        self.assertTrue(any("invalid_json" in row["warnings"] for row in failures))
        self.assertTrue(all(row["write_permission"] is False for row in failures))

    def test_duplicate_outputs_obey_duplicate_policy(self):
        self.run_runner(output_name="duplicate_policy")
        with self.assertRaises(FileExistsError):
            self.run_runner(output_name="duplicate_policy")

    def test_openai_provider_requires_live_unlock(self):
        with patch.dict("os.environ", {"ALLOW_LIVE_API": ""}, clear=False):
            with self.assertRaises(ValueError):
                self.run_runner(provider="openai", output_name="openai_no_unlock")

    def test_chat_completions_api_mode_is_recorded(self):
        result = run_proposal_runner(
            Namespace(
                project_root=str(ROOT),
                workspace=str(self.workspace),
                profile="configs/proposals/s2_portrait_candidate.v0.1.json",
                route_decisions=None,
                output_dir=str(self.workspace / "proposals" / "chat_mode"),
                run_id="s2-portrait-proposal:test",
                duplicate_policy="fail",
                provider="mock",
                api_mode="chat_completions",
                allow_live_api=False,
                weak_model="mock-weak",
                strong_model="mock-strong",
                weak_prompt=None,
                strong_prompt=None,
                env_file=None,
                max_items=None,
            )
        )
        manifest = json.loads(Path(result["outputs"]["proposal_run_manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["api_mode"], "chat_completions")

    def test_span_id_fallback_matches_text_unit_id(self):
        decisions = [
            route_decision_with_span(
                "tu:s3",
                "route:span-fallback",
                "strong_llm_proposal",
                "This important decision shows I resolved to keep better records.",
            )
        ]
        write_jsonl(self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl", decisions)
        result = self.run_runner(output_name="span_fallback")
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["text_unit_id"], "tu:s3")
        self.assertEqual(proposals[0]["proposal_status"], "candidate")

    def test_evidence_fallback_builds_packets_when_text_units_are_missing(self):
        (self.workspace / "raw" / "organization" / "text_units.jsonl").unlink()
        evidence_rows = [
            evidence_item("evidence:locomo:test:D1:1", "Gina likes careful project notes.", 1),
            evidence_item("evidence:locomo:test:D1:2", "Gina prefers reminders before meetings.", 2),
        ]
        write_jsonl(self.workspace / "evidence" / "evidence.jsonl", evidence_rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [evidence_route_decision("evidence:locomo:test:D1:2", "route:evidence", "weak_llm_proposal")],
        )

        result = self.run_runner(output_name="evidence_fallback")
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads(Path(result["outputs"]["proposal_run_manifest"]).read_text(encoding="utf-8"))

        self.assertEqual(len(proposals), 1)
        self.assertIsNone(proposals[0]["text_unit_id"])
        self.assertEqual(proposals[0]["evidence_refs"], ["evidence:locomo:test:D1:2"])
        self.assertEqual(proposals[0]["proposal_status"], "candidate")
        self.assertEqual(len(manifest["evidence_hash"]), 64)
        self.assertIsNone(manifest["text_units_hash"])

    def test_model_uncertain_becomes_human_review_required_not_reject(self):
        from tools.proposals.proposal_runner import clean_model_status

        self.assertEqual(clean_model_status("model_uncertain"), "human_review_required")

    def test_quote_warning_for_non_exact_quote(self):
        from tools.proposals.proposal_runner import exact_quote_warning

        warnings = exact_quote_warning("not in source", "source text", "context text", 300)
        self.assertIn("source_text_quote_not_exact", warnings)
        self.assertIn("source_text_quote_not_in_primary_text", warnings)

    def test_candidate_quote_outside_primary_text_is_downgraded_for_review(self):
        from tools.proposals.proposal_runner import ProviderResult

        class ContextQuoteProvider:
            def generate(self, *, prompt, model_id, input_packet):
                output = json.dumps(
                    {
                        "proposal_status": "candidate",
                        "candidate_text": "Gina prefers ocean-facing dance studios.",
                        "candidate_type": "preference",
                        "inference_level": "weak_inference",
                        "claim_strength": "direct",
                        "proposal_confidence": "medium",
                        "source_text_quote": "prefers ocean-facing dance studios",
                        "subject_contamination_risk": "low",
                        "privacy_class": "public_dataset",
                        "warnings": [],
                    }
                )
                return ProviderResult(
                    output_text=output,
                    model_id=model_id,
                    provider="mock",
                    estimated_input_tokens=10,
                    estimated_output_tokens=10,
                    cache_hit=None,
                    latency_ms=1,
                )

        rows = [
            text_unit("tu:p1", "Context says Gina prefers ocean-facing dance studios.", None),
            text_unit("tu:s2", "Gina is considering a location."),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:s2", "route:context-quote", "weak_llm_proposal", rows[1]["text"])],
        )

        with patch("tools.proposals.proposal_runner.build_provider", return_value=ContextQuoteProvider()):
            result = self.run_runner(output_name="context_quote_guardrail")
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(len(proposals), 1)
        row = proposals[0]
        self.assertEqual(row["proposal_status"], "human_review_required")
        self.assertEqual(row["candidate_text"], "")
        self.assertEqual(row["candidate_type"], "none")
        self.assertIn("source_text_quote_not_in_primary_text", row["warnings"])
        self.assertIn("candidate_downgraded_quote_guardrail", row["warnings"])
        self.assertFalse(row["write_permission"])

    def test_profile_safety_switches_must_remain_false(self):
        profile_path = self.workspace / "bad_profile.json"
        profile = json.loads((ROOT / "configs" / "proposals" / "s2_portrait_candidate.v0.1.json").read_text(encoding="utf-8"))
        profile["script_only_generates_candidates"] = True
        profile_path.write_text(json.dumps(profile), encoding="utf-8")
        with self.assertRaises(ValueError):
            run_proposal_runner(
                Namespace(
                    project_root=str(ROOT),
                    workspace=str(self.workspace),
                    profile=str(profile_path),
                    route_decisions=None,
                    output_dir=str(self.workspace / "proposals" / "bad_profile"),
                    run_id="s2-portrait-proposal:test",
                    duplicate_policy="fail",
                    provider="mock",
                    api_mode=None,
                    allow_live_api=False,
                    weak_model="mock-weak",
                    strong_model="mock-strong",
                    weak_prompt=None,
                    strong_prompt=None,
                    env_file=None,
                    max_items=None,
                )
            )

    def test_v02_mock_outputs_fact_hypothesis_uncertain_and_human_review(self):
        rows = [
            text_unit("tu:p1", "Conversation context.", None),
            text_unit("tu:fact", "I prefer careful project notes and enjoy planning."),
            text_unit("tu:hyp", "Wow Jon, you're so talented! Keep it up."),
            text_unit("tu:uncertain", "This is uncertain and could mean several things."),
            text_unit("tu:review", "This human review case has attribution and privacy risk."),
            text_unit("tu:reject", "The weather was mild today."),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [
                route_decision("tu:fact", "route:v02-fact", "weak_llm_proposal", rows[1]["text"]),
                route_decision("tu:hyp", "route:v02-hyp", "weak_llm_proposal", rows[2]["text"]),
                route_decision("tu:uncertain", "route:v02-uncertain", "weak_llm_proposal", rows[3]["text"]),
                route_decision("tu:review", "route:v02-review", "strong_llm_proposal", rows[4]["text"]),
                route_decision("tu:reject", "route:v02-reject", "weak_llm_proposal", rows[5]["text"]),
            ],
        )

        result = self.run_runner(
            output_name="v02_mock_cases",
            profile="configs/proposals/s2_portrait_proposal.v0.2.json",
        )
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        by_text_unit = {row["text_unit_id"]: row for row in proposals}
        self.assertEqual(by_text_unit["tu:fact"]["output_kind"], "portrait_fact_candidate")
        self.assertEqual(by_text_unit["tu:fact"]["epistemic_status"], "fact_candidate")
        self.assertTrue(by_text_unit["tu:fact"]["fact_candidate_text"])
        self.assertEqual(by_text_unit["tu:fact"]["hypothesis_text"], "")
        self.assertEqual(by_text_unit["tu:fact"]["candidate_type"], "project_context")

        self.assertEqual(by_text_unit["tu:hyp"]["output_kind"], "portrait_hypothesis_candidate")
        self.assertEqual(by_text_unit["tu:hyp"]["epistemic_status"], "hypothesis")
        self.assertTrue(by_text_unit["tu:hyp"]["hypothesis_text"])
        self.assertEqual(by_text_unit["tu:hyp"]["fact_candidate_text"], "")
        self.assertEqual(by_text_unit["tu:hyp"]["hypothesis_confidence"], "high")
        self.assertEqual(by_text_unit["tu:hyp"]["hypothesis_scope"], "event")
        self.assertEqual(by_text_unit["tu:hyp"]["commitment_level"], "low")
        self.assertEqual(by_text_unit["tu:hyp"]["promotion_readiness"], "not_ready")
        self.assertEqual(by_text_unit["tu:hyp"]["review_status"], "unreviewed_low_commitment")

        self.assertEqual(by_text_unit["tu:uncertain"]["output_kind"], "model_uncertain")
        self.assertEqual(by_text_unit["tu:uncertain"]["epistemic_status"], "uncertain")
        self.assertEqual(by_text_unit["tu:uncertain"]["candidate_type"], "none")
        self.assertEqual(by_text_unit["tu:review"]["output_kind"], "needs_human_review")
        self.assertEqual(by_text_unit["tu:review"]["epistemic_status"], "human_review_required")
        self.assertEqual(by_text_unit["tu:review"]["fact_candidate_text"], "")
        self.assertEqual(by_text_unit["tu:review"]["hypothesis_text"], "")
        self.assertEqual(by_text_unit["tu:review"]["candidate_text"], "")
        self.assertEqual(by_text_unit["tu:review"]["candidate_type"], "none")
        self.assertEqual(by_text_unit["tu:reject"]["output_kind"], "reject")
        self.assertEqual(by_text_unit["tu:reject"]["epistemic_status"], "reject")
        self.assertEqual(by_text_unit["tu:reject"]["fact_candidate_text"], "")
        self.assertEqual(by_text_unit["tu:reject"]["hypothesis_text"], "")
        self.assertEqual(by_text_unit["tu:reject"]["candidate_text"], "")
        self.assertEqual(by_text_unit["tu:reject"]["candidate_type"], "none")
        self.assertTrue(all(row["write_permission"] is False for row in proposals))

    def test_v02_fact_candidate_requires_primary_quote(self):
        from tools.proposals.proposal_runner import ProviderResult

        class ParentQuoteFactProvider:
            def generate(self, *, prompt, model_id, input_packet):
                output = json.dumps(
                    {
                        "output_kind": "portrait_fact_candidate",
                        "fact_candidate_text": "Gina uses dance for stress relief.",
                        "hypothesis_text": "",
                        "source_text_quote": "Dance is pretty much my go-to for stress relief.",
                        "source_text_quotes": [],
                        "supporting_observations": [],
                        "alternative_explanations": [],
                        "uncertainty_notes": [],
                        "candidate_type": "user_state",
                        "inference_level": "explicit",
                        "claim_strength": "direct",
                        "proposal_confidence": "high",
                        "hypothesis_status": "unknown",
                        "hypothesis_confidence": "unknown",
                        "hypothesis_scope": "unknown",
                        "commitment_level": "unknown",
                        "promotion_readiness": "unknown",
                        "subject_contamination_risk": "low",
                        "privacy_class": "public_dataset",
                        "warnings": [],
                    }
                )
                return ProviderResult(output, model_id, "mock", 10, 10, None, 1)

        rows = [
            text_unit("tu:turn", "Wow Jon, same here! Dance is pretty much my go-to for stress relief.", None, unit_type="turn"),
            text_unit("tu:sent", "Wow Jon, same here!", "tu:turn"),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:sent", "route:v02-primary-quote", "weak_llm_proposal", rows[1]["text"])],
        )

        with patch("tools.proposals.proposal_runner.build_provider", return_value=ParentQuoteFactProvider()):
            result = self.run_runner(
                output_name="v02_fact_primary_quote",
                profile="configs/proposals/s2_portrait_proposal.v0.2.json",
            )
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        row = proposals[0]
        self.assertEqual(row["output_kind"], "needs_human_review")
        self.assertEqual(row["fact_candidate_text"], "")
        self.assertEqual(row["candidate_type"], "none")
        self.assertIn("source_text_quote_not_in_primary_text", row["warnings"])
        self.assertIn("fact_candidate_downgraded_quote_guardrail", row["warnings"])

    def test_v02_hypothesis_using_neighbor_context_gets_warning(self):
        from tools.proposals.proposal_runner import ProviderResult

        class ParentQuoteHypothesisProvider:
            def generate(self, *, prompt, model_id, input_packet):
                output = json.dumps(
                    {
                        "output_kind": "portrait_hypothesis_candidate",
                        "fact_candidate_text": "",
                        "hypothesis_text": "Gina may use dance for stress relief.",
                        "source_text_quote": "Dance is pretty much my go-to for stress relief.",
                        "source_text_quotes": [],
                        "supporting_observations": [],
                        "alternative_explanations": [],
                        "uncertainty_notes": [],
                        "candidate_type": "user_state",
                        "inference_level": "direct_inference",
                        "claim_strength": "partial",
                        "proposal_confidence": "medium",
                        "hypothesis_status": "active",
                        "hypothesis_confidence": "medium",
                        "hypothesis_scope": "session",
                        "commitment_level": "low",
                        "promotion_readiness": "not_ready",
                        "subject_contamination_risk": "medium",
                        "privacy_class": "public_dataset",
                        "warnings": [],
                    }
                )
                return ProviderResult(output, model_id, "mock", 10, 10, None, 1)

        rows = [
            text_unit("tu:turn", "Wow Jon, same here! Dance is pretty much my go-to for stress relief.", None, unit_type="turn"),
            text_unit("tu:sent", "Wow Jon, same here!", "tu:turn"),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:sent", "route:v02-hyp-neighbor", "weak_llm_proposal", rows[1]["text"])],
        )

        with patch("tools.proposals.proposal_runner.build_provider", return_value=ParentQuoteHypothesisProvider()):
            result = self.run_runner(
                output_name="v02_hyp_neighbor_warning",
                profile="configs/proposals/s2_portrait_proposal.v0.2.json",
            )
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        row = proposals[0]
        self.assertEqual(row["output_kind"], "portrait_hypothesis_candidate")
        self.assertIn("source_text_quote_not_in_primary_text", row["warnings"])
        self.assertIn("hypothesis_uses_neighbor_context", row["warnings"])

    def test_v02_hypothesis_backfills_primary_quote_from_source_text_quotes(self):
        from tools.proposals.proposal_runner import ProviderResult

        class SourceQuotesOnlyHypothesisProvider:
            def generate(self, *, prompt, model_id, input_packet):
                output = json.dumps(
                    {
                        "output_kind": "portrait_hypothesis_candidate",
                        "fact_candidate_text": "",
                        "hypothesis_text": "Gina is interested in trying new dance moves with the other person.",
                        "source_text_quote": "",
                        "source_text_quotes": ["Let's explore some new dance moves."],
                        "supporting_observations": ["She proposes a shared future dance activity."],
                        "alternative_explanations": [],
                        "uncertainty_notes": [],
                        "candidate_type": "relationship_context",
                        "inference_level": "direct_inference",
                        "claim_strength": "partial",
                        "proposal_confidence": "medium",
                        "hypothesis_status": "active",
                        "hypothesis_confidence": "medium",
                        "hypothesis_scope": "session",
                        "commitment_level": "low",
                        "promotion_readiness": "needs_more_evidence",
                        "subject_contamination_risk": "medium",
                        "privacy_class": "public_dataset",
                        "warnings": [],
                    }
                )
                return ProviderResult(output, model_id, "mock", 10, 10, None, 1)

        rows = [
            text_unit("tu:p1", "Conversation context.", None),
            text_unit("tu:sent", "Let's explore some new dance moves."),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:sent", "route:v02-source-quotes-only", "weak_llm_proposal", rows[1]["text"])],
        )

        with patch("tools.proposals.proposal_runner.build_provider", return_value=SourceQuotesOnlyHypothesisProvider()):
            result = self.run_runner(
                output_name="v02_hyp_quote_backfill",
                profile="configs/proposals/s2_portrait_proposal.v0.2.json",
            )
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        row = proposals[0]
        self.assertEqual(row["output_kind"], "portrait_hypothesis_candidate")
        self.assertEqual(row["source_text_quote"], "Let's explore some new dance moves.")
        self.assertIn("source_text_quote_backfilled_from_source_text_quotes", row["warnings"])
        self.assertNotIn("source_text_quote_not_in_primary_text", row["warnings"])

    def test_v02_upgrades_script_only_target_signal_to_weak_llm(self):
        rows = [
            text_unit("tu:p1", "Conversation context.", None),
            text_unit("tu:script-signal", "I prefer quiet focused work and enjoy careful planning."),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:script-signal", "route:v02-script-signal", "script_only", rows[1]["text"])],
        )

        result = self.run_runner(
            output_name="v02_script_only_upgrade",
            profile="configs/proposals/s2_portrait_proposal.v0.2.json",
        )
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads(Path(result["outputs"]["proposal_run_manifest"]).read_text(encoding="utf-8"))

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["route_recommended"], "script_only")
        self.assertEqual(proposals[0]["route_used"], "weak_llm_proposal")
        self.assertEqual(proposals[0]["override_reason"], "script_only_upgraded_to_weak_for_target_signal")
        self.assertIn("script_only_upgraded_to_weak_for_target_signal", proposals[0]["warnings"])
        self.assertEqual(proposals[0]["output_kind"], "portrait_fact_candidate")
        self.assertTrue(proposals[0]["source_text_preview"])
        self.assertEqual(manifest["counts"]["model_call_rows"], 1)
        self.assertTrue(manifest["script_only_may_upgrade_to_weak"])

    def test_v02_script_only_upgrade_requires_explicit_profile_flag(self):
        rows = [
            text_unit("tu:p1", "Conversation context.", None),
            text_unit("tu:script-signal", "I prefer quiet focused work and enjoy careful planning."),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:script-signal", "route:v02-script-signal", "script_only", rows[1]["text"])],
        )
        profile_path = self.workspace / "s2_v02_no_script_upgrade.json"
        profile = json.loads((ROOT / "configs" / "proposals" / "s2_portrait_proposal.v0.2.json").read_text(encoding="utf-8"))
        profile["script_only_may_upgrade_to_weak"] = False
        profile_path.write_text(json.dumps(profile, ensure_ascii=False), encoding="utf-8")

        result = self.run_runner(
            output_name="v02_script_only_upgrade_disabled",
            profile=str(profile_path),
        )
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads(Path(result["outputs"]["proposal_run_manifest"]).read_text(encoding="utf-8"))

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["route_recommended"], "script_only")
        self.assertEqual(proposals[0]["route_used"], "script_only")
        self.assertEqual(proposals[0]["output_kind"], "skipped")
        self.assertIn("script_skipped_or_no_candidate", proposals[0]["warnings"])
        self.assertFalse(manifest["script_only_may_upgrade_to_weak"])
        self.assertEqual(manifest["counts"]["model_call_rows"], 0)

    def test_text_unit_packet_preserves_dialogue_metadata(self):
        rows = [
            text_unit("tu:turn", "Thanks! We just did a contemporary piece.", None, unit_type="turn"),
            text_unit(
                "tu:sent",
                "We just did a contemporary piece.",
                "tu:turn",
                evidence_ref="evidence:locomo:test:D1:7",
                canonical_evidence_ref="evidence:locomo:test:D1:7",
                section_type="conversation",
                perspective="target",
                speaker="Gina",
                subject_role="target",
                target_participant="Gina",
                target_subject_ids=["Gina"],
                subject_ids=["Gina"],
                subject_contamination_risk="low",
                raw_backpointer={
                    "source_file": None,
                    "locator": {
                        "kind": "conversation_turn_sentence",
                        "evidence_ref": "evidence:locomo:test:D1:7",
                        "char_start": 8,
                        "char_end": 43,
                    },
                },
            ),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:sent", "route:dialogue-metadata", "weak_llm_proposal", rows[1]["text"])],
        )

        inputs = load_inputs(
            Namespace(
                project_root=str(ROOT),
                workspace=str(self.workspace),
                profile="configs/proposals/s2_portrait_proposal.v0.2.json",
                route_decisions=None,
                output_dir=str(self.workspace / "proposals" / "packet_metadata"),
                run_id="s2-portrait-proposal:test:packet-metadata",
                duplicate_policy="fail",
                provider="mock",
                api_mode="responses",
                allow_live_api=False,
                weak_model="mock-weak",
                strong_model="mock-strong",
                weak_prompt=None,
                strong_prompt=None,
                env_file=None,
                max_items=None,
            )
        )
        packets, warnings = build_input_packets(inputs)

        self.assertFalse(warnings)
        self.assertEqual(len(packets), 1)
        packet = packets[0]
        self.assertEqual(packet["evidence_refs"], ["evidence:locomo:test:D1:7"])
        self.assertEqual(packet["speaker"], "Gina")
        self.assertEqual(packet["subject_role"], "target")
        self.assertEqual(packet["target_participant"], "Gina")
        self.assertEqual(packet["target_subject_ids"], ["Gina"])
        self.assertEqual(packet["subject_ids"], ["Gina"])
        self.assertEqual(packet["subject_contamination_risk"], "low")
        self.assertTrue(
            any(ref.get("locator", {}).get("kind") == "conversation_turn_sentence" for ref in packet["raw_backpointer_refs"])
        )

    def test_external_jsonl_provider_replays_outputs_through_validation(self):
        from tools.proposals.proposal_runner import short_hash

        rows = [
            text_unit("tu:p1", "Conversation context.", None),
            text_unit("tu:external", "I prefer careful project notes."),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:external", "route:external", "weak_llm_proposal", rows[1]["text"])],
        )
        proposal_input_id = f"s2pi:{short_hash('s2-portrait-proposal:test:tu:external:route:external')}"
        external_path = self.workspace / "external_model_outputs.jsonl"
        write_jsonl(
            external_path,
            [
                {
                    "proposal_input_id": proposal_input_id,
                    "provider": "subagent",
                    "model_id": "gpt-4o",
                    "output_text": json.dumps(
                        {
                            "output_kind": "portrait_fact_candidate",
                            "fact_candidate_text": "The target subject prefers careful project notes.",
                            "hypothesis_text": "",
                            "source_text_quote": "I prefer careful project notes.",
                            "source_text_quotes": ["I prefer careful project notes."],
                            "supporting_observations": [],
                            "alternative_explanations": [],
                            "uncertainty_notes": [],
                            "candidate_type": "preference",
                            "inference_level": "explicit",
                            "claim_strength": "direct",
                            "proposal_confidence": "high",
                            "hypothesis_status": "unknown",
                            "hypothesis_confidence": "unknown",
                            "hypothesis_scope": "unknown",
                            "commitment_level": "unknown",
                            "promotion_readiness": "unknown",
                            "subject_contamination_risk": "low",
                            "privacy_class": "public_dataset",
                            "warnings": [],
                        }
                    ),
                }
            ],
        )

        result = self.run_runner(
            provider="external_jsonl",
            output_name="external_jsonl_provider",
            profile="configs/proposals/s2_portrait_proposal.v0.2.json",
            external_model_outputs=str(external_path),
        )
        output_dir = Path(result["output_dir"])
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        failures = [
            json.loads(line)
            for line in Path(result["outputs"]["model_output_failures"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads(Path(result["outputs"]["proposal_run_manifest"]).read_text(encoding="utf-8"))
        model_call_inputs = [
            json.loads(line)
            for line in (output_dir / "model_call_inputs.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertFalse(failures)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["provider"], "subagent")
        self.assertEqual(proposals[0]["model_id"], "gpt-4o")
        self.assertEqual(proposals[0]["output_kind"], "portrait_fact_candidate")
        self.assertEqual(proposals[0]["source_text_quote"], "I prefer careful project notes.")
        self.assertEqual(len(model_call_inputs), 1)
        self.assertEqual(model_call_inputs[0]["proposal_input_id"], proposal_input_id)
        self.assertIn("prompt_text", model_call_inputs[0])
        self.assertIn("input_packet", model_call_inputs[0])
        self.assertEqual(manifest["provider"], "external_jsonl")
        self.assertFalse(manifest["live_api_enabled"])
        self.assertFalse(manifest["api_key_recorded"])
        self.assertEqual(len(manifest["model_call_inputs_hash"]), 64)
        self.assertEqual(len(manifest["external_model_outputs_hash"]), 64)

    def test_s2_memory_unit_packet_preserves_original_text_for_quote_guardrail(self):
        from tools.proposals.proposal_runner import short_hash

        processed_text = "The author says he recently lived in the garret and wore the same clothes."
        original_text = "This morning (we living lately in the garret,) I rose, put on my suit with great skirts, having not lately worn any other clothes but them."
        route_row = memory_route_decision(
            "memory:paired-text",
            "route:memory-original",
            "weak_llm_proposal",
            processed_text,
            original_text,
        )
        route_row["warnings"] = ["source_text_quote_not_in_primary_text", "source_text_quote_not_exact"]
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_row],
        )
        proposal_input_id = f"s2pi:{short_hash('s2-portrait-proposal:test:memory:paired-text:route:memory-original')}"
        external_path = self.workspace / "external_model_outputs_memory_original.jsonl"
        write_jsonl(
            external_path,
            [
                {
                    "proposal_input_id": proposal_input_id,
                    "provider": "subagent",
                    "model_id": "gpt-4o",
                    "output_text": json.dumps(
                        {
                            "output_kind": "portrait_fact_candidate",
                            "fact_candidate_text": "The target subject had recently been living in the garret.",
                            "hypothesis_text": "",
                            "source_text_quote": "we living lately in the garret",
                            "source_text_quotes": ["we living lately in the garret"],
                            "supporting_observations": ["The original source text states the living arrangement."],
                            "alternative_explanations": [],
                            "uncertainty_notes": [],
                            "candidate_type": "user_state",
                            "inference_level": "explicit",
                            "claim_strength": "direct",
                            "proposal_confidence": "high",
                            "hypothesis_status": "unknown",
                            "hypothesis_confidence": "unknown",
                            "hypothesis_scope": "unknown",
                            "commitment_level": "unknown",
                            "promotion_readiness": "needs_more_evidence",
                            "subject_contamination_risk": "low",
                            "privacy_class": "public_dataset",
                            "warnings": [],
                        }
                    ),
                }
            ],
        )

        result = self.run_runner(
            provider="external_jsonl",
            output_name="memory_original_quote",
            profile="configs/proposals/s2_portrait_proposal.v0.2.json",
            external_model_outputs=str(external_path),
        )
        proposals = [
            json.loads(line)
            for line in Path(result["outputs"]["proposals"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        model_call_inputs = [
            json.loads(line)
            for line in Path(result["outputs"]["model_call_inputs"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["output_kind"], "portrait_fact_candidate")
        self.assertEqual(proposals[0]["source_text"], processed_text)
        self.assertEqual(proposals[0]["original_text"], original_text)
        self.assertEqual(proposals[0]["processed_text"], processed_text)
        self.assertNotIn("source_text_quote_not_in_primary_text", proposals[0]["warnings"])
        self.assertNotIn("source_text_quote_not_exact", proposals[0]["warnings"])
        self.assertIn("s1_inherited_source_text_quote_not_in_processed_text", proposals[0]["warnings"])
        self.assertIn("s1_inherited_source_text_quote_not_exact", proposals[0]["warnings"])
        self.assertEqual(model_call_inputs[0]["input_packet"]["original_text"], original_text)
        self.assertEqual(model_call_inputs[0]["input_packet"]["processed_text"], processed_text)
        self.assertIn(
            "s1_inherited_source_text_quote_not_in_processed_text",
            model_call_inputs[0]["input_packet"]["warnings"],
        )

    def test_external_jsonl_missing_output_is_logged_as_failure(self):
        rows = [
            text_unit("tu:p1", "Conversation context.", None),
            text_unit("tu:external-missing", "I prefer careful project notes."),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:external-missing", "route:external-missing", "weak_llm_proposal", rows[1]["text"])],
        )
        external_path = self.workspace / "external_model_outputs_missing.jsonl"
        write_jsonl(external_path, [])

        result = self.run_runner(
            provider="external_jsonl",
            output_name="external_jsonl_missing",
            profile="configs/proposals/s2_portrait_proposal.v0.2.json",
            external_model_outputs=str(external_path),
        )
        failures = [
            json.loads(line)
            for line in Path(result["outputs"]["model_output_failures"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["output_kind"], "model_failure")
        self.assertIn("provider_error_redacted", failures[0]["warnings"])
        self.assertFalse(failures[0]["write_permission"])


if __name__ == "__main__":
    unittest.main()
