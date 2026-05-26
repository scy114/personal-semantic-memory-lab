import json
import shutil
import unittest
from argparse import Namespace
from pathlib import Path

from tools.proposals.proposal_runner import run_proposal_runner, short_hash


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "configs/proposals/s1_memory_candidate_proposal.v0.1.json"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def text_unit(text_unit_id: str, text: str, parent: str | None = "tu:p1", **overrides) -> dict:
    row = {
        "schema_version": "s0b.text_unit.v0.1",
        "text_unit_id": text_unit_id,
        "unit_type": "sentence" if parent else "paragraph",
        "text": text,
        "parent_text_unit_id": parent,
        "previous_text_unit_id": None,
        "next_text_unit_id": None,
        "raw_span_id": f"raw:test:{text_unit_id}",
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


def route_decision(text_unit_id: str, route_id: str, route: str) -> dict:
    return {
        "schema_version": "memory_proposal_router_pilot.v0.1",
        "route_run_id": "memory-proposal-router:test:s1-memory",
        "span_id": text_unit_id,
        "text_unit_id": text_unit_id,
        "raw_span_id": f"raw:test:{text_unit_id}",
        "evidence_ref": None,
        "source_layer": "s0b_text_unit",
        "section_type": "body",
        "perspective": "author",
        "retrieval_policy": "default_retrieval",
        "s2_policy": "candidate_allowed",
        "task_routes": [
            {
                "route_decision_id": route_id,
                "target_task": "s1_memory_candidate",
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


class S1MemoryCandidateProposalRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_s1_memory_candidate_proposal_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        (self.workspace / "raw").mkdir(parents=True)
        (self.workspace / "raw" / "test.txt").write_text("test source", encoding="utf-8")
        rows = [
            text_unit("tu:p1", "Conversation context.", None),
            text_unit("tu:event", "I lost my job yesterday and plan to start my own studio.", evidence_ref="evidence:test:event"),
            text_unit("tu:proc", "I usually write a short checklist before starting a workflow."),
            text_unit("tu:greeting", "Thanks!"),
            text_unit("tu:ambiguous", "This ambiguous note is uncertain about who made the decision."),
            text_unit("tu:review", "This human review case has unclear attribution and privacy risk."),
            text_unit("tu:multi", "I love detailed notes, and I plan to build a better archive this month."),
        ]
        write_jsonl(self.workspace / "raw" / "organization" / "text_units.jsonl", rows)
        decisions = [
            route_decision("tu:event", "route:event", "weak_llm_proposal"),
            route_decision("tu:proc", "route:proc", "weak_llm_proposal"),
            route_decision("tu:greeting", "route:greeting", "script_only"),
            route_decision("tu:ambiguous", "route:ambiguous", "weak_llm_proposal"),
            route_decision("tu:review", "route:review", "weak_llm_proposal"),
            route_decision("tu:multi", "route:multi", "strong_llm_proposal"),
        ]
        write_jsonl(self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl", decisions)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def run_runner(
        self,
        provider: str = "mock",
        output_name: str = "s1_memory_candidate",
        external_model_outputs: str | None = None,
        provider_concurrency: int = 1,
    ) -> dict:
        return run_proposal_runner(
            Namespace(
                project_root=str(ROOT),
                workspace=str(self.workspace),
                profile=PROFILE,
                route_decisions=None,
                output_dir=str(self.workspace / "proposals" / output_name),
                run_id="s1-memory-candidate-proposal:test",
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
                item_offset=0,
                sample_stride=1,
                external_model_outputs=external_model_outputs,
                provider_concurrency=provider_concurrency,
            )
        )

    def read_outputs(self, result: dict) -> tuple[list[dict], list[dict], dict]:
        output_dir = Path(result["output_dir"])
        proposals = [
            json.loads(line)
            for line in (output_dir / "proposal_outcomes.ai.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        failures = [
            json.loads(line)
            for line in (output_dir / "model_output_failures.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        manifest = json.loads((output_dir / "proposal_run_manifest.json").read_text(encoding="utf-8"))
        return proposals, failures, manifest

    def test_mock_s1_outputs_paired_candidates_and_no_canonical_writes(self):
        result = self.run_runner()
        proposals, failures, manifest = self.read_outputs(result)
        by_text_unit = {row["text_unit_id"]: row for row in proposals}

        self.assertFalse(failures)
        self.assertEqual(manifest["proposal_profile_id"], "s1_memory_candidate_proposal.v0.1")
        self.assertEqual(manifest["target_task"], "s1_memory_candidate")
        self.assertFalse(manifest["durable_writes_executed"])
        self.assertFalse(manifest["reviewed_units_written"])
        self.assertFalse(manifest["graph_truth_written"])
        self.assertIn("memory_candidates_hash", manifest)
        self.assertIn("preprocessing_decisions_hash", manifest)

        event = by_text_unit["tu:event"]
        self.assertEqual(event["output_kind"], "memory_candidate")
        self.assertEqual(event["memory_class"], "episodic")
        self.assertTrue(event["memory_candidate_text"])
        self.assertEqual(event["processed_text"], event["memory_candidate_text"])
        self.assertTrue(event["original_text"])
        self.assertEqual(event["source_text_quote"], "I lost my job yesterday and plan to start my own studio.")
        self.assertEqual(event["evidence_refs"], ["evidence:test:event"])
        self.assertTrue(event["llm_assist_used"])
        self.assertIn("memory_candidate_text", event["llm_assisted_s1_candidate"])
        self.assertFalse(event["write_permission"])

        procedural = by_text_unit["tu:proc"]
        self.assertEqual(procedural["output_kind"], "memory_candidate")
        self.assertEqual(procedural["memory_class"], "procedural")

        greeting = by_text_unit["tu:greeting"]
        self.assertEqual(greeting["proposal_status"], "skipped")
        self.assertEqual(greeting["original_text"], "Thanks!")
        self.assertEqual(greeting["memory_candidate_text"], "")
        self.assertFalse(greeting["llm_assist_used"])

        ambiguous = by_text_unit["tu:ambiguous"]
        self.assertEqual(ambiguous["output_kind"], "model_uncertain")
        self.assertEqual(ambiguous["memory_candidate_text"], "")
        self.assertEqual(ambiguous["memory_class"], "unknown")

        review = by_text_unit["tu:review"]
        self.assertEqual(review["output_kind"], "needs_human_review")
        self.assertEqual(review["memory_candidate_text"], "")

        multi = by_text_unit["tu:multi"]
        self.assertEqual(multi["output_kind"], "memory_candidate")
        self.assertTrue(multi["memory_candidate_text"])
        self.assertTrue(multi["source_text_quote"])

        self.assertFalse((self.workspace / "memory" / "memory_units.jsonl").exists())
        self.assertFalse((self.workspace / "portrait" / "reviewed_units.jsonl").exists())
        self.assertFalse((self.workspace / "portrait" / "current_portrait.json").exists())
        self.assertFalse((self.workspace / "graph" / "graph_edges.jsonl").exists())

    def test_mock_s1_provider_concurrency_keeps_ordered_outputs(self):
        result = self.run_runner(output_name="s1_memory_candidate_parallel", provider_concurrency=3)
        proposals, failures, manifest = self.read_outputs(result)

        self.assertFalse(failures)
        self.assertEqual(manifest["provider_concurrency"], 3)
        self.assertEqual([row["text_unit_id"] for row in proposals], ["tu:event", "tu:proc", "tu:greeting", "tu:ambiguous", "tu:review", "tu:multi"])

    def test_existing_deterministic_memory_candidate_is_carried_as_auxiliary_processing(self):
        write_jsonl(
            self.workspace / "memory" / "memory_candidates.jsonl",
            [
                {
                    "schema_version": "step1.memory_candidate.v0.1",
                    "candidate_id": "memcand:test:event",
                    "candidate_text": "Existing deterministic candidate.",
                    "candidate_type": "source_claim",
                    "memory_class": "episodic",
                    "evidence_quote": "I lost my job yesterday and plan to start my own studio.",
                    "evidence_refs": ["evidence:test:event"],
                    "backpointer_refs": ["evidence:test:event"],
                    "generation_method": "local_skill",
                    "inference_level": "explicit",
                    "confidence": "high",
                    "review_status": "accepted_for_experiment",
                }
            ],
        )
        result = self.run_runner(output_name="deterministic_processing")
        proposals, _, manifest = self.read_outputs(result)
        event = next(row for row in proposals if row["text_unit_id"] == "tu:event")

        self.assertEqual(event["deterministic_processing_status"], "succeeded")
        self.assertEqual(event["deterministic_s1_processing"]["memory_candidate"]["candidate_id"], "memcand:test:event")
        self.assertEqual(len(manifest["memory_candidates_hash"]), 64)

    def test_s1_script_only_with_deterministic_candidate_emits_proposal_without_llm(self):
        write_jsonl(
            self.workspace / "raw" / "organization" / "text_units.jsonl",
            [
                text_unit("tu:p1", "Conversation context.", None),
                text_unit("tu:script-det", "I prefer careful project notes.", evidence_ref="evidence:test:script-det"),
            ],
        )
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:script-det", "route:script-det", "script_only")],
        )
        write_jsonl(
            self.workspace / "memory" / "memory_candidates.jsonl",
            [
                {
                    "schema_version": "step1.memory_candidate.v0.1",
                    "candidate_id": "memcand:test:script-det",
                    "candidate_text": "The source says the subject prefers careful project notes.",
                    "candidate_type": "source_claim",
                    "memory_class": "semantic",
                    "evidence_quote": "I prefer careful project notes.",
                    "evidence_refs": ["evidence:test:script-det"],
                    "backpointer_refs": ["evidence:test:script-det"],
                    "generation_method": "local_skill",
                    "inference_level": "explicit",
                    "confidence": "high",
                    "review_status": "accepted_for_experiment",
                }
            ],
        )

        result = self.run_runner(output_name="script_only_deterministic")
        proposals, failures, manifest = self.read_outputs(result)

        self.assertFalse(failures)
        self.assertEqual(len(proposals), 1)
        row = proposals[0]
        self.assertEqual(row["route_recommended"], "script_only")
        self.assertEqual(row["route_used"], "script_only")
        self.assertEqual(row["output_kind"], "memory_candidate")
        self.assertEqual(row["memory_candidate_text"], "The source says the subject prefers careful project notes.")
        self.assertEqual(row["processed_text"], "The source says the subject prefers careful project notes.")
        self.assertEqual(row["memory_class"], "semantic")
        self.assertEqual(row["deterministic_processing_status"], "succeeded")
        self.assertFalse(row["llm_assist_used"])
        self.assertFalse(row["needs_llm_assist"])
        self.assertEqual(row["provider"], "none")
        self.assertEqual(row["model_id"], "")
        self.assertIn("script_only_deterministic_s1_candidate_proposal", row["warnings"])
        self.assertEqual(manifest["counts"]["model_call_rows"], 0)

    def test_s1_script_only_span_level_deterministic_candidate_is_context_not_proposal(self):
        write_jsonl(
            self.workspace / "raw" / "organization" / "text_units.jsonl",
            [
                text_unit("tu:span-p1", "Parent span text.", None, raw_span_id="raw:test:text_span:0001"),
                text_unit(
                    "tu:span-s1",
                    "I prefer careful project notes.",
                    "tu:span-p1",
                    raw_span_id="raw:test:text_span:0001",
                ),
            ],
        )
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:span-s1", "route:span-s1", "script_only")],
        )
        write_jsonl(
            self.workspace / "memory" / "memory_candidates.jsonl",
            [
                {
                    "schema_version": "step1.memory_candidate.v0.1",
                    "candidate_id": "memcand:test:span-0001",
                    "candidate_text": "A coarse span-level deterministic candidate.",
                    "candidate_type": "source_claim",
                    "memory_class": "episodic",
                    "evidence_quote": "Parent span text. I prefer careful project notes.",
                    "evidence_refs": ["evidence:test:span-0001"],
                    "source_specific_refs": ["text_span:0001"],
                    "generation_method": "script_generated",
                    "inference_level": "explicit",
                    "confidence": "high",
                    "review_status": "accepted_for_experiment",
                }
            ],
        )

        result = self.run_runner(output_name="script_only_context_deterministic")
        proposals, failures, manifest = self.read_outputs(result)

        self.assertFalse(failures)
        self.assertEqual(len(proposals), 1)
        row = proposals[0]
        self.assertEqual(row["output_kind"], "skipped")
        self.assertEqual(row["deterministic_processing_status"], "insufficient")
        self.assertIn("context_memory_candidate", row["deterministic_s1_processing"])
        self.assertEqual(row["memory_candidate_text"], "")
        self.assertFalse(row["llm_assist_used"])
        self.assertIn("script_only_deterministic_s1_candidate_not_exact", row["warnings"])
        self.assertEqual(manifest["counts"]["model_call_rows"], 0)

    def test_external_jsonl_replays_s1_payload_through_validation(self):
        write_jsonl(
            self.workspace / "raw" / "organization" / "text_units.jsonl",
            [
                text_unit("tu:p1", "Conversation context.", None),
                text_unit("tu:external", "I prefer careful project notes."),
            ],
        )
        write_jsonl(
            self.workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl",
            [route_decision("tu:external", "route:external", "weak_llm_proposal")],
        )
        proposal_input_id = f"s1mpi:{short_hash('s1-memory-candidate-proposal:test:tu:external:route:external')}"
        external_path = self.workspace / "external_model_outputs.jsonl"
        write_jsonl(
            external_path,
            [
                {
                    "proposal_input_id": proposal_input_id,
                    "provider": "subagent",
                    "model_id": "gpt-4o",
                    "output_text": {
                        "output_kind": "memory_candidate",
                        "memory_candidate_text": "The source says the subject prefers careful project notes.",
                        "memory_class": "semantic",
                        "source_text_quote": "I prefer careful project notes.",
                        "source_text_quotes": ["I prefer careful project notes."],
                        "supporting_observations": ["The source explicitly states a preference."],
                        "scope_hint": "document",
                        "temporal_hint": "unknown",
                        "compression_level": "light",
                        "inference_level": "explicit",
                        "proposal_confidence": "high",
                        "uncertainty_notes": [],
                        "warnings": [],
                    },
                }
            ],
        )

        result = self.run_runner(
            provider="external_jsonl",
            output_name="external_jsonl_s1",
            external_model_outputs=str(external_path),
        )
        proposals, failures, manifest = self.read_outputs(result)

        self.assertFalse(failures)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["provider"], "subagent")
        self.assertEqual(proposals[0]["model_id"], "gpt-4o")
        self.assertEqual(proposals[0]["output_kind"], "memory_candidate")
        self.assertEqual(proposals[0]["memory_candidate_text"], "The source says the subject prefers careful project notes.")
        self.assertEqual(proposals[0]["processed_text"], "The source says the subject prefers careful project notes.")
        self.assertEqual(proposals[0]["original_text"], "I prefer careful project notes.")
        self.assertEqual(manifest["provider"], "external_jsonl")
        self.assertFalse(manifest["api_key_recorded"])

    def test_item_offset_and_sample_stride_select_later_packets(self):
        result = run_proposal_runner(
            Namespace(
                project_root=str(ROOT),
                workspace=str(self.workspace),
                profile=PROFILE,
                route_decisions=None,
                output_dir=str(self.workspace / "proposals" / "offset_stride"),
                run_id="s1-memory-candidate-proposal:test",
                duplicate_policy="fail",
                provider="mock",
                api_mode="responses",
                allow_live_api=False,
                weak_model="mock-weak",
                strong_model="mock-strong",
                weak_prompt=None,
                strong_prompt=None,
                env_file=None,
                max_items=2,
                item_offset=1,
                sample_stride=2,
                external_model_outputs=None,
            )
        )
        proposals, failures, manifest = self.read_outputs(result)

        self.assertFalse(failures)
        self.assertEqual([row["text_unit_id"] for row in proposals], ["tu:proc", "tu:ambiguous"])
        self.assertEqual(manifest["item_selection"]["item_offset"], 1)
        self.assertEqual(manifest["item_selection"]["sample_stride"], 2)
        self.assertEqual(manifest["item_selection"]["max_items"], 2)


if __name__ == "__main__":
    unittest.main()
