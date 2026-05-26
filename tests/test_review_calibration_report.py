import json
import shutil
import unittest
from pathlib import Path

from tools.proposals.review_calibration_report import SamplingPolicy, build_review_calibration


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def proposal_row(
    proposal_id: str,
    output_kind: str,
    *,
    candidate_type: str = "none",
    quote: str = "",
    fact_text: str = "",
    hypothesis_text: str = "",
    observations: list[str] | None = None,
    evidence_refs: list[str] | None = None,
    raw_refs: list[dict] | None = None,
    warnings: list[str] | None = None,
    contamination: str = "unknown",
    source_text: str = "",
    source_text_preview: str = "",
) -> dict:
    return {
        "proposal_id": proposal_id,
        "proposal_run_id": "run:test",
        "proposal_profile_id": "s2_portrait_proposal.v0.2",
        "output_kind": output_kind,
        "proposal_status": output_kind,
        "epistemic_status": {
            "portrait_fact_candidate": "fact_candidate",
            "portrait_hypothesis_candidate": "hypothesis",
            "reject": "reject",
            "model_uncertain": "uncertain",
            "needs_human_review": "human_review_required",
            "skipped": "skipped",
            "model_failure": "failure",
        }.get(output_kind, "unknown"),
        "candidate_type": candidate_type,
        "source_text": source_text,
        "source_text_quote": quote,
        "source_text_preview": source_text_preview,
        "source_text_quotes": [quote] if quote else [],
        "fact_candidate_text": fact_text,
        "hypothesis_text": hypothesis_text,
        "supporting_observations": observations or [],
        "evidence_refs": evidence_refs or [],
        "raw_backpointer_refs": raw_refs or [],
        "warnings": warnings or [],
        "subject_contamination_risk": contamination,
        "write_permission": False,
    }


class ReviewCalibrationReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proposal_dir = ROOT / "users" / "_review_calibration_report_test"
        if self.proposal_dir.exists():
            shutil.rmtree(self.proposal_dir)
        self.proposal_dir.mkdir(parents=True)
        write_json(
            self.proposal_dir / "proposal_run_manifest.json",
            {
                "proposal_run_id": "run:test",
                "proposal_profile_id": "s2_portrait_proposal.v0.2",
                "workspace": "users/_review_calibration_report_test",
                "provider": "mock",
            },
        )

    def tearDown(self) -> None:
        if self.proposal_dir.exists():
            shutil.rmtree(self.proposal_dir)

    def run_report(self, rows: list[dict], failures: list[dict] | None = None) -> dict:
        write_jsonl(self.proposal_dir / "proposal_outcomes.ai.jsonl", rows)
        write_jsonl(self.proposal_dir / "model_output_failures.jsonl", failures or [])
        return build_review_calibration(
            proposal_dir=self.proposal_dir,
            write_items=True,
            policy=SamplingPolicy(sampling_seed="test-seed", sample_rate=0.5, min_sample_count=2),
        )

    def read_items(self) -> list[dict]:
        return [
            json.loads(line)
            for line in (self.proposal_dir / "review_calibration_items.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_classifies_fact_hypothesis_uncertain_contamination_and_failures(self):
        rows = [
            proposal_row(
                "p:fact",
                "portrait_fact_candidate",
                candidate_type="project_context",
                quote="I am building the project.",
                fact_text="The subject is building a project.",
                evidence_refs=["e:1"],
                raw_refs=[{"locator": "x"}],
            ),
            proposal_row(
                "p:hyp",
                "portrait_hypothesis_candidate",
                candidate_type="relationship_context",
                quote="Keep it up.",
                hypothesis_text="The subject takes an encouraging stance in this exchange.",
                evidence_refs=["e:2"],
                raw_refs=[{"locator": "y"}],
            ),
            proposal_row("p:uncertain", "model_uncertain"),
            proposal_row(
                "p:contaminated",
                "portrait_fact_candidate",
                candidate_type="preference",
                quote="not primary",
                fact_text="Contaminated fact.",
                evidence_refs=["e:3"],
                raw_refs=[{"locator": "z"}],
                warnings=["source_text_quote_not_in_primary_text"],
            ),
            proposal_row("p:reject", "reject"),
            proposal_row("p:skip", "skipped"),
        ]
        failures = [proposal_row("p:failure", "model_failure", warnings=["invalid_json"])]

        result = self.run_report(rows, failures)
        items = self.read_items()
        pools = {item["proposal_id"]: item["review_pool"] for item in items}

        self.assertEqual(pools["p:fact"], "fact_promotion_review_queue")
        self.assertEqual(pools["p:hyp"], "hypothesis_pool_candidates")
        self.assertEqual(pools["p:uncertain"], "uncertain_pool")
        self.assertEqual(pools["p:contaminated"], "contamination_risk_pool")
        self.assertEqual(pools["p:failure"], "provider_failure_pool")
        self.assertEqual(result["review_pool_counts"]["sampled_reject_skip_pool"], 2)
        self.assertTrue(all(item["write_permission"] is False for item in items))

        hyp_item = next(item for item in items if item["proposal_id"] == "p:hyp")
        self.assertFalse(hyp_item["requires_manual_review_now"])
        self.assertEqual(hyp_item["default_review_action"], "keep_low_commitment")
        self.assertIn("source_text", hyp_item)
        self.assertEqual(hyp_item["hypothesis_text"], "The subject takes an encouraging stance in this exchange.")
        self.assertEqual(hyp_item["source_text_quote"], "Keep it up.")
        self.assertEqual(hyp_item["sampling_policy_id"], "review_calibration_sampling.v0.1")
        self.assertEqual(hyp_item["sampling_seed"], "test-seed")
        self.assertEqual(hyp_item["sample_rate"], 0.5)

    def test_hypothesis_without_usable_text_or_support_goes_to_uncertain_pool(self):
        rows = [
            proposal_row(
                "p:bad-hyp",
                "portrait_hypothesis_candidate",
                candidate_type="relationship_context",
                hypothesis_text="",
                observations=[],
                evidence_refs=[],
            )
        ]

        self.run_report(rows)
        items = self.read_items()

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["review_pool"], "uncertain_pool")
        self.assertEqual(items[0]["default_review_action"], "needs_more_evidence")

    def test_hypothesis_with_only_evidence_ref_is_not_enough_support(self):
        rows = [
            proposal_row(
                "p:weak-hyp",
                "portrait_hypothesis_candidate",
                candidate_type="relationship_context",
                hypothesis_text="The subject may be encouraging in this exchange.",
                evidence_refs=["e:weak"],
            )
        ]

        self.run_report(rows)
        items = self.read_items()

        self.assertEqual(items[0]["review_pool"], "uncertain_pool")
        self.assertEqual(items[0]["review_priority"], "sample_only")

    def test_needs_human_review_requires_manual_review_now(self):
        rows = [proposal_row("p:human", "needs_human_review")]

        result = self.run_report(rows)
        items = self.read_items()

        self.assertEqual(items[0]["review_pool"], "uncertain_pool")
        self.assertEqual(items[0]["review_priority"], "high")
        self.assertTrue(items[0]["requires_manual_review_now"])
        self.assertEqual(result["review_burden_counts"]["requires_manual_review_now"], 1)

    def test_unknown_output_kind_is_not_silently_dropped(self):
        rows = [proposal_row("p:future", "future_new_kind")]

        result = self.run_report(rows)
        items = self.read_items()

        self.assertEqual(items[0]["review_pool"], "schema_unknown_pool")
        self.assertEqual(items[0]["review_priority"], "high")
        self.assertTrue(items[0]["requires_manual_review_now"])
        self.assertEqual(result["output_kind_counts"]["future_new_kind"], 1)

    def test_attribution_blocks_other_node_fact_from_fact_promotion_queue(self):
        rows = [
            proposal_row(
                "p:other-node",
                "portrait_fact_candidate",
                candidate_type="relationship_context",
                quote="Xiao Hong, you are really smart.",
                fact_text="Xiao Hong is really smart.",
                evidence_refs=["e:other"],
                raw_refs=[{"locator": "other"}],
            )
        ]

        result = self.run_report(rows)
        items = self.read_items()

        self.assertEqual(items[0]["review_pool"], "hypothesis_pool_candidates")
        self.assertEqual(items[0]["claim_type_hint"], "subject_belief_about_node")
        self.assertEqual(items[0]["fact_promotion_eligibility"], "ineligible")
        self.assertNotIn("fact_promotion_review_queue", result["review_pool_counts"])

    def test_attribution_preserves_relation_candidate_outside_fact_promotion(self):
        rows = [
            proposal_row(
                "p:relation",
                "portrait_fact_candidate",
                candidate_type="project_context",
                quote="I use tool X for this project.",
                fact_text="The subject uses tool X for this project.",
                evidence_refs=["e:tool"],
                raw_refs=[{"locator": "tool"}],
            )
        ]

        self.run_report(rows)
        items = self.read_items()

        self.assertEqual(items[0]["review_pool"], "future_relation_candidate")
        self.assertEqual(items[0]["claim_type_hint"], "subject_relation_to_node")
        self.assertEqual(items[0]["claim_target_node_type"], "tool")

    def test_report_is_written_and_preserves_no_write_boundaries(self):
        rows = [proposal_row("p:reject", "reject"), proposal_row("p:skip", "skipped")]

        result = self.run_report(rows)
        report_text = (self.proposal_dir / "review_calibration_report.md").read_text(encoding="utf-8")

        self.assertIn("No live API calls are made", report_text)
        self.assertIn("No durable memory writes are made", report_text)
        self.assertIn("sampled_reject_skip_pool", report_text)
        self.assertEqual(result["review_burden_counts"]["requires_manual_review_now"], 0)

    def test_sampled_skipped_items_preserve_source_text_preview(self):
        rows = [
            proposal_row(
                "p:skip-preview",
                "skipped",
                source_text_preview="Gina says contemporary dance really speaks to her.",
            )
        ]

        self.run_report(rows)
        items = self.read_items()
        report_text = (self.proposal_dir / "review_calibration_report.md").read_text(encoding="utf-8")

        self.assertEqual(items[0]["source_text_preview"], "Gina says contemporary dance really speaks to her.")
        self.assertIn("contemporary dance", report_text)

    def test_review_items_preserve_source_and_candidate_text_fields(self):
        rows = [
            proposal_row(
                "p:text-fields",
                "portrait_hypothesis_candidate",
                candidate_type="relationship_context",
                quote="Keep it up.",
                hypothesis_text="The subject encourages the other participant in this exchange.",
                source_text="Gina said: Keep it up.",
                source_text_preview="Gina said: Keep it up.",
                raw_refs=[{"locator": "source"}],
            )
        ]

        self.run_report(rows)
        items = self.read_items()

        self.assertEqual(items[0]["source_text"], "Gina said: Keep it up.")
        self.assertEqual(items[0]["source_text_preview"], "Gina said: Keep it up.")
        self.assertEqual(items[0]["source_text_quote"], "Keep it up.")
        self.assertEqual(items[0]["hypothesis_text"], "The subject encourages the other participant in this exchange.")
        self.assertEqual(items[0]["fact_candidate_text"], "")
        self.assertIn("candidate_text", items[0])

    def test_fact_promotion_queue_requires_primary_quote(self):
        rows = [
            proposal_row(
                "p:bad-primary-quote",
                "portrait_fact_candidate",
                candidate_type="preference",
                quote="I like dance.",
                fact_text="The subject likes dance.",
                evidence_refs=["e:bad"],
                raw_refs=[{"locator": "bad"}],
                warnings=["source_text_quote_not_in_primary_text"],
            )
        ]

        result = self.run_report(rows)
        items = self.read_items()

        self.assertEqual(items[0]["review_pool"], "contamination_risk_pool")
        self.assertNotIn("fact_promotion_review_queue", result["review_pool_counts"])

    def test_skip_audit_signals_identify_target_modeling_value(self):
        rows = [
            proposal_row(
                "p:target-skip",
                "skipped",
                source_text_preview="I prefer quiet focused work and enjoy careful planning.",
            )
        ]

        result = self.run_report(rows)
        items = self.read_items()
        signal = items[0]["skip_audit_signals"]

        self.assertEqual(signal["modeling_value_hint"], "high")
        self.assertEqual(signal["evidence_directness_hint"], "direct_self_report")
        self.assertEqual(signal["claim_target_clarity_hint"], "target_subject")
        self.assertEqual(signal["review_priority_hint"], "focused_sample")
        self.assertIn("self_modeling_signal", signal["signal_tags"])
        self.assertEqual(result["skip_audit_signal_counts"]["modeling_value_hint"]["high"], 1)

    def test_skip_audit_signals_preserve_non_target_value_and_risk(self):
        rows = [
            proposal_row(
                "p:other-skip",
                "skipped",
                source_text_preview="Hey Gina! Your store looks great; all your hard work paid off.",
                warnings=["non_target_subject_skipped_for_s2_portrait"],
            )
        ]

        self.run_report(rows)
        items = self.read_items()
        signal = items[0]["skip_audit_signals"]

        self.assertEqual(signal["modeling_value_hint"], "medium")
        self.assertEqual(signal["attribution_risk_hint"], "high")
        self.assertEqual(signal["promotion_risk_hint"], "high")
        self.assertEqual(signal["review_priority_hint"], "focused_sample")
        self.assertIn("non_target_source", signal["signal_tags"])
        self.assertIn("relation_or_node_signal", signal["signal_tags"])

    def test_skip_audit_signals_allow_low_value_no_review_priority(self):
        rows = [
            proposal_row(
                "p:low-skip",
                "skipped",
                source_text_preview="Hey Jon! Good to see you. What's up?",
            )
        ]

        self.run_report(rows)
        items = self.read_items()
        signal = items[0]["skip_audit_signals"]

        self.assertEqual(signal["modeling_value_hint"], "none")
        self.assertEqual(signal["review_priority_hint"], "none")
        self.assertIn("low_content_signal", signal["signal_tags"])


if __name__ == "__main__":
    unittest.main()
