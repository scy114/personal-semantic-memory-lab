import json
import shutil
import unittest
from pathlib import Path

from tools.proposals.proposal_pool_collector import collect_proposal_pools


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def review_item(proposal_id: str, pool: str, **overrides) -> dict:
    row = {
        "schema_version": "review_calibration_item.v0.2",
        "review_item_id": f"rci:{proposal_id}",
        "proposal_id": proposal_id,
        "proposal_run_id": "run:test",
        "proposal_profile_id": "s2_portrait_proposal.v0.2",
        "output_kind": "portrait_hypothesis_candidate",
        "epistemic_status": "hypothesis",
        "review_pool": pool,
        "review_priority": "low",
        "default_review_action": "keep_low_commitment",
        "calibration_action": "none",
        "requires_manual_review_now": False,
        "write_permission": False,
        "source_text": "I enjoy contemporary dance.",
        "source_text_preview": "I enjoy contemporary dance.",
        "source_text_quote": "I enjoy contemporary dance.",
        "fact_candidate_text": "",
        "hypothesis_text": "The subject may have a current interest in contemporary dance.",
        "candidate_text": "",
        "evidence_refs": ["e:1"],
        "raw_backpointer_refs": [{"locator": "x"}],
        "warnings": [],
        "claim_type_hint": "subject_intrinsic_fact",
        "claim_target_hint": "target_subject",
        "source_perspective": "target",
        "attribution_status_hint": "clear",
        "promotion_path": "hypothesis_pool_candidates",
        "fact_promotion_eligibility": "ineligible",
        "claim_attribution_annotation": {
            "claim_type_hint": "subject_intrinsic_fact",
            "fact_promotion_eligibility": "ineligible",
        },
        "skip_audit_signals": {},
    }
    row.update(overrides)
    return row


class ProposalPoolCollectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.proposal_dir = ROOT / "users" / "_proposal_pool_collector_test"
        if self.proposal_dir.exists():
            shutil.rmtree(self.proposal_dir)
        self.proposal_dir.mkdir(parents=True)
        write_json(
            self.proposal_dir / "proposal_run_manifest.json",
            {
                "proposal_run_id": "run:test",
                "proposal_profile_id": "s2_portrait_proposal.v0.2",
            },
        )

    def tearDown(self) -> None:
        if self.proposal_dir.exists():
            shutil.rmtree(self.proposal_dir)

    def read_jsonl(self, name: str) -> list[dict]:
        path = self.proposal_dir / name
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def test_collects_non_durable_fact_hypothesis_and_calibration_pools(self):
        write_jsonl(
            self.proposal_dir / "review_calibration_items.jsonl",
            [
                review_item(
                    "p:fact",
                    "fact_promotion_review_queue",
                    output_kind="portrait_fact_candidate",
                    epistemic_status="fact_candidate",
                    review_priority="high",
                    default_review_action="accept_for_promotion_review",
                    requires_manual_review_now=True,
                    fact_candidate_text="The subject enjoys contemporary dance.",
                    hypothesis_text="",
                    promotion_path="fact_promotion_review_queue",
                    fact_promotion_eligibility="eligible",
                ),
                review_item("p:hyp", "hypothesis_pool_candidates"),
                review_item(
                    "p:skip",
                    "sampled_reject_skip_pool",
                    output_kind="skipped",
                    epistemic_status="skipped",
                    default_review_action="sample_only",
                    hypothesis_text="",
                ),
            ],
        )

        result = collect_proposal_pools(proposal_dir=self.proposal_dir)
        facts = self.read_jsonl("fact_promotion_candidates.review.jsonl")
        hypotheses = self.read_jsonl("active_hypothesis_candidates.low_commitment.jsonl")
        calibration = self.read_jsonl("calibration_review_seed_items.jsonl")

        self.assertEqual(result["fact_pool_rows"], 1)
        self.assertEqual(result["hypothesis_pool_rows"], 1)
        self.assertEqual(result["calibration_seed_rows"], 1)
        self.assertFalse(result["durable_memory_written"])
        self.assertFalse(result["reviewed_portrait_units_written"])
        self.assertFalse(result["graph_truth_written"])
        self.assertFalse(result["automatic_acceptance_executed"])

        self.assertEqual(facts[0]["promotion_review_status"], "needs_strict_promotion_review")
        self.assertFalse(facts[0]["promotion_allowed_without_review"])
        self.assertTrue(facts[0]["requires_manual_review_now"])
        self.assertFalse(facts[0]["write_permission"])

        self.assertEqual(hypotheses[0]["hypothesis_status"], "active_unreviewed")
        self.assertEqual(hypotheses[0]["commitment_status"], "low_commitment")
        self.assertFalse(hypotheses[0]["requires_manual_review_now"])
        self.assertFalse(hypotheses[0]["write_permission"])

        self.assertEqual(calibration[0]["review_pool"], "sampled_reject_skip_pool")
        self.assertEqual(calibration[0]["calibration_action"], "none")
        self.assertFalse(calibration[0]["write_permission"])

    def test_can_build_review_items_if_missing(self):
        write_jsonl(
            self.proposal_dir / "proposal_outcomes.ai.jsonl",
            [
                {
                    "proposal_id": "p:hyp",
                    "proposal_run_id": "run:test",
                    "proposal_profile_id": "s2_portrait_proposal.v0.2",
                    "output_kind": "portrait_hypothesis_candidate",
                    "epistemic_status": "hypothesis",
                    "source_text_quote": "I enjoy contemporary dance.",
                    "source_text": "I enjoy contemporary dance.",
                    "hypothesis_text": "The subject may have a current interest in contemporary dance.",
                    "evidence_refs": ["e:1"],
                    "raw_backpointer_refs": [{"locator": "x"}],
                    "warnings": [],
                    "write_permission": False,
                }
            ],
        )
        write_jsonl(self.proposal_dir / "model_output_failures.jsonl", [])

        result = collect_proposal_pools(proposal_dir=self.proposal_dir)

        self.assertEqual(result["hypothesis_pool_rows"], 1)
        self.assertTrue((self.proposal_dir / "review_calibration_items.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
