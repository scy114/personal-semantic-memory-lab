import json
import shutil
import unittest
from pathlib import Path

from tools.maintenance.latest_view import resolve_latest_view_input
from tools.maintenance.s2_current_view_publisher import run_s2_current_view_publish


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def base_unit(unit_id: str = "unit-old") -> dict:
    return {
        "schema_version": "s2.reviewed_portrait_unit.v1",
        "unit_id": unit_id,
        "user_id": "Jon",
        "type": "preference",
        "content": "Jon prefers contemporary dance.",
        "evidence_refs": ["evidence:old"],
        "status": "active",
    }


def candidate(review_item_id: str = "review-s2-1", *, source_s1: str = "memory-new") -> dict:
    return {
        "schema_version": "maintenance.s2_incremental_candidate.v0.4",
        "candidate_status": "candidate_ready_for_review",
        "candidate_text": "Jon currently prefers salsa for the performance.",
        "candidate_type": "preference",
        "proposal_confidence": "high",
        "inference_level": "explicit",
        "proposal_id": "s2p-new",
        "proposal_input_id": "s2pi-new",
        "proposal_run_id": "run-new",
        "provider": "openai",
        "model_id": "claude-haiku-4-5",
        "output_kind": "portrait_fact_candidate",
        "route_used": "strong_llm_proposal",
        "review_item_id": review_item_id,
        "patch_id": "patch-s2-new",
        "operation_id": "op-s2",
        "source_s1_memory_id": source_s1,
        "refresh_reason": "split_current_and_historical_overlay",
        "evidence_refs": ["evidence:new"],
        "raw_backpointer_refs": [{"locator": {"display_ref": "D99:2"}}],
        "source_text_quote": "For the performance now, salsa is my first choice.",
        "proposal_row": {
            "target_participant": "Jon",
            "subject_id": "Jon",
            "privacy_class": "personal_low",
            "candidate_text": "Jon currently prefers salsa for the performance.",
            "proposal_confidence": "high",
            "inference_level": "explicit",
        },
    }


def decision(review_item_id: str, *, status: str = "approved_for_apply") -> dict:
    return {
        "schema_version": "maintenance.incremental_review_decision.v0.4",
        "review_item_id": review_item_id,
        "review_status": status,
        "human_decision": "approve_recommended" if status == "approved_for_apply" else "reject",
        "internal_review_action": "accept_patch" if status == "approved_for_apply" else "reject_patch",
    }


class V04S2CurrentViewPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_v04_s2_current_view_publish_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def test_publishes_reviewed_s2_candidate_and_marks_direct_old_unit_stale(self):
        base_path = self.workspace / "base_reviewed_units.jsonl"
        candidate_path = self.workspace / "s2_candidates.jsonl"
        decisions_path = self.workspace / "s2_review_decisions.jsonl"
        affected_path = self.workspace / "s2_affected.jsonl"
        write_jsonl(base_path, [base_unit()])
        write_jsonl(candidate_path, [candidate()])
        write_jsonl(decisions_path, [decision("review-s2-1")])
        write_jsonl(
            affected_path,
            [
                {
                    "new_s1_unit_id": "memory-new",
                    "affected_s2_unit_id": "unit-old",
                    "s2_delta_signal": "contradicts_profile",
                    "subject_scope_existing_s2_unit_ids": ["unit-old"],
                }
            ],
        )

        manifest = run_s2_current_view_publish(
            workspace=self.workspace,
            base_reviewed_units_jsonl=base_path,
            candidate_outputs_jsonl=candidate_path,
            review_decisions_jsonl=decisions_path,
            affected_s2_jsonl=affected_path,
        )

        current = read_jsonl(self.workspace / "s2_current" / "s2_current.jsonl")
        excluded = read_jsonl(self.workspace / "s2_current" / "s2_current_excluded.jsonl")
        transitions = read_jsonl(self.workspace / "s2_current" / "s2_status_transitions.jsonl")

        self.assertEqual(len(current), 1)
        self.assertEqual(current[0]["content"], "Jon currently prefers salsa for the performance.")
        self.assertEqual(current[0]["current_view_status"], "accepted_by_s2_review")
        self.assertEqual(excluded[0]["unit_id"], "unit-old")
        self.assertEqual(excluded[0]["current_view_status"], "historical_or_stale")
        self.assertFalse(excluded[0]["participates_in_default_query"])
        self.assertTrue(any(row["unit_id"] == "unit-old" and row["current_view_status"] == "historical_or_stale" for row in transitions))
        self.assertEqual(manifest["counts"]["current_active_count"], 1)
        self.assertEqual(manifest["counts"]["current_excluded_count"], 1)
        self.assertTrue(manifest["query_default_ready"])
        self.assertFalse(manifest["canonical_reviewed_units_rewritten"])
        self.assertFalse(manifest["current_portrait_written"])
        self.assertFalse((self.workspace / "portrait" / "reviewed_units.jsonl").exists())

    def test_unreviewed_candidate_is_excluded_by_default(self):
        candidate_path = self.workspace / "s2_candidates.jsonl"
        write_jsonl(candidate_path, [candidate()])

        manifest = run_s2_current_view_publish(
            workspace=self.workspace,
            candidate_outputs_jsonl=candidate_path,
        )

        self.assertEqual(manifest["counts"]["current_active_count"], 0)
        self.assertEqual(manifest["counts"]["current_excluded_count"], 1)
        excluded = read_jsonl(self.workspace / "s2_current" / "s2_current_excluded.jsonl")
        self.assertEqual(excluded[0]["current_view_status"], "excluded_pending_s2_review")

    def test_explicit_unreviewed_experiment_can_publish_candidate(self):
        candidate_path = self.workspace / "s2_candidates.jsonl"
        write_jsonl(candidate_path, [candidate()])

        manifest = run_s2_current_view_publish(
            workspace=self.workspace,
            candidate_outputs_jsonl=candidate_path,
            allow_unreviewed_experiment=True,
        )

        self.assertEqual(manifest["counts"]["current_active_count"], 1)
        current = read_jsonl(self.workspace / "s2_current" / "s2_current.jsonl")
        self.assertEqual(current[0]["current_view_status"], "accepted_for_experiment")
        self.assertTrue(current[0]["participates_in_default_query"])

    def test_writes_default_latest_view_alias_for_s2_consumers(self):
        candidate_path = self.workspace / "s2_candidates.jsonl"
        decisions_path = self.workspace / "s2_review_decisions.jsonl"
        write_jsonl(candidate_path, [candidate()])
        write_jsonl(decisions_path, [decision("review-s2-1")])

        run_s2_current_view_publish(
            workspace=self.workspace,
            candidate_outputs_jsonl=candidate_path,
            review_decisions_jsonl=decisions_path,
        )

        latest_path, resolution = resolve_latest_view_input(
            workspace=self.workspace,
            layer="s2",
            canonical_path=self.workspace / "portrait" / "reviewed_units.jsonl",
        )
        self.assertEqual(resolution["source"], "default_latest_view")
        self.assertEqual(latest_path, self.workspace / "maintenance" / "latest_views" / "s2_latest_view.jsonl")
        self.assertEqual(len(read_jsonl(latest_path)), 1)
        self.assertTrue((self.workspace / "maintenance" / "latest_views" / "s2_latest_view_excluded.jsonl").exists())
        self.assertTrue((self.workspace / "maintenance" / "latest_views" / "s2_status_transitions.jsonl").exists())

    def test_rejected_s2_review_excludes_candidate(self):
        candidate_path = self.workspace / "s2_candidates.jsonl"
        decisions_path = self.workspace / "s2_review_decisions.jsonl"
        write_jsonl(candidate_path, [candidate()])
        write_jsonl(decisions_path, [decision("review-s2-1", status="rejected")])

        manifest = run_s2_current_view_publish(
            workspace=self.workspace,
            candidate_outputs_jsonl=candidate_path,
            review_decisions_jsonl=decisions_path,
        )

        self.assertEqual(manifest["counts"]["current_active_count"], 0)
        excluded = read_jsonl(self.workspace / "s2_current" / "s2_current_excluded.jsonl")
        self.assertEqual(excluded[0]["current_view_status"], "excluded_by_s2_review")


if __name__ == "__main__":
    unittest.main()
