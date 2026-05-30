import tempfile
import unittest
from pathlib import Path

from tools.maintenance.incremental_review_decision_applier import (
    build_incremental_review_apply_bundle,
    build_incremental_review_apply_bundle_from_files,
    read_json,
    read_jsonl,
    write_json,
    write_jsonl,
)
from tools.maintenance.incremental_review_webui import build_review_decision, build_review_session_manifest


def review_item(item_id: str, *, layer: str, patch_action: str, allowed: list[str], default: str) -> dict:
    return {
        "review_item_id": item_id,
        "patch_id": f"patch:{item_id}",
        "patch_hash": f"hash:{item_id}",
        "source_context_hash": f"source:{item_id}",
        "operation_id": "op-apply",
        "layer": layer,
        "patch_action": patch_action,
        "allowed_review_actions": allowed,
        "default_recommendation": default,
        "context_cards": [
            {
                "role": "new_s1_candidate",
                "object_id": f"new:{item_id}",
                "text": "new text",
                "evidence_refs": [f"evidence:{item_id}"],
            },
            {
                "role": "existing_s1_candidate",
                "object_id": f"old:{item_id}",
                "text": "old text",
                "evidence_refs": [f"old-evidence:{item_id}"],
            },
        ],
        "evidence_refs": [f"evidence:{item_id}"],
    }


class V04IncrementalReviewDecisionApplierTests(unittest.TestCase):
    def test_build_apply_bundle_generates_overlay_and_refresh_scopes(self):
        s1 = review_item(
            "review-s1",
            layer="s1",
            patch_action="append_s1_contradiction_candidate",
            allowed=["split_current_vs_historical", "reject_patch"],
            default="split_current_vs_historical",
        )
        s2 = review_item(
            "review-s2",
            layer="s2",
            patch_action="propose_new_s2_unit",
            allowed=["route_s2_build", "reject_patch"],
            default="route_s2_build",
        )
        graph = review_item(
            "review-graph",
            layer="graph",
            patch_action="review_graph_patch_candidate",
            allowed=["route_graph_extraction", "reject_patch"],
            default="route_graph_extraction",
        )
        decisions = [
            build_review_decision(s1, review_action="", human_decision="approve_recommended", reviewed_at="2026-05-28T00:00:00+00:00"),
            build_review_decision(s2, review_action="", human_decision="approve_recommended", reviewed_at="2026-05-28T00:00:01+00:00"),
            build_review_decision(graph, review_action="", human_decision="approve_recommended", reviewed_at="2026-05-28T00:00:02+00:00"),
        ]
        manifest = build_review_session_manifest(
            [s1, s2, graph],
            decisions,
            finalized_at="2026-05-28T00:01:00+00:00",
        )

        bundle = build_incremental_review_apply_bundle(
            review_queue_items=[s1, s2, graph],
            review_decisions=decisions,
            review_session_manifest=manifest,
            generated_at="2026-05-28T00:02:00+00:00",
        )

        self.assertEqual(len(bundle["apply_plan_rows"]), 3)
        self.assertEqual(len(bundle["s1_overlay_rows"]), 1)
        self.assertEqual(len(bundle["s2_refresh_scope_rows"]), 1)
        self.assertEqual(len(bundle["graph_refresh_scope_rows"]), 1)
        self.assertFalse(bundle["report"]["write_permission"])
        self.assertFalse(bundle["report"]["apply_executed"])
        self.assertFalse(bundle["report"]["graph_truth_written"])
        self.assertEqual(bundle["s1_overlay_rows"][0]["current_candidate_ids"], ["new:review-s1"])
        self.assertEqual(bundle["s1_overlay_rows"][0]["historical_or_stale_candidate_ids"], ["old:review-s1"])

    def test_rejects_unfinalized_session(self):
        item = review_item("review-1", layer="s1", patch_action="append_s1_candidate", allowed=["accept_patch"], default="accept_patch")
        decision = build_review_decision(item, review_action="", human_decision="approve_recommended")
        manifest = build_review_session_manifest([item], [decision])
        manifest["review_session_status"] = "draft"

        with self.assertRaisesRegex(ValueError, "review_session_not_finalized"):
            build_incremental_review_apply_bundle(
                review_queue_items=[item],
                review_decisions=[decision],
                review_session_manifest=manifest,
            )

    def test_rejects_decision_hash_mismatch(self):
        item = review_item("review-1", layer="s1", patch_action="append_s1_candidate", allowed=["accept_patch"], default="accept_patch")
        decision = build_review_decision(item, review_action="", human_decision="approve_recommended")
        manifest = build_review_session_manifest([item], [decision])
        decision["review_notes"] = "changed after finalization"

        with self.assertRaisesRegex(ValueError, "review_session_decisions_hash_mismatch"):
            build_incremental_review_apply_bundle(
                review_queue_items=[item],
                review_decisions=[decision],
                review_session_manifest=manifest,
            )

    def test_from_files_writes_artifacts(self):
        item = review_item("review-1", layer="graph", patch_action="review_graph_patch_candidate", allowed=["route_graph_extraction"], default="route_graph_extraction")
        decision = build_review_decision(item, review_action="", human_decision="approve_recommended")
        manifest = build_review_session_manifest([item], [decision])

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            queue_path = root / "queue.jsonl"
            decisions_path = root / "decisions.jsonl"
            manifest_path = root / "manifest.json"
            output_dir = root / "out"
            write_jsonl(queue_path, [item])
            write_jsonl(decisions_path, [decision])
            write_json(manifest_path, manifest)

            bundle = build_incremental_review_apply_bundle_from_files(
                review_queue_jsonl=queue_path,
                review_decisions_jsonl=decisions_path,
                review_session_manifest_json=manifest_path,
                output_dir=output_dir,
            )

            self.assertTrue((output_dir / "incremental_apply_plan.jsonl").exists())
            self.assertTrue((output_dir / "graph_refresh_scope.jsonl").exists())
            self.assertTrue((output_dir / "incremental_apply_report.md").exists())
            self.assertEqual(read_json(output_dir / "incremental_apply_report.json")["apply_plan_count"], 1)
            self.assertEqual(len(read_jsonl(output_dir / "graph_refresh_scope.jsonl")), 1)
            self.assertIn("paths", bundle)


if __name__ == "__main__":
    unittest.main()
