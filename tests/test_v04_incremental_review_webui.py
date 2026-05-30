import tempfile
import unittest
from pathlib import Path

from tools.maintenance.incremental_review_webui import (
    available_human_decisions,
    build_batch_decisions,
    build_review_decision,
    build_review_session_manifest,
    filter_items,
    label_action,
    read_jsonl,
    render_detail_html,
    render_page,
    render_session_controls,
    upsert_review_decision,
    validate_review_decision,
    write_jsonl,
)


def review_item(item_id: str, *, priority: str, patch_action: str, allowed: list[str]) -> dict:
    return {
        "review_item_id": item_id,
        "patch_id": f"patch:{item_id}",
        "patch_hash": f"hash:{item_id}",
        "source_context_hash": f"source:{item_id}",
        "operation_id": "op-webui",
        "review_stage": "s1_review",
        "layer": "s1",
        "patch_action": patch_action,
        "queue_priority": priority,
        "allowed_review_actions": allowed,
        "default_recommendation": allowed[0] if allowed else "",
        "review_status": "pending_review",
        "review_summary": "Review this proposed incremental change.",
        "risk_flags": ["candidate_not_truth"],
        "review_rubric": ["Is the proposed action supported by evidence?"],
        "context_cards": [
            {
                "role": "target",
                "object_id": f"context:{item_id}",
                "overlay_status": "current",
                "text": "Original evidence text remains visible.",
                "evidence_refs": [f"evidence:{item_id}"],
                "warnings": [],
            }
        ],
    }


class V04IncrementalReviewWebuiTests(unittest.TestCase):
    def test_build_review_decision_preserves_hashes_and_does_not_apply(self):
        item = review_item("review-1", priority="high", patch_action="append_s1_contradiction_candidate", allowed=["split_current_vs_historical"])
        decision = build_review_decision(
            item,
            review_action="split_current_vs_historical",
            review_notes="Keep old as historical and use new as current.",
            edited_payload={"note": "edited"},
            reviewed_at="2026-05-28T00:00:00+00:00",
        )

        self.assertEqual(decision["review_status"], "approved_with_edit")
        self.assertEqual(decision["review_stage"], "s1_review")
        self.assertEqual(decision["patch_hash"], "hash:review-1")
        self.assertEqual(decision["source_context_hash"], "source:review-1")
        self.assertFalse(decision["write_permission"])
        self.assertFalse(decision["apply_executed"])
        self.assertEqual(decision["edited_payload"], {"note": "edited"})
        self.assertEqual(decision["internal_review_action"], "split_current_vs_historical")

    def test_rejects_action_not_allowed_for_item(self):
        item = review_item("review-1", priority="high", patch_action="append_s1_contradiction_candidate", allowed=["split_current_vs_historical"])
        self.assertIn("review_action_not_allowed_for_item", validate_review_decision(item, "accept_noop"))

    def test_human_decision_approves_recommended_internal_action(self):
        item = review_item(
            "review-1",
            priority="high",
            patch_action="append_s1_contradiction_candidate",
            allowed=["split_current_vs_historical", "reject_patch", "defer", "needs_more_evidence"],
        )
        item["default_recommendation"] = "split_current_vs_historical"

        decision = build_review_decision(item, review_action="", human_decision="approve_recommended")

        self.assertEqual(decision["human_decision"], "approve_recommended")
        self.assertEqual(decision["approved_recommended_action"], "split_current_vs_historical")
        self.assertEqual(decision["internal_review_action"], "split_current_vs_historical")
        self.assertEqual(decision["review_action"], "split_current_vs_historical")

    def test_available_human_decisions_hide_internal_action_menu(self):
        item = review_item(
            "review-1",
            priority="high",
            patch_action="append_s1_contradiction_candidate",
            allowed=["split_current_vs_historical", "reject_patch", "defer", "needs_more_evidence"],
        )
        item["default_recommendation"] = "split_current_vs_historical"

        self.assertEqual(
            available_human_decisions(item),
            ["approve_recommended", "approve_with_edit", "needs_more_evidence", "reject", "defer"],
        )

    def test_batch_accept_only_allows_low_duplicate_noop(self):
        low = review_item("review-low", priority="low", patch_action="record_duplicate_noop", allowed=["accept_noop"])
        high = review_item("review-high", priority="high", patch_action="append_s1_contradiction_candidate", allowed=["accept_patch"])
        decisions = build_batch_decisions([low, high], review_action="accept_noop")

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["review_item_id"], "review-low")
        self.assertTrue(decisions[0]["batch_decision"])

    def test_upsert_review_decision_replaces_existing_item_decision(self):
        item = review_item("review-1", priority="low", patch_action="record_duplicate_noop", allowed=["accept_noop", "defer"])
        first = build_review_decision(item, review_action="defer", reviewed_at="2026-05-28T00:00:00+00:00")
        second = build_review_decision(item, review_action="accept_noop", reviewed_at="2026-05-28T00:01:00+00:00")

        rows = upsert_review_decision([], first)
        rows = upsert_review_decision(rows, second)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["review_action"], "accept_noop")

    def test_filter_items_uses_latest_decision_status(self):
        item = review_item("review-1", priority="low", patch_action="record_duplicate_noop", allowed=["accept_noop"])
        decision = build_review_decision(item, review_action="accept_noop")
        filtered = filter_items([item], {item["review_item_id"]: decision}, {"status": ["approved_noop"]})

        self.assertEqual(filtered, [item])

    def test_filter_items_can_filter_by_review_stage(self):
        s1_item = review_item("review-s1", priority="low", patch_action="record_duplicate_noop", allowed=["accept_noop"])
        s2_item = review_item("review-s2", priority="medium", patch_action="propose_new_s2_unit", allowed=["route_s2_build"])
        s2_item["review_stage"] = "s2_review"
        s2_item["layer"] = "s2"

        filtered = filter_items([s1_item, s2_item], {}, {"stage": ["s2_review"]})

        self.assertEqual(filtered, [s2_item])

    def test_jsonl_helpers_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "decisions.jsonl"
            write_jsonl(path, [{"a": 1}, {"b": 2}])
            self.assertEqual(read_jsonl(path), [{"a": 1}, {"b": 2}])

    def test_chinese_action_labels_keep_original_action_code(self):
        self.assertEqual(label_action("accept_noop"), "确认无需修改 (accept_noop)")

    def test_render_page_includes_chinese_labels_and_auxiliary_translation(self):
        item = review_item("review-1", priority="low", patch_action="record_duplicate_noop", allowed=["accept_noop"])
        html = render_page(
            [item],
            {},
            {},
            {
                "review-1": {
                    "review_item_id": "review-1",
                    "review_summary_zh": "这是中文辅助说明。",
                    "translation_is_not_evidence": True,
                    "context_translations": [
                        {
                            "object_id": "context:review-1",
                            "text_zh": "原始证据文本仍然可见。",
                        }
                    ],
                }
            },
        )

        self.assertIn("增量审核队列", html)
        self.assertIn("确认无需修改 (accept_noop)", html)
        self.assertIn("S1 review (s1_review)", html)
        self.assertIn('name="stage"', html)
        self.assertIn('name="human_decision"', html)
        self.assertNotIn('name="review_action"', html)
        self.assertIn("推荐处理方案", html)
        self.assertIn("中文辅助说明", html)
        self.assertIn("原始证据文本仍然可见。", html)
        self.assertIn("不作为证据", html)
        self.assertIn("Original evidence text remains visible.", html)

    def test_render_page_shows_saved_banner_and_decision_summary(self):
        item = review_item("review-1", priority="low", patch_action="record_duplicate_noop", allowed=["accept_noop", "reject_patch"])
        decision = build_review_decision(item, review_action="", human_decision="reject")

        html = render_page([item], {item["review_item_id"]: decision}, {"item": ["review-1"], "saved": ["1"]})

        self.assertIn("已保存审核决定", html)
        self.assertIn("当前已保存决策", html)
        self.assertIn("decided", html)
        self.assertIn("decision-badge", html)
        self.assertIn("拒绝 (reject)", html)
        self.assertIn("拒绝修改 (reject_patch)", html)

    def test_render_detail_html_supports_partial_detail_updates(self):
        item = review_item("review-1", priority="high", patch_action="append_s1_candidate", allowed=["accept_patch", "reject_patch"])
        decision = build_review_decision(item, review_action="", human_decision="approve_recommended")

        html = render_detail_html(item, decision, saved=True)

        self.assertIn("已保存审核决定", html)
        self.assertIn("当前已保存决策", html)
        self.assertNotIn("<html", html)
        self.assertNotIn("<nav", html)

    def test_render_page_adds_partial_loading_script(self):
        item = review_item("review-1", priority="high", patch_action="append_s1_candidate", allowed=["accept_patch"])
        html = render_page([item], {}, {})

        self.assertIn('id="review-detail"', html)
        self.assertIn('partial", "detail"', html)

    def test_build_review_session_manifest_finalizes_partial_review_without_apply(self):
        first = review_item("review-1", priority="low", patch_action="record_duplicate_noop", allowed=["accept_noop"])
        second = review_item("review-2", priority="high", patch_action="append_s1_candidate", allowed=["accept_patch"])
        decision = build_review_decision(first, review_action="accept_noop", reviewed_at="2026-05-28T00:00:00+00:00")

        manifest = build_review_session_manifest(
            [first, second],
            [decision],
            finalized_at="2026-05-28T00:01:00+00:00",
        )

        self.assertEqual(manifest["review_session_status"], "finalized")
        self.assertEqual(manifest["queue_item_count"], 2)
        self.assertEqual(manifest["decision_count"], 1)
        self.assertEqual(manifest["undecided_count"], 1)
        self.assertEqual(manifest["decided_review_item_ids"], ["review-1"])
        self.assertEqual(manifest["undecided_review_item_ids"], ["review-2"])
        self.assertFalse(manifest["write_permission"])
        self.assertFalse(manifest["apply_executed"])

    def test_render_session_controls_shows_finalize_submit(self):
        item = review_item("review-1", priority="low", patch_action="record_duplicate_noop", allowed=["accept_noop"])
        decision = build_review_decision(item, review_action="accept_noop")

        html = render_session_controls([item], {item["review_item_id"]: decision})

        self.assertIn("提交本轮审核", html)
        self.assertIn("已审", html)

    def test_render_session_controls_shows_finalized_manifest_status(self):
        item = review_item("review-1", priority="low", patch_action="record_duplicate_noop", allowed=["accept_noop"])
        decision = build_review_decision(item, review_action="accept_noop", reviewed_at="2026-05-28T00:00:00+00:00")
        manifest = build_review_session_manifest([item], [decision], finalized_at="2026-05-28T00:01:00+00:00")

        html = render_session_controls([item], {item["review_item_id"]: decision}, manifest)

        self.assertIn("本轮审核已提交", html)


if __name__ == "__main__":
    unittest.main()
