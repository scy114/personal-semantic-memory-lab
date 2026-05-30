import tempfile
import unittest
from pathlib import Path

from tools.maintenance.s1_review_overlay_consumer import (
    build_s1_review_latest_view,
    build_s1_review_latest_view_from_files,
    read_jsonl,
    write_jsonl,
)


class V04S1ReviewOverlayConsumerTests(unittest.TestCase):
    def test_accepts_reviewed_current_candidate_and_excludes_historical_old_row(self):
        bundle = build_s1_review_latest_view(
            active_s1_rows=[
                {"memory_id": "memory-old", "content": "Contemporary is top pick."},
                {"memory_id": "memory-keep", "content": "Keep this."},
            ],
            incremental_s1_rows=[
                {"memory_id": "memory-new", "content": "Salsa replaces contemporary."},
            ],
            s1_review_overlay_rows=[
                {
                    "review_item_id": "review-1",
                    "patch_id": "patch-1",
                    "operation_id": "op-1",
                    "overlay_effect": "split_current_and_historical_overlay",
                    "current_candidate_ids": ["memory-new"],
                    "historical_or_stale_candidate_ids": ["memory-old"],
                }
            ],
            generated_at="2026-05-28T00:00:00+00:00",
        )

        latest_ids = [row["memory_id"] for row in bundle["latest_rows"]]
        excluded_ids = [row["memory_id"] for row in bundle["excluded_rows"]]
        self.assertEqual(latest_ids, ["memory-keep", "memory-new"])
        self.assertEqual(excluded_ids, ["memory-old"])
        self.assertEqual(bundle["latest_rows"][1]["_maintenance_review_latest_view"]["latest_view_status"], "active_after_review")
        self.assertEqual(bundle["excluded_rows"][0]["_maintenance_review_latest_view"]["latest_view_status"], "historical_or_stale")
        self.assertFalse(bundle["manifest"]["write_permission"])
        self.assertFalse(bundle["manifest"]["apply_executed"])

    def test_reports_missing_current_candidate_without_inventing_row(self):
        bundle = build_s1_review_latest_view(
            active_s1_rows=[],
            incremental_s1_rows=[],
            s1_review_overlay_rows=[
                {
                    "review_item_id": "review-1",
                    "current_candidate_ids": ["memory-missing"],
                    "historical_or_stale_candidate_ids": [],
                }
            ],
            generated_at="2026-05-28T00:00:00+00:00",
        )

        self.assertEqual(bundle["latest_rows"], [])
        self.assertEqual(bundle["manifest"]["missing_current_candidate_ids"], ["memory-missing"])

    def test_can_confirm_stale_row_from_existing_excluded_input(self):
        bundle = build_s1_review_latest_view(
            active_s1_rows=[],
            incremental_s1_rows=[{"memory_id": "memory-new", "content": "new"}],
            existing_excluded_s1_rows=[{"memory_id": "memory-old", "content": "old"}],
            s1_review_overlay_rows=[
                {
                    "review_item_id": "review-1",
                    "current_candidate_ids": ["memory-new"],
                    "historical_or_stale_candidate_ids": ["memory-old"],
                }
            ],
            generated_at="2026-05-28T00:00:00+00:00",
        )

        self.assertEqual([row["memory_id"] for row in bundle["latest_rows"]], ["memory-new"])
        self.assertEqual([row["memory_id"] for row in bundle["excluded_rows"]], ["memory-old"])
        self.assertEqual(bundle["manifest"]["missing_stale_candidate_ids"], [])
        self.assertEqual(
            bundle["excluded_rows"][0]["_maintenance_review_latest_view"]["reason"],
            "confirmed_historical_from_existing_excluded_input",
        )

    def test_writes_bundle_from_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            active = root / "active.jsonl"
            incremental = root / "incremental.jsonl"
            overlay = root / "overlay.jsonl"
            output_dir = root / "out"
            write_jsonl(active, [{"memory_id": "memory-old", "content": "old"}])
            write_jsonl(incremental, [{"memory_id": "memory-new", "content": "new"}])
            write_jsonl(
                overlay,
                [
                    {
                        "review_item_id": "review-1",
                        "patch_id": "patch-1",
                        "operation_id": "op-1",
                        "overlay_effect": "materialize_candidate_overlay",
                        "current_candidate_ids": ["memory-new"],
                        "historical_or_stale_candidate_ids": [],
                    }
                ],
            )

            bundle = build_s1_review_latest_view_from_files(
                active_s1_jsonl=active,
                incremental_s1_jsonl=incremental,
                s1_review_overlay_jsonl=overlay,
                output_dir=output_dir,
            )

            self.assertTrue((output_dir / "s1_latest_view_after_review.jsonl").exists())
            self.assertTrue((output_dir / "s1_latest_view_after_review_excluded.jsonl").exists())
            self.assertTrue((output_dir / "s1_latest_view_after_review_manifest.json").exists())
            self.assertTrue((output_dir / "s1_latest_view_after_review_report.md").exists())
            written = read_jsonl(output_dir / "s1_latest_view_after_review.jsonl")
            self.assertEqual([row["memory_id"] for row in written], ["memory-old", "memory-new"])
            self.assertEqual(bundle["manifest"]["latest_view_count"], 2)


if __name__ == "__main__":
    unittest.main()
