import shutil
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_jsonl
from tools.maintenance.v04_full_review_workflow_package_runner import (
    finalize_full_review_workflow_package,
    prepare_full_review_workflow_package,
)


ROOT = Path(__file__).resolve().parents[1]


class V04FullReviewWorkflowPackageRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_v04_full_review_workflow_package_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def test_prepare_stops_before_apply_until_human_review_is_finalized(self):
        prepare = prepare_full_review_workflow_package(workspace=self.workspace)
        package_dir = Path(prepare["package_dir"])

        self.assertTrue((package_dir / "review_queue" / "incremental_review_queue.jsonl").exists())
        self.assertTrue((package_dir / "human_review" / "review_decisions.jsonl").exists())
        self.assertTrue((package_dir / "HUMAN_REVIEW_INSTRUCTIONS.md").exists())
        self.assertFalse((package_dir / "human_review" / "review_session_manifest.json").exists())
        self.assertFalse(prepare["boundary"]["apply_executed"])
        self.assertEqual(prepare["counts"]["review_queue_items"], 3)

        with self.assertRaisesRegex(FileNotFoundError, "Human review is not finalized"):
            finalize_full_review_workflow_package(workspace=self.workspace, package_dir=package_dir)

    def test_auto_review_for_test_mode_exercises_full_review_apply_publish_loop(self):
        prepare = prepare_full_review_workflow_package(workspace=self.workspace)
        package_dir = Path(prepare["package_dir"])

        final = finalize_full_review_workflow_package(
            workspace=self.workspace,
            package_dir=package_dir,
            auto_review_for_test_mode=True,
        )

        self.assertEqual(final["acceptance_status"], "pass")
        self.assertTrue(final["acceptance_checks"]["human_review_session_finalized"])
        self.assertTrue(final["acceptance_checks"]["graph_visual_slices_nonzero"])
        self.assertEqual(final["counts"]["review_decision_count"], 3)
        self.assertTrue(Path(final["outputs"]["incremental_visual_review_html"]).exists())
        decisions = read_jsonl(Path(final["outputs"]["review_decisions"]))
        self.assertEqual({row["review_item_id"] for row in decisions}, {"review-s1-1", "review-s2-1", "review-graph-1"})
        self.assertFalse(final["boundary"]["durable_memory_written"])
        self.assertFalse(final["boundary"]["graph_truth_written"])


if __name__ == "__main__":
    unittest.main()
