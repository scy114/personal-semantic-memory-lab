import shutil
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_jsonl
from tools.maintenance.v04_incremental_workflow_smoke_runner import run_v04_incremental_workflow_smoke


ROOT = Path(__file__).resolve().parents[1]


class V04IncrementalWorkflowSmokeRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_v04_incremental_workflow_smoke_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def test_smoke_package_closes_post_review_incremental_publish_loop(self):
        manifest = run_v04_incremental_workflow_smoke(workspace=self.workspace)
        package_dir = Path(manifest["package_dir"])

        self.assertEqual(manifest["acceptance_status"], "pass")
        self.assertTrue((package_dir / "v04_incremental_workflow_smoke_manifest.json").exists())
        self.assertTrue((package_dir / "v04_incremental_workflow_smoke_report.md").exists())
        self.assertTrue(Path(manifest["outputs"]["incremental_visual_review_html"]).exists())

        s1_current = read_jsonl(Path(manifest["outputs"]["s1_current"]))
        s2_current = read_jsonl(Path(manifest["outputs"]["s2_current"]))
        changed = read_jsonl(Path(manifest["outputs"]["changed_graph_units"]))

        self.assertTrue(any(row.get("memory_id") == "memory-new" for row in s1_current))
        self.assertTrue(any("salsa" in row.get("content", "") for row in s2_current))
        self.assertTrue(any(row.get("graph_object_id") == "edge:jon-performance-preference" for row in changed))
        self.assertTrue(manifest["acceptance_checks"]["graph_visual_slices_nonzero"])
        self.assertFalse(manifest["boundary"]["durable_memory_written"])
        self.assertFalse(manifest["boundary"]["graph_truth_written"])


if __name__ == "__main__":
    unittest.main()
