import json
import shutil
import unittest
from pathlib import Path

from tools.maintenance.latest_view import resolve_latest_view_input
from tools.maintenance.s1_current_view_publisher import run_s1_current_view_publish


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def row(memory_id: str, content: str, *, status: str, reason: str, overlay_effect: str = "") -> dict:
    return {
        "memory_id": memory_id,
        "content": content,
        "_maintenance_review_latest_view": {
            "latest_view_status": status,
            "reason": reason,
            "review_item_id": "review-1",
            "patch_id": "patch-1",
            "operation_id": "op-1",
            "overlay_effect": overlay_effect,
        },
    }


class V04S1CurrentViewPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workspace = ROOT / "users" / "_v04_s1_current_view_publish_test"
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir(parents=True)

    def tearDown(self) -> None:
        if self.workspace.exists():
            shutil.rmtree(self.workspace)

    def seed_reviewed_s1(self, source_dir: Path) -> None:
        write_jsonl(
            source_dir / "s1_latest_view_after_review.jsonl",
            [
                row("memory-keep", "Keep this.", status="active_existing", reason="kept_from_active_s1_input"),
                row("memory-new", "New reviewed fact.", status="active_after_review", reason="accepted_by_review_overlay"),
            ],
        )
        write_jsonl(
            source_dir / "s1_latest_view_after_review_excluded.jsonl",
            [
                row(
                    "memory-old",
                    "Old fact.",
                    status="historical_or_stale",
                    reason="excluded_by_review_overlay",
                    overlay_effect="split_current_and_historical_overlay",
                )
            ],
        )
        write_json(source_dir / "s1_latest_view_after_review_manifest.json", {"schema_version": "test.s1_review"})

    def test_publishes_reviewed_s1_current_and_tags_old_memory(self):
        source_dir = self.workspace / "s1_after_review"
        self.seed_reviewed_s1(source_dir)

        manifest = run_s1_current_view_publish(workspace=self.workspace, source_dir=source_dir)
        current_dir = self.workspace / "s1_current"

        self.assertTrue((current_dir / "s1_current.jsonl").exists())
        self.assertTrue((current_dir / "s1_current_excluded.jsonl").exists())
        self.assertTrue((current_dir / "s1_status_transitions.jsonl").exists())
        self.assertTrue((current_dir / "s1_current_manifest.json").exists())
        self.assertTrue((current_dir / "s1_current_report.md").exists())
        self.assertEqual(manifest["counts"]["current_active_count"], 2)
        self.assertEqual(manifest["counts"]["current_excluded_count"], 1)
        self.assertTrue(manifest["publish_executed"])
        self.assertTrue(manifest["query_default_ready"])
        self.assertFalse(manifest["durable_writes_executed"])
        self.assertFalse(manifest["canonical_history_rewritten"])

        active_rows = read_jsonl(current_dir / "s1_current.jsonl")
        excluded_rows = read_jsonl(current_dir / "s1_current_excluded.jsonl")
        transitions = read_jsonl(current_dir / "s1_status_transitions.jsonl")

        self.assertEqual([item["memory_id"] for item in active_rows], ["memory-keep", "memory-new"])
        self.assertEqual(excluded_rows[0]["memory_id"], "memory-old")
        self.assertFalse(excluded_rows[0]["participates_in_default_query"])
        old_transition = [item for item in transitions if item["memory_id"] == "memory-old"][0]
        self.assertEqual(old_transition["current_view_status"], "historical_or_stale")
        self.assertFalse(old_transition["participates_in_default_query"])

    def test_writes_default_latest_view_alias_for_downstream_consumers(self):
        source_dir = self.workspace / "s1_after_review"
        self.seed_reviewed_s1(source_dir)

        run_s1_current_view_publish(workspace=self.workspace, source_dir=source_dir)

        latest_path, resolution = resolve_latest_view_input(
            workspace=self.workspace,
            layer="s1",
            canonical_path=self.workspace / "memory" / "memory_units.jsonl",
        )
        self.assertEqual(resolution["source"], "default_latest_view")
        self.assertEqual(latest_path, self.workspace / "maintenance" / "latest_views" / "s1_latest_view.jsonl")
        self.assertEqual([item["memory_id"] for item in read_jsonl(latest_path)], ["memory-keep", "memory-new"])
        self.assertTrue((self.workspace / "maintenance" / "latest_views" / "s1_latest_view_excluded.jsonl").exists())
        self.assertTrue((self.workspace / "maintenance" / "latest_views" / "s1_status_transitions.jsonl").exists())

    def test_republish_archives_previous_s1_current(self):
        first = self.workspace / "first"
        second = self.workspace / "second"
        self.seed_reviewed_s1(first)
        self.seed_reviewed_s1(second)
        write_jsonl(second / "s1_latest_view_after_review.jsonl", [row("memory-second", "Second.", status="active_after_review", reason="accepted")])

        run_s1_current_view_publish(workspace=self.workspace, source_dir=first)
        manifest = run_s1_current_view_publish(workspace=self.workspace, source_dir=second)

        self.assertTrue(manifest["previous_current_archive_dir"])
        self.assertTrue(Path(manifest["previous_current_archive_dir"]).exists())
        self.assertEqual([item["memory_id"] for item in read_jsonl(self.workspace / "s1_current" / "s1_current.jsonl")], ["memory-second"])


if __name__ == "__main__":
    unittest.main()
