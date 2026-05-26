import json
import tempfile
import unittest
from pathlib import Path

from tools.maintenance.latest_view import (
    build_latest_view,
    build_latest_view_from_files,
    read_jsonl,
    write_json,
    write_jsonl,
)


class V04LatestViewTests(unittest.TestCase):
    def test_filters_impacted_and_inactive_rows_without_mutating_input(self):
        rows = [
            {"memory_id": "m1", "status": "accepted_for_experiment", "text": "active"},
            {"memory_id": "m2", "status": "accepted_for_experiment", "text": "will be stale"},
            {"memory_id": "m3", "status": "deprecated", "text": "old"},
        ]
        original = json.loads(json.dumps(rows))

        bundle = build_latest_view(
            rows,
            layer="s1",
            impacted_ids=["m2"],
            operation_id="op-1",
            source_path="memory_units.jsonl",
            generated_at="2026-05-26T00:00:00+00:00",
        )

        self.assertEqual(rows, original)
        self.assertEqual([row["memory_id"] for row in bundle["active_rows"]], ["m1"])
        excluded_by_id = {row["memory_id"]: row for row in bundle["excluded_rows"]}
        self.assertEqual(
            excluded_by_id["m2"]["_maintenance_latest_view"]["latest_view_status"],
            "stale_candidate",
        )
        self.assertEqual(
            excluded_by_id["m3"]["_maintenance_latest_view"]["latest_view_status"],
            "excluded",
        )
        self.assertEqual(bundle["manifest"]["active_count"], 1)
        self.assertEqual(bundle["manifest"]["excluded_count"], 2)
        self.assertFalse(bundle["manifest"]["write_permission"])

    def test_excludes_rows_without_stable_object_id(self):
        bundle = build_latest_view(
            [{"text": "missing id"}],
            layer="s2",
            generated_at="2026-05-26T00:00:00+00:00",
        )

        self.assertEqual(bundle["active_rows"], [])
        self.assertEqual(
            bundle["excluded_rows"][0]["_maintenance_latest_view"]["reason"],
            "missing_object_id",
        )

    def test_builds_file_bundle_from_impact_report(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "graph_edges_table.jsonl"
            output_dir = root / "latest"
            report_path = root / "incremental_impact_report.json"
            write_jsonl(
                input_path,
                [
                    {"edge_id": "edge-1", "relation_type": "supports"},
                    {"edge_id": "edge-2", "relation_type": "updates"},
                ],
            )
            write_json(
                report_path,
                {
                    "operation_id": "op-graph",
                    "impacted": {"graph_candidate_ids": ["edge-2"]},
                },
            )

            bundle = build_latest_view_from_files(
                input_jsonl=input_path,
                layer="graph",
                output_dir=output_dir,
                impact_report=report_path,
            )

            self.assertEqual([row["edge_id"] for row in bundle["active_rows"]], ["edge-1"])
            self.assertEqual([row["edge_id"] for row in bundle["excluded_rows"]], ["edge-2"])
            self.assertTrue((output_dir / "graph_latest_view.jsonl").exists())
            self.assertTrue((output_dir / "graph_latest_view_excluded.jsonl").exists())
            self.assertTrue((output_dir / "graph_latest_view_manifest.json").exists())
            written_active = read_jsonl(output_dir / "graph_latest_view.jsonl")
            self.assertEqual(written_active[0]["_maintenance_latest_view"]["layer"], "graph")


if __name__ == "__main__":
    unittest.main()
