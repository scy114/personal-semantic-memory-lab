import tempfile
import unittest
from pathlib import Path

from tools.maintenance.incremental_rebuild_comparator import (
    build_incremental_vs_full_rebuild_report,
    build_incremental_vs_full_rebuild_report_from_files,
    read_jsonl,
    write_json,
    write_jsonl,
)


class V04IncrementalRebuildComparatorTests(unittest.TestCase):
    def test_reports_pass_when_incremental_matches_full_rebuild_observation(self):
        bundle = build_incremental_vs_full_rebuild_report(
            observation_payload={
                "schema_version": "maintenance.full_rebuild_observation.v0.4",
                "observation_name": "synthetic-pass",
                "observations": [
                    {
                        "new_s1_unit_id": "memory-dup",
                        "expected_s1_delta_type": "no_material_change",
                        "expected_s2_recommended_action": "no_s2_refresh_expected",
                        "expected_affected_s2_unit_ids": ["s2-1"],
                        "expected_graph_recommended_action": "no_graph_refresh_expected",
                        "expected_primary_graph_packet_ids": ["graph_packet:1"],
                    },
                    {
                        "new_s1_unit_id": "memory-new",
                        "expected_s1_delta_type": "new_unit",
                        "expected_s2_recommended_action": "route_s2_for_new_s1_unit",
                        "expected_affected_s2_unit_ids": [],
                        "expected_graph_recommended_action": "route_graph_extraction_for_new_packet",
                        "expected_primary_graph_packet_ids": [],
                    },
                ],
            },
            s1_delta_decisions=[
                {"new_s1_unit_id": "memory-dup", "delta_type": "no_material_change"},
                {"new_s1_unit_id": "memory-new", "delta_type": "new_unit"},
            ],
            s2_affected_rows=[
                {
                    "new_s1_unit_id": "memory-dup",
                    "recommended_action": "no_s2_refresh_expected",
                    "affected_s2_unit_id": "s2-1",
                },
                {
                    "new_s1_unit_id": "memory-new",
                    "recommended_action": "route_s2_for_new_s1_unit",
                    "affected_s2_unit_id": "",
                },
            ],
            graph_affected_rows=[
                {
                    "new_s1_unit_id": "memory-dup",
                    "recommended_action": "no_graph_refresh_expected",
                    "primary_graph_packet_ids": ["graph_packet:1"],
                },
                {
                    "new_s1_unit_id": "memory-new",
                    "recommended_action": "route_graph_extraction_for_new_packet",
                    "primary_graph_packet_ids": [],
                },
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        self.assertEqual(bundle["report"]["acceptance_status"], "pass")
        self.assertEqual(bundle["report"]["pass_count"], 2)
        self.assertEqual(bundle["report"]["fail_count"], 0)
        self.assertTrue(all(row["comparison_status"] == "pass" for row in bundle["comparison_rows"]))
        self.assertFalse(bundle["report"]["write_permission"])

    def test_reports_fail_when_incremental_misses_expected_refresh(self):
        bundle = build_incremental_vs_full_rebuild_report(
            observation_payload={
                "schema_version": "maintenance.full_rebuild_observation.v0.4",
                "observations": [
                    {
                        "new_s1_unit_id": "memory-new",
                        "expected_s1_delta_type": "new_unit",
                        "expected_graph_recommended_action": "route_graph_extraction_for_new_packet",
                        "expected_primary_graph_packet_ids": ["graph_packet:new"],
                    }
                ],
            },
            s1_delta_decisions=[{"new_s1_unit_id": "memory-new", "delta_type": "new_unit"}],
            s2_affected_rows=[],
            graph_affected_rows=[
                {
                    "new_s1_unit_id": "memory-new",
                    "recommended_action": "no_graph_refresh_expected",
                    "primary_graph_packet_ids": [],
                }
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        self.assertEqual(bundle["report"]["acceptance_status"], "fail")
        self.assertEqual(bundle["report"]["fail_count"], 1)
        self.assertIn("graph_recommended_action", bundle["comparison_rows"][0]["mismatches"][0])
        self.assertTrue(bundle["comparison_rows"][0]["incremental_is_not_full_rebuild"])

    def test_writes_report_bundle_from_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            observation_path = root / "observation.json"
            s1_path = root / "s1_delta.jsonl"
            s2_path = root / "s2_affected.jsonl"
            graph_path = root / "graph_affected.jsonl"
            output_dir = root / "out"
            write_json(
                observation_path,
                {
                    "schema_version": "maintenance.full_rebuild_observation.v0.4",
                    "observations": [
                        {
                            "new_s1_unit_id": "memory-1",
                            "expected_s1_delta_type": "new_unit",
                        }
                    ],
                },
            )
            write_jsonl(s1_path, [{"new_s1_unit_id": "memory-1", "delta_type": "new_unit"}])
            write_jsonl(s2_path, [])
            write_jsonl(graph_path, [])

            bundle = build_incremental_vs_full_rebuild_report_from_files(
                full_rebuild_observation_json=observation_path,
                s1_delta_decisions_jsonl=s1_path,
                s2_affected_rows_jsonl=s2_path,
                graph_affected_rows_jsonl=graph_path,
                output_dir=output_dir,
            )

            self.assertTrue((output_dir / "incremental_vs_full_rebuild_rows.jsonl").exists())
            self.assertTrue((output_dir / "incremental_vs_full_rebuild_report.json").exists())
            written = read_jsonl(output_dir / "incremental_vs_full_rebuild_rows.jsonl")
            self.assertEqual(written[0]["new_s1_unit_id"], "memory-1")
            self.assertEqual(bundle["report"]["acceptance_status"], "pass")


if __name__ == "__main__":
    unittest.main()
