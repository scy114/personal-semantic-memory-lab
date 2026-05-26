import tempfile
import unittest
from pathlib import Path

from tools.maintenance.s2_affected_unit_resolver import (
    build_s2_affected_unit_report,
    build_s2_affected_unit_report_from_files,
    read_jsonl,
    write_jsonl,
)


class V04S2AffectedUnitResolverTests(unittest.TestCase):
    def test_resolves_direct_s2_unit_and_index_by_evidence(self):
        bundle = build_s2_affected_unit_report(
            s1_delta_decisions=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-new",
                    "new_subject_id": "Mira",
                    "delta_type": "needs_review",
                    "new_evidence_refs": ["evidence:1"],
                    "new_source_refs": ["source-a"],
                }
            ],
            active_s2_rows=[
                {
                    "unit_id": "s2-1",
                    "user_id": "Mira",
                    "type": "preference",
                    "scope": "global",
                    "content": "Mira prefers architecture-first reviews.",
                    "evidence_refs": ["evidence:1"],
                }
            ],
            s2_index_rows=[
                {
                    "vector_entry_id": "s2idx-1",
                    "object_id": "s2-1",
                    "evidence_refs": ["evidence:1"],
                }
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        row = bundle["affected_rows"][0]
        self.assertEqual(row["affected_s2_unit_id"], "s2-1")
        self.assertEqual(row["match_type"], "evidence_overlap")
        self.assertEqual(row["s2_delta_signal"], "needs_review")
        self.assertEqual(row["s2_index_entry_ids"], ["s2idx-1"])
        self.assertEqual(bundle["report"]["impacted_s2_unit_ids"], ["s2-1"])
        self.assertEqual(bundle["report"]["impacted_s2_index_entry_ids"], ["s2idx-1"])
        self.assertFalse(row["write_permission"])
        self.assertFalse(bundle["report"]["write_permission"])

    def test_new_s1_without_direct_s2_hit_routes_to_new_profile_candidate(self):
        bundle = build_s2_affected_unit_report(
            s1_delta_decisions=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-new",
                    "new_subject_id": "Mira",
                    "delta_type": "new_unit",
                    "new_evidence_refs": ["evidence:2"],
                    "new_source_refs": ["source-b"],
                }
            ],
            active_s2_rows=[
                {
                    "unit_id": "s2-1",
                    "user_id": "Mira",
                    "content": "Existing profile unit.",
                    "evidence_refs": ["evidence:1"],
                }
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        row = bundle["affected_rows"][0]
        self.assertEqual(row["affected_s2_unit_id"], "")
        self.assertEqual(row["match_type"], "subject_scope_candidate")
        self.assertEqual(row["subject_scope_existing_s2_unit_ids"], ["s2-1"])
        self.assertEqual(row["s2_delta_signal"], "new_profile_unit")
        self.assertEqual(row["recommended_action"], "route_s2_for_new_s1_unit")
        self.assertEqual(bundle["report"]["impacted_s2_unit_ids"], [])

    def test_no_material_change_direct_hit_does_not_request_refresh(self):
        bundle = build_s2_affected_unit_report(
            s1_delta_decisions=[
                {
                    "operation_id": "op-1",
                    "new_s1_unit_id": "memory-new",
                    "new_subject_id": "Mira",
                    "delta_type": "no_material_change",
                    "new_evidence_refs": ["evidence:1"],
                }
            ],
            active_s2_rows=[
                {"unit_id": "s2-1", "user_id": "Mira", "evidence_refs": ["evidence:1"]},
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        row = bundle["affected_rows"][0]
        self.assertEqual(row["s2_delta_signal"], "no_material_change")
        self.assertEqual(row["recommended_action"], "no_s2_refresh_expected")
        self.assertEqual(row["current_vs_historical_policy"], "current_view_unchanged")

    def test_writes_report_bundle_from_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            decisions_path = root / "s1_delta_decisions.jsonl"
            s2_path = root / "active_s2.jsonl"
            index_path = root / "s2_index.jsonl"
            output_dir = root / "out"
            write_jsonl(
                decisions_path,
                [
                    {
                        "operation_id": "op-file",
                        "new_s1_unit_id": "memory-new",
                        "new_subject_id": "Mira",
                        "delta_type": "needs_review",
                        "new_evidence_refs": ["evidence:1"],
                    }
                ],
            )
            write_jsonl(s2_path, [{"unit_id": "s2-1", "user_id": "Mira", "evidence_refs": ["evidence:1"]}])
            write_jsonl(index_path, [{"vector_entry_id": "s2idx-1", "object_id": "s2-1"}])

            bundle = build_s2_affected_unit_report_from_files(
                s1_delta_decisions_jsonl=decisions_path,
                active_s2_jsonl=s2_path,
                s2_index_jsonl=index_path,
                output_dir=output_dir,
            )

            self.assertTrue((output_dir / "s2_affected_units.jsonl").exists())
            self.assertTrue((output_dir / "s2_affected_unit_report.json").exists())
            written = read_jsonl(output_dir / "s2_affected_units.jsonl")
            self.assertEqual(written[0]["operation_id"], "op-file")
            self.assertEqual(bundle["report"]["affected_row_count"], 1)


if __name__ == "__main__":
    unittest.main()
