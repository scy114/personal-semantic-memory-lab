import tempfile
import unittest
from pathlib import Path

from tools.maintenance.s1_delta_comparison import (
    build_s1_delta_report,
    build_s1_delta_report_from_files,
    read_jsonl,
    write_jsonl,
)


class V04S1DeltaComparisonTests(unittest.TestCase):
    def test_marks_same_text_and_evidence_as_no_material_change(self):
        active = [
            {
                "memory_id": "old-1",
                "subject_id": "Mira",
                "processed_text": "Mira wants architecture-first reviews.",
                "evidence_refs": ["evidence:1"],
            }
        ]
        new = [
            {
                "memory_id": "new-1",
                "subject_id": "Mira",
                "processed_text": "Mira wants architecture-first reviews.",
                "evidence_refs": ["evidence:1"],
            }
        ]

        bundle = build_s1_delta_report(
            new_rows=new,
            active_rows=active,
            operation_id="op-1",
            generated_at="2026-05-26T00:00:00+00:00",
        )

        decision = bundle["decisions"][0]
        self.assertEqual(decision["delta_type"], "no_material_change")
        self.assertEqual(decision["top_matches"][0]["existing_s1_unit_id"], "old-1")
        self.assertTrue(decision["script_is_not_semantic_truth"])
        self.assertFalse(decision["write_permission"])

    def test_marks_overlapping_evidence_as_duplicate_candidate(self):
        bundle = build_s1_delta_report(
            new_rows=[
                {
                    "memory_id": "new-1",
                    "subject_id": "Mira",
                    "processed_text": "Mira refined her architecture review preference.",
                    "evidence_refs": ["evidence:1"],
                }
            ],
            active_rows=[
                {
                    "memory_id": "old-1",
                    "subject_id": "Mira",
                    "processed_text": "Mira wants architecture-first reviews.",
                    "evidence_refs": ["evidence:1"],
                }
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        decision = bundle["decisions"][0]
        self.assertEqual(decision["delta_type"], "duplicate_candidate")
        self.assertIn("same_or_overlapping_evidence_refs", decision["review_reasons"])
        self.assertEqual(decision["decision_confidence"], "review_recommended")

    def test_marks_low_similarity_as_new_unit(self):
        bundle = build_s1_delta_report(
            new_rows=[
                {
                    "memory_id": "new-1",
                    "subject_id": "Mira",
                    "processed_text": "Mira started learning pottery.",
                    "evidence_refs": ["evidence:2"],
                }
            ],
            active_rows=[
                {
                    "memory_id": "old-1",
                    "subject_id": "Mira",
                    "processed_text": "Mira wants architecture-first reviews.",
                    "evidence_refs": ["evidence:1"],
                }
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        self.assertEqual(bundle["decisions"][0]["delta_type"], "new_unit")
        self.assertEqual(bundle["report"]["delta_type_counts"], {"new_unit": 1})
        self.assertEqual(bundle["report"]["sample_decisions"][0]["new_s1_unit_id"], "new-1")
        self.assertIn("new_unit", bundle["report"]["sample_decisions_by_delta_type"])
        self.assertIn("embedding_similarity", bundle["report"]["method_followups_not_yet_run"])

    def test_routes_medium_similarity_to_review_instead_of_claiming_contradiction(self):
        bundle = build_s1_delta_report(
            new_rows=[
                {
                    "memory_id": "new-1",
                    "subject_id": "Mira",
                    "processed_text": "Mira wants lightweight review notes before implementation.",
                    "evidence_refs": ["evidence:2"],
                }
            ],
            active_rows=[
                {
                    "memory_id": "old-1",
                    "subject_id": "Mira",
                    "processed_text": "Mira wants architecture-first reviews before implementation.",
                    "evidence_refs": ["evidence:1"],
                }
            ],
            generated_at="2026-05-26T00:00:00+00:00",
        )

        decision = bundle["decisions"][0]
        self.assertEqual(decision["delta_type"], "needs_review")
        self.assertEqual(decision["llm_assist_status"], "recommended")
        self.assertNotIn(decision["delta_type"], {"contradicts", "weakens"})

    def test_writes_report_bundle_from_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            new_path = root / "new_s1.jsonl"
            active_path = root / "active_s1.jsonl"
            output_dir = root / "out"
            write_jsonl(
                new_path,
                [{"memory_id": "new-1", "processed_text": "A new unrelated fact.", "evidence_refs": ["e2"]}],
            )
            write_jsonl(
                active_path,
                [{"memory_id": "old-1", "processed_text": "An old fact.", "evidence_refs": ["e1"]}],
            )

            bundle = build_s1_delta_report_from_files(
                new_s1_jsonl=new_path,
                active_s1_jsonl=active_path,
                output_dir=output_dir,
                operation_id="op-file",
            )

            self.assertTrue((output_dir / "s1_delta_decisions.jsonl").exists())
            self.assertTrue((output_dir / "s1_delta_report.json").exists())
            written = read_jsonl(output_dir / "s1_delta_decisions.jsonl")
            self.assertEqual(written[0]["operation_id"], "op-file")
            self.assertEqual(bundle["report"]["decision_count"], 1)


if __name__ == "__main__":
    unittest.main()
