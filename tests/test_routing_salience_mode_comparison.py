import json
import tempfile
import unittest
from pathlib import Path

from tools.routing_salience_mode_comparison import (
    compare_row,
    run_for_sidecar,
    sample_rows,
)


def sidecar_row(
    row_id: str,
    text: str,
    routes: dict[str, str] | None = None,
    group_scores: dict[str, float] | None = None,
    salience_scores: dict[str, float] | None = None,
    salience_dimensions: dict[str, float] | None = None,
    feature_summary: dict | None = None,
) -> dict:
    return {
        "schema_version": "routing.salience_sidecar.v0.2",
        "workspace_id": "test_workspace",
        "row_id": row_id,
        "unit_type": "sentence",
        "text_preview": text,
        "v0_1_routes": routes or {},
        "group_scores": group_scores or {},
        "salience_scores": salience_scores or {"value_score": 0.0, "risk_score": 0.0, "complexity_score": 0.0, "low_value_score": 0.0},
        "salience_dimensions": salience_dimensions or {},
        "feature_summary": feature_summary or {},
        "matched_terms": [],
        "write_permission": False,
    }


class RoutingSalienceModeComparisonTests(unittest.TestCase):
    def test_compare_row_keeps_complexity_as_assist_signal(self):
        row = sidecar_row(
            "complex",
            "Although the sentence is complex, it is not necessarily valuable.",
            salience_scores={"value_score": 0.0, "risk_score": 0.0, "complexity_score": 9.0, "low_value_score": 0.0},
        )

        compared = compare_row(row)

        self.assertEqual(compared["mode_scores"]["complexity_assist"], 9.0)
        self.assertLess(compared["profile_scores"]["s2_portrait_candidate"], 4.5)
        self.assertIn(compared["suggested_routes"]["s2_portrait_candidate"], {"skip_or_background_only", "script_only"})

    def test_low_value_text_is_not_promoted_by_ensemble(self):
        row = sidecar_row(
            "thanks",
            "Thanks!",
            routes={"s2_portrait_candidate": "script_only"},
            group_scores={"low_value_terms": -5.0},
            salience_scores={"value_score": -1.0, "risk_score": 0.0, "complexity_score": 0.0, "low_value_score": 5.0},
        )

        compared = compare_row(row)

        self.assertLess(compared["mode_scores"]["ensemble_profile"], 2.0)
        self.assertEqual(compared["suggested_routes"]["s1_memory_candidate"], "skip_or_background_only")

    def test_orthogonal_dimensions_override_legacy_mixed_scores(self):
        row = sidecar_row(
            "orthogonal",
            "This risky constraint should be reviewed separately from low-value noise.",
            group_scores={"low_value_terms": -5.0, "constraint_terms": 5.0, "risk_terms": 5.0},
            salience_scores={"value_score": -2.0, "risk_score": 7.0, "complexity_score": 1.0, "low_value_score": 5.0},
            salience_dimensions={
                "modeling_value_score": 12.0,
                "positive_salience_score": 10.0,
                "risk_score": 5.0,
                "constraint_score": 5.0,
                "processing_complexity_score": 1.0,
                "low_value_score": 5.0,
            },
        )

        compared = compare_row(row)

        self.assertEqual(compared["raw_signal_summary"]["modeling_value_raw"], 12.0)
        self.assertEqual(compared["raw_signal_summary"]["constraint_raw"], 5.0)
        self.assertNotEqual(compared["suggested_routes"]["s2_portrait_candidate"], "skip_or_background_only")

    def test_explicit_s1_memory_signal_lifts_first_person_event_only_for_s1(self):
        row = sidecar_row(
            "lost-job",
            "Unfortunately, I also lost my job at Door Dash this month.",
            feature_summary={"external_corpus_metrics": {"sklearn_tfidf": {"sum_tfidf": 3.0, "top_terms": [{"term": "lost job"}]}}},
            salience_scores={"value_score": 0.0, "risk_score": 0.0, "complexity_score": 3.0, "low_value_score": 0.0},
        )

        compared = compare_row(row)

        self.assertGreater(compared["raw_signal_summary"]["explicit_s1_memory_signal"], 0)
        self.assertEqual(compared["suggested_routes"]["s1_memory_candidate"], "script_only")
        self.assertEqual(compared["suggested_routes"]["s2_portrait_candidate"], "skip_or_background_only")

    def test_s1_memory_signal_does_not_lift_questions_without_source(self):
        row = sidecar_row(
            "question",
            "What got you into this biz?",
            feature_summary={"external_corpus_metrics": {"sklearn_tfidf": {"sum_tfidf": 3.0, "top_terms": [{"term": "biz"}]}}},
            salience_scores={"value_score": 0.0, "risk_score": 0.0, "complexity_score": 0.0, "low_value_score": 0.0},
        )

        compared = compare_row(row)

        self.assertEqual(compared["raw_signal_summary"]["explicit_s1_memory_signal"], 0.0)
        self.assertEqual(compared["suggested_routes"]["s1_memory_candidate"], "skip_or_background_only")

    def test_chinese_first_person_event_lifts_s1_memory_profile(self):
        row = sidecar_row(
            "zh-event",
            "我这个月失业了，正在准备开一家舞蹈工作室。",
            feature_summary={
                "external_metrics": {"wordfreq": {"zh_min_zipf": 3.2}},
                "external_corpus_metrics": {"sklearn_tfidf": {"sum_tfidf": 4.0, "top_terms": [{"term": "失业"}]}},
            },
            salience_scores={"value_score": 0.0, "risk_score": 0.0, "complexity_score": 3.0, "low_value_score": 0.0},
        )

        compared = compare_row(row)

        self.assertGreater(compared["raw_signal_summary"]["explicit_s1_memory_signal"], 0)
        self.assertIn(compared["suggested_routes"]["s1_memory_candidate"], {"script_only", "weak_llm_proposal"})

    def test_tfidf_high_seed_low_bucket_is_sampled(self):
        rows = [
            compare_row(
                sidecar_row(
                    "tfidf",
                    "Rare project phrase.",
                    feature_summary={"external_corpus_metrics": {"sklearn_tfidf": {"sum_tfidf": 7.0, "top_terms": [{"term": "rare project"}]}}},
                )
            ),
            compare_row(
                sidecar_row(
                    "seed",
                    "Mira needs privacy.",
                    group_scores={"preference_terms": 8.0, "constraint_terms": 5.0},
                    salience_scores={"value_score": 13.0, "risk_score": 2.0, "complexity_score": 1.0, "low_value_score": 0.0},
                )
            ),
        ]

        samples = sample_rows(rows, per_bucket=10, max_total=80)
        sampled = {row["row_id"]: row for row in samples}

        self.assertIn("tfidf", sampled)
        self.assertIn("tfidf_high_seed_low", sampled["tfidf"]["sample_buckets"])

    def test_run_for_sidecar_writes_comparison_outputs_without_modifying_input(self):
        rows = [
            sidecar_row(
                "a",
                "Mira needs privacy and evidence refs.",
                routes={"s1_memory_candidate": "weak_llm_proposal"},
                group_scores={"preference_terms": 4.0, "constraint_terms": 5.0, "evidence_directness_terms": 3.0},
                salience_scores={"value_score": 12.0, "risk_score": 2.0, "complexity_score": 3.0, "low_value_score": 0.0},
            ),
            sidecar_row(
                "b",
                "Thanks!",
                routes={"s1_memory_candidate": "skip_or_background_only"},
                group_scores={"low_value_terms": -5.0},
                salience_scores={"value_score": -1.0, "risk_score": 0.0, "complexity_score": 0.0, "low_value_score": 5.0},
            ),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            sidecar = Path(temp_dir) / "workspace" / "routing" / "sidecar" / "salience_sidecar.jsonl"
            sidecar.parent.mkdir(parents=True)
            original = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
            sidecar.write_text(original, encoding="utf-8")

            result = run_for_sidecar(sidecar)

            self.assertEqual(sidecar.read_text(encoding="utf-8"), original)
            self.assertTrue(Path(result["mode_comparison_rows"]).exists())
            self.assertTrue(Path(result["mode_comparison_samples"]).exists())
            self.assertTrue(Path(result["mode_comparison_table"]).exists())
            self.assertTrue(Path(result["mode_comparison_report"]).exists())


if __name__ == "__main__":
    unittest.main()
