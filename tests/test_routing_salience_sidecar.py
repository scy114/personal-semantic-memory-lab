import tempfile
import unittest
from pathlib import Path

from tools.routing_salience_sidecar import (
    collect_dynamic_terms,
    external_corpus_features,
    load_static_terms,
    optional_wheel_status,
    readability_metrics,
    run,
    salience_for_text,
    sentence_complexity_metrics,
    tokenize,
)


ROOT = Path(__file__).resolve().parents[1]


class RoutingSalienceSidecarTests(unittest.TestCase):
    def test_weighted_salience_lexicon_scores_value_and_low_value_terms(self):
        terms = load_static_terms(ROOT / "configs" / "routing" / "lexicons" / "salience_core_zh_en_v0.2.yaml")

        valuable = salience_for_text(
            "Mira needs privacy and wants evidence refs preserved before conclusions.",
            terms,
            {"Mira": {"base_weight": 4, "group": "dynamic_entity_terms", "source": "test", "confidence": "high", "license_note": "test", "polarity": "neutral"}},
        )
        low_value = salience_for_text("Thanks!", terms, {})

        self.assertGreater(valuable["salience_scores"]["value_score"], 10)
        self.assertGreater(valuable["salience_dimensions"]["modeling_value_score"], 10)
        self.assertGreater(valuable["salience_dimensions"]["constraint_score"], 0)
        self.assertEqual(valuable["salience_dimensions"]["risk_score"], valuable["salience_scores"]["risk_score"])
        self.assertGreater(low_value["salience_scores"]["low_value_score"], 0)
        self.assertGreater(low_value["salience_dimensions"]["low_value_score"], 0)
        self.assertLess(low_value["salience_scores"]["value_score"], valuable["salience_scores"]["value_score"])
        self.assertIn("readability_metrics", valuable["feature_summary"])
        self.assertIn("external_metrics", valuable["feature_summary"])

    def test_dynamic_entity_lexicon_uses_metadata_and_repeated_names(self):
        rows = [
            {"speaker": "Gina", "participant_ids": ["Gina", "Jon"], "text": "Gina met Jon near Navy Office."},
            {"speaker": "Gina", "participant_ids": ["Gina", "Jon"], "text": "Jon mentioned Navy Office again."},
        ]

        dynamic = collect_dynamic_terms(rows)

        self.assertIn("Gina", dynamic)
        self.assertIn("Jon", dynamic)
        self.assertIn("Navy Office", dynamic)
        self.assertEqual(dynamic["Gina"]["confidence"], "high")

    def test_readability_metrics_score_complex_sentences_higher_than_simple_ones(self):
        simple = readability_metrics("Thanks.")
        complex_sentence = readability_metrics(
            "Although the implementation remained reversible, the attribution boundary became difficult because several entities and claims were compressed into one sentence."
        )

        self.assertIsNotNone(simple["flesch_kincaid_grade"])
        self.assertIsNotNone(complex_sentence["flesch_kincaid_grade"])
        self.assertEqual(simple["readability_reliability"], "low_short_text")
        self.assertEqual(complex_sentence["readability_reliability"], "standard")
        self.assertGreater(complex_sentence["flesch_kincaid_grade"], simple["flesch_kincaid_grade"])

    def test_sentence_complexity_marks_short_text_unreliable(self):
        simple = sentence_complexity_metrics("Definitely!")
        complex_sentence = sentence_complexity_metrics(
            "Although the source text is short, it contains several clauses, entities, and qualifications because attribution remains ambiguous."
        )

        self.assertEqual(simple["syntactic_complexity_reliability"], "low_short_text")
        self.assertEqual(simple["syntactic_complexity_score"], 0)
        self.assertEqual(complex_sentence["syntactic_complexity_reliability"], "standard")
        self.assertGreater(complex_sentence["syntactic_complexity_score"], simple["syntactic_complexity_score"])

    def test_external_corpus_features_add_sklearn_tfidf_when_available(self):
        features = external_corpus_features(
            {
                "a": "Mira needs privacy and evidence refs for the router.",
                "b": "Thanks.",
            }
        )
        status = optional_wheel_status()
        if status["sklearn"] == "available":
            self.assertIn("a", features)
            self.assertIn("sklearn_tfidf", features["a"])
            self.assertGreaterEqual(features["a"]["sklearn_tfidf"]["sum_tfidf"], 0)
        else:
            self.assertEqual(features, {})

    def test_chinese_tokenization_and_salience_use_optional_jieba_path(self):
        terms = load_static_terms(ROOT / "configs" / "routing" / "lexicons" / "salience_core_zh_en_v0.2.yaml")
        text = "我这个月失业了，正在准备开一家舞蹈工作室。"

        tokens = tokenize(text)
        scored = salience_for_text(text, terms, {})

        status = optional_wheel_status()
        if status["jieba"] == "available":
            self.assertIn("失业", tokens)
            self.assertIn("工作室", tokens)
            self.assertEqual(scored["feature_summary"]["language_hints"]["tokenizer"], "jieba")
            if status["wordfreq"] == "available":
                self.assertIsNotNone(scored["feature_summary"]["external_metrics"]["wordfreq"]["zh_min_zipf"])
            else:
                self.assertEqual(scored["feature_summary"]["external_metrics"]["module_status"]["wordfreq"], "missing")
        self.assertGreater(scored["feature_summary"]["token_count"], 0)
        self.assertIn("失业", scored["feature_summary"]["top_keyphrases"])

    def test_chinese_tfidf_uses_segmented_terms(self):
        features = external_corpus_features(
            {
                "event": "我这个月失业了，正在准备开一家舞蹈工作室。",
                "low": "谢谢。",
            }
        )
        status = optional_wheel_status()
        if status["sklearn"] == "available" and status["jieba"] == "available":
            terms = [item["term"] for item in features["event"]["sklearn_tfidf"]["top_terms"]]
            self.assertIn("失业", terms)
            self.assertIn("工作室", terms)
        elif status["sklearn"] == "available":
            self.assertIn("event", features)

    def test_optional_wheel_status_reports_known_modules(self):
        status = optional_wheel_status()

        self.assertIn("textstat", status)
        self.assertIn("jieba", status)
        self.assertIn("vaderSentiment", status)
        self.assertIn("sklearn", status)
        self.assertIn(status["textstat"], {"available", "missing"})

    def test_sidecar_run_does_not_modify_route_decisions(self):
        workspace = ROOT / "users" / "_v01_blind_generic_text_section_20260520"
        route_path = workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl"
        before = route_path.read_text(encoding="utf-8") if route_path.exists() else None

        with tempfile.TemporaryDirectory() as temp_dir:
            output_name = Path(temp_dir).name
            result = run(
                type(
                    "Args",
                    (),
                    {
                        "project_root": str(ROOT),
                        "workspace": str(workspace),
                        "lexicon": "configs/routing/lexicons/salience_core_zh_en_v0.2.yaml",
                        "output_name": output_name,
                    },
                )()
            )
            self.assertTrue(Path(result["salience_sidecar"]).exists())
            self.assertTrue(Path(result["report"]).exists())

        after = route_path.read_text(encoding="utf-8") if route_path.exists() else None
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
