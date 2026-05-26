import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.routing_v021_feature_matrix_builder import run


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def make_resource_inventory(root: Path) -> Path:
    vocab = root / "external_references" / "routing_vocab"
    complex_dir = vocab / "CompLex"
    mrc_dir = vocab / "MRC"
    concreteness_dir = vocab / "Concreteness-Ratings"
    thuocl_dir = vocab / "THUOCL"
    stopwords_iso_dir = vocab / "stopwords-iso"
    chinese_stopwords_dir = vocab / "chinese-stopwords-goto456"
    en_dimlex_dir = vocab / "en_dimlex"
    chinese_dimlex_dir = vocab / "chinese-dimlex"
    for path in (
        complex_dir / "train",
        mrc_dir,
        concreteness_dir,
        thuocl_dir / "data",
        stopwords_iso_dir,
        chinese_stopwords_dir,
        en_dimlex_dir,
        chinese_dimlex_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)

    (complex_dir / "train" / "lcp_single_train.tsv").write_text(
        "\n".join(
            [
                "id\tcorpus\tsentence\ttoken\tcomplexity",
                "1\ttest\tThe prototype is hard.\tprototype\t0.7",
                "2\ttest\tThe deadline is hard.\tdeadline\t0.4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (mrc_dir / "mrc.csv").write_text(
        "\n".join(
            [
                "Word,Familiarity,Concreteness,Imageability,Age of Acquisition Rating",
                "prototype,300,420,500,560",
                "deadline,500,360,420,450",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (concreteness_dir / "ratings.csv").write_text(
        "\n".join(["Word,Conc.M", "prototype,3.6", "deadline,3.2"]) + "\n",
        encoding="utf-8",
    )
    (thuocl_dir / "data" / "THUOCL_IT.txt").write_text(
        "\u9879\u76ee 1000\n\u539f\u578b 800\n",
        encoding="utf-8",
    )
    (stopwords_iso_dir / "stopwords-iso.json").write_text(
        json.dumps({"en": ["the", "is"], "zh": ["\u8fd9\u4e2a"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    (chinese_stopwords_dir / "cn_stopwords.txt").write_text("\u4e86\n\u7684\n", encoding="utf-8")
    (en_dimlex_dir / "en_dimlex.xml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?><dimlex><entry id='1' word='because'><orths><orth><part>because</part></orth></orths></entry></dimlex>",
        encoding="utf-8",
    )
    (chinese_dimlex_dir / "chinese_dimlex.xml").write_text(
        "<?xml version='1.0' encoding='UTF-8'?><dimlex>"
        "<entry id='c1' word='\u56e0\u4e3a'><orths><orth><part>\u56e0\u4e3a</part></orth></orths></entry>"
        "<entry id='c2' word='\u5982\u679c...\u90a3\u4e48...'><orths><orth><part>\u5982\u679c...\u90a3\u4e48...</part></orth></orths></entry>"
        "</dimlex>",
        encoding="utf-8",
    )

    inventory = vocab / "routing_resource_inventory.csv"
    rows = [
        ("complex", "CompLex", "external_references/routing_vocab/CompLex"),
        ("mrc", "MRC", "external_references/routing_vocab/MRC"),
        ("concreteness_ratings_hf", "Concreteness", "external_references/routing_vocab/Concreteness-Ratings"),
        ("thuocl", "THUOCL", "external_references/routing_vocab/THUOCL"),
        ("word_importance", "Word Importance", "external_references/routing_vocab/missing-word-importance"),
        ("wordfreq", "wordfreq", "external_references/routing_vocab/wordfreq"),
        ("textstat", "textstat", "external_references/routing_vocab/textstat"),
        ("empath", "Empath", "external_references/routing_vocab/empath-client"),
        ("vader", "VADER", "external_references/routing_vocab/vaderSentiment"),
        ("textblob", "TextBlob", "external_references/routing_vocab/textblob"),
        ("yake", "YAKE", "external_references/routing_vocab/yake"),
        ("english_wordnet", "WordNet", "external_references/routing_vocab/english-wordnet"),
        ("omw", "OMW", "external_references/routing_vocab/OMW"),
        ("openhownet", "OpenHowNet", "external_references/routing_vocab/OpenHowNet"),
        ("nrc", "NRC", "external_references/routing_vocab/NRC"),
        ("stopwords_iso", "stopwords-iso", "external_references/routing_vocab/stopwords-iso"),
        ("chinese_stopwords_goto456", "Chinese stopwords", "external_references/routing_vocab/chinese-stopwords-goto456"),
        ("en_dimlex", "en_dimlex", "external_references/routing_vocab/en_dimlex"),
        ("chinese_dimlex", "Chinese DiMLex", "external_references/routing_vocab/chinese-dimlex"),
    ]
    with inventory.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "resource_id",
                "resource",
                "local_path",
                "license_status",
                "feature_family",
                "v021_use_note",
            ]
        )
        for resource_id, resource, local_path in rows:
            writer.writerow([resource_id, resource, local_path, "test", "test_feature", "test_use"])
    return inventory


class RoutingV021FeatureMatrixBuilderTests(unittest.TestCase):
    def test_run_writes_matrix_outputs_and_preserves_inputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = make_resource_inventory(root)
            workspace = root / "users" / "matrix_fixture"
            text_units = workspace / "raw" / "organization" / "text_units.jsonl"
            route_decisions = workspace / "routing" / "prebuild_s1" / "route_decisions.jsonl"
            write_jsonl(
                text_units,
                [
                    {
                        "text_unit_id": "tu1",
                        "unit_type": "sentence",
                        "workspace_id": "matrix_fixture",
                        "text": "The Prototype project connects Alice to a complex deadline because the evidence is fragmented.",
                    },
                    {
                        "text_unit_id": "tu2",
                        "unit_type": "sentence",
                        "workspace_id": "matrix_fixture",
                        "text": "\u56e0\u4e3a\u8fd9\u4e2a\u9879\u76ee\u539f\u578b\u9700\u8981\u590d\u76d8\uff0c\u6240\u4ee5\u9700\u8981\u8bb0\u5f55\u3002",
                    },
                ],
            )
            write_jsonl(
                route_decisions,
                [
                    {
                        "text_unit_id": "tu1",
                        "task_routes": [{"target_task": "s1_memory_candidate", "recommended_route": "weak_llm_proposal"}],
                    }
                ],
            )
            before_text_units = text_units.read_text(encoding="utf-8")
            before_routes = route_decisions.read_text(encoding="utf-8")

            outputs = run(
                type(
                    "Args",
                    (),
                    {
                        "project_root": str(root),
                        "workspace": str(workspace),
                        "resource_inventory": str(inventory),
                        "output_dir": None,
                        "max_items": 0,
                    },
                )()
            )

            self.assertEqual(text_units.read_text(encoding="utf-8"), before_text_units)
            self.assertEqual(route_decisions.read_text(encoding="utf-8"), before_routes)
            for path in outputs.values():
                self.assertTrue(Path(path).exists(), path)

            rows = [json.loads(line) for line in Path(outputs["matrix_jsonl"]).read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            by_id = {row["unit_id"]: row for row in rows}
            self.assertEqual(by_id["tu1"]["route_snapshot"]["s1_memory_candidate"], "weak_llm_proposal")
            self.assertGreater(by_id["tu1"]["axis_scores"]["lexical_complexity_score"], 0)
            self.assertEqual(by_id["tu1"]["feature_status"]["word_importance"], "missing_resource")
            self.assertEqual(by_id["tu1"]["feature_status"]["stopwords_iso"], "available")
            self.assertEqual(by_id["tu1"]["feature_status"]["en_dimlex"], "available")
            self.assertEqual(by_id["tu1"]["raw_features"]["sentence_complexity"]["discourse_marker_count"], 1)
            self.assertEqual(by_id["tu2"]["feature_status"]["chinese_dimlex"], "available")
            self.assertGreaterEqual(by_id["tu2"]["raw_features"]["sentence_complexity"]["discourse_marker_count"], 1)
            self.assertGreaterEqual(by_id["tu2"]["raw_features"]["thuocl"]["hit_count"], 1)
            self.assertGreater(by_id["tu2"]["axis_scores"]["domain_term_score"], 0)
            self.assertEqual(by_id["tu2"]["feature_status"]["chinese_stopwords_goto456"], "available")
            self.assertFalse(by_id["tu1"]["write_permission"])

    def test_low_information_short_text_is_reported_as_low_value_without_route_replacement(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inventory = make_resource_inventory(root)
            workspace = root / "users" / "short_fixture"
            write_jsonl(
                workspace / "raw" / "organization" / "text_units.jsonl",
                [{"text_unit_id": "short1", "unit_type": "sentence", "text": "OK."}],
            )

            outputs = run(
                type(
                    "Args",
                    (),
                    {
                        "project_root": str(root),
                        "workspace": str(workspace),
                        "resource_inventory": str(inventory),
                        "output_dir": None,
                        "max_items": 0,
                    },
                )()
            )

            row = json.loads(Path(outputs["matrix_jsonl"]).read_text(encoding="utf-8").splitlines()[0])
            self.assertGreaterEqual(row["axis_scores"]["low_value_score"], 5.0)
            self.assertEqual(row["route_snapshot"], {})
            manifest = json.loads(Path(outputs["manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["provider_calls"])
            self.assertFalse(manifest["route_decisions_replaced"])


if __name__ == "__main__":
    unittest.main()
