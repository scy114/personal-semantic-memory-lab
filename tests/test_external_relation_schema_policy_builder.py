import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.external_relation_schema_policy_builder import build_generated_policy


class ExternalRelationSchemaPolicyBuilderTests(unittest.TestCase):
    def test_builds_policy_from_external_schema_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "external_references" / "graph_relation_schemas"
            (root / "arf").mkdir(parents=True)
            (root / "dialogre").mkdir(parents=True)
            (root / "conceptnet" / "conceptnet5.wiki").mkdir(parents=True)
            (root / "wikidata").mkdir(parents=True)
            (root / "dbpedia").mkdir(parents=True)

            (root / "arf" / "arf_relation_types.txt").write_text("child_of\ntravel_to\n", encoding="utf-8")
            (root / "arf" / "arf_relation_schema_summary.json").write_text(
                json.dumps({"top_relation_types": [["child_of", 7], ["travel_to", 3]]}),
                encoding="utf-8",
            )
            (root / "dialogre" / "dialogre_relation_schema_summary.json").write_text(
                json.dumps({"relation_label_counts": [["per:spouse", 5], ["unanswerable", 100]]}),
                encoding="utf-8",
            )
            (root / "dialogre" / "dialogre_relation_types.txt").write_text("per:spouse\n", encoding="utf-8")
            (root / "conceptnet" / "conceptnet5.wiki" / "Relations.md").write_text(
                "| /r/PartOf | part-whole relation |\n| /r/RelatedTo | related relation |\n",
                encoding="utf-8",
            )
            (root / "wikidata" / "P26.json").write_text(
                json.dumps(
                    {
                        "entities": {
                            "P26": {
                                "labels": {"en": {"value": "spouse"}},
                                "aliases": {"en": [{"value": "married to"}]},
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "dbpedia" / "dbpedia_ontology_tbox.owl").write_text(
                "http://dbpedia.org/ontology/birthPlace",
                encoding="utf-8",
            )

            policy = build_generated_policy(root)

            self.assertEqual(policy["status"], "generated_from_downloaded_external_resources")
            self.assertTrue(policy["normalization_strategy"]["generated_from_external_resources_only"])
            self.assertFalse(policy["normalization_strategy"]["handwritten_project_relation_wordlist"])
            self.assertIn("child_of", policy["canonical_relation_types"])
            self.assertIn("spouse", policy["canonical_relation_types"])
            self.assertIn("part_of", policy["canonical_relation_types"])
            self.assertIn("birth_place", policy["canonical_relation_types"])
            self.assertEqual(policy["alias_map"]["child-of"], "child_of")
            self.assertEqual(policy["alias_map"]["per:spouse"], "spouse")
            self.assertEqual(policy["alias_map"]["/r/partof"], "part_of")
            self.assertEqual(policy["alias_map"]["married-to"], "spouse")
            self.assertNotIn("about", policy["alias_map"])
            self.assertGreaterEqual(len(policy["provenance"]["source_hashes"]), 5)


if __name__ == "__main__":
    unittest.main()
