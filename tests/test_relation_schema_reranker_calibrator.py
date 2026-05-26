import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_jsonl, write_jsonl
from tools.graph.relation_schema_reranker_calibrator import run_calibration


class RelationSchemaRerankerCalibratorTests(unittest.TestCase):
    def test_trains_and_reranks_from_proxy_provider_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            candidates_path = root / "relation_schema_candidates.jsonl"
            extraction_dir = root / "extraction"
            output_dir = root / "reranker"
            extraction_dir.mkdir(parents=True)

            candidate_rows = [
                {
                    "packet_id": "packet:1",
                    "relation_schema_candidates": [
                        {
                            "rank": 1,
                            "relation_type": "sibling",
                            "score": 1.0,
                            "source_count": 0,
                            "category": "external_wikidata_property",
                            "external_sources": ["wikidata"],
                            "aliases": ["brother"],
                            "retrieval_features": {
                                "lexical_score": 1.0,
                                "exact_alias_hits": [],
                                "matched_tokens": ["brother"],
                            },
                        },
                        {
                            "rank": 2,
                            "relation_type": "mother",
                            "score": 20.0,
                            "source_count": 5,
                            "category": "external_wikidata_property",
                            "external_sources": ["wikidata"],
                            "aliases": ["mother"],
                            "retrieval_features": {
                                "lexical_score": 20.0,
                                "exact_alias_hits": ["mother"],
                                "matched_tokens": ["mother"],
                            },
                        },
                    ],
                    "graph_is_not_proof": True,
                },
                {
                    "packet_id": "packet:2",
                    "relation_schema_candidates": [
                        {
                            "rank": 1,
                            "relation_type": "residence",
                            "score": 2.0,
                            "source_count": 0,
                            "category": "external_wikidata_property",
                            "external_sources": ["wikidata"],
                            "aliases": ["house"],
                            "retrieval_features": {
                                "lexical_score": 2.0,
                                "exact_alias_hits": [],
                                "matched_tokens": ["house"],
                            },
                        },
                        {
                            "rank": 2,
                            "relation_type": "father",
                            "score": 19.0,
                            "source_count": 4,
                            "category": "external_wikidata_property",
                            "external_sources": ["wikidata"],
                            "aliases": ["father"],
                            "retrieval_features": {
                                "lexical_score": 19.0,
                                "exact_alias_hits": ["father"],
                                "matched_tokens": ["father"],
                            },
                        },
                    ],
                    "graph_is_not_proof": True,
                },
            ]
            write_jsonl(candidates_path, candidate_rows)
            write_jsonl(
                extraction_dir / "graph_relation_candidates.jsonl",
                [
                    {
                        "source_packet_id": "packet:1",
                        "relation_type_hint": "mother",
                        "relation_schema_status": "selected_from_retrieved_schema",
                    },
                    {
                        "source_packet_id": "packet:2",
                        "relation_type_hint": "father",
                        "relation_schema_status": "selected_from_retrieved_schema",
                    },
                ],
            )

            manifest = run_calibration(
                relation_schema_candidates=candidates_path,
                extraction_dirs=[extraction_dir],
                output_dir=output_dir,
            )

            reranked = read_jsonl(output_dir / "relation_schema_candidates.reranked.jsonl")
            first_packet_relations = [row["relation_type"] for row in reranked[0]["relation_schema_candidates"]]
            feature_rows = read_jsonl(output_dir / "relation_schema_reranker_feature_rows.jsonl")
            labels = [row["label"] for row in feature_rows]

            self.assertEqual(manifest["boundary"]["graph_truth"], False)
            self.assertTrue((output_dir / "relation_schema_reranker.joblib").exists())
            self.assertIn(1, labels)
            self.assertIn(0, labels)
            self.assertEqual(first_packet_relations[0], "mother")
            self.assertTrue(reranked[0]["graph_is_not_proof"])


if __name__ == "__main__":
    unittest.main()
