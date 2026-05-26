import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.relation_schema_candidate_retriever import run_retriever


class RelationSchemaCandidateRetrieverTests(unittest.TestCase):
    def test_retrieves_packet_specific_relation_candidates_from_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            packets_path = workspace / "graph_construction_packets.jsonl"
            policy_path = root / "relation_policy.json"
            output_dir = workspace / "relation_schema_candidates"
            workspace.mkdir(parents=True)

            policy = {
                "schema_version": "test",
                "policy_id": "test",
                "canonical_relation_types": {
                    "child_of": {
                        "category": "external_arf_fiction",
                        "external_sources": ["arf"],
                        "source_counts": {"arf": 100},
                    },
                    "spouse": {
                        "category": "external_dialogue",
                        "external_sources": ["dialogre"],
                        "source_counts": {"dialogre": 20},
                    },
                    "located_in": {
                        "category": "external_arf_fiction",
                        "external_sources": ["arf"],
                        "source_counts": {"arf": 10},
                    },
                },
                "alias_map": {
                    "child-of": "child_of",
                    "child of": "child_of",
                    "mother": "child_of",
                    "spouse": "spouse",
                    "wife": "spouse",
                    "located-in": "located_in",
                },
            }
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            packets_path.write_text(
                json.dumps(
                    {
                        "packet_id": "packet:1",
                        "workspace_id": "workspace",
                        "original_text": "Mira lost her mother when she was fifteen.",
                        "graph_extraction_text": "Mira lost her mother when she was fifteen.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = run_retriever(
                workspace=workspace,
                input_packets=packets_path,
                relation_policy=policy_path,
                output_dir=output_dir,
                top_k=2,
            )

            rows = [
                json.loads(line)
                for line in (output_dir / "relation_schema_candidates.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            relation_types = [row["relation_type"] for row in rows[0]["relation_schema_candidates"]]
            self.assertEqual(manifest["counts"]["packet_count"], 1)
            self.assertEqual(manifest["boundary"]["provider_calls"], False)
            self.assertIn("child_of", relation_types)
            self.assertTrue((output_dir / "relation_schema_candidate_retrieval_report.md").exists())

    def test_does_not_rank_by_stopwords_or_alias_substrings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            packets_path = workspace / "graph_construction_packets.jsonl"
            policy_path = root / "relation_policy.json"
            output_dir = workspace / "relation_schema_candidates"
            workspace.mkdir(parents=True)

            policy = {
                "schema_version": "test",
                "policy_id": "test",
                "canonical_relation_types": {
                    "sibling": {"external_sources": ["wikidata"], "source_counts": {"wikidata": 1}},
                    "mother": {"external_sources": ["wikidata"], "source_counts": {"wikidata": 1}},
                    "has_part_s": {"external_sources": ["wikidata"], "source_counts": {"wikidata": 1}},
                },
                "alias_map": {
                    "brother or sister": "sibling",
                    "mother": "mother",
                    "mom": "mother",
                    "composed of": "has_part_s",
                },
            }
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            packets_path.write_text(
                json.dumps(
                    {
                        "packet_id": "packet:1",
                        "workspace_id": "workspace",
                        "original_text": "Her last moments were quiet; no family relation is named here.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            run_retriever(
                workspace=workspace,
                input_packets=packets_path,
                relation_policy=policy_path,
                output_dir=output_dir,
                top_k=3,
            )

            row = json.loads((output_dir / "relation_schema_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
            by_type = {candidate["relation_type"]: candidate for candidate in row["relation_schema_candidates"]}
            self.assertNotIn("mom", by_type.get("mother", {}).get("retrieval_features", {}).get("exact_alias_hits", []))
            self.assertNotIn("has_part_s", by_type)


if __name__ == "__main__":
    unittest.main()
