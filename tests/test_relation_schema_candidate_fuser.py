import json
import tempfile
import unittest
from pathlib import Path

from tools.graph.relation_schema_candidate_fuser import run_fuser


class RelationSchemaCandidateFuserTests(unittest.TestCase):
    def test_fuses_lexical_and_embedding_candidates_with_rrf_cap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            workspace.mkdir()
            lexical = workspace / "lexical.jsonl"
            embedding = workspace / "embedding.jsonl"
            output_dir = workspace / "fused"

            lexical.write_text(
                json.dumps(
                    {
                        "packet_id": "packet:1",
                        "workspace_id": "workspace",
                        "relation_schema_candidates": [
                            {"rank": 1, "relation_type": "suggests", "score": 9.0, "aliases": ["suggests"]},
                            {"rank": 2, "relation_type": "mother", "score": 3.0, "aliases": ["mother"]},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            embedding.write_text(
                json.dumps(
                    {
                        "packet_id": "packet:1",
                        "workspace_id": "workspace",
                        "relation_schema_candidates": [
                            {"rank": 1, "relation_type": "mother", "score": 0.7, "doc_kind": "canonical_relation_type"},
                            {"rank": 2, "relation_type": "schema_gap_hint_x", "score": 0.8, "doc_kind": "supplemental_relation_gap_hint"},
                            {"rank": 3, "relation_type": "mourned_over", "score": 0.6, "doc_kind": "canonical_relation_type"},
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = run_fuser(
                workspace=workspace,
                lexical_candidates=lexical,
                embedding_candidates=embedding,
                output_dir=output_dir,
                max_candidates=2,
                rrf_k=60,
            )

            row = json.loads((output_dir / "relation_schema_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
            relation_types = [candidate["relation_type"] for candidate in row["relation_schema_candidates"]]
            self.assertEqual(manifest["counts"]["candidate_rows"], 2)
            self.assertIn("mother", relation_types)
            self.assertNotIn("suggests", relation_types)
            self.assertNotIn("schema_gap_hint_x", relation_types)
            mother = next(candidate for candidate in row["relation_schema_candidates"] if candidate["relation_type"] == "mother")
            self.assertEqual(mother["retrieval_features"]["fusion_sources"], ["embedding", "lexical"])
            self.assertTrue(manifest["boundary"]["graph_is_not_proof"])


if __name__ == "__main__":
    unittest.main()
