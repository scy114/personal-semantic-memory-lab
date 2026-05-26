import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.graph.graph_construction_packet_builder import write_json
from tools.graph.relation_schema_embedding_index_builder import build_index
from tools.graph.relation_schema_embedding_retriever import run_embedding_retriever


class RelationSchemaEmbeddingRetrievalTests(unittest.TestCase):
    def test_gap_review_stays_supplemental_not_canonical(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            policy_path = root / "policy.json"
            external_root = root / "external_references"
            gap_path = root / "gap_rows.jsonl"
            output_dir = root / "index"
            policy_path.write_text(
                json.dumps(
                    {
                        "canonical_relation_types": {
                            "place_of_death": {"category": "external_wikidata_property"},
                            "mourned_over": {"category": "external_arf_fiction"},
                        },
                        "alias_map": {},
                    }
                ),
                encoding="utf-8",
            )
            gap_path.write_text(
                json.dumps(
                    {
                        "packet_id": "packet:1",
                        "gap_cluster_hint": "loss_or_grief",
                        "description": "grieves the loss of",
                        "quote": "Mira lost her mother.",
                        "source": "Mira",
                        "target": "mother",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            manifest = build_index(
                relation_policy=policy_path,
                external_root=external_root,
                output_dir=output_dir,
                embedding_model="test",
                gap_review_path=gap_path,
                skip_embeddings=True,
            )

            docs = [
                json.loads(line)
                for line in (output_dir / "relation_schema_embedding_docs.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            place_doc = next(doc for doc in docs if doc["relation_type"] == "place_of_death")
            gap_docs = [doc for doc in docs if doc["doc_kind"] == "supplemental_relation_gap_hint"]
            self.assertEqual(manifest["counts"]["supplemental_relation_gap_hint_doc_count"], 1)
            self.assertNotIn("grieves the loss of", place_doc["semantic_text"])
            self.assertEqual(len(gap_docs), 1)
            self.assertIn("do_not_use_as_provider_relation_type", gap_docs[0]["warnings"])

    def test_embedding_retriever_returns_canonical_and_hint_groups(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace = root / "workspace"
            index_dir = root / "index"
            output_dir = workspace / "out"
            packets_path = workspace / "graph_construction_packets.jsonl"
            workspace.mkdir(parents=True)
            index_dir.mkdir(parents=True)
            docs = [
                {
                    "relation_type": "father",
                    "doc_kind": "canonical_relation_type",
                    "category": "external_wikidata_property",
                    "semantic_text": "father parent 父亲",
                    "aliases": ["father", "父亲"],
                },
                {
                    "relation_type": "schema_gap_hint_1",
                    "doc_kind": "supplemental_relation_gap_hint",
                    "category": "observed_low_schema_gap",
                    "semantic_text": "family relation gap",
                    "warnings": ["do_not_use_as_provider_relation_type"],
                },
            ]
            (index_dir / "relation_schema_embedding_docs.jsonl").write_text(
                "".join(json.dumps(doc, ensure_ascii=False) + "\n" for doc in docs),
                encoding="utf-8",
            )
            np.save(index_dir / "relation_schema_embedding_matrix.npy", np.array([[1.0, 0.0], [0.8, 0.2]], dtype="float32"))
            write_json(
                index_dir / "relation_schema_embedding_manifest.json",
                {"embedding_model": "test-model", "counts": {"vector_dim": 2}},
            )
            packets_path.write_text(
                json.dumps({"packet_id": "packet:1", "workspace_id": "workspace", "original_text": "A is B's father."})
                + "\n",
                encoding="utf-8",
            )

            with patch(
                "tools.graph.relation_schema_embedding_retriever.encode_texts",
                return_value=np.array([[1.0, 0.0]], dtype="float32"),
            ):
                manifest = run_embedding_retriever(
                    workspace=workspace,
                    input_packets=packets_path,
                    index_dir=index_dir,
                    output_dir=output_dir,
                    top_k=2,
                    canonical_top_k=1,
                )

            row = json.loads((output_dir / "relation_schema_embedding_candidates.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(manifest["counts"]["packet_count"], 1)
            self.assertEqual(row["canonical_relation_schema_candidates"][0]["relation_type"], "father")
            self.assertEqual(row["relation_schema_candidates"][0]["relation_type"], "father")
            self.assertEqual(row["relation_schema_embedding_candidates"][1]["doc_kind"], "supplemental_relation_gap_hint")
            self.assertTrue(manifest["boundary"]["graph_is_not_proof"])


if __name__ == "__main__":
    unittest.main()
