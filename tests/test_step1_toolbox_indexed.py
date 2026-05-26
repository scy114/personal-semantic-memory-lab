import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.step1.step1_build_runner import run_build
from tools.step1.step1_index_runner import run_index
from tools.step1.step1_toolbox import LocalStep1Toolbox


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def deterministic_prebuild_args() -> dict:
    return {
        "pre_build_route_mode": "none",
        "route_policy": None,
        "proposal_provider": "mock",
        "proposal_profile": None,
        "api_mode": "responses",
        "allow_live_api": False,
        "external_model_outputs": None,
        "max_items": None,
        "item_offset": 0,
        "sample_stride": 1,
    }


class Step1ToolboxIndexedTests(unittest.TestCase):
    def build_and_index(self, temp_dir: str, run_scope: str = "full_s1_build") -> Path:
        workspace = Path(temp_dir) / "fixture_indexed"
        run_build(
            Namespace(
                project_root=str(ROOT),
                workspace="tests/fixtures/step1_toolbox",
                output_workspace=str(workspace),
                workspace_id="fixture_indexed",
                modeled_subject_id="Jon",
                run_id=f"s1-build:test:toolbox-indexed:{run_scope}",
                run_scope=run_scope,
                duplicate_policy="fail",
                **deterministic_prebuild_args(),
            )
        )
        run_index(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                workspace_id="fixture_indexed",
                run_id=f"s1-index:test:toolbox-indexed:{run_scope}",
                run_scope="full_s1_index",
                index_mode="lexical_and_embedding",
                output_path=str(workspace / "indexes"),
                duplicate_policy="fail",
                embedding_backend="hash",
                embedding_model="deterministic-hash-embedding-v0.1",
                embedding_dimension=3,
                embedding_batch_size=16,
                embedding_device=None,
                allow_non_lcoral_for_tests=True,
            )
        )
        return workspace

    def test_bm25_indexed_exact_query_finds_evidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_and_index(temp_dir)
            toolbox = LocalStep1Toolbox(workspace)

            results = toolbox.retrieve_evidence(
                "grant proposal",
                filters={"retrieval_mode": "bm25_indexed"},
                top_k=3,
                route_policy={"route": "exact", "target_subject_id": "Jon"},
            )

            self.assertTrue(results)
            self.assertEqual(results[0]["retrieval_mode"], "bm25_indexed")
            self.assertEqual(results[0]["retrieval_status"], "success")
            self.assertEqual(results[0]["retrieval_backend"], "bm25")
            self.assertEqual(results[0]["support_status"], "not_checked")
            self.assertIn("evidence:fixture_indexed:turn_001", results[0]["evidence_refs"])
            self.assertIn("retrieval_hit_not_support_check", results[0]["warnings"])
            self.assertEqual(results[0]["subject_match_status"], "target_match")
            self.assertEqual(results[0]["subject_risk_warning"], "")
            self.assertIn("bm25_raw_score", results[0]["score_components"])
            self.assertIn("bm25_normalized_score", results[0]["score_components"])
            self.assertEqual(results[0]["score_components"]["normalization_scope"], "current_hit_set")

    def test_stale_index_degrades_to_direct_scan(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_and_index(temp_dir)
            evidence_path = workspace / "evidence" / "evidence.jsonl"
            evidence_path.write_text(evidence_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            toolbox = LocalStep1Toolbox(workspace)

            results = toolbox.retrieve_evidence(
                "grant proposal",
                filters={"retrieval_mode": "bm25_indexed"},
                top_k=3,
            )

            self.assertTrue(results)
            self.assertEqual(results[0]["retrieval_mode"], "degraded_direct_scan")
            self.assertIn("degraded_direct_scan", results[0]["warnings"])
            self.assertTrue(any(warning.startswith("stale_index_source_hash") for warning in results[0]["warnings"]))

    def test_hash_embedding_backend_is_not_semantic_backend(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_and_index(temp_dir)
            toolbox = LocalStep1Toolbox(workspace)

            results = toolbox.retrieve_evidence(
                "grant proposal",
                filters={"retrieval_mode": "embedding_indexed"},
                top_k=3,
            )

            self.assertTrue(results)
            self.assertEqual(results[0]["retrieval_mode"], "degraded_direct_scan")
            self.assertIn("hash_embedding_backend_test_only", results[0]["warnings"])

    def test_embedding_indexed_uses_query_vector_and_preserves_refs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_and_index(temp_dir)
            index_root = workspace / "indexes"
            manifest = read_json(index_root / "step1_embedding_manifest.json")
            sidecar = read_json(index_root / "step1_embedding_vectors.json")
            entries = jsonl_rows(index_root / "step1_embedding_entries.jsonl")

            manifest["embedding_backend"] = "qwen_local"
            manifest["embedding_model"] = "synthetic-qwen-compatible-test"
            manifest["embedding_dimension"] = 3
            sidecar["embedding_backend"] = "qwen_local"
            sidecar["embedding_model"] = "synthetic-qwen-compatible-test"
            sidecar["embedding_dimension"] = 3
            vectors = []
            for entry in entries:
                if "evidence:fixture_indexed:turn_001" in entry.get("evidence_refs", []):
                    vectors.append([1.0, 0.0, 0.0])
                else:
                    vectors.append([0.0, 1.0, 0.0])
            sidecar["vectors"] = vectors
            write_json(index_root / "step1_embedding_manifest.json", manifest)
            write_json(index_root / "step1_embedding_vectors.json", sidecar)

            toolbox = LocalStep1Toolbox(workspace)
            results = toolbox.retrieve_evidence(
                "work focus",
                filters={"retrieval_mode": "embedding_indexed"},
                top_k=3,
                route_policy={"query_vector": [1.0, 0.0, 0.0]},
            )

            self.assertTrue(results)
            self.assertEqual(results[0]["retrieval_mode"], "embedding_indexed")
            self.assertEqual(results[0]["retrieval_backend"], "embedding")
            self.assertEqual(results[0]["support_status"], "not_checked")
            self.assertIn("evidence:fixture_indexed:turn_001", results[0]["evidence_refs"])
            self.assertIn("retrieval_hit_not_support_check", results[0]["warnings"])
            self.assertIn("embedding_raw_score", results[0]["score_components"])
            self.assertIn("embedding_normalized_score", results[0]["score_components"])
            self.assertEqual(results[0]["embedding_runtime"]["embedding_backend"], "qwen_local")
            self.assertEqual(results[0]["embedding_runtime"]["model_load_behavior"], "query_vector_supplied_no_model_load")

    def test_hybrid_indexed_preserves_memory_backpointers_and_summary_policy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_and_index(temp_dir)
            toolbox = LocalStep1Toolbox(workspace)

            memory_results = toolbox.retrieve_evidence(
                "poster grant",
                filters={"retrieval_mode": "hybrid_indexed", "include_memory_units": True},
                top_k=10,
                route_policy={"allow_hash_embedding_for_tests": True, "query_vector": [1.0, 0.0, 0.0]},
            )
            memory_hits = [result for result in memory_results if result["source_object_type"] == "memory_unit"]
            self.assertTrue(memory_hits)
            self.assertEqual(memory_hits[0]["retrieval_mode"], "hybrid_indexed")
            self.assertIn("fusion_weights", memory_hits[0]["score_components"])
            self.assertIn("final_score", memory_hits[0]["score_components"])
            self.assertTrue(memory_hits[0]["active_backends"])
            self.assertTrue(memory_hits[0]["evidence_refs"])
            self.assertTrue(memory_hits[0]["backpointer_refs"])
            self.assertIn("memory_unit_not_raw_evidence", memory_hits[0]["warnings"])

            summary_results = toolbox.retrieve_evidence(
                "session grant proposal",
                filters={"retrieval_mode": "bm25_indexed", "include_summaries": True, "item_layers": ["doc_level_summary"]},
                top_k=10,
            )
            summary_hits = [result for result in summary_results if result["source_object_type"] == "doc_level_summary"]
            self.assertTrue(summary_hits)
            self.assertIn("compressed_context_only", summary_hits[0]["truth_status"])
            self.assertIn("summary_hit_context_only", summary_hits[0]["warnings"])

    def test_auto_mode_selects_fresh_hybrid(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_and_index(temp_dir)
            index_root = workspace / "indexes"
            manifest = read_json(index_root / "step1_embedding_manifest.json")
            sidecar = read_json(index_root / "step1_embedding_vectors.json")
            manifest["embedding_backend"] = "qwen_local"
            sidecar["embedding_backend"] = "qwen_local"
            sidecar["vectors"] = [[1.0, 0.0, 0.0] for _ in sidecar["vectors"]]
            write_json(index_root / "step1_embedding_manifest.json", manifest)
            write_json(index_root / "step1_embedding_vectors.json", sidecar)

            toolbox = LocalStep1Toolbox(workspace)
            results = toolbox.retrieve_evidence(
                "grant proposal",
                filters={"retrieval_mode": "auto"},
                top_k=3,
                route_policy={"query_vector": [1.0, 0.0, 0.0]},
            )

            self.assertTrue(results)
            self.assertEqual(results[0]["requested_retrieval_mode"], "auto")
            self.assertEqual(results[0]["retrieval_mode"], "hybrid_indexed")
            self.assertIn("bm25", results[0]["active_backends"])
            self.assertIn("embedding", results[0]["active_backends"])
            self.assertEqual(results[0]["missing_backends"], [])
            self.assertEqual(results[0]["hybrid_status"], "full")
            self.assertEqual(results[0]["support_status"], "not_checked")

    def test_hybrid_indexed_reports_partial_when_one_backend_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_and_index(temp_dir)
            index_root = workspace / "indexes"
            manifest = read_json(index_root / "step1_embedding_manifest.json")
            sidecar = read_json(index_root / "step1_embedding_vectors.json")
            manifest["embedding_backend"] = "qwen_local"
            sidecar["embedding_backend"] = "qwen_local"
            sidecar["vectors"] = [[1.0, 0.0, 0.0] for _ in sidecar["vectors"]]
            write_json(index_root / "step1_embedding_manifest.json", manifest)
            write_json(index_root / "step1_embedding_vectors.json", sidecar)
            for path in (
                index_root / "step1_bm25_manifest.json",
                index_root / "step1_bm25_entries.jsonl",
                index_root / "step1_bm25_index.json",
            ):
                path.unlink()

            toolbox = LocalStep1Toolbox(workspace)
            results = toolbox.retrieve_evidence(
                "grant proposal",
                filters={"retrieval_mode": "hybrid_indexed"},
                top_k=3,
                route_policy={"query_vector": [1.0, 0.0, 0.0]},
            )

            self.assertTrue(results)
            self.assertEqual(results[0]["retrieval_mode"], "hybrid_indexed")
            self.assertEqual(results[0]["hybrid_status"], "partial")
            self.assertEqual(results[0]["active_backends"], ["embedding"])
            self.assertIn("bm25", results[0]["missing_backends"])
            self.assertIn("hybrid_bm25_path_unavailable", results[0]["warnings"])

    def test_subject_mismatch_warning_is_exposed_for_indexed_hits(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_and_index(temp_dir)
            entries_path = workspace / "indexes" / "step1_bm25_entries.jsonl"
            entries = jsonl_rows(entries_path)
            rewritten = []
            for entry in entries:
                if "evidence:fixture_indexed:turn_001" in entry.get("evidence_refs", []):
                    entry["locator"] = {
                        "kind": "conversation_turn",
                        "turn_index": "turn_001",
                        "speaker": "Maya",
                        "timestamp": "2026-05-10T10:00:00+08:00",
                    }
                    entry["subject_scope"] = "other_participant"
                rewritten.append(json.dumps(entry, ensure_ascii=False))
            entries_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

            toolbox = LocalStep1Toolbox(workspace)
            results = toolbox.retrieve_evidence(
                "grant proposal",
                filters={"retrieval_mode": "bm25_indexed"},
                top_k=3,
                route_policy={"route": "exact", "target_subject_id": "Jon"},
            )

            self.assertTrue(results)
            self.assertEqual(results[0]["subject_match_status"], "other_participant")
            self.assertEqual(results[0]["subject_risk_warning"], "subject_mismatch_risk")
            self.assertIn("subject_mismatch_risk", results[0]["warnings"])
            self.assertIn("subject_downranked", results[0]["warnings"])
            self.assertLess(results[0]["score_components"]["subject_adjustment_factor"], 1.0)

    def test_subject_aware_ranking_prefers_target_subject_over_other_participant(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "fixture_subject_rank"
            run_build(
                Namespace(
                    project_root=str(ROOT),
                    workspace="tests/fixtures/step1_toolbox",
                    output_workspace=str(workspace),
                    workspace_id="fixture_subject_rank",
                    modeled_subject_id="Jon",
                    run_id="s1-build:test:subject-rank",
                    run_scope="evidence_only",
                    duplicate_policy="fail",
                    **deterministic_prebuild_args(),
                )
            )

            evidence_path = workspace / "evidence" / "evidence.jsonl"
            evidence_rows = jsonl_rows(evidence_path)
            rewritten = []
            for row in evidence_rows:
                row["text"] = "The conversation discussed a grant proposal."
                if row["source_specific_ref"] == "turn_001":
                    row["speaker"] = "Maya"
                    row["subject_ids"] = ["Maya"]
                    row["locator"]["speaker"] = "Maya"
                if row["source_specific_ref"] == "turn_002":
                    row["speaker"] = "Jon"
                    row["subject_ids"] = ["Jon"]
                    row["locator"]["speaker"] = "Jon"
                rewritten.append(json.dumps(row, ensure_ascii=False))
            evidence_path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")

            run_index(
                Namespace(
                    project_root=str(ROOT),
                    workspace=str(workspace),
                    workspace_id="fixture_subject_rank",
                    run_id="s1-index:test:subject-rank",
                    run_scope="evidence_index_only",
                    index_mode="lexical",
                    output_path=str(workspace / "indexes"),
                    duplicate_policy="fail",
                    embedding_backend="hash",
                    embedding_model="deterministic-hash-embedding-v0.1",
                    embedding_dimension=3,
                    embedding_batch_size=16,
                    embedding_device=None,
                    allow_non_lcoral_for_tests=True,
                )
            )

            toolbox = LocalStep1Toolbox(workspace)
            results = toolbox.retrieve_evidence(
                "grant proposal",
                filters={"retrieval_mode": "bm25_indexed"},
                top_k=2,
                route_policy={"route": "exact", "target_subject_id": "Jon"},
            )

            self.assertEqual(len(results), 2)
            self.assertIn("evidence:fixture_subject_rank:turn_002", results[0]["evidence_refs"])
            self.assertEqual(results[0]["subject_match_status"], "target_match")
            self.assertEqual(results[0]["score_components"]["subject_adjustment_factor"], 1.0)
            self.assertIn("evidence:fixture_subject_rank:turn_001", results[1]["evidence_refs"])
            self.assertEqual(results[1]["subject_match_status"], "other_participant")
            self.assertIn("subject_downranked", results[1]["warnings"])
            self.assertLess(results[1]["score"], results[1]["score_components"]["retrieval_score_before_subject_adjustment"])


if __name__ == "__main__":
    unittest.main()
