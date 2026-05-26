import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.step1.step1_build_runner import run_build
from tools.step1.step1_index_runner import run_index


ROOT = Path(__file__).resolve().parents[1]


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


class Step1IndexRunnerTests(unittest.TestCase):
    def build_workspace(self, temp_dir: str, run_scope: str = "full_s1_build") -> Path:
        output_workspace = Path(temp_dir) / f"fixture_{run_scope}"
        run_build(
            Namespace(
                project_root=str(ROOT),
                workspace="tests/fixtures/step1_toolbox",
                output_workspace=str(output_workspace),
                workspace_id=f"fixture_{run_scope}",
                modeled_subject_id="Jon",
                run_id=f"s1-build:test:index:{run_scope}",
                run_scope=run_scope,
                duplicate_policy="fail",
                **deterministic_prebuild_args(),
            )
        )
        return output_workspace

    def run_index_for(self, workspace: Path, run_scope: str, index_mode: str = "lexical_and_embedding") -> dict:
        return run_index(
            Namespace(
                project_root=str(ROOT),
                workspace=str(workspace),
                workspace_id=workspace.name,
                run_id=f"s1-index:test:{workspace.name}:{run_scope}:{index_mode}",
                run_scope=run_scope,
                index_mode=index_mode,
                output_path=str(workspace / "indexes"),
                duplicate_policy="fail",
                embedding_backend="hash",
                embedding_model="deterministic-hash-embedding-v0.1",
                embedding_dimension=32,
                embedding_batch_size=16,
                embedding_device=None,
                allow_non_lcoral_for_tests=True,
            )
        )

    def test_evidence_index_only_builds_bm25_and_embedding(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_workspace(temp_dir, "full_s1_build")
            result = self.run_index_for(workspace, "evidence_index_only")

            bm25_manifest = json.loads((workspace / "indexes" / "step1_bm25_manifest.json").read_text(encoding="utf-8"))
            embedding_manifest = json.loads(
                (workspace / "indexes" / "step1_embedding_manifest.json").read_text(encoding="utf-8")
            )
            bm25_entries = jsonl_rows(workspace / "indexes" / "step1_bm25_entries.jsonl")
            embedding_entries = jsonl_rows(workspace / "indexes" / "step1_embedding_entries.jsonl")
            vectors = json.loads((workspace / "indexes" / "step1_embedding_vectors.json").read_text(encoding="utf-8"))

            self.assertEqual(result["index_units"], 2)
            self.assertEqual({row["source_object_type"] for row in bm25_entries}, {"raw_evidence"})
            self.assertEqual({row["source_object_type"] for row in embedding_entries}, {"raw_evidence"})
            self.assertEqual(bm25_manifest["run_scope"], "evidence_index_only")
            self.assertEqual(embedding_manifest["embedding_dimension"], 32)
            self.assertEqual(vectors["vector_count"], len(embedding_entries))
            self.assertEqual(vectors["index_entry_ids"], [row["index_entry_id"] for row in embedding_entries])
            self.assertTrue(all(row["raw_source_id"] for row in embedding_entries))
            self.assertTrue(all(row.get("locator") or row.get("locator_unavailable_reason") for row in embedding_entries))

    def test_evidence_plus_memory_excludes_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_workspace(temp_dir, "full_s1_build")
            self.run_index_for(workspace, "evidence_plus_memory_index")

            entries = jsonl_rows(workspace / "indexes" / "step1_bm25_entries.jsonl")
            object_types = {row["source_object_type"] for row in entries}
            self.assertEqual(object_types, {"raw_evidence", "memory_unit"})
            self.assertNotIn("doc_level_summary", object_types)

    def test_full_s1_index_marks_summary_hits_context_only(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_workspace(temp_dir, "full_s1_build")
            self.run_index_for(workspace, "full_s1_index")

            entries = jsonl_rows(workspace / "indexes" / "step1_embedding_entries.jsonl")
            summary_entries = [row for row in entries if row["source_object_type"] == "doc_level_summary"]
            self.assertEqual(len(summary_entries), 1)
            self.assertEqual(summary_entries[0]["item_layer"], "doc_level_summary")
            self.assertIn("summary_hit_context_only", summary_entries[0]["warnings"])
            self.assertIn("compressed_context_only", summary_entries[0]["truth_status"])

    def test_index_runner_does_not_mutate_canonical_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_workspace(temp_dir, "full_s1_build")
            evidence_before = (workspace / "evidence" / "evidence.jsonl").read_text(encoding="utf-8")
            memory_before = (workspace / "memory" / "memory_units.jsonl").read_text(encoding="utf-8")
            summary_before = (workspace / "memory" / "summaries.jsonl").read_text(encoding="utf-8")

            self.run_index_for(workspace, "full_s1_index")

            self.assertEqual(evidence_before, (workspace / "evidence" / "evidence.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(memory_before, (workspace / "memory" / "memory_units.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(summary_before, (workspace / "memory" / "summaries.jsonl").read_text(encoding="utf-8"))
            report = (workspace / "indexes" / "step1_index_build_report.md").read_text(encoding="utf-8")
            self.assertIn("canonical_assets_unchanged: `True`", report)

    def test_duplicate_policy_fail_blocks_existing_indexes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = self.build_workspace(temp_dir, "full_s1_build")
            self.run_index_for(workspace, "evidence_index_only")
            with self.assertRaisesRegex(FileExistsError, "Existing S1 index assets"):
                self.run_index_for(workspace, "evidence_index_only")


if __name__ == "__main__":
    unittest.main()
