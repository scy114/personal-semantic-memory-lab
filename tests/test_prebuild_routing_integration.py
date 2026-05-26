import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.step0b.s0b_section_policy_runner import run_section_policy
from tools.step1.step1_build_runner import run_build as run_s1_build
from tools.step2.step2_build_runner import run_build as run_s2_build


ROOT = Path(__file__).resolve().parents[1]
ROUTE_POLICY = "configs/routing/memory_proposal_router/heuristic_salience_v0.21.candidate.yaml"
S1_PROFILE = "configs/proposals/s1_memory_candidate_proposal.v0.1.json"
S2_PROFILE = "configs/proposals/s2_portrait_proposal.v0.2.json"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def create_plaintext_workspace(temp_dir: str) -> Path:
    workspace = Path(temp_dir) / "prebuild_plaintext_fixture"
    raw_dir = workspace / "raw"
    org_dir = raw_dir / "organization"
    org_dir.mkdir(parents=True)
    source_path = raw_dir / "notes.txt"
    source_path.write_text(
        "I lost my job yesterday and plan to start my own studio.\n\n"
        "I reported a privacy risk in the project because the database may crash, and I plan to repair it.\n\n"
        "I prefer careful project notes before I begin a workflow.\n\n"
        "Thanks!",
        encoding="utf-8",
    )
    raw_source = {
        "schema_version": "s0b.raw_source.v0.1",
        "raw_source_id": "raw:generic:prebuild_notes",
        "workspace_id": "prebuild_plaintext_fixture",
        "bundle_id": "bundle:generic:prebuild_notes",
        "source_type": "plain_text",
        "modality": "text",
        "local_path": str(source_path),
        "original_uri_or_path": str(source_path),
        "original_format": "txt",
        "content_hash": "test-hash-prebuild",
        "privacy_class": "public_dataset",
        "organization_degree": "medium",
        "processing_status": "ready_for_s1_intake",
        "inclusion_decision": "include",
        "adapter_recommendation": "generic_text_adapter",
        "coverage_notes": "Plaintext pre-build routing fixture.",
        "perspective_notes": "First-person note.",
        "quality_notes": "Plain text.",
        "source_specific_metadata": {
            "record_id": "prebuild_notes",
            "title": "Prebuild notes",
            "author_subject_id": "Test Subject",
            "modeled_subject_is_author": True,
            "max_segment_chars": 120,
        },
    }
    (org_dir / "raw_sources.jsonl").write_text(json.dumps(raw_source, ensure_ascii=False) + "\n", encoding="utf-8")
    (org_dir / "bundle.json").write_text(
        json.dumps(
            {
                "schema_version": "s0b.bundle.v0.1",
                "bundle_id": "bundle:generic:prebuild_notes",
                "workspace_id": "prebuild_plaintext_fixture",
                "modeled_subject_id": "Test Subject",
                "raw_root": str(raw_dir),
                "source_count": 1,
                "organization_degree": "medium",
                "processing_status": "ready_for_s1_intake",
                "default_adapter_recommendation": "generic_text_adapter",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    run_section_policy(
        Namespace(
            project_root=str(ROOT),
            workspace=str(workspace),
            run_id="s0b-section:test:prebuild",
            max_segment_chars=120,
            duplicate_policy="fail",
        )
    )
    return workspace


def s1_args(workspace: Path, output_workspace: Path, mode: str, run_id: str) -> Namespace:
    return Namespace(
        project_root=str(ROOT),
        workspace=str(workspace),
        output_workspace=str(output_workspace),
        workspace_id=output_workspace.name,
        modeled_subject_id="Test Subject",
        run_id=run_id,
        run_scope="evidence_plus_memory",
        duplicate_policy="fail",
        pre_build_route_mode=mode,
        route_policy=ROUTE_POLICY,
        proposal_provider="mock",
        proposal_profile=S1_PROFILE,
        api_mode="responses",
        allow_live_api=False,
        external_model_outputs=None,
        max_items=5,
        item_offset=0,
        sample_stride=1,
    )


def s2_args(workspace: Path, mode: str, run_id: str) -> Namespace:
    return Namespace(
        workspace=str(workspace),
        workspace_id=workspace.name,
        modeled_user_id="Test Subject",
        target_participant="Test Subject",
        run_id=run_id,
        max_units=10,
        duplicate_policy="fail",
        pre_build_route_mode=mode,
        route_policy=ROUTE_POLICY,
        proposal_provider="mock",
        proposal_profile=S2_PROFILE,
        api_mode="responses",
        allow_live_api=False,
        external_model_outputs=None,
        max_items=5,
        item_offset=0,
        sample_stride=1,
    )


class PreBuildRoutingIntegrationTests(unittest.TestCase):
    def test_s1_route_only_writes_route_artifacts_without_proposals_or_canonical_change(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = create_plaintext_workspace(temp_dir)
            output_workspace = Path(temp_dir) / "s1_route_only_output"
            result = run_s1_build(
                s1_args(
                    workspace,
                    output_workspace,
                    "route_only",
                    "s1-build:test:prebuild-route-only",
                )
            )

            prebuild = result["prebuild_routing"]
            route_dir = Path(prebuild["route"]["output_dir"])
            self.assertEqual(prebuild["mode"], "route_only")
            self.assertTrue((route_dir / "route_decisions.jsonl").exists())
            self.assertTrue((route_dir / "route_run_manifest.json").exists())
            self.assertIsNone(prebuild["proposal"])
            self.assertFalse((output_workspace / "proposals").exists())
            self.assertTrue((output_workspace / "memory" / "memory_units.jsonl").exists())
            units = read_jsonl(output_workspace / "memory" / "memory_units.jsonl")
            self.assertTrue(units)
            self.assertTrue(all(unit["original_text"] for unit in units))
            self.assertTrue(all(unit["processed_text"] for unit in units))
            self.assertTrue(all(unit["processing_method"] == "script" for unit in units))

            manifest = json.loads((output_workspace / "evidence" / "build_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["prebuild_routing"]["mode"], "route_only")
            self.assertFalse(manifest["prebuild_routing"]["canonical_writes_executed"])

    def test_s1_route_and_propose_uses_shared_runner_without_writing_durable_memory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = create_plaintext_workspace(temp_dir)
            output_workspace = Path(temp_dir) / "s1_route_and_propose_output"
            result = run_s1_build(
                s1_args(
                    workspace,
                    output_workspace,
                    "route_and_propose",
                    "s1-build:test:prebuild-route-and-propose",
                )
            )

            proposal_dir = Path(result["prebuild_routing"]["proposal"]["output_dir"])
            proposals = read_jsonl(proposal_dir / "proposal_outcomes.ai.jsonl")
            proposal_manifest = json.loads((proposal_dir / "proposal_run_manifest.json").read_text(encoding="utf-8"))
            self.assertTrue(proposals)
            self.assertTrue(all(row["write_permission"] is False for row in proposals))
            self.assertFalse(proposal_manifest["durable_writes_executed"])
            self.assertFalse(proposal_manifest["reviewed_units_written"])
            self.assertFalse(proposal_manifest["graph_truth_written"])
            self.assertTrue((output_workspace / "memory" / "memory_units.jsonl").exists())
            units = read_jsonl(output_workspace / "memory" / "memory_units.jsonl")
            llm_units = [unit for unit in units if unit["processing_method"] == "llm_assisted"]
            self.assertTrue(llm_units)
            self.assertTrue(all(unit["original_text"] for unit in llm_units))
            self.assertTrue(all(unit["processed_text"] for unit in llm_units))
            self.assertTrue(all(unit["evidence_refs"] == unit["backpointer_refs"] for unit in llm_units))
            self.assertTrue(all(unit["llm_assist_used"] is True for unit in llm_units))
            self.assertFalse((output_workspace / "portrait" / "reviewed_units.jsonl").exists())

    def test_s2_route_and_propose_materializes_from_proposal_candidates_not_all_memory_units(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = create_plaintext_workspace(temp_dir)
            s1_output = Path(temp_dir) / "s2_prebuild_s1_output"
            run_s1_build(s1_args(workspace, s1_output, "none", "s1-build:test:s2-prebuild-input"))

            result = run_s2_build(
                s2_args(
                    s1_output,
                    "route_and_propose",
                    "s2-build:test:prebuild-route-and-propose",
                )
            )

            prebuild = result["prebuild_routing"]
            proposal_dir = Path(prebuild["proposal"]["output_dir"])
            proposals = read_jsonl(proposal_dir / "proposal_outcomes.ai.jsonl")
            status = json.loads((s1_output / "checkpoints" / "phase2-status.json").read_text(encoding="utf-8"))
            reviewed = read_jsonl(s1_output / "portrait" / "reviewed_units.jsonl")
            materialized_proposals = [
                row
                for row in proposals
                if row.get("output_kind") in {"portrait_fact_candidate", "portrait_hypothesis_candidate"}
            ]

            self.assertTrue(proposals)
            self.assertTrue(all(row["write_permission"] is False for row in proposals))
            self.assertEqual(status["prebuild_routing"]["target_task"], "s2_portrait_candidate")
            self.assertEqual(status["build_source"], "s2_proposal_outcomes")
            self.assertTrue((s1_output / "portrait" / "reviewed_units.jsonl").exists())
            self.assertEqual(len(reviewed), min(len(materialized_proposals), 10))
            self.assertTrue(reviewed)
            self.assertTrue(all(unit["step1_origin"]["input_layer"] == "s2_proposal_outcome" for unit in reviewed))
            self.assertTrue(all(unit["proposal_origin"]["proposal_id"] for unit in reviewed))
            self.assertFalse((s1_output / "indexes" / "step2_user_model_embedding_manifest.json").exists())
            self.assertFalse((s1_output / "indexes" / "step2_user_model_embedding_entries.jsonl").exists())
            self.assertFalse((s1_output / "indexes" / "step2_user_model_embedding_vectors.npy").exists())


if __name__ == "__main__":
    unittest.main()
