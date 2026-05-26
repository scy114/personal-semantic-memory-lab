import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tools.step2.step2_user_model_index_runner import build_index, read_jsonl, resolve_workspace_identity


class Step2UserModelIndexRunnerTests(unittest.TestCase):
    def test_resolves_modeled_user_id_from_workspace_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "portrait").mkdir()
            (workspace / "manifest.yaml").write_text(
                "schema_version: s2.workspace_manifest.v0.1\n"
                "workspace_id: test_workspace\n"
                "modeled_user_id: Mira Chen\n",
                encoding="utf-8",
            )
            (workspace / "portrait" / "reviewed_units.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": "s2.reviewed_portrait_unit.v1",
                        "unit_id": "unit-mira-001",
                        "user_id": "Mira Chen",
                        "status": "active",
                        "content": "Mira prefers fixed runners.",
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            units = read_jsonl(workspace / "portrait" / "reviewed_units.jsonl")
            workspace_id, modeled_user_id = resolve_workspace_identity(
                workspace,
                Namespace(workspace_id=None, modeled_user_id=None),
                units,
            )

            self.assertEqual(workspace_id, "test_workspace")
            self.assertEqual(modeled_user_id, "Mira Chen")

    def test_resolves_modeled_user_id_from_reviewed_units_when_manifest_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "portrait").mkdir()
            units = [{"unit_id": "unit-mira-001", "user_id": "Mira Chen", "content": "Mira prefers evidence refs."}]

            workspace_id, modeled_user_id = resolve_workspace_identity(
                workspace,
                Namespace(workspace_id=None, modeled_user_id=None),
                units,
            )

            self.assertEqual(workspace_id, workspace.name)
            self.assertEqual(modeled_user_id, "Mira Chen")

    def test_build_index_updates_status_and_records_prebuild_proposal_lineage(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "portrait").mkdir(parents=True)
            (workspace / "checkpoints").mkdir()
            (workspace / "proposals" / "prebuild_s2_test").mkdir(parents=True)
            proposal_outcomes = workspace / "proposals" / "prebuild_s2_test" / "proposal_outcomes.ai.jsonl"
            proposal_manifest = workspace / "proposals" / "prebuild_s2_test" / "proposal_run_manifest.json"
            proposal_outcomes.write_text(json.dumps({"proposal_id": "s2p:test"}, ensure_ascii=False) + "\n", encoding="utf-8")
            proposal_manifest.write_text(json.dumps({"proposal_run_id": "prebuild-proposal:s2:test"}, ensure_ascii=False), encoding="utf-8")
            (workspace / "portrait" / "reviewed_units.jsonl").write_text(
                json.dumps(
                    {
                        "schema_version": "s2.reviewed_portrait_unit.v1",
                        "unit_id": "unit-mira-001",
                        "user_id": "Mira Chen",
                        "status": "active",
                        "content": "Mira prefers evidence refs.",
                        "type": "preference",
                        "memory_class": "semantic",
                        "scope": "global",
                        "confidence": "high",
                        "inference_level": "explicit",
                        "evidence_refs": ["evidence:test:1"],
                        "backpointer_refs": ["evidence:test:1"],
                        "source_refs": ["raw:test"],
                        "review_metadata": {"review_status": "accepted_for_experiment"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            (workspace / "checkpoints" / "phase2-status.json").write_text(
                json.dumps(
                    {
                        "workspace_id": "test_workspace",
                        "modeled_user_id": "Mira Chen",
                        "build_source": "s2_proposal_outcomes",
                        "completed_steps": ["reviewed_portrait_units"],
                        "latest_outputs": {"reviewed_units": "portrait/reviewed_units.jsonl"},
                        "prebuild_routing": {
                            "proposal": {
                                "proposal_run_id": "prebuild-proposal:s2:test",
                                "proposal_profile_id": "s2_portrait_proposal.v0.2",
                                "provider": "mock",
                                "outputs": {
                                    "proposals": str(proposal_outcomes),
                                    "proposal_run_manifest": str(proposal_manifest),
                                },
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

            with patch(
                "tools.step2.step2_user_model_index_runner.encode_qwen_texts",
                return_value=(np.zeros((1, 4), dtype="float32"), {"model": "mock-embed", "embedding_dim": 4}),
            ):
                manifest = build_index(
                    Namespace(
                        workspace=str(workspace),
                        workspace_id=None,
                        modeled_user_id=None,
                        output_path=None,
                        model="mock-embed",
                        batch_size=16,
                        max_length=512,
                        device="cpu",
                        duplicate_policy="fail",
                        write_model_suffix=False,
                    )
                )

            status = json.loads((workspace / "checkpoints" / "phase2-status.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["upstream_lineage"]["build_source"], "s2_proposal_outcomes")
            self.assertEqual(
                manifest["upstream_lineage"]["prebuild_proposal"]["proposal_run_id"],
                "prebuild-proposal:s2:test",
            )
            self.assertIn("step2_user_model_index", status["completed_steps"])
            self.assertEqual(status["step2_user_model_index"]["entry_count"], 1)
            self.assertEqual(
                status["latest_outputs"]["step2_user_model_embedding_manifest"],
                "indexes/step2_user_model_embedding_manifest.json",
            )


if __name__ == "__main__":
    unittest.main()
