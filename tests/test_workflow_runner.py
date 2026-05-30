import argparse
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tools.graph.graph_construction_packet_builder import read_jsonl
from tools.workflow_runner import (
    discover_workflow_graph_dir,
    discover_workflow_profile_dir,
    run_build_graph,
    run_incremental,
    run_status,
    workspace_status,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def make_graph_workspace(root: Path) -> Path:
    workspace = root / "workflow_graph_workspace"
    (workspace / "evidence").mkdir(parents=True)
    (workspace / "portrait").mkdir()
    (workspace / "memory").mkdir()
    (workspace / "proposals" / "prebuild_s2_test").mkdir(parents=True)
    (workspace / "manifest.yaml").write_text(
        "workspace_id: workflow_graph_workspace\nmodeled_user_id: Mira Chen\n",
        encoding="utf-8",
    )
    write_jsonl(
        workspace / "evidence" / "evidence.jsonl",
        [
            {
                "evidence_ref": "evidence:test:001",
                "source_id": "source:test",
                "step1_evidence_ref": "s1:evidence:001",
                "step1_source_ref": "s1:source:001",
                "participant": "Mira Chen",
                "target_participant": "Mira Chen",
                "subject_role": "target",
                "timestamp": "2026-05-23T00:00:00Z",
                "text": "Mira uses a graph notebook to track project decisions.",
                "privacy_class": "public_dataset",
            }
        ],
    )
    write_jsonl(
        workspace / "portrait" / "normalized_candidates.jsonl",
        [
            {
                "candidate_id": "npc-mira-001",
                "user_id": "Mira Chen",
                "candidate_text": "Mira uses a graph notebook for project decisions.",
                "candidate_type": "project_context",
                "source_refs": ["source:test"],
                "evidence_refs": ["evidence:test:001"],
                "backpointer_refs": ["s1:evidence:001"],
                "input_layer": "s2_proposal_outcome",
                "input_refs": ["s2p:mira-001"],
                "proposal_origin": {"proposal_id": "s2p:mira-001"},
                "confidence": "high",
                "inference_level": "explicit",
                "privacy_class": "public_dataset",
            }
        ],
    )
    write_jsonl(
        workspace / "portrait" / "review_decisions.jsonl",
        [
            {
                "decision_id": "review-mira-001",
                "candidate_id": "npc-mira-001",
                "review_status": "accepted_for_experiment",
                "review_action": "accept_for_experiment",
            }
        ],
    )
    write_jsonl(
        workspace / "portrait" / "reviewed_units.jsonl",
        [
            {
                "unit_id": "unit-mira-001",
                "user_id": "Mira Chen",
                "type": "project_context",
                "memory_class": "semantic",
                "content": "Mira uses a graph notebook for project decisions.",
                "scope": "project",
                "source_refs": ["source:test"],
                "evidence_refs": ["evidence:test:001"],
                "backpointer_refs": ["s1:evidence:001"],
                "evidence_summary": "Mira uses a graph notebook to track project decisions.",
                "confidence": "high",
                "inference_level": "explicit",
                "status": "active",
                "privacy_class": "public_dataset",
                "review_metadata": {
                    "review_status": "accepted_for_experiment",
                    "review_action": "accept_for_experiment",
                },
                "proposal_origin": {"proposal_id": "s2p:mira-001"},
                "step1_origin": {
                    "input_layer": "s2_proposal_outcome",
                    "input_refs": ["s2p:mira-001"],
                },
            }
        ],
    )
    write_jsonl(
        workspace / "memory" / "preprocessing_decisions.jsonl",
        [
            {
                "decision_id": "prep-mira-001",
                "input_layer": "evidence_item",
                "evidence_ref": "evidence:test:001",
                "route": "graph_candidate_needed",
                "warnings": [],
            }
        ],
    )
    write_jsonl(
        workspace / "proposals" / "prebuild_s2_test" / "proposal_outcomes.ai.jsonl",
        [
            {
                "proposal_id": "s2p:mira-001",
                "output_kind": "portrait_fact_candidate",
                "candidate_type": "project_context",
                "fact_candidate_text": "Mira uses a graph notebook for project decisions.",
                "source_text": "Mira uses a graph notebook to track project decisions.",
                "source_perspective": "Mira Chen",
                "attribution_status": "strict",
                "source_refs": ["source:test"],
                "evidence_refs": ["evidence:test:001"],
                "raw_backpointer_refs": ["s1:evidence:001"],
                "proposal_confidence": "high",
                "inference_level": "explicit",
                "write_permission": False,
            }
        ],
    )
    return workspace


class WorkflowRunnerTests(unittest.TestCase):
    def test_status_identifies_empty_and_current_view_assets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "status_workspace"
            workspace.mkdir()

            empty = workspace_status(workspace)
            self.assertTrue(empty["workspace_exists"])
            self.assertIsNone(empty["discovery"]["graph_dir"])

            write_jsonl(workspace / "maintenance" / "latest_views" / "s1_latest_view.jsonl", [{"unit_id": "s1"}])
            write_jsonl(workspace / "graph_current" / "graph_nodes_latest_view.jsonl", [{"node_id": "n1"}])
            write_jsonl(workspace / "graph_current" / "graph_edges_latest_view.jsonl", [{"edge_id": "e1"}])

            status = workspace_status(workspace)
            self.assertEqual(status["discovery"]["query_graph_context_mode"], "v03")
            self.assertTrue(any(row["kind"] == "s1_latest_view" and row["exists"] for row in status["artifacts"]))

            manifest = run_status(argparse.Namespace(workspace=str(workspace), run_id="status-test"))
            self.assertTrue(Path(manifest["outputs"]["workflow_manifest"]).exists())

    def test_query_graph_discovery_prefers_graph_current(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "query_workspace"
            workspace.mkdir()
            write_jsonl(workspace / "graph_v03_consolidation_provider_80" / "graph_nodes_table.jsonl", [{"node_id": "old"}])
            write_jsonl(workspace / "graph_v03_consolidation_provider_80" / "graph_edges_table.jsonl", [{"edge_id": "old-e"}])
            write_jsonl(workspace / "graph_current" / "graph_nodes_latest_view.jsonl", [{"node_id": "current"}])
            write_jsonl(workspace / "graph_current" / "graph_edges_latest_view.jsonl", [{"edge_id": "current-e"}])

            self.assertEqual(discover_workflow_graph_dir(workspace), (workspace / "graph_current").resolve())

    def test_query_graph_discovery_finds_latest_hash_suffixed_graph_and_matching_profile(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "query_workspace"
            workspace.mkdir()
            old_graph = workspace / "graph_v03_consolidation_old"
            new_graph = workspace / "graph_v03_consolidation_new"
            old_profile = workspace / "graph_v03_profile_communities_old"
            new_profile = workspace / "graph_v03_profile_communities_new"

            write_jsonl(old_graph / "graph_nodes_table.jsonl", [{"node_id": "old"}])
            write_jsonl(old_graph / "graph_edges_table.jsonl", [{"edge_id": "old-e"}])
            write_jsonl(old_profile / "graph_community_reports.jsonl", [{"community_id": "old-c"}])
            write_jsonl(new_graph / "graph_nodes_table.jsonl", [{"node_id": "new"}])
            write_jsonl(new_graph / "graph_edges_table.jsonl", [{"edge_id": "new-e"}])
            write_jsonl(new_profile / "graph_community_reports.jsonl", [{"community_id": "new-c"}])
            os.utime(old_graph, (1000, 1000))
            os.utime(new_graph, (2000, 2000))

            graph_dir = discover_workflow_graph_dir(workspace)
            self.assertEqual(graph_dir, new_graph.resolve())
            self.assertEqual(discover_workflow_profile_dir(workspace, graph_dir=graph_dir), new_profile.resolve())

    def test_incremental_prepare_requires_review_before_finalize_and_smoke_writes_workflow_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir) / "incremental_workspace"
            workspace.mkdir()
            prepare = run_incremental(
                argparse.Namespace(
                    workspace=str(workspace),
                    package_dir=None,
                    run_id="incremental-prepare-test",
                    incremental_command="prepare",
                    reset=False,
                    port=8765,
                    auto_review_for_test=False,
                )
            )
            self.assertEqual(prepare["status"], "waiting_for_human_review")
            self.assertTrue(Path(prepare["outputs"]["workflow_manifest"]).exists())

            with self.assertRaisesRegex(FileNotFoundError, "Human review is not finalized"):
                run_incremental(
                    argparse.Namespace(
                        workspace=str(workspace),
                        package_dir=prepare["outputs"]["package_dir"],
                        run_id="incremental-finalize-test",
                        incremental_command="finalize",
                        reset=False,
                        port=8765,
                        auto_review_for_test=False,
                    )
                )

            smoke_workspace = Path(temp_dir) / "incremental_smoke_workspace"
            smoke = run_incremental(
                argparse.Namespace(
                    workspace=str(smoke_workspace),
                    package_dir=None,
                    run_id="incremental-smoke-test",
                    incremental_command="smoke",
                    reset=True,
                    port=8765,
                    auto_review_for_test=False,
                )
            )
            self.assertEqual(smoke["status"], "pass")
            self.assertTrue(Path(smoke["outputs"]["workflow_manifest"]).exists())
            self.assertFalse(smoke["boundary"]["graph_truth_written"])

    def test_build_graph_mock_baseline_runs_to_quality_gate_without_truth_writes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = make_graph_workspace(Path(temp_dir))
            manifest = run_build_graph(
                argparse.Namespace(
                    project_root=str(Path(__file__).resolve().parents[1]),
                    workspace=str(workspace),
                    run_id="graph-smoke-test",
                    output_suffix="mock_smoke",
                    provider="mock_regex_baseline",
                    api_mode=None,
                    allow_live_api=False,
                    max_items=4,
                    provider_concurrency=1,
                    duplicate_policy="overwrite_generated",
                    relation_schema_candidates=None,
                    projection="review_aware_graph",
                    community_report_provider="extractive",
                    max_provider_reports=0,
                    quality_sample_limit=4,
                    visual=False,
                )
            )

            self.assertEqual(manifest["status"], "completed")
            self.assertFalse(manifest["provider_policy"]["mock_regex_baseline_is_main_fidelity_path"])
            self.assertFalse(manifest["boundary"]["graph_truth_written"])
            self.assertTrue((workspace / "graph_v03_final_quality_gate_mock_smoke" / "graph_final_quality_gate_summary.json").exists())
            nodes = read_jsonl(workspace / "graph_v03_consolidation_mock_smoke" / "graph_nodes_table.jsonl")
            self.assertGreaterEqual(len(nodes), 1)


if __name__ == "__main__":
    unittest.main()
