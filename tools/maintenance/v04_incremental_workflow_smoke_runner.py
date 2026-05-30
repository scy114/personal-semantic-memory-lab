"""Build and run a synthetic v0.4 incremental maintenance workflow package.

The smoke package validates the post-review merge/publish path end to end:

reviewed S1 overlay -> S1 current
reviewed S2 candidates -> S2 current
graph latest-view overlay -> graph_current publish
graph dependency / invalidation / visual review refresh

It uses synthetic evidence-bound rows and never calls a live provider, writes
durable memory, or writes graph truth.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.graph.graph_construction_packet_builder import file_hash, read_jsonl, write_json, write_text
from tools.maintenance.graph_candidate_latest_view_runner import run_graph_candidate_latest_view
from tools.maintenance.graph_current_view_publisher import run_graph_current_view_publish
from tools.maintenance.s1_current_view_publisher import run_s1_current_view_publish
from tools.maintenance.s1_review_overlay_consumer import build_s1_review_latest_view_from_files
from tools.maintenance.s2_current_view_publisher import run_s2_current_view_publish


SCHEMA_VERSION = "maintenance.v04_incremental_workflow_smoke.v0.4"
DEFAULT_PACKAGE_DIR_NAME = "maintenance_v04_workflow_smoke"
MANIFEST_FILENAME = "v04_incremental_workflow_smoke_manifest.json"
REPORT_FILENAME = "v04_incremental_workflow_smoke_report.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def seed_s1_inputs(input_dir: Path) -> dict[str, str]:
    active = input_dir / "s1_active_before_review.jsonl"
    incremental = input_dir / "s1_incremental_candidates.jsonl"
    overlay = input_dir / "s1_latest_view_overlay.jsonl"
    write_jsonl(
        active,
        [
            {
                "memory_id": "memory-old",
                "content": "Jon prefers contemporary dance for the performance.",
                "evidence_refs": ["evidence:old"],
                "raw_backpointer_refs": [{"display_ref": "D1:1"}],
                "graph_is_not_proof": True,
            },
            {
                "memory_id": "memory-keep",
                "content": "Jon is preparing a performance.",
                "evidence_refs": ["evidence:keep"],
                "raw_backpointer_refs": [{"display_ref": "D1:2"}],
                "graph_is_not_proof": True,
            },
        ],
    )
    write_jsonl(
        incremental,
        [
            {
                "memory_id": "memory-new",
                "content": "Jon currently prefers salsa for the performance.",
                "evidence_refs": ["evidence:new"],
                "raw_backpointer_refs": [{"display_ref": "D2:1"}],
                "graph_is_not_proof": True,
            }
        ],
    )
    write_jsonl(
        overlay,
        [
            {
                "review_item_id": "review-s1-1",
                "patch_id": "patch-s1-1",
                "operation_id": "op-smoke",
                "overlay_effect": "split_current_and_historical_overlay",
                "current_candidate_ids": ["memory-new"],
                "historical_or_stale_candidate_ids": ["memory-old"],
            }
        ],
    )
    return {"active": str(active), "incremental": str(incremental), "overlay": str(overlay)}


def seed_s2_inputs(input_dir: Path) -> dict[str, str]:
    base = input_dir / "s2_base_reviewed_units.jsonl"
    candidates = input_dir / "s2_incremental_candidates.jsonl"
    decisions = input_dir / "s2_review_decisions.jsonl"
    affected = input_dir / "s2_affected_units.jsonl"
    write_jsonl(
        base,
        [
            {
                "schema_version": "s2.reviewed_portrait_unit.v1",
                "unit_id": "s2-old",
                "user_id": "Jon",
                "type": "preference",
                "content": "Jon prefers contemporary dance for the performance.",
                "evidence_refs": ["evidence:old"],
                "status": "active",
            }
        ],
    )
    write_jsonl(
        candidates,
        [
            {
                "schema_version": "maintenance.s2_incremental_candidate.v0.4",
                "candidate_status": "candidate_ready_for_review",
                "candidate_text": "Jon currently prefers salsa for the performance.",
                "candidate_type": "preference",
                "proposal_confidence": "high",
                "inference_level": "explicit",
                "proposal_id": "s2p-new",
                "proposal_input_id": "s2pi-new",
                "proposal_run_id": "run-smoke",
                "provider": "synthetic",
                "model_id": "smoke-fixture",
                "output_kind": "portrait_fact_candidate",
                "route_used": "approved_synthetic_smoke",
                "review_item_id": "review-s2-1",
                "patch_id": "patch-s2-1",
                "operation_id": "op-smoke",
                "source_s1_memory_id": "memory-new",
                "refresh_reason": "split_current_and_historical_overlay",
                "evidence_refs": ["evidence:new"],
                "raw_backpointer_refs": [{"display_ref": "D2:1"}],
                "source_text_quote": "For the performance now, salsa is my first choice.",
                "proposal_row": {
                    "target_participant": "Jon",
                    "subject_id": "Jon",
                    "privacy_class": "personal_low",
                    "candidate_text": "Jon currently prefers salsa for the performance.",
                    "proposal_confidence": "high",
                    "inference_level": "explicit",
                },
            }
        ],
    )
    write_jsonl(
        decisions,
        [
            {
                "schema_version": "maintenance.incremental_review_decision.v0.4",
                "review_item_id": "review-s2-1",
                "review_status": "approved_for_apply",
                "human_decision": "approve_recommended",
                "internal_review_action": "accept_patch",
            }
        ],
    )
    write_jsonl(
        affected,
        [
            {
                "new_s1_unit_id": "memory-new",
                "affected_s2_unit_id": "s2-old",
                "s2_delta_signal": "contradicts_profile",
                "subject_scope_existing_s2_unit_ids": ["s2-old"],
            }
        ],
    )
    return {"base": str(base), "candidates": str(candidates), "decisions": str(decisions), "affected": str(affected)}


def graph_node(node_id: str, label: str) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "label": label,
        "entity_type": "person" if label == "Jon" else "activity",
        "entity_quality_hint": "stable",
        "evidence_refs": ["evidence:old" if "contemporary" in node_id else "evidence:new"],
        "graph_is_not_proof": True,
    }


def seed_graph_dir(graph_dir: Path, *, target_node_id: str, target_label: str, evidence_ref: str, quote: str) -> None:
    write_jsonl(
        graph_dir / "graph_nodes_table.jsonl",
        [
            graph_node("node:jon", "Jon"),
            graph_node(target_node_id, target_label),
        ],
    )
    write_jsonl(
        graph_dir / "graph_edges_table.jsonl",
        [
            {
                "edge_id": "edge:jon-performance-preference",
                "source_node_id": "node:jon",
                "target_node_id": target_node_id,
                "source_label": "Jon",
                "target_label": target_label,
                "relation_type": "prefers_for_performance",
                "generic_relation_review_hint": "not_generic",
                "evidence_refs": [evidence_ref],
                "source_text_quotes": [quote],
                "source_packet_ids": ["packet:smoke"],
                "candidate_ids": ["candidate:graph-smoke"],
                "graph_is_not_proof": True,
            }
        ],
    )
    write_jsonl(
        graph_dir / "graph_claims_table.jsonl",
        [
            {
                "claim_id": "claim:jon-performance-preference",
                "subject_node_id": "node:jon",
                "subject_endpoint_status": "local_entity_id",
                "evidence_refs": [evidence_ref],
                "source_packet_id": "packet:smoke",
                "candidate_id": "candidate:claim-smoke",
                "graph_is_not_proof": True,
            }
        ],
    )
    write_jsonl(
        graph_dir / "evidence_links.jsonl",
        [
            {
                "link_id": "link:jon-performance-preference",
                "owner_kind": "edge",
                "owner_id": "edge:jon-performance-preference",
                "evidence_ref": evidence_ref,
                "source_packet_id": "packet:smoke",
                "candidate_id": "candidate:evidence-link-smoke",
                "graph_is_not_proof": True,
            }
        ],
    )
    write_json(graph_dir / "graph_consolidation_manifest.json", {"schema_version": "synthetic.graph_consolidation_smoke"})


def seed_graph_inputs(input_dir: Path) -> dict[str, str]:
    base = input_dir / "graph_base_consolidated"
    incremental = input_dir / "graph_incremental_consolidated"
    seed_graph_dir(
        base,
        target_node_id="node:contemporary",
        target_label="contemporary dance",
        evidence_ref="evidence:old",
        quote="Jon prefers contemporary dance for the performance.",
    )
    seed_graph_dir(
        incremental,
        target_node_id="node:salsa",
        target_label="salsa",
        evidence_ref="evidence:new",
        quote="For the performance now, salsa is my first choice.",
    )
    return {"base": str(base), "incremental": str(incremental)}


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.4 Incremental Workflow Smoke Package",
        "",
        f"- generated_at: {manifest['generated_at']}",
        f"- workspace: `{manifest['workspace']}`",
        f"- package_dir: `{manifest['package_dir']}`",
        f"- acceptance_status: `{manifest['acceptance_status']}`",
        "",
        "## Stage Summary",
        "",
    ]
    for name, stage in manifest["stages"].items():
        lines.extend(
            [
                f"### {name}",
                "",
                f"- status: `{stage['status']}`",
                f"- manifest: `{stage.get('manifest', '')}`",
                f"- key_counts: `{json.dumps(stage.get('counts', {}), ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Workflow Boundary",
            "",
            "- This smoke package validates runner orchestration, artifacts, and current-view publication.",
            "- It uses synthetic rows and does not call a live provider.",
            "- It does not write durable memory, canonical graph truth, S3, or support-checker authority.",
            "- Graph visualization is audit support only.",
            "",
        ]
    )
    return "\n".join(lines)


def run_v04_incremental_workflow_smoke(
    *,
    workspace: Path,
    package_dir: Path | None = None,
    reset: bool = False,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    package_dir = (package_dir or workspace / DEFAULT_PACKAGE_DIR_NAME).resolve()
    if package_dir.exists():
        if not reset:
            raise FileExistsError(f"Smoke package already exists; pass reset=True to rebuild: {package_dir}")
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)
    input_dir = package_dir / "inputs"

    s1_inputs = seed_s1_inputs(input_dir)
    s2_inputs = seed_s2_inputs(input_dir)
    graph_inputs = seed_graph_inputs(input_dir)

    s1_after_review = package_dir / "s1_after_review"
    s1_bundle = build_s1_review_latest_view_from_files(
        active_s1_jsonl=Path(s1_inputs["active"]),
        incremental_s1_jsonl=Path(s1_inputs["incremental"]),
        s1_review_overlay_jsonl=Path(s1_inputs["overlay"]),
        output_dir=s1_after_review,
    )
    s1_publish = run_s1_current_view_publish(workspace=package_dir, source_dir=s1_after_review)

    s2_publish = run_s2_current_view_publish(
        workspace=package_dir,
        base_reviewed_units_jsonl=Path(s2_inputs["base"]),
        candidate_outputs_jsonl=Path(s2_inputs["candidates"]),
        review_decisions_jsonl=Path(s2_inputs["decisions"]),
        affected_s2_jsonl=Path(s2_inputs["affected"]),
    )

    graph_latest_base = package_dir / "graph_latest_base"
    graph_base_latest = run_graph_candidate_latest_view(
        incremental_graph_dir=Path(graph_inputs["base"]),
        output_dir=graph_latest_base,
    )
    graph_first_publish = run_graph_current_view_publish(workspace=package_dir, source_latest_view_dir=graph_latest_base)

    graph_latest_after_increment = package_dir / "graph_latest_after_increment"
    graph_increment_latest = run_graph_candidate_latest_view(
        base_graph_dir=Path(graph_inputs["base"]),
        incremental_graph_dir=Path(graph_inputs["incremental"]),
        output_dir=graph_latest_after_increment,
    )
    graph_second_publish = run_graph_current_view_publish(
        workspace=package_dir,
        source_latest_view_dir=graph_latest_after_increment,
    )

    visual_manifest_path = Path(graph_second_publish["maintenance_artifacts"]["incremental_visual_review_manifest"])
    visual_manifest = json.loads(visual_manifest_path.read_text(encoding="utf-8-sig"))
    changed_units = read_jsonl(Path(graph_second_publish["maintenance_artifacts"]["changed_graph_units"]))

    acceptance_checks = {
        "s1_current_has_active_rows": s1_publish["counts"]["current_active_count"] >= 1,
        "s1_current_has_excluded_rows": s1_publish["counts"]["current_excluded_count"] >= 1,
        "s2_current_has_active_rows": s2_publish["counts"]["current_active_count"] >= 1,
        "s2_current_has_excluded_rows": s2_publish["counts"]["current_excluded_count"] >= 1,
        "graph_publish_changed_units_nonzero": graph_second_publish["maintenance_counts"]["changed_graph_units"] > 0,
        "graph_visual_slices_nonzero": visual_manifest["counts"]["slices"] > 0,
        "graph_query_default_ready": graph_second_publish["query_default_ready"] is True,
        "no_durable_memory_write": not any(
            [
                s1_publish.get("durable_writes_executed"),
                s2_publish.get("durable_writes_executed"),
                graph_second_publish.get("durable_writes_executed"),
            ]
        ),
        "no_graph_truth_write": graph_second_publish.get("graph_truth_written") is False,
    }
    acceptance_status = "pass" if all(acceptance_checks.values()) else "fail"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "package_dir": str(package_dir),
        "acceptance_status": acceptance_status,
        "acceptance_checks": acceptance_checks,
        "inputs": {
            "s1": s1_inputs,
            "s2": s2_inputs,
            "graph": graph_inputs,
        },
        "stages": {
            "s1_after_review_latest_view": {
                "status": "pass",
                "manifest": s1_bundle["paths"]["manifest"],
                "counts": s1_bundle["manifest"],
            },
            "s1_current_publish": {
                "status": "pass",
                "manifest": s1_publish["paths"]["manifest"],
                "counts": s1_publish["counts"],
            },
            "s2_current_publish": {
                "status": "pass",
                "manifest": s2_publish["paths"]["manifest"],
                "counts": s2_publish["counts"],
            },
            "graph_initial_current_publish": {
                "status": "pass",
                "manifest": graph_first_publish["paths"]["manifest"],
                "counts": graph_first_publish["maintenance_counts"],
            },
            "graph_incremental_latest_view": {
                "status": "pass",
                "manifest": graph_increment_latest["paths"]["manifest"],
                "counts": graph_increment_latest["counts"],
            },
            "graph_current_publish_with_invalidation_and_visual": {
                "status": "pass",
                "manifest": graph_second_publish["paths"]["manifest"],
                "counts": graph_second_publish["maintenance_counts"],
            },
            "graph_visual_review_incremental": {
                "status": "pass",
                "manifest": str(visual_manifest_path),
                "counts": visual_manifest["counts"],
            },
        },
        "outputs": {
            "manifest": str(package_dir / MANIFEST_FILENAME),
            "report": str(package_dir / REPORT_FILENAME),
            "s1_current": s1_publish["paths"]["s1_current"],
            "s2_current": s2_publish["paths"]["s2_current"],
            "graph_current": graph_second_publish["current_graph_dir"],
            "changed_graph_units": graph_second_publish["maintenance_artifacts"]["changed_graph_units"],
            "incremental_visual_review_html": graph_second_publish["maintenance_artifacts"]["incremental_visual_review_html"],
        },
        "sample_changed_graph_units": changed_units[:5],
        "source_hashes": {
            "s1_overlay": file_hash(Path(s1_inputs["overlay"])),
            "s2_review_decisions": file_hash(Path(s2_inputs["decisions"])),
            "graph_incremental_edges": file_hash(Path(graph_inputs["incremental"]) / "graph_edges_table.jsonl"),
        },
        "boundary": {
            "synthetic_smoke_only": True,
            "provider_calls_executed": False,
            "graph_is_not_proof": True,
            "support_status": "not_checked",
            "durable_memory_written": False,
            "graph_truth_written": False,
            "support_checker_authority": False,
        },
    }
    write_json(package_dir / MANIFEST_FILENAME, manifest)
    write_text(package_dir / REPORT_FILENAME, render_report(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--package-dir", type=Path)
    parser.add_argument("--reset", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_v04_incremental_workflow_smoke(
        workspace=args.workspace,
        package_dir=args.package_dir,
        reset=args.reset,
    )
    print(
        json.dumps(
            {
                "manifest": manifest["outputs"]["manifest"],
                "report": manifest["outputs"]["report"],
                "acceptance_status": manifest["acceptance_status"],
                "outputs": manifest["outputs"],
            },
            ensure_ascii=False,
        )
    )
    return 0 if manifest["acceptance_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
