"""Thin workflow entrypoint over the existing memory/graph toolchain.

This module intentionally orchestrates existing runners instead of replacing
their implementation. It writes a workflow-level manifest/report so users and
agents can find the right artifacts without re-reading every reference doc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.graph.graph_candidate_consolidator import consolidate_graph_candidates
from tools.graph.graph_construction_packet_builder import build_graph_construction_packets
from tools.graph.graph_final_quality_gate_reporter import compute_quality_gate
from tools.graph.graph_package_router import build_graph_package_routes
from tools.graph.graph_profile_community_builder import run_builder as run_graph_profile_community_builder
from tools.graph.graph_query_retriever import discover_graph_dir, graph_dir_has_query_tables
from tools.graph.graph_relation_candidate_extractor import build_graph_relation_candidates
from tools.graph.graph_visual_review_bundle_builder import run_visual_review_bundle
from tools.graph.networkx_graph_utility_runner import run_networkx_utility
from tools.maintenance.v04_full_review_workflow_package_runner import (
    finalize_full_review_workflow_package,
    prepare_full_review_workflow_package,
)
from tools.maintenance.v04_incremental_workflow_smoke_runner import run_v04_incremental_workflow_smoke


SCHEMA_VERSION = "workflow.v041.manifest"
DEFAULT_WORKFLOW_ROOT = "workflow_runs"
BOUNDARY_FLAGS = {
    "durable_memory_written": False,
    "graph_truth_written": False,
    "support_checker_authority": False,
    "s3_written": False,
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug(value: str) -> str:
    allowed = []
    for char in value.lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_", ".", ":"}:
            allowed.append("-" if char == ":" else char)
    text = "".join(allowed).strip("-._")
    return text or "workflow"


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def default_run_id(command: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"v041-{slug(command)}-{stamp}"


def resolve_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def count_jsonl(path: Path) -> int:
    if not path.exists() or not path.is_file():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip())


def file_hash(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def workflow_output_dir(workspace: Path, run_id: str) -> Path:
    return workspace / DEFAULT_WORKFLOW_ROOT / run_id


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def command_row(module: str, args: list[str]) -> dict[str, Any]:
    return {"module": module, "argv": [sys.executable, "-m", module, *args]}


def run_module(module: str, args: list[str], *, cwd: Path) -> dict[str, Any]:
    cmd = [sys.executable, "-m", module, *args]
    completed = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)
    result = {
        "module": module,
        "argv": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }
    if completed.returncode != 0:
        raise RuntimeError(f"Workflow step failed: {module}\n{completed.stderr[-2000:]}")
    return result


def artifact_row(workspace: Path, rel_path: str, *, layer: str, kind: str) -> dict[str, Any]:
    path = workspace / rel_path
    suffix = path.suffix.lower()
    count = count_jsonl(path) if suffix == ".jsonl" else None
    return {
        "layer": layer,
        "kind": kind,
        "path": str(path),
        "relative_path": rel_path,
        "exists": path.exists(),
        "row_count": count,
        "hash": file_hash(path),
    }


def workspace_status(workspace: Path) -> dict[str, Any]:
    workspace = workspace.resolve()
    specs = [
        ("raw/organization/text_units.jsonl", "s0b", "text_units"),
        ("raw/organization/section_map.jsonl", "s0b", "section_map"),
        ("raw/organization/source_range_map.jsonl", "s0b", "source_range_map"),
        ("evidence/evidence.jsonl", "s1", "evidence"),
        ("memory/memory_units.jsonl", "s1", "memory_units"),
        ("indexes/step1_bm25_manifest.json", "s1_index", "bm25_manifest"),
        ("indexes/step1_embedding_manifest.json", "s1_index", "embedding_manifest"),
        ("portrait/normalized_candidates.jsonl", "s2", "normalized_candidates"),
        ("portrait/reviewed_units.jsonl", "s2", "reviewed_units"),
        ("indexes/step2_user_model_embedding_manifest.json", "s2_index", "embedding_manifest"),
        ("graph_v03_construction/graph_construction_packets.jsonl", "graph", "construction_packets"),
        ("graph_v03_construction/graph_route_decisions.jsonl", "graph", "route_decisions"),
        ("graph_v03_consolidation_provider_80/graph_nodes_table.jsonl", "graph", "nodes_provider_80"),
        ("graph_v03_consolidation/graph_nodes_table.jsonl", "graph", "nodes_default"),
        ("graph_current/graph_nodes_latest_view.jsonl", "graph_current", "nodes_latest"),
        ("maintenance/latest_views/s1_latest_view.jsonl", "maintenance", "s1_latest_view"),
        ("maintenance/latest_views/s2_latest_view.jsonl", "maintenance", "s2_latest_view"),
    ]
    artifacts = [artifact_row(workspace, rel, layer=layer, kind=kind) for rel, layer, kind in specs]
    layer_counts: dict[str, dict[str, int]] = {}
    for row in artifacts:
        layer = row["layer"]
        layer_counts.setdefault(layer, {"present": 0, "missing": 0})
        layer_counts[layer]["present" if row["exists"] else "missing"] += 1
    graph_dir = discover_workflow_graph_dir(workspace)
    profile_dir = discover_workflow_profile_dir(workspace, graph_dir=graph_dir)
    return {
        "workspace": str(workspace),
        "workspace_exists": workspace.exists(),
        "artifacts": artifacts,
        "layer_counts": layer_counts,
        "discovery": {
            "graph_dir": str(graph_dir) if graph_dir else None,
            "graph_profile_dir": str(profile_dir) if profile_dir else None,
            "query_graph_context_mode": "v03" if graph_dir else "disabled",
        },
    }


def discover_workflow_graph_dir(workspace: Path, explicit_graph_dir: Path | None = None) -> Path | None:
    if explicit_graph_dir:
        path = explicit_graph_dir.resolve()
        return path if graph_dir_has_query_tables(path) else None
    discovered = discover_graph_dir(workspace, None)
    if discovered:
        return discovered
    for name in ("graph_v03_consolidation_provider_80", "graph_v03_consolidation", "graph_v03_consolidation_provider"):
        path = workspace / name
        if graph_dir_has_query_tables(path):
            return path.resolve()
    latest = newest_valid_child_dir(workspace, "graph_v03_consolidation_", graph_dir_has_query_tables)
    if latest:
        return latest
    return None


def has_graph_community_reports(path: Path) -> bool:
    return (path / "graph_community_reports.jsonl").exists()


def newest_valid_child_dir(workspace: Path, prefix: str, predicate: Any) -> Path | None:
    if not workspace.exists():
        return None
    candidates = [
        path
        for path in workspace.iterdir()
        if path.is_dir() and path.name.startswith(prefix) and predicate(path)
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime).resolve()


def graph_suffix(path: Path, prefix: str) -> str | None:
    name = path.name
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix) :]
    return suffix or None


def discover_workflow_profile_dir(
    workspace: Path,
    explicit_profile_dir: Path | None = None,
    *,
    graph_dir: Path | None = None,
) -> Path | None:
    if explicit_profile_dir:
        path = explicit_profile_dir.resolve()
        return path if has_graph_community_reports(path) else None
    if graph_dir:
        suffix = graph_suffix(graph_dir, "graph_v03_consolidation_")
        if suffix:
            matched = workspace / f"graph_v03_profile_communities_{suffix}"
            if has_graph_community_reports(matched):
                return matched.resolve()
    for name in ("graph_v03_profile_communities", "graph_profile_communities"):
        path = workspace / name
        if has_graph_community_reports(path):
            return path.resolve()
    latest = newest_valid_child_dir(workspace, "graph_v03_profile_communities_", has_graph_community_reports)
    if latest:
        return latest
    return None


def write_workflow_artifacts(
    *,
    workspace: Path,
    run_id: str,
    command: str,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    output_dir = workflow_output_dir(workspace, run_id)
    supplied_boundary = dict(manifest.get("boundary", {}))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "workflow_command": command,
        "run_id": run_id,
        "workspace": str(workspace.resolve()),
        "output_dir": str(output_dir.resolve()),
        **manifest,
    }
    manifest["boundary"] = {**BOUNDARY_FLAGS, **supplied_boundary}
    manifest_path = output_dir / "workflow_manifest.json"
    report_path = output_dir / "workflow_report.md"
    manifest["outputs"] = {
        **manifest.get("outputs", {}),
        "workflow_manifest": str(manifest_path),
        "workflow_report": str(report_path),
    }
    write_json(manifest_path, manifest)
    write_text(report_path, render_workflow_report(manifest))
    return manifest


def render_workflow_report(manifest: dict[str, Any]) -> str:
    step_lines = []
    for step in manifest.get("steps", []):
        status = step.get("status", "unknown")
        name = step.get("name", "step")
        output = step.get("output_dir") or step.get("manifest") or ""
        step_lines.append(f"- `{name}`: `{status}` {output}")
    if not step_lines:
        step_lines = ["- none"]
    boundary = manifest.get("boundary", {})
    boundary_lines = [f"- {key}: `{value}`" for key, value in sorted(boundary.items())]
    return "\n".join(
        [
            f"# Workflow Report: {manifest.get('workflow_command')}",
            "",
            f"- run_id: `{manifest.get('run_id')}`",
            f"- workspace: `{manifest.get('workspace')}`",
            f"- status: `{manifest.get('status', 'completed')}`",
            "",
            "## Steps",
            "",
            *step_lines,
            "",
            "## Boundary",
            "",
            *boundary_lines,
            "",
        ]
    )


def run_status(args: argparse.Namespace) -> dict[str, Any]:
    workspace = resolve_path(args.workspace)
    status = workspace_status(workspace)
    return write_workflow_artifacts(
        workspace=workspace,
        run_id=args.run_id or default_run_id("status"),
        command="status",
        manifest={"status": "completed", "workspace_status": status},
    )


def run_build_full(args: argparse.Namespace) -> dict[str, Any]:
    project_root = resolve_path(args.project_root)
    workspace = resolve_path(args.workspace)
    run_id = args.run_id or default_run_id("build-full")
    steps: list[dict[str, Any]] = []

    common_prebuild = [
        "--pre-build-route-mode",
        args.pre_build_route_mode,
        "--proposal-provider",
        args.proposal_provider,
        "--duplicate-policy",
        args.duplicate_policy,
    ]
    if getattr(args, "api_mode", None):
        common_prebuild.extend(["--api-mode", args.api_mode])
    if getattr(args, "provider_profile", None):
        common_prebuild.extend(["--provider-profile", args.provider_profile])
    if getattr(args, "fallback_provider_profile", None):
        common_prebuild.extend(["--fallback-provider-profile", args.fallback_provider_profile])
    if args.allow_live_api:
        common_prebuild.append("--allow-live-api")
    if args.max_items is not None:
        common_prebuild.extend(["--max-items", str(args.max_items)])
    common_prebuild.extend(["--provider-concurrency", str(args.provider_concurrency)])

    step1_args = [
        "--project-root",
        str(project_root),
        "--workspace",
        str(workspace),
        "--run-scope",
        args.s1_run_scope,
        *common_prebuild,
    ]
    if args.workspace_id:
        step1_args.extend(["--workspace-id", args.workspace_id])
    if args.modeled_user_id:
        step1_args.extend(["--modeled-subject-id", args.modeled_user_id])
    steps.append({"name": "s1_build", "status": "running", "command": command_row("tools.step1.step1_build_runner", step1_args)})
    steps[-1].update({"status": "completed", "result": run_module("tools.step1.step1_build_runner", step1_args, cwd=project_root)})

    step1_index_args = [
        "--project-root",
        str(project_root),
        "--workspace",
        str(workspace),
        "--run-scope",
        args.s1_index_scope,
        "--index-mode",
        args.s1_index_mode,
        "--duplicate-policy",
        args.duplicate_policy,
        "--embedding-backend",
        args.s1_embedding_backend,
        "--embedding-model",
        args.s1_embedding_model,
        "--embedding-dimension",
        str(args.s1_embedding_dimension),
    ]
    if args.workspace_id:
        step1_index_args.extend(["--workspace-id", args.workspace_id])
    if args.allow_non_lcoral_for_tests:
        step1_index_args.append("--allow-non-lcoral-for-tests")
    steps.append({"name": "s1_index", "status": "running", "command": command_row("tools.step1.step1_index_runner", step1_index_args)})
    steps[-1].update({"status": "completed", "result": run_module("tools.step1.step1_index_runner", step1_index_args, cwd=project_root)})

    step2_args = [
        "--workspace",
        str(workspace),
        "--duplicate-policy",
        args.duplicate_policy,
        *common_prebuild,
    ]
    if args.workspace_id:
        step2_args.extend(["--workspace-id", args.workspace_id])
    if args.modeled_user_id:
        step2_args.extend(["--modeled-user-id", args.modeled_user_id])
    if args.target_participant:
        step2_args.extend(["--target-participant", args.target_participant])
    steps.append({"name": "s2_build", "status": "running", "command": command_row("tools.step2.step2_build_runner", step2_args)})
    steps[-1].update({"status": "completed", "result": run_module("tools.step2.step2_build_runner", step2_args, cwd=project_root)})

    step2_index_args = [
        "--workspace",
        str(workspace),
        "--duplicate-policy",
        args.duplicate_policy,
        "--device",
        args.s2_index_device,
        "--embedding-backend",
        args.s2_embedding_backend,
    ]
    if args.workspace_id:
        step2_index_args.extend(["--workspace-id", args.workspace_id])
    if args.modeled_user_id:
        step2_index_args.extend(["--modeled-user-id", args.modeled_user_id])
    steps.append({"name": "s2_index", "status": "running", "command": command_row("tools.step2.step2_user_model_index_runner", step2_index_args)})
    steps[-1].update({"status": "completed", "result": run_module("tools.step2.step2_user_model_index_runner", step2_index_args, cwd=project_root)})

    return write_workflow_artifacts(
        workspace=workspace,
        run_id=run_id,
        command="build-full",
        manifest={"status": "completed", "steps": steps, "workspace_status": workspace_status(workspace)},
    )


def run_build_graph(args: argparse.Namespace) -> dict[str, Any]:
    project_root = resolve_path(args.project_root)
    workspace = resolve_path(args.workspace)
    run_id = args.run_id or default_run_id("build-graph")
    suffix = args.output_suffix or short_hash(run_id)
    construction_dir = workspace / "graph_v03_construction"
    extraction_dir = workspace / f"graph_v03_extraction_{suffix}"
    consolidation_dir = workspace / f"graph_v03_consolidation_{suffix}"
    networkx_dir = workspace / f"graph_v03_networkx_utility_{suffix}"
    profile_dir = workspace / f"graph_v03_profile_communities_{suffix}"
    quality_dir = workspace / f"graph_v03_final_quality_gate_{suffix}"
    visual_dir = workspace / f"graph_v031_visual_review_{suffix}"

    steps: list[dict[str, Any]] = []
    construction = build_graph_construction_packets(workspace, construction_dir)
    steps.append({"name": "graph_construction_packets", "status": "completed", "manifest": construction})
    route = build_graph_package_routes(workspace, input_packets_path=construction_dir / "graph_construction_packets.jsonl", output_dir=construction_dir)
    steps.append({"name": "route_graph_extraction", "status": "completed", "manifest": route})
    extraction = build_graph_relation_candidates(
        workspace,
        project_root=project_root,
        input_packets_path=construction_dir / "graph_construction_packets.jsonl",
        output_dir=extraction_dir,
        provider=args.provider,
        api_mode=args.api_mode,
        allow_live_api=args.allow_live_api,
        provider_profile=getattr(args, "provider_profile", None),
        fallback_provider_profile=getattr(args, "fallback_provider_profile", None),
        route_decisions_path=construction_dir / "graph_route_decisions.jsonl",
        max_items=args.max_items,
        duplicate_policy=args.duplicate_policy,
        provider_concurrency=args.provider_concurrency,
        relation_schema_candidates_path=Path(args.relation_schema_candidates).resolve() if args.relation_schema_candidates else None,
    )
    steps.append({"name": "provider_graph_extraction", "status": "completed", "manifest": extraction})
    consolidation = consolidate_graph_candidates(workspace, extraction_dir=extraction_dir, output_dir=consolidation_dir)
    steps.append({"name": "graph_candidate_consolidation", "status": "completed", "manifest": consolidation})
    networkx_manifest = run_networkx_utility(workspace, graph_dir=consolidation_dir, output_dir=networkx_dir)
    steps.append({"name": "networkx_graph_utility", "status": "completed", "manifest": networkx_manifest})
    profile = run_graph_profile_community_builder(
        workspace,
        project_root=project_root,
        graph_dir=consolidation_dir,
        output_dir=profile_dir,
        projection=args.projection,
        community_report_provider=args.community_report_provider,
        allow_live_api=args.allow_live_api,
        max_provider_reports=args.max_provider_reports,
    )
    steps.append({"name": "graph_profile_community", "status": "completed", "manifest": profile})

    summary, samples, report = compute_quality_gate(consolidation_dir, sample_limit=args.quality_sample_limit)
    write_json(quality_dir / "graph_final_quality_gate_summary.json", summary)
    write_jsonl(quality_dir / "graph_final_quality_gate_samples.jsonl", samples)
    write_text(quality_dir / "graph_final_quality_gate_report.md", report)
    quality = {
        "output_dir": str(quality_dir),
        "summary": summary,
        "outputs": {
            "summary": str(quality_dir / "graph_final_quality_gate_summary.json"),
            "samples": str(quality_dir / "graph_final_quality_gate_samples.jsonl"),
            "report": str(quality_dir / "graph_final_quality_gate_report.md"),
        },
    }
    steps.append({"name": "graph_final_quality_gate", "status": "completed", "manifest": quality})

    visual = None
    if args.visual:
        visual = run_visual_review_bundle(
            workspace,
            graph_dir=consolidation_dir,
            networkx_dir=networkx_dir,
            profile_dir=profile_dir,
            output_dir=visual_dir,
            projection=args.projection,
        )
        steps.append({"name": "graph_visual_review", "status": "completed", "manifest": visual})

    manifest = {
        "status": "completed",
        "steps": steps,
        "outputs": {
            "graph_construction_dir": str(construction_dir),
            "graph_extraction_dir": str(extraction_dir),
            "graph_consolidation_dir": str(consolidation_dir),
            "networkx_dir": str(networkx_dir),
            "profile_dir": str(profile_dir),
            "quality_dir": str(quality_dir),
            "visual_dir": str(visual_dir) if visual else None,
        },
        "provider_policy": {
            "provider": args.provider,
            "allow_live_api": bool(args.allow_live_api),
            "mock_regex_baseline_is_main_fidelity_path": False,
        },
        "boundary": {
            "graph_is_not_proof": True,
            "write_permission": False,
        },
    }
    return write_workflow_artifacts(workspace=workspace, run_id=run_id, command="build-graph", manifest=manifest)


def run_query(args: argparse.Namespace) -> dict[str, Any]:
    project_root = resolve_path(args.project_root)
    workspace = resolve_path(args.workspace)
    run_id = args.run_id or default_run_id("query")
    output_dir = Path(args.output).resolve() if args.output else workflow_output_dir(workspace, run_id) / "query"
    graph_dir = discover_workflow_graph_dir(workspace, Path(args.v03_graph_dir).resolve() if args.v03_graph_dir else None)
    profile_dir = discover_workflow_profile_dir(
        workspace,
        Path(args.v03_graph_profile_dir).resolve() if args.v03_graph_profile_dir else None,
        graph_dir=graph_dir,
    )

    query_args = [
        "--workspace",
        str(workspace),
        "--output",
        str(output_dir),
        "--graph-context-mode",
        "v03" if graph_dir else "disabled",
        "--graph-rerank-mode",
        args.graph_rerank_mode,
        "--graph-community-mode",
        args.graph_community_mode,
        "--embedding-backend",
        args.embedding_backend,
    ]
    if args.questions:
        query_args.extend(["--questions", str(Path(args.questions).resolve())])
    if args.question:
        query_args.extend(["--question", args.question])
    if graph_dir:
        query_args.extend(["--v03-graph-dir", str(graph_dir)])
    if profile_dir:
        query_args.extend(["--v03-graph-profile-dir", str(profile_dir)])
    if args.v03_graph_community_assists:
        query_args.extend(["--v03-graph-community-assists", str(Path(args.v03_graph_community_assists).resolve())])
    if args.device:
        query_args.extend(["--device", args.device])

    step = {"name": "s2_query", "status": "running", "command": command_row("tools.step2.query_runner", query_args)}
    step.update({"status": "completed", "result": run_module("tools.step2.query_runner", query_args, cwd=project_root)})
    return write_workflow_artifacts(
        workspace=workspace,
        run_id=run_id,
        command="query",
        manifest={
            "status": "completed",
            "steps": [step],
            "outputs": {"query_output_dir": str(output_dir)},
            "query_context": {
                "graph_dir": str(graph_dir) if graph_dir else None,
                "graph_profile_dir": str(profile_dir) if profile_dir else None,
                "graph_context_mode": "v03" if graph_dir else "disabled",
            },
        },
    )


def run_incremental(args: argparse.Namespace) -> dict[str, Any]:
    workspace = resolve_path(args.workspace)
    run_id = args.run_id or default_run_id(f"incremental-{args.incremental_command}")
    if args.incremental_command == "prepare":
        result = prepare_full_review_workflow_package(
            workspace=workspace,
            package_dir=Path(args.package_dir).resolve() if args.package_dir else None,
            reset=args.reset,
            port=args.port,
        )
        status = "waiting_for_human_review"
    elif args.incremental_command == "finalize":
        result = finalize_full_review_workflow_package(
            workspace=workspace,
            package_dir=Path(args.package_dir).resolve() if args.package_dir else None,
            auto_review_for_test_mode=args.auto_review_for_test,
        )
        status = result.get("acceptance_status", "completed")
    elif args.incremental_command == "smoke":
        result = run_v04_incremental_workflow_smoke(
            workspace=workspace,
            package_dir=Path(args.package_dir).resolve() if args.package_dir else None,
            reset=args.reset,
        )
        status = result.get("acceptance_status", "completed")
    else:  # pragma: no cover
        raise ValueError(f"Unsupported incremental command: {args.incremental_command}")

    return write_workflow_artifacts(
        workspace=workspace,
        run_id=run_id,
        command=f"incremental {args.incremental_command}",
        manifest={
            "status": status,
            "steps": [{"name": f"incremental_{args.incremental_command}", "status": status, "manifest": result}],
            "outputs": {
                "package_dir": result.get("package_dir"),
                **(result.get("outputs") or {}),
            },
            "boundary": result.get("boundary", {}),
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="v0.41 thin workflow runner over existing tools.")
    parser.add_argument("--project-root", default=".")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--workspace", required=True)
    status.add_argument("--run-id", default=None)
    status.set_defaults(func=run_status)

    build_full = subparsers.add_parser("build-full")
    build_full.add_argument("--workspace", required=True)
    build_full.add_argument("--workspace-id", default=None)
    build_full.add_argument("--modeled-user-id", default=None)
    build_full.add_argument("--target-participant", default=None)
    build_full.add_argument("--run-id", default=None)
    build_full.add_argument("--s1-run-scope", default="full_s1_build")
    build_full.add_argument("--s1-index-scope", default="evidence_plus_memory_index")
    build_full.add_argument("--s1-index-mode", default="lexical")
    build_full.add_argument("--s1-embedding-backend", default="hash")
    build_full.add_argument("--s1-embedding-model", default="deterministic-hash-embedding-v0.1")
    build_full.add_argument("--s1-embedding-dimension", type=int, default=64)
    build_full.add_argument("--s2-index-device", default="cpu", choices=["auto", "cpu", "cuda"])
    build_full.add_argument("--s2-embedding-backend", default="hash", choices=["hash", "qwen_local"])
    build_full.add_argument("--pre-build-route-mode", default="route_and_propose", choices=["none", "route_only", "route_and_propose"])
    build_full.add_argument("--proposal-provider", default="mock", choices=["mock", "external_jsonl", "openai"])
    build_full.add_argument("--api-mode", default=None, choices=["responses", "chat_completions"])
    build_full.add_argument("--provider-profile", default=None)
    build_full.add_argument("--fallback-provider-profile", default=None)
    build_full.add_argument("--allow-live-api", action="store_true")
    build_full.add_argument("--max-items", type=int, default=None)
    build_full.add_argument("--provider-concurrency", type=int, default=1)
    build_full.add_argument("--duplicate-policy", default="fail", choices=["fail", "overwrite_generated"])
    build_full.add_argument("--allow-non-lcoral-for-tests", action="store_true")
    build_full.set_defaults(func=run_build_full)

    build_graph = subparsers.add_parser("build-graph")
    build_graph.add_argument("--workspace", required=True)
    build_graph.add_argument("--run-id", default=None)
    build_graph.add_argument("--output-suffix", default=None)
    build_graph.add_argument("--provider", default="mock_regex_baseline", choices=["mock", "mock_regex_baseline", "external_jsonl", "openai"])
    build_graph.add_argument("--api-mode", default=None, choices=["responses", "chat_completions"])
    build_graph.add_argument("--provider-profile", default=None)
    build_graph.add_argument("--fallback-provider-profile", default=None)
    build_graph.add_argument("--allow-live-api", action="store_true")
    build_graph.add_argument("--max-items", type=int, default=None)
    build_graph.add_argument("--provider-concurrency", type=int, default=1)
    build_graph.add_argument("--duplicate-policy", default="fail", choices=["fail", "overwrite_generated"])
    build_graph.add_argument("--relation-schema-candidates", default=None)
    build_graph.add_argument("--projection", default="review_aware_graph", choices=["full_candidate_graph", "review_aware_graph", "stable_core_graph"])
    build_graph.add_argument("--community-report-provider", default="extractive", choices=["extractive", "openai", "external_jsonl"])
    build_graph.add_argument("--max-provider-reports", type=int, default=0)
    build_graph.add_argument("--quality-sample-limit", type=int, default=12)
    build_graph.add_argument("--visual", action="store_true")
    build_graph.set_defaults(func=run_build_graph)

    query = subparsers.add_parser("query")
    query.add_argument("--workspace", required=True)
    query.add_argument("--run-id", default=None)
    query.add_argument("--output", default=None)
    query.add_argument("--questions", default=None)
    query.add_argument("--question", default=None)
    query.add_argument("--v03-graph-dir", default=None)
    query.add_argument("--v03-graph-profile-dir", default=None)
    query.add_argument("--v03-graph-community-assists", default=None)
    query.add_argument("--graph-rerank-mode", default="feature", choices=["disabled", "feature", "cross_encoder", "auto"])
    query.add_argument("--graph-community-mode", default="auto")
    query.add_argument("--embedding-backend", default="hash", choices=["hash", "qwen_local"])
    query.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    query.set_defaults(func=run_query)

    incremental = subparsers.add_parser("incremental")
    incremental.add_argument("--workspace", required=True)
    incremental.add_argument("--run-id", default=None)
    incremental.add_argument("--package-dir", default=None)
    incremental_sub = incremental.add_subparsers(dest="incremental_command", required=True)
    prepare = incremental_sub.add_parser("prepare")
    prepare.add_argument("--reset", action="store_true")
    prepare.add_argument("--port", type=int, default=8765)
    prepare.add_argument("--auto-review-for-test", action="store_true", help=argparse.SUPPRESS)
    prepare.set_defaults(func=run_incremental)
    finalize = incremental_sub.add_parser("finalize")
    finalize.add_argument("--auto-review-for-test", action="store_true")
    finalize.add_argument("--reset", action="store_true", help=argparse.SUPPRESS)
    finalize.add_argument("--port", type=int, default=8765, help=argparse.SUPPRESS)
    finalize.set_defaults(func=run_incremental)
    smoke = incremental_sub.add_parser("smoke")
    smoke.add_argument("--reset", action="store_true")
    smoke.add_argument("--auto-review-for-test", action="store_true", help=argparse.SUPPRESS)
    smoke.add_argument("--port", type=int, default=8765, help=argparse.SUPPRESS)
    smoke.set_defaults(func=run_incremental)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = args.func(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
