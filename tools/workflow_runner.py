"""Thin workflow entrypoint over the existing memory/graph toolchain.

This module intentionally orchestrates existing runners instead of replacing
their implementation. It writes a workflow-level manifest/report so users and
agents can find the right artifacts without re-reading every reference doc.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

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
from tools.providers.provider_profiles import ProviderProfile, resolve_provider_profile_bundle


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


def env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def profile_prefix(profile_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(profile_id or "")).strip("_").upper()
    return f"PSML_PROVIDER_{cleaned}" if cleaned else "PSML_PROVIDER"


def parse_env_file(path: Path) -> tuple[dict[str, str], list[str]]:
    values: dict[str, str] = {}
    warnings: list[str] = []
    if not path.exists():
        return values, [f"env_file_missing:{path}"]
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            warnings.append(f"env_line_ignored:{line_no}")
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            warnings.append(f"env_empty_key:{line_no}")
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        values[key] = value
    return values, warnings


@contextmanager
def temporary_env(overrides: dict[str, str]):
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def url_host(value: str) -> str:
    parsed = urlparse(str(value or ""))
    return parsed.netloc or ""


def is_official_openai_base_url(value: str) -> bool:
    return url_host(value).lower() == "api.openai.com"


def profile_key_source(profile_id: str, env_values: dict[str, str], profile: ProviderProfile | None) -> dict[str, Any]:
    if profile is None:
        return {"source": "none", "env_var": None, "env_var_set": False, "direct_in_env_file": False}
    prefix = profile_prefix(profile_id)
    direct_key = env_values.get(f"{prefix}_API_KEY", "")
    key_env = env_values.get(f"{prefix}_API_KEY_ENV") or os.environ.get(f"{prefix}_API_KEY_ENV")
    if direct_key:
        return {
            "source": "direct_profile_api_key",
            "env_var": None,
            "env_var_set": False,
            "direct_in_env_file": True,
        }
    if key_env:
        return {
            "source": "env_indirection",
            "env_var": key_env,
            "env_var_set": bool(os.environ.get(key_env)),
            "direct_in_env_file": False,
        }
    if profile_id == "legacy_openai":
        direct_legacy = bool(env_values.get("OPENAI_API_KEY", ""))
        return {
            "source": "legacy_openai_api_key",
            "env_var": "OPENAI_API_KEY",
            "env_var_set": bool(os.environ.get("OPENAI_API_KEY")),
            "direct_in_env_file": direct_legacy,
        }
    return {"source": "missing", "env_var": None, "env_var_set": False, "direct_in_env_file": False}


def profile_summary(role: str, profile: ProviderProfile | None, env_values: dict[str, str]) -> dict[str, Any] | None:
    if profile is None:
        return None
    key_source = profile_key_source(profile.profile_id, env_values, profile)
    return {
        "role": role,
        "profile_id": profile.profile_id,
        "provider": profile.provider,
        "api_mode": profile.api_mode,
        "weak_model": profile.weak_model,
        "strong_model": profile.strong_model,
        "base_url_host": url_host(profile.base_url),
        "base_url_is_official_openai": is_official_openai_base_url(profile.base_url),
        "api_key_present": bool(profile.api_key),
        "api_key_source": key_source["source"],
        "api_key_env_var": key_source["env_var"],
        "api_key_env_var_set": key_source["env_var_set"],
        "api_key_directly_in_env_file": key_source["direct_in_env_file"],
        "user_agent_present": bool(profile.user_agent),
        "max_retries": profile.max_retries,
        "retry_base_seconds": profile.retry_base_seconds,
        "retry_max_seconds": profile.retry_max_seconds,
    }


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
    if manifest.get("workflow_command") == "doctor" and manifest.get("provider_config_doctor"):
        return render_provider_doctor_report(manifest)
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


def render_provider_doctor_report(manifest: dict[str, Any]) -> str:
    doctor = manifest["provider_config_doctor"]
    selection = doctor.get("profile_selection") or {}
    counts = doctor.get("diagnostic_counts") or {}
    diagnostics = doctor.get("diagnostics") or []
    profiles = doctor.get("profiles") or {}
    live_api = doctor.get("live_api") or {}

    def profile_lines(profile: dict[str, Any] | None) -> list[str]:
        if not profile:
            return ["- none"]
        return [
            f"- profile_id: `{profile.get('profile_id')}`",
            f"- provider: `{profile.get('provider')}`",
            f"- api_mode: `{profile.get('api_mode')}`",
            f"- weak_model: `{profile.get('weak_model')}`",
            f"- strong_model: `{profile.get('strong_model')}`",
            f"- base_url_host: `{profile.get('base_url_host')}`",
            f"- official_openai_base_url: `{profile.get('base_url_is_official_openai')}`",
            f"- api_key_present: `{profile.get('api_key_present')}`",
            f"- api_key_source: `{profile.get('api_key_source')}`",
            f"- api_key_env_var: `{profile.get('api_key_env_var')}`",
            f"- direct_key_in_env_file: `{profile.get('api_key_directly_in_env_file')}`",
        ]

    diagnostic_lines = [
        f"- `{row.get('severity')}` `{row.get('check_id')}`: {row.get('message')}"
        + (f" Remediation: {row.get('remediation')}" if row.get("remediation") else "")
        for row in diagnostics
    ] or ["- none"]

    return "\n".join(
        [
            "# Provider / Config Doctor Report",
            "",
            f"- run_id: `{manifest.get('run_id')}`",
            f"- status: `{doctor.get('diagnostic_status')}`",
            f"- env_file: `{doctor.get('env_file')}`",
            f"- provider_resolved: `{doctor.get('provider_resolved')}`",
            f"- live_api_cli: `{live_api.get('allow_live_api_cli')}`",
            f"- live_api_env: `{live_api.get('allow_live_api_env')}`",
            f"- live_probe: `{live_api.get('doctor_performed_live_probe')}`",
            "",
            "## Profile Selection",
            "",
            f"- provider_profile_id: `{selection.get('provider_profile_id')}`",
            f"- provider_fallback_profile_id: `{selection.get('provider_fallback_profile_id')}`",
            f"- provider_fallback_enabled: `{selection.get('provider_fallback_enabled')}`",
            f"- api_mode: `{selection.get('api_mode')}`",
            f"- weak_model: `{selection.get('weak_model')}`",
            f"- strong_model: `{selection.get('strong_model')}`",
            "",
            "## Primary Profile",
            "",
            *profile_lines(profiles.get("primary")),
            "",
            "## Fallback Profile",
            "",
            *profile_lines(profiles.get("fallback")),
            "",
            "## Diagnostic Counts",
            "",
            f"- error: `{counts.get('error', 0)}`",
            f"- warning: `{counts.get('warning', 0)}`",
            f"- pass: `{counts.get('pass', 0)}`",
            f"- info: `{counts.get('info', 0)}`",
            "",
            "## Diagnostics",
            "",
            *diagnostic_lines,
            "",
            "## Boundary",
            "",
            "- Doctor does not call live provider APIs.",
            "- Doctor does not print API key values.",
            "- Fallback diagnostics cover provider/service fallback only, not output-quality fallback.",
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


def build_provider_diagnostics(
    *,
    project_root: Path,
    env_file: Path,
    provider: str,
    api_mode: str | None,
    provider_profile: str | None,
    fallback_provider_profile: str | None,
    allow_live_api: bool,
) -> dict[str, Any]:
    env_values, env_warnings = parse_env_file(env_file)
    diagnostics: list[dict[str, Any]] = []

    def add_diag(severity: str, check_id: str, message: str, remediation: str = "") -> None:
        diagnostics.append(
            {
                "severity": severity,
                "check_id": check_id,
                "message": message,
                "remediation": remediation,
            }
        )

    with temporary_env(env_values):
        resolved_provider = provider or os.environ.get("OPENAI_PROVIDER") or "openai"
        live_gate_env = env_truthy(os.environ.get("ALLOW_LIVE_API"))
        try:
            bundle = resolve_provider_profile_bundle(
                provider=resolved_provider,
                api_mode=api_mode,
                weak_model=None,
                strong_model=None,
                provider_profile=provider_profile,
                fallback_provider_profile=fallback_provider_profile,
            )
            bundle_error = None
        except Exception as exc:
            bundle = None
            bundle_error = f"{type(exc).__name__}: {exc}"

        if env_warnings:
            for warning in env_warnings:
                severity = "warning" if warning.startswith("env_line_ignored") else "info"
                add_diag(severity, "env_parse", warning, "检查 .env 是否存在拼写或格式问题。")
        if not env_file.exists():
            add_diag("warning", "env_file_missing", f"Env file not found: {env_file}", "如需 live provider，复制 .env.example 为 .env 并配置本地变量。")
        else:
            add_diag("info", "env_file_loaded", f"Env file parsed: {env_file}", "")

        direct_secret_keys = sorted(
            key
            for key, value in env_values.items()
            if key.endswith("API_KEY") and value.strip()
        )
        for key in direct_secret_keys:
            add_diag(
                "warning",
                "direct_api_key_in_env_file",
                f"{key} is set directly in the env file.",
                "更推荐使用 *_API_KEY_ENV 指向本机环境变量，避免复制或公开真实 key。",
            )

        if bundle_error:
            add_diag("error", "provider_bundle_resolution_failed", bundle_error, "检查 provider profile 名称、api_mode 和数值格式。")
            return {
                "project_root": str(project_root),
                "env_file": str(env_file),
                "env_file_exists": env_file.exists(),
                "provider_requested": provider,
                "provider_resolved": resolved_provider,
                "bundle_error": bundle_error,
                "diagnostics": diagnostics,
                "diagnostic_status": "fail",
            }

        assert bundle is not None
        primary = profile_summary("primary", bundle.primary, env_values)
        fallback = profile_summary("fallback", bundle.fallback, env_values)

        if resolved_provider == "openai":
            if not bundle.primary_profile_id:
                add_diag("error", "primary_profile_missing", "OpenAI-compatible provider has no primary profile.", "设置 PSML_PROVIDER_DEFAULT 或 --provider-profile。")
            elif bundle.primary_profile_id == "legacy_openai":
                add_diag(
                    "warning",
                    "legacy_openai_profile",
                    "Provider resolution is using legacy OPENAI_* fields.",
                    "如需便宜 provider 主路，设置 PSML_PROVIDER_DEFAULT=claude_secondary。",
                )
            else:
                add_diag("pass", "primary_profile_selected", f"Primary provider profile: {bundle.primary_profile_id}", "")

            if bundle.primary and not bundle.primary.api_key:
                add_diag("error", "primary_api_key_missing", f"Primary profile {bundle.primary.profile_id} has no API key.", "设置 profile 的 *_API_KEY_ENV 或本机环境变量。")
            elif bundle.primary:
                add_diag("pass", "primary_api_key_present", f"Primary profile {bundle.primary.profile_id} has an API key available.", "")

            if primary and primary["base_url_is_official_openai"]:
                severity = "warning" if bundle.primary_profile_id != "legacy_openai" else "info"
                add_diag(
                    severity,
                    "primary_official_openai_base_url",
                    f"Primary profile {bundle.primary_profile_id} points to api.openai.com.",
                    "如果目标是低成本 provider，确认默认 profile 是否应为 claude_secondary 或其它兼容网关。",
                )
            elif primary:
                add_diag("pass", "primary_non_openai_base_url", f"Primary base URL host: {primary['base_url_host']}", "")

            if bundle.primary and bundle.primary.api_mode == "responses" and "claude" in f"{bundle.weak_model} {bundle.strong_model}".lower():
                add_diag(
                    "warning",
                    "claude_model_with_responses_api_mode",
                    "Claude-like model name is paired with responses API mode.",
                    "OpenAI-compatible Claude gateways usually require chat_completions。",
                )
            if bundle.primary and bundle.primary.api_mode == "chat_completions":
                add_diag("pass", "chat_completions_mode", "API mode is chat_completions.", "")

            if bundle.primary and (not bundle.weak_model or not bundle.strong_model):
                add_diag("warning", "model_id_missing", "Weak or strong model id is empty after profile resolution.", "设置 *_WEAK_MODEL 和 *_STRONG_MODEL。")
            elif bundle.primary:
                add_diag("pass", "model_ids_resolved", f"Weak={bundle.weak_model}; strong={bundle.strong_model}", "")

            configured_fallback = fallback_provider_profile or os.environ.get("PSML_PROVIDER_FALLBACK") or os.environ.get("OPENAI_FALLBACK_PROVIDER_PROFILE") or ""
            if configured_fallback and not bundle.fallback_enabled:
                add_diag(
                    "warning",
                    "fallback_configured_but_disabled",
                    f"Fallback profile {configured_fallback} is configured but has no usable API key.",
                    "补齐 fallback profile key，或清空 fallback 配置避免误解。",
                )
            elif bundle.fallback_enabled:
                add_diag("pass", "fallback_enabled", f"Fallback provider profile enabled: {bundle.fallback_profile_id}", "")
            else:
                add_diag("info", "fallback_not_configured", "No provider fallback profile is enabled.", "这是可接受状态；当前默认是便宜 provider 主路。")
        else:
            add_diag("info", "non_openai_provider", f"Provider {resolved_provider} does not use OpenAI-compatible profile resolution.", "")

        if live_gate_env:
            add_diag("warning", "env_allows_live_api", "ALLOW_LIVE_API=true in env.", "确认这是本机私有配置，不要进入 public mirror。")
        elif allow_live_api:
            add_diag("info", "cli_allows_live_api", "--allow-live-api was supplied to doctor.", "Doctor still does not call a live API.")
        else:
            add_diag("pass", "live_api_not_auto_enabled", "Live API is not enabled by CLI or env.", "")

        public_risk_paths = [
            project_root / "users",
            project_root / "data",
            project_root / "external_references",
            project_root / "experiments",
            project_root / "reports",
        ]
        private_dirs_present = [str(path) for path in public_risk_paths if path.exists()]
        if private_dirs_present:
            add_diag("info", "private_work_dirs_present", "Private/generated working directories exist in this repo.", "这是私有主仓可接受；同步 public mirror 时必须排除。")
        else:
            add_diag("pass", "private_work_dirs_absent", "No private/generated working directories found at project root.", "")

        status = "fail" if any(row["severity"] == "error" for row in diagnostics) else "warn" if any(row["severity"] == "warning" for row in diagnostics) else "pass"
        return {
            "project_root": str(project_root),
            "env_file": str(env_file),
            "env_file_exists": env_file.exists(),
            "provider_requested": provider,
            "provider_resolved": resolved_provider,
            "live_api": {
                "allow_live_api_cli": bool(allow_live_api),
                "allow_live_api_env": live_gate_env,
                "doctor_performed_live_probe": False,
            },
            "legacy_openai": {
                "openai_provider": os.environ.get("OPENAI_PROVIDER"),
                "openai_api_mode": os.environ.get("OPENAI_API_MODE"),
                "openai_model_weak": os.environ.get("OPENAI_MODEL_WEAK"),
                "openai_model_strong": os.environ.get("OPENAI_MODEL_STRONG"),
                "openai_base_url_host": url_host(os.environ.get("OPENAI_BASE_URL", "")),
                "openai_api_key_present": bool(os.environ.get("OPENAI_API_KEY")),
                "openai_api_key_directly_in_env_file": bool(env_values.get("OPENAI_API_KEY", "")),
            },
            "profile_selection": {
                "provider_profile_arg": provider_profile,
                "fallback_provider_profile_arg": fallback_provider_profile,
                "psml_provider_default": os.environ.get("PSML_PROVIDER_DEFAULT"),
                "psml_provider_fallback": os.environ.get("PSML_PROVIDER_FALLBACK"),
                **bundle.manifest_fields(),
                "api_mode": bundle.api_mode,
                "weak_model": bundle.weak_model,
                "strong_model": bundle.strong_model,
            },
            "profiles": {
                "primary": primary,
                "fallback": fallback,
            },
            "diagnostics": diagnostics,
            "diagnostic_counts": {
                "error": sum(1 for row in diagnostics if row["severity"] == "error"),
                "warning": sum(1 for row in diagnostics if row["severity"] == "warning"),
                "pass": sum(1 for row in diagnostics if row["severity"] == "pass"),
                "info": sum(1 for row in diagnostics if row["severity"] == "info"),
            },
            "diagnostic_status": status,
        }


def run_doctor(args: argparse.Namespace) -> dict[str, Any]:
    project_root = resolve_path(args.project_root)
    workspace = resolve_path(args.workspace) if args.workspace else project_root
    env_file = resolve_path(args.env_file) if Path(args.env_file).is_absolute() else (project_root / args.env_file).resolve()
    doctor = build_provider_diagnostics(
        project_root=project_root,
        env_file=env_file,
        provider=args.provider,
        api_mode=args.api_mode,
        provider_profile=args.provider_profile,
        fallback_provider_profile=args.fallback_provider_profile,
        allow_live_api=args.allow_live_api,
    )
    return write_workflow_artifacts(
        workspace=workspace,
        run_id=args.run_id or default_run_id("doctor"),
        command="doctor",
        manifest={
            "status": doctor["diagnostic_status"],
            "provider_config_doctor": doctor,
            "steps": [
                {
                    "name": "provider_config_doctor",
                    "status": doctor["diagnostic_status"],
                    "manifest": {
                        "diagnostic_counts": doctor.get("diagnostic_counts"),
                        "profile_selection": doctor.get("profile_selection"),
                    },
                }
            ],
        },
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

    doctor = subparsers.add_parser("doctor")
    doctor.add_argument("--workspace", default=None)
    doctor.add_argument("--run-id", default=None)
    doctor.add_argument("--env-file", default=".env")
    doctor.add_argument("--provider", default=None, choices=["mock", "external_jsonl", "openai"])
    doctor.add_argument("--api-mode", default=None, choices=["responses", "chat_completions"])
    doctor.add_argument("--provider-profile", default=None)
    doctor.add_argument("--fallback-provider-profile", default=None)
    doctor.add_argument("--allow-live-api", action="store_true")
    doctor.set_defaults(func=run_doctor)

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
