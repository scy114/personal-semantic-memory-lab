"""Build incremental S1 candidate material from a v0.4 S0B batch.

This runner is the formal v0.4 S0B -> S1 incremental boundary. It consumes an
add-only S0B incremental batch, stages only that batch's evidence/text-unit
rows in a private workspace, runs S1 pre-build routing over the staged rows,
and emits incremental S1 candidate files.

It intentionally does not call tools.step1.step1_build_runner.run_build and it
does not write canonical memory/evidence files in the target workspace.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from tools.prebuild_routing import (
    DEFAULT_PRE_BUILD_ROUTE_MODE,
    DEFAULT_ROUTE_POLICY,
    DEFAULT_S1_PROPOSAL_PROFILE,
    PreBuildRoutingOptions,
    run_prebuild_routing,
)
from tools.step1.step1_build_runner import (
    BuildInputs,
    build_memory,
    build_s1_proposal_index,
    build_s1_reject_index,
    build_s1_route_skip_rows,
    load_s1_prebuild_proposal_rows,
    load_s1_prebuild_route_rows,
    validate_memory_result,
)


SCHEMA_VERSION = "maintenance.s1_incremental_build.v0.4"
MANIFEST_SCHEMA_VERSION = "maintenance.s1_incremental_build_manifest.v0.4"
REPORT_FILENAME = "s1_incremental_build_report.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def short_hash(value: str, length: int = 16) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:length]


def normalize(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def normalize_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def first_value(row: dict[str, Any], *field_names: str) -> str:
    for field_name in field_names:
        value = normalize(row.get(field_name))
        if value:
            return value
    return ""


def raw_row(batch_row: dict[str, Any]) -> dict[str, Any]:
    row = batch_row.get("raw_row") or {}
    if not isinstance(row, dict):
        raise ValueError("S0B incremental batch row raw_row must be an object")
    return row


def evidence_ref_from(row: dict[str, Any], batch_row: dict[str, Any]) -> str:
    evidence_refs = batch_row.get("evidence_refs") or []
    return (
        first_value(row, "evidence_ref", "canonical_evidence_ref")
        or (normalize(evidence_refs[0]) if evidence_refs else "")
        or first_value(batch_row, "object_id")
    )


def default_display_ref(evidence_ref: str) -> str:
    return evidence_ref.rsplit(":", 1)[-1] if ":" in evidence_ref else evidence_ref


def normalize_evidence_row(batch_row: dict[str, Any], *, modeled_subject_id: str) -> dict[str, Any]:
    row = raw_row(batch_row)
    evidence_ref = evidence_ref_from(row, batch_row)
    text = first_value(row, "text", "content", "evidence_quote")
    if not evidence_ref:
        raise ValueError("S0B incremental evidence row is missing evidence_ref")
    if not text:
        raise ValueError(f"S0B incremental evidence row is missing text: {evidence_ref}")
    source_id = first_value(row, "source_id") or first_value(batch_row, "source_id")
    raw_source_id = first_value(row, "raw_source_id") or first_value(batch_row, "source_id")
    speaker = first_value(row, "speaker", "subject_id", "participant") or modeled_subject_id
    source_specific_ref = first_value(row, "source_specific_ref", "display_ref") or default_display_ref(evidence_ref)
    display_ref = first_value(row, "display_ref") or source_specific_ref or evidence_ref
    target_subject_ids = row.get("target_subject_ids") or ([modeled_subject_id] if modeled_subject_id else [])
    subject_ids = row.get("subject_ids") or [first_value(row, "subject_id", "speaker") or speaker]
    normalized = {
        **row,
        "schema_version": row.get("schema_version") or "step1.raw_evidence.v0.1",
        "evidence_ref": evidence_ref,
        "canonical_evidence_ref": first_value(row, "canonical_evidence_ref") or evidence_ref,
        "source_id": source_id,
        "raw_source_id": raw_source_id,
        "source_version": first_value(row, "source_version") or first_value(batch_row, "source_version"),
        "source_specific_ref": source_specific_ref,
        "display_ref": display_ref,
        "text": text,
        "speaker": speaker,
        "participant": first_value(row, "participant") or speaker,
        "subject_id": first_value(row, "subject_id") or speaker,
        "subject_ids": normalize_unique(subject_ids),
        "target_subject_ids": normalize_unique(target_subject_ids),
        "target_participant": first_value(row, "target_participant") or modeled_subject_id,
        "subject_role": row.get("subject_role") or ("target" if speaker == modeled_subject_id else "context"),
        "source_type": row.get("source_type") or "incremental_s0b_batch",
        "section_type": row.get("section_type") or "conversation",
        "retrieval_policy": row.get("retrieval_policy") or "default_retrieval",
        "s2_policy": row.get("s2_policy") or "candidate_allowed",
        "s1_storage_policy": row.get("s1_storage_policy") or "ordinary_evidence",
        "privacy_class": row.get("privacy_class") or "unknown",
        "maintenance_status": row.get("maintenance_status") or batch_row.get("maintenance_status") or "active",
    }
    locator = normalized.get("locator")
    if not isinstance(locator, dict):
        normalized["locator"] = {
            "kind": "s0b_incremental_batch",
            "display_ref": display_ref,
            "source_file": batch_row.get("source_path") or "",
        }
    return normalized


def normalize_text_unit_row(batch_row: dict[str, Any], *, modeled_subject_id: str) -> dict[str, Any]:
    row = raw_row(batch_row)
    evidence_ref = evidence_ref_from(row, batch_row)
    text_unit_id = first_value(row, "text_unit_id", "id") or first_value(batch_row, "s0b_unit_id", "object_id")
    text = first_value(row, "text", "content") or evidence_ref
    if not text_unit_id:
        text_unit_id = "s0b:incremental:" + short_hash(evidence_ref + ":" + text)
    source_id = first_value(row, "source_id") or first_value(batch_row, "source_id")
    speaker = first_value(row, "speaker", "subject_id", "participant") or modeled_subject_id
    target_subject_ids = row.get("target_subject_ids") or ([modeled_subject_id] if modeled_subject_id else [])
    subject_ids = row.get("subject_ids") or [first_value(row, "subject_id", "speaker") or speaker]
    return {
        **row,
        "schema_version": row.get("schema_version") or "s0b.text_unit.v0.1",
        "text_unit_id": text_unit_id,
        "unit_type": row.get("unit_type") or "sentence",
        "raw_source_id": first_value(row, "raw_source_id") or first_value(batch_row, "source_id"),
        "source_id": source_id,
        "source_version": first_value(row, "source_version") or first_value(batch_row, "source_version"),
        "evidence_ref": evidence_ref,
        "canonical_evidence_ref": first_value(row, "canonical_evidence_ref") or evidence_ref,
        "text": text,
        "speaker": speaker,
        "subject_id": first_value(row, "subject_id") or speaker,
        "subject_ids": normalize_unique(subject_ids),
        "target_subject_ids": normalize_unique(target_subject_ids),
        "target_participant": first_value(row, "target_participant") or modeled_subject_id,
        "subject_role": row.get("subject_role") or ("target" if speaker == modeled_subject_id else "context"),
        "section_type": row.get("section_type") or "conversation",
        "retrieval_policy": row.get("retrieval_policy") or "default_retrieval",
        "s2_policy": row.get("s2_policy") or "candidate_allowed",
        "maintenance_status": row.get("maintenance_status") or batch_row.get("maintenance_status") or "active",
    }


def synthesize_text_unit_from_evidence(row: dict[str, Any]) -> dict[str, Any]:
    evidence_ref = row["evidence_ref"]
    return {
        "schema_version": "s0b.text_unit.v0.1",
        "text_unit_id": "s0b:incremental:" + short_hash(evidence_ref),
        "unit_type": "sentence",
        "raw_source_id": row.get("raw_source_id"),
        "source_id": row.get("source_id"),
        "source_version": row.get("source_version"),
        "evidence_ref": evidence_ref,
        "canonical_evidence_ref": row.get("canonical_evidence_ref") or evidence_ref,
        "text": row.get("text") or "",
        "speaker": row.get("speaker"),
        "subject_id": row.get("subject_id"),
        "subject_ids": row.get("subject_ids") or [],
        "target_subject_ids": row.get("target_subject_ids") or [],
        "target_participant": row.get("target_participant"),
        "subject_role": row.get("subject_role"),
        "section_type": row.get("section_type") or "conversation",
        "retrieval_policy": row.get("retrieval_policy") or "default_retrieval",
        "s2_policy": row.get("s2_policy") or "candidate_allowed",
        "maintenance_status": row.get("maintenance_status") or "active",
    }


def synthesize_evidence_from_text_unit(row: dict[str, Any], *, modeled_subject_id: str) -> dict[str, Any]:
    evidence_ref = row.get("evidence_ref") or "evidence:incremental:" + short_hash(row.get("text_unit_id") or row.get("text") or "")
    return normalize_evidence_row(
        {
            "row_kind": "evidence",
            "object_id": evidence_ref,
            "source_id": row.get("source_id"),
            "source_version": row.get("source_version"),
            "maintenance_status": row.get("maintenance_status"),
            "raw_row": {
                **row,
                "evidence_ref": evidence_ref,
                "canonical_evidence_ref": row.get("canonical_evidence_ref") or evidence_ref,
            },
        },
        modeled_subject_id=modeled_subject_id,
    )


def collect_incremental_rows(
    batch_rows: list[dict[str, Any]],
    *,
    modeled_subject_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence_rows: list[dict[str, Any]] = []
    text_unit_rows: list[dict[str, Any]] = []
    for batch_row in batch_rows:
        if batch_row.get("maintenance_status") not in {None, "", "active"}:
            continue
        row_kind = normalize(batch_row.get("row_kind"))
        if row_kind == "evidence":
            evidence_rows.append(normalize_evidence_row(batch_row, modeled_subject_id=modeled_subject_id))
        elif row_kind == "text_unit":
            text_unit_rows.append(normalize_text_unit_row(batch_row, modeled_subject_id=modeled_subject_id))
    if not evidence_rows and text_unit_rows:
        evidence_rows = [synthesize_evidence_from_text_unit(row, modeled_subject_id=modeled_subject_id) for row in text_unit_rows]
    if not text_unit_rows and evidence_rows:
        text_unit_rows = [synthesize_text_unit_from_evidence(row) for row in evidence_rows]
    return evidence_rows, text_unit_rows


def prepare_output_dir(output_dir: Path, duplicate_policy: str) -> None:
    outputs = [
        output_dir / "s1_incremental_preprocessing_decisions.jsonl",
        output_dir / "s1_incremental_memory_candidates.jsonl",
        output_dir / "s1_incremental_memory_units.jsonl",
        output_dir / "s1_incremental_build_manifest.json",
        output_dir / REPORT_FILENAME,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and duplicate_policy == "fail":
        raise FileExistsError("Existing S1 incremental outputs found: " + ", ".join(str(path) for path in existing))
    if duplicate_policy == "overwrite_generated":
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def stage_incremental_workspace(
    *,
    output_dir: Path,
    evidence_rows: list[dict[str, Any]],
    text_unit_rows: list[dict[str, Any]],
    workspace_id: str,
    modeled_subject_id: str,
) -> Path:
    staged = output_dir / "staged_s1_incremental_workspace"
    write_jsonl(staged / "evidence" / "evidence.jsonl", evidence_rows)
    write_jsonl(staged / "raw" / "organization" / "text_units.jsonl", text_unit_rows)
    write_jsonl(staged / "raw" / "organization" / "raw_sources.jsonl", [])
    write_json(
        staged / "raw" / "organization" / "bundle.json",
        {
            "schema_version": "maintenance.s1_incremental_staged_workspace.v0.4",
            "workspace_id": workspace_id,
            "modeled_subject_id": modeled_subject_id,
            "source": "s0b_incremental_batch",
        },
    )
    return staged


def incremental_build_inputs(
    *,
    project_root: Path,
    staged_workspace: Path,
    output_dir: Path,
    workspace_id: str,
    modeled_subject_id: str,
    run_id: str,
    started_at: str,
) -> BuildInputs:
    return BuildInputs(
        project_root=project_root,
        workspace=staged_workspace,
        output_workspace=output_dir,
        workspace_id=workspace_id,
        modeled_subject_id=modeled_subject_id,
        run_scope="s1_incremental_build",
        duplicate_policy="fail",
        run_id=run_id,
        started_at=started_at,
        latest_view_mode="off",
        s0b_latest_view_path=None,
        raw_sources=[],
        bundle={"workspace_id": workspace_id, "modeled_subject_id": modeled_subject_id},
    )


def rewrite_incremental_origin(rows: list[dict[str, Any]], *, run_id: str) -> None:
    for row in rows:
        origin = row.get("candidate_origin")
        if isinstance(origin, dict):
            origin["source_tool"] = "tools.maintenance.s1_incremental_build_runner"
            origin["source_workflow"] = "v0.4-s1-incremental-build"
            origin["source_run_id"] = run_id
            origin["notes"] = "Generated from staged S0B incremental batch evidence; not from the full Step1 build runner."
        row["incremental_run_id"] = run_id
        row["canonical_write"] = False
        row["write_permission"] = False
        warnings = row.get("warnings") or row.get("processing_warnings") or []
        if isinstance(warnings, list):
            row["warnings"] = sorted(set([*warnings, "incremental_candidate_not_canonical_memory"]))


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.4 S1 Incremental Build",
        "",
        f"- status: `{manifest['status']}`",
        f"- run_id: `{manifest['run_id']}`",
        f"- input batch rows: {manifest['counts']['input_batch_rows']}",
        f"- staged evidence rows: {manifest['counts']['staged_evidence_rows']}",
        f"- staged text unit rows: {manifest['counts']['staged_text_unit_rows']}",
        f"- route decisions: {manifest['counts']['route_decisions']}",
        f"- proposal rows: {manifest['counts']['proposal_rows']}",
        f"- incremental memory units: {manifest['counts']['memory_units']}",
        f"- LLM-assisted memory units: {manifest['counts']['llm_assisted_memory_units']}",
        "- full_build_runner_called: `false`",
        "- canonical_writes_executed: `false`",
        "- write_permission: `false`",
        "",
        "## Boundary",
        "",
        "- This runner consumes S0B incremental batch rows.",
        "- It stages temporary evidence/text-unit inputs for route/proposal only.",
        "- It does not call `tools.step1.step1_build_runner.run_build`.",
        "- It does not write canonical `memory/memory_units.jsonl` or graph truth.",
        "",
        "## Outputs",
        "",
    ]
    for name, value in manifest["outputs"].items():
        lines.append(f"- {name}: `{value}`")
    return "\n".join(lines) + "\n"


def run_s1_incremental_build(
    *,
    project_root: Path,
    s0b_incremental_batch_rows: Path,
    output_dir: Path,
    workspace_id: str,
    modeled_subject_id: str,
    run_id: str | None = None,
    pre_build_route_mode: str = DEFAULT_PRE_BUILD_ROUTE_MODE,
    route_policy: str = DEFAULT_ROUTE_POLICY,
    proposal_profile: str = DEFAULT_S1_PROPOSAL_PROFILE,
    proposal_provider: str = "openai",
    api_mode: str = "responses",
    allow_live_api: bool = False,
    env_file: str | None = ".env",
    external_model_outputs: str | None = None,
    max_items: int | None = None,
    item_offset: int = 0,
    sample_stride: int = 1,
    provider_concurrency: int = 1,
    duplicate_policy: str = "fail",
) -> dict[str, Any]:
    if duplicate_policy not in {"fail", "overwrite_generated"}:
        raise ValueError(f"Unsupported duplicate_policy: {duplicate_policy}")
    started_at = now_iso()
    resolved_run_id = run_id or f"s1-incremental:{workspace_id}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()
    s0b_incremental_batch_rows = s0b_incremental_batch_rows.resolve()
    prepare_output_dir(output_dir, duplicate_policy)

    batch_rows = read_jsonl(s0b_incremental_batch_rows)
    evidence_rows, text_unit_rows = collect_incremental_rows(batch_rows, modeled_subject_id=modeled_subject_id)
    if not evidence_rows:
        raise ValueError("S1 incremental build requires at least one evidence or text_unit row from the S0B batch")

    staged_workspace = stage_incremental_workspace(
        output_dir=output_dir,
        evidence_rows=evidence_rows,
        text_unit_rows=text_unit_rows,
        workspace_id=workspace_id,
        modeled_subject_id=modeled_subject_id,
    )
    routing_options = PreBuildRoutingOptions(
        mode=pre_build_route_mode,
        step_name="s1_incremental",
        target_task="s1_memory_candidate",
        route_policy=route_policy,
        proposal_profile=proposal_profile,
        proposal_provider=proposal_provider,
        api_mode=api_mode,
        provider_profile=None,
        fallback_provider_profile=None,
        allow_live_api=allow_live_api,
        env_file=env_file,
        external_model_outputs=external_model_outputs,
        max_items=max_items,
        item_offset=max(0, item_offset),
        sample_stride=max(1, sample_stride),
        duplicate_policy=duplicate_policy,
        provider_concurrency=max(1, provider_concurrency),
    )
    prebuild_result = run_prebuild_routing(
        project_root=project_root,
        route_workspace=staged_workspace,
        proposal_workspace=staged_workspace,
        output_workspace=output_dir,
        base_run_id=resolved_run_id,
        options=routing_options,
    )

    proposal_rows = [
        row
        for row in load_s1_prebuild_proposal_rows(prebuild_result)
        if row.get("output_kind") == "memory_candidate"
        and str(row.get("processed_text") or row.get("memory_candidate_text") or "").strip()
    ]
    proposal_index = build_s1_proposal_index(proposal_rows)
    reject_index = build_s1_reject_index(
        [
            *load_s1_prebuild_proposal_rows(prebuild_result),
            *build_s1_route_skip_rows(load_s1_prebuild_route_rows(prebuild_result)),
        ]
    )
    memory_result = build_memory(
        incremental_build_inputs(
            project_root=project_root,
            staged_workspace=staged_workspace,
            output_dir=output_dir,
            workspace_id=workspace_id,
            modeled_subject_id=modeled_subject_id,
            run_id=resolved_run_id,
            started_at=started_at,
        ),
        evidence_rows,
        proposal_index,
        reject_index,
    )
    validation = validate_memory_result(memory_result, evidence_rows)
    if not validation["valid"]:
        raise ValueError(f"S1 incremental memory validation failed: {validation}")

    rewrite_incremental_origin(memory_result.memory_candidates, run_id=resolved_run_id)
    rewrite_incremental_origin(memory_result.memory_units, run_id=resolved_run_id)

    decisions_path = output_dir / "s1_incremental_preprocessing_decisions.jsonl"
    candidates_path = output_dir / "s1_incremental_memory_candidates.jsonl"
    units_path = output_dir / "s1_incremental_memory_units.jsonl"
    manifest_path = output_dir / "s1_incremental_build_manifest.json"
    report_path = output_dir / REPORT_FILENAME
    write_jsonl(decisions_path, memory_result.preprocessing_decisions)
    write_jsonl(candidates_path, memory_result.memory_candidates)
    write_jsonl(units_path, memory_result.memory_units)

    route_counts = ((prebuild_result or {}).get("route") or {}).get("counts") or {}
    proposal_counts = ((prebuild_result or {}).get("proposal") or {}).get("counts") or {}
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "run_id": resolved_run_id,
        "runner": "tools.maintenance.s1_incremental_build_runner",
        "workflow": "v0.4-s1-incremental-build",
        "status": "completed",
        "workspace_id": workspace_id,
        "modeled_subject_id": modeled_subject_id,
        "started_at": started_at,
        "completed_at": now_iso(),
        "inputs": {
            "s0b_incremental_batch_rows": str(s0b_incremental_batch_rows),
            "s0b_incremental_batch_rows_hash": file_hash(s0b_incremental_batch_rows),
            "staged_workspace": str(staged_workspace),
        },
        "prebuild_routing": prebuild_result,
        "prebuild_routing_options": asdict(routing_options),
        "outputs": {
            "staged_evidence": str(staged_workspace / "evidence" / "evidence.jsonl"),
            "staged_text_units": str(staged_workspace / "raw" / "organization" / "text_units.jsonl"),
            "s1_incremental_preprocessing_decisions": str(decisions_path),
            "s1_incremental_memory_candidates": str(candidates_path),
            "s1_incremental_memory_units": str(units_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
        "counts": {
            "input_batch_rows": len(batch_rows),
            "staged_evidence_rows": len(evidence_rows),
            "staged_text_unit_rows": len(text_unit_rows),
            "route_decisions": int(route_counts.get("route_decisions") or 0),
            "route_input_items": int(route_counts.get("input_items") or 0),
            "proposal_rows": int(proposal_counts.get("proposal_rows") or 0),
            **memory_result.memory_stats,
        },
        "validation_summary": validation,
        "boundary": {
            "full_build_runner_called": False,
            "canonical_writes_executed": False,
            "durable_writes_executed": False,
            "graph_truth_written": False,
            "write_permission": False,
            "semantic_truth_status": "candidate_not_truth",
        },
    }
    write_json(manifest_path, manifest)
    report_path.write_text(render_report(manifest), encoding="utf-8")
    return {
        "manifest": manifest,
        "memory_result": memory_result,
        "prebuild_result": prebuild_result,
        "paths": manifest["outputs"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--s0b-incremental-batch-rows", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--modeled-subject-id", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--pre-build-route-mode", default=DEFAULT_PRE_BUILD_ROUTE_MODE)
    parser.add_argument("--route-policy", default=DEFAULT_ROUTE_POLICY)
    parser.add_argument("--proposal-profile", default=DEFAULT_S1_PROPOSAL_PROFILE)
    parser.add_argument("--proposal-provider", default="openai", choices=["mock", "external_jsonl", "openai"])
    parser.add_argument("--api-mode", default="responses")
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--external-model-outputs", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--item-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--provider-concurrency", type=int, default=1)
    parser.add_argument("--duplicate-policy", default="fail", choices=["fail", "overwrite_generated"])
    args = parser.parse_args()
    bundle = run_s1_incremental_build(
        project_root=Path(args.project_root),
        s0b_incremental_batch_rows=args.s0b_incremental_batch_rows,
        output_dir=args.output_dir,
        workspace_id=args.workspace_id,
        modeled_subject_id=args.modeled_subject_id,
        run_id=args.run_id,
        pre_build_route_mode=args.pre_build_route_mode,
        route_policy=args.route_policy,
        proposal_profile=args.proposal_profile,
        proposal_provider=args.proposal_provider,
        api_mode=args.api_mode,
        allow_live_api=args.allow_live_api,
        env_file=args.env_file,
        external_model_outputs=args.external_model_outputs,
        max_items=args.max_items,
        item_offset=args.item_offset,
        sample_stride=args.sample_stride,
        provider_concurrency=args.provider_concurrency,
        duplicate_policy=args.duplicate_policy,
    )
    print(json.dumps(bundle["manifest"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
