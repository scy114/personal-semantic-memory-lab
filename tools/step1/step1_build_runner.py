"""Step 1 build runner for S0B-to-S1 builds.

This is a small integration runner, not the final Step 1 build platform.
It reads S0B raw organization manifests, runs an explicit adapter, and writes
S1 source/evidence/build artifacts. It can optionally materialize conservative
evidence-bound memory candidates/units and session summaries when requested.
It intentionally does not build indexes, Step 2 assets, or final answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.prebuild_routing import (
    DEFAULT_S1_PROPOSAL_PROFILE,
    add_prebuild_arguments,
    options_from_args,
    run_prebuild_routing,
)


SUPPORTED_RUN_SCOPES = {"evidence_only", "evidence_plus_memory", "full_s1_build"}
SUPPORTED_DUPLICATE_POLICIES = {"fail", "overwrite_generated"}
BACKGROUND_ONLY = {"background_only"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def short_hash(value: str, length: int = 16) -> str:
    return sha256_text(value)[:length]


def safe_ref_part(value: Any) -> str:
    compact = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value or "unknown").strip()).strip("-")
    return compact or "unknown"


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def jsonl_hash(rows: list[dict[str, Any]]) -> str:
    return sha256_text(jsonl_text(rows))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonl_text(rows), encoding="utf-8")


def stable_run_id(workspace_id: str, run_scope: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"s1-build:{workspace_id}:{run_scope}:{stamp}"


def looks_like_external_uri(value: str) -> bool:
    lowered = value.lower()
    return "://" in lowered and not lowered.startswith("file://")


def resolve_path(project_root: Path, workspace: Path, source: dict[str, Any]) -> Path:
    raw = source.get("local_path") or source.get("original_uri_or_path")
    if not raw:
        raise ValueError(f"raw_source {source.get('raw_source_id')} has no local_path")
    if looks_like_external_uri(str(raw)):
        raise ValueError(
            "Step 1 Build Runner v0.1 supports local paths only; external URI must be "
            f"fetched and organized by S0-A before S1 Build: {raw}"
        )
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [project_root / path, workspace / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def parse_locomo_dia_id(dia_id: str) -> tuple[str, int, int]:
    day, turn = dia_id.split(":", 1)
    if not day.startswith("D"):
        raise ValueError(f"Unexpected LoCoMo dia_id: {dia_id}")
    return day, int(day[1:]), int(turn)


def validate_raw_sources(raw_sources: list[dict[str, Any]]) -> None:
    if not raw_sources:
        raise ValueError("S0B raw_sources.jsonl is empty; Step 1 build requires at least one raw source.")

    required_fields = [
        "raw_source_id",
        "workspace_id",
        "bundle_id",
        "source_type",
        "modality",
        "organization_degree",
        "processing_status",
        "inclusion_decision",
        "adapter_recommendation",
        "quality_notes",
    ]
    errors: list[str] = []
    for index, source in enumerate(raw_sources, 1):
        raw_source_id = source.get("raw_source_id") or f"row_{index}"
        missing = [field for field in required_fields if field not in source or source.get(field) in ("", None)]
        if not (source.get("local_path") or source.get("original_uri_or_path")):
            missing.append("local_path_or_original_uri_or_path")
        raw_path = source.get("local_path") or source.get("original_uri_or_path")
        if raw_path and looks_like_external_uri(str(raw_path)):
            errors.append(
                f"{raw_source_id}: external URI is not supported by S1 build runner v0.1; "
                "fetch/organize it in S0-A first"
            )
        if missing:
            errors.append(f"{raw_source_id}: missing required S0B fields: {', '.join(sorted(set(missing)))}")
    if errors:
        raise ValueError("S0B raw source intake validation failed: " + " | ".join(errors))


@dataclass
class BuildInputs:
    project_root: Path
    workspace: Path
    output_workspace: Path
    workspace_id: str
    modeled_subject_id: str
    run_scope: str
    duplicate_policy: str
    run_id: str
    started_at: str
    raw_sources: list[dict[str, Any]]
    bundle: dict[str, Any]


@dataclass
class OutputPreparation:
    checkpoint_paths: list[str]
    overwritten_paths: list[str]
    stale_higher_scope_assets: list[str]
    stale_asset_policy: str


@dataclass
class AdapterResult:
    source_manifest: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    adapter_routes: list[dict[str, Any]]
    source_stats: dict[str, Any]


@dataclass
class MemoryResult:
    preprocessing_decisions: list[dict[str, Any]]
    memory_candidates: list[dict[str, Any]]
    memory_units: list[dict[str, Any]]
    memory_stats: dict[str, Any]


@dataclass
class SummaryResult:
    summaries: list[dict[str, Any]]
    summary_stats: dict[str, Any]


def load_inputs(
    project_root: Path,
    workspace: Path,
    output_workspace: Path,
    run_scope: str,
    duplicate_policy: str,
    workspace_id: str | None,
    modeled_subject_id: str | None,
    run_id: str | None,
    started_at: str,
) -> BuildInputs:
    raw_sources_path = workspace / "raw" / "organization" / "raw_sources.jsonl"
    bundle_path = workspace / "raw" / "organization" / "bundle.json"
    if not raw_sources_path.exists():
        raise FileNotFoundError(f"Required S0B raw source manifest missing: {raw_sources_path}")

    raw_sources = read_jsonl(raw_sources_path)
    validate_raw_sources(raw_sources)
    bundle = read_json(bundle_path) if bundle_path.exists() else {}
    resolved_workspace_id = workspace_id or bundle.get("workspace_id") or workspace.name
    resolved_modeled_subject = modeled_subject_id or bundle.get("modeled_subject_id") or "unknown"
    resolved_run_id = run_id or stable_run_id(resolved_workspace_id, run_scope)

    return BuildInputs(
        project_root=project_root,
        workspace=workspace,
        output_workspace=output_workspace,
        workspace_id=str(resolved_workspace_id),
        modeled_subject_id=str(resolved_modeled_subject),
        run_scope=run_scope,
        duplicate_policy=duplicate_policy,
        run_id=resolved_run_id,
        started_at=started_at,
        raw_sources=raw_sources,
        bundle=bundle,
    )


def scope_assets(inputs: BuildInputs) -> list[Path]:
    assets = [
        inputs.output_workspace / "evidence" / "source_manifest.jsonl",
        inputs.output_workspace / "evidence" / "evidence.jsonl",
        inputs.output_workspace / "evidence" / "build_manifest.json",
    ]
    if inputs.run_scope in {"evidence_plus_memory", "full_s1_build"}:
        assets.extend(
            [
                inputs.output_workspace / "memory" / "preprocessing_decisions.jsonl",
                inputs.output_workspace / "memory" / "memory_candidates.jsonl",
                inputs.output_workspace / "memory" / "memory_units.jsonl",
                inputs.output_workspace / "memory" / "memory_build_manifest.json",
            ]
        )
    if inputs.run_scope == "full_s1_build":
        assets.append(inputs.output_workspace / "memory" / "summaries.jsonl")
    return assets


def stale_higher_scope_assets(inputs: BuildInputs) -> list[Path]:
    memory_assets = [
        inputs.output_workspace / "memory" / "preprocessing_decisions.jsonl",
        inputs.output_workspace / "memory" / "memory_candidates.jsonl",
        inputs.output_workspace / "memory" / "memory_units.jsonl",
        inputs.output_workspace / "memory" / "memory_build_manifest.json",
    ]
    summary_assets = [inputs.output_workspace / "memory" / "summaries.jsonl"]
    if inputs.run_scope == "evidence_only":
        return [path for path in [*memory_assets, *summary_assets] if path.exists()]
    if inputs.run_scope == "evidence_plus_memory":
        return [path for path in summary_assets if path.exists()]
    return []


def prepare_output_workspace(inputs: BuildInputs) -> OutputPreparation:
    existing = [path for path in scope_assets(inputs) if path.exists()]
    stale = stale_higher_scope_assets(inputs)
    checkpoint_paths: list[str] = []
    overwritten_paths: list[str] = []
    if (existing or stale) and inputs.duplicate_policy == "fail":
        raise FileExistsError(
            "Existing S1 build assets found and duplicate_policy=fail: "
            + ", ".join(str(path) for path in [*existing, *stale])
        )
    if (existing or stale) and inputs.duplicate_policy == "overwrite_generated":
        checkpoint_root = inputs.output_workspace / "checkpoints" / (
            "before-overwrite-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        for path in [*existing, *stale]:
            rel = path.relative_to(inputs.output_workspace)
            target = checkpoint_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            checkpoint_paths.append(str(target))
            overwritten_paths.append(str(path))
        for path in [*existing, *stale]:
            path.unlink()

    for rel in ("evidence", "reports", "checkpoints"):
        (inputs.output_workspace / rel).mkdir(parents=True, exist_ok=True)
    if inputs.run_scope in {"evidence_plus_memory", "full_s1_build"}:
        (inputs.output_workspace / "memory").mkdir(parents=True, exist_ok=True)
    return OutputPreparation(
        checkpoint_paths=checkpoint_paths,
        overwritten_paths=overwritten_paths,
        stale_higher_scope_assets=[str(path) for path in stale],
        stale_asset_policy=(
            "checkpointed_and_removed" if stale and inputs.duplicate_policy == "overwrite_generated"
            else "blocked_by_duplicate_policy_fail" if stale
            else "not_applicable"
        ),
    )


def copy_s0b_assets(inputs: BuildInputs) -> None:
    """Preserve the exact S0B input view under the output workspace."""

    org_dir = inputs.output_workspace / "raw" / "organization"
    org_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(org_dir / "raw_sources.jsonl", inputs.raw_sources)
    if inputs.bundle:
        write_json(org_dir / "bundle.json", inputs.bundle)


def build_evidence(inputs: BuildInputs) -> AdapterResult:
    source_manifest: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    adapter_routes: list[dict[str, Any]] = []
    source_stats: dict[str, Any] = {
        "included_raw_sources": 0,
        "background_only_raw_sources": 0,
        "unsupported_raw_sources": [],
    }

    for source in inputs.raw_sources:
        inclusion = source.get("inclusion_decision", "include")
        adapter_name = source.get("adapter_recommendation") or inputs.bundle.get("default_adapter_recommendation")
        if inclusion in BACKGROUND_ONLY:
            source_manifest.append(background_source_manifest(inputs, source, adapter_name))
            source_stats["background_only_raw_sources"] += 1
            continue
        if inclusion != "include":
            continue
        if source.get("processing_status") != "ready_for_s1_intake":
            raise ValueError(
                f"raw_source {source.get('raw_source_id')} is not ready_for_s1_intake: "
                f"{source.get('processing_status')}"
            )
        if adapter_name == "locomo_conversation_adapter":
            result = build_locomo_conversation(inputs, source)
        elif adapter_name == "fixture_conversation_adapter":
            result = build_fixture_conversation(inputs, source)
        elif adapter_name == "generic_text_adapter":
            result = build_generic_text(inputs, source)
        else:
            source_stats["unsupported_raw_sources"].append(
                {"raw_source_id": source.get("raw_source_id"), "adapter_recommendation": adapter_name}
            )
            raise ValueError(f"Unsupported adapter_recommendation: {adapter_name}")
        source_manifest.extend(result.source_manifest)
        evidence.extend(result.evidence)
        adapter_routes.extend(result.adapter_routes)
        source_stats["included_raw_sources"] += 1
        source_stats.update(result.source_stats)

    return AdapterResult(
        source_manifest=source_manifest,
        evidence=evidence,
        adapter_routes=adapter_routes,
        source_stats=source_stats,
    )


def background_source_manifest(inputs: BuildInputs, source: dict[str, Any], adapter_name: str | None) -> dict[str, Any]:
    raw_path = resolve_path(inputs.project_root, inputs.workspace, source)
    return {
        "schema_version": "step1.source_manifest_item.v1",
        "source_id": f"{source.get('raw_source_id')}:background",
        "raw_source_id": source.get("raw_source_id"),
        "source_type": source.get("source_type", "unknown"),
        "modality": source.get("modality", "unknown"),
        "dataset": source.get("source_specific_metadata", {}).get("dataset", ""),
        "dataset_or_project": source.get("source_specific_metadata", {}).get("dataset", ""),
        "record_id": source.get("source_specific_metadata", {}).get("record_id", ""),
        "path_or_uri": str(raw_path.resolve()),
        "source_path": str(raw_path.resolve()),
        "checksum_or_version": source.get("content_hash") or (file_hash(raw_path) if raw_path.exists() else ""),
        "privacy_class": source.get("privacy_class", "unknown"),
        "status": "background_only",
        "adapter_run_id": f"adapter_run:{inputs.workspace_id}:{source.get('raw_source_id')}:background",
        "source_specific_ref_policy": "background_only source; not converted into ordinary raw evidence.",
        "notes": "Recorded for provenance only. S1 evidence build does not treat this as ordinary raw evidence.",
        "adapter_recommendation": adapter_name or "",
    }


def build_locomo_conversation(inputs: BuildInputs, source: dict[str, Any]) -> AdapterResult:
    raw_path = resolve_path(inputs.project_root, inputs.workspace, source)
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    metadata = source.get("source_specific_metadata", {})
    record_id = metadata.get("record_id") or source.get("record_id")
    if not record_id:
        raise ValueError(f"LoCoMo source {source.get('raw_source_id')} has no record_id")
    record = next((row for row in data if row.get("sample_id") == record_id), None)
    if record is None:
        raise ValueError(f"LoCoMo record not found: {record_id}")

    conv = record["conversation"]
    participants = [conv.get("speaker_a", ""), conv.get("speaker_b", "")]
    adapter_run_id = f"adapter_run:{inputs.workspace_id}:{record_id}:dialogue:v0.1"
    source_id = f"locomo-{record_id}-raw"
    target = inputs.modeled_subject_id
    rows: list[dict[str, Any]] = []
    sessions: list[str] = []
    target_count = 0
    other_count = 0

    for session_index in range(1, 1000):
        session_key = f"session_{session_index}"
        turns = conv.get(session_key)
        if not turns:
            continue
        sessions.append(session_key)
        timestamp = conv.get(f"{session_key}_date_time")
        for turn in turns:
            dia_id = turn.get("dia_id")
            if not dia_id:
                raise ValueError(f"Missing LoCoMo dia_id in {record_id} {session_key}")
            day, day_index, turn_index = parse_locomo_dia_id(dia_id)
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            evidence_ref = f"evidence:locomo:{record_id}:{dia_id}"
            is_target = speaker == target
            target_count += 1 if is_target else 0
            other_count += 0 if is_target else 1
            item = {
                "schema_version": "step1.evidence_item.v1",
                "evidence_ref": evidence_ref,
                "canonical_evidence_ref": evidence_ref,
                "source_specific_ref": dia_id,
                "display_ref": f"LoCoMo {record_id} {dia_id}",
                "ref_aliases": [dia_id, evidence_ref],
                "source_id": source_id,
                "raw_source_id": source.get("raw_source_id"),
                "record_id": record_id,
                "dataset": "locomo",
                "source_type": "conversation",
                "modality": "text",
                "item_layer": "raw_evidence",
                "text": text,
                "content_hash": sha256_text(text),
                "speaker": speaker,
                "participant": speaker,
                "participant_ids": [p for p in participants if p],
                "subject_ids": [speaker] if speaker != "unknown" else [],
                "target_subject_ids": [target],
                "target_participant": target,
                "subject_role": "target" if is_target else "other_participant",
                "subject_contamination_risk": "none" if is_target else "medium",
                "timestamp": timestamp,
                "locator": {
                    "kind": "conversation_turn",
                    "dataset": "locomo",
                    "record_id": record_id,
                    "session": session_key,
                    "session_index": session_index,
                    "day": day,
                    "day_index": day_index,
                    "turn_index": turn_index,
                    "speaker": speaker,
                    "timestamp": timestamp,
                },
                "privacy_class": source.get("privacy_class", "unknown"),
                "extraction_method": "locomo_conversation_adapter",
                "extraction_confidence": 1.0,
                "adapter_run_id": adapter_run_id,
                "metadata": {
                    "legacy_locomo_ref": dia_id,
                    "modeled_subject_id": target,
                    "source_specific_ref_semantics": (
                        "LoCoMo source-local day/turn ref; preserved for compatibility, not core evidence model."
                    ),
                },
            }
            extra = {key: value for key, value in turn.items() if key not in {"speaker", "dia_id", "text"}}
            if extra:
                item["metadata"]["source_specific_turn_metadata"] = extra
            rows.append(item)

    source_manifest = [
        {
            "schema_version": "step1.source_manifest_item.v1",
            "source_id": source_id,
            "raw_source_id": source.get("raw_source_id"),
            "source_type": "conversation",
            "modality": "text",
            "dataset": "locomo",
            "dataset_or_project": "locomo",
            "record_id": record_id,
            "path_or_uri": str(raw_path.resolve()),
            "source_path": str(raw_path.resolve()),
            "checksum_or_version": file_hash(raw_path),
            "privacy_class": source.get("privacy_class", "unknown"),
            "status": "active",
            "adapter_run_id": adapter_run_id,
            "source_specific_ref_policy": (
                "LoCoMo D<day>:<turn> is source_specific_ref/display_ref; canonical evidence_ref is generated by S0-to-S1 adapter."
            ),
            "notes": f"LoCoMo raw conversation record {record_id}; generated by Step 1 build runner.",
        }
    ]
    route = {
        "raw_source_id": source.get("raw_source_id"),
        "adapter_name": "locomo_conversation_adapter",
        "adapter_version": "v0.1",
        "adapter_family": "conversation_dataset_adapter",
        "reason": "S0B marks this source as high-organization LoCoMo text with session/dia_id/speaker structure.",
        "warnings": ["LoCoMo day/turn refs are source_specific_ref/display_ref, not universal S1 evidence semantics."],
    }
    return AdapterResult(
        source_manifest=source_manifest,
        evidence=rows,
        adapter_routes=[route],
        source_stats={
            "session_count": len(sessions),
            "included_sessions": sessions,
            "target_utterance_count": target_count,
            "other_utterance_count": other_count,
        },
    )


def build_fixture_conversation(inputs: BuildInputs, source: dict[str, Any]) -> AdapterResult:
    """Normalize the tiny test fixture without treating it as a general schema."""

    raw_path = resolve_path(inputs.project_root, inputs.workspace, source)
    source_rows = read_jsonl(raw_path)
    adapter_run_id = f"adapter_run:{inputs.workspace_id}:{source.get('raw_source_id')}:fixture:v0.1"
    source_id = f"{source.get('raw_source_id')}:source"
    evidence: list[dict[str, Any]] = []
    target = inputs.modeled_subject_id
    for index, row in enumerate(source_rows, 1):
        text = row.get("text", "")
        source_specific_ref = row.get("source_specific_ref") or row.get("evidence_ref") or f"turn_{index:03d}"
        evidence_ref = f"evidence:{inputs.workspace_id}:{source_specific_ref}"
        speaker = row.get("speaker") or row.get("participant") or "unknown"
        locator = row.get("locator") or {
            "kind": "conversation_turn",
            "record_id": source.get("source_specific_metadata", {}).get("record_id", "fixture"),
            "turn_index": index,
            "speaker": speaker,
        }
        locator = dict(locator)
        if locator.get("kind") == "conversation_turn":
            locator.setdefault("session", "session_1")
            locator.setdefault("session_index", 1)
            locator.setdefault("record_id", source.get("source_specific_metadata", {}).get("record_id", "fixture"))
            locator.setdefault("speaker", speaker)
        evidence.append(
            {
                "schema_version": "step1.evidence_item.v1",
                "evidence_ref": evidence_ref,
                "canonical_evidence_ref": evidence_ref,
                "source_specific_ref": source_specific_ref,
                "display_ref": row.get("display_ref") or f"Fixture turn {index}",
                "ref_aliases": [source_specific_ref, evidence_ref],
                "source_id": source_id,
                "raw_source_id": source.get("raw_source_id"),
                "record_id": row.get("record_id") or source.get("source_specific_metadata", {}).get("record_id", "fixture"),
                "source_type": source.get("source_type", "conversation"),
                "modality": source.get("modality", "text"),
                "item_layer": "raw_evidence",
                "text": text,
                "content_hash": sha256_text(text),
                "speaker": speaker,
                "participant": speaker,
                "participant_ids": row.get("participant_ids", []),
                "subject_ids": row.get("subject_ids") or ([speaker] if speaker != "unknown" else []),
                "target_subject_ids": [target],
                "target_participant": target,
                "subject_role": "target" if speaker == target else "other_participant",
                "subject_contamination_risk": "none" if speaker == target else "medium",
                "timestamp": row.get("timestamp"),
                "locator": locator,
                "privacy_class": source.get("privacy_class", "unknown"),
                "extraction_method": "fixture_conversation_adapter",
                "extraction_confidence": 1.0,
                "adapter_run_id": adapter_run_id,
                "metadata": {"source_fixture_note": "fixture adapter output; not a universal evidence model"},
            }
        )
    source_manifest = [
        {
            "schema_version": "step1.source_manifest_item.v1",
            "source_id": source_id,
            "raw_source_id": source.get("raw_source_id"),
            "source_type": source.get("source_type", "conversation"),
            "modality": source.get("modality", "text"),
            "record_id": source.get("source_specific_metadata", {}).get("record_id", "fixture"),
            "path_or_uri": str(raw_path.resolve()),
            "source_path": str(raw_path.resolve()),
            "checksum_or_version": file_hash(raw_path),
            "privacy_class": source.get("privacy_class", "unknown"),
            "status": "active",
            "adapter_run_id": adapter_run_id,
            "source_specific_ref_policy": "Fixture refs are source_specific_ref/display_ref, not core evidence semantics.",
            "notes": "Synthetic fixture source for Step 1 build runner tests.",
        }
    ]
    return AdapterResult(
        source_manifest=source_manifest,
        evidence=evidence,
        adapter_routes=[{"raw_source_id": source.get("raw_source_id"), "adapter_name": "fixture_conversation_adapter"}],
        source_stats={"fixture_evidence_count": len(evidence)},
    )


def split_text_segments(text: str, max_chars: int = 1200) -> list[tuple[int, int, str]]:
    """Split plain text into paragraph-like spans without assuming dialogue turns."""

    segments: list[tuple[int, int, str]] = []
    current_parts: list[str] = []
    current_start: int | None = None
    current_end = 0

    for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, flags=re.S):
        raw = match.group(0).strip()
        if not raw:
            continue
        compact = " ".join(raw.split())
        if not compact:
            continue
        if current_start is None:
            current_start = match.start()
        projected = ("\n\n".join([*current_parts, compact])).strip()
        if current_parts and len(projected) > max_chars:
            segments.append((current_start, current_end, "\n\n".join(current_parts).strip()))
            current_parts = [compact]
            current_start = match.start()
        else:
            current_parts.append(compact)
        current_end = match.end()

    if current_parts and current_start is not None:
        segments.append((current_start, current_end, "\n\n".join(current_parts).strip()))
    return segments


def load_section_policy(inputs: BuildInputs, source: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load optional S0B section policy rows for a raw source."""

    section_map_path = inputs.workspace / "raw" / "organization" / "section_map.jsonl"
    if not section_map_path.exists():
        return {}
    raw_source_id = source.get("raw_source_id")
    rows = read_jsonl(section_map_path)
    policy: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("raw_source_id") != raw_source_id:
            continue
        if row.get("source_specific_ref"):
            policy[str(row["source_specific_ref"])] = row
        if row.get("raw_span_id"):
            policy[str(row["raw_span_id"])] = row
    return policy


def default_generic_text_policy(raw_span_id: str, source_specific_ref: str, start: int, end: int, segment_text: str) -> dict[str, Any]:
    return {
        "raw_span_id": raw_span_id,
        "source_specific_ref": source_specific_ref,
        "char_start": start,
        "char_end": end,
        "raw_text_hash": sha256_text(segment_text),
        "normalized_text_hash": sha256_text(" ".join(segment_text.split()).lower()),
        "final_section_type": "body",
        "perspective": "author",
        "s1_storage_policy": "ordinary_evidence",
        "retrieval_policy": "default_retrieval",
        "s2_policy": "candidate_allowed",
        "classification_method": "default_no_section_map",
        "confidence": "medium",
        "review_status": "auto",
        "llm_assist_status": "not_needed",
        "llm_assist_triggers": [],
    }


def load_generic_text_units(inputs: BuildInputs, source: dict[str, Any]) -> list[dict[str, Any]]:
    """Load S0B sentence text units for a generic text raw source when available."""

    text_units_path = inputs.workspace / "raw" / "organization" / "text_units.jsonl"
    if not text_units_path.exists():
        return []
    raw_source_id = source.get("raw_source_id")
    rows = read_jsonl(text_units_path)
    sentence_units = [
        row
        for row in rows
        if row.get("raw_source_id") == raw_source_id and row.get("unit_type") == "sentence"
    ]
    return sentence_units


def build_generic_text(inputs: BuildInputs, source: dict[str, Any]) -> AdapterResult:
    """Normalize a plain text source into S1 evidence items."""

    raw_path = resolve_path(inputs.project_root, inputs.workspace, source)
    text = raw_path.read_text(encoding="utf-8-sig")
    metadata = source.get("source_specific_metadata", {})
    record_id = metadata.get("record_id") or source.get("record_id") or raw_path.stem
    adapter_run_id = f"adapter_run:{inputs.workspace_id}:{source.get('raw_source_id')}:generic_text:v0.1"
    source_id = f"{source.get('raw_source_id')}:source"
    target = inputs.modeled_subject_id
    author_subject_id = metadata.get("author_subject_id") or metadata.get("author")
    modeled_subject_is_author = bool(metadata.get("modeled_subject_is_author"))
    subject_id = str(author_subject_id or target if modeled_subject_is_author else author_subject_id or "unknown")
    subject_role = "target" if subject_id == target or modeled_subject_is_author else "unknown"
    contamination_risk = "none" if subject_role == "target" else "low"
    text_units = load_generic_text_units(inputs, source)
    segments = split_text_segments(text, max_chars=int(metadata.get("max_segment_chars") or 1200))
    if not text_units and not segments:
        raise ValueError(f"generic_text_adapter found no text segments in {raw_path}")

    raw_part = safe_ref_part(source.get("raw_source_id"))
    section_policy = load_section_policy(inputs, source)
    skipped_by_policy = 0
    evidence: list[dict[str, Any]] = []
    if text_units:
        evidence_inputs = []
        for index, unit in enumerate(text_units, 1):
            source_specific_ref = str(unit.get("text_unit_id") or f"text_unit:{index:04d}")
            raw_span_id = str(unit.get("raw_span_id") or "")
            start = int(unit.get("char_start") or 0)
            end = int(unit.get("char_end") or start)
            segment_text = str(unit.get("text") or "")
            policy = section_policy.get(raw_span_id) or default_generic_text_policy(
                raw_span_id,
                source_specific_ref,
                start,
                end,
                segment_text,
            )
            for key in (
                "section_type",
                "s1_storage_policy",
                "retrieval_policy",
                "s2_policy",
                "raw_text_hash",
                "normalized_text_hash",
            ):
                if unit.get(key) is not None:
                    policy[key if key != "section_type" else "final_section_type"] = unit.get(key)
            evidence_inputs.append((index, source_specific_ref, raw_span_id, start, end, segment_text, policy, unit))
        evidence_granularity = "s0b_text_unit_sentence"
    else:
        evidence_inputs = []
        for index, (start, end, segment_text) in enumerate(segments, 1):
            source_specific_ref = f"text_span:{index:04d}"
            raw_span_id = f"{source.get('raw_source_id')}:{source_specific_ref}"
            policy = (
                section_policy.get(source_specific_ref)
                or section_policy.get(raw_span_id)
                or default_generic_text_policy(raw_span_id, source_specific_ref, start, end, segment_text)
            )
            evidence_inputs.append((index, source_specific_ref, raw_span_id, start, end, segment_text, policy, None))
        evidence_granularity = "text_span_fallback"

    for index, source_specific_ref, raw_span_id, start, end, segment_text, policy, text_unit in evidence_inputs:
        s1_storage_policy = policy.get("s1_storage_policy", "ordinary_evidence")
        if s1_storage_policy != "ordinary_evidence":
            skipped_by_policy += 1
            continue
        unit_suffix = f"unit-{index:04d}" if text_unit else f"span-{index:04d}"
        evidence_ref = f"evidence:{safe_ref_part(inputs.workspace_id)}:{raw_part}:{unit_suffix}"
        display_ref = (
            f"{metadata.get('title') or record_id} {source_specific_ref}"
            if text_unit
            else f"{metadata.get('title') or record_id} span {index}"
        )
        locator_kind = "text_unit_sentence" if text_unit else "text_span"
        ref_aliases = [source_specific_ref, evidence_ref, display_ref]
        if raw_span_id:
            ref_aliases.append(raw_span_id)
        if text_unit and text_unit.get("parent_text_unit_id"):
            ref_aliases.append(str(text_unit.get("parent_text_unit_id")))
        evidence.append(
            {
                "schema_version": "step1.evidence_item.v1",
                "evidence_ref": evidence_ref,
                "canonical_evidence_ref": evidence_ref,
                "source_specific_ref": source_specific_ref,
                "display_ref": display_ref,
                "ref_aliases": ref_aliases,
                "raw_span_id": policy.get("raw_span_id", raw_span_id),
                "text_unit_id": text_unit.get("text_unit_id") if text_unit else None,
                "parent_text_unit_id": text_unit.get("parent_text_unit_id") if text_unit else None,
                "paragraph_id": text_unit.get("paragraph_id") if text_unit else None,
                "sentence_id": text_unit.get("sentence_id") if text_unit else None,
                "source_id": source_id,
                "raw_source_id": source.get("raw_source_id"),
                "record_id": record_id,
                "source_type": source.get("source_type", "text"),
                "modality": source.get("modality", "text"),
                "item_layer": "raw_evidence",
                "text": segment_text,
                "content_hash": sha256_text(segment_text),
                "speaker": subject_id,
                "participant": subject_id,
                "participant_ids": [subject_id] if subject_id != "unknown" else [],
                "subject_ids": [subject_id] if subject_id != "unknown" else [],
                "target_subject_ids": [target],
                "target_participant": target,
                "subject_role": subject_role,
                "subject_scope": "target_only" if subject_role == "target" else "unknown_subject_text",
                "subject_contamination_risk": contamination_risk,
                "timestamp": metadata.get("date") or metadata.get("created_at"),
                "section_type": policy.get("final_section_type", "body"),
                "s1_storage_policy": s1_storage_policy,
                "retrieval_policy": policy.get("retrieval_policy", "default_retrieval"),
                "s2_policy": policy.get("s2_policy", "candidate_allowed"),
                "section_classification": {
                    "classification_method": policy.get("classification_method", "unknown"),
                    "confidence": policy.get("confidence", "unknown"),
                    "review_status": policy.get("review_status", "unknown"),
                    "llm_assist_status": policy.get("llm_assist_status", "unknown"),
                    "llm_assist_triggers": policy.get("llm_assist_triggers", []),
                    "raw_text_hash": policy.get("raw_text_hash"),
                    "normalized_text_hash": policy.get("normalized_text_hash"),
                },
                "locator": {
                    "kind": locator_kind,
                    "record_id": record_id,
                    "segment_index": index,
                    "char_start": start,
                    "char_end": end,
                    "source_file": raw_path.name,
                    "raw_span_id": raw_span_id,
                    "text_unit_id": text_unit.get("text_unit_id") if text_unit else None,
                    "parent_text_unit_id": text_unit.get("parent_text_unit_id") if text_unit else None,
                },
                "privacy_class": source.get("privacy_class", "unknown"),
                "extraction_method": "generic_text_adapter",
                "extraction_confidence": 0.95,
                "adapter_run_id": adapter_run_id,
                "metadata": {
                    "title": metadata.get("title"),
                    "author_subject_id": author_subject_id,
                    "modeled_subject_is_author": modeled_subject_is_author,
                    "source_specific_ref_semantics": (
                        "Plain-text segment ref; source-specific/display only, not universal evidence model."
                    ),
                    "section_policy_source": "section_map.jsonl" if section_policy else "default_no_section_map",
                    "evidence_granularity": evidence_granularity,
                    "text_structure_profile": text_unit.get("text_structure_profile") if text_unit else None,
                    "section_policy_profile": text_unit.get("section_policy_profile") if text_unit else None,
                },
            }
        )

    source_manifest = [
        {
            "schema_version": "step1.source_manifest_item.v1",
            "source_id": source_id,
            "raw_source_id": source.get("raw_source_id"),
            "source_type": source.get("source_type", "text"),
            "modality": source.get("modality", "text"),
            "record_id": record_id,
            "path_or_uri": str(raw_path.resolve()),
            "source_path": str(raw_path.resolve()),
            "checksum_or_version": source.get("content_hash") or file_hash(raw_path),
            "privacy_class": source.get("privacy_class", "unknown"),
            "status": "active",
            "adapter_run_id": adapter_run_id,
            "source_specific_ref_policy": (
                "S0B text_unit refs are source_specific_ref/display_ref when available; text_span refs are fallback only."
            ),
            "notes": (
                "Generic plain-text source normalized into S0B sentence text_unit evidence when available. "
                "No conversation day/turn assumptions."
            ),
        }
    ]
    return AdapterResult(
        source_manifest=source_manifest,
        evidence=evidence,
        adapter_routes=[{"raw_source_id": source.get("raw_source_id"), "adapter_name": "generic_text_adapter"}],
        source_stats={
            "generic_text_evidence_count": len(evidence),
            "generic_text_spans_skipped_by_section_policy": skipped_by_policy,
            "generic_text_units_skipped_by_section_policy": skipped_by_policy,
            "generic_text_evidence_granularity": evidence_granularity,
            "generic_text_text_unit_rows": len(text_units),
            "generic_text_section_policy_rows": len(section_policy),
        },
    )


def validate_result(result: AdapterResult) -> dict[str, Any]:
    refs = [item.get("evidence_ref") for item in result.evidence]
    duplicate_refs = sorted({ref for ref in refs if refs.count(ref) > 1})
    missing_locator = [
        item.get("evidence_ref")
        for item in result.evidence
        if "locator" not in item and "locator_unavailable_reason" not in item
    ]
    missing_raw_source = [item.get("evidence_ref") for item in result.evidence if not item.get("raw_source_id")]
    missing_adapter_run = [item.get("evidence_ref") for item in result.evidence if not item.get("adapter_run_id")]
    return {
        "source_manifest_items": len(result.source_manifest),
        "evidence_items": len(result.evidence),
        "unique_evidence_refs": len(set(refs)),
        "duplicate_evidence_refs": duplicate_refs,
        "missing_locator_or_reason": missing_locator,
        "missing_raw_source_id": missing_raw_source,
        "missing_adapter_run_id": missing_adapter_run,
        "all_evidence_has_locator_or_reason": not missing_locator,
        "valid": not duplicate_refs and not missing_locator and not missing_raw_source and not missing_adapter_run,
    }


def text_excerpt(value: str, max_chars: int = 240) -> str:
    compact = " ".join(str(value or "").split())
    return compact[:max_chars]


def normalized_text_key(value: str) -> str:
    return " ".join(str(value or "").split()).casefold()


def source_span_aliases(value: Any) -> set[str]:
    text = str(value or "")
    if not text:
        return set()
    aliases = {text}
    for match in re.findall(r"text_span:\d+", text):
        aliases.add(match)
    for match in re.findall(r"span-(\d+)", text):
        aliases.add(f"text_span:{match}")
    return aliases


def evidence_reference_values(item: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("evidence_ref", "canonical_evidence_ref", "raw_span_id", "source_specific_ref", "display_ref"):
        value = item.get(key)
        if value:
            refs.update(source_span_aliases(value))
    for value in item.get("ref_aliases") or []:
        refs.update(source_span_aliases(value))
    return refs


def exact_evidence_reference_values(item: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("evidence_ref", "canonical_evidence_ref", "source_specific_ref", "text_unit_id", "display_ref"):
        value = str(item.get(key) or "").strip()
        if value:
            refs.add(value)
    return refs


def proposal_reference_values(row: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("text_unit_id", "raw_span_id", "parent_text_unit_id", "source_text_quote"):
        value = row.get(key)
        if value:
            refs.update(source_span_aliases(value))
    for key in ("evidence_refs", "context_refs"):
        for value in row.get(key) or []:
            refs.update(source_span_aliases(value))
    for raw_backpointer in row.get("raw_backpointer_refs") or []:
        if not isinstance(raw_backpointer, dict):
            continue
        locator = raw_backpointer.get("locator") or {}
        for key in ("source_specific_ref", "raw_span_id"):
            if locator.get(key):
                refs.update(source_span_aliases(locator[key]))
    return refs


def proposal_primary_reference_values(row: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("evidence_ref", "canonical_evidence_ref", "source_specific_ref", "text_unit_id"):
        value = str(row.get(key) or "").strip()
        if value:
            refs.add(value)
    for value in row.get("evidence_refs") or []:
        value_text = str(value or "").strip()
        if value_text:
            refs.add(value_text)
    if not refs:
        for key in ("raw_span_id",):
            value = row.get(key)
            if value:
                refs.update(source_span_aliases(value))
    return refs


def raw_backpointer_refs_for_evidence(item: dict[str, Any]) -> list[dict[str, Any]]:
    locator = item.get("locator")
    if not isinstance(locator, dict):
        return []
    return [
        {
            "source_file": locator.get("source_file"),
            "locator": locator,
        }
    ]


def load_s1_prebuild_proposal_rows(prebuild_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not prebuild_result or prebuild_result.get("target_task") != "s1_memory_candidate":
        return []
    proposal = prebuild_result.get("proposal") or {}
    outputs = proposal.get("outputs") or {}
    path_value = outputs.get("proposals")
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    return read_jsonl(path)


def load_s1_prebuild_route_rows(prebuild_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not prebuild_result or prebuild_result.get("target_task") != "s1_memory_candidate":
        return []
    route = prebuild_result.get("route") or {}
    path_value = route.get("route_decisions")
    if not path_value:
        return []
    path = Path(path_value)
    if not path.exists():
        return []
    return read_jsonl(path)


def load_s1_prebuild_memory_candidate_rows(prebuild_result: dict[str, Any] | None) -> list[dict[str, Any]]:
    rows = load_s1_prebuild_proposal_rows(prebuild_result)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if row.get("output_kind") != "memory_candidate":
            continue
        processed_text = str(row.get("processed_text") or row.get("memory_candidate_text") or "").strip()
        if not processed_text:
            continue
        candidates.append(row)
    return candidates


def build_s1_proposal_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_primary_ref: dict[str, dict[str, Any]] = {}
    by_ref: dict[str, dict[str, Any]] = {}
    by_text: dict[str, dict[str, Any]] = {}
    for row in rows:
        for ref in proposal_primary_reference_values(row):
            by_primary_ref.setdefault(ref, row)
        for ref in proposal_reference_values(row):
            by_ref.setdefault(ref, row)
        for key in ("original_text", "source_text", "source_text_quote"):
            text_key = normalized_text_key(str(row.get(key) or ""))
            if text_key:
                by_text.setdefault(text_key, row)
    return {"by_primary_ref": by_primary_ref, "by_ref": by_ref, "by_text": by_text}


def route_row_to_s1_skip_row(row: dict[str, Any], route: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_kind": "skipped",
        "proposal_status": "skipped",
        "proposal_id": route.get("route_decision_id"),
        "proposal_input_id": row.get("text_unit_id") or row.get("span_id") or row.get("evidence_ref") or row.get("raw_span_id"),
        "route_used": route.get("recommended_route"),
        "route_recommended": route.get("recommended_route"),
        "evidence_ref": row.get("evidence_ref"),
        "canonical_evidence_ref": row.get("evidence_ref"),
        "source_specific_ref": row.get("text_unit_id") or row.get("span_id"),
        "text_unit_id": row.get("text_unit_id"),
        "raw_span_id": row.get("raw_span_id"),
        "parent_text_unit_id": row.get("parent_text_unit_id"),
        "original_text": row.get("text") or row.get("content") or "",
        "source_text": row.get("text") or row.get("content") or "",
        "warnings": sorted(set(["prebuild_route_skip_not_materialized", *(route.get("warnings") or [])])),
    }


def build_s1_route_skip_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    skip_rows: list[dict[str, Any]] = []
    for row in rows:
        for route in row.get("task_routes") or []:
            if route.get("target_task") != "s1_memory_candidate":
                continue
            if route.get("recommended_route") == "skip_or_background_only":
                skip_rows.append(route_row_to_s1_skip_row(row, route))
    return skip_rows


def build_s1_reject_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    by_primary_ref: dict[str, list[dict[str, Any]]] = {}
    by_ref: dict[str, list[dict[str, Any]]] = {}
    by_text: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("output_kind") not in {"reject", "skipped", "model_uncertain", "needs_human_review"}:
            continue
        for ref in proposal_primary_reference_values(row):
            by_primary_ref.setdefault(ref, []).append(row)
        for ref in proposal_reference_values(row):
            by_ref.setdefault(ref, []).append(row)
        for key in ("original_text", "source_text", "source_text_quote"):
            text_key = normalized_text_key(str(row.get(key) or ""))
            if text_key:
                by_text.setdefault(text_key, []).append(row)
    return {"by_primary_ref": by_primary_ref, "by_ref": by_ref, "by_text": by_text}


def select_s1_processed_proposal(
    item: dict[str, Any],
    proposal_index: dict[str, dict[str, dict[str, Any]]] | None,
) -> dict[str, Any] | None:
    if not proposal_index:
        return None
    by_primary_ref = proposal_index.get("by_primary_ref") or {}
    by_ref = proposal_index.get("by_ref") or {}
    for ref in exact_evidence_reference_values(item):
        if ref in by_primary_ref:
            return by_primary_ref[ref]
    text_key = normalized_text_key(str(item.get("text") or ""))
    if text_key:
        text_match = (proposal_index.get("by_text") or {}).get(text_key)
        if text_match:
            return text_match
    if (item.get("locator") or {}).get("kind") == "text_unit_sentence":
        return None
    for ref in evidence_reference_values(item):
        if ref in by_ref:
            return by_ref[ref]
    return None


def select_s1_reject_rows(
    item: dict[str, Any],
    reject_index: dict[str, dict[str, list[dict[str, Any]]]] | None,
) -> list[dict[str, Any]]:
    if not reject_index:
        return []
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    by_primary_ref = reject_index.get("by_primary_ref") or {}
    by_ref = reject_index.get("by_ref") or {}
    for ref in exact_evidence_reference_values(item):
        for row in by_primary_ref.get(ref, []):
            key = str(row.get("proposal_id") or row.get("proposal_input_id") or id(row))
            if key not in seen:
                matches.append(row)
                seen.add(key)
    text_key = normalized_text_key(str(item.get("text") or ""))
    if text_key:
        for row in (reject_index.get("by_text") or {}).get(text_key, []):
            key = str(row.get("proposal_id") or row.get("proposal_input_id") or id(row))
            if key not in seen:
                matches.append(row)
                seen.add(key)
    if (item.get("locator") or {}).get("kind") == "text_unit_sentence":
        return matches
    for ref in evidence_reference_values(item):
        for row in by_ref.get(ref, []):
            key = str(row.get("proposal_id") or row.get("proposal_input_id") or id(row))
            if key not in seen:
                matches.append(row)
                seen.add(key)
    return matches


def build_memory(
    inputs: BuildInputs,
    evidence: list[dict[str, Any]],
    s1_processed_proposal_index: dict[str, dict[str, dict[str, Any]]] | None = None,
    s1_reject_index: dict[str, dict[str, list[dict[str, Any]]]] | None = None,
) -> MemoryResult:
    """Build conservative evidence-bound S1 memory substrate.

    v0.1 policy:
    - every evidence item gets a preprocessing decision;
    - only target-subject utterances become memory candidates/units;
    - other-participant utterances remain relationship/context hints;
    - units stay close to source evidence and do not become S2 portrait facts.
    """

    target = inputs.modeled_subject_id
    decisions: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []

    for item in evidence:
        speaker = item.get("speaker") or item.get("participant") or "unknown"
        evidence_ref = item["evidence_ref"]
        source_specific_ref = item.get("source_specific_ref") or evidence_ref.rsplit(":", 1)[-1]
        display_ref = item.get("display_ref") or evidence_ref
        evidence_ref_key = short_hash(evidence_ref)
        is_target = speaker == target
        proposal_row = select_s1_processed_proposal(item, s1_processed_proposal_index)
        reject_rows = select_s1_reject_rows(item, s1_reject_index)
        proposal_veto = bool(is_target and not proposal_row and reject_rows)
        decision = (
            "proposal_rejected_or_uncertain"
            if proposal_veto
            else "portrait_candidate_needed"
            if is_target
            else "graph_candidate_needed"
        )
        decisions.append(
            {
                "schema_version": "step1.preprocessing_decision.v0.1",
                "decision_id": f"prep:{evidence_ref_key}",
                "evidence_ref": evidence_ref,
                "source_id": item.get("source_id"),
                "raw_source_id": item.get("raw_source_id"),
                "decision": decision,
                "route": decision,
                "reason": (
                    "Target participant utterance can feed evidence-bound S1 memory candidate/unit generation."
                    if is_target
                    else "Other participant turn may contain relationship/context; keep out of target memory units during S1 build."
                ),
                "target_subject_id": target,
                "target_participant": target,
                "speaker": speaker,
                "subject_scope": "target" if is_target else "other_participant_context",
                "subject_contamination_risk": item.get("subject_contamination_risk", "unknown"),
                "memory_class_hint": "episodic" if is_target else "relationship_context",
                "candidate_type_hint": "source_claim" if is_target else "relationship_context_hint",
                "memory_writer_action": (
                    "skip_due_to_prebuild_proposal"
                    if proposal_veto
                    else "candidate_input"
                    if is_target
                    else "context_or_relation_input"
                ),
                "source_specific_ref": source_specific_ref,
                "display_ref": display_ref,
                "warnings": (
                    ["prebuild_s1_proposal_rejected_or_uncertain"]
                    if proposal_veto
                    else []
                    if is_target
                    else ["not_target_subject", "do_not_write_as_target_memory_without_review"]
                ),
                "prebuild_proposal_refs": [
                    {
                        "proposal_id": row.get("proposal_id"),
                        "output_kind": row.get("output_kind"),
                        "proposal_status": row.get("proposal_status"),
                        "route_used": row.get("route_used"),
                        "provider": row.get("provider"),
                        "model_id": row.get("model_id"),
                    }
                    for row in reject_rows
                ],
                "created_at": now_iso(),
                "notes": "Generated by fixed Step 1 build runner from S1 evidence; no S2 assets read.",
            }
        )
        if not is_target:
            continue
        if proposal_veto:
            continue
        if item.get("s2_policy") in {"background_only", "blocked_from_portrait", "needs_review"}:
            continue

        text = item.get("text", "")
        candidate_id = f"candidate:{evidence_ref_key}"
        memory_id = f"memory:{evidence_ref_key}"
        source_type = str(item.get("source_type") or "")
        speech_verb = "wrote" if source_type in {"public_domain_text", "text", "document", "markdown"} else "said"
        script_candidate_text = f"{target} {speech_verb} in {display_ref}: {text}"
        if proposal_row:
            processed_text = str(
                proposal_row.get("processed_text") or proposal_row.get("memory_candidate_text") or ""
            ).strip()
            candidate_text = processed_text or script_candidate_text
            memory_class = str(proposal_row.get("memory_class") or "episodic")
            processing_method = "llm_assisted" if proposal_row.get("llm_assist_used") else "script"
            deterministic_processing_status = str(
                proposal_row.get("deterministic_processing_status") or "not_attempted"
            )
            deterministic_s1_processing = proposal_row.get("deterministic_s1_processing") or {}
            llm_assist_used = bool(proposal_row.get("llm_assist_used"))
            llm_assisted_s1_candidate = proposal_row.get("llm_assisted_s1_candidate") or {}
            processing_warnings = sorted(set(proposal_row.get("warnings") or []))
            generation_method = "llm_assisted_processed_text" if llm_assist_used else "script_generated"
            proposal_ref = {
                "proposal_id": proposal_row.get("proposal_id"),
                "proposal_input_id": proposal_row.get("proposal_input_id"),
                "proposal_run_id": proposal_row.get("proposal_run_id"),
                "provider": proposal_row.get("provider"),
                "model_id": proposal_row.get("model_id"),
                "route_used": proposal_row.get("route_used"),
            }
        else:
            candidate_text = script_candidate_text
            memory_class = "episodic"
            processing_method = "script"
            deterministic_processing_status = "succeeded"
            deterministic_s1_processing = {
                "script_candidate": {
                    "candidate_text": script_candidate_text,
                    "memory_class": "episodic",
                    "generation_method": "script_generated",
                }
            }
            llm_assist_used = False
            llm_assisted_s1_candidate = {}
            processing_warnings = []
            generation_method = "script_generated"
            proposal_ref = None
        raw_backpointer_refs = raw_backpointer_refs_for_evidence(item)
        origin = {
            "source_tool": "tools.step1.step1_build_runner",
            "source_workflow": "step1-build-workflow:evidence_plus_memory",
            "source_run_id": inputs.run_id,
            "generation_policy": (
                "prebuild-routed S1 processed_text materialization"
                if proposal_row
                else "memory-writer-safe boundary-aligned deterministic source-claim extraction"
            ),
            "model_id": proposal_ref.get("model_id") if proposal_ref else "",
            "notes": (
                "Generated from S1 raw evidence plus provenance-bound prebuild S1 processing; not from S2 portrait units."
                if proposal_row
                else "Generated from S1 raw evidence only; not from S2 portrait units."
            ),
        }
        if proposal_ref:
            origin["s1_processing_proposal"] = proposal_ref
        candidates.append(
            {
                "schema_version": "step1.memory_candidate.v0.1",
                "candidate_id": candidate_id,
                "source_refs": [item.get("source_id")] if item.get("source_id") else [],
                "raw_source_id": item.get("raw_source_id"),
                "evidence_refs": [evidence_ref],
                "backpointer_refs": [evidence_ref],
                "raw_backpointer_refs": raw_backpointer_refs,
                "original_text": text,
                "original_text_excerpt": text_excerpt(text),
                "processed_text": candidate_text,
                "processing_method": processing_method,
                "deterministic_processing_status": deterministic_processing_status,
                "deterministic_s1_processing": deterministic_s1_processing,
                "llm_assist_used": llm_assist_used,
                "llm_assisted_s1_candidate": llm_assisted_s1_candidate,
                "processing_warnings": processing_warnings,
                "candidate_text": candidate_text,
                "candidate_type": "source_claim",
                "memory_class": memory_class,
                "subject_id": target,
                "subject_role": item.get("subject_role", "target"),
                "target_subject_id": target,
                "generation_method": generation_method,
                "confidence": "high",
                "privacy_class": item.get("privacy_class", "unknown"),
                "review_status": "accepted_for_experiment",
                "created_at": now_iso(),
                "candidate_origin": origin,
                "source_specific_refs": [source_specific_ref],
                "display_refs": [display_ref],
                "warnings": processing_warnings,
                "section_type": item.get("section_type"),
                "s1_storage_policy": item.get("s1_storage_policy"),
                "retrieval_policy": item.get("retrieval_policy"),
                "s2_policy": item.get("s2_policy"),
                "notes": "S1 memory candidate only; not an S2 portrait unit.",
            }
        )
        units.append(
            {
                "schema_version": "step1.memory_unit.v0.1",
                "memory_id": memory_id,
                "memory_type": "source_claim",
                "memory_class": memory_class,
                "item_layer": "memory_unit",
                "content": candidate_text,
                "original_text": text,
                "original_text_excerpt": text_excerpt(text),
                "processed_text": candidate_text,
                "processing_method": processing_method,
                "deterministic_processing_status": deterministic_processing_status,
                "deterministic_s1_processing": deterministic_s1_processing,
                "llm_assist_used": llm_assist_used,
                "llm_assisted_s1_candidate": llm_assisted_s1_candidate,
                "processing_warnings": processing_warnings,
                "subject_id": target,
                "subject_role": item.get("subject_role", "target"),
                "target_subject_id": target,
                "source_refs": [item.get("source_id")] if item.get("source_id") else [],
                "raw_source_id": item.get("raw_source_id"),
                "evidence_refs": [evidence_ref],
                "backpointer_refs": [evidence_ref],
                "raw_backpointer_refs": raw_backpointer_refs,
                "generation_method": generation_method,
                "candidate_origin": origin,
                "confidence": "high",
                "privacy_class": item.get("privacy_class", "unknown"),
                "scope": "conversation",
                "status": "accepted_for_experiment",
                "created_at": now_iso(),
                "source_ref": {
                    "kind": "public_dataset_conversation",
                    "canonical_evidence_ref": evidence_ref,
                    "source_specific_ref": source_specific_ref,
                    "locator": display_ref,
                },
                "source_span": item.get("locator"),
                "source_specific_refs": [source_specific_ref],
                "display_refs": [display_ref],
                "evidence_quote": text,
                "evidence_summary": f"{target} source span at {display_ref}: {candidate_text}",
                "section_type": item.get("section_type"),
                "s1_storage_policy": item.get("s1_storage_policy"),
                "retrieval_policy": item.get("retrieval_policy"),
                "s2_policy": item.get("s2_policy"),
                "inference_level": "explicit",
                "index_targets": {"bm25": True, "embedding": True, "graph": False},
                "relation_candidates": [],
                "topic_tags": ["s1", "memory", "conversation"],
                "risk_notes": processing_warnings,
                "review_notes": (
                    "Accepted for public dataset experiment only. This S1 memory unit is evidence-bound substrate, "
                    "not an S2 portrait unit."
                ),
                "decay_policy": "review_on_conflict",
                "next_review_at": None,
                "supersedes": [],
                "superseded_by": [],
            }
        )

    memory_stats = {
        "preprocessing_decisions": len(decisions),
        "memory_candidates": len(candidates),
        "memory_units": len(units),
        "target_memory_units": len(units),
        "non_target_memory_units": 0,
        "script_processed_memory_units": sum(1 for unit in units if unit.get("processing_method") == "script"),
        "llm_assisted_memory_units": sum(1 for unit in units if unit.get("processing_method") == "llm_assisted"),
        "prebuild_proposal_vetoed_memory_units": sum(
            1
            for decision in decisions
            if decision.get("memory_writer_action") == "skip_due_to_prebuild_proposal"
        ),
    }
    return MemoryResult(decisions, candidates, units, memory_stats)


def validate_memory_result(memory: MemoryResult, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_refs = {item.get("evidence_ref") for item in evidence}
    missing_backpointers = [
        unit.get("memory_id")
        for unit in memory.memory_units
        if not unit.get("backpointer_refs")
    ]
    unresolved_backpointers = [
        unit.get("memory_id")
        for unit in memory.memory_units
        for ref in unit.get("backpointer_refs", [])
        if ref not in evidence_refs
    ]
    bad_status = [
        unit.get("memory_id")
        for unit in memory.memory_units
        if unit.get("status") != "accepted_for_experiment"
    ]
    missing_paired_text = [
        unit.get("memory_id")
        for unit in memory.memory_units
        if not unit.get("original_text") or not unit.get("processed_text")
    ]
    missing_processing_method = [
        unit.get("memory_id")
        for unit in memory.memory_units
        if unit.get("processing_method") not in {"script", "llm_assisted", "mixed", "none"}
    ]
    return {
        "evidence_items": len(evidence),
        "preprocessing_decisions": len(memory.preprocessing_decisions),
        "memory_candidates": len(memory.memory_candidates),
        "memory_units": len(memory.memory_units),
        "missing_backpointer_units": missing_backpointers,
        "unresolved_backpointer_units": unresolved_backpointers,
        "bad_status_units": bad_status,
        "missing_paired_text_units": missing_paired_text,
        "missing_processing_method_units": missing_processing_method,
        "all_units_have_backpointer_refs": not missing_backpointers,
        "all_unit_refs_resolve_to_evidence": not unresolved_backpointers,
        "all_units_status_experiment": not bad_status,
        "all_units_have_paired_text": not missing_paired_text,
        "all_units_have_processing_method": not missing_processing_method,
        "valid": (
            len(memory.preprocessing_decisions) == len(evidence)
            and not missing_backpointers
            and not unresolved_backpointers
            and not bad_status
            and not missing_paired_text
            and not missing_processing_method
        ),
    }


def truncate_text(text: str, limit: int = 140) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def looks_like_commitment(text: str) -> bool:
    lowered = text.lower()
    markers = [
        "i will",
        "i'll",
        "going to",
        "gonna",
        "plan",
        "want to",
        "wanna",
        "need to",
        "should",
        "let's",
        "we should",
        "can't wait",
    ]
    return any(marker in lowered for marker in markers)


def sort_key_from_locator(item: dict[str, Any]) -> tuple[int, str]:
    locator = item.get("locator") or {}
    raw_turn = locator.get("turn_index") or locator.get("turn_id") or item.get("turn_id") or ""
    if isinstance(raw_turn, int):
        return raw_turn, ""
    raw_text = str(raw_turn)
    digits = "".join(ch for ch in raw_text if ch.isdigit())
    if digits:
        return int(digits), raw_text
    return 0, raw_text


def build_summaries(inputs: BuildInputs, evidence: list[dict[str, Any]], source_evidence_hash: str) -> SummaryResult:
    non_conversation_refs: list[str] = []
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for item in evidence:
        locator = item.get("locator") or {}
        if locator.get("kind") != "conversation_turn" or not locator.get("session"):
            non_conversation_refs.append(str(item.get("evidence_ref") or ""))
            continue
        source_id = item.get("source_id") or "unknown_source"
        record_id = item.get("record_id") or locator.get("record_id") or "unknown_record"
        session = locator["session"]
        session_index = int(locator.get("session_index") or 0)
        grouped.setdefault((source_id, record_id, session, session_index), []).append(item)

    if non_conversation_refs and not grouped:
        return SummaryResult(
            summaries=[],
            summary_stats={
                "summaries": 0,
                "summary_type": "not_applicable_non_conversation",
                "non_conversation_evidence_items": len(non_conversation_refs),
                "summary_skip_reason": "non_conversation_plaintext_has_no_session_summary",
            },
        )

    summaries: list[dict[str, Any]] = []
    for (source_id, record_id, session, session_index), rows in sorted(grouped.items(), key=lambda pair: pair[0]):
        rows = sorted(rows, key=sort_key_from_locator)
        evidence_refs = [item["evidence_ref"] for item in rows]
        participants = sorted({item.get("speaker") or item.get("participant") or "unknown" for item in rows})
        target_rows = [item for item in rows if (item.get("speaker") or item.get("participant")) == inputs.modeled_subject_id]
        source_refs = sorted({item.get("source_id") for item in rows if item.get("source_id")})
        raw_source_ids = sorted({item.get("raw_source_id") for item in rows if item.get("raw_source_id")})
        display_refs = [item.get("display_ref") or item["evidence_ref"] for item in rows]
        source_specific_refs = [item.get("source_specific_ref") or item["evidence_ref"] for item in rows]
        timestamp = (rows[0].get("locator") or {}).get("timestamp") or rows[0].get("timestamp")
        adapter_run_id = rows[0].get("adapter_run_id", "")
        main_events = [
            {"evidence_ref": item["evidence_ref"], "text": truncate_text(item.get("text", ""))}
            for item in (target_rows or rows)[:3]
        ]
        explicit_commitments = [
            {"evidence_ref": item["evidence_ref"], "text": truncate_text(item.get("text", ""))}
            for item in target_rows
            if looks_like_commitment(item.get("text", ""))
        ][:5]
        participant_breakdown: dict[str, dict[str, Any]] = {}
        for participant in participants:
            participant_rows = [
                item
                for item in rows
                if (item.get("speaker") or item.get("participant") or "unknown") == participant
            ]
            snippets = [truncate_text(item.get("text", ""), 110) for item in participant_rows[:3]]
            more_count = max(0, len(participant_rows) - len(snippets))
            summary = " | ".join(snippets)
            if more_count:
                summary += f" (+{more_count} more turns)"
            participant_breakdown[participant] = {
                "evidence_refs": [item["evidence_ref"] for item in participant_rows],
                "summary": summary,
            }

        subject_scope = "target_only" if participants == [inputs.modeled_subject_id] else "mixed_participant"
        summary_text = (
            f"{session} contains {len(rows)} evidence turns involving {', '.join(participants)}. "
            f"{inputs.modeled_subject_id} has {len(target_rows)} target turns. "
            "Use evidence refs before asserting facts."
        )
        structured_summary = {
            "session_topic": f"extractive conversation summary for {session}",
            "main_events": main_events,
            "explicit_commitments": explicit_commitments,
            "participant_breakdown": participant_breakdown,
            "uncertainties": [],
        }
        group_key = json.dumps(
            {
                "workspace_id": inputs.workspace_id,
                "source_id": source_id,
                "record_id": record_id,
                "session": session,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        summary_id = f"summary:{short_hash(group_key)}"
        base = {
            "schema_version": "step1.summary_unit.v0.1",
            "summary_id": summary_id,
            "summary_type": "session_summary",
            "item_layer": "doc_level_summary",
            "workspace_id": inputs.workspace_id,
            "modeled_subject_id": inputs.modeled_subject_id,
            "source_id": source_id,
            "source_refs": source_refs,
            "raw_source_id": raw_source_ids[0] if raw_source_ids else "",
            "adapter_run_id": adapter_run_id,
            "record_id": record_id,
            "group_locator": {
                "kind": "conversation_session",
                "source_id": source_id,
                "record_id": record_id,
                "session": session,
                "session_index": session_index,
                "timestamp": timestamp,
            },
            "summary_text": summary_text,
            "structured_summary": structured_summary,
            "evidence_refs": evidence_refs,
            "source_specific_refs": source_specific_refs,
            "display_refs": display_refs,
            "backpointer_refs": list(evidence_refs),
            "participants": participants,
            "target_participant": inputs.modeled_subject_id,
            "subject_scope": subject_scope,
            "memory_class": "episodic",
            "coverage_policy": {
                "mode": "full_session",
                "included_evidence_count": len(evidence_refs),
                "excluded_evidence_count": 0,
                "coverage_notes": f"All evidence items with locator.session={session} were included.",
            },
            "generation_method": "script_assisted_local_summary",
            "generation_policy": "conservative_structured_extractive_v0.1",
            "generator": "tools.step1.step1_build_runner",
            "script_version": "s1_session_summary_v0.1",
            "model_id": "",
            "source_build_manifest_ref": "evidence/build_manifest.json",
            "source_evidence_hash": source_evidence_hash,
            "source_asset_version": inputs.run_id,
            "generated_from_evidence_count": len(evidence_refs),
            "confidence": "high",
            "privacy_class": rows[0].get("privacy_class", "unknown"),
            "status": "generated_context",
            "created_at": now_iso(),
            "warnings": ["doc_level_summary_not_raw_evidence"],
            "notes": "Summary is context/compression only. Factual claims require evidence refs and support checks.",
        }
        if subject_scope == "mixed_participant":
            base["warnings"].append("mixed_participant_context_requires_subject_lock")
        base["summary_hash"] = sha256_text(
            json.dumps(
                {
                    "summary_text": base["summary_text"],
                    "structured_summary": base["structured_summary"],
                    "evidence_refs": base["evidence_refs"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        summaries.append(base)

    return SummaryResult(
        summaries=summaries,
        summary_stats={
            "summaries": len(summaries),
            "summary_type": "session_summary",
            "non_conversation_evidence_items": len(non_conversation_refs),
        },
    )


def validate_summary_result(
    summaries: SummaryResult,
    evidence: list[dict[str, Any]],
    source_evidence_hash: str,
) -> dict[str, Any]:
    evidence_refs = {item.get("evidence_ref") for item in evidence}
    empty_evidence = [item.get("summary_id") for item in summaries.summaries if not item.get("evidence_refs")]
    unresolved_refs = [
        item.get("summary_id")
        for item in summaries.summaries
        for ref in item.get("evidence_refs", [])
        if ref not in evidence_refs
    ]
    missing_backpointers = [
        item.get("summary_id") for item in summaries.summaries if not item.get("backpointer_refs")
    ]
    wrong_layer = [
        item.get("summary_id") for item in summaries.summaries if item.get("item_layer") != "doc_level_summary"
    ]
    wrong_type = [
        item.get("summary_id") for item in summaries.summaries if item.get("summary_type") != "session_summary"
    ]
    stale_hash = [
        item.get("summary_id") for item in summaries.summaries if item.get("source_evidence_hash") != source_evidence_hash
    ]
    bad_counts = [
        item.get("summary_id")
        for item in summaries.summaries
        if item.get("generated_from_evidence_count") != len(item.get("evidence_refs", []))
    ]
    bad_status = [
        item.get("summary_id") for item in summaries.summaries if item.get("status") == "accepted_by_user"
    ]
    return {
        "summaries": len(summaries.summaries),
        "empty_evidence_summaries": empty_evidence,
        "unresolved_evidence_ref_summaries": unresolved_refs,
        "missing_backpointer_summaries": missing_backpointers,
        "wrong_item_layer_summaries": wrong_layer,
        "wrong_summary_type_summaries": wrong_type,
        "stale_source_hash_summaries": stale_hash,
        "bad_generated_count_summaries": bad_counts,
        "bad_status_summaries": bad_status,
        "valid": not any(
            [empty_evidence, unresolved_refs, missing_backpointers, wrong_layer, wrong_type, stale_hash, bad_counts, bad_status]
        ),
    }


def source_identity_coverage(evidence: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "evidence_items": len(evidence),
        "with_raw_source_id": sum(1 for item in evidence if item.get("raw_source_id")),
        "with_source_id": sum(1 for item in evidence if item.get("source_id")),
        "with_record_id": sum(1 for item in evidence if item.get("record_id")),
        "with_source_specific_ref": sum(1 for item in evidence if item.get("source_specific_ref")),
        "with_locator_or_unavailable_reason": sum(
            1 for item in evidence if item.get("locator") or item.get("locator_unavailable_reason")
        ),
    }


def output_object_count(
    result: AdapterResult,
    memory_result: MemoryResult | None,
    summary_result: SummaryResult | None,
) -> int:
    count = len(result.source_manifest) + len(result.evidence)
    if memory_result is not None:
        count += (
            len(memory_result.preprocessing_decisions)
            + len(memory_result.memory_candidates)
            + len(memory_result.memory_units)
        )
    if summary_result is not None:
        count += len(summary_result.summaries)
    return count


def transformation_policy(inputs: BuildInputs, result: AdapterResult) -> dict[str, Any]:
    adapter_names = sorted({route.get("adapter_name") for route in result.adapter_routes if route.get("adapter_name")})
    return {
        "run_scope": inputs.run_scope,
        "adapter_names": adapter_names,
        "adapter_routes": result.adapter_routes,
        "segmentation_policy": (
            "adapter_defined; conversation adapters emit turns; generic_text_adapter consumes S0B sentence "
            "text_units when available and falls back to text_span segments"
        ),
        "ref_policy": "adapters generate canonical evidence_ref and preserve source_specific_ref/display_ref separately",
        "locator_policy": "every evidence item must preserve locator or locator_unavailable_reason",
        "memory_preprocessing_policy": (
            "target-subject evidence becomes S1 source-claim candidates/units; other-subject evidence stays context hints"
            if inputs.run_scope in {"evidence_plus_memory", "full_s1_build"}
            else "memory preprocessing skipped for evidence_only"
        ),
        "summary_policy": (
            "conversation/session summaries only; non-conversation evidence is unsupported in full_s1_build v0.1"
            if inputs.run_scope == "full_s1_build"
            else "summary generation skipped"
        ),
        "background_only_rule": "background_only raw sources are recorded in source_manifest but not converted into ordinary raw_evidence",
        "no_step2_assets_generated": True,
    }


def write_build_outputs(
    inputs: BuildInputs,
    result: AdapterResult,
    preparation: OutputPreparation,
    prebuild_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_manifest_path = inputs.output_workspace / "evidence" / "source_manifest.jsonl"
    evidence_path = inputs.output_workspace / "evidence" / "evidence.jsonl"
    build_manifest_path = inputs.output_workspace / "evidence" / "build_manifest.json"
    checkpoint_path = inputs.output_workspace / "checkpoints" / "s1-build-status.json"
    report_path = inputs.output_workspace / "reports" / "s1-build-report.md"
    memory_result: MemoryResult | None = None
    memory_validation: dict[str, Any] | None = None
    summary_result: SummaryResult | None = None
    summary_validation: dict[str, Any] | None = None

    validation = validate_result(result)
    if not validation["valid"]:
        raise ValueError(f"Build validation failed: {validation}")

    source_evidence_hash = jsonl_hash(result.evidence)
    s1_prebuild_proposal_rows = load_s1_prebuild_proposal_rows(prebuild_result)
    s1_prebuild_route_skip_rows = build_s1_route_skip_rows(load_s1_prebuild_route_rows(prebuild_result))
    s1_processed_proposal_rows = [
        row
        for row in s1_prebuild_proposal_rows
        if row.get("output_kind") == "memory_candidate"
        and str(row.get("processed_text") or row.get("memory_candidate_text") or "").strip()
    ]
    s1_processed_proposal_index = build_s1_proposal_index(s1_processed_proposal_rows)
    s1_reject_index = build_s1_reject_index([*s1_prebuild_proposal_rows, *s1_prebuild_route_skip_rows])

    if inputs.run_scope in {"evidence_plus_memory", "full_s1_build"}:
        memory_result = build_memory(inputs, result.evidence, s1_processed_proposal_index, s1_reject_index)
        memory_validation = validate_memory_result(memory_result, result.evidence)
        if not memory_validation["valid"]:
            raise ValueError(f"Memory build validation failed: {memory_validation}")

    if inputs.run_scope == "full_s1_build":
        summary_result = build_summaries(inputs, result.evidence, source_evidence_hash)
        summary_validation = validate_summary_result(summary_result, result.evidence, source_evidence_hash)
        if not summary_validation["valid"]:
            raise ValueError(f"Summary build validation failed: {summary_validation}")

    write_jsonl(source_manifest_path, result.source_manifest)
    write_jsonl(evidence_path, result.evidence)
    if memory_result is not None:
        write_jsonl(inputs.output_workspace / "memory" / "preprocessing_decisions.jsonl", memory_result.preprocessing_decisions)
        write_jsonl(inputs.output_workspace / "memory" / "memory_candidates.jsonl", memory_result.memory_candidates)
        write_jsonl(inputs.output_workspace / "memory" / "memory_units.jsonl", memory_result.memory_units)
    if summary_result is not None:
        write_jsonl(inputs.output_workspace / "memory" / "summaries.jsonl", summary_result.summaries)

    raw_source_hashes: dict[str, str] = {}
    for source in inputs.raw_sources:
        path = resolve_path(inputs.project_root, inputs.workspace, source)
        raw_source_hashes[safe_relative(path, inputs.project_root)] = file_hash(path) if path.exists() else ""

    manifest = {
        "schema_version": "step1.build_manifest.v0.1",
        "run_id": inputs.run_id,
        "workflow": "step1-build-workflow",
        "runner": "tools.step1.step1_build_runner",
        "run_scope": inputs.run_scope,
        "run_type": "evidence_build",
        "started_at": inputs.started_at,
        "completed_at": now_iso(),
        "status": "completed",
        "workspace_id": inputs.workspace_id,
        "modeled_subject_id": inputs.modeled_subject_id,
        "adapter_routes": result.adapter_routes,
        "input_s0b_assets": {
            "bundle": "raw/organization/bundle.json",
            "raw_sources": "raw/organization/raw_sources.jsonl",
        },
        "input_raw_sources": [
            {
                "raw_source_id": source.get("raw_source_id"),
                "path": source.get("local_path") or source.get("original_uri_or_path"),
                "hash_sha256": source.get("content_hash"),
                "inclusion_decision": source.get("inclusion_decision"),
            }
            for source in inputs.raw_sources
        ],
        "input_raw_source_hashes": raw_source_hashes,
        "output_assets": {
            "source_manifest": "evidence/source_manifest.jsonl",
            "evidence_registry": "evidence/evidence.jsonl",
        },
        "output_hashes": {
            "evidence/source_manifest.jsonl": file_hash(source_manifest_path),
            "evidence/evidence.jsonl": source_evidence_hash,
        },
        "build_result_counts": {
            **{
                "source_manifest_items": len(result.source_manifest),
                "evidence_items": len(result.evidence),
            },
            **result.source_stats,
        },
        "output_counts": {
            "source_manifest_items": len(result.source_manifest),
            "evidence_items": len(result.evidence),
        },
        "ref_policy": {
            "canonical_evidence_ref_note": "Adapter-specific examples are not universal ref naming strategies.",
            "source_specific_ref_note": "Source-local refs are preserved as source_specific_ref/display_ref/ref_alias.",
            "compatibility_aliases": "Transitional aliases are for compatibility only; new artifacts should use canonical evidence_ref.",
        },
        "locator_policy": {
            "requirement": "Every evidence item must have locator or locator_unavailable_reason.",
            "note": "conversation_turn is one locator type, not the universal evidence model.",
        },
        "duplicate_policy": inputs.duplicate_policy,
        "checkpoint_paths": preparation.checkpoint_paths,
        "stale_higher_scope_assets": preparation.stale_higher_scope_assets,
        "stale_asset_policy": preparation.stale_asset_policy,
        "source_identity_coverage": source_identity_coverage(result.evidence),
        "write_result_counts": {
            "added": output_object_count(result, memory_result, summary_result),
            "updated": 0,
            "skipped": 0,
            "overwritten": len(preparation.overwritten_paths),
            "failed": 0,
        },
        "transformation_policy": transformation_policy(inputs, result),
        "validation_summary": validation,
        "limitations": [
            "This runner currently supports evidence_only builds.",
            "This runner does not build memory units, summaries, indexes, Step 2 assets, or final answers.",
            "LoCoMo-style refs remain source-specific and must not define core S1 evidence semantics.",
        ],
    }

    if memory_result is not None and memory_validation is not None:
        memory_manifest_path = inputs.output_workspace / "memory" / "memory_build_manifest.json"
        memory_outputs = {
            "preprocessing_decisions": "memory/preprocessing_decisions.jsonl",
            "memory_candidates": "memory/memory_candidates.jsonl",
            "memory_units": "memory/memory_units.jsonl",
            "memory_build_manifest": "memory/memory_build_manifest.json",
        }
        memory_manifest = {
            "schema_version": "step1.memory_build_manifest.v0.1",
            "run_id": inputs.run_id,
            "workflow": "step1-build-workflow",
            "runner": "tools.step1.step1_build_runner",
            "run_scope": inputs.run_scope,
            "run_type": "memory_preprocessing_and_unit_build",
            "started_at": manifest["started_at"],
            "completed_at": now_iso(),
            "status": "completed",
            "workspace_id": inputs.workspace_id,
            "modeled_subject_id": inputs.modeled_subject_id,
            "input_assets": {
                "evidence_registry": "evidence/evidence.jsonl",
                "evidence_build_manifest": "evidence/build_manifest.json",
            },
            "input_hashes": {
                "evidence/evidence.jsonl": file_hash(evidence_path),
            },
            "output_assets": memory_outputs,
            "output_hashes": {
                "memory/preprocessing_decisions.jsonl": file_hash(
                    inputs.output_workspace / "memory" / "preprocessing_decisions.jsonl"
                ),
                "memory/memory_candidates.jsonl": file_hash(inputs.output_workspace / "memory" / "memory_candidates.jsonl"),
                "memory/memory_units.jsonl": file_hash(inputs.output_workspace / "memory" / "memory_units.jsonl"),
            },
            "validation_summary": memory_validation,
            "generation_policy": {
                "method": (
                    "script_plus_prebuild_llm_processed_text"
                    if memory_result.memory_stats.get("llm_assisted_memory_units")
                    else "script_generated"
                ),
                "memory_writer_safe_boundary_aligned": True,
                "target_subject_rule": "Only target-subject utterances become S1 memory candidates/units.",
                "other_participant_rule": "Other participant utterances become graph/relationship context hints only.",
                "no_step2_assets_read": True,
                "paired_text_contract": "Every S1 memory unit carries original_text and processed_text with evidence refs.",
                "llm_processed_text_rule": (
                    "Prebuild LLM proposal rows may populate processed_text, but canonical evidence refs/backpointers "
                    "come from S1 evidence and original_text is preserved."
                ),
                "s1_processed_proposal_rows_available": len(s1_processed_proposal_rows),
                "s1_route_skip_rows_available": len(s1_prebuild_route_skip_rows),
            },
            "limitations": [
                "Generated units are source-claim substrate, not S2 reviewed portrait units.",
                (
                    "LLM-assisted processed_text was materialized for selected rows."
                    if memory_result.memory_stats.get("llm_assisted_memory_units")
                    else "No LLM-assisted processed_text was materialized."
                ),
            ],
        }
        write_json(memory_manifest_path, memory_manifest)
        manifest["output_assets"].update(memory_outputs)
        manifest["output_hashes"].update(memory_manifest["output_hashes"])
        manifest["output_hashes"]["memory/memory_build_manifest.json"] = file_hash(memory_manifest_path)
        manifest["build_result_counts"].update(memory_result.memory_stats)
        manifest["validation_summary"]["memory_validation"] = memory_validation
        manifest["limitations"] = [
            "This runner currently supports evidence_only, evidence_plus_memory, and full_s1_build.",
            "This runner does not build indexes, Step 2 assets, or final answers.",
            "Memory units are source-claim substrate and must not be treated as Step 2 portrait units.",
            "LoCoMo-style refs remain source-specific and must not define core S1 evidence semantics.",
        ]

    if summary_result is not None and summary_validation is not None:
        summary_output = {"summaries": "memory/summaries.jsonl"}
        manifest["output_assets"].update(summary_output)
        manifest["output_hashes"]["memory/summaries.jsonl"] = file_hash(
            inputs.output_workspace / "memory" / "summaries.jsonl"
        )
        manifest["build_result_counts"].update(summary_result.summary_stats)
        manifest["validation_summary"]["summary_validation"] = summary_validation
        manifest["limitations"] = [
            "This runner currently supports evidence_only, evidence_plus_memory, and full_s1_build.",
            "This runner does not build indexes, Step 2 assets, or final answers.",
            "Summaries are context compression only and must not replace raw evidence.",
            "LoCoMo-style refs remain source-specific and must not define core S1 evidence semantics.",
        ]

    if prebuild_result is not None:
        manifest["prebuild_routing"] = prebuild_result

    write_json(build_manifest_path, manifest)

    status = {
        "schema_version": "step1.build_status.v0.1",
        "workspace_id": inputs.workspace_id,
        "run_id": inputs.run_id,
        "run_scope": inputs.run_scope,
        "status": "completed",
        "completed_at": now_iso(),
        "outputs": manifest["output_assets"],
        "counts": manifest["build_result_counts"],
        "next_recommended_scope": (
            "evidence_plus_memory"
            if inputs.run_scope == "evidence_only"
            else "full_s1_build"
            if inputs.run_scope == "evidence_plus_memory"
            else "subagent_blind_trial"
        ),
        "blocking_findings": [],
        "non_blocking_findings": [
            (
                "Fixed evidence_only build runner is available; memory and summaries remain out of scope."
                if inputs.run_scope == "evidence_only"
                else "Fixed evidence_plus_memory build runner is available; summaries remain out of scope."
                if inputs.run_scope == "evidence_plus_memory"
                else "Fixed full_s1_build runner is available; subagent blind trial remains pending."
            )
        ],
    }
    if memory_validation is not None:
        status["memory_counts"] = memory_validation
    if summary_validation is not None:
        status["summary_counts"] = summary_validation
    write_json(checkpoint_path, status)

    report = build_report(inputs, manifest)
    report_path.write_text(report, encoding="utf-8")
    return {
        "workspace": str(inputs.output_workspace),
        "source_manifest": str(source_manifest_path),
        "evidence": str(evidence_path),
        "build_manifest": str(build_manifest_path),
        "checkpoint": str(checkpoint_path),
        "report": str(report_path),
        "counts": manifest["build_result_counts"],
        "prebuild_routing": prebuild_result,
    }


def build_report(inputs: BuildInputs, manifest: dict[str, Any]) -> str:
    counts = manifest["build_result_counts"]
    write_counts = manifest.get("write_result_counts", {})
    identity = manifest.get("source_identity_coverage", {})
    transform = manifest.get("transformation_policy", {})
    validation = manifest.get("validation_summary", {})
    limitations = "\n".join(f"- {item}" for item in manifest.get("limitations", []))
    checkpoint_paths = manifest.get("checkpoint_paths", [])
    checkpoint_lines = "\n".join(f"- `{path}`" for path in checkpoint_paths) if checkpoint_paths else "- none"
    stale_assets = manifest.get("stale_higher_scope_assets", [])
    stale_lines = "\n".join(f"- `{path}`" for path in stale_assets) if stale_assets else "- none"
    adapter_lines = "\n".join(
        f"- {route.get('raw_source_id')}: `{route.get('adapter_name')}`"
        for route in manifest.get("adapter_routes", [])
    ) or "- none"
    count_keys = [
        "source_manifest_items",
        "evidence_items",
        "included_raw_sources",
        "background_only_raw_sources",
        "session_count",
        "target_utterance_count",
        "other_utterance_count",
        "fixture_evidence_count",
        "preprocessing_decisions",
        "memory_candidates",
        "memory_units",
        "summaries",
    ]
    count_lines = "\n".join(f"- {key}: {counts[key]}" for key in count_keys if key in counts)
    return f"""# Step 1 Build Runner Report

## Status

- status: `{manifest["status"]}`
- run_id: `{manifest["run_id"]}`
- run_scope: `{inputs.run_scope}`
- duplicate_policy: `{inputs.duplicate_policy}`
- workspace_id: `{inputs.workspace_id}`
- output_workspace: `{inputs.output_workspace}`
- started_at: `{manifest["started_at"]}`
- completed_at: `{manifest["completed_at"]}`

## Outputs

- source_manifest: `evidence/source_manifest.jsonl`
- evidence_registry: `evidence/evidence.jsonl`
- build_manifest: `evidence/build_manifest.json`
- checkpoint: `checkpoints/s1-build-status.json`
{memory_output_lines(manifest)}

## Counts

{count_lines}

## Write Accounting

- added: {write_counts.get("added", 0)}
- updated: {write_counts.get("updated", 0)}
- skipped: {write_counts.get("skipped", 0)}
- overwritten: {write_counts.get("overwritten", 0)}
- failed: {write_counts.get("failed", 0)}

## Source Identity Coverage

- evidence_items: {identity.get("evidence_items", 0)}
- with_raw_source_id: {identity.get("with_raw_source_id", 0)}
- with_source_id: {identity.get("with_source_id", 0)}
- with_record_id: {identity.get("with_record_id", 0)}
- with_source_specific_ref: {identity.get("with_source_specific_ref", 0)}
- with_locator_or_unavailable_reason: {identity.get("with_locator_or_unavailable_reason", 0)}

## Adapter Routes

{adapter_lines}

## Transformation Policy

- segmentation_policy: {transform.get("segmentation_policy", "")}
- ref_policy: {transform.get("ref_policy", "")}
- locator_policy: {transform.get("locator_policy", "")}
- memory_preprocessing_policy: {transform.get("memory_preprocessing_policy", "")}
- summary_policy: {transform.get("summary_policy", "")}

## Validation Summary

- evidence_valid: {validation.get("valid")}
- duplicate_evidence_refs: {validation.get("duplicate_evidence_refs")}
- missing_locator_or_reason: {validation.get("missing_locator_or_reason")}
- missing_raw_source_id: {validation.get("missing_raw_source_id")}
- missing_adapter_run_id: {validation.get("missing_adapter_run_id")}

## Checkpoints And Stale Assets

- stale_asset_policy: `{manifest.get("stale_asset_policy")}`
- checkpoint_paths:
{checkpoint_lines}
- stale_higher_scope_assets:
{stale_lines}

## Boundary Notes

- S0B raw source is not S1 evidence.
- `background_only` raw sources are not converted into ordinary raw evidence.
- Source-specific refs are not the universal S1 evidence model.
- This runner does not build S1 indexes, S2 assets, or final answers.

## Limitations

{limitations}
"""


def memory_output_lines(manifest: dict[str, Any]) -> str:
    if "memory_units" not in manifest.get("output_assets", {}):
        return ""
    lines = "\n- preprocessing_decisions: `memory/preprocessing_decisions.jsonl`\n- memory_candidates: `memory/memory_candidates.jsonl`\n- memory_units: `memory/memory_units.jsonl`"
    if "memory_build_manifest" in manifest.get("output_assets", {}):
        lines += "\n- memory_build_manifest: `memory/memory_build_manifest.json`"
    if "summaries" in manifest.get("output_assets", {}):
        lines += "\n- summaries: `memory/summaries.jsonl`"
    return lines


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    started_at = now_iso()
    project_root = Path(args.project_root).resolve()
    workspace = (project_root / args.workspace).resolve() if not Path(args.workspace).is_absolute() else Path(args.workspace).resolve()
    output_workspace = (
        (project_root / args.output_workspace).resolve()
        if args.output_workspace and not Path(args.output_workspace).is_absolute()
        else Path(args.output_workspace).resolve()
        if args.output_workspace
        else workspace
    )
    if args.run_scope not in SUPPORTED_RUN_SCOPES:
        raise ValueError(f"Unsupported run_scope {args.run_scope}; supported: {sorted(SUPPORTED_RUN_SCOPES)}")
    if args.duplicate_policy not in SUPPORTED_DUPLICATE_POLICIES:
        raise ValueError(
            f"Unsupported duplicate_policy {args.duplicate_policy}; supported: {sorted(SUPPORTED_DUPLICATE_POLICIES)}"
        )
    inputs = load_inputs(
        project_root=project_root,
        workspace=workspace,
        output_workspace=output_workspace,
        run_scope=args.run_scope,
        duplicate_policy=args.duplicate_policy,
        workspace_id=args.workspace_id,
        modeled_subject_id=args.modeled_subject_id,
        run_id=args.run_id,
        started_at=started_at,
    )
    preparation = prepare_output_workspace(inputs)
    copy_s0b_assets(inputs)
    prebuild_options = options_from_args(
        args,
        step_name="s1",
        target_task="s1_memory_candidate",
        default_profile=DEFAULT_S1_PROPOSAL_PROFILE,
    )
    prebuild_result = run_prebuild_routing(
        project_root=project_root,
        route_workspace=workspace,
        proposal_workspace=workspace,
        output_workspace=output_workspace,
        base_run_id=inputs.run_id,
        options=prebuild_options,
    )
    result = build_evidence(inputs)
    return write_build_outputs(inputs, result, preparation, prebuild_result)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a Step 1 build from S0B organized raw sources.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current working directory.")
    parser.add_argument("--workspace", required=True, help="Input workspace containing raw/organization/raw_sources.jsonl.")
    parser.add_argument("--output-workspace", help="Output workspace. Defaults to --workspace.")
    parser.add_argument("--workspace-id", help="Override output workspace_id recorded in manifests.")
    parser.add_argument("--modeled-subject-id", help="Override modeled subject id.")
    parser.add_argument("--run-id", help="Override run id.")
    parser.add_argument("--run-scope", default="evidence_only", help="evidence_only, evidence_plus_memory, or full_s1_build.")
    parser.add_argument("--duplicate-policy", default="fail", help="fail or overwrite_generated.")
    add_prebuild_arguments(parser, default_profile=DEFAULT_S1_PROPOSAL_PROFILE)
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    output = run_build(args)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
