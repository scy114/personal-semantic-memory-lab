"""Build v0.3 graph construction packets from existing upstream assets.

This is the first graph-construction slice. It mirrors GraphRAG's first
practical step, preparing text-unit-level inputs for later extraction, while
consuming this project's own upstream S1/S2 assets directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "graph_v03.construction_packets.v0.1"
PACKET_SCHEMA_VERSION = "graph_v03.construction_packet.v0.1"
GRAPH_TEXT_UNIT_SCHEMA_VERSION = "graph_v03.graph_text_unit.v0.1"
GRAPH_CONTEXT_NEIGHBOR_WINDOW = 2


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def short_hash(value: str, length: int = 12) -> str:
    return sha256_text(value)[:length]


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}:{short_hash(value)}"


def file_hash(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.exists() else None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def string_list(value: Any) -> list[str]:
    out: list[str] = []
    for item in as_list(value):
        if item is None:
            continue
        if isinstance(item, dict):
            out.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def unique_strings(*values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for item in string_list(value):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def first_non_empty(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def load_modeled_user(workspace: Path, reviewed_units: list[dict[str, Any]]) -> str:
    manifest = workspace / "manifest.yaml"
    if manifest.exists():
        for line in manifest.read_text(encoding="utf-8-sig").splitlines():
            if line.startswith("modeled_user_id:") or line.startswith("target_participant:"):
                value = line.split(":", 1)[1].strip()
                if value:
                    return value
    for row in reviewed_units:
        value = row.get("user_id")
        if value:
            return str(value)
    return workspace.name


def referenced_proposal_ids(*row_sets: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for rows in row_sets:
        for row in rows:
            proposal_origin = row.get("proposal_origin") or {}
            for value in unique_strings(
                proposal_origin.get("proposal_id"),
                row.get("proposal_id"),
                row.get("proposal_refs"),
                row.get("input_refs") if row.get("input_layer") == "s2_proposal_outcome" else None,
                (row.get("step1_origin") or {}).get("input_refs")
                if (row.get("step1_origin") or {}).get("input_layer") == "s2_proposal_outcome"
                else None,
            ):
                if value.startswith("s2p:") or value.startswith("s1p:"):
                    ids.add(value)
    return ids


def load_proposal_rows(workspace: Path, allowed_proposal_ids: set[str]) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    paths: list[Path] = []
    if not allowed_proposal_ids:
        return rows, paths
    for path in sorted((workspace / "proposals").glob("**/proposal_outcomes.ai.jsonl")):
        loaded_from_path = False
        for row in read_jsonl(path):
            proposal_id = str(row.get("proposal_id") or "")
            if proposal_id not in allowed_proposal_ids:
                continue
            copy = dict(row)
            copy.setdefault("_source_path", str(path.relative_to(workspace)))
            rows.append(copy)
            loaded_from_path = True
        if loaded_from_path:
            paths.append(path)
    return rows, paths


def index_evidence(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in [
            row.get("evidence_ref"),
            row.get("canonical_evidence_ref"),
            row.get("source_specific_ref"),
            row.get("text_unit_id"),
            row.get("step1_evidence_ref"),
        ]:
            if key:
                index[str(key)] = row
        for alias in row.get("ref_aliases") or []:
            index[str(alias)] = row
    return index


def index_preprocessing_decisions(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        for key in [row.get("evidence_ref"), row.get("memory_id"), row.get("candidate_id")]:
            if key:
                index[str(key)].append(row)
    return dict(index)


def index_review_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("candidate_id")): row for row in rows if row.get("candidate_id")}


def evidence_ref(row: dict[str, Any]) -> str:
    return first_non_empty(row.get("evidence_ref"), row.get("canonical_evidence_ref"), row.get("step1_evidence_ref"))


def evidence_sort_key(index: int, row: dict[str, Any]) -> tuple[Any, ...]:
    locator = row.get("locator") or {}
    return (
        row.get("raw_source_id") or row.get("source_id") or row.get("record_id") or "",
        locator.get("day_index") if locator.get("day_index") is not None else row.get("day_index") or 0,
        locator.get("session_index") if locator.get("session_index") is not None else row.get("session_index") or 0,
        locator.get("turn_index") if locator.get("turn_index") is not None else row.get("turn_index") or index,
        index,
    )


def evidence_group_key(row: dict[str, Any]) -> tuple[str, str, str]:
    locator = row.get("locator") or {}
    return (
        str(row.get("raw_source_id") or row.get("source_id") or ""),
        str(locator.get("day") or locator.get("day_index") or row.get("day") or ""),
        str(locator.get("session") or locator.get("session_index") or row.get("session") or ""),
    )


def speaker_for_evidence(row: dict[str, Any]) -> str:
    locator = row.get("locator") or {}
    return first_non_empty(row.get("speaker"), row.get("participant"), locator.get("speaker"), default="unknown")


def evidence_display_line(row: dict[str, Any]) -> str:
    speaker = speaker_for_evidence(row)
    text = first_non_empty(row.get("text"), row.get("source_text"))
    return f"{speaker}: {text}" if speaker else text


def build_graph_text_units(evidence_rows: list[dict[str, Any]], workspace_id: str) -> list[dict[str, Any]]:
    """Create GraphRAG-style context envelopes from fine S0B/evidence units.

    S1/S2 can keep fine-grained evidence units, but graph construction needs a
    context-bearing text unit so pronouns, replies, and short dialogue turns are
    not routed as isolated fragments.
    """

    indexed = [(index, row) for index, row in enumerate(evidence_rows) if first_non_empty(row.get("text")) and evidence_ref(row)]
    grouped: dict[tuple[str, str, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for index, row in indexed:
        grouped[evidence_group_key(row)].append((index, row))

    units: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        ordered = sorted(group_rows, key=lambda item: evidence_sort_key(item[0], item[1]))
        rows_only = [row for _, row in ordered]
        for position, row in enumerate(rows_only):
            ref = evidence_ref(row)
            before = rows_only[max(0, position - GRAPH_CONTEXT_NEIGHBOR_WINDOW) : position]
            after = rows_only[position + 1 : position + 1 + GRAPH_CONTEXT_NEIGHBOR_WINDOW]
            context_rows = [*before, *after]
            context_refs = [evidence_ref(item) for item in context_rows if evidence_ref(item)]
            primary_text = first_non_empty(row.get("text"))
            route_lines = [evidence_display_line(item) for item in [*before, row, *after]]
            graph_text_unit_id = stable_id("graph_text_unit", f"{workspace_id}|{ref}|{GRAPH_CONTEXT_NEIGHBOR_WINDOW}")
            units.append(
                {
                    "schema_version": GRAPH_TEXT_UNIT_SCHEMA_VERSION,
                    "graph_text_unit_id": graph_text_unit_id,
                    "workspace_id": workspace_id,
                    "primary_evidence_ref": ref,
                    "primary_text": primary_text,
                    "route_text": "\n".join(line for line in route_lines if line),
                    "extraction_text": "\n".join(line for line in route_lines if line),
                    "context_evidence_refs": context_refs,
                    "previous_evidence_refs": [evidence_ref(item) for item in before if evidence_ref(item)],
                    "next_evidence_refs": [evidence_ref(item) for item in after if evidence_ref(item)],
                    "source_refs": unique_strings(row.get("source_id"), row.get("step1_source_ref")),
                    "raw_backpointer_refs": unique_strings(row.get("step1_evidence_ref"), row.get("source_specific_ref")),
                    "speaker": speaker_for_evidence(row),
                    "source_perspective": first_non_empty(row.get("participant"), row.get("target_participant"), default="unknown"),
                    "subject_role": first_non_empty(row.get("subject_role"), default="unknown"),
                    "context_window": {
                        "neighbor_before": len(before),
                        "neighbor_after": len(after),
                        "neighbor_window": GRAPH_CONTEXT_NEIGHBOR_WINDOW,
                    },
                    "context_requirement_hint": "neighbor_turn_needed" if context_refs else "local_turn_enough",
                    "context_usage": "routing_and_extraction_disambiguation",
                    "warnings": ["context_used_for_routing_not_primary_evidence"] if context_refs else [],
                    "graph_is_not_proof": True,
                }
            )
    return units


def index_graph_text_units(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("primary_evidence_ref")): row for row in rows if row.get("primary_evidence_ref")}


def evidence_text_for_refs(evidence_index: dict[str, dict[str, Any]], refs: list[str]) -> str:
    texts: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        row = evidence_index.get(ref)
        if not row:
            continue
        text = first_non_empty(row.get("text"), row.get("source_text"), row.get("display_ref"))
        if text and text not in seen:
            seen.add(text)
            texts.append(text)
    return "\n".join(texts)


def decision_refs(decision_index: dict[str, list[dict[str, Any]]], *keys: Any) -> list[str]:
    refs: list[str] = []
    for key in unique_strings(*keys):
        for decision in decision_index.get(key, []):
            decision_id = decision.get("decision_id")
            if decision_id:
                refs.append(str(decision_id))
    return unique_strings(refs)


def decision_warnings(decision_index: dict[str, list[dict[str, Any]]], *keys: Any) -> list[str]:
    warnings: list[str] = []
    for key in unique_strings(*keys):
        for decision in decision_index.get(key, []):
            warnings.extend(unique_strings(decision.get("warnings")))
            route = decision.get("route")
            if route:
                warnings.append(f"preprocessing_route:{route}")
    return unique_strings(warnings)


def make_packet(
    *,
    workspace_id: str,
    modeled_user_id: str,
    input_kind: str,
    input_ref: str,
    original_text: str,
    processed_text: str,
    evidence_refs: list[str],
    source_refs: list[str],
    raw_backpointer_refs: list[Any],
    source_perspective: str,
    subject_role: str,
    attribution_status: str,
    temporal_scope: dict[str, Any],
    confidence: str,
    inference_level: str,
    privacy_class: str,
    route_refs: list[str],
    proposal_refs: list[str],
    review_refs: list[str],
    warnings: list[str],
    primary_evidence_refs: list[str] | None = None,
    context_evidence_refs: list[str] | None = None,
    graph_text_unit_id: str = "",
    graph_route_text: str = "",
    graph_extraction_text: str = "",
    context_requirement_hint: str = "unknown",
    context_usage: str = "none",
) -> dict[str, Any]:
    final_warnings = unique_strings(warnings)
    if not evidence_refs:
        final_warnings = unique_strings(final_warnings, ["missing_evidence_refs"])
    if not original_text and not processed_text:
        final_warnings = unique_strings(final_warnings, ["missing_text"])
    packet_id = stable_id("graph_packet", f"{workspace_id}|{input_kind}|{input_ref}|{original_text}|{processed_text}")
    return {
        "schema_version": PACKET_SCHEMA_VERSION,
        "packet_id": packet_id,
        "workspace_id": workspace_id,
        "modeled_user_id": modeled_user_id,
        "input_kind": input_kind,
        "input_ref": input_ref,
        "original_text": original_text,
        "processed_text": processed_text,
        "primary_evidence_refs": primary_evidence_refs if primary_evidence_refs is not None else evidence_refs,
        "context_evidence_refs": context_evidence_refs or [],
        "graph_text_unit_id": graph_text_unit_id,
        "graph_route_text": graph_route_text,
        "graph_extraction_text": graph_extraction_text,
        "context_requirement_hint": context_requirement_hint,
        "context_usage": context_usage,
        "evidence_refs": evidence_refs,
        "source_refs": source_refs,
        "raw_backpointer_refs": raw_backpointer_refs,
        "source_perspective": source_perspective,
        "subject_role": subject_role,
        "attribution_status": attribution_status,
        "temporal_scope": temporal_scope,
        "confidence": confidence,
        "inference_level": inference_level,
        "privacy_class": privacy_class,
        "route_refs": route_refs,
        "proposal_refs": proposal_refs,
        "review_refs": review_refs,
        "warnings": final_warnings,
        "graph_is_not_proof": True,
    }


def evidence_packet(
    row: dict[str, Any],
    workspace_id: str,
    modeled_user_id: str,
    decision_index: dict[str, list[dict[str, Any]]],
    graph_text_unit_index: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    primary_ref = evidence_ref(row)
    if not primary_ref:
        return None
    graph_text_unit = (graph_text_unit_index or {}).get(primary_ref, {})
    context_refs = unique_strings(graph_text_unit.get("context_evidence_refs"))
    route_refs = decision_refs(decision_index, primary_ref)
    warnings = decision_warnings(decision_index, primary_ref)
    subject_role = first_non_empty(row.get("subject_role"), default="unknown")
    participant = first_non_empty(row.get("participant"), row.get("target_participant"), default="unknown")
    return make_packet(
        workspace_id=workspace_id,
        modeled_user_id=modeled_user_id,
        input_kind="evidence_item",
        input_ref=primary_ref,
        original_text=first_non_empty(row.get("text")),
        processed_text="",
        evidence_refs=[primary_ref],
        source_refs=unique_strings(row.get("source_id"), row.get("step1_source_ref")),
        raw_backpointer_refs=unique_strings(row.get("step1_evidence_ref"), row.get("source_specific_ref")),
        source_perspective=participant,
        subject_role=subject_role,
        attribution_status=first_non_empty(row.get("attribution_status"), default="source_text_only"),
        temporal_scope={"timestamp": row.get("timestamp")} if row.get("timestamp") else {},
        confidence="source",
        inference_level="source_text",
        privacy_class=first_non_empty(row.get("privacy_class"), default="unknown"),
        route_refs=route_refs,
        proposal_refs=[],
        review_refs=[],
        warnings=unique_strings(warnings, row.get("warnings"), row.get("subject_contamination_risk"), graph_text_unit.get("warnings")),
        primary_evidence_refs=[primary_ref],
        context_evidence_refs=context_refs,
        graph_text_unit_id=first_non_empty(graph_text_unit.get("graph_text_unit_id")),
        graph_route_text=first_non_empty(graph_text_unit.get("route_text")),
        graph_extraction_text=first_non_empty(graph_text_unit.get("extraction_text")),
        context_requirement_hint=first_non_empty(graph_text_unit.get("context_requirement_hint"), default="local_turn_enough"),
        context_usage=first_non_empty(graph_text_unit.get("context_usage"), default="routing_and_extraction_disambiguation" if context_refs else "none"),
    )


def reviewed_unit_packet(
    row: dict[str, Any],
    workspace_id: str,
    modeled_user_id: str,
    evidence_index: dict[str, dict[str, Any]],
    decision_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    unit_id = first_non_empty(row.get("unit_id"))
    if not unit_id:
        return None
    evidence_refs = unique_strings(row.get("evidence_refs"))
    original_text = first_non_empty(evidence_text_for_refs(evidence_index, evidence_refs), row.get("evidence_summary"))
    review_metadata = row.get("review_metadata") or {}
    step1_origin = row.get("step1_origin") or {}
    route_refs = decision_refs(
        decision_index,
        unit_id,
        step1_origin.get("input_refs"),
        evidence_refs,
    )
    warnings = unique_strings(
        row.get("warnings"),
        row.get("notes"),
        step1_origin.get("warnings"),
        decision_warnings(decision_index, unit_id, step1_origin.get("input_refs"), evidence_refs),
    )
    return make_packet(
        workspace_id=workspace_id,
        modeled_user_id=modeled_user_id,
        input_kind="reviewed_portrait_unit",
        input_ref=unit_id,
        original_text=original_text,
        processed_text=first_non_empty(row.get("content")),
        evidence_refs=evidence_refs,
        source_refs=unique_strings(row.get("source_refs")),
        raw_backpointer_refs=unique_strings(row.get("backpointer_refs")),
        source_perspective=first_non_empty(row.get("user_id"), default=modeled_user_id),
        subject_role="target" if str(row.get("user_id") or "") == modeled_user_id else "unknown",
        attribution_status=first_non_empty(review_metadata.get("review_status"), default="reviewed_unit"),
        temporal_scope=row.get("temporal_scope") or {},
        confidence=first_non_empty(row.get("confidence"), default="unknown"),
        inference_level=first_non_empty(row.get("inference_level"), default="unknown"),
        privacy_class=first_non_empty(row.get("privacy_class"), default="unknown"),
        route_refs=route_refs,
        proposal_refs=unique_strings((row.get("proposal_origin") or {}).get("proposal_id")),
        review_refs=unique_strings(review_metadata.get("review_action"), review_metadata.get("review_status")),
        warnings=warnings,
    )


def normalized_candidate_packet(
    row: dict[str, Any],
    workspace_id: str,
    modeled_user_id: str,
    evidence_index: dict[str, dict[str, Any]],
    review_index: dict[str, dict[str, Any]],
    decision_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    candidate_id = first_non_empty(row.get("candidate_id"))
    if not candidate_id:
        return None
    evidence_refs = unique_strings(row.get("evidence_refs"))
    review = review_index.get(candidate_id, {})
    original_text = first_non_empty(evidence_text_for_refs(evidence_index, evidence_refs), row.get("evidence_excerpt"))
    return make_packet(
        workspace_id=workspace_id,
        modeled_user_id=modeled_user_id,
        input_kind="normalized_candidate",
        input_ref=candidate_id,
        original_text=original_text,
        processed_text=first_non_empty(row.get("candidate_text")),
        evidence_refs=evidence_refs,
        source_refs=unique_strings(row.get("source_refs")),
        raw_backpointer_refs=unique_strings(row.get("backpointer_refs")),
        source_perspective=first_non_empty(row.get("subject_id"), row.get("user_id"), default=modeled_user_id),
        subject_role="target",
        attribution_status=first_non_empty(review.get("review_status"), row.get("review_status"), default="candidate"),
        temporal_scope=row.get("temporal_scope") or {},
        confidence=first_non_empty(row.get("confidence"), default="unknown"),
        inference_level=first_non_empty(row.get("inference_level"), default="unknown"),
        privacy_class=first_non_empty(row.get("privacy_class"), default="unknown"),
        route_refs=decision_refs(decision_index, candidate_id, evidence_refs),
        proposal_refs=unique_strings((row.get("proposal_origin") or {}).get("proposal_id")),
        review_refs=unique_strings(review.get("decision_id"), review.get("review_action"), review.get("review_status")),
        warnings=unique_strings(row.get("warnings"), row.get("step1_warnings"), decision_warnings(decision_index, candidate_id, evidence_refs)),
    )


def proposal_packet(
    row: dict[str, Any],
    workspace_id: str,
    modeled_user_id: str,
    evidence_index: dict[str, dict[str, Any]],
    decision_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    proposal_id = first_non_empty(row.get("proposal_id"))
    if not proposal_id:
        return None
    evidence_refs = unique_strings(row.get("evidence_refs"))
    original_text = first_non_empty(
        row.get("source_text"),
        row.get("original_text"),
        row.get("source_text_quote"),
        evidence_text_for_refs(evidence_index, evidence_refs),
    )
    processed_text = first_non_empty(
        row.get("fact_candidate_text"),
        row.get("hypothesis_text"),
        row.get("candidate_text"),
        row.get("memory_candidate_text"),
    )
    return make_packet(
        workspace_id=workspace_id,
        modeled_user_id=modeled_user_id,
        input_kind="proposal_outcome",
        input_ref=proposal_id,
        original_text=original_text,
        processed_text=processed_text,
        evidence_refs=evidence_refs,
        source_refs=unique_strings(row.get("source_refs"), row.get("text_unit_id")),
        raw_backpointer_refs=unique_strings(row.get("raw_backpointer_refs"), row.get("backpointer_refs")),
        source_perspective=first_non_empty(row.get("source_perspective"), row.get("speaker"), default="unknown"),
        subject_role=first_non_empty(row.get("subject_role"), default="unknown"),
        attribution_status=first_non_empty(row.get("attribution_status"), default="proposal"),
        temporal_scope=row.get("temporal_scope") or {},
        confidence=first_non_empty(row.get("proposal_confidence"), row.get("hypothesis_confidence"), default="unknown"),
        inference_level=first_non_empty(row.get("inference_level"), default="unknown"),
        privacy_class=first_non_empty(row.get("privacy_class"), default="unknown"),
        route_refs=unique_strings(row.get("route_decision_id"), row.get("route_run_id"), decision_refs(decision_index, evidence_refs)),
        proposal_refs=[proposal_id],
        review_refs=[],
        warnings=unique_strings(row.get("warnings"), row.get("validation_warnings"), decision_warnings(decision_index, evidence_refs)),
    )


def source_asset_hashes(workspace: Path, proposal_paths: list[Path]) -> dict[str, str | None]:
    paths = [
        workspace / "evidence" / "evidence.jsonl",
        workspace / "portrait" / "reviewed_units.jsonl",
        workspace / "portrait" / "normalized_candidates.jsonl",
        workspace / "portrait" / "review_decisions.jsonl",
        workspace / "memory" / "preprocessing_decisions.jsonl",
        workspace / "manifest.yaml",
    ]
    hashes: dict[str, str | None] = {}
    for path in paths:
        hashes[str(path.relative_to(workspace))] = file_hash(path)
    for path in proposal_paths:
        hashes[str(path.relative_to(workspace))] = file_hash(path)
    return hashes


def validate_packets(packets: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    packet_ids: set[str] = set()
    for packet in packets:
        packet_id = str(packet.get("packet_id") or "")
        if packet_id in packet_ids:
            warnings.append(f"duplicate_packet_id:{packet_id}")
        packet_ids.add(packet_id)
        if packet.get("graph_is_not_proof") is not True:
            warnings.append(f"graph_is_not_proof_missing:{packet_id}")
        if not packet.get("evidence_refs"):
            warnings.append(f"missing_evidence_refs:{packet_id}")
        if not packet.get("original_text") and not packet.get("processed_text"):
            warnings.append(f"missing_text:{packet_id}")
    return warnings


def build_report(workspace: Path, output_dir: Path, counts: dict[str, Any], warnings: list[str]) -> str:
    kind_lines = [f"- {kind}: {count}" for kind, count in counts["input_kind_counts"].items()]
    warning_lines = [f"- {warning}" for warning in warnings] if warnings else ["- none"]
    return "\n".join(
        [
            "# v0.3 Graph Construction Packet Report",
            "",
            f"- workspace: `{workspace.name}`",
            f"- output_dir: `{output_dir}`",
            "- stage: `graph_construction_packets`",
            "- graph_is_not_proof: `true`",
            "- graph_algorithms_run: `false`",
            "- adapter_layer_used: `false`",
            "",
            "## Counts",
            "",
            f"- packets: {counts['packet_count']}",
            f"- graph_text_units: {counts['graph_text_unit_count']}",
            f"- context_enveloped_evidence_packets: {counts['context_enveloped_evidence_packets']}",
            f"- missing_evidence_packets: {counts['missing_evidence_packets']}",
            f"- missing_text_packets: {counts['missing_text_packets']}",
            "",
            "## Input Kinds",
            "",
            *(kind_lines or ["- none"]),
            "",
            "## Validation Warnings",
            "",
            *warning_lines,
            "",
            "## Notes",
            "",
            "- Packets directly consume project upstream assets.",
            "- GraphRAG is used as stage-design reference only in this slice.",
            "- `graph_text_units.jsonl` is a GraphRAG-style context envelope over fine S0B/evidence units.",
            "- Neighbor context supports routing/extraction disambiguation but does not replace primary evidence.",
            "- Later extraction, merge, summary, and algorithms consume these packets.",
            "",
        ]
    )


def build_graph_construction_packets(workspace: Path, output_dir: Path | None = None) -> dict[str, Any]:
    workspace = workspace.resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")
    output_dir = (output_dir or workspace / "graph_v03_construction").resolve()
    workspace_id = workspace.name

    evidence_rows = read_jsonl(workspace / "evidence" / "evidence.jsonl")
    reviewed_units = read_jsonl(workspace / "portrait" / "reviewed_units.jsonl")
    normalized_candidates = read_jsonl(workspace / "portrait" / "normalized_candidates.jsonl")
    review_decisions = read_jsonl(workspace / "portrait" / "review_decisions.jsonl")
    preprocessing_decisions = read_jsonl(workspace / "memory" / "preprocessing_decisions.jsonl")
    allowed_proposal_ids = referenced_proposal_ids(normalized_candidates, reviewed_units, review_decisions)
    proposal_rows, proposal_paths = load_proposal_rows(workspace, allowed_proposal_ids)

    modeled_user_id = load_modeled_user(workspace, reviewed_units)
    evidence_index = index_evidence(evidence_rows)
    decision_index = index_preprocessing_decisions(preprocessing_decisions)
    review_index = index_review_decisions(review_decisions)
    graph_text_units = build_graph_text_units(evidence_rows, workspace_id)
    graph_text_unit_index = index_graph_text_units(graph_text_units)

    packets: list[dict[str, Any]] = []
    packets.extend(
        packet
        for row in evidence_rows
        if (packet := evidence_packet(row, workspace_id, modeled_user_id, decision_index, graph_text_unit_index)) is not None
    )
    packets.extend(
        packet
        for row in reviewed_units
        if (packet := reviewed_unit_packet(row, workspace_id, modeled_user_id, evidence_index, decision_index)) is not None
    )
    packets.extend(
        packet
        for row in normalized_candidates
        if (packet := normalized_candidate_packet(row, workspace_id, modeled_user_id, evidence_index, review_index, decision_index)) is not None
    )
    packets.extend(
        packet
        for row in proposal_rows
        if (packet := proposal_packet(row, workspace_id, modeled_user_id, evidence_index, decision_index)) is not None
    )

    counts = {
        "packet_count": len(packets),
        "input_kind_counts": dict(sorted(Counter(packet["input_kind"] for packet in packets).items())),
        "missing_evidence_packets": sum(1 for packet in packets if not packet.get("evidence_refs")),
        "missing_text_packets": sum(1 for packet in packets if not packet.get("original_text") and not packet.get("processed_text")),
        "graph_text_unit_count": len(graph_text_units),
        "context_enveloped_evidence_packets": sum(1 for packet in packets if packet.get("input_kind") == "evidence_item" and packet.get("context_evidence_refs")),
        "referenced_proposal_ids": sorted(allowed_proposal_ids),
        "source_asset_hashes": source_asset_hashes(workspace, proposal_paths),
    }
    validation_warnings = validate_packets(packets)

    output_dir.mkdir(parents=True, exist_ok=True)
    packets_path = output_dir / "graph_construction_packets.jsonl"
    graph_text_units_path = output_dir / "graph_text_units.jsonl"
    manifest_path = output_dir / "graph_construction_manifest.json"
    report_path = output_dir / "graph_construction_report.md"

    write_jsonl(packets_path, packets)
    write_jsonl(graph_text_units_path, graph_text_units)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "workspace_id": workspace_id,
        "modeled_user_id": modeled_user_id,
        "source_workspace": str(workspace),
        "output_dir": str(output_dir),
        "source_assets": {
            "evidence": "evidence/evidence.jsonl",
            "reviewed_units": "portrait/reviewed_units.jsonl",
            "normalized_candidates": "portrait/normalized_candidates.jsonl",
            "review_decisions": "portrait/review_decisions.jsonl",
            "preprocessing_decisions": "memory/preprocessing_decisions.jsonl",
            "proposal_outcomes": "proposals/**/proposal_outcomes.ai.jsonl",
        },
        "outputs": {
            "graph_construction_packets": str(packets_path),
            "graph_text_units": str(graph_text_units_path),
            "report": str(report_path),
        },
        "counts": counts,
        "validation_warnings": validation_warnings,
        "policies": {
            "adapter_layer_used": False,
            "external_runtime_dependency": False,
            "graphrag_usage": "stage_design_reference",
            "graph_is_not_proof": True,
            "graph_algorithms_run": False,
            "merge_performed": False,
            "graph_text_units_are_context_envelopes": True,
            "context_neighbors_are_not_primary_evidence": True,
        },
    }
    write_json(manifest_path, manifest)
    write_text(report_path, build_report(workspace, output_dir, counts, validation_warnings))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v0.3 graph construction packets from project upstream assets.")
    parser.add_argument("--workspace", required=True, type=Path, help="Workspace containing S1/S2 upstream assets.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <workspace>/graph_v03_construction.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_graph_construction_packets(args.workspace, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
