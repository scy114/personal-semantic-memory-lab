"""Resolve v0.4 incremental maintenance impact from existing artifacts.

This module is read-only. It maps operation scope ids to currently discoverable
S0B/S1/S2/index/query/graph artifacts and reports lineage gaps. It does not
mark stale rows or rebuild assets.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.maintenance.operation_log import read_jsonl as read_operation_log
from tools.maintenance.operation_log import validate_operation


SCHEMA_VERSION = "maintenance.incremental_impact_report.v0.4"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalize_ref(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("evidence_ref", "canonical_evidence_ref", "source_id", "source_ref", "id"):
            if value.get(key):
                return str(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value).strip()


def list_refs(row: dict[str, Any], *field_names: str) -> list[str]:
    refs: list[str] = []
    for field_name in field_names:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, list):
            refs.extend(normalize_ref(item) for item in value)
        else:
            refs.append(normalize_ref(value))
    seen: set[str] = set()
    cleaned: list[str] = []
    for ref in refs:
        if ref and ref not in seen:
            seen.add(ref)
            cleaned.append(ref)
    return cleaned


def row_id(row: dict[str, Any], *candidates: str) -> str:
    for candidate in candidates:
        value = row.get(candidate)
        if value:
            return str(value)
    return ""


def nested_evidence_refs(row: dict[str, Any]) -> list[str]:
    refs = list_refs(row, "evidence_refs")
    for quote in row.get("evidence_quotes") or []:
        if isinstance(quote, dict) and quote.get("evidence_ref"):
            refs.append(str(quote["evidence_ref"]))
    temporal_scope = row.get("temporal_scope") or {}
    if isinstance(temporal_scope, dict):
        for observed in temporal_scope.get("observed_at_refs") or []:
            if isinstance(observed, dict) and observed.get("display_ref"):
                refs.append(str(observed["display_ref"]))
    return sorted(set(refs))


class RefIndex:
    def __init__(self) -> None:
        self.by_source_id: dict[str, set[str]] = defaultdict(set)
        self.by_source_ref: dict[str, set[str]] = defaultdict(set)
        self.by_evidence_ref: dict[str, set[str]] = defaultdict(set)
        self.rows_by_id: dict[str, dict[str, Any]] = {}

    def add(self, object_id: str, row: dict[str, Any]) -> None:
        if object_id:
            self.rows_by_id[object_id] = row
            for ref in list_refs(row, "source_id", "raw_source_id"):
                self.by_source_id[ref].add(object_id)
            for ref in list_refs(row, "source_refs", "source_ref", "step1_source_ref"):
                self.by_source_ref[ref].add(object_id)
            for ref in list_refs(row, "evidence_ref", "step1_evidence_ref"):
                self.by_evidence_ref[ref].add(object_id)
            for ref in nested_evidence_refs(row):
                self.by_evidence_ref[ref].add(object_id)


def add_index_refs(index: RefIndex, object_id: str, row: dict[str, Any]) -> None:
    index.add(object_id, row)
    for ref in list_refs(row, "object_id"):
        index.by_source_ref[ref].add(object_id)


def build_workspace_indexes(workspace: Path) -> dict[str, RefIndex]:
    indexes = {
        "evidence": RefIndex(),
        "s1_memory": RefIndex(),
        "s2_units": RefIndex(),
        "s1_index_entries": RefIndex(),
        "s2_index_entries": RefIndex(),
        "graph_candidates": RefIndex(),
        "query_packets": RefIndex(),
    }

    for row in read_jsonl(workspace / "evidence" / "evidence.jsonl"):
        object_id = row_id(row, "evidence_ref", "id")
        indexes["evidence"].add(object_id, row)

    for row in read_jsonl(workspace / "memory" / "memory_units.jsonl"):
        object_id = row_id(row, "memory_id", "id")
        indexes["s1_memory"].add(object_id, row)

    for row in read_jsonl(workspace / "portrait" / "reviewed_units.jsonl"):
        object_id = row_id(row, "unit_id", "id")
        indexes["s2_units"].add(object_id, row)

    for filename in ("step1_bm25_entries.jsonl", "step1_embedding_entries.jsonl"):
        for row in read_jsonl(workspace / "indexes" / filename):
            object_id = row_id(row, "index_entry_id", "vector_entry_id", "object_id", "id")
            add_index_refs(indexes["s1_index_entries"], object_id, row)

    for row in read_jsonl(workspace / "indexes" / "step2_user_model_embedding_entries.jsonl"):
        object_id = row_id(row, "vector_entry_id", "object_id", "id")
        add_index_refs(indexes["s2_index_entries"], object_id, row)

    for candidate_path in workspace.glob("graph_v03_consolidation*/graph_*_table.jsonl"):
        for row in read_jsonl(candidate_path):
            object_id = row_id(row, "node_id", "edge_id", "claim_id", "evidence_link_id", "id")
            indexes["graph_candidates"].add(object_id, row)

    for query_path in (workspace / "queries").glob("**/*.json"):
        try:
            payload = json.loads(query_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            object_id = payload.get("packet_id") or payload.get("query_id") or str(query_path.relative_to(workspace))
            payload["_path"] = str(query_path)
            indexes["query_packets"].add(str(object_id), payload)

    return indexes


def resolve_ids(index: RefIndex, scope: dict[str, list[str]]) -> set[str]:
    resolved: set[str] = set()
    for source_id in scope.get("source_ids", []):
        resolved.update(index.by_source_id.get(source_id, set()))
        resolved.update(index.by_source_ref.get(source_id, set()))
    for source_version in scope.get("source_versions", []):
        resolved.update(index.by_source_ref.get(source_version, set()))
    for evidence_ref in scope.get("evidence_refs", []):
        resolved.update(index.by_evidence_ref.get(evidence_ref, set()))
    for s0b_unit_id in scope.get("s0b_unit_ids", []):
        resolved.update(index.by_source_ref.get(s0b_unit_id, set()))
    return resolved


def extend_scope_from_evidence(evidence_ids: set[str], evidence_index: RefIndex) -> list[str]:
    refs = set(evidence_ids)
    for evidence_id in evidence_ids:
        row = evidence_index.rows_by_id.get(evidence_id) or {}
        refs.update(list_refs(row, "evidence_ref"))
    return sorted(refs)


def resolve_operation_impact(workspace: Path, operation: dict[str, Any]) -> dict[str, Any]:
    validate_operation(operation)
    indexes = build_workspace_indexes(workspace)
    scope = dict(operation.get("scope") or {})
    scope.setdefault("evidence_refs", [])

    evidence_ids = resolve_ids(indexes["evidence"], scope)
    if evidence_ids:
        scope["evidence_refs"] = sorted(set(scope.get("evidence_refs", [])) | set(extend_scope_from_evidence(evidence_ids, indexes["evidence"])))

    impacted = {
        "evidence_refs": sorted(evidence_ids),
        "s1_unit_ids": sorted(resolve_ids(indexes["s1_memory"], scope)),
        "s2_unit_ids": sorted(resolve_ids(indexes["s2_units"], scope)),
        "s1_index_entry_ids": sorted(resolve_ids(indexes["s1_index_entries"], scope)),
        "s2_index_entry_ids": sorted(resolve_ids(indexes["s2_index_entries"], scope)),
        "graph_candidate_ids": sorted(resolve_ids(indexes["graph_candidates"], scope)),
        "query_packet_ids": sorted(resolve_ids(indexes["query_packets"], scope)),
    }

    recommended_actions = []
    if impacted["s1_unit_ids"]:
        recommended_actions.append("mark_s1_latest_view_stale")
    if impacted["s2_unit_ids"]:
        recommended_actions.append("mark_s2_latest_view_stale")
    if impacted["s1_index_entry_ids"] or impacted["s2_index_entry_ids"]:
        recommended_actions.append("invalidate_indexes")
    if impacted["graph_candidate_ids"]:
        recommended_actions.append("refresh_graph_candidates_after_s1_s2_delta")
    if impacted["query_packet_ids"]:
        recommended_actions.append("invalidate_query_packets")
    if not recommended_actions:
        recommended_actions.append("gap_review_no_impacted_artifacts_found")

    gaps = []
    if not impacted["evidence_refs"] and not impacted["s1_unit_ids"] and not impacted["s2_unit_ids"]:
        gaps.append("operation_scope_did_not_resolve_to_s0b_s1_or_s2_assets")
    if impacted["s2_unit_ids"] and not impacted["s1_unit_ids"]:
        gaps.append("s2_units_resolved_without_matching_s1_units")
    if impacted["s1_unit_ids"] and not impacted["s1_index_entry_ids"]:
        gaps.append("s1_units_resolved_without_matching_s1_index_entries")

    return {
        "schema_version": SCHEMA_VERSION,
        "workspace": str(workspace),
        "operation_id": operation["operation_id"],
        "operation_type": operation["operation_type"],
        "operation_status": operation["status"],
        "resolved_at": now_iso(),
        "input_scope": operation.get("scope", {}),
        "expanded_scope": scope,
        "impacted": impacted,
        "recommended_actions": recommended_actions,
        "gaps": gaps,
        "read_only": True,
        "write_permission": False,
    }


def resolve_latest_operation(log_path: Path, operation_id: str = "") -> dict[str, Any]:
    rows = read_operation_log(log_path)
    if not rows:
        raise ValueError(f"No operations found in {log_path}")
    if operation_id:
        matches = [row for row in rows if row["operation_id"] == operation_id]
        if not matches:
            raise ValueError(f"Operation not found: {operation_id}")
        return matches[-1]
    return rows[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--operation-log", required=True, type=Path)
    parser.add_argument("--operation-id", default="")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    operation = resolve_latest_operation(args.operation_log, args.operation_id)
    report = resolve_operation_impact(args.workspace, operation)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "incremental_impact_report.json", report)
    rows = [
        {
            "operation_id": report["operation_id"],
            "artifact_layer": layer,
            "artifact_id": artifact_id,
            "read_only": True,
            "write_permission": False,
        }
        for layer, artifact_ids in report["impacted"].items()
        for artifact_id in artifact_ids
    ]
    write_jsonl(args.output_dir / "incremental_impact_rows.jsonl", rows)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
