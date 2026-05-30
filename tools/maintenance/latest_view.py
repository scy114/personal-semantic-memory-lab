"""Build read-only latest-view projections for v0.4 maintenance.

Latest views are derived artifacts. They do not mutate canonical S0B/S1/S2,
index, graph, or query files. The first v0.4 slice uses them to keep stale or
deprecated rows out of active consumers while preserving audit rows separately.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "maintenance.latest_view.v0.4"
MANIFEST_SCHEMA_VERSION = "maintenance.latest_view_manifest.v0.4"
LATEST_VIEW_DIR = Path("maintenance") / "latest_views"

LAYER_IMPACT_KEYS = {
    "evidence": "evidence_refs",
    "s0b": "s0b_unit_ids",
    "s1": "s1_unit_ids",
    "s2": "s2_unit_ids",
    "s1_index": "s1_index_entry_ids",
    "s2_index": "s2_index_entry_ids",
    "graph": "graph_candidate_ids",
    "graph_nodes": "graph_candidate_ids",
    "graph_edges": "graph_candidate_ids",
    "graph_claims": "graph_candidate_ids",
    "query": "query_packet_ids",
}

LAYER_ID_FIELDS = {
    "evidence": ["evidence_ref", "id"],
    "s0b": ["s0b_unit_id", "text_unit_id", "raw_span_id", "unit_id", "evidence_ref", "id"],
    "s1": ["memory_id", "id"],
    "s2": ["unit_id", "id"],
    "s1_index": ["index_entry_id", "vector_entry_id", "object_id", "id"],
    "s2_index": ["vector_entry_id", "index_entry_id", "object_id", "id"],
    "graph": ["node_id", "edge_id", "claim_id", "evidence_link_id", "id"],
    "graph_nodes": ["node_id", "id"],
    "graph_edges": ["edge_id", "id"],
    "graph_claims": ["claim_id", "id"],
    "query": ["packet_id", "query_id", "id"],
}

STATUS_FIELDS = [
    "maintenance_status",
    "latest_view_status",
    "lifecycle_status",
    "s0b_status",
    "source_status",
    "status",
]

INACTIVE_STATUS_VALUES = {
    "deprecated",
    "deleted",
    "deleted_candidate",
    "hard_deleted",
    "inactive",
    "invalidated",
    "stale",
    "stale_candidate",
    "superseded",
}

SUPPORTED_LATEST_VIEW_MODES = {"auto", "require", "off"}


def default_latest_view_dir(workspace: Path) -> Path:
    return workspace / LATEST_VIEW_DIR


def default_latest_view_path(workspace: Path, layer: str) -> Path:
    return default_latest_view_dir(workspace) / f"{layer}_latest_view.jsonl"


def resolve_latest_view_input(
    *,
    workspace: Path,
    layer: str,
    canonical_path: Path,
    explicit_path: Path | None = None,
    mode: str = "auto",
) -> tuple[Path, dict[str, Any]]:
    if mode not in SUPPORTED_LATEST_VIEW_MODES:
        raise ValueError(f"Unsupported latest_view_mode: {mode}")
    if explicit_path is not None:
        if not explicit_path.exists():
            raise FileNotFoundError(f"Explicit latest-view path not found: {explicit_path}")
        return explicit_path, {
            "mode": mode,
            "layer": layer,
            "source": "explicit_latest_view",
            "path": str(explicit_path),
            "canonical_fallback_path": str(canonical_path),
            "warnings": [],
        }
    if mode == "off":
        return canonical_path, {
            "mode": mode,
            "layer": layer,
            "source": "canonical",
            "path": str(canonical_path),
            "warnings": ["latest_view_disabled"],
        }
    latest_path = default_latest_view_path(workspace, layer)
    if latest_path.exists():
        return latest_path, {
            "mode": mode,
            "layer": layer,
            "source": "default_latest_view",
            "path": str(latest_path),
            "canonical_fallback_path": str(canonical_path),
            "warnings": [],
        }
    if mode == "require":
        raise FileNotFoundError(f"Required latest-view path not found: {latest_path}")
    return canonical_path, {
        "mode": mode,
        "layer": layer,
        "source": "canonical_fallback",
        "path": str(canonical_path),
        "expected_latest_view_path": str(latest_path),
        "warnings": ["latest_view_not_found_using_canonical_fallback"],
    }


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


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
        if not isinstance(row, dict):
            raise ValueError(f"Latest-view input row must be an object at {path}:{line_number}")
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


def row_id(row: dict[str, Any], id_fields: list[str]) -> str:
    for field_name in id_fields:
        value = row.get(field_name)
        if value:
            return str(value)
    return ""


def normalize_ids(values: Iterable[Any] | None) -> set[str]:
    return {str(value).strip() for value in values or [] if str(value).strip()}


def impacted_ids_from_report(report: dict[str, Any], layer: str) -> set[str]:
    impact_key = LAYER_IMPACT_KEYS.get(layer)
    if not impact_key:
        raise ValueError(f"Unsupported latest-view layer: {layer}")
    impacted = report.get("impacted") or {}
    if not isinstance(impacted, dict):
        return set()
    return normalize_ids(impacted.get(impact_key))


def inactive_statuses(row: dict[str, Any]) -> list[str]:
    statuses: list[str] = []
    for field_name in STATUS_FIELDS:
        value = row.get(field_name)
        if isinstance(value, str) and value.strip().lower() in INACTIVE_STATUS_VALUES:
            statuses.append(f"{field_name}:{value.strip().lower()}")
    return statuses


def annotate_row(
    row: dict[str, Any],
    *,
    layer: str,
    object_id: str,
    latest_view_status: str,
    reason: str,
    operation_id: str,
    source_path: str,
    generated_at: str,
) -> dict[str, Any]:
    annotated = deepcopy(row)
    annotated["_maintenance_latest_view"] = {
        "schema_version": SCHEMA_VERSION,
        "layer": layer,
        "object_id": object_id,
        "latest_view_status": latest_view_status,
        "reason": reason,
        "operation_id": operation_id,
        "source_path": source_path,
        "generated_at": generated_at,
        "read_only": True,
        "write_permission": False,
    }
    return annotated


def build_latest_view(
    rows: list[dict[str, Any]],
    *,
    layer: str,
    id_fields: list[str] | None = None,
    impacted_ids: Iterable[Any] | None = None,
    operation_id: str = "",
    source_path: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    if layer not in LAYER_IMPACT_KEYS:
        raise ValueError(f"Unsupported latest-view layer: {layer}")

    resolved_id_fields = id_fields or LAYER_ID_FIELDS[layer]
    impacted = normalize_ids(impacted_ids)
    resolved_generated_at = generated_at or now_iso()

    active_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for row in rows:
        object_id = row_id(row, resolved_id_fields)
        status_hits = inactive_statuses(row)
        if not object_id:
            reason = "missing_object_id"
            latest_view_status = "excluded"
        elif status_hits:
            reason = "source_status_inactive:" + ",".join(status_hits)
            latest_view_status = "excluded"
        elif object_id in impacted:
            reason = "impacted_by_operation"
            latest_view_status = "stale_candidate"
        else:
            reason = "active"
            latest_view_status = "active"

        annotated = annotate_row(
            row,
            layer=layer,
            object_id=object_id,
            latest_view_status=latest_view_status,
            reason=reason,
            operation_id=operation_id,
            source_path=source_path,
            generated_at=resolved_generated_at,
        )
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if latest_view_status == "active":
            active_rows.append(annotated)
        else:
            excluded_rows.append(annotated)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "layer": layer,
        "source_path": source_path,
        "operation_id": operation_id,
        "id_fields": resolved_id_fields,
        "input_count": len(rows),
        "active_count": len(active_rows),
        "excluded_count": len(excluded_rows),
        "impacted_input_count": len(impacted),
        "reason_counts": dict(sorted(reason_counts.items())),
        "read_only": True,
        "write_permission": False,
        "generated_at": resolved_generated_at,
    }
    return {
        "active_rows": active_rows,
        "excluded_rows": excluded_rows,
        "manifest": manifest,
    }


def write_latest_view_bundle(output_dir: Path, layer: str, bundle: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    active_path = output_dir / f"{layer}_latest_view.jsonl"
    excluded_path = output_dir / f"{layer}_latest_view_excluded.jsonl"
    manifest_path = output_dir / f"{layer}_latest_view_manifest.json"
    write_jsonl(active_path, bundle["active_rows"])
    write_jsonl(excluded_path, bundle["excluded_rows"])
    write_json(manifest_path, bundle["manifest"])
    return {
        "active_path": str(active_path),
        "excluded_path": str(excluded_path),
        "manifest_path": str(manifest_path),
    }


def build_latest_view_from_files(
    *,
    input_jsonl: Path,
    layer: str,
    output_dir: Path,
    impact_report: Path | None = None,
    id_fields: list[str] | None = None,
) -> dict[str, Any]:
    report = read_json(impact_report) if impact_report else {}
    impacted = impacted_ids_from_report(report, layer) if report else set()
    operation_id = str(report.get("operation_id") or "") if report else ""
    bundle = build_latest_view(
        read_jsonl(input_jsonl),
        layer=layer,
        id_fields=id_fields,
        impacted_ids=impacted,
        operation_id=operation_id,
        source_path=str(input_jsonl),
    )
    paths = write_latest_view_bundle(output_dir, layer, bundle)
    bundle["paths"] = paths
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--layer", required=True, choices=sorted(LAYER_IMPACT_KEYS))
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--impact-report", type=Path)
    parser.add_argument("--id-field", action="append", default=[])
    args = parser.parse_args()

    bundle = build_latest_view_from_files(
        input_jsonl=args.input_jsonl,
        layer=args.layer,
        output_dir=args.output_dir,
        impact_report=args.impact_report,
        id_fields=args.id_field or None,
    )
    print(json.dumps(bundle["manifest"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
