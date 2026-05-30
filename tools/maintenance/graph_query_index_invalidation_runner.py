"""Emit graph query/index invalidation artifacts for v0.4 current graph changes.

The runner compares a current candidate graph view with an optional previous
current view and writes scoped invalidation rows. It does not rebuild BM25,
embedding, NetworkX, algorithm, query, or visualization assets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.graph.graph_construction_packet_builder import file_hash, read_jsonl, stable_id, write_json, write_jsonl, write_text
from tools.maintenance.graph_dependency_map_builder import (
    DEPENDENCY_MAP_FILENAME,
    MANIFEST_FILENAME as DEPENDENCY_MANIFEST_FILENAME,
    run_graph_dependency_map_build,
)


SCHEMA_VERSION = "maintenance.graph_query_index_invalidation.v0.4"
MANIFEST_FILENAME = "graph_query_index_invalidation_manifest.json"
REPORT_FILENAME = "graph_query_index_invalidation_report.md"
CHANGED_UNITS_FILENAME = "changed_graph_units.jsonl"
STALE_ASSETS_FILENAME = "stale_graph_query_assets.jsonl"

TABLES = {
    "node": {"id_field": "node_id", "active": "graph_nodes_latest_view.jsonl"},
    "edge": {"id_field": "edge_id", "active": "graph_edges_latest_view.jsonl"},
    "claim": {"id_field": "claim_id", "active": "graph_claims_latest_view.jsonl"},
    "evidence_link": {"id_field": "link_id", "active": "evidence_links_latest_view.jsonl"},
}

VOLATILE_KEYS = {
    "generated_at",
    "published_at",
    "previous_current_manifest",
    "previous_current_archive_dir",
}

ASSET_POLICIES = {
    "node": [
        "graph_bm25_units",
        "graph_embedding_units",
        "networkx_projection",
        "graph_algorithm_report",
        "graph_relation_neighborhood_cache",
        "graph_visual_review_bundle",
    ],
    "edge": [
        "graph_bm25_units",
        "graph_embedding_units",
        "networkx_projection",
        "graph_algorithm_report",
        "graph_relation_neighborhood_cache",
        "graph_evidence_path_cache",
        "graph_visual_review_bundle",
    ],
    "claim": [
        "graph_bm25_units",
        "graph_embedding_units",
        "graph_claim_query_units",
        "graph_contradiction_update_neighborhood_cache",
        "graph_visual_review_bundle",
    ],
    "evidence_link": [
        "graph_evidence_path_cache",
        "graph_bm25_units",
        "graph_embedding_units",
    ],
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def object_id(row: dict[str, Any], table: str) -> str:
    id_field = TABLES[table]["id_field"]
    value = str(row.get(id_field) or "").strip()
    if value:
        return value
    latest = row.get("_maintenance_graph_latest_view")
    if isinstance(latest, dict):
        return str(latest.get("object_id") or "").strip()
    return ""


def canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: canonicalize(v) for k, v in sorted(value.items()) if k not in VOLATILE_KEYS}
    if isinstance(value, list):
        return [canonicalize(item) for item in value]
    return value


def row_fingerprint(row: dict[str, Any]) -> str:
    payload = json.dumps(canonicalize(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return stable_id("graph_row_hash", payload)


def load_table(path: Path, table: str) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in read_jsonl(path / TABLES[table]["active"]):
        oid = object_id(row, table)
        if oid:
            rows[oid] = row
    return rows


def dependency_rows_by_object(dependency_map_path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in read_jsonl(dependency_map_path):
        oid = str(row.get("graph_object_id") or "").strip()
        if oid and oid not in rows:
            rows[oid] = row
    return rows


def dependency_map_path(current_graph_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    candidate = current_graph_dir / "dependency_map" / DEPENDENCY_MAP_FILENAME
    if not candidate.exists():
        run_graph_dependency_map_build(graph_dir=current_graph_dir, output_dir=current_graph_dir / "dependency_map")
    return candidate.resolve()


def change_type_for(
    *,
    current_row: dict[str, Any] | None,
    previous_row: dict[str, Any] | None,
) -> str:
    if previous_row is None and current_row is not None:
        return "new"
    if previous_row is not None and current_row is None:
        return "removed_from_current_view"
    if previous_row is not None and current_row is not None and row_fingerprint(previous_row) != row_fingerprint(current_row):
        return "changed"
    return "unchanged"


def build_changed_units(
    *,
    current_graph_dir: Path,
    previous_graph_dir: Path | None,
    dependency_rows: dict[str, dict[str, Any]],
    generated_at: str,
) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    no_previous = previous_graph_dir is None or not previous_graph_dir.exists()
    for table in TABLES:
        current_rows = load_table(current_graph_dir, table)
        previous_rows = {} if no_previous else load_table(previous_graph_dir, table)
        object_ids = sorted(set(current_rows) | set(previous_rows))
        for oid in object_ids:
            current_row = current_rows.get(oid)
            previous_row = previous_rows.get(oid)
            change_type = "initial_publish" if no_previous and current_row is not None else change_type_for(current_row=current_row, previous_row=previous_row)
            if change_type == "unchanged":
                continue
            dep = dependency_rows.get(oid, {})
            changed.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "change_id": stable_id("graph_change", f"{table}|{oid}|{change_type}|{row_fingerprint(current_row or previous_row or {})}"),
                    "graph_object_kind": table,
                    "graph_object_id": oid,
                    "change_type": change_type,
                    "current_present": current_row is not None,
                    "previous_present": previous_row is not None,
                    "current_row_hash": row_fingerprint(current_row) if current_row is not None else "",
                    "previous_row_hash": row_fingerprint(previous_row) if previous_row is not None else "",
                    "evidence_refs": dep.get("evidence_refs", []),
                    "source_packet_ids": dep.get("source_packet_ids", []),
                    "candidate_ids": dep.get("candidate_ids", []),
                    "s1_unit_ids": dep.get("s1_unit_ids", []),
                    "s2_unit_ids": dep.get("s2_unit_ids", []),
                    "depends_on_node_ids": dep.get("depends_on_node_ids", []),
                    "depends_on_edge_ids": dep.get("depends_on_edge_ids", []),
                    "depends_on_claim_ids": dep.get("depends_on_claim_ids", []),
                    "graph_is_not_proof": True,
                    "write_permission": False,
                    "generated_at": generated_at,
                }
            )
    return changed


def build_stale_assets(changed_units: list[dict[str, Any]], generated_at: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for unit in changed_units:
        kind = unit["graph_object_kind"]
        oid = unit["graph_object_id"]
        for asset_kind in ASSET_POLICIES.get(kind, []):
            key = (asset_kind, kind, oid)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "invalidation_id": stable_id("graph_invalidation", "|".join(key)),
                    "asset_kind": asset_kind,
                    "scope": "graph_object",
                    "graph_object_kind": kind,
                    "graph_object_id": oid,
                    "change_type": unit["change_type"],
                    "refresh_required": True,
                    "refresh_executed": False,
                    "reason": f"{kind}_{unit['change_type']}",
                    "evidence_refs": unit.get("evidence_refs", []),
                    "source_packet_ids": unit.get("source_packet_ids", []),
                    "graph_is_not_proof": True,
                    "write_permission": False,
                    "generated_at": generated_at,
                }
            )
    return rows


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.4 Graph Query/Index Invalidation",
        "",
        f"- generated_at: {manifest['generated_at']}",
        f"- current_graph_dir: `{manifest['inputs']['current_graph_dir']}`",
        f"- previous_graph_dir: `{manifest['inputs'].get('previous_graph_dir') or ''}`",
        f"- dependency_map: `{manifest['inputs']['dependency_map']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "## Counts",
        "",
        f"- changed_graph_units: {manifest['counts']['changed_graph_units']}",
        f"- stale_asset_rows: {manifest['counts']['stale_asset_rows']}",
        "",
        "### Change Types",
        "",
    ]
    for key, value in manifest["counts"]["change_type_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "### Asset Kinds", ""])
    for key, value in manifest["counts"]["asset_kind_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This runner only marks graph query/index/visual assets stale.",
            "- It does not rebuild indexes, NetworkX projections, algorithm reports, or visualization bundles.",
            "- It does not write graph truth or durable memory.",
            "- graph_is_not_proof=true.",
            "",
        ]
    )
    return "\n".join(lines)


def run_graph_query_index_invalidation(
    *,
    current_graph_dir: Path,
    previous_graph_dir: Path | None = None,
    dependency_map: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    current_graph_dir = current_graph_dir.resolve()
    previous_graph_dir = previous_graph_dir.resolve() if previous_graph_dir else None
    output_dir = (output_dir or current_graph_dir / "query_index_invalidation").resolve()
    generated_at = now_iso()

    dep_path = dependency_map_path(current_graph_dir, dependency_map)
    dependency_rows = dependency_rows_by_object(dep_path)
    changed_units = build_changed_units(
        current_graph_dir=current_graph_dir,
        previous_graph_dir=previous_graph_dir,
        dependency_rows=dependency_rows,
        generated_at=generated_at,
    )
    stale_assets = build_stale_assets(changed_units, generated_at)

    output_dir.mkdir(parents=True, exist_ok=True)
    changed_path = output_dir / CHANGED_UNITS_FILENAME
    stale_path = output_dir / STALE_ASSETS_FILENAME
    write_jsonl(changed_path, changed_units)
    write_jsonl(stale_path, stale_assets)

    change_type_counts = Counter(row["change_type"] for row in changed_units)
    asset_kind_counts = Counter(row["asset_kind"] for row in stale_assets)
    input_hashes = {
        "dependency_map": file_hash(dep_path),
        "dependency_manifest": file_hash(dep_path.parent / DEPENDENCY_MANIFEST_FILENAME),
    }
    for table, spec in TABLES.items():
        input_hashes[f"current_{table}"] = file_hash(current_graph_dir / spec["active"])
        if previous_graph_dir is not None:
            input_hashes[f"previous_{table}"] = file_hash(previous_graph_dir / spec["active"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "inputs": {
            "current_graph_dir": str(current_graph_dir),
            "previous_graph_dir": str(previous_graph_dir) if previous_graph_dir else "",
            "dependency_map": str(dep_path),
        },
        "output_dir": str(output_dir),
        "outputs": {
            "changed_graph_units": str(changed_path),
            "stale_graph_query_assets": str(stale_path),
            "report": str(output_dir / REPORT_FILENAME),
        },
        "counts": {
            "changed_graph_units": len(changed_units),
            "stale_asset_rows": len(stale_assets),
            "change_type_counts": dict(sorted(change_type_counts.items())),
            "asset_kind_counts": dict(sorted(asset_kind_counts.items())),
        },
        "input_hashes": input_hashes,
        "output_hashes": {
            "changed_graph_units": file_hash(changed_path),
            "stale_graph_query_assets": file_hash(stale_path),
        },
        "graph_is_not_proof": True,
        "graph_truth_written": False,
        "durable_writes_executed": False,
        "query_indexes_refreshed": False,
        "visual_review_refreshed": False,
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    write_text(output_dir / REPORT_FILENAME, render_report(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit v0.4 graph query/index invalidation artifacts.")
    parser.add_argument("--current-graph-dir", required=True)
    parser.add_argument("--previous-graph-dir")
    parser.add_argument("--dependency-map")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    manifest = run_graph_query_index_invalidation(
        current_graph_dir=Path(args.current_graph_dir),
        previous_graph_dir=Path(args.previous_graph_dir) if args.previous_graph_dir else None,
        dependency_map=Path(args.dependency_map) if args.dependency_map else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
