"""Build v0.4 latest-view overlays for candidate graph tables.

This runner combines an optional base consolidated graph directory with an
incremental consolidated graph directory. Incremental rows override base rows
with the same graph object id. The output is a read-only latest-view bundle for
query/visual consumers; it does not mutate canonical graph tables or publish
graph truth.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.graph.graph_construction_packet_builder import file_hash, read_jsonl, write_json, write_jsonl, write_text


SCHEMA_VERSION = "maintenance.graph_candidate_latest_view.v0.4"
REPORT_FILENAME = "graph_candidate_latest_view_report.md"
MANIFEST_FILENAME = "graph_candidate_latest_view_manifest.json"

TABLES = {
    "nodes": {
        "id_field": "node_id",
        "input": "graph_nodes_table.jsonl",
        "latest": "graph_nodes_latest_view.jsonl",
        "excluded": "graph_nodes_latest_view_excluded.jsonl",
    },
    "edges": {
        "id_field": "edge_id",
        "input": "graph_edges_table.jsonl",
        "latest": "graph_edges_latest_view.jsonl",
        "excluded": "graph_edges_latest_view_excluded.jsonl",
    },
    "claims": {
        "id_field": "claim_id",
        "input": "graph_claims_table.jsonl",
        "latest": "graph_claims_latest_view.jsonl",
        "excluded": "graph_claims_latest_view_excluded.jsonl",
    },
    "evidence_links": {
        "id_field": "link_id",
        "input": "evidence_links.jsonl",
        "latest": "evidence_links_latest_view.jsonl",
        "excluded": "evidence_links_latest_view_excluded.jsonl",
    },
}

INACTIVE_VALUES = {
    "deleted",
    "deprecated",
    "hard_deleted",
    "historical_only",
    "inactive",
    "invalidated",
    "stale",
    "stale_candidate",
    "superseded",
}
STATUS_FIELDS = ("maintenance_status", "latest_view_status", "lifecycle_status", "status")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        values = value
    else:
        values = [value]
    out: list[str] = []
    for item in values:
        if item is None:
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
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def row_id(row: dict[str, Any], id_field: str) -> str:
    return str(row.get(id_field) or "").strip()


def source_rows(source_dir: Path | None, table: str) -> list[dict[str, Any]]:
    if source_dir is None:
        return []
    path = source_dir / TABLES[table]["input"]
    return read_jsonl(path) if path.exists() else []


def inactive_status_reason(row: dict[str, Any]) -> str:
    for field in STATUS_FIELDS:
        value = str(row.get(field) or "").strip().lower()
        if value in INACTIVE_VALUES:
            return f"inactive_status:{field}:{value}"
    return ""


def table_exclusion_reason(table: str, row: dict[str, Any], active_owner_ids: set[str] | None = None) -> str:
    status_reason = inactive_status_reason(row)
    if status_reason:
        return status_reason
    if table == "nodes" and str(row.get("entity_quality_hint") or "") == "review_required":
        return "entity_review_required"
    if table == "edges":
        if str(row.get("generic_relation_review_hint") or "") in {"low_graph_value", "review_required"}:
            return "edge_generic_relation_review:" + str(row.get("generic_relation_review_hint") or "")
        if str(row.get("support_status") or "") == "contradicted":
            return "edge_support_contradicted"
    if table == "claims" and str(row.get("subject_endpoint_status") or "") == "unresolved":
        return "claim_subject_unresolved"
    if table == "evidence_links" and active_owner_ids is not None:
        owner_id = str(row.get("owner_id") or "").strip()
        if owner_id and owner_id not in active_owner_ids:
            return "evidence_owner_not_active"
    return ""


def annotate(
    row: dict[str, Any],
    *,
    table: str,
    object_id: str,
    source_layer: str,
    latest_view_status: str,
    reason: str,
    generated_at: str,
) -> dict[str, Any]:
    out = deepcopy(row)
    out["_maintenance_graph_latest_view"] = {
        "schema_version": SCHEMA_VERSION,
        "table": table,
        "object_id": object_id,
        "source_layer": source_layer,
        "latest_view_status": latest_view_status,
        "reason": reason,
        "generated_at": generated_at,
        "read_only": True,
        "write_permission": False,
    }
    out["graph_is_not_proof"] = True
    out["write_permission"] = False
    return out


def merge_table(
    *,
    table: str,
    base_rows: list[dict[str, Any]],
    incremental_rows: list[dict[str, Any]],
    generated_at: str,
    active_owner_ids: set[str] | None = None,
) -> dict[str, Any]:
    id_field = str(TABLES[table]["id_field"])
    active: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    incremental_ids = {row_id(row, id_field) for row in incremental_rows if row_id(row, id_field)}
    seen: set[str] = set()

    for source_layer, rows in (("base", base_rows), ("incremental", incremental_rows)):
        for row in rows:
            object_id = row_id(row, id_field)
            if not object_id:
                reason = "missing_object_id"
                latest_status = "excluded"
            elif source_layer == "base" and object_id in incremental_ids:
                reason = "overlaid_by_incremental_candidate"
                latest_status = "excluded"
            elif object_id in seen:
                reason = "duplicate_after_overlay"
                latest_status = "excluded"
            else:
                reason = table_exclusion_reason(table, row, active_owner_ids=active_owner_ids)
                latest_status = "excluded" if reason else "active"
                if not reason:
                    reason = "active"
                    seen.add(object_id)
            annotated = annotate(
                row,
                table=table,
                object_id=object_id,
                source_layer=source_layer,
                latest_view_status=latest_status,
                reason=reason,
                generated_at=generated_at,
            )
            reason_counts[reason] += 1
            if latest_status == "active":
                active.append(annotated)
            else:
                excluded.append(annotated)

    return {
        "active": active,
        "excluded": excluded,
        "counts": {
            "base_count": len(base_rows),
            "incremental_count": len(incremental_rows),
            "active_count": len(active),
            "excluded_count": len(excluded),
            "reason_counts": dict(sorted(reason_counts.items())),
        },
    }


def write_table_bundle(output_dir: Path, table: str, bundle: dict[str, Any]) -> dict[str, str]:
    spec = TABLES[table]
    active_path = output_dir / str(spec["latest"])
    excluded_path = output_dir / str(spec["excluded"])
    write_jsonl(active_path, bundle["active"])
    write_jsonl(excluded_path, bundle["excluded"])
    return {"latest": str(active_path), "excluded": str(excluded_path)}


def copy_latest_aliases(output_dir: Path, paths: dict[str, dict[str, str]]) -> None:
    latest_dir = output_dir / "latest_views"
    latest_dir.mkdir(parents=True, exist_ok=True)
    for table, table_paths in paths.items():
        for kind, source in table_paths.items():
            source_path = Path(source)
            target_name = source_path.name
            write_text(latest_dir / target_name, source_path.read_text(encoding="utf-8"))


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.4 Graph Candidate Latest View",
        "",
        f"- generated_at: {manifest['generated_at']}",
        f"- base_dir: `{manifest['inputs']['base_graph_dir']}`",
        f"- incremental_dir: `{manifest['inputs']['incremental_graph_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "## Counts",
        "",
    ]
    for table, counts in manifest["counts"].items():
        lines.extend(
            [
                f"### {table}",
                "",
                f"- base: {counts['base_count']}",
                f"- incremental: {counts['incremental_count']}",
                f"- active: {counts['active_count']}",
                f"- excluded: {counts['excluded_count']}",
                f"- reasons: `{json.dumps(counts['reason_counts'], ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- Latest views are read-only candidate projections.",
            "- No canonical graph table is rewritten.",
            "- No graph truth, durable memory, S2 truth, query index, or visualization bundle is written.",
            "- graph_is_not_proof=true.",
            "",
        ]
    )
    return "\n".join(lines)


def run_graph_candidate_latest_view(
    *,
    incremental_graph_dir: Path,
    output_dir: Path,
    base_graph_dir: Path | None = None,
) -> dict[str, Any]:
    base_graph_dir = base_graph_dir.resolve() if base_graph_dir else None
    incremental_graph_dir = incremental_graph_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_at = now_iso()

    node_bundle = merge_table(
        table="nodes",
        base_rows=source_rows(base_graph_dir, "nodes"),
        incremental_rows=source_rows(incremental_graph_dir, "nodes"),
        generated_at=generated_at,
    )
    active_node_ids = {str(row.get("node_id") or "") for row in node_bundle["active"] if row.get("node_id")}
    edge_bundle = merge_table(
        table="edges",
        base_rows=source_rows(base_graph_dir, "edges"),
        incremental_rows=source_rows(incremental_graph_dir, "edges"),
        generated_at=generated_at,
    )
    active_edge_ids = {str(row.get("edge_id") or "") for row in edge_bundle["active"] if row.get("edge_id")}
    claim_bundle = merge_table(
        table="claims",
        base_rows=source_rows(base_graph_dir, "claims"),
        incremental_rows=source_rows(incremental_graph_dir, "claims"),
        generated_at=generated_at,
    )
    active_claim_ids = {str(row.get("claim_id") or "") for row in claim_bundle["active"] if row.get("claim_id")}
    active_owner_ids = active_node_ids | active_edge_ids | active_claim_ids
    evidence_bundle = merge_table(
        table="evidence_links",
        base_rows=source_rows(base_graph_dir, "evidence_links"),
        incremental_rows=source_rows(incremental_graph_dir, "evidence_links"),
        generated_at=generated_at,
        active_owner_ids=active_owner_ids,
    )

    bundles = {
        "nodes": node_bundle,
        "edges": edge_bundle,
        "claims": claim_bundle,
        "evidence_links": evidence_bundle,
    }
    paths = {table: write_table_bundle(output_dir, table, bundle) for table, bundle in bundles.items()}
    copy_latest_aliases(output_dir, paths)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "inputs": {
            "base_graph_dir": str(base_graph_dir) if base_graph_dir else "",
            "incremental_graph_dir": str(incremental_graph_dir),
        },
        "input_hashes": {
            "base_manifest": file_hash(base_graph_dir / "graph_consolidation_manifest.json") if base_graph_dir else None,
            "incremental_manifest": file_hash(incremental_graph_dir / "graph_consolidation_manifest.json"),
        },
        "output_dir": str(output_dir),
        "counts": {table: bundle["counts"] for table, bundle in bundles.items()},
        "paths": {
            **paths,
            "latest_views_dir": str(output_dir / "latest_views"),
            "manifest": str(output_dir / MANIFEST_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
        },
        "policies": {
            "incremental_overlays_base_by_id": True,
            "low_graph_value_generic_edges_excluded": True,
            "review_required_entities_excluded": True,
            "evidence_links_follow_active_owner": True,
            "graph_is_not_proof": True,
            "write_permission": False,
            "graph_truth_written": False,
            "durable_memory_written": False,
            "query_indexes_refreshed": False,
            "visual_review_refreshed": False,
        },
        "boundary": "read_only_candidate_latest_view_no_graph_truth_write",
        "write_permission": False,
        "graph_truth_written": False,
        "durable_writes_executed": False,
        "graph_query_indexes_refreshed": False,
        "visual_review_refreshed": False,
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    write_text(output_dir / REPORT_FILENAME, render_report(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incremental-graph-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--base-graph-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_graph_candidate_latest_view(
        base_graph_dir=args.base_graph_dir,
        incremental_graph_dir=args.incremental_graph_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps({"manifest": manifest["paths"]["manifest"], "counts": manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
