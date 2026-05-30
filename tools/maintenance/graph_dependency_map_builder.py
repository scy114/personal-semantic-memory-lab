"""Build lineage maps for the current v0.4 candidate graph view.

This runner records which evidence/source/packet/candidate refs support each
current graph node, edge, claim, and evidence link. It is an audit and
incremental-maintenance artifact only; it does not write graph truth, durable
memory, or query indexes.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.graph.graph_construction_packet_builder import file_hash, read_jsonl, stable_id, write_json, write_jsonl, write_text


SCHEMA_VERSION = "maintenance.graph_dependency_map.v0.4"
MANIFEST_FILENAME = "graph_dependency_map_manifest.json"
REPORT_FILENAME = "graph_dependency_map_report.md"
DEPENDENCY_MAP_FILENAME = "graph_dependency_map.jsonl"
REVERSE_INDEX_FILENAME = "graph_dependency_reverse_index.json"

TABLES = {
    "node": {
        "id_field": "node_id",
        "active": "graph_nodes_latest_view.jsonl",
        "excluded": "graph_nodes_latest_view_excluded.jsonl",
    },
    "edge": {
        "id_field": "edge_id",
        "active": "graph_edges_latest_view.jsonl",
        "excluded": "graph_edges_latest_view_excluded.jsonl",
    },
    "claim": {
        "id_field": "claim_id",
        "active": "graph_claims_latest_view.jsonl",
        "excluded": "graph_claims_latest_view_excluded.jsonl",
    },
    "evidence_link": {
        "id_field": "link_id",
        "active": "evidence_links_latest_view.jsonl",
        "excluded": "evidence_links_latest_view_excluded.jsonl",
    },
}

COMMON_REF_FIELDS = {
    "candidate_ids": ("candidate_id", "candidate_ids", "source_candidate_ids", "target_candidate_ids", "subject_candidate_id"),
    "source_packet_ids": ("source_packet_id", "source_packet_ids", "packet_id", "packet_ids"),
    "evidence_refs": ("evidence_ref", "evidence_refs"),
    "raw_backpointer_refs": ("raw_backpointer_ref", "raw_backpointer_refs", "backpointer_refs"),
    "source_refs": ("source_ref", "source_refs", "raw_source_id", "raw_source_ids"),
    "s1_unit_ids": ("s1_unit_id", "s1_unit_ids", "memory_id", "memory_ids", "source_s1_unit_id", "source_s1_unit_ids"),
    "s2_unit_ids": ("s2_unit_id", "s2_unit_ids", "reviewed_unit_id", "reviewed_unit_ids", "source_s2_unit_id", "source_s2_unit_ids"),
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def string_list(value: Any) -> list[str]:
    out: list[str] = []
    for item in listify(value):
        if item is None:
            continue
        if isinstance(item, dict):
            text = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
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


def read_table(graph_dir: Path, table: str, *, include_excluded: bool) -> list[tuple[dict[str, Any], str]]:
    spec = TABLES[table]
    rows = [(row, "active") for row in read_jsonl(graph_dir / spec["active"])]
    if include_excluded:
        rows.extend((row, "excluded") for row in read_jsonl(graph_dir / spec["excluded"]))
    return rows


def object_id(row: dict[str, Any], table: str) -> str:
    id_field = TABLES[table]["id_field"]
    value = str(row.get(id_field) or "").strip()
    if value:
        return value
    latest = row.get("_maintenance_graph_latest_view")
    if isinstance(latest, dict):
        return str(latest.get("object_id") or "").strip()
    return ""


def latest_view_reason(row: dict[str, Any]) -> str:
    latest = row.get("_maintenance_graph_latest_view")
    if isinstance(latest, dict):
        return str(latest.get("reason") or "").strip()
    return ""


def collect_refs(row: dict[str, Any]) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for target, fields in COMMON_REF_FIELDS.items():
        values: list[Any] = []
        for field in fields:
            if field in row:
                values.append(row.get(field))
        refs[target] = unique_strings(*values)
    return refs


def dependency_node_refs(row: dict[str, Any], table: str) -> list[str]:
    refs: list[str] = []
    if table == "edge":
        refs = unique_strings(row.get("source_node_id"), row.get("target_node_id"))
    elif table == "claim":
        refs = unique_strings(row.get("subject_node_id"), row.get("object_node_id"))
    elif table == "evidence_link":
        if str(row.get("owner_kind") or "").strip() == "node":
            refs = unique_strings(row.get("owner_id"))
    return refs


def dependency_edge_refs(row: dict[str, Any], table: str) -> list[str]:
    if table == "evidence_link" and str(row.get("owner_kind") or "").strip() == "edge":
        return unique_strings(row.get("owner_id"))
    return []


def dependency_claim_refs(row: dict[str, Any], table: str) -> list[str]:
    if table == "evidence_link" and str(row.get("owner_kind") or "").strip() == "claim":
        return unique_strings(row.get("owner_id"))
    return []


def build_dependency_row(
    *,
    table: str,
    row: dict[str, Any],
    latest_view_status: str,
    generated_at: str,
    evidence_link_refs_by_owner: dict[str, dict[str, list[str]]],
) -> dict[str, Any]:
    graph_object_id = object_id(row, table)
    refs = collect_refs(row)
    linked_refs = evidence_link_refs_by_owner.get(graph_object_id, {})
    warnings = unique_strings(row.get("warnings"))
    if not refs["evidence_refs"] and not linked_refs.get("evidence_refs"):
        warnings = unique_strings(warnings, "missing_evidence_refs_for_dependency_map")
    if not refs["source_packet_ids"] and not linked_refs.get("source_packet_ids"):
        warnings = unique_strings(warnings, "missing_source_packet_ids_for_dependency_map")
    return {
        "schema_version": SCHEMA_VERSION,
        "dependency_id": stable_id("graph_dependency", f"{table}|{graph_object_id}|{latest_view_status}"),
        "graph_object_kind": table,
        "graph_object_id": graph_object_id,
        "latest_view_status": latest_view_status,
        "latest_view_reason": latest_view_reason(row),
        "candidate_ids": unique_strings(refs["candidate_ids"], linked_refs.get("candidate_ids")),
        "source_packet_ids": unique_strings(refs["source_packet_ids"], linked_refs.get("source_packet_ids")),
        "evidence_refs": unique_strings(refs["evidence_refs"], linked_refs.get("evidence_refs")),
        "raw_backpointer_refs": unique_strings(refs["raw_backpointer_refs"], linked_refs.get("raw_backpointer_refs")),
        "source_refs": unique_strings(refs["source_refs"], linked_refs.get("source_refs")),
        "s1_unit_ids": refs["s1_unit_ids"],
        "s2_unit_ids": refs["s2_unit_ids"],
        "depends_on_node_ids": dependency_node_refs(row, table),
        "depends_on_edge_ids": dependency_edge_refs(row, table),
        "depends_on_claim_ids": dependency_claim_refs(row, table),
        "support_status": str(row.get("support_status") or "not_checked"),
        "graph_is_not_proof": True,
        "candidate_not_truth": True,
        "write_permission": False,
        "warnings": warnings,
        "generated_at": generated_at,
    }


def evidence_link_refs_by_owner(evidence_link_rows: Iterable[tuple[dict[str, Any], str]]) -> dict[str, dict[str, list[str]]]:
    grouped: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))  # type: ignore[assignment]
    for row, _ in evidence_link_rows:
        owner_id = str(row.get("owner_id") or "").strip()
        if not owner_id:
            continue
        refs = collect_refs(row)
        for key, values in refs.items():
            grouped[owner_id][key] = unique_strings(grouped[owner_id].get(key), values)
    return {owner_id: dict(values) for owner_id, values in grouped.items()}


def build_reverse_index(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reverse: dict[str, dict[str, list[str]]] = {
        "by_evidence_ref": defaultdict(list),
        "by_source_packet_id": defaultdict(list),
        "by_candidate_id": defaultdict(list),
        "by_s1_unit_id": defaultdict(list),
        "by_s2_unit_id": defaultdict(list),
        "by_node_dependency": defaultdict(list),
        "by_edge_dependency": defaultdict(list),
        "by_claim_dependency": defaultdict(list),
    }
    for row in rows:
        graph_object_id = row["graph_object_id"]
        for value in row["evidence_refs"]:
            reverse["by_evidence_ref"][value].append(graph_object_id)
        for value in row["source_packet_ids"]:
            reverse["by_source_packet_id"][value].append(graph_object_id)
        for value in row["candidate_ids"]:
            reverse["by_candidate_id"][value].append(graph_object_id)
        for value in row["s1_unit_ids"]:
            reverse["by_s1_unit_id"][value].append(graph_object_id)
        for value in row["s2_unit_ids"]:
            reverse["by_s2_unit_id"][value].append(graph_object_id)
        for value in row["depends_on_node_ids"]:
            reverse["by_node_dependency"][value].append(graph_object_id)
        for value in row["depends_on_edge_ids"]:
            reverse["by_edge_dependency"][value].append(graph_object_id)
        for value in row["depends_on_claim_ids"]:
            reverse["by_claim_dependency"][value].append(graph_object_id)
    return {
        key: {ref: sorted(set(ids)) for ref, ids in value.items()}
        for key, value in reverse.items()
    }


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.4 Graph Dependency Map",
        "",
        f"- generated_at: {manifest['generated_at']}",
        f"- graph_dir: `{manifest['inputs']['graph_dir']}`",
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
                f"- active: {counts['active']}",
                f"- excluded: {counts['excluded']}",
                f"- dependency_rows: {counts['dependency_rows']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- This map is lineage for incremental maintenance and audit.",
            "- It is not graph truth and not support proof.",
            "- graph_is_not_proof=true.",
            "- Missing S1/S2 ids are warnings about lineage granularity, not automatic rejection.",
            "",
        ]
    )
    return "\n".join(lines)


def run_graph_dependency_map_build(
    *,
    graph_dir: Path,
    output_dir: Path | None = None,
    include_excluded: bool = True,
) -> dict[str, Any]:
    graph_dir = graph_dir.resolve()
    output_dir = (output_dir or graph_dir / "dependency_map").resolve()
    generated_at = now_iso()

    table_rows = {table: read_table(graph_dir, table, include_excluded=include_excluded) for table in TABLES}
    link_refs = evidence_link_refs_by_owner(table_rows["evidence_link"])

    dependency_rows: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {}
    warning_counts: Counter[str] = Counter()
    for table, rows in table_rows.items():
        table_dependency_rows: list[dict[str, Any]] = []
        for row, status in rows:
            dep = build_dependency_row(
                table=table,
                row=row,
                latest_view_status=status,
                generated_at=generated_at,
                evidence_link_refs_by_owner=link_refs,
            )
            table_dependency_rows.append(dep)
            warning_counts.update(dep["warnings"])
        dependency_rows.extend(table_dependency_rows)
        counts[table] = {
            "active": sum(1 for _, status in rows if status == "active"),
            "excluded": sum(1 for _, status in rows if status == "excluded"),
            "dependency_rows": len(table_dependency_rows),
        }

    reverse_index = build_reverse_index(dependency_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    dep_path = output_dir / DEPENDENCY_MAP_FILENAME
    reverse_path = output_dir / REVERSE_INDEX_FILENAME
    write_jsonl(dep_path, dependency_rows)
    write_json(reverse_path, reverse_index)

    input_hashes = {
        spec["active"]: file_hash(graph_dir / spec["active"])
        for spec in TABLES.values()
    }
    input_hashes.update(
        {
            spec["excluded"]: file_hash(graph_dir / spec["excluded"])
            for spec in TABLES.values()
        }
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "inputs": {
            "graph_dir": str(graph_dir),
            "include_excluded": include_excluded,
        },
        "output_dir": str(output_dir),
        "outputs": {
            "dependency_map": str(dep_path),
            "reverse_index": str(reverse_path),
            "report": str(output_dir / REPORT_FILENAME),
        },
        "counts": counts,
        "warning_counts": dict(sorted(warning_counts.items())),
        "input_hashes": input_hashes,
        "output_hashes": {
            "dependency_map": file_hash(dep_path),
            "reverse_index": file_hash(reverse_path),
        },
        "graph_is_not_proof": True,
        "graph_truth_written": False,
        "durable_writes_executed": False,
        "query_indexes_refreshed": False,
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    write_text(output_dir / REPORT_FILENAME, render_report(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build v0.4 graph dependency map artifacts.")
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--active-only", action="store_true")
    args = parser.parse_args()
    manifest = run_graph_dependency_map_build(
        graph_dir=Path(args.graph_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        include_excluded=not args.active_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
