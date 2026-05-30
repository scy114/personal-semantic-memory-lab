"""Refresh v0.31 visual review slices for affected v0.4 graph neighborhoods.

This runner is deliberately scoped: it consumes the published
``graph_current`` latest-view bundle plus graph change rows and writes a
bounded visual audit bundle. It does not rebuild graph query indexes, mutate
candidate graph truth, or write durable memory.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.graph.graph_construction_packet_builder import file_hash, read_jsonl, write_json, write_jsonl, write_text
from tools.graph.graph_visual_review_bundle_builder import (
    bounded_induced_subgraph,
    ego_slice,
    write_html_index,
    write_slice,
)
from tools.graph.networkx_graph_utility_runner import load_projection, node_label


SCHEMA_VERSION = "maintenance.graph_visual_review_incremental_refresh.v0.4"
MANIFEST_FILENAME = "graph_visual_review_incremental_manifest.json"
REPORT_FILENAME = "graph_visual_review_incremental_report.md"
SLICES_FILENAME = "graph_visual_review_slices.jsonl"
HTML_FILENAME = "graph_visual_review_index.html"
DEFAULT_OUTPUT_DIR_NAME = "graph_v031_visual_review_incremental"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def object_id(row: dict[str, Any], *names: str) -> str:
    for name in names:
        value = str(row.get(name) or "").strip()
        if value:
            return value
    return ""


def index_rows(rows: list[dict[str, Any]], *id_fields: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        oid = object_id(row, *id_fields)
        if oid:
            indexed[oid] = row
    return indexed


def changed_units_path(graph_current_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return graph_current_dir / "query_index_invalidation" / "changed_graph_units.jsonl"


def dependency_map_path(graph_current_dir: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    return graph_current_dir / "dependency_map" / "graph_dependency_map.jsonl"


def load_current_tables(graph_current_dir: Path) -> dict[str, Any]:
    nodes = read_jsonl(graph_current_dir / "graph_nodes_latest_view.jsonl")
    edges = read_jsonl(graph_current_dir / "graph_edges_latest_view.jsonl")
    claims = read_jsonl(graph_current_dir / "graph_claims_latest_view.jsonl")
    evidence_links = read_jsonl(graph_current_dir / "evidence_links_latest_view.jsonl")
    return {
        "nodes": nodes,
        "edges": edges,
        "claims": claims,
        "evidence_links": evidence_links,
        "node_by_id": index_rows(nodes, "node_id"),
        "edge_by_id": index_rows(edges, "edge_id"),
        "claim_by_id": index_rows(claims, "claim_id"),
        "evidence_link_by_id": index_rows(evidence_links, "link_id"),
    }


def endpoint_nodes(edge: dict[str, Any]) -> list[str]:
    return [
        node_id
        for node_id in [
            str(edge.get("source_node_id") or "").strip(),
            str(edge.get("target_node_id") or "").strip(),
        ]
        if node_id
    ]


def owner_neighborhood_nodes(
    *,
    owner_kind: str,
    owner_id: str,
    tables: dict[str, Any],
) -> list[str]:
    if owner_kind == "node":
        return [owner_id]
    if owner_kind == "edge":
        edge = tables["edge_by_id"].get(owner_id) or {}
        return endpoint_nodes(edge)
    if owner_kind == "claim":
        claim = tables["claim_by_id"].get(owner_id) or {}
        return [node_id for node_id in [str(claim.get("subject_node_id") or "").strip()] if node_id]
    return []


def slice_for_changed_unit(
    *,
    graph,
    change: dict[str, Any],
    tables: dict[str, Any],
    max_nodes: int,
    max_edges: int,
) -> tuple[str, str, str, Any, list[str]] | None:
    kind = str(change.get("graph_object_kind") or "")
    oid = str(change.get("graph_object_id") or "")
    change_type = str(change.get("change_type") or "")
    warnings = [
        "incremental_visual_review_slice",
        f"change_type:{change_type}",
    ]
    if change.get("current_present") is False:
        return None

    if kind == "node":
        if oid not in graph:
            return None
        subgraph, extra_warnings = ego_slice(graph, oid, max_nodes=max_nodes, max_edges=max_edges)
        warnings.extend(extra_warnings)
        title = f"changed node: {node_label(graph, oid)}"
        reason = "Inspect the affected node neighborhood after the current graph view changed."
        return "incremental_changed_node", title, reason, subgraph, warnings

    if kind == "edge":
        edge = tables["edge_by_id"].get(oid)
        if not edge:
            return None
        nodes = endpoint_nodes(edge)
        if not nodes:
            return None
        expanded = set(nodes)
        for node_id in nodes:
            if node_id in graph:
                expanded.update(str(item) for item in graph.predecessors(node_id))
                expanded.update(str(item) for item in graph.successors(node_id))
        subgraph, extra_warnings = bounded_induced_subgraph(
            graph,
            sorted(expanded),
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        warnings.extend(extra_warnings)
        source = edge.get("source_label") or edge.get("source_node_id") or ""
        target = edge.get("target_label") or edge.get("target_node_id") or ""
        relation = edge.get("relation_type") or "related_to"
        title = f"changed edge: {source} --{relation}--> {target}"
        reason = "Inspect both endpoints and their bounded relation neighborhood after an edge changed."
        return "incremental_changed_edge", title, reason, subgraph, warnings

    if kind == "claim":
        claim = tables["claim_by_id"].get(oid)
        if not claim:
            return None
        subject = str(claim.get("subject_node_id") or "").strip()
        if not subject or subject not in graph:
            return None
        subgraph, extra_warnings = ego_slice(graph, subject, max_nodes=max_nodes, max_edges=max_edges)
        warnings.extend(extra_warnings)
        title = f"changed claim neighborhood: {node_label(graph, subject)}"
        reason = "Inspect the subject neighborhood for a changed graph claim candidate."
        return "incremental_changed_claim", title, reason, subgraph, warnings

    if kind == "evidence_link":
        link = tables["evidence_link_by_id"].get(oid)
        if not link:
            return None
        nodes = owner_neighborhood_nodes(
            owner_kind=str(link.get("owner_kind") or ""),
            owner_id=str(link.get("owner_id") or ""),
            tables=tables,
        )
        if not nodes:
            return None
        subgraph, extra_warnings = bounded_induced_subgraph(
            graph,
            nodes,
            max_nodes=max_nodes,
            max_edges=max_edges,
        )
        warnings.extend(extra_warnings)
        title = f"changed evidence link: {oid}"
        reason = "Inspect the graph object neighborhood touched by a changed evidence link."
        return "incremental_changed_evidence_link", title, reason, subgraph, warnings

    return None


def write_no_change_html(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest.get("counts") or {}
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v0.4 增量图可视化审查</title>
  <style>
    body {{ margin: 0; font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif; background: #f5f7fb; color: #172033; }}
    main {{ max-width: 860px; margin: 0 auto; padding: 40px 22px; }}
    .panel {{ background: white; border: 1px solid #d8deea; border-radius: 10px; padding: 24px; box-shadow: 0 14px 35px rgba(15, 23, 42, 0.08); }}
    h1 {{ font-size: 22px; margin: 0 0 12px; }}
    p, li {{ font-size: 14px; line-height: 1.6; }}
    code {{ background: #eef3fb; padding: 2px 5px; border-radius: 4px; }}
  </style>
</head>
<body>
<main>
  <div class="panel">
    <h1>v0.4 增量图可视化审查</h1>
    <p>本次发布没有检测到变化的 graph units，因此没有生成受影响邻域切片。</p>
    <ul>
      <li>changed_graph_units: <code>{counts.get("changed_graph_units", 0)}</code></li>
      <li>visual slices: <code>{counts.get("slices", 0)}</code></li>
      <li>graph_is_not_proof=true</li>
      <li>visualization_is_audit_support_only=true</li>
    </ul>
  </div>
</main>
</body>
</html>
"""
    write_text(path, document)


def render_report(manifest: dict[str, Any], skipped: list[dict[str, Any]]) -> str:
    lines = [
        "# v0.4 Incremental Graph Visual Review Refresh",
        "",
        f"- generated_at: {manifest['generated_at']}",
        f"- workspace: `{manifest['workspace']}`",
        f"- graph_current_dir: `{manifest['inputs']['graph_current_dir']}`",
        f"- changed_graph_units: `{manifest['inputs']['changed_graph_units']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "## Counts",
        "",
        f"- changed_graph_units: {manifest['counts']['changed_graph_units']}",
        f"- slices: {manifest['counts']['slices']}",
        f"- skipped_changes: {manifest['counts']['skipped_changes']}",
        f"- no_changes_detected: {str(manifest['counts']['no_changes_detected']).lower()}",
        "",
        "## Boundary",
        "",
        "- Visualization is audit support only.",
        "- It refreshes affected bounded neighborhoods, not the whole graph by default.",
        "- It does not write graph truth, durable memory, query indexes, or support-checker authority.",
        "- graph_is_not_proof=true.",
        "",
    ]
    if skipped:
        lines.extend(["## Skipped Changes", ""])
        for row in skipped[:20]:
            lines.append(
                f"- {row.get('graph_object_kind')} `{row.get('graph_object_id')}`: {row.get('reason')}"
            )
        lines.append("")
    return "\n".join(lines)


def run_graph_visual_review_incremental_refresh(
    *,
    workspace: Path,
    graph_current_dir: Path | None = None,
    changed_graph_units: Path | None = None,
    dependency_map: Path | None = None,
    output_dir: Path | None = None,
    projection: str = "review_aware_graph",
    max_nodes_per_slice: int = 25,
    max_edges_per_slice: int = 60,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    graph_current_dir = (graph_current_dir or workspace / "graph_current").resolve()
    changed_path = changed_units_path(graph_current_dir, changed_graph_units)
    dependency_path = dependency_map_path(graph_current_dir, dependency_map)
    output_dir = (output_dir or workspace / DEFAULT_OUTPUT_DIR_NAME).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = load_current_tables(graph_current_dir)
    changes = read_jsonl(changed_path) if changed_path.exists() else []
    graph, projection_stats = load_projection(projection=projection, nodes=tables["nodes"], edges=tables["edges"])

    slice_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_slice_keys: set[tuple[str, str]] = set()
    for change in changes:
        result = slice_for_changed_unit(
            graph=graph,
            change=change,
            tables=tables,
            max_nodes=max_nodes_per_slice,
            max_edges=max_edges_per_slice,
        )
        if result is None:
            skipped.append(
                {
                    "graph_object_kind": change.get("graph_object_kind"),
                    "graph_object_id": change.get("graph_object_id"),
                    "change_type": change.get("change_type"),
                    "reason": "no_current_visible_neighborhood_or_removed_from_current_view",
                }
            )
            continue
        slice_kind, title, reason, subgraph, warnings = result
        key = (slice_kind, title)
        if key in seen_slice_keys:
            continue
        seen_slice_keys.add(key)
        slice_rows.append(
            write_slice(
                output_dir=output_dir,
                slice_kind=slice_kind,
                title=title,
                reason=reason,
                graph=subgraph,
                warnings=warnings,
            )
        )

    write_jsonl(output_dir / SLICES_FILENAME, slice_rows)
    generated_at = now_iso()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "workspace": str(workspace),
        "inputs": {
            "graph_current_dir": str(graph_current_dir),
            "changed_graph_units": str(changed_path),
            "dependency_map": str(dependency_path),
        },
        "output_dir": str(output_dir),
        "projection": projection,
        "source_hashes": {
            "graph_nodes_latest_view.jsonl": file_hash(graph_current_dir / "graph_nodes_latest_view.jsonl"),
            "graph_edges_latest_view.jsonl": file_hash(graph_current_dir / "graph_edges_latest_view.jsonl"),
            "graph_claims_latest_view.jsonl": file_hash(graph_current_dir / "graph_claims_latest_view.jsonl"),
            "evidence_links_latest_view.jsonl": file_hash(graph_current_dir / "evidence_links_latest_view.jsonl"),
            "changed_graph_units.jsonl": file_hash(changed_path),
            "graph_dependency_map.jsonl": file_hash(dependency_path),
        },
        "outputs": {
            "manifest": str(output_dir / MANIFEST_FILENAME),
            "report": str(output_dir / REPORT_FILENAME),
            "slices": str(output_dir / SLICES_FILENAME),
            "html_index": str(output_dir / HTML_FILENAME),
            "slices_dir": str(output_dir / "slices"),
        },
        "projection_stats": projection_stats,
        "counts": {
            "input_nodes": len(tables["nodes"]),
            "input_edges": len(tables["edges"]),
            "input_claims": len(tables["claims"]),
            "input_evidence_links": len(tables["evidence_links"]),
            "changed_graph_units": len(changes),
            "slices": len(slice_rows),
            "skipped_changes": len(skipped),
            "slice_nodes_total": sum(int(row.get("node_count") or 0) for row in slice_rows),
            "slice_edges_total": sum(int(row.get("edge_count") or 0) for row in slice_rows),
            "change_type_counts": dict(sorted(Counter(str(row.get("change_type") or "") for row in changes).items())),
            "change_kind_counts": dict(sorted(Counter(str(row.get("graph_object_kind") or "") for row in changes).items())),
            "no_changes_detected": len(changes) == 0,
        },
        "policies": {
            "visual_review_artifact_written": True,
            "visual_review_refreshed": bool(slice_rows),
            "affected_neighborhoods_only": True,
            "query_indexes_refreshed": False,
            "networkx_algorithm_report_refreshed": False,
            "graph_truth_written": False,
            "durable_memory_written": False,
            "support_checker_authority": False,
        },
        "boundary": {
            "graph_is_not_proof": True,
            "support_status": "not_checked",
            "write_permission": False,
            "visualization_is_audit_support_only": True,
        },
        "skipped_changes": skipped,
        "graph_is_not_proof": True,
        "graph_truth_written": False,
        "durable_writes_executed": False,
    }
    write_json(output_dir / MANIFEST_FILENAME, manifest)
    write_text(output_dir / REPORT_FILENAME, render_report(manifest, skipped))
    if slice_rows:
        write_html_index(output_dir / HTML_FILENAME, manifest, slice_rows)
    else:
        write_no_change_html(output_dir / HTML_FILENAME, manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--graph-current-dir", type=Path)
    parser.add_argument("--changed-graph-units", type=Path)
    parser.add_argument("--dependency-map", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--projection", default="review_aware_graph")
    parser.add_argument("--max-nodes-per-slice", type=int, default=25)
    parser.add_argument("--max-edges-per-slice", type=int, default=60)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_graph_visual_review_incremental_refresh(
        workspace=args.workspace,
        graph_current_dir=args.graph_current_dir,
        changed_graph_units=args.changed_graph_units,
        dependency_map=args.dependency_map,
        output_dir=args.output_dir,
        projection=args.projection,
        max_nodes_per_slice=args.max_nodes_per_slice,
        max_edges_per_slice=args.max_edges_per_slice,
    )
    print(
        json.dumps(
            {
                "manifest": manifest["outputs"]["manifest"],
                "html_index": manifest["outputs"]["html_index"],
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
