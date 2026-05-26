"""Build v0.31 graph visualization review bundles.

This is an auxiliary audit exporter. It borrows mature GraphRAG/LightRAG
patterns: GraphML snapshots for external graph tools, and bounded subgraphs for
human inspection. It does not create graph truth, durable memory, S3 assets, or
support-checker authority.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from tools.graph.graph_construction_packet_builder import (
    file_hash,
    read_jsonl,
    stable_id,
    write_json,
    write_jsonl,
    write_text,
)
from tools.graph.networkx_graph_utility_runner import (
    load_projection,
    node_label,
    string_list,
    write_graphml,
)


SCHEMA_VERSION = "graph_v031.visual_review_bundle.v0.1"
VISUAL_SLICE_SCHEMA_VERSION = "graph_v031.visual_review_slice.v0.1"
DEFAULT_GRAPH_DIR_NAME = "graph_v03_consolidation_provider_80"
DEFAULT_NETWORKX_DIR_NAME = "graph_v03_networkx_utility_provider_80"
DEFAULT_PROFILE_DIR_NAME = "graph_v03_profile_communities"
DEFAULT_OUTPUT_DIR_NAME = "graph_v031_visual_review"
WEAK_NODE_HINTS = {"context_dependent", "generic_fragment", "action_phrase", "review_required"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def uniq(values: list[Any], *, limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for item in string_list(value):
            if item not in seen:
                seen.add(item)
                out.append(item)
                if limit is not None and len(out) >= limit:
                    return out
    return out


def edge_label(data: dict[str, Any]) -> str:
    source = str(data.get("source_label") or data.get("source_node_id") or "")
    relation = str(data.get("relation_type") or "related_to")
    target = str(data.get("target_label") or data.get("target_node_id") or "")
    return f"{source} --{relation}--> {target}"


def graph_degree_ranking(graph: nx.MultiDiGraph) -> list[tuple[str, int]]:
    return sorted(graph.degree(), key=lambda item: (-int(item[1]), node_label(graph, str(item[0]))))


def copy_nodes(graph: nx.MultiDiGraph, node_ids: list[str]) -> nx.MultiDiGraph:
    sub = nx.MultiDiGraph(
        graph_is_not_proof=True,
        support_status="not_checked",
        write_permission=False,
        visualization_is_audit_support_only=True,
    )
    for node_id in node_ids:
        if node_id in graph:
            sub.add_node(node_id, **dict(graph.nodes[node_id]))
    return sub


def subgraph_from_edges(
    graph: nx.MultiDiGraph,
    *,
    edge_keys: list[tuple[str, str, str]],
    max_edges: int,
) -> nx.MultiDiGraph:
    sub = nx.MultiDiGraph(
        graph_is_not_proof=True,
        support_status="not_checked",
        write_permission=False,
        visualization_is_audit_support_only=True,
    )
    for source, target, key in edge_keys[:max_edges]:
        if source not in graph or target not in graph or not graph.has_edge(source, target, key):
            continue
        if source not in sub:
            sub.add_node(source, **dict(graph.nodes[source]))
        if target not in sub:
            sub.add_node(target, **dict(graph.nodes[target]))
        sub.add_edge(source, target, key=key, **dict(graph.edges[source, target, key]))
    return sub


def bounded_induced_subgraph(
    graph: nx.MultiDiGraph,
    node_ids: list[str],
    *,
    max_nodes: int,
    max_edges: int,
) -> tuple[nx.MultiDiGraph, list[str]]:
    available = [node_id for node_id in node_ids if node_id in graph]
    warnings: list[str] = []
    if len(available) > max_nodes:
        degree_order = {node_id: int(graph.degree(node_id)) for node_id in available}
        available = sorted(available, key=lambda node_id: (-degree_order[node_id], node_label(graph, node_id)))[:max_nodes]
        warnings.append("visual_slice_truncated_by_degree")
    allowed = set(available)
    edge_keys: list[tuple[str, str, str]] = []
    for source, target, key in graph.edges(keys=True):
        if source in allowed and target in allowed:
            edge_keys.append((str(source), str(target), str(key)))
    if len(edge_keys) > max_edges:
        warnings.append("visual_slice_edges_truncated")
    return subgraph_from_edges(graph, edge_keys=edge_keys, max_edges=max_edges), warnings


def ego_slice(
    graph: nx.MultiDiGraph,
    node_id: str,
    *,
    max_nodes: int,
    max_edges: int,
) -> tuple[nx.MultiDiGraph, list[str]]:
    neighbors = set(graph.predecessors(node_id)) | set(graph.successors(node_id)) | {node_id}
    sub, warnings = bounded_induced_subgraph(graph, list(neighbors), max_nodes=max_nodes, max_edges=max_edges)
    if not sub.number_of_edges() and node_id in graph:
        sub = copy_nodes(graph, [node_id])
        warnings.append("ego_slice_has_no_visible_edges")
    return sub, warnings


def community_slice(
    graph: nx.MultiDiGraph,
    community: dict[str, Any],
    *,
    max_nodes: int,
    max_edges: int,
) -> tuple[nx.MultiDiGraph, list[str]]:
    sub, warnings = bounded_induced_subgraph(
        graph,
        string_list(community.get("entity_ids")),
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    if community.get("activation_quality") == "review_heavy":
        warnings.append("community_activation_quality_review_heavy")
    if float(community.get("generic_relation_ratio") or 0.0) > 0.15:
        warnings.append("community_generic_relation_ratio_high")
    if float(community.get("weak_node_ratio") or 0.0) > 0.5:
        warnings.append("community_weak_node_ratio_high")
    return sub, warnings


def noisy_edge_slice(
    graph: nx.MultiDiGraph,
    *,
    max_edges: int,
) -> tuple[nx.MultiDiGraph, list[str]]:
    edge_keys: list[tuple[str, str, str]] = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        relation_type = str(data.get("relation_type") or "")
        generic_hint = str(data.get("generic_relation_review_hint") or "")
        if relation_type == "related_to_generic" or generic_hint not in {"", "not_generic"}:
            edge_keys.append((str(source), str(target), str(key)))
    warnings = ["noisy_or_generic_edge_review_slice"]
    if len(edge_keys) > max_edges:
        warnings.append("visual_slice_edges_truncated")
    return subgraph_from_edges(graph, edge_keys=edge_keys, max_edges=max_edges), warnings


def weak_node_slice(
    graph: nx.MultiDiGraph,
    *,
    max_nodes: int,
    max_edges: int,
) -> tuple[nx.MultiDiGraph, list[str]]:
    weak_nodes = [
        node_id
        for node_id, data in graph.nodes(data=True)
        if str(data.get("entity_quality_hint") or "") in WEAK_NODE_HINTS or graph.degree(node_id) == 0
    ]
    selected = sorted(weak_nodes, key=lambda node_id: (-int(graph.degree(node_id)), node_label(graph, str(node_id))))[:max_nodes]
    edge_keys: list[tuple[str, str, str]] = []
    allowed = set(selected)
    for source, target, key in graph.edges(keys=True):
        if source in allowed or target in allowed:
            edge_keys.append((str(source), str(target), str(key)))
    sub = subgraph_from_edges(graph, edge_keys=edge_keys, max_edges=max_edges)
    for node_id in selected:
        if node_id not in sub and node_id in graph:
            sub.add_node(node_id, **dict(graph.nodes[node_id]))
    warnings = ["weak_or_isolated_node_review_slice"]
    if len(weak_nodes) > max_nodes:
        warnings.append("visual_slice_nodes_truncated")
    if len(edge_keys) > max_edges:
        warnings.append("visual_slice_edges_truncated")
    return sub, warnings


def path_slice(
    graph: nx.MultiDiGraph,
    algorithm_row: dict[str, Any],
    *,
    max_edges: int,
) -> tuple[nx.MultiDiGraph, list[str]]:
    payload = algorithm_row.get("payload") or {}
    path_node_ids = string_list(payload.get("path_node_ids"))
    evidence_path = payload.get("evidence_path") if isinstance(payload.get("evidence_path"), list) else []
    edge_keys: list[tuple[str, str, str]] = []
    for segment in evidence_path:
        if not isinstance(segment, dict):
            continue
        source = str(segment.get("source_node_id") or "")
        target = str(segment.get("target_node_id") or "")
        relation = str(segment.get("relation_type") or "")
        if not graph.has_edge(source, target):
            continue
        for key, data in graph[source][target].items():
            if relation and str(data.get("relation_type") or "") != relation:
                continue
            edge_keys.append((source, target, str(key)))
            break
    if not edge_keys and len(path_node_ids) >= 2:
        for source, target in zip(path_node_ids, path_node_ids[1:]):
            if not graph.has_edge(source, target):
                continue
            key = next(iter(graph[source][target]))
            edge_keys.append((source, target, str(key)))
    sub = subgraph_from_edges(graph, edge_keys=edge_keys, max_edges=max_edges)
    for node_id in path_node_ids:
        if node_id in graph and node_id not in sub:
            sub.add_node(node_id, **dict(graph.nodes[node_id]))
    warnings = ["evidence_path_is_navigation_support_only"]
    if algorithm_row.get("graph_is_not_proof") is not True:
        warnings.append("source_algorithm_row_missing_graph_is_not_proof")
    return sub, warnings


def cytoscape_elements(graph: nx.MultiDiGraph) -> dict[str, Any]:
    nodes = [
        {
            "data": {
                "id": str(node_id),
                "label": data.get("label") or str(node_id),
                "description": data.get("description") or "",
                "entity_type": data.get("entity_type") or "",
                "entity_quality_hint": data.get("entity_quality_hint") or "",
                "evidence_refs": string_list(data.get("evidence_refs"))[:10],
                "raw_backpointer_refs": string_list(data.get("raw_backpointer_refs"))[:10],
                "source_refs": string_list(data.get("source_refs"))[:10],
                "source_text_quotes": string_list(data.get("source_text_quotes"))[:10],
                "warnings": string_list(data.get("warnings"))[:10],
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        }
        for node_id, data in graph.nodes(data=True)
    ]
    grouped_edges: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for source, target, key, data in graph.edges(keys=True, data=True):
        edge_id = str(data.get("edge_id") or key)
        relation_type = str(data.get("relation_type") or "related_to")
        generic_hint = str(data.get("generic_relation_review_hint") or "")
        group_key = (str(source), str(target), relation_type, generic_hint)
        if group_key not in grouped_edges:
            grouped_edges[group_key] = {
                "id": stable_id("graph_visual_edge_group", json.dumps(group_key, ensure_ascii=False)),
                "source": str(source),
                "target": str(target),
                "label": relation_type,
                "source_label": data.get("source_label") or str(source),
                "target_label": data.get("target_label") or str(target),
                "description": data.get("description") or "",
                "relation_type": relation_type,
                "raw_relation_types": [],
                "generic_relation_review_hint": generic_hint,
                "generic_relation_review_reasons": [],
                "evidence_count": 0,
                "evidence_refs": [],
                "raw_backpointer_refs": [],
                "source_refs": [],
                "source_text_quotes": [],
                "source_perspectives": [],
                "attribution_statuses": [],
                "confidence_hints": [],
                "temporal_scopes": [],
                "warnings": [],
                "source_edge_ids": [],
                "visual_edge_group_count": 0,
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        group = grouped_edges[group_key]
        group["source_edge_ids"].append(edge_id)
        group["visual_edge_group_count"] += 1
        group["raw_relation_types"] = uniq([group["raw_relation_types"], data.get("raw_relation_types")], limit=12)
        group["generic_relation_review_reasons"] = uniq(
            [group["generic_relation_review_reasons"], data.get("generic_relation_review_reasons")],
            limit=12,
        )
        group["evidence_refs"] = uniq([group["evidence_refs"], data.get("evidence_refs")], limit=20)
        group["raw_backpointer_refs"] = uniq([group["raw_backpointer_refs"], data.get("raw_backpointer_refs")], limit=20)
        group["source_refs"] = uniq([group["source_refs"], data.get("source_refs")], limit=20)
        group["source_text_quotes"] = uniq([group["source_text_quotes"], data.get("source_text_quotes")], limit=12)
        group["source_perspectives"] = uniq([group["source_perspectives"], data.get("source_perspective")], limit=8)
        group["attribution_statuses"] = uniq([group["attribution_statuses"], data.get("attribution_status")], limit=8)
        group["confidence_hints"] = uniq([group["confidence_hints"], data.get("confidence_hint")], limit=8)
        group["temporal_scopes"] = uniq([group["temporal_scopes"], data.get("temporal_scope")], limit=8)
        group["warnings"] = uniq([group["warnings"], data.get("warnings")], limit=20)
        group["evidence_count"] += int(data.get("evidence_count") or len(string_list(data.get("evidence_refs"))))
    edges = [{"data": data} for data in grouped_edges.values()]
    return {
        "nodes": nodes,
        "edges": edges,
        "visual_edge_groups": True,
        "raw_edge_count": graph.number_of_edges(),
        "visual_edge_count": len(edges),
    }


def slice_metadata(
    *,
    slice_id: str,
    slice_kind: str,
    title: str,
    reason: str,
    graph: nx.MultiDiGraph,
    warnings: list[str],
    graphml_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    evidence_refs = uniq(
        [data.get("evidence_refs") for _, data in graph.nodes(data=True)]
        + [data.get("evidence_refs") for _, _, _, data in graph.edges(keys=True, data=True)],
        limit=30,
    )
    relation_counts = Counter(str(data.get("relation_type") or "unknown") for _, _, _, data in graph.edges(keys=True, data=True))
    node_quality_counts = Counter(str(data.get("entity_quality_hint") or "unknown") for _, data in graph.nodes(data=True))
    return {
        "schema_version": VISUAL_SLICE_SCHEMA_VERSION,
        "slice_id": slice_id,
        "slice_kind": slice_kind,
        "title": title,
        "reason": reason,
        "node_count": graph.number_of_nodes(),
        "edge_count": graph.number_of_edges(),
        "relation_type_counts": dict(sorted(relation_counts.items())),
        "node_quality_counts": dict(sorted(node_quality_counts.items())),
        "evidence_refs": evidence_refs,
        "warnings": uniq([warnings], limit=30),
        "graphml_path": str(graphml_path),
        "json_path": str(json_path),
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
        "visualization_is_audit_support_only": True,
    }


def write_slice(
    *,
    output_dir: Path,
    slice_kind: str,
    title: str,
    reason: str,
    graph: nx.MultiDiGraph,
    warnings: list[str],
) -> dict[str, Any]:
    safe_id = stable_id("graph_visual_slice", json.dumps([slice_kind, title], ensure_ascii=False, sort_keys=True))
    safe_name = safe_id.split(":", 1)[1]
    slice_dir = output_dir / "slices"
    graphml_path = slice_dir / f"{slice_kind}_{safe_name}.graphml"
    json_path = slice_dir / f"{slice_kind}_{safe_name}.json"
    graph.graph.update(
        {
            "slice_kind": slice_kind,
            "slice_title": title,
            "graph_is_not_proof": True,
            "support_status": "not_checked",
            "write_permission": False,
            "visualization_is_audit_support_only": True,
        }
    )
    write_graphml(graphml_path, graph)
    write_json(
        json_path,
        {
            "schema_version": VISUAL_SLICE_SCHEMA_VERSION,
            "slice_id": safe_id,
            "slice_kind": slice_kind,
            "title": title,
            "reason": reason,
            "elements": cytoscape_elements(graph),
            "warnings": uniq([warnings], limit=30),
            "graph_is_not_proof": True,
            "support_status": "not_checked",
            "write_permission": False,
            "visualization_is_audit_support_only": True,
        },
    )
    return slice_metadata(
        slice_id=safe_id,
        slice_kind=slice_kind,
        title=title,
        reason=reason,
        graph=graph,
        warnings=warnings,
        graphml_path=graphml_path,
        json_path=json_path,
    )


def write_report(path: Path, manifest: dict[str, Any], slices: list[dict[str, Any]]) -> None:
    lines = [
        "# v0.31 Graph Visual Review Bundle",
        "",
        f"- workspace: `{manifest['workspace']}`",
        f"- graph_dir: `{manifest['graph_dir']}`",
        f"- projection: `{manifest['projection']}`",
        f"- slices: {manifest['counts']['slices']}",
        f"- nodes across slices: {manifest['counts']['slice_nodes_total']}",
        f"- edges across slices: {manifest['counts']['slice_edges_total']}",
        "",
        "## Boundary",
        "",
        "- visualization is audit support only;",
        "- graph_is_not_proof=true;",
        "- support_status=not_checked;",
        "- write_permission=false;",
        "- layout position, color, and size are not evidence.",
        "",
        "## Borrowed Mature Patterns",
        "",
        "- GraphRAG-style GraphML snapshot / Gephi review path;",
        "- LightRAG-style bounded subgraph retrieval by entity/label;",
        "- NetworkX as algorithm/projection layer, not final visualization UI.",
        "",
        "## Slices",
        "",
    ]
    for row in slices:
        lines.extend(
            [
                f"### {row['title']}",
                "",
                f"- kind: `{row['slice_kind']}`",
                f"- reason: {row['reason']}",
                f"- nodes: {row['node_count']}",
                f"- edges: {row['edge_count']}",
                f"- graphml: `{row['graphml_path']}`",
                f"- json: `{row['json_path']}`",
                f"- warnings: {', '.join(row['warnings'][:8]) if row['warnings'] else 'none'}",
                f"- evidence refs sample: {', '.join(row['evidence_refs'][:8]) if row['evidence_refs'] else 'none'}",
                "",
            ]
        )
    write_text(path, "\n".join(lines))


def _write_html_index_legacy_mojibake(path: Path, manifest: dict[str, Any], slices: list[dict[str, Any]]) -> None:
    slice_payloads: list[dict[str, Any]] = []
    for row in slices:
        json_path = Path(str(row.get("json_path") or ""))
        payload = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        slice_payloads.append(
            {
                "metadata": row,
                "elements": (payload.get("elements") or {"nodes": [], "edges": []}),
            }
        )
    payload_json = json.dumps(slice_payloads, ensure_ascii=False)
    manifest_json = json.dumps(
        {
            "workspace": manifest.get("workspace"),
            "projection": manifest.get("projection"),
            "counts": manifest.get("counts"),
            "boundary": manifest.get("boundary"),
        },
        ensure_ascii=False,
        indent=2,
    )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v0.31 图可视化审计</title>
  <script src="https://unpkg.com/cytoscape@3.29.2/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/layout-base/layout-base.js"></script>
  <script src="https://unpkg.com/cose-base/cose-base.js"></script>
  <script src="https://unpkg.com/cytoscape-fcose/cytoscape-fcose.js"></script>
  <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://unpkg.com/cytoscape-dagre/cytoscape-dagre.js"></script>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #18202a;
      --muted: #687385;
      --line: #d9dee7;
      --accent: #2563eb;
      --warn: #a16207;
      --danger: #b42318;
      --good: #047857;
      --weak: #d97706;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    main {{
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      min-height: 100vh;
    }}
    aside {{
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 20px;
      overflow: auto;
    }}
    section {{
      display: grid;
      grid-template-rows: auto minmax(420px, 1fr) auto;
      min-width: 0;
    }}
    h1 {{
      font-size: 20px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }}
    h2 {{
      font-size: 17px;
      margin: 0 0 8px;
    }}
    p, li, pre {{
      font-size: 13px;
      line-height: 1.45;
    }}
    pre {{
      white-space: pre-wrap;
      background: #eef2f7;
      padding: 10px;
      border-radius: 6px;
      overflow: auto;
    }}
    .muted {{ color: var(--muted); }}
    .toolbar {{
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 14px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      min-width: 0;
      flex-wrap: wrap;
    }}
    select, button {{
      min-height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
    }}
    button {{
      cursor: pointer;
    }}
    #sliceSelect {{
      flex: 1;
      min-width: 260px;
    }}
    #cy {{
      min-height: 620px;
      background:
        linear-gradient(#eef2f7 1px, transparent 1px),
        linear-gradient(90deg, #eef2f7 1px, transparent 1px),
        #fbfcfe;
      background-size: 28px 28px;
    }}
    .details {{
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 14px 16px;
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(280px, 0.6fr);
      gap: 16px;
    }}
    .pill {{
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: #edf3ff;
      color: #1d4ed8;
      font-size: 12px;
      margin: 2px 4px 2px 0;
    }}
    .legend {{
      display: grid;
      gap: 6px;
      margin: 12px 0;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }}
    .dot {{
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
      border: 1px solid #ffffff;
      box-shadow: 0 0 0 1px var(--line);
    }}
    .line-sample {{
      width: 22px;
      height: 0;
      border-top: 2px dashed #ef4444;
    }}
    .checkbox-row {{
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }}
    .edge-list {{
      margin-top: 10px;
      max-height: 220px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #f8fafc;
    }}
    .edge-card {{
      width: 100%;
      display: block;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      padding: 8px 10px;
      background: transparent;
      cursor: pointer;
    }}
    .edge-card:hover {{
      background: #eef4ff;
    }}
    .edge-title {{
      font-size: 13px;
      color: var(--ink);
      line-height: 1.35;
    }}
    .edge-meta {{
      font-size: 12px;
      color: var(--muted);
      margin-top: 3px;
      line-height: 1.35;
    }}
    .edge-card.generic .edge-title {{
      color: var(--danger);
    }}
    .warn {{
      color: var(--warn);
    }}
    .danger {{
      color: var(--danger);
    }}
    .fallback {{
      display: none;
      padding: 16px;
      color: var(--danger);
      background: #fff7ed;
      border-top: 1px solid #fed7aa;
    }}
    @media (max-width: 900px) {{
      main {{ grid-template-columns: 1fr; }}
      aside {{ border-right: none; border-bottom: 1px solid var(--line); }}
      .details {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
<main>
  <aside>
    <h1>v0.31 图可视化审计</h1>
    <p class="muted">可视化只用于人工审计，不是正确性证明。</p>
    <pre>{html.escape(manifest_json)}</pre>
    <h2>怎么看</h2>
    <ul>
      <li>节点越大，表示在当前切片里的局部连接越多。</li>
      <li>红色虚线边表示 generic 或需要复检的关系候选。</li>
      <li>橙色节点表示 weak / context-dependent / generic-fragment 候选。</li>
      <li>浏览器图只是快速审计；需要更成熟布局时打开 GraphML 到 Gephi/yEd。</li>
    </ul>
    <div class="legend">
      <div class="legend-row"><span class="dot" style="background:#2563eb"></span>人物/普通稳定节点</div>
      <div class="legend-row"><span class="dot" style="background:#059669"></span>项目/地点/材料类节点</div>
      <div class="legend-row"><span class="dot" style="background:#d97706"></span>弱节点或上下文依赖节点</div>
      <div class="legend-row"><span class="line-sample"></span>generic / review-heavy 边</div>
    </div>
  </aside>
  <section>
    <div class="toolbar">
      <label for="sliceSelect">切片</label>
      <select id="sliceSelect"></select>
      <label for="layoutSelect">布局</label>
      <select id="layoutSelect">
        <option value="auto">自动</option>
        <option value="fcose">语义簇 fCoSE</option>
        <option value="dagre">层级 Dagre</option>
        <option value="concentric">同心</option>
        <option value="cose">力导向</option>
        <option value="breadthfirst">层级</option>
        <option value="circle">圆形</option>
      </select>
      <label class="checkbox-row"><input id="edgeLabelToggle" type="checkbox">显示边标签</label>
      <button id="fitBtn">适配</button>
      <button id="layoutBtn">重排</button>
    </div>
    <div id="cy"></div>
    <div id="fallback" class="fallback">
      Cytoscape.js 没有加载成功。可以用 Gephi/yEd 打开 GraphML，或直接查看 JSON 切片。
    </div>
    <div class="details">
      <div>
        <h2 id="sliceTitle"></h2>
        <p id="sliceReason" class="muted"></p>
        <div id="sliceStats"></div>
        <div id="sliceWarnings"></div>
        <h2>边列表</h2>
        <div id="edgeList" class="edge-list"></div>
      </div>
      <div>
        <h2>选中对象</h2>
        <pre id="selectedDetails">点击节点或边查看属性。</pre>
      </div>
    </div>
  </section>
</main>
<script>
const SLICE_DATA = {payload_json};

function escapeText(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}

function localDegree(elements) {{
  const degree = new Map();
  for (const node of elements.nodes) degree.set(node.data.id, 0);
  for (const edge of elements.edges) {{
    degree.set(edge.data.source, (degree.get(edge.data.source) || 0) + 1);
    degree.set(edge.data.target, (degree.get(edge.data.target) || 0) + 1);
  }}
  return degree;
}}

function edgeArray(elements) {{
  return Array.isArray(elements.edges) ? elements.edges : [];
}}

function normalizeElements(raw) {{
  const degree = localDegree(raw);
  const nodes = raw.nodes.map(node => {{
    const d = node.data;
    const weak = ['context_dependent', 'generic_fragment', 'action_phrase', 'review_required'].includes(d.entity_quality_hint);
    const entityType = String(d.entity_type || '').toLowerCase();
    const semanticGroup = ['project', 'place', 'material', 'event', 'organization'].includes(entityType) ? 'object' : 'person_or_general';
    return {{
      data: {{
        ...d,
        degree: degree.get(d.id) || 0,
        weak: weak ? 'yes' : 'no',
        semanticGroup,
      }}
    }};
  }});
  const edges = edgeArray(raw).map(edge => {{
    const d = edge.data;
    const generic = d.relation_type === 'related_to_generic' || (d.generic_relation_review_hint && d.generic_relation_review_hint !== 'not_generic');
    return {{
      data: {{
        ...d,
        generic: generic ? 'yes' : 'no',
      }}
    }};
  }});
  return [...nodes, ...edges];
}}

function sliceKindLabel(kind) {{
  return {{
    ego_network: '实体邻域',
    community_cluster: '社区簇',
    noisy_generic_edges: '噪声/泛化边',
    weak_or_isolated_nodes: '弱节点/孤立节点',
    evidence_path: '证据路径'
  }}[kind] || kind;
}}

function defaultLayoutFor(slice) {{
  const kind = slice.metadata.slice_kind;
  if (kind === 'evidence_path') return 'dagre';
  if (kind === 'ego_network') return 'concentric';
  if (kind === 'community_cluster') return slice.metadata.node_count <= 12 ? 'circle' : 'fcose';
  if (kind === 'noisy_generic_edges') return 'concentric';
  return 'fcose';
}}

function layoutConfig(slice, requested) {{
  const name = requested === 'auto' ? defaultLayoutFor(slice) : requested;
  const base = {{ name, animate: false, fit: true, padding: 52 }};
  if (name === 'fcose') {{
    return {{
      ...base,
      quality: 'proof',
      randomize: false,
      nodeRepulsion: 9000,
      idealEdgeLength: 130,
      edgeElasticity: 0.35,
      gravity: 0.2,
      numIter: 1800
    }};
  }}
  if (name === 'dagre') {{
    return {{
      ...base,
      rankDir: 'LR',
      nodeSep: 90,
      edgeSep: 24,
      rankSep: 120
    }};
  }}
  if (name === 'concentric') {{
    return {{
      ...base,
      minNodeSpacing: 48,
      concentric: node => node.data('degree') || 1,
      levelWidth: () => 2
    }};
  }}
  if (name === 'breadthfirst') {{
    const nodes = slice.elements.nodes || [];
    const rootId = nodes[0]?.data?.id;
    return {{
      ...base,
      directed: true,
      spacingFactor: 1.4,
      roots: rootId ? `#${{CSS.escape(rootId)}}` : undefined
    }};
  }}
  if (name === 'circle') {{
    return {{ ...base, spacingFactor: 1.25 }};
  }}
  return {{
    ...base,
    name: 'cose',
    nodeRepulsion: 11000,
    idealEdgeLength: 150,
    edgeElasticity: 70,
    nestingFactor: 1.2,
    gravity: 0.25,
    numIter: 1200,
    randomize: false
  }};
}}

function applyEdgeLabelMode(cy) {{
  const show = document.getElementById('edgeLabelToggle').checked;
  cy.style()
    .selector('edge')
    .style('label', show ? 'data(label)' : '')
    .update();
}}

function relationHelp(edge) {{
  const relation = edge.relation_type || edge.label || 'related_to';
  const raw = Array.isArray(edge.raw_relation_types) && edge.raw_relation_types.length
    ? `原始关系: ${{edge.raw_relation_types.join(', ')}}`
    : '无原始关系类型';
  if (relation === 'related_to_generic' || (edge.generic_relation_review_hint && edge.generic_relation_review_hint !== 'not_generic')) {{
    return `泛化/待复检关系；${{raw}}；复检提示: ${{edge.generic_relation_review_hint || 'generic'}}`;
  }}
  return `规范化关系类型: ${{relation}}；${{raw}}`;
}}

function renderEdgeList(slice) {{
  const nodeLabels = new Map((slice.elements.nodes || []).map(node => [node.data.id, node.data.label || node.data.id]));
  const edges = edgeArray(slice.elements);
  const edgeList = document.getElementById('edgeList');
  if (!edges.length) {{
    edgeList.innerHTML = '<div class="edge-card"><div class="edge-meta">这个切片没有可见边。</div></div>';
    return;
  }}
  edgeList.innerHTML = edges.map(edge => {{
    const d = edge.data;
    const source = d.source_label || nodeLabels.get(d.source) || d.source;
    const target = d.target_label || nodeLabels.get(d.target) || d.target;
    const relation = d.relation_type || d.label || 'related_to';
    const generic = relation === 'related_to_generic' || (d.generic_relation_review_hint && d.generic_relation_review_hint !== 'not_generic');
    const description = d.description ? `<div class="edge-meta">${{escapeText(d.description)}}</div>` : '';
    const groupText = d.visual_edge_group_count && d.visual_edge_group_count > 1
      ? `；合并 ${{d.visual_edge_group_count}} 条原始边`
      : '';
    return `<button class="edge-card ${{generic ? 'generic' : ''}}" data-edge-id="${{escapeText(d.id)}}">`
      + `<div class="edge-title">${{escapeText(source)}} → <strong>${{escapeText(relation)}}</strong> → ${{escapeText(target)}}</div>`
      + `<div class="edge-meta">${{escapeText(relationHelp(d))}}</div>`
      + description
      + `<div class="edge-meta">证据 ${{(d.evidence_refs || []).length}} 条；warning ${{(d.warnings || []).length}} 条${{groupText}}</div>`
      + `</button>`;
  }}).join('');
}}

function renderMetadata(slice) {{
  const m = slice.metadata;
  document.getElementById('sliceTitle').textContent = m.title;
  document.getElementById('sliceReason').textContent = m.reason;
  document.getElementById('sliceStats').innerHTML = [
    `<span class="pill">${{escapeText(sliceKindLabel(m.slice_kind))}}</span>`,
    `<span class="pill">节点 ${{m.node_count}}</span>`,
    `<span class="pill">原始边 ${{m.edge_count}}</span>`,
    `<span class="pill">可视边 ${{edgeArray(slice.elements).length}}</span>`,
    `<span class="pill">graph_is_not_proof</span>`,
    `<span class="pill">not_checked</span>`
  ].join('');
  document.getElementById('sliceWarnings').innerHTML = (m.warnings || []).length
    ? `<p class="warn">警告：${{escapeText((m.warnings || []).join('，'))}}</p>`
    : '<p class="muted">警告：无</p>';
  document.getElementById('selectedDetails').textContent = '点击节点或边查看属性。';
  renderEdgeList(slice);
}}

function renderSlice(cy, slice) {{
  renderMetadata(slice);
  cy.elements().remove();
  cy.add(normalizeElements(slice.elements));
  applyEdgeLabelMode(cy);
  const requested = document.getElementById('layoutSelect').value;
  const config = layoutConfig(slice, requested);
  try {{
    cy.layout(config).run();
  }} catch (error) {{
    const fallback = config.name === 'dagre'
      ? {{ name: 'breadthfirst', directed: true, animate: false, fit: true, padding: 52 }}
      : {{ name: 'cose', animate: false, fit: true, padding: 52, nodeRepulsion: 10000, idealEdgeLength: 145, numIter: 1000 }};
    cy.layout(fallback).run();
  }}
}}

function init() {{
  const select = document.getElementById('sliceSelect');
  SLICE_DATA.forEach((slice, index) => {{
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${{sliceKindLabel(slice.metadata.slice_kind)}} | ${{slice.metadata.title}}`;
    select.appendChild(option);
  }});
  if (!window.cytoscape) {{
    document.getElementById('fallback').style.display = 'block';
    return;
  }}
  if (window.cytoscapeFcose) {{
    cytoscape.use(window.cytoscapeFcose);
  }}
  if (window.cytoscapeDagre) {{
    cytoscape.use(window.cytoscapeDagre);
  }}
  const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements: [],
    style: [
      {{
        selector: 'node',
        style: {{
          'label': 'data(label)',
          'font-size': 11,
          'text-wrap': 'wrap',
          'text-max-width': 110,
          'background-color': '#2563eb',
          'border-width': 1,
          'border-color': '#1d4ed8',
          'width': 'mapData(degree, 0, 20, 28, 76)',
          'height': 'mapData(degree, 0, 20, 28, 76)',
          'color': '#1f2937',
          'text-valign': 'bottom',
          'text-margin-y': 6
        }}
      }},
      {{
        selector: 'node[semanticGroup = "object"]',
        style: {{
          'background-color': '#059669',
          'border-color': '#047857'
        }}
      }},
      {{
        selector: 'node[weak = "yes"]',
        style: {{
          'background-color': '#d97706',
          'border-color': '#92400e'
        }}
      }},
      {{
        selector: 'edge',
        style: {{
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'target-arrow-color': '#64748b',
          'line-color': '#94a3b8',
          'width': 2,
          'label': '',
          'font-size': 9,
          'text-rotation': 'autorotate',
          'text-background-color': '#ffffff',
          'text-background-opacity': 0.75,
          'text-background-padding': 2
        }}
      }},
      {{
        selector: 'edge[generic = "yes"]',
        style: {{
          'line-color': '#ef4444',
          'target-arrow-color': '#ef4444',
          'line-style': 'dashed'
        }}
      }},
      {{
        selector: ':selected',
        style: {{
          'border-width': 4,
          'border-color': '#111827',
          'line-color': '#111827',
          'target-arrow-color': '#111827'
        }}
      }}
    ]
  }});
  cy.on('tap', 'node, edge', event => {{
    document.getElementById('selectedDetails').textContent = JSON.stringify(event.target.data(), null, 2);
  }});
  document.getElementById('edgeList').addEventListener('click', event => {{
    const card = event.target.closest('.edge-card');
    if (!card || !card.dataset.edgeId) return;
    const edge = cy.getElementById(card.dataset.edgeId);
    if (!edge || edge.empty()) return;
    cy.elements().unselect();
    edge.select();
    cy.animate({{ center: {{ eles: edge }}, duration: 220 }});
    document.getElementById('selectedDetails').textContent = JSON.stringify(edge.data(), null, 2);
  }});
  select.addEventListener('change', () => renderSlice(cy, SLICE_DATA[Number(select.value)]));
  document.getElementById('layoutSelect').addEventListener('change', () => renderSlice(cy, SLICE_DATA[Number(select.value)]));
  document.getElementById('edgeLabelToggle').addEventListener('change', () => applyEdgeLabelMode(cy));
  document.getElementById('fitBtn').addEventListener('click', () => cy.fit(undefined, 52));
  document.getElementById('layoutBtn').addEventListener('click', () => renderSlice(cy, SLICE_DATA[Number(select.value)]));
  renderSlice(cy, SLICE_DATA[0]);
}}

init();
</script>
</body>
</html>
"""
    write_text(path, document)


def write_html_index(path: Path, manifest: dict[str, Any], slices: list[dict[str, Any]]) -> None:
    slice_payloads: list[dict[str, Any]] = []
    for row in slices:
        json_path = Path(str(row.get("json_path") or ""))
        payload = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
        slice_payloads.append(
            {
                "metadata": row,
                "elements": (payload.get("elements") or {"nodes": [], "edges": []}),
            }
        )

    payload_json = json.dumps(slice_payloads, ensure_ascii=False).replace("</", "<\\/")
    manifest_summary = {
        "workspace": manifest.get("workspace"),
        "projection": manifest.get("projection"),
        "counts": manifest.get("counts"),
        "boundary": manifest.get("boundary"),
    }
    manifest_data_json = json.dumps(manifest_summary, ensure_ascii=False).replace("</", "<\\/")
    manifest_json = json.dumps(manifest_summary, ensure_ascii=False, indent=2)
    counts = manifest.get("counts") or {}
    summary_items = [
        ("投影", manifest.get("projection") or ""),
        ("切片", counts.get("slices", 0)),
        ("输入节点", counts.get("input_nodes", 0)),
        ("输入边", counts.get("input_edges", 0)),
        ("切片节点", counts.get("slice_nodes_total", 0)),
        ("切片边", counts.get("slice_edges_total", 0)),
    ]
    summary_html = "\n".join(
        f'<div class="summary-card"><span>{html.escape(str(label))}</span><strong>{html.escape(str(value))}</strong></div>'
        for label, value in summary_items
    )

    document = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>v0.31 图可视化审计</title>
  <script src="https://unpkg.com/cytoscape@3.29.2/dist/cytoscape.min.js"></script>
  <script src="https://unpkg.com/layout-base/layout-base.js"></script>
  <script src="https://unpkg.com/cose-base/cose-base.js"></script>
  <script src="https://unpkg.com/cytoscape-fcose/cytoscape-fcose.js"></script>
  <script src="https://unpkg.com/dagre@0.8.5/dist/dagre.min.js"></script>
  <script src="https://unpkg.com/cytoscape-dagre/cytoscape-dagre.js"></script>
  <style>
    :root {
      --bg: #f5f7fb;
      --panel: #ffffff;
      --ink: #172033;
      --muted: #667085;
      --line: #d8deea;
      --soft: #eef3fb;
      --accent: #2563eb;
      --accent-soft: #e8f0ff;
      --object: #079669;
      --weak: #c77700;
      --danger: #c24132;
      --danger-soft: #fff1f0;
      --warn: #9a6700;
      --shadow: 0 14px 35px rgba(15, 23, 42, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, "Microsoft YaHei", "PingFang SC", "Segoe UI", Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    main {
      display: grid;
      grid-template-columns: 340px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      padding: 20px;
      overflow: auto;
    }
    section {
      display: grid;
      grid-template-rows: auto minmax(560px, 1fr) minmax(280px, auto);
      min-width: 0;
    }
    h1 {
      font-size: 20px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }
    h2 {
      font-size: 16px;
      margin: 0 0 10px;
      letter-spacing: 0;
    }
    h3 {
      font-size: 14px;
      margin: 0 0 8px;
      letter-spacing: 0;
    }
    p, li, pre, summary {
      font-size: 13px;
      line-height: 1.45;
    }
    pre {
      white-space: pre-wrap;
      background: #f0f4fa;
      padding: 10px;
      border-radius: 6px;
      overflow: auto;
      max-height: 220px;
    }
    .muted { color: var(--muted); }
    .boundary {
      margin: 14px 0;
      padding: 10px 12px;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      background: #fff7ed;
      color: #8a4b0d;
      font-size: 13px;
      line-height: 1.45;
    }
    .summary-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
      margin: 14px 0;
    }
    .summary-card {
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fbfcff;
    }
    .summary-card span {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
    }
    .summary-card strong {
      display: block;
      font-size: 15px;
      overflow-wrap: anywhere;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      padding: 14px 16px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      min-width: 0;
      flex-wrap: wrap;
    }
    select, button {
      min-height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
      border-radius: 6px;
      padding: 6px 10px;
      font-size: 13px;
    }
    button { cursor: pointer; }
    button:hover { border-color: #aab7cf; }
    #sliceSelect {
      flex: 1;
      min-width: 260px;
    }
    #cy {
      min-height: 560px;
      background:
        linear-gradient(#e8edf5 1px, transparent 1px),
        linear-gradient(90deg, #e8edf5 1px, transparent 1px),
        #fbfcfe;
      background-size: 28px 28px;
    }
    .details {
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 14px 16px;
      display: grid;
      grid-template-columns: minmax(0, 0.95fr) minmax(360px, 1.05fr);
      gap: 16px;
    }
    .slice-card,
    .selected-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      box-shadow: var(--shadow);
      padding: 14px;
      min-width: 0;
    }
    .pill, .chip {
      display: inline-block;
      padding: 2px 7px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: #1d4ed8;
      font-size: 12px;
      margin: 2px 4px 2px 0;
      max-width: 100%;
      overflow-wrap: anywhere;
    }
    .chip.warn {
      background: #fff7df;
      color: var(--warn);
    }
    .chip.danger {
      background: var(--danger-soft);
      color: var(--danger);
    }
    .chip.neutral {
      background: #f1f5f9;
      color: #475569;
    }
    .legend {
      display: grid;
      gap: 7px;
      margin: 12px 0;
    }
    .legend-row {
      display: flex;
      align-items: center;
      gap: 8px;
      font-size: 13px;
      color: var(--muted);
    }
    .dot {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      display: inline-block;
      border: 1px solid #ffffff;
      box-shadow: 0 0 0 1px var(--line);
      flex: 0 0 auto;
    }
    .line-sample {
      width: 24px;
      height: 0;
      border-top: 2px dashed #ef4444;
      flex: 0 0 auto;
    }
    .checkbox-row {
      display: flex;
      align-items: center;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
      white-space: nowrap;
    }
    .edge-list {
      margin-top: 10px;
      max-height: 330px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #f8fafc;
    }
    .edge-card {
      width: 100%;
      display: block;
      text-align: left;
      border: 0;
      border-bottom: 1px solid var(--line);
      border-radius: 0;
      padding: 10px 12px;
      background: transparent;
      cursor: pointer;
    }
    .edge-card:hover {
      background: #eef4ff;
    }
    .edge-card.selected {
      background: #e8f0ff;
      box-shadow: inset 4px 0 0 var(--accent);
    }
    .edge-card.generic.selected {
      background: #fff1f0;
      box-shadow: inset 4px 0 0 var(--danger);
    }
    .edge-title {
      font-size: 13px;
      color: var(--ink);
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .edge-title strong {
      color: #1d4ed8;
    }
    .edge-meta {
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .edge-card.generic .edge-title,
    .edge-card.generic .edge-title strong {
      color: var(--danger);
    }
    .detail-title {
      font-size: 15px;
      line-height: 1.35;
      margin-bottom: 4px;
      overflow-wrap: anywhere;
    }
    .detail-subtitle {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 10px;
      overflow-wrap: anywhere;
    }
    .kv-grid {
      display: grid;
      grid-template-columns: 108px minmax(0, 1fr);
      gap: 6px 10px;
      margin: 10px 0;
      font-size: 13px;
    }
    .kv-grid dt {
      color: var(--muted);
    }
    .kv-grid dd {
      margin: 0;
      overflow-wrap: anywhere;
    }
    .detail-section {
      border-top: 1px solid var(--line);
      padding-top: 10px;
      margin-top: 10px;
    }
    .quote-list {
      margin: 6px 0 0;
      padding: 0;
      list-style: none;
      display: grid;
      gap: 6px;
    }
    .quote-list li {
      padding: 8px 10px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 7px;
      overflow-wrap: anywhere;
    }
    .warn {
      color: var(--warn);
    }
    .danger {
      color: var(--danger);
    }
    .fallback {
      display: none;
      padding: 16px;
      color: var(--danger);
      background: #fff7ed;
      border-top: 1px solid #fed7aa;
    }
    details {
      margin-top: 12px;
    }
    summary {
      cursor: pointer;
      color: var(--muted);
    }
    @media (max-width: 1100px) {
      main { grid-template-columns: 1fr; }
      aside { border-right: none; border-bottom: 1px solid var(--line); }
      section { grid-template-rows: auto minmax(480px, 1fr) auto; }
      .details { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
<main>
  <aside>
    <h1>v0.31 图可视化审计</h1>
    <p class="muted">用于人工检查图候选：看聚类、噪声边、弱节点、证据路径和查询扩展路径。</p>
    <div class="boundary">边、布局、颜色、大小都不是正确性证明。图只提供审计线索：graph_is_not_proof=true，support_status=not_checked。</div>
    <div class="summary-grid">
__MANIFEST_SUMMARY_HTML__
    </div>
    <h2>怎么看</h2>
    <ul>
      <li>点击图上的边，会同步定位到下面的边列表，并在右侧显示中文解释。</li>
      <li>边列表里的证据 refs / quote 是审计入口，不是最终支持证明。</li>
      <li>红色虚线边表示 generic 或需要复检，橙色节点表示弱节点或上下文依赖节点。</li>
      <li>需要成熟图工具时，打开同目录 GraphML 到 Gephi / yEd / Neo4j Bloom / Memgraph Lab。</li>
    </ul>
    <div class="legend">
      <div class="legend-row"><span class="dot" style="background:#2563eb"></span>人物 / 普通稳定节点</div>
      <div class="legend-row"><span class="dot" style="background:#079669"></span>项目 / 地点 / 材料 / 事件类节点</div>
      <div class="legend-row"><span class="dot" style="background:#c77700"></span>弱节点或上下文依赖节点</div>
      <div class="legend-row"><span class="line-sample"></span>generic / review-heavy 边</div>
    </div>
    <details>
      <summary>查看 manifest 摘要 JSON</summary>
      <pre>__MANIFEST_JSON__</pre>
    </details>
  </aside>
  <section>
    <div class="toolbar">
      <label for="sliceSelect">切片</label>
      <select id="sliceSelect"></select>
      <label for="layoutSelect">布局</label>
      <select id="layoutSelect">
        <option value="auto">自动</option>
        <option value="fcose">语义簇 fCoSE</option>
        <option value="dagre">层级 Dagre</option>
        <option value="concentric">同心</option>
        <option value="cose">力导向</option>
        <option value="breadthfirst">层级</option>
        <option value="circle">圆形</option>
      </select>
      <label class="checkbox-row"><input id="edgeLabelToggle" type="checkbox">显示边标签</label>
      <button id="fitBtn">适配</button>
      <button id="layoutBtn">重排</button>
    </div>
    <div id="cy"></div>
    <div id="fallback" class="fallback">
      Cytoscape.js 没有加载成功。可以用 Gephi/yEd 打开 GraphML，或直接查看 JSON 切片。
    </div>
    <div class="details">
      <div class="slice-card">
        <h2 id="sliceTitle"></h2>
        <p id="sliceReason" class="muted"></p>
        <div id="sliceStats"></div>
        <div id="sliceWarnings"></div>
        <h2>边列表</h2>
        <div id="edgeList" class="edge-list"></div>
      </div>
      <div class="selected-card">
        <h2>选中对象</h2>
        <div id="selectedDetails" class="detail-panel">点击节点或边查看中文解释、证据 refs 和 warnings。</div>
      </div>
    </div>
  </section>
</main>
<script>
const MANIFEST = __MANIFEST_DATA__;
const SLICE_DATA = __SLICE_DATA__;

const RELATION_LABELS = {
  supports: '支持 / 鼓励 / 帮助',
  plans: '计划 / 打算',
  participates_in: '参与 / 经历 / 访问',
  works_on: '从事 / 推进',
  updates: '更新状态',
  prefers: '偏好 / 重视',
  constrains: '限制 / 约束',
  uses: '使用',
  depends_on: '依赖',
  contradicts: '矛盾 / 冲突',
  related_to_generic: '泛化关系，需要复检'
};

function escapeText(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}

function asList(value) {
  if (Array.isArray(value)) return value.filter(item => item !== null && item !== undefined && String(item).trim() !== '').map(String);
  if (value === null || value === undefined || String(value).trim() === '') return [];
  return [String(value)];
}

function chipList(value, cssClass = 'neutral', empty = '无') {
  const values = asList(value);
  if (!values.length) return `<span class="chip neutral">${escapeText(empty)}</span>`;
  return values.slice(0, 12).map(item => `<span class="chip ${cssClass}">${escapeText(item)}</span>`).join('');
}

function quoteList(value, empty = '暂无 quote') {
  const values = asList(value);
  if (!values.length) return `<p class="muted">${escapeText(empty)}</p>`;
  return `<ul class="quote-list">${values.slice(0, 8).map(item => `<li>${escapeText(item)}</li>`).join('')}</ul>`;
}

function compact(value, empty = '无') {
  const values = asList(value);
  return values.length ? values.join('；') : empty;
}

function readableTemporalScope(item) {
  if (!item) return '';
  let parsed = null;
  if (typeof item === 'object') parsed = item;
  if (typeof item === 'string') {
    try {
      parsed = JSON.parse(item);
    } catch (error) {
      return item;
    }
  }
  if (!parsed || typeof parsed !== 'object') return String(item);
  if (parsed.timestamp) return `时间点：${parsed.timestamp}`;
  if (parsed.start || parsed.end) return `时间段：${parsed.start || '?'} 到 ${parsed.end || '?'}`;
  return Object.entries(parsed).map(([key, value]) => `${key}: ${value}`).join('；');
}

function temporalChipList(value) {
  const values = asList(value).map(readableTemporalScope).filter(Boolean);
  return chipList(values, 'neutral');
}

function localDegree(elements) {
  const degree = new Map();
  for (const node of elements.nodes || []) degree.set(node.data.id, 0);
  for (const edge of elements.edges || []) {
    degree.set(edge.data.source, (degree.get(edge.data.source) || 0) + 1);
    degree.set(edge.data.target, (degree.get(edge.data.target) || 0) + 1);
  }
  return degree;
}

function edgeArray(elements) {
  return Array.isArray(elements.edges) ? elements.edges : [];
}

function nodeLabelMap(slice) {
  return new Map((slice.elements.nodes || []).map(node => [node.data.id, node.data.label || node.data.id]));
}

function normalizeElements(raw) {
  const degree = localDegree(raw);
  const nodes = (raw.nodes || []).map(node => {
    const d = node.data;
    const weak = ['context_dependent', 'generic_fragment', 'action_phrase', 'review_required'].includes(d.entity_quality_hint);
    const entityType = String(d.entity_type || '').toLowerCase();
    const semanticGroup = ['project', 'place', 'material', 'event', 'organization'].includes(entityType) ? 'object' : 'person_or_general';
    return {
      data: {
        ...d,
        degree: degree.get(d.id) || 0,
        weak: weak ? 'yes' : 'no',
        semanticGroup,
      }
    };
  });
  const edges = edgeArray(raw).map(edge => {
    const d = edge.data;
    const generic = d.relation_type === 'related_to_generic' || (d.generic_relation_review_hint && d.generic_relation_review_hint !== 'not_generic');
    return {
      data: {
        ...d,
        generic: generic ? 'yes' : 'no',
      }
    };
  });
  return [...nodes, ...edges];
}

function sliceKindLabel(kind) {
  return {
    ego_network: '实体邻域',
    community_cluster: '社区簇',
    noisy_generic_edges: '噪声 / 泛化边',
    weak_or_isolated_nodes: '弱节点 / 孤立节点',
    evidence_path: '证据路径'
  }[kind] || kind;
}

function defaultLayoutFor(slice) {
  const kind = slice.metadata.slice_kind;
  if (kind === 'evidence_path') return 'dagre';
  if (kind === 'ego_network') return 'concentric';
  if (kind === 'community_cluster') return slice.metadata.node_count <= 12 ? 'circle' : 'fcose';
  if (kind === 'noisy_generic_edges') return 'concentric';
  return 'fcose';
}

function layoutConfig(slice, requested) {
  const name = requested === 'auto' ? defaultLayoutFor(slice) : requested;
  const base = { name, animate: false, fit: true, padding: 56 };
  if (name === 'fcose') {
    return {
      ...base,
      quality: 'proof',
      randomize: false,
      nodeRepulsion: 11000,
      idealEdgeLength: 155,
      edgeElasticity: 0.35,
      gravity: 0.15,
      numIter: 2200
    };
  }
  if (name === 'dagre') {
    return {
      ...base,
      rankDir: 'LR',
      nodeSep: 105,
      edgeSep: 30,
      rankSep: 150
    };
  }
  if (name === 'concentric') {
    return {
      ...base,
      minNodeSpacing: 54,
      concentric: node => node.data('degree') || 1,
      levelWidth: () => 2
    };
  }
  if (name === 'breadthfirst') {
    const nodes = slice.elements.nodes || [];
    const rootId = nodes[0]?.data?.id;
    return {
      ...base,
      directed: true,
      spacingFactor: 1.55,
      roots: rootId ? `#${CSS.escape(rootId)}` : undefined
    };
  }
  if (name === 'circle') {
    return { ...base, spacingFactor: 1.35 };
  }
  return {
    ...base,
    name: 'cose',
    nodeRepulsion: 12000,
    idealEdgeLength: 170,
    edgeElasticity: 80,
    nestingFactor: 1.2,
    gravity: 0.18,
    numIter: 1600,
    randomize: false
  };
}

function applyEdgeLabelMode(cy) {
  const show = document.getElementById('edgeLabelToggle').checked;
  cy.style()
    .selector('edge')
    .style('label', show ? 'data(label)' : '')
    .update();
}

function relationDisplay(edge) {
  const relation = edge.relation_type || edge.label || 'related_to';
  const human = RELATION_LABELS[relation];
  return human ? `${human} (${relation})` : relation;
}

function relationHelp(edge) {
  const relation = edge.relation_type || edge.label || 'related_to';
  const raw = asList(edge.raw_relation_types).length ? `原始关系: ${asList(edge.raw_relation_types).join(', ')}` : '没有原始关系类型';
  if (relation === 'related_to_generic' || (edge.generic_relation_review_hint && edge.generic_relation_review_hint !== 'not_generic')) {
    return `泛化 / 待复检关系；${raw}；复检提示: ${edge.generic_relation_review_hint || 'generic'}`;
  }
  return `规范化关系类型: ${relationDisplay(edge)}；${raw}`;
}

function renderEdgeList(slice) {
  const labels = nodeLabelMap(slice);
  const edges = edgeArray(slice.elements);
  const edgeList = document.getElementById('edgeList');
  if (!edges.length) {
    edgeList.innerHTML = '<div class="edge-card"><div class="edge-meta">这个切片没有可见边。</div></div>';
    return;
  }
  edgeList.innerHTML = edges.map(edge => {
    const d = edge.data;
    const source = d.source_label || labels.get(d.source) || d.source;
    const target = d.target_label || labels.get(d.target) || d.target;
    const relation = d.relation_type || d.label || 'related_to';
    const generic = relation === 'related_to_generic' || (d.generic_relation_review_hint && d.generic_relation_review_hint !== 'not_generic');
    const description = d.description ? `<div class="edge-meta">${escapeText(d.description)}</div>` : '';
    const groupText = d.visual_edge_group_count && d.visual_edge_group_count > 1
      ? `；合并 ${d.visual_edge_group_count} 条原始边`
      : '';
    const quoteText = asList(d.source_text_quotes).length ? `；quote ${asList(d.source_text_quotes).length} 条` : '';
    return `<button class="edge-card ${generic ? 'generic' : ''}" data-edge-id="${escapeText(d.id)}">`
      + `<div class="edge-title">${escapeText(source)} → <strong>${escapeText(relationDisplay(d))}</strong> → ${escapeText(target)}</div>`
      + `<div class="edge-meta">${escapeText(relationHelp(d))}</div>`
      + description
      + `<div class="edge-meta">证据 ${asList(d.evidence_refs).length} 条；warning ${asList(d.warnings).length} 条${quoteText}${groupText}</div>`
      + `</button>`;
  }).join('');
}

function renderMetadata(slice) {
  const m = slice.metadata;
  document.getElementById('sliceTitle').textContent = m.title;
  document.getElementById('sliceReason').textContent = m.reason;
  document.getElementById('sliceStats').innerHTML = [
    `<span class="pill">${escapeText(sliceKindLabel(m.slice_kind))}</span>`,
    `<span class="pill">节点 ${m.node_count}</span>`,
    `<span class="pill">原始边 ${m.edge_count}</span>`,
    `<span class="pill">可视边 ${edgeArray(slice.elements).length}</span>`,
    `<span class="pill">graph_is_not_proof</span>`,
    `<span class="pill">not_checked</span>`
  ].join('');
  document.getElementById('sliceWarnings').innerHTML = (m.warnings || []).length
    ? `<p class="warn">警告：${escapeText((m.warnings || []).join('；'))}</p>`
    : '<p class="muted">警告：无</p>';
  document.getElementById('selectedDetails').innerHTML = '点击节点或边查看中文解释、证据 refs 和 warnings。';
  renderEdgeList(slice);
}

function renderField(label, value) {
  return `<dt>${escapeText(label)}</dt><dd>${escapeText(value)}</dd>`;
}

function detailSection(title, body) {
  return `<div class="detail-section"><h3>${escapeText(title)}</h3>${body}</div>`;
}

function renderNodeDetails(data) {
  const title = data.label || data.id;
  const fields = [
    renderField('节点类型', data.entity_type || 'unknown'),
    renderField('质量提示', data.entity_quality_hint || '无'),
    renderField('局部连接数', data.degree ?? '0'),
    renderField('支持状态', data.support_status || 'not_checked'),
    renderField('可写入', String(data.write_permission === true)),
    renderField('不是证明', String(data.graph_is_not_proof === true))
  ].join('');
  const description = data.description
    ? detailSection('说明', `<p>${escapeText(data.description)}</p>`)
    : '';
  return `
    <div class="detail-title">${escapeText(title)}</div>
    <div class="detail-subtitle">节点 ID：${escapeText(data.id || '')}</div>
    <dl class="kv-grid">${fields}</dl>
    ${description}
    ${detailSection('证据 refs', chipList(data.evidence_refs, 'neutral'))}
    ${detailSection('原文 quote', quoteList(data.source_text_quotes))}
    ${detailSection('原始回指 / source refs', chipList([...asList(data.raw_backpointer_refs), ...asList(data.source_refs)], 'neutral'))}
    ${detailSection('Warnings', chipList(data.warnings, asList(data.warnings).length ? 'warn' : 'neutral'))}
  `;
}

function renderEdgeDetails(data) {
  const title = `${data.source_label || data.source} → ${relationDisplay(data)} → ${data.target_label || data.target}`;
  const generic = data.relation_type === 'related_to_generic' || (data.generic_relation_review_hint && data.generic_relation_review_hint !== 'not_generic');
  const fields = [
    renderField('关系类型', relationDisplay(data)),
    renderField('原始关系', compact(data.raw_relation_types)),
    renderField('source', data.source_label || data.source || ''),
    renderField('target', data.target_label || data.target || ''),
    renderField('来源视角', compact(data.source_perspectives)),
    renderField('归因状态', compact(data.attribution_statuses)),
    renderField('置信提示', compact(data.confidence_hints)),
    renderField('证据数量', data.evidence_count ?? asList(data.evidence_refs).length),
    renderField('合并原始边', data.visual_edge_group_count || 1),
    renderField('不是证明', String(data.graph_is_not_proof === true))
  ].join('');
  const genericBody = generic
    ? detailSection('为什么需要复检', chipList([data.generic_relation_review_hint, ...asList(data.generic_relation_review_reasons)], 'danger'))
    : '';
  const description = data.description
    ? detailSection('关系说明', `<p>${escapeText(data.description)}</p>`)
    : '';
  return `
    <div class="detail-title">${escapeText(title)}</div>
    <div class="detail-subtitle">可视边 ID：${escapeText(data.id || '')}</div>
    <dl class="kv-grid">${fields}</dl>
    ${description}
    ${genericBody}
    ${detailSection('证据 refs', chipList(data.evidence_refs, 'neutral'))}
    ${detailSection('原文 quote', quoteList(data.source_text_quotes))}
    ${detailSection('原始边 / 回指 / source refs', chipList([...asList(data.source_edge_ids), ...asList(data.raw_backpointer_refs), ...asList(data.source_refs)], 'neutral'))}
    ${detailSection('时间范围', temporalChipList(data.temporal_scopes))}
    ${detailSection('Warnings', chipList(data.warnings, asList(data.warnings).length ? 'warn' : 'neutral'))}
    <details><summary>查看原始对象 JSON</summary><pre>${escapeText(JSON.stringify(data, null, 2))}</pre></details>
  `;
}

function clearHighlights(cy) {
  cy.elements().removeClass('faded focus-node focus-edge');
}

function syncEdgeCard(edgeId) {
  document.querySelectorAll('.edge-card.selected').forEach(card => card.classList.remove('selected'));
  if (!edgeId) return;
  const card = document.querySelector(`.edge-card[data-edge-id="${CSS.escape(edgeId)}"]`);
  if (!card) return;
  card.classList.add('selected');
  card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
}

function selectEdge(cy, edge, shouldCenter = false) {
  if (!edge || edge.empty()) return;
  cy.elements().unselect();
  clearHighlights(cy);
  edge.select();
  edge.addClass('focus-edge');
  edge.source().addClass('focus-node');
  edge.target().addClass('focus-node');
  cy.elements().difference(edge.union(edge.source()).union(edge.target())).addClass('faded');
  syncEdgeCard(edge.id());
  document.getElementById('selectedDetails').innerHTML = renderEdgeDetails(edge.data());
  if (shouldCenter) cy.animate({ center: { eles: edge.union(edge.source()).union(edge.target()) }, duration: 220 });
}

function selectNode(cy, node) {
  if (!node || node.empty()) return;
  cy.elements().unselect();
  clearHighlights(cy);
  node.select();
  node.addClass('focus-node');
  const neighborhood = node.closedNeighborhood();
  cy.elements().difference(neighborhood).addClass('faded');
  syncEdgeCard(null);
  document.getElementById('selectedDetails').innerHTML = renderNodeDetails(node.data());
}

function renderSlice(cy, slice) {
  renderMetadata(slice);
  cy.elements().remove();
  cy.add(normalizeElements(slice.elements));
  clearHighlights(cy);
  applyEdgeLabelMode(cy);
  const requested = document.getElementById('layoutSelect').value;
  const config = layoutConfig(slice, requested);
  try {
    cy.layout(config).run();
  } catch (error) {
    const fallback = config.name === 'dagre'
      ? { name: 'breadthfirst', directed: true, animate: false, fit: true, padding: 56 }
      : { name: 'cose', animate: false, fit: true, padding: 56, nodeRepulsion: 10000, idealEdgeLength: 150, numIter: 1000 };
    cy.layout(fallback).run();
  }
}

function init() {
  const select = document.getElementById('sliceSelect');
  SLICE_DATA.forEach((slice, index) => {
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${sliceKindLabel(slice.metadata.slice_kind)} | ${slice.metadata.title}`;
    select.appendChild(option);
  });
  if (!window.cytoscape) {
    document.getElementById('fallback').style.display = 'block';
    return;
  }
  if (window.cytoscapeFcose) cytoscape.use(window.cytoscapeFcose);
  if (window.cytoscapeDagre) cytoscape.use(window.cytoscapeDagre);

  const cy = cytoscape({
    container: document.getElementById('cy'),
    elements: [],
    style: [
      {
        selector: 'node',
        style: {
          'label': 'data(label)',
          'font-size': 11,
          'text-wrap': 'wrap',
          'text-max-width': 118,
          'background-color': '#2563eb',
          'border-width': 1,
          'border-color': '#1d4ed8',
          'width': 'mapData(degree, 0, 20, 30, 78)',
          'height': 'mapData(degree, 0, 20, 30, 78)',
          'color': '#1f2937',
          'text-valign': 'bottom',
          'text-margin-y': 7
        }
      },
      {
        selector: 'node[semanticGroup = "object"]',
        style: {
          'background-color': '#079669',
          'border-color': '#047857'
        }
      },
      {
        selector: 'node[weak = "yes"]',
        style: {
          'background-color': '#c77700',
          'border-color': '#92400e'
        }
      },
      {
        selector: 'edge',
        style: {
          'curve-style': 'bezier',
          'target-arrow-shape': 'triangle',
          'target-arrow-color': '#64748b',
          'line-color': '#94a3b8',
          'width': 2.25,
          'label': '',
          'font-size': 9,
          'text-rotation': 'autorotate',
          'text-background-color': '#ffffff',
          'text-background-opacity': 0.82,
          'text-background-padding': 2
        }
      },
      {
        selector: 'edge[generic = "yes"]',
        style: {
          'line-color': '#ef4444',
          'target-arrow-color': '#ef4444',
          'line-style': 'dashed'
        }
      },
      {
        selector: '.faded',
        style: {
          'opacity': 0.16,
          'text-opacity': 0.08
        }
      },
      {
        selector: '.focus-node',
        style: {
          'border-width': 4,
          'border-color': '#111827'
        }
      },
      {
        selector: '.focus-edge',
        style: {
          'width': 5,
          'line-color': '#111827',
          'target-arrow-color': '#111827',
          'z-index': 999
        }
      },
      {
        selector: ':selected',
        style: {
          'border-width': 4,
          'border-color': '#111827',
          'line-color': '#111827',
          'target-arrow-color': '#111827'
        }
      }
    ]
  });
  window.__graphReviewCy = cy;

  cy.on('tap', 'edge', event => selectEdge(cy, event.target, false));
  cy.on('tap', 'node', event => selectNode(cy, event.target));
  cy.on('tap', event => {
    if (event.target !== cy) return;
    cy.elements().unselect();
    clearHighlights(cy);
    syncEdgeCard(null);
    document.getElementById('selectedDetails').innerHTML = '点击节点或边查看中文解释、证据 refs 和 warnings。';
  });
  document.getElementById('edgeList').addEventListener('click', event => {
    const card = event.target.closest('.edge-card');
    if (!card || !card.dataset.edgeId) return;
    const edge = cy.getElementById(card.dataset.edgeId);
    selectEdge(cy, edge, true);
  });
  select.addEventListener('change', () => renderSlice(cy, SLICE_DATA[Number(select.value)]));
  document.getElementById('layoutSelect').addEventListener('change', () => renderSlice(cy, SLICE_DATA[Number(select.value)]));
  document.getElementById('edgeLabelToggle').addEventListener('change', () => applyEdgeLabelMode(cy));
  document.getElementById('fitBtn').addEventListener('click', () => cy.fit(undefined, 56));
  document.getElementById('layoutBtn').addEventListener('click', () => renderSlice(cy, SLICE_DATA[Number(select.value)]));
  renderSlice(cy, SLICE_DATA[0]);
}

init();
</script>
</body>
</html>
"""
    document = (
        document.replace("__SLICE_DATA__", payload_json)
        .replace("__MANIFEST_DATA__", manifest_data_json)
        .replace("__MANIFEST_JSON__", html.escape(manifest_json))
        .replace("__MANIFEST_SUMMARY_HTML__", summary_html)
    )
    write_text(path, document)


def run_visual_review_bundle(
    workspace: Path,
    *,
    graph_dir: Path | None = None,
    networkx_dir: Path | None = None,
    profile_dir: Path | None = None,
    output_dir: Path | None = None,
    projection: str = "review_aware_graph",
    top_ego_count: int = 3,
    top_community_count: int = 3,
    top_evidence_path_count: int = 3,
    max_nodes_per_slice: int = 25,
    max_edges_per_slice: int = 60,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    graph_dir = (graph_dir or workspace / DEFAULT_GRAPH_DIR_NAME).resolve()
    networkx_dir = (networkx_dir or workspace / DEFAULT_NETWORKX_DIR_NAME).resolve()
    profile_dir = (profile_dir or workspace / DEFAULT_PROFILE_DIR_NAME).resolve()
    output_dir = (output_dir or workspace / DEFAULT_OUTPUT_DIR_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    node_path = graph_dir / "graph_nodes_table.jsonl"
    edge_path = graph_dir / "graph_edges_table.jsonl"
    nodes = read_jsonl(node_path)
    edges = read_jsonl(edge_path)
    communities = read_jsonl(profile_dir / "graph_communities.jsonl")
    community_reports = read_jsonl(profile_dir / "graph_community_reports.jsonl")
    algorithm_rows = read_jsonl(networkx_dir / "graph_algorithm_rows.jsonl")

    graph, projection_stats = load_projection(projection=projection, nodes=nodes, edges=edges)

    slice_rows: list[dict[str, Any]] = []
    selected_nodes = [node_id for node_id, _degree in graph_degree_ranking(graph)[:top_ego_count]]
    for node_id in selected_nodes:
        subgraph, warnings = ego_slice(graph, node_id, max_nodes=max_nodes_per_slice, max_edges=max_edges_per_slice)
        slice_rows.append(
            write_slice(
                output_dir=output_dir,
                slice_kind="ego_network",
                title=f"ego: {node_label(graph, node_id)}",
                reason="Inspect direct relation neighborhood around a high-degree or high-activation entity.",
                graph=subgraph,
                warnings=warnings,
            )
        )

    sorted_communities = sorted(
        communities,
        key=lambda row: (-float(row.get("activation_score") or row.get("rank") or 0.0), str(row.get("community_id") or "")),
    )[:top_community_count]
    report_by_community = {str(row.get("community_id") or ""): row for row in community_reports}
    for community in sorted_communities:
        community_id = str(community.get("community_id") or "")
        report = report_by_community.get(community_id, {})
        subgraph, warnings = community_slice(
            graph,
            community,
            max_nodes=max_nodes_per_slice,
            max_edges=max_edges_per_slice,
        )
        if report.get("provider_report_status") == "community_report_candidate":
            warnings.append("provider_community_report_available_but_not_truth")
        slice_rows.append(
            write_slice(
                output_dir=output_dir,
                slice_kind="community_cluster",
                title=f"community: {community.get('title') or community_id}",
                reason="Inspect whether a community is meaningful or review-heavy/noisy.",
                graph=subgraph,
                warnings=warnings,
            )
        )

    noisy_graph, noisy_warnings = noisy_edge_slice(graph, max_edges=max_edges_per_slice)
    slice_rows.append(
        write_slice(
            output_dir=output_dir,
            slice_kind="noisy_generic_edges",
            title="generic/noisy edge review",
            reason="Inspect generic or review-heavy relation candidates before they influence retrieval too strongly.",
            graph=noisy_graph,
            warnings=noisy_warnings,
        )
    )

    weak_graph, weak_warnings = weak_node_slice(graph, max_nodes=max_nodes_per_slice, max_edges=max_edges_per_slice)
    slice_rows.append(
        write_slice(
            output_dir=output_dir,
            slice_kind="weak_or_isolated_nodes",
            title="weak or isolated node review",
            reason="Inspect weak, context-dependent, generic-fragment, or isolated nodes.",
            graph=weak_graph,
            warnings=weak_warnings,
        )
    )

    evidence_path_rows = [
        row
        for row in algorithm_rows
        if row.get("algorithm") == "shortest_evidence_path" and row.get("projection") == projection
    ][:top_evidence_path_count]
    for row in evidence_path_rows:
        subgraph, warnings = path_slice(graph, row, max_edges=max_edges_per_slice)
        slice_rows.append(
            write_slice(
                output_dir=output_dir,
                slice_kind="evidence_path",
                title=f"path: {row.get('subject_label') or row.get('subject_id')}",
                reason="Inspect the candidate evidence/navigation path used by graph algorithms or query expansion.",
                graph=subgraph,
                warnings=warnings,
            )
        )

    write_jsonl(output_dir / "graph_visual_review_slices.jsonl", slice_rows)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "graph_dir": str(graph_dir),
        "networkx_dir": str(networkx_dir),
        "profile_dir": str(profile_dir),
        "output_dir": str(output_dir),
        "projection": projection,
        "source_hashes": {
            "graph_nodes_table.jsonl": file_hash(node_path),
            "graph_edges_table.jsonl": file_hash(edge_path),
            "graph_algorithm_rows.jsonl": file_hash(networkx_dir / "graph_algorithm_rows.jsonl"),
            "graph_communities.jsonl": file_hash(profile_dir / "graph_communities.jsonl"),
            "graph_community_reports.jsonl": file_hash(profile_dir / "graph_community_reports.jsonl"),
        },
        "projection_stats": projection_stats,
        "counts": {
            "input_nodes": len(nodes),
            "input_edges": len(edges),
            "input_communities": len(communities),
            "input_community_reports": len(community_reports),
            "input_algorithm_rows": len(algorithm_rows),
            "slices": len(slice_rows),
            "slice_nodes_total": sum(int(row.get("node_count") or 0) for row in slice_rows),
            "slice_edges_total": sum(int(row.get("edge_count") or 0) for row in slice_rows),
        },
        "viewer": {
            "html_index": str(output_dir / "graph_visual_review_index.html"),
            "renderer": "Cytoscape.js CDN with GraphML/JSON fallback",
            "visualization_is_audit_support_only": True,
        },
        "borrowed_patterns": [
            "GraphRAG GraphML snapshot and Gephi review pattern",
            "LightRAG bounded label/subgraph inspection pattern",
            "NetworkX projection/export layer",
        ],
        "boundary": {
            "graph_is_not_proof": True,
            "support_status": "not_checked",
            "write_permission": False,
            "visualization_is_audit_support_only": True,
        },
    }
    write_json(output_dir / "graph_visual_review_manifest.json", manifest)
    write_report(output_dir / "graph_visual_review_report.md", manifest, slice_rows)
    write_html_index(output_dir / "graph_visual_review_index.html", manifest, slice_rows)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v0.31 graph visualization review bundles.")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--networkx-dir", type=Path, default=None)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--projection", default="review_aware_graph")
    parser.add_argument("--top-ego-count", type=int, default=3)
    parser.add_argument("--top-community-count", type=int, default=3)
    parser.add_argument("--top-evidence-path-count", type=int, default=3)
    parser.add_argument("--max-nodes-per-slice", type=int, default=25)
    parser.add_argument("--max-edges-per-slice", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_visual_review_bundle(
        args.workspace,
        graph_dir=args.graph_dir,
        networkx_dir=args.networkx_dir,
        profile_dir=args.profile_dir,
        output_dir=args.output_dir,
        projection=args.projection,
        top_ego_count=args.top_ego_count,
        top_community_count=args.top_community_count,
        top_evidence_path_count=args.top_evidence_path_count,
        max_nodes_per_slice=args.max_nodes_per_slice,
        max_edges_per_slice=args.max_edges_per_slice,
    )
    print(json.dumps({"status": "ok", "output_dir": manifest["output_dir"], "counts": manifest["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
