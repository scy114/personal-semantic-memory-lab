"""Run v0.3 graph utility smoke over candidate graph tables with NetworkX.

This runner is the first graph-algorithm proof layer after candidate
consolidation. It loads audit JSONL graph tables into NetworkX projections,
computes basic graph signals, and writes interpretation-oriented reports.

It does not write graph truth, durable memory, S3 assets, or support-checker
authority.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import networkx as nx

from tools.graph.graph_construction_packet_builder import (
    file_hash,
    read_json,
    read_jsonl,
    stable_id,
    write_json,
    write_jsonl,
    write_text,
)


SCHEMA_VERSION = "graph_v03.networkx_utility.v0.1"
ALGORITHM_ROW_SCHEMA_VERSION = "graph_v03.graph_algorithm_row.v0.1"
PROJECTION_ROW_SCHEMA_VERSION = "graph_v03.graph_projection_stats.v0.1"
DEFAULT_GRAPH_DIR_NAME = "graph_v03_consolidation_provider_80"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_networkx_utility"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            text = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def graphml_value(value: Any) -> str | int | float | bool:
    if isinstance(value, bool | int | float | str):
        return value
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def graphml_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    clean = nx.MultiDiGraph()
    clean.graph.update({key: graphml_value(value) for key, value in graph.graph.items()})
    for node_id, data in graph.nodes(data=True):
        clean.add_node(node_id, **{key: graphml_value(value) for key, value in data.items()})
    for source, target, key, data in graph.edges(keys=True, data=True):
        clean.add_edge(source, target, key=key, **{key2: graphml_value(value) for key2, value in data.items()})
    return clean


def write_graphml(path: Path, graph: nx.MultiDiGraph) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graphml_graph(graph), path)


def projection_rules(name: str) -> dict[str, Any]:
    if name == "full_candidate_graph":
        return {
            "description": "All candidate nodes and edges.",
            "include_node_quality": "all",
            "exclude_node_quality": [],
            "exclude_generic_relation_review_hint": [],
            "exclude_generic_edges": False,
        }
    if name == "review_aware_graph":
        return {
            "description": "Exclude review-required nodes and low-value generic edges.",
            "include_node_quality": "all except review_required",
            "exclude_node_quality": ["review_required"],
            "exclude_generic_relation_review_hint": ["low_graph_value"],
            "exclude_generic_edges": False,
        }
    if name == "stable_core_graph":
        return {
            "description": "Stable nodes only, non-generic edges only.",
            "include_node_quality": ["stable"],
            "exclude_node_quality": ["candidate", "generic_fragment", "context_dependent", "action_phrase", "review_required"],
            "exclude_generic_relation_review_hint": ["low_graph_value", "needs_entity_resolution", "needs_schema_extension", "needs_attribution_review", "safe_generic"],
            "exclude_generic_edges": True,
        }
    raise ValueError(f"Unknown projection: {name}")


def include_node(node: dict[str, Any], projection: str) -> bool:
    quality = str(node.get("entity_quality_hint") or "")
    if projection == "full_candidate_graph":
        return True
    if projection == "review_aware_graph":
        return quality != "review_required"
    if projection == "stable_core_graph":
        return quality == "stable"
    raise ValueError(f"Unknown projection: {projection}")


def include_edge(edge: dict[str, Any], projection: str, included_nodes: set[str]) -> bool:
    if edge.get("source_node_id") not in included_nodes or edge.get("target_node_id") not in included_nodes:
        return False
    generic_hint = str(edge.get("generic_relation_review_hint") or "")
    relation_type = str(edge.get("relation_type") or "")
    if projection == "full_candidate_graph":
        return True
    if projection == "review_aware_graph":
        return generic_hint != "low_graph_value"
    if projection == "stable_core_graph":
        return relation_type != "related_to_generic" and generic_hint == "not_generic"
    raise ValueError(f"Unknown projection: {projection}")


def load_projection(
    *,
    projection: str,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
) -> tuple[nx.MultiDiGraph, dict[str, Any]]:
    graph = nx.MultiDiGraph(
        projection=projection,
        graph_is_not_proof=True,
        support_status="not_checked",
        write_permission=False,
    )
    included_node_ids: set[str] = set()
    excluded_node_counts: Counter[str] = Counter()
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        if include_node(node, projection):
            included_node_ids.add(node_id)
            graph.add_node(
                node_id,
                label=node.get("label") or "",
                entity_type=node.get("entity_type") or "unknown",
                entity_quality_hint=node.get("entity_quality_hint") or "",
                entity_quality_reasons=string_list(node.get("entity_quality_reasons")),
                evidence_refs=string_list(node.get("evidence_refs")),
                raw_backpointer_refs=string_list(node.get("raw_backpointer_refs")),
                source_refs=string_list(node.get("source_refs")),
                source_text_quotes=string_list(node.get("source_text_quotes")),
                graph_is_not_proof=True,
                support_status="not_checked",
                write_permission=False,
                warnings=string_list(node.get("warnings")),
            )
        else:
            excluded_node_counts[str(node.get("entity_quality_hint") or "unknown")] += 1

    excluded_edge_counts: Counter[str] = Counter()
    for edge in edges:
        edge_id = str(edge.get("edge_id") or stable_id("graph_edge_projection", json.dumps(edge, sort_keys=True)))
        if not include_edge(edge, projection, included_node_ids):
            excluded_edge_counts[str(edge.get("generic_relation_review_hint") or edge.get("relation_type") or "unknown")] += 1
            continue
        graph.add_edge(
            str(edge.get("source_node_id")),
            str(edge.get("target_node_id")),
            key=edge_id,
            edge_id=edge_id,
            relation_type=edge.get("relation_type") or "",
            raw_relation_types=string_list(edge.get("raw_relation_types")),
            source_label=edge.get("source_label") or "",
            target_label=edge.get("target_label") or "",
            evidence_refs=string_list(edge.get("evidence_refs")),
            raw_backpointer_refs=string_list(edge.get("raw_backpointer_refs")),
            source_refs=string_list(edge.get("source_refs")),
            source_text_quotes=string_list(edge.get("source_text_quotes")),
            source_perspective=edge.get("source_perspective") or "",
            attribution_status=edge.get("attribution_status") or "",
            confidence_hint=edge.get("confidence_hint") or "",
            temporal_scope=edge.get("temporal_scope") or {},
            description=edge.get("description") or "",
            evidence_count=int(edge.get("evidence_count") or 0),
            weight=float(edge.get("weight") or 1.0),
            generic_relation_review_hint=edge.get("generic_relation_review_hint") or "",
            generic_relation_review_reasons=string_list(edge.get("generic_relation_review_reasons")),
            graph_is_not_proof=True,
            support_status="not_checked",
            write_permission=False,
            warnings=string_list(edge.get("warnings")),
        )

    stats = projection_stats(graph, projection, excluded_node_counts, excluded_edge_counts)
    return graph, stats


def projection_stats(
    graph: nx.MultiDiGraph,
    projection: str,
    excluded_node_counts: Counter[str],
    excluded_edge_counts: Counter[str],
) -> dict[str, Any]:
    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    isolated_nodes = list(nx.isolates(graph.to_undirected()))
    self_loop_count = nx.number_of_selfloops(graph)
    weak_components = list(nx.weakly_connected_components(graph)) if node_count else []
    density = nx.density(graph) if node_count > 1 else 0.0
    small_sparse_warning = node_count < 10 or edge_count < max(3, node_count - 1)
    return {
        "schema_version": PROJECTION_ROW_SCHEMA_VERSION,
        "projection": projection,
        "projection_rules": projection_rules(projection),
        "node_count": node_count,
        "edge_count": edge_count,
        "density": round(float(density), 6),
        "isolated_node_count": len(isolated_nodes),
        "self_loop_count": self_loop_count,
        "weak_component_count": len(weak_components),
        "largest_weak_component_size": max((len(component) for component in weak_components), default=0),
        "excluded_node_quality_counts": dict(sorted(excluded_node_counts.items())),
        "excluded_edge_hint_counts": dict(sorted(excluded_edge_counts.items())),
        "small_or_sparse_graph_warning": small_sparse_warning,
        "graph_is_not_proof": True,
        "support_status": "not_checked",
    }


def simple_weighted_digraph(graph: nx.MultiDiGraph) -> nx.DiGraph:
    simple = nx.DiGraph()
    for node_id, data in graph.nodes(data=True):
        simple.add_node(node_id, **data)
    for source, target, data in graph.edges(data=True):
        weight = float(data.get("weight") or 1.0)
        evidence_count = int(data.get("evidence_count") or 0)
        if simple.has_edge(source, target):
            simple[source][target]["weight"] += weight
            simple[source][target]["evidence_count"] += evidence_count
            simple[source][target]["edge_count"] += 1
        else:
            simple.add_edge(source, target, weight=weight, evidence_count=evidence_count, edge_count=1)
    return simple


def simple_undirected_graph(graph: nx.MultiDiGraph) -> nx.Graph:
    undirected = nx.Graph()
    for node_id, data in graph.nodes(data=True):
        undirected.add_node(node_id, **data)
    for source, target, data in graph.edges(data=True):
        weight = float(data.get("weight") or 1.0)
        evidence_count = int(data.get("evidence_count") or 0)
        if undirected.has_edge(source, target):
            undirected[source][target]["weight"] += weight
            undirected[source][target]["evidence_count"] += evidence_count
            undirected[source][target]["edge_count"] += 1
        else:
            undirected.add_edge(source, target, weight=weight, evidence_count=evidence_count, edge_count=1)
    return undirected


def node_label(graph: nx.Graph, node_id: str) -> str:
    return str(graph.nodes[node_id].get("label") or node_id)


def top_nodes(values: dict[str, float], limit: int = 10) -> list[tuple[str, float]]:
    return sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]


def edge_between_summary(graph: nx.MultiDiGraph, source: str, target: str) -> dict[str, Any]:
    edge_data = graph.get_edge_data(source, target)
    if not edge_data:
        edge_data = graph.get_edge_data(target, source)
    if not edge_data:
        return {}
    best = sorted(
        edge_data.values(),
        key=lambda data: (int(data.get("evidence_count") or 0), str(data.get("relation_type") or "")),
        reverse=True,
    )[0]
    return {
        "relation_type": best.get("relation_type") or "",
        "evidence_refs": string_list(best.get("evidence_refs"))[:5],
        "generic_relation_review_hint": best.get("generic_relation_review_hint") or "",
        "warnings": string_list(best.get("warnings"))[:5],
    }


def evidence_path(graph: nx.MultiDiGraph, path: list[str]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for source, target in zip(path, path[1:]):
        edge_summary = edge_between_summary(graph, source, target)
        segments.append(
            {
                "source_node_id": source,
                "source_label": node_label(graph, source),
                "target_node_id": target,
                "target_label": node_label(graph, target),
                **edge_summary,
            }
        )
    return segments


def row(
    *,
    projection: str,
    algorithm: str,
    subject_id: str = "",
    subject_label: str = "",
    metric_value: float | int | None = None,
    payload: dict[str, Any] | None = None,
    interpretation: str = "",
    warning: str = "",
) -> dict[str, Any]:
    return {
        "schema_version": ALGORITHM_ROW_SCHEMA_VERSION,
        "row_id": stable_id("graph_algorithm_row", json.dumps([projection, algorithm, subject_id, payload or {}, warning], ensure_ascii=False, sort_keys=True)),
        "projection": projection,
        "algorithm": algorithm,
        "subject_id": subject_id,
        "subject_label": subject_label,
        "metric_value": metric_value,
        "payload": payload or {},
        "plain_language_interpretation": interpretation,
        "failure_or_misleading_risk": warning,
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }


def degree_rows(graph: nx.MultiDiGraph, projection: str) -> list[dict[str, Any]]:
    weighted = {
        node_id: sum(float(data.get("weight") or 1.0) for _, _, data in graph.edges(node_id, data=True))
        for node_id in graph.nodes
    }
    rows: list[dict[str, Any]] = []
    for node_id, value in top_nodes(dict(graph.degree()), 10):
        rows.append(
            row(
                projection=projection,
                algorithm="degree",
                subject_id=node_id,
                subject_label=node_label(graph, node_id),
                metric_value=int(value),
                payload={"entity_quality_hint": graph.nodes[node_id].get("entity_quality_hint", "")},
                interpretation="Degree counts direct graph connections in this projection.",
                warning="High degree can reflect conversational repetition or generic nodes, not truth.",
            )
        )
    for node_id, value in top_nodes(weighted, 10):
        rows.append(
            row(
                projection=projection,
                algorithm="weighted_degree",
                subject_id=node_id,
                subject_label=node_label(graph, node_id),
                metric_value=round(float(value), 6),
                payload={"entity_quality_hint": graph.nodes[node_id].get("entity_quality_hint", "")},
                interpretation="Weighted degree sums edge weights around the node.",
                warning="Weights are conservative candidate weights, not support proof.",
            )
        )
    return rows


def pagerank_rows(graph: nx.MultiDiGraph, projection: str) -> list[dict[str, Any]]:
    if graph.number_of_nodes() < 2 or graph.number_of_edges() < 1:
        return [
            row(
                projection=projection,
                algorithm="pagerank",
                warning="Graph is too small or edgeless for meaningful PageRank.",
            )
        ]
    simple = simple_weighted_digraph(graph)
    scores = nx.pagerank(simple, weight="weight")
    return [
        row(
            projection=projection,
            algorithm="pagerank",
            subject_id=node_id,
            subject_label=node_label(graph, node_id),
            metric_value=round(float(score), 8),
            payload={"entity_quality_hint": graph.nodes[node_id].get("entity_quality_hint", "")},
            interpretation="PageRank estimates graph importance from incoming/outgoing structure in this projection.",
            warning="PageRank is a ranking signal only; it does not prove evidence support.",
        )
        for node_id, score in top_nodes(scores, 10)
    ]


def ego_rows(graph: nx.MultiDiGraph, projection: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seed_ids = [node_id for node_id, _ in top_nodes(dict(graph.degree()), 5)]
    undirected = graph.to_undirected()
    for node_id in seed_ids:
        ego = nx.ego_graph(undirected, node_id, radius=1)
        rows.append(
            row(
                projection=projection,
                algorithm="ego_network_1hop",
                subject_id=node_id,
                subject_label=node_label(graph, node_id),
                metric_value=ego.number_of_nodes(),
                payload={
                    "neighbor_count": ego.number_of_nodes() - 1,
                    "neighbors": [
                        {"node_id": other, "label": node_label(graph, other), "entity_quality_hint": graph.nodes[other].get("entity_quality_hint", "")}
                        for other in sorted(set(ego.nodes) - {node_id})[:20]
                    ],
                },
                interpretation="One-hop ego network shows immediate graph neighborhood for audit and retrieval expansion.",
                warning="Neighbors may include context-dependent or candidate-only nodes.",
            )
        )
    return rows


def shortest_path_rows(graph: nx.MultiDiGraph, projection: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    undirected = graph.to_undirected()
    candidates = [node_id for node_id, _ in top_nodes(dict(graph.degree()), 8)]
    for source, target in combinations(candidates, 2):
        if len(rows) >= 8:
            break
        if source == target:
            continue
        try:
            path = nx.shortest_path(undirected, source, target)
        except nx.NetworkXNoPath:
            continue
        if len(path) <= 1:
            continue
        rows.append(
            row(
                projection=projection,
                algorithm="shortest_evidence_path",
                subject_id=f"{source}->{target}",
                subject_label=f"{node_label(graph, source)} -> {node_label(graph, target)}",
                metric_value=len(path) - 1,
                payload={"path_node_ids": path, "path_labels": [node_label(graph, node_id) for node_id in path], "evidence_path": evidence_path(graph, path)},
                interpretation="Shortest path gives a compact candidate evidence route between two graph nodes.",
                warning="Shortest path is a retrieval/navigation signal; every segment still needs evidence inspection.",
            )
        )
    if not rows:
        rows.append(row(projection=projection, algorithm="shortest_evidence_path", warning="No connected node pairs available for shortest path."))
    return rows


def community_rows(graph: nx.MultiDiGraph, projection: str) -> list[dict[str, Any]]:
    undirected = simple_undirected_graph(graph)
    if undirected.number_of_nodes() < 10 or undirected.number_of_edges() < 9:
        return [
            row(
                projection=projection,
                algorithm="community_detection",
                warning="Graph is too small or sparse; community interpretation is limited.",
            )
        ]
    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight"))
    rows: list[dict[str, Any]] = []
    for idx, community in enumerate(sorted(communities, key=lambda item: (-len(item), sorted(item)[0])), 1):
        if idx > 10:
            break
        labels = [node_label(graph, node_id) for node_id in sorted(community)]
        rows.append(
            row(
                projection=projection,
                algorithm="community_detection",
                subject_id=f"community:{idx}",
                subject_label=f"community {idx}",
                metric_value=len(community),
                payload={"node_ids": sorted(community), "labels": labels[:30]},
                interpretation="Community detection groups densely connected candidate nodes for review.",
                warning="Community membership can be driven by repeated dialogue topics or noisy generic nodes.",
            )
        )
    return rows


def node_similarity_rows(graph: nx.MultiDiGraph, projection: str) -> list[dict[str, Any]]:
    undirected = simple_undirected_graph(graph)
    if undirected.number_of_nodes() < 3:
        return [row(projection=projection, algorithm="node_similarity", warning="Graph is too small for node similarity.")]
    nodes = [node_id for node_id, _ in top_nodes(dict(undirected.degree()), 12)]
    scores: list[tuple[str, str, float]] = []
    for source, target in combinations(nodes, 2):
        source_neighbors = set(undirected.neighbors(source))
        target_neighbors = set(undirected.neighbors(target))
        union = source_neighbors | target_neighbors
        if not union:
            continue
        score = len(source_neighbors & target_neighbors) / len(union)
        if score > 0:
            scores.append((source, target, score))
    scores.sort(key=lambda item: (-item[2], node_label(graph, item[0]), node_label(graph, item[1])))
    if not scores:
        return [row(projection=projection, algorithm="node_similarity", warning="No non-zero neighbor-overlap similarity found.")]
    return [
        row(
            projection=projection,
            algorithm="node_similarity_jaccard",
            subject_id=f"{source}<->{target}",
            subject_label=f"{node_label(graph, source)} <-> {node_label(graph, target)}",
            metric_value=round(float(score), 6),
            payload={"method": "neighbor_jaccard"},
            interpretation="Neighbor-overlap similarity suggests candidate nodes that share graph context.",
            warning="Similarity can reflect shared noisy neighbors; it is not entity equivalence proof.",
        )
        for source, target, score in scores[:10]
    ]


def run_algorithms(graph: nx.MultiDiGraph, projection: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(degree_rows(graph, projection))
    rows.extend(pagerank_rows(graph, projection))
    rows.extend(ego_rows(graph, projection))
    rows.extend(shortest_path_rows(graph, projection))
    rows.extend(community_rows(graph, projection))
    rows.extend(node_similarity_rows(graph, projection))
    return rows


def write_report(path: Path, manifest: dict[str, Any], projection_rows: list[dict[str, Any]], algorithm_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v0.3 NetworkX Graph Utility Smoke",
        "",
        f"- workspace: `{manifest['workspace_id']}`",
        f"- graph_dir: `{manifest['graph_dir']}`",
        "- graph_is_not_proof: `true`",
        "- support_status: `not_checked`",
        "",
        "## Projection Summary",
        "",
    ]
    for stats in projection_rows:
        lines.extend(
            [
                f"### {stats['projection']}",
                "",
                f"- nodes: {stats['node_count']}",
                f"- edges: {stats['edge_count']}",
                f"- density: {stats['density']}",
                f"- weak_components: {stats['weak_component_count']}",
                f"- largest_component: {stats['largest_weak_component_size']}",
                f"- isolated_nodes: {stats['isolated_node_count']}",
                f"- small_or_sparse_warning: `{str(stats['small_or_sparse_graph_warning']).lower()}`",
                f"- rules: `{json.dumps(stats['projection_rules'], ensure_ascii=False, sort_keys=True)}`",
                "",
            ]
        )

    lines.extend(["## Algorithm Interpretation", ""])
    by_projection: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_item in algorithm_rows:
        by_projection[str(row_item["projection"])].append(row_item)

    for projection, rows_for_projection in by_projection.items():
        lines.extend([f"### {projection}", ""])
        for algorithm in ["pagerank", "degree", "ego_network_1hop", "shortest_evidence_path", "community_detection", "node_similarity_jaccard", "node_similarity"]:
            selected = [item for item in rows_for_projection if item["algorithm"] == algorithm]
            if not selected:
                continue
            lines.append(f"#### {algorithm}")
            for item in selected[:8]:
                if item.get("subject_label"):
                    lines.append(f"- `{item['subject_label']}`: {item.get('metric_value')} - {item['plain_language_interpretation']}")
                else:
                    lines.append(f"- warning: {item.get('failure_or_misleading_risk')}")
            lines.append("")

    lines.extend(
        [
            "## Boundary",
            "",
            "- NetworkX metrics are ranking/retrieval signals only.",
            "- Graph metrics are not evidence support proof.",
            "- Every downstream use must preserve evidence paths or explicit missing-evidence warnings.",
            "- No durable memory, graph truth, S3 asset, or support-checker authority is written.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def run_networkx_utility(
    workspace: Path,
    *,
    graph_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    graph_dir = (graph_dir or workspace / DEFAULT_GRAPH_DIR_NAME).resolve()
    output_dir = (output_dir or workspace / DEFAULT_OUTPUT_DIR_NAME).resolve()

    node_path = graph_dir / "graph_nodes_table.jsonl"
    edge_path = graph_dir / "graph_edges_table.jsonl"
    claim_path = graph_dir / "graph_claims_table.jsonl"
    evidence_path = graph_dir / "evidence_links.jsonl"
    consolidation_manifest_path = graph_dir / "graph_consolidation_manifest.json"

    nodes = read_jsonl(node_path)
    edges = read_jsonl(edge_path)
    claims = read_jsonl(claim_path)
    evidence_links = read_jsonl(evidence_path)
    consolidation_manifest = read_json(consolidation_manifest_path) if consolidation_manifest_path.exists() else {}

    projection_names = ["full_candidate_graph", "review_aware_graph", "stable_core_graph"]
    projection_rows: list[dict[str, Any]] = []
    algorithm_rows: list[dict[str, Any]] = []
    graphml_outputs: dict[str, str] = {}

    for projection in projection_names:
        graph, stats = load_projection(projection=projection, nodes=nodes, edges=edges)
        projection_rows.append(stats)
        algorithm_rows.extend(run_algorithms(graph, projection))
        graphml_path = output_dir / f"{projection}.graphml"
        write_graphml(graphml_path, graph)
        graphml_outputs[projection] = str(graphml_path)

    outputs = {
        "graph_projection_manifest": output_dir / "graph_projection_manifest.json",
        "graph_projection_stats": output_dir / "graph_projection_stats.jsonl",
        "graph_algorithm_rows": output_dir / "graph_algorithm_rows.jsonl",
        "graph_algorithm_report": output_dir / "graph_algorithm_report.md",
    }
    write_jsonl(outputs["graph_projection_stats"], projection_rows)
    write_jsonl(outputs["graph_algorithm_rows"], algorithm_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "workspace_id": workspace.name,
        "created_at": now_iso(),
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "networkx_version": nx.__version__,
        "upstream_consolidation_schema_version": consolidation_manifest.get("schema_version", ""),
        "source_asset_hashes": {
            str(node_path): file_hash(node_path),
            str(edge_path): file_hash(edge_path),
            str(claim_path): file_hash(claim_path),
            str(evidence_path): file_hash(evidence_path),
            str(consolidation_manifest_path): file_hash(consolidation_manifest_path),
        },
        "input_counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "claims": len(claims),
            "evidence_links": len(evidence_links),
        },
        "projection_counts": {
            row["projection"]: {
                "nodes": row["node_count"],
                "edges": row["edge_count"],
                "small_or_sparse_graph_warning": row["small_or_sparse_graph_warning"],
            }
            for row in projection_rows
        },
        "policies": {
            "graph_is_not_proof": True,
            "write_permission": False,
            "support_status": "not_checked",
            "networkx_metrics_are_signals_not_proof": True,
            "claims_loaded_as_side_table_not_topology": True,
            "durable_memory_written": False,
            "graph_truth_written": False,
            "s3_written": False,
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
        "graphml_outputs": graphml_outputs,
    }
    write_json(outputs["graph_projection_manifest"], manifest)
    write_report(outputs["graph_algorithm_report"], manifest, projection_rows, algorithm_rows)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NetworkX utility smoke on v0.3 candidate graph tables.")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_networkx_utility(args.workspace, graph_dir=args.graph_dir, output_dir=args.output_dir)
    print(json.dumps({"manifest": manifest["outputs"]["graph_projection_manifest"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
