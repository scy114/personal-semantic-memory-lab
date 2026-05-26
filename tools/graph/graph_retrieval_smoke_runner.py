"""Run v0.3 graph-aware retrieval smoke over candidate graph tables.

This runner checks whether the candidate graph can help retrieval before any
backend decision. It compares a simple lexical branch with a graph-neighborhood
branch, and writes retrieval packages for manual audit.

It does not produce final answers, support proof, graph truth, durable memory,
or S3 assets.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
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
from tools.graph.networkx_graph_utility_runner import (
    evidence_path,
    load_projection,
    node_label,
    simple_undirected_graph,
    string_list,
)


SCHEMA_VERSION = "graph_v03.retrieval_smoke.v0.1"
RETRIEVAL_ROW_SCHEMA_VERSION = "graph_v03.graph_retrieval_smoke_row.v0.1"
DEFAULT_GRAPH_DIR_NAME = "graph_v03_consolidation_provider_80"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_retrieval_smoke"
DEFAULT_PROJECTION = "review_aware_graph"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "or",
    "the",
    "to",
    "with",
    "what",
    "which",
    "who",
    "怎么",
    "什么",
    "哪些",
    "关系",
}

DEFAULT_QUERIES = [
    {
        "query_id": "entity_jon_dance_studio",
        "query_type": "entity_centered",
        "query": "Jon 和 dance studio 有什么关系？",
        "seed_terms": ["Jon", "dance studio"],
        "expected_retrieval_role": "Find direct and indirect candidate relations between a person and a project/place-like node.",
    },
    {
        "query_id": "project_dance_studio_status",
        "query_type": "project_centered",
        "query": "dance studio 这个项目目前有哪些计划、约束、支持和选址相关信息？",
        "seed_terms": ["dance studio", "plan", "support", "constraint", "location", "store", "business"],
        "expected_retrieval_role": "Expand a project node into planning, support, constraint, and location neighborhoods.",
    },
    {
        "query_id": "update_festival_choreography_performance",
        "query_type": "update_event_centered",
        "query": "Jon 和 festival / choreography / performance 这条线有什么关系？",
        "seed_terms": ["Jon", "festival", "choreography", "performance"],
        "expected_retrieval_role": "Inspect event/update paths that may connect a person to performance-related entities.",
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def tokens(value: Any) -> list[str]:
    text = normalize_text(value)
    raw = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text)
    return [item for item in raw if item and item not in STOPWORDS]


def token_score(query_tokens: set[str], text: Any) -> int:
    if not query_tokens:
        return 0
    text_tokens = set(tokens(text))
    return len(query_tokens & text_tokens)


def contains_phrase(text: Any, phrase: str) -> bool:
    phrase_norm = normalize_text(phrase)
    if not phrase_norm:
        return False
    return phrase_norm in normalize_text(text)


def row_id(query_id: str, payload: dict[str, Any]) -> str:
    return stable_id("graph_retrieval_smoke", json.dumps([query_id, payload], ensure_ascii=False, sort_keys=True))


def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id", ""),
        "label": node.get("label", ""),
        "entity_type": node.get("entity_type", ""),
        "entity_quality_hint": node.get("entity_quality_hint", ""),
        "evidence_refs": string_list(node.get("evidence_refs"))[:5],
        "warnings": string_list(node.get("warnings"))[:5],
    }


def compact_edge(edge: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_id": edge.get("edge_id", ""),
        "source_node_id": edge.get("source_node_id", ""),
        "source_label": edge.get("source_label", ""),
        "relation_type": edge.get("relation_type", ""),
        "target_node_id": edge.get("target_node_id", ""),
        "target_label": edge.get("target_label", ""),
        "evidence_refs": string_list(edge.get("evidence_refs"))[:5],
        "source_text_quotes": string_list(edge.get("source_text_quotes"))[:3],
        "generic_relation_review_hint": edge.get("generic_relation_review_hint", ""),
        "warnings": string_list(edge.get("warnings"))[:5],
    }


def compact_claim(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "claim_id": claim.get("claim_id", ""),
        "subject_node_id": claim.get("subject_node_id", ""),
        "subject_label": claim.get("subject_label", ""),
        "candidate_text": claim.get("candidate_text", ""),
        "evidence_refs": string_list(claim.get("evidence_refs"))[:5],
        "source_text_quote": claim.get("source_text_quote", ""),
        "warnings": string_list(claim.get("warnings"))[:5],
    }


def lexical_branch(
    query: str,
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    claims: list[dict[str, Any]],
    limit: int,
) -> dict[str, Any]:
    query_tokens = set(tokens(query))
    node_hits: list[tuple[int, dict[str, Any]]] = []
    for node in nodes:
        score = (
            token_score(query_tokens, node.get("label")) * 5
            + token_score(query_tokens, node.get("description")) * 2
            + token_score(query_tokens, " ".join(string_list(node.get("source_text_quotes")))) * 2
            + token_score(query_tokens, " ".join(string_list(node.get("warnings"))))
        )
        if score:
            node_hits.append((score, compact_node(node)))

    edge_hits: list[tuple[int, dict[str, Any]]] = []
    for edge in edges:
        text = " ".join(
            [
                str(edge.get("source_label") or ""),
                str(edge.get("target_label") or ""),
                str(edge.get("relation_type") or ""),
                str(edge.get("description") or ""),
                " ".join(string_list(edge.get("raw_relation_types"))),
                " ".join(string_list(edge.get("source_text_quotes"))),
            ]
        )
        score = token_score(query_tokens, text) * 3 + token_score(query_tokens, edge.get("relation_type")) * 2
        if score:
            edge_hits.append((score, compact_edge(edge)))

    claim_hits: list[tuple[int, dict[str, Any]]] = []
    for claim in claims:
        text = " ".join(
            [
                str(claim.get("subject_label") or ""),
                str(claim.get("candidate_text") or ""),
                str(claim.get("source_text_quote") or ""),
                str(claim.get("source_text_excerpt") or ""),
            ]
        )
        score = token_score(query_tokens, text) * 2
        if score:
            claim_hits.append((score, compact_claim(claim)))

    def sort_hits(items: list[tuple[int, dict[str, Any]]]) -> list[dict[str, Any]]:
        return [
            {"score": score, **payload}
            for score, payload in sorted(items, key=lambda item: (-item[0], json.dumps(item[1], ensure_ascii=False)))[:limit]
        ]

    return {
        "method": "simple_lexical_token_match",
        "query_tokens": sorted(query_tokens),
        "top_nodes": sort_hits(node_hits),
        "top_edges": sort_hits(edge_hits),
        "top_claims": sort_hits(claim_hits),
        "warnings": ["lexical_branch_is_keyword_baseline_not_semantic_answer"],
    }


def graph_seed_matches(
    graph: nx.MultiDiGraph,
    seed_terms: list[str],
    *,
    per_seed_limit: int,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for seed in seed_terms:
        exact_candidates: list[tuple[int, str]] = []
        fallback_candidates: list[tuple[int, str]] = []
        seed_tokens = set(tokens(seed))
        for node_id, data in graph.nodes(data=True):
            label = data.get("label") or node_id
            if normalize_text(label) == normalize_text(seed):
                exact_candidates.append((10, node_id))
                continue
            score = 0
            if contains_phrase(label, seed) or contains_phrase(seed, str(label)):
                score += 6
            score += token_score(seed_tokens, label) * 2
            if score:
                fallback_candidates.append((score, node_id))
        candidates = exact_candidates or fallback_candidates
        for score, node_id in sorted(candidates, key=lambda item: (-item[0], node_label(graph, item[1])))[:per_seed_limit]:
            if node_id in seen:
                continue
            seen.add(node_id)
            matches.append(
                {
                    "seed_term": seed,
                    "score": score,
                    "node_id": node_id,
                    "label": node_label(graph, node_id),
                    "entity_type": graph.nodes[node_id].get("entity_type", ""),
                    "entity_quality_hint": graph.nodes[node_id].get("entity_quality_hint", ""),
                    "evidence_refs": string_list(graph.nodes[node_id].get("evidence_refs"))[:5],
                    "warnings": string_list(graph.nodes[node_id].get("warnings"))[:5],
                }
            )
    return matches


def neighborhood_rows(
    graph: nx.MultiDiGraph,
    seed_node_ids: list[str],
    *,
    query_text: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_edges: set[str] = set()
    query_tokens = set(tokens(query_text))
    seed_node_set = set(seed_node_ids)
    for seed_node_id in seed_node_ids:
        if seed_node_id not in graph:
            continue
        for source, target, key, data in graph.out_edges(seed_node_id, keys=True, data=True):
            edge_id = str(data.get("edge_id") or key)
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            relation_type = data.get("relation_type", "")
            neighbor_label = node_label(graph, target)
            retrieval_score = (
                token_score(query_tokens, neighbor_label) * 4
                + token_score(query_tokens, relation_type) * 3
                + (10 if target in seed_node_set else 0)
            )
            rows.append(
                {
                    "direction": "outgoing",
                    "seed_node_id": seed_node_id,
                    "seed_label": node_label(graph, seed_node_id),
                    "neighbor_node_id": target,
                    "neighbor_label": neighbor_label,
                    "relation_type": relation_type,
                    "edge_id": edge_id,
                    "retrieval_score": retrieval_score,
                    "connects_seed_nodes": target in seed_node_set,
                    "evidence_refs": string_list(data.get("evidence_refs"))[:5],
                    "generic_relation_review_hint": data.get("generic_relation_review_hint", ""),
                    "warnings": string_list(data.get("warnings"))[:5],
                }
            )
        for source, target, key, data in graph.in_edges(seed_node_id, keys=True, data=True):
            edge_id = str(data.get("edge_id") or key)
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            relation_type = data.get("relation_type", "")
            neighbor_label = node_label(graph, source)
            retrieval_score = (
                token_score(query_tokens, neighbor_label) * 4
                + token_score(query_tokens, relation_type) * 3
                + (10 if source in seed_node_set else 0)
            )
            rows.append(
                {
                    "direction": "incoming",
                    "seed_node_id": seed_node_id,
                    "seed_label": node_label(graph, seed_node_id),
                    "neighbor_node_id": source,
                    "neighbor_label": neighbor_label,
                    "relation_type": relation_type,
                    "edge_id": edge_id,
                    "retrieval_score": retrieval_score,
                    "connects_seed_nodes": source in seed_node_set,
                    "evidence_refs": string_list(data.get("evidence_refs"))[:5],
                    "generic_relation_review_hint": data.get("generic_relation_review_hint", ""),
                    "warnings": string_list(data.get("warnings"))[:5],
                }
            )

    return sorted(
        rows,
        key=lambda item: (
            not item["connects_seed_nodes"],
            -item["retrieval_score"],
            item["generic_relation_review_hint"] != "not_generic",
            item["seed_label"],
            item["relation_type"],
            item["neighbor_label"],
        ),
    )[:limit]


def evidence_paths_between_seeds(graph: nx.MultiDiGraph, seed_node_ids: list[str], *, max_paths: int) -> list[dict[str, Any]]:
    undirected = simple_undirected_graph(graph)
    paths: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for source_index, source in enumerate(seed_node_ids):
        for target in seed_node_ids[source_index + 1 :]:
            pair = tuple(sorted([source, target]))
            if pair in seen_pairs or source not in undirected or target not in undirected:
                continue
            seen_pairs.add(pair)
            try:
                path = nx.shortest_path(undirected, source=source, target=target)
            except nx.NetworkXNoPath:
                continue
            if len(path) > 5:
                continue
            paths.append(
                {
                    "source_node_id": source,
                    "source_label": node_label(graph, source),
                    "target_node_id": target,
                    "target_label": node_label(graph, target),
                    "path_node_ids": path,
                    "path_labels": [node_label(graph, node_id) for node_id in path],
                    "segments": evidence_path(graph, path),
                }
            )
            if len(paths) >= max_paths:
                return paths
    return paths


def query_expansion_terms(neighborhood: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for item in neighborhood:
        for value in [item.get("neighbor_label"), item.get("relation_type")]:
            text = str(value or "").strip()
            if text:
                counts[text] += 1
    return [{"term": term, "count": count} for term, count in counts.most_common(limit)]


def graph_branch(
    query_spec: dict[str, Any],
    *,
    graph: nx.MultiDiGraph,
    seed_limit: int,
    neighbor_limit: int,
    path_limit: int,
) -> dict[str, Any]:
    seed_matches = graph_seed_matches(graph, string_list(query_spec.get("seed_terms")), per_seed_limit=seed_limit)
    seed_node_ids = [item["node_id"] for item in seed_matches]
    neighborhood = neighborhood_rows(graph, seed_node_ids, query_text=str(query_spec.get("query") or ""), limit=neighbor_limit)
    paths = evidence_paths_between_seeds(graph, seed_node_ids, max_paths=path_limit)
    warnings: list[str] = []
    if not seed_matches:
        warnings.append("no_seed_node_matched")
    if not neighborhood:
        warnings.append("no_graph_neighborhood_found")
    if not paths and len(seed_node_ids) > 1:
        warnings.append("no_short_evidence_path_between_seed_nodes")
    return {
        "method": "networkx_seed_neighborhood_and_shortest_evidence_path",
        "projection": str(graph.graph.get("projection") or ""),
        "matched_seed_nodes": seed_matches,
        "expanded_neighbors": neighborhood,
        "evidence_paths": paths,
        "query_expansion_terms": query_expansion_terms(neighborhood, limit=12),
        "warnings": warnings + ["graph_branch_is_candidate_retrieval_signal_not_support_proof"],
    }


def retrieval_row(
    query_spec: dict[str, Any],
    *,
    lexical: dict[str, Any],
    graph_result: dict[str, Any],
) -> dict[str, Any]:
    payload = {"query_id": query_spec["query_id"], "projection": graph_result.get("projection", "")}
    return {
        "schema_version": RETRIEVAL_ROW_SCHEMA_VERSION,
        "row_id": row_id(query_spec["query_id"], payload),
        "query_id": query_spec["query_id"],
        "query_type": query_spec["query_type"],
        "query": query_spec["query"],
        "seed_terms": string_list(query_spec.get("seed_terms")),
        "expected_retrieval_role": query_spec.get("expected_retrieval_role", ""),
        "lexical_branch": lexical,
        "graph_branch": graph_result,
        "retrieval_impact_summary": retrieval_impact_summary(lexical, graph_result),
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }


def retrieval_impact_summary(lexical: dict[str, Any], graph_result: dict[str, Any]) -> str:
    lexical_edge_count = len(lexical.get("top_edges") or [])
    graph_neighbor_count = len(graph_result.get("expanded_neighbors") or [])
    graph_path_count = len(graph_result.get("evidence_paths") or [])
    if graph_path_count:
        return "Graph branch adds auditable evidence paths between matched seed nodes."
    if graph_neighbor_count > lexical_edge_count:
        return "Graph branch expands the query into a broader relation neighborhood than lexical edge hits."
    if graph_neighbor_count:
        return "Graph branch complements lexical hits with typed neighboring relations."
    return "Graph branch did not add useful neighborhood evidence for this query."


def load_queries(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return DEFAULT_QUERIES
    rows = read_jsonl(path)
    return rows if rows else DEFAULT_QUERIES


def write_report(path: Path, manifest: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v0.3 Graph-Aware Retrieval Smoke",
        "",
        f"- workspace: `{manifest['workspace_id']}`",
        f"- graph_dir: `{manifest['graph_dir']}`",
        f"- projection: `{manifest['projection']}`",
        "- graph_is_not_proof: `true`",
        "- support_status: `not_checked`",
        "",
        "## 怎么读",
        "",
        "这份报告不是最终回答。它只检查图分支能不能把查询扩展成可审计的候选证据包。",
        "",
        "- `lexical_branch`: 关键词基线，看看不用图时能捞到什么。",
        "- `graph_branch.matched_seed_nodes`: 查询词在候选图里命中的点。",
        "- `graph_branch.expanded_neighbors`: 从命中点展开的一跳关系邻域。",
        "- `graph_branch.evidence_paths`: 命中点之间的最短候选证据路径。",
        "- `query_expansion_terms`: 图分支建议带入检索上下文的关系词/邻居词。",
        "",
        "## Query Results",
        "",
    ]
    for row in rows:
        graph_result = row["graph_branch"]
        lexical = row["lexical_branch"]
        lines.extend(
            [
                f"### {row['query_id']}",
                "",
                f"- query: {row['query']}",
                f"- type: `{row['query_type']}`",
                f"- retrieval impact: {row['retrieval_impact_summary']}",
                f"- lexical top nodes: {len(lexical.get('top_nodes') or [])}",
                f"- lexical top edges: {len(lexical.get('top_edges') or [])}",
                f"- matched seed nodes: {len(graph_result.get('matched_seed_nodes') or [])}",
                f"- expanded neighbors: {len(graph_result.get('expanded_neighbors') or [])}",
                f"- evidence paths: {len(graph_result.get('evidence_paths') or [])}",
                "",
            ]
        )
        if graph_result.get("matched_seed_nodes"):
            lines.append("Seed nodes:")
            for item in graph_result["matched_seed_nodes"][:8]:
                lines.append(f"- `{item['label']}` ({item['entity_type']}, {item['entity_quality_hint']})")
            lines.append("")
        if graph_result.get("expanded_neighbors"):
            lines.append("Top graph neighbors:")
            for item in graph_result["expanded_neighbors"][:8]:
                lines.append(
                    f"- `{item['seed_label']}` --{item['relation_type']}--> `{item['neighbor_label']}` "
                    f"[{', '.join(item.get('evidence_refs') or [])}]"
                )
            lines.append("")
        if graph_result.get("evidence_paths"):
            lines.append("Evidence paths:")
            for item in graph_result["evidence_paths"][:5]:
                lines.append(f"- {' -> '.join(item['path_labels'])}")
            lines.append("")
        if graph_result.get("warnings"):
            lines.append("Warnings:")
            for warning in graph_result["warnings"]:
                lines.append(f"- `{warning}`")
            lines.append("")

    lines.extend(
        [
            "## Boundary",
            "",
            "- Graph retrieval rows are candidate retrieval packages, not answers.",
            "- Evidence paths are navigation aids, not support proof.",
            "- Graph metrics and graph neighborhoods remain signals until a support checker exists.",
            "- No durable memory, graph truth, S3 asset, or support-checker authority is written.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def run_graph_retrieval_smoke(
    workspace: Path,
    *,
    graph_dir: Path | None = None,
    output_dir: Path | None = None,
    queries_path: Path | None = None,
    projection: str = DEFAULT_PROJECTION,
    limit: int = 10,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    graph_dir = (graph_dir or workspace / DEFAULT_GRAPH_DIR_NAME).resolve()
    output_dir = (output_dir or workspace / DEFAULT_OUTPUT_DIR_NAME).resolve()

    node_path = graph_dir / "graph_nodes_table.jsonl"
    edge_path = graph_dir / "graph_edges_table.jsonl"
    claim_path = graph_dir / "graph_claims_table.jsonl"
    evidence_path_file = graph_dir / "evidence_links.jsonl"
    consolidation_manifest_path = graph_dir / "graph_consolidation_manifest.json"

    nodes = read_jsonl(node_path)
    edges = read_jsonl(edge_path)
    claims = read_jsonl(claim_path)
    evidence_links = read_jsonl(evidence_path_file)
    consolidation_manifest = read_json(consolidation_manifest_path) if consolidation_manifest_path.exists() else {}
    graph, projection_stats = load_projection(projection=projection, nodes=nodes, edges=edges)

    rows: list[dict[str, Any]] = []
    for query_spec in load_queries(queries_path):
        lexical = lexical_branch(query_spec.get("query", ""), nodes=nodes, edges=edges, claims=claims, limit=limit)
        graph_result = graph_branch(
            query_spec,
            graph=graph,
            seed_limit=3,
            neighbor_limit=limit * 2,
            path_limit=limit,
        )
        rows.append(retrieval_row(query_spec, lexical=lexical, graph_result=graph_result))

    outputs = {
        "graph_retrieval_manifest": output_dir / "graph_retrieval_manifest.json",
        "graph_retrieval_smoke": output_dir / "graph_retrieval_smoke.jsonl",
        "graph_retrieval_smoke_report": output_dir / "graph_retrieval_smoke_report.md",
    }
    write_jsonl(outputs["graph_retrieval_smoke"], rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "workspace_id": workspace.name,
        "created_at": now_iso(),
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "projection": projection,
        "upstream_consolidation_schema_version": consolidation_manifest.get("schema_version", ""),
        "source_asset_hashes": {
            str(node_path): file_hash(node_path),
            str(edge_path): file_hash(edge_path),
            str(claim_path): file_hash(claim_path),
            str(evidence_path_file): file_hash(evidence_path_file),
            str(consolidation_manifest_path): file_hash(consolidation_manifest_path),
        },
        "input_counts": {
            "nodes": len(nodes),
            "edges": len(edges),
            "claims": len(claims),
            "evidence_links": len(evidence_links),
            "queries": len(rows),
        },
        "projection_stats": projection_stats,
        "policies": {
            "graph_is_not_proof": True,
            "write_permission": False,
            "support_status": "not_checked",
            "retrieval_smoke_not_final_answer": True,
            "lexical_branch_is_baseline": True,
            "graph_branch_is_candidate_signal": True,
            "durable_memory_written": False,
            "graph_truth_written": False,
            "s3_written": False,
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["graph_retrieval_manifest"], manifest)
    write_report(outputs["graph_retrieval_smoke_report"], manifest, rows)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run graph-aware retrieval smoke on v0.3 candidate graph tables.")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--graph-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--queries", type=Path, default=None)
    parser.add_argument("--projection", default=DEFAULT_PROJECTION)
    parser.add_argument("--limit", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_graph_retrieval_smoke(
        args.workspace,
        graph_dir=args.graph_dir,
        output_dir=args.output_dir,
        queries_path=args.queries,
        projection=args.projection,
        limit=args.limit,
    )
    print(json.dumps({"manifest": manifest["outputs"]["graph_retrieval_manifest"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
