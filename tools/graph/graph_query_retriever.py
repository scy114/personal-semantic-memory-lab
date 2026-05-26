"""v0.3 graph query retriever over consolidated candidate graph tables.

This module turns graph JSONL candidate tables into query-time retrieval
signals. It follows mature graph RAG patterns at a small local scale:
lexical retrieval, graph seed matching, selected-context anchoring,
neighborhood expansion, evidence paths, and RRF fusion.

It does not produce graph truth, support proof, durable memory, or S3 assets.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from tools.graph.graph_construction_packet_builder import read_jsonl, stable_id
from tools.graph.networkx_graph_utility_runner import (
    evidence_path,
    load_projection,
    node_label,
    simple_undirected_graph,
    string_list,
)


SCHEMA_VERSION = "graph_v03.query_retriever.v0.1"
DEFAULT_PROJECTION = "review_aware_graph"
DEFAULT_GRAPH_DIR_CANDIDATES = [
    "graph_v03_consolidation_provider_80",
    "graph_v03_consolidation",
    "graph_v03_prototype",
]

# Fallback guard for leaked unresolved reference nodes. The design source is
# GraphRAG/Graphiti-style guidance: resolve pronouns/articles during extraction
# or skip them; this list is only a query-time quarantine backstop.
UNRESOLVED_REFERENCE_LABELS = {
    "i",
    "me",
    "my",
    "you",
    "he",
    "him",
    "she",
    "her",
    "we",
    "us",
    "they",
    "them",
    "it",
    "this",
    "that",
    "these",
    "those",
    "someone",
    "something",
    "somewhere",
    "a place",
    "place",
    "spot",
    "thing",
}

WEAK_ENTITY_QUALITY_HINTS = {
    "review_required",
    "context_dependent",
    "generic_fragment",
    "action_phrase",
}

PATH_RISK_WARNINGS = {
    "reference_resolution_is_weak_surface",
    "context_dependency_warning",
    "context_only_evidence_role",
    "quote_not_found_in_excerpt",
    "entity_merge_requires_review",
}

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
    "how",
    "s",
    "about",
    "关系",
    "什么",
    "哪些",
    "怎么",
    "如何",
    "这个",
    "目前",
}


def normalize_text(value: Any) -> str:
    return str(value or "").strip().lower()


def tokens(value: Any) -> list[str]:
    text = normalize_text(value)
    raw = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", text)
    out: list[str] = []
    for item in raw:
        if not item or item in STOPWORDS:
            continue
        out.append(item)
    return out


def token_score(query_tokens: set[str], text: Any) -> int:
    if not query_tokens:
        return 0
    return len(query_tokens & set(tokens(text)))


def contains_phrase(text: Any, phrase: str) -> bool:
    phrase_norm = normalize_text(phrase)
    return bool(phrase_norm and phrase_norm in normalize_text(text))


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


@dataclass
class GraphQueryData:
    graph_dir: Path
    profile_dir: Path | None
    community_assists_path: Path | None
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    community_reports: list[dict[str, Any]]
    graph: nx.MultiDiGraph
    projection_stats: dict[str, Any]


_GRAPH_UNIT_EMBEDDING_CACHE: dict[tuple[str, str, int, tuple[str, ...]], np.ndarray] = {}


def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if high <= low:
        return [0.5 for _ in values]
    return [(value - low) / (high - low) for value in values]


class BM25Lite:
    def __init__(self, docs: list[dict[str, Any]], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokens(doc.get("text", "")) for doc in docs]
        self.doc_lengths = [len(items) for items in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        df: Counter[str] = Counter()
        for items in self.doc_tokens:
            df.update(set(items))
        self.idf = {
            term: math.log(1 + (len(self.docs) - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query: str) -> list[float]:
        query_terms = tokens(query)
        scores: list[float] = []
        for items, doc_len in zip(self.doc_tokens, self.doc_lengths):
            tf = Counter(items)
            score = 0.0
            for term in query_terms:
                if term not in tf:
                    continue
                denom = tf[term] + self.k1 * (1 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                score += self.idf.get(term, 0.0) * (tf[term] * (self.k1 + 1)) / (denom or 1.0)
            scores.append(score)
        return scores


def discover_graph_dir(workspace: Path, explicit: str | Path | None) -> Path | None:
    if explicit:
        path = Path(explicit).resolve()
        return path if (path / "graph_nodes_table.jsonl").exists() and (path / "graph_edges_table.jsonl").exists() else None
    for name in DEFAULT_GRAPH_DIR_CANDIDATES:
        path = workspace / name
        if (path / "graph_nodes_table.jsonl").exists() and (path / "graph_edges_table.jsonl").exists():
            return path
    return None


def discover_profile_dir(graph_dir: Path) -> Path | None:
    candidates = [
        graph_dir.parent / "graph_v03_profile_communities",
        graph_dir / "graph_v03_profile_communities",
    ]
    for path in candidates:
        if (path / "graph_community_reports.jsonl").exists():
            return path
    return None


def read_community_assist_rows(path: Path) -> list[dict[str, Any]]:
    path = path.resolve()
    if path.is_dir():
        path = path / "graph_community_report_assists.jsonl"
    return read_jsonl(path)


def apply_community_assists(reports: list[dict[str, Any]], assists_path: Path | None) -> tuple[list[dict[str, Any]], Path | None]:
    if assists_path is None:
        return reports, None
    assist_rows = [
        row
        for row in read_community_assist_rows(assists_path)
        if row.get("provider_report_status") == "community_report_candidate" and row.get("community_id")
    ]
    if not assist_rows:
        return reports, assists_path.resolve()
    by_id = {str(row.get("community_id") or ""): row for row in assist_rows}
    out: list[dict[str, Any]] = []
    for report in reports:
        community_id = str(report.get("community_id") or "")
        assist = by_id.get(community_id)
        if not assist:
            out.append(report)
            continue
        merged = dict(report)
        merged.update(assist)
        merged["community_report_assist_applied"] = True
        merged["warnings"] = string_list(report.get("warnings")) + [
            warning for warning in string_list(assist.get("warnings")) if warning not in string_list(report.get("warnings"))
        ]
        out.append(merged)
    return out, assists_path.resolve()


def load_graph_query_data(
    graph_dir: Path,
    *,
    projection: str,
    profile_dir: Path | None = None,
    community_assists_path: Path | None = None,
) -> GraphQueryData:
    nodes = read_jsonl(graph_dir / "graph_nodes_table.jsonl")
    edges = read_jsonl(graph_dir / "graph_edges_table.jsonl")
    claims = read_jsonl(graph_dir / "graph_claims_table.jsonl")
    profile_dir = profile_dir.resolve() if profile_dir else discover_profile_dir(graph_dir)
    community_reports = read_jsonl(profile_dir / "graph_community_reports.jsonl") if profile_dir else []
    community_reports, resolved_assists_path = apply_community_assists(community_reports, community_assists_path)
    graph, projection_stats = load_projection(projection=projection, nodes=nodes, edges=edges)
    return GraphQueryData(
        graph_dir=graph_dir,
        profile_dir=profile_dir,
        community_assists_path=resolved_assists_path,
        nodes=nodes,
        edges=edges,
        claims=claims,
        community_reports=community_reports,
        graph=graph,
        projection_stats=projection_stats,
    )


def node_unit(node: dict[str, Any]) -> dict[str, Any]:
    node_id = str(node.get("node_id") or "")
    text = " ".join(
        [
            str(node.get("label") or ""),
            str(node.get("description") or ""),
            str(node.get("entity_type") or ""),
            " ".join(string_list(node.get("source_text_quotes"))),
        ]
    )
    return {
        "unit_id": f"node:{node_id}",
        "kind": "node",
        "object_id": node_id,
        "text": text,
        "payload": node,
        "evidence_refs": string_list(node.get("evidence_refs")),
        "warnings": string_list(node.get("warnings")),
    }


def edge_unit(edge: dict[str, Any]) -> dict[str, Any]:
    edge_id = str(edge.get("edge_id") or "")
    text = " ".join(
        [
            str(edge.get("source_label") or ""),
            str(edge.get("relation_type") or ""),
            str(edge.get("target_label") or ""),
            str(edge.get("description") or ""),
            " ".join(string_list(edge.get("raw_relation_types"))),
            " ".join(string_list(edge.get("source_text_quotes"))),
        ]
    )
    return {
        "unit_id": f"edge:{edge_id}",
        "kind": "edge",
        "object_id": edge_id,
        "text": text,
        "payload": edge,
        "evidence_refs": string_list(edge.get("evidence_refs")),
        "warnings": string_list(edge.get("warnings")),
    }


def claim_unit(claim: dict[str, Any]) -> dict[str, Any]:
    claim_id = str(claim.get("claim_id") or "")
    text = " ".join(
        [
            str(claim.get("subject_label") or ""),
            str(claim.get("candidate_text") or ""),
            str(claim.get("source_text_quote") or ""),
            str(claim.get("source_text_excerpt") or ""),
        ]
    )
    return {
        "unit_id": f"claim:{claim_id}",
        "kind": "claim",
        "object_id": claim_id,
        "text": text,
        "payload": claim,
        "evidence_refs": string_list(claim.get("evidence_refs")),
        "warnings": string_list(claim.get("warnings")),
    }


def community_report_unit(report: dict[str, Any]) -> dict[str, Any]:
    report_id = str(report.get("report_id") or report.get("community_id") or "")
    text = " ".join(
        [
            str(report.get("title") or ""),
            str(report.get("summary") or ""),
            str(report.get("full_content") or ""),
            str(report.get("activation_quality") or ""),
        ]
    )
    return {
        "unit_id": f"community_report:{report_id}",
        "kind": "community_report",
        "object_id": report_id,
        "text": text,
        "payload": report,
        "evidence_refs": string_list(report.get("evidence_refs")),
        "warnings": string_list(report.get("warnings")),
    }


def graph_query_units(data: GraphQueryData) -> list[dict[str, Any]]:
    units = [node_unit(row) for row in data.nodes if row.get("node_id")]
    units.extend(edge_unit(row) for row in data.edges if row.get("edge_id"))
    units.extend(claim_unit(row) for row in data.claims if row.get("claim_id"))
    return [unit for unit in units if unit.get("text")]


def community_report_branch(query: str, reports: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    units = [community_report_unit(report) for report in reports if report.get("community_id")]
    rows = bm25_branch(query, units, limit=limit)
    for row in rows:
        row["branch"] = "community_report_match"
    return rows


def seed_terms(question: dict[str, Any], selected: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for value in listify(question.get("seed_terms")):
        text = str(value or "").strip()
        if text and text not in terms:
            terms.append(text)
    query_tokens = tokens(question.get("question", ""))
    latin_tokens = [token for token in query_tokens if re.fullmatch(r"[a-z0-9_]+", token, re.I)]
    for left, right in zip(latin_tokens, latin_tokens[1:]):
        phrase = f"{left} {right}"
        if phrase not in terms:
            terms.append(phrase)
    for token in query_tokens:
        if len(token) > 1 and token not in terms:
            terms.append(token)
    for candidate in selected[:4]:
        text = str(candidate.get("text") or "").strip()
        if text and len(text) <= 80 and text not in terms:
            terms.append(text)
    return terms[:16]


def node_payload(graph: nx.MultiDiGraph, node_id: str) -> dict[str, Any]:
    data = graph.nodes[node_id]
    return {
        "node_id": node_id,
        "label": node_label(graph, node_id),
        "entity_type": data.get("entity_type", "unknown"),
        "entity_quality_hint": data.get("entity_quality_hint", ""),
        "evidence_refs": string_list(data.get("evidence_refs"))[:5],
        "warnings": string_list(data.get("warnings"))[:5],
    }


def edge_payload(graph: nx.MultiDiGraph, source: str, target: str, key: str, data: dict[str, Any], *, seed_node_id: str = "") -> dict[str, Any]:
    return {
        "edge_id": str(data.get("edge_id") or key),
        "source_node_id": source,
        "target_node_id": target,
        "source_label": data.get("source_label") or node_label(graph, source),
        "target_label": data.get("target_label") or node_label(graph, target),
        "relation_type": data.get("relation_type", ""),
        "seed_node_id": seed_node_id,
        "seed_label": node_label(graph, seed_node_id) if seed_node_id else "",
        "neighbor_node_id": target if source == seed_node_id else source,
        "neighbor_label": node_label(graph, target if source == seed_node_id else source) if seed_node_id else "",
        "evidence_refs": string_list(data.get("evidence_refs"))[:5],
        "generic_relation_review_hint": data.get("generic_relation_review_hint", ""),
        "warnings": string_list(data.get("warnings"))[:5],
    }


def path_node_quality(graph: nx.MultiDiGraph, node_id: str, *, endpoint: bool, query_seed_ids: set[str]) -> dict[str, Any]:
    data = graph.nodes[node_id]
    label = node_label(graph, node_id)
    label_norm = normalize_text(label)
    quality_hint = str(data.get("entity_quality_hint") or "")
    entity_type = str(data.get("entity_type") or "unknown")
    warnings = string_list(data.get("warnings"))
    reasons: list[str] = []

    if quality_hint in WEAK_ENTITY_QUALITY_HINTS:
        reasons.append(f"weak_entity_quality:{quality_hint}")
    if label_norm in UNRESOLVED_REFERENCE_LABELS:
        reasons.append("unresolved_reference_label")
    if entity_type == "unknown" and len(tokens(label)) <= 1 and node_id not in query_seed_ids:
        reasons.append("unknown_short_bridge_node")
    for warning in warnings:
        if warning in PATH_RISK_WARNINGS or any(marker in warning for marker in PATH_RISK_WARNINGS):
            reasons.append(f"node_warning:{warning}")

    if endpoint and node_id in query_seed_ids:
        bridge_policy = "query_seed_endpoint"
    elif reasons and endpoint:
        bridge_policy = "weak_endpoint"
    elif reasons:
        bridge_policy = "noisy_bridge"
    else:
        bridge_policy = "clean_bridge"

    return {
        "node_id": node_id,
        "label": label,
        "entity_type": entity_type,
        "entity_quality_hint": quality_hint,
        "endpoint": endpoint,
        "bridge_policy": bridge_policy,
        "risk_reasons": sorted(set(reasons)),
    }


def path_segment_quality(segment: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not string_list(segment.get("evidence_refs")):
        reasons.append("path_segment_missing_evidence_refs")
    hint = str(segment.get("generic_relation_review_hint") or "")
    if hint and hint not in {"not_generic", "safe_generic"}:
        reasons.append(f"path_segment_generic_relation_hint:{hint}")
    for warning in string_list(segment.get("warnings")):
        if warning in PATH_RISK_WARNINGS or any(marker in warning for marker in PATH_RISK_WARNINGS):
            reasons.append(f"path_segment_warning:{warning}")
    return reasons


def annotate_path_quality(graph: nx.MultiDiGraph, path: list[str], segments: list[dict[str, Any]], query_seed_ids: set[str]) -> dict[str, Any]:
    node_qualities = [
        path_node_quality(graph, node_id, endpoint=index in {0, len(path) - 1}, query_seed_ids=query_seed_ids)
        for index, node_id in enumerate(path)
    ]
    segment_reasons = [reason for segment in segments for reason in path_segment_quality(segment)]
    noisy_bridges = [item for item in node_qualities[1:-1] if item["bridge_policy"] == "noisy_bridge"]
    weak_endpoints = [item for item in node_qualities if item["bridge_policy"] == "weak_endpoint"]

    reasons: list[str] = []
    for item in noisy_bridges + weak_endpoints:
        reasons.extend(item.get("risk_reasons", []))
    reasons.extend(segment_reasons)

    if noisy_bridges:
        status = "noisy_navigation"
        use_policy = "navigation_debug_only"
    elif weak_endpoints or segment_reasons:
        status = "caution"
        use_policy = "navigation_context_only"
    else:
        status = "clean"
        use_policy = "candidate_evidence_navigation"

    return {
        "path_quality_status": status,
        "path_use_policy": use_policy,
        "path_quality_reasons": sorted(set(reasons)),
        "path_node_quality": node_qualities,
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }


def bm25_branch(query: str, units: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    bm25 = BM25Lite(units)
    scored: list[tuple[float, dict[str, Any]]] = []
    for score, unit in zip(bm25.scores(query), units):
        if score <= 0:
            continue
        scored.append((score, unit))
    rows: list[dict[str, Any]] = []
    max_score = max((score for score, _ in scored), default=1.0)
    for rank, (score, unit) in enumerate(sorted(scored, key=lambda item: (-item[0], item[1]["unit_id"]))[:limit], 1):
        rows.append(
            {
                "branch": "bm25_graph_units",
                "rank": rank,
                "raw_score": round(score, 6),
                "norm_score": round(score / max_score, 6),
                "kind": unit["kind"],
                "object_id": unit["object_id"],
                "unit_id": unit["unit_id"],
                "text": unit["text"],
                "payload": unit["payload"],
                "evidence_refs": unit["evidence_refs"][:5],
                "warnings": unit["warnings"],
            }
        )
    return rows


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    vector = vector.astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def normalize_vector_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    if matrix.size == 0:
        return matrix
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def encode_graph_unit_texts(embedder: Any, texts: list[str]) -> np.ndarray:
    if hasattr(embedder, "encode_batch"):
        return normalize_vector_matrix(np.asarray(embedder.encode_batch(texts), dtype=np.float32))
    vectors = [normalize_vector(np.asarray(embedder.encode(text), dtype=np.float32)) for text in texts]
    return np.vstack(vectors).astype(np.float32) if vectors else np.zeros((0, 0), dtype=np.float32)


def graph_unit_embedding_vectors(
    *,
    graph_dir: Path,
    projection: str,
    units: list[dict[str, Any]],
    embedder: Any,
) -> np.ndarray:
    unit_signature = tuple(f"{unit['unit_id']}:{len(str(unit.get('text') or ''))}" for unit in units)
    key = (str(graph_dir.resolve()), projection, id(embedder), unit_signature)
    cached = _GRAPH_UNIT_EMBEDDING_CACHE.get(key)
    if cached is not None:
        return cached
    vectors = encode_graph_unit_texts(embedder, [str(unit.get("text") or "") for unit in units])
    _GRAPH_UNIT_EMBEDDING_CACHE[key] = vectors
    return vectors


def embedding_branch(
    query: str,
    units: list[dict[str, Any]],
    embedder: Any,
    *,
    graph_dir: Path,
    projection: str,
    limit: int,
) -> list[dict[str, Any]]:
    if embedder is None or not units:
        return []
    query_vector = normalize_vector(np.asarray(embedder.encode(query), dtype=np.float32))
    unit_vectors = graph_unit_embedding_vectors(graph_dir=graph_dir, projection=projection, units=units, embedder=embedder)
    if unit_vectors.size == 0:
        return []
    scored: list[tuple[float, dict[str, Any]]] = []
    for score, unit in zip([float(score) for score in unit_vectors @ query_vector], units):
        if score <= 0:
            continue
        scored.append((score, unit))
    rows: list[dict[str, Any]] = []
    max_score = max((score for score, _ in scored), default=1.0)
    for rank, (score, unit) in enumerate(sorted(scored, key=lambda item: (-item[0], item[1]["unit_id"]))[:limit], 1):
        rows.append(
            {
                "branch": "embedding_graph_units",
                "rank": rank,
                "raw_score": round(score, 6),
                "norm_score": round(score / max_score, 6),
                "kind": unit["kind"],
                "object_id": unit["object_id"],
                "unit_id": unit["unit_id"],
                "text": unit["text"],
                "payload": unit["payload"],
                "evidence_refs": unit["evidence_refs"][:5],
                "warnings": unit["warnings"],
            }
        )
    return rows


def seed_node_branch(graph: nx.MultiDiGraph, terms: list[str], *, per_seed_limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for term in terms:
        term_tokens = set(tokens(term))
        scored: list[tuple[int, str]] = []
        for node_id, data in graph.nodes(data=True):
            label = str(data.get("label") or node_id)
            score = 0
            if normalize_text(label) == normalize_text(term):
                score += 20
            elif contains_phrase(label, term) or contains_phrase(term, label):
                score += 10
            score += token_score(term_tokens, label) * 3
            if score:
                scored.append((score, node_id))
        for rank, (score, node_id) in enumerate(sorted(scored, key=lambda item: (-item[0], node_label(graph, item[1])))[:per_seed_limit], 1):
            key = f"{term}|{node_id}"
            if key in seen:
                continue
            seen.add(key)
            payload = node_payload(graph, node_id)
            rows.append(
                {
                    "branch": "seed_node_match",
                    "rank": rank,
                    "seed_term": term,
                    "raw_score": score,
                    "kind": "node",
                    "object_id": node_id,
                    "unit_id": f"node:{node_id}",
                    "text": payload["label"],
                    "evidence_refs": payload["evidence_refs"],
                    "warnings": payload["warnings"],
                    **payload,
                }
            )
    return rows


def anchor_branch(graph: nx.MultiDiGraph, selected: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    graph_refs: list[str] = []
    for candidate in selected:
        for ref in string_list(candidate.get("graph_refs")):
            if ref not in graph_refs:
                graph_refs.append(ref)
    rows: list[dict[str, Any]] = []
    for ref in graph_refs:
        if ref in graph:
            payload = node_payload(graph, ref)
            rows.append(
                {
                    "branch": "selected_context_anchor",
                    "rank": len(rows) + 1,
                    "kind": "node",
                    "object_id": ref,
                    "unit_id": f"node:{ref}",
                    "text": payload["label"],
                    "evidence_refs": payload["evidence_refs"],
                    "warnings": payload["warnings"],
                    **payload,
                }
            )
        for source, target, key, data in graph.edges(keys=True, data=True):
            edge_id = str(data.get("edge_id") or key)
            if ref not in {edge_id, source, target}:
                continue
            payload = edge_payload(graph, source, target, key, data)
            rows.append(
                {
                    "branch": "selected_context_anchor",
                    "rank": len(rows) + 1,
                    "kind": "edge",
                    "object_id": edge_id,
                    "unit_id": f"edge:{edge_id}",
                    "text": f"{payload['source_label']} --{payload['relation_type']}--> {payload['target_label']}",
                    "evidence_refs": payload["evidence_refs"],
                    "warnings": payload["warnings"],
                    **payload,
                }
            )
        if len(rows) >= limit:
            break
    return rows[:limit]


def neighborhood_branch(graph: nx.MultiDiGraph, seed_node_ids: list[str], query: str, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_edges: set[str] = set()
    query_terms = set(tokens(query))
    seed_set = set(seed_node_ids)
    for seed_node_id in seed_node_ids:
        if seed_node_id not in graph:
            continue
        edge_iter = list(graph.out_edges(seed_node_id, keys=True, data=True)) + list(graph.in_edges(seed_node_id, keys=True, data=True))
        for source, target, key, data in edge_iter:
            edge_id = str(data.get("edge_id") or key)
            if edge_id in seen_edges:
                continue
            seen_edges.add(edge_id)
            other = target if source == seed_node_id else source
            payload = edge_payload(graph, source, target, key, data, seed_node_id=seed_node_id)
            raw_score = (
                token_score(query_terms, payload["source_label"]) * 2
                + token_score(query_terms, payload["target_label"]) * 2
                + token_score(query_terms, payload["relation_type"]) * 3
                + (8 if other in seed_set else 0)
                + (2 if payload.get("generic_relation_review_hint") == "not_generic" else 0)
                + min(len(payload.get("evidence_refs") or []), 3)
            )
            rows.append(
                {
                    "branch": "graph_neighborhood_bfs_1hop",
                    "rank": 0,
                    "raw_score": raw_score,
                    "kind": "edge",
                    "object_id": edge_id,
                    "unit_id": f"edge:{edge_id}",
                    "text": f"{payload['source_label']} --{payload['relation_type']}--> {payload['target_label']}",
                    "connects_seed_nodes": other in seed_set,
                    "evidence_refs": payload["evidence_refs"],
                    "warnings": payload["warnings"],
                    **payload,
                }
            )
    rows = sorted(rows, key=lambda row: (-row["raw_score"], row.get("generic_relation_review_hint") != "not_generic", row["text"]))[:limit]
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def evidence_paths(graph: nx.MultiDiGraph, seed_node_ids: list[str], *, max_paths: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    undirected = simple_undirected_graph(graph)
    clean_or_caution_paths: list[dict[str, Any]] = []
    noisy_paths: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    query_seed_ids = set(seed_node_ids)
    for index, source in enumerate(seed_node_ids):
        for target in seed_node_ids[index + 1 :]:
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
            segments = evidence_path(graph, path)
            path_row = {
                "source_node_id": source,
                "source_label": node_label(graph, source),
                "target_node_id": target,
                "target_label": node_label(graph, target),
                "path_node_ids": path,
                "path_labels": [node_label(graph, node_id) for node_id in path],
                "segments": segments,
                **annotate_path_quality(graph, path, segments, query_seed_ids),
            }
            if path_row["path_quality_status"] == "noisy_navigation":
                noisy_paths.append(path_row)
            else:
                clean_or_caution_paths.append(path_row)
            if len(clean_or_caution_paths) >= max_paths:
                return clean_or_caution_paths, noisy_paths[:max_paths]
    return clean_or_caution_paths, noisy_paths[:max_paths]


def rrf_fuse(branches: dict[str, list[dict[str, Any]]], *, limit: int, k: int = 60) -> list[dict[str, Any]]:
    scores: dict[str, float] = Counter()
    branch_hits: dict[str, list[dict[str, Any]]] = {}
    for branch_name, rows in branches.items():
        for rank, row in enumerate(rows, 1):
            unit_id = str(row.get("unit_id") or f"{row.get('kind')}:{row.get('object_id')}")
            scores[unit_id] += 1.0 / (k + rank)
            branch_hits.setdefault(unit_id, []).append({**row, "branch": branch_name, "rank": rank})
    fused: list[dict[str, Any]] = []
    for unit_id, score in scores.items():
        hits = branch_hits[unit_id]
        best = hits[0]
        payload = best.get("payload") if isinstance(best.get("payload"), dict) else best
        evidence_refs: list[str] = []
        warnings: list[str] = []
        for hit in hits:
            for ref in string_list(hit.get("evidence_refs")):
                if ref not in evidence_refs:
                    evidence_refs.append(ref)
            for warning in string_list(hit.get("warnings")):
                if warning not in warnings:
                    warnings.append(warning)
        if not evidence_refs:
            warnings.append("graph_retrieval_item_has_no_evidence_refs")
        generic_hint = str(best.get("generic_relation_review_hint") or "")
        if not generic_hint and isinstance(payload, dict):
            generic_hint = str(payload.get("generic_relation_review_hint") or "")
        if generic_hint and generic_hint != "not_generic":
            warnings.append(f"generic_relation_hint:{generic_hint}")
        fused.append(
            {
                "unit_id": unit_id,
                "kind": best.get("kind"),
                "object_id": best.get("object_id"),
                "text": best.get("text", ""),
                "rrf_score": round(float(score), 8),
                "branches": sorted({hit["branch"] for hit in hits}),
                "branch_count": len({hit["branch"] for hit in hits}),
                "evidence_refs": evidence_refs[:8],
                "warnings": warnings,
                "payload": payload,
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )
    return sorted(fused, key=lambda row: (-row["rrf_score"], -row["branch_count"], row["unit_id"]))[:limit]


def compact_graph_rerank_text(item: dict[str, Any], *, max_chars: int = 1800) -> str:
    payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
    parts = [
        f"kind: {item.get('kind')}",
        f"text: {item.get('text', '')}",
        f"branches: {', '.join(string_list(item.get('branches')))}",
        f"evidence_refs: {', '.join(string_list(item.get('evidence_refs')))}",
    ]
    for key in (
        "source_label",
        "relation_type",
        "target_label",
        "description",
        "candidate_text",
        "source_text_quote",
        "source_text_excerpt",
        "label",
        "entity_type",
        "activation_quality",
        "summary",
        "full_content",
    ):
        value = payload.get(key)
        if value:
            parts.append(f"{key}: {value}")
    warnings = string_list(item.get("warnings"))
    if warnings:
        parts.append(f"warnings: {', '.join(warnings)}")
    text = "\n".join(str(part) for part in parts if str(part).strip())
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0].rstrip() + "..."


def feature_rerank_graph_items(query: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rerank graph candidates with auditable non-model quality signals.

    This is deliberately not a relation extractor or a truth scorer. It only
    reorders already-recalled graph query units using branch agreement,
    evidence availability, warning burden, and graph-unit type.
    """

    if not items:
        return []
    rrf_norms = minmax_normalize([float(item.get("rrf_score") or 0.0) for item in items])
    query_terms = set(tokens(query))
    rows: list[dict[str, Any]] = []
    for item, rrf_norm in zip(items, rrf_norms):
        row = dict(item)
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        warnings = string_list(row.get("warnings"))
        evidence_refs = string_list(row.get("evidence_refs"))
        branches = string_list(row.get("branches"))
        kind = str(row.get("kind") or "")
        generic_hint = str(payload.get("generic_relation_review_hint") or "")
        quality_hint = str(payload.get("entity_quality_hint") or "")
        text_overlap = min(1.0, token_score(query_terms, row.get("text", "")) / 3.0) if query_terms else 0.0

        score = 0.58 * float(rrf_norm)
        reasons = [f"rrf_norm:{round(float(rrf_norm), 4)}"]
        if len(branches) > 1:
            boost = min(0.16, 0.055 * (len(branches) - 1))
            score += boost
            reasons.append(f"branch_agreement:+{round(boost, 4)}")
        if evidence_refs:
            score += 0.10
            reasons.append("has_evidence_refs:+0.1")
        else:
            score -= 0.18
            reasons.append("missing_evidence_refs:-0.18")
        if kind == "edge":
            score += 0.09
            reasons.append("relation_unit:+0.09")
        elif kind == "claim":
            score += 0.05
            reasons.append("claim_unit:+0.05")
        elif kind == "community_report":
            activation_quality = str(payload.get("activation_quality") or "")
            score += 0.075
            reasons.append("community_report_unit:+0.075")
            if activation_quality == "review_heavy":
                score -= 0.08
                reasons.append("review_heavy_community:-0.08")
            elif activation_quality == "clean_candidate":
                score += 0.04
                reasons.append("clean_candidate_community:+0.04")
        elif kind == "node":
            score -= 0.035
            reasons.append("node_only_context:-0.035")
        if text_overlap:
            boost = 0.07 * text_overlap
            score += boost
            reasons.append(f"query_term_overlap:+{round(boost, 4)}")
        if generic_hint and generic_hint != "not_generic":
            score -= 0.18
            reasons.append(f"generic_relation_review:{generic_hint}:-0.18")
        if quality_hint in WEAK_ENTITY_QUALITY_HINTS:
            score -= 0.08
            reasons.append(f"weak_entity_quality:{quality_hint}:-0.08")
        if warnings:
            penalty = min(0.14, 0.028 * len(warnings))
            score -= penalty
            reasons.append(f"warning_burden:-{round(penalty, 4)}")

        row["pre_rerank_rrf_score"] = row.get("rrf_score")
        row["feature_rerank_score"] = round(float(score), 8)
        row["feature_rerank_reasons"] = reasons
        row["graph_rerank_score"] = row["feature_rerank_score"]
        row["graph_rerank_policy"] = "feature_quality_blend_v0.1"
        row["graph_rerank_reasons"] = reasons
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("graph_rerank_score") or 0.0),
            -float(row.get("pre_rerank_rrf_score") or 0.0),
            str(row.get("unit_id") or ""),
        ),
    )


def model_rerank_graph_items(
    query: str,
    items: list[dict[str, Any]],
    reranker: Any,
    *,
    alpha: float = 0.5,
    batch_size: int = 16,
) -> list[dict[str, Any]]:
    if not items or reranker is None:
        return items
    pairs = [[query, compact_graph_rerank_text(item)] for item in items]
    predict_kwargs = {"batch_size": batch_size, "show_progress_bar": False}
    try:
        raw_scores = reranker.predict(pairs, **predict_kwargs)
    except TypeError:
        raw_scores = reranker.predict(pairs)
    rerank_norms = minmax_normalize([float(score) for score in raw_scores])
    feature_norms = minmax_normalize([float(item.get("feature_rerank_score", item.get("rrf_score", 0.0)) or 0.0) for item in items])
    rows: list[dict[str, Any]] = []
    for item, raw_score, rerank_norm, feature_norm in zip(items, raw_scores, rerank_norms, feature_norms):
        row = dict(item)
        blend = float(alpha) * float(rerank_norm) + (1.0 - float(alpha)) * float(feature_norm)
        payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
        quality_guard_reasons: list[str] = []
        generic_hint = str(payload.get("generic_relation_review_hint") or "")
        if generic_hint and generic_hint != "not_generic":
            blend -= 0.18
            quality_guard_reasons.append(f"quality_guard_generic_relation:{generic_hint}:-0.18")
        if not string_list(row.get("evidence_refs")):
            blend -= 0.16
            quality_guard_reasons.append("quality_guard_missing_evidence_refs:-0.16")
        warning_count = len(string_list(row.get("warnings")))
        if warning_count:
            penalty = min(0.12, 0.025 * warning_count)
            blend -= penalty
            quality_guard_reasons.append(f"quality_guard_warning_burden:-{round(penalty, 4)}")
        row["cross_encoder_rerank_score"] = round(float(raw_score), 8)
        row["cross_encoder_rerank_norm"] = round(float(rerank_norm), 8)
        row["feature_rerank_norm"] = round(float(feature_norm), 8)
        row["graph_rerank_alpha"] = float(alpha)
        row["graph_rerank_score"] = round(blend, 8)
        row["graph_rerank_policy"] = "cross_encoder_feature_blend_v0.1"
        row["graph_rerank_reasons"] = string_list(row.get("feature_rerank_reasons")) + [
            f"cross_encoder_norm:{round(float(rerank_norm), 4)}",
            f"blend_alpha:{float(alpha)}",
        ] + quality_guard_reasons
        rows.append(row)
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("graph_rerank_score") or 0.0),
            -float(row.get("cross_encoder_rerank_norm") or 0.0),
            -float(row.get("pre_rerank_rrf_score") or 0.0),
            str(row.get("unit_id") or ""),
        ),
    )


def rerank_graph_items(
    query: str,
    items: list[dict[str, Any]],
    *,
    mode: str = "feature",
    reranker: Any | None = None,
    alpha: float = 0.5,
    batch_size: int = 16,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if mode == "disabled":
        return items, {"mode": "disabled", "used": False}
    feature_rows = feature_rerank_graph_items(query, items)
    if mode == "cross_encoder" or (mode == "auto" and reranker is not None):
        if reranker is None:
            return feature_rows, {
                "mode": mode,
                "used": True,
                "fallback": "feature_quality_blend_v0.1",
                "warnings": ["cross_encoder_reranker_not_available"],
            }
        return model_rerank_graph_items(query, feature_rows, reranker, alpha=alpha, batch_size=batch_size), {
            "mode": mode,
            "used": True,
            "model_blend": "cross_encoder_feature_blend_v0.1",
            "alpha": alpha,
            "batch_size": batch_size,
        }
    return feature_rows, {"mode": mode, "used": True, "model_blend": "feature_quality_blend_v0.1"}


def query_expansion_terms(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in rows:
        payload = row.get("payload", {})
        for value in [payload.get("neighbor_label"), payload.get("relation_type"), payload.get("target_label"), payload.get("source_label")]:
            text = str(value or "").strip()
            if text:
                counts[text] += 1
    return [{"term": term, "count": count} for term, count in counts.most_common(limit)]


def graph_item_rank_score(item: dict[str, Any]) -> float:
    return float(item.get("graph_rerank_score", item.get("rrf_score", 0.0)) or 0.0)


def community_context_row(item: dict[str, Any]) -> dict[str, Any]:
    payload = item.get("payload", {}) if isinstance(item.get("payload"), dict) else {}
    return {
        "graph_context_ref": stable_id("graphctx:v03community", str(item.get("unit_id") or item.get("object_id") or "")),
        "kind": "v03_ranked_community_report",
        "source_node_refs": string_list(payload.get("source_node_refs")),
        "source_edge_refs": string_list(payload.get("source_edge_refs")),
        "summary": payload.get("summary") or item.get("text") or "",
        "community_id": payload.get("community_id") or item.get("object_id"),
        "community_title": payload.get("title") or "",
        "activation_quality": payload.get("activation_quality") or "",
        "evidence_refs": string_list(item.get("evidence_refs")),
        "confidence": "community_report_candidate",
        "warnings": ["v03_community_report_is_context_routing_material_not_proof"] + string_list(item.get("warnings")),
        "retrieval_source": "v03_graph_query_retriever",
        "retrieval_rank_score": item.get("graph_rerank_score", item.get("rrf_score", item.get("norm_score"))),
        "pre_rerank_rrf_score": item.get("pre_rerank_rrf_score", item.get("rrf_score")),
        "graph_rerank_policy": item.get("graph_rerank_policy", "community_activation_branch"),
        "graph_rerank_reasons": string_list(item.get("graph_rerank_reasons")),
        "retrieval_branches": item.get("branches", [item.get("branch")]) if item.get("branches") else [item.get("branch")],
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }


def graph_context_from_ranked_items(
    ranked_items: list[dict[str, Any]],
    paths: list[dict[str, Any]],
    *,
    max_items: int,
    activated_communities: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for item in (activated_communities or [])[:2]:
        rows.append(community_context_row(item))

    relation_first_items = sorted(
        ranked_items,
        key=lambda item: (
            item.get("kind") != "edge",
            item.get("kind") != "claim",
            -graph_item_rank_score(item),
            str(item.get("unit_id") or ""),
        ),
    )
    for item in relation_first_items:
        payload = item.get("payload", {})
        kind = item.get("kind")
        if kind == "edge":
            source_label = payload.get("source_label") or payload.get("source_node_id") or ""
            target_label = payload.get("target_label") or payload.get("target_node_id") or ""
            relation_type = payload.get("relation_type") or "related_to"
            description = str(payload.get("description") or "").strip()
            summary = f"{source_label} --{relation_type}--> {target_label}"
            if description:
                summary = f"{summary}: {description}"
            rows.append(
                {
                    "graph_context_ref": stable_id("graphctx:v03ranked", item["unit_id"]),
                    "kind": "v03_ranked_relation",
                    "source_node_refs": [x for x in [payload.get("source_node_id"), payload.get("target_node_id")] if x],
                    "source_edge_refs": [str(payload.get("edge_id") or item.get("object_id"))],
                    "summary": summary,
                    "evidence_refs": string_list(item.get("evidence_refs")),
                    "confidence": payload.get("generic_relation_review_hint") or "candidate",
                    "warnings": ["v03_graph_retrieval_candidate_not_proof"] + string_list(item.get("warnings")),
                    "retrieval_source": "v03_graph_query_retriever",
                    "retrieval_rank_score": item.get("graph_rerank_score", item.get("rrf_score")),
                    "pre_rerank_rrf_score": item.get("pre_rerank_rrf_score", item.get("rrf_score")),
                    "graph_rerank_policy": item.get("graph_rerank_policy", "disabled"),
                    "graph_rerank_reasons": string_list(item.get("graph_rerank_reasons")),
                    "retrieval_branches": item.get("branches", []),
                    "graph_is_not_proof": True,
                    "support_status": "not_checked",
                    "write_permission": False,
                }
            )
        elif kind == "node":
            rows.append(
                {
                    "graph_context_ref": stable_id("graphctx:v03ranked", item["unit_id"]),
                    "kind": "v03_ranked_node",
                    "source_node_refs": [str(payload.get("node_id") or item.get("object_id"))],
                    "source_edge_refs": [],
                    "summary": item.get("text") or str(payload.get("label") or item.get("object_id")),
                    "evidence_refs": string_list(item.get("evidence_refs")),
                    "confidence": payload.get("entity_quality_hint") or "candidate",
                    "warnings": ["v03_graph_retrieval_candidate_not_proof"] + string_list(item.get("warnings")),
                    "retrieval_source": "v03_graph_query_retriever",
                    "retrieval_rank_score": item.get("graph_rerank_score", item.get("rrf_score")),
                    "pre_rerank_rrf_score": item.get("pre_rerank_rrf_score", item.get("rrf_score")),
                    "graph_rerank_policy": item.get("graph_rerank_policy", "disabled"),
                    "graph_rerank_reasons": string_list(item.get("graph_rerank_reasons")),
                    "retrieval_branches": item.get("branches", []),
                    "graph_is_not_proof": True,
                    "support_status": "not_checked",
                    "write_permission": False,
                }
            )
        elif kind == "community_report":
            rows.append(community_context_row(item))

    for path_item in paths:
        evidence_refs: list[str] = []
        edge_refs: list[str] = []
        for segment in path_item.get("segments", []):
            for ref in string_list(segment.get("edge_refs")):
                if ref not in edge_refs:
                    edge_refs.append(ref)
            for ref in string_list(segment.get("evidence_refs")):
                if ref not in evidence_refs:
                    evidence_refs.append(ref)
        rows.append(
            {
                "graph_context_ref": stable_id("graphctx:v03path", "|".join(path_item.get("path_node_ids", []))),
                "kind": "v03_evidence_path",
                "source_node_refs": string_list(path_item.get("path_node_ids")),
                "source_edge_refs": edge_refs,
                "summary": " -> ".join(str(x) for x in path_item.get("path_labels", [])),
                "evidence_refs": evidence_refs,
                "confidence": "candidate_path",
                "warnings": ["v03_evidence_path_is_navigation_not_support_proof"] + string_list(path_item.get("path_quality_reasons")),
                "retrieval_source": "v03_graph_query_retriever",
                "path_quality_status": path_item.get("path_quality_status", "clean"),
                "path_use_policy": path_item.get("path_use_policy", "candidate_evidence_navigation"),
                "path_node_quality": path_item.get("path_node_quality", []),
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        ref = row["graph_context_ref"]
        if ref in seen or not row.get("summary"):
            continue
        seen.add(ref)
        out.append(row)
    return out[:max_items]


def retrieve_graph_query_package(
    question: dict[str, Any],
    selected: list[dict[str, Any]],
    *,
    graph_dir: Path,
    profile_dir: Path | None = None,
    community_assists_path: Path | None = None,
    projection: str = DEFAULT_PROJECTION,
    top_k: int = 12,
    graph_unit_embedder: Any | None = None,
    graph_rerank_mode: str = "feature",
    graph_reranker: Any | None = None,
    graph_rerank_alpha: float = 0.5,
    graph_reranker_batch_size: int = 16,
    graph_community_mode: str = "auto",
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    data = load_graph_query_data(
        graph_dir,
        projection=projection,
        profile_dir=profile_dir,
        community_assists_path=community_assists_path,
    )
    units = graph_query_units(data)
    terms = seed_terms(question, selected)
    query_text = str(question.get("question") or "")
    bm25_rows = bm25_branch(query_text, units, limit=max(top_k * 3, top_k))
    embedding_rows = (
        embedding_branch(
            query_text,
            units,
            graph_unit_embedder,
            graph_dir=graph_dir,
            projection=projection,
            limit=max(top_k * 3, top_k),
        )
        if graph_unit_embedder is not None
        else []
    )
    seed_rows = seed_node_branch(data.graph, terms, per_seed_limit=3)
    anchor_rows = anchor_branch(data.graph, selected, limit=max(top_k * 2, top_k))
    community_enabled = graph_community_mode == "auto"
    community_rows = (
        community_report_branch(query_text, data.community_reports, limit=max(top_k * 2, top_k))
        if community_enabled and data.community_reports
        else []
    )
    seed_node_ids = []
    for row in seed_rows + anchor_rows:
        if row.get("kind") == "node" and row.get("object_id") not in seed_node_ids:
            seed_node_ids.append(str(row["object_id"]))
    neighborhood_rows = neighborhood_branch(data.graph, seed_node_ids, query_text, limit=max(top_k * 3, top_k))
    paths, noisy_paths = evidence_paths(data.graph, seed_node_ids, max_paths=top_k)
    branches = {
        "bm25_graph_units": bm25_rows,
        "seed_node_match": seed_rows,
        "selected_context_anchor": anchor_rows,
        "graph_neighborhood_bfs_1hop": neighborhood_rows,
    }
    if embedding_rows:
        branches["embedding_graph_units"] = embedding_rows
    if community_rows:
        branches["community_report_match"] = community_rows
    ranked_pool = rrf_fuse(branches, limit=max(top_k * 3, top_k))
    reranked_pool, rerank_status = rerank_graph_items(
        query_text,
        ranked_pool,
        mode=graph_rerank_mode,
        reranker=graph_reranker,
        alpha=graph_rerank_alpha,
        batch_size=graph_reranker_batch_size,
    )
    ranked_items = reranked_pool[:top_k]
    activated_communities = community_rows[:2]
    context = graph_context_from_ranked_items(ranked_items, paths, max_items=top_k, activated_communities=activated_communities)
    warnings: list[str] = []
    if not ranked_items:
        warnings.append("no_ranked_graph_items")
    if not seed_rows and not anchor_rows:
        warnings.append("no_seed_or_anchor_nodes")
    if data.projection_stats.get("small_or_sparse_graph_warning"):
        warnings.append("small_or_sparse_graph_limits_retrieval_reliability")
    package = {
        "schema_version": SCHEMA_VERSION,
        "query_id": question.get("query_id", ""),
        "query": query_text,
        "graph_dir": str(graph_dir),
        "projection": projection,
        "query_spec": {
            "query_id": question.get("query_id", ""),
            "query_type": question.get("route") or "s2_query",
            "query": query_text,
            "seed_terms": terms,
            "selected_context_anchor_count": len(anchor_rows),
        },
        "retrieval_policy": {
            "pattern_sources": ["GraphRAG local search", "Graphiti hybrid search", "LightRAG local/global/mix"],
            "fusion": "reciprocal_rank_fusion",
            "rerank": rerank_status,
            "branches": list(branches.keys()),
            "embedding_branch": "live_graph_unit_embeddings" if graph_unit_embedder is not None else "disabled_or_not_available",
            "community_report_branch": "enabled" if community_rows else ("disabled" if not community_enabled else "unavailable"),
            "graph_community_mode": graph_community_mode,
            "profile_dir": str(data.profile_dir) if data.profile_dir else None,
            "community_assists_path": str(data.community_assists_path) if data.community_assists_path else None,
            "graph_is_not_proof": True,
        },
        "branches": branches,
        "ranked_pool_before_rerank": ranked_pool,
        "ranked_items": ranked_items,
        "graph_branch": {
            "method": "bm25_seed_anchor_neighborhood_rrf_with_optional_rerank",
            "projection": projection,
            "matched_seed_nodes": [row for row in seed_rows if row.get("kind") == "node"],
            "expanded_neighbors": neighborhood_rows,
            "evidence_paths": paths,
            "noisy_navigation_paths": noisy_paths,
            "query_expansion_terms": query_expansion_terms(ranked_items, limit=12),
            "activated_communities": activated_communities,
            "ranked_items": ranked_items,
            "warnings": warnings + ["graph_branch_is_candidate_retrieval_signal_not_support_proof"],
        },
        "projection_stats": data.projection_stats,
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }
    status = {
        "mode": "v03_graph_query_retriever",
        "status": "available",
        "graph_dir": str(graph_dir),
        "projection": projection,
        "context_count": len(context),
        "ranked_item_count": len(ranked_items),
        "warnings": warnings,
    }
    return package, context, status
