"""Final v0.3 candidate graph quality gate reporter.

This is an audit/reporting tool. It does not modify graph tables, write graph
truth, or run support checking. It reads consolidated candidate graph tables and
surfaces whether the current graph is ready to be used as retrieval/algorithm
material.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None and str(item)]
    text = str(value)
    return [text] if text else []


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def warning_contains(row: dict[str, Any], *needles: str) -> bool:
    text = " ".join(string_list(row.get("warnings"))).lower()
    return any(needle.lower() in text for needle in needles)


def row_label(row: dict[str, Any]) -> str:
    if row.get("source_label") and row.get("target_label"):
        return f"{row.get('source_label')} --{row.get('relation_type')}--> {row.get('target_label')}"
    return str(row.get("label") or row.get("candidate_text") or row.get("summary") or row.get("node_id") or row.get("edge_id") or "")


def sample_row(row: dict[str, Any], category: str) -> dict[str, Any]:
    return {
        "category": category,
        "id": row.get("edge_id") or row.get("node_id") or row.get("claim_id") or row.get("decision_id"),
        "label": row_label(row),
        "relation_type": row.get("relation_type"),
        "raw_relation_types": string_list(row.get("raw_relation_types")),
        "entity_quality_hint": row.get("entity_quality_hint"),
        "generic_relation_review_hint": row.get("generic_relation_review_hint"),
        "evidence_refs": string_list(row.get("evidence_refs"))[:5],
        "raw_backpointer_refs": string_list(row.get("raw_backpointer_refs"))[:5],
        "source_text_quotes": string_list(row.get("source_text_quotes") or row.get("source_text_quote"))[:3],
        "warnings": string_list(row.get("warnings"))[:10],
        "graph_is_not_proof": row.get("graph_is_not_proof"),
        "support_status": row.get("support_status"),
        "write_permission": row.get("write_permission"),
    }


def top_degree_nodes(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], limit: int = 12) -> list[dict[str, Any]]:
    degrees: Counter[str] = Counter()
    for edge in edges:
        source = str(edge.get("source_node_id") or "")
        target = str(edge.get("target_node_id") or "")
        if source:
            degrees[source] += 1
        if target:
            degrees[target] += 1
    by_id = {str(node.get("node_id")): node for node in nodes}
    rows: list[dict[str, Any]] = []
    for node_id, degree in degrees.most_common(limit):
        node = by_id.get(node_id, {})
        rows.append(
            {
                "node_id": node_id,
                "label": node.get("label", ""),
                "entity_type": node.get("entity_type", ""),
                "entity_quality_hint": node.get("entity_quality_hint", ""),
                "degree": degree,
                "warnings": string_list(node.get("warnings"))[:6],
            }
        )
    return rows


def compute_quality_gate(graph_dir: Path, *, sample_limit: int = 12) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    nodes = read_jsonl(graph_dir / "graph_nodes_table.jsonl")
    edges = read_jsonl(graph_dir / "graph_edges_table.jsonl")
    claims = read_jsonl(graph_dir / "graph_claims_table.jsonl")
    evidence_links = read_jsonl(graph_dir / "evidence_links.jsonl")
    merge_decisions = read_jsonl(graph_dir / "graph_merge_decisions.jsonl")
    manifest = read_json(graph_dir / "graph_consolidation_manifest.json", default={})

    node_ids = {str(node.get("node_id")) for node in nodes if node.get("node_id")}
    edge_count = len(edges)
    node_count = len(nodes)
    claim_count = len(claims)

    relation_type_counts = Counter(str(edge.get("relation_type") or "unknown") for edge in edges)
    raw_relation_type_counts: Counter[str] = Counter()
    for edge in edges:
        raw_relation_type_counts.update(string_list(edge.get("raw_relation_types")))
    generic_review_counts = Counter(str(edge.get("generic_relation_review_hint") or "unknown") for edge in edges)
    entity_quality_counts = Counter(str(node.get("entity_quality_hint") or "unknown") for node in nodes)
    merge_decision_counts = Counter(str(row.get("decision") or "unknown") for row in merge_decisions)

    unresolved_edges = [
        edge
        for edge in edges
        if str(edge.get("source_node_id") or "") not in node_ids or str(edge.get("target_node_id") or "") not in node_ids
    ]
    unresolved_claims = [claim for claim in claims if claim.get("subject_node_id") and str(claim.get("subject_node_id")) not in node_ids]
    missing_evidence_edges = [edge for edge in edges if not string_list(edge.get("evidence_refs"))]
    missing_evidence_claims = [claim for claim in claims if not string_list(claim.get("evidence_refs"))]
    proof_violations = [
        row
        for row in [*nodes, *edges, *claims]
        if row.get("graph_is_not_proof") is not True or row.get("write_permission") is not False or row.get("support_status") != "not_checked"
    ]

    generic_edges = [edge for edge in edges if edge.get("relation_type") == "related_to_generic"]
    generic_ratio = len(generic_edges) / edge_count if edge_count else 0.0
    review_edge_rows = [edge for edge in edges if str(edge.get("generic_relation_review_hint") or "") != "not_generic"]
    quote_warning_edges = [edge for edge in edges if warning_contains(edge, "quote_not_found", "quote_from_context", "context_dependency")]
    quote_warning_claims = [claim for claim in claims if warning_contains(claim, "quote_not_found", "quote_from_context", "context_dependency")]
    review_nodes = [node for node in nodes if str(node.get("entity_quality_hint") or "") in {"review_required", "generic_fragment", "action_phrase", "context_dependent"}]
    unstable_node_ratio = len(review_nodes) / node_count if node_count else 0.0

    hard_blockers: list[str] = []
    warnings: list[str] = []
    if unresolved_edges:
        hard_blockers.append(f"unresolved edge endpoints: {len(unresolved_edges)}")
    if unresolved_claims:
        hard_blockers.append(f"unresolved claim subjects: {len(unresolved_claims)}")
    if missing_evidence_edges or missing_evidence_claims:
        hard_blockers.append(f"missing evidence rows: {len(missing_evidence_edges) + len(missing_evidence_claims)}")
    if proof_violations:
        hard_blockers.append(f"graph boundary violations: {len(proof_violations)}")
    if generic_ratio > 0.20:
        warnings.append(f"generic relation ratio is high: {generic_ratio:.3f}")
    if unstable_node_ratio > 0.50:
        warnings.append(f"candidate/review-heavy node ratio is high: {unstable_node_ratio:.3f}")
    if quote_warning_edges or quote_warning_claims:
        warnings.append(f"context/quote warnings require prompt/report separation: {len(quote_warning_edges)} edges, {len(quote_warning_claims)} claims")
    if not edge_count or not node_count:
        hard_blockers.append("empty candidate graph")

    if hard_blockers:
        status = "blocked"
    elif warnings:
        status = "pass_with_known_warnings"
    else:
        status = "pass"

    samples: list[dict[str, Any]] = []
    samples.extend(sample_row(row, "generic_relation") for row in generic_edges[:sample_limit])
    samples.extend(sample_row(row, "review_edge") for row in review_edge_rows[:sample_limit])
    samples.extend(sample_row(row, "quote_warning_edge") for row in quote_warning_edges[:sample_limit])
    samples.extend(sample_row(row, "quote_warning_claim") for row in quote_warning_claims[:sample_limit])
    samples.extend(sample_row(row, "review_or_weak_node") for row in review_nodes[:sample_limit])
    samples.extend(sample_row(row, "unresolved_or_missing_evidence") for row in [*unresolved_edges, *unresolved_claims, *missing_evidence_edges, *missing_evidence_claims][:sample_limit])

    summary = {
        "schema_version": "graph_v03.final_quality_gate.v0.1",
        "created_at": utc_now(),
        "graph_dir": str(graph_dir),
        "status": status,
        "closure_recommendation": "v0.3 can close as graph utility checkpoint" if status != "blocked" else "do not close v0.3 until blockers are fixed",
        "hard_blockers": hard_blockers,
        "warnings": warnings,
        "counts": {
            "node_count": node_count,
            "edge_count": edge_count,
            "claim_count": claim_count,
            "evidence_link_count": len(evidence_links),
            "merge_decision_count": len(merge_decisions),
            "unresolved_edge_endpoint_count": len(unresolved_edges),
            "unresolved_claim_subject_count": len(unresolved_claims),
            "missing_evidence_edge_count": len(missing_evidence_edges),
            "missing_evidence_claim_count": len(missing_evidence_claims),
            "graph_boundary_violation_count": len(proof_violations),
            "generic_relation_count": len(generic_edges),
            "generic_relation_ratio": round(generic_ratio, 6),
            "review_edge_count": len(review_edge_rows),
            "quote_warning_edge_count": len(quote_warning_edges),
            "quote_warning_claim_count": len(quote_warning_claims),
            "review_or_weak_node_count": len(review_nodes),
            "review_or_weak_node_ratio": round(unstable_node_ratio, 6),
        },
        "distributions": {
            "relation_type_counts": counter_dict(relation_type_counts),
            "raw_relation_type_top20": dict(raw_relation_type_counts.most_common(20)),
            "generic_relation_review_counts": counter_dict(generic_review_counts),
            "entity_quality_counts": counter_dict(entity_quality_counts),
            "merge_decision_counts": counter_dict(merge_decision_counts),
        },
        "top_degree_nodes": top_degree_nodes(nodes, edges),
        "manifest_policy": manifest.get("policies", {}),
        "source_asset_hashes": manifest.get("source_asset_hashes", {}),
        "interpretation": {
            "graph_is_not_proof": True,
            "graph_metrics_are_retrieval_signals_only": True,
            "context_quote_warnings_are_not_auto_failures": True,
            "weak_or_review_nodes_should_be_quarantined_or_downranked_not_deleted": True,
            "closure_scope": "candidate graph construction, graph algorithms, graph-aware retrieval, query integration, and lightweight prompt packaging",
            "non_goals": [
                "graph truth",
                "durable memory write",
                "S3 support checker authority",
                "production graph backend decision",
                "final semantic quality proof",
            ],
        },
    }
    return summary, samples, render_report(summary)


def render_report(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    distributions = summary["distributions"]
    blocker_lines = [f"- {item}" for item in summary["hard_blockers"]] or ["- none"]
    warning_lines = [f"- {item}" for item in summary["warnings"]] or ["- none"]
    relation_lines = [f"- `{key}`: {value}" for key, value in distributions["relation_type_counts"].items()]
    generic_lines = [f"- `{key}`: {value}" for key, value in distributions["generic_relation_review_counts"].items()]
    entity_lines = [f"- `{key}`: {value}" for key, value in distributions["entity_quality_counts"].items()]
    top_node_lines = [
        f"- {row['label']} ({row['entity_quality_hint']}): degree={row['degree']}"
        for row in summary["top_degree_nodes"][:10]
    ] or ["- none"]
    return "\n".join(
        [
            "# v0.3 Final Graph Quality Gate",
            "",
            f"Status: `{summary['status']}`",
            "",
            f"Recommendation: {summary['closure_recommendation']}.",
            "",
            "## Scope",
            "",
            "This report audits the consolidated v0.3 candidate graph before closure. It is not a support checker and does not mark graph facts as truth.",
            "",
            "## Core Counts",
            "",
            f"- nodes: {counts['node_count']}",
            f"- edges: {counts['edge_count']}",
            f"- claims: {counts['claim_count']}",
            f"- evidence_links: {counts['evidence_link_count']}",
            f"- merge_decisions: {counts['merge_decision_count']}",
            f"- unresolved_edge_endpoint_count: {counts['unresolved_edge_endpoint_count']}",
            f"- unresolved_claim_subject_count: {counts['unresolved_claim_subject_count']}",
            f"- missing_evidence_rows: {counts['missing_evidence_edge_count'] + counts['missing_evidence_claim_count']}",
            f"- graph_boundary_violation_count: {counts['graph_boundary_violation_count']}",
            f"- generic_relation_ratio: {counts['generic_relation_ratio']}",
            f"- review_or_weak_node_ratio: {counts['review_or_weak_node_ratio']}",
            "",
            "## Hard Blockers",
            "",
            *blocker_lines,
            "",
            "## Known Warnings",
            "",
            *warning_lines,
            "",
            "## Relation Type Coverage",
            "",
            *relation_lines,
            "",
            "## Generic Relation Review Hints",
            "",
            *generic_lines,
            "",
            "## Entity Quality Hints",
            "",
            *entity_lines,
            "",
            "## Top Degree Nodes",
            "",
            *top_node_lines,
            "",
            "## Interpretation",
            "",
            "- Candidate graph rows remain evidence-bound retrieval material.",
            "- `graph_is_not_proof=true`, `support_status=not_checked`, and `write_permission=false` remain mandatory boundaries.",
            "- Context quote warnings are acceptable for GraphRAG-style context envelopes only when they stay visible in audit views and are not treated as primary support.",
            "- Review-heavy/generic/action-phrase nodes should be downranked, quarantined, or inspected; they should not be silently deleted from the evidence trail.",
            "- A passing quality gate closes v0.3 as graph utility workflow closure, not as final semantic truth.",
            "",
            "## Sample Artifacts",
            "",
            "- `graph_final_quality_gate_samples.jsonl` contains examples for generic relations, review edges, quote warnings, weak/review nodes, and blockers if any.",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit consolidated v0.3 candidate graph quality before closure.")
    parser.add_argument("--graph-dir", required=True, help="Directory containing graph_nodes_table.jsonl and graph_edges_table.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Directory for final quality gate artifacts.")
    parser.add_argument("--sample-limit", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    summary, samples, report = compute_quality_gate(graph_dir, sample_limit=args.sample_limit)
    write_json(output_dir / "graph_final_quality_gate_summary.json", summary)
    write_jsonl(output_dir / "graph_final_quality_gate_samples.jsonl", samples)
    write_text(output_dir / "graph_final_quality_gate_report.md", report)
    print(str(output_dir / "graph_final_quality_gate_report.md"))


if __name__ == "__main__":
    main()
