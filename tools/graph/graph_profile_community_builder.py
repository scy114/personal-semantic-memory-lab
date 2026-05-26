"""Build v0.3 graph profile cards and community reports.

This is the first high-level graph memory layer after candidate graph
consolidation. It borrows the mature GraphRAG separation of communities and
community reports, while keeping this project's evidence discipline:
profiles/reports are routing and context material, not graph truth.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from tools.proposals.proposal_runner import (
    SUPPORTED_API_MODES,
    PromptPolicy,
    build_provider,
    estimate_tokens,
    file_hash as proposal_file_hash,
    load_dotenv,
    resolve_live_api,
)
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
    simple_undirected_graph,
    string_list,
)


SCHEMA_VERSION = "graph_v03.profile_community_builder.v0.1"
ENTITY_PROFILE_SCHEMA_VERSION = "graph_v03.entity_profile_card.v0.1"
RELATION_PROFILE_SCHEMA_VERSION = "graph_v03.relation_profile_card.v0.1"
COMMUNITY_SCHEMA_VERSION = "graph_v03.community.v0.1"
COMMUNITY_REPORT_SCHEMA_VERSION = "graph_v03.community_report.v0.1"
DEFAULT_GRAPH_DIR_NAME = "graph_v03_consolidation_provider_80"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_profile_communities"
DEFAULT_COMMUNITY_REPORT_PROMPT = "configs/prompts/graph_community_report_prompt.v0.3.md"
SUPPORTED_REPORT_PROVIDERS = {"extractive", "external_jsonl", "openai"}
WEAK_ENTITY_QUALITY_HINTS = {"context_dependent", "generic_fragment", "action_phrase", "review_required"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def uniq(values: list[Any], *, limit: int | None = None) -> list[str]:
    out: list[str] = []
    for value in values:
        for item in string_list(value):
            if item not in out:
                out.append(item)
                if limit is not None and len(out) >= limit:
                    return out
    return out


def edge_key(edge: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(edge.get("source_node_id") or ""),
        str(edge.get("target_node_id") or ""),
        str(edge.get("relation_type") or "related_to_generic"),
    )


def edge_summary(edge: dict[str, Any]) -> str:
    source = str(edge.get("source_label") or edge.get("source_node_id") or "")
    relation = str(edge.get("relation_type") or "related_to")
    target = str(edge.get("target_label") or edge.get("target_node_id") or "")
    description = str(edge.get("description") or "").strip()
    text = f"{source} --{relation}--> {target}"
    return f"{text}: {description}" if description else text


def strip_json_code_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def read_jsonl_index(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        key = str(row.get("proposal_input_id") or row.get("community_report_input_id") or row.get("community_id") or "")
        if key:
            rows[key] = row
    return rows


def parse_provider_json(text: str) -> tuple[dict[str, Any] | None, list[str]]:
    stripped = strip_json_code_fence(text)
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        return None, [f"provider_output_json_parse_failed:{exc.msg}"]
    if not isinstance(value, dict):
        return None, ["provider_output_must_be_json_object"]
    return value, []


def node_profile_cards(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_node_edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        for node_id in [edge.get("source_node_id"), edge.get("target_node_id")]:
            if node_id:
                by_node_edges[str(node_id)].append(edge)

    rows: list[dict[str, Any]] = []
    for node in sorted(nodes, key=lambda row: str(row.get("node_id") or "")):
        node_id = str(node.get("node_id") or "")
        if not node_id:
            continue
        related_edges = by_node_edges.get(node_id, [])
        relation_counts = Counter(str(edge.get("relation_type") or "related_to_generic") for edge in related_edges)
        evidence_refs = uniq([node.get("evidence_refs")] + [edge.get("evidence_refs") for edge in related_edges], limit=16)
        warnings = uniq([node.get("warnings")] + [edge.get("warnings") for edge in related_edges], limit=20)
        profile_text = " ".join(
            part
            for part in [
                str(node.get("label") or ""),
                str(node.get("entity_type") or ""),
                str(node.get("description") or ""),
                " ".join(edge_summary(edge) for edge in related_edges[:6]),
            ]
            if part
        )
        rows.append(
            {
                "schema_version": ENTITY_PROFILE_SCHEMA_VERSION,
                "profile_id": stable_id("graph_entity_profile", node_id),
                "node_id": node_id,
                "label": node.get("label") or "",
                "entity_type": node.get("entity_type") or "unknown",
                "entity_quality_hint": node.get("entity_quality_hint") or "",
                "description": node.get("description") or "",
                "profile_text": profile_text,
                "degree_hint": len(related_edges),
                "relation_type_counts": dict(sorted(relation_counts.items())),
                "top_relation_summaries": [edge_summary(edge) for edge in related_edges[:8]],
                "evidence_refs": evidence_refs,
                "raw_backpointer_refs": uniq([node.get("raw_backpointer_refs")] + [edge.get("raw_backpointer_refs") for edge in related_edges], limit=16),
                "source_node_refs": [node_id],
                "source_edge_refs": uniq([[edge.get("edge_id")] for edge in related_edges], limit=20),
                "warnings": warnings,
                "activation_role": "entity_profile_candidate",
                "profile_is_not_evidence_truth": True,
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )
    return rows


def relation_profile_cards(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        key = edge_key(edge)
        if key[0] and key[1]:
            grouped[key].append(edge)

    rows: list[dict[str, Any]] = []
    for (source_id, target_id, relation_type), items in sorted(grouped.items()):
        best = sorted(
            items,
            key=lambda row: (
                str(row.get("generic_relation_review_hint") or "") != "not_generic",
                -int(row.get("evidence_count") or 0),
                str(row.get("edge_id") or ""),
            ),
        )[0]
        generic_counts = Counter(str(row.get("generic_relation_review_hint") or "") for row in items)
        rows.append(
            {
                "schema_version": RELATION_PROFILE_SCHEMA_VERSION,
                "profile_id": stable_id("graph_relation_profile", json.dumps([source_id, target_id, relation_type], ensure_ascii=False)),
                "source_node_id": source_id,
                "target_node_id": target_id,
                "source_label": best.get("source_label") or "",
                "target_label": best.get("target_label") or "",
                "relation_type": relation_type,
                "relation_count": len(items),
                "summary": edge_summary(best),
                "relation_summaries": [edge_summary(edge) for edge in items[:8]],
                "generic_relation_review_hint_counts": dict(sorted(generic_counts.items())),
                "evidence_refs": uniq([edge.get("evidence_refs") for edge in items], limit=20),
                "raw_backpointer_refs": uniq([edge.get("raw_backpointer_refs") for edge in items], limit=20),
                "source_edge_refs": uniq([[edge.get("edge_id")] for edge in items], limit=20),
                "warnings": uniq([edge.get("warnings") for edge in items], limit=20),
                "activation_role": "relation_profile_candidate",
                "profile_is_not_evidence_truth": True,
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )
    return rows


def detect_communities(graph: nx.MultiDiGraph) -> list[set[str]]:
    undirected = simple_undirected_graph(graph)
    if undirected.number_of_nodes() == 0:
        return []
    if undirected.number_of_edges() == 0:
        return [{node_id} for node_id in sorted(undirected.nodes)]
    communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight"))
    if not communities:
        communities = [set(component) for component in nx.connected_components(undirected)]
    return [set(community) for community in communities]


def community_edges(graph: nx.MultiDiGraph, node_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, target, key, data in graph.edges(keys=True, data=True):
        if source in node_ids and target in node_ids:
            row = dict(data)
            row["source_node_id"] = source
            row["target_node_id"] = target
            row["edge_id"] = str(data.get("edge_id") or key)
            rows.append(row)
    return rows


def community_title(graph: nx.MultiDiGraph, node_ids: set[str], edges: list[dict[str, Any]]) -> str:
    degree_order = sorted(node_ids, key=lambda node_id: (-graph.degree(node_id), node_label(graph, node_id)))
    labels = [node_label(graph, node_id) for node_id in degree_order[:4]]
    relation_types = [str(edge.get("relation_type") or "") for edge in edges]
    common_relation = Counter(relation_types).most_common(1)
    suffix = f" / {common_relation[0][0]}" if common_relation and common_relation[0][0] else ""
    return f"{', '.join(labels)}{suffix}" if labels else "empty community"


def community_activation_metrics(graph: nx.MultiDiGraph, node_ids: set[str], edges: list[dict[str, Any]], evidence_refs: list[str], warnings: list[str]) -> dict[str, Any]:
    weak_nodes = [
        node_id
        for node_id in node_ids
        if str(graph.nodes[node_id].get("entity_quality_hint") or "") in WEAK_ENTITY_QUALITY_HINTS
    ]
    generic_edges = [
        edge
        for edge in edges
        if str(edge.get("relation_type") or "") == "related_to_generic"
        or str(edge.get("generic_relation_review_hint") or "") not in {"", "not_generic"}
    ]
    specific_edge_count = max(0, len(edges) - len(generic_edges))
    stable_node_count = max(0, len(node_ids) - len(weak_nodes))
    relation_diversity = len({str(edge.get("relation_type") or "") for edge in edges if edge.get("relation_type")})
    generic_relation_ratio = len(generic_edges) / len(edges) if edges else 0.0
    weak_node_ratio = len(weak_nodes) / len(node_ids) if node_ids else 0.0
    warning_penalty = min(4.0, len(warnings) * 0.08)
    activation_score = (
        specific_edge_count * 1.35
        + len(evidence_refs) * 0.18
        + stable_node_count * 0.22
        + relation_diversity * 0.35
        - len(generic_edges) * 0.55
        - len(weak_nodes) * 0.25
        - warning_penalty
    )
    if generic_relation_ratio >= 0.45 or weak_node_ratio >= 0.45:
        activation_quality = "review_heavy"
    elif generic_relation_ratio >= 0.25 or weak_node_ratio >= 0.25:
        activation_quality = "medium"
    else:
        activation_quality = "clean_candidate"
    return {
        "activation_score": round(max(0.1, float(activation_score)), 6),
        "activation_quality": activation_quality,
        "specific_edge_count": specific_edge_count,
        "generic_edge_count": len(generic_edges),
        "weak_node_count": len(weak_nodes),
        "generic_relation_ratio": round(float(generic_relation_ratio), 6),
        "weak_node_ratio": round(float(weak_node_ratio), 6),
        "relation_diversity": relation_diversity,
    }


def build_community_rows(graph: nx.MultiDiGraph, projection: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    communities = detect_communities(graph)
    community_rows_out: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    for index, node_ids in enumerate(sorted(communities, key=lambda item: (-len(item), sorted(item)[0])), 1):
        edges = community_edges(graph, node_ids)
        community_id = stable_id("graph_community", json.dumps([projection, index, sorted(node_ids)], sort_keys=True))
        node_refs = sorted(node_ids)
        edge_refs = uniq([[edge.get("edge_id")] for edge in edges], limit=None)
        evidence_refs = uniq(
            [graph.nodes[node_id].get("evidence_refs") for node_id in node_refs]
            + [edge.get("evidence_refs") for edge in edges],
            limit=30,
        )
        warnings = uniq(
            [graph.nodes[node_id].get("warnings") for node_id in node_refs]
            + [edge.get("warnings") for edge in edges],
            limit=30,
        )
        labels = [node_label(graph, node_id) for node_id in node_refs]
        title = community_title(graph, node_ids, edges)
        relation_summaries = [edge_summary(edge) for edge in edges[:12]]
        relation_type_counts = Counter(str(edge.get("relation_type") or "related_to_generic") for edge in edges)
        activation = community_activation_metrics(graph, node_ids, edges, evidence_refs, warnings)
        community_rows_out.append(
            {
                "schema_version": COMMUNITY_SCHEMA_VERSION,
                "community_id": community_id,
                "human_readable_id": index,
                "title": title,
                "level": 0,
                "parent": "",
                "children": [],
                "projection": projection,
                "entity_ids": node_refs,
                "relationship_ids": edge_refs,
                "text_unit_ids": uniq(
                    [graph.nodes[node_id].get("raw_backpointer_refs") for node_id in node_refs]
                    + [edge.get("raw_backpointer_refs") for edge in edges],
                    limit=50,
                ),
                "size": len(node_refs),
                "edge_count": len(edges),
                "relation_type_counts": dict(sorted(relation_type_counts.items())),
                **activation,
                "evidence_refs": evidence_refs,
                "warnings": warnings,
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )
        summary = (
            f"This candidate community centers on {', '.join(labels[:5])}. "
            f"It contains {len(node_refs)} nodes and {len(edges)} intra-community relations. "
            f"Activation quality is {activation['activation_quality']}."
        )
        full_content_parts = [
            summary,
            "Key relation candidates:",
            *[f"- {item}" for item in relation_summaries[:10]],
            "Evidence refs:",
            ", ".join(evidence_refs[:12]) if evidence_refs else "missing evidence refs",
        ]
        report_rows.append(
            {
                "schema_version": COMMUNITY_REPORT_SCHEMA_VERSION,
                "report_id": stable_id("graph_community_report", community_id),
                "community_id": community_id,
                "human_readable_id": index,
                "title": title,
                "summary": summary,
                "full_content": "\n".join(full_content_parts),
                "rank": activation["activation_score"],
                "activation_quality": activation["activation_quality"],
                "generic_relation_ratio": activation["generic_relation_ratio"],
                "weak_node_ratio": activation["weak_node_ratio"],
                "rating_explanation": "Rank is a retrieval activation score with penalties for generic relations and weak nodes, not importance proof.",
                "findings": [
                    {
                        "summary": "Representative relation candidates",
                        "explanation": " ; ".join(relation_summaries[:5]),
                        "evidence_refs": evidence_refs[:10],
                    }
                ],
                "source_node_refs": node_refs,
                "source_edge_refs": edge_refs,
                "evidence_refs": evidence_refs,
                "warnings": warnings + ["community_report_is_extractive_candidate_context_not_truth"],
                "activation_role": "community_expert_candidate",
                "report_is_not_evidence_truth": True,
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )
    return community_rows_out, report_rows


def build_community_report_model_input(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_input_id": stable_id("graph_community_report_input", str(report.get("community_id") or "")),
        "community_id": report.get("community_id") or "",
        "title": report.get("title") or "",
        "activation_quality": report.get("activation_quality") or "",
        "generic_relation_ratio": report.get("generic_relation_ratio"),
        "weak_node_ratio": report.get("weak_node_ratio"),
        "source_node_refs": report.get("source_node_refs") or [],
        "source_edge_refs": report.get("source_edge_refs") or [],
        "evidence_refs": report.get("evidence_refs") or [],
        "extractive_summary": report.get("summary") or "",
        "extractive_full_content": report.get("full_content") or "",
        "extractive_findings": report.get("findings") or [],
        "warnings": report.get("warnings") or [],
        "required_output_schema": {
            "output_kind": "community_report_candidate | model_uncertain | needs_human_review | reject",
            "title": "short title, may reuse input title",
            "summary": "evidence-bound activation summary, not proof",
            "findings": [
                {
                    "summary": "short finding",
                    "explanation": "what this community may help retrieve",
                    "evidence_refs": ["must be selected from input evidence_refs only"],
                }
            ],
            "retrieval_guidance": ["how this community should help query activation"],
            "query_expansion_terms": ["terms useful for graph-aware retrieval"],
            "warnings": ["include uncertainty, generic, attribution, or evidence limitations"],
        },
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
    }


def validated_provider_report(
    report: dict[str, Any],
    payload: dict[str, Any],
    *,
    provider: str,
    model_id: str,
    prompt: PromptPolicy,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    warnings = uniq([report.get("warnings"), payload.get("warnings")], limit=40)
    output_kind = str(payload.get("output_kind") or "").strip()
    if output_kind in {"reject", "model_uncertain", "needs_human_review"}:
        reason = str(payload.get("reason") or payload.get("uncertainty_note") or output_kind)
        updated = dict(report)
        updated["provider_report_status"] = output_kind
        updated["provider_report_reason"] = reason
        updated["warnings"] = uniq([warnings, [f"provider_report_status:{output_kind}"]], limit=50)
        return updated, {
            "community_id": report.get("community_id") or "",
            "failure_kind": output_kind,
            "reason": reason,
            "provider": provider,
            "model_id": model_id,
            "graph_is_not_proof": True,
        }
    if output_kind != "community_report_candidate":
        updated = dict(report)
        updated["provider_report_status"] = "schema_validation_failed"
        updated["warnings"] = uniq([warnings, ["provider_report_unsupported_output_kind"]], limit=50)
        return updated, {
            "community_id": report.get("community_id") or "",
            "failure_kind": "schema_validation_failed",
            "reason": f"unsupported output_kind: {output_kind or '<missing>'}",
            "provider": provider,
            "model_id": model_id,
            "graph_is_not_proof": True,
        }

    allowed_evidence_refs = set(string_list(report.get("evidence_refs")))
    provider_evidence_refs: list[str] = []
    findings: list[dict[str, Any]] = []
    invalid_refs: list[str] = []
    for finding in payload.get("findings") or []:
        if not isinstance(finding, dict):
            continue
        refs = []
        for ref in string_list(finding.get("evidence_refs")):
            if ref in allowed_evidence_refs:
                refs.append(ref)
                if ref not in provider_evidence_refs:
                    provider_evidence_refs.append(ref)
            else:
                invalid_refs.append(ref)
        findings.append(
            {
                "summary": str(finding.get("summary") or "")[:240],
                "explanation": str(finding.get("explanation") or "")[:1200],
                "evidence_refs": refs,
            }
        )

    summary = str(payload.get("summary") or "").strip()
    if not summary:
        updated = dict(report)
        updated["provider_report_status"] = "schema_validation_failed"
        updated["warnings"] = uniq([warnings, ["provider_report_missing_summary"]], limit=50)
        return updated, {
            "community_id": report.get("community_id") or "",
            "failure_kind": "schema_validation_failed",
            "reason": "community_report_candidate missing summary",
            "provider": provider,
            "model_id": model_id,
            "graph_is_not_proof": True,
        }

    if invalid_refs:
        warnings = uniq([warnings, ["provider_report_referenced_unknown_evidence_refs"]], limit=50)
    if not provider_evidence_refs:
        warnings = uniq([warnings, ["provider_report_has_no_valid_finding_evidence_refs"]], limit=50)

    retrieval_guidance = string_list(payload.get("retrieval_guidance"))[:8]
    query_expansion_terms = string_list(payload.get("query_expansion_terms"))[:16]
    full_content_parts = [
        summary,
        "Retrieval guidance:",
        *[f"- {item}" for item in retrieval_guidance],
        "Findings:",
        *[f"- {item.get('summary')}: {item.get('explanation')}" for item in findings[:8]],
        "Evidence refs:",
        ", ".join(provider_evidence_refs[:12]) if provider_evidence_refs else "no validated provider-selected evidence refs",
    ]
    updated = dict(report)
    updated["extractive_summary"] = report.get("summary") or ""
    updated["extractive_full_content"] = report.get("full_content") or ""
    updated["summary"] = summary
    updated["full_content"] = "\n".join(full_content_parts)
    updated["title"] = str(payload.get("title") or report.get("title") or "")[:180]
    updated["findings"] = findings or report.get("findings") or []
    updated["retrieval_guidance"] = retrieval_guidance
    updated["query_expansion_terms"] = query_expansion_terms
    updated["provider_selected_evidence_refs"] = provider_evidence_refs
    updated["provider_report_status"] = "community_report_candidate"
    updated["provider"] = provider
    updated["model_id"] = model_id
    updated["prompt_policy_id"] = prompt.policy_id
    updated["prompt_hash"] = prompt.prompt_hash
    updated["report_generation_method"] = "provider_schema_guided_with_extractive_fallback"
    updated["warnings"] = uniq([warnings, ["provider_report_is_activation_context_not_proof"]], limit=50)
    updated["report_is_not_evidence_truth"] = True
    updated["graph_is_not_proof"] = True
    updated["support_status"] = "not_checked"
    updated["write_permission"] = False
    return updated, None


def apply_provider_community_reports(
    reports: list[dict[str, Any]],
    *,
    project_root: Path,
    provider: str,
    api_mode: str,
    allow_live_api: bool,
    model_id: str,
    prompt_path: Path | str,
    external_model_outputs_path: Path | None,
    max_provider_reports: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if provider == "extractive" or max_provider_reports == 0:
        return reports, [], [], [], {
            "provider": "extractive",
            "status": "not_requested",
            "provider_report_count": 0,
            "failure_count": 0,
        }
    if provider not in SUPPORTED_REPORT_PROVIDERS:
        raise ValueError(f"Unsupported community report provider: {provider}")
    prompt_resolved = resolve_project_path(project_root, prompt_path)
    prompt = PromptPolicy(
        policy_id="graph_community_report_prompt.v0.3",
        path=prompt_resolved,
        text=prompt_resolved.read_text(encoding="utf-8"),
        prompt_hash=proposal_file_hash(prompt_resolved),
    )
    live_api_enabled, live_api_unlock_source = resolve_live_api(provider, allow_live_api)
    model_provider = build_provider(provider, api_mode, external_model_outputs_path) if provider in {"openai", "external_jsonl"} else None
    external_rows = read_jsonl_index(external_model_outputs_path) if provider == "external_jsonl" and external_model_outputs_path else {}

    sorted_reports = sorted(reports, key=lambda row: (-float(row.get("rank") or 0.0), str(row.get("community_id") or "")))
    selected_ids = {str(row.get("community_id") or "") for row in sorted_reports[:max_provider_reports]}
    output_reports: list[dict[str, Any]] = []
    model_call_inputs: list[dict[str, Any]] = []
    model_call_results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for report in reports:
        if str(report.get("community_id") or "") not in selected_ids:
            output_reports.append(report)
            continue
        input_packet = build_community_report_model_input(report)
        model_call_inputs.append(
            {
                "schema_version": "graph_v03.community_report_model_call_input.v0.1",
                "community_id": report.get("community_id") or "",
                "proposal_input_id": input_packet["proposal_input_id"],
                "provider": provider,
                "model_id": model_id,
                "prompt_policy_id": prompt.policy_id,
                "prompt_hash": prompt.prompt_hash,
                "estimated_input_tokens": estimate_tokens(prompt.text + json.dumps(input_packet, ensure_ascii=False)),
                "input_packet": input_packet,
                "graph_is_not_proof": True,
            }
        )
        try:
            started = time.perf_counter()
            if provider == "external_jsonl":
                row = external_rows.get(str(input_packet["proposal_input_id"])) or external_rows.get(str(report.get("community_id") or ""))
                if not row:
                    raise RuntimeError("external_jsonl missing output for community report input")
                raw_output = row.get("output_text", row.get("model_output"))
                output_text = json.dumps(raw_output, ensure_ascii=False) if isinstance(raw_output, dict) else str(raw_output or "")
                result_provider = "external_jsonl"
                result_model = str(row.get("model_id") or model_id)
                latency_ms = 0
                input_tokens = estimate_tokens(prompt.text)
                output_tokens = estimate_tokens(output_text)
            else:
                if model_provider is None:
                    raise RuntimeError("provider was not initialized")
                result = model_provider.generate(prompt=prompt, model_id=model_id, input_packet=input_packet)
                output_text = result.output_text
                result_provider = result.provider
                result_model = result.model_id
                latency_ms = result.latency_ms if result.latency_ms is not None else int((time.perf_counter() - started) * 1000)
                input_tokens = result.estimated_input_tokens
                output_tokens = result.estimated_output_tokens
            payload, parse_errors = parse_provider_json(output_text)
            model_call_results.append(
                {
                    "community_id": report.get("community_id") or "",
                    "proposal_input_id": input_packet["proposal_input_id"],
                    "provider": result_provider,
                    "model_id": result_model,
                    "prompt_policy_id": prompt.policy_id,
                    "prompt_hash": prompt.prompt_hash,
                    "estimated_input_tokens": input_tokens,
                    "estimated_output_tokens": output_tokens,
                    "latency_ms": latency_ms,
                    "parse_errors": parse_errors,
                    "graph_is_not_proof": True,
                }
            )
            if payload is None:
                updated = dict(report)
                updated["provider_report_status"] = "schema_validation_failed"
                updated["warnings"] = uniq([report.get("warnings"), parse_errors], limit=50)
                output_reports.append(updated)
                failures.append(
                    {
                        "community_id": report.get("community_id") or "",
                        "failure_kind": "schema_validation_failed",
                        "reason": "provider output was not valid strict JSON",
                        "provider": result_provider,
                        "model_id": result_model,
                        "warnings": parse_errors,
                        "graph_is_not_proof": True,
                    }
                )
                continue
            updated, failure = validated_provider_report(report, payload, provider=result_provider, model_id=result_model, prompt=prompt)
            output_reports.append(updated)
            if failure:
                failures.append(failure)
        except Exception as exc:
            updated = dict(report)
            updated["provider_report_status"] = "provider_call_failed"
            updated["warnings"] = uniq([report.get("warnings"), ["provider_error_redacted"]], limit=50)
            output_reports.append(updated)
            failures.append(
                {
                    "community_id": report.get("community_id") or "",
                    "failure_kind": "provider_call_failed",
                    "reason": f"{type(exc).__name__}: {exc}",
                    "provider": provider,
                    "model_id": model_id,
                    "warnings": ["provider_error_redacted"],
                    "graph_is_not_proof": True,
                }
            )

    status = {
        "provider": provider,
        "status": "completed",
        "provider_report_count": sum(1 for row in output_reports if row.get("provider_report_status") == "community_report_candidate"),
        "failure_count": len(failures),
        "selected_community_count": len(selected_ids),
        "prompt_path": str(prompt_resolved),
        "prompt_hash": prompt.prompt_hash,
        "api_mode": api_mode,
        "live_api_enabled": live_api_enabled,
        "live_api_unlock_source": live_api_unlock_source,
        "model_id": model_id,
    }
    return output_reports, model_call_inputs, model_call_results, failures, status


def write_report(path: Path, manifest: dict[str, Any], community_reports: list[dict[str, Any]]) -> None:
    lines = [
        "# v0.3 Graph Profile And Community Builder",
        "",
        f"- workspace: `{manifest['workspace']}`",
        f"- graph_dir: `{manifest['graph_dir']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- projection: `{manifest['projection']}`",
        "- graph_is_not_proof: `true`",
        "- support_status: `not_checked`",
        "",
        "## Counts",
        "",
        f"- entity_profile_cards: {manifest['counts']['entity_profile_cards']}",
        f"- relation_profile_cards: {manifest['counts']['relation_profile_cards']}",
        f"- communities: {manifest['counts']['communities']}",
        f"- community_reports: {manifest['counts']['community_reports']}",
        f"- provider_report_count: {manifest['counts'].get('provider_report_count', 0)}",
        f"- provider_report_failure_count: {manifest['counts'].get('provider_report_failure_count', 0)}",
        "",
        "## Provider Report Status",
        "",
        f"- provider: `{(manifest.get('community_report_provider') or {}).get('provider', 'extractive')}`",
        f"- status: `{(manifest.get('community_report_provider') or {}).get('status', '')}`",
        f"- model: `{(manifest.get('community_report_provider') or {}).get('model_id', '')}`",
        "",
        "## Top Community Reports",
        "",
    ]
    for report in sorted(community_reports, key=lambda row: (-float(row.get("rank") or 0.0), str(row.get("community_id") or "")))[:10]:
        lines.extend(
            [
                f"### {report['title']}",
                "",
                f"- rank: {report['rank']}",
                f"- nodes: {len(report.get('source_node_refs') or [])}",
                f"- edges: {len(report.get('source_edge_refs') or [])}",
                f"- evidence refs: {len(report.get('evidence_refs') or [])}",
                f"- provider status: `{report.get('provider_report_status', 'extractive')}`",
                f"- summary: {report['summary']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Boundary",
            "",
            "- Profiles and community reports are activation/context material only.",
            "- Description, summary, and full_content are helper fields, not evidence truth.",
            "- Retrieval can use these reports as community-level gates, but final support still needs evidence inspection.",
            "- No durable memory, graph truth, S3 asset, or support-checker authority is written.",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


def run_builder(
    workspace: Path,
    *,
    project_root: Path | None = None,
    graph_dir: Path | None = None,
    output_dir: Path | None = None,
    projection: str = "review_aware_graph",
    community_report_provider: str = "extractive",
    api_mode: str | None = None,
    allow_live_api: bool = False,
    model_id: str | None = None,
    prompt_path: Path | str = DEFAULT_COMMUNITY_REPORT_PROMPT,
    env_file: Path | str = ".env",
    external_model_outputs_path: Path | None = None,
    max_provider_reports: int = 0,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    project_root = (project_root or Path(".")).resolve()
    env_path = resolve_project_path(project_root, env_file)
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path, override=True)
    if community_report_provider not in SUPPORTED_REPORT_PROVIDERS:
        raise ValueError(f"Unsupported community_report_provider: {community_report_provider}")
    api_mode = api_mode or os.environ.get("OPENAI_API_MODE") or "responses"
    if api_mode not in SUPPORTED_API_MODES:
        raise ValueError(f"Unsupported api_mode: {api_mode}")
    model_id = model_id or os.environ.get("OPENAI_MODEL_WEAK") or "gpt-4o-mini"
    graph_dir = (graph_dir or workspace / DEFAULT_GRAPH_DIR_NAME).resolve()
    output_dir = (output_dir or workspace / DEFAULT_OUTPUT_DIR_NAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    node_path = graph_dir / "graph_nodes_table.jsonl"
    edge_path = graph_dir / "graph_edges_table.jsonl"
    claim_path = graph_dir / "graph_claims_table.jsonl"
    nodes = read_jsonl(node_path)
    edges = read_jsonl(edge_path)
    graph, projection_stats = load_projection(projection=projection, nodes=nodes, edges=edges)

    entity_profiles = node_profile_cards(nodes, edges)
    relation_profiles = relation_profile_cards(edges)
    communities, community_reports = build_community_rows(graph, projection)
    community_reports, provider_inputs, provider_results, provider_failures, provider_status = apply_provider_community_reports(
        community_reports,
        project_root=project_root,
        provider=community_report_provider,
        api_mode=api_mode,
        allow_live_api=allow_live_api,
        model_id=model_id,
        prompt_path=prompt_path,
        external_model_outputs_path=external_model_outputs_path.resolve() if external_model_outputs_path else None,
        max_provider_reports=max_provider_reports,
    )

    write_jsonl(output_dir / "graph_entity_profile_cards.jsonl", entity_profiles)
    write_jsonl(output_dir / "graph_relation_profile_cards.jsonl", relation_profiles)
    write_jsonl(output_dir / "graph_communities.jsonl", communities)
    write_jsonl(output_dir / "graph_community_reports.jsonl", community_reports)
    write_jsonl(output_dir / "graph_community_report_model_call_inputs.jsonl", provider_inputs)
    write_jsonl(output_dir / "graph_community_report_model_call_results.jsonl", provider_results)
    write_jsonl(output_dir / "graph_community_report_failures.jsonl", provider_failures)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "graph_dir": str(graph_dir),
        "output_dir": str(output_dir),
        "projection": projection,
        "community_report_provider": provider_status,
        "source_hashes": {
            "graph_nodes_table.jsonl": file_hash(node_path),
            "graph_edges_table.jsonl": file_hash(edge_path),
            "graph_claims_table.jsonl": file_hash(claim_path) if claim_path.exists() else None,
        },
        "projection_stats": projection_stats,
        "counts": {
            "entity_profile_cards": len(entity_profiles),
            "relation_profile_cards": len(relation_profiles),
            "communities": len(communities),
            "community_reports": len(community_reports),
            "provider_report_count": provider_status.get("provider_report_count", 0),
            "provider_report_failure_count": len(provider_failures),
            "provider_report_model_call_inputs": len(provider_inputs),
            "provider_report_model_call_results": len(provider_results),
        },
        "borrowed_patterns": [
            "GraphRAG-style separate community and community_report tables",
            "LightRAG-style low/high-level retrieval preparation",
            "NetworkX community detection as inspectable prototype layer",
        ],
        "boundary": {
            "profile_is_not_evidence_truth": True,
            "graph_is_not_proof": True,
            "support_status": "not_checked",
            "write_permission": False,
            "community_report_summary_is_helper_context": True,
        },
    }
    write_json(output_dir / "graph_profile_community_manifest.json", manifest)
    write_report(output_dir / "graph_profile_community_report.md", manifest, community_reports)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v0.3 graph profile cards and community reports.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--projection", default="review_aware_graph", choices=["full_candidate_graph", "review_aware_graph", "stable_core_graph"])
    parser.add_argument("--community-report-provider", default="extractive", choices=sorted(SUPPORTED_REPORT_PROVIDERS))
    parser.add_argument("--api-mode", default=None, choices=sorted(SUPPORTED_API_MODES))
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--prompt-path", default=DEFAULT_COMMUNITY_REPORT_PROMPT)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--external-model-outputs", default=None)
    parser.add_argument("--max-provider-reports", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run_builder(
        Path(args.workspace),
        project_root=Path(args.project_root),
        graph_dir=Path(args.graph_dir) if args.graph_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        projection=args.projection,
        community_report_provider=args.community_report_provider,
        api_mode=args.api_mode,
        allow_live_api=args.allow_live_api,
        model_id=args.model_id,
        prompt_path=args.prompt_path,
        env_file=args.env_file,
        external_model_outputs_path=Path(args.external_model_outputs) if args.external_model_outputs else None,
        max_provider_reports=args.max_provider_reports,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
