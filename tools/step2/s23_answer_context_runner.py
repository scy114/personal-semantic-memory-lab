#!/usr/bin/env python
"""Build Step 2.3 answer context and prompt view from Step 2.2 artifacts.

The full dual-branch retrieval package is an audit/debug source package.
``s23_answer_context.json`` is a lightweight but still debuggable intermediate
artifact. ``s23_prompt_context.md`` is the token-lean prompt view that a final
agent should consume by default.
"""

from __future__ import annotations

import argparse
import json
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
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path, required: bool = False) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def compact_text(text: Any, limit: int = 900) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").lower().split())


def unique_list(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text not in seen:
            out.append(text)
            seen.add(text)
    return out


def dual_branch_summary(dual_package: dict[str, Any]) -> dict[str, Any]:
    def counts(branch: dict[str, Any]) -> dict[str, int]:
        keys = [
            "bm25_candidates",
            "embedding_candidates",
            "overlap_candidates",
            "bm25_only_candidates",
            "embedding_only_candidates",
            "conflict_or_risk_candidates",
            "summary_context_candidates",
            "memory_unit_candidates",
            "raw_evidence_candidates",
        ]
        return {key: len(branch.get(key, [])) for key in keys if key in branch}

    return {
        "schema_version": dual_package.get("schema_version"),
        "step1_evidence_retrieval": counts(dual_package.get("step1_evidence_retrieval", {})),
        "step2_user_model_retrieval": counts(dual_package.get("step2_user_model_retrieval", {})),
        "graph_status": dual_package.get("graph_branch", {}).get("graph_status", "unknown"),
        "analysis_status": {
            "s3_status": dual_package.get("analysis_layer", {}).get("s3_status", "unknown"),
            "feedback_repair_status": dual_package.get("analysis_layer", {}).get("feedback_repair_status", "unknown"),
        },
        "support_status": dual_package.get("support_status", "not_checked"),
        "warnings": dual_package.get("warnings", []),
    }


def candidate_warnings(row: dict[str, Any]) -> list[str]:
    warnings = list(row.get("warnings", []) or [])
    layer = row.get("item_layer") or row.get("source_object_type") or row.get("object_type")
    if layer in {"doc_level_summary", "summary", "summary_context"}:
        warnings.append("summary_context_only")
    if layer in {"memory_unit", "s1_memory_unit"}:
        warnings.append("memory_unit_not_raw_evidence")
    if row.get("subject_match_status") in {"mixed_participant", "other_participant"}:
        warnings.append(f"subject_{row.get('subject_match_status')}")
    if row.get("support_status") in {None, "", "not_checked"}:
        warnings.append("retrieval_hit_not_support_check")
    return unique_list(warnings)


def graph_item_warnings(row: dict[str, Any]) -> list[str]:
    warnings = list(row.get("warnings", []) or [])
    if row.get("graph_is_not_proof") is not True:
        warnings.append("missing_graph_is_not_proof_flag")
    if row.get("support_status") in {None, "", "not_checked"}:
        warnings.append("graph_item_not_support_checked")
    if not row.get("evidence_refs"):
        warnings.append("graph_item_missing_evidence_refs")
    confidence = str(row.get("confidence") or "")
    if confidence in {"related_to_generic", "needs_entity_resolution", "needs_schema_extension", "needs_attribution_review", "low_graph_value"}:
        warnings.append(f"graph_relation_review:{confidence}")
    return unique_list(warnings)


def classify_graph_context_item(row: dict[str, Any]) -> str:
    kind = str(row.get("kind") or "")
    warnings = set(graph_item_warnings(row))
    confidence = str(row.get("confidence") or "")
    if kind == "v03_noisy_navigation_path" or row.get("path_quality_status") == "noisy_navigation":
        return "noisy_navigation_paths"
    if kind == "v03_evidence_path":
        return "evidence_paths"
    if "graph_item_missing_evidence_refs" in warnings:
        return "unresolved_or_unsafe_graph_items"
    if confidence in {"needs_entity_resolution", "needs_schema_extension", "needs_attribution_review", "low_graph_value"}:
        return "cautious_graph_relations"
    if kind in {"v03_relation_neighbor", "v03_ranked_relation"} and row.get("evidence_refs"):
        return "supported_graph_relations"
    if kind in {"v03_seed_node", "v03_ranked_node"}:
        return "query_expansion_terms"
    return "cautious_graph_relations"


def make_graph_answer_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": row.get("kind", "graph_context"),
        "summary": compact_text(row.get("summary"), 500),
        "source_node_refs": row.get("source_node_refs", []),
        "source_edge_refs": row.get("source_edge_refs", []),
        "evidence_refs": row.get("evidence_refs", []),
        "confidence": row.get("confidence", "candidate"),
        "warnings": graph_item_warnings(row),
        "retrieval_source": row.get("retrieval_source", "unknown"),
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
        "use_policy": "supporting_context_not_proof",
    }


def query_expansion_from_graph_package(graph_retrieval_package: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not graph_retrieval_package:
        return []
    terms = graph_retrieval_package.get("graph_branch", {}).get("query_expansion_terms", [])
    out: list[dict[str, Any]] = []
    for item in terms[:12]:
        term = str(item.get("term") or "").strip()
        if term:
            out.append({"term": term, "count": item.get("count", 1), "source": "v03_graph_retrieval"})
    return out


def build_graph_answer_context(
    graph_context: list[dict[str, Any]],
    graph_retrieval_package: dict[str, Any] | None,
) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "supported_graph_relations": [],
        "cautious_graph_relations": [],
        "evidence_paths": [],
        "noisy_navigation_paths": [],
        "query_expansion_terms": [],
        "unresolved_or_unsafe_graph_items": [],
    }
    for row in graph_context:
        bucket = classify_graph_context_item(row)
        buckets[bucket].append(make_graph_answer_item(row))

    package_paths = graph_retrieval_package.get("graph_branch", {}).get("evidence_paths", []) if graph_retrieval_package else []
    seen_path_summaries = {item["summary"] for item in buckets["evidence_paths"]}
    for path in package_paths[:8]:
        summary = " -> ".join(str(label) for label in path.get("path_labels", []))
        if not summary or summary in seen_path_summaries:
            continue
        evidence_refs: list[str] = []
        edge_refs: list[str] = []
        for segment in path.get("segments", []):
            for ref in segment.get("evidence_refs", []) or []:
                if ref not in evidence_refs:
                    evidence_refs.append(ref)
            edge_id = segment.get("edge_id")
            if edge_id and edge_id not in edge_refs:
                edge_refs.append(edge_id)
        buckets["evidence_paths"].append(
            {
                "kind": "v03_evidence_path",
                "summary": summary,
                "source_node_refs": path.get("path_node_ids", []),
                "source_edge_refs": edge_refs,
                "evidence_refs": evidence_refs,
                "confidence": "candidate_path",
                "warnings": ["v03_evidence_path_is_navigation_not_support_proof"] + unique_list(path.get("path_quality_reasons", [])),
                "retrieval_source": "v03_graph_retrieval_package",
                "path_quality_status": path.get("path_quality_status", "clean"),
                "path_use_policy": path.get("path_use_policy", "navigation_context_only"),
                "path_node_quality": path.get("path_node_quality", []),
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
                "use_policy": "navigation_context_only",
            }
        )
        seen_path_summaries.add(summary)

    package_noisy_paths = graph_retrieval_package.get("graph_branch", {}).get("noisy_navigation_paths", []) if graph_retrieval_package else []
    seen_noisy_summaries = {item["summary"] for item in buckets["noisy_navigation_paths"]}
    for path in package_noisy_paths[:8]:
        summary = " -> ".join(str(label) for label in path.get("path_labels", []))
        if not summary or summary in seen_path_summaries or summary in seen_noisy_summaries:
            continue
        evidence_refs: list[str] = []
        edge_refs: list[str] = []
        for segment in path.get("segments", []):
            for ref in segment.get("evidence_refs", []) or []:
                if ref not in evidence_refs:
                    evidence_refs.append(ref)
            edge_id = segment.get("edge_id")
            if edge_id and edge_id not in edge_refs:
                edge_refs.append(edge_id)
        buckets["noisy_navigation_paths"].append(
            {
                "kind": "v03_noisy_navigation_path",
                "summary": summary,
                "source_node_refs": path.get("path_node_ids", []),
                "source_edge_refs": edge_refs,
                "evidence_refs": evidence_refs,
                "confidence": "noisy_navigation",
                "warnings": ["v03_noisy_navigation_path_not_explanatory"] + unique_list(path.get("path_quality_reasons", [])),
                "retrieval_source": "v03_graph_retrieval_package",
                "path_quality_status": "noisy_navigation",
                "path_use_policy": "navigation_debug_only",
                "path_node_quality": path.get("path_node_quality", []),
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
                "use_policy": "navigation_debug_only",
            }
        )
        seen_noisy_summaries.add(summary)

    buckets["query_expansion_terms"].extend(query_expansion_from_graph_package(graph_retrieval_package))
    return {
        "schema_version": "s2.s23_graph_answer_context.v1",
        "mode": "graph_answer_context",
        "supported_graph_relations": buckets["supported_graph_relations"],
        "cautious_graph_relations": buckets["cautious_graph_relations"],
        "evidence_paths": buckets["evidence_paths"],
        "noisy_navigation_paths": buckets["noisy_navigation_paths"],
        "query_expansion_terms": buckets["query_expansion_terms"],
        "unresolved_or_unsafe_graph_items": buckets["unresolved_or_unsafe_graph_items"],
        "graph_is_not_proof": True,
        "support_status": "not_checked",
        "write_permission": False,
        "usage_rules": [
            "Use supported_graph_relations as answer context only when evidence refs are present.",
            "Use cautious_graph_relations as uncertain context, not asserted facts.",
            "Use evidence_paths for navigation and context organization, not as proof.",
            "Keep noisy_navigation_paths separate; use them for debugging or review, not explanation.",
            "Do not treat graph metrics or graph retrieval as support checks.",
        ],
    }


def classify_context_item(row: dict[str, Any], source: str, query_support: dict[str, Any]) -> dict[str, str]:
    layer = row.get("item_layer") or row.get("source_object_type") or row.get("object_type") or "unknown"
    object_type = row.get("object_type") or row.get("source_object_type") or layer
    support_status = row.get("support_status") or "not_checked"
    warnings = candidate_warnings(row)
    has_evidence_refs = bool(row.get("evidence_refs") or row.get("backpointer_refs"))
    query_status_for_item = query_support.get("support_status", "not_checked") if has_evidence_refs else "not_checked"

    if layer == "raw_evidence" or object_type == "raw_evidence":
        evidence_support_status = "checked" if query_support.get("support_status") == "checked" else support_status
        use_policy = "can_answer" if query_support.get("support_strength") == "direct" else "supporting_context"
        return {
            "type": "direct_evidence",
            "use_policy": use_policy,
            "evidence_support_status": evidence_support_status,
            "query_support_status": query_support.get("support_status", "not_checked"),
            "retrieval_status": support_status,
        }

    if layer == "doc_level_summary" or object_type in {"doc_level_summary", "summary_context"}:
        return {
            "type": "context_only",
            "use_policy": "background_only",
            "evidence_support_status": "not_checked",
            "query_support_status": "not_checked",
            "retrieval_status": support_status,
        }

    if layer == "memory_unit" or object_type == "memory_unit":
        status = "evidence_bound" if row.get("evidence_refs") or row.get("backpointer_refs") else "not_checked"
        use_policy = "supporting_context" if status == "evidence_bound" else "background_only"
        return {
            "type": "context_only",
            "use_policy": use_policy,
            "evidence_support_status": status,
            "query_support_status": query_status_for_item,
            "retrieval_status": support_status,
        }

    if source == "step2_selected_context":
        status = "evidence_bound" if row.get("evidence_refs") else support_status
        use_policy = "supporting_context" if status == "evidence_bound" else "background_only"
        if "no_evidence_refs" in warnings and status == "not_checked":
            use_policy = "background_only"
        item_type = "graph_relation" if graph_relation_from_text(str(row.get("text", ""))) else "s2_summary"
        return {
            "type": item_type,
            "use_policy": use_policy,
            "evidence_support_status": status,
            "query_support_status": query_status_for_item,
            "retrieval_status": support_status,
        }

    if "subject_other_participant" in warnings or "subject_mixed_participant" in warnings:
        return {
            "type": "context_only",
            "use_policy": "do_not_use_as_fact",
            "evidence_support_status": support_status,
            "query_support_status": query_status_for_item,
            "retrieval_status": support_status,
        }

    use_policy = "background_only" if support_status == "not_checked" else "supporting_context"
    return {
        "type": "context_only",
        "use_policy": use_policy,
        "evidence_support_status": support_status,
        "query_support_status": query_status_for_item,
        "retrieval_status": support_status,
    }


def retrieved_by_from_row(row: dict[str, Any], source: str, package_section: str) -> list[str]:
    views: list[str] = []
    score_components = row.get("score_components", {}) or {}
    active_backends = [str(backend) for backend in row.get("active_backends", [])]
    backend = str(row.get("retrieval_backend") or row.get("retrieval_method") or row.get("method") or "")
    if source == "memory_context_interpretation.direct_evidence":
        views.append("direct_evidence")
    if source == "step2_selected_context":
        views.append("s2_selected_context")
    if package_section == "overlap_candidates":
        views.extend(["bm25", "embedding"])
    if package_section in {"bm25_candidates", "bm25_only_candidates"}:
        views.append("bm25")
    if package_section in {"embedding_candidates", "embedding_only_candidates"}:
        views.append("embedding")
    if package_section == "raw_evidence_candidates":
        views.append("direct_evidence")
    if package_section == "summary_context_candidates":
        views.append("s2_summary")
    if package_section == "memory_unit_candidates":
        views.append("memory_unit")
    if positiveish(score_components.get("bm25")) or positiveish(score_components.get("bm25_normalized_score")):
        views.append("bm25")
    if positiveish(score_components.get("embedding")) or positiveish(score_components.get("embedding_normalized_score")):
        views.append("embedding")
    if "bm25" in active_backends or backend == "bm25":
        views.append("bm25")
    if "embedding" in active_backends or backend == "embedding":
        views.append("embedding")
    if graph_relation_from_text(str(row.get("text", ""))):
        views.append("graph")
    return unique_list(views)


def positiveish(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def retrieval_reason_summary(row: dict[str, Any], retrieved_by: list[str], text: str) -> list[str]:
    reasons: list[str] = []
    lower_text = text.lower()
    if "bm25" in retrieved_by:
        if "dance studio" in lower_text:
            reasons.append("bm25: lexical match on dance studio")
        else:
            reasons.append("bm25: lexical match")
    if "embedding" in retrieved_by:
        reasons.append("embedding: semantic similarity heuristic")
    relation = graph_relation_from_text(text)
    if relation:
        reasons.append(f"graph: {relation['subject']} --{relation['relation']}--> {relation['object']}")
    return unique_list(reasons[:3])


def human_label(identifier: str) -> str:
    text = str(identifier or "").strip()
    if text == "public_locomo_conv_30_jon":
        return "Jon"
    if text == "public_locomo_conv_30_gina":
        return "Gina"
    return text


def graph_relation_from_text(text: str) -> dict[str, str] | None:
    if " works_on " not in text:
        return None
    before, _, after = text.partition("; derived from ")
    subject, _, obj = before.partition(" works_on ")
    if not subject or not obj:
        return None
    support_text = after.split(": ", 1)[-1] if after else ""
    return {
        "subject": human_label(subject.strip()),
        "relation": "works_on",
        "object": obj.strip(),
        "support_text": support_text.strip(),
    }


def make_answer_item(
    row: dict[str, Any],
    *,
    source: str,
    query_support: dict[str, Any],
    package_section: str,
) -> dict[str, Any] | None:
    text = compact_text(row.get("text") or row.get("summary") or row.get("content") or "")
    if not text:
        return None
    classification = classify_context_item(row, source, query_support)
    source_candidate_id = (
        row.get("candidate_id")
        or row.get("retrieval_result_id")
        or row.get("index_entry_id")
        or row.get("object_id")
        or row.get("evidence_ref")
    )
    return {
        "text": text,
        "type": classification["type"],
        "use_policy": classification["use_policy"],
        "support_status": classification["evidence_support_status"],
        "evidence_support_status": classification["evidence_support_status"],
        "query_support_status": classification["query_support_status"],
        "retrieval_status": classification["retrieval_status"],
        "warnings": candidate_warnings(row),
        "retrieved_by": retrieved_by_from_row(row, source, package_section),
        "retrieval_reason_summary": retrieval_reason_summary(
            row,
            retrieved_by_from_row(row, source, package_section),
            text,
        ),
        "graph_relation": graph_relation_from_text(text),
        "audit_pointer": {
            "source": source,
            "package_section": package_section,
            "source_candidate_id": source_candidate_id,
            "evidence_refs": row.get("evidence_refs", []),
            "backpointer_refs": row.get("backpointer_refs", []),
        },
    }


def add_unique_item(items: list[dict[str, Any]], item: dict[str, Any] | None, seen: set[str]) -> None:
    if item is None:
        return
    pointer = item.get("audit_pointer", {})
    key = str(pointer.get("source_candidate_id") or item.get("text"))
    if key in seen:
        return
    items.append(item)
    seen.add(key)


def merge_key_for_item(item: dict[str, Any]) -> tuple[str, str, str]:
    text_key = normalize_text(item.get("text"))[:140]
    if not text_key:
        pointer = item.get("audit_pointer", {})
        evidence_refs = pointer.get("evidence_refs") or []
        text_key = "refs:" + "|".join(sorted(str(ref) for ref in evidence_refs))
    return (text_key, str(item.get("type")), str(item.get("evidence_support_status")))


def merge_answer_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for item in items:
        key = merge_key_for_item(item)
        if key not in merged:
            merged[key] = {**item, "audit_pointers": []}
        current = merged[key]
        current["warnings"] = unique_list(list(current.get("warnings", [])) + list(item.get("warnings", [])))
        current["retrieved_by"] = unique_list(list(current.get("retrieved_by", [])) + list(item.get("retrieved_by", [])))
        current["retrieval_reason_summary"] = unique_list(
            list(current.get("retrieval_reason_summary", [])) + list(item.get("retrieval_reason_summary", []))
        )[:4]
        pointer = item.get("audit_pointer")
        if pointer:
            current["audit_pointers"].append(pointer)
            current["audit_pointer"] = current["audit_pointers"][0]
        if current.get("use_policy") != "can_answer" and item.get("use_policy") == "can_answer":
            current["use_policy"] = "can_answer"
        if not current.get("graph_relation") and item.get("graph_relation"):
            current["graph_relation"] = item["graph_relation"]
        if len(str(item.get("text", ""))) > len(str(current.get("text", ""))):
            current["text"] = item["text"]
    return list(merged.values())


def prompt_warning_filter(warnings: list[str]) -> list[str]:
    allowed = {
        "no_evidence_refs",
        "summary_context_only",
        "memory_unit_not_raw_evidence",
        "subject_mixed_participant",
        "subject_other_participant",
        "retrieval_hit_not_support_check",
    }
    return [warning for warning in unique_list(warnings) if warning in allowed]


def extract_s23_prompt_items(answer_context: dict[str, Any]) -> list[dict[str, Any]]:
    items = merge_answer_items(answer_context.get("answer_context", []))
    prompt_items: list[dict[str, Any]] = []
    for item in items:
        use_policy = item.get("use_policy", "background_only")
        warnings = prompt_warning_filter(item.get("warnings", []))
        evidence_support_status = item.get("evidence_support_status") or item.get("support_status") or "not_checked"
        query_support_status = item.get("query_support_status", "not_checked")
        if use_policy == "can_answer" and evidence_support_status == "checked":
            warnings = [warning for warning in warnings if warning != "retrieval_hit_not_support_check"]
        if "no_evidence_refs" in warnings and evidence_support_status == "not_checked":
            use_policy = "background_only" if use_policy != "do_not_use_as_fact" else use_policy
            query_support_status = "not_checked"
        if "no_evidence_refs" not in warnings:
            warnings = [warning for warning in warnings if warning != "retrieval_hit_not_support_check"]
        prompt_items.append(
            {
                "text": str(item.get("text", "")),
                "type": item.get("type", "context_only"),
                "use_policy": use_policy,
                "evidence_support_status": evidence_support_status,
                "query_support_status": query_support_status,
                "retrieved_by": item.get("retrieved_by", []),
                "retrieval_reason_summary": item.get("retrieval_reason_summary", []),
                "graph_relation": item.get("graph_relation"),
                "warnings": warnings,
            }
        )
    return prompt_items


def short_ref_list(refs: list[Any], limit: int = 3) -> str:
    values = [str(ref) for ref in refs if str(ref).strip()]
    if not values:
        return "no refs"
    shown = values[:limit]
    suffix = f"; +{len(values) - limit} more" if len(values) > limit else ""
    return ", ".join(shown) + suffix


def compact_graph_warning_flags(warnings: list[Any]) -> list[str]:
    warning_text = " ".join(str(warning) for warning in warnings)
    flags: list[str] = []
    if "generic" in warning_text or "related_to_generic" in warning_text:
        flags.append("generic_relation")
    if "ambiguous" in warning_text or "uncertain" in warning_text:
        flags.append("ambiguous")
    if "no_evidence" in warning_text or "missing" in warning_text:
        flags.append("evidence_warning")
    if "review" in warning_text or "noisy" in warning_text:
        flags.append("needs_review")
    if warnings and not flags:
        flags.append("has_warning")
    return unique_list(flags)


def render_s23_prompt_context(answer_context: dict[str, Any], *, prompt_view: str = "answer") -> str:
    if prompt_view not in {"answer", "why", "audit"}:
        raise ValueError(f"Unsupported prompt_view: {prompt_view}")
    query = answer_context.get("query", {})
    items = extract_s23_prompt_items(answer_context)
    graph_answer_context = answer_context.get("graph_answer_context", {})
    lines = [
        "# Step 2.3 Prompt Context",
        "",
        f"Question: {query.get('question', '')}",
        "",
        "Use this as bounded memory context. Retrieval hits are not support checks.",
        "",
        "## Context Items",
        "",
    ]
    if not items:
        lines.append("- No usable context items.")
    for idx, item in enumerate(items, 1):
        warnings = ", ".join(item.get("warnings", [])) or "none"
        retrieved_by = "+".join(item.get("retrieved_by", [])) or "unknown"
        status_bits = (
            f"{item['type']} | {item['use_policy']} | "
            f"evidence:{item['evidence_support_status']} | query:{item['query_support_status']} | "
            f"retrieved_by:{retrieved_by}"
        )
        graph_relation = item.get("graph_relation")
        if graph_relation:
            text_line = f"{graph_relation['subject']} --{graph_relation['relation']}--> {graph_relation['object']}"
            support_text = graph_relation.get("support_text")
        else:
            text_line = item["text"]
            support_text = None
        lines.extend(
            [
                f"{idx}. [{status_bits}]",
                f"   {text_line}",
            ]
        )
        if support_text:
            lines.append(f"   - support_text: {support_text}")
        if item.get("retrieval_reason_summary"):
            lines.append(f"   - retrieval_reason_summary: {'; '.join(item['retrieval_reason_summary'])}")
        if warnings != "none":
            lines.append(f"   - warnings: {warnings}")

    if graph_answer_context:
        lines.extend(["", "## Graph Answer Context", ""])
        supported = graph_answer_context.get("supported_graph_relations", [])
        cautious = graph_answer_context.get("cautious_graph_relations", [])
        paths = graph_answer_context.get("evidence_paths", [])
        noisy_paths = graph_answer_context.get("noisy_navigation_paths", [])
        expansion_terms = graph_answer_context.get("query_expansion_terms", [])
        if supported:
            lines.append("Supported candidate relations:")
            for item in supported[:8]:
                refs = short_ref_list(item.get("evidence_refs", []))
                lines.append(f"- {item.get('summary', '')} (refs: {refs})")
        if cautious:
            lines.append("")
            lines.append("Cautious graph context:")
            for item in cautious[:5]:
                refs = short_ref_list(item.get("evidence_refs", []))
                flags = compact_graph_warning_flags(item.get("warnings", []))
                flag_text = f"; flags: {', '.join(flags)}" if flags else ""
                lines.append(f"- {item.get('summary', '')} (refs: {refs}{flag_text})")
        if paths and prompt_view in {"why", "audit"}:
            lines.append("")
            lines.append("Evidence paths for navigation only:")
            for item in paths[:5]:
                refs = short_ref_list(item.get("evidence_refs", []))
                lines.append(f"- {item.get('summary', '')} (refs: {refs})")
        if noisy_paths and prompt_view == "audit":
            lines.append("")
            lines.append("Noisy navigation paths for audit only:")
            for item in noisy_paths[:5]:
                flags = compact_graph_warning_flags(item.get("warnings", []))
                warning_text = ", ".join(flags) if flags else "has_warning"
                lines.append(f"- {item.get('summary', '')} (flags: {warning_text})")
        omitted_path_count = len(paths) if prompt_view == "answer" else 0
        omitted_noisy_count = len(noisy_paths) if prompt_view != "audit" else 0
        if omitted_path_count or omitted_noisy_count:
            lines.append("")
            lines.append(
                "Graph audit details omitted from this prompt view "
                f"(evidence_paths={omitted_path_count}, noisy_paths={omitted_noisy_count})."
            )
        if expansion_terms:
            terms = ", ".join(item.get("term", "") for item in expansion_terms[:12] if item.get("term"))
            if terms:
                lines.append("")
                lines.append(f"Graph query expansion terms: {terms}")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- Do not treat retrieval hits as factual support.",
            "- Direct evidence items are selected raw evidence text; retrieval rationale is only shown when it adds branch or graph information.",
            "- Use direct evidence for factual claims when support allows it.",
            "- Use summaries and memory units as context unless evidence support is explicit.",
            "- Use graph answer context as candidate relationship context; it is not proof.",
            "- Open expanded evidence/audit views only when evidence, feedback, debug, or repair is requested.",
        ]
    )
    return "\n".join(lines)


def build_s23_answer_context(
    *,
    query_dir: Path | None = None,
    dual_package: dict[str, Any] | None = None,
    interpretation: dict[str, Any] | None = None,
    selected_context: list[dict[str, Any]] | None = None,
    step2_candidates: list[dict[str, Any]] | None = None,
    graph_context: list[dict[str, Any]] | None = None,
    graph_retrieval_package: dict[str, Any] | None = None,
    max_items: int = 16,
) -> dict[str, Any]:
    query_dir = query_dir.resolve() if query_dir else None
    if query_dir is not None:
        dual_package = dual_package or read_json(query_dir / "dual_branch_retrieval_package.json", {})
        interpretation = interpretation or read_json(query_dir / "memory_context_interpretation.json", {})
        selected_context = selected_context if selected_context is not None else read_jsonl(query_dir / "selected_context.jsonl")
        step2_candidates = step2_candidates if step2_candidates is not None else read_jsonl(query_dir / "step2_candidates.jsonl")
        graph_context = graph_context if graph_context is not None else read_jsonl(query_dir / "graph_context.jsonl")
        graph_retrieval_package = graph_retrieval_package or read_json(query_dir / "graph_retrieval_package.json", {})

    dual_package = dual_package or {}
    interpretation = interpretation or {}
    selected_context = selected_context or []
    step2_candidates = step2_candidates or []
    graph_context = graph_context or []
    graph_retrieval_package = graph_retrieval_package or {}
    query_support = dual_package.get("step1_evidence_retrieval", {}).get("query_claim_support_check") or interpretation.get("query_claim_support_check") or {}
    graph_answer_context = build_graph_answer_context(graph_context, graph_retrieval_package)

    answer_items: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in interpretation.get("direct_evidence", [])[:6]:
        if not row.get("resolved", True):
            continue
        item = make_answer_item(
            {
                **row,
                "item_layer": "raw_evidence",
                "support_status": query_support.get("support_status", "not_checked"),
                "evidence_refs": [row.get("evidence_ref")] if row.get("evidence_ref") else row.get("evidence_refs", []),
            },
            source="memory_context_interpretation.direct_evidence",
            query_support=query_support,
            package_section="direct_evidence",
        )
        add_unique_item(answer_items, item, seen)

    step1_branch = dual_package.get("step1_evidence_retrieval", {})
    for section in ("overlap_candidates", "raw_evidence_candidates", "memory_unit_candidates", "summary_context_candidates"):
        for row in step1_branch.get(section, [])[:4]:
            add_unique_item(
                answer_items,
                make_answer_item(row, source="dual_branch.step1", query_support=query_support, package_section=section),
                seen,
            )

    for row in selected_context[:8]:
        add_unique_item(
            answer_items,
            make_answer_item(row, source="step2_selected_context", query_support=query_support, package_section="selected_context"),
            seen,
        )

    if len(answer_items) > max_items:
        answer_items = answer_items[:max_items]
    answer_items = merge_answer_items(answer_items)

    source_candidate_ids = unique_list(
        [
            item.get("audit_pointer", {}).get("source_candidate_id")
            for item in answer_items
            if item.get("audit_pointer", {}).get("source_candidate_id")
        ]
    )
    warnings = unique_list(
        [
            "answer_context_is_token_optimized_not_audit_complete",
            "retrieval_hit_not_support_check",
            *dual_package.get("warnings", []),
            *[warning for item in answer_items for warning in item.get("warnings", [])],
        ]
    )
    return {
        "schema_version": "s2.s23_answer_context.v1",
        "generated_at": utc_now(),
        "mode": "answer",
        "query": dual_package.get("query") or interpretation.get("query", {}),
        "answer_context": answer_items,
        "graph_answer_context": graph_answer_context,
        "warnings": warnings,
        "audit_pointer": {
            "source_package_path": "dual_branch_retrieval_package.json" if query_dir is not None else None,
            "source_interpretation_path": "memory_context_interpretation.json" if query_dir is not None else None,
            "source_selected_context_path": "selected_context.jsonl" if query_dir is not None else None,
            "source_graph_context_path": "graph_context.jsonl" if query_dir is not None else None,
            "source_graph_retrieval_package_path": "graph_retrieval_package.json" if query_dir is not None and graph_retrieval_package else None,
            "source_candidate_ids": source_candidate_ids,
        },
        "audit_summary": dual_branch_summary(dual_package),
        "usage_rules": [
            "This JSON is a lightweight intermediate/debug artifact, not the final prompt payload.",
            "Use s23_prompt_context.md or render_s23_prompt_context() as the default Step 2.3 prompt view.",
            "Do not treat retrieval hits as support checks.",
            "Open the full dual_branch_retrieval_package.json only for audit, feedback, or debugging.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight Step 2.3 answer context for an existing query directory.")
    parser.add_argument("--query-dir", required=True, help="Directory containing Step 2.2 query artifacts.")
    parser.add_argument("--output", default=None, help="Output JSON path. Defaults to <query-dir>/s23_answer_context.json.")
    parser.add_argument("--prompt-output", default=None, help="Prompt-view Markdown path. Defaults to <query-dir>/s23_prompt_context.md.")
    parser.add_argument(
        "--prompt-view",
        default="answer",
        choices=["answer", "why", "audit"],
        help="Prompt rendering mode. answer is default lightweight; why/audit expand graph evidence details.",
    )
    parser.add_argument("--max-items", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    query_dir = Path(args.query_dir).resolve()
    output = Path(args.output).resolve() if args.output else query_dir / "s23_answer_context.json"
    prompt_output = Path(args.prompt_output).resolve() if args.prompt_output else query_dir / "s23_prompt_context.md"
    context = build_s23_answer_context(query_dir=query_dir, max_items=args.max_items)
    write_json(output, context)
    prompt_output.parent.mkdir(parents=True, exist_ok=True)
    prompt_output.write_text(render_s23_prompt_context(context, prompt_view=args.prompt_view).rstrip() + "\n", encoding="utf-8")
    print(str(output))
    print(str(prompt_output))


if __name__ == "__main__":
    main()
