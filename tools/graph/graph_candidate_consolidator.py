"""Consolidate raw v0.3 graph candidates into audit graph tables.

This layer consumes raw entity/relation/claim candidates from schema-guided
graph extraction. It normalizes labels, validates endpoints and provenance,
records explicit merge/review decisions, and emits candidate graph tables.

It does not write graph truth, durable memory, S3 assets, or graph algorithm
outputs.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.graph.graph_construction_packet_builder import (
    file_hash,
    read_json,
    read_jsonl,
    sha256_text,
    stable_id,
    unique_strings,
    write_json,
    write_jsonl,
    write_text,
)


SCHEMA_VERSION = "graph_v03.candidate_consolidation.v0.1"
ENTITY_NORMALIZATION_SCHEMA_VERSION = "graph_v03.entity_normalization_candidate.v0.1"
RELATION_TYPE_NORMALIZATION_SCHEMA_VERSION = "graph_v03.relation_type_normalization.v0.1"
MERGE_DECISION_SCHEMA_VERSION = "graph_v03.merge_decision.v0.1"
ENTITY_RESOLUTION_CANDIDATE_SCHEMA_VERSION = "graph_v03.entity_resolution_candidate.v0.1"
NODE_TABLE_SCHEMA_VERSION = "graph_v03.node_table.v0.2"
EDGE_TABLE_SCHEMA_VERSION = "graph_v03.edge_table.v0.2"
CLAIM_TABLE_SCHEMA_VERSION = "graph_v03.claim_table.v0.1"
EVIDENCE_LINK_SCHEMA_VERSION = "graph_v03.evidence_link.v0.2"
DEFAULT_EXTRACTION_DIR_NAME = "graph_v03_extraction_provider_80"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_consolidation"
DEFAULT_RELATION_POLICY = "configs/graph/graph_relation_type_normalization.generated.v0.3.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_hash(value: str, length: int = 12) -> str:
    return sha256_text(value)[:length]


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


def first_non_empty(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def normalize_whitespace(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_entity_key(value: Any) -> str:
    text = normalize_whitespace(value).lower()
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "-", text)
    return text.strip("-")


def normalize_relation_label(value: Any) -> str:
    text = normalize_whitespace(value).lower()
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "_", text)
    return text.strip("_")


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def load_relation_policy(project_root: Path, policy_path: str | Path) -> dict[str, Any]:
    resolved = resolve_project_path(project_root, policy_path)
    policy = read_json(resolved)
    required = {"schema_version", "policy_id", "canonical_relation_types", "alias_map"}
    missing = sorted(key for key in required if key not in policy)
    if missing:
        raise ValueError("Relation normalization policy missing keys: " + ", ".join(missing))
    return policy


def most_common_non_empty(values: list[str], default: str = "unknown") -> str:
    clean = [value for value in values if value]
    if not clean:
        return default
    return Counter(clean).most_common(1)[0][0]


def row_warnings(*values: Any) -> list[str]:
    return unique_strings(*values)


def candidate_label(row: dict[str, Any]) -> str:
    return first_non_empty(row.get("candidate_text"), row.get("source_node_hint"), row.get("extracted_span"))


def relation_type_for(raw_type: str, policy: dict[str, Any]) -> tuple[str, str, list[str]]:
    warnings: list[str] = []
    raw_key = normalize_whitespace(raw_type).lower()
    alias_map = policy.get("alias_map") or {}
    canonical = policy.get("canonical_relation_types") or {}
    if raw_key in alias_map:
        return str(alias_map[raw_key]), "alias_map", warnings

    normalized = normalize_relation_label(raw_key)
    if normalized in canonical:
        return normalized, "canonical_direct", warnings

    warnings.append("unmapped_relation_type")
    return "related_to_generic", "generic_fallback", warnings


def relation_category_for(relation_type: str, policy: dict[str, Any]) -> str:
    canonical = policy.get("canonical_relation_types") or {}
    meta = canonical.get(relation_type) or {}
    return str(meta.get("category") or "generic")


def relation_schema_sources_for(relation_type: str, policy: dict[str, Any]) -> list[str]:
    canonical = policy.get("canonical_relation_types") or {}
    meta = canonical.get(relation_type) or {}
    return string_list(meta.get("external_sources"))


def endpoint_lookup_key(source_packet_id: str, local_entity_id: str) -> tuple[str, str]:
    return (str(source_packet_id or ""), str(local_entity_id or ""))


def source_hashes(paths: list[Path]) -> dict[str, str | None]:
    return {str(path): file_hash(path) for path in paths}


def evidence_link_rows(
    *,
    owner_id: str,
    owner_kind: str,
    candidate_id: str,
    row: dict[str, Any],
) -> list[dict[str, Any]]:
    evidence_refs = string_list(row.get("evidence_refs"))
    if not evidence_refs:
        evidence_refs = [""]
    rows: list[dict[str, Any]] = []
    for evidence_ref in evidence_refs:
        link_basis = json.dumps(
            {
                "owner_id": owner_id,
                "candidate_id": candidate_id,
                "evidence_ref": evidence_ref,
                "quote": row.get("source_text_quote") or "",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        rows.append(
            {
                "schema_version": EVIDENCE_LINK_SCHEMA_VERSION,
                "link_id": stable_id("graph_evidence_link", link_basis),
                "owner_id": owner_id,
                "owner_kind": owner_kind,
                "candidate_id": candidate_id,
                "evidence_ref": evidence_ref,
                "evidence_refs": evidence_refs if evidence_ref else [],
                "raw_backpointer_refs": string_list(row.get("raw_backpointer_refs")),
                "source_refs": string_list(row.get("source_refs")),
                "source_packet_id": row.get("source_packet_id") or "",
                "source_text_quote": row.get("source_text_quote") or "",
                "source_text_excerpt": row.get("source_text_excerpt") or "",
                "graph_is_not_proof": True,
                "write_permission": False,
                "support_status": "not_checked",
                "warnings": row_warnings(row.get("warnings"), [] if evidence_ref else ["missing_evidence_ref"]),
            }
        )
    return rows


def validate_quote_and_provenance(row: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    if not string_list(row.get("evidence_refs")):
        warnings.append("missing_evidence_refs")
    if not string_list(row.get("raw_backpointer_refs")):
        warnings.append("missing_raw_backpointer_refs")
    quote = normalize_whitespace(row.get("source_text_quote"))
    excerpt = normalize_whitespace(row.get("source_text_excerpt"))
    if not quote:
        warnings.append("missing_source_text_quote")
    elif excerpt and quote not in excerpt:
        warnings.append("quote_not_found_in_excerpt")
    if row.get("evidence_role") == "context":
        warnings.append("context_only_evidence_role")
    if row.get("graph_is_not_proof") is not True:
        warnings.append("graph_is_not_proof_not_true")
    if row.get("write_permission") is not False:
        warnings.append("write_permission_not_false")
    return warnings


def label_token_count(label: str) -> int:
    return len([part for part in re.split(r"[\s\-_]+", label.strip()) if part])


def is_upper_named_label(label: str) -> bool:
    text = label.strip()
    return bool(text) and text[0].isupper()


def entity_quality_for(
    *,
    label: str,
    entity_type: str,
    candidate_count: int,
    warnings: list[str],
) -> tuple[str, list[str]]:
    normalized_type = normalize_relation_label(entity_type)
    token_count = label_token_count(label)
    lower_label = label.strip().lower()
    reasons: list[str] = []

    if "entity_merge_requires_review" in warnings or "entity_type_conflict" in warnings:
        return "review_required", ["entity merge/type conflict requires review"]

    if normalized_type == "person" and is_upper_named_label(label):
        reasons.append("named person-like label")
        return "stable", reasons

    if normalized_type in {"project", "organization", "place", "event"} and candidate_count >= 2:
        reasons.append("repeated concrete entity type")
        return "stable", reasons

    if normalized_type in {"concept", "unknown"} and token_count >= 3 and lower_label == label.strip():
        reasons.append("lowercase multi-token phrase-like entity")
        return "action_phrase", reasons

    if normalized_type in {"concept", "unknown"} and token_count <= 2:
        reasons.append("short generic concept/unknown entity")
        return "generic_fragment", reasons

    if normalized_type == "person" and not is_upper_named_label(label):
        reasons.append("non-named person-like group")
        return "generic_fragment", reasons

    if any(
        warning in warnings
        for warning in (
            "context_dependency_warning",
            "context_only_evidence_role",
            "quote_not_found_in_excerpt",
        )
    ):
        reasons.append("context-dependent evidence or quote")
        return "context_dependent", reasons

    if normalized_type in {"project", "organization", "place", "event", "person"}:
        reasons.append("single concrete entity candidate")
        return "candidate", reasons

    return "candidate", ["default candidate quality"]


def generic_relation_review_for(
    *,
    relation_type: str,
    raw_relation_types: list[str],
    source_entity_type: str,
    target_entity_type: str,
    warnings: list[str],
) -> tuple[str, list[str]]:
    if relation_type != "related_to_generic":
        return "not_generic", []

    reasons: list[str] = []

    if any("attribution" in warning or "model_uncertain" in warning for warning in warnings):
        reasons.append("attribution or model uncertainty warning")
        return "needs_attribution_review", reasons

    if source_entity_type in {"concept", "unknown"} or target_entity_type in {"concept", "unknown"}:
        reasons.append("generic relation touches abstract/unknown endpoint")
        return "low_graph_value", reasons

    if "unmapped_relation_type" in warnings:
        reasons.append("relation label is outside the generated external schema")
        return "needs_schema_extension", reasons

    return "safe_generic", ["generic relation retained conservatively"]


def build_entity_groups(entities: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in entities:
        label = candidate_label(row)
        normalized_key = normalize_entity_key(label)
        if not normalized_key:
            normalized_key = "unresolved-" + short_hash(json.dumps(row, ensure_ascii=False, sort_keys=True))
        groups[normalized_key].append(row)
    return groups


def meaningful_entity_type_conflict(type_hints: list[str]) -> bool:
    strong_types = {value for value in type_hints if value and value != "unknown"}
    return len(strong_types) > 1


def entity_normalization_rows(
    groups: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    normalization_rows: list[dict[str, Any]] = []
    merge_decisions: list[dict[str, Any]] = []
    node_rows: list[dict[str, Any]] = []
    candidate_to_node_id: dict[str, str] = {}
    human_id = 1

    for normalized_key, rows in sorted(groups.items()):
        labels = [candidate_label(row) for row in rows]
        candidate_ids = [str(row.get("candidate_id")) for row in rows if row.get("candidate_id")]
        type_hints = [str(row.get("entity_type_hint") or "") for row in rows]
        primary_label = most_common_non_empty(labels, default=normalized_key)
        primary_type = most_common_non_empty(type_hints)
        node_id = stable_id("graph_node", f"{normalized_key}|{primary_type}")
        type_conflict = meaningful_entity_type_conflict(type_hints)
        validation_warnings = row_warnings(
            *(row.get("warnings") for row in rows),
            *(validate_quote_and_provenance(row) for row in rows),
            ["entity_type_conflict"] if type_conflict else [],
        )
        decision = "merge" if len(rows) > 1 and not type_conflict else "keep_separate"
        if len(rows) > 1 and type_conflict:
            decision = "review"

        normalization_rows.append(
            {
                "schema_version": ENTITY_NORMALIZATION_SCHEMA_VERSION,
                "normalization_id": stable_id("graph_entity_norm", normalized_key),
                "normalized_key": normalized_key,
                "canonical_label_candidate": primary_label,
                "entity_type_hint": primary_type,
                "candidate_ids": candidate_ids,
                "candidate_count": len(candidate_ids),
                "normalization_method": "exact_normalized_label",
                "decision_recommendation": decision,
                "graph_is_not_proof": True,
                "write_permission": False,
                "support_status": "not_checked",
                "warnings": validation_warnings,
            }
        )

        output_node_id = node_id if decision in {"merge", "keep_separate"} else ""
        merge_decisions.append(
            {
                "schema_version": MERGE_DECISION_SCHEMA_VERSION,
                "decision_id": stable_id("graph_merge_decision", f"entity|{normalized_key}"),
                "decision_kind": "entity_exact_normalized_label",
                "decision": decision,
                "candidate_ids": candidate_ids,
                "output_node_id": output_node_id,
                "normalized_key": normalized_key,
                "reason": "exact normalized entity label grouping; no semantic alias merge performed",
                "graph_is_not_proof": True,
                "write_permission": False,
                "support_status": "not_checked",
                "warnings": validation_warnings,
            }
        )

        if decision == "review":
            for row in rows:
                candidate_id = str(row.get("candidate_id") or "")
                row_node_id = stable_id("graph_node", f"{normalized_key}|{candidate_id}")
                row_warnings_list = row_warnings(
                    row.get("warnings"),
                    validate_quote_and_provenance(row),
                    ["entity_merge_requires_review"],
                )
                entity_quality_hint, entity_quality_reasons = entity_quality_for(
                    label=candidate_label(row),
                    entity_type=str(row.get("entity_type_hint") or "unknown"),
                    candidate_count=1,
                    warnings=row_warnings_list,
                )
                if candidate_id:
                    candidate_to_node_id[candidate_id] = row_node_id
                node_rows.append(
                    {
                        "schema_version": NODE_TABLE_SCHEMA_VERSION,
                        "node_id": row_node_id,
                        "human_readable_id": human_id,
                        "label": candidate_label(row),
                        "normalized_key": normalized_key,
                        "entity_type": str(row.get("entity_type_hint") or "unknown"),
                        "candidate_ids": [candidate_id] if candidate_id else [],
                        "source_packet_ids": unique_strings(row.get("source_packet_id")),
                        "evidence_refs": unique_strings(row.get("evidence_refs")),
                        "raw_backpointer_refs": unique_strings(row.get("raw_backpointer_refs")),
                        "source_refs": unique_strings(row.get("source_refs")),
                        "source_text_quotes": unique_strings(row.get("source_text_quote")),
                        "summary": "",
                        "description": first_non_empty(row.get("description")),
                        "entity_quality_hint": entity_quality_hint,
                        "entity_quality_reasons": entity_quality_reasons,
                        "helper_fields_not_evidence_truth": True,
                        "graph_is_not_proof": True,
                        "write_permission": False,
                        "support_status": "not_checked",
                        "warnings": row_warnings_list,
                    }
                )
                human_id += 1
        else:
            for candidate_id in candidate_ids:
                candidate_to_node_id[candidate_id] = node_id
            entity_quality_hint, entity_quality_reasons = entity_quality_for(
                label=primary_label,
                entity_type=primary_type,
                candidate_count=len(candidate_ids),
                warnings=validation_warnings,
            )
            node_rows.append(
                {
                    "schema_version": NODE_TABLE_SCHEMA_VERSION,
                    "node_id": node_id,
                    "human_readable_id": human_id,
                    "label": primary_label,
                    "normalized_key": normalized_key,
                    "entity_type": primary_type,
                    "candidate_ids": candidate_ids,
                    "source_packet_ids": unique_strings(*(row.get("source_packet_id") for row in rows)),
                    "evidence_refs": unique_strings(*(row.get("evidence_refs") for row in rows)),
                    "raw_backpointer_refs": unique_strings(*(row.get("raw_backpointer_refs") for row in rows)),
                    "source_refs": unique_strings(*(row.get("source_refs") for row in rows)),
                    "source_text_quotes": unique_strings(*(row.get("source_text_quote") for row in rows)),
                    "summary": "",
                    "description": first_non_empty(*(row.get("description") for row in rows)),
                    "entity_quality_hint": entity_quality_hint,
                    "entity_quality_reasons": entity_quality_reasons,
                    "helper_fields_not_evidence_truth": True,
                    "graph_is_not_proof": True,
                    "write_permission": False,
                    "support_status": "not_checked",
                    "warnings": validation_warnings,
                }
            )
            human_id += 1

    return normalization_rows, merge_decisions, node_rows, candidate_to_node_id


def build_endpoint_indexes(
    entities: list[dict[str, Any]],
    candidate_to_node_id: dict[str, str],
) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str], str]]:
    local_to_candidate: dict[tuple[str, str], str] = {}
    hint_to_candidate: dict[tuple[str, str], str] = {}
    hint_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for row in entities:
        candidate_id = str(row.get("candidate_id") or "")
        if not candidate_id:
            continue
        source_packet_id = str(row.get("source_packet_id") or "")
        local_id = str(row.get("local_entity_id") or "")
        if source_packet_id and local_id:
            local_to_candidate[endpoint_lookup_key(source_packet_id, local_id)] = candidate_id
        hint = normalize_entity_key(first_non_empty(row.get("source_node_hint"), row.get("candidate_text")))
        if source_packet_id and hint:
            hint_groups[(source_packet_id, hint)].append(candidate_id)

    for key, candidate_ids in hint_groups.items():
        unique = [candidate_id for candidate_id in dict.fromkeys(candidate_ids) if candidate_id in candidate_to_node_id]
        if len(unique) == 1:
            hint_to_candidate[key] = unique[0]
            continue
        unique_node_ids = {
            candidate_to_node_id[candidate_id]
            for candidate_id in unique
            if candidate_to_node_id.get(candidate_id)
        }
        if len(unique_node_ids) == 1:
            hint_to_candidate[key] = unique[0]
    return local_to_candidate, hint_to_candidate


def resolve_relation_endpoint(
    relation: dict[str, Any],
    *,
    endpoint: str,
    local_to_candidate: dict[tuple[str, str], str],
    hint_to_candidate: dict[tuple[str, str], str],
    candidate_to_node_id: dict[str, str],
) -> tuple[str, str, str]:
    source_packet_id = str(relation.get("source_packet_id") or "")
    if endpoint == "source":
        local_id = str(relation.get("source_local_entity_id") or "")
        hint = relation.get("source_node_hint")
    else:
        local_id = str(relation.get("target_local_entity_id") or "")
        hint = relation.get("target_node_hint")

    candidate_id = ""
    status = "unresolved"
    if source_packet_id and local_id:
        candidate_id = local_to_candidate.get(endpoint_lookup_key(source_packet_id, local_id), "")
        if candidate_id:
            status = "local_entity_id"

    if not candidate_id and source_packet_id and hint:
        candidate_id = hint_to_candidate.get((source_packet_id, normalize_entity_key(hint)), "")
        if candidate_id:
            status = "unique_hint_in_packet"

    node_id = candidate_to_node_id.get(candidate_id, "")
    if not node_id:
        status = "unresolved"
    return candidate_id, node_id, status


def resolve_claim_subject(
    claim: dict[str, Any],
    *,
    local_to_candidate: dict[tuple[str, str], str],
    hint_to_candidate: dict[tuple[str, str], str],
    candidate_to_node_id: dict[str, str],
) -> tuple[str, str, str]:
    source_packet_id = str(claim.get("source_packet_id") or "")
    local_id = str(claim.get("subject_local_entity_id") or "")
    hint = claim.get("source_node_hint")

    candidate_id = ""
    status = "unresolved"
    if source_packet_id and local_id:
        candidate_id = local_to_candidate.get(endpoint_lookup_key(source_packet_id, local_id), "")
        if candidate_id:
            status = "local_entity_id"

    if not candidate_id and source_packet_id and hint:
        candidate_id = hint_to_candidate.get((source_packet_id, normalize_entity_key(hint)), "")
        if candidate_id:
            status = "unique_hint_in_packet"

    node_id = candidate_to_node_id.get(candidate_id, "")
    if not node_id:
        status = "unresolved"
    return candidate_id, node_id, status


def relation_rows(
    relations: list[dict[str, Any]],
    *,
    policy: dict[str, Any],
    local_to_candidate: dict[tuple[str, str], str],
    hint_to_candidate: dict[tuple[str, str], str],
    candidate_to_node_id: dict[str, str],
    node_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    relation_type_rows: list[dict[str, Any]] = []
    merge_decisions: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    evidence_links: list[dict[str, Any]] = []
    edge_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in relations:
        raw_type = str(row.get("relation_type_hint") or "")
        normalized_type, method, type_warnings = relation_type_for(raw_type, policy)
        relation_category = relation_category_for(normalized_type, policy)
        relation_schema_sources = relation_schema_sources_for(normalized_type, policy)
        relation_candidate_id = str(row.get("candidate_id") or "")
        source_candidate_id, source_node_id, source_status = resolve_relation_endpoint(
            row,
            endpoint="source",
            local_to_candidate=local_to_candidate,
            hint_to_candidate=hint_to_candidate,
            candidate_to_node_id=candidate_to_node_id,
        )
        target_candidate_id, target_node_id, target_status = resolve_relation_endpoint(
            row,
            endpoint="target",
            local_to_candidate=local_to_candidate,
            hint_to_candidate=hint_to_candidate,
            candidate_to_node_id=candidate_to_node_id,
        )
        validation_warnings = row_warnings(row.get("warnings"), type_warnings, validate_quote_and_provenance(row))
        if not source_node_id:
            validation_warnings.append("unresolved_source_endpoint")
        if not target_node_id:
            validation_warnings.append("unresolved_target_endpoint")
        if source_node_id and target_node_id and source_node_id == target_node_id:
            validation_warnings.append("self_loop_candidate")
        source_node = node_by_id.get(source_node_id, {})
        target_node = node_by_id.get(target_node_id, {})
        generic_review_hint, generic_review_reasons = generic_relation_review_for(
            relation_type=normalized_type,
            raw_relation_types=[raw_type],
            source_entity_type=str(source_node.get("entity_type") or "unknown"),
            target_entity_type=str(target_node.get("entity_type") or "unknown"),
            warnings=validation_warnings,
        )

        relation_type_rows.append(
            {
                "schema_version": RELATION_TYPE_NORMALIZATION_SCHEMA_VERSION,
                "normalization_id": stable_id("graph_relation_type_norm", relation_candidate_id or json.dumps(row, sort_keys=True)),
                "relation_candidate_id": relation_candidate_id,
                "raw_relation_type": raw_type,
                "normalized_relation_type": normalized_type,
                "relation_category": relation_category,
                "relation_schema_sources": relation_schema_sources,
                "normalization_method": method,
                "source_candidate_id": source_candidate_id,
                "target_candidate_id": target_candidate_id,
                "source_endpoint_status": source_status,
                "target_endpoint_status": target_status,
                "generic_relation_review_hint": generic_review_hint,
                "generic_relation_review_reasons": generic_review_reasons,
                "graph_is_not_proof": True,
                "write_permission": False,
                "support_status": "not_checked",
                "warnings": validation_warnings,
            }
        )

        if not source_node_id or not target_node_id or source_node_id == target_node_id:
            merge_decisions.append(
                {
                    "schema_version": MERGE_DECISION_SCHEMA_VERSION,
                    "decision_id": stable_id("graph_merge_decision", f"relation_repair|{relation_candidate_id}"),
                    "decision_kind": "relation_endpoint_validation",
                    "decision": "repair_or_review",
                    "candidate_ids": [relation_candidate_id] if relation_candidate_id else [],
                    "output_edge_id": "",
                    "reason": "relation endpoint validation did not produce two distinct resolved nodes",
                    "graph_is_not_proof": True,
                    "write_permission": False,
                    "support_status": "not_checked",
                    "warnings": validation_warnings,
                }
            )
            continue

        temporal_key = json.dumps(row.get("temporal_scope") or {}, ensure_ascii=False, sort_keys=True)
        group_key = json.dumps(
            {
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "relation_type": normalized_type,
                "source_perspective": row.get("source_perspective") or "",
                "attribution_status": row.get("attribution_status") or "",
                "temporal_scope": temporal_key,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        edge_groups[group_key].append(
            {
                "row": row,
                "source_node_id": source_node_id,
                "target_node_id": target_node_id,
                "source_candidate_id": source_candidate_id,
                "target_candidate_id": target_candidate_id,
                "normalized_relation_type": normalized_type,
                "relation_category": relation_category,
                "relation_schema_sources": relation_schema_sources,
                "warnings": validation_warnings,
            }
        )

    for human_id, (group_key, grouped) in enumerate(sorted(edge_groups.items()), 1):
        parsed_key = json.loads(group_key)
        edge_id = stable_id("graph_edge", group_key)
        candidate_ids = [str(item["row"].get("candidate_id")) for item in grouped if item["row"].get("candidate_id")]
        warnings = row_warnings(*(item["warnings"] for item in grouped))
        merge_decision = "merge" if len(grouped) > 1 else "keep_separate"
        merge_decisions.append(
            {
                "schema_version": MERGE_DECISION_SCHEMA_VERSION,
                "decision_id": stable_id("graph_merge_decision", f"relation|{group_key}"),
                "decision_kind": "relation_structural_group",
                "decision": merge_decision,
                "candidate_ids": candidate_ids,
                "output_edge_id": edge_id,
                "reason": "grouped by source, target, normalized relation type, perspective, attribution, and temporal scope",
                "graph_is_not_proof": True,
                "write_permission": False,
                "support_status": "not_checked",
                "warnings": warnings,
            }
        )
        raw_relation_types = unique_strings(*(item["row"].get("relation_type_hint") for item in grouped))
        source_node = node_by_id.get(parsed_key["source_node_id"], {})
        target_node = node_by_id.get(parsed_key["target_node_id"], {})
        generic_review_hint, generic_review_reasons = generic_relation_review_for(
            relation_type=parsed_key["relation_type"],
            raw_relation_types=raw_relation_types,
            source_entity_type=str(source_node.get("entity_type") or "unknown"),
            target_entity_type=str(target_node.get("entity_type") or "unknown"),
            warnings=warnings,
        )
        edge_rows.append(
            {
                "schema_version": EDGE_TABLE_SCHEMA_VERSION,
                "edge_id": edge_id,
                "human_readable_id": human_id,
                "source_node_id": parsed_key["source_node_id"],
                "target_node_id": parsed_key["target_node_id"],
                "source_label": source_node.get("label", ""),
                "target_label": target_node.get("label", ""),
                "source_entity_type": source_node.get("entity_type", "unknown"),
                "target_entity_type": target_node.get("entity_type", "unknown"),
                "relation_type": parsed_key["relation_type"],
                "relation_category": relation_category_for(parsed_key["relation_type"], policy),
                "relation_schema_sources": relation_schema_sources_for(parsed_key["relation_type"], policy),
                "raw_relation_types": raw_relation_types,
                "candidate_ids": candidate_ids,
                "source_candidate_ids": unique_strings(*(item["source_candidate_id"] for item in grouped)),
                "target_candidate_ids": unique_strings(*(item["target_candidate_id"] for item in grouped)),
                "source_packet_ids": unique_strings(*(item["row"].get("source_packet_id") for item in grouped)),
                "source_perspective": parsed_key["source_perspective"],
                "attribution_status": parsed_key["attribution_status"],
                "temporal_scope": json.loads(parsed_key["temporal_scope"]) if parsed_key["temporal_scope"] else {},
                "evidence_refs": unique_strings(*(item["row"].get("evidence_refs") for item in grouped)),
                "raw_backpointer_refs": unique_strings(*(item["row"].get("raw_backpointer_refs") for item in grouped)),
                "source_refs": unique_strings(*(item["row"].get("source_refs") for item in grouped)),
                "source_text_quotes": unique_strings(*(item["row"].get("source_text_quote") for item in grouped)),
                "description": first_non_empty(*(item["row"].get("relation_description") for item in grouped), *(item["row"].get("candidate_text") for item in grouped)),
                "summary": "",
                "weight": 1.0,
                "evidence_count": len(unique_strings(*(item["row"].get("evidence_refs") for item in grouped))),
                "confidence_hint": most_common_non_empty([str(item["row"].get("confidence_hint") or "") for item in grouped], default="unknown"),
                "generic_relation_review_hint": generic_review_hint,
                "generic_relation_review_reasons": generic_review_reasons,
                "helper_fields_not_evidence_truth": True,
                "endpoint_labels_are_helper_fields_not_evidence_truth": True,
                "graph_is_not_proof": True,
                "write_permission": False,
                "support_status": "not_checked",
                "warnings": warnings,
            }
        )
        for item in grouped:
            evidence_links.extend(
                evidence_link_rows(
                    owner_id=edge_id,
                    owner_kind="edge",
                    candidate_id=str(item["row"].get("candidate_id") or ""),
                    row=item["row"],
                )
            )

    return relation_type_rows, merge_decisions, edge_rows, evidence_links


def claim_rows(
    claims: list[dict[str, Any]],
    *,
    local_to_candidate: dict[tuple[str, str], str],
    hint_to_candidate: dict[tuple[str, str], str],
    candidate_to_node_id: dict[str, str],
    node_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    evidence_links: list[dict[str, Any]] = []
    for human_id, claim in enumerate(claims, 1):
        candidate_id = str(claim.get("candidate_id") or "")
        subject_candidate_id, subject_node_id, subject_status = resolve_claim_subject(
            claim,
            local_to_candidate=local_to_candidate,
            hint_to_candidate=hint_to_candidate,
            candidate_to_node_id=candidate_to_node_id,
        )
        subject_node = node_by_id.get(subject_node_id, {})
        validation_warnings = row_warnings(claim.get("warnings"), validate_quote_and_provenance(claim))
        if not subject_node_id:
            validation_warnings.append("unresolved_claim_subject")
        claim_id = stable_id("graph_claim", candidate_id or json.dumps(claim, ensure_ascii=False, sort_keys=True))
        rows.append(
            {
                "schema_version": CLAIM_TABLE_SCHEMA_VERSION,
                "claim_id": claim_id,
                "human_readable_id": human_id,
                "candidate_id": candidate_id,
                "candidate_text": claim.get("candidate_text") or "",
                "claim_type_hint": claim.get("claim_type_hint") or "",
                "claim_status": claim.get("claim_status") or "",
                "subject_candidate_id": subject_candidate_id,
                "subject_node_id": subject_node_id,
                "subject_label": subject_node.get("label", claim.get("source_node_hint") or ""),
                "subject_entity_type": subject_node.get("entity_type", claim.get("entity_type_hint") or "unknown"),
                "subject_endpoint_status": subject_status,
                "source_packet_id": claim.get("source_packet_id") or "",
                "source_perspective": claim.get("source_perspective") or "",
                "attribution_status": claim.get("attribution_status") or "",
                "inference_level_hint": claim.get("inference_level_hint") or "",
                "temporal_scope": claim.get("temporal_scope") or {},
                "evidence_refs": string_list(claim.get("evidence_refs")),
                "raw_backpointer_refs": string_list(claim.get("raw_backpointer_refs")),
                "source_refs": string_list(claim.get("source_refs")),
                "source_text_quote": claim.get("source_text_quote") or "",
                "source_text_excerpt": claim.get("source_text_excerpt") or "",
                "confidence_hint": claim.get("confidence_hint") or "unknown",
                "helper_fields_not_evidence_truth": True,
                "graph_is_not_proof": True,
                "write_permission": False,
                "support_status": "not_checked",
                "warnings": validation_warnings,
            }
        )
        evidence_links.extend(
            evidence_link_rows(
                owner_id=claim_id,
                owner_kind="claim",
                candidate_id=candidate_id,
                row=claim,
            )
        )
    return rows, evidence_links


def node_evidence_links(node_rows: list[dict[str, Any]], entities_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    for node in node_rows:
        for candidate_id in node.get("candidate_ids") or []:
            row = entities_by_id.get(candidate_id)
            if not row:
                continue
            links.extend(
                evidence_link_rows(
                    owner_id=node["node_id"],
                    owner_kind="node",
                    candidate_id=candidate_id,
                    row=row,
                )
            )
    return links


def node_label_tokens(node: dict[str, Any]) -> set[str]:
    return {part for part in normalize_entity_key(node.get("label") or "").split("-") if len(part) > 1}


def node_ordered_label_tokens(node: dict[str, Any]) -> list[str]:
    return [part for part in normalize_entity_key(node.get("label") or "").split("-") if len(part) > 1]


def confidence_for_score(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "medium"
    return "low"


def entity_resolution_row(
    *,
    candidate_source: str,
    node_ids: list[str],
    labels: list[str],
    entity_types: list[str],
    candidate_ids: list[str],
    score: float,
    signals: list[str],
    evidence_refs: list[str],
    raw_backpointer_refs: list[str],
    source_refs: list[str],
    source_edge_refs: list[str],
    neighbor_node_refs: list[str],
    source_merge_candidate_id: str = "",
    normalized_key: str = "",
) -> dict[str, Any]:
    basis = json.dumps(
        {
            "candidate_source": candidate_source,
            "node_ids": sorted(node_ids),
            "candidate_ids": sorted(candidate_ids),
            "source_merge_candidate_id": source_merge_candidate_id,
            "signals": sorted(signals),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    warnings = [
        "entity_resolution_candidate_only",
        "do_not_silently_merge_entities",
    ]
    if not evidence_refs:
        warnings.append("missing_direct_shared_evidence")
    if score < 0.45:
        warnings.append("low_confidence_candidate")
    return {
        "schema_version": ENTITY_RESOLUTION_CANDIDATE_SCHEMA_VERSION,
        "resolution_candidate_id": stable_id("graph_entity_resolution_candidate", basis),
        "candidate_kind": "graph_entity_resolution_candidate",
        "candidate_source": candidate_source,
        "source_merge_candidate_id": source_merge_candidate_id,
        "normalized_key": normalized_key,
        "node_ids": node_ids,
        "labels": labels,
        "entity_types": entity_types,
        "candidate_ids": candidate_ids,
        "resolution_status": "candidate_only",
        "recommended_action": "review_before_merge",
        "score": round(score, 6),
        "confidence_hint": confidence_for_score(score),
        "signals": signals,
        "evidence_refs": evidence_refs,
        "raw_backpointer_refs": raw_backpointer_refs,
        "source_refs": source_refs,
        "source_edge_refs": source_edge_refs,
        "neighbor_node_refs": neighbor_node_refs,
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
        "warnings": warnings,
    }


def graph_context_by_node(edge_rows: list[dict[str, Any]]) -> dict[str, dict[str, list[str] | set[str]]]:
    contexts: dict[str, dict[str, list[str] | set[str]]] = defaultdict(
        lambda: {
            "neighbor_node_refs": set(),
            "relation_contexts": set(),
            "source_edge_refs": [],
        }
    )
    for edge in edge_rows:
        edge_id = str(edge.get("edge_id") or "")
        source_node_id = str(edge.get("source_node_id") or "")
        target_node_id = str(edge.get("target_node_id") or "")
        relation_type = str(edge.get("relation_type") or "")
        if not source_node_id or not target_node_id:
            continue
        source_context = contexts[source_node_id]
        target_context = contexts[target_node_id]
        source_context["neighbor_node_refs"].add(target_node_id)  # type: ignore[union-attr]
        target_context["neighbor_node_refs"].add(source_node_id)  # type: ignore[union-attr]
        source_context["relation_contexts"].add(f"out|{relation_type}|{target_node_id}")  # type: ignore[union-attr]
        target_context["relation_contexts"].add(f"in|{relation_type}|{source_node_id}")  # type: ignore[union-attr]
        if edge_id:
            source_context["source_edge_refs"].append(edge_id)  # type: ignore[union-attr]
            target_context["source_edge_refs"].append(edge_id)  # type: ignore[union-attr]
    return contexts


def entity_resolution_candidate_rows(
    *,
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    merge_candidates: list[dict[str, Any]],
    entities_by_id: dict[str, dict[str, Any]],
    candidate_to_node_id: dict[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, ...]] = set()
    node_by_id = {str(row.get("node_id")): row for row in node_rows if row.get("node_id")}
    graph_contexts = graph_context_by_node(edge_rows)

    for source_row in merge_candidates:
        candidate_ids = unique_strings(source_row.get("candidate_ids"))
        node_ids = unique_strings(*(candidate_to_node_id.get(candidate_id, "") for candidate_id in candidate_ids))
        labels = unique_strings(*(candidate_label(entities_by_id.get(candidate_id, {})) for candidate_id in candidate_ids))
        entity_types = unique_strings(*(entities_by_id.get(candidate_id, {}).get("entity_type_hint") for candidate_id in candidate_ids))
        if len(candidate_ids) < 2 or len(node_ids) < 2:
            continue
        pair_key = tuple(sorted(node_ids))
        if len(node_ids) == 2:
            seen_pairs.add(pair_key)
        rows.append(
            entity_resolution_row(
                candidate_source="upstream_merge_candidate",
                source_merge_candidate_id=str(source_row.get("merge_candidate_id") or source_row.get("candidate_id") or ""),
                normalized_key=str(source_row.get("normalized_key") or ""),
                node_ids=node_ids,
                labels=labels,
                entity_types=entity_types,
                candidate_ids=candidate_ids,
                score=0.85 if len(node_ids) <= 1 else 0.65,
                signals=unique_strings("upstream_merge_candidate", source_row.get("merge_basis") or source_row.get("signals")),
                evidence_refs=unique_strings(*(entities_by_id.get(candidate_id, {}).get("evidence_refs") for candidate_id in candidate_ids)),
                raw_backpointer_refs=unique_strings(*(entities_by_id.get(candidate_id, {}).get("raw_backpointer_refs") for candidate_id in candidate_ids)),
                source_refs=unique_strings(*(entities_by_id.get(candidate_id, {}).get("source_refs") for candidate_id in candidate_ids)),
                source_edge_refs=[],
                neighbor_node_refs=[],
            )
        )

    for index, left in enumerate(node_rows):
        left_id = str(left.get("node_id") or "")
        if not left_id:
            continue
        for right in node_rows[index + 1 :]:
            right_id = str(right.get("node_id") or "")
            if not right_id:
                continue
            pair_key = tuple(sorted([left_id, right_id]))
            if pair_key in seen_pairs:
                continue

            left_type = str(left.get("entity_type") or "unknown")
            right_type = str(right.get("entity_type") or "unknown")
            if meaningful_entity_type_conflict([left_type, right_type]):
                continue
            if "action_phrase" in {str(left.get("entity_quality_hint") or ""), str(right.get("entity_quality_hint") or "")}:
                continue

            left_tokens = node_label_tokens(left)
            right_tokens = node_label_tokens(right)
            left_ordered_tokens = node_ordered_label_tokens(left)
            right_ordered_tokens = node_ordered_label_tokens(right)
            token_overlap = left_tokens & right_tokens
            left_key = str(left.get("normalized_key") or "")
            right_key = str(right.get("normalized_key") or "")
            left_context = graph_contexts.get(left_id, {})
            right_context = graph_contexts.get(right_id, {})
            shared_neighbors = set(left_context.get("neighbor_node_refs") or set()) & set(right_context.get("neighbor_node_refs") or set())
            shared_relation_contexts = set(left_context.get("relation_contexts") or set()) & set(right_context.get("relation_contexts") or set())
            shared_evidence_refs = set(string_list(left.get("evidence_refs"))) & set(string_list(right.get("evidence_refs")))
            shared_raw_refs = set(string_list(left.get("raw_backpointer_refs"))) & set(string_list(right.get("raw_backpointer_refs")))
            shared_source_refs = set(string_list(left.get("source_refs"))) & set(string_list(right.get("source_refs")))

            signals: list[str] = []
            score = 0.0
            if left_type == right_type and left_type != "unknown":
                signals.append("same_entity_type")
                score += 0.15
            if shared_evidence_refs:
                signals.append("shared_evidence_refs")
                score += 0.25
            if shared_raw_refs:
                signals.append("shared_raw_backpointer_refs")
                score += 0.2
            if token_overlap:
                signals.append("label_token_overlap")
                score += 0.25
            if left_key and right_key and (left_key in right_key or right_key in left_key):
                signals.append("normalized_label_containment")
                score += 0.25
            if left_ordered_tokens and right_ordered_tokens and left_ordered_tokens[-1] == right_ordered_tokens[-1]:
                signals.append("terminal_label_token_overlap")
                score += 0.15
            if shared_neighbors:
                signals.append("shared_graph_neighbors")
                score += 0.15
            if shared_relation_contexts:
                signals.append("shared_relation_contexts")
                score += 0.2
            if shared_source_refs:
                signals.append("shared_source_refs")

            strong_signals = [
                signal
                for signal in signals
                if signal not in {"same_entity_type", "shared_source_refs", "shared_graph_neighbors"}
            ]
            lexical_or_structural_signals = {
                "label_token_overlap",
                "normalized_label_containment",
                "terminal_label_token_overlap",
                "shared_relation_contexts",
            }
            if not (set(signals) & lexical_or_structural_signals):
                continue
            if str(left.get("entity_quality_hint") or "") == "generic_fragment" and not (set(signals) & lexical_or_structural_signals):
                continue
            if str(right.get("entity_quality_hint") or "") == "generic_fragment" and not (set(signals) & lexical_or_structural_signals):
                continue
            if len(strong_signals) < 2 or score < 0.5:
                continue

            seen_pairs.add(pair_key)
            neighbor_refs = unique_strings(list(shared_neighbors))
            source_edge_refs = unique_strings(left_context.get("source_edge_refs"), right_context.get("source_edge_refs"))
            rows.append(
                entity_resolution_row(
                    candidate_source="derived_node_pair",
                    node_ids=[left_id, right_id],
                    labels=unique_strings(left.get("label"), right.get("label")),
                    entity_types=unique_strings(left_type, right_type),
                    candidate_ids=unique_strings(left.get("candidate_ids"), right.get("candidate_ids")),
                    score=min(score, 1.0),
                    signals=unique_strings(signals),
                    evidence_refs=unique_strings(list(shared_evidence_refs)),
                    raw_backpointer_refs=unique_strings(list(shared_raw_refs)),
                    source_refs=unique_strings(list(shared_source_refs)),
                    source_edge_refs=source_edge_refs,
                    neighbor_node_refs=neighbor_refs,
                )
            )

    return rows


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["counts"]
    lines = [
        "# v0.3 Graph Candidate Consolidation Report",
        "",
        f"- workspace: `{manifest['workspace_id']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        "- graph_is_not_proof: `true`",
        "- write_permission: `false`",
        "- support_status: `not_checked`",
        "",
        "## Counts",
        "",
        f"- entity_candidates: {counts['entity_candidate_count']}",
        f"- relation_candidates: {counts['relation_candidate_count']}",
        f"- claim_candidates: {counts['claim_candidate_count']}",
        f"- entity_normalization_rows: {counts['entity_normalization_count']}",
        f"- entity_resolution_candidates: {counts['entity_resolution_candidate_count']}",
        f"- relation_type_normalization_rows: {counts['relation_type_normalization_count']}",
        f"- merge_decisions: {counts['merge_decision_count']}",
        f"- graph_nodes: {counts['node_count']}",
        f"- graph_edges: {counts['edge_count']}",
        f"- graph_claims: {counts['claim_table_count']}",
        f"- evidence_links: {counts['evidence_link_count']}",
        f"- unresolved_endpoint_relations: {counts['unresolved_endpoint_relation_count']}",
        f"- unresolved_claim_subjects: {counts['unresolved_claim_subject_count']}",
        f"- missing_evidence_rows: {counts['missing_evidence_row_count']}",
        f"- generic_relation_ratio: {counts['generic_relation_ratio']}",
        f"- entity_quality_counts: `{json.dumps(counts['entity_quality_counts'], ensure_ascii=False, sort_keys=True)}`",
        f"- generic_relation_review_counts: `{json.dumps(counts['generic_relation_review_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Quality Signals",
        "",
        "- Edge endpoint labels are helper audit fields, not evidence truth.",
        "- Claim rows are candidate/covariate material; they are not graph truth.",
        "- Generic relation ratio should be inspected before graph algorithms run.",
        "- Entity quality hints and generic relation review hints are routing/audit signals, not truth.",
        "- Entity resolution candidates are review prompts only; they do not silently merge graph nodes.",
        "",
        "## Boundary",
        "",
        "- Candidate graph tables are audit/projection material only.",
        "- Summaries and descriptions are helper fields, not evidence truth.",
        "- No NetworkX metrics, graph truth, S3, durable memory, or support-checker authority is produced.",
        "",
    ]
    write_text(path, "\n".join(lines))


def consolidate_graph_candidates(
    workspace: Path,
    *,
    extraction_dir: Path | None = None,
    output_dir: Path | None = None,
    relation_policy_path: str | Path = DEFAULT_RELATION_POLICY,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    project_root = Path(__file__).resolve().parents[2]
    extraction_dir = (extraction_dir or workspace / DEFAULT_EXTRACTION_DIR_NAME).resolve()
    output_dir = (output_dir or workspace / DEFAULT_OUTPUT_DIR_NAME).resolve()
    policy_path = resolve_project_path(project_root, relation_policy_path)
    policy = load_relation_policy(project_root, policy_path)

    entity_path = extraction_dir / "graph_entity_candidates.jsonl"
    relation_path = extraction_dir / "graph_relation_candidates.jsonl"
    claim_path = extraction_dir / "graph_claim_candidates.jsonl"
    merge_candidate_path = extraction_dir / "graph_merge_candidates.jsonl"
    extraction_manifest_path = extraction_dir / "graph_extraction_manifest.json"

    entities = read_jsonl(entity_path)
    relations = read_jsonl(relation_path)
    claims = read_jsonl(claim_path)
    merge_candidates = read_jsonl(merge_candidate_path)
    extraction_manifest = read_json(extraction_manifest_path)

    groups = build_entity_groups(entities)
    entity_norm_rows, entity_merge_decisions, node_rows, candidate_to_node_id = entity_normalization_rows(groups)
    local_to_candidate, hint_to_candidate = build_endpoint_indexes(entities, candidate_to_node_id)
    node_by_id = {str(row.get("node_id")): row for row in node_rows if row.get("node_id")}
    relation_type_rows, relation_merge_decisions, edge_rows, edge_links = relation_rows(
        relations,
        policy=policy,
        local_to_candidate=local_to_candidate,
        hint_to_candidate=hint_to_candidate,
        candidate_to_node_id=candidate_to_node_id,
        node_by_id=node_by_id,
    )
    claim_table_rows, claim_links = claim_rows(
        claims,
        local_to_candidate=local_to_candidate,
        hint_to_candidate=hint_to_candidate,
        candidate_to_node_id=candidate_to_node_id,
        node_by_id=node_by_id,
    )
    entities_by_id = {str(row.get("candidate_id")): row for row in entities if row.get("candidate_id")}
    evidence_links = node_evidence_links(node_rows, entities_by_id) + edge_links + claim_links
    entity_resolution_candidates = entity_resolution_candidate_rows(
        node_rows=node_rows,
        edge_rows=edge_rows,
        merge_candidates=merge_candidates,
        entities_by_id=entities_by_id,
        candidate_to_node_id=candidate_to_node_id,
    )
    merge_decisions = entity_merge_decisions + relation_merge_decisions

    missing_evidence_count = sum(
        1
        for row in [*entities, *relations, *claims]
        if not string_list(row.get("evidence_refs"))
    )
    unresolved_endpoint_count = sum(
        1
        for row in relation_type_rows
        if row["source_endpoint_status"] == "unresolved" or row["target_endpoint_status"] == "unresolved"
    )
    generic_count = sum(1 for row in edge_rows if row.get("relation_type") == "related_to_generic")
    generic_ratio = round(generic_count / len(edge_rows), 6) if edge_rows else 0.0
    relation_counts = Counter(str(row.get("relation_type") or "") for row in edge_rows)
    unresolved_claim_subject_count = sum(1 for row in claim_table_rows if row.get("subject_endpoint_status") == "unresolved")
    entity_quality_counts = Counter(str(row.get("entity_quality_hint") or "") for row in node_rows)
    generic_review_counts = Counter(str(row.get("generic_relation_review_hint") or "") for row in edge_rows)
    entity_resolution_source_counts = Counter(
        str(row.get("candidate_source") or "") for row in entity_resolution_candidates
    )

    outputs = {
        "graph_entity_normalization_candidates": output_dir / "graph_entity_normalization_candidates.jsonl",
        "graph_entity_resolution_candidates": output_dir / "graph_entity_resolution_candidates.jsonl",
        "graph_relation_type_normalization": output_dir / "graph_relation_type_normalization.jsonl",
        "graph_merge_decisions": output_dir / "graph_merge_decisions.jsonl",
        "graph_nodes_table": output_dir / "graph_nodes_table.jsonl",
        "graph_edges_table": output_dir / "graph_edges_table.jsonl",
        "graph_claims_table": output_dir / "graph_claims_table.jsonl",
        "evidence_links": output_dir / "evidence_links.jsonl",
        "graph_consolidation_manifest": output_dir / "graph_consolidation_manifest.json",
        "graph_consolidation_report": output_dir / "graph_consolidation_report.md",
    }

    write_jsonl(outputs["graph_entity_normalization_candidates"], entity_norm_rows)
    write_jsonl(outputs["graph_entity_resolution_candidates"], entity_resolution_candidates)
    write_jsonl(outputs["graph_relation_type_normalization"], relation_type_rows)
    write_jsonl(outputs["graph_merge_decisions"], merge_decisions)
    write_jsonl(outputs["graph_nodes_table"], node_rows)
    write_jsonl(outputs["graph_edges_table"], edge_rows)
    write_jsonl(outputs["graph_claims_table"], claim_table_rows)
    write_jsonl(outputs["evidence_links"], evidence_links)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "workspace_id": workspace.name,
        "created_at": now_iso(),
        "extraction_dir": str(extraction_dir),
        "output_dir": str(output_dir),
        "relation_policy": str(policy_path),
        "relation_policy_hash": file_hash(policy_path),
        "source_asset_hashes": source_hashes(
            [
                entity_path,
                relation_path,
                claim_path,
                merge_candidate_path,
                extraction_manifest_path,
            ]
        ),
        "upstream_extraction_manifest": extraction_manifest.get("schema_version", ""),
        "counts": {
            "entity_candidate_count": len(entities),
            "relation_candidate_count": len(relations),
            "claim_candidate_count": len(claims),
            "upstream_merge_candidate_count": len(merge_candidates),
            "entity_normalization_count": len(entity_norm_rows),
            "entity_resolution_candidate_count": len(entity_resolution_candidates),
            "entity_resolution_source_counts": dict(sorted(entity_resolution_source_counts.items())),
            "relation_type_normalization_count": len(relation_type_rows),
            "merge_decision_count": len(merge_decisions),
            "node_count": len(node_rows),
            "edge_count": len(edge_rows),
            "claim_table_count": len(claim_table_rows),
            "evidence_link_count": len(evidence_links),
            "unresolved_endpoint_relation_count": unresolved_endpoint_count,
            "unresolved_claim_subject_count": unresolved_claim_subject_count,
            "missing_evidence_row_count": missing_evidence_count,
            "generic_relation_count": generic_count,
            "generic_relation_ratio": generic_ratio,
            "relation_type_counts": dict(sorted(relation_counts.items())),
            "entity_quality_counts": dict(sorted(entity_quality_counts.items())),
            "generic_relation_review_counts": dict(sorted(generic_review_counts.items())),
        },
        "policies": {
            "candidate_first": True,
            "graph_is_not_proof": True,
            "write_permission": False,
            "support_status": "not_checked",
            "no_silent_entity_merge": True,
            "relationship_merge_key_includes_relation_type_perspective_attribution_temporal_scope": True,
            "summaries_are_helper_fields_not_evidence_truth": True,
            "networkx_metrics_computed": False,
            "graph_truth_written": False,
            "durable_memory_written": False,
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["graph_consolidation_manifest"], manifest)
    write_report(outputs["graph_consolidation_report"], manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Consolidate v0.3 graph candidates into audit graph tables.")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--extraction-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--relation-policy", default=DEFAULT_RELATION_POLICY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = consolidate_graph_candidates(
        args.workspace,
        extraction_dir=args.extraction_dir,
        output_dir=args.output_dir,
        relation_policy_path=args.relation_policy,
    )
    print(json.dumps({"manifest": manifest["outputs"]["graph_consolidation_manifest"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
