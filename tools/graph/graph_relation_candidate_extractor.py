"""Extract v0.3 graph relation candidates from construction packets.

This is the first graph-construction extraction slice. It mirrors GraphRAG's
ordering:

    text units -> extraction -> merge candidates -> later graph algorithms

The implementation is candidate-first and evidence-bound. It does not write
graph truth, durable memory, or graph algorithms.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
    first_non_empty,
    read_json,
    read_jsonl,
    sha256_text,
    stable_id,
    unique_strings,
    write_json,
    write_jsonl,
    write_text,
)


SCHEMA_VERSION = "graph_v03.relation_extraction.v0.1"
ENTITY_SCHEMA_VERSION = "graph_v03.entity_candidate.v0.1"
RELATION_SCHEMA_VERSION = "graph_v03.relation_candidate.v0.1"
CLAIM_SCHEMA_VERSION = "graph_v03.claim_candidate.v0.1"
MERGE_SCHEMA_VERSION = "graph_v03.merge_candidate.v0.1"
FAILURE_SCHEMA_VERSION = "graph_v03.extraction_failure.v0.1"
SUPPORTED_PROVIDERS = {"mock", "mock_regex_baseline", "external_jsonl", "openai"}
SUPPORTED_DUPLICATE_POLICIES = {"fail", "overwrite_generated"}
DEFAULT_INPUT_DIR_NAME = "graph_v03_construction"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_construction"
DEFAULT_GRAPH_EXTRACTION_PROFILE = "configs/graph/graph_relation_extraction.v0.3.json"
MODEL_CALL_INPUTS_FILENAME = "graph_model_call_inputs.jsonl"
RELATION_SCHEMA_CANDIDATES_FILENAME = "relation_schema_candidates.jsonl"

GENERIC_ENTITY_TYPES = {"unknown", "concept", "event"}
LOW_VALUE_PATTERNS = [
    r"^\s*(thanks|thank you|ok|okay|yes|no|lol|haha|hmm|uh+)[!.?]*\s*$",
    r"^\s*(\d{4}|\w{3}\.\s+\d{1,2}(?:,\s+\d{4})?)[!.?]*\s*$",
]
ABBREVIATIONS = ["Mr.", "Mrs.", "Ms.", "Dr.", "St.", "Esq.", "Jr.", "Sr.", "Prof."]
INVALID_TARGET_HINTS = {"mr", "mrs", "ms", "dr", "sir", "he", "she", "him", "her", "they", "them", "i", "we", "you"}
QUOTE_TOKEN_RE = re.compile(r"[0-9a-zA-Z\u4e00-\u9fff]+")
QUOTE_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "she",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "was",
    "were",
    "which",
    "who",
    "with",
}

RELATION_PATTERNS: list[dict[str, Any]] = [
    {
        "relation_type": "uses",
        "pattern": r"\b(?:use|uses|used|using)\b\s+(?P<object>[^.;!?]+)",
        "entity_type_hint": "tool",
        "directionality": "forward",
        "confidence_hint": "high",
        "inference_level_hint": "explicit",
    },
    {
        "relation_type": "visited",
        "pattern": r"\b(?:(?:go|goes|went)\s+to|visit|visits|visited)\b\s+(?P<object>[^.;!?]+)",
        "entity_type_hint": "place",
        "directionality": "forward",
        "confidence_hint": "high",
        "inference_level_hint": "explicit",
    },
    {
        "relation_type": "read",
        "pattern": r"\b(?:read|reads|reading)\b\s+(?P<object>[^.;!?]+)",
        "entity_type_hint": "document",
        "directionality": "forward",
        "confidence_hint": "high",
        "inference_level_hint": "explicit",
    },
    {
        "relation_type": "works_on",
        "pattern": r"\b(?:work|works|worked|working)\s+on\s+(?P<object>[^.;!?]+)",
        "entity_type_hint": "project",
        "directionality": "forward",
        "confidence_hint": "high",
        "inference_level_hint": "explicit",
    },
    {
        "relation_type": "learns_from",
        "pattern": r"\b(?:learn|learns|learned|learning)\s+from\s+(?P<object>[^.;!?]+)",
        "entity_type_hint": "person",
        "directionality": "forward",
        "confidence_hint": "high",
        "inference_level_hint": "explicit",
    },
    {
        "relation_type": "depends_on",
        "pattern": r"\b(?:depend|depends|depended|depending)\s+on\s+(?P<object>[^.;!?]+)",
        "entity_type_hint": "project",
        "directionality": "forward",
        "confidence_hint": "medium",
        "inference_level_hint": "implicit",
    },
    {
        "relation_type": "met",
        "pattern": r"\b(?:met|meets|meeting)\b(?:\s+with)?\s+(?P<object>[^.;!?]+)",
        "entity_type_hint": "person",
        "directionality": "bidirectional",
        "confidence_hint": "medium",
        "inference_level_hint": "explicit",
    },
    {
        "relation_type": "encourages",
        "pattern": r"\b(?:encourage|encourages|encouraged)\b\s+(?P<object>[^.;!?]+)",
        "entity_type_hint": "person",
        "directionality": "forward",
        "confidence_hint": "medium",
        "inference_level_hint": "explicit",
    },
    {
        "relation_type": "criticizes",
        "pattern": r"\b(?:criticize|criticizes|criticized|rebuke|rebuked)\b\s+(?P<object>[^.;!?]+)",
        "entity_type_hint": "person",
        "directionality": "forward",
        "confidence_hint": "medium",
        "inference_level_hint": "explicit",
    },
]

CLAIM_PATTERNS: list[dict[str, Any]] = [
    {
        "claim_type_hint": "subject_belief_about_node",
        "pattern": r"\b(?:think|thinks|thought|believe|believes|believed)\b\s+(?P<object>[^.;!?]+)",
        "directionality": "unknown",
        "confidence_hint": "medium",
        "inference_level_hint": "implicit",
    },
    {
        "claim_type_hint": "subject_relation_to_node",
        "pattern": r"\b(?:want|wants|wanted|like|likes|liked|love|loves|loved|prefer|prefers|preferred|enjoy|enjoys|enjoyed)\b\s+(?P<object>[^.;!?]+)",
        "directionality": "unknown",
        "confidence_hint": "medium",
        "inference_level_hint": "implicit",
    },
    {
        "claim_type_hint": "node_state_from_subject_perspective",
        "pattern": r"\b(?:is|are|was|were|seems|seem|looks|look)\b\s+(?P<object>[^.;!?]+)",
        "directionality": "unknown",
        "confidence_hint": "low",
        "inference_level_hint": "implicit",
    },
    {
        "claim_type_hint": "interaction_hypothesis",
        "pattern": r"\b(?:say|says|said|tell|tells|told|ask|asks|asked|write|writes|wrote)\b\s+(?P<object>[^.;!?]+)",
        "directionality": "forward",
        "confidence_hint": "medium",
        "inference_level_hint": "implicit",
    },
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str | None:
    return sha256_text(path.read_text(encoding="utf-8", errors="ignore")) if path.exists() else None


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", normalize_whitespace(str(text or "")).lower()).strip("-")


def normalize_relation_type_hint(text: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", normalize_whitespace(str(text or "").lower()).replace("-", "_")).strip("_")


def clamp_excerpt(text: str, max_chars: int = 240) -> str:
    return normalize_whitespace(text)[:max_chars]


def load_graph_extraction_profile(project_root: Path, profile_path: Path | str) -> dict[str, Any]:
    resolved = resolve_project_path(project_root, profile_path)
    data = read_json(resolved)
    required = {
        "schema_version",
        "profile_id",
        "target_task",
        "prompt_policies",
        "output_kinds",
        "entity_types",
        "directionality_statuses",
        "attribution_statuses",
        "confidence_hints",
    }
    missing = sorted(key for key in required if key not in data)
    if missing:
        raise ValueError("Graph extraction profile missing required keys: " + ", ".join(missing))
    if data["target_task"] != "graph_relation_candidate":
        raise ValueError("Graph extraction profile target_task must be graph_relation_candidate")
    for prompt_key in ("weak", "strong"):
        if prompt_key not in data["prompt_policies"]:
            raise ValueError(f"Graph extraction profile prompt_policies missing {prompt_key}")
        if "policy_id" not in data["prompt_policies"][prompt_key] or "path" not in data["prompt_policies"][prompt_key]:
            raise ValueError(f"Graph extraction profile prompt_policies.{prompt_key} must include policy_id and path")
    data["_profile_path"] = str(resolved)
    data["_profile_hash"] = file_hash(resolved)
    return data


def load_graph_prompt(project_root: Path, profile: dict[str, Any], prompt_key: str) -> PromptPolicy:
    policy = profile["prompt_policies"][prompt_key]
    path = resolve_project_path(project_root, policy["path"])
    text = path.read_text(encoding="utf-8")
    return PromptPolicy(
        policy_id=str(policy["policy_id"]),
        path=path,
        text=text,
        prompt_hash=proposal_file_hash(path),
    )


def strip_json_code_fence(text: str) -> str:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def split_clauses(text: str) -> list[str]:
    normalized = normalize_whitespace(text)
    for index, abbreviation in enumerate(ABBREVIATIONS):
        normalized = normalized.replace(abbreviation, f"{abbreviation[:-1]}<DOT{index}>")
    parts = re.split(r"(?<=[.!?;:])\s+|\s+(?:and|but|then|so)\s+", normalized)
    restored: list[str] = []
    for part in parts:
        restored_part = part
        for index, abbreviation in enumerate(ABBREVIATIONS):
            restored_part = restored_part.replace(f"{abbreviation[:-1]}<DOT{index}>", abbreviation)
        restored.append(restored_part.strip())
    return [part for part in restored if part]


def looks_low_value(text: str) -> bool:
    stripped = normalize_whitespace(text)
    if not stripped:
        return True
    if len(stripped) < 10 and not re.search(r"[A-Za-z\u4e00-\u9fff]", stripped):
        return True
    return any(re.fullmatch(pattern, stripped, flags=re.IGNORECASE) for pattern in LOW_VALUE_PATTERNS)


def maybe_route_lane(packet: dict[str, Any], text: str) -> tuple[str, list[str]]:
    warnings: list[str] = []
    lowered = text.lower()
    if looks_low_value(text):
        return "no_useful_modeling_value", ["low_value_text"]
    if "?" in text:
        warnings.append("question_or_interaction_marker")
    if any(word in lowered for word in ["think", "thinks", "thought", "believe", "believes", "said", "told", "asked", "wrote"]):
        return "strong_llm_extraction", unique_strings(warnings, ["attribution_heavy"])
    if len(split_clauses(text)) > 1:
        return "strong_llm_extraction", unique_strings(warnings, ["multi_clause"])
    if any(token in lowered for token in ["use", "visit", "went", "read", "work", "learn"]):
        return "weak_llm_extraction", warnings
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text):
        return "heuristic_nlp_baseline", warnings
    return "weak_llm_extraction", warnings


def clean_phrase(value: str) -> str:
    text = normalize_whitespace(value)
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    text = re.sub(r"^(?:a|an|the|this|that|these|those|my|your|his|her|our|their)\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_target_phrase(text: str) -> str:
    phrase = clean_phrase(text)
    phrase = re.split(r"\b(?:after|because|when|while|if|though|although|for|with|at|in|on|from|to|of)\b", phrase, maxsplit=1)[0]
    return clean_phrase(phrase)


def valid_target_hint(text: str) -> bool:
    normalized = normalize_key(text)
    if not normalized:
        return False
    if normalized in INVALID_TARGET_HINTS:
        return False
    if len(normalized) < 3:
        return False
    return True


def infer_entity_type(text: str, fallback: str = "unknown") -> str:
    lowered = text.lower()
    if any(token in lowered for token in ["notebook", "tool", "runner", "api", "library", "model", "workflow", "algorithm"]):
        return "tool"
    if any(token in lowered for token in ["project", "report", "build", "plan", "experiment", "task"]):
        return "project"
    if any(token in lowered for token in ["chapel", "house", "street", "room", "city", "town", "place"]):
        return "place"
    if any(token in lowered for token in ["book", "liturgy", "note", "document", "text", "article", "paper"]):
        return "document"
    if any(token in lowered for token in ["meeting", "dinner", "walk", "visit", "journey"]):
        return "event"
    if re.search(r"\b(?:Mr|Mrs|Ms|Dr|Sir)\.?\s+[A-Z][a-z]+", text):
        return "person"
    if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text):
        return "person"
    return fallback


def extract_subject_hint(packet: dict[str, Any]) -> str:
    return first_non_empty(packet.get("source_perspective"), packet.get("participant"), packet.get("target_participant"), default="unknown")


def base_candidate_row(
    *,
    schema_version: str,
    candidate_kind: str,
    packet: dict[str, Any],
    source_text_excerpt: str,
    extracted_span: str,
    source_node_hint: str,
    target_node_hint: str,
    entity_type_hint: str,
    relation_type_hint: str,
    directionality: str,
    source_perspective: str,
    attribution_status: str,
    temporal_scope: dict[str, Any],
    evidence_refs: list[str],
    source_refs: list[str],
    raw_backpointer_refs: list[str],
    route_refs: list[str],
    proposal_refs: list[str],
    review_refs: list[str],
    confidence_hint: str,
    inference_level_hint: str,
    warnings: list[str],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": schema_version,
        "candidate_id": stable_id(
            "graph_candidate",
            "|".join(
                [
                    str(packet.get("packet_id") or ""),
                    candidate_kind,
                    source_node_hint,
                    relation_type_hint,
                    target_node_hint,
                    extracted_span,
                ]
            ),
        ),
        "candidate_kind": candidate_kind,
        "source_packet_id": str(packet.get("packet_id") or ""),
        "source_text_excerpt": source_text_excerpt,
        "extracted_span": extracted_span,
        "source_node_hint": source_node_hint,
        "target_node_hint": target_node_hint,
        "entity_type_hint": entity_type_hint,
        "relation_type_hint": relation_type_hint,
        "directionality": directionality,
        "source_perspective": source_perspective,
        "attribution_status": attribution_status,
        "temporal_scope": temporal_scope,
        "evidence_refs": evidence_refs,
        "source_refs": source_refs,
        "raw_backpointer_refs": raw_backpointer_refs,
        "route_refs": route_refs,
        "proposal_refs": proposal_refs,
        "review_refs": review_refs,
        "confidence_hint": confidence_hint,
        "inference_level_hint": inference_level_hint,
        "warnings": unique_strings(warnings),
        "graph_is_not_proof": True,
    }
    if extra:
        payload.update(extra)
    return payload


def packet_text(packet: dict[str, Any]) -> str:
    return first_non_empty(packet.get("original_text"), packet.get("processed_text"), default="")


def packet_warnings(packet: dict[str, Any]) -> list[str]:
    return [warning for warning in unique_strings(packet.get("warnings")) if warning.lower() not in {"none", "null"}]


def packet_primary_text(packet: dict[str, Any]) -> str:
    return first_non_empty(packet.get("original_text"), packet.get("primary_text"), packet.get("processed_text"), default="")


def packet_extraction_text(packet: dict[str, Any]) -> str:
    return first_non_empty(packet.get("graph_extraction_text"), packet.get("graph_route_text"), packet_text(packet), default="")


def compact_relation_schema_candidates(candidates: list[dict[str, Any]], *, limit: int = 50) -> list[dict[str, Any]]:
    compacted: list[dict[str, Any]] = []
    for candidate in candidates[: max(0, limit)]:
        compacted.append(
            {
                "rank": candidate.get("rank"),
                "relation_type": candidate.get("relation_type"),
                "score": candidate.get("score"),
                "category": candidate.get("category"),
                "aliases": unique_strings(candidate.get("aliases"))[:8],
                "external_sources": unique_strings(candidate.get("external_sources"))[:5],
            }
        )
    return compacted


def load_relation_schema_candidate_index(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Relation schema candidates file not found: {path}")
    index: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        packet_id = str(row.get("packet_id") or "")
        if not packet_id:
            continue
        index[packet_id] = compact_relation_schema_candidates(row.get("relation_schema_candidates") or [])
    return index


def build_graph_model_input(
    packet: dict[str, Any],
    lane: str,
    route_decision: dict[str, Any] | None,
    profile: dict[str, Any],
    relation_schema_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    allowed_relation_types = compact_relation_schema_candidates(relation_schema_candidates or [])
    return {
        "schema_version": "graph_v03.extraction_model_input.v0.1",
        "target_task": "graph_relation_candidate",
        "profile_id": profile["profile_id"],
        "source_packet_id": str(packet.get("packet_id") or ""),
        "recommended_route": lane,
        "route_decision_id": route_decision.get("route_decision_id") if route_decision else None,
        "route_reasons": unique_strings((route_decision or {}).get("routing_reasons")),
        "modeled_user_id": packet.get("modeled_user_id"),
        "source_perspective": extract_subject_hint(packet),
        "subject_role": first_non_empty(packet.get("subject_role"), default="unknown"),
        "attribution_status": first_non_empty(packet.get("attribution_status"), default="unknown"),
        "context_requirement_hint": first_non_empty(packet.get("context_requirement_hint"), default="unknown"),
        "context_usage": first_non_empty(packet.get("context_usage"), default="unknown"),
        "primary_text": packet_primary_text(packet),
        "processed_text": str(packet.get("processed_text") or ""),
        "extraction_text": packet_extraction_text(packet),
        "primary_evidence_refs": unique_strings(packet.get("primary_evidence_refs"), packet.get("evidence_refs")),
        "context_evidence_refs": unique_strings(packet.get("context_evidence_refs")),
        "evidence_refs": unique_strings(packet.get("evidence_refs")),
        "source_refs": unique_strings(packet.get("source_refs")),
        "raw_backpointer_refs": unique_strings(packet.get("raw_backpointer_refs")),
        "route_refs": unique_strings(packet.get("route_refs")),
        "proposal_refs": unique_strings(packet.get("proposal_refs")),
        "review_refs": unique_strings(packet.get("review_refs")),
        "temporal_scope": packet.get("temporal_scope") or {},
        "warnings": packet_warnings(packet),
        "allowed_output_kinds": profile["output_kinds"],
        "allowed_entity_types": profile["entity_types"],
        "allowed_directionality_statuses": profile["directionality_statuses"],
        "allowed_attribution_statuses": profile["attribution_statuses"],
        "allowed_confidence_hints": profile["confidence_hints"],
        "allowed_relation_types": allowed_relation_types,
        "relation_schema_candidate_count": len(allowed_relation_types),
        "relation_schema_policy": {
            "source": "external_relation_schema_retrieval",
            "instruction": "Prefer relation_type_hint values from allowed_relation_types. If none fits, use out_of_schema_relation and explain the natural-language relation in relation_description.",
            "allowed_relation_types_are_candidates_not_truth": True,
        },
        "graph_is_not_proof": True,
    }


def parse_graph_model_output(output_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = json.loads(strip_json_code_fence(output_text))
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    if not isinstance(payload, dict):
        return None, ["schema_validation_failed", "model_output_not_object"]
    return payload, []


def text_contains_quote(packet: dict[str, Any], quote: str) -> tuple[bool, str]:
    normalized_quote = normalize_whitespace(quote)
    if not normalized_quote:
        return False, "missing_quote"
    primary = normalize_whitespace(packet_primary_text(packet))
    extraction = normalize_whitespace(packet_extraction_text(packet))
    if normalized_quote in primary:
        return True, "primary"
    if normalized_quote in extraction:
        return True, "context"
    if quote_matches_text_approximately(normalized_quote, primary):
        return True, "primary_approximate"
    if quote_matches_text_approximately(normalized_quote, extraction):
        return True, "context_approximate"
    return False, "missing"


def quote_content_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for token in QUOTE_TOKEN_RE.findall(str(text or "").lower()):
        if not token or token.isdigit():
            continue
        if len(token) < 3 and not any("\u4e00" <= char <= "\u9fff" for char in token):
            continue
        if token in QUOTE_STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def quote_matches_text_approximately(quote: str, text: str) -> bool:
    quote_tokens = quote_content_tokens(quote)
    if len(quote_tokens) < 4:
        return False
    text_tokens = quote_content_tokens(text)
    if not text_tokens:
        return False
    text_token_set = set(text_tokens)
    coverage = sum(1 for token in quote_tokens if token in text_token_set) / max(1, len(quote_tokens))
    if coverage < 0.82:
        return False
    if len(quote_tokens) <= 8:
        return coverage >= 0.9
    return True


def evidence_refs_for_role(packet: dict[str, Any], evidence_role: str) -> list[str]:
    primary_refs = unique_strings(packet.get("primary_evidence_refs"), packet.get("evidence_refs"))
    context_refs = unique_strings(packet.get("context_evidence_refs"))
    if evidence_role.startswith("context"):
        return context_refs
    return primary_refs


def local_entity_index(entity_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in entity_rows:
        local_id = str(row.get("local_entity_id") or "")
        if local_id:
            index[local_id] = row
    return index


def relation_candidates_for_clause(
    packet: dict[str, Any],
    clause: str,
    lane: str,
    *,
    max_gleanings: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    source_text = packet_text(packet)
    source_excerpt = clamp_excerpt(source_text)
    source_perspective = extract_subject_hint(packet)
    primary_evidence_refs = unique_strings(packet.get("primary_evidence_refs"), packet.get("evidence_refs"))
    context_evidence_refs = unique_strings(packet.get("context_evidence_refs"))
    evidence_refs = primary_evidence_refs
    source_refs = unique_strings(packet.get("source_refs"))
    raw_backpointer_refs = unique_strings(packet.get("raw_backpointer_refs"))
    route_refs = unique_strings(packet.get("route_refs"))
    proposal_refs = unique_strings(packet.get("proposal_refs"))
    review_refs = unique_strings(packet.get("review_refs"))
    temporal_scope = packet.get("temporal_scope") or {}
    base_warnings = packet_warnings(packet)
    relation_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    glean_count = 0

    for rule in CLAIM_PATTERNS:
        if glean_count >= max_gleanings:
            warnings.append("gleaning_limit_reached")
            break
        match = re.search(rule["pattern"], clause, flags=re.IGNORECASE)
        if not match:
            continue
        target = extract_target_phrase(match.group("object"))
        if not valid_target_hint(target):
            continue
        glean_count += 1
        claim_rows.append(
            base_candidate_row(
                schema_version=CLAIM_SCHEMA_VERSION,
                candidate_kind="graph_claim_candidate",
                packet=packet,
                source_text_excerpt=source_excerpt,
                extracted_span=clause,
                source_node_hint=source_perspective,
                target_node_hint=target,
                entity_type_hint=infer_entity_type(target, fallback="concept"),
                relation_type_hint="",
                directionality=rule["directionality"],
                source_perspective=source_perspective,
                attribution_status=first_non_empty(packet.get("attribution_status"), default="unknown"),
                temporal_scope=temporal_scope,
                evidence_refs=evidence_refs,
                source_refs=source_refs,
                raw_backpointer_refs=raw_backpointer_refs,
                route_refs=route_refs,
                proposal_refs=proposal_refs,
                review_refs=review_refs,
                confidence_hint=first_non_empty(rule.get("confidence_hint"), packet.get("confidence"), default="unknown"),
                inference_level_hint=first_non_empty(rule.get("inference_level_hint"), packet.get("inference_level"), default="unknown"),
                warnings=unique_strings(base_warnings, warnings, [f"claim_type_hint:{rule['claim_type_hint']}"]),
                extra={
                    "candidate_text": clause,
                    "claim_type_hint": rule["claim_type_hint"],
                    "lane": lane,
                },
            )
        )
        warnings = []

    for rule in RELATION_PATTERNS:
        if glean_count >= max_gleanings:
            warnings.append("gleaning_limit_reached")
            break
        match = re.search(rule["pattern"], clause, flags=re.IGNORECASE)
        if not match:
            continue
        target = extract_target_phrase(match.group("object"))
        if not valid_target_hint(target):
            continue
        glean_count += 1
        relation_rows.append(
            base_candidate_row(
                schema_version=RELATION_SCHEMA_VERSION,
                candidate_kind="graph_relation_candidate",
                packet=packet,
                source_text_excerpt=source_excerpt,
                extracted_span=clause,
                source_node_hint=source_perspective,
                target_node_hint=target,
                entity_type_hint=rule["entity_type_hint"],
                relation_type_hint=rule["relation_type"],
                directionality=rule["directionality"],
                source_perspective=source_perspective,
                attribution_status=first_non_empty(packet.get("attribution_status"), default="unknown"),
                temporal_scope=temporal_scope,
                evidence_refs=evidence_refs,
                source_refs=source_refs,
                raw_backpointer_refs=raw_backpointer_refs,
                route_refs=route_refs,
                proposal_refs=proposal_refs,
                review_refs=review_refs,
                confidence_hint=first_non_empty(rule.get("confidence_hint"), packet.get("confidence"), default="unknown"),
                inference_level_hint=first_non_empty(rule.get("inference_level_hint"), packet.get("inference_level"), default="unknown"),
                warnings=unique_strings(base_warnings, warnings, [f"extraction_lane:{lane}"]),
                extra={
                    "candidate_text": clause,
                    "lane": lane,
                },
            )
        )
        warnings = []

    return relation_rows, claim_rows, warnings


def entity_candidates_for_clause(
    packet: dict[str, Any],
    clause: str,
    lane: str,
) -> list[dict[str, Any]]:
    source_text = packet_text(packet)
    source_excerpt = clamp_excerpt(source_text)
    source_perspective = extract_subject_hint(packet)
    primary_evidence_refs = unique_strings(packet.get("primary_evidence_refs"), packet.get("evidence_refs"))
    context_evidence_refs = unique_strings(packet.get("context_evidence_refs"))
    evidence_refs = primary_evidence_refs
    source_refs = unique_strings(packet.get("source_refs"))
    raw_backpointer_refs = unique_strings(packet.get("raw_backpointer_refs"))
    route_refs = unique_strings(packet.get("route_refs"))
    proposal_refs = unique_strings(packet.get("proposal_refs"))
    review_refs = unique_strings(packet.get("review_refs"))
    temporal_scope = packet.get("temporal_scope") or {}
    candidates: list[dict[str, Any]] = []

    named_entities = []
    if source_perspective and source_perspective != "unknown":
        named_entities.append(source_perspective)
    for token in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", clause):
        named_entities.append(token)

    for entity in unique_strings(named_entities):
        candidates.append(
            base_candidate_row(
                schema_version=ENTITY_SCHEMA_VERSION,
                candidate_kind="graph_entity_candidate",
                packet=packet,
                source_text_excerpt=source_excerpt,
                extracted_span=clause,
                source_node_hint=entity,
                target_node_hint="",
                entity_type_hint=infer_entity_type(entity, fallback="unknown"),
                relation_type_hint="",
                directionality="n/a",
                source_perspective=source_perspective,
                attribution_status=first_non_empty(packet.get("attribution_status"), default="unknown"),
                temporal_scope=temporal_scope,
                evidence_refs=evidence_refs,
                source_refs=source_refs,
                raw_backpointer_refs=raw_backpointer_refs,
                route_refs=route_refs,
                proposal_refs=proposal_refs,
                review_refs=review_refs,
                confidence_hint=first_non_empty(packet.get("confidence"), default="unknown"),
                inference_level_hint=first_non_empty(packet.get("inference_level"), default="unknown"),
                warnings=unique_strings(packet_warnings(packet), [f"extraction_lane:{lane}"]),
                extra={
                    "candidate_text": entity,
                    "lane": lane,
                },
            )
        )
    return candidates


def extract_packet_with_lane(
    packet: dict[str, Any],
    *,
    lane: str,
    lane_warnings: list[str] | None = None,
    max_gleanings: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    text = packet_text(packet)
    packet_warning_list = unique_strings(packet_warnings(packet), lane_warnings or [])
    if lane in {"skip_or_background_only", "no_useful_modeling_value"}:
        failure = {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "failure_id": stable_id("graph_extraction_failure", f"{packet.get('packet_id')}|no_useful_modeling_value"),
            "source_packet_id": str(packet.get("packet_id") or ""),
            "failure_kind": "no_useful_modeling_value",
            "reason": "graph package route selected skip/background for this packet",
            "extraction_lane": lane,
            "warnings": packet_warning_list,
            "graph_is_not_proof": True,
        }
        return [], [], [], [failure], packet_warning_list
    if lane in {"repair_or_review", "human_review"}:
        failure_kind = "repair_or_review_required" if lane == "repair_or_review" else "human_review_required"
        failure = {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "failure_id": stable_id("graph_extraction_failure", f"{packet.get('packet_id')}|{failure_kind}"),
            "source_packet_id": str(packet.get("packet_id") or ""),
            "failure_kind": failure_kind,
            "reason": "graph package route requires repair or human review before extraction",
            "extraction_lane": lane,
            "warnings": packet_warning_list,
            "graph_is_not_proof": True,
        }
        return [], [], [], [failure], packet_warning_list
    if lane in {"weak_llm_graph_extraction", "strong_llm_graph_extraction"}:
        failure = {
            "schema_version": FAILURE_SCHEMA_VERSION,
            "failure_id": stable_id("graph_extraction_failure", f"{packet.get('packet_id')}|llm_required"),
            "source_packet_id": str(packet.get("packet_id") or ""),
            "failure_kind": "llm_schema_extraction_required",
            "reason": "graph package route selected LLM schema-guided extraction; mock_regex_baseline will not substitute regex output",
            "extraction_lane": lane,
            "warnings": unique_strings(packet_warning_list, ["llm_route_not_executed_by_mock_regex_baseline"]),
            "graph_is_not_proof": True,
        }
        return [], [], [], [failure], unique_strings(packet_warning_list, ["llm_route_not_executed_by_mock_regex_baseline"])

    entity_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []

    clauses = split_clauses(text)
    if not clauses:
        failure_rows.append(
            {
                "schema_version": FAILURE_SCHEMA_VERSION,
                "failure_id": stable_id("graph_extraction_failure", f"{packet.get('packet_id')}|missing_text"),
                "source_packet_id": str(packet.get("packet_id") or ""),
                "failure_kind": "missing_text",
                "reason": "packet has no usable original or processed text",
                "extraction_lane": lane,
                "warnings": packet_warning_list,
                "graph_is_not_proof": True,
            }
        )
        return [], [], [], failure_rows, packet_warning_list

    glean_limit = max(1, max_gleanings)
    glean_used = 0
    for clause in clauses:
        if glean_used >= glean_limit:
            packet_warning_list = unique_strings(packet_warning_list, ["gleaning_limit_reached"])
            break
        relation_clause_rows: list[dict[str, Any]] = []
        claim_clause_rows: list[dict[str, Any]] = []
        clause_warnings: list[str] = []
        if lane != "entity_candidate_only":
            relation_clause_rows, claim_clause_rows, clause_warnings = relation_candidates_for_clause(
                packet,
                clause,
                lane,
                max_gleanings=glean_limit - glean_used,
            )
        entity_clause_rows = entity_candidates_for_clause(packet, clause, lane)
        relation_rows.extend(relation_clause_rows)
        claim_rows.extend(claim_clause_rows)
        entity_rows.extend(entity_clause_rows)
        glean_used += 1
        if clause_warnings:
            packet_warning_list = unique_strings(packet_warning_list, clause_warnings)

    if not entity_rows and not relation_rows and not claim_rows:
        failure_rows.append(
            {
                "schema_version": FAILURE_SCHEMA_VERSION,
                "failure_id": stable_id("graph_extraction_failure", f"{packet.get('packet_id')}|no_candidates"),
                "source_packet_id": str(packet.get("packet_id") or ""),
                "failure_kind": "no_candidates",
                "reason": "no entity, relation, or claim candidate extracted",
                "extraction_lane": lane,
                "warnings": packet_warning_list,
                "graph_is_not_proof": True,
            }
        )

    return entity_rows, relation_rows, claim_rows, failure_rows, packet_warning_list


def extract_packet(
    packet: dict[str, Any],
    *,
    max_gleanings: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    text = packet_text(packet)
    lane, lane_warnings = maybe_route_lane(packet, text)
    return extract_packet_with_lane(packet, lane=lane, lane_warnings=lane_warnings, max_gleanings=max_gleanings)


def load_external_payload_index(path: Path) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        raise FileNotFoundError(f"external_model_outputs not found: {path}")
    for row in read_jsonl(path):
        packet_id = first_non_empty(row.get("source_packet_id"), row.get("packet_id"), row.get("input_ref"))
        if packet_id:
            index[str(packet_id)].append(row)
    return dict(index)


def load_graph_route_decisions(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    decisions: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        packet_id = first_non_empty(row.get("source_packet_id"), row.get("packet_id"), row.get("input_ref"))
        if packet_id:
            decisions[str(packet_id)] = row
    return decisions


def lane_from_route_decision(packet: dict[str, Any], route_decisions: dict[str, dict[str, Any]]) -> tuple[str, list[str], dict[str, Any] | None]:
    packet_id = str(packet.get("packet_id") or "")
    decision = route_decisions.get(packet_id)
    if decision is None:
        lane, warnings = maybe_route_lane(packet, packet_text(packet))
        return lane, unique_strings(warnings, ["graph_route_decision_missing_mock_fallback"]), None
    lane = str(decision.get("recommended_route") or "skip_or_background_only")
    warnings = unique_strings(decision.get("warnings"), decision.get("routing_reasons"))
    return lane, warnings, decision


def normalize_external_rows(
    packet: dict[str, Any],
    payload_rows: list[dict[str, Any]],
    lane: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    entity_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for payload in payload_rows:
        payload_warnings = unique_strings(payload.get("warnings"))
        warnings = unique_strings(warnings, payload_warnings)
        if payload.get("failure"):
            failure_rows.append(
                {
                    "schema_version": FAILURE_SCHEMA_VERSION,
                    "failure_id": stable_id("graph_extraction_failure", f"{packet.get('packet_id')}|external"),
                    "source_packet_id": str(packet.get("packet_id") or ""),
                    "failure_kind": str(payload.get("failure_kind") or "external_failure"),
                    "reason": str(payload.get("failure") or payload.get("reason") or "external model output signaled failure"),
                    "extraction_lane": lane,
                    "warnings": warnings,
                    "graph_is_not_proof": True,
                }
            )
            continue
        for kind, rows, schema_version in [
            ("graph_entity_candidate", entity_rows, ENTITY_SCHEMA_VERSION),
            ("graph_relation_candidate", relation_rows, RELATION_SCHEMA_VERSION),
            ("graph_claim_candidate", claim_rows, CLAIM_SCHEMA_VERSION),
        ]:
            payload_rows_for_kind = payload.get(f"{kind.split('_')[1]}_candidates")
            if payload_rows_for_kind is None and payload.get("candidate_kind") == kind:
                payload_rows_for_kind = [payload]
            if payload_rows_for_kind is None:
                continue
            for row in payload_rows_for_kind:
                candidate = dict(row)
                candidate.setdefault("schema_version", schema_version)
                candidate.setdefault("candidate_kind", kind)
                candidate.setdefault("source_packet_id", str(packet.get("packet_id") or ""))
                candidate.setdefault("source_text_excerpt", clamp_excerpt(packet_text(packet)))
                candidate.setdefault("evidence_refs", unique_strings(packet.get("evidence_refs")))
                candidate.setdefault("source_refs", unique_strings(packet.get("source_refs")))
                candidate.setdefault("raw_backpointer_refs", unique_strings(packet.get("raw_backpointer_refs")))
                candidate.setdefault("route_refs", unique_strings(packet.get("route_refs")))
                candidate.setdefault("proposal_refs", unique_strings(packet.get("proposal_refs")))
                candidate.setdefault("review_refs", unique_strings(packet.get("review_refs")))
                candidate.setdefault("source_perspective", extract_subject_hint(packet))
                candidate.setdefault("attribution_status", first_non_empty(packet.get("attribution_status"), default="unknown"))
                candidate.setdefault("temporal_scope", packet.get("temporal_scope") or {})
                candidate.setdefault("graph_is_not_proof", True)
                candidate.setdefault("warnings", warnings)
                rows.append(candidate)
    return entity_rows, relation_rows, claim_rows, failure_rows, warnings


def graph_failure_row(
    packet: dict[str, Any],
    *,
    failure_kind: str,
    reason: str,
    lane: str,
    warnings: list[str] | None = None,
    suffix: str | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "failure_id": stable_id("graph_extraction_failure", f"{packet.get('packet_id')}|{failure_kind}|{suffix or reason}"),
        "source_packet_id": str(packet.get("packet_id") or ""),
        "failure_kind": failure_kind,
        "reason": reason,
        "extraction_lane": lane,
        "warnings": unique_strings(packet_warnings(packet), warnings or []),
        "graph_is_not_proof": True,
    }


def candidate_quote_warnings(packet: dict[str, Any], quote: str, candidate_label: str) -> tuple[bool, list[str], str]:
    found, evidence_role = text_contains_quote(packet, quote)
    if not found:
        return False, [f"{candidate_label}_quote_missing_or_not_in_source_text"], evidence_role
    warnings: list[str] = []
    if evidence_role.startswith("context"):
        warnings.extend([f"{candidate_label}_quote_from_context", "context_dependency_warning"])
        if not unique_strings(packet.get("context_evidence_refs")):
            warnings.append("context_evidence_refs_missing")
    if evidence_role.endswith("_approximate"):
        warnings.append(f"{candidate_label}_quote_approximate_match")
    return True, warnings, evidence_role


def normalize_model_payload(
    packet: dict[str, Any],
    payload: dict[str, Any],
    lane: str,
    model_id: str,
    provider_name: str,
    prompt: PromptPolicy,
    relation_schema_candidates: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    warnings = unique_strings(packet_warnings(packet), payload.get("warnings"), [f"extraction_lane:{lane}"])
    allowed_relation_types = {
        normalize_relation_type_hint(str(candidate.get("relation_type") or ""))
        for candidate in (relation_schema_candidates or [])
        if str(candidate.get("relation_type") or "").strip()
    }
    output_kind = str(payload.get("output_kind") or "").strip()
    if output_kind in {"reject", "model_uncertain", "needs_human_review"}:
        failure_kind = {
            "reject": "model_rejected_graph_extraction",
            "model_uncertain": "model_uncertain",
            "needs_human_review": "human_review_required",
        }[output_kind]
        reason = str(payload.get("reason") or payload.get("uncertainty_note") or output_kind)
        return [], [], [], [graph_failure_row(packet, failure_kind=failure_kind, reason=reason, lane=lane, warnings=warnings)], warnings
    if output_kind != "graph_bundle_candidate":
        return [], [], [], [
            graph_failure_row(
                packet,
                failure_kind="schema_validation_failed",
                reason=f"unsupported output_kind: {output_kind or '<missing>'}",
                lane=lane,
                warnings=unique_strings(warnings, ["unsupported_output_kind"]),
            )
        ], unique_strings(warnings, ["unsupported_output_kind"])

    entity_payloads = payload.get("entity_candidates")
    relation_payloads = payload.get("relation_candidates")
    claim_payloads = payload.get("claim_candidates", [])
    if not isinstance(entity_payloads, list) or not isinstance(relation_payloads, list) or not isinstance(claim_payloads, list):
        return [], [], [], [
            graph_failure_row(
                packet,
                failure_kind="schema_validation_failed",
                reason="entity_candidates, relation_candidates, and claim_candidates must be arrays",
                lane=lane,
                warnings=unique_strings(warnings, ["candidate_arrays_required"]),
            )
        ], unique_strings(warnings, ["candidate_arrays_required"])

    source_text = packet_text(packet)
    source_excerpt = clamp_excerpt(source_text)
    source_perspective = extract_subject_hint(packet)
    primary_evidence_refs = unique_strings(packet.get("primary_evidence_refs"), packet.get("evidence_refs"))
    context_evidence_refs = unique_strings(packet.get("context_evidence_refs"))
    evidence_refs = primary_evidence_refs
    source_refs = unique_strings(packet.get("source_refs"))
    raw_backpointer_refs = unique_strings(packet.get("raw_backpointer_refs"))
    route_refs = unique_strings(packet.get("route_refs"))
    proposal_refs = unique_strings(packet.get("proposal_refs"))
    review_refs = unique_strings(packet.get("review_refs"))
    temporal_scope = packet.get("temporal_scope") or {}
    attribution_status = first_non_empty(packet.get("attribution_status"), default="unknown")

    entity_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    seen_local_ids: set[str] = set()

    for index, entity in enumerate(entity_payloads):
        if not isinstance(entity, dict):
            failure_rows.append(graph_failure_row(packet, failure_kind="schema_validation_failed", reason="entity candidate must be an object", lane=lane, warnings=warnings, suffix=f"entity:{index}"))
            continue
        local_id = str(entity.get("local_entity_id") or "").strip()
        name = normalize_whitespace(str(entity.get("name") or entity.get("entity_name") or ""))
        quote = str(entity.get("source_text_quote") or "")
        if not local_id or not name:
            failure_rows.append(graph_failure_row(packet, failure_kind="schema_validation_failed", reason="entity candidate missing local_entity_id or name", lane=lane, warnings=warnings, suffix=f"entity:{index}"))
            continue
        if local_id in seen_local_ids:
            failure_rows.append(graph_failure_row(packet, failure_kind="schema_validation_failed", reason=f"duplicate local_entity_id: {local_id}", lane=lane, warnings=warnings, suffix=f"entity:{index}"))
            continue
        quote_ok, quote_warnings, evidence_role = candidate_quote_warnings(packet, quote, "entity")
        if not quote_ok:
            failure_rows.append(graph_failure_row(packet, failure_kind="quote_validation_failed", reason=f"entity quote missing or not found for {local_id}", lane=lane, warnings=unique_strings(warnings, quote_warnings), suffix=f"entity:{index}"))
            continue
        candidate_evidence_refs = evidence_refs_for_role(packet, evidence_role)
        seen_local_ids.add(local_id)
        entity_rows.append(
            base_candidate_row(
                schema_version=ENTITY_SCHEMA_VERSION,
                candidate_kind="graph_entity_candidate",
                packet=packet,
                source_text_excerpt=source_excerpt,
                extracted_span=quote,
                source_node_hint=name,
                target_node_hint="",
                entity_type_hint=str(entity.get("entity_type_hint") or "unknown"),
                relation_type_hint="",
                directionality="n/a",
                source_perspective=source_perspective,
                attribution_status=str(entity.get("attribution_status") or attribution_status),
                temporal_scope=temporal_scope,
                evidence_refs=candidate_evidence_refs,
                source_refs=source_refs,
                raw_backpointer_refs=raw_backpointer_refs,
                route_refs=route_refs,
                proposal_refs=proposal_refs,
                review_refs=review_refs,
                confidence_hint=str(entity.get("confidence_hint") or "unknown"),
                inference_level_hint=str(entity.get("inference_level_hint") or "explicit"),
                warnings=unique_strings(warnings, quote_warnings, entity.get("warnings")),
                extra={
                    "candidate_text": name,
                    "description": str(entity.get("description") or ""),
                    "local_entity_id": local_id,
                    "source_text_quote": quote,
                    "evidence_role": evidence_role,
                    "primary_evidence_refs": primary_evidence_refs,
                    "context_evidence_refs": context_evidence_refs,
                    "lane": lane,
                    "model_id": model_id,
                    "provider": provider_name,
                    "prompt_policy_id": prompt.policy_id,
                    "prompt_hash": prompt.prompt_hash,
                    "write_permission": False,
                },
            )
        )

    entity_by_local_id = local_entity_index(entity_rows)
    for index, relation in enumerate(relation_payloads):
        if not isinstance(relation, dict):
            failure_rows.append(graph_failure_row(packet, failure_kind="schema_validation_failed", reason="relation candidate must be an object", lane=lane, warnings=warnings, suffix=f"relation:{index}"))
            continue
        source_local_id = str(relation.get("source_local_entity_id") or "").strip()
        target_local_id = str(relation.get("target_local_entity_id") or "").strip()
        if source_local_id not in entity_by_local_id or target_local_id not in entity_by_local_id:
            failure_rows.append(
                graph_failure_row(
                    packet,
                    failure_kind="endpoint_validation_failed",
                    reason=f"relation endpoint not found in Step 1 entities: {source_local_id}->{target_local_id}",
                    lane=lane,
                    warnings=unique_strings(warnings, ["relation_endpoint_not_in_entity_candidates"]),
                    suffix=f"relation:{index}",
                )
            )
            continue
        quote = str(relation.get("source_text_quote") or "")
        quote_ok, quote_warnings, evidence_role = candidate_quote_warnings(packet, quote, "relation")
        if not quote_ok:
            failure_rows.append(graph_failure_row(packet, failure_kind="quote_validation_failed", reason="relation quote missing or not found", lane=lane, warnings=unique_strings(warnings, quote_warnings), suffix=f"relation:{index}"))
            continue
        candidate_evidence_refs = evidence_refs_for_role(packet, evidence_role)
        relation_type = normalize_relation_type_hint(str(relation.get("relation_type_hint") or "")) or "related_to"
        relation_schema_status = "not_supplied"
        relation_schema_warnings: list[str] = []
        if allowed_relation_types:
            if relation_type in allowed_relation_types:
                relation_schema_status = "selected_from_retrieved_schema"
            elif relation_type == "out_of_schema_relation":
                relation_schema_status = "model_declared_out_of_schema"
                relation_schema_warnings.append("relation_type_out_of_schema")
            else:
                relation_schema_status = "free_relation_type_not_in_retrieved_schema"
                relation_schema_warnings.append("relation_type_not_in_retrieved_schema_candidates")
        relation_rows.append(
            base_candidate_row(
                schema_version=RELATION_SCHEMA_VERSION,
                candidate_kind="graph_relation_candidate",
                packet=packet,
                source_text_excerpt=source_excerpt,
                extracted_span=quote,
                source_node_hint=str(entity_by_local_id[source_local_id].get("source_node_hint") or ""),
                target_node_hint=str(entity_by_local_id[target_local_id].get("source_node_hint") or ""),
                entity_type_hint=str(entity_by_local_id[target_local_id].get("entity_type_hint") or "unknown"),
                relation_type_hint=relation_type,
                directionality=str(relation.get("directionality_status") or relation.get("directionality") or "ambiguous"),
                source_perspective=source_perspective,
                attribution_status=str(relation.get("attribution_status") or attribution_status),
                temporal_scope=temporal_scope,
                evidence_refs=candidate_evidence_refs,
                source_refs=source_refs,
                raw_backpointer_refs=raw_backpointer_refs,
                route_refs=route_refs,
                proposal_refs=proposal_refs,
                review_refs=review_refs,
                confidence_hint=str(relation.get("confidence_hint") or "unknown"),
                inference_level_hint=str(relation.get("inference_level_hint") or "direct_inference"),
                warnings=unique_strings(warnings, quote_warnings, relation_schema_warnings, relation.get("warnings")),
                extra={
                    "candidate_text": str(relation.get("relation_description") or relation.get("why_related") or quote),
                    "source_local_entity_id": source_local_id,
                    "target_local_entity_id": target_local_id,
                    "relation_description": str(relation.get("relation_description") or ""),
                    "why_related": str(relation.get("why_related") or ""),
                    "source_text_quote": quote,
                    "evidence_role": evidence_role,
                    "primary_evidence_refs": primary_evidence_refs,
                    "context_evidence_refs": context_evidence_refs,
                    "relation_schema_status": relation_schema_status,
                    "allowed_relation_type_count": len(allowed_relation_types),
                    "lane": lane,
                    "model_id": model_id,
                    "provider": provider_name,
                    "prompt_policy_id": prompt.policy_id,
                    "prompt_hash": prompt.prompt_hash,
                    "write_permission": False,
                },
            )
        )

    for index, claim in enumerate(claim_payloads):
        if not isinstance(claim, dict):
            failure_rows.append(graph_failure_row(packet, failure_kind="schema_validation_failed", reason="claim candidate must be an object", lane=lane, warnings=warnings, suffix=f"claim:{index}"))
            continue
        subject_local_id = str(claim.get("subject_local_entity_id") or claim.get("source_local_entity_id") or "").strip()
        if subject_local_id and subject_local_id not in entity_by_local_id:
            failure_rows.append(graph_failure_row(packet, failure_kind="endpoint_validation_failed", reason=f"claim subject not found in Step 1 entities: {subject_local_id}", lane=lane, warnings=unique_strings(warnings, ["claim_subject_not_in_entity_candidates"]), suffix=f"claim:{index}"))
            continue
        quote = str(claim.get("source_text_quote") or "")
        quote_ok, quote_warnings, evidence_role = candidate_quote_warnings(packet, quote, "claim")
        if not quote_ok:
            failure_rows.append(graph_failure_row(packet, failure_kind="quote_validation_failed", reason="claim quote missing or not found", lane=lane, warnings=unique_strings(warnings, quote_warnings), suffix=f"claim:{index}"))
            continue
        candidate_evidence_refs = evidence_refs_for_role(packet, evidence_role)
        source_node = str(entity_by_local_id.get(subject_local_id, {}).get("source_node_hint") or source_perspective)
        claim_rows.append(
            base_candidate_row(
                schema_version=CLAIM_SCHEMA_VERSION,
                candidate_kind="graph_claim_candidate",
                packet=packet,
                source_text_excerpt=source_excerpt,
                extracted_span=quote,
                source_node_hint=source_node,
                target_node_hint=str(claim.get("target_node_hint") or ""),
                entity_type_hint=str(entity_by_local_id.get(subject_local_id, {}).get("entity_type_hint") or "unknown"),
                relation_type_hint="",
                directionality="unknown",
                source_perspective=source_perspective,
                attribution_status=str(claim.get("attribution_status") or attribution_status),
                temporal_scope=temporal_scope,
                evidence_refs=candidate_evidence_refs,
                source_refs=source_refs,
                raw_backpointer_refs=raw_backpointer_refs,
                route_refs=route_refs,
                proposal_refs=proposal_refs,
                review_refs=review_refs,
                confidence_hint=str(claim.get("confidence_hint") or "unknown"),
                inference_level_hint=str(claim.get("inference_level_hint") or "direct_inference"),
                warnings=unique_strings(warnings, quote_warnings, claim.get("warnings")),
                extra={
                    "candidate_text": str(claim.get("claim_text") or ""),
                    "claim_type_hint": str(claim.get("claim_type_hint") or "unknown"),
                    "claim_status": str(claim.get("status") or "unknown"),
                    "subject_local_entity_id": subject_local_id,
                    "source_text_quote": quote,
                    "evidence_role": evidence_role,
                    "primary_evidence_refs": primary_evidence_refs,
                    "context_evidence_refs": context_evidence_refs,
                    "lane": lane,
                    "model_id": model_id,
                    "provider": provider_name,
                    "prompt_policy_id": prompt.policy_id,
                    "prompt_hash": prompt.prompt_hash,
                    "write_permission": False,
                },
            )
        )

    if output_kind == "graph_bundle_candidate" and not entity_rows and not relation_rows and not claim_rows:
        failure_rows.append(
            graph_failure_row(
                packet,
                failure_kind="no_candidates",
                reason="model returned graph_bundle_candidate but no valid candidates survived validation",
                lane=lane,
                warnings=unique_strings(warnings, ["no_valid_candidates_after_validation"]),
            )
        )
    return entity_rows, relation_rows, claim_rows, failure_rows, warnings


def merge_candidate_rows(entity_rows: list[dict[str, Any]], relation_rows: list[dict[str, Any]], claim_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in entity_rows:
        key = ("entity", normalize_key(row.get("source_node_hint") or row.get("candidate_text") or ""), "")
        groups[key].append(row)
    for row in relation_rows:
        key = (
            "relation",
            normalize_key(row.get("source_node_hint") or ""),
            "|".join(
                [
                    normalize_key(row.get("relation_type_hint") or ""),
                    normalize_key(row.get("target_node_hint") or ""),
                ]
            ),
        )
        groups[key].append(row)
    for row in claim_rows:
        key = (
            "claim",
            normalize_key(row.get("source_node_hint") or ""),
            "|".join(
                [
                    normalize_key(row.get("claim_type_hint") or ""),
                    normalize_key(row.get("target_node_hint") or ""),
                ]
            ),
        )
        groups[key].append(row)

    merge_rows: list[dict[str, Any]] = []
    for (merge_kind, group_key, secondary_key), rows in groups.items():
        if len(rows) < 2:
            continue
        candidate_ids = unique_strings(*(row.get("candidate_id") for row in rows))
        evidence_refs = unique_strings(*(row.get("evidence_refs") for row in rows))
        merge_rows.append(
            {
                "schema_version": MERGE_SCHEMA_VERSION,
                "merge_candidate_id": stable_id(
                    "graph_merge_candidate",
                    f"{merge_kind}|{group_key}|{secondary_key}|{'|'.join(candidate_ids)}",
                ),
                "merge_kind": f"{merge_kind}_exact_match",
                "normalized_key": group_key if not secondary_key else f"{group_key}|{secondary_key}",
                "candidate_ids": candidate_ids,
                "candidate_count": len(candidate_ids),
                "source_packet_ids": unique_strings(*(row.get("source_packet_id") for row in rows)),
                "evidence_refs": evidence_refs,
                "source_refs": unique_strings(*(row.get("source_refs") for row in rows)),
                "raw_backpointer_refs": unique_strings(*(row.get("raw_backpointer_refs") for row in rows)),
                "route_refs": unique_strings(*(row.get("route_refs") for row in rows)),
                "proposal_refs": unique_strings(*(row.get("proposal_refs") for row in rows)),
                "review_refs": unique_strings(*(row.get("review_refs") for row in rows)),
                "confidence_hint": "medium",
                "reason": "exact normalized label/signature repetition across packets",
                "warnings": ["merge_candidate_only"],
                "graph_is_not_proof": True,
            }
        )
    return merge_rows


def source_asset_hashes(workspace: Path) -> dict[str, str | None]:
    paths = [
        workspace / "graph_v03_construction" / "graph_construction_packets.jsonl",
        workspace / "graph_v03_construction" / "graph_construction_manifest.json",
        workspace / "evidence" / "evidence.jsonl",
        workspace / "portrait" / "reviewed_units.jsonl",
        workspace / "portrait" / "normalized_candidates.jsonl",
        workspace / "portrait" / "review_decisions.jsonl",
        workspace / "memory" / "preprocessing_decisions.jsonl",
        workspace / "manifest.yaml",
    ]
    return {str(path.relative_to(workspace)): file_hash(path) for path in paths}


def prepare_outputs(output_dir: Path, duplicate_policy: str) -> None:
    if duplicate_policy not in SUPPORTED_DUPLICATE_POLICIES:
        raise ValueError(f"Unsupported duplicate_policy: {duplicate_policy}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_targets = [
        output_dir / "graph_entity_candidates.jsonl",
        output_dir / "graph_relation_candidates.jsonl",
        output_dir / "graph_claim_candidates.jsonl",
        output_dir / "graph_extraction_failures.jsonl",
        output_dir / "graph_merge_candidates.jsonl",
        output_dir / "graph_merge_decisions.jsonl",
        output_dir / MODEL_CALL_INPUTS_FILENAME,
        output_dir / "graph_extraction_manifest.json",
        output_dir / "graph_extraction_report.md",
    ]
    if duplicate_policy == "fail":
        conflicts = [path for path in existing_targets if path.exists()]
        if conflicts:
            raise FileExistsError("Existing graph extraction outputs found: " + ", ".join(str(path) for path in conflicts))


def build_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    relation_lines = [
        f"- {relation}: {count}"
        for relation, count in sorted(counts.get("relation_type_counts", {}).items())
    ]
    entity_lines = [
        f"- {entity_type}: {count}"
        for entity_type, count in sorted(counts.get("entity_type_counts", {}).items())
    ]
    claim_lines = [
        f"- {claim_type}: {count}"
        for claim_type, count in sorted(counts.get("claim_type_counts", {}).items())
    ]
    lane_lines = [f"- {lane}: {count}" for lane, count in sorted(counts.get("lane_counts", {}).items())]
    warning_lines = [f"- {warning}" for warning in manifest.get("validation_warnings", [])] or ["- none"]
    return "\n".join(
        [
            "# v0.3 Graph Relation Candidate Extraction Report",
            "",
            f"- workspace: `{manifest['workspace_id']}`",
            f"- output_dir: `{manifest['output_dir']}`",
            f"- provider: `{manifest['provider']}`",
            f"- max_gleanings: `{manifest['max_gleanings']}`",
            "- graph_is_not_proof: `true`",
            "- extraction is candidate-first and evidence-bound",
            "- LLM assist belongs here, not in summarization",
            "",
            "## Counts",
            "",
            f"- packets: {counts['packet_count']}",
            f"- entity_candidates: {counts['entity_candidate_count']}",
            f"- relation_candidates: {counts['relation_candidate_count']}",
            f"- claim_candidates: {counts['claim_candidate_count']}",
            f"- merge_candidates: {counts['merge_candidate_count']}",
            f"- failures: {counts['failure_count']}",
            f"- packets_without_candidates: {counts['packets_without_candidates']}",
            "",
            "## Lanes",
            "",
            *(lane_lines or ["- none"]),
            "",
            "## Relation Types",
            "",
            *(relation_lines or ["- none"]),
            "",
            "## Entity Types",
            "",
            *(entity_lines or ["- none"]),
            "",
            "## Claim Types",
            "",
            *(claim_lines or ["- none"]),
            "",
            "## Validation Warnings",
            "",
            *warning_lines,
            "",
            "## Notes",
            "",
            "- Relation extraction happens in the candidate layer, before merge and long before graph algorithms.",
            "- `graph_construction_packets.jsonl` remains the upstream packet layer.",
            "- `summarize_descriptions`-style summarization is a later stage, not the extraction step.",
            "- Merge rows are candidates only; no merge decisions are applied here.",
            "",
        ]
    )


def build_graph_relation_candidates(
    workspace: Path,
    *,
    project_root: Path | None = None,
    input_packets_path: Path | None = None,
    output_dir: Path | None = None,
    provider: str = "openai",
    api_mode: str | None = None,
    allow_live_api: bool = False,
    weak_model: str | None = None,
    strong_model: str | None = None,
    profile_path: Path | str = DEFAULT_GRAPH_EXTRACTION_PROFILE,
    weak_prompt_path: str | None = None,
    strong_prompt_path: str | None = None,
    env_file: Path | str = ".env",
    external_model_outputs_path: Path | None = None,
    route_decisions_path: Path | None = None,
    relation_schema_candidates_path: Path | None = None,
    max_gleanings: int = 2,
    max_items: int | None = None,
    item_offset: int = 0,
    sample_stride: int = 1,
    duplicate_policy: str = "fail",
    provider_concurrency: int = 1,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    project_root = (project_root or Path(".")).resolve()
    env_path = resolve_project_path(project_root, env_file)
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path, override=True)
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    provider_label = "mock_regex_baseline" if provider == "mock" else provider
    api_mode = api_mode or os.environ.get("OPENAI_API_MODE") or "responses"
    if api_mode not in SUPPORTED_API_MODES:
        raise ValueError(f"Unsupported api_mode: {api_mode}")
    live_api_enabled, live_api_unlock_source = resolve_live_api(provider, bool(allow_live_api))
    weak_model = weak_model or os.environ.get("OPENAI_MODEL_WEAK") or "gpt-4o-mini"
    strong_model = strong_model or os.environ.get("OPENAI_MODEL_STRONG") or "gpt-4o"
    profile = load_graph_extraction_profile(project_root, profile_path)
    if weak_prompt_path:
        profile["prompt_policies"]["weak"]["path"] = weak_prompt_path
    if strong_prompt_path:
        profile["prompt_policies"]["strong"]["path"] = strong_prompt_path
    weak_prompt = load_graph_prompt(project_root, profile, "weak")
    strong_prompt = load_graph_prompt(project_root, profile, "strong")

    output_dir = (output_dir or workspace / DEFAULT_OUTPUT_DIR_NAME).resolve()
    input_packets_path = (input_packets_path or output_dir / "graph_construction_packets.jsonl").resolve()
    if route_decisions_path is None:
        default_route_decisions = output_dir / "graph_route_decisions.jsonl"
        route_decisions_path = default_route_decisions if default_route_decisions.exists() else None
    prepare_outputs(output_dir, duplicate_policy)

    packets = read_jsonl(input_packets_path)
    if not packets:
        raise FileNotFoundError(f"No graph construction packets found at: {input_packets_path}")

    if item_offset:
        packets = packets[item_offset:]
    if sample_stride > 1:
        packets = packets[::sample_stride]
    if max_items is not None:
        packets = packets[: max(0, max_items)]

    external_payload_index: dict[str, list[dict[str, Any]]] = {}
    if provider == "external_jsonl":
        if external_model_outputs_path is None:
            raise ValueError("external_jsonl provider requires external_model_outputs_path")
        external_payload_index = load_external_payload_index(external_model_outputs_path)
    model_provider = build_provider(provider, api_mode, external_model_outputs_path) if provider == "openai" else None
    route_decisions = load_graph_route_decisions(route_decisions_path)
    relation_schema_candidate_index = load_relation_schema_candidate_index(relation_schema_candidates_path)
    provider_concurrency = max(1, int(provider_concurrency or 1))

    entity_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    claim_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    packet_warning_rows: list[dict[str, Any]] = []
    model_call_inputs: list[dict[str, Any]] = []
    model_call_result_rows: list[dict[str, Any]] = []
    lane_counts: Counter[str] = Counter()

    def call_openai_provider(packet_index: int, packet: dict[str, Any]) -> dict[str, Any]:
        lane, lane_warnings, route_decision = lane_from_route_decision(packet, route_decisions)
        prompt = strong_prompt if lane == "strong_llm_graph_extraction" else weak_prompt
        model_id = strong_model if lane == "strong_llm_graph_extraction" else weak_model
        schema_candidates = relation_schema_candidate_index.get(str(packet.get("packet_id") or ""), [])
        model_input = build_graph_model_input(packet, lane, route_decision, profile, schema_candidates)
        input_row = {
            "schema_version": "graph_v03.model_call_input.v0.1",
            "source_packet_id": str(packet.get("packet_id") or ""),
            "recommended_route": lane,
            "model_id": model_id,
            "provider": provider_label,
            "api_mode": api_mode,
            "prompt_policy_id": prompt.policy_id,
            "prompt_hash": prompt.prompt_hash,
            "prompt_path": str(prompt.path),
            "input_packet": model_input,
            "estimated_input_tokens": estimate_tokens(prompt.text + json.dumps(model_input, ensure_ascii=False)),
            "graph_is_not_proof": True,
        }
        try:
            started = time.perf_counter()
            result = model_provider.generate(prompt=prompt, model_id=model_id, input_packet=model_input) if model_provider else None
            if result is None:
                raise RuntimeError("openai provider was not initialized")
            payload, parse_errors = parse_graph_model_output(result.output_text)
            result_row = {
                "source_packet_id": str(packet.get("packet_id") or ""),
                "model_id": result.model_id,
                "provider": result.provider,
                "prompt_policy_id": prompt.policy_id,
                "prompt_hash": prompt.prompt_hash,
                "estimated_input_tokens": result.estimated_input_tokens,
                "estimated_output_tokens": result.estimated_output_tokens,
                "latency_ms": result.latency_ms if result.latency_ms is not None else int((time.perf_counter() - started) * 1000),
                "parse_errors": parse_errors,
            }
            return {
                "packet_index": packet_index,
                "input_row": input_row,
                "result_row": result_row,
                "payload": payload,
                "parse_errors": parse_errors,
                "result": result,
                "prompt": prompt,
                "lane": lane,
                "lane_warnings": lane_warnings,
                "exception": None,
            }
        except Exception as exc:  # pragma: no cover - exercised by live/provider integration.
            return {
                "packet_index": packet_index,
                "input_row": input_row,
                "result_row": None,
                "payload": None,
                "parse_errors": [],
                "result": None,
                "prompt": prompt,
                "lane": lane,
                "lane_warnings": lane_warnings,
                "exception": exc,
            }

    parallel_openai_results: dict[int, dict[str, Any]] = {}
    if provider == "openai" and provider_concurrency > 1 and len(packets) > 1:
        openai_packet_items = [
            (idx, packet)
            for idx, packet in enumerate(packets)
            if lane_from_route_decision(packet, route_decisions)[0] in {"weak_llm_graph_extraction", "strong_llm_graph_extraction"}
        ]
        if openai_packet_items:
            with ThreadPoolExecutor(max_workers=provider_concurrency) as executor:
                futures = {
                    executor.submit(call_openai_provider, idx, packet): idx
                    for idx, packet in openai_packet_items
                }
                for future in as_completed(futures):
                    result = future.result()
                    parallel_openai_results[int(result["packet_index"])] = result

    for packet_index, packet in enumerate(packets):
        lane, lane_warnings, route_decision = lane_from_route_decision(packet, route_decisions)
        lane_counts[lane] += 1
        if lane in {"skip_or_background_only", "repair_or_review", "human_review"}:
            _, _, _, routed_failures, routed_warnings = extract_packet_with_lane(
                packet,
                lane=lane,
                lane_warnings=lane_warnings,
                max_gleanings=max_gleanings,
            )
            failure_rows.extend(routed_failures)
            packet_warning_rows.append(
                {
                    "source_packet_id": str(packet.get("packet_id") or ""),
                    "warnings": unique_strings(packet_warnings(packet), lane_warnings, routed_warnings),
                }
            )
            continue
        if lane in {"weak_llm_graph_extraction", "strong_llm_graph_extraction"} and provider == "openai":
            provider_result = parallel_openai_results.get(packet_index) or call_openai_provider(packet_index, packet)
            model_call_inputs.append(provider_result["input_row"])
            try:
                if provider_result["exception"] is not None:
                    raise provider_result["exception"]
                payload = provider_result["payload"]
                parse_errors = provider_result["parse_errors"]
                result = provider_result["result"]
                prompt = provider_result["prompt"]
                if provider_result["result_row"] is not None:
                    model_call_result_rows.append(provider_result["result_row"])
                if payload is None:
                    failure_rows.append(
                        graph_failure_row(
                            packet,
                            failure_kind="schema_validation_failed",
                            reason="model output was not valid strict JSON",
                            lane=lane,
                            warnings=unique_strings(packet_warnings(packet), lane_warnings, parse_errors),
                        )
                    )
                    packet_warning_rows.append(
                        {
                            "source_packet_id": str(packet.get("packet_id") or ""),
                            "warnings": unique_strings(packet_warnings(packet), lane_warnings, parse_errors),
                        }
                    )
                    continue
                ext_entity_rows, ext_relation_rows, ext_claim_rows, ext_failure_rows, ext_warnings = normalize_model_payload(
                    packet,
                    payload,
                    lane,
                    result.model_id,
                    result.provider,
                    prompt,
                    relation_schema_candidate_index.get(str(packet.get("packet_id") or ""), []),
                )
                entity_rows.extend(ext_entity_rows)
                relation_rows.extend(ext_relation_rows)
                claim_rows.extend(ext_claim_rows)
                failure_rows.extend(ext_failure_rows)
                packet_warning_rows.append(
                    {
                        "source_packet_id": str(packet.get("packet_id") or ""),
                        "warnings": unique_strings(packet_warnings(packet), lane_warnings, ext_warnings),
                    }
                )
            except Exception as exc:
                failure_rows.append(
                    graph_failure_row(
                        packet,
                        failure_kind="provider_call_failed",
                        reason=f"{type(exc).__name__}: {exc}",
                        lane=lane,
                        warnings=unique_strings(packet_warnings(packet), lane_warnings, ["provider_error_redacted"]),
                    )
                )
                packet_warning_rows.append(
                    {
                        "source_packet_id": str(packet.get("packet_id") or ""),
                        "warnings": unique_strings(packet_warnings(packet), lane_warnings, ["provider_error_redacted"]),
                    }
                )
            continue
        if lane in {"weak_llm_graph_extraction", "strong_llm_graph_extraction"} and provider != "external_jsonl":
            _, _, _, routed_failures, routed_warnings = extract_packet_with_lane(
                packet,
                lane=lane,
                lane_warnings=lane_warnings,
                max_gleanings=max_gleanings,
            )
            failure_rows.extend(routed_failures)
            packet_warning_rows.append(
                {
                    "source_packet_id": str(packet.get("packet_id") or ""),
                    "warnings": unique_strings(packet_warnings(packet), lane_warnings, routed_warnings),
                }
            )
            continue
        if provider == "external_jsonl":
            payload_rows = external_payload_index.get(str(packet.get("packet_id") or ""), [])
            if not payload_rows:
                failure_rows.append(
                    {
                        "schema_version": FAILURE_SCHEMA_VERSION,
                        "failure_id": stable_id("graph_extraction_failure", f"{packet.get('packet_id')}|external_missing"),
                        "source_packet_id": str(packet.get("packet_id") or ""),
                        "failure_kind": "external_payload_missing",
                        "reason": "external_jsonl provider did not supply any payload for this packet",
                        "extraction_lane": lane,
                        "warnings": unique_strings(packet_warnings(packet), lane_warnings, ["external_payload_missing"]),
                        "graph_is_not_proof": True,
                    }
                )
                packet_warning_rows.append(
                    {
                        "source_packet_id": str(packet.get("packet_id") or ""),
                    "warnings": unique_strings(packet_warnings(packet), lane_warnings, ["external_payload_missing"]),
                    }
                )
                continue
            ext_entity_rows, ext_relation_rows, ext_claim_rows, ext_failure_rows, ext_warnings = normalize_external_rows(
                packet,
                payload_rows,
                lane,
            )
            entity_rows.extend(ext_entity_rows)
            relation_rows.extend(ext_relation_rows)
            claim_rows.extend(ext_claim_rows)
            failure_rows.extend(ext_failure_rows)
            packet_warning_rows.append(
                {
                    "source_packet_id": str(packet.get("packet_id") or ""),
                    "warnings": unique_strings(packet_warnings(packet), lane_warnings, ext_warnings),
                }
            )
            continue

        ext_entity_rows, ext_relation_rows, ext_claim_rows, ext_failure_rows, ext_warnings = extract_packet_with_lane(
            packet,
            lane=lane,
            lane_warnings=lane_warnings,
            max_gleanings=max_gleanings,
        )
        entity_rows.extend(ext_entity_rows)
        relation_rows.extend(ext_relation_rows)
        claim_rows.extend(ext_claim_rows)
        failure_rows.extend(ext_failure_rows)
        packet_warning_rows.append(
            {
                "source_packet_id": str(packet.get("packet_id") or ""),
                "warnings": unique_strings(packet_warnings(packet), lane_warnings, ext_warnings),
            }
        )

    merge_rows = merge_candidate_rows(entity_rows, relation_rows, claim_rows)
    merge_decisions: list[dict[str, Any]] = []
    if not merge_rows:
        merge_decisions = []

    entity_type_counts = Counter(str(row.get("entity_type_hint") or "unknown") for row in entity_rows)
    relation_type_counts = Counter(str(row.get("relation_type_hint") or "unknown") for row in relation_rows)
    claim_type_counts = Counter(str(row.get("claim_type_hint") or "unknown") for row in claim_rows)
    packets_without_candidates = sum(
        1
        for packet in packets
        if not any(
            row.get("source_packet_id") == str(packet.get("packet_id") or "")
            for row in entity_rows + relation_rows + claim_rows
        )
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "workspace_id": workspace.name,
        "output_dir": str(output_dir),
        "input_packets": str(input_packets_path),
        "provider": provider_label,
        "profile": str(profile.get("_profile_path")),
        "profile_id": profile["profile_id"],
        "profile_hash": profile.get("_profile_hash"),
        "weak_prompt_hash": weak_prompt.prompt_hash,
        "strong_prompt_hash": strong_prompt.prompt_hash,
        "api_mode": api_mode,
        "provider_concurrency": provider_concurrency,
        "live_api_enabled": live_api_enabled,
        "live_api_unlock_source": live_api_unlock_source,
        "api_key_recorded": False,
        "route_decisions": str(route_decisions_path) if route_decisions_path else None,
        "relation_schema_candidates": str(relation_schema_candidates_path) if relation_schema_candidates_path else None,
        "max_gleanings": max_gleanings,
        "counts": {
            "packet_count": len(packets),
            "entity_candidate_count": len(entity_rows),
            "relation_candidate_count": len(relation_rows),
            "claim_candidate_count": len(claim_rows),
            "merge_candidate_count": len(merge_rows),
            "failure_count": len(failure_rows),
            "packets_without_candidates": packets_without_candidates,
            "model_call_inputs": len(model_call_inputs),
            "model_call_rows": len(model_call_result_rows),
            "estimated_input_tokens": sum(int(row.get("estimated_input_tokens") or 0) for row in model_call_result_rows),
            "estimated_output_tokens": sum(int(row.get("estimated_output_tokens") or 0) for row in model_call_result_rows),
            "lane_counts": dict(sorted(lane_counts.items())),
            "entity_type_counts": dict(sorted(entity_type_counts.items())),
            "relation_type_counts": dict(sorted(relation_type_counts.items())),
            "claim_type_counts": dict(sorted(claim_type_counts.items())),
            "source_asset_hashes": source_asset_hashes(workspace),
        },
        "policies": {
            "graph_is_not_proof": True,
            "merge_performed": False,
            "llm_primary_path": "openai_schema_guided_extraction"
            if provider == "openai"
            else "external_jsonl_schema_replay"
            if provider == "external_jsonl"
            else "route_controlled_mock_regex_baseline",
            "heuristic_nlp_assist": True,
            "regex_extractor_role": "mock_regex_baseline_only",
            "route_decisions_consumed": bool(route_decisions),
            "relation_schema_candidates_consumed": bool(relation_schema_candidate_index),
            "missing_route_decisions_use_mock_fallback": not bool(route_decisions),
            "bounded_gleaning": True,
        },
        "validation_warnings": unique_strings(*(row.get("warnings") for row in packet_warning_rows)),
        "outputs": {
            "graph_entity_candidates": str(output_dir / "graph_entity_candidates.jsonl"),
            "graph_relation_candidates": str(output_dir / "graph_relation_candidates.jsonl"),
            "graph_claim_candidates": str(output_dir / "graph_claim_candidates.jsonl"),
            "graph_extraction_failures": str(output_dir / "graph_extraction_failures.jsonl"),
            "graph_merge_candidates": str(output_dir / "graph_merge_candidates.jsonl"),
            "graph_merge_decisions": str(output_dir / "graph_merge_decisions.jsonl"),
            "graph_model_call_inputs": str(output_dir / MODEL_CALL_INPUTS_FILENAME),
            "graph_extraction_manifest": str(output_dir / "graph_extraction_manifest.json"),
            "graph_extraction_report": str(output_dir / "graph_extraction_report.md"),
        },
    }

    write_jsonl(output_dir / "graph_entity_candidates.jsonl", entity_rows)
    write_jsonl(output_dir / "graph_relation_candidates.jsonl", relation_rows)
    write_jsonl(output_dir / "graph_claim_candidates.jsonl", claim_rows)
    write_jsonl(output_dir / "graph_extraction_failures.jsonl", failure_rows)
    write_jsonl(output_dir / "graph_merge_candidates.jsonl", merge_rows)
    write_jsonl(output_dir / "graph_merge_decisions.jsonl", merge_decisions)
    write_jsonl(output_dir / MODEL_CALL_INPUTS_FILENAME, model_call_inputs)
    write_json(output_dir / "graph_extraction_manifest.json", manifest)
    write_text(output_dir / "graph_extraction_report.md", build_report(manifest))

    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v0.3 graph relation candidates from construction packets.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--input-packets", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--provider", default="openai", choices=sorted(SUPPORTED_PROVIDERS))
    parser.add_argument("--api-mode", default=None, choices=sorted(SUPPORTED_API_MODES))
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--weak-model", default=None)
    parser.add_argument("--strong-model", default=None)
    parser.add_argument("--profile", default=DEFAULT_GRAPH_EXTRACTION_PROFILE)
    parser.add_argument("--weak-prompt", default=None)
    parser.add_argument("--strong-prompt", default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--external-model-outputs", default=None)
    parser.add_argument("--route-decisions", default=None)
    parser.add_argument("--relation-schema-candidates", default=None)
    parser.add_argument("--max-gleanings", type=int, default=2)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--item-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--provider-concurrency", type=int, default=1)
    parser.add_argument("--duplicate-policy", default="fail", choices=sorted(SUPPORTED_DUPLICATE_POLICIES))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    workspace = Path(args.workspace)
    input_packets = Path(args.input_packets).resolve() if args.input_packets else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    external_outputs = Path(args.external_model_outputs).resolve() if args.external_model_outputs else None
    route_decisions = Path(args.route_decisions).resolve() if args.route_decisions else None
    relation_schema_candidates = Path(args.relation_schema_candidates).resolve() if args.relation_schema_candidates else None
    build_graph_relation_candidates(
        workspace,
        project_root=project_root,
        input_packets_path=input_packets,
        output_dir=output_dir,
        provider=args.provider,
        api_mode=args.api_mode,
        allow_live_api=args.allow_live_api,
        weak_model=args.weak_model,
        strong_model=args.strong_model,
        profile_path=args.profile,
        weak_prompt_path=args.weak_prompt,
        strong_prompt_path=args.strong_prompt,
        env_file=args.env_file,
        external_model_outputs_path=external_outputs,
        route_decisions_path=route_decisions,
        relation_schema_candidates_path=relation_schema_candidates,
        max_gleanings=args.max_gleanings,
        max_items=args.max_items,
        item_offset=args.item_offset,
        sample_stride=args.sample_stride,
        duplicate_policy=args.duplicate_policy,
        provider_concurrency=args.provider_concurrency,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
