"""Config-driven dry-run memory proposal router.

This v0.1 runner is deliberately narrow:
- heuristic route decisions only;
- no LLM calls;
- no proposal generation;
- no durable memory writes;
- no automatic cascade.

The router is generic: it consumes text spans plus metadata and emits
multi-target route decisions. `target_task` is a routing parameter, not a
task-specific implementation fork.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from tools.routing_v021_feature_matrix_builder import (
    DEFAULT_RESOURCE_INVENTORY as V021_DEFAULT_RESOURCE_INVENTORY,
    InputUnit as V021InputUnit,
    build_feature_row as build_v021_feature_row,
    load_external_enumerable_resources as load_v021_external_enumerable_resources,
    load_resource_inventory as load_v021_resource_inventory,
    load_resource_store as load_v021_resource_store,
    resolve_project_path as resolve_v021_project_path,
    tfidf_features as v021_tfidf_features,
)
from tools.routing_salience_sidecar import load_static_terms as load_salience_terms
from tools.routing_salience_sidecar import salience_for_text


SUPPORTED_DUPLICATE_POLICIES = {"fail", "overwrite_generated"}
SCHEMA_VERSION = "memory_proposal_router_pilot.v0.1"
DEFAULT_POLICY_PATH = "configs/routing/memory_proposal_router/heuristic_v0.1.yaml"
REQUIRED_POLICY_KEYS = {
    "schema_version",
    "policy_id",
    "policy_version",
    "router_backend",
    "feature_extractor",
    "feature_extractor_version",
    "router_version",
    "matcher_version",
    "route_confidence_type",
    "lexicon_path",
    "target_tasks",
    "blocked_target_tasks",
    "background_section_types",
    "body_section_types",
    "excluded_retrieval_policies",
    "blocked_s2_policies",
    "punctuation_chars",
    "weights",
    "thresholds",
    "routes",
    "warnings",
    "unknown_metadata_policy",
    "guardrails",
}
REQUIRED_LEXICON_KEYS = {"schema_version", "lexicon_id", "lexicon_version", "term_groups"}
REQUIRED_ROUTES = {"background", "preprocessing_required", "script", "weak", "strong", "human"}
REQUIRED_WARNINGS = {
    "support_check_candidate",
    "s0b_section_review",
    "background_logged",
    "high_complexity",
    "needs_proposal_unit_segmentation",
    "raw_span_too_coarse",
    "length_feature_may_reflect_segmentation",
}
REQUIRED_THRESHOLDS = {
    "source_span_char_count_coarse",
    "source_span_token_count_coarse",
    "proposal_unit_char_count_gt",
    "proposal_unit_token_count_gt",
    "pronoun_terms_gte",
    "event_terms_per_100_tokens_high",
    "role_terms_per_100_tokens_high",
    "logical_markers_per_100_tokens_high",
    "importance_terms_per_100_tokens_high",
    "script_only_max_score",
    "weak_llm_max_score",
    "strong_llm_max_score",
    "route_score_max",
}
REQUIRED_WEIGHTS = {
    "proposal_unit_char_count_gt",
    "proposal_unit_token_count_gt",
    "punctuation_per_two_max",
    "event_terms_max",
    "role_terms_present",
    "pronoun_terms_gte",
    "logical_markers_present",
    "importance_terms_present",
    "unknown_low_confidence",
    "mixed_detected",
    "graph_relation_candidate_base",
    "graph_relation_candidate_with_relation_signal",
    "s2_portrait_candidate_base",
    "support_check_candidate_base",
    "s0b_section_review_uncertain",
    "event_terms_density_high",
    "role_terms_density_high",
    "logical_markers_density_high",
    "importance_terms_density_high",
}
V021_POLICY_SCHEMA = "memory_proposal_router.policy.v0.21.candidate"
V021_COMPAT_DEFAULT_POLICY = "configs/routing/memory_proposal_router/heuristic_salience_v0.2.yaml"
V021_REQUIRED_POLICY_KEYS = {
    "schema_version",
    "policy_id",
    "policy_version",
    "router_backend",
    "feature_extractor",
    "feature_extractor_version",
    "router_version",
    "matcher_version",
    "route_confidence_type",
    "target_tasks",
    "blocked_target_tasks",
    "weights",
    "thresholds",
    "routes",
    "guardrails",
}
V021_REQUIRED_WEIGHTS = {
    "value_score",
    "entity_salience_score",
    "domain_term_score",
    "keyphrase_score",
    "affect_score",
    "risk_score",
    "complexity_score",
    "low_value_score",
}
V021_REQUIRED_TASK_THRESHOLDS = {
    "low_value_skip_floor",
    "low_value_useful_floor",
    "skip_useful_floor",
    "skip_low_value_floor",
    "script_useful_floor",
    "weak_useful_floor",
    "weak_stress_floor",
    "weak_risk_floor",
    "strong_useful_floor",
    "strong_stress_floor",
    "strong_risk_floor",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def short_hash(value: str, length: int = 16) -> str:
    return sha256_text(value)[:length]


def read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def safe_workspace(project_root: Path, workspace: str) -> Path:
    path = Path(workspace)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    return ensure_within_allowed_roots(resolved, [project_root])


def resolve_config_path(project_root: Path, config_path: str | None, default_path: str) -> Path:
    path = Path(config_path or default_path)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    return ensure_within_allowed_roots(resolved, [project_root])


def ensure_within_allowed_roots(path: Path, roots: list[Path]) -> Path:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise ValueError(f"Unsafe path outside allowed roots: {resolved}")


def safe_output_dir(project_root: Path, workspace: Path, output_dir: str | None) -> Path:
    if output_dir:
        path = Path(output_dir)
        resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    else:
        resolved = workspace / "routing" / "memory_proposal_router"
    return ensure_within_allowed_roots(resolved, [workspace, project_root])


def resolve_backpointer_path(raw_path: str, workspace: Path, project_root: Path) -> Path:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [workspace / path, project_root / path]
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            safe = ensure_within_allowed_roots(candidate, [workspace, project_root])
            if safe.exists():
                return safe
        except ValueError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return ensure_within_allowed_roots(candidates[0], [workspace, project_root])


def safe_text(row: dict[str, Any]) -> str:
    for key in ("text", "content", "evidence_quote", "summary_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def sentence_count(text: str) -> int:
    parts = [part for part in re.split(r"[.!?。！？]+", text) if part.strip()]
    return max(1, len(parts))


def sentence_lengths(text: str) -> list[int]:
    return [len(part.strip()) for part in re.split(r"[.!?。！？]+", text) if part.strip()]


def rough_token_count(text: str) -> int:
    latin_words = re.findall(r"[A-Za-z0-9_]+", text)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    return len(latin_words) + len(cjk_chars)


def per_100(count: int, token_count: int) -> float:
    if token_count <= 0:
        return 0.0
    return round((count / token_count) * 100.0, 4)


def english_term_count(text: str, terms: list[str]) -> int:
    count = 0
    lowered = text.lower()
    for term in terms:
        escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
        count += len(re.findall(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", lowered))
    return count


def chinese_term_count(text: str, terms: list[str]) -> int:
    return sum(text.count(term) for term in terms)


@dataclass
class RouterPolicy:
    project_root: Path
    path: Path
    policy_hash: str
    lexicon_path: Path
    lexicon_hash: str
    data: dict[str, Any]
    lexicon: dict[str, Any]

    @property
    def policy_id(self) -> str:
        return str(self.data["policy_id"])

    @property
    def router_version(self) -> str:
        return str(self.data["router_version"])

    @property
    def feature_extractor_version(self) -> str:
        return str(self.data["feature_extractor_version"])

    @property
    def matcher_version(self) -> str:
        return str(self.data["matcher_version"])

    @property
    def route_confidence_type(self) -> str:
        return str(self.data["route_confidence_type"])

    @property
    def target_tasks(self) -> list[str]:
        return list(self.data["target_tasks"])

    @property
    def blocked_target_tasks(self) -> set[str]:
        return set(self.data.get("blocked_target_tasks", []))


def require_keys(mapping: dict[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(key for key in keys if key not in mapping)
    if missing:
        raise ValueError(f"{label} missing required keys: {', '.join(missing)}")


def validate_policy_schema(policy: dict[str, Any], lexicon: dict[str, Any]) -> None:
    require_keys(policy, REQUIRED_POLICY_KEYS, "Router policy")
    require_keys(lexicon, REQUIRED_LEXICON_KEYS, "Router lexicon")
    if not isinstance(policy["target_tasks"], list) or not policy["target_tasks"]:
        raise ValueError("Router policy target_tasks must be a non-empty list")
    for key in (
        "blocked_target_tasks",
        "background_section_types",
        "body_section_types",
        "excluded_retrieval_policies",
        "blocked_s2_policies",
        "punctuation_chars",
    ):
        if not isinstance(policy[key], list):
            raise ValueError(f"Router policy {key} must be a list")
    if policy["route_confidence_type"] != "heuristic_score_uncalibrated":
        raise ValueError("v0.1 router policy must use route_confidence_type=heuristic_score_uncalibrated")
    require_keys(policy["routes"], REQUIRED_ROUTES, "Router policy routes")
    require_keys(policy["warnings"], REQUIRED_WARNINGS, "Router policy warnings")
    require_keys(policy["thresholds"], REQUIRED_THRESHOLDS, "Router policy thresholds")
    require_keys(policy["weights"], REQUIRED_WEIGHTS, "Router policy weights")
    term_groups = lexicon["term_groups"]
    if not isinstance(term_groups, dict):
        raise ValueError("Router lexicon term_groups must be an object")
    for group_name, group in term_groups.items():
        if not isinstance(group, dict):
            raise ValueError(f"Router lexicon group {group_name} must be an object")
        for lang_key in ("english", "chinese"):
            if lang_key in group and not isinstance(group[lang_key], list):
                raise ValueError(f"Router lexicon group {group_name}.{lang_key} must be a list")


def validate_v021_policy_schema(policy: dict[str, Any], lexicon: dict[str, Any]) -> None:
    require_keys(policy, V021_REQUIRED_POLICY_KEYS, "v0.21 router policy")
    require_keys(lexicon, REQUIRED_LEXICON_KEYS, "Router lexicon")
    if policy["schema_version"] != V021_POLICY_SCHEMA:
        raise ValueError(f"v0.21 router policy must use schema_version={V021_POLICY_SCHEMA}")
    if policy["router_backend"] != "heuristic_salience_v0_21_calibrated":
        raise ValueError("v0.21 router policy must use router_backend=heuristic_salience_v0_21_calibrated")
    if policy["feature_extractor"] not in {"resource_matrix_v0_21", "salience_v0_21_calibrated"}:
        raise ValueError("v0.21 router policy must use a v0.21 resource-backed feature extractor")
    if not isinstance(policy["target_tasks"], list) or not policy["target_tasks"]:
        raise ValueError("v0.21 router policy target_tasks must be a non-empty list")
    require_keys(policy["routes"], REQUIRED_ROUTES, "v0.21 router policy routes")
    require_keys(policy["weights"], V021_REQUIRED_WEIGHTS, "v0.21 router policy weights")
    thresholds = policy.get("thresholds") or {}
    for task_key in ("s1", "s2"):
        task_thresholds = thresholds.get(task_key)
        if not isinstance(task_thresholds, dict):
            raise ValueError(f"v0.21 router policy thresholds.{task_key} must be an object")
        require_keys(task_thresholds, V021_REQUIRED_TASK_THRESHOLDS, f"v0.21 router policy thresholds.{task_key}")
    if "route_score_max" not in thresholds:
        raise ValueError("v0.21 router policy thresholds missing route_score_max")


def hydrate_v021_policy(project_root: Path, policy: dict[str, Any]) -> dict[str, Any]:
    """Add legacy-compatible metadata fields without changing v0.21 semantics."""

    base_path = resolve_config_path(project_root, V021_COMPAT_DEFAULT_POLICY, V021_COMPAT_DEFAULT_POLICY)
    base_policy = read_yaml(base_path)
    hydrated = dict(base_policy)
    hydrated.update(policy)
    hydrated.setdefault("lexicon_path", base_policy.get("lexicon_path"))
    hydrated.setdefault("salience_lexicon_path", base_policy.get("salience_lexicon_path"))
    for key in (
        "background_section_types",
        "body_section_types",
        "excluded_retrieval_policies",
        "blocked_s2_policies",
        "punctuation_chars",
        "warnings",
        "unknown_metadata_policy",
    ):
        hydrated.setdefault(key, base_policy.get(key))
    return hydrated


def load_policy(project_root: Path, policy_path: str | None) -> RouterPolicy:
    resolved_policy_path = resolve_config_path(project_root, policy_path, DEFAULT_POLICY_PATH)
    policy = read_yaml(resolved_policy_path)
    is_v021 = policy.get("schema_version") == V021_POLICY_SCHEMA
    if is_v021:
        policy = hydrate_v021_policy(project_root, policy)
    lexicon_path = resolve_config_path(project_root, str(policy["lexicon_path"]), str(policy["lexicon_path"]))
    lexicon = read_yaml(lexicon_path)
    if is_v021:
        validate_v021_policy_schema(policy, lexicon)
    else:
        validate_policy_schema(policy, lexicon)
    return RouterPolicy(
        project_root=project_root,
        path=resolved_policy_path,
        policy_hash=sha256_file(resolved_policy_path),
        lexicon_path=lexicon_path,
        lexicon_hash=sha256_file(lexicon_path),
        data=policy,
        lexicon=lexicon,
    )


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, text: str, row: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class RegexLexiconFeatureExtractor(FeatureExtractor):
    def __init__(self, policy: RouterPolicy) -> None:
        self.policy = policy
        self.term_groups = policy.lexicon.get("term_groups", {})

    def term_count(self, text: str, group_name: str) -> int:
        group = self.term_groups.get(group_name, {})
        return english_term_count(text, list(group.get("english", []))) + chinese_term_count(text, list(group.get("chinese", [])))

    def extract(self, text: str, row: dict[str, Any]) -> dict[str, Any]:
        section_type = row.get("section_type") or row.get("final_section_type")
        perspective = row.get("perspective")
        s2_policy = row.get("s2_policy") or "unknown"
        retrieval_policy = row.get("retrieval_policy") or "unknown"
        subject_risk = row.get("subject_contamination_risk") or row.get("subject_scope")
        subject_role = row.get("subject_role") or row.get("metadata", {}).get("subject_role") or "unknown"
        section_classification = row.get("section_classification") or {}
        classifier_confidence = (
            row.get("confidence")
            or row.get("script_confidence")
            or section_classification.get("confidence")
            or "unknown"
        )
        unknown_policy = self.policy.data["unknown_metadata_policy"]
        unknown_reasons: list[str] = []
        if section_type in (None, "", "unknown") or perspective in (None, "", "unknown"):
            if classifier_confidence == "low":
                unknown_reasons.append(unknown_policy["low_confidence_reason"])
            else:
                unknown_reasons.append(unknown_policy["missing_metadata_reason"])
        if perspective == "mixed" or subject_risk == "mixed":
            unknown_reasons.append(unknown_policy["mixed_reason"])

        background_sections = set(self.policy.data["background_section_types"])
        body_sections = set(self.policy.data["body_section_types"])
        excluded_retrieval = set(self.policy.data["excluded_retrieval_policies"])
        blocked_s2 = set(self.policy.data["blocked_s2_policies"])
        punctuation_chars = list(self.policy.data["punctuation_chars"])
        punctuation_count = sum(text.count(char) for char in punctuation_chars)
        source_span_text = str(row.get("source_span_text") or text)
        source_span_tokens = rough_token_count(source_span_text)
        proposal_unit_tokens = rough_token_count(text)
        lengths = sentence_lengths(text)
        max_sentence_char_count = max(lengths, default=len(text))
        avg_sentence_char_count = round(sum(lengths) / len(lengths), 4) if lengths else len(text)
        logical_marker_count = self.term_count(text, "logical_markers")
        clause_marker_count = punctuation_count + logical_marker_count
        event_term_count = self.term_count(text, "event_terms")
        role_term_count = self.term_count(text, "role_terms")
        pronoun_term_count = self.term_count(text, "pronoun_terms")
        first_person_term_count = self.term_count(text, "first_person_terms")
        other_person_term_count = self.term_count(text, "other_person_terms")
        importance_term_count = self.term_count(text, "importance_terms")
        thresholds = self.policy.data["thresholds"]
        unit_type = row.get("unit_type") or "source_span_fallback"
        source_span_is_coarse = (
            unit_type == "source_span_fallback"
            and (
                len(source_span_text) > thresholds["source_span_char_count_coarse"]
                or source_span_tokens > thresholds["source_span_token_count_coarse"]
            )
        )
        return {
            "source_span_char_count": len(source_span_text),
            "source_span_token_count": source_span_tokens,
            "proposal_unit_char_count": len(text),
            "proposal_unit_token_count": proposal_unit_tokens,
            "character_count": len(text),
            "rough_token_count": proposal_unit_tokens,
            "punctuation_count": punctuation_count,
            "punctuation_density": per_100(punctuation_count, proposal_unit_tokens),
            "sentence_count": len(lengths) or 1,
            "max_sentence_char_count": max_sentence_char_count,
            "avg_sentence_char_count": avg_sentence_char_count,
            "clause_marker_count": clause_marker_count,
            "max_sentence_length": max_sentence_char_count,
            "event_term_count": event_term_count,
            "event_terms_per_100_tokens": per_100(event_term_count, proposal_unit_tokens),
            "role_term_count": role_term_count,
            "role_terms_per_100_tokens": per_100(role_term_count, proposal_unit_tokens),
            "pronoun_term_count": pronoun_term_count,
            "pronoun_terms_per_100_tokens": per_100(pronoun_term_count, proposal_unit_tokens),
            "first_person_term_count": first_person_term_count,
            "first_person_terms_per_100_tokens": per_100(first_person_term_count, proposal_unit_tokens),
            "other_person_term_count": other_person_term_count,
            "other_person_terms_per_100_tokens": per_100(other_person_term_count, proposal_unit_tokens),
            "logical_marker_count": logical_marker_count,
            "logical_markers_per_100_tokens": per_100(logical_marker_count, proposal_unit_tokens),
            "importance_term_count": importance_term_count,
            "importance_terms_per_100_tokens": per_100(importance_term_count, proposal_unit_tokens),
            "unit_type": unit_type,
            "source_span_is_coarse": source_span_is_coarse,
            "section_type": section_type or "unknown",
            "perspective": perspective or "unknown",
            "classifier_confidence": classifier_confidence,
            "unknown_metadata_reasons": sorted(set(unknown_reasons)),
            "s2_policy": s2_policy,
            "retrieval_policy": retrieval_policy,
            "subject_contamination_signal": subject_risk or "unknown",
            "subject_role": subject_role,
            "source_layer": row.get("source_layer") or "unknown",
            "memory_class": row.get("memory_class"),
            "evidence_ref": row.get("evidence_ref"),
            "evidence_refs": row.get("evidence_refs") or [],
            "input_warnings": sorted(str(item) for item in (row.get("warnings") or [])),
            "is_non_target_subject": subject_role in {"other_participant", "third_party", "non_target"},
            "is_background_or_excluded": (
                section_type in background_sections
                or retrieval_policy in excluded_retrieval
                or s2_policy in blocked_s2
            ),
            "is_body_like": section_type in body_sections,
            "matcher_version": self.policy.matcher_version,
        }


class SalienceV02FeatureExtractor(RegexLexiconFeatureExtractor):
    def __init__(self, policy: RouterPolicy) -> None:
        super().__init__(policy)
        salience_path_value = str(policy.data.get("salience_lexicon_path") or "configs/routing/lexicons/salience_core_zh_en_v0.2.yaml")
        salience_path = Path(salience_path_value)
        self.salience_path = salience_path if salience_path.is_absolute() else policy.project_root / salience_path
        self.salience_terms = load_salience_terms(self.salience_path)

    def extract(self, text: str, row: dict[str, Any]) -> dict[str, Any]:
        features = super().extract(text, row)
        salience = salience_for_text(text, self.salience_terms, {})
        salience_scores = salience.get("salience_scores") or {}
        salience_dimensions = salience.get("salience_dimensions") or {}
        group_scores = salience.get("group_scores") or {}
        feature_summary = salience.get("feature_summary") or {}
        features.update(
            {
                "salience_feature_version": "salience_v0.2",
                "salience_dimensions": salience_dimensions,
                "salience_value_score": float(salience_dimensions.get("modeling_value_score", salience_scores.get("value_score", 0.0)) or 0.0),
                "salience_positive_score": float(salience_dimensions.get("positive_salience_score", 0.0) or 0.0),
                "salience_keyphrase_score": float(salience_dimensions.get("keyphrase_salience_score", 0.0) or 0.0),
                "salience_risk_score": float(salience_dimensions.get("risk_score", salience_scores.get("risk_score", 0.0)) or 0.0),
                "salience_constraint_score": float(salience_dimensions.get("constraint_score", 0.0) or 0.0),
                "salience_evidence_directness_score": float(salience_dimensions.get("evidence_directness_score", 0.0) or 0.0),
                "salience_attribution_signal_score": float(salience_dimensions.get("attribution_signal_score", 0.0) or 0.0),
                "salience_complexity_score": float(salience_dimensions.get("processing_complexity_score", salience_scores.get("complexity_score", 0.0)) or 0.0),
                "salience_low_value_score": float(salience_dimensions.get("low_value_score", salience_scores.get("low_value_score", 0.0)) or 0.0),
                "salience_group_scores": group_scores,
                "salience_top_keyphrases": feature_summary.get("top_keyphrases") or [],
                "salience_contains_chinese": bool((feature_summary.get("language_hints") or {}).get("contains_chinese")),
            }
        )
        return features


class ResourceMatrixV021FeatureExtractor(RegexLexiconFeatureExtractor):
    def __init__(self, policy: RouterPolicy) -> None:
        super().__init__(policy)
        inventory_path = resolve_v021_project_path(
            policy.project_root,
            str(policy.data.get("resource_inventory") or V021_DEFAULT_RESOURCE_INVENTORY),
        )
        self.inventory = load_v021_resource_inventory(policy.project_root, inventory_path)
        self.enumerable_resources = load_v021_external_enumerable_resources(self.inventory)
        self.store = load_v021_resource_store(self.inventory, self.enumerable_resources)

    def extract(self, text: str, row: dict[str, Any]) -> dict[str, Any]:
        section_type = row.get("section_type") or row.get("final_section_type")
        perspective = row.get("perspective")
        s2_policy = row.get("s2_policy") or "unknown"
        retrieval_policy = row.get("retrieval_policy") or "unknown"
        subject_role = row.get("subject_role") or row.get("metadata", {}).get("subject_role") or "unknown"
        background_sections = set(self.policy.data.get("background_section_types") or [])
        excluded_retrieval = set(self.policy.data.get("excluded_retrieval_policies") or [])
        blocked_s2 = set(self.policy.data.get("blocked_s2_policies") or [])
        token_count = rough_token_count(text)
        features = {
            "character_count": len(text),
            "rough_token_count": token_count,
            "sentence_count": sentence_count(text),
            "punctuation_count": sum(text.count(ch) for ch in self.policy.data.get("punctuation_chars", [])),
            "event_term_count": self.term_count(text, "event_terms"),
            "role_term_count": self.term_count(text, "role_terms"),
            "pronoun_term_count": self.term_count(text, "pronoun_terms"),
            "logical_marker_count": self.term_count(text, "logical_markers"),
            "importance_term_count": self.term_count(text, "importance_terms"),
            "first_person_term_count": self.term_count(text, "first_person_terms"),
            "other_person_term_count": self.term_count(text, "other_person_terms"),
            "section_type": section_type or "unknown",
            "perspective": perspective or "unknown",
            "s2_policy": s2_policy,
            "retrieval_policy": retrieval_policy,
            "subject_role": subject_role,
            "subject_contamination_risk": row.get("subject_contamination_risk") or row.get("subject_scope"),
            "source_layer": row.get("source_layer"),
            "evidence_ref": row.get("evidence_ref"),
            "evidence_refs": row.get("evidence_refs") or [],
            "raw_backpointer": row.get("raw_backpointer") or {},
            "input_warnings": row.get("warnings") or [],
            "source_span_is_coarse": False,
            "is_non_target_subject": subject_role in {"other", "other_participant", "non_target"},
            "is_background_or_excluded": (
                section_type in background_sections
                or retrieval_policy in excluded_retrieval
                or s2_policy in blocked_s2
            ),
            "matcher_version": self.policy.matcher_version,
        }
        unit_id = str(
            row.get("span_id")
            or row.get("text_unit_id")
            or row.get("evidence_ref")
            or row.get("memory_id")
            or short_hash(text)
        )
        source_path = Path(str(row.get("source_path") or self.policy.path))
        source_layer = str(row.get("source_layer") or "router_input")
        workspace_id = str(row.get("workspace_id") or row.get("record_id") or "unknown")
        unit = V021InputUnit(
            source_layer=source_layer,
            source_path=source_path,
            source_row_index=int(row.get("source_row_index") or 0),
            unit_id=unit_id,
            workspace_id=workspace_id,
            text=text,
            row=row,
        )
        tfidf_by_id = v021_tfidf_features({unit_id: text}, self.enumerable_resources)
        v021_row = build_v021_feature_row(
            unit,
            self.store,
            self.enumerable_resources,
            {},
            tfidf_by_id.get(unit_id, {"status": "not_applicable", "score": 0.0}),
        )
        axis = v021_row.get("axis_scores") or {}
        raw = v021_row.get("raw_features") or {}
        language = raw.get("language") or {}
        features.update(
            {
                "v021_feature_version": "resource_matrix_v0.21",
                "v021_axis_scores": axis,
                "v021_feature_status": v021_row.get("feature_status") or {},
                "v021_raw_features": raw,
                "v021_feature_warnings": v021_row.get("warnings") or [],
                "value_score": float(axis.get("value_score") or 0.0),
                "risk_score": float(axis.get("risk_score") or 0.0),
                "complexity_score": float(axis.get("complexity_score") or 0.0),
                "entity_salience_score": float(axis.get("entity_salience_score") or 0.0),
                "keyphrase_score": float(axis.get("keyphrase_score") or 0.0),
                "affect_score": float(axis.get("affect_score") or 0.0),
                "low_value_score": float(axis.get("low_value_score") or 0.0),
                "lexical_complexity_score": float(axis.get("lexical_complexity_score") or 0.0),
                "sentence_complexity_score": float(axis.get("sentence_complexity_score") or 0.0),
                "domain_term_score": float(axis.get("domain_term_score") or 0.0),
                "language_primary": str(language.get("primary") or "unknown"),
            }
        )
        return features


class RouterBackend(ABC):
    @abstractmethod
    def route_for_task(self, route_run_id: str, span_id: str, target_task: str, features: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class HeuristicRouterBackend(RouterBackend):
    def __init__(self, policy: RouterPolicy) -> None:
        self.policy = policy
        self.weights = policy.data["weights"]
        self.thresholds = policy.data["thresholds"]
        self.routes = policy.data["routes"]
        self.warnings = policy.data["warnings"]
        self.unknown_policy = policy.data["unknown_metadata_policy"]

    def complexity_reasons(self, features: dict[str, Any], target_task: str) -> list[str]:
        reasons: list[str] = []
        if features.get("source_span_is_coarse"):
            reasons.extend(
                [
                    self.warnings["needs_proposal_unit_segmentation"],
                    self.warnings["raw_span_too_coarse"],
                    self.warnings["length_feature_may_reflect_segmentation"],
                ]
            )
        if (
            features["proposal_unit_char_count"] > self.thresholds["proposal_unit_char_count_gt"]
            or features["proposal_unit_token_count"] > self.thresholds["proposal_unit_token_count_gt"]
        ):
            reasons.append("long_or_dense_proposal_unit")
        if features["sentence_count"] > 1 or features["punctuation_count"] >= 3:
            reasons.append("multi_clause_or_multi_sentence")
        if features["event_term_count"] >= 2:
            reasons.append("multiple_event_terms")
        if features["role_term_count"] > 0:
            reasons.append("role_terms")
        if features["pronoun_term_count"] >= self.thresholds["pronoun_terms_gte"]:
            reasons.append("pronoun_coreference")
        if features["logical_marker_count"] > 0:
            reasons.append("logical_structure")
        if features["importance_term_count"] > 0:
            reasons.append("implicit_importance_or_risk")
        if features["event_terms_per_100_tokens"] >= self.thresholds["event_terms_per_100_tokens_high"]:
            reasons.append("high_event_term_density")
        if features["role_terms_per_100_tokens"] >= self.thresholds["role_terms_per_100_tokens_high"]:
            reasons.append("high_role_term_density")
        if features["logical_markers_per_100_tokens"] >= self.thresholds["logical_markers_per_100_tokens_high"]:
            reasons.append("high_logical_marker_density")
        if features["importance_terms_per_100_tokens"] >= self.thresholds["importance_terms_per_100_tokens_high"]:
            reasons.append("high_importance_term_density")
        reasons.extend(features.get("unknown_metadata_reasons", []))
        if features.get("is_non_target_subject"):
            reasons.append("non_target_subject")
        if target_task in {"s2_portrait_candidate", "graph_relation_candidate"}:
            reasons.append(f"downstream_{target_task}_impact")
        if target_task == "support_check_candidate":
            reasons.append("future_support_check_work")
        if target_task == "s0b_section_review":
            reasons.append("s0b_section_policy_owned_review")
        return sorted(set(reasons))

    def route_key_for_score(self, score: int) -> str:
        if score <= self.thresholds["script_only_max_score"]:
            return "script"
        if score <= self.thresholds["weak_llm_max_score"]:
            return "weak"
        if score <= self.thresholds["strong_llm_max_score"]:
            return "strong"
        return "human"

    def score(self, features: dict[str, Any], target_task: str) -> int:
        score = 0
        score += self.weights["proposal_unit_char_count_gt"] if features["proposal_unit_char_count"] > self.thresholds["proposal_unit_char_count_gt"] else 0
        score += self.weights["proposal_unit_token_count_gt"] if features["proposal_unit_token_count"] > self.thresholds["proposal_unit_token_count_gt"] else 0
        score += min(self.weights["punctuation_per_two_max"], features["punctuation_count"] // 2)
        score += min(self.weights["event_terms_max"], features["event_term_count"])
        score += self.weights["role_terms_present"] if features["role_term_count"] > 0 else 0
        score += self.weights["pronoun_terms_gte"] if features["pronoun_term_count"] >= self.thresholds["pronoun_terms_gte"] else 0
        score += self.weights["logical_markers_present"] if features["logical_marker_count"] > 0 else 0
        score += self.weights["importance_terms_present"] if features["importance_term_count"] > 0 else 0
        score += self.weights["event_terms_density_high"] if features["event_terms_per_100_tokens"] >= self.thresholds["event_terms_per_100_tokens_high"] else 0
        score += self.weights["role_terms_density_high"] if features["role_terms_per_100_tokens"] >= self.thresholds["role_terms_per_100_tokens_high"] else 0
        score += self.weights["logical_markers_density_high"] if features["logical_markers_per_100_tokens"] >= self.thresholds["logical_markers_per_100_tokens_high"] else 0
        score += self.weights["importance_terms_density_high"] if features["importance_terms_per_100_tokens"] >= self.thresholds["importance_terms_per_100_tokens_high"] else 0
        reasons = set(features.get("unknown_metadata_reasons", []))
        if self.unknown_policy["low_confidence_reason"] in reasons and self.unknown_policy["low_confidence_adds_score"]:
            score += self.weights["unknown_low_confidence"]
        if self.unknown_policy["mixed_reason"] in reasons and self.unknown_policy["mixed_adds_score"]:
            score += self.weights["mixed_detected"]
        if target_task == "graph_relation_candidate":
            if features["role_term_count"] > 0 or features["logical_marker_count"] > 0:
                score += self.weights["graph_relation_candidate_with_relation_signal"]
            else:
                score += self.weights["graph_relation_candidate_base"]
        elif target_task == "s2_portrait_candidate":
            score += self.weights["s2_portrait_candidate_base"]
        elif target_task == "support_check_candidate":
            score += self.weights["support_check_candidate_base"]
        elif target_task == "s0b_section_review" and (
            self.unknown_policy["low_confidence_reason"] in reasons or features["section_type"] == "unknown"
        ):
            score += self.weights["s0b_section_review_uncertain"]
        return min(score, int(self.thresholds["route_score_max"]))

    def route_for_task(self, route_run_id: str, span_id: str, target_task: str, features: dict[str, Any]) -> dict[str, Any]:
        warnings: list[str] = []
        if target_task == "support_check_candidate":
            warnings.append(self.warnings["support_check_candidate"])
        if target_task == "s0b_section_review":
            warnings.append(self.warnings["s0b_section_review"])
        if features["is_background_or_excluded"]:
            route = self.routes["background"]
            warnings.append(self.warnings["background_logged"])
            score = 0
        elif target_task == "s2_portrait_candidate" and features.get("is_non_target_subject"):
            route = self.routes["background"]
            warnings.append("non_target_subject_skipped_for_s2_portrait")
            score = 0
        elif features.get("source_span_is_coarse"):
            route = self.routes["preprocessing_required"]
            warnings.extend(
                [
                    self.warnings["needs_proposal_unit_segmentation"],
                    self.warnings["raw_span_too_coarse"],
                    self.warnings["length_feature_may_reflect_segmentation"],
                ]
            )
            score = 0
        else:
            score = self.score(features, target_task)
            route_key = self.route_key_for_score(score)
            route = self.routes[route_key]
            if route_key == "human":
                warnings.append(self.warnings["high_complexity"])
        reasons = self.complexity_reasons(features, target_task) or ["low_complexity"]
        return {
            "route_decision_id": f"route:{short_hash(route_run_id + ':' + span_id + ':' + target_task)}",
            "target_task": target_task,
            "recommended_route": route["recommended_route"],
            "fallback_route": route["fallback_route"],
            "route_score": score,
            "route_score_max": self.thresholds["route_score_max"],
            "route_confidence": None,
            "route_confidence_type": self.policy.route_confidence_type,
            "cost_class": route["cost_class"],
            "routing_reasons": reasons,
            "write_permission": False,
            "warnings": sorted(set(warnings)),
        }


class HeuristicSalienceV02RouterBackend(HeuristicRouterBackend):
    def has_first_person_signal(self, text_features: dict[str, Any]) -> bool:
        return int(text_features.get("first_person_term_count") or 0) > 0

    def has_target_source_signal(self, features: dict[str, Any]) -> bool:
        return self.has_first_person_signal(features) or (
            features.get("source_layer") == "s1_memory_unit"
            and features.get("subject_role") == "target"
        )

    def s1_has_modeling_value(self, features: dict[str, Any]) -> bool:
        """S1 value is a gate: no value means no memory materialization lane."""
        if features.get("salience_low_value_score", 0.0) >= 5.0 and features.get("salience_positive_score", 0.0) <= 3.0:
            return False
        warnings = set(features.get("input_warnings") or [])
        if warnings.intersection({"unbalanced_bracket_fragment", "editorial_fragment", "fragment_like_sentence"}):
            if features.get("salience_positive_score", 0.0) <= 0 and int(features.get("rough_token_count") or 0) <= 3:
                return False
        if features.get("salience_positive_score", 0.0) > 0:
            return True
        if int(features.get("event_term_count") or 0) > 0:
            return True
        if self.has_first_person_signal(features) and int(features.get("rough_token_count") or 0) >= 4:
            return True
        return (
            float(features.get("salience_keyphrase_score") or 0.0) >= 2.0
            and int(features.get("rough_token_count") or 0) >= 4
            and int(features.get("character_count") or 0) >= 20
        )

    def s1_risk_score(self, features: dict[str, Any]) -> int:
        risk = 0
        warnings = set(features.get("input_warnings") or [])
        if warnings.intersection({"unbalanced_bracket_fragment", "source_text_quote_not_exact", "source_text_quote_not_in_primary_text"}):
            risk += 4
        if float(features.get("salience_risk_score") or 0.0) >= 5.0:
            risk += 4
        if float(features.get("salience_constraint_score") or 0.0) >= 5.0:
            risk += 2
        if float(features.get("salience_attribution_signal_score") or 0.0) >= 4.0:
            risk += 2
        if float(features.get("salience_complexity_score") or 0.0) >= 8.0:
            risk += 4
        elif float(features.get("salience_complexity_score") or 0.0) >= 6.0:
            risk += 2
        if int(features.get("pronoun_term_count") or 0) >= self.thresholds["pronoun_terms_gte"]:
            risk += 1
        unknown_reasons = set(features.get("unknown_metadata_reasons") or [])
        if self.unknown_policy["mixed_reason"] in unknown_reasons:
            risk += 3
        elif self.unknown_policy["low_confidence_reason"] in unknown_reasons:
            risk += 2
        return risk

    def route_for_s1_memory_candidate(self, route_run_id: str, span_id: str, features: dict[str, Any]) -> dict[str, Any]:
        warnings: list[str] = []
        if features["is_background_or_excluded"]:
            row = super().route_for_task(route_run_id, span_id, "s1_memory_candidate", features)
            row["routing_reasons"] = sorted(set(row["routing_reasons"] + self.salience_reasons(features, "s1_memory_candidate")))
            return row
        if features.get("source_span_is_coarse"):
            row = super().route_for_task(route_run_id, span_id, "s1_memory_candidate", features)
            row["routing_reasons"] = sorted(set(row["routing_reasons"] + self.salience_reasons(features, "s1_memory_candidate")))
            return row

        has_value = self.s1_has_modeling_value(features)
        risk_score = self.s1_risk_score(features)
        if not has_value:
            route = self.routes["background"]
            route_score = 0
            warnings.append("low_salience_or_low_value")
            warnings.append("s1_low_value_not_materialized")
            if risk_score >= 3:
                warnings.append("s1_low_value_high_risk_not_materialized")
            reasons = self.salience_reasons(features, "s1_memory_candidate") + ["s1_low_modeling_value"]
        elif risk_score >= 6:
            route = self.routes["strong"]
            route_score = min(int(self.thresholds["route_score_max"]), max(6, risk_score))
            warnings.append("s1_high_value_high_risk_requires_strong_llm")
            reasons = self.salience_reasons(features, "s1_memory_candidate") + ["s1_high_value_high_risk"]
        elif risk_score >= 3:
            route = self.routes["weak"]
            route_score = max(2, risk_score)
            warnings.append("s1_high_value_medium_risk_requires_llm")
            reasons = self.salience_reasons(features, "s1_memory_candidate") + ["s1_high_value_medium_risk"]
        else:
            route = self.routes["script"]
            route_score = 1
            reasons = self.salience_reasons(features, "s1_memory_candidate") + ["s1_high_value_low_risk"]

        return {
            "route_decision_id": f"route:{short_hash(route_run_id + ':' + span_id + ':s1_memory_candidate')}",
            "target_task": "s1_memory_candidate",
            "recommended_route": route["recommended_route"],
            "fallback_route": route["fallback_route"],
            "route_score": route_score,
            "route_score_max": self.thresholds["route_score_max"],
            "route_confidence": None,
            "route_confidence_type": self.policy.route_confidence_type,
            "cost_class": route["cost_class"],
            "routing_reasons": sorted(set(reasons)),
            "write_permission": False,
            "warnings": sorted(set(warnings)),
        }

    def s2_has_portrait_value(self, features: dict[str, Any]) -> bool:
        """S2 value means useful for portrait/claim/hypothesis modeling."""
        if features.get("salience_low_value_score", 0.0) >= 5.0 and features.get("salience_positive_score", 0.0) <= 3.0:
            return False
        groups = features.get("salience_group_scores") or {}
        target_source = self.has_target_source_signal(features)
        source_layer = str(features.get("source_layer") or "")
        memory_class = str(features.get("memory_class") or "")
        relation_signal = float(groups.get("relation_terms", 0.0) or 0.0) > 0
        project_signal = float(groups.get("project_tool_terms", 0.0) or 0.0) > 0
        preference_signal = float(groups.get("preference_terms", 0.0) or 0.0) > 0
        affect_signal = float(groups.get("affect_terms", 0.0) or 0.0) > 0
        event_signal = int(features.get("event_term_count") or 0) > 0 or float(groups.get("event_nominal_terms", 0.0) or 0.0) > 0
        constraint_or_risk = (
            float(features.get("salience_constraint_score") or 0.0) > 0
            or float(features.get("salience_risk_score") or 0.0) > 0
        )
        if source_layer == "s1_memory_unit" and features.get("subject_role") == "target":
            return True
        other_directed_interaction = (
            features.get("perspective") in {"author", "target"}
            and not target_source
            and int(features.get("other_person_term_count") or 0) > 0
            and float(features.get("salience_value_score") or 0.0) > 0
        )
        if preference_signal or relation_signal or affect_signal or constraint_or_risk:
            return True
        if project_signal and target_source:
            return True
        if target_source and event_signal and float(features.get("salience_value_score") or 0.0) >= 3.0:
            return True
        if source_layer == "s1_memory_unit" and memory_class == "episodic" and target_source:
            return True
        if source_layer == "s1_memory_unit" and memory_class in {"preference", "procedural", "constraint"}:
            return True
        return bool(other_directed_interaction)

    def s2_can_use_script_only(self, features: dict[str, Any], risk_score: int) -> bool:
        if risk_score > 1:
            return False
        if features.get("source_layer") != "s1_memory_unit" or features.get("subject_role") != "target":
            return False
        groups = features.get("salience_group_scores") or {}
        memory_class = str(features.get("memory_class") or "")
        direct_class = memory_class in {"preference", "procedural", "constraint"}
        direct_signal = (
            float(groups.get("preference_terms", 0.0) or 0.0) > 0
            or float(groups.get("constraint_terms", 0.0) or 0.0) > 0
        )
        return bool(direct_class and direct_signal)

    def s2_promotion_risk_score(self, features: dict[str, Any]) -> int:
        """Risk controls review strength; it does not decide whether the item has value."""
        risk = 0
        groups = features.get("salience_group_scores") or {}
        warnings = set(features.get("input_warnings") or [])
        if warnings.intersection({"unbalanced_bracket_fragment", "source_text_quote_not_exact", "source_text_quote_not_in_primary_text"}):
            risk += 4
        if float(features.get("salience_risk_score") or 0.0) >= 5.0:
            risk += 4
        elif float(features.get("salience_risk_score") or 0.0) > 0:
            risk += 2
        if float(features.get("salience_constraint_score") or 0.0) >= 5.0:
            risk += 2
        if float(features.get("salience_attribution_signal_score") or 0.0) >= 4.0:
            risk += 3
        if float(groups.get("relation_terms", 0.0) or 0.0) > 0:
            risk += 2
        if int(features.get("other_person_term_count") or 0) > 0 and not self.has_target_source_signal(features):
            risk += 3
        if float(features.get("salience_complexity_score") or 0.0) >= 8.0:
            risk += 4
        elif float(features.get("salience_complexity_score") or 0.0) >= 6.0:
            risk += 2
        unknown_reasons = set(features.get("unknown_metadata_reasons") or [])
        if self.unknown_policy["mixed_reason"] in unknown_reasons:
            risk += 3
        elif self.unknown_policy["low_confidence_reason"] in unknown_reasons:
            risk += 2
        if not features.get("evidence_ref") and not features.get("evidence_refs"):
            risk += 1
        return risk

    def route_for_s2_portrait_candidate(self, route_run_id: str, span_id: str, features: dict[str, Any]) -> dict[str, Any]:
        warnings: list[str] = []
        if features["is_background_or_excluded"] or features.get("is_non_target_subject") or features.get("source_span_is_coarse"):
            row = super().route_for_task(route_run_id, span_id, "s2_portrait_candidate", features)
            row["routing_reasons"] = sorted(set(row["routing_reasons"] + self.salience_reasons(features, "s2_portrait_candidate")))
            return row

        has_value = self.s2_has_portrait_value(features)
        risk_score = self.s2_promotion_risk_score(features)
        if not has_value:
            route = self.routes["background"]
            route_score = 0
            warnings.append("low_salience_or_low_value")
            warnings.append("s2_low_value_not_modeled")
            if risk_score >= 3:
                warnings.append("s2_low_value_high_risk_not_modeled")
            reasons = self.salience_reasons(features, "s2_portrait_candidate") + ["s2_low_portrait_value"]
        elif risk_score >= 6:
            route = self.routes["strong"]
            route_score = min(int(self.thresholds["route_score_max"]), max(6, risk_score))
            warnings.append("s2_high_value_high_promotion_risk_requires_strong_llm")
            reasons = self.salience_reasons(features, "s2_portrait_candidate") + ["s2_high_value_high_promotion_risk"]
        elif self.s2_can_use_script_only(features, risk_score):
            route = self.routes["script"]
            route_score = 1
            warnings.append("s2_direct_low_risk_script_lane")
            reasons = self.salience_reasons(features, "s2_portrait_candidate") + ["s2_high_value_low_risk_script_eligible"]
        else:
            route = self.routes["weak"]
            route_score = max(2, risk_score)
            if risk_score >= 3:
                warnings.append("s2_high_value_medium_promotion_risk_requires_llm")
                reasons = self.salience_reasons(features, "s2_portrait_candidate") + ["s2_high_value_medium_promotion_risk"]
            else:
                reasons = self.salience_reasons(features, "s2_portrait_candidate") + ["s2_high_value_low_promotion_risk"]

        return {
            "route_decision_id": f"route:{short_hash(route_run_id + ':' + span_id + ':s2_portrait_candidate')}",
            "target_task": "s2_portrait_candidate",
            "recommended_route": route["recommended_route"],
            "fallback_route": route["fallback_route"],
            "route_score": route_score,
            "route_score_max": self.thresholds["route_score_max"],
            "route_confidence": None,
            "route_confidence_type": self.policy.route_confidence_type,
            "cost_class": route["cost_class"],
            "routing_reasons": sorted(set(reasons)),
            "write_permission": False,
            "warnings": sorted(set(warnings)),
        }

    def salience_reasons(self, features: dict[str, Any], target_task: str) -> list[str]:
        reasons: list[str] = []
        groups = features.get("salience_group_scores") or {}
        if features.get("salience_low_value_score", 0.0) >= 5.0:
            reasons.append("salience_low_value_signal")
        if features.get("salience_value_score", 0.0) > 0:
            reasons.append("salience_value_signal")
        if features.get("salience_risk_score", 0.0) > 0:
            reasons.append("salience_risk_signal")
        if features.get("salience_constraint_score", 0.0) > 0:
            reasons.append("salience_constraint_signal")
        if groups.get("preference_terms", 0.0) > 0:
            reasons.append("salience_preference_signal")
        if groups.get("constraint_terms", 0.0) > 0:
            reasons.append("salience_constraint_signal")
        if groups.get("project_tool_terms", 0.0) > 0:
            reasons.append("salience_project_tool_signal")
        if groups.get("relation_terms", 0.0) > 0:
            reasons.append("salience_relation_signal")
        if self.has_target_source_signal(features):
            reasons.append("source_perspective_or_first_person_signal")
        if (
            target_task == "s2_portrait_candidate"
            and features.get("perspective") in {"author", "target"}
            and not self.has_target_source_signal(features)
            and int(features.get("other_person_term_count") or 0) > 0
            and features.get("salience_value_score", 0.0) > 0
        ):
            reasons.append("other_directed_interaction_review_signal")
        reasons.append(f"target_profile_{target_task}")
        return sorted(set(reasons))

    def no_modeling_signal(self, features: dict[str, Any], score: int, target_task: str) -> bool:
        if features.get("salience_low_value_score", 0.0) >= 5.0 and features.get("salience_positive_score", 0.0) <= 3.0:
            return True
        if target_task != "s0b_section_review" and score <= 0:
            return True
        return score <= 0 and features.get("salience_value_score", 0.0) <= 0 and int(features.get("event_term_count") or 0) <= 0

    def score(self, features: dict[str, Any], target_task: str) -> int:
        groups = features.get("salience_group_scores") or {}
        low_value = float(features.get("salience_low_value_score", 0.0) or 0.0)
        salience_value = max(0.0, float(features.get("salience_value_score", 0.0) or 0.0))
        positive_salience = max(0.0, float(features.get("salience_positive_score", 0.0) or 0.0))
        first_person = self.has_first_person_signal(features)
        event_signal = int(features.get("event_term_count") or 0) > 0 or float(groups.get("event_nominal_terms", 0.0) or 0.0) > 0
        preference_signal = float(groups.get("preference_terms", 0.0) or 0.0) > 0
        constraint_signal = float(features.get("salience_constraint_score", 0.0) or 0.0) > 0
        risk_signal = float(features.get("salience_risk_score", 0.0) or 0.0) > 0
        constraint_or_risk = constraint_signal or risk_signal
        project_signal = float(groups.get("project_tool_terms", 0.0) or 0.0) > 0
        relation_signal = float(groups.get("relation_terms", 0.0) or 0.0) > 0
        other_directed_interaction = (
            target_task == "s2_portrait_candidate"
            and features.get("perspective") in {"author", "target"}
            and not first_person
            and int(features.get("other_person_term_count") or 0) > 0
            and salience_value > 0
        )

        if low_value >= 5.0 and positive_salience <= 3.0:
            return 0

        score = 0
        if target_task == "s1_memory_candidate":
            if salience_value > 0 or event_signal:
                score += 1
            if first_person and (constraint_or_risk or project_signal):
                score += 1
            if constraint_or_risk or project_signal:
                score += 1
        elif target_task == "s2_portrait_candidate":
            if first_person and (preference_signal or constraint_or_risk or project_signal):
                score += 1
            if first_person and event_signal and salience_value >= 3.0:
                score += 1
            if constraint_or_risk and project_signal:
                score += 1
            if relation_signal and first_person:
                score += 1
            if other_directed_interaction:
                score += 2
        elif target_task == "graph_relation_candidate":
            if relation_signal or project_signal:
                score += 1
            if relation_signal and first_person:
                score += 1
        elif target_task == "support_check_candidate":
            if salience_value > 0 and (features.get("evidence_ref") or features.get("raw_backpointer")):
                score += 1
        elif target_task == "s0b_section_review":
            score += super().score(features, target_task)
        else:
            score += super().score(features, target_task)

        if features.get("salience_complexity_score", 0.0) >= 6.0:
            score += 1
        return min(score, int(self.thresholds["route_score_max"]))

    def route_for_task(self, route_run_id: str, span_id: str, target_task: str, features: dict[str, Any]) -> dict[str, Any]:
        if target_task == "s1_memory_candidate":
            return self.route_for_s1_memory_candidate(route_run_id, span_id, features)
        if target_task == "s2_portrait_candidate":
            return self.route_for_s2_portrait_candidate(route_run_id, span_id, features)
        if features["is_background_or_excluded"]:
            row = super().route_for_task(route_run_id, span_id, target_task, features)
            row["routing_reasons"] = sorted(set(row["routing_reasons"] + self.salience_reasons(features, target_task)))
            return row
        if target_task == "s2_portrait_candidate" and features.get("is_non_target_subject"):
            return super().route_for_task(route_run_id, span_id, target_task, features)
        if features.get("source_span_is_coarse"):
            return super().route_for_task(route_run_id, span_id, target_task, features)

        score = self.score(features, target_task)
        warnings: list[str] = []
        if self.no_modeling_signal(features, score, target_task):
            route = self.routes["background"]
            score = 0
            warnings.append("low_salience_or_low_value")
        else:
            route = self.routes[self.route_key_for_score(score)]
        return {
            "route_decision_id": f"route:{short_hash(route_run_id + ':' + span_id + ':' + target_task)}",
            "target_task": target_task,
            "recommended_route": route["recommended_route"],
            "fallback_route": route["fallback_route"],
            "route_score": score,
            "route_score_max": self.thresholds["route_score_max"],
            "route_confidence": None,
            "route_confidence_type": self.policy.route_confidence_type,
            "cost_class": route["cost_class"],
            "routing_reasons": self.salience_reasons(features, target_task) or ["low_salience"],
            "write_permission": False,
            "warnings": sorted(set(warnings)),
        }


class HeuristicSalienceV021CalibratedRouterBackend(RouterBackend):
    def __init__(self, policy: RouterPolicy) -> None:
        self.policy = policy
        self.weights = policy.data["weights"]
        self.thresholds = policy.data["thresholds"]
        self.routes = policy.data["routes"]

    def weighted_scores(self, features: dict[str, Any]) -> dict[str, float]:
        value = float(features.get("value_score") or 0.0)
        entity = float(features.get("entity_salience_score") or 0.0)
        domain = float(features.get("domain_term_score") or 0.0)
        keyphrase = float(features.get("keyphrase_score") or 0.0)
        affect = float(features.get("affect_score") or 0.0)
        risk = float(features.get("risk_score") or 0.0)
        complexity = float(features.get("complexity_score") or 0.0)
        low = float(features.get("low_value_score") or 0.0)
        useful = (
            value * float(self.weights.get("value_score") or 0.0)
            + entity * float(self.weights.get("entity_salience_score") or 0.0)
            + domain * float(self.weights.get("domain_term_score") or 0.0)
            + keyphrase * float(self.weights.get("keyphrase_score") or 0.0)
            + affect * float(self.weights.get("affect_score") or 0.0)
        )
        stress = (
            risk * float(self.weights.get("risk_score") or 0.0)
            + complexity * float(self.weights.get("complexity_score") or 0.0)
        )
        return {"useful": round(useful, 6), "stress": round(stress, 6), "low": round(low, 6)}

    def task_thresholds(self, target_task: str) -> dict[str, float]:
        task_key = "s1" if target_task == "s1_memory_candidate" else "s2"
        return self.thresholds[task_key]

    def route_key_for_v021(self, features: dict[str, Any], target_task: str) -> tuple[str, dict[str, float]]:
        scores = self.weighted_scores(features)
        task = self.task_thresholds(target_task)
        low = scores["low"]
        useful = scores["useful"]
        stress = scores["stress"]
        risk = float(features.get("risk_score") or 0.0)
        if low >= float(task["low_value_skip_floor"]) and useful < float(task["low_value_useful_floor"]):
            return "background", scores
        if useful < float(task["skip_useful_floor"]) and low >= float(task["skip_low_value_floor"]):
            return "background", scores
        if useful >= float(task["strong_useful_floor"]) and (
            stress >= float(task["strong_stress_floor"]) or risk >= float(task["strong_risk_floor"])
        ):
            return "strong", scores
        if useful >= float(task["weak_useful_floor"]) and (
            stress >= float(task["weak_stress_floor"]) or risk >= float(task["weak_risk_floor"])
        ):
            return "weak", scores
        if useful >= float(task["script_useful_floor"]):
            return "script", scores
        return "background", scores

    def route_score_for_key(self, route_key: str, scores: dict[str, float]) -> int:
        score_max = int(self.thresholds.get("route_score_max") or 12)
        if route_key == "background":
            return 0
        if route_key == "script":
            return 1
        if route_key == "weak":
            return max(2, min(score_max, int(round(scores["useful"] / 2.0))))
        if route_key == "strong":
            return max(6, min(score_max, int(round(max(scores["useful"], scores["stress"]) / 2.0))))
        return score_max

    def routing_reasons(self, features: dict[str, Any], target_task: str, route_key: str, scores: dict[str, float]) -> list[str]:
        reasons = [f"target_profile_{target_task}", f"v021_matrix_route_{route_key}"]
        if scores["low"] >= 5.0:
            reasons.append("v021_low_value_signal")
        if scores["useful"] > 0:
            reasons.append("v021_useful_signal")
        if scores["stress"] > 0:
            reasons.append("v021_risk_or_complexity_signal")
        if float(features.get("value_score") or 0.0) > 0:
            reasons.append("v021_value_score_signal")
        if float(features.get("entity_salience_score") or 0.0) > 0:
            reasons.append("v021_entity_salience_signal")
        if float(features.get("keyphrase_score") or 0.0) > 0:
            reasons.append("v021_keyphrase_signal")
        if float(features.get("domain_term_score") or 0.0) > 0:
            reasons.append("v021_domain_term_signal")
        if float(features.get("complexity_score") or 0.0) >= 6.0:
            reasons.append("v021_complexity_signal")
        if features.get("language_primary") in {"zh", "mixed"}:
            reasons.append("v021_chinese_or_mixed_language_signal")
        if features.get("is_background_or_excluded"):
            reasons.append("background_or_excluded_input")
        return sorted(set(reasons))

    def route_for_task(self, route_run_id: str, span_id: str, target_task: str, features: dict[str, Any]) -> dict[str, Any]:
        warnings = list(features.get("v021_feature_warnings") or [])
        if features.get("is_background_or_excluded"):
            route_key = "background"
            scores = self.weighted_scores(features)
            warnings.append("background_or_excluded_span_logged")
        elif target_task not in {"s1_memory_candidate", "s2_portrait_candidate"}:
            route_key = "weak"
            scores = self.weighted_scores(features)
            warnings.append("v021_policy_non_primary_target_task_fallback")
        else:
            route_key, scores = self.route_key_for_v021(features, target_task)
            if route_key == "background":
                warnings.append("v021_low_value_or_background")
        route = self.routes[route_key]
        route_score = self.route_score_for_key(route_key, scores)
        return {
            "route_decision_id": f"route:{short_hash(route_run_id + ':' + span_id + ':' + target_task)}",
            "target_task": target_task,
            "recommended_route": route["recommended_route"],
            "fallback_route": route["fallback_route"],
            "route_score": route_score,
            "route_score_max": int(self.thresholds.get("route_score_max") or 12),
            "route_confidence": None,
            "route_confidence_type": self.policy.route_confidence_type,
            "cost_class": route["cost_class"],
            "routing_reasons": self.routing_reasons(features, target_task, route_key, scores),
            "v021_useful_score": scores["useful"],
            "v021_stress_score": scores["stress"],
            "v021_low_value_score": scores["low"],
            "write_permission": False,
            "warnings": sorted(set(warnings)),
        }


def build_feature_extractor(policy: RouterPolicy) -> FeatureExtractor:
    registry = {
        "regex_lexicon": RegexLexiconFeatureExtractor,
        "salience_v0_2": SalienceV02FeatureExtractor,
        "resource_matrix_v0_21": ResourceMatrixV021FeatureExtractor,
        "salience_v0_21_calibrated": ResourceMatrixV021FeatureExtractor,
    }
    extractor_name = policy.data["feature_extractor"]
    if extractor_name not in registry:
        raise ValueError(f"Unsupported feature extractor: {extractor_name}")
    return registry[extractor_name](policy)


def build_router_backend(policy: RouterPolicy) -> RouterBackend:
    registry = {
        "heuristic": HeuristicRouterBackend,
        "heuristic_salience_v0_2": HeuristicSalienceV02RouterBackend,
        "heuristic_salience_v0_21_calibrated": HeuristicSalienceV021CalibratedRouterBackend,
    }
    backend_name = policy.data["router_backend"]
    if backend_name not in registry:
        raise ValueError(f"Unsupported router backend: {backend_name}")
    return registry[backend_name](policy)


def find_section_text(section_row: dict[str, Any], workspace: Path, project_root: Path) -> tuple[str, list[str]]:
    text = safe_text(section_row)
    if text:
        return text, []
    backpointer = section_row.get("raw_backpointer") or {}
    source_file = backpointer.get("source_file")
    locator = backpointer.get("locator") or {}
    if not source_file or locator.get("kind") != "text_span":
        return "", ["raw_text_unavailable"]
    raw_path = resolve_backpointer_path(str(source_file), workspace, project_root)
    try:
        raw = raw_path.read_text(encoding="utf-8-sig")
        start = int(locator.get("char_start"))
        end = int(locator.get("char_end"))
        return raw[start:end].strip(), []
    except (OSError, TypeError, ValueError) as exc:
        return "", [f"raw_text_unavailable:{type(exc).__name__}"]


def input_from_section(row: dict[str, Any], workspace: Path, project_root: Path) -> dict[str, Any]:
    text, warnings = find_section_text(row, workspace, project_root)
    return {
        "span_id": row.get("raw_span_id"),
        "raw_span_id": row.get("raw_span_id"),
        "text": text,
        "evidence_ref": None,
        "raw_backpointer": row.get("raw_backpointer") or {},
        "source_layer": "s0b_raw_span",
        "section_type": row.get("final_section_type") or row.get("section_type") or "unknown",
        "perspective": row.get("perspective") or "unknown",
        "confidence": row.get("confidence") or row.get("script_confidence") or "unknown",
        "retrieval_policy": row.get("retrieval_policy") or "unknown",
        "s2_policy": row.get("s2_policy") or "unknown",
        "warnings": warnings,
    }


def input_from_text_unit(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "span_id": row.get("text_unit_id"),
        "raw_span_id": row.get("raw_span_id"),
        "text_unit_id": row.get("text_unit_id"),
        "text": safe_text(row),
        "evidence_ref": row.get("evidence_ref") or row.get("canonical_evidence_ref"),
        "raw_backpointer": row.get("raw_backpointer") or {},
        "source_layer": "s0b_text_unit",
        "section_type": row.get("section_type") or "unknown",
        "perspective": row.get("perspective") or "unknown",
        "confidence": row.get("segmentation_confidence") or "unknown",
        "retrieval_policy": row.get("retrieval_policy") or "unknown",
        "s2_policy": row.get("s2_policy") or "unknown",
        "speaker": row.get("speaker"),
        "subject_role": row.get("subject_role"),
        "target_participant": row.get("target_participant"),
        "target_subject_ids": row.get("target_subject_ids") or [],
        "subject_ids": row.get("subject_ids") or [],
        "subject_contamination_risk": row.get("subject_contamination_risk"),
        "unit_type": row.get("unit_type") or "source_span_fallback",
        "paragraph_id": row.get("paragraph_id"),
        "sentence_id": row.get("sentence_id"),
        "parent_text_unit_id": row.get("parent_text_unit_id"),
        "previous_text_unit_id": row.get("previous_text_unit_id"),
        "next_text_unit_id": row.get("next_text_unit_id"),
        "warnings": row.get("warnings") or [],
    }


def input_from_evidence(row: dict[str, Any]) -> dict[str, Any]:
    section_classification = row.get("section_classification") or {}
    return {
        "span_id": row.get("raw_span_id") or row.get("evidence_ref"),
        "raw_span_id": row.get("raw_span_id"),
        "text": safe_text(row),
        "evidence_ref": row.get("evidence_ref") or row.get("canonical_evidence_ref"),
        "raw_backpointer": {"locator": row.get("locator"), "source_file": (row.get("locator") or {}).get("source_file")},
        "source_layer": "s1_raw_evidence",
        "section_type": row.get("section_type") or "unknown",
        "perspective": section_classification.get("perspective") or row.get("metadata", {}).get("perspective") or "unknown",
        "confidence": section_classification.get("confidence") or row.get("extraction_confidence") or "unknown",
        "retrieval_policy": row.get("retrieval_policy") or "unknown",
        "s2_policy": row.get("s2_policy") or "unknown",
        "subject_role": row.get("subject_role") or row.get("metadata", {}).get("subject_role"),
        "subject_contamination_risk": row.get("subject_contamination_risk") or row.get("subject_scope"),
        "warnings": [],
    }


def input_from_memory_unit(row: dict[str, Any]) -> dict[str, Any]:
    evidence_refs = row.get("evidence_refs") or []
    evidence_ref = row.get("evidence_ref") or (evidence_refs[0] if evidence_refs else None)
    warnings = sorted(set((row.get("warnings") or []) + (row.get("processing_warnings") or []) + (row.get("risk_notes") or [])))
    return {
        "span_id": row.get("memory_id"),
        "memory_id": row.get("memory_id"),
        "raw_span_id": (row.get("source_span") or {}).get("raw_span_id") or row.get("raw_span_id"),
        "text_unit_id": (row.get("source_span") or {}).get("text_unit_id") or row.get("text_unit_id"),
        "text": row.get("processed_text") or row.get("content") or row.get("original_text") or row.get("original_text_excerpt") or "",
        "original_text": row.get("original_text") or row.get("original_text_excerpt") or row.get("evidence_quote") or "",
        "processed_text": row.get("processed_text") or row.get("content") or "",
        "evidence_ref": evidence_ref,
        "evidence_refs": evidence_refs,
        "raw_backpointer": row.get("raw_backpointer_refs") or row.get("raw_backpointer") or {},
        "raw_backpointer_refs": row.get("raw_backpointer_refs") or [],
        "source_layer": "s1_memory_unit",
        "section_type": row.get("section_type") or "body",
        "perspective": "target",
        "confidence": row.get("confidence") or "unknown",
        "retrieval_policy": row.get("retrieval_policy") or "default_retrieval",
        "s2_policy": row.get("s2_policy") or "candidate_allowed",
        "speaker": row.get("subject_id") or row.get("target_subject_id"),
        "subject_role": row.get("subject_role") or "target",
        "target_participant": row.get("target_subject_id") or row.get("subject_id"),
        "target_subject_ids": [row.get("target_subject_id")] if row.get("target_subject_id") else [],
        "subject_ids": [row.get("subject_id")] if row.get("subject_id") else [],
        "subject_contamination_risk": row.get("subject_contamination_risk"),
        "unit_type": "s1_memory_unit",
        "memory_class": row.get("memory_class"),
        "memory_type": row.get("memory_type"),
        "processing_method": row.get("processing_method"),
        "warnings": warnings,
    }


def merge_inputs(section_inputs: list[dict[str, Any]], evidence_inputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for item in section_inputs:
        key = item.get("raw_span_id") or item.get("span_id")
        if key:
            by_key[str(key)] = item
    for evidence in evidence_inputs:
        key = evidence.get("raw_span_id") or evidence.get("evidence_ref") or evidence.get("span_id")
        if key and key in by_key:
            merged = {**by_key[key], **{k: v for k, v in evidence.items() if v not in (None, "", [])}}
            merged["warnings"] = sorted(set((by_key[key].get("warnings") or []) + (evidence.get("warnings") or [])))
            merged["source_layer"] = "s1_raw_evidence"
            by_key[key] = merged
        elif key:
            by_key[key] = evidence
    return [item for item in by_key.values() if item.get("text")]


@dataclass
class RouterInputs:
    project_root: Path
    workspace: Path
    output_dir: Path
    duplicate_policy: str
    route_run_id: str
    target_tasks: list[str]
    policy: RouterPolicy


def load_inputs(args: argparse.Namespace) -> RouterInputs:
    project_root = Path(args.project_root).resolve()
    workspace = safe_workspace(project_root, args.workspace)
    output_dir = safe_output_dir(project_root, workspace, args.output_dir)
    if args.duplicate_policy not in SUPPORTED_DUPLICATE_POLICIES:
        raise ValueError(f"Unsupported duplicate_policy: {args.duplicate_policy}")
    policy = load_policy(project_root, args.policy)
    target_tasks = [item.strip() for item in (args.target_tasks or ",".join(policy.target_tasks)).split(",") if item.strip()]
    blocked = policy.blocked_target_tasks.intersection(target_tasks)
    if blocked:
        raise ValueError(f"Blocked v0.1 target task(s): {', '.join(sorted(blocked))}")
    route_run_id = args.run_id or f"memory-proposal-router:{workspace.name}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    return RouterInputs(
        project_root=project_root,
        workspace=workspace,
        output_dir=output_dir,
        duplicate_policy=args.duplicate_policy,
        route_run_id=route_run_id,
        target_tasks=target_tasks,
        policy=policy,
    )


def prepare_outputs(output_dir: Path, duplicate_policy: str) -> None:
    outputs = [
        output_dir / "route_decisions.jsonl",
        output_dir / "route_run_manifest.json",
        output_dir / "route_summary.md",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and duplicate_policy == "fail":
        raise FileExistsError("Existing route outputs found: " + ", ".join(str(path) for path in existing))
    if duplicate_policy == "overwrite_generated":
        for path in existing:
            path.unlink()


def build_route_decisions(inputs: RouterInputs) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    section_rows = read_jsonl(inputs.workspace / "raw" / "organization" / "section_map.jsonl")
    text_unit_rows = read_jsonl(inputs.workspace / "raw" / "organization" / "text_units.jsonl")
    evidence_rows = read_jsonl(inputs.workspace / "evidence" / "evidence.jsonl")
    memory_unit_rows = read_jsonl(inputs.workspace / "memory" / "memory_units.jsonl")
    if inputs.target_tasks == ["s2_portrait_candidate"] and memory_unit_rows:
        items = [input_from_memory_unit(row) for row in memory_unit_rows]
        input_source = "s1_memory_units_for_s2"
    elif text_unit_rows:
        sentence_units = [row for row in text_unit_rows if row.get("unit_type") == "sentence"]
        paragraph_units = [row for row in text_unit_rows if row.get("unit_type") == "paragraph"]
        preferred_units = sentence_units or paragraph_units
        items = [input_from_text_unit(row) for row in preferred_units]
        input_source = "text_units_sentence_first" if sentence_units else "text_units_paragraph_fallback"
    else:
        section_inputs = [input_from_section(row, inputs.workspace, inputs.project_root) for row in section_rows]
        evidence_inputs = [input_from_evidence(row) for row in evidence_rows]
        items = merge_inputs(section_inputs, evidence_inputs)
        input_source = "section_or_evidence_fallback"
    extractor = build_feature_extractor(inputs.policy)
    router = build_router_backend(inputs.policy)
    created_at = now_iso()
    decisions: list[dict[str, Any]] = []

    for item in items:
        span_id = item.get("span_id") or item.get("evidence_ref") or item.get("raw_span_id")
        if not span_id:
            continue
        features = extractor.extract(str(item.get("text") or ""), item)
        task_routes = [
            router.route_for_task(inputs.route_run_id, str(span_id), target_task, features)
            for target_task in inputs.target_tasks
        ]
        decisions.append(
            {
                "schema_version": SCHEMA_VERSION,
                "route_run_id": inputs.route_run_id,
                "router_policy_id": inputs.policy.policy_id,
                "router_policy_path": str(inputs.policy.path),
                "router_policy_hash": inputs.policy.policy_hash,
                "lexicon_path": str(inputs.policy.lexicon_path),
                "lexicon_hash": inputs.policy.lexicon_hash,
                "router_version": inputs.policy.router_version,
                "feature_extractor_version": inputs.policy.feature_extractor_version,
                "matcher_version": inputs.policy.matcher_version,
                "created_at": created_at,
                "span_id": span_id,
                "memory_id": item.get("memory_id"),
                "raw_span_id": item.get("raw_span_id"),
                "text_unit_id": item.get("text_unit_id"),
                "text": item.get("text"),
                "original_text": item.get("original_text"),
                "processed_text": item.get("processed_text"),
                "evidence_ref": item.get("evidence_ref"),
                "evidence_refs": item.get("evidence_refs") or ([item.get("evidence_ref")] if item.get("evidence_ref") else []),
                "raw_backpointer": item.get("raw_backpointer") or {},
                "raw_backpointer_refs": item.get("raw_backpointer_refs") or [],
                "source_layer": item.get("source_layer"),
                "section_type": item.get("section_type"),
                "perspective": item.get("perspective"),
                "retrieval_policy": item.get("retrieval_policy"),
                "s2_policy": item.get("s2_policy"),
                "speaker": item.get("speaker"),
                "subject_role": item.get("subject_role"),
                "target_participant": item.get("target_participant"),
                "target_subject_ids": item.get("target_subject_ids") or [],
                "subject_ids": item.get("subject_ids") or [],
                "subject_contamination_risk": item.get("subject_contamination_risk"),
                "unit_type": item.get("unit_type"),
                "memory_class": item.get("memory_class"),
                "memory_type": item.get("memory_type"),
                "processing_method": item.get("processing_method"),
                "paragraph_id": item.get("paragraph_id"),
                "sentence_id": item.get("sentence_id"),
                "parent_text_unit_id": item.get("parent_text_unit_id"),
                "previous_text_unit_id": item.get("previous_text_unit_id"),
                "next_text_unit_id": item.get("next_text_unit_id"),
                "features": features,
                "task_routes": task_routes,
                "warnings": item.get("warnings") or [],
            }
        )

    route_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    background_logged = 0
    for decision in decisions:
        for route in decision["task_routes"]:
            route_counts[route["recommended_route"]] += 1
            task_counts[route["target_task"]] += 1
            if route["recommended_route"] == "skip_or_background_only":
                background_logged += 1

    guardrails = inputs.policy.data["guardrails"]
    manifest = {
        "schema_version": "memory_proposal_router_manifest.v0.1",
        "route_run_id": inputs.route_run_id,
        "router_policy_id": inputs.policy.policy_id,
        "router_policy_path": str(inputs.policy.path),
        "router_policy_hash": inputs.policy.policy_hash,
        "lexicon_path": str(inputs.policy.lexicon_path),
        "lexicon_hash": inputs.policy.lexicon_hash,
        "router_version": inputs.policy.router_version,
        "feature_extractor_version": inputs.policy.feature_extractor_version,
        "matcher_version": inputs.policy.matcher_version,
        "workspace": str(inputs.workspace),
        "output_dir": str(inputs.output_dir),
        "created_at": created_at,
        "input_source": input_source,
        "target_tasks": inputs.target_tasks,
        "s3_hypothesis_candidate_emitted": bool(guardrails["s3_hypothesis_candidate_emitted"]),
        "llm_calls_executed": bool(guardrails["llm_calls_executed"]),
        "proposal_generation_executed": bool(guardrails["proposal_generation_executed"]),
        "durable_writes_executed": bool(guardrails["durable_writes_executed"]),
        "automatic_cascade_executed": bool(guardrails["automatic_cascade_executed"]),
        "route_confidence_type": inputs.policy.route_confidence_type,
        "support_check_candidate_sets_support_status": bool(guardrails["support_check_candidate_sets_support_status"]),
        "write_permission": bool(guardrails["write_permission"]),
        "counts": {
            "input_items": len(items),
            "route_decisions": len(decisions),
            "task_route_count": sum(task_counts.values()),
            "background_or_skipped_task_routes": background_logged,
            "route_counts": dict(route_counts),
            "task_counts": dict(task_counts),
        },
    }
    return decisions, manifest


def render_summary(manifest: dict[str, Any], decisions: list[dict[str, Any]]) -> str:
    route_counts = manifest["counts"]["route_counts"]
    task_counts = manifest["counts"]["task_counts"]
    by_task: dict[str, Counter[str]] = defaultdict(Counter)
    for decision in decisions:
        for route in decision["task_routes"]:
            by_task[route["target_task"]][route["recommended_route"]] += 1

    lines = [
        "# Memory Proposal Router Dry-Run Summary",
        "",
        f"- route_run_id: `{manifest['route_run_id']}`",
        f"- router_policy_id: `{manifest['router_policy_id']}`",
        f"- workspace: `{manifest['workspace']}`",
        f"- input_items: {manifest['counts']['input_items']}",
        f"- route_decisions: {manifest['counts']['route_decisions']}",
        f"- task_route_count: {manifest['counts']['task_route_count']}",
        f"- route_confidence_type: `{manifest['route_confidence_type']}`",
        "",
        "## Boundaries",
        "",
        "- No LLM calls executed.",
        "- No proposal generation executed.",
        "- No durable writes executed.",
        "- No automatic cascade executed.",
        "- `support_check_candidate` routes future support-check work only; it does not set support status.",
        "- `s3_hypothesis_candidate` is not emitted in v0.1.",
        "- `write_permission=false` for all route decisions.",
        "",
        "## Route Counts",
        "",
    ]
    for route_name, count in sorted(route_counts.items()):
        lines.append(f"- {route_name}: {count}")
    lines.extend(["", "## Task Counts", ""])
    for task_name, count in sorted(task_counts.items()):
        lines.append(f"- {task_name}: {count}")
    lines.extend(["", "## Route Counts By Task", ""])
    for task_name, counts in sorted(by_task.items()):
        lines.append(f"### {task_name}")
        lines.append("")
        for route_name, count in sorted(counts.items()):
            lines.append(f"- {route_name}: {count}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def run_router(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_inputs(args)
    prepare_outputs(inputs.output_dir, inputs.duplicate_policy)
    decisions, manifest = build_route_decisions(inputs)
    write_jsonl(inputs.output_dir / "route_decisions.jsonl", decisions)
    write_json(inputs.output_dir / "route_run_manifest.json", manifest)
    (inputs.output_dir / "route_summary.md").write_text(render_summary(manifest, decisions), encoding="utf-8")
    return {
        "workspace": str(inputs.workspace),
        "output_dir": str(inputs.output_dir),
        "route_decisions": str(inputs.output_dir / "route_decisions.jsonl"),
        "route_run_manifest": str(inputs.output_dir / "route_run_manifest.json"),
        "route_summary": str(inputs.output_dir / "route_summary.md"),
        "counts": manifest["counts"],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the dry-run memory proposal router.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--policy", default=DEFAULT_POLICY_PATH)
    parser.add_argument("--target-tasks", default=None)
    parser.add_argument("--duplicate-policy", default="fail", choices=sorted(SUPPORTED_DUPLICATE_POLICIES))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    print(json.dumps(run_router(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
