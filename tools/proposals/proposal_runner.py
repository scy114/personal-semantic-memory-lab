"""Generic AI-assisted proposal runner.

This v0.1 runner connects memory proposal router decisions to reviewable
AI-assisted proposal outputs through a task profile. It is intentionally
conservative:
- no durable memory writes;
- no reviewed portrait unit writes;
- no current_portrait writes;
- no graph truth writes;
- no automatic acceptance;
- all outputs keep write_permission=false.

The default provider is `openai`, the OpenAI-compatible provider lane. Use
`mock` explicitly for fixture tests, dry-run plumbing checks, and deterministic
regression only.

The first enabled profile is `s2_portrait_candidate`; the runner itself is not
Step-2-specific.
"""

from __future__ import annotations

import argparse
import http.client
import hashlib
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - lcoral has python-dotenv, fallback keeps import safe.
    load_dotenv = None  # type: ignore[assignment]


SUPPORTED_DUPLICATE_POLICIES = {"fail", "overwrite_generated"}
SUPPORTED_PROVIDERS = {"mock", "mock_invalid", "openai", "external_jsonl"}
SUPPORTED_API_MODES = {"responses", "chat_completions"}
DEFAULT_PROFILE = "configs/proposals/s2_portrait_candidate.v0.1.json"
MODEL_CALL_INPUTS_FILENAME = "model_call_inputs.jsonl"
OUTPUT_KIND_TO_EPISTEMIC_STATUS = {
    "memory_candidate": "memory_candidate",
    "portrait_fact_candidate": "fact_candidate",
    "portrait_hypothesis_candidate": "hypothesis",
    "reject": "reject",
    "model_uncertain": "uncertain",
    "needs_human_review": "human_review_required",
    "skipped": "skipped",
    "model_failure": "failure",
}
SCRIPT_ONLY_WEAK_UPGRADE_PATTERNS = [
    r"\bi\s+(also\s+)?(prefer|like|love|enjoy|need|want|use|lost|have|am|can't|cannot)\b",
    r"\bi'?m\s+",
    r"\bmy\s+",
    r"\bme too\b",
    r"\bsame here\b",
    r"\bgo-to\b",
    r"\bspeaks to me\b",
    r"\blet'?s\b.*\b(plan|explore|discuss|meet|try|go)\b",
    r"\bwe should\b.*\b(plan|explore|discuss|meet|try|go)\b",
    r"\b我(也|喜欢|想|需要|用|不能|正在|计划)\b",
    r"\b我的\b",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def short_hash(value: str, length: int = 16) -> str:
    return sha256_text(value)[:length]


def source_text_preview(text: str, max_chars: int = 240) -> str:
    return " ".join(str(text or "").split())[:max_chars]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optional_file_hash(path: Path) -> str | None:
    return file_hash(path) if path.exists() else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_within(path: Path, roots: list[Path]) -> Path:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise ValueError(f"Unsafe path outside allowed roots: {resolved}")


def safe_workspace(project_root: Path, workspace: str) -> Path:
    path = Path(workspace)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    return ensure_within(resolved, [project_root])


def safe_output_dir(project_root: Path, workspace: Path, output_dir: str | None, output_dir_name: str) -> Path:
    if output_dir:
        path = Path(output_dir)
        resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    else:
        resolved = workspace / "proposals" / output_dir_name
    return ensure_within(resolved, [workspace, project_root])


def resolve_project_path(project_root: Path, value: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (project_root / path).resolve()
    return ensure_within(resolved, [project_root])


def default_route_decisions_path(workspace: Path) -> Path:
    return workspace / "routing" / "memory_proposal_router" / "route_decisions.jsonl"


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def exact_quote_warning(quote: str, text: str, context: str, max_quote_chars: int) -> list[str]:
    warnings: list[str] = []
    if quote and len(quote) > max_quote_chars:
        warnings.append("source_text_quote_too_long")
    if quote and quote not in text:
        warnings.append("source_text_quote_not_in_primary_text")
    if quote and quote not in text and quote not in context:
        warnings.append("source_text_quote_not_exact")
    return warnings


def inherited_warning_for_s2_memory_packet(warning: str) -> str:
    if warning == "source_text_quote_not_in_primary_text":
        return "s1_inherited_source_text_quote_not_in_processed_text"
    if warning == "source_text_quote_not_exact":
        return "s1_inherited_source_text_quote_not_exact"
    if warning == "source_text_quote_too_long":
        return "s1_inherited_source_text_quote_too_long"
    return warning


def inherited_warnings_for_s2_memory_packet(warnings: list[Any]) -> list[str]:
    return [inherited_warning_for_s2_memory_packet(str(warning)) for warning in warnings]


def packet_primary_text(packet: dict[str, Any]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for key in ("text", "original_text", "processed_text"):
        value = str(packet.get(key) or "")
        if value and value not in seen:
            parts.append(value)
            seen.add(value)
    return "\n".join(parts)


def exact_quote_warning_for_packet(quote: str, packet: dict[str, Any], max_quote_chars: int) -> list[str]:
    return exact_quote_warning(quote, packet_primary_text(packet), local_context_text(packet), max_quote_chars)


def local_context_text(packet: dict[str, Any]) -> str:
    return "\n".join(
        str(packet.get(key) or "")
        for key in ("parent_paragraph_text", "previous_sentence_text", "next_sentence_text")
        if packet.get(key)
    )


def quote_uses_local_context(quote: str, packet: dict[str, Any]) -> bool:
    primary_text = packet_primary_text(packet)
    if not quote or quote in primary_text:
        return False
    return quote in local_context_text(packet)


def choose_traceable_quote(payload: dict[str, Any], packet: dict[str, Any], max_quote_chars: int) -> tuple[str, list[str]]:
    quote = str(payload.get("source_text_quote") or "")[:max_quote_chars]
    if quote:
        return quote, []
    quoted_values = [str(item)[:max_quote_chars] for item in payload.get("source_text_quotes") or [] if str(item or "").strip()]
    primary_text = packet_primary_text(packet)
    context_text = local_context_text(packet)
    for candidate in quoted_values:
        if candidate in primary_text:
            return candidate, ["source_text_quote_backfilled_from_source_text_quotes"]
    for candidate in quoted_values:
        if candidate in context_text:
            return candidate, [
                "source_text_quote_backfilled_from_source_text_quotes",
                "hypothesis_uses_neighbor_context",
            ]
    if quoted_values:
        return "", ["source_text_quotes_not_in_primary_or_local_context"]
    return "", ["source_text_quote_missing"]


def normalize_non_candidate_row(row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("output_kind") or row.get("proposal_status") or "") in {
        "reject",
        "model_uncertain",
        "needs_human_review",
        "skipped",
        "model_failure",
        "human_review_required",
    }:
        row["fact_candidate_text"] = ""
        row["hypothesis_text"] = ""
        row["candidate_text"] = ""
        row["candidate_type"] = "none"
        row["candidate_type_hints"] = []
        row["hypothesis_status"] = "unknown"
        row["hypothesis_confidence"] = "unknown"
        row["hypothesis_scope"] = "unknown"
        row["commitment_level"] = "unknown"
        row["promotion_readiness"] = "unknown"
        row["memory_candidate_text"] = ""
        row["memory_class"] = "unknown"
        row["llm_assisted_s1_candidate"] = {}
    return row


def clean_model_status(status: str) -> str:
    if status == "model_uncertain":
        return "human_review_required"
    if status == "needs_human_review":
        return "human_review_required"
    if status in {"candidate", "reject"}:
        return status
    return "model_failure"


def profile_schema_family_data(profile: dict[str, Any]) -> str:
    return str(profile.get("schema_family") or ("s2_portrait_v02" if "output_kinds" in profile else "s2_portrait_v01"))


def profile_schema_family(profile: "ProposalProfile") -> str:
    return profile_schema_family_data(profile.data)


def profile_is_s1_memory_candidate(profile: "ProposalProfile") -> bool:
    return profile_schema_family(profile) == "s1_memory_candidate_proposal"


def profile_uses_output_kind(profile: "ProposalProfile") -> bool:
    return "output_kinds" in profile.data


def target_subject_packet(packet: dict[str, Any]) -> bool:
    perspective = str(packet.get("perspective") or "").lower()
    subject_role = str(packet.get("subject_role") or "").lower()
    speaker = str(packet.get("speaker") or "")
    target = str(packet.get("target_participant") or "")
    return (
        perspective in {"target", "target_subject", "author", "user"}
        or subject_role in {"target", "target_subject"}
        or (bool(speaker) and bool(target) and speaker == target)
    )


def has_script_only_upgrade_signal(packet: dict[str, Any]) -> bool:
    text = str(packet.get("text") or "")
    if not text.strip():
        return False
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in SCRIPT_ONLY_WEAK_UPGRADE_PATTERNS)


def should_upgrade_script_only_to_weak(inputs: "RunnerInputs", packet: dict[str, Any]) -> bool:
    return (
        bool(inputs.profile.data.get("script_only_may_upgrade_to_weak", False))
        and target_subject_packet(packet)
        and has_script_only_upgrade_signal(packet)
    )


def empty_review_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


@dataclass
class ProposalProfile:
    path: Path
    data: dict[str, Any]
    profile_hash: str

    @property
    def profile_id(self) -> str:
        return str(self.data["profile_id"])

    @property
    def target_task(self) -> str:
        return str(self.data["target_task"])

    @property
    def output_dir_name(self) -> str:
        return str(self.data.get("output_dir_name") or self.target_task)

    @property
    def max_quote_chars(self) -> int:
        return int(self.data.get("max_quote_chars", 300))

    @property
    def output_files(self) -> dict[str, str]:
        return dict(self.data["output_files"])

    @property
    def prompt_policies(self) -> dict[str, dict[str, str]]:
        return dict(self.data["prompt_policies"])

    def values(self, key: str) -> set[str]:
        return set(self.data[key])


@dataclass
class PromptPolicy:
    policy_id: str
    path: Path
    text: str
    prompt_hash: str


@dataclass
class ProviderResult:
    output_text: str
    model_id: str
    provider: str
    estimated_input_tokens: int
    estimated_output_tokens: int
    cache_hit: bool | None
    latency_ms: int | None


@dataclass
class RunnerInputs:
    project_root: Path
    workspace: Path
    output_dir: Path
    profile: ProposalProfile
    duplicate_policy: str
    proposal_run_id: str
    provider: str
    api_mode: str
    live_api_enabled: bool
    live_api_unlock_source: str | None
    weak_model: str
    strong_model: str
    weak_prompt: PromptPolicy
    strong_prompt: PromptPolicy
    route_decisions_path: Path
    max_items: int | None
    item_offset: int
    sample_stride: int
    text_units_path: Path
    evidence_path: Path
    section_map_path: Path
    memory_candidates_path: Path
    preprocessing_decisions_path: Path
    external_model_outputs_path: Path | None
    provider_concurrency: int


def validate_profile_schema(profile: dict[str, Any]) -> None:
    family = profile_schema_family_data(profile)
    required = {
        "schema_version",
        "profile_id",
        "target_task",
        "output_dir_name",
        "proposal_schema_version",
        "run_schema_version",
        "human_review_queue_schema_version",
        "proposal_id_prefix",
        "proposal_input_id_prefix",
        "output_files",
        "prompt_policies",
        "max_quote_chars",
        "script_only_generates_candidates",
        "human_review_calls_model_by_default",
        "proposal_statuses",
        "model_statuses",
    }
    missing = sorted(key for key in required if key not in profile)
    if missing:
        raise ValueError("Proposal profile missing required keys: " + ", ".join(missing))
    output_required = {"proposals", "manifest", "review_log", "human_review_queue", "model_output_failures", "report"}
    missing_outputs = sorted(key for key in output_required if key not in profile["output_files"])
    if missing_outputs:
        raise ValueError("Proposal profile output_files missing keys: " + ", ".join(missing_outputs))
    for prompt_key in ("weak", "strong"):
        if prompt_key not in profile["prompt_policies"]:
            raise ValueError(f"Proposal profile prompt_policies missing {prompt_key}")
        for field in ("policy_id", "path"):
            if field not in profile["prompt_policies"][prompt_key]:
                raise ValueError(f"Proposal profile prompt_policies.{prompt_key} missing {field}")
    list_keys = ["proposal_statuses", "model_statuses"]
    if family == "s1_memory_candidate_proposal":
        list_keys.extend(
            [
                "output_kinds",
                "epistemic_statuses",
                "memory_classes",
                "deterministic_processing_statuses",
                "scope_hints",
                "temporal_hints",
                "compression_levels",
                "inference_levels",
                "proposal_confidences",
            ]
        )
    else:
        list_keys.extend(
            [
                "candidate_types",
                "inference_levels",
                "claim_strengths",
                "proposal_confidences",
                "subject_contamination_risks",
                "privacy_classes",
            ]
        )
    for list_key in list_keys:
        if not isinstance(profile[list_key], list) or not profile[list_key]:
            raise ValueError(f"Proposal profile {list_key} must be a non-empty list")
    for optional_list_key in (
        "output_kinds",
        "epistemic_statuses",
        "hypothesis_statuses",
        "hypothesis_confidences",
        "hypothesis_scopes",
        "commitment_levels",
        "promotion_readinesses",
        "memory_classes",
        "deterministic_processing_statuses",
        "scope_hints",
        "temporal_hints",
        "compression_levels",
    ):
        if optional_list_key in profile and (not isinstance(profile[optional_list_key], list) or not profile[optional_list_key]):
            raise ValueError(f"Proposal profile {optional_list_key} must be a non-empty list when present")
    if profile["script_only_generates_candidates"] is not False:
        raise ValueError("proposal pilot requires script_only_generates_candidates=false")
    if "script_only_may_upgrade_to_weak" in profile and not isinstance(profile["script_only_may_upgrade_to_weak"], bool):
        raise ValueError("proposal profile script_only_may_upgrade_to_weak must be boolean when present")
    if profile["human_review_calls_model_by_default"] is not False:
        raise ValueError("proposal pilot requires human_review_calls_model_by_default=false")


def load_profile(project_root: Path, profile_path: str) -> ProposalProfile:
    resolved = resolve_project_path(project_root, profile_path)
    data = read_json(resolved)
    validate_profile_schema(data)
    return ProposalProfile(path=resolved, data=data, profile_hash=file_hash(resolved))


class ModelProvider:
    def generate(self, *, prompt: PromptPolicy, model_id: str, input_packet: dict[str, Any]) -> ProviderResult:
        raise NotImplementedError


class MockProvider(ModelProvider):
    def __init__(self, invalid: bool = False) -> None:
        self.invalid = invalid

    def generate(self, *, prompt: PromptPolicy, model_id: str, input_packet: dict[str, Any]) -> ProviderResult:
        started = time.perf_counter()
        if self.invalid:
            output = "{invalid json"
        else:
            text = str(input_packet.get("text") or "")
            max_quote_chars = int(input_packet.get("max_quote_chars") or 300)
            lower = text.lower()
            if prompt.policy_id.startswith("s1_memory_candidate"):
                if "human review" in lower:
                    output_kind = "needs_human_review"
                    memory_candidate_text = ""
                    memory_class = "unknown"
                    quote = ""
                    observations: list[str] = []
                    warnings = ["mock_provider_output", "mock_needs_human_review"]
                elif any(marker in lower for marker in ["uncertain", "ambiguous"]):
                    output_kind = "model_uncertain"
                    memory_candidate_text = ""
                    memory_class = "unknown"
                    quote = ""
                    observations = []
                    warnings = ["mock_provider_output", "mock_model_uncertain"]
                elif any(marker in lower for marker in ["thanks", "hello", "good morning", "good evening"]) and not any(
                    marker in lower for marker in ["prefer", "love", "enjoy", "plan", "goal", "lost", "started", "starting", "usually", "always", "workflow"]
                ):
                    output_kind = "reject"
                    memory_candidate_text = ""
                    memory_class = "unknown"
                    quote = text[: min(len(text), 120)]
                    observations = []
                    warnings = ["mock_provider_output", "mock_reject"]
                elif any(marker in lower for marker in ["usually", "always", "workflow", "process", "procedure"]):
                    output_kind = "memory_candidate"
                    memory_candidate_text = f"Evidence-bound procedural memory candidate: {text[:160]}"
                    memory_class = "procedural"
                    quote = text[:max_quote_chars]
                    observations = [f"Primary text describes a repeatable process or routine: {text[:160]}"]
                    warnings = ["mock_provider_output", "mock_memory_candidate"]
                elif any(marker in lower for marker in ["prefer", "love", "enjoy", "need", "want"]):
                    output_kind = "memory_candidate"
                    memory_candidate_text = f"Evidence-bound semantic memory candidate: {text[:160]}"
                    memory_class = "semantic"
                    quote = text[:max_quote_chars]
                    observations = [f"Primary text states a preference, need, or stable context: {text[:160]}"]
                    warnings = ["mock_provider_output", "mock_memory_candidate"]
                elif any(marker in lower for marker in ["plan", "goal", "start", "started", "starting", "build", "lost", "launched"]):
                    output_kind = "memory_candidate"
                    memory_candidate_text = f"Evidence-bound episodic memory candidate: {text[:160]}"
                    memory_class = "episodic"
                    quote = text[:max_quote_chars]
                    observations = [f"Primary text states an event, goal, or project: {text[:160]}"]
                    warnings = ["mock_provider_output", "mock_memory_candidate"]
                else:
                    output_kind = "reject"
                    memory_candidate_text = ""
                    memory_class = "unknown"
                    quote = text[: min(len(text), 120)]
                    observations = []
                    warnings = ["mock_provider_output", "mock_reject"]
                output = json.dumps(
                    {
                        "output_kind": output_kind,
                        "memory_candidate_text": memory_candidate_text,
                        "memory_class": memory_class,
                        "source_text_quote": quote,
                        "source_text_quotes": [quote] if quote else [],
                        "supporting_observations": observations,
                        "scope_hint": "event" if memory_class == "episodic" else "unknown",
                        "temporal_hint": "event_bound" if memory_class == "episodic" else "unknown",
                        "compression_level": "light" if output_kind == "memory_candidate" else "unknown",
                        "inference_level": "explicit" if output_kind == "memory_candidate" else "unknown",
                        "proposal_confidence": "medium" if output_kind == "memory_candidate" else "unknown",
                        "uncertainty_notes": ["mock uncertainty path"] if output_kind == "model_uncertain" else [],
                        "warnings": warnings,
                    },
                    ensure_ascii=False,
                )
            elif ".v0.2" in prompt.policy_id:
                if "human review" in lower:
                    output_kind = "needs_human_review"
                    fact_text = ""
                    hypothesis_text = ""
                    candidate_type = "none"
                    quote = ""
                    observations: list[str] = []
                    warnings = ["mock_provider_output", "mock_needs_human_review"]
                elif "uncertain" in lower:
                    output_kind = "model_uncertain"
                    fact_text = ""
                    hypothesis_text = ""
                    candidate_type = "none"
                    quote = ""
                    observations = []
                    warnings = ["mock_provider_output", "mock_model_uncertain"]
                elif any(marker in lower for marker in ["keep it up", "talented", "encouraging", "encourage"]):
                    output_kind = "portrait_hypothesis_candidate"
                    fact_text = ""
                    hypothesis_text = f"The target subject may be taking an encouraging stance in this local exchange: {text[:100]}"
                    candidate_type = "relationship_context"
                    quote = text[:max_quote_chars]
                    observations = [f"Primary text shows a local interaction stance: {text[:160]}"]
                    warnings = ["mock_provider_output", "mock_interaction_hypothesis"]
                elif any(marker in lower for marker in ["prefer", "love", "enjoy", "resolved", "important", "goal", "plan"]):
                    output_kind = "portrait_fact_candidate"
                    fact_text = f"Target subject states a portrait-relevant fact from source text: {text[:120]}"
                    hypothesis_text = ""
                    candidate_type = "project_context" if any(marker in lower for marker in ["goal", "plan", "project"]) else "preference"
                    quote = text[:max_quote_chars]
                    observations = []
                    warnings = ["mock_provider_output", "mock_fact_candidate"]
                else:
                    output_kind = "reject"
                    fact_text = ""
                    hypothesis_text = ""
                    candidate_type = "none"
                    quote = text[: min(len(text), 120)]
                    observations = []
                    warnings = ["mock_provider_output", "mock_reject"]
                output = json.dumps(
                    {
                        "output_kind": output_kind,
                        "fact_candidate_text": fact_text,
                        "hypothesis_text": hypothesis_text,
                        "source_text_quote": quote,
                        "source_text_quotes": [quote] if quote else [],
                        "supporting_observations": observations,
                        "alternative_explanations": [],
                        "uncertainty_notes": ["mock uncertainty path"] if output_kind == "model_uncertain" else [],
                        "candidate_type": candidate_type,
                        "inference_level": "explicit" if output_kind == "portrait_fact_candidate" else "direct_inference" if output_kind == "portrait_hypothesis_candidate" else "unknown",
                        "claim_strength": "direct" if output_kind == "portrait_fact_candidate" else "partial" if output_kind == "portrait_hypothesis_candidate" else "unknown",
                        "proposal_confidence": "medium",
                        "hypothesis_status": "active" if output_kind == "portrait_hypothesis_candidate" else "unknown",
                        "hypothesis_confidence": "high" if output_kind == "portrait_hypothesis_candidate" else "unknown",
                        "hypothesis_scope": "event" if output_kind == "portrait_hypothesis_candidate" else "unknown",
                        "commitment_level": "low" if output_kind == "portrait_hypothesis_candidate" else "unknown",
                        "promotion_readiness": "not_ready" if output_kind == "portrait_hypothesis_candidate" else "unknown",
                        "subject_contamination_risk": "medium" if output_kind == "portrait_hypothesis_candidate" else "unknown",
                        "privacy_class": "public_dataset",
                        "warnings": warnings,
                    },
                    ensure_ascii=False,
                )
            elif any(marker in lower for marker in ["prefer", "love", "enjoy", "resolved", "important", "great"]):
                status = "candidate"
                candidate_text = f"Potential portrait candidate from source text: {text[:120]}"
                candidate_type = "user_state" if "resolved" in lower else "preference"
                quote = text[:max_quote_chars]
                output = json.dumps(
                    {
                        "proposal_status": status,
                        "candidate_text": candidate_text,
                        "candidate_type": candidate_type,
                        "inference_level": "explicit" if status == "candidate" else "weak_inference",
                        "claim_strength": "direct" if status == "candidate" else "unknown",
                        "proposal_confidence": "medium",
                        "source_text_quote": quote,
                        "subject_contamination_risk": "unknown",
                        "privacy_class": "public_dataset",
                        "warnings": ["mock_provider_output"],
                    },
                    ensure_ascii=False,
                )
            else:
                status = "reject"
                candidate_text = ""
                candidate_type = "none"
                quote = text[: min(len(text), 120)]
                output = json.dumps(
                    {
                        "proposal_status": status,
                        "candidate_text": candidate_text,
                        "candidate_type": candidate_type,
                        "inference_level": "explicit" if status == "candidate" else "weak_inference",
                        "claim_strength": "direct" if status == "candidate" else "unknown",
                        "proposal_confidence": "medium",
                        "source_text_quote": quote,
                        "subject_contamination_risk": "unknown",
                        "privacy_class": "public_dataset",
                        "warnings": ["mock_provider_output"],
                    },
                    ensure_ascii=False,
                )
        return ProviderResult(
            output_text=output,
            model_id=model_id,
            provider="mock_invalid" if self.invalid else "mock",
            estimated_input_tokens=estimate_tokens(prompt.text + json.dumps(input_packet, ensure_ascii=False)),
            estimated_output_tokens=estimate_tokens(output),
            cache_hit=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class OpenAICompatibleProvider(ModelProvider):
    def __init__(self, api_mode: str) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for provider=openai.")
        if api_mode not in SUPPORTED_API_MODES:
            raise ValueError(f"Unsupported OpenAI-compatible api_mode: {api_mode}")
        self.api_key = api_key
        self.api_mode = api_mode
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        self.user_agent = os.environ.get("OPENAI_USER_AGENT", "curl/8.19.0")
        self.max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "2"))
        self.retry_base_seconds = float(os.environ.get("OPENAI_RETRY_BASE_SECONDS", "1.0"))
        self.retry_max_seconds = float(os.environ.get("OPENAI_RETRY_MAX_SECONDS", "12.0"))

    def generate(self, *, prompt: PromptPolicy, model_id: str, input_packet: dict[str, Any]) -> ProviderResult:
        started = time.perf_counter()
        if self.api_mode == "chat_completions":
            return self._generate_chat_completions(prompt=prompt, model_id=model_id, input_packet=input_packet, started=started)
        return self._generate_responses(prompt=prompt, model_id=model_id, input_packet=input_packet, started=started)

    def _post_json(self, endpoint: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": self.user_agent,
            },
            method="POST",
        )
        retryable_errors = (
            TimeoutError,
            ConnectionResetError,
            http.client.RemoteDisconnected,
            urllib.error.URLError,
        )
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    raw = response.read().decode("utf-8")
                return json.loads(raw)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                non_retryable_detail = any(
                    marker in detail.lower()
                    for marker in [
                        "auth_unavailable",
                        "no auth available",
                        "invalid api key",
                        "invalid_api_key",
                        "permission denied",
                        "unauthorized",
                    ]
                )
                if not non_retryable_detail and (exc.code in {408, 409, 429} or 500 <= exc.code <= 599):
                    last_exc = RuntimeError(f"OpenAI-compatible API retryable HTTP error {exc.code}: {detail}")
                    if attempt < self.max_retries:
                        retry_after = exc.headers.get("Retry-After")
                        if retry_after:
                            try:
                                delay = float(retry_after)
                            except ValueError:
                                delay = self.retry_base_seconds
                        else:
                            delay = self.retry_base_seconds * (2**attempt)
                        delay = min(self.retry_max_seconds, delay) + random.uniform(0, min(1.0, self.retry_base_seconds))
                        time.sleep(delay)
                        continue
                raise RuntimeError(f"OpenAI-compatible API error {exc.code}: {detail}") from exc
            except retryable_errors as exc:
                last_exc = exc
                if attempt >= self.max_retries:
                    break
                delay = min(self.retry_max_seconds, self.retry_base_seconds * (2**attempt))
                time.sleep(delay + random.uniform(0, min(1.0, self.retry_base_seconds)))
        raise RuntimeError("OpenAI-compatible API retryable transport failure") from last_exc

    def _generate_responses(
        self,
        *,
        prompt: PromptPolicy,
        model_id: str,
        input_packet: dict[str, Any],
        started: float,
    ) -> ProviderResult:
        body = {
            "model": model_id,
            "input": [
                {"role": "system", "content": prompt.text},
                {"role": "user", "content": json.dumps(input_packet, ensure_ascii=False)},
            ],
        }
        data = self._post_json("/responses", body)
        output_text = data.get("output_text")
        if not output_text:
            output_parts: list[str] = []
            for item in data.get("output", []):
                for content in item.get("content", []):
                    if content.get("type") in {"output_text", "text"} and content.get("text"):
                        output_parts.append(str(content["text"]))
            output_text = "\n".join(output_parts)
        if not output_text:
            raise RuntimeError("OpenAI response did not include output text.")
        usage = data.get("usage") or {}
        return ProviderResult(
            output_text=output_text,
            model_id=model_id,
            provider="openai",
            estimated_input_tokens=int(usage.get("input_tokens") or estimate_tokens(prompt.text)),
            estimated_output_tokens=int(usage.get("output_tokens") or estimate_tokens(output_text)),
            cache_hit=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )

    def _generate_chat_completions(
        self,
        *,
        prompt: PromptPolicy,
        model_id: str,
        input_packet: dict[str, Any],
        started: float,
    ) -> ProviderResult:
        body = {
            "model": model_id,
            "messages": [
                {"role": "system", "content": prompt.text},
                {"role": "user", "content": json.dumps(input_packet, ensure_ascii=False)},
            ],
        }
        data = self._post_json("/chat/completions", body)
        choices = data.get("choices") or []
        output_text = ""
        if choices:
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                output_text = content
            elif isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("text"):
                        parts.append(str(part["text"]))
                output_text = "\n".join(parts)
        if not output_text:
            raise RuntimeError("OpenAI-compatible chat completion did not include message content.")
        usage = data.get("usage") or {}
        return ProviderResult(
            output_text=output_text,
            model_id=model_id,
            provider="openai",
            estimated_input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or estimate_tokens(prompt.text)),
            estimated_output_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or estimate_tokens(output_text)),
            cache_hit=None,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


class ExternalJsonlProvider(ModelProvider):
    """Replay model outputs from JSONL keyed by proposal_input_id.

    This provider is intentionally offline. It lets a separate model runner,
    subagent, or local tool produce raw model JSON while this runner still owns
    schema validation, quote guardrails, output normalization, and manifests.
    """

    def __init__(self, outputs_path: Path) -> None:
        if not outputs_path.exists():
            raise ValueError(f"external_jsonl outputs file does not exist: {outputs_path}")
        self.outputs_path = outputs_path
        rows = read_jsonl(outputs_path)
        self.rows: dict[str, dict[str, Any]] = {}
        duplicates: list[str] = []
        for row in rows:
            proposal_input_id = str(row.get("proposal_input_id") or "")
            if not proposal_input_id:
                raise ValueError("external_jsonl row missing proposal_input_id")
            if proposal_input_id in self.rows:
                duplicates.append(proposal_input_id)
            self.rows[proposal_input_id] = row
        if duplicates:
            raise ValueError("external_jsonl duplicate proposal_input_id values: " + ", ".join(sorted(set(duplicates))))

    def generate(self, *, prompt: PromptPolicy, model_id: str, input_packet: dict[str, Any]) -> ProviderResult:
        started = time.perf_counter()
        proposal_input_id = str(input_packet.get("proposal_input_id") or "")
        row = self.rows.get(proposal_input_id)
        if row is None:
            raise RuntimeError("external_jsonl missing output for proposal_input_id")
        output_value = row.get("output_text", row.get("model_output"))
        if isinstance(output_value, dict):
            output_text = json.dumps(output_value, ensure_ascii=False)
        else:
            output_text = str(output_value or "")
        if not output_text.strip():
            raise RuntimeError("external_jsonl empty output_text")
        return ProviderResult(
            output_text=output_text,
            model_id=str(row.get("model_id") or model_id),
            provider=str(row.get("provider") or "external_jsonl"),
            estimated_input_tokens=int(row.get("estimated_input_tokens") or estimate_tokens(prompt.text + json.dumps(input_packet, ensure_ascii=False))),
            estimated_output_tokens=int(row.get("estimated_output_tokens") or estimate_tokens(output_text)),
            cache_hit=row.get("cache_hit"),
            latency_ms=int(row.get("latency_ms") or int((time.perf_counter() - started) * 1000)),
        )


def build_provider(provider_name: str, api_mode: str, external_outputs_path: Path | None = None) -> ModelProvider:
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "mock_invalid":
        return MockProvider(invalid=True)
    if provider_name == "openai":
        return OpenAICompatibleProvider(api_mode)
    if provider_name == "external_jsonl":
        if external_outputs_path is None:
            raise ValueError("provider=external_jsonl requires --external-model-outputs")
        return ExternalJsonlProvider(external_outputs_path)
    raise ValueError(f"Unsupported provider: {provider_name}")


def load_prompt(project_root: Path, relative_path: str, policy_id: str) -> PromptPolicy:
    path = resolve_project_path(project_root, relative_path)
    text = path.read_text(encoding="utf-8")
    return PromptPolicy(policy_id=policy_id, path=path, text=text, prompt_hash=file_hash(path))


def env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def resolve_live_api(provider: str, allow_live_api: bool) -> tuple[bool, str | None]:
    if provider != "openai":
        return False, None
    env_allows = env_truthy(os.environ.get("ALLOW_LIVE_API"))
    if allow_live_api:
        return True, "cli"
    if env_allows:
        return True, "env"
    raise ValueError("provider=openai requires --allow-live-api or ALLOW_LIVE_API=true in .env")


def load_inputs(args: argparse.Namespace) -> RunnerInputs:
    project_root = Path(args.project_root).resolve()
    if args.env_file and load_dotenv:
        env_path = resolve_project_path(project_root, args.env_file)
        load_dotenv(env_path)
    profile = load_profile(project_root, args.profile)
    workspace = safe_workspace(project_root, args.workspace)
    output_dir = safe_output_dir(project_root, workspace, args.output_dir, profile.output_dir_name)
    if args.duplicate_policy not in SUPPORTED_DUPLICATE_POLICIES:
        raise ValueError(f"Unsupported duplicate_policy: {args.duplicate_policy}")
    provider = args.provider or os.environ.get("OPENAI_PROVIDER") or "openai"
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    api_mode = getattr(args, "api_mode", None) or os.environ.get("OPENAI_API_MODE") or "responses"
    if api_mode not in SUPPORTED_API_MODES:
        raise ValueError(f"Unsupported api_mode: {api_mode}")
    live_api_enabled, live_api_unlock_source = resolve_live_api(provider, bool(args.allow_live_api))
    route_decisions_path = resolve_project_path(project_root, args.route_decisions) if args.route_decisions else default_route_decisions_path(workspace)
    ensure_within(route_decisions_path, [workspace, project_root])
    external_model_outputs_path = None
    if getattr(args, "external_model_outputs", None):
        external_model_outputs_path = resolve_project_path(project_root, args.external_model_outputs)
        ensure_within(external_model_outputs_path, [workspace, project_root])
    weak_model = args.weak_model or os.environ.get("OPENAI_MODEL_WEAK") or "mock-weak-model"
    strong_model = args.strong_model or os.environ.get("OPENAI_MODEL_STRONG") or "mock-strong-model"
    proposal_run_id = args.run_id or f"{profile.target_task}-proposal:{workspace.name}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    weak_prompt_profile = profile.prompt_policies["weak"]
    strong_prompt_profile = profile.prompt_policies["strong"]
    return RunnerInputs(
        project_root=project_root,
        workspace=workspace,
        output_dir=output_dir,
        profile=profile,
        duplicate_policy=args.duplicate_policy,
        proposal_run_id=proposal_run_id,
        provider=provider,
        api_mode=api_mode,
        live_api_enabled=live_api_enabled,
        live_api_unlock_source=live_api_unlock_source,
        weak_model=weak_model,
        strong_model=strong_model,
        weak_prompt=load_prompt(project_root, args.weak_prompt or weak_prompt_profile["path"], weak_prompt_profile["policy_id"]),
        strong_prompt=load_prompt(project_root, args.strong_prompt or strong_prompt_profile["path"], strong_prompt_profile["policy_id"]),
        route_decisions_path=route_decisions_path,
        max_items=args.max_items,
        item_offset=max(0, int(getattr(args, "item_offset", 0) or 0)),
        sample_stride=max(1, int(getattr(args, "sample_stride", 1) or 1)),
        text_units_path=workspace / "raw" / "organization" / "text_units.jsonl",
        evidence_path=workspace / "evidence" / "evidence.jsonl",
        section_map_path=workspace / "raw" / "organization" / "section_map.jsonl",
        memory_candidates_path=workspace / "memory" / "memory_candidates.jsonl",
        preprocessing_decisions_path=workspace / "memory" / "preprocessing_decisions.jsonl",
        external_model_outputs_path=external_model_outputs_path,
        provider_concurrency=max(1, int(getattr(args, "provider_concurrency", 1) or 1)),
    )


def output_path(inputs: RunnerInputs, key: str) -> Path:
    return inputs.output_dir / inputs.profile.output_files[key]


def model_call_inputs_path(inputs: RunnerInputs) -> Path:
    return inputs.output_dir / MODEL_CALL_INPUTS_FILENAME


def prepare_outputs(inputs: RunnerInputs) -> None:
    paths = [
        output_path(inputs, "proposals"),
        output_path(inputs, "manifest"),
        output_path(inputs, "review_log"),
        output_path(inputs, "human_review_queue"),
        output_path(inputs, "model_output_failures"),
        output_path(inputs, "report"),
        model_call_inputs_path(inputs),
    ]
    existing = [path for path in paths if path.exists()]
    if existing and inputs.duplicate_policy == "fail":
        raise FileExistsError("Existing proposal outputs found: " + ", ".join(str(path) for path in existing))
    if inputs.duplicate_policy == "overwrite_generated":
        for path in existing:
            path.unlink()


def rows_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows if row.get(key)}


def rows_by_any_key(rows: list[dict[str, Any]], keys: list[str]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in keys:
            value = row.get(key)
            if value:
                indexed.setdefault(str(value), row)
    return indexed


def route_for_target(decision: dict[str, Any], target_task: str) -> dict[str, Any] | None:
    for route in decision.get("task_routes") or []:
        if route.get("target_task") == target_task:
            return route
    return None


def make_backpointer_refs(decision: dict[str, Any], text_unit: dict[str, Any] | None) -> list[dict[str, Any]]:
    refs: list[dict[str, Any]] = []
    for source in (decision, text_unit or {}):
        raw = source.get("raw_backpointer")
        if isinstance(raw, dict) and raw:
            refs.append(raw)
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ref in refs:
        key = json.dumps(ref, ensure_ascii=False, sort_keys=True)
        if key not in seen:
            unique.append(ref)
            seen.add(key)
    return unique


def evidence_neighbor_maps(evidence_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    previous_by_ref: dict[str, dict[str, Any]] = {}
    next_by_ref: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(evidence_rows):
        evidence_ref = row.get("evidence_ref") or row.get("canonical_evidence_ref")
        if not evidence_ref:
            continue
        if index > 0:
            previous_by_ref[str(evidence_ref)] = evidence_rows[index - 1]
        if index + 1 < len(evidence_rows):
            next_by_ref[str(evidence_ref)] = evidence_rows[index + 1]
    return previous_by_ref, next_by_ref


def evidence_ref_for_decision(decision: dict[str, Any], evidence_rows_by_key: dict[str, dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    for key in ("evidence_ref", "span_id", "raw_span_id"):
        value = decision.get(key)
        if value and str(value) in evidence_rows_by_key:
            row = evidence_rows_by_key[str(value)]
            return str(row.get("evidence_ref") or row.get("canonical_evidence_ref") or value), row
    return None, None


def row_reference_values(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for key in (
        "evidence_ref",
        "canonical_evidence_ref",
        "source_specific_ref",
        "text_unit_id",
        "raw_span_id",
        "candidate_id",
    ):
        value = row.get(key)
        if value:
            values.add(str(value))
    for key in ("evidence_refs", "backpointer_refs", "raw_backpointer_refs", "source_specific_refs"):
        value = row.get(key)
        if isinstance(value, list):
            values.update(str(item) for item in value if item)
    return values


def rows_by_reference_values(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        for value in row_reference_values(row):
            indexed.setdefault(value, row)
    return indexed


def source_span_aliases(value: Any) -> set[str]:
    text = str(value or "")
    if not text:
        return set()
    aliases = {text}
    for match in re.findall(r"text_span:\d+", text):
        aliases.add(match)
    for match in re.findall(r"span-(\d+)", text):
        aliases.add(f"text_span:{match}")
    return aliases


def packet_exact_reference_values(packet: dict[str, Any]) -> set[str]:
    refs = {str(ref) for ref in packet.get("evidence_refs") or [] if ref}
    if packet.get("text_unit_id"):
        refs.add(str(packet["text_unit_id"]))
    return refs


def packet_context_reference_values(packet: dict[str, Any]) -> set[str]:
    refs: set[str] = set()
    for key in ("raw_span_id", "parent_text_unit_id"):
        if packet.get(key):
            refs.update(source_span_aliases(packet[key]))
    for ref in packet.get("context_refs") or []:
        refs.update(source_span_aliases(ref))
    return refs - packet_exact_reference_values(packet)


def slim_deterministic_memory_candidate(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "candidate_id",
        "candidate_text",
        "candidate_type",
        "memory_class",
        "evidence_quote",
        "evidence_refs",
        "backpointer_refs",
        "source_specific_refs",
        "generation_method",
        "inference_level",
        "confidence",
        "review_status",
    ]
    return {key: row.get(key) for key in keys if key in row}


def slim_preprocessing_decision(row: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "decision_id",
        "route",
        "memory_writer_action",
        "reason",
        "evidence_ref",
        "source_specific_ref",
        "speaker",
        "target_participant",
        "subject_contamination_risk",
    ]
    return {key: row.get(key) for key in keys if key in row}


def annotate_s1_processing(
    packet: dict[str, Any],
    memory_candidate_by_ref: dict[str, dict[str, Any]],
    preprocessing_decision_by_ref: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    exact_refs = packet_exact_reference_values(packet)
    context_refs = packet_context_reference_values(packet)
    memory_candidate = next((memory_candidate_by_ref[ref] for ref in exact_refs if ref in memory_candidate_by_ref), None)
    context_memory_candidate = next((memory_candidate_by_ref[ref] for ref in context_refs if ref in memory_candidate_by_ref), None)
    preprocessing_decision = next((preprocessing_decision_by_ref[ref] for ref in exact_refs if ref in preprocessing_decision_by_ref), None)
    context_preprocessing_decision = next((preprocessing_decision_by_ref[ref] for ref in context_refs if ref in preprocessing_decision_by_ref), None)
    deterministic_processing: dict[str, Any] = {}
    if memory_candidate:
        deterministic_processing["memory_candidate"] = slim_deterministic_memory_candidate(memory_candidate)
    elif context_memory_candidate:
        deterministic_processing["context_memory_candidate"] = slim_deterministic_memory_candidate(context_memory_candidate)
    if preprocessing_decision:
        deterministic_processing["preprocessing_decision"] = slim_preprocessing_decision(preprocessing_decision)
    elif context_preprocessing_decision:
        deterministic_processing["context_preprocessing_decision"] = slim_preprocessing_decision(context_preprocessing_decision)
    if memory_candidate or preprocessing_decision:
        status = "succeeded"
    elif deterministic_processing:
        status = "insufficient"
    else:
        status = "not_attempted"
    route = str(packet.get("route_recommended") or "")
    packet["deterministic_processing_status"] = status
    packet["deterministic_s1_processing"] = deterministic_processing
    packet["needs_llm_assist"] = route in {"weak_llm_proposal", "strong_llm_proposal", "split_or_segment_first", "human_review"}
    return packet


def packet_from_text_unit(
    inputs: RunnerInputs,
    decision: dict[str, Any],
    route: dict[str, Any],
    text_unit_id: str,
    text_unit: dict[str, Any],
    by_text_unit: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    parent = by_text_unit.get(str(text_unit.get("parent_text_unit_id") or ""))
    previous = by_text_unit.get(str(text_unit.get("previous_text_unit_id") or ""))
    next_item = by_text_unit.get(str(text_unit.get("next_text_unit_id") or ""))
    evidence_ref = decision.get("evidence_ref") or text_unit.get("evidence_ref") or text_unit.get("canonical_evidence_ref")
    context_refs = [
        ref
        for ref in (
            text_unit.get("parent_text_unit_id"),
            text_unit.get("previous_text_unit_id"),
            text_unit.get("next_text_unit_id"),
        )
        if ref
    ]
    return {
        "proposal_input_id": f"{inputs.profile.data['proposal_input_id_prefix']}:{short_hash(inputs.proposal_run_id + ':' + str(text_unit_id) + ':' + str(route.get('route_decision_id')))}",
        "target_task": inputs.profile.target_task,
        "text_unit_id": text_unit_id,
        "raw_span_id": text_unit.get("raw_span_id") or decision.get("raw_span_id"),
        "parent_text_unit_id": text_unit.get("parent_text_unit_id"),
        "previous_text_unit_id": text_unit.get("previous_text_unit_id"),
        "next_text_unit_id": text_unit.get("next_text_unit_id"),
        "route_decision_id": route.get("route_decision_id"),
        "route_run_id": decision.get("route_run_id"),
        "unit_type": text_unit.get("unit_type"),
        "text": text_unit.get("text") or "",
        "parent_paragraph_text": (parent or {}).get("text") or "",
        "previous_sentence_text": (previous or {}).get("text") or "",
        "next_sentence_text": (next_item or {}).get("text") or "",
        "section_type": text_unit.get("section_type") or decision.get("section_type") or "unknown",
        "perspective": text_unit.get("perspective") or decision.get("perspective") or "unknown",
        "speaker": text_unit.get("speaker") or decision.get("speaker"),
        "subject_role": text_unit.get("subject_role") or decision.get("subject_role"),
        "target_participant": text_unit.get("target_participant") or decision.get("target_participant"),
        "target_subject_ids": text_unit.get("target_subject_ids") or decision.get("target_subject_ids") or [],
        "subject_ids": text_unit.get("subject_ids") or decision.get("subject_ids") or [],
        "subject_contamination_risk": text_unit.get("subject_contamination_risk") or decision.get("subject_contamination_risk"),
        "s1_storage_policy": text_unit.get("s1_storage_policy") or "unknown",
        "retrieval_policy": text_unit.get("retrieval_policy") or decision.get("retrieval_policy") or "unknown",
        "s2_policy": text_unit.get("s2_policy") or decision.get("s2_policy") or "unknown",
        "route_recommended": route.get("recommended_route"),
        "route_score": route.get("route_score"),
        "route_score_max": route.get("route_score_max"),
        "route_reasons": route.get("routing_reasons") or [],
        "raw_backpointer_refs": make_backpointer_refs(decision, text_unit),
        "evidence_refs": [evidence_ref] if evidence_ref else [],
        "context_refs": context_refs,
        "warnings": sorted(set((decision.get("warnings") or []) + (text_unit.get("warnings") or []) + (route.get("warnings") or []))),
        "max_quote_chars": inputs.profile.max_quote_chars,
    }


def packet_from_evidence(
    inputs: RunnerInputs,
    decision: dict[str, Any],
    route: dict[str, Any],
    evidence_ref: str,
    evidence: dict[str, Any],
    previous: dict[str, Any] | None,
    next_item: dict[str, Any] | None,
) -> dict[str, Any]:
    locator = evidence.get("locator") or {}
    raw_backpointer = decision.get("raw_backpointer") or {
        "source_file": locator.get("source_file"),
        "locator": locator,
    }
    evidence_like = {
        "raw_backpointer": raw_backpointer,
    }
    metadata = evidence.get("metadata") or {}
    decision_perspective = decision.get("perspective")
    perspective = (
        decision_perspective
        if decision_perspective not in {None, "", "unknown"}
        else metadata.get("perspective") or evidence.get("subject_role") or "unknown"
    )
    context_refs = [ref for ref in ((previous or {}).get("evidence_ref"), (next_item or {}).get("evidence_ref")) if ref]
    return {
        "proposal_input_id": f"{inputs.profile.data['proposal_input_id_prefix']}:{short_hash(inputs.proposal_run_id + ':' + evidence_ref + ':' + str(route.get('route_decision_id')))}",
        "target_task": inputs.profile.target_task,
        "text_unit_id": None,
        "raw_span_id": decision.get("raw_span_id"),
        "parent_text_unit_id": None,
        "previous_text_unit_id": (previous or {}).get("evidence_ref"),
        "next_text_unit_id": (next_item or {}).get("evidence_ref"),
        "route_decision_id": route.get("route_decision_id"),
        "route_run_id": decision.get("route_run_id"),
        "unit_type": locator.get("kind") or evidence.get("source_type") or "evidence_item",
        "text": evidence.get("text") or "",
        "parent_paragraph_text": "",
        "previous_sentence_text": (previous or {}).get("text") or "",
        "next_sentence_text": (next_item or {}).get("text") or "",
        "section_type": evidence.get("section_type") or decision.get("section_type") or "conversation",
        "perspective": perspective,
        "speaker": evidence.get("speaker") or locator.get("speaker"),
        "subject_role": evidence.get("subject_role"),
        "target_participant": evidence.get("target_participant") or metadata.get("modeled_subject_id"),
        "target_subject_ids": evidence.get("target_subject_ids") or ([metadata.get("modeled_subject_id")] if metadata.get("modeled_subject_id") else []),
        "subject_ids": evidence.get("subject_ids") or [],
        "subject_contamination_risk": evidence.get("subject_contamination_risk"),
        "s1_storage_policy": "ordinary_evidence",
        "retrieval_policy": decision.get("retrieval_policy") or evidence.get("retrieval_policy") or "default_retrieval",
        "s2_policy": decision.get("s2_policy") or evidence.get("s2_policy") or "candidate_allowed",
        "route_recommended": route.get("recommended_route"),
        "route_score": route.get("route_score"),
        "route_score_max": route.get("route_score_max"),
        "route_reasons": route.get("routing_reasons") or [],
        "raw_backpointer_refs": make_backpointer_refs(decision, evidence_like),
        "evidence_refs": [evidence_ref],
        "context_refs": context_refs,
        "warnings": sorted(set((decision.get("warnings") or []) + (evidence.get("warnings") or []) + (route.get("warnings") or []))),
        "max_quote_chars": inputs.profile.max_quote_chars,
    }


def packet_from_memory_route_decision(
    inputs: RunnerInputs,
    decision: dict[str, Any],
    route: dict[str, Any],
) -> dict[str, Any]:
    evidence_refs = decision.get("evidence_refs") or []
    if decision.get("evidence_ref") and decision.get("evidence_ref") not in evidence_refs:
        evidence_refs = [decision.get("evidence_ref")] + evidence_refs
    return {
        "proposal_input_id": f"{inputs.profile.data['proposal_input_id_prefix']}:{short_hash(inputs.proposal_run_id + ':' + str(decision.get('memory_id') or decision.get('span_id')) + ':' + str(route.get('route_decision_id')))}",
        "target_task": inputs.profile.target_task,
        "memory_id": decision.get("memory_id"),
        "text_unit_id": decision.get("text_unit_id"),
        "raw_span_id": decision.get("raw_span_id"),
        "parent_text_unit_id": decision.get("parent_text_unit_id"),
        "previous_text_unit_id": decision.get("previous_text_unit_id"),
        "next_text_unit_id": decision.get("next_text_unit_id"),
        "route_decision_id": route.get("route_decision_id"),
        "route_run_id": decision.get("route_run_id"),
        "unit_type": decision.get("unit_type") or "s1_memory_unit",
        "text": decision.get("processed_text") or decision.get("text") or decision.get("original_text") or "",
        "original_text": decision.get("original_text") or decision.get("text") or "",
        "processed_text": decision.get("processed_text") or decision.get("text") or "",
        "parent_paragraph_text": "",
        "previous_sentence_text": "",
        "next_sentence_text": "",
        "section_type": decision.get("section_type") or "body",
        "perspective": decision.get("perspective") or "target",
        "speaker": decision.get("speaker"),
        "subject_role": decision.get("subject_role") or "target",
        "target_participant": decision.get("target_participant"),
        "target_subject_ids": decision.get("target_subject_ids") or [],
        "subject_ids": decision.get("subject_ids") or [],
        "subject_contamination_risk": decision.get("subject_contamination_risk"),
        "s1_storage_policy": "ordinary_evidence",
        "retrieval_policy": decision.get("retrieval_policy") or "default_retrieval",
        "s2_policy": decision.get("s2_policy") or "candidate_allowed",
        "memory_class": decision.get("memory_class"),
        "memory_type": decision.get("memory_type"),
        "processing_method": decision.get("processing_method"),
        "source_layer": decision.get("source_layer"),
        "route_recommended": route.get("recommended_route"),
        "route_score": route.get("route_score"),
        "route_score_max": route.get("route_score_max"),
        "route_reasons": route.get("routing_reasons") or [],
        "raw_backpointer_refs": decision.get("raw_backpointer_refs") or make_backpointer_refs(decision, None),
        "evidence_refs": evidence_refs,
        "context_refs": [],
        "warnings": sorted(
            set(
                inherited_warnings_for_s2_memory_packet(decision.get("warnings") or [])
                + (route.get("warnings") or [])
            )
        ),
        "max_quote_chars": inputs.profile.max_quote_chars,
    }


def build_input_packets(inputs: RunnerInputs) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    decisions = read_jsonl(inputs.route_decisions_path)
    text_units = read_jsonl(inputs.text_units_path)
    evidence_rows = read_jsonl(inputs.evidence_path)
    memory_candidates = read_jsonl(inputs.memory_candidates_path)
    preprocessing_decisions = read_jsonl(inputs.preprocessing_decisions_path)
    by_text_unit = rows_by_key(text_units, "text_unit_id")
    by_evidence = rows_by_any_key(evidence_rows, ["evidence_ref", "canonical_evidence_ref", "raw_span_id", "source_specific_ref"])
    memory_candidate_by_ref = rows_by_reference_values(memory_candidates)
    preprocessing_decision_by_ref = rows_by_reference_values(preprocessing_decisions)
    previous_by_evidence_ref, next_by_evidence_ref = evidence_neighbor_maps(evidence_rows)
    packets: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    candidate_index = 0
    for decision in decisions:
        route = route_for_target(decision, inputs.profile.target_task)
        if not route:
            continue
        if decision.get("source_layer") == "s1_memory_unit" and decision.get("memory_id"):
            packet = packet_from_memory_route_decision(inputs, decision, route)
        else:
            text_unit_id = decision.get("text_unit_id")
            text_unit = by_text_unit.get(str(text_unit_id)) if text_unit_id else None
            if not text_unit and decision.get("span_id") in by_text_unit:
                text_unit_id = decision.get("span_id")
                text_unit = by_text_unit[str(text_unit_id)]
            if not text_unit and decision.get("raw_span_id") in by_text_unit:
                text_unit_id = decision.get("raw_span_id")
                text_unit = by_text_unit[str(text_unit_id)]
            if text_unit:
                packet = packet_from_text_unit(inputs, decision, route, str(text_unit_id), text_unit, by_text_unit)
            else:
                evidence_ref, evidence = evidence_ref_for_decision(decision, by_evidence)
                if not evidence_ref or not evidence:
                    warnings.append(
                        {
                            "warning": "input_source_missing_for_route_decision",
                            "span_id": decision.get("span_id"),
                            "evidence_ref": decision.get("evidence_ref"),
                            "route_decision_id": route.get("route_decision_id"),
                        }
                    )
                    continue
                packet = packet_from_evidence(
                    inputs,
                    decision,
                    route,
                    evidence_ref,
                    evidence,
                    previous_by_evidence_ref.get(evidence_ref),
                    next_by_evidence_ref.get(evidence_ref),
                )
        packet = annotate_s1_processing(packet, memory_candidate_by_ref, preprocessing_decision_by_ref)
        include_packet = candidate_index >= inputs.item_offset and (
            (candidate_index - inputs.item_offset) % inputs.sample_stride == 0
        )
        candidate_index += 1
        if not include_packet:
            continue
        packets.append(packet)
        if inputs.max_items is not None and len(packets) >= inputs.max_items:
            break
    return packets, warnings


def validate_v02_model_payload(profile: ProposalProfile, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "output_kind",
        "fact_candidate_text",
        "hypothesis_text",
        "source_text_quote",
        "source_text_quotes",
        "supporting_observations",
        "alternative_explanations",
        "uncertainty_notes",
        "candidate_type",
        "inference_level",
        "claim_strength",
        "proposal_confidence",
        "hypothesis_status",
        "hypothesis_confidence",
        "hypothesis_scope",
        "commitment_level",
        "promotion_readiness",
        "subject_contamination_risk",
        "privacy_class",
        "warnings",
    ):
        if key not in payload:
            errors.append(f"missing_{key}")
    output_kind = str(payload.get("output_kind") or "")
    if output_kind not in profile.values("output_kinds"):
        errors.append("invalid_output_kind")
    if payload.get("candidate_type") not in profile.values("candidate_types"):
        errors.append("invalid_candidate_type")
    if payload.get("inference_level") not in profile.values("inference_levels"):
        errors.append("invalid_inference_level")
    if payload.get("claim_strength") not in profile.values("claim_strengths"):
        errors.append("invalid_claim_strength")
    if payload.get("proposal_confidence") not in profile.values("proposal_confidences"):
        errors.append("invalid_proposal_confidence")
    if payload.get("hypothesis_status") not in profile.values("hypothesis_statuses"):
        errors.append("invalid_hypothesis_status")
    if payload.get("hypothesis_confidence") not in profile.values("hypothesis_confidences"):
        errors.append("invalid_hypothesis_confidence")
    if payload.get("hypothesis_scope") not in profile.values("hypothesis_scopes"):
        errors.append("invalid_hypothesis_scope")
    if payload.get("commitment_level") not in profile.values("commitment_levels"):
        errors.append("invalid_commitment_level")
    if payload.get("promotion_readiness") not in profile.values("promotion_readinesses"):
        errors.append("invalid_promotion_readiness")
    if payload.get("subject_contamination_risk") not in profile.values("subject_contamination_risks"):
        errors.append("invalid_subject_contamination_risk")
    if payload.get("privacy_class") not in profile.values("privacy_classes"):
        errors.append("invalid_privacy_class")
    for list_key in ("source_text_quotes", "supporting_observations", "alternative_explanations", "uncertainty_notes", "warnings"):
        if not isinstance(payload.get(list_key), list):
            errors.append(f"{list_key}_must_be_list")
    fact_text = str(payload.get("fact_candidate_text") or "").strip()
    hypothesis_text = str(payload.get("hypothesis_text") or "").strip()
    quote = str(payload.get("source_text_quote") or "").strip()
    has_support = bool(quote or payload.get("source_text_quotes") or payload.get("supporting_observations"))
    if output_kind == "portrait_fact_candidate":
        if not fact_text:
            errors.append("fact_candidate_missing_text")
        if hypothesis_text:
            errors.append("fact_candidate_must_not_use_hypothesis_text")
        if not quote:
            errors.append("fact_candidate_missing_source_text_quote")
    elif output_kind == "portrait_hypothesis_candidate":
        if not hypothesis_text:
            errors.append("hypothesis_candidate_missing_text")
        if fact_text:
            errors.append("hypothesis_candidate_must_not_use_fact_text")
        if not has_support:
            errors.append("hypothesis_candidate_missing_support")
    elif output_kind in {"reject", "model_uncertain", "needs_human_review"}:
        if fact_text or hypothesis_text:
            errors.append("non_candidate_must_not_use_candidate_text")
        if payload.get("candidate_type") != "none":
            errors.append("non_candidate_must_use_candidate_type_none")
    return errors


def validate_s1_model_payload(profile: ProposalProfile, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "output_kind",
        "memory_candidate_text",
        "memory_class",
        "source_text_quote",
        "source_text_quotes",
        "supporting_observations",
        "scope_hint",
        "temporal_hint",
        "compression_level",
        "inference_level",
        "proposal_confidence",
        "uncertainty_notes",
        "warnings",
    ):
        if key not in payload:
            errors.append(f"missing_{key}")
    output_kind = str(payload.get("output_kind") or "")
    if output_kind not in profile.values("output_kinds"):
        errors.append("invalid_output_kind")
    if payload.get("memory_class") not in profile.values("memory_classes"):
        errors.append("invalid_memory_class")
    if payload.get("scope_hint") not in profile.values("scope_hints"):
        errors.append("invalid_scope_hint")
    if payload.get("temporal_hint") not in profile.values("temporal_hints"):
        errors.append("invalid_temporal_hint")
    if payload.get("compression_level") not in profile.values("compression_levels"):
        errors.append("invalid_compression_level")
    if payload.get("inference_level") not in profile.values("inference_levels"):
        errors.append("invalid_inference_level")
    if payload.get("proposal_confidence") not in profile.values("proposal_confidences"):
        errors.append("invalid_proposal_confidence")
    for list_key in ("source_text_quotes", "supporting_observations", "uncertainty_notes", "warnings"):
        if not isinstance(payload.get(list_key), list):
            errors.append(f"{list_key}_must_be_list")
    candidate_text = str(payload.get("memory_candidate_text") or "").strip()
    if output_kind == "memory_candidate":
        if not candidate_text:
            errors.append("memory_candidate_missing_text")
        if payload.get("memory_class") == "unknown":
            errors.append("memory_candidate_must_have_memory_class")
    elif output_kind in {"reject", "model_uncertain", "needs_human_review"}:
        if candidate_text:
            errors.append("non_candidate_must_not_use_memory_candidate_text")
        if payload.get("memory_class") != "unknown":
            errors.append("non_candidate_must_use_memory_class_unknown")
    return errors


def validate_v01_model_payload(profile: ProposalProfile, payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for key in (
        "proposal_status",
        "candidate_text",
        "candidate_type",
        "inference_level",
        "claim_strength",
        "proposal_confidence",
        "source_text_quote",
        "subject_contamination_risk",
        "privacy_class",
        "warnings",
    ):
        if key not in payload:
            errors.append(f"missing_{key}")
    if payload.get("proposal_status") not in profile.values("model_statuses"):
        errors.append("invalid_proposal_status")
    if payload.get("candidate_type") not in profile.values("candidate_types"):
        errors.append("invalid_candidate_type")
    if payload.get("inference_level") not in profile.values("inference_levels"):
        errors.append("invalid_inference_level")
    if payload.get("claim_strength") not in profile.values("claim_strengths"):
        errors.append("invalid_claim_strength")
    if payload.get("proposal_confidence") not in profile.values("proposal_confidences"):
        errors.append("invalid_proposal_confidence")
    if payload.get("subject_contamination_risk") not in profile.values("subject_contamination_risks"):
        errors.append("invalid_subject_contamination_risk")
    if payload.get("privacy_class") not in profile.values("privacy_classes"):
        errors.append("invalid_privacy_class")
    if not isinstance(payload.get("warnings"), list):
        errors.append("warnings_must_be_list")
    if payload.get("proposal_status") == "candidate" and not str(payload.get("candidate_text") or "").strip():
        errors.append("candidate_missing_text")
    if payload.get("proposal_status") != "candidate" and payload.get("candidate_type") != "none":
        errors.append("non_candidate_must_use_candidate_type_none")
    return errors


def validate_model_payload(profile: ProposalProfile, payload: dict[str, Any]) -> list[str]:
    if profile_is_s1_memory_candidate(profile):
        return validate_s1_model_payload(profile, payload)
    if profile_uses_output_kind(profile):
        return validate_v02_model_payload(profile, payload)
    return validate_v01_model_payload(profile, payload)


def parse_model_output(profile: ProposalProfile, result: ProviderResult) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        payload = json.loads(result.output_text)
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    if not isinstance(payload, dict):
        return None, ["schema_validation_failed", "model_output_not_object"]
    errors = validate_model_payload(profile, payload)
    if errors:
        return None, ["schema_validation_failed", *errors]
    return payload, []


def base_proposal_row(
    inputs: RunnerInputs,
    packet: dict[str, Any],
    route_used: str,
    model_id: str,
    provider: str,
    prompt: PromptPolicy | None,
) -> dict[str, Any]:
    source_text = str(packet.get("text") or "")
    original_text = str(packet.get("original_text") or source_text)
    processed_text = str(packet.get("processed_text") or "")
    return {
        "schema_version": inputs.profile.data["proposal_schema_version"],
        "proposal_profile_id": inputs.profile.profile_id,
        "proposal_profile_hash": inputs.profile.profile_hash,
        "proposal_id": f"{inputs.profile.data['proposal_id_prefix']}:{short_hash(inputs.proposal_run_id + ':' + packet['proposal_input_id'] + ':' + route_used)}",
        "proposal_input_id": packet["proposal_input_id"],
        "proposal_run_id": inputs.proposal_run_id,
        "target_task": inputs.profile.target_task,
        "text_unit_id": packet.get("text_unit_id"),
        "raw_span_id": packet.get("raw_span_id"),
        "parent_text_unit_id": packet.get("parent_text_unit_id"),
        "previous_text_unit_id": packet.get("previous_text_unit_id"),
        "next_text_unit_id": packet.get("next_text_unit_id"),
        "route_decision_id": packet.get("route_decision_id"),
        "route_run_id": packet.get("route_run_id"),
        "route_recommended": packet.get("route_recommended"),
        "route_used": route_used,
        "override_reason": packet.get("route_override_reason"),
        "model_id": model_id,
        "provider": provider,
        "prompt_policy_id": prompt.policy_id if prompt else None,
        "prompt_hash": prompt.prompt_hash if prompt else None,
        "output_kind": "skipped",
        "epistemic_status": OUTPUT_KIND_TO_EPISTEMIC_STATUS["skipped"],
        "proposal_status": "skipped",
        "fact_candidate_text": "",
        "hypothesis_text": "",
        "candidate_text": "",
        "candidate_type": "none",
        "candidate_type_hints": [],
        "inference_level": "weak_inference",
        "claim_strength": "unknown",
        "proposal_confidence": "low",
        "source_text": source_text,
        "original_text": original_text,
        "original_text_excerpt": source_text_preview(original_text),
        "processed_text": processed_text,
        "source_text_quote": "",
        "source_text_preview": source_text_preview(source_text),
        "source_text_quotes": [],
        "supporting_observations": [],
        "alternative_explanations": [],
        "uncertainty_notes": [],
        "max_quote_chars": inputs.profile.max_quote_chars,
        "evidence_refs": packet.get("evidence_refs") or [],
        "raw_backpointer_refs": packet.get("raw_backpointer_refs") or [],
        "context_refs": packet.get("context_refs") or [],
        "deterministic_processing_status": packet.get("deterministic_processing_status") or "not_attempted",
        "deterministic_s1_processing": packet.get("deterministic_s1_processing") or {},
        "needs_llm_assist": bool(packet.get("needs_llm_assist")),
        "llm_assist_used": False,
        "memory_candidate_text": "",
        "memory_class": "unknown",
        "llm_assisted_s1_candidate": {},
        "scope_hint": "unknown",
        "temporal_hint": "unknown",
        "compression_level": "unknown",
        "section_type": packet.get("section_type"),
        "perspective": packet.get("perspective"),
        "hypothesis_status": "unknown",
        "hypothesis_confidence": "unknown",
        "hypothesis_scope": "unknown",
        "commitment_level": "unknown",
        "promotion_readiness": "unknown",
        "subject_contamination_risk": "unknown",
        "privacy_class": "unknown",
        "review_status": "needs_review",
        "write_permission": False,
        "estimated_input_tokens": 0,
        "estimated_output_tokens": 0,
        "cache_hit": None,
        "latency_ms": None,
        "warnings": list(packet.get("warnings") or []),
    }


def skipped_row(inputs: RunnerInputs, packet: dict[str, Any], status_warning: str) -> dict[str, Any]:
    row = base_proposal_row(inputs, packet, str(packet.get("route_recommended") or "skipped"), "", "none", None)
    row["output_kind"] = "skipped"
    row["epistemic_status"] = OUTPUT_KIND_TO_EPISTEMIC_STATUS["skipped"]
    row["proposal_status"] = "skipped"
    row["warnings"] = sorted(set(row["warnings"] + [status_warning]))
    return row


def deterministic_s1_memory_candidate(packet: dict[str, Any]) -> dict[str, Any] | None:
    deterministic = packet.get("deterministic_s1_processing") or {}
    if not isinstance(deterministic, dict):
        return None
    candidate = deterministic.get("memory_candidate")
    return candidate if isinstance(candidate, dict) else None


def deterministic_candidate_is_usable(candidate: dict[str, Any]) -> bool:
    return bool(str(candidate.get("candidate_text") or "").strip()) and str(candidate.get("memory_class") or "unknown") != "unknown"


def s1_script_only_memory_candidate_row(inputs: RunnerInputs, packet: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    row = base_proposal_row(inputs, packet, "script_only", "", "none", None)
    memory_candidate_text = str(candidate.get("candidate_text") or "")
    memory_class = str(candidate.get("memory_class") or "unknown")
    source_text_quote = str(candidate.get("evidence_quote") or packet.get("text") or "")[: inputs.profile.max_quote_chars]
    supporting_observations = []
    for key in ("evidence_summary", "reason_for_candidate"):
        if candidate.get(key):
            supporting_observations.append(str(candidate[key]))
    row.update(
        {
            "output_kind": "memory_candidate",
            "epistemic_status": OUTPUT_KIND_TO_EPISTEMIC_STATUS["memory_candidate"],
            "proposal_status": "memory_candidate",
            "memory_candidate_text": memory_candidate_text,
            "processed_text": memory_candidate_text,
            "memory_class": memory_class,
            "candidate_text": memory_candidate_text,
            "source_text_quote": source_text_quote,
            "source_text_quotes": [source_text_quote] if source_text_quote else [],
            "supporting_observations": supporting_observations,
            "scope_hint": "event" if memory_class == "episodic" else "unknown",
            "temporal_hint": "event_bound" if memory_class == "episodic" else "unknown",
            "compression_level": "light",
            "inference_level": candidate.get("inference_level") or "explicit",
            "proposal_confidence": candidate.get("confidence") if candidate.get("confidence") in inputs.profile.values("proposal_confidences") else "unknown",
            "review_status": "needs_review",
            "llm_assist_used": False,
            "llm_assisted_s1_candidate": {},
            "warnings": sorted(set(row["warnings"] + ["script_only_deterministic_s1_candidate_proposal"])),
        }
    )
    row["warnings"] = sorted(
        set(
            row["warnings"]
            + exact_quote_warning_for_packet(source_text_quote, packet, inputs.profile.max_quote_chars)
        )
    )
    return row


def human_review_queue_row(inputs: RunnerInputs, packet: dict[str, Any]) -> dict[str, Any]:
    source_text = str(packet.get("text") or "")
    original_text = str(packet.get("original_text") or source_text)
    return {
        "schema_version": inputs.profile.data["human_review_queue_schema_version"],
        "proposal_profile_id": inputs.profile.profile_id,
        "proposal_input_id": packet["proposal_input_id"],
        "text_unit_id": packet.get("text_unit_id"),
        "route_decision_id": packet.get("route_decision_id"),
        "route_run_id": packet.get("route_run_id"),
        "route_recommended": "human_review",
        "route_reasons": packet.get("route_reasons") or [],
        "source_text": source_text,
        "original_text": original_text,
        "original_text_excerpt": source_text_preview(original_text),
        "memory_candidate_text": "",
        "deterministic_processing_status": packet.get("deterministic_processing_status") or "not_attempted",
        "deterministic_s1_processing": packet.get("deterministic_s1_processing") or {},
        "needs_llm_assist": bool(packet.get("needs_llm_assist")),
        "llm_assist_used": False,
        "parent_paragraph_text": packet.get("parent_paragraph_text") or "",
        "context_refs": packet.get("context_refs") or [],
        "raw_backpointer_refs": packet.get("raw_backpointer_refs") or [],
        "evidence_refs": packet.get("evidence_refs") or [],
        "warnings": packet.get("warnings") or [],
        "review_status": "needs_review",
    }


def proposal_from_model_payload(
    inputs: RunnerInputs,
    packet: dict[str, Any],
    route_used: str,
    result: ProviderResult,
    prompt: PromptPolicy,
    payload: dict[str, Any],
) -> dict[str, Any]:
    row = base_proposal_row(inputs, packet, route_used, result.model_id, result.provider, prompt)
    if profile_is_s1_memory_candidate(inputs.profile):
        output_kind = str(payload["output_kind"])
        source_text_quote = str(payload.get("source_text_quote") or "")[: inputs.profile.max_quote_chars]
        is_candidate_output = output_kind == "memory_candidate"
        memory_candidate_text = str(payload.get("memory_candidate_text") or "") if is_candidate_output else ""
        llm_candidate = {
            "memory_candidate_text": memory_candidate_text,
            "memory_class": payload.get("memory_class"),
            "source_text_quote": source_text_quote,
            "source_text_quotes": payload.get("source_text_quotes") or [],
            "supporting_observations": payload.get("supporting_observations") or [],
            "scope_hint": payload.get("scope_hint"),
            "temporal_hint": payload.get("temporal_hint"),
            "compression_level": payload.get("compression_level"),
            "inference_level": payload.get("inference_level"),
            "proposal_confidence": payload.get("proposal_confidence"),
            "uncertainty_notes": payload.get("uncertainty_notes") or [],
        } if is_candidate_output else {}
        row.update(
            {
                "output_kind": output_kind,
                "epistemic_status": OUTPUT_KIND_TO_EPISTEMIC_STATUS.get(output_kind, "failure"),
                "proposal_status": output_kind,
                "memory_candidate_text": memory_candidate_text,
                "processed_text": memory_candidate_text,
                "memory_class": payload.get("memory_class") if is_candidate_output else "unknown",
                "llm_assisted_s1_candidate": llm_candidate,
                "llm_assist_used": True,
                "candidate_text": memory_candidate_text,
                "source_text_quote": source_text_quote,
                "source_text_quotes": payload.get("source_text_quotes") or [],
                "supporting_observations": payload.get("supporting_observations") or [],
                "scope_hint": payload.get("scope_hint"),
                "temporal_hint": payload.get("temporal_hint"),
                "compression_level": payload.get("compression_level"),
                "inference_level": payload.get("inference_level"),
                "proposal_confidence": payload.get("proposal_confidence"),
                "uncertainty_notes": payload.get("uncertainty_notes") or [],
                "review_status": "needs_review" if is_candidate_output else "not_required",
                "estimated_input_tokens": result.estimated_input_tokens,
                "estimated_output_tokens": result.estimated_output_tokens,
                "cache_hit": result.cache_hit,
                "latency_ms": result.latency_ms,
                "warnings": sorted(set(row["warnings"] + list(payload.get("warnings") or []))),
            }
        )
        quote_warnings = exact_quote_warning(
            source_text_quote,
            packet_primary_text(packet),
            local_context_text(packet),
            inputs.profile.max_quote_chars,
        )
        row["warnings"] = sorted(set(row["warnings"] + quote_warnings))
        return normalize_non_candidate_row(row)

    if profile_uses_output_kind(inputs.profile):
        output_kind = str(payload["output_kind"])
        epistemic_status = OUTPUT_KIND_TO_EPISTEMIC_STATUS.get(output_kind, "failure")
        fact_text = str(payload.get("fact_candidate_text") or "")
        hypothesis_text = str(payload.get("hypothesis_text") or "")
        candidate_text = fact_text if output_kind == "portrait_fact_candidate" else hypothesis_text if output_kind == "portrait_hypothesis_candidate" else ""
        is_candidate_output = output_kind in {"portrait_fact_candidate", "portrait_hypothesis_candidate"}
        source_text_quote = str(payload.get("source_text_quote") or "")[: inputs.profile.max_quote_chars]
        quote_backfill_warnings: list[str] = []
        if output_kind == "portrait_hypothesis_candidate":
            source_text_quote, quote_backfill_warnings = choose_traceable_quote(
                payload,
                packet,
                inputs.profile.max_quote_chars,
            )
        row.update(
            {
                "output_kind": output_kind,
                "epistemic_status": epistemic_status,
                "proposal_status": output_kind,
                "fact_candidate_text": fact_text,
                "hypothesis_text": hypothesis_text,
                "candidate_text": candidate_text,
                "candidate_type": payload.get("candidate_type") if is_candidate_output else "none",
                "candidate_type_hints": payload.get("candidate_type_hints") or [],
                "inference_level": payload.get("inference_level"),
                "claim_strength": payload.get("claim_strength"),
                "proposal_confidence": payload.get("proposal_confidence"),
                "source_text_quote": source_text_quote,
                "source_text_quotes": payload.get("source_text_quotes") or [],
                "supporting_observations": payload.get("supporting_observations") or [],
                "alternative_explanations": payload.get("alternative_explanations") or [],
                "uncertainty_notes": payload.get("uncertainty_notes") or [],
                "hypothesis_status": payload.get("hypothesis_status"),
                "hypothesis_confidence": payload.get("hypothesis_confidence"),
                "hypothesis_scope": payload.get("hypothesis_scope"),
                "commitment_level": payload.get("commitment_level"),
                "promotion_readiness": payload.get("promotion_readiness"),
                "subject_contamination_risk": payload.get("subject_contamination_risk"),
                "privacy_class": payload.get("privacy_class"),
                "review_status": "unreviewed_low_commitment" if output_kind == "portrait_hypothesis_candidate" else "needs_review",
                "estimated_input_tokens": result.estimated_input_tokens,
                "estimated_output_tokens": result.estimated_output_tokens,
                "cache_hit": result.cache_hit,
                "latency_ms": result.latency_ms,
                "warnings": sorted(set(row["warnings"] + list(payload.get("warnings") or []) + quote_backfill_warnings)),
            }
        )
        quote_warnings = exact_quote_warning(
            source_text_quote,
            packet_primary_text(packet),
            local_context_text(packet),
            inputs.profile.max_quote_chars,
        )
        if output_kind == "portrait_hypothesis_candidate" and quote_uses_local_context(
            source_text_quote,
            packet,
        ):
            quote_warnings.append("hypothesis_uses_neighbor_context")
        row["warnings"] = sorted(set(row["warnings"] + quote_warnings))
        if output_kind == "portrait_fact_candidate" and any(
            warning in quote_warnings
            for warning in {
                "source_text_quote_too_long",
                "source_text_quote_not_exact",
                "source_text_quote_not_in_primary_text",
                "hypothesis_uses_neighbor_context",
            }
        ):
            row["output_kind"] = "needs_human_review"
            row["epistemic_status"] = OUTPUT_KIND_TO_EPISTEMIC_STATUS["needs_human_review"]
            row["proposal_status"] = "needs_human_review"
            row["review_status"] = "needs_review"
            row["warnings"] = sorted(set(row["warnings"] + ["fact_candidate_downgraded_quote_guardrail"]))
        return normalize_non_candidate_row(row)

    status = clean_model_status(str(payload["proposal_status"]))
    row.update(
        {
            "output_kind": status,
            "epistemic_status": status,
            "proposal_status": status,
            "candidate_text": str(payload.get("candidate_text") or ""),
            "candidate_type": payload.get("candidate_type") if status == "candidate" else "none",
            "inference_level": payload.get("inference_level"),
            "claim_strength": payload.get("claim_strength"),
            "proposal_confidence": payload.get("proposal_confidence"),
            "source_text_quote": str(payload.get("source_text_quote") or "")[: inputs.profile.max_quote_chars],
            "subject_contamination_risk": payload.get("subject_contamination_risk"),
            "privacy_class": payload.get("privacy_class"),
            "estimated_input_tokens": result.estimated_input_tokens,
            "estimated_output_tokens": result.estimated_output_tokens,
            "cache_hit": result.cache_hit,
            "latency_ms": result.latency_ms,
            "warnings": sorted(set(row["warnings"] + list(payload.get("warnings") or []) + (["model_uncertain"] if str(payload.get("proposal_status")) == "model_uncertain" else []))),
        }
    )
    quote_warnings = exact_quote_warning(
        str(payload.get("source_text_quote") or ""),
        packet_primary_text(packet),
        local_context_text(packet),
        inputs.profile.max_quote_chars,
    )
    row["warnings"] = sorted(set(row["warnings"] + quote_warnings))
    if status == "candidate" and any(
        warning in quote_warnings
        for warning in {
            "source_text_quote_too_long",
            "source_text_quote_not_exact",
            "source_text_quote_not_in_primary_text",
        }
    ):
        row["proposal_status"] = "human_review_required"
        row["candidate_text"] = ""
        row["candidate_type"] = "none"
        row["warnings"] = sorted(set(row["warnings"] + ["candidate_downgraded_quote_guardrail"]))
    if status == "human_review_required":
        row["candidate_text"] = ""
        row["candidate_type"] = "none"
    return normalize_non_candidate_row(row)


def failure_row(
    inputs: RunnerInputs,
    packet: dict[str, Any],
    route_used: str,
    result: ProviderResult | None,
    prompt: PromptPolicy | None,
    warnings: list[str],
    raw_output: str = "",
) -> dict[str, Any]:
    row = base_proposal_row(
        inputs,
        packet,
        route_used,
        result.model_id if result else "",
        result.provider if result else inputs.provider,
        prompt,
    )
    row["proposal_status"] = "model_failure"
    row["output_kind"] = "model_failure"
    row["epistemic_status"] = OUTPUT_KIND_TO_EPISTEMIC_STATUS["model_failure"]
    row["warnings"] = sorted(set(row["warnings"] + warnings))
    row = normalize_non_candidate_row(row)
    if result:
        row["estimated_input_tokens"] = result.estimated_input_tokens
        row["estimated_output_tokens"] = result.estimated_output_tokens
        row["cache_hit"] = result.cache_hit
        row["latency_ms"] = result.latency_ms
    return {
        **row,
        "raw_model_output": raw_output[:2000],
    }


def model_call_input_row(inputs: RunnerInputs, packet: dict[str, Any], route_used: str, prompt: PromptPolicy, model_id: str) -> dict[str, Any]:
    return {
        "schema_version": "proposal_model_call_input.v0.1",
        "proposal_run_id": inputs.proposal_run_id,
        "proposal_profile_id": inputs.profile.profile_id,
        "proposal_input_id": packet["proposal_input_id"],
        "route_recommended": packet.get("route_recommended"),
        "route_used": route_used,
        "model_id": model_id,
        "prompt_policy_id": prompt.policy_id,
        "prompt_hash": prompt.prompt_hash,
        "prompt_text": prompt.text,
        "input_packet": packet,
        "expected_external_output": {
            "proposal_input_id": packet["proposal_input_id"],
            "provider": "subagent",
            "model_id": model_id,
            "output_text": "{...strict JSON model output...}",
        },
    }


def redacted_failure_warnings(exc: Exception) -> list[str]:
    if isinstance(exc, RuntimeError):
        return [f"model_call_failed:{type(exc).__name__}", "provider_error_redacted"]
    return [f"model_call_failed:{type(exc).__name__}", "provider_error_redacted"]


def process_packet(
    inputs: RunnerInputs,
    provider: ModelProvider,
    packet: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    proposals: list[dict[str, Any]] = []
    human_queue: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    model_call_inputs: list[dict[str, Any]] = []

    route = str(packet.get("route_recommended") or "")
    if route == "script_only":
        deterministic_candidate = deterministic_s1_memory_candidate(packet) if profile_is_s1_memory_candidate(inputs.profile) else None
        if deterministic_candidate and deterministic_candidate_is_usable(deterministic_candidate):
            proposals.append(s1_script_only_memory_candidate_row(inputs, packet, deterministic_candidate))
            return proposals, human_queue, failures, model_call_inputs
        if should_upgrade_script_only_to_weak(inputs, packet):
            packet = {
                **packet,
                "route_override_reason": "script_only_upgraded_to_weak_for_target_signal",
                "warnings": sorted(set((packet.get("warnings") or []) + ["script_only_upgraded_to_weak_for_target_signal"])),
            }
            route = "weak_llm_proposal"
        else:
            if profile_is_s1_memory_candidate(inputs.profile):
                warning = (
                    "script_only_deterministic_s1_candidate_not_exact"
                    if packet.get("deterministic_processing_status") == "insufficient"
                    else "script_only_no_deterministic_s1_candidate"
                )
            else:
                warning = "script_skipped_or_no_candidate"
            proposals.append(skipped_row(inputs, packet, warning))
            return proposals, human_queue, failures, model_call_inputs
    if route == "skip_or_background_only":
        proposals.append(skipped_row(inputs, packet, "skip_or_background_only_no_candidate"))
        return proposals, human_queue, failures, model_call_inputs
    if route == "split_or_segment_first":
        proposals.append(skipped_row(inputs, packet, "preprocessing_required_no_candidate"))
        return proposals, human_queue, failures, model_call_inputs
    if route == "human_review":
        human_queue.append(human_review_queue_row(inputs, packet))
        return proposals, human_queue, failures, model_call_inputs
    if route not in {"weak_llm_proposal", "strong_llm_proposal"}:
        proposals.append(skipped_row(inputs, packet, "unsupported_route_no_candidate"))
        return proposals, human_queue, failures, model_call_inputs

    prompt = inputs.weak_prompt if route == "weak_llm_proposal" else inputs.strong_prompt
    model_id = inputs.weak_model if route == "weak_llm_proposal" else inputs.strong_model
    model_call_inputs.append(model_call_input_row(inputs, packet, route, prompt, model_id))
    try:
        result = provider.generate(prompt=prompt, model_id=model_id, input_packet=packet)
    except Exception as exc:
        failures.append(failure_row(inputs, packet, route, None, prompt, redacted_failure_warnings(exc)))
        return proposals, human_queue, failures, model_call_inputs
    payload, errors = parse_model_output(inputs.profile, result)
    if errors or payload is None:
        failures.append(failure_row(inputs, packet, route, result, prompt, errors, result.output_text))
        return proposals, human_queue, failures, model_call_inputs
    proposals.append(proposal_from_model_payload(inputs, packet, route, result, prompt, payload))
    return proposals, human_queue, failures, model_call_inputs


def process_packets(inputs: RunnerInputs, packets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    provider = build_provider(inputs.provider, inputs.api_mode, inputs.external_model_outputs_path)
    proposals: list[dict[str, Any]] = []
    human_queue: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    model_call_inputs: list[dict[str, Any]] = []

    concurrency = max(1, int(inputs.provider_concurrency or 1))
    if concurrency == 1 or len(packets) <= 1:
        ordered_results = [process_packet(inputs, provider, packet) for packet in packets]
    else:
        ordered_results: list[tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None] = [
            None
        ] * len(packets)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(process_packet, inputs, provider, packet): idx
                for idx, packet in enumerate(packets)
            }
            for future in as_completed(futures):
                ordered_results[futures[future]] = future.result()
        if any(result is None for result in ordered_results):
            raise RuntimeError("parallel provider processing produced missing packet results")

    for result in ordered_results:
        if result is None:
            continue
        packet_proposals, packet_human_queue, packet_failures, packet_model_call_inputs = result
        proposals.extend(packet_proposals)
        human_queue.extend(packet_human_queue)
        failures.extend(packet_failures)
        model_call_inputs.extend(packet_model_call_inputs)
    return proposals, human_queue, failures, model_call_inputs


def validate_outputs(inputs: RunnerInputs, proposals: list[dict[str, Any]], human_queue: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    all_rows = proposals + failures
    for row in all_rows:
        if row.get("write_permission") is not False:
            raise ValueError(f"Proposal row must keep write_permission=false: {row.get('proposal_id')}")
        if row.get("proposal_status") not in inputs.profile.values("proposal_statuses"):
            raise ValueError(f"Invalid proposal_status: {row.get('proposal_status')}")
        if profile_is_s1_memory_candidate(inputs.profile):
            output_kind = str(row.get("output_kind") or "")
            if output_kind not in inputs.profile.values("output_kinds"):
                raise ValueError(f"Invalid output_kind: {output_kind}")
            expected_epistemic = OUTPUT_KIND_TO_EPISTEMIC_STATUS.get(output_kind)
            if row.get("epistemic_status") != expected_epistemic:
                raise ValueError(f"Invalid epistemic_status for {output_kind}: {row.get('epistemic_status')}")
            if not (row.get("original_text") or row.get("original_text_excerpt")):
                raise ValueError("S1 proposal rows must preserve original_text or original_text_excerpt.")
            if row.get("deterministic_processing_status") not in inputs.profile.values("deterministic_processing_statuses"):
                raise ValueError(f"Invalid deterministic_processing_status: {row.get('deterministic_processing_status')}")
            if output_kind == "memory_candidate":
                if not str(row.get("memory_candidate_text") or "").strip():
                    raise ValueError("memory_candidate rows must carry memory_candidate_text.")
                if not str(row.get("processed_text") or "").strip():
                    raise ValueError("memory_candidate rows must carry processed_text.")
                if row.get("memory_class") not in inputs.profile.values("memory_classes") or row.get("memory_class") == "unknown":
                    raise ValueError("memory_candidate rows must carry concrete memory_class.")
            else:
                if row.get("memory_candidate_text") or row.get("candidate_text"):
                    raise ValueError("Non-candidate S1 rows must not carry memory candidate text.")
                if row.get("processed_text"):
                    raise ValueError("Non-candidate S1 rows must not carry processed_text.")
                if row.get("memory_class") != "unknown":
                    raise ValueError("Non-candidate S1 rows must use memory_class=unknown.")
        elif profile_uses_output_kind(inputs.profile):
            output_kind = str(row.get("output_kind") or "")
            if output_kind not in inputs.profile.values("output_kinds"):
                raise ValueError(f"Invalid output_kind: {output_kind}")
            expected_epistemic = OUTPUT_KIND_TO_EPISTEMIC_STATUS.get(output_kind)
            if row.get("epistemic_status") != expected_epistemic:
                raise ValueError(f"Invalid epistemic_status for {output_kind}: {row.get('epistemic_status')}")
            if output_kind not in {"portrait_fact_candidate", "portrait_hypothesis_candidate"} and row.get("candidate_type") != "none":
                raise ValueError("Non-candidate rows must use candidate_type=none.")
            if output_kind not in {"portrait_fact_candidate", "portrait_hypothesis_candidate"} and (
                row.get("fact_candidate_text") or row.get("hypothesis_text") or row.get("candidate_text")
            ):
                raise ValueError("Non-candidate rows must not carry candidate text fields.")
        elif row.get("proposal_status") in {"reject", "skipped", "model_failure", "human_review_required"} and row.get("candidate_type") != "none":
            raise ValueError("Non-candidate rows must use candidate_type=none.")
        if row.get("source_text_quote") and len(str(row["source_text_quote"])) > inputs.profile.max_quote_chars:
            raise ValueError("source_text_quote exceeds max_quote_chars")
    for item in human_queue:
        if item.get("review_status") != "needs_review":
            raise ValueError("Human review queue rows must default to needs_review")


def render_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# Proposal Run Report",
        "",
        f"- proposal_run_id: `{manifest['proposal_run_id']}`",
        f"- proposal_profile_id: `{manifest['proposal_profile_id']}`",
        f"- workspace: `{manifest['workspace']}`",
        f"- provider: `{manifest['provider']}`",
        f"- target_task: `{manifest['target_task']}`",
        "",
        "## Counts",
        "",
    ]
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- No durable memory writes executed.",
            "- No reviewed/canonical memory writes executed.",
            "- No current_portrait writes executed.",
            "- No graph truth writes executed.",
            "- No automatic acceptance executed.",
            "- All proposal rows keep `write_permission=false`.",
            "- Profiles may explicitly allow target-subject `script_only` rows with clear modeling signals to upgrade to weak LLM; scripts still do not generate candidates directly.",
            "- `human_review` routes are queued without model calls by default.",
            "- Retrieval/routing/model confidence is not support checking.",
        ]
    )
    if manifest.get("target_task") == "s2_portrait_candidate":
        lines.extend(
            [
                "- S2 `portrait_fact_candidate` rows are proposal-layer subject claim/fact candidates, not reviewed portrait facts and not current portrait truth.",
                "- S1-inherited quote warnings are labeled separately from quote warnings produced by the current S2 proposal.",
            ]
        )
    return "\n".join(lines) + "\n"


def build_manifest(
    inputs: RunnerInputs,
    packets: list[dict[str, Any]],
    packet_warnings: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    human_queue: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    model_call_inputs: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for row in proposals + failures:
        status_counts[str(row.get("proposal_status"))] = status_counts.get(str(row.get("proposal_status")), 0) + 1
        route_counts[str(row.get("route_used"))] = route_counts.get(str(row.get("route_used")), 0) + 1
    model_call_rows = [row for row in proposals + failures if row.get("provider") not in {"none", ""} and row.get("model_id")]
    return {
        "schema_version": inputs.profile.data["run_schema_version"],
        "proposal_run_id": inputs.proposal_run_id,
        "proposal_profile_id": inputs.profile.profile_id,
        "proposal_profile_path": str(inputs.profile.path),
        "proposal_profile_hash": inputs.profile.profile_hash,
        "workspace": str(inputs.workspace),
        "output_dir": str(inputs.output_dir),
        "input_route_decisions": str(inputs.route_decisions_path),
        "input_route_decisions_hash": optional_file_hash(inputs.route_decisions_path),
        "text_units_path": str(inputs.text_units_path),
        "text_units_hash": optional_file_hash(inputs.text_units_path),
        "evidence_path": str(inputs.evidence_path),
        "evidence_hash": optional_file_hash(inputs.evidence_path),
        "section_map_path": str(inputs.section_map_path),
        "section_map_hash": optional_file_hash(inputs.section_map_path),
        "memory_candidates_path": str(inputs.memory_candidates_path),
        "memory_candidates_hash": optional_file_hash(inputs.memory_candidates_path),
        "preprocessing_decisions_path": str(inputs.preprocessing_decisions_path),
        "preprocessing_decisions_hash": optional_file_hash(inputs.preprocessing_decisions_path),
        "model_call_inputs_path": str(model_call_inputs_path(inputs)),
        "model_call_inputs_hash": optional_file_hash(model_call_inputs_path(inputs)),
        "external_model_outputs_path": str(inputs.external_model_outputs_path) if inputs.external_model_outputs_path else None,
        "external_model_outputs_hash": optional_file_hash(inputs.external_model_outputs_path) if inputs.external_model_outputs_path else None,
        "target_task": inputs.profile.target_task,
        "provider": inputs.provider,
        "api_mode": inputs.api_mode,
        "live_api_enabled": inputs.live_api_enabled,
        "live_api_unlock_source": inputs.live_api_unlock_source,
        "weak_model_id": inputs.weak_model,
        "strong_model_id": inputs.strong_model,
        "provider_concurrency": inputs.provider_concurrency,
        "weak_prompt_policy_id": inputs.weak_prompt.policy_id,
        "strong_prompt_policy_id": inputs.strong_prompt.policy_id,
        "weak_prompt_hash": inputs.weak_prompt.prompt_hash,
        "strong_prompt_hash": inputs.strong_prompt.prompt_hash,
        "api_key_recorded": False,
        "durable_writes_executed": False,
        "reviewed_units_written": False,
        "current_portrait_written": False,
        "graph_truth_written": False,
        "automatic_acceptance_executed": False,
        "script_only_generates_candidates": bool(inputs.profile.data["script_only_generates_candidates"]),
        "script_only_may_upgrade_to_weak": bool(inputs.profile.data.get("script_only_may_upgrade_to_weak", False)),
        "human_review_calls_model_by_default": bool(inputs.profile.data["human_review_calls_model_by_default"]),
        "strict_schema_validation": True,
        "structured_output_enforced": False,
        "max_quote_chars": inputs.profile.max_quote_chars,
        "item_selection": {
            "item_offset": inputs.item_offset,
            "sample_stride": inputs.sample_stride,
            "max_items": inputs.max_items,
        },
        "created_at": now_iso(),
        "counts": {
            "proposal_inputs": len(packets),
            "input_packet_warnings": len(packet_warnings),
            "proposal_rows": len(proposals),
            "human_review_queue_rows": len(human_queue),
            "model_output_failures": len(failures),
            "model_call_rows": len(model_call_rows),
            "model_call_inputs": len(model_call_inputs),
            "estimated_input_tokens": sum(int(row.get("estimated_input_tokens") or 0) for row in model_call_rows),
            "estimated_output_tokens": sum(int(row.get("estimated_output_tokens") or 0) for row in model_call_rows),
            "status_counts": status_counts,
            "route_used_counts": route_counts,
        },
    }


def run_proposal_runner(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_inputs(args)
    prepare_outputs(inputs)
    packets, packet_warnings = build_input_packets(inputs)
    proposals, human_queue, failures, model_call_inputs = process_packets(inputs, packets)
    validate_outputs(inputs, proposals, human_queue, failures)

    write_jsonl(output_path(inputs, "proposals"), proposals)
    write_jsonl(output_path(inputs, "human_review_queue"), human_queue)
    write_jsonl(output_path(inputs, "model_output_failures"), failures)
    write_jsonl(model_call_inputs_path(inputs), model_call_inputs)
    empty_review_log(output_path(inputs, "review_log"))
    manifest = build_manifest(inputs, packets, packet_warnings, proposals, human_queue, failures, model_call_inputs)
    write_json(output_path(inputs, "manifest"), manifest)
    write_text(output_path(inputs, "report"), render_report(manifest))
    return {
        "workspace": str(inputs.workspace),
        "output_dir": str(inputs.output_dir),
        "proposal_run_id": inputs.proposal_run_id,
        "proposal_profile_id": inputs.profile.profile_id,
        "outputs": {
            "proposals": str(output_path(inputs, "proposals")),
            "human_review_queue": str(output_path(inputs, "human_review_queue")),
            "model_output_failures": str(output_path(inputs, "model_output_failures")),
            "model_call_inputs": str(model_call_inputs_path(inputs)),
            "proposal_review_log": str(output_path(inputs, "review_log")),
            "proposal_run_manifest": str(output_path(inputs, "manifest")),
            "proposal_run_report": str(output_path(inputs, "report")),
        },
        "counts": manifest["counts"],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reviewable AI-assisted proposals from router decisions.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--route-decisions", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--duplicate-policy", default="fail", choices=sorted(SUPPORTED_DUPLICATE_POLICIES))
    parser.add_argument("--provider", default=None, choices=sorted(SUPPORTED_PROVIDERS))
    parser.add_argument("--api-mode", default=None, choices=sorted(SUPPORTED_API_MODES))
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--weak-model", default=None)
    parser.add_argument("--strong-model", default=None)
    parser.add_argument("--weak-prompt", default=None)
    parser.add_argument("--strong-prompt", default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--item-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    parser.add_argument("--external-model-outputs", default=None)
    parser.add_argument(
        "--provider-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent provider calls. Default 1 preserves serial behavior.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_proposal_runner(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
