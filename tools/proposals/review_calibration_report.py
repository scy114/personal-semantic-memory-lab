"""Dry-run review/calibration report generator for S2 proposal outputs.

This tool is intentionally read-only with respect to memory assets. It reads a
proposal run directory and writes review/calibration artifacts beside it:
- review_calibration_report.md
- optionally review_calibration_items.jsonl

It does not call models, accept candidates, or write durable memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.proposals.claim_attribution_router import annotate_claim_attribution


DEFAULT_SAMPLING_POLICY_ID = "review_calibration_sampling.v0.1"
DEFAULT_SAMPLING_SEED = "review-calibration-v0.2"
DEFAULT_SAMPLE_RATE = 0.1
DEFAULT_MIN_SAMPLE_COUNT = 3
OUTPUT_KINDS = [
    "portrait_fact_candidate",
    "portrait_hypothesis_candidate",
    "reject",
    "model_uncertain",
    "needs_human_review",
    "skipped",
    "model_failure",
]
REVIEW_POOLS = [
    "fact_promotion_review_queue",
    "hypothesis_pool_candidates",
    "contamination_risk_pool",
    "uncertain_pool",
    "future_relation_candidate",
    "no_promotion",
    "sampled_reject_skip_pool",
    "provider_failure_pool",
    "schema_unknown_pool",
]
CLAIM_ATTRIBUTION_SCHEMA_VERSION = "claim_attribution_annotation.v0.2.1"
SKIP_AUDIT_SIGNAL_VERSION = "skip_audit_signals.heuristic.v0.1"
CONTAMINATION_WARNINGS = {
    "dialogue_subject_contamination_risk",
    "other_participant_dominant",
    "lacks_target_self_report_signal",
    "source_text_quote_not_in_primary_text",
    "source_text_quote_not_exact",
    "source_text_quote_too_long",
    "hypothesis_uses_neighbor_context",
    "fact_candidate_downgraded_quote_guardrail",
    "candidate_downgraded_quote_guardrail",
}
LOW_VALUE_PATTERNS = [
    r"\bhey\b.*\bwhat'?s up\b",
    r"\bgood to see you\b",
    r"\bthanks?\b",
    r"\bokay\b",
    r"\byeah\b[!.]?$",
    r"\bwow\b[!.]?$",
    r"\b哈哈\b|\b好的\b|\b谢谢\b",
]
SELF_MODELING_PATTERNS = [
    r"\bi\s+(also\s+)?(prefer|like|love|enjoy|need|want|use|lost|have|am|can't|cannot)\b",
    r"\bi'?m\s+",
    r"\bmy\s+",
    r"\bme too\b",
    r"\bsame here\b",
    r"\bgo-to\b",
    r"\bspeaks to me\b",
    r"\b我(也|喜欢|想|需要|用|不能|正在|计划)\b",
    r"\b我的\b",
]
RELATION_NODE_PATTERNS = [
    r"\byou\b|\byour\b",
    r"\blet'?s\b|\bwe should\b|\bwanna\b|\bcan't wait\b|\bin my corner\b",
    r"\bstore\b|\bstudio\b|\bproject\b|\bfestival\b|\bcompetition\b|\bdance class\b|\bmarley\b",
    r"\b一起\b|\b你\b|\b项目\b|\b工具\b",
]
AMBIGUOUS_CONTEXT_PATTERNS = [
    r"\bthat\b|\bit\b|\bthey\b|\bthem\b|\bthis\b",
    r"\bnext (fri|friday|week|month)\b",
    r"\byour\b|\byou\b",
]
PORTRAIT_RELEVANT_TYPES = {
    "preference",
    "goal",
    "constraint",
    "user_state",
    "procedural",
    "relationship_context",
    "project_context",
    "uncertainty",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


def ensure_within(path: Path, roots: list[Path]) -> Path:
    resolved = path.resolve()
    for root in roots:
        try:
            resolved.relative_to(root.resolve())
            return resolved
        except ValueError:
            continue
    raise ValueError(f"Unsafe path outside allowed roots: {resolved}")


def row_output_kind(row: dict[str, Any]) -> str:
    if row.get("output_kind"):
        return str(row["output_kind"])
    status = str(row.get("proposal_status") or "")
    if status == "candidate":
        return "portrait_fact_candidate"
    if status in {"human_review_required", "needs_human_review"}:
        return "needs_human_review"
    if status in {"reject", "skipped", "model_failure", "model_uncertain"}:
        return status
    return status or "unknown"


def has_hypothesis_support(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("source_text_quote") or "").strip()
        or row.get("source_text_quotes")
        or row.get("supporting_observations")
    )


def warning_set(row: dict[str, Any]) -> set[str]:
    return {str(item) for item in row.get("warnings") or []}


def has_contamination_signal(row: dict[str, Any]) -> bool:
    warnings = warning_set(row)
    return (
        str(row.get("subject_contamination_risk") or "") == "high"
        or bool(warnings & CONTAMINATION_WARNINGS)
    )


def has_clean_provenance(row: dict[str, Any]) -> bool:
    return bool(
        str(row.get("source_text_quote") or "").strip()
        and row.get("evidence_refs")
        and row.get("raw_backpointer_refs")
        and not has_contamination_signal(row)
    )


def has_primary_source_quote(row: dict[str, Any]) -> bool:
    warnings = warning_set(row)
    return bool(
        str(row.get("source_text_quote") or "").strip()
        and not (
            warnings
            & {
                "source_text_quote_not_in_primary_text",
                "source_text_quote_not_exact",
                "source_text_quote_too_long",
                "hypothesis_uses_neighbor_context",
            }
        )
    )


def is_portrait_relevant(row: dict[str, Any]) -> bool:
    candidate_type = str(row.get("candidate_type") or "")
    return candidate_type in PORTRAIT_RELEVANT_TYPES


def text_for_signal(row: dict[str, Any]) -> str:
    return " ".join(
        str(
            row.get("source_text_preview")
            or row.get("source_text_quote")
            or row.get("fact_candidate_text")
            or row.get("hypothesis_text")
            or row.get("candidate_text")
            or ""
        ).split()
    )


def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def has_non_target_signal(row: dict[str, Any]) -> bool:
    warnings = warning_set(row)
    perspective = str(row.get("perspective") or "").lower()
    return "non_target_subject_skipped_for_s2_portrait" in warnings or perspective in {"other_participant", "other_person"}


def skip_audit_signals(row: dict[str, Any]) -> dict[str, Any]:
    """Soft heuristic signals for skipped/rejected rows.

    These are review/calibration hints only. They are not truth labels, not
    promotion authority, and not graph proof.
    """

    output_kind = row_output_kind(row)
    text = text_for_signal(row)
    has_self = matches_any(text, SELF_MODELING_PATTERNS)
    has_relation = matches_any(text, RELATION_NODE_PATTERNS)
    is_low_value = matches_any(text, LOW_VALUE_PATTERNS) and not has_self
    non_target = has_non_target_signal(row)
    ambiguous_context = matches_any(text, AMBIGUOUS_CONTEXT_PATTERNS)
    tags: list[str] = []

    if output_kind not in {"skipped", "reject"}:
        return {
            "schema_version": SKIP_AUDIT_SIGNAL_VERSION,
            "modeling_value_hint": "unknown",
            "attribution_risk_hint": "unknown",
            "promotion_risk_hint": "unknown",
            "evidence_directness_hint": "unknown",
            "claim_target_clarity_hint": "unknown",
            "context_requirement_hint": "unknown",
            "review_priority_hint": "unknown",
            "signal_tags": [],
            "signal_source": "heuristic_skip_audit.v0.1",
            "signal_confidence": "low",
        }

    if has_self:
        tags.append("self_modeling_signal")
    if has_relation:
        tags.append("relation_or_node_signal")
    if non_target:
        tags.append("non_target_source")
    if ambiguous_context:
        tags.append("context_dependent_reference")
    if is_low_value:
        tags.append("low_content_signal")
    if output_kind == "reject":
        tags.append("model_reject_row")

    if is_low_value:
        modeling_value = "none"
        evidence_directness = "ambient_context"
        claim_target = "unknown"
        attribution_risk = "low" if not non_target else "medium"
    elif has_self and not non_target:
        modeling_value = "high"
        evidence_directness = "direct_self_report"
        claim_target = "target_subject"
        attribution_risk = "low"
    elif has_relation:
        modeling_value = "medium"
        evidence_directness = "inferred_relation"
        claim_target = "relationship" if re.search(r"\blet'?s\b|\bwe should\b|\bwanna\b", text, re.IGNORECASE) else "other_person"
        attribution_risk = "high" if non_target else "medium"
    else:
        modeling_value = "unknown" if text else "none"
        evidence_directness = "unknown"
        claim_target = "unknown"
        attribution_risk = "high" if non_target else "unknown"

    if is_low_value:
        context_requirement = "local_turn_enough"
    elif ambiguous_context and modeling_value in {"medium", "high"}:
        context_requirement = "neighbor_turn_needed"
    elif has_self and not ambiguous_context:
        context_requirement = "local_turn_enough"
    elif has_relation:
        context_requirement = "neighbor_turn_needed"
    else:
        context_requirement = "unknown"

    if attribution_risk == "high" or (non_target and modeling_value in {"medium", "high"}):
        promotion_risk = "high"
    elif modeling_value in {"medium", "high"}:
        promotion_risk = "medium"
    elif modeling_value == "none":
        promotion_risk = "low"
    else:
        promotion_risk = "unknown"

    if modeling_value == "high" or promotion_risk == "high" or context_requirement in {"paragraph_needed", "section_needed", "document_needed", "cross_document_needed"}:
        review_priority = "focused_sample"
    elif modeling_value == "medium" or context_requirement == "neighbor_turn_needed":
        review_priority = "sample"
    elif modeling_value == "none":
        review_priority = "none"
    else:
        review_priority = "sample"

    confidence = "medium" if text and tags else "low"
    return {
        "schema_version": SKIP_AUDIT_SIGNAL_VERSION,
        "modeling_value_hint": modeling_value,
        "attribution_risk_hint": attribution_risk,
        "promotion_risk_hint": promotion_risk,
        "evidence_directness_hint": evidence_directness,
        "claim_target_clarity_hint": claim_target,
        "context_requirement_hint": context_requirement,
        "review_priority_hint": review_priority,
        "signal_tags": sorted(set(tags)),
        "signal_source": "heuristic_skip_audit.v0.1",
        "signal_confidence": confidence,
    }


def deterministic_sample(rows: list[dict[str, Any]], *, sample_rate: float, min_count: int, seed: str) -> list[dict[str, Any]]:
    if not rows:
        return []
    target = max(min_count, int(round(len(rows) * sample_rate)))
    target = min(len(rows), target)
    keyed = sorted(rows, key=lambda row: str(row.get("proposal_id") or row.get("proposal_input_id") or json.dumps(row, sort_keys=True)))
    rng = random.Random(seed)
    selected_indexes = sorted(rng.sample(range(len(keyed)), target))
    return [keyed[index] for index in selected_indexes]


@dataclass
class SamplingPolicy:
    sampling_policy_id: str = DEFAULT_SAMPLING_POLICY_ID
    sampling_seed: str = DEFAULT_SAMPLING_SEED
    sample_rate: float = DEFAULT_SAMPLE_RATE
    min_sample_count: int = DEFAULT_MIN_SAMPLE_COUNT


def review_item(
    *,
    row: dict[str, Any],
    pool: str,
    priority: str,
    action: str,
    sample_reason: str,
    policy: SamplingPolicy,
) -> dict[str, Any]:
    output_kind = row_output_kind(row)
    epistemic_status = str(row.get("epistemic_status") or output_kind)
    proposal_id = str(row.get("proposal_id") or row.get("proposal_input_id") or "")
    annotation = row.get("claim_attribution_annotation") or annotate_claim_attribution(row)
    skip_signals = row.get("skip_audit_signals") or skip_audit_signals(row)
    return {
        "schema_version": "review_calibration_item.v0.2",
        "review_item_id": f"rci:{short_hash(proposal_id + ':' + pool + ':' + sample_reason)}",
        "proposal_id": proposal_id,
        "proposal_run_id": row.get("proposal_run_id"),
        "proposal_profile_id": row.get("proposal_profile_id"),
        "output_kind": output_kind,
        "epistemic_status": epistemic_status,
        "review_pool": pool,
        "review_priority": priority,
        "default_review_action": action,
        "calibration_action": "none",
        "review_sample_reason": sample_reason,
        "sampling_policy_id": policy.sampling_policy_id,
        "sampling_seed": policy.sampling_seed,
        "sample_rate": policy.sample_rate,
        "sample_reason": sample_reason,
        "requires_manual_review_now": priority == "high",
        "write_permission": False,
        "evidence_refs": row.get("evidence_refs") or [],
        "raw_backpointer_refs": row.get("raw_backpointer_refs") or [],
        "warnings": row.get("warnings") or [],
        "source_text": row.get("source_text") or row.get("source_text_preview") or "",
        "source_text_preview": row.get("source_text_preview") or "",
        "source_text_quote": row.get("source_text_quote") or "",
        "fact_candidate_text": row.get("fact_candidate_text") or "",
        "hypothesis_text": row.get("hypothesis_text") or "",
        "candidate_text": row.get("candidate_text") or "",
        "claim_attribution_annotation": annotation,
        "claim_type_hint": annotation.get("claim_type_hint"),
        "source_perspective": annotation.get("source_perspective"),
        "claim_target_hint": annotation.get("claim_target_hint"),
        "claim_target_node_type": annotation.get("claim_target_node_type"),
        "attribution_status_hint": annotation.get("attribution_status_hint"),
        "promotion_path": annotation.get("promotion_path"),
        "fact_promotion_eligibility": annotation.get("fact_promotion_eligibility"),
        "skip_audit_signals": skip_signals,
        "modeling_value_hint": skip_signals.get("modeling_value_hint"),
        "attribution_risk_hint": skip_signals.get("attribution_risk_hint"),
        "promotion_risk_hint": skip_signals.get("promotion_risk_hint"),
        "evidence_directness_hint": skip_signals.get("evidence_directness_hint"),
        "claim_target_clarity_hint": skip_signals.get("claim_target_clarity_hint"),
        "context_requirement_hint": skip_signals.get("context_requirement_hint"),
        "review_priority_hint": skip_signals.get("review_priority_hint"),
        "signal_tags": skip_signals.get("signal_tags") or [],
        "signal_source": skip_signals.get("signal_source"),
        "signal_confidence": skip_signals.get("signal_confidence"),
    }


def classify_review_items(rows: list[dict[str, Any]], failures: list[dict[str, Any]], policy: SamplingPolicy) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    reject_skip_rows: list[dict[str, Any]] = []

    for row in rows + failures:
        annotation = annotate_claim_attribution(row)
        signals = skip_audit_signals(row)
        row = {**row, "claim_attribution_annotation": annotation, "skip_audit_signals": signals}
        output_kind = row_output_kind(row)
        if output_kind == "model_failure":
            items.append(
                review_item(
                    row=row,
                    pool="provider_failure_pool",
                    priority="low",
                    action="archive_provider_failure",
                    sample_reason="provider_failure",
                    policy=policy,
                )
            )
            continue
        if has_contamination_signal(row):
            items.append(
                review_item(
                    row=row,
                    pool="contamination_risk_pool",
                    priority="high",
                    action="mark_subject_contamination",
                    sample_reason="contamination_signal",
                    policy=policy,
                )
            )
            continue
        if output_kind == "portrait_fact_candidate":
            if annotation.get("fact_promotion_eligibility") == "eligible":
                clean_and_relevant = has_clean_provenance(row) and has_primary_source_quote(row) and is_portrait_relevant(row)
                items.append(
                    review_item(
                        row=row,
                        pool="fact_promotion_review_queue",
                        priority="high" if clean_and_relevant else "medium",
                        action="accept_for_promotion_review" if clean_and_relevant else "needs_more_evidence",
                        sample_reason="fact_candidate_attribution_eligible",
                        policy=policy,
                    )
                )
                continue
            if annotation.get("fact_promotion_eligibility") == "ineligible":
                items.append(
                    review_item(
                        row=row,
                        pool=str(annotation.get("promotion_path") or "hypothesis_pool_candidates"),
                        priority="low",
                        action="keep_low_commitment",
                        sample_reason="fact_candidate_blocked_from_promotion_by_attribution",
                        policy=policy,
                    )
                )
                continue
            items.append(
                review_item(
                    row=row,
                    pool="uncertain_pool",
                    priority="high",
                    action="needs_more_evidence",
                    sample_reason="fact_candidate_attribution_uncertain",
                    policy=policy,
                )
            )
            continue
        if output_kind == "portrait_hypothesis_candidate":
            if str(row.get("hypothesis_text") or "").strip() and has_hypothesis_support(row):
                items.append(
                    review_item(
                        row=row,
                        pool="hypothesis_pool_candidates",
                        priority="low",
                        action="keep_low_commitment",
                        sample_reason="usable_low_commitment_hypothesis",
                        policy=policy,
                    )
                )
            else:
                items.append(
                    review_item(
                        row=row,
                        pool="uncertain_pool",
                        priority="sample_only",
                        action="needs_more_evidence",
                        sample_reason="hypothesis_missing_text_or_support",
                        policy=policy,
                    )
                )
            continue
        if output_kind in {"model_uncertain", "needs_human_review"}:
            items.append(
                review_item(
                    row=row,
                    pool="uncertain_pool",
                    priority="high" if output_kind == "needs_human_review" else "sample_only",
                    action="needs_more_evidence",
                    sample_reason=output_kind,
                    policy=policy,
                )
            )
            continue
        if output_kind in {"reject", "skipped"}:
            reject_skip_rows.append(row)
            continue
        items.append(
            review_item(
                row=row,
                pool="schema_unknown_pool",
                priority="high",
                action="needs_more_evidence",
                sample_reason="unknown_output_kind",
                policy=policy,
            )
        )

    for row in deterministic_sample(
        reject_skip_rows,
        sample_rate=policy.sample_rate,
        min_count=policy.min_sample_count,
        seed=policy.sampling_seed + ":reject_skip",
    ):
        items.append(
            review_item(
                row=row,
                pool="sampled_reject_skip_pool",
                priority="sample_only",
                action="sample_only",
                sample_reason=f"sampled_{row_output_kind(row)}",
                policy=policy,
            )
        )
    return items


def count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter = Counter(str(row.get(key) or "unknown") for row in rows)
    return dict(sorted(counter.items()))


def output_kind_counts(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(row_output_kind(row) for row in rows + failures)
    counts = {key: counter.get(key, 0) for key in OUTPUT_KINDS if counter.get(key, 0)}
    for key, value in sorted(counter.items()):
        if key not in OUTPUT_KINDS and value:
            counts[key] = value
    return counts


def review_burden_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "requires_manual_review_now": sum(1 for item in items if item.get("requires_manual_review_now") is True),
        "eligible_for_sampled_review": sum(1 for item in items if item.get("review_priority") == "sample_only"),
        "kept_low_commitment_without_immediate_review": sum(
            1
            for item in items
            if item.get("review_pool") == "hypothesis_pool_candidates"
            and item.get("requires_manual_review_now") is False
        ),
        "archived_provider_failures": sum(1 for item in items if item.get("review_pool") == "provider_failure_pool"),
    }


def annotation_counts(items: list[dict[str, Any]], key: str) -> dict[str, int]:
    return count_by_key(items, key)


def claim_attribution_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    annotations = [annotate_claim_attribution(row) for row in rows]
    return {
        "claim_type_hint": count_by_key(annotations, "claim_type_hint"),
        "fact_promotion_eligibility": count_by_key(annotations, "fact_promotion_eligibility"),
        "promotion_path": count_by_key(annotations, "promotion_path"),
    }


def skip_audit_signal_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    skip_rows = [row for row in rows if row_output_kind(row) == "skipped"]
    signals = [skip_audit_signals(row) for row in skip_rows]
    tag_counter: Counter[str] = Counter()
    for signal in signals:
        tag_counter.update(str(tag) for tag in signal.get("signal_tags") or [])
    return {
        "modeling_value_hint": count_by_key(signals, "modeling_value_hint"),
        "attribution_risk_hint": count_by_key(signals, "attribution_risk_hint"),
        "promotion_risk_hint": count_by_key(signals, "promotion_risk_hint"),
        "evidence_directness_hint": count_by_key(signals, "evidence_directness_hint"),
        "claim_target_clarity_hint": count_by_key(signals, "claim_target_clarity_hint"),
        "context_requirement_hint": count_by_key(signals, "context_requirement_hint"),
        "review_priority_hint": count_by_key(signals, "review_priority_hint"),
        "signal_tags": dict(sorted(tag_counter.items())),
    }


def render_examples(items: list[dict[str, Any]], rows_by_id: dict[str, dict[str, Any]], pool: str, limit: int = 3) -> list[str]:
    lines: list[str] = []
    for item in [candidate for candidate in items if candidate.get("review_pool") == pool][:limit]:
        row = rows_by_id.get(str(item.get("proposal_id"))) or {}
        text = (
            row.get("fact_candidate_text")
            or row.get("hypothesis_text")
            or row.get("candidate_text")
            or row.get("source_text_quote")
            or row.get("source_text_preview")
            or ""
        )
        text = " ".join(str(text).split())[:240]
        claim_type = item.get("claim_type_hint") or "unknown"
        eligibility = item.get("fact_promotion_eligibility") or "unknown"
        skip_signal = item.get("skip_audit_signals") or {}
        signal_text = ""
        if item.get("output_kind") in {"skipped", "reject"}:
            signal_text = (
                f" [value={skip_signal.get('modeling_value_hint')}, "
                f"risk={skip_signal.get('attribution_risk_hint')}, "
                f"context={skip_signal.get('context_requirement_hint')}, "
                f"review={skip_signal.get('review_priority_hint')}]"
            )
        lines.append(
            f"- `{item.get('proposal_id')}` {item.get('output_kind')} "
            f"[claim_type={claim_type}, fact_eligibility={eligibility}] "
            f"-> {item.get('default_review_action')}{signal_text}: {text}"
        )
    if not lines:
        lines.append("- none")
    return lines


def render_report(
    *,
    manifest: dict[str, Any],
    proposals_path: Path,
    failures_path: Path,
    items_path: Path | None,
    proposals: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    items: list[dict[str, Any]],
    policy: SamplingPolicy,
) -> str:
    proposal_run_id = manifest.get("proposal_run_id", "unknown")
    rows_by_id = {str(row.get("proposal_id") or row.get("proposal_input_id")): row for row in proposals + failures}
    output_counts = output_kind_counts(proposals, failures)
    pool_counts = count_by_key(items, "review_pool")
    priority_counts = count_by_key(items, "review_priority")
    burden_counts = review_burden_counts(items)
    all_attribution_counts = claim_attribution_counts(proposals + failures)
    all_skip_signal_counts = skip_audit_signal_counts(proposals + failures)
    lines = [
        "# Review / Calibration Report",
        "",
        f"- proposal_run_id: `{proposal_run_id}`",
        f"- proposal_profile_id: `{manifest.get('proposal_profile_id')}`",
        f"- workspace: `{manifest.get('workspace')}`",
        f"- provider: `{manifest.get('provider')}`",
        f"- proposals: `{proposals_path}`",
        f"- model_output_failures: `{failures_path}`",
        f"- review_items: `{items_path}`" if items_path else "- review_items: not written",
        f"- created_at: `{now_iso()}`",
        "",
        "## Sampling Policy",
        "",
        f"- sampling_policy_id: `{policy.sampling_policy_id}`",
        f"- sampling_seed: `{policy.sampling_seed}`",
        f"- sample_rate: `{policy.sample_rate}`",
        f"- min_sample_count: `{policy.min_sample_count}`",
        "",
        "## Output Kind Counts",
        "",
    ]
    for key, value in output_counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Review Pool Counts", ""])
    lines.append("")
    lines.append(
        "Review pool counts are bounded review/calibration item counts after sampling and queue selection."
    )
    lines.append("")
    for key in REVIEW_POOLS:
        if key in pool_counts:
            lines.append(f"- {key}: {pool_counts[key]}")
    lines.extend(["", "## Review Priority Counts", ""])
    for key, value in priority_counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Review Burden", ""])
    for key, value in burden_counts.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Claim Attribution Counts", ""])
    lines.append(f"- annotation_schema_version: `{CLAIM_ATTRIBUTION_SCHEMA_VERSION}`")
    lines.append(f"- counted_rows: `{len(proposals) + len(failures)}`")
    lines.append("- promotion_path counts are annotation-level counts across all proposal rows.")
    lines.append(
        "- These counts may differ from review_pool counts because review_pool counts are bounded sampled/queued review items."
    )
    lines.append("")
    lines.append("### claim_type_hint")
    lines.append("")
    for key, value in all_attribution_counts["claim_type_hint"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("### fact_promotion_eligibility")
    lines.append("")
    for key, value in all_attribution_counts["fact_promotion_eligibility"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")
    lines.append("### promotion_path")
    lines.append("")
    for key, value in all_attribution_counts["promotion_path"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Skip Audit Signal Counts", ""])
    lines.append(f"- signal_schema_version: `{SKIP_AUDIT_SIGNAL_VERSION}`")
    lines.append("- counted_rows: skipped proposal rows only")
    lines.append("- signals are heuristic sampling/calibration hints, not truth, not promotion authority, and not graph proof.")
    for section_key, counts in all_skip_signal_counts.items():
        lines.append("")
        lines.append(f"### {section_key}")
        lines.append("")
        if counts:
            for key, value in counts.items():
                lines.append(f"- {key}: {value}")
        else:
            lines.append("- none")
    lines.extend(
        [
            "",
            "## Pool Examples",
            "",
            "### fact_promotion_review_queue",
            "",
            *render_examples(items, rows_by_id, "fact_promotion_review_queue"),
            "",
            "### hypothesis_pool_candidates",
            "",
            *render_examples(items, rows_by_id, "hypothesis_pool_candidates"),
            "",
            "### contamination_risk_pool",
            "",
            *render_examples(items, rows_by_id, "contamination_risk_pool"),
            "",
            "### uncertain_pool",
            "",
            *render_examples(items, rows_by_id, "uncertain_pool"),
            "",
            "### future_relation_candidate",
            "",
            *render_examples(items, rows_by_id, "future_relation_candidate"),
            "",
            "### no_promotion",
            "",
            *render_examples(items, rows_by_id, "no_promotion"),
            "",
            "### sampled_reject_skip_pool",
            "",
            *render_examples(items, rows_by_id, "sampled_reject_skip_pool"),
            "",
            "### provider_failure_pool",
            "",
            *render_examples(items, rows_by_id, "provider_failure_pool"),
            "",
            "## Boundaries",
            "",
            "- No live API calls are made by this report generator.",
            "- No durable memory writes are made.",
            "- No reviewed portrait units are written.",
            "- No current portrait is written.",
            "- No graph truth is written.",
            "- No support status is set.",
            "- `write_permission=false` is preserved for review items.",
            "",
            "## Recommended Next Action",
            "",
            "Review this report shape and sampling behavior before running it over larger live outputs.",
            "",
        ]
    )
    return "\n".join(lines)


def build_review_calibration(
    *,
    proposal_dir: Path,
    proposals_path: Path | None = None,
    failures_path: Path | None = None,
    manifest_path: Path | None = None,
    report_path: Path | None = None,
    items_path: Path | None = None,
    write_items: bool = False,
    policy: SamplingPolicy | None = None,
) -> dict[str, Any]:
    policy = policy or SamplingPolicy()
    proposal_dir = proposal_dir.resolve()
    proposals_path = (proposals_path or proposal_dir / "proposal_outcomes.ai.jsonl").resolve()
    failures_path = (failures_path or proposal_dir / "model_output_failures.jsonl").resolve()
    manifest_path = (manifest_path or proposal_dir / "proposal_run_manifest.json").resolve()
    report_path = (report_path or proposal_dir / "review_calibration_report.md").resolve()
    items_path = (items_path or proposal_dir / "review_calibration_items.jsonl").resolve()
    for path in (proposals_path, failures_path, manifest_path, report_path, items_path):
        ensure_within(path, [proposal_dir])

    manifest = read_json(manifest_path)
    proposals = read_jsonl(proposals_path)
    failures = read_jsonl(failures_path)
    items = classify_review_items(proposals, failures, policy)
    write_text(
        report_path,
        render_report(
            manifest=manifest,
            proposals_path=proposals_path,
            failures_path=failures_path,
            items_path=items_path if write_items else None,
            proposals=proposals,
            failures=failures,
            items=items,
            policy=policy,
        ),
    )
    if write_items:
        write_jsonl(items_path, items)
    return {
        "proposal_dir": str(proposal_dir),
        "review_calibration_report": str(report_path),
        "review_calibration_items": str(items_path) if write_items else None,
        "proposal_rows": len(proposals),
        "failure_rows": len(failures),
        "review_item_rows": len(items),
        "output_kind_counts": output_kind_counts(proposals, failures),
        "review_pool_counts": count_by_key(items, "review_pool"),
        "review_burden_counts": review_burden_counts(items),
        "claim_attribution_counts": claim_attribution_counts(proposals + failures),
        "skip_audit_signal_counts": skip_audit_signal_counts(proposals + failures),
        "input_hashes": {
            "proposals_hash": file_hash(proposals_path),
            "failures_hash": file_hash(failures_path),
            "manifest_hash": file_hash(manifest_path),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a dry-run review/calibration report for proposal outputs.")
    parser.add_argument("--proposal-dir", required=True)
    parser.add_argument("--proposals", default=None)
    parser.add_argument("--failures", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument("--items", default=None)
    parser.add_argument("--write-items", action="store_true")
    parser.add_argument("--sampling-policy-id", default=DEFAULT_SAMPLING_POLICY_ID)
    parser.add_argument("--sampling-seed", default=DEFAULT_SAMPLING_SEED)
    parser.add_argument("--sample-rate", type=float, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--min-sample-count", type=int, default=DEFAULT_MIN_SAMPLE_COUNT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    proposal_dir = Path(args.proposal_dir).resolve()
    result = build_review_calibration(
        proposal_dir=proposal_dir,
        proposals_path=Path(args.proposals).resolve() if args.proposals else None,
        failures_path=Path(args.failures).resolve() if args.failures else None,
        manifest_path=Path(args.manifest).resolve() if args.manifest else None,
        report_path=Path(args.report).resolve() if args.report else None,
        items_path=Path(args.items).resolve() if args.items else None,
        write_items=bool(args.write_items),
        policy=SamplingPolicy(
            sampling_policy_id=args.sampling_policy_id,
            sampling_seed=args.sampling_seed,
            sample_rate=args.sample_rate,
            min_sample_count=args.min_sample_count,
        ),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
