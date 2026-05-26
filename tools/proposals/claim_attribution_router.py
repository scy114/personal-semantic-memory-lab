"""Mock/dry-run claim attribution annotations for proposal outcomes.

This module is intentionally small. It adds soft attribution and promotion
routing hints before review/calibration reporting. It does not create graph
nodes, write durable memory, or prove claim truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "claim_attribution_annotation.v0.2.1"
CLAIM_TYPES = {
    "subject_intrinsic_fact",
    "subject_belief_about_node",
    "subject_relation_to_node",
    "node_state_from_subject_perspective",
    "interaction_hypothesis",
    "no_useful_modeling_value",
    "unknown",
}
FACT_PROMOTION_ELIGIBILITY = {"eligible", "ineligible", "uncertain"}

NO_USEFUL_TEXT = {
    "ok",
    "okay",
    "yes",
    "no",
    "thanks",
    "thank you",
    "好的",
    "好",
    "嗯",
    "行",
}

POSITIVE_EVALUATION_WORDS = {
    "smart",
    "talented",
    "great",
    "excellent",
    "brilliant",
    "clever",
    "capable",
    "amazing",
    "聪明",
    "厉害",
    "优秀",
}

FIRST_PERSON_PATTERNS = [
    r"\bi prefer\b",
    r"\bi like\b",
    r"\bi need\b",
    r"\bi want\b",
    r"\bi cannot\b",
    r"\bi can't\b",
    r"\bi am\b",
    r"\bi'm\b",
    r"\bmy\b",
    r"\bme too\b",
    r"\bsame here\b",
    r"\bgo-to\b",
    r"\bspeaks to me\b",
    r"\b我喜欢\b",
    r"\b我需要\b",
    r"\b我想\b",
    r"\b我不能\b",
    r"\b我的\b",
]

RELATION_PATTERNS = [
    r"\bi should learn from\b",
    r"\blearn from you\b",
    r"\blet'?s\b.*\b(discuss|meet|work|collaborate|eat|dinner)\b",
    r"\bi use\b.*\b(tool|project|for this project)\b",
    r"\b向你学习\b",
    r"\b一起\b.*\b(讨论|吃饭|合作)\b",
]


def short_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:length]


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


def row_output_kind(row: dict[str, Any]) -> str:
    return str(row.get("output_kind") or row.get("proposal_status") or "unknown")


def compact_text(value: Any) -> str:
    if isinstance(value, list):
        return " ".join(str(item) for item in value if item)
    return str(value or "")


def evidence_text(row: dict[str, Any]) -> str:
    parts = [
        row.get("fact_candidate_text"),
        row.get("hypothesis_text"),
        row.get("candidate_text"),
        row.get("source_text_quote"),
        compact_text(row.get("source_text_quotes")),
        compact_text(row.get("supporting_observations")),
    ]
    return " ".join(str(part).strip() for part in parts if str(part or "").strip())


def normalized_text(row: dict[str, Any]) -> str:
    return " ".join(evidence_text(row).split())


def has_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def contains_positive_evaluation(text: str) -> bool:
    lower = text.lower()
    return any(word in lower for word in POSITIVE_EVALUATION_WORDS)


def is_direct_address_evaluation(text: str) -> bool:
    lower = text.lower()
    named_evaluation = re.search(
        r"\b([A-Z][A-Za-z]+)\s+is\s+(really\s+)?(smart|talented|great|excellent|brilliant|clever|capable)\b",
        text,
    )
    named_subject = named_evaluation.group(1).lower() if named_evaluation else ""
    return (
        (bool(re.search(r"\byou('?re| are)\b", lower)) and contains_positive_evaluation(lower))
        or bool(re.search(r"^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?,\s+you\b", text))
        or (bool(named_evaluation) and named_subject not in {"that", "this", "it"})
        or bool(re.search(r"你.*(聪明|厉害|优秀)", text))
    )


def is_question_or_compliment_about_other(text: str) -> bool:
    lower = text.lower()
    return (
        "?" in text
        and bool(re.search(r"\byou\b|\byour\b|\bdo you\b|\bgot any\b", lower))
    ) or is_direct_address_evaluation(text)


def node_type_hint(text: str) -> str:
    lower = text.lower()
    if re.search(r"\btool\b|\bapp\b|\bmodel\b|\bapi\b|工具", lower):
        return "tool"
    if re.search(r"\bproject\b|\bworkflow\b|\bplan\b|项目", lower):
        return "project"
    if re.search(r"\bdinner\b|\bmeeting\b|\bexchange\b|\bsession\b|饭|会议|交流", lower):
        return "event"
    if re.search(r"\b[A-Z][A-Za-z]+\b", text) or re.search(r"\byou\b|\byour\b|你|小红|小明", lower):
        return "person"
    return "unknown"


def claim_target_hint(text: str, claim_type: str) -> str:
    if claim_type == "subject_intrinsic_fact":
        return "target_subject"
    match = re.search(r"^([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+)?),\s+you\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\b([A-Z][A-Za-z]+)\s+is\s+(?:really\s+)?(?:smart|talented|great|excellent|brilliant|clever|capable)\b", text)
    if match:
        return match.group(1)
    if "you" in text.lower() or "你" in text:
        return "addressed_other_node"
    if "tool" in text.lower():
        return "tool"
    if "project" in text.lower():
        return "project"
    return "unknown"


def base_annotation(row: dict[str, Any]) -> dict[str, Any]:
    proposal_id = str(row.get("proposal_id") or row.get("proposal_input_id") or "")
    proposal_run_id = str(row.get("proposal_run_id") or "")
    return {
        "schema_version": SCHEMA_VERSION,
        "annotation_id": f"caa:{short_hash(proposal_run_id + ':' + proposal_id)}",
        "proposal_id": proposal_id,
        "proposal_run_id": proposal_run_id,
        "claim_type_hint": "unknown",
        "source_perspective": str(row.get("source_perspective") or row.get("perspective") or "unknown"),
        "claim_target_hint": "unknown",
        "claim_target_node_type": "unknown",
        "attribution_status_hint": "unknown",
        "promotion_path": "unknown",
        "fact_promotion_eligibility": "uncertain",
        "routing_reasons": [],
        "warnings": [],
        "write_permission": False,
    }


def finalize(
    annotation: dict[str, Any],
    *,
    claim_type: str,
    attribution_status: str,
    promotion_path: str,
    fact_eligibility: str,
    reasons: list[str],
    warnings: list[str] | None = None,
    target_hint: str | None = None,
    target_node_type: str | None = None,
) -> dict[str, Any]:
    annotation["claim_type_hint"] = claim_type
    annotation["attribution_status_hint"] = attribution_status
    annotation["promotion_path"] = promotion_path
    annotation["fact_promotion_eligibility"] = fact_eligibility
    annotation["routing_reasons"] = reasons
    annotation["warnings"] = sorted(set([*annotation.get("warnings", []), *(warnings or [])]))
    if target_hint is not None:
        annotation["claim_target_hint"] = target_hint
    if target_node_type is not None:
        annotation["claim_target_node_type"] = target_node_type
    if claim_type not in CLAIM_TYPES:
        raise ValueError(f"Invalid claim_type_hint: {claim_type}")
    if fact_eligibility not in FACT_PROMOTION_ELIGIBILITY:
        raise ValueError(f"Invalid fact_promotion_eligibility: {fact_eligibility}")
    return annotation


def annotate_claim_attribution(row: dict[str, Any]) -> dict[str, Any]:
    """Return one soft v0.2.1 attribution annotation for a proposal row."""

    annotation = base_annotation(row)
    text = normalized_text(row)
    lower = text.lower().strip()
    output_kind = row_output_kind(row)

    if row.get("warnings"):
        annotation["warnings"] = [str(item) for item in row.get("warnings") or []]

    if output_kind == "model_failure":
        return finalize(
            annotation,
            claim_type="no_useful_modeling_value",
            attribution_status="unknown",
            promotion_path="no_promotion",
            fact_eligibility="ineligible",
            reasons=["model_failure_has_no_claim"],
            target_node_type="unknown",
        )

    if not lower or lower in NO_USEFUL_TEXT:
        return finalize(
            annotation,
            claim_type="no_useful_modeling_value",
            attribution_status="explicit" if lower else "unknown",
            promotion_path="no_promotion",
            fact_eligibility="ineligible",
            reasons=["no_useful_modeling_value"],
            target_node_type="unknown",
        )

    if "encourages" in lower and "exchange" in lower:
        return finalize(
            annotation,
            claim_type="interaction_hypothesis",
            attribution_status="inferred",
            promotion_path="hypothesis_pool_candidates",
            fact_eligibility="ineligible",
            reasons=["local_interaction_hypothesis"],
            target_hint=claim_target_hint(text, "interaction_hypothesis"),
            target_node_type="person",
        )

    if is_direct_address_evaluation(text):
        return finalize(
            annotation,
            claim_type="subject_belief_about_node",
            attribution_status="inferred",
            promotion_path="hypothesis_pool_candidates",
            fact_eligibility="ineligible",
            reasons=["evaluation_about_other_node_not_intrinsic_fact"],
            warnings=["other_node_objective_mispromotion_risk"] if output_kind == "portrait_fact_candidate" else [],
            target_hint=claim_target_hint(text, "subject_belief_about_node"),
            target_node_type="person",
        )

    if has_pattern(lower, RELATION_PATTERNS):
        return finalize(
            annotation,
            claim_type="subject_relation_to_node",
            attribution_status="explicit" if "i " in lower or "我" in text else "inferred",
            promotion_path="future_relation_candidate",
            fact_eligibility="ineligible",
            reasons=["subject_relation_to_node"],
            target_hint=claim_target_hint(text, "subject_relation_to_node"),
            target_node_type=node_type_hint(text),
        )

    if is_question_or_compliment_about_other(text):
        return finalize(
            annotation,
            claim_type="subject_relation_to_node",
            attribution_status="inferred",
            promotion_path="hypothesis_pool_candidates",
            fact_eligibility="ineligible",
            reasons=["other_node_directed_turn_preserved_as_relation_or_interaction"],
            warnings=["fact_promotion_blocked_by_attribution"],
            target_hint=claim_target_hint(text, "subject_relation_to_node"),
            target_node_type=node_type_hint(text),
        )

    if output_kind == "portrait_hypothesis_candidate":
        return finalize(
            annotation,
            claim_type="interaction_hypothesis",
            attribution_status="inferred",
            promotion_path="hypothesis_pool_candidates",
            fact_eligibility="ineligible",
            reasons=["proposal_is_hypothesis_candidate"],
            target_hint=claim_target_hint(text, "interaction_hypothesis"),
            target_node_type=node_type_hint(text),
        )

    if output_kind == "portrait_fact_candidate" and has_pattern(lower, FIRST_PERSON_PATTERNS):
        return finalize(
            annotation,
            claim_type="subject_intrinsic_fact",
            attribution_status="explicit",
            promotion_path="fact_promotion_review_queue",
            fact_eligibility="eligible",
            reasons=["explicit_target_subject_self_report"],
            target_hint="target_subject",
            target_node_type="person",
        )

    if output_kind in {"reject", "skipped"}:
        return finalize(
            annotation,
            claim_type="no_useful_modeling_value",
            attribution_status="unknown",
            promotion_path="no_promotion",
            fact_eligibility="ineligible",
            reasons=[f"{output_kind}_row"],
            target_node_type="unknown",
        )

    return finalize(
        annotation,
        claim_type="unknown",
        attribution_status="ambiguous",
        promotion_path="uncertain_pool",
        fact_eligibility="uncertain",
        reasons=["insufficient_attribution_signal"],
        target_hint=claim_target_hint(text, "unknown"),
        target_node_type=node_type_hint(text),
    )


def annotate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [annotate_claim_attribution(row) for row in rows]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate mock/dry-run claim attribution annotations.")
    parser.add_argument("--input", required=True, help="Input proposal_outcomes.ai.jsonl or failures JSONL.")
    parser.add_argument("--output", required=True, help="Output claim_attribution_annotations.jsonl.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_jsonl(Path(args.input))
    write_jsonl(Path(args.output), annotate_rows(rows))
    print(json.dumps({"input_rows": len(rows), "annotation_rows": len(rows), "output": args.output}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
