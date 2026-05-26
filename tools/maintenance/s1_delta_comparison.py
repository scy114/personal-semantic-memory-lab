"""Compare new S1 units against an active S1 latest view.

This v0.4 helper emits candidate delta decisions for incremental maintenance.
It is a retrieval-style comparison report, not a truth classifier: ambiguous
semantic changes are routed to review or later LLM assist instead of being
declared as contradictions by script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "maintenance.s1_delta_decision.v0.4"
REPORT_SCHEMA_VERSION = "maintenance.s1_delta_report.v0.4"

DELTA_TYPES = {
    "new_unit",
    "duplicate_candidate",
    "strengthens",
    "weakens",
    "contradicts",
    "supersedes",
    "needs_review",
    "no_material_change",
}

WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"S1 delta input row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def memory_text(row: dict[str, Any]) -> str:
    for field_name in ("processed_text", "content", "memory_candidate_text", "original_text", "text"):
        value = normalize_text(row.get(field_name))
        if value:
            return value
    assisted = row.get("llm_assisted_s1_candidate") or {}
    if isinstance(assisted, dict):
        value = normalize_text(assisted.get("memory_candidate_text"))
        if value:
            return value
    return ""


def row_id(row: dict[str, Any]) -> str:
    for field_name in ("memory_id", "id", "candidate_id", "object_id"):
        value = row.get(field_name)
        if value:
            return str(value)
    text = memory_text(row)
    if text:
        return "s1row:" + sha256(text.encode("utf-8")).hexdigest()[:16]
    return ""


def list_refs(row: dict[str, Any], *field_names: str) -> list[str]:
    refs: list[str] = []
    for field_name in field_names:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for key in ("evidence_ref", "canonical_evidence_ref", "source_id", "id"):
                        if item.get(key):
                            refs.append(str(item[key]))
                            break
                else:
                    refs.append(str(item))
        elif isinstance(value, dict):
            for key in ("evidence_ref", "canonical_evidence_ref", "source_id", "id"):
                if value.get(key):
                    refs.append(str(value[key]))
                    break
        else:
            refs.append(str(value))
    seen: set[str] = set()
    normalized: list[str] = []
    for ref in refs:
        text = ref.strip()
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


def subject_id(row: dict[str, Any]) -> str:
    for field_name in ("subject_id", "target_subject_id", "user_id", "modeled_user_id"):
        value = row.get(field_name)
        if value:
            return str(value)
    return ""


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in WORD_RE.findall(text) if token.strip()]


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def token_cosine(left: str, right: str) -> float:
    left_counts = Counter(tokenize(left))
    right_counts = Counter(tokenize(right))
    if not left_counts or not right_counts:
        return 0.0
    common = set(left_counts) & set(right_counts)
    numerator = sum(left_counts[token] * right_counts[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left_counts.values()))
    right_norm = math.sqrt(sum(value * value for value in right_counts.values()))
    if not left_norm or not right_norm:
        return 0.0
    return numerator / (left_norm * right_norm)


def overlap_ratio(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / min(len(left_set), len(right_set))


def compare_pair(new_row: dict[str, Any], existing_row: dict[str, Any]) -> dict[str, Any]:
    new_text = memory_text(new_row)
    existing_text = memory_text(existing_row)
    new_evidence_refs = list_refs(new_row, "evidence_refs", "backpointer_refs")
    existing_evidence_refs = list_refs(existing_row, "evidence_refs", "backpointer_refs")
    new_source_refs = list_refs(new_row, "source_refs", "source_ref", "raw_source_id")
    existing_source_refs = list_refs(existing_row, "source_refs", "source_ref", "raw_source_id")
    new_subject = subject_id(new_row)
    existing_subject = subject_id(existing_row)
    exact_text_match = bool(new_text and existing_text and new_text == existing_text)
    evidence_overlap = overlap_ratio(new_evidence_refs, existing_evidence_refs)
    source_overlap = overlap_ratio(new_source_refs, existing_source_refs)
    same_subject = bool(new_subject and existing_subject and new_subject == existing_subject)
    jaccard = token_jaccard(new_text, existing_text)
    cosine = token_cosine(new_text, existing_text)

    score = max(jaccard, cosine)
    if exact_text_match:
        score = max(score, 1.0)
    if evidence_overlap:
        score = max(score, 0.95)
    elif same_subject and source_overlap:
        score = max(score, min(0.9, score + 0.12))
    elif same_subject:
        score = max(score, min(0.8, score + 0.05))

    return {
        "existing_s1_unit_id": row_id(existing_row),
        "existing_subject_id": existing_subject,
        "exact_text_match": exact_text_match,
        "same_subject": same_subject,
        "text_jaccard": round(jaccard, 6),
        "text_cosine": round(cosine, 6),
        "evidence_overlap": round(evidence_overlap, 6),
        "source_overlap": round(source_overlap, 6),
        "candidate_match_score": round(score, 6),
        "existing_evidence_refs": existing_evidence_refs,
        "existing_source_refs": existing_source_refs,
    }


def choose_delta_type(matches: list[dict[str, Any]]) -> tuple[str, list[str], str]:
    if not matches:
        return "new_unit", ["no_existing_s1_neighbor_found"], "script_candidate"

    best = matches[0]
    if best["exact_text_match"] and best["evidence_overlap"] > 0:
        return "no_material_change", ["same_text_and_same_evidence"], "script_candidate"
    if best["evidence_overlap"] > 0:
        return "duplicate_candidate", ["same_or_overlapping_evidence_refs"], "review_recommended"
    if best["candidate_match_score"] >= 0.82 and best["same_subject"]:
        return "duplicate_candidate", ["high_text_similarity_same_subject"], "review_recommended"
    if best["candidate_match_score"] >= 0.55:
        return "needs_review", ["near_neighbor_requires_semantic_delta_check"], "llm_assist_recommended"
    return "new_unit", ["nearest_neighbor_below_review_threshold"], "script_candidate"


def build_delta_decision(
    new_row: dict[str, Any],
    existing_rows: list[dict[str, Any]],
    *,
    operation_id: str,
    generated_at: str,
    top_k: int,
) -> dict[str, Any]:
    comparisons = [compare_pair(new_row, existing_row) for existing_row in existing_rows]
    comparisons.sort(key=lambda item: item["candidate_match_score"], reverse=True)
    top_matches = comparisons[:top_k]
    delta_type, reasons, confidence = choose_delta_type(top_matches)
    if delta_type not in DELTA_TYPES:
        raise ValueError(f"Unsupported delta_type: {delta_type}")

    new_unit_id = row_id(new_row)
    return {
        "schema_version": SCHEMA_VERSION,
        "operation_id": operation_id,
        "new_s1_unit_id": new_unit_id,
        "new_subject_id": subject_id(new_row),
        "new_evidence_refs": list_refs(new_row, "evidence_refs", "backpointer_refs"),
        "new_source_refs": list_refs(new_row, "source_refs", "source_ref", "raw_source_id"),
        "delta_type": delta_type,
        "decision_confidence": confidence,
        "review_reasons": reasons,
        "top_matches": top_matches,
        "new_row_excerpt": memory_text(new_row)[:500],
        "method": "provenance_plus_lexical_candidate_retrieval",
        "llm_assist_status": "recommended" if confidence == "llm_assist_recommended" else "not_run",
        "script_is_not_semantic_truth": True,
        "write_permission": False,
        "generated_at": generated_at,
    }


def build_s1_delta_report(
    *,
    new_rows: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    operation_id: str = "",
    top_k: int = 5,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_generated_at = generated_at or now_iso()
    decisions = [
        build_delta_decision(
            deepcopy(row),
            active_rows,
            operation_id=operation_id,
            generated_at=resolved_generated_at,
            top_k=top_k,
        )
        for row in new_rows
    ]
    delta_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    for decision in decisions:
        delta_counts[decision["delta_type"]] = delta_counts.get(decision["delta_type"], 0) + 1
        confidence_counts[decision["decision_confidence"]] = confidence_counts.get(decision["decision_confidence"], 0) + 1
    sample_decisions = [
        {
            "new_s1_unit_id": decision["new_s1_unit_id"],
            "delta_type": decision["delta_type"],
            "decision_confidence": decision["decision_confidence"],
            "review_reasons": decision["review_reasons"],
            "best_match": decision["top_matches"][0] if decision["top_matches"] else None,
            "new_row_excerpt": decision["new_row_excerpt"],
        }
        for decision in decisions[:10]
    ]
    sample_decisions_by_delta_type: dict[str, list[dict[str, Any]]] = {}
    for sample in sample_decisions:
        sample_decisions_by_delta_type.setdefault(sample["delta_type"], []).append(sample)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "operation_id": operation_id,
        "new_s1_count": len(new_rows),
        "active_s1_count": len(active_rows),
        "decision_count": len(decisions),
        "delta_type_counts": dict(sorted(delta_counts.items())),
        "decision_confidence_counts": dict(sorted(confidence_counts.items())),
        "method": "provenance_plus_lexical_candidate_retrieval",
        "method_followups_not_yet_run": ["embedding_similarity", "graph_entity_hints", "llm_semantic_delta_classification"],
        "sample_decisions": sample_decisions,
        "sample_decisions_by_delta_type": dict(sorted(sample_decisions_by_delta_type.items())),
        "boundary": "S1 delta decisions are candidate maintenance signals, not semantic truth.",
        "llm_assist_not_run": True,
        "read_only": True,
        "write_permission": False,
        "generated_at": resolved_generated_at,
    }
    return {"decisions": decisions, "report": report}


def write_report_bundle(output_dir: Path, bundle: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / "s1_delta_decisions.jsonl"
    report_path = output_dir / "s1_delta_report.json"
    write_jsonl(decisions_path, bundle["decisions"])
    write_json(report_path, bundle["report"])
    return {"decisions_path": str(decisions_path), "report_path": str(report_path)}


def build_s1_delta_report_from_files(
    *,
    new_s1_jsonl: Path,
    active_s1_jsonl: Path,
    output_dir: Path,
    operation_id: str = "",
    top_k: int = 5,
) -> dict[str, Any]:
    bundle = build_s1_delta_report(
        new_rows=read_jsonl(new_s1_jsonl),
        active_rows=read_jsonl(active_s1_jsonl),
        operation_id=operation_id,
        top_k=top_k,
    )
    bundle["paths"] = write_report_bundle(output_dir, bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-s1-jsonl", required=True, type=Path)
    parser.add_argument("--active-s1-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--operation-id", default="")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    bundle = build_s1_delta_report_from_files(
        new_s1_jsonl=args.new_s1_jsonl,
        active_s1_jsonl=args.active_s1_jsonl,
        output_dir=args.output_dir,
        operation_id=args.operation_id,
        top_k=args.top_k,
    )
    print(json.dumps(bundle["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
