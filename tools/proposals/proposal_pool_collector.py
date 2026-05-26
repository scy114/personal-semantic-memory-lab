"""Collect non-durable S2 proposal pools from review/calibration items.

This is a post-proposal organizer. It does not accept proposals, write durable
memory, write reviewed portrait units, create graph truth, or call models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.proposals.review_calibration_report import (
    SamplingPolicy,
    build_review_calibration,
    ensure_within,
    read_json,
    read_jsonl,
    short_hash,
    write_jsonl,
)


SCHEMA_VERSION = "s2.proposal_pool_collection.v0.1"
FACT_POOL_FILENAME = "fact_promotion_candidates.review.jsonl"
HYPOTHESIS_POOL_FILENAME = "active_hypothesis_candidates.low_commitment.jsonl"
CALIBRATION_SEED_FILENAME = "calibration_review_seed_items.jsonl"
MANIFEST_FILENAME = "proposal_pool_manifest.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def item_key(item: dict[str, Any], prefix: str) -> str:
    proposal_id = str(item.get("proposal_id") or item.get("review_item_id") or "")
    pool = str(item.get("review_pool") or "")
    return f"{prefix}:{short_hash(proposal_id + ':' + pool)}"


def common_source_fields(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "proposal_id": item.get("proposal_id"),
        "proposal_run_id": item.get("proposal_run_id"),
        "proposal_profile_id": item.get("proposal_profile_id"),
        "review_item_id": item.get("review_item_id"),
        "output_kind": item.get("output_kind"),
        "epistemic_status": item.get("epistemic_status"),
        "source_text": item.get("source_text") or "",
        "source_text_preview": item.get("source_text_preview") or "",
        "source_text_quote": item.get("source_text_quote") or "",
        "evidence_refs": item.get("evidence_refs") or [],
        "raw_backpointer_refs": item.get("raw_backpointer_refs") or [],
        "warnings": item.get("warnings") or [],
        "claim_attribution_annotation": item.get("claim_attribution_annotation") or {},
        "claim_type_hint": item.get("claim_type_hint"),
        "claim_target_hint": item.get("claim_target_hint"),
        "source_perspective": item.get("source_perspective"),
        "attribution_status_hint": item.get("attribution_status_hint"),
        "promotion_path": item.get("promotion_path"),
        "fact_promotion_eligibility": item.get("fact_promotion_eligibility"),
        "write_permission": False,
    }


def fact_pool_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "s2.fact_promotion_candidate.review.v0.1",
        "fact_pool_item_id": item_key(item, "fpc"),
        "pool": "fact_promotion_review_queue",
        "fact_candidate_text": item.get("fact_candidate_text") or item.get("candidate_text") or "",
        "promotion_review_status": "needs_strict_promotion_review",
        "promotion_allowed_without_review": False,
        "default_review_action": item.get("default_review_action"),
        "requires_manual_review_now": True,
        **common_source_fields(item),
    }


def hypothesis_pool_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "s2.active_hypothesis_candidate.v0.1",
        "hypothesis_pool_item_id": item_key(item, "hpc"),
        "pool": "hypothesis_pool_candidates",
        "hypothesis_text": item.get("hypothesis_text") or item.get("candidate_text") or "",
        "hypothesis_status": "active_unreviewed",
        "commitment_status": "low_commitment",
        "promotion_allowed_without_review": False,
        "requires_manual_review_now": False,
        "default_review_action": item.get("default_review_action"),
        **common_source_fields(item),
    }


def calibration_seed_row(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "s2.calibration_seed_item.v0.1",
        "calibration_seed_item_id": item_key(item, "csi"),
        "review_pool": item.get("review_pool"),
        "review_priority": item.get("review_priority"),
        "default_review_action": item.get("default_review_action"),
        "calibration_action": item.get("calibration_action") or "unlabeled",
        "review_sample_reason": item.get("review_sample_reason"),
        "skip_audit_signals": item.get("skip_audit_signals") or {},
        "modeling_value_hint": item.get("modeling_value_hint"),
        "attribution_risk_hint": item.get("attribution_risk_hint"),
        "promotion_risk_hint": item.get("promotion_risk_hint"),
        "context_requirement_hint": item.get("context_requirement_hint"),
        "requires_manual_review_now": bool(item.get("requires_manual_review_now")),
        **common_source_fields(item),
        "fact_candidate_text": item.get("fact_candidate_text") or "",
        "hypothesis_text": item.get("hypothesis_text") or "",
        "candidate_text": item.get("candidate_text") or "",
    }


def load_or_build_review_items(proposal_dir: Path, review_items_path: Path, *, build_if_missing: bool) -> list[dict[str, Any]]:
    if review_items_path.exists():
        return read_jsonl(review_items_path)
    if not build_if_missing:
        raise FileNotFoundError(f"Review items file not found: {review_items_path}")
    build_review_calibration(
        proposal_dir=proposal_dir,
        items_path=review_items_path,
        write_items=True,
        policy=SamplingPolicy(),
    )
    return read_jsonl(review_items_path)


def collect_proposal_pools(
    *,
    proposal_dir: Path,
    review_items_path: Path | None = None,
    build_review_items_if_missing: bool = True,
    fact_pool_path: Path | None = None,
    hypothesis_pool_path: Path | None = None,
    calibration_seed_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    proposal_dir = proposal_dir.resolve()
    review_items_path = (review_items_path or proposal_dir / "review_calibration_items.jsonl").resolve()
    fact_pool_path = (fact_pool_path or proposal_dir / FACT_POOL_FILENAME).resolve()
    hypothesis_pool_path = (hypothesis_pool_path or proposal_dir / HYPOTHESIS_POOL_FILENAME).resolve()
    calibration_seed_path = (calibration_seed_path or proposal_dir / CALIBRATION_SEED_FILENAME).resolve()
    manifest_path = (manifest_path or proposal_dir / MANIFEST_FILENAME).resolve()

    for path in (review_items_path, fact_pool_path, hypothesis_pool_path, calibration_seed_path, manifest_path):
        ensure_within(path, [proposal_dir])

    manifest_source_path = proposal_dir / "proposal_run_manifest.json"
    source_manifest = read_json(manifest_source_path) if manifest_source_path.exists() else {}
    items = load_or_build_review_items(
        proposal_dir,
        review_items_path,
        build_if_missing=build_review_items_if_missing,
    )

    fact_rows = [
        fact_pool_row(item)
        for item in items
        if item.get("review_pool") == "fact_promotion_review_queue"
    ]
    hypothesis_rows = [
        hypothesis_pool_row(item)
        for item in items
        if item.get("review_pool") == "hypothesis_pool_candidates"
    ]
    calibration_rows = [
        calibration_seed_row(item)
        for item in items
        if item.get("review_pool")
        in {
            "contamination_risk_pool",
            "uncertain_pool",
            "sampled_reject_skip_pool",
            "schema_unknown_pool",
        }
    ]

    write_jsonl(fact_pool_path, fact_rows)
    write_jsonl(hypothesis_pool_path, hypothesis_rows)
    write_jsonl(calibration_seed_path, calibration_rows)

    pool_counts = Counter(str(item.get("review_pool") or "unknown") for item in items)
    output = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "proposal_dir": str(proposal_dir),
        "proposal_run_id": source_manifest.get("proposal_run_id"),
        "proposal_profile_id": source_manifest.get("proposal_profile_id"),
        "source_review_items_path": str(review_items_path),
        "source_review_items_hash": file_hash(review_items_path),
        "fact_pool_path": str(fact_pool_path),
        "fact_pool_hash": file_hash(fact_pool_path),
        "hypothesis_pool_path": str(hypothesis_pool_path),
        "hypothesis_pool_hash": file_hash(hypothesis_pool_path),
        "calibration_seed_path": str(calibration_seed_path),
        "calibration_seed_hash": file_hash(calibration_seed_path),
        "review_item_rows": len(items),
        "fact_pool_rows": len(fact_rows),
        "hypothesis_pool_rows": len(hypothesis_rows),
        "calibration_seed_rows": len(calibration_rows),
        "review_pool_counts": dict(sorted(pool_counts.items())),
        "durable_memory_written": False,
        "reviewed_portrait_units_written": False,
        "current_portrait_written": False,
        "graph_truth_written": False,
        "automatic_acceptance_executed": False,
        "write_permission": False,
    }
    write_json(manifest_path, output)
    output["manifest_path"] = str(manifest_path)
    output["manifest_hash"] = file_hash(manifest_path)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect non-durable S2 proposal pools.")
    parser.add_argument("--proposal-dir", required=True)
    parser.add_argument("--review-items", default=None)
    parser.add_argument("--no-build-review-items", action="store_true")
    parser.add_argument("--fact-pool", default=None)
    parser.add_argument("--hypothesis-pool", default=None)
    parser.add_argument("--calibration-seed", default=None)
    parser.add_argument("--manifest", default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = collect_proposal_pools(
        proposal_dir=Path(args.proposal_dir),
        review_items_path=Path(args.review_items).resolve() if args.review_items else None,
        build_review_items_if_missing=not args.no_build_review_items,
        fact_pool_path=Path(args.fact_pool).resolve() if args.fact_pool else None,
        hypothesis_pool_path=Path(args.hypothesis_pool).resolve() if args.hypothesis_pool else None,
        calibration_seed_path=Path(args.calibration_seed).resolve() if args.calibration_seed else None,
        manifest_path=Path(args.manifest).resolve() if args.manifest else None,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
