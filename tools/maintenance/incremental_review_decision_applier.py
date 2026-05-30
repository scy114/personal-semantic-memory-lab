"""Convert finalized v0.4 review decisions into an incremental apply plan.

The applier is still plan-only. It validates that a human review session was
finalized, checks queue/decision hashes, and writes overlay/scope artifacts for
the next maintenance stage. It does not mutate canonical S1/S2/graph assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


APPLY_PLAN_SCHEMA_VERSION = "maintenance.incremental_review_apply_plan_row.v0.4"
REPORT_SCHEMA_VERSION = "maintenance.incremental_review_apply_report.v0.4"
EXPECTED_SESSION_SCHEMA_VERSION = "maintenance.incremental_review_session.v0.4"

APPLY_PLAN_FILENAME = "incremental_apply_plan.jsonl"
S1_OVERLAY_FILENAME = "s1_latest_view_overlay.jsonl"
S2_REFRESH_SCOPE_FILENAME = "s2_refresh_scope.jsonl"
GRAPH_REFRESH_SCOPE_FILENAME = "graph_refresh_scope.jsonl"
APPLY_REPORT_FILENAME = "incremental_apply_report.md"
APPLY_REPORT_JSON_FILENAME = "incremental_apply_report.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
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
            raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_hash(value: Any) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def decisions_by_item(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in decisions:
        item_id = str(row.get("review_item_id") or "")
        if item_id:
            latest[item_id] = row
    return latest


def queue_by_id(queue_items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item.get("review_item_id") or ""): item for item in queue_items if item.get("review_item_id")}


def validate_finalized_session(
    *,
    queue_items: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    session_manifest: dict[str, Any],
) -> dict[str, Any]:
    if session_manifest.get("schema_version") != EXPECTED_SESSION_SCHEMA_VERSION:
        raise ValueError(f"unsupported_session_schema_version:{session_manifest.get('schema_version')}")
    if session_manifest.get("review_session_status") != "finalized":
        raise ValueError("review_session_not_finalized")
    latest = decisions_by_item(decisions)
    queue_ids = [str(item.get("review_item_id") or "") for item in queue_items if item.get("review_item_id")]
    decided_ids = [item_id for item_id in queue_ids if item_id in latest]
    undecided_ids = [item_id for item_id in queue_ids if item_id not in latest]
    if session_manifest.get("queue_hash") != stable_hash(queue_items):
        raise ValueError("review_session_queue_hash_mismatch")
    if session_manifest.get("decisions_hash") != stable_hash([latest[item_id] for item_id in decided_ids]):
        raise ValueError("review_session_decisions_hash_mismatch")
    if int(session_manifest.get("decision_count") or 0) != len(decided_ids):
        raise ValueError("review_session_decision_count_mismatch")
    if int(session_manifest.get("undecided_count") or 0) != len(undecided_ids):
        raise ValueError("review_session_undecided_count_mismatch")
    if undecided_ids and not bool(session_manifest.get("allow_partial_finalize")):
        raise ValueError("review_session_has_unallowed_undecided_items")
    return {"latest_decisions": latest, "queue_ids": queue_ids, "decided_ids": decided_ids, "undecided_ids": undecided_ids}


def normalize(value: Any) -> str:
    return str(value or "").strip()


def first_context_card(item: dict[str, Any], role: str) -> dict[str, Any] | None:
    for card in item.get("context_cards") or []:
        if isinstance(card, dict) and normalize(card.get("role")) == role:
            return card
    return None


def context_object_ids(item: dict[str, Any], role: str) -> list[str]:
    ids: list[str] = []
    for card in item.get("context_cards") or []:
        if isinstance(card, dict) and normalize(card.get("role")) == role:
            object_id = normalize(card.get("object_id"))
            if object_id:
                ids.append(object_id)
    return ids


def apply_effect_for_action(action: str) -> str:
    return {
        "accept_noop": "record_no_material_change",
        "accept_patch": "materialize_candidate_overlay",
        "accept_with_edit": "materialize_edited_candidate_overlay",
        "split_current_vs_historical": "split_current_and_historical_overlay",
        "mark_stale_only": "mark_existing_stale_overlay",
        "route_s2_build": "enqueue_s2_refresh_scope",
        "route_graph_extraction": "enqueue_graph_refresh_scope",
        "reject_patch": "record_rejected_patch",
        "defer": "record_deferred_patch",
        "needs_more_evidence": "record_evidence_needed",
    }.get(action, "unknown_action")


def build_apply_plan_row(
    *,
    item: dict[str, Any],
    decision: dict[str, Any],
    generated_at: str,
) -> dict[str, Any]:
    action = normalize(decision.get("internal_review_action") or decision.get("review_action"))
    return {
        "schema_version": APPLY_PLAN_SCHEMA_VERSION,
        "review_item_id": item["review_item_id"],
        "patch_id": item.get("patch_id") or "",
        "patch_hash": item.get("patch_hash") or "",
        "source_context_hash": item.get("source_context_hash") or "",
        "operation_id": item.get("operation_id") or "",
        "layer": item.get("layer") or "",
        "patch_action": item.get("patch_action") or "",
        "human_decision": decision.get("human_decision") or "",
        "internal_review_action": action,
        "review_status": decision.get("review_status") or "",
        "apply_effect": apply_effect_for_action(action),
        "edited_payload": decision.get("edited_payload") or {},
        "review_notes": decision.get("review_notes") or "",
        "candidate_context_ids": context_object_ids(item, "new_s1_candidate"),
        "existing_context_ids": context_object_ids(item, "existing_s1_candidate"),
        "evidence_refs": item.get("evidence_refs") or [],
        "plan_only": True,
        "write_permission": False,
        "apply_executed": False,
        "durable_writes_executed": False,
        "graph_truth_written": False,
        "generated_at": generated_at,
    }


def build_s1_overlay_row(row: dict[str, Any]) -> dict[str, Any] | None:
    action = row["internal_review_action"]
    if row["layer"] != "s1":
        return None
    if action in {"reject_patch", "defer", "needs_more_evidence", "accept_noop"}:
        return None
    return {
        "schema_version": "maintenance.s1_latest_view_overlay_from_review.v0.4",
        "review_item_id": row["review_item_id"],
        "patch_id": row["patch_id"],
        "operation_id": row["operation_id"],
        "overlay_effect": row["apply_effect"],
        "current_candidate_ids": row["candidate_context_ids"] if action in {"accept_patch", "accept_with_edit", "split_current_vs_historical"} else [],
        "historical_or_stale_candidate_ids": row["existing_context_ids"] if action in {"split_current_vs_historical", "mark_stale_only"} else [],
        "review_status": row["review_status"],
        "plan_only": True,
        "write_permission": False,
        "apply_executed": False,
        "generated_at": row["generated_at"],
    }


def build_s2_refresh_scope_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row["layer"] != "s2" and row["internal_review_action"] != "route_s2_build":
        return None
    if row["internal_review_action"] in {"reject_patch", "defer", "needs_more_evidence", "accept_noop"}:
        return None
    return {
        "schema_version": "maintenance.s2_refresh_scope_from_review.v0.4",
        "review_item_id": row["review_item_id"],
        "patch_id": row["patch_id"],
        "operation_id": row["operation_id"],
        "refresh_reason": row["apply_effect"],
        "source_candidate_ids": row["candidate_context_ids"],
        "existing_context_ids": row["existing_context_ids"],
        "plan_only": True,
        "write_permission": False,
        "apply_executed": False,
        "generated_at": row["generated_at"],
    }


def build_graph_refresh_scope_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if row["layer"] != "graph" and row["internal_review_action"] != "route_graph_extraction":
        return None
    if row["internal_review_action"] in {"reject_patch", "defer", "needs_more_evidence", "accept_noop"}:
        return None
    return {
        "schema_version": "maintenance.graph_refresh_scope_from_review.v0.4",
        "review_item_id": row["review_item_id"],
        "patch_id": row["patch_id"],
        "operation_id": row["operation_id"],
        "refresh_reason": row["apply_effect"],
        "source_candidate_ids": row["candidate_context_ids"],
        "existing_context_ids": row["existing_context_ids"],
        "graph_is_not_proof": True,
        "plan_only": True,
        "write_permission": False,
        "apply_executed": False,
        "graph_truth_written": False,
        "generated_at": row["generated_at"],
    }


def build_report_markdown(report: dict[str, Any], paths: dict[str, str] | None = None) -> str:
    paths = paths or {}
    lines = [
        "# v0.4 Incremental Review Apply Plan Report",
        "",
        f"- generated_at: {report['generated_at']}",
        f"- review_session_status: {report['review_session_status']}",
        f"- queue_item_count: {report['queue_item_count']}",
        f"- decision_count: {report['decision_count']}",
        f"- undecided_count: {report['undecided_count']}",
        f"- apply_plan_rows: {report['apply_plan_count']}",
        f"- s1_overlay_rows: {report['s1_overlay_count']}",
        f"- s2_refresh_scope_rows: {report['s2_refresh_scope_count']}",
        f"- graph_refresh_scope_rows: {report['graph_refresh_scope_count']}",
        "",
        "## Boundary",
        "",
        "This runner is plan-only. It does not write canonical memory, durable memory, S2 portrait truth, or graph truth.",
        "",
        "## Action Counts",
        "",
    ]
    for action, count in sorted(report["internal_action_counts"].items()):
        lines.append(f"- {action}: {count}")
    if paths:
        lines.extend(["", "## Artifacts", ""])
        for name, path in sorted(paths.items()):
            lines.append(f"- {name}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def build_incremental_review_apply_bundle(
    *,
    review_queue_items: list[dict[str, Any]],
    review_decisions: list[dict[str, Any]],
    review_session_manifest: dict[str, Any],
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_generated_at = generated_at or now_iso()
    validation = validate_finalized_session(
        queue_items=review_queue_items,
        decisions=review_decisions,
        session_manifest=review_session_manifest,
    )
    latest = validation["latest_decisions"]
    by_id = queue_by_id(review_queue_items)
    apply_rows: list[dict[str, Any]] = []
    for item_id in validation["decided_ids"]:
        item = by_id[item_id]
        decision = latest[item_id]
        if decision.get("patch_hash") != item.get("patch_hash"):
            raise ValueError(f"patch_hash_mismatch:{item_id}")
        if decision.get("source_context_hash") != item.get("source_context_hash"):
            raise ValueError(f"source_context_hash_mismatch:{item_id}")
        apply_rows.append(build_apply_plan_row(item=item, decision=decision, generated_at=resolved_generated_at))

    s1_overlay_rows = [row for row in (build_s1_overlay_row(item) for item in apply_rows) if row]
    s2_refresh_scope_rows = [row for row in (build_s2_refresh_scope_row(item) for item in apply_rows) if row]
    graph_refresh_scope_rows = [row for row in (build_graph_refresh_scope_row(item) for item in apply_rows) if row]
    action_counts = Counter(row["internal_review_action"] for row in apply_rows)
    status_counts = Counter(row["review_status"] for row in apply_rows)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "review_session_status": review_session_manifest.get("review_session_status"),
        "queue_item_count": len(validation["queue_ids"]),
        "decision_count": len(validation["decided_ids"]),
        "undecided_count": len(validation["undecided_ids"]),
        "apply_plan_count": len(apply_rows),
        "s1_overlay_count": len(s1_overlay_rows),
        "s2_refresh_scope_count": len(s2_refresh_scope_rows),
        "graph_refresh_scope_count": len(graph_refresh_scope_rows),
        "internal_action_counts": dict(sorted(action_counts.items())),
        "review_status_counts": dict(sorted(status_counts.items())),
        "queue_hash": review_session_manifest.get("queue_hash"),
        "decisions_hash": review_session_manifest.get("decisions_hash"),
        "boundary": "plan_only_no_canonical_or_durable_writes",
        "write_permission": False,
        "apply_executed": False,
        "durable_writes_executed": False,
        "graph_truth_written": False,
        "generated_at": resolved_generated_at,
    }
    return {
        "apply_plan_rows": apply_rows,
        "s1_overlay_rows": s1_overlay_rows,
        "s2_refresh_scope_rows": s2_refresh_scope_rows,
        "graph_refresh_scope_rows": graph_refresh_scope_rows,
        "report": report,
    }


def write_apply_bundle(output_dir: Path, bundle: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "apply_plan": str(output_dir / APPLY_PLAN_FILENAME),
        "s1_overlay": str(output_dir / S1_OVERLAY_FILENAME),
        "s2_refresh_scope": str(output_dir / S2_REFRESH_SCOPE_FILENAME),
        "graph_refresh_scope": str(output_dir / GRAPH_REFRESH_SCOPE_FILENAME),
        "report_json": str(output_dir / APPLY_REPORT_JSON_FILENAME),
        "report_md": str(output_dir / APPLY_REPORT_FILENAME),
    }
    write_jsonl(Path(paths["apply_plan"]), bundle["apply_plan_rows"])
    write_jsonl(Path(paths["s1_overlay"]), bundle["s1_overlay_rows"])
    write_jsonl(Path(paths["s2_refresh_scope"]), bundle["s2_refresh_scope_rows"])
    write_jsonl(Path(paths["graph_refresh_scope"]), bundle["graph_refresh_scope_rows"])
    report_with_paths = {**bundle["report"], "paths": paths}
    write_json(Path(paths["report_json"]), report_with_paths)
    write_text(Path(paths["report_md"]), build_report_markdown(report_with_paths, paths))
    return paths


def build_incremental_review_apply_bundle_from_files(
    *,
    review_queue_jsonl: Path,
    review_decisions_jsonl: Path,
    review_session_manifest_json: Path,
    output_dir: Path,
) -> dict[str, Any]:
    bundle = build_incremental_review_apply_bundle(
        review_queue_items=read_jsonl(review_queue_jsonl),
        review_decisions=read_jsonl(review_decisions_jsonl),
        review_session_manifest=read_json(review_session_manifest_json),
    )
    bundle["paths"] = write_apply_bundle(output_dir, bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-queue-jsonl", required=True, type=Path)
    parser.add_argument("--review-decisions-jsonl", required=True, type=Path)
    parser.add_argument("--review-session-manifest-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    bundle = build_incremental_review_apply_bundle_from_files(
        review_queue_jsonl=args.review_queue_jsonl,
        review_decisions_jsonl=args.review_decisions_jsonl,
        review_session_manifest_json=args.review_session_manifest_json,
        output_dir=args.output_dir,
    )
    print(json.dumps(bundle["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
