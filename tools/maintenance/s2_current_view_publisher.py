"""Publish reviewed S2 incremental candidates as the current S2 view.

S2 history remains append-only. This publisher overlays reviewed incremental
portrait candidates onto an optional base reviewed-units file, tags excluded or
historical rows, and writes the default ``maintenance/latest_views/s2_latest_view.jsonl``
consumed by S2 query/index runners.

By default, unreviewed S2 candidates are not published. A bounded experiment
may pass ``--allow-unreviewed-experiment`` explicitly.
"""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.graph.graph_construction_packet_builder import file_hash, stable_id


SCHEMA_VERSION = "maintenance.s2_current_view_publish.v0.4"
TRANSITION_SCHEMA_VERSION = "maintenance.s2_status_transition.v0.4"
DEFAULT_CURRENT_DIR_NAME = "s2_current"
DEFAULT_LATEST_VIEW_DIR = Path("maintenance") / "latest_views"

CURRENT_FILENAME = "s2_current.jsonl"
EXCLUDED_FILENAME = "s2_current_excluded.jsonl"
TRANSITIONS_FILENAME = "s2_status_transitions.jsonl"
MANIFEST_FILENAME = "s2_current_manifest.json"
REPORT_FILENAME = "s2_current_report.md"

ACTIVE_BASE_STATUSES = {"active", "accepted", "accepted_for_experiment", "accepted_by_user", "current"}
INACTIVE_STATUSES = {"rejected", "archived", "superseded", "deprecated", "historical_only", "inactive", "stale"}
ACCEPTED_REVIEW_STATUSES = {"approved_for_apply", "approved_with_edit", "approved_noop"}
FAILED_REVIEW_STATUSES = {"rejected", "deferred", "needs_more_evidence"}
PUBLISHABLE_CANDIDATE_STATUSES = {"candidate_ready_for_review"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def normalize(value: Any) -> str:
    return str(value or "").strip()


def normalize_unique(values: Iterable[Any] | None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values or []:
        text = normalize(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
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
            raise ValueError(f"S2 current input row must be object at {path}:{line_number}")
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


def unit_id(row: dict[str, Any]) -> str:
    for field in ("unit_id", "id", "object_id"):
        text = normalize(row.get(field))
        if text:
            return text
    return ""


def decisions_by_review_item(decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in decisions:
        item_id = normalize(row.get("review_item_id"))
        if item_id:
            latest[item_id] = row
    return latest


def affected_by_s1_memory(affected_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    lookup: dict[str, list[dict[str, Any]]] = {}
    for row in affected_rows:
        memory_id = normalize(row.get("new_s1_unit_id"))
        if memory_id:
            lookup.setdefault(memory_id, []).append(row)
    return lookup


def status_for_base(row: dict[str, Any], stale_unit_ids: set[str]) -> tuple[str, bool, str]:
    item_id = unit_id(row)
    if item_id in stale_unit_ids:
        return "historical_or_stale", False, "directly_affected_by_reviewed_s2_increment"
    status = normalize(row.get("current_view_status") or row.get("status")).lower()
    if status in INACTIVE_STATUSES:
        return status, False, f"inactive_base_status:{status}"
    return "active_current_existing", True, "kept_from_base_s2_current"


def candidate_review_status(
    candidate: dict[str, Any],
    decisions: dict[str, dict[str, Any]],
    *,
    allow_unreviewed_experiment: bool,
) -> tuple[str, str, dict[str, Any] | None]:
    item_id = normalize(candidate.get("review_item_id"))
    decision = decisions.get(item_id)
    if decision:
        status = normalize(decision.get("review_status"))
        if status in ACCEPTED_REVIEW_STATUSES:
            return "accepted_by_s2_review", "accepted_review_decision", decision
        if status in FAILED_REVIEW_STATUSES:
            return "excluded_by_s2_review", f"failed_review_status:{status}", decision
        return "excluded_by_s2_review", f"unsupported_review_status:{status}", decision
    candidate_status = normalize(candidate.get("candidate_status"))
    if allow_unreviewed_experiment and candidate_status in PUBLISHABLE_CANDIDATE_STATUSES:
        return "accepted_for_experiment", "explicit_allow_unreviewed_experiment", None
    return "excluded_pending_s2_review", f"missing_s2_review_decision:{candidate_status or 'unknown'}", None


def candidate_content(candidate: dict[str, Any]) -> str:
    return (
        normalize(candidate.get("candidate_text"))
        or normalize(candidate.get("fact_candidate_text"))
        or normalize(candidate.get("hypothesis_text"))
        or normalize((candidate.get("proposal_row") or {}).get("candidate_text"))
    )


def candidate_to_reviewed_unit(
    candidate: dict[str, Any],
    *,
    current_status: str,
    review_reason: str,
    decision: dict[str, Any] | None,
    affected_rows: list[dict[str, Any]],
    published_at: str,
) -> dict[str, Any]:
    content = candidate_content(candidate)
    proposal_id = normalize(candidate.get("proposal_id"))
    source_s1_memory_id = normalize(candidate.get("source_s1_memory_id"))
    generated_unit_id = stable_id("s2unit:v04", "|".join([proposal_id, source_s1_memory_id, content]))
    proposal_row = candidate.get("proposal_row") if isinstance(candidate.get("proposal_row"), dict) else {}
    evidence_refs = normalize_unique(candidate.get("evidence_refs") or proposal_row.get("evidence_refs") or [])
    raw_refs = candidate.get("raw_backpointer_refs") or proposal_row.get("raw_backpointer_refs") or []
    source_refs = normalize_unique(candidate.get("source_refs") or proposal_row.get("source_refs") or [])
    affected_unit_ids = normalize_unique(row.get("affected_s2_unit_id") for row in affected_rows)
    subject_scope_ids = normalize_unique(ref for row in affected_rows for ref in (row.get("subject_scope_existing_s2_unit_ids") or []))
    warnings = normalize_unique(
        [
            *(candidate.get("warnings") or []),
            "s2_current_candidate_not_durable_memory",
            *(
                ["s2_old_unit_not_directly_resolved_subject_scope_only"]
                if subject_scope_ids and not affected_unit_ids
                else []
            ),
        ]
    )
    return {
        "schema_version": "s2.reviewed_portrait_unit.v1",
        "unit_id": generated_unit_id,
        "user_id": proposal_row.get("target_participant") or proposal_row.get("subject_id") or "",
        "type": candidate.get("candidate_type") or "unknown",
        "memory_class": proposal_row.get("memory_class") or "semantic",
        "content": content,
        "scope": proposal_row.get("scope_hint") or "unknown",
        "source_refs": source_refs,
        "evidence_refs": evidence_refs,
        "backpointer_refs": raw_refs,
        "raw_backpointer_refs": raw_refs,
        "evidence_summary": candidate.get("source_text_quote") or candidate.get("source_text_preview") or content,
        "confidence": candidate.get("proposal_confidence") or proposal_row.get("proposal_confidence") or "unknown",
        "inference_level": candidate.get("inference_level") or proposal_row.get("inference_level") or "",
        "status": "active" if current_status in {"accepted_by_s2_review", "accepted_for_experiment"} else "excluded",
        "current_view_status": current_status,
        "participates_in_default_query": current_status in {"accepted_by_s2_review", "accepted_for_experiment"},
        "privacy_class": proposal_row.get("privacy_class") or "",
        "review_metadata": {
            "review_status": (decision or {}).get("review_status") or current_status,
            "review_action": (decision or {}).get("internal_review_action") or (decision or {}).get("review_action") or "",
            "review_reason": review_reason,
            "review_item_id": candidate.get("review_item_id") or "",
        },
        "proposal_origin": {
            "proposal_id": proposal_id,
            "proposal_input_id": candidate.get("proposal_input_id") or "",
            "proposal_run_id": candidate.get("proposal_run_id") or "",
            "provider": candidate.get("provider") or "",
            "model_id": candidate.get("model_id") or "",
            "output_kind": candidate.get("output_kind") or "",
            "route_used": candidate.get("route_used") or "",
        },
        "step1_origin": {
            "input_layer": "s2_incremental_candidate",
            "input_refs": [proposal_id] if proposal_id else [],
            "source_s1_memory_id": source_s1_memory_id,
            "refresh_reason": candidate.get("refresh_reason") or "",
        },
        "s2_incremental_origin": {
            "candidate_status": candidate.get("candidate_status") or "",
            "patch_id": candidate.get("patch_id") or "",
            "operation_id": candidate.get("operation_id") or "",
            "route_decision_id": candidate.get("route_decision_id") or "",
            "affected_s2_unit_ids": affected_unit_ids,
            "subject_scope_existing_s2_unit_ids": subject_scope_ids,
        },
        "warnings": warnings,
        "_maintenance_s2_current": {
            "schema_version": SCHEMA_VERSION,
            "source": "incremental_candidate",
            "review_reason": review_reason,
            "published_at": published_at,
            "write_permission": False,
        },
    }


def annotate_base_row(
    row: dict[str, Any],
    *,
    current_status: str,
    participates: bool,
    reason: str,
    published_at: str,
) -> dict[str, Any]:
    out = deepcopy(row)
    out["current_view_status"] = current_status
    out["participates_in_default_query"] = participates
    if not participates and current_status == "historical_or_stale":
        out["status"] = "historical_only"
    out["_maintenance_s2_current"] = {
        "schema_version": SCHEMA_VERSION,
        "source": "base_reviewed_unit",
        "reason": reason,
        "published_at": published_at,
        "write_permission": False,
    }
    return out


def transition_row(
    *,
    object_id: str,
    source_bucket: str,
    current_status: str,
    participates: bool,
    reason: str,
    published_at: str,
    refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": TRANSITION_SCHEMA_VERSION,
        "layer": "s2",
        "unit_id": object_id,
        "source_bucket": source_bucket,
        "current_view_status": current_status,
        "transition_reason": reason,
        "participates_in_default_query": participates,
        "refs": refs or [],
        "write_permission": False,
        "durable_writes_executed": False,
        "current_portrait_written": False,
        "published_at": published_at,
    }


def stale_base_unit_ids_from_affected(affected_rows: list[dict[str, Any]]) -> set[str]:
    stale: set[str] = set()
    for row in affected_rows:
        signal = normalize(row.get("s2_delta_signal"))
        direct = normalize(row.get("affected_s2_unit_id"))
        if direct and signal in {"contradicts_profile", "supersedes_profile", "updates_profile", "historical_only"}:
            stale.add(direct)
    return stale


def build_s2_current_rows(
    *,
    base_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    review_decisions: list[dict[str, Any]],
    affected_rows: list[dict[str, Any]],
    allow_unreviewed_experiment: bool,
    published_at: str,
) -> dict[str, Any]:
    decisions = decisions_by_review_item(review_decisions)
    affected_lookup = affected_by_s1_memory(affected_rows)
    stale_ids = stale_base_unit_ids_from_affected(affected_rows)
    active: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    for row in base_rows:
        status, participates, reason = status_for_base(row, stale_ids)
        annotated = annotate_base_row(row, current_status=status, participates=participates, reason=reason, published_at=published_at)
        transitions.append(
            transition_row(
                object_id=unit_id(row),
                source_bucket="base",
                current_status=status,
                participates=participates,
                reason=reason,
                published_at=published_at,
            )
        )
        (active if participates else excluded).append(annotated)

    for candidate in candidate_rows:
        status, reason, decision = candidate_review_status(
            candidate,
            decisions,
            allow_unreviewed_experiment=allow_unreviewed_experiment,
        )
        source_s1 = normalize(candidate.get("source_s1_memory_id"))
        unit = candidate_to_reviewed_unit(
            candidate,
            current_status=status,
            review_reason=reason,
            decision=decision,
            affected_rows=affected_lookup.get(source_s1, []),
            published_at=published_at,
        )
        participates = bool(unit.get("participates_in_default_query"))
        transitions.append(
            transition_row(
                object_id=unit["unit_id"],
                source_bucket="incremental_candidate",
                current_status=status,
                participates=participates,
                reason=reason,
                published_at=published_at,
                refs=normalize_unique([candidate.get("proposal_id"), candidate.get("source_s1_memory_id"), candidate.get("review_item_id")]),
            )
        )
        (active if participates else excluded).append(unit)

    return {"active": active, "excluded": excluded, "transitions": transitions}


def archive_previous_current(current_dir: Path, history_dir: Path, slug: str) -> Path | None:
    if not current_dir.exists():
        return None
    history_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = history_dir / f"s2_current_previous_{slug}"
    shutil.copytree(current_dir, archive_dir)
    return archive_dir


def write_aliases(workspace: Path, current_dir: Path) -> dict[str, str]:
    latest_dir = workspace / DEFAULT_LATEST_VIEW_DIR
    latest_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        "s2_latest_view": latest_dir / "s2_latest_view.jsonl",
        "s2_latest_view_excluded": latest_dir / "s2_latest_view_excluded.jsonl",
        "s2_status_transitions": latest_dir / TRANSITIONS_FILENAME,
    }
    shutil.copy2(current_dir / CURRENT_FILENAME, aliases["s2_latest_view"])
    shutil.copy2(current_dir / EXCLUDED_FILENAME, aliases["s2_latest_view_excluded"])
    shutil.copy2(current_dir / TRANSITIONS_FILENAME, aliases["s2_status_transitions"])
    return {key: str(path) for key, path in aliases.items()}


def status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = normalize(row.get("current_view_status"))
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.4 S2 Current View Publish",
        "",
        f"- published_at: {manifest['published_at']}",
        f"- workspace: `{manifest['workspace']}`",
        f"- base_reviewed_units: `{manifest['inputs']['base_reviewed_units_jsonl']}`",
        f"- candidates: `{manifest['inputs']['candidate_outputs_jsonl']}`",
        f"- current_dir: `{manifest['current_dir']}`",
        f"- previous_current_archive_dir: `{manifest.get('previous_current_archive_dir') or ''}`",
        "",
        "## Counts",
        "",
        f"- current_active_count: {manifest['counts']['current_active_count']}",
        f"- current_excluded_count: {manifest['counts']['current_excluded_count']}",
        f"- status_transition_count: {manifest['counts']['status_transition_count']}",
        "",
        "## Status Counts",
        "",
    ]
    for status, count in sorted(manifest["counts"]["current_view_status_counts"].items()):
        lines.append(f"- {status}: {count}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- S2 history remains append-only.",
            "- This publishes a machine-consumed current S2 view for query/index consumers.",
            "- It does not write durable memory, canonical `portrait/reviewed_units.jsonl`, or `current_portrait.json`.",
            "- Unreviewed candidates require explicit `allow_unreviewed_experiment=true`.",
            "",
        ]
    )
    return "\n".join(lines)


def run_s2_current_view_publish(
    *,
    workspace: Path,
    candidate_outputs_jsonl: Path,
    base_reviewed_units_jsonl: Path | None = None,
    review_decisions_jsonl: Path | None = None,
    affected_s2_jsonl: Path | None = None,
    current_dir: Path | None = None,
    history_dir: Path | None = None,
    allow_unreviewed_experiment: bool = False,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    candidate_outputs_jsonl = candidate_outputs_jsonl.resolve()
    base_reviewed_units_jsonl = base_reviewed_units_jsonl.resolve() if base_reviewed_units_jsonl else None
    review_decisions_jsonl = review_decisions_jsonl.resolve() if review_decisions_jsonl else None
    affected_s2_jsonl = affected_s2_jsonl.resolve() if affected_s2_jsonl else None
    current_dir = (current_dir or workspace / DEFAULT_CURRENT_DIR_NAME).resolve()
    history_dir = (history_dir or workspace / "maintenance" / "published_views").resolve()
    if not candidate_outputs_jsonl.exists():
        raise FileNotFoundError(f"S2 candidate outputs not found: {candidate_outputs_jsonl}")

    published_at = now_iso()
    rows = build_s2_current_rows(
        base_rows=read_jsonl(base_reviewed_units_jsonl),
        candidate_rows=read_jsonl(candidate_outputs_jsonl),
        review_decisions=read_jsonl(review_decisions_jsonl),
        affected_rows=read_jsonl(affected_s2_jsonl),
        allow_unreviewed_experiment=allow_unreviewed_experiment,
        published_at=published_at,
    )

    slug = timestamp_slug()
    temp_dir = workspace / f".s2_current_tmp_{slug}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    write_jsonl(temp_dir / CURRENT_FILENAME, rows["active"])
    write_jsonl(temp_dir / EXCLUDED_FILENAME, rows["excluded"])
    write_jsonl(temp_dir / TRANSITIONS_FILENAME, rows["transitions"])

    previous_manifest_hash = file_hash(current_dir / MANIFEST_FILENAME)
    previous_archive = archive_previous_current(current_dir, history_dir, slug)
    if current_dir.exists():
        shutil.rmtree(current_dir)
    shutil.move(str(temp_dir), str(current_dir))
    alias_paths = write_aliases(workspace, current_dir)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "published_at": published_at,
        "workspace": str(workspace),
        "inputs": {
            "base_reviewed_units_jsonl": str(base_reviewed_units_jsonl) if base_reviewed_units_jsonl else "",
            "candidate_outputs_jsonl": str(candidate_outputs_jsonl),
            "review_decisions_jsonl": str(review_decisions_jsonl) if review_decisions_jsonl else "",
            "affected_s2_jsonl": str(affected_s2_jsonl) if affected_s2_jsonl else "",
        },
        "input_hashes": {
            "base_reviewed_units_jsonl": file_hash(base_reviewed_units_jsonl) if base_reviewed_units_jsonl else None,
            "candidate_outputs_jsonl": file_hash(candidate_outputs_jsonl),
            "review_decisions_jsonl": file_hash(review_decisions_jsonl) if review_decisions_jsonl else None,
            "affected_s2_jsonl": file_hash(affected_s2_jsonl) if affected_s2_jsonl else None,
            "previous_current_manifest": previous_manifest_hash,
        },
        "current_dir": str(current_dir),
        "previous_current_archive_dir": str(previous_archive) if previous_archive else "",
        "history_dir": str(history_dir),
        "counts": {
            "base_input_count": len(read_jsonl(base_reviewed_units_jsonl)),
            "candidate_input_count": len(read_jsonl(candidate_outputs_jsonl)),
            "review_decision_count": len(read_jsonl(review_decisions_jsonl)),
            "affected_s2_count": len(read_jsonl(affected_s2_jsonl)),
            "current_active_count": len(rows["active"]),
            "current_excluded_count": len(rows["excluded"]),
            "status_transition_count": len(rows["transitions"]),
            "current_view_status_counts": status_counts(rows["active"] + rows["excluded"]),
        },
        "paths": {
            "s2_current": str(current_dir / CURRENT_FILENAME),
            "s2_current_excluded": str(current_dir / EXCLUDED_FILENAME),
            "s2_status_transitions": str(current_dir / TRANSITIONS_FILENAME),
            "manifest": str(current_dir / MANIFEST_FILENAME),
            "report": str(current_dir / REPORT_FILENAME),
            **alias_paths,
        },
        "policies": {
            "history_append_only": True,
            "current_view_publish_executed": True,
            "query_default_ready": True,
            "stable_current_dir": DEFAULT_CURRENT_DIR_NAME,
            "default_latest_view_alias_written": True,
            "allow_unreviewed_experiment": allow_unreviewed_experiment,
            "durable_memory_written": False,
            "canonical_reviewed_units_rewritten": False,
            "current_portrait_written": False,
            "graph_truth_written": False,
        },
        "boundary": "published_current_s2_view_no_durable_or_canonical_history_write",
        "write_permission": False,
        "publish_executed": True,
        "query_default_ready": True,
        "durable_writes_executed": False,
        "canonical_reviewed_units_rewritten": False,
        "current_portrait_written": False,
        "graph_truth_written": False,
    }
    write_json(current_dir / MANIFEST_FILENAME, manifest)
    write_text(current_dir / REPORT_FILENAME, render_report(manifest))
    write_json(workspace / DEFAULT_LATEST_VIEW_DIR / "s2_current_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--candidate-outputs-jsonl", required=True, type=Path)
    parser.add_argument("--base-reviewed-units-jsonl", type=Path)
    parser.add_argument("--review-decisions-jsonl", type=Path)
    parser.add_argument("--affected-s2-jsonl", type=Path)
    parser.add_argument("--current-dir", type=Path)
    parser.add_argument("--history-dir", type=Path)
    parser.add_argument("--allow-unreviewed-experiment", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_s2_current_view_publish(
        workspace=args.workspace,
        candidate_outputs_jsonl=args.candidate_outputs_jsonl,
        base_reviewed_units_jsonl=args.base_reviewed_units_jsonl,
        review_decisions_jsonl=args.review_decisions_jsonl,
        affected_s2_jsonl=args.affected_s2_jsonl,
        current_dir=args.current_dir,
        history_dir=args.history_dir,
        allow_unreviewed_experiment=args.allow_unreviewed_experiment,
    )
    print(
        json.dumps(
            {
                "manifest": manifest["paths"]["manifest"],
                "current_dir": manifest["current_dir"],
                "counts": manifest["counts"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
