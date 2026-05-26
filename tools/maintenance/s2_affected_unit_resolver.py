"""Resolve S2 units affected by S1 delta decisions.

This v0.4 helper consumes S1 delta decisions and an active S2 latest view. It
emits a read-only affected-unit report and optional S2 index invalidation scope.
It does not rebuild S2, publish a current portrait, or mutate indexes.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "maintenance.s2_affected_unit.v0.4"
REPORT_SCHEMA_VERSION = "maintenance.s2_affected_unit_report.v0.4"

S2_DELTA_OUTPUTS = {
    "new_profile_unit",
    "duplicate_candidate",
    "refines_profile",
    "updates_profile",
    "contradicts_profile",
    "supersedes_profile",
    "historical_only",
    "needs_review",
    "no_material_change",
}

REQUIRES_S2_REFRESH = {
    "duplicate_candidate",
    "strengthens",
    "weakens",
    "contradicts",
    "supersedes",
    "needs_review",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
            raise ValueError(f"S2 affected resolver input row must be an object at {path}:{line_number}")
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


def normalize(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def normalize_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def list_refs(row: dict[str, Any], *field_names: str) -> list[str]:
    refs: list[str] = []
    for field_name in field_names:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for key in ("evidence_ref", "display_ref", "canonical_evidence_ref", "source_id", "id"):
                        if item.get(key):
                            refs.append(str(item[key]))
                            break
                else:
                    refs.append(str(item))
        elif isinstance(value, dict):
            for key in ("evidence_ref", "display_ref", "canonical_evidence_ref", "source_id", "id"):
                if value.get(key):
                    refs.append(str(value[key]))
                    break
        else:
            refs.append(str(value))
    temporal_scope = row.get("temporal_scope") or {}
    if isinstance(temporal_scope, dict):
        for observed in temporal_scope.get("observed_at_refs") or []:
            if isinstance(observed, dict):
                refs.append(normalize(observed.get("evidence_ref") or observed.get("display_ref")))
    for quote in row.get("evidence_quotes") or []:
        if isinstance(quote, dict):
            refs.append(normalize(quote.get("evidence_ref")))
    return normalize_unique(refs)


def s2_unit_id(row: dict[str, Any]) -> str:
    for field_name in ("unit_id", "id", "object_id"):
        value = row.get(field_name)
        if value:
            return str(value)
    return ""


def s2_subject_id(row: dict[str, Any]) -> str:
    for field_name in ("user_id", "subject_id", "target_subject_id", "modeled_user_id"):
        value = row.get(field_name)
        if value:
            return str(value)
    return ""


def index_entry_id(row: dict[str, Any]) -> str:
    for field_name in ("vector_entry_id", "index_entry_id", "object_id", "id"):
        value = row.get(field_name)
        if value:
            return str(value)
    return ""


def s2_delta_signal(s1_delta_type: str, has_direct_s2_hit: bool) -> tuple[str, str, str]:
    if s1_delta_type == "no_material_change":
        return "no_material_change", "no_s2_refresh_expected", "current_view_unchanged"
    if s1_delta_type == "new_unit":
        if has_direct_s2_hit:
            return "updates_profile", "route_s2_for_possible_update", "current_view_refresh_candidate"
        return "new_profile_unit", "route_s2_for_new_s1_unit", "current_view_add_candidate"
    if s1_delta_type == "duplicate_candidate":
        return "duplicate_candidate", "review_s2_duplicate_or_noop", "current_view_review_candidate"
    if s1_delta_type == "strengthens":
        return "refines_profile", "route_s2_for_refinement", "current_view_refresh_candidate"
    if s1_delta_type == "weakens":
        return "updates_profile", "route_s2_for_weakening_or_historical_note", "current_view_refresh_candidate"
    if s1_delta_type == "contradicts":
        return "contradicts_profile", "route_s2_for_contradiction_review", "current_view_refresh_candidate"
    if s1_delta_type == "supersedes":
        return "supersedes_profile", "mark_old_s2_unit_stale_candidate", "current_view_refresh_candidate"
    if s1_delta_type == "needs_review":
        return "needs_review", "route_s2_after_s1_delta_review", "current_view_review_candidate"
    return "needs_review", "unknown_s1_delta_type_review_required", "current_view_review_candidate"


def build_s2_index_lookup(s2_index_rows: list[dict[str, Any]]) -> dict[str, set[str]]:
    lookup: dict[str, set[str]] = {}
    for row in s2_index_rows:
        entry_id = index_entry_id(row)
        if not entry_id:
            continue
        for ref in normalize_unique([normalize(row.get("object_id")), *list_refs(row, "evidence_refs", "source_refs")]):
            lookup.setdefault(ref, set()).add(entry_id)
    return lookup


def resolve_s2_index_entries(
    *,
    unit_id: str,
    evidence_refs: list[str],
    index_lookup: dict[str, set[str]],
) -> list[str]:
    resolved: set[str] = set()
    resolved.update(index_lookup.get(unit_id, set()))
    for evidence_ref in evidence_refs:
        resolved.update(index_lookup.get(evidence_ref, set()))
    return sorted(resolved)


def build_affected_rows_for_decision(
    decision: dict[str, Any],
    *,
    s2_rows: list[dict[str, Any]],
    index_lookup: dict[str, set[str]],
    generated_at: str,
) -> list[dict[str, Any]]:
    s1_delta_type = normalize(decision.get("delta_type"))
    new_s1_unit_id = normalize(decision.get("new_s1_unit_id"))
    new_subject = normalize(decision.get("new_subject_id"))
    new_evidence_refs = list_refs(decision, "new_evidence_refs")
    new_source_refs = list_refs(decision, "new_source_refs")

    direct_hits: list[tuple[dict[str, Any], list[str]]] = []
    subject_hits: list[dict[str, Any]] = []
    for s2_row in s2_rows:
        unit_evidence_refs = list_refs(s2_row, "evidence_refs", "backpointer_refs")
        overlapping_evidence = sorted(set(new_evidence_refs) & set(unit_evidence_refs))
        if overlapping_evidence:
            direct_hits.append((s2_row, overlapping_evidence))
        elif new_subject and s2_subject_id(s2_row) == new_subject:
            subject_hits.append(s2_row)

    rows: list[dict[str, Any]] = []
    for s2_row, overlapping_evidence in direct_hits:
        unit_id = s2_unit_id(s2_row)
        s2_signal, recommended_action, current_policy = s2_delta_signal(s1_delta_type, True)
        evidence_refs = list_refs(s2_row, "evidence_refs", "backpointer_refs")
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "operation_id": normalize(decision.get("operation_id")),
                "new_s1_unit_id": new_s1_unit_id,
                "s1_delta_type": s1_delta_type,
                "new_evidence_refs": new_evidence_refs,
                "new_source_refs": new_source_refs,
                "new_subject_id": new_subject,
                "affected_s2_unit_id": unit_id,
                "affected_s2_subject_id": s2_subject_id(s2_row),
                "affected_s2_type": normalize(s2_row.get("type")),
                "affected_s2_scope": normalize(s2_row.get("scope")),
                "match_type": "evidence_overlap",
                "matched_evidence_refs": overlapping_evidence,
                "s2_delta_signal": s2_signal,
                "recommended_action": recommended_action,
                "current_vs_historical_policy": current_policy,
                "s2_index_entry_ids": resolve_s2_index_entries(
                    unit_id=unit_id,
                    evidence_refs=evidence_refs,
                    index_lookup=index_lookup,
                ),
                "s2_excerpt": normalize(s2_row.get("content"))[:500],
                "read_only": True,
                "write_permission": False,
                "generated_at": generated_at,
            }
        )

    if not direct_hits and s1_delta_type == "new_unit":
        s2_signal, recommended_action, current_policy = s2_delta_signal(s1_delta_type, False)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "operation_id": normalize(decision.get("operation_id")),
                "new_s1_unit_id": new_s1_unit_id,
                "s1_delta_type": s1_delta_type,
                "new_evidence_refs": new_evidence_refs,
                "new_source_refs": new_source_refs,
                "new_subject_id": new_subject,
                "affected_s2_unit_id": "",
                "affected_s2_subject_id": new_subject,
                "affected_s2_type": "",
                "affected_s2_scope": "",
                "match_type": "subject_scope_candidate" if subject_hits else "no_existing_s2_subject_scope",
                "matched_evidence_refs": [],
                "subject_scope_existing_s2_unit_ids": [s2_unit_id(row) for row in subject_hits if s2_unit_id(row)],
                "s2_delta_signal": s2_signal,
                "recommended_action": recommended_action,
                "current_vs_historical_policy": current_policy,
                "s2_index_entry_ids": [],
                "s2_excerpt": "",
                "read_only": True,
                "write_permission": False,
                "generated_at": generated_at,
            }
        )

    if not rows and s1_delta_type in REQUIRES_S2_REFRESH:
        s2_signal, recommended_action, current_policy = s2_delta_signal(s1_delta_type, False)
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "operation_id": normalize(decision.get("operation_id")),
                "new_s1_unit_id": new_s1_unit_id,
                "s1_delta_type": s1_delta_type,
                "new_evidence_refs": new_evidence_refs,
                "new_source_refs": new_source_refs,
                "new_subject_id": new_subject,
                "affected_s2_unit_id": "",
                "affected_s2_subject_id": new_subject,
                "affected_s2_type": "",
                "affected_s2_scope": "",
                "match_type": "s1_delta_requires_s2_review_but_no_direct_s2_hit",
                "matched_evidence_refs": [],
                "subject_scope_existing_s2_unit_ids": [s2_unit_id(row) for row in subject_hits if s2_unit_id(row)],
                "s2_delta_signal": s2_signal,
                "recommended_action": recommended_action,
                "current_vs_historical_policy": current_policy,
                "s2_index_entry_ids": [],
                "s2_excerpt": "",
                "read_only": True,
                "write_permission": False,
                "generated_at": generated_at,
            }
        )
    return rows


def build_s2_affected_unit_report(
    *,
    s1_delta_decisions: list[dict[str, Any]],
    active_s2_rows: list[dict[str, Any]],
    s2_index_rows: list[dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_generated_at = generated_at or now_iso()
    index_lookup = build_s2_index_lookup(s2_index_rows or [])
    affected_rows: list[dict[str, Any]] = []
    for decision in s1_delta_decisions:
        affected_rows.extend(
            build_affected_rows_for_decision(
                decision,
                s2_rows=active_s2_rows,
                index_lookup=index_lookup,
                generated_at=resolved_generated_at,
            )
        )

    match_type_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    impacted_s2_unit_ids: set[str] = set()
    impacted_s2_index_entry_ids: set[str] = set()
    for row in affected_rows:
        match_type_counts[row["match_type"]] = match_type_counts.get(row["match_type"], 0) + 1
        action_counts[row["recommended_action"]] = action_counts.get(row["recommended_action"], 0) + 1
        signal_counts[row["s2_delta_signal"]] = signal_counts.get(row["s2_delta_signal"], 0) + 1
        if row["affected_s2_unit_id"]:
            impacted_s2_unit_ids.add(row["affected_s2_unit_id"])
        impacted_s2_index_entry_ids.update(row.get("s2_index_entry_ids") or [])

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "s1_delta_decision_count": len(s1_delta_decisions),
        "active_s2_count": len(active_s2_rows),
        "affected_row_count": len(affected_rows),
        "impacted_s2_unit_ids": sorted(impacted_s2_unit_ids),
        "impacted_s2_index_entry_ids": sorted(impacted_s2_index_entry_ids),
        "match_type_counts": dict(sorted(match_type_counts.items())),
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "s2_delta_signal_counts": dict(sorted(signal_counts.items())),
        "current_vs_historical_policy": (
            "Directly affected active S2 units should be refreshed or marked stale in a derived latest view; "
            "historical/audit rows remain queryable with status warnings."
        ),
        "boundary": "S2 affected-unit rows are maintenance signals, not updated portrait truth.",
        "read_only": True,
        "write_permission": False,
        "generated_at": resolved_generated_at,
    }
    return {"affected_rows": affected_rows, "report": report}


def write_report_bundle(output_dir: Path, bundle: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "s2_affected_units.jsonl"
    report_path = output_dir / "s2_affected_unit_report.json"
    write_jsonl(rows_path, bundle["affected_rows"])
    write_json(report_path, bundle["report"])
    return {"affected_rows_path": str(rows_path), "report_path": str(report_path)}


def build_s2_affected_unit_report_from_files(
    *,
    s1_delta_decisions_jsonl: Path,
    active_s2_jsonl: Path,
    output_dir: Path,
    s2_index_jsonl: Path | None = None,
) -> dict[str, Any]:
    bundle = build_s2_affected_unit_report(
        s1_delta_decisions=read_jsonl(s1_delta_decisions_jsonl),
        active_s2_rows=read_jsonl(active_s2_jsonl),
        s2_index_rows=read_jsonl(s2_index_jsonl),
    )
    bundle["paths"] = write_report_bundle(output_dir, bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-delta-decisions-jsonl", required=True, type=Path)
    parser.add_argument("--active-s2-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--s2-index-jsonl", type=Path)
    args = parser.parse_args()

    bundle = build_s2_affected_unit_report_from_files(
        s1_delta_decisions_jsonl=args.s1_delta_decisions_jsonl,
        active_s2_jsonl=args.active_s2_jsonl,
        s2_index_jsonl=args.s2_index_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps(bundle["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
