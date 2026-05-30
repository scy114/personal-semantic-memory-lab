"""Consume reviewed S1 overlay plan rows and build a derived S1 latest view.

This runner is review-after-approval plumbing. It reads active S1 rows,
incremental S1 candidate rows, and `s1_latest_view_overlay.jsonl` emitted by
the review decision applier. It writes a derived latest view without mutating
canonical S1 memory files.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "maintenance.s1_review_latest_view.v0.4"
MANIFEST_SCHEMA_VERSION = "maintenance.s1_review_latest_view_manifest.v0.4"

LATEST_VIEW_FILENAME = "s1_latest_view_after_review.jsonl"
EXCLUDED_VIEW_FILENAME = "s1_latest_view_after_review_excluded.jsonl"
MANIFEST_FILENAME = "s1_latest_view_after_review_manifest.json"
REPORT_FILENAME = "s1_latest_view_after_review_report.md"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize(value: Any) -> str:
    return str(value or "").strip()


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
            raise ValueError(f"S1 review overlay input row must be an object at {path}:{line_number}")
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


def row_id(row: dict[str, Any]) -> str:
    for field_name in ("memory_id", "id"):
        value = normalize(row.get(field_name))
        if value:
            return value
    return ""


def index_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row_id(row): row for row in rows if row_id(row)}


def annotate(
    row: dict[str, Any],
    *,
    latest_view_status: str,
    reason: str,
    overlay_row: dict[str, Any] | None,
    generated_at: str,
) -> dict[str, Any]:
    annotated = deepcopy(row)
    annotated["_maintenance_review_latest_view"] = {
        "schema_version": SCHEMA_VERSION,
        "layer": "s1",
        "object_id": row_id(row),
        "latest_view_status": latest_view_status,
        "reason": reason,
        "review_item_id": (overlay_row or {}).get("review_item_id", ""),
        "patch_id": (overlay_row or {}).get("patch_id", ""),
        "operation_id": (overlay_row or {}).get("operation_id", ""),
        "overlay_effect": (overlay_row or {}).get("overlay_effect", ""),
        "generated_at": generated_at,
        "read_only": True,
        "write_permission": False,
    }
    return annotated


def active_and_stale_ids(overlay_rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    active: dict[str, dict[str, Any]] = {}
    stale: dict[str, dict[str, Any]] = {}
    for overlay in overlay_rows:
        for memory_id in overlay.get("current_candidate_ids") or []:
            text = normalize(memory_id)
            if text:
                active[text] = overlay
        for memory_id in overlay.get("historical_or_stale_candidate_ids") or []:
            text = normalize(memory_id)
            if text:
                stale[text] = overlay
    return active, stale


def build_s1_review_latest_view(
    *,
    active_s1_rows: list[dict[str, Any]],
    incremental_s1_rows: list[dict[str, Any]],
    s1_review_overlay_rows: list[dict[str, Any]],
    existing_excluded_s1_rows: list[dict[str, Any]] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_generated_at = generated_at or now_iso()
    active_candidate_ids, stale_candidate_ids = active_and_stale_ids(s1_review_overlay_rows)
    active_by_id = index_by_id(active_s1_rows)
    excluded_by_id = index_by_id(existing_excluded_s1_rows or [])
    incremental_by_id = index_by_id(incremental_s1_rows)

    latest_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    missing_current_candidate_ids: list[str] = []
    missing_stale_candidate_ids: list[str] = []

    for memory_id, row in active_by_id.items():
        if memory_id in stale_candidate_ids:
            excluded_rows.append(
                annotate(
                    row,
                    latest_view_status="historical_or_stale",
                    reason="excluded_by_review_overlay",
                    overlay_row=stale_candidate_ids[memory_id],
                    generated_at=resolved_generated_at,
                )
            )
        else:
            latest_rows.append(
                annotate(
                    row,
                    latest_view_status="active_existing",
                    reason="kept_from_active_s1_input",
                    overlay_row=None,
                    generated_at=resolved_generated_at,
                )
            )

    for memory_id, overlay in active_candidate_ids.items():
        row = incremental_by_id.get(memory_id) or active_by_id.get(memory_id)
        if row is None:
            missing_current_candidate_ids.append(memory_id)
            continue
        latest_rows.append(
            annotate(
                row,
                latest_view_status="active_after_review",
                reason="accepted_by_review_overlay",
                overlay_row=overlay,
                generated_at=resolved_generated_at,
            )
        )

    for memory_id in stale_candidate_ids:
        if memory_id not in active_by_id:
            row = excluded_by_id.get(memory_id)
            if row is None:
                missing_stale_candidate_ids.append(memory_id)
                continue
            excluded_rows.append(
                annotate(
                    row,
                    latest_view_status="historical_or_stale",
                    reason="confirmed_historical_from_existing_excluded_input",
                    overlay_row=stale_candidate_ids[memory_id],
                    generated_at=resolved_generated_at,
                )
            )

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "input_active_s1_count": len(active_s1_rows),
        "input_incremental_s1_count": len(incremental_s1_rows),
        "input_existing_excluded_s1_count": len(existing_excluded_s1_rows or []),
        "input_overlay_count": len(s1_review_overlay_rows),
        "latest_view_count": len(latest_rows),
        "excluded_count": len(excluded_rows),
        "accepted_current_candidate_count": len(active_candidate_ids),
        "historical_or_stale_candidate_count": len(stale_candidate_ids),
        "missing_current_candidate_ids": missing_current_candidate_ids,
        "missing_stale_candidate_ids": missing_stale_candidate_ids,
        "boundary": "derived_after_human_review_no_canonical_memory_mutation",
        "write_permission": False,
        "apply_executed": False,
        "durable_writes_executed": False,
        "generated_at": resolved_generated_at,
    }
    return {
        "latest_rows": latest_rows,
        "excluded_rows": excluded_rows,
        "manifest": manifest,
    }


def report_markdown(manifest: dict[str, Any], paths: dict[str, str] | None = None) -> str:
    paths = paths or {}
    lines = [
        "# v0.4 S1 Latest View After Review",
        "",
        f"- generated_at: {manifest['generated_at']}",
        f"- input_active_s1_count: {manifest['input_active_s1_count']}",
        f"- input_incremental_s1_count: {manifest['input_incremental_s1_count']}",
        f"- input_existing_excluded_s1_count: {manifest['input_existing_excluded_s1_count']}",
        f"- input_overlay_count: {manifest['input_overlay_count']}",
        f"- latest_view_count: {manifest['latest_view_count']}",
        f"- excluded_count: {manifest['excluded_count']}",
        f"- accepted_current_candidate_count: {manifest['accepted_current_candidate_count']}",
        f"- historical_or_stale_candidate_count: {manifest['historical_or_stale_candidate_count']}",
        "",
        "## Boundary",
        "",
        "This is a derived latest view after human review. It does not mutate canonical S1 memory or durable memory.",
        "",
    ]
    if manifest["missing_current_candidate_ids"] or manifest["missing_stale_candidate_ids"]:
        lines.extend(["## Warnings", ""])
        for memory_id in manifest["missing_current_candidate_ids"]:
            lines.append(f"- missing current candidate row: `{memory_id}`")
        for memory_id in manifest["missing_stale_candidate_ids"]:
            lines.append(f"- missing stale candidate row: `{memory_id}`")
        lines.append("")
    if paths:
        lines.extend(["## Artifacts", ""])
        for name, path in sorted(paths.items()):
            lines.append(f"- {name}: `{path}`")
    lines.append("")
    return "\n".join(lines)


def write_bundle(output_dir: Path, bundle: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "latest_view": str(output_dir / LATEST_VIEW_FILENAME),
        "excluded_view": str(output_dir / EXCLUDED_VIEW_FILENAME),
        "manifest": str(output_dir / MANIFEST_FILENAME),
        "report": str(output_dir / REPORT_FILENAME),
    }
    write_jsonl(Path(paths["latest_view"]), bundle["latest_rows"])
    write_jsonl(Path(paths["excluded_view"]), bundle["excluded_rows"])
    write_json(Path(paths["manifest"]), {**bundle["manifest"], "paths": paths})
    write_text(Path(paths["report"]), report_markdown(bundle["manifest"], paths))
    return paths


def build_s1_review_latest_view_from_files(
    *,
    active_s1_jsonl: Path,
    incremental_s1_jsonl: Path,
    s1_review_overlay_jsonl: Path,
    output_dir: Path,
    existing_excluded_s1_jsonl: Path | None = None,
) -> dict[str, Any]:
    bundle = build_s1_review_latest_view(
        active_s1_rows=read_jsonl(active_s1_jsonl),
        incremental_s1_rows=read_jsonl(incremental_s1_jsonl),
        s1_review_overlay_rows=read_jsonl(s1_review_overlay_jsonl),
        existing_excluded_s1_rows=read_jsonl(existing_excluded_s1_jsonl),
    )
    bundle["paths"] = write_bundle(output_dir, bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-s1-jsonl", required=True, type=Path)
    parser.add_argument("--incremental-s1-jsonl", required=True, type=Path)
    parser.add_argument("--s1-review-overlay-jsonl", required=True, type=Path)
    parser.add_argument("--existing-excluded-s1-jsonl", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    bundle = build_s1_review_latest_view_from_files(
        active_s1_jsonl=args.active_s1_jsonl,
        incremental_s1_jsonl=args.incremental_s1_jsonl,
        s1_review_overlay_jsonl=args.s1_review_overlay_jsonl,
        existing_excluded_s1_jsonl=args.existing_excluded_s1_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps(bundle["manifest"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
