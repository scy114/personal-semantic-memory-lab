"""Publish reviewed S1 latest-view rows as the current S1 view.

S0B/S1 history remains append-only. This publisher makes the reviewed S1
latest view the machine-consumed current view and records explicit status
transitions for old rows that should no longer participate in default query or
downstream S2 refresh.
"""

from __future__ import annotations

import argparse
import json
import shutil
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from tools.graph.graph_construction_packet_builder import file_hash


SCHEMA_VERSION = "maintenance.s1_current_view_publish.v0.4"
DEFAULT_CURRENT_DIR_NAME = "s1_current"
DEFAULT_LATEST_VIEW_DIR = Path("maintenance") / "latest_views"

CURRENT_FILENAME = "s1_current.jsonl"
EXCLUDED_FILENAME = "s1_current_excluded.jsonl"
TRANSITIONS_FILENAME = "s1_status_transitions.jsonl"
MANIFEST_FILENAME = "s1_current_manifest.json"
REPORT_FILENAME = "s1_current_report.md"

DEFAULT_SOURCE_ACTIVE = "s1_latest_view_after_review.jsonl"
DEFAULT_SOURCE_EXCLUDED = "s1_latest_view_after_review_excluded.jsonl"
DEFAULT_SOURCE_MANIFEST = "s1_latest_view_after_review_manifest.json"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


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
            raise ValueError(f"S1 current row must be an object at {path}:{line_number}")
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


def memory_id(row: dict[str, Any]) -> str:
    for field in ("memory_id", "id", "object_id"):
        value = str(row.get(field) or "").strip()
        if value:
            return value
    maintenance = row.get("_maintenance_review_latest_view") or row.get("_maintenance_latest_view") or {}
    if isinstance(maintenance, dict):
        return str(maintenance.get("object_id") or "").strip()
    return ""


def review_annotation(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("_maintenance_review_latest_view") or row.get("_maintenance_latest_view") or {}
    return value if isinstance(value, dict) else {}


def current_status_for_active(row: dict[str, Any]) -> str:
    annotation = review_annotation(row)
    status = str(annotation.get("latest_view_status") or "").strip()
    if status == "active_after_review":
        return "active_current_review_accepted"
    if status == "active_existing":
        return "active_current_existing"
    if status:
        return f"active_current:{status}"
    return "active_current"


def current_status_for_excluded(row: dict[str, Any]) -> str:
    annotation = review_annotation(row)
    status = str(annotation.get("latest_view_status") or "").strip()
    reason = str(annotation.get("reason") or "").strip()
    if status == "historical_or_stale":
        return "historical_or_stale"
    if "superseded" in reason:
        return "superseded"
    if "stale" in reason:
        return "stale"
    if status:
        return status
    return "excluded_from_current"


def annotate_current_row(
    row: dict[str, Any],
    *,
    current_view_status: str,
    participates_in_default_query: bool,
    published_at: str,
) -> dict[str, Any]:
    out = deepcopy(row)
    item_id = memory_id(row)
    annotation = review_annotation(row)
    out["current_view_status"] = current_view_status
    out["participates_in_default_query"] = participates_in_default_query
    out["_maintenance_s1_current"] = {
        "schema_version": SCHEMA_VERSION,
        "layer": "s1",
        "memory_id": item_id,
        "current_view_status": current_view_status,
        "participates_in_default_query": participates_in_default_query,
        "review_item_id": annotation.get("review_item_id", ""),
        "patch_id": annotation.get("patch_id", ""),
        "operation_id": annotation.get("operation_id", ""),
        "overlay_effect": annotation.get("overlay_effect", ""),
        "published_at": published_at,
        "read_only_current_projection": True,
        "write_permission": False,
    }
    return out


def transition_row(
    row: dict[str, Any],
    *,
    current_view_status: str,
    participates_in_default_query: bool,
    source_bucket: str,
    published_at: str,
) -> dict[str, Any]:
    annotation = review_annotation(row)
    return {
        "schema_version": "maintenance.s1_status_transition.v0.4",
        "layer": "s1",
        "memory_id": memory_id(row),
        "source_bucket": source_bucket,
        "current_view_status": current_view_status,
        "previous_latest_view_status": annotation.get("latest_view_status", ""),
        "transition_reason": annotation.get("reason", ""),
        "overlay_effect": annotation.get("overlay_effect", ""),
        "review_item_id": annotation.get("review_item_id", ""),
        "patch_id": annotation.get("patch_id", ""),
        "operation_id": annotation.get("operation_id", ""),
        "participates_in_default_query": participates_in_default_query,
        "write_permission": False,
        "durable_writes_executed": False,
        "published_at": published_at,
    }


def build_current_rows(
    *,
    active_rows: list[dict[str, Any]],
    excluded_rows: list[dict[str, Any]],
    published_at: str,
) -> dict[str, Any]:
    current_rows: list[dict[str, Any]] = []
    current_excluded_rows: list[dict[str, Any]] = []
    transitions: list[dict[str, Any]] = []

    for row in active_rows:
        status = current_status_for_active(row)
        current_rows.append(
            annotate_current_row(
                row,
                current_view_status=status,
                participates_in_default_query=True,
                published_at=published_at,
            )
        )
        transitions.append(
            transition_row(
                row,
                current_view_status=status,
                participates_in_default_query=True,
                source_bucket="active",
                published_at=published_at,
            )
        )

    for row in excluded_rows:
        status = current_status_for_excluded(row)
        current_excluded_rows.append(
            annotate_current_row(
                row,
                current_view_status=status,
                participates_in_default_query=False,
                published_at=published_at,
            )
        )
        transitions.append(
            transition_row(
                row,
                current_view_status=status,
                participates_in_default_query=False,
                source_bucket="excluded",
                published_at=published_at,
            )
        )

    return {
        "current_rows": current_rows,
        "current_excluded_rows": current_excluded_rows,
        "transitions": transitions,
    }


def archive_previous_current(current_dir: Path, history_dir: Path, slug: str) -> Path | None:
    if not current_dir.exists():
        return None
    history_dir.mkdir(parents=True, exist_ok=True)
    archive_dir = history_dir / f"s1_current_previous_{slug}"
    shutil.copytree(current_dir, archive_dir)
    return archive_dir


def write_aliases(workspace: Path, current_dir: Path) -> dict[str, str]:
    latest_dir = workspace / DEFAULT_LATEST_VIEW_DIR
    latest_dir.mkdir(parents=True, exist_ok=True)
    alias_paths = {
        "s1_latest_view": latest_dir / "s1_latest_view.jsonl",
        "s1_latest_view_excluded": latest_dir / "s1_latest_view_excluded.jsonl",
        "s1_status_transitions": latest_dir / TRANSITIONS_FILENAME,
    }
    shutil.copy2(current_dir / CURRENT_FILENAME, alias_paths["s1_latest_view"])
    shutil.copy2(current_dir / EXCLUDED_FILENAME, alias_paths["s1_latest_view_excluded"])
    shutil.copy2(current_dir / TRANSITIONS_FILENAME, alias_paths["s1_status_transitions"])
    return {key: str(path) for key, path in alias_paths.items()}


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.4 S1 Current View Publish",
        "",
        f"- published_at: {manifest['published_at']}",
        f"- workspace: `{manifest['workspace']}`",
        f"- source_active: `{manifest['inputs']['source_active_jsonl']}`",
        f"- source_excluded: `{manifest['inputs']['source_excluded_jsonl']}`",
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
            "- S0B/S1 history remains append-only.",
            "- This publishes a machine-consumed current S1 view after human review.",
            "- Old S1 rows excluded from current query are tagged in `s1_status_transitions.jsonl`.",
            "- No durable memory or canonical history file is rewritten.",
            "",
        ]
    )
    return "\n".join(lines)


def run_s1_current_view_publish(
    *,
    workspace: Path,
    source_dir: Path,
    current_dir: Path | None = None,
    history_dir: Path | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    source_dir = source_dir.resolve()
    current_dir = (current_dir or workspace / DEFAULT_CURRENT_DIR_NAME).resolve()
    history_dir = (history_dir or workspace / "maintenance" / "published_views").resolve()

    source_active = source_dir / DEFAULT_SOURCE_ACTIVE
    source_excluded = source_dir / DEFAULT_SOURCE_EXCLUDED
    source_manifest = source_dir / DEFAULT_SOURCE_MANIFEST
    if not source_active.exists():
        raise FileNotFoundError(f"Reviewed S1 active latest-view file not found: {source_active}")
    if not source_excluded.exists():
        raise FileNotFoundError(f"Reviewed S1 excluded latest-view file not found: {source_excluded}")

    published_at = now_iso()
    slug = timestamp_slug()
    temp_dir = workspace / f".s1_current_tmp_{slug}"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)

    rows = build_current_rows(
        active_rows=read_jsonl(source_active),
        excluded_rows=read_jsonl(source_excluded),
        published_at=published_at,
    )
    write_jsonl(temp_dir / CURRENT_FILENAME, rows["current_rows"])
    write_jsonl(temp_dir / EXCLUDED_FILENAME, rows["current_excluded_rows"])
    write_jsonl(temp_dir / TRANSITIONS_FILENAME, rows["transitions"])

    status_counts: dict[str, int] = {}
    for row in rows["transitions"]:
        status = str(row.get("current_view_status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1

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
            "source_dir": str(source_dir),
            "source_active_jsonl": str(source_active),
            "source_excluded_jsonl": str(source_excluded),
            "source_manifest_json": str(source_manifest) if source_manifest.exists() else "",
        },
        "input_hashes": {
            "source_active_jsonl": file_hash(source_active),
            "source_excluded_jsonl": file_hash(source_excluded),
            "source_manifest_json": file_hash(source_manifest),
            "previous_current_manifest": previous_manifest_hash,
        },
        "source_manifest": read_json(source_manifest),
        "current_dir": str(current_dir),
        "previous_current_archive_dir": str(previous_archive) if previous_archive else "",
        "history_dir": str(history_dir),
        "counts": {
            "current_active_count": len(rows["current_rows"]),
            "current_excluded_count": len(rows["current_excluded_rows"]),
            "status_transition_count": len(rows["transitions"]),
            "current_view_status_counts": dict(sorted(status_counts.items())),
        },
        "paths": {
            "s1_current": str(current_dir / CURRENT_FILENAME),
            "s1_current_excluded": str(current_dir / EXCLUDED_FILENAME),
            "s1_status_transitions": str(current_dir / TRANSITIONS_FILENAME),
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
            "durable_memory_written": False,
            "canonical_history_rewritten": False,
        },
        "boundary": "published_current_s1_view_no_durable_or_canonical_history_write",
        "write_permission": False,
        "publish_executed": True,
        "query_default_ready": True,
        "durable_writes_executed": False,
        "canonical_history_rewritten": False,
    }
    write_json(current_dir / MANIFEST_FILENAME, manifest)
    write_text(current_dir / REPORT_FILENAME, render_report(manifest))
    write_json(workspace / DEFAULT_LATEST_VIEW_DIR / "s1_current_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--current-dir", type=Path)
    parser.add_argument("--history-dir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_s1_current_view_publish(
        workspace=args.workspace,
        source_dir=args.source_dir,
        current_dir=args.current_dir,
        history_dir=args.history_dir,
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
