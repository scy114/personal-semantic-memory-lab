"""Append-only operation log for v0.4 incremental maintenance.

The operation log is the entry point for source/S0B/S1/S2 maintenance work.
It records intent, scope, lineage, artifacts, and status without mutating
derived memory or graph assets.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "maintenance.operation_log.v0.4"

OPERATION_TYPES = {
    "add_source",
    "supersede_source",
    "deprecate_source",
    "restore_source",
    "hard_delete_source_candidate",
    "add_s0b_batch",
    "refresh_s1_from_s0b_delta",
    "refresh_s2_from_s1_delta",
    "refresh_graph_from_s1s2_delta",
    "refresh_indexes",
    "publish_latest_view",
}

OPERATION_STATUSES = {
    "planned",
    "running",
    "completed",
    "failed",
    "rolled_back_candidate",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_strings(values: list[Any] | None) -> list[str]:
    seen: set[str] = set()
    normalized: list[str] = []
    for value in values or []:
        text = str(value).strip()
        if text and text not in seen:
            seen.add(text)
            normalized.append(text)
    return normalized


def stable_operation_id(
    *,
    workspace_id: str,
    operation_type: str,
    idempotency_key: str,
) -> str:
    seed = json.dumps(
        {
            "schema_version": SCHEMA_VERSION,
            "workspace_id": workspace_id,
            "operation_type": operation_type,
            "idempotency_key": idempotency_key,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return "op_" + sha256(seed.encode("utf-8")).hexdigest()[:16]


def default_scope() -> dict[str, list[str]]:
    return {
        "source_ids": [],
        "source_versions": [],
        "evidence_refs": [],
        "s0b_unit_ids": [],
        "s1_unit_ids": [],
        "s2_unit_ids": [],
        "index_entry_ids": [],
        "graph_candidate_ids": [],
        "query_packet_ids": [],
    }


def normalize_scope(scope: dict[str, Any] | None) -> dict[str, list[str]]:
    normalized = default_scope()
    for key, value in (scope or {}).items():
        if key not in normalized:
            raise ValueError(f"Unsupported operation scope key: {key}")
        if isinstance(value, list):
            normalized[key] = normalize_strings(value)
        else:
            normalized[key] = normalize_strings([value])
    return normalized


def build_operation(
    *,
    workspace_id: str,
    operation_type: str,
    idempotency_key: str,
    scope: dict[str, Any] | None = None,
    status: str = "planned",
    modeled_user_id: str = "",
    actor: str = "codex",
    operation_id: str = "",
    parent_operation_ids: list[Any] | None = None,
    supersedes_operation_ids: list[Any] | None = None,
    parameters: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    warnings: list[Any] | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    if operation_type not in OPERATION_TYPES:
        raise ValueError(f"Unsupported operation_type: {operation_type}")
    if status not in OPERATION_STATUSES:
        raise ValueError(f"Unsupported operation status: {status}")
    if not workspace_id:
        raise ValueError("workspace_id is required")
    if not idempotency_key:
        raise ValueError("idempotency_key is required")

    resolved_operation_id = operation_id or stable_operation_id(
        workspace_id=workspace_id,
        operation_type=operation_type,
        idempotency_key=idempotency_key,
    )

    operation = {
        "schema_version": SCHEMA_VERSION,
        "operation_id": resolved_operation_id,
        "operation_type": operation_type,
        "status": status,
        "workspace_id": workspace_id,
        "modeled_user_id": modeled_user_id,
        "actor": actor,
        "idempotency_key": idempotency_key,
        "scope": normalize_scope(scope),
        "parent_operation_ids": normalize_strings(parent_operation_ids),
        "supersedes_operation_ids": normalize_strings(supersedes_operation_ids),
        "parameters": parameters or {},
        "artifacts": artifacts or {},
        "warnings": normalize_strings(warnings),
        "created_at": created_at or now_iso(),
        "append_only": True,
        "write_permission": False,
    }
    validate_operation(operation)
    return operation


def validate_operation(operation: dict[str, Any]) -> None:
    required = [
        "schema_version",
        "operation_id",
        "operation_type",
        "status",
        "workspace_id",
        "idempotency_key",
        "scope",
        "created_at",
        "append_only",
        "write_permission",
    ]
    missing = [key for key in required if key not in operation]
    if missing:
        raise ValueError(f"Operation missing required fields: {missing}")
    if operation["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {operation['schema_version']}")
    if operation["operation_type"] not in OPERATION_TYPES:
        raise ValueError(f"Unsupported operation_type: {operation['operation_type']}")
    if operation["status"] not in OPERATION_STATUSES:
        raise ValueError(f"Unsupported operation status: {operation['status']}")
    if operation["append_only"] is not True:
        raise ValueError("operation log rows must declare append_only=true")
    if operation["write_permission"] is not False:
        raise ValueError("operation log rows must declare write_permission=false")
    normalize_scope(operation.get("scope"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Operation row must be an object at {path}:{line_number}")
        validate_operation(row)
        rows.append(row)
    return rows


def append_operation(log_path: Path, operation: dict[str, Any]) -> dict[str, Any]:
    validate_operation(operation)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(operation, ensure_ascii=False, sort_keys=True) + "\n")
    return operation


def latest_status_by_operation_id(rows: list[dict[str, Any]]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for row in rows:
        statuses[row["operation_id"]] = row["status"]
    return statuses


def build_manifest(log_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    type_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
        type_counts[row["operation_type"]] = type_counts.get(row["operation_type"], 0) + 1
    return {
        "schema_version": "maintenance.operation_log_manifest.v0.4",
        "operation_log_path": str(log_path),
        "operation_count": len(rows),
        "operation_type_counts": dict(sorted(type_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "append_only": True,
        "write_permission": False,
        "generated_at": now_iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operation-log", required=True, type=Path)
    parser.add_argument("--workspace-id", required=True)
    parser.add_argument("--operation-type", required=True, choices=sorted(OPERATION_TYPES))
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--modeled-user-id", default="")
    parser.add_argument("--status", default="planned", choices=sorted(OPERATION_STATUSES))
    parser.add_argument("--source-id", action="append", default=[])
    parser.add_argument("--source-version", action="append", default=[])
    parser.add_argument("--evidence-ref", action="append", default=[])
    parser.add_argument("--s0b-unit-id", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    operation = build_operation(
        workspace_id=args.workspace_id,
        modeled_user_id=args.modeled_user_id,
        operation_type=args.operation_type,
        status=args.status,
        idempotency_key=args.idempotency_key,
        scope={
            "source_ids": args.source_id,
            "source_versions": args.source_version,
            "evidence_refs": args.evidence_ref,
            "s0b_unit_ids": args.s0b_unit_id,
        },
    )
    if not args.dry_run:
        append_operation(args.operation_log, operation)
    print(json.dumps(operation, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
