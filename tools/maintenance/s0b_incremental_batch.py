"""Register add-only S0B incremental batches for v0.4 maintenance.

This helper records a new S0B/evidence batch as a derived maintenance artifact
and optionally appends an operation-log row. It does not mutate canonical
evidence, text unit, memory, portrait, index, graph, or query files.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from tools.maintenance.operation_log import append_operation, build_operation


SCHEMA_VERSION = "maintenance.s0b_incremental_batch.v0.4"
MANIFEST_SCHEMA_VERSION = "maintenance.s0b_incremental_batch_manifest.v0.4"


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
            raise ValueError(f"S0B batch row must be an object at {path}:{line_number}")
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


def stable_row_digest(row: dict[str, Any]) -> str:
    encoded = json.dumps(row, ensure_ascii=False, sort_keys=True)
    return sha256(encoded.encode("utf-8")).hexdigest()


def first_value(row: dict[str, Any], *field_names: str) -> str:
    for field_name in field_names:
        value = normalize(row.get(field_name))
        if value:
            return value
    return ""


def normalize_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def collect_source_ids(row: dict[str, Any]) -> list[str]:
    source_ids = [
        first_value(row, "source_id"),
        first_value(row, "raw_source_id"),
    ]
    for value in row.get("source_refs") or []:
        source_ids.append(normalize(value))
    source_ref = row.get("source_ref")
    if isinstance(source_ref, dict):
        source_ids.extend(normalize(source_ref.get(key)) for key in ("source_id", "raw_source_id"))
    elif source_ref:
        source_ids.append(normalize(source_ref))
    return normalize_unique(source_ids)


def build_batch_row(
    *,
    batch_id: str,
    row_kind: str,
    row: dict[str, Any],
    source_path: str,
    generated_at: str,
) -> dict[str, Any]:
    object_id = {
        "evidence": first_value(row, "evidence_ref", "canonical_evidence_ref", "id"),
        "text_unit": first_value(row, "text_unit_id", "id"),
        "section": first_value(row, "raw_span_id", "section_id", "id"),
    }[row_kind]
    evidence_refs = normalize_unique(
        [
            first_value(row, "evidence_ref", "canonical_evidence_ref"),
            *(row.get("evidence_refs") or []),
        ]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "row_kind": row_kind,
        "object_id": object_id,
        "source_ids": collect_source_ids(row),
        "evidence_refs": evidence_refs,
        "s0b_unit_ids": [object_id] if row_kind in {"text_unit", "section"} and object_id else [],
        "row_digest": stable_row_digest(row),
        "source_path": source_path,
        "raw_row": row,
        "add_only": True,
        "canonical_mutation": False,
        "read_only": True,
        "write_permission": False,
        "generated_at": generated_at,
    }


def build_s0b_incremental_batch(
    *,
    workspace: Path,
    batch_id: str,
    evidence_rows: list[dict[str, Any]] | None = None,
    text_unit_rows: list[dict[str, Any]] | None = None,
    section_rows: list[dict[str, Any]] | None = None,
    evidence_path: str = "",
    text_units_path: str = "",
    section_map_path: str = "",
    generated_at: str | None = None,
) -> dict[str, Any]:
    if not batch_id:
        raise ValueError("batch_id is required")
    resolved_generated_at = generated_at or now_iso()
    rows: list[dict[str, Any]] = []
    for row in evidence_rows or []:
        rows.append(
            build_batch_row(
                batch_id=batch_id,
                row_kind="evidence",
                row=row,
                source_path=evidence_path,
                generated_at=resolved_generated_at,
            )
        )
    for row in text_unit_rows or []:
        rows.append(
            build_batch_row(
                batch_id=batch_id,
                row_kind="text_unit",
                row=row,
                source_path=text_units_path,
                generated_at=resolved_generated_at,
            )
        )
    for row in section_rows or []:
        rows.append(
            build_batch_row(
                batch_id=batch_id,
                row_kind="section",
                row=row,
                source_path=section_map_path,
                generated_at=resolved_generated_at,
            )
        )

    duplicate_ids = sorted(
        object_id
        for object_id in {row["object_id"] for row in rows if row["object_id"]}
        if sum(1 for row in rows if row["object_id"] == object_id) > 1
    )
    if duplicate_ids:
        raise ValueError(f"Duplicate object ids inside S0B incremental batch: {duplicate_ids}")

    source_ids = normalize_unique(source_id for row in rows for source_id in row["source_ids"])
    evidence_refs = normalize_unique(evidence_ref for row in rows for evidence_ref in row["evidence_refs"])
    s0b_unit_ids = normalize_unique(s0b_id for row in rows for s0b_id in row["s0b_unit_ids"])
    row_kind_counts: dict[str, int] = {}
    for row in rows:
        row_kind_counts[row["row_kind"]] = row_kind_counts.get(row["row_kind"], 0) + 1

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "batch_id": batch_id,
        "workspace": str(workspace),
        "row_count": len(rows),
        "row_kind_counts": dict(sorted(row_kind_counts.items())),
        "source_ids": source_ids,
        "evidence_refs": evidence_refs,
        "s0b_unit_ids": s0b_unit_ids,
        "source_paths": {
            "evidence": evidence_path,
            "text_units": text_units_path,
            "section_map": section_map_path,
        },
        "add_only": True,
        "canonical_mutation": False,
        "read_only": True,
        "write_permission": False,
        "generated_at": resolved_generated_at,
    }
    return {"rows": rows, "manifest": manifest}


def write_batch_bundle(output_dir: Path, batch: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_id = batch["manifest"]["batch_id"]
    rows_path = output_dir / f"{batch_id}_s0b_incremental_batch_rows.jsonl"
    manifest_path = output_dir / f"{batch_id}_s0b_incremental_batch_manifest.json"
    write_jsonl(rows_path, batch["rows"])
    write_json(manifest_path, batch["manifest"])
    return {"rows_path": str(rows_path), "manifest_path": str(manifest_path)}


def register_s0b_incremental_batch(
    *,
    workspace: Path,
    batch_id: str,
    output_dir: Path,
    evidence_jsonl: Path | None = None,
    text_units_jsonl: Path | None = None,
    section_map_jsonl: Path | None = None,
    operation_log: Path | None = None,
    operation_status: str = "completed",
) -> dict[str, Any]:
    evidence_rows = read_jsonl(evidence_jsonl)
    text_unit_rows = read_jsonl(text_units_jsonl)
    section_rows = read_jsonl(section_map_jsonl)
    batch = build_s0b_incremental_batch(
        workspace=workspace,
        batch_id=batch_id,
        evidence_rows=evidence_rows,
        text_unit_rows=text_unit_rows,
        section_rows=section_rows,
        evidence_path=str(evidence_jsonl or ""),
        text_units_path=str(text_units_jsonl or ""),
        section_map_path=str(section_map_jsonl or ""),
    )
    paths = write_batch_bundle(output_dir, batch)
    operation = None
    if operation_log:
        operation = build_operation(
            workspace_id=workspace.name,
            operation_type="add_s0b_batch",
            idempotency_key=batch_id,
            status=operation_status,
            scope={
                "source_ids": batch["manifest"]["source_ids"],
                "evidence_refs": batch["manifest"]["evidence_refs"],
                "s0b_unit_ids": batch["manifest"]["s0b_unit_ids"],
            },
            artifacts={
                "s0b_incremental_batch_rows": paths["rows_path"],
                "s0b_incremental_batch_manifest": paths["manifest_path"],
            },
            warnings=["canonical_s0b_files_not_mutated"],
        )
        append_operation(operation_log, operation)
    return {"batch": batch, "paths": paths, "operation": operation}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--evidence-jsonl", type=Path)
    parser.add_argument("--text-units-jsonl", type=Path)
    parser.add_argument("--section-map-jsonl", type=Path)
    parser.add_argument("--operation-log", type=Path)
    parser.add_argument("--operation-status", default="completed")
    args = parser.parse_args()
    result = register_s0b_incremental_batch(
        workspace=args.workspace,
        batch_id=args.batch_id,
        output_dir=args.output_dir,
        evidence_jsonl=args.evidence_jsonl,
        text_units_jsonl=args.text_units_jsonl,
        section_map_jsonl=args.section_map_jsonl,
        operation_log=args.operation_log,
        operation_status=args.operation_status,
    )
    print(json.dumps(result["batch"]["manifest"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
