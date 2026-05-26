"""Compare v0.4 incremental maintenance output with a small full-rebuild observation.

The comparator is report-only. It does not run a rebuild. Instead it compares
incremental S1/S2/graph reports against an explicit observation file produced
from a small full-rebuild check or a controlled fixture.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "maintenance.incremental_vs_full_rebuild_row.v0.4"
REPORT_SCHEMA_VERSION = "maintenance.incremental_vs_full_rebuild_report.v0.4"
OBSERVATION_SCHEMA_VERSION = "maintenance.full_rebuild_observation.v0.4"


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
            raise ValueError(f"Comparator input row must be an object at {path}:{line_number}")
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


def group_rows_by_s1_unit(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = normalize(row.get("new_s1_unit_id"))
        if not key:
            continue
        grouped.setdefault(key, []).append(row)
    return grouped


def s1_decisions_by_unit(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize(row.get("new_s1_unit_id")): row for row in rows if normalize(row.get("new_s1_unit_id"))}


def collect_field(rows: list[dict[str, Any]], field_name: str) -> list[str]:
    values: list[Any] = []
    for row in rows:
        value = row.get(field_name)
        if isinstance(value, list):
            values.extend(value)
        else:
            values.append(value)
    return normalize_unique(values)


def compare_scalar(
    *,
    mismatches: list[str],
    field_name: str,
    expected: Any,
    actual: Any,
) -> bool:
    if expected is None:
        return True
    expected_text = normalize(expected)
    actual_text = normalize(actual)
    if expected_text == actual_text:
        return True
    mismatches.append(f"{field_name}: expected={expected_text!r} actual={actual_text!r}")
    return False


def compare_set(
    *,
    mismatches: list[str],
    field_name: str,
    expected: Any,
    actual_values: list[str],
) -> bool:
    if expected is None:
        return True
    expected_values = set(normalize_unique(expected if isinstance(expected, list) else [expected]))
    actual_set = set(actual_values)
    missing = sorted(expected_values - actual_set)
    extra = sorted(actual_set - expected_values)
    if not missing and not extra:
        return True
    if missing:
        mismatches.append(f"{field_name}: missing={missing}")
    if extra:
        mismatches.append(f"{field_name}: extra={extra}")
    return False


def normalize_observations(observation_payload: dict[str, Any]) -> list[dict[str, Any]]:
    if observation_payload.get("schema_version") not in {None, OBSERVATION_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported observation schema_version: {observation_payload.get('schema_version')}")
    observations = observation_payload.get("observations") or []
    if not isinstance(observations, list):
        raise ValueError("full rebuild observation must contain an observations list")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(observations, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Observation item must be an object at index {index}")
        if not normalize(item.get("new_s1_unit_id")):
            raise ValueError(f"Observation item missing new_s1_unit_id at index {index}")
        normalized.append(item)
    return normalized


def build_comparison_rows(
    *,
    observations: list[dict[str, Any]],
    s1_delta_decisions: list[dict[str, Any]],
    s2_affected_rows: list[dict[str, Any]],
    graph_affected_rows: list[dict[str, Any]],
    generated_at: str,
) -> list[dict[str, Any]]:
    s1_by_unit = s1_decisions_by_unit(s1_delta_decisions)
    s2_by_unit = group_rows_by_s1_unit(s2_affected_rows)
    graph_by_unit = group_rows_by_s1_unit(graph_affected_rows)
    rows: list[dict[str, Any]] = []

    for observation in observations:
        new_s1_unit_id = normalize(observation.get("new_s1_unit_id"))
        s1_row = s1_by_unit.get(new_s1_unit_id, {})
        s2_rows = s2_by_unit.get(new_s1_unit_id, [])
        graph_rows = graph_by_unit.get(new_s1_unit_id, [])
        mismatches: list[str] = []

        compare_scalar(
            mismatches=mismatches,
            field_name="s1_delta_type",
            expected=observation.get("expected_s1_delta_type"),
            actual=s1_row.get("delta_type"),
        )
        compare_scalar(
            mismatches=mismatches,
            field_name="s2_recommended_action",
            expected=observation.get("expected_s2_recommended_action"),
            actual=(s2_rows[0].get("recommended_action") if s2_rows else ""),
        )
        compare_scalar(
            mismatches=mismatches,
            field_name="graph_recommended_action",
            expected=observation.get("expected_graph_recommended_action"),
            actual=(graph_rows[0].get("recommended_action") if graph_rows else ""),
        )
        compare_set(
            mismatches=mismatches,
            field_name="affected_s2_unit_ids",
            expected=observation.get("expected_affected_s2_unit_ids"),
            actual_values=collect_field(s2_rows, "affected_s2_unit_id"),
        )
        compare_set(
            mismatches=mismatches,
            field_name="affected_s2_index_entry_ids",
            expected=observation.get("expected_affected_s2_index_entry_ids"),
            actual_values=collect_field(s2_rows, "s2_index_entry_ids"),
        )
        compare_set(
            mismatches=mismatches,
            field_name="affected_graph_packet_ids",
            expected=observation.get("expected_affected_graph_packet_ids"),
            actual_values=collect_field(graph_rows, "affected_graph_packet_ids"),
        )
        compare_set(
            mismatches=mismatches,
            field_name="primary_graph_packet_ids",
            expected=observation.get("expected_primary_graph_packet_ids"),
            actual_values=collect_field(graph_rows, "primary_graph_packet_ids"),
        )
        compare_set(
            mismatches=mismatches,
            field_name="affected_graph_candidate_ids",
            expected=observation.get("expected_affected_graph_candidate_ids"),
            actual_values=collect_field(graph_rows, "affected_graph_candidate_ids"),
        )

        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "new_s1_unit_id": new_s1_unit_id,
                "comparison_status": "pass" if not mismatches else "fail",
                "mismatches": mismatches,
                "expected": observation,
                "actual": {
                    "s1_delta_type": s1_row.get("delta_type"),
                    "s2_recommended_actions": collect_field(s2_rows, "recommended_action"),
                    "affected_s2_unit_ids": collect_field(s2_rows, "affected_s2_unit_id"),
                    "affected_s2_index_entry_ids": collect_field(s2_rows, "s2_index_entry_ids"),
                    "graph_recommended_actions": collect_field(graph_rows, "recommended_action"),
                    "affected_graph_packet_ids": collect_field(graph_rows, "affected_graph_packet_ids"),
                    "primary_graph_packet_ids": collect_field(graph_rows, "primary_graph_packet_ids"),
                    "context_graph_packet_ids": collect_field(graph_rows, "context_graph_packet_ids"),
                    "affected_graph_candidate_ids": collect_field(graph_rows, "affected_graph_candidate_ids"),
                },
                "incremental_is_not_full_rebuild": True,
                "read_only": True,
                "write_permission": False,
                "generated_at": generated_at,
            }
        )
    return rows


def build_incremental_vs_full_rebuild_report(
    *,
    observation_payload: dict[str, Any],
    s1_delta_decisions: list[dict[str, Any]],
    s2_affected_rows: list[dict[str, Any]],
    graph_affected_rows: list[dict[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_generated_at = generated_at or now_iso()
    observations = normalize_observations(observation_payload)
    rows = build_comparison_rows(
        observations=observations,
        s1_delta_decisions=s1_delta_decisions,
        s2_affected_rows=s2_affected_rows,
        graph_affected_rows=graph_affected_rows,
        generated_at=resolved_generated_at,
    )
    pass_count = sum(1 for row in rows if row["comparison_status"] == "pass")
    fail_count = len(rows) - pass_count
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "observation_schema_version": observation_payload.get("schema_version") or OBSERVATION_SCHEMA_VERSION,
        "observation_name": observation_payload.get("observation_name", ""),
        "observation_count": len(observations),
        "comparison_row_count": len(rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "acceptance_status": "pass" if fail_count == 0 else "fail",
        "boundary": "This compares incremental maintenance scope against a small full-rebuild observation; it does not prove semantic quality.",
        "read_only": True,
        "write_permission": False,
        "generated_at": resolved_generated_at,
    }
    return {"comparison_rows": rows, "report": report}


def write_report_bundle(output_dir: Path, bundle: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "incremental_vs_full_rebuild_rows.jsonl"
    report_path = output_dir / "incremental_vs_full_rebuild_report.json"
    write_jsonl(rows_path, bundle["comparison_rows"])
    write_json(report_path, bundle["report"])
    return {"comparison_rows_path": str(rows_path), "report_path": str(report_path)}


def build_incremental_vs_full_rebuild_report_from_files(
    *,
    full_rebuild_observation_json: Path,
    s1_delta_decisions_jsonl: Path,
    s2_affected_rows_jsonl: Path,
    graph_affected_rows_jsonl: Path,
    output_dir: Path,
) -> dict[str, Any]:
    bundle = build_incremental_vs_full_rebuild_report(
        observation_payload=read_json(full_rebuild_observation_json),
        s1_delta_decisions=read_jsonl(s1_delta_decisions_jsonl),
        s2_affected_rows=read_jsonl(s2_affected_rows_jsonl),
        graph_affected_rows=read_jsonl(graph_affected_rows_jsonl),
    )
    bundle["paths"] = write_report_bundle(output_dir, bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-rebuild-observation-json", required=True, type=Path)
    parser.add_argument("--s1-delta-decisions-jsonl", required=True, type=Path)
    parser.add_argument("--s2-affected-rows-jsonl", required=True, type=Path)
    parser.add_argument("--graph-affected-rows-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    bundle = build_incremental_vs_full_rebuild_report_from_files(
        full_rebuild_observation_json=args.full_rebuild_observation_json,
        s1_delta_decisions_jsonl=args.s1_delta_decisions_jsonl,
        s2_affected_rows_jsonl=args.s2_affected_rows_jsonl,
        graph_affected_rows_jsonl=args.graph_affected_rows_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps(bundle["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
