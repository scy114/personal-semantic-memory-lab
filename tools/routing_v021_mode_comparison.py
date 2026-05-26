"""Compare existing routes against v0.21 resource-backed feature axes.

This tool is read-only. It consumes ``v021_feature_matrix.jsonl`` and produces
inspectable comparison artifacts for calibration review. It does not replace
route decisions or write canonical build outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "routing.v021_mode_comparison.v0.1"
DEFAULT_OUTPUT_NAME = "v021_mode_comparison"
TARGET_TASKS = ("s1_memory_candidate", "s2_portrait_candidate")
ROUTE_RANK = {
    "": 0.0,
    "skip_or_background_only": 0.0,
    "split_or_segment_first": 1.5,
    "script_only": 3.0,
    "weak_llm_proposal": 5.5,
    "strong_llm_proposal": 8.0,
    "human_review": 9.0,
}
ACTIVE_ROUTES = {"script_only", "weak_llm_proposal", "strong_llm_proposal", "human_review"}
LLM_ROUTES = {"weak_llm_proposal", "strong_llm_proposal", "human_review"}


def clamp(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, value))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def route_score(route_snapshot: dict[str, str]) -> float:
    if not route_snapshot:
        return 0.0
    return max(ROUTE_RANK.get(route, 0.0) for route in route_snapshot.values())


def route_is_skipped(route_snapshot: dict[str, str]) -> bool:
    return bool(route_snapshot) and not any(route in ACTIVE_ROUTES for route in route_snapshot.values())


def route_is_llm(route_snapshot: dict[str, str]) -> bool:
    return any(route in LLM_ROUTES for route in route_snapshot.values())


def suggested_s1_route(axis: dict[str, float]) -> str:
    value = float(axis.get("value_score") or 0.0)
    risk = float(axis.get("risk_score") or 0.0)
    complexity = float(axis.get("complexity_score") or 0.0)
    entity = float(axis.get("entity_salience_score") or 0.0)
    keyphrase = float(axis.get("keyphrase_score") or 0.0)
    affect = float(axis.get("affect_score") or 0.0)
    domain = float(axis.get("domain_term_score") or 0.0)
    low = float(axis.get("low_value_score") or 0.0)
    useful = max(value, entity, domain, keyphrase * 0.6, affect * 0.35)
    if low >= 6.0 and useful < 3.0:
        return "skip_or_background_only"
    if useful < 2.0 and low >= 4.0:
        return "skip_or_background_only"
    if useful >= 4.0 and (risk >= 5.5 or complexity >= 6.0):
        return "strong_llm_proposal"
    if useful >= 3.0 and (risk >= 3.0 or complexity >= 4.0):
        return "weak_llm_proposal"
    if useful >= 3.0:
        return "script_only"
    return "skip_or_background_only"


def suggested_s2_route(axis: dict[str, float]) -> str:
    value = float(axis.get("value_score") or 0.0)
    risk = float(axis.get("risk_score") or 0.0)
    complexity = float(axis.get("complexity_score") or 0.0)
    entity = float(axis.get("entity_salience_score") or 0.0)
    low = float(axis.get("low_value_score") or 0.0)
    keyphrase = float(axis.get("keyphrase_score") or 0.0)
    useful = max(value, entity, keyphrase)
    if low >= 7.0 and useful < 2.5:
        return "skip_or_background_only"
    if useful < 1.5 and low >= 4.0:
        return "skip_or_background_only"
    if useful >= 3.0 and (risk >= 5.5 or complexity >= 6.5):
        return "strong_llm_proposal"
    if useful >= 2.0:
        return "weak_llm_proposal"
    if entity >= 2.0:
        return "script_only"
    return "weak_llm_proposal"


def extract_resource_statuses(row: dict[str, Any]) -> dict[str, str]:
    status = row.get("feature_status")
    if isinstance(status, dict):
        return {str(key): str(value) for key, value in status.items()}
    return {}


def compare_row(row: dict[str, Any]) -> dict[str, Any]:
    axis = {str(key): float(value or 0.0) for key, value in (row.get("axis_scores") or {}).items()}
    route_snapshot = {str(key): str(value) for key, value in (row.get("route_snapshot") or {}).items()}
    suggested_routes = {
        "s1_memory_candidate": suggested_s1_route(axis),
        "s2_portrait_candidate": suggested_s2_route(axis),
    }
    existing_score = route_score(route_snapshot)
    suggested_score = max(ROUTE_RANK.get(route, 0.0) for route in suggested_routes.values())
    useful_score = max(
        axis.get("value_score", 0.0),
        axis.get("entity_salience_score", 0.0),
        axis.get("keyphrase_score", 0.0),
        axis.get("domain_term_score", 0.0),
    )
    review_priority = clamp(
        useful_score * 0.45
        + axis.get("risk_score", 0.0) * 0.25
        + axis.get("complexity_score", 0.0) * 0.20
        + abs(suggested_score - existing_score) * 0.35
        - axis.get("low_value_score", 0.0) * 0.20
    )
    compared = {
        "schema_version": SCHEMA_VERSION,
        "workspace_id": row.get("workspace_id"),
        "source_layer": row.get("source_layer"),
        "unit_id": row.get("unit_id"),
        "unit_type": row.get("unit_type"),
        "text_preview": row.get("text_preview", ""),
        "existing_route_snapshot": route_snapshot,
        "suggested_routes": suggested_routes,
        "axis_scores": axis,
        "useful_signal_score": round(useful_score, 3),
        "existing_route_score": round(existing_score, 3),
        "suggested_route_score": round(suggested_score, 3),
        "route_delta_score": round(suggested_score - existing_score, 3),
        "review_priority_score": round(review_priority, 3),
        "resource_status": extract_resource_statuses(row),
        "sample_buckets": [],
        "review_label_placeholder": "",
        "write_permission": False,
    }
    compared["sample_buckets"] = bucket_names(compared)
    return compared


def bucket_names(row: dict[str, Any]) -> list[str]:
    axis = row["axis_scores"]
    routes = row["existing_route_snapshot"]
    suggested = row["suggested_routes"]
    buckets: list[str] = []
    value = float(axis.get("value_score") or 0.0)
    risk = float(axis.get("risk_score") or 0.0)
    complexity = float(axis.get("complexity_score") or 0.0)
    entity = float(axis.get("entity_salience_score") or 0.0)
    keyphrase = float(axis.get("keyphrase_score") or 0.0)
    low = float(axis.get("low_value_score") or 0.0)
    domain = float(axis.get("domain_term_score") or 0.0)
    useful = max(value, entity, keyphrase, domain)
    language = row.get("resource_status", {})

    if route_is_skipped(routes) and useful >= 4.0:
        buckets.append("old_skipped_but_high_v021_value")
    if route_is_llm(routes) and useful <= 2.0 and low >= 4.0:
        buckets.append("old_llm_but_low_v021_value")
    if useful >= 4.0 and risk >= 4.0:
        buckets.append("high_risk_and_high_value")
    if complexity >= 6.0 and useful < 3.0:
        buckets.append("high_complexity_unclear_value")
    if entity >= 4.0:
        buckets.append("high_entity_salience")
    if low >= 5.0:
        buckets.append("high_low_value")
    if suggested.get("s1_memory_candidate") != suggested.get("s2_portrait_candidate"):
        buckets.append("s1_s2_route_disagreement")
    if language.get("chinese_dimlex") == "available" and domain >= 2.0:
        buckets.append("chinese_specific_term_salience")
    if not buckets:
        buckets.append("general_calibration")
    return buckets


def select_samples(rows: list[dict[str, Any]], per_bucket: int, max_samples: int) -> list[dict[str, Any]]:
    bucket_order = (
        "old_skipped_but_high_v021_value",
        "old_llm_but_low_v021_value",
        "high_risk_and_high_value",
        "high_complexity_unclear_value",
        "high_entity_salience",
        "high_low_value",
        "s1_s2_route_disagreement",
        "chinese_specific_term_salience",
        "general_calibration",
    )
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for bucket in bucket_order:
        candidates = [row for row in rows if bucket in row["sample_buckets"]]
        candidates.sort(key=lambda item: item["review_priority_score"], reverse=True)
        for row in candidates[:per_bucket]:
            row_key = (str(row.get("source_layer")), str(row.get("unit_id")))
            if row_key in seen:
                continue
            selected.append(row)
            seen.add(row_key)
            if len(selected) >= max_samples:
                return selected
    return selected


CSV_COLUMNS = [
    "workspace_id",
    "source_layer",
    "unit_id",
    "unit_type",
    "existing_route_snapshot",
    "suggested_s1",
    "suggested_s2",
    "value_score",
    "risk_score",
    "complexity_score",
    "entity_salience_score",
    "keyphrase_score",
    "low_value_score",
    "domain_term_score",
    "useful_signal_score",
    "existing_route_score",
    "suggested_route_score",
    "route_delta_score",
    "review_priority_score",
    "sample_buckets",
    "text_preview",
]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            axis = row["axis_scores"]
            writer.writerow(
                {
                    "workspace_id": row["workspace_id"],
                    "source_layer": row["source_layer"],
                    "unit_id": row["unit_id"],
                    "unit_type": row["unit_type"],
                    "existing_route_snapshot": json.dumps(row["existing_route_snapshot"], ensure_ascii=False, sort_keys=True),
                    "suggested_s1": row["suggested_routes"]["s1_memory_candidate"],
                    "suggested_s2": row["suggested_routes"]["s2_portrait_candidate"],
                    "value_score": axis.get("value_score"),
                    "risk_score": axis.get("risk_score"),
                    "complexity_score": axis.get("complexity_score"),
                    "entity_salience_score": axis.get("entity_salience_score"),
                    "keyphrase_score": axis.get("keyphrase_score"),
                    "low_value_score": axis.get("low_value_score"),
                    "domain_term_score": axis.get("domain_term_score"),
                    "useful_signal_score": row["useful_signal_score"],
                    "existing_route_score": row["existing_route_score"],
                    "suggested_route_score": row["suggested_route_score"],
                    "route_delta_score": row["route_delta_score"],
                    "review_priority_score": row["review_priority_score"],
                    "sample_buckets": "|".join(row["sample_buckets"]),
                    "text_preview": row["text_preview"],
                }
            )


def render_report(matrix_path: Path, rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> str:
    bucket_counts = Counter(bucket for row in rows for bucket in row["sample_buckets"])
    existing_counts = Counter(
        route
        for row in rows
        for route in (row["existing_route_snapshot"].values() or ["missing_route"])
    )
    suggested_counts = Counter(
        f"{target}:{route}"
        for row in rows
        for target, route in row["suggested_routes"].items()
    )
    lines = [
        "# v0.21 Mode Comparison Report",
        "",
        f"Input matrix: `{matrix_path}`",
        "",
        f"- rows compared: {len(rows)}",
        f"- sampled rows: {len(samples)}",
        "",
        "## Existing Task Route Counts",
        "",
    ]
    if existing_counts:
        lines.extend(f"- `{key}`: {value}" for key, value in existing_counts.most_common())
    else:
        lines.append("- none")
    lines.extend(["", "## Suggested Task Route Counts", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in suggested_counts.most_common())
    lines.extend(["", "## Sample Bucket Counts", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in bucket_counts.most_common())
    lines.extend(["", "## Highest Review Priority Samples", ""])
    for row in sorted(samples, key=lambda item: item["review_priority_score"], reverse=True)[:20]:
        lines.append(
            f"- `{row['source_layer']}` `{row['unit_id']}` priority={row['review_priority_score']} "
            f"buckets={','.join(row['sample_buckets'])}: {str(row['text_preview'])[:160]}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Comparison only; no route decisions are replaced.",
            "- Scores are calibration signals, not truth.",
            "- Existing S1/S2 build outputs are not modified.",
            "- No provider calls, durable memory writes, graph writes, or S3 work.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, str]:
    matrix_path = Path(args.matrix).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else matrix_path.parent / DEFAULT_OUTPUT_NAME
    rows = [compare_row(row) for row in read_jsonl(matrix_path)]
    samples = select_samples(rows, per_bucket=args.per_bucket, max_samples=args.max_samples)

    rows_path = output_dir / "v021_mode_comparison_rows.jsonl"
    samples_path = output_dir / "v021_mode_comparison_samples.jsonl"
    table_path = output_dir / "v021_mode_comparison_table.csv"
    report_path = output_dir / "v021_mode_comparison_report.md"
    manifest_path = output_dir / "v021_mode_comparison_manifest.json"

    write_jsonl(rows_path, rows)
    write_jsonl(samples_path, samples)
    write_csv(table_path, rows)
    write_text(report_path, render_report(matrix_path, rows, samples))
    write_json(
        manifest_path,
        {
            "schema_version": "routing.v021_mode_comparison_manifest.v0.1",
            "input_matrix": str(matrix_path),
            "output_dir": str(output_dir),
            "row_count": len(rows),
            "sample_count": len(samples),
            "provider_calls": False,
            "route_decisions_replaced": False,
            "write_permission": False,
            "outputs": {
                "rows": str(rows_path),
                "samples": str(samples_path),
                "table": str(table_path),
                "report": str(report_path),
                "manifest": str(manifest_path),
            },
        },
    )
    return {
        "rows": str(rows_path),
        "samples": str(samples_path),
        "table": str(table_path),
        "report": str(report_path),
        "manifest": str(manifest_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare v0.21 feature matrix rows with existing route snapshots.")
    parser.add_argument("--matrix", required=True, help="Path to v021_feature_matrix.jsonl.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults beside matrix.")
    parser.add_argument("--per-bucket", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=80)
    return parser


def main() -> int:
    outputs = run(build_arg_parser().parse_args())
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
