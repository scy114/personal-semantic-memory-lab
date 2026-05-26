"""Compare v0.3 graph query evaluation runs across retrieval modes.

This reporter compares outputs from tools.step2.query_runner plus
graph_query_eval_reporter. It surfaces retrieval contribution and noise; it
does not judge answer correctness or evidence support.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.graph.graph_query_eval_reporter import analyze_query_dir, read_json, write_json, write_jsonl


def parse_run_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Run must be LABEL=PATH, got: {value}")
    label, path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"Missing run label in: {value}")
    return label, Path(path).resolve()


def load_run_rows(label: str, run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and (child / "route_decision.json").exists():
            row = analyze_query_dir(child)
            row["mode_label"] = label
            row["run_dir"] = str(run_dir)
            row["query_dir"] = str(child)
            rows.append(row)
    return rows


def value_rank(value: str) -> int:
    order = {
        "low": 0,
        "weak_or_noisy": 1,
        "organizes_known_context": 2,
        "adds_evidence_with_review": 3,
        "adds_evidence": 4,
    }
    return order.get(value, 0)


def branch_count(row: dict[str, Any], branch: str) -> int:
    counts = row.get("branch_counts", {}) or {}
    try:
        return int(counts.get(branch) or 0)
    except (TypeError, ValueError):
        return 0


def compare_rows(rows: list[dict[str, Any]], baseline_label: str) -> list[dict[str, Any]]:
    by_query: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        by_query.setdefault(str(row["query_id"]), {})[str(row["mode_label"])] = row

    comparisons: list[dict[str, Any]] = []
    for query_id, mode_rows in sorted(by_query.items()):
        baseline = mode_rows.get(baseline_label) or next(iter(mode_rows.values()))
        for label, row in sorted(mode_rows.items()):
            comparisons.append(
                {
                    "query_id": query_id,
                    "mode_label": label,
                    "question": row.get("question", ""),
                    "graph_value": row.get("graph_value", ""),
                    "graph_value_delta_vs_baseline": value_rank(str(row.get("graph_value"))) - value_rank(str(baseline.get("graph_value"))),
                    "supported_relation_delta_vs_baseline": int(row.get("supported_relation_count") or 0) - int(baseline.get("supported_relation_count") or 0),
                    "cautious_relation_delta_vs_baseline": int(row.get("cautious_relation_count") or 0) - int(baseline.get("cautious_relation_count") or 0),
                    "graph_only_evidence_delta_vs_baseline": int(row.get("graph_only_evidence_count") or 0) - int(baseline.get("graph_only_evidence_count") or 0),
                    "activated_community_delta_vs_baseline": int(row.get("activated_community_count") or 0) - int(baseline.get("activated_community_count") or 0),
                    "community_context_delta_vs_baseline": int(row.get("community_context_count") or 0) - int(baseline.get("community_context_count") or 0),
                    "review_heavy_community_delta_vs_baseline": int(row.get("review_heavy_community_count") or 0) - int(baseline.get("review_heavy_community_count") or 0),
                    "noisy_path_delta_vs_baseline": int(row.get("noisy_evidence_path_count") or 0) - int(baseline.get("noisy_evidence_path_count") or 0),
                    "graph_warning_delta_vs_baseline": int(row.get("graph_warning_count") or 0) - int(baseline.get("graph_warning_count") or 0),
                    "selected_context_count": row.get("selected_context_count", 0),
                    "graph_context_count": row.get("graph_context_count", 0),
                    "supported_relation_count": row.get("supported_relation_count", 0),
                    "cautious_relation_count": row.get("cautious_relation_count", 0),
                    "graph_only_evidence_count": row.get("graph_only_evidence_count", 0),
                    "community_report_match_count": branch_count(row, "community_report_match"),
                    "activated_community_count": row.get("activated_community_count", 0),
                    "community_context_count": row.get("community_context_count", 0),
                    "review_heavy_community_count": row.get("review_heavy_community_count", 0),
                    "noisy_evidence_path_count": row.get("noisy_evidence_path_count", 0),
                    "generic_or_warning_relation_count": row.get("generic_or_warning_relation_count", 0),
                    "graph_rerank_mode": row.get("graph_rerank_mode", ""),
                    "cross_encoder_reranked_count": row.get("cross_encoder_reranked_count", 0),
                    "embedding_graph_unit_count": branch_count(row, "embedding_graph_units"),
                    "bm25_graph_unit_count": branch_count(row, "bm25_graph_units"),
                    "seed_node_match_count": branch_count(row, "seed_node_match"),
                    "graph_neighborhood_count": branch_count(row, "graph_neighborhood_bfs_1hop"),
                    "relation_samples": row.get("relation_samples", []),
                    "activated_community_samples": row.get("activated_community_samples", []),
                    "graph_is_not_proof": True,
                    "support_status": "not_checked",
                }
            )
    return comparisons


def summarize(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]], run_dirs: dict[str, Path]) -> dict[str, Any]:
    by_mode: dict[str, dict[str, Any]] = {}
    for label in sorted(run_dirs):
        mode_rows = [row for row in rows if row["mode_label"] == label]
        value_counts: dict[str, int] = {}
        for row in mode_rows:
            value = str(row.get("graph_value") or "unknown")
            value_counts[value] = value_counts.get(value, 0) + 1
        by_mode[label] = {
            "query_count": len(mode_rows),
            "graph_value_counts": value_counts,
            "supported_relation_total": sum(int(row.get("supported_relation_count") or 0) for row in mode_rows),
            "cautious_relation_total": sum(int(row.get("cautious_relation_count") or 0) for row in mode_rows),
            "graph_only_evidence_total": sum(int(row.get("graph_only_evidence_count") or 0) for row in mode_rows),
            "community_report_match_total": sum(branch_count(row, "community_report_match") for row in mode_rows),
            "activated_community_total": sum(int(row.get("activated_community_count") or 0) for row in mode_rows),
            "community_context_total": sum(int(row.get("community_context_count") or 0) for row in mode_rows),
            "review_heavy_community_total": sum(int(row.get("review_heavy_community_count") or 0) for row in mode_rows),
            "noisy_path_total": sum(int(row.get("noisy_evidence_path_count") or 0) for row in mode_rows),
            "embedding_graph_unit_total": sum(branch_count(row, "embedding_graph_units") for row in mode_rows),
            "cross_encoder_reranked_total": sum(int(row.get("cross_encoder_reranked_count") or 0) for row in mode_rows),
        }
    return {
        "schema_version": "graph_v03.query_mode_comparison.v0.1",
        "run_dirs": {label: str(path) for label, path in run_dirs.items()},
        "mode_summary": by_mode,
        "comparison_row_count": len(comparisons),
        "graph_is_not_proof": True,
        "support_status": "not_checked",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "query_id",
        "mode_label",
        "graph_value",
        "graph_value_delta_vs_baseline",
        "supported_relation_count",
        "supported_relation_delta_vs_baseline",
        "cautious_relation_count",
        "cautious_relation_delta_vs_baseline",
        "graph_only_evidence_count",
        "graph_only_evidence_delta_vs_baseline",
        "community_report_match_count",
        "activated_community_count",
        "activated_community_delta_vs_baseline",
        "community_context_count",
        "community_context_delta_vs_baseline",
        "review_heavy_community_count",
        "review_heavy_community_delta_vs_baseline",
        "noisy_evidence_path_count",
        "noisy_path_delta_vs_baseline",
        "graph_rerank_mode",
        "cross_encoder_reranked_count",
        "embedding_graph_unit_count",
        "bm25_graph_unit_count",
        "seed_node_match_count",
        "graph_neighborhood_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(path: Path, summary: dict[str, Any], comparisons: list[dict[str, Any]], baseline_label: str) -> None:
    lines = [
        "# v0.3 Graph Query Mode Comparison",
        "",
        f"- baseline: `{baseline_label}`",
        "- graph_is_not_proof: `true`",
        "- support_status: `not_checked`",
        "",
        "## Mode Summary",
        "",
    ]
    for label, stats in summary["mode_summary"].items():
        lines.extend(
            [
                f"### {label}",
                "",
                f"- queries: {stats['query_count']}",
                f"- graph value counts: `{json.dumps(stats['graph_value_counts'], ensure_ascii=False, sort_keys=True)}`",
                f"- supported relations: {stats['supported_relation_total']}",
                f"- cautious relations: {stats['cautious_relation_total']}",
                f"- graph-only evidence refs: {stats['graph_only_evidence_total']}",
                f"- community report matches: {stats['community_report_match_total']}",
                f"- activated communities: {stats['activated_community_total']}",
                f"- community context rows: {stats['community_context_total']}",
                f"- review-heavy community rows: {stats['review_heavy_community_total']}",
                f"- noisy paths: {stats['noisy_path_total']}",
                f"- embedding graph unit hits: {stats['embedding_graph_unit_total']}",
                f"- cross-encoder reranked items: {stats['cross_encoder_reranked_total']}",
                "",
            ]
        )

    lines.extend(["## Query-Level Comparison", ""])
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in comparisons:
        by_query.setdefault(str(row["query_id"]), []).append(row)
    for query_id, rows in by_query.items():
        lines.extend([f"### {query_id}", ""])
        if rows:
            lines.append(f"- question: {rows[0].get('question', '')}")
        for row in rows:
            lines.append(
                "- "
                f"`{row['mode_label']}` value=`{row['graph_value']}` "
                f"supported={row['supported_relation_count']} "
                f"cautious={row['cautious_relation_count']} "
                f"graph_only_refs={row['graph_only_evidence_count']} "
                f"communities={row['activated_community_count']}/{row['community_context_count']} "
                f"review_heavy_communities={row['review_heavy_community_count']} "
                f"noisy_paths={row['noisy_evidence_path_count']} "
                f"embedding_units={row['embedding_graph_unit_count']} "
                f"rerank={row.get('graph_rerank_mode', '')}"
            )
        lines.append("")
        for row in rows:
            samples = row.get("activated_community_samples") or []
            if samples:
                lines.append(f"- `{row['mode_label']}` activated community samples: " + "; ".join(str(item) for item in samples[:3]))
        if any(row.get("activated_community_samples") for row in rows):
            lines.append("")

    lines.extend(
        [
            "## Boundary",
            "",
            "- This report compares retrieval modes, not final answer correctness.",
            "- Graph metrics, graph relations, and embedding similarity are retrieval signals only.",
            "- Evidence refs still require downstream support checking before factual use.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_comparison(run_args: list[str], output_dir: Path, baseline_label: str | None = None) -> dict[str, Any]:
    run_dirs = dict(parse_run_arg(value) for value in run_args)
    if not run_dirs:
        raise ValueError("At least one --run LABEL=PATH is required.")
    baseline = baseline_label or next(iter(run_dirs))
    rows: list[dict[str, Any]] = []
    for label, run_dir in run_dirs.items():
        rows.extend(load_run_rows(label, run_dir))
    comparisons = compare_rows(rows, baseline)
    summary = summarize(rows, comparisons, run_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "graph_query_mode_rows.jsonl", rows)
    write_jsonl(output_dir / "graph_query_mode_comparison_rows.jsonl", comparisons)
    write_csv(output_dir / "graph_query_mode_comparison_table.csv", comparisons)
    write_json(output_dir / "graph_query_mode_comparison_summary.json", summary)
    write_report(output_dir / "graph_query_mode_comparison_report.md", summary, comparisons, baseline)
    return {"output_dir": str(output_dir), "mode_count": len(run_dirs), "comparison_row_count": len(comparisons)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare v0.3 graph query runs across retrieval modes.")
    parser.add_argument("--run", action="append", required=True, help="Run in LABEL=PATH form. Repeat for multiple modes.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--baseline-label", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_comparison(args.run, Path(args.output_dir), args.baseline_label)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
