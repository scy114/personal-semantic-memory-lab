"""Summarize v0.3 graph query runs for manual quality review.

The reporter reads outputs produced by tools.step2.query_runner and compares
the S2 selected context with the graph answer context. It does not score truth;
it only surfaces retrieval contribution, noise, and audit burden.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None and str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if item is not None and str(item)]
    text = str(value)
    return [text] if text else []


def evidence_refs(rows: list[dict[str, Any]]) -> set[str]:
    refs: set[str] = set()
    for row in rows:
        refs.update(string_list(row.get("evidence_refs")))
    return refs


def warning_count(rows: list[dict[str, Any]]) -> int:
    return sum(len(string_list(row.get("warnings"))) for row in rows)


def relation_sample(rows: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    return [str(row.get("summary") or "") for row in rows[:limit] if row.get("summary")]


def community_title_sample(rows: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    titles: list[str] = []
    for row in rows:
        title = str(row.get("community_title") or row.get("title") or "").strip()
        if title and title not in titles:
            titles.append(title)
        if len(titles) >= limit:
            break
    return titles


def noisy_path_count(paths: list[dict[str, Any]]) -> int:
    weak_terms = {"we", "this", "it", "they", "them", "a place", "spot"}
    count = 0
    for row in paths:
        summary = str(row.get("summary") or "").lower()
        labels = [part.strip() for part in summary.split("->")]
        if any(label in weak_terms for label in labels):
            count += 1
    return count


def analyze_query_dir(query_dir: Path) -> dict[str, Any]:
    selected = read_jsonl(query_dir / "selected_context.jsonl")
    graph_context = read_jsonl(query_dir / "graph_context.jsonl")
    graph_answer = read_json(query_dir / "graph_answer_context.json", default={})
    graph_package = read_json(query_dir / "graph_retrieval_package.json", default={})
    route_decision = read_json(query_dir / "route_decision.json", default={})
    query_manifest = read_json(query_dir / "query_manifest.json", default={})

    supported = graph_answer.get("supported_graph_relations", []) or []
    cautious = graph_answer.get("cautious_graph_relations", []) or []
    paths = graph_answer.get("evidence_paths", []) or []
    noisy_navigation_paths = graph_answer.get("noisy_navigation_paths", []) or []
    unresolved = graph_answer.get("unresolved_or_unsafe_graph_items", []) or []
    ranked_items = graph_package.get("ranked_items", []) or []
    graph_branch = graph_package.get("graph_branch") or {}
    activated_communities = graph_branch.get("activated_communities", []) or []
    community_context = [row for row in graph_context if row.get("kind") == "v03_ranked_community_report"]
    rerank_policy = ((graph_package.get("retrieval_policy") or {}).get("rerank") or {})
    branch_counts = {
        name: len(rows or [])
        for name, rows in (graph_package.get("branches") or {}).items()
    }
    cross_encoder_reranked = sum(1 for row in ranked_items if "cross_encoder_rerank_score" in row)
    selected_refs = evidence_refs(selected)
    graph_refs = evidence_refs(graph_context)
    graph_only_refs = sorted(graph_refs - selected_refs)
    selected_only_refs = sorted(selected_refs - graph_refs)
    noisy_paths = len(noisy_navigation_paths) or noisy_path_count(paths)
    generic_count = sum(
        1
        for row in supported + cautious + graph_context
        if "related_to_generic" in str(row.get("summary") or "")
        or any("generic_relation" in warning for warning in string_list(row.get("warnings")))
    )
    relation_count = len(supported) + len(cautious)
    graph_value = "low"
    if supported and graph_only_refs and (generic_count or len(cautious) > 0):
        graph_value = "adds_evidence_with_review"
    elif supported and graph_only_refs:
        graph_value = "adds_evidence"
    elif supported:
        graph_value = "organizes_known_context"
    elif graph_context or noisy_paths:
        graph_value = "weak_or_noisy"

    return {
        "query_id": route_decision.get("query_id") or query_dir.name,
        "question": (
            route_decision.get("question")
            or (route_decision.get("query") or {}).get("question")
            or (query_manifest.get("query") or {}).get("question")
            or graph_package.get("query", "")
        ),
        "route": route_decision.get("route", ""),
        "selected_context_count": len(selected),
        "graph_context_count": len(graph_context),
        "supported_relation_count": len(supported),
        "cautious_relation_count": len(cautious),
        "evidence_path_count": len(paths),
        "unresolved_graph_item_count": len(unresolved),
        "ranked_item_count": len(ranked_items),
        "graph_rerank_mode": rerank_policy.get("mode", "unknown"),
        "graph_rerank_used": bool(rerank_policy.get("used")),
        "cross_encoder_reranked_count": cross_encoder_reranked,
        "branch_counts": branch_counts,
        "community_report_match_count": branch_counts.get("community_report_match", 0),
        "activated_community_count": len(activated_communities),
        "community_context_count": len(community_context),
        "review_heavy_community_count": sum(1 for row in community_context if row.get("activation_quality") == "review_heavy"),
        "selected_evidence_count": len(selected_refs),
        "graph_evidence_count": len(graph_refs),
        "graph_only_evidence_count": len(graph_only_refs),
        "selected_only_evidence_count": len(selected_only_refs),
        "generic_or_warning_relation_count": generic_count,
        "noisy_evidence_path_count": noisy_paths,
        "graph_warning_count": warning_count(graph_context),
        "graph_value": graph_value,
        "relation_samples": relation_sample(supported + cautious),
        "activated_community_samples": community_title_sample(community_context),
        "graph_only_evidence_refs": graph_only_refs[:20],
        "selected_only_evidence_refs": selected_only_refs[:20],
        "graph_is_not_proof": True,
        "support_status": "not_checked",
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "query_id",
        "graph_value",
        "selected_context_count",
        "graph_context_count",
        "supported_relation_count",
        "cautious_relation_count",
        "evidence_path_count",
        "graph_rerank_mode",
        "cross_encoder_reranked_count",
        "community_report_match_count",
        "activated_community_count",
        "community_context_count",
        "review_heavy_community_count",
        "graph_only_evidence_count",
        "generic_or_warning_relation_count",
        "noisy_evidence_path_count",
        "graph_warning_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_report(path: Path, run_summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    lines = [
        "# v0.3 Graph Query Evaluation",
        "",
        f"- workspace: `{run_summary.get('workspace', '')}`",
        f"- output: `{run_summary.get('output', '')}`",
        f"- questions: {len(rows)}",
        "- graph_is_not_proof: `true`",
        "- support_status: `not_checked`",
        "",
        "## 总体判断",
        "",
    ]
    value_counts: dict[str, int] = {}
    for row in rows:
        value_counts[row["graph_value"]] = value_counts.get(row["graph_value"], 0) + 1
    for key in sorted(value_counts):
        lines.append(f"- `{key}`: {value_counts[key]}")
    lines.extend(
        [
            "",
            "## 怎么读",
            "",
            "- `adds_evidence`: 图分支带来了主 S2 selected context 之外的 evidence refs。",
            "- `adds_evidence_with_review`: 图分支有增量证据，但包含 generic/谨慎关系，需要 review。",
            "- `adds_evidence_with_noisy_paths`: 图分支有增量证据，但 evidence path 出现代词/弱节点桥，不能直接解释。",
            "- `organizes_known_context`: 图分支主要把已命中的 evidence 组织成关系。",
            "- `weak_or_noisy`: 有图上下文但有效关系弱，或主要是噪声/谨慎项。",
            "- 所有图关系仍是候选关系，不是支持证明。",
            "",
            "## Query Samples",
            "",
        ]
    )
    for row in rows:
        lines.extend(
            [
                f"### {row['query_id']}",
                "",
                f"- question: {row.get('question', '')}",
                f"- graph value: `{row['graph_value']}`",
                f"- selected context: {row['selected_context_count']}",
                f"- graph context: {row['graph_context_count']}",
                f"- supported/cautious relations: {row['supported_relation_count']} / {row['cautious_relation_count']}",
                f"- evidence paths: {row['evidence_path_count']}",
                f"- graph-only evidence refs: {row['graph_only_evidence_count']}",
                f"- activated communities/context/review-heavy: {row['activated_community_count']} / {row['community_context_count']} / {row['review_heavy_community_count']}",
                f"- generic/warning relation count: {row['generic_or_warning_relation_count']}",
                f"- noisy evidence path count: {row['noisy_evidence_path_count']}",
                f"- graph warning count: {row['graph_warning_count']}",
                "",
            ]
        )
        if row["relation_samples"]:
            lines.append("Relation samples:")
            for sample in row["relation_samples"]:
                lines.append(f"- {sample}")
            lines.append("")
        if row["activated_community_samples"]:
            lines.append("Activated community samples:")
            for sample in row["activated_community_samples"]:
                lines.append(f"- {sample}")
            lines.append("")
        if row["graph_only_evidence_refs"]:
            lines.append("Graph-only evidence refs:")
            lines.append("- " + ", ".join(row["graph_only_evidence_refs"][:10]))
            lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "- This report evaluates retrieval contribution, not answer correctness.",
            "- Graph-only evidence is a candidate for inspection, not proof.",
            "- Generic/high-warning relations should be reviewed before being trusted.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_report(input_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = (output_dir or input_dir).resolve()
    run_summary = read_json(input_dir / "run-summary.json", default={})
    rows = []
    for child in sorted(input_dir.iterdir()):
        if child.is_dir() and (child / "route_decision.json").exists():
            rows.append(analyze_query_dir(child))
    write_jsonl(output_dir / "graph_query_eval_rows.jsonl", rows)
    write_json(output_dir / "graph_query_eval_summary.json", {"run_summary": run_summary, "rows": rows})
    write_csv(output_dir / "graph_query_eval_table.csv", rows)
    write_report(output_dir / "graph_query_eval_report.md", run_summary, rows)
    return {"input_dir": str(input_dir), "output_dir": str(output_dir), "query_count": len(rows)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v0.3 graph query evaluation outputs.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_report(Path(args.input_dir), Path(args.output_dir) if args.output_dir else None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
