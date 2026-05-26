"""Compare salience routing modes without changing route decisions."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


MODE_NAMES = (
    "baseline_v0_1_route",
    "seed_lexicon",
    "seed_plus_entity",
    "complexity_assist",
    "tfidf_salience",
    "external_wheels",
    "ensemble_profile",
)
TARGET_PROFILES = (
    "s1_memory_candidate",
    "s2_portrait_candidate",
    "graph_relation_candidate",
    "support_check_candidate",
)
ROUTE_BASELINE_SCORES = {
    "skip_or_background_only": 0.0,
    "split_or_segment_first": 1.5,
    "script_only": 2.5,
    "weak_llm_proposal": 5.5,
    "human_review": 6.5,
    "strong_llm_proposal": 8.0,
}
FIRST_PERSON_PATTERNS = (
    r"\bi\b",
    r"\bi'm\b",
    r"\bi've\b",
    r"\bi'll\b",
    r"\bmy\b",
    r"\bwe\b",
    r"\bwe're\b",
    r"\bwe've\b",
    r"\bour\b",
    r"我",
    r"我们",
    r"本人",
)
S1_MEMORY_CUE_TERMS = {
    "ad campaign",
    "business",
    "campaign",
    "customer",
    "customers",
    "deadline",
    "goal",
    "job",
    "launched",
    "learned",
    "location",
    "lost",
    "need",
    "offer",
    "offering",
    "plan",
    "project",
    "searching",
    "started",
    "store",
    "studio",
    "work",
    "working",
    "want",
    "wants",
    "失业",
    "工作",
    "项目",
    "计划",
    "准备",
    "正在",
    "开",
    "开店",
    "上线",
    "发布",
    "搬家",
    "学习",
    "需要",
    "想要",
    "喜欢",
    "讨厌",
}


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def clamp(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, value))


def normalized(value: float, cap: float) -> float:
    if cap <= 0:
        return 0.0
    return round(clamp((value / cap) * 10.0), 3)


def baseline_score(routes: dict[str, str]) -> float:
    if not routes:
        return 0.0
    return round(max(ROUTE_BASELINE_SCORES.get(route, 0.0) for route in routes.values()), 3)


def has_first_person_signal(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pattern, lower) for pattern in FIRST_PERSON_PATTERNS)


def s1_explicit_memory_signal(text: str, seed: float, tfidf_score: float, external: float, evidence: float, low: float) -> float:
    if low >= 5.0:
        return 0.0
    lower = text.lower()
    has_source = has_first_person_signal(text) or evidence > 0
    if not has_source:
        return 0.0
    cue_hits = sum(1 for cue in S1_MEMORY_CUE_TERMS if cue in lower)
    if cue_hits == 0 and max(seed, tfidf_score, external) < 4.0:
        return 0.0
    cue_score = min(1.2, cue_hits * 0.4)
    salience_score = min(1.0, max(seed, tfidf_score, external) / 6.0)
    return round(min(2.2, 0.8 + cue_score + salience_score), 3)


def route_is_skipped(routes: dict[str, str]) -> bool:
    return bool(routes) and all(route in {"skip_or_background_only", "split_or_segment_first"} for route in routes.values())


def route_is_active(routes: dict[str, str]) -> bool:
    return any(route in {"weak_llm_proposal", "strong_llm_proposal", "human_review"} for route in routes.values())


def external_score(feature_summary: dict[str, Any]) -> float:
    metrics = feature_summary.get("external_metrics") or {}
    score = 0.0
    wordfreq = metrics.get("wordfreq") or {}
    min_zipf = wordfreq.get("min_zipf")
    if isinstance(min_zipf, (int, float)):
        score += clamp((4.0 - float(min_zipf)) * 1.6, 0.0, 2.5)
    zh_min_zipf = wordfreq.get("zh_min_zipf")
    if isinstance(zh_min_zipf, (int, float)):
        score += clamp((4.0 - float(zh_min_zipf)) * 1.6, 0.0, 2.5)
    vader = metrics.get("vader") or {}
    compound = vader.get("compound")
    if isinstance(compound, (int, float)):
        score += abs(float(compound)) * 1.0
    textblob = metrics.get("textblob") or {}
    polarity = textblob.get("polarity")
    subjectivity = textblob.get("subjectivity")
    if isinstance(polarity, (int, float)):
        score += abs(float(polarity)) * 0.75
    if isinstance(subjectivity, (int, float)):
        score += float(subjectivity) * 0.5
    textstat = metrics.get("textstat") or {}
    difficult_words = textstat.get("difficult_words")
    grade = textstat.get("flesch_kincaid_grade")
    if isinstance(difficult_words, (int, float)):
        score += clamp(float(difficult_words) / 4.0, 0.0, 1.5)
    if isinstance(grade, (int, float)) and grade > 8:
        score += clamp((float(grade) - 8.0) / 6.0, 0.0, 1.0)
    empath = metrics.get("empath") or {}
    top_categories = empath.get("top_categories") or []
    if isinstance(top_categories, list):
        score += clamp(sum(float(item.get("score", 0.0) or 0.0) for item in top_categories) * 4.0, 0.0, 1.2)
    yake = metrics.get("yake") or {}
    keywords = yake.get("keywords") or []
    if isinstance(keywords, list):
        best_keyword_score = min((float(item.get("score", 1.0) or 1.0) for item in keywords), default=1.0)
        score += clamp((0.35 - best_keyword_score) * 4.0, 0.0, 1.2)
    lexical = metrics.get("lexical_diversity") or {}
    mtld = lexical.get("mtld")
    if isinstance(mtld, (int, float)) and mtld > 20:
        score += clamp((float(mtld) - 20.0) / 80.0, 0.0, 0.5)
    return round(clamp(score), 3)


def top_terms_for_mode(row: dict[str, Any], mode: str) -> list[str]:
    if mode == "seed_lexicon":
        return [
            str(hit.get("term"))
            for hit in row.get("matched_terms", [])
            if hit.get("group") not in {"dynamic_entity_terms", "low_value_terms"}
        ][:8]
    if mode == "seed_plus_entity":
        return [str(hit.get("term")) for hit in row.get("matched_terms", []) if hit.get("group") != "low_value_terms"][:8]
    if mode == "tfidf_salience":
        tfidf = (row.get("feature_summary", {}).get("external_corpus_metrics") or {}).get("sklearn_tfidf") or {}
        return [str(item.get("term")) for item in tfidf.get("top_terms", [])[:8]]
    if mode == "external_wheels":
        metrics = row.get("feature_summary", {}).get("external_metrics") or {}
        terms: list[str] = []
        terms.extend(str(item.get("term")) for item in (metrics.get("yake") or {}).get("keywords", [])[:5])
        terms.extend(str(item.get("category")) for item in (metrics.get("empath") or {}).get("top_categories", [])[:5])
        return [term for term in terms if term and term != "None"][:8]
    return []


def score_to_route(score: float, low_value_score: float, value_support: float) -> str:
    if low_value_score >= 5.0 and value_support < 5.0:
        return "skip_or_background_only"
    if score < 2.0:
        return "skip_or_background_only"
    if score < 4.5:
        return "script_only"
    if score < 7.0:
        return "weak_llm_proposal"
    return "strong_llm_proposal"


def score_to_profile_route(target: str, score: float, low_value_score: float, value_support: float) -> str:
    if target != "s1_memory_candidate":
        return score_to_route(score, low_value_score, value_support)
    if low_value_score >= 5.0 and value_support < 5.0:
        return "skip_or_background_only"
    if score < 2.0:
        return "skip_or_background_only"
    if score < 5.5:
        return "script_only"
    if score < 7.5:
        return "weak_llm_proposal"
    return "strong_llm_proposal"


def compare_row(row: dict[str, Any]) -> dict[str, Any]:
    group_scores = row.get("group_scores") or {}
    salience_scores = row.get("salience_scores") or {}
    salience_dimensions = row.get("salience_dimensions") or {}
    feature_summary = row.get("feature_summary") or {}
    routes = row.get("v0_1_routes") or {}

    low_raw = float(salience_dimensions.get("low_value_score", salience_scores.get("low_value_score", 0.0)) or 0.0)
    seed_raw = sum(float(value) for group, value in group_scores.items() if group not in {"dynamic_entity_terms", "low_value_terms"} and value > 0)
    entity_raw = float(group_scores.get("dynamic_entity_terms", 0.0) or 0.0)
    value_raw = float(salience_dimensions.get("modeling_value_score", salience_scores.get("value_score", 0.0)) or 0.0)
    risk_raw = float(salience_dimensions.get("risk_score", salience_scores.get("risk_score", 0.0)) or 0.0)
    constraint_raw = float(salience_dimensions.get("constraint_score", group_scores.get("constraint_terms", 0.0)) or 0.0)
    evidence_raw = float(salience_dimensions.get("evidence_directness_score", group_scores.get("evidence_directness_terms", 0.0)) or 0.0)
    complexity = float(salience_dimensions.get("processing_complexity_score", salience_scores.get("complexity_score", 0.0)) or 0.0)
    tfidf = float(
        ((feature_summary.get("external_corpus_metrics") or {}).get("sklearn_tfidf") or {}).get("sum_tfidf", 0.0)
        or salience_scores.get("external_tfidf_score", 0.0)
        or 0.0
    )
    external = external_score(feature_summary)

    seed = normalized(seed_raw, 25.0)
    value = normalized(value_raw, 15.0)
    entity = normalized(entity_raw, 25.0)
    seed_entity = normalized(seed_raw + entity_raw, 35.0)
    tfidf_score = normalized(tfidf, 8.0)
    risk = normalized(risk_raw, 10.0)
    constraint = normalized(constraint_raw, 10.0)
    evidence = normalized(evidence_raw, 10.0)
    low = normalized(low_raw, 10.0)
    baseline = baseline_score(routes)
    explicit_s1 = s1_explicit_memory_signal(str(row.get("text_preview", "")), seed, tfidf_score, external, evidence, low)
    value_support = max(seed + tfidf_score + external + entity * 0.4, normalized(value_raw, 15.0))
    low_penalty = low if value_support < 5.0 else low * 0.25

    s1 = clamp(0.25 * seed + 0.25 * value + 0.15 * entity + 0.15 * tfidf_score + 0.10 * external + 0.10 * complexity + explicit_s1 - 0.55 * low_penalty)
    s2 = clamp(0.20 * seed + 0.25 * value + 0.20 * entity + 0.10 * tfidf_score + 0.10 * external + 0.10 * risk + 0.05 * constraint + 0.05 * complexity - 0.60 * low_penalty)
    graph = clamp(0.15 * seed + 0.15 * value + 0.40 * entity + 0.15 * tfidf_score + 0.05 * external + 0.10 * complexity - 0.35 * low_penalty)
    support = clamp(0.15 * seed + 0.20 * value + 0.10 * entity + 0.15 * tfidf_score + 0.15 * external + 0.15 * complexity + 0.10 * evidence - 0.50 * low_penalty)
    ensemble = max(s1, s2, graph, support)

    mode_scores = {
        "baseline_v0_1_route": baseline,
        "seed_lexicon": round(seed, 3),
        "seed_plus_entity": round(seed_entity, 3),
        "complexity_assist": round(complexity, 3),
        "tfidf_salience": round(tfidf_score, 3),
        "external_wheels": round(external, 3),
        "ensemble_profile": round(ensemble, 3),
    }
    profile_scores = {
        "s1_memory_candidate": round(s1, 3),
        "s2_portrait_candidate": round(s2, 3),
        "graph_relation_candidate": round(graph, 3),
        "support_check_candidate": round(support, 3),
    }
    suggested_routes = {
        target: score_to_profile_route(target, score, low_raw, value_support)
        for target, score in profile_scores.items()
    }
    values = list(mode_scores.values())
    top_terms_by_mode = {mode: top_terms_for_mode(row, mode) for mode in MODE_NAMES}
    return {
        "schema_version": "routing.salience_mode_comparison.v0.1",
        "workspace_id": row.get("workspace_id"),
        "row_id": row.get("row_id"),
        "unit_type": row.get("unit_type"),
        "text_preview": row.get("text_preview", ""),
        "v0_1_routes": routes,
        "mode_scores": mode_scores,
        "profile_scores": profile_scores,
        "suggested_routes": suggested_routes,
        "top_terms_by_mode": top_terms_by_mode,
        "raw_signal_summary": {
            "modeling_value_raw": round(value_raw, 3),
            "seed_raw": round(seed_raw, 3),
            "dynamic_entity_raw": round(entity_raw, 3),
            "low_value_raw": round(low_raw, 3),
            "risk_raw": round(risk_raw, 3),
            "constraint_raw": round(constraint_raw, 3),
            "evidence_directness_raw": round(evidence_raw, 3),
            "tfidf_raw": round(tfidf, 3),
            "explicit_s1_memory_signal": explicit_s1,
        },
        "disagreement_score": round(max(values) - min(values), 3) if values else 0.0,
        "sample_buckets": [],
        "review_label_placeholder": "",
        "write_permission": False,
    }


def select_bucket(rows: list[dict[str, Any]], bucket: str, limit: int) -> list[dict[str, Any]]:
    if bucket == "baseline_skipped_ensemble_high":
        candidates = [r for r in rows if route_is_skipped(r["v0_1_routes"]) and r["mode_scores"]["ensemble_profile"] >= 6.0]
        return sorted(candidates, key=lambda r: r["mode_scores"]["ensemble_profile"], reverse=True)[:limit]
    if bucket == "baseline_active_ensemble_low":
        candidates = [r for r in rows if route_is_active(r["v0_1_routes"]) and r["mode_scores"]["ensemble_profile"] <= 2.5]
        return sorted(candidates, key=lambda r: r["mode_scores"]["baseline_v0_1_route"], reverse=True)[:limit]
    if bucket == "tfidf_high_seed_low":
        candidates = [r for r in rows if r["mode_scores"]["tfidf_salience"] >= 6.0 and r["mode_scores"]["seed_lexicon"] <= 2.5]
        return sorted(candidates, key=lambda r: r["mode_scores"]["tfidf_salience"], reverse=True)[:limit]
    if bucket == "seed_high_tfidf_low":
        candidates = [r for r in rows if r["mode_scores"]["seed_lexicon"] >= 6.0 and r["mode_scores"]["tfidf_salience"] <= 2.5]
        return sorted(candidates, key=lambda r: r["mode_scores"]["seed_lexicon"], reverse=True)[:limit]
    if bucket == "complexity_high":
        candidates = [r for r in rows if r["mode_scores"]["complexity_assist"] >= 6.0]
        return sorted(candidates, key=lambda r: r["mode_scores"]["complexity_assist"], reverse=True)[:limit]
    if bucket == "dynamic_entity_high":
        candidates = [r for r in rows if normalized(r["raw_signal_summary"]["dynamic_entity_raw"], 25.0) >= 6.0]
        return sorted(candidates, key=lambda r: r["raw_signal_summary"]["dynamic_entity_raw"], reverse=True)[:limit]
    if bucket == "low_value_high":
        candidates = [r for r in rows if r["raw_signal_summary"]["low_value_raw"] >= 5.0]
        return sorted(candidates, key=lambda r: r["raw_signal_summary"]["low_value_raw"], reverse=True)[:limit]
    if bucket == "disagreement_max":
        return sorted(rows, key=lambda r: r["disagreement_score"], reverse=True)[:limit]
    raise ValueError(f"Unknown sample bucket: {bucket}")


def sample_rows(rows: list[dict[str, Any]], per_bucket: int = 10, max_total: int = 80) -> list[dict[str, Any]]:
    buckets = (
        "baseline_skipped_ensemble_high",
        "baseline_active_ensemble_low",
        "tfidf_high_seed_low",
        "seed_high_tfidf_low",
        "complexity_high",
        "dynamic_entity_high",
        "low_value_high",
        "disagreement_max",
    )
    by_id = {str(row["row_id"]): row for row in rows}
    selected_ids: list[str] = []
    for bucket in buckets:
        for row in select_bucket(rows, bucket, per_bucket):
            row_id = str(row["row_id"])
            if bucket not in row["sample_buckets"]:
                row["sample_buckets"].append(bucket)
            if row_id not in selected_ids:
                selected_ids.append(row_id)
            if len(selected_ids) >= max_total:
                return [by_id[row_id] for row_id in selected_ids]
    return [by_id[row_id] for row_id in selected_ids]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workspace_id",
        "row_id",
        "text_preview",
        "v0_1_routes",
        *[f"score_{mode}" for mode in MODE_NAMES],
        *[f"profile_{target}" for target in TARGET_PROFILES],
        *[f"suggested_{target}" for target in TARGET_PROFILES],
        "sample_buckets",
        "review_label_placeholder",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat: dict[str, Any] = {
                "workspace_id": row["workspace_id"],
                "row_id": row["row_id"],
                "text_preview": row["text_preview"],
                "v0_1_routes": json.dumps(row["v0_1_routes"], ensure_ascii=False, sort_keys=True),
                "sample_buckets": "|".join(row["sample_buckets"]),
                "review_label_placeholder": row["review_label_placeholder"],
            }
            flat.update({f"score_{mode}": row["mode_scores"][mode] for mode in MODE_NAMES})
            flat.update({f"profile_{target}": row["profile_scores"][target] for target in TARGET_PROFILES})
            flat.update({f"suggested_{target}": row["suggested_routes"][target] for target in TARGET_PROFILES})
            writer.writerow(flat)


def render_report(sidecar_path: Path, rows: list[dict[str, Any]], samples: list[dict[str, Any]]) -> str:
    route_counter: Counter[str] = Counter()
    suggested_counter: Counter[str] = Counter()
    bucket_counter: Counter[str] = Counter()
    for row in rows:
        for target, route in row["v0_1_routes"].items():
            route_counter[f"{target}:{route}"] += 1
        for target, route in row["suggested_routes"].items():
            suggested_counter[f"{target}:{route}"] += 1
    for row in samples:
        bucket_counter.update(row["sample_buckets"])

    lines = [
        "# Salience Mode Comparison",
        "",
        f"Input sidecar: `{sidecar_path}`",
        "",
        f"Rows compared: {len(rows)}",
        f"Sample rows: {len(samples)}",
        "",
        "## Mode Summary",
        "",
        "| mode | avg_score | top_row_score |",
        "|---|---:|---:|",
    ]
    for mode in MODE_NAMES:
        scores = [float(row["mode_scores"][mode]) for row in rows]
        avg = round(sum(scores) / len(scores), 3) if scores else 0.0
        top = round(max(scores), 3) if scores else 0.0
        lines.append(f"| `{mode}` | {avg} | {top} |")

    lines.extend(["", "## Existing Route Counts", ""])
    for route, count in route_counter.most_common(20):
        lines.append(f"- `{route}`: {count}")
    lines.extend(["", "## Suggested Route Counts", ""])
    for route, count in suggested_counter.most_common(24):
        lines.append(f"- `{route}`: {count}")
    lines.extend(["", "## Sample Bucket Counts", ""])
    for bucket, count in bucket_counter.most_common():
        lines.append(f"- `{bucket}`: {count}")

    lines.extend(["", "## Sample Table", ""])
    lines.append("| bucket | row_id | ensemble | baseline | seed | entity | tfidf | external | complexity | suggested_s1 | suggested_s2 | text |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
    for row in samples[:80]:
        text = str(row["text_preview"]).replace("\n", " ").replace("|", "\\|")[:180]
        buckets = ",".join(row["sample_buckets"])
        mode = row["mode_scores"]
        lines.append(
            f"| `{buckets}` | `{row['row_id']}` | {mode['ensemble_profile']} | {mode['baseline_v0_1_route']} | "
            f"{mode['seed_lexicon']} | {mode['seed_plus_entity']} | {mode['tfidf_salience']} | {mode['external_wheels']} | "
            f"{mode['complexity_assist']} | `{row['suggested_routes']['s1_memory_candidate']}` | "
            f"`{row['suggested_routes']['s2_portrait_candidate']}` | {text} |"
        )

    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Comparison only.",
            "- No route decisions changed.",
            "- No durable memory writes.",
            "- No graph writes.",
            "- No provider calls.",
            "- Review labels are placeholders for later calibration.",
            "",
        ]
    )
    return "\n".join(lines)


def default_output_dir(sidecar_path: Path) -> Path:
    if sidecar_path.parent.name:
        return sidecar_path.parent.parent / "salience_mode_comparison_v0_1"
    return sidecar_path.parent / "salience_mode_comparison_v0_1"


def run_for_sidecar(sidecar_path: Path, output_dir: Path | None = None, per_bucket: int = 10, max_samples: int = 80) -> dict[str, str]:
    sidecar_path = sidecar_path.resolve()
    output_dir = (output_dir or default_output_dir(sidecar_path)).resolve()
    source_rows = read_jsonl(sidecar_path)
    compared = [compare_row(row) for row in source_rows]
    samples = sample_rows(compared, per_bucket=per_bucket, max_total=max_samples)

    rows_path = output_dir / "mode_comparison_rows.jsonl"
    samples_path = output_dir / "mode_comparison_samples.jsonl"
    table_path = output_dir / "mode_comparison_table.csv"
    report_path = output_dir / "mode_comparison_report.md"
    write_jsonl(rows_path, compared)
    write_jsonl(samples_path, samples)
    write_csv(table_path, compared)
    write_text(report_path, render_report(sidecar_path, compared, samples))
    return {
        "mode_comparison_rows": str(rows_path),
        "mode_comparison_samples": str(samples_path),
        "mode_comparison_table": str(table_path),
        "mode_comparison_report": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar", action="append", required=True, help="Path to salience_sidecar.jsonl. May be repeated.")
    parser.add_argument("--output-dir", help="Optional output directory. Only valid with one --sidecar.")
    parser.add_argument("--per-bucket", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=80)
    args = parser.parse_args()
    if args.output_dir and len(args.sidecar) != 1:
        raise SystemExit("--output-dir can only be used with exactly one --sidecar")
    results = []
    for sidecar in args.sidecar:
        results.append(
            run_for_sidecar(
                Path(sidecar),
                output_dir=Path(args.output_dir) if args.output_dir else None,
                per_bucket=args.per_bucket,
                max_samples=args.max_samples,
            )
        )
    print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
