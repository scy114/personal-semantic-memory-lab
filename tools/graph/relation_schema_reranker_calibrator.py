"""Calibrate relation schema candidate reranking from proxy extraction labels.

This tool does not create graph truth. It learns a lightweight reranker from
schema candidates and accepted provider relation candidates so later prompts can
receive better relation type candidates without hand-tuned weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from tools.graph.graph_construction_packet_builder import file_hash, read_jsonl, write_json, write_jsonl, write_text


SCHEMA_VERSION = "graph_v03.relation_schema_reranker_calibration.v0.1"
ROW_SCHEMA_VERSION = "graph_v03.relation_schema_reranker_feature_row.v0.1"
RERANKED_SCHEMA_VERSION = "graph_v03.packet_relation_schema_candidates.reranked.v0.1"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_relation_schema_reranker_calibration"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_relation_type(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return "_".join(part for part in text.split("_") if part)


def string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    out: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def unique_strings(value: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in value:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def load_positive_relation_types(extraction_dirs: list[Path]) -> dict[str, set[str]]:
    positives: dict[str, set[str]] = defaultdict(set)
    for extraction_dir in extraction_dirs:
        relation_path = extraction_dir / "graph_relation_candidates.jsonl"
        if not relation_path.exists():
            raise FileNotFoundError(f"graph_relation_candidates.jsonl not found: {relation_path}")
        for row in read_jsonl(relation_path):
            packet_id = str(row.get("source_packet_id") or "")
            relation_type = normalize_relation_type(row.get("relation_type_hint"))
            if not packet_id or not relation_type or relation_type == "out_of_schema_relation":
                continue
            if row.get("relation_schema_status") not in {
                "selected_from_retrieved_schema",
                "schema_known_but_not_retrieved",
                None,
                "",
            }:
                continue
            positives[packet_id].add(relation_type)
    return positives


def candidate_features(candidate: dict[str, Any]) -> dict[str, Any]:
    retrieval = candidate.get("retrieval_features") or {}
    exact_hits = string_list(retrieval.get("exact_alias_hits"))
    matched_tokens = string_list(retrieval.get("matched_tokens"))
    sources = string_list(candidate.get("external_sources"))
    aliases = string_list(candidate.get("aliases"))
    try:
        rank = float(candidate.get("rank") or 9999)
    except (TypeError, ValueError):
        rank = 9999.0
    try:
        score = float(candidate.get("score") or 0.0)
    except (TypeError, ValueError):
        score = 0.0
    source_count = int(candidate.get("source_count") or 0)
    return {
        "rank": rank,
        "inverse_rank": 1.0 / max(rank, 1.0),
        "retrieval_score": score,
        "log_retrieval_score": math.log1p(max(score, 0.0)),
        "lexical_score": float(retrieval.get("lexical_score") or 0.0),
        "log_lexical_score": math.log1p(max(float(retrieval.get("lexical_score") or 0.0), 0.0)),
        "exact_alias_hit_count": len(exact_hits),
        "matched_token_count": len(matched_tokens),
        "source_count": source_count,
        "log_source_count": math.log1p(max(source_count, 0)),
        "multi_source": int(len(sources) > 1),
        "alias_count": len(aliases),
        "category": str(candidate.get("category") or "unknown"),
        "relation_type": normalize_relation_type(candidate.get("relation_type")),
    }


def build_feature_rows(candidate_rows: list[dict[str, Any]], positives: dict[str, set[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for packet_row in candidate_rows:
        packet_id = str(packet_row.get("packet_id") or "")
        positive_types = positives.get(packet_id, set())
        if not positive_types:
            continue
        for candidate in packet_row.get("relation_schema_candidates") or []:
            relation_type = normalize_relation_type(candidate.get("relation_type"))
            features = candidate_features(candidate)
            label = int(relation_type in positive_types)
            rows.append(
                {
                    "schema_version": ROW_SCHEMA_VERSION,
                    "packet_id": packet_id,
                    "relation_type": relation_type,
                    "label": label,
                    "features": features,
                    "candidate": candidate,
                    "graph_is_not_proof": True,
                }
            )
    return rows


def train_classifier(feature_rows: list[dict[str, Any]]) -> tuple[Pipeline, dict[str, Any]]:
    labels = [int(row["label"]) for row in feature_rows]
    counts = Counter(labels)
    if len(counts) < 2:
        raise ValueError("Need both positive and negative relation candidate labels for calibration")

    min_class_count = min(counts.values())
    base = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=17)
    if min_class_count >= 2:
        cv = min(3, min_class_count)
        estimator: Any = CalibratedClassifierCV(base, method="sigmoid", cv=cv)
        calibration_method = f"calibrated_sigmoid_cv{cv}"
    else:
        estimator = base
        calibration_method = "uncalibrated_logistic_too_few_minority_examples"

    pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            ("classifier", estimator),
        ]
    )
    x_rows = [row["features"] for row in feature_rows]
    pipeline.fit(x_rows, labels)
    probabilities = [float(value[1]) for value in pipeline.predict_proba(x_rows)]
    metrics = ranking_metrics(feature_rows, probabilities)
    metrics["label_counts"] = dict(counts)
    metrics["calibration_method"] = calibration_method
    return pipeline, metrics


def ranking_metrics(feature_rows: list[dict[str, Any]], probabilities: list[float]) -> dict[str, Any]:
    labels = [int(row["label"]) for row in feature_rows]
    metrics: dict[str, Any] = {
        "row_count": len(feature_rows),
        "positive_count": sum(labels),
        "negative_count": len(labels) - sum(labels),
    }
    if len(set(labels)) > 1:
        metrics["average_precision_in_sample"] = round(float(average_precision_score(labels, probabilities)), 6)
        metrics["roc_auc_in_sample"] = round(float(roc_auc_score(labels, probabilities)), 6)
    by_packet: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for row, prob in zip(feature_rows, probabilities):
        baseline = float(row["features"].get("inverse_rank") or 0.0)
        by_packet[str(row["packet_id"])].append((int(row["label"]), baseline, prob))
    for key, index in (("baseline", 1), ("calibrated", 2)):
        recall_at_5 = []
        recall_at_10 = []
        mrr = []
        for items in by_packet.values():
            if not any(label for label, _, _ in items):
                continue
            ranked = sorted(items, key=lambda item: item[index], reverse=True)
            recall_at_5.append(float(any(label for label, _, _ in ranked[:5])))
            recall_at_10.append(float(any(label for label, _, _ in ranked[:10])))
            reciprocal = 0.0
            for rank, (label, _, _) in enumerate(ranked, 1):
                if label:
                    reciprocal = 1.0 / rank
                    break
            mrr.append(reciprocal)
        denom = max(1, len(mrr))
        metrics[f"{key}_recall_at_5_in_sample"] = round(sum(recall_at_5) / denom, 6)
        metrics[f"{key}_recall_at_10_in_sample"] = round(sum(recall_at_10) / denom, 6)
        metrics[f"{key}_mrr_in_sample"] = round(sum(mrr) / denom, 6)
    metrics["packet_count_with_labels"] = len(by_packet)
    return metrics


def rerank_candidate_rows(candidate_rows: list[dict[str, Any]], pipeline: Pipeline) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for packet_row in candidate_rows:
        candidates = list(packet_row.get("relation_schema_candidates") or [])
        if not candidates:
            out.append({**packet_row, "schema_version": RERANKED_SCHEMA_VERSION, "reranker_applied": False})
            continue
        feature_rows = [candidate_features(candidate) for candidate in candidates]
        probabilities = [float(value[1]) for value in pipeline.predict_proba(feature_rows)]
        scored = []
        for candidate, probability in zip(candidates, probabilities):
            updated = dict(candidate)
            updated["reranker_score"] = round(probability, 6)
            updated["baseline_rank"] = candidate.get("rank")
            scored.append(updated)
        scored.sort(key=lambda item: (-float(item.get("reranker_score") or 0.0), int(item.get("baseline_rank") or 9999)))
        for rank, candidate in enumerate(scored, 1):
            candidate["rank"] = rank
        out.append(
            {
                **packet_row,
                "schema_version": RERANKED_SCHEMA_VERSION,
                "relation_schema_candidates": scored,
                "reranker_applied": True,
                "graph_is_not_proof": True,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = ["packet_id", "relation_type", "label", "rank", "retrieval_score", "exact_alias_hit_count", "matched_token_count", "category"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            features = row.get("features") or {}
            writer.writerow(
                {
                    "packet_id": row.get("packet_id"),
                    "relation_type": row.get("relation_type"),
                    "label": row.get("label"),
                    "rank": features.get("rank"),
                    "retrieval_score": features.get("retrieval_score"),
                    "exact_alias_hit_count": features.get("exact_alias_hit_count"),
                    "matched_token_count": features.get("matched_token_count"),
                    "category": features.get("category"),
                }
            )


def render_report(manifest: dict[str, Any]) -> str:
    metrics = manifest["metrics"]
    return "\n".join(
        [
            "# Relation Schema Reranker Calibration Report",
            "",
            f"- candidate_rows: {manifest['counts']['candidate_packet_rows']}",
            f"- training_rows: {manifest['counts']['training_rows']}",
            f"- positives: {metrics.get('positive_count')}",
            f"- negatives: {metrics.get('negative_count')}",
            f"- calibration_method: `{metrics.get('calibration_method')}`",
            "- graph_is_not_proof: `true`",
            "- labels are proxy labels from provider/external extraction outputs, not graph truth",
            "",
            "## In-Sample Metrics",
            "",
            f"- baseline_recall_at_5: {metrics.get('baseline_recall_at_5_in_sample')}",
            f"- calibrated_recall_at_5: {metrics.get('calibrated_recall_at_5_in_sample')}",
            f"- baseline_mrr: {metrics.get('baseline_mrr_in_sample')}",
            f"- calibrated_mrr: {metrics.get('calibrated_mrr_in_sample')}",
            f"- average_precision: {metrics.get('average_precision_in_sample', 'n/a')}",
            f"- roc_auc: {metrics.get('roc_auc_in_sample', 'n/a')}",
            "",
            "## Boundary",
            "",
            "- This is a reranking calibration layer for prompt candidate selection.",
            "- It does not decide graph truth, support proof, or durable memory.",
            "- Use held-out workspaces before making it a default workflow dependency.",
            "",
        ]
    )


def run_calibration(
    *,
    relation_schema_candidates: Path,
    extraction_dirs: list[Path],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_rows = read_jsonl(relation_schema_candidates)
    positives = load_positive_relation_types(extraction_dirs)
    feature_rows = build_feature_rows(candidate_rows, positives)
    pipeline, metrics = train_classifier(feature_rows)
    reranked_rows = rerank_candidate_rows(candidate_rows, pipeline)

    model_path = output_dir / "relation_schema_reranker.joblib"
    joblib.dump(pipeline, model_path)
    write_jsonl(output_dir / "relation_schema_reranker_feature_rows.jsonl", feature_rows)
    write_csv(output_dir / "relation_schema_reranker_feature_rows.csv", feature_rows)
    write_jsonl(output_dir / "relation_schema_candidates.reranked.jsonl", reranked_rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "relation_schema_candidates": str(relation_schema_candidates.resolve()),
        "relation_schema_candidates_hash": file_hash(relation_schema_candidates),
        "extraction_dirs": [str(path.resolve()) for path in extraction_dirs],
        "extraction_hashes": {
            str((path / "graph_relation_candidates.jsonl").resolve()): file_hash(path / "graph_relation_candidates.jsonl")
            for path in extraction_dirs
        },
        "output_dir": str(output_dir),
        "outputs": {
            "model": str(model_path),
            "feature_rows": str(output_dir / "relation_schema_reranker_feature_rows.jsonl"),
            "feature_rows_csv": str(output_dir / "relation_schema_reranker_feature_rows.csv"),
            "reranked_candidates": str(output_dir / "relation_schema_candidates.reranked.jsonl"),
            "manifest": str(output_dir / "relation_schema_reranker_manifest.json"),
            "report": str(output_dir / "relation_schema_reranker_report.md"),
        },
        "counts": {
            "candidate_packet_rows": len(candidate_rows),
            "positive_packet_rows": len(positives),
            "training_rows": len(feature_rows),
        },
        "metrics": metrics,
        "boundary": {
            "graph_is_not_proof": True,
            "graph_truth": False,
            "support_status": "not_checked",
            "write_permission": False,
            "proxy_labels": True,
        },
    }
    write_json(output_dir / "relation_schema_reranker_manifest.json", manifest)
    write_text(output_dir / "relation_schema_reranker_report.md", render_report(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a calibrated reranker for graph relation schema candidates.")
    parser.add_argument("--relation-schema-candidates", required=True)
    parser.add_argument("--extraction-dir", action="append", required=True)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR_NAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = run_calibration(
        relation_schema_candidates=Path(args.relation_schema_candidates),
        extraction_dirs=[Path(path) for path in args.extraction_dir],
        output_dir=Path(args.output_dir),
    )
    print(json.dumps({"manifest": manifest["outputs"]["manifest"], "metrics": manifest["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
