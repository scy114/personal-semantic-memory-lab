"""Read-only comparison for the v0.21 trained router bundle.

This tool does not change canonical routing outputs. It loads the trained
classical bundle, evaluates it on a held-out or sampled slice of the proxy
calibration rows, and compares the bundle against the current candidate policy
and a simple heuristic baseline.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from tools.routing_v021_mode_comparison import suggested_s1_route, suggested_s2_route
from tools.routing_v021_policy_trainer import FEATURE_COLUMNS, ROUTES, TASKS, load_instances, route_gap_penalty


SCHEMA_VERSION = "routing.v021_policy_eval.v0.1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "reports/v021_policy_calibration/v021_calibration_training_rows.jsonl"
DEFAULT_BUNDLE = "reports/v021_policy_training/routing_v021_policy_bundle.joblib"
DEFAULT_CANDIDATE_POLICY = "configs/routing/memory_proposal_router/heuristic_salience_v0.21.candidate.yaml"
DEFAULT_OUTPUT_DIR = "reports/v021_policy_evaluation"
DEFAULT_SPLIT = "heldout"
DEFAULT_HOLDOUT_RATIO = 0.25
DEFAULT_SAMPLE_LIMIT = 120
SYSTEMS = ("bundle", "candidate_policy", "heuristic_baseline")
SYSTEM_ROW_KEYS = {
    "bundle": "bundle_prediction",
    "candidate_policy": "candidate_prediction",
    "heuristic_baseline": "baseline_prediction",
}


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


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


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def route_rank(route: str) -> int:
    try:
        return ROUTES.index(route)
    except ValueError:
        return 0


def weighted_route_penalty(true_route: str, pred_route: str, axis: dict[str, Any], task: str, sample_weight: float) -> float:
    return route_gap_penalty(true_route, pred_route, axis, task) * float(sample_weight)


def stable_hash_fraction(*parts: str) -> float:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12 - 1)


def choose_split(example_id: str, task: str, split: str, holdout_ratio: float) -> bool:
    fraction = stable_hash_fraction(example_id, task)
    if split == "heldout":
        return fraction >= holdout_ratio
    if split == "train":
        return fraction < holdout_ratio
    return True


def load_candidate_policy(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected policy format in {path}")
    return data


def score_candidate(axis: dict[str, Any], policy: dict[str, Any], task: str) -> tuple[float, float, str]:
    weights = policy.get("weights") or {}
    value = float(axis.get("value_score") or 0.0)
    entity = float(axis.get("entity_salience_score") or 0.0)
    domain = float(axis.get("domain_term_score") or 0.0)
    keyphrase = float(axis.get("keyphrase_score") or 0.0)
    affect = float(axis.get("affect_score") or 0.0)
    risk = float(axis.get("risk_score") or 0.0)
    complexity = float(axis.get("complexity_score") or 0.0)
    useful = (
        value * float(weights.get("value_score") or 0.0)
        + entity * float(weights.get("entity_salience_score") or 0.0)
        + domain * float(weights.get("domain_term_score") or 0.0)
        + keyphrase * float(weights.get("keyphrase_score") or 0.0)
        + affect * float(weights.get("affect_score") or 0.0)
    )
    stress = risk * float(weights.get("risk_score") or 0.0) + complexity * float(weights.get("complexity_score") or 0.0)
    task_key = "s1" if task == "s1_memory_candidate" else "s2"
    thresholds = policy.get("thresholds") or {}
    task_thresholds = thresholds.get(task_key) or {}
    low = float(axis.get("low_value_score") or 0.0)
    if low >= float(task_thresholds.get("low_value_skip_floor") or 0.0) and useful < float(task_thresholds.get("low_value_useful_floor") or 0.0):
        return useful, stress, "skip_or_background_only"
    if useful < float(task_thresholds.get("skip_useful_floor") or 0.0) and low >= float(task_thresholds.get("skip_low_value_floor") or 0.0):
        return useful, stress, "skip_or_background_only"
    if useful >= float(task_thresholds.get("strong_useful_floor") or 0.0) and (
        stress >= float(task_thresholds.get("strong_stress_floor") or 0.0)
        or risk >= float(task_thresholds.get("strong_risk_floor") or 0.0)
    ):
        return useful, stress, "strong_llm_proposal"
    if useful >= float(task_thresholds.get("weak_useful_floor") or 0.0) and (
        stress >= float(task_thresholds.get("weak_stress_floor") or 0.0)
        or risk >= float(task_thresholds.get("weak_risk_floor") or 0.0)
    ):
        return useful, stress, "weak_llm_proposal"
    if useful >= float(task_thresholds.get("script_useful_floor") or 0.0):
        return useful, stress, "script_only"
    return useful, stress, "skip_or_background_only"


def predict_candidate(axis: dict[str, Any], policy: dict[str, Any], task: str) -> dict[str, Any]:
    useful, stress, route = score_candidate(axis, policy, task)
    return {"route": route, "useful_score": round(float(useful), 6), "stress_score": round(float(stress), 6)}


def predict_baseline(axis: dict[str, Any], task: str) -> str:
    if task == "s1_memory_candidate":
        return suggested_s1_route(axis)
    return suggested_s2_route(axis)


def load_bundle(path: Path) -> dict[str, Any]:
    bundle = joblib.load(path)
    if not isinstance(bundle, dict):
        raise ValueError(f"Unexpected bundle type: {type(bundle)!r}")
    return bundle


def prepare_frame(input_path: Path, split: str, holdout_ratio: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame, meta = load_instances(input_path)
    if frame.empty:
        return frame, meta
    mask = [choose_split(str(row.example_id), str(row.task), split, holdout_ratio) for row in frame.itertuples(index=False)]
    frame = frame.loc[mask].reset_index(drop=True)
    meta = {**meta, "split": split, "holdout_ratio": holdout_ratio, "evaluated_instances": int(len(frame))}
    return frame, meta


def route_penalty_stats(rows: list[dict[str, Any]], system: str) -> tuple[float, int, int]:
    total_penalty = 0.0
    false_skips = 0
    over_strong = 0
    total_weight = 0.0
    for row in rows:
        pred = str(row[system])
        true = str(row["label"])
        axis = row["axis_scores"]
        weight = float(row["weight"])
        total_penalty += weighted_route_penalty(true, pred, axis, str(row["task"]), weight)
        total_weight += weight
        if true in {"weak_llm_proposal", "strong_llm_proposal", "human_review", "script_only"} and pred == "skip_or_background_only":
            false_skips += 1
        if true in {"skip_or_background_only", "script_only"} and pred in {"weak_llm_proposal", "strong_llm_proposal", "human_review"}:
            over_strong += 1
    return (total_penalty / max(total_weight, 1e-9), false_skips, over_strong)


def build_eval_rows(frame: pd.DataFrame, bundle: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    eval_rows: list[dict[str, Any]] = []
    for task_name in TASKS:
        task_frame = frame[frame["task"] == task_name].reset_index(drop=True)
        if task_frame.empty:
            continue
        feature_columns = list(bundle.get("feature_columns") or FEATURE_COLUMNS)
        X = task_frame[feature_columns].astype(float)
        task_bundle = bundle["tasks"][task_name]
        model = task_bundle["model"]
        bundle_pred = [str(item) for item in model.predict(X).tolist()]
        bundle_prob = None
        if hasattr(model, "predict_proba"):
            try:
                probs = np.asarray(model.predict_proba(X))
                bundle_prob = probs.max(axis=1).tolist()
            except Exception:
                bundle_prob = None
        for idx, row in task_frame.iterrows():
            axis = {column: float(row.get(column) or 0.0) for column in feature_columns}
            candidate = predict_candidate(axis, policy, task_name)
            baseline_route = predict_baseline(axis, task_name)
            eval_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": row["example_id"],
                    "task": task_name,
                    "dataset_id": row.get("dataset_id"),
                    "source_kind": row.get("source_kind"),
                    "source_path": row.get("source_path"),
                    "text_preview": row.get("text_preview"),
                    "label": row["label"],
                    "weight": float(row["weight"]),
                    "confidence": float(row["confidence"]),
                    "axis_scores": axis,
                    "bundle_prediction": bundle_pred[idx],
                    "bundle_confidence": float(bundle_prob[idx]) if bundle_prob is not None else None,
                    "candidate_prediction": candidate["route"],
                    "candidate_useful_score": candidate["useful_score"],
                    "candidate_stress_score": candidate["stress_score"],
                    "baseline_prediction": baseline_route,
                    "bundle_match": bundle_pred[idx] == row["label"],
                    "candidate_match": candidate["route"] == row["label"],
                    "baseline_match": baseline_route == row["label"],
                    "bundle_penalty": weighted_route_penalty(str(row["label"]), bundle_pred[idx], axis, task_name, float(row["weight"])),
                    "candidate_penalty": weighted_route_penalty(str(row["label"]), candidate["route"], axis, task_name, float(row["weight"])),
                    "baseline_penalty": weighted_route_penalty(str(row["label"]), baseline_route, axis, task_name, float(row["weight"])),
                    "write_permission": False,
                }
            )
    return eval_rows


def compute_metrics(rows: list[dict[str, Any]], system_key: str) -> dict[str, Any]:
    y_true = [str(row["label"]) for row in rows]
    y_pred = [str(row[system_key]) for row in rows]
    weights = np.asarray([float(row["weight"]) for row in rows], dtype=float)
    exact = float(np.average(np.asarray(y_true) == np.asarray(y_pred), weights=weights)) if rows else 0.0
    classes = sorted(set(y_true) | set(y_pred), key=route_rank)
    macro_f1 = float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0)) if rows else 0.0
    weighted_f1 = float(f1_score(y_true, y_pred, labels=classes, average="weighted", zero_division=0)) if rows else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if rows else 0.0
    acc = float(accuracy_score(y_true, y_pred, sample_weight=weights)) if rows else 0.0
    penalty, false_skips, over_strong = route_penalty_stats(rows, system_key)
    confusion: dict[str, dict[str, int]] = defaultdict(dict)
    for true_route in classes:
        for pred_route in classes:
            confusion[true_route][pred_route] = int(np.sum((np.asarray(y_true) == true_route) & (np.asarray(y_pred) == pred_route)))
    objective = penalty + (1.0 - exact) * 0.35 + (1.0 - macro_f1) * 0.15
    return {
        "accuracy_weighted": round(acc, 6),
        "exact_weighted": round(exact, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "route_penalty_weighted": round(penalty, 6),
        "objective": round(objective, 6),
        "false_skips": int(false_skips),
        "over_strong": int(over_strong),
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "class_count": len(classes),
        "row_count": len(rows),
    }


def select_error_samples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(
        rows,
        key=lambda row: (
            0 if row["bundle_prediction"] == "skip_or_background_only" and row["label"] != "skip_or_background_only" else 1,
            0 if row["label"] in {"skip_or_background_only", "script_only"} and row["bundle_prediction"] in {"weak_llm_proposal", "strong_llm_proposal", "human_review"} else 1,
            -float(row["bundle_penalty"]),
            -float(row["candidate_penalty"]),
        ),
    )
    return ranked[:limit]


def render_report(meta: dict[str, Any], task_metrics: dict[str, dict[str, Any]], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# v0.21 Policy Evaluation Report",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- input_path: `{meta['input_path']}`",
        f"- split: `{meta['split']}`",
        f"- holdout_ratio: `{meta['holdout_ratio']}`",
        f"- evaluated_instances: `{meta['evaluated_instances']}`",
        f"- skipped_labels: `{json.dumps(meta['skipped_labels'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Task Metrics",
        "",
    ]
    for task_name, metrics in task_metrics.items():
        lines.extend(
            [
                f"### {task_name}",
                "",
                "| system | objective | exact | macro_f1 | weighted_f1 | penalty | false_skips | over_strong |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for system_name in SYSTEMS:
            item = metrics[system_name]
            lines.append(
                f"| {system_name} | {item['objective']} | {item['exact_weighted']} | {item['macro_f1']} | {item['weighted_f1']} | {item['route_penalty_weighted']} | {item['false_skips']} | {item['over_strong']} |"
            )
        best_system = min(SYSTEMS, key=lambda name: metrics[name]["objective"])
        lines.extend(["", f"- best_system: `{best_system}`", ""])
    lines.extend(["## High-Risk Samples", ""])
    for row in samples[:30]:
        lines.append(
            f"- `{row['task']}` `{row['example_id']}` label=`{row['label']}` bundle=`{row['bundle_prediction']}` candidate=`{row['candidate_prediction']}` baseline=`{row['baseline_prediction']}`: {str(row['text_preview'])[:180]}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a read-only evaluation pass.",
            "- The trained bundle, candidate policy, and heuristic baseline are compared; none becomes canonical by this script alone.",
            "- Proxy labels remain proxy labels.",
            "- No provider calls, durable memory writes, graph writes, or S3 work were performed.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, str]:
    input_path = resolve_project_path(args.input)
    bundle_path = resolve_project_path(args.bundle)
    candidate_policy_path = resolve_project_path(args.candidate_policy)
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_bundle(bundle_path)
    candidate_policy = load_candidate_policy(candidate_policy_path)
    frame, meta = prepare_frame(input_path, args.split, args.holdout_ratio)
    if frame.empty:
        raise ValueError(f"No evaluation instances found in {input_path}")

    eval_rows = build_eval_rows(frame, bundle, candidate_policy)
    samples = select_error_samples(eval_rows, args.sample_limit) if args.sample_limit > 0 else []

    task_metrics: dict[str, dict[str, Any]] = {}
    for task_name in TASKS:
        task_rows = [row for row in eval_rows if row["task"] == task_name]
        if not task_rows:
            continue
        task_metrics[task_name] = {
            system: compute_metrics(task_rows, SYSTEM_ROW_KEYS[system])
            for system in SYSTEMS
        }

    rows_path = output_dir / "v021_policy_eval_rows.jsonl"
    samples_path = output_dir / "v021_policy_eval_samples.jsonl"
    table_path = output_dir / "v021_policy_eval_table.csv"
    report_path = output_dir / "v021_policy_eval_report.md"
    summary_path = output_dir / "v021_policy_eval_summary.json"

    write_jsonl(rows_path, eval_rows)
    write_jsonl(samples_path, samples)
    write_csv(
        table_path,
        [
            {
                "task": row["task"],
                "example_id": row["example_id"],
                "label": row["label"],
                "bundle_prediction": row["bundle_prediction"],
                "candidate_prediction": row["candidate_prediction"],
                "baseline_prediction": row["baseline_prediction"],
                "bundle_penalty": row["bundle_penalty"],
                "candidate_penalty": row["candidate_penalty"],
                "baseline_penalty": row["baseline_penalty"],
                "bundle_match": row["bundle_match"],
                "candidate_match": row["candidate_match"],
                "baseline_match": row["baseline_match"],
            }
            for row in eval_rows
        ],
        [
            "task",
            "example_id",
            "label",
            "bundle_prediction",
            "candidate_prediction",
            "baseline_prediction",
            "bundle_penalty",
            "candidate_penalty",
            "baseline_penalty",
            "bundle_match",
            "candidate_match",
            "baseline_match",
        ],
    )
    write_text(report_path, render_report({**meta, "input_path": str(input_path)}, task_metrics, samples))
    write_json(
        summary_path,
        {
            "schema_version": SCHEMA_VERSION,
            "input_path": str(input_path),
            "bundle_path": str(bundle_path),
            "candidate_policy_path": str(candidate_policy_path),
            "output_dir": str(output_dir),
            "split": args.split,
            "holdout_ratio": args.holdout_ratio,
            "evaluated_instances": len(eval_rows),
            "sample_count": len(samples),
            "task_metrics": task_metrics,
            "write_permission": False,
            "outputs": {
                "rows": str(rows_path),
                "samples": str(samples_path),
                "table": str(table_path),
                "report": str(report_path),
                "summary": str(summary_path),
            },
        },
    )
    return {
        "rows": str(rows_path),
        "samples": str(samples_path),
        "table": str(table_path),
        "report": str(report_path),
        "summary": str(summary_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate the v0.21 trained router bundle against proxy calibration rows.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Calibration training rows JSONL.")
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE, help="Trained joblib bundle path.")
    parser.add_argument("--candidate-policy", default=DEFAULT_CANDIDATE_POLICY, help="Candidate heuristic policy YAML.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for evaluation artifacts.")
    parser.add_argument("--split", choices=("heldout", "train", "all"), default=DEFAULT_SPLIT)
    parser.add_argument("--holdout-ratio", type=float, default=DEFAULT_HOLDOUT_RATIO)
    parser.add_argument("--sample-limit", type=int, default=DEFAULT_SAMPLE_LIMIT)
    return parser


def main() -> int:
    outputs = run(build_arg_parser().parse_args())
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
