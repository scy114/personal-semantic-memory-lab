"""Train classical route models from v0.21 calibration rows.

This tool is a read-only trainer: it consumes proxy-labeled calibration rows
and produces an auditable model bundle plus reports. It does not replace the
live router or write durable memory.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import warnings
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


SCHEMA_VERSION = "routing.v021_policy_training.v0.1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = "reports/v021_policy_calibration/v021_calibration_training_rows.jsonl"
DEFAULT_OUTPUT_DIR = "reports/v021_policy_training"
DEFAULT_BUNDLE_NAME = "routing_v021_policy_bundle.joblib"
ROUTES = (
    "skip_or_background_only",
    "script_only",
    "weak_llm_proposal",
    "strong_llm_proposal",
    "human_review",
)
TASKS = ("s1_memory_candidate", "s2_portrait_candidate")
FEATURE_COLUMNS = (
    "value_score",
    "risk_score",
    "complexity_score",
    "entity_salience_score",
    "keyphrase_score",
    "affect_score",
    "low_value_score",
    "lexical_complexity_score",
    "sentence_complexity_score",
    "domain_term_score",
    "language_en",
    "language_zh",
    "language_mixed",
    "language_unknown",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def clamp(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, value))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def route_rank(route: str) -> int:
    try:
        return ROUTES.index(route)
    except ValueError:
        return 0


def route_gap_penalty(true_route: str, pred_route: str, axis: dict[str, float], task: str) -> float:
    penalty = abs(route_rank(true_route) - route_rank(pred_route)) * 0.8
    useful = max(
        axis.get("value_score", 0.0),
        axis.get("entity_salience_score", 0.0),
        axis.get("domain_term_score", 0.0),
        axis.get("keyphrase_score", 0.0),
    )
    low_value = axis.get("low_value_score", 0.0)
    risk = axis.get("risk_score", 0.0)
    complexity = axis.get("complexity_score", 0.0)

    if true_route in {"weak_llm_proposal", "strong_llm_proposal", "human_review"} and pred_route == "skip_or_background_only" and useful >= 4.0:
        penalty += 2.0
    if true_route in {"skip_or_background_only", "script_only"} and pred_route in {"strong_llm_proposal", "human_review"}:
        penalty += 1.1
    if low_value >= 5.0 and pred_route in {"weak_llm_proposal", "strong_llm_proposal", "human_review"}:
        penalty += 0.7
    if task == "s2_portrait_candidate" and pred_route == "skip_or_background_only" and true_route != "skip_or_background_only":
        penalty += 1.2
    if risk >= 8.0 and pred_route == "script_only":
        penalty += 0.3
    if complexity >= 7.5 and pred_route == "skip_or_background_only" and true_route != "skip_or_background_only":
        penalty += 0.3
    return penalty


def language_features(value: str) -> dict[str, float]:
    language = (value or "unknown").lower()
    return {
        "language_en": 1.0 if language == "en" else 0.0,
        "language_zh": 1.0 if language == "zh" else 0.0,
        "language_mixed": 1.0 if language == "mixed" else 0.0,
        "language_unknown": 1.0 if language not in {"en", "zh", "mixed"} else 0.0,
    }


def load_instances(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows = read_jsonl(path)
    instances: list[dict[str, Any]] = []
    skipped_labels: Counter[str] = Counter()
    for row in rows:
        axis = row.get("axis_scores") or {}
        language = str(row.get("language") or "unknown")
        base_features = {key: to_float(axis.get(key)) for key in FEATURE_COLUMNS if key not in {"language_en", "language_zh", "language_mixed", "language_unknown"}}
        base_features.update(language_features(language))
        weight = to_float(row.get("weight"), 1.0)
        confidence = to_float(row.get("confidence"), 0.5)
        label_weight = clamp(weight * (0.5 + confidence), 0.05, 5.0)
        for task in TASKS:
            label = str((row.get("target_routes") or {}).get(task) or "").strip()
            if label not in ROUTES:
                skipped_labels[label or "missing"] += 1
                continue
            record = {
                "example_id": row.get("example_id"),
                "task": task,
                "label": label,
                "dataset_id": row.get("dataset_id"),
                "source_kind": row.get("source_kind"),
                "source_path": row.get("source_path"),
                "text_preview": row.get("text_preview"),
                "weight": label_weight,
                "confidence": confidence,
                **base_features,
            }
            instances.append(record)
    frame = pd.DataFrame(instances)
    meta = {
        "source_rows": len(rows),
        "training_instances": len(frame),
        "skipped_labels": dict(skipped_labels),
    }
    return frame, meta


def make_models(seed: int, class_count: int) -> dict[str, Callable[[], Any]]:
    models: dict[str, Callable[[], Any]] = {
        "logistic_regression": lambda: Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "decision_tree": lambda: DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "gradient_boosting": lambda: HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_depth=6,
            max_iter=220,
            random_state=seed,
        ),
    }
    if importlib.util.find_spec("lightgbm") is not None:
        from lightgbm import LGBMClassifier  # type: ignore

        models["lightgbm"] = lambda: LGBMClassifier(
            objective="multiclass",
            num_class=class_count,
            learning_rate=0.06,
            n_estimators=350,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
            verbosity=-1,
            force_col_wise=True,
        )
    return models


def safe_predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X))


def top_features_from_model(model: Any, feature_names: list[str], class_names: list[str]) -> dict[str, Any]:
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_)
        if coef.ndim == 1:
            coef = coef[None, :]
        return {
            "kind": "linear",
            "per_class_top_positive": {
                class_names[i]: [
                    {"feature": feature_names[j], "weight": round(float(coef[i, j]), 6)}
                    for j in np.argsort(coef[i])[::-1][:6]
                ]
                for i in range(min(len(class_names), coef.shape[0]))
            },
            "per_class_top_negative": {
                class_names[i]: [
                    {"feature": feature_names[j], "weight": round(float(coef[i, j]), 6)}
                    for j in np.argsort(coef[i])[:6]
                ]
                for i in range(min(len(class_names), coef.shape[0]))
            },
        }
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
        order = np.argsort(importances)[::-1][:10]
        return {
            "kind": "tree",
            "top_features": [
                {"feature": feature_names[i], "importance": round(float(importances[i]), 6)}
                for i in order
            ],
        }
    return {"kind": "unknown"}


def fit_with_weights(model: Any, X: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> Any:
    if isinstance(model, Pipeline):
        model.fit(X, y, model__sample_weight=weights)
        return model
    try:
        model.fit(X, y, sample_weight=weights)
    except TypeError:
        model.fit(X, y)
    return model


def evaluate_model(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    weights: np.ndarray,
    task_name: str,
    seed: int,
) -> dict[str, Any]:
    class_counts = y.value_counts()
    min_class_count = int(class_counts.min()) if not class_counts.empty else 1
    n_splits = max(2, min(5, min_class_count))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    classes = sorted(y.unique().tolist(), key=route_rank)
    oof_pred: list[str] = []
    oof_true: list[str] = []
    oof_weight: list[float] = []
    oof_penalty: list[float] = []

    for train_idx, valid_idx in splitter.split(X, y):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_valid = X.iloc[valid_idx]
        y_valid = y.iloc[valid_idx]
        w_train = weights[train_idx]
        w_valid = weights[valid_idx]

        model = model_factory()
        if task_name == "s1_memory_candidate" and hasattr(model, "set_params"):
            try:
                model.set_params(**{})
            except Exception:
                pass
        fit_with_weights(model, X_train, y_train, w_train)
        pred = safe_predict(model, X_valid)
        oof_pred.extend([str(item) for item in pred.tolist()])
        oof_true.extend([str(item) for item in y_valid.tolist()])
        oof_weight.extend([float(item) for item in w_valid.tolist()])
        for idx, true_route in enumerate(y_valid.tolist()):
            pred_route = str(pred[idx])
            axis = X_valid.iloc[idx].to_dict()
            oof_penalty.append(route_gap_penalty(str(true_route), pred_route, axis, task_name) * float(w_valid[idx]))

    exact = float(np.average(np.asarray(oof_pred) == np.asarray(oof_true), weights=np.asarray(oof_weight)))
    macro_f1 = float(f1_score(oof_true, oof_pred, labels=classes, average="macro"))
    weighted_f1 = float(f1_score(oof_true, oof_pred, labels=classes, average="weighted"))
    bal_acc = float(balanced_accuracy_score(oof_true, oof_pred))
    acc = float(accuracy_score(oof_true, oof_pred, sample_weight=oof_weight))
    penalty = float(np.sum(oof_penalty) / max(1e-9, np.sum(oof_weight)))

    confusion: dict[str, dict[str, int]] = defaultdict(dict)
    for true_route in classes:
        for pred_route in classes:
            confusion[true_route][pred_route] = int(np.sum((np.asarray(oof_true) == true_route) & (np.asarray(oof_pred) == pred_route)))

    return {
        "task": task_name,
        "n_splits": n_splits,
        "classes": classes,
        "accuracy_weighted": round(acc, 6),
        "exact_weighted": round(exact, 6),
        "balanced_accuracy": round(bal_acc, 6),
        "macro_f1": round(macro_f1, 6),
        "weighted_f1": round(weighted_f1, 6),
        "route_penalty_weighted": round(penalty, 6),
        "objective": round(penalty + (1.0 - exact) * 0.35 + (1.0 - macro_f1) * 0.15, 6),
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "oof_counts": {
            "true": dict(Counter(oof_true)),
            "pred": dict(Counter(oof_pred)),
        },
    }


def fit_final_model(model_factory: Callable[[], Any], X: pd.DataFrame, y: pd.Series, weights: np.ndarray) -> Any:
    model = model_factory()
    return fit_with_weights(model, X, y, weights)


def render_report(task_reports: list[dict[str, Any]], meta: dict[str, Any], best_by_task: dict[str, str], feature_names: list[str]) -> str:
    lines = [
        "# v0.21 Route Training Report",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- training_rows: `{meta['source_rows']}`",
        f"- training_instances: `{meta['training_instances']}`",
        f"- skipped_labels: `{json.dumps(meta['skipped_labels'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Best Models",
        "",
    ]
    for task, model_name in best_by_task.items():
        lines.append(f"- `{task}`: `{model_name}`")
    lines.extend(["", "## Task Metrics", ""])
    for report in task_reports:
        lines.extend(
            [
                f"### {report['task']}",
                "",
                f"- best_model: `{report['best_model']}`",
                f"- objective: `{report['best_objective']}`",
                f"- exact_weighted: `{report['best_metrics']['exact_weighted']}`",
                f"- macro_f1: `{report['best_metrics']['macro_f1']}`",
                f"- weighted_f1: `{report['best_metrics']['weighted_f1']}`",
                f"- route_penalty_weighted: `{report['best_metrics']['route_penalty_weighted']}`",
                "",
            ]
        )
        lines.append("Top models:")
        for item in report["models"][:5]:
            lines.append(
                f"- `{item['model_name']}` objective={item['objective']} exact={item['exact_weighted']} "
                f"macro_f1={item['macro_f1']} penalty={item['route_penalty_weighted']}"
            )
        lines.append("")
    lines.extend(
        [
            "## Features",
            "",
            "- Numeric feature columns:",
            "",
        ]
    )
    lines.extend(f"- `{name}`" for name in feature_names)
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a classical model training pass, not deep learning.",
            "- The trained bundle is auxiliary; it does not replace the live router by itself.",
            "- Proxy labels are still proxy labels.",
            "- No provider calls, durable memory writes, graph writes, or S3 work were performed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train classical router models from v0.21 calibration rows.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Calibration training rows JSONL.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for model bundle and reports.")
    parser.add_argument("--bundle-name", default=DEFAULT_BUNDLE_NAME, help="Joblib bundle name.")
    parser.add_argument("--seed", type=int, default=20260523)
    return parser


def run(args: argparse.Namespace) -> dict[str, str]:
    input_path = resolve_project_path(args.input)
    output_dir = resolve_project_path(args.output_dir)
    bundle_path = output_dir / args.bundle_name
    report_path = output_dir / "v021_policy_training_report.md"
    summary_path = output_dir / "v021_policy_training_summary.json"
    metrics_path = output_dir / "v021_policy_training_metrics.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, meta = load_instances(input_path)
    if frame.empty:
        raise ValueError(f"No valid training instances found in {input_path}")

    feature_names = list(FEATURE_COLUMNS)
    model_factories = make_models(args.seed, len(ROUTES))
    task_reports: list[dict[str, Any]] = []
    best_models: dict[str, str] = {}
    fitted_models: dict[str, Any] = {}
    metrics_by_task: dict[str, Any] = {}

    for task_name in TASKS:
        task_frame = frame[frame["task"] == task_name].reset_index(drop=True)
        if task_frame.empty:
            continue
        X = task_frame[feature_names].astype(float)
        y = task_frame["label"].astype(str)
        weights = task_frame["weight"].astype(float).to_numpy()

        model_summaries: list[dict[str, Any]] = []
        evaluated_models: dict[str, dict[str, Any]] = {}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
            warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
            for model_name, factory in model_factories.items():
                metrics = evaluate_model(factory, X, y, weights, task_name, args.seed)
                fitted_model = fit_final_model(factory, X, y, weights)
                evaluated_models[model_name] = {
                    "metrics": metrics,
                    "model": fitted_model,
                    "feature_importance": top_features_from_model(fitted_model, feature_names, metrics["classes"]),
                }
                model_summaries.append(
                    {
                        "model_name": model_name,
                        **metrics,
                        "feature_importance": evaluated_models[model_name]["feature_importance"],
                    }
                )

        ranked = sorted(model_summaries, key=lambda item: item["objective"])
        best = ranked[0]
        best_models[task_name] = best["model_name"]
        best_entry = evaluated_models[best["model_name"]]
        fitted_models[task_name] = {
            "model_name": best["model_name"],
            "model": best_entry["model"],
            "feature_importance": best_entry["feature_importance"],
            "metrics": best_entry["metrics"],
            "classes": best_entry["metrics"]["classes"],
        }
        task_report = {
            "task": task_name,
            "best_model": best["model_name"],
            "best_objective": best["objective"],
            "best_metrics": best,
            "models": ranked,
        }
        task_reports.append(task_report)
        metrics_by_task[task_name] = {
            "best_model": best["model_name"],
            "best_objective": best["objective"],
            "best_metrics": best,
            "model_summaries": ranked,
        }

    bundle = {
        "schema_version": SCHEMA_VERSION,
        "input_path": str(input_path),
        "feature_columns": feature_names,
        "tasks": fitted_models,
        "best_models": best_models,
        "meta": meta,
    }
    joblib.dump(bundle, bundle_path)

    write_json(
        summary_path,
        {
            "schema_version": SCHEMA_VERSION,
            "input_path": str(input_path),
            "output_dir": str(output_dir),
            "bundle_path": str(bundle_path),
            "report_path": str(report_path),
            "metrics_path": str(metrics_path),
            "best_models": best_models,
            "meta": meta,
            "write_permission": False,
        },
    )
    write_json(metrics_path, {"schema_version": SCHEMA_VERSION, "tasks": metrics_by_task})
    write_text(report_path, render_report(task_reports, meta, best_models, feature_names))
    return {
        "bundle": str(bundle_path),
        "report": str(report_path),
        "summary": str(summary_path),
        "metrics": str(metrics_path),
    }


def main() -> int:
    outputs = run(build_arg_parser().parse_args())
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
