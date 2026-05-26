"""Calibrate the v0.21 salience router from proxy data.

This tool is read-only with respect to canonical routing outputs.
It builds proxy training rows from:
- existing v0.21 feature matrices;
- existing v0.21 mode-comparison rows;
- external proxy datasets;
- weak labeling rules.

Then it searches a small classical parameter space and writes a candidate
policy plus audit artifacts. It does not call providers, does not write
durable memory, and does not replace the live router policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.routing_v021_feature_matrix_builder import (
    ExternalEnumerableResources,
    InputUnit,
    build_feature_row,
    load_external_enumerable_resources,
    load_resource_inventory,
    load_resource_store,
    resolve_project_path,
    tfidf_features,
)


SCHEMA_VERSION = "routing.v021_policy_calibration.v0.1"
DEFAULT_OUTPUT_DIR = "reports/v021_policy_calibration"
DEFAULT_CANDIDATE_NAME = "heuristic_salience_v0.21.candidate.yaml"
DEFAULT_RESOURCE_INVENTORY = "external_references/routing_vocab/routing_resource_inventory.csv"
DEFAULT_RANDOM_CANDIDATES = 96
DEFAULT_COORDINATE_PASSES = 2
DEFAULT_MAX_INTERNAL_ROWS = 400
DEFAULT_MAX_EXTERNAL_ROWS = 220
TARGET_TASKS = ("s1_memory_candidate", "s2_portrait_candidate")
ROUTE_ORDER = {
    "skip_or_background_only": 0,
    "script_only": 1,
    "weak_llm_proposal": 2,
    "strong_llm_proposal": 3,
    "human_review": 4,
}


def clamp(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, value))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def round3(value: float) -> float:
    return round(float(value), 3)


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_text(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    return ""


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def normalize_target_route(value: str | None) -> str:
    route = str(value or "").strip()
    if route in ROUTE_ORDER:
        return route
    if route in {"fact_candidate", "portrait_fact_candidate", "memory_candidate"}:
        return "weak_llm_proposal"
    if route in {"hypothesis", "portrait_hypothesis_candidate"}:
        return "strong_llm_proposal"
    return route or "skip_or_background_only"


def route_rank(route: str) -> int:
    return ROUTE_ORDER.get(route, 0)


def route_gap_penalty(predicted: str, target: str) -> float:
    pred_rank = route_rank(predicted)
    target_rank = route_rank(target)
    diff = abs(pred_rank - target_rank)
    penalty = diff * 0.85
    if target == "skip_or_background_only" and predicted in {"weak_llm_proposal", "strong_llm_proposal", "human_review"}:
        penalty += 2.0
    if target == "script_only" and predicted in {"strong_llm_proposal", "human_review"}:
        penalty += 1.4
    if target == "weak_llm_proposal" and predicted == "skip_or_background_only":
        penalty += 1.8
    if target == "strong_llm_proposal" and predicted == "skip_or_background_only":
        penalty += 2.6
    if target == "strong_llm_proposal" and predicted == "script_only":
        penalty += 1.3
    if target == "s1_memory_candidate":  # defensive only
        penalty += 0.0
    return penalty


def route_label_from_score(score: float, thresholds: dict[str, float]) -> str:
    if score < thresholds["script"]:
        return "skip_or_background_only"
    if score < thresholds["weak"]:
        return "script_only"
    if score < thresholds["strong"]:
        return "weak_llm_proposal"
    return "strong_llm_proposal"


def score_axes(axis: dict[str, Any], weights: dict[str, float], penalties: dict[str, float]) -> dict[str, float]:
    value = to_float(axis.get("value_score"))
    entity = to_float(axis.get("entity_salience_score"))
    domain = to_float(axis.get("domain_term_score"))
    keyphrase = to_float(axis.get("keyphrase_score"))
    affect = to_float(axis.get("affect_score"))
    risk = to_float(axis.get("risk_score"))
    complexity = to_float(axis.get("complexity_score"))
    low_value = to_float(axis.get("low_value_score"))
    useful = (
        value * weights["value"]
        + entity * weights["entity"]
        + domain * weights["domain"]
        + keyphrase * weights["keyphrase"]
        + affect * weights["affect"]
    )
    stress = risk * weights["risk"] + complexity * weights["complexity"]
    penalty = low_value * penalties["low_value"] + max(0.0, 10.0 - useful) * penalties["weakness"]
    return {
        "useful_score": round3(useful),
        "stress_score": round3(stress),
        "low_value_score": round3(low_value),
        "penalty_score": round3(penalty),
        "combined_score": round3(useful - penalty - stress * penalties["stress"]),
    }


def predict_route(axis: dict[str, Any], params: dict[str, Any], target_task: str) -> str:
    task = params["task"][target_task]
    weights = params["weights"]
    penalties = params["penalties"]
    scores = score_axes(axis, weights, penalties)
    useful = scores["useful_score"]
    stress = scores["stress_score"]
    low_value = scores["low_value_score"]

    if low_value >= task["low_value_skip_floor"] and useful < task["low_value_useful_floor"]:
        return "skip_or_background_only"
    if useful < task["skip_useful_floor"] and low_value >= task["skip_low_value_floor"]:
        return "skip_or_background_only"
    if useful >= task["strong_useful_floor"] and (stress >= task["strong_stress_floor"] or axis.get("risk_score", 0.0) >= task["strong_risk_floor"]):
        return "strong_llm_proposal"
    if useful >= task["weak_useful_floor"] and (stress >= task["weak_stress_floor"] or axis.get("risk_score", 0.0) >= task["weak_risk_floor"]):
        return "weak_llm_proposal"
    if useful >= task["script_useful_floor"]:
        return "script_only"
    return "skip_or_background_only"


@dataclass
class CalibrationExample:
    example_id: str
    source_kind: str
    dataset_id: str
    source_path: str
    source_row_index: int
    text_preview: str
    language: str
    target_routes: dict[str, str]
    axis_scores: dict[str, float]
    weight: float
    confidence: float
    label_family: str
    label_reason: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "source_kind": self.source_kind,
            "dataset_id": self.dataset_id,
            "source_path": self.source_path,
            "source_row_index": self.source_row_index,
            "text_preview": self.text_preview,
            "language": self.language,
            "target_routes": self.target_routes,
            "axis_scores": self.axis_scores,
            "weight": round3(self.weight),
            "confidence": round3(self.confidence),
            "label_family": self.label_family,
            "label_reason": self.label_reason,
            "metadata": self.metadata,
            "write_permission": False,
        }


def detect_language(text: str, fallback: str = "unknown") -> str:
    zh = len(re.findall(r"[\u4e00-\u9fff]", text))
    en = len(re.findall(r"[A-Za-z]", text))
    if zh > en and zh > 0:
        return "zh"
    if en > 0:
        return "en"
    return fallback


def first_nonempty(value: Any, fallback: str = "") -> str:
    text = safe_text(value)
    return text if text else fallback


def build_text_feature_rows(
    records: list[dict[str, Any]],
    store: Any,
    resources: ExternalEnumerableResources,
    source_layer: str,
    dataset_id: str,
    source_root: Path,
) -> list[dict[str, Any]]:
    if not records:
        return []
    units: list[InputUnit] = []
    for idx, record in enumerate(records):
        text = first_nonempty(record.get("text") or record.get("dialogue") or record.get("summary") or record.get("sentence"))
        source_path = Path(first_nonempty(record.get("source_path"), str(source_root / dataset_id)))
        units.append(
            InputUnit(
                source_layer=source_layer,
                source_path=source_path,
                source_row_index=idx,
                unit_id=str(record.get("example_id") or f"{dataset_id}:{idx + 1:06d}"),
                workspace_id=dataset_id,
                text=text,
                row={"workspace_id": dataset_id, "unit_type": record.get("unit_type") or "proxy_text"},
            )
        )
    tfidf_by_id = tfidf_features({unit.unit_id: unit.text for unit in units}, resources)
    feature_rows: list[dict[str, Any]] = []
    for unit in units:
        feature_rows.append(build_feature_row(unit, store, resources, {}, tfidf_by_id))
    return feature_rows


def internal_examples_from_feature_matrices(paths: list[Path], limit: int) -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
    seen = 0
    for path in paths:
        for row in read_jsonl(path):
            if seen >= limit:
                return examples
            axis = row.get("axis_scores") or {}
            route_snapshot = row.get("route_snapshot") or {}
            target_routes = {
                task: normalize_target_route(route_snapshot.get(task))
                for task in TARGET_TASKS
                if route_snapshot.get(task)
            }
            if not target_routes:
                continue
            examples.append(
                CalibrationExample(
                    example_id=f"internal_feature_matrix:{row.get('workspace_id')}:{row.get('source_layer')}:{row.get('unit_id')}",
                    source_kind="internal_feature_matrix",
                    dataset_id=str(row.get("workspace_id") or "unknown"),
                    source_path=str(path),
                    source_row_index=int(row.get("source_row_index") or 0),
                    text_preview=first_nonempty(row.get("text_preview")),
                    language=first_nonempty((row.get("raw_features") or {}).get("language", {}).get("primary"), "unknown"),
                    target_routes=target_routes,
                    axis_scores={k: round3(to_float(v)) for k, v in axis.items()},
                    weight=0.35,
                    confidence=0.35,
                    label_family="internal_route_snapshot",
                    label_reason="existing route snapshot used as weak calibration label",
                    metadata={
                        "source_layer": row.get("source_layer"),
                        "unit_id": row.get("unit_id"),
                        "feature_status": row.get("feature_status", {}),
                    },
                )
            )
            seen += 1
    return examples


def internal_examples_from_mode_comparison(paths: list[Path], limit: int) -> list[CalibrationExample]:
    examples: list[CalibrationExample] = []
    seen = 0
    for path in paths:
        for row in read_jsonl(path):
            if seen >= limit:
                return examples
            axis = row.get("axis_scores") or {}
            suggested_routes = row.get("suggested_routes") or {}
            target_routes = {
                task: normalize_target_route(suggested_routes.get(task))
                for task in TARGET_TASKS
                if suggested_routes.get(task)
            }
            if not target_routes:
                continue
            examples.append(
                CalibrationExample(
                    example_id=f"internal_mode_comparison:{row.get('workspace_id')}:{row.get('source_layer')}:{row.get('unit_id')}",
                    source_kind="internal_mode_comparison",
                    dataset_id=str(row.get("workspace_id") or "unknown"),
                    source_path=str(path),
                    source_row_index=0,
                    text_preview=first_nonempty(row.get("text_preview")),
                    language="unknown",
                    target_routes=target_routes,
                    axis_scores={k: round3(to_float(v)) for k, v in axis.items()},
                    weight=0.30,
                    confidence=0.30,
                    label_family="internal_comparison_signal",
                    label_reason="existing comparison suggestion used as weak calibration label",
                    metadata={
                        "sample_buckets": row.get("sample_buckets", []),
                        "review_priority_score": row.get("review_priority_score"),
                    },
                )
            )
            seen += 1
    return examples


def low_value_route(axis: dict[str, float], useful_floor: float = 2.5) -> str:
    low = axis.get("low_value_score", 0.0)
    useful = max(
        axis.get("value_score", 0.0),
        axis.get("entity_salience_score", 0.0),
        axis.get("domain_term_score", 0.0),
        axis.get("keyphrase_score", 0.0) * 0.7,
    )
    if low >= 6.0 and useful < useful_floor:
        return "skip_or_background_only"
    if low >= 4.0 and useful < useful_floor + 1.0:
        return "script_only"
    return "weak_llm_proposal" if useful >= useful_floor else "script_only"


def salience_route(axis: dict[str, float], task: str) -> str:
    useful = max(
        axis.get("value_score", 0.0),
        axis.get("entity_salience_score", 0.0),
        axis.get("domain_term_score", 0.0),
        axis.get("keyphrase_score", 0.0),
        axis.get("affect_score", 0.0) * 0.35,
    )
    low = axis.get("low_value_score", 0.0)
    risk = axis.get("risk_score", 0.0)
    complexity = axis.get("complexity_score", 0.0)
    if low >= 5.5 and useful < 2.5:
        return "skip_or_background_only"
    if task == "s1_memory_candidate":
        if useful < 2.5 and low >= 4.0:
            return "skip_or_background_only"
        if useful < 4.0:
            return "script_only"
        if useful < 6.5 or complexity >= 6.0 or risk >= 5.0:
            return "weak_llm_proposal"
        return "strong_llm_proposal"
    if useful < 2.0 and low >= 4.0:
        return "skip_or_background_only"
    if useful < 3.0:
        return "weak_llm_proposal"
    if complexity >= 6.5 or risk >= 5.0:
        return "strong_llm_proposal"
    return "weak_llm_proposal"


def claim_route(axis: dict[str, float], verdict: int) -> str:
    useful = max(axis.get("value_score", 0.0), axis.get("entity_salience_score", 0.0), axis.get("keyphrase_score", 0.0))
    if verdict > 0:
        if useful >= 4.0 or axis.get("risk_score", 0.0) >= 4.0 or axis.get("complexity_score", 0.0) >= 6.0:
            return "strong_llm_proposal"
        return "weak_llm_proposal"
    if useful < 2.0 and axis.get("low_value_score", 0.0) >= 4.0:
        return "skip_or_background_only"
    return "script_only"


def complexity_route(level: str, axis: dict[str, float], task: str) -> str:
    level = level.lower()
    complexity = axis.get("complexity_score", 0.0)
    useful = max(axis.get("value_score", 0.0), axis.get("keyphrase_score", 0.0), axis.get("entity_salience_score", 0.0))
    if task == "s1_memory_candidate":
        if level.endswith("ele"):
            return "skip_or_background_only" if useful < 2.0 else "script_only"
        if level.endswith("int"):
            return "script_only" if useful < 3.0 else "weak_llm_proposal"
        return "weak_llm_proposal" if complexity < 7.0 else "strong_llm_proposal"
    if level.endswith("ele"):
        return "script_only" if useful >= 1.5 else "skip_or_background_only"
    if level.endswith("int"):
        return "weak_llm_proposal"
    return "weak_llm_proposal" if complexity < 7.0 else "strong_llm_proposal"


def load_xdailydialog_records(dataset_root: Path, max_rows: int) -> list[dict[str, Any]]:
    data_dir = dataset_root / "data"
    if not data_dir.exists():
        data_dir = dataset_root
    records: list[dict[str, Any]] = []
    file_paths = sorted(
        path for path in data_dir.glob("*_human.txt")
        if path.is_file()
    )
    for path in file_paths:
        language = "zh" if "_zh_" in path.name.lower() else "en" if "_en_" in path.name.lower() else "unknown"
        for line_no, line in enumerate(path.read_text(encoding="utf-8-sig", errors="ignore").splitlines(), 1):
            if len(records) >= max_rows:
                return records
            if not line.strip():
                continue
            parts = line.split("\t")
            dialogue = safe_text(parts[0])
            if not dialogue:
                continue
            records.append(
                {
                    "example_id": f"xdailydialog:{path.stem}:{line_no}",
                    "dataset_id": "xdailydialog",
                    "source_path": str(path),
                    "source_row_index": line_no - 1,
                    "text": dialogue.replace("__eou__", " "),
                    "language": language,
                    "unit_type": "dialogue",
                    "metadata": {
                        "raw_fields": len(parts),
                        "labels": parts[1:],
                    },
                }
            )
    return records


def load_dialogsum_records(dataset_root: Path, max_rows: int) -> list[dict[str, Any]]:
    data_dir = dataset_root / "DialogSum_Data"
    if not data_dir.exists():
        data_dir = dataset_root
    files = [path for path in sorted(data_dir.glob("dialogsum.*.jsonl")) if path.is_file() and "hidden" not in path.name]
    records: list[dict[str, Any]] = []
    for path in files:
        for idx, row in enumerate(read_jsonl(path)):
            if len(records) >= max_rows:
                return records
            dialogue = safe_text(row.get("dialogue"))
            if not dialogue:
                continue
            records.append(
                {
                    "example_id": f"dialogsum:{path.stem}:{idx}",
                    "dataset_id": "dialogsum",
                    "source_path": str(path),
                    "source_row_index": idx,
                    "text": dialogue,
                    "language": "en",
                    "unit_type": "dialogue",
                    "metadata": {
                        "summary": safe_text(row.get("summary")),
                        "topic": safe_text(row.get("topic")),
                        "fname": safe_text(row.get("fname")),
                    },
                }
            )
    return records


def load_claimbuster_records(dataset_root: Path, max_rows: int) -> list[dict[str, Any]]:
    csv_path = dataset_root / "unzipped" / "ClaimBuster_Datasets" / "datasets" / "groundtruth.csv"
    rows = read_csv_rows(csv_path)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if len(records) >= max_rows:
            return records
        text = safe_text(row.get("Text"))
        if not text:
            continue
        verdict = int(float(row.get("Verdict") or 0))
        records.append(
            {
                "example_id": f"claimbuster:{row.get('File_id') or 'unknown'}:{row.get('Sentence_id') or idx}",
                "dataset_id": "claimbuster",
                "source_path": str(csv_path),
                "source_row_index": idx,
                "text": text,
                "language": "en",
                "unit_type": "sentence",
                "metadata": {
                    "verdict": verdict,
                    "speaker": safe_text(row.get("Speaker")),
                    "speaker_title": safe_text(row.get("Speaker_title")),
                    "file_id": safe_text(row.get("File_id")),
                    "sentiment": safe_text(row.get("Sentiment")),
                },
            }
        )
    return records


def load_onestopenglish_records(dataset_root: Path, max_rows: int) -> list[dict[str, Any]]:
    csv_path = dataset_root / "allfeatures-ose-final.csv"
    rows = read_csv_rows(csv_path)
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if len(records) >= max_rows:
            return records
        file_name = safe_text(row.get("fileName"))
        if not file_name:
            continue
        level = "unknown"
        m = re.search(r"-(adv|int|ele)\.txt$", file_name, flags=re.IGNORECASE)
        if m:
            level = m.group(1).lower()
        records.append(
            {
                "example_id": f"onestopenglish:{file_name}",
                "dataset_id": "onestopenglish",
                "source_path": str(csv_path),
                "source_row_index": idx,
                "text": file_name,
                "language": "en",
                "unit_type": "file_level_feature_proxy",
                "metadata": {
                    "reading_level": level,
                    "row": row,
                },
            }
        )
    return records


def load_lifelog_summary(dataset_root: Path) -> dict[str, Any]:
    zip_path = dataset_root / "DiaLog_v1.zip"
    json_path = dataset_root / "DiaLog_v1" / "DiaLog" / "DiaLog.json"
    label_path = dataset_root / "DiaLog_v1" / "DiaLog" / "label.txt"
    summary = {
        "dataset_id": "lifelog_dialog",
        "source_path": str(zip_path),
        "status": "label_only",
        "reason": "local checkout does not expose dialogue text beyond DiaLog.json event labels",
        "json_exists": json_path.exists(),
        "label_exists": label_path.exists(),
        "turn_sample": None,
        "event_labels": [],
    }
    if json_path.exists():
        try:
            data = read_json(json_path)
            if isinstance(data, list) and data:
                summary["turn_sample"] = data[0].get("turns")
                event_labels = set()
                for item in data[:20]:
                    for event in item.get("events", []):
                        for labels in event.values():
                            if isinstance(labels, list):
                                event_labels.update(str(label) for label in labels)
                summary["event_labels"] = sorted(event_labels)[:20]
        except Exception as exc:
            summary["status"] = "error"
            summary["error"] = type(exc).__name__
    return summary


def make_text_examples(
    records: list[dict[str, Any]],
    store: Any,
    resources: ExternalEnumerableResources,
) -> list[CalibrationExample]:
    if not records:
        return []
    feature_rows = build_text_feature_rows(
        records=records,
        store=store,
        resources=resources,
        source_layer="proxy_dataset",
        dataset_id=str(records[0]["dataset_id"]),
        source_root=Path(records[0]["source_path"]).parent,
    )
    examples: list[CalibrationExample] = []
    for record, feature in zip(records, feature_rows):
        axis = {k: round3(to_float(v)) for k, v in (feature.get("axis_scores") or {}).items()}
        dataset_id = str(record["dataset_id"])
        metadata = record.get("metadata") or {}
        label_family = "salience_proxy"
        confidence = 0.8
        weight = 0.9
        if dataset_id == "xdailydialog":
            label_family = "dialogue_proxy"
            target_routes = {
                "s1_memory_candidate": low_value_route(axis, useful_floor=2.8),
                "s2_portrait_candidate": salience_route(axis, "s2_portrait_candidate"),
            }
            confidence = 0.7
            weight = 0.8
            if axis.get("value_score", 0.0) >= 5.0 or axis.get("entity_salience_score", 0.0) >= 4.0:
                target_routes["s1_memory_candidate"] = "weak_llm_proposal"
        elif dataset_id == "dialogsum":
            label_family = "summary_proxy"
            target_routes = {
                "s1_memory_candidate": salience_route(axis, "s1_memory_candidate"),
                "s2_portrait_candidate": salience_route(axis, "s2_portrait_candidate"),
            }
            confidence = 0.95
            weight = 1.0
            if axis.get("complexity_score", 0.0) >= 6.0 or axis.get("risk_score", 0.0) >= 4.0:
                target_routes["s2_portrait_candidate"] = "strong_llm_proposal"
        elif dataset_id == "claimbuster":
            verdict = int(metadata.get("verdict") or 0)
            label_family = "factual_checkworthy_proxy"
            target_routes = {
                "s1_memory_candidate": claim_route(axis, verdict),
                "s2_portrait_candidate": "strong_llm_proposal" if verdict > 0 else claim_route(axis, verdict),
            }
            confidence = 1.0 if verdict != 0 else 0.8
            weight = 1.2 if verdict > 0 else 0.9
        else:
            target_routes = {
                "s1_memory_candidate": salience_route(axis, "s1_memory_candidate"),
                "s2_portrait_candidate": salience_route(axis, "s2_portrait_candidate"),
            }
        examples.append(
            CalibrationExample(
                example_id=str(record["example_id"]),
                source_kind="external_proxy_text",
                dataset_id=dataset_id,
                source_path=str(record["source_path"]),
                source_row_index=int(record["source_row_index"]),
                text_preview=str(feature.get("text_preview") or record.get("text") or "")[:300],
                language=first_nonempty(record.get("language"), "unknown"),
                target_routes=target_routes,
                axis_scores=axis,
                weight=weight,
                confidence=confidence,
                label_family=label_family,
                label_reason=f"{dataset_id} proxy-derived target routes",
                metadata={
                    "unit_type": record.get("unit_type"),
                    "raw_metadata": metadata,
                    "feature_status": feature.get("feature_status", {}),
                },
            )
        )
    return examples


def make_onestopenglish_examples(records: list[dict[str, Any]]) -> list[CalibrationExample]:
    if not records:
        return []
    rows: list[CalibrationExample] = []
    for idx, record in enumerate(records):
        metadata = record.get("metadata") or {}
        csv_row = metadata.get("row") or {}
        level = str(metadata.get("reading_level") or "unknown").lower()
        values = [to_float(csv_row.get(key)) for key in csv_row.keys() if key.startswith(("AoA_", "MRC", "DISC_", "POS_"))]
        aoa_values = [to_float(csv_row.get(key)) for key in csv_row.keys() if key.startswith("AoA_")]
        mrc_values = [to_float(csv_row.get(key)) for key in csv_row.keys() if key.startswith("MRC")]
        disc_values = [to_float(csv_row.get(key)) for key in csv_row.keys() if key.startswith("DISC_")]
        pos_values = [to_float(csv_row.get(key)) for key in csv_row.keys() if key.startswith("POS_")]
        complexity = clamp(
            mean(aoa_values) * 1.8
            + (10.0 - mean(mrc_values)) * 0.9
            + mean(disc_values) * 0.6
            + mean(pos_values) * 0.15
        )
        axis = {
            "value_score": 1.0,
            "risk_score": 0.0,
            "complexity_score": round3(complexity),
            "entity_salience_score": 0.0,
            "keyphrase_score": 0.0,
            "affect_score": 0.0,
            "low_value_score": 0.5,
            "domain_term_score": 0.0,
        }
        s1_route = complexity_route(level, axis, "s1_memory_candidate")
        s2_route = complexity_route(level, axis, "s2_portrait_candidate")
        rows.append(
            CalibrationExample(
                example_id=str(record["example_id"]),
                source_kind="external_proxy_numeric",
                dataset_id="onestopenglish",
                source_path=str(record["source_path"]),
                source_row_index=idx,
                text_preview=str(record["text"]),
                language="en",
                target_routes={
                    "s1_memory_candidate": s1_route,
                    "s2_portrait_candidate": s2_route,
                },
                axis_scores=axis,
                weight=0.85,
                confidence=0.82,
                label_family="complexity_proxy",
                label_reason=f"OneStopEnglish reading-level proxy ({level})",
                metadata={
                    "file_name": record["text"],
                    "reading_level": level,
                },
            )
        )
    return rows


def build_examples(project_root: Path, max_internal_rows: int, max_external_rows: int, inventory_path: Path) -> tuple[list[CalibrationExample], dict[str, Any]]:
    inventory = load_resource_inventory(project_root, inventory_path)
    enumerable_resources = load_external_enumerable_resources(inventory)
    store = load_resource_store(inventory, enumerable_resources)

    feature_matrix_paths = sorted(
        path
        for path in project_root.rglob("v021_feature_matrix*.jsonl")
        if "routing" in path.parts and "reports" not in path.parts
    )
    comparison_paths = sorted(
        path
        for path in project_root.rglob("v021_mode_comparison_rows.jsonl")
        if "routing" in path.parts and "reports" not in path.parts
    )
    internal_feature_examples = internal_examples_from_feature_matrices(feature_matrix_paths, limit=max_internal_rows)
    internal_comparison_examples = internal_examples_from_mode_comparison(comparison_paths, limit=max_internal_rows)

    external_root = project_root / "external_references" / "routing_calibration_datasets"
    xd_records = load_xdailydialog_records(external_root / "XDailyDialog", max_external_rows)
    ds_records = load_dialogsum_records(external_root / "DialogSum", max_external_rows)
    cb_records = load_claimbuster_records(external_root / "ClaimBuster", max_external_rows)
    ose_records = load_onestopenglish_records(external_root / "OneStopEnglish", max_external_rows)
    lifelog_summary = load_lifelog_summary(external_root / "Lifelog-DiaLog")

    external_examples = (
        make_text_examples(xd_records, store, enumerable_resources)
        + make_text_examples(ds_records, store, enumerable_resources)
        + make_text_examples(cb_records, store, enumerable_resources)
        + make_onestopenglish_examples(ose_records)
    )

    source_summary = {
        "project_root": str(project_root),
        "inventory_path": str(inventory_path),
        "resource_status": inventory.by_id,
        "feature_matrix_files": [
            {"path": str(path), "sha256": sha256_file(path), "row_count": len(read_jsonl(path))}
            for path in feature_matrix_paths
        ],
        "comparison_files": [
            {"path": str(path), "sha256": sha256_file(path), "row_count": len(read_jsonl(path))}
            for path in comparison_paths
        ],
        "external_dataset_counts": {
            "xdailydialog": len(xd_records),
            "dialogsum": len(ds_records),
            "claimbuster": len(cb_records),
            "onestopenglish": len(ose_records),
        },
        "external_dataset_notes": {
            "lifelog_dialog": lifelog_summary,
        },
    }
    return internal_feature_examples + internal_comparison_examples + external_examples, source_summary


def build_seed_params() -> dict[str, Any]:
    return {
        "weights": {
            "value": 1.05,
            "entity": 1.22,
            "domain": 1.0,
            "keyphrase": 0.58,
            "affect": 0.32,
            "risk": 0.88,
            "complexity": 0.82,
        },
        "penalties": {
            "low_value": 1.35,
            "stress": 0.20,
            "weakness": 0.10,
        },
        "task": {
            "s1_memory_candidate": {
                "low_value_skip_floor": 5.0,
                "low_value_useful_floor": 2.4,
                "skip_useful_floor": 2.0,
                "skip_low_value_floor": 4.0,
                "script_useful_floor": 3.0,
                "weak_useful_floor": 4.8,
                "weak_stress_floor": 4.1,
                "weak_risk_floor": 3.2,
                "strong_useful_floor": 6.5,
                "strong_stress_floor": 5.8,
                "strong_risk_floor": 5.2,
            },
            "s2_portrait_candidate": {
                "low_value_skip_floor": 5.5,
                "low_value_useful_floor": 2.0,
                "skip_useful_floor": 1.6,
                "skip_low_value_floor": 4.2,
                "script_useful_floor": 2.6,
                "weak_useful_floor": 3.7,
                "weak_stress_floor": 3.5,
                "weak_risk_floor": 2.9,
                "strong_useful_floor": 5.8,
                "strong_stress_floor": 5.3,
                "strong_risk_floor": 4.8,
            },
        },
    }


def normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    weights = params["weights"]
    penalties = params["penalties"]
    for key in weights:
        weights[key] = round3(clamp(float(weights[key]), 0.1, 3.0))
    for key in penalties:
        penalties[key] = round3(clamp(float(penalties[key]), 0.0, 3.0))
    for task, task_params in params["task"].items():
        for key in task_params:
            task_params[key] = round3(clamp(float(task_params[key]), 0.0, 10.0))
    return params


def mutate_params(params: dict[str, Any], rng: random.Random, scale: float = 0.15) -> dict[str, Any]:
    candidate = json.loads(json.dumps(params))
    for section in ("weights", "penalties"):
        for key, value in candidate[section].items():
            delta = rng.uniform(-scale, scale)
            candidate[section][key] = float(value) * (1.0 + delta)
    for task_name, task_params in candidate["task"].items():
        for key, value in task_params.items():
            if "floor" not in key:
                continue
            delta = rng.uniform(-scale * 1.5, scale * 1.5)
            candidate["task"][task_name][key] = float(value) + delta * max(1.0, float(value))
    return normalize_params(candidate)


def evaluate_candidate(examples: list[CalibrationExample], params: dict[str, Any]) -> dict[str, Any]:
    stats = {
        "exact": 0.0,
        "weighted_total": 0.0,
        "route_penalty": 0.0,
        "false_skip_high_value_penalty": 0.0,
        "over_strong_llm_penalty": 0.0,
        "s2_skip_too_much_penalty": 0.0,
        "low_value_llm_penalty": 0.0,
        "missing_provenance_penalty": 0.0,
        "count": 0.0,
        "target_route_counts": Counter(),
        "predicted_route_counts": Counter(),
        "task_counts": Counter(),
        "dataset_counts": Counter(),
    }
    per_task: dict[str, Counter[str]] = defaultdict(Counter)
    for example in examples:
        for target_task in TARGET_TASKS:
            target_route = example.target_routes.get(target_task)
            if not target_route:
                continue
            predicted = predict_route(example.axis_scores, params, target_task)
            weight = float(example.weight)
            stats["count"] += 1.0
            stats["weighted_total"] += weight
            stats["target_route_counts"][target_route] += 1
            stats["predicted_route_counts"][predicted] += 1
            stats["task_counts"][target_task] += 1
            stats["dataset_counts"][example.dataset_id] += 1
            per_task[target_task][f"{target_route}->{predicted}"] += 1
            penalty = route_gap_penalty(predicted, target_route)
            value = example.axis_scores.get("value_score", 0.0)
            useful = max(
                example.axis_scores.get("value_score", 0.0),
                example.axis_scores.get("entity_salience_score", 0.0),
                example.axis_scores.get("domain_term_score", 0.0),
                example.axis_scores.get("keyphrase_score", 0.0),
            )
            low_value = example.axis_scores.get("low_value_score", 0.0)
            risk = example.axis_scores.get("risk_score", 0.0)
            complexity = example.axis_scores.get("complexity_score", 0.0)
            if target_route in {"weak_llm_proposal", "strong_llm_proposal"} and predicted == "skip_or_background_only" and useful >= 4.0:
                stats["false_skip_high_value_penalty"] += weight * 1.8
                penalty += 1.8
            if target_route in {"skip_or_background_only", "script_only"} and predicted in {"strong_llm_proposal", "human_review"}:
                stats["over_strong_llm_penalty"] += weight * 1.2
                penalty += 1.2
            if target_task == "s2_portrait_candidate" and predicted == "skip_or_background_only" and target_route != "skip_or_background_only":
                stats["s2_skip_too_much_penalty"] += weight * 1.3
                penalty += 1.3
            if low_value >= 5.0 and predicted in {"weak_llm_proposal", "strong_llm_proposal"}:
                stats["low_value_llm_penalty"] += weight * 0.8
                penalty += 0.8
            if value >= 5.0 and predicted == "skip_or_background_only":
                stats["false_skip_high_value_penalty"] += weight * 0.7
                penalty += 0.7
            if example.weight < 0.5 and predicted == "human_review":
                stats["over_strong_llm_penalty"] += weight * 0.4
                penalty += 0.4
            stats["route_penalty"] += weight * penalty
            if predicted == target_route:
                stats["exact"] += weight
            stats["missing_provenance_penalty"] += 0.0 if example.source_path else weight * 0.1
            if risk >= 8.5 and predicted == "script_only":
                stats["over_strong_llm_penalty"] += weight * 0.3
                stats["route_penalty"] += weight * 0.3
            if complexity >= 7.5 and predicted == "skip_or_background_only" and target_route != "skip_or_background_only":
                stats["false_skip_high_value_penalty"] += weight * 0.3
                stats["route_penalty"] += weight * 0.3
    total = max(1.0, stats["weighted_total"])
    exact_rate = stats["exact"] / total
    objective = stats["route_penalty"] / total
    penalty_terms = (
        stats["false_skip_high_value_penalty"]
        + stats["over_strong_llm_penalty"]
        + stats["s2_skip_too_much_penalty"]
        + stats["low_value_llm_penalty"]
        + stats["missing_provenance_penalty"]
    ) / total
    return {
        "objective": round3(objective + penalty_terms * 0.25 - exact_rate * 0.15),
        "exact_match_weighted": round3(exact_rate),
        "route_penalty_weighted": round3(objective),
        "penalty_terms_weighted": round3(penalty_terms),
        "false_skip_high_value_penalty": round3(stats["false_skip_high_value_penalty"] / total),
        "over_strong_llm_penalty": round3(stats["over_strong_llm_penalty"] / total),
        "s2_skip_too_much_penalty": round3(stats["s2_skip_too_much_penalty"] / total),
        "low_value_llm_penalty": round3(stats["low_value_llm_penalty"] / total),
        "missing_provenance_penalty": round3(stats["missing_provenance_penalty"] / total),
        "target_route_counts": dict(stats["target_route_counts"]),
        "predicted_route_counts": dict(stats["predicted_route_counts"]),
        "task_counts": dict(stats["task_counts"]),
        "dataset_counts": dict(stats["dataset_counts"]),
        "confusion": {task: dict(counter) for task, counter in per_task.items()},
    }


def candidate_summary(candidate_id: str, params: dict[str, Any], metrics: dict[str, Any], source_tag: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "source_tag": source_tag,
        "objective": metrics["objective"],
        "exact_match_weighted": metrics["exact_match_weighted"],
        "route_penalty_weighted": metrics["route_penalty_weighted"],
        "penalty_terms_weighted": metrics["penalty_terms_weighted"],
        "weights": params["weights"],
        "penalties": params["penalties"],
        "task": params["task"],
        "dataset_counts": metrics["dataset_counts"],
        "task_counts": metrics["task_counts"],
        "target_route_counts": metrics["target_route_counts"],
        "predicted_route_counts": metrics["predicted_route_counts"],
        "confusion": metrics["confusion"],
        "write_permission": False,
    }


def seed_candidates() -> list[dict[str, Any]]:
    base = normalize_params(build_seed_params())
    candidates = [base]
    variants = [
        ("balanced", {"weights": {"value": 1.10, "entity": 1.28, "domain": 1.00, "keyphrase": 0.62, "affect": 0.30, "risk": 0.90, "complexity": 0.82}}),
        ("entity_heavier", {"weights": {"value": 1.00, "entity": 1.35, "domain": 1.05, "keyphrase": 0.58, "affect": 0.28, "risk": 0.88, "complexity": 0.80}}),
        ("keyphrase_heavier", {"weights": {"value": 1.00, "entity": 1.18, "domain": 1.05, "keyphrase": 0.72, "affect": 0.30, "risk": 0.88, "complexity": 0.80}}),
        ("risk_focused", {"weights": {"value": 1.00, "entity": 1.15, "domain": 0.95, "keyphrase": 0.55, "affect": 0.25, "risk": 1.05, "complexity": 0.95}, "penalties": {"stress": 0.26}}),
    ]
    for _, patch in variants:
        candidate = json.loads(json.dumps(base))
        for section, values in patch.items():
            for key, value in values.items():
                candidate[section][key] = value
        candidates.append(normalize_params(candidate))
    return candidates


def search_candidates(examples: list[CalibrationExample], random_count: int, coordinate_passes: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    pool: list[tuple[str, dict[str, Any], dict[str, Any]]] = []

    def add_candidate(source_tag: str, params: dict[str, Any]) -> None:
        metrics = evaluate_candidate(examples, params)
        pool.append((source_tag, params, metrics))

    for idx, params in enumerate(seed_candidates()):
        add_candidate(f"seed_{idx}", params)

    best_params = min(pool, key=lambda item: item[2]["objective"])[1]
    for idx in range(random_count):
        add_candidate(f"random_{idx}", mutate_params(best_params, rng, scale=0.20 if idx < random_count // 2 else 0.12))

    for pass_index in range(coordinate_passes):
        ranked = sorted(pool, key=lambda item: item[2]["objective"])
        base_params = ranked[0][1]
        for section in ("weights", "penalties"):
            for key in base_params[section]:
                for delta in (-0.18, -0.08, 0.08, 0.18):
                    candidate = json.loads(json.dumps(base_params))
                    candidate[section][key] = float(candidate[section][key]) * (1.0 + delta)
                    add_candidate(f"coord_{pass_index}_{section}_{key}_{delta:+.2f}", normalize_params(candidate))
        for task_name in base_params["task"]:
            for key in base_params["task"][task_name]:
                for delta in (-0.45, -0.20, 0.20, 0.45):
                    candidate = json.loads(json.dumps(base_params))
                    candidate["task"][task_name][key] = float(candidate["task"][task_name][key]) + delta
                    add_candidate(f"coord_{pass_index}_{task_name}_{key}_{delta:+.2f}", normalize_params(candidate))

    unique: dict[str, tuple[str, dict[str, Any], dict[str, Any]]] = {}
    for source_tag, params, metrics in pool:
        key = json.dumps({"weights": params["weights"], "penalties": params["penalties"], "task": params["task"]}, sort_keys=True)
        if key not in unique or metrics["objective"] < unique[key][2]["objective"]:
            unique[key] = (source_tag, params, metrics)
    ranked_candidates = sorted(unique.values(), key=lambda item: item[2]["objective"])
    return [candidate_summary(f"candidate_{idx:03d}", params, metrics, source_tag) for idx, (source_tag, params, metrics) in enumerate(ranked_candidates)]


def dump_yaml(value: Any, indent: int = 0) -> str:
    pad = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}{key}:")
                lines.append(dump_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}{key}: {yaml_scalar(item)}")
        return "\n".join(lines)
    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(dump_yaml(item, indent + 2))
            else:
                lines.append(f"{pad}- {yaml_scalar(item)}")
        return "\n".join(lines)
    return f"{pad}{yaml_scalar(value)}"


def yaml_scalar(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "null"
        if isinstance(value, float):
            return f"{value:.6f}".rstrip("0").rstrip(".")
        return str(value)
    text = str(value)
    if not text:
        return '""'
    if re.fullmatch(r"[A-Za-z0-9_.:/+-]+", text):
        return text
    return json.dumps(text, ensure_ascii=False)


def render_report(examples: list[CalibrationExample], candidates: list[dict[str, Any]], source_summary: dict[str, Any]) -> str:
    label_counts = Counter(example.label_family for example in examples)
    source_counts = Counter(example.source_kind for example in examples)
    dataset_counts = Counter(example.dataset_id for example in examples)
    if candidates:
        best = candidates[0]
    else:
        best = {}
    lines = [
        "# v0.21 Router Calibration Report",
        "",
        f"- schema_version: `{SCHEMA_VERSION}`",
        f"- examples: `{len(examples)}`",
        f"- candidates: `{len(candidates)}`",
        "",
        "## Example Sources",
        "",
    ]
    lines.extend(f"- `{key}`: {value}" for key, value in source_counts.most_common())
    lines.extend(["", "## Dataset Counts", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in dataset_counts.most_common())
    lines.extend(["", "## Label Families", ""])
    lines.extend(f"- `{key}`: {value}" for key, value in label_counts.most_common())
    lines.extend(["", "## External Dataset Notes", ""])
    for dataset_id, note in (source_summary.get("external_dataset_notes") or {}).items():
        lines.append(f"- `{dataset_id}`: {note.get('status')} - {note.get('reason')}")
    lines.extend(["", "## Best Candidate", ""])
    if best:
        lines.extend(
            [
                f"- candidate_id: `{best['candidate_id']}`",
                f"- source_tag: `{best['source_tag']}`",
                f"- objective: `{best['objective']}`",
                f"- exact_match_weighted: `{best['exact_match_weighted']}`",
                f"- route_penalty_weighted: `{best['route_penalty_weighted']}`",
                f"- penalty_terms_weighted: `{best['penalty_terms_weighted']}`",
            ]
        )
    lines.extend(["", "## Top Candidates", ""])
    for candidate in candidates[:12]:
        lines.append(
            f"- `{candidate['candidate_id']}` obj={candidate['objective']} exact={candidate['exact_match_weighted']} "
            f"source={candidate['source_tag']}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a proxy-data warm-start, not a learned router.",
            "- Proxy labels are not route truth.",
            "- Candidate policy is not the default policy until reviewed.",
            "- No provider calls, durable memory writes, graph writes, or S3 work were performed.",
            "",
        ]
    )
    return "\n".join(lines)


def build_candidate_policy(best: dict[str, Any], source_summary: dict[str, Any], examples: list[CalibrationExample], candidates: list[dict[str, Any]]) -> dict[str, Any]:
    weights = best["weights"]
    penalties = best["penalties"]
    task = best["task"]
    return {
        "schema_version": "memory_proposal_router.policy.v0.21.candidate",
        "policy_id": "memory-proposal-router-policy:v0.21:heuristic-salience-calibrated-candidate",
        "policy_version": "0.21-candidate",
        "router_backend": "heuristic_salience_v0_21_calibrated",
        "feature_extractor": "salience_v0_21_calibrated",
        "feature_extractor_version": "memory_proposal_features.salience.v0.21.calibrated",
        "router_version": "memory_proposal_router.heuristic_salience_backend.v0.21.calibrated",
        "matcher_version": "salience_zh_en_phrase_token.v0.2",
        "route_confidence_type": "heuristic_score_calibrated",
        "target_tasks": list(TARGET_TASKS),
        "blocked_target_tasks": ["s3_hypothesis_candidate"],
        "weights": {
            "value_score": weights["value"],
            "entity_salience_score": weights["entity"],
            "domain_term_score": weights["domain"],
            "keyphrase_score": weights["keyphrase"],
            "affect_score": weights["affect"],
            "risk_score": weights["risk"],
            "complexity_score": weights["complexity"],
            "low_value_score": penalties["low_value"],
        },
        "thresholds": {
            "s1": task["s1_memory_candidate"],
            "s2": task["s2_portrait_candidate"],
            "route_score_max": 12,
        },
        "routes": {
            "background": {
                "recommended_route": "skip_or_background_only",
                "fallback_route": "weak_llm_proposal",
                "cost_class": "none",
            },
            "preprocessing_required": {
                "recommended_route": "split_or_segment_first",
                "fallback_route": "weak_llm_proposal",
                "cost_class": "low",
            },
            "script": {
                "recommended_route": "script_only",
                "fallback_route": "weak_llm_proposal",
                "cost_class": "low",
            },
            "weak": {
                "recommended_route": "weak_llm_proposal",
                "fallback_route": "strong_llm_proposal",
                "cost_class": "medium",
            },
            "strong": {
                "recommended_route": "strong_llm_proposal",
                "fallback_route": "human_review",
                "cost_class": "high",
            },
            "human": {
                "recommended_route": "human_review",
                "fallback_route": "human_review",
                "cost_class": "high",
            },
        },
        "calibration": {
            "schema_version": SCHEMA_VERSION,
            "method": "proxy_warm_start_search",
            "objective": best["objective"],
            "exact_match_weighted": best["exact_match_weighted"],
            "route_penalty_weighted": best["route_penalty_weighted"],
            "penalty_terms_weighted": best["penalty_terms_weighted"],
            "best_candidate_id": best["candidate_id"],
            "candidate_count": len(candidates),
            "training_example_count": len(examples),
            "source_summary": {
                "feature_matrix_files": source_summary.get("feature_matrix_files", []),
                "comparison_files": source_summary.get("comparison_files", []),
                "external_dataset_counts": source_summary.get("external_dataset_counts", {}),
                "external_dataset_notes": source_summary.get("external_dataset_notes", {}),
            },
            "notes": [
                "proxy labels only",
                "not route truth",
                "not default policy",
                "no provider calls",
            ],
        },
        "guardrails": {
            "write_permission": False,
            "llm_calls_executed": False,
            "proposal_generation_executed": False,
            "durable_writes_executed": False,
            "automatic_cascade_executed": False,
            "support_check_candidate_sets_support_status": False,
            "s3_hypothesis_candidate_emitted": False,
        },
    }


def run(args: argparse.Namespace) -> dict[str, str]:
    project_root = resolve_project_path(Path.cwd(), getattr(args, "project_root", "."))
    inventory_path = resolve_project_path(project_root, getattr(args, "resource_inventory", DEFAULT_RESOURCE_INVENTORY))
    output_dir = resolve_project_path(project_root, args.output_dir) if args.output_dir else project_root / DEFAULT_OUTPUT_DIR
    candidate_path = resolve_project_path(project_root, args.candidate_path) if args.candidate_path else project_root / "configs" / "routing" / "memory_proposal_router" / DEFAULT_CANDIDATE_NAME

    examples, source_summary = build_examples(
        project_root=project_root,
        max_internal_rows=args.max_internal_rows,
        max_external_rows=args.max_external_rows,
        inventory_path=inventory_path,
    )
    candidates = search_candidates(
        examples=examples,
        random_count=args.random_candidates,
        coordinate_passes=args.coordinate_passes,
        seed=args.seed,
    )

    best = candidates[0]
    training_rows_path = output_dir / "v021_calibration_training_rows.jsonl"
    candidates_path = output_dir / "v021_calibration_candidates.jsonl"
    report_path = output_dir / "v021_calibration_report.md"
    summary_path = output_dir / "v021_calibration_summary.json"
    policy = build_candidate_policy(
        best=next(
            item for item in candidates if item["candidate_id"] == best["candidate_id"]
        ),
        source_summary=source_summary,
        examples=examples,
        candidates=candidates,
    )

    write_jsonl(training_rows_path, [example.as_json() for example in examples])
    write_jsonl(candidates_path, candidates)
    write_text(report_path, render_report(examples, candidates, source_summary))
    write_json(summary_path, {
        "schema_version": SCHEMA_VERSION,
        "example_count": len(examples),
        "candidate_count": len(candidates),
        "best_candidate_id": best["candidate_id"],
        "best_objective": best["objective"],
        "training_rows_path": str(training_rows_path),
        "candidates_path": str(candidates_path),
        "report_path": str(report_path),
        "candidate_policy_path": str(candidate_path),
        "source_summary": source_summary,
        "write_permission": False,
    })
    write_text(candidate_path, dump_yaml(policy) + "\n")
    return {
        "training_rows": str(training_rows_path),
        "candidates": str(candidates_path),
        "report": str(report_path),
        "summary": str(summary_path),
        "candidate_policy": str(candidate_path),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Proxy-data warm-start calibration for the v0.21 salience router.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current working directory.")
    parser.add_argument("--resource-inventory", default=DEFAULT_RESOURCE_INVENTORY, help="Relative or absolute resource inventory path.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory for calibration artifacts.")
    parser.add_argument("--candidate-path", default=None, help="Path for heuristic_salience_v0.21.candidate.yaml.")
    parser.add_argument("--max-internal-rows", type=int, default=DEFAULT_MAX_INTERNAL_ROWS)
    parser.add_argument("--max-external-rows", type=int, default=DEFAULT_MAX_EXTERNAL_ROWS)
    parser.add_argument("--random-candidates", type=int, default=DEFAULT_RANDOM_CANDIDATES)
    parser.add_argument("--coordinate-passes", type=int, default=DEFAULT_COORDINATE_PASSES)
    parser.add_argument("--seed", type=int, default=20260523)
    return parser


def main() -> int:
    outputs = run(build_arg_parser().parse_args())
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
