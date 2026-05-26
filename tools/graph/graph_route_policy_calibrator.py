"""Calibrate v0.3 graph package router parameters from external RE/OpenIE data.

The calibration data is proxy data. It warms up route parameters for graph
package selection; it is not route truth, graph truth, or extraction truth.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tools.graph.graph_construction_packet_builder import sha256_text, stable_id, write_json, write_jsonl, write_text
from tools.graph.graph_package_router import DEFAULT_POLICY, graph_route_features, graph_route_for_features


SCHEMA_VERSION = "graph_v03.route_policy_calibration.v0.1"
DEFAULT_DATASET_ROOT = "external_references/graph_route_calibration_datasets"
DEFAULT_OUTPUT_DIR = "external_references/graph_route_calibration_datasets/calibration"
DEFAULT_POLICY_OUTPUT = "configs/routing/graph_package_router/graph_route_policy.v0.3.candidate.json"


@dataclass
class ProxyExample:
    dataset_id: str
    split: str
    source_ref: str
    text: str
    target_route: str
    proxy_label: str
    metadata: dict[str, Any]


def file_hash(path: Path) -> str | None:
    return sha256_text(path.read_text(encoding="utf-8", errors="ignore")) if path.exists() else None


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_carb(root: Path, limit_per_split: int) -> Iterable[ProxyExample]:
    for split in ("dev", "test"):
        path = root / "CaRB" / "data" / "gold" / f"{split}.tsv"
        if not path.exists():
            continue
        seen: set[str] = set()
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                sentence, relation, arg1, arg2 = parts[:4]
                if sentence in seen:
                    continue
                seen.add(sentence)
                count += 1
                yield ProxyExample(
                    dataset_id="carb_openie",
                    split=split,
                    source_ref=f"CaRB/data/gold/{split}.tsv:{count}",
                    text=sentence,
                    target_route="nlp_openie_candidate",
                    proxy_label="openie_tuple_present",
                    metadata={"relation": relation, "arg1": arg1, "arg2": arg2},
                )
                if count >= limit_per_split:
                    break


def iter_dialogre(root: Path, limit_per_split: int) -> Iterable[ProxyExample]:
    for lang in ("en", "cn"):
        for split in ("train", "dev", "test"):
            path = root / "DialogRE" / "data_v2" / lang / "data" / f"{split}.json"
            if not path.exists():
                continue
            data = read_json(path)
            for index, row in enumerate(data[:limit_per_split], 1):
                turns = row[0] if row and isinstance(row[0], list) else []
                relations = row[1] if len(row) > 1 and isinstance(row[1], list) else []
                text = " ".join(str(turn) for turn in turns)
                route = "strong_llm_graph_extraction" if len(turns) >= 4 or len(relations) > 1 else "weak_llm_graph_extraction"
                yield ProxyExample(
                    dataset_id=f"dialogre_{lang}",
                    split=split,
                    source_ref=f"DialogRE/data_v2/{lang}/data/{split}.json:{index}",
                    text=text,
                    target_route=route,
                    proxy_label="dialogue_relation_context",
                    metadata={"turn_count": len(turns), "relation_count": len(relations), "language": lang},
                )


def iter_duie(root: Path, limit_per_split: int) -> Iterable[ProxyExample]:
    files = [
        ("dev", root / "DuIE-mirror-Bert-In-Relation-Extraction" / "duie_dev.json"),
        ("train", root / "DuIE-mirror-Bert-In-Relation-Extraction" / "train.json"),
    ]
    for split, path in files:
        if not path.exists():
            continue
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                spo_list = row.get("spo_list") or []
                text = str(row.get("text") or "")
                count += 1
                route = "strong_llm_graph_extraction" if len(spo_list) >= 5 else "weak_llm_graph_extraction"
                yield ProxyExample(
                    dataset_id="duie_chinese_spo",
                    split=split,
                    source_ref=f"DuIE-mirror-Bert-In-Relation-Extraction/{path.name}:{count}",
                    text=text,
                    target_route=route,
                    proxy_label="chinese_spo_present",
                    metadata={"spo_count": len(spo_list), "language": "zh"},
                )
                if count >= limit_per_split:
                    break


def retacred_label_counts(root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    retacred_root = root / "Re-TACRED" / "Re-TACRED"
    for split in ("train", "dev", "test"):
        path = retacred_root / f"{split}_id2label.json"
        if not path.exists():
            continue
        labels = read_json(path)
        counts = Counter(labels.values())
        out[split] = {
            "row_count": len(labels),
            "no_relation_count": counts.get("no_relation", 0),
            "positive_relation_count": len(labels) - counts.get("no_relation", 0),
            "top_labels": dict(counts.most_common(10)),
            "source_hash": file_hash(path),
            "status": "label_only_no_text",
        }
    return out


def make_packet(example: ProxyExample) -> dict[str, Any]:
    return {
        "packet_id": stable_id("graph_proxy_packet", f"{example.dataset_id}|{example.source_ref}"),
        "workspace_id": "external_graph_route_calibration",
        "modeled_user_id": "proxy_dataset",
        "input_kind": "external_proxy_example",
        "input_ref": example.source_ref,
        "original_text": example.text,
        "processed_text": "",
        "evidence_refs": [f"external:{example.source_ref}"],
        "source_refs": [example.dataset_id],
        "raw_backpointer_refs": [example.source_ref],
        "source_perspective": "proxy_dataset",
        "subject_role": "unknown",
        "attribution_status": "proxy_label",
        "temporal_scope": {},
        "confidence": "proxy",
        "inference_level": "proxy_label",
        "privacy_class": "public_dataset",
        "route_refs": [],
        "proposal_refs": [],
        "review_refs": [],
        "warnings": ["external_proxy_label_not_route_truth"],
        "graph_is_not_proof": True,
    }


def build_examples(root: Path, limit_per_dataset: int) -> list[ProxyExample]:
    per_split = max(1, limit_per_dataset // 3)
    examples = []
    examples.extend(iter_carb(root, per_split))
    examples.extend(iter_dialogre(root, per_split))
    examples.extend(iter_duie(root, per_split))
    return examples


def feature_rows(examples: list[ProxyExample]) -> list[dict[str, Any]]:
    rows = []
    for example in examples:
        packet = make_packet(example)
        features = graph_route_features(packet)
        features.update(
            {
                "dataset_id": example.dataset_id,
                "split": example.split,
                "source_ref": example.source_ref,
                "target_route": example.target_route,
                "proxy_label": example.proxy_label,
                "proxy_metadata": example.metadata,
            }
        )
        rows.append(features)
    return rows


def route_penalty(target: str, predicted: str) -> float:
    if target == predicted:
        return 0.0
    if predicted == "skip_or_background_only" and target in {
        "nlp_openie_candidate",
        "weak_llm_graph_extraction",
        "strong_llm_graph_extraction",
    }:
        return 8.0
    if predicted == "entity_candidate_only" and target in {"weak_llm_graph_extraction", "strong_llm_graph_extraction"}:
        return 7.0
    if predicted == "entity_candidate_only" and target == "nlp_openie_candidate":
        return 5.0
    if target == "strong_llm_graph_extraction" and predicted in {"weak_llm_graph_extraction", "nlp_openie_candidate"}:
        return 1.5
    if target in {"weak_llm_graph_extraction", "nlp_openie_candidate"} and predicted == "strong_llm_graph_extraction":
        return 0.75
    if target == "nlp_openie_candidate" and predicted == "weak_llm_graph_extraction":
        return 0.5
    return 2.0


def evaluate(rows: list[dict[str, Any]], policy: dict[str, float]) -> dict[str, Any]:
    exact = 0
    penalty = 0.0
    relation_targets = 0
    relation_predicted = 0
    confusion: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    for row in rows:
        predicted, _ = graph_route_for_features(row, policy)
        target = str(row["target_route"])
        exact += int(predicted == target)
        penalty += route_penalty(target, predicted)
        if target in {"nlp_openie_candidate", "weak_llm_graph_extraction", "strong_llm_graph_extraction"}:
            relation_targets += 1
            relation_predicted += int(predicted in {"nlp_openie_candidate", "weak_llm_graph_extraction", "strong_llm_graph_extraction"})
        confusion[f"{target}->{predicted}"] += 1
        predicted_counts[predicted] += 1
        target_counts[target] += 1
    total = len(rows) or 1
    relation_recall = relation_predicted / relation_targets if relation_targets else 1.0
    return {
        "exact_match": round(exact / total, 6),
        "mean_penalty": round(penalty / total, 6),
        "relation_route_recall": round(relation_recall, 6),
        "objective": round((penalty / total) + (1.0 - exact / total) + (1.0 - relation_recall), 6),
        "confusion": dict(confusion),
        "predicted_counts": dict(predicted_counts),
        "target_counts": dict(target_counts),
    }


def candidate_policies() -> Iterable[dict[str, float]]:
    base = dict(DEFAULT_POLICY)
    for nlp_surface in (4.0, 5.0, 6.0, 7.0):
        for nlp_context in (4.0, 5.0, 6.0, 7.0):
            for strong_utility in (3.5, 4.0, 4.5, 5.0, 5.5):
                for strong_context in (3.5, 4.0, 4.5, 5.0, 6.0):
                    for weak_relation in (2.0, 2.5, 3.0, 3.5):
                        for weak_utility in (3.0, 3.5, 4.0, 4.5):
                            for entity_ceiling in (1.0, 1.5, 2.0, 2.5):
                                policy = dict(base)
                                policy.update(
                                    {
                                        "entity_relation_ceiling": entity_ceiling,
                                        "nlp_surface_floor": nlp_surface,
                                        "nlp_context_ceiling": nlp_context,
                                        "strong_utility_floor": strong_utility,
                                        "strong_context_floor": strong_context,
                                        "weak_relation_floor": weak_relation,
                                        "weak_utility_floor": weak_utility,
                                    }
                                )
                                yield policy


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset_id",
        "split",
        "source_ref",
        "target_route",
        "proxy_label",
        "relation_surface_clarity",
        "relation_likelihood",
        "endpoint_quality",
        "directionality_certainty",
        "context_requirement",
        "attribution_risk",
        "evidence_quality",
        "graph_utility_hint",
        "merge_ambiguity",
        "low_graph_value",
        "text_excerpt",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# v0.3 Graph Route External Calibration Report",
        "",
        f"- training_example_count: {manifest['counts']['training_example_count']}",
        f"- candidate_count: {manifest['counts']['candidate_count']}",
        f"- best_objective: {manifest['best_evaluation']['objective']}",
        f"- exact_match: {manifest['best_evaluation']['exact_match']}",
        f"- mean_penalty: {manifest['best_evaluation']['mean_penalty']}",
        f"- relation_route_recall: {manifest['best_evaluation']['relation_route_recall']}",
        "",
        "## Dataset Counts",
        "",
    ]
    for dataset, count in sorted(manifest["counts"]["dataset_counts"].items()):
        lines.append(f"- `{dataset}`: {count}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Proxy labels are calibration signals, not route truth.",
            "- Re-TACRED is label-only in this local checkout; it informs no_relation prevalence but not feature fitting.",
            "- DocRED/HacRED are gated here because this checkout does not include usable text data.",
            "- No deep-learning training, provider calls, durable memory writes, graph truth writes, or S3 writes were executed.",
        ]
    )
    return "\n".join(lines) + "\n"


def calibrate_graph_route_policy(
    project_root: Path,
    *,
    dataset_root: Path | None = None,
    output_dir: Path | None = None,
    policy_output: Path | None = None,
    limit_per_dataset: int = 120,
) -> dict[str, Any]:
    dataset_root = (dataset_root or project_root / DEFAULT_DATASET_ROOT).resolve()
    output_dir = (output_dir or project_root / DEFAULT_OUTPUT_DIR).resolve()
    policy_output = (policy_output or project_root / DEFAULT_POLICY_OUTPUT).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_output.parent.mkdir(parents=True, exist_ok=True)

    examples = build_examples(dataset_root, limit_per_dataset)
    rows = feature_rows(examples)
    if not rows:
        raise RuntimeError(f"No calibration rows found under {dataset_root}")

    best_policy = dict(DEFAULT_POLICY)
    best_eval = evaluate(rows, best_policy)
    candidate_count = 1
    for policy in candidate_policies():
        candidate_count += 1
        result = evaluate(rows, policy)
        if result["objective"] < best_eval["objective"]:
            best_policy = policy
            best_eval = result

    dataset_counts = Counter(row["dataset_id"] for row in rows)
    target_counts = Counter(row["target_route"] for row in rows)
    source_hashes = {
        "CaRB/data/gold/dev.tsv": file_hash(dataset_root / "CaRB" / "data" / "gold" / "dev.tsv"),
        "CaRB/data/gold/test.tsv": file_hash(dataset_root / "CaRB" / "data" / "gold" / "test.tsv"),
        "DialogRE/data_v2/en/data/train.json": file_hash(dataset_root / "DialogRE" / "data_v2" / "en" / "data" / "train.json"),
        "DialogRE/data_v2/cn/data/train.json": file_hash(dataset_root / "DialogRE" / "data_v2" / "cn" / "data" / "train.json"),
        "DuIE-mirror-Bert-In-Relation-Extraction/duie_dev.json": file_hash(
            dataset_root / "DuIE-mirror-Bert-In-Relation-Extraction" / "duie_dev.json"
        ),
    }
    retacred_counts = retacred_label_counts(dataset_root)

    policy_doc = {
        "schema_version": "graph_v03.route_policy.v0.3.candidate",
        "policy_id": "graph-route-policy:v0.3:external-proxy-calibrated-candidate",
        "target_task": "graph_relation_candidate",
        "parameters": best_policy,
        "calibration": {
            "schema_version": SCHEMA_VERSION,
            "method": "external_proxy_grid_search_non_dl",
            "dataset_counts": dict(dataset_counts),
            "target_counts": dict(target_counts),
            "candidate_count": candidate_count,
            "best_evaluation": best_eval,
            "source_hashes": source_hashes,
            "retacred_label_only_counts": retacred_counts,
            "proxy_labels_are_not_truth": True,
            "deep_learning_training_executed": False,
        },
        "guardrails": {
            "llm_calls_executed": False,
            "durable_writes_executed": False,
            "graph_truth_written": False,
            "s3_written": False,
        },
    }
    write_json(policy_output, policy_doc)
    write_jsonl(output_dir / "graph_route_calibration_rows.jsonl", rows)
    write_csv(output_dir / "graph_route_calibration_rows.csv", rows)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "policy_output": str(policy_output),
        "counts": {
            "training_example_count": len(rows),
            "candidate_count": candidate_count,
            "dataset_counts": dict(dataset_counts),
            "target_counts": dict(target_counts),
        },
        "best_policy": best_policy,
        "best_evaluation": best_eval,
        "source_hashes": source_hashes,
        "retacred_label_only_counts": retacred_counts,
        "gated_or_reference_only": {
            "TACRED": "original TACRED text not included; Re-TACRED checkout contains relabel maps only",
            "DocRED": "official repo checkout contains code/download instructions, not usable text data",
            "HacRED": "not downloaded in this slice; locate official accessible data before use",
        },
        "outputs": {
            "policy": str(policy_output),
            "rows_jsonl": str(output_dir / "graph_route_calibration_rows.jsonl"),
            "rows_csv": str(output_dir / "graph_route_calibration_rows.csv"),
            "manifest": str(output_dir / "graph_route_calibration_manifest.json"),
            "report": str(output_dir / "graph_route_calibration_report.md"),
        },
    }
    write_json(output_dir / "graph_route_calibration_manifest.json", manifest)
    write_text(output_dir / "graph_route_calibration_report.md", render_report(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calibrate v0.3 graph package router from external proxy RE/OpenIE data.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--policy-output", default=None)
    parser.add_argument("--limit-per-dataset", type=int, default=120)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    project_root = Path(args.project_root).resolve()
    calibrate_graph_route_policy(
        project_root,
        dataset_root=Path(args.dataset_root).resolve() if args.dataset_root else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        policy_output=Path(args.policy_output).resolve() if args.policy_output else None,
        limit_per_dataset=args.limit_per_dataset,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
