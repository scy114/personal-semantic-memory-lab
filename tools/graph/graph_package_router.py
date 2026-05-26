"""Route v0.3 graph construction packets before extraction.

This is the graph-specific pre-extraction router. It scores graph construction
packets for relation-building value, not S1 memory value or S2 portrait value.
The output is a route plan consumed by graph extraction; it does not extract
relations and does not write graph truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.graph.graph_construction_packet_builder import (
    read_jsonl,
    sha256_text,
    stable_id,
    unique_strings,
    write_json,
    write_jsonl,
    write_text,
)


SCHEMA_VERSION = "graph_v03.package_router.v0.1"
FEATURE_SCHEMA_VERSION = "graph_v03.route_feature_matrix.v0.1"
ROUTE_SCHEMA_VERSION = "graph_v03.route_decision.v0.1"
TARGET_TASK = "graph_relation_candidate"
DEFAULT_INPUT_DIR_NAME = "graph_v03_construction"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_construction"

ROUTES = {
    "skip_or_background_only": {"cost_class": "none", "fallback_route": "entity_candidate_only"},
    "entity_candidate_only": {"cost_class": "low", "fallback_route": "weak_llm_graph_extraction"},
    "nlp_openie_candidate": {"cost_class": "low", "fallback_route": "weak_llm_graph_extraction"},
    "weak_llm_graph_extraction": {"cost_class": "medium", "fallback_route": "strong_llm_graph_extraction"},
    "strong_llm_graph_extraction": {"cost_class": "high", "fallback_route": "human_review"},
    "repair_or_review": {"cost_class": "medium", "fallback_route": "human_review"},
    "human_review": {"cost_class": "high", "fallback_route": "human_review"},
}

DEFAULT_POLICY = {
    "evidence_min": 5.0,
    "low_skip_floor": 7.0,
    "low_skip_relation_ceiling": 3.0,
    "entity_endpoint_floor": 3.0,
    "entity_relation_ceiling": 2.0,
    "nlp_surface_floor": 6.0,
    "nlp_endpoint_floor": 4.0,
    "nlp_direction_floor": 5.0,
    "nlp_attribution_ceiling": 4.0,
    "nlp_context_ceiling": 5.0,
    "strong_utility_floor": 5.0,
    "strong_context_floor": 6.0,
    "strong_attribution_floor": 5.0,
    "strong_merge_floor": 6.0,
    "weak_relation_floor": 3.5,
    "weak_utility_floor": 4.0,
}

LOW_VALUE_PATTERNS = [
    r"^\s*(thanks|thank you|ok|okay|yes|no|lol|haha|hmm|uh+)[!.?]*\s*$",
    r"^\s*(\d{4}|\w{3}\.\s+\d{1,2}(?:,\s+\d{4})?)[!.?]*\s*$",
]
ATTRIBUTION_MARKERS = r"\b(said|told|asked|wrote|heard|believe|believes|thought|claimed|according to)\b"
CONTEXT_MARKERS = r"\b(because|although|though|however|therefore|but|while|when|after|before|if|unless|despite)\b"
UPDATE_MARKERS = r"\b(change|changed|update|updated|replace|replaced|contradict|contradicts|conflict|conflicts|instead|no longer)\b"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str | None:
    return sha256_text(path.read_text(encoding="utf-8", errors="ignore")) if path.exists() else None


def normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def packet_text(packet: dict[str, Any]) -> str:
    return normalize_whitespace(
        packet.get("graph_route_text")
        or packet.get("graph_extraction_text")
        or packet.get("original_text")
        or packet.get("processed_text")
        or ""
    )


def primary_packet_text(packet: dict[str, Any]) -> str:
    return normalize_whitespace(packet.get("original_text") or packet.get("processed_text") or "")


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def clamp_score(value: float) -> float:
    return round(max(0.0, min(10.0, value)), 3)


def count_clauses(text: str) -> int:
    if not text:
        return 0
    return max(1, len(re.split(r"(?<=[.!?;:])\s+|\s+(?:and|but|then|so)\s+", text)))


def named_entity_like_terms(text: str) -> list[str]:
    english_names = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
    chinese_terms = re.findall(r"[\u4e00-\u9fff]{2,8}", text)
    return unique_strings(english_names, chinese_terms)


def has_low_value_text(text: str) -> bool:
    if not text:
        return True
    if len(text) < 10 and not re.search(r"[A-Za-z\u4e00-\u9fff]", text):
        return True
    return any(re.fullmatch(pattern, text, flags=re.IGNORECASE) for pattern in LOW_VALUE_PATTERNS)


def surface_relation_hits(text: str) -> int:
    """Cheap structural surface-relation signal.

    This is deliberately a small fallback signal, not the main extraction
    method. Mature RE/OpenIE datasets are used for calibration design; this
    runtime score only decides whether a packet is cheap enough for a baseline
    candidate before LLM extraction.
    """

    if not text:
        return 0
    hits = 0
    hits += len(
        re.findall(
            r"\b(?:uses?|using|used|depends?\s+on|works?\s+on|visited|visits?|met|supports?|criticizes?|replaced|replaces?|updated|updates?)\b\s+(?:a|an|the|this|that|my|your|his|her|our|their)?\s*[A-Za-z][^.;!?]{2,80}",
            text,
            flags=re.IGNORECASE,
        )
    )
    proper_count = len(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text))
    if proper_count >= 2 and re.search(r"\b(?:is|are|was|were|made|found|led|joined|plays|taught|began|became|born|died|worked|located|based)\b", text, flags=re.IGNORECASE):
        hits += 1
    hits += len(re.findall(r"\b(?:to|with|from|on|in|at|for|by|about)\s+[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*", text))
    hits += len(re.findall(r"[\u4e00-\u9fff]{2,}(?:使用|依赖|反对|支持|更新|替代|属于|位于|参加|负责)[\u4e00-\u9fffA-Za-z0-9_-]{1,}", text))
    return hits


def evidence_quality(packet: dict[str, Any]) -> float:
    score = 0.0
    if as_list(packet.get("evidence_refs")):
        score += 4.0
    if as_list(packet.get("source_refs")):
        score += 2.0
    if as_list(packet.get("raw_backpointer_refs")):
        score += 3.0
    if packet_text(packet):
        score += 1.0
    return clamp_score(score)


def graph_route_features(packet: dict[str, Any]) -> dict[str, Any]:
    text = packet_text(packet)
    primary_text = primary_packet_text(packet) or text
    context_ref_count = len(as_list(packet.get("context_evidence_refs")))
    named_terms = named_entity_like_terms(text)
    clause_count = count_clauses(primary_text)
    surface_hits = surface_relation_hits(text)
    attribution_hits = len(re.findall(ATTRIBUTION_MARKERS, primary_text, flags=re.IGNORECASE))
    context_hits = len(re.findall(CONTEXT_MARKERS, primary_text, flags=re.IGNORECASE))
    update_hits = len(re.findall(UPDATE_MARKERS, text, flags=re.IGNORECASE))
    speaker_turn_count = len(re.findall(r"\bSpeaker\s+\d+\s*:", text))
    speaker_count = len(set(re.findall(r"\bSpeaker\s+\d+\s*:", text)))
    warnings = unique_strings(packet.get("warnings"))
    evidence = evidence_quality(packet)
    low_value = 8.0 if has_low_value_text(text) else 0.0
    if len(text) < 25 and len(named_terms) < 2 and surface_hits == 0:
        low_value += 2.0

    endpoint = 0.0
    if packet.get("source_perspective"):
        endpoint += 2.0
    endpoint += min(5.0, len(named_terms) * 1.5)
    endpoint += min(3.0, speaker_count * 0.75)
    if surface_hits:
        endpoint += 1.0
    if re.search(r"\b(he|she|they|it|this|that|someone|something)\b", text, flags=re.IGNORECASE):
        endpoint -= 1.5

    relation_surface = min(10.0, surface_hits * 6.0)
    relation_likelihood = relation_surface * 0.45 + endpoint * 0.25 + (0.3 * min(len(text) / 40.0, 4.0))
    if speaker_turn_count >= 4:
        relation_likelihood += min(3.0, speaker_turn_count * 0.2)
    if context_ref_count:
        relation_likelihood += min(1.5, context_ref_count * 0.3)
    directionality = relation_surface
    if "?" in primary_text:
        directionality -= 1.0
    if attribution_hits:
        directionality -= min(3.0, attribution_hits)
    if update_hits:
        relation_likelihood += 1.0

    context_requirement = 0.0
    if clause_count > 1:
        context_requirement += min(5.0, (clause_count - 1) * 1.5)
    context_requirement += min(3.0, context_hits * 1.5)
    if context_ref_count:
        context_requirement += min(2.0, context_ref_count * 0.5)
    if packet.get("input_kind") in {"reviewed_portrait_unit", "proposal_outcome"}:
        context_requirement += 0.5
    if "neighbor" in " ".join(str(item) for item in warnings).lower():
        context_requirement += 2.0

    attribution_risk = 0.0
    attribution_status = str(packet.get("attribution_status") or "").lower()
    if attribution_status not in {"strict", "source_text_only", "source", ""}:
        attribution_risk += 3.0
    attribution_risk += min(4.0, attribution_hits * 2.0)
    if str(packet.get("subject_role") or "").lower() not in {"target", "target_subject", ""}:
        attribution_risk += 1.5

    graph_utility = relation_likelihood * 0.4 + endpoint * 0.3
    if speaker_turn_count >= 4:
        graph_utility += min(2.0, speaker_turn_count * 0.15)
    if context_ref_count:
        graph_utility += min(1.0, context_ref_count * 0.25)
    if packet.get("input_kind") in {"reviewed_portrait_unit", "normalized_candidate", "proposal_outcome"}:
        graph_utility += 1.5
    if update_hits:
        graph_utility += 2.0

    merge_ambiguity = 0.0
    if len(named_terms) == 1:
        merge_ambiguity += 2.0
    if re.search(r"\b(Mr|Mrs|Ms|Dr|Sir)\.?\s+[A-Z][a-z]+", text):
        merge_ambiguity += 1.0
    if endpoint < 3.0 and named_terms:
        merge_ambiguity += 2.0

    return {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "source_packet_id": str(packet.get("packet_id") or ""),
        "input_kind": str(packet.get("input_kind") or ""),
        "text_excerpt": text[:240],
        "relation_surface_clarity": clamp_score(relation_surface),
        "relation_likelihood": clamp_score(relation_likelihood),
        "endpoint_quality": clamp_score(endpoint),
        "directionality_certainty": clamp_score(directionality),
        "context_requirement": clamp_score(context_requirement),
        "attribution_risk": clamp_score(attribution_risk),
        "evidence_quality": evidence,
        "graph_utility_hint": clamp_score(graph_utility),
        "merge_ambiguity": clamp_score(merge_ambiguity),
        "low_graph_value": clamp_score(low_value),
        "surface_relation_hit_count": surface_hits,
        "named_entity_like_count": len(named_terms),
        "clause_count": clause_count,
        "attribution_marker_count": attribution_hits,
        "context_marker_count": context_hits,
        "update_marker_count": update_hits,
        "speaker_turn_count": speaker_turn_count,
        "speaker_count": speaker_count,
        "context_evidence_ref_count": context_ref_count,
        "feature_provenance": {
            "external_research_basis": [
                "TACRED/Re-TACRED no_relation proxy",
                "DocRED/Re-DocRED context proxy",
                "DialogRE dialogue attribution proxy",
                "DuIE/HacRED Chinese RE proxy",
                "OpenIE/CaRB tuple-density proxy",
            ],
            "runtime_external_model_used": False,
            "fallback_surface_signal": True,
        },
        "warnings": unique_strings(warnings),
        "graph_is_not_proof": True,
    }


def graph_route_for_features(features: dict[str, Any], policy: dict[str, float] | None = None) -> tuple[str, list[str]]:
    policy = dict(DEFAULT_POLICY if policy is None else policy)
    reasons: list[str] = []
    relation_surface = float(features["relation_surface_clarity"])
    relation_likelihood = float(features["relation_likelihood"])
    endpoint = float(features["endpoint_quality"])
    directionality = float(features["directionality_certainty"])
    context = float(features["context_requirement"])
    attribution = float(features["attribution_risk"])
    evidence = float(features["evidence_quality"])
    graph_utility = float(features["graph_utility_hint"])
    low_value = float(features["low_graph_value"])
    merge_ambiguity = float(features["merge_ambiguity"])

    if evidence < float(policy["evidence_min"]):
        return "repair_or_review", ["graph_route_missing_or_weak_evidence"]
    if low_value >= float(policy["low_skip_floor"]) and relation_likelihood < float(policy["low_skip_relation_ceiling"]):
        return "skip_or_background_only", ["graph_route_low_graph_value"]
    if (
        relation_surface >= float(policy["nlp_surface_floor"])
        and endpoint >= float(policy["nlp_endpoint_floor"])
        and directionality >= float(policy["nlp_direction_floor"])
        and attribution < float(policy["nlp_attribution_ceiling"])
        and context < float(policy["nlp_context_ceiling"])
    ):
        return "nlp_openie_candidate", ["graph_route_surface_relation_clear"]
    if graph_utility >= float(policy["strong_utility_floor"]) and (
        context >= float(policy["strong_context_floor"])
        or attribution >= float(policy["strong_attribution_floor"])
        or merge_ambiguity >= float(policy["strong_merge_floor"])
    ):
        return "strong_llm_graph_extraction", ["graph_route_high_utility_complex_or_attributed"]
    if relation_likelihood >= float(policy["weak_relation_floor"]) or graph_utility >= float(policy["weak_utility_floor"]):
        reasons.append("graph_route_relation_or_utility_signal")
        if relation_surface < 6.0:
            reasons.append("graph_route_weak_surface_relation_needs_llm")
        if context >= 4.0:
            reasons.append("graph_route_context_dependency")
        if attribution >= 4.0:
            reasons.append("graph_route_attribution_risk")
        return "weak_llm_graph_extraction", reasons
    if endpoint >= float(policy["entity_endpoint_floor"]) and relation_likelihood < float(policy["entity_relation_ceiling"]):
        return "entity_candidate_only", ["graph_route_entity_only_relation_unclear"]
    return "skip_or_background_only", ["graph_route_no_relation_signal"]


def route_row(packet: dict[str, Any], features: dict[str, Any], route_run_id: str, policy: dict[str, float] | None = None) -> dict[str, Any]:
    route, reasons = graph_route_for_features(features, policy)
    route_meta = ROUTES[route]
    route_score = 0 if route == "skip_or_background_only" else int(
        round(max(float(features["relation_likelihood"]), float(features["graph_utility_hint"])))
    )
    return {
        "schema_version": ROUTE_SCHEMA_VERSION,
        "route_decision_id": stable_id("graph_route", f"{route_run_id}|{packet.get('packet_id')}|{route}"),
        "route_run_id": route_run_id,
        "target_task": TARGET_TASK,
        "source_packet_id": str(packet.get("packet_id") or ""),
        "recommended_route": route,
        "fallback_route": route_meta["fallback_route"],
        "route_score": route_score,
        "route_score_max": 10,
        "route_confidence": None,
        "route_confidence_type": "graph_package_matrix_v0.1",
        "cost_class": route_meta["cost_class"],
        "routing_reasons": reasons,
        "graph_route_scores": {
            key: features[key]
            for key in [
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
            ]
        },
        "write_permission": False,
        "warnings": unique_strings(features.get("warnings")),
        "graph_is_not_proof": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_packet_id",
        "input_kind",
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
        "surface_relation_hit_count",
        "named_entity_like_count",
        "clause_count",
        "attribution_marker_count",
        "context_marker_count",
        "update_marker_count",
        "text_excerpt",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def load_policy(path: Path | None) -> dict[str, float]:
    policy = dict(DEFAULT_POLICY)
    if path is None:
        return policy
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("parameters") if isinstance(data.get("parameters"), dict) else data
    for key in DEFAULT_POLICY:
        if key in values:
            policy[key] = float(values[key])
    return policy


def build_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    lines = [
        "# v0.3 Graph Package Route Report",
        "",
        f"- workspace_id: `{manifest['workspace_id']}`",
        f"- packet_count: {counts['packet_count']}",
        f"- target_task: `{TARGET_TASK}`",
        f"- graph_is_not_proof: `true`",
        "",
        "## Route Counts",
        "",
    ]
    for route, count in counts["route_counts"].items():
        lines.append(f"- `{route}`: {count}")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- This router selects graph extraction lanes; it does not extract relations.",
            "- NLP/OpenIE baseline routes are candidate routes, not graph truth.",
            "- LLM routes still require schema-guided extraction and evidence validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_graph_package_routes(
    workspace: Path,
    *,
    input_packets_path: Path | None = None,
    output_dir: Path | None = None,
    route_run_id: str = "graph-route-v03",
    policy_path: Path | None = None,
    max_items: int | None = None,
    item_offset: int = 0,
    sample_stride: int = 1,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    output_dir = (output_dir or workspace / DEFAULT_OUTPUT_DIR_NAME).resolve()
    input_packets_path = (input_packets_path or output_dir / "graph_construction_packets.jsonl").resolve()
    packets = read_jsonl(input_packets_path)
    if item_offset:
        packets = packets[item_offset:]
    if sample_stride > 1:
        packets = packets[::sample_stride]
    if max_items is not None:
        packets = packets[: max(0, max_items)]
    if not packets:
        raise FileNotFoundError(f"No graph construction packets found at: {input_packets_path}")

    policy = load_policy(policy_path)
    feature_rows = [graph_route_features(packet) for packet in packets]
    route_rows = [route_row(packet, features, route_run_id, policy) for packet, features in zip(packets, feature_rows)]
    route_counts = Counter(row["recommended_route"] for row in route_rows)

    feature_jsonl = output_dir / "graph_route_feature_matrix.jsonl"
    feature_csv = output_dir / "graph_route_feature_matrix.csv"
    decisions_path = output_dir / "graph_route_decisions.jsonl"
    manifest_path = output_dir / "graph_route_manifest.json"
    report_path = output_dir / "graph_route_report.md"

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "workspace_id": workspace.name,
        "route_run_id": route_run_id,
        "target_task": TARGET_TASK,
        "input_packets": str(input_packets_path),
        "output_dir": str(output_dir),
        "counts": {
            "packet_count": len(packets),
            "route_counts": dict(sorted(route_counts.items())),
            "source_asset_hashes": {
                "graph_v03_construction/graph_construction_packets.jsonl": file_hash(input_packets_path),
            },
        },
        "policies": {
            "graph_specific_route": True,
            "s1_s2_memory_value_route_reused": False,
            "llm_calls_executed": False,
            "extraction_executed": False,
            "graph_is_not_proof": True,
            "deep_learning_training_executed": False,
            "policy_path": str(policy_path) if policy_path else None,
            "policy_parameters": policy,
        },
        "outputs": {
            "graph_route_feature_matrix_jsonl": str(feature_jsonl),
            "graph_route_feature_matrix_csv": str(feature_csv),
            "graph_route_decisions": str(decisions_path),
            "graph_route_manifest": str(manifest_path),
            "graph_route_report": str(report_path),
        },
    }
    write_jsonl(feature_jsonl, feature_rows)
    write_csv(feature_csv, feature_rows)
    write_jsonl(decisions_path, route_rows)
    write_json(manifest_path, manifest)
    write_text(report_path, build_report(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build v0.3 graph package feature matrix and route decisions.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--input-packets", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--route-run-id", default="graph-route-v03")
    parser.add_argument("--policy", default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--item-offset", type=int, default=0)
    parser.add_argument("--sample-stride", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    build_graph_package_routes(
        Path(args.workspace),
        input_packets_path=Path(args.input_packets).resolve() if args.input_packets else None,
        output_dir=Path(args.output_dir).resolve() if args.output_dir else None,
        route_run_id=args.route_run_id,
        policy_path=Path(args.policy).resolve() if args.policy else None,
        max_items=args.max_items,
        item_offset=args.item_offset,
        sample_stride=args.sample_stride,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
