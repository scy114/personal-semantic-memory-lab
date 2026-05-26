"""Step 2 build runner for evidence-bound user-model assets.

This is a v0.1 integration runner for the Step 2 Build workflow. It consumes
Step 1 evidence-bound assets and writes Step 2 canonical user-model artifacts.
It intentionally does not build Step 1 assets, Step 1 indexes, Step 2 embedding
indexes, query packets, final answers, or Step 3 latent models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.prebuild_routing import (
    DEFAULT_S2_PROPOSAL_PROFILE,
    add_prebuild_arguments,
    options_from_args,
    run_prebuild_routing,
)


SUPPORTED_DUPLICATE_POLICIES = {"fail", "overwrite_generated"}
ACTIVE_REVIEW_STATUSES = {"accepted_for_experiment", "active"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def short_hash(value: str, length: int = 12) -> str:
    return sha256_text(value)[:length]


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "item"


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}-{short_hash(value)}"


def existing_output_paths(workspace: Path) -> list[Path]:
    return [
        workspace / "manifest.yaml",
        workspace / "portrait" / "normalized_candidates.jsonl",
        workspace / "portrait" / "review_decisions.jsonl",
        workspace / "portrait" / "reviewed_units.jsonl",
        workspace / "portrait" / "current_portrait.md",
        workspace / "portrait" / "current_portrait.json",
        workspace / "graph" / "nodes.jsonl",
        workspace / "graph" / "edges.jsonl",
        workspace / "graph" / "graph-summary.md",
        workspace / "packets" / "base_assistance_packet.json",
        workspace / "reports" / "target-selection.md",
        workspace / "reports" / "protocol-review-report.md",
        workspace / "reports" / "phase2-pilot-report.md",
        workspace / "checkpoints" / "phase2-status.json",
        workspace / "checkpoints" / "run-log.md",
    ]


def prepare_outputs(workspace: Path, duplicate_policy: str) -> None:
    existing = [path for path in existing_output_paths(workspace) if path.exists()]
    if existing and duplicate_policy == "fail":
        raise FileExistsError(
            "Existing Step 2 build assets found and duplicate_policy=fail: "
            + ", ".join(str(path) for path in existing)
        )
    if existing and duplicate_policy == "overwrite_generated":
        checkpoint = workspace / "checkpoints" / ("before-step2-build-overwrite-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
        for path in existing:
            if not path.exists() or path.is_dir():
                continue
            rel = path.relative_to(workspace)
            target = checkpoint / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())


def choose_memory_units(memory_units: list[dict[str, Any]], target: str, max_units: int) -> list[dict[str, Any]]:
    target_lower = target.lower()
    selected: list[dict[str, Any]] = []
    for unit in memory_units:
        status = unit.get("status", "")
        if status not in ACTIVE_REVIEW_STATUSES:
            continue
        if unit.get("s2_policy") in {"background_only", "blocked_from_portrait", "needs_review"}:
            continue
        source_span = unit.get("source_span") or {}
        speaker = str(source_span.get("speaker") or unit.get("subject_id") or "").lower()
        content = str(unit.get("content") or "")
        if speaker and speaker != target_lower:
            continue
        if not speaker and f"{target} said" not in content and f"{target} wrote" not in content:
            continue
        if not unit.get("evidence_refs"):
            continue
        selected.append(unit)
    return selected[:max_units]


def classify_candidate(memory_unit: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    text = str(memory_unit.get("content") or "")
    lowered = text.lower()
    if any(word in lowered for word in ["prefer", "favorite", "favourite", "love", "enjoy", "stress relief", "go-to"]):
        return "preference", "semantic", "global", ["preference"]
    if any(word in lowered for word in ["job", "business", "store", "work", "career", "lost"]):
        return "life_stage", "episodic", "temporary", ["life_stage"]
    if any(word in lowered for word in ["plan", "routine", "usually", "schedule", "practice"]):
        return "routine", "procedural", "task", ["routine"]
    return "observation", memory_unit.get("memory_class", "episodic"), "conversation", ["observation"]


def evidence_excerpt(memory_unit: dict[str, Any]) -> str:
    quote = memory_unit.get("evidence_quote") or memory_unit.get("evidence_summary") or memory_unit.get("content") or ""
    display_refs = memory_unit.get("display_refs") or memory_unit.get("source_specific_refs") or []
    if display_refs:
        return f"{', '.join(display_refs)}: {quote}"
    return str(quote)


def candidate_from_memory_unit(index: int, target: str, unit: dict[str, Any]) -> dict[str, Any]:
    candidate_type, memory_class, scope, object_types = classify_candidate(unit)
    candidate_id = f"npc-{slugify(target)}-{index:03d}"
    node_id = f"{object_types[0]}:{short_hash(unit.get('memory_id') or unit.get('content', ''), 10)}"
    relation_type = {
        "preference": "has_preference",
        "life_stage": "has_life_stage",
        "routine": "has_routine",
        "observation": "has_observation",
    }.get(candidate_type, "has_observation")
    return {
        "candidate_id": candidate_id,
        "user_id": target,
        "input_layer": "memory_unit",
        "input_refs": [unit.get("memory_id", "")],
        "source_refs": list(unit.get("source_refs") or []),
        "evidence_refs": list(unit.get("evidence_refs") or []),
        "backpointer_refs": list(unit.get("backpointer_refs") or unit.get("evidence_refs") or []),
        "evidence_excerpt": evidence_excerpt(unit),
        "candidate_text": unit.get("content", ""),
        "candidate_type": candidate_type,
        "memory_class": memory_class,
        "scope": scope,
        "inference_level": unit.get("inference_level", "explicit"),
        "confidence": unit.get("confidence", "high"),
        "temporal_scope": {
            "validity": "current_or_historical_from_public_conversation",
            "observed_at_refs": [{"display_ref": ref} for ref in unit.get("display_refs", [])],
        },
        "privacy_class": unit.get("privacy_class", "public_dataset"),
        "subject_contamination_risk": "none",
        "step1_generation_method": unit.get("generation_method", ""),
        "step1_status": unit.get("status", ""),
        "step1_warnings": list(unit.get("warnings") or unit.get("risk_notes") or []),
        "subject_id": target,
        "object_ids": [node_id],
        "graph_projection": {
            "node_candidates": [
                {
                    "node_id": node_id,
                    "type": object_types[0],
                    "label": unit.get("content", "")[:80],
                }
            ],
            "edge_candidates": [
                {
                    "source_node": slugify(target),
                    "relation_type": relation_type,
                    "target_node": node_id,
                    "candidate_status": "pending_review",
                }
            ],
        },
        "review_status": "pending",
        "review_action": "",
        "notes": "Generated by fixed Step 2 Build runner from Step 1 memory_unit; not durable until review decision.",
    }


def proposal_candidate_type(row: dict[str, Any]) -> tuple[str, str, str, list[str], str]:
    output_kind = str(row.get("output_kind") or "")
    candidate_type_hint = str(row.get("candidate_type") or "unknown")
    if output_kind == "portrait_fact_candidate":
        if candidate_type_hint == "preference":
            return "preference", "semantic", "global", ["preference"], "has_preference"
        if candidate_type_hint == "procedural":
            return "routine", "procedural", "task", ["routine"], "has_routine"
        if candidate_type_hint in {"project_context", "goal"}:
            return "project_context", "semantic", "project", ["project"], "has_project_context"
        if candidate_type_hint == "relationship_context":
            return "relationship_context", "episodic", "relationship", ["relationship"], "has_relationship_context"
        return "observation", "episodic", "event", ["observation"], "has_observation"
    if row.get("hypothesis_scope") in {"project", "topic"}:
        return "working_hypothesis", "semantic", str(row.get("hypothesis_scope") or "topic"), ["hypothesis"], "has_working_hypothesis"
    if row.get("hypothesis_scope") == "relationship":
        return "relationship_hypothesis", "semantic", "relationship", ["hypothesis"], "has_working_hypothesis"
    return "working_hypothesis", "episodic", str(row.get("hypothesis_scope") or "event"), ["hypothesis"], "has_working_hypothesis"


def evidence_excerpt_from_proposal(row: dict[str, Any]) -> str:
    quote = row.get("source_text_quote") or row.get("source_text") or row.get("original_text") or ""
    refs = row.get("evidence_refs") or []
    return f"{', '.join(refs)}: {quote}" if refs else str(quote)


def candidate_from_proposal_row(index: int, target: str, row: dict[str, Any]) -> dict[str, Any] | None:
    output_kind = str(row.get("output_kind") or "")
    if output_kind not in {"portrait_fact_candidate", "portrait_hypothesis_candidate"}:
        return None
    candidate_text = str(row.get("fact_candidate_text") or row.get("hypothesis_text") or row.get("candidate_text") or "").strip()
    if not candidate_text:
        return None
    evidence_refs = list(row.get("evidence_refs") or [])
    if not evidence_refs:
        return None
    candidate_type, memory_class, scope, object_types, relation_type = proposal_candidate_type(row)
    candidate_id = f"npc-{slugify(target)}-{index:03d}"
    node_id = f"{object_types[0]}:{short_hash(row.get('proposal_id') or candidate_text, 10)}"
    confidence = str(row.get("proposal_confidence") or row.get("hypothesis_confidence") or "unknown")
    if confidence == "unknown" and output_kind == "portrait_fact_candidate":
        confidence = "medium"
    inference_level = str(row.get("inference_level") or "unknown")
    if inference_level == "unknown" and output_kind == "portrait_fact_candidate":
        inference_level = "direct_inference"
    temporal_validity = "hypothesis_from_public_source" if output_kind == "portrait_hypothesis_candidate" else "current_or_historical_from_public_conversation"
    return {
        "candidate_id": candidate_id,
        "user_id": target,
        "input_layer": "s2_proposal_outcome",
        "input_refs": [row.get("proposal_id", "")],
        "source_refs": list(row.get("source_refs") or []),
        "evidence_refs": evidence_refs,
        "backpointer_refs": list(row.get("raw_backpointer_refs") or row.get("backpointer_refs") or evidence_refs),
        "evidence_excerpt": evidence_excerpt_from_proposal(row),
        "candidate_text": candidate_text,
        "candidate_type": candidate_type,
        "memory_class": memory_class,
        "scope": scope,
        "inference_level": inference_level,
        "confidence": confidence,
        "temporal_scope": {
            "validity": temporal_validity,
            "observed_at_refs": [{"display_ref": ref} for ref in evidence_refs],
        },
        "privacy_class": row.get("privacy_class", "public_dataset"),
        "subject_contamination_risk": row.get("subject_contamination_risk") or "unknown",
        "step1_generation_method": row.get("processing_method") or row.get("deterministic_processing_status") or "",
        "step1_status": "proposal_backed",
        "step1_warnings": list(row.get("warnings") or []),
        "subject_id": target,
        "object_ids": [node_id],
        "proposal_origin": {
            "proposal_id": row.get("proposal_id"),
            "proposal_input_id": row.get("proposal_input_id"),
            "proposal_run_id": row.get("proposal_run_id"),
            "output_kind": output_kind,
            "route_used": row.get("route_used"),
            "provider": row.get("provider"),
            "model_id": row.get("model_id"),
            "review_status": row.get("review_status"),
        },
        "graph_projection": {
            "node_candidates": [
                {
                    "node_id": node_id,
                    "type": object_types[0],
                    "label": candidate_text[:80],
                }
            ],
            "edge_candidates": [
                {
                    "source_node": slugify(target),
                    "relation_type": relation_type,
                    "target_node": node_id,
                    "candidate_status": "pending_review",
                }
            ],
        },
        "review_status": "pending",
        "review_action": "",
        "notes": "Generated by Step 2 Build from S2 proposal outcome; proposal is evidence-linked and not user-approved durable memory.",
    }


def load_proposal_backed_candidates(prebuild_result: dict[str, Any] | None, target: str, max_units: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    if not prebuild_result or prebuild_result.get("mode") != "route_and_propose":
        return [], {}
    proposal_info = prebuild_result.get("proposal") or {}
    proposal_path = ((proposal_info.get("outputs") or {}).get("proposals"))
    if not proposal_path:
        return [], {}
    rows = read_jsonl(Path(proposal_path))
    counts = {
        "proposal_rows": len(rows),
        "proposal_candidates": 0,
        "proposal_rejects": 0,
        "proposal_review_or_uncertain": 0,
        "proposal_not_materialized": 0,
    }
    candidates: list[dict[str, Any]] = []
    for row in rows:
        output_kind = str(row.get("output_kind") or "")
        if output_kind in {"portrait_fact_candidate", "portrait_hypothesis_candidate"}:
            candidate = candidate_from_proposal_row(len(candidates) + 1, target, row)
            if candidate is not None:
                candidates.append(candidate)
                counts["proposal_candidates"] += 1
            else:
                counts["proposal_not_materialized"] += 1
        elif output_kind == "reject":
            counts["proposal_rejects"] += 1
        elif output_kind in {"model_uncertain", "needs_human_review", "model_failure", "skipped"}:
            counts["proposal_review_or_uncertain"] += 1
        else:
            counts["proposal_not_materialized"] += 1
        if len(candidates) >= max_units:
            break
    return candidates, counts


def review_decision(index: int, candidate: dict[str, Any], target: str) -> dict[str, Any]:
    if candidate.get("input_layer") == "s2_proposal_outcome":
        reason = "S2 proposal outcome accepted only for public-dataset experiment; no real-user approval implied."
    else:
        reason = "Direct target-subject Step 1 memory unit with evidence refs; accepted only for public-dataset experiment."
    return {
        "decision_id": f"review-{slugify(target)}-{index:03d}",
        "candidate_id": candidate["candidate_id"],
        "review_status": "accepted_for_experiment",
        "review_action": "accept_for_experiment",
        "reason": reason,
        "confidence_after_review": candidate.get("confidence", "high"),
        "contamination_risk_after_review": candidate.get("subject_contamination_risk", "none"),
        "merge_target_candidate_id": "",
        "supersedes_candidate_ids": [],
        "notes": "No real-user approval implied; accepted_by_user intentionally not used.",
    }


def reviewed_unit(index: int, candidate: dict[str, Any], decision: dict[str, Any], target: str) -> dict[str, Any]:
    return {
        "unit_id": f"unit-{slugify(target)}-{index:03d}",
        "user_id": target,
        "type": candidate["candidate_type"],
        "memory_class": candidate["memory_class"],
        "content": candidate["candidate_text"],
        "scope": candidate["scope"],
        "source_refs": candidate["source_refs"],
        "evidence_refs": candidate["evidence_refs"],
        "backpointer_refs": candidate["backpointer_refs"],
        "evidence_summary": candidate["evidence_excerpt"],
        "evidence_quotes": [
            {"evidence_ref": ref, "quote": candidate["evidence_excerpt"]}
            for ref in candidate["evidence_refs"]
        ],
        "confidence": decision.get("confidence_after_review", candidate["confidence"]),
        "inference_level": candidate["inference_level"],
        "status": "active",
        "temporal_scope": candidate["temporal_scope"],
        "privacy_class": candidate["privacy_class"],
        "review_policy": "public_dataset_protocol_review_only",
        "relation_candidates": candidate["graph_projection"]["edge_candidates"],
        "review_metadata": {
            "review_status": decision["review_status"],
            "review_action": decision["review_action"],
            "subject_contamination_checked": True,
            "contamination_risk": decision.get("contamination_risk_after_review", "none"),
        },
        "step1_origin": {
            "input_layer": candidate["input_layer"],
            "input_refs": candidate["input_refs"],
            "generation_method": candidate["step1_generation_method"],
            "step1_status": candidate["step1_status"],
            "warnings": candidate["step1_warnings"],
        },
        "proposal_origin": candidate.get("proposal_origin", {}),
        "notes": "Reviewed portrait unit for Step 2 experiment; not real-user-approved memory.",
    }


def build_graph(target: str, units: list[dict[str, Any]], workspace_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    root = slugify(target)
    nodes: dict[str, dict[str, Any]] = {
        root: {
            "node_id": root,
            "user_id": target,
            "type": "user",
            "label": target,
            "summary": "Modeled target participant for public dataset Step 2 build.",
            "source_refs": sorted({ref for unit in units for ref in unit.get("source_refs", [])}),
            "evidence_refs": units[0].get("evidence_refs", [])[:1] if units else [],
            "confidence": "high",
            "inference_level": "explicit",
            "status": "active",
            "temporal_scope": {"validity": "current_or_historical_from_public_conversation"},
            "privacy_class": "public_dataset",
            "aliases": [target],
            "properties": {"workspace_id": workspace_id, "target_participant": target},
        }
    }
    edges: list[dict[str, Any]] = []
    for unit in units:
        for relation in unit.get("relation_candidates", []):
            target_node = relation.get("target_node")
            if not target_node:
                continue
            nodes.setdefault(
                target_node,
                {
                    "node_id": target_node,
                    "user_id": target,
                    "type": target_node.split(":", 1)[0] if ":" in target_node else "concept",
                    "label": target_node.split(":", 1)[-1].replace("-", " ").title(),
                    "summary": unit["content"],
                    "source_refs": unit["source_refs"],
                    "evidence_refs": unit["evidence_refs"],
                    "confidence": unit["confidence"],
                    "inference_level": unit["inference_level"],
                    "status": "active",
                    "temporal_scope": unit["temporal_scope"],
                    "privacy_class": unit["privacy_class"],
                    "aliases": [],
                    "properties": {
                        "from_unit_id": unit["unit_id"],
                        "candidate_status": unit["review_metadata"]["review_status"],
                    },
                },
            )
            edge_id = stable_id("edge", f"{root}|{relation.get('relation_type')}|{target_node}|{unit['unit_id']}")
            edges.append(
                {
                    "edge_id": edge_id,
                    "user_id": target,
                    "source_node": root,
                    "relation_type": relation.get("relation_type", "has_observation"),
                    "target_node": target_node,
                    "summary": unit["content"],
                    "source_refs": unit["source_refs"],
                    "evidence_refs": unit["evidence_refs"],
                    "confidence": unit["confidence"],
                    "inference_level": unit["inference_level"],
                    "status": "active",
                    "temporal_scope": unit["temporal_scope"],
                    "properties": {
                        "candidate_status": unit["review_metadata"]["review_status"],
                        "unit_id": unit["unit_id"],
                        "graph_is_not_proof": True,
                    },
                }
            )
    return list(nodes.values()), edges


def current_portrait_markdown(target: str, units: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for unit in units:
        grouped[unit.get("type", "observation")].append(unit)
    sections = [f"# Current Portrait: {target}", "", "This is a public-dataset experimental portrait, not user-approved durable memory.", ""]
    for unit_type in sorted(grouped):
        sections.extend([f"## {unit_type.replace('_', ' ').title()}", ""])
        for unit in grouped[unit_type]:
            refs = ", ".join(unit.get("evidence_refs", [])[:3])
            sections.append(f"- {unit['content']} (`{refs}`)")
        sections.append("")
    return "\n".join(sections)


def current_portrait_json(target: str, units: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "s2.current_portrait.v0.1",
        "user_id": target,
        "truth_status": "reviewed_public_dataset_model_context_not_raw_evidence",
        "unit_count": len(units),
        "sections": [
            {
                "type": unit.get("type", "observation"),
                "unit_id": unit["unit_id"],
                "content": unit["content"],
                "memory_class": unit["memory_class"],
                "scope": unit["scope"],
                "confidence": unit["confidence"],
                "evidence_refs": unit["evidence_refs"],
            }
            for unit in units
        ],
    }


def base_packet(workspace_id: str, target: str, units: list[dict[str, Any]], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "s2.base_assistance_packet.v0.1",
        "workspace_id": workspace_id,
        "user_id": target,
        "truth_status": "base_model_context_not_final_answer",
        "selected_portrait_units": [
            {
                "unit_id": unit["unit_id"],
                "content": unit["content"],
                "memory_class": unit["memory_class"],
                "scope": unit["scope"],
                "confidence": unit["confidence"],
                "evidence_refs": unit["evidence_refs"],
                "support_status": "not_checked",
            }
            for unit in units
        ],
        "graph_context": {
            "nodes": [node["node_id"] for node in nodes],
            "edges": [edge["edge_id"] for edge in edges],
            "use_policy": "graph_context_not_proof",
        },
        "uncertainty": [
            "Public dataset model context is accepted_for_experiment only.",
            "Facts still require Step 1 evidence resolution and support checks at query time.",
        ],
    }


def validate_outputs(candidates: list[dict[str, Any]], decisions: list[dict[str, Any]], units: list[dict[str, Any]], nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    if not candidates:
        raise ValueError("Step 2 build produced no normalized portrait candidates.")
    if len(candidates) != len(decisions) or len(candidates) != len(units):
        raise ValueError("Candidate, decision, and reviewed unit counts must match.")
    for candidate in candidates:
        if not candidate.get("evidence_refs"):
            raise ValueError(f"Candidate missing evidence_refs: {candidate.get('candidate_id')}")
        if candidate.get("review_status") != "pending":
            raise ValueError(f"Candidate should remain pending before review decision: {candidate.get('candidate_id')}")
    for decision in decisions:
        if decision.get("review_status") == "accepted_by_user":
            raise ValueError("Public dataset build must not use accepted_by_user.")
    for unit in units:
        if not unit.get("memory_class") or not unit.get("evidence_refs"):
            raise ValueError(f"Reviewed unit missing memory_class or evidence_refs: {unit.get('unit_id')}")
        if unit.get("review_metadata", {}).get("review_status") != "accepted_for_experiment":
            raise ValueError(f"Unexpected reviewed unit status: {unit.get('unit_id')}")
    known_nodes = {node["node_id"] for node in nodes}
    for edge in edges:
        if edge.get("source_node") not in known_nodes or edge.get("target_node") not in known_nodes:
            raise ValueError(f"Graph edge endpoint cannot resolve: {edge.get('edge_id')}")
        if not edge.get("evidence_refs"):
            raise ValueError(f"Graph edge missing evidence_refs: {edge.get('edge_id')}")


def write_manifest_yaml(path: Path, workspace_id: str, target: str, run_id: str, counts: dict[str, int]) -> None:
    lines = [
        "schema_version: s2.workspace_manifest.v0.1",
        f"workspace_id: {workspace_id}",
        f"modeled_user_id: {target}",
        f"target_participant: {target}",
        f"run_id: {run_id}",
        "truth_status: public_dataset_experiment_not_user_approved",
        "outputs:",
        "  normalized_candidates: portrait/normalized_candidates.jsonl",
        "  review_decisions: portrait/review_decisions.jsonl",
        "  reviewed_units: portrait/reviewed_units.jsonl",
        "  current_portrait: portrait/current_portrait.md",
        "  graph_nodes: graph/nodes.jsonl",
        "  graph_edges: graph/edges.jsonl",
        "  base_assistance_packet: packets/base_assistance_packet.json",
        "counts:",
    ]
    lines.extend(f"  {key}: {value}" for key, value in counts.items())
    write_text(path, "\n".join(lines) + "\n")


def build_reports(workspace: Path, workspace_id: str, target: str, run_id: str, counts: dict[str, int], build_source: str) -> None:
    write_text(
        workspace / "reports" / "target-selection.md",
        f"""# Target Selection

- workspace_id: `{workspace_id}`
- modeled_user_id: `{target}`
- target_participant: `{target}`
- input_basis: Step 1 memory units with target speaker lock
- status: selected for public dataset experiment
""",
    )
    write_text(
        workspace / "reports" / "protocol-review-report.md",
        f"""# Protocol Review Report

- run_id: `{run_id}`
- review_status: `accepted_for_experiment`
- accepted_by_user used: `false`
- evidence discipline: every reviewed unit preserves evidence_refs/backpointer_refs
- build_source: `{build_source}`
- subject contamination policy: routed S2 proposal candidates or target-speaker Step 1 memory units only
- graph policy: graph context is relationship context, not proof
""",
    )
    write_text(
        workspace / "reports" / "phase2-pilot-report.md",
        f"""# Step 2 Build Runner Report

## Status

- run_id: `{run_id}`
- workspace_id: `{workspace_id}`
- modeled_user_id: `{target}`
- runner: `tools/step2/step2_build_runner.py`
- status: `completed`
- build_source: `{build_source}`

## Counts

- normalized_candidates: {counts["normalized_candidates"]}
- review_decisions: {counts["review_decisions"]}
- reviewed_units: {counts["reviewed_units"]}
- graph_nodes: {counts["graph_nodes"]}
- graph_edges: {counts["graph_edges"]}
- proposal_rows: {counts.get("proposal_rows", 0)}
- proposal_candidates: {counts.get("proposal_candidates", 0)}
- proposal_rejects: {counts.get("proposal_rejects", 0)}
- proposal_review_or_uncertain: {counts.get("proposal_review_or_uncertain", 0)}

## Boundaries

- In `route_and_propose` mode, this runner consumes S2 proposal outcomes before materializing experimental Step 2 assets.
- In deterministic regression mode, this runner consumes Step 1 memory units and evidence refs directly.
- It does not build S1 assets, S1 indexes, S2 embedding indexes, query packets, or final answers.
- It replaces ad-hoc temporary Step 2 build scripts for routine public-dataset S2 Build runs.
""",
    )


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    started_at = now_iso()
    workspace = Path(args.workspace).resolve()
    workspace_id = args.workspace_id or workspace.name
    target = args.target_participant or args.modeled_user_id
    if not target:
        raise ValueError("Step 2 build requires --modeled-user-id or --target-participant.")
    if args.duplicate_policy not in SUPPORTED_DUPLICATE_POLICIES:
        raise ValueError(f"Unsupported duplicate_policy: {args.duplicate_policy}")

    memory_units_path = workspace / "memory" / "memory_units.jsonl"
    evidence_path = workspace / "evidence" / "evidence.jsonl"
    build_manifest_path = workspace / "evidence" / "build_manifest.json"
    if not memory_units_path.exists():
        raise FileNotFoundError(f"Missing Step 1 memory units: {memory_units_path}")
    if not evidence_path.exists():
        raise FileNotFoundError(f"Missing Step 1 evidence registry: {evidence_path}")

    prepare_outputs(workspace, args.duplicate_policy)
    prebuild_options = options_from_args(
        args,
        step_name="s2",
        target_task="s2_portrait_candidate",
        default_profile=DEFAULT_S2_PROPOSAL_PROFILE,
    )
    prebuild_result = run_prebuild_routing(
        project_root=Path.cwd().resolve(),
        route_workspace=workspace,
        proposal_workspace=workspace,
        output_workspace=workspace,
        base_run_id=args.run_id or f"s2-build:{workspace_id}:{slugify(target)}",
        options=prebuild_options,
    )
    memory_units = read_jsonl(memory_units_path)
    candidates, proposal_counts = load_proposal_backed_candidates(prebuild_result, target, args.max_units)
    build_source = "s2_proposal_outcomes" if proposal_counts else "s1_memory_units_deterministic"
    if not proposal_counts:
        selected = choose_memory_units(memory_units, target, args.max_units)
        candidates = [candidate_from_memory_unit(index, target, unit) for index, unit in enumerate(selected, 1)]
    decisions = [review_decision(index, candidate, target) for index, candidate in enumerate(candidates, 1)]
    reviewed = [reviewed_unit(index, candidate, decisions[index - 1], target) for index, candidate in enumerate(candidates, 1)]
    nodes, edges = build_graph(target, reviewed, workspace_id)
    validate_outputs(candidates, decisions, reviewed, nodes, edges)

    write_jsonl(workspace / "portrait" / "normalized_candidates.jsonl", candidates)
    write_jsonl(workspace / "portrait" / "review_decisions.jsonl", decisions)
    write_jsonl(workspace / "portrait" / "reviewed_units.jsonl", reviewed)
    write_text(workspace / "portrait" / "current_portrait.md", current_portrait_markdown(target, reviewed))
    write_json(workspace / "portrait" / "current_portrait.json", current_portrait_json(target, reviewed))
    write_jsonl(workspace / "graph" / "nodes.jsonl", nodes)
    write_jsonl(workspace / "graph" / "edges.jsonl", edges)
    write_text(
        workspace / "graph" / "graph-summary.md",
        f"# User-rooted Graph Summary\n\n- user: `{target}`\n- nodes: {len(nodes)}\n- edges: {len(edges)}\n- policy: graph context is not proof\n",
    )
    write_json(workspace / "packets" / "base_assistance_packet.json", base_packet(workspace_id, target, reviewed, nodes, edges))

    run_id = args.run_id or f"s2-build:{workspace_id}:{slugify(target)}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    counts = {
        "normalized_candidates": len(candidates),
        "review_decisions": len(decisions),
        "reviewed_units": len(reviewed),
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "input_memory_units": len(memory_units),
        "build_source_s2_proposal_outcomes": 1 if build_source == "s2_proposal_outcomes" else 0,
    }
    counts.update(proposal_counts)
    write_manifest_yaml(workspace / "manifest.yaml", workspace_id, target, run_id, counts)
    status = {
        "schema_version": "s2.build_status.v0.1",
        "workspace_id": workspace_id,
        "modeled_user_id": target,
        "target_participant": target,
        "run_id": run_id,
        "current_step": "completed",
        "completed_steps": [
            "target_selection",
            "normalized_portrait_candidates",
            "confidence_risk_review",
            "reviewed_portrait_units",
            "current_portrait",
            "user_rooted_graph",
            "base_assistance_packet",
        ],
        "blocked": False,
        "blockers": [],
        "build_source": build_source,
        "started_at": started_at,
        "last_updated": now_iso(),
        "latest_outputs": {
            "normalized_candidates": "portrait/normalized_candidates.jsonl",
            "reviewed_units": "portrait/reviewed_units.jsonl",
            "graph_nodes": "graph/nodes.jsonl",
            "graph_edges": "graph/edges.jsonl",
            "base_assistance_packet": "packets/base_assistance_packet.json",
        },
    }
    if prebuild_result is not None:
        status["prebuild_routing"] = prebuild_result
    write_json(workspace / "checkpoints" / "phase2-status.json", status)
    write_text(
        workspace / "checkpoints" / "run-log.md",
        f"""# Step 2 Build Run Log

- run_id: `{run_id}`
- runner: `tools/step2/step2_build_runner.py`
- started_at: `{started_at}`
- completed_at: `{status["last_updated"]}`
- input_memory_units: `{memory_units_path}`
- input_evidence_registry: `{evidence_path}`
- input_evidence_build_manifest: `{build_manifest_path if build_manifest_path.exists() else "missing"}`
- build_source: `{build_source}`
- input_memory_units: {len(memory_units)}
- materialized_candidates: {len(candidates)}
- outputs_written: {", ".join(status["latest_outputs"].values())}
""",
    )
    build_reports(workspace, workspace_id, target, run_id, counts, build_source)
    return {
        "workspace": str(workspace),
        "run_id": run_id,
        "counts": counts,
        "outputs": status["latest_outputs"],
        "prebuild_routing": prebuild_result,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Step 2 user-model assets from Step 1 memory units.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument("--modeled-user-id", default=None)
    parser.add_argument("--target-participant", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-units", type=int, default=30)
    parser.add_argument("--duplicate-policy", default="fail", choices=sorted(SUPPORTED_DUPLICATE_POLICIES))
    add_prebuild_arguments(parser, default_profile=DEFAULT_S2_PROPOSAL_PROFILE)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_build(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
