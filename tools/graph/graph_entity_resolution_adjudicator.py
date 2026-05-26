"""LLM-assisted adjudication for graph entity resolution candidates.

This runner consumes graph_entity_resolution_candidates.jsonl and asks a model
to decide whether each candidate should be merged, kept separate, left
uncertain, or escalated for human review. It does not rewrite graph nodes or
edges; it only emits adjudication rows that can be audited and calibrated.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.proposals.proposal_runner import (
    SUPPORTED_API_MODES,
    PromptPolicy,
    build_provider,
    estimate_tokens,
    load_dotenv,
    resolve_live_api,
)
from tools.graph.graph_construction_packet_builder import (
    file_hash,
    read_json,
    read_jsonl,
    sha256_text,
    stable_id,
    unique_strings,
    write_json,
    write_jsonl,
    write_text,
)


SCHEMA_VERSION = "graph_v03.entity_resolution_adjudication_run.v0.1"
ADJUDICATION_SCHEMA_VERSION = "graph_v03.entity_merge_adjudication.v0.1"
FAILURE_SCHEMA_VERSION = "graph_v03.entity_merge_adjudication_failure.v0.1"
DEFAULT_CONSOLIDATION_DIR_NAME = "graph_v03_consolidation"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_entity_resolution_adjudication"
SUPPORTED_PROVIDERS = {"mock", "external_jsonl", "openai"}
SUPPORTED_DECISIONS = {"merge", "keep_separate", "uncertain", "needs_human_review"}
DEFAULT_PROMPT_TEXT = """You are adjudicating graph entity resolution candidates.

Decide whether the listed graph nodes refer to the same real-world entity.
Be strict about evidence. Related entities are not duplicates. Family members,
spouses, pronouns with unclear antecedents, roles, places, events, and concepts
must not be merged unless the evidence clearly identifies the same entity.

Return strict JSON only:
{
  "output_kind": "entity_merge_adjudication",
  "decision": "merge|keep_separate|uncertain|needs_human_review",
  "confidence_hint": "high|medium|low",
  "reason": "short explanation grounded in evidence",
  "supporting_signals": ["..."],
  "conflicting_signals": ["..."],
  "evidence_refs": ["..."],
  "warnings": ["..."],
  "graph_is_not_proof": true,
  "write_permission": false
}

Decision guidance:
- merge: evidence strongly says the nodes are the same entity or stable alias.
- keep_separate: evidence says they are different entities, roles, or related-but-distinct.
- uncertain: evidence is suggestive but insufficient.
- needs_human_review: high-impact ambiguity, conflicting evidence, or unsafe attribution.
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def short_hash(value: str, length: int = 16) -> str:
    return sha256_text(value)[:length]


def resolve_workspace(project_root: Path, workspace: Path) -> Path:
    return workspace.resolve() if workspace.is_absolute() else (project_root / workspace).resolve()


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def parse_model_json(output_text: str) -> tuple[dict[str, Any] | None, list[str]]:
    text = output_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None, ["invalid_json"]
    if not isinstance(payload, dict):
        return None, ["model_output_not_object"]
    return payload, []


def compact_node(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": node.get("node_id") or "",
        "label": node.get("label") or "",
        "normalized_key": node.get("normalized_key") or "",
        "entity_type": node.get("entity_type") or "",
        "candidate_ids": node.get("candidate_ids") or [],
        "entity_quality_hint": node.get("entity_quality_hint") or "",
        "entity_quality_reasons": node.get("entity_quality_reasons") or [],
        "evidence_refs": node.get("evidence_refs") or [],
        "raw_backpointer_refs": node.get("raw_backpointer_refs") or [],
        "source_text_quotes": node.get("source_text_quotes") or [],
        "warnings": node.get("warnings") or [],
    }


def compact_edge(edge: dict[str, Any], node_ids: set[str]) -> dict[str, Any] | None:
    if str(edge.get("source_node_id") or "") not in node_ids and str(edge.get("target_node_id") or "") not in node_ids:
        return None
    return {
        "edge_id": edge.get("edge_id") or "",
        "source_node_id": edge.get("source_node_id") or "",
        "source_label": edge.get("source_label") or "",
        "relation_type": edge.get("relation_type") or "",
        "target_node_id": edge.get("target_node_id") or "",
        "target_label": edge.get("target_label") or "",
        "description": edge.get("description") or "",
        "evidence_refs": edge.get("evidence_refs") or [],
        "warnings": edge.get("warnings") or [],
    }


def compact_evidence(evidence_rows: list[dict[str, Any]], owner_ids: set[str], evidence_refs: set[str], limit: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in evidence_rows:
        owner_id = str(row.get("owner_id") or "")
        row_refs = set(str(ref) for ref in row.get("evidence_refs") or [] if ref)
        direct_ref = str(row.get("evidence_ref") or "")
        if owner_id not in owner_ids and not (row_refs & evidence_refs) and direct_ref not in evidence_refs:
            continue
        out.append(
            {
                "owner_id": owner_id,
                "owner_kind": row.get("owner_kind") or "",
                "candidate_id": row.get("candidate_id") or "",
                "evidence_ref": direct_ref,
                "evidence_refs": row.get("evidence_refs") or [],
                "source_text_quote": row.get("source_text_quote") or "",
                "source_text_excerpt": row.get("source_text_excerpt") or "",
                "warnings": row.get("warnings") or [],
            }
        )
        if len(out) >= limit:
            break
    return out


def build_adjudication_input(
    candidate: dict[str, Any],
    *,
    nodes_by_id: dict[str, dict[str, Any]],
    edge_rows: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    evidence_limit: int,
    edge_limit: int,
) -> dict[str, Any]:
    node_ids = {str(node_id) for node_id in candidate.get("node_ids") or [] if str(node_id)}
    nodes = [compact_node(nodes_by_id[node_id]) for node_id in sorted(node_ids) if node_id in nodes_by_id]
    edges = [row for row in (compact_edge(edge, node_ids) for edge in edge_rows) if row is not None]
    edges = edges[: max(0, edge_limit)]
    owner_ids = set(node_ids)
    owner_ids.update(str(edge.get("edge_id") or "") for edge in edges if edge.get("edge_id"))
    evidence_refs = {str(ref) for ref in candidate.get("evidence_refs") or [] if str(ref)}
    return {
        "schema_version": "graph_v03.entity_resolution_adjudication_input.v0.1",
        "proposal_input_id": stable_id("graph_entity_resolution_adjudication_input", str(candidate.get("resolution_candidate_id") or "")),
        "target_task": "graph_entity_resolution_adjudication",
        "resolution_candidate_id": candidate.get("resolution_candidate_id") or "",
        "candidate": {
            "labels": candidate.get("labels") or [],
            "node_ids": candidate.get("node_ids") or [],
            "entity_types": candidate.get("entity_types") or [],
            "score": candidate.get("score"),
            "confidence_hint": candidate.get("confidence_hint") or "",
            "signals": candidate.get("signals") or [],
            "evidence_refs": candidate.get("evidence_refs") or [],
            "warnings": candidate.get("warnings") or [],
        },
        "nodes": nodes,
        "local_neighborhood_edges": edges,
        "evidence": compact_evidence(evidence_rows, owner_ids, evidence_refs, evidence_limit),
        "allowed_decisions": sorted(SUPPORTED_DECISIONS),
        "graph_is_not_proof": True,
        "write_permission": False,
    }


def mock_adjudication(input_packet: dict[str, Any]) -> dict[str, Any]:
    labels = [str(label).lower() for label in input_packet.get("candidate", {}).get("labels") or []]
    signals = set(input_packet.get("candidate", {}).get("signals") or [])
    warnings: list[str] = ["mock_provider_output"]
    if any("wife" in label for label in labels) and any("doctor" in label for label in labels):
        decision = "keep_separate"
        confidence = "high"
        reason = "One label refers to a spouse/wife while another refers to the Doctor; related is not same entity."
    elif {"label_token_overlap", "normalized_label_containment"} & signals and {"shared_evidence_refs", "shared_raw_backpointer_refs"} & signals:
        decision = "merge"
        confidence = "medium"
        reason = "The candidate has lexical alias signals plus shared evidence/backpointer support."
    else:
        decision = "uncertain"
        confidence = "low"
        reason = "The candidate is suggestive but does not have enough alias evidence for a merge."
    return {
        "output_kind": "entity_merge_adjudication",
        "decision": decision,
        "confidence_hint": confidence,
        "reason": reason,
        "supporting_signals": sorted(signals),
        "conflicting_signals": [],
        "evidence_refs": input_packet.get("candidate", {}).get("evidence_refs") or [],
        "warnings": warnings,
        "graph_is_not_proof": True,
        "write_permission": False,
    }


def normalize_adjudication(
    *,
    candidate: dict[str, Any],
    payload: dict[str, Any],
    provider: str,
    model_id: str,
    prompt: PromptPolicy,
    input_packet: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    resolution_candidate_id = str(candidate.get("resolution_candidate_id") or "")
    output_kind = str(payload.get("output_kind") or "")
    decision = str(payload.get("decision") or "")
    warnings = unique_strings(candidate.get("warnings"), payload.get("warnings"))
    if output_kind != "entity_merge_adjudication" or decision not in SUPPORTED_DECISIONS:
        return None, failure_row(
            candidate,
            "schema_validation_failed",
            f"unsupported output_kind/decision: {output_kind}/{decision}",
            warnings=unique_strings(warnings, ["schema_validation_failed"]),
        )
    if payload.get("graph_is_not_proof") is not True:
        warnings.append("graph_is_not_proof_not_true")
    if payload.get("write_permission") is not False:
        warnings.append("write_permission_not_false")
    row = {
        "schema_version": ADJUDICATION_SCHEMA_VERSION,
        "adjudication_id": stable_id("graph_entity_merge_adjudication", f"{resolution_candidate_id}|{provider}|{model_id}|{decision}|{payload.get('reason') or ''}"),
        "resolution_candidate_id": resolution_candidate_id,
        "node_ids": candidate.get("node_ids") or [],
        "labels": candidate.get("labels") or [],
        "candidate_score": candidate.get("score"),
        "candidate_confidence_hint": candidate.get("confidence_hint") or "",
        "candidate_signals": candidate.get("signals") or [],
        "decision": decision,
        "confidence_hint": str(payload.get("confidence_hint") or "unknown"),
        "reason": str(payload.get("reason") or ""),
        "supporting_signals": unique_strings(payload.get("supporting_signals")),
        "conflicting_signals": unique_strings(payload.get("conflicting_signals")),
        "evidence_refs": unique_strings(payload.get("evidence_refs"), candidate.get("evidence_refs")),
        "source_candidate_warnings": candidate.get("warnings") or [],
        "model_id": model_id,
        "provider": provider,
        "prompt_policy_id": prompt.policy_id,
        "prompt_hash": prompt.prompt_hash,
        "input_hash": short_hash(json.dumps(input_packet, ensure_ascii=False, sort_keys=True)),
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
        "merge_applied": False,
        "warnings": warnings,
    }
    return row, None


def failure_row(candidate: dict[str, Any], failure_kind: str, reason: str, *, warnings: list[str] | None = None) -> dict[str, Any]:
    resolution_candidate_id = str(candidate.get("resolution_candidate_id") or "")
    return {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "failure_id": stable_id("graph_entity_merge_adjudication_failure", f"{resolution_candidate_id}|{failure_kind}|{reason}"),
        "resolution_candidate_id": resolution_candidate_id,
        "labels": candidate.get("labels") or [],
        "failure_kind": failure_kind,
        "reason": reason,
        "graph_is_not_proof": True,
        "write_permission": False,
        "support_status": "not_checked",
        "warnings": warnings or [],
    }


def write_report(path: Path, manifest: dict[str, Any]) -> None:
    counts = manifest["counts"]
    lines = [
        "# v0.3 Graph Entity Resolution Adjudication Report",
        "",
        f"- workspace: `{manifest['workspace_id']}`",
        f"- output_dir: `{manifest['output_dir']}`",
        f"- provider: `{manifest['provider']}`",
        f"- model: `{manifest['model_id']}`",
        "- graph_is_not_proof: `true`",
        "- write_permission: `false`",
        "",
        "## Counts",
        "",
        f"- input_candidates: {counts['input_candidate_count']}",
        f"- adjudications: {counts['adjudication_count']}",
        f"- failures: {counts['failure_count']}",
        f"- decision_counts: `{json.dumps(counts['decision_counts'], ensure_ascii=False, sort_keys=True)}`",
        "",
        "## Boundary",
        "",
        "- LLM adjudication controls only merge recommendations.",
        "- This runner does not rewrite graph_nodes_table or graph_edges_table.",
        "- Merge decisions remain candidate material, not graph truth.",
        "",
    ]
    write_text(path, "\n".join(lines))


def run_entity_resolution_adjudicator(
    workspace: Path,
    *,
    project_root: Path | None = None,
    consolidation_dir: Path | None = None,
    output_dir: Path | None = None,
    provider: str = "openai",
    api_mode: str | None = None,
    allow_live_api: bool = False,
    model: str | None = None,
    env_file: str | Path = ".env",
    external_model_outputs: Path | None = None,
    max_items: int | None = None,
    item_offset: int = 0,
    min_score: float = 0.0,
    provider_concurrency: int = 1,
    evidence_limit: int = 12,
    edge_limit: int = 24,
) -> dict[str, Any]:
    project_root = (project_root or Path(__file__).resolve().parents[2]).resolve()
    workspace = resolve_workspace(project_root, workspace)
    env_path = resolve_project_path(project_root, env_file)
    if load_dotenv is not None and env_path.exists():
        load_dotenv(env_path, override=True)
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    api_mode = api_mode or os.environ.get("OPENAI_API_MODE") or "chat_completions"
    if api_mode not in SUPPORTED_API_MODES:
        raise ValueError(f"Unsupported api_mode: {api_mode}")
    live_api_enabled, live_api_unlock_source = resolve_live_api(provider, bool(allow_live_api))
    model = model or os.environ.get("OPENAI_MODEL_STRONG") or "gpt-4o"
    consolidation_dir = (consolidation_dir or workspace / DEFAULT_CONSOLIDATION_DIR_NAME).resolve()
    output_dir = (output_dir or workspace / DEFAULT_OUTPUT_DIR_NAME).resolve()

    candidate_path = consolidation_dir / "graph_entity_resolution_candidates.jsonl"
    node_path = consolidation_dir / "graph_nodes_table.jsonl"
    edge_path = consolidation_dir / "graph_edges_table.jsonl"
    evidence_path = consolidation_dir / "evidence_links.jsonl"
    manifest_path = consolidation_dir / "graph_consolidation_manifest.json"

    candidates = [row for row in read_jsonl(candidate_path) if float(row.get("score") or 0.0) >= min_score]
    candidates.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
    if item_offset:
        candidates = candidates[item_offset:]
    if max_items is not None:
        candidates = candidates[: max(0, max_items)]
    nodes = read_jsonl(node_path)
    edges = read_jsonl(edge_path)
    evidence = read_jsonl(evidence_path)
    consolidation_manifest = read_json(manifest_path) if manifest_path.exists() else {}
    nodes_by_id = {str(row.get("node_id")): row for row in nodes if row.get("node_id")}

    prompt = PromptPolicy(
        policy_id="graph_entity_resolution_adjudication_prompt.v0.1",
        path=Path("<embedded>"),
        text=DEFAULT_PROMPT_TEXT,
        prompt_hash=sha256_text(DEFAULT_PROMPT_TEXT),
    )
    model_provider = build_provider(provider, api_mode, external_model_outputs) if provider != "mock" else None
    provider_concurrency = max(1, int(provider_concurrency or 1))

    model_call_inputs: list[dict[str, Any]] = []
    model_call_results: list[dict[str, Any]] = []
    adjudications: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def call_provider(index: int, candidate: dict[str, Any]) -> dict[str, Any]:
        input_packet = build_adjudication_input(
            candidate,
            nodes_by_id=nodes_by_id,
            edge_rows=edges,
            evidence_rows=evidence,
            evidence_limit=evidence_limit,
            edge_limit=edge_limit,
        )
        input_row = {
            "schema_version": "graph_v03.entity_resolution_adjudication_model_input.v0.1",
            "resolution_candidate_id": candidate.get("resolution_candidate_id") or "",
            "model_id": model,
            "provider": provider,
            "api_mode": api_mode,
            "prompt_policy_id": prompt.policy_id,
            "prompt_hash": prompt.prompt_hash,
            "input_packet": input_packet,
            "estimated_input_tokens": estimate_tokens(prompt.text + json.dumps(input_packet, ensure_ascii=False)),
            "graph_is_not_proof": True,
        }
        try:
            started = time.perf_counter()
            if provider == "mock":
                payload = mock_adjudication(input_packet)
                result_row = {
                    "resolution_candidate_id": candidate.get("resolution_candidate_id") or "",
                    "model_id": model,
                    "provider": provider,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "parse_errors": [],
                }
                return {"index": index, "candidate": candidate, "input_row": input_row, "payload": payload, "parse_errors": [], "result_row": result_row, "exception": None}
            result = model_provider.generate(prompt=prompt, model_id=model, input_packet=input_packet) if model_provider else None
            if result is None:
                raise RuntimeError("model provider was not initialized")
            payload, parse_errors = parse_model_json(result.output_text)
            result_row = {
                "resolution_candidate_id": candidate.get("resolution_candidate_id") or "",
                "model_id": result.model_id,
                "provider": result.provider,
                "estimated_input_tokens": result.estimated_input_tokens,
                "estimated_output_tokens": result.estimated_output_tokens,
                "latency_ms": result.latency_ms,
                "parse_errors": parse_errors,
            }
            return {"index": index, "candidate": candidate, "input_row": input_row, "payload": payload, "parse_errors": parse_errors, "result_row": result_row, "exception": None}
        except Exception as exc:  # pragma: no cover - exercised by live/provider integration.
            return {"index": index, "candidate": candidate, "input_row": input_row, "payload": None, "parse_errors": [], "result_row": None, "exception": exc}

    results: dict[int, dict[str, Any]] = {}
    if provider_concurrency > 1 and len(candidates) > 1:
        with ThreadPoolExecutor(max_workers=provider_concurrency) as executor:
            futures = {executor.submit(call_provider, index, candidate): index for index, candidate in enumerate(candidates)}
            for future in as_completed(futures):
                result = future.result()
                results[int(result["index"])] = result

    for index, candidate in enumerate(candidates):
        result = results.get(index) or call_provider(index, candidate)
        model_call_inputs.append(result["input_row"])
        if result["result_row"]:
            model_call_results.append(result["result_row"])
        if result["exception"] is not None:
            failures.append(failure_row(candidate, "provider_call_failed", f"{type(result['exception']).__name__}: {result['exception']}", warnings=["provider_error_redacted"]))
            continue
        if result["payload"] is None:
            failures.append(failure_row(candidate, "schema_validation_failed", "model output was not valid strict JSON", warnings=result["parse_errors"]))
            continue
        row, failure = normalize_adjudication(
            candidate=candidate,
            payload=result["payload"],
            provider=provider,
            model_id=model,
            prompt=prompt,
            input_packet=result["input_row"]["input_packet"],
        )
        if row:
            adjudications.append(row)
        if failure:
            failures.append(failure)

    outputs = {
        "graph_entity_merge_adjudications": output_dir / "graph_entity_merge_adjudications.jsonl",
        "graph_entity_merge_adjudication_failures": output_dir / "graph_entity_merge_adjudication_failures.jsonl",
        "model_call_inputs": output_dir / "graph_entity_merge_adjudication_model_call_inputs.jsonl",
        "model_call_results": output_dir / "graph_entity_merge_adjudication_model_call_results.jsonl",
        "manifest": output_dir / "graph_entity_merge_adjudication_manifest.json",
        "report": output_dir / "graph_entity_merge_adjudication_report.md",
    }
    write_jsonl(outputs["graph_entity_merge_adjudications"], adjudications)
    write_jsonl(outputs["graph_entity_merge_adjudication_failures"], failures)
    write_jsonl(outputs["model_call_inputs"], model_call_inputs)
    write_jsonl(outputs["model_call_results"], model_call_results)

    decision_counts = Counter(str(row.get("decision") or "") for row in adjudications)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "workspace_id": workspace.name,
        "consolidation_dir": str(consolidation_dir),
        "output_dir": str(output_dir),
        "provider": provider,
        "api_mode": api_mode,
        "model_id": model,
        "prompt_policy_id": prompt.policy_id,
        "prompt_hash": prompt.prompt_hash,
        "live_api_enabled": live_api_enabled,
        "live_api_unlock_source": live_api_unlock_source,
        "api_key_recorded": False,
        "upstream_consolidation_manifest": consolidation_manifest.get("schema_version", ""),
        "source_asset_hashes": {
            "graph_entity_resolution_candidates": file_hash(candidate_path),
            "graph_nodes_table": file_hash(node_path),
            "graph_edges_table": file_hash(edge_path),
            "evidence_links": file_hash(evidence_path),
        },
        "counts": {
            "input_candidate_count": len(candidates),
            "adjudication_count": len(adjudications),
            "failure_count": len(failures),
            "decision_counts": dict(sorted(decision_counts.items())),
        },
        "policies": {
            "candidate_first": True,
            "llm_controls_merge_recommendation_only": True,
            "merge_applied": False,
            "graph_is_not_proof": True,
            "write_permission": False,
            "support_status": "not_checked",
            "no_graph_truth_written": True,
            "no_durable_memory_written": True,
        },
        "outputs": {key: str(value) for key, value in outputs.items()},
    }
    write_json(outputs["manifest"], manifest)
    write_report(outputs["report"], manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Adjudicate graph entity resolution candidates with an LLM.")
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--consolidation-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--provider", default="openai", choices=sorted(SUPPORTED_PROVIDERS))
    parser.add_argument("--api-mode", default=None, choices=sorted(SUPPORTED_API_MODES))
    parser.add_argument("--allow-live-api", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--external-model-outputs", type=Path, default=None)
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--item-offset", type=int, default=0)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--provider-concurrency", type=int, default=1)
    parser.add_argument("--evidence-limit", type=int, default=12)
    parser.add_argument("--edge-limit", type=int, default=24)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_entity_resolution_adjudicator(
        args.workspace,
        consolidation_dir=args.consolidation_dir,
        output_dir=args.output_dir,
        provider=args.provider,
        api_mode=args.api_mode,
        allow_live_api=args.allow_live_api,
        model=args.model,
        env_file=args.env_file,
        external_model_outputs=args.external_model_outputs,
        max_items=args.max_items,
        item_offset=args.item_offset,
        min_score=args.min_score,
        provider_concurrency=args.provider_concurrency,
        evidence_limit=args.evidence_limit,
        edge_limit=args.edge_limit,
    )
    print(json.dumps({"manifest": manifest["outputs"]["manifest"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
