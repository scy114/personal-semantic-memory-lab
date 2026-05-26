"""Resolve graph packets and candidates affected by S1/S2 maintenance deltas.

This v0.4 helper is read-only. It maps S1 delta decisions and S2 affected-unit
rows to graph construction packets, graph candidates, and downstream graph
asset refresh recommendations. It does not run extraction, merge entities,
write graph truth, or rebuild NetworkX/query/visualization artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "maintenance.graph_affected_packet.v0.4"
REPORT_SCHEMA_VERSION = "maintenance.graph_affected_packet_report.v0.4"

GRAPH_DELTA_OUTPUTS = {
    "new_node_candidate",
    "new_edge_candidate",
    "new_claim_candidate",
    "edge_superseded",
    "edge_weakened",
    "edge_contradicted",
    "edge_historical_only",
    "merge_candidate",
    "needs_review",
    "no_graph_change",
}

NO_REFRESH_S1_DELTAS = {"no_material_change"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Graph affected resolver input row must be an object at {path}:{line_number}")
        rows.append(row)
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalize(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def normalize_unique(values: Iterable[Any]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        text = normalize(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


def list_refs(row: dict[str, Any], *field_names: str) -> list[str]:
    refs: list[str] = []
    for field_name in field_names:
        value = row.get(field_name)
        if value is None:
            continue
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    for key in ("evidence_ref", "display_ref", "canonical_evidence_ref", "source_packet_id", "id"):
                        if item.get(key):
                            refs.append(str(item[key]))
                            break
                else:
                    refs.append(str(item))
        elif isinstance(value, dict):
            for key in ("evidence_ref", "display_ref", "canonical_evidence_ref", "source_packet_id", "id"):
                if value.get(key):
                    refs.append(str(value[key]))
                    break
        else:
            refs.append(str(value))
    return normalize_unique(refs)


def packet_id(row: dict[str, Any]) -> str:
    return normalize(row.get("packet_id") or row.get("source_packet_id") or row.get("id"))


def candidate_id(row: dict[str, Any]) -> str:
    for field_name in ("edge_id", "node_id", "claim_id", "link_id", "candidate_id", "id"):
        value = row.get(field_name)
        if value:
            return str(value)
    return ""


def candidate_kind(row: dict[str, Any]) -> str:
    schema = normalize(row.get("schema_version"))
    if row.get("edge_id") or "edge_table" in schema:
        return "edge"
    if row.get("node_id") or "node_table" in schema:
        return "node"
    if row.get("claim_id") or "claim_table" in schema:
        return "claim"
    if row.get("link_id") or "evidence_link" in schema:
        return "evidence_link"
    return normalize(row.get("candidate_kind") or "graph_candidate")


def build_packet_lookup(graph_packet_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_packet_id: dict[str, dict[str, Any]] = {}
    by_evidence_ref: dict[str, set[str]] = {}
    by_primary_evidence_ref: dict[str, set[str]] = {}
    by_context_evidence_ref: dict[str, set[str]] = {}
    for row in graph_packet_rows:
        pid = packet_id(row)
        if not pid:
            continue
        by_packet_id[pid] = row
        for ref in list_refs(row, "evidence_refs", "primary_evidence_refs", "context_evidence_refs", "input_ref"):
            by_evidence_ref.setdefault(ref, set()).add(pid)
        for ref in list_refs(row, "evidence_refs", "primary_evidence_refs", "input_ref"):
            by_primary_evidence_ref.setdefault(ref, set()).add(pid)
        for ref in list_refs(row, "context_evidence_refs"):
            by_context_evidence_ref.setdefault(ref, set()).add(pid)
    return {
        "by_packet_id": by_packet_id,
        "by_evidence_ref": by_evidence_ref,
        "by_primary_evidence_ref": by_primary_evidence_ref,
        "by_context_evidence_ref": by_context_evidence_ref,
    }


def build_candidate_lookup(graph_candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_packet_id: dict[str, set[str]] = {}
    by_evidence_ref: dict[str, set[str]] = {}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in graph_candidate_rows:
        cid = candidate_id(row)
        if not cid:
            continue
        rows_by_id[cid] = row
        for pid in list_refs(row, "source_packet_ids", "source_packet_id"):
            by_packet_id.setdefault(pid, set()).add(cid)
        for ref in list_refs(row, "evidence_refs", "evidence_ref"):
            by_evidence_ref.setdefault(ref, set()).add(cid)
    return {"by_packet_id": by_packet_id, "by_evidence_ref": by_evidence_ref, "rows_by_id": rows_by_id}


def s1_decisions_by_unit(s1_delta_decisions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {normalize(row.get("new_s1_unit_id")): row for row in s1_delta_decisions if normalize(row.get("new_s1_unit_id"))}


def graph_delta_signal(s1_delta_type: str, has_existing_candidates: bool) -> tuple[str, str, str]:
    if s1_delta_type in NO_REFRESH_S1_DELTAS:
        return "no_graph_change", "no_graph_refresh_expected", "active_graph_unchanged"
    if s1_delta_type == "new_unit":
        return "new_edge_candidate", "route_graph_extraction_for_new_packet", "active_graph_add_candidate"
    if s1_delta_type == "duplicate_candidate":
        return "needs_review", "review_existing_graph_candidates_for_duplicate", "review_graph_stale_candidate"
    if s1_delta_type == "strengthens":
        return "new_claim_candidate", "route_graph_extraction_for_strengthening", "active_graph_refresh_candidate"
    if s1_delta_type == "weakens":
        return "edge_weakened", "route_graph_extraction_for_weakening", "active_graph_refresh_candidate"
    if s1_delta_type == "contradicts":
        return "edge_contradicted", "route_graph_extraction_for_contradiction", "active_graph_refresh_candidate"
    if s1_delta_type == "supersedes":
        return "edge_superseded", "mark_existing_graph_candidates_stale", "active_graph_refresh_candidate"
    if s1_delta_type == "needs_review":
        return "needs_review", "review_graph_candidates_after_s1_s2_review", "review_graph_stale_candidate"
    if has_existing_candidates:
        return "needs_review", "review_existing_graph_candidates", "review_graph_stale_candidate"
    return "needs_review", "route_graph_extraction_after_review", "review_graph_stale_candidate"


def candidate_summary(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    kind_counts: dict[str, int] = {}
    ids: list[str] = []
    packet_ids: list[str] = []
    evidence_refs: list[str] = []
    for row in candidate_rows:
        kind = candidate_kind(row)
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        ids.append(candidate_id(row))
        packet_ids.extend(list_refs(row, "source_packet_ids", "source_packet_id"))
        evidence_refs.extend(list_refs(row, "evidence_refs", "evidence_ref"))
    return {
        "candidate_ids": normalize_unique(ids),
        "candidate_kind_counts": dict(sorted(kind_counts.items())),
        "source_packet_ids": normalize_unique(packet_ids),
        "evidence_refs": normalize_unique(evidence_refs),
    }


def collect_candidate_rows(
    *,
    packet_ids: list[str],
    candidate_lookup: dict[str, Any],
) -> list[dict[str, Any]]:
    candidate_ids: set[str] = set()
    for pid in packet_ids:
        candidate_ids.update(candidate_lookup["by_packet_id"].get(pid, set()))
    return [candidate_lookup["rows_by_id"][candidate_id] for candidate_id in sorted(candidate_ids)]


def build_affected_rows_for_s2_row(
    s2_row: dict[str, Any],
    *,
    s1_lookup: dict[str, dict[str, Any]],
    packet_lookup: dict[str, Any],
    candidate_lookup: dict[str, Any],
    generated_at: str,
) -> list[dict[str, Any]]:
    s1_unit_id = normalize(s2_row.get("new_s1_unit_id"))
    s1_decision = s1_lookup.get(s1_unit_id, {})
    s1_delta_type = normalize(s2_row.get("s1_delta_type") or s1_decision.get("delta_type"))
    evidence_refs = normalize_unique(
        [
            *list_refs(s2_row, "new_evidence_refs", "matched_evidence_refs"),
            *list_refs(s1_decision, "new_evidence_refs"),
        ]
    )
    primary_packet_ids = sorted(
        {
            pid
            for evidence_ref in evidence_refs
            for pid in packet_lookup["by_primary_evidence_ref"].get(evidence_ref, set())
        }
    )
    context_packet_ids = sorted(
        {
            pid
            for evidence_ref in evidence_refs
            for pid in packet_lookup["by_context_evidence_ref"].get(evidence_ref, set())
        }
    )
    packet_ids = sorted(set(primary_packet_ids) | set(context_packet_ids))
    candidate_rows = collect_candidate_rows(packet_ids=packet_ids, candidate_lookup=candidate_lookup)
    candidate_info = candidate_summary(candidate_rows)
    graph_signal, action, projection_policy = graph_delta_signal(s1_delta_type, bool(candidate_rows))

    if not packet_ids and not candidate_rows:
        match_type = "no_existing_graph_packet_for_new_or_changed_s1"
    elif primary_packet_ids and candidate_rows:
        match_type = "primary_packet_and_candidate_overlap"
    elif context_packet_ids and candidate_rows:
        match_type = "context_packet_and_candidate_overlap"
    elif primary_packet_ids:
        match_type = "primary_packet_overlap_without_materialized_candidate"
    elif context_packet_ids:
        match_type = "context_packet_overlap_without_materialized_candidate"
    else:
        match_type = "candidate_overlap_without_packet"

    return [
        {
            "schema_version": SCHEMA_VERSION,
            "operation_id": normalize(s2_row.get("operation_id") or s1_decision.get("operation_id")),
            "new_s1_unit_id": s1_unit_id,
            "s1_delta_type": s1_delta_type,
            "s2_delta_signal": normalize(s2_row.get("s2_delta_signal")),
            "affected_s2_unit_id": normalize(s2_row.get("affected_s2_unit_id")),
            "new_subject_id": normalize(s2_row.get("new_subject_id") or s1_decision.get("new_subject_id")),
            "evidence_refs": evidence_refs,
            "affected_graph_packet_ids": packet_ids,
            "primary_graph_packet_ids": primary_packet_ids,
            "context_graph_packet_ids": context_packet_ids,
            "affected_graph_candidate_ids": candidate_info["candidate_ids"],
            "candidate_kind_counts": candidate_info["candidate_kind_counts"],
            "candidate_source_packet_ids": candidate_info["source_packet_ids"],
            "candidate_evidence_refs": candidate_info["evidence_refs"],
            "match_type": match_type,
            "graph_delta_signal": graph_signal,
            "recommended_action": action,
            "graph_projection_latest_view_policy": projection_policy,
            "query_invalidation_hint": "invalidate_graph_query_assets" if graph_signal != "no_graph_change" else "no_query_invalidation_expected",
            "refresh_scope_hint": (
                "primary_packets_only"
                if graph_signal != "no_graph_change" and primary_packet_ids
                else "context_packets_review_only"
                if graph_signal != "no_graph_change" and context_packet_ids
                else "no_refresh"
            ),
            "visual_audit_hint": "show_stale_or_new_candidates_for_review" if graph_signal != "no_graph_change" else "no_visual_refresh_expected",
            "entity_merge_separate": True,
            "graph_is_not_proof": True,
            "read_only": True,
            "write_permission": False,
            "generated_at": generated_at,
        }
    ]


def build_graph_affected_packet_report(
    *,
    s1_delta_decisions: list[dict[str, Any]],
    s2_affected_rows: list[dict[str, Any]],
    graph_packet_rows: list[dict[str, Any]],
    graph_candidate_rows: list[dict[str, Any]],
    generated_at: str | None = None,
) -> dict[str, Any]:
    resolved_generated_at = generated_at or now_iso()
    packet_lookup = build_packet_lookup(graph_packet_rows)
    candidate_lookup = build_candidate_lookup(graph_candidate_rows)
    s1_lookup = s1_decisions_by_unit(s1_delta_decisions)
    affected_rows: list[dict[str, Any]] = []
    for s2_row in s2_affected_rows:
        affected_rows.extend(
            build_affected_rows_for_s2_row(
                s2_row,
                s1_lookup=s1_lookup,
                packet_lookup=packet_lookup,
                candidate_lookup=candidate_lookup,
                generated_at=resolved_generated_at,
            )
        )

    packet_ids = sorted({pid for row in affected_rows for pid in row["affected_graph_packet_ids"]})
    candidate_ids = sorted({cid for row in affected_rows for cid in row["affected_graph_candidate_ids"]})
    action_counts: dict[str, int] = {}
    signal_counts: dict[str, int] = {}
    match_counts: dict[str, int] = {}
    for row in affected_rows:
        action_counts[row["recommended_action"]] = action_counts.get(row["recommended_action"], 0) + 1
        signal_counts[row["graph_delta_signal"]] = signal_counts.get(row["graph_delta_signal"], 0) + 1
        match_counts[row["match_type"]] = match_counts.get(row["match_type"], 0) + 1

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "s1_delta_decision_count": len(s1_delta_decisions),
        "s2_affected_row_count": len(s2_affected_rows),
        "graph_packet_count": len(graph_packet_rows),
        "graph_candidate_count": len(graph_candidate_rows),
        "affected_row_count": len(affected_rows),
        "affected_graph_packet_ids": packet_ids,
        "affected_graph_candidate_ids": candidate_ids,
        "recommended_action_counts": dict(sorted(action_counts.items())),
        "graph_delta_signal_counts": dict(sorted(signal_counts.items())),
        "match_type_counts": dict(sorted(match_counts.items())),
        "graph_projection_latest_view_policy": "active graph excludes stale candidates; historical/review/audit graph may retain them with status warnings.",
        "graph_query_invalidation_policy": "Only graph-aware query assets touching affected packets/candidates should be refreshed; lexical/S1/S2 branches are separate.",
        "candidate_resolution_policy": "Graph candidates are resolved through source_packet_ids; consolidated node evidence_refs are not used as a broad candidate join key.",
        "boundary": "Graph affected-packet rows are maintenance signals, not graph truth or support proof.",
        "entity_merge_separate": True,
        "graph_is_not_proof": True,
        "read_only": True,
        "write_permission": False,
        "generated_at": resolved_generated_at,
    }
    return {"affected_rows": affected_rows, "report": report}


def read_many_jsonl(paths: list[Path | None]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(read_jsonl(path))
    return rows


def write_report_bundle(output_dir: Path, bundle: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "graph_affected_packets.jsonl"
    report_path = output_dir / "graph_affected_packet_report.json"
    write_jsonl(rows_path, bundle["affected_rows"])
    write_json(report_path, bundle["report"])
    return {"affected_rows_path": str(rows_path), "report_path": str(report_path)}


def build_graph_affected_packet_report_from_files(
    *,
    s1_delta_decisions_jsonl: Path,
    s2_affected_rows_jsonl: Path,
    graph_packets_jsonl: Path,
    output_dir: Path,
    graph_nodes_jsonl: Path | None = None,
    graph_edges_jsonl: Path | None = None,
    graph_claims_jsonl: Path | None = None,
    graph_evidence_links_jsonl: Path | None = None,
) -> dict[str, Any]:
    bundle = build_graph_affected_packet_report(
        s1_delta_decisions=read_jsonl(s1_delta_decisions_jsonl),
        s2_affected_rows=read_jsonl(s2_affected_rows_jsonl),
        graph_packet_rows=read_jsonl(graph_packets_jsonl),
        graph_candidate_rows=read_many_jsonl(
            [graph_nodes_jsonl, graph_edges_jsonl, graph_claims_jsonl, graph_evidence_links_jsonl]
        ),
    )
    bundle["paths"] = write_report_bundle(output_dir, bundle)
    return bundle


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s1-delta-decisions-jsonl", required=True, type=Path)
    parser.add_argument("--s2-affected-rows-jsonl", required=True, type=Path)
    parser.add_argument("--graph-packets-jsonl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--graph-nodes-jsonl", type=Path)
    parser.add_argument("--graph-edges-jsonl", type=Path)
    parser.add_argument("--graph-claims-jsonl", type=Path)
    parser.add_argument("--graph-evidence-links-jsonl", type=Path)
    args = parser.parse_args()

    bundle = build_graph_affected_packet_report_from_files(
        s1_delta_decisions_jsonl=args.s1_delta_decisions_jsonl,
        s2_affected_rows_jsonl=args.s2_affected_rows_jsonl,
        graph_packets_jsonl=args.graph_packets_jsonl,
        graph_nodes_jsonl=args.graph_nodes_jsonl,
        graph_edges_jsonl=args.graph_edges_jsonl,
        graph_claims_jsonl=args.graph_claims_jsonl,
        graph_evidence_links_jsonl=args.graph_evidence_links_jsonl,
        output_dir=args.output_dir,
    )
    print(json.dumps(bundle["report"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
