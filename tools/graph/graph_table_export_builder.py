"""Export v0.2 graph assets into v0.3 audit graph tables.

Slice 2 only exports and validates graph tables. It does not compute PageRank,
centrality, community detection, node similarity, or graph utility
interpretation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "graph_v03.table_export.v0.1"
NODE_SCHEMA_VERSION = "graph_v03.node_table.v0.1"
EDGE_SCHEMA_VERSION = "graph_v03.edge_table.v0.1"
EVIDENCE_LINK_SCHEMA_VERSION = "graph_v03.evidence_link.v0.1"
GENERIC_RELATION_TYPES = {"related_to", "mentions", "about", "associated_with", "has_observation"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def short_hash(value: str, length: int = 12) -> str:
    return sha256_text(value)[:length]


def file_hash(path: Path) -> str | None:
    return sha256_bytes(path.read_bytes()) if path.exists() else None


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def stable_id(prefix: str, value: str) -> str:
    return f"{prefix}:{short_hash(value)}"


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def string_list(value: Any) -> list[str]:
    result: list[str] = []
    for item in as_list(value):
        if item is None:
            continue
        if isinstance(item, dict):
            result.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
        else:
            text = str(item)
            if text:
                result.append(text)
    return result


def unique_strings(*values: Any) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for item in string_list(value):
            if item not in seen:
                seen.add(item)
                out.append(item)
    return out


def first_non_empty(*values: Any, default: str = "") -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return default


def load_modeled_user(workspace: Path) -> str:
    manifest = workspace / "manifest.yaml"
    if not manifest.exists():
        return workspace.name
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("modeled_user_id:") or line.startswith("target_participant:"):
            return line.split(":", 1)[1].strip() or workspace.name
    return workspace.name


def source_unit_ids_from_refs(*values: Any) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for ref in unique_strings(*values):
        candidate = ref
        if "text_span:" in ref:
            candidate = ref.split(":", 1)[-1] if ref.startswith("raw:") else ref
        if candidate and candidate not in seen:
            seen.add(candidate)
            ids.append(candidate)
    return ids


def raw_backpointer_refs(*values: Any) -> list[Any]:
    refs: list[Any] = []
    for value in values:
        for item in as_list(value):
            if item not in refs:
                refs.append(item)
    return refs


def load_proposal_index(workspace: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for path in (workspace / "proposals").glob("**/proposal_outcomes.ai.jsonl"):
        for row in read_jsonl(path):
            proposal_id = row.get("proposal_id")
            if proposal_id and proposal_id not in index:
                index[str(proposal_id)] = row
    return index


def load_evidence_index(workspace: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(workspace / "evidence" / "evidence.jsonl"):
        for key in [
            row.get("evidence_ref"),
            row.get("canonical_evidence_ref"),
            row.get("source_specific_ref"),
            row.get("text_unit_id"),
        ]:
            if key:
                index[str(key)] = row
        for alias in row.get("ref_aliases") or []:
            index[str(alias)] = row
    return index


def node_row(
    *,
    source: dict[str, Any],
    human_readable_id: int,
    workspace_id: str,
    modeled_user_id: str,
    proposal_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    node_id = str(source.get("node_id") or stable_id("node", json.dumps(source, sort_keys=True)))
    evidence_refs = unique_strings(source.get("evidence_refs"))
    source_refs = unique_strings(source.get("source_refs"))
    proposal_origin = source.get("proposal_origin") or {}
    proposal_refs = unique_strings(proposal_origin.get("proposal_id"))
    proposal_rows = [proposal_index[ref] for ref in proposal_refs if ref in proposal_index]
    route_refs = unique_strings(*(row.get("route_decision_id") for row in proposal_rows), *(row.get("route_run_id") for row in proposal_rows))
    backpointers = raw_backpointer_refs(source.get("raw_backpointer_refs"), source.get("backpointer_refs"), *(row.get("raw_backpointer_refs") for row in proposal_rows))
    source_unit_ids = source_unit_ids_from_refs(
        source.get("source_unit_ids"),
        evidence_refs,
        source_refs,
        *(row.get("text_unit_id") for row in proposal_rows),
    )
    warnings = unique_strings(source.get("warnings"), source.get("risk_notes"), *(row.get("warnings") for row in proposal_rows))
    properties = dict(source.get("properties") or {})
    properties.update(
        {
            "legacy_source": "graph/nodes.jsonl",
            "helper_fields_not_evidence_truth": True,
            "degree_metric_computed_in_slice2": False,
        }
    )
    return {
        "schema_version": NODE_SCHEMA_VERSION,
        "node_id": node_id,
        "human_readable_id": human_readable_id,
        "title": first_non_empty(source.get("label"), source.get("title"), node_id),
        "node_type": first_non_empty(source.get("type"), source.get("node_type"), default="unknown"),
        "description": first_non_empty(source.get("description"), source.get("summary")),
        "summary": first_non_empty(source.get("summary"), source.get("description")),
        "source_unit_ids": source_unit_ids,
        "evidence_refs": evidence_refs,
        "source_refs": source_refs,
        "raw_backpointer_refs": backpointers,
        "proposal_refs": proposal_refs,
        "route_refs": route_refs,
        "frequency": None,
        "degree": None,
        "confidence": source.get("confidence"),
        "inference_level": source.get("inference_level"),
        "temporal_scope": source.get("temporal_scope") or {},
        "privacy_class": source.get("privacy_class"),
        "properties": properties,
        "warnings": warnings,
        "graph_is_not_proof": True,
        "workspace_id": workspace_id,
        "modeled_user_id": modeled_user_id,
    }


def edge_row(
    *,
    source: dict[str, Any],
    human_readable_id: int,
    workspace_id: str,
    modeled_user_id: str,
    proposal_index: dict[str, dict[str, Any]],
    node_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    source_node_id = str(source.get("source_node") or source.get("source_node_id") or "")
    target_node_id = str(source.get("target_node") or source.get("target_node_id") or "")
    relation_type = str(source.get("relation_type") or "related_to")
    base_id = source.get("edge_id") or f"{source_node_id}|{relation_type}|{target_node_id}|{human_readable_id}"
    edge_id = str(base_id)
    evidence_refs = unique_strings(source.get("evidence_refs"))
    source_refs = unique_strings(source.get("source_refs"))
    proposal_origin = source.get("proposal_origin") or {}
    proposal_refs = unique_strings(proposal_origin.get("proposal_id"), (source.get("properties") or {}).get("proposal_id"))
    proposal_rows = [proposal_index[ref] for ref in proposal_refs if ref in proposal_index]
    route_refs = unique_strings(*(row.get("route_decision_id") for row in proposal_rows), *(row.get("route_run_id") for row in proposal_rows))
    backpointers = raw_backpointer_refs(source.get("raw_backpointer_refs"), source.get("backpointer_refs"), *(row.get("raw_backpointer_refs") for row in proposal_rows))
    source_unit_ids = source_unit_ids_from_refs(
        source.get("source_unit_ids"),
        evidence_refs,
        source_refs,
        *(row.get("text_unit_id") for row in proposal_rows),
    )
    warnings = unique_strings(source.get("warnings"), source.get("risk_notes"), *(row.get("warnings") for row in proposal_rows))
    evidence_count = len(evidence_refs)
    properties = dict(source.get("properties") or {})
    properties.update(
        {
            "legacy_source": "graph/edges.jsonl",
            "helper_fields_not_evidence_truth": True,
            "algorithm_metrics_computed_in_slice2": False,
            "weight_policy": "default_1_when_unknown; evidence_count and confidence are separate",
        }
    )
    privacy_class = source.get("privacy_class")
    if not privacy_class:
        privacy_class = (node_index.get(source_node_id) or {}).get("privacy_class")
    if not privacy_class:
        privacy_class = (node_index.get(target_node_id) or {}).get("privacy_class")
    if not privacy_class:
        privacy_class = "unknown"
    return {
        "schema_version": EDGE_SCHEMA_VERSION,
        "edge_id": edge_id,
        "human_readable_id": human_readable_id,
        "source_node_id": source_node_id,
        "target_node_id": target_node_id,
        "relation_type": relation_type,
        "description": first_non_empty(source.get("description"), source.get("summary")),
        "summary": first_non_empty(source.get("summary"), source.get("description")),
        "weight": 1.0,
        "weight_policy": "default_1_when_unknown",
        "evidence_count": evidence_count,
        "combined_degree": None,
        "source_unit_ids": source_unit_ids,
        "evidence_refs": evidence_refs,
        "source_refs": source_refs,
        "raw_backpointer_refs": backpointers,
        "proposal_refs": proposal_refs,
        "route_refs": route_refs,
        "temporal_scope": source.get("temporal_scope") or {},
        "confidence": source.get("confidence"),
        "inference_level": source.get("inference_level"),
        "privacy_class": privacy_class,
        "properties": properties,
        "warnings": warnings,
        "graph_is_not_proof": True,
        "workspace_id": workspace_id,
        "modeled_user_id": modeled_user_id,
    }


def evidence_link_rows(
    *,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    evidence_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for kind, items, id_field in [
        ("node", nodes, "node_id"),
        ("edge", edges, "edge_id"),
    ]:
        for item in items:
            item_id = str(item.get(id_field) or "")
            for evidence_ref in item.get("evidence_refs") or []:
                key = (kind, item_id, evidence_ref)
                if key in seen:
                    continue
                seen.add(key)
                evidence = evidence_index.get(evidence_ref, {})
                row = {
                    "schema_version": EVIDENCE_LINK_SCHEMA_VERSION,
                    "link_id": stable_id("evidence_link", "|".join(key)),
                    "graph_item_kind": kind,
                    "graph_item_id": item_id,
                    "evidence_ref": evidence_ref,
                    "source_unit_id": first_non_empty(evidence.get("text_unit_id"), *(item.get("source_unit_ids") or [])),
                    "raw_backpointer_refs": raw_backpointer_refs(item.get("raw_backpointer_refs"), evidence.get("locator")),
                    "quote": first_non_empty(evidence.get("text"), evidence.get("display_ref"), item.get("summary")),
                    "processed_text": "",
                    "original_text": first_non_empty(evidence.get("text")),
                    "support_role": "source_evidence",
                    "warnings": unique_strings(item.get("warnings")),
                    "graph_is_not_proof": True,
                }
                rows.append(row)
    return rows


def source_asset_hashes(workspace: Path, extra_paths: list[Path]) -> dict[str, str | None]:
    paths = [
        workspace / "graph" / "nodes.jsonl",
        workspace / "graph" / "edges.jsonl",
        workspace / "portrait" / "reviewed_units.jsonl",
        workspace / "memory" / "memory_units.jsonl",
        workspace / "evidence" / "evidence.jsonl",
        workspace / "manifest.yaml",
        *extra_paths,
    ]
    hashes: dict[str, str | None] = {}
    for path in paths:
        try:
            rel = str(path.relative_to(workspace))
        except ValueError:
            rel = str(path)
        hashes[rel] = file_hash(path)
    return hashes


def preflight_counts(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], source_hashes: dict[str, str | None]) -> dict[str, Any]:
    node_ids = {str(node.get("node_id")) for node in nodes}
    relation_counts = Counter(str(edge.get("relation_type") or "unknown") for edge in edges)
    missing_evidence_count = sum(1 for edge in edges if not edge.get("evidence_refs") and not edge.get("warnings"))
    unresolved_endpoint_count = sum(
        1
        for edge in edges
        if edge.get("source_node_id") not in node_ids or edge.get("target_node_id") not in node_ids
    )
    incident: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()
    self_loop_count = 0
    for edge in edges:
        source = str(edge.get("source_node_id") or "")
        target = str(edge.get("target_node_id") or "")
        incident[source] += 1
        incident[target] += 1
        pair_counts[(source, target)] += 1
        if source == target:
            self_loop_count += 1
    isolated_node_count = sum(1 for node_id in node_ids if incident[node_id] == 0)
    multi_edge_count = sum(count - 1 for count in pair_counts.values() if count > 1)
    generic_count = sum(count for relation, count in relation_counts.items() if relation in GENERIC_RELATION_TYPES)
    generic_ratio = (generic_count / len(edges)) if edges else 0.0
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "relation_type_counts": dict(sorted(relation_counts.items())),
        "missing_evidence_count": missing_evidence_count,
        "unresolved_endpoint_count": unresolved_endpoint_count,
        "isolated_node_count": isolated_node_count,
        "self_loop_count": self_loop_count,
        "multi_edge_count": multi_edge_count,
        "generic_relation_count": generic_count,
        "generic_relation_ratio": round(generic_ratio, 6),
        "source_asset_hashes": source_hashes,
    }


def validate_export(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    warnings: list[str] = []
    node_ids = {str(node.get("node_id")) for node in nodes}
    for node in nodes:
        if node.get("degree") is not None:
            warnings.append(f"node_degree_present:{node.get('node_id')}")
        if node.get("graph_is_not_proof") is not True:
            warnings.append(f"node_graph_is_not_proof_missing:{node.get('node_id')}")
    for edge in edges:
        edge_id = edge.get("edge_id")
        if edge.get("source_node_id") not in node_ids or edge.get("target_node_id") not in node_ids:
            warnings.append(f"edge_unresolved_endpoint:{edge_id}")
        if not edge.get("evidence_refs") and not edge.get("warnings"):
            warnings.append(f"edge_missing_evidence_without_warning:{edge_id}")
        if edge.get("combined_degree") is not None:
            warnings.append(f"edge_combined_degree_present:{edge_id}")
        if edge.get("graph_is_not_proof") is not True:
            warnings.append(f"edge_graph_is_not_proof_missing:{edge_id}")
    return warnings


def build_report(workspace: Path, output_dir: Path, counts: dict[str, Any], validation_warnings: list[str]) -> str:
    relation_lines = [
        f"- {relation}: {count}"
        for relation, count in counts.get("relation_type_counts", {}).items()
    ]
    warning_lines = [f"- {warning}" for warning in validation_warnings] if validation_warnings else ["- none"]
    return "\n".join(
        [
            "# v0.3 Graph Table Export Report",
            "",
            f"- workspace: `{workspace.name}`",
            f"- output_dir: `{output_dir}`",
            "- slice: `v0.3 Slice 2 graph table export`",
            "- graph_is_not_proof: `true`",
            "- algorithm_metrics_computed: `false`",
            "- community_or_centrality_interpretation: `false`",
            "",
            "## Preflight Counts",
            "",
            f"- nodes: {counts['node_count']}",
            f"- edges: {counts['edge_count']}",
            f"- missing_evidence_count: {counts['missing_evidence_count']}",
            f"- unresolved_endpoint_count: {counts['unresolved_endpoint_count']}",
            f"- isolated_node_count: {counts['isolated_node_count']}",
            f"- self_loop_count: {counts['self_loop_count']}",
            f"- multi_edge_count: {counts['multi_edge_count']}",
            f"- generic_relation_ratio: {counts['generic_relation_ratio']}",
            "",
            "## Relation Types",
            "",
            *(relation_lines or ["- none"]),
            "",
            "## Validation Warnings",
            "",
            *warning_lines,
            "",
            "## Notes",
            "",
            "- `description` and `summary` are helper fields only.",
            "- Evidence refs, quotes, and raw backpointers remain the evidence layer.",
            "- Slice 3 owns NetworkX `MultiDiGraph` loading and graph algorithms.",
            "",
        ]
    )


def export_graph_tables(workspace: Path, output_dir: Path | None = None) -> dict[str, Any]:
    workspace = workspace.resolve()
    output_dir = (output_dir or workspace / "graph_v03_prototype").resolve()
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")

    workspace_id = workspace.name
    modeled_user_id = load_modeled_user(workspace)
    proposal_index = load_proposal_index(workspace)
    evidence_index = load_evidence_index(workspace)

    legacy_nodes = read_jsonl(workspace / "graph" / "nodes.jsonl")
    legacy_edges = read_jsonl(workspace / "graph" / "edges.jsonl")
    if not legacy_nodes and not legacy_edges:
        raise FileNotFoundError(f"No v0.2 graph assets found under {workspace / 'graph'}")

    nodes = [
        node_row(
            source=row,
            human_readable_id=index,
            workspace_id=workspace_id,
            modeled_user_id=modeled_user_id,
            proposal_index=proposal_index,
        )
        for index, row in enumerate(legacy_nodes, 1)
    ]
    legacy_node_index = {str(node.get("node_id")): node for node in legacy_nodes}
    edges = [
        edge_row(
            source=row,
            human_readable_id=index,
            workspace_id=workspace_id,
            modeled_user_id=modeled_user_id,
            proposal_index=proposal_index,
            node_index=legacy_node_index,
        )
        for index, row in enumerate(legacy_edges, 1)
    ]

    evidence_links = evidence_link_rows(nodes=nodes, edges=edges, evidence_index=evidence_index)
    source_hashes = source_asset_hashes(workspace, [])
    counts = preflight_counts(nodes, edges, source_hashes)
    validation_warnings = validate_export(nodes, edges)

    output_dir.mkdir(parents=True, exist_ok=True)
    nodes_path = output_dir / "graph_nodes_table.jsonl"
    edges_path = output_dir / "graph_edges_table.jsonl"
    evidence_links_path = output_dir / "evidence_links.jsonl"
    manifest_path = output_dir / "graph_manifest.json"
    report_path = output_dir / "graph_table_export_report.md"

    write_jsonl(nodes_path, nodes)
    write_jsonl(edges_path, edges)
    write_jsonl(evidence_links_path, evidence_links)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "workspace_id": workspace_id,
        "modeled_user_id": modeled_user_id,
        "source_workspace": str(workspace),
        "output_dir": str(output_dir),
        "source_assets": {
            "legacy_graph_nodes": "graph/nodes.jsonl",
            "legacy_graph_edges": "graph/edges.jsonl",
            "portrait_units": "portrait/reviewed_units.jsonl",
            "evidence": "evidence/evidence.jsonl",
        },
        "outputs": {
            "graph_nodes_table": str(nodes_path),
            "graph_edges_table": str(edges_path),
            "evidence_links": str(evidence_links_path),
            "report": str(report_path),
        },
        "preflight_counts": counts,
        "validation_warnings": validation_warnings,
        "policies": {
            "slice": "v0.3 Slice 2",
            "graph_is_not_proof": True,
            "description_summary_are_helper_fields": True,
            "algorithm_metrics_computed": False,
            "community_or_centrality_interpretation": False,
            "edge_weight_policy": "default weight=1.0 when unknown; evidence_count and confidence are separate",
            "slice3_initial_graph_type": "networkx.MultiDiGraph",
        },
    }
    write_json(manifest_path, manifest)
    write_text(report_path, build_report(workspace, output_dir, counts, validation_warnings))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export v0.2 graph assets to v0.3 graph tables.")
    parser.add_argument("--workspace", required=True, type=Path, help="Workspace containing v0.2 graph assets.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Defaults to <workspace>/graph_v03_prototype.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = export_graph_tables(args.workspace, args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
