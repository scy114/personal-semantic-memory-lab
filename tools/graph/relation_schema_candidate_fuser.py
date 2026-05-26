"""Fuse lexical and embedding relation schema candidates.

The fuser is intentionally small: it uses Reciprocal Rank Fusion so token
control does not turn into another hand-tuned ranking project.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.graph.graph_construction_packet_builder import file_hash, read_jsonl, write_json, write_jsonl, write_text
from tools.graph.relation_schema_candidate_retriever import normalize_text, string_list, unique_strings


SCHEMA_VERSION = "graph_v03.relation_schema_candidate_fusion.v0.1"
ROW_SCHEMA_VERSION = "graph_v03.packet_relation_schema_candidates.fused.v0.1"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_relation_schema_candidates_fused"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_relation_type(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_")


def row_by_packet(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    for row in read_jsonl(path):
        packet_id = str(row.get("packet_id") or "")
        if packet_id:
            rows[packet_id] = row
    return rows


def canonical_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    if not row:
        return []
    candidates = row.get("relation_schema_candidates") or row.get("canonical_relation_schema_candidates") or []
    out: list[dict[str, Any]] = []
    for candidate in candidates:
        if candidate.get("doc_kind") and candidate.get("doc_kind") != "canonical_relation_type":
            continue
        relation_type = normalize_relation_type(candidate.get("relation_type"))
        if not relation_type or relation_type.startswith("schema_gap_hint_") or relation_type.startswith("zh_relation_hint_"):
            continue
        copied = dict(candidate)
        copied["relation_type"] = relation_type
        out.append(copied)
    return out


def merge_candidate_payload(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key in ("aliases", "external_sources", "examples_or_glosses", "warnings"):
        merged[key] = unique_strings(merged.get(key), incoming.get(key), limit=40)
    for key in ("category", "source_count", "doc_kind"):
        if not merged.get(key) and incoming.get(key):
            merged[key] = incoming.get(key)
    return merged


def fuse_packet_candidates(
    *,
    packet_id: str,
    lexical_row: dict[str, Any],
    embedding_row: dict[str, Any],
    max_candidates: int,
    rrf_k: int,
    include_lexical_only: bool,
) -> dict[str, Any]:
    candidates_by_type: dict[str, dict[str, Any]] = {}
    fusion_meta: dict[str, dict[str, Any]] = defaultdict(lambda: {"source_ranks": {}, "source_scores": {}, "rrf_score": 0.0})
    embedding_relation_types = {
        normalize_relation_type(candidate.get("relation_type"))
        for candidate in canonical_candidates(embedding_row)
    }

    for source_name, row in (("lexical", lexical_row), ("embedding", embedding_row)):
        for fallback_rank, candidate in enumerate(canonical_candidates(row), 1):
            relation_type = normalize_relation_type(candidate.get("relation_type"))
            if not relation_type:
                continue
            if source_name == "lexical" and not include_lexical_only and relation_type not in embedding_relation_types:
                continue
            rank = int(candidate.get("rank") or fallback_rank)
            try:
                source_score = float(candidate.get("score") or 0.0)
            except (TypeError, ValueError):
                source_score = 0.0
            if relation_type in candidates_by_type:
                candidates_by_type[relation_type] = merge_candidate_payload(candidates_by_type[relation_type], candidate)
            else:
                candidates_by_type[relation_type] = dict(candidate)
            fusion_meta[relation_type]["source_ranks"][source_name] = rank
            fusion_meta[relation_type]["source_scores"][source_name] = round(source_score, 6)
            fusion_meta[relation_type]["rrf_score"] += 1.0 / (rrf_k + rank)

    ranked: list[dict[str, Any]] = []
    for relation_type, candidate in candidates_by_type.items():
        meta = fusion_meta[relation_type]
        source_ranks = meta["source_ranks"]
        candidate = dict(candidate)
        candidate["score"] = round(float(meta["rrf_score"]), 8)
        candidate["retrieval_features"] = {
            "fusion_method": "reciprocal_rank_fusion",
            "rrf_k": rrf_k,
            "fusion_sources": sorted(source_ranks),
            "source_ranks": source_ranks,
            "source_scores": meta["source_scores"],
            "source_count": len(source_ranks),
        }
        ranked.append(candidate)

    ranked.sort(
        key=lambda candidate: (
            -float(candidate.get("score") or 0.0),
            -len((candidate.get("retrieval_features") or {}).get("fusion_sources") or []),
            str(candidate.get("relation_type") or ""),
        )
    )
    ranked = ranked[: max(0, max_candidates)]
    for rank, candidate in enumerate(ranked, 1):
        candidate["rank"] = rank

    text_preview = normalize_text(
        embedding_row.get("text_preview")
        or lexical_row.get("text_preview")
        or ""
    )[:300]
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "packet_id": packet_id,
        "workspace_id": embedding_row.get("workspace_id") or lexical_row.get("workspace_id"),
        "input_ref": embedding_row.get("input_ref") or lexical_row.get("input_ref"),
        "text_preview": text_preview,
        "relation_schema_candidates": ranked,
        "graph_is_not_proof": True,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "packet_id",
        "rank",
        "relation_type",
        "score",
        "fusion_sources",
        "source_ranks",
        "category",
        "external_sources",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_report(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Relation Schema Candidate Fusion Report",
        "",
        f"- workspace: `{manifest['workspace']}`",
        f"- packets: {manifest['counts']['packet_count']}",
        f"- max_candidates_per_packet: {manifest['max_candidates']}",
        f"- method: `reciprocal_rank_fusion`",
        f"- rrf_k: {manifest['rrf_k']}",
        f"- include_lexical_only: `{str(manifest['include_lexical_only']).lower()}`",
        "- graph_is_not_proof: `true`",
        "",
        "## Boundary",
        "",
        "- Read-only schema candidate fusion.",
        "- No provider calls.",
        "- No graph truth.",
        "- Only canonical relation candidates are emitted for provider prompts.",
        "- By default, lexical candidates only boost Qwen embedding candidates that are already present.",
        "",
        "## Samples",
        "",
    ]
    for row in rows[:8]:
        lines.extend(
            [
                f"### {row['packet_id']}",
                "",
                "- candidates: "
                + ", ".join(
                    f"{candidate['relation_type']}({candidate['rank']})"
                    for candidate in row.get("relation_schema_candidates", [])[:12]
                ),
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run_fuser(
    *,
    workspace: Path,
    lexical_candidates: Path,
    embedding_candidates: Path,
    output_dir: Path,
    max_candidates: int = 20,
    rrf_k: int = 60,
    include_lexical_only: bool = False,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    lexical_candidates = lexical_candidates.resolve()
    embedding_candidates = embedding_candidates.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    lexical = row_by_packet(lexical_candidates)
    embedding = row_by_packet(embedding_candidates)
    packet_ids = sorted(set(lexical) | set(embedding))
    rows = [
        fuse_packet_candidates(
            packet_id=packet_id,
            lexical_row=lexical.get(packet_id, {}),
            embedding_row=embedding.get(packet_id, {}),
            max_candidates=max_candidates,
            rrf_k=rrf_k,
            include_lexical_only=include_lexical_only,
        )
        for packet_id in packet_ids
    ]

    csv_rows: list[dict[str, Any]] = []
    for row in rows:
        for candidate in row.get("relation_schema_candidates") or []:
            features = candidate.get("retrieval_features") or {}
            csv_rows.append(
                {
                    "packet_id": row["packet_id"],
                    "rank": candidate.get("rank"),
                    "relation_type": candidate.get("relation_type"),
                    "score": candidate.get("score"),
                    "fusion_sources": ";".join(string_list(features.get("fusion_sources"))),
                    "source_ranks": json.dumps(features.get("source_ranks") or {}, ensure_ascii=False, sort_keys=True),
                    "category": candidate.get("category"),
                    "external_sources": ";".join(string_list(candidate.get("external_sources"))),
                }
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "lexical_candidates": str(lexical_candidates),
        "lexical_candidates_hash": file_hash(lexical_candidates),
        "embedding_candidates": str(embedding_candidates),
        "embedding_candidates_hash": file_hash(embedding_candidates),
        "output_dir": str(output_dir),
        "max_candidates": max_candidates,
        "rrf_k": rrf_k,
        "include_lexical_only": include_lexical_only,
        "counts": {
            "packet_count": len(rows),
            "candidate_rows": len(csv_rows),
            "packets_without_candidates": sum(1 for row in rows if not row.get("relation_schema_candidates")),
            "avg_candidates_per_packet": round(len(csv_rows) / max(1, len(rows)), 3),
        },
        "boundary": {
            "provider_calls": False,
            "graph_truth": False,
            "graph_is_not_proof": True,
            "write_permission": False,
        },
        "outputs": {
            "relation_schema_candidates": str(output_dir / "relation_schema_candidates.jsonl"),
            "relation_schema_candidates_csv": str(output_dir / "relation_schema_candidates.csv"),
            "manifest": str(output_dir / "relation_schema_candidate_fusion_manifest.json"),
            "report": str(output_dir / "relation_schema_candidate_fusion_report.md"),
        },
    }
    write_jsonl(output_dir / "relation_schema_candidates.jsonl", rows)
    write_csv(output_dir / "relation_schema_candidates.csv", csv_rows)
    write_json(output_dir / "relation_schema_candidate_fusion_manifest.json", manifest)
    write_text(output_dir / "relation_schema_candidate_fusion_report.md", render_report(manifest, rows))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fuse lexical and embedding relation schema candidates.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--lexical-candidates", required=True)
    parser.add_argument("--embedding-candidates", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-candidates", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--include-lexical-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace)
    output_dir = Path(args.output_dir) if args.output_dir else workspace / DEFAULT_OUTPUT_DIR_NAME
    manifest = run_fuser(
        workspace=workspace,
        lexical_candidates=Path(args.lexical_candidates),
        embedding_candidates=Path(args.embedding_candidates),
        output_dir=output_dir,
        max_candidates=max(1, int(args.max_candidates or 20)),
        rrf_k=max(1, int(args.rrf_k or 60)),
        include_lexical_only=bool(args.include_lexical_only),
    )
    print(json.dumps({"manifest": manifest["outputs"]["manifest"], "counts": manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
