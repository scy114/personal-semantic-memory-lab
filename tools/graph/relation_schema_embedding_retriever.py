"""Retrieve relation schema candidates with a local embedding index.

This is a read-only helper for v0.3 graph extraction. It uses the local
relation schema embedding index to find packet-specific relation candidates
before provider prompting.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.graph.graph_construction_packet_builder import read_json, read_jsonl, write_json, write_jsonl, write_text
from tools.graph.relation_schema_candidate_retriever import normalize_text, packet_text, string_list
from tools.graph.relation_schema_embedding_index_builder import DEFAULT_MODEL, encode_texts


SCHEMA_VERSION = "graph_v03.relation_schema_embedding_retrieval.v0.1"
ROW_SCHEMA_VERSION = "graph_v03.packet_relation_schema_embedding_candidates.v0.1"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_relation_schema_embedding_candidates"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl_local(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def packet_embedding_query_text(packet: dict[str, Any]) -> str:
    # Graph relation retrieval benefits from local context because relation
    # endpoints and attribution are often outside the primary sentence.
    parts = [
        packet.get("graph_extraction_text"),
        packet.get("graph_route_text"),
        packet.get("original_text"),
        packet.get("processed_text"),
    ]
    text = "\n".join(str(part or "") for part in parts if str(part or "").strip())
    return text or packet_text(packet)


def load_index(index_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    docs_path = index_dir / "relation_schema_embedding_docs.jsonl"
    matrix_path = index_dir / "relation_schema_embedding_matrix.npy"
    manifest_path = index_dir / "relation_schema_embedding_manifest.json"
    docs = read_jsonl_local(docs_path)
    matrix = np.load(matrix_path)
    manifest = read_json(manifest_path)
    if matrix.ndim != 2 or matrix.shape[0] != len(docs):
        raise ValueError(f"Embedding index shape mismatch: docs={len(docs)} matrix_shape={matrix.shape}")
    if matrix.shape[1] <= 0:
        raise ValueError("Embedding index has no vector dimension; rebuild without --skip-embeddings.")
    return docs, matrix.astype("float32"), manifest


def candidate_row(doc: dict[str, Any], score: float, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "relation_type": doc.get("relation_type"),
        "doc_kind": doc.get("doc_kind"),
        "score": round(float(score), 6),
        "category": doc.get("category"),
        "external_sources": string_list(doc.get("external_sources")),
        "aliases": string_list(doc.get("aliases"))[:16],
        "examples_or_glosses": string_list(doc.get("examples_or_glosses"))[:6],
        "warnings": string_list(doc.get("warnings")),
    }


def retrieve_candidates(
    query_vector: np.ndarray,
    docs: list[dict[str, Any]],
    matrix: np.ndarray,
    *,
    top_k: int,
    canonical_top_k: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scores = matrix @ query_vector.astype("float32")
    order = np.argsort(-scores)
    all_candidates: list[dict[str, Any]] = []
    canonical_candidates: list[dict[str, Any]] = []
    for doc_index in order:
        doc = docs[int(doc_index)]
        score = float(scores[int(doc_index)])
        if len(all_candidates) < top_k:
            all_candidates.append(candidate_row(doc, score, len(all_candidates) + 1))
        if doc.get("doc_kind") == "canonical_relation_type" and len(canonical_candidates) < canonical_top_k:
            canonical_candidates.append(candidate_row(doc, score, len(canonical_candidates) + 1))
        if len(all_candidates) >= top_k and len(canonical_candidates) >= canonical_top_k:
            break
    return all_candidates, canonical_candidates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "packet_id",
        "candidate_group",
        "rank",
        "relation_type",
        "doc_kind",
        "score",
        "category",
        "external_sources",
        "aliases",
        "warnings",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_report(manifest: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Relation Schema Embedding Retrieval Report",
        "",
        f"- workspace: `{manifest['workspace']}`",
        f"- packets: {manifest['counts']['packet_count']}",
        f"- index_docs: {manifest['counts']['index_doc_count']}",
        f"- vector_dim: {manifest['counts']['vector_dim']}",
        f"- top_k_all: {manifest['top_k_all']}",
        f"- top_k_canonical: {manifest['top_k_canonical']}",
        f"- embedding_model: `{manifest['embedding_model']}`",
        "- graph_is_not_proof: `true`",
        "",
        "## Boundary",
        "",
        "- Read-only semantic schema retrieval.",
        "- No provider calls.",
        "- No graph truth.",
        "- Supplemental hint docs are diagnostics; provider prompts should use canonical candidates unless explicitly reviewed.",
        "",
        "## Samples",
        "",
    ]
    for sample in samples[:8]:
        lines.extend(
            [
                f"### {sample['packet_id']}",
                "",
                f"- text_preview: {sample.get('text_preview', '')}",
                "- canonical_candidates: "
                + ", ".join(
                    f"{candidate['relation_type']}({candidate['score']})"
                    for candidate in sample.get("canonical_relation_schema_candidates", [])[:10]
                ),
                "- all_candidates: "
                + ", ".join(
                    f"{candidate['relation_type']}[{candidate['doc_kind']}]({candidate['score']})"
                    for candidate in sample.get("relation_schema_embedding_candidates", [])[:10]
                ),
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run_embedding_retriever(
    *,
    workspace: Path,
    input_packets: Path,
    index_dir: Path,
    output_dir: Path,
    top_k: int,
    canonical_top_k: int,
    embedding_model: str | None = None,
    batch_size: int = 8,
    max_length: int = 256,
    max_items: int | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    input_packets = input_packets.resolve()
    index_dir = index_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    docs, matrix, index_manifest = load_index(index_dir)
    packets = read_jsonl(input_packets)
    if max_items is not None:
        packets = packets[: max(0, max_items)]
    model_name = embedding_model or str(index_manifest.get("embedding_model") or DEFAULT_MODEL)
    query_texts = [packet_embedding_query_text(packet) for packet in packets]
    query_matrix = encode_texts(query_texts, model_name=model_name, batch_size=batch_size, max_length=max_length)

    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for packet, query_text, query_vector in zip(packets, query_texts, query_matrix):
        all_candidates, canonical_candidates = retrieve_candidates(
            query_vector,
            docs,
            matrix,
            top_k=top_k,
            canonical_top_k=canonical_top_k,
        )
        row = {
            "schema_version": ROW_SCHEMA_VERSION,
            "packet_id": str(packet.get("packet_id") or ""),
            "workspace_id": packet.get("workspace_id"),
            "input_ref": packet.get("input_ref"),
            "text_preview": normalize_text(query_text)[:300],
            "relation_schema_embedding_candidates": all_candidates,
            "canonical_relation_schema_candidates": canonical_candidates,
            "relation_schema_candidates": canonical_candidates,
            "graph_is_not_proof": True,
        }
        rows.append(row)
        for group, candidates in (
            ("all", all_candidates),
            ("canonical", canonical_candidates),
        ):
            for candidate in candidates:
                csv_rows.append(
                    {
                        "packet_id": row["packet_id"],
                        "candidate_group": group,
                        "rank": candidate["rank"],
                        "relation_type": candidate["relation_type"],
                        "doc_kind": candidate["doc_kind"],
                        "score": candidate["score"],
                        "category": candidate["category"],
                        "external_sources": ";".join(string_list(candidate.get("external_sources"))),
                        "aliases": ";".join(string_list(candidate.get("aliases"))[:8]),
                        "warnings": ";".join(string_list(candidate.get("warnings"))),
                    }
                )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "input_packets": str(input_packets),
        "index_dir": str(index_dir),
        "index_docs_hash": file_sha256(index_dir / "relation_schema_embedding_docs.jsonl"),
        "index_matrix_hash": file_sha256(index_dir / "relation_schema_embedding_matrix.npy"),
        "output_dir": str(output_dir),
        "embedding_model": model_name,
        "embedding_backend": "transformers_mean_pooling",
        "local_files_only": True,
        "top_k_all": top_k,
        "top_k_canonical": canonical_top_k,
        "batch_size": batch_size,
        "max_length": max_length,
        "counts": {
            "packet_count": len(packets),
            "index_doc_count": len(docs),
            "vector_dim": int(matrix.shape[1]),
            "candidate_rows": len(csv_rows),
        },
        "boundary": {
            "provider_calls": False,
            "graph_truth": False,
            "graph_is_not_proof": True,
            "write_permission": False,
        },
        "outputs": {
            "relation_schema_embedding_candidates": str(output_dir / "relation_schema_embedding_candidates.jsonl"),
            "relation_schema_embedding_candidates_csv": str(output_dir / "relation_schema_embedding_candidates.csv"),
            "manifest": str(output_dir / "relation_schema_embedding_retrieval_manifest.json"),
            "report": str(output_dir / "relation_schema_embedding_retrieval_report.md"),
        },
    }
    write_jsonl(output_dir / "relation_schema_embedding_candidates.jsonl", rows)
    write_csv(output_dir / "relation_schema_embedding_candidates.csv", csv_rows)
    write_json(output_dir / "relation_schema_embedding_retrieval_manifest.json", manifest)
    write_text(output_dir / "relation_schema_embedding_retrieval_report.md", render_report(manifest, rows))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve relation schema candidates with local embeddings.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--input-packets", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--canonical-top-k", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-items", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace)
    output_dir = Path(args.output_dir) if args.output_dir else workspace / DEFAULT_OUTPUT_DIR_NAME
    manifest = run_embedding_retriever(
        workspace=workspace,
        input_packets=Path(args.input_packets),
        index_dir=Path(args.index_dir),
        output_dir=output_dir,
        embedding_model=args.embedding_model,
        top_k=max(1, int(args.top_k or 40)),
        canonical_top_k=max(1, int(args.canonical_top_k or 40)),
        batch_size=max(1, int(args.batch_size or 8)),
        max_length=max(16, int(args.max_length or 256)),
        max_items=args.max_items,
    )
    print(json.dumps({"manifest": manifest["outputs"]["manifest"], "counts": manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
