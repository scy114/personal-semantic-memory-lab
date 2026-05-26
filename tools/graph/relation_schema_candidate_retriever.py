"""Retrieve packet-specific relation schema candidates for graph extraction.

This read-only helper moves external relation schemas closer to the provider
prompt without stuffing the full schema into every model call.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.graph.graph_construction_packet_builder import (
    file_hash,
    read_json,
    read_jsonl,
    sha256_text,
    write_json,
    write_jsonl,
    write_text,
)


try:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as SKLEARN_ENGLISH_STOP_WORDS
except Exception:  # pragma: no cover - sklearn is optional for this lightweight helper.
    SKLEARN_ENGLISH_STOP_WORDS = frozenset()


SCHEMA_VERSION = "graph_v03.relation_schema_candidates.v0.1"
ROW_SCHEMA_VERSION = "graph_v03.packet_relation_schema_candidates.v0.1"
DEFAULT_POLICY = "configs/graph/graph_relation_type_normalization.generated.v0.3.json"
DEFAULT_OUTPUT_DIR_NAME = "graph_v03_relation_schema_candidates"
TOKEN_RE = re.compile(r"[0-9a-zA-Z\u4e00-\u9fff]+")
FALLBACK_ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "was",
    "were",
    "which",
    "who",
    "with",
    "unknown",
}
ENGLISH_STOPWORDS = set(SKLEARN_ENGLISH_STOP_WORDS) | FALLBACK_ENGLISH_STOPWORDS


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def is_cjk_token(token: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in token)


def is_informative_token(token: str) -> bool:
    token = token.lower().strip()
    if not token or token.isdigit():
        return False
    if is_cjk_token(token):
        return len(token) >= 1
    if len(token) < 3:
        return False
    return token not in ENGLISH_STOPWORDS


def tokenize(value: Any) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(str(value or "")) if is_informative_token(token)]


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def string_list(value: Any) -> list[str]:
    out: list[str] = []
    for item in as_list(value):
        if item is None:
            continue
        if isinstance(item, dict):
            text = json.dumps(item, ensure_ascii=False, sort_keys=True)
        else:
            text = str(item).strip()
        if text:
            out.append(text)
    return out


def unique_strings(*values: Any, limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        for item in string_list(value):
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
            if limit is not None and len(out) >= limit:
                return out
    return out


def packet_text(packet: dict[str, Any]) -> str:
    primary_parts = [
        packet.get("original_text"),
        packet.get("processed_text"),
    ]
    primary = "\n".join(str(part or "") for part in primary_parts if str(part or "").strip())
    if primary.strip():
        return primary
    fallback_parts = [
        packet.get("graph_route_text"),
        packet.get("graph_extraction_text"),
    ]
    return "\n".join(str(part or "") for part in fallback_parts if str(part or "").strip())


def relation_aliases(policy: dict[str, Any]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = defaultdict(list)
    alias_map = policy.get("alias_map") or {}
    for alias, canonical in alias_map.items():
        aliases[str(canonical)].append(str(alias))
    return {key: unique_strings(value, limit=80) for key, value in aliases.items()}


def source_count(meta: dict[str, Any]) -> int:
    total = 0
    for count in (meta.get("source_counts") or {}).values():
        try:
            total += int(count)
        except (TypeError, ValueError):
            continue
    return total


def build_schema_docs(policy: dict[str, Any]) -> list[dict[str, Any]]:
    aliases_by_relation = relation_aliases(policy)
    docs: list[dict[str, Any]] = []
    for relation_type, meta in sorted((policy.get("canonical_relation_types") or {}).items()):
        if relation_type == "related_to_generic":
            continue
        aliases = aliases_by_relation.get(relation_type, [])
        source_ids = string_list(meta.get("external_sources"))
        category = str(meta.get("category") or "external_relation_schema")
        count = source_count(meta)
        search_text = " ".join(
            [
                str(relation_type),
                str(relation_type).replace("_", " "),
                " ".join(aliases),
            ]
        )
        docs.append(
            {
                "relation_type": relation_type,
                "aliases": aliases,
                "external_sources": source_ids,
                "category": category,
                "source_count": count,
                "tokens": Counter(tokenize(search_text)),
                "search_text_hash": sha256_text(search_text),
            }
        )
    return docs


def inverse_document_frequency(docs: list[dict[str, Any]]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for doc in docs:
        df.update(set((doc.get("tokens") or {}).keys()))
    total = max(1, len(docs))
    return {token: math.log((1 + total) / (1 + count)) + 1.0 for token, count in df.items()}


def alias_has_informative_tokens(alias: str) -> bool:
    return bool(tokenize(alias.replace("_", " ")))


def exact_alias_hits(text: str, aliases: list[str]) -> list[str]:
    normalized_text = normalize_text(text)
    hits: list[str] = []
    for alias in aliases:
        normalized_alias = normalize_text(alias.replace("_", " "))
        if len(normalized_alias) < 3 or not alias_has_informative_tokens(normalized_alias):
            continue
        if re.search(r"[a-zA-Z]", normalized_alias):
            pattern = r"(?<![0-9a-zA-Z])" + re.escape(normalized_alias) + r"(?![0-9a-zA-Z])"
            matched = re.search(pattern, normalized_text) is not None
        else:
            matched = normalized_alias in normalized_text
        if matched:
            hits.append(alias)
    return unique_strings(hits, limit=10)


def score_doc(query_tokens: Counter[str], text: str, doc: dict[str, Any], idf: dict[str, float]) -> tuple[float, dict[str, Any]]:
    doc_tokens: Counter[str] = doc.get("tokens") or Counter()
    lexical = 0.0
    matched_tokens: list[str] = []
    for token, query_count in query_tokens.items():
        if token not in doc_tokens:
            continue
        matched_tokens.append(token)
        lexical += (1.0 + math.log(query_count)) * (1.0 + math.log(doc_tokens[token])) * idf.get(token, 1.0)

    aliases = string_list(doc.get("aliases"))
    exact_hits = exact_alias_hits(text, [str(doc.get("relation_type") or ""), *aliases])
    exact_boost = 3.5 * len(exact_hits)
    count_prior = min(2.0, math.log1p(int(doc.get("source_count") or 0)) / 5.0)
    multi_source_boost = 0.4 if len(string_list(doc.get("external_sources"))) > 1 else 0.0
    if lexical <= 0.0 and not exact_hits:
        count_prior = 0.0
        multi_source_boost = 0.0
    score = lexical + exact_boost + count_prior + multi_source_boost
    features = {
        "lexical_score": round(lexical, 6),
        "exact_alias_hits": exact_hits,
        "source_count_prior": round(count_prior, 6),
        "multi_source_boost": multi_source_boost,
        "matched_tokens": sorted(matched_tokens)[:20],
    }
    return score, features


def retrieve_for_packet(packet: dict[str, Any], docs: list[dict[str, Any]], idf: dict[str, float], *, top_k: int) -> list[dict[str, Any]]:
    text = packet_text(packet)
    query_tokens = Counter(tokenize(text))
    scored: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for doc in docs:
        score, features = score_doc(query_tokens, text, doc, idf)
        if score <= 0:
            continue
        scored.append((score, doc, features))
    scored.sort(key=lambda item: (-item[0], -int(item[1].get("source_count") or 0), str(item[1].get("relation_type") or "")))
    out: list[dict[str, Any]] = []
    for rank, (score, doc, features) in enumerate(scored[:top_k], 1):
        out.append(
            {
                "rank": rank,
                "relation_type": doc["relation_type"],
                "score": round(score, 6),
                "aliases": string_list(doc.get("aliases"))[:12],
                "external_sources": string_list(doc.get("external_sources")),
                "category": doc.get("category"),
                "source_count": doc.get("source_count"),
                "retrieval_features": features,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "packet_id",
        "rank",
        "relation_type",
        "score",
        "category",
        "source_count",
        "external_sources",
        "exact_alias_hits",
        "matched_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def render_report(manifest: dict[str, Any], samples: list[dict[str, Any]]) -> str:
    lines = [
        "# Relation Schema Candidate Retrieval Report",
        "",
        f"- workspace: `{manifest['workspace']}`",
        f"- packets: {manifest['counts']['packet_count']}",
        f"- schema docs: {manifest['counts']['schema_doc_count']}",
        f"- top_k: {manifest['top_k']}",
        f"- policy: `{manifest['relation_policy_path']}`",
        f"- policy_hash: `{manifest['relation_policy_hash']}`",
        "",
        "## Boundary",
        "",
        "- Read-only schema retrieval.",
        "- No provider calls.",
        "- No graph truth.",
        "- Retrieved relation types are prompt candidates, not proof.",
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
                "- candidates: "
                + ", ".join(
                    f"{candidate['relation_type']}({candidate['score']})"
                    for candidate in sample.get("relation_schema_candidates", [])[:10]
                ),
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def run_retriever(
    *,
    workspace: Path,
    input_packets: Path,
    relation_policy: Path,
    output_dir: Path,
    top_k: int,
    max_items: int | None = None,
) -> dict[str, Any]:
    workspace = workspace.resolve()
    input_packets = input_packets.resolve()
    relation_policy = relation_policy.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    policy = read_json(relation_policy)
    docs = build_schema_docs(policy)
    idf = inverse_document_frequency(docs)
    packets = read_jsonl(input_packets)
    if max_items is not None:
        packets = packets[: max(0, max_items)]

    rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for packet in packets:
        candidates = retrieve_for_packet(packet, docs, idf, top_k=top_k)
        row = {
            "schema_version": ROW_SCHEMA_VERSION,
            "packet_id": str(packet.get("packet_id") or ""),
            "workspace_id": packet.get("workspace_id"),
            "input_ref": packet.get("input_ref"),
            "text_preview": normalize_text(packet_text(packet))[:240],
            "relation_schema_candidates": candidates,
            "graph_is_not_proof": True,
        }
        rows.append(row)
        for candidate in candidates:
            features = candidate.get("retrieval_features") or {}
            csv_rows.append(
                {
                    "packet_id": row["packet_id"],
                    "rank": candidate["rank"],
                    "relation_type": candidate["relation_type"],
                    "score": candidate["score"],
                    "category": candidate["category"],
                    "source_count": candidate["source_count"],
                    "external_sources": ";".join(string_list(candidate.get("external_sources"))),
                    "exact_alias_hits": ";".join(string_list(features.get("exact_alias_hits"))),
                    "matched_tokens": ";".join(string_list(features.get("matched_tokens"))),
                }
            )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": now_iso(),
        "workspace": str(workspace),
        "input_packets": str(input_packets),
        "relation_policy_path": str(relation_policy),
        "relation_policy_hash": file_hash(relation_policy),
        "output_dir": str(output_dir),
        "top_k": top_k,
        "counts": {
            "packet_count": len(packets),
            "schema_doc_count": len(docs),
            "candidate_rows": len(csv_rows),
            "packets_without_candidates": sum(1 for row in rows if not row["relation_schema_candidates"]),
        },
        "boundary": {
            "provider_calls": False,
            "graph_truth": False,
            "graph_is_not_proof": True,
        },
        "outputs": {
            "relation_schema_candidates": str(output_dir / "relation_schema_candidates.jsonl"),
            "relation_schema_candidates_csv": str(output_dir / "relation_schema_candidates.csv"),
            "manifest": str(output_dir / "relation_schema_candidate_retrieval_manifest.json"),
            "report": str(output_dir / "relation_schema_candidate_retrieval_report.md"),
        },
    }
    write_jsonl(output_dir / "relation_schema_candidates.jsonl", rows)
    write_csv(output_dir / "relation_schema_candidates.csv", csv_rows)
    write_json(output_dir / "relation_schema_candidate_retrieval_manifest.json", manifest)
    write_text(output_dir / "relation_schema_candidate_retrieval_report.md", render_report(manifest, rows))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Retrieve packet-specific relation schema candidates.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--input-packets", required=True)
    parser.add_argument("--relation-policy", default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-items", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    workspace = Path(args.workspace)
    output_dir = Path(args.output_dir) if args.output_dir else workspace / DEFAULT_OUTPUT_DIR_NAME
    manifest = run_retriever(
        workspace=workspace,
        input_packets=Path(args.input_packets),
        relation_policy=Path(args.relation_policy),
        output_dir=output_dir,
        top_k=max(1, int(args.top_k or 40)),
        max_items=args.max_items,
    )
    print(json.dumps({"manifest": manifest["outputs"]["manifest"], "counts": manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
