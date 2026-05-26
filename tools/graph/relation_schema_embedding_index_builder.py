"""Build a bilingual embedding index for graph relation schema retrieval.

The index is read-only auxiliary infrastructure. It embeds relation schema
documents, not user memories or graph truth.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.graph.graph_construction_packet_builder import file_hash, read_json, read_jsonl, sha256_text, write_json, write_jsonl, write_text


SCHEMA_VERSION = "graph_v03.relation_schema_embedding_index.v0.1"
DOC_SCHEMA_VERSION = "graph_v03.relation_schema_embedding_doc.v0.1"
DEFAULT_POLICY = "configs/graph/graph_relation_type_normalization.generated.v0.3.json"
DEFAULT_OUTPUT_DIR = "configs/graph/relation_schema_embedding_index"
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_EXTERNAL_ROOT = "external_references"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_embedding_backend() -> tuple[Any, Any, Any]:
    try:
        import torch  # type: ignore
        from transformers import AutoModel, AutoTokenizer  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Embedding index building requires optional dependencies: torch and transformers. "
            "Install the embedding extra or run with --skip-embeddings."
        ) from exc
    return torch, AutoModel, AutoTokenizer


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


def normalize_relation_type(value: Any) -> str:
    return re.sub(r"[^0-9a-zA-Z_\u4e00-\u9fff]+", "_", str(value or "").strip().lower().replace("-", "_")).strip("_")


def relation_aliases(policy: dict[str, Any]) -> dict[str, list[str]]:
    aliases: dict[str, list[str]] = defaultdict(list)
    for alias, canonical in (policy.get("alias_map") or {}).items():
        aliases[str(canonical)].append(str(alias))
    return {key: unique_strings(value, limit=160) for key, value in aliases.items()}


def source_count(meta: dict[str, Any]) -> int:
    total = 0
    for value in (meta.get("source_counts") or {}).values():
        try:
            total += int(value)
        except (TypeError, ValueError):
            continue
    return total


def wikidata_multilingual_terms(external_root: Path) -> dict[str, list[str]]:
    terms_by_relation: dict[str, list[str]] = defaultdict(list)
    property_map = {
        "P26": "spouse",
        "P22": "father",
        "P25": "mother",
        "P40": "child",
        "P3373": "sibling",
        "P1038": "relative",
        "P551": "residence",
        "P19": "place_of_birth",
        "P20": "place_of_death",
        "P463": "member_of",
        "P527": "has_part_s",
        "P361": "part_of",
        "P276": "location",
        "P159": "headquarters_location",
        "P108": "employer",
        "P69": "educated_at",
    }
    for prop, relation_type in property_map.items():
        path = external_root / "graph_relation_schemas" / "wikidata" / f"{prop}.json"
        if not path.exists():
            continue
        try:
            data = read_json(path)
            entity = (data.get("entities") or {}).get(prop) or {}
        except Exception:
            continue
        for lang in ("en", "zh", "zh-cn", "zh-hans"):
            label = ((entity.get("labels") or {}).get(lang) or {}).get("value")
            description = ((entity.get("descriptions") or {}).get(lang) or {}).get("value")
            terms_by_relation[relation_type].extend(string_list([label, description]))
            for alias_row in ((entity.get("aliases") or {}).get(lang) or []):
                terms_by_relation[relation_type].extend(string_list(alias_row.get("value")))
    return {key: unique_strings(value, limit=80) for key, value in terms_by_relation.items()}


def conceptnet_relation_descriptions(external_root: Path) -> dict[str, list[str]]:
    path = external_root / "graph_relation_schemas" / "conceptnet" / "conceptnet5.wiki" / "Relations.md"
    if not path.exists():
        return {}
    mapping = {
        "partof": "part_of",
        "hasa": "has_a",
        "usedfor": "used_for",
        "capableof": "capable_of",
        "atlocation": "location",
        "causes": "causes",
        "hasproperty": "has_property",
        "motivatedbygoal": "motivated_by_goal",
        "desires": "desires",
        "createdby": "created_by",
        "relatedto": "related_to_generic",
        "definedas": "defined_as",
        "locatednear": "located_near",
        "similarto": "similar_to",
        "madeof": "made_of",
    }
    out: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("| /r/"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 3:
            continue
        raw_name = cells[0].replace("/r/", "").lower()
        relation_type = mapping.get(raw_name)
        if relation_type:
            out[relation_type].extend([cells[1], cells[2]])
    return {key: unique_strings(value, limit=40) for key, value in out.items()}


def duie_predicate_terms(external_root: Path, *, max_rows: int = 5000) -> list[str]:
    root = external_root / "graph_route_calibration_datasets" / "DuIE-mirror-Bert-In-Relation-Extraction"
    terms: Counter[str] = Counter()
    for filename in ("duie_dev.json", "train.json", "new_train.json"):
        path = root / filename
        if not path.exists():
            continue
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for spo in row.get("spo_list") or []:
                    predicate = str(spo.get("predicate") or "").strip()
                    if predicate and "\ufffd" not in predicate:
                        terms[predicate] += 1
                count += 1
                if count >= max_rows:
                    break
    return [term for term, _ in terms.most_common(200)]


def chinese_literature_relation_terms(external_root: Path) -> list[str]:
    root = external_root / "graph_v03_validation_datasets" / "Chinese-Literature-NER-RE-Dataset"
    counts: Counter[str] = Counter()
    if not root.exists():
        return []
    for path in root.rglob("*.ann"):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.startswith("R"):
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                relation_type = parts[1].split()[0]
                if relation_type:
                    counts[relation_type] += 1
    return [term for term, _ in counts.most_common(50)]


def gap_review_docs(gap_review_path: Path | None) -> list[dict[str, Any]]:
    if gap_review_path is None or not gap_review_path.exists():
        return []
    docs: list[dict[str, Any]] = []
    for row in read_jsonl(gap_review_path):
        description = str(row.get("description") or "")
        cluster = str(row.get("gap_cluster_hint") or "")
        quote = str(row.get("quote") or "")
        source = str(row.get("source") or "")
        target = str(row.get("target") or "")
        if not any((description, cluster, quote, source, target)):
            continue
        relation_type = f"schema_gap_hint_{sha256_text(json.dumps(row, ensure_ascii=False, sort_keys=True))[:10]}"
        semantic_parts = unique_strings(
            [
                cluster.replace("_", " "),
                cluster,
                description,
                quote,
                source,
                target,
            ],
            limit=12,
        )
        docs.append(
            {
                "schema_version": DOC_SCHEMA_VERSION,
                "doc_id": f"relation_schema_doc:{sha256_text(relation_type)[:12]}",
                "relation_type": relation_type,
                "doc_kind": "supplemental_relation_gap_hint",
                "category": "observed_low_schema_gap",
                "external_sources": ["local_low_schema_gap_review"],
                "source_count": 0,
                "aliases": string_list([cluster, description]),
                "examples_or_glosses": semantic_parts,
                "semantic_text_en": " ".join(semantic_parts),
                "semantic_text_zh": "",
                "semantic_text": " ".join(semantic_parts),
                "language_coverage": ["en"],
                "source_paths": [str(gap_review_path)],
                "source_packet_id": row.get("packet_id"),
                "warnings": [
                    "supplemental_gap_hint_not_canonical_relation_type",
                    "do_not_use_as_provider_relation_type",
                ],
                "graph_is_not_proof": True,
            }
        )
    return docs


def build_embedding_docs(policy: dict[str, Any], external_root: Path, gap_review_path: Path | None = None) -> list[dict[str, Any]]:
    aliases_by_relation = relation_aliases(policy)
    wikidata_terms = wikidata_multilingual_terms(external_root)
    conceptnet_terms = conceptnet_relation_descriptions(external_root)
    local_gap_docs = gap_review_docs(gap_review_path)
    zh_relation_terms = unique_strings(duie_predicate_terms(external_root), chinese_literature_relation_terms(external_root), limit=300)

    docs: list[dict[str, Any]] = []
    for relation_type, meta in sorted((policy.get("canonical_relation_types") or {}).items()):
        if relation_type == "related_to_generic":
            continue
        aliases = aliases_by_relation.get(relation_type, [])
        external_sources = string_list(meta.get("external_sources"))
        category = str(meta.get("category") or "external_relation_schema")
        examples = unique_strings(
            wikidata_terms.get(relation_type, []),
            conceptnet_terms.get(relation_type, []),
            limit=80,
        )
        semantic_text_en = " ".join(
            unique_strings(
                [relation_type, relation_type.replace("_", " ")],
                aliases,
                examples,
                external_sources,
                [category],
                limit=260,
            )
        )
        semantic_text_zh = " ".join(
            term
            for term in examples
            if any("\u4e00" <= char <= "\u9fff" for char in term)
        )
        semantic_text = " ".join(unique_strings([semantic_text_en, semantic_text_zh], limit=2))
        docs.append(
            {
                "schema_version": DOC_SCHEMA_VERSION,
                "doc_id": f"relation_schema_doc:{sha256_text(relation_type)[:12]}",
                "relation_type": relation_type,
                "doc_kind": "canonical_relation_type",
                "category": category,
                "external_sources": external_sources,
                "source_count": source_count(meta),
                "aliases": aliases[:80],
                "examples_or_glosses": examples,
                "semantic_text_en": semantic_text_en,
                "semantic_text_zh": semantic_text_zh,
                "semantic_text": semantic_text,
                "language_coverage": [
                    lang
                    for lang, text in (("en", semantic_text_en), ("zh", semantic_text_zh))
                    if text.strip()
                ],
                "source_paths": string_list(meta.get("source_paths")),
                "graph_is_not_proof": True,
            }
        )

    docs.extend(local_gap_docs)

    for term in zh_relation_terms:
        relation_type = f"zh_relation_hint_{sha256_text(term)[:8]}"
        docs.append(
            {
                "schema_version": DOC_SCHEMA_VERSION,
                "doc_id": f"relation_schema_doc:{sha256_text(relation_type)[:12]}",
                "relation_type": relation_type,
                "doc_kind": "supplemental_chinese_relation_hint",
                "category": "external_chinese_relation_hint",
                "external_sources": ["duie_or_chinese_literature_relation_labels"],
                "source_count": 0,
                "aliases": [term],
                "examples_or_glosses": [term],
                "semantic_text_en": "",
                "semantic_text_zh": term,
                "semantic_text": term,
                "language_coverage": ["zh"],
                "source_paths": [],
                "warnings": ["supplemental_hint_not_canonical_relation_type"],
                "graph_is_not_proof": True,
            }
        )
    return docs


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def encode_texts(texts: list[str], *, model_name: str, batch_size: int, max_length: int) -> np.ndarray:
    torch, AutoModel, AutoTokenizer = load_embedding_backend()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True, trust_remote_code=True)
    model.eval()
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            output = model(**inputs)
            pooled = mean_pool(output.last_hidden_state, inputs["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype("float32"))
    if not vectors:
        return np.zeros((0, 0), dtype="float32")
    return np.vstack(vectors)


def render_report(manifest: dict[str, Any]) -> str:
    counts = manifest["counts"]
    return "\n".join(
        [
            "# Relation Schema Embedding Index Report",
            "",
            f"- model: `{manifest['embedding_model']}`",
            f"- doc_count: {counts['doc_count']}",
            f"- canonical_docs: {counts['canonical_doc_count']}",
            f"- supplemental_chinese_hint_docs: {counts['supplemental_chinese_hint_doc_count']}",
            f"- supplemental_relation_gap_hint_docs: {counts['supplemental_relation_gap_hint_doc_count']}",
            f"- vector_dim: {counts['vector_dim']}",
            "- graph_is_not_proof: `true`",
            "",
            "## Boundary",
            "",
            "- This is a schema retrieval index, not graph truth.",
            "- Supplemental Chinese hint docs are retrieval aids and are not canonical relation types.",
            "- Supplemental gap hint docs are retrieval diagnostics and must not be offered as provider relation types.",
            "- No provider calls were made.",
            "",
        ]
    )


def build_index(
    *,
    relation_policy: Path,
    external_root: Path,
    output_dir: Path,
    embedding_model: str,
    gap_review_path: Path | None = None,
    batch_size: int = 8,
    max_length: int = 256,
    max_docs: int | None = None,
    skip_embeddings: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = read_json(relation_policy)
    docs = build_embedding_docs(policy, external_root, gap_review_path)
    if max_docs is not None:
        docs = docs[: max(0, max_docs)]
    texts = [str(doc.get("semantic_text") or doc.get("relation_type") or "") for doc in docs]
    if skip_embeddings:
        matrix = np.zeros((len(docs), 0), dtype="float32")
    else:
        matrix = encode_texts(texts, model_name=embedding_model, batch_size=batch_size, max_length=max_length)

    docs_path = output_dir / "relation_schema_embedding_docs.jsonl"
    matrix_path = output_dir / "relation_schema_embedding_matrix.npy"
    manifest_path = output_dir / "relation_schema_embedding_manifest.json"
    report_path = output_dir / "relation_schema_embedding_report.md"
    write_jsonl(docs_path, docs)
    np.save(matrix_path, matrix)
    counts = {
        "doc_count": len(docs),
        "canonical_doc_count": sum(1 for doc in docs if doc.get("doc_kind") == "canonical_relation_type"),
        "supplemental_chinese_hint_doc_count": sum(1 for doc in docs if doc.get("doc_kind") == "supplemental_chinese_relation_hint"),
        "supplemental_relation_gap_hint_doc_count": sum(1 for doc in docs if doc.get("doc_kind") == "supplemental_relation_gap_hint"),
        "vector_dim": int(matrix.shape[1]) if matrix.ndim == 2 and matrix.shape[0] else 0,
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": now_iso(),
        "embedding_model": embedding_model,
        "embedding_backend": "transformers_mean_pooling",
        "local_files_only": True,
        "batch_size": batch_size,
        "max_length": max_length,
        "relation_policy": str(relation_policy.resolve()),
        "relation_policy_hash": file_hash(relation_policy),
        "external_root": str(external_root.resolve()),
        "gap_review_path": str(gap_review_path.resolve()) if gap_review_path else None,
        "gap_review_hash": file_hash(gap_review_path) if gap_review_path else None,
        "output_dir": str(output_dir),
        "outputs": {
            "docs": str(docs_path),
            "matrix": str(matrix_path),
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
        "counts": counts,
        "boundary": {
            "provider_calls": False,
            "graph_truth": False,
            "graph_is_not_proof": True,
            "write_permission": False,
        },
    }
    write_json(manifest_path, manifest)
    write_text(report_path, render_report(manifest))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build bilingual relation schema embedding index.")
    parser.add_argument("--relation-policy", default=DEFAULT_POLICY)
    parser.add_argument("--external-root", default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--embedding-model", default=DEFAULT_MODEL)
    parser.add_argument("--gap-review", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = build_index(
        relation_policy=Path(args.relation_policy),
        external_root=Path(args.external_root),
        output_dir=Path(args.output_dir),
        embedding_model=args.embedding_model,
        gap_review_path=Path(args.gap_review) if args.gap_review else None,
        batch_size=max(1, int(args.batch_size or 8)),
        max_length=max(16, int(args.max_length or 256)),
        max_docs=args.max_docs,
        skip_embeddings=bool(args.skip_embeddings),
    )
    print(json.dumps({"manifest": manifest["outputs"]["manifest"], "counts": manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
