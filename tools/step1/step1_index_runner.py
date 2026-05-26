"""Step 1 index runner for S1 query infrastructure.

This is a small integration runner, not the final Step 1 Query Toolbox.
It reads canonical Step 1 build assets and writes rebuildable lexical and
embedding index assets under indexes/. Index hits are retrieval signals only;
they are not evidence support, memory truth, Step 2 portrait units, or answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_RUN_SCOPES = {"evidence_index_only", "evidence_plus_memory_index", "full_s1_index"}
SUPPORTED_INDEX_MODES = {"lexical", "embedding", "lexical_and_embedding"}
SUPPORTED_DUPLICATE_POLICIES = {"fail", "overwrite_generated"}
EMBEDDING_BACKENDS = {"hash", "qwen_local"}
TOKENIZER_POLICY = "mixed_latin_word_cjk_unigram_bigram_v1"
CHUNKING_POLICY = "one_source_object_one_index_entry_v0.1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def short_hash(value: str, length: int = 16) -> str:
    return sha256_text(value)[:length]


def file_hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_hash(value: dict[str, Any]) -> str:
    return sha256_text(stable_json(value))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def jsonl_text(rows: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(jsonl_text(rows), encoding="utf-8")


def safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def stable_run_id(workspace_id: str, run_scope: str, index_mode: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"s1-index:{workspace_id}:{run_scope}:{index_mode}:{stamp}"


def conda_environment() -> str:
    return os.environ.get("CONDA_DEFAULT_ENV") or Path(sys.executable).parent.name


def ensure_lcoral_for_embedding(index_mode: str, allow_non_lcoral: bool) -> None:
    if index_mode == "lexical" or allow_non_lcoral:
        return
    exe = str(Path(sys.executable).resolve()).lower()
    env = conda_environment().lower()
    if "lcoral" not in exe and env != "lcoral":
        raise RuntimeError(
            "Embedding/vector index mode must run in the configured local lcoral environment. "
            f"Current python_executable={sys.executable!r}, conda_environment={conda_environment()!r}. "
            "Run with E:\\code\\anaconda\\envs\\lcoral\\python.exe or pass an explicit test-only override."
        )


def canonical_input_paths(workspace: Path) -> dict[str, Path]:
    return {
        "evidence/source_manifest.jsonl": workspace / "evidence" / "source_manifest.jsonl",
        "evidence/evidence.jsonl": workspace / "evidence" / "evidence.jsonl",
        "evidence/build_manifest.json": workspace / "evidence" / "build_manifest.json",
        "memory/memory_units.jsonl": workspace / "memory" / "memory_units.jsonl",
        "memory/memory_build_manifest.json": workspace / "memory" / "memory_build_manifest.json",
        "memory/summaries.jsonl": workspace / "memory" / "summaries.jsonl",
    }


def existing_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {name: file_hash(path) for name, path in paths.items() if path.exists()}


def tokenize(text: str) -> list[str]:
    latin = re.findall(r"[A-Za-z0-9_]+", text.lower())
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = [cjk[index] + cjk[index + 1] for index in range(len(cjk) - 1)]
    return latin + cjk + cjk_bigrams


def confidence_value(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    mapping = {"high": 0.9, "medium": 0.6, "low": 0.3, "unknown": None}
    return mapping.get(str(value).lower())


@dataclass
class IndexInputs:
    project_root: Path
    workspace: Path
    output_path: Path
    workspace_id: str
    run_scope: str
    index_mode: str
    duplicate_policy: str
    run_id: str
    started_at: str
    embedding_backend: str
    embedding_model: str
    embedding_dimension: int
    embedding_batch_size: int
    embedding_device: str | None
    allow_non_lcoral_for_tests: bool


@dataclass
class OutputPreparation:
    checkpoint_paths: list[str]
    overwritten_paths: list[str]


def load_inputs(args: argparse.Namespace) -> IndexInputs:
    project_root = Path(args.project_root).resolve()
    workspace = (project_root / args.workspace).resolve() if not Path(args.workspace).is_absolute() else Path(args.workspace).resolve()
    output_path = (
        (project_root / args.output_path).resolve()
        if args.output_path and not Path(args.output_path).is_absolute()
        else Path(args.output_path).resolve()
        if args.output_path
        else workspace / "indexes"
    )
    if args.run_scope not in SUPPORTED_RUN_SCOPES:
        raise ValueError(f"Unsupported run_scope {args.run_scope}; supported: {sorted(SUPPORTED_RUN_SCOPES)}")
    if args.index_mode not in SUPPORTED_INDEX_MODES:
        raise ValueError(f"Unsupported index_mode {args.index_mode}; supported: {sorted(SUPPORTED_INDEX_MODES)}")
    if args.duplicate_policy not in SUPPORTED_DUPLICATE_POLICIES:
        raise ValueError(
            f"Unsupported duplicate_policy {args.duplicate_policy}; supported: {sorted(SUPPORTED_DUPLICATE_POLICIES)}"
        )
    if args.embedding_backend not in EMBEDDING_BACKENDS:
        raise ValueError(f"Unsupported embedding_backend {args.embedding_backend}; supported: {sorted(EMBEDDING_BACKENDS)}")
    ensure_lcoral_for_embedding(args.index_mode, args.allow_non_lcoral_for_tests)
    build_manifest_path = workspace / "evidence" / "build_manifest.json"
    build_manifest = read_json(build_manifest_path) if build_manifest_path.exists() else {}
    workspace_id = args.workspace_id or build_manifest.get("workspace_id") or workspace.name
    run_id = args.run_id or stable_run_id(str(workspace_id), args.run_scope, args.index_mode)
    return IndexInputs(
        project_root=project_root,
        workspace=workspace,
        output_path=output_path,
        workspace_id=str(workspace_id),
        run_scope=args.run_scope,
        index_mode=args.index_mode,
        duplicate_policy=args.duplicate_policy,
        run_id=run_id,
        started_at=now_iso(),
        embedding_backend=args.embedding_backend,
        embedding_model=args.embedding_model,
        embedding_dimension=int(args.embedding_dimension),
        embedding_batch_size=int(getattr(args, "embedding_batch_size", 16)),
        embedding_device=getattr(args, "embedding_device", None),
        allow_non_lcoral_for_tests=bool(args.allow_non_lcoral_for_tests),
    )


def output_targets(inputs: IndexInputs) -> list[Path]:
    targets = [inputs.output_path / "step1_index_build_report.md"]
    if inputs.index_mode in {"lexical", "lexical_and_embedding"}:
        targets.extend(
            [
                inputs.output_path / "step1_bm25_manifest.json",
                inputs.output_path / "step1_bm25_entries.jsonl",
                inputs.output_path / "step1_bm25_index.json",
            ]
        )
    if inputs.index_mode in {"embedding", "lexical_and_embedding"}:
        targets.extend(
            [
                inputs.output_path / "step1_embedding_manifest.json",
                inputs.output_path / "step1_embedding_entries.jsonl",
                inputs.output_path / "step1_embedding_vectors.json",
            ]
        )
    return targets


def prepare_output(inputs: IndexInputs) -> OutputPreparation:
    existing = [path for path in output_targets(inputs) if path.exists()]
    if existing and inputs.duplicate_policy == "fail":
        raise FileExistsError(
            "Existing S1 index assets found and duplicate_policy=fail: " + ", ".join(str(path) for path in existing)
        )
    checkpoint_paths: list[str] = []
    overwritten_paths: list[str] = []
    if existing and inputs.duplicate_policy == "overwrite_generated":
        checkpoint_root = inputs.workspace / "checkpoints" / (
            "before-index-overwrite-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        )
        for path in existing:
            rel = path.relative_to(inputs.workspace) if path.resolve().is_relative_to(inputs.workspace.resolve()) else Path(path.name)
            target = checkpoint_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            checkpoint_paths.append(str(target))
            overwritten_paths.append(str(path))
    inputs.output_path.mkdir(parents=True, exist_ok=True)
    return OutputPreparation(checkpoint_paths=checkpoint_paths, overwritten_paths=overwritten_paths)


def validate_required_assets(inputs: IndexInputs) -> dict[str, Any]:
    paths = canonical_input_paths(inputs.workspace)
    missing: list[str] = []
    required = ["evidence/source_manifest.jsonl", "evidence/evidence.jsonl", "evidence/build_manifest.json"]
    if inputs.run_scope in {"evidence_plus_memory_index", "full_s1_index"}:
        required.extend(["memory/memory_units.jsonl", "memory/memory_build_manifest.json"])
    if inputs.run_scope == "full_s1_index":
        required.append("memory/summaries.jsonl")
    for name in required:
        if not paths[name].exists():
            missing.append(name)
    if missing:
        raise FileNotFoundError(f"Missing required canonical S1 assets for {inputs.run_scope}: {', '.join(missing)}")
    return {"required_assets": required, "input_hashes_before": existing_hashes(paths)}


def source_asset_versions(inputs: IndexInputs) -> dict[str, str]:
    versions: dict[str, str] = {}
    evidence_manifest = inputs.workspace / "evidence" / "build_manifest.json"
    if evidence_manifest.exists():
        manifest = read_json(evidence_manifest)
        versions["evidence/build_manifest.json"] = str(manifest.get("run_id") or manifest.get("completed_at") or "")
    memory_manifest = inputs.workspace / "memory" / "memory_build_manifest.json"
    if memory_manifest.exists():
        manifest = read_json(memory_manifest)
        versions["memory/memory_build_manifest.json"] = str(manifest.get("run_id") or manifest.get("completed_at") or "")
    summaries = inputs.workspace / "memory" / "summaries.jsonl"
    if summaries.exists():
        rows = read_jsonl(summaries)
        versions["memory/summaries.jsonl"] = ",".join(
            sorted({str(row.get("source_asset_version") or row.get("source_build_manifest_ref") or "") for row in rows})
        )
    return versions


def text_for_raw_evidence(item: dict[str, Any]) -> str:
    return str(item.get("text") or item.get("derived_text") or "")


def text_for_memory_unit(item: dict[str, Any]) -> str:
    return str(item.get("content") or item.get("memory_text") or item.get("candidate_text") or "")


def text_for_summary(item: dict[str, Any]) -> str:
    pieces: list[str] = []
    if item.get("summary_text"):
        pieces.append(str(item["summary_text"]))
    structured = item.get("structured_summary")
    if isinstance(structured, dict):
        if structured.get("session_topic"):
            pieces.append(str(structured["session_topic"]))
        for key in ("main_events", "explicit_commitments"):
            for event in structured.get(key, []) or []:
                if isinstance(event, dict) and event.get("text"):
                    pieces.append(str(event["text"]))
    return "\n".join(pieces)


def base_entry(
    *,
    inputs: IndexInputs,
    source_object_type: str,
    source_object_id: str,
    source_object_ref: str,
    source_object: dict[str, Any],
    text_for_index: str,
    source_object_version: str,
) -> dict[str, Any]:
    evidence_refs = list(source_object.get("evidence_refs") or source_object.get("backpointer_refs") or [])
    if source_object_type == "raw_evidence":
        evidence_ref = source_object.get("evidence_ref") or source_object.get("canonical_evidence_ref")
        evidence_refs = [evidence_ref] if evidence_ref else evidence_refs
    source_refs = list(source_object.get("source_refs") or [])
    if not source_refs and source_object.get("source_id"):
        source_refs = [source_object["source_id"]]
    locator = source_object.get("locator") or source_object.get("source_span")
    raw_source_id = source_object.get("raw_source_id")
    warnings = list(source_object.get("warnings") or [])
    if not raw_source_id:
        warnings.append("raw_source_id_unavailable")
    if not (locator or source_object.get("locator_unavailable_reason")):
        warnings.append("locator_unavailable")
    if source_object_type == "doc_level_summary":
        warnings.append("summary_hit_context_only")
    return {
        "schema_version": "step1.index_entry.v0.1",
        "index_entry_id": f"s1idx:{source_object_type}:{short_hash(source_object_id + ':' + text_for_index)}",
        "workspace_id": inputs.workspace_id,
        "source_object_id": source_object_id,
        "source_object_type": source_object_type,
        "source_object_ref": source_object_ref,
        "source_object_hash": source_object.get("content_hash") or object_hash(source_object),
        "source_object_version": source_object_version,
        "text_for_index": text_for_index,
        "text_hash": sha256_text(text_for_index),
        "text_token_count": len(tokenize(text_for_index)),
        "evidence_refs": evidence_refs,
        "source_refs": source_refs,
        "raw_source_id": raw_source_id,
        "item_layer": source_object.get("item_layer") or source_object_type,
        "modality": source_object.get("modality", "unknown"),
        "locator": locator,
        "locator_unavailable_reason": source_object.get("locator_unavailable_reason"),
        "source_specific_ref": source_object.get("source_specific_ref"),
        "display_ref": source_object.get("display_ref") or source_object_ref,
        "privacy_class": source_object.get("privacy_class", "unknown"),
        "scope": source_object.get("scope") or source_object.get("subject_scope") or "unknown",
        "confidence": source_object.get("confidence") or source_object.get("extraction_confidence") or "unknown",
        "confidence_numeric": confidence_value(source_object.get("confidence") or source_object.get("extraction_confidence")),
        "status": source_object.get("status", "active"),
        "subject_scope": source_object.get("subject_scope"),
        "subject_contamination_risk": source_object.get("subject_contamination_risk"),
        "warnings": warnings,
        "truth_status": truth_status(source_object_type),
    }


def truth_status(source_object_type: str) -> str:
    if source_object_type == "raw_evidence":
        return "raw_evidence_context; still requires support check before claim use"
    if source_object_type == "memory_unit":
        return "evidence_bound_derived_memory; not raw truth"
    if source_object_type == "doc_level_summary":
        return "compressed_context_only; not direct evidence"
    return "retrieval_context_only"


def build_index_units(inputs: IndexInputs) -> list[dict[str, Any]]:
    build_versions = source_asset_versions(inputs)
    units: list[dict[str, Any]] = []
    evidence = read_jsonl(inputs.workspace / "evidence" / "evidence.jsonl")
    evidence_version = build_versions.get("evidence/build_manifest.json", "")
    for item in evidence:
        text = text_for_raw_evidence(item)
        if not text.strip():
            continue
        ref = str(item.get("evidence_ref") or item.get("canonical_evidence_ref"))
        units.append(
            base_entry(
                inputs=inputs,
                source_object_type="raw_evidence",
                source_object_id=ref,
                source_object_ref=ref,
                source_object=item,
                text_for_index=text,
                source_object_version=evidence_version,
            )
        )

    if inputs.run_scope in {"evidence_plus_memory_index", "full_s1_index"}:
        memory_version = build_versions.get("memory/memory_build_manifest.json", "")
        for item in read_jsonl(inputs.workspace / "memory" / "memory_units.jsonl"):
            text = text_for_memory_unit(item)
            if not text.strip():
                continue
            ref = str(item.get("memory_id"))
            units.append(
                base_entry(
                    inputs=inputs,
                    source_object_type="memory_unit",
                    source_object_id=ref,
                    source_object_ref=ref,
                    source_object=item,
                    text_for_index=text,
                    source_object_version=memory_version,
                )
            )

    if inputs.run_scope == "full_s1_index":
        summary_version = build_versions.get("memory/summaries.jsonl", "")
        for item in read_jsonl(inputs.workspace / "memory" / "summaries.jsonl"):
            text = text_for_summary(item)
            if not text.strip():
                continue
            ref = str(item.get("summary_id"))
            entry = base_entry(
                inputs=inputs,
                source_object_type="doc_level_summary",
                source_object_id=ref,
                source_object_ref=ref,
                source_object=item,
                text_for_index=text,
                source_object_version=summary_version,
            )
            entry["source_evidence_hash"] = item.get("source_evidence_hash")
            entry["source_asset_version"] = item.get("source_asset_version")
            entry["generated_from_evidence_count"] = item.get("generated_from_evidence_count")
            units.append(entry)
    return units


def validate_index_units(units: list[dict[str, Any]]) -> dict[str, Any]:
    missing_text = [unit["index_entry_id"] for unit in units if not unit.get("text_for_index")]
    missing_raw_source = [
        unit["index_entry_id"]
        for unit in units
        if unit["source_object_type"] == "raw_evidence" and not unit.get("raw_source_id")
    ]
    missing_locator = [
        unit["index_entry_id"]
        for unit in units
        if unit["source_object_type"] == "raw_evidence"
        and not (unit.get("locator") or unit.get("locator_unavailable_reason"))
    ]
    duplicate_ids = duplicates(unit["index_entry_id"] for unit in units)
    return {
        "index_units": len(units),
        "missing_text_for_index": missing_text,
        "raw_evidence_missing_raw_source_id": missing_raw_source,
        "raw_evidence_missing_locator_or_reason": missing_locator,
        "duplicate_index_entry_ids": duplicate_ids,
        "valid": not any([missing_text, missing_raw_source, missing_locator, duplicate_ids]),
    }


def duplicates(values: Any) -> list[str]:
    seen: set[str] = set()
    dup: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            dup.add(text)
        seen.add(text)
    return sorted(dup)


def build_bm25(units: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    tokenized = [tokenize(unit["text_for_index"]) for unit in units]
    document_count = len(tokenized)
    avg_doc_len = sum(len(tokens) for tokens in tokenized) / document_count if document_count else 0.0
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    idf = {term: math.log(1.0 + (document_count - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
    entries: list[dict[str, Any]] = []
    for unit, tokens in zip(units, tokenized):
        term_counts = Counter(tokens)
        entries.append(
            {
                **without_text(unit),
                "token_count": len(tokens),
                "unique_token_count": len(term_counts),
                "top_terms": [term for term, _count in term_counts.most_common(20)],
            }
        )
    index = {
        "schema_version": "step1.bm25_index.v0.1",
        "tokenizer_policy": TOKENIZER_POLICY,
        "document_count": document_count,
        "avg_document_length": avg_doc_len,
        "k1": 1.5,
        "b": 0.75,
        "document_lengths": {unit["index_entry_id"]: len(tokens) for unit, tokens in zip(units, tokenized)},
        "idf": idf,
        "note": "BM25 index is lexical retrieval infrastructure, not factual support.",
    }
    return entries, index


def without_text(unit: dict[str, Any]) -> dict[str, Any]:
    row = dict(unit)
    row.pop("text_for_index", None)
    return row


def hash_embedding(text: str, dimension: int) -> list[float]:
    values = [0.0] * dimension
    tokens = tokenize(text) or [text]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for index in range(0, len(digest), 2):
            bucket = int.from_bytes(digest[index : index + 2], "big") % dimension
            sign = 1.0 if digest[index] % 2 == 0 else -1.0
            values[bucket] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [round(value / norm, 8) for value in values]


def build_embeddings(units: list[dict[str, Any]], inputs: IndexInputs) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    vectors, embedding_dimension, embedding_revision, embedding_policy = embed_texts(
        [unit["text_for_index"] for unit in units], inputs
    )
    entries: list[dict[str, Any]] = []
    index_entry_ids: list[str] = []
    for unit, vector in zip(units, vectors):
        index_entry_ids.append(unit["index_entry_id"])
        entries.append(
            {
                **without_text(unit),
                "embedding_backend": inputs.embedding_backend,
                "embedding_model": inputs.embedding_model,
                "embedding_model_revision": embedding_revision,
                "embedding_dimension": embedding_dimension,
                "embedding_policy": embedding_policy,
                "embedding_batch_size": inputs.embedding_batch_size,
                "embedding_device": inputs.embedding_device or "auto",
                "vector_sidecar_ref": "step1_embedding_vectors.json",
            }
        )
    sidecar = {
        "schema_version": "step1.embedding_vectors.v0.1",
        "alignment_key": "index_entry_id",
        "embedding_backend": inputs.embedding_backend,
        "embedding_model": inputs.embedding_model,
        "embedding_model_revision": embedding_revision,
        "embedding_dimension": embedding_dimension,
        "embedding_policy": embedding_policy,
        "embedding_batch_size": inputs.embedding_batch_size,
        "embedding_device": inputs.embedding_device or "auto",
        "index_entry_ids": index_entry_ids,
        "vectors": vectors,
        "vector_count": len(vectors),
        "note": "Embedding vectors are semantic retrieval infrastructure, not factual support.",
    }
    return entries, sidecar


def embed_texts(texts: list[str], inputs: IndexInputs) -> tuple[list[list[float]], int, str, str]:
    if inputs.embedding_backend == "hash":
        vectors = [hash_embedding(text, inputs.embedding_dimension) for text in texts]
        return (
            vectors,
            inputs.embedding_dimension,
            "not_applicable_for_hash_backend",
            "deterministic_hash_embedding_for_v0.1_integration_tests",
        )
    if inputs.embedding_backend == "qwen_local":
        return qwen_local_embeddings(texts, inputs)
    raise ValueError(f"Unsupported embedding backend: {inputs.embedding_backend}")


def qwen_local_embeddings(texts: list[str], inputs: IndexInputs) -> tuple[list[list[float]], int, str, str]:
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "qwen_local backend requires torch and transformers in the lcoral environment. "
            "No hash fallback is allowed for qwen_local."
        ) from exc

    try:
        device = torch.device(inputs.embedding_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        tokenizer = AutoTokenizer.from_pretrained(inputs.embedding_model, local_files_only=True)
        model = AutoModel.from_pretrained(inputs.embedding_model, local_files_only=True, trust_remote_code=True)
        model.to(device)
        model.eval()
        vectors: list[list[float]] = []
        for start in range(0, len(texts), inputs.embedding_batch_size):
            batch = texts[start : start + inputs.embedding_batch_size]
            encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.no_grad():
                output = model(**encoded)
                hidden = output.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            for row in pooled.detach().cpu().tolist():
                vectors.append([round(float(value), 8) for value in row])
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "qwen_local embedding failed. This backend must load the requested model locally and must not "
            f"fall back to hash. model={inputs.embedding_model!r}, python={sys.executable!r}"
        ) from exc

    dimension = len(vectors[0]) if vectors else 0
    revision = qwen_model_revision(inputs.embedding_model)
    return (
        vectors,
        dimension,
        revision,
        "transformers_local_qwen_mean_pool_normalized_embeddings_v0.1",
    )


def qwen_model_revision(model_id: str) -> str:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    cache_name = "models--" + model_id.replace("/", "--")
    refs_main = cache_root / cache_name / "refs" / "main"
    if refs_main.exists():
        return refs_main.read_text(encoding="utf-8").strip()
    return "unknown_local_revision"


def validate_embedding_sidecar(entries: list[dict[str, Any]], sidecar: dict[str, Any]) -> dict[str, Any]:
    entry_ids = [entry["index_entry_id"] for entry in entries]
    sidecar_ids = list(sidecar.get("index_entry_ids") or [])
    vectors = list(sidecar.get("vectors") or [])
    dimension = int(sidecar.get("embedding_dimension") or 0)
    bad_dimensions = [
        entry_id for entry_id, vector in zip(sidecar_ids, vectors) if not isinstance(vector, list) or len(vector) != dimension
    ]
    return {
        "embedding_entries": len(entries),
        "vector_count": len(vectors),
        "ids_match_entries": entry_ids == sidecar_ids,
        "vector_count_matches_entries": len(vectors) == len(entries),
        "bad_dimension_vectors": bad_dimensions,
        "valid": entry_ids == sidecar_ids and len(vectors) == len(entries) and not bad_dimensions,
    }


def source_object_counts(units: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter(unit["source_object_type"] for unit in units)
    return dict(sorted(counts.items()))


def included_layers(units: list[dict[str, Any]]) -> list[str]:
    return sorted({str(unit.get("item_layer") or unit.get("source_object_type")) for unit in units})


def common_manifest(
    inputs: IndexInputs,
    *,
    index_id: str,
    index_type: str,
    units: list[dict[str, Any]],
    validation: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_paths = canonical_input_paths(inputs.workspace)
    hashes = existing_hashes(source_paths)
    manifest: dict[str, Any] = {
        "schema_version": f"step1.{index_type}_manifest.v0.1",
        "workspace_id": inputs.workspace_id,
        "index_id": index_id,
        "index_type": index_type,
        "run_id": inputs.run_id,
        "workflow": "step1-index-workflow",
        "runner": "tools.step1.step1_index_runner",
        "run_scope": inputs.run_scope,
        "index_mode": inputs.index_mode,
        "built_from": {name: safe_relative(path, inputs.workspace) for name, path in source_paths.items() if path.exists()},
        "source_hashes": hashes,
        "source_asset_versions": source_asset_versions(inputs),
        "source_object_counts": source_object_counts(units),
        "included_layers": included_layers(units),
        "excluded_layers": excluded_layers(inputs.run_scope),
        "tokenizer_policy": TOKENIZER_POLICY,
        "chunking_policy": CHUNKING_POLICY,
        "python_executable": str(Path(sys.executable).resolve()),
        "conda_environment": conda_environment(),
        "built_at": now_iso(),
        "started_at": inputs.started_at,
        "builder": "step1_index_runner.v0.1",
        "status": "completed",
        "validation_summary": validation,
        "index_unit_policy": {
            "included_source_object_types": sorted(source_object_counts(units)),
            "text_for_index_policy": {
                "raw_evidence": "use evidence text/derived_text only",
                "memory_unit": "use memory content/candidate text only",
                "doc_level_summary": "use summary_text plus structured event snippets; context-only",
            },
            "excluded_statuses": ["rejected", "archived", "superseded"],
            "truth_status_note": "index entries are retrieval infrastructure, not truth source",
        },
        "warnings": warnings_for_units(units),
    }
    if extra:
        manifest.update(extra)
    return manifest


def excluded_layers(run_scope: str) -> list[str]:
    if run_scope == "evidence_index_only":
        return ["memory_unit", "doc_level_summary"]
    if run_scope == "evidence_plus_memory_index":
        return ["doc_level_summary"]
    return []


def warnings_for_units(units: list[dict[str, Any]]) -> list[str]:
    warnings: set[str] = set()
    for unit in units:
        for warning in unit.get("warnings") or []:
            warnings.add(str(warning))
    return sorted(warnings)


def write_outputs(
    inputs: IndexInputs,
    units: list[dict[str, Any]],
    preparation: OutputPreparation,
    input_validation: dict[str, Any],
) -> dict[str, Any]:
    outputs: dict[str, str] = {}
    validations: dict[str, Any] = {"index_units": validate_index_units(units)}
    if not validations["index_units"]["valid"]:
        raise ValueError(f"Index unit validation failed: {validations['index_units']}")

    if inputs.index_mode in {"lexical", "lexical_and_embedding"}:
        bm25_entries, bm25_index = build_bm25(units)
        bm25_validation = {"bm25_entries": len(bm25_entries), "document_count": bm25_index["document_count"], "valid": True}
        validations["bm25"] = bm25_validation
        write_jsonl(inputs.output_path / "step1_bm25_entries.jsonl", bm25_entries)
        write_json(inputs.output_path / "step1_bm25_index.json", bm25_index)
        bm25_manifest = common_manifest(
            inputs,
            index_id=f"step1-bm25:{inputs.workspace_id}:{short_hash(inputs.run_id)}",
            index_type="bm25",
            units=units,
            validation=bm25_validation,
            extra={
                "bm25_entries": "step1_bm25_entries.jsonl",
                "bm25_index": "step1_bm25_index.json",
                "bm25_policy": "in-run persistent lexical index files generated from S1 index units",
            },
        )
        write_json(inputs.output_path / "step1_bm25_manifest.json", bm25_manifest)
        outputs.update(
            {
                "bm25_entries": str(inputs.output_path / "step1_bm25_entries.jsonl"),
                "bm25_index": str(inputs.output_path / "step1_bm25_index.json"),
                "bm25_manifest": str(inputs.output_path / "step1_bm25_manifest.json"),
            }
        )

    if inputs.index_mode in {"embedding", "lexical_and_embedding"}:
        embedding_entries, sidecar = build_embeddings(units, inputs)
        embedding_validation = validate_embedding_sidecar(embedding_entries, sidecar)
        validations["embedding"] = embedding_validation
        if not embedding_validation["valid"]:
            raise ValueError(f"Embedding sidecar validation failed: {embedding_validation}")
        write_jsonl(inputs.output_path / "step1_embedding_entries.jsonl", embedding_entries)
        write_json(inputs.output_path / "step1_embedding_vectors.json", sidecar)
        embedding_manifest = common_manifest(
            inputs,
            index_id=f"step1-embedding:{inputs.workspace_id}:{short_hash(inputs.run_id)}",
            index_type="embedding",
            units=units,
            validation=embedding_validation,
            extra={
                "embedding_entries": "step1_embedding_entries.jsonl",
                "embedding_vectors": "step1_embedding_vectors.json",
                "embedding_backend": inputs.embedding_backend,
                "embedding_model": inputs.embedding_model,
                "embedding_model_revision": sidecar.get("embedding_model_revision"),
                "embedding_dimension": sidecar.get("embedding_dimension"),
                "embedding_policy": sidecar.get("embedding_policy"),
                "embedding_batch_size": inputs.embedding_batch_size,
                "embedding_device": inputs.embedding_device or "auto",
            },
        )
        write_json(inputs.output_path / "step1_embedding_manifest.json", embedding_manifest)
        outputs.update(
            {
                "embedding_entries": str(inputs.output_path / "step1_embedding_entries.jsonl"),
                "embedding_vectors": str(inputs.output_path / "step1_embedding_vectors.json"),
                "embedding_manifest": str(inputs.output_path / "step1_embedding_manifest.json"),
            }
        )

    after_hashes = existing_hashes(canonical_input_paths(inputs.workspace))
    canonical_unchanged = input_validation["input_hashes_before"] == after_hashes
    report_path = inputs.output_path / "step1_index_build_report.md"
    report = build_report(
        inputs=inputs,
        units=units,
        outputs=outputs,
        validations=validations,
        input_validation=input_validation,
        input_hashes_after=after_hashes,
        canonical_unchanged=canonical_unchanged,
        preparation=preparation,
    )
    report_path.write_text(report, encoding="utf-8")
    outputs["report"] = str(report_path)
    if not canonical_unchanged:
        raise RuntimeError("Canonical S1 assets changed during index build; index runner must not mutate them.")
    return {
        "workspace": str(inputs.workspace),
        "output_path": str(inputs.output_path),
        "run_id": inputs.run_id,
        "run_scope": inputs.run_scope,
        "index_mode": inputs.index_mode,
        "index_units": len(units),
        "outputs": outputs,
        "validations": validations,
    }


def build_report(
    *,
    inputs: IndexInputs,
    units: list[dict[str, Any]],
    outputs: dict[str, str],
    validations: dict[str, Any],
    input_validation: dict[str, Any],
    input_hashes_after: dict[str, str],
    canonical_unchanged: bool,
    preparation: OutputPreparation,
) -> str:
    counts = source_object_counts(units)
    count_lines = "\n".join(f"- {key}: {value}" for key, value in counts.items()) or "- none"
    outputs_lines = "\n".join(f"- {key}: `{value}`" for key, value in outputs.items()) or "- none"
    checkpoints = "\n".join(f"- `{path}`" for path in preparation.checkpoint_paths) or "- none"
    validations_text = json.dumps(validations, ensure_ascii=False, indent=2)
    return f"""# Step 1 Index Runner Report

## Status

- status: `completed`
- run_id: `{inputs.run_id}`
- workspace_id: `{inputs.workspace_id}`
- workspace: `{inputs.workspace}`
- output_path: `{inputs.output_path}`
- run_scope: `{inputs.run_scope}`
- index_mode: `{inputs.index_mode}`
- duplicate_policy: `{inputs.duplicate_policy}`
- python_executable: `{Path(sys.executable).resolve()}`
- conda_environment: `{conda_environment()}`
- started_at: `{inputs.started_at}`
- completed_at: `{now_iso()}`

## Outputs

{outputs_lines}

## Source Object Counts

{count_lines}

## Validation

```json
{validations_text}
```

## Canonical Asset Mutation Check

- canonical_assets_unchanged: `{canonical_unchanged}`
- before_hash_count: {len(input_validation.get("input_hashes_before", {}))}
- after_hash_count: {len(input_hashes_after)}

## Checkpoints

{checkpoints}

## Boundary Notes

- BM25 hit = lexical relevance, not factual proof.
- Embedding hit = semantic relevance, not factual proof.
- Memory-unit hit = evidence-bound derived memory, not raw truth.
- Summary hit = compressed context only; it must not enter support check as direct evidence without backpointer resolution.
- Step 2 must consume these indexes through Step 1 Query Toolbox, not by parsing index internals directly.
"""


def run_index(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_inputs(args)
    input_validation = validate_required_assets(inputs)
    preparation = prepare_output(inputs)
    units = build_index_units(inputs)
    return write_outputs(inputs, units, preparation, input_validation)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Step 1 lexical/embedding indexes from canonical S1 assets.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current working directory.")
    parser.add_argument("--workspace", required=True, help="Workspace containing canonical S1 build assets.")
    parser.add_argument("--workspace-id", help="Override workspace_id recorded in manifests.")
    parser.add_argument("--run-id", help="Override run id.")
    parser.add_argument(
        "--run-scope",
        default="evidence_index_only",
        help="evidence_index_only, evidence_plus_memory_index, or full_s1_index.",
    )
    parser.add_argument("--index-mode", default="lexical_and_embedding", help="lexical, embedding, or lexical_and_embedding.")
    parser.add_argument("--output-path", help="Index output path. Defaults to <workspace>/indexes.")
    parser.add_argument("--duplicate-policy", default="fail", help="fail or overwrite_generated.")
    parser.add_argument("--embedding-backend", default="hash", help="hash or qwen_local.")
    parser.add_argument(
        "--embedding-model",
        default="deterministic-hash-embedding-v0.1",
        help="Embedding model id. Use Qwen/Qwen3-Embedding-0.6B with --embedding-backend qwen_local.",
    )
    parser.add_argument("--embedding-dimension", type=int, default=64, help="Embedding vector dimension for hash backend.")
    parser.add_argument("--embedding-batch-size", type=int, default=16, help="Embedding batch size for local model backend.")
    parser.add_argument("--embedding-device", help="Optional local embedding device, e.g. cpu or cuda.")
    parser.add_argument(
        "--allow-non-lcoral-for-tests",
        action="store_true",
        help="Test-only override for unit tests; production embedding runs should use lcoral.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    output = run_index(args)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
