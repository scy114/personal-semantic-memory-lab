#!/usr/bin/env python
"""Generic Step 2 query runner.

This tool consumes an existing Step 2.1 user-model workspace and a question
file, then emits Step 2.2 query artifacts plus a Step 2.3-compatible usable
memory context. It deliberately does not contain dataset-specific answers.

The runner treats vector search as retrieval infrastructure, not truth.
Factual support still depends on evidence refs and downstream support checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.step1.step1_toolbox import LocalStep1Toolbox
from tools.graph.graph_query_retriever import discover_graph_dir, retrieve_graph_query_package
from tools.maintenance.latest_view import resolve_latest_view_input
from tools.step2.s23_answer_context_runner import build_s23_answer_context, dual_branch_summary, render_s23_prompt_context

DEFAULT_MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
HASH_MODEL_ID = "deterministic-hash-embedding-v0.1"
ACTIVE_STATUSES = {"active", "accepted", "accepted_for_experiment", "accepted_by_user", "current"}
INACTIVE_STATUSES = {"rejected", "archived", "superseded", "unresolved"}
KNOWN_ROUTES = {
    "evidence_first",
    "model_first",
    "graph_assisted",
    "hybrid",
    "procedural_assistance",
    "semantic",
    "exploratory",
    "insufficient_scope",
    "query_and_update_candidate_disabled",
}
LATIN_TOKEN_RE = re.compile(r"[a-z0-9_]+", re.I)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
V03_GRAPH_DIR_CANDIDATES = [
    "graph_v03_consolidation_provider_80",
    "graph_v03_consolidation",
    "graph_v03_prototype",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path, required: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def resolve_path(value: str | Path | None, base: Path = ROOT) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def tokenize(text: str) -> list[str]:
    """Mixed English/CJK tokenizer for Step 2 lexical retrieval.

    This is intentionally dependency-free. It supports English word tokens plus
    CJK unigrams and bigrams. It is not a final Chinese segmentation policy.
    """

    text = (text or "").lower()
    tokens = LATIN_TOKEN_RE.findall(text)
    cjk_chars = CJK_RE.findall(text)
    tokens.extend(cjk_chars)
    tokens.extend("".join(pair) for pair in zip(cjk_chars, cjk_chars[1:]))
    return tokens


class BM25Index:
    def __init__(self, rows: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75) -> None:
        self.rows = rows
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(row.get("text_for_retrieval", "")) for row in rows]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        df: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            df.update(set(tokens))
        self.doc_freq = dict(df)
        self.idf = {
            term: math.log(1.0 + ((len(rows) - freq + 0.5) / (freq + 0.5)))
            for term, freq in self.doc_freq.items()
        }

    def score(self, query: str, doc_index: int) -> float:
        query_terms = tokenize(query)
        if not query_terms or doc_index >= len(self.rows):
            return 0.0
        tf = self.term_freqs[doc_index]
        dl = self.doc_lengths[doc_index] or 0
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if not freq:
                continue
            denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
            score += self.idf.get(term, 0.0) * ((freq * (self.k1 + 1.0)) / denom)
        return float(score)

    def scores(self, query: str) -> list[float]:
        return [self.score(query, idx) for idx in range(len(self.rows))]

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": "s2.bm25_index_manifest.v1",
            "index_type": "bm25",
            "tokenizer_policy": "mixed_latin_word_cjk_unigram_bigram_v1",
            "k1": self.k1,
            "b": self.b,
            "document_count": len(self.rows),
            "average_document_length": self.avgdl,
            "vocabulary_size": len(self.doc_freq),
            "truth_status": "rebuildable_query_index_not_truth",
            "notes": [
                "BM25 is built in-memory from Step 2 query_units for this run.",
                "BM25 is lexical retrieval infrastructure, not evidence support.",
                "Chinese tokenization is dependency-free CJK unigram/bigram, not final segmentation.",
            ],
        }


def resolve_fusion_weights(question: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    if args.lexical_only:
        return {"bm25": 1.0, "embedding": 0.0, "source": "lexical_only"}
    if args.bm25_weight is not None or args.embedding_weight is not None:
        bm25 = args.bm25_weight
        embedding = args.embedding_weight
        if bm25 is None:
            bm25 = max(0.0, 1.0 - float(embedding or 0.0))
        if embedding is None:
            embedding = max(0.0, 1.0 - float(bm25 or 0.0))
        return {"bm25": float(bm25), "embedding": float(embedding), "source": "cli"}

    route = str(question.get("route") or "").lower()
    answer_type = str(question.get("expected_answer_type") or "").lower()
    if route == "evidence_first" or answer_type == "fact":
        return {"bm25": 0.55, "embedding": 0.45, "source": "route:evidence_first_or_fact"}
    if route in {"semantic", "exploratory"} or answer_type in {"semantic", "exploratory"}:
        return {"bm25": 0.25, "embedding": 0.75, "source": "route:semantic_or_exploratory"}
    return {"bm25": 0.35, "embedding": 0.65, "source": "route:hybrid_default"}


def bucket_for_score(score: float, args: argparse.Namespace) -> str:
    if score >= args.direct_threshold:
        return "direct_context"
    if score >= args.supporting_threshold:
        return "supporting_context"
    if score >= args.min_score:
        return "weak_context"
    return "excluded"


def normalize_vector_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def hash_embedding(text: str, dimension: int) -> np.ndarray:
    tokens = [token for token in str(text or "").lower().replace("\n", " ").split(" ") if token] or [str(text or "")]
    vector = np.zeros((dimension,), dtype="float32")
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    return vector.astype("float32")


def stable_id(prefix: str, value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.:-]+", "-", value.strip()).strip("-")
    return f"{prefix}:{safe or 'unknown'}"


def first_text(row: dict[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def evidence_refs_from(row: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for key in ("evidence_refs", "backpointer_refs"):
        for ref in listify(row.get(key)):
            if isinstance(ref, str) and ref not in refs:
                refs.append(ref)
    for rel in listify(row.get("relation_candidates")):
        if isinstance(rel, dict):
            for ref in listify(rel.get("evidence_refs")):
                if isinstance(ref, str) and ref not in refs:
                    refs.append(ref)
    return refs


def source_refs_from(row: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    for ref in listify(row.get("source_refs")):
        if isinstance(ref, str) and ref not in refs:
            refs.append(ref)
    return refs


def temporal_status(row: dict[str, Any]) -> str:
    explicit = row.get("temporal_status")
    if explicit:
        return str(explicit)
    temporal = row.get("temporal_scope")
    if isinstance(temporal, dict):
        validity = temporal.get("validity")
        if validity in {"current", "recurring", "historical", "temporary", "unknown"}:
            return str(validity)
    return "unknown"


def status_of(row: dict[str, Any]) -> str:
    return str(row.get("status") or row.get("review_status") or row.get("state") or "active")


@dataclass
class WorkspacePaths:
    workspace: Path
    manifest: Path
    reviewed_units: Path
    portrait_json: Path
    portrait_md: Path
    graph_nodes: Path
    graph_edges: Path
    base_packet: Path
    evidence: Path | None
    index_manifest: Path | None
    index_entries: Path | None
    index_vectors: Path | None


@dataclass
class WorkspaceAssets:
    paths: WorkspacePaths
    manifest: dict[str, Any]
    reviewed_units: list[dict[str, Any]]
    portrait_json: dict[str, Any]
    graph_nodes: list[dict[str, Any]]
    graph_edges: list[dict[str, Any]]
    base_packet: dict[str, Any]
    evidence: list[dict[str, Any]]
    index_manifest: dict[str, Any]
    index_entries: list[dict[str, Any]]
    vectors: np.ndarray | None


class Embedder:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(device)
        self.model.eval()

    def encode(self, text: str) -> np.ndarray:
        import torch

        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model(**inputs)
        hidden = output.last_hidden_state
        mask = inputs["attention_mask"]
        lengths = mask.sum(dim=1) - 1
        batch = torch.arange(hidden.size(0), device=hidden.device)
        pooled = hidden[batch, lengths]
        vec = pooled[0].detach().cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm:
            vec = vec / norm
        return vec

    def encode_batch(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        import torch

        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        vectors: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
                output = self.model(**inputs)
                hidden = output.last_hidden_state
                mask = inputs["attention_mask"]
                lengths = mask.sum(dim=1) - 1
                batch_indices = torch.arange(hidden.size(0), device=hidden.device)
                pooled = hidden[batch_indices, lengths].detach().cpu().numpy().astype(np.float32)
                norms = np.linalg.norm(pooled, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                vectors.append(pooled / norms)
        return np.vstack(vectors).astype(np.float32)


class HashEmbedder:
    def __init__(self, dimension: int) -> None:
        self.device = "not_applicable"
        self.model_id = HASH_MODEL_ID
        self.dimension = dimension

    def encode(self, text: str) -> np.ndarray:
        return hash_embedding(text, self.dimension)

    def encode_batch(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        return np.vstack([self.encode(text) for text in texts]) if texts else np.zeros((0, self.dimension), dtype=np.float32)


class GraphReranker:
    def __init__(self, model_id: str, device: str = "auto") -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
        self.model.eval()

    def predict(self, pairs: list[list[str]], *, batch_size: int = 16, show_progress_bar: bool = False):
        import torch

        scores: list[float] = []
        with torch.no_grad():
            for start in range(0, len(pairs), batch_size):
                batch = pairs[start : start + batch_size]
                left = [str(pair[0]) for pair in batch]
                right = [str(pair[1]) for pair in batch]
                inputs = self.tokenizer(
                    left,
                    right,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                output = self.model(**inputs)
                logits = output.logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                elif logits.ndim == 2:
                    logits = logits[:, -1]
                scores.extend(float(value) for value in logits.detach().cpu().tolist())
        return scores


def parse_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    data: dict[str, Any] = {"_raw": text}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped or stripped.startswith("-"):
            continue
        key, value = stripped.split(":", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def discover_evidence_path(workspace: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    candidates = [workspace / "evidence" / "evidence.jsonl"]
    repair_root = workspace / "repairs"
    if repair_root.exists():
        candidates.extend(sorted(repair_root.glob("*/evidence/evidence.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True))
    for path in candidates:
        if path.exists():
            return path
    return None


def discover_step1_root(workspace: Path, explicit: Path | None) -> Path | None:
    if explicit:
        return explicit
    candidates = [
        workspace / "step1",
        workspace,
        workspace / "evidence",
    ]
    for path in candidates:
        if (path / "evidence.jsonl").exists() or (path / "evidence" / "evidence.jsonl").exists():
            return path
    return None


def load_step1_toolbox(args: argparse.Namespace, workspace: Path) -> tuple[LocalStep1Toolbox | None, dict[str, Any]]:
    if args.disable_step1_toolbox:
        return None, {
            "status": "disabled",
            "root": None,
            "warnings": ["step1_toolbox_disabled_by_cli"],
        }
    root = discover_step1_root(workspace, resolve_path(args.step1_root))
    if root is None:
        return None, {
            "status": "unavailable",
            "root": None,
            "warnings": ["step1_toolbox_root_not_found"],
        }
    try:
        toolbox = LocalStep1Toolbox(root)
    except Exception as exc:  # pragma: no cover - defensive degraded mode.
        return None, {
            "status": "failed",
            "root": str(root),
            "warnings": [f"step1_toolbox_load_failed:{type(exc).__name__}:{exc}"],
        }
    if not toolbox.store.evidence:
        return None, {
            "status": "unavailable",
            "root": str(root),
            "warnings": ["step1_toolbox_has_no_evidence"],
        }
    warnings = []
    if not toolbox.store.memory_units:
        warnings.append("step1_toolbox_has_no_memory_units")
    return toolbox, {
        "status": "available",
        "root": str(root),
        "source_count": len(toolbox.store.sources),
        "evidence_count": len(toolbox.store.evidence),
        "memory_unit_count": len(toolbox.store.memory_units),
        "warnings": warnings,
    }


def step1_retrieval_filters(args: argparse.Namespace) -> dict[str, Any]:
    filters: dict[str, Any] = {
        "retrieval_mode": args.step1_retrieval_mode,
    }
    if args.step1_index_root:
        filters["index_root"] = str(resolve_path(args.step1_index_root))
    item_layers = ["raw_evidence"]
    if args.step1_include_memory_units:
        filters["include_memory_units"] = True
        item_layers.append("memory_unit")
    if args.step1_include_summaries:
        filters["include_summaries"] = True
        item_layers.append("doc_level_summary")
    filters["item_layers"] = item_layers
    if args.target_subject_id:
        filters["target_subject_id"] = args.target_subject_id
        filters["subject_ids"] = [args.target_subject_id]
    return filters


def step1_route_policy(question: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    route_policy: dict[str, Any] = {
        "route": question.get("route") or "mixed",
        "retrieval_mode": args.step1_retrieval_mode,
    }
    if args.step1_index_root:
        route_policy["index_root"] = str(resolve_path(args.step1_index_root))
    if args.target_subject_id:
        route_policy["target_subject_id"] = args.target_subject_id
    if args.step1_embedding_device:
        route_policy["embedding_device"] = args.step1_embedding_device
    if args.step1_allow_hash_embedding_for_tests:
        route_policy["allow_hash_embedding_for_tests"] = True
    return route_policy


def run_step1_retrieval(
    question: dict[str, Any],
    step1_toolbox: LocalStep1Toolbox | None,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if step1_toolbox is None:
        return [], {
            "status": "unavailable",
            "retrieval_mode": args.step1_retrieval_mode,
            "warnings": ["step1_toolbox_unavailable"],
        }
    if args.step1_retrieval_mode == "disabled":
        return [], {
            "status": "disabled",
            "retrieval_mode": "disabled",
            "warnings": ["step1_retrieval_disabled_by_cli"],
        }
    filters = step1_retrieval_filters(args)
    route_policy = step1_route_policy(question, args)
    results = step1_toolbox.retrieve_evidence(
        question["question"],
        filters=filters,
        top_k=args.step1_retrieval_top_k,
        route_policy=route_policy,
    )
    active_modes = sorted({str(row.get("retrieval_mode") or "unknown") for row in results})
    return results, {
        "status": "available",
        "retrieval_mode": args.step1_retrieval_mode,
        "result_count": len(results),
        "active_result_modes": active_modes,
        "index_root": filters.get("index_root"),
        "include_memory_units": bool(args.step1_include_memory_units),
        "include_summaries": bool(args.step1_include_summaries),
        "target_subject_id": args.target_subject_id,
        "warnings": sorted({warning for row in results for warning in row.get("warnings", [])}),
    }


def is_interrogative_query(text: str) -> bool:
    stripped = (text or "").strip()
    lowered = stripped.lower()
    interrogative_prefixes = (
        "what ",
        "when ",
        "where ",
        "who ",
        "whom ",
        "whose ",
        "why ",
        "how ",
        "which ",
        "does ",
        "do ",
        "did ",
        "is ",
        "are ",
        "was ",
        "were ",
        "can ",
        "could ",
        "would ",
        "should ",
    )
    return stripped.endswith("?") or lowered.startswith(interrogative_prefixes)


def make_query_claim_support_check(
    question: dict[str, Any],
    step1_results: list[dict[str, Any]],
    step1_toolbox: LocalStep1Toolbox | None,
) -> dict[str, Any]:
    evidence_refs: list[str] = []
    for result in step1_results:
        if result.get("item_layer") != "raw_evidence":
            continue
        for ref in listify(result.get("evidence_refs")):
            if isinstance(ref, str) and ref not in evidence_refs:
                evidence_refs.append(ref)
    if is_interrogative_query(question["question"]):
        return {
            "schema_version": "s2.query_claim_support_check.v1",
            "claim": question["question"],
            "support_status": "not_checked",
            "support_strength": "unknown",
            "evidence_refs": evidence_refs,
            "warnings": ["query_is_question_not_claim", "support_check_requires_claim_not_question"],
            "notes": [
                "The input is an interrogative query, not a factual claim.",
                "Retrieved evidence may answer the query, but query-answer support must be checked after an answer claim is formed.",
            ],
        }
    if step1_toolbox is None:
        return {
            "schema_version": "s2.query_claim_support_check.v1",
            "claim": question["question"],
            "support_status": "not_checked",
            "support_strength": "unknown",
            "evidence_refs": evidence_refs,
            "warnings": ["step1_toolbox_unavailable"],
            "notes": [
                "This check evaluates the user query claim itself, not merely selected context refs.",
            ],
        }
    if not evidence_refs:
        return {
            "schema_version": "s2.query_claim_support_check.v1",
            "claim": question["question"],
            "support_status": "not_checked",
            "support_strength": "unknown",
            "evidence_refs": [],
            "warnings": ["no_raw_evidence_refs_from_step1_retrieval"],
            "notes": [
                "No raw evidence refs were retrieved for the user query claim.",
            ],
        }
    support_check = step1_toolbox.check_claim_support(
        question["question"],
        evidence_refs,
        options={"default_ref_type": "evidence_ref"},
    )
    return {
        "schema_version": "s2.query_claim_support_check.v1",
        "claim": question["question"],
        "support_status": "checked",
        "support_strength": support_check.get("support_strength", "unknown"),
        "evidence_refs": evidence_refs,
        "step1_support_check": support_check,
        "warnings": [
            "query_claim_check_is_rule_based",
            "retrieved_context_support_does_not_imply_query_claim_support",
        ],
        "notes": [
            "This check evaluates the user query claim itself.",
            "Selected S2 context support checks are separate and must not be used as a substitute.",
        ],
    }


def default_index_paths(workspace: Path) -> tuple[Path | None, Path | None, Path | None]:
    index_dir = workspace / "indexes"
    manifest = index_dir / "step2_user_model_embedding_manifest.json"
    entries = index_dir / "step2_user_model_embedding_entries.jsonl"
    vectors = index_dir / "step2_user_model_embedding_vectors.npy"
    if manifest.exists() and entries.exists() and vectors.exists():
        return manifest, entries, vectors

    manifests = sorted(index_dir.glob("step2_user_model_embedding_manifest*.json"))
    for man in manifests:
        suffix = man.name.removeprefix("step2_user_model_embedding_manifest").removesuffix(".json")
        ent = index_dir / f"step2_user_model_embedding_entries{suffix}.jsonl"
        vec = index_dir / f"step2_user_model_embedding_vectors{suffix}.npy"
        if ent.exists() and vec.exists():
            return man, ent, vec
    return None, None, None


def load_workspace(args: argparse.Namespace) -> WorkspaceAssets:
    workspace = resolve_path(args.workspace)
    assert workspace is not None
    if not workspace.exists():
        raise FileNotFoundError(f"Workspace not found: {workspace}")

    default_manifest, default_entries, default_vectors = default_index_paths(workspace)
    evidence_path = discover_evidence_path(workspace, resolve_path(args.evidence))
    reviewed_units_path, reviewed_units_input = resolve_latest_view_input(
        workspace=workspace,
        layer="s2",
        canonical_path=workspace / "portrait" / "reviewed_units.jsonl",
        explicit_path=resolve_path(args.reviewed_units),
        mode=getattr(args, "latest_view_mode", "auto"),
    )
    graph_nodes_path, graph_nodes_input = resolve_latest_view_input(
        workspace=workspace,
        layer="graph_nodes",
        canonical_path=workspace / "graph" / "nodes.jsonl",
        explicit_path=resolve_path(args.graph_nodes),
        mode=getattr(args, "latest_view_mode", "auto"),
    )
    graph_edges_path, graph_edges_input = resolve_latest_view_input(
        workspace=workspace,
        layer="graph_edges",
        canonical_path=workspace / "graph" / "edges.jsonl",
        explicit_path=resolve_path(args.graph_edges),
        mode=getattr(args, "latest_view_mode", "auto"),
    )
    paths = WorkspacePaths(
        workspace=workspace,
        manifest=workspace / "manifest.yaml",
        reviewed_units=reviewed_units_path,
        portrait_json=resolve_path(args.portrait_json) or workspace / "portrait" / "current_portrait.json",
        portrait_md=workspace / "portrait" / "current_portrait.md",
        graph_nodes=graph_nodes_path,
        graph_edges=graph_edges_path,
        base_packet=resolve_path(args.base_packet) or workspace / "packets" / "base_assistance_packet.json",
        evidence=evidence_path,
        index_manifest=resolve_path(args.index_manifest) or default_manifest,
        index_entries=resolve_path(args.index_entries) or default_entries,
        index_vectors=resolve_path(args.index_vectors) or default_vectors,
    )

    vectors: np.ndarray | None = None
    index_manifest: dict[str, Any] = {}
    index_entries: list[dict[str, Any]] = []
    if not args.lexical_only:
        if not paths.index_manifest or not paths.index_entries or not paths.index_vectors:
            raise FileNotFoundError("Embedding index files were not found. Use --lexical-only or pass index paths.")
        index_manifest = read_json(paths.index_manifest)
        index_entries = read_jsonl(paths.index_entries)
        vectors = normalize_vector_matrix(np.load(paths.index_vectors))
        if len(index_entries) != vectors.shape[0]:
            raise ValueError(f"Index entry count {len(index_entries)} does not match vector rows {vectors.shape[0]}")

    manifest = parse_manifest(paths.manifest)
    manifest["_latest_view_inputs"] = {
        "s2_reviewed_units": reviewed_units_input,
        "graph_nodes": graph_nodes_input,
        "graph_edges": graph_edges_input,
    }
    return WorkspaceAssets(
        paths=paths,
        manifest=manifest,
        reviewed_units=read_jsonl(paths.reviewed_units),
        portrait_json=read_json(paths.portrait_json, default={}),
        graph_nodes=read_jsonl(paths.graph_nodes, required=False),
        graph_edges=read_jsonl(paths.graph_edges, required=False),
        base_packet=read_json(paths.base_packet, default={}),
        evidence=read_jsonl(paths.evidence, required=False) if paths.evidence else [],
        index_manifest=index_manifest,
        index_entries=index_entries,
        vectors=vectors,
    )


def load_questions(args: argparse.Namespace) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    if args.question:
        questions.append(
            {
                "query_id": args.query_id or "q001",
                "question": args.question,
                "route": args.route,
                "expected_answer_type": args.expected_answer_type,
                "reason": args.reason or "single CLI question",
                "needs_graph": args.needs_graph,
                "needs_temporal_filter": args.needs_temporal_filter,
            }
        )
    if args.questions:
        path = resolve_path(args.questions)
        assert path is not None
        if path.suffix.lower() == ".json":
            raw = read_json(path)
            questions.extend(raw if isinstance(raw, list) else raw.get("questions", []))
        else:
            questions.extend(read_jsonl(path))
    if not questions:
        raise ValueError("Provide --question or --questions. The generic runner has no built-in dataset questions.")

    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(questions, start=1):
        question = first_text(row, ["question", "query", "text"])
        if not question:
            raise ValueError(f"Question row {idx} is missing question/query/text")
        query_id = row.get("query_id") or row.get("id") or f"q{idx:03d}"
        route = str(row.get("route") or args.route or "hybrid")
        route_warnings = []
        if route not in KNOWN_ROUTES:
            route_warnings.append(f"unknown_route:{route}")
        metadata = row.get("metadata") or {}
        normalized.append(
            {
                "query_id": str(query_id),
                "question": question,
                "route": route,
                "expected_answer_type": str(row.get("expected_answer_type") or args.expected_answer_type or "unknown"),
                "reason": str(row.get("reason") or ""),
                "needs_graph": bool(row.get("needs_graph", args.needs_graph)),
                "needs_temporal_filter": bool(row.get("needs_temporal_filter", args.needs_temporal_filter)),
                "metadata": {**metadata, "route_warnings": route_warnings},
            }
        )
    return normalized


def make_query_unit(
    *,
    query_unit_id: str,
    artifact_type: str,
    artifact_ref: str,
    user_id: str,
    text: str,
    metadata: dict[str, Any],
    evidence_refs: list[str],
    source_refs: list[str],
    graph_refs: list[str],
    source_path: Path,
    status: str = "active",
) -> dict[str, Any]:
    return {
        "query_unit_id": query_unit_id,
        "artifact_type": artifact_type,
        "artifact_ref": artifact_ref,
        "user_id": user_id,
        "text_for_retrieval": text,
        "structured_metadata": metadata,
        "evidence_refs": evidence_refs,
        "source_refs": source_refs,
        "graph_refs": graph_refs,
        "source_path": str(source_path),
        "status": status,
        "query_eligible": status not in INACTIVE_STATUSES,
    }


def build_query_units(assets: WorkspaceAssets) -> list[dict[str, Any]]:
    workspace = assets.paths.workspace
    user_id = str(assets.base_packet.get("user_id") or assets.portrait_json.get("user_id") or assets.manifest.get("user_id") or workspace.name)
    units: list[dict[str, Any]] = []
    reviewed_by_id = {row.get("unit_id"): row for row in assets.reviewed_units if row.get("unit_id")}

    for row in assets.reviewed_units:
        unit_id = str(row.get("unit_id") or stable_id("unit", first_text(row, ["content", "summary"])[:50]))
        text = first_text(row, ["content", "summary", "evidence_summary", "text"])
        if not text:
            continue
        graph_refs = []
        for rel in listify(row.get("relation_candidates")):
            if isinstance(rel, dict):
                for key in ("target_node", "source_node", "edge_id"):
                    if rel.get(key):
                        graph_refs.append(str(rel[key]))
        metadata = {
            "memory_class": row.get("memory_class", "unknown"),
            "type": row.get("type", "unknown"),
            "scope": row.get("scope", "unknown"),
            "confidence": row.get("confidence", "unknown"),
            "inference_level": row.get("inference_level", "unknown"),
            "temporal_status": temporal_status(row),
            "privacy_class": row.get("privacy_class", "unknown"),
            "subject_contamination_risk": row.get("review_metadata", {}).get("contamination_risk", "unknown"),
        }
        units.append(
            make_query_unit(
                query_unit_id=stable_id("qu", unit_id),
                artifact_type="portrait_unit",
                artifact_ref=unit_id,
                user_id=str(row.get("user_id") or user_id),
                text=text,
                metadata=metadata,
                evidence_refs=evidence_refs_from(row),
                source_refs=source_refs_from(row),
                graph_refs=graph_refs,
                source_path=assets.paths.reviewed_units,
                status=status_of(row),
            )
        )

    for section, value in assets.portrait_json.items():
        if section in {"user_id", "schema_version"}:
            continue
        items = value if isinstance(value, list) else []
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            text = first_text(item, ["claim", "content", "summary", "question", "reason", "text"])
            if not text:
                continue
            ref = f"{section}:{idx}"
            metadata = {
                "memory_class": item.get("memory_class", "semantic"),
                "type": section,
                "scope": item.get("scope", "unknown"),
                "confidence": item.get("confidence", "unknown"),
                "inference_level": item.get("inference_level") or item.get("inference_type") or "unknown",
                "temporal_status": temporal_status(item),
                "privacy_class": item.get("privacy_class", "unknown"),
                "subject_contamination_risk": item.get("subject_contamination_risk", "unknown"),
            }
            units.append(
                make_query_unit(
                    query_unit_id=stable_id("qu:portrait", ref),
                    artifact_type="portrait_section",
                    artifact_ref=ref,
                    user_id=user_id,
                    text=text,
                    metadata=metadata,
                    evidence_refs=evidence_refs_from(item),
                    source_refs=source_refs_from(item),
                    graph_refs=[str(x) for x in listify(item.get("unit_ids"))],
                    source_path=assets.paths.portrait_json,
                    status=status_of(item),
                )
            )

    for row in assets.graph_nodes:
        node_id = str(row.get("node_id") or row.get("id") or stable_id("node", first_text(row, ["summary", "label"])[:50]))
        text = first_text(row, ["summary", "label", "text", "description"])
        if not text:
            continue
        metadata = {
            "memory_class": row.get("memory_class", "semantic"),
            "type": row.get("type", "graph_node"),
            "scope": row.get("scope", "relationship"),
            "confidence": row.get("confidence", "unknown"),
            "inference_level": row.get("inference_level", "unknown"),
            "temporal_status": temporal_status(row),
            "privacy_class": row.get("privacy_class", "unknown"),
            "subject_contamination_risk": row.get("subject_contamination_risk", "unknown"),
        }
        units.append(
            make_query_unit(
                query_unit_id=stable_id("qu:node", node_id),
                artifact_type="graph_node",
                artifact_ref=node_id,
                user_id=str(row.get("user_id") or user_id),
                text=text,
                metadata=metadata,
                evidence_refs=evidence_refs_from(row),
                source_refs=source_refs_from(row),
                graph_refs=[node_id],
                source_path=assets.paths.graph_nodes,
                status=status_of(row),
            )
        )

    for row in assets.graph_edges:
        edge_id = str(row.get("edge_id") or row.get("id") or stable_id("edge", first_text(row, ["summary", "label"])[:50]))
        text = first_text(row, ["summary", "label", "text", "description"])
        if not text:
            continue
        graph_refs = [edge_id]
        for key in ("source_node", "target_node", "source", "target"):
            if row.get(key):
                graph_refs.append(str(row[key]))
        metadata = {
            "memory_class": row.get("memory_class", "semantic"),
            "type": row.get("relation_type") or row.get("type") or "graph_edge",
            "scope": row.get("scope", "relationship"),
            "confidence": row.get("confidence", "unknown"),
            "inference_level": row.get("inference_level", "unknown"),
            "temporal_status": temporal_status(row),
            "privacy_class": row.get("privacy_class", "unknown"),
            "subject_contamination_risk": row.get("subject_contamination_risk", "unknown"),
        }
        units.append(
            make_query_unit(
                query_unit_id=stable_id("qu:edge", edge_id),
                artifact_type="graph_edge",
                artifact_ref=edge_id,
                user_id=str(row.get("user_id") or user_id),
                text=text,
                metadata=metadata,
                evidence_refs=evidence_refs_from(row),
                source_refs=source_refs_from(row),
                graph_refs=graph_refs,
                source_path=assets.paths.graph_edges,
                status=status_of(row),
            )
        )

    for group, value in assets.base_packet.items():
        if group in {"user_id", "packet_id", "schema_version"}:
            continue
        if not isinstance(value, list):
            continue
        for idx, item in enumerate(value, start=1):
            if not isinstance(item, dict):
                continue
            text = first_text(item, ["content", "claim", "summary", "guidance", "text"])
            if not text:
                continue
            unit_id = item.get("unit_id")
            source = reviewed_by_id.get(unit_id, {}) if unit_id else {}
            ref = f"{group}:{unit_id or idx}"
            metadata = {
                "memory_class": source.get("memory_class") or item.get("memory_class", "unknown"),
                "type": source.get("type") or item.get("type") or group,
                "scope": source.get("scope") or item.get("scope", "unknown"),
                "confidence": item.get("confidence") or source.get("confidence", "unknown"),
                "inference_level": item.get("inference_level") or source.get("inference_level", "unknown"),
                "temporal_status": temporal_status(source or item),
                "privacy_class": item.get("privacy_class") or source.get("privacy_class", "unknown"),
                "subject_contamination_risk": item.get("subject_contamination_risk", "unknown"),
            }
            units.append(
                make_query_unit(
                    query_unit_id=stable_id("qu:packet", ref),
                    artifact_type="packet_field",
                    artifact_ref=ref,
                    user_id=user_id,
                    text=text,
                    metadata=metadata,
                    evidence_refs=evidence_refs_from(item) or evidence_refs_from(source),
                    source_refs=source_refs_from(item) or source_refs_from(source),
                    graph_refs=[str(x) for x in listify(item.get("graph_refs"))],
                    source_path=assets.paths.base_packet,
                    status=status_of(item),
                )
            )
    return units


def build_vector_hit_maps(index_entries: list[dict[str, Any]], sims: np.ndarray | None) -> dict[str, dict[str, list[dict[str, Any]]]]:
    maps: dict[str, dict[str, list[dict[str, Any]]]] = {
        "query_unit_id": {},
        "object_id": {},
        "artifact_ref": {},
    }
    if sims is None:
        return maps
    for entry, raw_score in zip(index_entries, sims):
        embedding_raw = float(raw_score)
        hit = {
            "embedding_raw": embedding_raw,
            "embedding_norm": max(0.0, embedding_raw),
            "vector_entry_id": entry.get("vector_entry_id"),
            "source_object_hash": entry.get("source_object_hash"),
            "source_object_version": entry.get("source_object_version"),
            "entry": entry,
        }
        for field in ("query_unit_id", "object_id", "artifact_ref"):
            value = entry.get(field)
            if value:
                maps[field].setdefault(str(value), []).append(hit)
    return maps


def best_vector_hit_for(unit: dict[str, Any], maps: dict[str, dict[str, list[dict[str, Any]]]]) -> dict[str, Any] | None:
    alignment_attempts = [
        ("query_unit_id", unit.get("query_unit_id"), "exact_query_unit_id"),
        ("object_id", unit.get("artifact_ref"), "object_id_fallback"),
        ("artifact_ref", unit.get("artifact_ref"), "artifact_ref_fallback"),
    ]
    for map_name, key, alignment_method in alignment_attempts:
        if not key:
            continue
        hits = maps.get(map_name, {}).get(str(key), [])
        if not hits:
            continue
        best = max(hits, key=lambda row: row["embedding_raw"])
        result = dict(best)
        result["alignment_method"] = alignment_method
        result["all_vector_hits_count"] = len(hits)
        return result
    return None


def score_candidates(
    question: dict[str, Any],
    query_units: list[dict[str, Any]],
    assets: WorkspaceAssets,
    embedder: Embedder | None,
    bm25_index: BM25Index,
    args: argparse.Namespace,
    top_k: int,
    bm25_weight: float,
    embedding_weight: float,
    bm25_min_raw: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    question_text = question["question"]
    vector_maps = build_vector_hit_maps([], None)
    if embedder and assets.vectors is not None and assets.index_entries:
        q_vec = embedder.encode(question_text)
        sims = assets.vectors @ q_vec
        vector_maps = build_vector_hit_maps(assets.index_entries, sims)
    bm25_raw_scores = bm25_index.scores(question_text)
    max_bm25 = max(bm25_raw_scores) if bm25_raw_scores else 0.0
    bm25_is_usable = max_bm25 >= bm25_min_raw
    bm25_status = "usable" if bm25_is_usable else "diagnostic_only"

    candidates: list[dict[str, Any]] = []
    for idx, unit in enumerate(query_units):
        if not unit.get("query_eligible", True):
            continue
        object_id = str(unit["artifact_ref"])
        text = unit["text_for_retrieval"]
        bm25_raw = bm25_raw_scores[idx] if idx < len(bm25_raw_scores) else 0.0
        bm25_norm = (bm25_raw / max_bm25) if bm25_is_usable and max_bm25 > 0 else 0.0
        vector_hit = best_vector_hit_for(unit, vector_maps)
        embedding_raw = vector_hit["embedding_raw"] if vector_hit else None
        embedding_norm = vector_hit["embedding_norm"] if vector_hit else None
        if embedding_norm is not None:
            score = (embedding_weight * embedding_norm) + (bm25_weight * bm25_norm)
        elif embedding_weight > 0:
            score = bm25_weight * bm25_norm
        else:
            score = bm25_norm
        metadata = unit.get("structured_metadata", {})
        entry = vector_hit["entry"] if vector_hit else {}
        candidate_evidence_refs = unit.get("evidence_refs", []) or evidence_refs_from(entry)
        candidate_source_refs = unit.get("source_refs", []) or source_refs_from(entry)
        bucket = bucket_for_score(score, args)
        warnings = []
        if not candidate_evidence_refs:
            warnings.append("no_evidence_refs")
        if vector_hit is None:
            warnings.append("no_aligned_vector_entry")
        elif vector_hit["alignment_method"] != "exact_query_unit_id":
            warnings.append("vector_entry_aligned_by_object_id_fallback")
        if not bm25_is_usable:
            warnings.append("bm25_below_min_raw_threshold")
        candidates.append(
            {
                "candidate_id": stable_id("cand", unit["query_unit_id"]),
                "query_unit_id": unit["query_unit_id"],
                "object_type": unit["artifact_type"],
                "object_id": object_id,
                "text": text,
                "score": round(float(score), 6),
                "bucket": bucket,
                "score_components": {
                    "bm25": round(float(bm25_norm), 6),
                    "bm25_raw": round(float(bm25_raw), 6),
                    "embedding": round(float(embedding_norm), 6) if embedding_norm is not None else None,
                    "embedding_raw": round(float(embedding_raw), 6) if embedding_raw is not None else None,
                },
                "metadata": metadata,
                "evidence_refs": candidate_evidence_refs,
                "source_refs": candidate_source_refs,
                "graph_refs": unit.get("graph_refs", []),
                "source_path": unit.get("source_path"),
                "vector_entry_id": entry.get("vector_entry_id"),
                "source_object_hash": entry.get("source_object_hash"),
                "source_object_version": entry.get("source_object_version"),
                "vector_alignment": vector_hit.get("alignment_method") if vector_hit else None,
                "all_vector_hits_count": vector_hit.get("all_vector_hits_count") if vector_hit else 0,
                "warnings": warnings + listify(entry.get("warnings")),
            }
        )
    candidates.sort(key=lambda row: row["score"], reverse=True)
    diagnostics = {
        "bm25_status": bm25_status,
        "bm25_max_raw": max_bm25,
        "bm25_min_raw": bm25_min_raw,
        "bm25_document_count": len(query_units),
        "bm25_tokenizer_policy": bm25_index.manifest()["tokenizer_policy"],
        "fusion_weights": {
            "bm25": bm25_weight,
            "embedding": embedding_weight,
        },
    }
    return candidates[:top_k], diagnostics


def expand_graph_context_legacy_anchor(candidates: list[dict[str, Any]], assets: WorkspaceAssets, max_items: int) -> list[dict[str, Any]]:
    """Deprecated S2 graph expansion.

    This only expands around graph refs already attached to selected S2
    candidates. It is kept as a compatibility fallback for old workspaces that
    do not yet have v0.3 candidate graph tables.
    """

    anchor_refs: set[str] = set()
    for cand in candidates:
        anchor_refs.update(str(ref) for ref in cand.get("graph_refs", []) if ref)
        if cand["object_type"] in {"graph_node", "graph_edge"}:
            anchor_refs.add(str(cand["object_id"]))

    nodes = {str(row.get("node_id") or row.get("id")): row for row in assets.graph_nodes if row.get("node_id") or row.get("id")}
    edges = {str(row.get("edge_id") or row.get("id")): row for row in assets.graph_edges if row.get("edge_id") or row.get("id")}
    context: list[dict[str, Any]] = []

    for ref in sorted(anchor_refs):
        if ref in nodes:
            row = nodes[ref]
            context.append(
                {
                    "graph_context_ref": stable_id("graphctx:node", ref),
                    "kind": "node_hit",
                    "source_node_refs": [ref],
                    "source_edge_refs": [],
                    "summary": first_text(row, ["summary", "label", "text", "description"]),
                    "evidence_refs": evidence_refs_from(row),
                    "confidence": row.get("confidence", "unknown"),
                    "warnings": ["deprecated_legacy_anchor_graph_context"],
                }
            )
        if ref in edges:
            row = edges[ref]
            source = str(row.get("source_node") or row.get("source") or "")
            target = str(row.get("target_node") or row.get("target") or "")
            context.append(
                {
                    "graph_context_ref": stable_id("graphctx:edge", ref),
                    "kind": "edge_hit",
                    "source_node_refs": [x for x in [source, target] if x],
                    "source_edge_refs": [ref],
                    "summary": first_text(row, ["summary", "label", "text", "description"]),
                    "evidence_refs": evidence_refs_from(row),
                    "confidence": row.get("confidence", "unknown"),
                    "warnings": ["deprecated_legacy_anchor_graph_context"],
                }
            )

    for edge_id, row in edges.items():
        source = str(row.get("source_node") or row.get("source") or "")
        target = str(row.get("target_node") or row.get("target") or "")
        if source in anchor_refs or target in anchor_refs:
            context.append(
                {
                    "graph_context_ref": stable_id("graphctx:path", edge_id),
                    "kind": "path_expansion",
                    "source_node_refs": [x for x in [source, target] if x],
                    "source_edge_refs": [edge_id],
                    "summary": first_text(row, ["summary", "label", "text", "description"]),
                    "evidence_refs": evidence_refs_from(row),
                    "confidence": row.get("confidence", "unknown"),
                    "warnings": ["deprecated_legacy_anchor_graph_context"],
                }
            )

    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in context:
        key = item["graph_context_ref"]
        if key not in seen and item["summary"]:
            deduped.append(item)
            seen.add(key)
    return deduped[:max_items]


def discover_v03_graph_dir(workspace: Path, explicit: str | Path | None) -> Path | None:
    return discover_graph_dir(workspace, resolve_path(explicit) if explicit else None)


def v03_seed_terms(question: dict[str, Any], selected: list[dict[str, Any]]) -> list[str]:
    terms: list[str] = []
    for value in listify(question.get("seed_terms")):
        if isinstance(value, str) and value.strip() and value not in terms:
            terms.append(value.strip())
    query_tokens = tokenize(question.get("question", ""))
    latin_tokens = [token for token in query_tokens if LATIN_TOKEN_RE.fullmatch(token)]
    for left, right in zip(latin_tokens, latin_tokens[1:]):
        phrase = f"{left} {right}"
        if phrase not in terms:
            terms.append(phrase)
    for token in query_tokens:
        if len(token) > 2 and token not in terms:
            terms.append(token)
    for cand in selected[:4]:
        for value in (cand.get("text"),):
            text = str(value or "").strip()
            if text and len(text) <= 80 and text not in terms:
                terms.append(text)
    return terms[:12]


def graph_context_from_v03_package(package: dict[str, Any], max_items: int) -> list[dict[str, Any]]:
    graph_result = package.get("graph_branch", {})
    rows: list[dict[str, Any]] = []

    for item in graph_result.get("expanded_neighbors", []):
        edge_id = str(item.get("edge_id") or "")
        source = str(item.get("seed_node_id") or "")
        neighbor = str(item.get("neighbor_node_id") or "")
        if not edge_id:
            continue
        rows.append(
            {
                "graph_context_ref": stable_id("graphctx:v03edge", edge_id),
                "kind": "v03_relation_neighbor",
                "source_node_refs": [x for x in [source, neighbor] if x],
                "source_edge_refs": [edge_id],
                "summary": f"{item.get('seed_label', source)} --{item.get('relation_type', 'related_to')}--> {item.get('neighbor_label', neighbor)}",
                "evidence_refs": listify(item.get("evidence_refs")),
                "confidence": item.get("generic_relation_review_hint") or "candidate",
                "warnings": ["v03_graph_retrieval_candidate_not_proof"] + listify(item.get("warnings")),
                "retrieval_source": "v03_graph_retrieval",
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )

    for path_item in graph_result.get("evidence_paths", []):
        edge_refs: list[str] = []
        evidence_refs: list[str] = []
        for segment in path_item.get("segments", []):
            for ref in listify(segment.get("evidence_refs")):
                if ref not in evidence_refs:
                    evidence_refs.append(ref)
        rows.append(
            {
                "graph_context_ref": stable_id("graphctx:v03path", "|".join(path_item.get("path_node_ids", []))),
                "kind": "v03_evidence_path",
                "source_node_refs": listify(path_item.get("path_node_ids")),
                "source_edge_refs": edge_refs,
                "summary": " -> ".join(str(x) for x in path_item.get("path_labels", [])),
                "evidence_refs": evidence_refs,
                "confidence": "candidate_path",
                "warnings": ["v03_evidence_path_is_navigation_not_support_proof"],
                "retrieval_source": "v03_graph_retrieval",
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )

    for item in graph_result.get("matched_seed_nodes", []):
        node_id = str(item.get("node_id") or "")
        if not node_id:
            continue
        rows.append(
            {
                "graph_context_ref": stable_id("graphctx:v03seed", node_id),
                "kind": "v03_seed_node",
                "source_node_refs": [node_id],
                "source_edge_refs": [],
                "summary": f"{item.get('label', node_id)} ({item.get('entity_type', 'unknown')})",
                "evidence_refs": listify(item.get("evidence_refs")),
                "confidence": "candidate",
                "warnings": ["v03_graph_retrieval_candidate_not_proof"] + listify(item.get("warnings")),
                "retrieval_source": "v03_graph_retrieval",
                "graph_is_not_proof": True,
                "support_status": "not_checked",
                "write_permission": False,
            }
        )

    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = row["graph_context_ref"]
        if key not in seen and row.get("summary"):
            out.append(row)
            seen.add(key)
    return out[:max_items]


def build_v03_graph_retrieval_package(
    question: dict[str, Any],
    selected: list[dict[str, Any]],
    assets: WorkspaceAssets,
    args: argparse.Namespace,
    embedder: Embedder | None = None,
    graph_reranker: GraphReranker | None = None,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    graph_dir = discover_v03_graph_dir(assets.paths.workspace, args.v03_graph_dir)
    if graph_dir is None:
        return None, [], {
            "mode": "v03_graph_query_retriever",
            "status": "unavailable",
            "warnings": ["v03_graph_tables_not_found"],
        }
    return retrieve_graph_query_package(
        question,
        selected,
        graph_dir=graph_dir,
        profile_dir=resolve_path(args.v03_graph_profile_dir) if getattr(args, "v03_graph_profile_dir", None) else None,
        community_assists_path=resolve_path(args.v03_graph_community_assists) if getattr(args, "v03_graph_community_assists", None) else None,
        projection=args.v03_graph_projection,
        top_k=args.graph_top_k,
        graph_unit_embedder=embedder if getattr(args, "graph_unit_embedding_mode", "auto") in {"auto", "live"} else None,
        graph_rerank_mode=getattr(args, "graph_rerank_mode", "feature"),
        graph_reranker=graph_reranker,
        graph_rerank_alpha=getattr(args, "graph_rerank_alpha", 0.5),
        graph_reranker_batch_size=getattr(args, "graph_reranker_batch_size", 16),
        graph_community_mode=getattr(args, "graph_community_mode", "auto"),
    )


def build_graph_context(
    question: dict[str, Any],
    selected: list[dict[str, Any]],
    assets: WorkspaceAssets,
    args: argparse.Namespace,
    embedder: Embedder | None = None,
    graph_reranker: GraphReranker | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, dict[str, Any]]:
    mode = args.graph_context_mode
    if mode == "disabled" or not question.get("needs_graph"):
        return [], None, {"mode": mode, "status": "disabled", "warnings": []}

    if mode in {"auto", "v03"}:
        graph_embedder = embedder if getattr(args, "graph_unit_embedding_mode", "auto") in {"auto", "live"} else None
        package, context, status = build_v03_graph_retrieval_package(question, selected, assets, args, graph_embedder, graph_reranker)
        if status["status"] == "available" or mode == "v03":
            return context, package, status

    context = expand_graph_context_legacy_anchor(selected, assets, args.graph_top_k)
    return context, None, {
        "mode": "legacy_anchor_deprecated",
        "status": "available" if context else "empty",
        "context_count": len(context),
        "warnings": ["legacy_anchor_graph_context_deprecated_use_v03_graph_retrieval"],
    }


def adapt_resolved_ref_to_direct_evidence(resolved_ref: dict[str, Any], ref: str) -> dict[str, Any]:
    metadata = resolved_ref.get("metadata", {}) or {}
    return {
        "evidence_ref": ref,
        "resolved": bool(resolved_ref.get("resolved")),
        "text": resolved_ref.get("text", ""),
        "source_id": (resolved_ref.get("source_refs") or [metadata.get("source_id") or None])[0],
        "record_id": metadata.get("record_id"),
        "metadata": metadata,
        "warnings": listify(resolved_ref.get("resolution_warnings")),
        "resolution_source": "step1_toolbox",
    }


def resolve_evidence(
    evidence_refs: list[str],
    assets: WorkspaceAssets,
    step1_toolbox: LocalStep1Toolbox | None = None,
) -> list[dict[str, Any]]:
    if step1_toolbox is not None:
        return [
            adapt_resolved_ref_to_direct_evidence(
                step1_toolbox.resolve_ref(ref, "evidence_ref"),
                ref,
            )
            for ref in evidence_refs
        ]

    by_ref = {str(row.get("evidence_ref")): row for row in assets.evidence if row.get("evidence_ref")}
    resolved = []
    for ref in evidence_refs:
        row = by_ref.get(ref)
        resolved.append(
            {
                "evidence_ref": ref,
                "resolved": row is not None,
                "text": first_text(row or {}, ["text", "content", "utterance", "summary"]) if row else "",
                "source_id": (row or {}).get("source_id"),
                "record_id": (row or {}).get("record_id"),
                "metadata": (row or {}).get("metadata", {}),
                "warnings": [] if row else ["unresolved_evidence_ref"],
                "resolution_source": "local_metadata_resolution",
            }
        )
    return resolved


def ref_resolution_status(refs: list[str], resolved: list[dict[str, Any]]) -> str:
    resolved_count = sum(1 for row in resolved if row["resolved"])
    if refs and resolved_count == len(refs):
        return "direct_refs_resolved"
    if refs and resolved_count:
        return "partial_refs_resolved"
    if refs:
        return "unresolved_refs"
    return "unknown"


def make_evidence_checks(
    candidates: list[dict[str, Any]],
    assets: WorkspaceAssets,
    step1_toolbox: LocalStep1Toolbox | None = None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for cand in candidates:
        refs = cand.get("evidence_refs", [])
        resolved = resolve_evidence(refs, assets, step1_toolbox)
        ref_resolution = ref_resolution_status(refs, resolved)

        if step1_toolbox is not None:
            support_check = step1_toolbox.check_claim_support(
                cand["text"],
                refs,
                options={"default_ref_type": "evidence_ref"},
            )
            support_strength = support_check.get("support_strength", "unknown")
            checker = support_check.get("checker", {})
            warnings = listify(support_check.get("missing_refs")) + listify(support_check.get("notes"))
            if ref_resolution != "direct_refs_resolved":
                warnings.append("requires_resolved_evidence_refs_for_factual_answer")
            if support_strength in {"weak", "unknown"}:
                warnings.append("weak_or_unknown_step1_support")
            checks.append(
                {
                    "schema_version": "s2.evidence_check.v1",
                    "check_id": stable_id("support", cand["candidate_id"]),
                    "step1_support_check_id": support_check.get("check_id"),
                    "claim_or_context_text": cand["text"],
                    "source_candidate_refs": [cand["candidate_id"]],
                    "source_query_unit_refs": [cand["query_unit_id"]],
                    "evidence_refs": refs,
                    "resolved_evidence": resolved,
                    "ref_resolution_status": ref_resolution,
                    "support_strength": support_strength,
                    "support_checked": True,
                    "semantic_support_checked": False,
                    "support_check_source": "step1_toolbox_rule",
                    "support_check": support_check,
                    "checker": checker,
                    "warnings": [w for w in warnings if w],
                }
            )
            continue

        checks.append(
            {
                "schema_version": "s2.evidence_check.v1",
                "check_id": stable_id("support", cand["candidate_id"]),
                "claim_or_context_text": cand["text"],
                "source_candidate_refs": [cand["candidate_id"]],
                "source_query_unit_refs": [cand["query_unit_id"]],
                "evidence_refs": refs,
                "resolved_evidence": resolved,
                "ref_resolution_status": ref_resolution,
                "support_strength": "unknown",
                "support_checked": False,
                "semantic_support_checked": False,
                "support_check_source": "local_metadata_resolution_only",
                "checker": {
                    "type": "metadata_resolution",
                    "model_id": None,
                    "prompt_or_policy": "Resolve candidate evidence_refs only; no semantic entailment judgement.",
                },
                "warnings": [] if ref_resolution == "direct_refs_resolved" else ["requires_step1_claim_support_for_factual_answer"],
            }
        )
    return checks


def build_pruned_context_candidates(candidates: list[dict[str, Any]], selected_ids: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cand in candidates:
        bucket = cand.get("bucket", "excluded")
        selected = cand["candidate_id"] in selected_ids
        if selected:
            prune_status = "selected"
            use_policy = "candidate_context"
        elif bucket == "excluded":
            prune_status = "excluded"
            use_policy = "do_not_use"
        else:
            prune_status = "not_selected"
            use_policy = "audit_only"
        rows.append(
            {
                "context_ref": stable_id("ctx", cand["candidate_id"]),
                "source_candidate_ref": cand["candidate_id"],
                "source_query_unit_ref": cand["query_unit_id"],
                "object_type": cand["object_type"],
                "object_id": cand["object_id"],
                "score": cand["score"],
                "bucket": bucket,
                "prune_status": prune_status,
                "use_policy": use_policy,
                "evidence_refs": cand.get("evidence_refs", []),
                "graph_refs": cand.get("graph_refs", []),
                "metadata": cand.get("metadata", {}),
                "warnings": cand.get("warnings", []),
            }
        )
    return rows


def build_query_packet_quality(
    question: dict[str, Any],
    selected: list[dict[str, Any]],
    graph_context: list[dict[str, Any]],
    evidence_checks: list[dict[str, Any]],
    step1_retrieval_status: dict[str, Any],
    query_claim_support_check: dict[str, Any],
    packet: dict[str, Any],
    retrieval_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    route_warnings = question.get("metadata", {}).get("route_warnings", [])
    unresolved_refs = packet.get("uncertainty", {}).get("unresolved_evidence_refs", [])
    missing_fields = []
    for field in ("packet_id", "query", "selected_context", "evidence_checks", "uncertainty"):
        if field not in packet:
            missing_fields.append(field)

    support_strengths = [check.get("support_strength", "unknown") for check in evidence_checks]
    support_checked = [check for check in evidence_checks if check.get("support_checked")]
    if not selected:
        evidence_status = "not_applicable"
        evidence_notes = "No selected context required evidence checking."
    elif len(support_checked) != len(evidence_checks):
        evidence_status = "degraded"
        evidence_notes = "Some selected context only has metadata ref resolution, not Step 1 support checks."
    elif any(strength in {"direct", "partial", "contradicts"} for strength in support_strengths):
        evidence_status = "pass"
        evidence_notes = "Step 1 toolbox support checks ran for selected context; at least one candidate has direct/partial/contradictory support signal."
    else:
        evidence_status = "degraded"
        evidence_notes = "Step 1 toolbox support checks ran, but selected context has only weak/unknown support."
    if not selected:
        retrieval_status = "fail"
        retrieval_notes = "No selected context met the minimum score threshold."
    elif retrieval_diagnostics.get("bm25_status") == "diagnostic_only":
        retrieval_status = "degraded"
        retrieval_notes = "Selected context exists, but BM25 was diagnostic-only for this query."
    else:
        retrieval_status = "pass"
        retrieval_notes = "Selected context exists and BM25 was usable for this query."

    graph_status = "not_applicable"
    graph_notes = "Graph context was not requested."
    if question.get("needs_graph"):
        graph_status = "pass" if graph_context else "degraded"
        graph_notes = "Anchor-based graph context expansion only; not full graph search."

    packet_status = "pass" if not missing_fields and selected else "degraded"
    handoff_status = "ready"
    handoff_notes = "Ready for Step 2.3 with cautions; Step 1 support checks are rule-based and not a robust semantic judge."
    if missing_fields or unresolved_refs:
        handoff_status = "degraded"
    if not selected:
        handoff_status = "blocked"
        handoff_notes = "No selected context is available for Step 2.3."

    claim_support_strength = query_claim_support_check.get("support_strength", "unknown")
    if question.get("expected_answer_type") == "fact" and claim_support_strength in {"weak", "unknown", "contradicts"}:
        handoff_status = "degraded" if handoff_status == "ready" else handoff_status

    return {
        "schema_version": "s2.query_packet_quality.v1",
        "query_id": question["query_id"],
        "route_quality": {
            "status": "degraded" if route_warnings else "pass",
            "notes": "; ".join(route_warnings) if route_warnings else "Route is recognized by the current runner contract.",
        },
        "retrieval_quality": {
            "status": retrieval_status,
            "notes": retrieval_notes,
        },
        "graph_context_quality": {
            "status": graph_status,
            "notes": graph_notes,
        },
        "evidence_check_quality": {
            "status": evidence_status,
            "notes": evidence_notes,
        },
        "step1_retrieval_quality": {
            "status": step1_retrieval_status.get("status", "unknown"),
            "retrieval_mode": step1_retrieval_status.get("retrieval_mode"),
            "active_result_modes": step1_retrieval_status.get("active_result_modes", []),
            "result_count": step1_retrieval_status.get("result_count", 0),
            "warnings": step1_retrieval_status.get("warnings", []),
        },
        "query_claim_support_quality": {
            "status": query_claim_support_check.get("support_status", "unknown"),
            "support_strength": claim_support_strength,
            "notes": "This evaluates the user query claim itself; it is separate from selected context support checks.",
            "warnings": query_claim_support_check.get("warnings", []),
        },
        "packet_completeness": {
            "status": packet_status,
            "missing_or_weak_fields": missing_fields,
        },
        "metadata_handoff": {
            "status": "degraded" if unresolved_refs else "pass",
            "missing_fields": [],
            "unresolved_refs": unresolved_refs,
            "unpropagated_warnings": [],
        },
        "handoff_readiness": {
            "status": handoff_status,
            "required_step2_3_inputs": [
                "dynamic_assistance_packet.json",
                "evidence_checks.jsonl",
                "graph_context.jsonl",
                "pruned_context_candidates.jsonl",
                "route_decision.json",
                "query_packet_quality.json",
            ],
            "notes": handoff_notes,
        },
        "known_limitations": [
            "Evidence checks use Step 1 toolbox when available; current local checker is rule-based, not robust semantic entailment.",
            "Query-claim support check is rule-based and conservative; unknown means not established, not necessarily false.",
            "Graph context is anchor-based expansion only, not real graph search.",
            "BM25 is in-memory for this run, not a persistent lexical index workflow.",
        ],
    }


def positive_score(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0.0


def candidate_has_risk(row: dict[str, Any]) -> bool:
    warnings = {str(warning) for warning in row.get("warnings", [])}
    bucket = str(row.get("bucket") or "")
    item_layer = str(row.get("item_layer") or row.get("source_object_type") or "")
    object_type = str(row.get("object_type") or "")
    subject_status = str(row.get("subject_match_status") or "")
    confidence = row.get("confidence")
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    if bucket in {"weak_context", "excluded"}:
        return True
    if subject_status in {"mixed_participant", "other_participant"}:
        return True
    if item_layer == "doc_level_summary" or object_type in {"summary", "portrait_section"}:
        return True
    if "summary_hit_context_only" in warnings or "memory_unit_not_raw_evidence" in warnings:
        return True
    if "memory_summary_without_backpointer" in warnings or "unresolved_ref" in warnings:
        return True
    if any("subject" in warning or "risk" in warning or "unresolved" in warning for warning in warnings):
        return True
    if confidence is None:
        confidence = metadata.get("confidence")
    return isinstance(confidence, (int, float)) and float(confidence) < 0.5


def branch_projection(row: dict[str, Any], rank: int) -> dict[str, Any]:
    text = row.get("text") or row.get("content") or ""
    score_components = row.get("score_components", {})
    return {
        "rank": rank,
        "candidate_id": row.get("candidate_id") or row.get("result_id") or row.get("id"),
        "query_unit_id": row.get("query_unit_id"),
        "result_id": row.get("result_id"),
        "object_type": row.get("object_type") or row.get("source_object_type"),
        "object_id": row.get("object_id") or row.get("source_object_ref") or row.get("memory_id"),
        "item_layer": row.get("item_layer") or row.get("source_object_type"),
        "score": row.get("score"),
        "bucket": row.get("bucket"),
        "retrieval_method": row.get("retrieval_method") or row.get("method"),
        "retrieval_mode": row.get("retrieval_mode"),
        "retrieval_backend": row.get("retrieval_backend"),
        "support_status": row.get("support_status", "not_checked"),
        "subject_match_status": row.get("subject_match_status"),
        "subject_risk_warning": row.get("subject_risk_warning"),
        "evidence_refs": row.get("evidence_refs", []),
        "source_refs": row.get("source_refs", []),
        "backpointer_refs": row.get("backpointer_refs", []),
        "warnings": row.get("warnings", []),
        "score_components": score_components,
        "text": short_text(text),
    }


def classify_branch_candidates(rows: list[dict[str, Any]], *, source: str, selected_ids: set[str] | None = None) -> dict[str, Any]:
    bm25_candidates: list[dict[str, Any]] = []
    embedding_candidates: list[dict[str, Any]] = []
    overlap_candidates: list[dict[str, Any]] = []
    bm25_only_candidates: list[dict[str, Any]] = []
    embedding_only_candidates: list[dict[str, Any]] = []
    conflict_or_risk_candidates: list[dict[str, Any]] = []
    summary_context_candidates: list[dict[str, Any]] = []
    memory_unit_candidates: list[dict[str, Any]] = []
    raw_evidence_candidates: list[dict[str, Any]] = []

    for rank, row in enumerate(rows, start=1):
        score_components = row.get("score_components", {})
        bm25_hit = positive_score(score_components.get("bm25")) or positive_score(score_components.get("bm25_normalized_score"))
        embedding_hit = positive_score(score_components.get("embedding")) or positive_score(score_components.get("embedding_normalized_score"))
        active_backends = {str(backend) for backend in row.get("active_backends", [])}
        backend = str(row.get("retrieval_backend") or row.get("retrieval_method") or "")
        if "bm25" in active_backends or backend == "bm25":
            bm25_hit = True
        if "embedding" in active_backends or backend == "embedding":
            embedding_hit = True
        if backend == "hybrid":
            bm25_hit = bm25_hit or "bm25" in active_backends
            embedding_hit = embedding_hit or "embedding" in active_backends

        projected = branch_projection(row, rank)
        if bm25_hit:
            bm25_candidates.append(projected)
        if embedding_hit:
            embedding_candidates.append(projected)
        if bm25_hit and embedding_hit:
            overlap_candidates.append(projected)
        elif bm25_hit:
            bm25_only_candidates.append(projected)
        elif embedding_hit:
            embedding_only_candidates.append(projected)

        item_layer = str(row.get("item_layer") or row.get("source_object_type") or "")
        object_type = str(row.get("object_type") or "")
        warnings = {str(warning) for warning in row.get("warnings", [])}
        if item_layer == "doc_level_summary" or "summary_hit_context_only" in warnings or object_type in {"summary", "portrait_section"}:
            summary_context_candidates.append(projected)
        if item_layer == "memory_unit" or "memory_unit_not_raw_evidence" in warnings or object_type == "memory_unit":
            memory_unit_candidates.append(projected)
        if item_layer == "raw_evidence" or object_type == "raw_evidence":
            raw_evidence_candidates.append(projected)
        if candidate_has_risk(row):
            conflict_or_risk_candidates.append(projected)

    result = {
        "source": source,
        "bm25_candidates": bm25_candidates,
        "embedding_candidates": embedding_candidates,
        "overlap_candidates": overlap_candidates,
        "bm25_only_candidates": bm25_only_candidates,
        "embedding_only_candidates": embedding_only_candidates,
        "conflict_or_risk_candidates": conflict_or_risk_candidates,
        "summary_context_candidates": summary_context_candidates,
        "memory_unit_candidates": memory_unit_candidates,
        "support_status": "not_checked",
    }
    if raw_evidence_candidates:
        result["raw_evidence_candidates"] = raw_evidence_candidates
    if selected_ids is not None:
        result["selected_context_ids"] = sorted(selected_ids)
        result["excluded_context_ids"] = sorted(
            str(row.get("candidate_id"))
            for row in rows
            if row.get("candidate_id") and str(row.get("candidate_id")) not in selected_ids
        )
    return result


def derive_recommended_use_policy(query_claim_support_check: dict[str, Any]) -> dict[str, Any]:
    return {
        "factual_answer": {
            "prefer": "overlap candidates and BM25 raw-evidence candidates with checked support",
            "avoid": "embedding-only candidates as fact",
            "support_status": query_claim_support_check.get("support_status", "not_checked"),
            "support_strength": query_claim_support_check.get("support_strength", "unknown"),
        },
        "semantic_exploration": {
            "allow": "embedding-only candidates",
            "require": "subject-risk warning and later support check before durable claims",
        },
        "portrait_update": {
            "require": "multiple evidence refs plus S2.1 review path",
        },
        "summary_context": {
            "use": "background context only, not direct support",
        },
        "memory_unit": {
            "use": "evidence-bound substrate; preserve evidence refs and backpointers",
        },
        "subject_risky": {
            "use": "context only, not target-subject fact",
        },
    }


def build_dual_branch_retrieval_package(
    question: dict[str, Any],
    candidates: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    step1_retrieval_results: list[dict[str, Any]],
    step1_retrieval_status: dict[str, Any],
    query_claim_support_check: dict[str, Any],
    retrieval_diagnostics: dict[str, Any],
    weights: dict[str, Any],
) -> dict[str, Any]:
    selected_ids = {str(row["candidate_id"]) for row in selected}
    step2_branch = classify_branch_candidates(candidates, source="step2_user_model", selected_ids=selected_ids)
    step1_branch = classify_branch_candidates(step1_retrieval_results, source="step1_evidence")
    step1_branch["support_status"] = query_claim_support_check.get("support_status", "not_checked")
    step1_branch["query_claim_support_check"] = query_claim_support_check

    warnings = sorted(
        {
            "retrieval_hit_not_support_check",
            "hybrid_is_reference_view_not_only_contract",
            *[str(warning) for row in candidates for warning in row.get("warnings", [])],
            *[str(warning) for row in step1_retrieval_results for warning in row.get("warnings", [])],
            *[str(warning) for warning in step1_retrieval_status.get("warnings", [])],
            *[str(warning) for warning in query_claim_support_check.get("warnings", [])],
        }
    )

    return {
        "schema_version": "s2.dual_branch_retrieval_package.v1",
        "query": {
            "query_id": question["query_id"],
            "question": question["question"],
            "route": question.get("route"),
            "expected_answer_type": question.get("expected_answer_type"),
        },
        "query_intent_guess": {
            "route": question.get("route"),
            "needs_graph": question.get("needs_graph"),
            "needs_temporal_filter": question.get("needs_temporal_filter"),
        },
        "step1_evidence_retrieval": step1_branch,
        "step2_user_model_retrieval": step2_branch,
        "graph_branch": {
            "graph_candidates": [],
            "graph_status": "disabled",
            "graph_reason": "graph_search_not_implemented_in_v0_1",
        },
        "analysis_layer": {
            "s3_status": "not_invoked",
            "feedback_repair_status": "manual_future",
        },
        "recommended_use_policy": derive_recommended_use_policy(query_claim_support_check),
        "warnings": warnings,
        "retrieval_trace": {
            "step2_candidate_count": len(candidates),
            "step2_selected_count": len(selected),
            "step2_excluded_count": len(excluded),
            "step1_result_count": len(step1_retrieval_results),
            "step1_retrieval_status": step1_retrieval_status,
            "retrieval_diagnostics": retrieval_diagnostics,
            "fusion_weights": weights,
            "notes": [
                "BM25 and embedding branches are exposed before final interpretation.",
                "Hybrid/fusion scores are ranking views, not support checks.",
                "Graph search and Step 3 are explicit disabled/not-invoked branches in v0.1.",
            ],
        },
        "support_status": "not_checked",
    }


def packet_for_query(
    question: dict[str, Any],
    selected: list[dict[str, Any]],
    excluded: list[dict[str, Any]],
    graph_context: list[dict[str, Any]],
    evidence_checks: list[dict[str, Any]],
    step1_retrieval_results: list[dict[str, Any]],
    step1_retrieval_status: dict[str, Any],
    query_claim_support_check: dict[str, Any],
    retrieval_diagnostics: dict[str, Any],
    dual_branch_package: dict[str, Any],
    assets: WorkspaceAssets,
    args: argparse.Namespace,
    step1_toolbox: LocalStep1Toolbox | None,
    step1_status: dict[str, Any],
) -> dict[str, Any]:
    evidence_refs: list[str] = []
    for cand in selected:
        for ref in cand.get("evidence_refs", []):
            if ref not in evidence_refs:
                evidence_refs.append(ref)
    direct_evidence = resolve_evidence(evidence_refs, assets, step1_toolbox)
    unresolved = [row["evidence_ref"] for row in direct_evidence if not row["resolved"]]
    warnings = list(step1_status.get("warnings", []))
    if step1_toolbox is None:
        warnings.append("s1_toolbox_not_available_metadata_resolution_only")
    if unresolved:
        warnings.append("unresolved_evidence_refs_present")
    if args.lexical_only:
        warnings.append("embedding_disabled")
    if not selected:
        warnings.append("no_strong_context_found")
    if query_claim_support_check.get("support_strength") in {"weak", "unknown", "contradicts"}:
        warnings.append("query_claim_not_directly_supported_by_step1_check")

    return {
        "schema_version": "s2.dynamic_assistance_packet.v1",
        "packet_id": stable_id("s2packet", f"{question['query_id']}:{datetime.now(timezone.utc).timestamp()}"),
        "generated_at": utc_now(),
        "workspace": str(assets.paths.workspace),
        "query": question,
        "retrieval_methods": {
            "bm25": True,
            "bm25_status": retrieval_diagnostics["bm25_status"],
            "bm25_max_raw": retrieval_diagnostics["bm25_max_raw"],
            "bm25_min_raw": retrieval_diagnostics["bm25_min_raw"],
            "bm25_document_count": retrieval_diagnostics["bm25_document_count"],
            "bm25_tokenizer_policy": retrieval_diagnostics["bm25_tokenizer_policy"],
            "embedding": not args.lexical_only,
            "graph_expansion": bool(question.get("needs_graph", True)),
            "step1_support_check": "step1_toolbox_rule" if step1_toolbox else "metadata_resolution_only",
            "step1_toolbox": step1_status,
            "step1_retrieval": step1_retrieval_status,
            "fusion_weights": retrieval_diagnostics["fusion_weights"],
        },
        "step1_retrieval_results": step1_retrieval_results,
        "query_claim_support_check": query_claim_support_check,
        "dual_branch_retrieval_package_ref": "dual_branch_retrieval_package.json",
        "dual_branch_summary": dual_branch_summary(dual_branch_package),
        "selected_context": selected,
        "selected_portrait_units": [row for row in selected if row["object_type"] == "portrait_unit"],
        "selected_graph_context": graph_context,
        "direct_evidence": direct_evidence,
        "evidence_checks": evidence_checks,
        "excluded_context": excluded,
        "uncertainty": {
            "unresolved_evidence_refs": unresolved,
            "low_support_candidates": [row["candidate_id"] for row in selected if not row.get("evidence_refs")],
            "notes": [
                "Vector similarity is retrieval signal only.",
                "Use evidence_checks and resolved evidence before making factual claims.",
                "Selected context support does not imply the user query claim is supported; check query_claim_support_check separately.",
            ],
            "no_strong_context_found": not selected,
        },
        "answer_guidance": {
            "policy": "Use this packet as constrained memory context, not as a final answer.",
            "must_not": [
                "Do not cite vector score as evidence.",
                "Do not turn graph/path context into proof without evidence refs.",
                "Do not assert candidates with unresolved refs as facts.",
            ],
            "suggested_structure": ["direct evidence", "user-model context", "graph context", "uncertainty"],
        },
        "warnings": warnings,
    }


def interpret_packet(packet: dict[str, Any]) -> dict[str, Any]:
    allowed_claims: list[dict[str, Any]] = []
    cautious_claims: list[dict[str, Any]] = []
    forbidden_claims: list[dict[str, Any]] = []
    context_with_resolved_refs: list[dict[str, Any]] = []

    check_by_candidate: dict[str, dict[str, Any]] = {}
    for check in packet.get("evidence_checks", []):
        for cand_ref in check.get("source_candidate_refs", []):
            check_by_candidate[cand_ref] = check

    for cand in packet.get("selected_context", []):
        check = check_by_candidate.get(cand["candidate_id"], {})
        support = check.get("support_strength", "unknown")
        ref_resolution = check.get("ref_resolution_status", "unknown")
        item = {
            "candidate_id": cand["candidate_id"],
            "object_type": cand["object_type"],
            "object_id": cand["object_id"],
            "text": cand["text"],
            "support": support,
            "ref_resolution_status": ref_resolution,
            "support_checked": check.get("support_checked", False),
            "support_check_source": check.get("support_check_source", "unknown"),
            "semantic_support_checked": check.get("semantic_support_checked", False),
            "evidence_refs": cand.get("evidence_refs", []),
            "metadata": cand.get("metadata", {}),
        }
        if check.get("support_checked") and support == "direct":
            allowed_claims.append(item)
        elif check.get("support_checked") and support == "partial":
            cautious_claims.append(item)
        elif ref_resolution in {"direct_refs_resolved", "partial_refs_resolved"}:
            context_with_resolved_refs.append(item)
            cautious_claims.append(item)
        elif support == "unknown":
            cautious_claims.append(item)
        else:
            forbidden_claims.append(item)

    return {
        "schema_version": "s2.memory_context_interpretation.v1",
        "source_packet_id": packet["packet_id"],
        "query": packet["query"],
        "interpretation_status": "ready_with_cautions" if packet.get("warnings") else "ready",
        "allowed_claims": allowed_claims,
        "cautious_claims": cautious_claims,
        "context_with_resolved_refs": context_with_resolved_refs,
        "forbidden_claims": forbidden_claims,
        "graph_context": packet.get("selected_graph_context", []),
        "direct_evidence": packet.get("direct_evidence", []),
        "step1_retrieval_context": packet.get("step1_retrieval_results", []),
        "dual_branch_retrieval_package_ref": packet.get("dual_branch_retrieval_package_ref", "dual_branch_retrieval_package.json"),
        "dual_branch_summary": packet.get("dual_branch_summary", {}),
        "query_claim_support_check": packet.get("query_claim_support_check", {}),
        "uncertainty": packet.get("uncertainty", {}),
        "answer_policy": {
            "factual_claims_require": "semantic support check; ref resolution alone is not factual support",
            "portrait_context_use": "Use as user-model context, not raw proof.",
            "graph_context_use": "Use for relationship/navigation context, not standalone proof.",
            "step1_retrieval_use": "Use Step 1 retrieval as evidence/context candidates; retrieval hits are not support checks.",
            "final_answer": "Generated by the caller/agent using this constrained context.",
        },
    }


def short_text(text: str, limit: int = 220) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def render_context_markdown(packet: dict[str, Any], interpretation: dict[str, Any]) -> str:
    lines = [
        "# Usable Memory Context",
        "",
        f"- Query ID: `{packet['query']['query_id']}`",
        f"- Question: {packet['query']['question']}",
        f"- Route: `{packet['query'].get('route', 'hybrid')}`",
        f"- Source packet: `{packet['packet_id']}`",
        "",
        "## Answer Contract",
        "",
        "- This file is context for a final agent answer, not the final answer.",
        "- Use resolved evidence for factual claims.",
        "- Treat portrait and graph items as user-model context unless evidence checks support them.",
        "- Treat Step 1 retrieval hits as evidence/context candidates, not as support checks.",
        "- Do not cite vector scores as evidence.",
        "",
        "## Direct Evidence",
        "",
    ]
    direct_evidence = packet.get("direct_evidence", [])
    if direct_evidence:
        for row in direct_evidence:
            status = "resolved" if row.get("resolved") else "unresolved"
            text = row.get("text") or ""
            lines.append(f"- `{row['evidence_ref']}` ({status}): {text}")
    else:
        lines.append("- No direct evidence refs selected.")

    lines.extend(["", "## Query Claim Support", ""])
    query_support = interpretation.get("query_claim_support_check") or packet.get("query_claim_support_check", {})
    if query_support:
        refs = ", ".join(query_support.get("evidence_refs", [])) or "no raw evidence refs"
        warnings = ", ".join(query_support.get("warnings", [])) or "none"
        lines.append(f"- Status: `{query_support.get('support_status', 'unknown')}`")
        lines.append(f"- Support strength: `{query_support.get('support_strength', 'unknown')}`")
        lines.append(f"- Checked evidence refs: {refs}")
        lines.append(f"- Warnings: {warnings}")
        if query_support.get("support_strength") in {"weak", "unknown", "contradicts"}:
            lines.append("- Use policy: do not assert the query claim as a settled fact from Step 1 alone.")
    else:
        lines.append("- No query-claim support check available.")

    lines.extend(["", "## Dual-Branch Retrieval Package", ""])
    dual_summary = interpretation.get("dual_branch_summary") or packet.get("dual_branch_summary", {})
    if dual_summary:
        s1_branch = dual_summary.get("step1_evidence_retrieval", {})
        s2_branch = dual_summary.get("step2_user_model_retrieval", {})
        lines.append("- Retrieval hits are candidates, not support checks.")
        lines.append(
            "- Step 1 branch: "
            f"BM25={s1_branch.get('bm25_candidates', 0)}, "
            f"embedding={s1_branch.get('embedding_candidates', 0)}, "
            f"overlap={s1_branch.get('overlap_candidates', 0)}, "
            f"risk={s1_branch.get('conflict_or_risk_candidates', 0)}."
        )
        lines.append(
            "- Step 2 branch: "
            f"BM25={s2_branch.get('bm25_candidates', 0)}, "
            f"embedding={s2_branch.get('embedding_candidates', 0)}, "
            f"overlap={s2_branch.get('overlap_candidates', 0)}, "
            f"risk={s2_branch.get('conflict_or_risk_candidates', 0)}."
        )
        lines.append(
            f"- Graph branch: `{dual_summary.get('graph_status', 'unknown')}`."
        )
        if interpretation.get("dual_branch_retrieval_package_ref") or packet.get("dual_branch_retrieval_package_ref"):
            lines.append(
                f"- Audit package: `{interpretation.get('dual_branch_retrieval_package_ref') or packet.get('dual_branch_retrieval_package_ref')}`."
            )
    else:
        lines.append("- No dual-branch retrieval package provided.")

    lines.extend(["", "## Step 1 Query Evidence Context", ""])
    step1_results = interpretation.get("step1_retrieval_context") or packet.get("step1_retrieval_results", [])
    if step1_results:
        lines.append("- These are Step 1 retrieval candidates. They help evidence discipline but are not final support checks.")
        for row in step1_results[:8]:
            refs = ", ".join(row.get("evidence_refs", [])) or "no refs"
            warnings = ", ".join(row.get("warnings", [])) or "none"
            score = row.get("score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else str(score or "unknown")
            layer = row.get("item_layer") or row.get("source_object_type") or "unknown"
            subject = row.get("subject_match_status") or "unknown"
            support_status = row.get("support_status") or "not_checked"
            method = row.get("retrieval_method") or row.get("method") or "unknown"
            text = short_text(row.get("text", ""))
            use_policy = "raw evidence candidate" if layer == "raw_evidence" else "context only unless backpointers are checked"
            lines.append(
                f"- rank {row.get('rank', '?')} [{layer}] via `{method}` score={score_text}; "
                f"subject={subject}; support={support_status}; use={use_policy}; refs={refs}; warnings={warnings}; text: {text}"
            )
    else:
        lines.append("- No Step 1 retrieval candidates provided.")

    lines.extend(["", "## Allowed Claims", ""])
    if interpretation.get("allowed_claims"):
        for item in interpretation["allowed_claims"]:
            refs = ", ".join(item.get("evidence_refs", [])) or "no refs"
            lines.append(f"- [{item['object_type']}] {item['text']} (refs: {refs})")
    else:
        lines.append("- None.")

    lines.extend(["", "## Cautious Claims", ""])
    if interpretation.get("cautious_claims"):
        for item in interpretation["cautious_claims"]:
            refs = ", ".join(item.get("evidence_refs", [])) or "no refs"
            lines.append(f"- [{item['object_type']}] {item['text']} (support: {item['support']}; refs: {refs})")
    else:
        lines.append("- None.")

    lines.extend(["", "## Graph Context", ""])
    if interpretation.get("graph_context"):
        for item in interpretation["graph_context"]:
            refs = ", ".join(item.get("evidence_refs", [])) or "no refs"
            lines.append(f"- `{item['kind']}`: {item['summary']} (refs: {refs})")
    else:
        lines.append("- None.")

    lines.extend(["", "## Uncertainty", ""])
    for note in packet.get("uncertainty", {}).get("notes", []):
        lines.append(f"- {note}")
    for warning in packet.get("warnings", []):
        lines.append(f"- Warning: {warning}")
    return "\n".join(lines)


def run_one_query(
    question: dict[str, Any],
    assets: WorkspaceAssets,
    query_units: list[dict[str, Any]],
    bm25_index: BM25Index,
    embedder: Embedder | None,
    graph_reranker: GraphReranker | None,
    args: argparse.Namespace,
    step1_toolbox: LocalStep1Toolbox | None,
    step1_status: dict[str, Any],
) -> dict[str, Any]:
    out_dir = args.output / question["query_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "query_manifest.json", {"query": question, "workspace": str(assets.paths.workspace), "generated_at": utc_now()})
    route_decision = {
        "query_id": question["query_id"],
        "route": question["route"],
        "expected_answer_type": question["expected_answer_type"],
        "needs_graph": question["needs_graph"],
        "needs_temporal_filter": question["needs_temporal_filter"],
        "risk_flags": [
            "vector_similarity_is_not_truth",
            "step1_checker_is_rule_based" if step1_toolbox else "metadata_resolution_only",
        ],
    }
    write_json(out_dir / "route_decision.json", route_decision)
    write_jsonl(out_dir / "step2_query_units.jsonl", query_units)
    write_json(out_dir / "step2_bm25_manifest.json", bm25_index.manifest())

    weights = resolve_fusion_weights(question, args)
    candidates, retrieval_diagnostics = score_candidates(
        question,
        query_units,
        assets,
        embedder,
        bm25_index,
        args,
        args.candidate_pool_size,
        weights["bm25"],
        weights["embedding"],
        args.bm25_min_raw,
    )
    selected = [row for row in candidates if row["bucket"] != "excluded"][: args.top_k]
    selected_ids = {row["candidate_id"] for row in selected}
    excluded = [row for row in candidates if row["candidate_id"] not in selected_ids]
    graph_context, graph_retrieval_package, graph_context_status = build_graph_context(question, selected, assets, args, embedder, graph_reranker)
    evidence_checks = make_evidence_checks(selected, assets, step1_toolbox)
    step1_retrieval_results, step1_retrieval_status = run_step1_retrieval(question, step1_toolbox, args)
    query_claim_support_check = make_query_claim_support_check(question, step1_retrieval_results, step1_toolbox)
    dual_branch_package = build_dual_branch_retrieval_package(
        question,
        candidates,
        selected,
        excluded,
        step1_retrieval_results,
        step1_retrieval_status,
        query_claim_support_check,
        retrieval_diagnostics,
        weights,
    )
    if graph_retrieval_package is not None:
        dual_branch_package["graph_retrieval_package_ref"] = "graph_retrieval_package.json"
    dual_branch_package["graph_context_status"] = graph_context_status
    packet = packet_for_query(
        question,
        selected,
        excluded,
        graph_context,
        evidence_checks,
        step1_retrieval_results,
        step1_retrieval_status,
        query_claim_support_check,
        retrieval_diagnostics,
        dual_branch_package,
        assets,
        args,
        step1_toolbox,
        step1_status,
    )
    pruned_context = build_pruned_context_candidates(candidates, selected_ids)
    packet_quality = build_query_packet_quality(
        question,
        selected,
        graph_context,
        evidence_checks,
        step1_retrieval_status,
        query_claim_support_check,
        packet,
        retrieval_diagnostics,
    )
    interpretation = interpret_packet(packet)
    s23_answer_context = build_s23_answer_context(
        query_dir=out_dir,
        dual_package=dual_branch_package,
        interpretation=interpretation,
        selected_context=selected,
        step2_candidates=candidates,
        graph_context=graph_context,
        graph_retrieval_package=graph_retrieval_package,
    )

    write_jsonl(out_dir / "step2_candidates.jsonl", candidates)
    write_jsonl(out_dir / "step1_retrieval_results.jsonl", step1_retrieval_results)
    write_json(out_dir / "step1_retrieval_status.json", step1_retrieval_status)
    write_json(out_dir / "query_claim_support_check.json", query_claim_support_check)
    write_json(out_dir / "dual_branch_retrieval_package.json", dual_branch_package)
    if graph_retrieval_package is not None:
        write_json(out_dir / "graph_retrieval_package.json", graph_retrieval_package)
    write_jsonl(out_dir / "selected_context.jsonl", selected)
    write_jsonl(out_dir / "excluded_context.jsonl", excluded)
    write_jsonl(out_dir / "pruned_context_candidates.jsonl", pruned_context)
    write_jsonl(out_dir / "graph_context.jsonl", graph_context)
    write_json(out_dir / "graph_answer_context.json", s23_answer_context.get("graph_answer_context", {}))
    write_jsonl(out_dir / "evidence_checks.jsonl", evidence_checks)
    write_json(out_dir / "dynamic_assistance_packet.json", packet)
    write_json(out_dir / "query_packet_quality.json", packet_quality)
    write_json(out_dir / "memory_context_interpretation.json", interpretation)
    write_json(out_dir / "s23_answer_context.json", s23_answer_context)
    write_text(out_dir / "s23_prompt_context.md", render_s23_prompt_context(s23_answer_context))
    write_text(out_dir / "usable_memory_context.md", render_context_markdown(packet, interpretation))
    write_text(
        out_dir / "run-report.md",
        "\n".join(
            [
                f"# Query Run Report: {question['query_id']}",
                "",
                f"- Question: {question['question']}",
                f"- Route: {question['route']}",
                f"- Selected context: {len(selected)}",
                f"- Graph context: {len(graph_context)}",
                f"- Graph mode: {graph_context_status.get('mode')} / {graph_context_status.get('status')}",
                f"- Evidence checks: {len(evidence_checks)}",
                f"- Query packet quality: {packet_quality['handoff_readiness']['status']}",
                f"- BM25: {retrieval_diagnostics['bm25_status']}, max_raw={retrieval_diagnostics['bm25_max_raw']:.6f}, tokenizer={bm25_index.manifest()['tokenizer_policy']}",
                f"- Fusion weights: bm25={weights['bm25']}, embedding={weights['embedding']} ({weights['source']})",
                f"- Step 1 retrieval: {step1_retrieval_status.get('retrieval_mode')} / {step1_retrieval_status.get('active_result_modes')} / results={step1_retrieval_status.get('result_count', 0)}",
                f"- Query claim support: {query_claim_support_check.get('support_status')} / {query_claim_support_check.get('support_strength')}",
                f"- Dual-branch package: `dual_branch_retrieval_package.json`.",
                f"- Step 2.3 answer context artifact: `s23_answer_context.json`.",
                f"- Step 2.3 prompt context: `s23_prompt_context.md`.",
                f"- Default final-answer prompt should read `s23_prompt_context.md`; open answer/audit JSON only for debug.",
                f"- Graph: {graph_context_status.get('mode')} candidate context, not proof.",
                f"- Step 1 support: {'toolbox rule checker' if step1_toolbox else 'metadata resolution only'}.",
                f"- Output is a query/context package, not a fixed final answer.",
            ]
        ),
    )
    return {
        "query_id": question["query_id"],
        "question": question["question"],
        "output_dir": str(out_dir),
        "selected_context_count": len(selected),
        "graph_context_count": len(graph_context),
        "graph_context_mode": graph_context_status.get("mode"),
        "graph_context_status": graph_context_status.get("status"),
        "evidence_check_count": len(evidence_checks),
        "top_candidate": selected[0] if selected else None,
        "retrieval_diagnostics": retrieval_diagnostics,
        "step1_retrieval_status": step1_retrieval_status,
        "query_claim_support_check": {
            "support_status": query_claim_support_check.get("support_status"),
            "support_strength": query_claim_support_check.get("support_strength"),
            "evidence_ref_count": len(query_claim_support_check.get("evidence_refs", [])),
        },
        "fusion_weights": weights,
        "query_packet_quality": packet_quality["handoff_readiness"]["status"],
        "packet_path": str(out_dir / "dynamic_assistance_packet.json"),
        "dual_branch_package_path": str(out_dir / "dual_branch_retrieval_package.json"),
        "graph_retrieval_package_path": str(out_dir / "graph_retrieval_package.json") if graph_retrieval_package is not None else None,
        "s23_answer_context_path": str(out_dir / "s23_answer_context.json"),
        "s23_prompt_context_path": str(out_dir / "s23_prompt_context.md"),
        "usable_context_path": str(out_dir / "usable_memory_context.md"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic Step 2 query runner over existing S2.1 assets.")
    parser.add_argument("--workspace", required=True, help="Step 2.1 user workspace path.")
    parser.add_argument("--output", default=None, help="Output run directory. Defaults to <workspace>/queries/s2-query-run-<timestamp>.")
    parser.add_argument("--questions", default=None, help="JSONL or JSON question file.")
    parser.add_argument("--question", default=None, help="Single ad-hoc question.")
    parser.add_argument("--query-id", default=None, help="ID for --question.")
    parser.add_argument("--route", default="hybrid", help="Route label for missing question route.")
    parser.add_argument("--expected-answer-type", default="unknown")
    parser.add_argument("--reason", default="")
    parser.add_argument("--needs-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--needs-temporal-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--embedding-backend",
        default="qwen_local",
        choices=["qwen_local", "hash"],
        help="Embedding backend for query vectors. hash is deterministic and intended for integration smoke only.",
    )
    parser.add_argument("--hash-dimension", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lexical-only", action="store_true", help="Disable embedding query.")
    parser.add_argument("--bm25-weight", type=float, default=None, help="Explicit BM25 fusion weight. If omitted, route defaults are used.")
    parser.add_argument("--embedding-weight", type=float, default=None, help="Explicit embedding fusion weight. If omitted, route defaults are used.")
    parser.add_argument(
        "--bm25-min-raw",
        type=float,
        default=0.05,
        help="If max BM25 raw score is below this threshold, BM25 is treated as diagnostic-only for that query.",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--candidate-pool-size", type=int, default=24)
    parser.add_argument("--min-score", type=float, default=0.10, help="Candidates below this score are excluded from selected_context.")
    parser.add_argument("--direct-threshold", type=float, default=0.55, help="Score threshold for direct_context bucket.")
    parser.add_argument("--supporting-threshold", type=float, default=0.25, help="Score threshold for supporting_context bucket.")
    parser.add_argument("--graph-top-k", type=int, default=12)
    parser.add_argument(
        "--graph-unit-embedding-mode",
        default="auto",
        choices=["auto", "live", "disabled"],
        help=(
            "Optional graph-unit embedding branch for v0.3 graph retrieval. auto uses the same Step 2 "
            "embedder when embeddings are enabled; lexical-only runs disable it."
        ),
    )
    parser.add_argument(
        "--graph-rerank-mode",
        default="feature",
        choices=["disabled", "feature", "cross_encoder", "auto"],
        help=(
            "Rerank v0.3 graph retrieval candidates after RRF. feature uses auditable non-model quality "
            "signals; cross_encoder loads --graph-reranker-model and blends it with feature/RRF signals; "
            "auto uses cross_encoder only when a reranker is already loaded by this run."
        ),
    )
    parser.add_argument("--graph-reranker-model", default="BAAI/bge-reranker-base")
    parser.add_argument("--graph-reranker-device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--graph-reranker-batch-size", type=int, default=16)
    parser.add_argument(
        "--graph-rerank-alpha",
        type=float,
        default=0.5,
        help="Blend weight for cross-encoder score when --graph-rerank-mode cross_encoder is used.",
    )
    parser.add_argument(
        "--graph-community-mode",
        default="auto",
        choices=["auto", "disabled"],
        help="Enable or disable the v0.3 community/profile activation branch.",
    )
    parser.add_argument(
        "--graph-context-mode",
        default="auto",
        choices=["auto", "v03", "legacy_anchor", "disabled"],
        help=(
            "Graph context mode. auto prefers v0.3 candidate graph retrieval and falls back to the deprecated "
            "legacy anchor expansion. legacy_anchor is kept only for old workspaces."
        ),
    )
    parser.add_argument("--v03-graph-dir", default=None, help="Optional v0.3 consolidated candidate graph directory.")
    parser.add_argument("--v03-graph-profile-dir", default=None, help="Optional v0.3 graph profile/community directory.")
    parser.add_argument("--v03-graph-community-assists", default=None, help="Optional v0.3 community report assist overlay dir or JSONL file.")
    parser.add_argument("--v03-graph-projection", default="review_aware_graph", choices=["full_candidate_graph", "review_aware_graph", "stable_core_graph"])
    parser.add_argument("--reviewed-units", default=None)
    parser.add_argument("--portrait-json", default=None)
    parser.add_argument("--graph-nodes", default=None)
    parser.add_argument("--graph-edges", default=None)
    parser.add_argument("--base-packet", default=None)
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--latest-view-mode", default="auto", choices=["auto", "require", "off"])
    parser.add_argument("--step1-root", default=None, help="Optional Step 1 asset root. Defaults to the workspace or workspace/step1 when available.")
    parser.add_argument("--disable-step1-toolbox", action="store_true", help="Disable Step 1 toolbox support checks and use local metadata resolution only.")
    parser.add_argument(
        "--step1-retrieval-mode",
        default="auto",
        choices=["disabled", "direct_scan", "auto", "bm25_indexed", "embedding_indexed", "hybrid_indexed", "degraded_direct_scan"],
        help="Step 1 toolbox retrieval mode for query evidence comparison. Defaults to auto, which prefers hybrid indexed retrieval when fresh BM25 and embedding indexes are available. This does not replace Step 2 retrieval.",
    )
    parser.add_argument("--step1-index-root", default=None, help="Optional Step 1 index root for indexed retrieval modes.")
    parser.add_argument("--step1-retrieval-top-k", type=int, default=8)
    parser.add_argument("--step1-include-memory-units", action="store_true")
    parser.add_argument("--step1-include-summaries", action="store_true")
    parser.add_argument("--step1-embedding-device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--step1-allow-hash-embedding-for-tests", action="store_true")
    parser.add_argument("--target-subject-id", default=None, help="Target subject for S1 subject-risk metadata.")
    parser.add_argument("--index-manifest", default=None)
    parser.add_argument("--index-entries", default=None)
    parser.add_argument("--index-vectors", default=None)
    args = parser.parse_args()

    workspace = resolve_path(args.workspace)
    assert workspace is not None
    if args.output:
        args.output = resolve_path(args.output)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = workspace / "queries" / f"s2-query-run-{timestamp}"
    return args


def main() -> None:
    args = parse_args()
    assets = load_workspace(args)
    step1_toolbox, step1_status = load_step1_toolbox(args, assets.paths.workspace)
    questions = load_questions(args)
    query_units = build_query_units(assets)
    if not query_units:
        raise RuntimeError("No query units could be built from the Step 2.1 workspace.")
    bm25_index = BM25Index(query_units)

    embedder = None
    if not args.lexical_only:
        if args.embedding_backend == "hash":
            manifest_embedding = assets.index_manifest.get("embedding") if isinstance(assets.index_manifest, dict) else {}
            dimension = (
                args.hash_dimension
                or int(manifest_embedding.get("embedding_dim") or 0)
                or (assets.vectors.shape[1] if assets.vectors is not None and assets.vectors.ndim == 2 else 64)
            )
            embedder = HashEmbedder(dimension)
        else:
            embedder = Embedder(args.model_id, args.device)

    graph_reranker = None
    if args.graph_rerank_mode == "cross_encoder":
        graph_reranker = GraphReranker(args.graph_reranker_model, args.graph_reranker_device)

    args.output.mkdir(parents=True, exist_ok=True)
    run_results = [
        run_one_query(question, assets, query_units, bm25_index, embedder, graph_reranker, args, step1_toolbox, step1_status)
        for question in questions
    ]
    summary = {
        "schema_version": "s2.query_runner.summary.v1",
        "generated_at": utc_now(),
        "runner": str(Path(__file__).resolve()),
        "workspace": str(assets.paths.workspace),
        "output": str(args.output),
        "python": sys.executable,
        "model_id": None if args.lexical_only else (HASH_MODEL_ID if args.embedding_backend == "hash" else args.model_id),
        "embedding_backend": "disabled" if args.lexical_only else args.embedding_backend,
        "lexical_only": args.lexical_only,
        "graph_rerank": {
            "mode": args.graph_rerank_mode,
            "model_id": args.graph_reranker_model if graph_reranker is not None else None,
            "device": graph_reranker.device if graph_reranker is not None else None,
            "alpha": args.graph_rerank_alpha,
        },
        "graph_community": {
            "mode": args.graph_community_mode,
            "profile_dir": str(resolve_path(args.v03_graph_profile_dir)) if args.v03_graph_profile_dir else None,
            "assists": str(resolve_path(args.v03_graph_community_assists)) if args.v03_graph_community_assists else None,
        },
        "bm25": bm25_index.manifest(),
        "fusion_weights": {
            "mode": "lexical_only" if args.lexical_only else ("cli" if args.bm25_weight is not None or args.embedding_weight is not None else "route_based"),
            "bm25_cli": args.bm25_weight,
            "embedding_cli": args.embedding_weight,
        },
        "query_unit_count": len(query_units),
        "question_count": len(questions),
        "step1_toolbox": step1_status,
        "selection_thresholds": {
            "min_score": args.min_score,
            "supporting_threshold": args.supporting_threshold,
            "direct_threshold": args.direct_threshold,
        },
        "runs": run_results,
        "notes": [
            "This runner is generic over S2.1 assets and question files.",
            "It does not hard-code dataset-specific answers.",
            "It emits query/context packages; final natural-language answers are caller responsibility.",
            "BM25 is enabled as lexical retrieval infrastructure, not as a truth source.",
        ],
    }
    write_json(args.output / "run-summary.json", summary)
    write_text(
        args.output / "run-summary.md",
        "\n".join(
            [
                "# Step 2 Query Runner Summary",
                "",
                f"- Workspace: `{assets.paths.workspace}`",
                f"- Output: `{args.output}`",
                f"- Questions: {len(questions)}",
                f"- Query units: {len(query_units)}",
                f"- BM25: enabled, tokenizer `{bm25_index.manifest()['tokenizer_policy']}`",
                f"- Embedding: {'disabled' if args.lexical_only else (HASH_MODEL_ID if args.embedding_backend == 'hash' else args.model_id)}",
                f"- Step 1 toolbox: {step1_status.get('status')} ({step1_status.get('root')})",
                "",
                "## Runs",
                "",
                *[
                    f"- `{row['query_id']}`: selected={row['selected_context_count']}, graph={row['graph_context_count']}, context=`{row['usable_context_path']}`"
                    for row in run_results
                ],
            ]
        ),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
