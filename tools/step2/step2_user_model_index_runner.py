"""Build Step 2 user-model embedding indexes from canonical Step 2 assets.

This runner implements the Step 2 Build module 12. It builds rebuildable query
infrastructure only; it does not create canonical memory, graph truth, evidence
support, query packets, or final answers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from tools.maintenance.latest_view import resolve_latest_view_input


DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
HASH_MODEL = "deterministic-hash-embedding-v0.1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_simple_yaml_fields(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    fields: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        value = value.strip().strip("\"'")
        if key and value and not value.startswith("{") and not value.startswith("["):
            fields[key.strip()] = value
    return fields


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def optional_file_sha256(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        return None
    return f"sha256:{file_sha256(path)}"


def object_hash(obj: dict[str, Any]) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def safe_model_name(model: str) -> str:
    return model.replace("/", "__").replace("\\", "__")


def hash_embedding(text: str, dimension: int) -> np.ndarray:
    tokens = [token for token in text.lower().replace("\n", " ").split(" ") if token] or [text]
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


def entry_text(unit: dict[str, Any]) -> str:
    temporal_scope = unit.get("temporal_scope") or {}
    review_metadata = unit.get("review_metadata") or {}
    parts = [
        f"Content: {unit.get('content', '')}",
        f"Type: {unit.get('type', '')}",
        f"Memory class: {unit.get('memory_class', '')}",
        f"Scope: {unit.get('scope', '')}",
        f"Temporal: {unit.get('temporal_status') or temporal_scope.get('validity', '')}",
        f"Confidence: {unit.get('confidence', '')}",
        f"Inference level: {unit.get('inference_level', '')}",
        f"Evidence summary: {unit.get('evidence_summary', '')}",
        f"Review notes: {review_metadata.get('notes', '') if isinstance(review_metadata, dict) else ''}",
    ]
    return "\n".join(part for part in parts if part.strip() and not part.endswith(": "))


def build_entries(units: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for idx, unit in enumerate(units):
        unit_id = unit.get("unit_id")
        if not unit_id:
            raise ValueError("reviewed portrait unit missing unit_id")
        status = unit.get("status") or unit.get("review_status") or "active"
        review_metadata = unit.get("review_metadata") or {}
        if status in {"rejected", "archived", "superseded"}:
            continue
        entries.append(
            {
                "vector_entry_id": f"s2vec:{unit_id}",
                "row_index": idx,
                "object_type": "portrait_unit",
                "object_id": unit_id,
                "source_object_version": unit.get("schema_version", "s2.reviewed_portrait_unit.v1"),
                "source_object_hash": object_hash(unit),
                "text": entry_text(unit),
                "source_refs": list(unit.get("source_refs") or []),
                "evidence_refs": list(unit.get("evidence_refs") or []),
                "backpointer_refs": list(unit.get("backpointer_refs") or []),
                "graph_refs": list(unit.get("graph_refs") or unit.get("relation_candidates") or []),
                "memory_class": unit.get("memory_class", "unknown"),
                "scope": unit.get("scope", "unknown"),
                "temporal_status": unit.get("temporal_status") or (unit.get("temporal_scope") or {}).get("validity", "unknown"),
                "temporal_scope": unit.get("temporal_scope", {}),
                "confidence": unit.get("confidence", "unknown"),
                "inference_level": unit.get("inference_level", "unknown"),
                "privacy_class": unit.get("privacy_class", "unknown"),
                "subject_contamination_risk": (
                    review_metadata.get("contamination_risk")
                    or unit.get("subject_contamination_risk")
                    or "unknown"
                ),
                "status": "active" if status == "accepted_for_experiment" else status,
                "step1_origin": unit.get("step1_origin", {}),
                "warnings": list(unit.get("warnings") or []),
            }
        )
    return entries


def resolve_workspace_identity(workspace: Path, args: argparse.Namespace, units: list[dict[str, Any]]) -> tuple[str, str]:
    workspace_manifest = read_simple_yaml_fields(workspace / "manifest.yaml")
    current_portrait_path = workspace / "portrait" / "current_portrait.json"
    current_portrait = read_json(current_portrait_path) if current_portrait_path.exists() else {}
    workspace_id = args.workspace_id or workspace_manifest.get("workspace_id") or workspace.name
    modeled_user_id = (
        args.modeled_user_id
        or workspace_manifest.get("modeled_user_id")
        or current_portrait.get("user_id")
        or next((str(unit.get("user_id")) for unit in units if unit.get("user_id")), "")
    )
    return workspace_id, modeled_user_id


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    import torch

    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def encode_qwen_texts(
    *,
    model_name: str,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device_name: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer

    started = time.perf_counter()
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left", local_files_only=True, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, local_files_only=True, trust_remote_code=True).to(device)
    model.eval()
    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = model(**inputs)
            pooled = last_token_pool(outputs.last_hidden_state, inputs["attention_mask"])
            pooled = F.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.detach().cpu().float().numpy().astype("float32"))
    embeddings = np.vstack(vectors) if vectors else np.zeros((0, 0), dtype="float32")
    return embeddings, {
        "model": model_name,
        "model_class": model.__class__.__name__,
        "embedding_dim": int(embeddings.shape[1]) if embeddings.size else 0,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "device": str(device),
        "max_length": max_length,
        "pooling": "last_token",
    }


def encode_hash_texts(*, texts: list[str], dimension: int) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.perf_counter()
    vectors = np.vstack([hash_embedding(text, dimension) for text in texts]) if texts else np.zeros((0, dimension), dtype="float32")
    return vectors.astype("float32"), {
        "model": HASH_MODEL,
        "model_class": "deterministic_hash_embedding",
        "embedding_dim": dimension,
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "device": "not_applicable",
        "max_length": "not_applicable",
        "pooling": "not_applicable",
        "note": "Deterministic hash backend is for integration smoke only; it is not semantic retrieval quality proof.",
    }


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    workspace = Path(args.workspace).resolve()
    output_path = Path(args.output_path).resolve() if args.output_path else workspace / "indexes"
    reviewed_units_path, reviewed_units_input = resolve_latest_view_input(
        workspace=workspace,
        layer="s2",
        canonical_path=workspace / "portrait" / "reviewed_units.jsonl",
        explicit_path=Path(args.s2_latest_view).resolve() if getattr(args, "s2_latest_view", None) else None,
        mode=getattr(args, "latest_view_mode", "auto"),
    )
    if not reviewed_units_path.exists():
        raise FileNotFoundError(f"Missing reviewed units: {reviewed_units_path}")

    output_path.mkdir(parents=True, exist_ok=True)
    generic_manifest = output_path / "step2_user_model_embedding_manifest.json"
    generic_entries = output_path / "step2_user_model_embedding_entries.jsonl"
    generic_vectors = output_path / "step2_user_model_embedding_vectors.npy"
    existing = [path for path in [generic_manifest, generic_entries, generic_vectors] if path.exists()]
    if existing and args.duplicate_policy == "fail":
        raise FileExistsError(f"Step 2 embedding index already exists: {existing}")

    units = read_jsonl(reviewed_units_path)
    workspace_id, modeled_user_id = resolve_workspace_identity(workspace, args, units)
    status_path = workspace / "checkpoints" / "phase2-status.json"
    build_status = read_json(status_path) if status_path.exists() else {}
    prebuild_routing = build_status.get("prebuild_routing") if isinstance(build_status, dict) else None
    proposal_lineage = None
    if isinstance(prebuild_routing, dict) and isinstance(prebuild_routing.get("proposal"), dict):
        proposal_outputs = prebuild_routing["proposal"].get("outputs") or {}
        proposal_lineage = {
            "proposal_run_id": prebuild_routing["proposal"].get("proposal_run_id"),
            "proposal_profile_id": prebuild_routing["proposal"].get("proposal_profile_id"),
            "provider": prebuild_routing["proposal"].get("provider"),
            "proposal_outcomes": proposal_outputs.get("proposals"),
            "proposal_run_manifest": proposal_outputs.get("proposal_run_manifest"),
            "proposal_outcomes_hash": optional_file_sha256(proposal_outputs.get("proposals")),
            "proposal_run_manifest_hash": optional_file_sha256(proposal_outputs.get("proposal_run_manifest")),
        }
    entries = build_entries(units)
    if not entries:
        raise ValueError("No active reviewed units to index.")
    embedding_backend = getattr(args, "embedding_backend", "qwen_local")
    hash_dimension = int(getattr(args, "hash_dimension", 64))
    if embedding_backend == "hash":
        vectors, embedding_meta = encode_hash_texts(texts=[row["text"] for row in entries], dimension=hash_dimension)
        model_name = HASH_MODEL
    elif embedding_backend == "qwen_local":
        vectors, embedding_meta = encode_qwen_texts(
            model_name=args.model,
            texts=[row["text"] for row in entries],
            batch_size=args.batch_size,
            max_length=args.max_length,
            device_name=args.device,
        )
        model_name = args.model
    else:
        raise ValueError(f"Unsupported embedding backend: {embedding_backend}")

    write_jsonl(generic_entries, entries)
    np.save(generic_vectors, vectors)

    manifest = {
        "schema_version": "s2.embedding_index_manifest.v1",
        "index_id": f"step2_user_model_embedding__{safe_model_name(model_name)}__{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "index_type": "step2_user_model_embedding",
        "truth_status": "rebuildable_query_asset_not_truth",
        "workspace_id": workspace_id,
        "modeled_user_id": modeled_user_id,
        "built_from": {
            "reviewed_units": str(reviewed_units_path),
        },
        "latest_view_inputs": {
            "s2_reviewed_units": reviewed_units_input,
        },
        "upstream_lineage": {
            "step2_build_status": str(status_path) if status_path.exists() else None,
            "build_source": build_status.get("build_source") if isinstance(build_status, dict) else None,
            "prebuild_proposal": proposal_lineage,
        },
        "source_hashes": {
            "reviewed_units": f"sha256:{file_sha256(reviewed_units_path)}",
        },
        "source_asset_versions": {
            "reviewed_units": "s2.reviewed_portrait_unit.v1",
        },
        "index_scope": {
            "asset_types": ["portrait_unit"],
            "included_statuses": ["active", "accepted_for_experiment"],
            "excluded_statuses": ["rejected", "archived", "superseded"],
        },
        "embedding_backend": embedding_backend,
        "embedding_model": model_name,
        "embedding_model_revision": "",
        "chunking_policy": "one_vector_per_reviewed_portrait_unit",
        "max_length": args.max_length,
        "metadata_policy": "preserve_object_refs_evidence_refs_scope_temporal_confidence_inference_memory_class_warnings",
        "entries_path": str(generic_entries),
        "vectors_path": str(generic_vectors),
        "entry_count": len(entries),
        "embedding": embedding_meta,
        "built_at": datetime.now(timezone.utc).isoformat(),
        "builder": "tools/step2/step2_user_model_index_runner.py",
        "status": "active",
        "build_status": "completed",
        "warnings": [
            "This index is Step 2 query infrastructure, not truth.",
            "Vector similarity must not replace evidence refs or support checks.",
        ],
    }
    write_json(generic_manifest, manifest)
    if isinstance(build_status, dict) and status_path.exists():
        completed_steps = list(build_status.get("completed_steps") or [])
        if "step2_user_model_index" not in completed_steps:
            completed_steps.append("step2_user_model_index")
        latest_outputs = dict(build_status.get("latest_outputs") or {})
        latest_outputs.update(
            {
                "step2_user_model_embedding_manifest": "indexes/step2_user_model_embedding_manifest.json",
                "step2_user_model_embedding_entries": "indexes/step2_user_model_embedding_entries.jsonl",
                "step2_user_model_embedding_vectors": "indexes/step2_user_model_embedding_vectors.npy",
            }
        )
        build_status.update(
            {
                "completed_steps": completed_steps,
                "latest_outputs": latest_outputs,
                "step2_user_model_index": {
                    "status": "completed",
                    "truth_status": manifest["truth_status"],
                    "index_id": manifest["index_id"],
                    "entry_count": manifest["entry_count"],
                    "manifest": latest_outputs["step2_user_model_embedding_manifest"],
                    "entries": latest_outputs["step2_user_model_embedding_entries"],
                    "vectors": latest_outputs["step2_user_model_embedding_vectors"],
                    "source_hashes": manifest["source_hashes"],
                },
                "last_updated": manifest["built_at"],
            }
        )
        write_json(status_path, build_status)

    if args.write_model_suffix:
        suffix = safe_model_name(model_name)
        write_json(output_path / f"step2_user_model_embedding_manifest.{suffix}.json", manifest)
        write_jsonl(output_path / f"step2_user_model_embedding_entries.{suffix}.jsonl", entries)
        np.save(output_path / f"step2_user_model_embedding_vectors.{suffix}.npy", vectors)

    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--workspace-id", default=None)
    parser.add_argument("--modeled-user-id", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--embedding-backend", default="qwen_local", choices=["qwen_local", "hash"])
    parser.add_argument("--hash-dimension", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--duplicate-policy", default="fail", choices=["fail", "overwrite_generated"])
    parser.add_argument("--write-model-suffix", action="store_true")
    parser.add_argument("--latest-view-mode", default="auto", choices=["auto", "require", "off"])
    parser.add_argument("--s2-latest-view", default=None, help="Explicit S2 latest-view JSONL. Defaults to maintenance/latest_views/s2_latest_view.jsonl when present.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_index(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
