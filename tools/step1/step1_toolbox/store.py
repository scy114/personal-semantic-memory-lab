"""Local JSON/JSONL store helpers for Step 1 toolbox."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_existing(root: Path, candidates: list[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return root / candidates[0]


class LocalStep1Store:
    """Read-only local store for Step 1 fixture/canonical assets."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.source_manifest_path = first_existing(
            self.root,
            [
                "source_manifest.jsonl",
                "evidence/source_manifest.jsonl",
            ],
        )
        self.evidence_path = first_existing(
            self.root,
            [
                "evidence.jsonl",
                "evidence/evidence.jsonl",
            ],
        )
        self.memory_units_path = first_existing(
            self.root,
            [
                "memory_units.jsonl",
                "memory/memory_units.jsonl",
            ],
        )
        self.summaries_path = first_existing(
            self.root,
            [
                "summaries.jsonl",
                "memory/summaries.jsonl",
            ],
        )
        self.raw_sources_path = first_existing(
            self.root,
            [
                "raw/organization/raw_sources.jsonl",
                "organization/raw_sources.jsonl",
            ],
        )
        self.raw_bundle_path = first_existing(
            self.root,
            [
                "raw/organization/bundle.json",
                "organization/bundle.json",
            ],
        )
        self.sources = read_jsonl(self.source_manifest_path)
        self.evidence = read_jsonl(self.evidence_path)
        self.memory_units = read_jsonl(self.memory_units_path)
        self.summaries = read_jsonl(self.summaries_path)
        self.raw_sources = read_jsonl(self.raw_sources_path)
        self.raw_bundle = read_json(self.raw_bundle_path)

        self._evidence_by_ref: dict[str, dict[str, Any]] = {}
        for item in self.evidence:
            for key in (
                "evidence_ref",
                "canonical_evidence_ref",
                "source_specific_ref",
                "display_ref",
            ):
                ref = item.get(key)
                if isinstance(ref, str) and ref:
                    self._evidence_by_ref.setdefault(ref, item)
            for ref in item.get("ref_aliases", []) or []:
                if isinstance(ref, str) and ref:
                    self._evidence_by_ref.setdefault(ref, item)
        self._memory_by_id = {
            item.get("memory_id"): item
            for item in self.memory_units
            if item.get("memory_id")
        }
        self._summary_by_id = {
            item.get("summary_id"): item
            for item in self.summaries
            if item.get("summary_id")
        }
        self._source_by_id = {
            item.get("source_id"): item
            for item in self.sources
            if item.get("source_id")
        }
        self._raw_source_by_id = {
            item.get("raw_source_id"): item
            for item in self.raw_sources
            if item.get("raw_source_id")
        }

    def resolve_evidence(self, evidence_ref: str) -> dict[str, Any] | None:
        return self._evidence_by_ref.get(evidence_ref)

    def resolve_memory(self, memory_id: str) -> dict[str, Any] | None:
        return self._memory_by_id.get(memory_id)

    def resolve_summary(self, summary_id: str) -> dict[str, Any] | None:
        return self._summary_by_id.get(summary_id)

    def resolve_source(self, source_id: str) -> dict[str, Any] | None:
        return self._source_by_id.get(source_id)

    def resolve_raw_source(self, raw_source_id: str) -> dict[str, Any] | None:
        return self._raw_source_by_id.get(raw_source_id)

    def source_for_evidence(self, evidence_item: dict[str, Any]) -> dict[str, Any] | None:
        source_id = evidence_item.get("source_id")
        if not source_id:
            return None
        return self.resolve_source(source_id)

    def raw_source_for_evidence(self, evidence_item: dict[str, Any]) -> dict[str, Any] | None:
        raw_source_id = evidence_item.get("raw_source_id")
        if raw_source_id:
            return self.resolve_raw_source(raw_source_id)
        source = self.source_for_evidence(evidence_item) or {}
        source_raw_id = source.get("raw_source_id")
        if source_raw_id:
            return self.resolve_raw_source(source_raw_id)
        return None

    def s0_alignment_warnings(self, evidence_item: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        if not evidence_item.get("raw_source_id"):
            warnings.append("missing_raw_source_id")
        elif not self.resolve_raw_source(str(evidence_item["raw_source_id"])):
            warnings.append("raw_source_manifest_missing")
        if not evidence_item.get("adapter_run_id"):
            warnings.append("missing_adapter_run_id")
        if not evidence_item.get("modality"):
            warnings.append("missing_modality")
        if not evidence_item.get("locator"):
            warnings.append("missing_locator")
        if not evidence_item.get("source_specific_ref"):
            warnings.append("missing_source_specific_ref")
        if not evidence_item.get("display_ref"):
            warnings.append("missing_display_ref")
        return warnings

    def source_hashes(self) -> dict[str, str]:
        paths = {
            "evidence/source_manifest.jsonl": self.root / "evidence" / "source_manifest.jsonl",
            "evidence/evidence.jsonl": self.root / "evidence" / "evidence.jsonl",
            "evidence/build_manifest.json": self.root / "evidence" / "build_manifest.json",
            "memory/memory_units.jsonl": self.root / "memory" / "memory_units.jsonl",
            "memory/memory_build_manifest.json": self.root / "memory" / "memory_build_manifest.json",
            "memory/summaries.jsonl": self.root / "memory" / "summaries.jsonl",
        }
        fallback_paths = {
            "evidence/source_manifest.jsonl": self.source_manifest_path,
            "evidence/evidence.jsonl": self.evidence_path,
            "memory/memory_units.jsonl": self.memory_units_path,
            "memory/summaries.jsonl": self.summaries_path,
        }
        hashes: dict[str, str] = {}
        for name, path in paths.items():
            actual = path if path.exists() else fallback_paths.get(name)
            if actual and actual.exists():
                hashes[name] = file_hash(actual)
        return hashes
