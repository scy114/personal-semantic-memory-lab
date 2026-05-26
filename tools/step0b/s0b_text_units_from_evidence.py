"""Build S0B text units from text-like S1 evidence rows.

This is an organization-only adapter for evidence rows that already preserve
source structure, such as LoCoMo conversation turns. It emits recoverable text
units and does not create evidence truth, memory candidates, graph truth, or
support status.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.step0b.s0b_section_policy_runner import (
    sentence_segmentation_warnings,
    sha256_text,
    split_sentence_ranges,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def source_id(value: str) -> str:
    return value.replace("\\", "/").replace(" ", "_")


def evidence_backpointer(row: dict[str, Any], *, unit_kind: str, char_start: int | None = None, char_end: int | None = None) -> dict[str, Any]:
    locator = dict(row.get("locator") or {})
    locator["kind"] = unit_kind
    locator["evidence_ref"] = row.get("evidence_ref") or row.get("canonical_evidence_ref")
    if char_start is not None:
        locator["char_start"] = char_start
    if char_end is not None:
        locator["char_end"] = char_end
    return {"source_file": locator.get("source_file"), "locator": locator}


def build_units_for_evidence(row: dict[str, Any], workspace_id: str) -> list[dict[str, Any]]:
    text = str(row.get("text") or "").strip()
    evidence_ref = str(row.get("evidence_ref") or row.get("canonical_evidence_ref") or "")
    if not text or not evidence_ref:
        return []

    unit_base = source_id(evidence_ref)
    parent_id = f"{unit_base}:TURN"
    sentences = split_sentence_ranges(text, 0, len(text))
    sentence_ids = [f"{parent_id}.S{index:04d}" for index in range(1, len(sentences) + 1)]
    metadata = row.get("metadata") or {}
    locator = row.get("locator") or {}
    subject_role = row.get("subject_role") or metadata.get("subject_role") or "unknown"
    perspective = "target" if subject_role == "target" else subject_role or "unknown"
    raw_source_id = row.get("raw_source_id") or row.get("source_id")
    raw_span_id = row.get("raw_span_id") or evidence_ref
    common = {
        "schema_version": "s0b.text_unit.v0.1",
        "raw_span_id": raw_span_id,
        "raw_source_id": raw_source_id,
        "workspace_id": workspace_id,
        "bundle_id": row.get("bundle_id"),
        "evidence_ref": evidence_ref,
        "canonical_evidence_ref": row.get("canonical_evidence_ref") or evidence_ref,
        "source_type": row.get("source_type"),
        "section_type": row.get("section_type") or row.get("source_type") or "conversation",
        "perspective": perspective,
        "speaker": row.get("speaker") or locator.get("speaker"),
        "participant": row.get("participant"),
        "participant_ids": row.get("participant_ids") or [],
        "target_participant": row.get("target_participant") or metadata.get("modeled_subject_id"),
        "target_subject_ids": row.get("target_subject_ids") or ([metadata.get("modeled_subject_id")] if metadata.get("modeled_subject_id") else []),
        "subject_ids": row.get("subject_ids") or [],
        "subject_role": subject_role,
        "subject_contamination_risk": row.get("subject_contamination_risk"),
        "timestamp": row.get("timestamp"),
        "s1_storage_policy": "ordinary_evidence",
        "retrieval_policy": row.get("retrieval_policy") or "default_retrieval",
        "s2_policy": row.get("s2_policy") or "candidate_allowed",
        "segmentation_method": "script_rule",
    }
    units: list[dict[str, Any]] = [
        {
            **common,
            "text_unit_id": parent_id,
            "unit_type": "turn" if row.get("source_type") == "conversation" else "evidence_span",
            "paragraph_id": None,
            "sentence_id": None,
            "text": text,
            "char_start": 0,
            "char_end": len(text),
            "raw_text_hash": sha256_text(text),
            "normalized_text_hash": sha256_text(" ".join(text.split()).lower()),
            "segmentation_confidence": "medium" if len(sentences) > 1 else "high",
            "previous_text_unit_id": None,
            "next_text_unit_id": None,
            "parent_text_unit_id": None,
            "child_text_unit_ids": sentence_ids,
            "raw_backpointer": evidence_backpointer(row, unit_kind="conversation_turn" if row.get("source_type") == "conversation" else "evidence_span"),
            "warnings": [],
        }
    ]
    for index, (start, end, sentence_text) in enumerate(sentences, 1):
        warnings = sentence_segmentation_warnings(sentence_text)
        units.append(
            {
                **common,
                "text_unit_id": sentence_ids[index - 1],
                "unit_type": "sentence",
                "paragraph_id": None,
                "sentence_id": f"S{index:04d}",
                "text": sentence_text,
                "char_start": start,
                "char_end": end,
                "raw_text_hash": sha256_text(text[start:end]),
                "normalized_text_hash": sha256_text(" ".join(sentence_text.split()).lower()),
                "segmentation_confidence": "low" if warnings else "medium",
                "previous_text_unit_id": sentence_ids[index - 2] if index > 1 else None,
                "next_text_unit_id": sentence_ids[index] if index < len(sentence_ids) else None,
                "parent_text_unit_id": parent_id,
                "child_text_unit_ids": [],
                "raw_backpointer": evidence_backpointer(row, unit_kind="conversation_turn_sentence" if row.get("source_type") == "conversation" else "evidence_sentence", char_start=start, char_end=end),
                "warnings": warnings,
            }
        )
    return units


def link_sentence_neighbors(units: list[dict[str, Any]]) -> None:
    sentence_indexes = [index for index, row in enumerate(units) if row.get("unit_type") == "sentence"]
    for pos, unit_index in enumerate(sentence_indexes):
        row = units[unit_index]
        if row.get("previous_text_unit_id") is None and pos > 0:
            row["previous_text_unit_id"] = units[sentence_indexes[pos - 1]]["text_unit_id"]
        if row.get("next_text_unit_id") is None and pos + 1 < len(sentence_indexes):
            row["next_text_unit_id"] = units[sentence_indexes[pos + 1]]["text_unit_id"]


def build_text_units(evidence_rows: list[dict[str, Any]], workspace_id: str) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for row in evidence_rows:
        if row.get("modality") not in {None, "text"}:
            continue
        units.extend(build_units_for_evidence(row, workspace_id))
    link_sentence_neighbors(units)
    return units


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build S0B text_units.jsonl from text-like evidence rows.")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--evidence", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    workspace = Path(args.workspace).resolve()
    evidence_path = Path(args.evidence).resolve() if args.evidence else workspace / "evidence" / "evidence.jsonl"
    output_path = Path(args.output).resolve() if args.output else workspace / "raw" / "organization" / "text_units.jsonl"
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists; pass --overwrite to replace: {output_path}")
    rows = read_jsonl(evidence_path)
    units = build_text_units(rows, workspace.name)
    write_jsonl(output_path, units)
    print(json.dumps({"workspace": str(workspace), "evidence_rows": len(rows), "text_units": len(units), "output": str(output_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
