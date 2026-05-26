"""S0B section policy runner for generic text sources.

This v0.1 runner is deliberately conservative:
- script-first classification only;
- no LLM calls;
- no S1 evidence generation;
- no deletion of raw spans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SUPPORTED_DUPLICATE_POLICIES = {"fail", "overwrite_generated"}
HIGH_RISK_TYPES = {
    "front_matter",
    "license",
    "copyright",
    "editorial_preface",
    "editorial_note",
    "footnote",
    "endnote",
    "appendix",
    "bibliography",
    "person_list",
    "metadata",
    "ocr_noise",
    "page_header_footer",
}
BASE_PLAINTEXT_PROFILE = "base_plaintext"
BOOK_PLAINTEXT_PROFILE = "book_plaintext"
DIARY_PLAINTEXT_PROFILE = "diary_plaintext"
DIALOGUE_PLAINTEXT_PROFILE = "dialogue_plaintext"
KNOWN_TEXT_STRUCTURE_PROFILES = {
    BASE_PLAINTEXT_PROFILE,
    BOOK_PLAINTEXT_PROFILE,
    DIARY_PLAINTEXT_PROFILE,
    DIALOGUE_PLAINTEXT_PROFILE,
}

SENTENCE_END_CHARS = {".", "!", "?", "。", "！", "？"}
SENTENCE_CLOSE_CHARS = {'"', "'", "”", "’", ")", "]", "}", "）", "】", "》"}
NON_TERMINAL_ABBREVIATIONS = {
    "adm",
    "capt",
    "cf",
    "col",
    "dr",
    "e.g",
    "etc",
    "fig",
    "gen",
    "hon",
    "i.e",
    "jr",
    "lt",
    "mr",
    "mrs",
    "ms",
    "mt",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "no",
    "p",
    "pp",
    "prof",
    "rev",
    "sr",
    "st",
    "vs",
}
EXPLICIT_FRONT_MATTER_MARKERS = [
    "front matter",
    "fixture front matter",
    "synthetic fixture",
    "workflow testing",
    "should not become ordinary evidence",
    "should not become a portrait unit",
    "not become ordinary evidence",
    "not a portrait unit",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def looks_like_external_uri(value: str) -> bool:
    lowered = value.lower()
    return "://" in lowered and not lowered.startswith("file://")


def resolve_path(project_root: Path, workspace: Path, source: dict[str, Any]) -> Path:
    raw = source.get("local_path") or source.get("original_uri_or_path")
    if not raw:
        raise ValueError(f"raw_source {source.get('raw_source_id')} has no local_path")
    if looks_like_external_uri(str(raw)):
        raise ValueError(f"S0B section policy v0.1 supports local paths only: {raw}")
    path = Path(raw)
    if path.is_absolute():
        return path
    for candidate in (project_root / path, workspace / path):
        if candidate.exists():
            return candidate
    return project_root / path


def infer_text_structure_profile(source: dict[str, Any], bundle: dict[str, Any]) -> str:
    metadata = source.get("source_specific_metadata") or {}
    explicit = (
        metadata.get("text_structure_profile")
        or metadata.get("section_policy_profile")
        or source.get("text_structure_profile")
        or source.get("section_policy_profile")
        or bundle.get("text_structure_profile")
        or bundle.get("section_policy_profile")
    )
    if explicit:
        profile = str(explicit)
        if profile not in KNOWN_TEXT_STRUCTURE_PROFILES:
            raise ValueError(f"Unsupported text_structure_profile: {profile}")
        return profile
    return BASE_PLAINTEXT_PROFILE


def split_text_segments(text: str, max_chars: int = 1200) -> list[tuple[int, int, str]]:
    """Split plain text into paragraph-like spans without semantic rewriting."""

    segments: list[tuple[int, int, str]] = []
    current_parts: list[str] = []
    current_start: int | None = None
    current_end = 0

    for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, flags=re.S):
        raw = match.group(0).strip()
        if not raw:
            continue
        compact = " ".join(raw.split())
        if not compact:
            continue
        if current_start is None:
            current_start = match.start()
        projected = ("\n\n".join([*current_parts, compact])).strip()
        if current_parts and len(projected) > max_chars:
            segments.append((current_start, current_end, "\n\n".join(current_parts).strip()))
            current_parts = [compact]
            current_start = match.start()
        else:
            current_parts.append(compact)
        current_end = match.end()

    if current_parts and current_start is not None:
        segments.append((current_start, current_end, "\n\n".join(current_parts).strip()))
    return segments


def trim_span(raw: str, start: int, end: int) -> tuple[int, int, str]:
    while start < end and raw[start].isspace():
        start += 1
    while end > start and raw[end - 1].isspace():
        end -= 1
    return start, end, raw[start:end]


def split_paragraph_ranges(raw: str, start: int, end: int) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    section_text = raw[start:end]
    for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", section_text, flags=re.S):
        p_start, p_end, text = trim_span(raw, start + match.start(), start + match.end())
        if text.strip():
            ranges.append((p_start, p_end, " ".join(text.split())))
    if not ranges:
        p_start, p_end, text = trim_span(raw, start, end)
        if text.strip():
            ranges.append((p_start, p_end, " ".join(text.split())))
    return ranges


def previous_ascii_token(text: str, dot_index: int) -> str:
    match = re.search(r"([A-Za-z](?:[A-Za-z]|\.)*)$", text[:dot_index])
    return match.group(1) if match else ""


def is_decimal_period(text: str, index: int) -> bool:
    return index > 0 and index + 1 < len(text) and text[index - 1].isdigit() and text[index + 1].isdigit()


def is_initial_or_acronym_period(text: str, index: int) -> bool:
    token = previous_ascii_token(text, index)
    if len(token) == 1 and token.isupper():
        return True
    return bool(re.fullmatch(r"(?:[A-Z]\.)+[A-Z]?", token))


def is_non_terminal_abbreviation_period(text: str, index: int) -> bool:
    token = previous_ascii_token(text, index).lower().rstrip(".")
    return token in NON_TERMINAL_ABBREVIATIONS


def is_date_prefix_period(text: str, index: int) -> bool:
    prefix = text[: index + 1]
    numeric_date = r"(?:^|\s)\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\.$"
    month_date = r"(?:^|\s)(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\.\s+\d{1,2},\s+\d{4}\.$"
    return bool(re.search(numeric_date, prefix, flags=re.I) or re.search(month_date, prefix, flags=re.I))


def is_sentence_boundary(text: str, index: int) -> bool:
    char = text[index]
    if char not in SENTENCE_END_CHARS:
        return False
    if char == ".":
        if is_date_prefix_period(text, index):
            return False
        if is_decimal_period(text, index):
            return False
        if is_initial_or_acronym_period(text, index):
            return False
        if is_non_terminal_abbreviation_period(text, index):
            return False
    return True


def collect_sentence_trailing_closers(text: str, index: int) -> int:
    end_index = index + 1
    while end_index < len(text) and text[end_index] in SENTENCE_CLOSE_CHARS:
        end_index += 1
    return end_index


def sentence_segmentation_warnings(sentence_text: str) -> list[str]:
    warnings: list[str] = []
    stripped = sentence_text.strip()
    if not stripped:
        return warnings
    if stripped[0] in {"]", ")", "}", "）", "】"}:
        warnings.append("starts_with_closing_bracket")
    bracket_pairs = [("[", "]"), ("(", ")"), ("{", "}"), ("（", "）"), ("【", "】")]
    for opener, closer in bracket_pairs:
        if stripped.count(opener) != stripped.count(closer):
            warnings.append("unbalanced_bracket_fragment")
            break
    if re.search(r"\b[A-Za-z]{1,4}\.$", stripped):
        token = previous_ascii_token(stripped, len(stripped) - 1).lower().rstrip(".")
        if token in NON_TERMINAL_ABBREVIATIONS:
            warnings.append("ends_with_non_terminal_abbreviation")
    return warnings


def split_sentence_ranges(raw: str, start: int, end: int) -> list[tuple[int, int, str]]:
    ranges: list[tuple[int, int, str]] = []
    paragraph_text = raw[start:end]
    sentence_start = 0
    index = 0
    while index < len(paragraph_text):
        if not is_sentence_boundary(paragraph_text, index):
            index += 1
            continue
        sentence_end = collect_sentence_trailing_closers(paragraph_text, index)
        next_index = sentence_end
        while next_index < len(paragraph_text) and paragraph_text[next_index].isspace():
            next_index += 1
        if next_index < len(paragraph_text) and paragraph_text[next_index] in SENTENCE_END_CHARS:
            index = next_index
            continue
        s_start, s_end, text = trim_span(raw, start + sentence_start, start + sentence_end)
        if text.strip():
            ranges.append((s_start, s_end, " ".join(text.split())))
        sentence_start = next_index
        index = next_index
    if sentence_start < len(paragraph_text):
        s_start, s_end, text = trim_span(raw, start + sentence_start, end)
        if text.strip():
            ranges.append((s_start, s_end, " ".join(text.split())))
    if not ranges:
        s_start, s_end, text = trim_span(raw, start, end)
        if text.strip():
            ranges.append((s_start, s_end, " ".join(text.split())))
    return ranges


def build_text_units_for_section(
    section_row: dict[str, Any],
    raw_text: str,
    raw_path: Path,
) -> list[dict[str, Any]]:
    """Build recoverable paragraph/sentence text units without semantic interpretation."""

    units: list[dict[str, Any]] = []
    raw_span_id = str(section_row["raw_span_id"])
    section_start = int(section_row["char_start"])
    section_end = int(section_row["char_end"])
    paragraphs = split_paragraph_ranges(raw_text, section_start, section_end)

    for paragraph_index, (p_start, p_end, p_text) in enumerate(paragraphs, 1):
        paragraph_id = f"P{paragraph_index:04d}"
        paragraph_unit_id = f"{raw_span_id}:{paragraph_id}"
        sentences = split_sentence_ranges(raw_text, p_start, p_end)
        sentence_ids = [f"{paragraph_unit_id}.S{sentence_index:04d}" for sentence_index in range(1, len(sentences) + 1)]
        paragraph_unit = {
            "schema_version": "s0b.text_unit.v0.1",
            "text_unit_id": paragraph_unit_id,
            "raw_span_id": raw_span_id,
            "raw_source_id": section_row.get("raw_source_id"),
            "workspace_id": section_row.get("workspace_id"),
            "bundle_id": section_row.get("bundle_id"),
            "unit_type": "paragraph",
            "paragraph_id": paragraph_id,
            "sentence_id": None,
            "text": p_text,
            "char_start": p_start,
            "char_end": p_end,
            "raw_text_hash": sha256_text(raw_text[p_start:p_end]),
            "normalized_text_hash": sha256_text(" ".join(p_text.split()).lower()),
            "section_type": section_row.get("final_section_type") or section_row.get("section_type"),
            "text_structure_profile": section_row.get("text_structure_profile"),
            "section_policy_profile": section_row.get("section_policy_profile"),
            "perspective": section_row.get("perspective"),
            "s1_storage_policy": section_row.get("s1_storage_policy"),
            "retrieval_policy": section_row.get("retrieval_policy"),
            "s2_policy": section_row.get("s2_policy"),
            "segmentation_method": "script_rule",
            "segmentation_confidence": "medium" if len(sentences) > 1 else "high",
            "previous_text_unit_id": None,
            "next_text_unit_id": None,
            "parent_text_unit_id": None,
            "child_text_unit_ids": sentence_ids,
            "raw_backpointer": {
                "source_file": str(raw_path),
                "locator": {"kind": "text_span", "char_start": p_start, "char_end": p_end},
            },
            "warnings": [],
        }
        units.append(paragraph_unit)

        for sentence_index, (s_start, s_end, s_text) in enumerate(sentences, 1):
            sentence_unit_id = sentence_ids[sentence_index - 1]
            warnings = sentence_segmentation_warnings(s_text)
            units.append(
                {
                    "schema_version": "s0b.text_unit.v0.1",
                    "text_unit_id": sentence_unit_id,
                    "raw_span_id": raw_span_id,
                    "raw_source_id": section_row.get("raw_source_id"),
                    "workspace_id": section_row.get("workspace_id"),
                    "bundle_id": section_row.get("bundle_id"),
                    "unit_type": "sentence",
                    "paragraph_id": paragraph_id,
                    "sentence_id": f"S{sentence_index:04d}",
                    "text": s_text,
                    "char_start": s_start,
                    "char_end": s_end,
                    "raw_text_hash": sha256_text(raw_text[s_start:s_end]),
                    "normalized_text_hash": sha256_text(" ".join(s_text.split()).lower()),
                    "section_type": section_row.get("final_section_type") or section_row.get("section_type"),
                    "text_structure_profile": section_row.get("text_structure_profile"),
                    "section_policy_profile": section_row.get("section_policy_profile"),
                    "perspective": section_row.get("perspective"),
                    "s1_storage_policy": section_row.get("s1_storage_policy"),
                    "retrieval_policy": section_row.get("retrieval_policy"),
                    "s2_policy": section_row.get("s2_policy"),
                    "segmentation_method": "script_rule",
                    "segmentation_confidence": "low" if warnings else "medium",
                    "previous_text_unit_id": sentence_ids[sentence_index - 2] if sentence_index > 1 else None,
                    "next_text_unit_id": sentence_ids[sentence_index] if sentence_index < len(sentence_ids) else None,
                    "parent_text_unit_id": paragraph_unit_id,
                    "child_text_unit_ids": [],
                    "raw_backpointer": {
                        "source_file": str(raw_path),
                        "locator": {"kind": "text_span", "char_start": s_start, "char_end": s_end},
                    },
                    "warnings": warnings,
                }
            )
    return units


def classify_section(text: str, index: int, text_structure_profile: str = BASE_PLAINTEXT_PROFILE) -> tuple[str, str, str, str, str]:
    """Return section_type, confidence, perspective, attribution_status, quote_boundary_status."""

    compact = " ".join(text.split())
    lowered = compact.lower()
    date_like_body = (
        re.match(r"^\s*\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b", compact)
        or re.match(r"^\s*[A-Z][a-z]{2,8}\.\s+\d{1,2},", compact)
        or re.match(r"^\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.\s+\d", lowered)
    )
    if date_like_body:
        return "body", "high", "author", "unknown", "unknown"
    if any(marker in lowered for marker in ["project gutenberg", "gutenberg license", "end of the project gutenberg"]):
        return "license", "high", "editor", "unknown", "unknown"
    if any(marker in lowered for marker in ["copyright", "all rights reserved", "license"]):
        return "copyright", "medium", "editor", "unknown", "unknown"
    if any(marker in lowered for marker in ["transcriber's note", "editor's note", "editorial note"]):
        return "editorial_note", "high", "editor", "unknown", "unknown"
    if re.match(r"^\s*(footnote|endnote)\b", lowered) or re.match(r"^\s*\[[0-9ivxlcdm]+\]", lowered):
        return "footnote", "medium", "editor", "unknown", "unknown"
    if re.match(r"^\s*(appendix|bibliography|references)\b", lowered):
        return "appendix" if lowered.startswith("appendix") else "bibliography", "high", "editor", "unknown", "unknown"
    if re.match(r"^\s*(persons?|dramatis personae|list of)\b", lowered):
        return "person_list", "medium", "editor", "unknown", "unknown"
    if any(marker in lowered for marker in EXPLICIT_FRONT_MATTER_MARKERS):
        return "front_matter", "high", "editor", "unknown", "unknown"
    book_like_profile = text_structure_profile == BOOK_PLAINTEXT_PROFILE
    if book_like_profile and len(compact) < 80 and compact.isupper():
        return "front_matter" if index <= 3 else "metadata", "medium", "editor", "unknown", "unknown"
    if compact.startswith(("\"", "'", "\u201c")) or (len(compact) <= 800 and re.search(r"\bsaid\b.*[\"'\u201c]", lowered)):
        return "quote", "medium", "quoted_speaker", "ambiguous", "ambiguous"
    if book_like_profile and index <= 2 and len(compact) < 200:
        return "front_matter", "low", "unknown", "unknown", "unknown"
    return "body", "medium", "author", "unknown", "unknown"


def default_policies(section_type: str, confidence: str) -> tuple[str, str, str]:
    if section_type == "body":
        return "ordinary_evidence", "default_retrieval", "candidate_allowed"
    if section_type in {"front_matter", "license", "copyright", "metadata", "ocr_noise", "page_header_footer"}:
        return "recoverable_raw_only", "excluded_from_default_retrieval", "blocked_from_portrait"
    if section_type in {"footnote", "endnote", "editorial_note", "editorial_preface", "appendix", "bibliography", "person_list", "list"}:
        return "background_evidence", "explicit_query_only", "background_only"
    if section_type == "quote":
        return "background_evidence", "explicit_query_only", "background_only"
    return "needs_review", "needs_review", "needs_review"


def assist_triggers(
    section_type: str,
    confidence: str,
    perspective: str,
    s2_policy: str,
    source_priority: str,
) -> list[str]:
    triggers: list[str] = []
    if confidence == "low":
        triggers.append("low_script_confidence")
    if section_type in HIGH_RISK_TYPES and confidence != "high":
        triggers.append("high_risk_section")
    if perspective in {"mixed", "unknown"} and s2_policy == "candidate_allowed":
        triggers.append("perspective_ambiguity")
    if s2_policy in {"candidate_allowed", "needs_review"} and confidence != "high":
        triggers.append("high_downstream_impact")
    if source_priority == "high" and confidence != "high":
        triggers.append("high_value_source")
    return triggers


@dataclass
class SectionPolicyInputs:
    project_root: Path
    workspace: Path
    duplicate_policy: str
    max_segment_chars: int
    run_id: str
    text_structure_profile: str | None = None


def load_inputs(args: argparse.Namespace) -> SectionPolicyInputs:
    project_root = Path(args.project_root).resolve()
    workspace = (project_root / args.workspace).resolve() if not Path(args.workspace).is_absolute() else Path(args.workspace).resolve()
    if args.duplicate_policy not in SUPPORTED_DUPLICATE_POLICIES:
        raise ValueError(f"Unsupported duplicate_policy: {args.duplicate_policy}")
    text_structure_profile = getattr(args, "text_structure_profile", None)
    if text_structure_profile and text_structure_profile not in KNOWN_TEXT_STRUCTURE_PROFILES:
        raise ValueError(f"Unsupported text_structure_profile: {text_structure_profile}")
    return SectionPolicyInputs(
        project_root=project_root,
        workspace=workspace,
        duplicate_policy=args.duplicate_policy,
        max_segment_chars=args.max_segment_chars,
        run_id=args.run_id or f"s0b-section-policy:{workspace.name}:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        text_structure_profile=text_structure_profile,
    )


def prepare_outputs(workspace: Path, duplicate_policy: str) -> None:
    org = workspace / "raw" / "organization"
    outputs = [
        org / "section_map.jsonl",
        org / "text_units.jsonl",
        org / "source_range_map.jsonl",
        org / "llm_assist_queue.jsonl",
        org / "section_policy_report.json",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and duplicate_policy == "fail":
        raise FileExistsError("Existing S0B section policy outputs found: " + ", ".join(str(path) for path in existing))
    if duplicate_policy == "overwrite_generated":
        for path in existing:
            path.unlink()


def build_section_policy(inputs: SectionPolicyInputs) -> dict[str, Any]:
    org = inputs.workspace / "raw" / "organization"
    raw_sources_path = org / "raw_sources.jsonl"
    if not raw_sources_path.exists():
        raise FileNotFoundError(f"Missing S0B raw_sources.jsonl: {raw_sources_path}")
    raw_sources = read_jsonl(raw_sources_path)
    bundle = read_json(org / "bundle.json") if (org / "bundle.json").exists() else {}
    workspace_id = bundle.get("workspace_id") or inputs.workspace.name

    section_rows: list[dict[str, Any]] = []
    text_unit_rows: list[dict[str, Any]] = []
    range_rows: list[dict[str, Any]] = []
    queue_rows: list[dict[str, Any]] = []

    for source in raw_sources:
        if source.get("adapter_recommendation") != "generic_text_adapter":
            continue
        if source.get("inclusion_decision") not in {None, "include"}:
            continue
        raw_path = resolve_path(inputs.project_root, inputs.workspace, source)
        text = raw_path.read_text(encoding="utf-8-sig")
        metadata = source.get("source_specific_metadata", {})
        max_chars = int(metadata.get("max_segment_chars") or inputs.max_segment_chars)
        segments = split_text_segments(text, max_chars=max_chars)
        source_priority = str(source.get("source_priority") or metadata.get("source_priority") or "normal")
        text_structure_profile = inputs.text_structure_profile or infer_text_structure_profile(source, bundle)
        section_policy_profile = text_structure_profile
        included_ranges: list[dict[str, Any]] = []
        excluded_ranges: list[dict[str, Any]] = []

        for index, (start, end, span_text) in enumerate(segments, 1):
            source_specific_ref = f"text_span:{index:04d}"
            raw_span_id = f"{source.get('raw_source_id')}:{source_specific_ref}"
            section_type, confidence, perspective, attribution_status, quote_boundary_status = classify_section(
                span_text,
                index,
                text_structure_profile,
            )
            s1_policy, retrieval_policy, s2_policy = default_policies(section_type, confidence)
            triggers = assist_triggers(section_type, confidence, perspective, s2_policy, source_priority)
            llm_status = "queued" if triggers else "not_needed"
            review_status = "needs_llm_assist" if triggers else "auto"
            row = {
                "schema_version": "s0b.section_map.v0.1",
                "raw_span_id": raw_span_id,
                "raw_source_id": source.get("raw_source_id"),
                "workspace_id": workspace_id,
                "bundle_id": source.get("bundle_id") or bundle.get("bundle_id"),
                "source_specific_ref": source_specific_ref,
                "text_structure_profile": text_structure_profile,
                "section_policy_profile": section_policy_profile,
                "char_start": start,
                "char_end": end,
                "raw_text_hash": sha256_text(span_text),
                "normalized_text_hash": sha256_text(" ".join(span_text.split()).lower()),
                "ingested_at": source.get("ingested_at") or now_iso(),
                "time_source": source.get("time_source") or "unknown",
                "time_confidence": source.get("time_confidence") or "unknown",
                "source_created_at": source.get("source_created_at"),
                "source_modified_at": source.get("source_modified_at"),
                "source_published_at": source.get("source_published_at"),
                "observed_at": source.get("observed_at"),
                "temporal_coverage": source.get("temporal_coverage"),
                "script_section_guess": section_type,
                "script_confidence": confidence,
                "final_section_type": section_type,
                "perspective": perspective,
                "attribution_status": attribution_status,
                "quoted_speaker_id": None,
                "quoted_speaker_label": None,
                "quote_boundary_status": quote_boundary_status,
                "s1_storage_policy": s1_policy,
                "retrieval_policy": retrieval_policy,
                "s2_policy": s2_policy,
                "classification_method": "script_rule",
                "confidence": confidence,
                "review_status": review_status,
                "llm_assist_status": llm_status,
                "llm_assist_triggers": triggers,
                "raw_backpointer": {
                    "source_file": str(raw_path),
                    "locator": {"kind": "text_span", "char_start": start, "char_end": end},
                },
            }
            section_rows.append(row)
            text_unit_rows.extend(build_text_units_for_section(row, text, raw_path))
            range_item = {
                "range_id": source_specific_ref,
                "char_start": start,
                "char_end": end,
                "range_type": section_type,
            }
            if s1_policy == "ordinary_evidence":
                included_ranges.append(range_item)
            else:
                excluded_ranges.append(range_item)
            if triggers:
                queue_rows.append(
                    {
                        "schema_version": "s0b.llm_assist_queue.v0.1",
                        "queue_item_id": f"llm-assist:{sha256_text(raw_span_id)[:16]}",
                        "raw_span_id": raw_span_id,
                        "raw_source_id": source.get("raw_source_id"),
                        "workspace_id": workspace_id,
                        "trigger_reasons": triggers,
                        "current_section_guess": section_type,
                        "current_perspective_guess": perspective,
                        "current_s1_storage_policy": s1_policy,
                        "current_retrieval_policy": retrieval_policy,
                        "current_s2_policy": s2_policy,
                        "source_priority": source_priority,
                        "required_decision": ["final_section_type", "perspective", "s1_storage_policy", "retrieval_policy", "s2_policy"],
                        "raw_backpointer": row["raw_backpointer"],
                        "status": "queued",
                    }
                )

        range_rows.append(
            {
                "schema_version": "s0b.source_range_policy.v0.1",
                "raw_source_id": source.get("raw_source_id"),
                "workspace_id": workspace_id,
                "text_structure_profile": text_structure_profile,
                "section_policy_profile": section_policy_profile,
                "selected_ranges": included_ranges,
                "excluded_ranges": excluded_ranges,
                "range_selection_reason": "script-first section policy v0.1; ordinary body spans selected for default S1 intake",
                "range_selection_method": "script_rule",
                "raw_backpointer": {"source_file": str(raw_path)},
            }
        )

    return {
        "section_rows": section_rows,
        "text_unit_rows": text_unit_rows,
        "range_rows": range_rows,
        "queue_rows": queue_rows,
        "report": {
            "schema_version": "s0b.section_policy_report.v0.1",
            "run_id": inputs.run_id,
            "workspace": str(inputs.workspace),
            "status": "completed",
            "generated_at": now_iso(),
            "section_count": len(section_rows),
            "text_unit_count": len(text_unit_rows),
            "source_range_count": len(range_rows),
            "llm_assist_queue_count": len(queue_rows),
            "llm_assist_executed": False,
            "boundary": "section labels are routing metadata, not evidence truth",
        },
    }


def run_section_policy(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_inputs(args)
    prepare_outputs(inputs.workspace, inputs.duplicate_policy)
    result = build_section_policy(inputs)
    org = inputs.workspace / "raw" / "organization"
    write_jsonl(org / "section_map.jsonl", result["section_rows"])
    write_jsonl(org / "text_units.jsonl", result["text_unit_rows"])
    write_jsonl(org / "source_range_map.jsonl", result["range_rows"])
    write_jsonl(org / "llm_assist_queue.jsonl", result["queue_rows"])
    write_json(org / "section_policy_report.json", result["report"])
    return {
        "workspace": str(inputs.workspace),
        "section_map": str(org / "section_map.jsonl"),
        "text_units": str(org / "text_units.jsonl"),
        "source_range_map": str(org / "source_range_map.jsonl"),
        "llm_assist_queue": str(org / "llm_assist_queue.jsonl"),
        "counts": {
            "sections": len(result["section_rows"]),
            "text_units": len(result["text_unit_rows"]),
            "source_ranges": len(result["range_rows"]),
            "llm_assist_queue": len(result["queue_rows"]),
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run S0B section policy over organized generic text raw sources.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-segment-chars", type=int, default=1200)
    parser.add_argument("--text-structure-profile", default=None, choices=sorted(KNOWN_TEXT_STRUCTURE_PROFILES))
    parser.add_argument("--duplicate-policy", default="fail", choices=sorted(SUPPORTED_DUPLICATE_POLICIES))
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    print(json.dumps(run_section_policy(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
