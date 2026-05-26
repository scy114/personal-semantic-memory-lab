"""Build a read-only v0.21 resource-backed routing feature matrix.

This tool does not make route decisions. It turns existing S0B/S1/S2
artifacts plus downloaded external NLP resources into inspectable feature
rows for later scoring-policy comparison.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "routing.v021_feature_matrix.v0.1"
DEFAULT_RESOURCE_INVENTORY = "external_references/routing_vocab/routing_resource_inventory.csv"
DEFAULT_OUTPUT_NAME = "v021_feature_matrix"
DEFAULT_INPUTS = (
    ("s0b_text_unit", "raw/organization/text_units.jsonl"),
    ("s1_memory_candidate", "memory/memory_candidates.jsonl"),
    ("s1_memory_unit", "memory/memory_units.jsonl"),
    ("s2_proposal_backed_unit", "portrait/reviewed_units.jsonl"),
)


def clamp(value: float, lower: float = 0.0, upper: float = 10.0) -> float:
    return max(lower, min(upper, value))


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return round(float(value), digits)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


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
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def module_status(name: str) -> str:
    return "available" if importlib.util.find_spec(name) is not None else "package_missing"


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def normalize_token(token: str) -> str:
    return token.strip("'_-").lower()


def is_chinese_token(token: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", token))


@dataclass
class ExternalEnumerableResources:
    english_stopwords: set[str] = field(default_factory=set)
    chinese_stopwords: set[str] = field(default_factory=set)
    english_discourse_connectives: set[str] = field(default_factory=set)
    chinese_discourse_connectives: set[str] = field(default_factory=set)
    statuses: dict[str, dict[str, Any]] = field(default_factory=dict)

    def stopword_source_status(self) -> dict[str, Any]:
        return {
            "stopwords_iso": self.statuses.get("stopwords_iso", {}),
            "chinese_stopwords_goto456": self.statuses.get("chinese_stopwords_goto456", {}),
        }

    def connective_source_status(self) -> dict[str, Any]:
        return {
            "en_dimlex": self.statuses.get("en_dimlex", {}),
            "chinese_dimlex": self.statuses.get("chinese_dimlex", {}),
        }


def token_is_contentful(token: str, resources: ExternalEnumerableResources) -> bool:
    normalized = normalize_token(token)
    if not normalized:
        return False
    if is_chinese_token(normalized):
        return len(normalized) >= 2 and normalized not in resources.chinese_stopwords
    return len(normalized) > 2 and normalized not in resources.english_stopwords


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    if chunks and module_status("jieba") == "available":
        try:
            import jieba  # type: ignore

            for chunk in chunks:
                tokens.extend(token for token in jieba.lcut(chunk) if token.strip())
        except Exception:
            tokens.extend(chunks)
    else:
        tokens.extend(chunks)
    return tokens


def sentence_split(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[.!?\u3002\uff01\uff1f]+", text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def chinese_connective_hit_count(text: str, marker: str) -> int:
    marker = marker.strip()
    if not marker:
        return 0
    if "..." not in marker and "\u2026" not in marker:
        return text.count(marker)
    parts = [part for part in re.split(r"(?:\.{3,}|\u2026+)", marker) if part]
    if not parts:
        return 0
    pattern = ".*?".join(re.escape(part) for part in parts)
    return 1 if re.search(pattern, text) else 0


def language_scores(text: str, tokens: list[str], resources: ExternalEnumerableResources) -> dict[str, Any]:
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    total = zh_chars + latin_chars
    zh_score = (zh_chars / total) * 10 if total else 0.0
    en_score = (latin_chars / total) * 10 if total else 0.0
    if zh_score >= 6 and en_score >= 2:
        primary = "mixed"
    elif zh_score >= en_score and zh_score > 0:
        primary = "zh"
    elif en_score > 0:
        primary = "en"
    else:
        primary = "unknown"
    return {
        "primary": primary,
        "zh_score": round(zh_score, 3),
        "en_score": round(en_score, 3),
        "tokenizer": "jieba" if has_chinese(text) and module_status("jieba") == "available" else "regex",
        "token_count": len(tokens),
        "content_token_count": len([token for token in tokens if token_is_contentful(token, resources)]),
    }


def local_sentence_complexity(text: str, tokens: list[str], resources: ExternalEnumerableResources) -> dict[str, Any]:
    words = [normalize_token(token) for token in tokens if normalize_token(token)]
    sentences = sentence_split(text)
    lower = f" {text.lower()} "
    discourse_hits = []
    for marker in resources.english_discourse_connectives:
        hit_count = len(re.findall(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", lower))
        if hit_count:
            discourse_hits.append({"marker": marker, "language": "en", "count": hit_count})
    for marker in resources.chinese_discourse_connectives:
        hit_count = chinese_connective_hit_count(text, marker)
        if hit_count:
            discourse_hits.append({"marker": marker, "language": "zh", "count": hit_count})
    discourse_marker_count = sum(hit["count"] for hit in discourse_hits)
    punctuation_count = sum(text.count(ch) for ch in [",", ";", ":", "\uff0c", "\uff1b", "\uff1a"])
    parenthetical_count = sum(text.count(ch) for ch in ["(", ")", "[", "]", "\uff08", "\uff09"])
    quote_count = sum(text.count(ch) for ch in ['"', "'", "\u201c", "\u201d", "\u2018", "\u2019"])
    avg_sentence_length = len(words) / max(1, len(sentences))
    score = discourse_marker_count * 2 + min(3, punctuation_count)
    if avg_sentence_length >= 18:
        score += 2
    if avg_sentence_length >= 30:
        score += 2
    if parenthetical_count:
        score += 1
    reliability = "empty" if not words else ("low_short_text" if len(words) < 10 else "standard")
    return {
        "discourse_marker_count": discourse_marker_count,
        "top_discourse_markers": sorted(discourse_hits, key=lambda item: (-item["count"], item["marker"]))[:12],
        "clause_punctuation_count": punctuation_count,
        "parenthetical_marker_count": parenthetical_count,
        "quote_marker_count": quote_count,
        "avg_sentence_length": round(avg_sentence_length, 3) if words else 0.0,
        "score": round(clamp(float(score if reliability == "standard" else 0)), 3),
        "reliability": reliability,
    }


def capitalized_entity_candidates(text: str, resources: ExternalEnumerableResources) -> list[str]:
    candidates = []
    for match in re.finditer(r"\b(?:[A-Z][a-z]+(?:\s+|$)){1,4}", text):
        value = match.group(0).strip()
        if value and value.lower() not in resources.english_stopwords:
            candidates.append(value)
    return sorted(set(candidates))[:20]


def safe_text(row: dict[str, Any]) -> str:
    for key in (
        "original_text",
        "processed_text",
        "memory_candidate_text",
        "candidate_text",
        "claim_text",
        "fact_candidate_text",
        "hypothesis_text",
        "content",
        "text",
        "source_text",
        "evidence_quote",
        "summary_text",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    nested = row.get("llm_assisted_s1_candidate")
    if isinstance(nested, dict):
        value = nested.get("memory_candidate_text")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def unit_id_for_row(row: dict[str, Any], fallback: str) -> str:
    for key in (
        "text_unit_id",
        "memory_id",
        "memory_candidate_id",
        "proposal_id",
        "reviewed_unit_id",
        "unit_id",
        "claim_id",
        "evidence_ref",
        "raw_span_id",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback


@dataclass
class InputUnit:
    source_layer: str
    source_path: Path
    source_row_index: int
    unit_id: str
    workspace_id: str
    text: str
    row: dict[str, Any]


@dataclass
class ResourceInventory:
    project_root: Path
    path: Path
    rows: list[dict[str, str]]
    by_id: dict[str, dict[str, str]]

    def resource_path(self, resource_id: str) -> Path | None:
        row = self.by_id.get(resource_id)
        if not row:
            return None
        local_path = row.get("local_path") or ""
        if not local_path:
            return None
        return resolve_project_path(self.project_root, local_path)

    def resource_note(self, resource_id: str) -> dict[str, str]:
        row = self.by_id.get(resource_id) or {}
        return {
            "license_status": row.get("license_status", "unknown"),
            "feature_family": row.get("feature_family", ""),
            "use_note": row.get("v021_use_note", ""),
        }


def load_resource_inventory(project_root: Path, path: Path) -> ResourceInventory:
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle)]
    by_id = {str(row.get("resource_id", "")).strip(): row for row in rows if row.get("resource_id")}
    return ResourceInventory(project_root=project_root, path=path, rows=rows, by_id=by_id)


def fallback_resource_path(inventory: ResourceInventory, resource_id: str, relative: str) -> Path:
    configured = inventory.resource_path(resource_id)
    if configured is not None:
        return configured
    return inventory.project_root / relative


def load_stopwords_iso(path: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    json_path = path / "stopwords-iso.json"
    if not json_path.exists():
        return {}, {"status": "missing_resource", "path": str(path)}
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, {"status": "error", "path": str(json_path), "error": type(exc).__name__}
    result: dict[str, set[str]] = {}
    for language in ("en", "zh"):
        values = data.get(language)
        if isinstance(values, list):
            result[language] = {str(value).strip().lower() for value in values if str(value).strip()}
    return result, {
        "status": "available" if result else "empty",
        "path": str(json_path),
        "languages": sorted(result.keys()),
        "entry_count": sum(len(values) for values in result.values()),
        "license_status": "MIT",
    }


def load_chinese_stopwords_goto456(path: Path) -> tuple[set[str], dict[str, Any]]:
    if not path.exists():
        return set(), {"status": "missing_resource", "path": str(path)}
    words: set[str] = set()
    files = sorted(path.glob("*stopwords.txt"))
    for file_path in files:
        try:
            for line in file_path.read_text(encoding="utf-8-sig", errors="ignore").splitlines():
                value = line.strip()
                if value:
                    words.add(value)
        except Exception:
            continue
    return words, {
        "status": "available" if words else "empty",
        "path": str(path),
        "file_count": len(files),
        "entry_count": len(words),
        "license_status": "unverified",
    }


def load_en_dimlex_connectives(path: Path) -> tuple[set[str], dict[str, Any]]:
    xml_path = path / "en_dimlex.xml" if path.is_dir() else path
    if not xml_path.exists():
        return set(), {"status": "missing_resource", "path": str(xml_path)}
    connectives: set[str] = set()
    try:
        root = ET.parse(xml_path).getroot()
        for entry in root.findall(".//entry"):
            word = entry.attrib.get("word")
            if word:
                connectives.add(word.strip().lower())
            for part in entry.findall(".//part"):
                text = (part.text or "").strip()
                if text:
                    connectives.add(text.lower())
    except Exception as exc:
        return set(), {"status": "error", "path": str(xml_path), "error": type(exc).__name__}
    return connectives, {
        "status": "available" if connectives else "empty",
        "path": str(xml_path),
        "entry_count": len(connectives),
        "license_status": "CC-BY-NC-SA-4.0",
    }


def load_chinese_dimlex_connectives(path: Path) -> tuple[set[str], dict[str, Any]]:
    xml_path = path / "chinese_dimlex.xml" if path.is_dir() else path
    if not xml_path.exists():
        return set(), {"status": "missing_resource", "path": str(xml_path)}
    connectives: set[str] = set()
    try:
        root = ET.parse(xml_path).getroot()
        for entry in root.findall(".//entry"):
            word = entry.attrib.get("word")
            if word:
                connectives.add(word.strip())
            for part in entry.findall(".//part"):
                text = (part.text or "").strip()
                if text:
                    connectives.add(text)
    except Exception as exc:
        return set(), {"status": "error", "path": str(xml_path), "error": type(exc).__name__}
    return connectives, {
        "status": "available" if connectives else "empty",
        "path": str(xml_path),
        "entry_count": len(connectives),
        "license_status": "CC-BY-NC-SA-4.0",
    }


def load_external_enumerable_resources(inventory: ResourceInventory) -> ExternalEnumerableResources:
    resources = ExternalEnumerableResources()
    stopwords_iso_path = fallback_resource_path(inventory, "stopwords_iso", "external_references/routing_vocab/stopwords-iso")
    stopwords_by_lang, status = load_stopwords_iso(stopwords_iso_path)
    resources.statuses["stopwords_iso"] = status
    resources.english_stopwords.update(stopwords_by_lang.get("en", set()))
    resources.chinese_stopwords.update(stopwords_by_lang.get("zh", set()))

    zh_stopwords_path = fallback_resource_path(
        inventory,
        "chinese_stopwords_goto456",
        "external_references/routing_vocab/chinese-stopwords-goto456",
    )
    zh_words, status = load_chinese_stopwords_goto456(zh_stopwords_path)
    resources.statuses["chinese_stopwords_goto456"] = status
    resources.chinese_stopwords.update(zh_words)

    en_dimlex_path = fallback_resource_path(inventory, "en_dimlex", "external_references/routing_vocab/en_dimlex")
    connectives, status = load_en_dimlex_connectives(en_dimlex_path)
    resources.statuses["en_dimlex"] = status
    resources.english_discourse_connectives.update(connectives)

    zh_dimlex_path = fallback_resource_path(inventory, "chinese_dimlex", "external_references/routing_vocab/chinese-dimlex")
    zh_connectives, status = load_chinese_dimlex_connectives(zh_dimlex_path)
    resources.statuses["chinese_dimlex"] = status
    resources.chinese_discourse_connectives.update(zh_connectives)
    return resources


def load_complex_scores(path: Path | None) -> tuple[dict[str, float], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {"status": "missing_resource"}
    values: defaultdict[str, list[float]] = defaultdict(list)
    for file_path in list(path.glob("**/lcp_single_*.tsv")) + list(path.glob("**/lcp_multi_*.tsv")):
        try:
            with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    token = normalize_token(str(row.get("token", "")))
                    value = row.get("complexity")
                    if token and value not in (None, ""):
                        values[token].append(float(value))
        except Exception:
            continue
    scores = {token: sum(items) / len(items) for token, items in values.items() if items}
    return scores, {"status": "available" if scores else "empty", "entry_count": len(scores)}


def load_word_importance(path: Path | None, resources: ExternalEnumerableResources) -> tuple[dict[str, float], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {"status": "missing_resource"}
    if module_status("pandas") != "available" or module_status("pyarrow") != "available":
        return {}, {"status": "package_missing", "packages": ["pandas", "pyarrow"]}
    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        return {}, {"status": "error", "error": type(exc).__name__}
    values: defaultdict[str, list[float]] = defaultdict(list)
    try:
        for file_path in path.glob("**/*.parquet"):
            frame = pd.read_parquet(file_path)
            for _, row in frame.iterrows():
                context = row.get("context")
                labels = row.get("label")
                if context is None or labels is None:
                    continue
                label_values = [float(value) for value in labels]
                if not label_values:
                    continue
                high = max(label_values)
                low = min(label_values)
                span = max(0.0001, high - low)
                for token, label in zip(context, label_values):
                    normalized = normalize_token(str(token))
                    if normalized and token_is_contentful(normalized, resources):
                        values[normalized].append(((high - float(label)) / span) * 10.0)
    except Exception as exc:
        return {}, {"status": "error", "error": type(exc).__name__}
    scores = {token: sum(items) / len(items) for token, items in values.items() if items}
    return scores, {"status": "available" if scores else "empty", "entry_count": len(scores)}


def load_mrc(path: Path | None) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {"status": "missing_resource"}
    csv_paths = list(path.glob("**/*.csv"))
    if not csv_paths:
        return {}, {"status": "missing_resource_file"}
    rows: dict[str, dict[str, float]] = {}
    try:
        with csv_paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                word = normalize_token(str(row.get("Word", "")))
                if not word:
                    continue
                metrics: dict[str, float] = {}
                for source, target in (
                    ("Familiarity", "familiarity"),
                    ("Concreteness", "concreteness"),
                    ("Imageability", "imageability"),
                    ("Age of Acquisition Rating", "aoa"),
                ):
                    raw = row.get(source)
                    try:
                        value = float(raw) if raw not in (None, "") else 0.0
                    except ValueError:
                        value = 0.0
                    if value > 0:
                        metrics[target] = clamp((value / 700.0) * 10.0)
                if metrics:
                    rows[word] = metrics
    except Exception as exc:
        return {}, {"status": "error", "error": type(exc).__name__}
    return rows, {"status": "available" if rows else "empty", "entry_count": len(rows)}


def load_concreteness(path: Path | None) -> tuple[dict[str, float], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {"status": "missing_resource"}
    csv_paths = list(path.glob("**/*.csv"))
    if not csv_paths:
        return {}, {"status": "missing_resource_file"}
    scores: dict[str, float] = {}
    try:
        with csv_paths[0].open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                word = normalize_token(str(row.get("Word", "")))
                raw = row.get("Conc.M")
                if not word or raw in (None, ""):
                    continue
                scores[word] = clamp(((float(raw) - 1.0) / 4.0) * 10.0)
    except Exception as exc:
        return {}, {"status": "error", "error": type(exc).__name__}
    return scores, {"status": "available" if scores else "empty", "entry_count": len(scores)}


def load_thuocl(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if path is None or not path.exists():
        return {}, {"status": "missing_resource"}
    terms: dict[str, dict[str, Any]] = {}
    for file_path in path.glob("data/THUOCL_*.txt"):
        category = file_path.stem.replace("THUOCL_", "")
        try:
            with file_path.open("r", encoding="utf-8-sig") as handle:
                for line in handle:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    term = parts[0]
                    if len(term) < 2:
                        continue
                    freq = 0
                    if len(parts) > 1:
                        try:
                            freq = int(float(parts[1]))
                        except ValueError:
                            freq = 0
                    current = terms.get(term)
                    if current is None or freq > int(current.get("frequency", 0)):
                        terms[term] = {"category": category, "frequency": freq}
        except Exception:
            continue
    return terms, {"status": "available" if terms else "empty", "entry_count": len(terms)}


def load_nrc(paths: list[Path | None]) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    existing_paths = [path for path in paths if path is not None and path.exists()]
    if not existing_paths:
        return {}, {"status": "missing_resource"}
    values: defaultdict[str, dict[str, float]] = defaultdict(dict)
    zip_paths: list[Path] = []
    for path in existing_paths:
        if path.is_file() and path.suffix.lower() == ".zip":
            zip_paths.append(path)
        elif path.is_dir():
            zip_paths.extend(path.glob("*.zip"))
    for zip_path in zip_paths:
        try:
            with zipfile.ZipFile(zip_path) as zf:
                for name in zf.namelist():
                    lowered = name.lower()
                    if "wordlevel" in lowered and lowered.endswith(".txt"):
                        with zf.open(name) as handle:
                            for raw_line in handle:
                                line = raw_line.decode("utf-8", errors="ignore").strip()
                                parts = line.split("\t")
                                if len(parts) >= 3:
                                    word = normalize_token(parts[0])
                                    label = parts[1].lower()
                                    try:
                                        score = float(parts[2])
                                    except ValueError:
                                        continue
                                    if word and score > 0:
                                        values[word][label] = max(values[word].get(label, 0.0), score)
                    elif ("vad" in lowered or "worry" in lowered or "intensity" in lowered) and lowered.endswith((".txt", ".csv")):
                        with zf.open(name) as handle:
                            for raw_line in handle:
                                line = raw_line.decode("utf-8", errors="ignore").strip()
                                if not line or line.lower().startswith("word"):
                                    continue
                                parts = re.split(r"[\t,]", line)
                                if len(parts) >= 2:
                                    word = normalize_token(parts[0])
                                    if not word:
                                        continue
                                    for idx, raw_value in enumerate(parts[1:4], 1):
                                        try:
                                            score = float(raw_value)
                                        except ValueError:
                                            continue
                                        values[word][f"nrc_numeric_{idx}"] = max(values[word].get(f"nrc_numeric_{idx}", 0.0), score)
        except Exception:
            continue
    result = {word: dict(metrics) for word, metrics in values.items()}
    return result, {"status": "available" if result else "empty", "entry_count": len(result), "zip_count": len(zip_paths)}


@dataclass
class ResourceStore:
    inventory: ResourceInventory
    complex_scores: dict[str, float] = field(default_factory=dict)
    word_importance: dict[str, float] = field(default_factory=dict)
    mrc: dict[str, dict[str, float]] = field(default_factory=dict)
    concreteness: dict[str, float] = field(default_factory=dict)
    thuocl: dict[str, dict[str, Any]] = field(default_factory=dict)
    nrc: dict[str, dict[str, float]] = field(default_factory=dict)
    statuses: dict[str, dict[str, Any]] = field(default_factory=dict)


def load_resource_store(inventory: ResourceInventory, enumerable_resources: ExternalEnumerableResources) -> ResourceStore:
    store = ResourceStore(inventory=inventory)
    store.complex_scores, store.statuses["complex"] = load_complex_scores(inventory.resource_path("complex"))
    store.word_importance, store.statuses["word_importance"] = load_word_importance(
        inventory.resource_path("word_importance"),
        enumerable_resources,
    )
    store.mrc, store.statuses["mrc"] = load_mrc(inventory.resource_path("mrc"))
    store.concreteness, store.statuses["concreteness_ratings_hf"] = load_concreteness(inventory.resource_path("concreteness_ratings_hf"))
    store.thuocl, store.statuses["thuocl"] = load_thuocl(inventory.resource_path("thuocl"))
    store.nrc, store.statuses["nrc"] = load_nrc(
        [
            inventory.resource_path("nrc_emotion"),
            inventory.resource_path("nrc_vad"),
            inventory.resource_path("nrc_intensity"),
            inventory.resource_path("worrywords"),
            inventory.resource_path("nrc"),
        ]
    )
    for package_resource, package_name in (
        ("wordfreq", "wordfreq"),
        ("empath", "empath"),
        ("textstat", "textstat"),
        ("vader", "vaderSentiment"),
        ("textblob", "textblob"),
        ("yake", "yake"),
        ("english_wordnet", "nltk"),
        ("omw", "nltk"),
        ("openhownet", "OpenHowNet"),
    ):
        note = inventory.resource_note(package_resource)
        store.statuses.setdefault(package_resource, {"status": module_status(package_name), **note})
    store.statuses.update(enumerable_resources.statuses)
    return store


def values_for_terms(tokens: list[str], table: dict[str, float]) -> list[float]:
    values = []
    for token in tokens:
        value = table.get(normalize_token(token))
        if value is not None:
            values.append(float(value))
    return values


def dict_values_for_terms(tokens: list[str], table: dict[str, dict[str, float]], metric: str) -> list[float]:
    values = []
    for token in tokens:
        item = table.get(normalize_token(token))
        if item and metric in item:
            values.append(float(item[metric]))
    return values


def wordfreq_features(tokens: list[str], resources: ExternalEnumerableResources) -> dict[str, Any]:
    if module_status("wordfreq") != "available":
        return {"status": "package_missing"}
    try:
        from wordfreq import zipf_frequency  # type: ignore
    except Exception as exc:
        return {"status": "error", "error": type(exc).__name__}
    en_values = []
    zh_values = []
    for token in tokens:
        normalized = normalize_token(token)
        if not token_is_contentful(normalized, resources):
            continue
        if re.match(r"[a-z]", normalized):
            en_values.append(float(zipf_frequency(normalized, "en")))
        elif is_chinese_token(normalized):
            zh_values.append(float(zipf_frequency(normalized, "zh")))
    min_en = min(en_values) if en_values else None
    min_zh = min(zh_values) if zh_values else None
    rarity_inputs = [value for value in [min_en, min_zh] if value is not None]
    rarity = max((clamp((4.5 - value) * 2.5) for value in rarity_inputs), default=None)
    commonness_inputs = en_values + zh_values
    commonness = mean(commonness_inputs)
    return {
        "status": "available" if commonness_inputs else "not_applicable",
        "min_zipf_en": round_or_none(min_en),
        "min_zipf_zh": round_or_none(min_zh),
        "avg_zipf": round_or_none(commonness),
        "rarity_score": round_or_none(rarity),
    }


def textstat_features(text: str) -> dict[str, Any]:
    if module_status("textstat") != "available":
        return {"status": "package_missing"}
    try:
        import textstat  # type: ignore

        grade = float(textstat.flesch_kincaid_grade(text))
        difficult = float(textstat.difficult_words(text))
        fog = float(textstat.gunning_fog(text))
        return {
            "status": "available",
            "flesch_kincaid_grade": round(grade, 3),
            "gunning_fog": round(fog, 3),
            "difficult_words": round(difficult, 3),
            "difficulty_score": round(clamp((max(grade, fog) - 6.0) / 8.0 * 10.0 + min(2.0, difficult / 8.0)), 3),
        }
    except Exception as exc:
        return {"status": "error", "error": type(exc).__name__}


def affect_features(text: str, tokens: list[str], store: ResourceStore) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "not_applicable"}
    scores = []
    if module_status("vaderSentiment") == "available":
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

            vader = SentimentIntensityAnalyzer().polarity_scores(text)
            result["vader"] = vader
            scores.append(abs(float(vader.get("compound", 0.0))) * 10.0)
        except Exception as exc:
            result["vader_error"] = type(exc).__name__
    else:
        result["vader_status"] = "package_missing"
    if module_status("textblob") == "available":
        try:
            from textblob import TextBlob  # type: ignore

            sentiment = TextBlob(text).sentiment
            result["textblob"] = {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}
            scores.append(abs(float(sentiment.polarity)) * 6.0 + float(sentiment.subjectivity) * 4.0)
        except Exception as exc:
            result["textblob_error"] = type(exc).__name__
    else:
        result["textblob_status"] = "package_missing"
    nrc_hits: Counter[str] = Counter()
    for token in tokens:
        metrics = store.nrc.get(normalize_token(token))
        if not metrics:
            continue
        for label, value in metrics.items():
            if value > 0:
                nrc_hits[label] += 1
    if nrc_hits:
        result["nrc_hits"] = dict(nrc_hits.most_common(12))
        scores.append(clamp(sum(nrc_hits.values()) * 1.5))
    result["status"] = "available" if scores or nrc_hits else "not_applicable"
    result["affect_score"] = round_or_none(clamp(max(scores) if scores else 0.0))
    return result


def empath_features(text: str) -> dict[str, Any]:
    if module_status("empath") != "available":
        return {"status": "package_missing"}
    try:
        from empath import Empath  # type: ignore

        categories = Empath().analyze(text, normalize=True)
        top = sorted(((name, score) for name, score in categories.items() if score), key=lambda item: item[1], reverse=True)[:10]
        return {
            "status": "available" if top else "not_applicable",
            "top_categories": [{"category": name, "score": round(float(score), 4)} for name, score in top],
            "category_score": round(clamp(sum(float(score) for _, score in top) * 8.0), 3),
        }
    except Exception as exc:
        return {"status": "error", "error": type(exc).__name__}


def yake_features(text: str) -> dict[str, Any]:
    if module_status("yake") != "available":
        return {"status": "package_missing"}
    try:
        import yake  # type: ignore

        language = "zh" if has_chinese(text) else "en"
        keyword_text = " ".join(tokenize(text)) if language == "zh" else text
        extractor = yake.KeywordExtractor(lan=language, n=2, top=10)
        keywords = [{"term": term, "score": round(float(score), 6)} for term, score in extractor.extract_keywords(keyword_text)]
        best = min((item["score"] for item in keywords), default=None)
        return {
            "status": "available" if keywords else "not_applicable",
            "keywords": keywords,
            "keyphrase_score": round_or_none(clamp((0.35 - best) * 20.0 if best is not None else 0.0)),
        }
    except Exception as exc:
        return {"status": "error", "error": type(exc).__name__}


def wordnet_features(tokens: list[str], project_root: Path, resources: ExternalEnumerableResources) -> dict[str, Any]:
    if module_status("nltk") != "available":
        return {"status": "package_missing"}
    try:
        import nltk  # type: ignore
        from nltk.corpus import wordnet as wn  # type: ignore

        data_path = project_root / "external_references" / "routing_vocab" / "nltk_data"
        if data_path.exists() and str(data_path) not in nltk.data.path:
            nltk.data.path.append(str(data_path))
        packages_path = data_path / "packages"
        if packages_path.exists() and str(packages_path) not in nltk.data.path:
            nltk.data.path.append(str(packages_path))
        synset_counts = {}
        for token in tokens:
            normalized = normalize_token(token)
            if re.match(r"[a-z]", normalized) and token_is_contentful(normalized, resources):
                count = len(wn.synsets(normalized))
                if count:
                    synset_counts[normalized] = count
        return {
            "status": "available" if synset_counts else "not_applicable",
            "synset_counts": dict(sorted(synset_counts.items(), key=lambda item: item[1], reverse=True)[:12]),
            "lexical_network_score": round(clamp(sum(min(3, value) for value in synset_counts.values()) / 2.0), 3),
        }
    except LookupError:
        return {"status": "missing_data"}
    except Exception as exc:
        return {"status": "error", "error": type(exc).__name__}


def thuocl_features(tokens: list[str], text: str, store: ResourceStore) -> dict[str, Any]:
    if not store.thuocl:
        return {"status": store.statuses.get("thuocl", {}).get("status", "missing_resource")}
    hits: dict[str, dict[str, Any]] = {}
    token_set = {token for token in tokens if is_chinese_token(token)}
    for token in token_set:
        meta = store.thuocl.get(token)
        if meta:
            hits[token] = meta
    if has_chinese(text):
        candidates: set[str] = set()
        for chunk in re.findall(r"[\u4e00-\u9fff]+", text):
            for size in (2, 3, 4, 5):
                for idx in range(0, max(0, len(chunk) - size + 1)):
                    candidates.add(chunk[idx : idx + size])
        for candidate in candidates:
            meta = store.thuocl.get(candidate)
            if meta:
                hits.setdefault(candidate, meta)
            if len(hits) >= 30:
                break
    categories = Counter(str(meta.get("category", "unknown")) for meta in hits.values())
    frequency_score = sum(math.log10(max(1, int(meta.get("frequency", 0)))) for meta in hits.values())
    return {
        "status": "available" if hits else "not_applicable",
        "hit_count": len(hits),
        "category_counts": dict(categories.most_common(8)),
        "hits": [{"term": term, **meta} for term, meta in list(hits.items())[:20]],
        "domain_term_score": round(clamp(len(hits) * 1.5 + frequency_score / 6.0), 3),
    }


def lexical_resource_features(tokens: list[str], store: ResourceStore, resources: ExternalEnumerableResources) -> dict[str, Any]:
    content_tokens = [normalize_token(token) for token in tokens if token_is_contentful(token, resources)]
    importance_values = values_for_terms(content_tokens, store.word_importance)
    complex_values = values_for_terms(content_tokens, store.complex_scores)
    familiarity_values = dict_values_for_terms(content_tokens, store.mrc, "familiarity")
    mrc_concrete_values = dict_values_for_terms(content_tokens, store.mrc, "concreteness")
    aoa_values = dict_values_for_terms(content_tokens, store.mrc, "aoa")
    image_values = dict_values_for_terms(content_tokens, store.mrc, "imageability")
    concreteness_values = values_for_terms(content_tokens, store.concreteness)
    return {
        "word_importance": {
            "status": "available" if importance_values else store.statuses.get("word_importance", {}).get("status", "missing_resource"),
            "score": round_or_none(mean(importance_values)),
            "coverage_count": len(importance_values),
        },
        "complex": {
            "status": "available" if complex_values else store.statuses.get("complex", {}).get("status", "missing_resource"),
            "lexical_complexity_score": round_or_none((mean(complex_values) or 0.0) * 10.0) if complex_values else None,
            "coverage_count": len(complex_values),
        },
        "mrc": {
            "status": "available" if (familiarity_values or mrc_concrete_values or aoa_values) else store.statuses.get("mrc", {}).get("status", "missing_resource"),
            "familiarity_score": round_or_none(mean(familiarity_values)),
            "concreteness_score": round_or_none(mean(mrc_concrete_values)),
            "imageability_score": round_or_none(mean(image_values)),
            "aoa_score": round_or_none(mean(aoa_values)),
            "coverage_count": max(len(familiarity_values), len(mrc_concrete_values), len(aoa_values), len(image_values)),
        },
        "concreteness": {
            "status": "available" if concreteness_values else store.statuses.get("concreteness_ratings_hf", {}).get("status", "missing_resource"),
            "score": round_or_none(mean(concreteness_values)),
            "coverage_count": len(concreteness_values),
        },
    }


def tfidf_features(text_by_id: dict[str, str], resources: ExternalEnumerableResources) -> dict[str, dict[str, Any]]:
    if module_status("sklearn") != "available" or not text_by_id:
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except Exception:
        return {}
    ids = list(text_by_id.keys())
    texts = [text_by_id[row_id] for row_id in ids]

    def analyzer(text: str) -> list[str]:
        tokens = [normalize_token(token) for token in tokenize(text) if token_is_contentful(token, resources)]
        bigrams = [f"{tokens[idx]} {tokens[idx + 1]}" for idx in range(len(tokens) - 1)]
        return tokens + bigrams

    try:
        vectorizer = TfidfVectorizer(lowercase=True, analyzer=analyzer, max_features=4000, min_df=1)
        matrix = vectorizer.fit_transform(texts)
        names = vectorizer.get_feature_names_out()
    except ValueError:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for idx, row_id in enumerate(ids):
        row = matrix.getrow(idx)
        if row.nnz == 0:
            result[row_id] = {"status": "not_applicable", "top_terms": [], "max_tfidf": 0.0, "sum_tfidf": 0.0, "score": 0.0}
            continue
        pairs = sorted(zip(row.indices, row.data), key=lambda item: item[1], reverse=True)[:10]
        sum_tfidf = float(sum(row.data))
        result[row_id] = {
            "status": "available",
            "top_terms": [{"term": str(names[col]), "score": round(float(score), 4)} for col, score in pairs],
            "max_tfidf": round(float(max(row.data)), 4),
            "sum_tfidf": round(sum_tfidf, 4),
            "score": round(clamp(sum_tfidf / 3.0 * 10.0), 3),
        }
    return result


def route_snapshot_for(row: dict[str, Any], route_map: dict[str, dict[str, str]]) -> dict[str, str]:
    source_specific_refs = row.get("source_specific_refs")
    first_source_ref = source_specific_refs[0] if isinstance(source_specific_refs, list) and source_specific_refs else None
    candidates = [
        row.get("text_unit_id"),
        first_source_ref,
        row.get("raw_span_id"),
        row.get("evidence_ref"),
        row.get("canonical_evidence_ref"),
    ]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate in route_map:
            return route_map[candidate]
    return {}


def route_by_target(row: dict[str, Any]) -> dict[str, str]:
    routes: dict[str, str] = {}
    if isinstance(row.get("task_routes"), list):
        for route in row["task_routes"]:
            if isinstance(route, dict) and route.get("target_task"):
                routes[str(route["target_task"])] = str(route.get("recommended_route", ""))
    elif row.get("target_task"):
        routes[str(row["target_task"])] = str(row.get("recommended_route") or row.get("route_recommended") or row.get("route_used") or "")
    return {key: value for key, value in routes.items() if value}


def load_route_map(workspace: Path) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    route_dir = workspace / "routing"
    if not route_dir.exists():
        return {}, []
    route_files = sorted(route_dir.glob("**/route_decisions.jsonl"), key=lambda path: path.stat().st_mtime)
    route_map: dict[str, dict[str, str]] = {}
    sources: list[dict[str, Any]] = []
    for path in route_files:
        sources.append({"path": str(path), "sha256": sha256_file(path), "mtime": path.stat().st_mtime})
        for row in read_jsonl(path):
            routes = route_by_target(row)
            if not routes:
                continue
            for key in ("text_unit_id", "span_id", "raw_span_id", "evidence_ref", "raw_span_ref"):
                value = row.get(key)
                if isinstance(value, str) and value.strip():
                    route_map.setdefault(value.strip(), {}).update(routes)
    return route_map, sources


def collect_input_units(workspace: Path, max_items: int | None = None) -> tuple[list[InputUnit], list[dict[str, Any]]]:
    units: list[InputUnit] = []
    source_assets: list[dict[str, Any]] = []
    workspace_id = workspace.name
    for source_layer, relative in DEFAULT_INPUTS:
        path = workspace / relative
        if not path.exists():
            continue
        rows = read_jsonl(path)
        source_assets.append({"source_layer": source_layer, "path": str(path), "sha256": sha256_file(path), "row_count": len(rows)})
        for idx, row in enumerate(rows):
            text = safe_text(row)
            if not text:
                continue
            unit_id = unit_id_for_row(row, f"{source_layer}:{idx + 1:06d}")
            units.append(
                InputUnit(
                    source_layer=source_layer,
                    source_path=path,
                    source_row_index=idx,
                    unit_id=unit_id,
                    workspace_id=str(row.get("workspace_id") or workspace_id),
                    text=text,
                    row=row,
                )
            )
            if max_items and len(units) >= max_items:
                return units, source_assets
    return units, source_assets


def derive_axis_scores(features: dict[str, Any]) -> dict[str, float]:
    wordfreq = features["wordfreq"]
    lexical = features["lexical_resources"]
    textstat = features["textstat"]
    sentence = features["sentence_complexity"]
    empath = features["empath"]
    affect = features["affect"]
    thuocl = features["thuocl"]
    wordnet = features["wordnet"]
    yake = features["yake"]
    tfidf = features["tfidf"]
    entity = features["entity"]
    language = features["language"]

    rarity = float(wordfreq.get("rarity_score") or 0.0)
    importance = float((lexical.get("word_importance") or {}).get("score") or 0.0)
    complex_score = float((lexical.get("complex") or {}).get("lexical_complexity_score") or 0.0)
    familiarity = float((lexical.get("mrc") or {}).get("familiarity_score") or 0.0)
    aoa = float((lexical.get("mrc") or {}).get("aoa_score") or 0.0)
    concrete = max(
        float((lexical.get("mrc") or {}).get("concreteness_score") or 0.0),
        float((lexical.get("concreteness") or {}).get("score") or 0.0),
    )
    readability = float(textstat.get("difficulty_score") or 0.0)
    sentence_score = float(sentence.get("score") or 0.0)
    domain = float(thuocl.get("domain_term_score") or 0.0)
    lexical_network = float(wordnet.get("lexical_network_score") or 0.0)
    keyphrase = max(float(yake.get("keyphrase_score") or 0.0), float(tfidf.get("score") or 0.0))
    affect_score = float(affect.get("affect_score") or 0.0)
    empath_score = float(empath.get("category_score") or 0.0)
    entity_score = float(entity.get("score") or 0.0)
    content_tokens = int(language.get("content_token_count") or 0)
    avg_zipf = wordfreq.get("avg_zipf")
    common_high = isinstance(avg_zipf, (int, float)) and float(avg_zipf) >= 5.0
    low_value_score = 0.0
    if content_tokens == 0:
        low_value_score = 10.0
    elif content_tokens <= 2 and common_high and entity_score == 0 and keyphrase < 2:
        low_value_score = 8.0
    elif content_tokens <= 3 and entity_score == 0 and keyphrase < 2:
        low_value_score = 5.5

    complexity_score = max(complex_score, readability, sentence_score, clamp(aoa + max(0.0, 5.0 - familiarity) / 2.0))
    value_score = clamp(
        importance * 0.25
        + rarity * 0.15
        + entity_score * 0.20
        + domain * 0.15
        + keyphrase * 0.20
        + concrete * 0.05
        + lexical_network * 0.05
        - low_value_score * 0.35
    )
    risk_score = clamp(affect_score * 0.35 + empath_score * 0.15 + sentence_score * 0.10 + entity_score * 0.10)
    return {
        "value_score": round(value_score, 3),
        "risk_score": round(risk_score, 3),
        "complexity_score": round(clamp(complexity_score), 3),
        "entity_salience_score": round(clamp(entity_score), 3),
        "keyphrase_score": round(clamp(keyphrase), 3),
        "affect_score": round(clamp(max(affect_score, empath_score)), 3),
        "low_value_score": round(clamp(low_value_score), 3),
        "lexical_complexity_score": round(clamp(complex_score), 3),
        "sentence_complexity_score": round(clamp(sentence_score), 3),
        "domain_term_score": round(clamp(domain), 3),
    }


def build_feature_row(
    unit: InputUnit,
    store: ResourceStore,
    enumerable_resources: ExternalEnumerableResources,
    route_map: dict[str, dict[str, str]],
    tfidf: dict[str, Any],
) -> dict[str, Any]:
    tokens = tokenize(unit.text)
    normalized_tokens = [normalize_token(token) for token in tokens]
    language = language_scores(unit.text, tokens, enumerable_resources)
    lexical = lexical_resource_features(normalized_tokens, store, enumerable_resources)
    wordfreq = wordfreq_features(normalized_tokens, enumerable_resources)
    textstat = textstat_features(unit.text)
    sentence = local_sentence_complexity(unit.text, normalized_tokens, enumerable_resources)
    affect = affect_features(unit.text, normalized_tokens, store)
    empath = empath_features(unit.text)
    yake = yake_features(unit.text)
    thuocl = thuocl_features(tokens, unit.text, store)
    wordnet = wordnet_features(normalized_tokens, store.inventory.project_root, enumerable_resources)
    proper_names = capitalized_entity_candidates(unit.text, enumerable_resources)
    entity_score = clamp(len(proper_names) * 1.4 + float(thuocl.get("domain_term_score") or 0.0) * 0.5)
    features = {
        "language": language,
        "wordfreq": wordfreq,
        "lexical_resources": lexical,
        "textstat": textstat,
        "sentence_complexity": sentence,
        "affect": affect,
        "empath": empath,
        "yake": yake,
        "thuocl": thuocl,
        "wordnet": wordnet,
        "tfidf": tfidf,
        "entity": {"proper_name_candidates": proper_names, "score": round(entity_score, 3)},
        "external_enumerables": {
            "stopword_sources": enumerable_resources.stopword_source_status(),
            "connective_sources": enumerable_resources.connective_source_status(),
            "english_stopword_count": len(enumerable_resources.english_stopwords),
            "chinese_stopword_count": len(enumerable_resources.chinese_stopwords),
            "english_discourse_connective_count": len(enumerable_resources.english_discourse_connectives),
            "chinese_discourse_connective_count": len(enumerable_resources.chinese_discourse_connectives),
        },
    }
    axes = derive_axis_scores(features)
    statuses = {
        "wordfreq": wordfreq.get("status"),
        "word_importance": lexical["word_importance"].get("status"),
        "complex": lexical["complex"].get("status"),
        "mrc": lexical["mrc"].get("status"),
        "concreteness": lexical["concreteness"].get("status"),
        "textstat": textstat.get("status"),
        "empath": empath.get("status"),
        "affect": affect.get("status"),
        "thuocl": thuocl.get("status"),
        "wordnet": wordnet.get("status"),
        "yake": yake.get("status"),
        "tfidf": tfidf.get("status", "missing"),
        "stopwords_iso": enumerable_resources.statuses.get("stopwords_iso", {}).get("status"),
        "chinese_stopwords_goto456": enumerable_resources.statuses.get("chinese_stopwords_goto456", {}).get("status"),
        "en_dimlex": enumerable_resources.statuses.get("en_dimlex", {}).get("status"),
        "chinese_dimlex": enumerable_resources.statuses.get("chinese_dimlex", {}).get("status"),
    }
    warnings = []
    if all(value in {"missing_resource", "package_missing", "missing", "empty"} for value in statuses.values() if value):
        warnings.append("no_external_resource_features_available")
    if unit.source_layer == "s2_proposal_backed_unit":
        warnings.append("s2_unit_scores_are_diagnostics_not_truth")
    return {
        "schema_version": SCHEMA_VERSION,
        "workspace_id": unit.workspace_id,
        "source_layer": unit.source_layer,
        "source_path": str(unit.source_path),
        "source_row_index": unit.source_row_index,
        "unit_id": unit.unit_id,
        "unit_type": unit.row.get("unit_type") or unit.row.get("item_layer") or unit.row.get("candidate_type") or "unknown",
        "text_preview": unit.text[:300],
        "text_char_count": len(unit.text),
        "route_snapshot": route_snapshot_for(unit.row, route_map),
        "feature_status": statuses,
        "raw_features": features,
        "axis_scores": axes,
        "feature_provenance": {
            "source": "v0.21 resource-backed feature matrix",
            "route_authority": False,
            "provider_calls": False,
            "resource_ids": sorted(store.statuses.keys()),
        },
        "warnings": warnings,
        "write_permission": False,
    }


CSV_COLUMNS = [
    "workspace_id",
    "source_layer",
    "unit_id",
    "unit_type",
    "text_char_count",
    "language_primary",
    "language_zh_score",
    "language_en_score",
    "token_count",
    "content_token_count",
    "value_score",
    "risk_score",
    "complexity_score",
    "entity_salience_score",
    "keyphrase_score",
    "affect_score",
    "low_value_score",
    "lexical_complexity_score",
    "sentence_complexity_score",
    "domain_term_score",
    "wordfreq_status",
    "wordfreq_min_zipf_en",
    "wordfreq_min_zipf_zh",
    "wordfreq_rarity_score",
    "word_importance_status",
    "word_importance_score",
    "complex_status",
    "complex_lexical_complexity_score",
    "mrc_status",
    "mrc_familiarity_score",
    "mrc_concreteness_score",
    "mrc_aoa_score",
    "concreteness_status",
    "concreteness_score",
    "textstat_status",
    "textstat_grade",
    "empath_status",
    "thuocl_status",
    "thuocl_hit_count",
    "wordnet_status",
    "yake_status",
    "tfidf_status",
    "route_snapshot",
    "top_terms",
    "text_preview",
]


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    raw = row["raw_features"]
    axes = row["axis_scores"]
    lexical = raw["lexical_resources"]
    top_terms = []
    top_terms.extend(item.get("term") for item in raw.get("tfidf", {}).get("top_terms", [])[:5])
    top_terms.extend(item.get("term") for item in raw.get("yake", {}).get("keywords", [])[:5])
    top_terms.extend(item.get("term") for item in raw.get("thuocl", {}).get("hits", [])[:5])
    flat = {
        "workspace_id": row["workspace_id"],
        "source_layer": row["source_layer"],
        "unit_id": row["unit_id"],
        "unit_type": row["unit_type"],
        "text_char_count": row["text_char_count"],
        "language_primary": raw["language"].get("primary"),
        "language_zh_score": raw["language"].get("zh_score"),
        "language_en_score": raw["language"].get("en_score"),
        "token_count": raw["language"].get("token_count"),
        "content_token_count": raw["language"].get("content_token_count"),
        **axes,
        "wordfreq_status": raw["wordfreq"].get("status"),
        "wordfreq_min_zipf_en": raw["wordfreq"].get("min_zipf_en"),
        "wordfreq_min_zipf_zh": raw["wordfreq"].get("min_zipf_zh"),
        "wordfreq_rarity_score": raw["wordfreq"].get("rarity_score"),
        "word_importance_status": lexical["word_importance"].get("status"),
        "word_importance_score": lexical["word_importance"].get("score"),
        "complex_status": lexical["complex"].get("status"),
        "complex_lexical_complexity_score": lexical["complex"].get("lexical_complexity_score"),
        "mrc_status": lexical["mrc"].get("status"),
        "mrc_familiarity_score": lexical["mrc"].get("familiarity_score"),
        "mrc_concreteness_score": lexical["mrc"].get("concreteness_score"),
        "mrc_aoa_score": lexical["mrc"].get("aoa_score"),
        "concreteness_status": lexical["concreteness"].get("status"),
        "concreteness_score": lexical["concreteness"].get("score"),
        "textstat_status": raw["textstat"].get("status"),
        "textstat_grade": raw["textstat"].get("flesch_kincaid_grade"),
        "empath_status": raw["empath"].get("status"),
        "thuocl_status": raw["thuocl"].get("status"),
        "thuocl_hit_count": raw["thuocl"].get("hit_count"),
        "wordnet_status": raw["wordnet"].get("status"),
        "yake_status": raw["yake"].get("status"),
        "tfidf_status": raw["tfidf"].get("status"),
        "route_snapshot": json.dumps(row["route_snapshot"], ensure_ascii=False, sort_keys=True),
        "top_terms": json.dumps([term for term in top_terms if term], ensure_ascii=False),
        "text_preview": row["text_preview"],
    }
    return {column: flat.get(column) for column in CSV_COLUMNS}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))


def render_report(workspace: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> str:
    source_counts = Counter(row["source_layer"] for row in rows)
    status_counts: Counter[str] = Counter()
    language_counts = Counter(row["raw_features"]["language"]["primary"] for row in rows)
    for row in rows:
        for key, value in row["feature_status"].items():
            status_counts[f"{key}:{value}"] += 1
    top_value = sorted(rows, key=lambda item: item["axis_scores"]["value_score"], reverse=True)[:8]
    high_risk = sorted(rows, key=lambda item: item["axis_scores"]["risk_score"], reverse=True)[:8]
    high_low = sorted(rows, key=lambda item: item["axis_scores"]["low_value_score"], reverse=True)[:8]

    def sample_lines(title: str, sample_rows: list[dict[str, Any]], score_key: str) -> list[str]:
        lines = [f"## {title}", ""]
        if not sample_rows:
            return lines + ["- None.", ""]
        for row in sample_rows:
            lines.append(
                f"- `{row['source_layer']}` `{row['unit_id']}` "
                f"{score_key}={row['axis_scores'][score_key]}: {row['text_preview'][:140]}"
            )
        lines.append("")
        return lines

    lines = [
        "# v0.21 Feature Matrix Report",
        "",
        f"Workspace: `{workspace.name}`",
        "",
        "本报告只生成资源特征矩阵，不替换路由、不调用 provider、不写 durable memory、不写 graph。",
        "",
        "## Counts",
        "",
        f"- rows: {len(rows)}",
        f"- output_dir: `{manifest['output_dir']}`",
        "",
        "## Source Layers",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in sorted(source_counts.items()))
    lines.extend(["", "## Languages", ""])
    lines.extend(f"- {key}: {value}" for key, value in sorted(language_counts.items()))
    lines.extend(["", "## Resource Status Coverage", ""])
    lines.extend(f"- {key}: {value}" for key, value in sorted(status_counts.items()))
    lines.extend([""])
    lines.extend(sample_lines("High Value Diagnostics", top_value, "value_score"))
    lines.extend(sample_lines("High Risk Diagnostics", high_risk, "risk_score"))
    lines.extend(sample_lines("Low Value / Boilerplate Diagnostics", high_low, "low_value_score"))
    lines.extend(
        [
            "## Boundary",
            "",
            "- These scores are routing diagnostics, not truth.",
            "- Missing resources are reported as status fields instead of being converted into silent zero evidence.",
            "- S1 and S2 scoring profiles are intentionally not applied in this slice.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, str]:
    project_root = resolve_project_path(Path.cwd(), getattr(args, "project_root", "."))
    workspace = resolve_project_path(project_root, args.workspace)
    inventory_path = resolve_project_path(project_root, getattr(args, "resource_inventory", DEFAULT_RESOURCE_INVENTORY))
    output_dir_arg = getattr(args, "output_dir", None)
    output_dir = resolve_project_path(project_root, output_dir_arg) if output_dir_arg else workspace / "routing" / DEFAULT_OUTPUT_NAME
    max_items = getattr(args, "max_items", None) or None

    inventory = load_resource_inventory(project_root, inventory_path)
    enumerable_resources = load_external_enumerable_resources(inventory)
    store = load_resource_store(inventory, enumerable_resources)
    units, source_assets = collect_input_units(workspace, max_items=max_items)
    route_map, route_assets = load_route_map(workspace)
    tfidf_by_id = tfidf_features({unit.unit_id: unit.text for unit in units}, enumerable_resources)
    rows = [
        build_feature_row(
            unit,
            store,
            enumerable_resources,
            route_map,
            tfidf_by_id.get(unit.unit_id, {"status": "missing"}),
        )
        for unit in units
    ]

    matrix_jsonl = output_dir / "v021_feature_matrix.jsonl"
    matrix_csv = output_dir / "v021_feature_matrix.csv"
    manifest_path = output_dir / "v021_feature_matrix_manifest.json"
    report_path = output_dir / "v021_feature_matrix_report.md"

    manifest = {
        "schema_version": "routing.v021_feature_matrix_manifest.v0.1",
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "resource_inventory": str(inventory_path),
        "resource_inventory_hash": sha256_file(inventory_path) if inventory_path.exists() else None,
        "resource_statuses": store.statuses,
        "source_assets": source_assets,
        "route_assets": route_assets,
        "row_count": len(rows),
        "provider_calls": False,
        "route_decisions_replaced": False,
        "write_permission": False,
        "outputs": {
            "matrix_jsonl": str(matrix_jsonl),
            "matrix_csv": str(matrix_csv),
            "manifest": str(manifest_path),
            "report": str(report_path),
        },
    }
    write_jsonl(matrix_jsonl, rows)
    write_csv(matrix_csv, rows)
    write_json(manifest_path, manifest)
    write_text(report_path, render_report(workspace, rows, manifest))
    return manifest["outputs"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a read-only v0.21 routing feature matrix.")
    parser.add_argument("--project-root", default=".", help="Project root. Defaults to current working directory.")
    parser.add_argument("--workspace", required=True, help="Workspace path, absolute or relative to project root.")
    parser.add_argument("--resource-inventory", default=DEFAULT_RESOURCE_INVENTORY, help="Resource inventory CSV path.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to <workspace>/routing/v021_feature_matrix.")
    parser.add_argument("--max-items", type=int, default=0, help="Optional row cap for smoke runs.")
    return parser


def main() -> int:
    outputs = run(build_arg_parser().parse_args())
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
