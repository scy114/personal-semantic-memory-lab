"""Dry-run salience sidecar for memory proposal router outputs.

This evaluates router v0.2 salience signals without changing route decisions.
It is intentionally read-only with respect to canonical memory assets.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_LEXICON = "configs/routing/lexicons/salience_core_zh_en_v0.2.yaml"
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "or",
    "our",
    "she",
    "that",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}
OPTIONAL_WHEELS = (
    "jieba",
    "opencc",
    "textstat",
    "vaderSentiment",
    "empath",
    "wordfreq",
    "textblob",
    "spacy",
    "yake",
    "keybert",
    "sklearn",
    "nltk",
    "textacy",
    "lexical_diversity",
    "readability",
    "sentence_transformers",
)
SUBORDINATION_MARKERS = {
    "although",
    "because",
    "before",
    "after",
    "while",
    "when",
    "whereas",
    "unless",
    "since",
    "if",
    "though",
    "even though",
    "as long as",
    "so that",
}
COORDINATION_MARKERS = {"and", "but", "or", "nor", "yet", "so"}
CHINESE_STOPWORDS = {
    "了",
    "的",
    "地",
    "得",
    "和",
    "与",
    "或",
    "是",
    "在",
    "我",
    "我们",
    "你",
    "你们",
    "他",
    "她",
    "它",
    "他们",
    "她们",
    "这",
    "这个",
    "那",
    "那个",
    "一个",
    "一家",
    "正在",
}


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


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def safe_text(row: dict[str, Any]) -> str:
    for key in ("text", "content", "evidence_quote", "summary_text"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def has_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def normalize_chinese_text(text: str) -> str:
    if not has_chinese(text) or importlib.util.find_spec("opencc") is None:
        return text
    try:
        from opencc import OpenCC  # type: ignore

        return str(OpenCC("t2s").convert(text))
    except Exception:
        return text


def tokenize(text: str) -> list[str]:
    text = normalize_chinese_text(text)
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]*", text)
    chinese_chunks = re.findall(r"[\u4e00-\u9fff]+", text)
    if chinese_chunks and importlib.util.find_spec("jieba") is not None:
        try:
            import jieba  # type: ignore

            jieba.setLogLevel(logging.ERROR)
            for chunk in chinese_chunks:
                tokens.extend(token for token in jieba.lcut(chunk) if token.strip())
        except Exception:
            tokens.extend(chinese_chunks)
    else:
        tokens.extend(chinese_chunks)
    return tokens


def normalize_token(token: str) -> str:
    return token.strip("'_-").lower()


def is_chinese_token(token: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", token))


def token_is_contentful(token: str) -> bool:
    normalized = normalize_token(token)
    if not normalized:
        return False
    if is_chinese_token(normalized):
        return normalized not in CHINESE_STOPWORDS and len(normalized) >= 2
    return len(normalized) > 2 and normalized not in STOPWORDS


def optional_wheel_status() -> dict[str, str]:
    return {name: ("available" if importlib.util.find_spec(name) else "missing") for name in OPTIONAL_WHEELS}


def english_term_matches(text: str, term: str) -> list[tuple[int, int]]:
    escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
    return [(m.start(), m.end()) for m in re.finditer(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", text.lower())]


def chinese_term_matches(text: str, term: str) -> list[tuple[int, int]]:
    matches: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(term, start)
        if idx < 0:
            return matches
        matches.append((idx, idx + len(term)))
        start = idx + max(1, len(term))


@dataclass
class LexiconTerm:
    term: str
    group: str
    base_weight: float
    source: str
    license_note: str
    confidence: str
    language: str
    polarity: str


def load_static_terms(path: Path) -> list[LexiconTerm]:
    data = load_yaml(path)
    terms: list[LexiconTerm] = []
    for group_name, group in (data.get("groups") or {}).items():
        for language in ("english", "chinese"):
            for term in group.get(language, []) or []:
                terms.append(
                    LexiconTerm(
                        term=str(term),
                        group=str(group_name),
                        base_weight=float(group.get("base_weight", 0)),
                        source=str(group.get("source", "unknown")),
                        license_note=str(group.get("license_note", "")),
                        confidence=str(group.get("confidence", "unknown")),
                        language=language,
                        polarity=str(group.get("polarity", "neutral")),
                    )
                )
    return terms


def collect_input_rows(workspace: Path) -> list[dict[str, Any]]:
    text_units = read_jsonl(workspace / "raw" / "organization" / "text_units.jsonl")
    if text_units:
        sentence_units = [row for row in text_units if row.get("unit_type") == "sentence"]
        return sentence_units or [row for row in text_units if row.get("unit_type") == "paragraph"] or text_units
    evidence = read_jsonl(workspace / "evidence" / "evidence.jsonl")
    if evidence:
        return evidence
    return read_jsonl(workspace / "raw" / "organization" / "section_map.jsonl")


def collect_dynamic_terms(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates: Counter[str] = Counter()
    high_confidence: set[str] = set()
    for row in rows:
        for key in ("speaker", "participant", "target_participant"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                candidates[value.strip()] += 3
                high_confidence.add(value.strip())
        for key in ("participant_ids", "subject_ids", "target_subject_ids"):
            values = row.get(key) or []
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str) and value.strip():
                        candidates[value.strip()] += 3
                        high_confidence.add(value.strip())
        text = safe_text(row)
        for match in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", text):
            if normalize_token(match) not in STOPWORDS:
                candidates[match] += 1
    dynamic: dict[str, dict[str, Any]] = {}
    for term, count in candidates.items():
        if len(term) < 2:
            continue
        confidence = "high" if term in high_confidence else ("medium" if count >= 2 else "low")
        if confidence == "low" and count < 2:
            continue
        dynamic[term] = {
            "base_weight": 4 if confidence == "high" else 3,
            "group": "dynamic_entity_terms",
            "source": "metadata_or_capitalized_phrase",
            "confidence": confidence,
            "count": count,
            "license_note": "derived from local workspace text/metadata",
            "polarity": "neutral",
        }
    return dynamic


def route_by_target(row: dict[str, Any]) -> dict[str, str]:
    routes: dict[str, str] = {}
    for route in row.get("task_routes") or []:
        target = route.get("target_task")
        if target:
            routes[str(target)] = str(route.get("recommended_route"))
    return routes


def route_decision_map(workspace: Path) -> dict[str, dict[str, str]]:
    paths = list((workspace / "routing").glob("**/route_decisions.jsonl"))
    if not paths:
        return {}
    best = max(paths, key=lambda p: p.stat().st_mtime)
    mapping: dict[str, dict[str, str]] = {}
    for row in read_jsonl(best):
        span_id = row.get("text_unit_id") or row.get("span_id") or row.get("evidence_ref") or row.get("raw_span_id")
        if span_id:
            mapping[str(span_id)] = route_by_target(row)
    return mapping


def match_terms(text: str, static_terms: list[LexiconTerm], dynamic_terms: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for item in static_terms:
        spans = english_term_matches(text, item.term) if item.language == "english" else chinese_term_matches(text, item.term)
        for start, end in spans:
            hits.append(
                {
                    "term": item.term,
                    "group": item.group,
                    "base_weight": item.base_weight,
                    "source": item.source,
                    "confidence": item.confidence,
                    "license_note": item.license_note,
                    "polarity": item.polarity,
                    "span": [start, end],
                }
            )
    for term, meta in dynamic_terms.items():
        for start, end in english_term_matches(text, term):
            hits.append(
                {
                    "term": term,
                    "group": meta["group"],
                    "base_weight": meta["base_weight"],
                    "source": meta["source"],
                    "confidence": meta["confidence"],
                    "license_note": meta["license_note"],
                    "polarity": meta["polarity"],
                    "span": [start, end],
                }
            )
    return hits


def local_keyphrases(text: str) -> list[str]:
    tokens = [normalize_token(t) for t in tokenize(text)]
    tokens = [t for t in tokens if token_is_contentful(t)]
    counts = Counter(tokens)
    return [term for term, _ in counts.most_common(5)]


def sentence_split(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r"[.!?。！？]+", text) if part.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def sentence_complexity_metrics(text: str) -> dict[str, Any]:
    tokens = [normalize_token(t) for t in tokenize(text)]
    words = [t for t in tokens if t]
    sentences = sentence_split(text)
    lower = f" {text.lower()} "
    subordination_count = sum(len(re.findall(rf"(?<![a-z]){re.escape(marker)}(?![a-z])", lower)) for marker in SUBORDINATION_MARKERS)
    coordination_count = sum(1 for token in words if token in COORDINATION_MARKERS)
    punctuation_count = sum(text.count(ch) for ch in [",", ";", ":", "，", "；", "："])
    parenthetical_count = sum(text.count(ch) for ch in ["(", ")", "[", "]", "（", "）"])
    quote_count = sum(text.count(ch) for ch in ['"', "'", "“", "”", "‘", "’"])
    avg_sentence_length = len(words) / max(1, len(sentences))
    marker_score = subordination_count * 2 + coordination_count + min(3, punctuation_count)
    length_score = 0
    if avg_sentence_length >= 18:
        length_score += 2
    if avg_sentence_length >= 30:
        length_score += 2
    if parenthetical_count:
        marker_score += 1
    reliability = "empty" if not words else ("low_short_text" if len(words) < 10 else "standard")
    return {
        "subordination_marker_count": subordination_count,
        "coordination_marker_count": coordination_count,
        "clause_punctuation_count": punctuation_count,
        "parenthetical_marker_count": parenthetical_count,
        "quote_marker_count": quote_count,
        "avg_sentence_length": round(avg_sentence_length, 3) if words else 0.0,
        "syntactic_complexity_score": min(10, marker_score + length_score) if reliability == "standard" else 0,
        "syntactic_complexity_reliability": reliability,
    }


def count_syllables(word: str) -> int:
    word = normalize_token(word)
    if not word:
        return 0
    groups = re.findall(r"[aeiouy]+", word)
    count = len(groups)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability_metrics(text: str) -> dict[str, Any]:
    words = [normalize_token(t) for t in tokenize(text) if re.match(r"[A-Za-z]", t)]
    sentences = sentence_split(text)
    word_count = len(words)
    sentence_count = max(1, len(sentences))
    if word_count == 0:
        return {
            "word_count": 0,
            "sentence_count": sentence_count,
            "avg_sentence_length": 0.0,
            "avg_syllables_per_word": 0.0,
            "flesch_reading_ease": None,
            "flesch_kincaid_grade": None,
            "automated_readability_index": None,
            "coleman_liau_index": None,
            "gunning_fog": None,
            "smog_index": None,
            "complex_word_count": 0,
            "readability_reliability": "empty",
        }
    char_count = sum(len(w) for w in words)
    syllables = sum(count_syllables(w) for w in words)
    complex_words = [w for w in words if count_syllables(w) >= 3]
    avg_sentence_length = word_count / sentence_count
    avg_syllables = syllables / word_count
    fre = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables)
    fk = (0.39 * avg_sentence_length) + (11.8 * avg_syllables) - 15.59
    ari = (4.71 * (char_count / word_count)) + (0.5 * avg_sentence_length) - 21.43
    letters_per_100 = (char_count / word_count) * 100
    sentences_per_100 = (sentence_count / word_count) * 100
    cli = (0.0588 * letters_per_100) - (0.296 * sentences_per_100) - 15.8
    fog = 0.4 * (avg_sentence_length + 100 * (len(complex_words) / word_count))
    smog = (1.043 * math.sqrt(len(complex_words) * (30 / sentence_count)) + 3.1291) if sentence_count > 0 else None
    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 3),
        "avg_syllables_per_word": round(avg_syllables, 3),
        "flesch_reading_ease": round(fre, 3),
        "flesch_kincaid_grade": round(fk, 3),
        "automated_readability_index": round(ari, 3),
        "coleman_liau_index": round(cli, 3),
        "gunning_fog": round(fog, 3),
        "smog_index": round(smog, 3) if smog is not None else None,
        "complex_word_count": len(complex_words),
        "readability_reliability": "low_short_text" if word_count < 10 else "standard",
    }


def optional_external_metrics(text: str) -> dict[str, Any]:
    status = optional_wheel_status()
    metrics: dict[str, Any] = {"module_status": status}
    if status.get("textstat") == "available":
        try:
            import textstat  # type: ignore

            metrics["textstat"] = {
                "flesch_reading_ease": textstat.flesch_reading_ease(text),
                "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
                "gunning_fog": textstat.gunning_fog(text),
                "smog_index": textstat.smog_index(text),
                "automated_readability_index": textstat.automated_readability_index(text),
                "coleman_liau_index": textstat.coleman_liau_index(text),
                "difficult_words": textstat.difficult_words(text),
            }
        except Exception as exc:  # pragma: no cover - depends on optional package state.
            metrics["textstat_error"] = type(exc).__name__
    if status.get("vaderSentiment") == "available":
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

            metrics["vader"] = SentimentIntensityAnalyzer().polarity_scores(text)
        except Exception as exc:  # pragma: no cover
            metrics["vader_error"] = type(exc).__name__
    if status.get("wordfreq") == "available":
        try:
            from wordfreq import zipf_frequency  # type: ignore

            values = []
            zh_values = []
            for token in tokenize(text):
                if re.match(r"[A-Za-z]", token):
                    values.append(zipf_frequency(token, "en"))
                elif is_chinese_token(token) and token_is_contentful(token):
                    zh_values.append(zipf_frequency(token, "zh"))
            metrics["wordfreq"] = {
                "avg_zipf": round(sum(values) / len(values), 3) if values else None,
                "min_zipf": round(min(values), 3) if values else None,
                "zh_avg_zipf": round(sum(zh_values) / len(zh_values), 3) if zh_values else None,
                "zh_min_zipf": round(min(zh_values), 3) if zh_values else None,
            }
        except Exception as exc:  # pragma: no cover
            metrics["wordfreq_error"] = type(exc).__name__
    if status.get("textblob") == "available":
        try:
            from textblob import TextBlob  # type: ignore

            sentiment = TextBlob(text).sentiment
            metrics["textblob"] = {"polarity": sentiment.polarity, "subjectivity": sentiment.subjectivity}
        except Exception as exc:  # pragma: no cover
            metrics["textblob_error"] = type(exc).__name__
    if status.get("empath") == "available":
        try:
            from empath import Empath  # type: ignore

            categories = Empath().analyze(text, normalize=True)
            top_categories = sorted(((k, v) for k, v in categories.items() if v), key=lambda item: item[1], reverse=True)[:8]
            metrics["empath"] = {"top_categories": [{"category": k, "score": round(v, 4)} for k, v in top_categories]}
        except Exception as exc:  # pragma: no cover
            metrics["empath_error"] = type(exc).__name__
    if status.get("yake") == "available":
        try:
            import yake  # type: ignore

            keyword_text = " ".join(tokenize(text)) if has_chinese(text) else text
            extractor = yake.KeywordExtractor(lan="zh" if has_chinese(text) else "en", n=2, top=8)
            metrics["yake"] = {
                "keywords": [{"term": term, "score": round(float(score), 6)} for term, score in extractor.extract_keywords(keyword_text)]
            }
        except Exception as exc:  # pragma: no cover
            metrics["yake_error"] = type(exc).__name__
    if status.get("spacy") == "available":
        try:
            import spacy  # type: ignore

            nlp = None
            for model_name in ("en_core_web_sm", "zh_core_web_sm"):
                try:
                    nlp = spacy.load(model_name)
                    break
                except Exception:
                    continue
            if nlp is None:
                metrics["spacy"] = {"status": "available_no_language_model"}
            else:
                doc = nlp(text)
                metrics["spacy"] = {
                    "entity_count": len(doc.ents),
                    "entities": [{"text": ent.text, "label": ent.label_} for ent in doc.ents[:12]],
                    "noun_chunk_count": len(list(doc.noun_chunks)) if doc.has_annotation("DEP") else None,
                    "sentence_count": len(list(doc.sents)) if doc.has_annotation("SENT_START") else None,
                }
        except Exception as exc:  # pragma: no cover
            metrics["spacy_error"] = type(exc).__name__
    if status.get("nltk") == "available":
        try:
            from nltk import word_tokenize  # type: ignore

            metrics["nltk"] = {"token_count": len(word_tokenize(text))}
        except Exception as exc:  # pragma: no cover
            metrics["nltk_error"] = type(exc).__name__
    if status.get("lexical_diversity") == "available":
        try:
            from lexical_diversity import lex_div as ld  # type: ignore

            tokenized = [normalize_token(t) for t in tokenize(text)]
            metrics["lexical_diversity"] = {
                "ttr": round(ld.ttr(tokenized), 4) if tokenized else None,
                "mtld": round(ld.mtld(tokenized), 4) if tokenized else None,
            }
        except Exception as exc:  # pragma: no cover
            metrics["lexical_diversity_error"] = type(exc).__name__
    if status.get("keybert") == "available":
        metrics["keybert"] = {
            "status": "available_not_run",
            "reason": "requires explicit embedding model selection to avoid implicit model download",
        }
    if status.get("sentence_transformers") == "available":
        metrics["sentence_transformers"] = {
            "status": "available_not_run",
            "reason": "embedding model loading is intentionally not automatic in this dry-run sidecar",
        }
    if status.get("readability") == "available":
        metrics["readability"] = {"status": "available_not_run", "reason": "package API varies; local formulas and textstat adapter are primary"}
    return metrics


def external_corpus_features(text_by_id: dict[str, str]) -> dict[str, dict[str, Any]]:
    status = optional_wheel_status()
    if status.get("sklearn") != "available" or not text_by_id:
        return {}
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    except Exception:  # pragma: no cover
        return {}
    ids = list(text_by_id.keys())
    texts = [text_by_id[row_id] for row_id in ids]
    try:
        def analyzer(text: str) -> list[str]:
            tokens = [normalize_token(token) for token in tokenize(text) if token_is_contentful(token)]
            bigrams = [f"{tokens[idx]} {tokens[idx + 1]}" for idx in range(len(tokens) - 1)]
            return tokens + bigrams

        vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer=analyzer,
            max_features=2000,
            min_df=1,
        )
        matrix = vectorizer.fit_transform(texts)
        names = vectorizer.get_feature_names_out()
    except ValueError:
        return {}
    features: dict[str, dict[str, Any]] = {}
    for idx, row_id in enumerate(ids):
        row = matrix.getrow(idx)
        if row.nnz == 0:
            features[row_id] = {"sklearn_tfidf": {"top_terms": [], "max_tfidf": 0.0, "sum_tfidf": 0.0}}
            continue
        pairs = sorted(zip(row.indices, row.data), key=lambda item: item[1], reverse=True)[:8]
        features[row_id] = {
            "sklearn_tfidf": {
                "top_terms": [{"term": str(names[col]), "score": round(float(score), 4)} for col, score in pairs],
                "max_tfidf": round(float(max(row.data)), 4),
                "sum_tfidf": round(float(sum(row.data)), 4),
            }
        }
    return features


def complexity_proxy(text: str) -> dict[str, Any]:
    tokens = [normalize_token(t) for t in tokenize(text)]
    words = [t for t in tokens if t]
    long_words = [w for w in words if len(w) >= 9]
    punctuation = sum(text.count(ch) for ch in [",", ";", ":", "\uFF0C", "\uFF1B", "\uFF1A"])
    readability = readability_metrics(text)
    syntax = sentence_complexity_metrics(text)
    fk_grade = readability.get("flesch_kincaid_grade")
    readability_score = (
        max(0, min(10, int(float(fk_grade) // 2)))
        if fk_grade is not None and readability.get("readability_reliability") == "standard"
        else 0
    )
    score = min(
        10,
        max(
            readability_score,
            int(syntax.get("syntactic_complexity_score", 0)),
            punctuation + len(long_words) + (1 if len(words) > 25 else 0) + (2 if len(words) > 45 else 0),
        ),
    )
    return {
        "complexity_score": score,
        "token_count": len(words),
        "long_word_count": len(long_words),
        "punctuation_count": punctuation,
        "readability_metrics": readability,
        "sentence_complexity_metrics": syntax,
        "external_metrics": optional_external_metrics(text),
    }


def salience_for_text(text: str, static_terms: list[LexiconTerm], dynamic_terms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    hits = match_terms(text, static_terms, dynamic_terms)
    group_scores: defaultdict[str, float] = defaultdict(float)
    for hit in hits:
        group_scores[str(hit["group"])] += float(hit["base_weight"])
    value_groups = {k: v for k, v in group_scores.items() if k != "low_value_terms" and v > 0}
    low_value_score = abs(group_scores.get("low_value_terms", 0.0))
    comp = complexity_proxy(text)
    keyphrases = local_keyphrases(text)
    keyphrase_score = min(3, len(keyphrases))
    positive_salience_score = sum(value_groups.values())
    modeling_value_score = positive_salience_score + keyphrase_score
    risk_score = group_scores.get("risk_terms", 0.0)
    constraint_score = group_scores.get("constraint_terms", 0.0)
    evidence_directness_score = group_scores.get("evidence_directness_terms", 0.0)
    attribution_signal_score = group_scores.get("relation_terms", 0.0) + group_scores.get("dynamic_entity_terms", 0.0)
    legacy_net_value_score = modeling_value_score - low_value_score
    return {
        "matched_terms": hits[:40],
        "group_scores": dict(sorted(group_scores.items())),
        "salience_dimensions": {
            "modeling_value_score": round(modeling_value_score, 3),
            "positive_salience_score": round(positive_salience_score, 3),
            "keyphrase_salience_score": round(keyphrase_score, 3),
            "risk_score": round(risk_score, 3),
            "constraint_score": round(constraint_score, 3),
            "evidence_directness_score": round(evidence_directness_score, 3),
            "attribution_signal_score": round(attribution_signal_score, 3),
            "processing_complexity_score": comp["complexity_score"],
            "low_value_score": round(low_value_score, 3),
        },
        "salience_scores": {
            "value_score": round(legacy_net_value_score, 3),
            "risk_score": round(risk_score, 3),
            "complexity_score": comp["complexity_score"],
            "low_value_score": round(low_value_score, 3),
        },
        "feature_summary": {
            "top_keyphrases": keyphrases,
            "language_hints": {
                "contains_chinese": has_chinese(text),
                "tokenizer": "jieba" if has_chinese(text) and importlib.util.find_spec("jieba") is not None else "regex",
                "normalizer": "opencc_t2s" if has_chinese(text) and importlib.util.find_spec("opencc") is not None else "none",
            },
            "token_count": comp["token_count"],
            "long_word_count": comp["long_word_count"],
            "punctuation_count": comp["punctuation_count"],
            "readability_metrics": comp["readability_metrics"],
            "sentence_complexity_metrics": comp["sentence_complexity_metrics"],
            "external_metrics": comp["external_metrics"],
        },
    }


def render_report(workspace: Path, rows: list[dict[str, Any]]) -> str:
    total = len(rows)
    value_sorted = sorted(rows, key=lambda r: (r.get("salience_dimensions") or {}).get("modeling_value_score", r["salience_scores"]["value_score"]), reverse=True)
    low_sorted = sorted(rows, key=lambda r: (r.get("salience_dimensions") or {}).get("low_value_score", r["salience_scores"]["low_value_score"]), reverse=True)
    group_counter: Counter[str] = Counter()
    route_counter: Counter[str] = Counter()
    external_feature_counter: Counter[str] = Counter()
    module_status = rows[0].get("feature_summary", {}).get("external_metrics", {}).get("module_status", {}) if rows else {}
    for row in rows:
        group_counter.update(row.get("group_scores", {}).keys())
        external_feature_counter.update((row.get("feature_summary", {}).get("external_corpus_metrics") or {}).keys())
        for key, value in (row.get("feature_summary", {}).get("external_metrics") or {}).items():
            if key == "module_status":
                continue
            if isinstance(value, dict) and value.get("status") == "available_not_run":
                continue
            external_feature_counter[key] += 1
        for target, route in row.get("v0_1_routes", {}).items():
            route_counter[f"{target}:{route}"] += 1
    lines = [
        "# Salience Sidecar Audit",
        "",
        f"Workspace: `{workspace.name}`",
        "",
        f"Rows scored: {total}",
        "",
        "## Matched Group Counts",
        "",
    ]
    for group, count in group_counter.most_common():
        lines.append(f"- `{group}`: {count}")
    lines.extend(["", "## Existing Route Counts", ""])
    for route, count in route_counter.most_common(20):
        lines.append(f"- `{route}`: {count}")
    lines.extend(["", "## External Wheel Status", ""])
    for name, status in sorted(module_status.items()):
        lines.append(f"- `{name}`: {status}")
    lines.extend(["", "## External Feature Coverage", ""])
    if external_feature_counter:
        for name, count in external_feature_counter.most_common():
            lines.append(f"- `{name}`: {count}")
    else:
        lines.append("- none")
    lines.extend(["", "## Highest Value Samples", ""])
    for row in value_sorted[:12]:
        text = row["text_preview"].replace("\n", " ")
        dimensions = row.get("salience_dimensions") or {}
        lines.append(
            f"- value={dimensions.get('modeling_value_score', row['salience_scores']['value_score'])} "
            f"risk={dimensions.get('risk_score', row['salience_scores']['risk_score'])} "
            f"constraint={dimensions.get('constraint_score', 0)} "
            f"low={dimensions.get('low_value_score', row['salience_scores']['low_value_score'])} routes={row.get('v0_1_routes', {})} :: {text}"
        )
    lines.extend(["", "## Highest Low-Value Samples", ""])
    for row in [item for item in low_sorted if item["salience_scores"]["low_value_score"] > 0][:12]:
        text = row["text_preview"].replace("\n", " ")
        dimensions = row.get("salience_dimensions") or {}
        lines.append(
            f"- value={dimensions.get('modeling_value_score', row['salience_scores']['value_score'])} "
            f"low={dimensions.get('low_value_score', row['salience_scores']['low_value_score'])} "
            f"routes={row.get('v0_1_routes', {})} :: {text}"
        )
    lines.extend(["", "## Highest Complexity Samples", ""])
    for row in sorted(rows, key=lambda r: r["salience_scores"]["complexity_score"], reverse=True)[:12]:
        text = row["text_preview"].replace("\n", " ")
        readability = row.get("feature_summary", {}).get("readability_metrics", {})
        syntax = row.get("feature_summary", {}).get("sentence_complexity_metrics", {})
        lines.append(
            f"- complexity={row['salience_scores']['complexity_score']} "
            f"fk={readability.get('flesch_kincaid_grade')} fog={readability.get('gunning_fog')} "
            f"syntax={syntax.get('syntactic_complexity_score')} syntax_rel={syntax.get('syntactic_complexity_reliability')} "
            f"routes={row.get('v0_1_routes', {})} :: {text}"
        )
    lines.extend(["", "## Highest External TF-IDF Samples", ""])
    tfidf_sorted = sorted(
        rows,
        key=lambda r: r.get("feature_summary", {}).get("external_corpus_metrics", {}).get("sklearn_tfidf", {}).get("sum_tfidf", 0.0),
        reverse=True,
    )
    for row in [item for item in tfidf_sorted if item.get("feature_summary", {}).get("external_corpus_metrics", {}).get("sklearn_tfidf")][:12]:
        text = row["text_preview"].replace("\n", " ")
        tfidf = row.get("feature_summary", {}).get("external_corpus_metrics", {}).get("sklearn_tfidf", {})
        terms = ", ".join(term["term"] for term in tfidf.get("top_terms", [])[:5])
        lines.append(
            f"- tfidf_sum={tfidf.get('sum_tfidf')} tfidf_max={tfidf.get('max_tfidf')} terms=[{terms}] "
            f"routes={row.get('v0_1_routes', {})} :: {text}"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- Sidecar only.",
            "- No route decisions changed.",
            "- No durable memory writes.",
            "- No graph writes.",
            "- No provider calls.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, str]:
    project_root = Path(args.project_root).resolve()
    workspace = (project_root / args.workspace).resolve() if not Path(args.workspace).is_absolute() else Path(args.workspace).resolve()
    lexicon_path = (project_root / args.lexicon).resolve() if not Path(args.lexicon).is_absolute() else Path(args.lexicon).resolve()
    output_dir = (workspace / "routing" / args.output_name).resolve()
    rows = collect_input_rows(workspace)
    static_terms = load_static_terms(lexicon_path)
    dynamic_terms = collect_dynamic_terms(rows)
    v01_routes = route_decision_map(workspace)
    text_by_id: dict[str, str] = {}
    for row in rows:
        text = safe_text(row)
        row_id = row.get("text_unit_id") or row.get("evidence_ref") or row.get("raw_span_id")
        if text and row_id:
            text_by_id[str(row_id)] = text
    corpus_features = external_corpus_features(text_by_id)
    scored: list[dict[str, Any]] = []
    for row in rows:
        text = safe_text(row)
        row_id = row.get("text_unit_id") or row.get("evidence_ref") or row.get("raw_span_id")
        if not text or not row_id:
            continue
        salience = salience_for_text(text, static_terms, dynamic_terms)
        row_corpus_features = corpus_features.get(str(row_id), {})
        if row_corpus_features:
            salience["feature_summary"]["external_corpus_metrics"] = row_corpus_features
            salience["salience_scores"]["external_tfidf_score"] = row_corpus_features.get("sklearn_tfidf", {}).get("sum_tfidf", 0.0)
        scored.append(
            {
                "schema_version": "routing.salience_sidecar.v0.2",
                "workspace_id": workspace.name,
                "row_id": row_id,
                "unit_type": row.get("unit_type"),
                "text_preview": text[:240],
                "salience_scores": salience["salience_scores"],
                "salience_dimensions": salience["salience_dimensions"],
                "feature_summary": salience["feature_summary"],
                "group_scores": salience["group_scores"],
                "matched_terms": salience["matched_terms"],
                "v0_1_routes": v01_routes.get(str(row_id), {}),
                "write_permission": False,
            }
        )
    output_jsonl = output_dir / "salience_sidecar.jsonl"
    output_report = output_dir / "salience_sidecar_report.md"
    write_jsonl(output_jsonl, scored)
    write_text(output_report, render_report(workspace, scored))
    return {"salience_sidecar": str(output_jsonl), "report": str(output_report)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--lexicon", default=DEFAULT_LEXICON)
    parser.add_argument("--output-name", default="salience_sidecar_v0_2")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
