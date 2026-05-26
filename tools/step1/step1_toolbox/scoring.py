"""Small deterministic scoring helpers for Step 1 retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
MIXED_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokens(text: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(text or "")}


def bm25_tokens(text: str) -> list[str]:
    """Tokenization policy aligned with the Step 1 index runner."""

    raw_tokens = MIXED_TOKEN_RE.findall(text or "")
    latin = [token.lower() for token in raw_tokens if not ("\u4e00" <= token <= "\u9fff")]
    cjk = [token for token in raw_tokens if "\u4e00" <= token <= "\u9fff"]
    cjk_bigrams = [cjk[index] + cjk[index + 1] for index in range(len(cjk) - 1)]
    return latin + cjk + cjk_bigrams


def lexical_score(query: str, text: str) -> float:
    query_tokens = tokens(query)
    if not query_tokens:
        return 0.0
    text_tokens = tokens(text)
    overlap = query_tokens & text_tokens
    if not overlap:
        return 0.0
    return float(len(overlap)) + (len(overlap) / max(len(query_tokens), 1))


def bm25_score(query: str, text: str, bm25_index: dict[str, Any], index_entry_id: str) -> float:
    query_terms = bm25_tokens(query)
    if not query_terms:
        return 0.0
    doc_terms = bm25_tokens(text)
    if not doc_terms:
        return 0.0
    term_counts = Counter(doc_terms)
    idf = bm25_index.get("idf", {})
    doc_lengths = bm25_index.get("document_lengths", {})
    doc_len = float(doc_lengths.get(index_entry_id) or len(doc_terms))
    avg_len = float(bm25_index.get("avg_document_length") or max(doc_len, 1.0))
    k1 = float(bm25_index.get("k1") or 1.5)
    b = float(bm25_index.get("b") or 0.75)
    score = 0.0
    for term in query_terms:
        tf = term_counts.get(term, 0)
        if tf <= 0:
            continue
        term_idf = float(idf.get(term, 0.0))
        denominator = tf + k1 * (1.0 - b + b * (doc_len / max(avg_len, 1e-9)))
        score += term_idf * ((tf * (k1 + 1.0)) / max(denominator, 1e-9))
    return float(score)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(dot / (left_norm * right_norm))
