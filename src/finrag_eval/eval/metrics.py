"""Retrieval metrics — Recall@K, Precision@K, MRR, nDCG, evidence-hit, soft-recall.

Owner: Evaluation Lead

The id-based metrics take:
    - retrieved: list of retrieved chunk_ids in rank order
    - gold: set of gold-evidence chunk_ids
and return a float.

soft_recall_at_k is the exception: it needs the retrieved chunks' text and the
gold quotes, so it takes the richer RetrievalResult / Citation objects.

These are standard IR metrics; we implement them directly rather than depend
on a library so they're transparent and reproducible.
"""

from __future__ import annotations

import math

from finrag_eval.common import Citation, RetrievalResult
from finrag_eval.common.text import normalize_for_search


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of gold passages found in top-k retrieved."""
    if not gold:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & gold) / len(gold)


def precision_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Fraction of top-k retrieved that are gold."""
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return sum(1 for r in top_k if r in gold) / len(top_k)


def mean_reciprocal_rank(retrieved: list[str], gold: set[str]) -> float:
    """Reciprocal rank of the first gold passage; 0 if none retrieved."""
    for i, r in enumerate(retrieved, start=1):
        if r in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k. Binary relevance."""
    dcg = sum((1.0 / math.log2(i + 2)) if r in gold else 0.0 for i, r in enumerate(retrieved[:k]))
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evidence_hit_rate(retrieved: list[str], gold: set[str], k: int) -> float:
    """1.0 if ANY gold passage is in top-k, else 0.0. Useful per-question."""
    return 1.0 if any(r in gold for r in retrieved[:k]) else 0.0


def soft_recall_at_k(
    retrieved: list[RetrievalResult],
    gold: list[Citation],
    k: int,
) -> float:
    """Quote-containment recall: a superset of id-based recall@k.

    A gold citation counts as found if its chunk_id is in the top-k retrieved
    chunks, OR its quote (normalized) is contained in any top-k chunk's text.
    Citations without a quote fall back to the id check, so this is always
    >= recall_at_k on the same retrieval (assuming distinct gold chunk_ids).
    """
    if not gold:
        return 0.0
    top_k = retrieved[:k]
    top_k_ids = {r.chunk.chunk_id for r in top_k}
    top_k_texts = [normalize_for_search(r.chunk.text) for r in top_k]

    found = 0
    for citation in gold:
        if citation.chunk_id in top_k_ids:
            found += 1
            continue
        if citation.quote:
            needle = normalize_for_search(citation.quote)
            if needle and any(needle in text for text in top_k_texts):
                found += 1
    return found / len(gold)
