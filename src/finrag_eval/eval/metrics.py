"""Retrieval metrics — Recall@K, Precision@K, MRR, nDCG, evidence-hit.

Owner: Evaluation Lead

All metrics take:
    - retrieved: list of retrieved chunk_ids in rank order
    - gold: set of gold-evidence chunk_ids

And return a float.

These are standard IR metrics; we implement them directly rather than depend
on a library so they're transparent and reproducible.
"""

from __future__ import annotations

import math


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
    dcg = sum(
        (1.0 / math.log2(i + 2)) if r in gold else 0.0
        for i, r in enumerate(retrieved[:k])
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def evidence_hit_rate(retrieved: list[str], gold: set[str], k: int) -> float:
    """1.0 if ANY gold passage is in top-k, else 0.0. Useful per-question."""
    return 1.0 if any(r in gold for r in retrieved[:k]) else 0.0
