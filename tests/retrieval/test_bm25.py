"""Smoke tests for BM25 retriever — requires local corpus data."""

from __future__ import annotations

from pathlib import Path

import pytest

from finrag_eval.common import RetrievalResult
from finrag_eval.retrieval.bm25 import BM25Retriever, load_chunks_from_jsonl

QUERIES = [
    "what are the main risk factors for Apple?",
    "revenue growth in MD&A section",
    "cybersecurity risks and data breaches",
]

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def bm25_labeled():
    chunks = load_chunks_from_jsonl("labeled")
    r = BM25Retriever(index_path=Path("data/indexes/bm25_labeled"))
    r.index(chunks)
    return r


@pytest.fixture(scope="module")
def bm25_fixed():
    chunks = load_chunks_from_jsonl("fixed_size")
    r = BM25Retriever(index_path=Path("data/indexes/bm25_fixed_size"))
    r.index(chunks)
    return r


@pytest.mark.parametrize("query", QUERIES)
def test_labeled_returns_top5(bm25_labeled, query):
    results = bm25_labeled.retrieve(query, k=5)
    assert len(results) == 5
    for res in results:
        assert isinstance(res, RetrievalResult)
        assert res.score >= 0.0
        assert res.rank >= 1


@pytest.mark.parametrize("query", QUERIES)
def test_fixed_returns_top5(bm25_fixed, query):
    results = bm25_fixed.retrieve(query, k=5)
    assert len(results) == 5


def test_ranks_are_sequential(bm25_labeled):
    results = bm25_labeled.retrieve(QUERIES[0], k=5)
    assert [r.rank for r in results] == [1, 2, 3, 4, 5]


def test_scores_are_descending(bm25_labeled):
    results = bm25_labeled.retrieve(QUERIES[0], k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_latency_logged(bm25_labeled):
    bm25_labeled.retrieve(QUERIES[0], k=5)
    assert bm25_labeled.latency_ms() > 0


def test_chunk_id_format(bm25_labeled):
    results = bm25_labeled.retrieve(QUERIES[0], k=1)
    parts = results[0].chunk.chunk_id.split("_")
    assert len(parts) >= 3
