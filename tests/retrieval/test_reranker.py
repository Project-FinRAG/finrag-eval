"""Unit tests for RerankedRetriever and HybridRetriever — no model downloads, no API calls."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from finrag_eval.common import Chunk, RetrievalResult
from finrag_eval.retrieval.hybrid import HybridRetriever
from finrag_eval.retrieval.reranker import RerankedRetriever

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(chunk_id: str, text: str = "some text") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        filing_accession="0001234567-24-000001",
        ticker="AAPL",
        filing_type="10-K",
        section="Item 1",
        text=text,
        char_start=0,
        char_end=len(text),
        token_count=len(text.split()),
    )


def _make_result(
    chunk_id: str, score: float, rank: int, text: str = "some text"
) -> RetrievalResult:
    return RetrievalResult(chunk=_make_chunk(chunk_id, text), score=score, rank=rank)


def _stub_retriever(name: str, results: list[RetrievalResult]) -> MagicMock:
    r = MagicMock()
    r.name = name
    r.retrieve.return_value = results
    return r


# ---------------------------------------------------------------------------
# RerankedRetriever tests
# ---------------------------------------------------------------------------


class TestRerankedRetriever:
    def _make_reranker(
        self, base_results: list[RetrievalResult], scores: list[float], initial_k: int = 50
    ) -> RerankedRetriever:
        base = _stub_retriever("hybrid", base_results)
        reranker = RerankedRetriever(base_retriever=base, initial_k=initial_k)
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        reranker._model = mock_model
        return reranker

    def test_name_is_set(self) -> None:
        base = _stub_retriever("hybrid", [])
        r = RerankedRetriever(base_retriever=base)
        assert r.name == "hybrid+rerank"

    def test_empty_base_returns_empty(self) -> None:
        reranker = self._make_reranker([], [])
        assert reranker.retrieve("query", k=5) == []

    def test_reorders_by_reranker_score(self) -> None:
        # Base returns chunks in order c1, c2, c3
        # Reranker scores them so c3 > c1 > c2 — output should be c3, c1, c2
        base_results = [
            _make_result("c1", score=0.9, rank=1, text="text one"),
            _make_result("c2", score=0.8, rank=2, text="text two"),
            _make_result("c3", score=0.7, rank=3, text="text three"),
        ]
        reranker_scores = [0.5, 0.1, 0.9]  # c1=0.5, c2=0.1, c3=0.9
        reranker = self._make_reranker(base_results, reranker_scores)

        results = reranker.retrieve("query", k=3)

        assert [r.chunk.chunk_id for r in results] == ["c3", "c1", "c2"]

    def test_honors_k(self) -> None:
        base_results = [_make_result(f"c{i}", score=float(i), rank=i) for i in range(10)]
        scores = list(range(10, 0, -1))  # descending
        reranker = self._make_reranker(base_results, scores)

        results = reranker.retrieve("query", k=3)
        assert len(results) == 3

    def test_reassigns_rank(self) -> None:
        base_results = [
            _make_result("c1", score=0.9, rank=1),
            _make_result("c2", score=0.8, rank=2),
        ]
        reranker_scores = [0.2, 0.8]  # c2 should be rank 1 after reranking
        reranker = self._make_reranker(base_results, reranker_scores)

        results = reranker.retrieve("query", k=2)
        assert results[0].rank == 1
        assert results[1].rank == 2

    def test_reassigns_score_from_reranker(self) -> None:
        base_results = [_make_result("c1", score=0.9, rank=1)]
        reranker_scores = [0.42]
        reranker = self._make_reranker(base_results, reranker_scores)

        results = reranker.retrieve("query", k=1)
        assert results[0].score == pytest.approx(0.42)

    def test_pulls_initial_k_from_base(self) -> None:
        base = _stub_retriever("hybrid", [])
        reranker = RerankedRetriever(base_retriever=base, initial_k=25)
        reranker._model = MagicMock()
        reranker._model.predict.return_value = []

        reranker.retrieve("query", k=5)
        base.retrieve.assert_called_once_with("query", 25)


# ---------------------------------------------------------------------------
# HybridRetriever tests
# ---------------------------------------------------------------------------


class TestHybridRetriever:
    def _make_hybrid(
        self,
        bm25_results: list[RetrievalResult],
        dense_results: list[RetrievalResult],
    ) -> HybridRetriever:
        bm25 = _stub_retriever("bm25", bm25_results)
        dense = _stub_retriever("dense", dense_results)
        hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)
        return hybrid

    def test_name_is_hybrid(self) -> None:
        hybrid = self._make_hybrid([], [])
        assert hybrid.name == "hybrid"

    def test_rrf_fusion_order(self) -> None:
        # c1 is rank 1 in bm25, rank 2 in dense → high RRF score
        # c2 is rank 2 in bm25, rank 1 in dense → also high RRF score
        # c3 is rank 3 in bm25 only → lower score
        bm25_results = [
            _make_result("c1", score=3.0, rank=1),
            _make_result("c2", score=2.0, rank=2),
            _make_result("c3", score=1.0, rank=3),
        ]
        dense_results = [
            _make_result("c2", score=3.0, rank=1),
            _make_result("c1", score=2.0, rank=2),
        ]
        hybrid = self._make_hybrid(bm25_results, dense_results)

        results = hybrid.retrieve("query", k=3)
        chunk_ids = [r.chunk.chunk_id for r in results]

        # c1 and c2 both appear in both lists so they outscore c3
        assert "c3" not in chunk_ids[:2]
        assert set(chunk_ids[:2]) == {"c1", "c2"}

    def test_honors_k(self) -> None:
        bm25_results = [_make_result(f"c{i}", score=float(i), rank=i) for i in range(1, 6)]
        dense_results = [_make_result(f"c{i}", score=float(i), rank=i) for i in range(1, 6)]
        hybrid = self._make_hybrid(bm25_results, dense_results)

        results = hybrid.retrieve("query", k=3)
        assert len(results) == 3

    def test_ranks_start_at_one(self) -> None:
        bm25_results = [_make_result("c1", score=1.0, rank=1)]
        dense_results = [_make_result("c1", score=1.0, rank=1)]
        hybrid = self._make_hybrid(bm25_results, dense_results)

        results = hybrid.retrieve("query", k=1)
        assert results[0].rank == 1


# ---------------------------------------------------------------------------
# BM25Retriever unit tests (in-memory, no corpus file)
# ---------------------------------------------------------------------------


class TestBM25RetrieverUnit:
    def test_obvious_match_ranks_first(self) -> None:
        from finrag_eval.retrieval.bm25 import BM25Retriever

        chunks = [
            _make_chunk("c1", "apple revenue quarterly earnings"),
            _make_chunk("c2", "federal reserve interest rate decision"),
            _make_chunk("c3", "apple iphone sales growth market share"),
        ]
        r = BM25Retriever()
        r.index(chunks)

        results = r.retrieve("apple revenue", k=3)
        assert results[0].chunk.chunk_id in ("c1", "c3")

    def test_returns_k_results(self) -> None:
        from finrag_eval.retrieval.bm25 import BM25Retriever

        chunks = [_make_chunk(f"c{i}", f"text about topic {i}") for i in range(10)]
        r = BM25Retriever()
        r.index(chunks)

        results = r.retrieve("topic", k=4)
        assert len(results) == 4

    def test_ranks_are_sequential(self) -> None:
        from finrag_eval.retrieval.bm25 import BM25Retriever

        chunks = [_make_chunk(f"c{i}", f"document {i}") for i in range(5)]
        r = BM25Retriever()
        r.index(chunks)

        results = r.retrieve("document", k=5)
        assert [r.rank for r in results] == list(range(1, 6))
