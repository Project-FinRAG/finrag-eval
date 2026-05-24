"""Hybrid retriever combining BM25 and dense via reciprocal rank fusion.

Owner: Retrieval & Modeling Lead
"""

from __future__ import annotations

import time

from finrag_eval.common import Chunk, RetrievalResult
from finrag_eval.retrieval.bm25 import BM25Retriever
from finrag_eval.retrieval.dense import DenseRetriever


def _rrf(
    *rankings: list[RetrievalResult],
    k: int = 60,
) -> tuple[list[tuple[str, float]], dict[str, Chunk]]:
    scores: dict[str, float] = {}
    chunk_map: dict[str, Chunk] = {}
    for ranked_list in rankings:
        for result in ranked_list:
            cid = result.chunk.chunk_id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + result.rank)
            chunk_map[cid] = result.chunk
    return sorted(scores.items(), key=lambda x: x[1], reverse=True), chunk_map


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        rrf_k: int = 60,
        fetch_n: int = 50,
    ) -> None:
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.rrf_k = rrf_k
        self.fetch_n = fetch_n
        self._last_latency_ms: float = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def load(self) -> None:
        self.bm25.load()
        self.dense.load()

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        t0 = time.perf_counter()
        bm25_hits = self.bm25.retrieve(query, self.fetch_n)
        dense_hits = self.dense.retrieve(query, self.fetch_n)
        fused, chunk_map = _rrf(bm25_hits, dense_hits, k=self.rrf_k)
        results = [
            RetrievalResult(chunk=chunk_map[cid], score=score, rank=rank + 1)
            for rank, (cid, score) in enumerate(fused[:k])
        ]
        self._last_latency_ms = (time.perf_counter() - t0) * 1000
        return results

    def latency_ms(self) -> float:
        return self._last_latency_ms
