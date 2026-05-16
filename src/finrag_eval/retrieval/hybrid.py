"""Hybrid retriever combining BM25 and dense via reciprocal rank fusion.

Owner: Retrieval & Modeling Lead

RRF formula: score(d) = sum_i 1 / (k + rank_i(d))
where k is typically 60 and rank_i is the document's rank in retriever i.

Reference: Cormack et al. 2009, "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods"
"""

from __future__ import annotations

from finrag_eval.common import Chunk, RetrievalResult
from finrag_eval.retrieval.bm25 import BM25Retriever
from finrag_eval.retrieval.dense import DenseRetriever


class HybridRetriever:
    name = "hybrid"

    def __init__(
        self,
        bm25: BM25Retriever | None = None,
        dense: DenseRetriever | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.bm25 = bm25 or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.rrf_k = rrf_k

    def index(self, chunks: list[Chunk]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def load(self) -> None:
        self.bm25.load()
        self.dense.load()

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        # TODO(@retrieval-lead): retrieve k*3 from each, fuse via RRF, return top k
        raise NotImplementedError("HybridRetriever.retrieve is not yet implemented")
