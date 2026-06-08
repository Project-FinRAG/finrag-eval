"""Cross-encoder reranker wrapped around any base retriever.

Owner: Retrieval & Modeling Lead

Pattern: retrieve k_initial from a base retriever, then rerank with a
cross-encoder model that scores (query, passage) jointly. More accurate
but slower than bi-encoder retrieval.

Default model: cross-encoder/ms-marco-MiniLM-L-12-v2
"""

from __future__ import annotations

import time

from sentence_transformers import CrossEncoder

from finrag_eval.common import RetrievalResult
from finrag_eval.retrieval.base import Retriever


class RerankedRetriever:
    """Wraps any Retriever, retrieves an initial pool, reranks with cross-encoder."""

    def __init__(
        self,
        base_retriever: Retriever,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2",
        initial_k: int = 50,
    ) -> None:
        self.base = base_retriever
        self.reranker_model = reranker_model
        self.initial_k = initial_k
        self.name = f"{base_retriever.name}+rerank"
        self._model: CrossEncoder | None = None
        self._last_latency_ms: float = 0.0

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.reranker_model)
        return self._model

    def index(self, chunks) -> None:  # type: ignore[no-untyped-def]
        self.base.index(chunks)

    def load(self) -> None:
        self.base.load()

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        t0 = time.perf_counter()

        # Step 1: pull a wider candidate pool from the base retriever
        candidates = self.base.retrieve(query, self.initial_k)

        if not candidates:
            return []

        # Step 2: score each (query, passage) pair with the cross-encoder
        model = self._get_model()
        pairs = [(query, r.chunk.text) for r in candidates]
        scores = model.predict(pairs)

        # Step 3: sort by reranker score descending, keep top-k
        ranked = sorted(zip(scores, candidates, strict=True), key=lambda x: x[0], reverse=True)
        results = [
            RetrievalResult(chunk=item.chunk, score=float(score), rank=rank + 1)
            for rank, (score, item) in enumerate(ranked[:k])
        ]

        self._last_latency_ms = (time.perf_counter() - t0) * 1000
        return results

    def latency_ms(self) -> float:
        return self._last_latency_ms
        
