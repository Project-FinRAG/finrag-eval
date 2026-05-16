"""Cross-encoder reranker wrapped around any base retriever.

Owner: Retrieval & Modeling Lead

Pattern: retrieve k_initial from a base retriever, then rerank with a
cross-encoder model that scores (query, passage) jointly. More accurate
but slower than bi-encoder retrieval.

Default model: cross-encoder/ms-marco-MiniLM-L-12-v2
"""

from __future__ import annotations

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

    def index(self, chunks) -> None:  # type: ignore[no-untyped-def]
        self.base.index(chunks)

    def load(self) -> None:
        self.base.load()

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        # TODO(@retrieval-lead): get initial_k from base, score with cross-encoder,
        #                        return top-k by reranker score
        raise NotImplementedError("RerankedRetriever.retrieve is not yet implemented")
