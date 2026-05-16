"""Retriever Protocol — the contract every retriever must satisfy."""

from __future__ import annotations

from typing import Protocol

from finrag_eval.common import Chunk, RetrievalResult


class Retriever(Protocol):
    """Common interface for all retrieval strategies.

    Lifecycle:
        1. retriever = SomeRetriever(...)
        2. retriever.index(chunks)   # one-time, persists to disk
        3. retriever.retrieve(q, k)  # many times, fast
    """

    name: str
    """Human-readable name used in eval reports (e.g. 'bm25', 'dense', 'hybrid+rerank')."""

    def index(self, chunks: list[Chunk]) -> None:
        """Build the index from a list of chunks. Persists to disk."""
        ...

    def load(self) -> None:
        """Load a previously-built index from disk."""
        ...

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return the top-k most relevant chunks for a query, ranked by score."""
        ...
