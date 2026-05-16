"""BM25 sparse lexical retriever.

Owner: Retrieval & Modeling Lead

This is our sparse baseline. Uses `rank-bm25`. Cheap, fast, no GPU.
Often surprisingly competitive on factual lookup questions where exact
financial terminology matters.
"""

from __future__ import annotations

from pathlib import Path

from finrag_eval.common import Chunk, RetrievalResult


class BM25Retriever:
    name = "bm25"

    def __init__(self, index_path: Path | None = None) -> None:
        self.index_path = index_path
        # TODO(@retrieval-lead): hold rank_bm25.BM25Okapi instance + chunk lookup

    def index(self, chunks: list[Chunk]) -> None:
        # TODO(@retrieval-lead): tokenize each chunk.text, build BM25Okapi
        # TODO(@retrieval-lead): pickle to self.index_path with chunk metadata
        raise NotImplementedError("BM25Retriever.index is not yet implemented")

    def load(self) -> None:
        # TODO(@retrieval-lead): unpickle from self.index_path
        raise NotImplementedError("BM25Retriever.load is not yet implemented")

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        # TODO(@retrieval-lead): tokenize query, get top-k scores
        raise NotImplementedError("BM25Retriever.retrieve is not yet implemented")
