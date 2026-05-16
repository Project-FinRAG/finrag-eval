"""Dense retriever using sentence-transformer embeddings + Chroma.

Owner: Retrieval & Modeling Lead

Uses an embedding model (OpenAI text-embedding-3-small by default, or
sentence-transformers for offline) and ChromaDB for the vector index.
"""

from __future__ import annotations

from finrag_eval.common import Chunk, RetrievalResult


class DenseRetriever:
    name = "dense"

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        collection_name: str = "finrag_dense",
    ) -> None:
        self.embedding_model = embedding_model
        self.collection_name = collection_name
        # TODO(@retrieval-lead): init chromadb client

    def index(self, chunks: list[Chunk]) -> None:
        # TODO(@retrieval-lead): batch-embed chunks
        # TODO(@retrieval-lead): upsert to Chroma collection
        raise NotImplementedError("DenseRetriever.index is not yet implemented")

    def load(self) -> None:
        # TODO(@retrieval-lead): connect to existing collection
        raise NotImplementedError("DenseRetriever.load is not yet implemented")

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        # TODO(@retrieval-lead): embed query, query Chroma
        raise NotImplementedError("DenseRetriever.retrieve is not yet implemented")
