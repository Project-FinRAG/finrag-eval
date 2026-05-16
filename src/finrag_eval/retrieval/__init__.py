"""Retrieval package — find the most relevant chunks for a query.

Owner: Retrieval & Modeling Lead

Four retrievers, all implementing the Retriever Protocol:
    - BM25Retriever: sparse lexical (rank-bm25)
    - DenseRetriever: dense embeddings + cosine (sentence-transformers + Chroma)
    - HybridRetriever: BM25 + dense fused via reciprocal rank fusion
    - RerankedRetriever: any of the above + cross-encoder reranking

The Retriever Protocol is the most important interface in the codebase.
Every retriever must support .index() once and .retrieve() many times.
"""

from finrag_eval.retrieval.base import Retriever
from finrag_eval.retrieval.bm25 import BM25Retriever
from finrag_eval.retrieval.dense import DenseRetriever
from finrag_eval.retrieval.hybrid import HybridRetriever
from finrag_eval.retrieval.reranker import RerankedRetriever

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "RerankedRetriever",
    "Retriever",
]
