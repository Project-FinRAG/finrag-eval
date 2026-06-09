"""Build the dense Chroma index for the fixed_size_full chunking arm.

Mirrors the labeled dense index but over the regenerated full-corpus fixed-size
chunks, into a separate path/collection so the labeled index is untouched.
Safe to re-run: DenseRetriever.index() recreates the collection from scratch.

Cost: ~58,992 chunks via text-embedding-3-small (~$0.65, ~5 min).

Usage:
    uv run python scripts/13_build_dense_fixed.py
"""

from __future__ import annotations

from pathlib import Path

from finrag_eval.retrieval.bm25 import load_chunks_from_jsonl
from finrag_eval.retrieval.dense import DenseRetriever

INDEX_PATH = Path("data/indexes/chroma_dense_fixed_full")
COLLECTION = "finrag_dense_fixed"


def main() -> None:
    chunks = load_chunks_from_jsonl("fixed_size_full")
    print(f"Loaded {len(chunks):,} fixed_size_full chunks")

    retriever = DenseRetriever(
        index_path=INDEX_PATH,
        collection_name=COLLECTION,
    )
    retriever.index(chunks)


if __name__ == "__main__":
    main()