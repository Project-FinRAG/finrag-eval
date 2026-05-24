"""Build BM25 indexes for both chunking strategies.

Usage:
    python scripts/07_build_indexes.py
"""
from pathlib import Path
from finrag_eval.retrieval.bm25 import BM25Retriever, load_chunks_from_jsonl


def build(strategy: str, index_path: Path) -> None:
    print(f"\nBuilding BM25 index: strategy='{strategy}'")
    chunks = load_chunks_from_jsonl(strategy)  # type: ignore[arg-type]
    r = BM25Retriever(index_path=index_path)
    r.index(chunks)


if __name__ == "__main__":
    build("labeled",    Path("data/indexes/bm25_labeled"))
    build("fixed_size", Path("data/indexes/bm25_fixed_size"))
    print("\nAll indexes built.")
