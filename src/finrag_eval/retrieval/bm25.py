"""BM25 sparse lexical retriever.

Owner: Retrieval & Modeling Lead

This is our sparse baseline. Uses `rank-bm25`. Cheap, fast, no GPU.
Often surprisingly competitive on factual lookup questions where exact
financial terminology matters.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Literal

from rank_bm25 import BM25Okapi

from finrag_eval.common import Chunk, RetrievalResult

CHUNK_DIR = Path("data/processed/chunks")
INDEX_DIR = Path("data/indexes")


def load_chunks_from_jsonl(
    strategy: Literal["labeled", "fixed_size", "strict", "all"] = "labeled",
) -> list[Chunk]:
    """Load chunks from JSONL files, filtered by chunking strategy.

    labeled    = section_aware + hybrid_section_aware (186 filings, primary)
    fixed_size = fixed_size only (12 filings, baseline)
    strict     = section_aware only (163 filings, robustness check)
    all        = entire corpus
    """
    allowed: set[str] | None
    if strategy == "labeled":
        allowed = {"section_aware", "hybrid_section_aware"}
    elif strategy == "fixed_size":
        allowed = {"fixed_size"}
    elif strategy == "strict":
        allowed = {"section_aware"}
    else:
        allowed = None

    chunks = []
    for path in sorted(CHUNK_DIR.glob("*.jsonl")):
        with path.open() as f:
            for line in f:
                raw = json.loads(line)
                if allowed and raw.get("chunking_method") not in allowed:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=raw["chunk_id"],
                        filing_accession=raw["accession"],
                        ticker=raw["ticker"],
                        filing_type=raw["filing_type"],
                        section=raw.get("section_label"),
                        text=raw["text"],
                        char_start=raw.get("char_start", 0),
                        char_end=raw.get("char_end", len(raw["text"])),
                        token_count=raw.get("token_count", 0),
                    )
                )
    return chunks


class BM25Retriever:
    name = "bm25"

    def __init__(self, index_path: Path | None = None) -> None:
        self.index_path = index_path or INDEX_DIR / "bm25"
        self._index: BM25Okapi | None = None
        self._chunks: list[Chunk] = []
        self._build_time_ms: float = 0.0
        self._last_latency_ms: float = 0.0

    def index(self, chunks: list[Chunk]) -> None:
        """Tokenize chunks, build BM25 index, persist to disk."""
        self._chunks = chunks
        tokenized = [c.text.lower().split() for c in chunks]

        t0 = time.perf_counter()
        self._index = BM25Okapi(tokenized)
        self._build_time_ms = (time.perf_counter() - t0) * 1000

        self.index_path.mkdir(parents=True, exist_ok=True)
        with open(self.index_path / "index.pkl", "wb") as f:
            pickle.dump(self._index, f)
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._chunks, f)
        with open(self.index_path / "stats.json", "w") as f:
            json.dump(
                {"num_chunks": len(chunks), "build_time_ms": round(self._build_time_ms, 1)},
                f,
                indent=2,
            )

        print(f"BM25 index built: {len(chunks):,} chunks in {self._build_time_ms:.0f}ms")
        print(f"Saved to {self.index_path}")

    def load(self) -> None:
        """Load a previously-built index from disk."""
        with open(self.index_path / "index.pkl", "rb") as f:
            self._index = pickle.load(f)
        with open(self.index_path / "chunks.pkl", "rb") as f:
            self._chunks = pickle.load(f)

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Return top-k most relevant chunks for the query."""
        if self._index is None:
            raise RuntimeError("Index not built or loaded. Call .index() or .load() first.")

        t0 = time.perf_counter()
        tokens = query.lower().split()
        scores = self._index.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        results = [
            RetrievalResult(chunk=self._chunks[i], score=float(scores[i]), rank=rank + 1)
            for rank, i in enumerate(top_indices)
        ]

        self._last_latency_ms = (time.perf_counter() - t0) * 1000
        return results

    def latency_ms(self) -> float:
        return self._last_latency_ms

    def index_stats(self) -> dict[str, object]:
        return {
            "num_chunks": len(self._chunks),
            "build_time_ms": round(self._build_time_ms, 1),
            "index_path": str(self.index_path),
        }
