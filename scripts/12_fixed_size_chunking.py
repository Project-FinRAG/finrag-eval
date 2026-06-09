"""Regenerate fixed-size chunks over the full labeled corpus.

The original `fixed_size` chunks were a 12-filing baseline. To answer the
chunking axis fairly, we re-window the section-aware ("labeled") chunk text into
fixed-size windows over the SAME filings: identical source text (so gold quotes
match the same way across both arms), with the windowing as the only variable.

Output: data/processed/chunks/fixed_size_full.jsonl, tagged
chunking_method="fixed_size_full" (kept distinct from the old baseline).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import tiktoken

from finrag_eval.common import Chunk
from finrag_eval.retrieval.bm25 import load_chunks_from_jsonl

CHUNK_DIR = Path("data/processed/chunks")
OUTPUT_PATH = CHUNK_DIR / "fixed_size_full.jsonl"
CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 64
ENCODING = "cl100k_base"


def main() -> None:
    chunks = load_chunks_from_jsonl("labeled")
    print(f"Loaded {len(chunks):,} labeled chunks")

    by_filing: dict[tuple[str, str, str], list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_filing[(c.ticker, c.filing_accession, str(c.filing_type))].append(c)

    enc = tiktoken.get_encoding(ENCODING)
    step = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS

    n_filings = 0
    n_out = 0
    with OUTPUT_PATH.open("w", encoding="utf-8") as out:
        for (ticker, accession, filing_type), filing_chunks in sorted(by_filing.items()):
            filing_chunks.sort(key=lambda c: c.chunk_id)
            text = "\n".join(c.text for c in filing_chunks)
            tokens = enc.encode(text)

            idx = 0
            for start in range(0, len(tokens), step):
                window = tokens[start : start + CHUNK_SIZE_TOKENS]
                if not window:
                    break
                record = {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "accession": accession,
                    "chunk_id": f"{ticker}_{accession}_fixed_{idx:04d}",
                    "text": enc.decode(window),
                    "section_label": None,
                    "chunking_method": "fixed_size_full",
                    "token_count": len(window),
                }
                out.write(json.dumps(record) + "\n")
                idx += 1
                n_out += 1
                if start + CHUNK_SIZE_TOKENS >= len(tokens):
                    break
            n_filings += 1

    print(f"Wrote {n_out:,} fixed-size chunks across {n_filings} filings -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()