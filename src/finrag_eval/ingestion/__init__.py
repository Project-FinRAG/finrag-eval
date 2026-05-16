"""Ingestion package — fetch and parse SEC EDGAR filings.

Owner: Data & Application Lead

This package is responsible for everything from "give me filings" to "give me
clean, chunked, indexable text with metadata."

Public interface:
    - EdgarClient: fetches filings from SEC EDGAR with rate limiting
    - Chunker (Protocol): chunking strategy interface
    - FixedSizeChunker: fixed-token sliding window
    - SectionAwareChunker: respects 10-K/10-Q section boundaries
"""

from finrag_eval.ingestion.chunker import (
    Chunker,
    FixedSizeChunker,
    SectionAwareChunker,
)
from finrag_eval.ingestion.edgar_client import EdgarClient

__all__ = [
    "Chunker",
    "EdgarClient",
    "FixedSizeChunker",
    "SectionAwareChunker",
]
