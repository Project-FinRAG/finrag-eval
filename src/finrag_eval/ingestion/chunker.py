"""Chunking strategies for SEC filings.

Owner: Data & Application Lead

We compare two strategies:
    1. FixedSizeChunker: fixed-token sliding window with overlap
    2. SectionAwareChunker: respects 10-K/10-Q section boundaries

Both implement the Chunker Protocol so they're swappable in the eval matrix.
"""

from __future__ import annotations

from typing import Protocol

from finrag_eval.common import Chunk, Filing


class Chunker(Protocol):
    """Protocol for chunking strategies. Any chunker must implement this."""

    def chunk(self, filing: Filing) -> list[Chunk]:
        """Split a filing into retrievable chunks."""
        ...


class FixedSizeChunker:
    """Fixed-token sliding window chunker.

    Args:
        chunk_size_tokens: target tokens per chunk (default 512)
        overlap_tokens: overlap between adjacent chunks (default 64)
    """

    def __init__(self, chunk_size_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens

    def chunk(self, filing: Filing) -> list[Chunk]:
        # TODO(@data-lead): tokenize with tiktoken for the embedding model
        # TODO(@data-lead): sliding window with overlap
        # TODO(@data-lead): preserve char_start/char_end for citation back to source
        raise NotImplementedError("FixedSizeChunker.chunk is not yet implemented")


class SectionAwareChunker:
    """Section-aware chunker that respects filing structure.

    Splits at natural section boundaries (Item 1, Item 1A, Item 7, etc.),
    then subdivides oversized sections.

    Args:
        max_tokens_per_chunk: maximum tokens before subdividing (default 800)
        min_tokens_per_chunk: minimum tokens (don't fragment too small, default 100)
    """

    def __init__(
        self,
        max_tokens_per_chunk: int = 800,
        min_tokens_per_chunk: int = 100,
    ) -> None:
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.min_tokens_per_chunk = min_tokens_per_chunk

    def chunk(self, filing: Filing) -> list[Chunk]:
        # TODO(@data-lead): iterate filing.sections, subdivide as needed
        # TODO(@data-lead): preserve section name in Chunk.section
        # TODO(@data-lead): handle tables specially (don't split mid-table)
        raise NotImplementedError("SectionAwareChunker.chunk is not yet implemented")
