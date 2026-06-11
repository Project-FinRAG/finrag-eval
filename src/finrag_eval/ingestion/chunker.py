"""Chunking strategies for SEC filings.

Owner: Data & Application Lead

We compare two strategies:
    1. FixedSizeChunker: fixed-token sliding window with overlap
    2. SectionAwareChunker: respects 10-K/10-Q section boundaries

Both implement the Chunker Protocol so they're swappable in the eval matrix.
"""

from __future__ import annotations

import re
from typing import Protocol

import tiktoken

from finrag_eval.common import Chunk, Filing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ENCODING = "cl100k_base"

STANDARD_ITEMS: dict[str, str] = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "[Reserved]",
    "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accounting Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

ITEM_HEADER_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[a-c]?)\b\.?\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Chunker(Protocol):
    """Protocol for chunking strategies. Any chunker must implement this."""

    def chunk(self, filing: Filing) -> list[Chunk]:
        """Split a filing into retrievable chunks."""
        ...


# ---------------------------------------------------------------------------
# FixedSizeChunker
# ---------------------------------------------------------------------------


class FixedSizeChunker:
    """Fixed-token sliding window chunker.

    Tokenizes the full filing text with tiktoken (cl100k_base), then slides a
    window of ``chunk_size_tokens`` tokens with ``overlap_tokens`` overlap.
    char_start / char_end are approximated by decoding each window back to text
    and tracking the cumulative character offset.

    Args:
        chunk_size_tokens: target tokens per chunk (default 512)
        overlap_tokens: overlap between adjacent chunks (default 64)
    """

    def __init__(self, chunk_size_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self.chunk_size_tokens = chunk_size_tokens
        self.overlap_tokens = overlap_tokens
        self._enc: tiktoken.Encoding | None = None

    def _get_enc(self) -> tiktoken.Encoding:
        if self._enc is None:
            self._enc = tiktoken.get_encoding(ENCODING)
        return self._enc

    def chunk(self, filing: Filing) -> list[Chunk]:
        enc = self._get_enc()
        text = filing.raw_text
        tokens = enc.encode(text)
        step = self.chunk_size_tokens - self.overlap_tokens

        chunks: list[Chunk] = []
        char_cursor = 0
        for chunk_idx, start in enumerate(range(0, len(tokens), step)):
            window = tokens[start : start + self.chunk_size_tokens]
            if not window:
                break
            window_text = enc.decode(window)
            char_start = char_cursor
            char_end = char_start + len(window_text)
            chunks.append(
                Chunk(
                    chunk_id=f"{filing.ticker}_{filing.accession_number}_fixed_{chunk_idx:04d}",
                    filing_accession=filing.accession_number,
                    ticker=filing.ticker,
                    filing_type=filing.filing_type,
                    section=None,
                    text=window_text,
                    char_start=char_start,
                    char_end=char_end,
                    token_count=len(window),
                )
            )
            # Advance cursor by the non-overlapping portion only
            non_overlap = enc.decode(tokens[start : start + step])
            char_cursor += len(non_overlap)

        return chunks


# ---------------------------------------------------------------------------
# SectionAwareChunker
# ---------------------------------------------------------------------------


class SectionAwareChunker:
    """Section-aware chunker that respects filing structure.

    Splits at natural section boundaries (Item 1, Item 1A, Item 7, etc.)
    detected via regex, then subdivides oversized sections into
    ``max_tokens_per_chunk``-token windows with overlap equal to
    ``max_tokens_per_chunk - min_tokens_per_chunk``.

    Args:
        max_tokens_per_chunk: maximum tokens before subdividing (default 800)
        min_tokens_per_chunk: minimum tokens — sets overlap size (default 100)
    """

    def __init__(
        self,
        max_tokens_per_chunk: int = 800,
        min_tokens_per_chunk: int = 100,
    ) -> None:
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.min_tokens_per_chunk = min_tokens_per_chunk
        self._enc: tiktoken.Encoding | None = None

    def _get_enc(self) -> tiktoken.Encoding:
        if self._enc is None:
            self._enc = tiktoken.get_encoding(ENCODING)
        return self._enc

    @staticmethod
    def _find_sections(text: str) -> dict[str, tuple[int, int, str]]:
        """Detect Item sections; return {item_num: (char_start, char_end, title)}."""
        last_match: dict[str, re.Match[str]] = {}
        for m in ITEM_HEADER_RE.finditer(text):
            last_match[m.group(1).upper()] = m

        sorted_items = sorted(last_match.items(), key=lambda x: x[1].start())
        sections: dict[str, tuple[int, int, str]] = {}
        for i, (item_num, match) in enumerate(sorted_items):
            start = match.start()
            end = sorted_items[i + 1][1].start() if i + 1 < len(sorted_items) else len(text)
            title = match.group(2).strip() or STANDARD_ITEMS.get(item_num, "Unknown")
            sections[item_num] = (start, end, title)
        return sections

    def chunk(self, filing: Filing) -> list[Chunk]:
        enc = self._get_enc()
        text = filing.raw_text
        sections = self._find_sections(text)

        # Fall back to the full text as a single unnamed section if none found
        if not sections:
            sections = {"0": (0, len(text), "Full Document")}

        step = self.max_tokens_per_chunk - self.min_tokens_per_chunk
        chunks: list[Chunk] = []
        chunk_idx = 0

        for item_num in sorted(sections.keys(), key=lambda x: sections[x][0]):
            char_start, char_end, title = sections[item_num]
            section_text = text[char_start:char_end]
            section_label = f"Item {item_num} - {STANDARD_ITEMS.get(item_num, title)}"

            tokens = enc.encode(section_text)

            for i in range(0, len(tokens), step):
                window = tokens[i : i + self.max_tokens_per_chunk]
                if not window:
                    break
                window_text = enc.decode(window)
                w_char_start = char_start + len(enc.decode(tokens[:i]))
                w_char_end = w_char_start + len(window_text)
                chunks.append(
                    Chunk(
                        chunk_id=(
                            f"{filing.ticker}_{filing.accession_number}"
                            f"_item{item_num}_{chunk_idx:04d}"
                        ),
                        filing_accession=filing.accession_number,
                        ticker=filing.ticker,
                        filing_type=filing.filing_type,
                        section=section_label,
                        text=window_text,
                        char_start=w_char_start,
                        char_end=w_char_end,
                        token_count=len(window),
                    )
                )
                chunk_idx += 1

        return chunks
