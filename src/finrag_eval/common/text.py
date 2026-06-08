"""Shared text normalization for matching QA quotes against chunk text.

SEC filing HTML uses typographic punctuation (smart quotes, non-breaking
spaces); QA-pair quotes are typically typed with straight ASCII. Both are
normalized to a common form before substring matching, so the same logic
backs gold-quote verification and the soft-match retrieval metric.
"""

from __future__ import annotations

import re

QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",  # left single quote
        "\u2019": "'",  # right single quote
        "\u201c": '"',  # left double quote
        "\u201d": '"',  # right double quote
        "\u00a0": " ",  # non-breaking space
    }
)


def normalize_for_search(s: str) -> str:
    """Collapse whitespace, lowercase, and normalize smart quotes / NBSPs to ASCII."""
    return re.sub(r"\s+", " ", s.translate(QUOTE_TRANSLATION)).lower().strip()
