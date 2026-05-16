"""Held-out QA dataset — the foundation of the evaluation.

Owner: Evaluation Lead

Target: ~100 QA pairs, hand-curated with gold evidence citations.

Distribution targets:
    By question type (~25 each):
        - factual_lookup     (e.g., "What was AAPL's revenue in FY2024?")
        - multi_doc_synthesis (e.g., "Compare risk factors across MSFT 10-Ks 2022-24")
        - numerical_reasoning (e.g., "What is JPM's YoY change in net interest income?")
        - temporal_comparison (e.g., "How did Goldman's risk language change post-2022?")

    By difficulty (rough): 40% easy, 40% medium, 20% hard

Stored as JSONL for diff-friendly version control.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from finrag_eval.common import QAPair


class QADataset:
    """Load, save, and iterate the held-out QA dataset."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._pairs: list[QAPair] = []

    def load(self) -> None:
        # TODO(@eval-lead): read JSONL, parse each line as QAPair
        raise NotImplementedError("QADataset.load is not yet implemented")

    def save(self) -> None:
        # TODO(@eval-lead): write self._pairs as JSONL
        raise NotImplementedError("QADataset.save is not yet implemented")

    def __iter__(self) -> Iterator[QAPair]:
        return iter(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def filter(
        self,
        question_type: str | None = None,
        difficulty: str | None = None,
    ) -> list[QAPair]:
        # TODO(@eval-lead): return filtered subset for stratified analysis
        raise NotImplementedError("QADataset.filter is not yet implemented")
