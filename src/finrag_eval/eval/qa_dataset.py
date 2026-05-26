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
        """Read JSONL, parse each line as a QAPair."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"QA dataset not found at {self.path}. "
                "Expected JSONL with one QAPair per line."
            )
        pairs: list[QAPair] = []
        with self.path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    pairs.append(QAPair.model_validate_json(line))
                except Exception as e:
                    raise ValueError(
                        f"Failed to parse QA pair at line {line_no} of {self.path}: {e}"
                    ) from e
        self._pairs = pairs

    def save(self) -> None:
        """Write self._pairs as JSONL."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for pair in self._pairs:
                f.write(pair.model_dump_json() + "\n")

    def __iter__(self) -> Iterator[QAPair]:
        return iter(self._pairs)

    def __len__(self) -> int:
        return len(self._pairs)

    def filter(
        self,
        question_type: str | None = None,
        difficulty: str | None = None,
    ) -> list[QAPair]:
        """Return pairs matching the given filters. Both filters are optional and AND-combined."""
        result = list(self._pairs)
        if question_type is not None:
            result = [p for p in result if p.question_type == question_type]
        if difficulty is not None:
            result = [p for p in result if p.difficulty == difficulty]
        return result
