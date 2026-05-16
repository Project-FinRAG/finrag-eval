"""Smoke tests — verify the package imports and basic types work."""

from __future__ import annotations

from datetime import date

from finrag_eval import __version__
from finrag_eval.common import Citation, Filing, FilingType, QAPair, QuestionType


def test_version_is_set() -> None:
    assert __version__ == "0.1.0"


def test_filing_construction() -> None:
    f = Filing(
        ticker="AAPL",
        cik="0000320193",
        filing_type=FilingType.TEN_K,
        filing_date=date(2024, 11, 1),
        accession_number="0000320193-24-000123",
        raw_text="Annual report...",
    )
    assert f.ticker == "AAPL"
    assert f.filing_type == FilingType.TEN_K


def test_qa_pair_construction() -> None:
    qa = QAPair(
        qa_id="q-001",
        question="What was Apple's FY2024 revenue?",
        gold_answer="$391.0 billion",
        gold_evidence=[
            Citation(
                chunk_id="aapl-2024-10k-mda-0007",
                filing_accession="0000320193-24-000123",
                ticker="AAPL",
                section="Item 7 - MD&A",
            )
        ],
        question_type=QuestionType.FACTUAL_LOOKUP,
        difficulty="easy",
    )
    assert qa.question_type == QuestionType.FACTUAL_LOOKUP
    assert len(qa.gold_evidence) == 1
