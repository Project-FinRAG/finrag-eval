"""Shared types used across all FinRAG-Eval packages."""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class FilingType(str, Enum):
    TEN_K = "10-K"
    TEN_Q = "10-Q"
    EIGHT_K = "8-K"


class Filing(BaseModel):
    """A single SEC filing with metadata + raw text."""

    ticker: str
    cik: str
    filing_type: FilingType
    filing_date: date
    period_of_report: date | None = None
    accession_number: str
    raw_text: str
    sections: dict[str, str] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A retrievable chunk of text with provenance back to its source filing."""

    chunk_id: str
    filing_accession: str
    ticker: str
    filing_type: FilingType
    section: str | None = None
    text: str
    char_start: int
    char_end: int
    token_count: int


class RetrievalResult(BaseModel):
    """A retrieved chunk with its relevance score."""

    chunk: Chunk
    score: float
    rank: int


class Citation(BaseModel):
    """Pointer from an answer back to supporting evidence."""

    chunk_id: str
    filing_accession: str
    ticker: str
    section: str | None
    quote: str | None = None


class Answer(BaseModel):
    """The generated answer with supporting citations and metadata."""

    question: str
    answer_text: str
    citations: list[Citation]
    abstained: bool = False
    abstention_reason: str | None = None
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: int


class QuestionType(str, Enum):
    FACTUAL_LOOKUP = "factual_lookup"
    MULTI_DOC_SYNTHESIS = "multi_doc_synthesis"
    NUMERICAL_REASONING = "numerical_reasoning"
    TEMPORAL_COMPARISON = "temporal_comparison"


Difficulty = Literal["easy", "medium", "hard"]


class QAPair(BaseModel):
    """A held-out evaluation QA pair with gold evidence citations."""

    qa_id: str
    question: str
    gold_answer: str
    gold_evidence: list[Citation]
    question_type: QuestionType
    difficulty: Difficulty
    notes: str | None = None
