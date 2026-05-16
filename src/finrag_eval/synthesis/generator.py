"""Generator — produces grounded answers from query + retrieved passages.

Owner: Retrieval & Modeling Lead

The Generator is intentionally simple. Most of the project's intelligence
lives in the retriever and the eval. The Generator just composes a prompt,
calls an LLM, parses citations, and returns a structured Answer.
"""

from __future__ import annotations

from finrag_eval.common import Answer, RetrievalResult
from finrag_eval.common.config import settings


class Generator:
    """Calls an LLM to produce a grounded answer with citations."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.llm_quality_model

    def answer(
        self,
        question: str,
        passages: list[RetrievalResult],
    ) -> Answer:
        """Produce an answer grounded in the provided passages.

        Returns an Answer with:
            - the generated text
            - citations resolved to chunk_ids
            - cost and latency metadata
            - abstention flag if evidence was insufficient
        """
        # TODO(@retrieval-lead): format passages with stable IDs (P1, P2, ...)
        # TODO(@retrieval-lead): call LLM with QA_PROMPT, parse JSON response
        # TODO(@retrieval-lead): map citation IDs back to Chunk.chunk_id
        # TODO(@retrieval-lead): record input/output tokens, compute cost
        # TODO(@retrieval-lead): measure latency_ms
        raise NotImplementedError("Generator.answer is not yet implemented")
