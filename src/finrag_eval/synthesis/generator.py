"""Generator — produces grounded answers from query + retrieved passages.

Owner: Retrieval & Modeling Lead

The Generator is intentionally simple. Most of the project's intelligence
lives in the retriever and the eval. The Generator just composes a prompt,
calls an LLM, parses citations, and returns a structured Answer.
"""

from __future__ import annotations

import json
import time

from openai import OpenAI

from finrag_eval.common import Answer, Citation, RetrievalResult
from finrag_eval.common.config import settings
from finrag_eval.synthesis.prompts import QA_PROMPT

# OpenAI list prices in USD per 1M tokens, as (input, output). Verified against
# openai.com/api/pricing mid-2026; update when prices change. Does not model
# cached-input or Batch-API discounts.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compute call cost from token counts and the model price table."""
    input_price, output_price = MODEL_PRICING.get(model, (0.0, 0.0))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000


def _format_passages(
    passages: list[RetrievalResult],
) -> tuple[str, dict[str, RetrievalResult]]:
    """Render passages with stable IDs (P1, P2, ...) and a map back to chunks."""
    id_map: dict[str, RetrievalResult] = {}
    blocks: list[str] = []
    for i, result in enumerate(passages, start=1):
        pid = f"P{i}"
        id_map[pid] = result
        chunk = result.chunk
        header = f"[{pid}] {chunk.ticker} {chunk.filing_accession}"
        if chunk.section:
            header += f" — {chunk.section}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks), id_map


class Generator:
    """Calls an LLM to produce a grounded answer with citations."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model or settings.llm_quality_model
        self._client: OpenAI | None = None

    def _openai(self) -> OpenAI:
        if self._client is None:
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or the environment.")
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def answer(self, question: str, passages: list[RetrievalResult]) -> Answer:
        """Produce an answer grounded in the provided passages."""
        passages_block, id_map = _format_passages(passages)
        prompt = QA_PROMPT.format(question=question, passages=passages_block)

        client = self._openai()
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)

        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        cost_usd = estimate_cost_usd(self.model, input_tokens, output_tokens)

        content = response.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # One malformed response shouldn't abort a full eval run — record it
            # as an abstention and keep going.
            return Answer(
                question=question,
                answer_text="",
                citations=[],
                abstained=True,
                abstention_reason=f"unparseable model output: {content[:200]!r}",
                model=self.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
                latency_ms=latency_ms,
            )

        abstained = bool(data.get("abstain", False))
        citations = self._resolve_citations(data.get("citations") or [], id_map)

        return Answer(
            question=question,
            answer_text=str(data.get("answer", "")),
            citations=citations,
            abstained=abstained,
            abstention_reason=data.get("abstention_reason") if abstained else None,
            model=self.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )

    @staticmethod
    def _resolve_citations(
        cited_ids: list[object],
        id_map: dict[str, RetrievalResult],
    ) -> list[Citation]:
        """Map model-emitted passage IDs (P1, P3, ...) back to chunk citations."""
        citations: list[Citation] = []
        seen: set[str] = set()
        for raw in cited_ids:
            result = id_map.get(str(raw).strip())
            if result is None or result.chunk.chunk_id in seen:
                continue
            seen.add(result.chunk.chunk_id)
            chunk = result.chunk
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    filing_accession=chunk.filing_accession,
                    ticker=chunk.ticker,
                    section=chunk.section,
                    quote=None,
                )
            )
        return citations
