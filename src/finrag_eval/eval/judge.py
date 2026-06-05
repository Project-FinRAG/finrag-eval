"""LLM-as-judge for answer quality scoring.

Owner: Evaluation Lead

We score on 5 rubric dimensions:
    1. Correctness    — does the answer match ground truth?
    2. Completeness   — does it cover all required evidence?
    3. Faithfulness   — are claims grounded in retrieved passages?
    4. Citation       — are citations accurate and traceable?
    5. Abstention     — correctly abstaining when evidence is insufficient?

Critical methodology: a 20-30 question human-rated subset is compared with
the LLM judge's scores using Cohen's kappa. If kappa < 0.5, the judge is
unreliable and we need to revise the prompt or use a different model.
"""

from __future__ import annotations

import json

from openai import OpenAI
from pydantic import BaseModel

from finrag_eval.common import Answer, Citation, QAPair, RetrievalResult
from finrag_eval.common.config import settings

JUDGE_PROMPT = """You are a meticulous evaluator of financial question-answering
systems. You are given a question about SEC filings, a gold reference answer and
its supporting evidence, the passages retrieved and shown to the system, and the
system's answer with its citations. Score the system's answer on five dimensions.

Score each of the first four from 0.0 to 1.0:
1. correctness — Does the answer agree with the gold answer on the key facts and
   figures? 1.0 = fully correct; 0.0 = wrong or contradicts the gold answer.
   Penalize incorrect numbers heavily.
2. completeness — Does the answer cover every element the gold answer requires?
   Give partial credit for partial coverage.
3. faithfulness — Is every claim in the answer supported by the RETRIEVED passages
   (not outside knowledge)? Lower the score for any claim not grounded in the
   passages, even if it happens to be true.
4. citation_support — Do the system's cited passages actually contain the
   information the answer relies on? 1.0 = citations accurate and sufficient;
   0.0 = citations missing, wrong, or unsupportive.

Then judge:
5. abstention_correct (true/false) — Did the system make the right call? True if it
   answered when the passages were sufficient, or abstained when they were
   genuinely insufficient. False otherwise.

If the system abstained, it made no factual claims, so judge faithfulness and
citation_support on whatever claims (if any) it did make, and score correctness
and completeness low when an answer was clearly available from the passages.

Question:
{question}

Gold answer:
{gold_answer}

Gold supporting evidence:
{gold_evidence}

Passages retrieved and shown to the system:
{passages}

System answer:
{answer_text}

System cited chunk ids: {answer_citations}
System abstained: {abstained}

Return a JSON object with exactly these fields and nothing else:
  "correctness": number between 0 and 1,
  "completeness": number between 0 and 1,
  "faithfulness": number between 0 and 1,
  "citation_support": number between 0 and 1,
  "abstention_correct": boolean,
  "reasoning": a brief string justifying the scores
"""


class JudgeScore(BaseModel):
    correctness: float  # 0-1
    completeness: float  # 0-1
    faithfulness: float  # 0-1
    citation_support: float  # 0-1
    abstention_correct: bool
    reasoning: str


def _clamp01(value: object) -> float:
    try:
        score = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _format_passages(passages: list[RetrievalResult]) -> str:
    blocks: list[str] = []
    for result in passages:
        chunk = result.chunk
        header = f"[{chunk.chunk_id}] {chunk.ticker} {chunk.filing_accession}"
        if chunk.section:
            header += f" — {chunk.section}"
        blocks.append(f"{header}\n{chunk.text}")
    return "\n\n".join(blocks)


def _format_gold_evidence(evidence: list[Citation]) -> str:
    lines: list[str] = []
    for c in evidence:
        loc = f"[{c.chunk_id}] {c.ticker}"
        if c.section:
            loc += f" — {c.section}"
        lines.append(f"{loc}\n{c.quote or '(no quote provided)'}")
    return "\n\n".join(lines) if lines else "(none provided)"


class AnswerJudge:
    """LLM-as-judge scoring answers against gold answers + evidence."""

    def __init__(self, model: str | None = None) -> None:
        # Defaults to the quality model. Judging with the same model family that
        # generated the answer introduces self-preference bias — disclose it, or
        # set a different judge model.
        self.model = model or settings.llm_quality_model
        self._client: OpenAI | None = None

    def _openai(self) -> OpenAI:
        if self._client is None:
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY not set. Add it to .env or the environment.")
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def score(
        self,
        qa: QAPair,
        answer: Answer,
        passages: list[RetrievalResult] | None = None,
    ) -> JudgeScore:
        """Score one answer against its gold QA pair on the 5-dimension rubric.

        ``passages`` are the retrieved chunks shown to the generator — what
        faithfulness and citation_support are judged against. If omitted, the
        judge falls back to the gold evidence and those two dimensions should be
        read with that caveat.
        """
        evidence_block = (
            _format_passages(passages) if passages else _format_gold_evidence(qa.gold_evidence)
        )
        cited = ", ".join(c.chunk_id for c in answer.citations) or "(none)"

        prompt = JUDGE_PROMPT.format(
            question=qa.question,
            gold_answer=qa.gold_answer,
            gold_evidence=_format_gold_evidence(qa.gold_evidence),
            passages=evidence_block,
            answer_text=answer.answer_text or "(no answer text)",
            answer_citations=cited,
            abstained=answer.abstained,
        )

        client = self._openai()
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            # A judge parse failure is a measurement failure, not a system
            # property — surface it rather than silently scoring zero.
            raise ValueError(f"judge returned unparseable output: {content[:200]!r}") from exc

        return JudgeScore(
            correctness=_clamp01(data.get("correctness")),
            completeness=_clamp01(data.get("completeness")),
            faithfulness=_clamp01(data.get("faithfulness")),
            citation_support=_clamp01(data.get("citation_support")),
            abstention_correct=bool(data.get("abstention_correct", False)),
            reasoning=str(data.get("reasoning", "")),
        )

    def calibrate_against_humans(
        self,
        human_scores: list[JudgeScore],
        judge_scores: list[JudgeScore],
    ) -> dict[str, float]:
        """Compute inter-rater agreement (Cohen's kappa per dimension)."""
        # TODO(@eval-lead): bin to ordinal categories, compute kappa once a
        # 20-30 question human-rated subset exists.
        raise NotImplementedError("AnswerJudge.calibrate_against_humans is not yet implemented")
