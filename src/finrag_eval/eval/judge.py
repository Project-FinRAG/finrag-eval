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

from pydantic import BaseModel

from finrag_eval.common import Answer, QAPair


class JudgeScore(BaseModel):
    correctness: float  # 0-1
    completeness: float  # 0-1
    faithfulness: float  # 0-1
    citation_support: float  # 0-1
    abstention_correct: bool
    reasoning: str


class AnswerJudge:
    """LLM-as-judge scoring answers against gold answers + evidence."""

    def __init__(self, model: str | None = None) -> None:
        self.model = model

    def score(self, qa: QAPair, answer: Answer) -> JudgeScore:
        # TODO(@eval-lead): call LLM with structured rubric prompt
        # TODO(@eval-lead): return JudgeScore with reasoning chain
        raise NotImplementedError("AnswerJudge.score is not yet implemented")

    def calibrate_against_humans(
        self,
        human_scores: list[JudgeScore],
        judge_scores: list[JudgeScore],
    ) -> dict[str, float]:
        """Compute inter-rater agreement (Cohen's kappa per dimension)."""
        # TODO(@eval-lead): bin to ordinal categories, compute kappa
        raise NotImplementedError("AnswerJudge.calibrate_against_humans is not yet implemented")
