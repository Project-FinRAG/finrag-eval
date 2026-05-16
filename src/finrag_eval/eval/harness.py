"""EvalHarness — runs a retriever + generator over the full QA set.

Owner: Evaluation Lead

This is the entry point that produces the comparison table in the final
report. One harness run = one row in the results table.

Output is a structured EvalReport that includes per-question metrics,
aggregate metrics, cost, latency, and the model/config that produced it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pydantic import BaseModel

from finrag_eval.eval.judge import JudgeScore
from finrag_eval.eval.qa_dataset import QADataset


class PerQuestionResult(BaseModel):
    qa_id: str
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    evidence_hit: float
    judge_score: JudgeScore | None
    cost_usd: float
    latency_ms: int


class EvalReport(BaseModel):
    config_name: str
    retriever_name: str
    chunker_name: str
    generator_model: str
    judge_model: str
    timestamp: datetime
    commit_sha: str
    n_questions: int

    # Aggregate metrics
    mean_recall_at_10: float
    mean_mrr: float
    mean_ndcg_at_10: float
    mean_evidence_hit: float
    mean_faithfulness: float
    mean_correctness: float

    # Operational
    total_cost_usd: float
    mean_latency_ms: float
    p95_latency_ms: float

    per_question: list[PerQuestionResult]

    def save(self, path: Path) -> None:
        path.write_text(self.model_dump_json(indent=2))


class EvalHarness:
    """Run a single evaluation configuration end-to-end."""

    def __init__(
        self,
        retriever,  # type: ignore[no-untyped-def]
        generator,  # type: ignore[no-untyped-def]
        judge,      # type: ignore[no-untyped-def]
        dataset: QADataset,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.judge = judge
        self.dataset = dataset

    def run(self, config_name: str) -> EvalReport:
        # TODO(@eval-lead): iterate dataset, retrieve, generate, judge
        # TODO(@eval-lead): aggregate per-question results into EvalReport
        # TODO(@eval-lead): include commit SHA via git rev-parse for reproducibility
        raise NotImplementedError("EvalHarness.run is not yet implemented")
