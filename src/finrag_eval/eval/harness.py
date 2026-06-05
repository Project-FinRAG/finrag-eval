"""EvalHarness — runs a retriever + generator over the full QA set.

Owner: Evaluation Lead

This is the entry point that produces the comparison table in the final
report. One harness run = one row in the results table.

Output is a structured EvalReport that includes per-question metrics,
aggregate metrics, cost, latency, and the model/config that produced it.
"""

from __future__ import annotations

import logging
import math
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from finrag_eval.eval.judge import AnswerJudge, JudgeScore
from finrag_eval.eval.metrics import (
    evidence_hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from finrag_eval.eval.qa_dataset import QADataset
from finrag_eval.retrieval.base import Retriever
from finrag_eval.synthesis.generator import Generator

logger = logging.getLogger(__name__)


class PerQuestionResult(BaseModel):
    qa_id: str
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    evidence_hit: float
    judge_score: JudgeScore | None
    answer_text: str
    abstained: bool
    retrieved_ids: list[str]
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - rank) + ordered[high] * (rank - low)


def _git_commit_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.stdout.strip() or "unknown"


class EvalHarness:
    """Run a single evaluation configuration end-to-end."""

    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        judge: AnswerJudge,
        dataset: QADataset,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.judge = judge
        self.dataset = dataset

    def run(
        self,
        config_name: str,
        *,
        chunker_name: str,
        top_k: int = 10,
        retriever_name: str | None = None,
    ) -> EvalReport:
        n = len(self.dataset)
        if n == 0:
            raise RuntimeError("QADataset is empty — call dataset.load() before run().")

        per_question: list[PerQuestionResult] = []
        for i, qa in enumerate(self.dataset, start=1):
            logger.info("[%d/%d] %s", i, n, qa.qa_id)
            results = self.retriever.retrieve(qa.question, top_k)
            retrieved_ids = [r.chunk.chunk_id for r in results]
            gold = {c.chunk_id for c in qa.gold_evidence}

            answer = self.generator.answer(qa.question, results)

            judge_score: JudgeScore | None
            try:
                judge_score = self.judge.score(qa, answer, results)
            except Exception:
                logger.warning(
                    "Judge failed on %s; recording judge_score=None", qa.qa_id, exc_info=True
                )
                judge_score = None

            per_question.append(
                PerQuestionResult(
                    qa_id=qa.qa_id,
                    recall_at_10=recall_at_k(retrieved_ids, gold, 10),
                    mrr=mean_reciprocal_rank(retrieved_ids, gold),
                    ndcg_at_10=ndcg_at_k(retrieved_ids, gold, 10),
                    evidence_hit=evidence_hit_rate(retrieved_ids, gold, 10),
                    judge_score=judge_score,
                    answer_text=answer.answer_text,
                    abstained=answer.abstained,
                    retrieved_ids=retrieved_ids,
                    cost_usd=answer.cost_usd,
                    latency_ms=answer.latency_ms,
                )
            )

        scored = [pq.judge_score for pq in per_question if pq.judge_score is not None]
        latencies = [float(pq.latency_ms) for pq in per_question]
        resolved_retriever_name = (
            retriever_name or getattr(self.retriever, "name", None) or type(self.retriever).__name__
        )

        return EvalReport(
            config_name=config_name,
            retriever_name=resolved_retriever_name,
            chunker_name=chunker_name,
            generator_model=self.generator.model,
            judge_model=self.judge.model,
            timestamp=datetime.now(UTC),
            commit_sha=_git_commit_sha(),
            n_questions=n,
            mean_recall_at_10=_mean([pq.recall_at_10 for pq in per_question]),
            mean_mrr=_mean([pq.mrr for pq in per_question]),
            mean_ndcg_at_10=_mean([pq.ndcg_at_10 for pq in per_question]),
            mean_evidence_hit=_mean([pq.evidence_hit for pq in per_question]),
            mean_faithfulness=_mean([s.faithfulness for s in scored]),
            mean_correctness=_mean([s.correctness for s in scored]),
            total_cost_usd=sum(pq.cost_usd for pq in per_question),
            mean_latency_ms=_mean(latencies),
            p95_latency_ms=_percentile(latencies, 0.95),
            per_question=per_question,
        )
