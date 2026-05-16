"""Evaluation package — the crown jewel of FinRAG-Eval.

Owner: Evaluation Lead

This package is what makes the project a research contribution rather than
"another RAG demo." It contains:

    - qa_dataset: load/save the held-out QA pairs
    - metrics: retrieval metrics (Recall@K, MRR, nDCG, evidence-hit)
    - judge: LLM-as-judge for answer quality scoring
    - harness: end-to-end eval runner producing structured reports

Every change in retrieval/ or synthesis/ should trigger an eval regression.
"""

from finrag_eval.eval.harness import EvalHarness, EvalReport
from finrag_eval.eval.judge import AnswerJudge
from finrag_eval.eval.metrics import (
    evidence_hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from finrag_eval.eval.qa_dataset import QADataset

__all__ = [
    "AnswerJudge",
    "EvalHarness",
    "EvalReport",
    "QADataset",
    "evidence_hit_rate",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
]
