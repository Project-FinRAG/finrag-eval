"""End-to-end answer evaluation — retriever + generator + judge over the QA set.

One run = one configuration = one row in the final comparison table. Builds a
single retriever (matching scripts/08), generates a grounded answer per question,
judges it on the 5-dimension rubric, and writes a structured EvalReport.

Usage:
    uv run python scripts/11_eval_e2e.py --retriever bm25 --limit 2   # cheap smoke test
    uv run python scripts/11_eval_e2e.py --retriever dense
    uv run python scripts/11_eval_e2e.py --retriever hybrid --output data/eval_runs/e2e_hybrid.json

Notes:
    Every config (even bm25) calls the OpenAI API for generation AND judging,
    so OPENAI_API_KEY must be set. dense/hybrid additionally embed each query.
    Use --limit during development to validate wiring before paying for all 20.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from finrag_eval.common.config import settings
from finrag_eval.eval.harness import EvalHarness
from finrag_eval.eval.judge import AnswerJudge
from finrag_eval.eval.qa_dataset import QADataset
from finrag_eval.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
)
from finrag_eval.retrieval.bm25 import load_chunks_from_jsonl
from finrag_eval.synthesis.generator import Generator

logger = logging.getLogger(__name__)

DEFAULT_QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")
DEFAULT_INDEX_DIR = Path("data/indexes")

# The persistent dense index — connect read-only, never rebuild from here.
DENSE_INDEX_PATH = DEFAULT_INDEX_DIR / "chroma_dense_labeled"
DENSE_COLLECTION = "finrag_dense"

# strategy -> human-readable chunker label for the report
CHUNKER_NAMES = {"labeled": "section_aware", "fixed": "fixed_size"}


def _require_labeled(name: str, strategy: str) -> None:
    if strategy != "labeled":
        raise ValueError(
            f"--retriever {name} requires --strategy labeled "
            f"(the only dense index built is {DENSE_INDEX_PATH.name})."
        )


def _build_bm25(strategy: str) -> BM25Retriever:
    index_path = DEFAULT_INDEX_DIR / f"bm25_{strategy}"
    retriever = BM25Retriever(index_path=index_path)
    if (index_path / "index.pkl").exists():
        logger.info("Loading existing BM25 index from %s", index_path)
        retriever.load()
    else:
        logger.info("Building BM25 index for strategy=%r...", strategy)
        chunks = load_chunks_from_jsonl(strategy)  # type: ignore[arg-type]
        logger.info("  Loaded %d chunks", len(chunks))
        retriever.index(chunks)
    return retriever


def _build_dense() -> DenseRetriever:
    retriever = DenseRetriever(index_path=DENSE_INDEX_PATH, collection_name=DENSE_COLLECTION)
    logger.info(
        "Connecting to dense index at %s (collection %r)", DENSE_INDEX_PATH, DENSE_COLLECTION
    )
    retriever.load()
    return retriever


def build_retriever(name: str, strategy: str) -> Retriever:
    if name == "bm25":
        return _build_bm25(strategy)
    if name == "dense":
        _require_labeled(name, strategy)
        return _build_dense()
    if name == "hybrid":
        _require_labeled(name, strategy)
        return HybridRetriever(bm25=_build_bm25("labeled"), dense=_build_dense(), rrf_k=60)
    raise ValueError(f"Unknown retriever {name!r}.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end answer eval.")
    p.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    p.add_argument("--strategy", default="labeled")
    p.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    p.add_argument("--top-k", type=int, default=10, help="Passages fed to the generator.")
    p.add_argument("--limit", type=int, default=None, help="Evaluate only the first N pairs.")
    p.add_argument("--generator-model", default=None, help="Override generator model.")
    p.add_argument("--judge-model", default=None, help="Override judge model.")
    p.add_argument("--chunker-name", default=None, help="Override chunker label.")
    p.add_argument("--config-name", default=None)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY not set — generation and judging need it. Add it to .env.")

    chunker_name = args.chunker_name or CHUNKER_NAMES.get(args.strategy, args.strategy)
    config_name = args.config_name or f"{args.retriever}_{chunker_name}"

    retriever = build_retriever(args.retriever, args.strategy)
    generator = Generator(model=args.generator_model)
    judge = AnswerJudge(model=args.judge_model)

    dataset = QADataset(args.qa_path)
    dataset.load()
    if args.limit is not None:
        # Smoke-test convenience: evaluate only the first N pairs.
        dataset._pairs = dataset._pairs[: args.limit]

    harness = EvalHarness(retriever=retriever, generator=generator, judge=judge, dataset=dataset)
    report = harness.run(config_name, chunker_name=chunker_name, top_k=args.top_k)

    output = args.output or (settings.eval_runs_dir / f"e2e_{args.retriever}_{args.strategy}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    report.save(output)

    print()
    print(f"config         {report.config_name}")
    print(f"retriever      {report.retriever_name}")
    print(f"chunker        {report.chunker_name}")
    print(f"generator      {report.generator_model}")
    print(f"judge          {report.judge_model}")
    print(f"commit         {report.commit_sha}")
    print(f"n_questions    {report.n_questions}")
    print(f"recall@10      {report.mean_recall_at_10:.3f}")
    print(f"mrr            {report.mean_mrr:.3f}")
    print(f"ndcg@10        {report.mean_ndcg_at_10:.3f}")
    print(f"evidence_hit   {report.mean_evidence_hit:.3f}")
    print(f"faithfulness   {report.mean_faithfulness:.3f}")
    print(f"correctness    {report.mean_correctness:.3f}")
    print(f"total_cost_usd ${report.total_cost_usd:.4f}")
    print(f"latency_ms     mean={report.mean_latency_ms:.0f}  p95={report.p95_latency_ms:.0f}")
    print(f"\nsaved -> {output}")


if __name__ == "__main__":
    main()
