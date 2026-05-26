"""Retrieval evaluation harness — first real research-phase output.

Runs a retriever over the held-out QA dataset and reports Recall@k, MRR,
nDCG@10, and evidence-hit@10. This is the retrieval-only evaluation;
end-to-end answer evaluation (with a generator and judge) comes later.

Usage:
    uv run python scripts/08_eval_retrieval.py
    uv run python scripts/08_eval_retrieval.py --retriever bm25 --strategy labeled
    uv run python scripts/08_eval_retrieval.py --output data/eval_runs/bm25_labeled.json

Output:
    - Per-question table printed to stdout
    - Aggregate metrics overall and stratified by question_type
    - Optional JSON file with full results for later comparison
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from finrag_eval.common import QAPair
from finrag_eval.eval.metrics import (
    evidence_hit_rate,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)
from finrag_eval.eval.qa_dataset import QADataset
from finrag_eval.retrieval.bm25 import BM25Retriever, load_chunks_from_jsonl

DEFAULT_QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")
DEFAULT_INDEX_DIR = Path("data/indexes")


def build_retriever(name: str, strategy: str) -> BM25Retriever:
    """Build a retriever and index it on the requested chunk strategy.

    Today only BM25 is implemented; dense and hybrid will be added once
    Vidhee's DenseRetriever lands.
    """
    if name != "bm25":
        raise ValueError(
            f"Only 'bm25' is implemented today. Got {name!r}. "
            "Dense and hybrid retrievers are pending."
        )

    index_path = DEFAULT_INDEX_DIR / f"bm25_{strategy}"
    retriever = BM25Retriever(index_path=index_path)

    if (index_path / "index.pkl").exists():
        print(f"Loading existing BM25 index from {index_path}")
        retriever.load()
    else:
        print(f"Building BM25 index for strategy={strategy!r}...")
        chunks = load_chunks_from_jsonl(strategy)  # type: ignore[arg-type]
        print(f"  Loaded {len(chunks):,} chunks")
        retriever.index(chunks)

    return retriever


def evaluate_pair(
    pair: QAPair,
    retriever: BM25Retriever,
    k_values: list[int],
) -> dict:
    """Run retrieval for a single QA pair and compute all metrics."""
    gold_ids = {ev.chunk_id for ev in pair.gold_evidence}
    max_k = max(k_values)

    t0 = time.perf_counter()
    results = retriever.retrieve(pair.question, k=max_k)
    latency_ms = (time.perf_counter() - t0) * 1000

    retrieved_ids = [r.chunk.chunk_id for r in results]

    metrics: dict[str, float | int | str] = {
        "qa_id": pair.qa_id,
        "question_type": str(pair.question_type),
        "difficulty": pair.difficulty,
        "n_gold": len(gold_ids),
        "latency_ms": round(latency_ms, 1),
    }
    for k in k_values:
        metrics[f"recall@{k}"] = round(recall_at_k(retrieved_ids, gold_ids, k), 3)
        metrics[f"evidence_hit@{k}"] = round(evidence_hit_rate(retrieved_ids, gold_ids, k), 3)
    metrics["mrr"] = round(mean_reciprocal_rank(retrieved_ids, gold_ids), 3)
    metrics[f"ndcg@{max_k}"] = round(ndcg_at_k(retrieved_ids, gold_ids, max_k), 3)
    metrics["retrieved_ids"] = retrieved_ids

    return metrics


def aggregate(results: list[dict], k_values: list[int], group_by: str | None = None) -> dict:
    """Compute mean metrics across results, optionally grouped."""
    if group_by:
        groups: dict[str, list[dict]] = {}
        for r in results:
            key = r[group_by]
            groups.setdefault(key, []).append(r)
        return {
            group: aggregate(group_results, k_values, group_by=None)
            for group, group_results in groups.items()
        }

    if not results:
        return {}

    max_k = max(k_values)
    agg: dict[str, float | int] = {"n": len(results)}
    for k in k_values:
        agg[f"recall@{k}"] = round(mean(r[f"recall@{k}"] for r in results), 3)
        agg[f"evidence_hit@{k}"] = round(mean(r[f"evidence_hit@{k}"] for r in results), 3)
    agg["mrr"] = round(mean(r["mrr"] for r in results), 3)
    agg[f"ndcg@{max_k}"] = round(mean(r[f"ndcg@{max_k}"] for r in results), 3)
    agg["mean_latency_ms"] = round(mean(r["latency_ms"] for r in results), 1)
    return agg


def print_table(results: list[dict], k_values: list[int]) -> None:
    """Per-question results table."""
    print(f"\n{'='*100}")
    print(f"{'qa_id':<8} {'type':<22} {'diff':<7} {'n_gold':<7}", end="")
    for k in k_values:
        print(f" R@{k:<3}", end="")
    print(f"  {'MRR':<6} nDCG@{max(k_values)}  lat(ms)")
    print("=" * 100)

    for r in results:
        print(f"{r['qa_id']:<8} {str(r['question_type']):<22} "
              f"{r['difficulty']:<7} {r['n_gold']:<7}", end="")
        for k in k_values:
            print(f" {r[f'recall@{k}']:<5}", end="")
        print(f"  {r['mrr']:<6} {r[f'ndcg@{max(k_values)}']:<7} {r['latency_ms']}")
    print()


def print_aggregates(results: list[dict], k_values: list[int]) -> None:
    """Overall + stratified aggregates."""
    overall = aggregate(results, k_values)
    print(f"\n{'='*70}")
    print("AGGREGATE METRICS (overall)")
    print("=" * 70)
    for key, val in overall.items():
        print(f"  {key:<20} {val}")

    by_type = aggregate(results, k_values, group_by="question_type")
    print(f"\n{'='*70}")
    print("BY QUESTION TYPE")
    print("=" * 70)
    for qt, agg in by_type.items():
        print(f"\n  {qt}  (n={agg['n']})")
        for key, val in agg.items():
            if key == "n":
                continue
            print(f"    {key:<20} {val}")

    by_diff = aggregate(results, k_values, group_by="difficulty")
    print(f"\n{'='*70}")
    print("BY DIFFICULTY")
    print("=" * 70)
    for diff, agg in by_diff.items():
        print(f"\n  {diff}  (n={agg['n']})")
        for key, val in agg.items():
            if key == "n":
                continue
            print(f"    {key:<20} {val}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH,
                        help=f"Path to qa_pairs.jsonl (default: {DEFAULT_QA_PATH})")
    parser.add_argument("--retriever", default="bm25",
                        choices=["bm25"],
                        help="Retriever to evaluate (default: bm25)")
    parser.add_argument("--strategy", default="labeled",
                        choices=["labeled", "strict", "fixed_size", "all"],
                        help="Chunk-loading strategy (default: labeled = "
                             "section_aware + hybrid_section_aware)")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10],
                        help="k values for Recall@k and evidence_hit@k (default: 5 10)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Optional path to save results JSON for later comparison")
    args = parser.parse_args()

    print(f"Retriever:  {args.retriever}")
    print(f"Strategy:   {args.strategy}")
    print(f"QA path:    {args.qa_path}")
    print(f"k values:   {args.k}")
    print()

    # Load QA pairs
    dataset = QADataset(args.qa_path)
    dataset.load()
    print(f"Loaded {len(dataset)} QA pairs.\n")

    # Build/load retriever
    retriever = build_retriever(args.retriever, args.strategy)

    # Run evaluation
    print("\nRunning evaluation...")
    results = [evaluate_pair(pair, retriever, args.k) for pair in dataset]

    # Print results
    print_table(results, args.k)
    print_aggregates(results, args.k)

    # Save if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_payload = {
            "retriever": args.retriever,
            "strategy": args.strategy,
            "qa_path": str(args.qa_path),
            "k_values": args.k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_pairs": len(results),
            "per_question": results,
            "aggregate_overall": aggregate(results, args.k),
            "aggregate_by_type": aggregate(results, args.k, group_by="question_type"),
            "aggregate_by_difficulty": aggregate(results, args.k, group_by="difficulty"),
        }
        args.output.write_text(json.dumps(output_payload, indent=2, default=str))
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())