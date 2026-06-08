"""Retrieval evaluation harness — first real research-phase output.

Runs one or more retrievers over the held-out QA dataset and reports
Recall@k, MRR, nDCG@10, and evidence_hit@10. This is the retrieval-only
evaluation; end-to-end answer evaluation (generator + judge) comes later.

Usage:
    uv run python scripts/08_eval_retrieval.py
    uv run python scripts/08_eval_retrieval.py --retriever bm25 --strategy labeled
    uv run python scripts/08_eval_retrieval.py --retriever dense
    uv run python scripts/08_eval_retrieval.py --retriever hybrid
    uv run python scripts/08_eval_retrieval.py --retriever all --output data/eval_runs/compare.json

Note:
    dense, hybrid, and all require --strategy labeled: the only dense index
    built is chroma_dense_labeled (collection 'finrag_dense'). dense/hybrid
    runs embed each query via the OpenAI API, so OPENAI_API_KEY must be set
    (DenseRetriever loads it from the environment or a .env file).

Output:
    - Per-question table printed to stdout (per retriever)
    - Aggregate metrics overall and stratified by question_type and difficulty
    - In --retriever all mode, a side-by-side overall comparison table
    - Optional JSON file with full results for later comparison
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
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
from finrag_eval.retrieval import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    Retriever,
)
from finrag_eval.retrieval.bm25 import load_chunks_from_jsonl

DEFAULT_QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")
DEFAULT_INDEX_DIR = Path("data/indexes")

# The persistent dense index. Do NOT rebuild it from the eval harness — it is
# costly to recreate (~$0.60 + ~5 min). We only ever connect to it read-only.
DENSE_INDEX_PATH = DEFAULT_INDEX_DIR / "chroma_dense_labeled"
DENSE_COLLECTION = "finrag_dense"


def _require_labeled(name: str, strategy: str) -> None:
    """dense/hybrid/all only work against the labeled dense index."""
    if strategy != "labeled":
        raise ValueError(
            f"--retriever {name} requires --strategy labeled "
            f"(the only dense index built is {DENSE_INDEX_PATH.name}, "
            f"collection {DENSE_COLLECTION!r}). Got --strategy {strategy!r}."
        )


def _build_bm25(strategy: str) -> BM25Retriever:
    """Load the BM25 index for a strategy, building it only if absent."""
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


def _build_dense() -> DenseRetriever:
    """Connect to the existing persistent dense index (read-only, no rebuild)."""
    retriever = DenseRetriever(
        index_path=DENSE_INDEX_PATH,
        collection_name=DENSE_COLLECTION,
    )
    print(f"Connecting to dense index at {DENSE_INDEX_PATH} (collection {DENSE_COLLECTION!r})")
    retriever.load()
    return retriever


def build_retriever(name: str, strategy: str) -> Retriever:
    """Build a single retriever, indexing/loading as needed."""
    if name == "bm25":
        return _build_bm25(strategy)
    if name == "dense":
        _require_labeled(name, strategy)
        return _build_dense()
    if name == "hybrid":
        _require_labeled(name, strategy)
        bm25 = _build_bm25("labeled")
        dense = _build_dense()
        return HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)
    raise ValueError(f"Unknown retriever {name!r}.")


def build_all_retrievers(strategy: str) -> list[Retriever]:
    """Build bm25, dense, and hybrid once, sharing the bm25/dense instances."""
    _require_labeled("all", strategy)
    bm25 = _build_bm25("labeled")
    dense = _build_dense()
    hybrid = HybridRetriever(bm25=bm25, dense=dense, rrf_k=60)
    retrievers: list[Retriever] = [bm25, dense, hybrid]
    return retrievers


def evaluate_pair(
    pair: QAPair,
    retriever: Retriever,
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


def _run_payload(results: list[dict], k_values: list[int]) -> dict:
    """Per-run JSON payload shared by single-retriever and compare modes."""
    return {
        "per_question": results,
        "aggregate_overall": aggregate(results, k_values),
        "aggregate_by_type": aggregate(results, k_values, group_by="question_type"),
        "aggregate_by_difficulty": aggregate(results, k_values, group_by="difficulty"),
    }


def print_table(results: list[dict], k_values: list[int]) -> None:
    """Per-question results table."""
    print(f"\n{'=' * 100}")
    print(f"{'qa_id':<8} {'type':<22} {'diff':<7} {'n_gold':<7}", end="")
    for k in k_values:
        print(f" R@{k:<3}", end="")
    print(f"  {'MRR':<6} nDCG@{max(k_values)}  lat(ms)")
    print("=" * 100)

    for r in results:
        print(
            f"{r['qa_id']:<8} {r['question_type']!s:<22} {r['difficulty']:<7} {r['n_gold']:<7}",
            end="",
        )
        for k in k_values:
            print(f" {r[f'recall@{k}']:<5}", end="")
        print(f"  {r['mrr']:<6} {r[f'ndcg@{max(k_values)}']:<7} {r['latency_ms']}")
    print()


def print_aggregates(results: list[dict], k_values: list[int]) -> None:
    """Overall + stratified aggregates."""
    overall = aggregate(results, k_values)
    print(f"\n{'=' * 70}")
    print("AGGREGATE METRICS (overall)")
    print("=" * 70)
    for key, val in overall.items():
        print(f"  {key:<20} {val}")

    by_type = aggregate(results, k_values, group_by="question_type")
    print(f"\n{'=' * 70}")
    print("BY QUESTION TYPE")
    print("=" * 70)
    for qt, agg in by_type.items():
        print(f"\n  {qt}  (n={agg['n']})")
        for key, val in agg.items():
            if key == "n":
                continue
            print(f"    {key:<20} {val}")

    by_diff = aggregate(results, k_values, group_by="difficulty")
    print(f"\n{'=' * 70}")
    print("BY DIFFICULTY")
    print("=" * 70)
    for diff, agg in by_diff.items():
        print(f"\n  {diff}  (n={agg['n']})")
        for key, val in agg.items():
            if key == "n":
                continue
            print(f"    {key:<20} {val}")


def print_comparison(overall_by_retriever: dict[str, dict], k_values: list[int]) -> None:
    """Side-by-side overall metrics, one row per retriever."""
    max_k = max(k_values)
    columns: list[tuple[str, str]] = []
    for k in k_values:
        columns.append((f"R@{k}", f"recall@{k}"))
    columns.append(("MRR", "mrr"))
    columns.append((f"nDCG@{max_k}", f"ndcg@{max_k}"))
    for k in k_values:
        columns.append((f"EH@{k}", f"evidence_hit@{k}"))
    columns.append(("lat(ms)", "mean_latency_ms"))

    w = 11
    header = f"{'retriever':<10}" + "".join(f"{label:>{w}}" for label, _ in columns)
    print(f"\n{'=' * len(header)}")
    print("THREE-WAY COMPARISON (overall)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for name, agg in overall_by_retriever.items():
        row = f"{name:<10}" + "".join(f"{agg.get(key, ''):>{w}}" for _, key in columns)
        print(row)
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qa-path",
        type=Path,
        default=DEFAULT_QA_PATH,
        help=f"Path to qa_pairs.jsonl (default: {DEFAULT_QA_PATH})",
    )
    parser.add_argument(
        "--retriever",
        default="bm25",
        choices=["bm25", "dense", "hybrid", "all"],
        help="Retriever to evaluate (default: bm25). dense/hybrid/all require --strategy labeled.",
    )
    parser.add_argument(
        "--strategy",
        default="labeled",
        choices=["labeled", "strict", "fixed_size", "all"],
        help="Chunk-loading strategy (default: labeled = section_aware + hybrid_section_aware)",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[5, 10],
        help="k values for Recall@k and evidence_hit@k (default: 5 10)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save results JSON for later comparison",
    )
    args = parser.parse_args()

    print(f"Retriever:  {args.retriever}")
    print(f"Strategy:   {args.strategy}")
    print(f"QA path:    {args.qa_path}")
    print(f"k values:   {args.k}")
    print()

    # Load QA pairs (materialize so we can iterate once per retriever).
    dataset = QADataset(args.qa_path)
    dataset.load()
    pairs = list(dataset)
    print(f"Loaded {len(pairs)} QA pairs.\n")

    timestamp = datetime.now(UTC).isoformat()

    if args.retriever == "all":
        retrievers = build_all_retrievers(args.strategy)

        runs: dict[str, list[dict]] = {}
        overall_by_retriever: dict[str, dict] = {}
        for retriever in retrievers:
            print(f"\n\n{'#' * 80}")
            print(f"# RETRIEVER: {retriever.name}")
            print(f"{'#' * 80}")
            results = [evaluate_pair(pair, retriever, args.k) for pair in pairs]
            print_table(results, args.k)
            print_aggregates(results, args.k)
            runs[retriever.name] = results
            overall_by_retriever[retriever.name] = aggregate(results, args.k)

        print_comparison(overall_by_retriever, args.k)

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "mode": "compare",
                "retrievers": [r.name for r in retrievers],
                "strategy": args.strategy,
                "qa_path": str(args.qa_path),
                "k_values": args.k,
                "timestamp": timestamp,
                "n_pairs": len(pairs),
                "runs": {name: _run_payload(r, args.k) for name, r in runs.items()},
                "comparison_overall": overall_by_retriever,
            }
            args.output.write_text(json.dumps(payload, indent=2, default=str))
            print(f"\nResults saved to {args.output}")
        return 0

    # Single-retriever path (unchanged output shape).
    retriever = build_retriever(args.retriever, args.strategy)

    print("\nRunning evaluation...")
    results = [evaluate_pair(pair, retriever, args.k) for pair in pairs]

    print_table(results, args.k)
    print_aggregates(results, args.k)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_payload = {
            "retriever": args.retriever,
            "strategy": args.strategy,
            "qa_path": str(args.qa_path),
            "k_values": args.k,
            "timestamp": timestamp,
            "n_pairs": len(results),
            **_run_payload(results, args.k),
        }
        args.output.write_text(json.dumps(output_payload, indent=2, default=str))
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
