"""Experiment: does text-embedding-3-large beat 3-small for dense retrieval?

Builds a 3-large dense index over the labeled corpus (first run only, ~$4 / ~45 min),
then evaluates dense retrieval on the 70-pair QA set. Compares soft_recall@10 to the
3-small dense baseline (labeled, n=70): 0.371.

Usage:
    uv run python scripts/15_dense_large.py
"""

from __future__ import annotations

from pathlib import Path

from finrag_eval.eval.metrics import recall_at_k, soft_recall_at_k
from finrag_eval.eval.qa_dataset import QADataset
from finrag_eval.retrieval.bm25 import load_chunks_from_jsonl
from finrag_eval.retrieval.dense import DenseRetriever

QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")
LARGE_PATH = Path("data/indexes/chroma_dense_large_labeled")
LARGE_COLLECTION = "finrag_dense_large"
K_VALUES = [5, 10]
BASELINE_SMALL_S10 = 0.371  # 3-small dense, labeled, n=70


def main() -> None:
    retriever = DenseRetriever(
        embedding_model="text-embedding-3-large",
        index_path=LARGE_PATH,
        collection_name=LARGE_COLLECTION,
    )

    if LARGE_PATH.exists():
        print(f"Loading existing 3-large index from {LARGE_PATH}")
        retriever.load()
    else:
        print("Building 3-large dense index over labeled corpus (first run, ~$4 / ~45 min)...")
        chunks = load_chunks_from_jsonl("labeled")
        print(f"  Loaded {len(chunks):,} labeled chunks")
        retriever.index(chunks)

    qa = QADataset(QA_PATH)
    qa.load()
    print(f"\nEvaluating dense (3-large) on {len(qa)} QA pairs...\n")

    sums = {f"recall@{k}": 0.0 for k in K_VALUES}
    sums.update({f"soft_recall@{k}": 0.0 for k in K_VALUES})
    n = 0

    for qp in qa:
        gold_ids = {ev.chunk_id for ev in qp.gold_evidence}
        results = retriever.retrieve(qp.question, k=max(K_VALUES))
        retrieved_ids = [r.chunk.chunk_id for r in results]
        for k in K_VALUES:
            sums[f"recall@{k}"] += recall_at_k(retrieved_ids, gold_ids, k)
            sums[f"soft_recall@{k}"] += soft_recall_at_k(results, qp.gold_evidence, k)
        n += 1

    print("=" * 60)
    print(f"DENSE (text-embedding-3-large), labeled, n={n}")
    print("=" * 60)
    for metric, total in sums.items():
        print(f"  {metric:16s} {total / n:.3f}")
    print()
    large_s10 = sums["soft_recall@10"] / n
    print(f"3-large soft_recall@10:  {large_s10:.3f}")
    print(f"3-small soft_recall@10:  {BASELINE_SMALL_S10:.3f}  (baseline)")
    print(f"delta:                   {large_s10 - BASELINE_SMALL_S10:+.3f}")

if __name__ == "__main__":
    main()