"""Experiment: ticker pre-filtering on top of the 3-large dense index.

For each question, gpt-4o-mini extracts the single company it's about (from the
QUESTION text only — never the gold). If exactly one corpus company is identified,
the dense search is restricted to that ticker; otherwise it falls back to unfiltered
search (protecting cross-company questions). Evaluated on the 70-pair set, stacked on
the 3-large index. Compare to 3-large unfiltered: S@10 = 0.533.

Usage:
    uv run python scripts/16_dense_large_filtered.py
"""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from finrag_eval.eval.metrics import recall_at_k, soft_recall_at_k
from finrag_eval.eval.qa_dataset import QADataset
from finrag_eval.retrieval.bm25 import load_chunks_from_jsonl
from finrag_eval.retrieval.dense import DenseRetriever, _load_api_key

QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")
LARGE_PATH = Path("data/indexes/chroma_dense_large_labeled")
LARGE_COLLECTION = "finrag_dense_large"
K_VALUES = [5, 10]
BASELINE_LARGE_S10 = 0.533  # 3-large dense, unfiltered, labeled, n=70


def extract_ticker(client: OpenAI, question: str, valid: set[str]) -> str | None:
    """Ask gpt-4o-mini which single corpus company the question is about."""
    prompt = (
        f"Valid tickers: {', '.join(sorted(valid))}.\n"
        "Which SINGLE company is the question about? Reply with exactly one ticker "
        "from the list, or NONE if it concerns multiple companies or none of them.\n\n"
        f"Question: {question}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=8,
    )
    answer = (resp.choices[0].message.content or "").strip().upper()
    return answer if answer in valid else None


def main() -> None:
    valid_tickers = {c.ticker for c in load_chunks_from_jsonl("labeled")}
    print(f"{len(valid_tickers)} companies in corpus")

    retriever = DenseRetriever(
        embedding_model="text-embedding-3-large",
        index_path=LARGE_PATH,
        collection_name=LARGE_COLLECTION,
    )
    retriever.load()

    client = OpenAI(api_key=_load_api_key())

    qa = QADataset(QA_PATH)
    qa.load()
    print(f"Evaluating dense-3-large + ticker filter on {len(qa)} QA pairs...\n")

    sums = {f"recall@{k}": 0.0 for k in K_VALUES}
    sums.update({f"soft_recall@{k}": 0.0 for k in K_VALUES})
    n = 0
    n_filtered = 0

    for qp in qa:
        ticker = extract_ticker(client, qp.question, valid_tickers)
        where = {"ticker": ticker} if ticker else None
        if where:
            n_filtered += 1
        results = retriever.retrieve(qp.question, k=max(K_VALUES), where=where)
        retrieved_ids = [r.chunk.chunk_id for r in results]
        gold_ids = {ev.chunk_id for ev in qp.gold_evidence}
        for k in K_VALUES:
            sums[f"recall@{k}"] += recall_at_k(retrieved_ids, gold_ids, k)
            sums[f"soft_recall@{k}"] += soft_recall_at_k(results, qp.gold_evidence, k)
        n += 1

    print("=" * 60)
    print(f"DENSE 3-large + ticker pre-filter, labeled, n={n}")
    print(f"  routed to a company: {n_filtered}/{n}")
    print("=" * 60)
    for metric, total in sums.items():
        print(f"  {metric:16s} {total / n:.3f}")
    print()
    s10 = sums["soft_recall@10"] / n
    print(f"filtered soft_recall@10:    {s10:.3f}")
    print(f"unfiltered (3-large) S@10:  {BASELINE_LARGE_S10:.3f}  (baseline)")
    print(f"delta:                      {s10 - BASELINE_LARGE_S10:+.3f}")

if __name__ == "__main__":
    main()