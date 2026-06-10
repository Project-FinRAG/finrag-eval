"""Experiment: HyDE on top of the 3-large dense index.

For each question, gpt-4o-mini writes a short hypothetical 10-K-style passage that
would answer it (filing prose, "The Company..."), then we retrieve using THAT passage's
embedding instead of the raw question. This attacks the question<->filing phrasing gap.
Evaluated on the 70-pair set against the 3-large baseline (S@10 = 0.533). HyDE-only,
no ticker filter, to isolate its effect.

Usage:
    uv run python scripts/17_dense_large_hyde.py
"""

from __future__ import annotations

from pathlib import Path

from openai import OpenAI

from finrag_eval.eval.metrics import recall_at_k, soft_recall_at_k
from finrag_eval.eval.qa_dataset import QADataset
from finrag_eval.retrieval.dense import DenseRetriever, _load_api_key

QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")
LARGE_PATH = Path("data/indexes/chroma_dense_large_labeled")
LARGE_COLLECTION = "finrag_dense_large"
K_VALUES = [5, 10]
BASELINE_LARGE_S10 = 0.533  # 3-large dense, raw query, labeled, n=70


def generate_hyde(client: OpenAI, question: str) -> str:
    """Generate a hypothetical filing-style passage that would answer the question."""
    prompt = (
        "Write a short passage (2-4 sentences) stating the facts that answer the question "
        "below, and NAME the specific company the question is about. Write it as factual "
        "prose, like an excerpt from that company's annual report, not as a reply to a "
        "question.\n\n"
        f"Question: {question}"
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=160,
    )
    return (resp.choices[0].message.content or "").strip()


def main() -> None:
    retriever = DenseRetriever(
        embedding_model="text-embedding-3-large",
        index_path=LARGE_PATH,
        collection_name=LARGE_COLLECTION,
    )
    retriever.load()

    client = OpenAI(api_key=_load_api_key())

    qa = QADataset(QA_PATH)
    qa.load()
    print(f"Evaluating dense-3-large + HyDE on {len(qa)} QA pairs...\n")

    sums = {f"recall@{k}": 0.0 for k in K_VALUES}
    sums.update({f"soft_recall@{k}": 0.0 for k in K_VALUES})
    n = 0

    for qp in qa:
        hyde = generate_hyde(client, qp.question)
        results = retriever.retrieve(hyde, k=max(K_VALUES))
        retrieved_ids = [r.chunk.chunk_id for r in results]
        gold_ids = {ev.chunk_id for ev in qp.gold_evidence}
        for k in K_VALUES:
            sums[f"recall@{k}"] += recall_at_k(retrieved_ids, gold_ids, k)
            sums[f"soft_recall@{k}"] += soft_recall_at_k(results, qp.gold_evidence, k)
        n += 1

    print("=" * 60)
    print(f"DENSE 3-large + HyDE, labeled, n={n}")
    print("=" * 60)
    for metric, total in sums.items():
        print(f"  {metric:16s} {total / n:.3f}")
    print()
    s10 = sums["soft_recall@10"] / n
    print(f"HyDE soft_recall@10:        {s10:.3f}")
    print(f"raw-query (3-large) S@10:   {BASELINE_LARGE_S10:.3f}  (baseline)")
    print(f"delta:                      {s10 - BASELINE_LARGE_S10:+.3f}")


if __name__ == "__main__":
    main()