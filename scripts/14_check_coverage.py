"""Coverage audit: how much of the gold evidence is even findable in the corpus?

Separates "retrieval is weak" from "the corpus/QA caps recall". For every gold
citation, checks three levels against the indexed labeled corpus:

  1. filing indexed  - is the gold filing in the corpus at all?
  2. chunk present   - does the exact gold chunk_id exist?  (id-recall ceiling)
  3. quote findable  - does the gold quote text appear in that filing's chunks?
                       (soft-recall ceiling; quote-less golds fall back to #2,
                       mirroring soft_recall_at_k)

Headline = the soft-recall ceiling: the max mean soft-recall any retriever could
achieve here. Compare to the reported soft_recall@10 to see how much of the gap
is fixable retrieval vs a hard coverage cap.

Usage:
    uv run python scripts/14_check_coverage.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from finrag_eval.common.text import normalize_for_search
from finrag_eval.eval.qa_dataset import QADataset
from finrag_eval.retrieval.bm25 import load_chunks_from_jsonl

QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")
REPORTED_BEST_SOFT_AT_10 = 0.50  # labeled hybrid, n=20


def main() -> None:
    chunks = load_chunks_from_jsonl("labeled")
    print(f"Loaded {len(chunks):,} labeled corpus chunks")

    corpus_chunk_ids = {c.chunk_id for c in chunks}
    by_filing: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        by_filing[c.filing_accession].append(c.text)
    filing_text = {
        acc: normalize_for_search("\n".join(texts)) for acc, texts in by_filing.items()
    }
    print(f"  {len(filing_text)} distinct filings indexed\n")

    qa = QADataset(QA_PATH)
    qa.load()
    print(f"Loaded {len(qa)} QA pairs\n")

    total_gold = 0
    n_filing_ok = 0
    n_chunk_ok = 0
    n_quote_ok = 0
    per_qa_soft_ceiling: list[float] = []
    problems: list[str] = []

    for qp in qa:
        golds = qp.gold_evidence
        if not golds:
            per_qa_soft_ceiling.append(0.0)
            problems.append(f"{qp.qa_id}: NO gold_evidence")
            continue

        hits = 0
        for cit in golds:
            total_gold += 1
            filing_ok = cit.filing_accession in filing_text
            chunk_ok = cit.chunk_id in corpus_chunk_ids
            if cit.quote:
                quote_ok = normalize_for_search(cit.quote) in filing_text.get(
                    cit.filing_accession, ""
                )
            else:
                quote_ok = chunk_ok

            n_filing_ok += filing_ok
            n_chunk_ok += chunk_ok
            n_quote_ok += quote_ok
            hits += quote_ok

            if not filing_ok:
                problems.append(
                    f"{qp.qa_id} [{cit.ticker} {cit.filing_accession}]: FILING NOT INDEXED"
                )
            elif not quote_ok:
                problems.append(
                    f"{qp.qa_id} [{cit.ticker} {cit.filing_accession}]: "
                    f"quote not found in that filing's chunks"
                )
            elif not chunk_ok:
                problems.append(
                    f"{qp.qa_id} [{cit.ticker}]: chunk_id {cit.chunk_id} absent "
                    f"(soft-match can hit, id-recall cannot)"
                )

        per_qa_soft_ceiling.append(hits / len(golds))

    n_qa = len(qa)
    if total_gold == 0:
        print("No gold citations found — nothing to audit.")
        return

    mean_ceiling = sum(per_qa_soft_ceiling) / n_qa
    fully = sum(1 for v in per_qa_soft_ceiling if v == 1.0)
    zero = sum(1 for v in per_qa_soft_ceiling if v == 0.0)

    print("=" * 70)
    print("COVERAGE AUDIT  (labeled corpus)")
    print("=" * 70)
    print(f"Gold citations:                {total_gold}")
    print(f"  filing indexed:              {n_filing_ok}/{total_gold}  ({n_filing_ok / total_gold:.1%})")
    print(f"  chunk_id present (id ceil):  {n_chunk_ok}/{total_gold}  ({n_chunk_ok / total_gold:.1%})")
    print(f"  quote findable (soft ceil):  {n_quote_ok}/{total_gold}  ({n_quote_ok / total_gold:.1%})")
    print()
    print(f"SOFT-RECALL CEILING (max achievable mean soft-recall): {mean_ceiling:.3f}")
    print(
        f"  reported best (labeled hybrid soft@10) = {REPORTED_BEST_SOFT_AT_10:.2f}"
        f"  ->  fixable headroom {mean_ceiling - REPORTED_BEST_SOFT_AT_10:+.3f}"
    )
    print()
    print(f"QA fully covered:              {fully}/{n_qa}")
    print(f"QA with zero findable gold:    {zero}/{n_qa}")

    if problems:
        print("\n" + "=" * 70)
        print(f"ISSUES ({len(problems)})")
        print("=" * 70)
        for p in problems:
            print(f"  - {p}")
    else:
        print("\nClean — every gold citation's quote is findable in its filing.")


if __name__ == "__main__":
    main()








