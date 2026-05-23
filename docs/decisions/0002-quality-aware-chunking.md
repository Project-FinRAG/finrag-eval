# 0002 - Quality-Aware Three-Tier Chunking

**Status:** Accepted
**Date:** 2026-05-22
**Deciders:** FinRAG-Eval Team

## Context

SEC 10-K filings are heterogeneous. The "standard" Item-based structure (Item 1 Business, Item 1A Risk Factors, ..., Item 16 Form 10-K Summary) is followed by most filers, but a non-trivial minority deviate in ways that defeat naive section-header detection.

Our first pass at corpus ingestion used a binary classifier: if section detection passed all quality gates, emit section-aware chunks (chunks labeled with the Item they belong to); otherwise, fall back to fixed-size 512-token chunks with no semantic labels. This produced an 80/20 split on a 100-filing corpus, with 20% of filings receiving no section metadata at all.

Empirical investigation of the 20% revealed multiple distinct failure modes, several of which produce *partially usable* section maps — for example, 22 of 23 expected Items detected correctly but Item 7 (MD&A) is too small because the parser matched a table-of-contents entry instead of the real header. Discarding section structure entirely for these filings throws away usable information.

Additionally, the strict dominance gate (rejecting any filing where one section is >50% of the document) incorrectly rejected legitimate filings from large financial firms whose Item 8 (Financial Statements) genuinely is more than half the filing's length.

## Decision

Introduce a three-tier classification for filing-level chunking method:

| Tier | Quality bar | What it means |
|---|---|---|
| `section_aware` | All strict gates pass: ≥10 sections detected, coverage ≥80%, no Item exceeds 50% of doc (relaxed conditionally — see below), Item 7 ≥5,000 chars | Section boundaries are reliable; chunks carry trustworthy Item labels |
| `hybrid_section_aware` | At least one strict gate failed, but section map is usable: ≥10 sections detected, coverage ≥50%, at least one high-value Item (1, 1A, 7, 8) has ≥5,000 chars of content | Section labels exist but should be treated as lower-confidence. Downstream consumers can opt in or out. |
| `fixed_size` | Section detection unusable: fewer than 10 sections, insufficient coverage, or all high-value Items are stubs | Fall back to 512-token windows with positional labels (`chunk_0`, `chunk_1`, ...) and no Item metadata |

The tier is recorded in two places:

- **Filing-level**: `method_used` column in `data/processed/corpus_stats.csv`
- **Chunk-level**: `chunking_method` field in `data/processed/chunks/*.jsonl`

Both must agree per filing, allowing downstream code to filter at either granularity.

The dominance gate is conditionally relaxed: a filing may exceed the 50% dominance threshold if (a) the dominant Item is 8 or 15, (b) the ratio is ≤75%, (c) Item 7 is healthy, and (d) at least 12 sections were detected. This recovers legitimate large financial-statement filings from insurers and asset managers without admitting filings where the dominance reflects parser failure.

## Consequences

**Good:**
- Corpus shifts from 80/20 binary to 82.3% / 11.6% / 6.1% across the three tiers. 93.9% of filings now produce section-labeled chunks (up from 80%).
- The 6.1% remaining fallbacks are characterized by company and root cause rather than a single opaque category. The 12 filings are 3 companies × 4 years each (MS, C, INTC) with three distinct structural causes documented in `docs/corpus_v0.1.md`.
- Vidhee's retrieval comparison can run on the full section-labeled subset (186 filings) for maximum statistical power, or restricted to strict section-aware (163) as a robustness check.
- Harshmeet's QA pair work has clear guidance per tier: prefer `section_aware` for first ~50 pairs; use `hybrid_section_aware` thoughtfully (verify against the diagnostic columns); avoid `fixed_size` for section-specific questions.
- The 4-failure-mode taxonomy (incorporation by reference, non-standard header format, parser limitation, exhibit dominance) is itself a methodology contribution.

**Trade-off:**
- Downstream code is slightly more complex — must filter on `chunking_method` rather than assuming all chunks are equal quality.
- The "hybrid" tier is a confidence judgment, not a binary truth. A pair written against a `hybrid_section_aware` filing may cite a section that's labeled correctly *and* a section that isn't, in the same filing. Harshmeet's verification protocol catches this, but the failure mode exists.

## What we tried and rejected

Two parser-level fixes were attempted and rejected before this design:

1. **Heuristic section-scoring patch** — modify `find_sections` to score candidate matches (e.g., penalize matches inside TOC regions). Tried twice; both attempts broke previously-working control filings (AAPL regressed). Conclusion: any global change to matching logic has side effects on the 80% that already works.

2. **edgartools library spike** — evaluate replacing the in-house parser with the `edgartools` library. Tested on the 5 hardest fallback filings: recovered MSFT and BAC cleanly but failed on JPM (Item 7 = 396 chars), MS (Item 1A swallowed multiple sections), and C (returned garbage). Net result was 2 wins, 2 silent failures (wrongly-labeled chunks worse than honest fallback), 1 ambiguous. Conclusion: replacing the parser exchanges one set of failure modes for another, without net improvement.

The three-tier classification sidesteps the parser problem entirely: instead of trying to make every filing pass strict gates, we acknowledge the heterogeneity and route each filing to the most useful chunking strategy it can support.

## References

- Failure-mode taxonomy and per-filing classification: `data/metadata/corpus_manifest_v0.1.csv`
- Corpus characterization with example diagnostics: `docs/corpus_v0.1.md`
- Verification: `scripts/05_corpus_scale_out.py` (the three tiers), `scripts/verify_qa_pairs.py` (downstream contract)
- Spike script (kept for methodology evidence): `scripts/spike_edgartools.py`