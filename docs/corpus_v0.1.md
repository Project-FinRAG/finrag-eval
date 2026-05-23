# FinRAG-Eval Corpus v0.1

**Frozen:** 2026-05-22
**Manifest:** `data/metadata/corpus_manifest_v0.1.csv`
**Methodology:** [ADR-0002: Quality-Aware Three-Tier Chunking](decisions/0002-quality-aware-chunking.md)

The frozen v0.1 corpus is the input to all retrieval and evaluation work for the remainder of the project. This document describes what's in it and how to consume it.

## At a glance

| Property | Value |
|---|---|
| Filings | 198 |
| Companies | 49 (tech + financial sectors) |
| Filing type | 10-K |
| Fiscal years covered | 2022–2026 (4 years per company) |
| Total chunks | 59,822 |
| Source | SEC EDGAR via `sec-edgar-downloader` |

## Chunking method distribution

| Tier | Filings | Share | Chunks have section labels? |
|---|---|---|---|
| `section_aware` | 163 | 82.3% | Yes — strict quality bar |
| `hybrid_section_aware` | 23 | 11.6% | Yes — usable but at least one strict gate failed |
| `fixed_size` | 12 | 6.1% | No — 512-token positional windows |
| **Total section-labeled** | **186** | **93.9%** | |

See ADR-0002 for tier definitions and quality gates.

## The 12 fixed-size filings

Three companies, four years each, three distinct structural reasons:

| Company | Years | Failure mode | Root cause |
|---|---|---|---|
| MS (Morgan Stanley) | 2023–2026 | `non_standard_format` | Filing uses a visual section hierarchy that does not match the SEC's "Item N." prefix convention. Zero sections detected by the regex parser. |
| C (Citigroup) | 2023–2026 | `non_standard_format` | Same as MS — non-standard section formatting. |
| INTC (Intel) | 2023–2026 | `non_standard_format` | Item index appears at the end of the document; the actual content is structured differently. Parser finds 23 tiny stub sections totaling <1% of the document. |

These 12 filings are excluded from section-specific QA pairs and the section-aware retrieval comparison. They remain available as fixed-size chunks for general retrieval and as a baseline for the experiment.

## The 23 hybrid_section_aware filings

These passed enough quality gates to produce useful section-labeled chunks but failed at least one strict check:

| Company | Years | Failure mode | What's still usable |
|---|---|---|---|
| MSFT | 4 | `parser_limitation_item7` | All Items except Item 7. MD&A header is a TOC false match (~1,967 chars instead of real content). |
| BAC | 3 | `parser_limitation_item7` | All Items except Item 7. Same TOC issue as MSFT. |
| IBM | 4 | `incorporation_by_reference` | Item 1 (Business) and Item 1A (Risk Factors) are substantial. Item 7 (MD&A) is a 212-char pointer because IBM files MD&A separately in the annual report. |
| WFC | 4 | `incorporation_by_reference` | Item 1 (Business) is 35-46K chars and substantial. Items 1A, 7, 8 are stubs (~250-316 chars each) because WFC also incorporates by reference. |
| JPM | 4 | `dominant_section_parser_failure` | Items 1, 1A are reliable. Item 15 (Exhibits) spans 84% of the document because the parser misses the Item 8 boundary and Item 15 swallows financial statement content. |
| USB | 4 | `dominant_section_parser_failure` | Same pattern as JPM. Item 1 is reliable; the dominant section reflects parser failure. |

**Guidance for QA pair authors:** Use these filings cautiously. The sections in the "What's still usable" column carry trustworthy labels. The named-failure section in each filing should be avoided unless the QA pair explicitly tests retrieval robustness.

## The 5 section_aware filings with relaxed dominance

These pass `section_aware` only because of the conditional dominance gate (Item 8/15 allowed up to 75% if Item 7 is healthy):

| Company | Years | Why flagged |
|---|---|---|
| MET (MetLife) | 4 | Insurer. Item 8 (Financial Statements) is 48-55% of the filing — legitimately large given the company's financial complexity. Item 7 is 224-298K chars, healthy. |
| PRU (Prudential) | 4 | Same pattern — asset manager with massive Item 8 sections. |

Tagged `large_item8_legit` in the manifest for visibility. These filings produce trustworthy section-aware chunks; the tag flags the underlying characteristic for any downstream code that wants to be explicit about which filings used the relaxed gate.

## How to use the manifest

The manifest at `data/metadata/corpus_manifest_v0.1.csv` is the per-filing reference. Columns:

| Column | Description |
|---|---|
| `filing_id` | Unique per-filing key: `{TICKER}_{ACCESSION}` |
| `ticker` | Company ticker |
| `filing_type` | `10-K` for v0.1 (no 10-Qs yet) |
| `accession_number` | SEC accession (also the JSONL filename suffix) |
| `filing_year` | Year of filing (derived from accession YY field) |
| `chunking_method` | `section_aware` / `hybrid_section_aware` / `fixed_size` |
| `failure_mode` | `none` for clean filings, otherwise descriptive label |
| `sections_detected` | Count of Items found by the parser |
| `chunks_produced` | Number of chunks in the JSONL for this filing |
| `total_chars` | Cleaned text length |
| `captured_ratio` | Fraction of document text inside detected sections |
| `largest_section_item` | Item number of the dominant section (e.g., `8`, `1A`) |
| `largest_section_ratio` | Largest section size / total chars |
| `item_7_chars` | Length of detected Item 7 (MD&A). 0 means undetected. |
| `quality_check_passed` | Boolean — did the filing pass strict reliability? |
| `quality_check_reason` | Free-text from the reliability function |

## Common filtering recipes

```python
import pandas as pd
manifest = pd.read_csv('data/metadata/corpus_manifest_v0.1.csv')

# All section-labeled filings (for retrieval)
indexed = manifest[manifest['chunking_method'].isin(['section_aware', 'hybrid_section_aware'])]
# Result: 186 filings

# Strict section-aware only (for robustness comparison)
strict = manifest[manifest['chunking_method'] == 'section_aware']
# Result: 163 filings

# Filings safe for Item 7 QA pairs (excludes MSFT, BAC, IBM, WFC, JPM, USB across all years)
item7_safe = manifest[manifest['item_7_chars'] >= 5000]
# Result: ~150 filings

# Filings safe for cross-section synthesis QA pairs
all_high_value_present = manifest[
    (manifest['chunking_method'] == 'section_aware') &
    (manifest['item_7_chars'] >= 5000)
]
```

## Reproducibility

The corpus is regenerated by `scripts/05_corpus_scale_out.py` from cached SEC filings in `data/raw/sec-edgar-filings/`. To rebuild from scratch:

```bash
# This will hit SEC EDGAR — set your contact email
export SEC_CONTACT_EMAIL=you@example.com

rm -rf data/processed/chunks data/processed/corpus_stats.csv
uv run python scripts/05_corpus_scale_out.py
uv run python scripts/generate_corpus_manifest.py
```

With an empty cache, this takes ~10-15 minutes (SEC EDGAR rate limits at ~10 req/sec).

## What v0.1 does *not* include

- 10-Q filings (only 10-Ks for now)
- Filings older than the most recent 4 fiscal years per company
- Companies outside tech and financial sectors
- Sub-filing artifacts (exhibits, schedules) beyond what's inside the 10-K body

These are out of scope for v0.1 and may or may not be added in a future v0.2.