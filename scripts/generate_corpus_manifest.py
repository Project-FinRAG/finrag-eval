"""Generate data/metadata/corpus_manifest_v0.1.csv from corpus_stats.csv.

Adds a 'failure_mode' column to characterize each fixed-size and
hybrid_section_aware filing. For section_aware filings, failure_mode='none'.

The manifest is a stable per-filing reference for downstream code:
  - Vidhee's retriever uses it to filter on chunking_method or failure_mode
  - Harshmeet's QA work uses it to identify which filings have which Items
  - The EDA notebook references it for visualization
"""
from __future__ import annotations

import csv
from pathlib import Path

STATS_PATH = Path("data/processed/corpus_stats.csv")
MANIFEST_PATH = Path("data/metadata/corpus_manifest_v0.1.csv")

# Per-company failure mode classification, established empirically from
# the diagnostic work in ADR-0002. Companies not in this map are presumed
# to have section_aware filings; if any year for these companies is
# hybrid or fixed-size, the manifest will flag it without a known root cause.
FAILURE_MODE_BY_COMPANY = {
    # MD&A and Financial Statements content is incorporated by reference
    # from a separately-filed annual report; the 10-K itself contains stubs.
    "IBM": "incorporation_by_reference",
    "WFC": "incorporation_by_reference",
    # No regex-matchable Item headers; the filing uses a non-standard
    # visual hierarchy that doesn't follow the "Item N." prefix convention.
    "MS": "non_standard_format",
    "C": "non_standard_format",
    "INTC": "non_standard_format",
    # Real Item 7 header exists but parser matched a page-header repeat or
    # cross-reference instead. Section map is mostly usable.
    "MSFT": "parser_limitation_item7",
    "BAC": "parser_limitation_item7",
    # Item 8 (Financial Statements) or Item 15 (Exhibits) legitimately
    # dominates the document; for these the dominance gate was correctly
    # relaxed and the filing is section_aware. Listed here for visibility.
    "MET": "large_item8_legit",
    "PRU": "large_item8_legit",
    # Dominance is genuine parser failure — Item 15 swallows Item 8
    # content because intermediate boundaries are missed.
    "JPM": "dominant_section_parser_failure",
    "USB": "dominant_section_parser_failure",
}


def derive_failure_mode(ticker: str, method_used: str) -> str:
    """For fully section_aware filings outside the known-problem set,
    no failure mode applies. For everything else, look up by ticker."""
    if method_used == "section_aware" and ticker not in {"MET", "PRU"}:
        # MET and PRU pass section_aware via the relaxed dominance gate;
        # we still flag the underlying condition for visibility.
        return "none"
    return FAILURE_MODE_BY_COMPANY.get(ticker, "uncharacterized")


def derive_fiscal_year(accession: str) -> str:
    """Best-effort fiscal year from accession number's YY field.
    Accession format: NNNNNNNNNN-YY-NNNNNN. The YY is the filing year,
    typically one year after the fiscal year for 10-Ks. We report the
    filing year as a stable identifier — exact fiscal year requires
    parsing the document content."""
    parts = accession.split("-")
    if len(parts) >= 2 and len(parts[1]) == 2:
        yy = int(parts[1])
        # 22 → 2022, 26 → 2026 (assume 21st century)
        return str(2000 + yy)
    return "unknown"


def main():
    if not STATS_PATH.exists():
        raise SystemExit(f"FATAL: {STATS_PATH} not found. Run the ingestion pipeline first.")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    with STATS_PATH.open() as f:
        rows = list(csv.DictReader(f))

    # Order: section_aware first (alphabetical), then hybrid, then fixed_size
    tier_order = {"section_aware": 0, "hybrid_section_aware": 1, "fixed_size": 2}
    rows.sort(key=lambda r: (tier_order.get(r["method_used"], 9), r["ticker"], r["accession"]))

    manifest_rows = []
    for r in rows:
        ticker = r["ticker"]
        accession = r["accession"]
        method = r["method_used"]
        manifest_rows.append({
            "filing_id": f"{ticker}_{accession}",
            "ticker": ticker,
            "filing_type": r.get("filing_type", "10-K"),
            "accession_number": accession,
            "filing_year": derive_fiscal_year(accession),
            "chunking_method": method,
            "failure_mode": derive_failure_mode(ticker, method),
            "sections_detected": r.get("sections_detected", ""),
            "chunks_produced": r.get("chunks_produced", ""),
            "total_chars": r.get("total_chars", ""),
            "captured_ratio": r.get("captured_ratio", ""),
            "largest_section_item": r.get("largest_section_item", ""),
            "largest_section_ratio": r.get("largest_section_ratio", ""),
            "item_7_chars": r.get("item_7_chars", ""),
            "quality_check_passed": r.get("quality_check_passed", ""),
            "quality_check_reason": r.get("quality_check_reason", ""),
        })

    fieldnames = list(manifest_rows[0].keys())
    with MANIFEST_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Summary print
    from collections import Counter
    methods = Counter(r["chunking_method"] for r in manifest_rows)
    failures = Counter(r["failure_mode"] for r in manifest_rows)
    print(f"Wrote {MANIFEST_PATH} with {len(manifest_rows)} rows.\n")
    print("Chunking method distribution:")
    for m, n in sorted(methods.items()):
        print(f"  {m:25s} {n:4d}")
    print("\nFailure mode distribution:")
    for fm, n in sorted(failures.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {fm:35s} {n:4d}")


if __name__ == "__main__":
    main()