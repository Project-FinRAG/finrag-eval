"""Day 5: Scale the quality-aware chunking pipeline to the full corpus.

Iterates over 50 companies × 2 years × 2 filing types = ~200 filings.
Persists chunks + per-filing stats to disk so subsequent stages
(EDA, indexing, retrieval) don't need to re-fetch.

Run from repo root:
    uv run python scripts/05_corpus_scale_out.py

Output:
    data/processed/chunks/{TICKER}_{FILING_TYPE}_{ACCESSION}.jsonl
    data/processed/corpus_stats.csv
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

# ─── Config ─────────────────────────────────────────────────────────────────
TICKERS = [
    # Tech (25)
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "ORCL", "ADBE", "CRM",
    "AVGO", "CSCO", "INTC", "AMD", "QCOM", "IBM", "TXN", "NOW", "INTU",
    "PANW", "SNPS", "CDNS", "ANET", "KLAC", "MU", "AMAT",
    # Financial (25)
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "USB",
    "PNC", "COF", "TFC", "WTW", "AON", "CB", "MET", "PRU", "AIG", "ALL",
    "TRV", "BX", "KKR", "APO", "CME",
]

FILING_TYPES = ["10-K"]   # Start with 10-K only; add 10-Q after this works
YEARS_TO_FETCH = 2        # Most recent 2 years per filing type

COMPANY_NAME = os.environ.get("SEC_COMPANY_NAME", "FinRAG-Eval Team")
EMAIL = os.environ.get("SEC_CONTACT_EMAIL")
if not EMAIL:
    raise SystemExit("SEC_CONTACT_EMAIL environment variable is required.")

RAW_DIR = Path("./data/raw")
PROCESSED_DIR = Path("./data/processed")
CHUNKS_DIR = PROCESSED_DIR / "chunks"
STATS_PATH = PROCESSED_DIR / "corpus_stats.csv"

CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 64

MIN_COVERAGE = 0.65
MAX_SECTION_DOMINANCE = 0.50
MIN_ITEM_7_CHARS = 5_000

STANDARD_ITEMS = {
    "1": "Business", "1A": "Risk Factors", "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity", "2": "Properties", "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures", "5": "Market for Registrant's Common Equity",
    "6": "[Reserved]", "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures", "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accounting Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

ITEM_HEADER_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[a-c]?)\b\.?\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass
class Chunk:
    ticker: str
    filing_type: str
    accession: str
    chunk_id: str
    text: str
    section_label: str
    chunking_method: str
    token_count: int


# ─── Pipeline (lifted from Day 4) ───────────────────────────────────────────
def extract_primary_document(submission: str, doc_type: str) -> str:
    for block in submission.split("<DOCUMENT>")[1:]:
        type_match = re.search(r"<TYPE>([^\n<]+)", block)
        if type_match and type_match.group(1).strip() == doc_type:
            text_match = re.search(r"<TEXT>\s*(.*?)\s*</TEXT>", block, re.DOTALL)
            if text_match:
                return text_match.group(1)
    raise ValueError(f"No <DOCUMENT> with <TYPE>{doc_type} found")


def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def find_sections(text: str) -> dict[str, tuple[int, int, str]]:
    last_match: dict[str, re.Match[str]] = {}
    for m in ITEM_HEADER_RE.finditer(text):
        last_match[m.group(1).upper()] = m
    sorted_items = sorted(last_match.items(), key=lambda x: x[1].start())
    sections: dict[str, tuple[int, int, str]] = {}
    for i, (item_num, match) in enumerate(sorted_items):
        start = match.start()
        end = sorted_items[i + 1][1].start() if i + 1 < len(sorted_items) else len(text)
        title = match.group(2).strip() or STANDARD_ITEMS.get(item_num, "Unknown")
        sections[item_num] = (start, end, title)
    return sections


def is_section_detection_reliable(sections, total_chars: int) -> tuple[bool, str]:
    if total_chars == 0 or not sections:
        return False, "no sections detected"
    captured = sum(end - start for start, end, _ in sections.values())
    if captured / total_chars < MIN_COVERAGE:
        return False, f"low coverage ({captured / total_chars:.1%})"
    largest = max(end - start for start, end, _ in sections.values())
    if largest / total_chars > MAX_SECTION_DOMINANCE:
        return False, f"section dominance ({largest / total_chars:.1%})"
    item_7 = sections.get("7")
    if not item_7 or (item_7[1] - item_7[0]) < MIN_ITEM_7_CHARS:
        size = (item_7[1] - item_7[0]) if item_7 else 0
        return False, f"Item 7 too small ({size:,} chars)"
    return True, "all signals pass"


def chunk_section_aware(text, sections, ticker, filing_type, accession, encoder):
    chunks = []
    chunk_idx = 0
    step = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS
    for item_num in sorted(sections.keys(), key=lambda x: sections[x][0]):
        start, end, _ = sections[item_num]
        section_text = text[start:end]
        section_label = f"Item {item_num} - {STANDARD_ITEMS.get(item_num, 'Unknown')}"
        tokens = encoder.encode(section_text)
        for i in range(0, len(tokens), step):
            window = tokens[i : i + CHUNK_SIZE_TOKENS]
            if not window:
                break
            chunks.append(Chunk(
                ticker=ticker, filing_type=filing_type, accession=accession,
                chunk_id=f"{ticker}_{accession}_item{item_num}_{chunk_idx:04d}",
                text=encoder.decode(window), section_label=section_label,
                chunking_method="section_aware", token_count=len(window),
            ))
            chunk_idx += 1
    return chunks


def chunk_fixed_size(text, ticker, filing_type, accession, encoder):
    chunks = []
    tokens = encoder.encode(text)
    step = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS
    for i, idx in enumerate(range(0, len(tokens), step)):
        window = tokens[idx : idx + CHUNK_SIZE_TOKENS]
        if not window:
            break
        chunks.append(Chunk(
            ticker=ticker, filing_type=filing_type, accession=accession,
            chunk_id=f"{ticker}_{accession}_fixed_{i:04d}",
            text=encoder.decode(window), section_label=f"chunk_{i}",
            chunking_method="fixed_size", token_count=len(window),
        ))
    return chunks


def process_filing(ticker, filing_type, encoder, dl):
    """Returns list of (chunks, stats_row) for each filing of this type/ticker."""
    try:
        dl.get(filing_type, ticker, limit=YEARS_TO_FETCH)
    except Exception as e:
        return [], [{"ticker": ticker, "filing_type": filing_type, "error": str(e)}]

    filing_dir = RAW_DIR / "sec-edgar-filings" / ticker / filing_type
    if not filing_dir.exists():
        return [], [{"ticker": ticker, "filing_type": filing_type, "error": "no filings downloaded"}]

    results = []
    for accession_dir in sorted(filing_dir.iterdir()):
        if not accession_dir.is_dir():
            continue
        accession = accession_dir.name
        submission_files = list(accession_dir.glob("full-submission.txt"))
        if not submission_files:
            continue

        try:
            raw = submission_files[0].read_text(encoding="utf-8", errors="ignore")
            filing_html = extract_primary_document(raw, filing_type)
            text = html_to_text(filing_html)
            sections = find_sections(text)
            reliable, reason = is_section_detection_reliable(sections, len(text))

            if reliable:
                chunks = chunk_section_aware(text, sections, ticker, filing_type, accession, encoder)
                method = "section_aware"
            else:
                chunks = chunk_fixed_size(text, ticker, filing_type, accession, encoder)
                method = "fixed_size"

            stats = {
                "ticker": ticker,
                "filing_type": filing_type,
                "accession": accession,
                "total_chars": len(text),
                "sections_detected": len(sections),
                "quality_check_passed": reliable,
                "quality_check_reason": reason,
                "method_used": method,
                "chunks_produced": len(chunks),
                "error": "",
            }
            results.append((chunks, stats))
        except Exception as e:
            results.append(([], {"ticker": ticker, "filing_type": filing_type,
                                  "accession": accession, "error": str(e)}))

    return [(c, s) for c, s in results], None


def save_chunks(chunks, output_path):
    """Persist chunks as JSONL — one chunk per line."""
    with output_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def main():
    encoder = tiktoken.get_encoding("cl100k_base")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader(COMPANY_NAME, EMAIL, str(RAW_DIR))

    all_stats = []
    start_time = time.time()
    total_filings = len(TICKERS) * len(FILING_TYPES) * YEARS_TO_FETCH

    print(f"Scaling to {len(TICKERS)} companies × {len(FILING_TYPES)} filing types × "
          f"{YEARS_TO_FETCH} years = up to {total_filings} filings\n")

    filings_processed = 0
    for ticker_idx, ticker in enumerate(TICKERS, 1):
        for filing_type in FILING_TYPES:
            print(f"[{ticker_idx:2d}/{len(TICKERS)}] {ticker} {filing_type}...", end=" ", flush=True)
            try:
                results, error = process_filing(ticker, filing_type, encoder, dl)
                if error:
                    print(f"ERROR: {error[0].get('error', 'unknown')}")
                    all_stats.append(error[0])
                    continue
                if not results:
                    print("no filings found")
                    continue

                for chunks, stats in results:
                    if chunks:
                        output_path = CHUNKS_DIR / f"{ticker}_{filing_type}_{stats['accession']}.jsonl"
                        save_chunks(chunks, output_path)
                    all_stats.append(stats)
                    filings_processed += 1

                methods = [s["method_used"] for s in [r[1] for r in results] if "method_used" in s]
                chunk_counts = [s["chunks_produced"] for s in [r[1] for r in results] if "chunks_produced" in s]
                print(f"{len(results)} filings, {sum(chunk_counts)} chunks "
                      f"({', '.join(set(methods))})")
            except Exception as e:
                print(f"FATAL: {e}")
                all_stats.append({"ticker": ticker, "filing_type": filing_type, "error": str(e)})

    # Write stats CSV
    if all_stats:
        keys = ["ticker", "filing_type", "accession", "total_chars", "sections_detected",
                "quality_check_passed", "quality_check_reason", "method_used",
                "chunks_produced", "error"]
        with STATS_PATH.open("w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for s in all_stats:
                row = [str(s.get(k, "")).replace(",", ";") for k in keys]
                f.write(",".join(row) + "\n")

    elapsed = time.time() - start_time

    # ─── Summary ────────────────────────────────────────────────────────────
    successful = [s for s in all_stats if s.get("chunks_produced", 0) > 0]
    failed = [s for s in all_stats if s.get("error")]
    section_aware = [s for s in successful if s.get("method_used") == "section_aware"]
    fixed_size = [s for s in successful if s.get("method_used") == "fixed_size"]
    total_chunks = sum(s.get("chunks_produced", 0) for s in successful)

    print(f"\n{'=' * 70}")
    print("CORPUS SCALE-OUT SUMMARY")
    print(f"{'=' * 70}")
    print(f"Elapsed:              {elapsed / 60:.1f} minutes")
    print(f"Filings processed:    {len(successful)} successful, {len(failed)} failed")
    print(f"Section-aware:        {len(section_aware)}/{len(successful)} "
          f"({len(section_aware) / max(len(successful), 1):.0%})")
    print(f"Fixed-size:           {len(fixed_size)}/{len(successful)} "
          f"({len(fixed_size) / max(len(successful), 1):.0%})")
    print(f"Total chunks:         {total_chunks:,}")
    print(f"Stats written to:     {STATS_PATH}")
    print(f"Chunks written to:    {CHUNKS_DIR}/  ({len(list(CHUNKS_DIR.glob('*.jsonl')))} files)")

    if failed:
        print(f"\nFailed filings ({len(failed)}):")
        for f in failed:
            print(f"  {f['ticker']} {f.get('filing_type', '?')}: {f['error']}")


if __name__ == "__main__":
    main()