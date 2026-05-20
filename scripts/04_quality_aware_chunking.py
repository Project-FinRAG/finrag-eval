"""Day 4: Quality-aware chunking pipeline.

Combines Day 2's SGML extraction + Day 3's section detection with a
3-signal quality check that decides whether to use section-aware or
fixed-size chunking for each filing.

Quality check signals (filing trusted if ALL pass):
    1. Coverage: section content >= 65% of total text
    2. Dominance: no single section > 50% of total text
    3. MD&A sanity: Item 7 exists and is >= 5,000 chars

Filings passing all three -> section-aware chunks (labeled by Item)
Filings failing any check -> fixed-size chunks (labeled positionally)

Both paths produce chunks usable by retrieval. The eval framework
characterizes any performance difference between paths.

Run from repo root:
    uv run python scripts/04_quality_aware_chunking.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import os
import re
from dataclasses import dataclass
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

# ─── Config ─────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "BAC", "JPM", "GS", "WMT", "XOM"]
FILING_TYPE = "10-K"
NUM_FILINGS = 1
COMPANY_NAME = os.environ.get("SEC_COMPANY_NAME", "FinRAG-Eval Team")
EMAIL = os.environ.get("SEC_CONTACT_EMAIL")
if not EMAIL:
    raise SystemExit("SEC_CONTACT_EMAIL environment variable is required.")
DATA_DIR = Path("./data/raw")

CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 64

# Quality check thresholds
MIN_COVERAGE = 0.65
MAX_SECTION_DOMINANCE = 0.50
MIN_ITEM_7_CHARS = 5_000

# Standard 10-K Items
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


# ─── Data model ─────────────────────────────────────────────────────────────
@dataclass
class Chunk:
    """A retrievable chunk of text with provenance."""

    ticker: str
    chunk_id: str
    text: str
    section_label: str  # "Item 1A - Risk Factors" or "chunk_47" for fallback
    chunking_method: str  # "section_aware" or "fixed_size"
    token_count: int


# ─── Pipeline functions ────────────────────────────────────────────────────
def extract_primary_document(submission: str, doc_type: str) -> str:
    """Pull the <TEXT> block of the first <DOCUMENT> matching the <TYPE>."""
    for block in submission.split("<DOCUMENT>")[1:]:
        type_match = re.search(r"<TYPE>([^\n<]+)", block)
        if type_match and type_match.group(1).strip() == doc_type:
            text_match = re.search(r"<TEXT>\s*(.*?)\s*</TEXT>", block, re.DOTALL)
            if text_match:
                return text_match.group(1)
    raise ValueError(f"No <DOCUMENT> with <TYPE>{doc_type} found")


def html_to_text(html: str) -> str:
    """Strip HTML/iXBRL to clean plain text."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text


def find_sections(text: str) -> dict[str, tuple[int, int, str]]:
    """Detect Item sections; return {item_num: (start, end, title)}."""
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


def is_section_detection_reliable(
    sections: dict[str, tuple[int, int, str]],
    total_chars: int,
) -> tuple[bool, str]:
    """Run the 3-signal quality check.

    Returns (passes, reason) where reason explains the failure mode if any.
    """
    if total_chars == 0 or not sections:
        return False, "no text or no sections detected"

    # Signal 1: Coverage
    captured = sum(end - start for start, end, _ in sections.values())
    coverage = captured / total_chars
    if coverage < MIN_COVERAGE:
        return False, f"low coverage ({coverage:.1%} < {MIN_COVERAGE:.0%})"

    # Signal 2: Dominance
    largest = max(end - start for start, end, _ in sections.values())
    dominance = largest / total_chars
    if dominance > MAX_SECTION_DOMINANCE:
        return False, f"single section dominates ({dominance:.1%} > {MAX_SECTION_DOMINANCE:.0%})"

    # Signal 3: MD&A sanity
    item_7 = sections.get("7")
    if not item_7:
        return False, "Item 7 not detected"
    item_7_size = item_7[1] - item_7[0]
    if item_7_size < MIN_ITEM_7_CHARS:
        return False, f"Item 7 too small ({item_7_size:,} < {MIN_ITEM_7_CHARS:,} chars)"

    return True, "all signals pass"


def chunk_section_aware(
    text: str,
    sections: dict[str, tuple[int, int, str]],
    ticker: str,
    encoder: tiktoken.Encoding,
) -> list[Chunk]:
    """Section-aware chunking: split each section into token-bounded windows."""
    chunks: list[Chunk] = []
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
            chunks.append(
                Chunk(
                    ticker=ticker,
                    chunk_id=f"{ticker}_item{item_num}_{chunk_idx:04d}",
                    text=encoder.decode(window),
                    section_label=section_label,
                    chunking_method="section_aware",
                    token_count=len(window),
                )
            )
            chunk_idx += 1

    return chunks


def chunk_fixed_size(
    text: str,
    ticker: str,
    encoder: tiktoken.Encoding,
) -> list[Chunk]:
    """Fixed-size sliding window chunking with positional labels."""
    chunks: list[Chunk] = []
    tokens = encoder.encode(text)
    step = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS

    chunk_idx = 0
    for i in range(0, len(tokens), step):
        window = tokens[i : i + CHUNK_SIZE_TOKENS]
        if not window:
            break
        chunks.append(
            Chunk(
                ticker=ticker,
                chunk_id=f"{ticker}_fixed_{chunk_idx:04d}",
                text=encoder.decode(window),
                section_label=f"chunk_{chunk_idx}",
                chunking_method="fixed_size",
                token_count=len(window),
            )
        )
        chunk_idx += 1

    return chunks


def process_filing(
    ticker: str,
    encoder: tiktoken.Encoding,
) -> tuple[list[Chunk], dict[str, object]]:
    """Run the full pipeline on one filing. Returns (chunks, stats)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    Downloader(COMPANY_NAME, EMAIL, str(DATA_DIR)).get(
        FILING_TYPE, ticker, limit=NUM_FILINGS
    )

    filing_dir = DATA_DIR / "sec-edgar-filings" / ticker / FILING_TYPE
    submission_path = list(filing_dir.rglob("full-submission.txt"))[-1]
    raw = submission_path.read_text(encoding="utf-8", errors="ignore")

    filing_html = extract_primary_document(raw, FILING_TYPE)
    text = html_to_text(filing_html)
    sections = find_sections(text)

    reliable, reason = is_section_detection_reliable(sections, len(text))

    if reliable:
        chunks = chunk_section_aware(text, sections, ticker, encoder)
        method = "section_aware"
    else:
        chunks = chunk_fixed_size(text, ticker, encoder)
        method = "fixed_size"

    stats = {
        "ticker": ticker,
        "total_chars": len(text),
        "sections_detected": len(sections),
        "quality_check_passed": reliable,
        "quality_check_reason": reason,
        "method_used": method,
        "chunks_produced": len(chunks),
    }
    return chunks, stats


# ─── Run pipeline on all sanity-check tickers ───────────────────────────────
def main() -> None:
    encoder = tiktoken.get_encoding("cl100k_base")
    all_stats: list[dict[str, object]] = []

    for ticker in TICKERS:
        print(f"\n{'=' * 70}")
        print(f"Processing {ticker}...")
        print(f"{'=' * 70}")
        try:
            chunks, stats = process_filing(ticker, encoder)
            all_stats.append(stats)
            print(f"  Total text:       {stats['total_chars']:,} chars")
            print(f"  Sections found:   {stats['sections_detected']}")
            print(f"  Quality check:    {'PASS' if stats['quality_check_passed'] else 'FAIL'} ({stats['quality_check_reason']})")
            print(f"  Chunking method:  {stats['method_used']}")
            print(f"  Chunks produced:  {stats['chunks_produced']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            all_stats.append({"ticker": ticker, "error": str(e)})

    # ─── Summary table ──────────────────────────────────────────────────────
    print(f"\n\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    print(f"{'Ticker':<8} {'Chars':>10} {'Sections':>9} {'Quality':>9} {'Method':>15} {'Chunks':>8}")
    print(f"{'─' * 8} {'─' * 10} {'─' * 9} {'─' * 9} {'─' * 15} {'─' * 8}")

    section_aware_count = 0
    fixed_size_count = 0
    total_chunks = 0
    for s in all_stats:
        if "error" in s:
            print(f"{s['ticker']:<8} ERROR: {s['error']}")
            continue
        check = "PASS" if s["quality_check_passed"] else "FAIL"
        print(
            f"{s['ticker']:<8} {s['total_chars']:>10,} {s['sections_detected']:>9} "
            f"{check:>9} {s['method_used']:>15} {s['chunks_produced']:>8}"
        )
        if s["method_used"] == "section_aware":
            section_aware_count += 1
        else:
            fixed_size_count += 1
        total_chunks += s["chunks_produced"]

    total = section_aware_count + fixed_size_count
    if total > 0:
        print(f"\nSection-aware: {section_aware_count}/{total} ({section_aware_count / total:.0%})")
        print(f"Fixed-size:    {fixed_size_count}/{total} ({fixed_size_count / total:.0%})")
        print(f"Total chunks produced: {total_chunks:,}")


if __name__ == "__main__":
    main()