"""Day 3: Detect standard 10-K Item sections in the extracted primary document.

Extends Day 2's SGML extraction with regex-based section tagging. The standard
10-K Item structure (Item 1, 1A, 7, 7A, 8, ...) is mandated by SEC Regulation
S-K, so detection is reliable across companies and years.

This produces the structured input the SectionAwareChunker needs.

Run from repo root:
    uv run python scripts/03_section_tagging.py
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
import os
import re
from pathlib import Path

from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

# ─── Config ─────────────────────────────────────────────────────────────────
TICKER = "AAPL"
FILING_TYPE = "10-K"
NUM_FILINGS = 1
COMPANY_NAME = os.environ.get("SEC_COMPANY_NAME", "FinRAG-Eval Team")
EMAIL = os.environ.get("SEC_CONTACT_EMAIL")
if not EMAIL:
    raise SystemExit(
        "SEC_CONTACT_EMAIL environment variable is required. "
        "See .env.example."
    )
DATA_DIR = Path("./data/raw")

# Standard 10-K Items per SEC Regulation S-K
STANDARD_ITEMS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "1C": "Cybersecurity",
    "2": "Properties",
    "3": "Legal Proceedings",
    "4": "Mine Safety Disclosures",
    "5": "Market for Registrant's Common Equity",
    "6": "[Reserved]",
    "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements",
    "9": "Changes in and Disagreements With Accountants",
    "9A": "Controls and Procedures",
    "9B": "Other Information",
    "9C": "Disclosure Regarding Foreign Jurisdictions",
    "10": "Directors, Executive Officers and Corporate Governance",
    "11": "Executive Compensation",
    "12": "Security Ownership of Certain Beneficial Owners",
    "13": "Certain Relationships and Related Transactions",
    "14": "Principal Accounting Fees and Services",
    "15": "Exhibits, Financial Statement Schedules",
    "16": "Form 10-K Summary",
}

# Match "Item N", "ITEM N.", "Item NA.", at line start with optional title
ITEM_HEADER_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[a-c]?)\b\.?\s*(.*?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def extract_primary_document(submission: str, doc_type: str) -> str:
    """Pull the <TEXT> block of the first <DOCUMENT> matching the <TYPE>."""
    for block in submission.split("<DOCUMENT>")[1:]:
        type_match = re.search(r"<TYPE>([^\n<]+)", block)
        if type_match and type_match.group(1).strip() == doc_type:
            text_match = re.search(r"<TEXT>\s*(.*?)\s*</TEXT>", block, re.DOTALL)
            if text_match:
                return text_match.group(1)
    raise ValueError(f"No <DOCUMENT> with <TYPE>{doc_type} found")


def find_sections(text: str) -> dict[str, tuple[int, int, str]]:
    """Detect standard Item sections.

    Returns dict mapping item_number -> (start_char, end_char, detected_title).

    Each Item appears at least twice (TOC + actual section). We take the LAST
    occurrence of each as the section start, which always lands on the real
    section rather than the TOC entry.
    """
    last_match: dict[str, re.Match[str]] = {}
    for m in ITEM_HEADER_RE.finditer(text):
        last_match[m.group(1).upper()] = m  # overwrites; keeps last

    sorted_items = sorted(last_match.items(), key=lambda x: x[1].start())

    sections: dict[str, tuple[int, int, str]] = {}
    for i, (item_num, match) in enumerate(sorted_items):
        start = match.start()
        end = sorted_items[i + 1][1].start() if i + 1 < len(sorted_items) else len(text)
        title = match.group(2).strip() or STANDARD_ITEMS.get(item_num, "Unknown")
        sections[item_num] = (start, end, title)
    return sections


def first_sentence(section_text: str, max_len: int = 80) -> str:
    """Extract a clean first sentence preview."""
    lines = [ln.strip() for ln in section_text.split("\n") if ln.strip()]
    if len(lines) < 2:
        return "(empty)"
    body = " ".join(lines[1:])  # skip the header line itself
    match = re.search(r"^(.{20,}?[.!?])(?:\s|$)", body)
    sentence = match.group(1) if match else body[:max_len]
    if len(sentence) > max_len:
        sentence = sentence[: max_len - 1] + "…"
    return f'"{sentence}"'


def item_sort_key(item_num: str) -> tuple[int, str]:
    """Sort 1 < 1A < 1B < 1C < 2 < 3 ... < 9 < 9A < 9B < 9C < 10 ..."""
    num = int(re.match(r"\d+", item_num).group())
    suffix = item_num[len(str(num)):]
    return (num, suffix)


# ─── Run ────────────────────────────────────────────────────────────────────
print(f"Fetching {NUM_FILINGS} most recent {FILING_TYPE} for {TICKER}...")
DATA_DIR.mkdir(parents=True, exist_ok=True)
Downloader(COMPANY_NAME, EMAIL, str(DATA_DIR)).get(FILING_TYPE, TICKER, limit=NUM_FILINGS)

filing_dir = DATA_DIR / "sec-edgar-filings" / TICKER / FILING_TYPE
submission_path = list(filing_dir.rglob("full-submission.txt"))[-1]
print(f"Submission: {submission_path}\n")

raw = submission_path.read_text(encoding="utf-8", errors="ignore")
filing_html = extract_primary_document(raw, FILING_TYPE)
soup = BeautifulSoup(filing_html, "lxml")
for tag in soup(["script", "style"]):
    tag.decompose()
text = soup.get_text(separator="\n")
text = re.sub(r"\n{3,}", "\n\n", text)
text = re.sub(r"[ \t]+", " ", text)
print(f"Text extracted: {len(text):,} chars\n")

# ─── Detect ─────────────────────────────────────────────────────────────────
sections = find_sections(text)

print(f"{'Item':<8} {'Title':<48} {'Chars':>9}  First sentence")
print(f"{'─' * 8} {'─' * 48} {'─' * 9}  {'─' * 60}")

total_section_chars = 0
for item_num in sorted(sections.keys(), key=item_sort_key):
    start, end, _ = sections[item_num]
    chars = end - start
    total_section_chars += chars
    title = STANDARD_ITEMS.get(item_num, "Unknown")[:46]
    preview = first_sentence(text[start:end])
    print(f"Item {item_num:<3} {title:<48} {chars:>9,}  {preview}")

# ─── Stats ──────────────────────────────────────────────────────────────────
total_chars = len(text)
coverage = total_section_chars / total_chars * 100 if total_chars else 0
unassigned = total_chars - total_section_chars

print(f"\nTotal text:            {total_chars:>10,} chars")
print(f"Captured in sections:  {total_section_chars:>10,} chars  ({coverage:.1f}%)")
print(f"Unassigned:            {unassigned:>10,} chars  (iXBRL preamble + signatures)")

print("\nTop 5 sections by size (likely retrieval-valuable):")
top = sorted(sections.items(), key=lambda x: x[1][1] - x[1][0], reverse=True)[:5]
for item_num, (start, end, _) in top:
    title = STANDARD_ITEMS.get(item_num, "Unknown")[:45]
    print(f"  Item {item_num:<3}  {title:<45}  {end - start:>8,} chars")