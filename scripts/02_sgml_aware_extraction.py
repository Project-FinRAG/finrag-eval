"""Day 2: Pull AAPL's 10-K, extract only the primary document from the
SGML container, parse it, chunk it. Compare numbers to Day 1.
"""

from __future__ import annotations

import re
from pathlib import Path

import tiktoken
from bs4 import BeautifulSoup
from sec_edgar_downloader import Downloader

# ─── Config ─────────────────────────────────────────────────────────────────
TICKER = "AAPL"
FILING_TYPE = "10-K"
NUM_FILINGS = 1
COMPANY_NAME = "FinRAG-Eval Team"
EMAIL = "bhardwaj.may@northeastern.edu"
DATA_DIR = Path("./data/raw")
CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 64


def extract_primary_document(submission: str, doc_type: str) -> str:
    """Pull the <TEXT> block of the first <DOCUMENT> matching the given <TYPE>.

    SEC submissions are SGML-wrapped containers. The primary filing (e.g., 10-K)
    is one <DOCUMENT> block; subsequent blocks are exhibits we don't want.
    """
    blocks = submission.split("<DOCUMENT>")
    for block in blocks[1:]:  # skip the pre-DOCUMENT header
        type_match = re.search(r"<TYPE>([^\n<]+)", block)
        if not type_match:
            continue
        if type_match.group(1).strip() == doc_type:
            text_match = re.search(r"<TEXT>\s*(.*?)\s*</TEXT>", block, re.DOTALL)
            if text_match:
                return text_match.group(1)
    raise ValueError(f"No <DOCUMENT> with <TYPE>{doc_type} found")


# ─── Step 1: Fetch ──────────────────────────────────────────────────────────
print(f"Fetching {NUM_FILINGS} most recent {FILING_TYPE} for {TICKER}...")
DATA_DIR.mkdir(parents=True, exist_ok=True)
dl = Downloader(COMPANY_NAME, EMAIL, str(DATA_DIR))
dl.get(FILING_TYPE, TICKER, limit=NUM_FILINGS)

filing_dir = DATA_DIR / "sec-edgar-filings" / TICKER / FILING_TYPE
submission_files = list(filing_dir.rglob("full-submission.txt"))
if not submission_files:
    raise SystemExit(f"No submission file found in {filing_dir}")
filing_path = submission_files[-1]
print(f"Submission: {filing_path}")
print(f"Submission size: {filing_path.stat().st_size / 1_000_000:.1f} MB")

# ─── Step 2: Extract just the 10-K from the SGML container ──────────────────
print(f"\nExtracting <TYPE>{FILING_TYPE}</TYPE> block from SGML...")
raw_submission = filing_path.read_text(encoding="utf-8", errors="ignore")
filing_html = extract_primary_document(raw_submission, FILING_TYPE)
print(f"Primary document HTML: {len(filing_html):,} chars (vs {len(raw_submission):,} total submission)")

# ─── Step 3: HTML → text ────────────────────────────────────────────────────
print("\nParsing HTML → plain text...")
soup = BeautifulSoup(filing_html, "lxml")
# Drop script/style noise
for tag in soup(["script", "style"]):
    tag.decompose()
text = soup.get_text(separator="\n")
text = re.sub(r"\n{3,}", "\n\n", text)
text = re.sub(r"[ \t]+", " ", text)
print(f"Chars: {len(text):,}")
print(f"Words: {len(text.split()):,}")

# ─── Step 4: Chunk ──────────────────────────────────────────────────────────
print(f"\nChunking ({CHUNK_SIZE_TOKENS} tokens, {OVERLAP_TOKENS} overlap)...")
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode(text)
print(f"Total tokens: {len(tokens):,}")

step = CHUNK_SIZE_TOKENS - OVERLAP_TOKENS
chunks = [
    enc.decode(tokens[i : i + CHUNK_SIZE_TOKENS])
    for i in range(0, len(tokens), step)
    if tokens[i : i + CHUNK_SIZE_TOKENS]
]
print(f"Chunks produced: {len(chunks)}")
print(f"Avg chunk length: {sum(len(c) for c in chunks) // len(chunks):,} chars")

# ─── Step 5: Samples ────────────────────────────────────────────────────────
for label, chunk in [
    ("FIRST", chunks[0]),
    (f"MIDDLE [{len(chunks)//2}]", chunks[len(chunks) // 2]),
    (f"LAST [{len(chunks)-1}]", chunks[-1]),
]:
    print("\n" + "=" * 80)
    print(f"{label} CHUNK (first 500 chars):")
    print("=" * 80)
    print(chunk[:500])