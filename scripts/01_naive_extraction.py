"""Day 1: Pull Apple's most recent 10-K from EDGAR, parse it, chunk it.

Goal: prove the end-to-end pipeline works on one filing before generalizing.
Run from repo root:
    uv run python scripts/01_naive_extraction.py
"""


from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import os
import re
from pathlib import Path

import tiktoken
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
CHUNK_SIZE_TOKENS = 512
OVERLAP_TOKENS = 64

# ─── Step 1: Fetch ──────────────────────────────────────────────────────────
print(f"Fetching {NUM_FILINGS} most recent {FILING_TYPE} for {TICKER}...")
DATA_DIR.mkdir(parents=True, exist_ok=True)
dl = Downloader(COMPANY_NAME, EMAIL, str(DATA_DIR))
dl.get(FILING_TYPE, TICKER, limit=NUM_FILINGS)

# Locate the file the downloader created
filing_dir = DATA_DIR / "sec-edgar-filings" / TICKER / FILING_TYPE
candidates = list(filing_dir.rglob("*.txt")) + list(filing_dir.rglob("*.htm*"))
if not candidates:
    raise SystemExit(f"No filings found in {filing_dir}")
filing_path = max(candidates, key=lambda p: p.stat().st_size)  # main doc is largest
print(f"Filing: {filing_path}")
print(f"Size: {filing_path.stat().st_size / 1_000_000:.1f} MB")

# ─── Step 2: Strip to plain text ────────────────────────────────────────────
print("\nParsing HTML/XBRL → plain text...")
raw = filing_path.read_text(encoding="utf-8", errors="ignore")
soup = BeautifulSoup(raw, "lxml")
text = soup.get_text(separator="\n")
text = re.sub(r"\n{3,}", "\n\n", text)
text = re.sub(r"[ \t]+", " ", text)
print(f"Chars: {len(text):,}")
print(f"Words: {len(text.split()):,}")

# ─── Step 3: Chunk (fixed-size sliding window) ──────────────────────────────
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

# ─── Step 4: Eyeball samples ────────────────────────────────────────────────
for label, chunk in [
    ("FIRST", chunks[0]),
    (f"MIDDLE [{len(chunks)//2}]", chunks[len(chunks) // 2]),
    (f"LAST [{len(chunks)-1}]", chunks[-1]),
]:
    print("\n" + "=" * 80)
    print(f"{label} CHUNK (first 500 chars):")
    print("=" * 80)
    print(chunk[:500])
