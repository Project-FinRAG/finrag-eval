"""Verify the html_to_text fix recovers narrative text on previously broken filings."""
from pathlib import Path
import sys

# Add scripts dir to path so we can import the patched function
sys.path.insert(0, str(Path(__file__).parent))

# Import the patched html_to_text (after you've edited 05_corpus_scale_out.py)
from importlib import import_module
spec = import_module("05_corpus_scale_out")
html_to_text = spec.html_to_text
extract_primary_document = spec.extract_primary_document
find_sections = spec.find_sections

CASES = [
    ("AAPL (was working)", "data/raw/sec-edgar-filings/AAPL/10-K/0000320193-24-000123/full-submission.txt"),
    ("MSFT (was broken)", "data/raw/sec-edgar-filings/MSFT/10-K/0000950170-24-087843/full-submission.txt"),
    ("JPM (was broken)", "data/raw/sec-edgar-filings/JPM/10-K/0000019617-25-000270/full-submission.txt"),
    ("BAC (was broken)", "data/raw/sec-edgar-filings/BAC/10-K/0000070858-25-000139/full-submission.txt"),
]

for label, path in CASES:
    print(f"\n=== {label} ===")
    raw = Path(path).read_text(encoding="utf-8", errors="ignore")
    html = extract_primary_document(raw, "10-K")
    text = html_to_text(html)
    sections = find_sections(text)
    print(f"  Extracted text length: {len(text):,} chars")
    print(f"  Sections detected:     {len(sections)}")
    print(f"  Section keys found:    {sorted(sections.keys())[:10]}{'...' if len(sections) > 10 else ''}")
    print(f"  First 300 chars of text:")
    print(f"    {text[:300]!r}")
    # Look for the smoking gun — if "us-gaap" appears in first 5000 chars, contamination remains
    if "us-gaap" in text[:5000].lower():
        print(f"  ⚠️  XBRL contamination STILL PRESENT in first 5000 chars")
    else:
        print(f"  ✓ No XBRL contamination in first 5000 chars")