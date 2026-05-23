# scripts/spike_edgartools.py
"""
Spike: evaluate whether edgartools handles our problem filings out-of-the-box.

Run this FIRST tomorrow morning. The output decides Path A vs Path B.

Decision rule:
- If 4+ of MSFT/JPM/BAC/MS/C show real Item 1, 1A, 7 content -> Path B (switch)
- If 0-2 show real content -> Path A (keep patching find_sections)
- If 3 -> read the failure cases manually before deciding

Setup:
    uv add --dev edgartools
    SEC_CONTACT_EMAIL=... uv run python scripts/spike_edgartools.py
"""
from __future__ import annotations

import os
import sys

try:
    from edgar import Company, set_identity
except ImportError:
    sys.exit("Run `uv add --dev edgartools` first.")

email = os.environ.get("SEC_CONTACT_EMAIL")
if not email:
    sys.exit("SEC_CONTACT_EMAIL environment variable is required.")

set_identity(f"FinRAG-Eval Team {email}")

# The five tickers that fail our in-house parser
PROBLEM_TICKERS = ["MSFT", "JPM", "BAC", "MS", "C"]
# Plus AAPL as a control — should work everywhere
CONTROL_TICKERS = ["AAPL"]

KEY_ITEMS = ["Item 1", "Item 1A", "Item 7", "Item 7A"]


def evaluate(ticker: str) -> dict:
    """Pull latest 10-K via edgartools and report what content we get for key items."""
    try:
        company = Company(ticker)
        filings = company.get_filings(form="10-K")
        if not filings:
            return {"ticker": ticker, "error": "no 10-K filings returned"}
        latest = filings.latest()
        tk = latest.obj()  # Returns TenK object

        result = {
            "ticker": ticker,
            "filing_date": str(latest.filing_date),
            "items": {},
        }
        # tk.items can be dict-like or attr-accessor depending on version
        for item in KEY_ITEMS:
            try:
                content = tk[item] if hasattr(tk, "__getitem__") else getattr(tk, item.lower().replace(" ", "_"), None)
                if content is None:
                    result["items"][item] = "MISSING"
                elif isinstance(content, str):
                    result["items"][item] = f"{len(content):,} chars"
                else:
                    result["items"][item] = f"present ({type(content).__name__})"
            except Exception as e:
                result["items"][item] = f"error: {e!r}"
        return result
    except Exception as e:
        return {"ticker": ticker, "error": f"top-level: {e!r}"}


def main() -> None:
    print("=" * 70)
    print("edgartools spike — evaluating section extraction on problem filings")
    print("=" * 70)

    for ticker in PROBLEM_TICKERS + CONTROL_TICKERS:
        print(f"\n--- {ticker} ---")
        result = evaluate(ticker)
        if "error" in result:
            print(f"  FAILED: {result['error']}")
            continue
        print(f"  filing_date: {result['filing_date']}")
        for item, status in result["items"].items():
            print(f"  {item:>8}: {status}")

    print()
    print("=" * 70)
    print("DECISION RULE")
    print("=" * 70)
    print("Count problem tickers (MSFT/JPM/BAC/MS/C) where Item 1, 1A, 7 all show")
    print("real content (not MISSING, not error):")
    print("  4-5 wins -> Path B (switch to edgartools)")
    print("  0-2 wins -> Path A (keep patching find_sections)")
    print("  3 wins   -> read the failure cases and decide manually")


if __name__ == "__main__":
    main()