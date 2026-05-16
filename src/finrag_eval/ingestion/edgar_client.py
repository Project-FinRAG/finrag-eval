"""EDGAR client — fetches filings with rate limiting and caching.

Owner: Data & Application Lead

Implementation notes:
    - EDGAR requires a User-Agent header with real contact info (see settings).
    - Rate limit is 10 req/sec; we stay polite at 8.
    - Cache raw filings to disk to avoid re-fetching during development.
    - Use `sec-edgar-downloader` as a starting point or roll our own.
"""

from __future__ import annotations

from datetime import date

from finrag_eval.common import Filing, FilingType
from finrag_eval.common.config import settings


class EdgarClient:
    """Fetches filings from SEC EDGAR.

    Example:
        >>> client = EdgarClient()
        >>> filing = client.fetch_filing("AAPL", FilingType.TEN_K, year=2024)
        >>> print(filing.sections.keys())
    """

    def __init__(
        self,
        user_agent: str | None = None,
        rate_limit_per_second: int | None = None,
    ) -> None:
        self.user_agent = user_agent or settings.edgar_user_agent
        self.rate_limit = rate_limit_per_second or settings.edgar_rate_limit_per_second

    def fetch_filing(
        self,
        ticker: str,
        filing_type: FilingType,
        year: int,
    ) -> Filing:
        """Fetch a single filing for a ticker, type, and year.

        Args:
            ticker: stock ticker (e.g. "AAPL")
            filing_type: 10-K, 10-Q, or 8-K
            year: filing year

        Returns:
            Parsed Filing with raw text and section-tagged content.

        Raises:
            NotImplementedError: this is a stub.
        """
        # TODO(@data-lead): implement using sec-edgar-downloader or requests
        # TODO(@data-lead): respect rate limit via tenacity or asyncio.Semaphore
        # TODO(@data-lead): cache raw HTML to settings.raw_filings_dir
        # TODO(@data-lead): parse with BeautifulSoup, tag sections (Item 1, 1A, 7, etc.)
        raise NotImplementedError("EdgarClient.fetch_filing is not yet implemented")

    def fetch_company_filings(
        self,
        ticker: str,
        filing_types: list[FilingType],
        start_date: date,
        end_date: date,
    ) -> list[Filing]:
        """Fetch all filings of given types for a ticker within a date range."""
        # TODO(@data-lead): batch implementation calling fetch_filing
        raise NotImplementedError("EdgarClient.fetch_company_filings is not yet implemented")
