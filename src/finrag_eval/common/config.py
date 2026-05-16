"""Centralized configuration loaded from environment variables."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # LLM providers
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_bulk_model: str = "gpt-4o-mini"
    llm_quality_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # EDGAR
    edgar_user_agent: str = Field(
        default="FinRAG-Eval Team contact@example.com",
        description="Required by SEC EDGAR. Must include real contact info.",
    )
    edgar_rate_limit_per_second: int = 8

    # Storage
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "finrag"
    postgres_user: str = "finrag"
    postgres_password: str = "changeme"

    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Paths
    data_dir: Path = Path("./data")
    raw_filings_dir: Path = Path("./data/raw")
    processed_dir: Path = Path("./data/processed")
    index_dir: Path = Path("./data/indexes")
    eval_runs_dir: Path = Path("./data/eval_runs")

    # Cost controls
    weekly_budget_usd: float = 75.0
    enable_response_cache: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
