# FinRAG-Eval — Makefile
# Run `make help` to see available targets.

.PHONY: help install lint format typecheck test check eval ingest index serve ui clean docker-up docker-down

help:  ## Show this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n\nTargets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
install:  ## Install package + dev dependencies
	uv sync --all-extras
	pre-commit install

# ─────────────────────────────────────────────────────────────────────────────
# Quality gates (what CI runs)
# ─────────────────────────────────────────────────────────────────────────────
lint:  ## Run ruff linter
	uv run ruff check src tests

format:  ## Auto-format code with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:  ## Run mypy
	uv run mypy src

test:  ## Run pytest
	uv run pytest -m "not slow and not integration"

test-all:  ## Run all tests including slow and integration
	uv run pytest

check: lint typecheck test  ## Run all quality gates (lint + typecheck + test)

# ─────────────────────────────────────────────────────────────────────────────
# Project commands
# ─────────────────────────────────────────────────────────────────────────────
ingest:  ## Pull filings from EDGAR. Usage: make ingest COMPANIES=AAPL,MSFT YEARS=2024
	uv run finrag ingest --companies $(COMPANIES) --years $(YEARS)

index:  ## Build all retrieval indexes from ingested filings
	uv run finrag index

ask:  ## Ask a single question. Usage: make ask Q="What is..." CONFIG=hybrid
	uv run finrag ask --question "$(Q)" --config $(CONFIG)

eval:  ## Run evaluation. Usage: make eval CONFIG=hybrid_section_aware
	uv run finrag eval --config $(CONFIG)

eval-all:  ## Run full evaluation matrix across all 6 configurations
	uv run finrag eval --all-configs

# ─────────────────────────────────────────────────────────────────────────────
# Servers
# ─────────────────────────────────────────────────────────────────────────────
serve:  ## Run FastAPI service locally on :8001
	uv run uvicorn apps.api.main:app --reload --port 8001

ui:  ## Run Streamlit dashboard on :8501
	uv run streamlit run apps/frontend/app.py

# ─────────────────────────────────────────────────────────────────────────────
# Infrastructure
# ─────────────────────────────────────────────────────────────────────────────
docker-up:  ## Start Postgres + Chroma via docker compose
	docker compose up -d

docker-down:  ## Stop docker compose services
	docker compose down

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
clean:  ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
