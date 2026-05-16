# FinRAG-Eval

[![CI](https://github.com/YOUR_ORG/finrag-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/finrag-eval/actions/workflows/ci.yml)
[![Eval Regression](https://github.com/YOUR_ORG/finrag-eval/actions/workflows/eval-regression.yml/badge.svg)](https://github.com/YOUR_ORG/finrag-eval/actions/workflows/eval-regression.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> An evaluation-first financial document intelligence system for reliable SEC filing question answering.

FinRAG-Eval is a research and engineering project comparing retrieval architectures for financial question answering over SEC filings. Unlike most RAG demos, the differentiator here is **measurement**: we treat evaluation engineering as the central contribution rather than the chatbot itself.

## What This Project Answers

> Which combination of retrieval architecture (sparse / dense / hybrid+rerank) and chunking strategy (fixed-size vs. section-aware) produces the strongest faithfulness–cost trade-off for SEC filing question answering, as measured by a held-out evidence-grounded QA benchmark?

We answer this with a 6-cell experimental grid (3 retrieval × 2 chunking strategies) evaluated against ~100 hand-curated QA pairs with gold evidence citations.

## Architecture

```
                    ┌─────────────────────┐
                    │   SEC EDGAR API     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Ingestion Pipeline │  ← parsing, chunking, metadata
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        ┌─────▼─────┐   ┌──────▼──────┐  ┌──────▼──────┐
        │  BM25     │   │  Dense      │  │  Hybrid     │
        │  Index    │   │  Embeddings │  │  + Rerank   │
        └─────┬─────┘   └──────┬──────┘  └──────┬──────┘
              └────────────────┼────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Synthesis Layer    │  ← prompt + LLM + citations
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
        ┌─────▼─────┐   ┌──────▼──────┐  ┌──────▼──────┐
        │  FastAPI  │   │  Streamlit  │  │  Eval       │
        │  Service  │   │  Dashboard  │  │  Harness    │
        └───────────┘   └─────────────┘  └─────────────┘
```

The four core packages (`ingestion`, `retrieval`, `synthesis`, `eval`) each expose a clean interface so retrieval/chunking variants are swappable in experiments.

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- Docker + Docker Compose (for Postgres + vector DB)
- OpenAI API key and/or Anthropic API key

### Setup

```bash
# Clone
git clone https://github.com/YOUR_ORG/finrag-eval.git
cd finrag-eval

# Install dependencies
uv sync                          # or: pip install -e ".[dev]"

# Configure environment
cp .env.example .env
# Edit .env with your API keys and EDGAR user-agent

# Spin up services
docker compose up -d             # Postgres + Chroma

# Verify install
make check                       # runs lint + type-check + tests
```

### Run a Single Question End-to-End

```bash
make ingest COMPANIES=AAPL,MSFT YEARS=2024
make index
make ask Q="What did Apple identify as its top risk factors in its most recent 10-K?"
```

### Run the Full Eval Suite

```bash
make eval CONFIG=hybrid_section_aware
```

## Project Structure

```
finrag-eval/
├── src/finrag_eval/
│   ├── ingestion/         # EDGAR fetching, parsing, chunking
│   ├── retrieval/         # BM25, dense, hybrid, reranking (common interface)
│   ├── synthesis/         # Prompts, LLM client, citation extraction
│   ├── eval/              # QA dataset, metrics, RAGAS, LLM-judge harness
│   └── common/            # Shared types, config, logging
├── apps/
│   ├── api/               # FastAPI service
│   └── frontend/          # Streamlit demo + eval dashboard
├── tests/                 # Unit + integration tests per package
├── notebooks/             # Exploratory analysis (not production code)
├── data/                  # Gitignored; populate via `make ingest`
├── docs/
│   ├── decisions/         # Architecture Decision Records (ADRs)
│   └── eval-methodology.md
└── scripts/               # One-off utilities
```

## Evaluation Methodology

We measure both **retrieval quality** and **answer quality** independently, plus the trade-off against cost and latency.

### Retrieval Metrics
- Recall@K (K=5, 10, 20)
- Precision@K
- Mean Reciprocal Rank (MRR)
- Normalized Discounted Cumulative Gain (nDCG)
- Evidence-hit rate (gold passage in top-K)

### Answer Quality Rubric
- **Correctness** — does the answer match ground truth?
- **Completeness** — does it cover all required evidence?
- **Faithfulness** — are claims grounded in retrieved passages?
- **Citation support** — are citations accurate and traceable?
- **Abstention** — does the system correctly refuse when evidence is insufficient?

Answer quality is scored by both a human-rated sample (n=20–30) and an LLM-as-judge over the full set. We report inter-rater agreement (Cohen's κ) between the two to validate judge reliability.

See [`docs/eval-methodology.md`](docs/eval-methodology.md) for full details.

## Development Workflow

This project follows a "work like a company" model: protected main branch, PR-based development, code review required, CI must pass.

### Branch Naming

```
feat/<area>-<short-description>      e.g., feat/retrieval-bm25-baseline
fix/<area>-<short-description>       e.g., fix/ingestion-rate-limit
eval/<experiment-name>               e.g., eval/hybrid-vs-dense
docs/<topic>                         e.g., docs/adr-chunking-decision
```

### Commit Messages

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat(retrieval): add cross-encoder reranker
fix(ingestion): respect EDGAR 10 req/sec limit
eval(judge): calibrate LLM judge against human ratings
docs(adr): record decision on section-aware chunking
```

### Pull Requests

Every PR must:
- [ ] Pass CI (lint, type-check, tests)
- [ ] Pass eval-regression if it touches `retrieval/` or `synthesis/`
- [ ] Have at least 1 reviewer approval (2 for changes to eval/)
- [ ] Update relevant docs

See [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md).

### Common Commands

```bash
make lint            # Run ruff
make format          # Auto-format code
make typecheck       # Run mypy
make test            # Run pytest
make check           # All of the above
make eval            # Run evaluation suite
make ingest          # Pull filings from EDGAR
make index           # Build retrieval indexes
make serve           # Run FastAPI locally
make ui              # Run Streamlit locally
```

## Key Results

> _To be populated as experiments complete. The headline numbers go here so anyone scanning the repo sees the contribution immediately._

| Configuration | Recall@10 | Faithfulness | Cost/Query | Latency (p50) |
|---|---|---|---|---|
| BM25 + fixed-chunk | — | — | — | — |
| Dense + fixed-chunk | — | — | — | — |
| Hybrid + fixed-chunk | — | — | — | — |
| BM25 + section-aware | — | — | — | — |
| Dense + section-aware | — | — | — | — |
| Hybrid+rerank + section-aware | — | — | — | — |

## Team

- **Member 1** — Data & Application Lead (ingestion, infra, deployment)
- **Member 2** — Evaluation Lead (QA dataset, metrics, judge calibration)
- **Member 3** — Retrieval & Modeling Lead (BM25, dense, hybrid, rerank)

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- SEC EDGAR for public filing access
- [RAGAS](https://github.com/explodinggradients/ragas) for evaluation primitives
- [FinanceBench](https://arxiv.org/abs/2311.11944) for benchmark methodology inspiration
