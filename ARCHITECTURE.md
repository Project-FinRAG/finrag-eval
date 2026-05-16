# Architecture

## Design Principles

1. **Swappable components.** Every retriever, chunker, and judge implements a common interface so we can run controlled experiments without rewriting code.
2. **Evaluation is a first-class citizen.** The `eval/` package is not a notebook — it's a tested module with CI regression gates.
3. **Reproducibility over cleverness.** Every experiment is parameterized via config, logged with seed + commit SHA, and reproducible from a single command.
4. **Cost-conscious by default.** Caching is on by default. Bulk eval uses small models. Large models are explicit opt-in.

## System Components

### Ingestion (`src/finrag_eval/ingestion/`)

Owner: Data & Application Lead

Responsibilities:
- Fetch filings from SEC EDGAR with rate limiting and caching
- Parse HTML/iXBRL into structured sections
- Implement multiple chunking strategies behind a common `Chunker` interface
- Persist filing metadata to Postgres

Key contracts:
- `EdgarClient.fetch_filing(ticker, filing_type, year) -> Filing`
- `Chunker.chunk(filing: Filing) -> list[Chunk]`

### Retrieval (`src/finrag_eval/retrieval/`)

Owner: Retrieval & Modeling Lead

Responsibilities:
- Implement BM25, dense, hybrid, and reranked retrieval behind a common `Retriever` interface
- Build and persist indexes
- Expose a single `retrieve(query, k)` method per variant

Key contracts:
- `Retriever.index(chunks: list[Chunk]) -> None`
- `Retriever.retrieve(query: str, k: int) -> list[RetrievalResult]`

### Synthesis (`src/finrag_eval/synthesis/`)

Owner: Retrieval & Modeling Lead (shared)

Responsibilities:
- Compose prompts from query + retrieved passages
- Call LLM with structured output (answer + citations)
- Detect insufficient evidence → abstain

Key contracts:
- `Generator.answer(query: str, passages: list[RetrievalResult]) -> Answer`

### Evaluation (`src/finrag_eval/eval/`)

Owner: Evaluation Lead

Responsibilities:
- Maintain the held-out QA dataset (~100 pairs)
- Compute retrieval metrics (Recall@K, MRR, nDCG, evidence-hit)
- Score answers (LLM-as-judge + human rubric)
- Track cost and latency per query
- Produce comparison reports

Key contracts:
- `EvalHarness.run(retriever, generator, dataset) -> EvalReport`

## Data Flow

```
EDGAR → Ingestion → Chunks → Index (Chroma/BM25)
                                 │
                                 ▼
                          Retriever.retrieve()
                                 │
                                 ▼
                          Generator.answer()
                                 │
                          ┌──────┴──────┐
                          ▼             ▼
                       Answer       EvalHarness
                                        │
                                        ▼
                                   EvalReport
```

## Storage

- **Postgres** — filing metadata, chunk metadata, eval results
- **ChromaDB** — vector index (alternative: pgvector if we want a single store)
- **Local disk** — raw filings (cached), BM25 index pickle

## External Services

- **OpenAI API** — GPT-4o-mini for bulk eval, GPT-4o for production answers
- **Anthropic API** — Claude as alternative LLM provider for cross-validation
- **SEC EDGAR** — public filings (free, 10 req/sec limit, requires User-Agent)

## Architecture Decisions

Material decisions are recorded as ADRs in [`docs/decisions/`](docs/decisions/). Examples we'll need:
- ADR 0001: Why a single package vs. monorepo with workspaces
- ADR 0002: Chunking strategy (fixed-size vs. section-aware default)
- ADR 0003: Vector store choice (Chroma vs. pgvector)
- ADR 0004: LLM provider strategy (OpenAI only vs. multi-provider)
- ADR 0005: Eval dataset construction methodology

When you make a meaningful architectural choice, write a 1-page ADR. Future-you will thank present-you.
