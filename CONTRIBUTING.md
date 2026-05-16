# Contributing to FinRAG-Eval

This is a 7-week capstone project. We work like a small team, not like classmates.

## Setup

```bash
git clone https://github.com/YOUR_ORG/finrag-eval.git
cd finrag-eval
uv sync --all-extras
pre-commit install
cp .env.example .env
# Fill in your API keys
docker compose up -d
make check  # verify everything works
```

## Workflow

1. **Pick or create a ticket** on the GitHub Projects board
2. **Create a branch** from `main` following the naming convention
3. **Write code + tests** locally; run `make check` before pushing
4. **Open a PR** using the template; tag a reviewer
5. **Address review comments**; CI must pass
6. **Squash-merge** when approved

### Branch Naming

```
feat/<area>-<short-description>      # e.g., feat/retrieval-bm25-baseline
fix/<area>-<short-description>       # e.g., fix/ingestion-rate-limit
eval/<experiment-name>               # e.g., eval/hybrid-vs-dense
docs/<topic>                         # e.g., docs/adr-chunking-decision
```

### Commits

[Conventional Commits](https://www.conventionalcommits.org/). Examples:

```
feat(retrieval): add cross-encoder reranker
fix(ingestion): respect EDGAR 10 req/sec limit
eval(judge): calibrate LLM judge against human ratings
docs(adr): record decision on section-aware chunking
test(metrics): add nDCG edge cases
chore(deps): bump ragas to 0.3
```

## Code Standards

- **Python 3.12+**, type hints on every function (mypy strict mode)
- **Ruff** for linting + formatting (auto-applied by pre-commit)
- **Docstrings** on every public class and function. Brief is fine, but explain the *why*, not just the *what*
- **No `print()`** in production code — use `structlog`
- **No magic strings** for config — use `settings` from `common/config.py`
- **Tests** for any non-trivial logic. Smoke tests are enough for thin wrappers; real tests for metrics, parsing, prompt construction

## Code Review

Every PR needs at least 1 approval. PRs touching `eval/` need 2 approvals because changes there invalidate prior results.

**Reviewing well:**
- Read the code, not just the diff summary
- Run it locally if it's non-trivial
- Ask "what could go wrong here?"
- Approve or request changes within 24 hours so the author isn't blocked

**Receiving review:**
- Don't take it personally
- If you disagree, push back with reasoning, not feeling
- If a comment makes the code better, just make the change

## Architecture Decisions

If your PR makes a meaningful architectural choice (vector store, chunking default, eval methodology), write an ADR in `docs/decisions/`. See [`docs/decisions/0001-record-architecture-decisions.md`](docs/decisions/0001-record-architecture-decisions.md) for the format.

## Communication

- **Discord/Slack**: async, daily
- **Weekly sync**: Tuesdays after class, ~30 min
- **Mid-week check-in**: optional, only if someone's blocked
- **No silent stuck.** If you're blocked for >2 hours, ask.

## Definition of Done

A task is done when:
- [ ] Code is written, tested, and reviewed
- [ ] CI is green
- [ ] Docs are updated if relevant
- [ ] PR is merged to `main`
- [ ] If it was an experiment, results are committed under `data/eval_runs/`
