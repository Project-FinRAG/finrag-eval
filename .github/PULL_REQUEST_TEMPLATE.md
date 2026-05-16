## Summary

<!-- One or two sentences: what does this PR do and why? -->

## Type of Change

- [ ] `feat` — new feature
- [ ] `fix` — bug fix
- [ ] `eval` — evaluation methodology or dataset change
- [ ] `refactor` — code change that neither fixes a bug nor adds a feature
- [ ] `docs` — documentation only
- [ ] `test` — adding or fixing tests
- [ ] `chore` — tooling, dependencies, CI

## Scope

- [ ] `ingestion`
- [ ] `retrieval`
- [ ] `synthesis`
- [ ] `eval`
- [ ] `api`
- [ ] `frontend`
- [ ] `infra/ci`
- [ ] `docs`

## Checklist

- [ ] CI passes (lint + type-check + tests)
- [ ] If touching `retrieval/` or `synthesis/`, eval regression considered
- [ ] If touching `eval/`, methodology change documented in `docs/eval-methodology.md`
- [ ] Tests added or updated
- [ ] Docs updated (README, ARCHITECTURE, or ADR if architectural)
- [ ] No secrets, API keys, or filing data committed
- [ ] No `print()` left in production code (use `structlog`)

## Architectural Decision?

<!-- If this PR makes a meaningful architectural choice, add an ADR
     under docs/decisions/ and link it here. -->

ADR: <!-- e.g., docs/decisions/0003-vector-store.md or N/A -->

## How to Test

<!-- Commands a reviewer should run to verify. -->

```bash
# example
make check
make eval CONFIG=hybrid_section_aware
```

## Screenshots / Eval Output

<!-- If UI changes or eval results, paste here. -->
