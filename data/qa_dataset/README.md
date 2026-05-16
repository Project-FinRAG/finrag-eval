# QA Dataset

This directory holds the held-out QA pairs used to evaluate the system.

**This is the only `data/` content tracked in git** — it's small, hand-curated,
and the team's primary intellectual artifact.

## File

`qa_pairs.jsonl` — one `QAPair` per line (see `src/finrag_eval/common/types.py`).

## Construction

See `docs/eval-methodology.md` for the full construction protocol.
