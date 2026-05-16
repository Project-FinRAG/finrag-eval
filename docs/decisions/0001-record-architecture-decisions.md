# 0001 - Record Architecture Decisions

**Status:** Accepted
**Date:** 2026-05-15
**Deciders:** FinRAG-Eval Team

## Context

Throughout the project we will make architectural choices that aren't obvious from the code itself: which vector store, why fixed-size vs. section-aware chunking is the default, how we decided on the LLM provider mix, etc. Two months from now nobody will remember why we made these decisions, including us.

## Decision

We will use lightweight Architecture Decision Records (ADRs) following Michael Nygard's template. Each ADR is a short markdown file in `docs/decisions/` named `NNNN-short-title.md`.

Each ADR has four sections:

- **Status** — Proposed, Accepted, Deprecated, or Superseded
- **Context** — what's the situation that's forcing this decision?
- **Decision** — what did we choose?
- **Consequences** — what becomes easier, what becomes harder?

## Consequences

**Good:**
- Future contributors (and our future selves) can understand *why*, not just *what*
- Forces us to think before we commit to a path
- A great artifact for the final report and demo

**Trade-off:**
- One more thing to write. We commit to writing an ADR only for choices that are genuinely architectural (would be costly to reverse later), not for every routine decision.

## References

- Michael Nygard, ["Documenting Architecture Decisions"](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions)
