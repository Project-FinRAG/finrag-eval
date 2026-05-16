# Evaluation Methodology

This document is the source of truth for how FinRAG-Eval evaluates its systems. If you change anything in `src/finrag_eval/eval/`, update this document in the same PR.

## Overview

We evaluate two things independently, then study their trade-off against cost and latency:

1. **Retrieval quality** — does the system find the right passages?
2. **Answer quality** — does it produce a correct, grounded, well-cited answer?

Both are measured against a held-out QA dataset of ~100 hand-curated pairs.

---

## The QA Dataset

### Target Composition (~100 pairs)

| Dimension | Distribution |
|---|---|
| Question Type | 25 factual lookup, 25 multi-doc synthesis, 25 numerical reasoning, 25 temporal comparison |
| Difficulty | 40% easy, 40% medium, 20% hard |
| Companies | spread across the 30 in-scope companies |
| Filings | mix of 10-K and 10-Q across 2022–2024 |

### Construction Protocol

1. **Two annotators** each draft questions targeting specific filing sections
2. Each question must include:
   - The question text
   - The gold answer (free text)
   - Gold evidence citations (1–5 specific chunks from specific filings)
   - Question type and difficulty
3. A **third annotator** independently verifies that the gold evidence actually supports the gold answer
4. Inter-annotator disagreements are resolved by team discussion or the question is dropped

### Storage

JSONL at `data/qa_dataset/qa_pairs.jsonl`. Each line is a `QAPair` (see `common/types.py`). This file IS committed to git despite the `data/` gitignore — it's small, hand-curated, and the team's primary intellectual artifact.

---

## Retrieval Metrics

All metrics computed at K ∈ {5, 10, 20}.

| Metric | Definition | What it tells us |
|---|---|---|
| Recall@K | fraction of gold passages in top-K | "Did we find everything?" |
| Precision@K | fraction of top-K that are gold | "How much noise did we pull?" |
| MRR | reciprocal rank of first gold hit | "How fast did we find the first one?" |
| nDCG@K | rank-discounted gain over ideal | "Are gold hits near the top?" |
| Evidence-hit | 1.0 if any gold in top-K else 0.0 | Coarse but interpretable for end users |

**Why these specifically:**
- Recall@K is the primary headline metric for retrieval comparison
- MRR captures the "find at least one good thing fast" property that matters for grounded QA
- nDCG handles the rank-dependent value of retrieval
- Evidence-hit is the simplest end-user-meaningful metric ("did we even surface the answer?")

---

## Answer Quality Rubric

Each answer is scored on 5 dimensions, each on a [0, 1] scale (continuous, not categorical).

### 1. Correctness
Does the answer match the gold answer? Allow paraphrase but not contradiction. Partial credit for partially-correct answers.

### 2. Completeness
Does the answer cover everything the question asked? E.g., "compare X and Y" with only X mentioned is incomplete.

### 3. Faithfulness
Are all factual claims grounded in the retrieved passages? An answer that's *correct* but cites nothing or hallucinates is unfaithful even if right.

### 4. Citation Support
Are the citations actually traceable to passages? Did the system cite the right passages?

### 5. Abstention Correctness (boolean)
- If gold answer exists and system answered: 1 if answer is good, 0 if abstained
- If gold has no answer and system abstained: 1
- If gold has no answer and system invented one: 0 (worst failure)

---

## LLM-as-Judge

We use an LLM to score answer quality at scale because human-rating 100 answers × 6 configurations = 600 scores is impractical.

**Critical:** an unvalidated LLM judge is worthless. So we calibrate.

### Calibration Protocol

1. Sample 20–30 (question, answer) pairs across all 6 configurations
2. Each team member rates them independently using the same rubric
3. Run the LLM judge on the same set
4. Compute **Cohen's κ** per rubric dimension between (avg human, judge)
5. **Acceptance threshold:** κ ≥ 0.5 ("moderate agreement") per dimension
6. If a dimension fails, iterate on the judge prompt or switch judge model

This calibration result is reported in the final paper. Without it, the eval is unreliable.

### Judge Model

Default: GPT-4o for judging (more capable). Bulk eval uses GPT-4o-mini for the *answer generation* (under test); the judge is held constant.

We could ablate this too, but cross-judge variance is a stretch goal.

---

## Cost and Latency Tracking

Every evaluation run records:
- Total cost in USD (computed from token counts × per-model rates)
- Latency p50, p95, p99 in milliseconds per query
- Per-question breakdown of cost and latency

This lets us produce the Pareto frontier of quality-vs-cost in the final report.

---

## Reporting

Each `EvalReport` contains:
- Configuration name, retriever, chunker, generator model, judge model
- Timestamp + commit SHA (reproducibility)
- Aggregate metrics (mean, median, p95 of each)
- Per-question results (for error analysis)

Reports are stored at `data/eval_runs/<timestamp>-<config>.json`.

The final paper's comparison table is generated by aggregating these reports — never hand-typed.

---

## What This Methodology Is *Not*

- **Not a real-world deployment evaluation** — we don't measure user satisfaction or business outcomes
- **Not a multi-turn conversation evaluation** — single-question scope only
- **Not a hallucination benchmark in the strict sense** — faithfulness is rubric-scored, not detection-style

These are scope limits we accept for the 7-week timeline.
