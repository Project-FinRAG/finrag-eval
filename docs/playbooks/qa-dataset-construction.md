# QA Dataset Construction Playbook

**Owner:** Evaluation Lead
**Target:** ~100 hand-curated QA pairs with gold evidence citations
**File:** `data/qa_dataset/qa_pairs.jsonl` (committed; one JSON object per line)

---

## What this dataset is for

The QA dataset is the **foundation of every retrieval and answer-quality experiment** in this project. It is the held-out evaluation set against which we measure:

- Whether our retriever finds the right chunks (Recall@K, MRR, nDCG)
- Whether the generated answer is correct, grounded, and faithful
- How different retrieval architectures and chunking strategies trade off

A weak QA dataset produces weak experiments. A strong one makes the whole project credible. This is the single most important deliverable from the Eval Lead, and the quality bar is high.

---

## Target composition

**Total: ~100 pairs.** Aim for balance across two dimensions.

### By question type (~25 each)

| Type | What it tests | Example |
|---|---|---|
| `factual_lookup` | Can the system find a specific fact in one passage? | *"What was Apple's total net sales in FY2024?"* |
| `multi_doc_synthesis` | Can the system combine evidence across multiple chunks or filings? | *"Compare risk factors related to AI competition disclosed in Microsoft's 2023 vs 2024 10-Ks."* |
| `numerical_reasoning` | Can the system answer questions requiring numerical interpretation? | *"What was JPMorgan's YoY change in net interest income from 2023 to 2024?"* |
| `temporal_comparison` | Can the system reason about changes over time? | *"How did Goldman Sachs' cybersecurity risk language evolve between 2023 and 2024?"* |

### By difficulty (rough target)

| Difficulty | Share | Characteristics |
|---|---|---|
| `easy` | ~40% | Single chunk contains the answer. Question maps directly to text. |
| `medium` | ~40% | 2–3 chunks needed. Some synthesis or interpretation required. |
| `hard` | ~20% | Multi-document, requires synthesis across sections or filings. |

### By company / sector

Spread across the 49 companies in our corpus. Don't make all questions about AAPL — diversify across tech and financial sectors. Aim for at least 30 of the 49 companies to appear in some QA pair.

---

## What makes a GOOD QA pair

1. **Specific enough to have one defensible answer.** "What does Apple do?" is bad. "What was Apple's reported FY2024 services revenue?" is good.
2. **General enough to test retrieval, not exact-string matching.** Don't quote the filing verbatim in the question. Paraphrase.
3. **Answer is genuinely in the gold evidence.** You should be able to construct the gold answer using ONLY the cited chunks.
4. **Doesn't require world knowledge beyond the filings.** "What does GAAP stand for?" is bad (general knowledge). "How does Apple define its operating segments per the 10-K?" is good (filing-specific).
5. **Has clear citation paths.** You can point to specific chunks as gold evidence.

## What makes a BAD QA pair

1. **Yes/no or single-word answers.** Too easy to guess. "Is Apple a tech company?" — bad.
2. **Multiple valid answers.** "What are Apple's biggest risks?" — too open. Specify which kind of risk, or ask for the top three as ranked in the filing.
3. **Questions requiring computation we can't verify.** "What is the discount rate that justifies Apple's current valuation?" — requires modeling we can't ground in the filing.
4. **Trivia.** "Who signed the 10-K on behalf of Apple?" — technically answerable from the filing, but useless as a retrieval/synthesis test.
5. **Ambiguous time references.** "What was Apple's revenue last year?" — relative to what date? Use absolute references: "FY2024" not "last year."

---

## The JSONL schema

Each line in `qa_pairs.jsonl` is a single JSON object. Fields:

```json
{
  "qa_id": "q-001",
  "question": "...",
  "gold_answer": "...",
  "gold_evidence": [
    {
      "chunk_id": "AAPL_0000320193-24-000123_item7_0007",
      "filing_accession": "0000320193-24-000123",
      "ticker": "AAPL",
      "section": "Item 7 - MD&A",
      "quote": "Optional verbatim quote from the chunk that justifies the answer"
    }
  ],
  "question_type": "factual_lookup",
  "difficulty": "easy",
  "notes": "Optional free-text annotator notes about edge cases"
}
```

Field reference:

| Field | Type | Required | Notes |
|---|---|---|---|
| `qa_id` | string | yes | Stable ID, format `q-NNN` (e.g., `q-001`, `q-042`) |
| `question` | string | yes | The question text |
| `gold_answer` | string | yes | Free-text gold answer (a paragraph or two is fine) |
| `gold_evidence` | array of Citation | yes | 1–5 supporting chunks. Most questions need 1–3. |
| `gold_evidence[].chunk_id` | string | yes | Exact chunk ID from `data/processed/chunks/*.jsonl` |
| `gold_evidence[].filing_accession` | string | yes | Accession number of the source filing |
| `gold_evidence[].ticker` | string | yes | Company ticker |
| `gold_evidence[].section` | string | yes for section-aware chunks | Section label if available, else `null` |
| `gold_evidence[].quote` | string | optional | Verbatim quote supporting the answer; helpful for evaluation |
| `question_type` | enum | yes | `factual_lookup`, `multi_doc_synthesis`, `numerical_reasoning`, `temporal_comparison` |
| `difficulty` | enum | yes | `easy`, `medium`, `hard` |
| `notes` | string | optional | Annotator notes |

---

## Construction process (per pair)

1. **Pick a company and a filing.** Start with the section-aware ones (clean chunk labels). AAPL, BAC, GS are good starters.
2. **Open the chunks file:** `data/processed/chunks/AAPL_10-K_<accession>.jsonl`
3. **Read a section.** Pick something substantive — Risk Factors (Item 1A), MD&A (Item 7), Business (Item 1).
4. **Formulate a question** that's specific, answerable from the section, and not exact-string-matched.
5. **Find the gold evidence.** Identify the 1–3 specific chunks (by `chunk_id`) that contain the answer.
6. **Write the gold answer.** Synthesize the answer from the evidence chunks. Quote when useful.
7. **Tag question type and difficulty.**
8. **Verify.** Read the gold evidence again and ask: "Could a reader produce the gold answer using only these chunks?" If no, revise.
9. **Append the JSON line** to `data/qa_dataset/qa_pairs.jsonl`.

### Pacing target

- 5–10 pairs per hour once you're warmed up
- ~100 pairs total = 10–20 hours of work
- Don't try to do all 100 in one sitting. Two or three focused sessions across the week is better than one marathon.

---

## Validation protocol

After ~30 pairs, do a **calibration check** with the team:

1. Pick 5 random pairs from your dataset.
2. Have Mayank (Data Lead) independently try to answer each using only the gold evidence.
3. If his answers match the gold answer ≥4/5 times, your pairs are well-grounded.
4. If not, revise the gold evidence (it's probably under-specified) or revise the gold answer (it's probably over-specified).

This is the "inter-annotator agreement" sanity check. It's also the foundation for the LLM-as-judge calibration we'll do later.

---

## Tips that save time

1. **Browse one filing fully before writing questions.** Don't jump between filings — context-switching kills throughput. Pick AAPL 10-K, read sections, write 5–10 pairs, then move on.
2. **Use the JSONL files directly.** They're at `data/processed/chunks/{TICKER}_10-K_{ACCESSION}.jsonl`. Each line is a chunk with its full text and section label. `head -1` a file to see the structure.
3. **Reuse evidence chunks across pairs.** If a chunk is rich, you can write 2–3 different questions about it.
4. **Tag difficulty conservatively.** When in doubt, mark `medium` over `easy`. Easy questions are too easy to game.
5. **Save quotes in `gold_evidence[].quote`.** Even if optional, it makes review and LLM-judge prompting much easier later.

---

## Where the chunks live

```
data/processed/chunks/
├── AAPL_10-K_0000320193-25-000079.jsonl       # 114 chunks, section-aware
├── BAC_10-K_0000070858-26-000157.jsonl        # 485 chunks, section-aware
├── GS_10-K_0000886982-26-000091.jsonl         # 559 chunks, section-aware
├── JPM_10-K_0001628280-26-008131.jsonl        # 814 chunks, fixed-size fallback
├── MSFT_10-K_0000950170-25-100235.jsonl       # 232 chunks, fixed-size fallback
└── ...(94 more files)
```

Section-aware files have chunks with labels like `"Item 1A - Risk Factors"`. Fixed-size files have positional labels like `"chunk_47"`. Start with section-aware filings — your QA pairs will have cleaner citations.

Sample a chunk:

```bash
head -1 data/processed/chunks/AAPL_10-K_0000320193-25-000079.jsonl | python -m json.tool
```

---

## When you're stuck

- **Can't find a good question in a section?** Skip it. Not every section has rich QA material. MD&A and Risk Factors usually yield the most.
- **Question feels obvious?** Make it more specific. "What does Apple sell?" → "Which product category accounted for the largest share of Apple's services revenue in FY2024?"
- **Question feels too hard?** Break it into two. A multi-doc synthesis question can become a multi-doc question + a focused factual lookup.
- **Not sure about difficulty?** Default to medium. We'll recalibrate after the first 30 pairs.

---

## Definition of done (for the full dataset)

The QA dataset is done when:

- [ ] ~100 pairs in `data/qa_dataset/qa_pairs.jsonl`
- [ ] Distribution roughly matches the targets (25 per question type, 40/40/20 difficulty split)
- [ ] At least 30 distinct companies represented
- [ ] Calibration check passed (Mayank's answers match gold ≥80% on a 5-pair sample)
- [ ] Schema validates (one JSON object per line, all required fields present)
- [ ] PR opened with the file + this playbook updated if anything changed

That's the foundation everything else in the project measures against. Take care with it.
