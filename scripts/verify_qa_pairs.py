"""Verify every QA pair in data/qa_dataset/qa_pairs.jsonl points at chunks
that actually exist in our corpus.

Checks per pair:
  - qa_id is unique
  - All required schema fields present
  - For each gold_evidence entry:
      * The chunk_id exists in data/processed/chunks/
      * The ticker and accession in the citation match the chunk's metadata
      * If 'quote' is provided, it appears in the chunk's text (with some
        normalization for whitespace)

Exit code 0 if all pairs valid, non-zero otherwise.

Usage:
    uv run python scripts/verify_qa_pairs.py
    uv run python scripts/verify_qa_pairs.py path/to/qa_pairs.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")
CHUNKS_DIR = Path("data/processed/chunks")

REQUIRED_TOP_FIELDS = {
    "qa_id", "question", "gold_answer", "gold_evidence",
    "question_type", "difficulty",
}
REQUIRED_EVIDENCE_FIELDS = {"chunk_id", "filing_accession", "ticker"}
ALLOWED_QUESTION_TYPES = {
    "factual_lookup", "multi_doc_synthesis",
    "numerical_reasoning", "temporal_comparison",
}
ALLOWED_DIFFICULTIES = {"easy", "medium", "hard"}


QUOTE_TRANSLATION = str.maketrans({
    "\u2018": "'", "\u2019": "'",  # smart single quotes
    "\u201C": '"', "\u201D": '"',  # smart double quotes
    "\u00A0": " ",                  # non-breaking space
})


def normalize_for_search(s: str) -> str:
    """Collapse whitespace, lowercase, normalize smart quotes and NBSPs.

    SEC filings use typographic punctuation in HTML output (U+2019 right single
    quote, U+00A0 non-breaking space, etc.). QA pair quotes are often typed with
    straight ASCII equivalents. Normalize both to ASCII before matching.
    """
    s = s.translate(QUOTE_TRANSLATION)
    return re.sub(r"\s+", " ", s).lower().strip()


def load_chunk_index() -> dict[str, dict]:
    """Map chunk_id -> chunk dict for every chunk in the corpus."""
    if not CHUNKS_DIR.exists():
        sys.exit(f"FATAL: {CHUNKS_DIR} not found. Run the ingestion pipeline first.")
    index: dict[str, dict] = {}
    for jsonl_path in CHUNKS_DIR.glob("*.jsonl"):
        with jsonl_path.open() as f:
            for line in f:
                c = json.loads(line)
                index[c["chunk_id"]] = c
    return index


def verify(qa_path: Path) -> int:
    if not qa_path.exists():
        sys.exit(f"FATAL: {qa_path} not found.")

    print(f"Loading chunk index from {CHUNKS_DIR}...")
    chunk_index = load_chunk_index()
    print(f"  Loaded {len(chunk_index):,} chunks across {len(list(CHUNKS_DIR.glob('*.jsonl')))} files")
    print()

    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    n_pairs = 0

    with qa_path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"line {line_no}: invalid JSON: {e}")
                continue
            n_pairs += 1
            qa_id = pair.get("qa_id", f"<line {line_no}>")

            # Required fields
            missing = REQUIRED_TOP_FIELDS - set(pair.keys())
            if missing:
                errors.append(f"{qa_id}: missing required fields: {missing}")
                continue

            # Unique qa_id
            if qa_id in seen_ids:
                errors.append(f"{qa_id}: duplicate qa_id")
            seen_ids.add(qa_id)

            # Enum validation
            if pair["question_type"] not in ALLOWED_QUESTION_TYPES:
                errors.append(
                    f"{qa_id}: question_type {pair['question_type']!r} not in "
                    f"{sorted(ALLOWED_QUESTION_TYPES)}"
                )
            if pair["difficulty"] not in ALLOWED_DIFFICULTIES:
                errors.append(
                    f"{qa_id}: difficulty {pair['difficulty']!r} not in "
                    f"{sorted(ALLOWED_DIFFICULTIES)}"
                )

            # Evidence
            evidence = pair.get("gold_evidence", [])
            if not isinstance(evidence, list) or not evidence:
                errors.append(f"{qa_id}: gold_evidence must be a non-empty list")
                continue

            for i, ev in enumerate(evidence):
                tag = f"{qa_id}.evidence[{i}]"
                missing_ev = REQUIRED_EVIDENCE_FIELDS - set(ev.keys())
                if missing_ev:
                    errors.append(f"{tag}: missing fields: {missing_ev}")
                    continue
                cid = ev["chunk_id"]
                if cid not in chunk_index:
                    errors.append(f"{tag}: chunk_id {cid!r} not found in corpus")
                    continue
                chunk = chunk_index[cid]
                if chunk["ticker"] != ev["ticker"]:
                    errors.append(
                        f"{tag}: ticker mismatch — citation says {ev['ticker']!r}, "
                        f"chunk says {chunk['ticker']!r}"
                    )
                if chunk["accession"] != ev["filing_accession"]:
                    errors.append(
                        f"{tag}: accession mismatch — citation says "
                        f"{ev['filing_accession']!r}, chunk says {chunk['accession']!r}"
                    )
                # Optional quote check
                quote = ev.get("quote")
                if quote:
                    if normalize_for_search(quote) not in normalize_for_search(chunk["text"]):
                        warnings.append(
                            f"{tag}: quote not found in chunk text (may be paraphrased)"
                        )

    print(f"Verified {n_pairs} QA pairs.")
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    print()
    if errors:
        print("ERRORS:")
        for e in errors:
            print(f"  ✗ {e}")
        print()
    if warnings:
        print("WARNINGS:")
        for w in warnings:
            print(f"  ⚠ {w}")
        print()

    if errors:
        return 1
    print("✓ All QA pairs valid.")
    return 0


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else QA_PATH
    sys.exit(verify(path))
