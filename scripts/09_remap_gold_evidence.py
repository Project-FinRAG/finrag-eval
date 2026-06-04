"""Re-map gold_evidence chunk_ids to the chunk that actually contains each quote.

Some QA pairs reference chunk_ids that exist but point at the wrong chunk (e.g.
the opening chunk of a section instead of the chunk holding the answer). Each
gold_evidence entry carries the verbatim `quote` the author intended, so this
uses those quotes as ground truth: for each entry it finds the labeled chunk in
the same filing whose text contains the quote, and (with --write) rewrites the
chunk_id to that chunk. It reconstructs intent from the authors' own quotes — it
does not invent gold. Route the resulting diff through the Eval Lead.

Modes:
    uv run python scripts/09_remap_gold_evidence.py            # dry-run: show proposed remaps
    uv run python scripts/09_remap_gold_evidence.py --check    # CI gate: fail if a quote isn't in its chunk
    uv run python scripts/09_remap_gold_evidence.py --write     # apply (only if everything resolves uniquely)
"""

from __future__ import annotations

import argparse
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

from finrag_eval.common import Chunk
from finrag_eval.retrieval.bm25 import load_chunks_from_jsonl

DEFAULT_QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")


def norm(s: str) -> str:
    """Normalize text for robust substring matching across extraction artifacts."""
    s = unicodedata.normalize("NFKC", s)
    for a, b in (
        ("\u2019", "'"), ("\u2018", "'"),
        ("\u201c", '"'), ("\u201d", '"'),
        ("\u2013", "-"), ("\u2014", "-"),
        ("\xa0", " "),
    ):
        s = s.replace(a, b)
    return " ".join(s.split())


def build_index(strategy: str) -> dict[str, list[tuple[str, str]]]:
    """filing_accession -> list of (chunk_id, normalized_text) for the index subset."""
    chunks: list[Chunk] = load_chunks_from_jsonl(strategy)  # type: ignore[arg-type]
    by_acc: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for c in chunks:
        by_acc[c.filing_accession].append((c.chunk_id, norm(c.text)))
    return by_acc


def find_chunks(quote: str, accession: str, index: dict[str, list[tuple[str, str]]]) -> list[str]:
    """chunk_ids in the filing whose text contains the quote (normalized)."""
    nq = norm(quote)
    if not nq:
        return []
    hits = [cid for cid, ntext in index.get(accession, []) if nq in ntext]
    if not hits:
        low = nq.lower()
        hits = [cid for cid, ntext in index.get(accession, []) if low in ntext.lower()]
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--strategy", default="labeled",
                        choices=["labeled", "strict", "fixed_size", "all"],
                        help="Chunk subset to match against (default: labeled = what's indexed)")
    parser.add_argument("--check", action="store_true",
                        help="Validate current chunk_ids contain their quotes; exit 1 on any failure")
    parser.add_argument("--write", action="store_true",
                        help="Rewrite chunk_ids in place (only if every evidence resolves uniquely)")
    args = parser.parse_args()

    raw_lines = [ln for ln in args.qa_path.read_text().splitlines() if ln.strip()]
    records = [json.loads(ln) for ln in raw_lines]
    index = build_index(args.strategy)

    changed = [False] * len(records)
    rows: list[tuple[str, int, str, str, str, str]] = []
    n_changed = n_unresolved = n_check_fail = 0

    for j, rec in enumerate(records):
        qa_id = rec["qa_id"]
        for i, ev in enumerate(rec.get("gold_evidence", [])):
            old = ev["chunk_id"]
            section = ev.get("section", "")
            quote = ev.get("quote", "")
            acc = ev["filing_accession"]
            hits = find_chunks(quote, acc, index)

            if args.check:
                if not (norm(quote) and old in hits):
                    n_check_fail += 1
                    rows.append((qa_id, i, section, "QUOTE NOT IN CHUNK", old, ""))
                continue

            if old in hits:
                rows.append((qa_id, i, section, "ok", old, old))
            elif len(hits) == 1:
                n_changed += 1
                rows.append((qa_id, i, section, "REMAP", old, hits[0]))
                if args.write:
                    ev["chunk_id"] = hits[0]
                    changed[j] = True
            elif not hits:
                n_unresolved += 1
                rows.append((qa_id, i, section, "NOT FOUND", old, ""))
            else:
                n_unresolved += 1
                rows.append((qa_id, i, section, f"AMBIGUOUS({len(hits)})", old, "  ".join(hits[:3])))

    if args.check:
        for qa_id, i, section, status, old, _ in rows:
            print(f"  {qa_id} ev{i} [{section}] {status}: {old}")
        print(f"\n{len(records)} pairs checked; {n_check_fail} evidence entries fail quote-containment.")
        return 1 if n_check_fail else 0

    w = max((len(r[4]) for r in rows), default=10)
    for qa_id, i, section, status, old, new in rows:
        tail = f"  ->  {new}" if status == "REMAP" else (f"  [{new}]" if status.startswith("AMBIG") else "")
        print(f"  {qa_id:<7} ev{i} [{section:<30}] {status:<18} {old:<{w}}{tail}")
    print(f"\nProposed: {n_changed} remap(s), {n_unresolved} unresolved.")

    if args.write:
        if n_unresolved:
            print("\nRefusing to write: resolve NOT FOUND / AMBIGUOUS first "
                  "(a quote may span a chunk boundary, sit in a non-labeled chunk, "
                  "or differ in wording). No changes made.")
            return 1
        out = [json.dumps(records[j]) if changed[j] else raw_lines[j] for j in range(len(records))]
        args.qa_path.write_text("\n".join(out) + "\n")
        print(f"\nWrote {n_changed} remap(s) to {args.qa_path}. Review the git diff.")
    else:
        print("\nDry-run. Re-run with --write to apply, or --check for the CI gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())