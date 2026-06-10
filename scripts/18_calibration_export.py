"""Export a blind human-rating sheet for judge calibration (correctness).

Samples answered questions from a saved e2e EvalReport and writes two files:
  - a rating sheet (CSV): question / gold answer / gold evidence / system answer
    + a blank `human_correctness` column for a rater to fill. NO judge scores, so
    the rating is blind.
  - a judge sidecar (CSV): the judge's scores per item, joined back by item_id at
    scoring time (scripts/19).

Rate `human_correctness` on {0, 0.5, 1}: 0 = wrong / unsupported / blank,
0.5 = partially correct, 1 = fully correct — judged against the gold answer.

Usage:
    uv run python scripts/18_calibration_export.py \
        --report data/eval_runs/e2e_dense-large_labeled.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from finrag_eval.eval.qa_dataset import QADataset

DEFAULT_REPORT = Path("data/eval_runs/e2e_dense-large_labeled.json")
DEFAULT_QA_PATH = Path("data/qa_dataset/qa_pairs.jsonl")
DEFAULT_SHEET = Path("data/eval_runs/calibration_sheet.csv")
DEFAULT_SIDECAR = Path("data/eval_runs/calibration_judge_scores.csv")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a blind calibration rating sheet.")
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    p.add_argument("--n", type=int, default=30, help="Number of items to sample.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    p.add_argument("--judge-scores", type=Path, default=DEFAULT_SIDECAR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    report = json.loads(args.report.read_text())
    per_question = report.get("per_question", [])

    qa = QADataset(args.qa_path)
    qa.load()
    gold = {p.qa_id: p for p in qa}

    eligible = [row for row in per_question if row.get("judge_score") is not None]
    if not eligible:
        raise SystemExit(f"No scored answers in {args.report}.")

    n = min(args.n, len(eligible))
    sample = random.Random(args.seed).sample(eligible, n)
    sample.sort(key=lambda r: r["qa_id"])

    args.sheet.parent.mkdir(parents=True, exist_ok=True)

    with args.sheet.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["item_id", "qa_id", "question", "gold_answer", "gold_evidence",
             "system_answer", "human_correctness"]
        )
        for i, row in enumerate(sample, start=1):
            pair = gold.get(row["qa_id"])
            evidence = (
                " | ".join(c.quote for c in pair.gold_evidence if c.quote) if pair else ""
            )
            w.writerow([
                i,
                row["qa_id"],
                pair.question if pair else "(question not found)",
                (pair.gold_answer if pair else "") or "",
                evidence,
                row.get("answer_text", ""),
                "",  # rater fills this
            ])

    with args.judge_scores.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["item_id", "qa_id", "judge_correctness", "judge_completeness",
             "judge_faithfulness", "judge_citation_support", "judge_abstention_correct"]
        )
        for i, row in enumerate(sample, start=1):
            js = row["judge_score"]
            w.writerow([
                i, row["qa_id"], js["correctness"], js["completeness"],
                js["faithfulness"], js["citation_support"], js["abstention_correct"],
            ])

    bins = {"0  (<0.33)": 0, "0.5 (0.33-0.67)": 0, "1  (>=0.67)": 0}
    for row in sample:
        c = float(row["judge_score"]["correctness"])
        key = "0  (<0.33)" if c < 1 / 3 else ("0.5 (0.33-0.67)" if c < 2 / 3 else "1  (>=0.67)")
        bins[key] += 1

    print(f"Wrote {n} items.")
    print(f"  rating sheet  -> {args.sheet}   (hand to the rater; blind)")
    print(f"  judge sidecar -> {args.judge_scores}   (keep; used by scripts/19)")
    print("  judge correctness spread in this sample:")
    for key, count in bins.items():
        print(f"    {key:16s} {count}")


if __name__ == "__main__":
    main()