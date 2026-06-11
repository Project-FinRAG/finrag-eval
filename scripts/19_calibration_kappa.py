"""Compute judge-vs-human agreement (Cohen's kappa) from a filled rating sheet.

Reads the human-filled rating sheet (scripts/18) + the judge sidecar, joins on
item_id, and reports raw agreement, Cohen's kappa, quadratic-weighted kappa, and
a confusion matrix on correctness — via AnswerJudge.calibrate_against_humans.

Usage:
    uv run python scripts/19_calibration_kappa.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from finrag_eval.eval.judge import AnswerJudge, JudgeScore

DEFAULT_SHEET = Path("data/eval_runs/calibration_sheet.csv")
DEFAULT_SIDECAR = Path("data/eval_runs/calibration_judge_scores.csv")


def _stub(correctness: float) -> JudgeScore:
    """A JudgeScore carrying only correctness; the other fields are unused
    placeholders (we call calibrate with dimensions=["correctness"])."""
    return JudgeScore(
        correctness=correctness,
        completeness=0.0,
        faithfulness=0.0,
        citation_support=0.0,
        abstention_correct=False,
        reasoning="",
    )


def _bin(score: float) -> int:
    if score < 1 / 3:
        return 0
    if score < 2 / 3:
        return 1
    return 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge-vs-human kappa from a rating sheet.")
    p.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    p.add_argument("--judge-scores", type=Path, default=DEFAULT_SIDECAR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with args.judge_scores.open(newline="") as f:
        judge_by_id = {r["item_id"]: r for r in csv.DictReader(f)}

    human_scores: list[JudgeScore] = []
    judge_scores: list[JudgeScore] = []
    skipped = 0
    with args.sheet.open(newline="") as f:
        for r in csv.DictReader(f):
            raw = (r.get("human_correctness") or "").strip()
            item_id = r.get("item_id", "")
            if not raw or item_id not in judge_by_id:
                skipped += 1
                continue
            human_scores.append(_stub(float(raw)))
            judge_scores.append(_stub(float(judge_by_id[item_id]["judge_correctness"])))

    if not human_scores:
        raise SystemExit("No rated rows found — fill the human_correctness column first.")

    out = AnswerJudge().calibrate_against_humans(
        human_scores, judge_scores, dimensions=["correctness"]
    )
    kappa = out["correctness_kappa"]
    weighted = out["correctness_kappa_weighted"]
    agreement = out["correctness_agreement"]
    n = int(out["correctness_n"])

    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for h, j in zip(human_scores, judge_scores):
        matrix[_bin(h.correctness)][_bin(j.correctness)] += 1

    def fmt(v: float) -> str:
        return "n/a (no rating variance)" if math.isnan(v) else f"{v:.3f}"

    print("=" * 52)
    print(f"Judge-vs-human calibration (correctness), n={n}")
    if skipped:
        print(f"  ({skipped} unrated/unmatched rows skipped)")
    print("=" * 52)
    print(f"  raw agreement       {agreement:.3f}")
    print(f"  Cohen's kappa       {fmt(kappa)}")
    print(f"  weighted kappa      {fmt(weighted)}")
    print()
    print("  confusion (rows = human 0/0.5/1, cols = judge 0/0.5/1):")
    for i in range(3):
        print(f"    {['0  ', '0.5', '1  '][i]}  {matrix[i]}")
    print()
    if math.isnan(kappa):
        print("  All items in one bin — add lower/partial-scored items for a meaningful kappa.")
    elif kappa >= 0.6:
        print("  >= 0.6: substantial agreement — correctness is well-supported.")
    elif kappa >= 0.4:
        print("  0.4-0.6: moderate — clears the ~0.5 bar; note as a limitation.")
    else:
        print("  < 0.4: weak — revisit the judge prompt or judge model.")


if __name__ == "__main__":
    main()