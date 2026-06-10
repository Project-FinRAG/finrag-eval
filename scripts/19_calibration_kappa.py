"""Compute judge-vs-human agreement (Cohen's kappa) from a filled rating sheet.

Reads the human-filled rating sheet (scripts/18) + the judge sidecar, joins on
item_id, bins both scores to ordinal {0=low, 1=partial, 2=high}, and reports raw
agreement, Cohen's kappa, and quadratic-weighted kappa on correctness.

Usage:
    uv run python scripts/19_calibration_kappa.py
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

DEFAULT_SHEET = Path("data/eval_runs/calibration_sheet.csv")
DEFAULT_SIDECAR = Path("data/eval_runs/calibration_judge_scores.csv")


def _to_ordinal(score: float) -> int:
    """Bin a 0-1 score into thirds so {0,0.5,1} and the judge's float align."""
    if score < 1 / 3:
        return 0
    if score < 2 / 3:
        return 1
    return 2


def _kappa(matrix: list[list[int]], k: int, *, weighted: bool) -> float:
    """Cohen's kappa over a k x k confusion matrix; quadratic weights if asked."""
    total = sum(sum(row) for row in matrix)
    if total == 0:
        return float("nan")
    row_marg = [sum(matrix[i]) for i in range(k)]
    col_marg = [sum(matrix[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:
        if not weighted:
            return 0.0 if i == j else 1.0
        return ((i - j) / (k - 1)) ** 2

    observed = sum(w(i, j) * matrix[i][j] for i in range(k) for j in range(k)) / total
    expected = sum(
        w(i, j) * row_marg[i] * col_marg[j] for i in range(k) for j in range(k)
    ) / (total * total)
    if expected == 0:
        return float("nan")
    return 1.0 - observed / expected


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Judge-vs-human kappa from a rating sheet.")
    p.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    p.add_argument("--judge-scores", type=Path, default=DEFAULT_SIDECAR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with args.judge_scores.open(newline="") as f:
        judge_by_id = {r["item_id"]: r for r in csv.DictReader(f)}

    pairs: list[tuple[int, int]] = []
    skipped = 0
    with args.sheet.open(newline="") as f:
        for r in csv.DictReader(f):
            raw = (r.get("human_correctness") or "").strip()
            item_id = r.get("item_id", "")
            if not raw or item_id not in judge_by_id:
                skipped += 1
                continue
            pairs.append(
                (_to_ordinal(float(raw)), _to_ordinal(float(judge_by_id[item_id]["judge_correctness"])))
            )

    if not pairs:
        raise SystemExit("No rated rows found — fill the human_correctness column first.")

    k = 3
    matrix = [[0 for _ in range(k)] for _ in range(k)]
    for h, j in pairs:
        matrix[h][j] += 1

    n = len(pairs)
    agreement = sum(matrix[i][i] for i in range(k)) / n
    kappa = _kappa(matrix, k, weighted=False)
    weighted = _kappa(matrix, k, weighted=True)

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
    for i in range(k):
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