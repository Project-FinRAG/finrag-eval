"""Analyze faithfulness-correctness gap, conditioned on abstention status.

Hypothesis: abstentions are trivially faithful (no unsupported claims)
but score 0 on correctness, inflating the gap. This script tests whether
the gap collapses when conditioned on answered-only queries.

Usage:
    uv run python scripts/25_gap_analysis.py results/e2e_*.json
"""

import json
import sys
from pathlib import Path


def analyze_file(path: Path) -> dict:
    """Break faithfulness and correctness down by abstained vs answered."""
    with open(path) as f:
        data = json.load(f)

    records = data.get("per_question", [])
    if not records:
        # Fallback: try list format
        records = data if isinstance(data, list) else []

    answered = [r for r in records if not r.get("abstained", False)]
    abstained = [r for r in records if r.get("abstained", False)]

    def avg(lst: list, key: str) -> float:
        if not lst:
            return 0.0
        return sum(r["judge_score"][key] for r in lst) / len(lst)

    return {
        "config": data.get("config_name", path.stem),
        "retriever": data.get("retriever_name", "unknown"),
        "n_total": len(records),
        "answered": {
            "n": len(answered),
            "faith": round(avg(answered, "faithfulness"), 3),
            "corr": round(avg(answered, "correctness"), 3),
        },
        "abstained": {
            "n": len(abstained),
            "faith": round(avg(abstained, "faithfulness"), 3),
            "corr": round(avg(abstained, "correctness"), 3),
        },
        "all": {
            "faith": round(avg(records, "faithfulness"), 3),
            "corr": round(avg(records, "correctness"), 3),
        },
    }


def print_report(stats: dict) -> None:
    """Pretty-print the gap analysis for one config."""
    ans = stats["answered"]
    abst = stats["abstained"]
    total = stats["all"]

    print(f"\n{'=' * 60}")
    print(f"  {stats['config']}")
    print(f"{'=' * 60}")
    print(f"  {'Bucket':<12} {'N':>4}  {'Faith':>7}  {'Corr':>7}  {'Gap':>7}")
    print(f"  {'-' * 42}")
    print(
        f"  {'Answered':<12} {ans['n']:>4}"
        f"  {ans['faith']:>7.3f}  {ans['corr']:>7.3f}"
        f"  {ans['faith'] - ans['corr']:>+7.3f}"
    )
    print(
        f"  {'Abstained':<12} {abst['n']:>4}"
        f"  {abst['faith']:>7.3f}  {abst['corr']:>7.3f}"
        f"  {abst['faith'] - abst['corr']:>+7.3f}"
    )
    print(f"  {'-' * 42}")
    overall_gap = total["faith"] - total["corr"]
    print(
        f"  {'ALL':<12} {stats['n_total']:>4}"
        f"  {total['faith']:>7.3f}  {total['corr']:>7.3f}"
        f"  {overall_gap:>+7.3f}"
    )

    abs_pct = abst["n"] / stats["n_total"] * 100 if stats["n_total"] else 0
    answered_gap = ans["faith"] - ans["corr"]
    print(f"\n  Abstention rate: {abst['n']}/{stats['n_total']} ({abs_pct:.0f}%)")
    print(f"  Overall gap:     {overall_gap:+.3f}")
    print(f"  Answered-only:   {answered_gap:+.3f}")

    reduction = (1 - answered_gap / overall_gap) * 100 if overall_gap > 0 else 0
    print(f"  Gap reduction:   {reduction:.0f}%")

    if overall_gap > 0.05 and answered_gap < overall_gap * 0.5:
        print("  -> Gap COLLAPSES. Abstentions are the primary driver.")
    elif overall_gap > 0.05:
        print("  -> Gap PERSISTS for answered queries. Not purely abstention-driven.")
    else:
        print("  -> Minimal gap overall.")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/25_gap_analysis.py results/e2e_*.json")
        sys.exit(1)

    all_stats = []
    for fpath in sorted(sys.argv[1:]):
        stats = analyze_file(Path(fpath))
        print_report(stats)
        all_stats.append(stats)

    # Summary table
    if len(all_stats) > 1:
        print(f"\n{'=' * 60}")
        print("  SUMMARY")
        print(f"{'=' * 60}")
        print(f"  {'Config':<28} {'Abst%':>6} {'AllGap':>7}" f" {'AnsGap':>7} {'Reduct':>7}")
        print(f"  {'-' * 58}")
        for s in all_stats:
            abst_pct = s["abstained"]["n"] / s["n_total"] * 100
            all_gap = s["all"]["faith"] - s["all"]["corr"]
            ans_gap = s["answered"]["faith"] - s["answered"]["corr"]
            reduction = (1 - ans_gap / all_gap) * 100 if all_gap > 0 else 0
            print(
                f"  {s['config']:<28} {abst_pct:>5.0f}%"
                f" {all_gap:>+7.3f} {ans_gap:>+7.3f} {reduction:>6.0f}%"
            )


if __name__ == "__main__":
    main()
