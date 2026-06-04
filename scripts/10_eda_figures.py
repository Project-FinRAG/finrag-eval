"""Generate publication-ready PNG figures for the EDA report.

Reads the v0.1 corpus manifest and chunk files, produces 6 figures saved to
docs/figures/. Designed to be idempotent: clears existing figures before
regenerating so the script is the single source of truth for what appears
in the report.

Usage:
    uv run python scripts/10_eda_figures.py

Output:
    docs/figures/fig01_filing_size_by_sector.png
    docs/figures/fig02_tier_distribution.png
    docs/figures/fig03_tier_by_sector.png
    docs/figures/fig04_section_label_distribution.png
    docs/figures/fig05_item7_chars_by_tier.png
    docs/figures/fig06_failure_mode_distribution.png
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["font.size"] = 10

ROOT = Path.cwd()
if not (ROOT / "pyproject.toml").exists():
    ROOT = ROOT.parent

MANIFEST_PATH = ROOT / "data/metadata/corpus_manifest_v0.1.csv"
CHUNKS_DIR = ROOT / "data/processed/chunks"
FIGURES_DIR = ROOT / "docs/figures"

# Sector taxonomy (authoritative, matches notebook 01_corpus_eda.py)
TECH = {
    "AAPL", "MSFT", "GOOGL", "META", "AMZN", "NVDA", "AMD", "INTC", "CSCO",
    "ORCL", "CRM", "ADBE", "NOW", "INTU", "IBM", "TXN", "QCOM", "AVGO",
    "MU", "AMAT", "LRCX", "KLAC", "PANW", "ANET", "ACN", "CDNS", "SNPS",
}
FINANCIAL = {
    "JPM", "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF",
    "SCHW", "BLK", "V", "MA", "AXP", "SPGI", "MCO", "ICE", "CME", "BRK-B",
    "AIG", "MET", "PRU", "ALL", "TRV", "WTW", "AON", "APO", "BX", "CB", "KKR",
}

TIER_ORDER = ["section_aware", "hybrid_section_aware", "fixed_size"]
TIER_LABELS = {
    "section_aware": "Section-aware\n(strict)",
    "hybrid_section_aware": "Hybrid\nsection-aware",
    "fixed_size": "Fixed-size\nfallback",
}
TIER_COLORS = {
    "section_aware": "#2E7D32",
    "hybrid_section_aware": "#F9A825",
    "fixed_size": "#C62828",
}


def sector_of(ticker: str) -> str:
    if ticker in TECH:
        return "tech"
    if ticker in FINANCIAL:
        return "financial"
    return "other"


def load_manifest() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST_PATH)
    df["sector"] = df["ticker"].apply(sector_of)
    assert df["sector"].ne("other").all(), \
        f"Uncategorized tickers: {sorted(df[df['sector']=='other']['ticker'].unique())}"
    return df


def fig01_filing_size_by_sector(manifest: pd.DataFrame) -> None:
    """Filing size distribution, stratified by sector."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for sector, color in [("tech", "#1976D2"), ("financial", "#7B1FA2")]:
        sub = manifest[manifest["sector"] == sector]
        ax.hist(
            sub["total_chars"] / 1_000_000,
            bins=25,
            alpha=0.6,
            label=f"{sector.capitalize()} (n={len(sub)})",
            color=color,
            edgecolor="white",
        )
    ax.set_xlabel("Filing size (million characters)")
    ax.set_ylabel("Number of filings")
    ax.set_title("Filing size distribution by sector")
    ax.legend(loc="upper right")
    fig.savefig(FIGURES_DIR / "fig01_filing_size_by_sector.png")
    plt.close(fig)


def fig02_tier_distribution(manifest: pd.DataFrame) -> None:
    """Bar chart of the three chunking tiers."""
    counts = manifest["chunking_method"].value_counts().reindex(TIER_ORDER, fill_value=0)
    pct = (counts / len(manifest) * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        [TIER_LABELS[t] for t in TIER_ORDER],
        counts.values,
        color=[TIER_COLORS[t] for t in TIER_ORDER],
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, n, p in zip(bars, counts.values, pct.values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            n + 3,
            f"{n}\n({p}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Number of filings")
    ax.set_title("Chunking tier distribution (N=198)")
    ax.set_ylim(0, max(counts.values) * 1.15)
    fig.savefig(FIGURES_DIR / "fig02_tier_distribution.png")
    plt.close(fig)


def fig03_tier_by_sector(manifest: pd.DataFrame) -> None:
    """Stacked bar chart of tier within each sector."""
    ct = pd.crosstab(manifest["sector"], manifest["chunking_method"])
    ct = ct.reindex(columns=TIER_ORDER, fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottoms = [0] * len(ct_pct)
    for tier in TIER_ORDER:
        ax.bar(
            ct_pct.index,
            ct_pct[tier],
            bottom=bottoms,
            color=TIER_COLORS[tier],
            edgecolor="white",
            linewidth=1.5,
            label=TIER_LABELS[tier].replace("\n", " "),
        )
        for i, sector in enumerate(ct_pct.index):
            v = ct_pct[tier].iloc[i]
            if v > 3:
                ax.text(
                    i,
                    bottoms[i] + v / 2,
                    f"{v:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                    fontweight="bold",
                )
            bottoms[i] += v
    ax.set_ylabel("Percent of sector's filings")
    ax.set_title("Chunking tier distribution by sector")
    ax.set_ylim(0, 105)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        framealpha=0.95,
        fontsize=9,
    )
    ax.set_xticklabels([s.capitalize() for s in ct_pct.index])
    fig.savefig(FIGURES_DIR / "fig03_tier_by_sector.png")
    plt.close(fig)


def fig04_section_label_distribution() -> None:
    """Top-10 Items by chunk count across the full section-labeled corpus."""
    items: Counter = Counter()
    total = 0
    for f in CHUNKS_DIR.glob("*.jsonl"):
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                c = json.loads(line)
                if c.get("chunking_method") in {"section_aware", "hybrid_section_aware"}:
                    total += 1
                    sec = c.get("section_label", "")
                    if " - " in sec:
                        items[sec.split(" - ")[0]] += 1

    top10 = items.most_common(10)
    labels = [item for item, _ in top10][::-1]
    counts = [n for _, n in top10][::-1]
    pcts = [n / total * 100 for n in counts]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, counts, color="#1976D2", edgecolor="white")
    for bar, n, p in zip(bars, counts, pcts, strict=True):
        ax.text(
            n + max(counts) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{n:,} ({p:.1f}%)",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel("Number of chunks")
    ax.set_title(f"Section-label distribution across {total:,} section-labeled chunks")
    ax.set_xlim(0, max(counts) * 1.18)
    fig.savefig(FIGURES_DIR / "fig04_section_label_distribution.png")
    plt.close(fig)


def fig05_item7_chars_by_tier(manifest: pd.DataFrame) -> None:
    """Boxplot of Item 7 character count by tier."""
    data = [
        manifest[manifest["chunking_method"] == t]["item_7_chars"].values
        for t in TIER_ORDER
    ]
    labels = [TIER_LABELS[t] for t in TIER_ORDER]
    colors = [TIER_COLORS[t] for t in TIER_ORDER]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bp = ax.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.5)

    ax.set_yscale("symlog")
    ax.set_ylabel("Item 7 (MD&A) character count (symlog scale)")
    ax.set_title("Item 7 content by chunking tier")
    fig.savefig(FIGURES_DIR / "fig05_item7_chars_by_tier.png")
    plt.close(fig)


def fig06_failure_mode_distribution(manifest: pd.DataFrame) -> None:
    """Horizontal bar of failure modes, with 'none' separated."""
    fm = manifest["failure_mode"].value_counts()
    # Reorder: put 'none' last so the failure modes are prominent
    fm = fm.drop("none", errors="ignore")
    # Friendly labels
    rename = {
        "non_standard_format": "Non-standard format\n(MS, C, INTC)",
        "incorporation_by_reference": "Incorporation by reference\n(IBM, WFC)",
        "dominant_section_parser_failure": "Dominant-section parser failure\n(JPM, USB)",
        "large_item8_legit": "Large Item 8 (legitimate)\n(MET, PRU)",
        "parser_limitation_item7": "Item 7 parser limitation\n(MSFT, BAC)",
    }
    color_map = {
        "non_standard_format": "#C62828",
        "incorporation_by_reference": "#EF6C00",
        "dominant_section_parser_failure": "#F9A825",
        "large_item8_legit": "#2E7D32",
        "parser_limitation_item7": "#F9A825",
    }
    fm = fm.sort_values()
    labels = [rename.get(idx, idx) for idx in fm.index]
    colors = [color_map.get(idx, "#666666") for idx in fm.index]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(labels, fm.values, color=colors, edgecolor="white")
    for bar, n in zip(bars, fm.values, strict=True):
        ax.text(
            n + 0.15,
            bar.get_y() + bar.get_height() / 2,
            f"{n}",
            va="center",
            fontsize=10,
        )
    ax.set_xlabel("Number of filings")
    ax.set_title("Failure modes and recovery categories (43 non-clean filings)")
    ax.set_xlim(0, max(fm.values) * 1.18)
    fig.savefig(FIGURES_DIR / "fig06_failure_mode_distribution.png")
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    # Clear stale figures
    for existing in FIGURES_DIR.glob("fig*.png"):
        existing.unlink()
    print(f"Output directory: {FIGURES_DIR}")
    print()

    manifest = load_manifest()
    print(f"Loaded manifest: {len(manifest)} filings, {manifest['ticker'].nunique()} companies")
    print(f"Sectors: {manifest['sector'].value_counts().to_dict()}")
    print()

    print("Generating figures...")
    fig01_filing_size_by_sector(manifest); print("  fig01_filing_size_by_sector.png")
    fig02_tier_distribution(manifest); print("  fig02_tier_distribution.png")
    fig03_tier_by_sector(manifest); print("  fig03_tier_by_sector.png")
    fig04_section_label_distribution(); print("  fig04_section_label_distribution.png")
    fig05_item7_chars_by_tier(manifest); print("  fig05_item7_chars_by_tier.png")
    fig06_failure_mode_distribution(manifest); print("  fig06_failure_mode_distribution.png")
    print()
    print("Done.")


if __name__ == "__main__":
    main()