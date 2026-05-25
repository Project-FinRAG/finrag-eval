# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # FinRAG-Eval — Corpus EDA (v0.1)
#
# **Frozen corpus:** 198 SEC 10-K filings, 49 companies × 4 years, 59,822 chunks.
# **Methodology:** [ADR-0002 Quality-Aware Three-Tier Chunking](../docs/decisions/0002-quality-aware-chunking.md)
# **Reference:** [`data/metadata/corpus_manifest_v0.1.csv`](../data/metadata/corpus_manifest_v0.1.csv)
#
# This notebook characterizes the corpus that powers FinRAG-Eval's retrieval
# and evaluation work. The narrative arc:
#
# 1. **What's in the corpus** — scale, composition, filing-size distribution
# 2. **The three-tier chunking story** — why a binary section-aware vs fixed-size
#    classification was insufficient, and what we replaced it with
# 3. **Methodology validation** — temporal stability of failure modes, sanity
#    checks on the relaxed dominance gate
# 4. **Implications for downstream work** — what retrieval and evaluation should
#    do with this corpus
#
# All numbers cited in the markdown narrative are computed in the code cells
# directly below them. If you re-run the notebook against an updated corpus,
# regenerate the manifest first via `scripts/generate_corpus_manifest.py`.

# %% [markdown]
# ## Setup

# %%
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110

CORPUS_STATS = Path("../data/processed/corpus_stats.csv")
MANIFEST = Path("../data/metadata/corpus_manifest_v0.1.csv")
CHUNKS_DIR = Path("../data/processed/chunks")

# Authoritative sector taxonomy for the v0.1 corpus.
# Note: the sector map in scripts/05_corpus_scale_out.py is incomplete (omits
# AON, APO, BX, CB, KKR, CDNS, SNPS). This notebook's classification is the
# authoritative one for analysis; a future PR should backport this into the
# ingestion script.
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


def sector_of(ticker: str) -> str:
    if ticker in TECH:
        return "tech"
    if ticker in FINANCIAL:
        return "financial"
    return "other"


stats = pd.read_csv(CORPUS_STATS)
manifest = pd.read_csv(MANIFEST)
manifest["sector"] = manifest["ticker"].apply(sector_of)

assert manifest["sector"].ne("other").all(), \
    f"Uncategorized tickers: {sorted(manifest[manifest['sector']=='other']['ticker'].unique())}"

print(f"Loaded {len(stats)} filing stats rows and {len(manifest)} manifest rows.")
print(f"Sector coverage complete: {manifest['sector'].value_counts().to_dict()}")

# %% [markdown]
# ## Section 1 — Corpus at a glance
#
# The v0.1 corpus targets the most-traded U.S. equities across two sectors
# where 10-K filings are most analytically valuable: large-cap technology
# and financial services. We pulled four fiscal years per company where
# available, producing 198 filings total.
#
# **Scale:**
# - 198 filings
# - 49 companies (27 tech, 31 financial — note Berkshire is counted as financial)
# - Fiscal years 2022–2026 (4 years per company is typical; some companies
#   have 3 or 5 due to differing fiscal calendars and FY2026 still rolling in)
# - 59,822 chunks
# - 117 million characters of cleaned filing text

# %%
print(f"Filings:     {len(manifest)}")
print(f"Companies:   {manifest['ticker'].nunique()}")
print(f"Years:       {sorted(manifest['filing_year'].unique())}")
print(f"Chunks:      {stats['chunks_produced'].sum():,}")
print(f"Text chars:  {stats['total_chars'].sum():,}")
print(f"Mean chunks/filing: {stats['chunks_produced'].mean():.1f}")

# %% [markdown]
# ### 1a. Distribution by sector and year

# %%
sector_year = manifest.pivot_table(
    index="sector", columns="filing_year",
    values="ticker", aggfunc="count", fill_value=0,
)
sector_year["TOTAL"] = sector_year.sum(axis=1)
print("Filings by sector × year:")
print(sector_year)

# %% [markdown]
# Filings-per-year varies modestly: 2022 and 2026 are tail years
# (some companies hadn't been in the corpus yet, others haven't filed FY2026 yet).
# The middle years (2023, 2024, 2025) are nearly complete for the full set.

# %% [markdown]
# ### 1b. Filing size distribution
#
# Financial filings — especially from large banks and insurers — are
# dramatically longer than tech filings. The financial sector averages
# ~780K characters per filing vs ~360K for tech.

# %%
print("Filing size by sector (chars):")
print(manifest.groupby("sector")["total_chars"].agg(
    count="count", mean="mean", median="median", min="min", max="max"
).round(0))

# %%
fig, ax = plt.subplots(figsize=(10, 4))
for sector in ["tech", "financial"]:
    sub = manifest[manifest["sector"] == sector]
    ax.hist(sub["total_chars"] / 1_000_000, bins=25, alpha=0.6,
            label=f"{sector} (n={len(sub)})", edgecolor="white")
ax.set_xlabel("Filing size (million chars)")
ax.set_ylabel("Number of filings")
ax.set_title("Filing size distribution by sector")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 1c. The largest and smallest filings

# %%
print("Largest 10 filings:")
print(manifest.nlargest(10, "total_chars")[
    ["ticker", "filing_year", "total_chars", "chunking_method"]
].to_string(index=False))

print("\nSmallest 10 filings:")
print(manifest.nsmallest(10, "total_chars")[
    ["ticker", "filing_year", "total_chars", "chunking_method"]
].to_string(index=False))

# %% [markdown]
# The largest filings are insurance and asset-management 10-Ks (MET, PRU,
# BRK-B) — driven by their detailed Item 8 (Financial Statements) sections.
# The smallest filings are tech 10-Ks like AAPL and NVDA, which are
# substantially shorter despite still covering all required Items.

# %% [markdown]
# ## Section 2 — The three-tier chunking story
#
# This is the methodology section. The naive approach to chunking 10-Ks is
# binary: try to detect Item boundaries; if section detection passes quality
# gates, emit semantically-labeled chunks; otherwise fall back to fixed-size
# 512-token windows.
#
# Our v0 pipeline used that binary split and produced an 80/20 result on the
# initial 100-filing corpus. Empirical investigation of the 20% revealed that
# treating them as homogeneous fallbacks discarded substantial usable
# information. ADR-0002 documents the redesign — what follows visualizes the
# outcome.

# %% [markdown]
# ### 2a. Tier distribution

# %%
tier_counts = manifest["chunking_method"].value_counts()
tier_pct = (tier_counts / len(manifest) * 100).round(1)
tier_summary = pd.DataFrame({
    "count": tier_counts,
    "share_%": tier_pct,
})
tier_summary.loc["TOTAL"] = [len(manifest), 100.0]
print(tier_summary)

labeled_count = (manifest["chunking_method"] != "fixed_size").sum()
print(f"\nTotal section-labeled: {labeled_count}/{len(manifest)} "
      f"({labeled_count / len(manifest):.1%})")

# %% [markdown]
# **Three tiers:**
# - `section_aware` (163, 82.3%): All strict quality gates pass — Item
#   boundaries are reliable, every Item is well-populated
# - `hybrid_section_aware` (23, 11.6%): At least one strict gate failed,
#   but the section map is still usable for most Items
# - `fixed_size` (12, 6.1%): Section detection failed entirely — falls back
#   to positional 512-token windows
#
# 93.9% of filings carry section labels in v0.1 (up from 80% in the binary
# classification). The 6.1% remaining fallbacks are characterized by company
# and root cause rather than left as opaque failures.

# %% [markdown]
# ### 2b. Tier distribution by sector
#
# Are the failures concentrated in one sector? If so, the methodology has
# different implications for tech-focused vs finance-focused retrieval.

# %%
sector_tier = pd.crosstab(manifest["sector"], manifest["chunking_method"])
sector_tier_pct = sector_tier.div(sector_tier.sum(axis=1), axis=0) * 100
print("Tier distribution by sector (count):")
print(sector_tier)
print("\nTier distribution by sector (% within sector):")
print(sector_tier_pct.round(1))

# %% [markdown]
# Failures skew financial — 8 of 12 fixed-size filings and 15 of 23 hybrid
# filings are financial sector. Tech filings parse cleanly more often
# (87% strict section_aware) because their structure is more uniform.
# Financial filings have more variation in formatting conventions and more
# legitimate cases where Item 8 (Financial Statements) is huge.

# %% [markdown]
# ### 2c. The four failure modes
#
# ADR-0002 names four root causes for non-section_aware filings.
# This is the methodology contribution: rather than discarding 12% of the
# corpus as opaque "fallbacks," we characterize what went wrong and why.

# %%
fm_counts = manifest["failure_mode"].value_counts()
print("Failure mode distribution:")
print(fm_counts)
print()
print("Failure mode × chunking_method:")
print(pd.crosstab(manifest["failure_mode"], manifest["chunking_method"]))

# %% [markdown]
# **The four named failure modes, by company:**
#
# | Failure mode | Filings | Companies | What's broken | What's usable |
# |---|---|---|---|---|
# | `non_standard_format` | 12 | MS, C, INTC (4 yrs each) | Filing uses a visual section hierarchy that doesn't match the SEC's "Item N." prefix convention; parser detects zero sections | Nothing structural — chunks emit as fixed-size positional windows |
# | `incorporation_by_reference` | 8 | IBM, WFC (4 yrs each) | MD&A (Item 7) and Financial Statements (Item 8) are filed separately in the annual report; only stubs in the 10-K | Item 1 (Business), Item 1A (Risk Factors) are full |
# | `dominant_section_parser_failure` | 8 | JPM, USB (4 yrs each) | Item 15 (Exhibits) header is detected but Item 8 boundary is missed, so Item 15 swallows financial-statement content | Items 1, 1A reliable; the dominant section reflects parser failure, not real document structure |
# | `large_item8_legit` | 8 | MET, PRU (4 yrs each) | Not a failure — Item 8 is genuinely >50% of the filing because the company is an insurer with massive financial-statement disclosures | Everything; passes via the relaxed dominance gate |
# | `parser_limitation_item7` | 7 | MSFT (4 yrs), BAC (3 yrs) | Real Item 7 header exists but parser matched a table-of-contents entry first (~2K chars, not real MD&A) | Item 1, 1A, 8 all reliable; only Item 7 affected |
#
# Note that BAC has 3 hybrid years and 1 strict section_aware year — failure
# modes aren't always 4/4 within a company. The other 7 multi-year failure
# cases are perfectly stable across all 4 years.

# %% [markdown]
# ### 2d. Diagnostic: Item 7 character counts by tier
#
# Item 7 (MD&A) is the most-asked-about section in financial QA. Whether
# our chunker can find real Item 7 content is a useful proxy for tier
# quality. Here's the distribution of Item 7 chars per filing, by tier:

# %%
item7_by_tier = manifest.groupby("chunking_method")["item_7_chars"].agg(
    count="count", mean="mean", median="median", min="min", max="max"
).round(0)
print("Item 7 chars by chunking_method:")
print(item7_by_tier)

# %% [markdown]
# The contrast is sharp:
#
# - **section_aware** filings have a median of 67K chars of Item 7 content
#   (and a mean of 112K — pulled higher by the dense MD&A sections in
#   financial filings)
# - **hybrid_section_aware** filings have a median of 322 chars of Item 7
#   (a TOC stub or "incorporated by reference" pointer)
# - **fixed_size** filings have 0 chars of Item 7 (no sections detected)
#
# This is exactly the signal we want: the tier label predicts Item 7
# usability. Downstream consumers (QA-pair authors, the retrieval evaluator)
# can filter by tier and know what to expect.

# %% [markdown]
# ## Section 3 — Methodology validation
#
# Three questions to answer about the three-tier classification:
#
# 1. **Temporal stability:** Are failures year-specific (a one-off bad parse)
#    or company-specific (the company always files in a way our parser
#    can't handle)? If failures are temporal, our methodology generalizes
#    poorly to future filings; if they're company-specific, we know exactly
#    when to expect them.
# 2. **Dominance gate sanity:** Does the relaxed dominance gate (allowing
#    Item 8 up to 75%) admit filings that *should* have been rejected?
# 3. **Corpus expansion:** Did going from 100 to 198 filings reveal new
#    failure modes, or are the four named modes exhaustive?

# %% [markdown]
# ### 3a. Temporal stability of failure modes

# %%
# For each non-clean company, show year-by-year tier assignment
problem_tickers = manifest[manifest["failure_mode"] != "none"]["ticker"].unique()
year_tier = manifest[manifest["ticker"].isin(problem_tickers)].pivot_table(
    index="ticker", columns="filing_year", values="chunking_method",
    aggfunc="first",
)
print("Tier per year for companies with any non-section_aware filing:")
print(year_tier.fillna("-"))

# %% [markdown]
# **Result: failures are overwhelmingly company-specific, not year-specific.**
#
# - 7 of 8 multi-year failure companies fail the same way in every year of
#   their corpus history (4-out-of-4 consistency)
# - The one exception is BAC, which has 3 failed years and 1 passing year —
#   suggesting their filing format varied between years
#
# This is a positive finding. It means the four failure modes generalize:
# when v0.2 of the corpus adds future filings from these companies, we can
# predict which tier they'll fall into. It also means the methodology
# describes structural realities of the filings, not parser flakiness.

# %% [markdown]
# ### 3b. Dominance gate sanity check
#
# The relaxed gate allows Item 8 (or Item 15) to exceed 50% of the document
# *if* Item 7 is healthy (≥5,000 chars) and at least 12 sections were
# detected. Without this relaxation, MET and PRU's filings would have been
# rejected as section_aware despite being well-structured.

# %%
relaxed = manifest[manifest["failure_mode"] == "large_item8_legit"]
print(f"Filings recovered by relaxed dominance gate: {len(relaxed)}")
print("\nDiagnostic columns for these filings:")
print(relaxed[[
    "ticker", "filing_year", "largest_section_item",
    "largest_section_ratio", "item_7_chars", "sections_detected",
]].to_string(index=False))

# %% [markdown]
# **Result: the relaxed gate is doing exactly what it should.**
#
# Each of the 8 recovered filings:
# - Has Item 8 (Financial Statements) as the dominant section
# - Has Item 7 (MD&A) in healthy condition (200K+ chars)
# - Has 22–23 sections detected (well above the 12-section floor)
# - Has largest-section ratio of 0.48–0.55, comfortably below the 0.75 ceiling
#
# No filings sneak in via the relaxed gate; the criteria are strict enough
# that genuine parser failures (e.g., JPM/USB with Item 15 swallowing Item 8)
# are still routed to the hybrid tier.

# %% [markdown]
# ### 3c. Did corpus expansion reveal new failure modes?
#
# v0 of the corpus had 100 filings (50 companies × 2 years) and identified
# the four failure modes after manual diagnostic work. v0.1 doubled the
# corpus to 198 filings. Did the larger corpus introduce any new failure
# modes that didn't appear in the smaller sample?

# %%
uncharacterized = manifest[manifest["failure_mode"] == "uncharacterized"]
print(f"Uncharacterized non-clean filings: {len(uncharacterized)}")
if len(uncharacterized) > 0:
    print(uncharacterized[[
        "ticker", "filing_year", "chunking_method", "quality_check_reason"
    ]].to_string(index=False))
else:
    print("All non-clean filings are covered by the four named failure modes.")

# %% [markdown]
# Zero uncharacterized filings. The four failure modes captured every
# non-clean case in v0.1 — corpus expansion did not surface a fifth failure
# class. This isn't proof the taxonomy is complete (v0.2 might find new
# patterns), but it's evidence that the named modes describe the dominant
# failure structures.

# %% [markdown]
# ## Section 4 — Implications for retrieval and evaluation
#
# The three-tier structure isn't just descriptive — it actively shapes
# the downstream workstreams.

# %% [markdown]
# ### 4a. Section-label distribution across section_aware chunks
#
# Section-aware chunks carry semantic Item labels. The distribution
# across Items determines what kinds of QA pairs are well-supported.

# %%
sample_files = [f for f in CHUNKS_DIR.glob("*.jsonl")
                if not f.name.startswith(("MS_", "C_", "INTC_"))][:30]
items = Counter()
total_chunks = 0
for f in sample_files:
    with f.open() as fh:
        for line in fh:
            c = json.loads(line)
            if c.get("chunking_method") == "section_aware":
                total_chunks += 1
                sec = c.get("section_label", "")
                if " - " in sec:
                    items[sec.split(" - ")[0]] += 1

print(f"Sample: {len(sample_files)} files, {total_chunks:,} section_aware chunks")
print("\nTop 10 section labels by chunk count:")
for item, n in items.most_common(10):
    print(f"  {item:25s} {n:5d}  ({n/total_chunks:.1%})")

# %% [markdown]
# Item 8 (Financial Statements) dominates by chunk count — roughly 37% of
# section_aware chunks come from Item 8. Item 7 (MD&A), Item 1A (Risk Factors),
# and Item 1 (Business) follow. This distribution matches the analytical
# content of 10-Ks: Financial Statements and MD&A together account for
# more than half of section_aware retrievable text.

# %% [markdown]
# ### 4b. What this means for the QA workstream
#
# - **First ~50 QA pairs** should target `section_aware` filings exclusively
#   to validate the retrieval system on the highest-quality subset.
#   AAPL, GOOGL, GS, NVDA, MA, V are reliable choices.
# - **Section-specific questions** (e.g., "What did Apple disclose about
#   cybersecurity?") work best on `section_aware` filings; `hybrid_section_aware`
#   filings can be used but the QA author must verify the relevant Item is
#   not in the affected section
# - **Item 7 questions** should avoid MSFT, BAC, IBM, WFC, JPM, USB (all
#   compromised in MD&A)
# - **The 12 fixed_size filings** (MS, C, INTC across all years) are
#   excluded from section-specific QA — useable only for fully-free-form
#   questions where retrieval-strategy matters less

# %% [markdown]
# ### 4c. What this means for the retrieval workstream
#
# The retrieval comparison (section-aware vs fixed-size) operates on
# overlapping subsets of the corpus:
#
# - **Primary experiment:** Section-aware retrieval on the 186 section-labeled
#   filings (163 strict + 23 hybrid) versus fixed-size retrieval on the
#   same 186, evaluated on QA pairs that reference those filings
# - **Robustness check:** Restrict to the 163 strict section_aware filings
#   only, confirm the comparison result holds
# - **Baseline:** Fixed-size retrieval on all 198 filings (the experimental
#   condition that doesn't use section information at all)
#
# Vidhee's BM25 retriever supports this via the `labeled` / `strict` /
# `fixed_size` / `all` chunk-loading strategies that map directly to
# the tier taxonomy.

# %% [markdown]
# ## Headline findings
#
# What goes in the final report's methodology section:
#
# 1. **Corpus scale:** 198 SEC 10-K filings, 49 companies × 4 years, 60K
#    chunks across 117M characters of cleaned text. Tech and financial
#    sectors only; fiscal years 2022–2026.
#
# 2. **Quality-aware three-tier chunking:** Filings classified as
#    `section_aware` (163, 82.3%), `hybrid_section_aware` (23, 11.6%), or
#    `fixed_size` (12, 6.1%). Total 93.9% section-labeled — a meaningful
#    improvement over the 80% baseline of the binary classification.
#
# 3. **Four named failure modes:** Non-clean filings characterized by
#    structural root cause: non-standard format (MS, C, INTC),
#    incorporation by reference (IBM, WFC), dominant-section parser
#    failure (JPM, USB), and parser limitation on Item 7 (MSFT, BAC).
#    Plus 8 filings legitimately recovered by the relaxed dominance gate
#    (MET, PRU as insurers with genuinely large Item 8).
#
# 4. **Temporal stability:** Failure modes are company-specific, not
#    year-specific. 7 of 8 multi-year failure companies fail identically
#    across all 4 years. This means the taxonomy generalizes — future
#    filings from these companies are predictable in tier assignment.
#
# 5. **Tier label is informative:** Median Item 7 character count is 67K
#    for `section_aware` filings, 322 for `hybrid_section_aware`, 0 for
#    `fixed_size`. The tier label predicts content usability cleanly.
#
# 6. **Production handoff:** The per-filing manifest at
#    `data/metadata/corpus_manifest_v0.1.csv` is the stable contract for
#    downstream code. Retrieval filters on `chunking_method`; QA pair
#    authors filter on `failure_mode`. Both workstreams have explicit
#    guidance per tier.