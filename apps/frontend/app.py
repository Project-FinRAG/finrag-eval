"""Streamlit demo — FinRAG-Eval.

The demonstration interface from the project plan: answers analyst-style questions
over SEC filings using the strongest-performing configuration (dense +
text-embedding-3-large + ticker pre-filter), with transparent source traces. Runs
the same Generator the evaluation harness scores, so the demo IS the evaluated system.

Owner: Data & Application Lead

Run:
    uv run streamlit run apps/frontend/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from finrag_eval.common.config import settings
from finrag_eval.retrieval.bm25 import load_chunks_from_jsonl
from finrag_eval.retrieval.dense import DenseRetriever
from finrag_eval.synthesis.generator import Generator

LARGE_INDEX = Path("data/indexes/chroma_dense_large_labeled")
LARGE_COLLECTION = "finrag_dense_large"
EMBED_MODEL = "text-embedding-3-large"
TOP_K = 10
ALL = "__ALL__"

NAMES = {
    "AAPL": "Apple", "ADBE": "Adobe", "AIG": "American International Group",
    "ALL": "Allstate", "AMAT": "Applied Materials", "AMD": "AMD", "AMZN": "Amazon",
    "ANET": "Arista Networks", "AON": "Aon", "APO": "Apollo Global", "AVGO": "Broadcom",
    "AXP": "American Express", "BAC": "Bank of America", "BLK": "BlackRock",
    "BX": "Blackstone", "CB": "Chubb", "CDNS": "Cadence", "CME": "CME Group",
    "COF": "Capital One", "CRM": "Salesforce", "CSCO": "Cisco", "GOOGL": "Alphabet (Google)",
    "GS": "Goldman Sachs", "IBM": "IBM", "INTU": "Intuit", "JPM": "JPMorgan Chase",
    "KKR": "KKR", "KLAC": "KLA", "MET": "MetLife", "META": "Meta", "MSFT": "Microsoft",
    "MU": "Micron", "NOW": "ServiceNow", "NVDA": "NVIDIA", "ORCL": "Oracle",
    "PANW": "Palo Alto Networks", "PNC": "PNC Financial", "PRU": "Prudential",
    "QCOM": "Qualcomm", "SCHW": "Charles Schwab", "SNPS": "Synopsys", "TFC": "Truist",
    "TRV": "Travelers", "TXN": "Texas Instruments", "USB": "U.S. Bancorp",
    "WFC": "Wells Fargo", "WTW": "Willis Towers Watson",
}

st.set_page_config(page_title="FinRAG-Eval", page_icon="📊", layout="centered")


@st.cache_resource(show_spinner="Connecting to the 3-large index…")
def get_retriever() -> DenseRetriever:
    r = DenseRetriever(
        embedding_model=EMBED_MODEL,
        index_path=LARGE_INDEX,
        collection_name=LARGE_COLLECTION,
    )
    r.load()
    return r


@st.cache_resource(show_spinner="Loading generator…")
def get_generator() -> Generator:
    return Generator()


@st.cache_resource(show_spinner="Indexing company tickers…")
def get_tickers() -> list[str]:
    return sorted({c.ticker for c in load_chunks_from_jsonl("labeled")})


def company_label(ticker: str) -> str:
    if ticker == ALL:
        return "All companies (no filter)"
    return f"{NAMES.get(ticker, ticker)} ({ticker})"


def trace_expander(rank: int, r) -> None:  # noqa: ANN001 — RetrievalResult
    ch = r.chunk
    trace = f"#{rank} · {ch.ticker} · {ch.filing_type} · {ch.filing_accession}"
    if ch.section:
        trace += f" · {ch.section}"
    with st.expander(trace):
        st.text(ch.text)


st.title("FinRAG-Eval")
st.caption("Evaluation-first financial document intelligence over SEC filings.")

if not settings.openai_api_key:
    st.error("OPENAI_API_KEY is not set — add it to .env.")
    st.stop()

tab_demo, tab_eval = st.tabs(["Ask a Question", "Evaluation Results"])

with tab_demo:
    company = st.selectbox(
        "Company",
        [ALL, *get_tickers()],
        format_func=company_label,
        help="Only these companies are in the corpus (tech + financials). Type to search.",
    )
    question = st.text_input(
        "Question",
        placeholder="What did the company identify as its top risk factors?",
    )

    if st.button("Ask", type="primary") and question.strip():
        if not LARGE_INDEX.exists():
            st.error(f"3-large index not found at {LARGE_INDEX}. Build it with scripts/15.")
            st.stop()
        where = None if company == ALL else {"ticker": company}
        try:
            retriever = get_retriever()
            generator = get_generator()
            with st.spinner("Retrieving evidence…"):
                results = retriever.retrieve(question, k=TOP_K, where=where)
            with st.spinner("Generating grounded answer…"):
                answer = generator.answer(question, results)
        except Exception as exc:  # noqa: BLE001 — surface any failure in the UI
            st.error(f"Pipeline error: {exc}")
            st.stop()

        if answer.abstained or not answer.answer_text.strip():
            st.warning("Abstained — the retrieved passages lacked sufficient evidence to answer.")
        else:
            st.write(answer.answer_text)

        c1, c2, c3 = st.columns(3)
        c1.metric("Company", "All" if company == ALL else company)
        c2.metric("Passages", len(results))
        c3.metric("Cost", f"${answer.cost_usd:.4f}")

        st.divider()
        ranked = list(enumerate(results, 1))
        cited_ids = {c.chunk_id for c in answer.citations}
        cited_items = [(i, r) for i, r in ranked if r.chunk.chunk_id in cited_ids]
        st.caption(f"Sources — {len(cited_items)} of {len(results)} passages cited")
        for i, r in cited_items:
            trace_expander(i, r)
        if st.toggle("Show all retrieved passages"):
            for i, r in ranked:
                if r.chunk.chunk_id not in cited_ids:
                    trace_expander(i, r)

with tab_eval:
    m1, m2, m3 = st.columns(3)
    m1.metric("Best soft-recall@10", "0.590")
    m2.metric("Best correctness", "0.561")
    m3.metric("Judge κ (weighted)", "0.70")
    st.caption("Comparative evaluation over n=70 questions. Full analysis in the final report.")

    st.subheader("End-to-end answer quality")
    e2e = pd.DataFrame(
        {
            "retriever": ["bm25", "dense (3-small)", "dense (3-large)"],
            "correctness": [0.249, 0.447, 0.561],
            "faithfulness": [0.700, 0.749, 0.867],
            "gen cost (70 Q)": ["$1.02", "$0.98", "$1.04"],
        }
    ).set_index("retriever")
    st.table(e2e)
    st.bar_chart(e2e[["correctness", "faithfulness"]])
    st.caption("Correctness tracks retrieval quality at flat cost; faithfulness leads throughout — retrieval is the bottleneck.")

    with st.expander("Full retrieval comparison (soft-recall@10)"):
        retr = pd.DataFrame(
            {
                "configuration": [
                    "bm25", "hybrid fusion", "cross-encoder rerank", "HyDE (entity-named)",
                    "dense (3-small)", "dense (3-large)", "dense (3-large) + ticker filter",
                ],
                "soft-recall@10": [0.129, 0.286, 0.326, 0.507, 0.371, 0.533, 0.590],
                "verdict": [
                    "vocabulary mismatch", "dilutes dense", "Pareto-dominated", "wash, +cost",
                    "baseline", "biggest win", "best config",
                ],
            }
        ).set_index("configuration")
        st.table(retr)
        st.caption("Every generative/fusion addition hurt; only embedding quality and the ticker filter helped.")