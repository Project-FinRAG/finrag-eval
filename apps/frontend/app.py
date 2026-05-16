"""Streamlit dashboard — demo + eval visualization.

Owner: Data & Application Lead
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="FinRAG-Eval", page_icon="📊", layout="wide")

st.title("FinRAG-Eval")
st.caption("Evaluation-first financial document intelligence over SEC filings.")

tab_demo, tab_eval = st.tabs(["Ask a Question", "Evaluation Dashboard"])

with tab_demo:
    st.subheader("Ask a question about SEC filings")
    question = st.text_input("Question", placeholder="What did Apple identify as its top risk factors?")
    config = st.selectbox("Retrieval config", ["bm25", "dense", "hybrid", "hybrid+rerank"])
    if st.button("Ask"):
        st.info("[stub] Would call /ask endpoint here")

with tab_eval:
    st.subheader("Configuration Comparison")
    st.info("[stub] Eval results table + charts go here")
