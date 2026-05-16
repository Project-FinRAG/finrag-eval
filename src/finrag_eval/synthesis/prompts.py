"""Prompt templates for the synthesis layer.

Owner: Retrieval & Modeling Lead

Two prompts:
    QA_PROMPT: standard grounded QA with required citations
    ABSTENTION_PROMPT: explicit instruction to abstain when evidence is weak

Both must produce JSON-structured output for reliable parsing.
"""

QA_PROMPT = """You are a financial analyst assistant. Answer the user's question
using ONLY the provided SEC filing passages. Cite the specific passages that
support each claim. If the passages don't contain enough information to answer
confidently, abstain and explain what's missing.

Question: {question}

Passages:
{passages}

Return a JSON object with these fields:
  - "answer": your answer (string)
  - "citations": list of passage IDs that support the answer
  - "abstain": true if you cannot answer from the passages
  - "abstention_reason": null if abstain=false, otherwise a short reason

Output the JSON object only, with no surrounding prose.
"""

ABSTENTION_PROMPT = """You are evaluating whether the provided passages contain
sufficient evidence to answer the question. Be strict: if the answer requires
inference beyond what the passages directly state, you must abstain.

Question: {question}

Passages:
{passages}

Return JSON: {{"sufficient_evidence": true|false, "reason": "..."}}
"""
