"""Synthesis package — turn retrieved passages into grounded answers.

Owner: Retrieval & Modeling Lead (shared)

Public interface:
    - Generator: produces an Answer from a query and retrieved passages
    - Prompts: prompt templates for QA and abstention
"""

from finrag_eval.synthesis.generator import Generator
from finrag_eval.synthesis.prompts import ABSTENTION_PROMPT, QA_PROMPT

__all__ = ["ABSTENTION_PROMPT", "QA_PROMPT", "Generator"]
