"""FastAPI service exposing the QA pipeline.

Owner: Data & Application Lead
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="FinRAG-Eval API", version="0.1.0")


class AskRequest(BaseModel):
    question: str
    config: str = "hybrid"
    k: int = 10


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, str]:
    # TODO(@data-lead): wire to Retriever + Generator
    return {"answer": f"[stub] would answer: {req.question}", "config": req.config}
