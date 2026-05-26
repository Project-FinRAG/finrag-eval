"""Dense retriever using OpenAI embeddings + ChromaDB.

Owner: Retrieval & Modeling Lead (interim)

Uses OpenAI's text-embedding-3-small (1536-dim) for embeddings and ChromaDB
for the persistent vector index. The retriever satisfies the Retriever
Protocol (`name`, `index`, `load`, `retrieve`) and is a drop-in alternative
to BM25Retriever.

Indexing 60K chunks costs ~$0.60 against OpenAI's API and takes ~5 minutes.
Per-query embedding adds ~50ms of API latency to retrieval.
"""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Any, cast

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.config import Settings
from dotenv import load_dotenv
from openai import APIError, OpenAI, RateLimitError

from finrag_eval.common import Chunk, RetrievalResult

INDEX_DIR = Path("data/indexes")
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_BATCH_SIZE = 100
MAX_RETRIES = 5


def _load_api_key() -> str:
    """Load OPENAI_API_KEY from environment or .env file."""
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. Either export it in your shell or "
            "add it to a .env file in the repo root:\n"
            "  OPENAI_API_KEY=sk-proj-...\n"
        )
    return api_key


class DenseRetriever:
    """Dense retriever backed by OpenAI embeddings and ChromaDB.

    Satisfies the Retriever Protocol — interchangeable with BM25Retriever
    and HybridRetriever in the eval harness.
    """

    name = "dense"

    def __init__(
        self,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        index_path: Path | None = None,
        collection_name: str = "finrag_dense",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.embedding_model = embedding_model
        self.index_path = index_path or INDEX_DIR / "chroma_dense"
        self.collection_name = collection_name
        self.batch_size = batch_size

        self._client: OpenAI | None = None
        self._chroma: ClientAPI | None = None
        self._collection: Collection | None = None
        self._last_latency_ms: float = 0.0

    # ---- internal helpers ------------------------------------------------

    def _ensure_openai(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=_load_api_key())
        return self._client

    def _ensure_chroma(self) -> ClientAPI:
        if self._chroma is None:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(
                path=str(self.index_path),
                settings=Settings(anonymized_telemetry=False),
            )
        return self._chroma

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts with retry-on-rate-limit."""
        client = self._ensure_openai()

        for attempt in range(MAX_RETRIES):
            try:
                resp = client.embeddings.create(
                    model=self.embedding_model,
                    input=texts,
                )
                return [d.embedding for d in resp.data]
            except RateLimitError:
                wait = 2**attempt
                print(f"  rate-limited, sleeping {wait}s (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(wait)
            except APIError as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                wait = 2**attempt
                print(f"  API error ({e!r}), retrying in {wait}s")
                time.sleep(wait)

        raise RuntimeError(f"Failed to embed batch after {MAX_RETRIES} attempts")

    # ---- public API (Retriever Protocol) ---------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """Embed chunks in batches, upsert into ChromaDB. Persists to disk."""
        chroma = self._ensure_chroma()
        # Recreate collection if it exists (avoid mixing old + new embeddings)
        with contextlib.suppress(Exception):
            chroma.delete_collection(self.collection_name)
        self._collection = chroma.create_collection(
            name=self.collection_name,
            metadata={
                "embedding_model": self.embedding_model,
                "hnsw:space": "cosine",
            },
        )

        n = len(chunks)
        print(f"Embedding {n:,} chunks in batches of {self.batch_size}...")
        t0 = time.perf_counter()

        for start in range(0, n, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embed_batch(texts)

            self._collection.add(
                ids=[c.chunk_id for c in batch],
                documents=texts,
                embeddings=embeddings,  # type: ignore[arg-type]
                metadatas=[
                    {
                        "ticker": c.ticker,
                        "filing_accession": c.filing_accession,
                        "filing_type": str(c.filing_type),
                        "section": c.section or "",
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                        "token_count": c.token_count,
                    }
                    for c in batch
                ],
            )

            done = min(start + self.batch_size, n)
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta_s = (n - done) / rate if rate > 0 else 0
            print(f"  {done:>6,}/{n:,} chunks  ({rate:.0f} chunks/s, ETA {eta_s:.0f}s)")

        total = time.perf_counter() - t0
        print(f"Indexed {n:,} chunks in {total:.0f}s ({total / 60:.1f} min)")
        print(f"Persisted to {self.index_path}")

    def load(self) -> None:
        """Connect to existing persistent ChromaDB collection."""
        chroma = self._ensure_chroma()
        try:
            self._collection = chroma.get_collection(self.collection_name)
        except Exception as e:
            raise RuntimeError(
                f"No collection named {self.collection_name!r} at {self.index_path}. "
                "Run .index() first to build the index."
            ) from e

    def retrieve(self, query: str, k: int = 10) -> list[RetrievalResult]:
        """Embed the query, query Chroma for top-k nearest neighbors."""
        if self._collection is None:
            raise RuntimeError("Index not built or loaded. Call .index() or .load() first.")

        t0 = time.perf_counter()

        # Embed the query (single text, but use the batch path for consistency)
        query_embedding = self._embed_batch([query])[0]

        # Query the collection
        result = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns lists-of-lists (one inner list per query); we only have one.
        # The query returns typed dict where docs/metadatas/distances are Optional;
        # they're always present given our include= args, so cast for mypy.
        ids: list[str] = result["ids"][0]
        # result["documents"], ["metadatas"], ["distances"] are Optional in the
        # type stubs, but always present given our include= args. Cast each
        # inner list explicitly so mypy understands the unpacking that follows.
        docs = cast(list[str], result["documents"][0])  # type: ignore[index]
        metas = cast(list[dict[str, Any]], result["metadatas"][0])  # type: ignore[index]
        distances = cast(list[float], result["distances"][0])  # type: ignore[index]

        # Build RetrievalResult objects. Chroma's cosine distance is 1 - cos_sim,
        # so score = 1 - distance (higher = more relevant).
        from finrag_eval.common.types import FilingType

        results: list[RetrievalResult] = []
        for rank, (cid, doc, meta, dist) in enumerate(
            zip(ids, docs, metas, distances, strict=True), start=1
        ):
            chunk = Chunk(
                chunk_id=cid,
                filing_accession=meta["filing_accession"],
                ticker=meta["ticker"],
                filing_type=FilingType(meta["filing_type"]),
                section=meta.get("section") or None,
                text=doc,
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", len(doc))),
                token_count=int(meta.get("token_count", 0)),
            )
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=1.0 - float(dist),
                    rank=rank,
                )
            )

        self._last_latency_ms = (time.perf_counter() - t0) * 1000
        return results

    def latency_ms(self) -> float:
        return self._last_latency_ms
