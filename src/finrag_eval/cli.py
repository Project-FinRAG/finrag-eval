"""FinRAG-Eval CLI — entry point for ingest, index, ask, eval commands."""

from __future__ import annotations

import typer

app = typer.Typer(help="FinRAG-Eval: evaluation-first financial RAG.")


@app.command()
def ingest(
    companies: str = typer.Option(..., help="Comma-separated tickers, e.g. AAPL,MSFT"),
    years: str = typer.Option(..., help="Comma-separated years, e.g. 2022,2023,2024"),
    filing_types: str = typer.Option("10-K,10-Q", help="Filing types to fetch"),
) -> None:
    """Fetch filings from EDGAR for the given companies and years."""
    typer.echo(f"[stub] Would ingest {companies} for years {years}")
    # TODO: wire up EdgarClient


@app.command()
def index(
    chunker: str = typer.Option("section_aware", help="fixed_size or section_aware"),
) -> None:
    """Build retrieval indexes from ingested filings."""
    typer.echo(f"[stub] Would build indexes with {chunker} chunker")


@app.command()
def ask(
    question: str = typer.Option(..., "--question", "-q"),
    config: str = typer.Option("hybrid", help="Retrieval config name"),
) -> None:
    """Ask a single question against the indexed corpus."""
    typer.echo(f"[stub] Would answer: {question!r} using config={config}")


@app.command(name="eval")
def run_eval(
    config: str = typer.Option(None, help="Single config name; omit with --all-configs"),
    all_configs: bool = typer.Option(False, "--all-configs"),
    smoke: bool = typer.Option(False, help="Run a small smoke test"),
    max_questions: int = typer.Option(None, help="Limit number of questions"),
    output: str = typer.Option("eval-results.json"),
) -> None:
    """Run the evaluation suite."""
    typer.echo(f"[stub] Would run eval: config={config} all={all_configs} smoke={smoke}")


if __name__ == "__main__":
    app()
