"""End-to-end ingestion pipeline CLI.

Orchestrates: arXiv fetch → optional PDF parse → chunk → embed → FAISS upsert.

Usage:
    python -m src.ingestion.pipeline --max-results 500
    python -m src.ingestion.pipeline --max-results 100 --full-pdf
    python -m src.ingestion.pipeline --categories hep-ex hep-ph --days-back 30
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.config import settings

app = typer.Typer(
    name="cern-ingest",
    help="Ingest CERN/HEP papers from arXiv into the FAISS vector store.",
    add_completion=False,
)
console = Console()


@app.command()
def ingest(
    max_results: int = typer.Option(
        settings.max_results_default,
        "--max-results", "-n",
        help="Maximum number of papers to fetch from arXiv.",
    ),
    categories: Optional[List[str]] = typer.Option(
        None,
        "--categories", "-c",
        help="arXiv category codes (repeatable). Defaults to config.",
    ),
    days_back: Optional[int] = typer.Option(
        None,
        "--days-back", "-d",
        help="Only fetch papers from the last N days.",
    ),
    full_pdf: bool = typer.Option(
        False,
        "--full-pdf",
        help="Download and parse full PDFs (slow, requires PyMuPDF).",
    ),
    skip_fetch: bool = typer.Option(
        False,
        "--skip-fetch",
        help="Skip arXiv fetch, use existing raw/papers.jsonl.",
    ),
    force_reindex: bool = typer.Option(
        False,
        "--force-reindex",
        help="Rebuild the FAISS index from scratch even if it exists.",
    ),
) -> None:
    """Run the full ingestion pipeline: fetch → chunk → embed → index."""

    settings.ensure_directories()
    settings.configure_langsmith()

    # ── Step 1: Fetch from arXiv ─────────────────────────────────────────────
    raw_path = settings.raw_data_path / "papers.jsonl"

    if skip_fetch and raw_path.exists():
        console.print(f"[yellow]⏭  Skipping fetch — using existing {raw_path}[/yellow]")
        from src.ingestion.arxiv_fetcher import ArxivFetcher
        fetcher = ArxivFetcher()
        papers = fetcher.load_from_jsonl(raw_path)
    else:
        from src.ingestion.arxiv_fetcher import ArxivFetcher
        cats = categories or settings.arxiv_categories
        fetcher = ArxivFetcher(categories=cats)

        console.print(
            f"\n[bold cyan]🔭 CERN Knowledge Navigator — Ingestion Pipeline[/bold cyan]\n"
            f"  Categories : {cats}\n"
            f"  Max results: {max_results}\n"
            f"  Days back  : {days_back or 'all time'}\n"
            f"  Full PDF   : {full_pdf}\n"
        )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching papers from arXiv...", total=None)
            papers = fetcher.fetch_papers(
                max_results=max_results, days_back=days_back
            )
            progress.update(task, description=f"Fetched {len(papers)} papers ✓")

        fetcher.save_to_jsonl(papers, raw_path)

    if not papers:
        console.print("[red]No papers fetched. Aborting.[/red]")
        raise typer.Exit(1)

    # ── Step 2: Optional full PDF extraction ─────────────────────────────────
    if full_pdf:
        from src.ingestion.pdf_parser import PdfParser
        parser = PdfParser()
        console.print(f"[cyan]📄 Extracting full text from {len(papers)} PDFs...[/cyan]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
        ) as progress:
            task = progress.add_task("Downloading PDFs", total=len(papers))
            for i, paper in enumerate(papers):
                papers[i] = parser.enrich_paper_with_full_text(paper)
                progress.advance(task)

        # Save enriched papers
        enriched_path = settings.raw_data_path / "papers_full.jsonl"
        with open(enriched_path, "w", encoding="utf-8") as f:
            for p in papers:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        console.print(f"[green]✓ Saved enriched papers → {enriched_path}[/green]")

    # ── Step 3: Chunk ────────────────────────────────────────────────────────
    from src.ingestion.chunker import PaperChunker
    chunker = PaperChunker()

    console.print("\n[cyan]✂️  Chunking papers...[/cyan]")
    docs = chunker.chunk_papers(papers)
    console.print(f"[green]✓ Created {len(docs)} document chunks[/green]")

    # ── Step 4: Embed + Index ─────────────────────────────────────────────────
    from src.rag.vector_store import CernVectorStore

    vs = CernVectorStore()

    # Check if index already exists
    if not force_reindex and vs.index_exists():
        console.print(
            f"\n[yellow]ℹ  FAISS index already exists at {settings.faiss_index_path}. "
            "Use --force-reindex to rebuild.[/yellow]"
        )
        console.print("[yellow]⏭  Skipping embed + index step.[/yellow]")
    else:
        console.print("\n[cyan]🧮 Embedding chunks and building FAISS index...[/cyan]")
        console.print(
            f"  Model  : {settings.embedding_model}\n"
            f"  Chunks : {len(docs)}\n"
            f"  (This may take a few minutes on first run)\n"
        )

        start = time.perf_counter()
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Embedding & indexing...", total=None)
            vs.build_index(docs)
            elapsed = time.perf_counter() - start
            progress.update(task, description=f"Done in {elapsed:.1f}s ✓")

        console.print(f"[green]✓ FAISS index built in {elapsed:.1f}s[/green]")
        console.print(f"[green]✓ Index saved → {settings.faiss_index_path}[/green]")

    console.print(
        "\n[bold green]🎉 Ingestion complete![/bold green]\n"
        f"  Run the API : [cyan]python -m src.api.main[/cyan]\n"
        f"  Run the MCP : [cyan]python -m src.mcp_server.server[/cyan]\n"
        f"  Dashboard   : [cyan]streamlit run src/evaluation/dashboard.py[/cyan]\n"
    )


if __name__ == "__main__":
    app()
