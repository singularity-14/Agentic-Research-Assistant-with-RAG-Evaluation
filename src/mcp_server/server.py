"""FastMCP server — CERN Knowledge Navigator MCP tools, resources & prompts.

Transports:
    stdio  : Claude Desktop / Cursor (run: python -m src.mcp_server.server)
    HTTP   : Remote agents via SSE  (run: python -m src.mcp_server.server --http)

Tools exposed:
    search_cern_docs(query, top_k)   — semantic + hybrid retrieval
    get_paper_summary(arxiv_id)      — single-paper summary via RAG
    list_indexed_categories()        — show indexed arXiv categories

Resources exposed:
    cern://papers/latest             — last 7 days of indexed papers

Prompts:
    physics_qa_template              — structured Q&A prompt for physics queries

Usage with Claude Desktop:
    Add to claude_desktop_config.json:
    {
      "mcpServers": {
        "cern-navigator": {
          "command": "python",
          "args": ["-m", "src.mcp_server.server"],
          "cwd": "/path/to/cern-knowledge-navigator"
        }
      }
    }
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastmcp import FastMCP
from loguru import logger

from src.config import settings

# ── Configure LangSmith before loading any LangChain components ──────────────
settings.configure_langsmith()

# ── IMPORTANT: Never use print() in stdio mode — it corrupts the JSON-RPC stream
# All logging goes to stderr only.
logger.remove()
logger.add(sys.stderr, level="INFO", format="{time} | {level} | {message}")

# ── MCP Server instance ───────────────────────────────────────────────────────
mcp = FastMCP(
    name="CERN Knowledge Navigator",
    instructions=(
        "A scientific knowledge assistant specialising in CERN high-energy physics. "
        "Search thousands of HEP papers from arXiv categories: hep-ex, hep-ph, "
        "physics.acc-ph. Ask about particle physics experiments, accelerator design, "
        "Higgs boson measurements, LHC operations, or any CERN research topic."
    ),
)


# ── Lazy-loaded RAG components ────────────────────────────────────────────────
_rag_chain = None
_fetcher = None


def _get_rag():
    global _rag_chain
    if _rag_chain is None:
        from src.rag.chain import get_rag_chain
        _rag_chain = get_rag_chain()
    return _rag_chain


def _get_fetcher():
    global _fetcher
    if _fetcher is None:
        from src.ingestion.arxiv_fetcher import ArxivFetcher
        _fetcher = ArxivFetcher()
    return _fetcher


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
async def search_cern_docs(
    query: str,
    top_k: int = 5,
) -> str:
    """Search CERN scientific documentation using semantic RAG retrieval.

    Performs hybrid BM25 + dense vector search over indexed HEP papers,
    followed by cross-encoder reranking and Groq LLM answer generation.

    Args:
        query: Scientific question or keyword search (e.g. 'Higgs boson mass
               measurement at ATLAS', 'LHC beam emittance').
        top_k: Number of source documents to retrieve (1-10, default 5).

    Returns:
        A scientifically grounded answer with source citations.
    """
    top_k = max(1, min(top_k, 10))  # clamp to [1, 10]

    try:
        rag = _get_rag()
        result = rag.invoke(query)

        answer = result["answer"]
        sources = result["sources"][:top_k]

        # Build formatted source list
        source_lines = []
        for i, doc in enumerate(sources, 1):
            meta = doc.metadata
            source_lines.append(
                f"[{i}] {meta.get('title', 'Unknown title')} "
                f"| arXiv:{meta.get('arxiv_id', 'N/A')} "
                f"| {meta.get('category', '')} "
                f"| {meta.get('published_date', '')}"
            )

        sources_str = "\n".join(source_lines) if source_lines else "No sources found."

        return (
            f"{answer}\n\n"
            f"── Sources Retrieved ──\n"
            f"{sources_str}"
        )

    except FileNotFoundError:
        return (
            "⚠️  The CERN knowledge base has not been indexed yet. "
            "Please run: python -m src.ingestion.pipeline"
        )
    except Exception as exc:
        logger.error(f"search_cern_docs error: {exc}")
        return f"An error occurred during retrieval: {str(exc)}"


@mcp.tool()
async def get_paper_summary(arxiv_id: str) -> str:
    """Get a concise AI-generated summary of a specific CERN/HEP paper by arXiv ID.

    Fetches the paper metadata from arXiv and generates a structured summary
    covering: problem statement, methodology, key findings, and significance.

    Args:
        arxiv_id: The arXiv paper identifier (e.g. '2401.12345' or
                  '2401.12345v2'). Both short and full IDs are accepted.

    Returns:
        A structured summary of the paper (problem, method, findings, impact).
    """
    # Clean up the ID
    arxiv_id = arxiv_id.strip().split("v")[0]  # remove version suffix

    try:
        fetcher = _get_fetcher()
        paper = fetcher.fetch_by_id(arxiv_id)

        if not paper:
            return f"Paper arXiv:{arxiv_id} not found. Verify the ID is correct."

        # Use RAG chain to generate a structured summary from the abstract
        summary_query = (
            f"Provide a structured summary of this paper. "
            f"Title: {paper['title']}. "
            f"Abstract: {paper['abstract']}\n\n"
            f"Cover: (1) Problem being solved, (2) Methodology, "
            f"(3) Key findings, (4) Scientific significance."
        )

        rag = _get_rag()
        llm = rag._get_llm()
        from langchain_core.messages import HumanMessage

        response = llm.invoke([HumanMessage(content=summary_query)])
        summary = response.content

        authors_str = ", ".join(paper.get("authors", [])[:5])
        if len(paper.get("authors", [])) > 5:
            authors_str += f" et al."

        return (
            f"📄 {paper['title']}\n"
            f"   arXiv:{paper['arxiv_id']} | {paper['published_date'][:10]}\n"
            f"   Authors: {authors_str}\n"
            f"   Category: {paper.get('category', 'N/A')}\n\n"
            f"{summary}\n\n"
            f"🔗 {paper.get('entry_url', '')}"
        )

    except Exception as exc:
        logger.error(f"get_paper_summary error for {arxiv_id}: {exc}")
        return f"Failed to summarise arXiv:{arxiv_id}: {str(exc)}"


@mcp.tool()
async def list_indexed_categories() -> str:
    """List the arXiv categories currently indexed in the CERN knowledge base.

    Returns the configured physics categories and their descriptions,
    along with instructions for re-ingesting with different categories.

    Returns:
        Human-readable list of indexed categories with descriptions.
    """
    category_descriptions = {
        "hep-ex": "High Energy Physics - Experiment (ATLAS, CMS, LHCb, ALICE measurements)",
        "hep-ph": "High Energy Physics - Phenomenology (Standard Model theory, BSM physics)",
        "physics.acc-ph": "Accelerator Physics (LHC beam dynamics, magnet design, RF systems)",
        "hep-th": "High Energy Physics - Theory (QFT, string theory, formal aspects)",
        "nucl-ex": "Nuclear Experiment (heavy ion collisions, nuclear structure)",
    }

    indexed = settings.arxiv_categories
    lines = ["📚 Indexed arXiv Categories:\n"]
    for cat in indexed:
        desc = category_descriptions.get(cat, "No description available.")
        lines.append(f"  ✓ {cat} — {desc}")

    not_indexed = [c for c in category_descriptions if c not in indexed]
    if not_indexed:
        lines.append("\n📋 Available (not yet indexed):")
        for cat in not_indexed:
            desc = category_descriptions[cat]
            lines.append(f"  ○ {cat} — {desc}")

    lines.append(
        "\n💡 To add more categories, update ARXIV_CATEGORIES in .env "
        "and re-run: python -m src.ingestion.pipeline --force-reindex"
    )
    return "\n".join(lines)


# ── Resources ─────────────────────────────────────────────────────────────────

@mcp.resource("cern://papers/latest")
async def latest_papers() -> str:
    """Latest HEP papers published to arXiv in the last 7 days.

    Fetches live from arXiv API — results are not limited to indexed papers.
    """
    try:
        fetcher = _get_fetcher()
        papers = fetcher.fetch_latest(days=7)

        if not papers:
            return "No recent papers found in the last 7 days."

        lines = [
            f"📰 Latest HEP Papers (last 7 days) — {len(papers)} found\n",
        ]
        for i, p in enumerate(papers[:20], 1):
            authors = ", ".join(p.get("authors", [])[:3])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            lines.append(
                f"{i:2}. [{p.get('arxiv_id', '')}] {p.get('title', '')}\n"
                f"    {authors} | {p.get('published_date', '')[:10]} | {p.get('category', '')}"
            )

        if len(papers) > 20:
            lines.append(f"\n... and {len(papers) - 20} more.")

        return "\n".join(lines)

    except Exception as exc:
        logger.error(f"latest_papers resource error: {exc}")
        return f"Failed to fetch latest papers: {str(exc)}"


# ── Prompts ───────────────────────────────────────────────────────────────────

@mcp.prompt()
def physics_qa_template(topic: str = "particle physics") -> str:
    """Structured prompt template for physics Q&A using CERN data.

    Returns a ready-to-use prompt that instructs the LLM to answer physics
    questions using only CERN documentation, with proper citations.

    Args:
        topic: The physics topic or experiment to focus on
               (e.g. 'Higgs boson', 'LHC luminosity', 'b-quark decays').

    Returns:
        A formatted prompt string for physics Q&A.
    """
    return (
        f"You are a CERN physicist specialising in {topic}.\n\n"
        "When answering questions:\n"
        "1. Use the search_cern_docs tool to retrieve relevant papers first.\n"
        "2. Ground all claims in retrieved context — never hallucinate data.\n"
        "3. Quote specific measurements with uncertainties (e.g. 125.09 ± 0.24 GeV).\n"
        "4. Cite the arXiv ID for every factual claim.\n"
        "5. If results are contradictory across papers, mention the discrepancy.\n\n"
        f"Begin by searching for: '{topic}'"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

def run():
    """Run the MCP server in stdio mode (default for Claude Desktop / Cursor)."""
    import argparse

    parser = argparse.ArgumentParser(description="CERN Knowledge Navigator MCP Server")
    parser.add_argument(
        "--http",
        action="store_true",
        help="Run in HTTP/SSE mode instead of stdio.",
    )
    parser.add_argument("--host", default=settings.mcp_host)
    parser.add_argument("--port", type=int, default=settings.mcp_port)
    args = parser.parse_args()

    if args.http:
        logger.info(f"Starting MCP server (HTTP/SSE) on {args.host}:{args.port}")
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.info("Starting MCP server in stdio mode (for Claude Desktop / Cursor)")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    run()
