"""FastAPI route handlers for the CERN Knowledge Navigator REST API."""

from __future__ import annotations

import json
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger

from src.api.models import (
    HealthResponse,
    MetricsResponse,
    PaperResponse,
    PapersListResponse,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from src.config import settings

router = APIRouter()


def _doc_to_source(doc) -> SourceDocument:
    """Convert a LangChain Document to a SourceDocument response model."""
    meta = doc.metadata
    return SourceDocument(
        arxiv_id=meta.get("arxiv_id", ""),
        title=meta.get("title", "Unknown"),
        authors=meta.get("authors", ""),
        published_date=meta.get("published_date", ""),
        category=meta.get("category", ""),
        chunk_index=meta.get("chunk_index", 0),
        rerank_score=meta.get("rerank_score"),
        entry_url=meta.get("entry_url"),
    )


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check() -> HealthResponse:
    """Check API health and index status."""
    from src.rag.vector_store import CernVectorStore

    vs = CernVectorStore()
    return HealthResponse(
        status="ok",
        index_loaded=vs.index_exists(),
        embedding_model=settings.embedding_model,
        llm_model=settings.groq_model,
    )


# ── Query ─────────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest) -> QueryResponse:
    """Ask a question against the CERN scientific knowledge base.

    Performs hybrid retrieval (BM25 + dense + cross-encoder reranking),
    optionally applies HyDE query enhancement, then generates a grounded
    answer using Groq LLM with LangSmith observability.
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Use /query/stream for streaming responses.",
        )

    try:
        from src.rag.chain import get_rag_chain

        chain = get_rag_chain()
        result = chain.invoke(request.query)

        sources = [
            _doc_to_source(doc)
            for doc in result["sources"][: request.top_k]
        ]

        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=sources,
            source_count=len(sources),
            model_used=settings.groq_model,
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge base not indexed. "
                "Run: python -m src.ingestion.pipeline"
            ),
        )
    except Exception as exc:
        logger.error(f"Query error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/query/stream", tags=["RAG"])
async def query_stream(request: QueryRequest):
    """Stream an answer as Server-Sent Events.

    Connect with EventSource in the browser or `httpx` in Python:
        response = httpx.stream("POST", "/query/stream", json={...})
    """
    try:
        from src.rag.chain import get_rag_chain

        chain = get_rag_chain()

        def event_generator():
            for chunk in chain.stream(request.query):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Knowledge base not indexed. Run ingestion pipeline first.",
        )
    except Exception as exc:
        logger.error(f"Stream error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Papers ────────────────────────────────────────────────────────────────────

@router.get("/papers", response_model=PapersListResponse, tags=["Papers"])
async def list_papers(
    category: Optional[str] = Query(None, description="Filter by arXiv category."),
    limit: int = Query(50, ge=1, le=200, description="Max papers to return."),
) -> PapersListResponse:
    """List papers stored in the raw data directory.

    Papers are read from data/raw/papers.jsonl — the file created by the
    ingestion pipeline.
    """
    try:
        from src.ingestion.arxiv_fetcher import ArxivFetcher

        fetcher = ArxivFetcher()
        all_papers = fetcher.load_from_jsonl()

        if category:
            all_papers = [p for p in all_papers if p.get("category") == category]

        papers = all_papers[:limit]
        return PapersListResponse(
            total=len(all_papers),
            papers=[
                PaperResponse(
                    arxiv_id=p.get("arxiv_id", ""),
                    title=p.get("title", ""),
                    authors=p.get("authors", []),
                    abstract=p.get("abstract", ""),
                    published_date=p.get("published_date", ""),
                    category=p.get("category", ""),
                    pdf_url=p.get("pdf_url"),
                    entry_url=p.get("entry_url"),
                )
                for p in papers
            ],
        )
    except Exception as exc:
        logger.error(f"List papers error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/papers/{arxiv_id}", response_model=PaperResponse, tags=["Papers"])
async def get_paper(arxiv_id: str) -> PaperResponse:
    """Fetch metadata for a specific paper by its arXiv ID.

    Looks up the paper in the local JSONL cache first; if not found,
    queries the arXiv API directly.
    """
    try:
        from src.ingestion.arxiv_fetcher import ArxivFetcher

        fetcher = ArxivFetcher()

        # Try local cache first
        all_papers = fetcher.load_from_jsonl()
        paper = next(
            (p for p in all_papers if p.get("arxiv_id") == arxiv_id), None
        )

        # Fall back to live arXiv lookup
        if not paper:
            paper = fetcher.fetch_by_id(arxiv_id)

        if not paper:
            raise HTTPException(
                status_code=404, detail=f"Paper arXiv:{arxiv_id} not found."
            )

        return PaperResponse(
            arxiv_id=paper.get("arxiv_id", ""),
            title=paper.get("title", ""),
            authors=paper.get("authors", []),
            abstract=paper.get("abstract", ""),
            published_date=paper.get("published_date", ""),
            category=paper.get("category", ""),
            pdf_url=paper.get("pdf_url"),
            entry_url=paper.get("entry_url"),
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Get paper error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


# ── Metrics ───────────────────────────────────────────────────────────────────

@router.get("/metrics", response_model=MetricsResponse, tags=["Evaluation"])
async def latest_metrics() -> MetricsResponse:
    """Return the most recent RAGAS evaluation scores.

    Reads from the evaluation results stored in data/evaluation/results/.
    Run python -m src.evaluation.ragas_runner to generate scores.
    """
    try:
        from src.evaluation.metrics_store import MetricsStore

        store = MetricsStore()
        latest = store.get_latest()

        if not latest:
            return MetricsResponse(
                run_id=None,
                timestamp=None,
                faithfulness=None,
                answer_relevancy=None,
                context_precision=None,
                context_recall=None,
                question_count=None,
                faithfulness_threshold=settings.faithfulness_threshold,
                passes_threshold=None,
            )

        return MetricsResponse(
            **latest,
            faithfulness_threshold=settings.faithfulness_threshold,
            passes_threshold=(
                (latest.get("faithfulness") or 0.0)
                >= settings.faithfulness_threshold
            ),
        )
    except Exception as exc:
        logger.error(f"Metrics error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))
