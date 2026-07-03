"""Pydantic request/response models for the FastAPI REST service."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request Models ────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Body for POST /query."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The scientific question to ask the CERN knowledge base.",
        examples=["What is the measured Higgs boson mass at ATLAS?"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of source documents to retrieve.",
    )
    stream: bool = Field(
        default=False,
        description="If true, stream the response as server-sent events.",
    )


# ── Response Models ───────────────────────────────────────────────────────────

class SourceDocument(BaseModel):
    """Metadata for a retrieved source document."""
    arxiv_id: str = Field(description="Short arXiv paper ID.")
    title: str = Field(description="Paper title.")
    authors: str = Field(description="Author list (possibly truncated).")
    published_date: str = Field(description="Publication date (ISO format).")
    category: str = Field(description="arXiv category code.")
    chunk_index: int = Field(default=0, description="Chunk position within the paper.")
    rerank_score: Optional[float] = Field(
        default=None, description="Cross-encoder rerank score."
    )
    entry_url: Optional[str] = Field(default=None, description="arXiv paper URL.")


class QueryResponse(BaseModel):
    """Response from POST /query."""
    query: str
    answer: str
    sources: List[SourceDocument]
    source_count: int
    model_used: str


class PaperResponse(BaseModel):
    """Response for GET /papers/{arxiv_id}."""
    arxiv_id: str
    title: str
    authors: List[str]
    abstract: str
    published_date: str
    category: str
    pdf_url: Optional[str]
    entry_url: Optional[str]


class PapersListResponse(BaseModel):
    """Response for GET /papers."""
    total: int
    papers: List[PaperResponse]


class HealthResponse(BaseModel):
    """Response for GET /health."""
    status: str
    index_loaded: bool
    embedding_model: str
    llm_model: str
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    """Response for GET /metrics — latest RAGAS scores."""
    run_id: Optional[str]
    timestamp: Optional[str]
    faithfulness: Optional[float]
    answer_relevancy: Optional[float]
    context_precision: Optional[float]
    context_recall: Optional[float]
    question_count: Optional[int]
    faithfulness_threshold: float
    passes_threshold: Optional[bool]
