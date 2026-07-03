"""FastAPI application entry point.

Runs the CERN Knowledge Navigator REST API on port 8000.

Usage:
    python -m src.api.main
    uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.api.routes import router
from src.config import settings

# Configure LangSmith on startup
settings.configure_langsmith()
settings.ensure_directories()

app = FastAPI(
    title="CERN Knowledge Navigator API",
    description=(
        "Production RAG system over CERN/HEP scientific papers. "
        "Exposes semantic search, paper retrieval, and RAGAS evaluation metrics. "
        "Backed by FAISS + BGE embeddings + Groq LLM + LangSmith observability."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(router, prefix="")


@app.on_event("startup")
async def on_startup():
    logger.info("CERN Knowledge Navigator API starting up...")
    logger.info(f"LLM model  : {settings.groq_model}")
    logger.info(f"Embeddings : {settings.embedding_model}")
    logger.info(f"Index path : {settings.faiss_index_path}")
    logger.info(f"LangSmith  : {settings.langsmith_tracing}")


@app.get("/", tags=["System"])
async def root():
    return {
        "name": "CERN Knowledge Navigator",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


def run():
    """Start the Uvicorn server."""
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    run()
