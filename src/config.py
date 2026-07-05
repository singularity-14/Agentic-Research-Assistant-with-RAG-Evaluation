"""CERN Knowledge Navigator — centralised configuration.

All settings are read from the .env file in the project root.
Never hard-code credentials; always use this module.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root is two levels up from this file (src/config.py → project root)
ROOT_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Centralised settings — populated from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LangSmith Observability ──────────────────────────────────────────────
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com", alias="LANGSMITH_ENDPOINT"
    )
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="cern-knowledge-navigator", alias="LANGSMITH_PROJECT"
    )

    # ── Groq LLM ────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile", alias="GROQ_MODEL"
    )
    groq_judge_model: str = Field(
        default="llama-3.1-8b-instant", alias="GROQ_JUDGE_MODEL"
    )

    # ── Gemini (judge LLM — preferred over Groq when key is set) ────────────
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_judge_model: str = Field(
        default="gemini-1.5-flash", alias="GEMINI_JUDGE_MODEL"
    )

    @property
    def use_gemini_judge(self) -> bool:
        """True when a real Gemini API key is configured (not the placeholder)."""
        key = self.gemini_api_key
        return bool(key) and key not in ("", "YOUR_GEMINI_API_KEY_HERE")

    # ── Embeddings (Local HuggingFace) ───────────────────────────────────────
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5", alias="EMBEDDING_MODEL"
    )
    embedding_device: str = Field(default="cpu", alias="EMBEDDING_DEVICE")

    # ── Vector Store (FAISS — local) ────────────────────────────────────────
    faiss_index_path: Path = Field(
        default=ROOT_DIR / "data" / "faiss_index", alias="FAISS_INDEX_PATH"
    )
    index_name: str = Field(default="cern_papers", alias="INDEX_NAME")

    # ── Data Ingestion ───────────────────────────────────────────────────────
    max_results_default: int = Field(default=500, alias="MAX_RESULTS_DEFAULT")
    arxiv_delay_seconds: float = Field(default=3.0, alias="ARXIV_DELAY_SECONDS")
    arxiv_categories: Any = Field(
        default=["hep-ex", "hep-ph", "physics.acc-ph"], alias="ARXIV_CATEGORIES"
    )

    # ── MCP Server ───────────────────────────────────────────────────────────
    mcp_auth_token: str = Field(
        default="changeme-replace-with-secure-random-token", alias="MCP_AUTH_TOKEN"
    )
    mcp_host: str = Field(default="0.0.0.0", alias="MCP_HOST")
    mcp_port: int = Field(default=8001, alias="MCP_PORT")

    # ── FastAPI REST Service ─────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    # ── RAG Chain ────────────────────────────────────────────────────────────
    top_k_retrieval: int = Field(default=10, alias="TOP_K_RETRIEVAL")
    top_k_rerank: int = Field(default=5, alias="TOP_K_RERANK")
    enable_hyde: bool = Field(default=True, alias="ENABLE_HYDE")
    chunk_size: int = Field(default=512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=50, alias="CHUNK_OVERLAP")

    # ── Evaluation ───────────────────────────────────────────────────────────
    faithfulness_threshold: float = Field(
        default=0.8, alias="FAITHFULNESS_THRESHOLD"
    )
    eval_results_path: Path = Field(
        default=ROOT_DIR / "data" / "evaluation" / "results",
        alias="EVAL_RESULTS_PATH",
    )
    golden_qa_path: Path = Field(
        default=ROOT_DIR / "data" / "evaluation" / "golden_qa.json",
        alias="GOLDEN_QA_PATH",
    )

    # ── Derived paths (not from env) ─────────────────────────────────────────
    @property
    def raw_data_path(self) -> Path:
        return ROOT_DIR / "data" / "raw"

    @property
    def processed_data_path(self) -> Path:
        return ROOT_DIR / "data" / "processed"

    @field_validator("arxiv_categories", mode="before")
    @classmethod
    def parse_categories(cls, v):
        """Accept both a comma-separated string and a list."""
        if isinstance(v, str):
            return [c.strip() for c in v.split(",") if c.strip()]
        return v

    def configure_langsmith(self) -> None:
        """Push LangSmith env vars so LangChain picks them up automatically."""
        if self.langsmith_tracing and self.langsmith_api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint
            os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
            os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project

    def ensure_directories(self) -> None:
        """Create all required data directories on first run."""
        dirs = [
            self.raw_data_path,
            self.processed_data_path,
            self.faiss_index_path,
            self.eval_results_path,
            self.golden_qa_path.parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


# ── Singleton ────────────────────────────────────────────────────────────────
settings = Settings()
