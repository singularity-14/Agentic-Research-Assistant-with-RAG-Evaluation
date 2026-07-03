"""FAISS vector store wrapper — build, save, load, and search.

Wraps LangChain's FAISS integration with persistence and metadata support.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from loguru import logger

from src.config import settings
from src.rag.embeddings import get_embeddings


class CernVectorStore:
    """Manages the FAISS vector store for CERN paper chunks."""

    def __init__(self, index_path: Optional[Path] = None) -> None:
        self.index_path = index_path or settings.faiss_index_path
        self.index_name = settings.index_name
        self._store: Optional[FAISS] = None

    # ── Build ────────────────────────────────────────────────────────────────

    def build_index(self, documents: List[Document]) -> None:
        """Embed all documents and build a new FAISS index from scratch.

        Args:
            documents: List of LangChain Documents to index.
        """
        if not documents:
            raise ValueError("Cannot build index from empty document list.")

        logger.info(f"Building FAISS index from {len(documents)} documents...")
        embeddings = get_embeddings()

        # Build index in batches to track progress
        batch_size = 256
        if len(documents) <= batch_size:
            self._store = FAISS.from_documents(documents, embeddings)
        else:
            # Build from first batch then add remaining in chunks
            self._store = FAISS.from_documents(documents[:batch_size], embeddings)
            for start in range(batch_size, len(documents), batch_size):
                batch = documents[start : start + batch_size]
                self._store.add_documents(batch)
                logger.debug(
                    f"Indexed batch {start // batch_size + 1} "
                    f"({start + len(batch)}/{len(documents)})"
                )

        self.save()
        logger.success(f"FAISS index built and saved to {self.index_path}")

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Save the current FAISS index to disk."""
        if self._store is None:
            raise RuntimeError("No index to save. Call build_index() first.")
        self.index_path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.index_path), index_name=self.index_name)
        logger.debug(f"FAISS index saved → {self.index_path}/{self.index_name}")

    def load(self) -> "CernVectorStore":
        """Load FAISS index from disk.

        Returns:
            Self (for chaining).

        Raises:
            FileNotFoundError: If the index files do not exist.
        """
        if not self.index_exists():
            raise FileNotFoundError(
                f"FAISS index not found at {self.index_path}. "
                "Run: python -m src.ingestion.pipeline"
            )
        embeddings = get_embeddings()
        self._store = FAISS.load_local(
            str(self.index_path),
            embeddings,
            index_name=self.index_name,
            allow_dangerous_deserialization=True,
        )
        logger.info(f"FAISS index loaded from {self.index_path}")
        return self

    def index_exists(self) -> bool:
        """Return True if a saved FAISS index exists on disk."""
        return (self.index_path / f"{self.index_name}.faiss").exists()

    # ── Search ───────────────────────────────────────────────────────────────

    @property
    def store(self) -> FAISS:
        """Lazy-load the store on first access."""
        if self._store is None:
            self.load()
        return self._store  # type: ignore[return-value]

    def similarity_search(
        self,
        query: str,
        k: int = settings.top_k_retrieval,
    ) -> List[Document]:
        """Dense vector similarity search.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of most similar Documents.
        """
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_scores(
        self,
        query: str,
        k: int = settings.top_k_retrieval,
    ) -> List[Tuple[Document, float]]:
        """Dense search returning (Document, score) tuples.

        Args:
            query: Search query string.
            k: Number of results to return.

        Returns:
            List of (Document, cosine_score) tuples, descending score.
        """
        return self.store.similarity_search_with_relevance_scores(query, k=k)

    def as_retriever(self, k: int = settings.top_k_retrieval):
        """Return a LangChain-compatible retriever interface.

        Args:
            k: Number of documents to retrieve.

        Returns:
            VectorStoreRetriever instance.
        """
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Add new documents to an existing index.

        Args:
            documents: New documents to upsert.
        """
        self.store.add_documents(documents)
        self.save()
        logger.info(f"Added {len(documents)} documents to FAISS index.")

    def get_all_documents(self) -> List[Document]:
        """Retrieve all stored documents (for BM25 indexing).

        Returns:
            List of all Document objects in the store.
        """
        # FAISS stores docs in its docstore dict
        return list(self.store.docstore._dict.values())  # type: ignore[attr-defined]
