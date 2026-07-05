"""Hybrid retriever: BM25 + FAISS dense, fused with RRF + cross-encoder reranker.

Pipeline:
    1. BM25 keyword search → top-K candidates
    2. FAISS dense search → top-K candidates
    3. Reciprocal Rank Fusion (RRF) to merge lists
    4. Cross-encoder reranker (ms-marco-MiniLM-L-6-v2) for final ordering
    5. Return top-K reranked results
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from loguru import logger

from src.config import settings
from src.rag.vector_store import CernVectorStore


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Lazy-load the cross-encoder model (cached singleton)."""
    try:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder: cross-encoder/ms-marco-MiniLM-L-6-v2")
        model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            max_length=512,
        )
        logger.success("Cross-encoder loaded.")
        return model
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. Reranking disabled."
        )
        return None


def reciprocal_rank_fusion(
    results_lists: List[List[Document]],
    k: int = 60,
) -> List[Tuple[Document, float]]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    Args:
        results_lists: List of ranked document lists.
        k: RRF constant (typically 60).

    Returns:
        List of (Document, rrf_score) tuples, sorted descending.
    """
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}

    for result_list in results_lists:
        for rank, doc in enumerate(result_list):
            # Use content hash as a unique key
            doc_id = hash(doc.page_content[:200])
            key = str(doc_id)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            doc_map[key] = doc

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_map[key], score) for key, score in sorted_items]


class HybridRetriever:
    """Hybrid BM25 + dense retriever with cross-encoder reranking."""

    def __init__(
        self,
        vector_store: Optional[CernVectorStore] = None,
        top_k_retrieval: int = settings.top_k_retrieval,
        top_k_rerank: int = settings.top_k_rerank,
        use_reranker: bool = True,
    ) -> None:
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.use_reranker = use_reranker

        # Vector store (lazy-loaded)
        self._vs = vector_store or CernVectorStore()
        self._bm25_retriever: Optional[BM25Retriever] = None

    def _get_bm25(self) -> BM25Retriever:
        """Build or return the BM25 retriever from the FAISS docstore."""
        if self._bm25_retriever is None:
            logger.info("Building BM25 index from FAISS docstore...")
            all_docs = self._vs.get_all_documents()
            self._bm25_retriever = BM25Retriever.from_documents(
                all_docs, k=self.top_k_retrieval
            )
            logger.success(f"BM25 index built from {len(all_docs)} documents.")
        return self._bm25_retriever

    def retrieve(self, query: str) -> List[Document]:
        """Run hybrid retrieval with optional reranking.

        Args:
            query: User query string.

        Returns:
            Top-K reranked documents.
        """
        # 1. BM25 search
        bm25 = self._get_bm25()
        bm25_results = bm25.invoke(query)

        # 2. Dense search
        dense_results = self._vs.similarity_search(query, k=self.top_k_retrieval)

        # 3. RRF fusion
        fused = reciprocal_rank_fusion([bm25_results, dense_results])
        candidates = [doc for doc, _ in fused[: self.top_k_retrieval * 2]]

        logger.debug(
            f"Retrieved {len(bm25_results)} BM25 + {len(dense_results)} dense "
            f"→ {len(candidates)} after RRF"
        )

        # 4. Cross-encoder reranking
        if self.use_reranker and candidates:
            candidates = self._rerank(query, candidates)

        return candidates[: self.top_k_rerank]

    def _rerank(
        self, query: str, candidates: List[Document]
    ) -> List[Document]:
        """Score candidates with cross-encoder and re-sort.

        Args:
            query: Original user query.
            candidates: Candidate documents from fusion step.

        Returns:
            Candidates sorted by cross-encoder score (descending).
        """
        cross_encoder = _load_cross_encoder()
        if cross_encoder is None:
            return candidates

        pairs = [(query, doc.page_content[:512]) for doc in candidates]
        scores = cross_encoder.predict(pairs)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        # Attach rerank score to metadata for transparency
        for doc, score in ranked:
            doc.metadata["rerank_score"] = float(score)

        logger.debug(f"Reranked {len(candidates)} → top score: {ranked[0][1]:.3f}")
        return [doc for doc, _ in ranked]

    def as_langchain_retriever(self) -> BaseRetriever:
        """Return an EnsembleRetriever for use in LangChain chains.

        Returns:
            LangChain EnsembleRetriever (BM25 + FAISS, equal weights).
        """
        # Lazy import: langchain.retrievers.__init__ eagerly loads
        # ContextualCompressionRetriever which breaks on langchain_core>=0.3.60
        # (langchain_core.memory was removed). Only import when actually needed.
        from langchain.retrievers.ensemble import EnsembleRetriever  # noqa: PLC0415
        bm25 = self._get_bm25()
        dense = self._vs.as_retriever(k=self.top_k_retrieval)
        return EnsembleRetriever(
            retrievers=[bm25, dense],
            weights=[0.4, 0.6],
        )
