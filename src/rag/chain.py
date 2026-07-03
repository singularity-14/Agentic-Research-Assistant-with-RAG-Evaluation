"""Full LangChain RAG chain with hybrid retrieval, HyDE, and Groq generation.

Pipeline:
    User Query
        → [Optional] HyDE enhancement
        → Hybrid retriever (BM25 + FAISS + cross-encoder reranker)
        → Contextual compression (LLM Extractor filters irrelevant context)
        → Groq LLM generation
        → Streamed answer with source citations
        → LangSmith tracing (when enabled)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from loguru import logger

from src.config import settings
from src.rag.hyde import HyDEQueryEnhancer
from src.rag.retriever import HybridRetriever
from src.rag.vector_store import CernVectorStore


# ── Prompt Template ──────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = """You are CERN Knowledge Navigator, an expert AI assistant \
specialising in high-energy physics, accelerator physics, and CERN scientific research.

You answer questions using the retrieved scientific context below. Follow these rules:
1. Ground every claim in the provided context. Do not invent facts.
2. If the context doesn't contain enough information, say so explicitly.
3. Use scientific terminology accurately. Include relevant equations or values when helpful.
4. Cite the source paper (title and arXiv ID) at the end of your answer.
5. Be concise: prefer 3-6 sentence answers unless detail is explicitly requested.

Retrieved Context:
{context}"""

RAG_HUMAN_PROMPT = "{question}"

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", RAG_HUMAN_PROMPT),
    ]
)


def _format_docs(docs: List[Document]) -> str:
    """Format retrieved documents into a context string for the prompt."""
    formatted_parts: List[str] = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = (
            f"[{i}] {meta.get('title', 'Unknown')} "
            f"(arXiv:{meta.get('arxiv_id', 'N/A')}, "
            f"cat:{meta.get('category', 'N/A')}, "
            f"date:{meta.get('published_date', 'N/A')})"
        )
        formatted_parts.append(f"{source}\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted_parts)


# ── Chain Builder ─────────────────────────────────────────────────────────────

class CernRAGChain:
    """Full RAG pipeline: retrieve → compress → generate."""

    def __init__(
        self,
        use_hyde: bool = settings.enable_hyde,
        use_compression: bool = True,
    ) -> None:
        self.use_hyde = use_hyde
        self.use_compression = use_compression
        self._retriever: Optional[HybridRetriever] = None
        self._llm: Optional[ChatGroq] = None
        self._chain = None

    # ── Private setup ────────────────────────────────────────────────────────

    def _get_llm(self) -> ChatGroq:
        if self._llm is None:
            self._llm = ChatGroq(
                model=settings.groq_model,
                api_key=settings.groq_api_key,
                temperature=0.1,
                max_tokens=1024,
            )
        return self._llm

    def _get_retriever(self) -> HybridRetriever:
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def _build_chain(self):
        """Build and cache the full LangChain runnable chain."""
        if self._chain is not None:
            return self._chain

        llm = self._get_llm()
        hyde = HyDEQueryEnhancer(enabled=self.use_hyde)
        hybrid = self._get_retriever()

        def retrieve_with_hyde(query: str) -> List[Document]:
            enhanced = hyde.enhance(query)
            return hybrid.retrieve(enhanced)

        if self.use_compression:
            # Add LLM-based contextual compression to filter noisy chunks
            compressor = LLMChainExtractor.from_llm(llm)
            base_retriever = hybrid.as_langchain_retriever()
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )
            retrieve_step = compression_retriever
        else:
            retrieve_step = RunnableLambda(retrieve_with_hyde)

        self._chain = (
            {
                "context": retrieve_step | RunnableLambda(_format_docs),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )
        return self._chain

    # ── Public API ────────────────────────────────────────────────────────────

    def invoke(self, query: str) -> Dict[str, Any]:
        """Run a full RAG query and return answer + sources.

        Args:
            query: User's question.

        Returns:
            Dict with 'answer' (str) and 'sources' (List[Document]).
        """
        settings.configure_langsmith()
        chain = self._build_chain()

        # Retrieve sources separately for the response payload
        sources = self._get_retriever().retrieve(
            HyDEQueryEnhancer(enabled=self.use_hyde).enhance(query)
        )

        answer = chain.invoke(query)

        logger.info(
            f"RAG query complete | sources={len(sources)} | "
            f"answer_len={len(answer)}"
        )
        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "source_count": len(sources),
        }

    def stream(self, query: str) -> Iterator[str]:
        """Stream the RAG answer token-by-token.

        Args:
            query: User's question.

        Yields:
            Answer text chunks as they are generated.
        """
        settings.configure_langsmith()
        chain = self._build_chain()
        yield from chain.stream(query)

    def retrieve_only(self, query: str) -> List[Document]:
        """Run retrieval only — useful for evaluation and debugging.

        Args:
            query: User's question.

        Returns:
            Retrieved and reranked Documents.
        """
        enhanced = HyDEQueryEnhancer(enabled=self.use_hyde).enhance(query)
        return self._get_retriever().retrieve(enhanced)


@lru_cache(maxsize=1)
def get_rag_chain() -> CernRAGChain:
    """Return a cached singleton RAG chain instance."""
    return CernRAGChain()
