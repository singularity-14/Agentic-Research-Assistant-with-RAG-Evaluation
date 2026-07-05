"""Hypothetical Document Embedding (HyDE) query enhancement.

HyDE improves retrieval by:
1. Using the LLM to generate a hypothetical "ideal answer" document.
2. Embedding that hypothetical document.
3. Using the hypothetical embedding to search the vector store.

This brings the query embedding closer to the space of real answer documents,
especially useful for short or under-specified queries.

Reference: Gao et al. (2022) — "Precise Zero-Shot Dense Retrieval without
Relevance Labels" — https://arxiv.org/abs/2212.10496
"""

from __future__ import annotations

from typing import Optional

from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

from src.config import settings


HYDE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a CERN physicist writing a concise, factual paragraph that "
            "directly answers the following question based on high-energy physics "
            "knowledge. Write exactly one paragraph (3-5 sentences). "
            "This paragraph will be used to search a scientific document database.",
        ),
        ("human", "{query}"),
    ]
)


class HyDEQueryEnhancer:
    """Generate a hypothetical document to improve retrieval quality."""

    def __init__(self, enabled: bool = settings.enable_hyde) -> None:
        self.enabled = enabled
        self._chain = None

    def _get_chain(self):
        """Lazily build the HyDE generation chain."""
        if self._chain is None:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            if "NVIDIA_GLM_API_KEY" in os.environ:
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=os.environ["NVIDIA_GLM_API_KEY"],
                    model="mistralai/mistral-medium-3.5-128b",
                    temperature=0.7,
                    max_tokens=300,
                    top_p=1.0,
                )
            elif settings.use_gemini_judge:
                from langchain_google_genai import ChatGoogleGenerativeAI
                os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
                llm = ChatGoogleGenerativeAI(
                    model=settings.gemini_judge_model,
                    temperature=0.3,
                    max_output_tokens=300,
                )
            else:
                llm = ChatGroq(
                    model=settings.groq_model,
                    api_key=settings.groq_api_key,
                    temperature=0.3,
                    max_tokens=300,
                )
            self._chain = HYDE_PROMPT | llm | StrOutputParser()
        return self._chain

    def enhance(self, query: str) -> str:
        """Generate a hypothetical document for the query.

        If HyDE is disabled or generation fails, returns the original query.

        Args:
            query: Original user query.

        Returns:
            Hypothetical document text (or original query as fallback).
        """
        if not self.enabled:
            return query

        import time
        for attempt in range(3):
            try:
                time.sleep(0.1)  # Avoid rate limit and 503s
                chain = self._get_chain()
                hypothetical_doc = chain.invoke({"query": query})
                logger.debug(
                    f"HyDE generated ({len(hypothetical_doc)} chars) for: {query[:60]}..."
                )
                return hypothetical_doc
            except Exception as exc:
                if attempt == 2:
                    logger.warning(f"HyDE generation failed after 3 attempts, using original query: {exc}")
                    return query
                else:
                    logger.warning(f"HyDE generation failed (attempt {attempt+1}), retrying: {exc}")
                    time.sleep(0.5)
