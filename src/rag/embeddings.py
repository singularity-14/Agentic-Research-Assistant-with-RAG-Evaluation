"""HuggingFace BGE embedding wrapper for LangChain.

Uses BAAI/bge-small-en-v1.5 — a high-quality, compact embedding model
that runs fully locally with no API key required.

BGE models expect a special instruction prefix for retrieval queries.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

from src.config import settings


# BGE models perform better with this query instruction prefix
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a cached singleton embedding model instance.

    The model is downloaded from HuggingFace Hub on first call (~30 MB)
    and cached locally in ~/.cache/huggingface/.

    Returns:
        LangChain-compatible HuggingFaceEmbeddings instance.
    """
    logger.info(
        f"Loading embedding model: {settings.embedding_model} "
        f"(device={settings.embedding_device})"
    )

    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": settings.embedding_device},
        encode_kwargs={
            "normalize_embeddings": True,   # cosine similarity via dot product
            "batch_size": 32,
        },
        query_instruction=BGE_QUERY_INSTRUCTION,
    )

    logger.success(f"Embedding model loaded: {settings.embedding_model}")
    return embeddings


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of text strings.

    Args:
        texts: Raw text strings to embed.

    Returns:
        List of embedding vectors (one per input text).
    """
    model = get_embeddings()
    return model.embed_documents(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single search query with the BGE instruction prefix.

    Args:
        query: User search query.

    Returns:
        Embedding vector for the query.
    """
    model = get_embeddings()
    return model.embed_query(query)
