"""Text chunker — converts raw paper text into LangChain Documents.

Uses RecursiveCharacterTextSplitter with token-aware sizing, preserving
per-chunk metadata (paper_id, title, authors, date, category).
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger

from src.config import settings


class PaperChunker:
    """Split paper text into overlapping chunks for embedding."""

    def __init__(
        self,
        chunk_size: int = settings.chunk_size,
        chunk_overlap: int = settings.chunk_overlap,
    ) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_paper(self, paper: Dict[str, Any]) -> List[Document]:
        """Convert a single paper dict into a list of LangChain Documents.

        The text fed to the splitter is:
            Title + Authors + Abstract  (for abstract-only ingestion)
            Title + Authors + Full text (for PDF-parsed papers)

        Args:
            paper: Paper metadata dict from ArxivFetcher or PdfParser.

        Returns:
            List of Documents, each with rich metadata.
        """
        # Build the text content to chunk
        authors_str = ", ".join(paper.get("authors", [])[:5])
        if len(paper.get("authors", [])) > 5:
            authors_str += f" et al. ({len(paper['authors'])} authors)"

        full_text: str = paper.get("full_text") or paper.get("abstract", "")
        content = (
            f"Title: {paper.get('title', '')}\n"
            f"Authors: {authors_str}\n"
            f"Published: {paper.get('published_date', '')[:10]}\n"
            f"Category: {paper.get('category', '')}\n\n"
            f"{full_text}"
        )

        # Build per-chunk metadata
        base_metadata: Dict[str, Any] = {
            "arxiv_id": paper.get("arxiv_id", ""),
            "title": paper.get("title", ""),
            "authors": authors_str,
            "published_date": paper.get("published_date", "")[:10],
            "category": paper.get("category", ""),
            "pdf_url": paper.get("pdf_url", ""),
            "entry_url": paper.get("entry_url", ""),
            "source": "abstract" if not paper.get("full_text") else "full_pdf",
        }

        chunks = self.splitter.create_documents(
            texts=[content],
            metadatas=[base_metadata],
        )

        # Tag each chunk with its position index
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_total"] = len(chunks)

        return chunks

    def chunk_papers(self, papers: List[Dict[str, Any]]) -> List[Document]:
        """Chunk a list of papers and return all Documents.

        Args:
            papers: List of paper metadata dicts.

        Returns:
            Flattened list of Documents across all papers.
        """
        all_docs: List[Document] = []
        skipped = 0

        for paper in papers:
            if not paper.get("abstract") and not paper.get("full_text"):
                skipped += 1
                continue
            chunks = self.chunk_paper(paper)
            all_docs.extend(chunks)

        logger.info(
            f"Chunked {len(papers) - skipped} papers → {len(all_docs)} chunks "
            f"(skipped {skipped} with no text)"
        )
        return all_docs
