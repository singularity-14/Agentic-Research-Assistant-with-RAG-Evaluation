"""Optional PDF full-text extractor using PyMuPDF.

Only invoked when the --full-pdf flag is passed to the ingestion pipeline.
Falls back gracefully if a PDF cannot be downloaded or parsed.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import httpx
from loguru import logger


class PdfParser:
    """Download and extract full text from arXiv PDFs using PyMuPDF."""

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    def extract_text_from_path(self, pdf_path: Path) -> Optional[str]:
        """Extract text from a local PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text string, or None on failure.
        """
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(str(pdf_path))
            text_parts: list[str] = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            full_text = "\n".join(text_parts).strip()
            return full_text if full_text else None
        except Exception as exc:
            logger.warning(f"PDF text extraction failed for {pdf_path}: {exc}")
            return None

    def extract_text_from_url(self, pdf_url: str) -> Optional[str]:
        """Download a PDF from URL and extract its text synchronously.

        Args:
            pdf_url: Direct URL to the PDF (e.g. arXiv PDF link).

        Returns:
            Extracted text string, or None on failure.
        """
        try:
            with httpx.Client(timeout=self.timeout, follow_redirects=True) as client:
                response = client.get(pdf_url)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                    tmp.write(response.content)
                    tmp_path = Path(tmp.name)

            text = self.extract_text_from_path(tmp_path)
            tmp_path.unlink(missing_ok=True)
            return text

        except httpx.TimeoutException:
            logger.warning(f"Timeout downloading PDF: {pdf_url}")
            return None
        except Exception as exc:
            logger.warning(f"Failed to fetch PDF from {pdf_url}: {exc}")
            return None

    def enrich_paper_with_full_text(
        self, paper: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add full_text field to a paper dict by downloading its PDF.

        Args:
            paper: Paper metadata dict with a pdf_url field.

        Returns:
            Updated paper dict with full_text key added.
        """
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            logger.debug(f"No PDF URL for {paper.get('arxiv_id', 'unknown')}")
            return paper

        logger.info(f"Downloading PDF for {paper.get('arxiv_id', '')} ...")
        full_text = self.extract_text_from_url(pdf_url)

        if full_text:
            paper["full_text"] = full_text
            logger.success(
                f"Extracted {len(full_text):,} chars from {paper.get('arxiv_id', '')}"
            )
        else:
            logger.warning(
                f"Could not extract full text for {paper.get('arxiv_id', '')}. "
                "Using abstract only."
            )

        return paper
