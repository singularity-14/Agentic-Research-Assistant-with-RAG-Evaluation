"""arXiv API fetcher — pulls HEP paper metadata for CERN categories.

Usage:
    from src.ingestion.arxiv_fetcher import ArxivFetcher
    fetcher = ArxivFetcher()
    papers = await fetcher.fetch_papers(max_results=500)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import arxiv
from loguru import logger

from src.config import settings


class ArxivFetcher:
    """Fetch paper metadata from the arXiv API for HEP categories."""

    # arXiv categories relevant to CERN physics
    DEFAULT_CATEGORIES: List[str] = settings.arxiv_categories

    def __init__(
        self,
        categories: Optional[List[str]] = None,
        delay_seconds: float = settings.arxiv_delay_seconds,
    ) -> None:
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=delay_seconds,
            num_retries=3,
        )

    def _build_query(self, days_back: Optional[int] = None) -> str:
        """Build an arXiv search query string for the configured categories."""
        cat_queries = [f"cat:{cat}" for cat in self.categories]
        query = " OR ".join(cat_queries)
        return query

    def fetch_papers(
        self,
        max_results: int = settings.max_results_default,
        days_back: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch papers synchronously from arXiv API.

        Args:
            max_results: Maximum number of papers to retrieve.
            days_back: If set, only papers from the last N days.

        Returns:
            List of paper metadata dicts.
        """
        query = self._build_query(days_back)
        logger.info(
            f"Fetching up to {max_results} papers | categories={self.categories}"
        )

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers: List[Dict[str, Any]] = []
        cutoff_date: Optional[datetime] = None
        if days_back:
            cutoff_date = datetime.now(tz=timezone.utc) - timedelta(days=days_back)

        for result in self.client.results(search):
            if cutoff_date and result.published < cutoff_date:
                logger.debug(
                    f"Stopping — paper {result.entry_id} older than cutoff."
                )
                break

            # Determine which configured category this paper belongs to
            matched_category = self._match_category(result.categories)

            paper: Dict[str, Any] = {
                "arxiv_id": result.get_short_id(),
                "title": result.title.strip(),
                "authors": [a.name for a in result.authors],
                "abstract": result.summary.strip(),
                "published_date": result.published.isoformat(),
                "updated_date": result.updated.isoformat(),
                "category": matched_category,
                "all_categories": result.categories,
                "pdf_url": result.pdf_url,
                "entry_url": result.entry_id,
                "journal_ref": result.journal_ref,
                "doi": result.doi,
            }
            papers.append(paper)

        logger.success(f"Fetched {len(papers)} papers from arXiv.")
        return papers

    def fetch_by_id(self, arxiv_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single paper by its arXiv ID.

        Args:
            arxiv_id: Short arXiv ID, e.g. '2401.12345'.

        Returns:
            Paper metadata dict or None if not found.
        """
        search = arxiv.Search(id_list=[arxiv_id])
        results = list(self.client.results(search))
        if not results:
            logger.warning(f"Paper not found: {arxiv_id}")
            return None

        result = results[0]
        return {
            "arxiv_id": result.get_short_id(),
            "title": result.title.strip(),
            "authors": [a.name for a in result.authors],
            "abstract": result.summary.strip(),
            "published_date": result.published.isoformat(),
            "updated_date": result.updated.isoformat(),
            "category": self._match_category(result.categories),
            "all_categories": result.categories,
            "pdf_url": result.pdf_url,
            "entry_url": result.entry_id,
            "journal_ref": result.journal_ref,
            "doi": result.doi,
        }

    def fetch_latest(self, days: int = 7) -> List[Dict[str, Any]]:
        """Fetch papers published in the last N days.

        Args:
            days: Look-back window in days.

        Returns:
            List of recent paper metadata dicts.
        """
        return self.fetch_papers(max_results=200, days_back=days)

    def save_to_jsonl(
        self, papers: List[Dict[str, Any]], output_path: Optional[Path] = None
    ) -> Path:
        """Persist fetched papers as JSONL for the ingestion pipeline.

        Args:
            papers: List of paper dicts to save.
            output_path: Target file. Defaults to data/raw/papers.jsonl.

        Returns:
            Path to the saved file.
        """
        if output_path is None:
            output_path = settings.raw_data_path / "papers.jsonl"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for paper in papers:
                f.write(json.dumps(paper, ensure_ascii=False) + "\n")

        logger.success(f"Saved {len(papers)} papers → {output_path}")
        return output_path

    def load_from_jsonl(self, path: Optional[Path] = None) -> List[Dict[str, Any]]:
        """Load previously saved paper metadata from JSONL.

        Args:
            path: Source file path. Defaults to data/raw/papers.jsonl.

        Returns:
            List of paper dicts.
        """
        if path is None:
            path = settings.raw_data_path / "papers.jsonl"

        if not path.exists():
            logger.warning(f"No raw data found at {path}. Run ingest first.")
            return []

        papers: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    papers.append(json.loads(line))

        logger.info(f"Loaded {len(papers)} papers from {path}")
        return papers

    # ── Private helpers ──────────────────────────────────────────────────────

    def _match_category(self, paper_categories: List[str]) -> str:
        """Return the first configured category that matches this paper."""
        for cat in self.categories:
            if any(cat in pc for pc in paper_categories):
                return cat
        # Return the paper's primary category if none match
        return paper_categories[0] if paper_categories else "unknown"
