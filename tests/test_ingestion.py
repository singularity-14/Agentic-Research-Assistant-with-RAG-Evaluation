"""Tests for the data ingestion pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.ingestion.arxiv_fetcher import ArxivFetcher
from src.ingestion.chunker import PaperChunker


class TestArxivFetcher:
    def test_build_query_default_categories(self):
        fetcher = ArxivFetcher()
        assert len(fetcher.categories) > 0

    def test_match_category(self):
        fetcher = ArxivFetcher(categories=["hep-ex", "hep-ph"])
        assert fetcher._match_category(["hep-ex.01", "hep-ph"]) == "hep-ex"
        assert fetcher._match_category(["nucl-th"]) == "nucl-th"

    @patch("arxiv.Client")
    def test_fetch_papers_returns_list(self, mock_client_cls):
        mock_result = MagicMock()
        mock_result.get_short_id.return_value = "2401.00001"
        mock_result.title = "Test Paper"
        mock_result.authors = [MagicMock(name="Author One")]
        mock_result.summary = "Abstract text."
        mock_result.published.isoformat.return_value = "2024-01-01T00:00:00Z"
        mock_result.updated.isoformat.return_value = "2024-01-02T00:00:00Z"
        mock_result.categories = ["hep-ex"]
        mock_result.pdf_url = "https://arxiv.org/pdf/2401.00001"
        mock_result.entry_id = "https://arxiv.org/abs/2401.00001"
        mock_result.journal_ref = None
        mock_result.doi = None

        mock_client = MagicMock()
        mock_client.results.return_value = [mock_result]
        mock_client_cls.return_value = mock_client

        fetcher = ArxivFetcher(categories=["hep-ex"])
        fetcher.client = mock_client
        papers = fetcher.fetch_papers(max_results=1)

        assert len(papers) == 1
        assert papers[0]["arxiv_id"] == "2401.00001"
        assert papers[0]["title"] == "Test Paper"

    def test_save_and_load_jsonl(self, tmp_path):
        fetcher = ArxivFetcher()
        papers = [
            {"arxiv_id": "2401.00001", "title": "Paper A", "abstract": "Abstract A"},
            {"arxiv_id": "2401.00002", "title": "Paper B", "abstract": "Abstract B"},
        ]
        path = tmp_path / "test.jsonl"
        fetcher.save_to_jsonl(papers, path)
        loaded = fetcher.load_from_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["arxiv_id"] == "2401.00001"


class TestPaperChunker:
    def test_chunk_paper_basic(self):
        chunker = PaperChunker(chunk_size=200, chunk_overlap=20)
        paper = {
            "arxiv_id": "2401.00001",
            "title": "Higgs Boson Mass Measurement",
            "authors": ["Alice Smith", "Bob Jones"],
            "abstract": "We present a precise measurement of the Higgs boson mass. " * 10,
            "published_date": "2024-01-01",
            "category": "hep-ex",
        }
        chunks = chunker.chunk_paper(paper)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["arxiv_id"] == "2401.00001"
            assert chunk.metadata["title"] == "Higgs Boson Mass Measurement"
            assert "chunk_index" in chunk.metadata

    def test_chunk_papers_skips_empty(self):
        chunker = PaperChunker()
        papers = [
            {"arxiv_id": "A", "title": "Paper A", "abstract": "Some content here for testing."},
            {"arxiv_id": "B", "title": "Paper B"},  # no abstract
        ]
        chunks = chunker.chunk_papers(papers)
        arxiv_ids = {c.metadata["arxiv_id"] for c in chunks}
        assert "A" in arxiv_ids
        assert "B" not in arxiv_ids

    def test_chunk_has_metadata(self):
        chunker = PaperChunker(chunk_size=100, chunk_overlap=10)
        paper = {
            "arxiv_id": "2401.12345",
            "title": "Test",
            "authors": ["A", "B"],
            "abstract": "Short abstract text. " * 5,
            "published_date": "2024-06-01",
            "category": "hep-ph",
        }
        chunks = chunker.chunk_paper(paper)
        for ch in chunks:
            assert ch.metadata["category"] == "hep-ph"
            assert ch.metadata["source"] == "abstract"
