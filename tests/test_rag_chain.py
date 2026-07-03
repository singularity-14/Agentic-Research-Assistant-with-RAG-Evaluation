"""Tests for the RAG chain components."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document


class TestHyDEQueryEnhancer:
    def test_passthrough_when_disabled(self):
        from src.rag.hyde import HyDEQueryEnhancer

        enhancer = HyDEQueryEnhancer(enabled=False)
        query = "What is the Higgs boson mass?"
        result = enhancer.enhance(query)
        assert result == query

    def test_returns_original_on_failure(self):
        from src.rag.hyde import HyDEQueryEnhancer

        enhancer = HyDEQueryEnhancer(enabled=True)
        mock_chain = MagicMock()
        mock_chain.invoke.side_effect = Exception("API error")
        enhancer._chain = mock_chain
        result = enhancer.enhance("test query")
        assert result == "test query"


class TestReciprocalRankFusion:
    def test_rrf_merges_lists(self):
        from src.rag.retriever import reciprocal_rank_fusion

        doc_a = Document(page_content="Higgs boson discovery at ATLAS")
        doc_b = Document(page_content="LHC beam energy and luminosity")
        doc_c = Document(page_content="Muon spectrometer alignment")

        list1 = [doc_a, doc_b]
        list2 = [doc_a, doc_c]

        fused = reciprocal_rank_fusion([list1, list2])
        assert len(fused) >= 2
        # doc_a appears in both lists so should have highest score
        top_doc, top_score = fused[0]
        assert "Higgs" in top_doc.page_content

    def test_rrf_empty_lists(self):
        from src.rag.retriever import reciprocal_rank_fusion

        fused = reciprocal_rank_fusion([[], []])
        assert fused == []


class TestDocFormatting:
    def test_format_docs_with_metadata(self):
        from src.rag.chain import _format_docs

        docs = [
            Document(
                page_content="The Higgs boson mass is 125 GeV.",
                metadata={
                    "title": "Higgs Measurement",
                    "arxiv_id": "2401.00001",
                    "category": "hep-ex",
                    "published_date": "2024-01-01",
                },
            )
        ]
        formatted = _format_docs(docs)
        assert "Higgs Measurement" in formatted
        assert "2401.00001" in formatted
        assert "125 GeV" in formatted
