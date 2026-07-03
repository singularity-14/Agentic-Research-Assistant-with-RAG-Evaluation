"""Tests for the evaluation framework."""

from __future__ import annotations

import json
import pytest

from src.evaluation.golden_dataset import GOLDEN_QA, save_golden_dataset


class TestGoldenDataset:
    def test_dataset_size(self):
        assert len(GOLDEN_QA) == 50

    def test_all_have_required_fields(self):
        for qa in GOLDEN_QA:
            assert "question" in qa, f"Missing 'question' in: {qa}"
            assert "ground_truth" in qa, f"Missing 'ground_truth' in: {qa}"
            assert len(qa["question"]) > 10
            assert len(qa["ground_truth"]) > 20

    def test_save_and_load(self, tmp_path):
        path = tmp_path / "golden_qa.json"
        save_golden_dataset(path)
        with open(path, encoding="utf-8") as f:
            loaded = json.load(f)
        assert len(loaded) == len(GOLDEN_QA)
        assert loaded[0]["question"] == GOLDEN_QA[0]["question"]


class TestMetricsStore:
    def test_save_and_retrieve(self, tmp_path):
        from src.evaluation.metrics_store import MetricsStore

        store = MetricsStore(db_path=tmp_path / "test_metrics.db")
        metrics = {
            "run_id": "test001",
            "faithfulness": 0.85,
            "answer_relevancy": 0.78,
            "context_precision": 0.82,
            "context_recall": 0.75,
            "question_count": 10,
        }
        store.save(metrics)
        latest = store.get_latest()

        assert latest is not None
        assert latest["faithfulness"] == 0.85
        assert latest["run_id"] == "test001"

    def test_get_history_empty(self, tmp_path):
        from src.evaluation.metrics_store import MetricsStore

        store = MetricsStore(db_path=tmp_path / "empty.db")
        assert store.get_history() == []

    def test_get_latest_empty(self, tmp_path):
        from src.evaluation.metrics_store import MetricsStore

        store = MetricsStore(db_path=tmp_path / "empty2.db")
        assert store.get_latest() is None
