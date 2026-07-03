"""SQLite-backed metrics store for persisting RAGAS evaluation results."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from sqlalchemy import Column, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.config import settings


class Base(DeclarativeBase):
    pass


class EvalResult(Base):
    __tablename__ = "eval_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(64), nullable=False, unique=True)
    timestamp = Column(String(32), nullable=False)
    faithfulness = Column(Float, nullable=True)
    answer_relevancy = Column(Float, nullable=True)
    context_precision = Column(Float, nullable=True)
    context_recall = Column(Float, nullable=True)
    question_count = Column(Integer, nullable=True)
    raw_results = Column(Text, nullable=True)  # JSON blob of per-question results


class MetricsStore:
    """Persist and query RAGAS evaluation results."""

    DB_FILE = "data/evaluation/results/metrics.db"

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or (settings.eval_results_path / "metrics.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.db_path}", echo=False
        )
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)

    def save(self, metrics: Dict[str, Any]) -> str:
        """Persist an evaluation run to the database.

        Args:
            metrics: Dict with faithfulness, answer_relevancy, etc.

        Returns:
            The run_id string assigned to this run.
        """
        run_id = metrics.get("run_id") or str(uuid.uuid4())[:8]
        ts = datetime.now(tz=timezone.utc).isoformat()

        raw = json.dumps(metrics.get("per_question", []), ensure_ascii=False)

        with Session(self.engine) as session:
            result = EvalResult(
                run_id=run_id,
                timestamp=ts,
                faithfulness=metrics.get("faithfulness"),
                answer_relevancy=metrics.get("answer_relevancy"),
                context_precision=metrics.get("context_precision"),
                context_recall=metrics.get("context_recall"),
                question_count=metrics.get("question_count"),
                raw_results=raw,
            )
            session.add(result)
            session.commit()

        logger.info(f"Saved eval run {run_id} → {self.db_path}")
        return run_id

    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Return the most recent evaluation run as a dict."""
        with Session(self.engine) as session:
            row = (
                session.query(EvalResult)
                .order_by(EvalResult.id.desc())
                .first()
            )
            if not row:
                return None
            return {
                "run_id": row.run_id,
                "timestamp": row.timestamp,
                "faithfulness": row.faithfulness,
                "answer_relevancy": row.answer_relevancy,
                "context_precision": row.context_precision,
                "context_recall": row.context_recall,
                "question_count": row.question_count,
            }

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return historical evaluation runs for dashboard charting."""
        with Session(self.engine) as session:
            rows = (
                session.query(EvalResult)
                .order_by(EvalResult.id.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "run_id": r.run_id,
                    "timestamp": r.timestamp,
                    "faithfulness": r.faithfulness,
                    "answer_relevancy": r.answer_relevancy,
                    "context_precision": r.context_precision,
                    "context_recall": r.context_recall,
                    "question_count": r.question_count,
                }
                for r in reversed(rows)
            ]
