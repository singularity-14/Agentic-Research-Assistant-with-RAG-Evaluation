"""RAGAS evaluation runner with Groq as the judge LLM.

Usage:
    python -m src.evaluation.ragas_runner                         # full 50-Q eval
    python -m src.evaluation.ragas_runner --sample 10            # quick CI subset
    python -m src.evaluation.ragas_runner --ci                    # exit non-zero if below threshold
"""

from __future__ import annotations

import sys
import uuid
from typing import Any, Dict, List, Optional

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import settings

app = typer.Typer(name="cern-eval", add_completion=False)
console = Console()


def _build_ragas_llm():
    """Return Groq LLM wrapped for RAGAS."""
    from langchain_groq import ChatGroq
    from ragas.llms import LangchainLLMWrapper

    llm = ChatGroq(
        model=settings.groq_judge_model,
        api_key=settings.groq_api_key,
        temperature=0.0,
    )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """Return local BGE embeddings wrapped for RAGAS."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from src.rag.embeddings import get_embeddings

    return LangchainEmbeddingsWrapper(get_embeddings())


def run_evaluation(
    questions: List[Dict[str, Any]],
    sample: Optional[int] = None,
    ci_mode: bool = False,
) -> Dict[str, Any]:
    """Run RAGAS evaluation on a list of Q&A pairs.

    Args:
        questions: List of dicts with 'question' and 'ground_truth' keys.
        sample: If set, only evaluate this many questions (random sample).

    Returns:
        Dict of aggregated RAGAS metrics plus per-question results.
    """
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.run_config import RunConfig
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    # CI mode: only evaluate faithfulness to minimise Groq API calls (10 vs 40).
    # This avoids rate-limiting and keeps the CI gate fast and reliable.
    if ci_mode:
        logger.info("CI mode: evaluating faithfulness only (10 API calls).")
        active_metrics = [faithfulness]
    else:
        active_metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    from src.rag.chain import get_rag_chain

    if sample and sample < len(questions):
        import random
        questions = random.sample(questions, sample)

    chain = get_rag_chain()
    ragas_llm = _build_ragas_llm()
    ragas_emb = _build_ragas_embeddings()

    # Initialise metrics with Groq as judge
    for m in active_metrics:
        m.llm = ragas_llm
        if hasattr(m, "embeddings"):
            m.embeddings = ragas_emb
    metrics = active_metrics

    # Build evaluation dataset
    samples: List[SingleTurnSample] = []
    per_question: List[Dict[str, Any]] = []

    logger.info(f"Running RAG inference on {len(questions)} questions...")
    for i, qa in enumerate(questions, 1):
        q = qa["question"]
        gt = qa.get("ground_truth", "")

        try:
            result = chain.invoke(q)
            answer = result["answer"]
            contexts = [doc.page_content for doc in result["sources"]]
        except Exception as exc:
            logger.warning(f"[{i}/{len(questions)}] Inference failed: {exc}")
            answer = ""
            contexts = []

        samples.append(
            SingleTurnSample(
                user_input=q,
                retrieved_contexts=contexts,
                response=answer,
                reference=gt,
            )
        )
        per_question.append({"question": q, "answer": answer, "context_count": len(contexts)})
        logger.info(f"  [{i}/{len(questions)}] {q[:60]}...")

    dataset = EvaluationDataset(samples=samples)

    logger.info("Running RAGAS evaluation...")
    # max_workers=1: sequential evaluation avoids Groq free-tier 6k TPM rate limit.
    # timeout=120s per job; max_retries=5 with backoff handles transient 429s.
    run_cfg = RunConfig(max_workers=1, timeout=120, max_retries=5)
    ragas_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=run_cfg,
        raise_exceptions=False,
    )

    scores = ragas_result.to_pandas().mean(numeric_only=True).to_dict()

    return {
        "run_id": str(uuid.uuid4())[:8],
        "faithfulness": round(scores.get("faithfulness", 0.0), 4),
        "answer_relevancy": round(scores.get("answer_relevancy", 0.0), 4),
        "context_precision": round(scores.get("context_precision", 0.0), 4),
        "context_recall": round(scores.get("context_recall", 0.0), 4),
        "question_count": len(questions),
        "per_question": per_question,
    }


@app.command()
def evaluate_cmd(
    sample: Optional[int] = typer.Option(
        None, "--sample", "-n", help="Evaluate only N questions (fast CI mode)."
    ),
    ci: bool = typer.Option(
        False, "--ci", help="Exit with code 1 if faithfulness < threshold."
    ),
    dataset_path: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Path to custom golden QA JSON file."
    ),
) -> None:
    """Run RAGAS evaluation and display results."""
    settings.configure_langsmith()
    settings.ensure_directories()

    from src.evaluation.golden_dataset import load_golden_dataset
    from src.evaluation.metrics_store import MetricsStore

    questions = load_golden_dataset()
    if dataset_path:
        import json
        with open(dataset_path) as f:
            questions = json.load(f)

    console.print(
        f"\n[bold cyan]🔬 CERN Knowledge Navigator — RAGAS Evaluation[/bold cyan]\n"
        f"  Questions  : {len(questions)}\n"
        f"  Sample     : {sample or 'all'}\n"
        f"  Judge LLM  : {settings.groq_judge_model}\n"
        f"  Threshold  : faithfulness ≥ {settings.faithfulness_threshold}\n"
    )

    metrics = run_evaluation(questions, sample=sample, ci_mode=ci)

    # Persist
    store = MetricsStore()
    store.save(metrics)

    # Display table
    table = Table(title="RAGAS Evaluation Results", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Score", style="bold")
    table.add_column("Status", style="bold")

    threshold = settings.faithfulness_threshold

    def _status(val: float, thr: float = 0.7) -> str:
        return "[green]✓ PASS[/green]" if val >= thr else "[red]✗ FAIL[/red]"

    table.add_row("Faithfulness", f"{metrics['faithfulness']:.4f}", _status(metrics["faithfulness"], threshold))
    table.add_row("Answer Relevancy", f"{metrics['answer_relevancy']:.4f}", _status(metrics["answer_relevancy"]))
    table.add_row("Context Precision", f"{metrics['context_precision']:.4f}", _status(metrics["context_precision"]))
    table.add_row("Context Recall", f"{metrics['context_recall']:.4f}", _status(metrics["context_recall"]))

    console.print(table)
    console.print(f"\n  Run ID : {metrics['run_id']}")
    console.print(f"  Saved  : {store.db_path}\n")

    if ci and metrics["faithfulness"] < threshold:
        console.print(
            f"[bold red]❌ CI GATE FAILED: faithfulness {metrics['faithfulness']:.4f} "
            f"< threshold {threshold}[/bold red]"
        )
        raise typer.Exit(1)

    console.print("[bold green]✅ Evaluation complete.[/bold green]")


if __name__ == "__main__":
    app()
