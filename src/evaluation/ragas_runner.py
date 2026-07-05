"""RAGAS evaluation runner with Groq as the judge LLM.

Usage:
    python -m src.evaluation.ragas_runner                         # full 50-Q eval
    python -m src.evaluation.ragas_runner --sample 10            # quick CI subset
    python -m src.evaluation.ragas_runner --ci                    # exit non-zero if below threshold
"""

from __future__ import annotations

import os
import sys
import uuid
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from src.config import settings

app = typer.Typer(name="cern-eval", add_completion=False)
console = Console()


def _build_ragas_llm():
    """Return judge LLM wrapped for RAGAS.

    Priority:
      1. Gemini (GEMINI_API_KEY set) — high rate limits, ideal for evaluation
      2. Groq 70B fallback — used when no Gemini key is configured
    """
    from ragas.llms import LangchainLLMWrapper

    if "NVIDIA_GLM_API_KEY" in os.environ:
        from langchain_openai import ChatOpenAI
        logger.info("Judge LLM: NVIDIA (mistral-medium-3.5-128b)")
        llm = ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_GLM_API_KEY"],
            model="mistralai/mistral-medium-3.5-128b",
            temperature=0.0,
            request_timeout=600,
        )
    elif settings.use_gemini_judge:
        from langchain_google_genai import ChatGoogleGenerativeAI
        # Set env var so the underlying google-generativeai SDK picks it up
        # correctly regardless of key format (AQ.* keys fail with google_api_key=)
        os.environ["GOOGLE_API_KEY"] = settings.gemini_api_key
        logger.info(f"Judge LLM: Gemini ({settings.gemini_judge_model})")
        llm = ChatGoogleGenerativeAI(
            model=settings.gemini_judge_model,
            temperature=0.0,
        )
    else:
        from langchain_groq import ChatGroq
        logger.info(f"Judge LLM: Groq ({settings.groq_model})")
        llm = ChatGroq(
            model=settings.groq_model,  # 70B for reliable scores
            api_key=settings.groq_api_key,
            temperature=0.0,
        )
    return LangchainLLMWrapper(llm)


def _build_ragas_embeddings():
    """Return local BGE embeddings wrapped for RAGAS."""
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from src.rag.embeddings import get_embeddings

    return LangchainEmbeddingsWrapper(get_embeddings())


def _invoke_with_retry(chain, query: str, max_retries: int = 3) -> dict:
    """Invoke the RAG chain with exponential backoff on 429 rate-limit errors."""
    import time
    last_exc: Exception = RuntimeError("No attempts made")
    for attempt in range(max_retries):
        try:
            time.sleep(0.1)  # Global sleep to avoid hitting limit across multiple invocations
            return chain.invoke(query)
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate_limit" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
                wait = 2 ** attempt * 15  # 15s → 30s → 60s
                logger.warning(
                    f"Rate limit hit (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait}s..."
                )
                time.sleep(wait)
                last_exc = exc
            else:
                raise
    raise last_exc



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

    # CI mode: only gate on faithfulness to save API quota.
    # 4 metrics × 10 questions = 40+ LLM calls; faithfulness-only = ~10 calls.
    # Faithfulness is the only metric in the CI pass/fail gate.
    if ci_mode:
        logger.info("CI mode: evaluating faithfulness only (saves 75% API quota).")
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
    cache_path = "data/evaluation/rag_cache.json"
    cache = {}
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cache = json.load(f)
        except Exception:
            pass

    for i, qa in enumerate(questions, 1):
        q = qa["question"]
        gt = qa.get("ground_truth", "")

        if q in cache:
            logger.info(f"[{i}/{len(questions)}] Cache hit for query: {q[:30]}...")
            answer = cache[q]["answer"]
            contexts = cache[q]["contexts"]
        else:
            try:
                result = _invoke_with_retry(chain, q)
                answer = result["answer"]
                # Use sources from the same invoke() call so contexts match
                # what the LLM actually received — fixes the faithfulness scoring.
                sources = result.get("sources", [])
                contexts = [doc.page_content for doc in sources if doc.page_content.strip()]
                # Fallback: if compression stripped everything, use raw retrieval
                if not contexts:
                    logger.warning(f"[{i}/{len(questions)}] Empty contexts — using raw retrieval.")
                    from src.rag.hyde import HyDEQueryEnhancer
                    from src.config import settings as cfg
                    raw_docs = chain._get_retriever().retrieve(
                        HyDEQueryEnhancer(enabled=cfg.enable_hyde).enhance(q)
                    )
                    contexts = [doc.page_content for doc in raw_docs if doc.page_content.strip()]
            except Exception as exc:
                logger.warning(f"[{i}/{len(questions)}] Inference failed after retries: {exc}")
                answer = ""
                contexts = []
            
        if q not in cache and answer:
            cache[q] = {"answer": answer, "contexts": contexts}
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f)

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
    # max_workers=1: sequential to respect free-tier rate limits.
    # timeout=600s to allow massive models like Nemotron-550B enough time.
    run_cfg = RunConfig(max_workers=1, timeout=600, max_retries=10)
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
    no_hyde: bool = typer.Option(
        False, "--no-hyde", help="Disable HyDE query enhancement (saves ~50% LLM tokens)."
    ),
    dataset_path: Optional[str] = typer.Option(
        None, "--dataset", "-d", help="Path to custom golden QA JSON file."
    ),
) -> None:
    """Run RAGAS evaluation and display results."""
    settings.configure_langsmith()
    settings.ensure_directories()

    # Temporarily disable HyDE if requested (saves LLM API calls during eval)
    if no_hyde:
        import src.config as _cfg_mod
        _cfg_mod.settings.enable_hyde = False
        logger.info("HyDE disabled for this evaluation run (--no-hyde).")

    from src.evaluation.golden_dataset import load_golden_dataset
    from src.evaluation.metrics_store import MetricsStore

    questions = load_golden_dataset()
    if dataset_path:
        import json
        with open(dataset_path) as f:
            questions = json.load(f)

    import os
    judge_label = (
        "NVIDIA (mistralai/mistral-medium-3.5-128b)"
        if "NVIDIA_GLM_API_KEY" in os.environ
        else (f"Gemini ({settings.gemini_judge_model})" if settings.use_gemini_judge else f"Groq ({settings.groq_model})")
    )
    console.print(
        f"\n[bold cyan]🔬 CERN Knowledge Navigator — RAGAS Evaluation[/bold cyan]\n"
        f"  Questions  : {len(questions)}\n"
        f"  Sample     : {sample or 'all'}\n"
        f"  Judge LLM  : {judge_label}\n"
        f"  Mode       : {'CI (faithfulness only)' if ci else 'Full (4 metrics)'}\n"
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
    if ci:
        table.add_row("Answer Relevancy",  "[dim]N/A (CI)[/dim]", "[dim]—[/dim]")
        table.add_row("Context Precision", "[dim]N/A (CI)[/dim]", "[dim]—[/dim]")
        table.add_row("Context Recall",    "[dim]N/A (CI)[/dim]", "[dim]—[/dim]")
    else:
        table.add_row("Answer Relevancy",  f"{metrics['answer_relevancy']:.4f}",  _status(metrics["answer_relevancy"]))
        table.add_row("Context Precision", f"{metrics['context_precision']:.4f}", _status(metrics["context_precision"]))
        table.add_row("Context Recall",    f"{metrics['context_recall']:.4f}",    _status(metrics["context_recall"]))

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
