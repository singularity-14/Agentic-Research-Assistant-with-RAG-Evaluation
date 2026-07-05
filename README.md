<div align="center">

# ⚛️ Agentic Research Assistant with RAG Evaluation

> **Note:** Due to model rate limit errors, the evaluation is currently configured to sample only 2 queries.

**Production-grade RAG system over CERN/HEP scientific papers**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg)](https://fastapi.tiangolo.com)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.5-purple.svg)](https://github.com/jlowin/fastmcp)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.2-orange.svg)](https://ragas.io)
[![LangSmith](https://img.shields.io/badge/LangSmith-observability-green.svg)](https://smith.langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Semantic search over arXiv HEP papers via MCP tools, REST API, and a RAGAS evaluation dashboard*

</div>

---

## 🏗️ Architecture

```
CERN Public Docs (arXiv: hep-ex, hep-ph, physics.acc-ph)
        ↓
  Document Ingestion Pipeline  (async, CLI)
        ↓
  Chunking  (RecursiveCharacterTextSplitter, 512 tokens, 50 overlap)
        ↓
  Embedding  (BAAI/bge-small-en-v1.5 — local HuggingFace, no API key)
        ↓
  Vector Store  (FAISS, local)
        ↓
  ┌────────────────────────────────────────────┐
  │         Hybrid Retriever                   │
  │  BM25 + FAISS Dense (RRF fusion)           │
  │  Cross-encoder reranker (ms-marco-MiniLM)  │
  │  HyDE query enhancement (optional)         │
  └────────────────────────────────────────────┘
        ↓
  ┌────────────────────────────────────────────┐
  │         FastMCP Server                     │
  │  @tool: search_cern_docs()                 │
  │  @tool: get_paper_summary()                │
  │  @tool: list_indexed_categories()          │
  │  @resource: cern://papers/latest           │
  │  @prompt: physics_qa_template              │
  └────────────────────────────────────────────┘
        ↓
  LangChain RAG Chain  (Groq llama-3.3-70b-versatile)
  LangSmith Tracing    (observability)
        ↓
  FastAPI REST API  (:8000)
        ↓
  RAGAS Evaluation Dashboard  (Streamlit :8501)
```

---

## 🗂️ Project Structure

```
cern-knowledge-navigator/
├── .env.example                        # Environment variable template
├── .env                                # Your credentials (never committed)
├── .gitignore
├── .github/
│   └── workflows/
│       └── ragas_eval.yml              # CI: fails PR if faithfulness < 0.8
├── Dockerfile                          # Multi-stage, non-root runtime
├── docker-compose.yml                  # API + MCP + Dashboard services
├── pyproject.toml                      # Project metadata & entry points
├── requirements.txt                    # Pinned dependencies
│
├── src/
│   ├── config.py                       # Pydantic settings (reads .env)
│   ├── ingestion/
│   │   ├── arxiv_fetcher.py            # arXiv API → paper metadata
│   │   ├── chunker.py                  # RecursiveCharacterTextSplitter
│   │   ├── pdf_parser.py               # Optional PyMuPDF full-text
│   │   └── pipeline.py                 # CLI: fetch → chunk → embed → index
│   ├── rag/
│   │   ├── embeddings.py               # BGE-small-en-v1.5 (local)
│   │   ├── vector_store.py             # FAISS build/save/load/search
│   │   ├── retriever.py                # Hybrid BM25+dense + RRF + reranker
│   │   ├── hyde.py                     # HyDE query enhancement
│   │   └── chain.py                    # Full LangChain RAG chain
│   ├── mcp_server/
│   │   ├── server.py                   # FastMCP tools, resources, prompts
│   │   └── auth.py                     # Bearer token middleware
│   ├── api/
│   │   ├── main.py                     # FastAPI app entry point
│   │   ├── routes.py                   # /query /papers /health /metrics
│   │   └── models.py                   # Pydantic request/response schemas
│   └── evaluation/
│       ├── golden_dataset.py           # 50 curated HEP Q&A pairs
│       ├── ragas_runner.py             # RAGAS evaluate() with Groq judge
│       ├── metrics_store.py            # SQLite persistence for scores
│       └── dashboard.py               # Streamlit evaluation dashboard
│
├── data/
│   ├── raw/                            # papers.jsonl (arXiv metadata)
│   ├── processed/                      # Chunked documents
│   ├── faiss_index/                    # FAISS .faiss + .pkl files
│   └── evaluation/
│       ├── golden_qa.json              # 50 Q&A pairs
│       └── results/                    # RAGAS JSONL + metrics.db
│
├── notebooks/
│   └── 01_data_exploration.ipynb
└── tests/
    ├── test_ingestion.py
    ├── test_rag_chain.py
    ├── test_mcp_tools.py
    └── test_evaluation.py
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
Python 3.11+
pip or uv
```

### 2. Clone & Install

```bash
git clone https://github.com/your-username/cern-knowledge-navigator.git
cd cern-knowledge-navigator

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and fill in your credentials:

```bash
copy .env.example .env     # Windows
# cp .env.example .env     # Linux/Mac
```

Edit `.env`:

```env
# Required
GROQ_API_KEY=gsk_your_key_here
LANGSMITH_API_KEY=lsv2_your_key_here
LANGSMITH_PROJECT=cern-knowledge-navigator
LANGSMITH_TRACING=true

# Optional (defaults are fine)
GROQ_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
MAX_RESULTS_DEFAULT=500
```

> **No extra API keys needed** — embeddings run locally via HuggingFace (BGE model, ~30 MB download on first run).

### 4. Ingest Papers

```bash
# Quick start: fetch 100 papers (abstracts only, ~2 min)
python -m src.ingestion.pipeline --max-results 100

# Full ingest: 500 papers (~10 min)
python -m src.ingestion.pipeline --max-results 500

# With full PDF text (slow, requires --full-pdf flag)
python -m src.ingestion.pipeline --max-results 50 --full-pdf

# Specific categories only
python -m src.ingestion.pipeline --categories hep-ex --max-results 200
```

### 5. Start the REST API

```bash
python -m src.api.main
# → http://localhost:8000
# → http://localhost:8000/docs  (Swagger UI)
```

### 6. Start the MCP Server

```bash
# stdio mode (Claude Desktop / Cursor integration)
python -m src.mcp_server.server

# HTTP/SSE mode (remote agent integration)
python -m src.mcp_server.server --http
```

### 7. Run Evaluation Dashboard

```bash
streamlit run src/evaluation/dashboard.py
# → http://localhost:8501
```

---

## 📡 API Reference

### `POST /query`

Ask a question against the CERN knowledge base.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the measured Higgs boson mass at ATLAS?",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "query": "What is the measured Higgs boson mass at ATLAS?",
  "answer": "The ATLAS experiment measured the Higgs boson mass to be...",
  "sources": [
    {
      "arxiv_id": "2207.00043",
      "title": "Measurement of the Higgs boson mass...",
      "category": "hep-ex",
      "published_date": "2022-07-01",
      "rerank_score": 0.94
    }
  ],
  "source_count": 5,
  "model_used": "llama-3.3-70b-versatile"
}
```

### `GET /papers`

```bash
curl "http://localhost:8000/papers?category=hep-ex&limit=20"
```

### `GET /papers/{arxiv_id}`

```bash
curl http://localhost:8000/papers/2401.12345
```

### `GET /health`

```bash
curl http://localhost:8000/health
```

### `GET /metrics`

Returns latest RAGAS evaluation scores:

```bash
curl http://localhost:8000/metrics
```

---

## 🤖 MCP Integration

### Claude Desktop

Add to `claude_desktop_config.json` (`%APPDATA%\Claude\` on Windows):

```json
{
  "mcpServers": {
    "cern-navigator": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "F:\\Github\\New folder"
    }
  }
}
```

### Cursor IDE

Add to `.cursor/mcp.json` in your project:

```json
{
  "mcpServers": {
    "cern-navigator": {
      "command": "python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "F:\\Github\\New folder"
    }
  }
}
```

### Available MCP Tools

| Tool | Description |
|---|---|
| `search_cern_docs(query, top_k)` | Semantic RAG search over HEP papers |
| `get_paper_summary(arxiv_id)` | AI summary of a specific arXiv paper |
| `list_indexed_categories()` | Show indexed physics categories |

**Resource:** `cern://papers/latest` — Live feed of last 7 days of HEP papers

---

## 📊 RAGAS Evaluation

Run evaluation against the 50-question golden dataset:

```bash
# Full evaluation (50 questions, ~15-20 min with Groq rate limits)
python -m src.evaluation.ragas_runner

# Quick CI subset (2 questions, ~1 min)
python -m src.evaluation.ragas_runner --sample 2

# CI mode: exits with code 1 if faithfulness < 0.8
python -m src.evaluation.ragas_runner --sample 2 --ci
```

### Metrics

| Metric | Description | Target |
|---|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? | ≥ 0.80 |
| **Answer Relevancy** | Does the answer address the query? | ≥ 0.70 |
| **Context Precision** | Are retrieved chunks actually relevant? | ≥ 0.70 |
| **Context Recall** | Does context contain necessary information? | ≥ 0.70 |

> **Judge LLM**: `llama-3.1-8b-instant` via Groq (free tier, fast inference)

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Services:
#   API       → http://localhost:8000
#   MCP (SSE) → http://localhost:8001
#   Dashboard → http://localhost:8501

# Stop
docker-compose down
```

### Required GitHub Secrets (for CI)

Add to `Settings → Secrets → Actions`:

| Secret | Value |
|---|---|
| `GROQ_API_KEY` | Your Groq API key |
| `LANGSMITH_API_KEY` | Your LangSmith API key |

---

## 🧪 Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific suite
pytest tests/test_ingestion.py -v
pytest tests/test_evaluation.py -v
```

---

## 🔧 Configuration Reference

All settings are loaded from `.env` via `src/config.py` (Pydantic `BaseSettings`).

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Generation LLM |
| `GROQ_JUDGE_MODEL` | `llama-3.1-8b-instant` | RAGAS judge LLM |
| `LANGSMITH_API_KEY` | — | LangSmith observability |
| `LANGSMITH_PROJECT` | `cern-knowledge-navigator` | LangSmith project name |
| `LANGSMITH_TRACING` | `true` | Enable LangSmith tracing |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Local HF embedding model |
| `MAX_RESULTS_DEFAULT` | `500` | Papers to ingest |
| `ARXIV_CATEGORIES` | `hep-ex,hep-ph,physics.acc-ph` | Target arXiv categories |
| `TOP_K_RETRIEVAL` | `10` | Retrieval pool size |
| `TOP_K_RERANK` | `5` | Final results after reranking |
| `ENABLE_HYDE` | `true` | HyDE query enhancement |
| `MCP_AUTH_TOKEN` | `changeme` | Bearer token for MCP HTTP |
| `FAITHFULNESS_THRESHOLD` | `0.8` | CI gate threshold |

---

## 🏅 Tech Stack

| Component | Technology |
|---|---|
| **LLM (generation)** | Groq `llama-3.3-70b-versatile` |
| **RAGAS judge** | Groq `llama-3.1-8b-instant` |
| **Embeddings** | `BAAI/bge-small-en-v1.5` (local) |
| **Cross-encoder reranker** | `ms-marco-MiniLM-L-6-v2` (local) |
| **Vector store** | FAISS (local) |
| **Hybrid search** | BM25 + FAISS, RRF fusion |
| **RAG framework** | LangChain 0.3 |
| **MCP server** | FastMCP 2.5 |
| **REST API** | FastAPI + Uvicorn |
| **Evaluation** | RAGAS 0.2 |
| **Dashboard** | Streamlit + Plotly |
| **Observability** | LangSmith |
| **Containerisation** | Docker + Docker Compose |
| **CI/CD** | GitHub Actions |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
