# PWC RAG App (POC)

Modular Retrieval-Augmented Generation proof-of-concept tailored to PwC-style AI/ML engineering workflows. The current sprint focused on laying the foundation for LangChain/LangGraph integration, chunk management, and observability.

## Architecture Overview

```text
Documents → Chunker → Embeddings → ChromaDB → Retrieval → LLM Routing → Answer
```

* **`backend/app/config.py`** – Centralized dataclass configuration for embeddings, LLM tiers, chunking, retrieval, vector DB, and observability knobs.
* **`backend/app/pipeline/chunker.py`** – Recursive text splitter that preserves metadata, overlap, and chunk indices for downstream tracking.
* **`backend/app/rag.py`** – Core pipeline: indexes sample docs into ChromaDB, retrieves via cosine similarity, routes queries across Router/Worker/Synthesis tiers (Ollama), and emits latency metrics.
* **`backend/app/main.py`** – FastAPI service with CORS, `/query` plus observability endpoints (`/stats`, `/metrics`, `/config`).
* **`backend/requirements.txt`** – FastAPI, ChromaDB, Ollama client, plus LangChain/LangGraph, BM25, tiktoken, dotenv.

Front-end (`frontend/`) and document storage (`data/`) are placeholders for future integration.

## Prerequisites

1. Python 3.11+ (tested on 3.13 via uv) and `pip`.
2. [Ollama](https://ollama.com) running locally (`ollama serve`).
3. Pull lightweight models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5:3b
   ```
   (Optional) For higher quality tiers later: `gemma3:12b`, `mistral-small`.

## Setup & Run

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Overview

| Endpoint | Method | Description |
| --- | --- | --- |
| `/` | GET | Health + version info |
| `/query` | POST | Run RAG pipeline. Body fields: `question`, `use_routing` (bool), `top_k` (int), `strategy` (`semantic`, `hybrid`, `adaptive` – semantic active today). Returns answer, citations, latency metrics. |
| `/stats` | GET | Vector store stats (doc count, collection, metric). |
| `/metrics` | GET | Last 10 query metrics (latency breakdown, model tier). |
| `/config` | GET | Current config snapshot (embedding/LLM/retrieval/chunking settings). |

### Example Query

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "use_routing": true, "top_k": 3}' | jq .
```

### Observability

```bash
curl http://localhost:8000/stats   # vector store stats
curl http://localhost:8000/metrics # recent latency + model usage
curl http://localhost:8000/config  # live configuration snapshot
```

## Current Capabilities

- Config-driven multi-model routing (Router/Worker/Synthesis) using Ollama.
- ChromaDB persistent vector store with cosine similarity and chunk metadata.
- Recursive chunker ready for ingestion workflows.
- Latency metrics capture (embedding/retrieval/generation) surfaced via API.
- LangChain/LangGraph dependencies installed for upcoming agentic workflows.

## Roadmap / Next Steps

1. Integrate chunker + ingestion endpoints so new documents can be uploaded and tracked in real time.
2. Implement hybrid/adaptive retrieval (semantic + BM25, learnable routing) leveraging `RetrievalStrategy` enum.
3. Introduce LangGraph state machine for agentic orchestration (router → retriever → synthesis).
4. Add token usage tracking (tiktoken), structured logging, and evaluation harness.
5. Build lightweight frontend (Vite/React) calling the enhanced API.
6. Add CI + tests (unit + integration) for the modular pipeline.
