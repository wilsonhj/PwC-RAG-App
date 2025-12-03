# PWC RAG App (POC)

Minimal Retrieval-Augmented Generation (RAG) proof-of-concept tailored to PwC-style AI consulting work.

## Structure

- `backend/`
  - `app/main.py` – FastAPI app exposing a `/query` endpoint.
  - `app/rag.py` – Extremely simple in-memory "RAG" over a few hard-coded chunks.
  - `requirements.txt` – Python dependencies.
- `data/` – Placeholder for documents if you later move beyond in-memory chunks.
- `frontend/` – Placeholder for a UI client (e.g., React/Next.js or simple SPA).

## Running the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Then, test the API:

```bash
curl -X POST \
  http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What does this POC demonstrate?"}'
```

You should receive an answer plus a list of cited chunk IDs.

## Next steps

- Replace the naive keyword retriever with embeddings + a vector store.
- Connect to a local or hosted LLM and pass retrieved chunks into the prompt.
- Add a simple frontend (e.g., Vite/React) calling the `/query` endpoint.
- Introduce evaluation, logging, and observability once the basics work.
