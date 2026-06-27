# AI Lawyer

AI Lawyer is a local legal assistant web app with a React frontend and a FastAPI backend. The backend talks to Ollama, streams model output to the UI, and can attach retrieval context from a local ChromaDB store.

## Project Layout

- `frontend/` - Vite + React UI
- `backend/` - FastAPI API, Ollama stream proxy, and RAG lookup
- `rag/` - local vector database and indexing assets used by the backend

## Features

- Chat UI with streaming assistant responses
- FastAPI `/chat` endpoint
- Ollama integration using `http://localhost:11434/api/chat`
- Optional RAG context from ChromaDB

## Prerequisites

- Node.js 18+
- Python 3.11+
- Ollama installed and running
- A pulled Ollama model named `gemma4:e2b`

## Environment Variables

Create a `backend/.env` file if you need to override defaults:

```env
SYSTEM_PROMPT={"role":"system","content":"You are a legal assistant."}
CHROMA_PATH=../rag/chroma_db
EMBED_MODEL=BAAI/bge-m3
```

If `SYSTEM_PROMPT` is not set, the backend falls back to its built-in Vietnamese legal assistant prompt.

## Install and Run

### 1. Start Ollama

Make sure Ollama is running locally and the model exists:

```bash
ollama pull gemma4:e2b
ollama serve
```

### 2. Run the backend

From the `backend/` directory:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install chromadb sentence-transformers
uvicorn main:app --reload --port 8000
```

The extra `chromadb` and `sentence-transformers` installs are required by the current backend code, even though they are not listed in `backend/requirements.txt` yet.

### 3. Run the frontend

From the `frontend/` directory:

```bash
npm install
npm run dev
```

By default the UI talks to `http://localhost:8000/chat`.

## API

### `POST /chat`

Request body:

```json
{
  "messages": [
    { "role": "user", "content": "..." }
  ]
}
```

Response:

- `text/plain` stream of assistant text chunks

### `GET /ping`

Health check endpoint.

## Notes

- The backend currently streams plain text chunks, so the frontend reads `response.body` instead of calling `res.json()`.
- The ChromaDB data lives under `rag/chroma_db/` by default.
- If you replace the model name, update the backend Ollama request in `backend/main.py`.
