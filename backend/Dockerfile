FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt backend/
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r backend/requirements.txt

COPY backend/main.py backend/

ENV CHROMA_PATH=/app/rag/chroma_db \
    EMBED_MODEL=BAAI/bge-m3 \
    OLLAMA_HOST=http://ollama:11434 \
    HF_HOME=/root/.cache/huggingface

EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
