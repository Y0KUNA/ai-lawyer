FROM python:3.13-slim

WORKDIR /app

RUN pip install uv


COPY pyproject.toml uv.lock ./

RUN uv sync

COPY src ./src
COPY knowledge ./knowledge

ENV PYTHONPATH=/app/src

CMD ["uv", "run", "uvicorn",  "movie_to_novel_adapter.api:app",  "--host", "0.0.0.0",  "--port","8000"]