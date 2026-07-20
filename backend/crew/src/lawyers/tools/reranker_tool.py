from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import List, Type
from sentence_transformers import SentenceTransformer
import numpy as np


class RerankerInput(BaseModel):
    chunks: List[dict]
    query: str
    top_k: int = 8


class RerankerTool(BaseTool):
    """Reranks chunks using an embedding similarity model and returns top_k chunks."""
    name : str = "RerankerTool"
    description : str = "Reranks a list of chunks against a query and returns the top_k items."
    args_schema: Type[BaseModel]  = RerankerInput

    _embed = None

    def _ensure_model(self):
        if self._embed is None:
            self._embed = SentenceTransformer("BAAI/bge-m3")

    def _run(self, chunks: List[dict], query: str, top_k: int = 8) -> List[dict]:
        if not chunks:
            return []
        self._ensure_model()
        q_vec = self._embed.encode(query)
        scores = []
        for c in chunks:
            text = c.get("text", "")
            c_vec = self._embed.encode(text)
            # cosine
            denom = np.linalg.norm(q_vec) * np.linalg.norm(c_vec)
            sim = float(np.dot(q_vec, c_vec) / denom) if denom > 0 else 0.0
            scores.append((sim, c))

        scores.sort(key=lambda x: x[0], reverse=True)
        top = [c for _, c in scores[:top_k]]
        return top
