from typing import Dict, List
from sentence_transformers import CrossEncoder
import os
class RerankerService:
    def __init__(self):
        self.model = CrossEncoder(
    "BAAI/bge-reranker-v2-m3",
    max_length=512,
    device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
)

    def rerank(self, chunks: List[Dict], query: str, top_k: int = 8) -> List[Dict]:
        if not chunks or not query:
            return []

        
        pairs = [
            (query, chunk["text"])
            for chunk in chunks
        ]
        scores = self.model.predict(pairs)  
        scored_chunks = []

        for chunk, score in zip(chunks, scores):

            c = dict(chunk)

            c["rerank_score"] = float(score)

            scored_chunks.append(c)

        scored_chunks.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return scored_chunks[:top_k]