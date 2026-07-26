from typing import List
from sentence_transformers import CrossEncoder
import os

from .evidence import Evidence


class RerankerService:
    def __init__(self):
        self.model = CrossEncoder(
            "BAAI/bge-reranker-v2-m3",
            max_length=512,
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
        )

    @staticmethod
    def _get_chunk_text(chunk: Evidence) -> str:
        return str(chunk.text or "")

    def rerank(self, chunks: List[Evidence], query: str, top_k: int = 8) -> List[Evidence]:
        if not chunks or not query:
            return []

        pairs = [
            (query, self._get_chunk_text(chunk))
            for chunk in chunks
        ]
        scores = self.model.predict(pairs)
        scored_chunks = []

        for chunk, score in zip(chunks, scores):
            scored_chunks.append(
                Evidence(
                    text=chunk.text,
                    source=chunk.source,
                    law_id=chunk.law_id,
                    article=chunk.article,
                    clause=chunk.clause,
                    point=chunk.point,
                    authority_type=chunk.authority_type,
                    score=float(score),
                    retrieval_source=chunk.retrieval_source,
                    source_query=chunk.source_query,
                    metadata=chunk.metadata or {},
                )
            )

        scored_chunks.sort(
            key=lambda x: x.score,
            reverse=True
        )

        return scored_chunks[:top_k]