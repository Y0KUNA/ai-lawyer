from typing import List
from .evidence import Evidence


class CoverageEvaluator:
    def evaluate(self, chunks: List[Evidence], issue: str) -> float:
        if not chunks:
            return 0.0

        def score_of(chunk: Evidence) -> float:
            return chunk.score or 0.0

        relevance_scores = [score_of(chunk) for chunk in chunks]
        relevance = sum(relevance_scores) / len(relevance_scores)

        def metadata_of(chunk: Evidence):
            return chunk.metadata or {}

        has_specific_article = any(
            metadata_of(chunk).get("dieu") for chunk in chunks
        )
        specificity_bonus = 0.1 if has_specific_article else 0.0

        return min(1.0, relevance * 0.9 + specificity_bonus)
