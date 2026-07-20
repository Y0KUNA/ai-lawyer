from typing import List, Dict


class CoverageEvaluator:
    def evaluate(self, chunks: List[Dict], issue: str) -> float:
        if not chunks:
            return 0.0

        authority_types = set()
        law_ids = set()
        sources = set()
        articles = set()
        for chunk in chunks:
            metadata = chunk.get("metadata", {}) or {}
            authority_types.add(metadata.get("loai_van_ban", "unknown"))
            law_ids.add(metadata.get("so_hieu", metadata.get("ten_luat", "")))
            sources.add(metadata.get("source", chunk.get("source_query", "")))
            articles.add((metadata.get("so_hieu"), metadata.get("dieu")))
            
            articles.add(
                (
                    metadata.get("so_hieu"),
                    metadata.get("dieu"),
                )
            )   
        type_score = min(1.0, len(authority_types) / 5)
        id_score = min(1.0, len(law_ids) / max(1, len(chunks)))
        source_score = min(1.0, len(sources) / max(1, len(chunks)))
        article_score = min(
                    1.0,
                    len(articles) / 8
                )
        scores = [
            c["retrieval_score"]
            for c in chunks
            if c.get("retrieval_score") is not None
        ]

        avg_score = (
            sum(scores)/len(scores)
            if scores else 0
        )
        return (
            type_score*0.2 +
            id_score*0.2 +
            article_score*0.2 +
            source_score*0.1 +
            avg_score*0.3
        )
