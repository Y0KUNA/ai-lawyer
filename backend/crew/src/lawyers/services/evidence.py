from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Evidence:
    text: str
    source: str
    law_id: Optional[str] = None
    article: Optional[int] = None
    clause: Optional[str] = None
    point: Optional[str] = None
    authority_type: str = "law_document"
    score: float = 0.0
    retrieval_source: str = "chroma"
    source_query: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @staticmethod
    def _parse_int(value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @classmethod
    def from_dict(cls, chunk: Dict[str, Any], default_source: str = "chroma") -> "Evidence":
        metadata = chunk.get("metadata", {}) or {}
        law_id = metadata.get("law_code") or metadata.get("so_hieu") or metadata.get("id") or chunk.get("id")
        article_number = metadata.get("article_number") or metadata.get("dieu") or chunk.get("article")
        clause_number = metadata.get("clause_number") or metadata.get("khoan") or chunk.get("clause")
        point_value = metadata.get("point") or metadata.get("diem") or chunk.get("point")
        return cls(
            text=str(chunk.get("text", "") or ""),
            source=str(chunk.get("source_query", "") or metadata.get("source", "")) or default_source,
            law_id=str(law_id) if law_id is not None else None,
            article=cls._parse_int(article_number),
            clause=str(clause_number) if clause_number not in (None, "") else None,
            point=str(point_value) if point_value not in (None, "") else None,
            authority_type=str(metadata.get("authority_type", "law_document") or "law_document"),
            score=float(chunk.get("rerank_score", chunk.get("retrieval_score", 0.0)) or 0.0),
            retrieval_source=str(chunk.get("retrieval_source", default_source) or default_source),
            source_query=str(chunk.get("source_query") or "") if chunk.get("source_query") not in (None, "") else None,
            metadata=metadata,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "source": self.source,
            "law_id": self.law_id,
            "article": self.article,
            "clause": self.clause,
            "point": self.point,
            "authority_type": self.authority_type,
            "score": self.score,
            "retrieval_source": self.retrieval_source,
            "source_query": self.source_query,
            "metadata": self.metadata or {},
        }

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "law_id": self.law_id,
            "article": self.article,
            "clause": self.clause,
            "point": self.point,
            "authority_type": self.authority_type,
            "score": round(self.score, 4),
            "retrieval_source": self.retrieval_source,
            "source": self.source,
        }
