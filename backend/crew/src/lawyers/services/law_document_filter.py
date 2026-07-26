import re
from typing import Dict, List, Optional

from .evidence import Evidence

# Các loại văn bản luật được phép trong retrieval pipeline.
LAW_DOCUMENT_TYPES = frozenset({
    "luật",
    "bộ luật",
    "hiến pháp",
})

# Giá trị document_type trong ChromaDB (khác hoa/thường).
CHROMA_LAW_TYPES = [
    "Luật",
    "Bộ luật",
    "Hiến pháp",
    "LUẬT",
    "BỘ LUẬT",
    "HIẾN PHÁP",
]

# Dấu hiệu nội dung thuộc án lệ / bản án / tài liệu tố tụng — loại khỏi pipeline.
CASE_LAW_MARKERS = (
    "án lệ",
    "an le",
    "bản án",
    "ban an",
    "quyết định số",
    "quyet dinh so",
    "tòa án nhân dân",
    "toa an nhan dan",
    "hội đồng xét xử",
    "hoi dong xet xu",
    "phán quyết",
    "phan quyet",
    "tài liệu tố tụng",
    "tai lieu to tung",
    "bị cáo",
    "bi cao",
    "nguyên đơn",
    "nguyen don",
    "bị đơn",
    "bi don",
)

# Dấu hiệu tích cực cho văn bản luật (dùng khi chunk không có metadata document_type).
LAW_TEXT_MARKERS = (
    "bộ luật",
    "bo luat",
    "hiến pháp",
    "hien phap",
    "luật số",
    "luat so",
    "điều ",
    "dieu ",
    "/qh",
)

_RE_LAW_NUMBER = re.compile(r"\b\d{2,3}/\d{4}/QH\d+\b", re.IGNORECASE)
_RE_ARTICLE = re.compile(r"\bĐiều\s+\d+", re.IGNORECASE)


class LawDocumentFilter:
    @staticmethod
    def chroma_where_clauses() -> List[Optional[Dict]]:
        """Thử lần lượt: lọc luật + hiệu lực → chỉ hiệu lực → không lọc metadata."""
        return [
            {
                "$and": [
                    {"status": {"$ne": "het_hieu_luc"}},
                    {"document_type": {"$in": CHROMA_LAW_TYPES}},
                ]
            },
            {"status": {"$ne": "het_hieu_luc"}},
            None,
        ]

    @staticmethod
    def _get_text(chunk: Evidence) -> str:
        return chunk.text or ""

    @staticmethod
    def _get_metadata(chunk: Evidence) -> Dict:
        return chunk.metadata or {}

    @staticmethod
    def _get_retrieval_source(chunk: Evidence) -> str:
        return chunk.retrieval_source or ""

    @staticmethod
    def is_case_law_text(text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        return any(marker in lower for marker in CASE_LAW_MARKERS)

    @staticmethod
    def is_law_document_type(document_type: str) -> bool:
        normalized = (document_type or "").strip().lower()
        if not normalized:
            return False
        return any(law_type in normalized for law_type in LAW_DOCUMENT_TYPES)

    @staticmethod
    def has_law_text_signals(text: str) -> bool:
        if not text:
            return False
        lower = text.lower()
        if any(marker in lower for marker in LAW_TEXT_MARKERS):
            return True
        if _RE_LAW_NUMBER.search(text):
            return True
        if _RE_ARTICLE.search(text):
            return True
        return False

    @classmethod
    def keep_chunk(cls, chunk: Evidence) -> bool:
        metadata = cls._get_metadata(chunk)
        text = cls._get_text(chunk)
        # `article_title` is the chunker's field for the Điều's heading
        # (e.g. "Đặt cọc" for "Điều 328. Đặt cọc").
        heading = str(metadata.get("article_title", "") or "")
        combined = f"{heading}\n{text}"

        if cls.is_case_law_text(combined):
            return False

        document_type = str(metadata.get("document_type", "") or "")
        if document_type:
            return cls.is_law_document_type(document_type)

        return cls.has_law_text_signals(combined)

    @classmethod
    def filter_chunks(cls, chunks: List[Evidence]) -> List[Evidence]:
        return [chunk for chunk in chunks if cls.keep_chunk(chunk)]

    @staticmethod
    def law_focused_query(query: str) -> str:
        base = (query or "").strip()
        if not base:
            return base
        lowered = base.lower()
        if any(term in lowered for term in ("luật", "luat", "bộ luật", "bo luat", "điều", "dieu")):
            return f"{base} chỉ tìm văn bản luật, không án lệ, không bản án"
        return f"{base} luật điều khoản văn bản pháp luật chỉ tìm văn bản luật, không án lệ, không bản án, không quyết định tòa án"