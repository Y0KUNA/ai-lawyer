"""
domain_relevance_filter.py
------------------------------------------------------------
Lọc bỏ các chunk đến từ những bộ luật "chuyên biệt" (áp dụng cho một
nhóm đối tượng hẹp: phạm nhân, quân nhân, doanh nghiệp bảo hiểm...)
khi câu hỏi của người dùng KHÔNG thực sự hỏi về nhóm đối tượng đó.

Lý do cần bộ lọc này: nhiều luật chuyên biệt có các điều khoản dùng
từ ngữ giống hệt luật lao động/dân sự phổ thông (ví dụ "bảo hiểm xã
hội", "hưởng chế độ"), khiến embedding + reranker bị đánh lừa bởi
tương đồng câu chữ bề mặt, xếp nhầm các chunk này lên hạng cao dù
sai hoàn toàn đối tượng áp dụng.

Cách dùng: gọi DomainRelevanceFilter.filter_chunks(chunks, issue)
ngay sau (hoặc thay cho) LawDocumentFilter.filter_chunks(chunks) ở
những vị trí tương ứng trong retrieval_pipeline_service.py.
------------------------------------------------------------
"""

import unicodedata
from typing import Dict, List

from .evidence import Evidence


def _normalize(text: str) -> str:
    """Lowercase + bỏ dấu, để so khớp từ khóa không phụ thuộc cách gõ dấu."""
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text


# ------------------------------------------------------------------
# Danh sách luật "chuyên biệt": chỉ giữ lại chunk từ các luật này nếu
# câu hỏi (issue) chứa ít nhất 1 từ khóa xác nhận đúng domain.
#
# Khớp theo SUBSTRING trên `law_name` đã chuẩn hóa (không dấu, thường),
# nên "Luật Thi hành án hình sự" và "LUẬT THI HÀNH ÁN HÌNH SỰ" đều khớp.
#
# TODO: bổ sung thêm domain khác nếu phát hiện case tương tự khi test
# (ví dụ Luật Xử lý vi phạm hành chính, Luật Đặc xá...).
# ------------------------------------------------------------------
_SPECIALIZED_DOMAINS: Dict[str, List[str]] = {
    "thi hanh an hinh su": [
        "pham nhan", "thi hanh an hinh su", "trai giam", "chap hanh an",
        "tu nhan", "an phat tu", "trai tam giam",
    ],
    "nghia vu quan su": [
        "nghia vu quan su", "quan nhan", "binh si", "ha si quan",
        "quan doi", "nhap ngu", "xuat ngu", "tai ngu",
    ],
    "kinh doanh bao hiem": [
        "kinh doanh bao hiem", "doanh nghiep bao hiem", "bao hiem xe co gioi",
        "bao hiem chay no", "bao hiem phi nhan tho", "hop dong bao hiem thuong mai",
        "moi gioi bao hiem",
    ],
}


class DomainRelevanceFilter:
    @staticmethod
    def _get_law_name(chunk: Evidence) -> str:
        metadata = chunk.metadata or {}
        return str(metadata.get("law_name", "") or "")

    @staticmethod
    def is_domain_mismatch(chunk: Evidence, issue: str) -> bool:
        """True nếu chunk thuộc 1 domain chuyên biệt NHƯNG issue không
        có từ khóa xác nhận domain đó -> nên loại bỏ chunk này."""
        law_name_norm = _normalize(DomainRelevanceFilter._get_law_name(chunk))
        if not law_name_norm:
            return False

        issue_norm = _normalize(issue or "")

        for domain_key, trigger_keywords in _SPECIALIZED_DOMAINS.items():
            if domain_key in law_name_norm:
                has_trigger = any(kw in issue_norm for kw in trigger_keywords)
                return not has_trigger  # mismatch nếu KHÔNG có từ khóa xác nhận

        return False

    @staticmethod
    def filter_chunks(chunks: List[Evidence], issue: str) -> List[Evidence]:
        if not chunks:
            return []
        return [
            chunk for chunk in chunks
            if not DomainRelevanceFilter.is_domain_mismatch(chunk, issue)
        ]