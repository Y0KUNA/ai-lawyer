import re
from dataclasses import dataclass
from typing import List, Optional

from .evidence import Evidence


@dataclass(frozen=True)
class ArticleRef:
    dieu: Optional[int] = None
    khoan: Optional[int] = None
    diem: Optional[str] = None

    def is_specific(self) -> bool:
        return self.dieu is not None or self.khoan is not None or self.diem is not None

    def satisfies(self, found: "ArticleRef") -> bool:
        """True when `found` matches every level specified in this requirement."""
        if self.dieu is not None and found.dieu != self.dieu:
            return False
        if self.khoan is not None and found.khoan != self.khoan:
            return False
        if self.diem is not None and found.diem != self.diem:
            return False
        return True


class CitationVerifierService:
    _RE_DIEU = re.compile(r"Điều\s+(\d+)", re.IGNORECASE)
    _RE_KHOAN = re.compile(r"Khoản\s+(\d+)", re.IGNORECASE)
    _RE_DIEM = re.compile(r"Điểm\s+([a-zA-ZđĐ])", re.IGNORECASE)
    _RE_ARTICLE_CITATION = re.compile(
        r"Điều\s+(\d+)"
        r"(?:[\.\:]?\s*Khoản\s+(\d+))?"
        r"(?:[\.\:]?\s*Điểm\s+([a-zA-ZđĐ]))?",
        re.IGNORECASE,
    )

    def verify(
        self,
        chunks: List[Evidence],
        issue: str,
        document_hints: Optional[List[str]] = None,
    ) -> List[Evidence]:
        if not chunks:
            return []

        required_refs = self.extract_article_refs(issue)
        if required_refs:
            return [
                chunk
                for chunk in chunks
                if self._chunk_matches_article_refs(chunk, required_refs)
            ]

        hints = [h.strip() for h in (document_hints or []) if h and h.strip()]
        if hints:
            return [chunk for chunk in chunks if self._matches_document_hint(chunk, hints)]

        return chunks

    def extract_article_refs(self, text: str) -> List[ArticleRef]:
        if not text:
            return []

        refs: List[ArticleRef] = []
        seen = set()
        for match in self._RE_ARTICLE_CITATION.finditer(text):
            dieu = int(match.group(1))
            khoan = int(match.group(2)) if match.group(2) else None
            diem = match.group(3).lower() if match.group(3) else None
            ref = ArticleRef(dieu=dieu, khoan=khoan, diem=diem)
            key = (ref.dieu, ref.khoan, ref.diem)
            if key not in seen:
                seen.add(key)
                refs.append(ref)
        return refs

    def extract_chunk_refs(self, chunk: Evidence) -> List[ArticleRef]:
        metadata = chunk.metadata or {}
        text = chunk.text or ""
        heading = (
            str(metadata.get("heading", "") or "")
            if isinstance(metadata, dict)
            else ""
        )
        refs: List[ArticleRef] = []
        seen = set()

        meta_dieu = metadata.get("article_number") or metadata.get("dieu")
        if meta_dieu not in (None, ""):
            try:
                refs.append(ArticleRef(dieu=int(meta_dieu)))
            except (TypeError, ValueError):
                pass

        meta_khoan = metadata.get("clause_number") or metadata.get("khoan")
        if meta_khoan not in (None, ""):
            try:
                refs.append(ArticleRef(dieu=refs[0].dieu if refs else None, khoan=int(meta_khoan)))
            except (TypeError, ValueError):
                pass

        meta_diem = metadata.get("point") or metadata.get("diem")
        if meta_diem not in (None, ""):
            refs.append(ArticleRef(dieu=refs[0].dieu if refs else None, khoan=refs[0].khoan if refs else None, diem=str(meta_diem)))

        for source in (heading, text):
            for match in self._RE_ARTICLE_CITATION.finditer(source):
                dieu = int(match.group(1))
                khoan = int(match.group(2)) if match.group(2) else None
                diem = match.group(3).lower() if match.group(3) else None
                ref = ArticleRef(dieu=dieu, khoan=khoan, diem=diem)
                key = (ref.dieu, ref.khoan, ref.diem)
                if key not in seen:
                    seen.add(key)
                    refs.append(ref)

        if not refs and isinstance(metadata, dict) and metadata.get("section_type") == "dieu":
            dieu_match = self._RE_DIEU.search(heading)
            if dieu_match:
                refs.append(ArticleRef(dieu=int(dieu_match.group(1))))

        return refs

    @staticmethod
    def _get_chunk_text(chunk: Evidence) -> str:
        return str(chunk.text or "")

    @staticmethod
    def _get_chunk_metadata(chunk: Evidence):
        return chunk.metadata or {}

    def _chunk_matches_article_refs(self, chunk: Evidence, required_refs: List[ArticleRef]) -> bool:
        chunk_refs = self.extract_chunk_refs(chunk)
        if chunk_refs:
            return any(
                required.satisfies(found)
                for required in required_refs
                for found in chunk_refs
            )

        text = self._get_chunk_text(chunk)
        metadata = self._get_chunk_metadata(chunk)
        searchable = " ".join(
            str(part)
            for part in (text, metadata.get("heading", ""))
            if part
        )
        for required in required_refs:
            if required.dieu is not None and re.search(
                rf"\bĐiều\s+{required.dieu}(?:[\.\:]|\b)",
                searchable,
                re.IGNORECASE,
            ):
                if required.khoan is None or re.search(
                    rf"\bKhoản\s+{required.khoan}(?:[\.\:]|\b)",
                    searchable,
                    re.IGNORECASE,
                ):
                    if required.diem is None or re.search(
                        rf"\bĐiểm\s+{re.escape(required.diem)}(?:[\.\:]|\b)",
                        searchable,
                        re.IGNORECASE,
                    ):
                        return True
        return False

    def _matches_document_hint(self, chunk: Evidence, document_hints: List[str]) -> bool:
        metadata = self._get_chunk_metadata(chunk)
        haystack = " ".join(
            str(part)
            for part in (
                self._get_chunk_text(chunk),
                metadata.get("law_code", ""),
                metadata.get("law_name", ""),
                metadata.get("article_title", ""),
                metadata.get("heading", ""),
            )
            if part
        )
        return any(hint in haystack for hint in document_hints)
