import hashlib
from typing import List
class ChunkBuilder:
    def __init__(
        self,
        max_chunk_chars: int = 900,
        overlap_chars: int = 120,
    ):
        self.max_chunk_chars = max_chunk_chars
        self.overlap_chars = overlap_chars
    # =====================================================
    # PUBLIC
    # =====================================================
    def build(
        self,
        law: LawDocument,
    ) -> List[Chunk]:
        chunks = []
        for article in law.articles:
            chunks.extend(
                self._build_article(
                    law,
                    article,
                )
            )
        return chunks
    # =====================================================
    # ARTICLE
    # =====================================================
    def _build_article(
        self,
        law: LawDocument,
        article: Article,
    ) -> List[Chunk]:
        article_text = self._article_text(article)
        if len(article_text) <= self.max_chunk_chars:
            return [
                self._make_chunk(
                    law=law,
                    article=article,
                    clause=None,
                    body=article_text,
                    chunk_index=0,
                    total_chunks=1,
                )
            ]
        chunks = []
        for clause in article.clauses:
            chunks.extend(
                self._build_clause(
                    law,
                    article,
                    clause,
                )
            )
        return chunks
    # =====================================================
    # CLAUSE
    # =====================================================
    def _build_clause(
        self,
        law,
        article,
        clause,
    ):
        body = self._clause_text(clause)
        if len(body) <= self.max_chunk_chars:
            return [
                self._make_chunk(
                    law,
                    article,
                    clause,
                    body,
                    0,
                    1,
                )
            ]
        pieces = self._split_soft(body)
        result = []
        total = len(pieces)
        for idx, piece in enumerate(pieces):
            result.append(
                self._make_chunk(
                    law,
                    article,
                    clause,
                    piece,
                    idx,
                    total,
                )
            )
        return result
    # =====================================================
    # CHUNK
    # =====================================================
    def _make_chunk(
        self,
        law,
        article,
        clause,
        body,
        chunk_index,
        total_chunks,
    ):
        embedding = self._embedding_text(
            law,
            article,
            clause,
            body,
        )
        metadata = self._metadata(
            law,
            article,
            clause,
            body,
            chunk_index,
            total_chunks,
        )
        raw = (
            law.law_code
            + str(article.number)
            + str(clause.number if clause else 0)
            + str(chunk_index)
        )
        chunk_id = hashlib.sha1(
            raw.encode()
        ).hexdigest()
        return Chunk(
            chunk_id=chunk_id,
            text=body,
            embedding_text=embedding,
            metadata=metadata,
        )
    # =====================================================
    # ARTICLE TEXT
    # =====================================================
    def _article_text(
        self,
        article,
    ):
        text = []
        if article.intro:
            text.extend(article.intro)
        for clause in article.clauses:
            text.append(
                self._clause_text(clause)
            )
        return "\n".join(text).strip()
    # =====================================================
    # CLAUSE TEXT
    # =====================================================
    def _clause_text(
        self,
        clause,
    ):
        lines = []
        if clause.title:
            lines.append(clause.title)
        if clause.content:
            lines.append(clause.content)
        for point in clause.points:
            lines.append(
                f"{point.point}) {point.content}"
            )
        return "\n".join(lines)
    # =====================================================
    # SPLIT
    # =====================================================
    def _split_soft(
        self,
        text,
    ):
        if len(text) <= self.max_chunk_chars:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = min(
                start + self.max_chunk_chars,
                len(text),
            )
            if end == len(text):
                chunks.append(
                    text[start:].strip()
                )
                break
            cut = text.rfind(
                "\n",
                start,
                end,
            )
            if cut < 0:
                cut = text.rfind(
                    ". ",
                    start,
                    end,
                )
            if cut < 0:
                cut = end
            chunks.append(
                text[start:cut].strip()
            )
            start = max(
                cut - self.overlap_chars,
                cut,
            )
        return chunks
    # =====================================================
    # EMBEDDING TEXT
    # =====================================================
    def _embedding_text(
        self,
        law,
        article,
        clause,
        body,
    ):
        lines = [
            f"Văn bản: {law.law_name}",
            f"Số hiệu: {law.law_code}",
            f"Loại: {law.document_type}",
        ]
        if article.chapter:
            lines.append(article.chapter)
        if article.section:
            lines.append(article.section)
        lines.append(
            f"Điều {article.number}. {article.title}"
        )
        if clause:
            lines.append(
                f"Khoản {clause.number}"
            )
        lines.append(body)
        return "\n".join(lines)
    # =====================================================
    # METADATA
    # =====================================================
    def _metadata(
        self,
        law,
        article,
        clause,
        body,
        chunk_index,
        total_chunks,
    ):
        return {
            "law_name": law.law_name,
            "law_code": law.law_code,
            "document_type": law.document_type,
            "article": article.number,
            "article_title": article.title,
            "chapter": article.chapter,
            "section": article.section,
            "clause": clause.number if clause else None,
            "effective_date": law.effective_date,
            "status": law.status,
            "keywords": extract_keywords(body),
            "citation": build_citation(
                law,
                article,
                clause,
            ),
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
        }