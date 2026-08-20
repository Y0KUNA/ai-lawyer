import logging
import re
from typing import Dict, List, Optional

from crewai import LLM
from .evidence import Evidence
from .query_expansion_service import QueryExpansionService
from .hyde_service import HyDEService
from .rag_service import RAGService
from .reranker_service import RerankerService
from .citation_service import CitationVerifierService
from .coverage_service import CoverageEvaluator
from .normalize_service import NormalizeService
from .law_document_filter import LawDocumentFilter
from .domain_relevance_filter import DomainRelevanceFilter
from concurrent.futures import ThreadPoolExecutor, as_completed
logger = logging.getLogger(__name__)

# ---- Cấu hình cho bước compact/trim -----------------------------------
# Các trường metadata KHÔNG mang giá trị nội dung (debug/vận hành/không dùng
# tới trong bước suy luận của agent) -> loại bỏ khỏi output cuối cùng.
_METADATA_DROP_KEYS = {
    "id", "Favicon", "Image", "Extras", "Subpages", "Crawl Date",
    "Summary", "Highlights", "Search Time", "CostDollars", "Score",
    "retrieval_rank", "retrieval_score", "rerank_score",
}

# Các trường metadata luật muốn giữ lại (nếu có)
_LAW_METADATA_KEEP_KEYS = {
    "heading", "law_code", "ten_luat", "loai_van_ban", "status", "ngay_hieu_luc",
}

# Nếu 1 chunk raw-text (dạng scrape Exa: Title/URL/.../Text/...) dài hơn
# ngưỡng này thì mới áp dụng trích lọc đoạn liên quan thay vì giữ nguyên.
_LONG_TEXT_CHAR_THRESHOLD = 4000
# Khi trích lọc đoạn liên quan, giữ tối đa n ký tự quanh mỗi từ khóa khớp.
_KEYWORD_WINDOW_CHARS = 2000


class RetrievalPipelineService:
    def __init__(self, llm: LLM):
        self.query_expander = QueryExpansionService()
        self.hyde = HyDEService(llm)
        self.rag = RAGService()
        self.reranker = RerankerService()
        self.citation_verifier = CitationVerifierService()
        self.coverage_evaluator = CoverageEvaluator()
        self.normalizer = NormalizeService()

    def run(self, issue: str, n_queries: int = 3, n_results: int = 8, top_k: int = 8, coverage_threshold: float = 0.5) -> Dict:
        issue = self.normalizer.normalize_law_name(issue)
        issue = self.normalizer.normalize_article(issue)
        queries = self.query_expander.expand(issue, n_queries)
        logger.info("Query expansion generated %d queries", len(queries))

        chunks = self.rag.search(queries, n_results=n_results)
        chunks = LawDocumentFilter.filter_chunks(chunks)
        chunks = DomainRelevanceFilter.filter_chunks(chunks, issue)
        logger.info("RAG returned %d law-only evidences", len(chunks))
        logger.info("RAG evidence list: %s", [e.to_log_dict() for e in chunks])

        coverage = self.coverage_evaluator.evaluate(chunks, issue)
        logger.info("Coverage score: %.3f", coverage)
        hyde_queries = []

        if coverage < coverage_threshold:
            hyde_queries = self.hyde.generate(issue, 2)
            queries = list(dict.fromkeys(
                queries + hyde_queries
            ))
            chunks = self.rag.search(
                queries,
                n_results=n_results,
            )
            chunks = LawDocumentFilter.filter_chunks(chunks)
            chunks = DomainRelevanceFilter.filter_chunks(chunks, issue) 
            coverage = self.coverage_evaluator.evaluate(
                chunks,
                issue,
            )
            logger.info(
                "HyDE generated %d queries",
                len(hyde_queries),
            )
        logger.info("After HyDE augmentation, total queries: %d", len(queries))
        logger.info(
            "Coverage after HyDE: %.3f",
            coverage,
        )
        verified = chunks
        document_hints = self._collect_document_hints(chunks)
        if coverage < coverage_threshold:
            additional = self._run_exa_fallback(queries[:5], n_results=n_results)
            logger.info(
                "EXA returned %d evidences",
                len(additional),
            )
            logger.info("EXA evidence list: %s", [e.to_log_dict() for e in additional])
            if additional:
                additional = LawDocumentFilter.filter_chunks(additional)
                chunks = chunks + additional
                coverage = self.coverage_evaluator.evaluate(chunks, issue)
                logger.info("Added %d EXA fallback evidences, new total %d", len(additional), len(chunks))
                logger.info("Merged evidence list after EXA: %s", [e.to_log_dict() for e in chunks])
                document_hints = self._collect_document_hints(chunks)
                verified = self.citation_verifier.verify(
                    chunks,
                    issue,
                    document_hints=document_hints,
                )
        reranked = self.reranker.rerank(verified or chunks, issue, top_k=top_k)
        logger.info("Reranker evidence list: %s", [e.to_log_dict() for e in reranked])
        verified = LawDocumentFilter.filter_chunks(
            self.citation_verifier.verify(
                reranked,
                issue,
                document_hints=document_hints,
            )
        )
        logger.info("Verified %d evidences after citation filter", len(verified))
        logger.info("Citation verifier final evidence list: %s", [e.to_log_dict() for e in verified])

        compact_chunks = self._compact_chunks(verified, issue)
        logger.info(
            "Compacted %d chunks (chars before=%d, after=%d)",
            len(compact_chunks),
            sum(len(self._get_chunk_text(e)) for e in verified),
            sum(len(self._get_chunk_text(c)) for c in compact_chunks),
        )

        return {
            "issue": issue,
            "queries": queries,
            "coverage": coverage,
            "chunks": compact_chunks,
        }
    def run_many(self, issues: List[str], **kwargs) -> List[Dict]:
        results = [None] * len(issues)
        with ThreadPoolExecutor(max_workers=min(len(issues), 5)) as pool:
            futures = {pool.submit(self.run, issue, **kwargs): i for i, issue in enumerate(issues)}
            for f in as_completed(futures):
                i = futures[f]
                try:
                    results[i] = f.result()
                except Exception:
                    logger.exception("Retrieval failed for issue: %s", issues[i])
                    results[i] = {"issue": issues[i], "queries": [], "coverage": 0.0, "chunks": []}
        return results
    @staticmethod
    def _get_chunk_text(chunk: Evidence) -> str:
        return str(chunk.text or "")

    @staticmethod
    def _get_chunk_metadata(chunk: Evidence) -> Dict:
        return chunk.metadata or {}

    @staticmethod
    def _get_chunk_value(chunk: Evidence, key: str, default=None):
        return getattr(chunk, key, default)

    @staticmethod
    def _collect_document_hints(chunks: List[Evidence]) -> List[str]:
        return list(dict.fromkeys(
            str(RetrievalPipelineService._get_chunk_metadata(chunk).get("law_code", "")).strip()
            for chunk in chunks
            if RetrievalPipelineService._get_chunk_metadata(chunk).get("law_code")
        ))

    # ------------------------------------------------------------------
    # Bước compact/trim: giảm token đưa vào context của agent mà không
    # làm mất nội dung pháp lý cốt lõi (điều luật, trích dẫn, nguồn).
    # ------------------------------------------------------------------
    def _compact_chunks(self, chunks: List[Evidence], issue: str) -> List[Evidence]:
        keywords = self._extract_keywords(issue)
        compacted = []
        for chunk in chunks:
            text = self._get_chunk_text(chunk)

            # 1) Nếu là raw text kiểu scrape (Exa) có cấu trúc Title/URL/.../Text/...
            #    thì chỉ giữ Title + Text, bỏ Image/Favicon/Extras/Summary/Highlights/
            #    Search Time/CostDollars vốn không có giá trị nội dung.
            parsed_title, parsed_text = self._parse_scraped_block(text)
            if parsed_text is not None:
                text = parsed_text

            # 2) Nếu văn bản vẫn quá dài (ví dụ nguyên văn cả bộ luật), chỉ giữ
            #    các đoạn có chứa từ khóa liên quan đến issue/điều luật, thay vì
            #    cắt cứng theo số ký tự (tránh mất đúng đoạn cần thiết).
            if len(text) > _LONG_TEXT_CHAR_THRESHOLD:
                text = self._extract_relevant_sections(text, keywords)

            metadata = self._get_chunk_metadata(chunk)
            compact_metadata = {
                k: v for k, v in metadata.items()
                if k in _LAW_METADATA_KEEP_KEYS and v not in (None, "")
            }
            if parsed_title:
                compact_metadata.setdefault("heading", parsed_title)

            compact_chunk = Evidence(
                text=text.strip(),
                source=self._get_chunk_value(chunk, "source_query") or chunk.source or "retrieval_pipeline",
                law_id=chunk.law_id,
                article=chunk.article,
                clause=chunk.clause,
                point=chunk.point,
                authority_type=chunk.authority_type,
                score=float(score) if (score := self._get_chunk_value(chunk, "score", self._get_chunk_value(chunk, "rerank_score", self._get_chunk_value(chunk, "retrieval_score")))) is not None else 0.0,
                retrieval_source=self._get_chunk_value(chunk, "retrieval_source") or chunk.retrieval_source,
                source_query=self._get_chunk_value(chunk, "source_query"),
                metadata=compact_metadata,
            )

            compacted.append(compact_chunk)

        return compacted

    @staticmethod
    def _parse_scraped_block(text: str):
        """Rút Title + Text từ 1 block scrape kiểu:
        Title: ...\nURL: ...\n...\nText: ...\nSummary: ...\nHighlights: [...]\nSearch Time: ...
        Trả về (title, text_content) hoặc (None, None) nếu không khớp định dạng.
        """
        if "Text:" not in text or "Title:" not in text:
            return None, None

        title_match = re.search(r"Title:\s*(.+)", text)
        title = title_match.group(1).strip() if title_match else None

        # Lấy phần sau "Text:" và trước "Summary:"/"Highlights:"/"Search Time:" (cái nào đến trước)
        text_match = re.search(r"\nText:\s*(.*?)(?:\nSummary:|\nHighlights:|\nSearch Time:|$)", text, re.S)
        if not text_match:
            return title, None

        body = text_match.group(1).strip()
        result = f"{title}\n\n{body}" if title else body
        return title, result

    @staticmethod
    def _extract_keywords(issue: str) -> List[str]:
        # Từ khóa đơn giản: các từ có nghĩa dài hơn 3 ký tự trong issue,
        # cộng thêm các từ khóa pháp lý phổ biến để nhận diện đoạn liên quan.
        base = re.findall(r"[A-Za-zÀ-ỹ]{4,}", issue)
        legal_hints = ["Điều", "Khoản", "hợp đồng", "đặt cọc", "sở hữu", "hiệu lực"]
        return list(dict.fromkeys(base + legal_hints))

    @staticmethod
    def _extract_relevant_sections(text: str, keywords: List[str]) -> str:
        """Với văn bản quá dài, chỉ giữ các đoạn (paragraph) chứa từ khóa liên
        quan, kèm một cửa sổ ký tự xung quanh mỗi lần khớp, thay vì cắt cứng
        theo độ dài -> tránh mất nội dung liên quan nằm ở cuối văn bản.
        """
        if not keywords:
            return text[:_LONG_TEXT_CHAR_THRESHOLD] + "\n[... nội dung đã rút gọn ...]"

        lower_text = text.lower()
        matches = []
        for kw in keywords:
            for m in re.finditer(re.escape(kw.lower()), lower_text):
                start = max(0, m.start() - _KEYWORD_WINDOW_CHARS // 2)
                end = min(len(text), m.end() + _KEYWORD_WINDOW_CHARS // 2)
                matches.append((start, end))

        if not matches:
            return text[:_LONG_TEXT_CHAR_THRESHOLD] + "\n[... nội dung đã rút gọn, không tìm thấy đoạn khớp từ khóa ...]"

        # Gộp các khoảng chồng lấn lại để tránh trùng lặp nội dung
        matches.sort()
        merged = [matches[0]]
        for start, end in matches[1:]:
            last_start, last_end = merged[-1]
            if start <= last_end:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))

        sections = [text[s:e].strip() for s, e in merged]
        return "\n\n[...]\n\n".join(sections)

    @staticmethod
    def _normalize_exa_results(res) -> List[Evidence]:
        if isinstance(res, list):
            normalized = []
            for item in res:
                if isinstance(item, dict):
                    normalized.append(Evidence.from_dict(item, default_source="exa"))
                else:
                    normalized.append(Evidence(text=str(item), source="exa", metadata={}, retrieval_source="exa"))
            return normalized

        if isinstance(res, dict):
            results = res.get("results") or res.get("data") or []
            if isinstance(results, list):
                return [
                    Evidence.from_dict(
                        {
                            "id": item.get("id") if isinstance(item, dict) else None,
                            "text": item.get("text") or item.get("title") or str(item) if isinstance(item, dict) else str(item),
                            "metadata": item.get("metadata", {}) if isinstance(item, dict) else {},
                        },
                        default_source="exa",
                    )
                    for item in results
                ]
            return [Evidence(text=str(res), source="exa", metadata={}, retrieval_source="exa")]

        return [Evidence(text=str(res), source="exa", metadata={}, retrieval_source="exa")]

    def _run_exa_fallback(self, queries: List[str], n_results: int = 5) -> List[Evidence]:
        try:
            from crewai_tools import EXASearchTool, ScrapeWebsiteTool
        except Exception:
            logger.exception("EXA fallback tools not available")
            return []

        added = []
        if EXASearchTool is not None:
            try:
                exa = EXASearchTool()
                for q in queries:
                    law_query = LawDocumentFilter.law_focused_query(q)
                    try:
                        res = exa._run(search_query=law_query)
                        items = self._normalize_exa_results(res)
                        for item in items:
                            item.source_query = q
                            item.retrieval_source = "exa"
                            added.append(item)
                    except Exception:
                        continue
            except Exception:
                logger.exception("EXA fallback initialization failed")

        if ScrapeWebsiteTool is not None and added:
            try:
                crawler = ScrapeWebsiteTool()
                crawled = []
                for item in added:
                    link = (item.metadata or {}).get("url") or item.text
                    try:
                        page = crawler._run(str(link))
                        crawled.append(Evidence(
                            text=str(page),
                            source=str(link or "exa_crawl"),
                            metadata={"source": link},
                            source_query=item.source_query,
                            retrieval_source="exa_crawl",
                        ))
                    except Exception:
                        continue
                if crawled:
                    return LawDocumentFilter.filter_chunks(crawled)
            except Exception:
                logger.exception("Website crawling failed during EXA fallback")

        return LawDocumentFilter.filter_chunks(added)