import logging
from typing import Dict, List
from crewai import LLM
from .query_expansion_service import QueryExpansionService
from .hyde_service import HyDEService
from .rag_service import RAGService
from .reranker_service import RerankerService
from .citation_service import CitationVerifierService
from .coverage_service import CoverageEvaluator
from .normalize_service import NormalizeService


logger = logging.getLogger(__name__)


class RetrievalPipelineService:
    def __init__(self, llm: LLM):
        self.query_expander = QueryExpansionService()
        self.hyde = HyDEService(llm)
        self.rag = RAGService()
        self.reranker = RerankerService()
        self.citation_verifier = CitationVerifierService()
        self.coverage_evaluator = CoverageEvaluator()
        self.normalizer = NormalizeService()

    def run(self, issue: str, n_queries: int = 3, n_results: int = 8, top_k: int = 8, coverage_threshold: float = 0.5) -> List[Dict]:
        issue = self.normalizer.normalize_law_name(issue)
        issue = self.normalizer.normalize_article(issue)
        queries = self.query_expander.expand(issue, n_queries)
        logger.info("Query expansion generated %d queries", len(queries))
        
        chunks = self.rag.search(queries, n_results=n_results)
        logger.info("RAG returned %d chunks", len(chunks))

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
        citations = [str(chunk.get("metadata", {}).get("so_hieu", "")) for chunk in chunks if chunk.get("metadata", {}).get("so_hieu")]
        if coverage < coverage_threshold:
            additional = self._run_exa_fallback(queries[:5], n_results=n_results)
            logger.info(
                "EXA returned %d chunks",
                len(additional),
            )
            if additional:
                chunks = chunks + additional
                coverage = self.coverage_evaluator.evaluate(chunks, issue)
                verified = self.citation_verifier.verify(chunks, citations)
                logger.info("Added %d EXA fallback chunks, new total %d", len(additional), len(chunks))
                citations = [str(chunk.get("metadata", {}).get("so_hieu", "")) for chunk in chunks if chunk.get("metadata", {}).get("so_hieu")]
        reranked = self.reranker.rerank(verified or chunks, issue, top_k=top_k)
        verified = self.citation_verifier.verify(reranked, citations)
        logger.info("Verified %d chunks after citation filter", len(verified))
        
        logger.info("Reranked result count: %d", len(reranked))
        return {
            "issue": issue,
            "queries": queries,
            "coverage": coverage,
            "chunks": verified,
        }

    def _run_exa_fallback(self, queries: List[str], n_results: int = 5) -> List[Dict]:
        try:
            from crewai_tools import EXASearchTool, ScrapeWebsiteTool
        except Exception:
            return []

        added = []
        if EXASearchTool is not None:
            try:
                exa = EXASearchTool()
                for q in queries:
                    try:
                        res = exa._run(q)
                        if isinstance(res, list):
                            for item in res:
                                added.append({
                                    "id": item.get("id"),
                                    "text": item.get("text", str(item)),
                                    "metadata": item.get("metadata", {}),
                                    "source_query": q,
                                    "retrieval_source": "exa",
                                })
                        else:
                            added.append({
                                "id": None,
                                "text": str(res),
                                "metadata": {},
                                "source_query": q,
                                "retrieval_source": "exa",
                            })
                    except Exception:
                        continue
            except Exception:
                logger.exception("EXA fallback initialization failed")

        if ScrapeWebsiteTool is not None and added:
            try:
                crawler = ScrapeWebsiteTool()
                crawled = []
                for item in added:
                    link = item.get("metadata", {}).get("url") or item.get("text")
                    try:
                        page = crawler._run(str(link))
                        crawled.append({
                            "id": None,
                            "text": str(page),
                            "metadata": {"source": link},
                            "source_query": item.get("source_query"),
                            "retrieval_source": "exa_crawl",
                        })
                    except Exception:
                        continue
                if crawled:
                    return crawled
            except Exception:
                logger.exception("Website crawling failed during EXA fallback")

        return added
