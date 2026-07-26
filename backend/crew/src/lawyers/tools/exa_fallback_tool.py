from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import List, Optional, Type
import logging

try:
    from crewai_tools import EXASearchTool, ScrapeWebsiteTool
except Exception:
    EXASearchTool = None
    ScrapeWebsiteTool = None

from ..services.law_document_filter import LawDocumentFilter


class EXAFallbackInput(BaseModel):
    queries: List[str]
    existing_chunks: Optional[List[dict]] = None
    n_results: int = 5
    threshold: float = 0.5


class EXAFallbackTool(BaseTool):
    """Fallback that calls EXA and crawling when coverage is low. Agent should not call EXA directly."""
    name : str  = "EXAFallbackTool"
    description : str  = "If coverage is below threshold, internally call EXA and crawl results, then return new chunks."
    args_schema: Type[BaseModel]  = EXAFallbackInput

    @staticmethod
    def _normalize_exa_results(res) -> List[dict]:
        if isinstance(res, list):
            return [item if isinstance(item, dict) else {"text": str(item)} for item in res]

        if isinstance(res, dict):
            results = res.get("results") or res.get("data") or []
            if isinstance(results, list):
                return [
                    {
                        "id": item.get("id") if isinstance(item, dict) else None,
                        "text": item.get("text") or item.get("title") or str(item) if isinstance(item, dict) else str(item),
                        "metadata": item.get("metadata", {}) if isinstance(item, dict) else {},
                    }
                    for item in results
                ]
            return [{"text": str(res)}]

        return [{"text": str(res)}]

    def _run(self, queries: List[str], existing_chunks: Optional[List[dict]] = None, n_results: int = 5, threshold: float = 0.5) -> List[dict]:
        existing = existing_chunks or []
        denom = max(1, len(queries) * max(1, n_results))
        coverage = len(existing) / denom
        if coverage >= threshold:
            return []

        added = []
        # Try to call EXA search tool if available
        if EXASearchTool is not None:
            try:
                exa = EXASearchTool()
                for q in queries:
                    law_query = LawDocumentFilter.law_focused_query(q)
                    try:
                        res = exa._run(search_query=law_query)
                        items = self._normalize_exa_results(res)
                        for item in items:
                            added.append({
                                "id": item.get("id"),
                                "text": item.get("text", str(item)),
                                "metadata": item.get("metadata", {}),
                                "source_query": q,
                            })
                    except Exception:
                        continue
            except Exception:
                logging.exception("EXA call failed")

        # If ScrapeWebsiteTool available and EXA returned links, try crawling (best effort)
        if ScrapeWebsiteTool is not None and added:
            try:
                crawler = ScrapeWebsiteTool()
                crawled = []
                for a in added:
                    link = a.get("metadata", {}).get("url") or a.get("text")
                    try:
                        page = crawler._run(str(link))
                        crawled.append({"id": None, "text": str(page), "metadata": {"source": link}, "source_query": a.get("source_query")})
                    except Exception:
                        continue
                if crawled:
                    return LawDocumentFilter.filter_chunks(crawled)
            except Exception:
                logging.exception("Website crawling failed during EXA fallback")
        return LawDocumentFilter.filter_chunks(added)
