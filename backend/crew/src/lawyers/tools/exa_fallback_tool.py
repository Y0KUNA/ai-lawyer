from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import List, Optional, Type
import logging

try:
    from crewai_tools import EXASearchTool, ScrapeWebsiteTool
except Exception:
    EXASearchTool = None
    ScrapeWebsiteTool = None


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
                    try:
                        res = exa._run(q)
                        # If EXA returns structured items, try to convert
                        if isinstance(res, list):
                            for item in res:
                                added.append({"id": item.get("id" , None), "text": item.get("text", str(item)), "metadata": item.get("metadata", {}), "source_query": q})
                        else:
                            added.append({"id": None, "text": str(res), "metadata": {}, "source_query": q})
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
                    return crawled
            except Exception:
                logging.exception("Website crawling failed during EXA fallback")
        return added
