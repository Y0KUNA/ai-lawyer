import os
from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr
from typing import List, Type
from crewai import LLM
from ..services.retrieval_pipeline_service import RetrievalPipelineService


class RetrievalPipelineInput(BaseModel):
    issue: str
    n_queries: int = 3
    n_results: int = 8
    top_k: int = 8
    coverage_threshold: float = 0.5


class RetrievalPipelineTool(BaseTool):
    """One-stop retrieval pipeline that the Research Agent uses.

    It internally composes query expansion, HyDE, RAG, reranking, citation verification,
    and low-coverage EXA fallback through a single service layer.
    """
    name :str = "RetrievalPipelineTool"
    description : str = "Run the full retrieval pipeline and return authoritative chunks."
    args_schema: Type[BaseModel]  = RetrievalPipelineInput
    _service: RetrievalPipelineService = PrivateAttr()
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        local_llm = LLM(
            model="ollama/gemma4:e2b",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
        self._service = RetrievalPipelineService(local_llm)

    def _run(self, issue: str, n_queries: int = 3, n_results: int = 8, top_k: int = 8, coverage_threshold: float = 0.5) -> List[dict]:
        return self._service.run(issue, n_queries=n_queries, n_results=n_results, top_k=top_k, coverage_threshold=coverage_threshold)
