from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import List, Optional, Type

from ..services.citation_service import CitationVerifierService


class CitationVerifierInput(BaseModel):
    chunks: List[dict]
    issue: str
    document_hints: Optional[List[str]] = None


class CitationVerifierTool(BaseTool):
    """Verifies whether retrieved chunks match article-level citations from the issue."""
    name: str = "CitationVerifierTool"
    description: str = (
        "Checks that retrieved chunks match specific Điều/Khoản/Điểm citations "
        "from the issue, not just the parent law document. Pure Python checks, no LLM."
    )
    args_schema: Type[BaseModel] = CitationVerifierInput

    def _run(
        self,
        chunks: List[dict],
        issue: str,
        document_hints: Optional[List[str]] = None,
    ) -> List[dict]:
        return CitationVerifierService().verify(chunks, issue, document_hints=document_hints)
