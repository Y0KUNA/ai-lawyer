from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import List, Type


class CitationVerifierInput(BaseModel):
    chunks: List[dict]
    citations: List[str]


class CitationVerifierTool(BaseTool):
    """Verifies whether citation strings exist inside chunks using Python-only checks."""
    name : str  = "CitationVerifierTool"
    description : str  = "Checks that citation text appears in retrieved chunks. Pure Python checks, no LLM."
    args_schema: Type[BaseModel]  = CitationVerifierInput

    def _run(self, chunks: List[dict], citations: List[str]) -> List[dict]:
        if not chunks or not citations:
            return []
        kept = []
        for c in chunks:
            text = c.get("text", "") or ""
            meta = c.get("metadata", {}) or {}
            keep = False
            for cit in citations:
                if not cit:
                    continue
                if cit in text:
                    keep = True
                    break
                # also check metadata fields as joined string
                if any(cit in str(v) for v in meta.values()):
                    keep = True
                    break
            if keep:
                kept.append(c)
        return kept
