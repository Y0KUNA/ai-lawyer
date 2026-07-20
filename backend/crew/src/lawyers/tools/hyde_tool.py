from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import List, Type


class HyDEInput(BaseModel):
    issue: str
    n: int = 3


class HyDETool(BaseTool):
    """Lightweight HyDE: generate hypothetical answers to expand retrieval."""
    name : str  = "HyDETool"
    description : str  = "Generate hypothetical document snippets (HyDE) from an issue to improve recall."
    args_schema: Type[BaseModel]  = HyDEInput

    def _run(self, issue: str, n: int = 3) -> List[str]:
        if not issue:
            return []
        out = []
        for i in range(max(1, n)):
            out.append(f"Giả định về {issue} - gợi ý tìm kiếm {i+1}")
        return out
