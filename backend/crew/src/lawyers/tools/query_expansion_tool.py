from crewai.tools import BaseTool
from pydantic import BaseModel
from typing import List, Type


class QueryExpansionInput(BaseModel):
    issue: str
    n_queries: int = 3


class QueryExpansionTool(BaseTool):
    """Simple query expansion: produces a small set of search queries from an issue string."""
    name : str  = "QueryExpansionTool"
    description : str = "Expand an issue string into multiple search queries."
    args_schema: Type[BaseModel]  = QueryExpansionInput

    def _run(self, issue: str, n_queries: int = 3) -> List[str]:
        if not issue:
            return []
        base = issue.strip()
        queries = [base]
        if n_queries >= 2:
            queries.append(f"{base} văn bản pháp luật")
        if n_queries >= 3:
            queries.append(f"{base} án lệ")
        # pad with slight variations if requested
        i = 3
        while len(queries) < n_queries:
            queries.append(f"{base} {i}")
            i += 1
        return queries[:n_queries]
