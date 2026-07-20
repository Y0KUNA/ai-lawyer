from typing import List


class QueryExpansionService:
    def expand(self, issue: str, n_queries: int = 3) -> List[str]:
        if not issue:
            return []
        base = issue.strip()
        queries = [base]
        if n_queries >= 2:
            queries.append(f"{base} văn bản pháp luật")
        if n_queries >= 3:
            queries.append(f"{base} án lệ")
        if n_queries >= 4:
            queries.append(f"{base} điều khoản")
        if n_queries >= 5:
            queries.append(f"{base} quy định pháp luật")

        i = 5
        while len(queries) < n_queries:
            queries.append(f"{base} {i}")
            i += 1

        return queries[:n_queries]
