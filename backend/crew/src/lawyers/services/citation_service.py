import re
from typing import List, Dict


class CitationVerifierService:
    citation_patterns = [
        r"Điều\s+\d+",
        r"Khoản\s+\d+",
        r"Điểm\s+[a-zA-Z0-9]+",
        r"Chương\s+\w+",
        r"Mục\s+\w+",
    ]

    def verify(self, chunks: List[Dict], citations: List[str]) -> List[Dict]:
        if not chunks:
            return []
        if not citations:
            return chunks

        patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.citation_patterns]
        verified = []
        for chunk in chunks:
            text = chunk.get("text", "") or ""
            metadata = chunk.get("metadata", {}) or {}
            metadata_text = " ".join(str(v) for v in metadata.values() if v)
            if self._matches_citation(text, metadata_text, citations, patterns):
                verified.append(chunk)
        return verified

    def _matches_citation(self, text: str, metadata_text: str, citations: List[str], patterns: List[re.Pattern]) -> bool:
        for citation in citations:
            if not citation:
                continue
            if citation in text or citation in metadata_text:
                return True
            if any(pattern.search(citation) for pattern in patterns) and any(pattern.search(text) or pattern.search(metadata_text) for pattern in patterns):
                return True
        return False
