from typing import Optional
import re


class NormalizeService:
    law_aliases = {
        "BLDS": "Bộ luật Dân sự",
        "Bộ luật dân sự": "Bộ luật Dân sự",
        "Civil Code": "Bộ luật Dân sự",
    }

    number_map = {
        "một": 1,
        "hai": 2,
        "ba": 3,
        "bốn": 4,
        "năm": 5,
        "sáu": 6,
        "bảy": 7,
        "tám": 8,
        "chín": 9,
        "mười": 10,
    }

    def normalize_law_name(self, text: str) -> str:
        if not text:
            return text
        normalized = text.strip()
        return self.law_aliases.get(normalized, normalized)

    def normalize_article(self, text: str) -> str:
        if not text:
            return text
        normalized = text
        normalized = re.sub(r"Điều\s+([a-zA-Z\s]+)", self._normalize_article_match, normalized, flags=re.IGNORECASE)
        return normalized

    def _normalize_article_match(self, match: re.Match) -> str:
        word = match.group(1).strip().lower()
        words = word.split()
        if words and all(w in self.number_map for w in words):
            value = sum(self.number_map[w] for w in words)
            return f"Điều {value}"
        return match.group(0)
