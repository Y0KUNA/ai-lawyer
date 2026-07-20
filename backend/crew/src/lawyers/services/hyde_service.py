from typing import List
from crewai import LLM


class HyDEService:
    def __init__(self, llm: LLM):
        self.llm = llm

    def generate(self, issue: str, n: int = 3) -> List[str]:
        if not issue:
            return []

        prompt = (
            f"Bạn là một luật sư. Dựa trên yêu cầu pháp lý sau, "
            f"hãy viết {n} đoạn văn bản giả định ngắn gọn mô tả nội dung "
            f"văn bản pháp lý hoặc án lệ có thể áp dụng cho vấn đề. "
            f"Mỗi đoạn nên là một cụm từ tìm kiếm hiệu quả, không dài hơn 50 từ. "
            f"Tập trung vào luật Việt Nam và dùng tiếng Việt.\n\n"
            f"Yêu cầu: {issue}"
        )

        response = self.llm.call(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        if not response:
            return []

        text = str(response)

        items = [
            line.strip()
            for line in text.split("\n")
            if line.strip()
        ]

        return items[:n]