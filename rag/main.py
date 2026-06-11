from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import chromadb
import requests
import json
from sentence_transformers import SentenceTransformer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model  = SentenceTransformer("BAAI/bge-m3")
client = chromadb.PersistentClient(path="./chroma_db")
coll   = client.get_collection("luat_vn")


class ChatRequest(BaseModel):
    question: str


def retrieve_context(question: str, n_results: int = 5):
    """Truy xuất chunks liên quan, ưu tiên luật còn hiệu lực."""
    q_vec = model.encode(question).tolist()

    # Thử filter luật còn hiệu lực trước
    results = coll.query(
        query_embeddings=[q_vec],
        n_results=n_results,
        where={"status": {"$ne": "het_hieu_luc"}},  # bỏ luật hết hạn
    )

    # Nếu không đủ kết quả thì fallback không filter
    if len(results["documents"][0]) < 2:
        results = coll.query(query_embeddings=[q_vec], n_results=n_results)

    docs   = results["documents"][0]
    metas  = results["metadatas"][0]
    return docs, metas


def build_prompt(question: str, docs: list, metas: list) -> str:
    context_parts = []
    for doc, meta in zip(docs, metas):
        header = (
            f"[{meta['loai_van_ban']} {meta['so_hieu']} — {meta['heading']}]"
            f" (Hiệu lực: {meta['ngay_hieu_luc']})"
        )
        context_parts.append(f"{header}\n{doc}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""Bạn là chuyên gia tư vấn pháp luật Việt Nam, trả lời chính xác và dễ hiểu.

NGUYÊN TẮC:
- Chỉ dựa vào các quy định pháp luật được cung cấp bên dưới.
- Trích dẫn rõ số hiệu văn bản và điều khoản khi trả lời.
- Nếu thông tin không đủ để trả lời, hãy nói rõ và gợi ý người dùng tham khảo thêm.
- Không bịa đặt quy định pháp luật.

=== CÁC QUY ĐỊNH LIÊN QUAN ===
{context}

=== CÂU HỎI ===
{question}

=== TRẢ LỜI ==="""


@app.post("/chat")
def chat(req: ChatRequest):
    docs, metas = retrieve_context(req.question)
    prompt = build_prompt(req.question, docs, metas)

    # Gọi Ollama với streaming
    def generate():
        with requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3", "prompt": prompt, "stream": True},
            stream=True,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if token := chunk.get("response"):
                        yield token
                    if chunk.get("done"):
                        # Gửi nguồn trích dẫn sau khi stream xong
                        sources = [
                            {
                                "so_hieu":  m["so_hieu"],
                                "ten_luat": m["ten_luat"],
                                "heading":  m["heading"],
                            }
                            for m in metas
                        ]
                        # Gửi dưới dạng JSON đặc biệt để FE nhận ra
                        yield f"\n\n__SOURCES__{json.dumps(sources, ensure_ascii=False)}"

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")