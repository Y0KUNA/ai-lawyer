from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import time
from dotenv import load_dotenv
import os
import json
from pathlib import Path
# ── RAG: thêm 2 import này ──────────────────────────────────────────
import chromadb
from sentence_transformers import SentenceTransformer
# ────────────────────────────────────────────────────────────────────

print("=================================")
print("RUNNING MY MAIN.PY")
print("=================================")

load_dotenv()

# ── RAG: khởi tạo 1 lần khi app start ───────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
CHROMA_DIR = Path(os.getenv("CHROMA_PATH", BASE_DIR.parent / "rag" / "chroma_db"))
# Resolve relative path so "../rag/chroma_db" hoạt động đúng
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = (BASE_DIR / CHROMA_DIR).resolve()

_embed_model = SentenceTransformer(os.getenv("EMBED_MODEL", "BAAI/bge-m3"))
_chroma      = chromadb.PersistentClient(path=str(CHROMA_DIR))
_collection  = _chroma.get_collection("luat_vn")
print("CHROMA_DIR:", CHROMA_DIR)  # thêm dòng này trước get_collection
_collection = _chroma.get_collection("luat_vn")
def retrieve_law_context(messages: list, n_results: int = 5) -> str:
    """
    Lấy câu hỏi cuối cùng của user, embed rồi tìm chunks luật liên quan.
    Trả về chuỗi context để nhét vào system prompt.
    """
    # Lấy nội dung tin nhắn cuối cùng của user
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        return ""

    q_vec = _embed_model.encode(user_query).tolist()

    results = _collection.query(
        query_embeddings=[q_vec],
        n_results=n_results,
        where={"status": {"$ne": "het_hieu_luc"}},  # bỏ luật hết hiệu lực
    )

    # Fallback nếu filter trả về quá ít kết quả
    if len(results["documents"][0]) < 2:
        results = _collection.query(query_embeddings=[q_vec], n_results=n_results)

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        header = (
            f"[{meta.get('loai_van_ban','')} {meta.get('so_hieu','')} "
            f"— {meta.get('heading','')}]"
        )
        context_parts.append(f"{header}\n{doc}")

    return "\n\n---\n\n".join(context_parts)
# ────────────────────────────────────────────────────────────────────


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=".*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    messages: list


@app.post("/chat")
def chat(req: ChatRequest):
    system_prompt_raw = os.getenv("SYSTEM_PROMPT")
    try:
        if system_prompt_raw:
            SYSTEM_PROMPT = json.loads(system_prompt_raw)
        else:
            raise ValueError("empty")
    except Exception:
        SYSTEM_PROMPT = {
            "role": "system",
            "content": (
                "Bạn là luật sư.\n\n"
                "Nhiệm vụ:\n"
                "- Tóm tắt vụ án\n"
                "- Giải thích các điều luật\n"
                "- Phân tích chi tiết các tình huống pháp lý\n"
                "- Trả lời ngắn gọn và chính xác"
            ),
        }

    # ── RAG: truy xuất context rồi gắn vào system prompt ────────────
    law_context = retrieve_law_context(req.messages)
    if law_context:
        rag_note = (
            "\n\n=== CÁC QUY ĐỊNH PHÁP LUẬT LIÊN QUAN ===\n"
            f"{law_context}\n"
            "==========================================\n"
            "Hãy dựa vào các quy định trên để trả lời. "
            "Trích dẫn số hiệu văn bản và điều khoản cụ thể khi có thể. "
            "Nếu thông tin không đủ, hãy nói rõ thay vì bịa đặt."
        )
        # Tạo bản sao để không mutate object gốc
        system_with_rag = {
            "role": "system",
            "content": SYSTEM_PROMPT["content"] + rag_note,
        }
    else:
        system_with_rag = SYSTEM_PROMPT
    # ────────────────────────────────────────────────────────────────

    messages = [system_with_rag, *req.messages]  # ← thay SYSTEM_PROMPT bằng system_with_rag

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "gemma4:e2b",
                "messages": messages,
                "stream": True,
            },
            stream=True,
            timeout=(10, None),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"Cannot connect to Ollama: {exc}") from exc

    def stream_answer():
        start = time.time()
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    print("INVALID OLLAMA CHUNK:", line)
                    continue

                content = chunk.get("message", {}).get("content")
                if content:
                    yield content

                if chunk.get("done"):
                    break
        finally:
            response.close()
            print("OLLAMA TIME:", time.time() - start)

    return StreamingResponse(stream_answer(), media_type="text/plain; charset=utf-8")


@app.get("/ping")
def ping():
    return {"status": "ok"}


print("===== ROUTES =====")
for route in app.routes:
    print(route.path, route.methods)