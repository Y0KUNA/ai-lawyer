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
from crew.src.lawyers.services.chroma_service import ChromaService
# ── RAG: thêm 2 import này ──────────────────────────────────────────
import chromadb
from sentence_transformers import SentenceTransformer
# ────────────────────────────────────────────────────────────────────
# ── CREW: thêm import cho AI Lawyer Crew ────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent / "crew" / "src"))
from crew.src.lawyers.crew import AILawyerCrew
# ────────────────────────────────────────────────────────────────────
import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    handlers=[
        logging.FileHandler("logs/crew.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
print("=================================")
print("RUNNING MY MAIN.PY")
print("=================================")

load_dotenv()

# ── RAG: khởi tạo 1 lần khi app start ───────────────────────────────
BASE_DIR   = Path(__file__).parent.resolve()
CHROMA_DIR = Path(os.getenv("CHROMA_PATH", BASE_DIR.parent / "chroma_db"))
# Resolve relative path so "../chroma_db" hoạt động đúng
if not CHROMA_DIR.is_absolute():
    CHROMA_DIR = (BASE_DIR / CHROMA_DIR).resolve()

_embed_model = None
_chroma = None
_collection = None

try:
    _embed_model = SentenceTransformer(os.getenv("EMBED_MODEL", "BAAI/bge-m3"))
    ChromaService.initialize()
    _collection = ChromaService.get_collection()
    print("CHROMA_DIR:", CHROMA_DIR)
    print("[OK] ChromaDB collection 'luat_vn' loaded successfully")
except Exception as e:
    print(f"[WARNING] Warning: Could not load ChromaDB collection: {e}")
    print("  RAG will be disabled but chat will still work")
def retrieve_law_context(messages: list, n_results: int = 5) -> str:
    """
    Lấy câu hỏi cuối cùng của user, embed rồi tìm chunks luật liên quan.
    Trả về chuỗi context để nhét vào system prompt.
    """
    # Kiểm tra nếu ChromaDB không khả dụng
    if _collection is None or _embed_model is None:
        return ""
    
    # Lấy nội dung tin nhắn cuối cùng của user
    user_query = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            user_query = msg.get("content", "")
            break

    if not user_query:
        return ""

    try:
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
    except Exception as e:
        print(f"[WARNING] Error retrieving law context: {e}")
        return ""
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


class CaseAnalysisRequest(BaseModel):
    """Request model for AI Lawyer Crew analysis"""
    case_description: str


class CaseAnalysisResponse(BaseModel):
    """Response model for AI Lawyer Crew analysis"""
    analysis: str
    status: str = "completed"


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
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        response = requests.post(
            f"{ollama_host}/api/chat",
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


@app.post("/analyze")
async def analyze_case(req: CaseAnalysisRequest):
    try:
        print("CASE:")
        print(req.case_description)
        print("\n" + "="*50)
        print("ANALYZING CASE WITH AI LAWYER CREW")
        print("="*50)
        
        crew = AILawyerCrew()
        result = await crew.crew().kickoff_async(
            inputs={"case_description": req.case_description}
        )
        
        print("[OK] Analysis completed")
        print("="*50 + "\n")
        
        return CaseAnalysisResponse(
            analysis=str(result),
            status="completed"
        )
    except Exception as e:
        print(f"[ERROR] Error analyzing case: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing case: {str(e)}"
        )


print("===== ROUTES =====")
for route in app.routes:
    print(route.path, route.methods)