from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
import time
from dotenv import load_dotenv
import os
import json
print("=================================")
print("RUNNING MY MAIN.PY")
print("=================================")

# load .env if present (should be valid KEY=VALUE format)
load_dotenv()


app = FastAPI()

# Allow requests from the frontend dev server
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
    # Try to load a JSON system prompt from env, otherwise use a simple default
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

    # Expect req.messages to be a list of message objects like {role, content}
    messages = [SYSTEM_PROMPT, *req.messages]

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
