# AI Lawyer

AI Lawyer là một trợ lý pháp lý (tiếng Việt) chạy local, gồm frontend React,
backend FastAPI, và một hệ **multi-agent CrewAI** chuyên phân tích vụ việc
pháp lý theo phương pháp IRAC (Issue – Rule – Application – Conclusion), có
retrieval tăng cường (RAG) trên kho văn bản luật Việt Nam lưu trong ChromaDB.

Model ngôn ngữ chạy qua **Ollama** (local, không gọi API ngoài).

## Kiến trúc tổng quan

```
frontend (Vite + React)  →  backend (FastAPI)  →  Ollama (LLM local)
                                    │
                                    ├─ /chat     → chat thường + RAG context (tìm nhanh, 1 lượt)
                                    └─ /analyze  → AI Lawyer Crew (multi-agent, phân tích IRAC đầy đủ)
                                                        │
                                                        ├─ ChromaDB (backend/chroma_db, chroma_db/)
                                                        └─ SentenceTransformer (BAAI/bge-m3) để embed
```

### AI Lawyer Crew (`backend/crew/`)

Hệ agent dùng CrewAI, định nghĩa tại `backend/crew/src/lawyers/`:

**Agents** (`config/agents.yaml`, `crew.py`):
- `legal_request_analyzer` – phân tích yêu cầu/vụ việc đầu vào
- `legal_issue_identifier` – tách vụ việc thành danh sách các vấn đề pháp lý
- `legal_researcher` – tra cứu căn cứ pháp lý cho từng vấn đề (dùng tool retrieval)
- `irac_legal_reasoning_specialist` – lập luận theo IRAC
- `legal_verification_and_synthesis_specialist` – kiểm chứng trích dẫn và tổng hợp câu trả lời cuối

**Tasks** (`config/tasks.yaml`, chạy tuần tự): `analyze_legal_request` →
`identify_legal_issues` → `research_relevant_laws` →
`apply_irac_legal_reasoning` → `verify_and_synthesize_legal_response`

**Tools** (`src/lawyers/tools/`):
- `retrieval_pipeline_tool.py` – pipeline tìm kiếm chính (mở rộng câu hỏi → tìm ChromaDB → rerank)
- `query_expansion_tool.py` – sinh nhiều biến thể truy vấn từ 1 vấn đề pháp lý
- `reranker_tool.py` – xếp hạng lại kết quả truy xuất
- `hyde_tool.py` – Hypothetical Document Embeddings để cải thiện truy hồi
- `citation_verifier_tool.py` – xác minh trích dẫn điều luật khớp với nguồn
- `exa_fallback_tool.py` – tìm kiếm web dự phòng qua Exa khi RAG nội bộ không đủ
- `rag_query_tool.py`, `custom_tool.py`

**Services** (`src/lawyers/services/`): `retrieval_pipeline_service.py`,
`query_expansion_service.py`, `rag_service.py`, `chroma_service.py`,
`normalize_service.py`, v.v. — chứa logic nghiệp vụ thực sự, các tool ở trên
chỉ là lớp bọc (wrapper) để agent gọi được.

## Cấu trúc thư mục

```
backend/
  main.py                 FastAPI app: /chat, /analyze, /ping
  requirements.txt
  Dockerfile
  chroma_db/               (dữ liệu ChromaDB dùng bởi backend khi chạy Docker)
  crew/
    src/lawyers/
      agents.yaml, tasks.yaml   (config/)
      crew.py               định nghĩa AILawyerCrew (agents + tasks)
      tools/                 tool wrapper cho agent
      services/               logic nghiệp vụ (RAG, rerank, HyDE, normalize...)
  fine-tune/                script/tài nguyên fine-tune (nếu có)
  evaluate_retrieval_pipeline.py   script đánh giá pipeline truy hồi

frontend/
  src/
    App.jsx, main.jsx
    components/            ChatWindow, ChatInput, MessageBubble, SideBar, LoadingIndicator
  Dockerfile, nginx.conf

rag/
  index_documen.py          script index văn bản luật vào ChromaDB
  main.py

data/
  chunk.py, chunk_new.py, chunk_builder.py   script tiền xử lý/chunk văn bản luật
  law-clean/, rag_chunks/    dữ liệu luật đã làm sạch / đã chunk

docker-compose.yml         chạy ollama + backend + frontend cùng lúc
```

## Yêu cầu môi trường

- Node.js 18+
- Python 3.11+
- Ollama đã cài và đang chạy, với model đã pull (mặc định `gemma4:e2b` /
  `gemma4:e4b` — kiểm tra lại tên model đang cấu hình trong `backend/main.py`
  và `crew.py`)
- (Tuỳ chọn) API key của [Exa](https://exa.ai) nếu muốn dùng `exa_fallback_tool`

## Biến môi trường

Tạo file `backend/.env`:

```env
SYSTEM_PROMPT={"role":"system","content":"You are a legal assistant."}
CHROMA_PATH=../chroma_db
EMBED_MODEL=BAAI/bge-m3
OLLAMA_HOST=http://localhost:11434
EXA_API_KEY=your_exa_api_key   # nếu dùng exa_fallback_tool
```

Nếu không đặt `SYSTEM_PROMPT`, backend dùng prompt luật sư tiếng Việt mặc định
được hard-code sẵn trong `main.py`.

## Cài đặt & chạy (thủ công)

### 1. Ollama

```bash
ollama pull gemma4:e2b
ollama serve
```

### 2. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Lần chạy đầu tiên sẽ tải model embedding `BAAI/bge-m3` về máy (khá nặng, cần
kết nối mạng và vài GB dung lượng cache Hugging Face).

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Mặc định UI gọi API tại `http://localhost:8000`.

## Chạy bằng Docker Compose

```bash
docker compose up --build
```

Khởi động 3 service: `ollama` (11434), `backend` (8000), `frontend` (5173).
Nhớ `docker exec` vào container `ollama` để `ollama pull` model trước khi
dùng, vì image gốc không có sẵn model nào.

## API

### `POST /chat`
Chat nhanh, streaming, có RAG context tự động chèn vào system prompt.

```json
{ "messages": [{ "role": "user", "content": "..." }] }
```
Trả về: stream `text/plain`.

### `POST /analyze`
Chạy toàn bộ AI Lawyer Crew để phân tích một vụ việc theo IRAC.

```json
{ "case_description": "Mô tả vụ việc..." }
```
Trả về:
```json
{ "analysis": "...", "status": "completed" }
```
Lưu ý: endpoint này chạy nhiều agent tuần tự nên thời gian phản hồi lâu hơn
`/chat` đáng kể (không streaming).

### `GET /ping`
Health check, trả về `{"status": "ok"}`.

## Dữ liệu & Retrieval

- Văn bản luật gốc/đã xử lý nằm trong `data/` (`law-clean/`, `rag_chunks/`).
- `rag/index_documen.py` dùng để embed và nạp chunk vào ChromaDB.
- Có 2 thư mục ChromaDB trong repo (`chroma_db/` ở root và
  `backend/chroma_db/`) — kiểm tra biến `CHROMA_PATH` để biết backend đang
  trỏ vào thư mục nào trước khi debug retrieval.
- `backend/evaluate_retrieval_pipeline.py` dùng để đánh giá chất lượng pipeline
  truy hồi (coverage, độ chính xác trích dẫn...).

## Ghi chú / vấn đề đã biết

- Model chạy qua Ollama hiện dùng là bản nhẹ (`gemma4:e2b`/`e4b`), tool-calling
  không hoàn toàn ổn định — ví dụ agent `legal_researcher` đôi khi gộp nhiều
  vấn đề pháp lý thành một truy vấn duy nhất thay vì tra cứu song song từng
  vấn đề. Xem service `retrieval_pipeline_service.py` /
  `query_expansion_service.py` nếu cần chỉnh lại hành vi này.
- `backend/main.py` khởi tạo `SentenceTransformer` và `ChromaService` một lần
  khi app start; nếu ChromaDB không tải được, `/chat` vẫn chạy nhưng không có
  RAG context (`/analyze` vẫn cần Crew hoạt động đầy đủ).
- Đường dẫn `CHROMA_PATH` là tương đối theo thư mục chạy `uvicorn` — nếu đổi
  cách chạy (venv vs Docker), kiểm tra lại đường dẫn resolve đúng thư mục
  ChromaDB mong muốn.