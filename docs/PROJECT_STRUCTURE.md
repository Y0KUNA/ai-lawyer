# Cấu trúc dự án

Mục tiêu của cấu trúc này là tách rõ mã cần để chạy sản phẩm khỏi các tác vụ
nghiên cứu và dữ liệu sinh cục bộ.

```text
backend/       API FastAPI và AI Lawyer Crew đang phục vụ ứng dụng
frontend/      Giao diện React/Vite
data/          Nguồn luật, corpus đã xử lý và tập dữ liệu fine-tune
scripts/       Indexing, chuẩn bị dữ liệu, benchmark, evaluation, diagnostics
tests/         Test tự động và fixture có thể dùng lại
artifacts/     Kết quả sinh cục bộ (được bỏ qua bởi Git)
legacy/        Dữ liệu/công cụ lịch sử, không tham gia runtime
rag/           Prototype RAG cũ; backend/ mới là dịch vụ chính
```

## Lệnh tác vụ thủ công

Chạy từ thư mục gốc repository:

```powershell
python scripts/data_preparation/chunk_laws_uts_v2.py
python scripts/indexing/load_chroma_bge_m3.py
python scripts/evaluation/evaluate_retrieval_pipeline.py
python scripts/benchmarking/benchmark_ai_lawyer.py
```

Kết quả benchmark và evaluation mặc định được ghi vào `artifacts/`; không
trộn lẫn với mã nguồn hoặc fixture.
