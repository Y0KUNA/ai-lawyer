"""
evaluate_retrieval_pipeline.py
------------------------------------------------------------
Đánh giá chất lượng của RetrievalPipelineService trong
backend/crew/src/lawyers/services/retrieval_pipeline_service.py

Pipeline này gồm nhiều tầng: query expansion -> RAG (ChromaDB) ->
coverage evaluation -> HyDE fallback (nếu coverage thấp) -> EXA web
fallback (nếu vẫn thấp) -> reranker (cross-encoder) -> citation
verification -> compact.

Script này KHÔNG đi qua FastAPI /chat, mà gọi thẳng service bằng
Python để đo chính xác retrieval quality + latency từng pipeline,
tách biệt hoàn toàn khỏi tốc độ sinh text của LLM cuối cùng.

CÁCH CHẠY (bắt buộc chạy từ thư mục backend/, vì code gốc dùng
import kiểu `from crew.src.lawyers...` dựa vào cwd):

  cd ai-lawyer/backend
  pip install -r requirements.txt --break-system-packages
  pip install crewai crewai-tools --break-system-packages   # nếu chưa có
  # đặt file này vào thư mục backend/ rồi chạy:
  python evaluate_retrieval_pipeline.py --test-file retrieval_test_cases.json

Yêu cầu Ollama đang chạy (dùng cho HyDE) và ChromaDB đã có dữ liệu
(CHROMA_PATH trỏ đúng, mặc định ../chroma_db theo chroma_service.py).
------------------------------------------------------------
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ── Cho phép import `crew.src.lawyers...` giống hệt cách main.py làm ──
sys.path.insert(0, str(Path(__file__).parent))

# ── Đọc .env giống hệt main.py để lấy đúng CHROMA_PATH, EMBEDDING_MODEL... ──
from dotenv import load_dotenv
load_dotenv()

DEFAULT_TEST_FILE = "retrieval_test_cases.json"
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e4b")  # sửa nếu tên model khác
DEFAULT_OLLAMA_HOST = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


# ------------------------------------------------------------------
# Bắt log INFO của chính pipeline để biết HyDE/EXA có được kích hoạt
# không, mà không cần sửa code gốc.
# ------------------------------------------------------------------
class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records: List[str] = []

    def emit(self, record):
        self.records.append(record.getMessage())

    def reset(self):
        self.records = []

    def hyde_triggered(self) -> bool:
        return any("HyDE generated" in r and not r.startswith("HyDE generated 0") for r in self.records)

    def exa_triggered(self) -> bool:
        return any(r.startswith("EXA returned") and not r.startswith("EXA returned 0") for r in self.records)

    def coverage_values(self) -> List[float]:
        vals = []
        for r in self.records:
            if r.startswith("Coverage score:") or r.startswith("Coverage after HyDE:"):
                try:
                    vals.append(float(r.split(":")[-1].strip()))
                except ValueError:
                    pass
        return vals


def build_service(chroma_path: Optional[str] = None):
    """Khởi tạo RetrievalPipelineService giống hệt cách RetrievalPipelineTool làm."""
    from crewai import LLM
    from crew.src.lawyers.services.chroma_service import ChromaService
    from crew.src.lawyers.services.retrieval_pipeline_service import RetrievalPipelineService

    # Khởi tạo ChromaService TRƯỚC, với path chỉ định rõ ràng nếu có —
    # tránh việc RAGService() bên trong tự init với path mặc định sai.
    if chroma_path:
        ChromaService.initialize(path=chroma_path)
        print(f"[INFO] ChromaDB path (chỉ định thủ công): {chroma_path}")
    else:
        ChromaService.initialize()
        print(f"[INFO] ChromaDB path (theo CHROMA_PATH trong .env / mặc định code)")

    collection = ChromaService.get_collection()
    print(f"[INFO] Collection '{ChromaService._collection_name}' "
          f"có {collection.count() if collection else 0} chunk đã index")

    llm = LLM(model=f"ollama/{DEFAULT_MODEL}", base_url=DEFAULT_OLLAMA_HOST)
    return RetrievalPipelineService(llm)


# ------------------------------------------------------------------
# Đánh giá 1 case: so khớp chunk trả về với ground truth.
# Ưu tiên so khớp chính xác theo metadata (law_code/article) nếu case
# có khai báo; nếu không, fallback sang so khớp từ khóa trong text.
# ------------------------------------------------------------------
def evaluate_case(chunks: List, case: Dict) -> Dict:
    expected_law_code = case.get("expected_law_code")
    expected_article = case.get("expected_article")
    expected_keywords = case.get("expected_keywords", [])

    hit = False
    hit_rank = None
    matched_metadata = False

    for rank, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata or {}
        text = (chunk.text or "").lower()

        if expected_law_code or expected_article:
            law_ok = (not expected_law_code) or (
                str(metadata.get("law_code", "")).strip() == str(expected_law_code).strip()
            )
            article_ok = (not expected_article) or (chunk.article == expected_article)
            if law_ok and article_ok and (expected_law_code or expected_article):
                hit = True
                matched_metadata = True
                hit_rank = rank
                break
        elif expected_keywords:
            if any(kw.lower() in text for kw in expected_keywords):
                hit = True
                hit_rank = rank
                break

    return {
        "hit": hit,
        "hit_rank": hit_rank,
        "matched_via": "metadata" if matched_metadata else ("keyword" if hit else None),
        "reciprocal_rank": (1.0 / hit_rank) if hit_rank else 0.0,
    }


def run_evaluation(test_cases: List[Dict], n_queries: int, n_results: int, top_k: int,
                    coverage_threshold: float, chroma_path: Optional[str] = None) -> Dict:
    service = build_service(chroma_path=chroma_path)

    log_capture = _LogCapture()
    pipeline_logger = logging.getLogger("crew.src.lawyers.services.retrieval_pipeline_service")
    pipeline_logger.setLevel(logging.INFO)
    pipeline_logger.addHandler(log_capture)

    results = []
    for i, case in enumerate(test_cases, start=1):
        issue = case["issue"]
        print(f"[{i}/{len(test_cases)}] {issue[:60]}...")
        log_capture.reset()

        start = time.perf_counter()
        try:
            output = service.run(
                issue,
                n_queries=n_queries,
                n_results=n_results,
                top_k=top_k,
                coverage_threshold=coverage_threshold,
            )
            elapsed = time.perf_counter() - start
            chunks = output.get("chunks", [])
            eval_result = evaluate_case(chunks, case)

            # In ra danh sách luật/điều thật sự lọt vào top-k SAU rerank,
            # để kiểm tra thủ công xem reranker có lọc sạch chunk sai domain không.
            print("    Chunks sau rerank (final):")
            for rank, c in enumerate(chunks, start=1):
                law_name = (c.metadata or {}).get("law_name", "?")
                print(f"      [{rank}] score={c.score:.3f} | {law_name} - Điều {c.article}"
                      f"{f' khoản {c.clause}' if c.clause else ''} | {c.text[:70]}...")

            results.append({
                "issue": issue,
                "latency_sec": round(elapsed, 3),
                "num_chunks_returned": len(chunks),
                "final_coverage": round(output.get("coverage", 0.0), 3),
                "num_queries_used": len(output.get("queries", [])),
                "hyde_triggered": log_capture.hyde_triggered(),
                "exa_triggered": log_capture.exa_triggered(),
                "final_chunks_law_articles": [
                    {"law_name": (c.metadata or {}).get("law_name", "?"),
                     "article": c.article, "clause": c.clause, "score": round(c.score, 3)}
                    for c in chunks
                ],
                **eval_result,
            })
            status = "✓ HIT" if eval_result["hit"] else "✗ MISS"
            print(f"    {status} | {elapsed:.2f}s | coverage={output.get('coverage', 0):.2f} | "
                  f"chunks={len(chunks)} | hyde={log_capture.hyde_triggered()} | exa={log_capture.exa_triggered()}")

        except Exception as e:
            elapsed = time.perf_counter() - start
            print(f"    [LỖI] {e}")
            results.append({
                "issue": issue,
                "latency_sec": round(elapsed, 3),
                "error": str(e),
                "hit": False,
                "reciprocal_rank": 0.0,
            })

    pipeline_logger.removeHandler(log_capture)
    return summarize(results)


def summarize(results: List[Dict]) -> Dict:
    n = len(results)
    hits = [r for r in results if r.get("hit")]
    errors = [r for r in results if "error" in r]
    latencies = [r["latency_sec"] for r in results if "latency_sec" in r]
    coverages = [r["final_coverage"] for r in results if "final_coverage" in r]
    rr = [r.get("reciprocal_rank", 0.0) for r in results]
    hyde_count = sum(1 for r in results if r.get("hyde_triggered"))
    exa_count = sum(1 for r in results if r.get("exa_triggered"))

    def stats(values):
        if not values:
            return {}
        return {
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "min": round(min(values), 3),
            "max": round(max(values), 3),
        }

    return {
        "n_cases": n,
        "n_errors": len(errors),
        "hit_rate": round(len(hits) / n, 3) if n else 0.0,
        "mrr": round(sum(rr) / n, 3) if n else 0.0,  # Mean Reciprocal Rank
        "latency_sec": stats(latencies),
        "final_coverage": stats(coverages),
        "hyde_trigger_rate": round(hyde_count / n, 3) if n else 0.0,
        "exa_trigger_rate": round(exa_count / n, 3) if n else 0.0,
        "raw_results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Đánh giá RetrievalPipelineService")
    parser.add_argument("--test-file", default=DEFAULT_TEST_FILE)
    parser.add_argument("--n-queries", type=int, default=3)
    parser.add_argument("--n-results", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--chroma-path", default=None,
                         help="Chỉ định thủ công đường dẫn chroma_db nếu .env không có/không đúng")
    parser.add_argument("--output", default="retrieval_eval_results.json")
    args = parser.parse_args()

    with open(args.test_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print(f"Đang đánh giá {len(test_cases)} case, model={DEFAULT_MODEL}, "
          f"ollama_host={DEFAULT_OLLAMA_HOST}\n")

    summary = run_evaluation(
        test_cases,
        n_queries=args.n_queries,
        n_results=args.n_results,
        top_k=args.top_k,
        coverage_threshold=args.coverage_threshold,
        chroma_path=args.chroma_path,
    )

    print("\n" + "=" * 60)
    print("TỔNG KẾT")
    print("=" * 60)
    print(json.dumps(
        {k: v for k, v in summary.items() if k != "raw_results"},
        indent=2, ensure_ascii=False,
    ))

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Đã lưu chi tiết vào {args.output}")


if __name__ == "__main__":
    main()