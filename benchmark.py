"""
benchmark_ai_lawyer.py
------------------------------------------------------------
Script đo số liệu định lượng cho dự án ai-lawyer (Y0KUNA/ai-lawyer)
để lấy dữ liệu đưa vào CV.

Đo 3 nhóm:
  1. Độ trễ (latency) & throughput của endpoint /chat (streaming)
  2. Quy mô dữ liệu RAG (số chunk/document trong ChromaDB)
  3. Độ chính xác truy hồi RAG (đơn giản, dựa trên từ khoá kỳ vọng)

Cách chạy:
  1. Đảm bảo Ollama đang chạy: `ollama serve`
  2. Đảm bảo backend đang chạy: `uvicorn main:app --reload --port 8000`
     (chạy từ thư mục backend/ của repo ai-lawyer)
  3. cd vào thư mục chứa script này rồi chạy:
       pip install requests chromadb --break-system-packages
       python benchmark_ai_lawyer.py --chroma-path /path/to/ai-lawyer/rag/chroma_db

Kết quả in ra console + lưu vào benchmark_results.json
------------------------------------------------------------
"""

import argparse
import json
import time
import statistics
import sys

import requests

CHAT_URL_DEFAULT = "http://127.0.0.1:8000/chat"

# Câu hỏi mẫu dùng để đo latency / throughput.
# Bạn nên thay bằng câu hỏi thực tế phù hợp với domain pháp lý của app.
SAMPLE_QUESTIONS = [
    "Hợp đồng lao động thử việc có thời hạn tối đa bao lâu?",
    "Người lao động bị sa thải trái luật thì được bồi thường gì?",
    "Điều kiện để ly hôn đơn phương là gì?",
    "Thời hiệu khởi kiện tranh chấp hợp đồng dân sự là bao lâu?",
    "Doanh nghiệp có bắt buộc phải đóng bảo hiểm xã hội cho nhân viên thử việc không?",
]

# Bộ test đơn giản cho độ chính xác RAG: mỗi câu hỏi kèm 1-2 từ khoá
# kỳ vọng phải xuất hiện trong câu trả lời nếu RAG hoạt động đúng.
# Sửa lại cho khớp với bộ dữ liệu pháp luật thật của bạn.
RAG_ACCURACY_CASES = [
    # ============================================================
    # 1. LUẬT LAO ĐỘNG
    # ============================================================

    {
        "question": "Hợp đồng lao động thử việc có thời hạn tối đa bao lâu?",
        "expected_keywords": ["60 ngày", "60 ngày làm việc", "60"]
    },
    {
        "question": "Người lao động bị sa thải trái pháp luật thì được bồi thường như thế nào?",
        "expected_keywords": ["tiền lương", "bồi thường", "nhận người lao động trở lại"]
    },
    {
        "question": "Doanh nghiệp có bắt buộc phải đóng bảo hiểm xã hội cho người lao động thử việc không?",
        "expected_keywords": ["bảo hiểm xã hội", "hợp đồng thử việc", "không"]
    },
    {
        "question": "Người lao động được nghỉ phép năm bao nhiêu ngày?",
        "expected_keywords": ["12 ngày", "12 ngày làm việc"]
    },
    {
        "question": "Thời giờ làm việc bình thường của người lao động tối đa bao nhiêu giờ trong một ngày?",
        "expected_keywords": ["8 giờ", "8 giờ trong một ngày"]
    },

    # ============================================================
    # 2. LUẬT DÂN SỰ
    # ============================================================

    {
        "question": "Thời hiệu khởi kiện tranh chấp hợp đồng dân sự là bao lâu?",
        "expected_keywords": ["3 năm", "ba năm"]
    },
    {
        "question": "Người từ đủ bao nhiêu tuổi có năng lực hành vi dân sự đầy đủ?",
        "expected_keywords": ["18 tuổi", "đủ 18 tuổi"]
    },
    {
        "question": "Giao dịch dân sự vô hiệu thì hậu quả pháp lý như thế nào?",
        "expected_keywords": ["khôi phục lại tình trạng ban đầu", "hoàn trả", "hoàn trả cho nhau"]
    },
    {
        "question": "Hợp đồng đặt cọc có phải lập thành văn bản không?",
        "expected_keywords": ["đặt cọc", "văn bản", "thỏa thuận"]
    },
    {
        "question": "Bên nhận đặt cọc từ chối giao kết hoặc thực hiện hợp đồng thì phải trả lại tiền đặt cọc như thế nào?",
        "expected_keywords": ["trả lại", "gấp đôi", "phạt cọc"]
    },

    # ============================================================
    # 3. HÔN NHÂN VÀ GIA ĐÌNH
    # ============================================================

    {
        "question": "Điều kiện để được ly hôn đơn phương là gì?",
        "expected_keywords": ["tình trạng hôn nhân", "trầm trọng", "đời sống chung"]
    },
    {
        "question": "Con dưới 36 tháng tuổi khi ly hôn thường được giao cho ai trực tiếp nuôi?",
        "expected_keywords": ["36 tháng", "mẹ", "người mẹ"]
    },
    {
        "question": "Tài sản chung của vợ chồng khi ly hôn được chia như thế nào?",
        "expected_keywords": ["chia đôi", "công sức đóng góp", "hoàn cảnh"]
    },
    {
        "question": "Vợ chồng có được thỏa thuận chia tài sản chung trong thời kỳ hôn nhân không?",
        "expected_keywords": ["thỏa thuận", "chia tài sản chung"]
    },
    {
        "question": "Cha mẹ có nghĩa vụ cấp dưỡng cho con sau khi ly hôn không?",
        "expected_keywords": ["cấp dưỡng", "con", "nghĩa vụ"]
    },

    # ============================================================
    # 4. ĐẤT ĐAI
    # ============================================================

    {
        "question": "Điều kiện để được cấp Giấy chứng nhận quyền sử dụng đất là gì?",
        "expected_keywords": ["Giấy chứng nhận", "quyền sử dụng đất", "điều kiện"]
    },
    {
        "question": "Hợp đồng chuyển nhượng quyền sử dụng đất có bắt buộc phải công chứng hoặc chứng thực không?",
        "expected_keywords": ["công chứng", "chứng thực", "quyền sử dụng đất"]
    },
    {
        "question": "Khi Nhà nước thu hồi đất thì người sử dụng đất có được bồi thường không?",
        "expected_keywords": ["bồi thường", "thu hồi đất"]
    },
    {
        "question": "Đất không có Giấy chứng nhận có được chuyển nhượng không?",
        "expected_keywords": ["Giấy chứng nhận", "chuyển nhượng", "điều kiện"]
    },
    {
        "question": "Thời hạn sử dụng đất ở của hộ gia đình và cá nhân là bao lâu?",
        "expected_keywords": ["lâu dài", "ổn định lâu dài", "đất ở"]
    },

    # ============================================================
    # 5. HỢP ĐỒNG VÀ THƯƠNG MẠI
    # ============================================================

    {
        "question": "Điều kiện để một hợp đồng dân sự có hiệu lực là gì?",
        "expected_keywords": ["năng lực hành vi dân sự", "tự nguyện", "nội dung"]
    },
    {
        "question": "Một bên vi phạm hợp đồng thì có thể phải chịu những trách nhiệm gì?",
        "expected_keywords": ["bồi thường thiệt hại", "phạt vi phạm", "thực hiện đúng hợp đồng"]
    },
    {
        "question": "Phạt vi phạm hợp đồng có cần phải được các bên thỏa thuận trước không?",
        "expected_keywords": ["thỏa thuận", "phạt vi phạm"]
    },
    {
        "question": "Bồi thường thiệt hại do vi phạm hợp đồng bao gồm những khoản thiệt hại nào?",
        "expected_keywords": ["thiệt hại thực tế", "tổn thất", "lợi ích"]
    },
    {
        "question": "Một bên có được đơn phương chấm dứt hợp đồng khi bên còn lại vi phạm nghĩa vụ không?",
        "expected_keywords": ["đơn phương chấm dứt", "vi phạm", "nghĩa vụ"]
    },

    # ============================================================
    # 6. THỪA KẾ
    # ============================================================

    {
        "question": "Thời hiệu để người thừa kế yêu cầu chia di sản là bao lâu?",
        "expected_keywords": ["30 năm", "30 năm đối với bất động sản"]
    },
    {
        "question": "Người thừa kế có được quyền từ chối nhận di sản không?",
        "expected_keywords": ["từ chối nhận di sản", "từ chối", "di sản"]
    },
    {
        "question": "Những người nào được hưởng thừa kế theo pháp luật ở hàng thừa kế thứ nhất?",
        "expected_keywords": ["vợ", "chồng", "cha đẻ", "mẹ đẻ", "con"]
    },
    {
        "question": "Con chưa thành niên có được hưởng thừa kế dù không có tên trong di chúc không?",
        "expected_keywords": ["chưa thành niên", "di chúc", "phần di sản"]
    },
    {
        "question": "Di chúc bằng miệng có được pháp luật công nhận không?",
        "expected_keywords": ["di chúc bằng miệng", "công nhận", "tính mạng"]
    },

    # ============================================================
    # 7. BẢO HIỂM XÃ HỘI
    # ============================================================

    {
        "question": "Người lao động tham gia bảo hiểm xã hội bắt buộc được hưởng những chế độ nào?",
        "expected_keywords": ["ốm đau", "thai sản", "hưu trí", "tử tuất"]
    },
    {
        "question": "Điều kiện để người lao động được hưởng lương hưu là gì?",
        "expected_keywords": ["đủ tuổi", "đóng bảo hiểm xã hội", "20 năm"]
    },
    {
        "question": "Người lao động nghỉ việc có được hưởng bảo hiểm thất nghiệp không?",
        "expected_keywords": ["bảo hiểm thất nghiệp", "điều kiện", "trợ cấp thất nghiệp"]
    },
    {
        "question": "Thời gian hưởng trợ cấp thất nghiệp được xác định như thế nào?",
        "expected_keywords": ["3 tháng", "12 tháng", "thời gian đóng"]
    },
]


def measure_chat_latency(chat_url: str, questions: list[str], timeout: int = 120):
    """Đo time-to-first-byte, tổng thời gian, và tốc độ ký tự/giây cho mỗi câu hỏi."""
    results = []
    for q in questions:
        payload = {"messages": [{"role": "user", "content": q}]}
        start = time.perf_counter()
        first_chunk_time = None
        total_chars = 0

        try:
            with requests.post(chat_url, json=payload, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue
                    if first_chunk_time is None:
                        first_chunk_time = time.perf_counter()
                    total_chars += len(chunk)
        except Exception as e:
            print(f"[LỖI] Câu hỏi '{q[:30]}...': {e}", file=sys.stderr)
            continue

        end = time.perf_counter()
        ttft = (first_chunk_time - start) if first_chunk_time else None
        total_time = end - start
        chars_per_sec = total_chars / total_time if total_time > 0 else 0

        results.append({
            "question": q,
            "time_to_first_chunk_sec": round(ttft, 3) if ttft else None,
            "total_time_sec": round(total_time, 3),
            "total_chars": total_chars,
            "chars_per_sec": round(chars_per_sec, 1),
        })
        print(f"  ✓ '{q[:40]}...' -> TTFT={ttft:.2f}s, tổng={total_time:.2f}s, "
              f"{chars_per_sec:.0f} ký tự/s" if ttft else f"  ✗ '{q[:40]}...' không nhận được phản hồi")

    return results


def summarize_latency(results: list[dict]):
    ttfts = [r["time_to_first_chunk_sec"] for r in results if r["time_to_first_chunk_sec"]]
    totals = [r["total_time_sec"] for r in results]
    speeds = [r["chars_per_sec"] for r in results if r["chars_per_sec"]]

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
        "time_to_first_chunk_sec": stats(ttfts),
        "total_time_sec": stats(totals),
        "chars_per_sec": stats(speeds),
        "n_requests": len(results),
    }


def count_chroma_documents(chroma_path: str, collection_name: str | None = None):
    """Đếm số chunk/document đã index trong ChromaDB."""
    try:
        import chromadb
    except ImportError:
        print("[CẢNH BÁO] Chưa cài chromadb, bỏ qua bước đếm dữ liệu RAG.", file=sys.stderr)
        return None

    try:
        client = chromadb.PersistentClient(path=chroma_path)
        collections = client.list_collections()
        info = []
        for col in collections:
            count = col.count()
            info.append({"collection": col.name, "num_chunks": count})
        return info
    except Exception as e:
        print(f"[LỖI] Không đọc được ChromaDB tại {chroma_path}: {e}", file=sys.stderr)
        return None


def measure_rag_accuracy(chat_url: str, cases: list[dict], timeout: int = 120):
    """Đo tỷ lệ câu trả lời chứa đúng từ khoá kỳ vọng (proxy đơn giản cho độ chính xác RAG)."""
    hits = 0
    details = []
    for case in cases:
        payload = {"messages": [{"role": "user", "content": case["question"]}]}
        try:
            r = requests.post(chat_url, json=payload, timeout=timeout)
            r.raise_for_status()
            answer = r.text.lower()
        except Exception as e:
            print(f"[LỖI] {e}", file=sys.stderr)
            details.append({"question": case["question"], "hit": False, "error": str(e)})
            continue

        hit = any(kw.lower() in answer for kw in case["expected_keywords"])
        hits += hit
        details.append({"question": case["question"], "hit": hit})

    accuracy = hits / len(cases) if cases else 0
    return {"accuracy": round(accuracy, 3), "n_cases": len(cases), "details": details}


def main():
    parser = argparse.ArgumentParser(description="Đo số liệu định lượng cho ai-lawyer")
    parser.add_argument("--chat-url", default=CHAT_URL_DEFAULT)
    parser.add_argument("--chroma-path", default=None,
                         help="Đường dẫn tới thư mục rag/chroma_db trong repo")
    parser.add_argument("--skip-accuracy", action="store_true",
                         help="Bỏ qua bước đo độ chính xác RAG")
    args = parser.parse_args()

    print("=== 1. Đo latency & throughput của /chat ===")
    latency_results = measure_chat_latency(args.chat_url, SAMPLE_QUESTIONS)
    latency_summary = summarize_latency(latency_results)
    print(json.dumps(latency_summary, indent=2, ensure_ascii=False))

    rag_stats = None
    if args.chroma_path:
        print("\n=== 2. Đếm dữ liệu đã index trong ChromaDB ===")
        rag_stats = count_chroma_documents(args.chroma_path)
        print(json.dumps(rag_stats, indent=2, ensure_ascii=False))
    else:
        print("\n(Bỏ qua đếm ChromaDB — truyền --chroma-path để đo)")

    accuracy_result = None
    if not args.skip_accuracy:
        print("\n=== 3. Đo độ chính xác truy hồi RAG (proxy từ khoá) ===")
        accuracy_result = measure_rag_accuracy(args.chat_url, RAG_ACCURACY_CASES)
        print(json.dumps(accuracy_result, indent=2, ensure_ascii=False))

    output = {
        "latency_summary": latency_summary,
        "latency_raw": latency_results,
        "rag_document_stats": rag_stats,
        "rag_accuracy": accuracy_result,
    }

    with open("benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n✅ Đã lưu kết quả chi tiết vào benchmark_results.json")


if __name__ == "__main__":
    main()