#!/usr/bin/env python3
"""
benchmark_rag.py
------------------------------------------------------------
Benchmark hệ thống RAG của ai-lawyer. Đo 3 nhóm chỉ số:

  A. End-to-end qua endpoint /chat (streaming)
     - Độ chính xác câu trả lời (answer accuracy, dựa trên từ khoá kỳ vọng)
     - Thời gian: TTFT, tổng thời gian, ký tự/giây, p50/p95/p99, tỷ lệ lỗi

  B. Retrieval-only trực tiếp trên ChromaDB (không qua LLM)
     - Hit@k, Precision@k, MRR
     - Thời gian embed + thời gian truy vấn vector

  C. Tải đồng thời (tuỳ chọn)
     - Throughput (req/s) và latency khi chạy N request song song

Cách chạy:
  pip install requests chromadb
  python benchmark_rag.py \
      --dataset rag_cases.json \
      --chroma-path ../rag/chroma_db \
      --embed-model nomic-embed-text \
      --concurrency 4

Định dạng dataset (JSON):
  [
    {"question": "...", "expected_keywords": ["60 ngày", "60"], "category": "lao dong"},
    ...
  ]
  ("category" là tuỳ chọn, dùng để thống kê theo nhóm luật.)
  Nếu không truyền --dataset, script sẽ thử import RAG_ACCURACY_CASES từ
  benchmark_ai_lawyer.py nằm cùng thư mục.
------------------------------------------------------------
"""

import argparse
import importlib.util
import json
import statistics
import sys
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

CHAT_URL_DEFAULT = "http://127.0.0.1:8000/chat"
OLLAMA_URL_DEFAULT = "http://127.0.0.1:11434"


# ============================================================
# Tiện ích
# ============================================================

def normalize(text: str, strip_accents: bool = False) -> str:
    """Chuẩn hoá unicode (NFC), hạ chữ thường; tuỳ chọn bỏ dấu tiếng Việt."""
    text = unicodedata.normalize("NFC", text).lower()
    if strip_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
        text = text.replace("đ", "d")
    return text


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def describe(values: list[float]) -> dict:
    values = [v for v in values if v is not None]
    if not values:
        return {}
    return {
        "mean": round(statistics.mean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 95), 3),
        "p99": round(percentile(values, 99), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "stdev": round(statistics.pstdev(values), 3) if len(values) > 1 else 0.0,
    }


def load_cases(path: str | None) -> list[dict]:
    if path:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    legacy = Path(__file__).with_name("benchmark_ai_lawyer.py")
    if legacy.exists():
        spec = importlib.util.spec_from_file_location("legacy_bench", legacy)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.RAG_ACCURACY_CASES
    sys.exit("Không tìm thấy dataset. Truyền --dataset <file.json>.")


def is_match(text: str, keywords: list[str], mode: str, strip_accents: bool) -> bool:
    t = normalize(text, strip_accents)
    found = [normalize(k, strip_accents) in t for k in keywords]
    return all(found) if mode == "all" else any(found)


# ============================================================
# A. End-to-end qua /chat
# ============================================================

def ask_chat(chat_url: str, question: str, timeout: int) -> dict:
    """Gửi 1 câu hỏi (streaming), đo TTFT + tổng thời gian + lấy lại câu trả lời."""
    payload = {"messages": [{"role": "user", "content": question}]}
    start = time.perf_counter()
    first = None
    parts: list[str] = []
    try:
        with requests.post(chat_url, json=payload, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            r.encoding = "utf-8"
            for chunk in r.iter_content(chunk_size=None, decode_unicode=True):
                if not chunk:
                    continue
                if first is None:
                    first = time.perf_counter()
                parts.append(chunk)
    except Exception as e:
        return {"question": question, "error": str(e),
                "total_time_sec": time.perf_counter() - start}

    end = time.perf_counter()
    answer = "".join(parts)
    total = end - start
    gen_time = (end - first) if first else None
    return {
        "question": question,
        "answer": answer,
        "ttft_sec": (first - start) if first else None,
        "total_time_sec": total,
        "total_chars": len(answer),
        "chars_per_sec": len(answer) / total if total > 0 else 0,
        "gen_chars_per_sec": (len(answer) / gen_time) if gen_time and gen_time > 0 else None,
        "error": None if answer else "empty_response",
    }


def run_e2e(args, cases: list[dict]) -> dict:
    print(f"\n=== A. End-to-end /chat ({len(cases)} câu x {args.repeats} lượt) ===")

    if args.warmup > 0:
        print(f"  Warm-up {args.warmup} request (loại khỏi thống kê, tránh cold-start của Ollama)...")
        for _ in range(args.warmup):
            ask_chat(args.chat_url, cases[0]["question"], args.timeout)

    runs = []          # mọi lượt đo (dùng cho latency)
    first_pass = {}    # lượt đầu mỗi câu (dùng cho accuracy)
    for rep in range(args.repeats):
        for i, case in enumerate(cases, 1):
            res = ask_chat(args.chat_url, case["question"], args.timeout)
            res["category"] = case.get("category", "all")
            runs.append(res)
            if rep == 0:
                first_pass[case["question"]] = (case, res)
            if res.get("error"):
                print(f"  [{rep+1}.{i:02d}] ✗ LỖI: {res['error'][:80]}")
            else:
                print(f"  [{rep+1}.{i:02d}] TTFT={res['ttft_sec']:.2f}s "
                      f"tổng={res['total_time_sec']:.2f}s "
                      f"{res['chars_per_sec']:.0f} ký tự/s")

    ok = [r for r in runs if not r.get("error")]
    latency = {
        "n_requests": len(runs),
        "n_errors": len(runs) - len(ok),
        "error_rate": round((len(runs) - len(ok)) / len(runs), 3) if runs else 0,
        "ttft_sec": describe([r["ttft_sec"] for r in ok]),
        "total_time_sec": describe([r["total_time_sec"] for r in ok]),
        "chars_per_sec": describe([r["chars_per_sec"] for r in ok]),
    }

    # Độ chính xác câu trả lời: lỗi tính là trả lời sai (mẫu số = tổng số câu)
    hits, per_cat, details = 0, defaultdict(lambda: [0, 0]), []
    for q, (case, res) in first_pass.items():
        cat = case.get("category", "all")
        hit = (not res.get("error")) and is_match(
            res["answer"], case["expected_keywords"], args.match, args.strip_accents)
        hits += hit
        per_cat[cat][0] += hit
        per_cat[cat][1] += 1
        details.append({
            "question": q,
            "category": cat,
            "hit": bool(hit),
            "total_time_sec": round(res["total_time_sec"], 3),
            "error": res.get("error"),
            "answer_preview": (res.get("answer") or "")[:200],
        })

    accuracy = {
        "answer_accuracy": round(hits / len(cases), 3) if cases else 0,
        "n_cases": len(cases),
        "match_mode": args.match,
        "by_category": {c: {"accuracy": round(h / n, 3), "n": n} for c, (h, n) in per_cat.items()},
        "details": details,
    }
    return {"latency": latency, "accuracy": accuracy}


# ============================================================
# B. Retrieval-only trên ChromaDB
# ============================================================

def ollama_embed(base_url: str, model: str, text: str, timeout: int = 60) -> list[float]:
    """Hỗ trợ cả /api/embed (mới) và /api/embeddings (cũ)."""
    r = requests.post(f"{base_url}/api/embed", json={"model": model, "input": text}, timeout=timeout)
    if r.status_code == 200 and r.json().get("embeddings"):
        return r.json()["embeddings"][0]
    r = requests.post(f"{base_url}/api/embeddings", json={"model": model, "prompt": text}, timeout=timeout)
    r.raise_for_status()
    return r.json()["embedding"]


def get_collection(chroma_path: str, name: str | None):
    import chromadb
    client = chromadb.PersistentClient(path=chroma_path)
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    if not names:
        sys.exit(f"ChromaDB tại {chroma_path} không có collection nào.")
    if name is None:
        if len(names) > 1:
            sys.exit(f"Có nhiều collection {names}. Hãy chỉ định bằng --collection.")
        name = names[0]
    return client.get_collection(name), names


def run_retrieval(args, cases: list[dict]) -> dict | None:
    print(f"\n=== B. Retrieval-only trên ChromaDB (top_k={args.top_k}) ===")
    try:
        col, names = get_collection(args.chroma_path, args.collection)
    except ImportError:
        print("  [BỎ QUA] Chưa cài chromadb (pip install chromadb).")
        return None
    except Exception as e:
        print(f"  [BỎ QUA] Không mở được ChromaDB: {e}")
        return None

    n_chunks = col.count()
    print(f"  Collection '{col.name}': {n_chunks} chunk")

    ks = [k for k in (1, 3, 5, 10) if k <= args.top_k]
    if args.top_k not in ks:
        ks.append(args.top_k)

    embed_times, query_times = [], []
    rr_list, details = [], []
    hit_at = {k: 0 for k in ks}
    prec_at = {k: [] for k in ks}
    n_err = 0

    for i, case in enumerate(cases, 1):
        q = case["question"]
        try:
            t0 = time.perf_counter()
            if args.embed_model:
                emb = ollama_embed(args.ollama_url, args.embed_model, q)
                t1 = time.perf_counter()
                res = col.query(query_embeddings=[emb], n_results=args.top_k,
                                include=["documents", "distances"])
            else:
                t1 = t0  # collection tự embed, không tách riêng được
                res = col.query(query_texts=[q], n_results=args.top_k,
                                include=["documents", "distances"])
            t2 = time.perf_counter()
        except Exception as e:
            n_err += 1
            details.append({"question": q, "error": str(e)})
            print(f"  [{i:02d}] ✗ LỖI: {str(e)[:80]}")
            continue

        docs = res["documents"][0]
        rel = [is_match(d, case["expected_keywords"], args.match, args.strip_accents) for d in docs]
        first_rank = next((r for r, ok in enumerate(rel, 1) if ok), None)

        rr_list.append(1 / first_rank if first_rank else 0.0)
        for k in ks:
            hit_at[k] += any(rel[:k])
            prec_at[k].append(sum(rel[:k]) / k)
        if args.embed_model:
            embed_times.append(t1 - t0)
        query_times.append(t2 - t1)

        details.append({
            "question": q,
            "first_relevant_rank": first_rank,
            "relevant_flags": rel,
            "top_distance": round(res["distances"][0][0], 4) if res.get("distances") else None,
            "embed_time_sec": round(t1 - t0, 4) if args.embed_model else None,
            "query_time_sec": round(t2 - t1, 4),
        })
        print(f"  [{i:02d}] rank={first_rank or '-':>2} "
              f"query={(t2 - t1) * 1000:.0f}ms")

    n_eval = len(cases) - n_err
    return {
        "collection": col.name,
        "all_collections": names,
        "n_chunks": n_chunks,
        "top_k": args.top_k,
        "n_eval": n_eval,
        "n_errors": n_err,
        "hit_at_k": {f"hit@{k}": round(hit_at[k] / n_eval, 3) if n_eval else 0 for k in ks},
        "precision_at_k": {f"p@{k}": round(statistics.mean(prec_at[k]), 3) if prec_at[k] else 0 for k in ks},
        "mrr": round(statistics.mean(rr_list), 3) if rr_list else 0,
        "embed_time_sec": describe(embed_times),
        "vector_query_time_sec": describe(query_times),
        "note": ("Relevance = chunk chứa từ khoá kỳ vọng (proxy). "
                 "Nên bổ sung ground-truth chunk/doc id để đo chính xác hơn."),
        "details": details,
    }


# ============================================================
# C. Tải đồng thời
# ============================================================

def run_load(args, cases: list[dict]) -> dict:
    n = args.concurrency
    questions = [c["question"] for c in cases] * args.load_rounds
    print(f"\n=== C. Tải đồng thời: {len(questions)} request, {n} luồng ===")

    t0 = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(ask_chat, args.chat_url, q, args.timeout) for q in questions]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.perf_counter() - t0

    ok = [r for r in results if not r.get("error")]
    out = {
        "concurrency": n,
        "n_requests": len(results),
        "n_errors": len(results) - len(ok),
        "wall_time_sec": round(wall, 2),
        "throughput_req_per_sec": round(len(ok) / wall, 3) if wall > 0 else 0,
        "ttft_sec": describe([r["ttft_sec"] for r in ok]),
        "total_time_sec": describe([r["total_time_sec"] for r in ok]),
    }
    print(f"  Xong trong {wall:.1f}s | throughput={out['throughput_req_per_sec']} req/s "
          f"| lỗi={out['n_errors']}/{len(results)}")
    return out


# ============================================================
# Báo cáo
# ============================================================

def print_summary(report: dict):
    print("\n" + "=" * 64)
    print("TÓM TẮT")
    print("=" * 64)

    e2e = report.get("e2e")
    if e2e:
        lat, acc = e2e["latency"], e2e["accuracy"]
        print(f"[E2E] Độ chính xác câu trả lời : {acc['answer_accuracy']*100:.1f}% "
              f"({acc['n_cases']} câu, match={acc['match_mode']})")
        for c, v in acc["by_category"].items():
            if len(acc["by_category"]) > 1:
                print(f"        - {c:<18}: {v['accuracy']*100:.1f}% (n={v['n']})")
        for key, label in (("ttft_sec", "TTFT"), ("total_time_sec", "Tổng thời gian")):
            s = lat.get(key) or {}
            if s:
                print(f"[E2E] {label:<14}: mean={s['mean']}s  p50={s['median']}s  "
                      f"p95={s['p95']}s  max={s['max']}s")
        cps = lat.get("chars_per_sec") or {}
        if cps:
            print(f"[E2E] Tốc độ sinh            : ~{cps['mean']:.0f} ký tự/s")
        print(f"[E2E] Tỷ lệ lỗi              : {lat['error_rate']*100:.1f}% "
              f"({lat['n_errors']}/{lat['n_requests']})")

    ret = report.get("retrieval")
    if ret:
        print(f"[RAG] Số chunk đã index      : {ret['n_chunks']}")
        print("[RAG] " + "  ".join(f"{k}={v*100:.1f}%" for k, v in ret["hit_at_k"].items()))
        print(f"[RAG] MRR                    : {ret['mrr']}")
        q = ret.get("vector_query_time_sec") or {}
        if q:
            print(f"[RAG] Truy vấn vector        : mean={q['mean']*1000:.0f}ms  p95={q['p95']*1000:.0f}ms")
        em = ret.get("embed_time_sec") or {}
        if em:
            print(f"[RAG] Embed câu hỏi          : mean={em['mean']*1000:.0f}ms  p95={em['p95']*1000:.0f}ms")

    load = report.get("load")
    if load:
        print(f"[LOAD] {load['concurrency']} luồng               : "
              f"{load['throughput_req_per_sec']} req/s, "
              f"p95 tổng={load['total_time_sec'].get('p95', 'n/a')}s, "
              f"lỗi={load['n_errors']}/{load['n_requests']}")


def main():
    p = argparse.ArgumentParser(description="Benchmark RAG cho ai-lawyer")
    p.add_argument("--chat-url", default=CHAT_URL_DEFAULT)
    p.add_argument("--dataset", default=None, help="File JSON chứa bộ câu hỏi + từ khoá kỳ vọng")
    p.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu (chạy thử nhanh)")
    p.add_argument("--repeats", type=int, default=1, help="Số lượt lặp để đo latency ổn định hơn")
    p.add_argument("--warmup", type=int, default=1, help="Số request khởi động (không tính)")
    p.add_argument("--timeout", type=int, default=180)
    p.add_argument("--match", choices=["any", "all"], default="any",
                   help="any: chứa ít nhất 1 từ khoá; all: phải chứa tất cả")
    p.add_argument("--strip-accents", action="store_true", help="So khớp không phân biệt dấu")

    p.add_argument("--chroma-path", default=None, help="Thư mục ChromaDB (bật đo retrieval)")
    p.add_argument("--collection", default=None)
    p.add_argument("--embed-model", default=None,
                   help="Model embedding của Ollama (nếu bỏ trống, dùng embedding function của collection)")
    p.add_argument("--ollama-url", default=OLLAMA_URL_DEFAULT)
    p.add_argument("--top-k", type=int, default=5)

    p.add_argument("--concurrency", type=int, default=0, help=">1 để chạy test tải đồng thời")
    p.add_argument("--load-rounds", type=int, default=1)

    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument("--output", default="benchmark_results.json")
    args = p.parse_args()

    cases = load_cases(args.dataset)
    if args.limit:
        cases = cases[:args.limit]
    print(f"Đã nạp {len(cases)} câu hỏi.")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "config": {k: v for k, v in vars(args).items()},
        "n_cases": len(cases),
    }

    if not args.skip_e2e:
        report["e2e"] = run_e2e(args, cases)
    if args.chroma_path:
        report["retrieval"] = run_retrieval(args, cases)
    else:
        print("\n(Bỏ qua phần B — truyền --chroma-path để đo retrieval)")
    if args.concurrency and args.concurrency > 1:
        report["load"] = run_load(args, cases)

    print_summary(report)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Đã lưu kết quả chi tiết vào {out}")


if __name__ == "__main__":
    main()