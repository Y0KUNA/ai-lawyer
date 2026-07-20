import json
import os
import chromadb
from sentence_transformers import SentenceTransformer
import torch
import time
from pathlib import Path
# ── Config ──────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR.parent / "data" / "rag_chunks" / "rag_corpus.jsonl"
JSONL_PATH   = file_path
CHROMA_PATH  = BASE_DIR.parent / "chroma_db"
CHECKPOINT_F = "index_checkpoint.txt"  # lưu chunk_id đã index
BATCH_SIZE   = 32                         # nhỏ hơn để tránh lỗi compaction
# ────────────────────────────────────────────────────────────────────

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()}")

model = SentenceTransformer("BAAI/bge-m3", device=device)

client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(
    name="luat_vn",
    metadata={"hnsw:space": "cosine"}
)


def safe_meta(c: dict) -> dict:
    keys = ["so_hieu", "ten_luat", "loai_van_ban", "section_type",
            "heading", "ngay_hieu_luc", "status", "filename"]
    return {k: str(c.get(k) or "") for k in keys}


# Đọc checkpoint — những chunk_id đã index thành công
indexed_ids = set()
if os.path.exists(CHECKPOINT_F):
    with open(CHECKPOINT_F, "r", encoding="utf-8") as f:
        indexed_ids = set(line.strip() for line in f if line.strip())
    print(f"Checkpoint: đã có {len(indexed_ids)} chunks, tiếp tục từ đó...")


# Đọc toàn bộ chunks
all_chunks = []
with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            all_chunks.append(json.loads(line))

# Lọc ra những chunk chưa index
pending = [c for c in all_chunks if c["chunk_id"] not in indexed_ids]
print(f"Tổng: {len(all_chunks)} | Đã có: {len(indexed_ids)} | Còn lại: {len(pending)}")

if not pending:
    print("✅ Đã index hết rồi!")
    exit()


# Index theo batch với retry
checkpoint_file = open(CHECKPOINT_F, "a", encoding="utf-8")

total_done = len(indexed_ids)

for i in range(0, len(pending), BATCH_SIZE):
    batch = pending[i : i + BATCH_SIZE]
    texts = [c["text"]     for c in batch]
    ids   = [c["chunk_id"] for c in batch]
    metas = [safe_meta(c)  for c in batch]

    # Retry tối đa 3 lần nếu lỗi
    for attempt in range(3):
        try:
            embeddings = model.encode(texts, show_progress_bar=False).tolist()
            collection.add(documents=texts, embeddings=embeddings, ids=ids, metadatas=metas)

            # Ghi checkpoint ngay sau khi add thành công
            for chunk_id in ids:
                checkpoint_file.write(chunk_id + "\n")
            checkpoint_file.flush()

            total_done += len(batch)
            print(f"  [{total_done}/{len(all_chunks)}] batch {i//BATCH_SIZE + 1} ✓")
            break  # thành công, thoát retry

        except Exception as e:
            print(f"  ⚠️  Lỗi attempt {attempt+1}/3: {e}")
            if attempt < 2:
                time.sleep(3)  # đợi 3 giây rồi thử lại
            else:
                print(f"  ❌ Bỏ qua batch này sau 3 lần thất bại")

checkpoint_file.close()
print(f"\n✅ Hoàn tất! Tổng đã index: {total_done}/{len(all_chunks)}")