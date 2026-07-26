"""
load_chroma_bge_m3.py
Reads the JSONL corpus produced by chunk_laws_uts_v2_fixed.py, embeds each
chunk with BAAI/bge-m3, and loads everything into a persistent ChromaDB
collection.
Install:
    pip install chromadb sentence-transformers --break-system-packages
Run:
    python load_chroma_bge_m3.py
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Iterator, List, Dict, Any
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
# ============================================================
# CONFIG
# ============================================================
JSONL_PATH = Path("D:\\Huy\\AI-Lawyer\\ai-lawyer\\data\\rag_chunks_uts_v2\\rag_corpus.jsonl")
CHROMA_DIR = Path("D:/Huy/AI-Lawyer/ai-lawyer/chroma_db")
COLLECTION_NAME = "luat_vn"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32          # tune down if you hit OOM on GPU/CPU
MAX_SEQ_LENGTH = 8192    # bge-m3 supports long context, unlike bge-large
# BGE convention: passages are embedded as-is, only the *query* side gets an
# instruction prefix at search time. Do not add this prefix to chunks.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
# Which text field to embed. `embedding_text` (built in the chunker) already
# includes law name / article / chapter context, which usually improves
# retrieval quality over embedding the bare chunk `text`.
EMBED_FIELD = "embedding_text"
# Fields that go into chunk.metadata in the chunker output and must stay
# flat (str/int/float/bool) for Chroma. We just pass everything through
# except the two big text fields, which become `document` + are dropped
# from metadata to avoid duplicating large text twice.
TEXT_FIELDS = {"text", "embedding_text"}
# ============================================================
# MODEL
# ============================================================
def load_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = MAX_SEQ_LENGTH
    return model
def embed_passages(model: SentenceTransformer, texts: List[str]) -> List[List[float]]:
    """Embed document/passage chunks (no instruction prefix)."""
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # bge-m3 dense vectors are meant to be L2-normalized
        show_progress_bar=False,
    )
    return vectors.tolist()
def embed_query(model: SentenceTransformer, query: str) -> List[float]:
    """Embed a search query (with the BGE instruction prefix)."""
    vector = model.encode(
        QUERY_INSTRUCTION + query,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()
# ============================================================
# JSONL READER
# ============================================================
def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
def batched(iterable, size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
def record_to_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    """Strip out the big text fields + chunk_id, keep everything else.
    Also guards against None values, which Chroma metadata rejects."""
    meta = {
        k: v for k, v in record.items() if k not in TEXT_FIELDS and k != "chunk_id"
    }
    for k, v in meta.items():
        if v is None:
            meta[k] = ""
    return meta
# ============================================================
# LOAD PIPELINE
# ============================================================
def get_collection() -> chromadb.Collection:
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    # We supply our own bge-m3 embeddings, so no embedding_function here.
    # cosine space matches normalized bge-m3 vectors.
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    return collection
def load_into_chroma() -> None:
    if not JSONL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {JSONL_PATH}. Hãy chạy chunk_laws_uts_v2_fixed.py trước."
        )
    model = load_model()
    collection = get_collection()
    total = 0
    for batch in batched(iter_jsonl(JSONL_PATH), BATCH_SIZE):
        ids = [r["chunk_id"] for r in batch]
        texts_to_embed = [r[EMBED_FIELD] for r in batch]
        documents = [r["text"] for r in batch]
        metadatas = [record_to_metadata(r) for r in batch]
        embeddings = embed_passages(model, texts_to_embed)
        # upsert (not add) so re-running the script is idempotent
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total += len(batch)
        print(f"Đã nạp {total:,} chunks...", end="\r")
    print(f"\nHoàn tất. Tổng số chunks trong collection '{COLLECTION_NAME}': "
          f"{collection.count():,}")
# ============================================================
# QUERY EXAMPLE
# ============================================================
def search(query: str, n_results: int = 5, where: Dict[str, Any] | None = None):
    model = load_model()
    collection = get_collection()
    query_vector = embed_query(model, query)
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=n_results,
        where=where,  # e.g. {"document_type": "Bộ luật"}
    )
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        print(f"[{dist:.4f}] Điều {meta.get('article_number')} "
              f"Khoản {meta.get('clause_number') or '-'} "
              f"({meta.get('law_name')})")
        print(doc[:200].replace("\n", " "))
        print("-" * 60)
if __name__ == "__main__":
    load_into_chroma()
    # Ví dụ tìm kiếm sau khi nạp xong:
    # search("phạt cọc khi một bên từ chối giao kết hợp đồng")