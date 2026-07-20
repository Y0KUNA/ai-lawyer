"""
chunk_laws_hf.py — Chia chunk văn bản pháp luật từ dataset HuggingFace
                    th1nhng0/vietnamese-legal-documents cho RAG

Khác với bản gốc (đọc file .doc/.docx cục bộ), script này TẢI TRỰC TIẾP
văn bản từ HuggingFace Hub:

    metadata      — 153k văn bản, 16 trường metadata (số hiệu, ngày ban hành,
                    cơ quan ban hành, tình trạng hiệu lực, ...)
    content       — ~149k văn bản có nội dung HTML đầy đủ (join theo `id`)
    relationships — quan hệ giữa các văn bản (sửa đổi, thay thế, viện dẫn, ...)

Nguồn: https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents

Output:
    data/rag_chunks/
        rag_corpus.jsonl  — mỗi dòng là 1 chunk, sẵn để embed
        _stats.json       — thống kê tổng quan

Cài đặt:
    pip install datasets huggingface_hub pyarrow beautifulsoup4 lxml pandas tqdm

Lưu ý kỹ thuật: config 'content' và 'relationships' được đọc TRỰC TIẾP từ
file parquet bằng pyarrow (không qua `datasets(streaming=True)`), vì cột
`content_html` là kiểu `large_string` — khi `datasets` ép về kiểu `string`
(offset int32) sẽ báo lỗi:
    pyarrow.lib.ArrowInvalid: Failed casting from large_string to string:
    input array too large
Đọc thẳng bằng pyarrow giữ nguyên kiểu large_string nên tránh được lỗi này.
"""

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path

import pyarrow.parquet as pq
from bs4 import BeautifulSoup
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

# ─── CẤU HÌNH ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/rag_chunks")
MAX_CHUNK_CHARS = 1500   # Độ dài tối đa mỗi chunk (ký tự)
OVERLAP_CHARS   = 150    # Overlap giữa các chunk

# Chỉ lấy các loại văn bản dưới đây. Đặt = None để lấy TẤT CẢ loại văn bản.
LOAI_VAN_BAN_FILTER = ["Luật", "Bộ luật"]

# Giới hạn số văn bản xử lý (debug nhanh). Đặt None để chạy full.
MAX_DOCS = None

# Có kéo thêm bảng quan hệ (sửa đổi / thay thế / viện dẫn) để làm giàu
# metadata "sua_doi_luat" hay không. Bật lên sẽ chậm hơn một chút vì phải
# quét qua config `relationships` (898k dòng, streaming).
INCLUDE_RELATIONSHIPS = True

DATASET_NAME = "vohuutridung/vietnamese-legal-documents"

# Map tình trạng hiệu lực (tiếng Việt, tự do) → mã trạng thái chuẩn hoá.
# Nếu gặp giá trị lạ không có trong map, sẽ tự động "slugify" chuỗi gốc.
STATUS_MAP = {
    "còn hiệu lực":              "dang_hieu_luc",
    "hết hiệu lực toàn bộ":      "het_hieu_luc_toan_bo",
    "hết hiệu lực một phần":     "het_hieu_luc_mot_phan",
    "chưa có hiệu lực":          "chua_co_hieu_luc",
    "ngưng hiệu lực":            "ngung_hieu_luc",
    "đã bị sửa đổi":             "da_sua_doi",
    "đã bị sửa đổi, bổ sung":    "da_sua_doi",
    "đã bị thay thế":            "da_thay_the",
    "không xác định":            "khong_xac_dinh",
}
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def slugify(text: str) -> str:
    """Chuyển chuỗi tiếng Việt bất kỳ → snake_case không dấu, dùng làm mã trạng thái fallback."""
    if not text:
        return "khong_ro"
    text = text.replace("Đ", "D").replace("đ", "d")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return text or "khong_ro"


def map_status(tinh_trang_hieu_luc: str) -> str:
    if not tinh_trang_hieu_luc:
        return "khong_xac_dinh"
    key = tinh_trang_hieu_luc.strip().lower()
    return STATUS_MAP.get(key, slugify(tinh_trang_hieu_luc))


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: TẢI DATASET TỪ HUGGINGFACE — metadata, content, (relationships)
# ══════════════════════════════════════════════════════════════════════════════

def load_metadata_df():
    print("→ Tải config 'metadata' ...")
    ds = load_dataset(DATASET_NAME, "metadata", split="data")
    df = ds.to_pandas()
    df["id"] = df["id"].astype(str)
    print(f"  {len(df):,} văn bản (trước khi lọc)")
    return df


def filter_metadata_df(df):
    if LOAI_VAN_BAN_FILTER:
        df = df[df["loai_van_ban"].isin(LOAI_VAN_BAN_FILTER)]
        print(f"  → còn {len(df):,} văn bản sau khi lọc loai_van_ban={LOAI_VAN_BAN_FILTER}")
    if MAX_DOCS:
        df = df.head(MAX_DOCS)
        print(f"  → giới hạn MAX_DOCS={MAX_DOCS}")
    return df


_PARQUET_REVISION = "refs/convert/parquet"


def _list_parquet_files(config: str) -> list:
    """Liệt kê các file parquet (đã auto-convert) của 1 config trên HF Hub."""
    api = HfApi()
    files = api.list_repo_files(DATASET_NAME, repo_type="dataset", revision=_PARQUET_REVISION)
    prefix = f"{config}/"
    matched = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    return matched


def _iter_parquet_rows(config: str, columns: list):
    """
    Yield từng dòng (dict) của 1 config, đọc TRỰC TIẾP bằng pyarrow thay vì
    qua `datasets` streaming — tránh lỗi cast large_string→string khi cột
    chứa chuỗi rất lớn (content_html). Bộ nhớ được giữ thấp nhờ đọc theo
    batch (record_batch), không load cả file vào RAM cùng lúc.
    """
    parquet_files = _list_parquet_files(config)
    if not parquet_files:
        raise RuntimeError(
            f"Không tìm thấy file parquet nào cho config '{config}'. "
            f"Kiểm tra lại tên dataset hoặc revision '{_PARQUET_REVISION}'."
        )
    for fname in parquet_files:
        local_path = hf_hub_download(
            DATASET_NAME, fname, repo_type="dataset", revision=_PARQUET_REVISION
        )
        pf = pq.ParquetFile(local_path)
        for batch in pf.iter_batches(batch_size=2048, columns=columns):
            cols = {name: batch.column(name).to_pylist() for name in batch.schema.names}
            n = len(next(iter(cols.values()))) if cols else 0
            for i in range(n):
                yield {k: v[i] for k, v in cols.items()}


def load_content_for_ids(target_ids: set) -> dict:
    """
    Đọc config 'content' trực tiếp bằng pyarrow và chỉ giữ lại content_html
    của những id nằm trong target_ids (không tải hết ~3.6GB vào RAM).
    """
    print(f"→ Đọc config 'content' (pyarrow) để lấy {len(target_ids):,} văn bản cần dùng ...")
    result = {}
    remaining = set(target_ids)
    for row in tqdm(_iter_parquet_rows("content", columns=["id", "content_html"]),
                     desc="  duyệt content"):
        rid = str(row["id"])
        if rid in remaining:
            result[rid] = row["content_html"]
            remaining.discard(rid)
            if not remaining:
                break
    print(f"  → tìm thấy nội dung cho {len(result):,}/{len(target_ids):,} văn bản "
          f"({len(target_ids) - len(result):,} chỉ có bản scan PDF, không có HTML)")
    return result


def load_relationships_for_ids(target_ids: set) -> dict:
    """
    Đọc config 'relationships' trực tiếp bằng pyarrow và gom các quan hệ mà
    doc_id nằm trong target_ids, ví dụ 'Sửa đổi bổ sung', 'Thay thế', ...
    Trả về: {doc_id: [{"other_doc_id":..., "relationship":...}, ...]}
    """
    print("→ Đọc config 'relationships' (pyarrow) ...")
    result = {}
    for row in tqdm(_iter_parquet_rows("relationships",
                                        columns=["doc_id", "other_doc_id", "relationship"]),
                     desc="  duyệt relationships"):
        doc_id = str(row["doc_id"])
        if doc_id in target_ids:
            result.setdefault(doc_id, []).append({
                "other_doc_id": str(row["other_doc_id"]),
                "relationship": row["relationship"],
            })
    return result


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: HTML → PLAIN TEXT
# ══════════════════════════════════════════════════════════════════════════════

def html_to_text(content_html: str) -> str:
    """content_html là HTML thô của trang chi tiết văn bản trên vbpl.vn.
    Chuyển thành plain text, giữ xuống dòng theo <p>/<div>/<tr> để
    parse_sections() ở bước sau vẫn nhận diện được Điều/Chương theo dòng."""
    soup = BeautifulSoup(content_html, "lxml")

    # Loại bỏ script/style nếu có
    for tag in soup(["script", "style"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Dọn dòng trống thừa
    lines = [l.strip() for l in text.splitlines()]
    lines = [l for l in lines if l]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: METADATA VĂN BẢN — lấy trực tiếp từ cột dataset, không cần regex header
# ══════════════════════════════════════════════════════════════════════════════

def build_meta(row: dict, status: str, relationships_by_doc: dict, meta_df_by_id: dict) -> dict:
    meta = {
        "doc_id":          row["id"],
        "so_hieu":         row.get("so_ky_hieu"),
        "ten_luat":        row.get("title"),
        "co_quan_bh":      row.get("co_quan_ban_hanh"),
        "ngay_ban_hanh":   row.get("ngay_ban_hanh"),
        "ngay_hieu_luc":   row.get("ngay_co_hieu_luc"),
        "ngay_het_hieu_luc": row.get("ngay_het_hieu_luc") or None,
        "nguoi_ky":        row.get("nguoi_ky"),
        "chuc_danh_ky":    row.get("chuc_danh"),
        "loai_van_ban":    row.get("loai_van_ban"),
        "nganh":           row.get("nganh"),
        "linh_vuc":        row.get("linh_vuc"),
        "pham_vi":         row.get("pham_vi"),
        "tinh_trang_goc":  row.get("tinh_trang_hieu_luc"),
        "sua_doi_luat":    None,
        "status":          status,
    }

    # Làm giàu "sua_doi_luat" từ bảng quan hệ, nếu có bật INCLUDE_RELATIONSHIPS
    rels = relationships_by_doc.get(row["id"], [])
    if rels:
        interesting = [r for r in rels if any(
            kw in (r["relationship"] or "").lower()
            for kw in ("sửa đổi", "thay thế", "hết hiệu lực")
        )]
        if interesting:
            parts = []
            for r in interesting[:5]:
                other_title = meta_df_by_id.get(r["other_doc_id"], {}).get("title", r["other_doc_id"])
                parts.append(f"{r['relationship']}: {other_title}")
            meta["sua_doi_luat"] = " | ".join(parts)

    return meta


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: TÁCH CÁC PHẦN CẤU TRÚC (Điều, Chương, Mục, Phụ lục)
# — logic giữ nguyên như bản gốc, chạy trên plain text đã convert từ HTML —
# ══════════════════════════════════════════════════════════════════════════════

RE_DIEU    = re.compile(r"^(Điều\s+\d+[\.\:]?\s*.{0,120})$", re.IGNORECASE)
RE_CHUONG  = re.compile(r"^(Chương\s+[IVXLCDM\d]+[\.\:]?\s*.{0,120})$", re.IGNORECASE)
RE_MUC     = re.compile(r"^(Mục\s+\d+[\.\:]?\s*.{0,120})$", re.IGNORECASE)
RE_PHUCLUC = re.compile(r"^(PHỤ LỤC.{0,120})$", re.IGNORECASE)
RE_HEADER  = re.compile(
    r"^(?:QUỐC HỘI|CHÍNH PHỦ|CỘNG HÒA XÃ HỘI|Độc lập|________|\d{1,3}$|Số thứ tự|Mã số)",
    re.IGNORECASE
)


def parse_sections(text: str) -> list:
    lines = text.splitlines()
    sections = []
    current = {"section_type": "mo_dau", "heading": "Mở đầu", "content": []}
    current_chuong = ""
    current_muc = ""

    start_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if RE_HEADER.match(stripped) or len(stripped) < 3:
            continue
        if re.match(r"^Căn cứ", stripped, re.IGNORECASE) or RE_DIEU.match(stripped):
            start_idx = i
            break

    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped:
            current["content"].append("")
            continue

        if RE_CHUONG.match(stripped):
            current_chuong = stripped
            current_muc = ""
            current["content"].append(stripped)
            continue

        if RE_MUC.match(stripped):
            current_muc = stripped
            current["content"].append(stripped)
            continue

        if RE_DIEU.match(stripped):
            if current["content"]:
                sections.append({**current, "content": "\n".join(current["content"]).strip()})
            current = {
                "section_type": "dieu",
                "heading":      stripped,
                "chuong":       current_chuong,
                "muc":          current_muc,
                "content":      [],
            }
            continue

        if RE_PHUCLUC.match(stripped):
            if current["content"]:
                sections.append({**current, "content": "\n".join(current["content"]).strip()})
            current = {
                "section_type": "phu_luc",
                "heading":      stripped,
                "chuong":       "",
                "muc":          "",
                "content":      [],
            }
            continue

        current["content"].append(stripped)

    if current["content"]:
        sections.append({**current, "content": "\n".join(current["content"]).strip()})

    result = [s for s in sections if len(s["content"]) >= 20]
    return result


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 5: CHUNK — chia section dài thành các chunk nhỏ hơn (giữ nguyên bản gốc)
# ══════════════════════════════════════════════════════════════════════════════

def split_into_chunks(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break

        cut = -1
        for sep in ("\n", ". ", " "):
            pos = text.rfind(sep, start + overlap, end)
            if pos > start:
                cut = pos + len(sep)
                break
        if cut == -1:
            cut = end

        chunks.append(text[start:cut].strip())
        start = cut - overlap

    return [c for c in chunks if c]


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 6: TẠO CHUNK RECORDS đầy đủ metadata
# ══════════════════════════════════════════════════════════════════════════════

def make_chunks(sections: list, meta: dict) -> list:
    chunks = []
    for sec in sections:
        texts = split_into_chunks(sec["content"])
        for i, text in enumerate(texts):
            law_context = f"[{meta['loai_van_ban'] or 'VĂN BẢN'} {meta['so_hieu'] or ''}] {meta['ten_luat'] or ''}\n"
            if sec["section_type"] == "dieu":
                law_context += f"{sec.get('chuong','')}\n{sec.get('muc','')}\n{sec['heading']}\n".strip() + "\n"
            heading = sec.get("heading", "").strip()
            chunk = {
                "chunk_id": (
                    f"{meta['so_hieu']}"
                    f"__{sec['section_type']}"
                    f"__{hashlib.md5(heading.encode()).hexdigest()[:8]}"
                    f"__{i}"
                ),

                "text": law_context + text,
                "char_len": len(text),

                "section_type": sec["section_type"],
                "heading":      sec["heading"],
                "chuong":       sec.get("chuong", ""),
                "muc":          sec.get("muc", ""),
                "chunk_idx":    i,
                "total_chunks": len(texts),

                "doc_id":          meta["doc_id"],
                "so_hieu":         meta["so_hieu"],
                "ten_luat":        meta["ten_luat"],
                "loai_van_ban":    meta["loai_van_ban"],
                "co_quan_bh":      meta["co_quan_bh"],
                "ngay_ban_hanh":   meta["ngay_ban_hanh"],
                "ngay_hieu_luc":   meta["ngay_hieu_luc"],
                "ngay_het_hieu_luc": meta["ngay_het_hieu_luc"],
                "nguoi_ky":        meta["nguoi_ky"],
                "chuc_danh_ky":    meta["chuc_danh_ky"],
                "nganh":           meta["nganh"],
                "linh_vuc":        meta["linh_vuc"],
                "pham_vi":         meta["pham_vi"],
                "sua_doi_luat":    meta["sua_doi_luat"],

                "status":          meta["status"],
                "tinh_trang_goc":  meta["tinh_trang_goc"],
            }
            chunks.append(chunk)

    merged = []
    for chunk in chunks:
        if merged and chunk["char_len"] < 100 and chunk["section_type"] == merged[-1]["section_type"]:
            merged[-1]["text"]     += "\n" + chunk["text"]
            merged[-1]["char_len"] += chunk["char_len"]
            merged[-1]["total_chunks"] -= 1
        else:
            merged.append(chunk)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════════════════

def process_doc(row: dict, content_html: str, status: str,
                 relationships_by_doc: dict, meta_df_by_id: dict) -> list | None:
    try:
        text = html_to_text(content_html)
    except Exception as e:
        print(f"    ✗ Lỗi convert HTML (id={row['id']}): {e}")
        return None

    meta     = build_meta(row, status, relationships_by_doc, meta_df_by_id)
    sections = parse_sections(text)
    chunks   = make_chunks(sections, meta)
    return chunks


def main():
    output_file = OUTPUT_DIR / "rag_corpus.jsonl"
    stats_file  = OUTPUT_DIR / "_stats.json"

    print(f"\n{'='*60}")
    print(f"  Chunk Laws (HuggingFace: {DATASET_NAME}) → RAG Corpus")
    print(f"  Lọc loai_van_ban: {LOAI_VAN_BAN_FILTER or 'TẤT CẢ'}")
    print(f"  Output: {output_file}")
    print(f"{'='*60}\n")

    # 1) Metadata
    meta_df = load_metadata_df()
    meta_df = filter_metadata_df(meta_df)
    meta_df_by_id = meta_df.set_index("id").to_dict(orient="index")
    for rid, d in meta_df_by_id.items():
        d["id"] = rid
    target_ids = set(meta_df_by_id.keys())

    if not target_ids:
        print("✗ Không có văn bản nào khớp bộ lọc — dừng lại.")
        return

    # 2) Nội dung HTML (chỉ những id cần dùng)
    content_by_id = load_content_for_ids(target_ids)

    # 3) Quan hệ pháp lý (tuỳ chọn)
    relationships_by_doc = {}
    if INCLUDE_RELATIONSHIPS:
        relationships_by_doc = load_relationships_for_ids(target_ids)

    # 4) Xử lý & ghi chunk
    total_docs   = 0
    total_chunks = 0
    errors       = []
    no_content   = []
    stats_per_status = {}

    print(f"\n→ Xử lý {len(target_ids):,} văn bản ...")
    with output_file.open("w", encoding="utf-8") as out:
        for doc_id, row in tqdm(meta_df_by_id.items(), desc="  chunking"):
            content_html = content_by_id.get(doc_id)
            if not content_html:
                no_content.append(doc_id)
                continue

            status = map_status(row.get("tinh_trang_hieu_luc"))
            chunks = process_doc(row, content_html, status, relationships_by_doc, meta_df_by_id)

            if chunks is None:
                errors.append(doc_id)
                stats_per_status.setdefault(status, {"docs": 0, "chunks": 0, "errors": 0})
                stats_per_status[status]["errors"] += 1
                continue

            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            total_docs   += 1
            total_chunks += len(chunks)
            stats_per_status.setdefault(status, {"docs": 0, "chunks": 0, "errors": 0})
            stats_per_status[status]["docs"] += 1
            stats_per_status[status]["chunks"] += len(chunks)

    # 5) Ghi thống kê
    stats = {
        "generated_at":        time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dataset":      DATASET_NAME,
        "loai_van_ban_filter": LOAI_VAN_BAN_FILTER,
        "total_docs":          total_docs,
        "total_chunks":        total_chunks,
        "avg_chunks_per_doc":  round(total_chunks / total_docs, 1) if total_docs else 0,
        "docs_without_content": len(no_content),
        "max_chunk_chars":     MAX_CHUNK_CHARS,
        "overlap_chars":       OVERLAP_CHARS,
        "by_status":           stats_per_status,
        "errors":              errors,
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  ✅ Xong: {total_chunks:,} chunks từ {total_docs:,} văn bản")
    print(f"  ⚠ {len(no_content):,} văn bản không có content HTML (chỉ có bản scan PDF)")
    print(f"  Output: {output_file}")
    if errors:
        print(f"  ⚠ {len(errors)} văn bản lỗi — xem {stats_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()