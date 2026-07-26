"""
chunk_laws_uts_vlc.py — Chia chunk văn bản pháp luật từ dataset HuggingFace
                         undertheseanlp/UTS_VLC cho RAG

Dataset nguồn: https://huggingface.co/datasets/undertheseanlp/UTS_VLC
    - Chỉ 942 dòng tổng cộng, chia 4 split: 2021 (110), 2023 (208),
      2026_01 (318, bản gốc có lỗi trùng lặp), 2026 (306, BẢN ĐÃ SỬA — khuyên dùng)
    - Split "2026" = snapshot ĐÃ XÁC THỰC đang còn hiệu lực (Hiến pháp 2013 +
      6 Bộ luật + 299 Luật), đối chiếu chéo với vbpl.vn và 5 nguồn khác
      (Công báo Chính phủ, vanban.chinhphu.vn, vietlaw.quochoi.vn,
      thuvienphapluat.vn, luatvietnam.vn), đã khử trùng theo Số hiệu.
    - CHỈ có Hiến pháp/Luật/Bộ luật (cấp 1-2) — KHÔNG có Nghị định, Thông tư,
      Quyết định... Nếu RAG cần cả văn bản dưới luật, phải ghép thêm nguồn khác
      (vd. script chunk_laws_hf.py dùng vohuutridung/vietnamese-legal-documents).
    - Chỉ 1 config duy nhất ("default"), mỗi dòng đã có sẵn "content" là
      Markdown sạch — KHÔNG cần đọc pyarrow / join nhiều bảng như 2 dataset
      trước (th1nhng0 / vohuutridung), nên script này đơn giản hơn nhiều.

Output:
    data/rag_chunks_uts_vlc/
        rag_corpus.jsonl  — mỗi dòng là 1 chunk, sẵn để embed
        _stats.json       — thống kê tổng quan

Cài đặt:
    pip install datasets pandas tqdm
"""

import hashlib
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# ─── CẤU HÌNH ─────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("data/rag_chunks_uts_vlc")
MAX_CHUNK_CHARS = 1500   # Độ dài tối đa mỗi chunk (ký tự)
OVERLAP_CHARS   = 150    # Overlap giữa các chunk

DATASET_NAME = "undertheseanlp/UTS_VLC"

# Split nào để dùng. Khuyên dùng "2026" (bản đã xác thực đang hiệu lực).
# Đặt "2026_01" / "2023" / "2021" chỉ khi cần tái lập bản gốc / dữ liệu lịch sử
# (những split này CÓ CHỨA văn bản đã hết hiệu lực toàn bộ, không dùng cho
# tư vấn pháp luật hiện hành).
SPLIT_NAME = "2026"

# Giới hạn số văn bản xử lý (debug nhanh). Đặt None để chạy full.
MAX_DOCS = None

# Map cột `type` của dataset → nhãn loại văn bản tiếng Việt dùng trong chunk.
TYPE_MAP = {
    "constitution": "Hiến pháp",
    "code":         "Bộ luật",
    "law":          "Luật",
}

# Trạng thái hiệu lực gán theo split — dataset KHÔNG có cột status riêng cho
# từng văn bản, nhưng theo dataset card, split "2026" đã được lọc chỉ giữ
# "Còn hiệu lực" ∪ "Hết hiệu lực một phần" (tức là còn giá trị áp dụng), nên
# ta gán nhãn tổng quát tương ứng thay vì "khong_xac_dinh" như 2 dataset kia.
SPLIT_STATUS_LABEL = {
    "2026":    "hieu_luc_da_xac_thuc",       # verified in-force (2026-06 snapshot)
    "2026_01": "hieu_luc_chua_xac_thuc",     # bản gốc có lỗi trùng lặp/bịa, chưa nên dùng
    "2023":    "snapshot_lich_su_2023",      # có thể chứa VB đã hết hiệu lực
    "2021":    "snapshot_lich_su_2021",      # có thể chứa VB đã hết hiệu lực
}
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: TÁCH HEADER (block "**Key:** value" đầu file) KHỎI THÂN VĂN BẢN
# ══════════════════════════════════════════════════════════════════════════════
# Mỗi `content` có dạng:
#   # <Tiêu đề>
#
#   **Số hiệu:** 91/2015/QH13
#   **English:** ...          (tuỳ văn bản, có thể không có)
#   **Ngày hiệu lực:** ...    (tuỳ văn bản, có thể không có)
#
#   ---
#
#   <thân văn bản luật, bắt đầu bằng "BỘ LUẬT / DÂN SỰ / Căn cứ ...">

_HEADER_FIELD_RE = re.compile(r"^\*\*(.+?):\*\*\s*(.*)$")
_HEADER_KEY_MAP = {
    "số hiệu":       "so_hieu_header",
    "ngày hiệu lực": "ngay_hieu_luc",
    "ngày ban hành": "ngay_ban_hanh",
    "english":       "ten_tieng_anh",
}


def split_header_and_body(content: str) -> tuple[dict, str]:
    """Trả về (dict các field trong header, phần thân văn bản còn lại)."""
    parts = content.split("\n---\n", 1)
    if len(parts) == 1:
        # Không tìm thấy dòng phân cách "---" — coi như không có header,
        # toàn bộ content là thân văn bản.
        return {}, content

    header_block, body = parts[0], parts[1]
    extra = {}
    for line in header_block.splitlines():
        m = _HEADER_FIELD_RE.match(line.strip())
        if not m:
            continue
        key_raw, value = m.group(1).strip().lower(), m.group(2).strip()
        key = _HEADER_KEY_MAP.get(key_raw, key_raw.replace(" ", "_"))
        if value:
            extra[key] = value
    return extra, body.strip()


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: MARKDOWN → PLAIN TEXT (làm sạch cú pháp markdown trong thân văn bản)
# ══════════════════════════════════════════════════════════════════════════════

_MD_HEADER_RE = re.compile(r"^#{1,6}\s*")
_MD_BOLD_RE   = re.compile(r"\*\*(.+?)\*\*")
_MD_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_MD_LINK_RE   = re.compile(r"\[([^\]]*)\]\(([^)]*)\)")
_MD_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
_MD_HR_RE     = re.compile(r"^\s*-{3,}\s*$")


def markdown_to_text(body_md: str) -> str:
    lines = body_md.splitlines()
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line or _MD_HR_RE.match(line):
            continue
        line = _MD_HEADER_RE.sub("", line)
        line = _MD_LINK_RE.sub(r"\1", line)
        line = _MD_BOLD_RE.sub(r"\1", line)
        line = _MD_ITALIC_RE.sub(r"\1", line)
        line = _MD_BULLET_RE.sub("", line)
        line = line.strip()
        if line:
            cleaned.append(line)
    return "\n".join(cleaned)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: TÁCH CÁC PHẦN CẤU TRÚC (Điều, Chương, Mục, Phụ lục)
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
# BƯỚC 4: CHUNK — chia section dài thành các chunk nhỏ hơn
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
# BƯỚC 5: TẠO CHUNK RECORDS đầy đủ metadata
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

                "doc_id":        meta["doc_id"],
                "so_hieu":       meta["so_hieu"],
                "ten_luat":      meta["ten_luat"],
                "loai_van_ban":  meta["loai_van_ban"],
                "ngay_ban_hanh": meta["ngay_ban_hanh"],
                "ngay_hieu_luc": meta["ngay_hieu_luc"],
                "ten_tieng_anh": meta["ten_tieng_anh"],

                "status":        meta["status"],
                "split_nguon":   meta["split_nguon"],
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

def process_doc(row: dict) -> list | None:
    try:
        header_fields, body_md = split_header_and_body(row["content"])
        text = markdown_to_text(body_md)
    except Exception as e:
        print(f"    ✗ Lỗi xử lý (id={row.get('id')}): {e}")
        return None

    meta = {
        "doc_id":        row.get("id"),
        "so_hieu":       row.get("id"),  # cột id CHÍNH LÀ số hiệu (vd "91/2015/QH13")
        "ten_luat":      row.get("title"),
        "loai_van_ban":  TYPE_MAP.get(row.get("type"), row.get("type")),
        "ngay_ban_hanh": header_fields.get("ngay_ban_hanh"),
        "ngay_hieu_luc": header_fields.get("ngay_hieu_luc"),
        "ten_tieng_anh": header_fields.get("ten_tieng_anh"),
        "status":        SPLIT_STATUS_LABEL.get(SPLIT_NAME, "khong_xac_dinh"),
        "split_nguon":   SPLIT_NAME,
    }

    sections = parse_sections(text)
    return make_chunks(sections, meta)


def main():
    output_file = OUTPUT_DIR / "rag_corpus.jsonl"
    stats_file  = OUTPUT_DIR / "_stats.json"

    print(f"\n{'='*60}")
    print(f"  Chunk Laws (HuggingFace: {DATASET_NAME}, split={SPLIT_NAME}) → RAG Corpus")
    print(f"  Output: {output_file}")
    print(f"{'='*60}\n")

    if SPLIT_NAME != "2026":
        print(f"⚠ Cảnh báo: split '{SPLIT_NAME}' KHÔNG phải bản đã xác thực đang hiệu lực. "
              f"Có thể chứa văn bản đã hết hiệu lực toàn bộ. Dùng split='2026' cho luật hiện hành.\n")

    print(f"→ Tải split '{SPLIT_NAME}' ...")
    ds = load_dataset(DATASET_NAME, split=SPLIT_NAME)
    if MAX_DOCS:
        ds = ds.select(range(min(MAX_DOCS, len(ds))))
    print(f"  {len(ds):,} văn bản\n")

    total_docs   = 0
    total_chunks = 0
    errors       = []
    stats_by_type = {}

    with output_file.open("w", encoding="utf-8") as out:
        for row in tqdm(ds, desc="  chunking"):
            chunks = process_doc(row)
            if chunks is None:
                errors.append(row.get("id"))
                continue
            if not chunks:
                continue

            for chunk in chunks:
                out.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            total_docs   += 1
            total_chunks += len(chunks)
            loai = row.get("type", "khong_ro")
            stats_by_type.setdefault(loai, {"docs": 0, "chunks": 0})
            stats_by_type[loai]["docs"] += 1
            stats_by_type[loai]["chunks"] += len(chunks)

    stats = {
        "generated_at":       time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dataset":     DATASET_NAME,
        "split":              SPLIT_NAME,
        "total_docs":         total_docs,
        "total_chunks":       total_chunks,
        "avg_chunks_per_doc": round(total_chunks / total_docs, 1) if total_docs else 0,
        "max_chunk_chars":    MAX_CHUNK_CHARS,
        "overlap_chars":      OVERLAP_CHARS,
        "by_type":            stats_by_type,
        "errors":             errors,
        "note": (
            "Dataset undertheseanlp/UTS_VLC chỉ bao phủ Hiến pháp/Luật/Bộ luật "
            "(cấp 1-2 trong hệ thống VBQPPL) — không có Nghị định/Thông tư/Quyết định. "
            "Luật sửa đổi bổ sung được để riêng, chưa hợp nhất vào luật gốc."
        ),
    }
    stats_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  ✅ Xong: {total_chunks:,} chunks từ {total_docs:,} văn bản")
    print(f"  Output: {output_file}")
    if errors:
        print(f"  ⚠ {len(errors)} văn bản lỗi — xem {stats_file}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()