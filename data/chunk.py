"""
chunk_laws_uts_v2_fixed.py
Corrected version - parses Vietnamese legal documents into RAG-ready,
ChromaDB-compatible chunks.

Fixes applied (see accompanying explanation):
 1. Added missing imports (hashlib, json, time, Counter) at the top.
 2. Added missing MAX_CHUNK_CHARS constant.
 3. Added missing build_metadata() function, with FLAT metadata
    (str/int/float only) so it can be inserted directly into ChromaDB
    (nested dict/list values are not supported by Chroma metadata).
 4. Clause/Point dataclasses now have safe defaults and are built with
    an accumulating string instead of a list-that-was-never-joined.
 5. Removed the buggy "join only after the whole loop ends" code that
    silently corrupted every clause/point except the very last one
    parsed. Content is now finalized correctly, clause by clause.
 6. Fixed Point handling (`point.text` did not exist -> AttributeError).
 7. Fixed a possible infinite loop in split_soft() when a newline/period
    was found very close to `start`.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

# ============================================================
# CONFIG
# ============================================================
DATASET_NAME = "undertheseanlp/UTS_VLC"
SPLIT = "2026"

# Anchor to this script's location, not the current working directory, so
# the output folder is always in the same place regardless of where/how
# you launch the script (terminal, IDE run button, etc.).
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "rag_chunks_uts_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_DOCS = None

# Max characters for an article to be kept as a single chunk before
# it gets split at clause/point level.
MAX_CHUNK_CHARS = 1200

TYPE_MAP = {
    "constitution": "Hiến pháp",
    "code": "Bộ luật",
    "law": "Luật",
}
STATUS_MAP = {
    "2026": "verified",
    "2026_01": "unverified",
    "2023": "historical",
    "2021": "historical",
}

# ============================================================
# REGEX
# ============================================================
RE_CHAPTER = re.compile(r"^Chương\s+([IVXLCDM0-9]+)\.?\s*(.*)$", re.IGNORECASE)
RE_SECTION = re.compile(r"^Mục\s+(\d+)\.?\s*(.*)$", re.IGNORECASE)
RE_ARTICLE = re.compile(r"^Điều\s+(\d+)\.?\s*(.*)$", re.IGNORECASE)
RE_CLAUSE = re.compile(r"^(\d+)\.\s*(.*)$")
RE_POINT = re.compile(r"^([a-zđ])\)\s*(.*)$", re.IGNORECASE)
RE_HEADER = re.compile(r"^\*\*(.+?)\:\*\*\s*(.*)$")

# ============================================================
# DATA MODEL
# ============================================================
@dataclass
class Point:
    point: str
    content: str = ""


@dataclass
class Clause:
    number: int
    title: str
    content: str = ""
    points: List[Point] = field(default_factory=list)


@dataclass
class Article:
    number: int
    title: str
    chapter: str = ""
    section: str = ""
    intro: List[str] = field(default_factory=list)
    clauses: List[Clause] = field(default_factory=list)


@dataclass
class LawDocument:
    law_code: str
    law_name: str
    document_type: str
    effective_date: str = ""
    issue_date: str = ""
    english_name: str = ""
    status: str = ""
    articles: List[Article] = field(default_factory=list)


# ============================================================
# HEADER
# ============================================================
HEADER_MAP = {
    "số hiệu": "law_code",
    "ngày hiệu lực": "effective_date",
    "ngày ban hành": "issue_date",
    "english": "english_name",
}


def split_header(content: str):
    parts = content.split("\n---\n", 1)
    if len(parts) == 1:
        return {}, content
    header, body = parts
    meta = {}
    for line in header.splitlines():
        m = RE_HEADER.match(line.strip())
        if not m:
            continue
        key = m.group(1).strip().lower()
        key = HEADER_MAP.get(key, key)
        meta[key] = m.group(2).strip()
    return meta, body


# ============================================================
# MARKDOWN CLEANER
# ============================================================
def markdown_to_text(md: str):
    md = re.sub(r"^#{1,6}\s*", "", md, flags=re.MULTILINE)
    md = re.sub(r"\*\*(.*?)\*\*", r"\1", md)
    md = re.sub(r"\*(.*?)\*", r"\1", md)
    md = re.sub(r"\[(.*?)\]\((.*?)\)", r"\1", md)
    md = re.sub(r"^\s*[-*+]\s+", "", md, flags=re.MULTILINE)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ============================================================
# PARSER
# ============================================================
def _append_line(existing: str, line: str) -> str:
    """Append a line to an accumulating text field, joined with \n."""
    return f"{existing}\n{line}".strip() if existing else line


def parse_articles(text: str) -> List[Article]:
    articles: List[Article] = []
    article: Optional[Article] = None
    clause: Optional[Clause] = None
    point: Optional[Point] = None
    current_chapter = ""
    current_section = ""

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        m = RE_CHAPTER.match(line)
        if m:
            current_chapter = line
            current_section = ""
            continue

        m = RE_SECTION.match(line)
        if m:
            current_section = line
            continue

        m = RE_ARTICLE.match(line)
        if m:
            if article:
                if point:
                    clause.points.append(point)
                    point = None
                if clause:
                    article.clauses.append(clause)
                    clause = None
                articles.append(article)
            article = Article(
                number=int(m.group(1)),
                title=m.group(2),
                chapter=current_chapter,
                section=current_section,
            )
            continue

        if article is None:
            continue

        m = RE_CLAUSE.match(line)
        if m:
            if point:
                clause.points.append(point)
                point = None
            if clause:
                article.clauses.append(clause)
            clause = Clause(
                number=int(m.group(1)),
                title=m.group(2),
            )
            continue

        m = RE_POINT.match(line)
        if m and clause:
            if point:
                clause.points.append(point)
            point = Point(point=m.group(1))
            point.content = m.group(2).strip()
            continue

        if point:
            point.content = _append_line(point.content, line)
        elif clause:
            clause.content = _append_line(clause.content, line)
        else:
            article.intro.append(line)

    # Flush whatever is left open at end of document.
    if article:
        if point:
            clause.points.append(point)
            point = None
        if clause:
            article.clauses.append(clause)
            clause = None
        articles.append(article)

    return articles


# ============================================================
# DOCUMENT PARSER
# ============================================================
def parse_document(row) -> LawDocument:
    header, body = split_header(row["content"])
    body = markdown_to_text(body)
    return LawDocument(
        law_code=row["id"],
        law_name=row["title"],
        document_type=TYPE_MAP.get(row["type"], row["type"]),
        effective_date=header.get("effective_date", ""),
        issue_date=header.get("issue_date", ""),
        english_name=header.get("english_name", ""),
        status=STATUS_MAP.get(SPLIT, ""),
        articles=parse_articles(body),
    )


# ============================================================
# DATASET
# ============================================================
def load_laws():
    ds = load_dataset(DATASET_NAME, split=SPLIT)
    if MAX_DOCS:
        ds = ds.select(range(min(MAX_DOCS, len(ds))))
    return ds


@dataclass
class Chunk:
    chunk_id: str
    text: str
    embedding_text: str
    metadata: dict


LEGAL_KEYWORDS = [
    "đặt cọc",
    "hợp đồng",
    "phạt cọc",
    "nghĩa vụ",
    "bồi thường",
    "quyền sở hữu",
    "quyền sử dụng đất",
    "tài sản",
    "vợ chồng",
    "thừa kế",
    "thế chấp",
    "bảo lãnh",
    "đăng ký",
]


def extract_keywords(text: str) -> List[str]:
    text = text.lower()
    result = [kw for kw in LEGAL_KEYWORDS if kw in text]
    return sorted(set(result))


def build_citation(law, article, clause=None, point=None) -> dict:
    citation = {
        "law_name": law.law_name,
        "law_code": law.law_code,
        "article": article.number,
        "article_title": article.title,
    }
    if clause:
        citation["clause"] = clause.number
    if point:
        citation["point"] = point.point
    return citation


def build_embedding_text(law, article, body, clause=None) -> str:
    parts = [
        f"Văn bản: {law.law_name}",
        f"Số hiệu: {law.law_code}",
        f"Loại: {law.document_type}",
    ]
    if article.chapter:
        parts.append(article.chapter)
    if article.section:
        parts.append(article.section)
    parts.append(f"Điều {article.number}. {article.title}")
    if clause:
        parts.append(f"Khoản {clause.number}")
    parts.append("Nội dung:")
    parts.append(body)
    return "\n".join(parts)


def build_metadata(
    law: LawDocument,
    article: Article,
    clause: Optional[Clause],
    point: Optional[Point],
    idx: int,
    total: int,
    chunk_text: str,
) -> dict:
    """
    Build FLAT metadata for the chunk.

    ChromaDB metadata values must be str/int/float/bool - never a dict
    or a list. Nested structures (like the citation) are serialized to
    a JSON string; list values (keywords) are joined into a string.
    None values are replaced with safe defaults (0 / "") since Chroma
    rejects None as a metadata value.
    """
    citation = build_citation(law, article, clause, point)
    return {
        "law_code": law.law_code,
        "law_name": law.law_name,
        "document_type": law.document_type,
        "effective_date": law.effective_date,
        "issue_date": law.issue_date,
        "english_name": law.english_name,
        "status": law.status,
        "chapter": article.chapter,
        "section": article.section,
        "article_number": article.number,
        "article_title": article.title,
        "clause_number": clause.number if clause else 0,
        "clause_title": clause.title if clause else "",
        "point": point.point if point else "",
        "chunk_index": idx,
        "chunk_total": total,
        "keywords": ", ".join(extract_keywords(chunk_text)),
        "citation_json": json.dumps(citation, ensure_ascii=False),
    }


def split_soft(text: str, max_chars: int = 900, overlap: int = 120) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        if end == n:
            piece = text[start:].strip()
            if piece:
                chunks.append(piece)
            break

        cut = text.rfind("\n", start, end)
        if cut <= start:
            cut = text.rfind(". ", start, end)
        if cut <= start:
            cut = end

        piece = text[start:cut].strip()
        if piece:
            chunks.append(piece)

        # Guarantee forward progress even if `cut - overlap` would land
        # at/behind the current start (this used to cause an infinite loop).
        next_start = cut - overlap
        if next_start <= start:
            next_start = cut
        start = next_start

    return chunks


def make_chunk_id(parts: List[Any], seen_keys: Dict[str, int]) -> str:
    """
    Build a stable, collision-safe chunk_id.

    `parts` should uniquely identify a chunk's *position* in the corpus
    (split, law_code, article, clause, point, idx). We include SPLIT so
    that if you ever load multiple splits/versions of the same law into
    one Chroma collection, their ids don't collide and silently overwrite
    each other via upsert.

    On top of that, `seen_keys` guards against the rarer case where the
    parser itself produces two structurally-identical keys for genuinely
    different content (e.g. a stray numbered list mis-detected as a
    "khoản" duplicates a real khoản number). Instead of silently colliding,
    we detect it and append a disambiguating suffix, so BOTH chunks are
    kept - and we can report how often this happened.
    """
    base_key = "|".join(str(p) for p in parts)
    key = base_key
    if key in seen_keys:
        seen_keys[key] += 1
        key = f"{base_key}|dup{seen_keys[base_key]}"
    else:
        seen_keys[key] = 1
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


# ============================================================
# BUILD CHUNKS
# ============================================================
def build_chunks(law: LawDocument, seen_keys: Dict[str, int]) -> List[Chunk]:
    chunks: List[Chunk] = []

    for article in law.articles:
        # ----------------------------------------------------
        # Build full article text
        # ----------------------------------------------------
        article_parts = []
        if article.intro:
            article_parts.append("\n".join(article.intro))
        for clause in article.clauses:
            clause_text = []
            if clause.title:
                clause_text.append(clause.title)
            if clause.content:
                clause_text.append(clause.content)
            for point in clause.points:
                clause_text.append(f"{point.point}) {point.content}")
            article_parts.append("\n".join(clause_text))
        article_body = "\n".join(article_parts).strip()

        # ----------------------------------------------------
        # Điều ngắn (short article -> single chunk)
        # ----------------------------------------------------
        if len(article_body) <= MAX_CHUNK_CHARS:
            metadata = build_metadata(law, article, None, None, 0, 1, article_body)
            embedding_text = build_embedding_text(law, article, article_body)
            chunk_id = make_chunk_id(
                [SPLIT, law.law_code, "article", article.number],
                seen_keys,
            )
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    text=article_body,
                    embedding_text=embedding_text,
                    metadata=metadata,
                )
            )
            continue

        # ----------------------------------------------------
        # Điều dài (long article -> split by clause / point)
        # ----------------------------------------------------
        for clause in article.clauses:
            clause_lines = []
            if clause.title:
                clause_lines.append(clause.title)
            if clause.content:
                clause_lines.append(clause.content)

            # =================================================
            # Khoản không có điểm
            # =================================================
            if not clause.points:
                clause_body = "\n".join(clause_lines).strip()
                pieces = split_soft(clause_body)
                for idx, piece in enumerate(pieces):
                    metadata = build_metadata(
                        law, article, clause, None, idx, len(pieces), piece
                    )
                    embedding_text = build_embedding_text(
                        law, article, piece, clause
                    )
                    chunk_id = make_chunk_id(
                        [
                            SPLIT,
                            law.law_code,
                            "article", article.number,
                            "clause", clause.number,
                            "idx", idx,
                        ],
                        seen_keys,
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=piece,
                            embedding_text=embedding_text,
                            metadata=metadata,
                        )
                    )
                continue

            # =================================================
            # Khoản có điểm
            # =================================================
            for point in clause.points:
                point_body = point.content.strip()
                pieces = split_soft(point_body)
                for idx, piece in enumerate(pieces):
                    metadata = build_metadata(
                        law, article, clause, point, idx, len(pieces), piece
                    )
                    embedding_text = build_embedding_text(
                        law, article, piece, clause
                    )
                    chunk_id = make_chunk_id(
                        [
                            SPLIT,
                            law.law_code,
                            "article", article.number,
                            "clause", clause.number,
                            "point", point.point,
                            "idx", idx,
                        ],
                        seen_keys,
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=piece,
                            embedding_text=embedding_text,
                            metadata=metadata,
                        )
                    )

    return chunks


# ============================================================
# EXPORT
# ============================================================
def export_chunk(fp, chunk: Chunk) -> None:
    record = {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "embedding_text": chunk.embedding_text,
        **chunk.metadata,
    }
    fp.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# PIPELINE
# ============================================================
def process_dataset():
    output_jsonl = OUTPUT_DIR / "rag_corpus.jsonl"
    stats_path = OUTPUT_DIR / "_stats.json"

    dataset = load_laws()

    total_docs = 0
    total_chunks = 0
    total_articles = 0
    total_clauses = 0
    total_points = 0
    chunk_per_doc = []
    chunk_per_type = Counter()
    article_per_type = Counter()
    errors = []

    # Shared across the ENTIRE run (not per-document), so we also catch
    # duplicate keys caused by two different rows sharing the same law_code.
    seen_keys: Dict[str, int] = {}

    with output_jsonl.open("w", encoding="utf-8") as fout:
        for row in dataset:
            try:
                law = parse_document(row)
                chunks = build_chunks(law, seen_keys)
                for chunk in chunks:
                    export_chunk(fout, chunk)

                total_docs += 1
                total_chunks += len(chunks)
                chunk_per_doc.append(len(chunks))

                article_count = len(law.articles)
                total_articles += article_count
                article_per_type[law.document_type] += article_count
                chunk_per_type[law.document_type] += len(chunks)

                for article in law.articles:
                    total_clauses += len(article.clauses)
                    for clause in article.clauses:
                        total_points += len(clause.points)
            except Exception as e:
                errors.append({"law": row.get("id"), "error": str(e)})
                continue

    stats = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": DATASET_NAME,
        "split": SPLIT,
        "documents": total_docs,
        "articles": total_articles,
        "clauses": total_clauses,
        "points": total_points,
        "chunks": total_chunks,
        "avg_chunk_per_doc": round(sum(chunk_per_doc) / len(chunk_per_doc), 2)
        if chunk_per_doc
        else 0,
        "chunk_by_document_type": dict(chunk_per_type),
        "article_by_document_type": dict(article_per_type),
        "errors": len(errors),
        # How many chunk keys collided and had to be disambiguated with a
        # "|dupN" suffix. Should normally be 0 - a nonzero value usually
        # means the source text has malformed numbering (e.g. a stray
        # numbered list mis-parsed as a "khoản") and is worth investigating.
        "duplicate_keys_resolved": sum(v - 1 for v in seen_keys.values() if v > 1),
    }

    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    if errors:
        error_path = OUTPUT_DIR / "_errors.json"
        with error_path.open("w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print("Chunk generation completed")
    print(f"Documents : {total_docs:,}")
    print(f"Articles  : {total_articles:,}")
    print(f"Clauses   : {total_clauses:,}")
    print(f"Points    : {total_points:,}")
    print(f"Chunks    : {total_chunks:,}")
    print(f"Errors    : {len(errors)}")
    print("=" * 60)


# ============================================================
# MAIN
# ============================================================
def main():
    process_dataset()


if __name__ == "__main__":
    main()