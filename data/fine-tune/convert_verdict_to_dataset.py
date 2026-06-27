"""
convert_verdict_to_dataset.py
==============================
Chuyển bản án PDF/text thực tế → dataset fine-tune dạng luật sư tư vấn.

Mỗi bản án sinh ra nhiều góc nhìn khác nhau:
  1. Người ở vị trí nguyên đơn hỏi trước khi kiện
  2. Người ở vị trí bị đơn hỏi khi bị kiện
  3. Phân tích chứng cứ: cái gì quyết định thắng/thua
  4. Lộ trình hành động nếu ở vị trí tương tự
  5. Bài học rút ra từ sai lầm của các bên
  6. Câu hỏi pháp lý cụ thể từ tình tiết vụ án

Cách chạy:
  pip install anthropic pymupdf tqdm
  export ANTHROPIC_API_KEY=sk-ant-...

  # Xử lý 1 file
  python convert_verdict_to_dataset.py --input banan.pdf

  # Xử lý toàn bộ folder
  python convert_verdict_to_dataset.py --folder ./verdicts/

  # Xem prompt mẫu, không gọi API
  python convert_verdict_to_dataset.py --input banan.pdf --dry-run
"""

import argparse
import json
import time
from pathlib import Path

import anthropic

# ── Cấu hình ────────────────────────────────────────────────────────────────
OUTPUT_DIR  = Path("./dataset")
OUTPUT_FILE = OUTPUT_DIR / "verdict_dataset.jsonl"
MODEL       = "claude-sonnet-4-6"
MAX_TOKENS  = 8000
# ─────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPTS = {
    "layperson": (
        "Bạn là luật sư tư vấn pháp luật Việt Nam với 15 năm kinh nghiệm xử lý "
        "tranh chấp dân sự, đất đai, lao động. Bạn đang nói chuyện với người dân "
        "bình thường. Dùng ngôn ngữ đơn giản, ví von dễ hiểu. Ưu tiên lời khuyên "
        "thực tế 'cần làm gì ngay' thay vì giảng luật. Nói thật dù có khó nghe."
    ),
    "professional": (
        "Bạn là luật sư tư vấn pháp luật Việt Nam với 15 năm kinh nghiệm tranh tụng. "
        "Tư vấn cho người có hiểu biết pháp lý cơ bản. Dùng thuật ngữ pháp lý chuẩn, "
        "trích dẫn điều khoản cụ thể, phân tích rủi ro định lượng. "
        "Đánh giá thực tế, không né tránh."
    ),
}


# ── 7 góc nhìn sinh dataset từ mỗi bản án ───────────────────────────────────

PERSPECTIVES = [

    {
        "id": "plaintiff_before",
        "user_type": "layperson",
        "description": "Người dùng ở vị trí nguyên đơn, hỏi TRƯỚC khi khởi kiện",
        "prompt": """
Dựa trên bản án sau, hãy tạo 1 cặp hỏi-đáp:
- Người hỏi đang ở VỊ TRÍ NGUYÊN ĐƠN, chưa kiện, đang hỏi "tôi có nên kiện không?"
- Câu hỏi phải tự nhiên như người dân thật sự hỏi (không biết luật)
- Câu trả lời: luật sư phân tích vị thế, khuyên có nên kiện không, rủi ro gì

Câu trả lời PHẢI có cấu trúc:
1. Nhận định ngay: vị thế nguyên đơn mạnh hay yếu ở điểm nào
2. Căn cứ pháp lý chính (1-2 điều luật quan trọng nhất)
3. Lời khuyên: nên làm gì ngay bây giờ
4. Rủi ro cần biết
5. Xác suất thắng ước tính (dựa trên kết quả thực tế của bản án này)

BẢN ÁN:
{verdict_text}

Trả về JSON:
{{
  "user": "<câu hỏi tự nhiên của người dân>",
  "assistant": "<câu trả lời luật sư chi tiết, có cấu trúc 5 phần>",
  "metadata": {{
    "perspective": "plaintiff_before",
    "user_type": "layperson",
    "domain": "<lĩnh vực: dat_dai/dan_su/lao_dong/boi_thuong>",
    "verdict_outcome": "<nguyên đơn thắng/thua/thắng một phần>",
    "key_law": "<điều luật quan trọng nhất>"
  }}
}}
""",
    },

    {
        "id": "defendant_response",
        "user_type": "layperson",
        "description": "Người dùng ở vị trí bị đơn, vừa nhận được đơn kiện",
        "prompt": """
Dựa trên bản án sau, tạo 1 cặp hỏi-đáp:
- Người hỏi ở VỊ TRÍ BỊ ĐƠN, vừa nhận giấy triệu tập tòa, hoảng loạn
- Câu hỏi: "Tôi bị kiện rồi, phải làm gì bây giờ?"
- Câu trả lời: luật sư bình tĩnh hóa, phân tích vị thế bị đơn, chiến lược phòng thủ

Câu trả lời phải:
1. Trấn an và định hướng ngay
2. Phân tích điểm mạnh/yếu của bị đơn trong tình huống này
3. Những việc CẦN LÀM NGAY (thu thập chứng cứ gì?)
4. Những việc TUYỆT ĐỐI KHÔNG làm
5. Chiến lược: nên thương lượng hay ra tòa? Tại sao?

BẢN ÁN:
{verdict_text}

Trả về JSON với format tương tự, "perspective": "defendant_response"
""",
    },

    {
        "id": "evidence_analysis",
        "user_type": "professional",
        "description": "Phân tích chứng cứ: cái gì quyết định thắng thua",
        "prompt": """
Dựa trên bản án sau, tạo 1 cặp hỏi-đáp CHUYÊN SÂU về chứng cứ:
- Người hỏi: luật sư/người có hiểu biết pháp lý, hỏi "Điều gì thực sự quyết định kết quả vụ này?"
- Câu trả lời: phân tích pháp lý sâu, focus vào chứng cứ và lập luận

Câu trả lời phải phân tích:
1. Chứng cứ THEN CHỐT quyết định kết quả (cụ thể từng chứng cứ)
2. Lập luận nào của mỗi bên mạnh/yếu và tại sao
3. Điều luật được áp dụng và lý do chọn điều luật đó
4. Sai lầm pháp lý (nếu có) của mỗi bên
5. Bài học: nếu làm lại, nên chuẩn bị khác thế nào?

BẢN ÁN:
{verdict_text}

Trả về JSON, "perspective": "evidence_analysis", "user_type": "professional"
""",
    },

    {
        "id": "action_roadmap",
        "user_type": "layperson",
        "description": "Người có tình huống tương tự hỏi lộ trình hành động",
        "prompt": """
Dựa trên bản án sau, tạo 1 cặp hỏi-đáp về LỘ TRÌNH HÀNH ĐỘNG:
- Người hỏi: đang có tình huống TƯƠNG TỰ nguyên đơn trong bản án, hỏi "tôi phải làm gì từng bước?"
- KHÔNG được nhắc đến bản án — như thể đây là tư vấn độc lập

Câu trả lời phải là lộ trình cụ thể theo thời gian:
- Ngay hôm nay: làm gì? Thu thập gì?
- Tuần này: gửi văn bản gì, đến đâu?
- Nếu bên kia không hợp tác: bước tiếp theo?
- Khởi kiện: tòa nào? Hồ sơ gồm gì? Mất bao lâu? Chi phí sơ bộ?
- Thời hiệu: có hạn nộp đơn không? Bao lâu?

BẢN ÁN:
{verdict_text}

Trả về JSON, "perspective": "action_roadmap"
""",
    },

    {
        "id": "negotiation_strategy",
        "user_type": "layperson",
        "description": "Chiến lược thương lượng trước khi ra tòa",
        "prompt": """
Dựa trên bản án sau, tạo 1 cặp hỏi-đáp về THƯƠNG LƯỢNG:
- Người hỏi: đang ở vị trí nguyên đơn, bên kia vừa đề nghị thương lượng, hỏi "chấp nhận bao nhiêu là hợp lý?"
- Câu hỏi tự nhiên, không biết luật

Câu trả lời phải:
1. Đánh giá vị thế thương lượng (mạnh/yếu dựa trên chứng cứ thực tế)
2. Mức yêu cầu tối đa hợp lý
3. Mức chấp nhận tối thiểu (không nên xuống dưới vì sao)
4. Điều khoản không được bỏ qua trong thỏa thuận
5. Dấu hiệu bên kia câu giờ → nên dừng thương lượng và kiện
6. Cách ghi nhận thỏa thuận có giá trị pháp lý

BẢN ÁN:
{verdict_text}

Trả về JSON, "perspective": "negotiation_strategy"
""",
    },

    {
        "id": "lesson_learned",
        "user_type": "layperson",
        "description": "Bài học thực tế rút ra từ vụ án cho người bình thường",
        "prompt": """
Dựa trên bản án sau, tạo 1 cặp hỏi-đáp dạng "bài học phòng ngừa":
- Người hỏi: chưa có tranh chấp nhưng lo ngại về tình huống tương tự
- Câu hỏi kiểu: "Làm thế nào để không rơi vào tình huống như thế này?"

Câu trả lời phải:
1. Tóm tắt ngắn gọn: tranh chấp này xảy ra vì lý do gì cốt lõi
2. Sai lầm phổ biến dẫn đến tranh chấp (cụ thể từ vụ này)
3. Cách phòng ngừa: cần làm gì từ đầu (giấy tờ, chứng cứ, hành động)
4. Nếu đã lỡ rơi vào rồi: xử lý thế nào để ít thiệt hại nhất
5. Câu hỏi tự kiểm tra: "Nếu tôi đang ở vị trí đó, tôi có... không?"

BẢN ÁN:
{verdict_text}

Trả về JSON, "perspective": "lesson_learned"
""",
    },

    {
        "id": "specific_legal_qa",
        "user_type": "both",
        "description": "Sinh 3 cặp Q&A từ các tình tiết pháp lý cụ thể trong bản án",
        "prompt": """
Dựa trên bản án sau, hãy tạo 3 cặp hỏi-đáp NGẮN từ các tình tiết pháp lý CỤ THỂ:

Ví dụ tình tiết có thể khai thác:
- Thời hiệu thừa kế
- Giá trị pháp lý của di chúc
- Công sức trông nom tài sản được tính thế nào
- Yêu cầu phản tố là gì
- Giấy chứng nhận QSDĐ có thể bị hủy trong trường hợp nào
(Chọn 3 tình tiết PHÙ HỢP với bản án này)

Mỗi cặp: câu hỏi ngắn gọn tự nhiên + câu trả lời 3-5 câu súc tích, trích dẫn điều luật.

BẢN ÁN:
{verdict_text}

Trả về JSON:
{{
  "qa_pairs": [
    {{
      "user": "...",
      "assistant": "...",
      "metadata": {{
        "perspective": "specific_legal_qa",
        "user_type": "layperson",
        "domain": "...",
        "key_law": "..."
      }}
    }},
    ... (3 phần tử)
  ]
}}
""",
    },
]


# ── PDF reader ───────────────────────────────────────────────────────────────

def read_pdf(path: Path) -> str:
    """Đọc text từ PDF bằng pymupdf (fitz)."""
    try:
        import fitz  # pymupdf
        doc  = fitz.open(str(path))
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except ImportError:
        raise ImportError("Cần cài: pip install pymupdf")


def read_file(path: Path) -> str:
    """Đọc văn bản từ PDF hoặc text file."""
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    return path.read_text(encoding="utf-8")


# ── Generator ────────────────────────────────────────────────────────────────

class VerdictConverter:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def convert_perspective(
        self,
        verdict_text: str,
        perspective: dict,
        max_retries: int = 3,
    ) -> list[dict]:
        """Sinh dataset cho 1 góc nhìn, trả về list các cặp hỏi-đáp."""
        prompt = perspective["prompt"].format(verdict_text=verdict_text[:12000])

        for attempt in range(max_retries):
            try:
                response = self.client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=(
                        "Bạn là chuyên gia tạo dataset huấn luyện AI pháp luật Việt Nam. "
                        "Luôn trả về JSON hợp lệ, không có text ngoài JSON. "
                        "Nội dung phải thực tế, cụ thể, đúng pháp luật. "
                        "Câu trả lời của luật sư phải dài ít nhất 200 từ, có cấu trúc rõ ràng."
                    ),
                    messages=[{"role": "user", "content": prompt}],
                )

                raw = response.content[0].text.strip()
                # Bóc JSON khỏi markdown fence
                if "```" in raw:
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                raw = raw.strip()

                parsed = json.loads(raw)

                # Xử lý 2 format: single object hoặc {qa_pairs: [...]}
                if "qa_pairs" in parsed:
                    items = parsed["qa_pairs"]
                elif "user" in parsed:
                    items = [parsed]
                else:
                    print(f"    ⚠️  Format lạ: {list(parsed.keys())}")
                    return []

                # Thêm system prompt vào mỗi item
                results = []
                for item in items:
                    if not item.get("user") or not item.get("assistant"):
                        continue
                    if len(item["assistant"]) < 150:
                        continue  # bỏ câu trả lời quá ngắn

                    user_type = item.get("metadata", {}).get("user_type", "layperson")
                    item["system"] = SYSTEM_PROMPTS.get(
                        user_type,
                        SYSTEM_PROMPTS["layperson"],
                    )
                    results.append(item)

                return results

            except json.JSONDecodeError as e:
                print(f"    ⚠️  JSON lỗi attempt {attempt+1}: {e}")
            except anthropic.RateLimitError:
                wait = 30 * (attempt + 1)
                print(f"    ⏳ Rate limit, đợi {wait}s...")
                time.sleep(wait)
            except Exception as e:
                print(f"    ❌ Lỗi attempt {attempt+1}: {e}")
                time.sleep(5)

        return []

    def convert_verdict(self, verdict_path: Path) -> list[dict]:
        """Xử lý 1 bản án, sinh tất cả góc nhìn."""
        print(f"\n📄 Đang xử lý: {verdict_path.name}")
        verdict_text = read_file(verdict_path)
        print(f"   → {len(verdict_text):,} ký tự")

        all_samples = []

        for perspective in PERSPECTIVES:
            print(f"   ⚙️  Góc nhìn: {perspective['id']}")
            samples = self.convert_perspective(verdict_text, perspective)
            print(f"      → {len(samples)} mẫu")
            all_samples.extend(samples)
            time.sleep(1)  # tránh rate limit

        print(f"   ✅ Tổng: {len(all_samples)} mẫu từ bản án này")
        return all_samples


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Chuyển bản án → dataset fine-tune")
    parser.add_argument("--input",   type=str, help="Đường dẫn 1 file PDF/txt")
    parser.add_argument("--folder",  type=str, help="Thư mục chứa nhiều bản án")
    parser.add_argument("--dry-run", action="store_true", help="Xem prompt mẫu, không gọi API")
    args = parser.parse_args()

    if not args.input and not args.folder:
        parser.error("Cần chỉ định --input hoặc --folder")

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Thu thập danh sách file
    files: list[Path] = []
    if args.input:
        files.append(Path(args.input))
    if args.folder:
        folder = Path(args.folder)
        files.extend(folder.glob("*.pdf"))
        files.extend(folder.glob("*.txt"))

    print(f"Tìm thấy {len(files)} bản án")

    if args.dry_run:
        if files:
            verdict_text = read_file(files[0])[:3000]
            print(f"\n=== DRY RUN — Prompt mẫu (góc nhìn: plaintiff_before) ===\n")
            print(PERSPECTIVES[0]["prompt"].format(verdict_text=verdict_text))
        return

    converter = VerdictConverter()
    total_samples = 0

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for path in files:
            try:
                samples = converter.convert_verdict(path)
                for s in samples:
                    # Thêm nguồn gốc
                    s.setdefault("metadata", {})["source_file"] = path.name
                    out_f.write(json.dumps(s, ensure_ascii=False) + "\n")
                out_f.flush()
                total_samples += len(samples)
            except Exception as e:
                print(f"   ❌ Lỗi xử lý {path.name}: {e}")

    print(f"\n{'─'*50}")
    print(f"✅ Hoàn tất!")
    print(f"   Bản án đã xử lý : {len(files)}")
    print(f"   Tổng mẫu sinh ra : {total_samples}")
    print(f"   Trung bình/bản án: {total_samples // max(len(files), 1)}")
    print(f"   Output           : {OUTPUT_FILE.resolve()}")

    # Thống kê
    counts: dict[str, int] = {}
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                p = obj.get("metadata", {}).get("perspective", "unknown")
                counts[p] = counts.get(p, 0) + 1
            except Exception:
                pass

    print("\n📊 Phân phối theo góc nhìn:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"   {k:<25}: {v}")


if __name__ == "__main__":
    main()
