# Lưu trữ các lần nộp

`output/` và `output.zip` ở thư mục gốc bị `.gitignore` (chúng bị ghi đè mỗi lượt
chạy). Thư mục này giữ bản sao **được git theo dõi** của từng artifact, kèm
checksum và cấu hình đã tạo ra nó, để so sánh công bằng giữa các lần.

Mỗi thư mục con có:

```text
<id>-<ngày>-<phương pháp>/
├── output.zip     # đúng tệp đã nộp (hoặc đã dựng)
├── json/          # bản giải nén để xem diff bằng git
└── MANIFEST.md    # điểm, checksum, cấu hình, thống kê
```

| Lần | Ngày | Phương pháp | Điểm BTC |
|---|---|---|---:|
| [01](01-20260727-qwen8b/) | 27/07/2026 | Qwen3-8B sinh mention + E5 retrieval + LLM rerank | **14.4255** |
| [02](02-20260730-predict-v2-cpu/) | 30/07/2026 | `predict-v2` CPU, không selector | chưa nộp |
| 03 | 30/07/2026 | Kaggle, GLiNER + corrector | **27.1398** |
| 04 | 31/07/2026 | thêm bộ loại span, `reject_margin=1.0` | **27.5217** |

## Lịch sử thành phần

| Lần | text | assertion | candidates | tổng |
|---|---:|---:|---:|---:|
| 01 | 16.4048 | 20.1874 | 8.6197 | 14.4255 |
| 03 | 30.2302 | 32.5915 | 20.7332 | 27.1398 |
| 04 | 30.6012 | 33.0650 | 21.0545 | 27.5217 |
| *tham chiếu* | *32.1820* | *35.2687* | *19.1084* | *27.8786* |

`WER` trên trang kết quả là **tỷ lệ lỗi**, nên `text = 100 − WER`. Cả bốn mốc đều
tái lập chính xác dưới `0.3·text + 0.3·assertion + 0.4·candidates`.

Từ lần 03 sang 04 chỉ đổi đúng một thứ — bật bộ loại span — nên +0.3819 quy được
trọn vẹn cho nó. Hai ước lượng độc lập (từ text và từ assertion) đều cho ra
**~23 span bị bỏ**; chúng khớp nhau nghĩa là bộ loại gần như chỉ bỏ đúng span
thừa, vì bỏ nhầm span đúng sẽ làm hai ước lượng lệch nhau.

Mỗi lần nộp chỉ trả về một con số, nên **mỗi lần chỉ đổi một thứ** — trộn nhiều
thay đổi thì không quy được nguyên nhân.

## Cách thêm một lần nộp

```bash
ID=03-$(date +%Y%m%d)-predict-v2-kaggle
mkdir -p "submissions/$ID/json"
cp output.zip "submissions/$ID/"
cp output/*.json "submissions/$ID/json/"
shasum -a 256 "submissions/$ID/output.zip"
```

Rồi viết `MANIFEST.md` theo mẫu của các lần trước.
