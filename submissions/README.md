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
| [02](02-20260730-predict-v2-cpu/) | 30/07/2026 | `predict-v2` CPU (GLiNER + exact-alias), **không** có selector | chưa nộp |

Lần chạy Kaggle có đủ selector Qwen sẽ là lần 03; khi có kết quả thì thêm thư mục
mới thay vì ghi đè lần cũ.

## Cách thêm một lần nộp

```bash
ID=03-$(date +%Y%m%d)-predict-v2-kaggle
mkdir -p "submissions/$ID/json"
cp output.zip "submissions/$ID/"
cp output/*.json "submissions/$ID/json/"
shasum -a 256 "submissions/$ID/output.zip"
```

Rồi viết `MANIFEST.md` theo mẫu của các lần trước.
