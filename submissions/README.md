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
| 05 | 01/08/2026 | nâng sàn ngưỡng GLiNER lên 0.30 | 26.8959 |

## Lịch sử thành phần

| Lần | text | assertion | candidates | tổng |
|---|---:|---:|---:|---:|
| 01 | 16.4048 | 20.1874 | 8.6197 | 14.4255 |
| 03 | 30.2302 | 32.5915 | 20.7332 | 27.1398 |
| 04 | 30.6012 | 33.0650 | 21.0545 | 27.5217 |
| 05 | 30.0350 | 32.3106 | 20.4804 | 26.8959 |
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


## Kết luận rút ra từ lần 05

Nâng ngưỡng làm **cả ba thành phần cùng giảm**. Giải ngược cho biết dải điểm
GLiNER 0.15–0.30 chỉ chứa **~57% rác**, dưới vạch hoà vốn 60.8%, nên cắt là lỗ.

Đặt cạnh nhau hai tín hiệu phân biệt rác:

| tín hiệu | tỉ lệ rác trong dải bị cắt |
|---|---:|
| điểm GLiNER (dải thấp nhất) | 55–58% |
| teacher Qwen (`margin=1.0`) | 65–69% |
| *hoà vốn* | *60.8%* |

Precision tổng thể của pipeline là 50–64%. Dải **thấp nhất** của GLiNER chỉ 57% —
gần bằng mức trung bình. Nghĩa là **điểm tin cậy của GLiNER gần như không phân
biệt được đúng/sai**, nên vặn ngưỡng là ngõ cụt theo cả hai hướng.

Teacher Qwen phân biệt tốt hơn, nhưng trần của nó cũng chỉ khoảng +2 điểm nếu
giữ được precision 67% khi bỏ tới 1.000 span — mà precision chắc chắn tụt khi bỏ
sâu. Kết luận: **mọi lever vặn tham số còn lại đều chỉ đáng ±1 điểm.**

## Bộ nhãn giả — kết quả kiểm định (02/08/2026)

Gán nhãn 100 tài liệu bằng `gpt-5.6-terra` (3.183 concept) và `gpt-5.6-sol`
(3.533). Lưu tại `data/pseudo_gt/`.

| Phép kiểm | Kết quả |
|---|---|
| 1. Cấu trúc | **HỎNG** — 3.183–3.533 concept so với `G` 1.150–1.700 suy từ điểm thật. Hai model gán nhãn rộng gấp ~2 lần. |
| 2. Tái lập điểm | lệch +2.5 đến +7.1 điểm, luôn cao hơn thực tế |
| 3. Bảo toàn thứ tự | **ĐẠT**, kể cả ở cặp chênh 0.63 điểm |

Phép 3 mới là thứ quyết định, và nó đạt ở đúng biên độ cần dùng:

```text
BTC   : gốc 27.5217 -> sàn 0.30 26.8959   = -0.6258
terra : gốc 32.455  -> sàn 0.30 32.129    = -0.3252   đúng dấu
sol   : gốc 32.812  -> sàn 0.30 32.566    = -0.2467   đúng dấu
```

Cả hai bộ nhãn dự đoán đúng dấu của một thay đổi chênh 0.63 điểm — thứ mà bốn
lượt nộp trước phải trả bằng hạn mức để biết. **Độ lớn bị nén khoảng một nửa**,
nên chỉ dùng để so sánh tương đối, không đọc như điểm thật.

### Đã dùng ngay để bác bỏ một hướng

Chênh lệch phân bố type gợi ý phải hạ ngưỡng cho `KẾT_QUẢ_XÉT_NGHIỆM` (ta 26 so
với 193 của nhãn) và `TÊN_XÉT_NGHIỆM` (227 so với 579). Đo cục bộ thì **cả hai
đều làm điểm giảm**:

| cấu hình | concept | terra | sol |
|---|---:|---:|---:|
| gốc | 2.497 | **32.455** | **32.812** |
| KẾT_QUẢ 0.35→0.20 | 2.573 | 32.252 | 32.617 |
| KẾT_QUẢ 0.10 + TÊN_XN 0.08 | 2.726 | 32.061 | 32.424 |
| sàn 0.40 | 1.722 | 30.956 | 30.861 |
| sàn 0.50 | 1.378 | 28.104 | 27.495 |

Span điểm thấp mà GLiNER đề xuất cho hai type đó **không trùng** với span nhãn
đánh dấu — thêm vào chỉ tăng số concept thừa. Vặn ngưỡng đã hết đường theo **cả
hai** hướng, và lần này biết được mà không tốn lượt nộp nào.
