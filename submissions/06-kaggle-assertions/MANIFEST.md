# Lần 06 — bật assertion bằng luật

| Trường | Giá trị |
|---|---|
| SHA-256 | `9ee9a86c93fc…` |
| Kích thước | 72,362 bytes |
| Concept | 2,371 |
| Có assertion | **123 (5.2%)** — 103 `isNegated`, 22 `isHistorical` |
| Có candidate | 171 (không đổi) |
| **Điểm BTC** | **27.5691** (từ 27.5217) |
| `WER` / text | 69.3988 / 30.6012 — **không đổi** |
| `J_assertion` | **33.2229** (từ 33.0650) |
| `J_candidates` | 21.0545 — **không đổi** |

## Đúng một biến thay đổi

So với lần 04 (27.5217): **2.366/2.371 span trùng khớp**, 5 span mới, 0 span mất.
Năm span chênh gần như chắc chắn là dao động của teacher Qwen ở sát ngưỡng do
thành phần batch khác nhau, không phải thay đổi chủ ý. Candidate và phân bố type
giữ nguyên.

Nên chênh lệch điểm quy được trọn vẹn cho assertion.

## Dự đoán trước khi chạy, và kết quả đo lại

| | terra | sol |
|---|---:|---:|
| dự đoán cục bộ (trên output lần 04) | +0.372 | +0.397 |
| đo trên output Kaggle thật | **+0.370** | **+0.400** |

Khớp tới 0.003 — luật chạy trên Kaggle đúng như chạy ở máy.

Hệ số nén của nhãn giả đo ở cặp trước là ~0.5 (nhãn dự −0.33 cho thay đổi thật
−0.63), nên **điểm thật kỳ vọng +0.5 đến +0.8**, tức khoảng **28.0–28.3**.

Nếu rơi ngoài khoảng đó thì chính con số đó cho biết hệ số nén thật, dùng được
cho mọi lần đo sau.

## Kiểm tra

`medical-coder validate` PASS: đủ 100 tệp, schema hợp lệ, mọi span thoả
`raw_text[start:end] == text`, ZIP đúng 100 member dưới `output/`.


## Kết quả: dự đoán chệch 10-17 lần

Dự đoán +0.5 đến +0.8, thực tế **+0.0474**. `text` và `candidates` không đổi tới
4 chữ số nên toàn bộ quy về assertion, đúng như thiết kế một-biến.

Suy ngược precision thật ~**52%** — sát vạch hoà vốn 52.5%, chứ không phải 84%
đo trên nhãn giả. Xem `../README.md` mục "Lần 06" để biết vì sao đồng thuận hai
bộ nhãn không khử được thiên lệch chung.

Giữ lại cấu hình này vì vẫn dương, nhưng assertion **không phải một lever** như
đã tưởng: trần thật khoảng +0.14 điểm tổng.
