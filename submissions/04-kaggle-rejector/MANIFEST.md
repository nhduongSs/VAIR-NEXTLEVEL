# Lần 04 — Kaggle, GLiNER + corrector + bộ loại span

| Trường | Giá trị |
|---|---|
| Ngày nộp | 31/07/2026 |
| **Điểm tổng** | **27.5217** |
| `WER` / text | 69.3988 / **30.6012** |
| `J_assertion` | 33.0650 |
| `J_candidates` | 21.0545 |
| SHA-256 | `63318d83c89b...` (xem `shasum -a 256 output.zip`) |
| Concept | 2,366 |

## Cấu hình

GLiNER `urchade/gliner_multi-v2.1` + Qwen3-4B-Instruct-2507 làm teacher chính,
`reject_margin=1.0`, không có teacher phụ nên **không có additions**.

| Type | Lần 03 (CPU + corrector) | Lần 04 |
|---|---:|---:|
| TRIỆU_CHỨNG | — | 1,180 |
| CHẨN_ĐOÁN | — | 624 |
| THUỐC | — | 309 |
| TÊN_XÉT_NGHIỆM | — | 227 |
| KẾT_QUẢ_XÉT_NGHIỆM | — | 26 |

So với bản CPU không selector (2,497 concept, 529 CHẨN_ĐOÁN):

* corrector đổi type **95** span TRIỆU_CHỨNG → CHẨN_ĐOÁN;
* bộ loại bỏ **131** span;
* thêm **0** span, đúng như thiết kế khi thiếu teacher phụ.

## Hiệu chỉnh một ước lượng sai

Trước khi có tệp này, tôi suy ngược từ hai phương trình điểm và kết luận bộ loại
bỏ "~23 span" với precision gần như tuyệt đối. Con số thật là **131**, tức tôi
sai 5,7 lần.

Giải lại với `k = 131` đã biết: trong 131 span bị bỏ có khoảng **85–90 span thật
sự thừa**, tức precision của bộ loại là **65–69%**.

Vẫn trên vạch hoà vốn 60.8% nên nó có lời, nhưng biên an toàn mỏng hơn nhiều so
với tôi tưởng. Hệ quả: hạ `reject_margin` xuống 0.0 sẽ bỏ nhiều span hơn nhưng
precision sẽ tụt, và có thể rơi xuống dưới vạch hoà vốn.

Bài học: đừng suy ngược một đại lượng khi có thể đo thẳng. Bảng hiệu chuẩn
`rejection_report()` thêm sau đó chính là để không phải đoán lần nữa.

## Hiệu chuẩn bộ loại — `rejection_stats.json`

2.634 span được teacher chấm. Phân bố margin:

| phân vị | margin |
|---|---:|
| p1 | −18.50 |
| p5 | −1.50 |
| p10 | 10.00 |
| p50 | **20.62** |
| p90 | 23.00 |

* teacher nói **CÓ** với **94.7%** span, trung vị **20.6 logit**;
* toàn dải `−2 … +2` chỉ có **29 span (1.1%)**.

### Hai hệ quả, cả hai đều bác bỏ kế hoạch đang định làm

**Vặn `reject_margin` là vô nghĩa.** Đường cong gần như phẳng: margin −1.0 bỏ 149
span, margin 3.0 bỏ 124 — chênh 25 span trên cả dải 4 logit. Mọi giá trị trong
khoảng thường dùng đều cho gần đúng một kết quả.

**`teacher_decides` còn tệ hơn.** Với `decide_margin=0.0`, teacher giữ 94.7% số
span. Áp lên 5.741 ứng viên thô sẽ emit 4.000–5.000 concept — lệch theo đúng
hướng **đắt nhất** của thang điểm, vì mỗi concept thừa bị đếm hai lần vào cả ba
mẫu số.

### Điều rút ra

Teacher **không phải một bộ phân loại**; nó là một cái gật đầu rất tự tin. Chỉ
phần đuôi từ chối 5.3% là mang thông tin — và phần đó đã được khai thác rồi,
chính là +0.3819 của lần nộp này.

Nguyên nhân sâu hơn: prompt đang hỏi *"đây có phải khái niệm y khoa không?"*,
trong khi câu hỏi đúng phải là *"người gán nhãn có đánh dấu span này không?"*.
Phần lớn span của GLiNER **đều** mang tính y khoa, nên câu trả lời "có" là đúng
mà vô dụng.
