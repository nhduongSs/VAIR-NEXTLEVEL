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
