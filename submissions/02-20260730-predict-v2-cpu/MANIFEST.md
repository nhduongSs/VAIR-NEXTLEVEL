# Lần dựng 02 — `predict-v2` CPU (chưa nộp)

| Trường | Giá trị |
|---|---|
| Ngày dựng | 30/07/2026 |
| Trạng thái | **Chưa nộp, chưa có điểm** |
| SHA-256 | `12322556bec85dd91fd2ceda5ac19ce0200ae86b056ae274b9fed0a83737a6a2` |
| Kích thước | 73,928 bytes |

Đây là mốc tham chiếu của nhánh `predict-v2` khi chạy **không có GPU**, tức là
thiếu cả hai bước Qwen (corrector và consensus additions). Bản đầy đủ chạy trên
Kaggle sẽ là lần 03.

## Cấu hình

```bash
medical-coder predict-v2 \
  --input-dir input --output-dir output_v2 \
  --icd-kb data/terminology/icd10_vn.tsv \
  --rxnorm-kb data/terminology/rxnorm.tsv \
  --zip-path output_v2.zip
```

| Thành phần | Giá trị |
|---|---|
| NER | `urchade/gliner_multi-v2.1` (0.289B), CPU |
| Ngưỡng | TRIỆU 0.20 / CHẨN 0.25 / THUỐC 0.30 / TÊN_XN 0.15 / KẾT_QUẢ_XN 0.35 |
| Selector | **không có** (cần GPU) |
| ICD KB | ICD-10 **tiếng Việt** TT06/2026/TT-BYT, 15,845 mã |
| RxNorm KB | Current Prescribable Content, 32,983 RxCUI (đã loại SCDC) |
| Candidate | chỉ khi alias khớp duy nhất, tối đa 1 mã |
| Assertion | rỗng, có chủ ý |
| Thời gian | ~55 giây cho 100 bản ghi |

## Thống kê output

| Chỉ số | Lần 01 | Lần 02 |
|---|---:|---:|
| Tổng concept | 2,266 | 2,497 |
| Bản ghi rỗng | 2 | **0** |
| TRIỆU_CHỨNG | 869 | 1,347 |
| CHẨN_ĐOÁN | 377 | 529 |
| THUỐC | 298 | 349 |
| TÊN_XÉT_NGHIỆM | 425 | 239 |
| KẾT_QUẢ_XÉT_NGHIỆM | 297 | 33 |
| Concept có candidate | 188 | 160 |
| Tổng mã xuất ra | 460 | **160** |
| Nhãn assertion | 266 | **0** |

## Điểm yếu đã biết

* Thiếu corrector nên `CHẨN_ĐOÁN` còn thấp (529 so với 668 của lời giải 27.8786).
* Thiếu additions nên `TÊN_XÉT_NGHIỆM` (239 so với 402) và `KẾT_QUẢ_XÉT_NGHIỆM`
  (33 so với 123) thấp rõ rệt.
* Vẫn lọt tiêu đề mục dạng biến thể, ví dụ `Các triệu chứng hiện tại` trong
  `30.json`; danh sách nhãn mục chỉ so bằng nên không bắt được hậu tố.
* Ngưỡng lấy nguyên từ lời giải tham chiếu, vốn được chọn khi **đã có** selector,
  nên chưa chắc tối ưu cho cấu hình không selector này.

**Chưa có ground truth nên chưa đo được lần nào tốt hơn.** Mọi so sánh ở trên là
về phân bố và cơ chế, không phải về điểm.
