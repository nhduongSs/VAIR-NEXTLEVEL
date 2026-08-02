# Lần nộp 01 — Qwen3-8B sinh mention

| Trường | Giá trị |
|---|---|
| Ngày nộp | 27/07/2026, 18:02 (GMT+7) |
| Trạng thái | Đã chấm, 100/100 bản ghi |
| **Điểm tổng** | **14.4255** |
| SHA-256 | `c4f4689b8f8db11227f27c56f3b2b7bff8a906a10c4b6bf1c217c7ad4c6a242f` |
| Kích thước | 82,924 bytes |

## Điểm thành phần

| Thành phần | Giá trị công bố | Điểm thực |
|---|---:|---:|
| `WER` | 83.5952 | **16.4048** (= 100 − WER) |
| `J_assertion` | 20.1874 | 20.1874 |
| `J_candidates` | 8.6197 | 8.6197 |

```text
0.3 × 16.4048 + 0.3 × 20.1874 + 0.4 × 8.6197 = 14.4255
```

`WER` là **tỷ lệ lỗi**, không phải điểm — nên `text` là thành phần yếu nhất chứ
không phải mạnh nhất.

## Cấu hình

| Thành phần | Giá trị |
|---|---|
| GPU | Tesla P100-PCIE-16GB (`sm_60`) |
| PyTorch | 2.10.0+cu126 |
| LLM | `Qwen/Qwen3-8B`, NF4 4-bit — sinh mention dạng JSON rồi căn offset |
| Retriever | `intfloat/multilingual-e5-small` |
| Tham số kê khai | 8.318B |
| ICD KB | CDC ICD-10-CM FY2026 (**tiếng Anh**) |
| RxNorm KB | Current Prescribable Content 07/2026 |
| Candidate | `retrieval_top_k=20`, tối đa 3 mã/entity, LLM rerank |

## Thống kê output

| Chỉ số | Giá trị |
|---|---:|
| Tổng concept | 2,266 |
| Trung bình / bản ghi | 22.66 |
| Bản ghi rỗng | 2 (`75`, `96`) |
| TRIỆU_CHỨNG | 869 |
| TÊN_XÉT_NGHIỆM | 425 |
| CHẨN_ĐOÁN | 377 |
| THUỐC | 298 |
| KẾT_QUẢ_XÉT_NGHIỆM | 297 |
| Concept có candidate | 188 |
| Tổng mã xuất ra | 460 |
| Nhãn assertion | 266 |

## Vì sao thay phương pháp

1. **NER sinh văn bản.** Bắt Qwen viết lại mention rồi căn ngược về text gốc:
   mention nào bị diễn giải lại là bị bước alignment loại bỏ, và mention sinh ra
   không có score nên không có nút vặn precision/recall.
2. **KB tiếng Anh.** Mention `viêm túi mật` về nguyên tắc không thể khớp alias
   nào trong CDC ICD-10-CM.
3. **Emit quá tay.** 460 mã trên 188 concept và 266 nhãn assertion, trong khi
   scorer đếm mỗi concept thừa hai lần và trả Jaccard 1.0 cho dự đoán rỗng đúng
   với ground truth rỗng.

Chi tiết ở [`../../KET_QUA_NOP_LAN_01.md`](../../KET_QUA_NOP_LAN_01.md) và
[`../../PHAN_TICH_DOI_THU_VA_CAI_TIEN.md`](../../PHAN_TICH_DOI_THU_VA_CAI_TIEN.md).
