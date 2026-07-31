# Kế hoạch nộp — 5 lượt/ngày, cách nhau 10 phút

Mỗi lượt nộp trả về đúng **ba con số**. Với hạn mức 5 lượt/ngày, phải coi mỗi lượt
là một phép đo được thiết kế trước, không phải một lần thử vận may.

## Trạng thái

| Lần | Thư mục | Concept | Điểm |
|---|---|---:|---:|
| 01 | `submissions/01-20260727-qwen8b/` | 2,266 | 14.4255 |
| 04 | `submissions/04-kaggle-rejector/` | 2,366 | **27.5217** ← tốt nhất |

Đội đầu bảng hơn 50. Tính toán cho thấy toàn bộ khoảng cách nằm ở **span thừa**:
precision hiện tại 50–64%, và nếu loại sạch span thừa thì điểm trần là 43–76 tuỳ
`G`. Nên mọi phép đo dưới đây đều nhắm vào một câu hỏi: **bỏ bớt span thì được
hay mất?**

## Hôm nay: ba lượt, đo đường cong ngưỡng

Ba gói dưới đây chỉ khác nhau **đúng một tham số** là ngưỡng GLiNER. Ba điểm đủ
dựng đường cong và chốt `G` — ẩn số đang chi phối mọi ước lượng khác.

| Thứ tự | Thư mục nộp | Concept |
|---|---|---:|
| 1 | `submissions/05-probe-threshold-050/output.zip` | 1,378 |
| 2 | `submissions/07-probe-threshold-065/output.zip` | 890 |
| 3 | `submissions/06-probe-threshold-035/output.zip` | 1,919 |

Nộp theo đúng thứ tự này. Nếu lượt 1 tụt mạnh thì dừng, không nộp lượt 2 — đường
cong đã đủ dốc để kết luận, và để dành lượt cho hướng khác.

### Cảnh báo về so sánh

Ba gói này chạy **CPU, không có corrector và không có bộ loại span**, trong khi
mốc 27.5217 chạy trên Kaggle có cả hai. Nên **đừng so trực tiếp từng gói với
27.5217** — chênh lệch có phần đến từ việc thiếu GPU.

So sánh có ý nghĩa là **giữa ba gói với nhau**: chúng khác nhau đúng một tham số,
nên hình dạng đường cong là thật. Tìm được ngưỡng tốt nhất rồi mới chạy lại trên
Kaggle kèm corrector và bộ loại để lấy điểm thật.

## Lượt dự phòng

| Thư mục | Đo cái gì |
|---|---|
| `submissions/08-probe-no-candidates/output.zip` | Bỏ sạch candidate, giữ nguyên ngưỡng gốc. Thành phần này chiếm 40% điểm mà ta chỉ emit 171 mã trên 933 concept chẩn đoán/thuốc. Nếu ground truth phần lớn rỗng thì emit mã đang **làm hại**. Repo tham chiếu đo được việc bỏ hết mã chỉ tốn 0.0036 — nếu đúng vậy thì gói này gần như hoà, và ta biết candidate không phải chỗ đáng đầu tư. |

## Đọc kết quả thế nào

Với gói `05` (bỏ 1.119 span so với bản 2.497), ngưỡng hoà vốn là **60.8%** rác
trong số span bị bỏ:

| tỉ lệ rác thật | `text` dự kiến |
|---|---|
| 50% | 30.6 → ~25.0 |
| 60% | ~31.3 (hoà) |
| 70% | ~38.9 |
| 80% | ~48.0 |
| 90% | ~59.3 |

* **tăng mạnh** → giả thuyết "emit thừa gần gấp đôi" đúng; đi tiếp theo hướng siết
  precision, và ngưỡng tối ưu có thể còn cao hơn 0.50;
* **quanh mức cũ** → dải điểm thấp lẫn lộn; chuyển sang bộ loại bằng Qwen thay vì
  cắt thô theo ngưỡng, vì nó xét ngữ cảnh chứ không chỉ điểm số;
* **giảm mạnh** → dải điểm thấp chứa nhiều concept thật; bỏ hướng siết precision,
  chuyển sang ranh giới span và assertion.

## Sau khi có ba điểm

1. Khớp đường cong, chốt `G` và ngưỡng tối ưu.
2. Chạy Kaggle với ngưỡng đó **kèm** corrector và bộ loại → đây mới là lượt nộp
   để ăn điểm.
3. Dùng bảng `rejection_report()` in ra cuối lượt chạy để chọn `reject_margin`
   bằng dữ liệu thay vì đoán.

## Nguyên tắc

* mỗi lượt đổi **đúng một thứ**, nếu không thì không quy được nguyên nhân;
* ghi điểm vào `submissions/<tên>/MANIFEST.md` ngay khi có, kèm ba chỉ số thành
  phần chứ không chỉ điểm tổng;
* `WER` trên trang kết quả là **tỷ lệ lỗi**, nên `text = 100 − WER`.
