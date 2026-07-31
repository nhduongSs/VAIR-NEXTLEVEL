# Kế hoạch nộp — 5 lượt/ngày, cách nhau 10 phút

Không dựng sẵn cả loạt. Mỗi lượt: dựng **đúng một** phương án tốt nhất theo hiểu
biết hiện tại, nộp, đợi ba chỉ số trả về, rồi mới quyết bước sau. Kết quả nào
cũng đổi hướng đi tiếp, nên dựng trước là thừa.

## Trạng thái

| Lần | Thư mục | Concept | Điểm |
|---|---|---:|---:|
| 01 | `submissions/01-20260727-qwen8b/` | 2,266 | 14.4255 |
| 04 | `submissions/04-kaggle-rejector/` | 2,366 | **27.5217** ← tốt nhất |

Đội đầu bảng hơn 50. Tính toán cho thấy toàn bộ khoảng cách nằm ở **span thừa**:
precision hiện tại 50–64%, và nếu loại sạch span thừa thì trần là 43–76 tuỳ `G`.

## Lượt tiếp theo: nâng sàn ngưỡng lên 0.30

Chạy notebook `notebooks/Viettel_AI_Race_Kaggle_Predict_V2.ipynb` rồi nộp
`/kaggle/working/output.zip`.

**Đổi đúng một thứ** so với cấu hình đã cho 27.5217: sàn ngưỡng GLiNER 0.30.
Corrector giữ nguyên, `reject_margin` giữ nguyên 1.0.

| Type | trước | sau |
|---|---:|---:|
| TRIỆU_CHỨNG | 0.20 | **0.30** |
| CHẨN_ĐOÁN | 0.25 | **0.30** |
| TÊN_XÉT_NGHIỆM | 0.15 | **0.30** |
| THUỐC | 0.30 | 0.30 |
| KẾT_QUẢ_XÉT_NGHIỆM | 0.35 | 0.35 |

### Vì sao chọn lever này

Hai lever đang mở, chọn cái có kỳ vọng dương rõ hơn:

| | Kỳ vọng |
|---|---|
| hạ `reject_margin` | Đã đo: margin 1.0 bỏ 131 span, precision 65–69%, hoà vốn 60.8%. Dải thêm khi hạ margin có precision **thấp hơn** dải đã bỏ → nằm sát vạch hoà vốn, phương sai cao. |
| **nâng ngưỡng GLiNER** | Bỏ các span điểm **thấp nhất**, nhóm khả nghi nhất. Ngưỡng hiện tại lấy từ repo tham chiếu, vốn được chọn **khi có bước additions** bù lại recall — ta không bật additions nên operating point khác hẳn và ngưỡng đang quá thấp. |

Nâng **vừa phải** chứ không nhảy thẳng lên 0.50: tỉ lệ rác giảm dần khi bỏ sâu,
nên đáy dải (0.15–0.30) là chỗ an toàn nhất để cắt trước. Trên bản CPU, sàn 0.30
bỏ 361 span; trên Kaggle con số sẽ khác vì bộ loại đã bỏ 131 span, có thể trùng
một phần.

### Đọc kết quả

Hoà vốn ở **60.8%** rác trong số span bị bỏ.

* **tăng** → giả thuyết "emit thừa" đúng; lượt sau nâng tiếp lên 0.40;
* **đứng yên** → dải 0.15–0.30 lẫn lộn; chuyển sang bộ loại Qwen vì nó xét ngữ
  cảnh chứ không chỉ điểm số, và dùng bảng hiệu chuẩn để chọn margin;
* **giảm** → dải điểm thấp chứa nhiều concept thật hơn tưởng; bỏ hẳn hướng siết
  precision, chuyển sang ranh giới span và assertion.

Gửi lại **cả ba chỉ số** (`WER`, `J_assertion`, `J_candidates`), không chỉ điểm
tổng — phân rã mới cho biết cái gì đang xảy ra. Kèm bảng `rejector:` in ở cuối
log, vì nó cho cả đường cong margin trong một lượt chạy.

## Sau khi lưu kết quả

```bash
ID=09-$(date +%Y%m%d)-threshold-030
mkdir -p "submissions/$ID/json"
cp output.zip "submissions/$ID/"
cd "submissions/$ID" && unzip -j output.zip 'output/*' -d json/
shasum -a 256 output.zip
```

Rồi viết `MANIFEST.md` kèm ba chỉ số thành phần.

## Nguyên tắc

* mỗi lượt đổi **đúng một thứ**;
* `WER` trên trang kết quả là **tỷ lệ lỗi**, nên `text = 100 − WER`;
* ghi điểm vào MANIFEST ngay khi có, kèm phân rã thành phần.
