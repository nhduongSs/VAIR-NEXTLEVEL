# Lần 05 — phép thử ngưỡng 0.50 (chưa nộp)

| Trường | Giá trị |
|---|---|
| SHA-256 | `18053fb6fa837d088dbadb23e22e162eb19226582da9fcb4b9904574d3c32185` |
| Kích thước | 50,032 bytes |
| Concept | **1,378** (13.8/bản ghi) — so với 2,497 của lần 04 |
| Chạy trên | CPU, không GPU, ~55 giây |

```bash
medical-coder predict-v2 \
  --input-dir input --output-dir output \
  --icd-kb data/terminology/icd10_vn.tsv \
  --rxnorm-kb data/terminology/rxnorm.tsv \
  --threshold TRIỆU_CHỨNG=0.50 --threshold CHẨN_ĐOÁN=0.50 \
  --threshold THUỐC=0.50 --threshold TÊN_XÉT_NGHIỆM=0.50 \
  --threshold KẾT_QUẢ_XÉT_NGHIỆM=0.50 \
  --zip-path output.zip
```

## Đây không phải một lần nộp để ăn điểm

Đây là **phép đo**. Nó bỏ 1.119 span nằm trong dải điểm GLiNER thấp (0.15–0.50)
để đo trực tiếp tỉ lệ rác trong dải đó — con số mà không suy luận nào từ ba chỉ
số công bố thay thế được.

Ngưỡng hoà vốn: `p/k > 1/(1 + 2·text/α) = 60.8%`.

| tỉ lệ rác thật trong 1.119 span bị bỏ | `text` dự kiến |
|---|---|
| 50% | 30.6 → ~25.0 (tệ đi) |
| 60% | 30.6 → ~31.3 (hoà) |
| 70% | 30.6 → ~38.9 |
| 80% | 30.6 → ~48.0 |
| 90% | 30.6 → ~59.3 |

Dải kết quả rất rộng, nên **kết quả nào cũng cho biết nhiều**:

* điểm tăng mạnh → giả thuyết "ta emit thừa gấp đôi" đúng, và đường tới 50+ là
  tiếp tục siết precision;
* điểm giảm → dải điểm thấp chứa nhiều concept thật hơn ta tưởng, và phải chuyển
  hướng sang sửa ranh giới span cùng assertion.

Ngoài ra, ghép `text` của lần này với lần 04 (cùng pipeline, khác đúng một tham
số, `P` chênh 43%) sẽ chốt được `G` chặt hơn nhiều so với vùng 1.150–1.600 hiện
tại — mà `G` là ẩn số chi phối mọi ước lượng khác.

## Lưu ý

Bản ghi `96` rỗng. Concept có candidate giảm còn 131 (lần 04 là 160).
