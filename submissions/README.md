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
| [06](06-kaggle-assertions/) | 02/08/2026 | bật assertion bằng luật | **27.5691** |

## Lịch sử thành phần

| Lần | text | assertion | candidates | tổng |
|---|---:|---:|---:|---:|
| 01 | 16.4048 | 20.1874 | 8.6197 | 14.4255 |
| 03 | 30.2302 | 32.5915 | 20.7332 | 27.1398 |
| 04 | 30.6012 | 33.0650 | 21.0545 | 27.5217 |
| 05 | 30.0350 | 32.3106 | 20.4804 | 26.8959 |
| 06 | 30.6012 | 33.2229 | 21.0545 | **27.5691** |
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


## Lần 06 — nhãn giả dự thừa 8 lần ở thay đổi về assertion

`text` và `candidates` giống hệt lần 04 tới 4 chữ số, nên toàn bộ +0.0474 quy về
assertion: 33.0650 → 33.2229.

**Dự đoán +0.5 đến +0.8, thực tế +0.0474** — chệch 10–17 lần.

| thay đổi | nhãn giả dự | thật | tỉ lệ |
|---|---:|---:|---:|
| ngưỡng sàn 0.30 (span) | −0.290 | −0.626 | 2.16x |
| bật assertion | +0.385 | +0.047 | **0.12x** |

Suy ngược precision thật: thành phần assertion tăng 0.1579 trên mẫu số ~3.200,
tức lãi ròng chỉ ~5 đơn vị tử số từ 123 assertion đã emit. Mỗi cái đúng +1, sai
−1, nên khoảng **64 đúng / 59 sai — precision ~52%**, ngay sát vạch hoà vốn
52.5%, chứ không phải 84% đo trên nhãn giả.

### Vì sao đồng thuận hai bộ nhãn không cứu được

Cả `terra` lẫn `sol` đều gán assertion cho 24–27% concept trong khi mức thật
~15%. Chúng **thiên lệch cùng một hướng**, nên việc chúng đồng ý với nhau
(+0.372 và +0.397) không khử được thiên lệch — nó chỉ xác nhận hai model cùng họ
`gpt-5.6` mắc cùng một lỗi. Lập luận "hai bộ đồng ý nên đáng tin" đã dùng để
chọn cấu hình này là **sai**.

### Kết luận về phạm vi dùng được của nhãn giả

| dùng cho | tin được? |
|---|---|
| quyết định về **span** (ngưỡng, loại bỏ, thêm) | có, nhưng độ lớn thiếu ~2 lần |
| quyết định về **assertion** | **không** — thiên lệch cùng chiều với thay đổi |

Trần thật của assertion: 123 assertion mua được 0.1579 điểm thành phần. Kể cả
bắt đúng toàn bộ ~355 concept thật sự có assertion và không sai cái nào, trần
cũng chỉ khoảng **+0.14 điểm tổng** — không phải +1.75 đến +2.5 như ước tính
trước đó từ phương trình điểm.
