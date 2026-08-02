# Những gì đã biết về tập test ẩn

Tổng hợp mọi thứ suy ra được về ground truth của Vòng 1, tính đến 02/08/2026,
sau 6 lượt nộp có điểm.

Mỗi khẳng định được gắn mức tin cậy:

| Nhãn | Nghĩa |
|---|---|
| **[SỰ KIỆN]** | Số liệu BTC công bố, hoặc đo trực tiếp trên tệp |
| **[SUY RA]** | Giải từ số liệu công bố, có nêu giả định |
| **[ƯỚC LƯỢNG]** | Có căn cứ nhưng phụ thuộc giả định chưa kiểm chứng |
| **[BÁC BỎ]** | Từng tin, đã bị dữ liệu bác |

---

## 1. Hàm chấm điểm — đã xác định chắc chắn

**[SỰ KIỆN]**

```text
final = 0.3 × text + 0.3 × assertion + 0.4 × candidates
text  = 100 − WER
```

Trường `WER` trên trang kết quả là **tỷ lệ lỗi**, không phải điểm. Đây là điểm dễ
đọc nhầm nhất và từng khiến báo cáo lần nộp 01 mô tả `text` là thành phần mạnh
nhất trong khi nó là yếu nhất.

Công thức tái lập **chính xác tới 4 chữ số trên cả 6 mốc đã biết**:

| Lượt | text | assertion | candidates | tính ra | BTC công bố |
|---|---:|---:|---:|---:|---:|
| 01 | 16.4048 | 20.1874 | 8.6197 | 14.4255 | 14.4255 |
| 03 | 30.2302 | 32.5915 | 20.7332 | 27.1398 | 27.1398 |
| 04 | 30.6012 | 33.0650 | 21.0545 | 27.5217 | 27.5217 |
| 05 | 30.0350 | 32.3106 | 20.4804 | 26.8958 | 26.8959 |
| 06 | 30.6012 | 33.2229 | 21.0545 | 27.5690 | 27.5691 |
| tham chiếu | 32.1820 | 35.2687 | 19.1084 | 27.8786 | 27.8786 |

Sáu điểm dữ liệu, ba trong đó từ pipeline hoàn toàn khác nhau — đủ để coi công
thức là chắc chắn.

### 1.1. Cơ chế phạt kép

**[SUY RA]** Mỗi concept dự đoán **không khớp** ground truth bị đếm **hai lần**
vào mẫu số của cả ba thành phần:

```text
text       = Σ(khớp) (1 − WER) / (n_gold + 2·n_thừa)
assertion  = Σ(khớp) J_assert  / (n_gold_assertable + 2·n_thừa_assertable)
candidates = Σ(khớp) w·J_cand  / (Σ_gold w + 2·Σ_thừa w),  w = |mã gold| + 1
```

Hệ quả định lượng: **một concept thừa đắt gấp đôi một concept bỏ sót**. Và một
concept thừa mang 3 mã tốn `2×(3+1) = 8` đơn vị mẫu số thay vì 2.

Đây là tính chất chi phối mọi quyết định thiết kế. Nó chưa được BTC xác nhận
trực tiếp, nhưng mô hình này khớp mọi quan sát và giải thích được vì sao lần nộp
01 (emit nhiều, kém chính xác) thua xa lần 04 ở cùng mức phát thải.

### 1.2. Jaccard có ba trường hợp biên

**[SỰ KIỆN]** Theo đề bài mục 11.3:

* gold rỗng **và** dự đoán rỗng → **1.0**
* gold rỗng, dự đoán không rỗng → **0**
* gold không rỗng, dự đoán rỗng → **0**

Trường hợp đầu là lý do emit `assertions: []` không hề "bỏ trắng": phần lớn
concept có gold rỗng, và mỗi cái cho trọn 1.0 miễn phí.

---

## 2. Ground truth ẩn

### 2.1. Số lượng concept — `G`

**[SUY RA]** `G ≈ 1.150 – 1.700`.

Giải từ `text = α·M/(G + 2S)` với `M + S = P` đã biết, ràng buộc `M ≤ G` và
`α ≤ 1`. Ở lượt 04 (`P = 2366`, `text = 0.306012`):

```text
G = α·M/0.306012 − 2(2366 − M),   M ≥ 1201
```

**Ta đang emit ~2.370 concept, tức nhiều hơn ground truth khoảng 1,4–2 lần.**

### 2.2. Chất lượng ranh giới span — `α`

**[ƯỚC LƯỢNG]** `α ≈ 0.85 – 0.95` (trung bình `1 − WER` trên các cặp đã khớp).

Nghĩa là **ranh giới span không phải vấn đề**. Khi đã khớp được một concept, ta
lấy gần đúng toàn bộ điểm text của nó. Điều này quan trọng vì nó loại bỏ cả một
hướng tối ưu: chỉnh ranh giới không mang lại gì đáng kể.

### 2.3. Precision và recall

**[SUY RA]** precision **50–64%**, tức **900–1.250 trong ~2.370 span không khớp
gì cả**.

Con số này vững qua mọi giả thuyết `G`. Recall thì không — nó phụ thuộc mạnh vào
`G` nên dải rất rộng (48–90%).

### 2.4. Tỷ lệ concept có assertion — `q`

**[SUY RA]** khoảng **15–17%** concept assertable có assertion không rỗng
(`q ≈ 0.83–0.85` là tỷ lệ rỗng).

Suy từ `assertion = q·M_a/D_a` khi emit toàn rỗng. Được xác nhận độc lập ở lượt
06: 123 assertion emit ra chỉ lãi ròng ~5 đơn vị tử số, khớp với giả thuyết mức
thật thấp hơn nhiều so với 24–27% mà nhãn giả gán.

### 2.5. Phân bố loại

**[ƯỚC LƯỢNG]** Nhãn giả (chiết khấu ~2 lần) gợi ý ta đang thiếu nhiều nhất ở
`KẾT_QUẢ_XÉT_NGHIỆM` và `TÊN_XÉT_NGHIỆM`:

| Loại | ta emit | nhãn giả `terra` | ÷2 |
|---|---:|---:|---:|
| KẾT_QUẢ_XÉT_NGHIỆM | 26 | 193 | 96 |
| TÊN_XÉT_NGHIỆM | 227 | 579 | 289 |
| CHẨN_ĐOÁN | 624 | 798 | 399 |
| TRIỆU_CHỨNG | 1.184 | 1.323 | 661 |
| THUỐC | 310 | 290 | 145 |

**Nhưng hạ ngưỡng để lấp hai chỗ thiếu đã đo là LỖ** (mục 5.3). Span điểm thấp
GLiNER đề xuất cho hai loại đó không trùng span nhãn đánh dấu.

---

## 3. Giá trị của từng tín hiệu

Đây là phần quan trọng nhất: **hai tín hiệu ta có đều gần như mù.**

| Tín hiệu | Khả năng phân biệt | Kết luận |
|---|---|---|
| Điểm tin cậy GLiNER | dải thấp nhất chỉ **55–58%** rác, gần bằng precision trung bình 50–64% | **gần như không mang thông tin** |
| Margin teacher Qwen | **65–69%** đúng trên phần bỏ, nhưng nói CÓ với **94.7%** span | chỉ đuôi từ chối 5.3% là dùng được |
| Luật assertion | 84% trên nhãn giả, **~52%** trên thực tế | sát vạch hoà vốn |

### 3.1. Phân bố margin của teacher

**[SỰ KIỆN]** Đo trên 2.634 span (`submissions/04-kaggle-rejector/rejection_stats.json`):

| phân vị | margin |
|---|---:|
| p1 | −18.50 |
| p5 | −1.50 |
| p50 | **20.62** |
| p90 | 23.00 |

Toàn dải `−2 … +2` chỉ có **29 span (1.1%)**. Nên đường cong "margin → số span
bỏ" gần như **phẳng**: margin −1.0 bỏ 149, margin 3.0 bỏ 124.

**Hệ quả: `reject_margin` không phải một tham số điều chỉnh được.** Mọi giá trị
trong dải thường dùng cho cùng một kết quả.

---

## 4. Các ngưỡng hoà vốn

**[SUY RA]** Ba công thức chi phối mọi quyết định.

### 4.1. Bỏ bớt span

Bỏ `k` span trong đó `p` thực sự thừa. Có lời khi:

```text
p/k > 1 / (1 + 2·text/α)   ≈ 60.8%   (text = 0.306, α = 0.95)
```

Lý do bất đối xứng: bỏ nhầm span **đúng** chỉ mất `α` ở tử số (concept gold vẫn
nằm trong mẫu số dưới dạng bỏ sót), còn bỏ đúng span **thừa** giảm được 2 đơn vị
mẫu số.

### 4.2. Emit assertion

Trên một concept đã khớp: gold rỗng + emit rỗng cho 1.0 miễn phí, nên emit nhầm
mất 1. Có lời khi:

```text
precision > 1 / (1 + e)   ≈ 52.5%   (e = độ đúng của TẬP nhãn ≈ 0.90)
```

### 4.3. Giá trị biên của việc thêm concept khớp đúng

```text
Δfinal ≈ (0.3·α + 0.3·q) / D   mỗi concept,   D = G + 2S ≈ 3.900
```

khoảng **+0.014 điểm mỗi concept** bắt thêm được đúng.

---

## 5. Toàn bộ thí nghiệm đã đo

### 5.1. Đổi kiến trúc NER — lượt 01 → 03

**+12.71 điểm.** Thay Qwen sinh mention bằng GLiNER trả offset trực tiếp.

Hai bộ span chỉ trùng nhau **36%**; ở cùng mức phát thải (2.261 so với 2.366)
bộ của GLiNER hơn **1,87 lần** về text. Dấu vết cho biết chỗ hỏng là **ranh
giới**: span dài quá 40 ký tự ở lượt 01 có 102 cái, lượt 04 chỉ 7.

**Bài học:** để model sinh text rồi căn ngược offset là sai lầm kiến trúc, không
phải vấn đề chất lượng model.

### 5.2. Bộ loại span bằng teacher — lượt 03 → 04

**+0.3819.** Bỏ 131 span, precision 65–69%.

### 5.3. Nâng ngưỡng GLiNER — lượt 04 → 05

**−0.6258.** Dải 0.15–0.30 chỉ chứa ~57% rác, dưới vạch hoà vốn 60.8%.

Đo cục bộ sau đó xác nhận **hạ** ngưỡng cũng lỗ. Vặn ngưỡng hết đường **cả hai
hướng**.

### 5.4. Bật assertion bằng luật — lượt 04 → 06

**+0.0474.** `text` và `candidates` không đổi tới 4 chữ số, nên toàn bộ quy về
assertion (33.0650 → 33.2229).

Suy ngược: lãi ròng ~5 đơn vị tử số từ 123 assertion → **~64 đúng / 59 sai,
precision ~52%**.

### 5.5. Đã đo cục bộ và bác bỏ, không tốn lượt nộp

| Hướng | Kết quả đo |
|---|---|
| hạ ngưỡng `KẾT_QUẢ_XÉT_NGHIỆM` 0.35→0.10 | giảm 0.37 (cả hai bộ nhãn) |
| hạ thêm `TÊN_XÉT_NGHIỆM` 0.15→0.08 | giảm 0.39 |
| `teacher_decides` (Qwen chọn span) | sẽ emit 4.000–5.000 concept |
| `isFamily` | terra tụt từ +0.389 xuống +0.199 |
| assertion suy theo tiêu đề mục | lệch 2.1 lần giữa hai bộ nhãn |

---

## 6. Bộ nhãn giả — phạm vi dùng được

**[SỰ KIỆN]** `gpt-5.6-terra` 3.183 concept, `gpt-5.6-sol` 3.533, phần giao
2.761, Jaccard 0.698. Lưu ở `data/pseudo_gt/`.

### 6.1. So với pipeline

| | |
|---|---:|
| pipeline bắt được phần lõi | **44.1%** |
| pipeline emit ngoài lõi | 48.6% |
| span đúng ranh giới nhưng **sai type** | chỉ **127** |

Con số cuối quan trọng: **sai type không phải vấn đề chính**, vấn đề là span
hoàn toàn không có.

### 6.2. Hệ số nén — khác nhau theo loại thay đổi

**[SỰ KIỆN]** Đo trên hai thay đổi đã có điểm thật:

| thay đổi | nhãn giả dự | thật | tỉ lệ |
|---|---:|---:|---:|
| ngưỡng sàn 0.30 (**span**) | −0.290 | −0.626 | **2.16x** |
| bật assertion (**assertion**) | +0.385 | +0.047 | **0.12x** |

| dùng cho | tin được? |
|---|---|
| quyết định về **span** | có — dấu đúng, độ lớn thiếu ~2 lần |
| quyết định về **assertion** | **không** |

### 6.3. Vì sao đồng thuận hai bộ nhãn không khử được thiên lệch

Cả hai đều gán assertion cho 24–27% concept trong khi mức thật ~15%. Chúng lệch
**cùng một hướng**, nên việc chúng cho cùng con số (+0.372 và +0.397) **không**
phải bằng chứng độc lập — chỉ xác nhận hai model cùng họ `gpt-5.6` mắc cùng một
lỗi.

**Quy tắc rút ra:** đồng thuận giữa các model cùng họ chỉ khử được **nhiễu**,
không khử được **thiên lệch hệ thống**. Muốn khử thiên lệch phải có nguồn khác
loại — người gán nhãn, hoặc chính điểm BTC trả về.

---

## 7. Những dự đoán đã sai

Ghi lại để nhận ra kiểu lập luận nào hay hỏng.

| Dự đoán | Thực tế | Nguyên nhân |
|---|---|---|
| "recall ~99%, precision ~55%" | recall không xác định được | lấy biên của vùng khả dĩ rồi đọc như nghiệm duy nhất |
| bộ loại bỏ "~23 span, precision gần tuyệt đối" | **131 span**, precision 65–69% | suy ngược một đại lượng đo thẳng được |
| "hạ `reject_margin` sẽ ăn" | đường cong phẳng, không đổi gì | chưa xem phân bố trước khi đề xuất |
| "`teacher_decides` sẽ tốt hơn" | sẽ emit 4.000–5.000 concept | giả định margin có thang giống nhau ở mọi tập span |
| "nâng ngưỡng có kỳ vọng dương rõ hơn bộ loại" | **lỗ 0.63** | xếp hạng hai lever bằng trực giác thay vì đo |
| "assertion đáng +1.75 đến +2.5" | trần thật **+0.14** | ngoại suy từ phương trình điểm mà không kiểm tra độ nhạy |
| "assertion sẽ được +0.5 đến +0.8" | **+0.047** | tự bác bỏ đúng cảnh báo của chính mình bằng lập luận đồng thuận |

**Mẫu hình chung:** sai ở chỗ ngoại suy từ mô hình mà chưa đo, và ở chỗ trình bày
kết quả biên như thể là nghiệm xác định. Những lần đo thẳng (phân bố margin, quét
ngưỡng cục bộ, so bộ nhãn) đều cho kết luận đúng.

---

## 8. Câu hỏi còn mở

1. **`G` chính xác bằng bao nhiêu?** Chốt được nó sẽ chốt luôn precision và
   recall. Cách rẻ nhất: một lượt nộp với số concept khác hẳn (ví dụ 1.400), rồi
   giải hệ hai phương trình.
2. **Ta bỏ sót loại span nào?** Nhãn giả nói ta chỉ bắt 44% phần lõi, nhưng hạ
   ngưỡng không lấp được — nghĩa là GLiNER **không đề xuất** những span đó ở bất
   kỳ ngưỡng nào. Cần biết chúng trông thế nào.
3. **Hệ số nén 2.16x cho thay đổi span có ổn định không?** Hiện chỉ có **một**
   điểm dữ liệu. Mọi kế hoạch fine-tune sẽ dựa vào nó.
4. **Nhóm đầu bảng (>50 điểm) làm gì khác?** Với `text ≈ 50` thì họ phải đạt
   `M/(G+2S) ≈ 0.53` — tức precision và recall đều cao hơn hẳn. Gần như chắc
   chắn là model NER đã fine-tune, không phải zero-shot.

---

## 9. Trạng thái hiện tại

| | |
|---|---|
| Điểm tốt nhất | **27.5691** (lượt 06) |
| Tham chiếu công khai | 27.8786 |
| Nhóm đầu bảng | hơn 50 |

**Đã hết đường:** vặn ngưỡng (cả hai hướng), `reject_margin`, `teacher_decides`,
assertion (trần +0.14).

**Còn biên độ:** chỉ còn **recall span**, và nó đòi một model NER tốt hơn chứ
không phải một tham số khác. Mọi lever tham số đã đo đều nằm trong dải ±0.6 điểm,
trong khi khoảng cách tới nhóm đầu là ~22 điểm.
