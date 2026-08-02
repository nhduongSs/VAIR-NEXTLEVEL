# Phân tích lời giải 27.8786 và cải tiến pipeline

Tài liệu này ghi lại kết quả nghiên cứu repository công khai
[`duongtruongbinh/viettel_ai_race_task2`](https://github.com/duongtruongbinh/viettel_ai_race_task2)
(điểm BTC **27.8786**) và các thay đổi đã áp dụng vào lời giải của chúng ta.

Baseline của ta: **14.4255**.

## 1. Giải mã công thức chấm

Điểm công bố của cả hai lời giải khớp chính xác với công thức:

```text
final = 0.3 × text_score + 0.3 × assertions_score + 0.4 × candidates_score
```

| Lời giải | text | assertions | candidates | final |
|---|---:|---:|---:|---:|
| Ta, lần nộp 01 | 16.4048 | 20.1874 | 8.6197 | **14.4255** |
| Repo tham chiếu (`improved_v2`) | 32.1820 | 35.2687 | 19.1084 | **27.8786** |

Hệ quả quan trọng: trường `WER = 83.5952` trên trang kết quả là **tỷ lệ lỗi**,
không phải điểm. `text_score = 100 − WER = 16.4048`. Trước đây tài liệu của ta
trình bày 83.5952 như thành phần mạnh nhất; thực tế đó là thành phần **yếu nhất**.

## 2. Cơ chế quyết định: spurious concept bị đếm hai lần

Repo tham chiếu có scorer nội bộ. Điểm mấu chốt là mỗi concept dự đoán **không
khớp** ground truth bị cộng **hai lần** vào mẫu số của cả ba thành phần:

```text
text       = Σ_matched (1 − WER) / (n_truth + 2 × n_spurious)
assertions = Σ_matched J_assert  / (n_truth_assertable + 2 × n_spurious_assertable)
candidates = Σ_matched J_cand × w / (Σ_truth w + 2 × Σ_spurious w),  w = n_codes + 1
```

Hai hệ quả trực tiếp:

1. **Concept thừa đắt gấp đôi concept thiếu.** Precision quan trọng hơn recall.
2. **Candidate thừa còn đắt hơn nữa.** Một concept spurious mang 3 mã tốn
   `2 × (3+1) = 8` đơn vị mẫu số, thay vì 2 nếu không mang mã nào.

Thêm vào đó, khi ground truth của một concept có `candidates` rỗng thì dự đoán
rỗng đạt Jaccard 1.0 còn dự đoán bất kỳ mã nào đạt 0.0.

Ta đã port scorer này thành `medical_coder.scoring`; nó tự chấm ground truth
bằng 1.0 và tái lập chính xác cả hai mốc điểm công bố ở Mục 1.

## 3. Những gì repo tham chiếu làm khác

| Hạng mục | Lần nộp 01 của ta | Repo tham chiếu |
|---|---|---|
| NER | Qwen3-8B tự sinh mention dạng JSON | GLiNER zero-shot, có score từng span |
| Ngưỡng | không có | ngưỡng riêng theo từng type |
| Assertion | emit 266 nhãn | **để rỗng hoàn toàn** |
| Candidate | LLM rerank, tối đa 3 mã | chỉ emit khi alias trùng khớp duy nhất, tối đa 1 mã |
| ICD KB | CDC ICD-10-CM (tiếng Anh) | danh mục ICD-10 tiếng Việt TT06/2026 của Bộ Y tế |

Ablation đáng chú ý nhất trong `docs/04_findings.md` của họ: **bỏ toàn bộ
candidate chỉ làm candidate Jaccard giảm 0.0036**. Nghĩa là 40% trọng số của
metric gần như hoàn toàn được quyết định bởi chất lượng khớp concept và mức độ
tiết chế khi emit mã, chứ không phải bởi việc tra đúng mã.

## 4. Thay đổi đã áp dụng

### 4.1. Thay NER sinh văn bản bằng GLiNER (`predict-v2`)

Kiến trúc cũ bắt Qwen viết lại từng mention rồi căn ngược về văn bản gốc. Mọi
mention bị model diễn giải lại đều bị bước alignment loại bỏ, và mention sinh ra
không có score nên không có nút vặn precision/recall.

GLiNER trả offset ký tự kèm score, nên `text[start:end]` luôn đúng nguyên văn và
có thể đặt ngưỡng riêng cho từng type.

Phân bố concept thay đổi rõ rệt và bám sát repo tham chiếu:

| Type | Lần nộp 01 | `predict-v2` | Repo tham chiếu |
|---|---:|---:|---:|
| TRIỆU_CHỨNG | 869 | 1,348 | 1,405 |
| CHẨN_ĐOÁN | 377 | 529 | 668 |
| THUỐC | 298 | 349 | 346 |
| TÊN_XÉT_NGHIỆM | 425 | 253 | 402 |
| KẾT_QUẢ_XÉT_NGHIỆM | 297 | 38 | 123 |
| Bản ghi rỗng | 2 | 0 | — |

### 4.2. Knowledge base ICD-10 tiếng Việt

Nguồn: *Phụ lục Bảng danh mục mã ICD-10 tiếng Việt* ban hành kèm Thông tư
06/2026/TT-BYT, đã đưa vào `data/kb/raw/`. KB cũ là bản tiếng Anh của CDC nên
một mention như `viêm túi mật` về nguyên tắc không thể khớp alias nào.

Builder của ta lấy thêm một cột mà repo tham chiếu bỏ qua: cột synonym
`HƯỚNG DẪN MÃ HÓA BỔ SUNG CỦA WHO 2019`. Cột này lẫn lộn giữa synonym sạch
(`Bệnh tả cổ điển`) và ghi chú bao hàm/loại trừ, nên `iter_who_synonyms` chỉ giữ
phần synonym. Kết quả: +2,866 alias, tỷ lệ khớp duy nhất trên mention chẩn đoán
của ta tăng từ 9.0% lên 10.9%.

### 4.3. Liên kết mã theo hướng precision-first

Chỉ emit mã khi mention khớp **đúng một** alias; tối đa 1 mã; mã ICD 3 ký tự
được remap sang nhánh `.9` khi nhánh đó tồn tại. Số mã xuất ra giảm từ 460 (trên
188 concept) xuống 163 (trên 163 concept).

### 4.4. Assertion để rỗng

Đây là quyết định theo số liệu chứ không phải bỏ sót: repo tham chiếu đo được
`isNegated` tách biệt ở AUC 0.497 — ngang mức ngẫu nhiên — và mọi rule họ thử đều
emit thừa. Ta emit 266 nhãn và được 20.19; họ không emit nhãn nào và được 35.27.
Vì vậy `predict-v2` không có cờ bật/tắt nửa vời cho phần này; muốn khôi phục thì
phải có dữ liệu validation có nhãn trước.

### 4.5. RxNorm: loại bỏ term type SCDC

Khi dựng KB thuốc lần đầu, ta index cả SCDC (hoạt chất + hàm lượng). Kiểm tra
trên ví dụ chính thức cho thấy đây là lỗi: mọi mã gold trong ví dụ đều là **SCD**
(`amlodipine 10 mg po daily` → 308135 “amlodipine 10 MG Oral Tablet”), trong khi
`329526` là SCDC “amlodipine 10 MG”. Mention sau khi bỏ route/tần suất khớp đúng
bề mặt SCDC, nên việc index SCDC biến “không trả lời” thành “trả lời sai một cách
tự tin” đúng ở nhóm mention nhiều khả năng có mã gold nhất.

SCDC bị loại theo **concept**, không theo dòng, vì một RxCUI dạng SCDC vẫn có các
dòng TMSY khác.

### 4.6. Hướng đã thử và loại bỏ

Ta thử nối SCDC → SCD qua `RXNREL` để cứu các mention có hàm lượng, chỉ emit khi
SCDC dẫn tới đúng một SCD. Trên toàn bộ RxNorm thì 74.8% bề mặt SCDC là duy nhất,
nhưng đúng những thuốc phổ biến trong dữ liệu lại đa trị (`amlodipine 10 mg` ứng
với 5 SCD). Cách này chỉ cứu được 4/349 mention nên đã bị loại.

Kết luận: mention thuốc có hàm lượng không thể phân giải duy nhất bằng từ vựng.
Điều này nhất quán với conditional code precision 0.231 của repo tham chiếu.

## 4.7. Selector chạy GPU: corrector và consensus additions

`medical_coder.selector` port hai vai trò còn thiếu, cả hai quyết định bằng
**next-token logits trên tập nhãn cố định**, không sinh văn bản tự do. Khác biệt
này quan trọng: một câu trả lời được sinh ra có thể là bất cứ thứ gì, còn logit
trên các chữ số 0-5 là một quyết định đóng, không thể phá schema.

* **Corrector** — với mọi span `TRIỆU_CHỨNG`, đọc logit 5 chiều và đổi sang
  `CHẨN_ĐOÁN` khi teacher không đồng ý. Đây là lỗi type lớn nhất của GLiNER (đọc
  bệnh mạn tính thành triệu chứng) và là nguyên nhân ta chỉ có 529 chẩn đoán so
  với 668 của tham chiếu.
* **Additions** — với span điểm thấp không chồng lấn baseline, đọc logit 6 chiều
  (0 = không phải khái niệm) ở **cả hai** teacher; chỉ thêm khi hai teacher cùng
  chọn một type, type đó **không mang candidate**, và cả hai vượt margin so với
  `NONE`.

Additions bắt buộc đồng thuận còn correction thì không: correction chỉ đổi nhãn
của span đã tồn tại, còn addition tạo ra span mới — chỉ cái sau mới sinh được
concept thừa.

Ngân sách tham số (đếm trên weights gốc, quantization không làm giảm số kê khai):

| Model | Vai trò | Tham số |
|---|---|---:|
| `urchade/gliner_multi-v2.1` | NER | 0.289B |
| `Qwen/Qwen3-4B-Instruct-2507` | teacher chính | 4.022B |
| `Qwen/Qwen3.5-4B` | teacher phụ | 4.206B |
| **Tổng** | | **8.517B** |

Teacher phụ là tuỳ chọn; thiếu nó thì chạy corrector-only (4.311B) và bỏ
additions. Phương án một teacher mạnh: `Qwen3-8B` + GLiNER = 8.489B, vẫn dưới 9B.
`build_selector` sẽ dừng chương trình nếu tổng vượt 9B.

## 4.8. Loại header và cắt tiền tố chung

Thay bộ regex tự chế bằng cách của repo tham chiếu: chỉ loại span khi văn bản
**bằng đúng** một nhãn mục (`tiền sử`, `chẩn đoán`, `kết quả xét nghiệm`…). Đây
là phép so bằng chứ không phải blacklist chuỗi con, vì chính những từ đó nằm
trong một mention dài hơn (`kết quả xét nghiệm glucose cao`) vẫn là khái niệm
thật. Kèm theo `trim_generic_prefix` cắt `dấu hiệu/biểu hiện/tình trạng/hội
chứng` khi phần còn lại vẫn đủ hai từ — đây là lever boundary duy nhất mà repo
tham chiếu giữ lại sau ablation.

## 5. Cách chạy

```bash
python -m pip install -e '.[v2]'
```

```bash
medical-coder predict-v2 \
  --input-dir input \
  --output-dir output_v2 \
  --icd-kb data/terminology/icd10_vn.tsv \
  --rxnorm-kb data/terminology/rxnorm.tsv \
  --zip-path output_v2.zip
```

Toàn bộ 100 bản ghi chạy khoảng 55 giây trên CPU, không cần GPU.

Bản đầy đủ có selector (cần GPU):

```bash
medical-coder predict-v2 \
  --input-dir input --output-dir output \
  --icd-kb data/terminology/icd10_vn.tsv \
  --rxnorm-kb data/terminology/rxnorm.tsv \
  --device cuda \
  --primary-teacher Qwen/Qwen3-4B-Instruct-2507 \
  --secondary-teacher Qwen/Qwen3.5-4B \
  --teacher-quantization 4bit \
  --zip-path output.zip
```

Trên Kaggle dùng notebook
[`notebooks/Viettel_AI_Race_Kaggle_Predict_V2.ipynb`](notebooks/Viettel_AI_Race_Kaggle_Predict_V2.ipynb):
tự dò source/input đã attach, xử lý wheel `sm_60` cho P100, dựng cả hai KB, chạy
smoke test 2 bản ghi trước, rồi chạy đủ 100 và đóng gói ZIP. Teacher phụ được tải
theo kiểu chấp nhận thất bại — nếu không có thì tự động chạy corrector-only thay
vì hỏng cả lượt chạy.

Dựng lại KB:

```bash
python -m pip install -e '.[kb]'
medical-coder build-icd-kb
```

RxNorm không được phân phối kèm repo; tải
`RxNorm_full_prescribe_<ngày>.zip` từ NLM rồi dựng bằng
`medical_coder.rxnorm_kb.build`.

## 6. Việc còn lại

1. **Gán nhãn 15–20 bản ghi** để `medical-coder score` hoạt động. Hiện chưa có
   ground truth nên mọi thay đổi vẫn phải suy luận gián tiếp; đây là việc chặn
   đường mọi tuning tiếp theo.
2. **Type corrector TRIỆU_CHỨNG → CHẨN_ĐOÁN.** Đây là lỗi type lớn nhất của
   GLiNER và giải thích khoảng cách 529 với 668 ở nhóm chẩn đoán. Repo tham chiếu
   dùng next-token logits của Qwen3-4B trên tập nhãn cố định, không sinh tự do.
3. **Bổ sung span cho type không có candidate** (`TÊN_XÉT_NGHIỆM`,
   `KẾT_QUẢ_XÉT_NGHIỆM`), nơi ta còn thấp hơn tham chiếu rõ rệt (253 với 402, và
   38 với 123). Họ yêu cầu hai teacher đồng thuận trước khi thêm.
4. **Tune lại ngưỡng theo dữ liệu của ta.** Ngưỡng hiện tại lấy nguyên từ repo
   tham chiếu và được chọn khi *đã có* selector, nên chưa chắc tối ưu cho cấu
   hình không selector của ta — đặc biệt `KẾT_QUẢ_XÉT_NGHIỆM = 0.35`.
