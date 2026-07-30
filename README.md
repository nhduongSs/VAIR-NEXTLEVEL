# Viettel AI Race — Clinical Concept Extraction

Pipeline self-host để:

1. phát hiện 5 loại khái niệm y khoa;
2. phân loại `isNegated`, `isFamily`, `isHistorical`;
3. căn span chính xác trên văn bản gốc theo `[start, end)`;
4. truy hồi và ánh xạ chẩn đoán sang ICD-10, thuốc sang RxNorm/RxCUI;
5. kiểm tra schema và tạo đúng `output.zip`.

Giải pháp chính thức **không dùng OpenAI API hoặc API inference bên ngoài**.

## Kết quả nộp lần đầu

Lần nộp đầu tiên ngày 27/07/2026 đã được hệ thống của BTC chấm đủ 100/100
record, đạt **14.4255**. Các metric được công bố là `WER = 83.5952`,
`J_assertion = 20.1874` và `J_candidates = 8.6197`.

`WER = 83.5952` là **tỷ lệ lỗi**, không phải điểm: `text_score = 100 − WER =
16.4048`. Kiểm chứng bằng chính công thức của đề bài:

```text
0.3 × 16.4048 + 0.3 × 20.1874 + 0.4 × 8.6197 = 14.4255
```

Nói cách khác `text` là thành phần **yếu nhất**, không phải mạnh nhất.

Gói [output.zip](output.zip) trong workspace là artifact của lần nộp này và đã
được kiểm tra lại đủ 100 JSON, schema/span hợp lệ, ZIP không lỗi. Báo cáo đầy đủ
về cấu hình, checksum, phân bố output và hướng cải thiện có trong
[KET_QUA_NOP_LAN_01.md](KET_QUA_NOP_LAN_01.md).

## Pipeline `predict-v2` (precision-first, chạy CPU)

Sau khi nghiên cứu một lời giải công khai đạt 27.8786, workspace có thêm một
pipeline thứ hai không dùng sinh văn bản:

```text
raw text
→ GLiNER spans (ngưỡng riêng theo type, offset nguyên văn)
→ Qwen corrector: TRIỆU_CHỨNG → CHẨN_ĐOÁN            (tuỳ chọn, cần GPU)
→ Qwen consensus additions cho type không có candidate (tuỳ chọn, cần 2 teacher)
→ trim generic prefix + loại header
→ exact-alias linking (ICD-10 tiếng Việt TT06 + RxNorm), chỉ emit khi khớp duy nhất
→ assertions rỗng
→ validator → output.zip
```

Không có GPU thì bỏ hai bước Qwen; đủ 100 bản ghi chạy khoảng 55 giây trên CPU:

```bash
python -m pip install -e '.[v2]'
medical-coder predict-v2 \
  --input-dir input \
  --output-dir output_v2 \
  --icd-kb data/terminology/icd10_vn.tsv \
  --rxnorm-kb data/terminology/rxnorm.tsv \
  --zip-path output_v2.zip
```

Bản đầy đủ trên Kaggle (GPU): notebook
[Viettel_AI_Race_Kaggle_Predict_V2.ipynb](notebooks/Viettel_AI_Race_Kaggle_Predict_V2.ipynb).
Tổng tham số kê khai 8.517B (GLiNER 0.289B + hai teacher 4B), dưới giới hạn 9B;
`build_selector` dừng chương trình nếu vượt.

Lý do của từng lựa chọn — và các hướng đã thử rồi loại — nằm ở
[PHAN_TICH_DOI_THU_VA_CAI_TIEN.md](PHAN_TICH_DOI_THU_VA_CAI_TIEN.md).

## Chấm điểm cục bộ

`medical_coder.scoring` hiện thực lại công thức của BTC, bao gồm quy tắc đếm
**hai lần** mỗi concept thừa. Scorer tự chấm ground truth bằng 1.0 và tái lập
đúng cả hai mốc điểm đã công bố.

```bash
medical-coder score --output-dir output_v2 --truth-dir data/labelled --per-record
```

Cần có ground truth tự gán nhãn; chưa gán nhãn thì chưa tune được gì đáng tin.

## Kiến trúc dưới 9B

| Thành phần | Vai trò | Tham số khai báo |
|---|---|---:|
| `Qwen/Qwen3-8B` | NER, assertion và rerank candidate | 8,200M |
| `intfloat/multilingual-e5-small` | retrieval đa ngôn ngữ | 118M |
| Luật + lexical index + validator | offset/schema/ZIP | 0 |
| **Tổng** |  | **8,318M** |

Chi tiết có trong [model manifest](config/model_manifest.json). Quantization 4-bit
chỉ giảm VRAM, không làm giảm số tham số được kê khai. Cùng một Qwen được dùng ở
nhiều stage nhưng chỉ có một bộ weights.

Pipeline:

```text
raw text
→ Qwen3-8B trích mention + assertion
→ deterministic alignment trên raw text
→ lexical + E5 retrieval từ ICD/RxNorm local
→ Qwen3-8B chỉ chọn trong code đã truy hồi
→ allowlist + schema validator
→ output.zip
```

Không cho LLM tự sinh mã ngoài knowledge base. Đây là khác biệt quan trọng so
với baseline API trước đây.

## Cài đặt

Core và test:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Inference local:

```bash
python -m pip install -e '.[local]'
```

Trên Kaggle dùng:

```bash
python -m pip install -r requirements-kaggle.txt
```

Không cần `.env` hay API key.

## Knowledge base bắt buộc để có điểm candidate

Chuẩn bị:

```text
data/terminology/icd10.tsv
data/terminology/rxnorm.tsv
```

Header tối thiểu:

```tsv
code	label	aliases
K21.9	Gastro-esophageal reflux disease without esophagitis	trào ngược dạ dày thực quản|GERD
```

Xem [quy cách terminology](data/terminology/README.md). Nếu thiếu một KB,
pipeline vẫn sinh JSON hợp lệ nhưng `candidates` của loại tương ứng sẽ rỗng.
Đó chỉ là fallback kỹ thuật, không phải cấu hình cạnh tranh.

## Chạy thử

```bash
medical-coder predict \
  --input-dir input \
  --output-dir output \
  --model-path /path/to/Qwen3-8B \
  --embedding-model /path/to/multilingual-e5-small \
  --icd-kb data/terminology/icd10.tsv \
  --rxnorm-kb data/terminology/rxnorm.tsv \
  --quantization 4bit \
  --workers 1 \
  --ids 1,2 \
  --overwrite
```

Chạy subset không tạo ZIP.

## Chạy toàn bộ

```bash
medical-coder predict \
  --input-dir input \
  --output-dir output \
  --cache-dir .medical_coder_cache \
  --model-path /path/to/Qwen3-8B \
  --embedding-model /path/to/multilingual-e5-small \
  --icd-kb data/terminology/icd10.tsv \
  --rxnorm-kb data/terminology/rxnorm.tsv \
  --quantization 4bit \
  --workers 1 \
  --max-candidates 3 \
  --retrieval-top-k 20 \
  --overwrite \
  --zip-path output.zip
```

Kết quả:

```text
output.zip
└── output/
    ├── 1.json
    ├── ...
    └── 100.json
```

Hướng dẫn chi tiết cho Kaggle: [KAGGLE.md](KAGGLE.md).

Notebook `Run All` sẵn sàng upload lên Kaggle:

```text
notebooks/Viettel_AI_Race_Kaggle_Run_All.ipynb
```

Notebook tự dò các Dataset đã attach, chạy smoke test, chạy đủ 100 record,
validate lại ZIP và hiển thị liên kết tải `/kaggle/working/output.zip`.

## Cache và chống trộn output

Response JSON hợp lệ được cache theo nội dung input, model, prompt và candidate
được truy hồi. Mỗi output còn có provenance riêng trong `output/.provenance/`.

Nếu output cũ đến từ model/KB khác hoặc pipeline API cũ, chương trình dừng thay
vì bỏ qua rồi tạo một submission trộn. Dùng `--overwrite` để tái tạo. Thư mục
provenance không được đưa vào ZIP.

Mention LLM trả về nhưng không căn được vào văn bản gốc bị loại và ghi ở:

```text
.medical_coder_cache/<id>.alignment_issues.json
```

## Kiểm tra output

```bash
medical-coder validate \
  --input-dir input \
  --output-dir output \
  --zip-path output.zip
```

Validator kiểm tra:

- đủ tệp tương ứng input;
- JSON top-level là list;
- type/assertion/candidate đúng điều kiện;
- ICD/RxCUI đúng định dạng và allowlist nếu được truyền;
- entity được sắp theo position và không trùng;
- `raw_text[start:end] == text`;
- ZIP đọc lại được và member nằm dưới `output/`.

## Các tùy chọn chính

```text
--model-path PATH           snapshot model self-host
--quantization 4bit         cấu hình phù hợp GPU Kaggle 16 GB
--embedding-model PATH      E5 local; dùng "none" để chỉ lexical
--embedding-device cpu      tránh tranh VRAM với Qwen
--icd-kb PATH               terminology ICD local
--rxnorm-kb PATH            terminology RxNorm local
--retrieval-top-k 20        pool mã hợp lệ đưa vào reranker
--max-candidates 3          số mã tối đa xuất ra
--ids 1,2,3                 smoke test
--overwrite                 tạo lại output của cùng một cấu hình
--no-zip                    không tạo output.zip
```

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Test không load model và không dùng mạng.

## Lộ trình tăng điểm

Ưu tiên theo ảnh hưởng đến metric:

1. tạo validation có nhãn thủ công trên 10–20 record đại diện;
2. làm giàu alias ICD/RxNorm tiếng Việt, tên thương mại và viết tắt;
3. tune ngưỡng candidate theo Jaccard thay vì luôn trả top-k;
4. đánh giá lỗi span/type/assertion theo từng section;
5. sau khi có dữ liệu nhãn, fine-tune XLM-R-base đa nhiệm và chỉ thêm weights
   nếu ensemble thực sự tăng điểm. Khi đó tổng dự kiến khoảng 8,60B, vẫn dưới
   giả thuyết 9B;
6. cố định snapshot/checksum của model và KB trước khi bàn giao.

Không nên fine-tune Qwen3-8B ngay khi chưa có validation đáng tin cậy; với Kaggle
Free, đầu tư vào retrieval, weak supervision và calibration thường hiệu quả hơn
so với QLoRA mù.
