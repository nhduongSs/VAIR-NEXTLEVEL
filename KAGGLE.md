# Chạy self-host trên Kaggle Free GPU

## 1. Kiến trúc được dùng

| Model | Vai trò | Tham số khai báo |
|---|---|---:|
| Qwen3-8B | NER, assertion, rerank candidate | 8,200M |
| multilingual-e5-small | semantic retrieval | 118M |
| Luật/lexical index/validator | offset, alias, schema, ZIP | 0 |
| **Tổng** |  | **8,318M** |

Tổng được tính trên model gốc, không lấy kích thước sau quantization để lách
giới hạn. Qwen được tái sử dụng ở nhiều stage nhưng weights chỉ đóng gói một lần.

## 2. Chuẩn bị Kaggle Dataset

Notebook hoàn chỉnh đã nhúng package `medical_coder`, vì vậy Dataset hiện tại chỉ
cần có:

1. thư mục `input/` gồm `1.txt` đến `100.txt`.

Weights và terminology có thể attach trước để tiết kiệm thời gian, nhưng không
bắt buộc. Nếu không thấy source hợp lệ trong Dataset, notebook tự giải nén source
nhúng vào `/kaggle/working/embedded_viettel_ai_race`; không cần tạo thêm Dataset
source.

Notebook đã được cấu hình ưu tiên input hiện tại:

```text
/kaggle/input/datasets/thanhhiepvo/viettelairace/input
```

Nếu Kaggle mount Dataset bằng alias ngắn như `/kaggle/input/viettelairace`,
notebook sẽ tự dò fallback.

Nếu chưa attach Qwen/E5, đặt Hugging Face token trong Kaggle Add-ons/Secrets với
một trong các tên `HF_TOKEN`, `HF_KEY`, `HUGGINGFACE_TOKEN` hoặc
`HUGGINGFACE_KEY`. Notebook sẽ tải snapshot public về `/kaggle/working/models`,
sau đó xóa token khỏi state và ép Transformers chạy offline. Đây chỉ là tải
weights; inference vẫn hoàn toàn self-host.

Nếu chưa attach knowledge base, notebook tự tải và chuyển đổi:

- ICD-10-CM FY2026 Code Descriptions từ CDC;
- RxNorm Current Prescribable Content ngày 06/07/2026 từ NLM.

Hai archive được khóa checksum và parser từ chối chạy nếu số lượng dưới 50.000
mã ICD hoặc 20.000 RxCUI. KB tự tạo là baseline tiếng Anh chính thức; để tăng
điểm tiếng Việt, vẫn nên attach bản đã bổ sung alias tiếng Việt. File attach
luôn được ưu tiên hơn file tự tải.

Lần chạy đầu cần **Internet On** nếu notebook phải cài wheel, tải weights hoặc
terminology. Sau bước provision, model loader bị khóa `local_files_only=True` và
không gọi inference API. Muốn chạy với Internet Off ngay từ đầu thì phải attach
sẵn toàn bộ dependencies, weights và terminology. Model ID truyền cho CLI luôn
là đường dẫn local dưới `/kaggle/input` hoặc `/kaggle/working`, không phải URL.

## 3. Kiểm tra GPU

```bash
!nvidia-smi
```

Pipeline dùng một worker để không nạp trùng model. Qwen3-8B được nạp NF4 4-bit.
Notebook Run All dùng E5 trên GPU để xây semantic index nhanh hơn và chia sẻ
đúng một E5 giữa ICD/RxNorm. Nếu gặp OOM, đổi `EMBEDDING_DEVICE = "cpu"`.

Tesla P100 có compute capability `sm_60`. Wheel Kaggle hiện tại
`torch 2.10.0+cu128` không chứa kiến trúc này. Notebook tự phát hiện P100 và cài
`torch==2.10.0` từ index CUDA 12.6, sau đó bắt buộc kiểm tra:

```text
CUDA capability: (6, 0)
compiled arches: [..., 'sm_60', ...]
```

Nếu đã chạy notebook cũ hoặc đã `import torch`, chọn **Restart Session** trước
khi Run All notebook mới. Không thể thay binary Torch một cách an toàn trong
kernel đã import nó.

## 4. Kết quả lần nộp đầu tiên

Lần chạy hoàn tất trên Kaggle đã tạo đủ 100 output, được kiểm tra schema/span/ZIP
tại local và đã nộp thành công ngày 27/07/2026. Điểm BTC công bố là **14.4255**
trên 100 record (`WER = 83.5952`, `J_assertion = 20.1874`,
`J_candidates = 8.6197`).

Xem [KET_QUA_NOP_LAN_01.md](KET_QUA_NOP_LAN_01.md) để biết checksum của ZIP,
thống kê output và cấu hình chính xác. Đây là baseline hợp lệ để so sánh các lần
chạy sau.

Để ổn định trên P100, source hiện tại ép Qwen sinh JSON có wrapper, dừng khi JSON
hoàn chỉnh, bỏ riêng entity có nhãn ngoài schema và chia normalization thành batch
2 entity (tự hạ xuống batch 1 nếu OOM). Các thay đổi này không gọi inference API.

## 5. Cài đặt

```bash
%cd "/kaggle/working/Viettel AI Race"
!python -m pip install -q -r requirements-kaggle.txt
```

Không cấu hình `OPENAI_API_KEY`.

## 6. Smoke test

```bash
!medical-coder predict \
  --input-dir /kaggle/input/viettel-test/input \
  --output-dir /kaggle/working/output \
  --cache-dir /kaggle/working/cache \
  --model-path /kaggle/input/qwen3-8b \
  --embedding-model /kaggle/input/multilingual-e5-small \
  --icd-kb /kaggle/input/medical-terminology/icd10.tsv \
  --rxnorm-kb /kaggle/input/medical-terminology/rxnorm.tsv \
  --quantization 4bit \
  --workers 1 \
  --ids 1,2 \
  --overwrite
```

Kiểm tra log, JSON, offset và candidate trước khi chạy toàn bộ.

## 7. Chạy đủ 100 bản ghi

```bash
!medical-coder predict \
  --input-dir /kaggle/input/viettel-test/input \
  --output-dir /kaggle/working/output \
  --cache-dir /kaggle/working/cache \
  --model-path /kaggle/input/qwen3-8b \
  --embedding-model /kaggle/input/multilingual-e5-small \
  --icd-kb /kaggle/input/medical-terminology/icd10.tsv \
  --rxnorm-kb /kaggle/input/medical-terminology/rxnorm.tsv \
  --quantization 4bit \
  --workers 1 \
  --max-candidates 3 \
  --retrieval-top-k 20 \
  --zip-path /kaggle/working/output.zip
```

Notebook dọn output ở đầu lượt chạy. Sau smoke test, full run không cần
`--overwrite`: output cùng cấu hình sẽ được nhận diện và bỏ qua để có thể resume
sau lỗi. Chỉ thêm `--overwrite` khi chủ động tạo lại toàn bộ output. Cache của
response hợp lệ vẫn được tái sử dụng.

## 8. Kiểm tra gói nộp

```bash
!medical-coder validate \
  --input-dir /kaggle/input/viettel-test/input \
  --output-dir /kaggle/working/output \
  --zip-path /kaggle/working/output.zip

!unzip -l /kaggle/working/output.zip | head
```

ZIP cuối chỉ chứa `output/1.json` đến `output/100.json`; cache, provenance,
weights và source không nằm trong `output.zip`.

## 9. Fallback thực tế

- Nếu 4-bit backend lỗi trên GPU được cấp, thử nâng `bitsandbytes`, rồi test lại
  1 record. Không chuyển sang API.
- Nếu Qwen3-8B không thể chạy ổn định, fallback hợp lệ là Qwen3-4B 4-bit/FP16;
  chất lượng có thể thấp hơn nhưng còn nhiều VRAM.
- Không load hai bản Qwen khác nhau vào gói nộp nếu tổng tham số của mọi weights
  được hiểu là cộng dồn.
- Không truncate văn bản để né context; pipeline sẽ dừng nếu vượt giới hạn nhằm
  tránh tạo offset/coverage sai.
