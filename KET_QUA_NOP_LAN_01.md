# Kết quả nộp lần 01 — Viettel AI Race

## 1. Mốc nộp và điểm chính thức

| Trường | Giá trị |
|---|---|
| Thời điểm nộp | 27/07/2026, 18:02 (GMT+7) |
| Trạng thái trên hệ thống chấm | Đã chấm |
| Điểm tổng | **14.4255** |
| `num_records` | 100 |
| `num_scored` | 100 |
| `WER` | 83.5952 |
| `J_assertion` | 20.1874 |
| `J_candidates` | 8.6197 |

Các giá trị trên được chép nguyên trạng từ trang kết quả của Ban Tổ chức. Đây là
mốc baseline của lần nộp đầu tiên; không suy diễn công thức chấm chi tiết ngoài
những metric đã được hệ thống công bố.

## 2. Cấu hình code đã dùng

Pipeline chạy self-host hoàn toàn trên Kaggle Free GPU, không gọi OpenAI API hay
Hugging Face Inference API.

| Thành phần | Cấu hình thực thi |
|---|---|
| GPU | Tesla P100-PCIE-16GB, compute capability `sm_60` |
| PyTorch | `2.10.0+cu126`, có kernel `sm_60` |
| LLM | `Qwen/Qwen3-8B`, NF4 4-bit |
| Retriever | `intfloat/multilingual-e5-small` |
| Tổng weights kê khai | 8.318B / giới hạn giả định 9B |
| Terminology | ICD-10-CM FY2026 và RxNorm Current Prescribable Content 07/2026 |
| Candidate pool | `retrieval_top_k=20`, xuất tối đa 3 mã/entity |

Mã nguồn hiện thực trong `src/medical_coder/` và notebook
`notebooks/Viettel_AI_Race_Kaggle_Run_All.ipynb` đã bao gồm các bảo vệ cần thiết
cho P100/Kaggle:

- source fallback nhúng trực tiếp trong notebook;
- tự thay wheel CUDA 12.8 không hỗ trợ P100 bằng PyTorch CUDA 12.6;
- ép Qwen tiếp tục JSON với wrapper `entities`/`mappings`, sau đó dừng khi JSON
  ngoài cùng hoàn chỉnh;
- chuẩn hóa các biến thể JSON thông dụng của model và loại riêng entity có nhãn
  ngoài năm nhãn hợp lệ, thay vì làm hỏng cả record;
- retrieval chỉ chạy local, candidate được lọc theo knowledge base;
- normalization xử lý theo batch 2 entity, tự hạ xuống batch 1 khi gặp CUDA OOM;
- deterministic alignment, schema validation và kiểm tra ZIP trước khi nộp.

## 3. Artifact đã nộp và kiểm chứng local

Artifact nộp là [output.zip](output.zip), được tạo từ thư mục
[output/](output/). Đợt kiểm tra local sau khi hoàn tất xác nhận:

- có đủ chính xác 100 file `output/1.json` đến `output/100.json`;
- ZIP chứa đúng 100 member theo thứ tự số, không có cache/source/weights;
- tất cả JSON tuân thủ schema, type/assertion/candidate hợp lệ và mọi span thỏa
  `raw_text[start:end] == text`;
- ZIP đọc được, không có member hỏng.

| Artifact | Giá trị |
|---|---|
| Kích thước ZIP | 82,924 bytes (xấp xỉ 81 KB) |
| SHA-256 ZIP | `c4f4689b8f8db11227f27c56f3b2b7bff8a906a10c4b6bf1c217c7ad4c6a242f` |
| SHA-256 notebook | `e8b56c29d42bb5fee586b2774a2ab62f73e4a75d8fef07a5e21160ee3e6616e5` |
| SHA-256 model manifest | `86a29eadc35eb73201c7986a12380846888f51bb498bcee3933fb2711fa22206` |

## 4. Thống kê output lần nộp

| Chỉ số | Giá trị |
|---|---:|
| Tổng entity xuất ra | 2,266 |
| Trung bình entity/bản ghi | 22.66 |
| Nhỏ nhất / lớn nhất | 0 / 66 |
| Bản ghi rỗng | `75`, `96` |
| TRIỆU_CHỨNG | 869 |
| TÊN_XÉT_NGHIỆM | 425 |
| KẾT_QUẢ_XÉT_NGHIỆM | 297 |
| CHẨN_ĐOÁN | 377 |
| THUỐC | 298 |
| `isNegated` | 78 |
| `isHistorical` | 176 |
| `isFamily` | 12 |

Trường `candidates` có mặt đúng ở 377 chẩn đoán và 298 thuốc. Trong đó 43 chẩn
đoán và 145 thuốc có ít nhất một candidate không rỗng, với tổng cộng 98 mã ICD
và 362 RxCUI được xuất theo lượt candidate. Điều này nhất quán với `J_candidates = 8.6197`: candidate
normalization là điểm nghẽn lớn nhất cần ưu tiên ở lần nộp tiếp theo.

Hai record rỗng vẫn là JSON list hợp lệ và đã được hệ thống chấm đủ 100 record;
chúng là nhóm cần xem lại đầu tiên khi cải thiện recall.

## 5. Diễn giải để tối ưu lần nộp sau

Điểm lần đầu xác nhận pipeline đã hoàn thành end-to-end và được BTC chấm hợp lệ,
nhưng đây mới là baseline. Thứ tự ưu tiên cải thiện nên là:

1. kiểm tra hai output rỗng và các record có số entity bất thường;
2. tăng recall/span typing của NER, đặc biệt với các mention bị alignment loại;
3. làm giàu alias tiếng Việt, tên thương mại, hoạt chất và viết tắt cho ICD/RxNorm;
4. cải thiện candidate coverage trước, sau đó tune số candidate/threshold theo
   `J_candidates`, không đơn giản trả thêm mã;
5. rà assertion ở các record có cue phủ định, tiền sử và quan hệ gia đình.

Mọi lần chạy sau cần giữ lại ZIP, checksum, score và cấu hình để so sánh công
bằng với baseline này.
