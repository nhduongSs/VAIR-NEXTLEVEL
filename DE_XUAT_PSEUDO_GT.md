# Đề xuất: dùng LLM ngoài để tạo nhãn, và dùng nhãn đó thế nào cho an toàn

## 1. Vấn đề đang chặn

Mỗi lần nộp trả về đúng **một con số**. Bốn lần nộp = bốn bit thông tin. Với tốc
độ đó không thể dò được các tham số như `reject_margin`, ngưỡng GLiNER theo type,
hay chính sách assertion.

Thứ đang thiếu không phải model mạnh hơn — mà là **một thước đo cục bộ**.

## 2. Luật có cho phép không

Có. Mục 5.1 của đề bài:

> Đề bài không cung cấp tập train và **yêu cầu thí sinh sử dụng giải pháp nằm
> ngoài lời giải chính để tạo thêm dữ liệu huấn luyện**.

Mục 17 nói rõ hơn về vai trò: *"LLM chủ yếu dùng cho tạo dữ liệu, reranking hoặc
xử lý ca khó"*. Giới hạn 9B và cấm API ngoài áp lên **pipeline nộp bài**, không
áp lên khâu chuẩn bị dữ liệu.

**Ranh giới tuyệt đối không được vượt:** OpenAI chỉ chạy **offline**, khi ta ngồi
phát triển. Nếu một lượt inference tạo `output.zip` mà có gọi API ngoài thì bài
bị loại. Mọi thứ trong `predict-v2` phải chạy được với Internet Off.

**Nghĩa vụ khai báo:** mục 5.2 bắt buộc bàn giao "code tạo dữ liệu" và "toàn bộ
dữ liệu nhóm sử dụng". Nên script gán nhãn, prompt, và bộ nhãn sinh ra đều phải
nằm trong repo và được nêu trong README. Không giấu.

## 3. Ưu tiên số một: assertion, không phải span

Ba thành phần điểm không giống nhau về mức độ một LLM ngoài giúp được:

| Thành phần | Trọng số | Dư địa | GPT giúp được không |
|---|---:|---:|---|
| **assertion** | 0.3 | +20.2 | **Rất** — phủ định / gia đình / tiền sử là bài toán ngôn ngữ thuần |
| candidates | 0.4 | +31.7 | Vừa — ta đã hơn repo tham chiếu (+1.62) |
| text / span | 0.3 | +20.9 | **Kém** — ranh giới span là *quy ước* của BTC, GPT đoán mò như ta |

Hiện ta emit **zero assertion** và được 33.07. Đó là nước đi đúng khi mù, nhưng
nó bỏ trắng toàn bộ các concept thật sự có assertion.

### Ngưỡng hoà vốn

Trên một concept đã khớp:

* GT rỗng, ta emit rỗng → Jaccard **1.0** (đang được miễn phí)
* GT rỗng, ta emit gì đó → **0** (mất trắng)
* GT có nhãn, ta emit rỗng → **0** (đang mất trắng)
* GT có nhãn, ta emit đúng tập → **1.0**

Gọi `pi` là precision của bộ phát hiện "concept này có assertion", `e` là độ đúng
của *tập* nhãn khi đã biết là có. Có lời khi:

```text
pi * e > 1 - pi      <=>      pi > 1 / (1 + e)
```

| `e` | precision tối thiểu |
|---|---|
| 1.0 | 50.0% |
| 0.9 | 52.6% |
| 0.8 | 55.6% |
| 0.7 | 58.8% |

Ngưỡng này **thấp**. Chỉ cần chắc hơn 56% là đã có lời.

### Lợi ích tối đa

Nếu bắt đúng hết các concept có assertion, với `q` là tỉ lệ concept assertable
trong GT có nhãn rỗng:

| `q` | assertion | Δ điểm tổng |
|---|---|---|
| 0.80 | 33.07 → 41.33 | **+2.48** |
| 0.85 | 33.07 → 38.90 | **+1.75** |
| 0.90 | 33.07 → 36.74 | **+1.10** |

So với khoảng cách còn lại tới repo tham chiếu là 0.357 điểm, đây là lever lớn
nhất còn chưa động tới.

## 4. Kiến trúc: GPT không được nằm trong đường chạy

Đây là chỗ dễ làm sai nhất.

```text
OFFLINE (một lần, có OpenAI)          INFERENCE (nộp bài, không Internet)
─────────────────────────────         ──────────────────────────────────
GPT gán assertion trên span của ta
        ↓
bộ nhãn assertion (~2500 concept)
        ↓
chọn ngưỡng cho classifier Qwen  ───→  Qwen (đã có sẵn) chấm assertion
        ↓                               bằng next-token logits, tự host
ngưỡng + prompt được cố định            ↓
                                        emit assertion khi vượt ngưỡng
```

Qwen teacher **đã chạy sẵn** trên Kaggle cho corrector và bộ loại span. Thêm một
vai trò assertion dùng đúng cơ chế logits đó là gần như miễn phí về hạ tầng, và
hoàn toàn self-host.

GPT chỉ làm đúng một việc: cho ta biết **đặt ngưỡng ở đâu**.

## 5. Quy trình kỹ thuật

### 5.1. Gán nhãn

Đầu vào cho mỗi concept: mention + **dòng chứa nó + dòng tiêu đề mục gần nhất**
(dùng lại `line_context()` đã có trong `selector.py`).

Đầu ra: `{"assertions": ["isNegated", ...]}` — chỉ thế.

Không hỏi GPT về position. Không hỏi về span. Không hỏi về mã ICD.

### 5.2. Không tin position do model trả về

Nếu sau này mở rộng sang gán nhãn cả span, mọi offset phải đi qua
`alignment.py` rồi `validation.py`, đúng như pipeline hiện tại. Ràng buộc
`raw[start:end] == text` không bao giờ được nới.

### 5.3. Kiểm định trước khi tin

Đã có sẵn [`tools/calibrate_pseudo_gt.py`](tools/calibrate_pseudo_gt.py):

1. **cấu trúc** — số concept phải nằm trong vùng `G ≈ 1150–1600` giải ngược từ
   điểm thật; lệch xa là sai ngay, khỏi chấm tiếp;
2. **tái lập điểm** — chấm lại các lần nộp đã biết điểm;
3. **bảo toàn thứ tự** — quan trọng nhất, vì ta dùng nhãn để *chọn* phương án
   chứ không phải để *đo*.

Hỏng phép 3 thì bộ nhãn vô dụng dù phép 2 đẹp đến đâu: tối ưu theo nó sẽ đẩy
pipeline đi sai hướng một cách êm ái, mỗi vòng đều thấy "cải thiện" mà điểm thật
đứng yên.

## 6. Chi phí

100 bản ghi, ~2500 concept, mỗi concept một đoạn ngữ cảnh ngắn. Ước tính
150–250K token đầu vào cho một lượt gán nhãn đầy đủ. Vài đô la. Không phải yếu tố
cần cân nhắc.

## 7. Bảo mật khoá

* khoá đặt trong biến môi trường `OPENAI_API_KEY`, **không** viết vào file nào
  trong repo;
* `.env` đã nằm trong `.gitignore`;
* script gán nhãn đọc từ môi trường và dừng với thông báo rõ nếu thiếu;
* bộ nhãn sinh ra thì commit (cần cho bàn giao), khoá thì không.

## 8. Điều kiện dừng

Bỏ hướng này nếu:

* bộ nhãn hỏng phép 3 (xếp sai thứ tự) sau hai lần sửa prompt;
* precision assertion do GPT ước tính dưới 56% — dưới ngưỡng hoà vốn;
* một lần nộp thật cho kết quả ngược dấu với dự đoán của nhãn giả.

Điều kiện cuối là quan trọng nhất: **mọi kết luận từ nhãn giả phải được xác nhận
bằng một lần nộp thật** trước khi xây tiếp lên nó.

## 9. Cần từ phía bạn

1. `OPENAI_API_KEY` (đặt vào môi trường, không dán vào chat cũng được — tôi sẽ
   viết script đọc từ env).
2. **`output.zip` của lần nộp 03 và 04** đang nằm trên Kaggle. Không có chúng thì
   phép kiểm 2 và 3 không chạy được, mà đó chính là phần khiến toàn bộ hướng này
   an toàn thay vì tự huyễn hoặc.

## 10. Thứ tự thực hiện

| Bước | Việc | Cần gì |
|---|---|---|
| 1 | lưu output lần 03, 04 vào `submissions/` | bạn tải từ Kaggle |
| 2 | script gán nhãn assertion + prompt | key |
| 3 | chạy kiểm định 3 phép | bước 1 + 2 |
| 4 | thêm vai trò assertion cho Qwen teacher | — |
| 5 | chọn ngưỡng theo nhãn giả | bước 3 + 4 |
| 6 | nộp thử, đối chiếu dự đoán với thực tế | — |

Bước 1 làm được ngay và không tốn gì. Bước 6 là bước quyết định hướng này sống
hay chết.
