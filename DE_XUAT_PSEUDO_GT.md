# Đề xuất: tạo nhãn để fine-tune, KHÔNG dùng API ngoài

## 1. Đính chính về luật

Trước đó tôi kết luận "dùng OpenAI để tạo nhãn là được phép" và dẫn mục 5.1. Đọc
lại thì kết luận đó **không đứng vững**.

Trong toàn bộ tài liệu, phần duy nhất được đánh dấu là lời Ban Tổ chức nằm ở mục
5.3:

> Thí sinh tự chuẩn bị tài nguyên tính toán. Tuy nhiên, với những giải pháp
> LLM/agent chỉ cho phép thí sinh self-host model mà không được sử dụng API
> ngoài, model self-host có độ lớn tối đa là 9B params.

Câu này **không nói rõ** lệnh cấm API chỉ áp lên khâu inference.

Hai câu tôi từng dựa vào đều **không phải** lời BTC, mà là diễn giải do chính tài
liệu này viết ra:

* mục 5.1 — *"Đề bài … yêu cầu thí sinh sử dụng giải pháp nằm ngoài lời giải
  chính để tạo thêm dữ liệu huấn luyện"*: không có dấu trích dẫn, là câu văn của
  tài liệu;
* mục 5.3 — *"không gọi OpenAI API … **trong quá trình tạo kết quả**"*: cụm giới
  hạn phạm vi này cũng do tài liệu tự thêm.

Nói cách khác, tôi lấy diễn giải của chính mình làm bằng chứng cho chính nó.

## 2. Rủi ro bất đối xứng

| | Nếu dùng OpenAI |
|---|---|
| Được | Nhãn chất lượng cao hơn Qwen self-host một mức **chưa đo được** |
| Mất | Mục 5.2 bắt buộc nộp "code tạo dữ liệu" và "toàn bộ dữ liệu"; BTC sẽ **thấy** dấu vết OpenAI khi dựng lại. Nếu họ đọc luật theo nghĩa rộng thì **bị loại**, sau khi đã đầu tư toàn bộ công sức. |

Đánh đổi này không đáng. Nhất là khi có phương án khác **không hề thua kém rõ
rệt** mà an toàn tuyệt đối.

## 3. Phương án thay thế: dùng chính Qwen đang chạy

Ta **không cần** OpenAI. Qwen teacher đã chạy self-host trên Kaggle cho corrector
và bộ loại span. Dùng chính nó để sinh nhãn thì hợp lệ dưới **mọi cách đọc** của
luật — không có API nào được gọi, không có weights nào ngoài hạn mức.

Và có bằng chứng nó đủ tốt cho việc này: đo trực tiếp từ hai lần nộp,

| tín hiệu phân biệt rác | tỉ lệ đúng |
|---|---:|
| điểm tin cậy GLiNER | 55–58% |
| **teacher Qwen3-4B** | **65–69%** |

Qwen đã phân biệt tốt hơn hẳn tín hiệu ta đang dùng. Nhãn của nó không hoàn hảo,
nhưng nó **tốt hơn cái đang có** — và đó là toàn bộ điều cần thiết để cải thiện.

## 4. Vì sao fine-tune mới là chỗ có 22 điểm

Ba lần nộp gần nhất đã đo được biên độ của việc vặn tham số:

| Thay đổi | Δ điểm |
|---|---:|
| bật bộ loại span | +0.38 |
| nâng sàn ngưỡng lên 0.30 | −0.63 |
| *trần lý thuyết của bộ loại* | *~+2* |

Trong khi khoảng cách tới nhóm đầu là **hơn 22 điểm**. Vặn tham số đã hết đường.

Nguyên nhân gốc đã rõ: **điểm tin cậy của GLiNER gần như không mang thông tin về
đúng/sai** — dải thấp nhất của nó chỉ 57% rác, gần bằng precision trung bình
50–64% của cả pipeline. Không ngưỡng nào cứu được một tín hiệu không phân biệt.

Fine-tune sửa đúng chỗ đó: dạy model cho điểm cao vào span thật và điểm thấp vào
span rác, trên đúng phân bố văn bản của cuộc thi.

## 5. Ngân sách tham số

| Vai trò | Model | Tham số |
|---|---|---:|
| sinh nhãn (offline) | `Qwen/Qwen3-8B` | 8.200B |
| NER sau fine-tune (nộp bài) | GLiNER `multi-v2.1` | 0.289B |
| **Tổng khai báo** | | **8.489B** < 9B |

Kê khai cả model sinh nhãn cho chắc, dù nó không chạy lúc inference. Vẫn dưới hạn
mức, nên không cần tranh cãi phạm vi nào cả.

Nếu muốn giữ thêm teacher cho corrector thì dùng **đúng một** Qwen cho cả hai vai
trò, đừng nạp hai model khác nhau.

## 6. Các bước

| Bước | Việc | Ghi chú |
|---|---|---|
| 1 | Qwen3-8B chấm từng span ứng viên trên 100 văn bản, ghi nhãn + độ tin cậy | chạy Kaggle, offline, một lần |
| 2 | Kiểm định bộ nhãn bằng `tools/calibrate_pseudo_gt.py` | ba phép: cấu trúc, tái lập điểm, **bảo toàn thứ tự** |
| 3 | Fine-tune GLiNER trên bộ nhãn đó | 0.289B, chạy được trên Kaggle |
| 4 | Nộp thử, đối chiếu dự đoán với thực tế | bước quyết định hướng này sống hay chết |

Bước 2 không được bỏ. Nhãn do model sinh chỉ có ích nếu nó **xếp hạng** các
phương án giống scorer thật; hỏng phép bảo toàn thứ tự thì tối ưu theo nó sẽ đẩy
pipeline đi sai hướng một cách êm ái.

## 7. Nếu vẫn muốn dùng OpenAI

Cách duy nhất đúng đắn là **hỏi thẳng Ban Tổ chức** trước khi làm:

> Cho phép dùng LLM API ngoài để gán nhãn dữ liệu huấn luyện offline không, với
> điều kiện pipeline nộp bài hoàn toàn self-host và không gọi API nào?

Được trả lời "có" thì nâng cấp bước 1 sang GPT sau, phần còn lại của kế hoạch
không đổi. Chưa được trả lời thì **đừng làm** — không nên đặt cược cả giải vào
một cách đọc luật.

## 8. Điều kiện dừng

* bộ nhãn hỏng phép bảo toàn thứ tự sau hai lần sửa prompt;
* GLiNER sau fine-tune không cải thiện trên chính bộ nhãn giữ lại;
* một lần nộp thật cho kết quả ngược dấu với dự đoán.
