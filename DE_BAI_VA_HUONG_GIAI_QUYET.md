# Hệ thống AI nhận diện, chuẩn hóa và suy luận khái niệm y khoa từ văn bản tự do

## 1. Mục đích tài liệu

Tài liệu này diễn giải lại đầy đủ bài toán xây dựng hệ thống AI xử lý văn bản y khoa tự do, đồng thời đề xuất một hướng giải quyết có thể triển khai trong khuôn khổ cuộc thi.

Hệ thống cần biến dữ liệu lâm sàng phi cấu trúc như ghi chú bác sĩ, giấy ra viện, kết quả xét nghiệm và hồ sơ bệnh án điện tử thành dữ liệu có cấu trúc. Kết quả phải:

1. phát hiện đúng các cụm từ mang ý nghĩa y khoa;
2. phân loại các cụm từ theo nhóm yêu cầu;
3. xác định ngữ cảnh của triệu chứng, chẩn đoán và thuốc;
4. ánh xạ chẩn đoán sang ICD-10 và thuốc sang RxNorm;
5. biểu diễn được các quan hệ cần thiết giữa các khái niệm;
6. bảo toàn chính xác vị trí ký tự của từng cụm từ trong văn bản gốc;
7. xuất kết quả theo đúng định dạng JSON quy định.

---

## 2. Phát biểu bài toán

### 2.1. Bài toán tổng quát

Cho một văn bản y khoa tiếng Việt ở dạng tự do, xây dựng hệ thống NLP/LLM có khả năng thực hiện đồng thời các nhiệm vụ:

- **Clinical Named Entity Recognition (Clinical NER):** nhận diện các thực thể y khoa và thông tin có ý nghĩa lâm sàng.
- **Entity Typing:** gán loại cho từng thực thể.
- **Assertion Detection:** xác định thực thể có bị phủ định, thuộc tiền sử hay liên quan đến người thân của bệnh nhân hay không.
- **Medical Concept Normalization/Entity Linking:** liên kết chẩn đoán với mã ICD-10 và thuốc với mã RxNorm.
- **Relation Extraction/Ontological Reasoning:** xác định quan hệ giữa các thực thể trong cùng ngữ cảnh, ví dụ quan hệ giữa tên xét nghiệm và kết quả xét nghiệm.

Đây là một bài toán đa nhiệm. Chất lượng cuối cùng không chỉ phụ thuộc vào khả năng “hiểu nội dung” mà còn phụ thuộc mạnh vào độ chính xác của span ký tự, schema đầu ra và khả năng truy hồi mã chuẩn.

### 2.2. Bối cảnh

Dữ liệu lâm sàng thực tế thường:

- không có cấu trúc thống nhất;
- chứa nhiều cách viết cho cùng một khái niệm;
- trộn tiếng Việt, tiếng Anh, tên thương mại, tên hoạt chất và từ viết tắt;
- chứa lỗi chính tả, lỗi gõ dấu và ký hiệu chuyên ngành;
- sử dụng dấu phẩy làm dấu thập phân;
- viết nhiều thông tin trong một câu dài;
- lược bỏ chủ ngữ hoặc quan hệ ngữ pháp;
- chứa thông tin phủ định, tiền sử và bệnh sử gia đình;
- chứa tên xét nghiệm, giá trị, đơn vị và khoảng tham chiếu sát nhau.

Vì vậy, giải pháp chỉ dựa vào từ điển sẽ có độ bao phủ thấp, còn giải pháp chỉ dựa vào LLM dễ sai vị trí ký tự, sinh mã không tồn tại hoặc không tuân thủ schema. Hướng phù hợp là một pipeline lai giữa mô hình học máy, truy hồi từ cơ sở tri thức và luật xác định.

---

## 3. Đặc tả đầu vào

### 3.1. Định dạng

Mỗi bản ghi là một tệp `.txt` chứa một văn bản y khoa tự do. Nguồn văn bản có thể là:

- ghi chú khám lâm sàng;
- ghi chú của bác sĩ;
- giấy xuất viện;
- kết quả xét nghiệm;
- kết quả chẩn đoán hình ảnh;
- hồ sơ sức khỏe điện tử (EHR);
- các ghi chú lâm sàng khác.

Mỗi văn bản có thể chứa nhiều hơn một khái niệm y khoa.

### 3.2. Ví dụ

```text
Bệnh nhân bị bệnh 1 tuần nay, ho đờm xanh, tức ngực, đau thượng vị, ợ hơi, được chẩn đoán mắc bệnh trào ngược dạ dày - thực quản.
```

Các khái niệm cần phát hiện trong ví dụ gồm:

- triệu chứng: `ho đờm xanh`, `tức ngực`, `đau thượng vị`, `ợ hơi`;
- chẩn đoán: `bệnh trào ngược dạ dày - thực quản`;
- candidate ICD-10 phù hợp cho chẩn đoán: ví dụ `K21.0`, `K21.9`.

### 3.3. Yêu cầu bảo toàn văn bản

Không được làm thay đổi văn bản trước khi tính vị trí thực thể. Các thao tác chuẩn hóa như:

- chuyển chữ hoa thành chữ thường;
- chuẩn hóa Unicode;
- thay nhiều khoảng trắng bằng một khoảng trắng;
- đổi dấu phẩy thập phân thành dấu chấm;
- sửa lỗi chính tả;

chỉ được thực hiện trên một bản sao phục vụ suy luận. Kết quả cuối cùng phải được chiếu ngược về chỉ số ký tự của văn bản gốc.

---

## 4. Đặc tả đầu ra

### 4.1. Định dạng tổng thể

Với mỗi tệp đầu vào `<id>.txt`, hệ thống tạo một tệp `<id>.json`. Nội dung tệp JSON là một mảng các object, mỗi object biểu diễn một khái niệm y khoa:

```json
[
  {
    "text": "cụm từ xuất hiện nguyên văn",
    "type": "TRIỆU_CHỨNG",
    "assertions": [],
    "position": [0, 10]
  }
]
```

### 4.2. Các trường dữ liệu

| Trường | Kiểu dữ liệu | Bắt buộc | Ý nghĩa |
|---|---|---:|---|
| `text` | string | Có | Cụm từ được trích nguyên văn từ input |
| `position` | array gồm 2 số nguyên | Có | Vị trí bắt đầu và kết thúc của thực thể trong input |
| `type` | string | Có | Loại khái niệm y khoa |
| `assertions` | array string | Có | Các thuộc tính ngữ cảnh của thực thể |
| `candidates` | array string | Có điều kiện | Danh sách mã ICD-10 hoặc RxNorm dự đoán; xuất cho `CHẨN_ĐOÁN` và `THUỐC` |

Ví dụ chính thức của Vòng 1 bỏ trường `candidates` đối với `TRIỆU_CHỨNG`. Vì vậy, serializer nên bám đúng format này: mọi object có `text`, `type`, `assertions`, `position`; chỉ `CHẨN_ĐOÁN` và `THUỐC` có thêm `candidates`. Không trả về `null`.

### 4.3. Nhãn thực thể

Hệ thống chỉ xuất một trong năm nhãn sau:

| Nhãn | Nội dung |
|---|---|
| `TRIỆU_CHỨNG` | Dấu hiệu hoặc triệu chứng bệnh nhân gặp phải |
| `TÊN_XÉT_NGHIỆM` | Tên xét nghiệm hoặc chỉ số xét nghiệm |
| `KẾT_QUẢ_XÉT_NGHIỆM` | Giá trị và đơn vị của một xét nghiệm |
| `CHẨN_ĐOÁN` | Chẩn đoán/bệnh do bác sĩ xác định |
| `THUỐC` | Thuốc, hoạt chất hoặc chế phẩm điều trị |

Một số phần mô tả tổng quan có đề cập “thông tin bệnh nhân”, nhưng danh sách nhãn đầu ra chính thức không có nhãn tương ứng. Vì vậy, hệ thống không nên tự thêm nhãn như `TUỔI`, `GIỚI_TÍNH` hoặc `THÔNG_TIN_BỆNH_NHÂN` nếu ban tổ chức chưa xác nhận.

### 4.4. Assertion

Trường `assertions` chỉ áp dụng cho:

- `TRIỆU_CHỨNG`;
- `CHẨN_ĐOÁN`;
- `THUỐC`.

Mỗi thực thể có thể có từ 0 đến 3 assertion:

| Assertion | Ý nghĩa | Ví dụ |
|---|---|---|
| `isNegated` | Khái niệm bị phủ định | `không ho`, `chưa ghi nhận sốt` |
| `isFamily` | Khái niệm thuộc người thân/gia đình | `bố bệnh nhân bị tăng huyết áp` |
| `isHistorical` | Khái niệm thuộc tiền sử | `tiền sử hen`, `đã từng dùng aspirin` |

Các assertion không loại trừ lẫn nhau. Ví dụ, một bệnh trong câu “mẹ bệnh nhân không có tiền sử đái tháo đường” có thể đồng thời là `isFamily`, `isHistorical` và `isNegated`.

`TÊN_XÉT_NGHIỆM` và `KẾT_QUẢ_XÉT_NGHIỆM` nên có `assertions: []`.

### 4.5. Candidate mapping

Trường `candidates` chỉ có nội dung đối với:

- `CHẨN_ĐOÁN`: danh sách mã ICD-10;
- `THUỐC`: danh sách mã RxNorm/RxCUI.

Theo ví dụ chính thức của Vòng 1, các loại còn lại **không có trường `candidates`**. Đây là format nên dùng khi nộp bài, trừ khi JSON Schema hoặc code evaluator do Ban Tổ chức cung cấp có quy định khác.

Danh sách candidate nên:

1. được sắp xếp theo độ phù hợp giảm dần;
2. không chứa mã trùng lặp;
3. chỉ chứa mã tồn tại trong cơ sở tri thức được cung cấp;
4. không trộn mã ICD-10 và RxNorm;
5. ưu tiên mã cụ thể khi văn bản có đủ bằng chứng, nhưng giữ mã tổng quát hơn ở các vị trí sau nếu còn mơ hồ.

### 4.6. Quy ước vị trí ký tự

Ví dụ chính thức của Vòng 1 xác nhận `position` dùng chỉ số bắt đầu từ 0 và mốc kết thúc loại trừ, tức `[start, end)`. Với tất cả thực thể trong ví dụ, `end - start` đúng bằng độ dài chuỗi `text`. Điều này cụ thể hơn câu mô tả ban đầu “vị trí tính từ 0 đến n - 1”, vốn có thể khiến người đọc hiểu nhầm mốc cuối là inclusive.

Có thể kiểm tra trực tiếp:

```python
text[start:end] == entity_text
```

Với ví dụ ngắn ở trên:

| Thực thể | Position |
|---|---:|
| `ho đờm xanh` | `[30, 41]` |
| `tức ngực` | `[43, 51]` |
| `đau thượng vị` | `[53, 66]` |
| `ợ hơi` | `[68, 73]` |
| `bệnh trào ngược dạ dày - thực quản` | `[94, 128]` |

### 4.7. Ví dụ JSON hoàn chỉnh

Ví dụ dưới đây sử dụng đúng quy ước end-exclusive và cách bỏ trường `candidates` khi không áp dụng:

```json
[
  {
    "text": "ho đờm xanh",
    "position": [30, 41],
    "type": "TRIỆU_CHỨNG",
    "assertions": []
  },
  {
    "text": "tức ngực",
    "position": [43, 51],
    "type": "TRIỆU_CHỨNG",
    "assertions": []
  },
  {
    "text": "đau thượng vị",
    "position": [53, 66],
    "type": "TRIỆU_CHỨNG",
    "assertions": []
  },
  {
    "text": "ợ hơi",
    "position": [68, 73],
    "type": "TRIỆU_CHỨNG",
    "assertions": []
  },
  {
    "text": "bệnh trào ngược dạ dày - thực quản",
    "position": [94, 128],
    "type": "CHẨN_ĐOÁN",
    "assertions": [],
    "candidates": ["K21.0", "K21.9"]
  }
]
```

---

## 5. Dữ liệu và thể thức Vòng 1

Tập test gồm 100 bản ghi:

```text
test/
└── input/
    ├── 1.txt
    ├── 2.txt
    ├── ...
    └── 100.txt
```

### 5.1. Gói kết quả dự đoán

Thí sinh nộp duy nhất tệp `output.zip`. Sau khi giải nén, gói kết quả phải có đúng cấu trúc:

```text
output.zip
└── output/
    ├── 1.json
    ├── 2.json
    ├── ...
    └── 100.json
```

Mỗi tệp `output/<id>.json` là kết quả của đúng tệp `test/input/<id>.txt`. Tệp JSON chứa một list các object khái niệm y khoa theo schema tại Mục 4.

Checklist trước khi nén:

- thư mục gốc bên trong ZIP có tên chính xác là `output`;
- có đủ `1.json` đến `100.json`, không thừa hoặc thiếu;
- không đặt thêm một tầng thư mục như `output/output/`;
- mỗi tệp parse được bằng JSON parser;
- dùng UTF-8;
- mọi position dùng end-exclusive;
- mọi span được đối chiếu trên chính nội dung tệp `.txt` gốc;
- không đưa log, trace, confidence hoặc trường nội bộ vào submission.

Đề bài không cung cấp tập train và yêu cầu thí sinh sử dụng giải pháp nằm ngoài lời giải chính để tạo thêm dữ liệu huấn luyện. Điều này làm cho chiến lược tạo dữ liệu, weak supervision và khai thác cơ sở tri thức trở thành một phần quan trọng của lời giải.

### 5.2. Yêu cầu bàn giao source code đối với nhóm đứng đầu

Trước khi Vòng 1 kết thúc, Ban Tổ chức yêu cầu khoảng 15 đội đứng đầu gửi riêng source code để dựng lại lời giải và đánh giá trên private test. Mục tiêu là kiểm tra khả năng tái lập và ngăn việc hard-code output theo public test.

Gói bàn giao phải gồm:

- toàn bộ code xử lý dữ liệu;
- code tạo dữ liệu hoặc chuẩn bị dữ liệu;
- code huấn luyện;
- code inference và sinh `output.zip`;
- toàn bộ dữ liệu nhóm sử dụng;
- model weights;
- file README hướng dẫn cài đặt và chạy.

Nếu Ban Tổ chức không thể cài đặt, đội thi có thể được liên hệ để hỗ trợ trong một khoảng thời gian giới hạn. Không hỗ trợ kịp thời có thể dẫn đến bị loại.

Vì vậy, ngay từ đầu lời giải phải có khả năng tái lập:

- cố định seed và ghi rõ phiên bản thư viện;
- không phụ thuộc đường dẫn tuyệt đối trên máy phát triển;
- khai báo rõ phần cứng, RAM/VRAM và thời gian chạy;
- có lệnh duy nhất hoặc script rõ ràng để chạy inference;
- kiểm tra checksum/version của model và knowledge base;
- không gọi dịch vụ bên ngoài bí mật hoặc API không được mô tả;
- ghi rõ license và nguồn của dữ liệu bổ sung.

### 5.3. Giới hạn tài nguyên và API

Thông tin bổ sung từ Ban Tổ chức:

> Thí sinh tự chuẩn bị tài nguyên tính toán. Tuy nhiên, với những giải pháp
> LLM/agent chỉ cho phép thí sinh self-host model mà không được sử dụng API
> ngoài, model self-host có độ lớn tối đa là 9B params.

Để thận trọng, lời giải này áp dụng giả thuyết chặt hơn: **tổng tham số của mọi
bộ weights khác nhau được đóng gói không vượt 9B**, không chỉ từng model riêng
lẻ. Quantization không làm thay đổi số tham số được kê khai.

Cấu hình phù hợp Kaggle Free GPU:

| Thành phần | Vai trò | Tham số khai báo |
|---|---|---:|
| Qwen3-8B | NER, assertion, rerank candidate | 8,200M |
| multilingual-e5-small | retrieval đa ngôn ngữ | 118M |
| Luật, lexical index, validator | offset, alias, schema, ZIP | 0 |
| **Tổng** |  | **8,318M** |

Qwen3-8B được tái sử dụng cho nhiều stage nhưng chỉ đóng gói một bộ weights.
Model được nạp 4-bit để vừa VRAM; E5 chạy CPU hoặc được nạp tuần tự. Mọi
candidate phải đến từ knowledge base ICD/RxNorm local. Tuyệt đối không gọi
OpenAI API, Hugging Face Inference API hoặc dịch vụ inference ngoài trong quá
trình tạo kết quả.

Sau khi có dữ liệu gán nhãn và benchmark cục bộ, có thể cân nhắc thêm
XLM-R-base khoảng 278M cho NER/assertion đa nhiệm. Tổng khi đó xấp xỉ 8,596B,
vẫn dưới giả thuyết 9B. Không nên đóng gói checkpoint này nếu chưa fine-tune
hoặc chưa chứng minh được mức tăng điểm.

### 5.4. Ví dụ chính thức của Vòng 1

Ví dụ là một danh sách thuốc trước nhập viện. Toàn bộ 11 thuốc được gán `isHistorical`; các triệu chứng/chỉ định đi kèm không có assertion:

| STT | Text | Type | Candidates | Assertions | Position |
|---:|---|---|---|---|---:|
| 1 | `amlodipine 10 mg po daily` | `THUỐC` | `["308135"]` | `["isHistorical"]` | `[58, 83]` |
| 2 | `aspirin 81 mg po daily` | `THUỐC` | `["243670"]` | `["isHistorical"]` | `[89, 111]` |
| 3 | `metoprolol succinate xl 50 mg po daily` | `THUỐC` | `["866436"]` | `["isHistorical"]` | `[117, 155]` |
| 4 | `guaifenesin ml po q6h:prn` | `THUỐC` | `["392085"]` | `["isHistorical"]` | `[161, 186]` |
| 5 | `ho` | `TRIỆU_CHỨNG` | — | `[]` | `[196, 198]` |
| 6 | `nystatin oral suspension 5 ml po qid:prn` | `THUỐC` | `["7597"]` | `["isHistorical"]` | `[204, 244]` |
| 7 | `đau nhức` | `TRIỆU_CHỨNG` | — | `[]` | `[254, 262]` |
| 8 | `acetaminophen 325-650 mg po q6h:prn` | `THUỐC` | `["313782"]` | `["isHistorical"]` | `[268, 303]` |
| 9 | `sốt đau` | `TRIỆU_CHỨNG` | — | `[]` | `[313, 320]` |
| 10 | `pravastatin 40 mg po daily` | `THUỐC` | `["904475"]` | `["isHistorical"]` | `[326, 352]` |
| 11 | `docusate sodium 100 mg po bid` | `THUỐC` | `["1099279"]` | `["isHistorical"]` | `[358, 387]` |
| 12 | `táo bón` | `TRIỆU_CHỨNG` | — | `[]` | `[397, 404]` |
| 13 | `senna 8.6 mg po bid:prn` | `THUỐC` | `["312935"]` | `["isHistorical"]` | `[410, 433]` |
| 14 | `táo bón` | `TRIỆU_CHỨNG` | — | `[]` | `[443, 450]` |
| 15 | `clonazepam 0.5 mg po qam:prn` | `THUỐC` | `["197527"]` | `["isHistorical"]` | `[457, 485]` |
| 16 | `lo âu` | `TRIỆU_CHỨNG` | — | `[]` | `[495, 500]` |
| 17 | `clonazepam 1.5 mg po qhs` | `THUỐC` | `["197528"]` | `["isHistorical"]` | `[507, 531]` |
| 18 | `lo âu` | `TRIỆU_CHỨNG` | — | `[]` | `[541, 546]` |
| 19 | `mất ngủ` | `TRIỆU_CHỨNG` | — | `[]` | `[547, 554]` |

Các kết luận trực tiếp từ ví dụ:

- span thuốc bao gồm tên, hàm lượng, đường dùng và tần suất khi các thành phần nằm liền nhau;
- thuốc “trước nhập viện” được đánh dấu `isHistorical`;
- cùng một text như `táo bón` hoặc `lo âu` xuất hiện nhiều lần phải tạo nhiều entity theo từng position;
- `candidates` của thuốc là chuỗi RxCUI, không phải số JSON;
- các object triệu chứng không chứa trường `candidates`;
- position là end-exclusive vì hiệu `end - start` bằng độ dài `text`;
- cần đọc offset trên tệp đầu vào gốc; bản input được hiển thị trên trang thể lệ có thể đã làm thay đổi xuống dòng hoặc khoảng trắng.

---

## 6. Các điểm cần xác nhận với ban tổ chức

Trước khi chốt code nộp bài, cần hỏi hoặc kiểm tra tài liệu bổ sung cho các vấn đề sau:

1. Vị trí được tính theo Unicode code point, UTF-16 code unit, byte UTF-8 hay cách đếm ký tự của Python?
2. Có giới hạn số candidate cho mỗi thực thể không, ví dụ top-1, top-3 hay top-5?
3. Candidate có cần giữ thứ tự xếp hạng không? Công thức Jaccard cho thấy thứ tự có thể không ảnh hưởng, nhưng điều này chưa được nói trực tiếp.
4. Phiên bản ICD-10 và RxNorm nào được sử dụng?
5. Chỉ được trả mã có trong gói tri thức cung cấp hay có thể dùng mã ngoài?
6. “Quan hệ giữa các khái niệm” được biểu diễn bằng trường nào? Schema Vòng 1 chỉ có `assertions`, chưa có trường `relations`.
7. Tên xét nghiệm và kết quả xét nghiệm có cần liên kết với nhau không? Nếu có, quan hệ được ghi ở mức entity ID hay position?
8. Có cho phép hai span chồng lấn hoặc lồng nhau không?
9. “Chẩn đoán phân biệt”, “nghi ngờ”, “theo dõi” được coi là `CHẨN_ĐOÁN` hay cần assertion riêng?
10. Thuốc đang dùng, thuốc đã ngừng và dị ứng thuốc có cùng được gán `THUỐC` không?
11. Với kết quả xét nghiệm, span có bao gồm đơn vị, dấu bất thường (`H`, `L`, `+`, `-`) và khoảng tham chiếu không?
12. WER được tính trên chuỗi text đã ghép theo thứ tự nào, và evaluator ghép entity dự đoán với ground truth ra sao?
13. `J_assertions(i)` và `J_candidates(i)` được gom ở cấp bản ghi hay là trung bình trên các entity đã ghép?
14. Có bắt buộc bỏ trường `candidates` ở loại không áp dụng, hay `candidates: []` cũng được chấp nhận?

Nếu không nhận được câu trả lời, giải pháp nên cô lập các lựa chọn trên trong file cấu hình để có thể đổi quy ước mà không sửa pipeline lõi.

---

## 7. Đề xuất kiến trúc giải pháp

### 7.1. Kiến trúc tổng thể

```text
Văn bản gốc
    │
    ├── Tiền xử lý có bảo toàn offset
    │
    ├── Nhận diện và phân loại span
    │       ├── Mô hình Clinical NER
    │       ├── Bộ luật xét nghiệm/thuốc
    │       └── Từ điển y khoa
    │
    ├── Hợp nhất và giải quyết span xung đột
    │
    ├── Phân loại assertion
    │
    ├── Chuẩn hóa khái niệm
    │       ├── CHẨN_ĐOÁN → ICD-10
    │       └── THUỐC → RxNorm
    │
    ├── Trích xuất quan hệ
    │
    ├── Kiểm tra ontology và tính nhất quán
    │
    └── Kiểm tra schema → JSON
```

Khuyến nghị xây dựng pipeline theo module thay vì một prompt LLM duy nhất. Cách này giúp đo riêng từng lỗi, thay thế từng mô hình và bảo đảm đầu ra xác định.

### 7.2. Bước 1 — Tiền xử lý bảo toàn offset

Tạo hai phiên bản:

- `raw_text`: bất biến, dùng để lấy `text` và `position`;
- `normalized_text`: dùng cho tìm kiếm, NER và chuẩn hóa.

Mỗi ký tự/token trong bản chuẩn hóa cần có ánh xạ về vị trí trong `raw_text`. Có thể dùng một mảng:

```text
normalized_index -> raw_index
```

Các xử lý nên hỗ trợ:

- chuẩn hóa Unicode NFC/NFKC có kiểm soát;
- chuẩn hóa khoảng trắng;
- tách câu dựa trên dấu câu và các pattern y khoa;
- nhận diện viết tắt;
- tạo phiên bản không dấu để truy hồi;
- chuẩn hóa số thập phân;
- không thay đổi bản gốc.

### 7.3. Bước 2 — Nhận diện và phân loại thực thể

Nên kết hợp ba nguồn dự đoán.

#### A. Mô hình span-based NER

Fine-tune một encoder hỗ trợ tiếng Việt hoặc đa ngôn ngữ để dự đoán span và nhãn. Hai lựa chọn:

- token classification với BIO/BILOU;
- span classification, chấm điểm trực tiếp mọi cặp start-end hợp lệ.

Span classification phù hợp hơn nếu dữ liệu có thực thể dài và dấu câu phức tạp. Token classification đơn giản hơn và có nhiều thư viện sẵn.

#### B. Luật có độ chính xác cao

Luật đặc biệt hữu ích cho:

- tên xét nghiệm viết hoa hoặc viết tắt: `WBC`, `NEUT%`, `HbA1c`;
- kết quả dạng số, đơn vị và cờ bất thường;
- hàm lượng thuốc: `0.4 MG/ML`, `500 mg`;
- dạng bào chế và đường dùng;
- các cấu trúc như `X: value unit`;
- cụm từ báo hiệu chẩn đoán: `chẩn đoán`, `kết luận`, `mắc`;
- cụm từ báo hiệu tiền sử hoặc phủ định.

Luật không nên tự quyết toàn bộ nhãn mà nên tạo candidate span hoặc feature cho mô hình hợp nhất.

#### C. Dictionary matching

Xây dựng alias dictionary từ ICD-10, RxNorm và dữ liệu bổ sung:

- tên chuẩn;
- tên đồng nghĩa;
- tên không dấu;
- từ viết tắt;
- biến thể dấu gạch nối/khoảng trắng;
- tên hoạt chất, tên thương mại và dạng bào chế;
- lỗi gõ phổ biến có kiểm soát.

Dùng exact/fuzzy matching để tăng recall, sau đó lọc theo ngữ cảnh.

#### Hợp nhất span

Mỗi span candidate có:

```text
(start, end, type, source, confidence)
```

Quy tắc gợi ý:

1. loại span không thỏa `raw_text[start:end] == text`;
2. hợp nhất các dự đoán trùng span và cùng nhãn;
3. ưu tiên span đầy đủ hơn nếu hai span cùng loại và một span bao chứa span kia;
4. ưu tiên rule độ chính xác cao cho kết quả xét nghiệm;
5. sử dụng điểm calibration riêng cho mỗi nguồn;
6. xử lý span chồng lấn theo quy định chính thức của cuộc thi.

### 7.4. Bước 3 — Phân loại assertion

Sử dụng mô hình phân loại đa nhãn với đầu vào gồm:

- câu chứa thực thể;
- một câu trước và một câu sau nếu cần;
- marker đánh dấu span, ví dụ `[ENTITY] ... [/ENTITY]`;
- loại thực thể.

Đầu ra gồm ba xác suất độc lập:

```text
P(isNegated), P(isFamily), P(isHistorical)
```

Kết hợp mô hình với scope rules:

- phủ định: `không`, `chưa`, `không ghi nhận`, `loại trừ`, `âm tính với`;
- gia đình: `bố`, `mẹ`, `anh/chị/em`, `con`, `gia đình`, `họ hàng`;
- tiền sử: `tiền sử`, `trước đây`, `đã từng`, `cách đây`, `mạn tính`.

Quan trọng nhất là xác định **phạm vi tác động**. Trong câu “không sốt, ho nhẹ”, từ “không” chỉ phủ định `sốt`, không phủ định `ho nhẹ`.

Không nên đánh đồng:

- `đã dùng thuốc` với phủ định;
- `loại trừ bệnh X` với chẩn đoán xác định;
- `chưa dùng thuốc X` với tiền sử dùng thuốc;
- `tiền sử gia đình` với tiền sử của chính bệnh nhân.

### 7.5. Bước 4 — Chuẩn hóa chẩn đoán sang ICD-10

Áp dụng pipeline retrieval–reranking:

#### Tạo chỉ mục ICD-10

Mỗi mã gồm:

- code;
- tên chuẩn;
- mô tả;
- synonym/alias;
- mã cha và mã con;
- chương/nhóm bệnh;
- embedding ngữ nghĩa.

#### Truy hồi candidate

Kết hợp:

- BM25 trên tên và synonym;
- fuzzy lexical matching;
- dense retrieval bằng sentence embedding;
- tra cứu exact alias;
- ưu tiên candidate phù hợp với từ khóa giải phẫu, mức độ và biến chứng.

Hợp nhất top candidate bằng Reciprocal Rank Fusion hoặc tổng trọng số đã hiệu chỉnh.

#### Reranking

Cross-encoder hoặc LLM reranker nhận:

- mention;
- câu/ngữ cảnh xung quanh;
- tên và mô tả mã ICD;
- bằng chứng về vị trí giải phẫu, cấp/mạn, có/không biến chứng.

Reranker chỉ được chọn trong tập mã đã truy hồi, không được tự sinh mã mới.

#### Kiểm tra phân cấp

- nếu ngữ cảnh không đủ cụ thể, không ép chọn mã con quá chi tiết;
- nếu candidate cha và con cùng xuất hiện, xếp hạng dựa trên bằng chứng;
- loại mã không phù hợp với giới tính, độ tuổi hoặc giải phẫu nếu có luật chắc chắn;
- giữ top-k theo yêu cầu submission.

### 7.6. Bước 5 — Chuẩn hóa thuốc sang RxNorm

Phân tích mention thuốc thành các thành phần:

```text
ingredient + strength + dose form + route + brand
```

Ví dụ:

```text
Chlorpheniramine 0.4 MG/ML
```

được tách thành:

- hoạt chất: `chlorpheniramine`;
- nồng độ: `0.4 mg/mL`;
- dạng bào chế/đường dùng: lấy từ ngữ cảnh nếu có.

Candidate generation:

1. exact match theo tên chuẩn hoặc normalized string;
2. match hoạt chất;
3. match hoạt chất + hàm lượng;
4. match brand/generic;
5. dense retrieval cho biến thể tiếng Việt;
6. rerank dựa trên độ khớp thành phần.

Thứ tự ưu tiên khi có đủ dữ kiện:

```text
ingredient + exact strength + exact dose form
> ingredient + exact strength
> ingredient only
```

Phải xác thực RxCUI trong cơ sở dữ liệu được cung cấp để ngăn hallucination.

### 7.7. Bước 6 — Trích xuất quan hệ

Do schema quan hệ chưa được mô tả, module quan hệ nên tạo biểu diễn nội bộ độc lập:

```json
{
  "source_entity_id": "e1",
  "target_entity_id": "e2",
  "relation": "HAS_RESULT",
  "confidence": 0.98
}
```

Các quan hệ hữu ích:

- `HAS_RESULT`: tên xét nghiệm → kết quả xét nghiệm;
- `TREATED_BY`: chẩn đoán/triệu chứng → thuốc;
- `INDICATES`: kết quả xét nghiệm → chẩn đoán;
- `ASSOCIATED_WITH`: quan hệ lâm sàng tổng quát.

Trong phạm vi đề hiện tại, nên ưu tiên `HAS_RESULT`, vì có thể xác định tương đối chắc chắn từ cấu trúc `tên: giá trị đơn vị`. Chỉ đưa quan hệ vào submission khi ban tổ chức cung cấp schema chính thức; nếu không, module này vẫn hữu ích để kiểm tra và ghép đúng tên xét nghiệm với kết quả.

### 7.8. Bước 7 — Ontological reasoning và kiểm tra nhất quán

Lớp kiểm tra sau cùng áp dụng các ràng buộc:

- chỉ `CHẨN_ĐOÁN` có ICD-10;
- chỉ `THUỐC` có RxNorm;
- `candidates` không xuất hiện ở ba loại còn lại theo ví dụ Vòng 1;
- chỉ ba loại được phép có assertion;
- mỗi code phải tồn tại trong knowledge base;
- span phải nằm trong giới hạn văn bản;
- nội dung span phải khớp tuyệt đối với `text`;
- không có object trùng hoàn toàn;
- mã cha/con không được xếp hạng vô lý khi ngữ cảnh đủ cụ thể;
- kết quả xét nghiệm nên có tên xét nghiệm liên quan gần đó;
- vị trí entity được sắp tăng dần để output ổn định.

### 7.9. Bước 8 — Validation và xuất JSON

Validator phải chạy trước khi đóng gói:

```python
assert 0 <= start < end <= len(raw_text)
assert raw_text[start:end] == entity["text"]
assert entity["type"] in ALLOWED_TYPES
assert set(entity["assertions"]) <= ALLOWED_ASSERTIONS

if entity["type"] in {"CHẨN_ĐOÁN", "THUỐC"}:
    assert "candidates" in entity
    assert candidates_are_valid(entity)
else:
    assert "candidates" not in entity
```

Nên định nghĩa JSON Schema và unit test cho các trường hợp:

- văn bản Unicode có dấu;
- entity ở đầu/cuối tệp;
- nhiều entity giống hệt nhau ở các vị trí khác nhau;
- số thập phân dùng dấu phẩy;
- xuống dòng Windows/Unix;
- span chứa dấu ngoặc, `%`, `/`, `-`;
- file không có entity;
- assertion kết hợp.

---

## 8. Chiến lược dữ liệu huấn luyện

### 8.1. Dữ liệu weak supervision

Tạo nhãn bạc bằng cách:

1. dò tên/alias ICD-10 và RxNorm trong tập văn bản y khoa;
2. dùng luật nhận diện xét nghiệm và kết quả;
3. dùng từ khóa xác định assertion;
4. cho một mô hình mạnh hoặc LLM gán nhãn;
5. chỉ giữ mẫu khi nhiều nguồn đồng thuận;
6. kiểm tra thủ công một tập nhỏ để đo precision.

### 8.2. Sinh dữ liệu tổng hợp

Tạo template theo phong cách ghi chép lâm sàng:

```text
Bệnh nhân [tuổi/giới], [triệu chứng]. Chẩn đoán [bệnh].
Tiền sử [bệnh/thuốc].
[xét nghiệm]: [giá trị] [đơn vị].
```

Sau đó biến đổi có kiểm soát:

- thêm/bớt dấu tiếng Việt;
- thay đồng nghĩa;
- chuyển tên đầy đủ thành viết tắt;
- thay dấu `:` bằng khoảng trắng hoặc dấu `=`;
- hoán đổi thứ tự thông tin;
- chèn lỗi chính tả nhẹ;
- thay dấu thập phân;
- đổi chữ hoa/chữ thường;
- tạo phủ định, tiền sử và ngữ cảnh gia đình;
- tạo hard negative chứa thuật ngữ nhưng không phải thực thể mục tiêu.

Mọi dữ liệu tổng hợp phải giữ được nhãn span chính xác sau biến đổi.

### 8.3. Active learning

Nếu có nguồn lực gán nhãn thủ công:

1. huấn luyện baseline;
2. chạy trên dữ liệu chưa nhãn;
3. chọn mẫu có entropy cao, mô hình bất đồng hoặc candidate normalization gần điểm nhau;
4. chuyên gia sửa nhãn;
5. bổ sung vào tập train và lặp lại.

Active learning giúp tập trung công sức vào viết tắt, span ranh giới khó và mã dễ nhầm.

### 8.4. Chia tập dữ liệu

Nên chia theo **bản ghi/bệnh nhân**, không chia theo câu, để tránh rò rỉ các đoạn gần giống nhau:

- train: 70–80%;
- validation: 10–15%;
- local test: 10–15%.

Tập validation cần có đủ:

- năm loại entity;
- ba assertion;
- ICD phổ biến và hiếm;
- thuốc generic/brand/hàm lượng;
- lỗi chính tả và viết tắt;
- xét nghiệm có đơn vị và không đơn vị.

---

## 9. Ba mức giải pháp

### 9.1. Baseline nhanh

- dictionary + regex để phát hiện entity;
- cue words cho assertion;
- BM25/fuzzy matching cho ICD-10 và RxNorm;
- rule ghép xét nghiệm với kết quả;
- validator output.

Ưu điểm: nhanh, dễ debug, không cần nhiều dữ liệu.  
Nhược điểm: recall thấp, khó xử lý diễn đạt mới và mơ hồ ngữ cảnh.

### 9.2. Giải pháp khuyến nghị cho cuộc thi

- NER encoder fine-tune + dictionary + rule;
- assertion classifier đa nhãn + scope rules;
- hybrid retrieval cho ICD/RxNorm;
- cross-encoder hoặc LLM reranking;
- ontology constraints;
- confidence calibration;
- validator chặt chẽ;
- ensemble 2–3 mô hình NER nếu ngân sách tính toán cho phép.

Đây là lựa chọn cân bằng tốt giữa độ chính xác, khả năng kiểm soát và thời gian triển khai.

### 9.3. Giải pháp LLM/agent

LLM có thể hỗ trợ:

- tạo dữ liệu tổng hợp;
- mở rộng synonym;
- phân tích các ca khó;
- rerank candidate;
- giải thích quan hệ;
- kiểm tra chéo output.

Không nên để LLM tự do sinh JSON và mã y tế ở bước cuối. Nên giới hạn LLM bằng:

- danh sách candidate đã truy hồi;
- constrained decoding/JSON Schema;
- tool tra cứu knowledge base;
- kiểm tra span và code bằng chương trình;
- fallback về mô hình/rule khi output không hợp lệ.

Agent orchestration chỉ thực sự cần khi mỗi agent có công cụ riêng, ví dụ:

1. agent phát hiện entity;
2. agent tra ICD;
3. agent tra RxNorm;
4. agent kiểm tra assertion;
5. agent validator.

Với 100 bản ghi, pipeline module hóa thường đơn giản, nhanh và ổn định hơn một hệ multi-agent hoàn toàn tự trị.

---

## 10. Huấn luyện đa nhiệm và hàm mất mát

Nếu huấn luyện chung một encoder, có thể tối ưu:

```text
L_total =
    λ1 * L_NER
  + λ2 * L_assertion
  + λ3 * L_relation
  + λ4 * L_linking
```

Trong giai đoạn đầu, nên huấn luyện từng module riêng để dễ tìm lỗi. Chỉ dùng multi-task learning sau khi đã có baseline và đủ dữ liệu.

Đối với lớp hiếm:

- class weights hoặc focal loss;
- oversampling mẫu `isFamily`, `isNegated`;
- hard-negative mining;
- threshold riêng cho từng assertion;
- calibration trên validation.

---

## 11. Metric đánh giá Vòng 1

### 11.1. Công thức điểm cuối

Điểm Vòng 1 là tổng có trọng số của ba thành phần:

```text
final_score =
    0.3 × text_score
  + 0.3 × assertions_score
  + 0.4 × candidates_score
```

Candidate mapping có trọng số lớn nhất, chiếm 40% tổng điểm. Tuy vậy, candidate chỉ được chấm có ý nghĩa khi hệ thống đã phát hiện và phân loại được khái niệm tương ứng.

### 11.2. Điểm nội dung khái niệm

Trường `text` được đánh giá bằng Word Error Rate (WER):

```text
WER = (S + D + I) / N
```

Trong đó:

- `S`: số phép thay thế từ;
- `D`: số phép xóa từ;
- `I`: số phép chèn từ;
- `N`: số từ trong ground truth.

Theo ký hiệu của đề bài, với `i` là một sample:

```text
text_score = (1 / |test|) × Σ(i ∈ test) [1 - WER(i)]
```

Tài liệu chưa mô tả cách ghép và sắp thứ tự các entity trước khi tính WER. Do đó, evaluator local cần được điều chỉnh ngay khi Ban Tổ chức cung cấp code chấm hoặc giải thích chính thức.

### 11.3. Jaccard similarity

Assertion và candidate được so sánh theo tập hợp. Với trường `X`:

```text
J_X(i) = 1
    nếu ground_truth_X(i) và prediction_X(i) đều rỗng

J_X(i) = 0
    nếu ground_truth_X(i) rỗng nhưng prediction_X(i) không rỗng

J_X(i) = |ground_truth_X(i) ∩ prediction_X(i)|
         ---------------------------------------
         |ground_truth_X(i) ∪ prediction_X(i)|
    trong các trường hợp còn lại
```

Nếu ground truth không rỗng nhưng dự đoán rỗng, công thức giao/hợp ở trường hợp cuối cho kết quả 0.

### 11.4. Điểm assertion

Assertion được xét trên các khái niệm thuộc `CHẨN_ĐOÁN`, `THUỐC` và `TRIỆU_CHỨNG`. Độ tương đồng Jaccard của các assertion tương ứng được tổng hợp thành `J_assertions(i)`, sau đó lấy trung bình trên toàn bộ test:

```text
assertions_score =
    (1 / |test|) × Σ(i ∈ test) J_assertions(i)
```

### 11.5. Điểm candidate

Candidate cũng được chấm bằng Jaccard. Điểm candidate của mỗi sample được gán trọng số theo số mã ground truth của các khái niệm trong sample.

Đặt:

```text
w_i = Σ(k ∈ i) [|ground_truth(k)| + 1]
```

Khi đó:

```text
candidates_score =
    Σ(i ∈ test) [J_candidates(i) × w_i]
    ------------------------------------
             Σ(i ∈ test) w_i
```

Số hạng `+1` bảo đảm mỗi khái niệm vẫn đóng góp trọng số kể cả khi tập candidate ground truth rỗng.

### 11.6. Quy tắc phạt sai loại

Nếu dự đoán đúng `text` nhưng sai `type`, ví dụ dự đoán `CHẨN_ĐOÁN` trong khi ground truth là `TRIỆU_CHỨNG`, khái niệm bị tính hai lần:

1. một lần bỏ sót khái niệm đúng;
2. một lần tạo thêm khái niệm sai.

Cả hai lần đều nhận 0 ở cả ba nhóm metric. Vì vậy, type classification là “cổng” ảnh hưởng đồng thời đến text, assertion và candidate.

### 11.7. Hệ quả đối với chiến lược tối ưu

- Không nên trả toàn bộ top-k code không qua ngưỡng: Jaccard giảm khi thêm candidate sai.
- Nên hiệu chỉnh ngưỡng riêng cho ICD-10 và RxNorm trên validation.
- Candidate chiếm 40%, nhưng không được đánh đổi bằng việc tạo quá nhiều entity giả.
- Phải tối ưu cả ranh giới span và cách giữ nguyên cụm từ vì `text` được chấm bằng WER.
- Không nên gán assertion chỉ vì có một cue word ở xa; một assertion thừa làm giảm Jaccard.
- Với list ground truth rỗng, dự đoán rỗng đạt Jaccard bằng 1; dự đoán thừa đạt 0.
- Vì sai type bị phạt kép, các ca type không chắc chắn nên được calibration thay vì luôn chọn nhãn có logit cao nhất.
- Thứ tự candidate về lý thuyết không ảnh hưởng Jaccard vì metric thao tác trên tập hợp, nhưng vẫn nên xuất thứ tự điểm giảm dần để ổn định và dễ debug.

### 11.8. Metric chẩn đoán nên theo dõi thêm

Ngoài metric chính thức, quá trình phát triển nên theo dõi:

- strict span + type Precision/Recall/F1;
- F1 riêng cho từng loại entity;
- assertion micro/macro F1;
- Accuracy@1, Recall@k và MRR cho normalization;
- tỷ lệ code không hợp lệ, mục tiêu bằng 0;
- tỷ lệ file JSON hợp lệ;
- tỷ lệ `raw_text[start:end] == entity.text`;
- thời gian xử lý mỗi bản ghi.

Các metric phụ giúp xác định module gây lỗi; điểm chính thức dùng để lựa chọn checkpoint và threshold cuối cùng.

---

## 12. Phân tích lỗi

Mỗi lần đánh giá nên phân lỗi thành:

- bỏ sót entity;
- phát hiện thừa;
- sai ranh giới span;
- sai loại;
- sai scope phủ định;
- nhầm tiền sử với hiện tại;
- nhầm bệnh nhân với người nhà;
- đúng entity nhưng không truy hồi được code;
- có đúng code trong top-k nhưng rerank sai;
- chọn mã quá tổng quát/quá cụ thể;
- lỗi format/offset/schema.

Nên lưu trace cho từng entity:

```json
{
  "span_sources": ["ner", "dictionary"],
  "ner_score": 0.93,
  "assertion_scores": {
    "isNegated": 0.02,
    "isFamily": 0.01,
    "isHistorical": 0.87
  },
  "retrieved_candidates": [],
  "final_candidates": []
}
```

Trace chỉ dùng khi phát triển, không đưa vào file submission.

---

## 13. Cấu trúc mã nguồn đề xuất

```text
project/
├── configs/
│   ├── pipeline.yaml
│   └── labels.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   ├── synthetic/
│   └── knowledge_base/
├── src/
│   ├── preprocessing/
│   │   ├── normalize.py
│   │   └── offsets.py
│   ├── ner/
│   ├── assertions/
│   ├── normalization/
│   │   ├── icd_retriever.py
│   │   ├── rxnorm_retriever.py
│   │   └── reranker.py
│   ├── relations/
│   ├── reasoning/
│   ├── validation/
│   │   ├── schema.py
│   │   └── validate_output.py
│   └── pipeline.py
├── scripts/
│   ├── build_kb.py
│   ├── generate_synthetic.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
├── tests/
├── output/
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 14. Luồng suy luận tham khảo

```python
def process_record(raw_text):
    normalized_text, offset_map = normalize_with_offsets(raw_text)

    spans = []
    spans += ner_model.predict(normalized_text)
    spans += lab_rules.predict(normalized_text)
    spans += medication_rules.predict(normalized_text)
    spans += dictionary_matcher.predict(normalized_text)

    entities = merge_and_project_to_raw(spans, offset_map, raw_text)

    for entity in entities:
        if entity.type in {"TRIỆU_CHỨNG", "CHẨN_ĐOÁN", "THUỐC"}:
            entity.assertions = assertion_model.predict(raw_text, entity)
        else:
            entity.assertions = []

        if entity.type == "CHẨN_ĐOÁN":
            retrieved = icd_retriever.search(entity.text, raw_text, top_n=30)
            entity.candidates = icd_reranker.rank(retrieved)[:TOP_K]
        elif entity.type == "THUỐC":
            retrieved = rxnorm_retriever.search(entity.text, raw_text, top_n=30)
            entity.candidates = rxnorm_reranker.rank(retrieved)[:TOP_K]
        else:
            entity.candidates = None  # serializer sẽ bỏ trường này

    relations = relation_extractor.predict(raw_text, entities)
    entities = ontology_checker.apply(entities, relations)
    output = serializer.to_submission_schema(entities)
    validate(raw_text, output)
    return output
```

---

## 15. Lộ trình triển khai

### Giai đoạn 1 — Chốt đặc tả và baseline

- cố định schema Vòng 1, quy ước position end-exclusive và cấu trúc `output.zip`;
- xác nhận top-k, phiên bản knowledge base và các chi tiết evaluator còn thiếu;
- xây validator;
- xây exact dictionary matching, regex xét nghiệm và assertion cues;
- tạo pipeline chạy đủ 100 file;
- có submission hợp lệ đầu tiên.

### Giai đoạn 2 — Dữ liệu và NER

- xây bộ annotation guideline;
- tạo dữ liệu weak/synthetic;
- gán nhãn thủ công một tập validation nhỏ nhưng chất lượng cao;
- fine-tune NER;
- phân tích lỗi span theo từng loại.

### Giai đoạn 3 — Chuẩn hóa khái niệm

- xây chỉ mục ICD-10/RxNorm;
- triển khai lexical + dense retrieval;
- tạo tập positive/negative pair;
- huấn luyện hoặc cấu hình reranker;
- đánh giá Recall@k trước khi tối ưu top-1.

### Giai đoạn 4 — Assertion và quan hệ

- xây scope rules;
- huấn luyện assertion classifier;
- ghép tên xét nghiệm–kết quả;
- bổ sung ontology constraints.

### Giai đoạn 5 — Tối ưu end-to-end

- calibration threshold theo công thức điểm chính thức;
- tối ưu tập candidate theo Jaccard thay vì mặc định luôn trả top-k;
- hard-negative mining;
- ensemble nếu có lợi;
- test hồi quy offset/schema;
- benchmark tốc độ và bộ nhớ;
- đóng gói `output.zip` và source/model/data có thể tái lập trên private test.

---

## 16. Rủi ro và biện pháp giảm thiểu

| Rủi ro | Tác động | Biện pháp |
|---|---|---|
| Đếm offset khác runtime của BTC | Sai toàn bộ span dù nhận diện đúng | Kiểm tra trên raw file; test Unicode; cô lập serializer |
| Không có train set | NER/assertion yếu | Weak supervision, synthetic data, active learning |
| LLM sinh mã không tồn tại | Sai normalization | Chỉ chọn từ candidate trong KB |
| Dữ liệu y khoa nhiều viết tắt | Bỏ sót/nhầm nghĩa | Từ điển viết tắt + ngữ cảnh + hard negatives |
| Sai scope phủ định | Gán sai assertion | Rule theo phạm vi + classifier |
| ICD quá chi tiết | Chọn mã con sai | Rerank theo bằng chứng và hierarchy |
| RxNorm nhầm hàm lượng/dạng bào chế | Sai RxCUI | Parse cấu phần thuốc trước khi match |
| Dùng normalization làm lệch vị trí | JSON sai | Offset map và invariant test |
| Rule và model xung đột | Output không ổn định | Confidence calibration + precedence rõ ràng |
| Schema quan hệ chưa rõ | Không thể xuất đúng | Dùng relation nội bộ, chờ schema chính thức |
| Trả quá nhiều candidate | Jaccard giảm vì tăng phần hợp sai | Chọn tập theo threshold được tune trên validation |
| Sai type | Bị tính hai lần và nhận 0 ở cả ba metric | Calibration, confusion analysis, hard negatives |
| Không tái lập được private test | Có nguy cơ bị loại dù public score cao | Đóng gói code/data/weights/README và test môi trường sạch |
| Rò rỉ dữ liệu qua sinh mẫu | Điểm local ảo | Deduplicate và chia theo bản ghi |
| Hạn chế giấy phép dữ liệu chuẩn | Không thể phát hành/triển khai | Kiểm tra license trước khi phân phối KB |

---

## 17. Kết luận và khuyến nghị

Giải pháp nên được triển khai theo kiến trúc lai:

1. mô hình NER để hiểu ngữ cảnh và tăng độ bao phủ;
2. luật và từ điển để bắt các cấu trúc y khoa có tính quy ước;
3. retrieval–reranking để ánh xạ ICD-10/RxNorm;
4. classifier đa nhãn kết hợp scope rules cho assertion;
5. ontology constraints và validator để loại lỗi logic, mã và schema;
6. LLM chủ yếu dùng cho tạo dữ liệu, reranking hoặc xử lý ca khó, không dùng làm nguồn sinh mã tự do.

Ưu tiên triển khai theo thứ tự:

```text
schema/offset đúng
→ baseline end-to-end
→ tăng recall của NER
→ tăng Recall@k của retrieval
→ tối ưu reranking và ngưỡng candidate theo Jaccard
→ tối ưu assertion/relations
→ ensemble, calibration và đóng gói tái lập
```

Một hệ thống có mô hình mạnh nhưng sai offset, sai type hoặc sinh mã ngoài knowledge base vẫn có thể nhận điểm rất thấp. Candidate chiếm 40% điểm, nhưng sai type bị phạt kép ở cả ba metric; vì vậy cần tối ưu end-to-end bằng đúng scorer của Vòng 1 thay vì tối ưu từng module tách rời. Tính đúng đắn của pipeline, validator và khả năng tái lập trên private test phải được xem là thành phần cốt lõi.

---

## 18. Cập nhật triển khai và nộp bài lần đầu

Ngày 27/07/2026, pipeline self-host Qwen3-8B + multilingual-e5-small đã chạy đủ
100 input trên Kaggle P100, tạo một ZIP đúng schema và được BTC chấm thành công.
Điểm baseline nhận được là **14.4255** với `WER = 83.5952`,
`J_assertion = 20.1874` và `J_candidates = 8.6197`.

Lần chạy xác nhận khả năng tái lập end-to-end: source nhúng, PyTorch CUDA 12.6
cho `sm_60`, inference local, căn chỉnh offset, terminology local và validator
đều hoạt động. Mức `J_candidates` thấp cho thấy ưu tiên kỹ thuật tiếp theo là
tăng coverage/chất lượng alias ICD-10 và RxNorm, sau đó tune tập candidate theo
Jaccard. Báo cáo artifact, checksum và thống kê đầy đủ được lưu tại
[KET_QUA_NOP_LAN_01.md](KET_QUA_NOP_LAN_01.md).
