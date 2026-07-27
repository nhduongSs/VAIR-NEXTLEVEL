# Định dạng knowledge base local

Pipeline chính thức không cho Qwen tự sinh mã từ trí nhớ. Cần chuẩn bị hai tệp:

```text
data/terminology/icd10.tsv
data/terminology/rxnorm.tsv
```

CSV hoặc TSV phải có header:

```tsv
code	label	aliases
K21.9	Gastro-esophageal reflux disease without esophagitis	trào ngược dạ dày thực quản|GERD
```

`aliases` là tùy chọn; nhiều alias được phân cách bằng `|`. JSONL cũng được hỗ
trợ, mỗi dòng có dạng:

```json
{"code":"K21.9","label":"Gastro-esophageal reflux disease without esophagitis","aliases":["trào ngược dạ dày thực quản","GERD"]}
```

Yêu cầu:

- ICD dùng đúng phiên bản mà Ban Tổ chức chấm;
- RxNorm dùng RxCUI dạng chuỗi;
- bổ sung alias tiếng Việt, tên thương mại, tên hoạt chất và viết tắt;
- giữ lại nguồn, phiên bản, license và script tạo tệp để bàn giao;
- không đưa một mã vào nhiều dòng; nếu có nhiều tên, gom vào `aliases`.

Repository không kèm dữ liệu ICD/RxNorm vì workspace hiện chưa có bản được Ban
Tổ chức xác nhận và một số nguồn có điều khoản phân phối riêng.
