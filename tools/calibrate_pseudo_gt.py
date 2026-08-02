"""Kiểm định một bộ ground truth giả trước khi dùng nó để tối ưu.

Bộ nhãn do LLM sinh ra chỉ có ích nếu nó **xếp hạng** các thay đổi giống scorer
thật. Ta không kiểm chứng được điều đó bằng chính nó, nhưng có bốn điểm chấm
thật từ BTC — đủ để bác bỏ một bộ nhãn tồi.

Ba phép kiểm, từ rẻ tới đắt:

1. **Cấu trúc** — số concept phải nằm trong vùng đã giải ngược từ điểm thật.
   Bộ nhãn 3.000 concept là sai ngay, không cần chấm gì thêm.
2. **Tái lập điểm** — chấm các lần nộp đã biết điểm; sai số tuyệt đối cho biết
   bộ nhãn lệch bao nhiêu.
3. **Bảo toàn thứ tự** — quan trọng nhất. Ta dùng bộ nhãn để CHỌN giữa các
   phương án, nên nó chỉ cần xếp đúng thứ tự, không cần đúng giá trị.

Một bộ nhãn hỏng phép 3 thì vô dụng dù phép 2 có đẹp đến đâu.

    python tools/calibrate_pseudo_gt.py data/pseudo_gt
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medical_coder.scoring import load_records, score_corpus  # noqa: E402

# Điểm chính thức của BTC, kèm thư mục output đã tạo ra chúng.
KNOWN = [
    ("01 Qwen3-8B sinh mention", "submissions/01-20260727-qwen8b/json", 14.4255),
    ("04 thêm bộ loại span", "submissions/04-kaggle-rejector/json", 27.5217),
]

# Vùng khả dĩ giải ngược từ ba chỉ số công bố (xem PHAN_TICH mục phân tích ngược).
G_RANGE = (1150, 1600)


def main(pseudo_dir: Path) -> int:
    truth = load_records(pseudo_dir)
    if not truth:
        raise SystemExit(f"Không đọc được nhãn nào trong {pseudo_dir}")

    total = sum(len(v) for v in truth.values())
    print(f"Bộ nhãn: {len(truth)} bản ghi, {total} concept "
          f"({total / len(truth):.1f}/bản ghi)\n")

    print("1. CẤU TRÚC")
    low, high = G_RANGE
    verdict = "nằm trong vùng" if low <= total <= high else "NGOÀI vùng"
    print(f"   G suy ngược từ điểm thật: {low}-{high} concept")
    print(f"   bộ nhãn này            : {total} concept -> {verdict}")
    if not low <= total <= high:
        print("   => nhãn lệch xa; sửa prompt trước khi dùng để tối ưu.")
    print()

    print("2. TÁI LẬP ĐIỂM")
    rows = []
    for name, directory, official in KNOWN:
        path = Path(directory)
        if not path.is_dir():
            print(f"   {name:26s} THIẾU output ({directory})")
            continue
        score = score_corpus(load_records(path), truth)
        local = score.final_score * 100
        rows.append((name, local, official))
        print(f"   {name:26s} nhãn giả {local:7.3f} | BTC {official:7.4f} "
              f"| lệch {local - official:+7.3f}")
    print()

    print("3. BẢO TOÀN THỨ TỰ  (phép quan trọng nhất)")
    if len(rows) < 2:
        print("   cần ít nhất hai lần nộp có điểm; hãy lưu output.zip của mỗi lần.")
        return 1
    ok = True
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            name_i, local_i, off_i = rows[i]
            name_j, local_j, off_j = rows[j]
            same = (local_i < local_j) == (off_i < off_j)
            ok &= same
            mark = "đúng" if same else "SAI"
            print(f"   {name_i[:2]} vs {name_j[:2]}: "
                  f"nhãn giả {local_j - local_i:+7.3f} | "
                  f"BTC {off_j - off_i:+7.4f} -> {mark}")
    print()
    if ok:
        print("KẾT LUẬN: bộ nhãn xếp đúng thứ tự mọi cặp -> dùng được để CHỌN")
        print("phương án. Vẫn chỉ tin các chênh lệch lớn, và mọi kết luận phải")
        print("được xác nhận lại bằng một lần nộp thật.")
        return 0
    print("KẾT LUẬN: bộ nhãn xếp SAI thứ tự -> chưa dùng để tối ưu được.")
    print("Tối ưu theo nó sẽ đẩy pipeline đi sai hướng một cách êm ái.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    raise SystemExit(main(Path(sys.argv[1])))
