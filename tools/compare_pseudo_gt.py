"""So hai bộ nhãn giả với nhau và với output hiện tại của pipeline.

Nhãn do model sinh không phải chân lý. Ba model mạnh nhất chỉ đồng ý với nhau
62-72% trên mẫu thử, nên thứ đáng tin là **phần giao**, còn phần lệch cho biết
vùng bản thân các model cũng không chắc.

    python tools/compare_pseudo_gt.py data/pseudo_gt/gpt-5.6-terra \\
                                      data/pseudo_gt/gpt-5.6-sol \\
                                      submissions/04-kaggle-rejector/json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def load_spans(directory: Path, with_type: bool = True) -> set[tuple]:
    spans = set()
    for path in directory.glob("*.json"):
        if not path.stem.isdigit():
            continue
        for concept in json.loads(path.read_text(encoding="utf-8")):
            key = (path.stem, concept["position"][0], concept["position"][1])
            spans.add(key + ((concept["type"],) if with_type else ()))
    return spans


def load_concepts(directory: Path) -> list[dict]:
    return [
        concept
        for path in directory.glob("*.json")
        if path.stem.isdigit()
        for concept in json.loads(path.read_text(encoding="utf-8"))
    ]


def jaccard(left: set, right: set) -> float:
    return len(left & right) / len(left | right) if (left | right) else 1.0


def main(label_a: Path, label_b: Path, prediction: Path) -> int:
    a, b, p = (load_spans(d) for d in (label_a, label_b, prediction))
    core = a & b

    print(f"{'nguồn':38s} {'span':>7s}")
    for name, spans in ((label_a.name, a), (label_b.name, b), (prediction.name, p)):
        print(f"  {name:36s} {len(spans):7d}")
    print()

    print(f"Đồng thuận hai bộ nhãn: Jaccard {jaccard(a, b):.3f}")
    print(f"  cùng đánh dấu (LÕI)         {len(core):6d}")
    print(f"  chỉ {label_a.name:24s} {len(a - b):6d}")
    print(f"  chỉ {label_b.name:24s} {len(b - a):6d}")
    print()

    print("Pipeline so với LÕI — đây là thứ quyết định điểm:")
    hit = len(core & p)
    print(f"  bắt được lõi        {hit:6d}/{len(core)} = {hit / len(core):.1%}  (recall)")
    print(f"  emit ngoài lõi      {len(p - core):6d}/{len(p)} = {len(p - core) / len(p):.1%}")
    print(f"  bỏ sót lõi          {len(core - p):6d}")
    print()

    loose_a, loose_b, loose_p = (load_spans(d, False) for d in (label_a, label_b, prediction))
    loose_core = loose_a & loose_b
    loose_hit = len(loose_core & loose_p)
    print("Bỏ qua type, chỉ so ranh giới:")
    print(f"  bắt được lõi        {loose_hit:6d}/{len(loose_core)} = {loose_hit / len(loose_core):.1%}")
    gap = loose_hit - hit
    print(f"  chênh lệch          {gap:6d} span đúng ranh giới nhưng SAI TYPE")
    print()

    print("Phân bố type:")
    for name, directory in ((label_a.name, label_a), (label_b.name, label_b),
                            (prediction.name, prediction)):
        counts = Counter(c["type"] for c in load_concepts(directory))
        print(f"  {name:28s} " + "  ".join(f"{k[:9]}={v}" for k, v in counts.most_common()))
    print()

    print("Assertion — pipeline đang emit 0:")
    for name, directory in ((label_a.name, label_a), (label_b.name, label_b)):
        concepts = load_concepts(directory)
        flagged = sum(1 for c in concepts if c.get("assertions"))
        labels = sum(len(c.get("assertions") or []) for c in concepts)
        print(f"  {name:28s} {flagged:5d}/{len(concepts)} concept có nhãn "
              f"({flagged / len(concepts):.0%}), {labels} nhãn")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    raise SystemExit(main(*(Path(arg) for arg in sys.argv[1:])))
