"""Liệt kê các model mà khoá hiện tại truy cập được.

Tên model thay đổi nhanh hơn bất kỳ tài liệu nào, nên đừng đoán — hỏi thẳng API
rồi truyền tên chính xác cho `medical-coder label-gpt --model ...`.

    python tools/list_openai_models.py
    python tools/list_openai_models.py gpt      # lọc theo chuỗi con
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from medical_coder.gpt_labeler import load_dotenv  # noqa: E402


def main(needle: str | None) -> int:
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Thiếu OPENAI_API_KEY — xem .env.example")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Chưa cài SDK: python -m pip install openai") from error

    models = OpenAI().models.list()
    names = sorted(model.id for model in models)
    if needle:
        names = [name for name in names if needle.lower() in name.lower()]
    if not names:
        print("không có model nào khớp")
        return 1
    print(f"{len(names)} model truy cập được:\n")
    for name in names:
        print(" ", name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else None))
