"""Build a Vietnamese ICD-10 terminology table from the official MoH catalog.

Source: *Phụ lục Bảng danh mục mã ICD-10 tiếng Việt*, issued with Thông tư
06/2026/TT-BYT (TT06), published by Bộ Y tế. The catalog ships in
``data/kb/raw/`` and is the reason a Vietnamese note can be linked at all: the
CDC ICD-10-CM descriptions used by the first submission are English only, so a
mention like ``viêm túi mật`` could never match an alias.

Three Vietnamese surfaces are harvested per row:

* ``TÊN BỆNH`` -> ``MÃ BỆNH``                        (leaf, e.g. A00.0)
* ``TÊN NHÓM BỆNH 3 KÝ TỰ`` -> ``MÃ NHÓM BỆNH 3 KÝ TỰ`` (3-char category, A00)
* ``HƯỚNG DẪN MÃ HÓA BỔ SUNG CỦA WHO 2019`` -> ``MÃ BỆNH`` (WHO synonyms)

The third column is not used by the reference solution. It is a mixed field: some
rows are clean synonyms (``Bệnh tả cổ điển``), others are inclusion notes
(``Bao gồm: ...``) or dagger/asterisk cross-references. :func:`iter_who_synonyms`
keeps only the clean synonym surfaces, which adds ~2.9k aliases and lifts the
unique-exact-match rate on our diagnosis mentions from 9.0% to 10.9%.

Output is the ``code / label / aliases`` TSV that :mod:`medical_coder.terminology`
already reads, so the result is a drop-in for ``--icd-kb``.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path

CATEGORY_CODE_COLUMN = "MÃ NHÓM BỆNH 3 KÝ TỰ"
CATEGORY_NAME_COLUMN = "TÊN NHÓM BỆNH 3 KÝ TỰ"
LEAF_CODE_COLUMN = "MÃ BỆNH"
LEAF_NAME_COLUMN = "TÊN BỆNH"
WHO_SYNONYM_COLUMN = "HƯỚNG DẪN MÃ HÓA BỔ SUNG CỦA WHO 2019"

# COVID-19 codes from QĐ 98, absent from the TT06 annex.
COVID_CODES = (
    ("U07.1", "COVID-19, vi rút được xác định"),
    ("U07.2", "COVID-19, vi rút không được xác định"),
)

# Inclusion/exclusion prose in the WHO guidance column, never a usable synonym.
_GUIDANCE_PROSE = re.compile(
    r"(bao gồm|loại trừ|dùng thêm|sử dụng|xem |mã hóa)", re.IGNORECASE
)


def dotted_code(value: str) -> str:
    """Normalize an ICD-10 code to dotted form (``A001`` -> ``A00.1``)."""
    code = str(value).strip().upper().replace(".", "")
    return f"{code[:3]}.{code[3:]}" if len(code) > 3 else code


def clean_name(value: str) -> str:
    return " ".join(str(value).replace("・", " ").split()).strip(" ,;:")


def iter_who_synonyms(value: str):
    """Yield usable Vietnamese synonyms from the WHO guidance column.

    Rejects the whole cell when it carries cross-reference markup (``†``, ``*``,
    parenthesised codes) or reads as inclusion/exclusion prose, because those
    surfaces are not names a clinician would write in a note.
    """
    raw = str(value).strip()
    if not raw or raw.isdigit():
        return
    if "†" in raw or "*" in raw or "(" in raw or _GUIDANCE_PROSE.search(raw):
        return
    for part in re.split(r"[\n;]+", raw):
        candidate = part.strip().strip("+-–— ")
        if not candidate or candidate.endswith(":") or candidate.isdigit():
            return
        if 4 <= len(candidate) <= 90:
            yield candidate


def build_alias_table(xlsx_path: Path) -> dict[str, dict[str, object]]:
    """Return ``{code: {"label": str, "aliases": list[str]}}`` from the catalog."""
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "Building the Vietnamese ICD KB needs pandas and openpyxl: "
            "python -m pip install pandas openpyxl"
        ) from error

    sheet = pd.read_excel(xlsx_path, sheet_name=0, header=None, dtype=str).fillna("")

    header_row = None
    for index in range(min(15, len(sheet))):
        if any(str(cell).strip().upper() == LEAF_CODE_COLUMN for cell in sheet.iloc[index]):
            header_row = index
            break
    if header_row is None:
        raise ValueError(f"Could not find a '{LEAF_CODE_COLUMN}' header in {xlsx_path}")

    header = [str(cell).strip() for cell in sheet.iloc[header_row]]
    body = sheet.iloc[header_row + 1 :].reset_index(drop=True)
    body.columns = header

    def column(name: str) -> str | None:
        for candidate in header:
            if candidate.strip().upper() == name:
                return candidate
        return None

    leaf_code = column(LEAF_CODE_COLUMN)
    leaf_name = column(LEAF_NAME_COLUMN)
    if leaf_code is None or leaf_name is None:
        raise ValueError(f"Missing {LEAF_CODE_COLUMN}/{LEAF_NAME_COLUMN} in {xlsx_path}")
    category_code = column(CATEGORY_CODE_COLUMN)
    category_name = column(CATEGORY_NAME_COLUMN)
    synonym_column = column(WHO_SYNONYM_COLUMN)

    table: dict[str, dict[str, object]] = {}

    def add(raw_code: str, raw_name: str) -> None:
        code = dotted_code(raw_code)
        if len(code.replace(".", "")) < 3:
            return
        name = clean_name(raw_name)
        if not name or name.lower() == "nan":
            return
        entry = table.setdefault(code, {"label": name, "aliases": []})
        aliases: list[str] = entry["aliases"]  # type: ignore[assignment]
        if name != entry["label"] and name not in aliases:
            aliases.append(name)

    for _, row in body.iterrows():
        add(row[leaf_code], row[leaf_name])
        if category_code is not None and category_name is not None:
            add(row[category_code], row[category_name])
        if synonym_column is not None:
            for synonym in iter_who_synonyms(row[synonym_column]):
                add(row[leaf_code], synonym)

    for code, name in COVID_CODES:
        add(code, name)
    return table


def write_terminology_tsv(table: dict[str, dict[str, object]], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["code", "label", "aliases"])
        for code in sorted(table):
            entry = table[code]
            writer.writerow([code, entry["label"], "|".join(entry["aliases"])])  # type: ignore[arg-type]
    return len(table)


def build(xlsx_path: Path, destination: Path) -> int:
    return write_terminology_tsv(build_alias_table(xlsx_path), destination)
