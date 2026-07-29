"""Merge Vietnamese alias patches into downloaded ICD-10 / RxNorm TSV files.

Usage:
    python scripts/build_vn_terminology.py \
        --icd-input  /path/to/icd10_en.tsv \
        --rxnorm-input /path/to/rxnorm_en.tsv \
        --icd-patch  data/terminology/vn_icd10_aliases.tsv \
        --rxnorm-patch data/terminology/vn_rxnorm_aliases.tsv \
        --icd-output  data/terminology/icd10.tsv \
        --rxnorm-output data/terminology/rxnorm.tsv

The patch files contain two columns (no label): code  aliases (pipe-separated).
The script appends patch aliases to existing aliases for matching codes. Codes in
the patch that do not exist in the base file are skipped with a warning.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def load_patch(path: Path) -> dict[str, list[str]]:
    """Return {code: [alias, ...]} from a two-column TSV patch file."""
    patches: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            code = str(row.get("code") or "").strip()
            raw = str(row.get("aliases") or "").strip()
            if not code or not raw:
                continue
            aliases = [a.strip() for a in raw.split("|") if a.strip()]
            if aliases:
                patches.setdefault(code, []).extend(aliases)
    return patches


def merge(input_path: Path, patch: dict[str, list[str]], output_path: Path) -> None:
    rows: list[dict[str, str]] = []
    with input_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        fieldnames = list(reader.fieldnames or [])
        if "code" not in fieldnames or "label" not in fieldnames:
            sys.exit(f"ERROR: {input_path} must have 'code' and 'label' columns")
        if "aliases" not in fieldnames:
            fieldnames.append("aliases")
        for row in reader:
            rows.append(dict(row))

    found_codes = {row["code"] for row in rows}
    skipped = sorted(set(patch) - found_codes)
    if skipped:
        print(f"WARNING: {len(skipped)} patch codes not in base file: {skipped[:10]}...")

    patched = 0
    for row in rows:
        extra = patch.get(row["code"])
        if not extra:
            continue
        existing = [a.strip() for a in (row.get("aliases") or "").split("|") if a.strip()]
        merged = list(dict.fromkeys(existing + extra))  # dedup, preserve order
        row["aliases"] = "|".join(merged)
        patched += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  {input_path.name} → {output_path.name}: {len(rows)} rows, {patched} patched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--icd-input", required=True, type=Path)
    parser.add_argument("--rxnorm-input", required=True, type=Path)
    parser.add_argument("--icd-patch", default=Path("data/terminology/vn_icd10_aliases.tsv"), type=Path)
    parser.add_argument("--rxnorm-patch", default=Path("data/terminology/vn_rxnorm_aliases.tsv"), type=Path)
    parser.add_argument("--icd-output", default=Path("data/terminology/icd10.tsv"), type=Path)
    parser.add_argument("--rxnorm-output", default=Path("data/terminology/rxnorm.tsv"), type=Path)
    args = parser.parse_args()

    print("Loading patches...")
    icd_patch = load_patch(args.icd_patch)
    rxnorm_patch = load_patch(args.rxnorm_patch)
    print(f"  ICD patch: {len(icd_patch)} codes")
    print(f"  RxNorm patch: {len(rxnorm_patch)} codes")

    print("Merging ICD-10...")
    merge(args.icd_input, icd_patch, args.icd_output)

    print("Merging RxNorm...")
    merge(args.rxnorm_input, rxnorm_patch, args.rxnorm_output)

    print("Done.")


if __name__ == "__main__":
    main()
