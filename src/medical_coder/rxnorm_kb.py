"""Build an RxNorm terminology table from RxNorm Current Prescribable Content.

The archive (``RxNorm_full_prescribe_<date>.zip``, public, no UMLS licence) is not
redistributable with the repository, so this module builds the TSV from a local
copy — see KAGGLE.md for the download step.

Only names that a clinician would actually write are kept. The term types below
cover the ingredient / brand / clinical-drug surfaces that appear in the Vòng 1
example (``amlodipine 10 mg po daily`` -> 308135, a clinical drug), while
dropping dose-form and pack noise that never matches a mention.
"""
from __future__ import annotations

import csv
import zipfile
from pathlib import Path

# RXNCONSO.RRF column offsets (pipe-delimited, no header).
RXCUI, SAB, TTY, STR, SUPPRESS = 0, 11, 12, 14, 16

# Concept classes a mention may legitimately resolve to. Every gold RxCUI in the
# Vòng 1 example is an SCD (``amlodipine 10 mg po daily`` -> 308135, "amlodipine
# 10 MG Oral Tablet"), plus ingredients and brand names for bare drug words.
#
# SCDC (ingredient + strength, e.g. 329526 "amlodipine 10 MG") is deliberately
# excluded. A dosed mention cleans to exactly an SCDC surface, so indexing it
# turns "no answer" into a *confidently wrong* answer on precisely the mentions
# most likely to carry a gold code. SCDC maps one-to-many onto SCDs by dose form,
# which is unresolvable from the mention — so under a unique-match policy the
# right output is silence.
CONCEPT_TERM_TYPES = frozenset({"SCD", "SBD", "BN", "IN", "PIN", "MIN"})

# Name rows worth indexing once a concept qualifies above. SY/PSN/TMSY are
# alternate surfaces of the same RxCUI (243670 SY "ASA 81 MG Oral Tablet").
NAME_TERM_TYPES = CONCEPT_TERM_TYPES | {"SY", "PSN", "TMSY"}

# Compact surfaces preferred as the display label.
LABEL_TERM_TYPES = frozenset({"IN", "PIN", "BN"})


def _iter_rows(handle):
    for line in handle:
        fields = line.decode("utf-8", "replace").rstrip("\n").split("|")
        if len(fields) <= SUPPRESS:
            continue
        if fields[SAB] != "RXNORM" or fields[SUPPRESS] not in ("N", ""):
            continue
        name = fields[STR].strip()
        if name:
            yield fields[RXCUI].strip(), fields[TTY], name


def build_alias_table(archive: Path) -> dict[str, dict[str, object]]:
    """Index every name of each RxCUI that qualifies as a linkable concept.

    Qualification is decided per *concept*, not per row: an SCDC RxCUI also
    carries TMSY rows, so filtering by row term type alone would let it back in.
    """
    with zipfile.ZipFile(archive) as bundle:
        member = next(name for name in bundle.namelist() if name.endswith("RXNCONSO.RRF"))

        with bundle.open(member) as handle:
            qualified = {
                rxcui
                for rxcui, term_type, _ in _iter_rows(handle)
                if term_type in CONCEPT_TERM_TYPES
            }

        table: dict[str, dict[str, object]] = {}
        with bundle.open(member) as handle:
            for rxcui, term_type, name in _iter_rows(handle):
                if rxcui not in qualified or term_type not in NAME_TERM_TYPES:
                    continue
                entry = table.setdefault(rxcui, {"label": name, "aliases": []})
                aliases: list[str] = entry["aliases"]  # type: ignore[assignment]
                if term_type in LABEL_TERM_TYPES and len(name) < len(str(entry["label"])):
                    aliases.append(str(entry["label"]))
                    entry["label"] = name
                elif name != entry["label"] and name not in aliases:
                    aliases.append(name)
    return table


def write_terminology_tsv(table: dict[str, dict[str, object]], destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["code", "label", "aliases"])
        for code in sorted(table, key=lambda value: int(value) if value.isdigit() else 0):
            entry = table[code]
            writer.writerow([code, entry["label"], "|".join(entry["aliases"])])  # type: ignore[arg-type]
    return len(table)


def build(archive: Path, destination: Path) -> int:
    return write_terminology_tsv(build_alias_table(archive), destination)
