"""Local reading of the Vòng 1 host metric.

    final_score = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score

This is *our reading* of the published formula, not the official grader. It is
reconciled against the only two public data points we have:

* our submission 01 (14.4255 from WER 83.5952 / J_assert 20.1874 / J_cand 8.6197)
* the reference solution's 27.8786 (32.1820 / 35.2687 / 19.1084)

Both reproduce exactly under `0.3*text + 0.3*assert + 0.4*cand`, which confirms
the published `WER` field is an *error rate* and `text_score = 1 - WER`.

Choices the organisers left unspecified are all localised here so a single edit
switches convention when the official scorer is released:

* A prediction matches a ground-truth concept iff **same type** and **overlapping
  character span**, matched greedily by largest overlap. A right-text/wrong-type
  prediction therefore cannot match its twin: it is a brand-new concept scoring 0
  everywhere, which is exactly the double penalty described in the rules.
* An unmatched ("spurious") prediction is counted **twice** in every applicable
  denominator. That is the mechanism that makes over-emission so expensive, and
  it is the single most important property to optimise against.
* Aggregation is global over concepts, not a mean of per-record means.

Scoring the ground truth against itself returns 1.0.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from .models import EntityType

WEIGHTS = {"text": 0.3, "assertions": 0.3, "candidates": 0.4}

ASSERTABLE_TYPES = {
    EntityType.SYMPTOM.value,
    EntityType.DIAGNOSIS.value,
    EntityType.MEDICATION.value,
}
CANDIDATE_TYPES = {
    EntityType.DIAGNOSIS.value,
    EntityType.MEDICATION.value,
}


def jaccard(prediction: set[str], truth: set[str]) -> float:
    """Jaccard with the host edge cases: both empty -> 1, one empty -> 0."""
    if not prediction and not truth:
        return 1.0
    if not prediction or not truth:
        return 0.0
    return len(prediction & truth) / len(prediction | truth)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein error rate, (S + D + I) / N."""
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
            )
        previous = current
    return previous[len(hyp)] / len(ref)


def _span(concept: Mapping) -> tuple[int, int]:
    start, end = concept["position"]
    return int(start), int(end)


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def match_concepts(
    predictions: Sequence[Mapping],
    truth: Sequence[Mapping],
) -> tuple[dict[int, int], list[int], list[int]]:
    """Greedily pair prediction -> truth on (same type, largest overlap)."""
    pairs = []
    for pi, prediction in enumerate(predictions):
        prediction_span = _span(prediction)
        for ti, gold in enumerate(truth):
            if gold.get("type") != prediction.get("type"):
                continue
            overlap = _overlap(prediction_span, _span(gold))
            if overlap > 0:
                pairs.append((overlap, pi, ti))
    pairs.sort(key=lambda item: (-item[0], item[1], item[2]))

    matched: dict[int, int] = {}
    used_predictions: set[int] = set()
    used_truth: set[int] = set()
    for _, pi, ti in pairs:
        if pi in used_predictions or ti in used_truth:
            continue
        matched[pi] = ti
        used_predictions.add(pi)
        used_truth.add(ti)
    unmatched_predictions = [i for i in range(len(predictions)) if i not in used_predictions]
    unmatched_truth = [i for i in range(len(truth)) if i not in used_truth]
    return matched, unmatched_predictions, unmatched_truth


@dataclass
class RecordScore:
    record_id: str
    text: float
    assertions: float
    candidates: float
    predicted: int
    expected: int


@dataclass
class Score:
    text_score: float
    assertions_score: float
    candidates_score: float
    final_score: float
    per_record: list[RecordScore] = field(default_factory=list)

    def format_report(self) -> str:
        return (
            f"text       {self.text_score * 100:8.4f}\n"
            f"assertions {self.assertions_score * 100:8.4f}\n"
            f"candidates {self.candidates_score * 100:8.4f}\n"
            f"final      {self.final_score * 100:8.4f}"
        )


def score_corpus(
    predictions: Mapping[str, list],
    truth: Mapping[str, list],
    record_ids: Sequence[str] | None = None,
) -> Score:
    """Score predictions against ground truth using the host formula.

    text        = sum_matched (1 - WER) / (n_truth + 2 * n_spurious)
    assertions  = sum_matched J_assert / (n_truth_assertable + 2 * n_spurious_assertable)
    candidates  = sum_matched J_cand * w / (sum_truth w + 2 * sum_spurious w),
                  where w = len(truth candidates) + 1
    """
    if record_ids is None:
        record_ids = sorted(
            truth,
            key=lambda value: (0, int(value)) if value.isdigit() else (1, value),
        )

    text_numerator = text_denominator = 0.0
    assertion_numerator = assertion_denominator = 0.0
    candidate_numerator = candidate_denominator = 0.0
    per_record: list[RecordScore] = []

    for record_id in record_ids:
        gold = truth[record_id]
        predicted = predictions.get(record_id, [])
        matched, spurious, missed = match_concepts(predicted, gold)

        record_text = sum(
            max(0.0, 1.0 - word_error_rate(gold[ti]["text"], predicted[pi]["text"]))
            for pi, ti in matched.items()
        )
        record_text_denominator = len(gold) + 2 * len(spurious)
        text_numerator += record_text
        text_denominator += record_text_denominator

        assertable_gold = [
            i for i in range(len(gold)) if gold[i].get("type") in ASSERTABLE_TYPES
        ]
        assertable_spurious = [
            i for i in spurious if predicted[i].get("type") in ASSERTABLE_TYPES
        ]
        record_assertions = sum(
            jaccard(
                set(predicted[pi].get("assertions") or []),
                set(gold[ti].get("assertions") or []),
            )
            for pi, ti in matched.items()
            if gold[ti].get("type") in ASSERTABLE_TYPES
        )
        record_assertion_denominator = len(assertable_gold) + 2 * len(assertable_spurious)
        assertion_numerator += record_assertions
        assertion_denominator += record_assertion_denominator

        record_candidates = 0.0
        record_candidate_denominator = 0.0
        for pi, ti in matched.items():
            if gold[ti].get("type") not in CANDIDATE_TYPES:
                continue
            weight = len(gold[ti].get("candidates") or []) + 1
            record_candidates += weight * jaccard(
                set(predicted[pi].get("candidates") or []),
                set(gold[ti].get("candidates") or []),
            )
            record_candidate_denominator += weight
        for ti in missed:
            if gold[ti].get("type") in CANDIDATE_TYPES:
                record_candidate_denominator += len(gold[ti].get("candidates") or []) + 1
        for pi in spurious:
            if predicted[pi].get("type") in CANDIDATE_TYPES:
                record_candidate_denominator += 2 * (
                    len(predicted[pi].get("candidates") or []) + 1
                )
        candidate_numerator += record_candidates
        candidate_denominator += record_candidate_denominator

        per_record.append(
            RecordScore(
                record_id=record_id,
                text=record_text / record_text_denominator if record_text_denominator else 1.0,
                assertions=(
                    record_assertions / record_assertion_denominator
                    if record_assertion_denominator
                    else 1.0
                ),
                candidates=(
                    record_candidates / record_candidate_denominator
                    if record_candidate_denominator
                    else 1.0
                ),
                predicted=len(predicted),
                expected=len(gold),
            )
        )

    text = text_numerator / text_denominator if text_denominator else 1.0
    assertions = (
        assertion_numerator / assertion_denominator if assertion_denominator else 1.0
    )
    candidates = (
        candidate_numerator / candidate_denominator if candidate_denominator else 1.0
    )
    final = (
        WEIGHTS["text"] * text
        + WEIGHTS["assertions"] * assertions
        + WEIGHTS["candidates"] * candidates
    )
    return Score(text, assertions, candidates, final, per_record)


def load_records(directory: Path) -> dict[str, list]:
    return {
        path.stem: json.loads(path.read_text(encoding="utf-8"))
        for path in directory.glob("*.json")
        if path.stem.isdigit()
    }
