from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .models import AlignedEntity, AssertionType, EntityType, ExtractedMention


ASSERTION_ORDER = {
    AssertionType.NEGATED: 0,
    AssertionType.FAMILY: 1,
    AssertionType.HISTORICAL: 2,
}


@dataclass(frozen=True)
class AlignmentIssue:
    text: str
    type: str
    start_hint: int
    reason: str


def _all_exact_occurrences(raw_text: str, mention: str) -> list[tuple[int, int]]:
    return [(match.start(), match.end()) for match in re.finditer(re.escape(mention), raw_text)]


def _all_casefold_occurrences(raw_text: str, mention: str) -> list[tuple[int, int]]:
    return [
        (match.start(), match.end())
        for match in re.finditer(re.escape(mention), raw_text, flags=re.IGNORECASE)
    ]


def _all_flexible_whitespace_occurrences(
    raw_text: str,
    mention: str,
) -> list[tuple[int, int]]:
    parts = re.split(r"\s+", mention.strip())
    if not parts:
        return []
    pattern = r"\s+".join(re.escape(part) for part in parts)
    return [
        (match.start(), match.end())
        for match in re.finditer(pattern, raw_text, flags=re.IGNORECASE)
    ]


def _choose_occurrence(
    occurrences: Iterable[tuple[int, int]],
    start_hint: int,
    used_occurrences: set[tuple[int, int, EntityType]],
    entity_type: EntityType,
) -> tuple[int, int] | None:
    available = [
        item
        for item in occurrences
        if (item[0], item[1], entity_type) not in used_occurrences
    ]
    if not available:
        return None
    return min(available, key=lambda item: (abs(item[0] - start_hint), item[0]))


def _sanitize_assertions(
    entity_type: EntityType,
    assertions: Iterable[AssertionType],
) -> list[AssertionType]:
    if entity_type not in {
        EntityType.SYMPTOM,
        EntityType.DIAGNOSIS,
        EntityType.MEDICATION,
    }:
        return []
    return sorted(set(assertions), key=ASSERTION_ORDER.__getitem__)


def align_mentions(
    raw_text: str,
    mentions: Iterable[ExtractedMention],
) -> tuple[list[AlignedEntity], list[AlignmentIssue]]:
    """Align exact LLM quotes to raw text and return end-exclusive positions."""

    aligned: list[AlignedEntity] = []
    issues: list[AlignmentIssue] = []
    used_occurrences: set[tuple[int, int, EntityType]] = set()

    for mention in mentions:
        predicted_text = mention.text.strip()
        if not predicted_text:
            issues.append(
                AlignmentIssue(
                    text=mention.text,
                    type=mention.type.value,
                    start_hint=mention.start_hint,
                    reason="empty mention after trimming",
                )
            )
            continue

        occurrence = _choose_occurrence(
            _all_exact_occurrences(raw_text, predicted_text),
            mention.start_hint,
            used_occurrences,
            mention.type,
        )
        if occurrence is None:
            occurrence = _choose_occurrence(
                _all_casefold_occurrences(raw_text, predicted_text),
                mention.start_hint,
                used_occurrences,
                mention.type,
            )
        if occurrence is None:
            occurrence = _choose_occurrence(
                _all_flexible_whitespace_occurrences(raw_text, predicted_text),
                mention.start_hint,
                used_occurrences,
                mention.type,
            )

        if occurrence is None:
            issues.append(
                AlignmentIssue(
                    text=predicted_text,
                    type=mention.type.value,
                    start_hint=mention.start_hint,
                    reason="quote not found in raw text",
                )
            )
            continue

        start, end = occurrence
        used_occurrences.add((start, end, mention.type))
        aligned.append(
            AlignedEntity(
                text=raw_text[start:end],
                type=mention.type,
                assertions=_sanitize_assertions(mention.type, mention.assertions),
                position=(start, end),
            )
        )

    # Deduplicate identical objects, but preserve the same text at distinct offsets.
    unique: dict[tuple[int, int, EntityType], AlignedEntity] = {}
    for entity in aligned:
        key = (entity.position[0], entity.position[1], entity.type)
        unique.setdefault(key, entity)

    return (
        sorted(
            unique.values(),
            key=lambda entity: (
                entity.position[0],
                entity.position[1],
                entity.type.value,
            ),
        ),
        issues,
    )

