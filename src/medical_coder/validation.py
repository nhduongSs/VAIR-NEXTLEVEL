from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from .models import AssertionType, EntityType


ICD10_RE = re.compile(r"^[A-Z][0-9]{2}(?:\.[0-9A-Z]{1,4})?$")
RXCUI_RE = re.compile(r"^[0-9]+$")
ALLOWED_TYPES = {item.value for item in EntityType}
ALLOWED_ASSERTIONS = {item.value for item in AssertionType}
CANDIDATE_TYPES = {EntityType.DIAGNOSIS.value, EntityType.MEDICATION.value}
ASSERTION_TYPES = {
    EntityType.SYMPTOM.value,
    EntityType.DIAGNOSIS.value,
    EntityType.MEDICATION.value,
}


class SubmissionValidationError(ValueError):
    pass


def sanitize_candidates(
    entity_type: EntityType,
    candidates: Iterable[str],
    max_candidates: int,
    allowlist: set[str] | None = None,
) -> list[str]:
    pattern = ICD10_RE if entity_type == EntityType.DIAGNOSIS else RXCUI_RE
    result: list[str] = []
    for value in candidates:
        candidate = str(value).strip().upper() if entity_type == EntityType.DIAGNOSIS else str(value).strip()
        if not pattern.fullmatch(candidate):
            continue
        if allowlist is not None and candidate not in allowlist:
            continue
        if candidate not in result:
            result.append(candidate)
        if len(result) >= max_candidates:
            break
    return result


def validate_submission_record(raw_text: str, entities: list[dict[str, Any]]) -> None:
    if not isinstance(entities, list):
        raise SubmissionValidationError("Top-level JSON must be a list")

    seen: set[tuple[int, int, str]] = set()
    previous_position: tuple[int, int] | None = None

    for index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            raise SubmissionValidationError(f"Entity {index} must be an object")

        required = {"text", "type", "assertions", "position"}
        missing = required - set(entity)
        if missing:
            raise SubmissionValidationError(f"Entity {index} missing fields: {sorted(missing)}")

        entity_type = entity["type"]
        if entity_type not in ALLOWED_TYPES:
            raise SubmissionValidationError(f"Entity {index} has invalid type: {entity_type!r}")

        has_candidates = "candidates" in entity
        if (entity_type in CANDIDATE_TYPES) != has_candidates:
            raise SubmissionValidationError(
                f"Entity {index}: candidates field does not match type {entity_type}"
            )

        position = entity["position"]
        if (
            not isinstance(position, list)
            or len(position) != 2
            or not all(isinstance(value, int) and not isinstance(value, bool) for value in position)
        ):
            raise SubmissionValidationError(f"Entity {index} has invalid position")
        start, end = position
        if not (0 <= start < end <= len(raw_text)):
            raise SubmissionValidationError(
                f"Entity {index} position [{start}, {end}] is outside text length {len(raw_text)}"
            )
        if raw_text[start:end] != entity["text"]:
            raise SubmissionValidationError(
                f"Entity {index} text does not match raw_text[{start}:{end}]"
            )

        assertions = entity["assertions"]
        if not isinstance(assertions, list) or any(
            item not in ALLOWED_ASSERTIONS for item in assertions
        ):
            raise SubmissionValidationError(f"Entity {index} has invalid assertions")
        if entity_type not in ASSERTION_TYPES and assertions:
            raise SubmissionValidationError(
                f"Entity {index}: assertions are not allowed for {entity_type}"
            )
        if len(assertions) != len(set(assertions)):
            raise SubmissionValidationError(f"Entity {index} has duplicate assertions")

        if has_candidates:
            candidates = entity["candidates"]
            if not isinstance(candidates, list) or any(
                not isinstance(item, str) for item in candidates
            ):
                raise SubmissionValidationError(f"Entity {index} has invalid candidates")
            pattern = ICD10_RE if entity_type == EntityType.DIAGNOSIS.value else RXCUI_RE
            if any(not pattern.fullmatch(item) for item in candidates):
                raise SubmissionValidationError(f"Entity {index} has malformed candidate code")
            if len(candidates) != len(set(candidates)):
                raise SubmissionValidationError(f"Entity {index} has duplicate candidates")

        current = (start, end)
        if previous_position is not None and current < previous_position:
            raise SubmissionValidationError("Entities are not sorted by position")
        previous_position = current

        identity = (start, end, entity_type)
        if identity in seen:
            raise SubmissionValidationError(f"Duplicate entity at index {index}")
        seen.add(identity)


def validate_output_directory(input_dir: Path, output_dir: Path) -> list[str]:
    errors: list[str] = []
    input_files = sorted(input_dir.glob("*.txt"), key=lambda path: int(path.stem))
    expected_names = {f"{path.stem}.json" for path in input_files}
    actual_names = {path.name for path in output_dir.glob("*.json")}

    missing = sorted(expected_names - actual_names)
    extra = sorted(actual_names - expected_names)
    if missing:
        errors.append(f"Missing output files: {', '.join(missing)}")
    if extra:
        errors.append(f"Unexpected output files: {', '.join(extra)}")

    for input_path in input_files:
        output_path = output_dir / f"{input_path.stem}.json"
        if not output_path.exists():
            continue
        try:
            raw_text = input_path.read_text(encoding="utf-8")
            entities = json.loads(output_path.read_text(encoding="utf-8"))
            validate_submission_record(raw_text, entities)
        except (OSError, UnicodeError, json.JSONDecodeError, SubmissionValidationError) as exc:
            errors.append(f"{output_path.name}: {exc}")
    return errors

