from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from medical_coder.models import AlignedEntity, AssertionType, EntityType
from medical_coder.pipeline import create_submission_zip
from medical_coder.validation import (
    SubmissionValidationError,
    sanitize_candidates,
    validate_submission_record,
)


class ValidationTests(unittest.TestCase):
    def test_submission_schema_omits_candidates_for_symptom(self) -> None:
        entity = AlignedEntity(
            text="ho",
            type=EntityType.SYMPTOM,
            assertions=[],
            position=(0, 2),
        )
        result = entity.to_submission_dict()

        self.assertNotIn("candidates", result)
        validate_submission_record("ho", [result])

    def test_medication_has_candidates_and_historical_assertion(self) -> None:
        entity = AlignedEntity(
            text="aspirin",
            type=EntityType.MEDICATION,
            assertions=[AssertionType.HISTORICAL],
            position=(0, 7),
            candidates=["1191"],
        )
        result = entity.to_submission_dict()

        self.assertEqual(["1191"], result["candidates"])
        validate_submission_record("aspirin", [result])

    def test_invalid_span_is_rejected(self) -> None:
        with self.assertRaises(SubmissionValidationError):
            validate_submission_record(
                "ho",
                [
                    {
                        "text": "ho",
                        "type": "TRIỆU_CHỨNG",
                        "assertions": [],
                        "position": [0, 1],
                    }
                ],
            )

    def test_candidate_sanitization(self) -> None:
        actual = sanitize_candidates(
            EntityType.DIAGNOSIS,
            [" k21.9 ", "NOT-A-CODE", "K21.9", "K21.0", "K22.0"],
            max_candidates=2,
        )
        self.assertEqual(["K21.9", "K21.0"], actual)

    def test_zip_has_required_output_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "output"
            output_dir.mkdir()
            (output_dir / "1.json").write_text(
                json.dumps([], ensure_ascii=False),
                encoding="utf-8",
            )
            zip_path = root / "output.zip"

            create_submission_zip(output_dir, zip_path)

            with zipfile.ZipFile(zip_path) as archive:
                self.assertEqual(["output/1.json"], archive.namelist())


if __name__ == "__main__":
    unittest.main()

