from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from medical_coder.models import (
    AssertionType,
    CandidatePrediction,
    EntityType,
    ExtractedMention,
    ExtractionResponse,
    NormalizationResponse,
)
from medical_coder.pipeline import PipelineConfig, process_file
from medical_coder.validation import validate_submission_record


class FakeExtractor:
    def extract(self, record_id: str, raw_text: str) -> ExtractionResponse:
        return ExtractionResponse(
            entities=[
                ExtractedMention(
                    text="aspirin",
                    type=EntityType.MEDICATION,
                    assertions=[AssertionType.HISTORICAL],
                    start_hint=raw_text.index("aspirin"),
                ),
                ExtractedMention(
                    text="ho",
                    type=EntityType.SYMPTOM,
                    assertions=[AssertionType.NEGATED],
                    start_hint=raw_text.index("ho"),
                ),
            ]
        )

    def normalize(
        self,
        record_id: str,
        raw_text: str,
        entities: list,
    ) -> NormalizationResponse:
        return NormalizationResponse(
            mappings=[CandidatePrediction(entity_index=0, candidates=["1191"])]
        )


class PipelineTests(unittest.TestCase):
    def test_process_file_end_to_end_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            cache_dir = root / "cache"
            input_dir.mkdir()
            raw_text = "Tiền sử dùng aspirin và không ho."
            input_path = input_dir / "1.txt"
            input_path.write_text(raw_text, encoding="utf-8")
            config = PipelineConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                cache_dir=cache_dir,
                model="fake",
                reasoning_effort="none",
                workers=1,
                max_candidates=3,
                overwrite=False,
                selected_ids=None,
                icd_allowlist=None,
                rxnorm_allowlist=None,
            )

            record_id, count, issues = process_file(
                input_path,
                config,
                FakeExtractor(),  # type: ignore[arg-type]
            )

            output = json.loads((output_dir / "1.json").read_text(encoding="utf-8"))
            self.assertEqual("1", record_id)
            self.assertEqual(2, count)
            self.assertEqual([], issues)
            self.assertEqual(["1191"], output[0]["candidates"])
            self.assertNotIn("candidates", output[1])
            validate_submission_record(raw_text, output)


if __name__ == "__main__":
    unittest.main()

