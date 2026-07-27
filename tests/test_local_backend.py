from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from medical_coder.local_llm import LocalMedicalExtractor, extract_json_object
from medical_coder.models import AlignedEntity, AssertionType, EntityType
from medical_coder.terminology import TerminologyIndex, normalize_term


class FakeGenerator:
    def generate(self, *, instructions: str, input_text: str) -> str:
        if "<normalization_input>" in input_text:
            return (
                '{"mappings":[{"entity_index":0,'
                '"candidates":["K21.9","Z99.9"]}]}'
            )
        return '{"entities":[]}'


class SingleEntityGenerator:
    def generate(self, *, instructions: str, input_text: str) -> str:
        return (
            '{"text":"Thiếu men G6PD","type":"CHẨN_ĐOÁN",'
            '"assertions":[{"isNegated":false,"isFamily":false,'
            '"isHistorical":true}],"start_hint":0}'
        )


class MixedEntityTypeGenerator:
    def generate(self, *, instructions: str, input_text: str) -> str:
        return (
            '{"entities":['
            '{"text":"chảy máu chân răng","type":"TRIỆU_CHỨNG",'
            '"assertions":[],"start_hint":0},'
            '{"text":"vệ sinh răng miệng kém","type":"NGUYÊN_NHÂN",'
            '"assertions":[],"start_hint":22}'
            "]}"
        )


class LocalBackendTests(unittest.TestCase):
    def test_extract_json_object_ignores_fence_and_surrounding_text(self) -> None:
        actual = extract_json_object(
            'Kết quả:\n```json\n{"entities": []}\n```\n'
        )
        self.assertEqual({"entities": []}, actual)

    def test_vietnamese_retrieval_matches_without_diacritics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "icd.tsv"
            path.write_text(
                "code\tlabel\taliases\n"
                "K21.9\tGastro-esophageal reflux disease\t"
                "trào ngược dạ dày thực quản|GERD\n"
                "I10\tEssential hypertension\ttăng huyết áp\n",
                encoding="utf-8",
            )
            index = TerminologyIndex(path)

            result = index.retrieve("trao nguoc da day thuc quan", top_k=2)

            self.assertEqual("K21.9", result[0].code)
            self.assertTrue(result[0].exact)

    def test_normalize_term_never_changes_raw_text_in_place(self) -> None:
        raw = "Tăng  huyết áp"
        self.assertEqual("tang huyet ap", normalize_term(raw))
        self.assertEqual("Tăng  huyết áp", raw)

    def test_extractor_accepts_single_entity_and_boolean_assertions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            extractor = LocalMedicalExtractor(
                model_path="fake-local-model",
                quantization="none",
                cache_dir=Path(temporary) / "cache",
                icd_terminology=None,
                rxnorm_terminology=None,
                embedding_model=None,
                embedding_device="cpu",
                retrieval_top_k=10,
                max_candidates=3,
                max_input_tokens=4096,
                max_new_tokens=1024,
                generator=SingleEntityGenerator(),  # type: ignore[arg-type]
            )

            result = extractor.extract("1", "Thiếu men G6PD")

            self.assertEqual(1, len(result.entities))
            self.assertEqual("Thiếu men G6PD", result.entities[0].text)
            self.assertEqual(
                [AssertionType.HISTORICAL],
                result.entities[0].assertions,
            )

    def test_extractor_drops_only_entities_outside_submission_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            extractor = LocalMedicalExtractor(
                model_path="fake-local-model",
                quantization="none",
                cache_dir=Path(temporary) / "cache",
                icd_terminology=None,
                rxnorm_terminology=None,
                embedding_model=None,
                embedding_device="cpu",
                retrieval_top_k=10,
                max_candidates=3,
                max_input_tokens=4096,
                max_new_tokens=1024,
                generator=MixedEntityTypeGenerator(),  # type: ignore[arg-type]
            )

            result = extractor.extract(
                "1",
                "chảy máu chân răng do vệ sinh răng miệng kém",
            )

            self.assertEqual(["chảy máu chân răng"], [item.text for item in result.entities])

    def test_normalizer_rejects_code_outside_retrieved_choices(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kb_path = root / "icd.tsv"
            kb_path.write_text(
                "code\tlabel\taliases\n"
                "K21.9\tGastro-esophageal reflux disease\t"
                "trào ngược dạ dày thực quản\n",
                encoding="utf-8",
            )
            extractor = LocalMedicalExtractor(
                model_path="fake-local-model",
                quantization="none",
                cache_dir=root / "cache",
                icd_terminology=kb_path,
                rxnorm_terminology=None,
                embedding_model=None,
                embedding_device="cpu",
                retrieval_top_k=10,
                max_candidates=3,
                max_input_tokens=4096,
                max_new_tokens=1024,
                generator=FakeGenerator(),  # type: ignore[arg-type]
            )
            raw_text = "Chẩn đoán trào ngược dạ dày thực quản."
            start = raw_text.index("trào ngược")
            entity = AlignedEntity(
                text="trào ngược dạ dày thực quản",
                type=EntityType.DIAGNOSIS,
                assertions=[],
                position=(start, start + len("trào ngược dạ dày thực quản")),
            )

            result = extractor.normalize("1", raw_text, [entity])

            self.assertEqual(["K21.9"], result.mappings[0].candidates)


if __name__ == "__main__":
    unittest.main()
