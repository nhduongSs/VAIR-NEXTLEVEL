import unittest

from medical_coder.exact_link import (
    ExactAliasIndex,
    PrecisionFirstLinker,
    clean_mention,
    strip_diacritics,
)
from medical_coder.gliner_ner import (
    ScoredSpan,
    extend_medication_span,
    iter_chunks,
    resolve_overlaps,
)
from medical_coder.icd_vn import dotted_code, iter_who_synonyms
from medical_coder.models import EntityType
from medical_coder.terminology import TerminologyEntry


def index(*entries):
    return ExactAliasIndex(
        TerminologyEntry(code=code, label=label, aliases=tuple(aliases))
        for code, label, aliases in entries
    )


class MentionCleaningTests(unittest.TestCase):
    def test_drug_route_and_frequency_are_stripped(self):
        self.assertEqual(
            clean_mention("amlodipine 10 mg po daily", EntityType.MEDICATION),
            "amlodipine 10 mg",
        )

    def test_diagnosis_mentions_are_left_alone(self):
        self.assertEqual(
            clean_mention("viêm  phổi", EntityType.DIAGNOSIS), "viêm phổi"
        )

    def test_strip_diacritics_handles_vietnamese_d(self):
        self.assertEqual(strip_diacritics("đái tháo đường"), "dai thao duong")


class ExactAliasIndexTests(unittest.TestCase):
    def test_matches_label_and_alias_case_insensitively(self):
        table = index(("K21.9", "Bệnh trào ngược dạ dày thực quản", ["GERD"]))
        self.assertEqual(table.lookup("bệnh trào ngược dạ dày thực quản"), {"K21.9"})
        self.assertEqual(table.lookup("gerd"), {"K21.9"})

    def test_falls_back_to_diacritic_insensitive_match(self):
        table = index(("E11.9", "Đái tháo đường típ 2", []))
        self.assertEqual(table.lookup("dai thao duong tip 2"), {"E11.9"})


class PrecisionFirstLinkerTests(unittest.TestCase):
    def test_unique_match_emits_one_code(self):
        linker = PrecisionFirstLinker(icd_index=index(("I50.9", "Suy tim", [])))
        self.assertEqual(linker.link("suy tim", EntityType.DIAGNOSIS), ["I50.9"])

    def test_ambiguous_match_stays_silent(self):
        table = index(("A00", "Bệnh tả", []), ("A00.9", "Bệnh tả", []))
        linker = PrecisionFirstLinker(icd_index=table)
        self.assertEqual(linker.link("bệnh tả", EntityType.DIAGNOSIS), [])

    def test_bare_category_is_remapped_to_its_unspecified_leaf(self):
        table = index(("J18", "Viêm phổi", []), ("J18.9", "Viêm phổi, không đặc hiệu", []))
        linker = PrecisionFirstLinker(icd_index=table)
        self.assertEqual(linker.link("viêm phổi", EntityType.DIAGNOSIS), ["J18.9"])

    def test_remap_is_skipped_when_the_leaf_does_not_exist(self):
        linker = PrecisionFirstLinker(icd_index=index(("J18", "Viêm phổi", [])))
        self.assertEqual(linker.link("viêm phổi", EntityType.DIAGNOSIS), ["J18"])

    def test_non_candidate_types_never_link(self):
        linker = PrecisionFirstLinker(icd_index=index(("I50.9", "Suy tim", [])))
        self.assertEqual(linker.link("suy tim", EntityType.SYMPTOM), [])


class GlinerHelperTests(unittest.TestCase):
    def test_chunks_are_verbatim_and_offsets_are_exact(self):
        text = "dòng một\ndòng hai\ndòng ba\n"
        for start, chunk in iter_chunks(text, 12):
            self.assertEqual(text[start : start + len(chunk)], chunk)
        self.assertEqual("".join(chunk for _, chunk in iter_chunks(text, 12)), text)

    def test_medication_span_absorbs_dosage_tokens(self):
        text = "dùng amlodipine 10 mg po daily cho bệnh nhân"
        end = extend_medication_span(text, 5, 15)
        self.assertEqual(text[5:end], "amlodipine 10 mg po daily")

    def test_medication_span_stops_at_an_indication(self):
        text = "paracetamol 500 mg để hạ sốt"
        end = extend_medication_span(text, 0, 11)
        self.assertEqual(text[0:end], "paracetamol 500 mg")

    def test_overlap_resolution_prefers_the_higher_score(self):
        spans = [
            ScoredSpan(0, 20, EntityType.SYMPTOM, 0.3),
            ScoredSpan(5, 12, EntityType.DIAGNOSIS, 0.9),
        ]
        self.assertEqual(resolve_overlaps(spans), [spans[1]])


class IcdCatalogTests(unittest.TestCase):
    def test_dotted_code_normalization(self):
        self.assertEqual(dotted_code("A001"), "A00.1")
        self.assertEqual(dotted_code("A00"), "A00")
        self.assertEqual(dotted_code("a00.1"), "A00.1")

    def test_clean_synonyms_are_kept(self):
        self.assertEqual(list(iter_who_synonyms("Bệnh tả cổ điển")), ["Bệnh tả cổ điển"])

    def test_inclusion_notes_and_cross_references_are_rejected(self):
        self.assertEqual(list(iter_who_synonyms("Bao gồm: nhiễm trùng do salmonella")), [])
        self.assertEqual(list(iter_who_synonyms("viêm khớp† (M01.3-*)")), [])
        self.assertEqual(list(iter_who_synonyms("23")), [])


if __name__ == "__main__":
    unittest.main()
