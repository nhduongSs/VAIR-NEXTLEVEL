import unittest

from medical_coder.scoring import (
    jaccard,
    match_concepts,
    score_corpus,
    word_error_rate,
)


def concept(text, type_, position, assertions=None, candidates=None):
    value = {
        "text": text,
        "type": type_,
        "assertions": assertions or [],
        "position": list(position),
    }
    if candidates is not None:
        value["candidates"] = candidates
    return value


class JaccardTests(unittest.TestCase):
    def test_both_empty_scores_one(self):
        self.assertEqual(jaccard(set(), set()), 1.0)

    def test_one_side_empty_scores_zero(self):
        self.assertEqual(jaccard({"a"}, set()), 0.0)
        self.assertEqual(jaccard(set(), {"a"}), 0.0)

    def test_partial_overlap(self):
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)


class WordErrorRateTests(unittest.TestCase):
    def test_identical_is_zero(self):
        self.assertEqual(word_error_rate("đau thượng vị", "đau thượng vị"), 0.0)

    def test_substitution_counts_once(self):
        self.assertAlmostEqual(word_error_rate("đau thượng vị", "đau hạ vị"), 1 / 3)

    def test_insertion_can_exceed_one(self):
        self.assertGreater(word_error_rate("ho", "ho đờm xanh nhiều"), 1.0)


class MatchingTests(unittest.TestCase):
    def test_matches_on_type_and_overlap(self):
        predictions = [concept("tức ngực", "TRIỆU_CHỨNG", (43, 51))]
        truth = [concept("tức ngực", "TRIỆU_CHỨNG", (43, 51))]
        matched, spurious, missed = match_concepts(predictions, truth)
        self.assertEqual(matched, {0: 0})
        self.assertEqual((spurious, missed), ([], []))

    def test_wrong_type_cannot_match_its_twin(self):
        """The rules' double penalty: right text, wrong type is a new concept."""
        predictions = [concept("tức ngực", "CHẨN_ĐOÁN", (43, 51))]
        truth = [concept("tức ngực", "TRIỆU_CHỨNG", (43, 51))]
        matched, spurious, missed = match_concepts(predictions, truth)
        self.assertEqual(matched, {})
        self.assertEqual((spurious, missed), ([0], [0]))


class ScoreCorpusTests(unittest.TestCase):
    def test_truth_against_itself_is_one(self):
        truth = {
            "1": [
                concept("ho", "TRIỆU_CHỨNG", (0, 2), ["isNegated"]),
                concept("viêm phổi", "CHẨN_ĐOÁN", (5, 14), [], ["J18.9"]),
            ]
        }
        self.assertAlmostEqual(score_corpus(truth, truth).final_score, 1.0)

    def test_empty_prediction_beats_wrong_candidate_when_truth_is_empty(self):
        truth = {"1": [concept("viêm phổi", "CHẨN_ĐOÁN", (0, 9), [], [])]}
        silent = {"1": [concept("viêm phổi", "CHẨN_ĐOÁN", (0, 9), [], [])]}
        guessing = {"1": [concept("viêm phổi", "CHẨN_ĐOÁN", (0, 9), [], ["J18.9"])]}
        self.assertEqual(score_corpus(silent, truth).candidates_score, 1.0)
        self.assertEqual(score_corpus(guessing, truth).candidates_score, 0.0)

    def test_spurious_concept_is_counted_twice(self):
        truth = {"1": [concept("ho", "TRIỆU_CHỨNG", (0, 2))]}
        predictions = {
            "1": [
                concept("ho", "TRIỆU_CHỨNG", (0, 2)),
                concept("sốt", "TRIỆU_CHỨNG", (10, 13)),
            ]
        }
        # one matched concept over a denominator of 1 truth + 2 * 1 spurious
        self.assertAlmostEqual(score_corpus(predictions, truth).text_score, 1 / 3)

    def test_extra_candidates_inflate_the_spurious_denominator(self):
        truth = {"1": [concept("viêm phổi", "CHẨN_ĐOÁN", (0, 9), [], ["J18.9"])]}
        lean = {
            "1": [
                concept("viêm phổi", "CHẨN_ĐOÁN", (0, 9), [], ["J18.9"]),
                concept("lao", "CHẨN_ĐOÁN", (40, 43), [], []),
            ]
        }
        greedy = {
            "1": [
                concept("viêm phổi", "CHẨN_ĐOÁN", (0, 9), [], ["J18.9"]),
                concept("lao", "CHẨN_ĐOÁN", (40, 43), [], ["A15.0", "A15.9", "A16.9"]),
            ]
        }
        self.assertGreater(
            score_corpus(lean, truth).candidates_score,
            score_corpus(greedy, truth).candidates_score,
        )


if __name__ == "__main__":
    unittest.main()
