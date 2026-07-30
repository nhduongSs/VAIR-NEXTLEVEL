"""Selector logic tests.

The Qwen teachers need a GPU, so these drive :class:`SpanSelector` with a stub
that returns fixed logits. That covers the parts that actually decide output —
which spans get re-typed, and which additions survive consensus — without
loading any weights.
"""
import unittest

from medical_coder.gliner_ner import ScoredSpan
from medical_coder.models import EntityType
from medical_coder.selector import ORDERED_TYPES, SpanSelector, line_context


class StubTeacher:
    """Returns a scripted score row per prompt, in call order."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.entity_digits = [[i] for i in range(5)]
        self.none_digits = [5]
        self.parameters = 1_000_000
        self.prompts = []

    def chat_prompt(self, system, user):
        return user

    def group_scores(self, prompts, groups, batch_size, max_length):
        self.prompts.extend(prompts)
        return [self.rows.pop(0) for _ in prompts]


def five_way_row(entity_type):
    return [10.0 if t is entity_type else 0.0 for t in ORDERED_TYPES]


def six_way_row(entity_type, margin):
    row = [0.0] * 5
    row[ORDERED_TYPES.index(entity_type)] = margin
    return row + [0.0]


class LineContextTests(unittest.TestCase):
    def test_includes_the_preceding_non_empty_line_as_a_section_hint(self):
        text = "Tiền sử bệnh\n\nbệnh nhân bị ho nhiều\n"
        context = line_context(text, 22, 24)
        self.assertIn("Tiền sử bệnh", context)
        self.assertIn("bệnh nhân bị ho nhiều", context)

    def test_first_line_has_no_section_hint(self):
        self.assertEqual(line_context("đau ngực nhiều", 0, 8), "đau ngực nhiều")


class CorrectorTests(unittest.TestCase):
    def test_symptom_is_retyped_when_the_teacher_reads_a_diagnosis(self):
        text = "bệnh nhân bị tăng huyết áp"
        spans = [ScoredSpan(13, 26, EntityType.SYMPTOM, 0.4)]
        selector = SpanSelector(StubTeacher([five_way_row(EntityType.DIAGNOSIS)]))
        result = selector.correct_types(text, spans)
        self.assertEqual(result[0].type, EntityType.DIAGNOSIS)
        self.assertEqual((result[0].start, result[0].end), (13, 26))

    def test_symptom_is_left_alone_when_the_teacher_agrees(self):
        text = "bệnh nhân bị đau đầu"
        spans = [ScoredSpan(13, 20, EntityType.SYMPTOM, 0.4)]
        selector = SpanSelector(StubTeacher([five_way_row(EntityType.SYMPTOM)]))
        self.assertEqual(selector.correct_types(text, spans)[0].type, EntityType.SYMPTOM)

    def test_other_types_are_never_sent_to_the_corrector(self):
        text = "paracetamol 500 mg"
        spans = [ScoredSpan(0, 18, EntityType.MEDICATION, 0.9)]
        teacher = StubTeacher([])
        self.assertEqual(SpanSelector(teacher).correct_types(text, spans), spans)
        self.assertEqual(teacher.prompts, [])


class AdditionTests(unittest.TestCase):
    def setUp(self):
        self.text = "bệnh nhân sốt cao và mệt mỏi nhiều"
        self.baseline = [ScoredSpan(10, 17, EntityType.SYMPTOM, 0.5)]
        self.raw = [ScoredSpan(21, 28, EntityType.SYMPTOM, 0.05)]

    def _selector(self, primary_row, secondary_row):
        return SpanSelector(StubTeacher([primary_row]), StubTeacher([secondary_row]))

    def test_added_when_both_teachers_agree_with_margin(self):
        selector = self._selector(
            six_way_row(EntityType.SYMPTOM, 2.0), six_way_row(EntityType.SYMPTOM, 2.0)
        )
        additions = selector.propose_additions(self.text, self.baseline, self.raw)
        self.assertEqual([(s.start, s.end, s.type) for s in additions],
                         [(21, 28, EntityType.SYMPTOM)])

    def test_rejected_when_teachers_disagree(self):
        selector = self._selector(
            six_way_row(EntityType.SYMPTOM, 2.0), six_way_row(EntityType.TEST_NAME, 2.0)
        )
        self.assertEqual(selector.propose_additions(self.text, self.baseline, self.raw), [])

    def test_rejected_when_the_margin_over_none_is_too_small(self):
        selector = self._selector(
            six_way_row(EntityType.SYMPTOM, 0.2), six_way_row(EntityType.SYMPTOM, 2.0)
        )
        self.assertEqual(selector.propose_additions(self.text, self.baseline, self.raw), [])

    def test_candidate_bearing_types_are_never_added(self):
        """A spurious diagnosis is charged to the candidate denominator as well."""
        selector = self._selector(
            six_way_row(EntityType.DIAGNOSIS, 5.0), six_way_row(EntityType.DIAGNOSIS, 5.0)
        )
        self.assertEqual(selector.propose_additions(self.text, self.baseline, self.raw), [])

    def test_spans_overlapping_the_baseline_are_not_reconsidered(self):
        overlapping = [ScoredSpan(12, 16, EntityType.SYMPTOM, 0.05)]
        selector = self._selector(
            six_way_row(EntityType.SYMPTOM, 5.0), six_way_row(EntityType.SYMPTOM, 5.0)
        )
        self.assertEqual(selector.propose_additions(self.text, self.baseline, overlapping), [])

    def test_no_additions_without_a_second_teacher(self):
        selector = SpanSelector(StubTeacher([]))
        self.assertEqual(selector.propose_additions(self.text, self.baseline, self.raw), [])


if __name__ == "__main__":
    unittest.main()
