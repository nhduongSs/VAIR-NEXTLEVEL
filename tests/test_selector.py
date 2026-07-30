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
    """Scores where `entity_type` beats NONE by `margin`.

    The other four types sit far below so that `max()` picks `entity_type` even
    when the margin is negative — i.e. when NONE wins.
    """
    row = [-1e3] * 5
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


class RejectorTests(unittest.TestCase):
    """Dropping baseline spans the teacher reads as non-concepts."""

    def setUp(self):
        self.text = "Các triệu chứng hiện tại và đau đầu nhiều"
        self.spans = [
            ScoredSpan(0, 24, EntityType.SYMPTOM, 0.3),   # tiêu đề mục
            ScoredSpan(28, 40, EntityType.SYMPTOM, 0.6),  # khái niệm thật
        ]

    def test_span_is_dropped_when_none_wins_by_the_margin(self):
        selector = SpanSelector(
            StubTeacher([six_way_row(EntityType.SYMPTOM, -3.0),
                         six_way_row(EntityType.SYMPTOM, 4.0)])
        )
        kept = selector.reject_spans(self.text, self.spans, margin=1.0)
        self.assertEqual([(s.start, s.end) for s in kept], [(28, 40)])

    def test_span_survives_when_none_wins_by_less_than_the_margin(self):
        selector = SpanSelector(
            StubTeacher([six_way_row(EntityType.SYMPTOM, -0.5),
                         six_way_row(EntityType.SYMPTOM, 4.0)])
        )
        self.assertEqual(len(selector.reject_spans(self.text, self.spans, margin=1.0)), 2)

    def test_a_second_teacher_must_also_call_it_junk(self):
        """Dropping is irreversible, so one dissenting vote is enough to keep."""
        selector = SpanSelector(
            StubTeacher([six_way_row(EntityType.SYMPTOM, -3.0),
                         six_way_row(EntityType.SYMPTOM, 4.0)]),
            StubTeacher([six_way_row(EntityType.SYMPTOM, 2.0),
                         six_way_row(EntityType.SYMPTOM, 4.0)]),
        )
        self.assertEqual(len(selector.reject_spans(self.text, self.spans, margin=1.0)), 2)

    def test_rejection_is_off_unless_a_margin_is_configured(self):
        teacher = StubTeacher([])
        selector = SpanSelector(teacher)
        self.assertIsNone(selector.reject_margin)
        self.assertEqual(selector.select(self.text, [], []), [])
        self.assertEqual(teacher.prompts, [])


class BreakEvenTests(unittest.TestCase):
    """The arithmetic that justifies rejecting at all."""

    @staticmethod
    def gain(text_score, alpha, dropped, truly_spurious, denominator, numerator):
        new_numerator = numerator - alpha * (dropped - truly_spurious)
        return new_numerator / (denominator - 2 * truly_spurious) - text_score

    def test_rejecting_only_spurious_spans_always_gains(self):
        d, n = 3625.0, 0.302302 * 3625.0
        self.assertGreater(self.gain(0.302302, 0.79, 200, 200, d, n), 0)

    def test_break_even_matches_the_closed_form(self):
        """Profitable while dropped/spurious < 1 + 2*text/alpha."""
        text, alpha, d = 0.302302, 0.79, 3625.0
        n = text * d
        ratio = 1 + 2 * text / alpha
        spurious = 200
        self.assertGreater(self.gain(text, alpha, int(spurious * (ratio - 0.1)), spurious, d, n), 0)
        self.assertLess(self.gain(text, alpha, int(spurious * (ratio + 0.1)), spurious, d, n), 0)


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
