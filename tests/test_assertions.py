"""Luật assertion — điều quan trọng nhất là nó KHÔNG bắn quá tay."""
import unittest

from medical_coder.assertions import detect
from medical_coder.models import AssertionType, EntityType

SYMPTOM = EntityType.SYMPTOM


def names(document, start, end, entity_type=SYMPTOM, **kw):
    return [a.value for a in detect(document, start, end, entity_type, **kw)]


class NegationTests(unittest.TestCase):
    def test_detects_a_plain_negation(self):
        text = "Bệnh nhân không sốt"
        self.assertEqual(names(text, 16, 19), ["isNegated"])

    def test_detects_a_multiword_cue(self):
        text = "Chưa ghi nhận đau ngực"
        self.assertEqual(names(text, 14, 22), ["isNegated"])

    def test_clause_break_closes_the_scope(self):
        """'không sốt, nhưng có ho' — ho KHÔNG bị phủ định."""
        text = "Bệnh nhân không sốt, nhưng có ho nhiều"
        self.assertEqual(names(text, 30, 32), [])

    def test_full_stop_closes_the_scope(self):
        text = "Không sốt. Bệnh nhân ho nhiều"
        self.assertEqual(names(text, 21, 23), [])

    def test_cue_beyond_the_window_is_ignored(self):
        text = "không " + "x" * 60 + " ho"
        self.assertEqual(names(text, len(text) - 2, len(text)), [])

    def test_never_crosses_a_line_boundary(self):
        text = "Không sốt\nho nhiều"
        self.assertEqual(names(text, 10, 12), [])


class HistoryTests(unittest.TestCase):
    def test_local_cue_marks_history(self):
        text = "Tiền sử hen phế quản"
        self.assertEqual(names(text, 8, 20), ["isHistorical"])

    def test_section_heading_is_ignored_by_default(self):
        """Mặc định KHÔNG suy theo tiêu đề mục: bản đó gán 27.6% concept với
        precision 57-62%, và lợi ích của nó lệch 2.1 lần giữa hai bộ nhãn."""
        text = "1. Tiền sử bệnh\n    Bệnh nhân đau đầu nhiều"
        self.assertEqual(names(text, 31, 38), [])

    def test_section_heading_used_when_explicitly_enabled(self):
        text = "1. Tiền sử bệnh\n    Bệnh nhân đau đầu nhiều"
        self.assertEqual(names(text, 31, 38, use_sections=True), ["isHistorical"])


class ScopeTests(unittest.TestCase):
    def test_test_name_never_gets_assertions(self):
        text = "Không thấy công thức máu"
        self.assertEqual(names(text, 11, 24, EntityType.TEST_NAME), [])

    def test_test_result_never_gets_assertions(self):
        text = "Tiền sử 120 mg/dL"
        self.assertEqual(names(text, 8, 17, EntityType.TEST_RESULT), [])

    def test_family_is_off_by_default(self):
        text = "Mẹ bệnh nhân bị tăng huyết áp"
        self.assertEqual(names(text, 16, 29, EntityType.DIAGNOSIS), [])

    def test_family_when_explicitly_enabled(self):
        text = "Mẹ bị tăng huyết áp"
        self.assertIn(
            AssertionType.FAMILY.value,
            names(text, 6, 19, EntityType.DIAGNOSIS, enable_family=True),
        )

    def test_negation_and_history_can_combine(self):
        text = "Tiền sử không hen"
        self.assertEqual(
            sorted(names(text, 14, 17)), ["isHistorical", "isNegated"]
        )

    def test_plain_mention_gets_nothing(self):
        self.assertEqual(names("Bệnh nhân ho nhiều", 10, 12), [])


if __name__ == "__main__":
    unittest.main()
