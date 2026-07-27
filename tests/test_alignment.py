from __future__ import annotations

import unittest

from medical_coder.alignment import align_mentions
from medical_coder.models import AssertionType, EntityType, ExtractedMention


class AlignmentTests(unittest.TestCase):
    def test_repeated_mentions_use_distinct_offsets(self) -> None:
        text = "Bệnh nhân ho, sau đó vẫn ho."
        mentions = [
            ExtractedMention(
                text="ho",
                type=EntityType.SYMPTOM,
                assertions=[],
                start_hint=10,
            ),
            ExtractedMention(
                text="ho",
                type=EntityType.SYMPTOM,
                assertions=[],
                start_hint=28,
            ),
        ]

        entities, issues = align_mentions(text, mentions)

        self.assertEqual([], issues)
        self.assertEqual([(10, 12), (25, 27)], [item.position for item in entities])
        self.assertTrue(all(text[s:e] == item.text for item in entities for s, e in [item.position]))

    def test_casefold_and_whitespace_repair_preserves_raw_quote(self) -> None:
        text = "Đã dùng ASPIRIN   81 mg trước nhập viện."
        mentions = [
            ExtractedMention(
                text="aspirin 81 mg",
                type=EntityType.MEDICATION,
                assertions=[AssertionType.HISTORICAL],
                start_hint=8,
            )
        ]

        entities, issues = align_mentions(text, mentions)

        self.assertEqual([], issues)
        self.assertEqual("ASPIRIN   81 mg", entities[0].text)
        self.assertEqual((8, 23), entities[0].position)
        self.assertEqual([AssertionType.HISTORICAL], entities[0].assertions)

    def test_missing_quote_is_dropped(self) -> None:
        mentions = [
            ExtractedMention(
                text="không tồn tại",
                type=EntityType.DIAGNOSIS,
                assertions=[],
                start_hint=0,
            )
        ]

        entities, issues = align_mentions("Văn bản ngắn.", mentions)

        self.assertEqual([], entities)
        self.assertEqual(1, len(issues))

    def test_assertions_removed_from_test_result(self) -> None:
        text = "WBC: 14,43"
        mentions = [
            ExtractedMention(
                text="14,43",
                type=EntityType.TEST_RESULT,
                assertions=[AssertionType.NEGATED],
                start_hint=5,
            )
        ]

        entities, _ = align_mentions(text, mentions)

        self.assertEqual([], entities[0].assertions)


if __name__ == "__main__":
    unittest.main()
