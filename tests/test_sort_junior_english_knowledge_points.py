import unittest

from scripts.sort_junior_english_knowledge_points import (
    split_output_labels,
    validate_ordered_output,
)


class SortKnowledgePointsTests(unittest.TestCase):
    def test_split_output_labels_removes_only_separator_whitespace_and_empty_items(self):
        self.assertEqual(split_output_labels(" A ; B;;A "), ["A", "B", "A"])
        self.assertEqual(split_output_labels(""), [])


    def test_validate_ordered_output_preserves_duplicate_labels_as_a_multiset(self):
        result = validate_ordered_output(
            "核心;辅助;核心",
            {"ordered_output": ["辅助", "核心", "核心"]},
        )
        self.assertEqual(result, ["辅助", "核心", "核心"])


    def test_validate_ordered_output_rejects_invalid_model_contract(self):
        invalid_cases = [
            ("A;B", {"ordered_output": ["A"]}),
            ("A;B", {"ordered_output": ["A", "C"]}),
            ("A;B", {"ordered_output": "A;B"}),
            ("A;B", {}),
        ]
        for source, payload in invalid_cases:
            with self.subTest(source=source, payload=payload):
                with self.assertRaises(ValueError):
                    validate_ordered_output(source, payload)
