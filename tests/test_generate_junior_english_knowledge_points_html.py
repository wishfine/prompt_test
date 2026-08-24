import unittest

from scripts.generate_junior_english_knowledge_points_html import (
    build_summary,
    generate_html,
)


def sample_rows():
    return [
        {
            "question_id": "q1",
            "input": "题干：There ___ a meeting。",
            "original_output": "时态;there be",
            "output": "there be;时态",
            "ordered_output": ["there be", "时态"],
            "sort_status": "success",
        },
        {
            "question_id": "q2",
            "input": "题干：She went home.",
            "original_output": "一般过去时",
            "output": "一般过去时",
            "ordered_output": ["一般过去时"],
            "sort_status": "success",
        },
        {
            "question_id": "q3",
            "input": "题干：bad row",
            "output": "A;B",
            "sort_status": "error",
            "sort_error": "模型输出不是合法标签数组",
        },
    ]


class KnowledgePointsHtmlTests(unittest.TestCase):
    def test_build_summary_counts_status_changes_and_primary_labels(self):
        summary = build_summary(sample_rows())
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["success"], 2)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["changed"], 1)
        self.assertEqual(summary["unchanged"], 1)
        self.assertEqual(
            summary["top_primary"][0], {"label": "there be", "count": 1}
        )

    def test_generate_html_contains_review_fields_and_filter_hooks(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "review.html"
            generate_html(sample_rows(), output_path, "初中英语知识点排序验收")
            html = output_path.read_text(encoding="utf-8")

        self.assertIn("初中英语知识点排序验收", html)
        self.assertIn("q1", html)
        self.assertIn("原始知识点顺序", html)
        self.assertIn("排序后知识点顺序", html)
        self.assertIn('<details class="question-details" open>', html)
        self.assertIn("人工复核", html)
        self.assertIn("localStorage", html)
        self.assertIn('data-status="changed"', html)
        self.assertIn("筛选", html)


if __name__ == "__main__":
    unittest.main()
