# -*- coding: utf-8 -*-

import json
import tempfile
import unittest
from pathlib import Path

from tests.teacher_label_regression import evaluate


class TeacherLabelRegressionTests(unittest.TestCase):
    def test_evaluate_reports_raw_and_final_metrics_separately(self) -> None:
        rows = [
            {
                "question_id": "q1",
                "difficulty_level_raw": "基础题",
                "difficulty_rating": {"difficulty_level": "中等题"},
                "postprocess_actions": [
                    {"rule": "medium_fix", "from": "基础题", "to": "中等题"}
                ],
            },
            {
                "question_id": "q2",
                "difficulty_level_raw": "中等题",
                "difficulty_rating": {"difficulty_level": "中等题"},
                "postprocess_actions": [],
            },
        ]
        labels = {"q1": "中等", "q2": "较易"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = evaluate(path, labels)

        self.assertEqual(result["evaluated"], 2)
        self.assertEqual(result["exact_match_rate"], 0.5)
        self.assertEqual(result["prediction_distribution"]["中等题"], 2)
        self.assertEqual(result["raw_evaluation"]["evaluated"], 2)
        self.assertEqual(result["raw_evaluation"]["exact_match_rate"], 0.0)
        self.assertEqual(result["raw_evaluation"]["prediction_distribution"]["基础题"], 1)
        self.assertEqual(result["postprocess_rules"]["medium_fix"]["improved"], 1)


if __name__ == "__main__":
    unittest.main()
