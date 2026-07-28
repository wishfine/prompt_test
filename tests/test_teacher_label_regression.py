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

    def test_evaluate_prefers_multi_call_raw_consensus(self) -> None:
        row = {
            "question_id": "q1",
            "difficulty_level_raw": "基础题",
            "multi_call_raw_level": "中等题",
            "difficulty_rating": {"difficulty_level": "中等题"},
            "postprocess_actions": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            result = evaluate(path, {"q1": "中等"})

        self.assertEqual(result["raw_evaluation"]["exact_match_rate"], 1.0)
        self.assertEqual(
            result["raw_evaluation"]["prediction_distribution"]["中等题"],
            1,
        )

    def test_evaluate_reports_ensemble_calibration_effect(self) -> None:
        rows = [
            {
                "question_id": "q1",
                "multi_call_raw_level": "送分题",
                "difficulty_rating": {"difficulty_level": "基础题"},
                "postprocess_actions": [],
                "lite_self_consistency": {
                    "majority_level_before_calibration": "送分题",
                    "calibration_actions": [
                        {
                            "rule": "structured_easy_disagreement_guard",
                            "from": "送分题",
                            "to": "基础题",
                        }
                    ],
                },
            },
            {
                "question_id": "q2",
                "multi_call_raw_level": "送分题",
                "difficulty_rating": {"difficulty_level": "基础题"},
                "postprocess_actions": [],
                "lite_self_consistency": {
                    "majority_level_before_calibration": "送分题",
                    "calibration_actions": [
                        {
                            "rule": "structured_easy_disagreement_guard",
                            "from": "送分题",
                            "to": "基础题",
                        }
                    ],
                },
            },
        ]
        labels = {"q1": "较易", "q2": "容易"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n" for row in rows
                ),
                encoding="utf-8",
            )
            result = evaluate(path, labels)

        self.assertEqual(
            result["ensemble_baseline_evaluation"]["exact_match_rate"],
            0.5,
        )
        stats = result["ensemble_calibration_rules"][
            "structured_easy_disagreement_guard"
        ]
        self.assertEqual(stats["triggered"], 2)
        self.assertEqual(stats["improved"], 1)
        self.assertEqual(stats["worsened"], 1)


if __name__ == "__main__":
    unittest.main()
