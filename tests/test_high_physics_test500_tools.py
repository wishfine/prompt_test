# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_high_physics_test500 as builder  # noqa: E402
import evaluate_high_physics_test500 as evaluator  # noqa: E402


class HighPhysicsTest500ToolTests(unittest.TestCase):
    def test_question_projection_removes_all_review_labels(self) -> None:
        row = {
            "question_id": "1",
            "stem": "题目",
            "analysis": "解析",
            "reviewed_difficulty_level": "难度5档",
            "review_reason": "标签理由",
            "prior_label_stage1": "难度4档",
        }
        projected = builder.question_projection(row)
        self.assertEqual(projected, {
            "question_id": "1",
            "stem": "题目",
            "analysis": "解析",
        })

    def test_evaluator_reports_exact_within_one_and_severe(self) -> None:
        labels = {
            "1": {"reviewed_difficulty_level": "难度1档"},
            "2": {"reviewed_difficulty_level": "难度3档"},
            "3": {"reviewed_difficulty_level": "难度5档"},
        }
        predictions = {
            "1": {"final_difficulty_level": "难度1档"},
            "2": {"final_difficulty_level": "难度4档"},
            "3": {"final_difficulty_level": "难度2档"},
        }
        report = evaluator.evaluate(
            labels,
            predictions,
            "final_difficulty_level",
        )
        self.assertEqual(report["exact_match_rate"], 0.3333)
        self.assertEqual(report["within_one_level_rate"], 0.6667)
        self.assertEqual(report["severe_deviation_count"], 1)
        self.assertEqual(report["under_predicted"], 1)


if __name__ == "__main__":
    unittest.main()
