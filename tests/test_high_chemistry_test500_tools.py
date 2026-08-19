# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "tools" / "build_high_chemistry_test500.py"
EVALUATOR_PATH = ROOT / "tools" / "evaluate_high_chemistry_test500.py"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HighChemistryTest500ToolTests(unittest.TestCase):
    def test_builder_separates_question_from_reference_labels(self) -> None:
        self.assertTrue(BUILDER_PATH.exists())
        builder = load_module(BUILDER_PATH, "build_high_chemistry_test500")
        row = {
            "question_id": "1",
            "parent_id": "1",
            "stem": "题目",
            "options": "A.甲",
            "analysis": "解析",
            "sub_questions": [],
            "reviewed_difficulty_level": "难度3档",
            "review_reason": "标签理由",
            "manual_difficulty_level": "难度3档",
            "confidence": "高",
        }
        blind = builder.question_projection(row)
        labels = builder.label_projection(row)
        self.assertEqual(blind["stem"], "题目")
        self.assertNotIn("reviewed_difficulty_level", blind)
        self.assertNotIn("review_reason", blind)
        self.assertNotIn("manual_difficulty_level", blind)
        self.assertEqual(labels["reviewed_difficulty_level"], "难度3档")
        self.assertNotIn("stem", labels)
        self.assertNotIn("analysis", labels)

    def test_evaluator_reports_step1_final_and_review_diagnostics(self) -> None:
        self.assertTrue(EVALUATOR_PATH.exists())
        evaluator = load_module(
            EVALUATOR_PATH,
            "evaluate_high_chemistry_test500",
        )
        labels = {
            "1": {"reviewed_difficulty_level": "难度2档"},
            "2": {"reviewed_difficulty_level": "难度4档"},
        }
        predictions = {
            "1": {
                "difficulty_level_step1": "难度3档",
                "final_difficulty_level": "难度2档",
                "difficulty_rating_stage1": {
                    "original_predicted_accuracy": 86.0,
                },
                "verification": {
                    "has_structural_revision": True,
                    "review_requires_manual": False,
                    "multiplier_reasonableness": "合理",
                    "feature_corrections_applied": [{"field": "step_count"}],
                    "review_action": "建议降一档",
                },
            },
            "2": {
                "difficulty_level_step1": "难度4档",
                "final_difficulty_level": "难度4档",
                "difficulty_rating_stage1": {
                    "original_predicted_accuracy": 52.0,
                },
                "verification": {
                    "has_structural_revision": False,
                    "review_requires_manual": False,
                    "multiplier_reasonableness": "合理",
                    "feature_corrections_applied": [],
                    "review_action": "维持",
                },
            },
        }
        report = evaluator.build_report(labels, predictions)
        self.assertEqual(report["step1"]["exact_match_rate"], 0.5)
        self.assertEqual(report["final"]["exact_match_rate"], 1.0)
        self.assertEqual(
            report["review_diagnostics"]["supported_feature_correction_count"],
            1,
        )
        self.assertEqual(
            report["accuracy_scale_diagnostics"]["unique_original_accuracy_count"],
            2,
        )


if __name__ == "__main__":
    unittest.main()
