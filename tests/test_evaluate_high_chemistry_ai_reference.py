from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools" / "evaluate_high_chemistry_ai_reference.py"
SPEC = importlib.util.spec_from_file_location("evaluate_high_chemistry_ai_reference", PATH)
assert SPEC and SPEC.loader
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class HighChemistryAiReferenceEvaluationTests(unittest.TestCase):
    def test_physics_aligned_core_metrics(self) -> None:
        labels = {
            "q1": {"reviewed_difficulty_level": "难度1档"},
            "q2": {"reviewed_difficulty_level": "难度4档"},
            "q3": {"standard_level": 5},
        }
        predictions = {
            "q1": {"final_difficulty_level": "难度1档", "difficulty_level_step1": "难度1档"},
            "q2": {"final_difficulty_level": "难度3档", "difficulty_level_step1": "难度4档"},
            "q3": {"final_difficulty_level": "难度5档", "difficulty_level_step1": "难度5档"},
        }
        report = evaluation.evaluate(labels, predictions, "final_difficulty_level")
        self.assertEqual(report["exact_match_rate"], 0.6667)
        self.assertEqual(report["within_one_level_rate"], 1.0)
        self.assertEqual(report["mae"], 0.3333)
        self.assertEqual(report["severe_deviation_count"], 0)
        self.assertIn("quadratic_weighted_kappa", report)
        self.assertIn("confusion_matrix", report)


if __name__ == "__main__":
    unittest.main()
