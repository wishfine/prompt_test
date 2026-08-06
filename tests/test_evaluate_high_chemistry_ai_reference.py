# -*- coding: utf-8 -*-
"""高中化学参考标签评测工具测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "evaluate_high_chemistry_ai_reference.py"
SPEC = importlib.util.spec_from_file_location("chemistry_evaluator", TOOL_PATH)
assert SPEC and SPEC.loader
evaluator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluator
SPEC.loader.exec_module(evaluator)


def prediction(final_level: str, raw_base: str, triggers: list[str]) -> dict:
    return {
        "final_difficulty_level": final_level,
        "difficulty_rating_stage1": {
            "base_difficulty_level_model_raw": raw_base,
            "structural_level_audit": {"triggers": triggers},
        },
    }


class EssentialTaskThresholdDiagnosticsTests(unittest.TestCase):
    def test_reports_rule_outcomes(self):
        labels = {
            "correct": {"reviewed_difficulty_level": "难度3档"},
            "incorrect": {"reviewed_difficulty_level": "难度2档"},
            "under": {"reviewed_difficulty_level": "难度4档"},
            "unrelated": {"reviewed_difficulty_level": "难度3档"},
        }
        predictions = {
            "correct": prediction("难度3档", "难度2档", ["essential_task_count_at_least_five"]),
            "incorrect": prediction("难度3档", "难度2档", ["essential_task_count_at_least_five"]),
            "under": prediction("难度3档", "难度2档", ["essential_task_count_at_least_five"]),
            "unrelated": prediction("难度3档", "难度2档", []),
        }
        self.assertEqual(
            evaluator.essential_task_threshold_diagnostics(labels, predictions),
            {
                "trigger_count": 3,
                "correct_promotions": 1,
                "incorrect_promotions": 1,
                "still_underestimated": 1,
                "net_benefit": 0,
            },
        )
