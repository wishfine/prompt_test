# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import build_high_physics_test500 as builder  # noqa: E402
import evaluate_high_physics_test500 as evaluator  # noqa: E402
import upgrade_high_physics_v3_results as upgrader  # noqa: E402


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

    def test_evaluator_reports_accuracy_scale_audit_diagnostics(self) -> None:
        predictions = {
            "1": {
                "difficulty_rating_stage1": {
                    "accuracy_anchor": "教材直接原型",
                    "original_predicted_accuracy": 94.0,
                    "accuracy_scale_audit": {
                        "metadata_complete": True,
                        "anchor_range_consistent": True,
                        "low_structure_score_conflict": False,
                        "option_probability_multiplication_risk": False,
                        "error_risk_local_adjustment_confirmed": True,
                        "unsupported_boundary_evidence": [],
                    },
                }
            },
            "2": {
                "difficulty_rating_stage1": {
                    "accuracy_anchor": "熟悉标准模型",
                    "original_predicted_accuracy": 78.0,
                    "accuracy_scale_audit": {
                        "metadata_complete": True,
                        "anchor_range_consistent": False,
                        "low_structure_score_conflict": True,
                        "option_probability_multiplication_risk": True,
                        "error_risk_local_adjustment_confirmed": False,
                        "unsupported_boundary_evidence": [
                            "MODEL_NOT_FULLY_EXPLICIT"
                        ],
                    },
                }
            },
        }
        report = evaluator.accuracy_scale_diagnostics(predictions)
        self.assertEqual(report["records_with_stage1"], 2)
        self.assertEqual(report["metadata_complete_count"], 2)
        self.assertEqual(report["anchor_range_inconsistent_count"], 1)
        self.assertEqual(report["low_structure_score_conflict_count"], 1)
        self.assertEqual(report["option_probability_multiplication_risk_count"], 1)
        self.assertEqual(report["error_risk_not_local_count"], 1)
        self.assertEqual(report["unsupported_boundary_evidence_count"], 1)
        self.assertEqual(report["unique_original_accuracy_count"], 2)
        self.assertEqual(
            report["anchor_distribution"],
            {"教材直接原型": 1, "熟悉标准模型": 1},
        )

    def test_v3_upgrade_recalculates_final_level_with_bucket_guard(self) -> None:
        row = {
            "question_id": "1",
            "pipeline_version": "high_physics_two_stage_v3",
            "difficulty_rating_stage1": {
                "difficulty_level_step1": "难度4档",
                "high_difficulty_feature_count": 2,
            },
            "difficulty_level_step1": "难度4档",
            "verification": {
                "rating_reasonableness": "偏低",
                "adjusted_difficulty_level": "难度5档",
                "multiplier_reasonableness": "不合理",
                "reviewed_high_difficulty_features": [
                    "多对象强耦合",
                    "多过程或多状态强耦合",
                    "多约束联合",
                    "复杂分类讨论",
                ],
            },
            "final_difficulty_level": "难度5档",
            "input_quality": {"input_sufficiency": "充分"},
        }
        upgraded = upgrader.upgrade_record(row)
        self.assertEqual(upgraded["final_difficulty_level"], "难度4档")
        self.assertTrue(upgraded["needs_manual_review"])
        self.assertIn("乘数桶变化", upgraded["final_adjustment"])
        self.assertEqual(
            upgraded["pipeline_version"],
            "high_physics_two_stage_v4",
        )
        self.assertEqual(
            upgraded["upgraded_from_pipeline_version"],
            "high_physics_two_stage_v3",
        )


if __name__ == "__main__":
    unittest.main()
