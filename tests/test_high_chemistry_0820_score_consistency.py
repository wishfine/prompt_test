from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import high_chemistry_pipeline_core_0820 as core


def direct_features() -> dict[str, object]:
    return {
        "step_count": "1-2步",
        "required_task_breadth": "单一规则任务",
        "model_explicitness": "模型完全显性",
        "model_relation": "单一模型",
        "reasoning_chain": "直接套用",
        "representation_conversion": "无转换",
        "evidence_relation": "直接给定",
        "calculation_model": "无定量计算",
        "calculation_complexity": "直接判断",
        "information_conversion": "无信息转换",
        "experiment_requirement": "无",
        "route_design_requirement": "无",
        "process_structure": "单阶段",
        "subquestion_dependency": "无多问",
        "shared_model_across_subquestions": False,
        "constraint_structure": "单一约束",
    }


class ScoreFeatureConsistencyTests(unittest.TestCase):
    def test_explicit_reason_band_conflict_is_repairable(self) -> None:
        rating = {
            "features": direct_features(),
            "reason": "各项均为常规综合，未达到拔高结构，正确率位于58—85区间。",
            "predicted_accuracy": 55,
        }
        issues = core.detect_stage1_score_feature_conflicts(rating)
        reason_issues = [
            issue for issue in issues
            if issue["boundary"] == "reason—分数一致性"
        ]
        self.assertEqual(len(reason_issues), 1)
        self.assertTrue(reason_issues[0]["repairable"])

    def test_reason_excluding_below_58_conflicts_with_score_55(self) -> None:
        rating = {
            "features": direct_features(),
            "reason": "各选项独立，仍属常规综合，未进入58以下区间。",
            "predicted_accuracy": 55,
        }
        issues = core.detect_stage1_score_feature_conflicts(rating)
        self.assertTrue(any(
            issue["boundary"] == "reason—分数一致性"
            for issue in issues
        ))

    def test_boundary_comparison_is_not_mistaken_for_final_band(self) -> None:
        rating = {
            "features": direct_features(),
            "reason": "本题低于85但高于58，需比较58—85与38—58两个相邻区间。",
            "predicted_accuracy": 62,
        }
        issues = core.detect_stage1_score_feature_conflicts(rating)
        self.assertFalse(any(
            issue["boundary"] == "reason—分数一致性"
            for issue in issues
        ))

    def test_negated_band_phrase_is_not_treated_as_final_band(self) -> None:
        rating = {
            "features": direct_features(),
            "reason": "本题未达到低于38分的压轴结构，最终位于38—58区间。",
            "predicted_accuracy": 52,
        }
        issues = core.detect_stage1_score_feature_conflicts(rating)
        self.assertFalse(any(
            issue["boundary"] == "reason—分数一致性"
            for issue in issues
        ))

    def test_single_value_object_wrapper_is_unwrapped(self) -> None:
        self.assertEqual(
            core._unwrap_scalar_wrapper({"value": "直接判断"}),
            ("直接判断", True),
        )
        ambiguous = {"left": "直接判断", "right": "简单计算"}
        self.assertEqual(
            core._unwrap_scalar_wrapper(ambiguous),
            (ambiguous, False),
        )

    def test_incomplete_stage2_uses_auditable_neutral_verification(self) -> None:
        recovered = core.build_neutral_stage2_verification(
            {
                "difficulty_level_step1": "难度3档",
                "original_predicted_accuracy": 62,
                "high_difficulty_features": ["多约束联合"],
            },
            validation_error="第二阶段缺少字段：analysis",
        )
        self.assertEqual(
            recovered["adjacent_boundary_review"]["boundaries_checked"],
            ["85边界", "58边界"],
        )
        self.assertEqual(recovered["reviewed_original_predicted_accuracy"], 62)
        self.assertEqual(recovered["reviewed_high_difficulty_features"], ["多约束联合"])
        self.assertFalse(recovered["has_structural_revision"])
        self.assertTrue(recovered["stage2_output_recovered"])

    def test_single_rule_direct_structure_below_88_is_audit_only(self) -> None:
        issues = core.detect_stage1_score_feature_conflicts({
            "features": direct_features(),
            "predicted_accuracy": 80,
        })
        self.assertEqual(issues[0]["boundary"], "88边界")
        self.assertIs(issues[0]["repairable"], False)

    def test_independent_basic_structure_below_85_is_conflict(self) -> None:
        features = direct_features()
        features["required_task_breadth"] = "2-3个异质必要任务"
        issues = core.detect_stage1_score_feature_conflicts({
            "features": features,
            "predicted_accuracy": 82,
        })
        self.assertEqual(issues[0]["boundary"], "85边界")
        self.assertIs(issues[0]["repairable"], True)

    def test_consistent_direct_score_has_no_conflict(self) -> None:
        issues = core.detect_stage1_score_feature_conflicts({
            "features": direct_features(),
            "predicted_accuracy": 88,
        })
        self.assertEqual(issues, [])

    def test_joint_constraints_are_not_mechanically_raised(self) -> None:
        features = direct_features()
        features["constraint_structure"] = "多约束联合筛选"
        issues = core.detect_stage1_score_feature_conflicts({
            "features": features,
            "predicted_accuracy": 82,
        })
        self.assertEqual(issues, [])

    def test_single_rule_below_85_is_not_automatically_repaired(self) -> None:
        issues = core.detect_stage1_score_feature_conflicts({
            "features": direct_features(),
            "predicted_accuracy": 82,
        })
        self.assertEqual(issues[0]["boundary"], "88边界")
        self.assertIs(issues[0]["repairable"], False)

    def test_simple_causal_structure_is_not_a_strict_85_conflict(self) -> None:
        features = direct_features()
        features["required_task_breadth"] = "2-3个异质必要任务"
        features["reasoning_chain"] = "简单因果"
        issues = core.detect_stage1_score_feature_conflicts({
            "features": features,
            "predicted_accuracy": 82,
        })
        self.assertEqual(issues, [])

    def test_light_conversion_is_not_a_strict_85_conflict(self) -> None:
        features = direct_features()
        features["required_task_breadth"] = "4个及以上异质必要任务"
        features["representation_conversion"] = "一次常规转换"
        issues = core.detect_stage1_score_feature_conflicts({
            "features": features,
            "predicted_accuracy": 82,
        })
        self.assertEqual(issues, [])

    def test_85_repair_cannot_cross_into_level_one(self) -> None:
        issues = [{"boundary": "85边界", "repairable": True}]
        self.assertFalse(
            core.score_feature_repair_crosses_boundary(issues, 87.5)
        )
        self.assertTrue(
            core.score_feature_repair_crosses_boundary(issues, 88)
        )

    def test_88_audit_does_not_activate_cross_boundary_guard(self) -> None:
        issues = [{"boundary": "88边界", "repairable": False}]
        self.assertFalse(
            core.score_feature_repair_crosses_boundary(issues, 89)
        )


if __name__ == "__main__":
    unittest.main()
