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
