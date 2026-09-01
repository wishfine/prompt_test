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
        "calculation_model": "无定量计算",
        "calculation_complexity": "直接判断",
        "process_structure": "单阶段",
        "subquestion_dependency": "无多问",
        "shared_model_across_subquestions": False,
        "constraint_structure": "单一约束",
    }


class ScoreFeatureConsistencyTests(unittest.TestCase):
    def test_single_rule_direct_structure_below_88_is_conflict(self) -> None:
        issues = core.detect_stage1_score_feature_conflicts({
            "features": direct_features(),
            "predicted_accuracy": 80,
        })
        self.assertEqual(issues[0]["boundary"], "88边界")

    def test_independent_basic_structure_below_85_is_conflict(self) -> None:
        features = direct_features()
        features["required_task_breadth"] = "2-3个异质必要任务"
        issues = core.detect_stage1_score_feature_conflicts({
            "features": features,
            "predicted_accuracy": 82,
        })
        self.assertEqual(issues[0]["boundary"], "85边界")

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


if __name__ == "__main__":
    unittest.main()
