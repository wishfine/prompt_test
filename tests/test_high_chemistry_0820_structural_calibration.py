from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import high_chemistry_pipeline_core_0820 as core


class Level4EvidenceGroupTests(unittest.TestCase):
    def test_single_isolated_field_does_not_trigger_information_group(self) -> None:
        groups = core.detect_level4_evidence_groups(
            {"information_conversion": "流程或图谱反推"}
        )
        self.assertFalse(groups["信息反推"])

    def test_information_inference_uses_conversion_and_reasoning(self) -> None:
        groups = core.detect_level4_evidence_groups(
            {
                "information_conversion": "流程或图谱反推",
                "reasoning_chain": "逆向推理或临界分析",
            }
        )
        self.assertTrue(groups["信息反推"])

    def test_complex_quantitative_group_accepts_range_model(self) -> None:
        groups = core.detect_level4_evidence_groups(
            {
                "calculation_complexity": "参数或范围计算",
                "parameter_operation": "双参数",
            }
        )
        self.assertTrue(groups["复杂定量"])

    def test_advanced_experiment_needs_supporting_structure(self) -> None:
        isolated = core.detect_level4_evidence_groups(
            {"experiment_requirement": "控制变量或异常分析"}
        )
        supported = core.detect_level4_evidence_groups(
            {
                "experiment_requirement": "控制变量或异常分析",
                "constraint_structure": "多约束联合筛选",
            }
        )
        self.assertFalse(isolated["高阶实验或路线"])
        self.assertTrue(supported["高阶实验或路线"])


class StructuralAccuracyCalibrationTests(unittest.TestCase):
    def test_level4_evidence_crosses_58_from_any_level3_score(self) -> None:
        features = {
            "calculation_complexity": "多方程联立",
            "equation_structure": "2-3个方程联立",
        }
        for model_accuracy in (62.0, 72.0, 84.5):
            calibrated, actions, groups = (
                core._apply_structural_accuracy_calibration(
                    model_accuracy=model_accuracy,
                    features=features,
                )
            )
            self.assertEqual(calibrated, 57.9)
            self.assertTrue(groups["复杂定量"])
            self.assertEqual(actions[-1]["rule"], "level4_structure_accuracy_cap")

    def test_level3_score_without_complete_group_is_unchanged(self) -> None:
        calibrated, actions, groups = core._apply_structural_accuracy_calibration(
            model_accuracy=72.0,
            features={"calculation_complexity": "多方程联立"},
        )
        self.assertEqual(calibrated, 72.0)
        self.assertEqual(actions, [])
        self.assertFalse(any(groups.values()))

    def test_first_band_still_requires_single_rule_task(self) -> None:
        calibrated, actions, _ = core._apply_structural_accuracy_calibration(
            model_accuracy=90.0,
            features={"required_task_breadth": "2-3个异质必要任务"},
        )
        self.assertEqual(calibrated, 87.9)
        self.assertEqual(actions[0]["rule"], "multiple_required_tasks_accuracy_cap")


class PromptAlignmentTests(unittest.TestCase):
    def test_prompt_uses_same_five_level4_groups_without_fixed_reason_phrase(self) -> None:
        prompt = (ROOT / "prompts" / "高中化学难度打标提示词_0820.txt").read_text(
            encoding="utf-8"
        )
        for name in (
            "信息反推",
            "联合条件",
            "复杂定量",
            "反应或模型复杂化",
            "高阶实验或路线",
        ):
            self.assertIn(name, prompt)
        self.assertNotIn('必须写出“共同判据：____”', prompt)


if __name__ == "__main__":
    unittest.main()
