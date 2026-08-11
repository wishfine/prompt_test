from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "src" / "chemistry_observable_features.py"
RUNTIME_PATH = ROOT / "src" / "chemistry_difficulty_rating_with_cache.py"
BASE_PROMPT_PATH = ROOT / "prompts" / "初中化学难度打标提示词.txt"
AUDIT_PROMPT_PATH = (
    ROOT / "prompts" / "初中化学难度打标提示词_审计增强AB.txt"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_v5_features() -> dict:
    return {
        "longest_solution_chain": ["识别题目要求", "应用课内规则"],
        "task_groups": [{"task_type": "性质与反应判断", "count": 1}],
        "rule_families": ["性质用途或现象判断"],
        "curriculum_topics": ["U2-2"],
        "parallel_task_relation": "单一答题目标",
        "solution_topology": "单线性常规链",
        "reaction_structure": "单一反应",
        "condition_operations": [],
        "representation_operations": [],
        "evidence_operations": [],
        "experiment_operation": "无",
        "experiment_task_structure": "无实验判断",
        "visual_task_structure": "无必要视觉信息",
        "graph_table_operation": "无",
        "error_analysis_operation": "无误差分析",
        "calculation_operations": [],
        "new_information_operation": "无新信息",
    }


class ChemistryBranchAbsorptionV7Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features = load_module("chemistry_features_v7_test", FEATURE_PATH)
        cls.runtime = load_module("chemistry_runtime_v7_test", RUNTIME_PATH)

    def test_frozen_v5_remains_default_and_v7_is_audit_only_extension(self) -> None:
        self.assertEqual(
            self.features.OBSERVABLE_FEATURE_FIELDS,
            self.features.OBSERVABLE_V5_FEATURE_FIELDS,
        )
        self.assertEqual(len(self.features.OBSERVABLE_FEATURE_FIELDS), 17)
        self.assertEqual(len(self.features.OBSERVABLE_V7_FEATURE_FIELDS), 19)
        self.assertIn(
            "interference_type", self.features.OBSERVABLE_V7_FEATURE_FIELDS
        )
        self.assertIn(
            "expression_type", self.features.OBSERVABLE_V7_FEATURE_FIELDS
        )

    def test_v7_single_choice_audit_fields_validate(self) -> None:
        item = stable_v5_features()
        item["interference_type"] = "多个选项规则切换"
        item["expression_type"] = "原因或结论规范表达"

        validated = self.features.validate_observable_features(item)

        self.assertEqual(
            self.runtime.observable_feature_schema_version(validated),
            "chemistry_observable_v7",
        )
        self.assertEqual(
            validated["interference_type"], "多个选项规则切换"
        )

    def test_v7_fields_do_not_change_postprocess_candidates(self) -> None:
        v5 = stable_v5_features()
        v7 = {
            **v5,
            "interference_type": "规范表述易错",
            "expression_type": "原因或结论规范表达",
        }
        data = {"stem": "说明氧气支持燃烧的原因。"}

        v5_result = self.runtime.postprocess_chemistry_difficulty(
            {
                "features": v5,
                "coarse_difficulty": "送分/基础区间（1-2档）",
                "reasoning": {
                    "core_basis": "课内规则直接应用。",
                    "hard_point": "需要规范表达。",
                    "why_not_lower": "不是原词复现。",
                    "why_not_higher": "没有连续推导。",
                },
                "difficulty_level": "基础题",
            },
            data,
        )
        v7_result = self.runtime.postprocess_chemistry_difficulty(
            {
                "features": v7,
                "coarse_difficulty": "送分/基础区间（1-2档）",
                "reasoning": {
                    "core_basis": "课内规则直接应用。",
                    "hard_point": "需要规范表达。",
                    "why_not_lower": "不是原词复现。",
                    "why_not_higher": "没有连续推导。",
                },
                "difficulty_level": "基础题",
            },
            data,
        )

        self.assertEqual(
            v7_result["postprocess_candidate_actions"],
            v5_result["postprocess_candidate_actions"],
        )
        self.assertEqual(v7_result["difficulty_level"], v5_result["difficulty_level"])

    def test_program_metrics_include_options_images_and_question_items(self) -> None:
        metrics = self.runtime.derive_question_structure_metrics(
            {
                "stem": "观察图片，完成下列问题。",
                "options": "A. 甲\nB. 乙\nC. 丙\nD. 丁",
                "stem_pic_url": ["https://example.com/a.png", ""],
                "sub_questions": [
                    {"stem": "（1）判断现象"},
                    {"stem": "（2）说明原因"},
                ],
            }
        )

        self.assertEqual(metrics["option_count"], 4)
        self.assertEqual(metrics["stem_image_count"], 1)
        self.assertEqual(metrics["question_item_count"], 4)
        self.assertIn(metrics["question_text_length_band"], {
            "60字及以下", "61-100字", "101-300字", "300字以上"
        })

    def test_response_diagnostics_are_audit_only_and_detect_token_mismatch(self) -> None:
        complete = self.runtime.derive_response_diagnostics(
            raw_text='{"difficulty_level":"中等题"}',
            candidate={"difficulty_level": "中等题"},
            prompt_tokens=100,
            completion_tokens=20,
            total_tokens=121,
        )

        self.assertEqual(complete["response_status"], "complete")
        self.assertTrue(complete["structured_output_json_complete"])
        self.assertFalse(complete["token_usage_consistent"])
        self.assertEqual(complete["token_anomaly_flags"], ["token_sum_mismatch"])

    def test_audit_prompt_is_separate_and_has_non_mechanical_guidance(self) -> None:
        base = BASE_PROMPT_PATH.read_text(encoding="utf-8")
        audit = AUDIT_PROMPT_PATH.read_text(encoding="utf-8")

        self.assertNotIn('"interference_type"', base)
        self.assertNotIn('"expression_type"', base)
        self.assertIn('"interference_type"', audit)
        self.assertIn('"expression_type"', audit)
        self.assertIn("干扰类型只记录实际参与解题的负担", audit)
        self.assertIn("表达类型只记录答案必须采用的形式", audit)
        self.assertIn("不得单独触发升档或降档", audit)

        self.runtime.load_prompt_config(str(AUDIT_PROMPT_PATH))
        loaded = self.runtime.DIFFICULTY_RATING_PROMPT_PREFIX
        self.assertIn("## 一、总原则：先记录事实，再判等级", loaded)
        self.assertIn("## 审计增强 A/B", loaded)
        self.assertLess(
            loaded.index("## 审计增强 A/B"),
            loaded.index("## 输入题目信息"),
        )


if __name__ == "__main__":
    unittest.main()
