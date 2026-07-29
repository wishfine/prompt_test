from __future__ import annotations

import copy
import importlib.util
import json
import re
import runpy
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "chemistry_difficulty_rating_with_cache.py"
PROMPT_PATH = ROOT / "prompts" / "初中化学难度打标提示词.txt"

SPEC = importlib.util.spec_from_file_location(
    "chemistry_difficulty_rating_runtime",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
chemistry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chemistry)


def valid_rating(level: str = "中等题") -> dict:
    features = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
    features.update(
        {
            "reasoning_depth": "2-3层",
            "reasoning_direction": "正向推导",
            "knowledge_relation": "同模块深度关联",
            "representation_conversion": "一次表征转换",
            "reaction_relation": "2-3个并列或简单连续反应",
            "constraint_complexity": "单一约束",
            "evidence_relation": "多条清晰证据联合",
            "experiment_requirement": "控制变量、现象解释或数据归纳",
            "graph_table_requirement": "多组比较归纳",
            "calculation_model": "单一方程式或关系式",
            "unfamiliar_information_transfer": "课内直接原型",
            "subquestion_dependency": "多问共享模型但无答案依赖",
        }
    )
    coarse = {
        "送分题": "送分/基础区间（1-2档）",
        "基础题": "送分/基础区间（1-2档）",
        "中等题": "基础/中等区间（2-3档）",
        "拔高题": "中等/拔高区间（3-4档）",
        "压轴题": "拔高/压轴区间（4-5档）",
    }[level]
    return {
        "features": features,
        "coarse_difficulty": coarse,
        "reasoning": {
            "core_basis": "测试依据",
            "hard_point": "测试难点",
            "why_not_lower": "不降档",
            "why_not_higher": "不升档",
        },
        "difficulty_level": level,
    }


class ChemistryRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_image_mode = chemistry.CHEMISTRY_IMAGE_MODE

    def tearDown(self) -> None:
        chemistry.CHEMISTRY_IMAGE_MODE = self.old_image_mode

    def test_lite_temperature_matches_physics_runtime(self) -> None:
        self.assertEqual(
            chemistry.resolve_temperature("doubao-seed-2.0-lite", "0"),
            1.0,
        )
        self.assertEqual(
            chemistry.resolve_temperature("doubao-seed-2.0-mini", "0"),
            0.0,
        )
        self.assertIsNone(
            chemistry.resolve_temperature("doubao-seed-2.0-pro", ""),
        )

    def test_source_label_is_not_sent_and_is_renamed_in_output(self) -> None:
        row = {
            "question_id": 1234567890123456789,
            "stem": "题干",
            "difficulty": 4,
            "teacher_label": "较难",
            "sub_questions": [
                {"stem": "小题", "difficulty": 2},
            ],
        }
        safe = chemistry.sanitize_question_data(row)
        self.assertNotIn("difficulty", safe)
        self.assertNotIn("teacher_label", safe)
        self.assertNotIn("difficulty", safe["sub_questions"][0])
        output = chemistry.make_output_base(row)
        self.assertNotIn("difficulty", output)
        self.assertEqual(output["source_difficulty_untrusted"], 4)
        self.assertEqual(
            output["source_teacher_label_untrusted"],
            "较难",
        )
        self.assertEqual(
            output["sub_questions"][0]["source_difficulty_untrusted"],
            2,
        )

    def test_auto_image_mode_uses_visual_reference_only(self) -> None:
        chemistry.CHEMISTRY_IMAGE_MODE = "auto"
        ordinary = {
            "stem": "下列说法正确的是",
            "analysis": "根据教材结论可知答案。",
            "stem_pic_url": "https://example.com/full.png",
            "analysis_pic_url": "https://example.com/answer.png",
        }
        self.assertEqual(chemistry.select_image_fields(ordinary)[0], [])

        visual = {
            "stem": "根据下图中的四幅微观示意图回答问题：<image>",
            "analysis": "由粒子构成关系判断。",
            "stem_pic_url": (
                "https://example.com/1.png,"
                "https://example.com/2.png"
            ),
        }
        selected, _ = chemistry.select_image_fields(visual)
        self.assertEqual(selected, ["stem_pic_url"])
        content = chemistry.build_user_content(visual, selected)
        images = [
            item["image_url"]
            for item in content
            if item.get("type") == "input_image"
        ]
        self.assertEqual(
            images,
            [
                "https://example.com/1.png",
                "https://example.com/2.png",
            ],
        )

    def test_missing_analysis_can_request_analysis_image(self) -> None:
        chemistry.CHEMISTRY_IMAGE_MODE = "auto"
        row = {
            "stem": "完成实验探究题。",
            "analysis": "",
            "analysis_pic_url": "https://example.com/analysis.png",
        }
        self.assertEqual(
            chemistry.select_image_fields(row)[0],
            ["analysis_pic_url"],
        )

    def test_full_structured_text_includes_subquestion_analysis(self) -> None:
        row = {
            "stem": "母题",
            "options": "A. 甲",
            "analysis": "母题解析",
            "sub_questions": [
                {
                    "question_id": "2",
                    "stem": "小题题干",
                    "options": "B. 乙",
                    "analysis": "小题解析",
                }
            ],
        }
        text = chemistry.construct_question_content(row)
        for expected in (
            "母题",
            "A. 甲",
            "母题解析",
            "小题题干",
            "B. 乙",
            "小题解析",
        ):
            self.assertIn(expected, text)

    def test_missing_feature_is_rejected_instead_of_defaulted(self) -> None:
        rating = valid_rating()
        rating["features"].pop("evidence_relation")
        with self.assertRaises(chemistry.ChemistrySchemaError):
            chemistry.validate_rating_contract(rating)

    def test_extra_or_fuzzy_feature_key_is_rejected(self) -> None:
        rating = valid_rating()
        rating["features"]["not_reasoning_depth"] = "2-3层"
        with self.assertRaises(chemistry.ChemistrySchemaError):
            chemistry.validate_rating_contract(rating)

    def test_coarse_interval_must_contain_final_level(self) -> None:
        rating = valid_rating("中等题")
        rating["coarse_difficulty"] = "送分/基础区间（1-2档）"
        with self.assertRaises(chemistry.ChemistrySchemaError):
            chemistry.validate_rating_contract(rating)

    def test_direct_retrieval_basic_is_audited_without_writeback(self) -> None:
        rating = valid_rating("基础题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertTrue(
            any(
                "仅审计，不自动降档" in item
                for item in result["feature_audit_flags"]
            )
        )

    def test_easy_application_is_audited_without_writeback(self) -> None:
        rating = valid_rating("送分题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "1层",
                "reasoning_direction": "正向推导",
                "evidence_relation": "单一证据直接对应",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "送分题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertTrue(
            any(
                "仅审计，不自动升档" in item
                for item in result["feature_audit_flags"]
            )
        )

    def test_weak_easy_features_do_not_stack_into_basic_writeback(self) -> None:
        rating = valid_rating("送分题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "constraint_complexity": "单一约束",
                "evidence_relation": "单一证据直接对应",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "送分题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_complete_equation_model_basic_is_raised_to_medium(self) -> None:
        rating = valid_rating("基础题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "2-3层",
                "reasoning_direction": "正向推导",
                "reaction_relation": "单一直接反应",
                "constraint_complexity": "单一约束",
                "calculation_model": "单一方程式或关系式",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "中等题")

    def test_medium_is_not_raised_by_one_weak_high_signal(self) -> None:
        rating = valid_rating("中等题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "正向推导",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
                "experiment_requirement": "控制变量、现象解释或数据归纳",
                "graph_table_requirement": "多组比较归纳",
                "calculation_model": "单一方程式或关系式",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_medium_with_two_decisive_signals_is_raised_to_hard(self) -> None:
        rating = valid_rating("中等题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "逆向推导",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "需要排除竞争解释",
                "experiment_requirement": "方案设计、评价或补充实验",
            }
        )
        original = copy.deepcopy(rating)
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(len(result["postprocess_actions"]), 1)
        self.assertEqual(rating, original)

    def test_coupled_hard_is_raised_only_one_level_to_final(self) -> None:
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "6层及以上",
                "reasoning_direction": "分类讨论或综合推导",
                "knowledge_relation": "多模块深度融合",
                "representation_conversion": "宏观-微观-符号-定量多重转换",
                "reaction_relation": "多反应连续转化",
                "constraint_complexity": "多层嵌套约束",
                "evidence_relation": "证据冲突、筛选或多层排除",
                "experiment_requirement": "多阶段探究与定量误差",
                "graph_table_requirement": "多图表耦合建模",
                "calculation_model": "多重守恒、差量、联立或分类",
                "unfamiliar_information_transfer": "完全陌生模型现场建立",
                "subquestion_dependency": "多问存在结果或任务链依赖",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "压轴题")
        self.assertEqual(len(result["postprocess_actions"]), 1)

    def test_special_structural_final_is_not_mechanically_lowered(self) -> None:
        rating = valid_rating("压轴题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "逆向推导",
                "representation_conversion": "宏观-微观-符号-定量多重转换",
                "calculation_model": "单一方程式或关系式",
                "unfamiliar_information_transfer": "迁移后建立关系",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "压轴题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_prompt_example_is_valid_and_uses_core12_enums(self) -> None:
        namespace = runpy.run_path(str(PROMPT_PATH))
        prefix = namespace["DIFFICULTY_RATING_PROMPT_PREFIX"]
        suffix = namespace["DIFFICULTY_RATING_PROMPT_SUFFIX"]
        self.assertTrue(prefix)
        self.assertTrue(suffix)
        blocks = re.findall(
            r"```json\s*(\{.*?\})\s*```",
            prefix,
            flags=re.DOTALL,
        )
        self.assertEqual(len(blocks), 1)
        example = json.loads(blocks[0])
        self.assertEqual(
            set(example["features"]),
            set(chemistry.FEATURE_DEFAULTS),
        )
        for field, value in example["features"].items():
            self.assertIn(
                value,
                chemistry.ALLOWED_FEATURE_VALUES[field],
            )
        for placeholder in (
            "五个选项之一",
            "四个选项之一",
            "三个选项之一",
        ):
            self.assertNotIn(placeholder, prefix)
        self.assertIn("冻结Core-12特征", suffix)
        self.assertNotIn("冻结18维特征", suffix)
        self.assertGreaterEqual(prefix.count("教师等级："), 16)
        self.assertIn(
            "不能看到题目要求“判断”就自动解释为一步应用",
            prefix,
        )
        self.assertIn(
            "没有答案依赖不能单独作为压轴降为拔高的依据",
            prefix,
        )


if __name__ == "__main__":
    unittest.main()
