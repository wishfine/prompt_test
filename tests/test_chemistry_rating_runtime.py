from __future__ import annotations

import copy
import importlib.util
import json
import re
import runpy
import tempfile
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
            "core_basis": (
                "纵向D=3个有效化学决策；"
                "有效覆盖W=2项；"
                "任务广度B=共享模型但无答案依赖。"
                "关键任务边：操作→现象→结论。"
            ),
            "hard_point": "测试难点",
            "why_not_lower": "不降档",
            "why_not_higher": "不升档",
        },
        "difficulty_level": level,
    }


class ChemistryRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_image_mode = chemistry.CHEMISTRY_IMAGE_MODE
        self.old_writeback = getattr(
            chemistry,
            "CHEMISTRY_ENABLE_LEVEL_WRITEBACK",
            False,
        )
        self.old_final_boundary_guard = getattr(
            chemistry,
            "CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD",
            False,
        )
        self.old_final_boundary_guard_writeback = getattr(
            chemistry,
            "CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK",
            False,
        )
        self.old_teacher_distribution_guard = getattr(
            chemistry,
            "CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS",
            False,
        )
        self.old_teacher_distribution_guard_writeback = getattr(
            chemistry,
            "CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK",
            False,
        )
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = True
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = False
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = (
            False
        )

    def tearDown(self) -> None:
        chemistry.CHEMISTRY_IMAGE_MODE = self.old_image_mode
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = (
            self.old_writeback
        )
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = (
            self.old_final_boundary_guard
        )
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK = (
            self.old_final_boundary_guard_writeback
        )
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = (
            self.old_teacher_distribution_guard
        )
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = (
            self.old_teacher_distribution_guard_writeback
        )

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

    def test_run_signature_changes_with_final_guard_writeback(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.jsonl"
            prompt_path = root / "prompt.txt"
            input_path.write_text('{"question_id":"q1"}\n', encoding="utf-8")
            prompt_path.write_text("prompt", encoding="utf-8")
            chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK = (
                False
            )
            config_off = chemistry.build_run_config(
                input_path,
                prompt_path,
                seed=42,
                num=None,
            )
            chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK = (
                True
            )
            config_on = chemistry.build_run_config(
                input_path,
                prompt_path,
                seed=42,
                num=None,
            )

        self.assertNotEqual(
            chemistry.build_run_signature(config_off),
            chemistry.build_run_signature(config_on),
        )

    def test_run_signature_changes_with_teacher_distribution_guard(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.jsonl"
            prompt_path = root / "prompt.txt"
            input_path.write_text('{"question_id":"q1"}\n', encoding="utf-8")
            prompt_path.write_text("prompt", encoding="utf-8")
            chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = False
            config_off = chemistry.build_run_config(
                input_path,
                prompt_path,
                seed=42,
                num=None,
            )
            chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
            config_on = chemistry.build_run_config(
                input_path,
                prompt_path,
                seed=42,
                num=None,
            )

        self.assertNotEqual(
            chemistry.build_run_signature(config_off),
            chemistry.build_run_signature(config_on),
        )

    def test_resume_rejects_unsigned_or_mismatched_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "results.jsonl"
            output_path.write_text(
                '{"question_id":"q1"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "缺少run_signature"):
                chemistry.ensure_output_run_signature(
                    output_path,
                    "expected",
                )

            output_path.write_text(
                '{"question_id":"q1","run_signature":"other"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "运行签名不一致"):
                chemistry.ensure_output_run_signature(
                    output_path,
                    "expected",
                )

            output_path.write_text(
                '{"question_id":"q1","run_signature":"expected"}\n',
                encoding="utf-8",
            )
            chemistry.ensure_output_run_signature(
                output_path,
                "expected",
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

    def test_complete_equation_model_basic_is_raised_to_medium(
        self,
    ) -> None:
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
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "core12_basic_to_medium_complete_model",
        )

    def test_disabled_writeback_keeps_raw_level_and_audits_candidate(
        self,
    ) -> None:
        old_value = getattr(
            chemistry,
            "CHEMISTRY_ENABLE_LEVEL_WRITEBACK",
            None,
        )
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        try:
            rating = valid_rating("基础题")
            rating["features"] = copy.deepcopy(
                chemistry.FEATURE_DEFAULTS
            )
            rating["features"].update(
                {
                    "reasoning_depth": "2-3层",
                    "reasoning_direction": "正向推导",
                    "reaction_relation": "单一直接反应",
                    "constraint_complexity": "单一约束",
                    "calculation_model": "单一方程式或关系式",
                }
            )
            result = chemistry.postprocess_chemistry_difficulty(
                rating,
                {},
            )
        finally:
            if old_value is None:
                del chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK
            else:
                chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = old_value

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertFalse(result["postprocess_writeback_enabled"])
        self.assertEqual(
            result["postprocess_candidate_actions"][0]["rule"],
            "core12_basic_to_medium_complete_model",
        )

    def test_disabled_writeback_preserves_model_coarse_interval(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        rating = valid_rating("基础题")
        rating["coarse_difficulty"] = "基础/中等区间（2-3档）"

        result = chemistry.postprocess_chemistry_difficulty(
            rating,
            {},
        )

        self.assertEqual(
            result["coarse_difficulty"],
            "基础/中等区间（2-3档）",
        )
        self.assertEqual(
            result["coarse_difficulty_raw"],
            "基础/中等区间（2-3档）",
        )
        self.assertEqual(
            result["coarse_difficulty_final"],
            "基础/中等区间（2-3档）",
        )

    def test_enabled_writeback_syncs_final_coarse_interval(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = True
        rating = valid_rating("基础题")
        rating["coarse_difficulty"] = "送分/基础区间（1-2档）"

        result = chemistry.postprocess_chemistry_difficulty(
            rating,
            {},
        )

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(
            result["coarse_difficulty_raw"],
            "送分/基础区间（1-2档）",
        )
        self.assertEqual(
            result["coarse_difficulty_final"],
            "基础/中等区间（2-3档）",
        )

    def test_high_structure_basic_uses_complete_model_guard(
        self,
    ) -> None:
        rating = valid_rating("基础题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "2-3层",
                "reasoning_direction": "正向推导",
                "reaction_relation": "2-3个并列或简单连续反应",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
                "subquestion_dependency": "多问共享模型但无答案依赖",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(len(result["postprocess_actions"]), 1)
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "core12_basic_to_medium_complete_model",
        )

    def test_complete_model_does_not_require_four_structure_families(
        self,
    ) -> None:
        rating = valid_rating("基础题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "2-3层",
                "reasoning_direction": "正向推导",
                "reaction_relation": "2-3个并列或简单连续反应",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "core12_basic_to_medium_complete_model",
        )

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

    def test_low_structure_medium_is_lowered_to_basic(self) -> None:
        rating = valid_rating("中等题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "1层",
                "reasoning_direction": "正向推导",
                "knowledge_relation": "单一知识点",
                "evidence_relation": "单一证据直接对应",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(
            rating,
            {"stem": "根据一个显性现象判断物质性质。"},
        )
        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "core12_medium_to_basic_low_structure",
        )

    def test_independent_shallow_experiment_tasks_do_not_veto_lowering(
        self,
    ) -> None:
        rating = valid_rating("中等题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "1层",
                "reasoning_direction": "正向推导",
                "experiment_requirement": "基础操作或读数",
                "subquestion_dependency": "多问相互独立",
            }
        )
        data = {
            "stem": "完成气体制取、检验和装置选择。",
            "sub_questions": [
                {"stem": "选择发生装置"},
                {"stem": "检验生成的气体"},
            ],
        }
        result = chemistry.postprocess_chemistry_difficulty(rating, data)
        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "core12_medium_to_basic_low_structure",
        )

    def test_shallow_purification_text_does_not_veto_lowering(self) -> None:
        rating = valid_rating("中等题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "1层",
                "reasoning_direction": "正向推导",
                "reaction_relation": "单一直接反应",
                "subquestion_dependency": "多问相互独立",
            }
        )
        data = {
            "stem": "选择除杂试剂并说明检验方法和操作步骤。",
            "sub_questions": [
                {"stem": "选择除杂试剂"},
                {"stem": "检验杂质是否除尽"},
            ],
        }
        result = chemistry.postprocess_chemistry_difficulty(rating, data)
        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "core12_medium_to_basic_low_structure",
        )

    def test_single_choice_text_does_not_override_frozen_features(
        self,
    ) -> None:
        rating = valid_rating("中等题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "1层",
                "reasoning_direction": "正向推导",
                "reaction_relation": "单一直接反应",
            }
        )
        data = {
            "stem": (
                "除去下列物质中的少量杂质，所用试剂和操作方法"
                "都正确的是"
            ),
            "options": (
                "A. 加盐酸后蒸发 B. 加活性炭后过滤 "
                "C. 通入溶液 D. 加试剂后过滤"
            ),
        }
        result = chemistry.postprocess_chemistry_difficulty(rating, data)
        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "core12_medium_to_basic_low_structure",
        )

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

    def test_d4_5_shared_coupled_hard_is_not_forced_to_final(self) -> None:
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "分类讨论或综合推导",
                "reaction_relation": "多反应连续转化",
                "constraint_complexity": "多层嵌套约束",
                "evidence_relation": "证据冲突、筛选或多层排除",
                "experiment_requirement": "多阶段探究与定量误差",
                "graph_table_requirement": "拐点、平台或分段反推",
                "calculation_model": "多重守恒、差量、联立或分类",
                "subquestion_dependency": "多问共享模型但无答案依赖",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_experimental_d4_5_final_boundary_guard_raises_coupled_hard(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = True
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "分类讨论或综合推导",
                "knowledge_relation": "多模块深度融合",
                "representation_conversion": "宏观-微观-符号-定量多重转换",
                "reaction_relation": "多反应连续转化",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
                "experiment_requirement": "多阶段探究与定量误差",
                "graph_table_requirement": "多图表耦合建模",
                "calculation_model": "多重守恒、差量、联立或分类",
                "unfamiliar_information_transfer": "迁移后建立关系",
                "subquestion_dependency": "多问存在结果或任务链依赖",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(result["difficulty_level"], "压轴题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "core12_hard_to_final_4_5_coupled_guard",
        )

    def test_experimental_final_guard_is_auditable_without_writeback(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = True
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "分类讨论或综合推导",
                "knowledge_relation": "多模块深度融合",
                "representation_conversion": "宏观-微观-符号-定量多重转换",
                "reaction_relation": "多反应连续转化",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
                "experiment_requirement": "多阶段探究与定量误差",
                "graph_table_requirement": "多图表耦合建模",
                "calculation_model": "多重守恒、差量、联立或分类",
                "unfamiliar_information_transfer": "迁移后建立关系",
                "subquestion_dependency": "多问存在结果或任务链依赖",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertEqual(
            result["postprocess_candidate_level"],
            "压轴题",
        )
        self.assertEqual(
            result["postprocess_candidate_actions"][0]["rule"],
            "core12_hard_to_final_4_5_coupled_guard",
        )
        self.assertEqual(
            result["final_boundary_guard_candidate_level"],
            "压轴题",
        )
        self.assertEqual(
            result["final_boundary_guard_candidate_action"]["rule"],
            "core12_hard_to_final_4_5_coupled_guard",
        )

    def test_dedicated_final_guard_writeback_applies_without_general_writeback(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = True
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK = True
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "分类讨论或综合推导",
                "knowledge_relation": "多模块深度融合",
                "representation_conversion": "宏观-微观-符号-定量多重转换",
                "reaction_relation": "多反应连续转化",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
                "experiment_requirement": "多阶段探究与定量误差",
                "graph_table_requirement": "多图表耦合建模",
                "calculation_model": "多重守恒、差量、联立或分类",
                "unfamiliar_information_transfer": "迁移后建立关系",
                "subquestion_dependency": "多问存在结果或任务链依赖",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(result["difficulty_level"], "压轴题")
        self.assertTrue(result["postprocess_writeback_enabled"])
        self.assertTrue(
            result["final_boundary_guard_writeback_enabled"]
        )
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "core12_hard_to_final_4_5_coupled_guard",
        )

    def test_dedicated_final_guard_writeback_ignores_other_candidates(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = True
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK = True
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
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

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertEqual(result["postprocess_candidate_level"], "中等题")
        self.assertEqual(
            result["final_boundary_guard_candidate_level"],
            "基础题",
        )
        self.assertFalse(result["automatic_level_change_applied"])

    def test_teacher_guard_audits_easy_with_basic_experiment_as_basic(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("送分题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"]["experiment_requirement"] = "基础操作或读数"

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(result["difficulty_level"], "送分题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "基础题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_easy_to_basic_experiment_application",
        )

    def test_teacher_guard_audits_linked_basic_as_medium(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("基础题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "1层",
                "knowledge_relation": "同模块简单关联",
                "representation_conversion": "一次表征转换",
                "constraint_complexity": "多个相互关联约束",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "中等题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_basic_to_medium_linked_application",
        )

    def test_teacher_guard_does_not_promote_plain_segment_graph_medium(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("中等题")
        rating["features"]["graph_table_requirement"] = (
            "拐点、平台或分段反推"
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "中等题",
        )
        self.assertIsNone(
            result["teacher_distribution_guard_candidate_action"],
        )

    def test_teacher_guard_promotes_strong_segment_graph_chain_as_hard(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("中等题")
        rating["features"].update(
            {
                "knowledge_relation": "跨模块融合",
                "representation_conversion": "两类表征连续转换",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
                "experiment_requirement": "控制变量、现象解释或数据归纳",
                "graph_table_requirement": "拐点、平台或分段反推",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "拔高题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_medium_to_hard_strong_graph_chain",
        )

    def test_teacher_guard_requires_shared_model_for_new_information(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("中等题")
        rating["features"]["unfamiliar_information_transfer"] = (
            "给定新信息直接应用"
        )
        rating["features"]["subquestion_dependency"] = "无多问"

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "中等题",
        )
        self.assertIsNone(
            result["teacher_distribution_guard_candidate_action"],
        )

    def test_teacher_guard_promotes_shared_new_information_model(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("中等题")
        rating["features"].update(
            {
                "knowledge_relation": "跨模块融合",
                "unfamiliar_information_transfer": "给定新信息直接应用",
                "subquestion_dependency": "多问共享模型但无答案依赖",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "拔高题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_medium_to_hard_shared_new_information",
        )

    def test_teacher_guard_applies_basic_floor_to_parallel_air_exposure(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("送分题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        data = {
            "stem": "将下列物质长期暴露在空气中，质量减小的是",
            "options": (
                "A. 氢氧化钠\nB. 浓盐酸\n"
                "C. 氢氧化钙\nD. 氯化钠"
            ),
        }

        result = chemistry.postprocess_chemistry_difficulty(rating, data)

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "基础题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_easy_to_basic_parallel_application_floor",
        )

    def test_teacher_guard_applies_medium_floor_to_reaction_conversion(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("送分题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        data = {
            "stem": "在给定条件下，下列物质的转化能实现的是",
            "options": (
                "A. Cu→CuO→CuSO4\nB. Fe→FeCl3→Fe(OH)3\n"
                "C. CO2→CO→CaCO3\nD. NaCl→NaOH→Na2CO3"
            ),
        }

        result = chemistry.postprocess_chemistry_difficulty(rating, data)

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "中等题",
        )
        action = result["teacher_distribution_guard_candidate_action"]
        self.assertEqual(
            action["rule"],
            "teacher_easy_to_medium_reaction_conversion_floor",
        )
        self.assertEqual(action["from"], "送分题")
        self.assertEqual(action["to"], "中等题")

    def test_teacher_guard_can_write_back_two_level_severe_floor(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = (
            True
        )
        rating = valid_rating("送分题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        data = {
            "stem": "下列物质转化在给定条件下能够实现的是",
            "options": (
                "A. Cu→CuO→CuSO4\nB. Fe→FeCl2→Fe(OH)2\n"
                "C. CO2→CO→CaCO3\nD. NaCl→NaOH→Na2CO3"
            ),
        }

        result = chemistry.postprocess_chemistry_difficulty(rating, data)

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertTrue(
            result["teacher_distribution_guard_writeback_applied"]
        )
        self.assertEqual(len(result["postprocess_actions"]), 1)
        self.assertEqual(result["postprocess_actions"][0]["level_distance"], 2)
        self.assertEqual(
            result["coarse_difficulty"],
            "基础/中等区间（2-3档）",
        )

    def test_general_level_setter_still_rejects_two_level_change(
        self,
    ) -> None:
        rating = valid_rating("送分题")
        with self.assertRaisesRegex(
            ValueError,
            "后处理调整距离超出该规则许可范围",
        ):
            chemistry.set_level_with_reason(
                rating,
                "中等题",
                "普通规则不得跨两档",
            )

    def test_teacher_guard_promotes_equation_and_type_validation(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("基础题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        data = {
            "stem": "下列化学方程式及所属基本反应类型都正确的是",
            "options": (
                "A. 2H2+O2=2H2O，化合反应\n"
                "B. CaCO3=CaO+CO2，分解反应\n"
                "C. Fe+HCl=FeCl2+H2，置换反应\n"
                "D. NaOH+HCl=NaCl+H2O，复分解反应"
            ),
        }

        result = chemistry.postprocess_chemistry_difficulty(rating, data)

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "中等题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_basic_to_medium_reaction_validation_floor",
        )

    def test_teacher_guard_promotes_dense_project_experiment_as_hard(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("中等题")
        rating["features"].update(
            {
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
                "experiment_requirement": "控制变量、现象解释或数据归纳",
            }
        )
        data = {
            "stem": (
                "项目式学习：【活动一】实验1和实验2完成试剂制取；"
                "【活动二】实验3研究变量，进一步探究并设计实验4，"
                "根据多组数据解释现象并得出结论。"
            ),
            "options": "",
        }

        result = chemistry.postprocess_chemistry_difficulty(rating, data)

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "拔高题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_medium_to_hard_dense_project_floor",
        )

    def test_teacher_guard_audits_complex_model_hard_as_final(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "分类讨论或综合推导",
                "knowledge_relation": "跨模块融合",
                "representation_conversion": "宏观-微观-符号-定量多重转换",
                "reaction_relation": "2-3个并列或简单连续反应",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
                "experiment_requirement": "无",
                "graph_table_requirement": "无",
                "calculation_model": "多重守恒、差量、联立或分类",
                "unfamiliar_information_transfer": "课内直接原型",
                "subquestion_dependency": "无多问",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "压轴题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_hard_to_final_complex_model",
        )

    def test_final_guard_respects_low_quantitative_experiment_ceiling(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = True
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "分类讨论或综合推导",
                "knowledge_relation": "跨模块融合",
                "representation_conversion": "两类表征连续转换",
                "reaction_relation": "需要分情况判断的反应模型",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "需要排除竞争解释",
                "experiment_requirement": "多阶段探究与定量误差",
                "graph_table_requirement": "多组比较归纳",
                "calculation_model": "口算或直接比例",
                "subquestion_dependency": "多问存在结果或任务链依赖",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(
            rating,
            {"stem": "根据多组实验判断未知溶液并评价方案", "options": ""},
        )

        self.assertEqual(
            result["final_boundary_guard_candidate_level"],
            "拔高题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "拔高题",
        )
        self.assertIn(
            "缺少压轴级定量、四重表征或多图表耦合",
            result["final_promotion_ceiling_reason"],
        )

    def test_final_guard_respects_short_independent_calculation_ceiling(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = True
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "正向推导",
                "knowledge_relation": "跨模块融合",
                "representation_conversion": "宏观-微观-符号-定量多重转换",
                "reaction_relation": "多反应连续转化",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "多条清晰证据联合",
                "experiment_requirement": "无",
                "graph_table_requirement": "无",
                "calculation_model": "多重守恒、差量、联立或分类",
                "subquestion_dependency": "多问相互独立",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(
            rating,
            {
                "stem": "已知两种燃料的混合物质量和产物质量，求耗氧量。",
                "options": "",
            },
        )

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "拔高题",
        )
        self.assertIn(
            "短题中的独立常规定量任务",
            result["final_promotion_ceiling_reason"],
        )

    def test_final_guard_respects_parallel_research_questions_ceiling(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = True
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "分类讨论或综合推导",
                "knowledge_relation": "跨模块融合",
                "representation_conversion": "宏观-微观-符号-定量多重转换",
                "reaction_relation": "多反应连续转化",
                "constraint_complexity": "多个相互关联约束",
                "evidence_relation": "需要排除竞争解释",
                "experiment_requirement": "方案设计、评价或补充实验",
                "graph_table_requirement": "拐点、平台或分段反推",
                "calculation_model": "多重守恒、差量、联立或分类",
                "unfamiliar_information_transfer": "迁移后建立关系",
                "subquestion_dependency": "多问共享模型但无答案依赖",
            }
        )
        data = {
            "stem": (
                "问题1：认识物质；问题2：比较性质；问题3：设计实验；"
                "问题4：完成计算；问题5：评价方案。"
            ),
            "options": "",
        }

        result = chemistry.postprocess_chemistry_difficulty(rating, data)

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "拔高题",
        )
        self.assertIn(
            "并列研究问题",
            result["final_promotion_ceiling_reason"],
        )

    def test_teacher_guard_writeback_is_independent_and_adjacent(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = (
            True
        )
        rating = valid_rating("中等题")
        rating["features"].update(
            {
                "knowledge_relation": "跨模块融合",
                "unfamiliar_information_transfer": "给定新信息直接应用",
                "subquestion_dependency": "多问共享模型但无答案依赖",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertTrue(
            result["teacher_distribution_guard_writeback_applied"]
        )
        self.assertEqual(len(result["postprocess_actions"]), 1)
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "teacher_medium_to_hard_shared_new_information",
        )

    def test_teacher_guard_does_not_promote_weak_level_only_signal(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        rating = valid_rating("中等题")
        rating["features"] = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
        rating["features"].update(
            {
                "reasoning_depth": "2-3层",
                "reasoning_direction": "正向推导",
            }
        )

        result = chemistry.postprocess_chemistry_difficulty(rating, {})

        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "中等题",
        )
        self.assertIsNone(
            result["teacher_distribution_guard_candidate_action"]
        )

    def test_d4_5_independent_high_signals_do_not_force_final(self) -> None:
        rating = valid_rating("拔高题")
        rating["features"].update(
            {
                "reasoning_depth": "4-5层",
                "reasoning_direction": "分类讨论或综合推导",
                "reaction_relation": "多反应连续转化",
                "constraint_complexity": "多层嵌套约束",
                "experiment_requirement": "方案设计、评价或补充实验",
                "calculation_model": "单一守恒或多反应计算",
                "subquestion_dependency": "多问相互独立",
            }
        )
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])

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

    def test_reasoning_wording_is_not_used_for_level_writeback(
        self,
    ) -> None:
        rating = valid_rating("中等题")
        rating["reasoning"]["core_basis"] = "仅描述纵向推理，没有广度诊断。"
        result = chemistry.postprocess_chemistry_difficulty(rating, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_stable_profile_records_final_anchor_v2_audit_mode(
        self,
    ) -> None:
        result = chemistry.postprocess_chemistry_difficulty(
            valid_rating("中等题"),
            {},
        )
        self.assertEqual(
            result["postprocess_profile"],
            "chemistry_core12_teacher_distribution_v2_severe_floor_audit_first",
        )

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
        self.assertGreaterEqual(prefix.count("教师等级："), 17)
        self.assertIn(
            "不能看到题目要求“判断”就自动解释为一步应用",
            prefix,
        )
        self.assertIn("用纵向深度 D 确定基准档", prefix)
        self.assertIn(
            "独立任务不增加纵向深度D，但会增加整题任务广度B",
            prefix,
        )
        self.assertIn(
            "中等题进入拔高比较的结构广度通道",
            prefix,
        )
        self.assertIn(
            "至少4个非重复有效任务",
            prefix,
        )
        self.assertIn(
            "至少2项属于实质应用、解释、实验分析、图表比较或计算",
            prefix,
        )
        self.assertIn(
            "严重低估安全底线",
            prefix,
        )
        self.assertIn(
            "多选项连续转化链",
            prefix,
        )
        self.assertIn(
            "短题中两个彼此独立的常规定量任务",
            prefix,
        )
        self.assertIn(
            "题干并列提出四个以上研究问题",
            prefix,
        )
        self.assertNotIn(
            "多个独立选项分别一步判断、同一规则重复填空、"
            "只共享生活背景或图片、装置中直接填写名称和现象"
            "但没有连续依赖，均不能据此升中等。",
            prefix,
        )
        self.assertIn(
            "先由 D 确定基准档，再由 B/W 做至多一个相邻档校准",
            prefix,
        )
        self.assertIn(
            "独立小问不能机械累加到 `reasoning_depth`",
            prefix,
        )
        self.assertIn(
            "`core_basis`必须说明D、B、W和至少一条真实任务边",
            prefix,
        )
        self.assertNotIn("B/W 只通过三类受控结构校准", prefix)
        self.assertNotIn("受控广度通道必须同时满足", prefix)
        self.assertNotIn("高密度共享模型通道必须同时满足", prefix)
        self.assertNotIn("4—5层复杂主模型通道必须同时满足", prefix)
        self.assertNotIn(
            "多个选项分别涉及不同教材结论，至少进入基础题",
            prefix,
        )
        self.assertNotIn("异质事实核验只建立基础题下限", prefix)
        self.assertNotIn("并列核验类型=", prefix)
        self.assertNotIn("D/B/W 相邻校准矩阵", prefix)
        self.assertNotIn("入口E=", prefix)
        self.assertNotIn("广度校准=开启/关闭", prefix)
        self.assertNotIn("有限横向广度", prefix)
        self.assertIn(
            "`reasoning_depth=6层及以上`不是压轴题的必要条件",
            prefix,
        )
        self.assertIn(
            "4—5层拔高/压轴对照",
            prefix,
        )
        example_12 = prefix.split(
            "### 示例12：压轴题——陌生装置、温控和纯度测定",
            1,
        )[1].split("### 补充示例：", 1)[0]
        self.assertIn("`reasoning_depth=4-5层`", example_12)
        self.assertNotIn(
            "`unfamiliar_information_transfer=完全陌生模型现场建立`",
            example_12,
        )
        supplementary_final = prefix.split(
            "### 补充示例：压轴题——部分变质中的守恒与差量连续计算",
            1,
        )[1].split("### 示例13：", 1)[0]
        self.assertIn(
            "`reasoning_depth=4-5层`",
            supplementary_final,
        )
        self.assertNotIn(
            "`constraint_complexity=多层嵌套约束`",
            supplementary_final,
        )


if __name__ == "__main__":
    unittest.main()
