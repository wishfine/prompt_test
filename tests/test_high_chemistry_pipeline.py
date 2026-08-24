# -*- coding: utf-8 -*-
"""高中化学两阶段难度 Pipeline 的核心行为测试（V14 精炼：连续分数与结构档位解耦 + 严格4档门槛 + 跨两档守卫）。"""

from __future__ import annotations

import copy
import inspect
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import high_chemistry_pipeline_core as core  # noqa: E402
except ModuleNotFoundError:
    core = None


def base_features(**overrides):
    features = {
        "knowledge_L1": ["化学基本概念"],
        "knowledge_L2": ["物质分类与化学用语"],
        "knowledge_points": ["物质分类"],
        "knowledge_count": "1个",
        "knowledge_scope": "单知识点",
        "substance_count": "1种",
        "substance_relation": "单一物质",
        "reaction_count": "0-1个",
        "reaction_relation": "无反应链",
        "competing_reaction": "无",
        "process_structure": "单阶段",
        "primary_problem_structure": "概念辨析",
        "step_count": "1-2步",
        "required_task_breadth": "单一规则任务",
        "subquestion_dependency": "无多问",
        "shared_model_across_subquestions": False,
        "model_explicitness": "模型完全显性",
        "model_relation": "单一模型",
        "reasoning_chain": "直接套用",
        "representation_conversion": "无转换",
        "evidence_relation": "直接给定",
        "hidden_conditions": "无",
        "critical_condition": "无临界",
        "classification_discussion": "无",
        "constraint_structure": "无约束",
        "chemistry_methods": [],
        "calculation_model": "无定量计算",
        "equation_structure": "无方程",
        "calculation_complexity": "直接判断",
        "parameter_operation": "无参数",
        "information_carrier": "纯文字",
        "information_conversion": "无信息转换",
        "experiment_requirement": "无",
        "route_design_requirement": "无",
        "context_type": "纯化学",
        "context_load": "纯包装",
        "error_risk": "无明显易错点",
    }
    features.update(overrides)
    return features


@unittest.skipIf(core is None, "高中化学核心模块尚未实现")
class AccuracyAndSchemaTests(unittest.TestCase):
    def test_strict_stage_schemas_match_current_contracts(self) -> None:
        stage1 = core.build_stage1_output_schema()
        self.assertEqual(
            set(stage1["required"]),
            {"features", "reason", "predicted_accuracy"},
        )
        self.assertFalse(stage1["additionalProperties"])
        self.assertEqual(
            set(stage1["properties"]["features"]["required"]),
            set(core.REQUIRED_FEATURE_FIELDS),
        )
        stage2 = core.build_stage2_output_schema()
        self.assertIn("reviewed_original_predicted_accuracy", stage2["required"])
        self.assertFalse(stage2["additionalProperties"])
        
        # Stage 2 correctable fields check (taxonomy fields excluded)
        self.assertNotIn("knowledge_L1", core.STAGE2_CORRECTABLE_FEATURE_FIELDS)
        self.assertNotIn("knowledge_L2", core.STAGE2_CORRECTABLE_FEATURE_FIELDS)
        self.assertNotIn("knowledge_count", core.STAGE2_CORRECTABLE_FEATURE_FIELDS)
        self.assertNotIn("knowledge_scope", core.STAGE2_CORRECTABLE_FEATURE_FIELDS)
        self.assertNotIn("knowledge_points", core.STAGE2_CORRECTABLE_FEATURE_FIELDS)
        self.assertNotIn("chemistry_methods", core.STAGE2_CORRECTABLE_FEATURE_FIELDS)
        self.assertEqual(
            set(core.STAGE2_CORRECTABLE_FEATURE_FIELDS),
            set(core.REQUIRED_FEATURE_FIELDS)
            - core.PROGRAM_DERIVED_FEATURE_FIELDS
            - {"knowledge_points", "knowledge_L2", "chemistry_methods"},
        )

        # Stage 2 corrections anyOf variants check
        corrections_schema = stage2["properties"]["feature_corrections"]["items"]
        self.assertIn("anyOf", corrections_schema)
        variant_fields = {
            variant["properties"]["field"]["enum"][0]
            for variant in corrections_schema["anyOf"]
        }
        self.assertEqual(variant_fields, set(core.STAGE2_CORRECTABLE_FEATURE_FIELDS))

    def test_continuous_accuracy_boundaries_are_fixed(self) -> None:
        cases = [
            (88, "难度1档"),
            (87.999, "难度2档"),
            (85, "难度2档"),
            (84.999, "难度3档"),
            (58, "难度3档"),
            (57.999, "难度4档"),
            (38, "难度4档"),
            (37.999, "难度5档"),
        ]
        for score, expected in cases:
            with self.subTest(score=score):
                self.assertEqual(core.map_accuracy_to_level(score), expected)

    def test_feature_schema_accepts_complete_chemistry_features(self) -> None:
        core.validate_feature_schema(base_features())

    def test_feature_schema_rejects_missing_field(self) -> None:
        features = base_features()
        del features["reaction_relation"]
        with self.assertRaises(ValueError):
            core.validate_feature_schema(features)

    def test_feature_schema_rejects_invalid_enum_value(self) -> None:
        features = base_features(step_count="100步")
        with self.assertRaises(ValueError):
            core.validate_feature_schema(features)

    def test_normalize_stage1_rating_coerces_shared_model_string_values(self) -> None:
        normalized, log = core.normalize_stage1_rating({
            "features": base_features(shared_model_across_subquestions="是"),
            "reason": "测试",
            "predicted_accuracy": 80,
        })
        self.assertIs(normalized["features"]["shared_model_across_subquestions"], True)
        self.assertEqual(log[0]["field"], "shared_model_across_subquestions")

    def test_normalize_stage1_rating_derives_l1_from_valid_l2_with_log(self) -> None:
        normalized, log = core.normalize_stage1_rating({
            "features": base_features(
                knowledge_L1=["元素化学"],
                knowledge_L2=["物质分类与化学用语", "化学反应与能量"],
            ),
            "reason": "测试",
            "predicted_accuracy": 80,
        })
        self.assertEqual(
            set(normalized["features"]["knowledge_L1"]),
            {"化学基本概念", "化学反应原理"},
        )
        l1_log = [item for item in log if item["field"] == "knowledge_L1"]
        self.assertEqual(len(l1_log), 1)
        self.assertIn("固定父级", l1_log[0]["reason"])


@unittest.skipIf(core is None, "高中化学核心模块尚未实现")
class StructuralLevelConstraintTests(unittest.TestCase):
    def test_structural_constraint_signature_has_no_accuracy_parameters(self) -> None:
        sig = inspect.signature(core.derive_structural_level_constraint)
        params = list(sig.parameters.keys())
        self.assertNotIn("original_accuracy", params)
        self.assertNotIn("predicted_accuracy", params)
        self.assertNotIn("score", params)
        self.assertEqual(params, ["features", "high_names"])

    def test_strict_direct_prototype_sets_floor_and_ceiling_to_level_one(self) -> None:
        features = base_features()
        constraint = core.derive_structural_level_constraint(features, [])
        self.assertEqual(constraint["difficulty_floor"], "难度1档")
        self.assertEqual(constraint["difficulty_ceiling"], "难度1档")
        self.assertEqual(constraint["rule_ids"], ["direct_prototype_exact_1"])
        self.assertFalse(constraint["constraint_conflict"])

    def test_calculation_model_sets_floor_to_at_least_level_two(self) -> None:
        features = base_features(
            calculation_model="常规化学计量",
            step_count="1-2步",
            reasoning_chain="简单因果",
        )
        constraint = core.derive_structural_level_constraint(features, [])
        self.assertEqual(constraint["difficulty_floor"], "难度2档")
        self.assertEqual(constraint["difficulty_ceiling"], "难度2档")

    def test_basic_explicit_application_sets_ceiling_to_level_two(self) -> None:
        features = base_features(
            step_count="1-2步",
            model_explicitness="模型完全显性",
            reasoning_chain="简单因果",
            substance_relation="相互独立",
        )
        constraint = core.derive_structural_level_constraint(features, [])
        self.assertEqual(constraint["difficulty_ceiling"], "难度2档")

    def test_parallel_basic_bundle_sets_ceiling_to_level_two(self) -> None:
        features = base_features(
            required_task_breadth="4个及以上异质必要任务",
            substance_relation="相互独立",
            reaction_relation="无反应链",
            process_structure="单阶段",
            step_count="1-2步",
            model_explicitness="模型完全显性",
            reasoning_chain="简单因果",
            information_conversion="无信息转换",
        )
        constraint = core.derive_structural_level_constraint(features, [])
        self.assertEqual(constraint["difficulty_ceiling"], "难度2档")
        self.assertIn("parallel_basic_bundle_strict_ceiling_2", constraint["rule_ids"])

    def test_standard_chain_sets_floor_to_level_three(self) -> None:
        features = base_features(
            step_count="3-5步",
            substance_relation="前后转化依赖",
            reaction_relation="显性顺序衔接",
            reaction_count="2-3个",
            reasoning_chain="多层因果",
            calculation_model="常规化学计量",
            required_task_breadth="2-3个异质必要任务",
        )
        constraint = core.derive_structural_level_constraint(features, [])
        self.assertEqual(constraint["difficulty_floor"], "难度3档")
        self.assertIn("standard_chain_floor_3", constraint["rule_ids"])

    def test_compressed_high_burden_sets_floor_to_level_four(self) -> None:
        features = base_features(
            step_count="3-5步",
            model_explicitness="半隐含模型",
            reasoning_chain="多层因果",
            model_relation="模型切换",
            information_conversion="多源信息联合转换",
        )
        constraint = core.derive_structural_level_constraint(features, [])
        self.assertEqual(constraint["difficulty_floor"], "难度4档")
        self.assertIn("compressed_high_burden_floor_4", constraint["rule_ids"])

    def test_ordinary_process_flow_does_not_force_floor_four(self) -> None:
        """3-5步、显性流程、单模型、前后转化、简单因果、无复杂计算 -> 不应被强制4档。"""
        features = base_features(
            step_count="3-5步",
            process_structure="多阶段显性流程",
            model_relation="单一模型",
            model_explicitness="模型完全显性",
            substance_relation="前后转化依赖",
            reasoning_chain="简单因果",
            calculation_model="常规化学计量",
            information_conversion="直接读取",
        )
        high = core.detect_high_difficulty_features(features)
        constraint = core.derive_structural_level_constraint(features, high.names)
        self.assertEqual(constraint["difficulty_floor"], "难度2档")
        self.assertEqual(constraint["difficulty_ceiling"], "难度3档")
        # 验证 70 分（3档）在该结构下保持 3 档
        res = core.enrich_stage1_rating({"features": features, "reason": "测试", "predicted_accuracy": 70.0})
        self.assertEqual(res["difficulty_level_step1"], "难度3档")

    def test_strong_multistage_flow_path_a_sets_floor_to_level_four(self) -> None:
        """路径 A: 6-8步+模型切换+多阶段强依赖 -> floor=4。"""
        features = base_features(
            step_count="6-8步",
            process_structure="多阶段强依赖",
            model_relation="模型切换",
            substance_relation="前后转化依赖",
        )
        high = core.detect_high_difficulty_features(features)
        constraint = core.derive_structural_level_constraint(features, high.names)
        self.assertEqual(constraint["difficulty_floor"], "难度4档")

    def test_coupled_system_with_extra_burden_path_b_sets_floor_to_level_four(self) -> None:
        """路径 B: 6-8步+模型切换+体系依赖+多源信息联合转换 -> floor=4。"""
        features = base_features(
            step_count="6-8步",
            model_relation="模型切换",
            substance_relation="前后转化依赖",
            information_conversion="多源信息联合转换",
        )
        high = core.detect_high_difficulty_features(features)
        constraint = core.derive_structural_level_constraint(features, high.names)
        self.assertEqual(constraint["difficulty_floor"], "难度4档")

    def test_coupled_system_without_extra_burden_does_not_force_floor_four(self) -> None:
        """3-5步+模型切换+前后转化依赖+无额外强负担 -> 不应被强制4档。"""
        features = base_features(
            step_count="3-5步",
            model_relation="模型切换",
            model_explicitness="模型完全显性",
            substance_relation="前后转化依赖",
            reaction_relation="显性顺序衔接",
            reaction_count="2-3个",
            reasoning_chain="简单因果",
            information_conversion="直接读取",
            calculation_model="常规化学计量",
        )
        high = core.detect_high_difficulty_features(features)
        constraint = core.derive_structural_level_constraint(features, high.names)
        self.assertEqual(constraint["difficulty_floor"], "难度2档")

    def test_complex_quantitative_floor_four_strictly_tracks_high_names(self) -> None:
        # Case 1: 常规多步计算无高难 -> floor 3 (不强制4档，标准常规综合)
        regular_calc = base_features(
            step_count="3-5步",
            calculation_model="平衡常数或Ka/Kb/Ksp",
            calculation_complexity="多步计算",
            parameter_operation="无参数",
            model_explicitness="模型完全显性",
            reasoning_chain="简单因果",
            information_conversion="直接读取",
            substance_relation="同一反应体系",
        )
        high1 = core.detect_high_difficulty_features(regular_calc)
        self.assertNotIn("复杂定量、参数或范围", high1.names)
        c1 = core.derive_structural_level_constraint(regular_calc, high1.names)
        self.assertEqual(c1["difficulty_floor"], "难度3档")

        # Case 2: 严格命中高难复杂定量 -> floor 4
        strict_complex = base_features(
            step_count="3-5步",
            calculation_model="平衡常数或Ka/Kb/Ksp",
            calculation_complexity="多方程联立",
            equation_structure="2-3个方程联立",
            parameter_operation="双参数",
            substance_relation="同一反应体系",
        )
        high2 = core.detect_high_difficulty_features(strict_complex)
        self.assertIn("复杂定量、参数或范围", high2.names)
        c2 = core.derive_structural_level_constraint(strict_complex, high2.names)
        self.assertEqual(c2["difficulty_floor"], "难度4档")

    def test_regular_comprehensive_sets_ceiling_to_level_three(self) -> None:
        features = base_features(
            step_count="3-5步",
            substance_relation="同一反应体系",
            model_relation="单一模型",
            process_structure="单阶段",
            calculation_model="常规化学计量",
        )
        constraint = core.derive_structural_level_constraint(features, [])
        self.assertEqual(constraint["difficulty_ceiling"], "难度3档")

    def test_score_84_and_85_on_exact_structural_three_yield_final_level_three_with_intact_raw_scores(self) -> None:
        features = base_features(
            step_count="3-5步",
            substance_relation="前后转化依赖",
            reaction_relation="显性顺序衔接",
            reaction_count="2-3个",
            reasoning_chain="多层因果",
            calculation_model="多步化学计量",
            required_task_breadth="2-3个异质必要任务",
        )
        # Score 84 (score level 3)
        res84 = core.enrich_stage1_rating({
            "features": copy.deepcopy(features),
            "reason": "测试84分",
            "predicted_accuracy": 84.0,
        })
        self.assertEqual(res84["model_predicted_accuracy_raw"], 84.0)
        self.assertEqual(res84["original_predicted_accuracy"], 84.0)
        self.assertEqual(res84["difficulty_level_from_score"], "难度3档")
        self.assertEqual(res84["difficulty_level_step1"], "难度3档")

        # Score 85 (score level 2, but structural floor 3)
        res85 = core.enrich_stage1_rating({
            "features": copy.deepcopy(features),
            "reason": "测试85分",
            "predicted_accuracy": 85.0,
        })
        self.assertEqual(res85["model_predicted_accuracy_raw"], 85.0)
        self.assertEqual(res85["original_predicted_accuracy"], 85.0)
        self.assertEqual(res85["difficulty_level_from_score"], "难度2档")
        self.assertEqual(res85["difficulty_level_step1"], "难度3档")
        self.assertTrue(res85["structural_constraint_applied"])

    def test_score_57_and_58_on_regular_structural_three_yield_final_level_three_with_intact_raw_scores(self) -> None:
        features = base_features(
            step_count="3-5步",
            substance_relation="同一反应体系",
            model_explicitness="模型完全显性",
            information_conversion="直接读取",
            model_relation="单一模型",
            process_structure="单阶段",
            reasoning_chain="简单因果",
            calculation_model="常规化学计量",
            required_task_breadth="2-3个异质必要任务",
        )
        # Score 58 (score level 3)
        res58 = core.enrich_stage1_rating({
            "features": copy.deepcopy(features),
            "reason": "测试58分",
            "predicted_accuracy": 58.0,
        })
        self.assertEqual(res58["model_predicted_accuracy_raw"], 58.0)
        self.assertEqual(res58["original_predicted_accuracy"], 58.0)
        self.assertEqual(res58["difficulty_level_from_score"], "难度3档")
        self.assertEqual(res58["difficulty_level_step1"], "难度3档")

        # Score 57 (score level 4, but structural ceiling 3)
        res57 = core.enrich_stage1_rating({
            "features": copy.deepcopy(features),
            "reason": "测试57分",
            "predicted_accuracy": 57.0,
        })
        self.assertEqual(res57["model_predicted_accuracy_raw"], 57.0)
        self.assertEqual(res57["original_predicted_accuracy"], 57.0)
        self.assertEqual(res57["difficulty_level_from_score"], "难度4档")
        self.assertEqual(res57["difficulty_level_step1"], "难度3档")
        self.assertTrue(res57["structural_constraint_applied"])

    def test_two_level_structural_disagreement_guard_moves_at_most_one_level(self) -> None:
        """若 score_level 为 2 档，但结构 floor 为 4 档，Stage1 最多移动到 3 档，并标记 severe disagreement。"""
        features = base_features(
            step_count="6-8步",
            process_structure="多阶段强依赖",
            model_relation="模型切换",
            substance_relation="前后转化依赖",
        )
        output = core.enrich_stage1_rating({
            "features": features,
            "reason": "测试",
            "predicted_accuracy": 86.0,  # score level: 难度2档
        })
        self.assertEqual(output["difficulty_level_from_score"], "难度2档")
        self.assertEqual(output["structural_level_constraint"]["difficulty_floor"], "难度4档")
        self.assertEqual(output["difficulty_level_step1"], "难度3档")  # 移动一档到3档，而不是直跳4档
        self.assertTrue(output["structural_severe_disagreement"])
        self.assertTrue(output["needs_manual_review"])

    def test_stage2_without_feature_corrections_maintains_stage1_level_even_with_subjective_score_change(self) -> None:
        features = base_features(step_count="3-5步")
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=70.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 62.0,
                "has_structural_revision": False,
                "adjacent_boundary_review": {"verdict": "维持"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertEqual(reviewed["reviewed_original_predicted_accuracy_model_raw"], 62.0)
        self.assertEqual(reviewed["reviewed_original_predicted_accuracy"], 70.0)
        self.assertEqual(reviewed["reviewed_difficulty_level"], "难度3档")
        self.assertEqual(reviewed["adjusted_difficulty_level"], "难度3档")
        self.assertFalse(reviewed["auto_adjustment_eligible"])

    def test_stage2_valid_feature_correction_recalculates_constraint_and_adjusts_at_most_one_level(self) -> None:
        features = base_features()
        reviewed = core.recalculate_verification(
            current_level="难度1档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=90.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [
                    {
                        "field": "step_count",
                        "from": "1-2步",
                        "to": "3-5步",
                        "evidence": "解析显示存在三个连续有效决策",
                    }
                ],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 86.0,
                "has_structural_revision": True,
                "adjacent_boundary_review": {"verdict": "应更难一档"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertTrue(reviewed["has_structural_revision"])
        self.assertTrue(reviewed["auto_adjustment_eligible"])
        self.assertEqual(reviewed["reviewed_direction"], "应更难一档")
        self.assertEqual(reviewed["adjusted_difficulty_level"], "难度2档")


@unittest.skipIf(core is None, "高中化学核心模块尚未实现")
class MultiplierAndHighFeatureTests(unittest.TestCase):
    def test_multiplier_boundaries(self) -> None:
        expected = {0: 1.0, 2: 1.0, 3: 0.85, 4: 0.70, 8: 0.70}
        for count, multiplier in expected.items():
            self.assertEqual(core.multiplier_for_high_count(count), multiplier)

    def test_chemistry_multiplier_requires_math_style_trigger_combo(self) -> None:
        features = base_features(
            constraint_structure="多约束联合筛选",
            critical_condition="隐含终点或有效区间",
            hidden_conditions="多个隐含条件",
            reasoning_chain="逆向推理或临界分析",
            information_carrier="多载体综合",
            information_conversion="多源信息联合转换",
            evidence_relation="证据链相互支持",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 80}
        )
        self.assertEqual(output["high_difficulty_feature_count"], 3)
        self.assertTrue(output["high_difficulty_multiplier_enabled"])
        self.assertFalse(output["multiplier_triggered"])
        self.assertEqual(output["multiplier_candidate"], 0.85)
        self.assertEqual(output["multiplier_applied"], 1.0)
        self.assertEqual(output["predicted_accuracy"], 80.0)

    def test_multiplier_floor_keeps_nonfinal_high_combo_in_level_four(self) -> None:
        features = base_features(
            reaction_count="4-6个",
            reaction_relation="前后反应强依赖",
            process_structure="多阶段强依赖",
            model_relation="多模型耦合",
            step_count="6-8步",
            reasoning_chain="多层因果",
            constraint_structure="多约束联合筛选",
            information_carrier="多载体综合",
            information_conversion="多源信息联合转换",
            evidence_relation="证据链相互支持",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 45}
        )
        self.assertTrue(output["multiplier_triggered"])
        self.assertEqual(output["multiplier_applied"], 0.70)
        self.assertTrue(output["multiplier_final_level_guard_applied"])
        self.assertEqual(output["predicted_accuracy"], 38.0)
        self.assertEqual(output["difficulty_level_step1"], "难度4档")

    def test_strong_final_combo_can_cross_from_four_to_five(self) -> None:
        features = base_features(
            reaction_count="4-6个",
            reaction_relation="前后反应强依赖",
            process_structure="多阶段强依赖",
            step_count="6-8步",
            reasoning_chain="多层因果",
            constraint_structure="多约束联合筛选",
            experiment_requirement="方案设计或误差反演",
            information_carrier="多载体综合",
            information_conversion="多源信息联合转换",
            evidence_relation="证据链相互支持",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 42}
        )
        self.assertTrue(output["multiplier_triggered"])
        self.assertFalse(output["multiplier_final_level_guard_applied"])
        self.assertEqual(output["difficulty_level_step1"], "难度5档")


@unittest.skipIf(core is None, "高中化学核心模块尚未实现")
class FinalizationAndPreparationTests(unittest.TestCase):
    def test_question_preparation_removes_labels_recursively_without_mutation(self) -> None:
        source = {
            "question_id": 123,
            "difficulty": 5,
            "stem": "如图完成实验",
            "analysis": "",
            "sub_questions": [
                {"question_id": "2", "stem": "二", "analysis": "有解析", "difficulty": 4},
                {"question_id": "1", "stem": "一", "analysis": "", "difficulty": 2},
            ],
        }
        before = copy.deepcopy(source)
        prepared = core.prepare_question(source, image_mode="off")
        self.assertEqual(source, before)
        self.assertNotIn("difficulty", prepared.question)
        self.assertTrue(
            all("difficulty" not in item for item in prepared.question["sub_questions"])
        )
        self.assertEqual(
            [item["question_id"] for item in prepared.question["sub_questions"]],
            ["1", "2"],
        )
        self.assertTrue(prepared.input_quality["has_analysis"])
        self.assertEqual(prepared.input_quality["input_sufficiency"], "部分缺失")

    def test_stage2_is_audit_only_by_default(self) -> None:
        result = core.finalize_level(
            current_level="难度3档",
            review_action="建议升一档",
            model_suggested_level="难度4档",
            input_sufficiency="充分",
        )
        self.assertEqual(result.final_level, "难度3档")
        self.assertFalse(result.auto_adjustment_applied)
        self.assertIn("维持", result.adjustment_desc)

    def test_even_when_enabled_final_adjustment_is_at_most_one_level(self) -> None:
        result = core.finalize_level(
            current_level="难度3档",
            review_action="建议升一档",
            model_suggested_level="难度5档",
            input_sufficiency="充分",
            auto_adjustment_enabled=True,
        )
    def test_same_system_simple_causal_floor_3_rule(self) -> None:
        """测试 same_system_simple_causal_floor_3 (同一反应体系+简单因果+多异质任务 1-2步) 提升为 3 档 floor 并不被 ceiling2 压制。"""
        feats = base_features(
            step_count="1-2步",
            required_task_breadth="4个及以上异质必要任务",
            substance_relation="同一反应体系",
            process_structure="单阶段",
            model_explicitness="模型完全显性",
            model_relation="单一模型",
            reasoning_chain="简单因果",
        )
        constraint = core.derive_structural_level_constraint(feats, [])
        self.assertEqual(constraint["difficulty_floor"], "难度3档")
        self.assertIn("same_system_simple_causal_floor_3", constraint["rule_ids"])
        self.assertNotIn("basic_explicit_application_ceiling_2", constraint["rule_ids"])
        self.assertNotIn("parallel_basic_bundle_strict_ceiling_2", constraint["rule_ids"])

    def test_heterogeneous_multicarrier_floor_4_rule(self) -> None:
        """测试 heterogeneous_multicarrier_floor_4 (4+异质任务+多载体综合+额外处理负担) 提升为 4 档 floor 并解除 ceiling3。"""
        feats = base_features(
            step_count="1-2步",
            required_task_breadth="4个及以上异质必要任务",
            information_carrier="多载体综合",
            calculation_model="常规化学计量",
            model_explicitness="模型完全显性",
            reasoning_chain="直接套用",
            information_conversion="直接读取",
        )
        constraint = core.derive_structural_level_constraint(feats, [])
        self.assertEqual(constraint["difficulty_floor"], "难度4档")
        self.assertIn("heterogeneous_multicarrier_floor_4", constraint["rule_ids"])
        self.assertNotIn("regular_comprehensive_ceiling_3", constraint["rule_ids"])

    def test_stage2_direction_gates_candidate_4(self) -> None:
        """测试 Candidate 4 Stage 2 方向门槛：3->2 要求 2+ non-breadth 结构组修正、直接套用且无真实依赖链；4->3 仅在依赖链保留但消除4档时允许。"""
        # Case 1: 3 -> 2 只改了 required_task_breadth -> 不通过 3->2 gate
        orig_feats = base_features(
            step_count="1-2步",
            required_task_breadth="4个及以上异质必要任务",
            model_explicitness="模型完全显性",
            reasoning_chain="简单因果",
        )
        rev = core.recalculate_verification(
            current_level="难度3档",
            original_features=orig_feats,
            original_high_count=0,
            original_high_features=[],
            original_accuracy=65.0,
            allow_auto_adjustment=True,
            verification={
                "confidence": "高",
                "adjacent_boundary_review": {"verdict": "应更简单一档"},
                "input_sufficiency_review": {"status": "充分"},
                "feature_corrections": [
                    {"field": "required_task_breadth", "from": "4个及以上异质必要任务", "to": "2-3个异质必要任务", "reason": "修改任务广度"}
                ],
                "reviewed_original_predicted_accuracy": 86.0,
            },
        )
        self.assertFalse(rev["auto_adjustment_eligible"])
        self.assertEqual(rev["adjusted_difficulty_level"], "难度3档")
        self.assertFalse(rev["three_to_two_basicization_supported"])

        # Case 2: 3 -> 2 虽然有两组修正但保留了 reaction_relation="前后反应强依赖" -> 被 _has_real_dependency 拦截
        orig_feats2 = base_features(
            step_count="1-2步",
            required_task_breadth="2-3个异质必要任务",
            model_explicitness="半隐含模型",
            reasoning_chain="简单因果",
            reaction_relation="前后反应强依赖",
        )
        rev2 = core.recalculate_verification(
            current_level="难度3档",
            original_features=orig_feats2,
            original_high_count=0,
            original_high_features=[],
            original_accuracy=65.0,
            allow_auto_adjustment=True,
            verification={
                "confidence": "高",
                "adjacent_boundary_review": {"verdict": "应更简单一档"},
                "input_sufficiency_review": {"status": "充分"},
                "feature_corrections": [
                    {"field": "model_explicitness", "from": "半隐含模型", "to": "模型完全显性", "reason": "模型显性"},
                    {"field": "reasoning_chain", "from": "简单因果", "to": "直接套用", "reason": "直接套用"},
                ],
                "reviewed_original_predicted_accuracy": 86.0,
            },
        )
        self.assertFalse(rev2["auto_adjustment_eligible"])
        self.assertEqual(rev2["adjusted_difficulty_level"], "难度3档")
        self.assertFalse(rev2["three_to_two_basicization_supported"])

    def test_validate_stage1_semantic_consistency(self) -> None:
        """测试 Candidate 5 第一阶段结构语义自洽性校验 (4大逻辑冲突拦截)。"""
        # 1. 多层因果 + 1-2步 -> ValueError
        feats1 = base_features(reasoning_chain="多层因果", step_count="1-2步")
        with self.assertRaises(ValueError) as cm1:
            core.validate_stage1_semantic_consistency(feats1)
        self.assertIn("reasoning_chain=多层因果 但 step_count=1-2步", str(cm1.exception))

        # 2. 直接套用 + 3步及以上 -> ValueError
        feats2_a = base_features(reasoning_chain="直接套用", step_count="3-5步")
        with self.assertRaises(ValueError) as cm2_a:
            core.validate_stage1_semantic_consistency(feats2_a)
        self.assertIn("reasoning_chain=直接套用", str(cm2_a.exception))

        feats2_b = base_features(reasoning_chain="直接套用", step_count="6-8步")
        with self.assertRaises(ValueError) as cm2_b:
            core.validate_stage1_semantic_consistency(feats2_b)
        self.assertIn("reasoning_chain=直接套用", str(cm2_b.exception))

        # 3. 多问递进任务链 + 无后问依赖前问且无共享模型 -> ValueError
        feats3 = base_features(
            required_task_breadth="多问递进任务链",
            subquestion_dependency="无多问",
            shared_model_across_subquestions=False,
        )
        with self.assertRaises(ValueError) as cm3:
            core.validate_stage1_semantic_consistency(feats3)
        self.assertIn("required_task_breadth=多问递进任务链，但没有答案依赖或共享模型", str(cm3.exception))

        # 4. 强多阶段流程 + 1-2步 -> ValueError
        feats4 = base_features(process_structure="多阶段强依赖", step_count="1-2步")
        with self.assertRaises(ValueError) as cm4:
            core.validate_stage1_semantic_consistency(feats4)
        self.assertIn("process_structure 为强多阶段结构，但 step_count=1-2步", str(cm4.exception))

        # 合法通过测试
        valid_feats = base_features(
            reasoning_chain="简单因果",
            step_count="1-2步",
            required_task_breadth="2-3个异质必要任务",
            subquestion_dependency="无多问",
            process_structure="单阶段",
        )
        # 应正常返回不抛异常
        core.validate_stage1_semantic_consistency(valid_feats)

    def test_enrich_stage1_rating_records_validation_retry_metadata(self) -> None:
        """测试 enrich_stage1_rating 正确透传并记录语义重试统计元数据。"""
        rating = {
            "predicted_accuracy": 75.0,
            "features": base_features(),
        }
        reasons = ["结构语义冲突：reasoning_chain=多层因果 但 step_count=1-2步。"]
        enriched = core.enrich_stage1_rating(
            rating,
            validation_retry_count=1,
            validation_retry_reasons=reasons,
        )
        self.assertEqual(enriched["stage1_validation_retry_count"], 1)
        self.assertEqual(enriched["stage1_validation_retry_reasons"], reasons)

    def test_parallel_basic_bundle_strict_requires_one_to_two_steps(self) -> None:
        """测试 parallel_basic_bundle_strict 严格限定 step_count=='1-2步'，3-5步及以上题不被误压成 ceiling2。"""
        feats_1_2 = base_features(
            step_count="1-2步",
            required_task_breadth="4个及以上异质必要任务",
            substance_relation="相互独立",
            reaction_relation="无反应链",
            process_structure="单阶段",
            model_explicitness="模型完全显性",
            model_relation="单一模型",
            reasoning_chain="直接套用",
            information_conversion="无信息转换",
            evidence_relation="直接给定",
        )
        constraint_1_2 = core.derive_structural_level_constraint(feats_1_2, [])
        self.assertIn("parallel_basic_bundle_strict_ceiling_2", constraint_1_2["rule_ids"])
        self.assertEqual(constraint_1_2["difficulty_ceiling"], "难度2档")

        # 3-5 步虽然其他特征并列，但绝不命中 parallel_basic_bundle_strict
        feats_3_5 = base_features(
            step_count="3-5步",
            required_task_breadth="4个及以上异质必要任务",
            substance_relation="相互独立",
            reaction_relation="无反应链",
            process_structure="单阶段",
            model_explicitness="模型完全显性",
            model_relation="单一模型",
            reasoning_chain="直接套用",
            information_conversion="无信息转换",
            evidence_relation="直接给定",
        )
        constraint_3_5 = core.derive_structural_level_constraint(feats_3_5, [])
        self.assertNotIn("parallel_basic_bundle_strict_ceiling_2", constraint_3_5["rule_ids"])


if __name__ == "__main__":
    unittest.main()

