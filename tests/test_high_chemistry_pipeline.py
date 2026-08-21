# -*- coding: utf-8 -*-
"""高中化学两阶段难度 Pipeline 的核心行为测试。"""

from __future__ import annotations

import copy
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
        features.pop("critical_condition")
        with self.assertRaisesRegex(ValueError, "critical_condition"):
            core.validate_feature_schema(features)

    def test_feature_schema_requires_task_breadth(self) -> None:
        features = base_features()
        features.pop("required_task_breadth")
        with self.assertRaisesRegex(ValueError, "required_task_breadth"):
            core.validate_feature_schema(features)

    def test_feature_schema_rejects_inconsistent_taxonomy(self) -> None:
        with self.assertRaisesRegex(ValueError, "knowledge_L1"):
            core.validate_feature_schema(
                base_features(
                    knowledge_L1=["有机化学"],
                    knowledge_L2=["物质分类与化学用语"],
                )
            )

    def test_normalization_derives_l1_from_valid_l2(self) -> None:
        rating = {
            "features": base_features(
                knowledge_L1=["元素化学"],
                knowledge_L2=[
                    "原子结构与元素周期律",
                    "水溶液中的离子平衡",
                    "实验探究与方案设计",
                ],
            ),
            "reason": "测试",
            "predicted_accuracy": 70,
        }

        normalized, log = core.normalize_stage1_rating(rating)

        self.assertEqual(
            normalized["features"]["knowledge_L1"],
            ["化学基本概念", "化学反应原理", "化学实验"],
        )
        self.assertTrue(
            any(item.get("field") == "knowledge_L1" for item in log)
        )


@unittest.skipIf(core is None, "高中化学核心模块尚未实现")
class ChemistryHighFeatureTests(unittest.TestCase):
    def test_many_independent_substances_are_not_high(self) -> None:
        detected = core.detect_high_difficulty_features(
            base_features(
                substance_count="7种及以上",
                substance_relation="相互独立",
                reaction_count="4-6个",
                reaction_relation="并列独立",
            )
        )
        self.assertNotIn("多物质强耦合", detected.names)
        self.assertNotIn("多反应或多阶段强耦合", detected.names)

    def test_true_multi_substance_network_is_high(self) -> None:
        detected = core.detect_high_difficulty_features(
            base_features(
                substance_count="4-6种",
                substance_relation="组成—性质—反应网络",
                reaction_count="4-6个",
                reaction_relation="多路径反应网络",
                evidence_relation="证据链相互支持",
                model_relation="多模型耦合",
            )
        )
        self.assertIn("多物质强耦合", detected.names)

    def test_competition_critical_and_two_way_classification_share_one_node(self) -> None:
        detected = core.detect_high_difficulty_features(
            base_features(
                competing_reaction="多反应竞争并需筛选",
                reaction_relation="多路径反应网络",
                evidence_relation="证据冲突需排除",
                hidden_conditions="单个隐含条件",
                critical_condition="需要推导过量不足边界",
                classification_discussion="2类讨论",
                reasoning_chain="逆向推理或临界分析",
            )
        )
        overlap_node = {
            "竞争反应与副反应判断",
            "隐含临界或过量不足",
            "复杂分类讨论",
        }
        self.assertEqual(len(overlap_node.intersection(detected.names)), 1)
        self.assertTrue(detected.suppressed_overlaps)

    def test_four_distinct_high_structures_apply_point_seven(self) -> None:
        features = base_features(
            substance_count="4-6种",
            substance_relation="组成—性质—反应网络",
            reaction_count="4-6个",
            reaction_relation="前后反应强依赖",
            evidence_relation="证据链相互支持",
            process_structure="多阶段强依赖",
            model_relation="多模型耦合",
            constraint_structure="多约束联合筛选",
            equation_structure="2-3个方程联立",
            reasoning_chain="多层因果",
            information_carrier="多载体综合",
            information_conversion="多源信息联合转换",
        )
        detected = core.detect_high_difficulty_features(features)
        self.assertGreaterEqual(len(detected.names), 4)
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 80},
            multiplier_enabled=True,
        )
        self.assertEqual(output["multiplier_applied"], 0.70)
        self.assertEqual(output["predicted_accuracy"], 56.0)
        self.assertEqual(output["difficulty_level_step1"], "难度4档")

    def test_active_count_is_not_used_as_high_count(self) -> None:
        features = base_features(
            knowledge_scope="同模块跨章节",
            substance_count="2-3种",
            substance_relation="同一反应体系",
            reaction_count="2-3个",
            reaction_relation="显性顺序衔接",
            process_structure="两阶段显性流程",
            representation_conversion="一次常规转换",
            information_carrier="单一图表",
            information_conversion="直接读取",
        )
        active = core.detect_active_features(features)
        high = core.detect_high_difficulty_features(features)
        self.assertGreaterEqual(len(active), 5)
        self.assertEqual(high.names, [])

    def test_strong_dependent_stages_do_not_require_four_reaction_nodes(self) -> None:
        detected = core.detect_high_difficulty_features(
            base_features(
                reaction_count="2-3个",
                reaction_relation="前后反应强依赖",
                process_structure="多阶段强依赖",
                step_count="6-8步",
                reasoning_chain="多层因果",
            )
        )
        self.assertIn("多反应或多阶段强耦合", detected.names)

    def test_short_explicit_sequence_is_not_strong_stage_coupling(self) -> None:
        detected = core.detect_high_difficulty_features(
            base_features(
                reaction_count="2-3个",
                reaction_relation="显性顺序衔接",
                process_structure="两阶段显性流程",
                step_count="3-5步",
            )
        )
        self.assertNotIn("多反应或多阶段强耦合", detected.names)

    def test_full_dependent_organic_route_keeps_information_and_chain_as_distinct_high_features(self) -> None:
        """路线反推与下游设计共享中间体时，不能因图中箭头显性而压成普通4档。"""
        features = base_features(
            knowledge_L1=["有机化学"],
            knowledge_L2=["有机合成与推断"],
            knowledge_points=["中间体结构反推", "同分异构体筛选", "合成路线设计", "官能团转化"],
            substance_count="4-6种",
            substance_relation="前后转化依赖",
            reaction_count="4-6个",
            reaction_relation="显性顺序衔接",
            process_structure="多阶段显性流程",
            primary_problem_structure="有机合成",
            step_count="6-8步",
            subquestion_dependency="后问依赖前问",
            shared_model_across_subquestions=True,
            model_explicitness="半隐含模型",
            model_relation="同一模型多状态",
            reasoning_chain="多层因果",
            representation_conversion="多次同类转换",
            evidence_relation="证据链相互支持",
            hidden_conditions="单个隐含条件",
            constraint_structure="多约束联合筛选",
            information_carrier="工艺流程图",
            information_conversion="流程或图谱反推",
            route_design_requirement="合成路线设计",
            context_type="有机合成",
            context_load="需要信息转换",
        )
        detected = core.detect_high_difficulty_features(features)
        self.assertIn("多反应或多阶段强耦合", detected.names)
        self.assertIn("高层级信息转换", detected.names)
        self.assertIn("高阶实验、合成或分离设计", detected.names)

        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 42},
            multiplier_enabled=True,
        )
        self.assertTrue(output["multiplier_triggered"])
        self.assertEqual(output["difficulty_level_step1"], "难度5档")
        self.assertFalse(output["multiplier_final_level_guard_applied"])

    def test_quantitative_multi_model_dependency_is_a_high_feature(self) -> None:
        detected = core.detect_high_difficulty_features(
            base_features(
                model_relation="多模型耦合",
                calculation_model="多模型定量耦合",
                calculation_complexity="多步计算",
                equation_structure="单方程",
                step_count="6-8步",
                reasoning_chain="多层因果",
            )
        )
        self.assertIn("多模型或多平衡耦合", detected.names)


@unittest.skipIf(core is None, "高中化学核心模块尚未实现")
class PipelineAndInputTests(unittest.TestCase):
    def test_multiplier_boundaries(self) -> None:
        expected = {0: 1.0, 2: 1.0, 3: 0.85, 4: 0.70, 8: 0.70}
        for count, multiplier in expected.items():
            self.assertEqual(core.multiplier_for_high_count(count), multiplier)

    def test_enrichment_preserves_raw_score_and_separates_counts(self) -> None:
        features = base_features(
            reaction_count="4-6个",
            reaction_relation="前后反应强依赖",
            process_structure="多阶段强依赖",
            step_count="6-8步",
            constraint_structure="多约束联合筛选",
            reasoning_chain="多层因果",
            information_carrier="多载体综合",
            information_conversion="多源信息联合转换",
            evidence_relation="证据链相互支持",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 80},
            multiplier_enabled=True,
        )
        self.assertEqual(output["original_predicted_accuracy"], 80.0)
        self.assertEqual(output["high_difficulty_feature_count"], 3)
        self.assertEqual(output["multiplier_applied"], 0.85)
        self.assertEqual(output["predicted_accuracy"], 68.0)
        self.assertGreater(output["active_feature_count"], 3)

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
        self.assertEqual(output["difficulty_level_step1"], "难度3档")

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

    def test_independent_multi_concept_question_cannot_stay_in_level_one(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(
                    knowledge_points=["物质分类", "电解质", "胶体", "氧化物"],
                    substance_count="4-6种",
                    substance_relation="相互独立",
                ),
                "reason": "四个独立基础判断",
                "predicted_accuracy": 88,
            }
        )
        self.assertEqual(output["difficulty_level_step1"], "难度2档")
        self.assertEqual(
            output["stage1_structural_guard_actions"][0]["rule"],
            "independent_multi_concept_level_one_floor",
        )

    def test_multiple_required_tasks_cannot_stay_in_level_one(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(
                    knowledge_points=["离子共存", "离子颜色"],
                    substance_count="2-3种",
                    substance_relation="相互独立",
                    required_task_breadth="2-3个异质必要任务",
                ),
                "reason": "两个不可合并的基础判断",
                "predicted_accuracy": 88,
            }
        )
        self.assertEqual(output["difficulty_level_step1"], "难度2档")
        self.assertEqual(
            output["stage1_structural_guard_actions"][0]["rule"],
            "multiple_required_tasks_level_one_floor",
        )

    def test_standard_stoichiometry_cannot_stay_in_level_one(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(
                    knowledge_L2=["物质的量与化学计量"],
                    knowledge_points=["反应热", "盖斯定律"],
                    substance_count="2-3种",
                    substance_relation="同一反应体系",
                    calculation_model="常规化学计量",
                    calculation_complexity="简单计算",
                ),
                "reason": "一步热化学计算",
                "predicted_accuracy": 88,
            }
        )
        self.assertEqual(output["difficulty_level_step1"], "难度2档")
        self.assertEqual(
            output["stage1_structural_guard_actions"][0]["rule"],
            "standard_stoichiometry_level_one_floor",
        )

    def test_low_structure_independent_concept_question_is_recovered_from_level_three(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(
                    knowledge_points=["物质分类", "电解质"],
                    substance_count="2-3种",
                    substance_relation="相互独立",
                    reasoning_chain="简单因果",
                    constraint_structure="单一约束",
                ),
                "reason": "多个独立基础判断",
                "predicted_accuracy": 82,
            }
        )
        self.assertEqual(output["difficulty_level_step1"], "难度2档")
        self.assertEqual(
            output["stage1_structural_guard_actions"][0]["rule"],
            "low_structure_independent_concept_recovery",
        )

    def test_standard_stoichiometry_chain_cannot_stay_in_level_two(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(
                    knowledge_L2=["物质的量与化学计量"],
                    knowledge_points=["物质的量", "化学方程式", "质量关系"],
                    substance_count="2-3种",
                    substance_relation="同一反应体系",
                    reaction_count="2-3个",
                    step_count="3-5步",
                    reasoning_chain="简单因果",
                    calculation_model="常规化学计量",
                    calculation_complexity="简单计算",
                ),
                "reason": "同一反应体系的连续计量",
                "predicted_accuracy": 86,
            }
        )
        self.assertEqual(output["difficulty_level_step1"], "难度3档")
        self.assertEqual(
            output["stage1_structural_guard_actions"][0]["rule"],
            "standard_stoichiometry_chain_level_two_floor",
        )

    def test_review_score_is_locked_without_supported_feature_revision(self) -> None:
        features = base_features()
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
        self.assertEqual(
            reviewed["reviewed_original_predicted_accuracy_model_raw"], 62.0
        )
        self.assertEqual(reviewed["reviewed_original_predicted_accuracy"], 70.0)
        self.assertFalse(reviewed["has_structural_revision"])

    def test_review_score_can_change_after_supported_feature_revision(self) -> None:
        features = base_features()
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=70.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [
                    {
                        "field": "step_count",
                        "from": "1-2步",
                        "to": "3-5步",
                        "evidence": "解析包含三个前后关联决策",
                    }
                ],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 62.0,
                "has_structural_revision": True,
                "adjacent_boundary_review": {"verdict": "维持"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertEqual(reviewed["reviewed_original_predicted_accuracy"], 62.0)
        self.assertTrue(reviewed["has_structural_revision"])

    def test_review_rejects_mismatched_from_value_and_locks_score(self) -> None:
        features = base_features()
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=70.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [
                    {
                        "field": "step_count",
                        "from": "6-8步",
                        "to": "3-5步",
                        "evidence": "from 与第一阶段事实不一致",
                    }
                ],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 62.0,
                "has_structural_revision": True,
                "adjacent_boundary_review": {"verdict": "维持"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertEqual(reviewed["reviewed_original_predicted_accuracy"], 70.0)
        self.assertFalse(reviewed["has_structural_revision"])
        self.assertEqual(len(reviewed["feature_corrections_rejected"]), 1)

    def test_program_derived_feature_correction_cannot_authorize_score_change(self) -> None:
        features = base_features(
            knowledge_points=["物质分类", "胶体性质"],
            knowledge_count="2-3个",
            knowledge_scope="同章节综合",
        )
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=70.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [
                    {
                        "field": "knowledge_scope",
                        "from": "同章节综合",
                        "to": "单知识点",
                        "evidence": "模型试图修改程序派生字段",
                    }
                ],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 88.0,
                "has_structural_revision": True,
                "adjacent_boundary_review": {"verdict": "应更简单一档"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertEqual(reviewed["reviewed_original_predicted_accuracy"], 70.0)
        self.assertFalse(reviewed["has_structural_revision"])
        self.assertEqual(reviewed["review_action"], "维持")
        self.assertEqual(len(reviewed["feature_corrections_applied"]), 0)
        self.assertIn(
            "程序派生字段",
            reviewed["feature_corrections_rejected"][0]["reason"],
        )

    def test_stage2_auto_adjustment_uses_evidence_guards(self) -> None:
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
        self.assertTrue(reviewed["auto_adjustment_eligible"])
        self.assertEqual(reviewed["reviewed_direction"], "应更难一档")
        self.assertEqual(reviewed["rating_reasonableness"], "偏低")
        self.assertEqual(reviewed["adjusted_difficulty_level"], "难度2档")

    def test_stage2_low_confidence_or_direction_conflict_blocks_auto_adjustment(self) -> None:
        features = base_features()
        for confidence, verdict in (("低", "应更难一档"), ("高", "应更简单一档")):
            with self.subTest(confidence=confidence, verdict=verdict):
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
                        "adjacent_boundary_review": {"verdict": verdict},
                        "confidence": confidence,
                        "input_sufficiency_review": {"status": "充分"},
                        "high_feature_overlap_review": [],
                    },
                )
                self.assertFalse(reviewed["auto_adjustment_eligible"])
                self.assertEqual(reviewed["rating_reasonableness"], "合理")
                self.assertEqual(reviewed["adjusted_difficulty_level"], "难度1档")
                self.assertTrue(reviewed["review_requires_manual"])

    def test_stage2_58_boundary_structural_channel_promotes_three_to_four(self) -> None:
        features = base_features(
            step_count="3-5步",
            reasoning_chain="多层因果",
            information_carrier="多载体综合",
            information_conversion="多源信息联合转换",
        )
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=60.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 60.0,
                "has_structural_revision": False,
                "adjacent_boundary_review": {"verdict": "维持"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertTrue(reviewed["chemistry_58_boundary_promotion_candidate"])
        self.assertTrue(reviewed["auto_adjustment_eligible"])
        self.assertEqual(reviewed["reviewed_direction"], "应更难一档")
        self.assertEqual(reviewed["adjusted_difficulty_level"], "难度4档")

    def test_stage2_58_boundary_representation_channel_requires_decisive_structure(self) -> None:
        features = base_features(
            step_count="3-5步",
            reasoning_chain="多层因果",
            representation_conversion="一次常规转换",
        )
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=68.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 68.0,
                "has_structural_revision": False,
                "adjacent_boundary_review": {"verdict": "维持"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertFalse(reviewed["chemistry_58_boundary_promotion_candidate"])
        self.assertFalse(reviewed["auto_adjustment_eligible"])
        self.assertEqual(reviewed["adjusted_difficulty_level"], "难度3档")

    def test_stage2_58_boundary_representation_channel_accepts_reaction_chain(self) -> None:
        features = base_features(
            substance_count="2-3种",
            substance_relation="同一反应体系",
            reaction_count="2-3个",
            step_count="3-5步",
            reasoning_chain="多层因果",
            representation_conversion="一次常规转换",
        )
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=65.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 65.0,
                "has_structural_revision": False,
                "adjacent_boundary_review": {"verdict": "维持"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertTrue(reviewed["chemistry_58_boundary_promotion_candidate"])
        self.assertTrue(reviewed["auto_adjustment_eligible"])
        self.assertEqual(reviewed["adjusted_difficulty_level"], "难度4档")

    def test_stage2_structural_cluster_promotes_model_switch_with_multistage_flow(self) -> None:
        features = base_features(
            substance_count="4-6种",
            substance_relation="前后转化依赖",
            reaction_count="2-3个",
            reaction_relation="显性顺序衔接",
            process_structure="多阶段显性流程",
            step_count="3-5步",
            model_explicitness="半隐含模型",
            model_relation="模型切换",
            reasoning_chain="多层因果",
            representation_conversion="一次常规转换",
            evidence_relation="证据链相互支持",
            information_conversion="单次关系转换",
        )
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=62.0,
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
        self.assertTrue(reviewed["chemistry_structural_cluster_promotion_candidate"])
        self.assertTrue(reviewed["auto_adjustment_eligible"])
        self.assertEqual(reviewed["adjusted_difficulty_level"], "难度4档")

    def test_stage2_structural_cluster_rejects_isolated_model_switch(self) -> None:
        features = base_features(
            step_count="3-5步",
            model_explicitness="半隐含模型",
            model_relation="模型切换",
            reasoning_chain="多层因果",
            representation_conversion="一次常规转换",
        )
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=62.0,
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
        self.assertFalse(reviewed["chemistry_structural_cluster_promotion_candidate"])
        self.assertFalse(reviewed["auto_adjustment_eligible"])

    def test_supported_feature_revision_may_change_multiplier_without_blocking_auto_adjustment(self) -> None:
        features = base_features(
            reaction_count="4-6个",
            reaction_relation="前后反应强依赖",
            model_relation="模型切换",
            process_structure="单阶段",
            step_count="3-5步",
            reasoning_chain="多层因果",
            representation_conversion="一次常规转换",
        )
        reviewed = core.recalculate_verification(
            current_level="难度3档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=62.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [{
                    "field": "process_structure",
                    "from": "单阶段",
                    "to": "多阶段强依赖",
                    "evidence": "后续反应必须以上一阶段中间体为条件",
                }],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 62.0,
                "has_structural_revision": True,
                "adjacent_boundary_review": {"verdict": "应更难一档"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertTrue(reviewed["has_structural_revision"])
        self.assertTrue(reviewed["auto_adjustment_eligible"])
        self.assertEqual(reviewed["multiplier_reasonableness"], "合理")
        self.assertEqual(reviewed["adjusted_difficulty_level"], "难度4档")

    def test_stage2_representation_channel_rejects_score_above_68(self) -> None:
        features = base_features(
            step_count="3-5步",
            reasoning_chain="多层因果",
            representation_conversion="一次常规转换",
        )
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
                "reviewed_original_predicted_accuracy": 70.0,
                "has_structural_revision": False,
                "adjacent_boundary_review": {"verdict": "维持"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertFalse(reviewed["chemistry_58_boundary_promotion_candidate"])
        self.assertFalse(reviewed["auto_adjustment_eligible"])

    def test_stage2_never_automatically_downgrades_two_to_one(self) -> None:
        features = base_features(step_count="3-5步")
        reviewed = core.recalculate_verification(
            current_level="难度2档",
            original_high_count=0,
            original_high_features=[],
            original_accuracy=86.0,
            original_features=features,
            allow_auto_adjustment=True,
            verification={
                "feature_corrections": [
                    {
                        "field": "step_count",
                        "from": "3-5步",
                        "to": "1-2步",
                        "evidence": "复核认为只有一步",
                    }
                ],
                "reviewed_high_difficulty_features": [],
                "reviewed_original_predicted_accuracy": 90.0,
                "has_structural_revision": True,
                "adjacent_boundary_review": {"verdict": "应更简单一档"},
                "confidence": "高",
                "input_sufficiency_review": {"status": "充分"},
                "high_feature_overlap_review": [],
            },
        )
        self.assertFalse(reviewed["auto_adjustment_eligible"])
        self.assertEqual(reviewed["adjusted_difficulty_level"], "难度2档")
        self.assertTrue(reviewed["review_requires_manual"])

    def test_experiment_task_does_not_force_cross_content_module(self) -> None:
        features = base_features(
            knowledge_L1=["化学反应原理", "化学实验"],
            knowledge_L2=["水溶液中的离子平衡", "实验探究与方案设计"],
            knowledge_points=["酸碱平衡", "实验方案评价"],
            knowledge_count="2-3个",
            knowledge_scope="跨模块综合",
            primary_problem_structure="实验探究",
            experiment_requirement="方案设计或误差反演",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 70}
        )
        self.assertEqual(output["features"]["knowledge_scope"], "同章节综合")

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
        self.assertEqual(result.final_level, "难度4档")
        self.assertTrue(result.needs_manual_review)

    def test_structural_revision_requires_concrete_feature_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "结构修订"):
            core.validate_structural_revision_evidence(
                {
                    "has_structural_revision": True,
                    "feature_corrections": [],
                    "missed_features": ["无"],
                }
            )

    def test_structural_revision_accepts_feature_correction_or_real_omission(self) -> None:
        core.validate_structural_revision_evidence(
            {
                "has_structural_revision": True,
                "feature_corrections": [
                    {
                        "field": "step_count",
                        "from": "3-5步",
                        "to": "1-2步",
                        "evidence": "各选项彼此独立，不构成连续链",
                    }
                ],
                "missed_features": ["无"],
            }
        )
        core.validate_structural_revision_evidence(
            {
                "has_structural_revision": True,
                "feature_corrections": [],
                "missed_features": ["遗漏了共享物质流依赖"],
            }
        )


class InitialRedStateTests(unittest.TestCase):
    def test_high_chemistry_core_module_exists(self) -> None:
        self.assertIsNotNone(core, "尚未实现 high_chemistry_pipeline_core")


if __name__ == "__main__":
    unittest.main()
