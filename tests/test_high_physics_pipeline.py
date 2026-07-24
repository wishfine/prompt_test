# -*- coding: utf-8 -*-
"""高中物理两阶段难度 Pipeline 的离线单元测试。"""

from __future__ import annotations

import copy
import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import high_physics_pipeline_core as core  # noqa: E402


def base_features(**overrides):
    features = {
        "knowledge_L1": ["力学"],
        "knowledge_L2": ["运动学"],
        "knowledge_points": ["匀变速直线运动"],
        "knowledge_count": "1个",
        "knowledge_scope": "单知识点",
        "knowledge_depth": "基础概念",
        "primary_problem_structure": "直接计算",
        "step_count": "1-2步",
        "process_count": "单过程",
        "object_count": "单对象",
        "object_relation": "无对象关系",
        "state_count": "1个",
        "state_transition": "无状态转换",
        "process_state_relation": "单一关系",
        "constraint_structure": "无约束",
        "subquestion_dependency": "无多问",
        "shared_model_across_subquestions": False,
        "model_explicitness": "模型完全显性",
        "model_relation": "单一模型",
        "reasoning_chain": "直接套用",
        "hidden_conditions": "无",
        "critical_state": "无临界",
        "classification_discussion": "无",
        "variable_relation": "无变量关系",
        "physics_methods": [],
        "formula_count": "0-1个",
        "equation_structure": "无方程",
        "calculation_complexity": "直接判断",
        "parameter_operation": "无参数",
        "numerical_complexity": "无数值计算",
        "unit_conversion": "无",
        "information_carrier": "纯文字",
        "graph_structure": "无图表",
        "drawing_requirement": "无",
        "experiment_requirement": "无",
        "context_type": "纯物理",
        "context_load": "纯包装",
        "error_risk": "无明显易错点",
    }
    features.update(overrides)
    return features


class AccuracyMappingTests(unittest.TestCase):
    def test_continuous_accuracy_boundaries(self) -> None:
        cases = [
            (100, "难度1档"),
            (88, "难度1档"),
            (87.999, "难度2档"),
            (85, "难度2档"),
            (84.999, "难度3档"),
            (58, "难度3档"),
            (57.999, "难度4档"),
            (38, "难度4档"),
            (37.999, "难度5档"),
            (0, "难度5档"),
        ]
        for accuracy, expected in cases:
            with self.subTest(accuracy=accuracy):
                self.assertEqual(core.map_accuracy_to_level(accuracy), expected)


class HighDifficultyFeatureTests(unittest.TestCase):
    def test_multiple_objects_without_strong_coupling_is_not_high(self) -> None:
        features = base_features(
            object_count="三个及以上对象",
            object_relation="对象相互独立",
        )
        detected = core.detect_high_difficulty_features(features)
        self.assertNotIn("多对象强耦合", detected.names)

    def test_standard_two_object_model_is_not_automatically_high(self) -> None:
        features = base_features(
            object_count="两个对象",
            object_relation="共同受约束",
            model_relation="单一模型",
            equation_structure="单方程",
            reasoning_chain="简单因果",
        )
        detected = core.detect_high_difficulty_features(features)
        self.assertNotIn("多对象强耦合", detected.names)

    def test_hidden_critical_does_not_duplicate_boundary_only_state_signal(self) -> None:
        features = base_features(
            process_count="两个过程",
            state_count="3个及以上",
            state_transition="连续演化",
            process_state_relation="连续变化伴随边界",
            hidden_conditions="单个隐含条件",
            critical_state="隐含临界",
            reasoning_chain="逆向推理或临界分析",
        )
        detected = core.detect_high_difficulty_features(features)
        self.assertIn("隐含临界状态", detected.names)
        self.assertNotIn("多过程或多状态强耦合", detected.names)

    def test_four_distinct_high_features_are_preserved(self) -> None:
        features = base_features(
            object_count="三个及以上对象",
            object_relation="双向耦合",
            process_count="三个及以上过程",
            state_count="3个及以上",
            state_transition="离散状态转换",
            process_state_relation="前后状态强依赖",
            constraint_structure="多约束联合筛选",
            model_relation="多模型耦合",
            equation_structure="4个以上方程或不等式组",
            reasoning_chain="多层因果",
        )
        detected = core.detect_high_difficulty_features(features)
        self.assertEqual(
            detected.names,
            [
                "多对象强耦合",
                "多过程或多状态强耦合",
                "多约束联合",
                "多模型切换或耦合",
            ],
        )

    def test_each_physics_high_difficulty_category_has_a_strict_trigger(self) -> None:
        cases = [
            (
                "复杂分类讨论",
                {
                    "classification_discussion": "3类讨论",
                    "model_relation": "模型切换",
                    "equation_structure": "2-3个方程联立",
                },
            ),
            (
                "复杂参数、范围或极值",
                {
                    "parameter_operation": "双参数",
                    "calculation_complexity": "参数或范围计算",
                    "equation_structure": "4个以上方程或不等式组",
                    "variable_relation": "分段或非线性关系",
                },
            ),
            (
                "高层级图像信息转换",
                {
                    "graph_structure": "多图联合转换",
                    "variable_relation": "函数或图像关系",
                    "reasoning_chain": "多层因果",
                },
            ),
            (
                "跨模块深度综合",
                {
                    "knowledge_scope": "跨模块综合",
                    "model_relation": "模型切换",
                    "step_count": "6-8步",
                },
            ),
            (
                "高阶实验设计或误差反演",
                {
                    "experiment_requirement": "误差反演",
                    "reasoning_chain": "逆向推理或临界分析",
                },
            ),
        ]
        for expected, overrides in cases:
            with self.subTest(expected=expected):
                detected = core.detect_high_difficulty_features(
                    base_features(**overrides)
                )
                self.assertIn(expected, detected.names)

    def test_single_parameter_complex_classification_can_trigger_parameter_high(self) -> None:
        detected = core.detect_high_difficulty_features(
            base_features(
                parameter_operation="单参数",
                calculation_complexity="参数或范围计算",
                classification_discussion="3类讨论",
                variable_relation="分段或非线性关系",
            )
        )
        self.assertIn("复杂参数、范围或极值", detected.names)
        self.assertTrue(detected.possible_overlap_groups)

    def test_active_feature_count_is_separate_from_high_feature_count(self) -> None:
        features = base_features(
            knowledge_scope="同模块跨章节",
            process_count="两个过程",
            object_count="两个对象",
            object_relation="对象相互独立",
            state_count="2个",
            state_transition="离散状态转换",
            graph_structure="单图关系转换",
        )
        active = core.detect_active_features(features)
        high = core.detect_high_difficulty_features(features)
        self.assertGreaterEqual(len(active), 5)
        self.assertEqual(high.names, [])

    def test_feature_schema_rejects_missing_required_field(self) -> None:
        features = base_features()
        features.pop("critical_state")
        with self.assertRaisesRegex(ValueError, "critical_state"):
            core.validate_feature_schema(features)

    def test_feature_schema_rejects_inconsistent_knowledge_levels(self) -> None:
        with self.assertRaisesRegex(ValueError, "knowledge_L1"):
            core.validate_feature_schema(
                base_features(
                    knowledge_L1=["热学"],
                    knowledge_L2=["运动学"],
                )
            )

    def test_feature_schema_rejects_extra_declared_l1_module(self) -> None:
        with self.assertRaisesRegex(ValueError, "knowledge_L1"):
            core.validate_feature_schema(
                base_features(
                    knowledge_L1=["力学", "电磁学"],
                    knowledge_L2=["运动学"],
                )
            )

    def test_feature_schema_rejects_duplicate_list_values(self) -> None:
        cases = [
            ("knowledge_L1", ["力学", "力学"]),
            ("knowledge_L2", ["运动学", "运动学"]),
            ("physics_methods", ["守恒思想", "守恒思想"]),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "不得包含重复值"):
                    core.validate_feature_schema(base_features(**{field: value}))

    def test_feature_schema_rejects_non_string_taxonomy_items_cleanly(self) -> None:
        for field in (
            "knowledge_L1",
            "knowledge_L2",
            "knowledge_points",
            "physics_methods",
        ):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "字符串"):
                    core.validate_feature_schema(
                        base_features(**{field: [{"name": "非法对象"}]})
                    )

    def test_high_feature_evidence_contains_every_trigger_field(self) -> None:
        detected = core.detect_high_difficulty_features(
            base_features(
                object_count="两个对象",
                object_relation="共同受约束",
                model_relation="单一模型",
                equation_structure="2-3个方程联立",
                reasoning_chain="多层因果",
                process_count="两个过程",
                state_count="2个",
                state_transition="离散状态转换",
                process_state_relation="前后状态强依赖",
            )
        )
        evidence = {item["name"]: item for item in detected.evidence}
        self.assertEqual(
            set(evidence["多对象强耦合"]["fields"]),
            {
                "object_count",
                "object_relation",
                "model_relation",
                "equation_structure",
                "reasoning_chain",
            },
        )
        self.assertIn(
            "state_transition",
            evidence["多过程或多状态强耦合"]["fields"],
        )


class MultiplierTests(unittest.TestCase):
    def test_multiplier_boundaries_for_zero_two_three_four_and_five(self) -> None:
        expected = {
            0: 1.0,
            2: 1.0,
            3: 0.85,
            4: 0.70,
            5: 0.70,
        }
        for count, multiplier in expected.items():
            with self.subTest(count=count):
                self.assertEqual(
                    core.multiplier_for_high_count(count),
                    multiplier,
                )

    def test_exactly_three_high_features_apply_point_eight_five(self) -> None:
        features = base_features(
            object_count="两个对象",
            object_relation="双向耦合",
            equation_structure="2-3个方程联立",
            reasoning_chain="多层因果",
            process_count="三个及以上过程",
            state_count="3个及以上",
            state_transition="离散状态转换",
            process_state_relation="前后状态强依赖",
            constraint_structure="多约束联合筛选",
        )
        rating = {
            "features": features,
            "reason": "测试",
            "predicted_accuracy": 80.0,
        }
        output = core.enrich_stage1_rating(rating)
        self.assertEqual(output["original_predicted_accuracy"], 80.0)
        self.assertEqual(output["high_difficulty_feature_count"], 3)
        self.assertEqual(output["multiplier_applied"], 0.85)
        self.assertEqual(output["predicted_accuracy"], 68.0)
        self.assertEqual(output["difficulty_level_step1"], "难度3档")

    def test_adjusted_accuracy_is_rounded_to_one_decimal(self) -> None:
        features = base_features(
            object_count="两个对象",
            object_relation="双向耦合",
            equation_structure="2-3个方程联立",
            reasoning_chain="多层因果",
            process_count="三个及以上过程",
            state_count="3个及以上",
            state_transition="离散状态转换",
            process_state_relation="前后状态强依赖",
            constraint_structure="多约束联合筛选",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 83.0}
        )
        self.assertEqual(output["predicted_accuracy"], 70.5)

    def test_four_or_more_high_features_prioritize_point_seven(self) -> None:
        features = base_features(
            object_count="三个及以上对象",
            object_relation="双向耦合",
            process_count="三个及以上过程",
            state_count="3个及以上",
            state_transition="离散状态转换",
            process_state_relation="前后状态强依赖",
            constraint_structure="多约束联合筛选",
            model_relation="多模型耦合",
            equation_structure="4个以上方程或不等式组",
            reasoning_chain="多层因果",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 80.0}
        )
        self.assertEqual(output["high_difficulty_feature_count"], 4)
        self.assertEqual(output["multiplier_applied"], 0.70)
        self.assertEqual(output["predicted_accuracy"], 56.0)
        self.assertEqual(output["difficulty_level_step1"], "难度4档")

    def test_fewer_than_three_high_features_do_not_adjust_accuracy(self) -> None:
        features = base_features(
            critical_state="隐含临界",
            hidden_conditions="单个隐含条件",
            reasoning_chain="逆向推理或临界分析",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 80.0}
        )
        self.assertEqual(output["high_difficulty_feature_count"], 1)
        self.assertEqual(output["multiplier_applied"], 1.0)
        self.assertEqual(output["predicted_accuracy"], 80.0)

    def test_knowledge_count_is_derived_from_points(self) -> None:
        features = base_features(
            knowledge_points=["牛顿第二定律", "动能定理", "机械能守恒"],
            knowledge_count="1个",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 80.0}
        )
        self.assertEqual(output["features"]["knowledge_count"], "2-3个")
        self.assertEqual(output["knowledge_count_model_raw"], "1个")

    def test_knowledge_scope_is_derived_from_two_level_taxonomy(self) -> None:
        features = base_features(
            knowledge_L1=["力学"],
            knowledge_L2=["运动学", "机械能"],
            knowledge_points=["匀变速直线运动", "动能定理"],
            knowledge_scope="跨模块综合",
        )
        output = core.enrich_stage1_rating(
            {"features": features, "reason": "测试", "predicted_accuracy": 80.0}
        )
        self.assertEqual(
            output["features"]["knowledge_scope"],
            "同模块跨章节",
        )
        self.assertEqual(output["knowledge_scope_model_raw"], "跨模块综合")

    def test_out_of_range_raw_accuracy_is_rejected_instead_of_silently_clamped(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 到 100"):
            core.enrich_stage1_rating(
                {
                    "features": base_features(),
                    "reason": "测试",
                    "predicted_accuracy": 120,
                }
            )

    def test_accuracy_scale_audit_records_consistent_anchor_without_changing_score(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(),
                "reason": "单一教材模板直接判断。",
                "accuracy_anchor": "教材直接原型",
                "boundary_crossing_evidence": [],
                "accuracy_self_check": {
                    "below_88_justified": False,
                    "below_85_justified": False,
                    "options_treated_as_independent_tasks": False,
                    "error_risk_only_used_for_local_adjustment": True,
                },
                "predicted_accuracy": 94.0,
            }
        )
        self.assertEqual(output["original_predicted_accuracy"], 94.0)
        self.assertEqual(output["predicted_accuracy"], 94.0)
        self.assertEqual(output["accuracy_anchor"], "教材直接原型")
        self.assertEqual(
            output["accuracy_scale_audit"],
            {
                "metadata_complete": True,
                "anchor_range_consistent": True,
                "below_88_justified": False,
                "below_85_evidence_present": True,
                "unsupported_boundary_evidence": [],
                "low_structure_score_conflict": False,
                "option_probability_multiplication_risk": False,
                "error_risk_local_adjustment_confirmed": True,
                "complex_anchor_conflict": False,
                "high_burden_score_conflict": False,
                "heterogeneous_task_breadth_conflict": False,
                "standard_model_score_inflation_risk": False,
            },
        )

    def test_accuracy_scale_audit_flags_low_structure_score_without_crossing_evidence(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(error_risk="明显易错点"),
                "reason": "模型完全显性，但因容易混淆给出较低正确率。",
                "accuracy_anchor": "熟悉标准模型",
                "boundary_crossing_evidence": [],
                "accuracy_self_check": {
                    "below_88_justified": True,
                    "below_85_justified": False,
                    "options_treated_as_independent_tasks": False,
                    "error_risk_only_used_for_local_adjustment": False,
                },
                "predicted_accuracy": 78.0,
            }
        )
        audit = output["accuracy_scale_audit"]
        self.assertFalse(audit["anchor_range_consistent"])
        self.assertFalse(audit["below_85_evidence_present"])
        self.assertTrue(audit["low_structure_score_conflict"])
        self.assertFalse(audit["error_risk_local_adjustment_confirmed"])
        self.assertEqual(output["original_predicted_accuracy"], 78.0)

    def test_accuracy_scale_audit_accepts_supported_conceptual_conflict_evidence(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(
                    knowledge_points=["电场强度", "电势", "电势能"],
                    knowledge_count="2-3个",
                    knowledge_depth="标准模型",
                    primary_problem_structure="概念辨析",
                    reasoning_chain="多层因果",
                    error_risk="明显易错点",
                ),
                "reason": "需要联合辨析场强、电势、电势能和电荷正负。",
                "accuracy_anchor": "常规综合",
                "boundary_crossing_evidence": [
                    "CONCEPTUAL_MODEL_CONFLICT"
                ],
                "accuracy_self_check": {
                    "below_88_justified": True,
                    "below_85_justified": True,
                    "options_treated_as_independent_tasks": False,
                    "error_risk_only_used_for_local_adjustment": True,
                },
                "predicted_accuracy": 78.0,
            }
        )
        audit = output["accuracy_scale_audit"]
        self.assertTrue(audit["anchor_range_consistent"])
        self.assertTrue(audit["below_85_evidence_present"])
        self.assertEqual(audit["unsupported_boundary_evidence"], [])
        self.assertFalse(audit["low_structure_score_conflict"])

    def test_accuracy_scale_audit_flags_high_burden_score_and_anchor_conflicts(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(
                    knowledge_depth="标准模型",
                    primary_problem_structure="复合题",
                    step_count="9-12步",
                    process_count="三个及以上过程",
                    state_count="3个及以上",
                    process_state_relation="前后状态强依赖",
                    subquestion_dependency="后问依赖前问",
                    shared_model_across_subquestions=True,
                    model_relation="模型切换",
                    reasoning_chain="多层因果",
                    formula_count="4-6个",
                    equation_structure="4个以上方程或不等式组",
                ),
                "reason": "课内标准模型组成的长链递进综合题。",
                "accuracy_anchor": "常规综合",
                "boundary_crossing_evidence": [
                    "THREE_PLUS_DEPENDENT_DECISIONS",
                    "DEPENDENT_SUBQUESTIONS",
                ],
                "accuracy_self_check": {
                    "below_88_justified": True,
                    "below_85_justified": True,
                    "options_treated_as_independent_tasks": False,
                    "error_risk_only_used_for_local_adjustment": True,
                },
                "predicted_accuracy": 72.0,
            }
        )
        audit = output["accuracy_scale_audit"]
        self.assertTrue(audit["complex_anchor_conflict"])
        self.assertTrue(audit["high_burden_score_conflict"])
        self.assertTrue(audit["standard_model_score_inflation_risk"])
        self.assertFalse(audit["heterogeneous_task_breadth_conflict"])

    def test_accuracy_scale_audit_flags_heterogeneous_independent_task_breadth(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(
                    knowledge_points=["器材选择", "仪器读数", "误差分析", "图像处理"],
                    knowledge_count="4个及以上",
                    primary_problem_structure="实验探究",
                    subquestion_dependency="相互独立",
                    formula_count="4-6个",
                    experiment_requirement="标准数据处理",
                ),
                "reason": "多个相互独立但异质的实验评分任务。",
                "accuracy_anchor": "低结构基础应用",
                "boundary_crossing_evidence": [],
                "accuracy_self_check": {
                    "below_88_justified": False,
                    "below_85_justified": False,
                    "options_treated_as_independent_tasks": False,
                    "error_risk_only_used_for_local_adjustment": True,
                },
                "predicted_accuracy": 89.0,
            }
        )
        audit = output["accuracy_scale_audit"]
        self.assertTrue(audit["heterogeneous_task_breadth_conflict"])
        self.assertFalse(audit["high_burden_score_conflict"])

    def test_accuracy_scale_metadata_rejects_unknown_controlled_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "accuracy_anchor"):
            core.enrich_stage1_rating(
                {
                    "features": base_features(),
                    "reason": "测试",
                    "accuracy_anchor": "自定义锚点",
                    "boundary_crossing_evidence": [],
                    "predicted_accuracy": 90.0,
                }
            )
        with self.assertRaisesRegex(ValueError, "boundary_crossing_evidence"):
            core.enrich_stage1_rating(
                {
                    "features": base_features(),
                    "reason": "测试",
                    "accuracy_anchor": "教材直接原型",
                    "boundary_crossing_evidence": ["题目容易出错"],
                    "predicted_accuracy": 90.0,
                }
            )

    def test_legacy_stage1_without_accuracy_metadata_remains_readable(self) -> None:
        output = core.enrich_stage1_rating(
            {
                "features": base_features(),
                "reason": "旧结果",
                "predicted_accuracy": 90.0,
            }
        )
        self.assertFalse(output["accuracy_scale_audit"]["metadata_complete"])
        self.assertEqual(
            set(output["accuracy_scale_audit"]["missing_metadata_fields"]),
            {
                "accuracy_anchor",
                "boundary_crossing_evidence",
                "accuracy_self_check",
            },
        )


class Stage1NormalizationTests(unittest.TestCase):
    def test_controlled_aliases_are_normalized_and_logged(self) -> None:
        rating = {
            "features": base_features(
                state_count="三个及以上",
                object_count="四个对象",
                numerical_complexity="科学记数",
                information_carrier="纯文字加示意图",
                variable_relation="函数关系",
                knowledge_L2=["万有引力"],
            ),
            "reason": "测试",
            "predicted_accuracy": 70.0,
        }
        normalized, log = core.normalize_stage1_rating(rating)
        features = normalized["features"]
        self.assertEqual(features["state_count"], "3个及以上")
        self.assertEqual(features["object_count"], "三个及以上对象")
        self.assertEqual(
            features["numerical_complexity"],
            "常规小数或科学记数",
        )
        self.assertEqual(features["information_carrier"], "多载体综合")
        self.assertEqual(features["variable_relation"], "函数或图像关系")
        self.assertEqual(
            features["knowledge_L2"],
            ["曲线运动与万有引力"],
        )
        self.assertEqual(features["knowledge_L1"], ["力学"])
        self.assertTrue(log)
        self.assertTrue(
            all(
                set(item) == {"field", "raw", "normalized", "action"}
                for item in log
            )
        )

    def test_unknown_physics_methods_are_dropped_and_duplicates_are_deduplicated(self) -> None:
        rating = {
            "features": base_features(
                physics_methods=[
                    "隔离法",
                    "控制变量法",
                    "整体法与隔离法",
                ]
            ),
            "reason": "测试",
            "predicted_accuracy": 70.0,
        }
        normalized, log = core.normalize_stage1_rating(rating)
        self.assertEqual(
            normalized["features"]["physics_methods"],
            ["整体法与隔离法"],
        )
        self.assertIn(
            {
                "field": "physics_methods",
                "raw": "控制变量法",
                "normalized": None,
                "action": "drop_unknown_method",
            },
            log,
        )

    def test_knowledge_l1_is_derived_from_normalized_l2(self) -> None:
        rating = {
            "features": base_features(
                knowledge_L1=["力学", "电磁学"],
                knowledge_L2=["热学", "原子与原子核"],
            ),
            "reason": "测试",
            "predicted_accuracy": 70.0,
        }
        normalized, log = core.normalize_stage1_rating(rating)
        self.assertEqual(
            normalized["features"]["knowledge_L1"],
            ["热学", "近代物理"],
        )
        self.assertTrue(
            any(
                item["field"] == "knowledge_L1"
                and item["action"] == "derive_from_knowledge_L2"
                for item in log
            )
        )

    def test_ambiguous_range_relation_uses_other_fields(self) -> None:
        parameterized = {
            "features": base_features(
                variable_relation="范围关系",
                parameter_operation="单参数",
            ),
            "reason": "测试",
            "predicted_accuracy": 70.0,
        }
        plain = copy.deepcopy(parameterized)
        plain["features"]["parameter_operation"] = "无参数"
        normalized_parameterized, _ = core.normalize_stage1_rating(
            parameterized
        )
        normalized_plain, _ = core.normalize_stage1_rating(plain)
        self.assertEqual(
            normalized_parameterized["features"]["variable_relation"],
            "分段或非线性关系",
        )
        self.assertEqual(
            normalized_plain["features"]["variable_relation"],
            "无变量关系",
        )

    def test_ambiguous_model_and_object_relations_use_conservative_context(self) -> None:
        rating = {
            "features": base_features(
                model_relation="同一模型多对象",
                object_relation="相互作用",
                state_count="1个",
                state_transition="无状态转换",
                equation_structure="单方程",
            ),
            "reason": "测试",
            "predicted_accuracy": 70.0,
        }
        normalized, _ = core.normalize_stage1_rating(rating)
        self.assertEqual(
            normalized["features"]["model_relation"],
            "单一模型",
        )
        self.assertEqual(
            normalized["features"]["object_relation"],
            "单向影响",
        )

    def test_enriched_result_preserves_raw_features_and_normalization_log(self) -> None:
        raw = {
            "features": base_features(state_count="三个及以上"),
            "reason": "测试",
            "predicted_accuracy": 70.0,
        }
        normalized, log = core.normalize_stage1_rating(raw)
        enriched = core.enrich_stage1_rating(
            normalized,
            features_model_raw=raw["features"],
            normalization_log=log,
        )
        self.assertEqual(
            enriched["features_model_raw"]["state_count"],
            "三个及以上",
        )
        self.assertEqual(enriched["features"]["state_count"], "3个及以上")
        self.assertTrue(enriched["enum_normalization_applied"])
        self.assertEqual(enriched["enum_normalization_log"], log)

    def test_normalization_leaves_non_string_method_for_schema_error(self) -> None:
        raw = {
            "features": base_features(
                physics_methods=[{"name": "非法对象"}]
            ),
            "reason": "测试",
            "predicted_accuracy": 70.0,
        }
        normalized, _ = core.normalize_stage1_rating(raw)
        with self.assertRaisesRegex(ValueError, "字符串"):
            core.enrich_stage1_rating(normalized)


class InputPreparationTests(unittest.TestCase):
    def test_difficulty_is_removed_and_subquestions_are_not_mutated(self) -> None:
        source = {
            "question_id": "100",
            "difficulty": "5",
            "reviewed_difficulty_level": "难度5档",
            "review_reason": "该字段绝不能发送给模型",
            "prior_label_stage1": "难度4档",
            "stem": "测试题",
            "analysis": "",
            "sub_questions": [
                {
                    "question_id": "102",
                    "stem": "第二问",
                    "analysis": "解析2",
                    "difficulty": "4",
                },
                {"question_id": "101", "stem": "第一问", "analysis": "解析1"},
            ],
            "stem_image_url": "",
            "analysis_image_url": "",
        }
        snapshot = copy.deepcopy(source)
        prepared = core.prepare_question(source, image_mode="auto")
        self.assertNotIn("difficulty", prepared.question)
        self.assertNotIn("reviewed_difficulty_level", prepared.question)
        self.assertNotIn("review_reason", prepared.question)
        self.assertNotIn("prior_label_stage1", prepared.question)
        self.assertTrue(
            all(
                "difficulty" not in item
                for item in prepared.question["sub_questions"]
            )
        )
        self.assertEqual(
            [item["question_id"] for item in prepared.question["sub_questions"]],
            ["101", "102"],
        )
        self.assertEqual(source, snapshot)
        self.assertTrue(prepared.input_quality["has_analysis"])
        self.assertTrue(prepared.input_quality["subquestion_analysis_available"])

    def test_text_insufficient_without_required_image_is_flagged(self) -> None:
        source = {
            "question_id": "100",
            "difficulty": "4",
            "stem": "",
            "options": "",
            "analysis": "",
            "sub_questions": [],
            "stem_image_url": "",
            "analysis_image_url": "",
        }
        prepared = core.prepare_question(source, image_mode="auto")
        self.assertEqual(prepared.input_quality["input_sufficiency"], "信息不足")
        self.assertTrue(prepared.input_quality["image_required"])

    def test_auto_image_mode_sends_stem_image_but_not_unneeded_analysis_image(self) -> None:
        source = {
            "question_id": "100",
            "stem": "如图所示，求物体速度。",
            "analysis": "由速度公式直接计算即可。",
            "sub_questions": [],
            "stem_image_url": "https://example.com/stem.png",
            "analysis_image_url": "https://example.com/analysis.png",
        }
        prepared = core.prepare_question(source, image_mode="auto")
        self.assertEqual(
            prepared.selected_image_urls,
            ["https://example.com/stem.png"],
        )

    def test_auto_image_mode_sends_analysis_image_when_text_analysis_is_missing(self) -> None:
        source = {
            "question_id": "100",
            "stem": "如图所示，求物体速度。",
            "analysis": "",
            "sub_questions": [],
            "stem_image_url": "https://example.com/stem.png",
            "analysis_image_url": "https://example.com/analysis.png",
        }
        prepared = core.prepare_question(source, image_mode="auto")
        self.assertEqual(
            prepared.selected_image_urls,
            [
                "https://example.com/stem.png",
                "https://example.com/analysis.png",
            ],
        )


class FinalAdjustmentTests(unittest.TestCase):
    def test_overrated_moves_to_simpler_level(self) -> None:
        result = core.finalize_level(
            current_level="难度3档",
            reasonableness="偏高",
            model_suggested_level="难度2档",
            multiplier_reasonableness="合理",
            input_sufficiency="充分",
        )
        self.assertEqual(result.final_level, "难度2档")
        self.assertFalse(result.needs_manual_review)

    def test_underrated_moves_to_harder_level(self) -> None:
        result = core.finalize_level(
            current_level="难度3档",
            reasonableness="偏低",
            model_suggested_level="难度4档",
            multiplier_reasonableness="合理",
            input_sufficiency="充分",
        )
        self.assertEqual(result.final_level, "难度4档")

    def test_cross_two_level_suggestion_is_clamped_and_flagged(self) -> None:
        result = core.finalize_level(
            current_level="难度3档",
            reasonableness="偏低",
            model_suggested_level="难度5档",
            multiplier_reasonableness="合理",
            input_sufficiency="充分",
        )
        self.assertEqual(result.final_level, "难度4档")
        self.assertTrue(result.needs_manual_review)
        self.assertEqual(result.model_suggested_level, "难度5档")

    def test_unreasonable_multiplier_requires_manual_review(self) -> None:
        result = core.finalize_level(
            current_level="难度4档",
            reasonableness="合理",
            model_suggested_level="难度4档",
            multiplier_reasonableness="不合理",
            input_sufficiency="充分",
        )
        self.assertEqual(result.final_level, "难度4档")
        self.assertTrue(result.needs_manual_review)

    def test_review_crossing_multiplier_bucket_requires_manual_review(self) -> None:
        result = core.finalize_level(
            current_level="难度3档",
            reasonableness="合理",
            model_suggested_level="难度3档",
            multiplier_reasonableness="合理",
            input_sufficiency="充分",
            original_high_count=2,
            reviewed_high_count=3,
        )
        self.assertTrue(result.needs_manual_review)

    def test_inconsistent_overrated_suggestion_is_not_applied(self) -> None:
        result = core.finalize_level(
            current_level="难度4档",
            reasonableness="偏高",
            model_suggested_level="难度4档",
            multiplier_reasonableness="合理",
            input_sufficiency="充分",
        )
        self.assertEqual(result.final_level, "难度4档")
        self.assertTrue(result.needs_manual_review)

    def test_inconsistent_underrated_suggestion_is_not_applied(self) -> None:
        result = core.finalize_level(
            current_level="难度4档",
            reasonableness="偏低",
            model_suggested_level="难度4档",
            multiplier_reasonableness="合理",
            input_sufficiency="充分",
        )
        self.assertEqual(result.final_level, "难度4档")
        self.assertTrue(result.needs_manual_review)

    def test_unreasonable_multiplier_and_overrated_review_does_not_downgrade(self) -> None:
        result = core.finalize_level(
            current_level="难度5档",
            reasonableness="偏高",
            model_suggested_level="难度4档",
            multiplier_reasonableness="不合理",
            input_sufficiency="充分",
            original_high_count=2,
            reviewed_high_count=4,
        )
        self.assertEqual(result.final_level, "难度5档")
        self.assertTrue(result.needs_manual_review)

    def test_multiplier_bucket_change_blocks_underrated_upgrade(self) -> None:
        result = core.finalize_level(
            current_level="难度4档",
            reasonableness="偏低",
            model_suggested_level="难度5档",
            multiplier_reasonableness="不合理",
            input_sufficiency="充分",
            original_high_count=2,
            reviewed_high_count=4,
        )
        self.assertEqual(result.final_level, "难度4档")
        self.assertTrue(result.needs_manual_review)
        self.assertIn("乘数桶变化", result.adjustment_desc)


class VerificationRecalculationTests(unittest.TestCase):
    def test_program_recalculates_review_multiplier_accuracy_and_level(self) -> None:
        verification = core.recalculate_verification(
            current_level="难度4档",
            original_high_count=2,
            verification={
                "reviewed_original_predicted_accuracy": 80.0,
                "reviewed_high_difficulty_features": [
                    "多对象强耦合",
                    "多过程或多状态强耦合",
                    "多约束联合",
                ],
            },
        )
        self.assertEqual(verification["reviewed_high_difficulty_feature_count"], 3)
        self.assertEqual(verification["reviewed_multiplier"], 0.85)
        self.assertEqual(verification["reviewed_predicted_accuracy"], 68.0)
        self.assertEqual(verification["reviewed_difficulty_level"], "难度3档")
        self.assertEqual(verification["multiplier_reasonableness"], "不合理")
        self.assertEqual(verification["rating_reasonableness"], "偏高")
        self.assertEqual(verification["adjusted_difficulty_level"], "难度3档")

    def test_reviewed_accuracy_must_be_between_zero_and_one_hundred(self) -> None:
        with self.assertRaisesRegex(ValueError, "0 到 100"):
            core.recalculate_verification(
                current_level="难度3档",
                original_high_count=0,
                verification={
                    "reviewed_original_predicted_accuracy": 120,
                    "reviewed_high_difficulty_features": [],
                },
            )


class PromptAssetTests(unittest.TestCase):
    def test_prompt_config_contains_both_stages_and_continuous_boundaries(self) -> None:
        path = ROOT / "prompts" / "高中物理难度打标提示词.txt"
        namespace: dict[str, str] = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {}, namespace)
        stage1 = namespace["FEATURE_EXTRACTION_PROMPT_PREFIX"]
        stage2 = namespace["VERIFICATION_PROMPT_PREFIX"]
        self.assertIn("original_predicted_accuracy", stage2)
        self.assertIn("high_difficulty_feature_count", stage2)
        self.assertIn("multiplier_applied", stage2)
        self.assertIn("predicted_accuracy >= 88", stage1)
        self.assertIn("85 <= predicted_accuracy < 88", stage1)
        self.assertIn("58 <= predicted_accuracy < 85", stage1)
        self.assertIn("38 <= predicted_accuracy < 58", stage1)
        self.assertIn("predicted_accuracy < 38", stage1)
        self.assertIn("组合乘数效应", stage1)
        self.assertIn("普通高考的全体考生", stage1)
        self.assertIn("92—96", stage1)
        self.assertIn("模板分数", stage1)
        self.assertIn("局部模型熟悉度", stage1)
        self.assertIn("整题完成负担", stage1)
        self.assertIn("多个异质评分任务", stage1)
        self.assertIn("这些范围互相重叠", stage1)
        self.assertIn("reviewed_original_predicted_accuracy", stage2)
        self.assertIn("不要自行输出 multiplier", stage2)
        self.assertIn("不是难度4档或难度5档的必要条件", stage2)
        self.assertIn("accuracy_anchor", stage1)
        self.assertIn("boundary_crossing_evidence", stage1)
        self.assertIn("accuracy_self_check", stage1)
        self.assertIn("普通单选题或多选题整体只算一个作答任务", stage1)
        self.assertIn("error_risk 不负责决定正确率所属的大区间", stage1)
        self.assertIn("CONCEPTUAL_MODEL_CONFLICT", stage1)
        self.assertIn("低于 85", stage1)
        self.assertIn("accuracy_scale_audit", stage2)
        blocks = re.findall(r'\{\n  "features":.*?\n\}', stage1, re.S)
        self.assertTrue(blocks)
        example = json.loads(blocks[-1])
        core.validate_feature_schema(example["features"])
        self.assertIn(
            example["accuracy_anchor"],
            core.ACCURACY_ANCHOR_RANGES,
        )
        self.assertTrue(
            set(example["boundary_crossing_evidence"])
            <= core.BOUNDARY_CROSSING_EVIDENCE
        )


if __name__ == "__main__":
    unittest.main()
