# -*- coding: utf-8 -*-
"""高中化学两阶段难度 Pipeline 的离线单元测试。"""

from __future__ import annotations

import copy
import importlib.util
import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "src" / "high_chemistry_pipeline_core.py"
SPEC = importlib.util.spec_from_file_location("high_chemistry_pipeline_core", CORE_PATH)
assert SPEC and SPEC.loader
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def base_features(**overrides):
    features = {
        "knowledge_L1": ["化学基本概念与定量关系"],
        "knowledge_L2": ["物质组成、分类与分散系"],
        "knowledge_points": ["物质分类"],
        "knowledge_count": "1个",
        "knowledge_scope": "单知识点",
        "knowledge_depth": "基础概念",
        "step_count": "1-2步",
        "substance_count": "1种",
        "reaction_count": "0个",
        "reaction_relation": "无反应关系",
        "state_count": "1个",
        "process_state_relation": "单一关系",
        "constraint_structure": "无约束",
        "subquestion_dependency": "无多问",
        "shared_model_across_subquestions": False,
        "model_explicitness": "模型完全显性",
        "model_relation": "单一模型",
        "reasoning_chain": "直接判断",
        "hidden_conditions": "无",
        "critical_condition": "无临界",
        "classification_discussion": "无",
        "variable_relation": "无变量关系",
        "chemistry_methods": [],
        "equation_structure": "无方程",
        "calculation_complexity": "无需计算",
        "stoichiometric_calculation": "无",
        "equilibrium_calculation": "无",
        "information_carrier": "纯文字",
        "graph_structure": "无图表",
        "experiment_requirement": "无",
        "synthesis_route": "无",
        "separation_purification": "无",
        "context_type": "纯化学",
    }
    features.update(overrides)
    return features


def stage1_rating(features=None, accuracy=90.0):
    return {
        "features": copy.deepcopy(features or base_features()),
        "reason": "教材直接概念辨析，只有一个评分任务。",
        "score_evidence": "教材直接概念辨析，只有一个评分任务。",
        "score_band": core.score_band_for_accuracy(accuracy),
        "local_model_familiarity": "教材直接结论",
        "whole_question_burden": "低",
        "task_completion_structure": "单一评分任务",
        "threshold_review": {
            "can_reach_88": accuracy >= 88,
            "can_reach_75": accuracy >= 75,
            "can_reach_55": accuracy >= 55,
            "can_reach_35": accuracy >= 35,
        },
        "threshold_evidence": {
            "boundary_88": "教材直接结论。",
            "boundary_75": "没有隐藏关系。",
            "boundary_55": "没有长链。",
            "boundary_35": "没有压轴结构。",
        },
        "predicted_accuracy": accuracy,
    }


class AccuracyAndSchemaTests(unittest.TestCase):
    def test_accuracy_boundaries(self):
        cases = [
            (88, "难度1档"), (87.999, "难度2档"),
            (75, "难度2档"), (74.999, "难度3档"),
            (55, "难度3档"), (54.999, "难度4档"),
            (35, "难度4档"), (34.999, "难度5档"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(core.map_accuracy_to_level(value), expected)

    def test_score_bands_follow_raw_accuracy_intervals(self):
        cases = [
            (88, "88及以上"), (87.9, "75至87.9"),
            (75, "75至87.9"), (74.9, "55至74.9"),
            (55, "55至74.9"), (54.9, "35至54.9"),
            (35, "35至54.9"), (34.9, "35以下"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(core.score_band_for_accuracy(value), expected)

    def test_l1_must_match_l2(self):
        features = base_features(
            knowledge_L1=["有机化学"],
            knowledge_L2=["物质组成、分类与分散系"],
        )
        with self.assertRaisesRegex(ValueError, "不一致"):
            core.validate_feature_schema(features)

    def test_experiment_is_a_knowledge_module(self):
        features = base_features(
            knowledge_L1=["化学实验与探究"],
            knowledge_L2=["检验、鉴别与分离提纯"],
            knowledge_points=["蒸馏"],
            separation_purification="直接选择操作",
        )
        core.validate_feature_schema(features)

    def test_industrial_process_is_context_not_l1(self):
        self.assertNotIn("工业流程", core.KNOWLEDGE_L1)
        features = base_features(
            information_carrier="工艺流程图",
            context_type="工业流程",
        )
        core.validate_feature_schema(features)


class HighDifficultyFeatureTests(unittest.TestCase):
    def test_single_keyword_does_not_trigger_reaction_network(self):
        detected = core.detect_high_difficulty_features(
            base_features(substance_count="7种及以上")
        )
        self.assertNotIn("多物质多反应网络强耦合", detected.names)

    def test_reaction_network_joint_trigger(self):
        detected = core.detect_high_difficulty_features(base_features(
            substance_count="7种及以上",
            reaction_count="4个及以上",
            reaction_relation="多阶段强依赖反应链",
            model_relation="模型切换",
            reasoning_chain="多层因果",
        ))
        self.assertIn("多物质多反应网络强耦合", detected.names)

    def test_four_high_features_apply_point_seven_multiplier(self):
        features = base_features(
            substance_count="7种及以上",
            reaction_count="4个及以上",
            reaction_relation="竞争或副反应",
            reasoning_chain="多层因果",
            constraint_structure="多约束联合筛选",
            classification_discussion="3类讨论",
            model_relation="多模型或多平衡耦合",
            process_state_relation="前后状态强依赖",
        )
        detected = core.detect_high_difficulty_features(features)
        self.assertGreaterEqual(len(detected.names), 4)
        self.assertEqual(core.multiplier_for_high_count(len(detected.names)), 0.70)

    def test_advanced_design_requires_reasoning_and_constraint(self):
        simple = core.detect_high_difficulty_features(base_features(
            synthesis_route="自主设计或路线评价",
        ))
        self.assertNotIn("高阶实验合成或分离方案设计", simple.names)
        complex_case = core.detect_high_difficulty_features(base_features(
            synthesis_route="自主设计或路线评价",
            reasoning_chain="多层因果",
            constraint_structure="单一约束",
        ))
        self.assertIn("高阶实验合成或分离方案设计", complex_case.names)


class Stage1Tests(unittest.TestCase):
    def test_normalization_uses_whitelist_and_derives_l1(self):
        rating = stage1_rating(base_features(
            knowledge_L1=["有机化学"],
            knowledge_L2=["仪器操作与实验安全", "仪器操作与实验安全"],
            chemistry_methods=["质量守恒", "不存在的方法"],
        ), 80)
        rating["threshold_review"]["can_reach_88"] = True
        normalized, log = core.normalize_stage1_rating(rating)
        self.assertEqual(normalized["features"]["knowledge_L2"], ["实验基础与安全"])
        self.assertEqual(normalized["features"]["knowledge_L1"], ["化学实验与探究"])
        self.assertEqual(normalized["features"]["chemistry_methods"], ["守恒思想"])
        self.assertEqual(normalized["threshold_review"], {
            "can_reach_88": False, "can_reach_75": True,
            "can_reach_55": True, "can_reach_35": True,
        })
        self.assertTrue(any(item["action"] == "audit_only_fallback" for item in log))

    def test_legacy_combined_l2_expands_without_losing_knowledge_domain(self):
        rating = stage1_rating(base_features(
            knowledge_L2=["化学用语与化学计量"],
        ), 90)
        normalized, _ = core.normalize_stage1_rating(rating)
        self.assertEqual(normalized["features"]["knowledge_L2"], [
            "化学用语与物质组成", "物质的量与化学计量",
        ])
        self.assertEqual(normalized["features"]["knowledge_L1"], ["化学基本概念与定量关系"])

    def test_common_calculation_aliases_are_normalized(self):
        rating = stage1_rating(base_features(
            calculation_complexity="参数计算",
            stoichiometric_calculation="守恒计算",
        ), 80)
        normalized, _ = core.normalize_stage1_rating(rating)
        self.assertEqual(
            normalized["features"]["calculation_complexity"], "参数或范围计算"
        )
        self.assertEqual(
            normalized["features"]["stoichiometric_calculation"], "守恒差量或混合计算"
        )

    def test_operation_and_industry_aliases_are_normalized(self):
        rating = stage1_rating(base_features(
            experiment_requirement="多步实验操作",
            context_type="工业生产",
        ), 80)
        normalized, _ = core.normalize_stage1_rating(rating)
        self.assertEqual(normalized["features"]["experiment_requirement"], "多步操作组合")
        self.assertEqual(normalized["features"]["context_type"], "工业流程")

    def test_recent_stage1_enum_aliases_are_normalized(self):
        rating = stage1_rating(base_features(
            context_type="实验制备",
            critical_condition="显性临界过量条件",
            state_count="2-3种",
            experiment_requirement="定量实验与数据处理",
        ), 80)
        normalized, _ = core.normalize_stage1_rating(rating)
        self.assertEqual(normalized["features"]["context_type"], "实验探究")
        self.assertEqual(normalized["features"]["critical_condition"], "显性临界或过量条件")
        self.assertEqual(normalized["features"]["state_count"], "2个")
        self.assertEqual(normalized["features"]["experiment_requirement"], "标准数据处理")

    def test_experiment_feasibility_alias_is_normalized(self):
        for value in ("方案可行性评价", "实验探究与方案评价"):
            with self.subTest(value=value):
                rating = stage1_rating(base_features(experiment_requirement=value), 80)
                normalized, _ = core.normalize_stage1_rating(rating)
                self.assertEqual(
                    normalized["features"]["experiment_requirement"], "方案设计或可行性评价"
                )

    def test_unknown_feature_uses_audit_only_fallback_without_disabling_multiplier(self):
        rating = stage1_rating(base_features(
            reaction_relation="未定义的反应关系",
            substance_count="7种及以上",
            reaction_count="4个及以上",
            reasoning_chain="多层因果",
            constraint_structure="多约束联合筛选",
            model_relation="多模型或多平衡耦合",
            process_state_relation="前后状态强依赖",
        ), 80)
        normalized, log = core.normalize_stage1_rating(rating)
        enriched = core.enrich_stage1_rating(
            normalized, features_model_raw=rating["features"], normalization_log=log
        )
        self.assertEqual(
            enriched["features"]["reaction_relation"], core.AUDIT_ONLY_FEATURE_VALUE
        )
        self.assertGreaterEqual(enriched["high_difficulty_feature_count"], 2)
        self.assertEqual(enriched["multiplier_applied"], 1.0)

    def test_unknown_knowledge_l2_uses_audit_only_knowledge_module(self):
        rating = stage1_rating(base_features(
            knowledge_L2=["配位化合物"],
            knowledge_points=["配位化合物"],
        ), 80)
        normalized, log = core.normalize_stage1_rating(rating)
        enriched = core.enrich_stage1_rating(
            normalized, features_model_raw=rating["features"], normalization_log=log
        )
        self.assertEqual(normalized["features"]["knowledge_L2"], [core.AUDIT_ONLY_KNOWLEDGE_L2])
        self.assertEqual(normalized["features"]["knowledge_L1"], [core.AUDIT_ONLY_KNOWLEDGE_L1])
        self.assertTrue(any(
            item["field"] == "knowledge_L2" and item["action"] == "audit_only_fallback"
            for item in log
        ))

    def test_model_explicitness_alias_is_normalized(self):
        rating = stage1_rating(base_features(model_explicitness="完全显性"), 80)
        normalized, _ = core.normalize_stage1_rating(rating)
        self.assertEqual(normalized["features"]["model_explicitness"], "模型完全显性")

    def test_enrichment_derives_knowledge_fields(self):
        features = base_features(
            knowledge_L1=["化学基本概念与定量关系", "化学反应原理"],
            knowledge_L2=["氧化还原反应", "电化学"],
            knowledge_points=["氧化还原反应", "原电池", "电极反应", "电子守恒"],
            knowledge_count="1个",
            knowledge_scope="单知识点",
        )
        enriched = core.enrich_stage1_rating(stage1_rating(features, 80))
        self.assertEqual(enriched["features"]["knowledge_count"], "4个及以上")
        self.assertEqual(enriched["features"]["knowledge_scope"], "跨模块综合")

    def test_threshold_review_is_derived_from_accuracy(self):
        rating = stage1_rating(accuracy=80)
        rating["threshold_review"]["can_reach_88"] = True
        normalized, _ = core.normalize_stage1_rating(rating)
        self.assertEqual(normalized["threshold_review"], {
            "can_reach_88": False, "can_reach_75": True,
            "can_reach_55": True, "can_reach_35": True,
        })

    def test_stage1_without_boundary_fields_is_enriched(self):
        rating = stage1_rating(accuracy=80)
        rating.pop("threshold_review")
        rating.pop("threshold_evidence")
        normalized, log = core.normalize_stage1_rating(rating)
        enriched = core.enrich_stage1_rating(
            normalized, features_model_raw=rating["features"], normalization_log=log
        )
        self.assertEqual(enriched["difficulty_level_step1"], "难度2档")
        self.assertEqual(enriched["threshold_review"], {
            "can_reach_88": False, "can_reach_75": True,
            "can_reach_55": True, "can_reach_35": True,
        })
        self.assertEqual(
            enriched["threshold_evidence"]["boundary_75"], rating["score_evidence"]
        )

    def test_score_band_mismatch_is_normalized_and_logged(self):
        rating = stage1_rating(accuracy=80)
        rating["score_band"] = "55至74.9"
        normalized, log = core.normalize_stage1_rating(rating)
        self.assertEqual(normalized["score_band"], "75至87.9")
        self.assertEqual(normalized["score_band_model_raw"], "55至74.9")
        self.assertTrue(any(item["field"] == "score_band" for item in log))

    def test_multiplier_is_applied_after_original_accuracy(self):
        features = base_features(
            substance_count="7种及以上",
            reaction_count="4个及以上",
            reaction_relation="竞争或副反应",
            reasoning_chain="多层因果",
            constraint_structure="多约束联合筛选",
            classification_discussion="3类讨论",
            model_relation="多模型或多平衡耦合",
            process_state_relation="前后状态强依赖",
        )
        enriched = core.enrich_stage1_rating(stage1_rating(features, 80))
        self.assertEqual(enriched["original_predicted_accuracy"], 80)
        self.assertEqual(enriched["multiplier_applied"], 0.70)
        self.assertEqual(enriched["predicted_accuracy"], 56)
        self.assertEqual(enriched["difficulty_level_step1"], "难度3档")

    def test_low_structure_score_conflict_is_audited_not_overwritten(self):
        enriched = core.enrich_stage1_rating(stage1_rating(accuracy=70))
        self.assertEqual(enriched["original_predicted_accuracy"], 70)
        self.assertTrue(enriched["accuracy_scale_audit"]["low_structure_score_conflict"])

    def test_three_state_and_multi_reaction_boundary_risks_are_audited(self):
        features = base_features(
            step_count="6-8步",
            reaction_count="4个及以上",
            reaction_relation="多阶段强依赖反应链",
            state_count="3个及以上",
            process_state_relation="前后状态强依赖",
            reasoning_chain="多层因果",
        )
        enriched = core.enrich_stage1_rating(stage1_rating(features, 60))
        audit = enriched["accuracy_scale_audit"]
        self.assertTrue(audit["three_state_boundary_review_risk"])
        self.assertTrue(audit["multi_reaction_boundary_review_risk"])


class InputPreparationTests(unittest.TestCase):
    def test_fillblank_without_options_is_not_anomaly(self):
        prepared = core.prepare_question({
            "question_id": "1", "structure_type": "fillblank",
            "stem": "填写化学方程式", "options": "", "analysis": "略",
            "sub_questions": [],
        })
        self.assertFalse(prepared.input_quality["option_anomaly"])

    def test_choice_without_options_is_anomaly(self):
        prepared = core.prepare_question({
            "question_id": "2", "structure_type": "danxuan",
            "stem": "选择正确说法", "options": "", "analysis": "略",
            "sub_questions": [],
        })
        self.assertTrue(prepared.input_quality["option_anomaly"])
        self.assertEqual(prepared.input_quality["input_sufficiency"], "不足")

    def test_empty_parent_stem_with_complete_children_is_valid(self):
        prepared = core.prepare_question({
            "question_id": "3", "structure_type": "fuhe", "stem": "",
            "options": "", "analysis": "", "stem_image_url": "https://example/stem.png",
            "sub_questions": [
                {"question_id": "3-1", "stem": "第一问"},
                {"question_id": "3-2", "stem": "第二问"},
            ],
        })
        self.assertEqual(prepared.input_quality["content_mode"], "subquestion_complete")
        self.assertNotEqual(prepared.input_quality["input_sufficiency"], "不足")

    def test_empty_stem_with_image_is_image_dependent(self):
        prepared = core.prepare_question({
            "question_id": "4", "structure_type": "danxuan", "stem": "",
            "options": "A.甲\nB.乙", "analysis": "",
            "stem_image_url": "https://example/stem.png", "sub_questions": [],
            "difficulty": "5", "percent_correct": "60", "answered_count": "20",
        })
        self.assertEqual(prepared.input_quality["content_mode"], "image_dependent")
        self.assertEqual(prepared.selected_image_urls, ["https://example/stem.png"])
        self.assertNotIn("difficulty", prepared.question)
        self.assertNotIn("percent_correct", prepared.question)
        self.assertNotIn("answered_count", prepared.question)
        self.assertEqual(prepared.source_difficulty_untrusted, "5")

    def test_subquestions_are_sorted_and_child_analysis_counts(self):
        prepared = core.prepare_question({
            "question_id": "5", "structure_type": "fuhe", "stem": "综合题",
            "options": "", "analysis": "", "sub_questions": [
                {"question_id": "10", "stem": "第十问", "analysis": "解析十"},
                {"question_id": "2", "stem": "第二问", "analysis": "解析二"},
            ],
        })
        self.assertEqual(
            [item["question_id"] for item in prepared.question["sub_questions"]],
            ["2", "10"],
        )
        self.assertTrue(prepared.input_quality["subquestion_analysis_available"])
        self.assertTrue(prepared.input_quality["has_analysis"])

    def test_analysis_image_is_not_sent_when_text_is_complete(self):
        prepared = core.prepare_question({
            "question_id": "6", "structure_type": "fillblank",
            "stem": "写出反应方程式", "options": "", "analysis": "依据守恒配平。",
            "analysis_image_url": "https://example/answer.png", "sub_questions": [],
        })
        self.assertEqual(prepared.selected_image_urls, [])


class VerificationTests(unittest.TestCase):
    def test_no_structural_revision_forces_original_values(self):
        stage1 = core.enrich_stage1_rating(stage1_rating(accuracy=80))
        result = core.recalculate_verification(stage1, {
            "difficulty_source": "无新证据",
            "feature_corrections": [],
            "missed_features": ["无"],
            "has_structural_revision": False,
            "adjacent_boundary_review": {
                "boundaries_checked": ["75边界", "55边界"],
                "verdict": "维持",
                "decisive_evidence": ["特征与解析一致"],
            },
            "confidence": "高",
            "reviewed_original_predicted_accuracy": 20,
            "reviewed_high_difficulty_features": [],
            "analysis": "维持",
        })
        self.assertEqual(result["reviewed_original_predicted_accuracy"], 80)
        self.assertEqual(result["reviewed_difficulty_level"], "难度2档")

    def test_high_feature_revision_recalculates_multiplier(self):
        features = base_features(
            substance_count="7种及以上",
            reaction_count="4个及以上",
            reaction_relation="竞争或副反应",
            reasoning_chain="多层因果",
            constraint_structure="多约束联合筛选",
            classification_discussion="3类讨论",
            model_relation="多模型或多平衡耦合",
            process_state_relation="前后状态强依赖",
        )
        stage1 = core.enrich_stage1_rating(stage1_rating(features, 80))
        reviewed_names = stage1["high_difficulty_features"][:2]
        result = core.recalculate_verification(stage1, {
            "difficulty_source": "两类触发共享同一反应网络，去除重复计数",
            "feature_corrections": [],
            "missed_features": ["无"],
            "has_structural_revision": True,
            "adjacent_boundary_review": {
                "boundaries_checked": ["55边界", "35边界"],
                "verdict": "应更简单一档",
                "decisive_evidence": ["高难特征重复计数"],
            },
            "confidence": "高",
            "reviewed_original_predicted_accuracy": 80,
            "reviewed_high_difficulty_features": reviewed_names,
            "analysis": "保留两个独立高难结构",
        })
        self.assertTrue(result["has_structural_revision"])
        self.assertEqual(result["reviewed_high_difficulty_feature_count"], 2)
        self.assertEqual(result["reviewed_multiplier_applied"], 1.0)

    def test_disabled_auto_adjust_routes_changed_level_to_manual_review(self):
        final = core.finalize_level(
            current_level="难度3档",
            reasonableness="偏低",
            model_suggested_level="难度4档",
            multiplier_reasonableness="合理",
            input_sufficiency="充分",
            original_high_count=0,
            reviewed_high_count=0,
            enable_auto_adjust=False,
        )
        self.assertEqual(final.final_level, "难度3档")
        self.assertTrue(final.needs_manual_review)

    def test_correction_with_stale_original_value_is_rejected(self):
        stage1 = core.enrich_stage1_rating(stage1_rating(accuracy=80))
        result = core.recalculate_verification(stage1, {
            "difficulty_source": "模型显性度复核",
            "feature_corrections": [{
                "field": "model_explicitness", "original_value": "隐含模型",
                "reviewed_value": "半隐含模型", "evidence": "题干明确给出部分关系",
            }],
            "missed_features": ["无"], "has_structural_revision": True,
            "adjacent_boundary_review": {
                "boundaries_checked": ["75边界", "55边界"], "verdict": "维持",
                "decisive_evidence": ["原值与第一阶段记录不一致"],
            },
            "confidence": "高", "reviewed_original_predicted_accuracy": 70,
            "reviewed_high_difficulty_features": [], "analysis": "不采纳",
        })
        self.assertFalse(result["has_structural_revision"])
        self.assertEqual(len(result["unsupported_feature_corrections"]), 1)
        self.assertEqual(result["reviewed_original_predicted_accuracy"], 80)

    def test_auto_adjust_requires_one_step_direction_and_boundary_agreement(self):
        final = core.finalize_level(
            current_level="难度3档", reasonableness="偏低",
            model_suggested_level="难度5档", multiplier_reasonableness="合理",
            input_sufficiency="充分", original_high_count=0,
            reviewed_high_count=0, enable_auto_adjust=True,
        )
        self.assertEqual(final.final_level, "难度4档")
        self.assertTrue(final.needs_manual_review)

    def test_multiplier_bucket_change_always_routes_to_manual_review(self):
        final = core.finalize_level(
            current_level="难度3档", reasonableness="偏高",
            model_suggested_level="难度2档", multiplier_reasonableness="不合理",
            input_sufficiency="充分", original_high_count=2,
            reviewed_high_count=3, enable_auto_adjust=True,
        )
        self.assertEqual(final.final_level, "难度3档")
        self.assertTrue(final.needs_manual_review)


class PromptAssetTests(unittest.TestCase):
    def test_prompt_compiles_and_exposes_four_sections(self):
        path = ROOT / "prompts" / "高中化学难度打标提示词.txt"
        source = path.read_text(encoding="utf-8")
        namespace = {}
        exec(compile(source, str(path), "exec"), namespace)
        for name in (
            "FEATURE_EXTRACTION_PROMPT_PREFIX",
            "FEATURE_EXTRACTION_PROMPT_SUFFIX",
            "VERIFICATION_PROMPT_PREFIX",
            "VERIFICATION_PROMPT_SUFFIX",
        ):
            self.assertTrue(namespace.get(name))

    def test_prompt_contains_every_required_feature(self):
        text = (ROOT / "prompts" / "高中化学难度打标提示词.txt").read_text(encoding="utf-8")
        for field in core.REQUIRED_FEATURE_FIELDS:
            self.assertIn(field, text)

    def test_prompt_defines_type_aware_empty_options(self):
        text = (ROOT / "prompts" / "高中化学难度打标提示词.txt").read_text(encoding="utf-8")
        self.assertIn("填空、主观和复合题可以没有 options", text)
        self.assertIn("文字题干为空但有题干图片", text)

    def test_prompt_keeps_industry_out_of_l1(self):
        text = (ROOT / "prompts" / "高中化学难度打标提示词.txt").read_text(encoding="utf-8")
        self.assertIn("知识模块只标注解题必需的化学知识，不按工业流程、生活生产等情境分类", text)

    def test_prompt_requires_continuous_accuracy_estimate_and_option_detail(self):
        text = (ROOT / "prompts" / "高中化学难度打标提示词.txt").read_text(encoding="utf-8")
        for phrase in (
            "local_model_familiarity", "whole_question_burden",
            "task_completion_structure", "score_evidence",
            "不得把各选项判断正确率相乘",
            "不得把四项正确率机械相乘",
            "先输出 score_band",
            "不要输出 threshold_review 或 threshold_evidence",
        ):
            self.assertIn(phrase, text)

    def test_stage1_prompt_requires_one_score_band_not_four_boundary_reviews(self):
        path = ROOT / "prompts" / "高中化学难度打标提示词.txt"
        namespace = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
        stage1_prompt = (
            namespace["FEATURE_EXTRACTION_PROMPT_PREFIX"]
            + namespace["FEATURE_EXTRACTION_PROMPT_SUFFIX"]
        )
        self.assertIn("score_band", stage1_prompt)
        self.assertIn("不要输出 threshold_review 或 threshold_evidence", stage1_prompt)

    def test_prompt_defines_task_modeling_without_task_unit_schema(self):
        text = (ROOT / "prompts" / "高中化学难度打标提示词.txt").read_text(encoding="utf-8")
        for phrase in ("评分输出不等于实质化学任务", "答案不复用不等于不共享模型"):
            self.assertIn(phrase, text)
        self.assertNotIn('"task_units"', text)

class RunnerAssetTests(unittest.TestCase):
    def setUp(self):
        self.path = ROOT / "src" / "high_chemistry_difficulty_rating_and_verify.py"
        self.source = self.path.read_text(encoding="utf-8")

    def test_runner_compiles(self):
        compile(self.source, str(self.path), "exec")
        self.assertIn('"high_chemistry_two_stage_v9_score_band_accuracy"', self.source)
        self.assertIn("_stage1_repair_feedback", self.source)
        self.assertIn("_is_retriable_image_download_timeout", self.source)
        self.assertIn("timeout while downloading url", self.source)

    def test_runner_has_required_controls(self):
        for flag in (
            "--input", "--output", "--errors", "--prompt", "--concurrency",
            "--limit", "--per-level", "--no-cache", "--image-mode",
        ):
            self.assertIn(flag, self.source)

    def test_runner_uses_environment_credentials(self):
        self.assertIn('os.getenv("API_KEY"', self.source)
        self.assertNotRegex(
            self.source,
            re.compile(r'API_KEY\s*=\s*["\'][0-9a-f]{20,}["\']', re.I),
        )

    def test_runner_defaults_to_supplied_dataset_and_prompt(self):
        self.assertIn('"high-chemistry-sample25k.jsonl"', self.source)
        self.assertIn('"高中化学难度打标提示词.txt"', self.source)
        self.assertTrue((ROOT / "data" / "samples" / "high-chemistry-sample25k.jsonl").exists())

    def test_runner_never_sends_source_labels(self):
        for field in ("difficulty", "percent_correct", "answered_count"):
            self.assertIn(field, core.UNTRUSTED_LABEL_FIELDS)
        self.assertIn("source_difficulty_untrusted", self.source)

class DatasetCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data_path = ROOT / "data" / "samples" / "high-chemistry-sample25k.jsonl"

    def test_all_25000_rows_are_type_aware_compatible(self):
        rows = 0
        empty_options = 0
        option_anomalies = 0
        empty_stems = 0
        insufficient_empty_stems = []
        empty_stems_without_selected_images = []
        with self.data_path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                rows += 1
                if not str(row.get("options", "")).strip():
                    empty_options += 1
                if not str(row.get("stem", "")).strip():
                    empty_stems += 1
                prepared = core.prepare_question(row, image_mode="auto")
                if prepared.input_quality["option_anomaly"]:
                    option_anomalies += 1
                if not str(row.get("stem", "")).strip() and prepared.input_quality["input_sufficiency"] == "不足":
                    insufficient_empty_stems.append(row.get("question_id"))
                if not str(row.get("stem", "")).strip() and not prepared.selected_image_urls:
                    empty_stems_without_selected_images.append(row.get("question_id"))
        self.assertEqual(rows, 25000)
        self.assertEqual(empty_options, 9224)
        self.assertEqual(option_anomalies, 0)
        self.assertEqual(empty_stems, 23)
        self.assertEqual(insufficient_empty_stems, [])
        self.assertEqual(empty_stems_without_selected_images, [])


if __name__ == "__main__":
    unittest.main()
