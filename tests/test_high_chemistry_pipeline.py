# -*- coding: utf-8 -*-
"""高中化学单阶段难度 Pipeline 的离线单元测试。"""

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
        "local_model_familiarity": "教材直接结论",
        "whole_question_burden": "低",
        "task_completion_structure": "单一评分任务",
        "predicted_accuracy": accuracy,
    }


class AccuracyAndSchemaTests(unittest.TestCase):
    def test_accuracy_boundaries(self):
        cases = [
            (88, "难度1档"), (87.999, "难度2档"),
            (85, "难度2档"), (84.999, "难度3档"),
            (58, "难度3档"), (57.999, "难度4档"),
            (38, "难度4档"), (37.999, "难度5档"),
        ]
        for value, expected in cases:
            with self.subTest(value=value):
                self.assertEqual(core.map_accuracy_to_level(value), expected)

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
    def test_preparation_derives_l1_without_alias_mapping(self):
        rating = stage1_rating(base_features(
            knowledge_L1=["有机化学"],
            knowledge_L2=["实验基础与安全"],
        ), 80)
        prepared = core.prepare_stage1_rating(rating)
        self.assertEqual(prepared["features"]["knowledge_L2"], ["实验基础与安全"])
        self.assertEqual(prepared["features"]["knowledge_L1"], ["化学实验与探究"])

    def test_unknown_enum_is_rejected_instead_of_repaired(self):
        rating = stage1_rating(base_features(reaction_relation="未定义的反应关系"), 80)
        with self.assertRaisesRegex(ValueError, "reaction_relation 非法值"):
            core.prepare_stage1_rating(rating)

    def test_exact_array_duplicates_are_deduplicated(self):
        rating = stage1_rating(base_features(
            knowledge_L2=["物质组成、分类与分散系", "物质组成、分类与分散系"],
            knowledge_points=["氧化还原", "氧化还原"],
            chemistry_methods=["守恒思想", "守恒思想"],
        ), 80)
        prepared = core.prepare_stage1_rating(rating)
        self.assertEqual(prepared["features"]["knowledge_L2"], ["物质组成、分类与分散系"])
        self.assertEqual(prepared["features"]["knowledge_points"], ["氧化还原"])
        self.assertEqual(prepared["features"]["chemistry_methods"], ["守恒思想"])

    def test_strict_output_schema_uses_canonical_enums(self):
        schema = core.build_stage1_output_schema()
        self.assertFalse(schema["additionalProperties"])
        features = schema["properties"]["features"]
        self.assertFalse(features["additionalProperties"])
        self.assertEqual(set(features["required"]), set(core.REQUIRED_FEATURE_FIELDS))
        reaction_enum = features["properties"]["reaction_relation"]["enum"]
        self.assertEqual(set(reaction_enum), core.FEATURE_OPTIONS["reaction_relation"])
        self.assertNotIn("未确定（仅审计）", reaction_enum)

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

    def test_stage1_rating_does_not_require_threshold_fields(self):
        enriched = core.enrich_stage1_rating(stage1_rating(accuracy=80))
        self.assertEqual(enriched["original_predicted_accuracy"], 80)

    def test_overlapping_high_features_are_grouped_before_multiplier(self):
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
        self.assertEqual(enriched["high_difficulty_feature_count"], 4)
        self.assertEqual(enriched["effective_high_difficulty_feature_count"], 2)
        self.assertEqual(enriched["multiplier_applied"], 1.0)
        self.assertEqual(enriched["predicted_accuracy"], 80)
        self.assertEqual(enriched["difficulty_level"], "难度3档")

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


class PromptAssetTests(unittest.TestCase):
    def test_prompt_compiles_and_exposes_single_stage_sections(self):
        path = ROOT / "prompts" / "高中化学难度打标提示词.txt"
        source = path.read_text(encoding="utf-8")
        namespace = {}
        exec(compile(source, str(path), "exec"), namespace)
        for name in (
            "FEATURE_EXTRACTION_PROMPT_PREFIX",
            "FEATURE_EXTRACTION_PROMPT_SUFFIX",
        ):
            self.assertTrue(namespace.get(name))
        self.assertNotIn("VERIFICATION_PROMPT_PREFIX", namespace)
        self.assertNotIn("VERIFICATION_PROMPT_SUFFIX", namespace)

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

    def test_prompt_requires_interval_positioning_and_option_detail(self):
        text = (ROOT / "prompts" / "高中化学难度打标提示词.txt").read_text(encoding="utf-8")
        for phrase in (
            "local_model_familiarity", "whole_question_burden",
            "task_completion_structure", "原始正确率的四个边界与五档区间",
            "按相邻边界距离给出连续分数",
            "完整作答稳定性校准",
            "完整完成本题全部必要任务的概率",
            "核心拔高任务",
            "取得整题答案不可省略",
            "且至少两个任务相互依赖",
            "核心拔高任务数为0",
            "存在1个核心拔高任务",
            "核心拔高任务数为2个及以上",
            "原始正确率低于38时，必须分别指出两个核心拔高任务",
            "整体拔高证据",
            "若难点表现为整体拔高证据",
            "关键错误传播",
            "核心判定：课内常规综合",
            "主要难度来自理解题意并正确衔接这些常规步骤",
            "只能用于排除5档，不能推翻4档判断",
            "前后依赖检查",
            "能否独立建模并得到该阶段答案",
            "共同背景但各阶段都能独立建模",
            "有效决策负担", "失分传播与整题完成",
            "不得先选取边界值、整数中点或任何示例值作为模板分数",
            "不得把各选项判断正确率相乘",
            "不得把四项正确率机械相乘",
            "共享复杂模型",
        ):
            self.assertIn(phrase, text)
        self.assertNotIn('"predicted_accuracy": 72.0', text)

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
        self.assertIn('"high_chemistry_single_stage_v3_deduplicated_features"', self.source)
        self.assertIn('"type": "json_schema"', self.source)
        self.assertIn('"strict": True', self.source)
        self.assertIn("build_stage1_output_schema", self.source)
        self.assertIn("_is_retriable_image_download_timeout", self.source)
        self.assertIn("timeout while downloading url", self.source)
        self.assertNotIn("call_stage2", self.source)
        self.assertNotIn("ENABLE_STAGE2_AUTO_ADJUST", self.source)
        self.assertNotIn('"difficulty_rating_stage1": stage1', self.source)
        self.assertNotIn('"needs_manual_review"', self.source)

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
