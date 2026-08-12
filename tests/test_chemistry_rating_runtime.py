import json
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import junior_chemistry_schema as schema


def rating(level="基础题", topic_ids=None):
    return {
        "features": {
            "knowledge": {"topic_ids": topic_ids or ["U02_T03"]},
            "task_count": "1项",
            "knowledge_distribution": "单一知识点",
            "chemical_object_distribution": "单一化学对象",
            "step_count": "1步",
            "task_relation": "单项任务",
            "solution_method": "一条规则直接应用",
            "classification_discussion": "无",
            "reverse_tracing": "无",
            "visual_content": "无图片信息",
            "visual_item_count": "无图片",
            "visual_complexity": "无图片信息",
            "information_operation": "无需额外提取",
            "reaction_count": "0个",
            "reaction_relation": "无反应关系",
            "experiment_operation": "基本操作或读数判断",
            "experiment_analysis": "无",
            "experiment_design": "无",
            "error_analysis": "无",
            "calculation_type": "无",
            "calculation_steps": "无",
            "calculation_structure": "无任何计算",
            "special_method": "无",
            "hidden_condition_count": "0个",
            "hidden_condition_type": "无",
            "condition_relation": "无条件限制",
            "interference_type": "无",
            "expression_type": "无",
            "subjective_response": "无",
            "given_information": "题干未提供新增规则",
            "cross_subject": "无",
            "curriculum_scope": {"scope": "within_junior", "extra_points": []},
        },
        "reasoning": {
            "knowledge_points": "涉及教材知识点。",
            "solution_process": "完成一步判断。",
            "main_difficulty_factors": "基础规则应用。",
            "level_basis": "判为基础题。",
        },
        "difficulty_level": level,
    }


class JuniorChemistrySchemaTests(unittest.TestCase):
    def test_curriculum_map_is_loaded(self):
        topics = schema.load_curriculum_topics()
        self.assertGreaterEqual(len(topics), 58)
        self.assertEqual(topics["U10_T02"]["unit_id"], "U10")

    def test_duplicate_topic_ids_are_deduplicated_without_level_change(self):
        value = rating("中等题", ["U02_T03", "U02_T03", "U10_T01"])
        result = schema.postprocess_chemistry_difficulty(value, {})

        self.assertEqual(
            result["features"]["knowledge"]["topic_ids"],
            ["U02_T03", "U10_T01"],
        )
        self.assertEqual(result["difficulty_level"], "中等题")
        topic_schema = schema.rating_tool_definition()["parameters"]["properties"][
            "features"
        ]["properties"]["knowledge"]["properties"]["topic_ids"]
        self.assertTrue(topic_schema["uniqueItems"])

    def test_exactly_thirty_single_choice_core_features(self):
        self.assertEqual(len(schema.FEATURE_OPTIONS), 30)
        self.assertTrue(all(isinstance(values, tuple) for values in schema.FEATURE_OPTIONS.values()))
        value = rating()
        for field in schema.FEATURE_OPTIONS:
            self.assertIsInstance(value["features"][field], str)

    def test_postprocess_computes_coverage_without_writeback(self):
        result = schema.postprocess_chemistry_difficulty(
            rating("送分题", ["U02_T03", "U10_T01"]), {}
        )
        knowledge = result["features"]["knowledge"]
        self.assertEqual(knowledge["knowledge_point_count"], 2)
        self.assertEqual(knowledge["unit_count"], 2)
        self.assertTrue(knowledge["cross_unit"])
        self.assertEqual(
            result["features"]["knowledge_distribution"], "跨单元不同知识点"
        )
        self.assertEqual(result["difficulty_level"], "送分题")
        self.assertFalse(result["postprocess"]["writeback_applied"])
        self.assertEqual(result["postprocess"]["candidates"], [])

    def test_error_analysis_is_written_back_to_medium_floor(self):
        value = rating("送分题")
        value["features"]["error_analysis"] = "量筒读数误差"
        result = schema.postprocess_chemistry_difficulty(value, {"stem": "判断量筒读数误差。"})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertTrue(result["postprocess"]["writeback_applied"])
        self.assertEqual(result["postprocess_actions"][0]["rule"], "T1_error_analysis_floor")

    def test_question_statistics_do_not_create_difficulty_floors(self):
        long_stem = "化" * 101
        result = schema.postprocess_chemistry_difficulty(
            rating("送分题"), {"stem": long_stem, "options": "A.甲 B.乙 C.丙 D.丁"}
        )
        stats = result["postprocess"]["question_statistics"]
        self.assertEqual(stats["stem_char_count"], 101)
        self.assertEqual(stats["stem_length_band"], "101-300字")
        self.assertEqual(stats["option_count"], 4)
        self.assertEqual(result["difficulty_level"], "送分题")
        self.assertFalse(result["postprocess"]["writeback_applied"])

        result = schema.postprocess_chemistry_difficulty(
            rating("送分题"), {"sub_questions": [{"stem": str(i)} for i in range(5)]}
        )
        self.assertEqual(result["difficulty_level"], "送分题")

        value = rating("送分题")
        value["features"]["visual_content"] = "仪器图"
        value["features"]["visual_item_count"] = "4幅及以上"
        value["features"]["visual_complexity"] = "多个同类型图像"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "送分题")

    def test_related_enum_fields_are_normalized_without_retry(self):
        value = rating()
        value["features"]["visual_content"] = "仪器图"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["visual_item_count"], "1幅")
        self.assertEqual(result["features"]["visual_complexity"], "单一同类型图像")

        value = rating()
        value["features"]["subjective_response"] = "有"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["subjective_response"], "无")
        self.assertGreaterEqual(
            len(result["postprocess"]["feature_normalization_actions"]), 1
        )

    def test_common_aliases_are_normalized_without_retry(self):
        value = rating()
        value["features"]["experiment_operation"] = "仪器识别"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(
            result["features"]["experiment_operation"], "仪器识别或名称"
        )

        value = rating()
        value["features"]["calculation_type"] = "反应后气体质量"
        value["features"]["calculation_steps"] = "1步"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(
            result["features"]["calculation_type"],
            "反应后气体沉淀固体质量",
        )

    def test_complete_json_with_unescaped_reasoning_quotes_is_recovered(self):
        raw = json.dumps(rating("送分题"), ensure_ascii=False)
        malformed = raw.replace(
            "完成一步判断。", '使用"是否生成新物质"规则完成判断。'
        )
        parsed, mode = schema.parse_model_json_text(malformed)
        self.assertEqual(mode, "local_recovered")
        self.assertEqual(parsed["difficulty_level"], "送分题")
        self.assertIn('"是否生成新物质"', parsed["reasoning"]["solution_process"])

    def test_complete_json_with_missing_reasoning_quote_is_recovered(self):
        raw = json.dumps(rating("中等题"), ensure_ascii=False)
        malformed = raw.replace(
            '"solution_process": "完成一步判断。"',
            '"solution_process":完成一步判断。"',
        )
        parsed, mode = schema.parse_model_json_text(malformed)
        self.assertEqual(mode, "local_recovered")
        self.assertEqual(parsed["reasoning"]["solution_process"], "完成一步判断。")

    def test_thinking_prefix_before_complete_json_is_ignored(self):
        raw = json.dumps(rating("基础题"), ensure_ascii=False)
        parsed, mode = schema.parse_model_json_text(
            "内部分析文本不应进入结果。<think_end>" + raw
        )
        self.assertEqual(mode, "local_recovered")
        self.assertEqual(parsed["difficulty_level"], "基础题")

    def test_truncated_json_is_not_repaired(self):
        raw = json.dumps(rating("基础题"), ensure_ascii=False)
        parsed, mode = schema.parse_model_json_text(raw[:-20])
        self.assertEqual(parsed, {})
        self.assertEqual(mode, "failed")

    def test_nonreaction_multistep_calculation_has_own_structure(self):
        value = rating("中等题", ["U04_T06"])
        value["features"].update({
            "calculation_type": "相对分子质量或元素质量计算",
            "calculation_steps": "2-3步",
            "calculation_structure": "无任何计算",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(
            result["features"]["calculation_structure"], "多步常规计算"
        )

    def test_cross_field_value_is_normalized_without_retry(self):
        value = rating("拔高题")
        value["features"]["calculation_type"] = "多个化学反应计算"
        value["features"]["calculation_steps"] = "4步及以上"
        value["features"]["special_method"] = "元素守恒"
        value["features"]["calculation_structure"] = "多个化学反应计算"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["calculation_type"], "化学方程式计算")
        self.assertEqual(
            result["features"]["calculation_structure"],
            "多个化学反应计算",
        )
        actions = result["postprocess"]["feature_normalization_actions"]
        self.assertTrue(any(action["field"] == "calculation_type" for action in actions))
        self.assertEqual(result["difficulty_level"], "拔高题")

    def test_unknown_topic_id_is_recovered_only_from_unique_knowledge_text(self):
        value = rating("中等题", ["U05_T05"])
        value["reasoning"]["knowledge_points"] = "本题考查质量守恒定律。"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(
            result["features"]["knowledge"]["topic_ids"], ["U05_T01"]
        )
        actions = result["postprocess"]["feature_normalization_actions"]
        self.assertTrue(any(action["field"] == "knowledge.topic_ids" for action in actions))

    def test_unknown_topic_id_without_unique_evidence_still_fails(self):
        value = rating("中等题", ["U05_T99"])
        value["reasoning"]["knowledge_points"] = "本题涉及化学知识。"
        with self.assertRaisesRegex(schema.ChemistrySchemaError, "无法唯一映射"):
            schema.postprocess_chemistry_difficulty(value, {})

    def test_normalization_never_erases_explicit_high_difficulty_evidence(self):
        value = rating("拔高题")
        value["features"].update({
            "visual_content": "曲线图",
            "visual_item_count": "1幅",
            "visual_complexity": "图像拐点或分段分析",
            "information_operation": "图像拐点或分段分析",
            "reaction_count": "3个",
            "reaction_relation": "多个反应连续",
            "calculation_type": "多类计算综合",
            "calculation_steps": "4步及以上",
            "calculation_structure": "多个化学反应计算",
            "special_method": "差量法、元素守恒",
            "hidden_condition_count": "多个关联条件",
            "hidden_condition_type": "多类条件联合",
            "condition_relation": "多个关联条件",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        features = result["features"]
        self.assertEqual(features["reaction_count"], "2-3个")
        self.assertEqual(features["reaction_relation"], "多个反应连续")
        self.assertEqual(features["special_method"], "多种特殊方法联合")
        self.assertEqual(features["hidden_condition_count"], "2个")
        self.assertEqual(features["condition_relation"], "多个关联条件")
        self.assertEqual(features["visual_complexity"], "复杂高难图像")

    def test_consistency_uses_nonzero_relation_evidence_over_zero_counts(self):
        value = rating("拔高题")
        value["features"].update({
            "reaction_count": "0个",
            "reaction_relation": "反应先后或过量不足",
            "hidden_condition_count": "0个",
            "hidden_condition_type": "多类条件联合",
            "condition_relation": "多个关联条件",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["reaction_count"], "2-3个")
        self.assertEqual(
            result["features"]["reaction_relation"], "反应先后或过量不足"
        )
        self.assertEqual(result["features"]["hidden_condition_count"], "2个")
        self.assertEqual(result["features"]["condition_relation"], "多个关联条件")

    def test_giveaway_review_uses_only_concrete_teacher_anchors(self):
        material_confusion = rating("送分题", ["U11_T02"])
        material_confusion["features"].update({
            "chemical_object_distribution": "不同类多个化学对象",
            "step_count": "1步",
            "solution_method": "多条规则分别判断",
            "interference_type": "多个选项规则切换",
        })
        combustion_facts = rating("送分题", ["U02_T03"])
        combustion_facts["features"].update({
            "chemical_object_distribution": "同类多个化学对象",
            "experiment_analysis": "实验现象判断",
            "solution_method": "多条规则分别判断",
        })
        for value in (material_confusion, combustion_facts):
            with self.subTest(features=value["features"]):
                result = schema.postprocess_chemistry_difficulty(value, {})
                self.assertEqual(result["difficulty_level"], "基础题")
                self.assertEqual(
                    result["postprocess_actions"][-1]["rule"],
                    "S1_giveaway_to_foundation_teacher_anchor",
                )

        simple_classification = rating("送分题", ["U11_T02"])
        simple_classification["features"].update({
            "chemical_object_distribution": "同类多个化学对象",
            "interference_type": "易混概念",
        })
        unchanged = schema.postprocess_chemistry_difficulty(simple_classification, {})
        self.assertEqual(unchanged["difficulty_level"], "送分题")

    def test_teacher_sensitive_single_topic_paths_use_specific_evidence(self):
        cases = [
            (
                ["U04_T05"],
                {"calculation_type": "化合价或化学式计算", "calculation_steps": "1步", "calculation_structure": "一步或口算"},
                {"stem": "CuO中氧元素为-2价，求铜元素的化合价。"},
            ),
            (
                ["U09_T03"], {},
                {"stem": "与固体溶解度无关的因素是（ ）。"},
            ),
            (
                ["U03_T01"], {},
                {"stem": "物质发生三态变化的主要原因是（ ）。", "options": "分子大小；分子本身；分子间隔；运动状态"},
            ),
            (
                ["U01_T04"], {"visual_content": "仪器图", "visual_item_count": "4幅及以上", "visual_complexity": "多个同类型图像"},
                {"stem": "在过滤操作中，不需要用到的仪器是（ ）。<image>"},
            ),
        ]
        for topic_ids, feature_updates, data in cases:
            with self.subTest(topic_ids=topic_ids, data=data):
                value = rating("送分题", topic_ids)
                value["features"].update(feature_updates)
                result = schema.postprocess_chemistry_difficulty(value, data)
                self.assertEqual(result["difficulty_level"], "基础题")

    def test_sensitive_topic_name_alone_does_not_raise_giveaway(self):
        value = rating("送分题", ["U11_T01"])
        result = schema.postprocess_chemistry_difficulty(
            value, {"stem": "下列食物中富含蛋白质的是（ ）。"}
        )
        self.assertEqual(result["difficulty_level"], "送分题")

    def test_uncommon_formula_details_raise_without_catching_simple_formula_item(self):
        detailed = rating("送分题", ["U04_T04", "U04_T05"])
        detailed["features"].update({
            "chemical_object_distribution": "不同类多个化学对象",
            "interference_type": "易混概念",
        })
        detailed_result = schema.postprocess_chemistry_difficulty(detailed, {
            "stem": "下列物质的化学式书写正确的是（ ）。",
            "options": "氧化铁FeO；甲烷CH4；氯化铵NH3Cl；硝酸NO3",
        })
        self.assertEqual(detailed_result["difficulty_level"], "基础题")

        simple = rating("送分题", ["U04_T04", "U04_T05"])
        simple["features"].update({
            "chemical_object_distribution": "不同类多个化学对象",
            "interference_type": "易混概念",
        })
        simple_result = schema.postprocess_chemistry_difficulty(simple, {
            "stem": "某同学制作的试剂标签中化学式书写正确的是（ ）。",
            "options": "氯化钠NaCl；水H2O；氧气O2；二氧化碳CO2",
        })
        self.assertEqual(simple_result["difficulty_level"], "送分题")

    def test_element_deficiency_comparison_and_example_writeback_have_foundation_floor(self):
        deficiency = rating("送分题", ["U11_T01"])
        deficiency["features"]["chemical_object_distribution"] = "同类多个化学对象"
        deficiency_result = schema.postprocess_chemistry_difficulty(deficiency, {
            "stem": "青少年缺乏下列哪种元素会得佝偻病？",
            "options": "A.铁 B.锌 C.钙 D.氟",
        })
        self.assertEqual(deficiency_result["difficulty_level"], "基础题")

        example = rating("送分题", ["U01_T01"])
        example_result = schema.postprocess_chemistry_difficulty(example, {
            "stem": "下列过程中主要发生化学变化的是____。A.粮食酿酒 B.海水蒸发 C.苹果榨汁 D.______",
        })
        self.assertEqual(example_result["difficulty_level"], "基础题")

    def test_question_image_evidence_preserves_life_sign_features(self):
        value = rating("送分题", ["U11_T03"])
        value["features"].update({
            "visual_content": "无图片信息",
            "visual_item_count": "4幅及以上",
            "visual_complexity": "多个同类型图像",
        })
        result = schema.postprocess_chemistry_difficulty(
            value,
            {
                "stem": "下列常见标识错误的是（ ）<image>",
                "options": "节能标志；节水标志；禁止带火种；有毒品",
                "image_input_used": True,
            },
        )
        self.assertEqual(result["features"]["visual_content"], "生活标识图")
        self.assertEqual(result["features"]["visual_item_count"], "4幅及以上")
        self.assertEqual(result["difficulty_level"], "送分题")
        self.assertFalse(result["postprocess"]["writeback_applied"])

    def test_every_model_feature_finishes_in_its_own_enum(self):
        value = rating("基础题")
        for field in schema.FEATURE_OPTIONS:
            value["features"][field] = "__非法自由文本__"
        result = schema.postprocess_chemistry_difficulty(value, {})
        for field, options in schema.FEATURE_OPTIONS.items():
            self.assertIn(result["features"][field], options, field)
        self.assertIn(result["difficulty_level"], schema.LEVELS)
        self.assertGreaterEqual(
            len(result["postprocess"]["feature_normalization_actions"]),
            len(schema.FEATURE_OPTIONS),
        )

    def test_open_semantic_fields_have_legal_nonempty_residuals(self):
        for field, residual in schema.FEATURE_RESIDUAL_OPTIONS.items():
            self.assertIn(field, schema.FEATURE_OPTIONS)
            self.assertIn(residual, schema.FEATURE_OPTIONS[field])
            self.assertNotEqual(residual, schema.FEATURE_DEFAULTS[field])
        for field, combined in schema.FEATURE_MULTI_OPTIONS.items():
            self.assertIn(field, schema.FEATURE_OPTIONS)
            self.assertIn(combined, schema.FEATURE_OPTIONS[field])
        for field, aliases in schema.FEATURE_ALIASES.items():
            for target in aliases.values():
                self.assertIn(target, schema.FEATURE_OPTIONS[field])

    def test_unknown_open_semantic_value_keeps_positive_evidence(self):
        source = rating("基础题")["features"]
        for field, residual in schema.FEATURE_RESIDUAL_OPTIONS.items():
            normalized, reason = schema.canonicalize_feature_value(
                field, "明确存在但目录未列出的任务", source
            )
            self.assertEqual(normalized, residual, field)
            self.assertIn("不回落为无", reason)

    def test_multiple_values_converge_to_joint_enum_instead_of_none(self):
        value = rating("中等题")
        value["features"]["experiment_operation"] = "仪器识别或名称;装置选择或连接"
        value["features"]["expression_type"] = "化学方程式书写;计算过程书写"
        value["features"]["subjective_response"] = "有"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(
            result["features"]["experiment_operation"], "多项实验操作联合"
        )
        self.assertEqual(
            result["features"]["expression_type"], "多类规范表达联合"
        )

    def test_calculation_structure_is_derived_when_model_crosses_fields(self):
        value = rating("中等题")
        value["features"].update({
            "calculation_type": "相对分子质量或元素质量计算",
            "calculation_steps": "2-3步",
            "calculation_structure": "相对分子质量和元素质量计算",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(
            result["features"]["calculation_structure"], "多步常规计算"
        )

    def test_legal_but_incompatible_calculation_structure_is_corrected(self):
        value = rating("中等题")
        value["features"].update({
            "calculation_type": "相对分子质量或元素质量计算",
            "calculation_steps": "2-3步",
            "calculation_structure": "含杂质多步质量分数",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(
            result["features"]["calculation_structure"], "多步常规计算"
        )

    def test_short_numeric_answer_is_not_forced_to_reason_explanation(self):
        value = rating("基础题")
        value["features"]["expression_type"] = "数值或简短答案填写"
        value["features"]["subjective_response"] = "有"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(
            result["features"]["expression_type"], "数值或简短答案填写"
        )
        self.assertEqual(result["features"]["subjective_response"], "有")

    def test_independent_task_count_does_not_inflate_solution_steps(self):
        value = rating("中等题", ["U07_T01", "U02_T03", "U10_T01", "U10_T02"])
        value["features"]["task_count"] = "4项及以上"
        value["features"]["step_count"] = "1步"
        value["features"]["task_relation"] = "多项独立"
        result = schema.validate_rating_contract(value)
        self.assertEqual(result["features"]["task_count"], "4项及以上")
        self.assertEqual(result["features"]["step_count"], "1步")

    def test_cross_unit_and_task_count_do_not_suggest_medium_by_themselves(self):
        value = rating("基础题", ["U07_T01", "U02_T03", "U10_T01", "U10_T02"])
        value["features"]["task_count"] = "4项及以上"
        value["features"]["task_relation"] = "多项独立"
        result = schema.postprocess_chemistry_difficulty(value, {})
        candidate_codes = {
            candidate["code"] for candidate in result["postprocess"]["candidates"]
        }
        self.assertNotIn("K3_multi_unit_composite", candidate_codes)
        self.assertNotIn("B2_basic_multiple_tasks", candidate_codes)
        self.assertEqual(result["difficulty_level"], "基础题")

    def test_basic_review_uses_multiple_teacher_factors_without_step_veto(self):
        value = rating("基础题", ["U02_T03", "U07_T01", "U10_T01"])
        value["features"].update({
            "task_count": "4项及以上",
            "task_relation": "多项独立",
            "step_count": "1步",
            "solution_method": "多条规则分别判断",
            "visual_content": "仪器图",
            "visual_item_count": "4幅及以上",
            "visual_complexity": "多个同类型图像",
            "information_operation": "比较或整理多条信息",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        action = result["postprocess_actions"][-1]
        self.assertEqual(
            action["rule"],
            "M1_basic_to_medium_teacher_factor_review",
        )

    def test_basic_review_does_not_promote_counts_or_cross_unit_alone(self):
        value = rating("基础题", ["U02_T03", "U07_T01", "U10_T01"])
        value["features"].update({
            "task_count": "4项及以上",
            "task_relation": "多项独立",
            "step_count": "1步",
            "solution_method": "一条规则直接应用",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertFalse(result["postprocess"]["writeback_applied"])

    def test_decisive_special_method_is_not_blocked_by_four_step_threshold(self):
        value = rating("中等题", ["U05_T03", "U10_T03"])
        value["features"]["calculation_type"] = "含杂质计算"
        value["features"]["calculation_steps"] = "2-3步"
        value["features"]["special_method"] = "极值法"
        result = schema.postprocess_chemistry_difficulty(value, {})
        candidates = {
            candidate["rule"]: candidate
            for candidate in result["postprocess"]["candidates"]
        }
        self.assertIn("H1_medium_decisive_task", candidates)
        self.assertIn("不得仅因不足4步", candidates["H1_medium_decisive_task"]["reason"])
        self.assertEqual(result["difficulty_level"], "中等题")

    def test_medium_local_difficulty_review_does_not_require_four_steps(self):
        reverse_case = rating("中等题")
        reverse_case["features"].update({
            "step_count": "2-3步",
            "reverse_tracing": "有",
            "hidden_condition_count": "1个",
            "hidden_condition_type": "反应先后",
            "condition_relation": "单一条件",
        })
        result = schema.postprocess_chemistry_difficulty(reverse_case, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(
            result["postprocess_actions"][-1]["rule"],
            "R1_medium_to_hard_calibrated_local_difficulty",
        )

        reaction_case = rating("中等题")
        reaction_case["features"].update({
            "step_count": "2-3步",
            "reaction_count": "2-3个",
            "reaction_relation": "反应先后或过量不足",
            "hidden_condition_count": "1个",
            "hidden_condition_type": "过量或不足",
            "condition_relation": "单一条件",
        })
        result = schema.postprocess_chemistry_difficulty(reaction_case, {})
        self.assertEqual(result["difficulty_level"], "拔高题")

    def test_medium_review_requires_anchor_and_multiple_supporting_features(self):
        value = rating("中等题", ["U05_T03", "U10_T03"])
        value["features"]["task_count"] = "2-3项"
        value["features"]["step_count"] = "4-5步"
        value["features"]["task_relation"] = "前后依赖"
        value["features"]["solution_method"] = "定性与定量联合"
        value["features"]["information_operation"] = "多来源信息筛选联合"
        value["features"]["calculation_type"] = "多类计算综合"
        value["features"]["calculation_steps"] = "4步及以上"
        value["features"]["special_method"] = "元素守恒"
        value["features"]["condition_relation"] = "多个关联条件"
        value["features"]["hidden_condition_count"] = "2个"
        value["features"]["hidden_condition_type"] = "多类条件联合"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        action = result["postprocess_actions"][-1]
        self.assertEqual(action["rule"], "R1_medium_to_hard_multi_feature_review")
        self.assertGreaterEqual(action["evidence"]["review_score"], 5)

    def test_hard_review_requires_multiple_types_of_final_level_evidence(self):
        value = rating("拔高题", ["U05_T03", "U10_T03", "U09_T05"])
        value["features"]["task_count"] = "4项及以上"
        value["features"]["step_count"] = "6步及以上"
        value["features"]["task_relation"] = "前后依赖"
        value["features"]["solution_method"] = "定性与定量联合"
        value["features"]["information_operation"] = "多来源信息筛选联合"
        value["features"]["calculation_type"] = "多类计算综合"
        value["features"]["calculation_steps"] = "4步及以上"
        value["features"]["reaction_count"] = "2-3个"
        value["features"]["reaction_relation"] = "多个反应连续"
        value["features"]["special_method"] = "极值法"
        value["features"]["hidden_condition_count"] = "2个"
        value["features"]["hidden_condition_type"] = "多类条件联合"
        value["features"]["condition_relation"] = "多个关联条件"
        value["features"]["given_information"] = "题干给出多条新事实或规则"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "压轴题")
        action = result["postprocess_actions"][-1]
        self.assertEqual(action["rule"], "R2_hard_to_final_multi_feature_review")
        self.assertIn(
            "多来源信息、多反应计算与至少两个隐藏条件联合",
            action["evidence"]["matched_paths"],
        )

    def test_hard_review_accepts_calibrated_pressure_paths(self):
        cases = []

        classification = rating("拔高题")
        classification["features"].update({
            "classification_discussion": "多情况分类讨论",
            "reverse_tracing": "有",
            "calculation_type": "多类计算综合",
            "calculation_steps": "4步及以上",
            "hidden_condition_count": "2个",
            "hidden_condition_type": "过量或不足",
            "condition_relation": "多个关联条件",
        })
        cases.append(classification)

        continuous_reactions = rating("拔高题")
        continuous_reactions["features"].update({
            "reaction_count": "4个及以上",
            "reaction_relation": "多个反应连续",
            "experiment_analysis": "多个实验分析任务联合",
            "error_analysis": "装置或方案导致误差",
            "calculation_type": "多类计算综合",
            "calculation_steps": "4步及以上",
            "special_method": "差量法",
        })
        cases.append(continuous_reactions)

        for value in cases:
            with self.subTest(features=value["features"]):
                result = schema.postprocess_chemistry_difficulty(value, {})
                self.assertEqual(result["difficulty_level"], "压轴题")
                self.assertEqual(
                    result["postprocess_actions"][-1]["rule"],
                    "R2_hard_to_final_multi_feature_review",
                )

    def test_hard_review_uses_concrete_tasks_without_low_count_vetoes(self):
        staged_graph = rating("拔高题")
        staged_graph["features"].update({
            "step_count": "4-5步",
            "information_operation": "图像拐点或分段分析",
            "reaction_count": "2-3个",
            "reaction_relation": "多个反应连续",
            "experiment_analysis": "多个实验分析任务联合",
            "error_analysis": "装置或方案导致误差",
            "calculation_type": "多类计算综合",
            "calculation_steps": "4步及以上",
            "calculation_structure": "多个化学反应计算",
            "hidden_condition_count": "2个",
            "hidden_condition_type": "图像拐点或分段",
            "condition_relation": "多个关联条件",
        })

        shared_composition = rating("拔高题", ["U05_T03", "U08_T02", "U10_T02"])
        shared_composition["features"].update({
            "chemical_object_distribution": "不同类多个化学对象",
            "step_count": "4-5步",
            "solution_method": "定性与定量联合",
            "reaction_count": "2-3个",
            "reaction_relation": "多个反应连续",
            "calculation_type": "多类计算综合",
            "calculation_steps": "4步及以上",
            "calculation_structure": "多个化学反应计算",
            "special_method": "元素守恒",
            "hidden_condition_count": "2个",
            "hidden_condition_type": "物质或溶液状态",
            "condition_relation": "多个关联条件",
            "interference_type": "体系质量关系易错",
        })

        qualitative_remainder = rating("拔高题", ["U08_T04", "U10_T06"])
        qualitative_remainder["features"].update({
            "task_count": "4项及以上",
            "task_relation": "前后依赖",
            "chemical_object_distribution": "不同类多个化学对象",
            "step_count": "4-5步",
            "solution_method": "连续推导",
            "classification_discussion": "单一情况讨论",
            "reverse_tracing": "有",
            "reaction_count": "2-3个",
            "reaction_relation": "多个反应连续",
            "experiment_analysis": "多个实验分析任务联合",
            "hidden_condition_count": "2个",
            "hidden_condition_type": "多类条件联合",
            "condition_relation": "多个关联条件",
            "interference_type": "多种剩余情况或竞争解释",
        })

        expected_paths = (
            "分阶段图像或多来源数据中判断2-3个连续反应并完成关联计算",
            "同一复杂体系中用2-3个反应和守恒反推组成或纯度",
        )
        for value, expected_path in zip(
            (staged_graph, shared_composition),
            expected_paths,
        ):
            with self.subTest(expected_path=expected_path):
                result = schema.postprocess_chemistry_difficulty(value, {})
                self.assertEqual(result["difficulty_level"], "压轴题")
                action = result["postprocess_actions"][-1]
                self.assertIn(expected_path, action["evidence"]["matched_paths"])
                self.assertIn("未被当作单项否决条件", action["reason"])

        qualitative_result = schema.postprocess_chemistry_difficulty(
            qualitative_remainder, {}
        )
        self.assertEqual(qualitative_result["difficulty_level"], "拔高题")
        self.assertEqual(qualitative_result["postprocess_actions"], [])

    def test_hard_review_rejects_broad_feature_accumulation(self):
        classification_without_reverse = rating("拔高题")
        classification_without_reverse["features"].update({
            "classification_discussion": "单一情况讨论",
            "calculation_type": "多类计算综合",
            "calculation_steps": "4步及以上",
            "calculation_structure": "多个化学反应计算",
            "special_method": "极值法",
            "hidden_condition_count": "2个",
            "hidden_condition_type": "过量或不足",
            "condition_relation": "多个关联条件",
        })
        broad_multi_source = rating("拔高题")
        broad_multi_source["features"].update({
            "task_count": "4项及以上",
            "task_relation": "前后依赖",
            "step_count": "4-5步",
            "information_operation": "多来源信息筛选联合",
            "reaction_count": "4个及以上",
            "reaction_relation": "多个反应连续",
            "calculation_type": "多类计算综合",
            "calculation_steps": "4步及以上",
            "calculation_structure": "多个化学反应计算",
            "hidden_condition_count": "2个",
            "hidden_condition_type": "反应条件",
            "condition_relation": "多个关联条件",
            "interference_type": "体系质量关系易错",
        })
        staged_without_validation = rating("拔高题")
        staged_without_validation["features"].update({
            "task_count": "4项及以上",
            "task_relation": "前后依赖",
            "step_count": "4-5步",
            "information_operation": "多来源信息筛选联合",
            "reaction_count": "2-3个",
            "reaction_relation": "多个反应连续",
            "calculation_type": "含杂质计算",
            "calculation_steps": "4步及以上",
            "calculation_structure": "含杂质多步质量分数",
            "special_method": "元素守恒",
            "hidden_condition_count": "2个",
            "hidden_condition_type": "过量或不足",
            "condition_relation": "多个关联条件",
            "interference_type": "体系质量关系易错",
        })
        for value in (
            classification_without_reverse,
            broad_multi_source,
            staged_without_validation,
        ):
            with self.subTest(features=value["features"]):
                result = schema.postprocess_chemistry_difficulty(value, {})
                self.assertEqual(result["difficulty_level"], "拔高题")
                self.assertEqual(result["postprocess_actions"], [])

    def test_hard_review_accepts_single_chain_and_independent_final_paths(self):
        multi_source = rating("拔高题")
        multi_source["features"].update({
            "task_count": "4项及以上",
            "task_relation": "多项独立",
            "step_count": "4-5步",
            "information_operation": "多来源信息筛选联合",
            "reaction_count": "4个及以上",
            "reaction_relation": "多个反应连续",
            "hidden_condition_count": "2个",
            "hidden_condition_type": "多类条件联合",
            "condition_relation": "多个关联条件",
            "given_information": "题干给出多条新事实或规则",
        })
        result = schema.postprocess_chemistry_difficulty(multi_source, {})
        self.assertEqual(result["difficulty_level"], "压轴题")

        error_case = rating("拔高题")
        error_case["features"].update({
            "reaction_count": "4个及以上",
            "reaction_relation": "多个反应连续",
            "error_analysis": "装置或方案导致误差",
        })
        result = schema.postprocess_chemistry_difficulty(error_case, {})
        self.assertEqual(result["difficulty_level"], "压轴题")

    def test_hard_review_does_not_promote_independent_comprehensive_tasks(self):
        value = rating("拔高题")
        value["features"].update({
            "task_count": "4项及以上",
            "task_relation": "多项独立",
            "information_operation": "多来源信息筛选联合",
            "calculation_type": "多类计算综合",
            "calculation_steps": "4步及以上",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertFalse(result["postprocess"]["writeback_applied"])

    def test_hard_review_does_not_promote_one_special_method(self):
        value = rating("拔高题")
        value["features"].update({
            "calculation_type": "含杂质计算",
            "calculation_steps": "4步及以上",
            "special_method": "极值法",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertFalse(result["postprocess"]["writeback_applied"])

    def test_upper_level_review_never_cascades_two_levels(self):
        value = rating("中等题", ["U05_T03", "U10_T03", "U09_T05"])
        value["features"]["task_count"] = "4项及以上"
        value["features"]["step_count"] = "6步及以上"
        value["features"]["task_relation"] = "多条任务链汇合"
        value["features"]["solution_method"] = "定性与定量联合"
        value["features"]["information_operation"] = "图像拐点或分段分析"
        value["features"]["calculation_type"] = "多类计算综合"
        value["features"]["calculation_steps"] = "4步及以上"
        value["features"]["special_method"] = "多种特殊方法联合"
        value["features"]["hidden_condition_count"] = "3个及以上"
        value["features"]["hidden_condition_type"] = "多类条件联合"
        value["features"]["condition_relation"] = "多层嵌套条件"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(len(result["postprocess_actions"]), 1)

    def test_out_of_scope_requires_specific_content(self):
        value = rating(topic_ids=[])
        value["features"]["curriculum_scope"] = {
            "scope": "out_of_scope", "extra_points": ["物质的量"],
        }
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["curriculum_scope"]["scope"], "out_of_scope")

    def test_tool_schema_forces_every_core_enum(self):
        tool = schema.rating_tool_definition()
        self.assertTrue(tool["strict"])
        self.assertEqual(tool["name"], schema.TOOL_NAME)
        parameters = tool["parameters"]
        self.assertFalse(parameters["additionalProperties"])
        feature_schema = parameters["properties"]["features"]
        self.assertFalse(feature_schema["additionalProperties"])
        for field, options in schema.MODEL_FEATURE_OPTIONS.items():
            self.assertEqual(feature_schema["properties"][field]["enum"], list(options))
        self.assertIn("2幅", schema.FEATURE_OPTIONS["visual_item_count"])
        self.assertIn("3幅", schema.FEATURE_OPTIONS["visual_item_count"])
        self.assertNotIn("2-3幅", schema.FEATURE_OPTIONS["visual_item_count"])

    def test_prompt_documents_every_runtime_enum(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        self.assertIn("30个细粒度特征", prompt)
        self.assertEqual(prompt.count("## 输入题目信息"), 1)
        self.assertGreaterEqual(prompt.count("【Case"), 25)
        for field, options in schema.MODEL_FEATURE_OPTIONS.items():
            self.assertIn(field, prompt)
            for value in options:
                self.assertIn(value, prompt)
        for value in (*schema.SCOPES, *schema.LEVELS):
            self.assertIn(value, prompt)
        documented = {
            field: tuple(re.findall(r"`([^`]+)`", values))
            for field, values in re.findall(
                r"#### \d+\. ([a-z_]+)\s*\n只能是：([^\n]+)", prompt
            )
        }
        self.assertEqual(set(documented), set(schema.MODEL_FEATURE_OPTIONS))
        for field, options in schema.MODEL_FEATURE_OPTIONS.items():
            self.assertEqual(documented[field], options, field)
        self.assertIn("task_count只描述工作量，不能单独决定档位", prompt)
        self.assertIn("不能因`跨单元不同知识点`或`多单元综合`自动升档", prompt)
        self.assertIn("普通选择题中的错误选项不算干扰", prompt)
        self.assertIn("题干字数、选项数、小问数、图片数只用于审计", prompt)
        self.assertIn("多个高难环节共同决定答案", prompt)
        self.assertIn("4-5步", prompt)
        self.assertIn("不能单独否决压轴题", prompt)
        self.assertIn("【Case 23：氧分子与含氧物质】", prompt)
        self.assertIn("【Case 52：石青分阶段受热】", prompt)
        self.assertIn("【Case 57：锌与两种盐溶液反应后的滤液滤渣】", prompt)
        self.assertIn("【Case 92：单一高难实验或计算 vs 多类型高难任务耦合】", prompt)
        self.assertIn("【Case 47：装置作用、顺序、改进与计算联合】", prompt)
        self.assertNotIn("【Case 93", prompt)
        case_numbers = [int(value) for value in re.findall(r"【Case (\d+)", prompt)]
        self.assertEqual(case_numbers, list(range(1, 93)))
        self.assertNotIn("同档锚点", prompt)
        self.assertNotIn("补充边界", prompt)
        self.assertIn("【Case 17：利用化合价代数和求化合价或化学式】", prompt)
        self.assertIn("【Case 59：常见化学式再认 vs 化合价关系应用】", prompt)
        self.assertIn("禁止借用相邻字段的枚举", prompt)
        self.assertIn("必须按指定的严格JSON Schema输出一个完整对象", prompt)
        self.assertNotIn("必须通过指定函数提交", prompt)
        self.assertIn("calculation_structure", prompt)

    def test_prompt_json_example_follows_runtime_contract(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        start = prompt.index('\n{\n  "features"') + 1
        decoder = json.JSONDecoder()
        example, _ = decoder.raw_decode(prompt[start:])
        schema.validate_rating_contract(example)

    def test_runtime_uses_prompt_json_with_local_strict_schema_and_no_retry(self):
        runtime = (ROOT / "src" / "chemistry_difficulty_rating_with_cache.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"structured_output_mode": "prompt_json_local_strict_schema"', runtime)
        self.assertNotIn('"text": {"format":', runtime)
        self.assertNotIn('"tools": [junior_schema.rating_tool_definition()]', runtime)
        self.assertNotIn('"parallel_tool_calls": False', runtime)
        self.assertIn("for _single_http_attempt in range(1)", runtime)
        self.assertNotIn("--retries", runtime)
        self.assertNotIn("args.retries", runtime)
        self.assertNotIn("--no-cache", runtime)
        self.assertNotIn("args.no_cache", runtime)
        self.assertIn("USE_CACHE = True", runtime)
        self.assertIn("正在用第一道真实题验证前缀缓存与JSON输出兼容性", runtime)
        self.assertIn("for q in to_process[1:]", runtime)
        self.assertIn('return "model_validation_error"', runtime)
        self.assertIn('preflight_status == "request_error"', runtime)

    def test_runtime_supports_isolated_prompt_cache_files_for_ab_runs(self):
        runtime = (ROOT / "src" / "chemistry_difficulty_rating_with_cache.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"CHEMISTRY_CACHE_FILE_PATH"', runtime)
        self.assertIn('"chemistry_prompt_cache.json"', runtime)
        self.assertIn("第一题未通过本地Schema", runtime)
        self.assertIn('output_data["feature_normalization_actions"]', runtime)
        self.assertNotIn("MAX_SCHEMA_RETRIES", runtime)
        self.assertNotIn("repair_feedback", runtime)
        self.assertIn("junior_schema.parse_model_json_text", runtime)
        self.assertNotIn("json_repair", runtime)
        self.assertIn('"structured_output_json_complete": False', runtime)
        self.assertIn('"token_anomaly_flags": []', runtime)
        self.assertIn("completion_tokens_abnormally_high", runtime)

    def test_junior_runtime_has_no_legacy_or_source_label_logic(self):
        paths = [
            ROOT / "src" / "chemistry_difficulty_rating_with_cache.py",
            ROOT / "src" / "junior_chemistry_schema.py",
        ]
        source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
        for forbidden in (
            "UNTRUSTED" + "_LABEL_FIELDS",
            "source_difficulty" + "_untrusted",
            "reasoning" + "_direction",
            "knowledge" + "_relation",
            "representation" + "_conversion",
            "unfamiliar_information" + "_transfer",
            "CORE" + "12",
            "ENUM" + "_ALIASES",
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
