import json
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

    def test_no_alias_or_free_generated_feature_value_is_accepted(self):
        value = rating()
        value["features"]["experiment_operation"] = "仪器识别"
        with self.assertRaises(schema.ChemistrySchemaError):
            schema.validate_rating_contract(value)

        value = rating()
        value["features"]["calculation_type"] = "反应后气体质量"
        with self.assertRaises(schema.ChemistrySchemaError):
            schema.validate_rating_contract(value)

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

    def test_decisive_special_method_is_not_blocked_by_four_step_threshold(self):
        value = rating("中等题", ["U05_T03", "U10_T03"])
        value["features"]["calculation_type"] = "含杂质计算"
        value["features"]["calculation_steps"] = "2-3步"
        value["features"]["calculation_structure"] = "含杂质多步质量分数"
        value["features"]["special_method"] = "极值法"
        result = schema.postprocess_chemistry_difficulty(value, {})
        candidates = {
            candidate["rule"]: candidate
            for candidate in result["postprocess"]["candidates"]
        }
        self.assertIn("H1_medium_decisive_task", candidates)
        self.assertIn("不得仅因不足4步", candidates["H1_medium_decisive_task"]["reason"])
        self.assertEqual(result["difficulty_level"], "中等题")

    def test_medium_review_requires_anchor_and_multiple_supporting_features(self):
        value = rating("中等题", ["U05_T03", "U10_T03"])
        value["features"]["task_count"] = "2-3项"
        value["features"]["step_count"] = "4-5步"
        value["features"]["task_relation"] = "前后依赖"
        value["features"]["solution_method"] = "定性与定量联合"
        value["features"]["information_operation"] = "多来源信息筛选联合"
        value["features"]["calculation_type"] = "多类计算综合"
        value["features"]["calculation_steps"] = "4步及以上"
        value["features"]["calculation_structure"] = "多模型综合计算"
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
        value["features"]["information_operation"] = "图像拐点或分段分析"
        value["features"]["calculation_type"] = "多类计算综合"
        value["features"]["calculation_steps"] = "4步及以上"
        value["features"]["calculation_structure"] = "多个化学反应计算"
        value["features"]["special_method"] = "极值法"
        value["features"]["hidden_condition_count"] = "2个"
        value["features"]["hidden_condition_type"] = "多类条件联合"
        value["features"]["condition_relation"] = "多个关联条件"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "压轴题")
        action = result["postprocess_actions"][-1]
        self.assertEqual(action["rule"], "R2_hard_to_final_multi_feature_review")
        self.assertGreaterEqual(action["evidence"]["review_score"], 7)

    def test_upper_level_review_never_cascades_two_levels(self):
        value = rating("中等题", ["U05_T03", "U10_T03", "U09_T05"])
        value["features"]["task_count"] = "4项及以上"
        value["features"]["step_count"] = "6步及以上"
        value["features"]["task_relation"] = "多条任务链汇合"
        value["features"]["solution_method"] = "定性与定量联合"
        value["features"]["information_operation"] = "图像拐点或分段分析"
        value["features"]["calculation_type"] = "多类计算综合"
        value["features"]["calculation_steps"] = "4步及以上"
        value["features"]["calculation_structure"] = "多模型综合计算"
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
        for field, options in schema.FEATURE_OPTIONS.items():
            self.assertEqual(feature_schema["properties"][field]["enum"], list(options))

    def test_prompt_documents_every_runtime_enum(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        self.assertIn("30个细粒度特征", prompt)
        self.assertEqual(prompt.count("## 输入题目信息"), 1)
        self.assertGreaterEqual(prompt.count("【Case"), 25)
        for field, options in schema.FEATURE_OPTIONS.items():
            self.assertIn(field, prompt)
            for value in options:
                self.assertIn(value, prompt)
        for value in (*schema.SCOPES, *schema.LEVELS):
            self.assertIn(value, prompt)
        self.assertIn("task_count只描述工作量，不能单独决定档位", prompt)
        self.assertIn("不能因`跨单元不同知识点`或`多单元综合`自动升档", prompt)
        self.assertIn("普通选择题中的错误选项不算干扰", prompt)
        self.assertIn("题干字数、选项数、小问数、图片数只用于审计", prompt)

    def test_prompt_json_example_follows_runtime_contract(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        start = prompt.index('\n{\n  "features"') + 1
        decoder = json.JSONDecoder()
        example, _ = decoder.raw_decode(prompt[start:])
        schema.validate_rating_contract(example)

    def test_runtime_forces_function_call_and_keeps_no_retry(self):
        runtime = (ROOT / "src" / "chemistry_difficulty_rating_with_cache.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('"tools": [junior_schema.rating_tool_definition()]', runtime)
        self.assertIn('"tool_choice": {', runtime)
        self.assertIn('"parallel_tool_calls": False', runtime)
        self.assertIn("for retry in range(1)", runtime)
        self.assertNotIn("MAX_SCHEMA_RETRIES", runtime)
        self.assertNotIn("repair_feedback", runtime)

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
