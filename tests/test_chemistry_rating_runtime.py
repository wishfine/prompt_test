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
        self.assertEqual(result["postprocess"]["candidates"][0]["candidate_level"], "基础题")

    def test_error_analysis_is_written_back_to_medium_floor(self):
        value = rating("送分题")
        value["features"]["error_analysis"] = "量筒读数误差"
        result = schema.postprocess_chemistry_difficulty(value, {"stem": "判断量筒读数误差。"})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertTrue(result["postprocess"]["writeback_applied"])
        self.assertEqual(result["postprocess_actions"][0]["rule"], "T1_error_analysis_floor")

    def test_question_statistics_and_objective_floors(self):
        long_stem = "化" * 101
        result = schema.postprocess_chemistry_difficulty(
            rating("送分题"), {"stem": long_stem, "options": "A.甲 B.乙 C.丙 D.丁"}
        )
        stats = result["postprocess"]["question_statistics"]
        self.assertEqual(stats["stem_char_count"], 101)
        self.assertEqual(stats["stem_length_band"], "101-300字")
        self.assertEqual(stats["option_count"], 4)
        self.assertEqual(result["difficulty_level"], "中等题")

        result = schema.postprocess_chemistry_difficulty(
            rating("送分题"), {"sub_questions": [{"stem": str(i)} for i in range(5)]}
        )
        self.assertEqual(result["difficulty_level"], "基础题")

        value = rating("送分题")
        value["features"]["visual_content"] = "仪器图"
        value["features"]["visual_item_count"] = "4幅及以上"
        value["features"]["visual_complexity"] = "多个同类型图像"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "基础题")

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
        self.assertGreaterEqual(prompt.count("【Case"), 25)
        for field, options in schema.FEATURE_OPTIONS.items():
            self.assertIn(field, prompt)
            for value in options:
                self.assertIn(value, prompt)
        for value in (*schema.SCOPES, *schema.LEVELS):
            self.assertIn(value, prompt)

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
