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
            "knowledge_coverage": "单一知识点",
            "chemical_object_distribution": "单一化学对象",
            "task_structure": "单一任务",
            "step_count": "1步",
            "solution_method": "一条规则直接应用",
            "information_carrier": "无额外信息",
            "information_operation": "无需额外提取",
            "reaction_structure": "无反应",
            "experiment_operation": "无",
            "experiment_analysis": "无",
            "experiment_design": "无",
            "error_analysis": "无",
            "calculation_type": "无",
            "calculation_structure": "无计算",
            "special_method": "无",
            "condition_structure": "无隐藏条件",
            "interference_type": "无",
            "expression_type": "无",
            "curriculum_scope": {"scope": "within_junior", "extra_points": []},
        },
        "reasoning": {
            "knowledge_points": "涉及一个课内知识点。",
            "solution_process": ["应用一条规则完成判断"],
            "main_difficulty_factors": "无额外复杂任务。",
            "level_basis": "按教师边界判为基础题。",
        },
        "difficulty_level": level,
    }


class JuniorChemistrySchemaTests(unittest.TestCase):
    def test_curriculum_map_is_loaded(self):
        topics = schema.load_curriculum_topics()
        self.assertGreaterEqual(len(topics), 58)
        self.assertEqual(topics["U10_T02"]["unit_id"], "U10")

    def test_exactly_nineteen_explicit_core_features(self):
        self.assertEqual(len(schema.FEATURE_OPTIONS), 18)
        value = rating()
        self.assertEqual(
            set(value["features"]),
            {"knowledge", *schema.FEATURE_OPTIONS, "curriculum_scope"},
        )
        self.assertEqual(1 + len(schema.FEATURE_OPTIONS), 19)

    def test_removed_duplicate_fields_are_not_in_schema(self):
        removed = {
            "task_count", "knowledge_distribution", "task_relation",
            "classification_discussion", "reverse_tracing", "visual_content",
            "visual_item_count", "visual_complexity", "reaction_count",
            "reaction_relation", "calculation_steps", "hidden_condition_count",
            "hidden_condition_type", "condition_relation", "subjective_response",
            "given_information", "cross_subject",
        }
        self.assertTrue(removed.isdisjoint(schema.FEATURE_OPTIONS))
        properties = schema.rating_json_schema()["properties"]["features"]["properties"]
        self.assertTrue(removed.isdisjoint(properties))

    def test_schema_requires_every_enum_and_solution_process_list(self):
        validated = schema.validate_rating_contract(rating())
        self.assertIsInstance(validated["reasoning"]["solution_process"], list)
        bad = rating()
        bad["reasoning"]["solution_process"] = "一步判断"
        with self.assertRaises(schema.ChemistrySchemaError):
            schema.validate_rating_contract(bad)

    def test_every_feature_value_is_its_own_enum(self):
        result = schema.postprocess_chemistry_difficulty(rating(), {})
        for field, options in schema.FEATURE_OPTIONS.items():
            self.assertIn(result["features"][field], options)

    def test_cross_field_calculation_value_is_repaired_without_retry(self):
        value = rating()
        value["features"]["calculation_type"] = "多个化学反应计算"
        value["features"]["calculation_structure"] = "多个反应连续计算"
        value["features"]["special_method"] = "元素守恒"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["calculation_type"], "化学方程式计算")
        self.assertEqual(result["features"]["calculation_structure"], "多个反应连续计算")

    def test_duplicate_topic_ids_are_deduplicated(self):
        result = schema.postprocess_chemistry_difficulty(
            rating("中等题", ["U02_T03", "U02_T03", "U10_T01"]), {}
        )
        knowledge = result["features"]["knowledge"]
        self.assertEqual(knowledge["topic_ids"], ["U02_T03", "U10_T01"])
        self.assertEqual(knowledge["knowledge_point_count"], 2)
        self.assertEqual(knowledge["unit_count"], 2)
        self.assertTrue(knowledge["cross_unit"])

    def test_error_analysis_has_medium_floor(self):
        value = rating("送分题")
        value["features"]["error_analysis"] = "量筒读数误差"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(result["postprocess_actions"][0]["rule"], "T1_error_analysis_floor")

    def test_cross_unit_alone_does_not_promote_foundation(self):
        value = rating("基础题", ["U02_T03", "U10_T01"])
        value["features"]["knowledge_coverage"] = "跨两个单元"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "基础题")

    def test_two_independent_medium_signals_can_promote_foundation(self):
        value = rating("基础题", ["U02_T03", "U10_T01"])
        value["features"]["knowledge_coverage"] = "跨两个单元"
        value["features"]["information_operation"] = "比较或整理多条信息"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "中等题")

    def test_one_hard_signal_only_triggers_review(self):
        value = rating("中等题")
        value["features"]["special_method"] = "极值法"
        value["features"]["calculation_type"] = "含杂质计算"
        value["features"]["calculation_structure"] = "一步或口算"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertFalse(result["postprocess"]["candidates"][-1]["writeback_allowed"])

    def test_two_independent_hard_signals_can_promote_medium(self):
        value = rating("中等题")
        value["features"]["special_method"] = "极值法"
        value["features"]["calculation_type"] = "含杂质计算"
        value["features"]["calculation_structure"] = "含杂质或反应后体系计算"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "拔高题")

    def test_hard_to_final_requires_three_signal_groups(self):
        value = rating("拔高题")
        value["features"].update({
            "task_structure": "前后依赖任务",
            "step_count": "4-5步",
            "reaction_structure": "反应先后或过量不足",
            "calculation_type": "含杂质计算",
            "calculation_structure": "含杂质或反应后体系计算",
            "special_method": "元素守恒",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "压轴题")

    def test_prompt_catalog_and_example_follow_runtime_contract(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        schema.validate_prompt_feature_catalog(prompt)
        self.assertEqual(prompt.count("## 输入题目信息"), 1)
        start = prompt.index('\n{\n  "features"') + 1
        example, _ = json.JSONDecoder().raw_decode(prompt[start:])
        schema.validate_rating_contract(example)

    def test_runtime_has_no_source_label_or_retry_logic(self):
        runtime = (ROOT / "src" / "chemistry_difficulty_rating_with_cache.py").read_text(encoding="utf-8")
        source = runtime + (ROOT / "src" / "junior_chemistry_schema.py").read_text(encoding="utf-8")
        self.assertNotIn("UNTRUSTED" + "_LABEL_FIELDS", source)
        self.assertNotIn("source_difficulty" + "_untrusted", source)
        self.assertIn("for _single_http_attempt in range(1)", runtime)
        self.assertNotIn("--retries", runtime)


if __name__ == "__main__":
    unittest.main()
