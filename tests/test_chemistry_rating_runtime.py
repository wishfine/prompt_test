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
            "task_structure": "单一任务",
            "step_count": "1步",
            "information_complexity": "无额外信息处理",
            "reaction_structure": "无反应",
            "experiment_operation": ["无"],
            "experiment_analysis": ["无"],
            "experiment_design": ["无"],
            "error_analysis": "无",
            "calculation_structure": "无计算",
            "special_method": "无",
            "difficulty_obstacle": "无明显障碍",
            "expression_requirement": ["无"],
            "curriculum_scope": {"scope": "within_junior", "extra_points": []},
        },
        "reasoning": {
            "solution_process": ["应用一条规则完成判断"],
            "level_basis": "按教师边界判为基础题。",
        },
        "difficulty_level": level,
    }


class JuniorChemistrySchemaTests(unittest.TestCase):
    def test_curriculum_map_is_loaded(self):
        topics = schema.load_curriculum_topics()
        self.assertGreaterEqual(len(topics), 58)
        self.assertEqual(topics["U10_T02"]["unit_id"], "U10")

    def test_exactly_thirteen_explicit_core_features(self):
        self.assertEqual(len(schema.FEATURE_OPTIONS), 12)
        value = rating()
        self.assertEqual(
            set(value["features"]),
            {"knowledge", *schema.FEATURE_OPTIONS, "curriculum_scope"},
        )
        self.assertEqual(1 + len(schema.FEATURE_OPTIONS), 13)

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

    def test_task_relation_and_calculation_structure_do_not_repeat_counts(self):
        self.assertEqual(
            schema.FEATURE_OPTIONS["task_structure"],
            ("单一任务", "多个独立任务", "多项任务共享同一模型", "前后依赖任务", "多条任务链汇合"),
        )
        self.assertFalse(any("步" in item for item in schema.FEATURE_OPTIONS["calculation_structure"]))

    def test_experiment_responsibilities_are_separate(self):
        self.assertTrue({
            "experiment_operation", "experiment_analysis",
            "experiment_design", "error_analysis",
        }.issubset(schema.FEATURE_OPTIONS))

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
            value = result["features"][field]
            if field in schema.MULTI_FEATURE_FIELDS:
                self.assertTrue(value)
                self.assertTrue(all(item in options for item in value))
            else:
                self.assertIn(value, options)

    def test_legacy_calculation_fields_merge_without_retry(self):
        value = rating()
        del value["features"]["calculation_structure"]
        value["features"]["calculation_type"] = "多个化学反应计算"
        value["features"]["calculation_complexity"] = "多个反应连续计算"
        value["features"]["special_method"] = "元素守恒"
        result = schema.postprocess_chemistry_difficulty(value, {})
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

    def test_generic_operation_consequence_does_not_force_medium(self):
        value = rating("基础题")
        value["features"]["error_analysis"] = "实验操作导致误差"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertFalse(result["postprocess"]["writeback_applied"])

    def test_cross_unit_alone_does_not_promote_foundation(self):
        value = rating("基础题", ["U02_T03", "U10_T01"])
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(result["features"]["knowledge"]["coverage_type"], "跨两个单元")

    def test_multiple_signals_do_not_promote_foundation(self):
        value = rating("基础题", ["U02_T03", "U10_T01"])
        value["features"]["information_complexity"] = "比较整理多条信息"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "基础题")

    def test_single_adjacent_signal_only_triggers_review(self):
        value = rating("中等题")
        value["features"]["special_method"] = "极值法"
        value["features"]["calculation_structure"] = "直接数值或比例关系"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertFalse(result["postprocess"]["writeback_applied"])
        self.assertFalse(result["postprocess"]["candidates"][-1]["writeback_allowed"])

    def test_two_generic_hard_signal_groups_only_trigger_review(self):
        value = rating("中等题")
        value["features"]["special_method"] = "极值法"
        value["features"]["calculation_structure"] = "含杂质或反应后体系计算"
        value["features"]["information_complexity"] = "图表装置或流程推断"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertFalse(result["postprocess"]["candidates"][-1]["writeback_allowed"])

    def test_medium_review_requires_verified_expression_knowledge_path(self):
        value = rating("基础题", ["U02_T03", "U05_T02", "U06_T03", "U10_T01"])
        value["features"]["expression_requirement"] = ["化学方程式书写"]
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertIn(
            "至少4个知识点且要求书写化学方程式",
            result["postprocess"]["candidates"][-1]["evidence"]["matched_review_paths"],
        )

    def test_reaction_order_path_promotes_medium_to_hard(self):
        value = rating("中等题", ["U08_T02", "U11_T04"])
        value["features"]["reaction_structure"] = "反应先后或过量不足"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertIn(
            "需要处理反应先后或过量不足",
            result["postprocess"]["candidates"][-1]["evidence"]["matched_review_paths"],
        )

    def test_generic_final_evidence_only_triggers_review(self):
        value = rating("拔高题")
        value["features"].update({
            "task_structure": "前后依赖任务",
            "step_count": "6步及以上",
            "information_complexity": "拐点分段或多来源联合",
            "reaction_structure": "反应先后或过量不足",
            "calculation_structure": "多模型综合计算",
            "special_method": "多种特殊方法联合",
            "difficulty_obstacle": "多层嵌套条件或竞争解释",
        })
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertFalse(result["postprocess"]["candidates"][-1]["writeback_allowed"])

    def test_same_calculation_dimension_counts_only_once(self):
        value = rating("中等题")
        value["features"]["calculation_structure"] = "含杂质或反应后体系计算"
        value["features"]["special_method"] = "极值法"
        result = schema.postprocess_chemistry_difficulty(value, {})
        candidate = result["postprocess"]["candidates"][-1]
        self.assertEqual(set(candidate["evidence"]["evidence_groups"]), {"计算"})
        self.assertEqual(result["difficulty_level"], "中等题")

    def test_reaction_and_its_calculation_count_as_one_evidence_group(self):
        value = rating("中等题")
        value["features"]["reaction_structure"] = "2-3个连续反应"
        value["features"]["calculation_structure"] = "多个反应连续计算"
        result = schema.postprocess_chemistry_difficulty(value, {})
        candidate = result["postprocess"]["candidates"][-1]
        self.assertEqual(set(candidate["evidence"]["evidence_groups"]), {"反应与计算"})
        self.assertEqual(result["difficulty_level"], "中等题")

    def test_error_analysis_and_experiment_design_count_as_one_experiment_group(self):
        value = rating("中等题")
        value["features"]["experiment_design"] = ["实验方案评价"]
        value["features"]["error_analysis"] = "定量实验误差分析"
        result = schema.postprocess_chemistry_difficulty(value, {})
        candidate = result["postprocess"]["candidates"][-1]
        self.assertEqual(set(candidate["evidence"]["evidence_groups"]), {"实验"})
        self.assertEqual(result["difficulty_level"], "中等题")

    def test_transparent_one_step_and_direct_reading_do_not_promote_giveaway(self):
        value = rating("送分题")
        value["features"]["information_complexity"] = "直接读取单一信息"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["difficulty_level"], "送分题")

    def test_new_runtime_rejects_legacy_fields(self):
        value = rating()
        value["features"]["calculation_complexity"] = "多个反应连续计算"
        with self.assertRaises(schema.ChemistrySchemaError):
            schema.postprocess_chemistry_difficulty(value, {}, allow_legacy_fields=False)

    def test_multi_value_expression_accepts_multiple_legal_values(self):
        value = rating()
        value["features"]["expression_requirement"] = [
            "化学方程式书写", "计算过程书写",
        ]
        validated = schema.validate_rating_contract(value)
        self.assertEqual(len(validated["features"]["expression_requirement"]), 2)

    def test_legacy_experiment_task_splits_without_losing_tasks(self):
        value = rating()
        del value["features"]["experiment_operation"]
        del value["features"]["experiment_analysis"]
        value["features"]["experiment_task"] = [
            "装置选择或连接",
            "装置作用或实验目的", "根据现象或数据得出结论",
        ]
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["experiment_operation"], [
            "装置选择或连接",
        ])
        self.assertEqual(result["features"]["experiment_analysis"], [
            "装置作用或实验目的", "根据现象或数据得出结论",
        ])

    def test_legacy_obstacle_fields_keep_the_stronger_signal(self):
        value = rating()
        del value["features"]["difficulty_obstacle"]
        value["features"]["condition_structure"] = "无隐藏条件"
        value["features"]["interference_type"] = "多种剩余情况或竞争解释"
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(
            result["features"]["difficulty_obstacle"],
            "多层嵌套条件或竞争解释",
        )

    def test_missing_scope_is_repaired_without_retry(self):
        value = rating()
        del value["features"]["curriculum_scope"]
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["curriculum_scope"]["scope"], "within_junior")

    def test_parser_repairs_trailing_comma_and_inner_quotes(self):
        value = rating()
        text = json.dumps(value, ensure_ascii=False)
        text = text.replace("应用一条规则完成判断", '判断D选项中"生成物"是否正确')
        text = text[:-1] + ",}"
        parsed, mode = schema.parse_model_json_text(text)
        self.assertEqual(mode, "local_recovered")
        self.assertEqual(parsed["difficulty_level"], "基础题")

    def test_prompt_catalog_and_example_follow_runtime_contract(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        schema.validate_prompt_feature_catalog(prompt)
        self.assertEqual(prompt.count("## 输入题目信息"), 1)
        start = prompt.index('\n{\n  "features"') + 1
        example, _ = json.JSONDecoder().raw_decode(prompt[start:])
        schema.validate_rating_contract(example)

    def test_each_level_uses_the_same_nonoverlapping_fine_grained_dimensions(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        dimensions = (
            "题型与任务关系", "知识点", "信息处理", "解题步骤", "反应结构",
            "实验要求", "计算结构与方法", "条件与易错点", "表达要求",
        )
        for dimension in dimensions:
            self.assertEqual(prompt.count(f"**{dimension}：**"), 5)
        self.assertNotIn("**表达与范围：**", prompt)

    def test_prompt_follows_task_feature_case_boundary_warning_order(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        ordered_markers = (
            "**识别具体任务类型。**",
            "**确认抽象特征。**",
            "**用档内真实例题校准。**",
            "**比较相邻档位。**",
            "**用共用误判警示兜底。**",
        )
        positions = [prompt.index(marker) for marker in ordered_markers]
        self.assertEqual(positions, sorted(positions))
        self.assertGreater(prompt.index("### 五档共用误判警示"), prompt.index("### 4 / 5：拔高题与压轴题"))

    def test_case_coverage_is_not_lost_after_prompt_simplification(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        level_sections = {}
        for level in range(1, 6):
            start = prompt.index(f"### 难度{level}")
            end_marker = f"### 难度{level + 1}" if level < 5 else "## 三、教师边界例题"
            end = prompt.index(end_marker, start)
            level_sections[level] = prompt[start:end]
        minimum_cases = {1: 6, 2: 10, 3: 8, 4: 7, 5: 5}
        for level, minimum in minimum_cases.items():
            self.assertGreaterEqual(level_sections[level].count("【Case "), minimum)

        boundary = prompt[prompt.index("## 三、教师边界例题"):prompt.index("## 四、受控特征")]
        minimum_boundary_cases = {
            "### 1 / 2": 6,
            "### 2 / 3": 5,
            "### 3 / 4": 4,
            "### 4 / 5": 4,
        }
        starts = [(name, boundary.index(name)) for name in minimum_boundary_cases]
        for index, (name, start) in enumerate(starts):
            end = starts[index + 1][1] if index + 1 < len(starts) else boundary.index("### 五档共用误判警示")
            self.assertGreaterEqual(boundary[start:end].count("【Case "), minimum_boundary_cases[name])

    def test_runtime_has_no_source_label_or_retry_logic(self):
        runtime = (ROOT / "src" / "chemistry_difficulty_rating_with_cache.py").read_text(encoding="utf-8")
        source = runtime + (ROOT / "src" / "junior_chemistry_schema.py").read_text(encoding="utf-8")
        self.assertNotIn("UNTRUSTED" + "_LABEL_FIELDS", source)
        self.assertNotIn("source_difficulty" + "_untrusted", source)
        self.assertIn("for _single_http_attempt in range(1)", runtime)
        self.assertNotIn("--retries", runtime)


if __name__ == "__main__":
    unittest.main()
