import sys
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import junior_chemistry_schema as schema


def rating(level="基础题", topic_ids=None):
    return {
        "difficulty_level": level,
        "features": {
            "knowledge": {"topic_ids": topic_ids or ["U02_T03"]},
            "solution_process": {
                "step_count": "2-3步",
                "task_types": ["实验操作判断"],
                "key_steps": ["判断实验操作"],
                "task_relation": "单项任务",
            },
            "information_processing": ["无"],
            "reaction_processes": {"processes": ["无"], "requires_condition_selection": False},
            "experiment_tasks": ["基本操作判断"],
            "calculation": {"has_calculation": False, "calculation_steps": "无", "types": ["无"], "special_methods": ["无"]},
            "difficulty_conditions": {"hidden_conditions": ["无"], "interference_points": ["无"]},
            "expression_requirements": ["无"],
            "question_context": {"unfamiliar_materials": ["无"], "interdisciplinary_context": ["无"]},
            "curriculum_scope": {"scope": "within_junior", "extra_points": []},
        },
        "reasoning": {
            "knowledge_points": "涉及教材知识点。",
            "solution_process": "完成实验操作判断。",
            "main_difficulty_factors": "操作判断。",
            "level_basis": "判为基础题。",
        },
    }


class JuniorChemistrySchemaTests(unittest.TestCase):
    def test_curriculum_map_is_loaded(self):
        topics = schema.load_curriculum_topics()
        self.assertGreaterEqual(len(topics), 58)
        self.assertEqual(topics["U10_T02"]["unit_id"], "U10")

    def test_postprocess_computes_coverage_without_writeback(self):
        result = schema.postprocess_chemistry_difficulty(rating("送分题", ["U02_T03", "U10_T01"]), {})
        knowledge = result["features"]["knowledge"]
        self.assertEqual(knowledge["knowledge_point_count"], 2)
        self.assertEqual(knowledge["unit_count"], 2)
        self.assertTrue(knowledge["cross_unit"])
        self.assertEqual(result["difficulty_level"], "送分题")
        self.assertFalse(result["postprocess"]["writeback_applied"])
        self.assertEqual(result["postprocess"]["candidates"][0]["candidate_level"], "基础题")

    def test_out_of_scope_requires_specific_content(self):
        value = rating(topic_ids=[])
        value["features"]["curriculum_scope"] = {"scope": "out_of_scope", "extra_points": ["物质的量"]}
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["curriculum_scope"]["scope"], "out_of_scope")

    def test_rejects_unknown_feature_shape(self):
        value = rating()
        value["features"] = {"legacy_feature": "unused"}
        with self.assertRaises(schema.ChemistrySchemaError):
            schema.validate_rating_contract(value)

    def test_rejects_unknown_topic(self):
        with self.assertRaises(schema.ChemistrySchemaError):
            schema.postprocess_chemistry_difficulty(rating(topic_ids=["U99_T99"]), {})

    def test_absent_feature_uses_none_enum(self):
        value = rating()
        value["features"]["information_processing"] = ["无"]
        value["features"]["experiment_tasks"] = ["无"]
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertEqual(result["features"]["information_processing"], ["无"])

    def test_rejects_unknown_enum_and_mixed_none(self):
        value = rating()
        value["features"]["information_processing"] = ["一次表征转换"]
        with self.assertRaises(schema.ChemistrySchemaError):
            schema.validate_rating_contract(value)

        value = rating()
        value["features"]["experiment_tasks"] = ["无", "基本操作判断"]
        with self.assertRaises(schema.ChemistrySchemaError):
            schema.validate_rating_contract(value)

    def test_calculation_fields_must_be_consistent(self):
        value = rating()
        value["features"]["calculation"]["has_calculation"] = True
        with self.assertRaises(schema.ChemistrySchemaError):
            schema.validate_rating_contract(value)

        value = rating()
        value["features"]["solution_process"]["task_types"] = ["实验操作判断", "计算"]
        value["features"]["solution_process"]["task_relation"] = "前后依赖"
        value["features"]["calculation"] = {
            "has_calculation": True,
            "calculation_steps": "2-3步",
            "types": ["化学方程式计算"],
            "special_methods": ["质量守恒"],
        }
        result = schema.validate_rating_contract(value)
        self.assertTrue(result["features"]["calculation"]["has_calculation"])

    def test_audit_candidate_has_evidence_and_never_writes_back(self):
        value = rating("送分题", ["U02_T03", "U10_T01"])
        value["features"]["difficulty_conditions"]["interference_points"] = ["易混概念"]
        result = schema.postprocess_chemistry_difficulty(value, {})
        self.assertGreaterEqual(len(result["postprocess"]["candidates"]), 2)
        for candidate in result["postprocess"]["candidates"]:
            self.assertIn("reason", candidate)
            self.assertIn("evidence", candidate)
            self.assertFalse(candidate["writeback_applied"])
        self.assertEqual(result["difficulty_level"], "送分题")
        self.assertEqual(result["postprocess"]["final_level"], "送分题")

    def test_all_prompt_json_examples_follow_runtime_contract(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        starts = []
        position = 0
        while True:
            position = prompt.find("\n{\n  \"features\"", position)
            if position < 0:
                break
            starts.append(position + 1)
            position += 2
        # Few-shot仅保留边界有效信息；完整JSON只在“输出要求”中出现一次。
        self.assertEqual(len(starts), 1)
        for start in starts:
            depth = 0
            in_string = False
            escaped = False
            end = None
            for index in range(start, len(prompt)):
                char = prompt[index]
                if in_string:
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        end = index + 1
                        break
            self.assertIsNotNone(end)
            schema.validate_rating_contract(json.loads(prompt[start:end]))

    def test_prompt_documents_every_runtime_enum(self):
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(encoding="utf-8")
        for enum_values in schema.ENUMS.values():
            for value in enum_values:
                self.assertIn(value, prompt)
        for value in schema.STEPS | schema.RELATIONS | schema.SCOPES | schema.LEVELS:
            self.assertIn(value, prompt)

    def test_junior_runtime_has_no_legacy_feature_or_source_label_logic(self):
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
        ):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
