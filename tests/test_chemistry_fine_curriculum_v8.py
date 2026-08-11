from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "src" / "chemistry_observable_features.py"
CATALOG_PATH = ROOT / "src" / "chemistry_curriculum_catalog.py"
RUNTIME_PATH = ROOT / "src" / "chemistry_difficulty_rating_with_cache.py"
BASE_PROMPT_PATH = ROOT / "prompts" / "初中化学难度打标提示词.txt"
AUDIT_PROMPT_PATH = (
    ROOT / "prompts" / "初中化学难度打标提示词_58知识点审计AB.txt"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stable_v5_features() -> dict:
    return {
        "longest_solution_chain": ["识别题目要求", "应用课内规则"],
        "task_groups": [{"task_type": "性质与反应判断", "count": 1}],
        "rule_families": ["性质用途或现象判断"],
        "curriculum_topics": ["U4-3"],
        "parallel_task_relation": "单一答题目标",
        "solution_topology": "单线性常规链",
        "reaction_structure": "无反应任务",
        "condition_operations": [],
        "representation_operations": ["宏观对象→化学符号"],
        "evidence_operations": [],
        "experiment_operation": "无",
        "experiment_task_structure": "无实验判断",
        "visual_task_structure": "无必要视觉信息",
        "graph_table_operation": "无",
        "error_analysis_operation": "无误差分析",
        "calculation_operations": [],
        "new_information_operation": "无新信息",
    }


class ChemistryFineCurriculumV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_module("chemistry_curriculum_catalog_test", CATALOG_PATH)
        cls.features = load_module("chemistry_features_v8_test", FEATURE_PATH)
        cls.runtime = load_module("chemistry_runtime_v8_test", RUNTIME_PATH)

    def test_catalog_has_58_unique_topics_and_complete_coarse_mapping(self) -> None:
        self.assertEqual(len(self.catalog.FINE_CURRICULUM_TOPIC_NAMES), 58)
        self.assertEqual(
            set(self.catalog.FINE_CURRICULUM_TOPIC_NAMES),
            set(self.catalog.FINE_CURRICULUM_TOPIC_TO_COARSE),
        )
        self.assertEqual(
            self.catalog.FINE_CURRICULUM_TOPIC_TO_COARSE["U04_T06"],
            "U4-3",
        )
        self.assertEqual(
            self.catalog.FINE_CURRICULUM_TOPIC_TO_COARSE["U10_T09"],
            "U10-3",
        )

    def test_v8_is_optional_and_frozen_v5_remains_default(self) -> None:
        self.assertEqual(
            self.features.OBSERVABLE_FEATURE_FIELDS,
            self.features.OBSERVABLE_V5_FEATURE_FIELDS,
        )
        self.assertEqual(len(self.features.OBSERVABLE_FEATURE_FIELDS), 17)
        self.assertEqual(len(self.features.OBSERVABLE_V8_FEATURE_FIELDS), 19)

    def test_v8_derives_fine_counts_and_mapping_without_changing_grade_fields(self) -> None:
        item = stable_v5_features()
        item["fine_curriculum_topics"] = ["U04_T04", "U04_T06"]
        item["out_of_scope_items"] = []

        validated = self.features.validate_observable_features(item)
        metrics = self.features.derive_observable_metrics(validated)

        self.assertEqual(metrics["fine_curriculum_topic_count"], 2)
        self.assertEqual(metrics["fine_curriculum_unit_count"], 1)
        self.assertEqual(metrics["fine_mapped_coarse_topics"], ["U4-3"])
        self.assertTrue(metrics["fine_coarse_topic_consistent"])
        self.assertEqual(metrics["curriculum_scope"], "within_junior")
        self.assertEqual(
            self.runtime.observable_feature_schema_version(validated),
            "chemistry_observable_v8",
        )

    def test_v8_audit_fields_do_not_change_postprocess_candidates(self) -> None:
        v5 = stable_v5_features()
        v8 = {
            **v5,
            "fine_curriculum_topics": ["U04_T04"],
            "out_of_scope_items": [],
        }
        data = {"stem": "写出水的化学式。"}

        def rating(features: dict) -> dict:
            return {
                "features": features,
                "coarse_difficulty": "送分/基础区间（1-2档）",
                "reasoning": {
                    "core_basis": "书写一个课内化学式。",
                    "hard_point": "规范书写。",
                    "why_not_lower": "需自主生成化学符号。",
                    "why_not_higher": "没有连续推导。",
                },
                "difficulty_level": "基础题",
            }

        v5_result = self.runtime.postprocess_chemistry_difficulty(
            rating(v5), data
        )
        v8_result = self.runtime.postprocess_chemistry_difficulty(
            rating(v8), data
        )

        self.assertEqual(
            v8_result["postprocess_candidate_actions"],
            v5_result["postprocess_candidate_actions"],
        )
        self.assertEqual(v8_result["difficulty_level"], "基础题")

    def test_coarse_mismatch_is_audited_instead_of_causing_schema_retry(self) -> None:
        item = stable_v5_features()
        item["fine_curriculum_topics"] = ["U06_T03"]
        item["out_of_scope_items"] = []

        validated = self.features.validate_observable_features(item)
        metrics = self.features.derive_observable_metrics(validated)

        self.assertFalse(metrics["fine_coarse_topic_consistent"])
        self.assertEqual(metrics["fine_mapped_coarse_topics"], ["U6-2"])

    def test_out_of_scope_requires_specific_items(self) -> None:
        item = stable_v5_features()
        item["new_information_operation"] = "依赖题干未给出的超纲化学知识"
        item["fine_curriculum_topics"] = ["U05_T01"]
        item["out_of_scope_items"] = []
        with self.assertRaisesRegex(ValueError, "out_of_scope_items"):
            self.features.validate_observable_features(item)

        item["out_of_scope_items"] = ["物质的量", "转移电子数"]
        validated = self.features.validate_observable_features(item)
        metrics = self.features.derive_observable_metrics(validated)
        self.assertEqual(metrics["curriculum_scope"], "out_of_scope")
        self.assertEqual(metrics["out_of_scope_items"], ["物质的量", "转移电子数"])

    def test_audit_prompt_is_separate_and_contains_all_topic_ids(self) -> None:
        base = BASE_PROMPT_PATH.read_text(encoding="utf-8")
        audit = AUDIT_PROMPT_PATH.read_text(encoding="utf-8")

        self.assertNotIn('"fine_curriculum_topics"', base)
        self.assertIn('@include 初中化学难度打标提示词.txt', audit)
        self.assertIn('"fine_curriculum_topics"', audit)
        self.assertIn('"out_of_scope_items"', audit)
        for topic_id in self.catalog.FINE_CURRICULUM_TOPIC_NAMES:
            self.assertIn(topic_id, audit)
        self.assertIn("不得单独触发升档或降档", audit)


if __name__ == "__main__":
    unittest.main()
