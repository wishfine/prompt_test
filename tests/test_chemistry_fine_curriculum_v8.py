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


def fine_only_v9_features() -> dict:
    features = stable_v5_features()
    features.pop("curriculum_topics")
    features["fine_curriculum_topics"] = ["U04_T04", "U04_T06"]
    features["out_of_scope_items"] = []
    return features


class ChemistryFineCurriculumV8Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_module("chemistry_curriculum_catalog_test", CATALOG_PATH)
        cls.features = load_module("chemistry_features_v8_test", FEATURE_PATH)
        cls.runtime = load_module("chemistry_runtime_v8_test", RUNTIME_PATH)

    def setUp(self) -> None:
        self.runtime.EXPECTED_MODEL_FEATURE_SCHEMA_VERSION = ""

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

    def test_v9_model_contract_uses_only_fine_topics_and_derives_coarse_topics(self) -> None:
        item = fine_only_v9_features()

        validated = self.features.validate_observable_features(item)
        metrics = self.features.derive_observable_metrics(validated)

        self.assertNotIn(
            "curriculum_topics",
            self.features.OBSERVABLE_V9_MODEL_FEATURE_FIELDS,
        )
        self.assertEqual(
            set(validated["curriculum_topics"]),
            {"U4-3"},
        )
        self.assertTrue(metrics["fine_coarse_topic_consistent"])

    def test_v9_allows_pure_out_of_scope_without_forced_fine_topic_mapping(self) -> None:
        item = fine_only_v9_features()
        item["fine_curriculum_topics"] = []
        item["new_information_operation"] = "依赖题干未给出的超纲化学知识"
        item["out_of_scope_items"] = ["物质的量", "转移电子数"]

        validated = self.features.validate_observable_features(item)

        self.assertEqual(validated["curriculum_topics"], [])
        self.assertEqual(
            self.features.derive_observable_metrics(validated)["curriculum_scope"],
            "out_of_scope",
        )

    def test_audit_prompt_requires_v9_instead_of_accepting_v5_fallback(self) -> None:
        self.runtime.load_prompt_config(str(AUDIT_PROMPT_PATH))
        rating = {
            "features": stable_v5_features(),
            "coarse_difficulty": "送分/基础区间（1-2档）",
            "reasoning": {
                "core_basis": "一条课内规则。",
                "hard_point": "无。",
                "why_not_lower": "需要判断。",
                "why_not_higher": "没有连续推导。",
            },
            "difficulty_level": "基础题",
        }

        with self.assertRaisesRegex(
            self.runtime.ChemistrySchemaError,
            "chemistry_observable_v9",
        ):
            self.runtime.validate_rating_contract(rating)

        rating["features"] = fine_only_v9_features()
        validated = self.runtime.validate_rating_contract(rating)
        self.assertEqual(
            validated["model_feature_schema_version"],
            "chemistry_observable_v9",
        )
        self.assertEqual(validated["features"]["curriculum_topics"], ["U4-3"])

        result = self.runtime.postprocess_chemistry_difficulty(
            rating,
            {"stem": "写出水的化学式。"},
        )
        self.assertEqual(
            result["feature_schema_version"],
            "chemistry_observable_v9",
        )
        self.assertEqual(
            result["postprocess_profile"],
            "chemistry_observable_v9_fine_only_curriculum_audit_v1",
        )

    def test_v9_schema_repair_feedback_tells_model_to_replace_coarse_field(self) -> None:
        feedback = self.runtime.build_schema_repair_feedback(
            self.runtime.ChemistrySchemaError(
                "chemistry_observable_v9审计Prompt必须输出"
                "细知识点唯一课程合同"
            ),
            {"features": stable_v5_features()},
        )

        self.assertIn("删除curriculum_topics", feedback)
        self.assertIn("保留fine_curriculum_topics", feedback)

    def test_v9_run_config_records_expected_model_contract(self) -> None:
        self.runtime.load_prompt_config(str(AUDIT_PROMPT_PATH))

        config = self.runtime.build_run_config(
            BASE_PROMPT_PATH,
            AUDIT_PROMPT_PATH,
            seed=42,
            num=2,
        )

        self.assertEqual(
            config["expected_model_feature_schema_version"],
            "chemistry_observable_v9",
        )

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
        self.assertIn("chemistry_observable_v9", audit)
        self.assertIn("不要输出curriculum_topics", audit)
        self.assertLess(
            audit.index("课程越界前置检查"),
            audit.index("### fine_curriculum_topics"),
        )
        for topic_id in self.catalog.FINE_CURRICULUM_TOPIC_NAMES:
            self.assertIn(topic_id, audit)
        self.assertIn("不得单独触发升档或降档", audit)


if __name__ == "__main__":
    unittest.main()
