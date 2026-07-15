# -*- coding: utf-8 -*-
"""V7 兼容基线资源的静态验收。"""

from __future__ import annotations

import unittest
from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[1]


class V7CompatAssetTests(unittest.TestCase):
    def test_archived_prompt_is_python_config_and_complete(self) -> None:
        path = ROOT / "prompts" / "archive" / "初中物理难度打标提示词_v7_best.txt"
        namespace: dict[str, str] = {}
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), {}, namespace)
        prefix = namespace.get("DIFFICULTY_RATING_PROMPT_PREFIX", "")
        suffix = namespace.get("DIFFICULTY_RATING_PROMPT_SUFFIX", "")
        self.assertIn("真实边界判定 few-shot 示例", prefix)
        self.assertIn("【示例26", prefix)
        self.assertIn("全面的难度分析和评级", suffix)

    def test_legacy_reference_compiles_and_contains_v7_final_definition(self) -> None:
        path = ROOT / "src" / "legacy" / "physics_difficulty_rating_v7_reference.py"
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        self.assertIn("V7 边界小修版", source)
        self.assertGreaterEqual(source.count("def postprocess_physics_difficulty"), 1)


class ProductionPromptAssetTests(unittest.TestCase):
    def load_prefix(self) -> str:
        path = ROOT / "prompts" / "初中物理难度打标提示词.txt"
        namespace: dict[str, str] = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {}, namespace)
        return namespace["DIFFICULTY_RATING_PROMPT_PREFIX"]

    def test_production_prompt_keeps_old_baseline_depth_and_two_layer_examples(self) -> None:
        prefix = self.load_prefix()
        self.assertGreaterEqual(len(prefix), 17500)
        self.assertLess(len(prefix), 23000)
        self.assertGreaterEqual(prefix.count("### 代表性例题"), 5)
        self.assertIn("## 相邻档位边界校准 few-shot", prefix)

    def test_production_prompt_has_ten_non_versioned_boundary_examples(self) -> None:
        prefix = self.load_prefix()
        self.assertEqual(len(re.findall(r"【边界示例\d+】", prefix)), 10)
        self.assertNotRegex(prefix, r"V5|V6|V7")
        self.assertNotIn("回收中等保护", prefix)
        self.assertNotIn("压轴保护恢复", prefix)

    def test_production_prompt_resolves_diagram_and_multi_blank_conflicts(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("完整受力图、凸透镜光线作图", prefix)
        self.assertRegex(prefix, r"完整受力图、凸透镜光线作图[^。]{0,80}至少基础题")
        self.assertRegex(prefix, r"跨不同知识点[^。]{0,100}至少基础题")

    def test_production_prompt_uses_teacher_five_dimension_rubric(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("教师五维定档主标准", prefix)
        for dimension in ["知识量", "过程/对象", "数学工具", "情境", "思维层次"]:
            self.assertIn(dimension, prefix)
        self.assertIn("实际有效推理约 3-4 步", prefix)
        self.assertIn("实际有效推理约 5-6 步", prefix)
        self.assertIn("实际有效推理 7 步以上", prefix)

    def test_production_prompt_does_not_use_features_as_postprocess_triggers(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("18 个 features 只用于客观记录和审计", prefix)
        self.assertIn("不得根据单个 feature 机械升降档", prefix)
        self.assertNotIn("凡是需要物理公式代入", prefix)
        self.assertNotIn("同时出现 9 步以上复杂推理", prefix)

    def test_batch_script_defaults_to_production_prompt(self) -> None:
        source = (ROOT / "src" / "physics_difficulty_rating_with_cache.py").read_text(encoding="utf-8")
        self.assertIn('"prompts", "初中物理难度打标提示词.txt"', source)
        self.assertNotIn('default_prompt =', source)

    def test_final_prompt_json_example_has_no_duplicate_keys(self) -> None:
        prefix = self.load_prefix()
        marker = "合法 JSON 示例"
        start = prefix.index("{", prefix.index(marker))

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            output: dict[str, object] = {}
            for key, value in pairs:
                if key in output:
                    raise ValueError(f"重复 JSON key: {key}")
                output[key] = value
            return output

        try:
            parsed, _ = json.JSONDecoder(object_pairs_hook=reject_duplicates).raw_decode(prefix[start:])
        except (ValueError, json.JSONDecodeError) as exc:
            self.fail(str(exc))
        self.assertEqual(set(parsed["features"]), {
            "step_count", "formula_count", "calculation_complexity", "reasoning_chain",
            "problem_structure", "additional_structure", "information_carrier", "reality_question",
            "subquestion_dependency", "knowledge_count", "knowledge_diff", "cross_module",
            "state_count", "constraint_count", "variable_relation", "experiment_requirement",
            "graph_table_requirement", "error_risk",
        })
        self.assertIn(parsed["difficulty_level"], ["送分题", "基础题", "中等题", "拔高题", "压轴题"])
        self.assertNotIn("difficulty_level", parsed["reasoning"])


if __name__ == "__main__":
    unittest.main()
