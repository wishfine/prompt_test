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
        self.assertIn("单个静止物体的教材原型受力图", prefix)
        self.assertIn("一条凸透镜特殊光线", prefix)
        self.assertRegex(prefix, r"复杂受力分析[^。]{0,100}至少基础题")
        self.assertIn("同一小节的多个直接识记空", prefix)
        self.assertRegex(prefix, r"跨不同知识点[^。]{0,100}至少基础题")

    def test_production_prompt_uses_direct_retrieval_bundle_easy_boundary(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("直接检索束", prefix)
        self.assertIn("每个空或选项都能独立由教材事实直接作答", prefix)
        self.assertIn("不需要共同物理过程、规律选择或条件联动", prefix)
        self.assertIn("分子动理论知识结构图", prefix)
        self.assertIn("多个空不等于多个应用步骤", prefix)

    def test_production_prompt_uses_five_dimension_anchor_with_task_structure_check(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("教师五维定档主标准", prefix)
        self.assertIn("真实解题任务结构", prefix)
        for dimension in ["直接识别", "显性应用", "常规分析", "决定性转换", "全链耦合"]:
            self.assertIn(dimension, prefix)
        self.assertIn("步骤数不是档位门槛", prefix)

    def test_production_prompt_does_not_treat_choice_count_or_simple_application_as_easy(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("四个选项本身既不升档也不降档", prefix)
        self.assertIn("需要把生活情境映射到物理规律", prefix)
        self.assertNotIn("四个短选项或一步因果；只要", prefix)
        self.assertNotIn("一步生活原型对应，不必然排除送分", prefix)

    def test_production_prompt_allows_single_question_internal_chain_to_be_final(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("不以存在多个小问或前问结果复用为必要条件", prefix)
        self.assertIn("单个设问内部", prefix)

    def test_production_prompt_contains_sample_derived_boundary_corrections(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("四个选项不会自动排除送分题", prefix)
        self.assertIn("高密度概念辨析", prefix)
        self.assertIn("一个决定性转换", prefix)
        self.assertIn("整题的完整推理链", prefix)

    def test_production_prompt_has_no_unavailable_score_rate_variable(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("得分率", prefix)
        self.assertNotRegex(prefix, r"(?<![A-Za-z])P(?:≥|<|时)")

    def test_production_prompt_uses_stable_adjacent_boundary_table(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("最短且完整的有效解题链", prefix)
        self.assertIn("不能只统计最后一问从已知答案出发的局部步骤", prefix)
        self.assertIn("相邻档位稳定决策表", prefix)
        self.assertIn("3-5步本身不能证明达到中等题", prefix)
        self.assertIn("6-8步也可以判压轴题", prefix)
        self.assertIn("步骤数只作支持证据，不作为单独门槛", prefix)
        self.assertNotIn("向上复核：防止专家视角压缩步骤", prefix)

    def test_production_prompt_has_sample_anchored_hard_and_final_examples(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("两条关系线", prefix)
        self.assertIn("反推图线身份", prefix)
        self.assertIn("多开关电路中包含小灯泡", prefix)
        self.assertIn("多安全约束和功率边界筛选", prefix)
        self.assertNotIn("将天平改装为液体密度测量仪", prefix)

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
