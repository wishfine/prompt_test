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


class FusedPromptAssetTests(unittest.TestCase):
    def test_final_prompt_has_nine_non_versioned_calibration_examples(self) -> None:
        path = ROOT / "prompts" / "初中物理难度打标提示词.txt"
        namespace: dict[str, str] = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {}, namespace)
        prefix = namespace["DIFFICULTY_RATING_PROMPT_PREFIX"]
        self.assertEqual(len(re.findall(r"【校准示例\d+】", prefix)), 9)
        self.assertNotRegex(prefix, r"V[1-9]")

    def test_final_prompt_json_example_has_no_duplicate_keys(self) -> None:
        path = ROOT / "prompts" / "初中物理难度打标提示词.txt"
        namespace: dict[str, str] = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {}, namespace)
        prefix = namespace["DIFFICULTY_RATING_PROMPT_PREFIX"]
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


if __name__ == "__main__":
    unittest.main()
