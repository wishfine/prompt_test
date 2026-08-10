from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chemistry_difficulty_rating_with_cache.py"
SPEC = importlib.util.spec_from_file_location(
    "chemistry_stable_contract",
    MODULE_PATH,
)
assert SPEC and SPEC.loader
chemistry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chemistry)


class ChemistryStableContractTests(unittest.TestCase):
    def test_historical_core12_contract_remains_readable(self) -> None:
        rating = {
            "features": copy.deepcopy(chemistry.FEATURE_DEFAULTS),
            "coarse_difficulty": "送分/基础区间（1-2档）",
            "reasoning": {
                "core_basis": "单一熟悉教材原型",
                "hard_point": "无实质卡点",
                "why_not_lower": "已为最低档",
                "why_not_higher": "没有第二次化学决策",
            },
            "difficulty_level": "送分题",
        }

        validated = chemistry.validate_rating_contract(rating)

        self.assertEqual(validated["difficulty_level"], "送分题")
        self.assertNotIn("boundary_features", validated)
        self.assertNotIn("curriculum_span", validated)

    def test_production_prompt_uses_observable_v3_not_boundary8(self) -> None:
        prompt = (
            ROOT / "prompts" / "初中化学难度打标提示词.txt"
        ).read_text(encoding="utf-8")

        self.assertIn("15项可观测特征协议", prompt)
        self.assertIn('"longest_solution_chain"', prompt)
        self.assertIn('"curriculum_topics"', prompt)
        self.assertIn("量筒俯仰视误差链", prompt)
        self.assertIn("关键是三个前后依赖判断", prompt)
        self.assertNotIn('"boundary_features"', prompt)
        self.assertNotIn("Boundary-8 固定字段", prompt)


if __name__ == "__main__":
    unittest.main()
