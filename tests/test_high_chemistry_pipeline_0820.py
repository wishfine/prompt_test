# -*- coding: utf-8 -*-
"""高中化学 0820 第一阶段边界判断的纯函数测试。"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "src" / "high_chemistry_pipeline_core_0820.py"
SPEC = importlib.util.spec_from_file_location("high_chemistry_pipeline_core_0820", CORE_PATH)
assert SPEC and SPEC.loader
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def boundary_review(decision: str) -> dict[str, str]:
    return {
        "decision": decision,
        "decisive_chemical_task": "先确定实际反应，再判断产物。",
        "why_not_other_side": "该关系是得到答案前不可省略的前提。",
    }


class Boundary85ReviewTests(unittest.TestCase):
    def test_stage1_schema_requires_boundary_review(self) -> None:
        schema = core.build_stage1_output_schema()
        self.assertIn("boundary_85_review", schema["required"])
        review = schema["properties"]["boundary_85_review"]
        self.assertEqual(
            set(review["properties"]["decision"]["enum"]),
            {"保持85及以上", "进入85以下"},
        )

    def test_boundary_review_accepts_matching_score(self) -> None:
        core.validate_boundary_85_review(boundary_review("保持85及以上"), 85)
        core.validate_boundary_85_review(boundary_review("进入85以下"), 84.9)

    def test_boundary_review_rejects_score_on_wrong_side(self) -> None:
        with self.assertRaisesRegex(ValueError, "小于85"):
            core.validate_boundary_85_review(boundary_review("保持85及以上"), 82)
        with self.assertRaisesRegex(ValueError, "不小于85"):
            core.validate_boundary_85_review(boundary_review("进入85以下"), 85)


if __name__ == "__main__":
    unittest.main()
