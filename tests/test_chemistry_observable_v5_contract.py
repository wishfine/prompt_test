from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
FEATURE_PATH = ROOT / "src" / "chemistry_observable_features.py"
RUNTIME_PATH = ROOT / "src" / "chemistry_difficulty_rating_with_cache.py"
PROMPT_PATH = ROOT / "prompts" / "初中化学难度打标提示词.txt"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def current_features() -> dict:
    return {
        "longest_solution_chain": [
            "读取图表中的反应阶段",
            "根据方程式建立质量关系",
            "利用质量守恒求目标量",
        ],
        "task_groups": [
            {"task_type": "图表与数据", "count": 1},
            {"task_type": "定量计算", "count": 2},
        ],
        "rule_families": ["图表与数据", "定量计算"],
        "curriculum_topics": ["U5-2", "U9-3"],
        "parallel_task_relation": "共享同一化学模型的关联任务",
        "solution_topology": "单线性常规链",
        "reaction_structure": "单一反应",
        "condition_operations": [],
        "representation_operations": [
            "图表数据→化学关系",
            "化学方程式→定量关系",
        ],
        "evidence_operations": ["多证据共同成立"],
        "experiment_operation": "无",
        "experiment_task_structure": "无实验判断",
        "visual_task_structure": "共享装置流程或图表模型",
        "graph_table_operation": "拐点平台或分段",
        "error_analysis_operation": "无误差分析",
        "calculation_operations": ["单一方程式", "单一守恒"],
        "new_information_operation": "无新信息",
    }


def rating(level: str = "中等题") -> dict:
    coarse_by_level = {
        "送分题": "送分/基础区间（1-2档）",
        "基础题": "送分/基础区间（1-2档）",
        "中等题": "中等/拔高区间（3-4档）",
        "拔高题": "中等/拔高区间（3-4档）",
        "压轴题": "压轴区间（5档）",
    }
    return {
        "features": current_features(),
        "coarse_difficulty": coarse_by_level[level],
        "reasoning": {
            "core_basis": "最长必要链包含三个前后依赖的化学决策。",
            "hard_point": "需要连续使用图表、方程式和守恒关系。",
            "why_not_lower": "不是单点直接匹配。",
            "why_not_higher": "没有竞争反应或分类讨论。",
        },
        "difficulty_level": level,
    }


class ChemistryObservableV5ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features = load_module("chemistry_observable_v5_features", FEATURE_PATH)
        cls.runtime = load_module("chemistry_observable_v5_runtime", RUNTIME_PATH)

    def test_current_contract_has_seventeen_fields_without_ra_counts(self) -> None:
        self.assertEqual(len(self.features.OBSERVABLE_FEATURE_FIELDS), 17)
        self.assertNotIn(
            "direct_retrieval_task_count",
            self.features.OBSERVABLE_FEATURE_FIELDS,
        )
        self.assertNotIn(
            "rule_application_task_count",
            self.features.OBSERVABLE_FEATURE_FIELDS,
        )
        self.assertIn("solution_topology", self.features.OBSERVABLE_FEATURE_FIELDS)
        self.assertIn(
            "experiment_task_structure",
            self.features.OBSERVABLE_FEATURE_FIELDS,
        )

    def test_current_contract_validates_as_observable_v5(self) -> None:
        validated = self.runtime.validate_feature_contract(current_features())

        self.assertEqual(set(validated), set(self.runtime.OBSERVABLE_FEATURE_FIELDS))
        self.assertEqual(
            self.runtime.observable_feature_schema_version(validated),
            "chemistry_observable_v5",
        )

    def test_historical_nineteen_field_v4_remains_readable(self) -> None:
        historical = copy.deepcopy(current_features())
        historical["direct_retrieval_task_count"] = 1
        historical["rule_application_task_count"] = 2

        validated = self.runtime.validate_feature_contract(historical)

        self.assertEqual(
            set(validated),
            set(self.runtime.OBSERVABLE_V4_FEATURE_FIELDS),
        )
        self.assertEqual(
            self.runtime.observable_feature_schema_version(validated),
            "chemistry_observable_v4",
        )

    def test_current_contract_avoids_generic_core12_writeback(self) -> None:
        item = rating("基础题")
        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "根据图表和方程式完成定量计算。"},
        )

        self.assertEqual(result["feature_schema_version"], "chemistry_observable_v5")
        self.assertEqual(result["postprocess_candidate_actions"], [])

    def test_prompt_documents_task_granularity_without_ra_fields(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("17项可观测特征协议", prompt)
        self.assertNotIn('"direct_retrieval_task_count"', prompt)
        self.assertNotIn('"rule_application_task_count"', prompt)
        self.assertIn("不同化学命题或不同作答目标", prompt)
        self.assertIn("同一规则只表示B不增加", prompt)
        self.assertIn("独立选项不增加D", prompt)


if __name__ == "__main__":
    unittest.main()
