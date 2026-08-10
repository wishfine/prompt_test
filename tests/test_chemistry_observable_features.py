from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chemistry_observable_features.py"
PROTOCOL_PATH = ROOT / "prompts" / "初中化学难度打标提示词.txt"


def load_module():
    if not MODULE_PATH.exists():
        raise AssertionError(f"候选特征模块尚未实现: {MODULE_PATH}")
    spec = importlib.util.spec_from_file_location(
        "chemistry_observable_features",
        MODULE_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def valid_features() -> dict:
    return {
        "longest_solution_chain": [
            "从图像平台读取氢气总质量",
            "根据反应方程式求硫酸锌质量",
            "利用质量守恒求溶液总质量",
            "计算溶质质量分数",
        ],
        "task_groups": [
            {"task_type": "图表与数据", "count": 1},
            {"task_type": "定量计算", "count": 2},
        ],
        "rule_families": [
            "图表与数据",
            "性质与反应判断",
            "定量计算",
        ],
        "curriculum_units": ["U5", "U9"],
        "reaction_structure": "单一反应",
        "condition_operations": [],
        "representation_operations": [
            "图表数据→化学关系",
            "化学方程式→定量关系",
        ],
        "evidence_operations": ["多证据共同成立"],
        "experiment_operation": "无",
        "graph_table_operation": "拐点平台或分段",
        "calculation_operations": ["单一方程式", "单一守恒"],
        "new_information_operation": "无新信息",
    }


class ChemistryObservableFeatureTests(unittest.TestCase):
    def test_candidate_module_exists(self) -> None:
        self.assertTrue(MODULE_PATH.exists())

    def test_validates_observable_contract_and_derives_dbwu(self) -> None:
        module = load_module()

        validated = module.validate_observable_features(valid_features())
        derived = module.derive_observable_metrics(validated)

        self.assertEqual(derived["longest_chain_steps"], 4)
        self.assertEqual(derived["effective_task_count"], 3)
        self.assertEqual(derived["task_group_count"], 2)
        self.assertEqual(derived["rule_family_count"], 3)
        self.assertEqual(derived["curriculum_unit_count"], 2)
        self.assertTrue(derived["has_task_dependency"])

    def test_rejects_legacy_abstract_feature_contract(self) -> None:
        module = load_module()
        legacy = {
            "reasoning_depth": "2-3层",
            "reasoning_direction": "正向推导",
        }

        with self.assertRaisesRegex(ValueError, "字段集不匹配"):
            module.validate_observable_features(legacy)

    def test_rejects_duplicate_or_empty_observable_evidence(self) -> None:
        module = load_module()
        features = valid_features()
        features["curriculum_units"] = ["U5", "U5"]

        with self.assertRaisesRegex(ValueError, "curriculum_units.*重复"):
            module.validate_observable_features(features)

        features = valid_features()
        features["longest_solution_chain"] = []
        with self.assertRaisesRegex(ValueError, "longest_solution_chain"):
            module.validate_observable_features(features)

    def test_rejects_graph_conversion_without_graph_task(self) -> None:
        module = load_module()
        features = valid_features()
        features["graph_table_operation"] = "无"

        with self.assertRaisesRegex(ValueError, "图表转换"):
            module.validate_observable_features(features)

    def test_formal_protocol_documents_counting_boundaries(self) -> None:
        self.assertTrue(PROTOCOL_PATH.exists())
        text = PROTOCOL_PATH.read_text(encoding="utf-8")

        for expected in (
            "longest_solution_chain",
            "task_groups",
            "curriculum_units",
            "纯算术",
            "独立选项",
            "不得直接根据预想难度填写",
        ):
            self.assertIn(expected, text)


if __name__ == "__main__":
    unittest.main()
