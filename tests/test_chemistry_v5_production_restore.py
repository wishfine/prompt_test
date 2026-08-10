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


def production_features() -> dict:
    return {
        "longest_solution_chain": [
            "读取多问共享的实验或流程信息",
            "确定反应阶段与关键中间量",
            "建立多反应定量关系",
            "利用前述结果计算后续目标量",
            "核验最终组成或结果",
        ],
        "task_groups": [
            {"task_type": "实验操作与探究", "count": 3},
            {"task_type": "定量计算", "count": 4},
        ],
        "rule_families": ["实验操作与探究", "定量计算"],
        "curriculum_topics": ["U1-2", "U5-2", "U9-3"],
        "parallel_task_relation": "共享同一化学模型的关联任务",
        "solution_topology": "多阶段反应网络",
        "reaction_structure": "产物进入后一反应",
        "condition_operations": ["条件切换"],
        "representation_operations": ["化学方程式→定量关系"],
        "evidence_operations": ["多证据共同成立"],
        "experiment_operation": "多阶段定量探究",
        "experiment_task_structure": "控制变量或数据归纳",
        "visual_task_structure": "共享装置流程或图表模型",
        "graph_table_operation": "多组比较",
        "error_analysis_operation": "无误差分析",
        "calculation_operations": ["多反应定量关系"],
        "new_information_operation": "无新信息",
    }


def hard_rating() -> dict:
    return {
        "features": production_features(),
        "coarse_difficulty": "中等/拔高区间（3-4档）",
        "reasoning": {
            "core_basis": "多问共享同一实验模型并形成五步定量链。",
            "hard_point": "需要把前问中间量继续用于多反应计算。",
            "why_not_lower": "不是并列的一步常规任务。",
            "why_not_higher": "模型原始判断认为仍属于熟悉综合链。",
        },
        "difficulty_level": "拔高题",
    }


class ChemistryV5ProductionRestoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.features = load_module(
            "chemistry_v5_restore_features",
            FEATURE_PATH,
        )
        cls.runtime = load_module(
            "chemistry_v5_restore_runtime",
            RUNTIME_PATH,
        )

    def setUp(self) -> None:
        self.original_flags = {
            "enabled": self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS,
            "writeback": self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK,
        }
        self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = True

    def tearDown(self) -> None:
        self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = (
            self.original_flags["enabled"]
        )
        self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = (
            self.original_flags["writeback"]
        )

    def test_production_contract_is_v5_seventeen_fields(self) -> None:
        self.assertEqual(len(self.features.OBSERVABLE_FEATURE_FIELDS), 17)
        self.assertEqual(
            set(self.features.OBSERVABLE_FEATURE_FIELDS),
            set(production_features()),
        )
        self.assertNotIn(
            "response_operations",
            self.features.OBSERVABLE_FEATURE_FIELDS,
        )
        self.assertNotIn(
            "cross_subject_operations",
            self.features.OBSERVABLE_FEATURE_FIELDS,
        )

    def test_prompt_requests_v5_without_two_experimental_fields(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("17项可观测特征协议", prompt)
        self.assertNotIn('"response_operations"', prompt)
        self.assertNotIn('"cross_subject_operations"', prompt)
        self.assertNotIn("### 4. response_operations", prompt)
        self.assertNotIn("### 6. cross_subject_operations", prompt)

    def test_program_metrics_remain_available_for_v5_output(self) -> None:
        result = self.runtime.postprocess_chemistry_difficulty(
            hard_rating(),
            {
                "stem": "完成下列任务。（1）判断阶段。（2）求中间量。",
                "sub_questions": [
                    {"stem": "任务1"},
                    {"stem": "任务2"},
                    {"stem": "任务3"},
                ],
            },
        )

        self.assertEqual(result["feature_schema_version"], "chemistry_observable_v5")
        self.assertEqual(
            result["observable_metrics"]["explicit_subquestion_count"],
            3,
        )
        self.assertGreater(
            result["observable_metrics"]["question_text_char_count"],
            0,
        )

    def test_narrow_dense_multiquestion_quantitative_chain_promotes_to_final(self) -> None:
        result = self.runtime.postprocess_chemistry_difficulty(
            hard_rating(),
            {
                "stem": "共享装置中的多阶段反应与定量计算。",
                "sub_questions": [
                    {"stem": "任务1"},
                    {"stem": "任务2"},
                    {"stem": "任务3"},
                    {"stem": "任务4"},
                ],
            },
        )

        self.assertEqual(result["difficulty_level"], "压轴题")
        self.assertEqual(len(result["postprocess_actions"]), 1)
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "teacher_hard_to_final_dense_multiquestion_quantitative_chain",
        )

    def test_narrow_final_rule_requires_four_explicit_subquestions(self) -> None:
        result = self.runtime.postprocess_chemistry_difficulty(
            hard_rating(),
            {
                "stem": "共享装置中的多阶段反应与定量计算。",
                "sub_questions": [
                    {"stem": "任务1"},
                    {"stem": "任务2"},
                    {"stem": "任务3"},
                ],
            },
        )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_narrow_final_rule_requires_advanced_calculation(self) -> None:
        item = hard_rating()
        item["features"]["calculation_operations"] = ["单一方程式"]

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {
                "stem": "共享装置中的常规单一方程式计算。",
                "sub_questions": [
                    {"stem": "任务1"},
                    {"stem": "任务2"},
                    {"stem": "任务3"},
                    {"stem": "任务4"},
                ],
            },
        )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_narrow_final_rule_accepts_safe_historical_enum_aliases(self) -> None:
        item = hard_rating()
        item["features"]["task_groups"].append(
            {"task_type": "误差分析", "count": 1}
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {
                "stem": "共享装置中的多阶段反应、误差分析与定量计算。",
                "sub_questions": [
                    {"stem": "任务1"},
                    {"stem": "任务2"},
                    {"stem": "任务3"},
                    {"stem": "任务4"},
                ],
            },
        )

        self.assertEqual(result["difficulty_level"], "压轴题")
        self.assertIn(
            "task_groups.task_type",
            {
                action["field"]
                for action in result["feature_normalization_actions"]
            },
        )


if __name__ == "__main__":
    unittest.main()
