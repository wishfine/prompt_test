from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "chemistry_difficulty_rating_with_cache.py"
PROMPT_PATH = ROOT / "prompts" / "初中化学难度打标提示词.txt"


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        "chemistry_observable_runtime_integration",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observable_features() -> dict:
    return {
        "longest_solution_chain": [
            "从图像平台读取生成气体质量",
            "根据化学方程式建立反应物质量关系",
            "利用质量守恒求反应后溶液质量",
            "计算反应后溶质质量分数",
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


def rating(level: str = "中等题") -> dict:
    coarse = {
        "送分题": "送分/基础区间（1-2档）",
        "基础题": "基础/中等区间（2-3档）",
        "中等题": "基础/中等区间（2-3档）",
        "拔高题": "中等/拔高区间（3-4档）",
        "压轴题": "拔高/压轴区间（4-5档）",
    }[level]
    return {
        "features": observable_features(),
        "coarse_difficulty": coarse,
        "reasoning": {
            "core_basis": "最长必要链为四个化学决策。",
            "hard_point": "图像、方程式与守恒连续衔接。",
            "why_not_lower": "不是单点直接应用。",
            "why_not_higher": "没有多反应分类与多阶段耦合。",
        },
        "difficulty_level": level,
    }


class ChemistryObservableRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_runtime()

    def test_formal_prompt_uses_observable_v2_output_contract(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        for field in self.runtime.OBSERVABLE_FEATURE_FIELDS:
            self.assertIn(f'"{field}"', prompt)
        for legacy_field in (
            "reasoning_depth",
            "reasoning_direction",
            "constraint_complexity",
            "evidence_relation",
            "unfamiliar_information_transfer",
            "subquestion_dependency",
        ):
            self.assertNotIn(f'"{legacy_field}"', prompt)

    def test_rating_contract_accepts_observable_features(self) -> None:
        validated = self.runtime.validate_rating_contract(rating())
        self.assertEqual(
            set(validated["features"]),
            set(self.runtime.OBSERVABLE_FEATURE_FIELDS),
        )

    def test_postprocess_derives_metrics_without_replacing_features(self) -> None:
        original = rating()
        result = self.runtime.postprocess_chemistry_difficulty(
            copy.deepcopy(original),
            {"stem": "根据图像和化学方程式完成计算。", "options": ""},
        )

        self.assertEqual(
            result["feature_schema_version"],
            "chemistry_observable_v2",
        )
        self.assertEqual(
            result["postprocess_profile"],
            "chemistry_observable_v2_teacher_distribution_v2_safe",
        )
        self.assertEqual(result["observable_metrics"]["longest_chain_steps"], 4)
        self.assertEqual(result["observable_metrics"]["effective_task_count"], 3)
        self.assertEqual(
            result["derived_core12_projection"]["reasoning_depth"],
            "4-5层",
        )
        self.assertEqual(result["features"], original["features"])

    def test_schema_retry_message_does_not_request_legacy_core12(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("上次输出未通过Core-12 schema", source)
        self.assertIn("上次输出未通过化学特征schema", source)

    def test_observable_projection_is_fact_derived(self) -> None:
        projection = self.runtime.project_observable_to_core12(
            observable_features()
        )
        self.assertEqual(projection["reasoning_depth"], "4-5层")
        self.assertEqual(
            projection["graph_table_requirement"],
            "拐点、平台或分段反推",
        )
        self.assertEqual(
            projection["calculation_model"],
            "单一守恒或多反应计算",
        )
        self.assertEqual(
            projection["knowledge_relation"],
            "跨模块融合",
        )

    def test_observable_chain_drives_basic_to_medium_candidate(self) -> None:
        item = rating("基础题")
        item["features"]["longest_solution_chain"] = [
            "根据现象确定参加反应的物质",
            "书写化学方程式建立质量关系",
            "根据方程式计算生成物质量",
        ]
        item["features"]["graph_table_operation"] = "直接读数"
        item["features"]["representation_operations"] = [
            "化学方程式→定量关系"
        ]
        item["features"]["calculation_operations"] = ["单一方程式"]

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "根据反应现象和方程式完成计算。", "options": ""},
        )

        self.assertEqual(
            result["postprocess_candidate_level"],
            "中等题",
        )
        self.assertEqual(
            result["postprocess_candidate_actions"][0]["rule"],
            "core12_basic_to_medium_complete_model",
        )

    def test_observable_projection_does_not_reuse_legacy_final_guard(
        self,
    ) -> None:
        item = rating("拔高题")
        item["features"].update(
            {
                "longest_solution_chain": [
                    "根据装置限制确定反应条件",
                    "由反应现象确定中间产物",
                    "把中间产物用于后一反应",
                    "联合图表数据建立计算关系",
                    "根据计算结果评价实验方案",
                ],
                "task_groups": [
                    {"task_type": "实验操作与探究", "count": 2},
                    {"task_type": "图表与数据", "count": 1},
                    {"task_type": "定量计算", "count": 1},
                ],
                "rule_families": [
                    "实验操作与探究",
                    "性质与反应判断",
                    "图表与数据",
                    "定量计算",
                ],
                "curriculum_units": ["U5", "U9", "U10"],
                "reaction_structure": "产物进入后一反应",
                "condition_operations": ["条件切换", "干扰条件排除"],
                "representation_operations": [
                    "宏观现象→化学符号",
                    "化学方程式→定量关系",
                    "图表数据→化学关系",
                ],
                "evidence_operations": ["多证据共同成立"],
                "experiment_operation": "多阶段定量探究",
                "graph_table_operation": "多图表联合",
                "calculation_operations": ["单一守恒", "多反应定量关系"],
                "new_information_operation": "新关系被多个任务共同使用",
            }
        )
        old_enabled = self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS
        old_writeback = (
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
        )
        try:
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = True
            result = self.runtime.postprocess_chemistry_difficulty(
                item,
                {"stem": "根据装置、图表和实验数据完成探究。"},
            )
        finally:
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = old_enabled
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = (
                old_writeback
            )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertIsNone(
            result["teacher_distribution_guard_candidate_action"]
        )

    def test_prompt_restores_detailed_boundary_calibration(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        required_anchors = (
            "同一熟悉分类规则检查四个候选项时，task_groups只记1个有效任务",
            "普通方案正误判断不等于方案评价",
            "中间量、纯算术和重复代入不单独增加最长链",
            "标准实验、常规计算与决定性卡点的边界",
            "工业流程、未知组成与守恒联立",
            "熟悉类别的一条固定规则直接判断",
            "多规则综合填空的受控广度",
            "同深度不同耦合",
            "固定基团转换关系",
            "枚举防错表",
        )
        for anchor in required_anchors:
            self.assertIn(anchor, prompt)
        self.assertGreaterEqual(prompt.count("### 示例"), 13)

    def test_prompt_uses_curriculum_units_as_task_boundaries(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        for anchor in (
            "课程单元任务边界",
            "U1：直接仪器识别",
            "U4/U5：物质组成、化学用语与方程式",
            "U9/U10：溶液、酸碱盐与证据链",
            "单元名称不是难度先验",
        ):
            self.assertIn(anchor, prompt)

    def test_schema_retry_budget_allows_three_repairs(self) -> None:
        self.assertGreaterEqual(self.runtime.MAX_SCHEMA_RETRIES, 3)

    def test_legacy_core12_remains_readable_for_historical_results(self) -> None:
        legacy = copy.deepcopy(self.runtime.FEATURE_DEFAULTS)
        validated = self.runtime.validate_feature_contract(legacy)
        self.assertEqual(validated, legacy)


if __name__ == "__main__":
    unittest.main()
