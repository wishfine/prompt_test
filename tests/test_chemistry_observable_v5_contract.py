from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
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
        "response_operations": ["图表读取或归纳", "定量计算"],
        "curriculum_topics": ["U5-2", "U9-3"],
        "cross_subject_operations": [],
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


def stable_v5_features() -> dict:
    features = current_features()
    features.pop("response_operations")
    features.pop("cross_subject_operations")
    return features


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

    def test_current_contract_is_stable_v5_and_keeps_v6_readable(self) -> None:
        self.assertEqual(len(self.features.OBSERVABLE_FEATURE_FIELDS), 17)
        self.assertEqual(len(self.features.OBSERVABLE_V5_FEATURE_FIELDS), 17)
        self.assertEqual(len(self.features.OBSERVABLE_V6_FEATURE_FIELDS), 19)
        self.assertEqual(
            self.features.OBSERVABLE_FEATURE_FIELDS,
            self.features.OBSERVABLE_V5_FEATURE_FIELDS,
        )
        self.assertIn(
            "response_operations",
            self.features.OBSERVABLE_V6_FEATURE_FIELDS,
        )
        self.assertIn(
            "cross_subject_operations",
            self.features.OBSERVABLE_V6_FEATURE_FIELDS,
        )
        self.assertNotIn(
            "response_operations",
            self.features.OBSERVABLE_FEATURE_FIELDS,
        )
        self.assertNotIn(
            "cross_subject_operations",
            self.features.OBSERVABLE_FEATURE_FIELDS,
        )
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

    def test_new_observable_options_cover_repeated_real_semantics(self) -> None:
        self.assertIn(
            "化学式组成计算",
            self.features.CALCULATION_OPERATIONS,
        )
        self.assertIn(
            "流程或关系图解析",
            self.features.GRAPH_TABLE_OPERATIONS,
        )
        self.assertIn(
            "跨学科语义或模型应用",
            self.features.RULE_FAMILIES,
        )

    def test_zero_count_groups_and_unknown_extra_fields_are_local_audit_repairs(
        self,
    ) -> None:
        item = stable_v5_features()
        item["task_groups"].append(
            {"task_type": "实验操作与探究", "count": 0}
        )
        item["condition_conditions"] = []

        normalized, actions = self.features.normalize_observable_features(item)
        validated = self.features.validate_observable_features(normalized)
        flags = self.features.observable_feature_quality_flags(
            validated,
            actions,
        )

        self.assertNotIn("condition_conditions", normalized)
        self.assertNotIn(
            {"task_type": "实验操作与探究", "count": 0},
            validated["task_groups"],
        )
        self.assertIn("structural_schema_repaired", flags)

    def test_missing_reasoning_is_rebuilt_from_existing_top_level_reasons(
        self,
    ) -> None:
        item = rating("中等题")
        item.pop("reasoning")
        item.update(
            {
                "hard_point": "需要建立一个完整方程式计算模型。",
                "why_not_lower": "不是一次透明匹配。",
                "why_not_higher": "没有多反应或分类模型。",
            }
        )

        validated = self.runtime.validate_rating_contract(item)

        self.assertEqual(
            validated["reasoning"]["core_basis"],
            "需要建立一个完整方程式计算模型。",
        )
        self.assertNotIn("hard_point", validated)
        self.assertTrue(validated["rating_schema_normalization_actions"])

    def test_malformed_coarse_reasoning_spill_is_repaired_locally(self) -> None:
        item = rating("中等题")
        item.pop("reasoning")
        item["coarse_difficulty"] = (
            '基础/中等区间（2-3档  "reasoning": {\n'
            '  "core_basis": "建立方程式计量关系并计算目标质量。'
        )
        item.update(
            {
                "hard_point": "建立方程式计量关系。",
                "why_not_lower": "不是一步直接比例。",
                "why_not_higher": "没有高级定量结构。",
            }
        )

        validated = self.runtime.validate_rating_contract(item)

        self.assertEqual(
            validated["coarse_difficulty"],
            "基础/中等区间（2-3档）",
        )
        self.assertEqual(
            validated["reasoning"]["core_basis"],
            "建立方程式计量关系并计算目标质量。",
        )
        self.assertTrue(validated["rating_schema_normalization_actions"])

    def test_text_table_operation_does_not_require_an_image_structure(self) -> None:
        item = stable_v5_features()
        item["representation_operations"] = ["图表数据→化学关系"]
        item["graph_table_operation"] = "多组比较"
        item["visual_task_structure"] = "无必要视觉信息"

        normalized, _ = self.features.normalize_observable_features(item)
        validated = self.features.validate_observable_features(normalized)

        self.assertEqual(validated["graph_table_operation"], "多组比较")
        self.assertEqual(
            validated["visual_task_structure"],
            "无必要视觉信息",
        )

    def test_formula_composition_misplaced_in_calculation_keeps_both_facts(
        self,
    ) -> None:
        item = stable_v5_features()
        item["representation_operations"] = []
        item["calculation_operations"] = ["化学符号→定量关系"]

        normalized, _ = self.features.normalize_observable_features(item)
        validated = self.features.validate_observable_features(normalized)

        self.assertEqual(
            validated["representation_operations"],
            ["化学符号→定量关系"],
        )
        self.assertEqual(
            validated["calculation_operations"],
            ["化学式组成计算"],
        )

    def test_rule_families_are_concrete_answer_operations(self) -> None:
        self.assertEqual(
            self.features.RULE_FAMILIES
            - {"其他未归类规则（仅审计）"},
            {
                "教材事实直接匹配",
                "分类或概念辨析",
                "化学用语书写",
                "化学用语含义辨析",
                "性质用途或现象判断",
                "反应关系或条件判断",
                "实验操作规范",
                "作用目的或原因解释",
                "异常失败或误差诊断",
                "图表读取或数据归纳",
                "证据推断或鉴别除杂",
                "定量关系与计算",
                "方案设计或评价",
                "新信息迁移",
                "跨学科语义或模型应用",
            },
        )

    def test_prompt_uses_chinese_enums_without_machine_code_contract(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "`rule_families`不再复刻task_type",
            prompt,
        )
        self.assertFalse(
            hasattr(self.features, "OBSERVABLE_ENUM_CODE_TO_LABEL_BY_FIELD")
        )
        for family in self.features.RULE_FAMILIES - {
            "其他未归类规则（仅审计）"
        }:
            self.assertIn(family, prompt)
        for code in ("T_FACT", "R_CAUSE", "C_EXCESS", "K_DIFFERENCE"):
            self.assertNotIn(f'"{code}"', prompt)
            self.assertNotIn(f"`{code}`", prompt)
        self.assertNotIn("无法归类，仅审计", prompt)
        self.assertNotIn("*_OTHER", prompt)
        self.assertNotIn(
            "实际需要切换的回答规则族数组，枚举与task_type相同",
            prompt,
        )

    def test_prompt_prevents_observed_retry_field_drift(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        for instruction in (
            "控制变量只能填入experiment_operation",
            "排除一个候选",
            "拐点平台或分段、流程或关系图解析只能填入`graph_table_operation`",
            "操作偏差只能填入error_analysis_operation",
            "单一比例必须写“直接比例”",
            "化学方程式不是task_type",
            "作用目的或原因解释只能填入rule_families",
            "rule_families不能填写表征转换",
            "排除多个候选解释",
            "单一分解反应必须写“单一反应”",
            "task_groups.count必须是1到20的整数",
        ):
            self.assertIn(instruction, prompt)

    def test_legacy_coarse_rule_families_normalize_without_retry(self) -> None:
        item = current_features()
        item["rule_families"] = [
            "直接事实与概念",
            "化学用语",
            "性质与反应判断",
            "实验操作与探究",
            "图表与数据",
            "证据推断",
            "定量计算",
            "方案设计与评价",
            "新信息应用",
        ]

        normalized, actions = self.features.normalize_observable_features(
            item
        )
        validated = self.features.validate_observable_features(normalized)

        self.assertEqual(
            validated["rule_families"],
            [
                "教材事实直接匹配",
                "化学用语书写",
                "性质用途或现象判断",
                "实验操作规范",
                "图表读取或数据归纳",
                "证据推断或鉴别除杂",
                "定量关系与计算",
                "方案设计或评价",
                "新信息迁移",
            ],
        )
        self.assertTrue(
            any(action["field"] == "rule_families" for action in actions)
        )

    def test_normalization_repairs_observed_cross_field_enum_drift(self) -> None:
        item = current_features()
        item["rule_families"] = ["图表读取或数据归纳", "定量关系与计算"]
        item["condition_operations"] = [
            "控制变量",
            "排除一个候选",
            "操作偏差到最终结果方向",
            "拐点平台或分段",
        ]
        item["evidence_operations"] = ["多证据共同成立"]
        item["experiment_operation"] = "无"
        item["graph_table_operation"] = "无"
        item["error_analysis_operation"] = "无误差分析"
        item["calculation_operations"] = ["单一比例"]

        normalized, actions = self.features.normalize_observable_features(
            item
        )
        validated = self.features.validate_observable_features(normalized)

        self.assertEqual(validated["condition_operations"], [])
        self.assertEqual(validated["experiment_operation"], "变量控制")
        self.assertEqual(
            validated["evidence_operations"],
            ["多证据共同成立", "排除一个候选"],
        )
        self.assertEqual(
            validated["error_analysis_operation"],
            "操作偏差到最终结果方向",
        )
        self.assertEqual(
            validated["graph_table_operation"],
            "拐点平台或分段",
        )
        self.assertEqual(validated["calculation_operations"], ["直接比例"])
        self.assertTrue(actions)

    def test_normalization_repairs_observed_task_and_count_variants(self) -> None:
        item = current_features()
        item["rule_families"] = ["反应关系或条件判断"]
        item["task_groups"] = [
            {"task_type": "化学方程式", "count": "2"},
            {"task_type": "性质与应用推断", "count": 1},
        ]

        normalized, _ = self.features.normalize_observable_features(item)
        validated = self.features.validate_observable_features(normalized)

        self.assertEqual(
            validated["task_groups"],
            [
                {"task_type": "化学用语", "count": 2},
                {"task_type": "性质与反应判断", "count": 1},
            ],
        )

    def test_normalization_repairs_rule_family_values_in_task_type(self) -> None:
        expected_task_types = {
            "教材事实直接匹配": "直接事实与概念",
            "分类或概念辨析": "直接事实与概念",
            "化学用语书写": "化学用语",
            "化学用语含义辨析": "化学用语",
            "性质用途或现象判断": "性质与反应判断",
            "反应关系或条件判断": "性质与反应判断",
            "实验操作规范": "实验操作与探究",
            "作用目的或原因解释": "实验操作与探究",
            "异常失败或误差诊断": "实验操作与探究",
            "图表读取或数据归纳": "图表与数据",
            "证据推断或鉴别除杂": "证据推断",
            "定量关系与计算": "定量计算",
            "方案设计或评价": "方案设计与评价",
            "新信息迁移": "新信息应用",
        }

        for rule_family, task_type in expected_task_types.items():
            item = current_features()
            item["rule_families"] = [rule_family]
            item["task_groups"] = [
                {"task_type": rule_family, "count": 1},
                {"task_type": task_type, "count": 2},
            ]

            with self.subTest(rule_family=rule_family):
                normalized, actions = (
                    self.features.normalize_observable_features(item)
                )
                self.assertEqual(
                    normalized["task_groups"],
                    [{"task_type": task_type, "count": 3}],
                )
                self.assertTrue(
                    any(
                        action["field"] == "task_groups.task_type"
                        and action["from"] == rule_family
                        and action["to"] == task_type
                        for action in actions
                    )
                )
                validated = self.features.validate_observable_features(
                    normalized
                )
                self.assertEqual(
                    validated["task_groups"], normalized["task_groups"]
                )

    def test_normalization_repairs_today_task_type_aliases(self) -> None:
        item = current_features()
        item["task_groups"] = [
            {"task_type": "概念辨析", "count": 1},
            {"task_type": "能量转化判断", "count": 2},
            {"task_type": "成分推断", "count": 3},
        ]

        normalized, _ = self.features.normalize_observable_features(item)

        self.assertEqual(
            normalized["task_groups"],
            [
                {"task_type": "直接事实与概念", "count": 1},
                {"task_type": "性质与反应判断", "count": 2},
                {"task_type": "证据推断", "count": 3},
            ],
        )
        self.features.validate_observable_features(normalized)

    def test_normalization_repairs_today_single_enum_aliases(self) -> None:
        item = current_features()
        item["reaction_structure"] = "单一分解反应"
        item["error_analysis_operation"] = "读数偏差到最终结果方向"

        normalized, _ = self.features.normalize_observable_features(item)

        self.assertEqual(normalized["reaction_structure"], "单一反应")
        self.assertEqual(
            normalized["error_analysis_operation"],
            "操作偏差到最终结果方向",
        )
        self.features.validate_observable_features(normalized)

    def test_normalization_repairs_rule_family_cross_field_drift(self) -> None:
        item = current_features()
        item["rule_families"] = [
            "微观粒子→化学符号",
            "化学符号→定量关系",
            "范围或边界判断",
            "微观粒子表征分析",
        ]
        item["representation_operations"] = []
        item["condition_operations"] = []

        normalized, _ = self.features.normalize_observable_features(item)

        self.assertEqual(
            normalized["rule_families"],
            [
                "化学用语书写",
                "定量关系与计算",
                "反应关系或条件判断",
                "分类或概念辨析",
            ],
        )
        self.assertEqual(
            normalized["representation_operations"],
            ["微观粒子→化学符号", "化学符号→定量关系"],
        )
        self.assertEqual(
            normalized["condition_operations"], ["范围或边界"]
        )
        self.features.validate_observable_features(normalized)

    def test_normalization_repairs_today_cross_field_operations(self) -> None:
        item = current_features()
        item["solution_topology"] = "排除多个候选解释"
        item["condition_operations"] = [
            "拐点分段",
            "差量",
            "单一守恒",
            "排除干扰条件排除",
            "反应条件判断",
        ]
        item["evidence_operations"] = [
            "范围条件筛选",
            "排除三个候选",
            "组分消元或组成不变量",
        ]
        item["graph_table_operation"] = "无"
        item["calculation_operations"] = []

        normalized, _ = self.features.normalize_observable_features(item)

        self.assertEqual(
            normalized["solution_topology"],
            "未知组分消元或组成不变量",
        )
        self.assertEqual(
            normalized["condition_operations"],
            ["干扰条件排除", "条件直接读取", "范围或边界"],
        )
        self.assertEqual(
            normalized["evidence_operations"], ["排除多个候选解释"]
        )
        self.assertEqual(
            normalized["calculation_operations"],
            ["差量", "单一守恒", "组分消元或组成不变量"],
        )
        self.assertEqual(
            normalized["graph_table_operation"], "拐点平台或分段"
        )
        self.features.validate_observable_features(normalized)

    def test_normalization_repairs_today_operation_aliases(self) -> None:
        item = current_features()
        item["representation_operations"] = [
            "微观粒子→宏观现象",
            "宏观对象→微观粒子",
        ]
        item["calculation_operations"] = [
            "质量守恒",
            "未知组分消元或组成不变量",
            "多个反应定量关系",
        ]
        item["solution_topology"] = "未知组分消元或组成不变量"

        normalized, _ = self.features.normalize_observable_features(item)

        self.assertEqual(
            normalized["representation_operations"],
            ["微观粒子→宏观含义", "宏观现象→微观粒子"],
        )
        self.assertEqual(
            normalized["calculation_operations"],
            ["单一守恒", "组分消元或组成不变量", "多反应定量关系"],
        )
        self.features.validate_observable_features(normalized)

    def test_normalization_repairs_remaining_today_field_drift(self) -> None:
        item = current_features()
        item["task_groups"] = [{"task_type": "证据推断", "count": 2}]
        item["solution_topology"] = "多证据共同成立"
        item["representation_operations"] = [
            "宏观现象→化学关系",
            "宏观现象→宏观含义",
        ]
        item["evidence_operations"] = [
            "排除干扰条件",
            "排除干扰",
        ]
        item["condition_operations"] = ["分段条件", "拐点边界"]
        item["calculation_operations"] = ["未知组成或量反推"]
        item["graph_table_operation"] = "无"

        normalized, _ = self.features.normalize_observable_features(item)

        self.assertEqual(
            normalized["solution_topology"], "未知组成或量反推"
        )
        self.assertEqual(
            normalized["representation_operations"],
            ["宏观现象→化学符号"],
        )
        self.assertEqual(
            normalized["evidence_operations"], ["多证据共同成立"]
        )
        self.assertCountEqual(
            normalized["condition_operations"],
            ["干扰条件排除", "条件切换", "范围或边界"],
        )
        self.assertEqual(
            normalized["graph_table_operation"], "拐点平台或分段"
        )
        self.assertEqual(normalized["calculation_operations"], [])
        self.features.validate_observable_features(normalized)

    def test_normalization_does_not_guess_graph_operation_from_conversion(
        self,
    ) -> None:
        item = current_features()
        item["representation_operations"] = ["图表数据→化学关系"]
        item["graph_table_operation"] = "无"

        normalized, _ = self.features.normalize_observable_features(item)

        self.assertEqual(
            normalized["graph_table_operation"],
            "其他未归类图表操作（仅审计）",
        )
        self.features.validate_observable_features(normalized)

    def test_normalization_moves_rule_family_out_of_calculation_field(
        self,
    ) -> None:
        item = current_features()
        item["task_groups"] = [{"task_type": "直接事实与概念", "count": 1}]
        item["rule_families"] = []
        item["calculation_operations"] = ["定量关系与计算"]

        normalized, _ = self.features.normalize_observable_features(item)

        self.assertEqual(normalized["calculation_operations"], [])
        self.assertEqual(normalized["rule_families"], ["定量关系与计算"])
        self.features.validate_observable_features(normalized)

    def test_historical_v6_contract_remains_readable(self) -> None:
        validated = self.runtime.validate_feature_contract(current_features())

        self.assertEqual(
            set(validated),
            set(self.runtime.OBSERVABLE_V6_FEATURE_FIELDS),
        )
        self.assertEqual(
            self.runtime.observable_feature_schema_version(validated),
            "chemistry_observable_v6",
        )

    def test_historical_v5_contract_remains_readable(self) -> None:
        historical = current_features()
        historical.pop("response_operations")
        historical.pop("cross_subject_operations")

        validated = self.runtime.validate_feature_contract(historical)

        self.assertEqual(
            set(validated),
            set(self.runtime.OBSERVABLE_V5_FEATURE_FIELDS),
        )
        self.assertEqual(
            self.runtime.observable_feature_schema_version(validated),
            "chemistry_observable_v5",
        )

    def test_new_observable_enums_are_valid_and_unknowns_are_audited(self) -> None:
        item = current_features()
        item["response_operations"] = [
            "完整命题正误辨析",
            "规范原因表达",
        ]
        item["cross_subject_operations"] = [
            "物理过程或物理量关系",
        ]
        validated = self.runtime.validate_feature_contract(item)
        metrics = self.features.derive_observable_metrics(validated)

        self.assertEqual(metrics["response_operation_count"], 2)
        self.assertEqual(metrics["cross_subject_operation_count"], 1)

        item["response_operations"] = ["泛泛理解题意"]
        normalized, actions = self.features.normalize_observable_features(item)
        validated = self.runtime.validate_feature_contract(normalized)
        flags = self.features.observable_feature_quality_flags(
            validated,
            actions,
        )
        self.assertEqual(
            validated["response_operations"],
            ["其他未归类作答操作（仅审计）"],
        )
        self.assertIn("fallback_enum:response_operations", flags)

    def test_supercurricular_chemistry_is_not_cross_subject(self) -> None:
        item = current_features()
        item["new_information_operation"] = (
            "依赖题干未给出的超纲化学知识"
        )
        item["cross_subject_operations"] = []

        validated = self.runtime.validate_feature_contract(item)
        projection = self.runtime.project_observable_to_core12(validated)

        self.assertEqual(
            projection["unfamiliar_information_transfer"],
            "完全陌生模型现场建立",
        )

    def test_response_breadth_is_audited_but_not_written_back(self) -> None:
        item = rating("基础题")
        item["features"]["task_groups"] = [
            {"task_type": "实验操作与探究", "count": 2},
            {"task_type": "化学用语", "count": 2},
        ]
        item["features"]["rule_families"] = [
            "实验操作与探究",
            "化学用语",
        ]
        item["features"]["response_operations"] = [
            "实验操作规范",
            "异常或失败原因诊断",
            "化学用语书写",
        ]
        item["features"]["parallel_task_relation"] = (
            "不同规则的独立任务"
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "分别判断实验操作、分析失败原因并书写化学式。"},
        )

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_basic_to_medium_multi_rule_breadth_candidate",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_v5_multi_rule_multitopic_boundary_writes_basic_to_medium(
        self,
    ) -> None:
        item = rating("基础题")
        item["features"] = stable_v5_features()
        item["features"].update(
            {
                "task_groups": [
                    {"task_type": "性质与反应判断", "count": 2},
                    {"task_type": "化学用语", "count": 2},
                ],
                "rule_families": [
                    "性质用途或现象判断",
                    "反应关系或条件判断",
                    "化学用语书写",
                ],
                "curriculum_topics": ["U1-2", "U4-3"],
                "parallel_task_relation": "不同规则的独立任务",
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(item, {})

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_basic_to_medium_multi_rule_multitopic",
        )
        self.assertTrue(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_v5_multi_rule_multitopic_boundary_requires_every_threshold(
        self,
    ) -> None:
        features = stable_v5_features()
        features.update(
            {
                "task_groups": [
                    {"task_type": "性质与反应判断", "count": 2},
                    {"task_type": "化学用语", "count": 2},
                ],
                "rule_families": [
                    "性质用途或现象判断",
                    "反应关系或条件判断",
                    "化学用语书写",
                ],
                "curriculum_topics": ["U1-2", "U4-3"],
            }
        )
        variants = {
            "W不足": {
                "task_groups": [
                    {"task_type": "性质与反应判断", "count": 1},
                    {"task_type": "化学用语", "count": 2},
                ]
            },
            "B不足": {
                "rule_families": [
                    "性质用途或现象判断",
                    "化学用语书写",
                ]
            },
            "T不足": {"curriculum_topics": ["U1-2"]},
        }

        for label, changes in variants.items():
            candidate = copy.deepcopy(features)
            candidate.update(changes)
            with self.subTest(label=label):
                self.assertIsNone(
                    self.runtime.observable_multi_rule_multitopic_medium_signal(
                        candidate
                    )
                )

    def test_v5_parallel_reaction_multitopic_candidate_is_audit_only(
        self,
    ) -> None:
        item = rating("基础题")
        item["features"] = stable_v5_features()
        item["features"].update(
            {
                "task_groups": [
                    {"task_type": "性质与反应判断", "count": 2},
                    {"task_type": "化学用语", "count": 2},
                ],
                "rule_families": [
                    "反应关系或条件判断",
                    "化学用语书写",
                ],
                "curriculum_topics": ["U5-2", "U6-2", "U8-2"],
                "reaction_structure": "多个并列反应",
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(item, {})

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_basic_to_medium_parallel_reaction_multitopic_candidate",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_v5_oxygen_parallel_phenomena_multitopic_writes_to_medium(
        self,
    ) -> None:
        item = rating("基础题")
        item["features"] = stable_v5_features()
        item["features"].update(
            {
                "task_groups": [
                    {"task_type": "性质与反应判断", "count": 4},
                ],
                "rule_families": ["性质用途或现象判断"],
                "curriculum_topics": ["U2-2", "U4-2", "U10-2"],
                "parallel_task_relation": "同一规则下多个对象",
                "reaction_structure": "多个并列反应",
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(item, {})

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_basic_to_medium_parallel_phenomena_multitopic",
        )
        self.assertTrue(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_v5_parallel_phenomena_without_oxygen_stays_audit_only(
        self,
    ) -> None:
        item = rating("基础题")
        item["features"] = stable_v5_features()
        item["features"].update(
            {
                "task_groups": [
                    {"task_type": "性质与反应判断", "count": 4},
                ],
                "rule_families": ["性质用途或现象判断"],
                "curriculum_topics": ["U5-2", "U6-2", "U8-2"],
                "parallel_task_relation": "同一规则下多个对象",
                "reaction_structure": "多个并列反应",
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(item, {})

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_basic_to_medium_parallel_reaction_multitopic_candidate",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_v5_parallel_reaction_multitopic_candidate_requires_all_signals(
        self,
    ) -> None:
        features = stable_v5_features()
        features.update(
            {
                "task_groups": [
                    {"task_type": "性质与反应判断", "count": 2},
                    {"task_type": "化学用语", "count": 2},
                ],
                "rule_families": [
                    "反应关系或条件判断",
                    "化学用语书写",
                ],
                "curriculum_topics": ["U5-2", "U6-2", "U8-2"],
                "reaction_structure": "多个并列反应",
            }
        )
        variants = {
            "W不足": {
                "task_groups": [
                    {"task_type": "性质与反应判断", "count": 1},
                    {"task_type": "化学用语", "count": 2},
                ]
            },
            "T不足": {"curriculum_topics": ["U5-2", "U6-2"]},
            "B过宽": {
                "rule_families": [
                    "反应关系或条件判断",
                    "化学用语书写",
                    "定量关系与计算",
                ]
            },
            "不是并列反应": {"reaction_structure": "单一反应"},
        }

        for label, changes in variants.items():
            candidate = copy.deepcopy(features)
            candidate.update(changes)
            with self.subTest(label=label):
                self.assertIsNone(
                    self.runtime.observable_parallel_reaction_multitopic_medium_candidate_signal(
                        candidate
                    )
                )

    def test_v5_high_density_evidence_writes_medium_to_hard(self) -> None:
        item = rating("中等题")
        item["features"] = stable_v5_features()
        item["features"].update(
            {
                "rule_families": [
                    "反应关系或条件判断",
                    "实验操作规范",
                    "作用目的或原因解释",
                    "图表读取或数据归纳",
                    "证据推断或鉴别除杂",
                    "定量关系与计算",
                ],
                "evidence_operations": ["多证据共同成立"],
                "calculation_operations": ["单一方程式"],
                "experiment_operation": "方案评价或补充实验",
                "experiment_task_structure": "方案设计或评价",
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(item, {})

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_medium_to_hard_high_density_evidence",
        )
        self.assertTrue(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_v5_high_density_evidence_rejects_parallel_routine_tasks(
        self,
    ) -> None:
        features = stable_v5_features()
        features.update(
            {
                "rule_families": [
                    "反应关系或条件判断",
                    "实验操作规范",
                    "作用目的或原因解释",
                    "图表读取或数据归纳",
                    "证据推断或鉴别除杂",
                    "定量关系与计算",
                ],
                "evidence_operations": ["多证据共同成立"],
                "experiment_operation": "数据归纳",
                "experiment_task_structure": "控制变量或数据归纳",
                "parallel_task_relation": "不同规则的独立任务",
                "solution_topology": "单线性常规链",
            }
        )

        self.assertIsNone(
            self.runtime.observable_high_density_evidence_hard_signal(
                features
            )
        )

    def test_v5_high_density_evidence_requires_six_rules_and_joint_evidence(
        self,
    ) -> None:
        features = stable_v5_features()
        features.update(
            {
                "rule_families": [
                    "反应关系或条件判断",
                    "实验操作规范",
                    "作用目的或原因解释",
                    "图表读取或数据归纳",
                    "证据推断或鉴别除杂",
                    "定量关系与计算",
                ],
                "evidence_operations": ["多证据共同成立"],
            }
        )
        five_rules = copy.deepcopy(features)
        five_rules["rule_families"] = five_rules["rule_families"][:5]
        no_joint_evidence = copy.deepcopy(features)
        no_joint_evidence["evidence_operations"] = ["单证据直接匹配"]

        self.assertIsNone(
            self.runtime.observable_high_density_evidence_hard_signal(
                five_rules
            )
        )
        self.assertIsNone(
            self.runtime.observable_high_density_evidence_hard_signal(
                no_joint_evidence
            )
        )

    def test_program_derives_text_length_and_explicit_subquestion_count(self) -> None:
        data = {
            "stem": "某同学完成下列实验。（1）写出现象。（2）解释原因。",
            "options": "A.甲 B.乙",
            "analysis": "这部分不应计入题面字数。",
            "sub_questions": [
                {"stem": "写出现象", "analysis": "略"},
                {"stem": "解释原因", "analysis": "略"},
            ],
        }

        metrics = self.runtime.derive_question_structure_metrics(data)

        self.assertEqual(metrics["explicit_subquestion_count"], 2)
        self.assertGreater(metrics["question_text_char_count"], 10)
        self.assertLessEqual(metrics["question_text_char_count"], 40)

    def test_cross_field_experiment_enum_reports_the_correct_destination(self) -> None:
        invalid = current_features()
        invalid["experiment_operation"] = "多仪器或多条件比较"
        invalid["experiment_task_structure"] = "多仪器或多条件比较"

        prepared = self.runtime.validate_rating_contract(
            {**rating(), "features": invalid}
        )

        self.assertEqual(
            prepared["features"]["experiment_operation"],
            "其他未归类实验操作（仅审计）",
        )
        self.assertEqual(
            prepared["features"]["experiment_task_structure"],
            "多仪器或多条件比较",
        )
        self.assertTrue(prepared["feature_normalization_actions"])
        self.assertIn(
            "fallback_enum:experiment_operation",
            prepared["feature_contract_quality_flags"],
        )

    def test_schema_repair_feedback_includes_invalid_json_and_field_hint(self) -> None:
        invalid_rating = rating()
        invalid_rating["features"]["experiment_operation"] = (
            "多仪器或多条件比较"
        )
        error = self.runtime.ChemistrySchemaError(
            "experiment_operation不在合法枚举中: '多仪器或多条件比较'；"
            "该值属于experiment_task_structure"
        )

        feedback = self.runtime.build_schema_repair_feedback(
            error,
            invalid_rating,
        )

        self.assertIn("该值属于experiment_task_structure", feedback)
        self.assertIn('"experiment_operation": "多仪器或多条件比较"', feedback)
        self.assertIn("experiment_operation描述做了什么实验认知操作", feedback)
        self.assertIn("experiment_task_structure描述实验任务怎样组织", feedback)

    def test_physics_style_normalization_repairs_observed_enum_variants(self) -> None:
        invalid = current_features()
        invalid["task_groups"] = [
            {"task_type": "误差分析", "count": 1},
            {"task_type": "实验操作与探究", "count": 1},
        ]
        invalid["rule_families"] = ["误差分析"]
        invalid["representation_operations"] = [
            "宏观名称→化学符号",
            "化学式→定量关系",
        ]
        invalid["experiment_operation"] = "方案设计与评价"
        invalid["experiment_task_structure"] = "方案设计或评价"

        normalized, actions = self.features.normalize_observable_features(
            invalid
        )
        validated = self.features.validate_observable_features(normalized)

        self.assertEqual(
            validated["task_groups"],
            [{"task_type": "实验操作与探究", "count": 2}],
        )
        self.assertEqual(
            validated["rule_families"],
            ["异常失败或误差诊断"],
        )
        self.assertEqual(
            validated["representation_operations"],
            ["宏观对象→化学符号", "化学符号→定量关系"],
        )
        self.assertTrue(actions)

    def test_normalization_repairs_field_typo_duplicates_and_cross_field_values(self) -> None:
        invalid = current_features()
        invalid["new_ininformation_operation"] = invalid.pop(
            "new_information_operation"
        )
        invalid["curriculum_topics"] = ["U5-2", "U5-2", "U9-3"]
        invalid["condition_operations"] = ["多证据共同成立"]
        invalid["evidence_operations"] = ["分类讨论"]

        normalized, actions = self.features.normalize_observable_features(
            invalid
        )
        validated = self.features.validate_observable_features(normalized)

        self.assertNotIn("new_ininformation_operation", validated)
        self.assertEqual(validated["curriculum_topics"], ["U5-2", "U9-3"])
        self.assertEqual(validated["condition_operations"], ["分类讨论"])
        self.assertEqual(validated["evidence_operations"], ["多证据共同成立"])
        self.assertTrue(actions)

    def test_normalization_moves_representation_value_out_of_calculation_field(self) -> None:
        invalid = current_features()
        invalid["representation_operations"] = [
            "化学方程式→定量关系",
        ]
        invalid["calculation_operations"] = [
            "化学符号→定量关系",
            "单一守恒",
        ]

        normalized, actions = self.features.normalize_observable_features(
            invalid
        )
        validated = self.features.validate_observable_features(normalized)

        self.assertEqual(
            validated["representation_operations"],
            ["化学方程式→定量关系", "化学符号→定量关系"],
        )
        self.assertEqual(
            validated["calculation_operations"],
            ["单一守恒", "化学式组成计算"],
        )
        self.assertTrue(
            any(
                action.get("field") == "calculation_operations"
                for action in actions
            )
        )

    def test_unknown_component_invariant_is_a_valid_observable_structure(self) -> None:
        item = current_features()
        item["longest_solution_chain"] = [
            "由酸的质量确定混合氧化物中的总氧量",
            "利用元素守恒消去未知组分比例",
            "将总氧量用于还原后的剩余固体计算",
        ]
        item["solution_topology"] = "未知组分消元或组成不变量"
        item["calculation_operations"] = [
            "单一守恒",
            "组分消元或组成不变量",
        ]

        validated = self.runtime.validate_feature_contract(item)
        projection = self.runtime.project_observable_to_core12(validated)

        self.assertEqual(
            validated["solution_topology"],
            "未知组分消元或组成不变量",
        )
        self.assertEqual(
            projection["calculation_model"],
            "多重守恒、差量、联立或分类",
        )

    def test_v6_requires_invariant_topology_and_operation_to_agree(self) -> None:
        topology_only = current_features()
        topology_only["solution_topology"] = (
            "未知组分消元或组成不变量"
        )
        with self.assertRaisesRegex(
            self.runtime.ChemistrySchemaError,
            "solution_topology.*calculation_operations",
        ):
            self.runtime.validate_feature_contract(topology_only)

        operation_only = current_features()
        operation_only["calculation_operations"].append(
            "组分消元或组成不变量"
        )
        with self.assertRaisesRegex(
            self.runtime.ChemistrySchemaError,
            "calculation_operations.*solution_topology",
        ):
            self.runtime.validate_feature_contract(operation_only)

    def test_unknown_enum_still_fails_after_safe_normalization(self) -> None:
        invalid = current_features()
        invalid["representation_operations"] = ["看起来很难的转换"]

        prepared = self.runtime.validate_rating_contract(
            {
                **rating(),
                "features": invalid,
            }
        )

        self.assertEqual(
            prepared["features"]["representation_operations"],
            ["其他未归类表征操作（仅审计）"],
        )
        self.assertIn(
            "fallback_enum:representation_operations",
            prepared["feature_contract_quality_flags"],
        )

    def test_internal_fallbacks_are_audited_and_excluded_from_counts(self) -> None:
        coded = stable_v5_features()
        coded["task_groups"] = [
            {"task_type": "直接事实与概念", "count": 2},
            {"task_type": "模型自造任务", "count": 5},
        ]
        coded["rule_families"] = ["教材事实直接匹配", "模型自造规则"]
        coded["condition_operations"] = ["模型自造条件"]
        coded["calculation_operations"] = ["模型自造计算"]

        normalized, actions = self.features.normalize_observable_features(coded)
        validated = self.features.validate_observable_features(normalized)
        metrics = self.features.derive_observable_metrics(validated)
        flags = self.features.observable_feature_quality_flags(
            validated,
            actions,
        )

        self.assertEqual(metrics["effective_task_count"], 2)
        self.assertEqual(metrics["task_group_count"], 1)
        self.assertEqual(metrics["rule_family_count"], 1)
        self.assertNotIn(
            "其他未归类计算操作（仅审计）",
            metrics["advanced_calculation_operations"],
        )
        self.assertIn("fallback_enum:task_groups.task_type", flags)
        self.assertIn("fallback_enum:rule_families", flags)
        self.assertIn("fallback_enum:condition_operations", flags)
        self.assertIn("fallback_enum:calculation_operations", flags)

    def test_open_list_fields_use_internal_fallback_without_schema_retry(self) -> None:
        item = current_features()
        item["response_operations"] = ["模型自造主观作答动作"]
        item["cross_subject_operations"] = ["模型自造跨学科动作"]

        normalized, actions = self.features.normalize_observable_features(item)
        validated = self.features.validate_observable_features(normalized)
        flags = self.features.observable_feature_quality_flags(
            validated,
            actions,
        )
        metrics = self.features.derive_observable_metrics(validated)

        self.assertEqual(
            validated["response_operations"],
            ["其他未归类作答操作（仅审计）"],
        )
        self.assertEqual(
            validated["cross_subject_operations"],
            ["其他未归类跨学科操作（仅审计）"],
        )
        self.assertIn("fallback_enum:response_operations", flags)
        self.assertIn("fallback_enum:cross_subject_operations", flags)
        self.assertEqual(metrics["response_operation_count"], 0)
        self.assertEqual(metrics["cross_subject_operation_count"], 0)

    def test_missing_calculation_evidence_is_quality_flag_not_schema_error(self) -> None:
        item = stable_v5_features()
        item["task_groups"] = [{"task_type": "定量计算", "count": 1}]
        item["calculation_operations"] = []

        normalized, actions = self.features.normalize_observable_features(item)
        validated = self.features.validate_observable_features(normalized)
        flags = self.features.observable_feature_quality_flags(
            validated,
            actions,
        )

        self.assertEqual(validated["calculation_operations"], [])
        self.assertIn("incomplete_calculation_evidence", flags)

    def test_ambiguous_graph_consistency_uses_internal_fallback(self) -> None:
        item = stable_v5_features()
        item["representation_operations"] = ["图表数据→化学关系"]
        item["graph_table_operation"] = "无"
        item["visual_task_structure"] = "无必要视觉信息"

        normalized, actions = self.features.normalize_observable_features(item)
        validated = self.features.validate_observable_features(normalized)
        flags = self.features.observable_feature_quality_flags(
            validated,
            actions,
        )

        self.assertEqual(
            validated["graph_table_operation"],
            "其他未归类图表操作（仅审计）",
        )
        self.assertEqual(
            validated["visual_task_structure"],
            "无必要视觉信息",
        )
        self.assertNotEqual(validated["graph_table_operation"], "直接读数")
        self.assertIn("fallback_enum:graph_table_operation", flags)
        self.assertNotIn("fallback_enum:visual_task_structure", flags)

    def test_semantic_feature_repair_blocks_writeback(self) -> None:
        item = rating("拔高题")
        item["features"] = stable_v5_features()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "建立总量关系",
                    "利用差量确定反应量",
                    "反推未知组成",
                    "把中间量代入后一反应",
                    "联立并核验最终结果",
                ],
                "solution_topology": "未知组成或量反推",
                "reaction_structure": "产物进入后一反应",
                "calculation_operations": [
                    "差量",
                    "多反应定量关系",
                ],
            }
        )
        item["feature_schema_repair_kind"] = "semantic"

        result = self.runtime.postprocess_chemistry_difficulty(item, {})

        self.assertIn(
            "semantic_schema_repaired",
            result["feature_contract_quality_flags"],
        )
        self.assertFalse(result["writeback_eligible"])
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_feature_repair_merge_preserves_first_rating_decision(self) -> None:
        original = rating("中等题")
        repaired = {
            "features": {
                **stable_v5_features(),
                "calculation_operations": ["差量"],
            },
            "difficulty_level": "压轴题",
            "reasoning": {
                "core_basis": "重试时改写了理由",
            },
        }

        merged = self.runtime.merge_feature_repair_candidate(
            original,
            repaired,
        )

        self.assertEqual(merged["difficulty_level"], "中等题")
        self.assertEqual(merged["reasoning"], original["reasoning"])
        self.assertEqual(
            merged["coarse_difficulty"],
            original["coarse_difficulty"],
        )
        self.assertEqual(
            merged["features"]["calculation_operations"],
            ["差量"],
        )

    def test_fallback_quality_blocks_automatic_teacher_writeback(self) -> None:
        item = rating("拔高题")
        item["features"] = stable_v5_features()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "建立总量关系",
                    "利用差量确定反应量",
                    "反推未知组成",
                    "把中间量代入后一反应",
                    "联立并核验最终结果",
                ],
                "solution_topology": "未知组成或量反推",
                "reaction_structure": "产物进入后一反应",
                "condition_operations": [
                    "其他未归类条件操作（仅审计）"
                ],
                "calculation_operations": ["差量", "多反应定量关系"],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(item, {})

        self.assertIsNotNone(
            result["teacher_distribution_guard_candidate_action"]
        )
        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )
        self.assertIn(
            "fallback_enum:condition_operations",
            result["feature_contract_quality_flags"],
        )

    def test_rating_records_normalization_actions_without_schema_retry(self) -> None:
        item = rating()
        item["features"]["representation_operations"] = [
            "宏观物质→化学符号"
        ]

        result = self.runtime.validate_rating_contract(item)

        self.assertEqual(
            result["features"]["representation_operations"],
            ["宏观对象→化学符号"],
        )
        self.assertTrue(result["feature_normalization_actions"])

    def test_generic_repair_feedback_lists_allowed_values_for_bad_enum(self) -> None:
        invalid_rating = rating()
        error = self.runtime.ChemistrySchemaError(
            "representation_operations包含非法枚举: ['未知转换']"
        )

        feedback = self.runtime.build_schema_repair_feedback(
            error,
            invalid_rating,
        )

        self.assertIn("representation_operations只能从", feedback)
        self.assertIn("宏观对象→化学符号", feedback)
        self.assertNotIn("X_OBJECT_TO_SYMBOL", feedback)
        self.assertNotIn("其他未归类表征操作", feedback)

    def test_model_feature_repairs_are_semantic_after_local_normalization(self) -> None:
        self.assertEqual(
            self.runtime.classify_feature_schema_repair(
                self.runtime.ChemistrySchemaError(
                    "task_groups.count必须是1到20的整数"
                )
            ),
            "semantic",
        )
        self.assertEqual(
            self.runtime.classify_feature_schema_repair(
                self.runtime.ChemistrySchemaError(
                    "存在图表转换时graph_table_operation不能为无"
                )
            ),
            "semantic",
        )
        self.assertEqual(
            self.runtime.classify_feature_schema_repair(
                self.runtime.ChemistrySchemaError(
                    "可观测特征字段集不匹配; "
                    "missing=['experiment_operation']; extra=[]"
                )
            ),
            "semantic",
        )
        self.assertEqual(
            self.runtime.classify_feature_schema_repair(
                self.runtime.ChemistrySchemaError(
                    "rule_families不能为空"
                )
            ),
            "semantic",
        )
        self.assertEqual(
            self.runtime.classify_feature_schema_repair(
                self.runtime.ChemistrySchemaError(
                    "longest_solution_chain必须包含1到12个必要化学决策步骤"
                )
            ),
            "semantic",
        )

    def test_schema_retry_audit_preserves_first_and_accepted_candidates(self) -> None:
        first = rating("基础题")
        first["features"] = stable_v5_features()
        accepted = copy.deepcopy(first)
        accepted["features"]["calculation_operations"] = ["差量"]
        candidates = [
            {
                "attempt": 0,
                "repair_mode": "full_rating",
                "repair_kind": "semantic",
                "candidate": copy.deepcopy(first),
                "accepted": False,
                "error": "字段语义冲突",
            },
            {
                "attempt": 1,
                "repair_mode": "features_only",
                "repair_kind": "semantic",
                "candidate": {"features": accepted["features"]},
                "accepted": True,
                "error": "",
            },
        ]

        audit = self.runtime.build_schema_retry_audit(
            first_candidate=first,
            accepted_candidate=accepted,
            schema_candidates=candidates,
            schema_retry_count=1,
            repair_kinds=["semantic"],
        )

        self.assertEqual(audit["first_attempt_level"], "基础题")
        self.assertEqual(audit["accepted_attempt_level"], "基础题")
        self.assertFalse(audit["schema_retry_changed_level"])
        self.assertTrue(audit["schema_retry_changed_features"])
        self.assertEqual(audit["schema_repair_mode"], "features_only")
        self.assertEqual(audit["schema_repair_kind"], "semantic")
        self.assertEqual(
            audit["difficulty_rating_first_attempt"]["difficulty_level"],
            "基础题",
        )

    def test_process_retry_only_replaces_features_and_freezes_first_judgment(self) -> None:
        first = rating("基础题")
        first["features"] = stable_v5_features()
        first["features"].pop("calculation_operations")
        repaired_features = stable_v5_features()
        calls = []

        async def fake_call(*args, **kwargs):
            features_only = bool(kwargs.get("features_only_repair"))
            calls.append(features_only)
            candidate = (
                {"features": copy.deepcopy(repaired_features)}
                if features_only
                else copy.deepcopy(first)
            )
            return (
                candidate,
                json.dumps(candidate, ensure_ascii=False),
                0.01,
                10,
                5,
                15,
                {
                    "image_input_used": False,
                    "question_text_char_count": 8,
                    "explicit_subquestion_count": 1,
                },
            )

        async def run_case(output_path: str, error_path: str) -> None:
            await self.runtime.process_single_question(
                {"question_id": "schema-repair-1", "stem": "判断水的组成。"},
                object(),
                asyncio.Semaphore(1),
                output_path,
                error_path,
                retries=1,
                timeout_sec=1,
            )

        original_call = self.runtime.call_model_with_cache
        original_schema_retries = self.runtime.MAX_SCHEMA_RETRIES
        writeback_names = (
            "CHEMISTRY_ENABLE_LEVEL_WRITEBACK",
            "CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD",
            "CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK",
            "CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS",
            "CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK",
        )
        original_writebacks = {
            name: getattr(self.runtime, name) for name in writeback_names
        }
        try:
            self.runtime.call_model_with_cache = fake_call
            self.runtime.MAX_SCHEMA_RETRIES = 2
            for name in writeback_names:
                setattr(self.runtime, name, False)
            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = str(Path(tmpdir) / "output.jsonl")
                error_path = str(Path(tmpdir) / "errors.jsonl")
                asyncio.run(run_case(output_path, error_path))
                row = json.loads(
                    Path(output_path).read_text(encoding="utf-8").strip()
                )
                self.assertFalse(Path(error_path).exists())
        finally:
            self.runtime.call_model_with_cache = original_call
            self.runtime.MAX_SCHEMA_RETRIES = original_schema_retries
            for name, value in original_writebacks.items():
                setattr(self.runtime, name, value)

        self.assertEqual(calls, [False, True])
        self.assertEqual(row["schema_retry_count"], 1)
        self.assertEqual(row["schema_repair_mode"], "features_only")
        self.assertEqual(row["first_attempt_level"], "基础题")
        self.assertEqual(row["accepted_attempt_level"], "基础题")
        self.assertFalse(row["schema_retry_changed_level"])
        self.assertTrue(row["schema_retry_changed_features"])
        self.assertEqual(
            row["difficulty_rating_raw"]["reasoning"],
            first["reasoning"],
        )
        self.assertEqual(
            row["difficulty_rating_first_attempt"]["reasoning"],
            first["reasoning"],
        )

    def test_historical_nineteen_field_v4_remains_readable(self) -> None:
        historical = copy.deepcopy(current_features())
        historical.pop("response_operations")
        historical.pop("cross_subject_operations")
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

        self.assertEqual(result["feature_schema_version"], "chemistry_observable_v6")
        self.assertEqual(result["postprocess_candidate_actions"], [])
        self.assertEqual(
            result["observable_metrics"]["explicit_subquestion_count"],
            1,
        )
        self.assertGreater(
            result["observable_metrics"]["question_text_char_count"],
            0,
        )

    def test_prompt_documents_task_granularity_without_ra_fields(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("17项可观测特征协议", prompt)
        self.assertNotIn('"direct_retrieval_task_count"', prompt)
        self.assertNotIn('"rule_application_task_count"', prompt)
        self.assertIn("非重复有效化学任务", prompt)
        self.assertIn("同一规则下多个对象", prompt)
        self.assertIn("浏览独立选项", prompt)
        self.assertIn("多个独立选项或小问可以增加任务量，但不能依次累计成长链", prompt)
        self.assertNotIn("response_operations", prompt)
        self.assertNotIn("cross_subject_operations", prompt)

    def test_prompt_decouples_task_count_from_easy_level(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "任何单个特征均不能单独决定等级",
            prompt,
        )
        self.assertIn("多幅候选图片不自动增加难度", prompt)
        self.assertIn("仍可为送分题", prompt)
        self.assertIn("固定分类规则", prompt)

    def test_prompt_keeps_dense_linear_chain_as_hard_candidate(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("高密度线性综合链", prompt)
        self.assertIn("方法熟悉或主线线性不能单独否决拔高", prompt)

    def test_prompt_separates_experiment_operation_and_structure(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("实际完成的实验认知操作", prompt)
        self.assertIn("experiment_task_structure回答“任务怎样组织”", prompt)
        self.assertIn(
            'experiment_operation="基础操作或读数"',
            prompt,
        )
        self.assertIn(
            'experiment_task_structure="多仪器或多条件比较"',
            prompt,
        )

    def test_prompt_separates_distinct_experiment_rules_and_invariant_elimination(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        for anchor in (
            "同属实验操作与探究不等于使用同一具体规则",
            "试剂作用、失败原因、性质用途和操作目的",
            "未知组分消元或组成不变量",
            "calculation_operations不得填写表征转换值",
            "化学符号→定量关系属于representation_operations",
        ):
            self.assertIn(anchor, prompt)

    def test_invariant_elimination_can_form_hard_to_final_audit_candidate(self) -> None:
        item = rating("拔高题")
        item["features"].update(
            {
                "longest_solution_chain": [
                    "由酸的质量确定混合氧化物中的总氧量",
                    "利用元素守恒消去未知组分比例",
                    "将总氧量用于还原阶段",
                    "计算还原前后的质量变化",
                    "核验剩余固体质量与组分无关",
                ],
                "solution_topology": "未知组分消元或组成不变量",
                "reaction_structure": "多个并列反应",
                "calculation_operations": [
                    "单一守恒",
                    "组分消元或组成不变量",
                ],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "未知比例的混合氧化物经酸溶和还原后求剩余质量。"},
        )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_hard_to_final_deep_quantitative_chain",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )
        self.assertEqual(result["postprocess_actions"], [])


if __name__ == "__main__":
    unittest.main()
