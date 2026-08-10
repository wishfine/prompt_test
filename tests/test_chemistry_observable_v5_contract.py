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

    def test_cross_field_experiment_enum_reports_the_correct_destination(self) -> None:
        invalid = current_features()
        invalid["experiment_operation"] = "多仪器或多条件比较"
        invalid["experiment_task_structure"] = "多仪器或多条件比较"

        with self.assertRaisesRegex(
            self.runtime.ChemistrySchemaError,
            "该值属于experiment_task_structure",
        ):
            self.runtime.validate_feature_contract(invalid)

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
        self.assertEqual(validated["rule_families"], ["实验操作与探究"])
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
        self.assertEqual(validated["calculation_operations"], ["单一守恒"])
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

    def test_unknown_enum_still_fails_after_safe_normalization(self) -> None:
        invalid = current_features()
        invalid["representation_operations"] = ["看起来很难的转换"]

        with self.assertRaisesRegex(
            self.runtime.ChemistrySchemaError,
            "representation_operations",
        ):
            self.runtime.validate_feature_contract(invalid)

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

    def test_prompt_decouples_task_count_from_easy_level(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("W只记录任务量事实，不设置最低难度", prompt)
        self.assertIn("W=4仍可判为送分题", prompt)
        self.assertIn("固定分类规则", prompt)

    def test_prompt_keeps_dense_linear_chain_as_hard_candidate(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("高密度线性综合链", prompt)
        self.assertIn("方法熟悉或主线线性不能单独否决拔高", prompt)

    def test_prompt_separates_experiment_operation_and_structure(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("experiment_operation回答“实际做了什么操作”", prompt)
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


if __name__ == "__main__":
    unittest.main()
