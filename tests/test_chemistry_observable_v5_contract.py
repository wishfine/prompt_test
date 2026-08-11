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

    def test_rule_families_are_concrete_answer_operations(self) -> None:
        self.assertEqual(
            self.features.RULE_FAMILIES,
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
            },
        )

    def test_prompt_exposes_only_the_concrete_rule_family_enum(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "`rule_families`不再复刻task_type",
            prompt,
        )
        for family in self.features.RULE_FAMILIES:
            self.assertIn(f"- {family}", prompt)
        self.assertNotIn(
            "实际需要切换的回答规则族数组，枚举与task_type相同",
            prompt,
        )

    def test_prompt_prevents_observed_retry_field_drift(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        for instruction in (
            "控制变量只能填入experiment_operation",
            "排除候选只能填入evidence_operations",
            "拐点、平台或分段只能填入graph_table_operation",
            "操作偏差只能填入error_analysis_operation",
            "单一比例必须写成直接比例",
            "化学方程式不是task_type",
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

    def test_new_observable_enums_are_strict_and_auditable(self) -> None:
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
        with self.assertRaisesRegex(
            self.runtime.ChemistrySchemaError,
            "response_operations",
        ):
            self.runtime.validate_feature_contract(item)

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
        self.assertIn("不同化学命题或不同作答目标", prompt)
        self.assertIn("同一规则只表示B不增加", prompt)
        self.assertIn("独立选项不增加D", prompt)
        self.assertNotIn("response_operations", prompt)
        self.assertNotIn("cross_subject_operations", prompt)

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

        self.assertEqual(result["difficulty_level"], "压轴题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_hard_to_final_deep_quantitative_chain",
        )
        self.assertTrue(
            result["teacher_distribution_guard_writeback_applied"]
        )


if __name__ == "__main__":
    unittest.main()
