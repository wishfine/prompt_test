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

    def test_prompt_final_path_has_priority_over_hard_dense_path(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "先检查压轴题的两条路径；若压轴路径成立，不得再用拔高题的高密度综合链截停",
            prompt,
        )
        self.assertIn(
            "反应、图表、实验、计算、证据或条件中至少两类共同参与",
            prompt,
        )
        self.assertNotIn(
            "反应、实验、图表、计算、证据或条件中至少三类共同参与",
            prompt,
        )

    def test_prompt_final_definition_keeps_deep_linear_path(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "压轴题有“复杂模型耦合”和“单主线深定量”两条并列路径",
            prompt,
        )
        self.assertIn(
            "仅当各任务可沿显性节点分别解决，且不满足单主线深定量路径时",
            prompt,
        )
        self.assertIn(
            "单主线深定量路径已成立时，即使主线清晰，也按压轴题比较",
            prompt,
        )

    def test_prompt_does_not_double_count_a_supplied_equation(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "核验题干已给出的熟悉方程式",
            prompt,
        )
        self.assertIn(
            "不得重复计作自主书写方程式或新增一个任务",
            prompt,
        )

    def test_prompt_treats_difference_method_as_a_hard_boundary(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "差量是决定性建模方法时，应进入拔高题比较",
            prompt,
        )
        self.assertIn(
            "不得仅因差量关系属于熟悉方法而停在中等题",
            prompt,
        )

    def test_prompt_does_not_merge_distinct_facts_by_answer_form(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "同一种作答形式不等于同一条具体化学命题",
            prompt,
        )
        self.assertIn(
            "科学家成就、元素缺乏症、性质用途或实验现象",
            prompt,
        )

    def test_prompt_keeps_heterogeneous_subjective_breadth_above_basic(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "四项以上非重复任务",
            prompt,
        )
        self.assertIn(
            "规范现象、操作目的、失败原因、化学用语书写或含义解释",
            prompt,
        )

    def test_prompt_names_decisive_hard_methods_without_chain_inflation(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "反应后完整体系质量",
            prompt,
        )
        self.assertIn(
            "未知组分消元或组成不变量",
            prompt,
        )
        self.assertIn(
            "由结论反推操作或控制变量方案",
            prompt,
        )

    def test_prompt_example_10_contains_five_real_decisions(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "由盐酸用量建立总碱量关系→由固体增重确定吸收CO₂量→"
            "将CO₂量换算为已变质NaOH量→由总量扣出未变质NaOH量→求质量比",
            prompt,
        )

    def test_prompt_example_12_separates_three_symbolic_response_levels(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "直接识别名称或从给定选项认出符号、化学式",
            prompt,
        )
        self.assertIn(
            "自主写出一个元素符号、离子符号或由化合价推出一个化学式",
            prompt,
        )
        self.assertIn(
            "元素符号、离子符号、化合价、化学式、方程式或数字含义中的多类规则",
            prompt,
        )
        self.assertIn(
            "多个空都重复同一种符号书写、化合价推式或数字含义规则时仍按同一规则处理",
            prompt,
        )

    def test_prompt_example_23_separates_proposition_object_order_and_images(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("同一命题的不同措辞", prompt)
        self.assertIn("同一固定规则判断多个对象", prompt)
        self.assertIn("前者/后者或先后顺序条件", prompt)
        self.assertIn("多图独立不同规则判断", prompt)
        self.assertIn(
            "同一透明分类规则判断多个对象时仍可为送分题",
            prompt,
        )
        self.assertIn(
            "多个对象分别需要核验不同教材事实",
            prompt,
        )

    def test_prompt_controlled_breadth_distinguishes_subjective_responses(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "多处规范原因、作用、目的或失败诊断",
            prompt,
        )
        self.assertIn(
            "单个主观回答只改变作答形式，不自动升档",
            prompt,
        )

    def test_prompt_final_paths_do_not_require_unrelated_extra_signals(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "不得再追加陌生信息、跨单元、多阶段实验或分类讨论作为必要条件",
            prompt,
        )
        self.assertIn(
            "定性证据网络不以存在定量计算为必要条件",
            prompt,
        )
        self.assertIn(
            "只有一个最终求解目标或主线清晰不能否决该特殊压轴口径",
            prompt,
        )

    def test_hard_reason_must_audit_both_final_paths(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "若最终判为拔高题，why_not_higher必须分别核对两条压轴路径",
            prompt,
        )
        self.assertIn(
            "单主线深定量路径具体缺少哪一项",
            prompt,
        )
        self.assertIn(
            "不得只写“缺少深度耦合、多模块或多阶段”",
            prompt,
        )

    def test_repeated_single_conservation_stays_hard_counterexample(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "多次重复使用同一种元素守恒",
            prompt,
        )
        self.assertIn(
            "仍是同一个清晰模型中的重复应用",
            prompt,
        )
        self.assertIn(
            "不能仅因链长达到5步或反应数量较多判为压轴题",
            prompt,
        )

    def test_prompt_lists_complete_new_information_enum_and_bare_topics(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("依赖题干未给出的超纲化学知识", prompt)
        self.assertIn(
            "curriculum_topics数组只能输出裸编码",
            prompt,
        )
        self.assertIn("`U3-2`（原子结构）", prompt)
        self.assertNotIn("U3-2原子结构", prompt)

    def test_topic_name_suffix_is_safely_normalized_to_bare_code(self) -> None:
        item = production_features()
        item["curriculum_topics"] = [
            "U3-2原子结构",
            "U3-3（元素）",
        ]

        normalized, actions = self.features.normalize_observable_features(
            item
        )
        validated = self.features.validate_observable_features(normalized)

        self.assertEqual(
            validated["curriculum_topics"],
            ["U3-2", "U3-3"],
        )
        self.assertTrue(
            any(
                action["field"] == "curriculum_topics"
                for action in actions
            )
        )

    def test_mismatched_topic_name_is_not_silently_normalized(self) -> None:
        item = production_features()
        item["curriculum_topics"] = ["U3-2元素"]

        normalized, _ = self.features.normalize_observable_features(item)

        with self.assertRaisesRegex(ValueError, "curriculum_topics"):
            self.features.validate_observable_features(normalized)

    def test_prompt_does_not_require_model_to_name_derived_span(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "课程跨度与并列/耦合术语由程序派生，理由不强制自行命名",
            prompt,
        )
        self.assertNotIn("若任务彼此独立，必须明确写“跨单元并列”", prompt)

    def test_prompt_declares_runtime_shape_limits(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "longest_solution_chain包含1至12步，每步1至80字且不得重复",
            prompt,
        )
        self.assertIn(
            "task_groups包含1至12组，task_type不得重复",
            prompt,
        )

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
        item = hard_rating()
        # 隔离“深定量压轴链”这条独立路径，本测试只验证
        # 高密度多问规则自身仍要求四个显式小问。
        item["features"]["solution_topology"] = "单线性常规链"
        result = self.runtime.postprocess_chemistry_difficulty(
            item,
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

    def test_v5_deep_quantitative_chain_writes_back_to_final(self) -> None:
        result = self.runtime.postprocess_chemistry_difficulty(
            hard_rating(),
            {
                "stem": "多阶段反应网络中使用多反应定量关系。",
                "sub_questions": [
                    {"stem": "任务1"},
                    {"stem": "任务2"},
                    {"stem": "任务3"},
                ],
            },
        )

        self.assertEqual(result["difficulty_level"], "压轴题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "teacher_hard_to_final_deep_quantitative_chain",
        )

    def test_v5_deep_quantitative_chain_requires_a_reaction_task(self) -> None:
        item = hard_rating()
        item["features"]["reaction_structure"] = "无反应任务"

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {
                "stem": "饱和溶液析晶中的质量差与组成计算。",
                "sub_questions": [
                    {"stem": "任务1"},
                    {"stem": "任务2"},
                    {"stem": "任务3"},
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
