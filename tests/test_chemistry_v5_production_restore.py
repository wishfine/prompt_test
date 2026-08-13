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

    def test_prompt_absorbs_reviewed_branch_case_anchors(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        for anchor in (
            "【Case 4：纯净物/混合物直接分类】",
            "【Case 9：四种物质的燃烧现象】",
            "【Case 8：读取指示剂表格后判断溶液】",
            "【Case 9：反应关系网络逐项验证】",
            "【Case 7：锌与两种盐溶液反应后的滤液滤渣】",
        ):
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, prompt)

    def test_absorbed_branch_guidance_keeps_seventeen_field_contract(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "`features`必须且只能包含以下17项",
            prompt,
        )
        self.assertNotIn("features中的30项", prompt)
        self.assertNotIn("30个细粒度特征", prompt)
        self.assertNotIn("`knowledge_distribution`", prompt)
        self.assertNotIn("`chemical_object_distribution`", prompt)
        self.assertNotIn("`step_count`", prompt)

    def test_prompt_final_path_has_priority_over_hard_dense_path(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "若候选等级为拔高题或压轴题，优先核验4/5边界",
            prompt,
        )
        self.assertIn(
            "任一路径成立后，不得再追加",
            prompt,
        )
        self.assertNotIn(
            "反应、实验、图表、计算、证据或条件中至少三类共同参与",
            prompt,
        )

    def test_prompt_handles_endpoint_reasoning_without_inventing_difficulty(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'difficulty_level="送分题"时，why_not_lower固定写“已是最低档，无更低档可比较”',
            prompt,
        )
        self.assertIn(
            'difficulty_level="压轴题"时，why_not_higher固定写“已是最高档，无更高档可比较”',
            prompt,
        )
        self.assertIn(
            "无额外难点，主要要求为",
            prompt,
        )
        self.assertIn(
            "不得为了填满字段虚构难点",
            prompt,
        )

    def test_prompt_uses_adjacent_boundary_review_without_global_final_priming(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        validation = prompt[
            prompt.index("## 五、定档前校验") :
            prompt.index("## 六、输出要求")
        ]

        self.assertIn("初定等级后只核验其相邻边界", validation)
        self.assertIn(
            "若候选等级为拔高题或压轴题，优先核验4/5边界",
            validation,
        )
        self.assertNotIn("先检查压轴两条路径并复核4/5边界", validation)

    def test_prompt_does_not_use_chain_counts_as_level_definitions(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        level_section = prompt[
            prompt.index("## 二、教师五档难度") :
            prompt.index("## 三、教师相邻边界例题")
        ]
        boundary_section = prompt[
            prompt.index("## 三、教师相邻边界例题") :
            prompt.index("## 四、17项可观测特征协议")
        ]

        self.assertNotIn("2至4个必要化学决策", level_section)
        self.assertNotIn("5个以上必要化学决策", level_section)
        self.assertNotIn("约5至6个以上必要决策", boundary_section)
        self.assertIn("较深的连续必要链", level_section)

    def test_prompt_calibrates_common_and_deep_chain_splitting(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        section = prompt[
            prompt.index("### 1. longest_solution_chain") :
            prompt.index("### 2. task_groups")
        ]

        for anchor in (
            "直接识记或透明分类：1步",
            "题给关系→应用该关系：2步",
            "标准方程式计算：确定反应关系→建立计量关系→求目标量",
            "误差分析：偏差→实际量→最终结果",
            "不得合并成笼统的“分析并计算”",
            "阶段判断、中间量或剩余量确定、后一反应关系",
        ):
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, section)

    def test_prompt_preserves_sequential_and_quantitative_coupling(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "前一阶段的产物、中间量或剩余量决定后一反应时，不得填写“多个并列反应”",
            prompt,
        )
        self.assertIn(
            "多个方程式共享未知量、中间量或总量关系时，还应记录“多反应定量关系”",
            prompt,
        )
        self.assertIn(
            "借助守恒或前后阶段不变量消去未知组分时，应记录“组分消元或组成不变量”",
            prompt,
        )

    def test_prompt_adds_single_value_tie_breaks_and_evidence_entry(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        for anchor in (
            "多阶段定量探究 > 方案评价或补充实验 > 方案设计 > 数据归纳 > 现象解释 > 变量控制 > 基础操作或读数 > 无",
            "方案设计或评价 > 操作偏差因果链 > 控制变量或数据归纳 > 多仪器或多条件比较 > 名称或单点规范匹配 > 无实验判断",
            "共享装置流程或图表模型 > 多图独立不同规则判断 > 多图独立同规则识别 > 单图直接识别 > 无必要视觉信息",
            "要求根据现象、实验结果、数据或事实去证明、排除、鉴别或确定某个化学结论或候选解释",
            "普通题干已知条件用于常规计算或直接应用，不因此记录evidence_operations",
        ):
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, prompt)

    def test_prompt_resolves_symbol_and_carbon_case_boundaries(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "自主解释一个化学符号中指定位置数字的含义，或根据规则写出一个规范化学符号",
            prompt,
        )
        self.assertIn(
            "从选项中直接辨认一个熟悉位置的数字含义仍可为送分题",
            prompt,
        )
        self.assertIn(
            "性质判断、反应关系、检验方法和转化关系之间切换",
            prompt,
        )

    def test_prompt_names_feature_container_shapes_explicitly(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "`longest_solution_chain`按自由文本数组输出，`task_groups`按规定对象结构输出",
            prompt,
        )
        self.assertNotIn("自由文本链和任务对象按", prompt)

    def test_prompt_case_ablation_keeps_only_unique_representatives(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        level_cases = prompt[
            prompt.index("## 二、教师五档难度") :
            prompt.index("## 三、教师相邻边界例题")
        ]
        boundary_cases = prompt[
            prompt.index("## 三、教师相邻边界例题") :
            prompt.index("## 四、17项可观测特征协议")
        ]

        self.assertEqual(prompt.count("## 三、教师相邻边界例题"), 1)
        self.assertEqual(level_cases.count("【Case "), 31)
        self.assertEqual(boundary_cases.count("【Case "), 27)
        for retained in (
            "【Case 2：量筒直接读数】",
            "【Case 9：四种物质的燃烧现象】",
            "【Case 8：读取指示剂表格后判断溶液】",
            "【Case 4：多处规范实验表达】",
            "【Case 5：异质实验规则切换】",
            "【Case 11：元素周期表、粒子结构与化学用语联合】",
            "【Case 9：反应关系网络逐项验证】",
            "【Case 5：酸碱盐混合后的废液判断】",
        ):
            with self.subTest(retained=retained):
                self.assertIn(retained, level_cases)
        for removed in (
            "【Case 3：空气组成直接识记】",
            "【Case 6：多种物质性质用途】",
            "【Case 1：量筒或天平误差因果链】",
            "【Case 1：差量法决定分解比例】",
            "【Case 1：循环工业流程与定量误差】",
            "【Case 39：同一命题 vs 跨章节辨认】",
            "【Case 59：高难常规计算 vs 复杂分段反应网络】",
        ):
            with self.subTest(removed=removed):
                self.assertNotIn(removed, prompt)

    def test_prompt_keeps_transparent_rule_separate_from_object_facts(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "所有候选答案都可由同一个显性判据直接推出",
            prompt,
        )
        self.assertIn(
            "逐项查看多个候选，也仍是同一透明规则的机械筛选",
            prompt,
        )
        self.assertIn(
            "共同判据不足以推出答案、必须分别回忆对象特有事实",
            prompt,
        )

    def test_prompt_resolves_remaining_adjacent_case_conflicts(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        for anchor in (
            "升档来自两个必要判断与顺序约束共同决定答案，不来自题面出现“前者/后者”字样",
            "难度来自非重复反应事实或回答规则的异质性，不来自选项数量",
            "已明确给出仰视或俯视，只判断读数相对实际量的偏大或偏小，不继续推导实验结果",
            "直接读取一个正确刻度仍为送分题",
            "一次范围或边界计算即可确定结果，后续计算常规",
        ):
            with self.subTest(anchor=anchor):
                self.assertIn(anchor, prompt)
        self.assertNotIn("四项以上非重复反应事实", prompt)
        self.assertNotIn("直接应用一次极值法", prompt)

    def test_prompt_moves_non_generalizable_cases_to_special_policy(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        boundary_cases = prompt[
            prompt.index("## 三、教师相邻边界例题") :
            prompt.index("## 四、17项可观测特征协议")
        ]
        special = boundary_cases[
            boundary_cases.index("### 特殊教师口径") :
        ]

        for special_case in (
            "【Case 41：文字转译实际成为必要步骤】",
            "【Case 56：其他学科模型进入关键化学推导】",
            "【Case 62：题干未提供的课程越界知识】",
            "【Case 64：固定基团转换关系】",
        ):
            with self.subTest(special_case=special_case):
                self.assertIn(special_case, special)
        self.assertIn("不能作为普通相邻边界外推", special)

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

    def test_prompt_clean_contract_decouples_tasks_and_rules(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "即使使用同一rule_family，也应分别计入有效任务",
            prompt,
        )
        self.assertIn(
            "只有回答规则本身发生切换时，才增加rule_families",
            prompt,
        )
        self.assertNotIn("#### 17项可观测特征落点", prompt)
        self.assertIn(
            "若不同对象必须分别检索不同的对象—结论事实",
            prompt,
        )

    def test_prompt_clean_contract_defines_high_risk_feature_boundaries(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        for anchor in (
            "普通选择题排除错误选项，也不等于“排除候选解释”",
            "只从最终平台读取一个稳定数值属于“直接读数”",
            "普通总质量相减或剩余量计算不算差量",
            "一条反应关系加普通代数整理不算联立",
            "选择对后续模型具有决定作用的最具体结构",
            "流程或关系图解析",
            "化学式组成计算",
            "跨学科语义或模型应用",
        ):
            self.assertIn(anchor, prompt)

    def test_prompt_output_example_is_a_clean_single_equation_model(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        output_section = prompt[prompt.index("## 六、输出要求") :]

        self.assertIn('"evidence_operations": []', output_section)
        self.assertIn('"graph_table_operation": "无"', output_section)
        self.assertIn(
            '"calculation_operations": ["单一方程式"]',
            output_section,
        )
        self.assertNotIn(
            '"evidence_operations": ["多证据共同成立"]',
            output_section,
        )
        self.assertIn(
            "存在多个任务、规则切换或前后依赖时",
            output_section,
        )
        self.assertIn(
            "不得为了满足模板虚构不存在的广度、切换或依赖",
            output_section,
        )

    def test_prompt_front_loads_only_the_execution_order(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        opening = prompt[
            prompt.index("## 一、") : prompt.index("## 二、")
        ]
        feature_section = prompt[
            prompt.index("## 四、17项可观测特征协议") :
            prompt.index("## 五、定档前校验")
        ]

        self.assertIn("## 一、判定顺序：先记录事实，再判等级", opening)
        self.assertIn("最终作答目标、有效任务和回答规则", opening)
        self.assertIn("17项特征填写完成后即冻结", opening)
        self.assertNotIn("以下操作可以成为最长连续解题链中的一步", opening)
        self.assertNotIn("### 分析优先级", opening)
        self.assertNotIn("程序只使用稳定区间和客观统计", opening)
        self.assertIn("阅读题干、观察图片", feature_section)
        self.assertNotIn("§", prompt)

    def test_prompt_feature_protocol_records_facts_not_difficulty_rules(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        error_section = prompt[
            prompt.index("### 15. error_analysis_operation") :
            prompt.index("### 16. calculation_operations")
        ]
        curriculum_section = prompt[
            prompt.index("### 4. curriculum_topics") :
            prompt.index("### 5. parallel_task_relation")
        ]
        experiment_structure_section = prompt[
            prompt.index("### 12. experiment_task_structure") :
            prompt.index("### 13. visual_task_structure")
        ]

        self.assertNotIn("基础题", error_section)
        self.assertNotIn("中等题", error_section)
        self.assertNotIn("更高档", error_section)
        self.assertNotIn("历史V2", curriculum_section)
        self.assertNotIn("不能只因都属于实验题就压成", experiment_structure_section)
        self.assertIn("普通错误选项不属于“干扰条件排除”", prompt)

    def test_prompt_parallel_relation_uses_task_based_decision_order(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        section = prompt[
            prompt.index("### 5. parallel_task_relation") :
            prompt.index("### 6. solution_topology")
        ]

        self.assertIn("前一任务得到的中间结论、参数、物质状态或数据", section)
        self.assertIn("合并重复任务后是否只剩一个有效任务", section)
        self.assertIn("使用同一`rule_family`", section)

    def test_prompt_output_example_keeps_chain_inside_one_task(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        output_section = prompt[prompt.index("## 六、输出要求") :]

        self.assertIn(
            '"task_groups": [\n      {"task_type": "定量计算", "count": 1}\n    ]',
            output_section,
        )
        self.assertIn(
            '"rule_families": [\n      "定量关系与计算"\n    ]',
            output_section,
        )
        self.assertIn('"parallel_task_relation": "单一答题目标"', output_section)
        self.assertNotIn('{"task_type": "性质与反应判断", "count": 1}', output_section)

    def test_prompt_mechanically_maps_coarse_level_and_shortens_suffix(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "`coarse_difficulty`在`difficulty_level`确定后机械生成，不参与难度判断",
            prompt,
        )
        for mapping in (
            "送分题、基础题 → 送分/基础区间（1-2档）",
            "中等题 → 基础/中等区间（2-3档）",
            "拔高题 → 中等/拔高区间（3-4档）",
            "压轴题 → 拔高/压轴区间（4-5档）",
        ):
            self.assertIn(mapping, prompt)
        suffix = prompt[prompt.index("DIFFICULTY_RATING_PROMPT_SUFFIX") :]
        self.assertIn("先填写并冻结17项features", suffix)
        self.assertIn("最后机械生成coarse_difficulty", suffix)
        self.assertNotIn("若附有图片", suffix)

    def test_prompt_does_not_double_count_a_supplied_equation(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "核验题干已给出的熟悉化学方程式",
            prompt,
        )
        self.assertIn(
            "不得重复计作自主书写方程式或新增一个任务",
            prompt,
        )

    def test_prompt_treats_difference_method_as_a_hard_boundary(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "由反应前后质量差反求反应量",
            prompt,
        )
        self.assertIn(
            "差量/守恒必须由学生自主选择并成为决定性建模方法",
            prompt,
        )

    def test_prompt_does_not_merge_distinct_facts_by_answer_form(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "不能因作答形式都属于主观表达而合并成同一规则",
            prompt,
        )

    def test_prompt_does_not_promote_every_distinct_fact_by_count(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "多个对象、较长背景或多幅候选图片不自动增加难度",
            prompt,
        )
        self.assertIn(
            "需要分别回忆四种物质的燃烧现象，不能压成同一结论重复核验",
            prompt,
        )
        self.assertIn(
            "同一教材结论的一眼重复",
            prompt,
        )

    def test_prompt_treats_given_equation_check_as_one_task(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "题干已完整给出反应原理或化学方程式",
            prompt,
        )
        self.assertIn(
            "只记录一个有效任务",
            prompt,
        )
        self.assertIn(
            "不得拆成两个连续步骤",
            prompt,
        )

    def test_prompt_keeps_heterogeneous_subjective_breadth_above_basic(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "难度来自非重复反应事实或回答规则的异质性，不来自选项数量",
            prompt,
        )
        self.assertIn(
            "作用、目的、失败原因和规范现象分别调用不同化学依据",
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
            "由结论反推操作和方案评价",
            prompt,
        )

    def test_prompt_case_58_uses_independent_relations_not_step_inflation(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "总碱量关系与固体增重差量分别约束不同未知量",
            prompt,
        )
        self.assertNotIn("由盐酸用量建立总碱量关系→", prompt)

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

    def test_prompt_boundaries_separate_objects_order_and_images(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("两个必要判断与顺序约束共同决定答案", prompt)
        self.assertIn("多图独立不同规则判断", prompt)
        self.assertIn(
            "所有候选答案都可由同一个显性判据直接推出",
            prompt,
        )
        self.assertIn(
            "必须分别回忆对象特有事实",
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
            "单个主观回答只改变作答形式，不自动增加实质任务",
            prompt,
        )

    def test_prompt_treats_distinct_normative_explanations_as_real_rules(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "作用、目的、失败原因和规范现象分别调用不同化学依据时，必须按不同具体回答规则记录",
            prompt,
        )
        self.assertIn(
            "不能因作答形式都属于主观表达而合并成同一规则",
            prompt,
        )

    def test_prompt_treats_symbol_positions_as_distinct_rules(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "左前方、正上方、右上角和右下角数字",
            prompt,
        )
        self.assertIn(
            "原子个数、化合价、离子电荷和分子中原子个数",
            prompt,
        )
        self.assertIn(
            "不能因都在解释数字含义而合并成同一规则",
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

    def test_prompt_distinguishes_repeated_conservation_from_cross_constraints(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "总碱量关系与固体增重差量分别约束不同未知量",
            prompt,
        )
        self.assertIn(
            "不是同一种元素守恒的重复应用",
            prompt,
        )

    def test_prompt_separates_course_boundary_exception_from_final_paths(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "这不是第三条普遍压轴路径",
            prompt,
        )
        self.assertIn(
            "多个题干未提供且无法由初中知识推出的高中或竞赛规律",
            prompt,
        )

    def test_prompt_uses_cases_only_for_same_structure_cross_check(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("教师Case只作为边界锚点，不是题型模板", prompt)
        self.assertIn("存在高度同型Case时比较新增或缺少的真实任务结构", prompt)
        self.assertNotIn(
            "先从Case中选择结构最接近的一对低档侧/高档侧",
            prompt,
        )

    def test_teacher_levels_and_boundaries_precede_feature_protocol(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        levels = prompt.index("## 二、教师五档难度")
        boundaries = prompt.index("## 三、教师相邻边界例题")
        features = prompt.index("## 四、17项可观测特征协议")
        self.assertLess(levels, boundaries)
        self.assertLess(boundaries, features)

    def test_all_teacher_examples_keep_concrete_three_part_format(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        levels = [
            ("### 难度1——送分题", "### 难度2——基础题", "送分题", 6),
            ("### 难度2——基础题", "### 难度3——中等题", "基础题", 9),
            ("### 难度3——中等题", "### 难度4——拔高题", "中等题", 8),
            ("### 难度4——拔高题", "### 难度5——压轴题", "拔高题", 3),
            ("### 难度5——压轴题", "## 三、教师相邻边界例题", "压轴题", 5),
        ]

        self.assertNotIn("#### 来自教师复核分支的真实锚点", prompt)
        for start_marker, end_marker, level, expected_cases in levels:
            start = prompt.index(start_marker)
            end = prompt.index(end_marker, start + len(start_marker))
            section = prompt[start:end]
            self.assertIn("#### 代表性例题", section)
            self.assertEqual(section.count("【Case "), expected_cases)
            self.assertEqual(section.count("题目："), expected_cases)
            self.assertEqual(
                section.count(f"教师等级：{level}。"),
                expected_cases,
            )
            self.assertEqual(section.count("判定："), expected_cases)

    def test_prompt_output_example_covers_all_declared_task_groups(
        self,
    ) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "确定反应→建立计量关系→求目标量",
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
            "指出缺失的具体任务边",
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
            "不能仅因链较长或反应数量较多判为压轴题",
            prompt,
        )

    def test_prompt_lists_complete_new_information_enum_and_bare_topics(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("依赖题干未给出的超纲化学知识", prompt)
        self.assertIn(
            "只输出裸编码",
            prompt,
        )
        self.assertIn("`U3-2`原子结构", prompt)
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

    def test_mismatched_topic_name_uses_internal_audit_fallback(self) -> None:
        item = production_features()
        item["curriculum_topics"] = ["U3-2元素"]

        normalized, actions = self.features.normalize_observable_features(item)
        validated = self.features.validate_observable_features(normalized)
        flags = self.features.observable_feature_quality_flags(
            validated,
            actions,
        )

        self.assertEqual(validated["curriculum_topics"], ["U_OTHER"])
        self.assertIn("fallback_enum:curriculum_topics", flags)

    def test_prompt_does_not_require_model_to_name_derived_span(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "任务之间是否共享中间结论或模型由`parallel_task_relation`记录",
            prompt,
        )
        self.assertNotIn("历史V2", prompt)

    def test_prompt_declares_runtime_shape_limits(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "包含1至12个具体必要化学决策，每步1至80字且不得重复",
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

    def test_dense_guard_blocks_short_single_reaction_breadth(self) -> None:
        item = hard_rating()
        item["features"]["longest_solution_chain"] = [
            "读取反应前后质量差",
            "按给定方程式建立计量关系",
            "求出目标物质质量",
            "计算目标质量分数",
        ]
        item["features"]["reaction_structure"] = "单一反应"
        item["features"]["solution_topology"] = "单线性常规链"

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {
                "stem": "围绕一个反应设置六个并列探究任务。",
                "sub_questions": [
                    {"stem": f"任务{i}"}
                    for i in range(1, 7)
                ],
            },
        )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_dense_guard_keeps_exceptionally_broad_single_reaction(self) -> None:
        item = hard_rating()
        item["features"]["longest_solution_chain"] = [
            "读取反应前后质量差",
            "按给定方程式建立计量关系",
            "求出目标物质质量",
            "计算目标质量分数",
        ]
        item["features"]["reaction_structure"] = "单一反应"
        item["features"]["solution_topology"] = "单线性常规链"

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {
                "stem": "围绕一个共享模型设置八个关联探究任务。",
                "sub_questions": [
                    {"stem": f"任务{i}"}
                    for i in range(1, 9)
                ],
            },
        )

        self.assertEqual(result["difficulty_level"], "压轴题")
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

    def test_v5_deep_quantitative_chain_is_audit_only(self) -> None:
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

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "压轴题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_hard_to_final_deep_quantitative_chain",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )
        self.assertIn(
            "仅保留候选审计",
            result[
                "teacher_distribution_guard_writeback_blocked_reason"
            ],
        )

    def test_qualitative_evidence_network_is_audit_only_final_candidate(
        self,
    ) -> None:
        item = hard_rating()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "整理实验后的离子组成",
                    "根据现象排除不可能离子",
                    "结合电荷与共存关系确定必然组分",
                    "筛选获得唯一结论的检验方案",
                ],
                "task_groups": [
                    {"task_type": "化学用语", "count": 1},
                    {"task_type": "性质与反应判断", "count": 2},
                    {"task_type": "证据推断", "count": 2},
                    {"task_type": "方案设计与评价", "count": 1},
                ],
                "rule_families": [
                    "化学用语",
                    "性质与反应判断",
                    "证据推断",
                    "方案设计与评价",
                ],
                "parallel_task_relation": "共享同一化学模型的关联任务",
                "solution_topology": "未知组分消元或组成不变量",
                "reaction_structure": "先后竞争或过量不足",
                "condition_operations": ["干扰条件排除", "范围或边界"],
                "evidence_operations": [
                    "多证据共同成立",
                    "排除多个候选解释",
                ],
                "experiment_operation": "现象解释",
                "experiment_task_structure": "控制变量或数据归纳",
                "graph_table_operation": "无",
                "calculation_operations": [],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "结合多组离子现象排除候选并设计唯一检验方案。"},
        )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "压轴题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_hard_to_final_qualitative_evidence_network_candidate",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_qualitative_candidate_rejects_generic_scheme_evaluation(
        self,
    ) -> None:
        item = hard_rating()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "读取实验现象",
                    "比较两组方案",
                    "排除不合理候选",
                    "选择补充实验",
                ],
                "task_groups": [
                    {"task_type": "实验操作与探究", "count": 3},
                    {"task_type": "证据推断", "count": 2},
                    {"task_type": "方案设计与评价", "count": 1},
                ],
                "rule_families": [
                    "实验操作与探究",
                    "证据推断",
                    "方案设计与评价",
                ],
                "parallel_task_relation": "共享同一化学模型的关联任务",
                "solution_topology": "条件分支或范围筛选",
                "reaction_structure": "多个并列反应",
                "condition_operations": ["干扰条件排除"],
                "evidence_operations": [
                    "多证据共同成立",
                    "排除多个候选解释",
                ],
                "experiment_operation": "方案评价或补充实验",
                "experiment_task_structure": "方案设计或评价",
                "graph_table_operation": "无",
                "calculation_operations": [],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "比较并列实验方案并选择补充实验。"},
        )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "拔高题",
        )
        self.assertIsNone(
            result["teacher_distribution_guard_candidate_action"]
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

    def test_deep_guard_blocks_low_density_two_conservation_chain(self) -> None:
        item = hard_rating()
        item["features"].update(
            {
                "task_groups": [
                    {"task_type": "定量计算", "count": 4},
                ],
                "rule_families": ["定量计算", "性质与反应判断"],
                "solution_topology": "未知组成或量反推",
                "reaction_structure": "产物进入后一反应",
                "condition_operations": [],
                "evidence_operations": [],
                "experiment_operation": "无",
                "experiment_task_structure": "无实验判断",
                "graph_table_operation": "无",
                "calculation_operations": ["单一守恒", "差量"],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {
                "stem": "沿同一条清晰主线完成两次守恒计算。",
                "sub_questions": [
                    {"stem": "任务1"},
                    {"stem": "任务2"},
                    {"stem": "任务3"},
                ],
            },
        )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_deep_guard_does_not_trust_support_claims_for_simple_chain(
        self,
    ) -> None:
        item = hard_rating()
        item["features"].update(
            {
                "task_groups": [
                    {"task_type": "定量计算", "count": 2},
                ],
                "rule_families": ["定量计算", "性质与反应判断"],
                "solution_topology": "未知组成或量反推",
                "reaction_structure": "产物进入后一反应",
                "condition_operations": ["条件切换"],
                "evidence_operations": ["多证据共同成立"],
                "experiment_operation": "无",
                "experiment_task_structure": "无实验判断",
                "graph_table_operation": "无",
                "calculation_operations": ["单一守恒", "联立"],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {
                "stem": "燃料不完全燃烧后沿同一主线重复使用元素守恒。",
                "sub_questions": [
                    {"stem": "写出方程式"},
                    {"stem": "完成守恒计算"},
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
