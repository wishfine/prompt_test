# -*- coding: utf-8 -*-
"""V7 兼容基线资源的静态验收。"""

from __future__ import annotations

import unittest
from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[1]


class V7CompatAssetTests(unittest.TestCase):
    def test_archived_prompt_is_python_config_and_complete(self) -> None:
        path = ROOT / "prompts" / "archive" / "初中物理难度打标提示词_v7_best.txt"
        namespace: dict[str, str] = {}
        source = path.read_text(encoding="utf-8")
        exec(compile(source, str(path), "exec"), {}, namespace)
        prefix = namespace.get("DIFFICULTY_RATING_PROMPT_PREFIX", "")
        suffix = namespace.get("DIFFICULTY_RATING_PROMPT_SUFFIX", "")
        self.assertIn("真实边界判定 few-shot 示例", prefix)
        self.assertIn("【示例26", prefix)
        self.assertIn("全面的难度分析和评级", suffix)

    def test_legacy_reference_compiles_and_contains_v7_final_definition(self) -> None:
        path = ROOT / "src" / "legacy" / "physics_difficulty_rating_v7_reference.py"
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        self.assertIn("V7 边界小修版", source)
        self.assertGreaterEqual(source.count("def postprocess_physics_difficulty"), 1)


class ProductionPromptAssetTests(unittest.TestCase):
    def load_prefix(self) -> str:
        path = ROOT / "prompts" / "初中物理难度打标提示词.txt"
        namespace: dict[str, str] = {}
        exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), {}, namespace)
        return namespace["DIFFICULTY_RATING_PROMPT_PREFIX"]

    def test_production_prompt_keeps_old_baseline_depth_and_two_layer_examples(self) -> None:
        prefix = self.load_prefix()
        self.assertGreaterEqual(len(prefix), 17500)
        # Prompt 长度不是评级约束；这里只防止无意中的失控膨胀。
        self.assertLess(len(prefix), 27000)
        self.assertGreaterEqual(prefix.count("### 代表性例题"), 5)
        self.assertIn("## 相邻档位边界校准 few-shot", prefix)

    def test_production_prompt_has_ten_non_versioned_boundary_examples(self) -> None:
        prefix = self.load_prefix()
        self.assertEqual(len(re.findall(r"【边界示例\d+】", prefix)), 10)
        self.assertNotRegex(prefix, r"V5|V6|V7")
        self.assertNotIn("回收中等保护", prefix)
        self.assertNotIn("压轴保护恢复", prefix)

    def test_production_prompt_resolves_diagram_and_multi_blank_conflicts(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("单个静止物体的教材原型受力图", prefix)
        self.assertIn("一条凸透镜特殊光线", prefix)
        self.assertRegex(prefix, r"复杂受力分析[^。]{0,100}至少基础题")
        self.assertIn("同一小节的多个直接识记空", prefix)
        self.assertRegex(prefix, r"多个不同实验原理[^。]{0,100}至少基础题")

    def test_production_prompt_uses_direct_retrieval_bundle_easy_boundary(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("直接检索束", prefix)
        self.assertIn("每个空或选项都能独立由教材事实直接作答", prefix)
        self.assertIn("不需要共同物理过程、规律选择或条件联动", prefix)
        self.assertIn("分子动理论知识结构图", prefix)
        self.assertIn("多个空不等于多个应用步骤", prefix)

    def test_production_prompt_uses_five_dimension_anchor_with_task_structure_check(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("教师五维定档主标准", prefix)
        self.assertIn("真实解题任务结构", prefix)
        for dimension in ["直接识别", "显性应用", "常规分析", "决定性转换", "全链耦合"]:
            self.assertIn(dimension, prefix)
        self.assertIn("步骤数不是档位门槛", prefix)

    def test_production_prompt_does_not_treat_choice_count_or_simple_application_as_easy(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("四个选项本身既不升档也不降档", prefix)
        self.assertIn("条件与唯一物理结论之间的一次透明映射", prefix)
        self.assertIn("需要在多个规律中选择", prefix)
        self.assertNotIn("四个短选项或一步因果；只要", prefix)
        self.assertNotIn("一步生活原型对应，不必然排除送分", prefix)

    def test_production_prompt_allows_single_question_internal_chain_to_be_final(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("不以存在多个小问或前问结果复用为必要条件", prefix)
        self.assertIn("单个设问内部", prefix)

    def test_production_prompt_contains_sample_derived_boundary_corrections(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("四个选项不会自动排除送分题", prefix)
        self.assertIn("高密度概念辨析", prefix)
        self.assertIn("一个决定性转换", prefix)
        self.assertIn("整题的完整推理链", prefix)

    def test_production_prompt_has_no_unavailable_score_rate_variable(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("得分率", prefix)
        self.assertNotRegex(prefix, r"(?<![A-Za-z])P(?:≥|<|时)")

    def test_production_prompt_uses_stable_adjacent_boundary_table(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("最短且完整的有效解题链", prefix)
        self.assertIn("不能只统计最后一问从已知答案出发的局部步骤", prefix)
        self.assertIn("相邻档位稳定决策表", prefix)
        self.assertIn("3-5步本身不能证明达到中等题", prefix)
        self.assertIn("6-8步也可以判压轴题", prefix)
        self.assertIn("步骤数只作支持证据，不作为单独门槛", prefix)
        self.assertNotIn("向上复核：防止专家视角压缩步骤", prefix)

    def test_independent_options_are_not_accumulated_into_medium_workload(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("四个彼此独立的选项不是四步", prefix)
        self.assertIn("以最高难选项自身的有效推理链计步", prefix)
        self.assertNotIn("中等题的常见标志是“整体工作量和联合辨析”", prefix)

    def test_final_boundary_keeps_six_to_eight_step_enum_without_five_step_anchor(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("6-8步也可以判压轴题", prefix)
        self.assertNotIn("实际约5—6步也可进入压轴比较", prefix)
        self.assertNotIn("实际约5—6步的高密度完整链", prefix)
        self.assertNotIn("实际约5—6步只有", prefix)

    def test_independent_questions_use_only_the_hardest_question_step_chain(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("多个相互独立的直接小问可填“3-5步”", prefix)
        self.assertIn("step_count 仍按最高难小问或最高难选项自身的连续推理链填写", prefix)

    def test_parallel_concepts_keep_step_depth_and_have_a_breadth_gate(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("通常判为基础题或中等题", prefix)
        self.assertIn("横向有效任务广度的严格判定", prefix)
        self.assertIn("选项或小问彼此独立，只表示它们不能机械累加为连续推理步骤", prefix)
        self.assertIn("至少两个分析型任务分别是什么", prefix)
        self.assertIn("不能作为横向广度升档依据", prefix)

    def test_production_prompt_uses_depth_breadth_dual_track(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("纵向深度", prefix)
        self.assertIn("横向广度", prefix)
        self.assertIn("step_count 不机械累加", prefix)
        self.assertIn("共享同一物理过程、实验逻辑、状态变量、图像关系或装置工作机制", prefix)
        self.assertIn("完全无关的教材事实", prefix)

    def test_composite_features_use_field_specific_scopes(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("knowledge_count 记录整道题实际要求调用且彼此不同的知识点并集", prefix)
        self.assertIn("information_carrier 记录整题实际参与作答的主要信息载体", prefix)
        self.assertIn("多个状态只有在题目要求比较、串联或统一分析它们时", prefix)
        self.assertIn("多个约束只有共同参与同一求解、范围或有效解筛选时", prefix)
        self.assertIn("仍按最高难单项记录", prefix)

    def test_medium_features_allow_truthful_breadth_values(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("通过横向广度达到中等时可以是0-1个", prefix)
        self.assertIn("通过横向广度达到中等时可以是4个及以上", prefix)
        self.assertIn("最高难单项可以只是简单因果推理", prefix)
        self.assertIn("载体形式不决定等级", prefix)
        self.assertIn("实验要求：可以为无", prefix)

    def test_horizontal_breadth_excludes_transparent_or_repeated_tasks(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("至少两个任务属于分析型任务", prefix)
        self.assertIn("直接读取一个点", prefix)
        self.assertIn("仅改变知识点名称但认知动作相同", prefix)
        self.assertIn("横向广度最多支持相邻升一档", prefix)

    def test_horizontal_breadth_requires_shared_physics_not_shared_packaging(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("共享同一物理过程、实验逻辑、状态变量、图像关系或装置工作机制", prefix)
        self.assertIn("仅共享故事背景、图片、物品名称、比赛场景、科技产品或生活主题", prefix)
        self.assertIn("不构成共同物理结构", prefix)
        self.assertIn("若将某个选项替换为另一个生活物品", prefix)
        self.assertNotIn("同一情境中的有效任务广度", prefix)
        self.assertNotIn("同一情境中完成多个非重复的应用型任务", prefix)

    def test_single_model_chain_does_not_count_intermediate_quantities_as_steps(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("有效步骤不是中间量或自然语言箭头的数量", prefix)
        self.assertIn("传感器电阻变化→总电阻变化→总电流变化", prefix)
        self.assertIn("一次模型识别加一次规律应用", prefix)
        self.assertIn("中途需要更换物理模型", prefix)

    def test_dynamic_circuit_counts_new_physical_relations_not_arrows(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("只要求沿同一关系链得到一个最终趋势", prefix)
        self.assertIn("进一步判断局部电压、分压比例、功率", prefix)
        self.assertIn("多个电表示数之间的关系", prefix)
        self.assertIn("按实际新增关系计入连续分析", prefix)

    def test_transparent_mapping_is_limited_to_one_clear_knowledge_target(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("一次透明映射通常只对应一个明确知识目标", prefix)
        self.assertIn("多个不同实验原理、跨小节概念、规范测量步骤或多个不同物理属性", prefix)
        self.assertIn("非零起点相减", prefix)

    def test_breadth_core_basis_names_shared_mechanism_and_two_analytical_tasks(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("core_basis 必须列出共同物理结构", prefix)
        self.assertIn("至少两个分析型任务分别是什么", prefix)
        self.assertIn("同一生活情境", prefix)

    def test_breadth_requires_shared_device_specific_information(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("至少两个分析型任务必须共同使用", prefix)
        self.assertIn("同一项装置特有信息、状态关系或中间结论", prefix)
        self.assertIn("只需分别调用通用教材结论", prefix)
        self.assertIn("通常只是共同背景，不是共同机制", prefix)

    def test_feature_truth_has_priority_over_level_appearance(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("features 与 difficulty_level 必须相互一致", prefix)
        self.assertIn("features 的事实真实性优先于表面档位一致性", prefix)
        self.assertIn("可以呈现非典型组合", prefix)
        self.assertIn("step_count=1-2步", prefix)
        self.assertIn("reasoning_chain=简单因果推理", prefix)

    def test_high_features_require_actual_solution_structure_not_keywords(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("若题目涉及极值、范围、不等式", prefix)
        self.assertIn("若实际求解确实需要极值、范围、不等式", prefix)
        self.assertIn("仅出现相关词语、陌生装置名称或生活背景", prefix)
        self.assertIn("不得填写高阶特征", prefix)

    def test_easy_formula_boundary_distinguishes_direct_substitution(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("题面已经明确唯一关系的一步直接代入不必然排除送分题", prefix)
        self.assertIn("自主选择适用规律或公式", prefix)
        self.assertIn("先求中间量", prefix)

    def test_parallel_module_coverage_is_not_cross_module_fusion(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("cross_module 只记录不同模块是否在同一推理链中发生融合", prefix)
        self.assertIn("仍填“同一模块内部”", prefix)
        self.assertIn("由 knowledge_count 记录知识覆盖广度", prefix)

    def test_easy_boundary_has_no_subquestion_count_threshold(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("小题数量≥3", prefix)
        self.assertIn("多个小问若跨知识点", prefix)

    def test_boundary_few_shot_feature_values_are_single_legal_enums(self) -> None:
        prefix = self.load_prefix()
        allowed = {
            "step_count": {"1-2步", "3-5步", "6-8步", "9-12步", "12步以上"},
            "formula_count": {"0-1个", "2-3个", "4-6个", "7个以上"},
            "calculation_complexity": {"口算或直接判断", "简单笔算", "多公式联立", "复杂方程或范围计算"},
            "reasoning_chain": {"直接套用", "简单因果推理", "多层因果推理", "逆向推理或临界分析"},
            "problem_structure": {"概念判断", "直接计算", "实验探究", "图像表格分析", "电路综合", "力学综合", "热学综合", "光学声学综合", "跨模块综合"},
            "information_carrier": {"纯文字", "单图识别", "电路图", "实验装置图", "图像或表格", "多图表综合"},
            "reality_question": {"是", "否"},
            "subquestion_dependency": {"无多问", "多问但相互独立", "多问且层层递进"},
            "knowledge_count": {"1个", "2-3个", "4个及以上"},
            "state_count": {"单状态", "双状态", "多状态", "连续变化或临界状态"},
            "constraint_count": {"无约束", "单一约束", "多约束"},
            "variable_relation": {"无变量关系", "简单正反比", "图像函数关系", "多变量耦合关系"},
            "experiment_requirement": {"无", "基础操作或读数", "控制变量或故障分析", "方案设计或误差评价"},
            "graph_table_requirement": {"无", "直接读数", "多组比较归纳", "图像反推或外推"},
            "error_risk": {"无明显易错点", "轻微易错点", "明显易错点", "高易错点"},
        }
        for line in prefix.splitlines():
            if not line.startswith("核心特征："):
                continue
            for item in re.split(r"[,，]", line.removeprefix("核心特征：").rstrip("。")):
                key, value = item.strip().split("=", 1)
                self.assertIn(key, allowed)
                self.assertIn(value, allowed[key], msg=f"非法 few-shot 枚举: {key}={value}")

    def test_json_core_basis_demonstrates_five_dimension_anchor(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("五维中过程/对象和思维层次落在中等档", prefix)
        self.assertIn("知识量与数学工具提供常规支撑", prefix)

    def test_knowledge_section_uses_structural_context_not_high_level_prior(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("物理高难题知识点：大概率判为 4-5 档", prefix)
        self.assertIn("容易承载高难结构的知识情境：最终等级仍由任务结构决定", prefix)
        self.assertIn("装置名称或知识点类别本身不构成升档依据", prefix)
        self.assertIn("多状态电路中的多重安全量程约束", prefix)
        self.assertIn("隐含控制逻辑与参数筛选并存的继电器控制", prefix)
        self.assertIn("非线性元件图像反推与多状态约束综合", prefix)

    def test_basic_to_medium_examples_distinguish_states_objects_and_modules(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("两个需要分别建模的状态", prefix)
        self.assertIn("多个相互作用的研究对象", prefix)
        self.assertIn("电学与热学、力学与热学", prefix)
        self.assertNotIn("力学+浮力", prefix)

    def test_medium_definition_and_examples_cover_shared_structure_and_finite_breadth(self) -> None:
        prefix = self.load_prefix()
        self.assertNotIn("数据归纳与高密度概念辨析", prefix)
        self.assertIn("围绕同一概念的充分必要条件、反例、特殊边界或规范表述辨析", prefix)
        self.assertNotIn("必须反复区分必要条件", prefix)
        self.assertIn("鱼缸增氧泵原理选择题", prefix)
        self.assertIn("形成共同机制下的横向有效任务广度", prefix)

    def test_glass_tube_example_records_multilayer_reasoning(self) -> None:
        prefix = self.load_prefix()
        section = prefix[prefix.index("【边界示例7】"):prefix.index("【边界示例8】")]
        self.assertIn("reasoning_chain=多层因果推理", section)
        self.assertNotIn("reasoning_chain=逆向推理或临界分析", section)

    def test_production_prompt_has_sample_anchored_hard_and_final_examples(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("两条关系线", prefix)
        self.assertIn("反推图线身份", prefix)
        self.assertIn("多开关电路中包含小灯泡", prefix)
        self.assertIn("多安全约束和功率边界筛选", prefix)
        self.assertNotIn("将天平改装为液体密度测量仪", prefix)

    def test_production_prompt_does_not_use_features_as_postprocess_triggers(self) -> None:
        prefix = self.load_prefix()
        self.assertIn("18 个 features 只用于客观记录和审计", prefix)
        self.assertIn("不得根据单个 feature 机械升降档", prefix)
        self.assertNotIn("凡是需要物理公式代入", prefix)
        self.assertNotIn("同时出现 9 步以上复杂推理", prefix)

    def test_batch_script_defaults_to_production_prompt(self) -> None:
        source = (ROOT / "src" / "physics_difficulty_rating_with_cache.py").read_text(encoding="utf-8")
        self.assertIn('"prompts", "初中物理难度打标提示词.txt"', source)
        self.assertNotIn('default_prompt =', source)

    def test_batch_output_records_progressive_chain_ab_switch(self) -> None:
        source = (ROOT / "src" / "physics_difficulty_rating_with_cache.py").read_text(encoding="utf-8")
        self.assertIn('"progressive_final_chain_enabled": ENABLE_PROGRESSIVE_FINAL_CHAIN', source)

    def test_final_prompt_json_example_has_no_duplicate_keys(self) -> None:
        prefix = self.load_prefix()
        marker = "合法 JSON 示例"
        start = prefix.index("{", prefix.index(marker))

        def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
            output: dict[str, object] = {}
            for key, value in pairs:
                if key in output:
                    raise ValueError(f"重复 JSON key: {key}")
                output[key] = value
            return output

        try:
            parsed, _ = json.JSONDecoder(object_pairs_hook=reject_duplicates).raw_decode(prefix[start:])
        except (ValueError, json.JSONDecodeError) as exc:
            self.fail(str(exc))
        self.assertEqual(set(parsed["features"]), {
            "step_count", "formula_count", "calculation_complexity", "reasoning_chain",
            "problem_structure", "additional_structure", "information_carrier", "reality_question",
            "subquestion_dependency", "knowledge_count", "knowledge_diff", "cross_module",
            "state_count", "constraint_count", "variable_relation", "experiment_requirement",
            "graph_table_requirement", "error_risk",
        })
        self.assertIn(parsed["difficulty_level"], ["送分题", "基础题", "中等题", "拔高题", "压轴题"])
        self.assertNotIn("difficulty_level", parsed["reasoning"])


if __name__ == "__main__":
    unittest.main()
