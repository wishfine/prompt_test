# -*- coding: utf-8 -*-
"""GPT-5.6 主标签 Hybrid5d 候选资源验收。"""

from __future__ import annotations

import unittest
from pathlib import Path
import json
import re


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "初中物理难度打标提示词_gpt56_hybrid.txt"
SCRIPT_PATH = ROOT / "src" / "physics_difficulty_rating_with_cache.py"


class GPT56HybridPromptAssetTests(unittest.TestCase):
    def load_config(self) -> tuple[str, str]:
        namespace: dict[str, str] = {}
        source = PROMPT_PATH.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT_PATH), "exec"), {}, namespace)
        return (
            namespace["DIFFICULTY_RATING_PROMPT_PREFIX"],
            namespace["DIFFICULTY_RATING_PROMPT_SUFFIX"],
        )

    def test_prompt_is_parseable_and_keeps_full_hybrid5d_structure(self) -> None:
        prefix, suffix = self.load_config()
        self.assertGreater(len(prefix), 15000)
        self.assertIn("五维定档主标准", prefix)
        self.assertIn("18 个 features", prefix)
        self.assertIn("全面的难度分析和评级", suffix)

    def test_prompt_uses_self_contained_rating_standard_without_label_provenance(self) -> None:
        prefix, _ = self.load_config()
        self.assertNotIn("GPT-5.6 专家复核裁定", prefix)
        self.assertNotIn("教师原标签只作为辅助参考", prefix)
        self.assertNotIn("1066 道题", prefix)
        self.assertNotIn("得分率", prefix)
        self.assertIn("只根据当前题目的真实解题任务定档", prefix)

    def test_easy_boundary_allows_one_familiar_standard_operation(self) -> None:
        prefix, _ = self.load_config()
        self.assertIn("没有第二次物理决策", prefix)
        self.assertIn("一步直接代入", prefix)
        self.assertIn("直接图像读数", prefix)
        self.assertIn("单一教材原型作图或接线", prefix)
        self.assertIn("多个独立空或选项", prefix)
        self.assertIn("同一种直接检索规则", prefix)

    def test_basic_medium_boundary_uses_structure_not_option_count(self) -> None:
        prefix, _ = self.load_config()
        self.assertIn("彼此独立地调用不同的通用教材结论", prefix)
        self.assertIn("通常仍是基础题", prefix)
        self.assertIn("充分条件、必要条件或反例", prefix)
        self.assertIn("共享题目特有的中间结论", prefix)
        self.assertIn("完整标准实验流程", prefix)
        self.assertIn("3—4个有效物理决策", prefix)

    def test_hard_has_transform_and_dense_complete_chain_channels(self) -> None:
        prefix, _ = self.load_config()
        self.assertIn("决定性转换通道", prefix)
        self.assertIn("高密度完整链通道", prefix)
        self.assertIn("约5—6个有效物理决策", prefix)
        self.assertIn("不要求必须存在单一神奇卡点", prefix)
        self.assertIn("每一步都是常规公式", prefix)
        self.assertIn("不等于整体只是常规中等题", prefix)

    def test_prompt_distinguishes_low_structure_independence_from_full_model_load(self) -> None:
        prefix, _ = self.load_config()
        self.assertIn("低结构独立任务", prefix)
        self.assertIn("非平凡独立任务", prefix)
        self.assertIn("不把选项数机械换算为 step_count", prefix)
        self.assertIn("完整任务负担", prefix)
        self.assertIn("最多只支持向相邻高一档比较", prefix)
        self.assertIn("至少两种不同的实质活动", prefix)
        self.assertIn("不能单独支持压轴题", prefix)

    def test_prompt_records_independent_subquestions_field_by_field(self) -> None:
        prefix, _ = self.load_config()
        self.assertIn("step_count：按最高难单项", prefix)
        self.assertIn("knowledge_count：按整题实际涉及的知识点并集", prefix)
        self.assertIn("state_count 和 constraint_count", prefix)
        self.assertIn("cross_module：只有多个模块在同一解题链中互相提供条件", prefix)
        self.assertIn("模型共享但答案互不依赖", prefix)

    def test_prompt_distinguishes_arithmetic_and_physical_intermediate_quantities(self) -> None:
        prefix, _ = self.load_config()
        self.assertIn("算术中间量", prefix)
        self.assertIn("物理中间量", prefix)
        self.assertIn("再调用另一条物理规律", prefix)

    def test_knowledge_context_headings_do_not_supply_level_priors(self) -> None:
        prefix, _ = self.load_config()
        self.assertNotIn("物理高难题知识点：大概率判为 4-5 档", prefix)
        self.assertNotIn("物理中等难度知识点：大概率判为 2-4 档", prefix)
        self.assertIn("知识点名称本身不决定等级", prefix)

    def test_few_shots_do_not_expose_question_ids(self) -> None:
        prefix, _ = self.load_config()
        self.assertNotRegex(prefix, r"question_id\s*[：:]")

    def test_prompt_contains_severe_deviation_audit_patterns(self) -> None:
        prefix, _ = self.load_config()
        self.assertNotIn("1066 题 GPT-5.6 裁定的结构共性校准", prefix)
        self.assertIn("相变—浮力—排开体积—液面", prefix)
        self.assertIn("多图表与多公式", prefix)
        self.assertIn("额定限制、极值与图像共同约束", prefix)
        self.assertIn("多组构型下的等效判定", prefix)

    def test_final_requires_decisive_network_operation(self) -> None:
        prefix, _ = self.load_config()
        self.assertIn("压轴候选必须同时具有", prefix)
        self.assertIn("分类、临界、边界验证、有效解筛选或图像反推", prefix)

    def test_final_uses_model_dependency_and_full_network(self) -> None:
        prefix, _ = self.load_config()
        self.assertIn("答案依赖", prefix)
        self.assertIn("模型依赖", prefix)
        self.assertIn("复杂状态—参数—约束网络", prefix)
        self.assertIn("全链耦合", prefix)

    def test_untrusted_source_difficulty_is_forbidden(self) -> None:
        prefix, _ = self.load_config()
        self.assertNotIn("输入数据原有 difficulty", prefix)
        self.assertNotIn("输入数据可信度", prefix)

    def test_output_example_is_valid_json(self) -> None:
        prefix, _ = self.load_config()
        match = re.search(r"合法 JSON 示例：\s*(\{.*?\n\})\s*\n\s*注意：", prefix, re.S)
        self.assertIsNotNone(match)
        parsed = json.loads(match.group(1))
        self.assertEqual(len(parsed["features"]), 18)
        self.assertIn(parsed["difficulty_level"], ["送分题", "基础题", "中等题", "拔高题", "压轴题"])

    def test_few_shot_feature_values_are_single_legal_enums(self) -> None:
        prefix, _ = self.load_config()
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


class GPT56HybridScriptAssetTests(unittest.TestCase):
    def test_script_supports_gpt56_profile_and_audit_switch(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertIn('"gpt56_hybrid"', source)
        self.assertIn("postprocess_gpt56_hybrid", source)
        self.assertIn('"gpt56_independence_guard_enabled"', source)
        self.assertIn('"gpt56_structural_calibration_enabled"', source)
        self.assertIn('"gpt56_severe_deviation_guards_enabled"', source)
        self.assertIn('"progressive_final_chain_effective"', source)


if __name__ == "__main__":
    unittest.main()
