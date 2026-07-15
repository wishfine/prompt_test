# -*- coding: utf-8 -*-
"""物理难度后处理的离线单元测试；不需要访问模型服务。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import physics_difficulty_rating_with_cache as rating  # noqa: E402


def features(**overrides: str) -> dict[str, str]:
    value = dict(rating.FEATURE_DEFAULTS)
    value.update(overrides)
    return value


def result(level: str, **overrides: str) -> dict:
    return {
        "features": features(**overrides),
        "coarse_difficulty": "基础/中等区间（2-3档）",
        "reasoning": {
            "core_basis": "测试用例",
            "hard_point": "",
            "why_not_lower": "",
            "why_not_higher": "",
        },
        "difficulty_level": level,
    }


class PhysicsPostprocessTests(unittest.TestCase):
    def postprocess(self, level: str, stem: str, **feature_values: str) -> dict:
        return rating.postprocess_physics_difficulty(result(level, **feature_values), {"question_id": "test", "stem": stem})

    def test_direct_single_concept_stays_easy(self) -> None:
        output = self.postprocess("送分题", "声音的音调由什么决定？")
        self.assertEqual(output["difficulty_level"], "送分题")
        self.assertEqual(output["postprocess_actions"], [])

    def test_simple_formula_application_is_at_least_basic(self) -> None:
        output = self.postprocess(
            "送分题",
            "已知 F=10N，S=2m2，求压强。",
            formula_count="0-1个",
            calculation_complexity="简单笔算",
            reasoning_chain="简单因果推理",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_basic_lab_direct_reading_does_not_auto_upgrade(self) -> None:
        output = self.postprocess(
            "基础题",
            "读出温度计的示数。",
            experiment_requirement="基础操作或读数",
            information_carrier="实验装置图",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_standard_experiment_strong_signal_overrides_independent_subquestions(self) -> None:
        output = self.postprocess(
            "基础题",
            "在探究液体压强实验中，根据多组表格数据归纳规律。",
            step_count="3-5步",
            reasoning_chain="简单因果推理",
            problem_structure="实验探究",
            information_carrier="图像或表格",
            subquestion_dependency="多问但相互独立",
            knowledge_count="2-3个",
            experiment_requirement="控制变量或故障分析",
            graph_table_requirement="多组比较归纳",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_standard_rule_diagram_is_not_upgraded_by_weak_signals(self) -> None:
        output = self.postprocess(
            "基础题",
            "根据小磁针N极指向，画出磁感线方向并标出电源正极。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            information_carrier="电路图",
            knowledge_count="2-3个",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_shallow_cross_module_points_stay_basic(self) -> None:
        output = self.postprocess(
            "基础题",
            "多个小问分别判断物态变化、质量和一个基础力学概念。",
            reasoning_chain="简单因果推理",
            subquestion_dependency="多问但相互独立",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_cross_module_with_multiple_supporting_signals_reaches_medium(self) -> None:
        output = self.postprocess(
            "基础题",
            "阅读材料后提取信息，建立简单比例关系并完成跨模块分析。",
            step_count="3-5步",
            reasoning_chain="简单因果推理",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
            variable_relation="简单正反比",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_regular_pressure_scale_stays_medium(self) -> None:
        output = self.postprocess(
            "中等题",
            "压敏电阻压力秤，读出图像并计算电流。",
            step_count="3-5步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            information_carrier="图像或表格",
            knowledge_count="2-3个",
            state_count="双状态",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_explicit_relay_chain_stays_basic(self) -> None:
        output = self.postprocess(
            "基础题",
            "温度自动报警器中，电磁继电器吸合后电铃报警。",
            reasoning_chain="简单因果推理",
            state_count="双状态",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_textbook_lens_recognition_is_not_upgraded_by_subject_word(self) -> None:
        output = self.postprocess("送分题", "照相机镜头相当于什么透镜？")
        self.assertEqual(output["difficulty_level"], "送分题")

    def test_spatial_diagram_basic_is_not_automatically_downgraded(self) -> None:
        output = self.postprocess(
            "基础题",
            "根据条形磁铁静止方向，在图中标出地理方位。",
            information_carrier="单图识别",
            reasoning_chain="简单因果推理",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_common_multistate_control_circuit_stays_medium(self) -> None:
        output = self.postprocess(
            "中等题",
            "电热水壶主加热、保温和干烧保护的电路判断。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            state_count="多状态",
            constraint_count="多约束",
            knowledge_count="2-3个",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_standard_experiment_design_at_three_to_five_steps_stays_medium(self) -> None:
        output = self.postprocess(
            "中等题",
            "测量海螺壳密度，完成常规实验过程、方法交流与误差评价。",
            step_count="3-5步",
            problem_structure="实验探究",
            subquestion_dependency="多问且层层递进",
            experiment_requirement="方案设计或误差评价",
            knowledge_count="2-3个",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_glass_tube_process_can_remain_hard(self) -> None:
        output = self.postprocess(
            "拔高题",
            "玻璃管汲水：松开、下移、压紧、上提的顺序。",
            step_count="6-8步",
            reasoning_chain="逆向推理或临界分析",
            problem_structure="力学综合",
            state_count="多状态",
            constraint_count="单一约束",
            knowledge_count="2-3个",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_three_to_five_step_hidden_model_is_not_downgraded_from_hard(self) -> None:
        output = self.postprocess(
            "拔高题",
            "曳引式电梯中上下钢丝绳质量变化，需建立隐含受力模型判断运行状态。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="力学综合",
            knowledge_count="2-3个",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_three_states_normalizes_to_multiple_states(self) -> None:
        normalized = rating.normalize_features({"state_count": "三状态"})
        self.assertEqual(normalized["state_count"], "多状态")

    def test_lite_temperature_is_fixed_to_one(self) -> None:
        resolver = getattr(rating, "resolve_temperature", None)
        self.assertIsNotNone(resolver)
        if resolver is not None:
            self.assertEqual(resolver("doubao-seed-2.0-lite", "0"), 1.0)
            self.assertEqual(resolver("doubao-seed-2.0-mini", "0"), 0.0)

    def test_project_without_validation_is_not_final(self) -> None:
        output = self.postprocess(
            "拔高题",
            "设计带压敏电阻传感器的自动控制装置。",
            step_count="6-8步",
            reasoning_chain="逆向推理或临界分析",
            problem_structure="跨模块综合",
            cross_module="跨模块综合",
            state_count="多状态",
            constraint_count="多约束",
            variable_relation="多变量耦合关系",
            calculation_complexity="多公式联立",
            experiment_requirement="方案设计或误差评价",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_complex_final_is_preserved(self) -> None:
        output = self.postprocess(
            "压轴题",
            "多对象、多过程的检测方案，需分类讨论并验证可行性。",
            step_count="9-12步",
            reasoning_chain="逆向推理或临界分析",
            problem_structure="跨模块综合",
            cross_module="跨模块综合",
            state_count="连续变化或临界状态",
            constraint_count="多约束",
            variable_relation="多变量耦合关系",
            calculation_complexity="复杂方程或范围计算",
            experiment_requirement="方案设计或误差评价",
        )
        self.assertEqual(output["difficulty_level"], "压轴题")

    def test_final_is_not_downgraded_only_for_missing_upgrade_evidence(self) -> None:
        output = self.postprocess(
            "压轴题",
            "含两个连续状态的综合题。",
            step_count="6-8步",
            reasoning_chain="逆向推理或临界分析",
            state_count="多状态",
            constraint_count="多约束",
        )
        self.assertEqual(output["difficulty_level"], "压轴题")

    def test_untrusted_source_difficulty_is_renamed(self) -> None:
        safe = rating.sanitize_question_data({"question_id": "1", "difficulty": 3, "stem": "题目"})
        output = rating.make_output_base({"question_id": "1", "difficulty": 3, "stem": "题目"})
        self.assertNotIn("difficulty", safe)
        self.assertNotIn("difficulty", output)
        self.assertEqual(output["source_difficulty_untrusted"], 3)


if __name__ == "__main__":
    unittest.main()
