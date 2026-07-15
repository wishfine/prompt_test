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
    def setUp(self) -> None:
        self.original_profile = rating.RATING_PROFILE
        rating.RATING_PROFILE = "generalized"

    def tearDown(self) -> None:
        rating.RATING_PROFILE = self.original_profile

    def postprocess(self, level: str, stem: str, **feature_values: str) -> dict:
        return rating.postprocess_physics_difficulty(result(level, **feature_values), {"question_id": "test", "stem": stem})

    def test_v7_stable_profile_is_available(self) -> None:
        self.assertIn("v7_stable", rating.VALID_RATING_PROFILES)

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


class V7CompatPostprocessTests(unittest.TestCase):
    """锁定历史 120/133 版本最关键的语义保护行为。"""

    def setUp(self) -> None:
        self.original_profile = getattr(rating, "RATING_PROFILE", None)
        self.assertIsNotNone(self.original_profile, "脚本必须提供 RATING_PROFILE")
        rating.RATING_PROFILE = "v7_compat"

    def tearDown(self) -> None:
        if self.original_profile is not None:
            rating.RATING_PROFILE = self.original_profile

    def postprocess(self, level: str, stem: str, **feature_values: str) -> dict:
        return rating.postprocess_physics_difficulty(
            result(level, **feature_values),
            {"question_id": "v7-compat-test", "stem": stem},
        )

    def test_low_calculation_high_modeling_reaches_hard(self) -> None:
        output = self.postprocess(
            "中等题",
            "用手端起餐盘并保持水平静止，画出受力并分析支点、力臂和动力变化。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="力学综合",
            additional_structure="力学约束",
            knowledge_count="2-3个",
            error_risk="明显易错点",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_regular_pressure_scale_stays_medium(self) -> None:
        output = self.postprocess(
            "中等题",
            "普通压敏电阻压力秤，依据 R-F 图像、欧姆定律和电表量程求最大测量压力。",
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

    def test_project_validation_keeps_final_channel(self) -> None:
        output = self.postprocess(
            "拔高题",
            "项目挑战：将托盘天平改装成液体密度测量仪，设计标尺并验证量程能否覆盖全部待测液体。",
            step_count="6-8步",
            formula_count="4-6个",
            calculation_complexity="复杂方程或范围计算",
            reasoning_chain="逆向推理或临界分析",
            problem_structure="跨模块综合",
            additional_structure="实验探究",
            information_carrier="多图表综合",
            reality_question="是",
            subquestion_dependency="多问且层层递进",
            knowledge_count="4个及以上",
            knowledge_diff="高",
            cross_module="跨模块综合",
            state_count="多状态",
            constraint_count="多约束",
            variable_relation="多变量耦合关系",
            experiment_requirement="方案设计或误差评价",
            graph_table_requirement="图像反推或外推",
            error_risk="高易错点",
        )
        self.assertEqual(output["difficulty_level"], "压轴题")
        self.assertEqual(output["difficulty_level_raw"], "拔高题")
        self.assertTrue(output["postprocess_actions"])


class V7StablePostprocessTests(unittest.TestCase):
    """旧 V7 主体不变，只修正已复现的稳定边界和规则波动。"""

    def setUp(self) -> None:
        self.original_profile = rating.RATING_PROFILE
        rating.RATING_PROFILE = "v7_stable"

    def tearDown(self) -> None:
        rating.RATING_PROFILE = self.original_profile

    def postprocess(self, level: str, stem: str, sub_questions: list | None = None, **feature_values: str) -> dict:
        return rating.postprocess_physics_difficulty(
            result(level, **feature_values),
            {"question_id": "v7-stable-test", "stem": stem, "sub_questions": sub_questions or []},
        )

    def test_routine_two_level_heater_is_medium(self) -> None:
        output = self.postprocess(
            "拔高题",
            "家用电热水壶有加热挡和保温挡，先求吸热量、功率，再按明确电路计算两段电热丝长度。",
            sub_questions=[{"stem": "求吸热量。"}, {"stem": "求保温功率。"}, {"stem": "补全电路并计算长度。"}],
            step_count="6-8步",
            formula_count="4-6个",
            calculation_complexity="多公式联立",
            reasoning_chain="多层因果推理",
            problem_structure="跨模块综合",
            subquestion_dependency="多问且层层递进",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
            state_count="双状态",
            constraint_count="单一约束",
            variable_relation="简单正反比",
        )
        self.assertEqual(output["difficulty_level"], "中等题")
        self.assertEqual(len(output["postprocess_actions"]), 1)
        self.assertEqual(output["postprocess_actions"][0]["rule"], "v7_stable_routine_heater_guard")

    def test_standard_circuit_connection_stays_basic(self) -> None:
        output = self.postprocess(
            "基础题",
            "根据电路图将实物连接完整，并为电压表选择量程。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            additional_structure="电路约束",
            information_carrier="电路图",
            knowledge_count="2-3个",
            constraint_count="单一约束",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_round_trip_material_question_reaches_medium(self) -> None:
        output = self.postprocess(
            "基础题",
            "雷达料位器记录电磁波往返时间，先识别波段，再计算液面高度并判断罐底液体压强变化。",
            sub_questions=[{"stem": "识别波段。"}, {"stem": "计算液面高度。"}, {"stem": "判断压强变化。"}],
            step_count="3-5步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="简单因果推理",
            problem_structure="跨模块综合",
            subquestion_dependency="多问但相互独立",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
            variable_relation="简单正反比",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_two_independent_direct_calculations_stay_basic(self) -> None:
        output = self.postprocess(
            "中等题",
            "太阳能热水器中分别直接计算能量转化效率和额定电流，据此选择空气开关。",
            sub_questions=[{"stem": "求效率。"}, {"stem": "求额定电流并选择空气开关。"}],
            step_count="3-5步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="多层因果推理",
            problem_structure="跨模块综合",
            subquestion_dependency="多问但相互独立",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
            state_count="单状态",
            constraint_count="单一约束",
            variable_relation="简单正反比",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_four_standard_experiments_reach_medium(self) -> None:
        output = self.postprocess(
            "基础题",
            "四个实验分别涉及杠杆平衡调节、托里拆利实验误差、扩散观察和分子间隙解释。",
            sub_questions=[{"stem": str(i)} for i in range(4)],
            step_count="1-2步",
            reasoning_chain="简单因果推理",
            problem_structure="实验探究",
            information_carrier="多图表综合",
            subquestion_dependency="多问但相互独立",
            knowledge_count="4个及以上",
            cross_module="跨模块综合",
            experiment_requirement="基础操作或读数",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_simple_buoyancy_state_change_stays_basic(self) -> None:
        output = self.postprocess(
            "基础题",
            "悬浮的鹦鹉螺把海水吸进气室后将怎样运动？",
            step_count="1-2步",
            reasoning_chain="简单因果推理",
            problem_structure="力学综合",
            knowledge_count="1个",
            state_count="双状态",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_explicit_multistate_control_choice_stays_medium(self) -> None:
        output = self.postprocess(
            "中等题",
            "电热水壶沸腾时主加热盘停止，降温后副加热盘保温，干烧时全部断开，选择符合要求的电路。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            information_carrier="电路图",
            knowledge_count="2-3个",
            state_count="多状态",
            constraint_count="多约束",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_magnetic_direction_diagram_stays_basic(self) -> None:
        output = self.postprocess(
            "基础题",
            "根据小磁针N极指向画出磁感线方向，并标出电源正极。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            information_carrier="单图识别",
            knowledge_count="2-3个",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_cross_module_parallel_choice_stays_basic(self) -> None:
        output = self.postprocess(
            "基础题",
            "火箭升空形成神箭凌日，四个选项分别判断重力势能、热机效率、运动相对性和光的直线传播。",
            step_count="1-2步",
            reasoning_chain="简单因果推理",
            problem_structure="力学综合",
            information_carrier="单图识别",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_two_mode_changeover_circuit_reaches_medium(self) -> None:
        output = self.postprocess(
            "基础题",
            "用太阳能板、可充电电池、单刀双掷开关和电动机设计电路：供电时电动机工作，充电时电动机不工作。",
            step_count="3-5步",
            reasoning_chain="简单因果推理",
            problem_structure="电路综合",
            additional_structure="电路约束",
            information_carrier="电路图",
            knowledge_count="1个",
            state_count="单状态",
            constraint_count="单一约束",
        )
        self.assertEqual(output["difficulty_level"], "中等题")


class FusedPostprocessTests(unittest.TestCase):
    """融合最终版：稳定边界优先，禁止单个易波动 feature 触发升档。"""

    def setUp(self) -> None:
        self.original_profile = rating.RATING_PROFILE
        self.assertIn("fused", rating.VALID_RATING_PROFILES)
        rating.RATING_PROFILE = "fused"

    def tearDown(self) -> None:
        rating.RATING_PROFILE = self.original_profile

    def postprocess(self, level: str, stem: str, sub_questions: list | None = None, **feature_values: str) -> dict:
        return rating.postprocess_physics_difficulty(
            result(level, **feature_values),
            {"question_id": "fused-test", "stem": stem, "sub_questions": sub_questions or []},
        )

    def test_standard_circuit_connection_stays_basic(self) -> None:
        output = self.postprocess(
            "基础题",
            "根据电路图将实物连接完整，并为电压表选择量程。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            additional_structure="电路约束",
            information_carrier="电路图",
            knowledge_count="2-3个",
            constraint_count="单一约束",
            error_risk="明显易错点",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_standard_circuit_connection_can_correct_medium_to_basic(self) -> None:
        output = self.postprocess(
            "中等题",
            "根据电路图将实物连接完整。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            additional_structure="电路约束",
            information_carrier="电路图",
            knowledge_count="2-3个",
            constraint_count="单一约束",
            error_risk="明显易错点",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_textbook_quantity_estimate_stays_easy(self) -> None:
        output = self.postprocess(
            "送分题",
            "关于公交车的说法，与实际相符的是：普通公交车长度约10m。",
            reasoning_chain="简单因果推理",
            knowledge_count="4个及以上",
            error_risk="轻微易错点",
        )
        self.assertEqual(output["difficulty_level"], "送分题")

    def test_photo_scale_estimate_stays_basic(self) -> None:
        output = self.postprocess(
            "基础题",
            "由照片估测，凤蝶两翅展开的实际长度约为多少？",
            reasoning_chain="简单因果推理",
            information_carrier="单图识别",
            knowledge_count="1个",
            error_risk="轻微易错点",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_independent_device_facts_and_direct_calculation_stay_basic(self) -> None:
        output = self.postprocess(
            "基础题",
            "电动晾衣装置：(1)判断电磁铁磁性；(2)判断电流方向；(3)匀速上升时求功和功率。",
            step_count="3-5步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="简单因果推理",
            problem_structure="跨模块综合",
            additional_structure="跨模块",
            reality_question="是",
            subquestion_dependency="多问但相互独立",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_four_standard_experiment_items_reach_medium_despite_knowledge_drift(self) -> None:
        output = rating.postprocess_physics_difficulty(
            result(
                "基础题",
                problem_structure="实验探究",
                additional_structure="图像表格",
                information_carrier="多图表综合",
                subquestion_dependency="多问但相互独立",
                knowledge_count="2-3个",
                cross_module="同一模块内部",
                experiment_requirement="基础操作或读数",
                graph_table_requirement="直接读数",
                error_risk="轻微易错点",
            ),
            {
                "question_id": "test",
                "stem": "四个独立实验。",
                "sub_questions": [
                    {"stem": "调节杠杆。"},
                    {"stem": "托里拆利实验误差。"},
                    {"stem": "扩散实验为什么不能搅拌？"},
                    {"stem": "水和酒精混合实验。"},
                ],
            },
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_standard_magnetic_drawing_stays_basic_despite_feature_drift(self) -> None:
        output = self.postprocess(
            "基础题",
            "根据小磁针N极指向，画出磁感线方向并标出电源正极。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            additional_structure="电路约束",
            information_carrier="电路图",
            knowledge_count="2-3个",
            constraint_count="单一约束",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_routine_two_mode_heater_is_not_force_downgraded_from_hard(self) -> None:
        output = self.postprocess(
            "拔高题",
            "家用电热水壶有加热和保温两个挡位，电路结构明确，计算吸热量、功率并按给定规格确定两段电热丝长度。",
            step_count="6-8步",
            formula_count="4-6个",
            calculation_complexity="多公式联立",
            reasoning_chain="多层因果推理",
            problem_structure="跨模块综合",
            additional_structure="跨模块",
            subquestion_dependency="多问且层层递进",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
            state_count="双状态",
            constraint_count="单一约束",
            variable_relation="简单正反比",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_independent_direct_calculations_do_not_force_medium_to_basic(self) -> None:
        output = self.postprocess(
            "中等题",
            "太阳能热水器包含两个相互独立的小问。",
            sub_questions=[
                {"stem": "直接代入比热容和效率公式计算效率。"},
                {"stem": "由额定功率直接计算电流并选择空气开关。"},
            ],
            step_count="3-5步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="简单因果推理",
            problem_structure="跨模块综合",
            subquestion_dependency="多问但相互独立",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
            state_count="单状态",
            constraint_count="单一约束",
            variable_relation="简单正反比",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_cross_module_material_with_round_trip_model_reaches_medium(self) -> None:
        output = self.postprocess(
            "基础题",
            "雷达料位器通过电磁波往返时间监测液面。",
            sub_questions=[
                {"stem": "根据波谱判断电磁波种类。"},
                {"stem": "由往返时间求单程距离，再求液面高度。"},
                {"stem": "解释液面下降时罐底压强变化。"},
            ],
            step_count="3-5步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="简单因果推理",
            problem_structure="跨模块综合",
            information_carrier="图像或表格",
            subquestion_dependency="多问但相互独立",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
            variable_relation="简单正反比",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_multiple_standard_experiment_analyses_reach_medium(self) -> None:
        output = self.postprocess(
            "基础题",
            "请运用所学物理知识分析下列实验。",
            sub_questions=[
                {"stem": "判断杠杆是否平衡并调节平衡螺母。"},
                {"stem": "读取水银柱并判断混入空气造成的误差方向。"},
                {"stem": "说明扩散实验为什么不能搅动液体。"},
                {"stem": "根据混合体积判断分子间存在间隙。"},
            ],
            step_count="1-2步",
            reasoning_chain="简单因果推理",
            problem_structure="实验探究",
            additional_structure="实验探究",
            information_carrier="多图表综合",
            subquestion_dependency="多问但相互独立",
            knowledge_count="4个及以上",
            cross_module="跨模块综合",
            experiment_requirement="基础操作或读数",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_routine_automatic_kettle_stays_medium_when_one_feature_drifts(self) -> None:
        output = self.postprocess(
            "中等题",
            "电热水壶包含加热、保温和干烧保护，开关状态和工作链均已明确，只判断各状态工作情况。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            cross_module="跨模块综合",
            state_count="多状态",
            constraint_count="多约束",
            knowledge_count="2-3个",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_project_with_range_coverage_keeps_final_despite_feature_drift(self) -> None:
        output = self.postprocess(
            "拔高题",
            "液体密度测量挑战赛：将天平改装为密度测量仪，设计标尺，并判断量程是否覆盖全部待测液体，验证方案可行性。",
            step_count="6-8步",
            calculation_complexity="多公式联立",
            reasoning_chain="多层因果推理",
            problem_structure="跨模块综合",
            subquestion_dependency="多问且层层递进",
            knowledge_count="4个及以上",
            cross_module="跨模块综合",
            state_count="多状态",
            constraint_count="单一约束",
            variable_relation="图像函数关系",
            experiment_requirement="方案设计或误差评价",
        )
        self.assertEqual(output["difficulty_level"], "压轴题")

    def test_single_object_textbook_force_diagram_can_stay_easy(self) -> None:
        output = self.postprocess(
            "送分题",
            "画出静止在水平地面上篮球的受力示意图。",
            information_carrier="单图识别",
        )
        self.assertEqual(output["difficulty_level"], "送分题")

    def test_life_application_is_not_downgraded_to_easy(self) -> None:
        output = self.postprocess(
            "基础题",
            "制作水瓶琴时，改变瓶内水量后判断音调变化。",
            reasoning_chain="简单因果推理",
            problem_structure="光学声学综合",
            information_carrier="单图识别",
            knowledge_count="2-3个",
            error_risk="轻微易错点",
        )
        self.assertEqual(output["difficulty_level"], "基础题")

    def test_standard_reflection_experiment_stays_medium(self) -> None:
        output = self.postprocess(
            "中等题",
            "探究光的反射定律，依据实验步骤和多组数据解释并评估实验结论。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="实验探究",
            subquestion_dependency="多问且层层递进",
            experiment_requirement="控制变量或故障分析",
            graph_table_requirement="多组比较归纳",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_standard_reflection_experiment_reaches_medium_despite_feature_drift(self) -> None:
        output = self.postprocess(
            "基础题",
            "在探究光的反射定律实验中，多次改变入射角，分析表格数据、纸板折转现象和光路可逆性。",
            step_count="1-2步",
            reasoning_chain="简单因果推理",
            problem_structure="实验探究",
            information_carrier="单图识别",
            knowledge_count="2-3个",
            experiment_requirement="基础操作或读数",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_pressure_scale_stays_medium_despite_step_and_constraint_drift(self) -> None:
        output = self.postprocess(
            "中等题",
            "普通压力秤使用压敏电阻R-F图像、欧姆定律和电表量程求最大测量压力。",
            step_count="6-8步",
            formula_count="2-3个",
            calculation_complexity="多公式联立",
            reasoning_chain="多层因果推理",
            problem_structure="电路综合",
            constraint_count="多约束",
            variable_relation="图像函数关系",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_process_sequence_hard_is_not_promoted_to_final(self) -> None:
        output = self.postprocess(
            "拔高题",
            "两端开口玻璃管汲水，判断压紧管口、快速上提和松开管口的正确操作顺序。",
            step_count="6-8步",
            reasoning_chain="逆向推理或临界分析",
            state_count="多状态",
            constraint_count="多约束",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_experiment_error_evaluation_hard_is_not_promoted_to_final(self) -> None:
        output = self.postprocess(
            "拔高题",
            "测量多种螺栓质量和体积，根据图像判断材料，并评价气泡猜想、提出新猜想。",
            step_count="6-8步",
            calculation_complexity="简单笔算",
            reasoning_chain="多层因果推理",
            problem_structure="实验探究",
            subquestion_dependency="多问且层层递进",
            constraint_count="单一约束",
            variable_relation="图像函数关系",
            experiment_requirement="方案设计或误差评价",
            graph_table_requirement="图像反推或外推",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_non_project_nine_step_deep_coupling_reaches_final(self) -> None:
        output = self.postprocess(
            "拔高题",
            "多个起重工具共同作用，需要联立机械效率、功率、速度关系并根据多项条件完成筛选。",
            step_count="9-12步",
            formula_count="4-6个",
            calculation_complexity="多公式联立",
            reasoning_chain="逆向推理或临界分析",
            problem_structure="力学综合",
            knowledge_count="4个及以上",
            state_count="双状态",
            constraint_count="多约束",
            variable_relation="多变量耦合关系",
        )
        self.assertEqual(output["difficulty_level"], "压轴题")

    def test_plate_hidden_lever_model_reaches_hard(self) -> None:
        output = self.postprocess(
            "中等题",
            "用手端起餐盘并保持水平静止，先画受力，再判断支点、力臂和动力如何变化。",
            sub_questions=[{"stem": "画受力。"}, {"stem": "分析支点、力臂和动力。"}],
            step_count="3-5步",
            reasoning_chain="简单因果推理",
            problem_structure="力学综合",
            information_carrier="单图识别",
            subquestion_dependency="多问但相互独立",
            knowledge_count="2-3个",
            knowledge_diff="低",
            variable_relation="简单正反比",
            error_risk="轻微易错点",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_glass_tube_process_reaches_hard_despite_state_feature_drift(self) -> None:
        output = self.postprocess(
            "中等题",
            "用两端开口的长玻璃管从桶中汲水，从压紧或松开管口、快速上提或下移中选择正确操作顺序。",
            step_count="3-5步",
            reasoning_chain="多层因果推理",
            problem_structure="力学综合",
            reality_question="是",
            state_count="双状态",
            error_risk="明显易错点",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_air_conditioner_safety_design_reaches_hard_without_feature_luck(self) -> None:
        output = self.postprocess(
            "中等题",
            "学校给教室加装两台空调，同学们参与线路的设计并完成电路连接，结合空气开关和导线载流量判断安全方案。",
            step_count="3-5步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="简单因果推理",
            problem_structure="跨模块综合",
            subquestion_dependency="多问但相互独立",
            knowledge_count="4个及以上",
            cross_module="跨模块综合",
            constraint_count="单一约束",
            experiment_requirement="无",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_routine_heat_chain_is_not_downgraded_from_medium(self) -> None:
        output = self.postprocess(
            "中等题",
            "燃气热水器把水加热，已知质量、温升、效率和热值，依次求吸热量及天然气体积。",
            sub_questions=[{"stem": "求吸热量。"}, {"stem": "求天然气体积。"}],
            step_count="1-2步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="简单因果推理",
            problem_structure="热学综合",
            subquestion_dependency="多问但相互独立",
            knowledge_count="2-3个",
            variable_relation="简单正反比",
        )
        self.assertEqual(output["difficulty_level"], "中等题")

    def test_validated_project_hard_reaches_final_despite_feature_drift(self) -> None:
        output = self.postprocess(
            "拔高题",
            "项目式自动饮水装置需要设计控制与加热电路，依据传感器图像、安全电流和温度范围筛选方案，并判断是否可行。",
            step_count="3-5步",
            formula_count="4-6个",
            calculation_complexity="多公式联立",
            reasoning_chain="多层因果推理",
            problem_structure="跨模块综合",
            information_carrier="多图表综合",
            reality_question="是",
            subquestion_dependency="多问且层层递进",
            knowledge_count="4个及以上",
            cross_module="跨模块综合",
            state_count="单状态",
            constraint_count="单一约束",
            variable_relation="简单正反比",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "压轴题")

    def test_project_temperature_control_medium_reaches_hard_semantically(self) -> None:
        output = self.postprocess(
            "中等题",
            "【项目要求】制作电加热装置并自动控温，使用热敏电阻、电磁继电器和电热丝，结合安全工作电流、温度范围并说明第二种方案是否满足自动控温要求。",
            step_count="3-5步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="简单因果推理",
            problem_structure="跨模块综合",
            knowledge_count="2-3个",
            cross_module="跨模块综合",
            constraint_count="单一约束",
            variable_relation="简单正反比",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "拔高题")

    def test_complex_expression_and_adjustment_hard_reaches_final(self) -> None:
        output = self.postprocess(
            "拔高题",
            "自动饮水装置含压敏电阻图像和进水控制机构，前三问完成电路计算与受力分析，再用物理量符号写出表达式，最后说明怎样调整结构。",
            sub_questions=[{"stem": "压强。"}, {"stem": "传感电路。"}, {"stem": "用物理量符号写出表达式。"}, {"stem": "怎样调整结构。"}],
            step_count="3-5步",
            formula_count="4-6个",
            calculation_complexity="多公式联立",
            reasoning_chain="多层因果推理",
            problem_structure="跨模块综合",
            information_carrier="多图表综合",
            subquestion_dependency="多问且层层递进",
            knowledge_count="4个及以上",
            cross_module="跨模块综合",
            constraint_count="单一约束",
            variable_relation="简单正反比",
            graph_table_requirement="直接读数",
        )
        self.assertEqual(output["difficulty_level"], "压轴题")


if __name__ == "__main__":
    unittest.main()
