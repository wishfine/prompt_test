from __future__ import annotations

import copy
from pathlib import Path
import unittest

from src import chemistry_postprocess_fxz as postprocess


ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = ROOT / "prompts" / "初中化学难度打标提示词_zyl.txt"


HARD_TO_FINAL_RULES = {
    "teacher_hard_to_final_dense_multiquestion_quantitative_chain",
    "teacher_hard_to_final_multistage_multiquestion_multireaction",
    "teacher_hard_to_final_strict_deep_quantitative_chain",
}


def hard_rating() -> dict:
    return {
        "features": {
            "longest_solution_chain": [
                "读取多问共享信息",
                "确定第一阶段反应",
                "建立多反应定量关系",
                "计算后一阶段目标量",
                "核验最终组成",
            ],
            "task_groups": [
                {"task_type": "性质与反应判断", "count": 3},
                {"task_type": "定量计算", "count": 4},
            ],
            "rule_families": [
                "反应关系或条件判断",
                "定量关系与计算",
            ],
            "curriculum_topics": ["U5-1", "U5-2"],
            "parallel_task_relation": "共享同一化学模型的关联任务",
            "solution_topology": "多阶段反应网络",
            "reaction_structure": "产物进入后一反应",
            "condition_operations": ["条件切换"],
            "representation_operations": ["化学方程式→定量关系"],
            "evidence_operations": [],
            "experiment_operation": "无",
            "experiment_task_structure": "无实验判断",
            "visual_task_structure": "无必要视觉信息",
            "graph_table_operation": "无",
            "error_analysis_operation": "无误差分析",
            "calculation_operations": ["多反应定量关系"],
            "new_information_operation": "无新信息",
        },
        "coarse_difficulty": "中等/拔高区间（3-4档）",
        "reasoning": {
            "core_basis": "多问共享多阶段反应模型。",
            "hard_point": "需要建立多反应定量关系。",
            "why_not_lower": "存在连续定量链。",
            "why_not_higher": "原始模型判断为拔高。",
        },
        "difficulty_level": "拔高题",
    }


def easy_rating() -> dict:
    item = hard_rating()
    item["features"].update(
        {
            "longest_solution_chain": ["直接填写教材事实"],
            "task_groups": [
                {"task_type": "直接事实与概念", "count": 4}
            ],
            "rule_families": ["教材事实直接匹配"],
            "curriculum_topics": ["U1-1"],
            "parallel_task_relation": "同一规则下多个对象",
            "solution_topology": "单点直接回答",
            "reaction_structure": "无反应任务",
            "condition_operations": [],
            "representation_operations": [],
            "evidence_operations": [],
            "experiment_operation": "无",
            "experiment_task_structure": "无实验判断",
            "visual_task_structure": "无必要视觉信息",
            "graph_table_operation": "无",
            "error_analysis_operation": "无误差分析",
            "calculation_operations": [],
            "new_information_operation": "无新信息",
        }
    )
    item["coarse_difficulty"] = "送分/基础区间（1-2档）"
    item["difficulty_level"] = "送分题"
    return item


class ChemistryFxzTeacherUpdateTests(unittest.TestCase):
    def test_all_hard_to_final_rules_are_candidate_only(self) -> None:
        self.assertTrue(
            HARD_TO_FINAL_RULES.issubset(
                postprocess.TEACHER_GUARD_CANDIDATE_ONLY_RULES
            )
        )
        self.assertTrue(
            postprocess.is_teacher_guard_candidate_only_rule(
                "teacher_hard_to_final_future_rule"
            )
        )

    def test_dense_hard_to_final_candidate_does_not_write_back(self) -> None:
        result = postprocess.postprocess_chemistry_difficulty(
            copy.deepcopy(hard_rating()),
            {
                "stem": "多问共享多阶段反应与定量关系。",
                "sub_questions": [
                    {"stem": f"任务{i}"} for i in range(1, 5)
                ],
            },
            teacher_distribution_guards_enabled=True,
            teacher_distribution_guards_writeback_enabled=True,
        )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "压轴题",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )
        self.assertEqual(result["postprocess_actions"], [])
        self.assertIn(
            "拔高→压轴规则已关闭写回",
            result["teacher_distribution_guard_writeback_blocked_reason"],
        )

    def test_prompt_anchors_haber_bond_energy_as_teacher_final(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        for text in (
            "哈伯法合成氨",
            "键能—反应热模型",
            "教师等级：压轴题",
            "不得被同题其余初中常规小问平均降档",
            "两路制气",
            "通气先后",
            "末端气体定量结果反推产量",
        ):
            self.assertIn(text, prompt)

    def test_prompt_sets_four_fill_blank_questions_to_basic_floor(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")

        self.assertIn("四个及以上真实填空小问至少判为基础题", prompt)

    def test_four_fill_blank_subquestions_write_easy_back_to_basic(
        self,
    ) -> None:
        result = postprocess.postprocess_chemistry_difficulty(
            easy_rating(),
            {
                "stem": "请完成填空。",
                "sub_questions": [
                    {"stem": f"填空{i}：______", "options": ""}
                    for i in range(1, 5)
                ],
            },
            teacher_distribution_guards_enabled=True,
            teacher_distribution_guards_writeback_enabled=True,
        )

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "teacher_easy_to_basic_four_fill_blank_subquestions",
        )

    def test_three_fill_blank_subquestions_do_not_trigger_floor(self) -> None:
        result = postprocess.postprocess_chemistry_difficulty(
            easy_rating(),
            {
                "stem": "请完成填空。",
                "sub_questions": [
                    {"stem": f"填空{i}：______", "options": ""}
                    for i in range(1, 4)
                ],
            },
            teacher_distribution_guards_enabled=True,
            teacher_distribution_guards_writeback_enabled=True,
        )

        self.assertEqual(result["difficulty_level"], "送分题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_parallel_single_step_conversions_do_not_form_reaction_floor(
        self,
    ) -> None:
        data = {
            "stem": "在给定条件下，下列物质间的转化不能实现的是",
            "options": (
                "A. Fe → Fe2O3\n"
                "B. CaCO3 → CO2\n"
                "C. H2 → H2O\n"
                "D. Ca(OH)2 → NaOH"
            ),
        }

        self.assertIsNone(postprocess.reaction_validation_floor_signal(data))

    def test_new_teacher_labels_and_sample_are_id_aligned(self) -> None:
        label_path = (
            ROOT
            / "data/labeled/"
            "lite_chemistry_random500_seed20260814_teacher_label.jsonl"
        )
        sample_path = (
            ROOT
            / "data/samples/"
            "lite_chemistry_random500_seed20260814_sample.jsonl"
        )

        import json

        labels = [
            json.loads(line)
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        samples = [
            json.loads(line)
            for line in sample_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        label_ids = [str(row["question_id"]) for row in labels]
        sample_ids = [str(row["question_id"]) for row in samples]

        self.assertEqual(len(label_ids), 500)
        self.assertEqual(len(set(label_ids)), 500)
        self.assertEqual(sample_ids, label_ids)
        target = next(
            row
            for row in labels
            if str(row["question_id"]) == "2754474790292570112"
        )
        self.assertEqual(target["standard_level_name"], "压轴题")
        additional_target = next(
            row
            for row in labels
            if str(row["question_id"]) == "2811150975861800960"
        )
        self.assertEqual(additional_target["standard_level_name"], "压轴题")


if __name__ == "__main__":
    unittest.main()
