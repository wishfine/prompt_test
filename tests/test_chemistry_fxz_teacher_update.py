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


class ChemistryFxzTeacherUpdateTests(unittest.TestCase):
    def test_all_hard_to_final_rules_are_candidate_only(self) -> None:
        self.assertTrue(
            HARD_TO_FINAL_RULES.issubset(
                postprocess.TEACHER_GUARD_CANDIDATE_ONLY_RULES
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
        ):
            self.assertIn(text, prompt)

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


if __name__ == "__main__":
    unittest.main()
