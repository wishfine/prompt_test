from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "chemistry_difficulty_rating_with_cache.py"
SPEC = importlib.util.spec_from_file_location("chemistry_boundary_v4", MODULE_PATH)
assert SPEC and SPEC.loader
chemistry = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chemistry)


def boundary(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "task_count_W": "1",
        "rule_family_count_B": "1",
        "retrieval_pattern": "单一可复用规则",
        "curriculum_refs": ["U2"],
        "response_requirement": "选择或短填",
        "trap_complexity": "无明显陷阱",
        "quantitative_path": "无定量",
        "curriculum_scope": "初中课内",
    }
    value.update(overrides)
    return value


def features() -> dict[str, str]:
    value = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
    value.update(
        {
            "reasoning_depth": "2-3层",
            "reasoning_direction": "正向推导",
            "knowledge_relation": "同模块深度关联",
            "representation_conversion": "一次表征转换",
            "reaction_relation": "2-3个并列或简单连续反应",
            "constraint_complexity": "单一约束",
            "evidence_relation": "多条清晰证据联合",
            "experiment_requirement": "控制变量、现象解释或数据归纳",
            "graph_table_requirement": "多组比较归纳",
            "calculation_model": "单一方程式或关系式",
            "subquestion_dependency": "多问共享模型但无答案依赖",
        }
    )
    return value


def rating(level: str, boundary_value: dict[str, object]) -> dict[str, object]:
    low_features = copy.deepcopy(chemistry.FEATURE_DEFAULTS)
    low_features.update(
        {
            "reasoning_depth": "1层",
            "reasoning_direction": "正向推导",
            "knowledge_relation": "同模块简单关联",
        }
    )
    coarse = {
        "送分题": "送分/基础区间（1-2档）",
        "基础题": "送分/基础区间（1-2档）",
        "中等题": "基础/中等区间（2-3档）",
        "拔高题": "中等/拔高区间（3-4档）",
        "压轴题": "拔高/压轴区间（4-5档）",
    }[level]
    return {
        "features": low_features,
        "boundary_features": boundary_value,
        "coarse_difficulty": coarse,
        "reasoning": {
            "core_basis": "D、B、W与课程坐标测试",
            "hard_point": "测试",
            "why_not_lower": "测试",
            "why_not_higher": "测试",
        },
        "difficulty_level": level,
    }


class ChemistryBoundaryV4Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.old_general = chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK
        self.old_teacher = chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS
        self.old_teacher_writeback = (
            chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
        )
        self.old_boundary = chemistry.CHEMISTRY_ENABLE_BOUNDARY_V4_GUARDS
        self.old_boundary_writeback = (
            chemistry.CHEMISTRY_ENABLE_BOUNDARY_V4_GUARDS_WRITEBACK
        )
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = False
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = False
        chemistry.CHEMISTRY_ENABLE_BOUNDARY_V4_GUARDS = True
        chemistry.CHEMISTRY_ENABLE_BOUNDARY_V4_GUARDS_WRITEBACK = False

    def tearDown(self) -> None:
        chemistry.CHEMISTRY_ENABLE_LEVEL_WRITEBACK = self.old_general
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = self.old_teacher
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = (
            self.old_teacher_writeback
        )
        chemistry.CHEMISTRY_ENABLE_BOUNDARY_V4_GUARDS = self.old_boundary
        chemistry.CHEMISTRY_ENABLE_BOUNDARY_V4_GUARDS_WRITEBACK = (
            self.old_boundary_writeback
        )

    def test_w_three_and_four_are_distinct(self) -> None:
        self.assertEqual(chemistry.BOUNDARY_W_ORDER["3"], 3)
        self.assertEqual(chemistry.BOUNDARY_W_ORDER["4"], 4)
        self.assertNotIn("3-4", chemistry.BOUNDARY_W_ORDER)

    def test_curriculum_refs_are_strict_and_span_is_derived(self) -> None:
        validated = chemistry.validate_boundary_contract(
            boundary(curriculum_refs=["U2", "U6", "GENERAL"])
        )
        self.assertEqual(
            chemistry.derive_curriculum_span(validated),
            "跨2个单元",
        )
        with self.assertRaises(chemistry.ChemistrySchemaError):
            chemistry.validate_boundary_contract(
                boundary(curriculum_refs=["U2", "U2"])
            )

    def test_foundational_tool_does_not_create_fake_cross_unit_span(self) -> None:
        validated = chemistry.validate_boundary_contract(
            boundary(curriculum_refs=["U6", "GENERAL"])
        )
        self.assertEqual(
            chemistry.derive_curriculum_span(validated),
            "单单元",
        )

    def test_two_chemical_writing_tasks_do_not_reach_medium(self) -> None:
        signals = chemistry.boundary_basic_to_medium_evidence(
            boundary(
                task_count_W="2",
                rule_family_count_B="1",
                response_requirement="自主化学用语或方程式书写",
            )
        )
        self.assertEqual(signals, [])

    def test_three_reason_tasks_need_shared_rule_load(self) -> None:
        low = chemistry.boundary_basic_to_medium_evidence(
            boundary(
                task_count_W="3",
                rule_family_count_B="1",
                response_requirement="规范原因或现象表达",
            )
        )
        high = chemistry.boundary_basic_to_medium_evidence(
            boundary(
                task_count_W="3",
                rule_family_count_B="2",
                response_requirement="规范原因或现象表达",
            )
        )
        self.assertEqual(low, [])
        self.assertTrue(high)

    def test_four_tasks_and_two_rule_families_reach_medium(self) -> None:
        signals = chemistry.boundary_basic_to_medium_evidence(
            boundary(
                task_count_W="4",
                rule_family_count_B="2",
                curriculum_refs=["U2", "U6"],
            )
        )
        self.assertTrue(signals)

    def test_high_school_final_requires_objective_solve_signal(self) -> None:
        high = boundary(
            task_count_W="3",
            rule_family_count_B="3及以上",
            curriculum_refs=["HS"],
            curriculum_scope="高中内容",
            quantitative_path="特殊方法或综合定量",
        )
        self.assertFalse(
            chemistry.confirmed_high_school_scope_signal(
                high,
                {"stem": "根据所给材料判断物质性质"},
            )
        )
        self.assertTrue(
            chemistry.confirmed_high_school_scope_signal(
                high,
                {
                    "stem": "计算该样品的物质的量并求电子转移数",
                    "analysis": "使用 n=m/M 和氧化还原电子守恒。",
                },
            )
        )

    def test_boundary_contract_is_required_by_rating_contract(self) -> None:
        rating = {
            "features": features(),
            "coarse_difficulty": "基础/中等区间（2-3档）",
            "reasoning": {
                "core_basis": "测试",
                "hard_point": "测试",
                "why_not_lower": "测试",
                "why_not_higher": "测试",
            },
            "difficulty_level": "中等题",
        }
        with self.assertRaises(chemistry.ChemistrySchemaError):
            chemistry.validate_rating_contract(rating)
        rating["boundary_features"] = boundary()
        validated = chemistry.validate_rating_contract(rating)
        self.assertEqual(validated["curriculum_span"], "单单元")

    def test_boundary_candidate_is_audited_without_writeback(self) -> None:
        result = chemistry.postprocess_chemistry_difficulty(
            rating(
                "基础题",
                boundary(
                    task_count_W="4",
                    rule_family_count_B="2",
                    curriculum_refs=["U2", "U6"],
                ),
            ),
            {"stem": "综合判断四项不同化学任务"},
        )
        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["boundary_v4_guard_candidate_level"],
            "中等题",
        )
        self.assertEqual(
            result["boundary_v4_guard_candidate_action"]["rule"],
            "boundary_v4_basic_to_medium_load_floor",
        )

    def test_boundary_candidate_can_be_explicitly_written_back(self) -> None:
        chemistry.CHEMISTRY_ENABLE_BOUNDARY_V4_GUARDS_WRITEBACK = True
        result = chemistry.postprocess_chemistry_difficulty(
            rating(
                "基础题",
                boundary(task_count_W="4", rule_family_count_B="2"),
            ),
            {"stem": "综合判断四项不同化学任务"},
        )
        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertTrue(result["automatic_level_change_applied"])

    def test_verified_teacher_guard_has_priority_over_boundary_writeback(
        self,
    ) -> None:
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
        chemistry.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = True
        chemistry.CHEMISTRY_ENABLE_BOUNDARY_V4_GUARDS_WRITEBACK = True
        value = rating(
            "送分题",
            boundary(
                task_count_W="4",
                rule_family_count_B="3及以上",
                retrieval_pattern="多规则混合任务",
                curriculum_refs=["U1", "U2"],
                response_requirement="规范原因或现象表达",
            ),
        )
        value["features"]["experiment_requirement"] = "基础操作或读数"

        result = chemistry.postprocess_chemistry_difficulty(
            value,
            {"stem": "完成多项化学实验与规范解释"},
        )

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["boundary_v4_guard_candidate_level"],
            "中等题",
        )
        self.assertEqual(result["combined_guard_candidate_level"], "基础题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "teacher_easy_to_basic_experiment_application",
        )
        self.assertFalse(result["boundary_v4_guard_writeback_applied"])

    def test_v4_prompt_contains_curriculum_coordinate_and_exact_w_bins(self) -> None:
        prompt = (ROOT / "prompts" / "初中化学难度打标提示词.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn("curriculum_refs", prompt)
        self.assertIn("U1—U11", prompt)
        self.assertIn(
            "`task_count_W`：`1`、`2`、`3`、`4`、`5及以上`",
            prompt,
        )
        self.assertNotIn('"task_count_W": "1 | 2 | 3-4 | 5及以上"', prompt)


if __name__ == "__main__":
    unittest.main()
