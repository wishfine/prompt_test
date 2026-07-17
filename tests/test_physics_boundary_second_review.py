# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "physics_boundary_second_review.py"
spec = importlib.util.spec_from_file_location("physics_boundary_second_review", SCRIPT)
assert spec and spec.loader
review = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = review
spec.loader.exec_module(review)


def make_item(level: str, **features: str) -> dict:
    values = copy.deepcopy(review.rating.FEATURE_DEFAULTS)
    values.update(features)
    return {
        "question_id": "test-question",
        "stem": "测试题",
        "difficulty_rating": {
            "features": values,
            "reasoning": {
                "core_basis": "测试理由",
                "hard_point": "无",
                "why_not_lower": "测试",
                "why_not_higher": "测试",
            },
            "difficulty_level": level,
        },
        "postprocess_actions": [],
        "boundary_review_any_postprocess_adjustment": False,
    }


class BoundarySingleRunTests(unittest.TestCase):
    def test_source_item_keeps_single_run_without_vote(self) -> None:
        item = make_item("中等题")
        merged = review.prepare_source_item(item, "run1.jsonl")
        self.assertEqual(merged["boundary_review_source_prediction"], "中等题")
        self.assertEqual(merged["boundary_review_input_mode"], "single_no_vote")
        self.assertNotIn("boundary_review_consensus_level", merged)


class BoundarySelectionTests(unittest.TestCase):
    def test_mismatch_audit_includes_model_reference_and_neighbor_levels(self) -> None:
        item = make_item("中等题")
        item["adjudication_reference_level"] = "压轴题"
        candidate = review.select_boundary_candidate(item, "all")
        self.assertEqual(candidate["allowed_levels"], ["基础题", "中等题", "拔高题", "压轴题"])

    def test_clear_middle_structure_is_not_selected_in_selective_mode(self) -> None:
        item = make_item(
            "中等题",
            step_count="3-5步",
            formula_count="2-3个",
            calculation_complexity="简单笔算",
            reasoning_chain="多层因果推理",
            knowledge_count="2-3个",
            state_count="双状态",
            constraint_count="单一约束",
        )
        candidate = review.select_boundary_candidate(item, "selective")
        self.assertFalse(candidate["selected"])

    def test_low_structure_basic_is_selected(self) -> None:
        item = make_item("基础题", reasoning_chain="简单因果推理")
        candidate = review.select_boundary_candidate(item, "selective")
        self.assertTrue(candidate["selected"])
        self.assertEqual(candidate["allowed_levels"], ["送分题", "基础题", "中等题"])

    def test_postprocessed_item_is_selected(self) -> None:
        item = make_item("中等题")
        item["postprocess_actions"] = [{"from": "基础题", "to": "中等题", "rule": "test"}]
        candidate = review.select_boundary_candidate(item, "selective")
        self.assertTrue(candidate["selected"])


class BoundaryApplicationTests(unittest.TestCase):
    def review_result(self, level: str, confidence: str = "高") -> dict:
        return {
            "review_level": level,
            "acceptable_levels": [level],
            "boundary_status": "明确归档",
            "confidence": confidence,
            "decision": "调整",
            "effective_task_summary": "有效任务",
            "boundary_basis": "边界证据",
            "first_pass_issue": "首轮误判",
        }

    def test_high_confidence_adjacent_adjustment_is_applied(self) -> None:
        apply, reason = review.should_apply_review(
            "中等题",
            self.review_result("拔高题"),
            ["基础题", "中等题", "拔高题"],
            ["高"],
        )
        self.assertTrue(apply)
        self.assertIn("相邻档", reason)

    def test_medium_confidence_adjustment_is_not_applied_by_default(self) -> None:
        apply, _ = review.should_apply_review(
            "中等题",
            self.review_result("拔高题", "中"),
            ["基础题", "中等题", "拔高题"],
            ["高"],
        )
        self.assertFalse(apply)

    def test_non_adjacent_or_disallowed_adjustment_is_rejected(self) -> None:
        apply, _ = review.should_apply_review(
            "基础题",
            self.review_result("拔高题"),
            ["送分题", "基础题", "中等题"],
            ["高"],
        )
        self.assertFalse(apply)

    def test_apply_preserves_first_pass_and_updates_coarse_level(self) -> None:
        item = make_item("中等题")
        item["difficulty_rating_before_review"] = copy.deepcopy(item["difficulty_rating"])
        candidate = {
            "current_level": "中等题",
            "allowed_levels": ["基础题", "中等题", "拔高题"],
        }
        applied, _ = review.apply_review_to_item(item, candidate, self.review_result("拔高题"), ["高"])
        self.assertTrue(applied)
        self.assertEqual(item["difficulty_rating_before_review"]["difficulty_level"], "中等题")
        self.assertEqual(item["difficulty_rating"]["difficulty_level"], "拔高题")
        self.assertEqual(item["difficulty_rating"]["coarse_difficulty"], "中等/拔高区间（3-4档）")

    def test_adjacent_boundary_can_accept_both_reference_and_model(self) -> None:
        result = self.review_result("中等题")
        result["acceptable_levels"] = ["基础题", "中等题"]
        result["boundary_status"] = "相邻边界均可"
        self.assertIsNone(review.validate_review_result(result, ["基础题", "中等题", "拔高题"]))
        self.assertEqual(
            review.classify_reference_disagreement("基础题", "中等题", result),
            "相邻边界均可",
        )

    def test_reference_and_model_errors_are_classified_separately(self) -> None:
        model_supported = self.review_result("中等题")
        self.assertEqual(
            review.classify_reference_disagreement("基础题", "中等题", model_supported),
            "参考标签需修订",
        )
        reference_supported = self.review_result("基础题")
        self.assertEqual(
            review.classify_reference_disagreement("基础题", "中等题", reference_supported),
            "模型确实误判",
        )


class BoundaryPromptTests(unittest.TestCase):
    def test_prompt_has_conservative_and_structured_output_constraints(self) -> None:
        prompt = review.DEFAULT_REVIEW_PROMPT.read_text(encoding="utf-8")
        self.assertIn("证据不足", prompt)
        self.assertIn("最多移动一档", prompt)
        self.assertIn('"review_level"', prompt)
        self.assertIn('"acceptable_levels"', prompt)
        self.assertIn("相邻边界均可", prompt)
        self.assertIn("两张及以上定量关系图", prompt)


if __name__ == "__main__":
    unittest.main()
