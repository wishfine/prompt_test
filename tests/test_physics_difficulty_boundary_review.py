# -*- coding: utf-8 -*-
"""相邻边界复核的离线测试，不访问模型服务。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physics_difficulty_boundary_review as boundary  # noqa: E402
import physics_difficulty_rating_with_cache as rating  # noqa: E402


def make_item(level: str, *, features: dict[str, str] | None = None, **extra) -> dict:
    values = dict(rating.FEATURE_DEFAULTS)
    values.update(features or {})
    item = {
        "question_id": f"test-{level}",
        "stem": "根据题意完成判断。",
        "options": "A.甲 B.乙",
        "analysis": "根据一个显性物理规律即可判断。",
        "difficulty_rating": {
            "features": values,
            "reasoning": {
                "core_basis": "测试",
                "hard_point": "无",
                "why_not_lower": "旧解释",
                "why_not_higher": "未达到拔高题。",
            },
            "difficulty_level": level,
            "postprocess_actions": [],
        },
    }
    item.update(extra)
    return item


class BoundaryRouteTests(unittest.TestCase):
    def test_low_structure_medium_routes_to_basic_medium(self) -> None:
        route = boundary.route_boundary_review(make_item("中等题"), "risk")
        self.assertTrue(route["selected"])
        self.assertEqual(route["review_boundary"], ["基础题", "中等题"])

    def test_postprocess_action_has_boundary_priority(self) -> None:
        item = make_item("拔高题")
        item["difficulty_rating"]["postprocess_actions"] = [
            {"rule": "test", "from": "中等题", "to": "拔高题", "evidence": ["x"]}
        ]
        route = boundary.route_boundary_review(item, "risk")
        self.assertEqual(route["review_boundary"], ["中等题", "拔高题"])

    def test_all_scope_assigns_one_pair_only(self) -> None:
        item = make_item(
            "中等题",
            features={
                "step_count": "3-5步",
                "reasoning_chain": "多层因果推理",
                "state_count": "双状态",
            },
        )
        route = boundary.route_boundary_review(item, "all")
        self.assertEqual(len(route["review_boundary"]), 2)
        self.assertIn("中等题", route["review_boundary"])


class ImageRouteTests(unittest.TestCase):
    def test_plain_figure_reference_does_not_trigger_auto_image(self) -> None:
        item = make_item(
            "基础题",
            stem="如图所示，判断物体名称。",
            stem_pic_url="https://example.com/simple.png",
        )
        route = boundary.route_images(item, "auto")
        self.assertTrue(route["image_available"])
        self.assertFalse(route["image_included"])

    def test_multigraph_or_complex_circuit_triggers_auto_image(self) -> None:
        item = make_item(
            "拔高题",
            stem="图甲和图乙给出热敏电阻曲线及含开关、滑动变阻器、电压表的电路图。",
            stem_pic_url="https://example.com/a.png,https://example.com/b.png",
            features={
                "information_carrier": "多图表综合",
                "graph_table_requirement": "图像反推或外推",
            },
        )
        route = boundary.route_images(item, "auto")
        self.assertTrue(route["image_included"])
        self.assertEqual(len(route["selected_urls"]), 2)


class ReviewGateTests(unittest.TestCase):
    def test_boundary_review_cannot_recommend_third_level(self) -> None:
        item = make_item("中等题")
        route = boundary.route_boundary_review(item, "risk")
        image_route = boundary.route_images(item, "off")
        review = boundary.normalize_review(
            {
                "review_boundary": "基础题|中等题",
                "current_level": "中等题",
                "recommended_level": "拔高题",
                "boundary_status": "明确归档",
                "acceptable_levels": ["拔高题"],
                "confidence": "高",
                "effective_decision_count": 5,
                "has_structural_revision": True,
                "feature_corrections": {"step_count": "6-8步"},
                "decisive_evidence": [],
                "postprocess_rule_review": "not_applicable",
                "invalidated_rules": [],
                "image_reviewed": False,
                "image_adds_new_evidence": False,
                "new_image_evidence": [],
                "reason": "测试",
            }
        )
        self.assertIn("超出", boundary.validate_review(review, route, image_route) or "")

    def test_high_confidence_writeback_requires_two_verifiable_excerpts(self) -> None:
        item = make_item("中等题")
        route = boundary.route_boundary_review(item, "risk")
        review = boundary.normalize_review(
            {
                "review_boundary": "基础题|中等题",
                "current_level": "中等题",
                "recommended_level": "基础题",
                "boundary_status": "明确归档",
                "acceptable_levels": ["基础题"],
                "confidence": "高",
                "effective_decision_count": 2,
                "has_structural_revision": True,
                "feature_corrections": {"step_count": "1-2步"},
                "decisive_evidence": [
                    {
                        "source_field": "analysis",
                        "source_excerpt": "一个显性物理规律",
                        "finding": "只需一次应用",
                    }
                ],
                "postprocess_rule_review": "not_applicable",
                "invalidated_rules": [],
                "image_reviewed": False,
                "image_adds_new_evidence": False,
                "new_image_evidence": [],
                "reason": "证据不足两条",
            }
        )
        allowed, reason = boundary.should_apply_review(item, route, review)
        self.assertFalse(allowed)
        self.assertIn("两条", reason)

    def test_apply_review_changes_only_one_level_and_resyncs_reasoning(self) -> None:
        item = make_item("中等题")
        review = {
            "recommended_level": "基础题",
            "feature_corrections": {"step_count": "1-2步"},
            "decisive_evidence": [{"finding": "低结构"}],
        }
        boundary.apply_review(item, review)
        final = item["difficulty_rating"]
        self.assertEqual(final["difficulty_level"], "基础题")
        self.assertIn("与中等题相比", final["reasoning"]["why_not_higher"])
        self.assertEqual(final["postprocess_actions"][-1]["rule"], "adjacent_boundary_review")


if __name__ == "__main__":
    unittest.main()
