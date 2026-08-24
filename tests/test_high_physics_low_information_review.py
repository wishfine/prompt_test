# -*- coding: utf-8 -*-
"""低信息负担组合复核页的筛选与渲染测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import generate_high_physics_low_information_review as review  # noqa: E402


def low_information_features(**overrides):
    features = {
        "information_carrier": "单一示意图",
        "graph_structure": "无图表",
        "drawing_requirement": "无",
        "experiment_requirement": "无",
        "context_type": "纯物理",
        "context_load": "纯包装",
        "step_count": "3-5步",
        "process_count": "两个过程",
        "state_count": "2个",
        "model_relation": "同一模型多状态",
        "constraint_structure": "单一约束",
        "reasoning_chain": "多层因果",
        "hidden_conditions": "无",
        "critical_state": "无临界",
        "classification_discussion": "无",
    }
    features.update(overrides)
    return features


class LowInformationReviewTests(unittest.TestCase):
    def test_selects_only_exact_low_information_level4_agreements(self) -> None:
        rows = [
            {
                "question_id": "keep",
                "stem": "题目一",
                "final_difficulty_level": "难度4档",
                "difficulty_rating_stage1": {
                    "features": low_information_features(),
                    "original_predicted_accuracy": 52.0,
                    "predicted_accuracy": 52.0,
                    "high_difficulty_feature_count": 0,
                },
            },
            {
                "question_id": "wrong-level",
                "final_difficulty_level": "难度3档",
                "difficulty_rating_stage1": {
                    "features": low_information_features(),
                },
            },
            {
                "question_id": "wrong-feature",
                "final_difficulty_level": "难度4档",
                "difficulty_rating_stage1": {
                    "features": low_information_features(context_type="生活应用"),
                },
            },
        ]
        labels = {
            "keep": "难度4档",
            "wrong-level": "难度4档",
            "wrong-feature": "难度4档",
        }

        selected = review.select_level4_agreements(rows, labels)

        self.assertEqual([item["question_id"] for item in selected], ["keep"])

    def test_rendered_page_exposes_structure_and_feedback_controls(self) -> None:
        rows = [
            {
                "question_id": "keep",
                "stem": "题目一",
                "options": "A. 甲\nB. 乙",
                "analysis": "解析一",
                "stem_image_url": "https://example.invalid/stem.png",
                "analysis_image_url": "https://example.invalid/analysis.png",
                "difficulty_rating_stage1": {
                    "features": low_information_features(),
                    "original_predicted_accuracy": 52.0,
                    "predicted_accuracy": 52.0,
                    "high_difficulty_feature_count": 0,
                    "high_difficulty_features": [],
                },
            }
        ]

        page = review.render_review_html(rows)

        self.assertIn("低信息负担组合", page)
        self.assertIn("题目ID：keep", page)
        self.assertIn("原始预测正确率", page)
        self.assertIn("保持难度4档", page)
        self.assertIn("下调为难度3档", page)
        self.assertIn("导出老师反馈", page)
        self.assertIn("单一示意图", page)
        self.assertIn("3-5步", page)
        self.assertIn('src="https://example.invalid/stem.png"', page)
        self.assertIn('src="https://example.invalid/analysis.png"', page)
        self.assertIn("题目图片（点击放大）", page)
        self.assertIn("解析图片（点击放大）", page)
        self.assertNotIn("题干图片 URL", page)
        self.assertNotIn("解析图片 URL", page)
        self.assertNotIn(">题目一<", page)
        self.assertNotIn("A. 甲", page)
        self.assertNotIn("解析一", page)


if __name__ == "__main__":
    unittest.main()
