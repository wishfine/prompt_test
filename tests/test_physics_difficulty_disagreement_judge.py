# -*- coding: utf-8 -*-
"""匿名分歧裁判的离线测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physics_difficulty_disagreement_judge as judge  # noqa: E402
import physics_difficulty_rating_with_cache as rating  # noqa: E402


def item(question_id: str, level: str, structure: str = "概念判断") -> dict:
    features = dict(rating.FEATURE_DEFAULTS)
    features["problem_structure"] = structure
    return {
        "question_id": question_id,
        "stem": f"{question_id}题干",
        "analysis": f"{question_id}官方解析",
        "difficulty_rating": {
            "difficulty_level": level,
            "features": features,
            "reasoning": {},
        },
    }


class CandidateTests(unittest.TestCase):
    def test_candidate_pair_accepts_only_two_adjacent_levels(self) -> None:
        self.assertEqual(
            judge.candidate_pair(["基础题", "中等题", "中等题"]),
            ("基础题", "中等题"),
        )
        self.assertIsNone(judge.candidate_pair(["基础题", "拔高题"]))
        self.assertIsNone(judge.candidate_pair(["送分题", "基础题", "中等题"]))

    def test_majority_level_is_deterministic(self) -> None:
        self.assertEqual(
            judge.majority_level(["中等题", "拔高题", "中等题"]),
            "中等题",
        )
        self.assertEqual(judge.majority_level(["基础题", "中等题"]), "基础题")

    def test_review_label_parser(self) -> None:
        self.assertEqual(
            judge.parse_review_human_label(
                {
                    "verdict": "correct",
                    "model_difficulty_level": "中等题",
                    "human_notes": "",
                }
            ),
            "中等题",
        )
        self.assertEqual(
            judge.parse_review_human_label(
                {
                    "verdict": "wrong",
                    "model_difficulty_level": "中等题",
                    "human_notes": "较难，需要建模。",
                }
            ),
            "拔高题",
        )


class FewshotTests(unittest.TestCase):
    def test_fewshots_are_balanced_and_exclude_target(self) -> None:
        references = {
            "target": item("target", "基础题"),
            "b1": item("b1", "基础题"),
            "b2": item("b2", "基础题"),
            "m1": item("m1", "中等题"),
            "m2": item("m2", "中等题"),
        }
        labels = {
            "target": "基础题",
            "b1": "基础题",
            "b2": "基础题",
            "m1": "中等题",
            "m2": "中等题",
        }
        values = judge.select_balanced_fewshots(
            "target",
            references["target"],
            ("基础题", "中等题"),
            references,
            labels,
            {},
            per_level=2,
            seed=1,
        )
        self.assertEqual(len(values), 4)
        self.assertEqual(
            sorted(value["teacher_level"] for value in values),
            ["中等题", "中等题", "基础题", "基础题"],
        )
        self.assertNotIn("target题干", str(values))

    def test_anonymous_content_does_not_leak_votes_or_current_level(self) -> None:
        target = item("target", "中等题")
        fewshots = [
            {
                "teacher_level": "基础题",
                "stem_summary": "样例",
                "official_analysis_summary": "解析",
                "teacher_boundary_note": "说明",
                "similarity_score": 1,
            },
            {
                "teacher_level": "中等题",
                "stem_summary": "样例",
                "official_analysis_summary": "解析",
                "teacher_boundary_note": "说明",
                "similarity_score": 1,
            },
        ]
        content = judge.build_judge_content(
            target,
            ("基础题", "中等题"),
            fewshots,
            "balanced",
        )
        self.assertNotIn("run_predictions", content)
        self.assertNotIn("majority_level", content)
        self.assertNotIn("首轮等级", content)
        self.assertNotIn("多数票", content)


class JudgmentTests(unittest.TestCase):
    def test_judgment_must_choose_one_boundary_level(self) -> None:
        pair = ("中等题", "拔高题")
        value = judge.normalize_judgment(
            {
                "review_boundary": "中等题|拔高题",
                "lower_level": "中等题",
                "upper_level": "拔高题",
                "judge_role": "balanced",
                "chosen_level": "压轴题",
                "upper_threshold_met": True,
                "confidence": "高",
                "effective_decision_count": 8,
                "decisive_structures": [],
                "missing_upper_requirements": [],
                "evidence": [],
                "reason": "测试",
            }
        )
        self.assertIn(
            "chosen_level",
            judge.validate_judgment(value, pair, "balanced") or "",
        )

    def test_threshold_boolean_must_match_choice(self) -> None:
        pair = ("基础题", "中等题")
        value = judge.normalize_judgment(
            {
                "review_boundary": "基础题|中等题",
                "lower_level": "基础题",
                "upper_level": "中等题",
                "judge_role": "balanced",
                "chosen_level": "基础题",
                "upper_threshold_met": True,
                "confidence": "高",
                "effective_decision_count": 2,
                "decisive_structures": [],
                "missing_upper_requirements": ["没有连续过程"],
                "evidence": [],
                "reason": "测试",
            }
        )
        self.assertIn(
            "不一致",
            judge.validate_judgment(value, pair, "balanced") or "",
        )


class SummaryTests(unittest.TestCase):
    def test_summary_reports_target_and_per_level_metrics(self) -> None:
        base_items = {
            "q1": item("q1", "基础题"),
            "q2": item("q2", "中等题"),
        }
        predictions = {
            "q1": ["基础题", "基础题", "基础题"],
            "q2": ["基础题", "中等题", "中等题"],
        }
        cases = judge.build_cases(["q1", "q2"], base_items, predictions)
        report = judge.summarize(
            cases,
            {},
            {"q1": "基础题", "q2": "中等题"},
        )
        self.assertEqual(report["final_correct"], 2)
        self.assertEqual(report["teacher_distribution"], {"基础题": 1, "中等题": 1})
        self.assertEqual(report["prediction_distribution"], {"基础题": 1, "中等题": 1})
        self.assertEqual(report["per_level_metrics"]["基础题"]["f1"], 1.0)
        self.assertEqual(report["target_90_percent"]["required_correct"], 2)


def valid_judgment(
    pair: tuple[str, str],
    role: str,
    chosen: str,
) -> dict:
    return {
        "review_boundary": f"{pair[0]}|{pair[1]}",
        "lower_level": pair[0],
        "upper_level": pair[1],
        "judge_role": role,
        "chosen_level": chosen,
        "upper_threshold_met": chosen == pair[1],
        "confidence": "高",
        "effective_decision_count": 3,
        "decisive_structures": ["连续分析"],
        "missing_upper_requirements": [] if chosen == pair[1] else ["未超过下档上限"],
        "evidence": [],
        "reason": "测试判断",
    }


class DualJudgeTests(unittest.IsolatedAsyncioTestCase):
    async def test_dual_agreement_skips_arbiter(self) -> None:
        pair = ("基础题", "中等题")

        async def fake_call(*_args, **kwargs):
            role = kwargs["role"]
            return valid_judgment(pair, role, "中等题"), {}, 0.01, ""

        with patch.object(judge, "call_with_semaphore", side_effect=fake_call):
            result = await judge.judge_case(
                item("q", "基础题"),
                pair,
                [],
                "dual",
                "prompt",
                object(),
                object(),
                "glm-5.2",
                "glm-5.2",
                "glm-5.2",
                "",
                1,
                30,
            )
        self.assertEqual(result["chosen_level"], "中等题")
        self.assertEqual(result["decision_source"], "dual_judges_agree")
        self.assertEqual(len(result["calls"]), 2)

    async def test_dual_disagreement_calls_arbiter(self) -> None:
        pair = ("中等题", "拔高题")

        async def fake_call(*_args, **kwargs):
            role = kwargs["role"]
            chosen = {
                "upper_threshold": "拔高题",
                "lower_ceiling": "中等题",
                "arbiter": "中等题",
            }[role]
            return valid_judgment(pair, role, chosen), {}, 0.01, ""

        with patch.object(judge, "call_with_semaphore", side_effect=fake_call):
            result = await judge.judge_case(
                item("q", "中等题"),
                pair,
                [],
                "dual",
                "prompt",
                object(),
                object(),
                "glm-5.2",
                "glm-5.2",
                "glm-5.2",
                "",
                1,
                30,
            )
        self.assertEqual(result["chosen_level"], "中等题")
        self.assertEqual(result["decision_source"], "arbiter")
        self.assertEqual(len(result["calls"]), 3)


if __name__ == "__main__":
    unittest.main()
