# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physics_difficulty_lite_ensemble as ensemble  # noqa: E402


def item(question_id: str, level: str, actions: list[dict] | None = None) -> dict:
    return {
        "question_id": question_id,
        "difficulty_rating": {
            "difficulty_level": level,
            "features": {},
            "reasoning": {},
        },
        "postprocess_actions": actions or [],
    }


class LiteEnsembleTests(unittest.TestCase):
    def test_majority_vote(self) -> None:
        self.assertEqual(
            ensemble.choose_level(["基础题", "中等题", "中等题"]),
            ("中等题", "majority"),
        )

    def test_unanimous_vote(self) -> None:
        self.assertEqual(
            ensemble.choose_level(["拔高题", "拔高题", "拔高题"]),
            ("拔高题", "unanimous"),
        )

    def test_all_different_uses_ordinal_median(self) -> None:
        self.assertEqual(
            ensemble.choose_level(["基础题", "中等题", "拔高题"]),
            ("中等题", "median_tiebreak"),
        )

    def test_five_run_vote_tie_uses_all_prediction_median(self) -> None:
        self.assertEqual(
            ensemble.choose_level(
                ["送分题", "送分题", "中等题", "压轴题", "压轴题"]
            ),
            ("中等题", "median_tiebreak"),
        )

    def test_representative_prefers_result_without_postprocess_action(self) -> None:
        rows = [
            item("q", "中等题", [{"rule": "test"}]),
            item("q", "中等题"),
            item("q", "基础题"),
        ]
        self.assertEqual(ensemble.representative_run(rows, "中等题"), 1)


if __name__ == "__main__":
    unittest.main()
