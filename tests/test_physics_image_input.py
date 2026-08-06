# -*- coding: utf-8 -*-
"""初中物理首轮条件传图的离线回归测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import physics_difficulty_rating_with_cache as rating  # noqa: E402


class PhysicsImageRoutingTests(unittest.TestCase):
    def test_off_preserves_text_only_behavior(self) -> None:
        row = {
            "stem": "如图所示，分析滑轮组。<image>",
            "stem_pic_url": "https://example.com/pulley.png",
        }
        route = rating.select_rating_images(row, "off")
        self.assertFalse(route["image_included"])
        self.assertEqual(route["selected_urls"], [])

        content = rating.build_model_user_content("题目", None, [])
        self.assertIsInstance(content, str)

    def test_auto_includes_graph_relation(self) -> None:
        row = {
            "stem": "根据 s-t 图像比较甲乙速度。<image>",
            "stem_pic_url": "https://example.com/st.png",
        }
        route = rating.select_rating_images(row, "auto")
        self.assertTrue(route["image_included"])
        self.assertIn("https://example.com/st.png", route["selected_urls"])
        self.assertTrue(any("曲线" in reason for reason in route["reasons"]))

    def test_auto_excludes_decorative_picture(self) -> None:
        row = {
            "stem": "无人机如图所示。已知电动机功率，直接求电流。<image>",
            "stem_pic_url": "https://example.com/drone.png",
        }
        route = rating.select_rating_images(row, "auto")
        self.assertFalse(route["image_included"])
        self.assertEqual(route["reasons"], ["结构化文字足以支持定档"])

    def test_auto_includes_multiple_visual_options(self) -> None:
        row = {
            "stem": "选择正确的实验操作图。",
            "options": "A.<image> B.<image> C.<image> D.<image>",
            "stem_pic_url": "https://example.com/options.png",
        }
        route = rating.select_rating_images(row, "auto")
        self.assertTrue(route["image_included"])
        self.assertTrue(any("多个选项" in reason for reason in route["reasons"]))

    def test_raw_image_is_preferred_over_externalized_page(self) -> None:
        row = {
            "stem": "根据波形图判断音调。<image>",
            "stem_pic_url": (
                "https://example.com/raw.png,"
                "https://example.com/image/externalized/page.png"
            ),
        }
        route = rating.select_rating_images(row, "auto")
        self.assertEqual(route["selected_urls"], ["https://example.com/raw.png"])

    def test_image_content_uses_responses_mixed_content_schema(self) -> None:
        original_prefix = rating.DIFFICULTY_RATING_PROMPT_PREFIX
        original_suffix = rating.DIFFICULTY_RATING_PROMPT_SUFFIX
        rating.DIFFICULTY_RATING_PROMPT_PREFIX = "PREFIX"
        rating.DIFFICULTY_RATING_PROMPT_SUFFIX = "SUFFIX"
        try:
            content = rating.build_model_user_content(
                "QUESTION",
                None,
                ["https://example.com/question.png"],
            )
        finally:
            rating.DIFFICULTY_RATING_PROMPT_PREFIX = original_prefix
            rating.DIFFICULTY_RATING_PROMPT_SUFFIX = original_suffix

        self.assertIsInstance(content, list)
        self.assertEqual(content[0]["type"], "input_text")
        self.assertIn("PREFIX", content[0]["text"])
        self.assertEqual(
            content[1],
            {"type": "input_image", "image_url": "https://example.com/question.png"},
        )


if __name__ == "__main__":
    unittest.main()
