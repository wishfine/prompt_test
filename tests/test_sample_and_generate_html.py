import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "sample_and_generate_html.py"
SPEC = importlib.util.spec_from_file_location("sample_and_generate_html", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SampleAndGenerateHtmlTests(unittest.TestCase):
    def test_sample_plan_scales_to_requested_total(self) -> None:
        self.assertEqual(
            MODULE.build_sample_plan(1500),
            {
                "送分题": 300,
                "基础题": 360,
                "中等题": 360,
                "拔高题": 300,
                "压轴题": 180,
            },
        )
        self.assertEqual(sum(MODULE.build_sample_plan(1501).values()), 1501)

    def test_review_count_uses_dynamic_placeholder(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("__REVIEW_COUNT__题", template)
        self.assertIn("annotations___REVIEW_SCOPE__", template)

    def test_unmarked_questions_default_to_model_accepted(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("未标注即视为“模型判定合理”", template)
        self.assertIn("default_model_accepted", template)
        self.assertIn("human_reviewed: manuallyRejected", template)

    def test_dashboard_counts_only_manual_exceptions(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("已标异常", template)
        self.assertIn("const correct = total - wrong", template)
        self.assertIn("人工标记不准", template)

    def test_typing_a_reason_marks_question_wrong(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("annotations[qid].verdict = 'wrong'", template)

    def test_corrected_level_is_structured_and_marks_exception(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("function saveCorrectedLevel", template)
        self.assertIn("human_difficulty_level", template)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("建议正确档位", source)
        self.assertIn("输入内容后会自动标记", source)

    def test_split_image_urls_deduplicates_and_preserves_order(self) -> None:
        self.assertEqual(
            MODULE.split_image_urls("a.png, b.png, a.png, ,c.png"),
            ["a.png", "b.png", "c.png"],
        )
        self.assertEqual(
            MODULE.split_image_urls(["a.png", " b.png ", "a.png"]),
            ["a.png", "b.png"],
        )

    def test_contains_image_reference_checks_text_and_nested_values(self) -> None:
        self.assertTrue(MODULE.contains_image_reference("如下图所示"))
        self.assertTrue(
            MODULE.contains_image_reference(
                [{"analysis": "根据图中关系可知"}]
            )
        )
        self.assertFalse(MODULE.contains_image_reference("纯文字直接判断"))

    def test_externalized_screenshot_is_primary_and_originals_are_supporting(
        self,
    ) -> None:
        primary, supporting = MODULE.partition_visualization_images(
            "https://img.example/original.png,"
            "https://img.example/image/externalized/full.png,"
            "https://img.example/diagram.png"
        )
        self.assertEqual(
            primary,
            ["https://img.example/image/externalized/full.png"],
        )
        self.assertEqual(
            supporting,
            [
                "https://img.example/original.png",
                "https://img.example/diagram.png",
            ],
        )

    def test_without_externalized_screenshot_all_images_remain_primary(
        self,
    ) -> None:
        primary, supporting = MODULE.partition_visualization_images(
            "https://img.example/a.png,https://img.example/b.png"
        )
        self.assertEqual(
            primary,
            ["https://img.example/a.png", "https://img.example/b.png"],
        )
        self.assertEqual(supporting, [])

    def test_two_stem_pngs_show_second_and_fold_first(self) -> None:
        section = MODULE.render_image_section(
            "https://img.example/first.png,"
            "https://img.example/second.png",
            title="题干图示",
            kind="stem",
        )
        primary_index = section.index("https://img.example/second.png")
        supporting_index = section.index("https://img.example/first.png")
        self.assertLess(primary_index, supporting_index)
        self.assertIn("完整题干图示", section)
        self.assertIn("查看原始题图 / 备用图片（1张）", section)

    def test_two_analysis_pngs_remain_visible_without_order_rule(self) -> None:
        primary, supporting = MODULE.partition_visualization_images(
            "https://img.example/first.png,"
            "https://img.example/second.png",
            prefer_second_png_when_two=False,
        )
        self.assertEqual(
            primary,
            [
                "https://img.example/first.png",
                "https://img.example/second.png",
            ],
        )
        self.assertEqual(supporting, [])

    def test_image_section_preserves_supporting_images_in_collapsed_details(
        self,
    ) -> None:
        section = MODULE.render_image_section(
            "https://img.example/source.png,"
            "https://img.example/image/externalized/full.png",
            title="题干图示",
            kind="stem",
        )
        self.assertIn('data-image-role="stem-primary"', section)
        self.assertIn('data-image-role="stem-supporting"', section)
        self.assertIn('<details class="supporting-images">', section)
        self.assertIn("查看原始题图 / 备用图片（1张）", section)
        self.assertIn("openImagePreview(this)", section)
        self.assertIn('data-image-source="externalized"', section)

    def test_missing_required_image_is_not_reported_as_no_image(self) -> None:
        section = MODULE.render_image_section(
            "",
            title="题干图示",
            kind="stem",
            content_requires_image=True,
        )
        self.assertIn("题干图示资源缺失", section)
        self.assertIn("media-missing", section)
        self.assertNotIn("该题无题干图示", section)

    def test_visualization_css_enlarges_images_and_rating_text(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("max-width: 1440px", template)
        self.assertIn("max-height: none", template)
        self.assertIn("max-width: min(100%, 980px)", template)
        self.assertIn("image-layout-long-document", template)
        self.assertIn("function applyAdaptiveImageSizing", template)
        self.assertIn("font-size: 21px", template)
        self.assertIn("font-size: 18px", template)
        self.assertIn("font-size: 17px", template)
        self.assertIn("image-lightbox", template)

    def test_generated_html_uses_smart_image_sections(self) -> None:
        samples = {
            3: [
                {
                    "question_id": "q-1",
                    "stem_pic_url": (
                        "https://img.example/source.png,"
                        "https://img.example/image/externalized/stem.png"
                    ),
                    "analysis_pic_url": (
                        "https://img.example/image/externalized/analysis.png"
                    ),
                    "difficulty_rating": {
                        "difficulty_level": "中等题",
                        "features": {"step_count": "3-5步"},
                        "reasoning": {"core_basis": "常规连续分析。"},
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "preview.html"
            MODULE.generate_html_file(samples, str(output))
            rendered = output.read_text(encoding="utf-8")

        primary_index = rendered.index(
            "https://img.example/image/externalized/stem.png"
        )
        supporting_index = rendered.index("https://img.example/source.png")
        self.assertLess(primary_index, supporting_index)
        self.assertIn("完整截图优先，原始题图可展开", rendered)
        self.assertIn('<details class="supporting-images">', rendered)

    def test_build_level_html_paths_creates_one_stable_path_per_level(self) -> None:
        paths = MODULE.build_level_html_paths(
            Path("outputs/visualizations/recent3years_review1000.html")
        )

        self.assertEqual(
            paths,
            {
                1: Path(
                    "outputs/visualizations/"
                    "recent3years_review1000_difficulty1_easy.html"
                ),
                2: Path(
                    "outputs/visualizations/"
                    "recent3years_review1000_difficulty2_basic.html"
                ),
                3: Path(
                    "outputs/visualizations/"
                    "recent3years_review1000_difficulty3_medium.html"
                ),
                4: Path(
                    "outputs/visualizations/"
                    "recent3years_review1000_difficulty4_hard.html"
                ),
                5: Path(
                    "outputs/visualizations/"
                    "recent3years_review1000_difficulty5_final.html"
                ),
            },
        )

    def test_generate_split_level_html_files_writes_five_independent_pages(
        self,
    ) -> None:
        samples = {
            level: [
                {
                    "question_id": f"q-{level}",
                    "difficulty_rating": {
                        "difficulty_level": level_name,
                        "features": {},
                        "reasoning": {},
                    },
                }
            ]
            for level, level_name in enumerate(
                ["送分题", "基础题", "中等题", "拔高题", "压轴题"],
                1,
            )
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            base = Path(tmp_dir) / "review1000.html"
            outputs = MODULE.generate_split_level_html_files(samples, base)

            self.assertEqual(len(outputs), 5)
            for level, output in outputs.items():
                rendered = output.read_text(encoding="utf-8")
                self.assertIn(f"q-{level}", rendered)
                expected_scope = MODULE.LEVEL_FILE_SLUGS[level]
                self.assertIn(
                    f"physics_difficulty_annotations_{expected_scope}",
                    rendered,
                )
                self.assertIn(
                    f"physics_difficulty_human_annotations_{expected_scope}.jsonl",
                    rendered,
                )
                for other_level in range(1, 6):
                    if other_level != level:
                        self.assertNotIn(f"q-{other_level}", rendered)


if __name__ == "__main__":
    unittest.main()
