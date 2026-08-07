import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "sample_and_generate_chemistry_html.py"
SPEC = importlib.util.spec_from_file_location(
    "sample_and_generate_chemistry_html",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rated_item(question_id: str, level: str) -> dict:
    return {
        "question_id": question_id,
        "difficulty_rating": {
            "difficulty_level": level,
            "features": {
                "reasoning_depth": "2-3层",
                "reasoning_direction": "正向推导",
                "knowledge_relation": "同模块深度关联",
                "representation_conversion": "两类表征连续转换",
                "reaction_relation": "单个反应或无反应",
                "constraint_complexity": "单一约束",
                "evidence_relation": "单一证据直接对应",
                "experiment_requirement": "无",
                "graph_table_requirement": "无",
                "calculation_model": "无",
                "unfamiliar_information_transfer": "无",
                "subquestion_dependency": "无多问",
            },
            "reasoning": {
                "core_basis": "测试依据",
                "hard_point": "测试卡点",
                "why_not_lower": "不能降低",
                "why_not_higher": "不能升高",
            },
        },
    }


class ChemistryVisualizationTests(unittest.TestCase):
    def test_sample_plan_scales_to_exact_requested_total(self) -> None:
        plan = MODULE.build_sample_plan(37)
        self.assertEqual(sum(plan.values()), 37)
        self.assertEqual(set(plan), set(MODULE.SAMPLE_PLAN))

    def test_externalized_image_is_primary_and_original_is_collapsed(
        self,
    ) -> None:
        section = MODULE.render_image_section(
            "https://example.com/original.png,"
            "https://example.com/image/externalized/full.png",
            title="题干图示",
            kind="stem",
        )
        primary_index = section.index(
            "https://example.com/image/externalized/full.png"
        )
        supporting_index = section.index(
            "https://example.com/original.png"
        )
        self.assertLess(primary_index, supporting_index)
        self.assertIn(
            '<details class="supporting-images">',
            section,
        )
        self.assertIn(
            "查看原始题图 / 备用图片（1张）",
            section,
        )
        self.assertIn(
            'data-image-source="externalized"',
            section,
        )

    def test_visualization_matches_physics_font_and_image_controls(
        self,
    ) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("max-width: 1440px", template)
        self.assertIn("font-size: 21px", template)
        self.assertIn("font-size: 18px", template)
        self.assertIn("font-size: 17px", template)
        self.assertIn("image-layout-long-document", template)
        self.assertIn("function applyAdaptiveImageSizing", template)
        self.assertIn("image-lightbox", template)

    def test_human_review_uses_default_acceptance_exception_flow(
        self,
    ) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("const correct = total - wrong;", template)
        self.assertIn(
            "review_source: manuallyRejected ? "
            "'manual_exception' : 'default_model_accepted'",
            template,
        )
        self.assertIn(
            "annotations[qid].verdict = 'wrong';",
            template,
        )
        self.assertIn(
            "输入修改意见后自动标记为异常",
            template,
        )

    def test_missing_required_chemistry_image_is_explicit(self) -> None:
        self.assertTrue(
            MODULE.contains_image_reference("根据微观示意图回答")
        )
        section = MODULE.render_image_section(
            "",
            title="题干图示",
            kind="stem",
            content_requires_image=True,
        )
        self.assertIn("题干图示资源缺失", section)
        self.assertIn("media-missing", section)

    def test_all_results_renders_core12_and_exports_aligned_images(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            results = root / "rated.jsonl"
            source = root / "source.jsonl"
            output_jsonl = root / "nested" / "visualized.jsonl"
            output_html = root / "nested" / "review.html"

            rows = [
                rated_item("1001", "基础题"),
                rated_item("1002", "中等题"),
            ]
            results.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )
            source.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question_id": 1001,
                                "stem_pic_url": (
                                    "https://example.com/1001.png,"
                                    "https://example.com/image/"
                                    "externalized/1001-full.png"
                                ),
                                "analysis_pic_url": "",
                            }
                        ),
                        json.dumps(
                            {
                                "question_id": "1002",
                                "stem_pic_url": "",
                                "analysis_pic_url": "https://example.com/a.png",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            argv = [
                str(SCRIPT),
                "-i",
                str(results),
                "-v2",
                str(source),
                "-oj",
                str(output_jsonl),
                "-oh",
                str(output_html),
                "--all-results",
            ]
            with patch.object(sys, "argv", argv):
                MODULE.main()

            exported = [
                json.loads(line)
                for line in output_jsonl.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            rendered = output_html.read_text(encoding="utf-8")

            self.assertEqual(len(exported), 2)
            self.assertEqual(
                exported[0]["stem_pic_url"],
                (
                    "https://example.com/1001.png,"
                    "https://example.com/image/"
                    "externalized/1001-full.png"
                ),
            )
            primary_index = rendered.index(
                "https://example.com/image/externalized/1001-full.png"
            )
            supporting_index = rendered.index(
                "https://example.com/1001.png"
            )
            self.assertLess(primary_index, supporting_index)
            self.assertIn(
                "查看原始题图 / 备用图片（1张）",
                rendered,
            )
            self.assertIn("纵向推理深度 D", rendered)
            self.assertIn("两类表征连续转换", rendered)
            self.assertIn("corrected-level-select", rendered)
            self.assertIn(
                "人工评议验收（默认模型判定合理）",
                rendered,
            )
            self.assertIn("✗ 模型判定不准", rendered)
            self.assertIn("✓ 恢复默认合理", rendered)
            self.assertIn("仅标错，暂不指定档位", rendered)
            self.assertNotIn("✓ 模型判定合理</button>", rendered)
            self.assertNotIn("— 清除状态", rendered)
            self.assertNotIn("来源难度（不可信）", rendered)


if __name__ == "__main__":
    unittest.main()
