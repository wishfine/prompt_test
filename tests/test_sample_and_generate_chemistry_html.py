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
        "source_difficulty_untrusted": 3,
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
                                "stem_pic_url": "https://example.com/1001.png",
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
                "https://example.com/1001.png",
            )
            self.assertIn("纵向推理深度 D", rendered)
            self.assertIn("两类表征连续转换", rendered)
            self.assertIn("corrected-level-select", rendered)
            self.assertIn("来源难度（不可信）", rendered)


if __name__ == "__main__":
    unittest.main()
