from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "generate_chemistry_review_by_level_fxz.py"
SPEC = importlib.util.spec_from_file_location(
    "generate_chemistry_review_by_level_fxz",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(question_id: str, level: str, *, stem: str = "常规化学题") -> dict:
    return {
        "question_id": question_id,
        "parent_id": question_id,
        "stem": stem,
        "options": "",
        "analysis": "",
        "sub_questions": [],
        "difficulty_rating": {
            "difficulty_level": level,
            "features": {},
            "reasoning": {},
        },
    }


class GenerateChemistryReviewByLevelFxzTests(unittest.TestCase):
    def test_poetry_easy_violation_blocks_all_output(self) -> None:
        rows = [
            row(
                "poetry-1",
                "送分题",
                stem="下列诗词中涉及化学变化的是",
            )
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            with self.assertRaisesRegex(
                MODULE.PoetryFloorViolation,
                "poetry-1",
            ):
                MODULE.generate_split_review_files(
                    rows,
                    output_dir=output_dir,
                    prefix="review",
                    expected_count=1,
                    release_label="test",
                )
            self.assertEqual(list(output_dir.iterdir()), [])

    def test_valid_rows_generate_five_single_level_pages(self) -> None:
        levels = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
        rows = [row(f"q-{index}", level) for index, level in enumerate(levels, 1)]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = MODULE.generate_split_review_files(
                rows,
                output_dir=output_dir,
                prefix="review",
                expected_count=5,
                release_label="test",
            )

            self.assertEqual(result, {1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
            for level_num, level_name in enumerate(levels, 1):
                jsonl_path = output_dir / f"review_{level_num}.jsonl"
                html_path = output_dir / f"review_{level_num}.html"
                self.assertTrue(jsonl_path.exists())
                self.assertTrue(html_path.exists())
                saved = [
                    json.loads(line)
                    for line in jsonl_path.read_text(encoding="utf-8").splitlines()
                ]
                self.assertEqual(saved[0]["difficulty_rating"]["difficulty_level"], level_name)
                rendered = html_path.read_text(encoding="utf-8")
                self.assertIn(f'id="level-{level_num}"', rendered)
                for absent in set(range(1, 6)) - {level_num}:
                    self.assertNotIn(f'id="level-{absent}"', rendered)


if __name__ == "__main__":
    unittest.main()
