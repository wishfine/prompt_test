# -*- coding: utf-8 -*-
"""高中物理老师反馈题集导入工具测试。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import import_high_physics_teacher_feedback as importer  # noqa: E402


class TeacherFeedbackImportTests(unittest.TestCase):
    def test_import_preserves_jsonl_order_and_reports_count(self) -> None:
        rows = [
            {"question_id": "900000000000000001", "stem": "题目甲"},
            {"question_id": "900000000000000002", "stem": "题目乙"},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "source.jsonl"
            target = directory / "target.jsonl"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            report = importer.import_teacher_feedback_set(source, target)

            imported = [
                json.loads(line)
                for line in target.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(imported, rows)
            self.assertEqual(report["imported_count"], 2)
            self.assertEqual(report["question_id_count"], 2)

    def test_import_rejects_duplicate_question_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            source = directory / "source.jsonl"
            target = directory / "target.jsonl"
            source.write_text(
                "\n".join(
                    (
                        '{"question_id":"900000000000000001","stem":"甲"}',
                        '{"question_id":"900000000000000001","stem":"乙"}',
                    )
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "重复 question_id"):
                importer.import_teacher_feedback_set(source, target)


if __name__ == "__main__":
    unittest.main()
