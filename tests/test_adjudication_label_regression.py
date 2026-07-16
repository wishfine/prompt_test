# -*- coding: utf-8 -*-

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.adjudication_label_regression import evaluate, load_adjudicated_labels


class AdjudicationLabelRegressionTests(unittest.TestCase):
    def test_loads_string_ids_and_five_level_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["题目ID", "最终裁定档", "裁定置信度"])
                writer.writeheader()
                writer.writerow(
                    {
                        "题目ID": "3670061256279695360",
                        "最终裁定档": "压轴题",
                        "裁定置信度": "高",
                    }
                )
            labels, confidence, fields = load_adjudicated_labels(path)

        self.assertEqual(labels, {"3670061256279695360": "压轴题"})
        self.assertEqual(confidence, {"3670061256279695360": "高"})
        self.assertIn("最终裁定档", fields)

    def test_evaluate_reports_raw_final_rules_and_confidence_slices(self) -> None:
        rows = [
            {
                "question_id": "q1",
                "difficulty_level_raw": "中等题",
                "difficulty_rating": {"difficulty_level": "拔高题"},
                "postprocess_actions": [
                    {"rule": "medium_to_hard", "from": "中等题", "to": "拔高题"}
                ],
            },
            {
                "question_id": "q2",
                "difficulty_level_raw": "基础题",
                "difficulty_rating": {"difficulty_level": "基础题"},
                "postprocess_actions": [],
            },
        ]
        labels = {"q1": "拔高题", "q2": "中等题"}
        confidence = {"q1": "高", "q2": "中"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = evaluate(path, labels, confidence)

        self.assertEqual(result["evaluated"], 2)
        self.assertEqual(result["exact_match_rate"], 0.5)
        self.assertEqual(result["raw_evaluation"]["exact_match_rate"], 0.0)
        self.assertEqual(result["postprocess_rules"]["medium_to_hard"]["improved"], 1)
        self.assertEqual(result["confidence_slices"]["高"]["exact_match_rate"], 1.0)
        self.assertEqual(result["confidence_slices"]["中"]["exact_match_rate"], 0.0)

    def test_script_runs_directly_from_repository_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_path = root / "labels.csv"
            questions_path = root / "questions.jsonl"
            results_path = root / "results.jsonl"
            with labels_path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["题目ID", "最终裁定档", "裁定置信度"])
                writer.writeheader()
                writer.writerow({"题目ID": "q1", "最终裁定档": "基础题", "裁定置信度": "高"})
            questions_path.write_text('{"question_id":"q1","stem":"test"}\n', encoding="utf-8")
            results_path.write_text(
                '{"question_id":"q1","difficulty_rating":{"difficulty_level":"基础题"}}\n',
                encoding="utf-8",
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    "tests/adjudication_label_regression.py",
                    "--csv",
                    str(labels_path),
                    "--jsonl",
                    str(questions_path),
                    "--evaluate",
                    str(results_path),
                ],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"exact_match_rate": 1.0', completed.stdout)


if __name__ == "__main__":
    unittest.main()
