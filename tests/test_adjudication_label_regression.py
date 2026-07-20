# -*- coding: utf-8 -*-

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.adjudication_label_regression import (
    evaluate,
    export_mismatches,
    load_adjudicated_labels,
)


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

    def test_loads_rereviewed_primary_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rereviewed_labels.csv"
            with path.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["题目ID", "原GPT裁定档", "修订后主标签", "修订后置信度"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "题目ID": "3670061256279695360",
                        "原GPT裁定档": "压轴题",
                        "修订后主标签": "拔高题",
                        "修订后置信度": "高",
                    }
                )
            labels, confidence, fields = load_adjudicated_labels(path)

        self.assertEqual(labels, {"3670061256279695360": "拔高题"})
        self.assertEqual(confidence, {"3670061256279695360": "高"})
        self.assertIn("修订后主标签", fields)

    def test_evaluate_reports_raw_final_rules_and_confidence_slices(self) -> None:
        rows = [
            {
                "question_id": "q1",
                "difficulty_level_raw": "中等题",
                "difficulty_rating": {"difficulty_level": "拔高题"},
                "difficulty_level_before_review": "中等题",
                "postprocess_actions": [
                    {"rule": "medium_to_hard", "from": "中等题", "to": "拔高题"}
                ],
                "boundary_review": {
                    "selected": True,
                    "applied": True,
                    "error": "",
                    "result": {"confidence": "高"},
                },
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
        self.assertEqual(result["before_boundary_review_evaluation"]["exact_match_rate"], 0.0)
        self.assertEqual(result["postprocess_rules"]["medium_to_hard"]["unchanged"], 1)
        self.assertEqual(result["boundary_review"]["stats"]["improved"], 1)
        self.assertEqual(result["boundary_review"]["transitions"]["中等题->拔高题"]["improved"], 1)
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

    def test_evaluate_reports_verification_agent_net_effect(self) -> None:
        rows = [
            {
                "question_id": "q1",
                "difficulty_level_raw": "中等题",
                "difficulty_level_before_verification": "中等题",
                "difficulty_rating": {"difficulty_level": "拔高题"},
                "verification_agent": {
                    "selected": True,
                    "applied": True,
                    "from": "中等题",
                    "to": "拔高题",
                    "error": "",
                    "blind_review": {"confidence": "高"},
                },
            },
            {
                "question_id": "q2",
                "difficulty_level_raw": "基础题",
                "difficulty_level_before_verification": "基础题",
                "difficulty_rating": {"difficulty_level": "基础题"},
                "verification_agent": {
                    "selected": False,
                    "applied": False,
                    "from": "基础题",
                    "to": "基础题",
                    "error": "",
                    "blind_review": {},
                },
            },
        ]
        labels = {"q1": "拔高题", "q2": "中等题"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent_results.jsonl"
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            result = evaluate(path, labels)

        self.assertEqual(result["before_verification_evaluation"]["exact_match_rate"], 0.0)
        self.assertEqual(result["verification_agent"]["stats"]["selected"], 1)
        self.assertEqual(result["verification_agent"]["stats"]["applied"], 1)
        self.assertEqual(result["verification_agent"]["stats"]["improved"], 1)
        self.assertEqual(
            result["verification_agent"]["transitions"]["中等题->拔高题"]["improved"],
            1,
        )

    def test_export_mismatches_keeps_single_run_rows_and_reference_metadata(self) -> None:
        rows = [
            {"question_id": "q1", "difficulty_rating": {"difficulty_level": "基础题"}},
            {"question_id": "q2", "difficulty_rating": {"difficulty_level": "中等题"}},
        ]
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "run1.jsonl"
            target = Path(directory) / "run1_mismatches.jsonl"
            source.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )
            count = export_mismatches(
                source,
                target,
                {"q1": "基础题", "q2": "拔高题"},
                {"q1": "高", "q2": "中"},
            )
            exported = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(exported[0]["question_id"], "q2")
        self.assertEqual(exported[0]["adjudication_reference_level"], "拔高题")
        self.assertEqual(exported[0]["adjudication_prediction_level"], "中等题")


if __name__ == "__main__":
    unittest.main()
