from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "tools" / "evaluate_chemistry_difficulty.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "evaluate_chemistry_difficulty",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ChemistryEvaluationTests(unittest.TestCase):
    def test_report_includes_distribution_recall_and_collapse_warning(
        self,
    ) -> None:
        evaluation = load_module()
        labels = {
            "q1": {"standard_level": 4, "standard_level_name": "拔高题"},
            "q2": {"standard_level": 5, "standard_level_name": "压轴题"},
            "q3": {"standard_level": 5, "standard_level_name": "压轴题"},
            "q4": {"standard_level": 5, "standard_level_name": "压轴题"},
        }
        predictions = {
            "q1": {"predicted_level": 4, "predicted_level_name": "拔高题"},
            "q2": {"predicted_level": 4, "predicted_level_name": "拔高题"},
            "q3": {"predicted_level": 4, "predicted_level_name": "拔高题"},
            "q4": {"predicted_level": 5, "predicted_level_name": "压轴题"},
        }

        report, _ = evaluation.evaluate_predictions(
            labels,
            predictions,
            error_ids=set(),
            error_messages={},
        )

        self.assertEqual(report["label_distribution"]["压轴题"], 3)
        self.assertEqual(report["prediction_distribution"]["压轴题"], 1)
        self.assertEqual(report["distribution_l1_count"], 4)
        self.assertEqual(report["distribution_total_variation"], 0.5)
        self.assertEqual(
            report["per_level_metrics"]["压轴题"]["recall"],
            0.333333,
        )
        self.assertEqual(report["top_two_level_recall"], 1.0)
        self.assertEqual(report["severe_deviation_count"], 0)
        self.assertTrue(
            any(
                item["level"] == "压轴题"
                and item["type"] == "prediction_count_below_half"
                for item in report["distribution_warnings"]
            )
        )

    def test_candidate_level_source_is_supported(self) -> None:
        evaluation = load_module()
        item = {
            "question_id": "q1",
            "difficulty_rating": {
                "difficulty_level": "拔高题",
                "postprocess_original_level": "拔高题",
                "postprocess_candidate_level": "压轴题",
                "final_boundary_guard_candidate_level": "压轴题",
                "teacher_distribution_guard_candidate_level": "中等题",
            },
        }

        level_name, level_number = evaluation.extract_prediction(
            item,
            "postprocess-candidate",
        )

        self.assertEqual(level_name, "压轴题")
        self.assertEqual(level_number, 5)

        level_name, level_number = evaluation.extract_prediction(
            item,
            "teacher-distribution-guard-candidate",
        )

        self.assertEqual(level_name, "中等题")
        self.assertEqual(level_number, 3)

        level_name, level_number = evaluation.extract_prediction(
            item,
            "final-boundary-guard-candidate",
        )

        self.assertEqual(level_name, "压轴题")
        self.assertEqual(level_number, 5)

    def test_nested_postprocess_original_level_is_supported(self) -> None:
        evaluation = load_module()
        item = {
            "question_id": "q1",
            "difficulty_rating": {
                "difficulty_level": "压轴题",
                "postprocess": {
                    "original_level": "拔高题",
                    "final_level": "压轴题",
                },
            },
        }

        level_name, level_number = evaluation.extract_prediction(
            item,
            "pre-postprocess",
        )

        self.assertEqual(level_name, "拔高题")
        self.assertEqual(level_number, 4)

    def test_high_chemistry_output_and_jsonl_labels_are_supported(self) -> None:
        evaluation = load_module()
        item = {
            "question_id": "q1",
            "difficulty_level_step1": "难度4档",
            "final_difficulty_level": "难度3档",
        }
        level_name, level_number = evaluation.extract_prediction(item, "final")
        self.assertEqual(level_name, "难度3档")
        self.assertEqual(level_number, 3)
        level_name, level_number = evaluation.extract_prediction(item, "pre-postprocess")
        self.assertEqual(level_name, "难度4档")
        self.assertEqual(level_number, 4)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "labels.jsonl"
            path.write_text(
                json.dumps({"question_id": "q1", "standard_level": 3, "reason": "常规综合"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            labels = evaluation.load_labels(path)
        self.assertEqual(labels["q1"]["standard_level"], 3)

    def test_mixed_run_signatures_are_rejected(self) -> None:
        evaluation = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mixed.jsonl"
            rows = [
                {
                    "question_id": "q1",
                    "run_signature": "signature-a",
                    "difficulty_rating": {
                        "difficulty_level": "基础题",
                        "general_level_writeback_enabled": False,
                        "final_boundary_guard_enabled": True,
                        "final_boundary_guard_writeback_enabled": False,
                    },
                },
                {
                    "question_id": "q2",
                    "run_signature": "signature-b",
                    "difficulty_rating": {
                        "difficulty_level": "中等题",
                        "general_level_writeback_enabled": False,
                        "final_boundary_guard_enabled": True,
                        "final_boundary_guard_writeback_enabled": False,
                    },
                },
            ]
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "混合运行签名"):
                evaluation.validate_prediction_run_consistency(path)

    def test_partially_signed_predictions_are_rejected(self) -> None:
        evaluation = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "partial.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "question_id": "q1",
                                "run_signature": "signature-a",
                            }
                        ),
                        json.dumps({"question_id": "q2"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "部分记录缺少"):
                evaluation.validate_prediction_run_consistency(path)


if __name__ == "__main__":
    unittest.main()
