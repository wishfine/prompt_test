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
    def test_run_consistency_reports_schema_retry_and_normalization(self) -> None:
        evaluation = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "predictions.jsonl"
            rows = [
                {
                    "question_id": "q1",
                    "schema_retry_count": 1,
                    "schema_validation_errors": ["bad enum"],
                    "feature_normalization_actions": [
                        {"field": "task_groups.task_type"}
                    ],
                },
                {
                    "question_id": "q2",
                    "schema_retry_count": 0,
                    "schema_validation_errors": [],
                    "feature_normalization_actions": [],
                },
            ]
            path.write_text(
                "".join(
                    json.dumps(row, ensure_ascii=False) + "\n"
                    for row in rows
                ),
                encoding="utf-8",
            )

            result = evaluation.validate_prediction_run_consistency(path)

        diagnostics = result["schema_diagnostics"]
        self.assertEqual(diagnostics["retry_rows"], 1)
        self.assertEqual(diagnostics["retry_total"], 1)
        self.assertEqual(diagnostics["normalization_rows"], 1)
        self.assertEqual(diagnostics["normalization_action_total"], 1)
        self.assertEqual(
            diagnostics["normalization_fields"],
            {"task_groups.task_type": 1},
        )

    def test_human_review_jsonl_is_supported_as_label_source(self) -> None:
        evaluation = load_module()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "human.jsonl"
            rows = [
                {
                    "question_id": "q1",
                    "human_difficulty_level": "基础题",
                    "human_notes": "需要一次规则应用",
                },
                {
                    "question_id": "q2",
                    "human_difficulty_level": "",
                    "human_notes": "未给最终档",
                },
            ]
            path.write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                encoding="utf-8",
            )

            labels = evaluation.load_labels(path)

        self.assertEqual(set(labels), {"q1"})
        self.assertEqual(labels["q1"]["standard_level"], 2)
        self.assertEqual(labels["q1"]["reason"], "需要一次规则应用")

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
                "boundary_v4_guard_candidate_level": "基础题",
                "combined_guard_candidate_level": "压轴题",
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

        level_name, level_number = evaluation.extract_prediction(
            item,
            "boundary-v4-guard-candidate",
        )

        self.assertEqual(level_name, "基础题")
        self.assertEqual(level_number, 2)

        level_name, level_number = evaluation.extract_prediction(
            item,
            "combined-guard-candidate",
        )

        self.assertEqual(level_name, "压轴题")
        self.assertEqual(level_number, 5)

    def test_rule_attribution_reports_helped_hurt_and_net(self) -> None:
        evaluation = load_module()
        labels = {
            "q1": {"standard_level": 2, "standard_level_name": "基础题"},
            "q2": {"standard_level": 1, "standard_level_name": "送分题"},
            "q3": {"standard_level": 3, "standard_level_name": "中等题"},
        }
        predictions = {
            "q1": {
                "predicted_level": 2,
                "predicted_level_name": "基础题",
                "postprocess_original_level": "送分题",
                "selected_action": {"rule": "boundary_rule"},
            },
            "q2": {
                "predicted_level": 2,
                "predicted_level_name": "基础题",
                "postprocess_original_level": "送分题",
                "selected_action": {"rule": "boundary_rule"},
            },
            "q3": {
                "predicted_level": 3,
                "predicted_level_name": "中等题",
                "postprocess_original_level": "中等题",
                "selected_action": None,
            },
        }

        report, _ = evaluation.evaluate_predictions(
            labels,
            predictions,
            error_ids=set(),
            error_messages={},
        )

        self.assertEqual(
            report["postprocess_rule_attribution"]["boundary_rule"],
            {
                "triggered": 2,
                "helped": 1,
                "hurt": 1,
                "unchanged": 0,
                "net": 0,
            },
        )
        self.assertEqual(report["postprocess_net_improvement"], 0)

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
