# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import physics_difficulty_production_pipeline as production  # noqa: E402


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def rated_item(
    question_id: str,
    level: str,
    *,
    unanimous: bool = True,
    calibration: bool = False,
) -> dict:
    predictions = [level, level, level]
    if not unanimous:
        predictions = ["送分题", "送分题", "基础题"]
    actions = []
    method = "unanimous" if unanimous else "structured_easy_guard"
    if calibration:
        actions.append(
            {
                "rule": "structured_easy_disagreement_guard",
                "from": "送分题",
                "to": "基础题",
            }
        )
    features = {field: "test" for field in production.REQUIRED_FEATURES}
    reasoning = {
        field: "test" for field in production.REQUIRED_REASONING_FIELDS
    }
    level_index = production.LEVELS.index(level)
    lower = production.LEVELS[level_index - 1] if level_index > 0 else None
    higher = (
        production.LEVELS[level_index + 1]
        if level_index + 1 < len(production.LEVELS)
        else None
    )
    return {
        "question_id": question_id,
        "difficulty_rating": {
            "difficulty_level": level,
            "features": features,
            "reasoning": reasoning,
            "adjacent_lower_level": lower,
            "adjacent_higher_level": higher,
            "adjacent_reasoning_normalized": True,
        },
        "lite_self_consistency": {
            "run_count": 3,
            "run_predictions": predictions,
            "unanimous": unanimous,
            "decision_method": method,
            "calibration_actions": actions,
        },
        "api_prompt_tokens": 10,
        "api_completion_tokens": 2,
        "api_total_tokens": 12,
        "api_time_use": 1.5,
    }


class PhysicsProductionPipelineTests(unittest.TestCase):
    def test_validate_complete_output_requires_exact_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.jsonl"
            write_jsonl(path, [rated_item("q1", "基础题")])
            with self.assertRaisesRegex(ValueError, "缺少 1"):
                production.validate_complete_output(
                    path,
                    {"q1", "q2"},
                )

    def test_validate_complete_output_checks_ensemble_run_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.jsonl"
            row = rated_item("q1", "基础题")
            row["lite_self_consistency"]["run_count"] = 5
            write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "运行次数不是 3"):
                production.validate_complete_output(
                    path,
                    {"q1"},
                    expected_run_count=3,
                )

    def test_validate_partial_run_rejects_foreign_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.jsonl"
            write_jsonl(path, [rated_item("other", "中等题")])
            with self.assertRaisesRegex(ValueError, "当前输入之外"):
                production.validate_partial_run(path, {"q1"})

    def test_validate_complete_output_rejects_cross_level_reasoning(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.jsonl"
            row = rated_item("q1", "基础题")
            row["difficulty_rating"]["adjacent_higher_level"] = "拔高题"
            write_jsonl(path, [row])
            with self.assertRaisesRegex(ValueError, "非法等级 1"):
                production.validate_complete_output(path, {"q1"})

    def test_manifest_refuses_unprotected_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_path = root / "batch_run1.jsonl"
            write_jsonl(run_path, [rated_item("q1", "基础题")])
            with self.assertRaisesRegex(ValueError, "没有生产清单保护"):
                production.ensure_manifest_compatible(
                    root / "batch_manifest.json",
                    {"pipeline_version": "x"},
                    [run_path],
                )

    def test_manifest_refuses_signature_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            production.write_json_atomic(
                manifest,
                {"signature": {"pipeline_version": "old", "input": "a"}},
            )
            with self.assertRaisesRegex(ValueError, "签名与本次配置不一致"):
                production.ensure_manifest_compatible(
                    manifest,
                    {"pipeline_version": "new", "input": "a"},
                    [],
                )

    def test_monitoring_summary_reports_drift_and_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "input.jsonl"
            write_jsonl(
                input_path,
                [
                    {"question_id": "q1", "stem": "a"},
                    {"question_id": "q2", "stem": "b"},
                ],
            )
            run_paths = []
            error_paths = []
            for index in range(1, 4):
                run_path = root / f"run{index}.jsonl"
                write_jsonl(
                    run_path,
                    [
                        rated_item("q1", "基础题"),
                        rated_item("q2", "基础题"),
                    ],
                )
                run_paths.append(run_path)
                error_path = root / f"run{index}_errors.jsonl"
                error_path.write_text("", encoding="utf-8")
                error_paths.append(error_path)
            final_path = root / "final.jsonl"
            write_jsonl(
                final_path,
                [
                    rated_item("q1", "基础题"),
                    rated_item(
                        "q2",
                        "基础题",
                        unanimous=False,
                        calibration=True,
                    ),
                ],
            )
            summary = production.build_monitoring_summary(
                input_path=input_path,
                run_paths=run_paths,
                error_paths=error_paths,
                final_path=final_path,
                signature={
                    "git_commit": "abc",
                    "input_sha256": production.sha256_file(input_path),
                },
            )

        self.assertEqual(summary["input_rows"], 2)
        self.assertEqual(summary["final_rows"], 2)
        self.assertEqual(summary["unanimous_count"], 1)
        self.assertEqual(summary["disagreement_count"], 1)
        self.assertEqual(
            summary["calibration_rules"]["structured_easy_disagreement_guard"],
            1,
        )
        self.assertEqual(summary["api_prompt_tokens"], 60)
        self.assertEqual(summary["api_completion_tokens"], 12)
        self.assertEqual(summary["api_total_tokens"], 72)
        self.assertTrue(summary["warnings"])


if __name__ == "__main__":
    unittest.main()
