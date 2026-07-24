# -*- coding: utf-8 -*-
"""高中物理两阶段批处理脚本的静态契约测试。"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "high_physics_difficulty_rating_and_verify.py"
sys.path.insert(0, str(ROOT / "src"))

import high_physics_difficulty_rating_and_verify as runner  # noqa: E402


class RunnerAssetTests(unittest.TestCase):
    def test_runner_exists_and_compiles(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        compile(source, str(RUNNER), "exec")
        self.assertIn('"high_physics_two_stage_v6"', source)

    def test_runner_exposes_required_operational_controls(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        for flag in (
            "--input",
            "--output",
            "--errors",
            "--prompt",
            "--concurrency",
            "--limit",
            "--no-cache",
            "--image-mode",
        ):
            self.assertIn(flag, source)

    def test_runner_uses_environment_credentials_and_both_stages(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn('os.getenv("API_KEY"', source)
        self.assertNotRegex(
            source,
            re.compile(r'API_KEY\s*=\s*["\'][0-9a-f]{20,}["\']', re.I),
        )
        self.assertIn("FEATURE_EXTRACTION_PROMPT_PREFIX", source)
        self.assertIn("VERIFICATION_PROMPT_PREFIX", source)
        self.assertIn("enrich_stage1_rating", source)
        self.assertIn("normalize_stage1_rating", source)
        self.assertIn("finalize_level", source)

    def test_verification_rejects_duplicate_high_features(self) -> None:
        value = {
            "difficulty_source": "测试",
            "feature_corrections": [],
            "missed_features": ["无"],
            "reviewed_original_predicted_accuracy": 68.0,
            "reviewed_high_difficulty_features": ["多约束联合", "多约束联合"],
            "analysis": "测试",
        }
        with self.assertRaisesRegex(ValueError, "不得重复"):
            runner.validate_verification(value)

    def test_verification_rejects_empty_required_text(self) -> None:
        value = {
            "difficulty_source": " ",
            "feature_corrections": [],
            "missed_features": ["无"],
            "reviewed_original_predicted_accuracy": 68.0,
            "reviewed_high_difficulty_features": [],
            "analysis": "测试",
        }
        with self.assertRaisesRegex(ValueError, "difficulty_source"):
            runner.validate_verification(value)

    def test_verification_validates_correction_structure_and_list_items(self) -> None:
        base = {
            "difficulty_source": "测试",
            "feature_corrections": ["不是对象"],
            "missed_features": ["无"],
            "reviewed_original_predicted_accuracy": 68.0,
            "reviewed_high_difficulty_features": [],
            "analysis": "测试",
        }
        with self.assertRaisesRegex(ValueError, "必须为对象"):
            runner.validate_verification(base)

        base["feature_corrections"] = []
        base["reviewed_high_difficulty_features"] = [{"name": "多约束联合"}]
        with self.assertRaisesRegex(ValueError, "每项必须为字符串"):
            runner.validate_verification(base)

    def test_verification_rejects_non_feature_correction_field(self) -> None:
        value = {
            "difficulty_source": "测试",
            "feature_corrections": [
                {
                    "field": "difficulty_level_step1",
                    "from": "难度4档",
                    "to": "难度3档",
                    "evidence": "档位不属于 feature 字段",
                }
            ],
            "missed_features": ["无"],
            "reviewed_original_predicted_accuracy": 68.0,
            "reviewed_high_difficulty_features": [],
            "analysis": "测试",
        }
        with self.assertRaisesRegex(ValueError, "非法 feature 修正字段"):
            runner.validate_verification(value)

    def test_stage2_error_record_preserves_paid_stage1_result(self) -> None:
        record = runner.build_pipeline_error(
            output_base={"question_id": "100"},
            error=RuntimeError("第二阶段请求失败"),
            stage1={"predicted_accuracy": 68.0},
            stage1_usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            stage1_elapsed=1.25,
        )
        self.assertEqual(record["failed_stage"], "stage2")
        self.assertEqual(
            record["difficulty_rating_stage1"]["predicted_accuracy"],
            68.0,
        )
        self.assertEqual(record["api_stage1_usage"]["total_tokens"], 15)

    def test_per_level_sampling_is_balanced_and_does_not_mutate_rows(self) -> None:
        rows = [
            {"question_id": f"{level}-{index}", "difficulty": str(level)}
            for level in range(1, 6)
            for index in range(3)
        ]
        sampled = runner.sample_questions_per_level(rows, per_level=2, seed=7)
        counts = {}
        for row in sampled:
            counts[row["difficulty"]] = counts.get(row["difficulty"], 0) + 1
        self.assertEqual(counts, {str(level): 2 for level in range(1, 6)})
        self.assertEqual(len(rows), 15)


if __name__ == "__main__":
    unittest.main()
