# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import run_disagreement_model_benchmark as benchmark  # noqa: E402


class BenchmarkConfigTests(unittest.TestCase):
    def test_default_model_list_has_expected_models(self) -> None:
        self.assertEqual(len(benchmark.DEFAULT_MODELS), 10)
        self.assertIn("glm-5.2", benchmark.DEFAULT_MODELS)
        self.assertIn("doubao-seed-2.1-turbo", benchmark.DEFAULT_MODELS)

    def test_mini_uses_temperature_zero_only(self) -> None:
        self.assertEqual(
            benchmark.configured_temperature("doubao-seed-2.0-mini"),
            "0",
        )
        self.assertIsNone(
            benchmark.configured_temperature("doubao-seed-2.0-lite")
        )
        self.assertIsNone(benchmark.configured_temperature("qwen3-max"))

    def test_command_uses_balanced_single_judge_and_auto_api(self) -> None:
        args = argparse.Namespace(
            run=list(benchmark.DEFAULT_RUNS),
            labels=benchmark.DEFAULT_LABELS,
            reference_questions=benchmark.DEFAULT_REFERENCES,
            prompt=benchmark.DEFAULT_PROMPT,
            fewshot_per_level=3,
            concurrency=10,
            timeout=180,
            retries=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            command = benchmark.build_command(
                args,
                "doubao-seed-2.0-mini",
                Path(tmp),
            )
        self.assertIn("balanced", command)
        self.assertIn("auto", command)
        temperature_index = command.index("--temperature")
        self.assertEqual(command[temperature_index + 1], "0")
        self.assertNotIn("--arbiter-model", command)

    def test_model_slug_removes_unsafe_characters(self) -> None:
        self.assertEqual(benchmark.model_slug(" glm-5.2\u00a0"), "glm-5.2")


if __name__ == "__main__":
    unittest.main()
