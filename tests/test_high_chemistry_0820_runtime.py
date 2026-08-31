from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# The runtime dependencies are installed in the server venv.  Stub them here so
# the local unit test can exercise pure response-validation helpers as well.
for module_name in ("aiofiles", "aiohttp", "json_repair"):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))
dotenv = sys.modules.setdefault("dotenv", types.ModuleType("dotenv"))
dotenv.load_dotenv = lambda: None
tqdm_package = sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
tqdm_asyncio = sys.modules.setdefault("tqdm.asyncio", types.ModuleType("tqdm.asyncio"))
tqdm_asyncio.tqdm = object()

import high_chemistry_difficulty_rating_and_verify_0820 as runtime


class ResponseCompletionTest(unittest.TestCase):
    def test_completed_response_is_accepted(self) -> None:
        self.assertIsNone(
            runtime._response_completion_error(
                {"status": "completed", "usage": {"output_tokens": 123}},
                "第一阶段",
            )
        )

    def test_incomplete_response_reports_reason_and_usage(self) -> None:
        error = runtime._response_completion_error(
            {
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"output_tokens": 8000},
            },
            "第一阶段",
        )

        self.assertEqual(
            error,
            "第一阶段模型响应未完成：status=incomplete，"
            "reason=max_output_tokens，output_tokens=8000",
        )

    def test_missing_status_remains_compatible(self) -> None:
        self.assertIsNone(runtime._response_completion_error({}, "第二阶段"))

    def test_output_budgets_cover_large_structured_responses(self) -> None:
        self.assertEqual(runtime.STAGE1_MAX_OUTPUT_TOKENS, 8000)
        self.assertEqual(runtime.STAGE2_MAX_OUTPUT_TOKENS, 5000)

    def test_runtime_defaults_to_one_total_attempt(self) -> None:
        args = runtime.build_parser().parse_args([])
        self.assertEqual(args.retries, 1)


if __name__ == "__main__":
    unittest.main()
