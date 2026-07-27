import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "sample_jsonl_excluding.py"
SPEC = importlib.util.spec_from_file_location("sample_jsonl_excluding", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SampleJsonlExcludingTests(unittest.TestCase):
    def test_even_stratified_sample_is_reproducible(self) -> None:
        rows = [
            {"question_id": str(index), "difficulty": str(index % 5 + 1)}
            for index in range(100)
        ]
        first, _, first_distribution = MODULE.sample_rows(
            rows, 50, 20260727, "difficulty"
        )
        second, _, second_distribution = MODULE.sample_rows(
            rows, 50, 20260727, "difficulty"
        )
        self.assertEqual(first, second)
        self.assertEqual(first_distribution, second_distribution)
        self.assertEqual(first_distribution, {str(i): 10 for i in range(1, 6)})

    def test_small_stratum_is_filled_from_other_strata(self) -> None:
        rows = [
            {"question_id": f"a-{index}", "difficulty": "1"}
            for index in range(2)
        ] + [
            {"question_id": f"b-{index}", "difficulty": "2"}
            for index in range(20)
        ]
        sampled, _, distribution = MODULE.sample_rows(
            rows, 10, 7, "difficulty"
        )
        self.assertEqual(len(sampled), 10)
        self.assertEqual(distribution, {"1": 2, "2": 8})


if __name__ == "__main__":
    unittest.main()
