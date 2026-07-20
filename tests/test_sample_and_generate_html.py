import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "sample_and_generate_html.py"
SPEC = importlib.util.spec_from_file_location("sample_and_generate_html", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SampleAndGenerateHtmlTests(unittest.TestCase):
    def test_unmarked_questions_default_to_model_accepted(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("未标注即视为“模型判定合理”", template)
        self.assertIn("default_model_accepted", template)
        self.assertIn("human_reviewed: manuallyRejected", template)

    def test_dashboard_counts_only_manual_exceptions(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("已标异常", template)
        self.assertIn("const correct = total - wrong", template)
        self.assertIn("人工标记不准", template)

    def test_typing_a_reason_marks_question_wrong(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("annotations[qid].verdict = 'wrong'", template)

    def test_corrected_level_is_structured_and_marks_exception(self) -> None:
        template = MODULE.HTML_TEMPLATE
        self.assertIn("function saveCorrectedLevel", template)
        self.assertIn("human_difficulty_level", template)
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("建议正确档位", source)
        self.assertIn("输入内容后会自动标记", source)


if __name__ == "__main__":
    unittest.main()
