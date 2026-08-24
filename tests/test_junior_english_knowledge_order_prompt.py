import unittest
from pathlib import Path


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "初中英语知识点排序提示词.txt"


class JuniorEnglishKnowledgeOrderPromptTests(unittest.TestCase):
    def test_prompt_has_context_specific_priority_rules_and_examples(self):
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        required_fragments = [
            "填空题和翻译题",
            "直接对应空格中要填写的固定搭配",
            "语篇主题 > 语篇体裁",
            "请求允许",
            "make a promise",
            "记叙文",
        ]
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, prompt)


if __name__ == "__main__":
    unittest.main()
