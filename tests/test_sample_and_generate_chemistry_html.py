import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "sample_and_generate_chemistry_html.py"
SPEC = importlib.util.spec_from_file_location(
    "sample_and_generate_chemistry_html",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SampleAndGenerateChemistryHtmlTests(unittest.TestCase):
    def test_stem_visualization_uses_only_last_image(self) -> None:
        section = MODULE.render_chemistry_image_section(
            "https://img.example/first.png,"
            "https://img.example/second.png,"
            "https://img.example/final.png",
            title="题干图示",
            kind="stem",
        )

        self.assertIn("https://img.example/final.png", section)
        self.assertNotIn("https://img.example/first.png", section)
        self.assertNotIn("https://img.example/second.png", section)
        self.assertIn("题干多图时仅展示最后一张", section)

    def test_analysis_visualization_keeps_single_last_image(self) -> None:
        section = MODULE.render_chemistry_image_section(
            ["a.png", "b.png"],
            title="解析图示",
            kind="analysis",
        )

        self.assertIn("b.png", section)
        self.assertNotIn('src="a.png"', section)

    def test_teacher_priority_features_are_ordered_first(self) -> None:
        item = {
            "question_text_char_count": 128,
            "explicit_subquestion_count": 4,
            "stem_pic_url": "a.png,b.png,c.png,d.png",
        }
        rating = {
            "features": {
                "longest_solution_chain": ["a", "b", "c"],
                "task_groups": [
                    {"task_type": "实验操作与探究", "count": 4}
                ],
                "rule_families": ["实验操作规范", "异常失败或误差诊断"],
                "curriculum_topics": ["U1-2", "U9-3"],
                "parallel_task_relation": "不同规则的独立任务",
                "solution_topology": "条件分支或范围筛选",
                "reaction_structure": "单一反应",
                "condition_operations": ["分类讨论"],
                "representation_operations": [],
                "evidence_operations": ["多证据共同成立"],
                "experiment_operation": "方案评价或补充实验",
                "experiment_task_structure": "操作偏差因果链",
                "visual_task_structure": "多图独立不同规则判断",
                "graph_table_operation": "多组比较",
                "error_analysis_operation": "操作偏差到最终结果方向",
                "calculation_operations": ["单一方程式"],
                "new_information_operation": "无新信息",
            },
            "observable_metrics": {
                "longest_chain_steps": 3,
                "effective_task_count": 4,
                "rule_family_count": 2,
                "curriculum_topic_count": 2,
                "curriculum_unit_count": 2,
                "curriculum_span_summary": "跨单元并列（U1-2、U9-3）",
            },
        }

        fields = MODULE.build_priority_feature_items(item, rating)
        labels = [field[0] for field in fields]

        self.assertEqual(
            labels[:10],
            [
                "最长解题链",
                "有效任务数",
                "明示小问数",
                "知识课题数",
                "规则族数",
                "解题拓扑",
                "误差分析",
                "计算操作",
                "条件操作",
                "图像任务结构",
            ],
        )
        values = dict(fields)
        self.assertEqual(values["题干字数"], "128")
        self.assertEqual(values["题干图片资源数"], "4")

    def test_generated_page_matches_physics_review_workflow(self) -> None:
        samples = {
            3: [
                {
                    "question_id": "q-1",
                    "question_text_char_count": 80,
                    "explicit_subquestion_count": 1,
                    "stem_pic_url": "first.png,last.png",
                    "analysis_pic_url": "analysis.png",
                    "difficulty_rating": {
                        "difficulty_level": "中等题",
                        "features": {
                            "longest_solution_chain": ["读取数据"],
                            "task_groups": [
                                {"task_type": "图表与数据", "count": 1}
                            ],
                            "rule_families": ["图表读取或数据归纳"],
                            "curriculum_topics": ["U9-3"],
                            "parallel_task_relation": "单一答题目标",
                            "solution_topology": "单点直接回答",
                            "reaction_structure": "无反应任务",
                            "condition_operations": [],
                            "representation_operations": [],
                            "evidence_operations": [],
                            "experiment_operation": "无",
                            "experiment_task_structure": "无实验判断",
                            "visual_task_structure": "单图直接识别",
                            "graph_table_operation": "直接读数",
                            "error_analysis_operation": "无误差分析",
                            "calculation_operations": [],
                            "new_information_operation": "无新信息",
                        },
                        "observable_metrics": {
                            "longest_chain_steps": 1,
                            "effective_task_count": 1,
                            "rule_family_count": 1,
                            "curriculum_topic_count": 1,
                            "curriculum_unit_count": 1,
                            "curriculum_span_summary": "单一课题（U9-3）",
                        },
                        "reasoning": {"core_basis": "直接读取。"},
                    },
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = Path(tmp_dir) / "chemistry.html"
            MODULE.generate_html_file(samples, str(output))
            rendered = output.read_text(encoding="utf-8")

        self.assertIn("未标注即视为“模型判定合理”", rendered)
        self.assertIn("建议正确档位", rendered)
        self.assertIn("last.png", rendered)
        self.assertNotIn('src="first.png"', rendered)
        self.assertIn("关键可观测证据", rendered)
        self.assertIn("全部17项特征", rendered)


if __name__ == "__main__":
    unittest.main()
