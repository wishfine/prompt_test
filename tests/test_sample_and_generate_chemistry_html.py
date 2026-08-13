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
    def test_parse_level_plan_accepts_requested_five_counts(self) -> None:
        self.assertEqual(
            MODULE.parse_level_plan("120,120,120,90,50"),
            {
                "送分题": 120,
                "基础题": 120,
                "中等题": 120,
                "拔高题": 90,
                "压轴题": 50,
            },
        )

    def test_select_rows_by_level_plan_is_exact_and_reproducible(self) -> None:
        grouped = {
            level: [
                {
                    "question_id": f"{level}-{index}",
                    "difficulty_rating": {"difficulty_level": level},
                }
                for index in range(10)
            ]
            for level in MODULE.LEVEL_MAP
        }
        plan = {level: 3 for level in MODULE.LEVEL_MAP}

        first = MODULE.select_rows_by_level_plan(grouped, plan, seed=7)
        second = MODULE.select_rows_by_level_plan(grouped, plan, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(
            {level: len(rows) for level, rows in first.items()},
            {level: 3 for level in MODULE.LEVEL_MAP},
        )

    def test_select_rows_by_level_plan_rejects_short_level(self) -> None:
        grouped = {level: [] for level in MODULE.LEVEL_MAP}
        grouped["送分题"] = [{"question_id": "only-one"}]

        with self.assertRaisesRegex(ValueError, "送分题仅1题"):
            MODULE.select_rows_by_level_plan(
                grouped,
                {"送分题": 2},
                seed=1,
            )

    def test_select_rows_by_level_plan_fills_shortage_from_adjacent_level(
        self,
    ) -> None:
        grouped = {level: [] for level in MODULE.LEVEL_MAP}
        grouped["送分题"] = [{"question_id": "easy-1"}]
        grouped["基础题"] = [
            {"question_id": f"basic-{index}"}
            for index in range(6)
        ]
        plan = {
            "送分题": 3,
            "基础题": 2,
            "中等题": 0,
            "拔高题": 0,
            "压轴题": 0,
        }

        first = MODULE.select_rows_by_level_plan(
            grouped,
            plan,
            seed=7,
            allow_cross_level_fill=True,
        )
        second = MODULE.select_rows_by_level_plan(
            grouped,
            plan,
            seed=7,
            allow_cross_level_fill=True,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first["送分题"]), 1)
        self.assertEqual(len(first["基础题"]), 4)
        selected_ids = [
            row["question_id"]
            for rows in first.values()
            for row in rows
        ]
        self.assertEqual(len(selected_ids), 5)
        self.assertEqual(len(set(selected_ids)), 5)

    def test_cross_level_fill_rejects_insufficient_total_pool(self) -> None:
        grouped = {level: [] for level in MODULE.LEVEL_MAP}
        grouped["送分题"] = [{"question_id": "easy-1"}]

        with self.assertRaisesRegex(ValueError, "全部档位合计仅1题"):
            MODULE.select_rows_by_level_plan(
                grouped,
                {"送分题": 2},
                seed=1,
                allow_cross_level_fill=True,
            )

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
            labels,
            [
                "最长解题链",
                "任务组",
                "解题方法",
                "计算操作",
                "误差分析",
                "知识点跨度",
                "关键条件处理",
                "并列/关联任务",
                "图表操作",
                "解题任务结构",
                "实验任务结构",
                "图像任务结构",
                "题干字数",
            ],
        )
        self.assertNotIn("规则族数", labels)
        self.assertNotIn("小问数", labels)
        self.assertNotIn("知识课题数", labels)
        self.assertNotIn("题干图片资源数", labels)
        values = dict(fields)
        self.assertEqual(values["最长解题链"], "1.a → 2.b → 3.c")
        self.assertEqual(values["任务组"], "实验操作与探究×4")
        self.assertEqual(
            values["知识点跨度"],
            "跨单元并列（U1-2 化学实验与科学探究、U9-3 溶质的质量分数）",
        )
        self.assertEqual(values["题干字数"], "128")

        rendered_grid = MODULE._render_feature_grid(
            fields,
            css_class="priority-details",
        )
        self.assertIn("task-structure-start", rendered_grid)
        self.assertEqual(rendered_grid.count("task-structure-card"), 3)

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
        self.assertNotIn("小问数", rendered)
        self.assertNotIn("明示小问数", rendered)
        self.assertNotIn("有效任务数", rendered)
        self.assertNotIn("知识课题数", rendered)
        self.assertNotIn("题干图片资源数", rendered)
        self.assertIn("任务组", rendered)
        self.assertIn("解题方法", rendered)
        self.assertIn("关键条件处理", rendered)
        self.assertIn("解题任务结构", rendered)
        self.assertNotIn("作答规则族", rendered)
        self.assertNotIn(">作答规则<", rendered)
        self.assertNotIn(">条件操作<", rendered)
        self.assertNotIn("规则族数", rendered)
        self.assertNotIn("解题拓扑", rendered)
        self.assertNotIn("冻结评级版本", rendered)
        self.assertNotIn("可视化不改档", rendered)
        self.assertIn("1.读取数据", rendered)
        self.assertIn("U9-3 溶质的质量分数", rendered)


if __name__ == "__main__":
    unittest.main()
