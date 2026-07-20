import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "physics_difficulty_agent_pipeline.py"
SPEC = importlib.util.spec_from_file_location("physics_difficulty_agent_pipeline", SCRIPT)
agent = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = agent
SPEC.loader.exec_module(agent)


def result_item(level="中等题", *, raw=None, features=None, actions=None):
    return {
        "question_id": "10001",
        "stem": "某物理题题干",
        "options": "A. 甲\nB. 乙",
        "analysis": "根据题意完成判断。",
        "difficulty_level_raw": raw or level,
        "postprocess_actions": actions or [],
        "source_difficulty_untrusted": 5,
        "difficulty_rating": {
            "difficulty_level": level,
            "features": features
            or {
                "step_count": "3-5步",
                "formula_count": "2-3个",
                "calculation_complexity": "简单笔算",
                "reasoning_chain": "多层因果推理",
                "problem_structure": "力学综合",
                "information_carrier": "纯文字",
                "subquestion_dependency": "无多问",
                "knowledge_count": "2-3个",
                "cross_module": "同一模块内部",
                "state_count": "双状态",
                "constraint_count": "单一约束",
                "variable_relation": "简单正反比",
                "experiment_requirement": "无",
                "graph_table_requirement": "无",
            },
            "reasoning": {"core_basis": "首轮理由不应进入盲审。"},
        },
    }


class RiskRoutingTests(unittest.TestCase):
    def test_routes_any_first_stage_postprocess_adjustment(self):
        item = result_item(
            actions=[{"rule": "medium_to_hard", "from": "中等题", "to": "拔高题"}],
            level="拔高题",
            raw="中等题",
        )
        route = agent.route_verification_risk(item)
        self.assertTrue(route["selected"])
        self.assertIn("首轮发生后处理调整", route["reasons"])
        self.assertEqual(route["allowed_directions"], ["down"])

    def test_routes_low_structure_medium_downward(self):
        features = result_item()["difficulty_rating"]["features"]
        features.update(
            {
                "step_count": "1-2步",
                "formula_count": "0-1个",
                "calculation_complexity": "口算或直接判断",
                "reasoning_chain": "直接套用",
                "problem_structure": "概念判断",
                "knowledge_count": "1个",
                "state_count": "单状态",
                "constraint_count": "无约束",
                "variable_relation": "无变量关系",
            }
        )
        route = agent.route_verification_risk(result_item(level="中等题", features=features))
        self.assertTrue(route["selected"])
        self.assertEqual(route["allowed_directions"], ["down"])

    def test_does_not_route_well_aligned_medium(self):
        route = agent.route_verification_risk(result_item())
        self.assertFalse(route["selected"])
        self.assertEqual(route["allowed_directions"], [])


class BlindInputTests(unittest.TestCase):
    def test_blind_content_excludes_first_stage_and_source_labels(self):
        item = result_item(
            actions=[{"rule": "secret_rule", "from": "基础题", "to": "中等题"}]
        )
        content = agent.build_blind_review_content(item)
        self.assertIn("某物理题题干", content)
        self.assertIn("根据题意完成判断", content)
        self.assertNotIn("中等题", content)
        self.assertNotIn("首轮理由", content)
        self.assertNotIn("secret_rule", content)
        self.assertNotIn("source_difficulty_untrusted", content)


class ConservativeGateTests(unittest.TestCase):
    def review(self, level, confidence="高", boundary="明确归档", acceptable=None):
        return {
            "review_level": level,
            "acceptable_levels": acceptable or [level],
            "boundary_status": boundary,
            "confidence": confidence,
            "effective_task_summary": "独立识别了真实任务结构。",
            "effective_decision_count": 5,
            "task_structure": "决定性转换",
            "structural_evidence": ["存在明确的图像反推和隐含条件"],
            "boundary_basis": "与相邻档相比存在明确差异。",
        }

    def test_applies_high_confidence_adjacent_change_in_routed_direction(self):
        route = {"selected": True, "allowed_directions": ["up"], "reasons": ["结构偏高"]}
        apply, reason = agent.should_apply_blind_review(
            "中等题", route, self.review("拔高题"), {"高"}
        )
        self.assertTrue(apply)
        self.assertIn("高置信度", reason)

    def test_rejects_two_level_jump(self):
        route = {"selected": True, "allowed_directions": ["up"], "reasons": ["结构偏高"]}
        apply, reason = agent.should_apply_blind_review(
            "基础题", route, self.review("拔高题"), {"高"}
        )
        self.assertFalse(apply)
        self.assertIn("超过一档", reason)

    def test_rejects_direction_not_supported_by_router(self):
        route = {"selected": True, "allowed_directions": ["down"], "reasons": ["结构偏低"]}
        apply, reason = agent.should_apply_blind_review(
            "中等题", route, self.review("拔高题"), {"高"}
        )
        self.assertFalse(apply)
        self.assertIn("方向", reason)

    def test_rejects_medium_confidence(self):
        route = {"selected": True, "allowed_directions": ["up"], "reasons": ["结构偏高"]}
        apply, reason = agent.should_apply_blind_review(
            "中等题", route, self.review("拔高题", confidence="中"), {"高"}
        )
        self.assertFalse(apply)
        self.assertIn("置信度", reason)

    def test_keeps_current_when_review_declares_boundary_including_current(self):
        route = {"selected": True, "allowed_directions": ["down"], "reasons": ["边界"]}
        review = self.review(
            "基础题",
            boundary="相邻边界均可",
            acceptable=["基础题", "中等题"],
        )
        apply, reason = agent.should_apply_blind_review("中等题", route, review, {"高"})
        self.assertFalse(apply)
        self.assertIn("当前等级仍可接受", reason)

    def test_rejects_review_without_valid_decision_count(self):
        route = {"selected": True, "allowed_directions": ["up"], "reasons": ["结构偏高"]}
        review = self.review("拔高题")
        review["effective_decision_count"] = "约5步"
        apply, reason = agent.should_apply_blind_review("中等题", route, review, {"高"})
        self.assertFalse(apply)
        self.assertIn("effective_decision_count", reason)

    def test_rejects_review_with_unknown_task_structure(self):
        route = {"selected": True, "allowed_directions": ["up"], "reasons": ["结构偏高"]}
        review = self.review("拔高题")
        review["task_structure"] = "非常难的题"
        apply, reason = agent.should_apply_blind_review("中等题", route, review, {"高"})
        self.assertFalse(apply)
        self.assertIn("task_structure", reason)

    def test_rejects_upward_review_whose_structure_does_not_support_target(self):
        route = {"selected": True, "allowed_directions": ["up"], "reasons": ["结构偏高"]}
        review = self.review("拔高题")
        review.update(
            {
                "effective_decision_count": 3,
                "task_structure": "常规分析",
                "structural_evidence": ["三个常规步骤"],
                "boundary_basis": "方法均为显性常规公式。",
            }
        )
        apply, reason = agent.should_apply_blind_review("中等题", route, review, {"高"})
        self.assertFalse(apply)
        self.assertIn("结构证据不足", reason)

    def test_rejects_final_upgrade_without_two_strong_final_signals(self):
        route = {"selected": True, "allowed_directions": ["up"], "reasons": ["结构偏高"]}
        review = self.review("压轴题")
        review.update(
            {
                "effective_decision_count": 7,
                "task_structure": "高密度综合链",
                "structural_evidence": ["存在多状态"],
                "boundary_basis": "步骤较长，但没有分类或边界验证。",
            }
        )
        apply, reason = agent.should_apply_blind_review("拔高题", route, review, {"高"})
        self.assertFalse(apply)
        self.assertIn("结构证据不足", reason)


class AuditAndResumeTests(unittest.TestCase):
    def test_apply_verified_level_preserves_complete_first_stage_snapshot(self):
        item = result_item(level="中等题")
        original = json.loads(json.dumps(item["difficulty_rating"], ensure_ascii=False))
        agent.apply_verified_level(item, "拔高题")
        self.assertEqual(item["difficulty_rating_before_verification"], original)
        self.assertEqual(item["difficulty_rating"]["difficulty_level"], "拔高题")
        self.assertEqual(item["difficulty_rating_before_verification"]["difficulty_level"], "中等题")

    def test_resume_rejects_rows_from_different_agent_configuration(self):
        expected = "signature-new"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "question_id": "q1",
                        "verification_agent": {"run_signature": "signature-old"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "配置不一致"):
                agent.validate_resume_output(str(path), expected)


if __name__ == "__main__":
    unittest.main()
