import copy
import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "src" / "chemistry_basic_medium_boundary_review.py"
spec = importlib.util.spec_from_file_location(
    "chemistry_basic_medium_boundary_review",
    SCRIPT,
)
review = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(review)


def make_item(
    level="基础题",
    *,
    tasks=4,
    rules=None,
    topics=None,
    response=None,
):
    rules = rules or ["性质用途或现象判断"]
    topics = topics or ["U2-2", "U6-2"]
    response = response or []
    return {
        "question_id": "q1",
        "stem": "四个选项分别需要判断不同化学事实。",
        "difficulty_rating": {
            "difficulty_level": level,
            "features": {
                "longest_solution_chain": ["识别对象", "分别核验规则"],
                "task_groups": [{"task_type": "性质与反应判断", "count": tasks}],
                "rule_families": rules,
                "curriculum_topics": topics,
                "response_operations": response,
            },
            "observable_metrics": {
                "effective_task_count": tasks,
                "longest_chain_steps": 2,
                "rule_family_count": len(rules),
                "curriculum_topic_count": len(topics),
            },
        },
    }


class ChemistryBasicMediumBoundaryReviewTests(unittest.TestCase):
    def test_routes_only_plausible_basic_to_medium_boundary(self):
        candidate = review.select_boundary_candidate(make_item())
        self.assertTrue(candidate["selected"])
        self.assertEqual(candidate["allowed_levels"], ["基础题", "中等题"])

        simple = make_item(tasks=1, topics=["U2-2"])
        simple["difficulty_rating"]["features"]["longest_solution_chain"] = [
            "直接识记"
        ]
        simple["difficulty_rating"]["observable_metrics"].update(
            effective_task_count=1,
            longest_chain_steps=1,
            curriculum_topic_count=1,
        )
        self.assertFalse(review.select_boundary_candidate(simple)["selected"])
        self.assertFalse(
            review.select_boundary_candidate(make_item(level="中等题"))["selected"]
        )

    def test_high_confidence_review_requires_decisive_medium_evidence(self):
        allowed = ["基础题", "中等题"]
        valid = {
            "review_level": "中等题",
            "confidence": "高",
            "decisive_evidence": ["异质规则切换"],
            "effective_task_summary": "四个选项切换多类规则。",
            "boundary_basis": "超过单一规则一步应用。",
            "first_pass_issue": "首轮将异质规则压成同一规则。",
        }
        self.assertIsNone(review.validate_review_result(valid, allowed))

        weak = copy.deepcopy(valid)
        weak["decisive_evidence"] = ["选项或对象数量多"]
        self.assertIsNotNone(review.validate_review_result(weak, allowed))

        multi_dimension = copy.deepcopy(valid)
        multi_dimension["decisive_evidence"] = [
            "同一主题多性质维度综合辨析"
        ]
        self.assertIsNone(
            review.validate_review_result(multi_dimension, allowed)
        )

    def test_writeback_is_narrow_and_auditable(self):
        item = make_item()
        candidate = review.select_boundary_candidate(item)
        result = {
            "review_level": "中等题",
            "confidence": "高",
            "decisive_evidence": ["多类化学符号含义互扰"],
            "effective_task_summary": "需区分多类符号位置含义。",
            "boundary_basis": "存在多类规则辨析和互扰。",
            "first_pass_issue": "首轮误认为同一规则重复。",
        }
        applied, reason = review.apply_review_to_item(
            item,
            candidate,
            result,
            writeback=True,
        )
        self.assertTrue(applied, reason)
        self.assertEqual(
            item["difficulty_rating"]["difficulty_level"],
            "中等题",
        )
        self.assertEqual(
            item["postprocess_actions"][-1]["rule"],
            "chemistry_basic_to_medium_boundary_review",
        )


if __name__ == "__main__":
    unittest.main()
