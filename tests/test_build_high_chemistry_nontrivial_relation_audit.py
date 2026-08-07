# -*- coding: utf-8 -*-

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "build_high_chemistry_nontrivial_relation_audit.py"
SPEC = importlib.util.spec_from_file_location("nontrivial_relation_audit", TOOL_PATH)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def label(question_id: str, level: str) -> dict:
    return {"question_id": question_id, "reviewed_difficulty_level": level}


def prediction(question_id: str, level: str, nontrivial_count: int) -> dict:
    return {
        "question_id": question_id,
        "final_difficulty_level": level,
        "difficulty_rating_stage1": {"nontrivial_task_count": nontrivial_count, "features": {}},
    }


class BuildNontrivialRelationAuditTests(unittest.TestCase):
    def test_builds_target_and_control_groups(self):
        labels = {
            "a": label("a", "难度3档"),
            "b": label("b", "难度2档"),
            "c": label("c", "难度3档"),
            "excluded": label("excluded", "难度3档"),
        }
        predictions = {
            "a": prediction("a", "难度2档", 0),
            "b": prediction("b", "难度2档", 0),
            "c": prediction("c", "难度3档", 2),
            "excluded": prediction("excluded", "难度2档", 1),
        }

        rows = audit.build_rows(labels, predictions)

        self.assertEqual([row["question_id"] for row in rows], ["a", "b", "c"])
        self.assertEqual(
            [row["audit_group"] for row in rows],
            [
                "A_目标漏识别_3到2_非平凡为0",
                "B_正确2档_负对照",
                "C_正确3档_正对照",
            ],
        )
        self.assertEqual(rows[0]["manual_requires_stem_specific_relation"], "")

    def test_rejects_misaligned_question_ids(self):
        with self.assertRaisesRegex(ValueError, "question_id 不一致"):
            audit.build_rows(
                {"a": label("a", "难度3档")},
                {"b": prediction("b", "难度2档", 0)},
            )
