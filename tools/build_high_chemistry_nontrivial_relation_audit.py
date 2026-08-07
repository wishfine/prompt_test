#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""导出高中化学“题干特有关系建立”离线人工审计表。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


LEVELS = {f"难度{value}档" for value in range(1, 6)}


def read_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in rows:
                raise ValueError(f"{path}:{line_number} question_id 重复：{question_id}")
            rows[question_id] = row
    return rows


def label_level(row: dict[str, Any]) -> str:
    level = row.get("reviewed_difficulty_level")
    if level not in LEVELS:
        raise ValueError(f"标签缺少有效 reviewed_difficulty_level：{row.get('question_id')}")
    return level


def prediction_level(row: dict[str, Any]) -> str:
    level = row.get("final_difficulty_level")
    if level not in LEVELS:
        raise ValueError(f"预测缺少有效 final_difficulty_level：{row.get('question_id')}")
    return level


def audit_group(label: str, prediction: str, nontrivial_count: int) -> str | None:
    if label == "难度3档" and prediction == "难度2档" and nontrivial_count == 0:
        return "A_目标漏识别_3到2_非平凡为0"
    if label == "难度2档" and prediction == "难度2档":
        return "B_正确2档_负对照"
    if label == "难度3档" and prediction == "难度3档":
        return "C_正确3档_正对照"
    return None


def build_rows(
    labels: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    if labels.keys() != predictions.keys():
        missing = sorted(labels.keys() - predictions.keys())
        unexpected = sorted(predictions.keys() - labels.keys())
        raise ValueError(
            "标签与预测 question_id 不一致："
            f"缺少预测 {len(missing)} 道，额外预测 {len(unexpected)} 道"
        )

    rows: list[dict[str, Any]] = []
    for question_id in sorted(labels):
        label = label_level(labels[question_id])
        prediction = predictions[question_id]
        predicted_level = prediction_level(prediction)
        stage1 = prediction.get("difficulty_rating_stage1", {})
        features = stage1.get("features", {})
        nontrivial_count = int(stage1.get("nontrivial_task_count", 0))
        group = audit_group(label, predicted_level, nontrivial_count)
        if group is None:
            continue

        source = labels[question_id]
        rows.append(
            {
                "audit_group": group,
                "question_id": question_id,
                "reference_level": label,
                "predicted_level_v7": predicted_level,
                "reference_reason": source.get("reason", ""),
                "stem": prediction.get("stem", ""),
                "options": prediction.get("options", ""),
                "analysis": prediction.get("analysis", ""),
                "structure_type": prediction.get("structure_type", ""),
                "task_units": json.dumps(stage1.get("task_units", []), ensure_ascii=False),
                "task_units_relation": stage1.get("task_units_relation", ""),
                "essential_task_count": stage1.get("essential_task_count", ""),
                "nontrivial_task_count": nontrivial_count,
                "substantive_step_count": stage1.get("substantive_step_count", ""),
                "whole_question_burden": features.get("whole_question_burden", ""),
                "reasoning_chain": features.get("reasoning_chain", ""),
                "model_relation": features.get("model_relation", ""),
                "constraint_structure": features.get("constraint_structure", ""),
                "model_conversion_required": stage1.get("model_conversion_required", ""),
                "intermediate_result_reuse": stage1.get("intermediate_result_reuse", ""),
                "manual_requires_stem_specific_relation": "",
                "manual_relation_type": "",
                "manual_relation_description": "",
                "manual_parallel_textbook_rule": "",
                "manual_basis": "",
                "manual_reviewer": "",
            }
        )
    return rows


def write_rows(rows: list[dict[str, Any]], output: Path) -> dict[str, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else ["audit_group", "question_id"]
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return dict(Counter(row["audit_group"] for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows = build_rows(read_by_id(args.labels), read_by_id(args.predictions))
    counts = write_rows(rows, args.output)
    print(json.dumps({"output": str(args.output), "row_count": len(rows), "groups": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
