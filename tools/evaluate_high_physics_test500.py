# -*- coding: utf-8 -*-
"""评测高中物理两阶段Pipeline输出与GPT-5.6复核标签。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LEVELS = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
LEVEL_INDEX = {level: index + 1 for index, level in enumerate(LEVELS)}


def read_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("question_id") or "")
            if not qid:
                raise ValueError(f"{path} 第 {line_number} 行缺少 question_id")
            if qid in rows:
                raise ValueError(f"{path} question_id 重复：{qid}")
            rows[qid] = row
    return rows


def evaluate(
    labels: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    prediction_field: str,
) -> dict[str, Any]:
    matched = sorted(labels.keys() & predictions.keys())
    confusion = {
        truth: {prediction: 0 for prediction in LEVELS}
        for truth in LEVELS
    }
    exact = within_one = severe = over = under = 0
    absolute_error = 0
    truth_dist: Counter[str] = Counter()
    pred_dist: Counter[str] = Counter()
    valid = 0

    for qid in matched:
        truth = labels[qid].get("reviewed_difficulty_level")
        prediction = predictions[qid].get(prediction_field)
        if truth not in LEVEL_INDEX or prediction not in LEVEL_INDEX:
            continue
        valid += 1
        truth_dist[truth] += 1
        pred_dist[prediction] += 1
        confusion[truth][prediction] += 1
        gap = LEVEL_INDEX[prediction] - LEVEL_INDEX[truth]
        absolute_error += abs(gap)
        exact += gap == 0
        within_one += abs(gap) <= 1
        severe += abs(gap) >= 2
        over += gap > 0
        under += gap < 0

    per_level_accuracy = {}
    for level in LEVELS:
        total = sum(confusion[level].values())
        per_level_accuracy[level] = (
            round(confusion[level][level] / total, 4) if total else None
        )
    return {
        "prediction_field": prediction_field,
        "label_count": len(labels),
        "prediction_count": len(predictions),
        "matched_ids": len(matched),
        "evaluated": valid,
        "exact_match_rate": round(exact / valid, 4) if valid else None,
        "within_one_level_rate": round(within_one / valid, 4) if valid else None,
        "mae": round(absolute_error / valid, 4) if valid else None,
        "severe_deviation_count": severe,
        "over_predicted": over,
        "under_predicted": under,
        "label_distribution": dict(truth_dist),
        "prediction_distribution": dict(pred_dist),
        "per_level_accuracy": per_level_accuracy,
        "confusion_matrix": confusion,
        "missing_prediction_ids": len(labels.keys() - predictions.keys()),
        "unexpected_prediction_ids": len(predictions.keys() - labels.keys()),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = read_by_id(Path(args.labels))
    predictions = read_by_id(Path(args.predictions))
    reports = {
        "final": evaluate(labels, predictions, "final_difficulty_level"),
        "step1": evaluate(labels, predictions, "difficulty_level_step1"),
    }
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
