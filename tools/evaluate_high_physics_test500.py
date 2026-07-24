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


def accuracy_scale_diagnostics(
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """汇总 V5 第一阶段正确率标尺的软审计信号。"""
    anchor_dist: Counter[str] = Counter()
    score_dist: Counter[float] = Counter()
    records_with_stage1 = metadata_complete = 0
    anchor_inconsistent = low_structure_conflict = 0
    option_risk = error_risk_not_local = unsupported_count = 0

    for row in predictions.values():
        stage1 = row.get("difficulty_rating_stage1")
        if not isinstance(stage1, dict):
            continue
        records_with_stage1 += 1
        anchor = stage1.get("accuracy_anchor")
        if isinstance(anchor, str) and anchor:
            anchor_dist[anchor] += 1
        try:
            score_dist[float(stage1["original_predicted_accuracy"])] += 1
        except (KeyError, TypeError, ValueError):
            pass
        audit = stage1.get("accuracy_scale_audit")
        if not isinstance(audit, dict):
            continue
        metadata_complete += audit.get("metadata_complete") is True
        anchor_inconsistent += audit.get("anchor_range_consistent") is False
        low_structure_conflict += (
            audit.get("low_structure_score_conflict") is True
        )
        option_risk += (
            audit.get("option_probability_multiplication_risk") is True
        )
        error_risk_not_local += (
            audit.get("error_risk_local_adjustment_confirmed") is False
        )
        unsupported = audit.get("unsupported_boundary_evidence")
        if isinstance(unsupported, list):
            unsupported_count += len(unsupported)

    return {
        "records_with_stage1": records_with_stage1,
        "metadata_complete_count": metadata_complete,
        "anchor_range_inconsistent_count": anchor_inconsistent,
        "low_structure_score_conflict_count": low_structure_conflict,
        "option_probability_multiplication_risk_count": option_risk,
        "error_risk_not_local_count": error_risk_not_local,
        "unsupported_boundary_evidence_count": unsupported_count,
        "unique_original_accuracy_count": len(score_dist),
        "top_original_accuracy_values": [
            {"score": score, "count": count}
            for score, count in score_dist.most_common(15)
        ],
        "anchor_distribution": dict(anchor_dist),
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
        "accuracy_scale_diagnostics": accuracy_scale_diagnostics(
            predictions
        ),
    }
    print(json.dumps(reports, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
