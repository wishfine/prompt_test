#!/usr/bin/env python3
"""按高中物理口径评测高中化学流程档位与独立 AI 盲标档位。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


LEVELS = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
LEVEL_INDEX = {level: index + 1 for index, level in enumerate(LEVELS)}


def quadratic_weighted_kappa(truth_values: list[str], prediction_values: list[str]) -> float | None:
    if not truth_values or len(truth_values) != len(prediction_values):
        return None
    size = len(LEVELS)
    observed = [[0 for _ in range(size)] for _ in range(size)]
    truth_counts = [0 for _ in range(size)]
    prediction_counts = [0 for _ in range(size)]
    for truth, prediction in zip(truth_values, prediction_values):
        truth_index, prediction_index = LEVEL_INDEX[truth] - 1, LEVEL_INDEX[prediction] - 1
        observed[truth_index][prediction_index] += 1
        truth_counts[truth_index] += 1
        prediction_counts[prediction_index] += 1
    observed_disagreement = expected_disagreement = 0.0
    denominator = float((size - 1) ** 2)
    sample_count = len(truth_values)
    for truth_index in range(size):
        for prediction_index in range(size):
            weight = (truth_index - prediction_index) ** 2 / denominator
            observed_disagreement += weight * observed[truth_index][prediction_index]
            expected_disagreement += (
                weight * truth_counts[truth_index] * prediction_counts[prediction_index] / sample_count
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return round(1.0 - observed_disagreement / expected_disagreement, 4)


def read_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path} 第 {line_number} 行缺少 question_id")
            if question_id in rows:
                raise ValueError(f"{path} question_id 重复：{question_id}")
            rows[question_id] = row
    return rows


def label_level(row: dict[str, Any]) -> str | None:
    direct = row.get("reviewed_difficulty_level")
    if direct in LEVEL_INDEX:
        return direct
    try:
        return f"难度{int(row['standard_level'])}档"
    except (KeyError, TypeError, ValueError):
        return None


def evaluate(labels: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]], prediction_field: str) -> dict[str, Any]:
    matched = sorted(labels.keys() & predictions.keys())
    confusion = {truth: {prediction: 0 for prediction in LEVELS} for truth in LEVELS}
    exact = within_one = severe = over = under = absolute_error = valid = 0
    truth_dist: Counter[str] = Counter()
    pred_dist: Counter[str] = Counter()
    truth_values: list[str] = []
    prediction_values: list[str] = []
    for question_id in matched:
        truth = label_level(labels[question_id])
        prediction = predictions[question_id].get(prediction_field)
        if truth not in LEVEL_INDEX or prediction not in LEVEL_INDEX:
            continue
        valid += 1
        truth_values.append(truth)
        prediction_values.append(prediction)
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
    per_level_accuracy = {
        level: round(confusion[level][level] / sum(confusion[level].values()), 4)
        if sum(confusion[level].values()) else None
        for level in LEVELS
    }
    return {
        "prediction_field": prediction_field,
        "label_count": len(labels),
        "prediction_count": len(predictions),
        "matched_ids": len(matched),
        "evaluated": valid,
        "exact_match_rate": round(exact / valid, 4) if valid else None,
        "within_one_level_rate": round(within_one / valid, 4) if valid else None,
        "mae": round(absolute_error / valid, 4) if valid else None,
        "quadratic_weighted_kappa": quadratic_weighted_kappa(truth_values, prediction_values),
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


def review_diagnostics(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    records_with_verification = structural_revision_count = manual_review_count = 0
    multiplier_bucket_change_count = final_differs_from_step1_count = 0
    direction_distribution: Counter[str] = Counter()
    for row in predictions.values():
        final_differs_from_step1_count += row.get("final_difficulty_level") != row.get("difficulty_level_step1")
        manual_review_count += row.get("needs_manual_review") is True
        verification = row.get("verification")
        if not isinstance(verification, dict):
            continue
        records_with_verification += 1
        structural_revision_count += verification.get("has_structural_revision") is True
        multiplier_bucket_change_count += verification.get("multiplier_reasonableness") == "不合理"
        direction = verification.get("reviewed_direction")
        if isinstance(direction, str) and direction:
            direction_distribution[direction] += 1
    return {
        "records_with_verification": records_with_verification,
        "structural_revision_count": structural_revision_count,
        "multiplier_bucket_change_count": multiplier_bucket_change_count,
        "top_level_manual_review_count": manual_review_count,
        "final_differs_from_step1_count": final_differs_from_step1_count,
        "reviewed_direction_distribution": dict(direction_distribution),
    }


def accuracy_scale_diagnostics(predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    familiarity: Counter[str] = Counter()
    burden: Counter[str] = Counter()
    task_structure: Counter[str] = Counter()
    score_dist: Counter[float] = Counter()
    records_with_stage1 = threshold_inconsistent = low_structure_conflict = 0
    high_burden_conflict = three_state_risk = multi_reaction_risk = 0
    for row in predictions.values():
        stage1 = row.get("difficulty_rating_stage1")
        if not isinstance(stage1, dict):
            continue
        records_with_stage1 += 1
        for field, counter in (("local_model_familiarity", familiarity), ("whole_question_burden", burden), ("task_completion_structure", task_structure)):
            value = stage1.get(field)
            if isinstance(value, str) and value:
                counter[value] += 1
        try:
            score_dist[float(stage1["original_predicted_accuracy"])] += 1
        except (KeyError, TypeError, ValueError):
            pass
        audit = stage1.get("accuracy_scale_audit")
        if not isinstance(audit, dict):
            continue
        threshold_inconsistent += audit.get("threshold_review_consistent") is False
        low_structure_conflict += audit.get("low_structure_score_conflict") is True
        high_burden_conflict += audit.get("high_burden_score_conflict") is True
        three_state_risk += audit.get("three_state_boundary_review_risk") is True
        multi_reaction_risk += audit.get("multi_reaction_boundary_review_risk") is True
    return {
        "records_with_stage1": records_with_stage1,
        "threshold_review_inconsistent_count": threshold_inconsistent,
        "low_structure_score_conflict_count": low_structure_conflict,
        "high_burden_score_conflict_count": high_burden_conflict,
        "three_state_boundary_review_risk_count": three_state_risk,
        "multi_reaction_boundary_review_risk_count": multi_reaction_risk,
        "unique_original_accuracy_count": len(score_dist),
        "top_original_accuracy_values": [{"score": score, "count": count} for score, count in score_dist.most_common(15)],
        "local_model_familiarity_distribution": dict(familiarity),
        "whole_question_burden_distribution": dict(burden),
        "task_completion_structure_distribution": dict(task_structure),
    }


def mismatch_rows(labels: dict[str, dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for question_id in sorted(labels.keys() & predictions.keys()):
        truth = label_level(labels[question_id])
        prediction = predictions[question_id].get("final_difficulty_level")
        if truth not in LEVEL_INDEX or prediction not in LEVEL_INDEX or truth == prediction:
            continue
        row = predictions[question_id]
        rows.append({
            "question_id": question_id,
            "reference_level": truth,
            "final_level": prediction,
            "step1_level": row.get("difficulty_level_step1", ""),
            "gap": LEVEL_INDEX[prediction] - LEVEL_INDEX[truth],
            "reference_confidence": labels[question_id].get("confidence", ""),
            "reference_reason": labels[question_id].get("reason", ""),
            "pipeline_reason": (row.get("difficulty_rating_stage1") or {}).get("reason", ""),
            "stem": row.get("stem", ""),
        })
    return rows


def write_mismatches(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "question_id", "reference_level", "final_level", "step1_level", "gap",
            "reference_confidence", "reference_reason", "pipeline_reason", "stem",
        ])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--mismatches", required=True)
    args = parser.parse_args()
    labels = read_by_id(Path(args.labels))
    predictions = read_by_id(Path(args.predictions))
    report = {
        "final": evaluate(labels, predictions, "final_difficulty_level"),
        "step1": evaluate(labels, predictions, "difficulty_level_step1"),
        "accuracy_scale_diagnostics": accuracy_scale_diagnostics(predictions),
        "review_diagnostics": review_diagnostics(predictions),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    mismatch_path = Path(args.mismatches)
    mismatches = mismatch_rows(labels, predictions)
    write_mismatches(mismatch_path, mismatches)
    final = report["final"]
    print(
        "评测完成：\n"
        f"  ACC：{final['exact_match_rate']}\n"
        f"  ±1档：{final['within_one_level_rate']}\n"
        f"  MAE：{final['mae']}\n"
        f"  报告：{report_path}\n"
        f"  错配样本：{mismatch_path}"
    )


if __name__ == "__main__":
    main()
