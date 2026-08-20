# -*- coding: utf-8 -*-
"""独立评测高中化学500题两阶段结果。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


LEVELS = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
LEVEL_INDEX = {level: index + 1 for index, level in enumerate(LEVELS)}


def quadratic_weighted_kappa(
    truth_values: list[str],
    prediction_values: list[str],
) -> float | None:
    """计算五档有序标签的 quadratic weighted kappa。"""
    if not truth_values or len(truth_values) != len(prediction_values):
        return None
    size = len(LEVELS)
    observed = [[0 for _ in range(size)] for _ in range(size)]
    truth_counts = [0 for _ in range(size)]
    prediction_counts = [0 for _ in range(size)]
    for truth, prediction in zip(truth_values, prediction_values):
        truth_index = LEVEL_INDEX[truth] - 1
        prediction_index = LEVEL_INDEX[prediction] - 1
        observed[truth_index][prediction_index] += 1
        truth_counts[truth_index] += 1
        prediction_counts[prediction_index] += 1

    observed_disagreement = 0.0
    expected_disagreement = 0.0
    denominator = float((size - 1) ** 2)
    sample_count = len(truth_values)
    for truth_index in range(size):
        for prediction_index in range(size):
            weight = (
                (truth_index - prediction_index) ** 2 / denominator
            )
            observed_disagreement += (
                weight * observed[truth_index][prediction_index]
            )
            expected_disagreement += (
                weight
                * truth_counts[truth_index]
                * prediction_counts[prediction_index]
                / sample_count
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return round(1.0 - observed_disagreement / expected_disagreement, 4)


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
    truth_values: list[str] = []
    prediction_values: list[str] = []
    valid = 0

    for qid in matched:
        truth = labels[qid].get("reviewed_difficulty_level")
        prediction = predictions[qid].get(prediction_field)
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
        "quadratic_weighted_kappa": quadratic_weighted_kappa(
            truth_values,
            prediction_values,
        ),
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


def review_diagnostics(
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """区分二阶段改档建议、乘数桶争议和顶层人工复核标记。"""
    records_with_verification = 0
    structural_revision_count = 0
    explicit_adjacent_adjustment_manual_count = 0
    multiplier_bucket_change_count = 0
    top_level_manual_review_count = 0
    supported_feature_correction_count = 0
    high_feature_set_changed_count = 0
    final_differs_from_step1_count = 0
    auto_adjustment_eligible_count = 0
    chemistry_58_boundary_promotion_candidate_count = 0
    auto_downgrade_two_to_one_blocked_count = 0
    reviewed_direction_distribution: Counter[str] = Counter()

    for row in predictions.values():
        top_level_manual_review_count += (
            row.get("needs_manual_review") is True
        )
        final_differs_from_step1_count += (
            row.get("final_difficulty_level")
            != row.get("difficulty_level_step1")
        )
        verification = row.get("verification")
        if not isinstance(verification, dict):
            continue
        records_with_verification += 1
        structural_revision_count += (
            verification.get("has_structural_revision") is True
        )
        explicit_adjacent_adjustment_manual_count += (
            verification.get("review_requires_manual") is True
        )
        multiplier_bucket_change_count += (
            verification.get("multiplier_reasonableness") == "不合理"
        )
        corrections = verification.get("supported_feature_corrections")
        if not isinstance(corrections, list):
            corrections = verification.get("feature_corrections_applied")
        if isinstance(corrections, list):
            supported_feature_correction_count += len(corrections)
        high_feature_set_changed_count += (
            verification.get("high_difficulty_features_changed") is True
        )
        auto_adjustment_eligible_count += (
            verification.get("auto_adjustment_eligible") is True
        )
        chemistry_58_boundary_promotion_candidate_count += (
            verification.get("chemistry_58_boundary_promotion_candidate")
            is True
        )
        auto_downgrade_two_to_one_blocked_count += (
            verification.get("auto_downgrade_two_to_one_blocked") is True
        )
        direction = verification.get("reviewed_direction")
        if not isinstance(direction, str) or not direction:
            direction = verification.get("review_action")
        if isinstance(direction, str) and direction:
            reviewed_direction_distribution[direction] += 1
    return {
        "records_with_verification": records_with_verification,
        "structural_revision_count": structural_revision_count,
        "explicit_adjacent_adjustment_manual_count": (
            explicit_adjacent_adjustment_manual_count
        ),
        "multiplier_bucket_change_count": multiplier_bucket_change_count,
        "top_level_manual_review_count": top_level_manual_review_count,
        "supported_feature_correction_count": (
            supported_feature_correction_count
        ),
        "high_feature_set_changed_count": high_feature_set_changed_count,
        "final_differs_from_step1_count": final_differs_from_step1_count,
        "auto_adjustment_eligible_count": auto_adjustment_eligible_count,
        "chemistry_58_boundary_promotion_candidate_count": (
            chemistry_58_boundary_promotion_candidate_count
        ),
        "auto_downgrade_two_to_one_blocked_count": (
            auto_downgrade_two_to_one_blocked_count
        ),
        "reviewed_direction_distribution": dict(
            reviewed_direction_distribution
        ),
    }


def accuracy_scale_diagnostics(
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """汇总高中化学第一阶段正确率与乘数审计信号。"""
    anchor_dist: Counter[str] = Counter()
    familiarity_dist: Counter[str] = Counter()
    burden_dist: Counter[str] = Counter()
    task_structure_dist: Counter[str] = Counter()
    score_dist: Counter[float] = Counter()
    records_with_stage1 = metadata_complete = 0
    anchor_inconsistent = low_structure_conflict = 0
    option_risk = error_risk_not_local = unsupported_count = 0
    complex_anchor_conflict = high_burden_score_conflict = 0
    heterogeneous_task_conflict = standard_model_inflation = 0
    threshold_inconsistent = threshold_evidence_incomplete = 0
    three_state_boundary_risk = 0
    multi_experiment_high_score_conflict = 0
    multiplier_enabled_count = 0
    multiplier_triggered_count = 0
    multiplier_final_level_guard_count = 0

    for row in predictions.values():
        stage1 = row.get("difficulty_rating_stage1")
        if not isinstance(stage1, dict):
            continue
        records_with_stage1 += 1
        anchor = stage1.get("accuracy_anchor")
        if isinstance(anchor, str) and anchor:
            anchor_dist[anchor] += 1
        for field, counter in (
            ("local_model_familiarity", familiarity_dist),
            ("whole_question_burden", burden_dist),
            ("task_completion_structure", task_structure_dist),
        ):
            value = stage1.get(field)
            if isinstance(value, str) and value:
                counter[value] += 1
        try:
            score_dist[float(stage1["original_predicted_accuracy"])] += 1
        except (KeyError, TypeError, ValueError):
            pass
        multiplier_enabled_count += (
            stage1.get("high_difficulty_multiplier_enabled") is True
        )
        multiplier_triggered_count += (
            stage1.get("multiplier_triggered") is True
        )
        multiplier_final_level_guard_count += (
            stage1.get("multiplier_final_level_guard_applied") is True
        )
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
        complex_anchor_conflict += (
            audit.get("complex_anchor_conflict") is True
        )
        high_burden_score_conflict += (
            audit.get("high_burden_score_conflict") is True
        )
        heterogeneous_task_conflict += (
            audit.get("heterogeneous_task_breadth_conflict") is True
        )
        standard_model_inflation += (
            audit.get("standard_model_score_inflation_risk") is True
        )
        threshold_inconsistent += (
            audit.get("threshold_review_consistent") is False
        )
        threshold_evidence_incomplete += (
            audit.get("threshold_evidence_complete") is False
        )
        three_state_boundary_risk += (
            audit.get("three_state_boundary_review_risk") is True
        )
        multi_experiment_high_score_conflict += (
            audit.get("multi_experiment_high_score_conflict") is True
        )

    score_count = sum(score_dist.values())
    most_common_score, most_common_count = (
        score_dist.most_common(1)[0] if score_dist else (None, 0)
    )
    top_5_count = sum(count for _, count in score_dist.most_common(5))
    return {
        "records_with_stage1": records_with_stage1,
        "metadata_complete_count": metadata_complete,
        "anchor_range_inconsistent_count": anchor_inconsistent,
        "low_structure_score_conflict_count": low_structure_conflict,
        "option_probability_multiplication_risk_count": option_risk,
        "error_risk_not_local_count": error_risk_not_local,
        "unsupported_boundary_evidence_count": unsupported_count,
        "complex_anchor_conflict_count": complex_anchor_conflict,
        "high_burden_score_conflict_count": high_burden_score_conflict,
        "heterogeneous_task_breadth_conflict_count": (
            heterogeneous_task_conflict
        ),
        "standard_model_score_inflation_risk_count": (
            standard_model_inflation
        ),
        "threshold_review_inconsistent_count": threshold_inconsistent,
        "threshold_evidence_incomplete_count": threshold_evidence_incomplete,
        "three_state_boundary_review_risk_count": three_state_boundary_risk,
        "multi_experiment_high_score_conflict_count": (
            multi_experiment_high_score_conflict
        ),
        "unique_original_accuracy_count": len(score_dist),
        "top_original_accuracy_values": [
            {"score": score, "count": count}
            for score, count in score_dist.most_common(15)
        ],
        "most_common_score": most_common_score,
        "most_common_score_count": most_common_count,
        "most_common_score_share": (
            round(most_common_count / score_count, 4)
            if score_count else None
        ),
        "top_5_score_share": (
            round(top_5_count / score_count, 4)
            if score_count else None
        ),
        "multiplier_enabled_count": multiplier_enabled_count,
        "multiplier_triggered_count": multiplier_triggered_count,
        "multiplier_final_level_guard_count": (
            multiplier_final_level_guard_count
        ),
        "anchor_distribution": dict(anchor_dist),
        "local_model_familiarity_distribution": dict(familiarity_dist),
        "whole_question_burden_distribution": dict(burden_dist),
        "task_completion_structure_distribution": dict(task_structure_dist),
    }


def build_report(
    labels: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "final": evaluate(labels, predictions, "final_difficulty_level"),
        "step1": evaluate(labels, predictions, "difficulty_level_step1"),
        "accuracy_scale_diagnostics": accuracy_scale_diagnostics(predictions),
        "review_diagnostics": review_diagnostics(predictions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = read_by_id(Path(args.labels))
    predictions = read_by_id(Path(args.predictions))
    report = build_report(labels, predictions)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
