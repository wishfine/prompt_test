#!/usr/bin/env python3
"""Evaluate chemistry difficulty predictions and monitor level collapse."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LEVEL_NAME_TO_NUMBER = {
    "送分题": 1,
    "基础题": 2,
    "中等题": 3,
    "拔高题": 4,
    "压轴题": 5,
}
LEVEL_NUMBER_TO_NAME = {
    value: key for key, value in LEVEL_NAME_TO_NUMBER.items()
}
LEVEL_NAMES = list(LEVEL_NAME_TO_NUMBER)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按 question_id 评测化学难度预测并监控档位分布",
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--errors")
    parser.add_argument(
        "--level-source",
        choices=[
            "final",
            "pre-postprocess",
            "postprocess-candidate",
            "final-boundary-guard-candidate",
        ],
        default="final",
    )
    parser.add_argument("--report", required=True)
    parser.add_argument("--mismatches", required=True)
    return parser.parse_args()


def jsonl_items(
    path: Path,
) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path} 第 {line_number} 行不是合法 JSON: {exc}"
                ) from exc
            if isinstance(item, dict):
                yield line_number, item


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            question_id = str(row.get("question_id", "")).strip()
            if not question_id:
                continue
            level = int(row["standard_level"])
            if level not in LEVEL_NUMBER_TO_NAME:
                raise ValueError(
                    f"ID={question_id} 的标准等级非法: {level}"
                )
            if question_id in labels:
                raise ValueError(
                    f"标签中存在重复 question_id: {question_id}"
                )
            labels[question_id] = {
                "standard_stars": row.get("standard_stars", ""),
                "standard_level": level,
                "standard_level_name": (
                    row.get("standard_level_name")
                    or LEVEL_NUMBER_TO_NAME[level]
                ),
                "reason": (
                    row.get("reason")
                    or row.get("teacher_reason")
                    or row.get("review_reason")
                    or ""
                ),
            }
    return labels


def extract_prediction(
    item: dict[str, Any],
    level_source: str,
) -> tuple[str | None, int | None]:
    rating = item.get("difficulty_rating")
    level_name = None
    if isinstance(rating, dict):
        if level_source == "pre-postprocess":
            level_name = rating.get("postprocess_original_level")
        elif level_source == "postprocess-candidate":
            level_name = rating.get("postprocess_candidate_level")
        elif level_source == "final-boundary-guard-candidate":
            level_name = rating.get(
                "final_boundary_guard_candidate_level"
            )
        level_name = level_name or rating.get("difficulty_level")
    if level_name is None:
        level_name = item.get("difficulty_level")
    if level_name is not None:
        level_name = str(level_name).strip()
    return (
        level_name,
        LEVEL_NAME_TO_NUMBER.get(level_name) if level_name else None,
    )


def load_predictions(
    path: Path,
    level_source: str,
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    predictions: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for line_number, item in jsonl_items(path):
        question_id = str(item.get("question_id", "")).strip()
        if not question_id:
            raise ValueError(
                f"{path} 第 {line_number} 行缺少 question_id"
            )
        if question_id in predictions:
            duplicates.append(question_id)
            continue
        level_name, level_number = extract_prediction(
            item,
            level_source,
        )
        rating = item.get("difficulty_rating")
        if not isinstance(rating, dict):
            rating = {}
        predictions[question_id] = {
            "predicted_level_name": level_name,
            "predicted_level": level_number,
            "stem": str(item.get("stem", "") or ""),
            "postprocess_original_level": rating.get(
                "postprocess_original_level"
            ),
            "postprocess_candidate_level": rating.get(
                "postprocess_candidate_level"
            ),
            "final_boundary_guard_candidate_level": rating.get(
                "final_boundary_guard_candidate_level"
            ),
            "postprocess_trace": rating.get("postprocess_trace", []),
            "postprocess_candidate_actions": rating.get(
                "postprocess_candidate_actions",
                [],
            ),
        }
    return predictions, sorted(set(duplicates))


def load_error_ids(
    path: Path | None,
) -> tuple[set[str], dict[str, str]]:
    if path is None or not path.exists():
        return set(), {}
    ids: set[str] = set()
    messages: dict[str, str] = {}
    for _, item in jsonl_items(path):
        question_id = str(item.get("question_id", "")).strip()
        if question_id:
            ids.add(question_id)
            messages[question_id] = str(
                item.get("rating_error", "")
            )
    return ids, messages


def safe_rate(
    numerator: int,
    denominator: int,
) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def evaluate_predictions(
    labels: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
    *,
    error_ids: set[str],
    error_messages: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    attempted_ids = set(predictions) | error_ids
    evaluable_ids = sorted(attempted_ids & set(labels))
    legal_ids = [
        question_id
        for question_id in evaluable_ids
        if predictions.get(question_id, {}).get("predicted_level")
        in LEVEL_NUMBER_TO_NAME
    ]

    confusion: Counter[tuple[int, int]] = Counter()
    label_distribution: Counter[str] = Counter()
    prediction_distribution: Counter[str] = Counter()
    exact = 0
    within_one = 0
    severe = 0
    absolute_error_sum = 0
    mismatch_rows: list[dict[str, Any]] = []

    for question_id in evaluable_ids:
        label = labels[question_id]
        prediction = predictions.get(question_id, {})
        actual = int(label["standard_level"])
        actual_name = LEVEL_NUMBER_TO_NAME[actual]
        predicted = prediction.get("predicted_level")
        label_distribution[actual_name] += 1
        if predicted in LEVEL_NUMBER_TO_NAME:
            predicted_name = LEVEL_NUMBER_TO_NAME[predicted]
            prediction_distribution[predicted_name] += 1
            difference = abs(predicted - actual)
            confusion[(actual, predicted)] += 1
            absolute_error_sum += difference
            exact += difference == 0
            within_one += difference <= 1
            severe += difference >= 2
            status = "correct" if difference == 0 else "mismatch"
        else:
            difference = None
            status = (
                "request_error"
                if question_id in error_ids
                else "invalid_prediction"
            )

        if status != "correct":
            mismatch_rows.append(
                {
                    "question_id": question_id,
                    "status": status,
                    "standard_stars": label.get(
                        "standard_stars",
                        "",
                    ),
                    "standard_level": actual,
                    "standard_level_name": actual_name,
                    "predicted_level": (
                        predicted if predicted is not None else ""
                    ),
                    "predicted_level_name": prediction.get(
                        "predicted_level_name"
                    )
                    or "",
                    "absolute_error": (
                        difference if difference is not None else ""
                    ),
                    "standard_reason": label.get("reason", ""),
                    "stem": prediction.get("stem", ""),
                    "postprocess_original_level": prediction.get(
                        "postprocess_original_level"
                    )
                    or "",
                    "postprocess_candidate_level": prediction.get(
                        "postprocess_candidate_level"
                    )
                    or "",
                    "postprocess_trace": json.dumps(
                        prediction.get("postprocess_trace", []),
                        ensure_ascii=False,
                    ),
                    "postprocess_candidate_actions": json.dumps(
                        prediction.get(
                            "postprocess_candidate_actions",
                            [],
                        ),
                        ensure_ascii=False,
                    ),
                    "rating_error": error_messages.get(
                        question_id,
                        "",
                    ),
                }
            )

    per_level_metrics: dict[str, dict[str, Any]] = {}
    for level_name, level in LEVEL_NAME_TO_NUMBER.items():
        tp = confusion[(level, level)]
        support = sum(
            confusion[(level, predicted)]
            for predicted in range(1, 6)
        )
        predicted_count = sum(
            confusion[(actual, level)]
            for actual in range(1, 6)
        )
        precision = safe_rate(tp, predicted_count)
        recall = safe_rate(tp, support)
        f1 = (
            round(
                2 * precision * recall / (precision + recall),
                6,
            )
            if precision is not None
            and recall is not None
            and precision + recall
            else None
        )
        per_level_metrics[level_name] = {
            "support": support,
            "predicted": predicted_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    label_dist = {
        level: label_distribution.get(level, 0)
        for level in LEVEL_NAMES
    }
    prediction_dist = {
        level: prediction_distribution.get(level, 0)
        for level in LEVEL_NAMES
    }
    distribution_l1 = sum(
        abs(label_dist[level] - prediction_dist[level])
        for level in LEVEL_NAMES
    )
    legal_count = len(legal_ids)
    attempted_count = len(evaluable_ids)
    distribution_warnings = [
        {
            "type": "prediction_count_below_half",
            "level": level,
            "label_count": label_dist[level],
            "prediction_count": prediction_dist[level],
        }
        for level in LEVEL_NAMES
        if label_dist[level] > 0
        and prediction_dist[level] * 2 < label_dist[level]
    ]

    top_two_actual = sum(
        label_dist[level] for level in ("拔高题", "压轴题")
    )
    top_two_correct = sum(
        confusion[(actual, predicted)]
        for actual in (4, 5)
        for predicted in (4, 5)
    )
    report = {
        "evaluable_attempted_ids": attempted_count,
        "legal_prediction_ids": legal_count,
        "exact_matches": exact,
        "accuracy_on_legal_predictions": safe_rate(
            exact,
            legal_count,
        ),
        "strict_accuracy": safe_rate(exact, attempted_count),
        "coverage_within_attempted": safe_rate(
            legal_count,
            attempted_count,
        ),
        "within_one_level_rate": safe_rate(
            within_one,
            legal_count,
        ),
        "mae": (
            round(absolute_error_sum / legal_count, 6)
            if legal_count
            else None
        ),
        "severe_deviation_count": severe,
        "label_distribution": label_dist,
        "prediction_distribution": prediction_dist,
        "distribution_l1_count": distribution_l1,
        "distribution_total_variation": (
            round(distribution_l1 / (2 * legal_count), 6)
            if legal_count
            else None
        ),
        "distribution_warnings": distribution_warnings,
        "top_two_level_recall": safe_rate(
            top_two_correct,
            top_two_actual,
        ),
        "per_level_metrics": per_level_metrics,
        "confusion_matrix": {
            str(actual): {
                str(predicted): confusion[(actual, predicted)]
                for predicted in range(1, 6)
            }
            for actual in range(1, 6)
        },
        "mismatch_count": len(mismatch_rows),
    }
    return report, mismatch_rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "status",
        "standard_stars",
        "standard_level",
        "standard_level_name",
        "predicted_level",
        "predicted_level_name",
        "absolute_error",
        "standard_reason",
        "stem",
        "postprocess_original_level",
        "postprocess_candidate_level",
        "postprocess_trace",
        "postprocess_candidate_actions",
        "rating_error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels).expanduser().resolve()
    predictions_path = Path(args.predictions).expanduser().resolve()
    errors_path = (
        Path(args.errors).expanduser().resolve()
        if args.errors
        else None
    )
    report_path = Path(args.report).expanduser().resolve()
    mismatches_path = Path(args.mismatches).expanduser().resolve()

    if not predictions_path.exists():
        raise FileNotFoundError(
            f"预测文件不存在: {predictions_path}"
        )
    labels = load_labels(labels_path)
    predictions, duplicate_ids = load_predictions(
        predictions_path,
        args.level_source,
    )
    error_ids, error_messages = load_error_ids(errors_path)
    report, mismatch_rows = evaluate_predictions(
        labels,
        predictions,
        error_ids=error_ids,
        error_messages=error_messages,
    )
    report.update(
        {
            "labels_file": str(labels_path),
            "predictions_file": str(predictions_path),
            "errors_file": (
                str(errors_path) if errors_path else None
            ),
            "level_source": args.level_source,
            "clean_label_ids": len(labels),
            "prediction_unique_ids": len(predictions),
            "error_unique_ids": len(error_ids),
            "duplicate_prediction_ids": duplicate_ids,
            "prediction_ids_without_clean_label": sorted(
                set(predictions) - set(labels)
            ),
        }
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(mismatches_path, mismatch_rows)

    print(f"本次模型输出唯一 ID: {len(predictions)}")
    print(
        "其中有干净标准标签: "
        f"{report['evaluable_attempted_ids']}"
    )
    print(f"合法难度预测: {report['legal_prediction_ids']}")
    print(f"完全一致: {report['exact_matches']}")
    print(
        "Accuracy（合法预测）: "
        f"{report['accuracy_on_legal_predictions']}"
    )
    print(
        "Strict Accuracy（失败也计错）: "
        f"{report['strict_accuracy']}"
    )
    print(
        "相差不超过一档: "
        f"{report['within_one_level_rate']}"
    )
    print(f"MAE: {report['mae']}")
    print(f"严重偏差: {report['severe_deviation_count']}")
    print(f"标签分布: {report['label_distribution']}")
    print(f"预测分布: {report['prediction_distribution']}")
    print(
        "分布L1/总变差: "
        f"{report['distribution_l1_count']}/"
        f"{report['distribution_total_variation']}"
    )
    if report["distribution_warnings"]:
        print(
            "分布报警: "
            + json.dumps(
                report["distribution_warnings"],
                ensure_ascii=False,
            )
        )
    print(f"报告: {report_path}")
    print(f"错题: {mismatches_path}")


if __name__ == "__main__":
    main()
