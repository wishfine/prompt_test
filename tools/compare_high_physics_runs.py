# -*- coding: utf-8 -*-
"""比较多次高中物理运行的指标均值、波动和逐边界翻转率。"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_high_physics_test500 import (
    LEVEL_INDEX,
    evaluate,
    read_by_id,
)


def stability_diagnostics(
    labels: dict[str, dict[str, Any]],
    runs: list[dict[str, dict[str, Any]]],
    *,
    prediction_field: str = "final_difficulty_level",
) -> dict[str, Any]:
    """在所有运行共同成功的题目上计算预测一致性和边界翻转。"""
    if not runs:
        raise ValueError("至少需要一次预测结果")
    common_ids = set(labels)
    for run in runs:
        common_ids &= set(run)

    valid_ids = sorted(
        question_id
        for question_id in common_ids
        if labels[question_id].get("reviewed_difficulty_level")
        in LEVEL_INDEX
        and all(
            run[question_id].get(prediction_field) in LEVEL_INDEX
            for run in runs
        )
    )
    all_equal = 0
    span_distribution: Counter[int] = Counter()
    boundary_flips = {
        "88_boundary_1_vs_2plus": 0,
        "85_boundary_1to2_vs_3plus": 0,
        "58_boundary_1to3_vs_4plus": 0,
        "38_boundary_1to4_vs_5": 0,
    }
    thresholds = [
        ("88_boundary_1_vs_2plus", 1),
        ("85_boundary_1to2_vs_3plus", 2),
        ("58_boundary_1to3_vs_4plus", 3),
        ("38_boundary_1to4_vs_5", 4),
    ]

    for question_id in valid_ids:
        values = [
            LEVEL_INDEX[run[question_id][prediction_field]]
            for run in runs
        ]
        if len(set(values)) == 1:
            all_equal += 1
        span_distribution[max(values) - min(values)] += 1
        for name, threshold in thresholds:
            sides = {value <= threshold for value in values}
            if len(sides) > 1:
                boundary_flips[name] += 1

    count = len(valid_ids)
    return {
        "common_evaluated": count,
        "all_runs_same_prediction_count": all_equal,
        "all_runs_same_prediction_rate": (
            round(all_equal / count, 4) if count else None
        ),
        "prediction_span_distribution": {
            str(span): value
            for span, value in sorted(span_distribution.items())
        },
        "boundary_flip_counts": boundary_flips,
        "boundary_flip_rates": {
            name: round(value / count, 4) if count else None
            for name, value in boundary_flips.items()
        },
    }


def summarize_metrics(
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """汇总多次运行核心指标的均值和总体标准差。"""
    fields = [
        "exact_match_rate",
        "within_one_level_rate",
        "mae",
        "quadratic_weighted_kappa",
        "severe_deviation_count",
    ]
    summary: dict[str, dict[str, float | None]] = {}
    for field in fields:
        values = [
            float(report[field])
            for report in reports
            if report.get(field) is not None
        ]
        summary[field] = {
            "mean": round(statistics.fmean(values), 4) if values else None,
            "population_std": (
                round(statistics.pstdev(values), 4) if values else None
            ),
            "min": round(min(values), 4) if values else None,
            "max": round(max(values), 4) if values else None,
        }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument(
        "--prediction-field",
        default="final_difficulty_level",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = read_by_id(Path(args.labels))
    run_paths = [Path(value) for value in args.predictions]
    runs = [read_by_id(path) for path in run_paths]
    reports = [
        evaluate(labels, run, args.prediction_field)
        for run in runs
    ]
    output = {
        "prediction_field": args.prediction_field,
        "runs": [
            {
                "path": str(path),
                **report,
            }
            for path, report in zip(run_paths, reports)
        ],
        "summary": summarize_metrics(reports),
        "stability": stability_diagnostics(
            labels,
            runs,
            prediction_field=args.prediction_field,
        ),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
