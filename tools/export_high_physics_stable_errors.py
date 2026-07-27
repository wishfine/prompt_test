# -*- coding: utf-8 -*-
"""导出多次高中物理运行中预测同档且始终错误的稳定错题。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evaluate_high_physics_test500 import LEVEL_INDEX, read_by_id


def stable_error_rows(
    labels: dict[str, dict[str, Any]],
    runs: list[dict[str, dict[str, Any]]],
    *,
    prediction_field: str = "final_difficulty_level",
) -> list[dict[str, Any]]:
    if len(runs) < 2:
        raise ValueError("稳定错题至少需要两次运行")
    common_ids = set(labels)
    for run in runs:
        common_ids &= set(run)

    rows: list[dict[str, Any]] = []
    for question_id in sorted(common_ids):
        truth = labels[question_id].get("reviewed_difficulty_level")
        predictions = [
            run[question_id].get(prediction_field)
            for run in runs
        ]
        if (
            truth not in LEVEL_INDEX
            or any(value not in LEVEL_INDEX for value in predictions)
            or len(set(predictions)) != 1
            or predictions[0] == truth
        ):
            continue
        stable_prediction = predictions[0]
        source = runs[0][question_id]
        diagnostics = []
        for index, run in enumerate(runs, start=1):
            result = run[question_id]
            stage1 = result.get("difficulty_rating_stage1") or {}
            features = stage1.get("features") or {}
            diagnostics.append({
                "run": index,
                "prediction": result.get(prediction_field),
                "original_predicted_accuracy": stage1.get(
                    "original_predicted_accuracy"
                ),
                "predicted_accuracy": stage1.get("predicted_accuracy"),
                "step_count": features.get("step_count"),
                "whole_question_burden": stage1.get(
                    "whole_question_burden"
                ),
                "task_completion_structure": stage1.get(
                    "task_completion_structure"
                ),
                "model_explicitness": features.get("model_explicitness"),
                "state_count": features.get("state_count"),
                "process_state_relation": features.get(
                    "process_state_relation"
                ),
                "shared_model_across_subquestions": features.get(
                    "shared_model_across_subquestions"
                ),
                "high_difficulty_features": stage1.get(
                    "high_difficulty_features"
                ),
                "threshold_review": stage1.get("threshold_review"),
                "threshold_evidence": stage1.get("threshold_evidence"),
                "reason": stage1.get("reason"),
            })
        rows.append({
            "question_id": question_id,
            "reviewed_difficulty_level": truth,
            "stable_prediction": stable_prediction,
            "transition": (
                f"{LEVEL_INDEX[truth]}_to_"
                f"{LEVEL_INDEX[stable_prediction]}"
            ),
            "stem": source.get("stem"),
            "options": source.get("options"),
            "analysis": source.get("analysis"),
            "input_quality": source.get("input_quality"),
            "run_diagnostics": diagnostics,
        })
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prediction-field",
        default="final_difficulty_level",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    labels = read_by_id(Path(args.labels))
    runs = [read_by_id(Path(path)) for path in args.predictions]
    rows = stable_error_rows(
        labels,
        runs,
        prediction_field=args.prediction_field,
    )
    output_dir = Path(args.output_dir)
    write_jsonl(output_dir / "stable_errors_all.jsonl", rows)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["transition"], []).append(row)
    for transition, group_rows in grouped.items():
        write_jsonl(
            output_dir / f"error_{transition}.jsonl",
            group_rows,
        )

    summary = {
        "stable_error_count": len(rows),
        "transition_distribution": dict(
            sorted(Counter(row["transition"] for row in rows).items())
        ),
        "output_dir": str(output_dir),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
