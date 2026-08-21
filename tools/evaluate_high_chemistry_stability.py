# -*- coding: utf-8 -*-
"""高中化学 3 跑稳定性与特征/档位一致性分析工具。"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import tools.evaluate_high_chemistry_test500 as eval_tool

LEVELS = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]


def compute_stability_metrics(
    labels: dict[str, dict[str, Any]],
    runs: list[dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    if len(runs) < 2:
        raise ValueError("至少需要提供 2 次运行结果进行一致性分析")

    common_qids = sorted(labels.keys())
    for run in runs:
        common_qids = [qid for qid in common_qids if qid in run]

    total_q = len(common_qids)
    if total_q == 0:
        raise ValueError("没有共同包含的题目")

    # 1. Pairwise agreement
    pair_agreements = {}
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            same_step1 = sum(
                1 for q in common_qids
                if runs[i][q].get("difficulty_level_step1") == runs[j][q].get("difficulty_level_step1")
            )
            same_final = sum(
                1 for q in common_qids
                if runs[i][q].get("final_difficulty_level") == runs[j][q].get("final_difficulty_level")
            )
            pair_agreements[f"run{i+1}_run{j+1}_step1_agreement"] = round(same_step1 / total_q, 4)
            pair_agreements[f"run{i+1}_run{j+1}_final_agreement"] = round(same_final / total_q, 4)

    # 2. All runs agreement
    step1_all_same = sum(
        1 for q in common_qids
        if len({runs[i][q].get("difficulty_level_step1") for i in range(len(runs))}) == 1
    )
    final_all_same = sum(
        1 for q in common_qids
        if len({runs[i][q].get("final_difficulty_level") for i in range(len(runs))}) == 1
    )
    score_level_all_same = sum(
        1 for q in common_qids
        if len({
            runs[i][q].get("difficulty_rating_stage1", {}).get("difficulty_level_from_score")
            or eval_tool.MAP_TO_LEVEL(runs[i][q].get("difficulty_rating_stage1", {}).get("predicted_accuracy", 50))
            if hasattr(eval_tool, "MAP_TO_LEVEL") else runs[i][q].get("difficulty_level_step1")
            for i in range(len(runs))
        }) == 1
    )

    # 3. Raw accuracy range statistics
    ranges = []
    ge_5 = 0
    ge_10 = 0
    cross_88 = 0
    cross_85 = 0
    cross_58 = 0
    cross_38 = 0

    for q in common_qids:
        raw_scores = [
            float(runs[i][q].get("difficulty_rating_stage1", {}).get("model_predicted_accuracy_raw")
                  or runs[i][q].get("difficulty_rating_stage1", {}).get("original_predicted_accuracy", 0))
            for i in range(len(runs))
        ]
        r = max(raw_scores) - min(raw_scores)
        ranges.append(r)
        if r >= 5.0:
            ge_5 += 1
        if r >= 10.0:
            ge_10 += 1
        if any(s >= 88 for s in raw_scores) and any(s < 88 for s in raw_scores):
            cross_88 += 1
        if any(s >= 85 for s in raw_scores) and any(s < 85 for s in raw_scores):
            cross_85 += 1
        if any(s >= 58 for s in raw_scores) and any(s < 58 for s in raw_scores):
            cross_58 += 1
        if any(s >= 38 for s in raw_scores) and any(s < 38 for s in raw_scores):
            cross_38 += 1

    # 4. Feature consistency
    feature_keys = list(runs[0][common_qids[0]].get("difficulty_rating_stage1", {}).get("features", {}).keys())
    per_feature_all_same_rate = {}
    for fkey in feature_keys:
        same_count = 0
        for q in common_qids:
            fvals = [
                str(runs[i][q].get("difficulty_rating_stage1", {}).get("features", {}).get(fkey))
                for i in range(len(runs))
            ]
            if len(set(fvals)) == 1:
                same_count += 1
        per_feature_all_same_rate[fkey] = round(same_count / total_q, 4)

    # 5. Structural constraint actions
    override_counts = [
        sum(1 for q in common_qids if runs[i][q].get("difficulty_rating_stage1", {}).get("structural_constraint_applied") is True)
        for i in range(len(runs))
    ]
    manual_review_counts = [
        sum(1 for q in common_qids if runs[i][q].get("needs_manual_review") is True)
        for i in range(len(runs))
    ]

    return {
        "total_evaluated_questions": total_q,
        "pairwise_agreements": pair_agreements,
        "step1_all_runs_same_rate": round(step1_all_same / total_q, 4),
        "final_all_runs_same_rate": round(final_all_same / total_q, 4),
        "score_level_all_runs_same_rate": round(score_level_all_same / total_q, 4),
        "raw_accuracy_range_mean": round(statistics.mean(ranges), 2),
        "raw_accuracy_range_median": round(statistics.median(ranges), 2),
        "raw_accuracy_range_p90": round(statistics.quantiles(ranges, n=10)[8], 2) if len(ranges) >= 10 else None,
        "raw_accuracy_range_ge_5_count": ge_5,
        "raw_accuracy_range_ge_10_count": ge_10,
        "raw_score_cross_88_count": cross_88,
        "raw_score_cross_85_count": cross_85,
        "raw_score_cross_58_count": cross_58,
        "raw_score_cross_38_count": cross_38,
        "structural_constraint_override_counts_per_run": override_counts,
        "manual_review_counts_per_run": manual_review_counts,
        "per_feature_all_runs_same_rate": per_feature_all_same_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="3-Run Stability Analysis for High Chemistry")
    parser.add_argument("--labels", required=True, help="Labels JSONL path")
    parser.add_argument("--runs", nargs="+", required=True, help="Run prediction JSONL paths")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    labels = eval_tool.read_by_id(Path(args.labels))
    run_dicts = [eval_tool.read_by_id(Path(p)) for p in args.runs]

    metrics = compute_stability_metrics(labels, run_dicts)
    report_json = json.dumps(metrics, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(report_json + "\n", encoding="utf-8")

    print(report_json)


if __name__ == "__main__":
    main()
