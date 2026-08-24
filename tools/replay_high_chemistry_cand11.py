# -*- coding: utf-8 -*-
"""高中化学 Candidate 11 离线 Frozen Replay 验证脚本。

将现有 Candidate 10 的三跑原始输出输入 Candidate 11 Core 重新计算 Step 1 和 Final 档位，
输出：
  1. 全量 500 题各 Run 的 Step1 / Final Accuracy、QWK、各档位 Recall/Precision
  2. 三跑档位一致性（3-run agreement rate）与稳定题准确率
  3. 309 题未审计稳定集与 191/180 题人工复核集在各版本间的变动对比
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import high_chemistry_pipeline_core as core

LEVELS = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
LEVEL_INDEX = {level: index + 1 for index, level in enumerate(LEVELS)}


def quadratic_weighted_kappa(
    truth_values: list[str],
    prediction_values: list[str],
) -> float | None:
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
            weight = (truth_index - prediction_index) ** 2 / denominator
            observed_disagreement += weight * observed[truth_index][prediction_index]
            expected_disagreement += (
                weight * truth_counts[truth_index] * prediction_counts[prediction_index] / sample_count
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return round(1.0 - observed_disagreement / expected_disagreement, 4)


def load_labels(path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        lvl = (
            row.get("revalidated_difficulty_level")
            or row.get("reviewed_difficulty_level")
            or row.get("manual_difficulty_level")
        )
        labels[str(row["question_id"])] = lvl
    return labels


def replay_single_run(run_path: Path, labels: dict[str, str]) -> list[dict[str, Any]]:
    rows = [json.loads(l) for l in run_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    results = []
    for item in rows:
        qid = str(item["question_id"])
        target = labels.get(qid)
        s1 = item.get("difficulty_rating_stage1") or {}
        raw_features = s1.get("features_model_raw") or s1.get("features") or {}
        raw_accuracy = s1.get("model_predicted_accuracy_raw", s1.get("original_predicted_accuracy", 50.0))
        reason = s1.get("reason", "")

        parsed = {
            "features": raw_features,
            "predicted_accuracy": raw_accuracy,
            "reason": reason,
        }
        normalized, norm_log = core.normalize_stage1_rating(parsed)
        enriched = core.enrich_stage1_rating(
            normalized,
            features_model_raw=raw_features,
            normalization_log=norm_log,
        )

        v = item.get("verification")
        final_lvl = enriched["difficulty_level_step1"]
        auto_adj_applied = False
        if v and isinstance(v, dict):
            try:
                recalc = core.recalculate_verification(
                    current_level=enriched["difficulty_level_step1"],
                    original_high_count=enriched["high_difficulty_feature_count"],
                    original_high_features=enriched["high_difficulty_features"],
                    original_accuracy=enriched["original_predicted_accuracy"],
                    original_features=enriched["features"],
                    allow_auto_adjustment=True,
                    verification=v,
                )
                final_res = core.finalize_level(
                    current_level=enriched["difficulty_level_step1"],
                    review_action={"合理": "维持", "偏高": "建议降一档", "偏低": "建议升一档"}.get(
                        recalc["rating_reasonableness"], "维持"
                    ),
                    model_suggested_level=recalc["adjusted_difficulty_level"],
                    input_sufficiency="充分",
                    auto_adjustment_enabled=True,
                )
                if recalc["multiplier_reasonableness"] != "合理":
                    final_lvl = enriched["difficulty_level_step1"]
                    auto_adj_applied = False
                else:
                    final_lvl = final_res.final_level
                    auto_adj_applied = final_res.auto_adjustment_applied
            except Exception:
                final_lvl = enriched["difficulty_level_step1"]

        results.append({
            "question_id": qid,
            "target": target,
            "step1_pred": enriched["difficulty_level_step1"],
            "final_pred": final_lvl,
            "auto_adj_applied": auto_adj_applied,
            "rules": enriched.get("structural_level_constraint", {}).get("rule_ids", []),
            "cand10_step1": item.get("difficulty_level_step1"),
            "cand10_final": item.get("final_difficulty_level"),
        })
    return results


def print_run_metrics(name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    s1_correct = sum(1 for r in results if r["step1_pred"] == r["target"])
    fin_correct = sum(1 for r in results if r["final_pred"] == r["target"])
    s1_acc = s1_correct / total
    fin_acc = fin_correct / total
    
    y_true = [r["target"] for r in results]
    y_pred_fin = [r["final_pred"] for r in results]
    qwk = quadratic_weighted_kappa(y_true, y_pred_fin)

    level_stats = {}
    for lvl in LEVELS:
        true_sub = [r for r in results if r["target"] == lvl]
        pred_sub = [r for r in results if r["final_pred"] == lvl]
        recall = (sum(1 for r in true_sub if r["final_pred"] == lvl) / len(true_sub)) if true_sub else 0.0
        precision = (sum(1 for r in pred_sub if r["target"] == lvl) / len(pred_sub)) if pred_sub else 0.0
        level_stats[lvl] = {
            "count": len(true_sub),
            "pred_count": len(pred_sub),
            "recall": recall,
            "precision": precision,
        }

    return {
        "name": name,
        "total": total,
        "s1_acc": s1_acc,
        "final_acc": fin_acc,
        "qwk": qwk,
        "level_stats": level_stats,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    labels_path = root / "data" / "high_chemistry" / "test_sets" / "chatgpt_reference_20260807_test500_labels.jsonl"
    labels = load_labels(labels_path)

    run_paths = [
        root / "outputs" / "model_runs" / f"high_chemistry_v22_cand10_run{i}.jsonl"
        for i in [1, 2, 3]
    ]

    all_runs_results = []
    print("=" * 70)
    print(" Candidate 11 Frozen Replay on Candidate 10 Runs (500 Questions)")
    print("=" * 70)

    for i, p in enumerate(run_paths, 1):
        if not p.exists():
            print(f"File not found: {p}")
            continue
        res = replay_single_run(p, labels)
        all_runs_results.append(res)
        metrics = print_run_metrics(f"Run {i}", res)
        print(f"\n--- {metrics['name']} ---")
        print(f"Step 1 Acc: {metrics['s1_acc']:.2%}")
        print(f"Final  Acc: {metrics['final_acc']:.2%}")
        print(f"QWK       : {metrics['qwk']:.4f}")
        for lvl in LEVELS:
            st = metrics['level_stats'][lvl]
            print(f"  {lvl} (N={st['count']:3d}, Pred={st['pred_count']:3d}): Recall={st['recall']:.2%}, Prec={st['precision']:.2%}")

    if len(all_runs_results) == 3:
        # Consistency analysis
        agree_step1 = 0
        agree_final = 0
        agree_final_correct = 0
        
        # Build map by question_id for accurate multi-run pairing
        map_r1 = {r["question_id"]: r for r in all_runs_results[0]}
        map_r2 = {r["question_id"]: r for r in all_runs_results[1]}
        map_r3 = {r["question_id"]: r for r in all_runs_results[2]}

        for qid, target in labels.items():
            if qid in map_r1 and qid in map_r2 and qid in map_r3:
                p_s1 = [map_r1[qid]["step1_pred"], map_r2[qid]["step1_pred"], map_r3[qid]["step1_pred"]]
                p_fin = [map_r1[qid]["final_pred"], map_r2[qid]["final_pred"], map_r3[qid]["final_pred"]]
                if len(set(p_s1)) == 1:
                    agree_step1 += 1
                if len(set(p_fin)) == 1:
                    agree_final += 1
                    if p_fin[0] == target:
                        agree_final_correct += 1

        print("\n" + "=" * 70)
        print(" Three-Run Consistency Analysis")
        print("=" * 70)
        print(f"Step 1 3-Run Identical Questions: {agree_step1} / {len(labels)} ({agree_step1/len(labels):.2%})")
        print(f"Final  3-Run Identical Questions: {agree_final} / {len(labels)} ({agree_final/len(labels):.2%})")
        print(f"Stable Subset Final Accuracy    : {agree_final_correct} / {agree_final} ({agree_final_correct/agree_final:.2%})")

        mean_s1 = sum(sum(1 for r in run_res if r["step1_pred"] == r["target"]) / len(run_res) for run_res in all_runs_results) / 3
        mean_fin = sum(sum(1 for r in run_res if r["final_pred"] == r["target"]) / len(run_res) for run_res in all_runs_results) / 3
        print(f"\nMean Step 1 Acc: {mean_s1:.2%}")
        print(f"Mean Final  Acc: {mean_fin:.2%}")

if __name__ == "__main__":
    main()
