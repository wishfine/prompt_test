#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测高中化学500题两阶段结果并生成评测报告与 Mismatch 记录。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


LEVELS = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
LEVEL_INDEX = {level: index + 1 for index, level in enumerate(LEVELS)}

LEVEL_NAME_TO_NUMBER = {
    "送分题": 1,
    "基础题": 2,
    "中等题": 3,
    "拔高题": 4,
    "压轴题": 5,
    "难度1档": 1,
    "难度2档": 2,
    "难度3档": 3,
    "难度4档": 4,
    "难度5档": 5,
    "1档": 1,
    "2档": 2,
    "3档": 3,
    "4档": 4,
    "5档": 5,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 question_id 评测化学难度预测并生成报告")
    parser.add_argument("--labels", required=True, help="标准标签 jsonl / csv 文件路径")
    parser.add_argument("--predictions", required=True, help="模型预测 jsonl 文件路径")
    parser.add_argument("--errors", help="错误日志 jsonl 路径（可选）")
    parser.add_argument(
        "--level-source",
        choices=("pre-postprocess", "final", "step1"),
        default="pre-postprocess",
        help="评测档位来源：pre-postprocess/step1 (第一阶段原始), final (最终复核后)",
    )
    parser.add_argument("--report", required=True, help="输出评测 JSON 报告路径")
    parser.add_argument("--mismatches", required=True, help="输出 Mismatches CSV 文件路径")
    return parser.parse_args()


def jsonl_items(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_number} 行不是合法 JSON: {exc}") from exc
            if isinstance(item, dict):
                yield line_number, item


def normalize_level(val: Any) -> tuple[str | None, int | None]:
    if val is None:
        return None, None
    s = str(val).strip()
    num = LEVEL_NAME_TO_NUMBER.get(s)
    if num is None:
        try:
            num = int(s)
        except ValueError:
            num = None
    if num is None or num not in (1, 2, 3, 4, 5):
        return None, None
    return f"难度{num}档", num


def load_labels(path: Path) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    if path.suffix.lower() in {".jsonl", ".json"}:
        for line_number, row in jsonl_items(path):
            qid = str(row.get("question_id") or "").strip()
            if not qid:
                continue
            raw_lvl = (
                row.get("revalidated_difficulty_level")
                or row.get("reviewed_difficulty_level")
                or row.get("manual_difficulty_level")
                or row.get("difficulty_level")
                or row.get("human_difficulty_level")
                or row.get("standard_level")
                or row.get("difficulty")
                or row.get("previous_reference_difficulty_level")
            )
            lvl_name, lvl_num = normalize_level(raw_lvl)
            if lvl_num is None:
                continue
            labels[qid] = {
                "question_id": qid,
                "standard_level": lvl_num,
                "standard_level_name": lvl_name,
                "reason": str(
                    row.get("revalidated_reason")
                    or row.get("review_reason")
                    or row.get("manual_label_reason")
                    or row.get("manual_reason")
                    or row.get("reason")
                    or ""
                ),
            }
    else:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                qid = str(row.get("question_id") or "").strip()
                if not qid:
                    continue
                raw_lvl = (
                    row.get("standard_level")
                    or row.get("difficulty_level")
                    or row.get("standard_level_name")
                )
                lvl_name, lvl_num = normalize_level(raw_lvl)
                if lvl_num is None:
                    continue
                labels[qid] = {
                    "question_id": qid,
                    "standard_level": lvl_num,
                    "standard_level_name": lvl_name,
                    "reason": str(row.get("reason") or row.get("review_reason") or ""),
                }
    return labels


def extract_prediction(item: dict[str, Any], level_source: str) -> tuple[str | None, int | None]:
    raw_lvl = None
    if level_source in ("pre-postprocess", "step1"):
        raw_lvl = (
            item.get("difficulty_level_step1")
            or (item.get("difficulty_rating_stage1", {}).get("difficulty_level_step1") if isinstance(item.get("difficulty_rating_stage1"), dict) else None)
            or (item.get("difficulty_rating", {}).get("postprocess_original_level") if isinstance(item.get("difficulty_rating"), dict) else None)
            or item.get("difficulty_level")
        )
    elif level_source == "final":
        raw_lvl = (
            item.get("final_difficulty_level")
            or item.get("difficulty_level")
            or (item.get("difficulty_rating", {}).get("difficulty_level") if isinstance(item.get("difficulty_rating"), dict) else None)
        )
    else:
        raw_lvl = item.get("final_difficulty_level") or item.get("difficulty_level_step1") or item.get("difficulty_level")

    return normalize_level(raw_lvl)


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
            weight = ((truth_index - prediction_index) ** 2) / denominator
            observed_disagreement += weight * observed[truth_index][prediction_index]
            expected_disagreement += (
                weight * truth_counts[truth_index] * prediction_counts[prediction_index] / sample_count
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return round(1.0 - observed_disagreement / expected_disagreement, 4)


def main() -> None:
    args = parse_args()
    labels = load_labels(Path(args.labels))
    pred_path = Path(args.predictions)

    predictions: dict[str, dict[str, Any]] = {}
    for line_number, item in jsonl_items(pred_path):
        qid = str(item.get("question_id") or "").strip()
        if not qid:
            continue
        lvl_name, lvl_num = extract_prediction(item, args.level_source)
        predictions[qid] = {
            "item": item,
            "level_name": lvl_name,
            "level_num": lvl_num,
        }

    matched_ids = sorted(set(labels.keys()) & set(predictions.keys()))
    truth_values: list[str] = []
    pred_values: list[str] = []
    truth_counts: Counter[str] = Counter()
    pred_counts: Counter[str] = Counter()
    confusion = {t: {p: 0 for p in LEVELS} for t in LEVELS}
    mismatches: list[dict[str, Any]] = []

    exact_correct = 0
    within_one = 0
    total_abs_diff = 0
    severe_disagreement = 0

    for qid in matched_ids:
        t_name = labels[qid]["standard_level_name"]
        t_num = labels[qid]["standard_level"]
        p_name = predictions[qid]["level_name"]
        p_num = predictions[qid]["level_num"]
        if p_name is None or p_num is None:
            continue

        truth_values.append(t_name)
        pred_values.append(p_name)
        truth_counts[t_name] += 1
        pred_counts[p_name] += 1
        confusion[t_name][p_name] += 1

        abs_diff = abs(t_num - p_num)
        total_abs_diff += abs_diff
        if abs_diff == 0:
            exact_correct += 1
            within_one += 1
        elif abs_diff == 1:
            within_one += 1
        else:
            severe_disagreement += 1

        if abs_diff > 0:
            item_raw = predictions[qid]["item"]
            s1_info = item_raw.get("difficulty_rating_stage1") or {}
            features = s1_info.get("features") or {}
            mismatches.append({
                "question_id": qid,
                "standard_level": t_name,
                "predicted_level": p_name,
                "level_source": args.level_source,
                "abs_diff": abs_diff,
                "direction": "偏高" if p_num > t_num else "偏低",
                "predicted_accuracy": s1_info.get("predicted_accuracy") or item_raw.get("predicted_accuracy"),
                "high_feature_count": s1_info.get("high_difficulty_feature_count", 0),
                "high_features": s1_info.get("high_difficulty_features", []),
                "structural_rules": s1_info.get("structural_level_constraint", {}).get("rule_ids", []),
                "reason": s1_info.get("reason", "") or item_raw.get("reason", ""),
                "teacher_reason": labels[qid].get("reason", ""),
                "step_count": features.get("step_count", ""),
                "task_breadth": features.get("required_task_breadth", ""),
                "reasoning_chain": features.get("reasoning_chain", ""),
                "model_relation": features.get("model_relation", ""),
            })

    valid_count = len(truth_values)
    acc = round(exact_correct / valid_count, 4) if valid_count else 0.0
    within_one_acc = round(within_one / valid_count, 4) if valid_count else 0.0
    mae = round(total_abs_diff / valid_count, 4) if valid_count else 0.0
    qwk = quadratic_weighted_kappa(truth_values, pred_values)

    per_level_metrics: dict[str, dict[str, Any]] = {}
    for lvl in LEVELS:
        tp = confusion[lvl][lvl]
        fn = sum(confusion[lvl][p] for p in LEVELS if p != lvl)
        fp = sum(confusion[t][lvl] for t in LEVELS if t != lvl)
        total_t = tp + fn
        total_p = tp + fp
        rec = round(tp / total_t, 4) if total_t else 0.0
        prec = round(tp / total_p, 4) if total_p else 0.0
        f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
        per_level_metrics[lvl] = {
            "true_count": total_t,
            "pred_count": total_p,
            "correct": tp,
            "recall": rec,
            "precision": prec,
            "f1": f1,
        }

    print(f"本次模型输出唯一 ID: {len(predictions)}")
    print(f"其中有干净标准标签: {valid_count}")
    print(f"合法难度预测: {valid_count}")
    print(f"完全一致: {exact_correct}")
    print(f"Accuracy（合法预测）: {acc:.2%}")
    print(f"Strict Accuracy（失败也计错）: {exact_correct / len(predictions):.2%}" if predictions else "None")
    print(f"相差不超过一档: {within_one_acc:.2%}")
    print(f"MAE: {mae}")
    print(f"Quadratic Weighted Kappa (QWK): {qwk}")
    print(f"严重偏差（跨2档以上）: {severe_disagreement}")
    print(f"标签分布: {dict(truth_counts)}")
    print(f"预测分布: {dict(pred_counts)}")
    print("\n--- 各档位详细 Precision / Recall ---")
    for lvl in LEVELS:
        m = per_level_metrics[lvl]
        print(f"  {lvl}: Recall={m['recall']:.2%} ({m['correct']}/{m['true_count']}), Precision={m['precision']:.2%} ({m['correct']}/{m['pred_count']}), F1={m['f1']:.4f}")

    report = {
        "dataset_summary": {
            "total_predictions": len(predictions),
            "matched_labels": valid_count,
            "level_source": args.level_source,
        },
        "metrics": {
            "accuracy": acc,
            "within_one_accuracy": within_one_acc,
            "mae": mae,
            "qwk": qwk,
            "severe_disagreement_count": severe_disagreement,
        },
        "per_level_metrics": per_level_metrics,
        "confusion_matrix": confusion,
        "truth_distribution": dict(truth_counts),
        "prediction_distribution": dict(pred_counts),
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n报告: {report_path.resolve()}")

    mismatches_path = Path(args.mismatches)
    mismatches_path.parent.mkdir(parents=True, exist_ok=True)
    if mismatches:
        keys = list(mismatches[0].keys())
        with mismatches_path.open("w", encoding="utf-8-sig", newline="") as h:
            writer = csv.DictWriter(h, fieldnames=keys)
            writer.writeheader()
            for row in mismatches:
                row_copy = dict(row)
                if isinstance(row_copy.get("high_features"), list):
                    row_copy["high_features"] = "; ".join(map(str, row_copy["high_features"]))
                if isinstance(row_copy.get("structural_rules"), list):
                    row_copy["structural_rules"] = "; ".join(map(str, row_copy["structural_rules"]))
                writer.writerow(row_copy)
    print(f"错题: {mismatches_path.resolve()}")


if __name__ == "__main__":
    main()
