# -*- coding: utf-8 -*-
"""教师标签匹配、分层抽样与模型结果回归评估。

不读取也不使用 JSONL 的 ``difficulty`` 字段。ID 始终作为字符串处理，避免 19 位
整数在表格软件或浮点转换中失真。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


TEACHER_TO_LEVEL = {"容易": "送分题", "较易": "基础题", "中等": "中等题", "较难": "拔高题", "困难": "压轴题"}
LEVEL_ORDER = {"送分题": 1, "基础题": 2, "中等题": 3, "拔高题": 4, "压轴题": 5}
DEFAULT_CSV = "data/labeled/physics_teacher_labels_0714.csv"
DEFAULT_JSONL = "data/labeled/physics_difficulty_tiku_data_0714_1000.jsonl"


def load_labels(path: Path) -> tuple[dict[str, str], int, list[str]]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if "ID" not in fields or "难度" not in fields:
            raise ValueError(f"CSV 必须含 ID、难度字段，实际字段为：{fields}")
        for row in reader:
            question_id = str(row.get("ID") or "").strip()
            label = str(row.get("难度") or "").strip()
            if question_id and label in TEACHER_TO_LEVEL:
                labels[question_id] = label
    return labels, len(labels), fields


def load_questions(path: Path) -> dict[str, dict[str, Any]]:
    questions: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "").strip()
            if not question_id:
                raise ValueError(f"JSONL 第 {line_number} 行缺少 question_id")
            if question_id in questions:
                raise ValueError(f"JSONL question_id 重复：{question_id}")
            questions[question_id] = item
    return questions


def match_report(labels: dict[str, str], questions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    matched = sorted(set(labels) & set(questions))
    distribution = Counter(labels[question_id] for question_id in matched)
    return {
        "csv_effective_labels": len(labels),
        "jsonl_questions": len(questions),
        "matched": len(matched),
        "csv_unmatched": len(set(labels) - set(questions)),
        "jsonl_unmatched": len(set(questions) - set(labels)),
        "teacher_distribution": {label: distribution.get(label, 0) for label in TEACHER_TO_LEVEL},
    }


def write_stratified_sample(labels: dict[str, str], questions: dict[str, dict[str, Any]], output: Path, per_label: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    grouped: dict[str, list[str]] = defaultdict(list)
    for question_id in sorted(set(labels) & set(questions)):
        grouped[labels[question_id]].append(question_id)
    selected: list[str] = []
    counts: dict[str, int] = {}
    for label in TEACHER_TO_LEVEL:
        ids = grouped[label]
        chosen = rng.sample(ids, min(per_label, len(ids)))
        selected.extend(chosen)
        counts[label] = len(chosen)
    rng.shuffle(selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for question_id in selected:
            # 评测输入中既不放教师标签，也不放 JSONL 旧 difficulty。
            clean = {key: value for key, value in questions[question_id].items() if key != "difficulty"}
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")
    return counts


def extract_prediction(item: dict[str, Any]) -> str | None:
    rating = item.get("difficulty_rating")
    if isinstance(rating, dict) and rating.get("difficulty_level") in LEVEL_ORDER:
        return rating["difficulty_level"]
    return None


def extract_raw_prediction(item: dict[str, Any]) -> str | None:
    raw = item.get("difficulty_level_raw")
    if raw in LEVEL_ORDER:
        return raw
    rating = item.get("difficulty_rating_raw")
    if isinstance(rating, dict) and rating.get("difficulty_level") in LEVEL_ORDER:
        return rating["difficulty_level"]
    return None


def evaluate(results_path: Path, labels: dict[str, str]) -> dict[str, Any]:
    rows: list[tuple[str, str, str, str | None, list[dict[str, Any]]]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "").strip()
            prediction = extract_prediction(item)
            if question_id in labels and prediction:
                rows.append((question_id, TEACHER_TO_LEVEL[labels[question_id]], prediction, extract_raw_prediction(item), item.get("postprocess_actions") or []))
    if not rows:
        raise ValueError("结果中没有可与教师 CSV 匹配的有效预测")
    total = len(rows)
    exact = sum(target == prediction for _, target, prediction, _, _ in rows)
    distances = [abs(LEVEL_ORDER[target] - LEVEL_ORDER[prediction]) for _, target, prediction, _, _ in rows]
    labels_in_order = list(LEVEL_ORDER)
    confusion = {target: {prediction: 0 for prediction in labels_in_order} for target in labels_in_order}
    per_level_total = Counter()
    per_level_correct = Counter()
    over = under = 0
    rule_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for _, target, prediction, raw, actions in rows:
        confusion[target][prediction] += 1
        per_level_total[target] += 1
        if target == prediction:
            per_level_correct[target] += 1
        if LEVEL_ORDER[prediction] > LEVEL_ORDER[target]:
            over += 1
        elif LEVEL_ORDER[prediction] < LEVEL_ORDER[target]:
            under += 1
        raw_error = abs(LEVEL_ORDER[raw] - LEVEL_ORDER[target]) if raw in LEVEL_ORDER else None
        final_error = abs(LEVEL_ORDER[prediction] - LEVEL_ORDER[target])
        for action in actions:
            rule = str(action.get("rule") or "unknown")
            rule_stats[rule]["triggered"] += 1
            if raw_error is None or final_error == raw_error:
                rule_stats[rule]["unchanged"] += 1
            elif final_error < raw_error:
                rule_stats[rule]["improved"] += 1
            else:
                rule_stats[rule]["worsened"] += 1
    return {
        "evaluated": total,
        "exact_match_rate": round(exact / total, 4),
        "within_one_level_rate": round(sum(distance <= 1 for distance in distances) / total, 4),
        "mae": round(sum(distances) / total, 4),
        "severe_deviation_count": sum(distance >= 2 for distance in distances),
        "over_predicted": over,
        "under_predicted": under,
        "confusion_matrix": confusion,
        "per_level_accuracy": {level: round(per_level_correct[level] / per_level_total[level], 4) if per_level_total[level] else None for level in labels_in_order},
        "postprocess_rules": {rule: dict(counter) for rule, counter in sorted(rule_stats.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="教师标签基准：匹配检查、分层抽样和回归评估")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--write-stratified", help="写入不含 difficulty/教师标签的固定分层样本 JSONL")
    parser.add_argument("--per-label", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--evaluate", help="读取模型输出 JSONL，按 CSV 教师标签统计回归指标")
    args = parser.parse_args()

    labels, _, fields = load_labels(Path(args.csv))
    questions = load_questions(Path(args.jsonl))
    report = match_report(labels, questions)
    report["csv_fields"] = fields
    if args.write_stratified:
        report["stratified_sample"] = write_stratified_sample(labels, questions, Path(args.write_stratified), args.per_label, args.seed)
        report["stratified_seed"] = args.seed
    if args.evaluate:
        report["evaluation"] = evaluate(Path(args.evaluate), labels)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
