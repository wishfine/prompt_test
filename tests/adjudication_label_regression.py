# -*- coding: utf-8 -*-
"""GPT-5.6 物理难度裁定标签的匹配与回归评估。

裁定 CSV 直接使用五档标签，不再经过教师“容易/较易”同义映射。
题目 ID 始终作为字符串读取。
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from tests.teacher_label_regression import (
        LEVEL_ORDER,
        extract_prediction,
        extract_raw_prediction,
        load_questions,
        summarize_predictions,
    )
except ModuleNotFoundError:  # 支持 python tests/adjudication_label_regression.py
    from teacher_label_regression import (  # type: ignore[no-redef]
        LEVEL_ORDER,
        extract_prediction,
        extract_raw_prediction,
        load_questions,
        summarize_predictions,
    )


DEFAULT_CSV = "data/labeled/physics_adjudicated_labels_gpt56_1066.csv"
DEFAULT_JSONL = "data/labeled/physics_difficulty_tiku_data_0714_1000.jsonl"


def load_adjudicated_labels(path: Path) -> tuple[dict[str, str], dict[str, str], list[str]]:
    labels: dict[str, str] = {}
    confidence: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        required = {"题目ID", "最终裁定档"}
        if not required.issubset(fields):
            raise ValueError(f"裁定 CSV 必须含 {sorted(required)}，实际字段为：{fields}")
        for row in reader:
            question_id = str(row.get("题目ID") or "").strip()
            label = str(row.get("最终裁定档") or "").strip()
            if question_id and label in LEVEL_ORDER:
                labels[question_id] = label
                confidence[question_id] = str(row.get("裁定置信度") or "").strip()
    return labels, confidence, fields


def evaluate(
    results_path: Path,
    labels: dict[str, str],
    confidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows: list[tuple[str, str, str, str | None, list[dict[str, Any]]]] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "").strip()
            prediction = extract_prediction(item)
            if question_id in labels and prediction:
                rows.append(
                    (
                        question_id,
                        labels[question_id],
                        prediction,
                        extract_raw_prediction(item),
                        item.get("postprocess_actions") or [],
                    )
                )
    if not rows:
        raise ValueError("结果中没有可与 GPT-5.6 裁定 CSV 匹配的有效预测")

    final_summary = summarize_predictions([(target, prediction) for _, target, prediction, _, _ in rows])
    raw_summary = summarize_predictions(
        [(target, raw) for _, target, _, raw, _ in rows if raw in LEVEL_ORDER]
    )
    rule_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for _, target, prediction, raw, actions in rows:
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

    confidence_slices: dict[str, dict[str, Any]] = {}
    if confidence:
        for value in sorted({confidence.get(question_id, "") for question_id, *_ in rows} - {""}):
            confidence_slices[value] = summarize_predictions(
                [
                    (target, prediction)
                    for question_id, target, prediction, _, _ in rows
                    if confidence.get(question_id) == value
                ]
            )

    return {
        **final_summary,
        "raw_evaluation": raw_summary,
        "postprocess_rules": {rule: dict(counter) for rule, counter in sorted(rule_stats.items())},
        "confidence_slices": confidence_slices,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-5.6 物理难度裁定标签回归评估")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--evaluate", required=True, help="模型输出 JSONL")
    args = parser.parse_args()

    labels, confidence, fields = load_adjudicated_labels(Path(args.csv))
    questions = load_questions(Path(args.jsonl))
    matched = sorted(set(labels) & set(questions))
    distribution = Counter(labels[question_id] for question_id in matched)
    report: dict[str, Any] = {
        "label_source": "GPT-5.6 adjudication",
        "csv_effective_labels": len(labels),
        "jsonl_questions": len(questions),
        "matched": len(matched),
        "csv_unmatched": len(set(labels) - set(questions)),
        "jsonl_unmatched": len(set(questions) - set(labels)),
        "adjudicated_distribution": {
            level: distribution.get(level, 0) for level in LEVEL_ORDER
        },
        "confidence_distribution": dict(Counter(confidence.get(question_id, "") for question_id in matched)),
        "csv_fields": fields,
        "evaluation": evaluate(Path(args.evaluate), labels, confidence),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
