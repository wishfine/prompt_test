# -*- coding: utf-8 -*-
"""评估相邻边界复核的首轮、反事实写回与可接受边界指标。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from tests.teacher_label_regression import LEVEL_ORDER, TEACHER_TO_LEVEL, summarize_predictions
except ModuleNotFoundError:
    from teacher_label_regression import LEVEL_ORDER, TEACHER_TO_LEVEL, summarize_predictions  # type: ignore[no-redef]


def load_labels(path: Path) -> tuple[dict[str, str], list[str]]:
    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        id_field = next(
            (field for field in ("题目ID", "ID", "question_id") if field in fields),
            "",
        )
        label_field = next(
            (
                field
                for field in (
                    "修订后主标签",
                    "最终裁定档",
                    "difficulty_level",
                    "teacher_level",
                    "label",
                    "难度",
                )
                if field in fields
            ),
            "",
        )
        if not id_field or not label_field:
            raise ValueError(f"无法识别标签 CSV 的 ID/标签字段：{fields}")
        for row in reader:
            question_id = str(row.get(id_field) or "").strip()
            raw_label = str(row.get(label_field) or "").strip()
            label = TEACHER_TO_LEVEL.get(raw_label, raw_label)
            if question_id and label in LEVEL_ORDER:
                labels[question_id] = label
    return labels, fields


def extract_level(item: dict[str, Any]) -> str:
    rating = item.get("difficulty_rating")
    if isinstance(rating, dict) and rating.get("difficulty_level") in LEVEL_ORDER:
        return str(rating["difficulty_level"])
    return ""


def evaluate(path: Path, labels: dict[str, str]) -> dict[str, Any]:
    baseline_rows: list[tuple[str, str]] = []
    proposed_rows: list[tuple[str, str]] = []
    recommended_selected_rows: list[tuple[str, str]] = []
    acceptance_correct = 0
    selected_count = 0
    valid_selected_count = 0
    overall_acceptance_correct = 0
    stats: Counter[str] = Counter()
    transition_stats: dict[str, Counter[str]] = {}
    reasoning_violations = 0

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "")
            if question_id not in labels:
                continue
            target = labels[question_id]
            baseline = extract_level(item)
            if baseline not in LEVEL_ORDER:
                continue
            agent = item.get("boundary_review_agent")
            agent = agent if isinstance(agent, dict) else {}
            review = agent.get("review")
            review = review if isinstance(review, dict) else {}
            proposed = baseline
            if agent.get("would_apply") and review.get("recommended_level") in LEVEL_ORDER:
                proposed = str(review["recommended_level"])

            baseline_rows.append((target, baseline))
            proposed_rows.append((target, proposed))
            if agent.get("selected"):
                selected_count += 1
                recommended = str(review.get("recommended_level") or "")
                if recommended in LEVEL_ORDER:
                    valid_selected_count += 1
                    recommended_selected_rows.append((target, recommended))
                    acceptable = set(review.get("acceptable_levels") or [])
                    if target in acceptable:
                        acceptance_correct += 1
                        overall_acceptance_correct += 1
                if agent.get("error"):
                    stats["selected_error"] += 1
            elif target == baseline:
                overall_acceptance_correct += 1
            if agent.get("would_apply"):
                stats["would_apply"] += 1
                before_error = abs(LEVEL_ORDER[baseline] - LEVEL_ORDER[target])
                after_error = abs(LEVEL_ORDER[proposed] - LEVEL_ORDER[target])
                outcome = "improved" if after_error < before_error else ("worsened" if after_error > before_error else "unchanged")
                stats[outcome] += 1
                transition = f"{baseline}->{proposed}"
                transition_stats.setdefault(transition, Counter())[outcome] += 1

            difficulty_rating = item.get("difficulty_rating")
            reasoning = difficulty_rating.get("reasoning") if isinstance(difficulty_rating, dict) else {}
            expected_lower = difficulty_rating.get("adjacent_lower_level") if isinstance(difficulty_rating, dict) else None
            expected_higher = difficulty_rating.get("adjacent_higher_level") if isinstance(difficulty_rating, dict) else None
            lower_text = str((reasoning or {}).get("why_not_lower") or "")
            higher_text = str((reasoning or {}).get("why_not_higher") or "")
            if expected_lower and f"与{expected_lower}相比" not in lower_text:
                reasoning_violations += 1
            if expected_higher and f"与{expected_higher}相比" not in higher_text:
                reasoning_violations += 1

    if not baseline_rows:
        raise ValueError("结果中没有可与标签匹配的题目")
    return {
        "label_count": len(labels),
        "evaluated": len(baseline_rows),
        "baseline": summarize_predictions(baseline_rows),
        "counterfactual_high_confidence_writeback": summarize_predictions(proposed_rows),
        "selected_reviewer_recommendation": summarize_predictions(recommended_selected_rows),
        "selected_count": selected_count,
        "valid_selected_review_count": valid_selected_count,
        "selected_label_acceptance_rate": (
            round(acceptance_correct / valid_selected_count, 4)
            if valid_selected_count
            else None
        ),
        "overall_label_acceptance_rate": round(
            overall_acceptance_correct / len(baseline_rows),
            4,
        ),
        "writeback_gate": dict(stats),
        "writeback_transitions": {
            transition: dict(values)
            for transition, values in sorted(transition_stats.items())
        },
        "adjacent_reasoning_violation_count": reasoning_violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="相邻边界复核回归评估")
    parser.add_argument("--csv", required=True, help="教师或裁定标签 CSV")
    parser.add_argument("--evaluate", required=True, help="相邻边界复核输出 JSONL")
    args = parser.parse_args()
    labels, fields = load_labels(Path(args.csv))
    report = evaluate(Path(args.evaluate), labels)
    report["label_source"] = args.csv
    report["csv_fields"] = fields
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
