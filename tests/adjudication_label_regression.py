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
        if "题目ID" not in fields:
            raise ValueError(f"裁定 CSV 必须含题目ID字段，实际字段为：{fields}")
        label_field = next(
            (field for field in ("修订后主标签", "最终裁定档") if field in fields),
            "",
        )
        if not label_field:
            raise ValueError(
                "裁定 CSV 必须含修订后主标签或最终裁定档字段，"
                f"实际字段为：{fields}"
            )
        confidence_field = next(
            (field for field in ("修订后置信度", "裁定置信度") if field in fields),
            "",
        )
        for row in reader:
            question_id = str(row.get("题目ID") or "").strip()
            label = str(row.get(label_field) or "").strip()
            if question_id and label in LEVEL_ORDER:
                labels[question_id] = label
                confidence[question_id] = str(row.get(confidence_field) or "").strip()
    return labels, confidence, fields


def evaluate(
    results_path: Path,
    labels: dict[str, str],
    confidence: dict[str, str] | None = None,
) -> dict[str, Any]:
    rows: list[
        tuple[
            str,
            str,
            str,
            str | None,
            list[dict[str, Any]],
            str,
            dict[str, Any],
            str,
            dict[str, Any],
        ]
    ] = []
    with results_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "").strip()
            prediction = extract_prediction(item)
            if question_id in labels and prediction:
                before_review = str(item.get("difficulty_level_before_review") or "").strip()
                if before_review not in LEVEL_ORDER:
                    before_rating = item.get("difficulty_rating_before_review")
                    if isinstance(before_rating, dict):
                        before_review = str(before_rating.get("difficulty_level") or "").strip()
                if before_review not in LEVEL_ORDER:
                    before_review = prediction
                before_verification = str(
                    item.get("difficulty_level_before_verification") or ""
                ).strip()
                if before_verification not in LEVEL_ORDER:
                    before_verification = prediction
                rows.append(
                    (
                        question_id,
                        labels[question_id],
                        prediction,
                        extract_raw_prediction(item),
                        item.get("postprocess_actions") or [],
                        before_review,
                        item.get("boundary_review") if isinstance(item.get("boundary_review"), dict) else {},
                        before_verification,
                        item.get("verification_agent")
                        if isinstance(item.get("verification_agent"), dict)
                        else {},
                    )
                )
    if not rows:
        raise ValueError("结果中没有可与 GPT-5.6 裁定 CSV 匹配的有效预测")

    final_summary = summarize_predictions(
        [(target, prediction) for _, target, prediction, _, _, _, _, _, _ in rows]
    )
    before_review_summary = summarize_predictions(
        [(target, before_review) for _, target, _, _, _, before_review, _, _, _ in rows]
    )
    before_verification_summary = summarize_predictions(
        [
            (target, before_verification)
            for _, target, _, _, _, _, _, before_verification, _ in rows
        ]
    )
    raw_summary = summarize_predictions(
        [(target, raw) for _, target, _, raw, _, _, _, _, _ in rows if raw in LEVEL_ORDER]
    )
    rule_stats: dict[str, Counter[str]] = defaultdict(Counter)
    for _, target, _, raw, actions, before_review, _, _, _ in rows:
        raw_error = abs(LEVEL_ORDER[raw] - LEVEL_ORDER[target]) if raw in LEVEL_ORDER else None
        postprocess_error = abs(LEVEL_ORDER[before_review] - LEVEL_ORDER[target])
        for action in actions:
            rule = str(action.get("rule") or "unknown")
            rule_stats[rule]["triggered"] += 1
            if raw_error is None or postprocess_error == raw_error:
                rule_stats[rule]["unchanged"] += 1
            elif postprocess_error < raw_error:
                rule_stats[rule]["improved"] += 1
            else:
                rule_stats[rule]["worsened"] += 1

    review_stats: Counter[str] = Counter()
    review_transitions: dict[str, Counter[str]] = defaultdict(Counter)
    review_confidence: Counter[str] = Counter()
    has_review_data = False
    for _, target, prediction, _, _, before_review, review_data, _, _ in rows:
        if not review_data:
            continue
        has_review_data = True
        review_stats["enabled"] += 1
        if review_data.get("selected"):
            review_stats["selected"] += 1
        else:
            review_stats["not_selected"] += 1
        if review_data.get("error"):
            review_stats["request_failed"] += 1
        result = review_data.get("result")
        if isinstance(result, dict) and result:
            review_stats["valid_response"] += 1
            confidence_value = str(result.get("confidence") or "")
            if confidence_value:
                review_confidence[confidence_value] += 1
        if not review_data.get("applied"):
            continue
        review_stats["applied"] += 1
        before_error = abs(LEVEL_ORDER[before_review] - LEVEL_ORDER[target])
        final_error = abs(LEVEL_ORDER[prediction] - LEVEL_ORDER[target])
        transition = f"{before_review}->{prediction}"
        review_transitions[transition]["triggered"] += 1
        if final_error < before_error:
            outcome = "improved"
        elif final_error > before_error:
            outcome = "worsened"
        else:
            outcome = "unchanged"
        review_stats[outcome] += 1
        review_transitions[transition][outcome] += 1

    agent_stats: Counter[str] = Counter()
    agent_transitions: dict[str, Counter[str]] = defaultdict(Counter)
    agent_confidence: Counter[str] = Counter()
    agent_disagreement: Counter[str] = Counter()
    agent_audit_pairs: list[tuple[str, str, str]] = []
    has_agent_data = False
    for _, target, prediction, _, _, _, _, before_verification, agent_data in rows:
        if not agent_data:
            continue
        has_agent_data = True
        agent_stats["enabled"] += 1
        agent_stats["selected" if agent_data.get("selected") else "not_selected"] += 1
        if agent_data.get("error"):
            agent_stats["request_failed"] += 1
        mode = str(agent_data.get("mode") or "auto_apply")
        agent_stats[mode] += 1
        if agent_data.get("would_apply"):
            agent_stats["would_apply"] += 1
        result = agent_data.get("blind_review")
        if isinstance(result, dict) and result:
            agent_stats["valid_response"] += 1
            confidence_value = str(result.get("confidence") or "")
            if confidence_value:
                agent_confidence[confidence_value] += 1
            review_level = str(result.get("review_level") or "")
            if agent_data.get("selected") and review_level in LEVEL_ORDER:
                agent_audit_pairs.append((target, before_verification, review_level))
                if review_level == before_verification:
                    agent_disagreement["agreed"] += 1
                else:
                    agent_disagreement["disagreed"] += 1
                    before_error = abs(LEVEL_ORDER[before_verification] - LEVEL_ORDER[target])
                    review_error = abs(LEVEL_ORDER[review_level] - LEVEL_ORDER[target])
                    if review_error < before_error:
                        agent_disagreement["blind_better"] += 1
                    elif review_error > before_error:
                        agent_disagreement["before_better"] += 1
                    else:
                        agent_disagreement["equal_distance"] += 1
        if not agent_data.get("applied"):
            continue
        agent_stats["applied"] += 1
        before_error = abs(LEVEL_ORDER[before_verification] - LEVEL_ORDER[target])
        final_error = abs(LEVEL_ORDER[prediction] - LEVEL_ORDER[target])
        transition = f"{before_verification}->{prediction}"
        agent_transitions[transition]["triggered"] += 1
        if final_error < before_error:
            outcome = "improved"
        elif final_error > before_error:
            outcome = "worsened"
        else:
            outcome = "unchanged"
        agent_stats[outcome] += 1
        agent_transitions[transition][outcome] += 1

    confidence_slices: dict[str, dict[str, Any]] = {}
    if confidence:
        for value in sorted({confidence.get(question_id, "") for question_id, *_ in rows} - {""}):
            confidence_slices[value] = summarize_predictions(
                [
                    (target, prediction)
                    for question_id, target, prediction, _, _, _, _, _, _ in rows
                    if confidence.get(question_id) == value
                ]
            )

    return {
        **final_summary,
        "raw_evaluation": raw_summary,
        "before_boundary_review_evaluation": before_review_summary,
        "before_verification_evaluation": before_verification_summary,
        "postprocess_rules": {rule: dict(counter) for rule, counter in sorted(rule_stats.items())},
        "boundary_review": (
            {
                "stats": dict(review_stats),
                "confidence": dict(review_confidence),
                "transitions": {
                    transition: dict(counter)
                    for transition, counter in sorted(review_transitions.items())
                },
            }
            if has_review_data
            else {}
        ),
        "verification_agent": (
            {
                "stats": dict(agent_stats),
                "confidence": dict(agent_confidence),
                "transitions": {
                    transition: dict(counter)
                    for transition, counter in sorted(agent_transitions.items())
                },
                "audit_comparison": (
                    {
                        "selected_before_evaluation": summarize_predictions(
                            [(target, before) for target, before, _ in agent_audit_pairs]
                        ),
                        "blind_review_evaluation": summarize_predictions(
                            [(target, blind) for target, _, blind in agent_audit_pairs]
                        ),
                        "disagreement": {
                            key: agent_disagreement.get(key, 0)
                            for key in (
                                "agreed",
                                "disagreed",
                                "blind_better",
                                "before_better",
                                "equal_distance",
                            )
                        },
                    }
                    if agent_audit_pairs
                    else {}
                ),
            }
            if has_agent_data
            else {}
        ),
        "confidence_slices": confidence_slices,
    }


def export_mismatches(
    results_path: Path,
    output_path: Path,
    labels: dict[str, str],
    confidence: dict[str, str] | None = None,
) -> int:
    """导出单次运行全部错题，不做跨运行投票或合并。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    exported = 0
    with results_path.open("r", encoding="utf-8") as source, output_path.open(
        "w", encoding="utf-8"
    ) as target_file:
        for line in source:
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "").strip()
            prediction = extract_prediction(item)
            reference = labels.get(question_id)
            if not prediction or reference not in LEVEL_ORDER or prediction == reference:
                continue
            exported_item = dict(item)
            exported_item["adjudication_reference_level"] = reference
            exported_item["adjudication_reference_confidence"] = (
                (confidence or {}).get(question_id, "")
            )
            exported_item["adjudication_prediction_level"] = prediction
            target_file.write(json.dumps(exported_item, ensure_ascii=False) + "\n")
            exported += 1
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT-5.6 物理难度裁定标签回归评估")
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--evaluate", required=True, help="模型输出 JSONL")
    parser.add_argument(
        "--export-mismatches",
        default="",
        help="可选：导出该次运行全部错题 JSONL，供单次边界复核使用",
    )
    args = parser.parse_args()

    labels, confidence, fields = load_adjudicated_labels(Path(args.csv))
    questions = load_questions(Path(args.jsonl))
    matched = sorted(set(labels) & set(questions))
    distribution = Counter(labels[question_id] for question_id in matched)
    report: dict[str, Any] = {
        "label_source": str(Path(args.csv)),
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
    if args.export_mismatches:
        report["exported_mismatches"] = export_mismatches(
            Path(args.evaluate),
            Path(args.export_mismatches),
            labels,
            confidence,
        )
        report["mismatch_output"] = str(Path(args.export_mismatches).resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
