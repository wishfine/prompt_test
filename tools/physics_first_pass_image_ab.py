# -*- coding: utf-8 -*-
"""准备并评估初中物理首轮条件传图 A/B 实验。

prepare 只抽取 auto 路由命中的题，避免把未传图题重新调用一次造成随机性混淆。
evaluate 将候选结果覆盖到固定基线，再按教师反馈计算净增益和分档指标。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import physics_difficulty_rating_with_cache as rating  # noqa: E402


LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS, 1)}
QUESTION_FIELDS = (
    "parent_id",
    "question_id",
    "stem",
    "options",
    "analysis",
    "sub_questions",
    "stem_pic_url",
    "analysis_pic_url",
)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} 不是 JSON 对象")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def question_id(row: Mapping[str, Any]) -> str:
    value = row.get("question_id")
    if value in (None, ""):
        raise ValueError("发现缺少 question_id 的记录")
    return str(value)


def index_unique(rows: Iterable[Dict[str, Any]], source: str) -> Dict[str, Dict[str, Any]]:
    values: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = question_id(row)
        if qid in values:
            raise ValueError(f"{source} 存在重复 question_id={qid}")
        values[qid] = row
    return values


def predicted_level(row: Mapping[str, Any]) -> str:
    nested = row.get("difficulty_rating")
    if isinstance(nested, dict) and nested.get("difficulty_level") in LEVEL_INDEX:
        return str(nested["difficulty_level"])
    for key in ("difficulty_level", "difficulty_level_raw", "model_difficulty_level"):
        if row.get(key) in LEVEL_INDEX:
            return str(row[key])
    return ""


def teacher_level(row: Mapping[str, Any]) -> str:
    value = row.get("human_difficulty_level")
    if value in LEVEL_INDEX:
        return str(value)
    if row.get("verdict") == "correct" and row.get("model_difficulty_level") in LEVEL_INDEX:
        return str(row["model_difficulty_level"])
    return ""


def prepare(args: argparse.Namespace) -> None:
    baseline_path = Path(args.baseline)
    output_path = Path(args.output)
    rows = read_jsonl(baseline_path)
    selected: List[Dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    available = 0
    total_urls = 0

    for row in rows:
        route = rating.select_rating_images(row, "auto")
        available += int(route["image_available"])
        if not route["image_included"]:
            continue
        clean = {
            field: row.get(field)
            for field in QUESTION_FIELDS
            if row.get(field) not in (None, "", [], {})
        }
        selected.append(clean)
        total_urls += int(route["selected_url_count"])
        reason_counts.update(route["reasons"])

    write_jsonl(output_path, selected)
    manifest = {
        "baseline": str(baseline_path),
        "baseline_count": len(rows),
        "image_available_count": available,
        "auto_selected_count": len(selected),
        "auto_selected_ratio": round(len(selected) / len(rows), 6) if rows else 0,
        "selected_image_url_count": total_urls,
        "route_reason_counts": dict(reason_counts.most_common()),
        "output": str(output_path),
        "experiment_rule": (
            "仅重跑 auto 条件传图命中的题；未命中题在评估时复用固定基线，"
            "隔离 Lite temperature=1 的额外随机波动"
        ),
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def classification_metrics(
    labels: Mapping[str, str],
    predictions: Mapping[str, str],
) -> Dict[str, Any]:
    qids = sorted(set(labels) & set(predictions))
    confusion = {
        truth: {prediction: 0 for prediction in LEVELS}
        for truth in LEVELS
    }
    exact = within_one = absolute_error = severe = over = under = 0
    for qid in qids:
        truth = labels[qid]
        prediction = predictions[qid]
        confusion[truth][prediction] += 1
        delta = LEVEL_INDEX[prediction] - LEVEL_INDEX[truth]
        exact += int(delta == 0)
        within_one += int(abs(delta) <= 1)
        absolute_error += abs(delta)
        severe += int(abs(delta) >= 2)
        over += int(delta > 0)
        under += int(delta < 0)

    per_level: Dict[str, Any] = {}
    for level in LEVELS:
        true_positive = confusion[level][level]
        support = sum(confusion[level].values())
        predicted = sum(confusion[truth][level] for truth in LEVELS)
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_level[level] = {
            "support": support,
            "predicted": predicted,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }

    evaluated = len(qids)
    return {
        "evaluated": evaluated,
        "correct": exact,
        "accuracy": round(exact / evaluated, 6) if evaluated else 0.0,
        "within_one_level_rate": round(within_one / evaluated, 6) if evaluated else 0.0,
        "mae": round(absolute_error / evaluated, 6) if evaluated else 0.0,
        "severe_deviation_count": severe,
        "over_predicted": over,
        "under_predicted": under,
        "confusion_matrix": confusion,
        "per_level_metrics": per_level,
    }


def evaluate(args: argparse.Namespace) -> None:
    baseline_rows = index_unique(read_jsonl(Path(args.baseline)), "baseline")
    candidate_rows = index_unique(read_jsonl(Path(args.candidate)), "candidate")
    control_rows = (
        index_unique(read_jsonl(Path(args.control)), "control")
        if args.control
        else {}
    )
    annotation_rows = index_unique(read_jsonl(Path(args.annotations)), "annotations")

    labels = {
        qid: teacher_level(row)
        for qid, row in annotation_rows.items()
        if teacher_level(row)
    }
    baseline_predictions = {
        qid: predicted_level(row)
        for qid, row in baseline_rows.items()
        if predicted_level(row)
    }
    merged_predictions = dict(baseline_predictions)
    merged_predictions.update(
        {
            qid: predicted_level(row)
            for qid, row in candidate_rows.items()
            if predicted_level(row)
        }
    )

    common = sorted(set(labels) & set(baseline_predictions))
    selected_common = sorted(set(common) & set(candidate_rows))
    improved = worsened = changed = unchanged = 0
    transitions: Counter[str] = Counter()
    cases: List[Dict[str, Any]] = []
    for qid in selected_common:
        truth = labels[qid]
        before = baseline_predictions[qid]
        after = merged_predictions[qid]
        before_correct = before == truth
        after_correct = after == truth
        improved += int(not before_correct and after_correct)
        worsened += int(before_correct and not after_correct)
        changed += int(before != after)
        unchanged += int(before == after)
        transitions[f"{before} -> {after}"] += 1
        if before != after or before != truth:
            candidate = candidate_rows[qid]
            cases.append({
                "question_id": qid,
                "teacher_level": truth,
                "baseline_level": before,
                "candidate_level": after,
                "effect": (
                    "improved" if not before_correct and after_correct else
                    "worsened" if before_correct and not after_correct else
                    "unchanged_error" if not after_correct else "changed_but_correct"
                ),
                "image_input_used": bool(candidate.get("image_input_used")),
                "image_input_route_reasons": candidate.get("image_input_route_reasons") or [],
                "postprocess_actions": candidate.get("postprocess_actions") or [],
            })

    baseline_metrics = classification_metrics(labels, baseline_predictions)
    candidate_metrics = classification_metrics(labels, merged_predictions)
    selected_labels = {qid: labels[qid] for qid in selected_common}
    selected_before = {qid: baseline_predictions[qid] for qid in selected_common}
    selected_after = {qid: merged_predictions[qid] for qid in selected_common}
    report = {
        "experiment": "first-pass-conditional-image-ab-v1",
        "label_source": args.annotations,
        "baseline_source": args.baseline,
        "candidate_source": args.candidate,
        "baseline": baseline_metrics,
        "candidate_with_unselected_baseline_reuse": candidate_metrics,
        "delta": {
            "correct": candidate_metrics["correct"] - baseline_metrics["correct"],
            "accuracy_points": round(
                (candidate_metrics["accuracy"] - baseline_metrics["accuracy"]) * 100,
                4,
            ),
            "improved": improved,
            "worsened": worsened,
            "net_improvement": improved - worsened,
        },
        "selected_subset": {
            "expected": len(candidate_rows),
            "evaluated": len(selected_common),
            "changed": changed,
            "unchanged": unchanged,
            "baseline": classification_metrics(selected_labels, selected_before),
            "candidate": classification_metrics(selected_labels, selected_after),
            "transitions": dict(transitions.most_common()),
        },
        "candidate_usage": {
            "image_input_used_count": sum(
                bool(row.get("image_input_used")) for row in candidate_rows.values()
            ),
            "prompt_tokens": sum(int(row.get("api_prompt_tokens", 0) or 0) for row in candidate_rows.values()),
            "completion_tokens": sum(int(row.get("api_completion_tokens", 0) or 0) for row in candidate_rows.values()),
            "total_tokens": sum(int(row.get("api_total_tokens", 0) or 0) for row in candidate_rows.values()),
        },
        "changed_or_incorrect_selected_cases": cases,
    }
    if control_rows:
        merged_control = dict(baseline_predictions)
        merged_control.update(
            {
                qid: predicted_level(row)
                for qid, row in control_rows.items()
                if predicted_level(row)
            }
        )
        control_metrics = classification_metrics(labels, merged_control)
        paired_ids = sorted(set(labels) & set(control_rows) & set(candidate_rows))
        auto_improved = auto_worsened = same = different = 0
        for qid in paired_ids:
            truth = labels[qid]
            control_level = merged_control[qid]
            candidate_level = merged_predictions[qid]
            auto_improved += int(control_level != truth and candidate_level == truth)
            auto_worsened += int(control_level == truth and candidate_level != truth)
            same += int(control_level == candidate_level)
            different += int(control_level != candidate_level)
        report["control_source"] = args.control
        report["control_text_only_with_unselected_baseline_reuse"] = control_metrics
        report["candidate_image_vs_control_text_only"] = {
            "paired_count": len(paired_ids),
            "control_correct": sum(merged_control[qid] == labels[qid] for qid in paired_ids),
            "candidate_correct": sum(merged_predictions[qid] == labels[qid] for qid in paired_ids),
            "auto_improved": auto_improved,
            "auto_worsened": auto_worsened,
            "net_improvement": auto_improved - auto_worsened,
            "same_prediction": same,
            "different_prediction": different,
            "full_set_accuracy_delta_points": round(
                (candidate_metrics["accuracy"] - control_metrics["accuracy"]) * 100,
                4,
            ),
        }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="初中物理首轮条件传图 A/B 工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="抽取 auto 路由命中的实验题")
    prepare_parser.add_argument("--baseline", required=True)
    prepare_parser.add_argument("--output", required=True)
    prepare_parser.set_defaults(handler=prepare)

    evaluate_parser = subparsers.add_parser("evaluate", help="固定基线合并并评测候选结果")
    evaluate_parser.add_argument("--baseline", required=True)
    evaluate_parser.add_argument("--candidate", required=True)
    evaluate_parser.add_argument(
        "--control",
        help="可选：同一 auto 路由子集的 text-only 重跑结果，用于配对比较",
    )
    evaluate_parser.add_argument("--annotations", required=True)
    evaluate_parser.add_argument("--output", required=True)
    evaluate_parser.set_defaults(handler=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
