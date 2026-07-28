# -*- coding: utf-8 -*-
"""合并多次 Doubao Lite 独立评级结果。

本工具不再调用裁判模型。每次首轮评级独立运行完整 Prompt 与后处理，
随后按 question_id 做确定性多数票；票数相同时使用五档序数中位数。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import physics_difficulty_rating_with_cache as rating


LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
PIPELINE_VERSION = "doubao-lite-self-consistency-v1"


def load_jsonl(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    order: List[str] = []
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "").strip()
            if not question_id:
                raise ValueError(f"{path}:{line_number} 缺少 question_id")
            if question_id in rows:
                raise ValueError(f"{path} question_id 重复：{question_id}")
            level = extract_level(item)
            if level not in LEVEL_INDEX:
                raise ValueError(
                    f"{path}:{line_number} question_id={question_id} 缺少合法最终等级"
                )
            order.append(question_id)
            rows[question_id] = item
    return order, rows


def extract_level(item: Dict[str, Any]) -> str:
    result = item.get("difficulty_rating")
    return str(result.get("difficulty_level") or "") if isinstance(result, dict) else ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def choose_level(predictions: Sequence[str]) -> Tuple[str, str]:
    if not predictions:
        raise ValueError("至少需要一个预测")
    if any(level not in LEVEL_INDEX for level in predictions):
        raise ValueError(f"存在非法等级：{predictions}")
    counts = Counter(predictions)
    maximum = max(counts.values())
    leaders = [level for level, count in counts.items() if count == maximum]
    if len(leaders) == 1:
        method = "unanimous" if len(counts) == 1 else "majority"
        return leaders[0], method

    ordered = sorted(LEVEL_INDEX[level] for level in predictions)
    median_index = ordered[len(ordered) // 2]
    median_level = LEVELS[median_index]
    return median_level, "median_tiebreak"


def representative_run(
    rows: Sequence[Dict[str, Any]],
    chosen_level: str,
) -> int:
    candidates = [
        index for index, item in enumerate(rows) if extract_level(item) == chosen_level
    ]
    if not candidates:
        raise ValueError(f"没有输出 {chosen_level} 的代表运行")
    # 优先采用没有后处理动作的同档结果，避免理由只描述一次偶发规则触发。
    return min(
        candidates,
        key=lambda index: (
            bool(rows[index].get("postprocess_actions")),
            index,
        ),
    )


def merge_question(
    question_id: str,
    rows: Sequence[Dict[str, Any]],
    run_paths: Sequence[Path],
    run_hashes: Sequence[str],
) -> Dict[str, Any]:
    predictions = [extract_level(item) for item in rows]
    chosen_level, method = choose_level(predictions)
    selected_index = representative_run(rows, chosen_level)
    output = copy.deepcopy(rows[selected_index])
    result = output.get("difficulty_rating")
    if not isinstance(result, dict):
        raise ValueError(f"question_id={question_id} 缺少 difficulty_rating")
    result["difficulty_level"] = chosen_level
    rating.sync_coarse_difficulty(result)
    rating.sync_final_adjacent_reasoning(result)

    counts = Counter(predictions)
    output["multi_call_final_level"] = chosen_level
    output["lite_self_consistency"] = {
        "pipeline_version": PIPELINE_VERSION,
        "run_count": len(rows),
        "run_predictions": predictions,
        "vote_counts": {level: counts[level] for level in LEVELS if counts[level]},
        "unanimous": len(counts) == 1,
        "decision_method": method,
        "representative_run": selected_index + 1,
        "run_files": [str(path) for path in run_paths],
        "run_sha256": list(run_hashes),
    }
    output["api_prompt_tokens"] = sum(
        int(item.get("api_prompt_tokens") or 0) for item in rows
    )
    output["api_completion_tokens"] = sum(
        int(item.get("api_completion_tokens") or 0) for item in rows
    )
    output["api_total_tokens"] = sum(
        int(item.get("api_total_tokens") or 0) for item in rows
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="合并3次或更多 Doubao Lite 独立评级结果"
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="独立评级 JSONL，可重复指定；建议使用3或5次",
    )
    parser.add_argument("-o", "--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_paths = [Path(path).resolve() for path in args.run]
    if len(run_paths) < 3:
        raise ValueError("自一致性聚合至少需要3次独立结果")
    if len(run_paths) % 2 == 0:
        raise ValueError("建议使用奇数次独立结果，当前输入为偶数")

    loaded = [load_jsonl(path) for path in run_paths]
    first_order = loaded[0][0]
    expected_ids = set(first_order)
    for path, (order, _rows) in zip(run_paths[1:], loaded[1:]):
        current_ids = set(order)
        if current_ids != expected_ids:
            missing = len(expected_ids - current_ids)
            extra = len(current_ids - expected_ids)
            raise ValueError(
                f"{path} 与首个结果 question_id 不一致：缺少 {missing}，多出 {extra}"
            )

    run_hashes = [sha256_file(path) for path in run_paths]
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    distribution: Counter[str] = Counter()
    method_distribution: Counter[str] = Counter()
    unanimous_count = 0

    with output_path.open("w", encoding="utf-8") as handle:
        for question_id in first_order:
            question_rows = [rows[question_id] for _order, rows in loaded]
            output = merge_question(
                question_id,
                question_rows,
                run_paths,
                run_hashes,
            )
            level = str(output["multi_call_final_level"])
            method = str(output["lite_self_consistency"]["decision_method"])
            distribution[level] += 1
            method_distribution[method] += 1
            unanimous_count += bool(output["lite_self_consistency"]["unanimous"])
            handle.write(json.dumps(output, ensure_ascii=False) + "\n")

    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "questions": len(first_order),
        "run_count": len(run_paths),
        "unanimous_count": unanimous_count,
        "disagreement_count": len(first_order) - unanimous_count,
        "decision_methods": dict(method_distribution),
        "prediction_distribution": dict(distribution),
        "run_files": [str(path) for path in run_paths],
        "run_sha256": run_hashes,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    summary_path = output_path.with_suffix(output_path.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("本实验用于检验：只使用多次 Doubao Lite 独立首轮评级和确定性多数票，")
    print("能否在不引入低可靠裁判模型的情况下提高稳定性和准确率。")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
