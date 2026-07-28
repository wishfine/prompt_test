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
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import physics_difficulty_rating_with_cache as rating


LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
PIPELINE_VERSION = "doubao-lite-self-consistency-v1"
EASY_BOUNDARY_LEVELS = {"送分题", "基础题"}
MEASUREMENT_INSTRUMENT_PATTERN = re.compile(
    r"刻度尺|量筒|秒表|停表|温度计|电流表|电压表|弹簧测力计|天平|游码"
)


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


def extract_raw_level(item: Dict[str, Any]) -> str:
    level = str(item.get("difficulty_level_raw") or "")
    if level in LEVEL_INDEX:
        return level
    result = item.get("difficulty_rating_raw")
    if isinstance(result, dict):
        level = str(result.get("difficulty_level") or "")
    return level if level in LEVEL_INDEX else ""


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


def structured_easy_to_basic_evidence(
    rows: Sequence[Dict[str, Any]],
) -> List[str]:
    """提取送分/基础分歧中的显性应用证据。

    这里只检查判为基础题的独立运行。规则刻意排除“知识点多、空多、题干长”
    等弱信号，避免再次把纯教材事实和直接检索束整体吸入基础题。
    """

    evidence: List[str] = []
    for item in rows:
        if extract_level(item) != "基础题":
            continue
        result = item.get("difficulty_rating")
        features = result.get("features") if isinstance(result, dict) else {}
        if not isinstance(features, dict):
            features = {}

        if features.get("reasoning_chain") == "简单因果推理":
            evidence.append("基础题分歧结果识别到简单因果应用")

        information_carrier = features.get("information_carrier")
        if information_carrier == "单图识别":
            evidence.append("需要从单图恢复题目关系")
        elif information_carrier == "电路图":
            evidence.append("需要读取电路图关系")

        graph_requirement = features.get("graph_table_requirement")
        if graph_requirement in {"直接读数", "多组比较归纳", "图像反推或外推"}:
            evidence.append(f"存在{graph_requirement}任务")

        stem = str(item.get("stem") or "")
        if (
            features.get("experiment_requirement") == "基础操作或读数"
            and MEASUREMENT_INSTRUMENT_PATTERN.search(stem)
        ):
            evidence.append("涉及规范测量仪器操作或读数")

    # 保持稳定顺序并去重，便于审计与回归测试。
    return list(dict.fromkeys(evidence))


def merge_question(
    question_id: str,
    rows: Sequence[Dict[str, Any]],
    run_paths: Sequence[Path],
    run_hashes: Sequence[str],
    easy_requires_unanimity: bool = False,
    structured_easy_guard: bool = False,
) -> Dict[str, Any]:
    if easy_requires_unanimity and structured_easy_guard:
        raise ValueError("送分题全票保护与结构化送分保护不能同时启用")

    predictions = [extract_level(item) for item in rows]
    raw_predictions = [extract_raw_level(item) for item in rows]
    valid_raw_predictions = [level for level in raw_predictions if level in LEVEL_INDEX]
    raw_level = ""
    raw_method = ""
    if len(valid_raw_predictions) == len(rows):
        raw_level, raw_method = choose_level(valid_raw_predictions)
    majority_level, method = choose_level(predictions)
    chosen_level = majority_level
    calibration_actions: List[Dict[str, Any]] = []
    easy_boundary_disagreement = (
        majority_level == "送分题"
        and "基础题" in predictions
        and set(predictions).issubset(EASY_BOUNDARY_LEVELS)
    )
    if (
        easy_requires_unanimity
        and easy_boundary_disagreement
    ):
        chosen_level = "基础题"
        method = "easy_unanimity_guard"
        calibration_actions.append(
            {
                "rule": "easy_requires_unanimity",
                "from": "送分题",
                "to": "基础题",
                "evidence": ["三次Lite未一致判为送分题", "至少一次独立结果判为基础题"],
            }
        )
    elif structured_easy_guard and easy_boundary_disagreement:
        structured_evidence = structured_easy_to_basic_evidence(rows)
        if structured_evidence:
            chosen_level = "基础题"
            method = "structured_easy_guard"
            calibration_actions.append(
                {
                    "rule": "structured_easy_disagreement_guard",
                    "from": "送分题",
                    "to": "基础题",
                    "evidence": structured_evidence,
                }
            )
    selected_index = representative_run(rows, chosen_level)
    output = copy.deepcopy(rows[selected_index])
    result = output.get("difficulty_rating")
    if not isinstance(result, dict):
        raise ValueError(f"question_id={question_id} 缺少 difficulty_rating")
    result["difficulty_level"] = chosen_level
    rating.sync_coarse_difficulty(result)
    rating.sync_final_adjacent_reasoning(result)

    counts = Counter(predictions)
    ordered_counts = sorted(counts.values(), reverse=True)
    vote_margin = ordered_counts[0] - (ordered_counts[1] if len(ordered_counts) > 1 else 0)
    output["multi_call_final_level"] = chosen_level
    output["multi_call_raw_level"] = raw_level
    output["lite_self_consistency"] = {
        "pipeline_version": PIPELINE_VERSION,
        "run_count": len(rows),
        "run_predictions": predictions,
        "raw_run_predictions": raw_predictions,
        "raw_consensus_level": raw_level,
        "raw_decision_method": raw_method,
        "vote_counts": {level: counts[level] for level in LEVELS if counts[level]},
        "vote_margin": vote_margin,
        "candidate_levels": [level for level in LEVELS if counts[level]],
        "unanimous": len(counts) == 1,
        "decision_method": method,
        "majority_level_before_calibration": majority_level,
        "calibration_actions": calibration_actions,
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
    parser.add_argument(
        "--easy-requires-unanimity",
        action="store_true",
        help="送分/基础分歧中，只有全部独立结果均为送分题才输出送分题",
    )
    parser.add_argument(
        "--structured-easy-guard",
        action="store_true",
        help=(
            "送分/基础分歧中，仅当基础题结果识别到因果应用、读图、图表、"
            "电路图关系或规范测量证据时输出基础题"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.easy_requires_unanimity and args.structured_easy_guard:
        raise ValueError(
            "--easy-requires-unanimity 与 --structured-easy-guard 不能同时启用"
        )
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
                easy_requires_unanimity=args.easy_requires_unanimity,
                structured_easy_guard=args.structured_easy_guard,
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
        "easy_requires_unanimity": args.easy_requires_unanimity,
        "structured_easy_guard": args.structured_easy_guard,
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
