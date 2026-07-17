# -*- coding: utf-8 -*-
"""对单次首轮物理难度结果进行可审计的第二阶段边界复核。

不做多次投票。可直接输入某一次完整结果，或由回归脚本导出的该次
全部错题；复核器独立判断唯一等级，或声明相邻两档均可接受。
高置信度且合法的复核结论最多调整一档。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import sys
import time
from asyncio import Lock, Semaphore
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiofiles
import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import physics_difficulty_rating_with_cache as rating  # noqa: E402


load_dotenv()

DEFAULT_REVIEW_PROMPT = ROOT / "prompts" / "初中物理难度边界复核提示词.txt"
LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
VALID_REVIEW_MODES = {"selective", "broad", "all"}
OUTPUT_LOCK = Lock()


STRUCTURE_SCORE_MAPS: Dict[str, Dict[str, int]] = {
    "step_count": {"1-2步": 0, "3-5步": 2, "6-8步": 4, "9-12步": 6, "12步以上": 7},
    "formula_count": {"0-1个": 0, "2-3个": 1, "4-6个": 3, "7个以上": 4},
    "calculation_complexity": {"口算或直接判断": 0, "简单笔算": 1, "多公式联立": 3, "复杂方程或范围计算": 4},
    "reasoning_chain": {"直接套用": 0, "简单因果推理": 1, "多层因果推理": 3, "逆向推理或临界分析": 4},
    "state_count": {"单状态": 0, "双状态": 1, "多状态": 3, "连续变化或临界状态": 4},
    "constraint_count": {"无约束": 0, "单一约束": 1, "多约束": 3},
    "variable_relation": {"无变量关系": 0, "简单正反比": 1, "图像函数关系": 2, "多变量耦合关系": 4},
    "experiment_requirement": {"无": 0, "基础操作或读数": 1, "控制变量或故障分析": 2, "方案设计或误差评价": 4},
    "graph_table_requirement": {"无": 0, "直接读数": 1, "多组比较归纳": 2, "图像反推或外推": 4},
    "knowledge_count": {"1个": 0, "2-3个": 1, "4个及以上": 2},
    "subquestion_dependency": {"无多问": 0, "多问但相互独立": 1, "多问且层层递进": 3},
    "cross_module": {"同一模块内部": 0, "跨模块综合": 2},
}


def load_jsonl(path: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    order: List[str] = []
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON: {exc}") from exc
            question_id = item.get("question_id")
            if question_id is None:
                raise ValueError(f"{path}:{line_number} 缺少 question_id")
            key = str(question_id)
            if key in rows:
                raise ValueError(f"{path} 存在重复 question_id={key}")
            level = extract_final_level(item)
            if level not in LEVEL_INDEX:
                raise ValueError(f"{path}:{line_number} 缺少合法最终等级")
            order.append(key)
            rows[key] = item
    return order, rows


def extract_final_level(item: Dict[str, Any]) -> Optional[str]:
    difficulty_rating = item.get("difficulty_rating")
    if isinstance(difficulty_rating, dict):
        value = difficulty_rating.get("difficulty_level")
        if value in LEVEL_INDEX:
            return str(value)
    value = item.get("difficulty_level_raw")
    return str(value) if value in LEVEL_INDEX else None


def adjacent_review_levels(current: str) -> List[str]:
    index = LEVEL_INDEX[current]
    start = max(0, index - 1)
    end = min(len(LEVELS), index + 2)
    return LEVELS[start:end]


def review_levels_for_item(item: Dict[str, Any], current: str) -> List[str]:
    """错题复核覆盖模型、参考标签及其相邻边界；普通复核只看相邻档。"""
    reference = str(item.get("adjudication_reference_level") or "").strip()
    if reference not in LEVEL_INDEX:
        return adjacent_review_levels(current)
    low = max(0, min(LEVEL_INDEX[current], LEVEL_INDEX[reference]) - 1)
    high = min(len(LEVELS) - 1, max(LEVEL_INDEX[current], LEVEL_INDEX[reference]) + 1)
    return LEVELS[low : high + 1]


def structural_score(features: Dict[str, Any]) -> int:
    return sum(mapping.get(str(features.get(field, "")), 0) for field, mapping in STRUCTURE_SCORE_MAPS.items())


def prepare_source_item(item: Dict[str, Any], source_file: str) -> Dict[str, Any]:
    current = extract_final_level(item)
    if current not in LEVEL_INDEX:
        raise ValueError("源结果中存在非法等级")
    merged = copy.deepcopy(item)
    before = copy.deepcopy(merged.get("difficulty_rating") or {})
    before["difficulty_level"] = current
    rating.sync_coarse_difficulty(before)
    merged["difficulty_rating"] = before
    merged["boundary_review_source_file"] = os.path.basename(source_file)
    merged["boundary_review_source_prediction"] = current
    merged["boundary_review_source_api_total_tokens"] = int(item.get("api_total_tokens", 0) or 0)
    merged["boundary_review_source_postprocess_actions"] = copy.deepcopy(
        item.get("postprocess_actions") or []
    )
    merged["boundary_review_input_mode"] = "single_no_vote"
    merged["boundary_review_any_postprocess_adjustment"] = bool(item.get("postprocess_actions"))
    merged["difficulty_rating_before_review"] = copy.deepcopy(before)
    return merged


def select_boundary_candidate(item: Dict[str, Any], mode: str = "selective") -> Dict[str, Any]:
    if mode not in VALID_REVIEW_MODES:
        raise ValueError(f"不支持的 review mode: {mode}")
    current = extract_final_level(item)
    if current not in LEVEL_INDEX:
        raise ValueError("待复核结果缺少合法等级")
    difficulty_rating = item.get("difficulty_rating") or {}
    features = difficulty_rating.get("features") if isinstance(difficulty_rating, dict) else {}
    features = features if isinstance(features, dict) else {}
    score = structural_score(features)
    reasons: List[str] = []

    if item.get("boundary_review_any_postprocess_adjustment") or item.get("postprocess_actions"):
        reasons.append("首轮发生过相邻档后处理调整")

    if mode in {"selective", "broad", "all"}:
        if mode == "all":
            reasons.append("全量复核模式")
        else:
            thresholds = {
                "selective": {
                    "送分题": (None, 3),
                    "基础题": (1, 7),
                    "中等题": (5, 15),
                    "拔高题": (11, 25),
                    "压轴题": (21, None),
                },
                "broad": {
                    "送分题": (None, 2),
                    "基础题": (2, 6),
                    "中等题": (7, 13),
                    "拔高题": (13, 22),
                    "压轴题": (24, None),
                },
            }[mode]
            low, high = thresholds[current]
            if current == "送分题" and high is not None and score >= high:
                reasons.append("送分档结构负担偏高")
            elif current == "压轴题" and low is not None and score <= low:
                reasons.append("压轴档结构证据偏弱")
            else:
                if low is not None and score <= low:
                    reasons.append(f"{current}结构分数靠近低一档")
                if high is not None and score >= high:
                    reasons.append(f"{current}结构分数靠近高一档")

    return {
        "selected": bool(reasons),
        "current_level": current,
        "allowed_levels": review_levels_for_item(item, current),
        "structural_score": score,
        "reasons": reasons,
    }


def build_review_content(item: Dict[str, Any], candidate: Dict[str, Any]) -> str:
    difficulty_rating = item.get("difficulty_rating") or {}
    first_pass_summary = {
        "本次首轮最终等级": item.get("boundary_review_source_prediction")
        or candidate["current_level"],
        "features": difficulty_rating.get("features", {}),
        "reasoning": difficulty_rating.get("reasoning", {}),
        "postprocess_actions": item.get("postprocess_actions", []),
        "本次首轮后处理动作": item.get("boundary_review_source_postprocess_actions", []),
        "边界筛选原因": candidate["reasons"],
        "结构审计分数": candidate["structural_score"],
    }
    return "\n\n".join(
        [
            "【当前题目】\n" + rating.construct_question_content(rating.sanitize_question_data(item)),
            "【允许等级】\n" + "、".join(candidate["allowed_levels"]),
            "【首轮最终等级】\n" + candidate["current_level"],
            "【首轮待审信息】\n" + json.dumps(first_pass_summary, ensure_ascii=False, indent=2),
            "请严格按复核 JSON 格式输出。",
        ]
    )


def canonical_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"高", "high"}:
        return "高"
    if text in {"中", "medium", "mid"}:
        return "中"
    if text in {"低", "low"}:
        return "低"
    return ""


def normalize_review_result(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    review_level = source.get("review_level") or source.get("difficulty_level")
    acceptable_levels = source.get("acceptable_levels")
    if not isinstance(acceptable_levels, list):
        acceptable_levels = [review_level] if review_level else []
    return {
        "review_level": str(review_level or "").strip(),
        "acceptable_levels": [str(level).strip() for level in acceptable_levels if str(level).strip()],
        "boundary_status": str(source.get("boundary_status", "") or "").strip(),
        "confidence": canonical_confidence(source.get("confidence")),
        "decision": str(source.get("decision", "") or "").strip(),
        "effective_task_summary": str(source.get("effective_task_summary", "") or "").strip(),
        "boundary_basis": str(source.get("boundary_basis", "") or "").strip(),
        "first_pass_issue": str(source.get("first_pass_issue", "") or "").strip(),
    }


def validate_review_result(result: Dict[str, Any], allowed_levels: Sequence[str]) -> Optional[str]:
    if result.get("review_level") not in allowed_levels:
        return "review_level 不在允许等级中"
    acceptable_levels = result.get("acceptable_levels")
    if not isinstance(acceptable_levels, list) or not acceptable_levels:
        return "acceptable_levels 不能为空"
    if len(acceptable_levels) > 2 or len(set(acceptable_levels)) != len(acceptable_levels):
        return "acceptable_levels 只能包含一个或两个不重复等级"
    if any(level not in allowed_levels for level in acceptable_levels):
        return "acceptable_levels 含允许范围外等级"
    if result.get("review_level") not in acceptable_levels:
        return "review_level 必须包含在 acceptable_levels 中"
    if len(acceptable_levels) == 2:
        indices = sorted(LEVEL_INDEX[level] for level in acceptable_levels)
        if indices[1] - indices[0] != 1:
            return "acceptable_levels 的两个等级必须相邻"
    if result.get("boundary_status") not in {"明确归档", "相邻边界均可"}:
        return "boundary_status 非法"
    if result.get("boundary_status") == "明确归档" and len(acceptable_levels) != 1:
        return "明确归档时 acceptable_levels 只能有一个等级"
    if result.get("boundary_status") == "相邻边界均可" and len(acceptable_levels) != 2:
        return "相邻边界均可时 acceptable_levels 必须有两个等级"
    if result.get("confidence") not in {"高", "中", "低"}:
        return "confidence 非法"
    for field in ["effective_task_summary", "boundary_basis", "first_pass_issue"]:
        if not result.get(field):
            return f"缺少 {field}"
    return None


def classify_reference_disagreement(
    reference_level: str,
    first_pass_level: str,
    review_result: Dict[str, Any],
) -> str:
    """将错题区分为模型错误、参考标签问题或真实相邻边界。"""
    acceptable = set(review_result.get("acceptable_levels") or [])
    if reference_level not in LEVEL_INDEX:
        return "无参考标签"
    reference_ok = reference_level in acceptable
    model_ok = first_pass_level in acceptable
    if reference_ok and model_ok:
        return "相邻边界均可"
    if reference_ok and not model_ok:
        return "模型确实误判"
    if model_ok and not reference_ok:
        return "参考标签需修订"
    return "双方均需修订"


def should_apply_review(
    current_level: str,
    review_result: Dict[str, Any],
    allowed_levels: Sequence[str],
    accepted_confidences: Iterable[str],
) -> Tuple[bool, str]:
    error = validate_review_result(review_result, allowed_levels)
    if error:
        return False, error
    target = review_result["review_level"]
    if target == current_level:
        return False, "复核保持首轮等级"
    if abs(LEVEL_INDEX[target] - LEVEL_INDEX[current_level]) != 1:
        return False, "复核调整超过一档"
    if review_result["confidence"] not in set(accepted_confidences):
        return False, "复核置信度未达到自动调整阈值"
    return True, "高置信度相邻档复核调整"


async def call_review_model(
    prompt: str,
    content: str,
    allowed_levels: Sequence[str],
    session: aiohttp.ClientSession,
    model_name: str,
    temperature: Optional[float],
    retries: int,
    timeout_sec: int,
) -> Tuple[Dict[str, Any], float, int, int, int, str]:
    started = time.time()
    prompt_tokens = completion_tokens = total_tokens = 0
    last_error = ""
    for attempt in range(retries):
        payload: Dict[str, Any] = {
            "model": model_name,
            "input": [{"role": "user", "content": prompt + "\n\n" + content}],
            "thinking": {"type": "disabled"},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            async with session.post(
                f"{rating.BASE_URL}responses",
                json=payload,
                headers={"Authorization": f"Bearer {rating.API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                if response.status == 200:
                    body = await response.json()
                    usage = body.get("usage", {})
                    prompt_tokens += int(usage.get("input_tokens", 0) or 0)
                    completion_tokens += int(usage.get("output_tokens", 0) or 0)
                    total_tokens += int(usage.get("total_tokens", 0) or 0)
                    output_text = ""
                    for output_item in body.get("output", []):
                        if output_item.get("type") != "message":
                            continue
                        for output_content in output_item.get("content", []):
                            if output_content.get("type") == "output_text":
                                output_text = output_content.get("text", "")
                    parsed = normalize_review_result(rating.parse_model_response(output_text))
                    validation_error = validate_review_result(parsed, allowed_levels)
                    if not validation_error:
                        return parsed, time.time() - started, prompt_tokens, completion_tokens, total_tokens, ""
                    last_error = validation_error
                else:
                    error_text = await response.text()
                    last_error = f"HTTP {response.status}: {error_text[:300]}"
                    if response.status < 500 and response.status != 429:
                        break
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = str(exc)
        except Exception as exc:  # 保留单题失败并继续批处理。
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2 ** attempt + random.random())
    return {}, time.time() - started, prompt_tokens, completion_tokens, total_tokens, last_error or "复核响应无效"


async def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    async with OUTPUT_LOCK:
        async with aiofiles.open(path, "a", encoding="utf-8") as handle:
            await handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def get_processed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    processed: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                processed.add(str(json.loads(line)["question_id"]))
            except Exception:
                continue
    return processed


def apply_review_to_item(
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    review_result: Dict[str, Any],
    accepted_confidences: Iterable[str],
) -> Tuple[bool, str]:
    current = candidate["current_level"]
    apply, reason = should_apply_review(current, review_result, candidate["allowed_levels"], accepted_confidences)
    if apply:
        difficulty_rating = item.setdefault("difficulty_rating", {})
        difficulty_rating["difficulty_level"] = review_result["review_level"]
        rating.sync_coarse_difficulty(difficulty_rating)
    return apply, reason


async def process_item(
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    prompt: str,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    output_path: str,
    error_path: str,
    model_name: str,
    temperature: Optional[float],
    accepted_confidences: Sequence[str],
    retries: int,
    timeout_sec: int,
) -> None:
    async with semaphore:
        review_result: Dict[str, Any] = {}
        elapsed = prompt_tokens = completion_tokens = total_tokens = 0
        error = ""
        applied = False
        decision_reason = "未进入第二阶段复核"
        if candidate["selected"]:
            content = build_review_content(item, candidate)
            review_result, elapsed, prompt_tokens, completion_tokens, total_tokens, error = await call_review_model(
                prompt,
                content,
                candidate["allowed_levels"],
                session,
                model_name,
                temperature,
                retries,
                timeout_sec,
            )
            if review_result:
                applied, decision_reason = apply_review_to_item(item, candidate, review_result, accepted_confidences)
            else:
                decision_reason = "复核请求失败，保持首轮等级"

        before = candidate["current_level"]
        after = extract_final_level(item) or before
        reference_level = str(item.get("adjudication_reference_level") or "").strip()
        disagreement_classification = (
            classify_reference_disagreement(reference_level, before, review_result)
            if review_result
            else "复核失败或未复核"
        )
        source_token_values = item.get("boundary_review_source_api_total_tokens")
        if not isinstance(source_token_values, list):
            source_token_values = [item.get("api_total_tokens", 0)]
        source_tokens = sum(int(value or 0) for value in source_token_values)
        item["boundary_review"] = {
            "enabled": True,
            "selected": candidate["selected"],
            "mode": item.get("boundary_review_mode"),
            "model": model_name,
            "temperature": temperature,
            "selection_reasons": candidate["reasons"],
            "structural_score": candidate["structural_score"],
            "allowed_levels": candidate["allowed_levels"],
            "result": review_result,
            "applied": applied,
            "from": before,
            "to": after,
            "reference_level": reference_level,
            "disagreement_classification": disagreement_classification,
            "decision_reason": decision_reason,
            "error": error,
            "api_time_use": round(float(elapsed), 2),
            "api_prompt_tokens": int(prompt_tokens),
            "api_completion_tokens": int(completion_tokens),
            "api_total_tokens": int(total_tokens),
        }
        item["difficulty_level_before_review"] = before
        item["difficulty_level_after_review"] = after
        item["boundary_review_applied"] = applied
        item["boundary_disagreement_classification"] = disagreement_classification
        item["pipeline_api_total_tokens"] = source_tokens + int(total_tokens)
        await append_jsonl(output_path, item)
        if error:
            await append_jsonl(
                error_path,
                {"question_id": item.get("question_id"), "boundary_review_error": error, "candidate": candidate},
            )


def prepare_items(input_path: str, mode: str) -> Tuple[List[Dict[str, Any]], Counter[str]]:
    base_order, source_rows = load_jsonl(input_path)
    prepared: List[Dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for question_id in base_order:
        merged = prepare_source_item(source_rows[question_id], input_path)
        merged["boundary_review_mode"] = mode
        candidate = select_boundary_candidate(merged, mode)
        merged["_boundary_candidate"] = candidate
        if candidate["selected"]:
            stats["selected"] += 1
            for reason in candidate["reasons"]:
                stats[reason] += 1
        else:
            stats["not_selected"] += 1
        prepared.append(merged)
    return prepared, stats


async def main() -> None:
    parser = argparse.ArgumentParser(description="初中物理难度第二阶段边界复核")
    parser.add_argument("-i", "--input", required=True, help="单次首轮 JSONL；不执行跨运行投票")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--error", required=True)
    parser.add_argument("-p", "--prompt", default=str(DEFAULT_REVIEW_PROMPT))
    parser.add_argument("-c", "--concurrency", type=int, default=20)
    parser.add_argument("-t", "--timeout", type=int, default=180)
    parser.add_argument("-r", "--retries", type=int, default=3)
    parser.add_argument("--review-mode", choices=sorted(VALID_REVIEW_MODES), default="selective")
    parser.add_argument("--model", default=os.getenv("BOUNDARY_REVIEW_MODEL_NAME") or rating.MODEL_NAME)
    parser.add_argument("--temperature", default=os.getenv("BOUNDARY_REVIEW_TEMPERATURE", ""))
    parser.add_argument("--accept-confidence", default=os.getenv("BOUNDARY_REVIEW_ACCEPT_CONFIDENCE", "高"))
    parser.add_argument("--max-review-calls", type=int, default=None, help="烟测用；只调用前 N 个候选，其余保持共识等级")
    parser.add_argument("--dry-run", action="store_true", help="只统计候选，不发请求、不写输出")
    args = parser.parse_args()

    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"找不到复核提示词: {args.prompt}")
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    items, stats = prepare_items(args.input, args.review_mode)
    print(f"单次输入: {args.input}，题目数: {len(items)}")
    print(f"复核模式: {args.review_mode}，候选数: {stats['selected']}，跳过数: {stats['not_selected']}")
    print("候选原因:", json.dumps(dict(stats), ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        return

    selected_seen = 0
    for item in items:
        candidate = item["_boundary_candidate"]
        if candidate["selected"]:
            selected_seen += 1
            if args.max_review_calls is not None and selected_seen > args.max_review_calls:
                candidate["selected"] = False
                candidate["reasons"] = candidate["reasons"] + ["超过 max-review-calls，未调用复核模型"]

    accepted_confidences = [value.strip() for value in args.accept_confidence.split(",") if value.strip()]
    if any(value not in {"高", "中", "低"} for value in accepted_confidences):
        raise ValueError("--accept-confidence 只能由高、中、低组成")
    temperature = rating.resolve_temperature(args.model, args.temperature)
    processed = get_processed_ids(args.output)
    pending = [item for item in items if str(item.get("question_id")) not in processed]
    print(f"已完成: {len(processed)}，待写入: {len(pending)}")
    print(f"复核模型: {args.model}，temperature={temperature}，自动调整置信度={accepted_confidences}")
    if not pending:
        return

    semaphore = Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    progress = tqdm(total=len(pending), unit="item", desc="Boundary Review Progress")
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for item in pending:
            candidate = item.pop("_boundary_candidate")
            tasks.append(
                asyncio.create_task(
                    process_item(
                        item,
                        candidate,
                        prompt,
                        session,
                        semaphore,
                        args.output,
                        args.error,
                        args.model,
                        temperature,
                        accepted_confidences,
                        args.retries,
                        args.timeout,
                    )
                )
            )
        for task in asyncio.as_completed(tasks):
            await task
            progress.update(1)
    progress.close()
    print(f"第二阶段复核完成: {os.path.abspath(args.output)}")
    print(f"复核错误日志: {os.path.abspath(args.error)}")


if __name__ == "__main__":
    started = time.time()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("收到中断信号，已停止复核。")
    except Exception as exc:
        print(f"边界复核失败: {exc}")
        raise
    finally:
        print(f"复核耗时: {round((time.time() - started) / 60, 2)} 分钟")
