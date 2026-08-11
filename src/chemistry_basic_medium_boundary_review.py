# -*- coding: utf-8 -*-
"""对化学首轮结果中的“基础题 / 中等题”进行独立二阶段复核。

候选路由只追求召回，不直接改档。只有复核模型给出高置信度，
且明确命中可核验的中等题决定性证据时，才允许从基础题升一档。
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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiofiles
import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import chemistry_difficulty_rating_with_cache as rating  # noqa: E402


load_dotenv()

DEFAULT_REVIEW_PROMPT = ROOT / "prompts" / "初中化学基础中等边界复核提示词.txt"
LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
OUTPUT_LOCK = Lock()

DECISIVE_EVIDENCE = {
    "异质规则切换",
    "同一主题多性质维度综合辨析",
    "多类化学符号含义互扰",
    "多个规范主观表达",
    "连续实验或流程链",
    "标准误差原因与结果链",
}
ALL_EVIDENCE = DECISIVE_EVIDENCE | {
    "同一规则重复核验",
    "选项或对象数量多",
    "跨章节但都是低负担识记",
    "无中等决定性证据",
}


def extract_final_level(item: Dict[str, Any]) -> str:
    result = item.get("difficulty_rating")
    if isinstance(result, dict) and result.get("difficulty_level") in LEVELS:
        return str(result["difficulty_level"])
    raw = item.get("difficulty_level_raw")
    return str(raw) if raw in LEVELS else ""


def _int_metric(
    features: Dict[str, Any],
    metrics: Dict[str, Any],
    metric_name: str,
    fallback: int,
) -> int:
    try:
        return int(metrics.get(metric_name, fallback) or fallback)
    except (TypeError, ValueError):
        return fallback


def select_boundary_candidate(item: Dict[str, Any]) -> Dict[str, Any]:
    """宽召回路由：只复核已判基础、但存在整题广度或表达负担的题。"""
    current = extract_final_level(item)
    result = item.get("difficulty_rating") or {}
    features = result.get("features") if isinstance(result, dict) else {}
    metrics = result.get("observable_metrics") if isinstance(result, dict) else {}
    features = features if isinstance(features, dict) else {}
    metrics = metrics if isinstance(metrics, dict) else {}

    task_groups = features.get("task_groups") or []
    fallback_tasks = sum(
        int(group.get("count", 0) or 0)
        for group in task_groups
        if isinstance(group, dict)
    )
    task_count = _int_metric(
        features,
        metrics,
        "effective_task_count",
        fallback_tasks,
    )
    chain_steps = _int_metric(
        features,
        metrics,
        "longest_chain_steps",
        len(features.get("longest_solution_chain") or []),
    )
    rule_count = _int_metric(
        features,
        metrics,
        "rule_family_count",
        len(features.get("rule_families") or []),
    )
    topic_count = _int_metric(
        features,
        metrics,
        "curriculum_topic_count",
        len(features.get("curriculum_topics") or []),
    )
    response_count = len(features.get("response_operations") or [])

    reasons: List[str] = []
    if current == "基础题":
        if task_count >= 4:
            reasons.append("至少4个实质任务，需复核是否真实切换规则")
        if task_count >= 2 and topic_count >= 2:
            reasons.append("多课题任务，需复核是低负担并列还是异质辨析")
        if response_count >= 2:
            reasons.append("多处主观表达，需复核是否分别组织化学证据")
        if chain_steps >= 3 or rule_count >= 3:
            reasons.append("链长或规则数接近中等边界")

    return {
        "selected": bool(reasons),
        "current_level": current,
        "allowed_levels": ["基础题", "中等题"],
        "selection_reasons": reasons,
        "audit_metrics": {
            "effective_task_count": task_count,
            "longest_chain_steps": chain_steps,
            "rule_family_count": rule_count,
            "curriculum_topic_count": topic_count,
            "response_operation_count": response_count,
        },
    }


def canonical_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    return {
        "high": "高",
        "medium": "中",
        "mid": "中",
        "low": "低",
    }.get(text, str(value or "").strip())


def normalize_review_result(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    evidence = source.get("decisive_evidence")
    if not isinstance(evidence, list):
        evidence = [evidence] if evidence else []
    return {
        "review_level": str(source.get("review_level", "") or "").strip(),
        "confidence": canonical_confidence(source.get("confidence")),
        "decisive_evidence": [
            str(item).strip() for item in evidence if str(item or "").strip()
        ],
        "effective_task_summary": str(
            source.get("effective_task_summary", "") or ""
        ).strip(),
        "boundary_basis": str(source.get("boundary_basis", "") or "").strip(),
        "first_pass_issue": str(source.get("first_pass_issue", "") or "").strip(),
    }


def validate_review_result(
    result: Dict[str, Any],
    allowed_levels: Sequence[str],
) -> Optional[str]:
    if result.get("review_level") not in allowed_levels:
        return "review_level不在允许的基础/中等边界中"
    if result.get("confidence") not in {"高", "中", "低"}:
        return "confidence非法"
    evidence = result.get("decisive_evidence")
    if not isinstance(evidence, list) or not evidence:
        return "decisive_evidence不能为空"
    if len(evidence) != len(set(evidence)):
        return "decisive_evidence不能重复"
    if any(item not in ALL_EVIDENCE for item in evidence):
        return "decisive_evidence包含非法值"
    if result.get("review_level") == "中等题" and not (
        set(evidence) & DECISIVE_EVIDENCE
    ):
        return "判为中等题时缺少决定性中等证据"
    for field in (
        "effective_task_summary",
        "boundary_basis",
        "first_pass_issue",
    ):
        if not result.get(field):
            return f"缺少{field}"
    return None


def build_review_content(
    item: Dict[str, Any],
    candidate: Dict[str, Any],
) -> str:
    result = item.get("difficulty_rating") or {}
    first_pass = {
        "首轮等级": candidate["current_level"],
        "features": result.get("features", {}),
        "reasoning": result.get("reasoning", {}),
        "候选原因": candidate["selection_reasons"],
        "审计计数": candidate["audit_metrics"],
    }
    safe = rating.sanitize_question_data(item)
    return "\n\n".join(
        [
            "【题目与解析】\n" + rating.construct_question_content(safe),
            "【只允许的等级】\n基础题、中等题",
            "【首轮待审材料】\n"
            + json.dumps(first_pass, ensure_ascii=False, indent=2),
            "请严格按JSON格式输出，不得使用题号、来源或分布作证据。",
        ]
    )


def apply_review_to_item(
    item: Dict[str, Any],
    candidate: Dict[str, Any],
    review_result: Dict[str, Any],
    *,
    writeback: bool,
) -> Tuple[bool, str]:
    error = validate_review_result(
        review_result,
        candidate["allowed_levels"],
    )
    if error:
        return False, error
    if not writeback:
        return False, "复核仅审计，写回关闭"
    if review_result["review_level"] != "中等题":
        return False, "复核保持基础题"
    if review_result["confidence"] != "高":
        return False, "复核置信度未达到写回阈值"
    if not set(review_result["decisive_evidence"]) & DECISIVE_EVIDENCE:
        return False, "复核缺少决定性中等证据"

    difficulty_rating = item.setdefault("difficulty_rating", {})
    before = str(difficulty_rating.get("difficulty_level") or "基础题")
    difficulty_rating["difficulty_level"] = "中等题"
    rating.sync_coarse_difficulty(difficulty_rating)
    action = {
        "rule": "chemistry_basic_to_medium_boundary_review",
        "from": before,
        "to": "中等题",
        "reason": review_result["boundary_basis"],
        "evidence": copy.deepcopy(review_result["decisive_evidence"]),
    }
    item.setdefault("postprocess_actions", []).append(action)
    difficulty_rating.setdefault("postprocess_actions", []).append(
        copy.deepcopy(action)
    )
    return True, "高置信度且有决定性证据，升为中等题"


async def call_review_model(
    prompt: str,
    content: str,
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
                        for part in output_item.get("content", []):
                            if part.get("type") == "output_text":
                                output_text = part.get("text", "")
                    parsed = normalize_review_result(
                        rating.parse_model_response(output_text)
                    )
                    error = validate_review_result(
                        parsed,
                        ["基础题", "中等题"],
                    )
                    if error is None:
                        return (
                            parsed,
                            time.time() - started,
                            prompt_tokens,
                            completion_tokens,
                            total_tokens,
                            "",
                        )
                    last_error = error
                else:
                    last_error = f"HTTP {response.status}: {(await response.text())[:300]}"
                    if response.status < 500 and response.status != 429:
                        break
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2**attempt + random.random())
    return (
        {},
        time.time() - started,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        last_error or "复核响应无效",
    )


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path}:{line_number}缺少question_id")
            if question_id in seen:
                raise ValueError(f"{path}存在重复question_id={question_id}")
            seen.add(question_id)
            rows.append(item)
    return rows


def get_processed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    return {
        str(item.get("question_id"))
        for item in load_jsonl(path)
        if item.get("question_id") is not None
    }


async def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    async with OUTPUT_LOCK:
        async with aiofiles.open(path, "a", encoding="utf-8") as handle:
            await handle.write(json.dumps(item, ensure_ascii=False) + "\n")


async def process_item(
    source_item: Dict[str, Any],
    prompt: str,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    output_path: str,
    error_path: str,
    model_name: str,
    temperature: Optional[float],
    retries: int,
    timeout_sec: int,
    writeback: bool,
    review_allowed: bool = True,
) -> None:
    async with semaphore:
        item = copy.deepcopy(source_item)
        candidate = select_boundary_candidate(item)
        before = extract_final_level(item)
        review_result: Dict[str, Any] = {}
        elapsed = prompt_tokens = completion_tokens = total_tokens = 0
        error = ""
        applied = False
        reason = "未进入基础/中等复核"
        if candidate["selected"] and not review_allowed:
            reason = "超过max-review-calls，未调用复核模型"
        elif candidate["selected"]:
            (
                review_result,
                elapsed,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                error,
            ) = await call_review_model(
                prompt,
                build_review_content(item, candidate),
                session,
                model_name,
                temperature,
                retries,
                timeout_sec,
            )
            if review_result:
                applied, reason = apply_review_to_item(
                    item,
                    candidate,
                    review_result,
                    writeback=writeback,
                )
            else:
                reason = "复核请求失败，保持首轮等级"

        after = extract_final_level(item) or before
        item["basic_medium_boundary_review"] = {
            "enabled": True,
            "selected": candidate["selected"],
            "selection_reasons": candidate["selection_reasons"],
            "audit_metrics": candidate["audit_metrics"],
            "allowed_levels": candidate["allowed_levels"],
            "result": review_result,
            "writeback_enabled": writeback,
            "applied": applied,
            "from": before,
            "to": after,
            "decision_reason": reason,
            "error": error,
            "api_time_use": round(float(elapsed), 2),
            "api_prompt_tokens": int(prompt_tokens),
            "api_completion_tokens": int(completion_tokens),
            "api_total_tokens": int(total_tokens),
        }
        item["difficulty_level_before_basic_medium_review"] = before
        item["difficulty_level_after_basic_medium_review"] = after
        item["basic_medium_boundary_review_applied"] = applied
        item["pipeline_api_total_tokens"] = int(
            item.get("api_total_tokens", 0) or 0
        ) + int(total_tokens)
        await append_jsonl(output_path, item)
        if error:
            await append_jsonl(
                error_path,
                {
                    "question_id": item.get("question_id"),
                    "basic_medium_boundary_review_error": error,
                    "candidate": candidate,
                },
            )


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="初中化学基础题/中等题独立边界复核"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--error", required=True)
    parser.add_argument("-p", "--prompt", default=str(DEFAULT_REVIEW_PROMPT))
    parser.add_argument("-c", "--concurrency", type=int, default=30)
    parser.add_argument("-t", "--timeout", type=int, default=180)
    parser.add_argument("-r", "--retries", type=int, default=3)
    parser.add_argument(
        "--model",
        default=os.getenv("CHEMISTRY_BOUNDARY_REVIEW_MODEL_NAME")
        or rating.MODEL_NAME,
    )
    parser.add_argument(
        "--temperature",
        default=os.getenv("CHEMISTRY_BOUNDARY_REVIEW_TEMPERATURE", "0"),
    )
    parser.add_argument(
        "--writeback",
        choices=("0", "1"),
        default=os.getenv("CHEMISTRY_BASIC_MEDIUM_REVIEW_WRITEBACK", "0"),
    )
    parser.add_argument("--max-review-calls", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    selected = [row for row in rows if select_boundary_candidate(row)["selected"]]
    print(
        f"输入{len(rows)}题，基础/中等边界复核候选{len(selected)}题，"
        f"写回={'开启' if args.writeback == '1' else '关闭'}"
    )
    if args.dry_run:
        return
    if not os.path.exists(args.prompt):
        raise FileNotFoundError(f"找不到复核提示词: {args.prompt}")
    prompt = Path(args.prompt).read_text(encoding="utf-8")

    review_ids: Optional[set[str]] = None
    if args.max_review_calls is not None:
        review_ids = {
            str(row.get("question_id"))
            for row in selected[: max(0, args.max_review_calls)]
        }

    processed = get_processed_ids(args.output)
    pending = [row for row in rows if str(row.get("question_id")) not in processed]
    temperature = rating.resolve_temperature(args.model, args.temperature)
    semaphore = Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    progress = tqdm(
        total=len(pending),
        unit="item",
        desc="Chemistry Basic/Medium Review",
    )
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                process_item(
                    row,
                    prompt,
                    session,
                    semaphore,
                    args.output,
                    args.error,
                    args.model,
                    temperature,
                    args.retries,
                    args.timeout,
                    args.writeback == "1",
                    review_ids is None
                    or str(row.get("question_id")) in review_ids,
                )
            )
            for row in pending
        ]
        for task in asyncio.as_completed(tasks):
            await task
            progress.update(1)
    progress.close()
    print(f"复核完成: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    asyncio.run(main())
