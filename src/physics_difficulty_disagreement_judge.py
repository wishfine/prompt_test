# -*- coding: utf-8 -*-
"""多次冻结评级后的匿名相邻边界裁判实验。

三次评级一致时直接保留；三次不一致时隐藏运行身份、票数和首轮结论，只把
相邻候选、题目、官方解析、边界标准和平衡真实样例发给裁判 API。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from asyncio import Lock, Semaphore
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiofiles
import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import physics_difficulty_boundary_review as boundary_review  # noqa: E402
import physics_difficulty_rating_with_cache as rating  # noqa: E402


load_dotenv()

PIPELINE_VERSION = "anonymous-disagreement-judge-v2"
DEFAULT_PROMPT = ROOT / "prompts" / "初中物理匿名边界裁判提示词.txt"
LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
OUTPUT_LOCK = Lock()
QUESTION_FIELDS = ("stem", "options", "analysis", "sub_questions")
SOURCE_FIELDS = set(QUESTION_FIELDS)
VALID_ROLES = {"balanced", "upper_threshold", "lower_ceiling", "arbiter"}
VALID_CONFIDENCE = {"高", "中", "低"}

TEACHER_ALIAS = {
    "容易": "送分题",
    "较易": "基础题",
    "中等": "中等题",
    "较难": "拔高题",
    "困难": "压轴题",
}


def extract_level(item: Dict[str, Any]) -> Optional[str]:
    difficulty_rating = item.get("difficulty_rating")
    if isinstance(difficulty_rating, dict):
        level = difficulty_rating.get("difficulty_level")
        if level in LEVEL_INDEX:
            return str(level)
    return None


def extract_features(item: Dict[str, Any]) -> Dict[str, str]:
    difficulty_rating = item.get("difficulty_rating")
    features = difficulty_rating.get("features") if isinstance(difficulty_rating, dict) else {}
    return rating.normalize_features(features)


def load_jsonl_map(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "").strip()
            if not question_id:
                raise ValueError(f"{path}:{line_number} 缺少 question_id")
            if question_id in rows:
                raise ValueError(f"{path} question_id 重复：{question_id}")
            rows[question_id] = item
    return rows


def parse_review_human_label(row: Dict[str, Any]) -> Optional[str]:
    model_level = str(row.get("model_difficulty_level") or "").strip()
    if row.get("verdict") == "correct" and model_level in LEVEL_INDEX:
        return model_level
    notes = str(row.get("human_notes") or "").strip()
    # 优先匹配完整系统档名，再匹配教师同义名称。
    for value in ("压轴题", "拔高题", "中等题", "基础题", "送分题"):
        if value in notes:
            return value
    for alias in ("困难", "较难", "中等", "较易", "容易"):
        if alias in notes:
            return TEACHER_ALIAS[alias]
    return None


def load_labels(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """支持教师默认复核 JSONL、教师 CSV 和 GPT 裁定 CSV。"""
    labels: Dict[str, str] = {}
    notes: Dict[str, str] = {}
    if path.lower().endswith(".jsonl"):
        for question_id, row in load_jsonl_map(path).items():
            label = parse_review_human_label(row)
            if label in LEVEL_INDEX:
                labels[question_id] = label
                notes[question_id] = str(row.get("human_notes") or "").strip()
        return labels, notes

    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        id_field = next((field for field in ("题目ID", "ID", "question_id") if field in fields), "")
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
        notes_field = next(
            (field for field in ("复核依据", "详细分析", "human_notes", "说明") if field in fields),
            "",
        )
        if not id_field or not label_field:
            raise ValueError(f"无法识别标签 CSV 的 ID/标签字段：{fields}")
        for row in reader:
            question_id = str(row.get(id_field) or "").strip()
            raw_label = str(row.get(label_field) or "").strip()
            label = TEACHER_ALIAS.get(raw_label, raw_label)
            if question_id and label in LEVEL_INDEX:
                labels[question_id] = label
                notes[question_id] = str(row.get(notes_field) or "").strip() if notes_field else ""
    return labels, notes


def merge_runs(paths: Sequence[str]) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, List[str]]]:
    if len(paths) < 2:
        raise ValueError("至少需要两次独立评级结果")
    run_maps = [load_jsonl_map(path) for path in paths]
    common_ids = sorted(set.intersection(*(set(run_map) for run_map in run_maps)))
    if not common_ids:
        raise ValueError("多次结果之间没有共同 question_id")
    base_items: Dict[str, Dict[str, Any]] = {}
    predictions: Dict[str, List[str]] = {}
    for question_id in common_ids:
        values: List[str] = []
        for run_map in run_maps:
            level = extract_level(run_map[question_id])
            if level not in LEVEL_INDEX:
                raise ValueError(f"question_id={question_id} 某次结果缺少合法等级")
            values.append(level)
        base_items[question_id] = run_maps[0][question_id]
        predictions[question_id] = values
    return common_ids, base_items, predictions


def candidate_pair(values: Sequence[str]) -> Optional[Tuple[str, str]]:
    unique = sorted(set(values), key=LEVEL_INDEX.get)
    if len(unique) != 2:
        return None
    if LEVEL_INDEX[unique[1]] - LEVEL_INDEX[unique[0]] != 1:
        return None
    return unique[0], unique[1]


def majority_level(values: Sequence[str]) -> str:
    counts = Counter(values)
    top_count = max(counts.values())
    tied = sorted(
        (level for level, count in counts.items() if count == top_count),
        key=LEVEL_INDEX.get,
    )
    # 偶数次调用出现平票时保守取较低档；三次实验不存在平票。
    return tied[0]


def build_cases(
    ids: Sequence[str],
    base_items: Dict[str, Dict[str, Any]],
    predictions: Dict[str, List[str]],
) -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}
    for question_id in ids:
        values = predictions[question_id]
        unique = sorted(set(values), key=LEVEL_INDEX.get)
        pair = candidate_pair(values)
        cases[question_id] = {
            "question_id": question_id,
            "base_item": base_items[question_id],
            "run_predictions": list(values),
            "candidate_levels": unique,
            "candidate_pair": list(pair) if pair else [],
            "unanimous": len(unique) == 1,
            "majority_level": majority_level(values),
        }
    return cases


def _short_text(value: Any, limit: int) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= limit else text[:limit] + "…"


def case_similarity(target: Dict[str, Any], reference: Dict[str, Any]) -> int:
    target_features = extract_features(target)
    reference_features = extract_features(reference)
    weights = {
        "problem_structure": 6,
        "information_carrier": 4,
        "experiment_requirement": 3,
        "graph_table_requirement": 3,
        "subquestion_dependency": 2,
        "state_count": 2,
        "cross_module": 2,
        "calculation_complexity": 2,
        "reasoning_chain": 1,
    }
    score = sum(
        weight
        for field, weight in weights.items()
        if target_features.get(field) == reference_features.get(field)
    )
    target_text = rating.construct_question_content(target)
    reference_text = rating.construct_question_content(reference)
    topic_terms = (
        "电路",
        "浮力",
        "杠杆",
        "滑轮",
        "压强",
        "密度",
        "透镜",
        "反射",
        "折射",
        "声",
        "热",
        "实验",
        "图像",
    )
    score += sum(2 for term in topic_terms if term in target_text and term in reference_text)
    return score


def select_balanced_fewshots(
    question_id: str,
    target: Dict[str, Any],
    pair: Tuple[str, str],
    reference_questions: Dict[str, Dict[str, Any]],
    labels: Dict[str, str],
    label_notes: Dict[str, str],
    per_level: int,
    seed: int,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    for level in pair:
        candidates: List[Tuple[int, str, Dict[str, Any]]] = []
        for reference_id, reference in reference_questions.items():
            if reference_id == question_id or labels.get(reference_id) != level:
                continue
            candidates.append((case_similarity(target, reference), reference_id, reference))
        candidates.sort(
            key=lambda value: (
                -value[0],
                hashlib.sha256(f"{seed}:{question_id}:{value[1]}".encode()).hexdigest(),
            )
        )
        if len(candidates) < per_level:
            raise ValueError(f"{pair} 的 {level} 可用校准样例不足 {per_level} 个")
        for score, reference_id, reference in candidates[:per_level]:
            selected.append(
                {
                    "teacher_level": level,
                    "stem_summary": _short_text(reference.get("stem"), 260),
                    "official_analysis_summary": _short_text(reference.get("analysis"), 420),
                    "teacher_boundary_note": _short_text(label_notes.get(reference_id), 180)
                    or "教师接受该题归入此档。",
                    "similarity_score": score,
                }
            )
    rng = random.Random(
        int(hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()[:16], 16)
    )
    rng.shuffle(selected)
    return selected


ROLE_INSTRUCTIONS = {
    "balanced": "中立比较两档：只有上档最低必要结构得到题目与解析支持时才选上档，否则选下档。",
    "upper_threshold": "逐项核验上档最低门槛。不要替上档补充题目中不存在的结构；门槛全部满足才选上档。",
    "lower_ceiling": "检查下档是否足以完整解释本题。只有存在下档无法覆盖的决定性结构时才选上档。",
    "arbiter": "比较两份匿名裁判意见，核对它们是否忠实于当前题与边界标准，给出最终二选一结论。",
}


def question_payload(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        field: item.get(field)
        for field in QUESTION_FIELDS
        if item.get(field) not in (None, "", [], {})
    }


def dynamic_output_template(pair: Tuple[str, str], role: str) -> Dict[str, Any]:
    return {
        "review_boundary": f"{pair[0]}|{pair[1]}",
        "lower_level": pair[0],
        "upper_level": pair[1],
        "judge_role": role,
        "chosen_level": f"只能填写 {pair[0]} 或 {pair[1]}",
        "upper_threshold_met": "布尔值",
        "confidence": "高/中/低",
        "effective_decision_count": "非负整数",
        "decisive_structures": ["当前题真实存在的结构，最多4条"],
        "missing_upper_requirements": ["未满足的上档必要条件；选择上档时可为空"],
        "evidence": [
            {
                "source_field": "stem/options/analysis/sub_questions",
                "source_excerpt": "当前题原文片段",
                "finding": "该原文对边界判断的意义",
            }
        ],
        "reason": "只解释当前相邻边界。",
    }


def build_judge_content(
    item: Dict[str, Any],
    pair: Tuple[str, str],
    fewshots: Sequence[Dict[str, Any]],
    role: str,
    candidate_arguments: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    if pair not in boundary_review.BOUNDARY_RUBRICS:
        raise ValueError(f"缺少边界规则：{pair}")
    payload: Dict[str, Any] = {
        "judge_role": role,
        "role_instruction": ROLE_INSTRUCTIONS[role],
        "lower_level": pair[0],
        "upper_level": pair[1],
        "boundary_rubric": boundary_review.BOUNDARY_RUBRICS[pair],
        "calibration_examples": list(fewshots),
        "current_question": question_payload(item),
        "required_output_template": dynamic_output_template(pair, role),
    }
    if candidate_arguments is not None:
        payload["anonymous_candidate_arguments"] = list(candidate_arguments)
    content = "【匿名相邻边界裁判输入】\n" + json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    )
    # 防止后续重构意外泄漏多次运行票数。
    forbidden = ("run_predictions", "majority_level", "vote_count", "首轮等级", "多数票")
    if any(token in content for token in forbidden):
        raise ValueError("匿名裁判输入泄漏运行身份或票数")
    return content + "\n\n请只输出 required_output_template 对应的合法 JSON。"


def canonical_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    return {"high": "高", "medium": "中", "mid": "中", "low": "低"}.get(
        text,
        str(value or "").strip(),
    )


def normalize_judgment(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    structures = source.get("decisive_structures")
    missing = source.get("missing_upper_requirements")
    evidence = source.get("evidence")
    return {
        "review_boundary": str(source.get("review_boundary") or "").replace("／", "/").strip(),
        "lower_level": str(source.get("lower_level") or "").strip(),
        "upper_level": str(source.get("upper_level") or "").strip(),
        "judge_role": str(source.get("judge_role") or "").strip(),
        "chosen_level": str(source.get("chosen_level") or "").strip(),
        "upper_threshold_met": source.get("upper_threshold_met"),
        "confidence": canonical_confidence(source.get("confidence")),
        "effective_decision_count": source.get("effective_decision_count"),
        "decisive_structures": [str(v).strip() for v in structures or [] if str(v).strip()],
        "missing_upper_requirements": [str(v).strip() for v in missing or [] if str(v).strip()],
        "evidence": [dict(v) for v in evidence or [] if isinstance(v, dict)],
        "reason": str(source.get("reason") or "").strip(),
    }


def validate_judgment(
    judgment: Dict[str, Any],
    pair: Tuple[str, str],
    role: str,
) -> Optional[str]:
    valid_boundaries = {f"{pair[0]}|{pair[1]}", f"{pair[0]}/{pair[1]}", f"{pair[0]}、{pair[1]}"}
    if judgment.get("review_boundary") not in valid_boundaries:
        return f"review_boundary 必须是 {pair[0]}|{pair[1]}"
    if judgment.get("lower_level") != pair[0] or judgment.get("upper_level") != pair[1]:
        return "lower_level 或 upper_level 与指定边界不一致"
    if judgment.get("judge_role") != role:
        return f"judge_role 必须是 {role}"
    chosen = judgment.get("chosen_level")
    if chosen not in pair:
        return f"chosen_level 只能是 {pair[0]} 或 {pair[1]}"
    if not isinstance(judgment.get("upper_threshold_met"), bool):
        return "upper_threshold_met 必须是布尔值"
    if judgment["upper_threshold_met"] != (chosen == pair[1]):
        return "upper_threshold_met 与 chosen_level 不一致"
    if judgment.get("confidence") not in VALID_CONFIDENCE:
        return "confidence 必须是高、中、低"
    count = judgment.get("effective_decision_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return "effective_decision_count 必须是非负整数"
    if not judgment.get("reason"):
        return "缺少 reason"
    if len(judgment.get("decisive_structures") or []) > 4:
        return "decisive_structures 最多4条"
    return None


def _response_text(body: Dict[str, Any]) -> str:
    text = ""
    for output_item in body.get("output", []):
        if output_item.get("type") != "message":
            continue
        for content in output_item.get("content", []):
            if content.get("type") == "output_text":
                text = str(content.get("text") or "")
    return text


def api_mode_order(model_name: str, configured_mode: str) -> List[str]:
    """返回接口尝试顺序；GLM 优先使用 Chat Completions。"""
    if configured_mode != "auto":
        return [configured_mode]
    if "glm" in str(model_name).lower():
        return ["chat_completions", "responses"]
    return ["responses", "chat_completions"]


def build_api_request(
    api_mode: str,
    model_name: str,
    user_content: str,
    temperature: Optional[float],
) -> Tuple[str, Dict[str, Any]]:
    if api_mode == "responses":
        payload: Dict[str, Any] = {
            "model": model_name,
            "input": [{"role": "user", "content": user_content}],
        }
        # thinking 是豆包 Responses 接口的扩展字段，不发送给 GLM 等模型。
        if "doubao" in str(model_name).lower():
            payload["thinking"] = {"type": "disabled"}
        endpoint = "responses"
    elif api_mode == "chat_completions":
        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_content}],
            "stream": False,
        }
        endpoint = "chat/completions"
    else:
        raise ValueError(f"不支持的 API 模式：{api_mode}")
    if temperature is not None:
        payload["temperature"] = temperature
    return endpoint, payload


def extract_api_text(api_mode: str, body: Dict[str, Any]) -> str:
    if api_mode == "responses":
        return _response_text(body)
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(part.get("text") or "")
            for part in content
            if isinstance(part, dict)
        )
    return str(content or "")


def extract_api_usage(api_mode: str, body: Dict[str, Any]) -> Dict[str, int]:
    raw_usage = body.get("usage") or {}
    if api_mode == "responses":
        prompt_tokens = int(raw_usage.get("input_tokens", 0) or 0)
        completion_tokens = int(raw_usage.get("output_tokens", 0) or 0)
    else:
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(raw_usage.get("completion_tokens", 0) or 0)
    total_tokens = int(
        raw_usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


async def call_judge_model(
    prompt: str,
    content: str,
    pair: Tuple[str, str],
    role: str,
    session: aiohttp.ClientSession,
    model_name: str,
    temperature: Optional[float],
    api_mode: str,
    retries: int,
    timeout_sec: int,
) -> Tuple[Dict[str, Any], Dict[str, int], float, str, str]:
    started = time.time()
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    last_error = ""
    validation_feedback = ""
    for attempt in range(retries):
        retry_content = content
        if validation_feedback:
            retry_content += (
                "\n\n【上一次输出校验失败】\n"
                + validation_feedback
                + f"\n指定边界只能是 {pair[0]}|{pair[1]}，只能选择其中一个等级。"
            )
        user_content = prompt + "\n\n" + retry_content
        endpoint_errors: List[str] = []
        stop_retrying = False
        for current_api_mode in api_mode_order(model_name, api_mode):
            endpoint, payload = build_api_request(
                current_api_mode,
                model_name,
                user_content,
                temperature,
            )
            try:
                async with session.post(
                    f"{rating.BASE_URL}{endpoint}",
                    json=payload,
                    headers={"Authorization": f"Bearer {rating.API_KEY}"},
                    timeout=aiohttp.ClientTimeout(total=timeout_sec),
                ) as response:
                    if response.status == 200:
                        body = await response.json()
                        for key, value in extract_api_usage(
                            current_api_mode,
                            body,
                        ).items():
                            usage[key] += value
                        judgment = normalize_judgment(
                            rating.parse_model_response(
                                extract_api_text(current_api_mode, body)
                            )
                        )
                        error = validate_judgment(judgment, pair, role)
                        if not error:
                            return (
                                judgment,
                                usage,
                                time.time() - started,
                                "",
                                current_api_mode,
                            )
                        last_error = f"{current_api_mode} 输出校验失败: {error}"
                        validation_feedback = last_error
                        # 接口已连通，仅输出格式无效；下一次重试带校验反馈，
                        # 不在同一次尝试中切换另一接口并重复生成。
                        break
                    response_text = await response.text()
                    endpoint_error = (
                        f"{current_api_mode} HTTP {response.status}: "
                        f"{response_text[:500]}"
                    )
                    endpoint_errors.append(endpoint_error)
                    if response.status in {401, 403, 429}:
                        stop_retrying = response.status in {401, 403}
                        break
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                endpoint_errors.append(f"{current_api_mode}: {exc}")
            except Exception as exc:
                endpoint_errors.append(f"{current_api_mode}: {exc}")
        if endpoint_errors:
            last_error = " | ".join(endpoint_errors)
            validation_feedback = last_error
        if stop_retrying:
            break
        if attempt < retries - 1:
            await asyncio.sleep(2**attempt + random.random())
    return {}, usage, time.time() - started, last_error or "裁判响应无效", ""


async def call_with_semaphore(
    semaphore: Semaphore,
    **kwargs: Any,
) -> Tuple[Dict[str, Any], Dict[str, int], float, str, str]:
    async with semaphore:
        return await call_judge_model(**kwargs)


def judgment_for_arbiter(value: Dict[str, Any], name: str) -> Dict[str, Any]:
    return {
        "anonymous_name": name,
        "chosen_level": value.get("chosen_level"),
        "effective_decision_count": value.get("effective_decision_count"),
        "decisive_structures": value.get("decisive_structures") or [],
        "missing_upper_requirements": value.get("missing_upper_requirements") or [],
        "evidence": value.get("evidence") or [],
        "reason": value.get("reason"),
    }


async def judge_case(
    item: Dict[str, Any],
    pair: Tuple[str, str],
    fewshots: Sequence[Dict[str, Any]],
    strategy: str,
    prompt: str,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    judge_model: str,
    second_judge_model: str,
    arbiter_model: str,
    temperature_raw: str,
    api_mode: str,
    retries: int,
    timeout_sec: int,
) -> Dict[str, Any]:
    calls: List[Dict[str, Any]] = []

    async def run(role: str, model: str, arguments: Optional[Sequence[Dict[str, Any]]] = None):
        # Lite 服务端固定为 1；其他模型默认不发送 temperature，
        # 仅当调用方显式传入 --temperature 时才设置，避免模型网关不兼容。
        temperature = rating.resolve_temperature(model, temperature_raw)
        content = build_judge_content(item, pair, fewshots, role, arguments)
        result, usage, elapsed, error, used_api_mode = await call_with_semaphore(
            semaphore,
            prompt=prompt,
            content=content,
            pair=pair,
            role=role,
            session=session,
            model_name=model,
            temperature=temperature,
            api_mode=api_mode,
            retries=retries,
            timeout_sec=timeout_sec,
        )
        calls.append(
            {
                "role": role,
                "model": model,
                "temperature": temperature,
                "api_mode": used_api_mode,
                "result": result,
                "usage": usage,
                "elapsed_seconds": round(elapsed, 3),
                "error": error,
            }
        )
        return result, error

    if strategy == "balanced":
        result, error = await run("balanced", judge_model)
        return {
            "chosen_level": result.get("chosen_level") if result else None,
            "decision_source": "balanced_judge" if result else "fallback_majority",
            "calls": calls,
            "error": error,
        }

    first_task = asyncio.create_task(run("upper_threshold", judge_model))
    second_task = asyncio.create_task(run("lower_ceiling", second_judge_model))
    (first, first_error), (second, second_error) = await asyncio.gather(
        first_task,
        second_task,
    )
    if first and second and first.get("chosen_level") == second.get("chosen_level"):
        return {
            "chosen_level": first["chosen_level"],
            "decision_source": "dual_judges_agree",
            "calls": calls,
            "error": "",
        }
    if not first and not second:
        return {
            "chosen_level": None,
            "decision_source": "fallback_majority",
            "calls": calls,
            "error": (
                "双裁判均未形成有效结果，已跳过仲裁。"
                f" 裁判甲: {first_error or '未知错误'}；"
                f"裁判乙: {second_error or '未知错误'}"
            ),
        }
    if not arbiter_model:
        return {
            "chosen_level": None,
            "decision_source": "fallback_majority",
            "calls": calls,
            "error": first_error or second_error or "双裁判意见不一致且未配置仲裁模型",
        }
    arguments = [
        judgment_for_arbiter(first, "裁判甲") if first else {"anonymous_name": "裁判甲", "error": first_error},
        judgment_for_arbiter(second, "裁判乙") if second else {"anonymous_name": "裁判乙", "error": second_error},
    ]
    arbiter, arbiter_error = await run("arbiter", arbiter_model, arguments)
    return {
        "chosen_level": arbiter.get("chosen_level") if arbiter else None,
        "decision_source": "arbiter" if arbiter else "fallback_majority",
        "calls": calls,
        "error": arbiter_error,
    }


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_signature(args: argparse.Namespace, prompt: str) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "run_hashes": [sha256_file(path) for path in args.run],
        "labels_hash": sha256_file(args.labels),
        "reference_questions_hash": sha256_file(args.reference_questions),
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
        "strategy": args.strategy,
        "judge_model": args.judge_model,
        "second_judge_model": args.second_judge_model,
        "arbiter_model": args.arbiter_model,
        "temperature": args.temperature,
        "api_mode": args.api_mode,
        "fewshot_per_level": args.fewshot_per_level,
        "seed": args.seed,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def processed_ids(path: str, signature: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            agent = item.get("anonymous_boundary_judge")
            if not isinstance(agent, dict) or agent.get("run_signature") != signature:
                raise ValueError("输出文件含不同运行配置，请更换输出文件")
            if agent.get("error"):
                continue
            done.add(str(item.get("question_id") or ""))
    return done


async def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    async with OUTPUT_LOCK:
        async with aiofiles.open(path, "a", encoding="utf-8") as handle:
            await handle.write(json.dumps(item, ensure_ascii=False) + "\n")


async def process_case(
    case: Dict[str, Any],
    labels: Dict[str, str],
    reference_questions: Dict[str, Dict[str, Any]],
    label_notes: Dict[str, str],
    prompt: str,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    args: argparse.Namespace,
    signature: str,
) -> None:
    question_id = case["question_id"]
    output = copy.deepcopy(case["base_item"])
    pair_values = case["candidate_pair"]
    chosen = case["majority_level"]
    fewshots: List[Dict[str, Any]] = []
    judge_result: Dict[str, Any] = {
        "chosen_level": chosen,
        "decision_source": "unanimous" if case["unanimous"] else "fallback_majority",
        "calls": [],
        "error": "",
    }
    if not case["unanimous"] and len(pair_values) == 2:
        pair = (pair_values[0], pair_values[1])
        fewshots = select_balanced_fewshots(
            question_id,
            case["base_item"],
            pair,
            reference_questions,
            labels,
            label_notes,
            args.fewshot_per_level,
            args.seed,
        )
        judge_result = await judge_case(
            case["base_item"],
            pair,
            fewshots,
            args.strategy,
            prompt,
            session,
            semaphore,
            args.judge_model,
            args.second_judge_model,
            args.arbiter_model,
            args.temperature,
            args.api_mode,
            args.retries,
            args.timeout,
        )
        chosen = judge_result.get("chosen_level") or case["majority_level"]
    elif not case["unanimous"]:
        judge_result["error"] = "候选不是单个相邻边界，当前版本回退多数结果"

    total_usage = Counter()
    for call in judge_result.get("calls") or []:
        total_usage.update(call.get("usage") or {})
    output["multi_call_final_level"] = chosen
    output["anonymous_boundary_judge"] = {
        "pipeline_version": PIPELINE_VERSION,
        "run_signature": signature,
        "unanimous": case["unanimous"],
        # 票数只保存在输出审计中，从未发送给裁判。
        "run_predictions_audit_only": case["run_predictions"],
        "candidate_levels": case["candidate_levels"],
        "majority_fallback": case["majority_level"],
        "fewshot_count": len(fewshots),
        "strategy": args.strategy,
        "chosen_level": chosen,
        "decision_source": judge_result.get("decision_source"),
        "calls": judge_result.get("calls") or [],
        "error": judge_result.get("error") or "",
        "usage": dict(total_usage),
    }
    await append_jsonl(args.output, output)
    if judge_result.get("error"):
        await append_jsonl(
            args.error,
            {
                "question_id": question_id,
                "candidate_levels": case["candidate_levels"],
                "error": judge_result["error"],
            },
        )


def load_latest_output(path: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[str(item.get("question_id") or "")] = item
    return rows


def summarize(
    cases: Dict[str, Dict[str, Any]],
    outputs: Dict[str, Dict[str, Any]],
    labels: Dict[str, str],
) -> Dict[str, Any]:
    ids = [question_id for question_id in cases if question_id in labels]
    if not ids:
        return {"evaluated": 0}
    run_count = len(next(iter(cases.values()))["run_predictions"])
    run_correct = [0] * run_count
    majority_correct = final_correct = oracle_correct = 0
    unanimous_ids: List[str] = []
    disagreement_ids: List[str] = []
    judge_correct = 0
    judge_valid = 0
    improved = worsened = unchanged = 0
    boundary_stats: Dict[str, Counter[str]] = defaultdict(Counter)
    decision_sources: Counter[str] = Counter()
    total_usage: Counter[str] = Counter()
    target_distribution: Counter[str] = Counter()
    prediction_distribution: Counter[str] = Counter()
    confusion: Dict[str, Counter[str]] = {
        level: Counter() for level in LEVELS
    }
    dual_cases = dual_agreements = arbiter_cases = 0
    errors = 0

    for question_id in ids:
        case = cases[question_id]
        target = labels[question_id]
        target_distribution[target] += 1
        for index, prediction in enumerate(case["run_predictions"]):
            run_correct[index] += prediction == target
        majority = case["majority_level"]
        majority_correct += majority == target
        oracle_correct += target in case["candidate_levels"]
        if case["unanimous"]:
            unanimous_ids.append(question_id)
        else:
            disagreement_ids.append(question_id)

        output = outputs.get(question_id)
        chosen = (
            str(output.get("multi_call_final_level") or "")
            if isinstance(output, dict)
            else majority
        )
        if chosen not in LEVEL_INDEX:
            chosen = majority
        final_correct += chosen == target
        prediction_distribution[chosen] += 1
        confusion[target][chosen] += 1
        before_error = abs(LEVEL_INDEX[majority] - LEVEL_INDEX[target])
        after_error = abs(LEVEL_INDEX[chosen] - LEVEL_INDEX[target])
        if after_error < before_error:
            improved += 1
        elif after_error > before_error:
            worsened += 1
        else:
            unchanged += 1

        if output:
            agent = output.get("anonymous_boundary_judge") or {}
            decision_sources[str(agent.get("decision_source") or "unknown")] += 1
            total_usage.update(agent.get("usage") or {})
            calls = agent.get("calls") or []
            judge_calls = [
                call
                for call in calls
                if call.get("role") in {"upper_threshold", "lower_ceiling"}
                and isinstance(call.get("result"), dict)
                and call["result"].get("chosen_level") in LEVEL_INDEX
            ]
            if len(judge_calls) == 2:
                dual_cases += 1
                if (
                    judge_calls[0]["result"]["chosen_level"]
                    == judge_calls[1]["result"]["chosen_level"]
                ):
                    dual_agreements += 1
            if any(call.get("role") == "arbiter" for call in calls):
                arbiter_cases += 1
            if agent.get("error"):
                errors += 1
            if not case["unanimous"] and agent.get("decision_source") != "fallback_majority":
                judge_valid += 1
                judge_correct += chosen == target
            if not case["unanimous"]:
                pair_name = "/".join(case["candidate_levels"])
                boundary_stats[pair_name]["evaluated"] += 1
                boundary_stats[pair_name]["correct"] += chosen == target

    unanimous_correct = sum(
        cases[q]["majority_level"] == labels[q] for q in unanimous_ids
    )
    disagreement_oracle = sum(
        labels[q] in cases[q]["candidate_levels"] for q in disagreement_ids
    )
    per_level_metrics: Dict[str, Dict[str, Any]] = {}
    for level in LEVELS:
        true_positive = confusion[level][level]
        support = target_distribution[level]
        predicted = prediction_distribution[level]
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_level_metrics[level] = {
            "support": support,
            "predicted": predicted,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    target_correct_for_90 = (9 * len(ids) + 9) // 10
    required_net_gain_for_90 = max(0, target_correct_for_90 - majority_correct)
    required_disagreement_accuracy_for_90 = (
        (target_correct_for_90 - unanimous_correct) / len(disagreement_ids)
        if disagreement_ids
        else 0.0
    )
    return {
        "evaluated": len(ids),
        "run_accuracy": [
            round(value / len(ids), 4) for value in run_correct
        ],
        "majority_accuracy": round(majority_correct / len(ids), 4),
        "candidate_oracle_accuracy": round(oracle_correct / len(ids), 4),
        "unanimous_count": len(unanimous_ids),
        "unanimous_accuracy": round(unanimous_correct / len(unanimous_ids), 4)
        if unanimous_ids
        else None,
        "disagreement_count": len(disagreement_ids),
        "disagreement_candidate_recall": round(
            disagreement_oracle / len(disagreement_ids),
            4,
        )
        if disagreement_ids
        else None,
        "judge_valid_count": judge_valid,
        "judge_accuracy_on_valid_disagreements": round(
            judge_correct / judge_valid,
            4,
        )
        if judge_valid
        else None,
        "dual_judge_valid_cases": dual_cases,
        "dual_judge_agreement_rate": round(dual_agreements / dual_cases, 4)
        if dual_cases
        else None,
        "arbiter_case_count": arbiter_cases,
        "final_accuracy": round(final_correct / len(ids), 4),
        "final_correct": final_correct,
        "improved_vs_majority": improved,
        "worsened_vs_majority": worsened,
        "unchanged_vs_majority": unchanged,
        "net_improvement": improved - worsened,
        "errors": errors,
        "decision_sources": dict(decision_sources),
        "teacher_distribution": dict(target_distribution),
        "prediction_distribution": dict(prediction_distribution),
        "confusion_matrix": {
            target: {
                prediction: confusion[target][prediction]
                for prediction in LEVELS
            }
            for target in LEVELS
        },
        "per_level_metrics": per_level_metrics,
        "boundary_accuracy": {
            name: {
                **dict(values),
                "accuracy": round(values["correct"] / values["evaluated"], 4)
                if values["evaluated"]
                else None,
            }
            for name, values in sorted(boundary_stats.items())
        },
        "target_90_percent": {
            "required_correct": target_correct_for_90,
            "required_net_gain_vs_majority": required_net_gain_for_90,
            "required_disagreement_accuracy": round(
                required_disagreement_accuracy_for_90,
                4,
            ),
        },
        "usage": dict(total_usage),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="多次物理评级的匿名相邻边界裁判")
    parser.add_argument("--run", action="append", required=True, help="可重复指定，建议3次冻结结果")
    parser.add_argument("--labels", required=True, help="教师复核 JSONL 或标签 CSV")
    parser.add_argument("--reference-questions", required=True, help="含题干、解析的校准题 JSONL")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--error", required=True)
    parser.add_argument("-p", "--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("--strategy", choices=("balanced", "dual"), default="balanced")
    parser.add_argument("--judge-model", default="doubao-seed-2.0-lite")
    parser.add_argument("--second-judge-model", default="doubao-seed-2.0-lite")
    parser.add_argument("--arbiter-model", default="")
    parser.add_argument(
        "--temperature",
        default="",
        help="非 Lite 模型采样温度；留空则不发送。Lite 始终固定为1。",
    )
    parser.add_argument(
        "--api-mode",
        choices=("auto", "responses", "chat_completions"),
        default="auto",
        help="模型 API 协议；auto 对 GLM 优先 Chat Completions，对豆包优先 Responses。",
    )
    parser.add_argument("--fewshot-per-level", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("-c", "--concurrency", type=int, default=15)
    parser.add_argument("-t", "--timeout", type=int, default=180)
    parser.add_argument("-r", "--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="只用第一道分歧题验证模型、接口和输出格式，不执行全量。",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过全量运行前的单题连通性预检。",
    )
    parser.add_argument(
        "--disagreements-only",
        action="store_true",
        help="结果文件只保存分歧题；汇总仍按全部共同题目计算。",
    )
    return parser


async def run_preflight(
    disagreement: Sequence[Dict[str, Any]],
    labels: Dict[str, str],
    reference_questions: Dict[str, Dict[str, Any]],
    label_notes: Dict[str, str],
    prompt: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not disagreement:
        return {"skipped": True, "reason": "没有分歧题"}
    case = disagreement[0]
    pair_values = case["candidate_pair"]
    if len(pair_values) != 2:
        raise RuntimeError("第一道分歧题不是合法相邻边界，无法预检")
    pair = (pair_values[0], pair_values[1])
    fewshots = select_balanced_fewshots(
        case["question_id"],
        case["base_item"],
        pair,
        reference_questions,
        labels,
        label_notes,
        args.fewshot_per_level,
        args.seed,
    )
    content = build_judge_content(
        case["base_item"],
        pair,
        fewshots,
        "balanced",
    )
    connector = aiohttp.TCPConnector(limit=2)
    async with aiohttp.ClientSession(connector=connector) as session:
        judgment, usage, elapsed, error, used_api_mode = await call_judge_model(
            prompt=prompt,
            content=content,
            pair=pair,
            role="balanced",
            session=session,
            model_name=args.judge_model,
            temperature=rating.resolve_temperature(
                args.judge_model,
                args.temperature,
            ),
            api_mode=args.api_mode,
            retries=args.retries,
            timeout_sec=args.timeout,
        )
    report = {
        "question_id": case["question_id"],
        "candidate_pair": list(pair),
        "model": args.judge_model,
        "api_mode": used_api_mode,
        "chosen_level": judgment.get("chosen_level") if judgment else None,
        "usage": usage,
        "elapsed_seconds": round(elapsed, 3),
        "error": error,
    }
    print("单题连通性预检:", json.dumps(report, ensure_ascii=False, sort_keys=True))
    if error or not judgment:
        raise RuntimeError(
            "单题预检失败，已阻止全量空跑："
            + (error or "模型未返回合法裁判结果")
        )
    return report


async def main() -> None:
    args = build_parser().parse_args()
    if args.fewshot_per_level < 1:
        raise ValueError("--fewshot-per-level 必须大于0")
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    ids, base_items, predictions = merge_runs(args.run)
    cases = build_cases(ids, base_items, predictions)
    labels, label_notes = load_labels(args.labels)
    reference_questions = load_jsonl_map(args.reference_questions)
    signature = build_run_signature(args, prompt)

    disagreement = [case for case in cases.values() if not case["unanimous"]]
    pair_distribution = Counter(
        "/".join(case["candidate_levels"]) for case in disagreement
    )
    preflight = summarize(cases, {}, labels)
    print("本实验用于检验：匿名裁判能否在三次分歧题上达到足以令总体准确率超过90%的选择能力。")
    print(f"共同题目: {len(ids)}，一致题: {len(ids)-len(disagreement)}，分歧题: {len(disagreement)}")
    print("分歧边界:", json.dumps(dict(pair_distribution), ensure_ascii=False, sort_keys=True))
    print("调用裁判前指标:", json.dumps(preflight, ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        return
    if not args.skip_preflight or args.preflight_only:
        await run_preflight(
            disagreement,
            labels,
            reference_questions,
            label_notes,
            prompt,
            args,
        )
    if args.preflight_only:
        return

    for path in (args.output, args.error):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    done = processed_ids(args.output, signature)
    pending = [
        case
        for question_id, case in cases.items()
        if question_id not in done
        and (not args.disagreements_only or not case["unanimous"])
    ]
    print(f"已完成: {len(done)}，待写入: {len(pending)}，策略: {args.strategy}")
    if pending:
        semaphore = Semaphore(args.concurrency)
        connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
        progress = tqdm(total=len(pending), unit="item", desc="Anonymous Judge Progress")
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                asyncio.create_task(
                    process_case(
                        case,
                        labels,
                        reference_questions,
                        label_notes,
                        prompt,
                        session,
                        semaphore,
                        args,
                        signature,
                    )
                )
                for case in pending
            ]
            for task in asyncio.as_completed(tasks):
                await task
                progress.update(1)
        progress.close()

    outputs = load_latest_output(args.output)
    report = summarize(cases, outputs, labels)
    report.update(
        {
            "pipeline_version": PIPELINE_VERSION,
            "run_signature": signature,
            "strategy": args.strategy,
            "judge_model": args.judge_model,
            "second_judge_model": args.second_judge_model,
            "arbiter_model": args.arbiter_model,
            "temperature": args.temperature,
            "api_mode": args.api_mode,
            "disagreements_only": args.disagreements_only,
            "fewshot_per_level": args.fewshot_per_level,
            "label_source": args.labels,
        }
    )
    report_path = args.output + ".summary.json"
    Path(report_path).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("最终实验结果:", json.dumps(report, ensure_ascii=False, sort_keys=True))
    print("结果文件:", os.path.abspath(args.output))
    print("汇总文件:", os.path.abspath(report_path))


if __name__ == "__main__":
    started = time.time()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("收到中断信号，已停止匿名裁判实验。")
    finally:
        print(f"匿名裁判实验耗时: {round((time.time() - started) / 60, 2)} 分钟")
