# -*- coding: utf-8 -*-
"""高中化学两阶段难度评级 Pipeline。

流程：模型提取化学结构特征和原始正确率；程序检测十类高难特征、应用
1.00/0.85/0.70 乘数并映射五档；第二次模型调用只复核结构事实和相邻边界；
程序重算并输出可审计结果。原 difficulty、percent_correct、answered_count
不会发送给模型。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any

try:
    import aiohttp
    import json_repair
    from dotenv import load_dotenv
    from tqdm.asyncio import tqdm
except ImportError as exc:  # pragma: no cover - 项目 venv 中执行
    raise RuntimeError(
        "缺少运行依赖，请启用项目 venv 并安装 aiohttp、json-repair、python-dotenv、tqdm"
    ) from exc

from high_chemistry_pipeline_core import (
    HIGH_DIFFICULTY_FEATURE_NAMES,
    REQUIRED_FEATURE_FIELDS,
    enrich_stage1_rating,
    finalize_level,
    normalize_stage1_rating,
    prepare_question,
    recalculate_verification,
)


load_dotenv()

API_KEY = os.getenv("API_KEY", "not-needed")
BASE_URL = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1").rstrip("/") + "/"
MODEL_NAME = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")
TEMPERATURE_RAW = os.getenv("TEMPERATURE", "")
ENABLE_STAGE2_AUTO_ADJUST = os.getenv("ENABLE_STAGE2_AUTO_ADJUST", "0").strip() == "1"

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "samples" / "high-chemistry-sample25k.jsonl"
DEFAULT_PROMPT = PROJECT_ROOT / "prompts" / "高中化学难度打标提示词.txt"
DEFAULT_OUTPUT = OUTPUTS_ROOT / "model_runs" / "high_chemistry_two_stage.jsonl"
DEFAULT_ERRORS = OUTPUTS_ROOT / "model_runs" / "high_chemistry_two_stage_errors.jsonl"
DEFAULT_CACHE = OUTPUTS_ROOT / "cache" / "high_chemistry_stage1_prefix_cache.json"

CACHE_EXPIRE_SECONDS = 5 * 24 * 3600
FILE_LOCK = asyncio.Lock()
CACHE_LOCK = asyncio.Lock()
CACHE_CREATE_LOCK = asyncio.Lock()

FEATURE_EXTRACTION_PROMPT_PREFIX = ""
FEATURE_EXTRACTION_PROMPT_SUFFIX = ""
VERIFICATION_PROMPT_PREFIX = ""
VERIFICATION_PROMPT_SUFFIX = ""


class PrefixCacheState:
    def __init__(self, response_id: str, cache_path: Path):
        self.response_id = response_id
        self.cache_path = cache_path
        self.refresh_lock = asyncio.Lock()


def resolve_temperature(model_name: str, raw_value: str) -> float | None:
    if "lite" in model_name.lower():
        return 1.0
    value = str(raw_value or "").strip()
    return float(value) if value else None


TEMPERATURE = resolve_temperature(MODEL_NAME, TEMPERATURE_RAW)


def load_prompt_config(path: str | Path) -> None:
    global FEATURE_EXTRACTION_PROMPT_PREFIX
    global FEATURE_EXTRACTION_PROMPT_SUFFIX
    global VERIFICATION_PROMPT_PREFIX
    global VERIFICATION_PROMPT_SUFFIX
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"找不到 Prompt：{prompt_path}")
    namespace: dict[str, Any] = {}
    source = prompt_path.read_text(encoding="utf-8")
    exec(compile(source, str(prompt_path), "exec"), namespace)
    names = (
        "FEATURE_EXTRACTION_PROMPT_PREFIX",
        "FEATURE_EXTRACTION_PROMPT_SUFFIX",
        "VERIFICATION_PROMPT_PREFIX",
        "VERIFICATION_PROMPT_SUFFIX",
    )
    missing = [name for name in names if not namespace.get(name)]
    if missing:
        raise ValueError(f"Prompt 缺少变量：{', '.join(missing)}")
    FEATURE_EXTRACTION_PROMPT_PREFIX = str(namespace[names[0]])
    FEATURE_EXTRACTION_PROMPT_SUFFIX = str(namespace[names[1]])
    VERIFICATION_PROMPT_PREFIX = str(namespace[names[2]])
    VERIFICATION_PROMPT_SUFFIX = str(namespace[names[3]])


def _prefix_hash() -> str:
    value = f"{MODEL_NAME}\n{FEATURE_EXTRACTION_PROMPT_PREFIX}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def _read_cache(path: Path) -> dict[str, Any]:
    async with CACHE_LOCK:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}


async def _write_cache(path: Path, value: dict[str, Any]) -> None:
    async with CACHE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _valid_cache(entry: dict[str, Any]) -> bool:
    return bool(
        entry.get("response_id")
        and int(entry.get("expire_at", 0)) > int(time.time())
        and entry.get("prefix_hash") == _prefix_hash()
        and entry.get("model_name") == MODEL_NAME
    )


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}


def _extract_output_text(response_json: dict[str, Any]) -> str:
    if isinstance(response_json.get("output_text"), str):
        return response_json["output_text"]
    chunks: list[str] = []
    for item in response_json.get("output", []):
        if not isinstance(item, dict):
            continue
        for content in item.get("content", []):
            if isinstance(content, dict) and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _parse_json_object(text: str) -> dict[str, Any]:
    value = json_repair.loads(text)
    if not isinstance(value, dict):
        raise ValueError("模型输出必须为 JSON 对象")
    return value


def _usage(response_json: dict[str, Any]) -> dict[str, int]:
    usage = response_json.get("usage") or {}
    return {
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }


async def _post_response(
    session: aiohttp.ClientSession,
    payload: dict[str, Any],
    timeout: int,
) -> tuple[int, dict[str, Any] | None, str]:
    async with session.post(
        f"{BASE_URL}responses",
        json=payload,
        headers=_headers(),
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        text = await response.text()
        if response.status != 200:
            return response.status, None, text
        try:
            return response.status, json.loads(text), ""
        except json.JSONDecodeError:
            return response.status, None, text


async def create_prefix_cache(
    session: aiohttp.ClientSession,
    cache_path: Path,
    retries: int,
    timeout: int,
) -> str | None:
    expire_at = int(time.time()) + CACHE_EXPIRE_SECONDS
    payload: dict[str, Any] = {
        "model": MODEL_NAME,
        "input": [{"role": "user", "content": FEATURE_EXTRACTION_PROMPT_PREFIX}],
        "thinking": {"type": "disabled"},
        "expire_at": expire_at,
        "caching": {"type": "enabled", "prefix": True},
    }
    if TEMPERATURE is not None:
        payload["temperature"] = TEMPERATURE
    for attempt in range(retries):
        try:
            status, body, error = await _post_response(session, payload, timeout)
            if status == 200 and body and body.get("id"):
                response_id = str(body["id"])
                await _write_cache(cache_path, {
                    "response_id": response_id,
                    "expire_at": expire_at,
                    "prefix_hash": _prefix_hash(),
                    "model_name": MODEL_NAME,
                })
                return response_id
            if status != 429 and status < 500:
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError):
            pass
        if attempt < retries - 1:
            await asyncio.sleep(2 ** attempt + random.random())
    return None


async def get_or_create_prefix_cache(
    session: aiohttp.ClientSession,
    cache_path: Path,
    retries: int,
    timeout: int,
) -> str | None:
    async with CACHE_CREATE_LOCK:
        entry = await _read_cache(cache_path)
        if _valid_cache(entry):
            return str(entry["response_id"])
        return await create_prefix_cache(session, cache_path, retries, timeout)


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def construct_question_text(question: dict[str, Any], input_quality: dict[str, Any]) -> str:
    return (
        "【输入质量】\n" + _json_block(input_quality)
        + "\n\n【题目数据】\n" + _json_block(question)
    )


def _content_with_images(text: str, image_urls: list[str]) -> str | list[dict[str, Any]]:
    if not image_urls:
        return text
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    content.extend({"type": "input_image", "image_url": url} for url in image_urls)
    return content


def validate_verification(result: dict[str, Any]) -> dict[str, Any]:
    required = {
        "difficulty_source", "feature_corrections", "missed_features",
        "has_structural_revision", "adjacent_boundary_review", "confidence",
        "reviewed_original_predicted_accuracy", "reviewed_high_difficulty_features",
        "analysis",
    }
    missing = sorted(required - result.keys())
    if missing:
        raise ValueError(f"第二阶段缺少字段：{missing}")
    required_text = ("difficulty_source", "analysis")
    for field in required_text:
        if not str(result.get(field, "")).strip():
            raise ValueError(f"{field} 必须为非空字符串")
    if type(result.get("has_structural_revision")) is not bool:
        raise ValueError("has_structural_revision 必须为布尔值")
    corrections = result.get("feature_corrections")
    if not isinstance(corrections, list):
        raise ValueError("feature_corrections 必须为列表")
    valid_corrections = []
    ignored_corrections = []
    for correction in corrections:
        if not isinstance(correction, dict):
            raise ValueError("feature_corrections 每项必须为对象")
        correction_missing = {
            "field", "original_value", "reviewed_value", "evidence"
        } - correction.keys()
        if correction_missing:
            raise ValueError(f"feature_corrections 缺少字段：{sorted(correction_missing)}")
        if "reviewed_value" not in correction:
            raise ValueError("feature_corrections 缺少 reviewed_value")
        if not str(correction.get("evidence", "")).strip():
            raise ValueError("feature_corrections.evidence 不得为空")
        if correction.get("field") == "high_difficulty_features":
            ignored_corrections.append(correction)
            continue
        if correction.get("field") not in REQUIRED_FEATURE_FIELDS:
            raise ValueError(f"feature_corrections 含非法字段：{correction.get('field')!r}")
        valid_corrections.append(correction)
    result["feature_corrections"] = valid_corrections
    if ignored_corrections:
        result["ignored_feature_corrections"] = ignored_corrections
    names = result.get("reviewed_high_difficulty_features")
    if not isinstance(names, list):
        raise ValueError("reviewed_high_difficulty_features 必须为列表")
    if len(names) != len(set(names)):
        raise ValueError("reviewed_high_difficulty_features 不得重复")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("reviewed_high_difficulty_features 每项必须为字符串")
    invalid = [name for name in names if name not in HIGH_DIFFICULTY_FEATURE_NAMES]
    if invalid:
        raise ValueError(f"reviewed_high_difficulty_features 含非法值：{invalid}")
    if result.get("confidence") not in {"高", "中", "低"}:
        raise ValueError("confidence 非法")
    missed_features = result.get("missed_features")
    if (
        not isinstance(missed_features, list)
        or not missed_features
        or any(not isinstance(value, str) or not value.strip() for value in missed_features)
    ):
        raise ValueError("missed_features 必须为非空字符串列表")
    boundary = result.get("adjacent_boundary_review")
    if not isinstance(boundary, dict):
        raise ValueError("adjacent_boundary_review 必须为对象")
    if boundary.get("verdict") not in {"维持", "应更简单一档", "应更难一档"}:
        raise ValueError("adjacent_boundary_review.verdict 非法")
    legal_boundaries = {"88边界", "85边界", "58边界", "38边界"}
    if (
        not isinstance(boundary.get("boundaries_checked"), list)
        or not boundary["boundaries_checked"]
        or any(value not in legal_boundaries for value in boundary["boundaries_checked"])
    ):
        raise ValueError("boundaries_checked 必须为非空列表")
    if (
        not isinstance(boundary.get("decisive_evidence"), list)
        or not boundary["decisive_evidence"]
        or any(not isinstance(value, str) or not value.strip() for value in boundary["decisive_evidence"])
    ):
        raise ValueError("decisive_evidence 必须为非空列表")
    try:
        accuracy = float(result.get("reviewed_original_predicted_accuracy"))
    except (TypeError, ValueError) as exc:
        raise ValueError("reviewed_original_predicted_accuracy 必须为数值") from exc
    if not 0 <= accuracy <= 100:
        raise ValueError("reviewed_original_predicted_accuracy 必须位于0到100")
    return copy.deepcopy(result)


async def call_stage1(
    *,
    session: aiohttp.ClientSession,
    question_text: str,
    image_urls: list[str],
    cache_state: PrefixCacheState | None,
    retries: int,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, int], float]:
    cache_id = cache_state.response_id if cache_state else None
    dynamic_text = question_text + FEATURE_EXTRACTION_PROMPT_SUFFIX
    started = time.time()
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    last_error = ""
    repair_feedback: str | None = None
    for attempt in range(retries):
        use_cache = bool(cache_id and repair_feedback is None)
        prompt_text = dynamic_text if use_cache else FEATURE_EXTRACTION_PROMPT_PREFIX + "\n\n" + dynamic_text
        if repair_feedback:
            prompt_text += "\n\n【格式修复要求】\n" + repair_feedback + "\n请重新输出完整合法JSON。"
        payload: dict[str, Any] = {
            "model": MODEL_NAME,
            "input": [{"role": "user", "content": _content_with_images(prompt_text, image_urls)}],
            "thinking": {"type": "disabled"},
            "max_output_tokens": 5000,
        }
        if use_cache:
            payload["previous_response_id"] = cache_id
        if TEMPERATURE is not None:
            payload["temperature"] = TEMPERATURE
        try:
            status, body, error = await _post_response(session, payload, timeout)
            if status == 200 and body:
                current = _usage(body)
                for key in total_usage:
                    total_usage[key] += current[key]
                parsed = _parse_json_object(_extract_output_text(body))
                try:
                    raw_features = copy.deepcopy(parsed.get("features"))
                    normalized, log = normalize_stage1_rating(parsed)
                    enriched = enrich_stage1_rating(
                        normalized,
                        features_model_raw=raw_features,
                        normalization_log=log,
                    )
                except ValueError as exc:
                    if repair_feedback is None and attempt < retries - 1:
                        repair_feedback = f"上一次校验失败：{exc}\n上一次输出：\n{_json_block(parsed)}"
                        last_error = str(exc)
                        continue
                    raise
                return enriched, total_usage, time.time() - started
            last_error = f"HTTP {status}: {error[:400]}"
            if use_cache and "PreviousResponseNotFound" in error and cache_state:
                async with cache_state.refresh_lock:
                    refreshed = await create_prefix_cache(session, cache_state.cache_path, retries, timeout)
                    if not refreshed:
                        raise RuntimeError("第一阶段前缀缓存刷新失败")
                    cache_state.response_id = refreshed
                    cache_id = refreshed
                continue
            if status != 429 and status < 500:
                break
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"第一阶段请求失败：{last_error}")


async def call_stage2(
    *,
    session: aiohttp.ClientSession,
    question_text: str,
    image_urls: list[str],
    stage1: dict[str, Any],
    retries: int,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, int], float]:
    text = (
        VERIFICATION_PROMPT_PREFIX + "\n\n【题目信息】\n" + question_text
        + "\n\n【第一阶段与程序处理结果】\n" + _json_block(stage1)
        + VERIFICATION_PROMPT_SUFFIX
    )
    started = time.time()
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    last_error = ""
    for attempt in range(retries):
        payload: dict[str, Any] = {
            "model": MODEL_NAME,
            "input": [{"role": "user", "content": _content_with_images(text, image_urls)}],
            "thinking": {"type": "disabled"},
            "max_output_tokens": 3000,
        }
        if TEMPERATURE is not None:
            payload["temperature"] = TEMPERATURE
        try:
            status, body, error = await _post_response(session, payload, timeout)
            if status == 200 and body:
                current = _usage(body)
                for key in total_usage:
                    total_usage[key] += current[key]
                parsed = validate_verification(_parse_json_object(_extract_output_text(body)))
                return recalculate_verification(
                    stage1,
                    parsed,
                    allow_auto_adjustment=ENABLE_STAGE2_AUTO_ADJUST,
                ), total_usage, time.time() - started
            last_error = f"HTTP {status}: {error[:400]}"
            if status != 429 and status < 500:
                break
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2 ** attempt + random.random())
    raise RuntimeError(f"第二阶段请求失败：{last_error}")


async def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    async with FILE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def load_processed_ids(path: Path) -> set[str]:
    processed: set[str] = set()
    if not path.exists():
        return processed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
                if row.get("final_difficulty_level") and row.get("question_id") is not None:
                    processed.add(str(row["question_id"]))
            except (json.JSONDecodeError, TypeError):
                continue
    return processed


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"跳过第{line_number}行非法JSON：{exc}")
                continue
            if isinstance(value, dict):
                questions.append(value)
    return questions


def sample_questions_per_level(
    questions: list[dict[str, Any]], *, per_level: int, seed: int | None
) -> list[dict[str, Any]]:
    if per_level <= 0:
        raise ValueError("per_level 必须大于0")
    groups = {str(level): [] for level in range(1, 6)}
    for row in questions:
        label = str(row.get("difficulty") or "").strip()
        if label in groups:
            groups[label].append(row)
    insufficient = {label: len(rows) for label, rows in groups.items() if len(rows) < per_level}
    if insufficient:
        raise ValueError(f"以下旧档位不足每档{per_level}道：{insufficient}")
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for label in ("1", "2", "3", "4", "5"):
        sampled.extend(rng.sample(groups[label], per_level))
    rng.shuffle(sampled)
    return sampled


def build_stage2_fallback(
    output_base: dict[str, Any],
    stage1: dict[str, Any],
    error: Exception,
    usage1: dict[str, int],
    elapsed1: float,
) -> dict[str, Any]:
    return {
        **output_base,
        "pipeline_version": "high_chemistry_two_stage_v4",
        "model_name": MODEL_NAME,
        "difficulty_rating_stage1": stage1,
        "difficulty_level_step1": stage1["difficulty_level_step1"],
        "verification": None,
        "final_difficulty_level": stage1["difficulty_level_step1"],
        "final_adjustment": "第二阶段失败，维持第一阶段并转人工复核",
        "needs_manual_review": True,
        "pipeline_warning": f"stage2_failed: {type(error).__name__}: {error}",
        "api_stage1_time_seconds": round(elapsed1, 2),
        "api_stage1_usage": usage1,
    }


async def process_question(
    *,
    source: dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    error_path: Path,
    cache_state: PrefixCacheState | None,
    image_mode: str,
    retries: int,
    timeout: int,
) -> None:
    async with semaphore:
        prepared = prepare_question(source, image_mode=image_mode)
        output_base = copy.deepcopy(prepared.question)
        output_base["input_quality"] = prepared.input_quality
        output_base["selected_image_urls"] = prepared.selected_image_urls
        if prepared.source_difficulty_untrusted is not None:
            output_base["source_difficulty_untrusted"] = prepared.source_difficulty_untrusted
        question_text = construct_question_text(prepared.question, prepared.input_quality)
        stage1 = None
        usage1 = None
        elapsed1 = None
        try:
            stage1, usage1, elapsed1 = await call_stage1(
                session=session, question_text=question_text,
                image_urls=prepared.selected_image_urls, cache_state=cache_state,
                retries=retries, timeout=timeout,
            )
            try:
                verification, usage2, elapsed2 = await call_stage2(
                    session=session, question_text=question_text,
                    image_urls=prepared.selected_image_urls, stage1=stage1,
                    retries=retries, timeout=timeout,
                )
            except Exception as exc:
                await append_jsonl(
                    output_path,
                    build_stage2_fallback(output_base, stage1, exc, usage1, elapsed1),
                )
                await append_jsonl(error_path, {
                    **output_base, "stage": "stage2", "error_type": type(exc).__name__, "error": str(exc)
                })
                return
            final = finalize_level(
                current_level=stage1["difficulty_level_step1"],
                reasonableness=verification["rating_reasonableness"],
                model_suggested_level=verification["adjusted_difficulty_level"],
                multiplier_reasonableness=verification["multiplier_reasonableness"],
                input_sufficiency=prepared.input_quality["input_sufficiency"],
                original_high_count=stage1["high_difficulty_feature_count"],
                reviewed_high_count=verification["reviewed_high_difficulty_feature_count"],
                enable_auto_adjust=ENABLE_STAGE2_AUTO_ADJUST,
            )
            total_usage = {key: usage1[key] + usage2[key] for key in usage1}
            await append_jsonl(output_path, {
                **output_base,
                "pipeline_version": "high_chemistry_two_stage_v4",
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "stage2_auto_adjustment_enabled": ENABLE_STAGE2_AUTO_ADJUST,
                "difficulty_rating_stage1": stage1,
                "difficulty_level_step1": stage1["difficulty_level_step1"],
                "verification": verification,
                "reviewed_high_difficulty_feature_count": verification["reviewed_high_difficulty_feature_count"],
                "model_suggested_level": final.model_suggested_level,
                "final_difficulty_level": final.final_level,
                "final_adjustment": final.adjustment_desc,
                "needs_manual_review": final.needs_manual_review or verification["review_requires_manual"],
                "api_stage1_time_seconds": round(elapsed1, 2),
                "api_stage2_time_seconds": round(elapsed2, 2),
                "api_stage1_usage": usage1,
                "api_stage2_usage": usage2,
                "api_total_usage": total_usage,
            })
        except Exception as exc:
            await append_jsonl(error_path, {
                **output_base,
                "stage": "stage1" if stage1 is None else "pipeline",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "difficulty_rating_stage1": stage1,
                "api_stage1_usage": usage1,
                "api_stage1_time_seconds": elapsed1,
            })


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="高中化学两阶段难度评级")
    parser.add_argument("-i", "--input", default=str(DEFAULT_INPUT))
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("-e", "--errors", default=str(DEFAULT_ERRORS))
    parser.add_argument("-p", "--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("-c", "--concurrency", type=int, default=30)
    sample = parser.add_mutually_exclusive_group()
    sample.add_argument("-n", "--limit", "--num", type=int, default=None)
    sample.add_argument("--per-level", type=int, default=None)
    parser.add_argument("-t", "--timeout", type=int, default=300)
    parser.add_argument("-r", "--retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--image-mode", choices=("off", "auto", "all"), default="auto")
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE))
    parser.add_argument("--task-batch-size", type=int, default=1000)
    return parser


async def run(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)
    error_path = Path(args.errors)
    cache_path = Path(args.cache_file)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    load_prompt_config(args.prompt)
    questions = load_questions(input_path)
    print(f"加载题目：{len(questions)}道")
    if args.per_level is not None:
        questions = sample_questions_per_level(questions, per_level=args.per_level, seed=args.seed)
    elif args.limit is not None:
        rng = random.Random(args.seed)
        questions = rng.sample(questions, min(args.limit, len(questions)))
    processed = load_processed_ids(output_path)
    pending = [row for row in questions if str(row.get("question_id") or "") not in processed]
    print(f"已完成：{len(processed)}；待处理：{len(pending)}")
    if not pending:
        return
    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=max(2, args.concurrency * 2))
    async with aiohttp.ClientSession(connector=connector) as session:
        cache_state = None
        if not args.no_cache:
            cache_id = await get_or_create_prefix_cache(session, cache_path, args.retries, args.timeout)
            if not cache_id:
                raise RuntimeError("第一阶段前缀缓存初始化失败")
            cache_state = PrefixCacheState(cache_id, cache_path)
        batch_size = max(args.concurrency, args.task_batch_size)
        progress = tqdm(total=len(pending), desc="High Chemistry Pipeline", unit="item")
        for start in range(0, len(pending), batch_size):
            tasks = [asyncio.create_task(process_question(
                source=row, session=session, semaphore=semaphore,
                output_path=output_path, error_path=error_path,
                cache_state=cache_state, image_mode=args.image_mode,
                retries=args.retries, timeout=args.timeout,
            )) for row in pending[start:start + batch_size]]
            for task in asyncio.as_completed(tasks):
                await task
                progress.update(1)
        progress.close()
    print(f"结果：{output_path.resolve()}")
    print(f"错误：{error_path.resolve()}")


def main() -> None:
    args = build_parser().parse_args()
    started = time.time()
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        print("收到中断信号，已安全退出")
    finally:
        print(f"耗时：{(time.time() - started) / 60:.2f}分钟")


if __name__ == "__main__":
    main()
