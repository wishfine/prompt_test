# -*- coding: utf-8 -*-
"""高中物理两阶段难度评级 Pipeline。

流程：
  1. 模型提取结构化 features，并给出乘数前原始正确率；
  2. 程序检测高难特征，按数量应用 1.00 / 0.85 / 0.70 乘数；
  3. 程序按连续正确率区间映射第一步五档；
  4. 第二次模型调用复核 features、正确率、高难触发、重复计数、乘数和档位；
  5. 程序依据“合理/偏高/偏低”最多调整一档，并标记人工复核项。

支持 OpenAI-compatible Responses API、第一阶段前缀缓存、并发、重试、
JSONL 断点续跑、题干/解析图片输入及 token 统计。不会向模型发送原始
``difficulty`` 字段；该字段只以 ``source_difficulty_untrusted`` 留在输出中。
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
    import aiofiles
    import aiohttp
    import json_repair
    from dotenv import load_dotenv
    from tqdm.asyncio import tqdm
except ImportError as exc:  # pragma: no cover - 服务器 venv 中执行
    raise RuntimeError(
        "缺少运行依赖，请在项目 venv 中安装/启用 aiofiles、aiohttp、"
        "json-repair、python-dotenv、tqdm"
    ) from exc

from high_physics_pipeline_core import (
    HIGH_DIFFICULTY_FEATURE_NAMES,
    enrich_stage1_rating,
    finalize_level,
    normalize_level,
    prepare_question,
)


load_dotenv()

API_KEY = os.getenv("API_KEY", "not-needed")
BASE_URL = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1").rstrip("/") + "/"
MODEL_NAME = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")
TEMPERATURE_RAW = os.getenv("TEMPERATURE", "")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "high-physics-sample25k.jsonl"
DEFAULT_PROMPT = ROOT / "prompts" / "高中物理难度打标提示词.txt"
DEFAULT_OUTPUT = ROOT / "outputs" / "model_runs" / "high_physics_two_stage.jsonl"
DEFAULT_ERRORS = ROOT / "outputs" / "model_runs" / "high_physics_two_stage_errors.jsonl"
DEFAULT_CACHE = ROOT / "outputs" / "cache" / "high_physics_stage1_prefix_cache.json"

CACHE_EXPIRE_SECONDS = 5 * 24 * 3600
FILE_LOCK = asyncio.Lock()
CACHE_LOCK = asyncio.Lock()
CACHE_CREATE_LOCK = asyncio.Lock()

FEATURE_EXTRACTION_PROMPT_PREFIX = ""
FEATURE_EXTRACTION_PROMPT_SUFFIX = ""
VERIFICATION_PROMPT_PREFIX = ""
VERIFICATION_PROMPT_SUFFIX = ""


class PrefixCacheState:
    """进程内共享前缀缓存 ID，避免每道题重复读取缓存文件。"""

    def __init__(self, response_id: str, cache_path: Path):
        self.response_id = response_id
        self.cache_path = cache_path
        self.refresh_lock = asyncio.Lock()


def resolve_temperature(model_name: str, raw_value: str) -> float | None:
    """Doubao Lite 服务端固定 temperature=1；其他模型遵循环境变量。"""
    if "lite" in model_name.lower():
        return 1.0
    raw_value = str(raw_value or "").strip()
    return float(raw_value) if raw_value else None


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
    required = (
        "FEATURE_EXTRACTION_PROMPT_PREFIX",
        "FEATURE_EXTRACTION_PROMPT_SUFFIX",
        "VERIFICATION_PROMPT_PREFIX",
        "VERIFICATION_PROMPT_SUFFIX",
    )
    missing = [name for name in required if not namespace.get(name)]
    if missing:
        raise ValueError(f"Prompt 缺少变量：{', '.join(missing)}")
    FEATURE_EXTRACTION_PROMPT_PREFIX = str(namespace[required[0]])
    FEATURE_EXTRACTION_PROMPT_SUFFIX = str(namespace[required[1]])
    VERIFICATION_PROMPT_PREFIX = str(namespace[required[2]])
    VERIFICATION_PROMPT_SUFFIX = str(namespace[required[3]])
    print("成功加载高中物理两阶段 Prompt")


def _prefix_hash() -> str:
    payload = f"{MODEL_NAME}\n{FEATURE_EXTRACTION_PROMPT_PREFIX}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def _read_cache(path: Path) -> dict[str, Any]:
    async with CACHE_LOCK:
        if not path.exists():
            return {}
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as handle:
                return json.loads(await handle.read())
        except Exception:
            return {}


async def _write_cache(path: Path, value: dict[str, Any]) -> None:
    async with CACHE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w", encoding="utf-8") as handle:
            await handle.write(json.dumps(value, ensure_ascii=False, indent=2))


def _valid_cache(entry: dict[str, Any]) -> bool:
    return bool(
        entry
        and entry.get("model_name") == MODEL_NAME
        and entry.get("prefix_hash") == _prefix_hash()
        and int(entry.get("expire_at", 0)) > int(time.time())
        and entry.get("response_id")
    )


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }


def _extract_output_text(response_json: dict[str, Any]) -> str:
    pieces: list[str] = []
    for item in response_json.get("output") or []:
        if item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if content.get("type") == "output_text":
                pieces.append(str(content.get("text") or ""))
    return "\n".join(pieces).strip()


def _parse_json_object(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("模型响应为空")
    repaired = json_repair.repair_json(text, return_objects=True)
    if not isinstance(repaired, dict):
        raise ValueError("模型响应不是 JSON 对象")
    return repaired


def _usage(response_json: dict[str, Any]) -> dict[str, int]:
    usage = response_json.get("usage") or {}
    return {
        "input_tokens": int(usage.get("input_tokens") or 0),
        "output_tokens": int(usage.get("output_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
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
            return response.status, json.loads(text), text
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
            status, body, error_text = await _post_response(
                session, payload, timeout
            )
            if status == 200 and body and body.get("id"):
                response_id = str(body["id"])
                await _write_cache(
                    cache_path,
                    {
                        "response_id": response_id,
                        "expire_at": expire_at,
                        "prefix_hash": _prefix_hash(),
                        "model_name": MODEL_NAME,
                    },
                )
                print(f"第一阶段前缀缓存创建成功：{response_id}")
                return response_id
            print(f"创建前缀缓存失败 ({status})：{error_text[:240]}")
            if 400 <= status < 500:
                return None
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt == retries - 1:
                print(f"创建前缀缓存最终失败：{exc}")
                return None
        await asyncio.sleep(2**attempt + random.random())
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
    """构造发送给模型的题目信息；输入已由 prepare_question 删除 difficulty。"""
    return (
        "【输入质量】\n"
        + _json_block(input_quality)
        + "\n\n【题目 JSON】\n"
        + _json_block(question)
    )


def _content_with_images(text: str, image_urls: list[str]) -> str | list[dict[str, Any]]:
    if not image_urls:
        return text
    content: list[dict[str, Any]] = [{"type": "input_text", "text": text}]
    for url in image_urls:
        content.append({"type": "input_image", "image_url": url})
    return content


def validate_verification(result: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise ValueError("第二阶段响应必须为对象")
    required = (
        "difficulty_source",
        "feature_corrections",
        "missed_features",
        "reviewed_high_difficulty_features",
        "multiplier_reasonableness",
        "rating_reasonableness",
        "adjusted_difficulty_level",
        "analysis",
    )
    missing = [field for field in required if field not in result]
    if missing:
        raise ValueError(f"第二阶段缺少字段：{', '.join(missing)}")
    for field in ("difficulty_source", "analysis"):
        value = result[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} 必须为非空字符串")
    if not isinstance(result["feature_corrections"], list):
        raise ValueError("feature_corrections 必须为数组")
    correction_fields = {"field", "from", "to", "evidence"}
    for index, correction in enumerate(result["feature_corrections"]):
        if not isinstance(correction, dict):
            raise ValueError(f"feature_corrections[{index}] 必须为对象")
        correction_missing = correction_fields - correction.keys()
        if correction_missing:
            raise ValueError(
                f"feature_corrections[{index}] 缺少字段："
                f"{sorted(correction_missing)}"
            )
    if not isinstance(result["missed_features"], list):
        raise ValueError("missed_features 必须为数组")
    if any(not isinstance(name, str) for name in result["missed_features"]):
        raise ValueError("missed_features 每项必须为字符串")
    reviewed = result["reviewed_high_difficulty_features"]
    if not isinstance(reviewed, list):
        raise ValueError("reviewed_high_difficulty_features 必须为数组")
    if any(not isinstance(name, str) for name in reviewed):
        raise ValueError(
            "reviewed_high_difficulty_features 每项必须为字符串"
        )
    if len(reviewed) != len(set(reviewed)):
        raise ValueError("reviewed_high_difficulty_features 不得重复")
    invalid_high = [
        name for name in reviewed if name not in HIGH_DIFFICULTY_FEATURE_NAMES
    ]
    if invalid_high:
        raise ValueError(f"第二阶段含非法高难特征：{invalid_high}")
    if result["multiplier_reasonableness"] not in {"合理", "不合理"}:
        raise ValueError("multiplier_reasonableness 只能为合理/不合理")
    if result["rating_reasonableness"] not in {"合理", "偏高", "偏低"}:
        raise ValueError("rating_reasonableness 只能为合理/偏高/偏低")
    level = normalize_level(result["adjusted_difficulty_level"])
    if not level:
        raise ValueError("adjusted_difficulty_level 必须为难度1档到难度5档")
    normalized = copy.deepcopy(result)
    normalized["adjusted_difficulty_level"] = level
    return normalized


def build_pipeline_error(
    *,
    output_base: dict[str, Any],
    error: Exception,
    stage1: dict[str, Any] | None = None,
    stage1_usage: dict[str, int] | None = None,
    stage1_elapsed: float | None = None,
) -> dict[str, Any]:
    """构造可续跑的错误记录；第二阶段失败时保留已付费的第一阶段结果。"""
    record = {
        **copy.deepcopy(output_base),
        "pipeline_version": "high_physics_two_stage_v2",
        "model_name": MODEL_NAME,
        "failed_stage": "stage2" if stage1 is not None else "stage1",
        "rating_error": str(error),
    }
    if stage1 is not None:
        record["difficulty_rating_stage1"] = stage1
        record["difficulty_level_step1"] = stage1.get(
            "difficulty_level_step1"
        )
        record["api_stage1_usage"] = stage1_usage or {}
        record["api_stage1_time_seconds"] = (
            round(stage1_elapsed, 2) if stage1_elapsed is not None else None
        )
    return record


async def call_stage1(
    *,
    session: aiohttp.ClientSession,
    question_text: str,
    image_urls: list[str],
    cache_state: PrefixCacheState | None,
    retries: int,
    timeout: int,
) -> tuple[dict[str, Any], dict[str, int], float]:
    cache_id = cache_state.response_id if cache_state is not None else None

    dynamic_text = question_text + FEATURE_EXTRACTION_PROMPT_SUFFIX
    started = time.time()
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    last_error = ""
    for attempt in range(retries):
        prompt_text = (
            dynamic_text
            if cache_id
            else FEATURE_EXTRACTION_PROMPT_PREFIX + "\n\n" + dynamic_text
        )
        payload: dict[str, Any] = {
            "model": MODEL_NAME,
            "input": [
                {
                    "role": "user",
                    "content": _content_with_images(prompt_text, image_urls),
                }
            ],
            "thinking": {"type": "disabled"},
            "max_output_tokens": 4000,
        }
        if cache_id:
            payload["previous_response_id"] = cache_id
        if TEMPERATURE is not None:
            payload["temperature"] = TEMPERATURE
        try:
            status, body, error_text = await _post_response(session, payload, timeout)
            if status == 200 and body:
                current_usage = _usage(body)
                for key in total_usage:
                    total_usage[key] += current_usage[key]
                parsed = _parse_json_object(_extract_output_text(body))
                enriched = enrich_stage1_rating(parsed)
                return enriched, total_usage, time.time() - started
            last_error = f"HTTP {status}: {error_text[:400]}"
            if cache_id and "PreviousResponseNotFound" in error_text:
                if cache_state is None:
                    raise RuntimeError("第一阶段缓存状态缺失")
                async with cache_state.refresh_lock:
                    if cache_state.response_id == cache_id:
                        refreshed = await create_prefix_cache(
                            session,
                            cache_state.cache_path,
                            retries,
                            timeout,
                        )
                        if not refreshed:
                            raise RuntimeError("第一阶段前缀缓存刷新失败")
                        cache_state.response_id = refreshed
                    cache_id = cache_state.response_id
                continue
            if status != 429 and status < 500:
                break
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2**attempt + random.random())
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
    review_text = (
        VERIFICATION_PROMPT_PREFIX
        + "\n\n【题目信息】\n"
        + question_text
        + "\n\n【第一阶段与程序处理结果】\n"
        + _json_block(stage1)
        + VERIFICATION_PROMPT_SUFFIX
    )
    started = time.time()
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    last_error = ""
    for attempt in range(retries):
        payload: dict[str, Any] = {
            "model": MODEL_NAME,
            "input": [
                {
                    "role": "user",
                    "content": _content_with_images(review_text, image_urls),
                }
            ],
            "thinking": {"type": "disabled"},
            "max_output_tokens": 2500,
        }
        if TEMPERATURE is not None:
            payload["temperature"] = TEMPERATURE
        try:
            status, body, error_text = await _post_response(session, payload, timeout)
            if status == 200 and body:
                current_usage = _usage(body)
                for key in total_usage:
                    total_usage[key] += current_usage[key]
                parsed = _parse_json_object(_extract_output_text(body))
                return (
                    validate_verification(parsed),
                    total_usage,
                    time.time() - started,
                )
            last_error = f"HTTP {status}: {error_text[:400]}"
            if status != 429 and status < 500:
                break
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2**attempt + random.random())
    raise RuntimeError(f"第二阶段请求失败：{last_error}")


async def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    async with FILE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "a", encoding="utf-8") as handle:
            await handle.write(json.dumps(value, ensure_ascii=False) + "\n")


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
        question_text = construct_question_text(
            prepared.question, prepared.input_quality
        )
        output_base = copy.deepcopy(prepared.question)
        if prepared.source_difficulty_untrusted is not None:
            output_base["source_difficulty_untrusted"] = (
                prepared.source_difficulty_untrusted
            )
        output_base["input_quality"] = prepared.input_quality
        output_base["selected_image_urls"] = prepared.selected_image_urls
        stage1: dict[str, Any] | None = None
        usage1: dict[str, int] | None = None
        elapsed1: float | None = None
        try:
            stage1, usage1, elapsed1 = await call_stage1(
                session=session,
                question_text=question_text,
                image_urls=prepared.selected_image_urls,
                cache_state=cache_state,
                retries=retries,
                timeout=timeout,
            )
            verification, usage2, elapsed2 = await call_stage2(
                session=session,
                question_text=question_text,
                image_urls=prepared.selected_image_urls,
                stage1=stage1,
                retries=retries,
                timeout=timeout,
            )
            reviewed_high_count = len(
                verification["reviewed_high_difficulty_features"]
            )
            final = finalize_level(
                current_level=stage1["difficulty_level_step1"],
                reasonableness=verification["rating_reasonableness"],
                model_suggested_level=verification["adjusted_difficulty_level"],
                multiplier_reasonableness=verification[
                    "multiplier_reasonableness"
                ],
                input_sufficiency=prepared.input_quality["input_sufficiency"],
                original_high_count=stage1["high_difficulty_feature_count"],
                reviewed_high_count=reviewed_high_count,
            )
            total_usage = {
                key: usage1[key] + usage2[key]
                for key in ("input_tokens", "output_tokens", "total_tokens")
            }
            result = {
                **output_base,
                "pipeline_version": "high_physics_two_stage_v2",
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "difficulty_rating_stage1": stage1,
                "difficulty_level_step1": stage1["difficulty_level_step1"],
                "verification": verification,
                "reviewed_high_difficulty_feature_count": reviewed_high_count,
                "model_suggested_level": final.model_suggested_level,
                "final_difficulty_level": final.final_level,
                "final_adjustment": final.adjustment_desc,
                "needs_manual_review": final.needs_manual_review,
                "api_stage1_time_seconds": round(elapsed1, 2),
                "api_stage2_time_seconds": round(elapsed2, 2),
                "api_stage1_usage": usage1,
                "api_stage2_usage": usage2,
                "api_total_usage": total_usage,
            }
            await append_jsonl(output_path, result)
        except Exception as exc:
            await append_jsonl(
                error_path,
                build_pipeline_error(
                    output_base=output_base,
                    error=exc,
                    stage1=stage1,
                    stage1_usage=usage1,
                    stage1_elapsed=elapsed1,
                ),
            )


def load_questions(path: Path) -> list[dict[str, Any]]:
    questions: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"跳过第 {line_number} 行非法 JSON：{exc}")
                continue
            if isinstance(row, dict):
                questions.append(row)
    return questions


def sample_questions_per_level(
    questions: list[dict[str, Any]],
    *,
    per_level: int,
    seed: int | None,
) -> list[dict[str, Any]]:
    """仅在抽样阶段读取旧 difficulty，按五档等量抽样；不修改原始记录。"""
    if per_level <= 0:
        raise ValueError("per_level 必须大于 0")
    groups: dict[str, list[dict[str, Any]]] = {
        str(level): [] for level in range(1, 6)
    }
    for row in questions:
        label = str(row.get("difficulty") or "").strip()
        if label in groups:
            groups[label].append(row)
    insufficient = {
        label: len(rows)
        for label, rows in groups.items()
        if len(rows) < per_level
    }
    if insufficient:
        raise ValueError(
            f"以下档位不足每档 {per_level} 道：{insufficient}"
        )
    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for label in ("1", "2", "3", "4", "5"):
        sampled.extend(rng.sample(groups[label], per_level))
    rng.shuffle(sampled)
    return sampled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="高中物理两阶段难度评级")
    parser.add_argument("-i", "--input", default=str(DEFAULT_INPUT))
    parser.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("-e", "--errors", default=str(DEFAULT_ERRORS))
    parser.add_argument("-p", "--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("-c", "--concurrency", type=int, default=30)
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument("-n", "--limit", "--num", type=int, default=None)
    sample_group.add_argument(
        "--per-level",
        type=int,
        default=None,
        help="按原始旧标签1—5档各抽取指定题数；标签仅用于抽样，不发送给模型",
    )
    parser.add_argument("-t", "--timeout", type=int, default=300)
    parser.add_argument("-r", "--retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--image-mode",
        choices=("off", "auto", "all"),
        default="auto",
        help="off=不传图片；auto=文本引用图片/文本不足时传；all=传全部图片",
    )
    parser.add_argument("--cache-file", default=str(DEFAULT_CACHE))
    parser.add_argument(
        "--task-batch-size",
        type=int,
        default=1000,
        help="每批创建的 asyncio 任务数，避免全量数据一次性占用内存",
    )
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
    print(f"加载题目：{len(questions)} 道")

    if args.per_level is not None:
        questions = sample_questions_per_level(
            questions,
            per_level=args.per_level,
            seed=args.seed,
        )
        print(
            f"分层抽样：每档 {args.per_level} 道，共 {len(questions)} 道，"
            f"seed={args.seed}"
        )
    elif args.limit is not None:
        if args.seed is not None:
            random.seed(args.seed)
        questions = random.sample(questions, min(args.limit, len(questions)))
        print(f"抽样处理：{len(questions)} 道，seed={args.seed}")

    processed = load_processed_ids(output_path)
    pending = [
        row
        for row in questions
        if str(row.get("question_id") or "") not in processed
    ]
    print(f"已完成：{len(processed)}；待处理：{len(pending)}")
    print(
        f"模型={MODEL_NAME}，temperature={TEMPERATURE}，"
        f"cache={'off' if args.no_cache else 'on'}，image_mode={args.image_mode}"
    )
    if not pending:
        return

    connector = aiohttp.TCPConnector(limit=max(2, args.concurrency * 2))
    semaphore = asyncio.Semaphore(args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        cache_state: PrefixCacheState | None = None
        if not args.no_cache:
            cache_id = await get_or_create_prefix_cache(
                session, cache_path, args.retries, args.timeout
            )
            if not cache_id:
                raise RuntimeError("第一阶段前缀缓存初始化失败")
            cache_state = PrefixCacheState(cache_id, cache_path)
        batch_size = max(args.concurrency, args.task_batch_size)
        progress = tqdm(total=len(pending), desc="High Physics Pipeline", unit="item")
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            tasks = [
                asyncio.create_task(
                    process_question(
                        source=row,
                        session=session,
                        semaphore=semaphore,
                        output_path=output_path,
                        error_path=error_path,
                        cache_state=cache_state,
                        image_mode=args.image_mode,
                        retries=args.retries,
                        timeout=args.timeout,
                    )
                )
                for row in batch
            ]
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
        print(f"耗时：{(time.time() - started) / 60:.2f} 分钟")


if __name__ == "__main__":
    main()
