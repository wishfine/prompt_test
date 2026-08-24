# -*- coding: utf-8 -*-
"""高中化学独立两阶段难度评级 Pipeline（V21：连续分数与冻结结构约束解耦）。

流程：
  1. 模型提取结构化 features，并给出连续原始预测正确率；
  2. 程序冻结 features，检测高难特征并应用乘数，得到 score_level；
  3. 程序由冻结 features 派生 StructuralLevelConstraint (floor/ceiling)；
  4. 程序应用结构上下限约束确定 Step 1 档位（严格限制跨档，杜绝微小分数噪声跨档）；
  5. 第二阶段独立模型作为结构审计器，复核 features 与高难特征；
  6. 仅在有合法 feature 修正时程序重算结构约束，最多调整一档，并标记人工复核项。

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

import high_chemistry_pipeline_core as chemistry_core
from high_chemistry_pipeline_core import (
    FinalizationResult,
    HIGH_DIFFICULTY_FEATURE_NAMES,
    REQUIRED_FEATURE_FIELDS,
    Stage1SemanticConsistencyError,
    build_stage1_output_schema,
    build_stage1_semantic_repair_schema,
    build_stage2_output_schema,
    enrich_stage1_rating,
    finalize_level as _chemistry_finalize_level,
    normalize_stage1_rating,
    prepare_question,
    recalculate_verification,
    validate_structural_revision_evidence,
)


load_dotenv()

API_KEY = os.getenv("API_KEY", "not-needed")
BASE_URL = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1").rstrip("/") + "/"
MODEL_NAME = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")
TEMPERATURE_RAW = os.getenv("TEMPERATURE", "")
ENABLE_STAGE2_AUTO_ADJUST = (
    os.getenv("ENABLE_STAGE2_AUTO_ADJUST", "1").strip() == "1"
)
ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER = (
    os.getenv("ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER", "1").strip()
    == "1"
)
chemistry_core.CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER_ENABLED = (
    ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data" / "high-chemistry-sample25k.jsonl"
DEFAULT_PROMPT = ROOT / "prompts" / "高中化学难度打标提示词.txt"
DEFAULT_OUTPUT = ROOT / "outputs" / "model_runs" / "high_chemistry_two_stage.jsonl"
DEFAULT_ERRORS = ROOT / "outputs" / "model_runs" / "high_chemistry_two_stage_errors.jsonl"
DEFAULT_CACHE = ROOT / "outputs" / "cache" / "high_chemistry_stage1_prefix_cache.json"
PIPELINE_VERSION = "high_chemistry_two_stage_v22_candidate_10"
PROMPT_VERSION = "high_chemistry_prompt_v22_candidate_10"
STRUCTURAL_CONSTRAINT_VERSION = "structural_constraint_v22_candidate_10"
PROMPT_SHA256 = ""
CORE_SHA256 = hashlib.sha256(
    (ROOT / "src" / "high_chemistry_pipeline_core.py").read_bytes()
).hexdigest()
SUBJECT_DISPLAY_NAME = "高中化学"
PROGRESS_DESCRIPTION = "High Chemistry Pipeline"

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


def _validate_prompt_stage_separation(
    stage1_prefix: str,
    stage1_suffix: str,
    stage2_prefix: str,
) -> None:
    """Fail fast when program-only postprocessing rules leak into stage 1."""
    del stage2_prefix  # Stage 2 is intentionally allowed to see program metadata.
    stage1_prompt = f"{stage1_prefix}\n{stage1_suffix}"
    forbidden_markers = (
        "0.85",
        "0.70",
        "乘数",
        "高难特征",
        "high_difficulty_feature_count",
        "multiplier_applied",
        "difficulty_level_step1",
        "程序稍后会用",
        "难度1档",
        "难度2档",
        "难度3档",
        "难度4档",
        "难度5档",
    )
    leaked = [marker for marker in forbidden_markers if marker in stage1_prompt]
    if leaked:
        raise ValueError(
            "第一阶段 Prompt 泄露后处理规则：" + "、".join(leaked)
        )


def _restrict_stage1_model_output(value: dict[str, Any]) -> dict[str, Any]:
    """Keep only fields that belong to the stage-1 model contract."""
    return {
        key: copy.deepcopy(value.get(key))
        for key in ("features", "reason", "predicted_accuracy")
    }


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
    _validate_prompt_stage_separation(
        str(namespace[required[0]]),
        str(namespace[required[1]]),
        str(namespace[required[2]]),
    )
    global PROMPT_SHA256
    PROMPT_SHA256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    FEATURE_EXTRACTION_PROMPT_PREFIX = str(namespace[required[0]])
    FEATURE_EXTRACTION_PROMPT_SUFFIX = str(namespace[required[1]])
    VERIFICATION_PROMPT_PREFIX = str(namespace[required[2]])
    VERIFICATION_PROMPT_SUFFIX = str(namespace[required[3]])
    print(f"成功加载{SUBJECT_DISPLAY_NAME}两阶段 Prompt")


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
        "has_structural_revision",
        "adjacent_boundary_review",
        "confidence",
        "reviewed_original_predicted_accuracy",
        "reviewed_high_difficulty_features",
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
    normalized = copy.deepcopy(result)
    correction_fields = {"field", "from", "to", "evidence"}
    valid_corrections: list[dict[str, Any]] = []
    normalization_log: list[dict[str, Any]] = []
    for index, correction in enumerate(result["feature_corrections"]):
        if not isinstance(correction, dict):
            raise ValueError(f"feature_corrections[{index}] 必须为对象")
        correction_missing = correction_fields - correction.keys()
        if correction_missing:
            raise ValueError(
                f"feature_corrections[{index}] 缺少字段："
                f"{sorted(correction_missing)}"
            )
        if correction["field"] not in REQUIRED_FEATURE_FIELDS:
            normalization_log.append(
                {
                    "action": "ignore_non_feature_correction",
                    "field": correction["field"],
                    "evidence": correction["evidence"],
                }
            )
            continue
        valid_corrections.append(copy.deepcopy(correction))
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
    try:
        reviewed_accuracy = float(
            result["reviewed_original_predicted_accuracy"]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "reviewed_original_predicted_accuracy 必须为数值"
        ) from exc
    if not 0.0 <= reviewed_accuracy <= 100.0:
        raise ValueError(
            "reviewed_original_predicted_accuracy 必须在 0 到 100 之间"
        )
    normalized["feature_corrections"] = valid_corrections
    normalized["verification_normalization_log"] = normalization_log
    if not isinstance(result["has_structural_revision"], bool):
        raise ValueError("has_structural_revision 必须为布尔值")
    boundary_review = result["adjacent_boundary_review"]
    if not isinstance(boundary_review, dict):
        raise ValueError("adjacent_boundary_review 必须为对象")
    boundary_fields = {
        "boundaries_checked",
        "verdict",
        "decisive_evidence",
    }
    boundary_missing = boundary_fields - boundary_review.keys()
    if boundary_missing:
        raise ValueError(
            "adjacent_boundary_review 缺少字段："
            f"{sorted(boundary_missing)}"
        )
    checked = boundary_review["boundaries_checked"]
    if (
        not isinstance(checked, list)
        or not checked
        or any(
            value not in {"88边界", "85边界", "58边界", "38边界"}
            for value in checked
        )
    ):
        raise ValueError("boundaries_checked 必须为非空合法边界数组")
    if boundary_review["verdict"] not in {
        "维持",
        "应更简单一档",
        "应更难一档",
    }:
        raise ValueError("adjacent_boundary_review.verdict 含非法值")
    decisive_evidence = boundary_review["decisive_evidence"]
    if (
        not isinstance(decisive_evidence, list)
        or not decisive_evidence
        or any(
            not isinstance(value, str) or not value.strip()
            for value in decisive_evidence
        )
    ):
        raise ValueError("decisive_evidence 必须为非空字符串数组")
    if result["confidence"] not in {"高", "中", "低"}:
        raise ValueError("confidence 只能是高、中或低")
    normalized["reviewed_original_predicted_accuracy"] = reviewed_accuracy
    validate_structural_revision_evidence(normalized)
    overlap_review = normalized.get("high_feature_overlap_review")
    if not isinstance(overlap_review, list):
        raise ValueError("high_feature_overlap_review 必须为数组")
    for index, item in enumerate(overlap_review):
        if not isinstance(item, dict):
            raise ValueError(f"high_feature_overlap_review[{index}] 必须为对象")
        missing_overlap = {"features", "resolution", "reason"} - item.keys()
        if missing_overlap:
            raise ValueError(
                f"high_feature_overlap_review[{index}] 缺少字段："
                f"{sorted(missing_overlap)}"
            )
        if not isinstance(item["features"], list) or any(
            name not in HIGH_DIFFICULTY_FEATURE_NAMES
            for name in item["features"]
        ):
            raise ValueError(
                f"high_feature_overlap_review[{index}].features 含非法值"
            )
    input_review = normalized.get("input_sufficiency_review")
    if not isinstance(input_review, dict):
        raise ValueError("input_sufficiency_review 必须为对象")
    if input_review.get("status") not in {"充分", "部分缺失", "信息不足"}:
        raise ValueError("input_sufficiency_review.status 含非法值")
    missing_information = input_review.get("missing_information")
    if not isinstance(missing_information, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in missing_information
    ):
        raise ValueError(
            "input_sufficiency_review.missing_information 必须为字符串数组"
        )
    return normalized


def finalize_verified_level(
    *,
    current_level: str,
    reasonableness: str,
    model_suggested_level: Any,
    multiplier_reasonableness: str,
    input_sufficiency: str,
    original_high_count: int | None = None,
    reviewed_high_count: int | None = None,
) -> FinalizationResult:
    """化学自有的最多一档改档适配器。"""
    del original_high_count, reviewed_high_count
    action = {
        "合理": "维持",
        "偏高": "建议降一档",
        "偏低": "建议升一档",
    }.get(reasonableness, "维持")
    result = _chemistry_finalize_level(
        current_level=current_level,
        review_action=action,
        model_suggested_level=model_suggested_level,
        input_sufficiency=input_sufficiency,
        auto_adjustment_enabled=ENABLE_STAGE2_AUTO_ADJUST,
    )
    if multiplier_reasonableness != "合理":
        return FinalizationResult(
            final_level=current_level,
            needs_manual_review=True,
            model_suggested_level=result.model_suggested_level,
            adjustment_desc=(
                f"乘数复核不一致·维持{current_level}·转人工复核"
            ),
            auto_adjustment_applied=False,
        )
    return result


class Stage2CallError(RuntimeError):
    """第二阶段调用或校验异常，携带完整 debug 诊断信息。"""

    def __init__(
        self,
        message: str,
        *,
        http_status: int | None = None,
        output_text_length: int | None = None,
        output_text_tail: str | None = None,
        parsed_keys: list[str] | None = None,
        usage: dict[str, int] | None = None,
        incomplete_details: Any | None = None,
        validation_error: str | None = None,
    ) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.output_text_length = output_text_length
        self.output_text_tail = output_text_tail
        self.parsed_keys = parsed_keys
        self.usage = usage
        self.incomplete_details = incomplete_details
        self.validation_error = validation_error


def build_pipeline_error(
    *,
    output_base: dict[str, Any],
    error: Exception,
    stage1: dict[str, Any] | None = None,
    stage1_usage: dict[str, int] | None = None,
    stage1_elapsed: float | None = None,
) -> dict[str, Any]:
    """构造可续跑的错误记录；第二阶段失败时保留已付费的第一阶段结果及完整元数据。"""
    record = {
        **copy.deepcopy(output_base),
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "structural_constraint_version": STRUCTURAL_CONSTRAINT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "core_sha256": CORE_SHA256,
        "model_name": MODEL_NAME,
        "high_difficulty_multiplier_enabled": (
            ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER
        ),
        "failed_stage": "stage2" if stage1 is not None else "stage1",
        "rating_error": str(error),
    }
    if isinstance(error, Stage2CallError):
        record["stage2_http_status"] = error.http_status
        record["stage2_output_text_length"] = error.output_text_length
        record["stage2_output_text_tail"] = error.output_text_tail
        record["stage2_parsed_keys"] = error.parsed_keys
        record["stage2_usage"] = error.usage
        record["stage2_incomplete_details"] = error.incomplete_details
        record["stage2_validation_error"] = error.validation_error

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


def build_stage2_fallback_result(
    *,
    output_base: dict[str, Any],
    stage1: dict[str, Any],
    stage2_error: Exception,
    stage1_usage: dict[str, int] | None,
    stage1_elapsed: float | None,
) -> dict[str, Any]:
    """第二阶段失败时保留第一阶段档位，并显式转人工复核，携带完整 debug 诊断信息与版本元数据。"""
    level = stage1["difficulty_level_step1"]
    usage = stage1_usage or {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    reviewed_high_count = int(
        stage1.get("high_difficulty_feature_count") or 0
    )
    usage2 = (
        stage2_error.usage
        if isinstance(stage2_error, Stage2CallError) and stage2_error.usage
        else {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
    )
    res: dict[str, Any] = {
        **copy.deepcopy(output_base),
        "pipeline_version": PIPELINE_VERSION,
        "prompt_version": PROMPT_VERSION,
        "structural_constraint_version": STRUCTURAL_CONSTRAINT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "core_sha256": CORE_SHA256,
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "high_difficulty_multiplier_enabled": (
            ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER
        ),
        "stage2_auto_adjustment_enabled": ENABLE_STAGE2_AUTO_ADJUST,
        "difficulty_rating_stage1": copy.deepcopy(stage1),
        "difficulty_level_step1": level,
        "verification": None,
        "verification_status": "failed_fallback_to_stage1",
        "stage2_error": str(stage2_error),
        "reviewed_high_difficulty_feature_count": reviewed_high_count,
        "model_suggested_level": level,
        "final_difficulty_level": level,
        "final_adjustment": (
            f"第二阶段失败·回退第一阶段{level}·转人工复核"
        ),
        "needs_manual_review": True,
        "api_stage1_time_seconds": (
            round(stage1_elapsed, 2)
            if stage1_elapsed is not None
            else None
        ),
        "api_stage2_time_seconds": None,
        "api_stage1_usage": copy.deepcopy(usage),
        "api_stage2_usage": copy.deepcopy(usage2),
        "api_total_usage": {
            k: int(usage.get(k, 0)) + int(usage2.get(k, 0))
            for k in ("input_tokens", "output_tokens", "total_tokens")
        },
    }
    if isinstance(stage2_error, Stage2CallError):
        res["stage2_http_status"] = stage2_error.http_status
        res["stage2_output_text_length"] = stage2_error.output_text_length
        res["stage2_output_text_tail"] = stage2_error.output_text_tail
        res["stage2_parsed_keys"] = stage2_error.parsed_keys
        res["stage2_usage"] = stage2_error.usage
        res["stage2_incomplete_details"] = stage2_error.incomplete_details
        res["stage2_validation_error"] = stage2_error.validation_error
    return res


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
    repair_feedback: str | None = None
    validation_retry_reasons: list[str] = []
    for attempt in range(retries):
        use_cache = bool(cache_id and repair_feedback is None)
        if repair_feedback is not None:
            prompt_text = (
                FEATURE_EXTRACTION_PROMPT_PREFIX
                + "\n\n"
                + dynamic_text
                + "\n\n【格式修复要求】\n"
                + repair_feedback
                + "\n请重新输出完整合法 JSON。不得省略任何必需 "
                "features 或 predicted_accuracy。"
            )
        else:
            prompt_text = (
                dynamic_text
                if use_cache
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
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "high_chemistry_stage1_rating",
                    "strict": True,
                    "schema": build_stage1_output_schema(),
                }
            },
        }
        if use_cache:
            payload["previous_response_id"] = cache_id
        if TEMPERATURE is not None:
            payload["temperature"] = TEMPERATURE
        try:
            status, body, error_text = await _post_response(session, payload, timeout)
            if status == 200 and body:
                current_usage = _usage(body)
                for key in total_usage:
                    total_usage[key] += current_usage[key]
                parsed = _restrict_stage1_model_output(
                    _parse_json_object(_extract_output_text(body))
                )
                try:
                    raw_features = copy.deepcopy(parsed.get("features"))
                    normalized, normalization_log = normalize_stage1_rating(
                        parsed
                    )
                    enriched = enrich_stage1_rating(
                        normalized,
                        features_model_raw=raw_features,
                        normalization_log=normalization_log,
                        validation_retry_count=len(validation_retry_reasons),
                        validation_retry_reasons=validation_retry_reasons,
                    )
                except ValueError as exc:
                    validation_retry_reasons.append(str(exc))
                    if attempt < retries - 1:
                        repair_feedback = (
                            f"上一次 JSON 校验失败：{exc}\n"
                            "上一次输出如下：\n"
                            + _json_block(parsed)
                        )
                        last_error = str(exc)
                        continue
                    raise
                return enriched, total_usage, time.time() - started
            last_error = f"HTTP {status}: {error_text[:400]}"
            if use_cache and "PreviousResponseNotFound" in error_text:
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
    last_call_error: Stage2CallError | None = None
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
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "high_chemistry_stage2_verification",
                    "strict": True,
                    "schema": build_stage2_output_schema(),
                }
            },
        }
        if TEMPERATURE is not None:
            payload["temperature"] = TEMPERATURE
        try:
            status, body, error_text = await _post_response(session, payload, timeout)
            if status == 200 and body:
                current_usage = _usage(body)
                for key in total_usage:
                    total_usage[key] += current_usage[key]
                incomplete_details = copy.deepcopy(body.get("incomplete_details"))
                response_status = body.get("status")
                if response_status == "incomplete":
                    last_call_error = Stage2CallError(
                        f"第二阶段响应 incomplete: {incomplete_details}",
                        http_status=200,
                        usage=copy.deepcopy(total_usage),
                        incomplete_details=incomplete_details,
                        validation_error="response_status_incomplete",
                    )
                    last_error = str(last_call_error)
                    if attempt < retries - 1:
                        await asyncio.sleep(2**attempt + random.random())
                        continue
                    raise last_call_error

                output_text = _extract_output_text(body)
                output_len = len(output_text)
                output_tail = output_text[-300:] if output_text else ""
                try:
                    parsed = _parse_json_object(output_text)
                except Exception as parse_exc:
                    last_call_error = Stage2CallError(
                        f"第二阶段响应 JSON 解析失败：{parse_exc}",
                        http_status=200,
                        output_text_length=output_len,
                        output_text_tail=output_tail,
                        usage=copy.deepcopy(total_usage),
                        incomplete_details=incomplete_details,
                        validation_error=str(parse_exc),
                    )
                    last_error = str(last_call_error)
                    if attempt < retries - 1:
                        await asyncio.sleep(2**attempt + random.random())
                        continue
                    raise last_call_error

                parsed_keys = list(parsed.keys()) if isinstance(parsed, dict) else []
                try:
                    validated = validate_verification(parsed)
                except Exception as val_exc:
                    last_call_error = Stage2CallError(
                        f"第二阶段校验失败：{val_exc}",
                        http_status=200,
                        output_text_length=output_len,
                        output_text_tail=output_tail,
                        parsed_keys=parsed_keys,
                        usage=copy.deepcopy(total_usage),
                        incomplete_details=incomplete_details,
                        validation_error=str(val_exc),
                    )
                    last_error = str(last_call_error)
                    if attempt < retries - 1:
                        await asyncio.sleep(2**attempt + random.random())
                        continue
                    raise last_call_error

                return (
                    recalculate_verification(
                        current_level=stage1["difficulty_level_step1"],
                        original_high_count=stage1[
                            "high_difficulty_feature_count"
                        ],
                        original_high_features=stage1[
                            "high_difficulty_features"
                        ],
                        original_accuracy=stage1[
                            "original_predicted_accuracy"
                        ],
                        original_features=stage1["features"],
                        allow_auto_adjustment=ENABLE_STAGE2_AUTO_ADJUST,
                        verification=validated,
                    ),
                    total_usage,
                    time.time() - started,
                )
            last_call_error = Stage2CallError(
                f"HTTP {status}: {error_text[:400]}",
                http_status=status,
                validation_error=error_text[:400],
            )
            last_error = str(last_call_error)
            if status != 429 and status < 500:
                break
        except (aiohttp.ClientError, asyncio.TimeoutError, ValueError) as exc:
            last_call_error = Stage2CallError(
                str(exc),
                validation_error=str(exc),
            )
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2**attempt + random.random())
    if last_call_error is not None:
        raise last_call_error
    raise Stage2CallError(f"第二阶段请求失败：{last_error}")


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
            try:
                verification, usage2, elapsed2 = await call_stage2(
                    session=session,
                    question_text=question_text,
                    image_urls=prepared.selected_image_urls,
                    stage1=stage1,
                    retries=retries,
                    timeout=timeout,
                )
            except Exception as stage2_exc:
                fallback = build_stage2_fallback_result(
                    output_base=output_base,
                    stage1=stage1,
                    stage2_error=stage2_exc,
                    stage1_usage=usage1,
                    stage1_elapsed=elapsed1,
                )
                await append_jsonl(output_path, fallback)
                await append_jsonl(
                    error_path,
                    build_pipeline_error(
                        output_base=output_base,
                        error=stage2_exc,
                        stage1=stage1,
                        stage1_usage=usage1,
                        stage1_elapsed=elapsed1,
                    ),
                )
                return
            reviewed_high_count = len(
                verification["reviewed_high_difficulty_features"]
            )
            final = finalize_verified_level(
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
            unresolved_stage1_severe = (
                stage1.get("structural_severe_disagreement") is True
                and not (
                    verification.get("has_structural_revision") is True
                    and verification.get("reviewed_structural_severe_disagreement") is False
                )
            )
            needs_manual_review = (
                unresolved_stage1_severe
                or final.needs_manual_review
                or verification.get("review_requires_manual") is True
            )
            result = {
                **output_base,
                "pipeline_version": PIPELINE_VERSION,
                "prompt_version": PROMPT_VERSION,
                "structural_constraint_version": STRUCTURAL_CONSTRAINT_VERSION,
                "prompt_sha256": PROMPT_SHA256,
                "core_sha256": CORE_SHA256,
                "model_name": MODEL_NAME,
                "temperature": TEMPERATURE,
                "high_difficulty_multiplier_enabled": ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER,
                "stage2_auto_adjustment_enabled": (
                    ENABLE_STAGE2_AUTO_ADJUST
                ),
                "difficulty_rating_stage1": stage1,
                "difficulty_level_step1": stage1["difficulty_level_step1"],
                "verification": verification,
                "reviewed_high_difficulty_feature_count": reviewed_high_count,
                "model_suggested_level": final.model_suggested_level,
                "final_difficulty_level": final.final_level,
                "final_adjustment": (
                    f"二阶段建议改档但未满足结构证据守卫·维持"
                    f"{final.final_level}·转人工复核"
                    if verification.get("review_requires_manual") is True
                    else (
                        f"第一阶段存在两档严重结构分歧·维持"
                        f"{final.final_level}·转人工复核"
                        if unresolved_stage1_severe
                        else final.adjustment_desc
                    )
                ),
                "needs_manual_review": needs_manual_review,
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
    parser = argparse.ArgumentParser(description=f"{SUBJECT_DISPLAY_NAME}两阶段难度评级")
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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="兼容断点续跑参数（本脚本默认已自动断点续跑）",
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
        progress = tqdm(total=len(pending), desc=PROGRESS_DESCRIPTION, unit="item")
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
