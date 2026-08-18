# -*- coding: utf-8 -*-
"""初中化学难度批量评级（FXZ 正式精简主线）。

只保留当前 V5 十七项可观测特征生产路径、Schema 修复、现行有效教师
边界写回规则，以及 API/缓存/并发/断点续跑。已删除旧 Core-12 通用
自动改档分支与已停用的 audit-only 候选规则及其专属信号函数。
现行有效规则的判断逻辑和先后顺序不做重写。

现行有效规则直接读取 V5 特征与 V5 派生指标，不保留 Core-12 兼容投影。
"""

import os
import sys
import json
import re
import random
import time
import hashlib
import asyncio
import copy
import aiofiles
import aiohttp
import argparse
from pathlib import Path
try:
    import json_repair
except Exception:
    class _JsonRepairFallback:
        @staticmethod
        def loads(text):
            return json.loads(text)
    json_repair = _JsonRepairFallback()
from typing import Dict, Any, Optional, List, Tuple, Sequence
from tqdm.asyncio import tqdm
from asyncio import Lock, Semaphore
from dotenv import load_dotenv

try:
    from chemistry_observable_features_fxz import (
        OBSERVABLE_ENUM_VALUES_BY_FIELD,
        OBSERVABLE_FALLBACK_LABELS,
        normalize_observable_features,
        observable_feature_quality_flags,
        validate_observable_features,
    )
except ModuleNotFoundError:
    from src.chemistry_observable_features_fxz import (
        OBSERVABLE_ENUM_VALUES_BY_FIELD,
        OBSERVABLE_FALLBACK_LABELS,
        normalize_observable_features,
        observable_feature_quality_flags,
        validate_observable_features,
    )

try:
    from chemistry_postprocess_fxz import (
        ChemistrySchemaError,
        VISUAL_REFERENCE_RE,
        derive_question_structure_metrics,
        postprocess_chemistry_difficulty,
        visible_text,
    )
except ModuleNotFoundError:
    from src.chemistry_postprocess_fxz import (
        ChemistrySchemaError,
        VISUAL_REFERENCE_RE,
        derive_question_structure_metrics,
        postprocess_chemistry_difficulty,
        visible_text,
    )

# -------------------------- 0. API 基础配置 --------------------------
load_dotenv()

API_KEY = os.getenv("API_KEY", "not-needed")
BASE_URL = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1")
if not BASE_URL.endswith("/"):
    BASE_URL += "/"
MODEL_NAME = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")


def resolve_temperature(model_name: str, raw_value: str) -> Optional[float]:
    """与物理正式脚本一致：Lite 服务端固定 temperature=1。"""
    if "lite" in str(model_name).lower():
        return 1.0
    value = str(raw_value or "").strip()
    return float(value) if value else None


TEMPERATURE = resolve_temperature(
    MODEL_NAME,
    os.getenv("TEMPERATURE", ""),
)
RATING_PROFILE = os.getenv(
    "RATING_PROFILE",
    "chemistry_stable",
).strip().lower()
VALID_RATING_PROFILES = {"chemistry_stable"}
if RATING_PROFILE not in VALID_RATING_PROFILES:
    raise ValueError(
        f"不支持的 RATING_PROFILE={RATING_PROFILE!r}；"
        f"可选值：{', '.join(sorted(VALID_RATING_PROFILES))}"
    )

USE_CACHE = os.getenv("USE_CACHE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}
CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = os.getenv(
    "CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS",
    "1",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = os.getenv(
    "CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK",
    "1",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

CHEMISTRY_IMAGE_MODE = os.getenv(
    "CHEMISTRY_IMAGE_MODE",
    "auto",
).strip().lower()
if CHEMISTRY_IMAGE_MODE not in {"off", "auto", "all"}:
    raise ValueError(
        f"不支持的 CHEMISTRY_IMAGE_MODE={CHEMISTRY_IMAGE_MODE!r}；"
        "可选值：off, auto, all"
    )
CHEMISTRY_IMAGE_DETAIL = os.getenv(
    "CHEMISTRY_IMAGE_DETAIL",
    "default",
).strip().lower()
if CHEMISTRY_IMAGE_DETAIL not in {
    "default",
    "adaptive",
    "auto",
    "low",
    "high",
    "xhigh",
}:
    raise ValueError(
        f"不支持的 CHEMISTRY_IMAGE_DETAIL={CHEMISTRY_IMAGE_DETAIL!r}；"
        "可选值：default, adaptive, auto, low, high, xhigh（实验值）"
    )
MAX_SCHEMA_RETRIES = int(os.getenv("CHEMISTRY_SCHEMA_RETRIES", "3"))

UNTRUSTED_LABEL_FIELDS = {
    "difficulty",
    "teacher_label",
    "teacher_difficulty",
    "label",
    "难度",
}
ADAPTIVE_XHIGH_IMAGE_RE = re.compile(
    r"(实验装置|装置图|装置|仪器|量筒|天平|滴定管|长颈漏斗|"
    r"分液漏斗|试管|烧杯|集气瓶|导管|发生装置|收集装置)"
)
ADAPTIVE_HIGH_IMAGE_RE = re.compile(
    r"(曲线|坐标图|关系图|拐点|分段图|流程图|流程|工艺|"
    r"表格|数据表|折线图|柱状图)"
)
VISUAL_PLACEHOLDER_RE = re.compile(
    r"(<img\b|<image\b|\[image\]|\[图片\]|\【图片\】|图片缺失|见图|如下图)",
    re.IGNORECASE,
)

FILE_LOCK = Lock()
CACHE_LOCK = Lock()
CACHE_GET_LOCK = Lock()

CACHE_EXPIRE_DAYS = 6
CACHE_EXPIRE_SECONDS = CACHE_EXPIRE_DAYS * 24 * 3600
CACHE_FILE_PATH = "chemistry_prompt_cache.json"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT_PATH = (
    PROJECT_ROOT / "prompts" / "初中化学难度打标提示词.txt"
)
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "chemistry_sampled_5000_per_difficulty_v2.jsonl"
)

DIFFICULTY_RATING_PROMPT_PREFIX = ""
DIFFICULTY_RATING_PROMPT_SUFFIX = ""
CURRENT_RUN_SIGNATURE = ""
CURRENT_RUN_CONFIG: Dict[str, Any] = {}

# -------------------------- 1. 提示词加载 --------------------------
def load_prompt_config(prompt_path: str) -> None:
    """动态解析提示词文件，支持 Python 变量格式与纯文本格式。"""
    global DIFFICULTY_RATING_PROMPT_PREFIX, DIFFICULTY_RATING_PROMPT_SUFFIX

    if not os.path.exists(prompt_path):
        print(f"错误: 找不到提示词文件 {prompt_path}！")
        sys.exit(1)

    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 优先兼容物理/数学已有 Python 变量结构
    try:
        namespace: Dict[str, Any] = {}
        exec(content, namespace)
        prefix = namespace.get("DIFFICULTY_RATING_PROMPT_PREFIX")
        suffix = namespace.get("DIFFICULTY_RATING_PROMPT_SUFFIX")
        if prefix and suffix:
            DIFFICULTY_RATING_PROMPT_PREFIX = str(prefix)
            DIFFICULTY_RATING_PROMPT_SUFFIX = str(suffix)
            print("成功以 Python 变量结构解析提示词")
            return
    except Exception:
        pass

    # 兼容纯文本提示词
    if "## 输入题目信息" in content:
        parts = content.split("## 输入题目信息")
        DIFFICULTY_RATING_PROMPT_PREFIX = parts[0] + "## 输入题目信息"
        DIFFICULTY_RATING_PROMPT_SUFFIX = "\n\n请根据以上信息，对题目进行全面的难度分析和评级。"
        print("成功以纯文本标志位结构切分并解析提示词")
        return

    raise ValueError("提示词格式不正确：既不是有效 Python 变量结构，也没有包含 '## 输入题目信息' 分割标志。")

# -------------------------- 2. 前缀缓存模块 --------------------------
def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_config(
    input_path: str | Path,
    prompt_path: str | Path,
    *,
    seed: int,
    num: Optional[int],
) -> Dict[str, Any]:
    return {
        "signature_version": "chemistry-rating-fxz-v5-production-v1",
        "input_sha256": compute_file_hash(input_path),
        "prompt_sha256": compute_file_hash(prompt_path),
        "script_sha256": compute_file_hash(Path(__file__).resolve()),
        "rating_profile": RATING_PROFILE,
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "image_mode": CHEMISTRY_IMAGE_MODE,
        "image_detail": CHEMISTRY_IMAGE_DETAIL,
        "cache_enabled": USE_CACHE,
        "seed": seed,
        "num": num,
        "teacher_distribution_guards_enabled": (
            CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS
        ),
        "teacher_distribution_guards_writeback_enabled": (
            CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
        ),
    }


def build_run_signature(config: Dict[str, Any]) -> str:
    canonical = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def ensure_output_run_signature(
    output_path: str | Path,
    expected_signature: str,
) -> None:
    path = Path(output_path)
    if not path.exists() or path.stat().st_size == 0:
        return
    signatures: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"断点文件第{line_number}行不是合法JSON，拒绝续跑"
                ) from exc
            signature = str(item.get("run_signature", "")).strip()
            if not signature:
                raise ValueError(
                    f"断点文件第{line_number}行缺少run_signature，"
                    "拒绝与旧版或未知配置结果混写；请更换输出文件"
                )
            signatures.add(signature)
    if signatures != {expected_signature}:
        raise ValueError(
            "断点文件运行签名不一致，拒绝混写；"
            f"期望={expected_signature}，已有={sorted(signatures)}"
        )


async def load_cache() -> Dict[str, Any]:
    async with CACHE_LOCK:
        if not os.path.exists(CACHE_FILE_PATH):
            return {}
        try:
            async with aiofiles.open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                content = await f.read()
                return json.loads(content) if content else {}
        except Exception as e:
            print(f"加载缓存文件失败: {e}")
            return {}

async def save_cache(cache_data: Dict[str, Any]) -> None:
    async with CACHE_LOCK:
        try:
            async with aiofiles.open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
                await f.write(json.dumps(cache_data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"保存缓存文件失败: {e}")


def is_cache_valid(cache_entry: Dict[str, Any], current_time: int) -> bool:
    if not cache_entry:
        return False
    if current_time >= int(cache_entry.get("expire_at", 0)):
        return False
    return (
        cache_entry.get("prefix_hash", "") == compute_text_hash(DIFFICULTY_RATING_PROMPT_PREFIX)
        and cache_entry.get("model_name") == MODEL_NAME
    )

async def get_valid_cache() -> Optional[Dict[str, Any]]:
    cache_data = await load_cache()
    cache_entry = cache_data.get("prompt_prefix_cache")
    if is_cache_valid(cache_entry, int(time.time())):
        return cache_entry
    return None

async def set_cache(response_id: str, expire_at: int) -> None:
    cache_data = await load_cache()
    cache_data["prompt_prefix_cache"] = {
        "response_id": response_id,
        "expire_at": expire_at,
        "prefix_hash": compute_text_hash(DIFFICULTY_RATING_PROMPT_PREFIX),
        "model_name": MODEL_NAME,
        "created_at": int(time.time()),
    }
    await save_cache(cache_data)

async def create_prefix_cache(session: aiohttp.ClientSession, retries: int, timeout_sec: int) -> Optional[str]:
    current_time = int(time.time())
    expire_at = current_time + CACHE_EXPIRE_SECONDS

    payload = {
        "model": MODEL_NAME,
        "input": [{"role": "user", "content": DIFFICULTY_RATING_PROMPT_PREFIX}],
        "thinking": {"type": "disabled"},
        "expire_at": expire_at,
        "caching": {"type": "enabled", "prefix": True},
    }

    t1 = time.time()
    for attempt in range(retries):
        try:
            async with session.post(
                f"{BASE_URL}responses",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"创建前缀缓存失败 (状态码: {response.status}): {error_text[:200]}")
                    if 400 <= response.status < 500:
                        return None
                    await asyncio.sleep(2 ** attempt)
                    continue

                result = await response.json()
                response_id = result.get("id")
                if response_id:
                    await set_cache(response_id, expire_at)
                    print(f"前缀缓存创建成功，耗时: {time.time() - t1:.2f}秒，缓存ID: {response_id}")
                    return response_id
        except Exception as e:
            backoff = (2 ** attempt) + random.uniform(0, 1)
            if attempt == retries - 1:
                print(f"创建前缀缓存最终失败: {e}")
                return None
            print(f"创建前缀缓存异常，{backoff:.2f}秒后重试: {e}")
            await asyncio.sleep(backoff)
    return None

async def get_or_create_cache(session: aiohttp.ClientSession, retries: int, timeout_sec: int) -> Optional[str]:
    async with CACHE_GET_LOCK:
        cache_entry = await get_valid_cache()
        if cache_entry:
            return cache_entry["response_id"]
        print("未找到有效缓存，正在向服务器创建前缀缓存...")
        return await create_prefix_cache(session, retries, timeout_sec)

# -------------------------- 3. 化学特征 schema 与归一化 --------------------------
def build_schema_repair_feedback(
    error: Exception,
    invalid_candidate: Dict[str, Any],
) -> str:
    """给Schema重试提供定点修复信息和上一版JSON。"""
    error_text = str(error)
    hints = [f"上次输出未通过化学特征schema：{error_text}"]
    for field, allowed in OBSERVABLE_ENUM_VALUES_BY_FIELD.items():
        if field not in error_text:
            continue
        allowed_values = sorted(
            value
            for value in allowed
            if value not in OBSERVABLE_FALLBACK_LABELS
            and value != "U_OTHER"
        )
        hints.append(
            f"{field}只能从以下中文枚举中逐字复制："
            + "、".join(allowed_values)
            + "。"
        )
    if "experiment_operation" in error_text:
        hints.append(
            "experiment_operation描述实际实验认知操作；"
            "不要填写实验任务组织结构。"
        )
    if "experiment_task_structure" in error_text:
        hints.append(
            "experiment_task_structure描述实验任务怎样组织；"
            "不要填写基础操作、变量控制等operation枚举。"
        )
    serialized = json.dumps(invalid_candidate, ensure_ascii=False, indent=2)
    if len(serialized) > 16000:
        serialized = serialized[:16000] + "\n...（已截断）"
    hints.extend(
        [
            "请以上次JSON为底稿，只修复字段、枚举和一致性错误；"
            "不要为了通过校验而抬高或压低等级。",
            serialized,
        ]
    )
    return "\n".join(hints)


def is_feature_schema_error(error: Exception) -> bool:
    """判断错误是否可通过只重生成features修复。"""
    text = str(error)
    top_level_markers = (
        "模型输出必须是JSON对象",
        "模型返回空对象",
        "顶层字段",
        "difficulty_level",
        "coarse_difficulty",
        "reasoning必须",
        "reasoning四个字段",
    )
    return not any(marker in text for marker in top_level_markers)


def classify_feature_schema_repair(error: Exception) -> str:
    """需要模型重生成的features统一视为语义修复并禁止自动写回。"""
    _ = error
    return "semantic"


def build_feature_schema_repair_feedback(
    error: Exception,
    invalid_candidate: Dict[str, Any],
) -> str:
    """生成只修复features的请求，冻结等级和理由。"""
    features = invalid_candidate.get("features", invalid_candidate)
    feedback = build_schema_repair_feedback(error, {"features": features})
    return (
        feedback
        + '\n本次只输出一个JSON对象：{"features": {...}}。'
        "不得输出或改写difficulty_level、coarse_difficulty、reasoning。"
    )


def merge_feature_repair_candidate(
    original_candidate: Dict[str, Any],
    repair_candidate: Dict[str, Any],
    repair_kind: str = "format",
) -> Dict[str, Any]:
    """只合并修复后的features，保持首轮难度结论和理由不变。"""
    if not isinstance(original_candidate, dict):
        return copy.deepcopy(original_candidate)
    repaired_features = (
        repair_candidate.get("features")
        if isinstance(repair_candidate, dict)
        else None
    )
    if not isinstance(repaired_features, dict):
        return copy.deepcopy(original_candidate)
    merged = copy.deepcopy(original_candidate)
    merged["features"] = copy.deepcopy(repaired_features)
    merged["feature_schema_repair_kind"] = repair_kind
    return merged


def build_schema_retry_audit(
    *,
    first_candidate: Dict[str, Any],
    accepted_candidate: Dict[str, Any],
    schema_candidates: List[Dict[str, Any]],
    schema_retry_count: int,
    repair_kinds: Sequence[str],
) -> Dict[str, Any]:
    """构建首轮与接纳候选的可比较审计字段。"""
    first = copy.deepcopy(first_candidate or accepted_candidate or {})
    accepted = copy.deepcopy(accepted_candidate or first_candidate or {})
    if not schema_retry_count:
        repair_mode = "none"
        repair_kind = "none"
    else:
        repair_mode = (
            "features_only"
            if any(
                item.get("repair_mode") == "features_only"
                for item in schema_candidates
            )
            else "full_rating"
        )
        unique_kinds = list(dict.fromkeys(repair_kinds))
        repair_kind = (
            "semantic"
            if "semantic" in unique_kinds
            else (unique_kinds[-1] if unique_kinds else "format")
        )
    return {
        "difficulty_rating_first_attempt": first,
        "first_attempt_level": str(first.get("difficulty_level", "") or ""),
        "accepted_attempt_level": str(
            accepted.get("difficulty_level", "") or ""
        ),
        "schema_retry_changed_level": (
            first.get("difficulty_level") != accepted.get("difficulty_level")
        ),
        "schema_retry_changed_features": (
            first.get("features") != accepted.get("features")
        ),
        "schema_candidates": copy.deepcopy(schema_candidates),
        "schema_repair_mode": repair_mode,
        "schema_repair_kind": repair_kind,
    }


def sanitize_question_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """递归隔离来源难度标签，模型输入和后处理均不得读取。"""
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if key not in UNTRUSTED_LABEL_FIELDS
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        return copy.deepcopy(value)

    return clean(data)


def make_output_base(data: Dict[str, Any]) -> Dict[str, Any]:
    """保留来源审计，但不继续用可信字段名保存原difficulty。"""
    def rename(value: Any) -> Any:
        if isinstance(value, dict):
            output: Dict[str, Any] = {}
            for key, item in value.items():
                target = (
                    f"source_{key}_untrusted"
                    if key in UNTRUSTED_LABEL_FIELDS
                    else key
                )
                output[target] = rename(item)
            return output
        if isinstance(value, list):
            return [rename(item) for item in value]
        return copy.deepcopy(value)

    return rename(data)


def construct_question_content(data: Dict[str, Any]) -> str:
    """将数据记录拼装成标准的打标输入文本；对齐物理脚本，兼容 sub_questions。"""
    parts: List[str] = []
    stem = str(data.get("stem", "") or "").strip()
    options = str(data.get("options", "") or "").strip()
    analysis = str(data.get("analysis", "") or "").strip()

    if stem:
        parts.append(f"【题干】\n{stem}")
    if options:
        parts.append(f"【选项】\n{options}")
    if analysis:
        parts.append(f"【解析】\n{analysis}")

    sub_questions = list(data.get("sub_questions", []) or [])
    if sub_questions:
        try:
            sub_questions = sorted(
                sub_questions,
                key=lambda x: int(x.get("question_id", 0))
                if isinstance(x, dict)
                else 0,
            )
        except Exception:
            pass
        parts.append("【小题】")
        for i, sq in enumerate(sub_questions, 1):
            parts.append(f"  小题{i}:")
            if isinstance(sq, dict):
                sq_stem = str(sq.get("stem", "") or "").strip()
                sq_options = str(sq.get("options", "") or "").strip()
                sq_analysis = str(sq.get("analysis", "") or "").strip()
                if sq_stem:
                    parts.append(f"    题干: {sq_stem}")
                if sq_options:
                    parts.append(f"    选项: {sq_options}")
                if sq_analysis:
                    parts.append(f"    解析: {sq_analysis}")
            else:
                parts.append(f"    题干: {sq}")

    return "\n\n".join(parts)


def _collect_analysis_text(data: Dict[str, Any]) -> str:
    parts = [str(data.get("analysis", "") or "").strip()]
    for item in data.get("sub_questions", []) or []:
        if isinstance(item, dict):
            parts.append(str(item.get("analysis", "") or "").strip())
    return "\n".join(part for part in parts if part)


def extract_image_urls(value: Any) -> List[str]:
    """兼容单URL、逗号拼接URL和URL列表，保持原顺序并去重。"""
    if isinstance(value, (list, tuple)):
        candidates = [str(item or "").strip() for item in value]
    else:
        candidates = [
            part.strip()
            for part in re.split(r"[,，\s]+", str(value or "").strip())
        ]
    urls: List[str] = []
    for candidate in candidates:
        if (
            candidate.startswith(("http://", "https://"))
            and candidate not in urls
        ):
            urls.append(candidate)
    return urls


def select_input_image_urls(
    data: Dict[str, Any],
    field: str,
) -> List[str]:
    """题干仅发送最后一张图，解析保留全部图片。"""
    urls = extract_image_urls(data.get(field))
    if field == "stem_pic_url":
        return urls[-1:]
    return urls


def select_image_fields(data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """按物理项目经验，仅在题面真实依赖视觉关系时发送图片。"""
    available = {
        field: extract_image_urls(data.get(field))
        for field in ("stem_pic_url", "analysis_pic_url")
        if extract_image_urls(data.get(field))
    }
    if CHEMISTRY_IMAGE_MODE == "off" or not available:
        return [], [
            "图片模式关闭"
            if CHEMISTRY_IMAGE_MODE == "off"
            else "无可用图片URL"
        ]
    if CHEMISTRY_IMAGE_MODE == "all":
        return list(available), ["all模式：发送全部可用图片字段"]

    stem_text = visible_text(data, include_analysis=False)
    analysis_text = _collect_analysis_text(data)
    selected: List[str] = []
    reasons: List[str] = []
    if "stem_pic_url" in available and (
        not str(data.get("stem", "") or "").strip()
        or VISUAL_REFERENCE_RE.search(stem_text)
        or VISUAL_PLACEHOLDER_RE.search(stem_text)
    ):
        selected.append("stem_pic_url")
        reasons.append("题干明确依赖图表/装置/流程/微观或空间关系")

    if "analysis_pic_url" in available and (
        not analysis_text
        or VISUAL_PLACEHOLDER_RE.search(analysis_text)
        or (
            VISUAL_REFERENCE_RE.search(analysis_text)
            and "stem_pic_url" not in available
        )
    ):
        selected.append("analysis_pic_url")
        reasons.append("结构化解析缺失或解析中的关键视觉关系无法由题干图补足")

    if not selected:
        reasons.append("结构化文字足以支持定档，不发送图片")
    return selected, reasons


def resolve_image_detail(
    data: Dict[str, Any],
    selected_image_fields: Optional[Sequence[str]] = None,
) -> str:
    """将全局图片策略解析为单题实际 detail。

    adaptive 只使用题面可见文字路由，不读解析，避免答案或解析
    表述改变输入画质。实验装置和仪器细节使用 xhigh；曲线、流程、
    数据表使用 high；其余保持 API 默认画质。
    """
    selected = list(selected_image_fields or [])
    if not selected:
        return "default"
    if CHEMISTRY_IMAGE_DETAIL != "adaptive":
        return CHEMISTRY_IMAGE_DETAIL

    question_text = visible_text(data, include_analysis=False)
    if ADAPTIVE_XHIGH_IMAGE_RE.search(question_text):
        return "xhigh"
    if ADAPTIVE_HIGH_IMAGE_RE.search(question_text):
        return "high"
    return "default"


def build_user_content(
    data: Dict[str, Any],
    selected_image_fields: Optional[Sequence[str]] = None,
) -> Any:
    """完整文字始终先发，图片只作为必要视觉补充。"""
    selected = list(
        selected_image_fields
        if selected_image_fields is not None
        else select_image_fields(data)[0]
    )
    dynamic_text = (
        f"{construct_question_content(data)}"
        f"{DIFFICULTY_RATING_PROMPT_SUFFIX}\n\n"
        "完整结构化文字是主要题面来源；不要从URL、题号或来源标签推断难度。"
    )
    if not selected:
        return dynamic_text

    resolved_image_detail = resolve_image_detail(data, selected)

    content: List[Dict[str, str]] = [
        {"type": "input_text", "text": dynamic_text}
    ]
    labels = {
        "stem_pic_url": (
            "下面是按需发送的题干图片。只补充文字无法表达的装置连接、"
            "流程箭头、曲线阶段、表格对应、粒子分布或空间关系。"
        ),
        "analysis_pic_url": (
            "下面是按需发送的解析图片。只核对必要解题链和视觉关系，"
            "不得因为看到答案而降低难度。"
        ),
    }
    seen = set()
    for field in selected:
        urls = select_input_image_urls(data, field)
        if urls:
            content.append({"type": "input_text", "text": labels[field]})
        for url in urls:
            if url in seen:
                continue
            image_part = {"type": "input_image", "image_url": url}
            if resolved_image_detail != "default":
                image_part["detail"] = resolved_image_detail
            content.append(image_part)
            seen.add(url)
    return content


def parse_model_response(response_text: str) -> Dict[str, Any]:
    """容错并修复 JSON 输出。"""
    if not response_text:
        return {}
    try:
        parsed = json_repair.loads(response_text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    try:
        clean_text = response_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```", 1)[1].split("```", 1)[0]
        parsed = json_repair.loads(clean_text.strip())
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    try:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json_repair.loads(response_text[start:end + 1])
            return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    return {}


async def call_model_with_cache(
    data: Dict[str, Any],
    session: aiohttp.ClientSession,
    retries: int,
    timeout_sec: int,
    repair_feedback: str = "",
    features_only_repair: bool = False,
) -> Tuple[
    Dict[str, Any],
    str,
    float,
    int,
    int,
    int,
    Dict[str, Any],
]:
    selected_fields, selection_reasons = select_image_fields(data)
    selected_urls = [
        url
        for field in selected_fields
        for url in select_input_image_urls(data, field)
    ]
    resolved_image_detail = resolve_image_detail(data, selected_fields)
    question_structure_metrics = derive_question_structure_metrics(data)
    image_status = {
        "question_input_mode": f"text_first_image_{CHEMISTRY_IMAGE_MODE}",
        "question_text_input_used": True,
        "structured_text_char_count": len(construct_question_content(data)),
        "image_mode": CHEMISTRY_IMAGE_MODE,
        "image_detail_strategy": CHEMISTRY_IMAGE_DETAIL,
        "image_detail": resolved_image_detail,
        "image_input_requested": bool(selected_fields),
        "image_input_used": False,
        "image_input_fields": selected_fields,
        "image_input_url_count": len(dict.fromkeys(selected_urls)),
        "image_selection_reasons": selection_reasons,
        "http_retry_count": 0,
        **question_structure_metrics,
    }

    response_id: Optional[str] = None
    if USE_CACHE:
        response_id = await get_or_create_cache(session, retries, timeout_sec)
        if not response_id:
            print("警告: 无法获取有效缓存 ID，终止单题请求")
            return {}, "", 0.0, 0, 0, 0, image_status

    for retry in range(retries):
        image_status["http_retry_count"] = retry
        user_content = build_user_content(data, selected_fields)
        if repair_feedback:
            repair_tail = (
                "本次只能输出{\"features\": {...}}；不要输出或改写"
                "difficulty_level、coarse_difficulty、reasoning。"
                if features_only_repair
                else "保持实质难度判断不变，只修复缺失字段、非法枚举或JSON格式。"
            )
            repair_part = {
                "type": "input_text",
                "text": (
                    "【上次输出修复要求】\n"
                    + repair_feedback
                    + "\n"
                    + repair_tail
                ),
            }
            if isinstance(user_content, list):
                user_content = [*user_content, repair_part]
            else:
                user_content = [
                    {"type": "input_text", "text": user_content},
                    repair_part,
                ]

        if USE_CACHE:
            request_content = user_content
        else:
            prefix_part = {
                "type": "input_text",
                "text": DIFFICULTY_RATING_PROMPT_PREFIX,
            }
            if isinstance(user_content, list):
                request_content = [prefix_part, *user_content]
            else:
                request_content = [
                    prefix_part,
                    {"type": "input_text", "text": user_content},
                ]
        payload = {
            "model": MODEL_NAME,
            "input": [{"role": "user", "content": request_content}],
            "thinking": {"type": "disabled"},
        }
        if USE_CACHE:
            payload["previous_response_id"] = response_id
        if TEMPERATURE is not None:
            payload["temperature"] = TEMPERATURE
        t1 = time.time()
        try:
            async with session.post(
                f"{BASE_URL}responses",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    output_text = ""
                    if "output" in result:
                        for item in result["output"]:
                            if item.get("type") == "message" and "content" in item:
                                for content_item in item["content"]:
                                    if content_item.get("type") == "output_text":
                                        output_text = content_item.get("text", "")
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    parsed_result = parse_model_response(output_text)
                    image_status["image_input_used"] = bool(selected_fields)
                    return (
                        parsed_result,
                        output_text,
                        time.time() - t1,
                        prompt_tokens,
                        completion_tokens,
                        total_tokens,
                        image_status,
                    )

                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    print(f"接口限流(429)，等待 {retry_after} 秒后进行第 {retry + 1} 次重试...")
                    await asyncio.sleep(retry_after)
                    continue

                error_text = await response.text()
                print(f"API请求失败 (状态码: {response.status}): {error_text[:200]}")
                if USE_CACHE and "InvalidParameter.PreviousResponseNotFound" in error_text:
                    print("检测到服务器缓存丢失，正在重建缓存...")
                    new_response_id = await create_prefix_cache(session, retries, timeout_sec)
                    if not new_response_id:
                        return {}, "", 0.0, 0, 0, 0, image_status
                    response_id = new_response_id
                    continue
                if response.status >= 500:
                    backoff = (2 ** retry) + random.uniform(0, 1)
                    print(f"服务器故障({response.status})，{backoff:.2f}秒后重试 (第{retry + 1}次)...")
                    await asyncio.sleep(backoff)
                    continue
                if 400 <= response.status < 500:
                    return {}, "", 0.0, 0, 0, 0, image_status
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            backoff = (2 ** retry) + random.uniform(0, 1)
            if retry == retries - 1:
                print(
                    "网络异常最终失败 "
                    f"[{type(e).__name__}]: {e!r}"
                )
                return {}, "", 0.0, 0, 0, 0, image_status
            print(
                f"网络出现异常 [{type(e).__name__}]: {e!r}，"
                f"将进行退避 {backoff:.2f} 秒后重试..."
            )
            await asyncio.sleep(backoff)
        except Exception as e:
            print(
                f"运行过程中请求异常 "
                f"[{type(e).__name__}]: {e!r}"
            )
            if retry == retries - 1:
                return {}, "", 0.0, 0, 0, 0, image_status
            if USE_CACHE:
                new_response_id = await create_prefix_cache(session, retries, timeout_sec)
                if new_response_id:
                    response_id = new_response_id
            await asyncio.sleep(1)

    return {}, "", 0.0, 0, 0, 0, image_status

# -------------------------- 6. 并发处理 --------------------------
async def process_single_question(
    data: Dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    output_path: str,
    error_path: str,
    retries: int,
    timeout_sec: int,
) -> None:
    async with semaphore:
        question_id = data.get("question_id", "unknown")
        safe_data = sanitize_question_data(data)
        total_time = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        schema_retry_count = 0
        schema_errors: List[str] = []
        schema_attempts: List[Dict[str, Any]] = []
        schema_candidates: List[Dict[str, Any]] = []
        schema_repair_kinds: List[str] = []
        repair_feedback = ""
        features_only_repair = False
        active_repair_kind = "none"
        first_model_candidate: Dict[str, Any] = {}
        feature_repair_base_candidate: Dict[str, Any] = {}
        accepted_candidate: Dict[str, Any] = {}
        raw_result: Dict[str, Any] = {}
        raw_text = ""
        image_status: Dict[str, Any] = {}

        while True:
            try:
                (
                    candidate,
                    raw_text,
                    time_use,
                    prompt_tokens,
                    completion_tokens,
                    call_tokens,
                    image_status,
                ) = await call_model_with_cache(
                    safe_data,
                    session,
                    retries,
                    timeout_sec,
                    repair_feedback=repair_feedback,
                    features_only_repair=features_only_repair,
                )
                response_candidate = copy.deepcopy(candidate)
                if not first_model_candidate and response_candidate:
                    first_model_candidate = copy.deepcopy(response_candidate)
                if features_only_repair and feature_repair_base_candidate:
                    candidate = merge_feature_repair_candidate(
                        feature_repair_base_candidate,
                        response_candidate,
                        repair_kind=active_repair_kind,
                    )
                elif (
                    isinstance(candidate, dict)
                    and "features" in candidate
                    and "difficulty_level" in candidate
                ):
                    feature_repair_base_candidate = copy.deepcopy(candidate)
                total_time += time_use
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += call_tokens

                try:
                    if not candidate:
                        raise ChemistrySchemaError("模型返回空对象或JSON解析失败")
                    rating_result = postprocess_chemistry_difficulty(
                        candidate,
                        safe_data,
                        teacher_distribution_guards_enabled=(
                            CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS
                        ),
                        teacher_distribution_guards_writeback_enabled=(
                            CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
                        ),
                    )
                except ChemistrySchemaError as exc:
                    schema_errors.append(str(exc))
                    failed_attempt = {
                        "attempt": schema_retry_count,
                        "error": str(exc),
                        "repair_mode": (
                            "features_only"
                            if features_only_repair
                            else "full_rating"
                        ),
                        "repair_kind": active_repair_kind,
                        "candidate": response_candidate,
                        "accepted": False,
                    }
                    schema_attempts.append(copy.deepcopy(failed_attempt))
                    schema_candidates.append(copy.deepcopy(failed_attempt))
                    if schema_retry_count >= MAX_SCHEMA_RETRIES:
                        raise RuntimeError(
                            f"schema校验重试耗尽({MAX_SCHEMA_RETRIES}): {exc}"
                        ) from exc
                    schema_retry_count += 1
                    feature_error = is_feature_schema_error(exc)
                    if feature_error:
                        repair_kind = classify_feature_schema_repair(exc)
                        schema_repair_kinds.append(repair_kind)
                        active_repair_kind = (
                            "semantic"
                            if "semantic" in schema_repair_kinds
                            else "format"
                        )
                        if isinstance(candidate, dict) and candidate.get(
                            "features"
                        ):
                            feature_repair_base_candidate = copy.deepcopy(
                                candidate
                            )
                    else:
                        active_repair_kind = "full_rating"
                    features_only_repair = bool(
                        feature_repair_base_candidate and feature_error
                    )
                    repair_feedback = (
                        build_feature_schema_repair_feedback(
                            exc,
                            candidate,
                        )
                        if features_only_repair
                        else build_schema_repair_feedback(
                            exc,
                            candidate,
                        )
                    )
                    continue

                accepted_candidate = copy.deepcopy(candidate)
                raw_result = copy.deepcopy(accepted_candidate)
                schema_candidates.append(
                    {
                        "attempt": schema_retry_count,
                        "error": "",
                        "repair_mode": (
                            "features_only"
                            if features_only_repair
                            else "full_rating"
                        ),
                        "repair_kind": active_repair_kind,
                        "candidate": copy.deepcopy(response_candidate),
                        "accepted": True,
                    }
                )
                schema_audit = build_schema_retry_audit(
                    first_candidate=(
                        first_model_candidate or accepted_candidate
                    ),
                    accepted_candidate=accepted_candidate,
                    schema_candidates=schema_candidates,
                    schema_retry_count=schema_retry_count,
                    repair_kinds=schema_repair_kinds,
                )

                output_data = make_output_base(data)
                output_data["rating_profile"] = RATING_PROFILE
                output_data["model_name"] = MODEL_NAME
                output_data["temperature"] = TEMPERATURE
                output_data["cache_enabled"] = USE_CACHE
                output_data["run_signature"] = CURRENT_RUN_SIGNATURE
                output_data["run_config"] = copy.deepcopy(
                    CURRENT_RUN_CONFIG
                )
                output_data["difficulty_rating_raw"] = copy.deepcopy(raw_result)
                output_data["difficulty_level_raw"] = str(
                    raw_result.get("difficulty_level", "") or ""
                )
                output_data["postprocess_actions"] = copy.deepcopy(
                    rating_result.get("postprocess_actions", [])
                )
                output_data["difficulty_rating"] = rating_result
                output_data["api_time_use"] = round(total_time, 2)
                output_data["api_prompt_tokens"] = total_prompt_tokens
                output_data["api_completion_tokens"] = total_completion_tokens
                output_data["api_total_tokens"] = total_tokens
                output_data.update(image_status)
                output_data["schema_retry_count"] = schema_retry_count
                output_data["schema_validation_errors"] = schema_errors
                output_data["schema_attempts"] = schema_attempts
                output_data.update(schema_audit)
                output_data["feature_normalization_actions"] = copy.deepcopy(
                    rating_result.get("feature_normalization_actions", [])
                )
                output_data["model_input_audit"] = {
                    "source_difficulty_sent": False,
                    "structured_text_primary": True,
                    "image_mode": CHEMISTRY_IMAGE_MODE,
                    "image_detail_strategy": CHEMISTRY_IMAGE_DETAIL,
                    "image_detail": image_status.get(
                        "image_detail",
                        "default",
                    ),
                    "selected_image_fields": image_status.get(
                        "image_input_fields",
                        [],
                    ),
                    "question_text_char_count": image_status.get(
                        "question_text_char_count",
                        0,
                    ),
                    "explicit_subquestion_count": image_status.get(
                        "explicit_subquestion_count",
                        1,
                    ),
                }
                async with FILE_LOCK:
                    async with aiofiles.open(
                        output_path,
                        "a",
                        encoding="utf-8",
                    ) as f:
                        await f.write(
                            json.dumps(output_data, ensure_ascii=False) + "\n"
                        )
                return
            except Exception as e:
                error_data = make_output_base(data)
                error_data["run_signature"] = CURRENT_RUN_SIGNATURE
                error_data["run_config"] = copy.deepcopy(
                    CURRENT_RUN_CONFIG
                )
                error_data["rating_error"] = (
                    f"question_id={question_id}; error={str(e)}"
                )
                error_data["last_model_text"] = raw_text
                error_data["difficulty_rating_raw"] = raw_result
                error_data["api_time_use"] = round(total_time, 2)
                error_data["api_prompt_tokens"] = total_prompt_tokens
                error_data["api_completion_tokens"] = total_completion_tokens
                error_data["api_total_tokens"] = total_tokens
                error_data.update(image_status)
                error_data["schema_retry_count"] = schema_retry_count
                error_data["schema_validation_errors"] = schema_errors
                error_data["schema_attempts"] = schema_attempts
                error_data.update(
                    build_schema_retry_audit(
                        first_candidate=first_model_candidate,
                        accepted_candidate=accepted_candidate,
                        schema_candidates=schema_candidates,
                        schema_retry_count=schema_retry_count,
                        repair_kinds=schema_repair_kinds,
                    )
                )
                async with FILE_LOCK:
                    async with aiofiles.open(
                        error_path,
                        "a",
                        encoding="utf-8",
                    ) as f:
                        await f.write(
                            json.dumps(error_data, ensure_ascii=False) + "\n"
                        )
                return


async def process_with_progress(
    data: Dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    pbar: tqdm,
    output_path: str,
    error_path: str,
    retries: int,
    timeout_sec: int,
) -> None:
    await process_single_question(data, session, semaphore, output_path, error_path, retries, timeout_sec)
    pbar.update(1)


def get_processed_question_ids(output_path: str) -> set:
    processed = set()
    if not os.path.exists(output_path):
        return processed
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    qid = item.get("question_id")
                    if qid is not None and str(qid).strip():
                        processed.add(str(qid))
                except Exception:
                    continue
    except Exception as e:
        print(f"扫描断点文件出错: {e}")
    return processed

# -------------------------- 7. 主执行流 --------------------------
async def main_batch_run() -> None:
    global USE_CACHE, CURRENT_RUN_SIGNATURE, CURRENT_RUN_CONFIG
    parser = argparse.ArgumentParser(description="初中化学难度评级多线程并发批量打标脚本 (带 Cache 优化)")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default=str(DEFAULT_PROMPT_PATH),
        help="化学打标提示词文件路径",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default=str(DEFAULT_INPUT_PATH),
        help="输入待打标 JSONL 数据集路径",
    )
    parser.add_argument("-o", "--output", type=str, default="chemistry_difficulty_rated_results.jsonl", help="输出保存打标结果的 JSONL 路径")
    parser.add_argument("-e", "--error", type=str, default="chemistry_difficulty_errors.jsonl", help="输出保存失败结果的 JSONL 路径")
    parser.add_argument("-c", "--concurrency", type=int, default=15, help="最大并发限制，默认 15")
    parser.add_argument("-t", "--timeout", type=int, default=180, help="单次 API 调用超时时间，默认 180 秒")
    parser.add_argument("-r", "--retries", type=int, default=3, help="失败最大重试次数，默认 3")
    parser.add_argument("-n", "--num", type=int, default=None, help="测试打标的限制数量（留空表示全部打标）")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样/打乱的种子，默认 42")
    parser.add_argument("--no-cache", action="store_true", help="禁用前缀缓存，每题发送完整提示词")
    args = parser.parse_args()

    random.seed(args.seed)
    if args.no_cache:
        USE_CACHE = False
    load_prompt_config(args.prompt)
    print(f"评级配置: {RATING_PROFILE}")
    print(f"模型: {MODEL_NAME}")
    print(
        "temperature: "
        + ("服务端默认" if TEMPERATURE is None else str(TEMPERATURE))
    )
    print(f"图片模式: {CHEMISTRY_IMAGE_MODE}")
    print(f"图片细节: {CHEMISTRY_IMAGE_DETAIL}")
    print("前缀缓存: " + ("启用" if USE_CACHE else "禁用"))

    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在，终止运行！")
        sys.exit(1)

    CURRENT_RUN_CONFIG = build_run_config(
        args.input,
        args.prompt,
        seed=args.seed,
        num=args.num,
    )
    CURRENT_RUN_SIGNATURE = build_run_signature(CURRENT_RUN_CONFIG)
    ensure_output_run_signature(
        args.output,
        CURRENT_RUN_SIGNATURE,
    )
    print(f"运行签名: {CURRENT_RUN_SIGNATURE}")

    print("正在加载待打标数据集...")
    questions: List[Dict[str, Any]] = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                questions.append(json.loads(line))
            except Exception:
                continue
    print(f"成功加载题目数据，共计 {len(questions)} 道题目。")

    if args.num is not None:
        questions = random.sample(questions, min(args.num, len(questions)))
        print(f"参数 -n 生效，随机抽样其中 {len(questions)} 道题进行测试。")
    else:
        random.shuffle(questions)
        print("全部打标启动：题目次序已随机打乱。")

    processed_ids = get_processed_question_ids(args.output)
    to_process = [
        q
        for q in questions
        if str(q.get("question_id", "")) not in processed_ids
    ]
    print(f"数据比对完成: 已完成数 {len(processed_ids)}，待处理数 {len(to_process)}")

    if not to_process:
        print("所有题目都已完成打标！")
        return

    semaphore = Semaphore(args.concurrency)
    pbar = tqdm(total=len(to_process), unit="item", desc="Chemistry Rating Progress")

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        if USE_CACHE:
            await get_or_create_cache(session, args.retries, args.timeout)
        tasks = [
            asyncio.create_task(
                process_with_progress(q, session, semaphore, pbar, args.output, args.error, args.retries, args.timeout)
            )
            for q in to_process
        ]
        if tasks:
            await asyncio.gather(*tasks)

    pbar.close()
    print("\n✨ 化学多线程批量打标运行结束！")
    print(f"👉 成功保存打标结果至: {os.path.abspath(args.output)}")
    print(f"👉 失败重试错误日志在: {os.path.abspath(args.error)}")


if __name__ == "__main__":
    start_time = time.time()
    try:
        asyncio.run(main_batch_run())
    except KeyboardInterrupt:
        print("\n收到键盘中断信号，程序已安全退出。")
    except Exception as e:
        print(f"\n批量运行中遇到未捕获异常: {e}")
    print(f"本次打标运行耗时: {round((time.time() - start_time) / 60, 2)} 分钟。")


