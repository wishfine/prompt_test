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
        OBSERVABLE_FEATURE_FIELDS,
        OBSERVABLE_ENUM_VALUES_BY_FIELD,
        OBSERVABLE_FALLBACK_LABELS,
        derive_observable_metrics,
        normalize_observable_features,
        observable_feature_quality_flags,
        validate_observable_features,
    )
except ModuleNotFoundError:
    from src.chemistry_observable_features_fxz import (
        OBSERVABLE_FEATURE_FIELDS,
        OBSERVABLE_ENUM_VALUES_BY_FIELD,
        OBSERVABLE_FALLBACK_LABELS,
        derive_observable_metrics,
        normalize_observable_features,
        observable_feature_quality_flags,
        validate_observable_features,
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
VISUAL_REFERENCE_RE = re.compile(
    r"(如图|图中|下图|图示|示意图|装置图|实验装置|流程图|"
    r"曲线|坐标图|关系图|图像|图象|表格|微观示意|粒子图|"
    r"看图|观察图|由图|据图|结合图)"
)
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

LEVEL_MAP = {
    "送分题": 1,
    "基础题": 2,
    "中等题": 3,
    "拔高题": 4,
    "压轴题": 5,
}

VALID_LEVELS = set(LEVEL_MAP.keys())

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
class ChemistrySchemaError(ValueError):
    """模型输出不满足当前V5十七项可观测特征契约。"""


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


def is_observable_feature_contract(features: Any) -> bool:
    """当前生产文件只接受V5十七项字段。"""
    return bool(
        isinstance(features, dict)
        and set(features) == set(OBSERVABLE_FEATURE_FIELDS)
    )


def observable_deep_quantitative_final_signal(
    features: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None,
) -> bool:
    """V5窄“拔高→压轴”深定量信号；广义版本只保留审计。"""
    if not is_observable_feature_contract(features):
        return False
    validated = validate_observable_features(features)
    metrics = derive_observable_metrics(validated)
    calculation_operations = set(validated["calculation_operations"])
    advanced_calculations = calculation_operations & {
        "组分消元或组成不变量",
        "差量",
        "多反应定量关系",
        "联立",
        "范围或分类计算",
    }
    question_metrics = derive_question_structure_metrics(data or {})
    simple_calculation_claim = bool(
        "联立" in calculation_operations
        and calculation_operations
        <= {"单一方程式", "单一守恒", "直接比例", "联立"}
    )
    few_explicit_questions = (
        2 <= question_metrics["explicit_subquestion_count"] <= 3
    )
    low_density_repeated_conservation_chain = bool(
        validated["solution_topology"] == "未知组成或量反推"
        and validated["reaction_structure"] == "产物进入后一反应"
        and metrics["effective_task_count"] <= 4
        and len(validated["rule_families"]) <= 2
        and (simple_calculation_claim or few_explicit_questions)
        and validated["experiment_operation"] == "无"
        and validated["graph_table_operation"] == "无"
    )
    return bool(
        len(validated["longest_solution_chain"]) >= 5
        and validated["reaction_structure"] != "无反应任务"
        and not low_density_repeated_conservation_chain
        and validated["solution_topology"]
        in {
            "条件分支或范围筛选",
            "未知组成或量反推",
            "未知组分消元或组成不变量",
            "双来源交叉验证",
            "多阶段反应网络",
        }
        and advanced_calculations
    )


def observable_strict_deep_quantitative_final_signal(
    features: Dict[str, Any],
    data: Optional[Dict[str, Any]] = None,
) -> bool:
    """识别可安全写回的深定量压轴严格子集。

    广义深定量信号不直接写回。本函数只保留严格交叉证据：真实分类讨论与范围计算共同出现；
    或范围边界继续约束多反应定量链；或多反应结构中的组成不变量同时
    由联立，或由差量与多反应定量关系共同约束；或六步以上两组反应数据
    连续反推未知组成。各路径都同时要求结构、操作与链深，单个方法名不能触发。
    """
    if not is_observable_feature_contract(features):
        return False
    if not observable_deep_quantitative_final_signal(features, data):
        return False
    validated = validate_observable_features(features)
    calculation_operations = set(
        validated["calculation_operations"]
    )
    chain_steps = len(validated["longest_solution_chain"])
    branch_range_signal = bool(
        validated["solution_topology"] == "条件分支或范围筛选"
        and "范围或分类计算" in calculation_operations
        and (
            "分类讨论" in validated["condition_operations"]
            or (
                chain_steps >= 5
                and "范围或边界"
                in validated["condition_operations"]
                and validated["reaction_structure"]
                not in {"无反应任务", "单一反应"}
            )
        )
    )
    strong_invariant_signal = bool(
        validated["solution_topology"]
        == "未知组分消元或组成不变量"
        and validated["reaction_structure"]
        not in {"无反应任务", "单一反应"}
        and (
            "联立" in calculation_operations
            or {
                "差量",
                "多反应定量关系",
            }.issubset(calculation_operations)
        )
    )
    deep_unknown_amount_signal = bool(
        chain_steps >= 6
        and validated["solution_topology"] == "未知组成或量反推"
        and validated["reaction_structure"] == "产物进入后一反应"
        and "多反应定量关系" in calculation_operations
    )
    return bool(
        branch_range_signal
        or strong_invariant_signal
        or deep_unknown_amount_signal
    )


def observable_dense_multiquestion_final_signal(
    features: Dict[str, Any],
    data: Dict[str, Any],
) -> bool:
    """识别窄压轴结构。

    信号同时依赖模型可核验事实与程序题面统计：至少四个显式小问、
    七项有效任务、四步最长链，并包含差量、多反应、联立、组分消元
    或范围分类计算。若只有单一反应和四步常规链，则至少需要八个显式
    小问才保留该广度通道，避免把普通综合探究连续抬成压轴。题干长度、
    课程跨度和小问数量均不能单独触发。
    """
    if not is_observable_feature_contract(features):
        return False
    validated = validate_observable_features(features)
    metrics = derive_observable_metrics(validated)
    question_metrics = derive_question_structure_metrics(data)
    advanced_calculations = {
        "组分消元或组成不变量",
        "差量",
        "多反应定量关系",
        "联立",
        "范围或分类计算",
    }
    return bool(
        metrics["longest_chain_steps"] >= 4
        and metrics["effective_task_count"] >= 7
        and question_metrics["explicit_subquestion_count"] >= 4
        and set(validated["calculation_operations"])
        & advanced_calculations
        and not (
            validated["reaction_structure"] == "单一反应"
            and metrics["longest_chain_steps"] == 4
            and question_metrics["explicit_subquestion_count"] < 8
        )
    )


def observable_multistage_multiquestion_multireaction_final_signal(
    features: Dict[str, Any],
    data: Dict[str, Any],
) -> bool:
    """识别多阶段/双来源的多问多反应压轴窄通道。

    该信号不依赖模型对任务数和链长的精确分拆：程序确定
    存在至少四个显式小问，模型同时识别出多阶段反应网络或
    双来源交叉验证，以及多反应定量关系。这一结构补充任务数或
    链长少拆时 dense 规则可能遗漏的压轴题。
    """
    if not is_observable_feature_contract(features):
        return False
    validated = validate_observable_features(features)
    question_metrics = derive_question_structure_metrics(data)
    return bool(
        question_metrics["explicit_subquestion_count"] >= 4
        and validated["solution_topology"]
        in {"多阶段反应网络", "双来源交叉验证"}
        and validated["reaction_structure"] != "无反应任务"
        and "多反应定量关系"
        in validated["calculation_operations"]
    )


def observable_double_source_multireaction_final_signal(
    features: Dict[str, Any],
) -> bool:
    """识别双来源交叉验证与多反应定量共同出现的压轴窄通道。

    不依赖小问数、任务数或链长的精确分拆；任一信号缺失均不触发。
    """
    if not is_observable_feature_contract(features):
        return False
    validated = validate_observable_features(features)
    return bool(
        validated["solution_topology"] == "双来源交叉验证"
        and "多反应定量关系"
        in validated["calculation_operations"]
    )


def _repair_coarse_reasoning_spill(rating_result: Dict[str, Any]) -> bool:
    """Repair a json_repair shape where reasoning leaked into coarse text."""
    raw_coarse = rating_result.get("coarse_difficulty")
    if not isinstance(raw_coarse, str) or '"reasoning"' not in raw_coarse:
        return False
    coarse_prefixes = {
        "送分/基础区间（1-2档": "送分/基础区间（1-2档）",
        "基础/中等区间（2-3档": "基础/中等区间（2-3档）",
        "中等/拔高区间（3-4档": "中等/拔高区间（3-4档）",
        "拔高/压轴区间（4-5档": "拔高/压轴区间（4-5档）",
    }
    normalized_coarse = next(
        (
            value
            for prefix, value in coarse_prefixes.items()
            if raw_coarse.strip().startswith(prefix)
        ),
        None,
    )
    core_match = re.search(
        r'"core_basis"\s*:\s*"(?P<core>.+)$',
        raw_coarse,
        flags=re.DOTALL,
    )
    if normalized_coarse is None or core_match is None:
        return False
    core_basis = core_match.group("core").strip().rstrip('"},').strip()
    if not core_basis:
        return False
    rating_result["coarse_difficulty"] = normalized_coarse
    if not str(rating_result.get("core_basis", "")).strip():
        rating_result["core_basis"] = core_basis
    return True


def validate_rating_contract(rating_result: Any) -> Dict[str, Any]:
    """校验固定顶层、V5特征、理由和相邻粗区间。"""
    if not isinstance(rating_result, dict):
        raise ChemistrySchemaError("模型输出必须是JSON对象")
    prepared = copy.deepcopy(rating_result)
    coarse_reasoning_spill_repaired = _repair_coarse_reasoning_spill(prepared)
    original_reasoning = copy.deepcopy(prepared.get("reasoning"))
    legacy_reason_fields = {
        field: copy.deepcopy(prepared.get(field))
        for field in (
            "core_basis",
            "hard_point",
            "why_not_lower",
            "why_not_higher",
            "reason",
        )
        if field in prepared
    }
    normalize_reasoning_schema(prepared)
    rating_schema_normalization_actions: List[Dict[str, Any]] = []
    if original_reasoning != prepared.get("reasoning") or legacy_reason_fields:
        rating_schema_normalization_actions.append(
            {
                "field": "reasoning",
                "from": original_reasoning or legacy_reason_fields,
                "to": copy.deepcopy(prepared.get("reasoning")),
                "reason": "顶层理由字段确定性合并为reasoning",
            }
        )
    if coarse_reasoning_spill_repaired:
        rating_schema_normalization_actions.append(
            {
                "field": "coarse_difficulty/reasoning",
                "from": rating_result.get("coarse_difficulty"),
                "to": {
                    "coarse_difficulty": prepared.get("coarse_difficulty"),
                    "reasoning": copy.deepcopy(prepared.get("reasoning")),
                },
                "reason": "json_repair导致的粗区间与reasoning粘连确定性拆分",
            }
        )
    required = {
        "features",
        "coarse_difficulty",
        "reasoning",
        "difficulty_level",
    }
    missing = sorted(required - set(prepared))
    if missing:
        raise ChemistrySchemaError(f"顶层字段缺失: {missing}")

    level = str(prepared.get("difficulty_level", "")).strip()
    if level not in VALID_LEVELS:
        raise ChemistrySchemaError(f"difficulty_level非法: {level!r}")
    coarse = str(prepared.get("coarse_difficulty", "")).strip()
    valid_coarse = {
        "送分/基础区间（1-2档）",
        "基础/中等区间（2-3档）",
        "中等/拔高区间（3-4档）",
        "拔高/压轴区间（4-5档）",
    }
    if coarse not in valid_coarse:
        raise ChemistrySchemaError(f"coarse_difficulty非法: {coarse!r}")
    coarse_levels = {
        "送分/基础区间（1-2档）": {"送分题", "基础题"},
        "基础/中等区间（2-3档）": {"基础题", "中等题"},
        "中等/拔高区间（3-4档）": {"中等题", "拔高题"},
        "拔高/压轴区间（4-5档）": {"拔高题", "压轴题"},
    }
    if level not in coarse_levels[coarse]:
        raise ChemistrySchemaError(
            f"coarse_difficulty={coarse!r}不包含最终等级{level!r}"
        )

    reasoning = prepared.get("reasoning")
    reason_fields = {
        "core_basis",
        "hard_point",
        "why_not_lower",
        "why_not_higher",
    }
    if not isinstance(reasoning, dict) or set(reasoning) != reason_fields:
        raise ChemistrySchemaError(
            "reasoning必须且只能包含core_basis、hard_point、"
            "why_not_lower、why_not_higher"
        )
    if any(not str(reasoning.get(field, "")).strip() for field in reason_fields):
        raise ChemistrySchemaError("reasoning四个字段均不得为空")

    raw_features = prepared["features"]
    if not isinstance(raw_features, dict):
        raise ChemistrySchemaError("features必须是JSON对象")
    normalized_features, normalization_actions = (
        normalize_observable_features(raw_features)
    )
    if not isinstance(normalized_features, dict) or set(normalized_features) != set(
        OBSERVABLE_FEATURE_FIELDS
    ):
        actual = set(normalized_features) if isinstance(normalized_features, dict) else set()
        missing = sorted(set(OBSERVABLE_FEATURE_FIELDS) - actual)
        extra = sorted(actual - set(OBSERVABLE_FEATURE_FIELDS))
        raise ChemistrySchemaError(
            f"V5可观测特征字段集不匹配: missing={missing}, extra={extra}"
        )
    try:
        prepared["features"] = validate_observable_features(
            normalized_features
        )
    except ValueError as exc:
        raise ChemistrySchemaError(str(exc)) from exc
    prepared["feature_normalization_actions"] = normalization_actions
    prepared["feature_contract_quality_flags"] = (
        observable_feature_quality_flags(
            prepared["features"], normalization_actions
        )
    )
    if prepared.get("feature_schema_repair_kind") == "semantic":
        prepared["feature_contract_quality_flags"].append(
            "semantic_schema_repaired"
        )
    prepared["feature_contract_quality_flags"] = list(
        dict.fromkeys(prepared["feature_contract_quality_flags"])
    )
    prepared["rating_schema_normalization_actions"] = (
        rating_schema_normalization_actions
    )
    return prepared

# -------------------------- 4. 后处理纠偏规则 --------------------------
def normalize_reasoning_schema(rating_result: Dict[str, Any]) -> None:
    reasoning = rating_result.get("reasoning")
    reason = rating_result.get("reason")
    normalized = {
        "core_basis": "",
        "hard_point": "",
        "why_not_lower": "",
        "why_not_higher": "",
    }
    if isinstance(reasoning, dict):
        normalized.update(reasoning)
    elif isinstance(reason, dict):
        normalized.update(reason)
    elif isinstance(reasoning, str) and reasoning:
        normalized["core_basis"] = reasoning
    elif isinstance(reason, str) and reason:
        normalized["core_basis"] = reason
    for field in (
        "core_basis",
        "hard_point",
        "why_not_lower",
        "why_not_higher",
    ):
        top_level_value = rating_result.get(field)
        if not normalized[field] and isinstance(top_level_value, str):
            normalized[field] = top_level_value
    if not normalized["core_basis"] and normalized["hard_point"]:
        normalized["core_basis"] = normalized["hard_point"]
    rating_result["reasoning"] = normalized
    rating_result.pop("reason", None)
    for field in (
        "core_basis",
        "hard_point",
        "why_not_lower",
        "why_not_higher",
    ):
        rating_result.pop(field, None)


def set_level_with_reason(
    rating_result: Dict[str, Any],
    level: str,
    core_basis_prefix: str,
    *,
    rule: str = "chemistry_adjacent_calibration",
    evidence: Optional[Sequence[str]] = None,
    max_level_distance: int = 1,
) -> None:
    """设置后处理难度，并记录可审计的改档轨迹。

    v6.1 说明：
    - 不改变任何分类规则，只把每一次自动升/降档记录到 postprocess_trace；
    - 后续由 sync_reasoning_after_postprocess() 统一同步 why_not_lower / why_not_higher，
      避免最终档位与原始模型解释互相矛盾。
    """
    previous_level = rating_result.get("difficulty_level", "")
    if previous_level not in LEVEL_MAP or level not in LEVEL_MAP:
        raise ValueError(
            f"后处理档位非法: {previous_level!r} -> {level!r}"
        )
    level_distance = abs(
        LEVEL_MAP[previous_level] - LEVEL_MAP[level]
    )
    if previous_level != level and not (
        1 <= level_distance <= max_level_distance
    ):
        raise ValueError(
            "后处理调整距离超出该规则许可范围: "
            f"{previous_level} -> {level}, "
            f"max_level_distance={max_level_distance}"
        )
    rating_result.setdefault("postprocess_original_level", previous_level)
    rating_result.setdefault("postprocess_trace", [])
    if previous_level != level:
        rating_result["postprocess_trace"].append({
            "rule": rule,
            "from": previous_level,
            "to": level,
            "level_distance": level_distance,
            "evidence": list(evidence or [core_basis_prefix]),
            "reason": core_basis_prefix,
        })
    rating_result["postprocess_note"] = core_basis_prefix
    rating_result["difficulty_level"] = level

    reasoning = rating_result.setdefault("reasoning", {
        "core_basis": "",
        "hard_point": "",
        "why_not_lower": "",
        "why_not_higher": "",
    })
    original_basis = reasoning.get("core_basis", "")
    reasoning["core_basis"] = f"【{core_basis_prefix}】。原始依据：{original_basis}"


def sync_coarse_difficulty(rating_result: Dict[str, Any]) -> None:
    level = rating_result.get("difficulty_level", "")
    if level in ["送分题", "基础题"]:
        rating_result["coarse_difficulty"] = "送分/基础区间（1-2档）"
    elif level == "中等题":
        rating_result["coarse_difficulty"] = "基础/中等区间（2-3档）"
    elif level == "拔高题":
        rating_result["coarse_difficulty"] = "中等/拔高区间（3-4档）"
    elif level == "压轴题":
        rating_result["coarse_difficulty"] = "拔高/压轴区间（4-5档）"


def contains_any(text: str, keywords: List[str]) -> bool:
    return any(k in text for k in keywords)


def visible_text(data: Dict[str, Any], include_analysis: bool = False) -> str:
    parts = [str(data.get("stem", "") or ""), str(data.get("options", "") or "")]
    if include_analysis:
        parts.append(str(data.get("analysis", "") or ""))
    for sq in data.get("sub_questions", []) or []:
        if isinstance(sq, dict):
            parts.append(str(sq.get("stem", "") or ""))
            parts.append(str(sq.get("options", "") or ""))
            if include_analysis:
                parts.append(str(sq.get("analysis", "") or ""))
    return "\n".join(parts)


def derive_question_structure_metrics(data: Dict[str, Any]) -> Dict[str, int]:
    """由结构化题面确定性统计字数和显式设问数。

    只统计题干、选项和子题题面，不读取解析、图片URL或标签。结构化
    ``sub_questions`` 优先；缺少该字段时才保守识别（1）、①等编号。
    单一整题记为1个作答目标，避免把“没有子题数组”误写成零任务。
    """
    text = visible_text(data, include_analysis=False)
    text_without_urls = re.sub(r"https?://\S+", "", text)
    text_without_placeholders = re.sub(
        r"(?:\[图片\]|【图片】|<image[^>]*>|\{\{image[^}]*\}\})",
        "",
        text_without_urls,
        flags=re.IGNORECASE,
    )
    question_text_char_count = len(
        re.sub(r"\s+", "", text_without_placeholders)
    )

    sub_questions = data.get("sub_questions", []) or []
    if isinstance(sub_questions, list) and sub_questions:
        explicit_subquestion_count = len(sub_questions)
    else:
        stem = str(data.get("stem", "") or "")
        markers = re.findall(
            r"(?:[①②③④⑤⑥⑦⑧⑨⑩]|[（\(][1-9]\d?[）\)])",
            stem,
        )
        explicit_subquestion_count = max(1, len(markers))

    return {
        "question_text_char_count": question_text_char_count,
        "explicit_subquestion_count": explicit_subquestion_count,
    }


def fill_blank_subquestion_count(data: Dict[str, Any]) -> int:
    """统计结构化填空小问，避免把选择项计作小问。"""
    sub_questions = data.get("sub_questions", []) or []
    if not isinstance(sub_questions, list):
        return 0
    return sum(
        1
        for sub_question in sub_questions
        if isinstance(sub_question, dict)
        and not str(sub_question.get("options", "") or "").strip()
    )


def count_choice_options(data: Dict[str, Any]) -> int:
    """统计显式 A-D 选项，仅作客观结构门控。"""
    options = str(data.get("options", "") or "")
    return len(
        re.findall(
            r"(?m)^\s*[A-DＡ-Ｄ][\.、．:：\)]",
            options,
        )
    )


def count_reaction_arrows(text: str) -> int:
    return len(
        re.findall(
            r"(?:→|->|⇒|↔|⇌|\\xrightarrow|\\rightarrow|\\mathop\{?→)",
            text,
        )
    )


def parallel_application_floor_signal(
    data: Dict[str, Any],
) -> Optional[str]:
    """识别不应落入纯直接检索的窄结构。

    这些信号只建立基础题下限，不按物质名直接定档。所有判断均来自
    题干和选项，不读取解析，也不依赖模型自报的 Core-12。
    """
    text = visible_text(data, include_analysis=False)
    if count_choice_options(data) < 4:
        return None

    if (
        re.search(
            r"长期(?:暴露|露置)|长时间(?:暴露|露置)|"
            r"久置|敞口放置",
            text,
        )
    ):
        return "多种物质在空气中变化需要分别应用吸水、挥发或反应规则"

    numbered_items = re.findall(
        r"(?:[①②③④⑤⑥⑦⑧⑨⑩]|"
        r"(?<!\d)[（\(][1-9][）\)])",
        text,
    )
    if (
        len(numbered_items) >= 5
        and re.search(r"(?:分别)?(?:放入|加入).{0,20}水", text)
        and "充分搅拌" in text
        and "得到溶液" in text
    ):
        return "多种材料需逐项判断分散体系形成条件"

    property_families = sum(
        bool(re.search(pattern, text))
        for pattern in (
            r"用途|清洁|清洗|去油污",
            r"腐蚀|安全|皮肤|危险",
            r"变质|久置|空气|密封",
            r"指示剂|酚酞|石蕊|酸碱性",
        )
    )
    if property_families >= 3:
        return "同一物质的用途、安全、保存或指示剂性质需切换多类规则"

    periodic_families = sum(
        bool(re.search(pattern, text))
        for pattern in (
            r"元素符号|符号为",
            r"质子数|电子数|原子序数",
            r"相对原子质量|原子质量",
            r"属于.{0,3}(?:金属|非金属)元素",
        )
    )
    if periodic_families >= 3:
        return "元素信息题同时核验符号、粒子数、类别或相对原子质量"

    return None


def observable_multi_rule_multitopic_medium_signal(
    model_features: Dict[str, Any],
) -> Optional[str]:
    """V5下窄化的“基础→中等”多规则跨课题信号。

    只接受正式 V5 十七项契约，并同时要求至少四项任务、
    三类具体回答规则和两个课题；题长、小问数或单纯跨课题
    都不能单独触发。
    """
    if not is_observable_feature_contract(model_features):
        return None
    metrics = derive_observable_metrics(model_features)
    if not (
        metrics["effective_task_count"] >= 4
        and metrics["rule_family_count"] >= 3
        and metrics["curriculum_topic_count"] >= 2
    ):
        return None
    return (
        "至少四项非重复任务横跨两个课题，"
        "且需切换至少三类具体回答规则"
    )


def observable_parallel_phenomena_multitopic_medium_signal(
    model_features: Dict[str, Any],
) -> Optional[str]:
    """V5下可写回的氧气现象跨课题窄信号。

    U2-2将规则限制为“氧气/燃烧现象与其他反应现象并列辨析”的作用域，
    并不作为单独升档依据；任务量、课题跨度、并列反应和现象规则
    缺一不可。
    """
    if not is_observable_feature_contract(model_features):
        return None
    metrics = derive_observable_metrics(model_features)
    if not (
        metrics["effective_task_count"] >= 4
        and metrics["curriculum_topic_count"] >= 3
        and metrics["rule_family_count"] <= 2
        and "U2-2" in model_features.get("curriculum_topics", [])
        and "性质用途或现象判断"
        in model_features.get("rule_families", [])
        and model_features.get("parallel_task_relation")
        == "同一规则下多个对象"
        and model_features.get("reaction_structure") == "多个并列反应"
    ):
        return None
    return (
        "至少四项氧气/燃烧及其他反应现象核验横跨三个课题，"
        "各项需分别核对反应条件、产物状态或规范现象"
    )


def observable_high_density_evidence_hard_signal(
    model_features: Dict[str, Any],
) -> Optional[str]:
    """V5下窄化的“中等→拔高”高密度证据信号。

    六类以上具体回答规则只有在多证据必须共同成立，且实验中存在
    方案设计、方案评价或多阶段定量探究这类决定性操作时，才形成拔高下限。
    多个独立的常规数据归纳、现象解释或规则切换不写回。
    """
    if not is_observable_feature_contract(model_features):
        return None
    metrics = derive_observable_metrics(model_features)
    if not (
        metrics["rule_family_count"] >= 6
        and "多证据共同成立"
        in model_features.get("evidence_operations", [])
        and model_features.get("experiment_operation")
        in {"方案设计", "方案评价或补充实验", "多阶段定量探究"}
    ):
        return None
    return (
        "至少六类具体回答规则共同参与，"
        "多条证据需联合成立，且存在决定性实验设计、"
        "评价或多阶段定量探究"
    )


def measuring_cylinder_error_chain_signal(
    data: Dict[str, Any],
) -> Optional[str]:
    """识别量筒俯仰视导致体积或配制误差的连续关系链。"""
    text = visible_text(data, include_analysis=True)
    if "量筒" not in text or not re.search(r"俯视|仰视", text):
        return None
    if not re.search(
        r"实际体积|实际取出|取液体积|配制结果|浓度|"
        r"示数.{0,8}(?:偏大|偏小|大于|小于)|"
        r"(?:偏大|偏小|大于|小于).{0,8}(?:示数|实际)|误差",
        text,
    ):
        return None
    return "量筒俯仰视需连续判断示数、实际体积及误差方向"


def reaction_validation_floor_signal(
    data: Dict[str, Any],
) -> Optional[str]:
    """识别至少属于中等题比较的多选项反应核验。"""
    text = visible_text(data, include_analysis=False)
    stem = str(data.get("stem", "") or "")
    if count_choice_options(data) < 4:
        return None

    if (
        re.search(
            r"转化|给定条件|一定条件|各步反应|实现下列",
            stem,
        )
        and count_reaction_arrows(text) >= 2
    ):
        return "多个候选连续转化链需逐段核验反应物、条件和产物"

    if (
        "化学方程式" in text
        and re.search(
            r"反应类型|基本反应类型|化合反应|分解反应|"
            r"置换反应|复分解反应",
            text,
        )
    ):
        return "每个候选同时核验方程式事实、配平条件和反应类型"

    return None


def _valid_operations(features: Dict[str, Any], field: str) -> set[str]:
    return set(features[field]) - OBSERVABLE_FALLBACK_LABELS


def _has_cross_module_fusion(metrics: Dict[str, Any]) -> bool:
    return bool(
        metrics["curriculum_unit_count"] >= 2
        and metrics["curriculum_coupling_type"]
        not in {"同单元跨课题并列", "跨单元并列"}
    )


def _has_nontrivial_chain(features: Dict[str, Any], metrics: Dict[str, Any]) -> bool:
    return bool(
        _valid_operations(features, "condition_operations")
        or _valid_operations(features, "representation_operations")
        or _valid_operations(features, "evidence_operations")
        or _valid_operations(features, "calculation_operations")
        or features["reaction_structure"] != "无反应任务"
        or features["experiment_operation"] != "无"
        or features["graph_table_operation"] != "无"
        or features["new_information_operation"] != "无新信息"
        or metrics["longest_chain_steps"] > 1
    )


def _has_advanced_experiment(features: Dict[str, Any]) -> bool:
    return bool(
        features["experiment_operation"]
        in {
            "变量控制",
            "现象解释",
            "数据归纳",
            "方案设计",
            "方案评价或补充实验",
            "多阶段定量探究",
        }
        or features["error_analysis_operation"]
        in {"多因素误差比较", "定量误差修正"}
        or (
            features["error_analysis_operation"]
            in {
                "读数偏差到实际量判断",
                "操作偏差到最终结果方向",
            }
            and features["experiment_operation"] == "基础操作或读数"
        )
    )


def _has_high_evidence(features: Dict[str, Any]) -> bool:
    return bool(
        _valid_operations(features, "evidence_operations")
        & {
            "多证据共同成立",
            "排除一个候选",
            "排除多个候选解释",
            "处理冲突证据",
            "补充实验获得唯一结论",
        }
    )


def coordinated_multigraph_reaction_signal(
    features: Dict[str, Any],
    metrics: Dict[str, Any],
    data: Dict[str, Any],
) -> bool:
    """识别同一反应进程中三幅以上关联图像的联合反推。"""
    text = visible_text(data, include_analysis=False)
    figure_markers: set[str] = set()
    for left, right in re.findall(
        r"([甲乙丙丁戊己])\s*图|图\s*([甲乙丙丁戊己])",
        text,
    ):
        figure_markers.add(left or right)
    return bool(
        len(figure_markers) >= 3
        and re.search(r"同一.{0,8}(?:实验|反应|过程)", text)
        and re.search(r"随.{0,6}(?:时间|反应).{0,8}变化", text)
        and features["graph_table_operation"] == "拐点平台或分段"
        and _has_cross_module_fusion(metrics)
        and metrics["representation_operation_count"] >= 2
        and metrics["condition_operation_count"] >= 2
        and _has_high_evidence(features)
    )


def cross_module_knowledge_breadth_signal(
    features: Dict[str, Any],
    metrics: Dict[str, Any],
    data: Dict[str, Any],
) -> bool:
    """识别跨多个生活板块、含大量独立判断的知识归纳题。"""
    text = visible_text(data, include_analysis=False)
    numbered_items = re.findall(
        r"[①②③④⑤⑥⑦⑧⑨⑩]|(?<!\d)[（\(][1-9][）\)]",
        text,
    )
    knowledge_sections = re.findall(
        r"化学与[\u4e00-\u9fff]{1,8}",
        text,
    )
    return bool(
        len(numbered_items) >= 8
        and len(knowledge_sections) >= 3
        and metrics["longest_chain_steps"] <= 1
        and _has_cross_module_fusion(metrics)
        and metrics["representation_operation_count"] == 0
        and features["experiment_operation"] == "无"
        and features["graph_table_operation"] == "无"
        and metrics["calculation_operation_count"] == 0
        and features["parallel_task_relation"]
        in {"同一规则下多个对象", "不同规则的独立任务"}
    )


def controllable_gas_scheme_signal(
    features: Dict[str, Any],
    metrics: Dict[str, Any],
    data: Dict[str, Any],
) -> bool:
    """识别可随开随停装置对多组气体制备方案的双约束筛选。"""
    text = visible_text(data, include_analysis=False)
    numbered_items = re.findall(
        r"[①②③④⑤⑥⑦⑧⑨⑩]|(?<!\d)[（\(][1-9][）\)]",
        text,
    )
    return bool(
        len(numbered_items) >= 4
        and contains_any(text, ["制备气体", "制取气体"])
        and re.search(
            r"控制反应的(?:发生与停止|发生和停止)|"
            r"随时控制反应|随开随停|启普发生器",
            text,
        )
        and metrics["condition_operation_count"] >= 2
        and features["experiment_operation"] != "无"
    )


def multi_activity_project_signal(
    features: Dict[str, Any],
    metrics: Dict[str, Any],
    data: Dict[str, Any],
) -> bool:
    """识别至少两项活动、三项实验构成的项目式探究链。"""
    text = visible_text(data, include_analysis=False)
    activity_markers = set(
        re.findall(r"活动\s*[一二三四五六123456]", text)
    )
    experiment_markers = set(
        re.findall(r"实验\s*[一二三四五六123456]", text)
    )
    return bool(
        contains_any(text, ["项目式学习", "项目学习"])
        and len(activity_markers) >= 2
        and len(experiment_markers) >= 3
        and metrics["condition_operation_count"] >= 2
        and _has_advanced_experiment(features)
    )


def dense_project_experiment_signal(
    features: Dict[str, Any],
    metrics: Dict[str, Any],
    data: Dict[str, Any],
) -> bool:
    text = visible_text(data, include_analysis=False)
    experiment_markers = set(re.findall(
        r"实验[一二三四五六123456]",
        text,
    ))
    all_markers = set(re.findall(
        r"(?:活动|任务|实验)[一二三四五六123456]",
        text,
    ))
    return bool(
        len(experiment_markers) >= 4
        and len(all_markers) >= 6
        and metrics["condition_operation_count"] >= 2
        and _has_high_evidence(features)
        and _has_advanced_experiment(features)
    )


def strong_segment_graph_chain_signal(
    features: Dict[str, Any],
    metrics: Dict[str, Any],
) -> bool:
    return bool(
        features["graph_table_operation"] == "拐点平台或分段"
        and _has_cross_module_fusion(metrics)
        and metrics["representation_operation_count"] >= 2
        and metrics["condition_operation_count"] >= 2
        and _has_high_evidence(features)
        and _has_advanced_experiment(features)
    )


def sync_reasoning_after_postprocess(rating_result: Dict[str, Any]) -> None:
    """后处理改档后的解释同步层。

    只在 postprocess_trace 非空时生效；不改变 difficulty_level 和 features。
    目标是解决“最终档位已被后处理改成 X，但 why_not_higher 仍沿用原模型解释”的前后矛盾问题。
    """
    trace = rating_result.get("postprocess_trace") or []
    if not trace:
        return

    final_level = rating_result.get("difficulty_level", "")
    reason_text = "；".join(str(item.get("reason", "")) for item in trace if item.get("reason"))
    if not reason_text:
        reason_text = str(rating_result.get("postprocess_note", "")) or "后处理规则修正"

    reasoning = rating_result.setdefault("reasoning", {
        "core_basis": "",
        "hard_point": "",
        "why_not_lower": "",
        "why_not_higher": "",
    })

    if final_level == "送分题":
        reasoning["why_not_lower"] = "送分题已经是最低难度档，无更低档。"
        reasoning["why_not_higher"] = f"后处理最终判为送分题，原因：{reason_text}。题目只涉及低阶直接识记或常识匹配，不需要提升到基础题。"
    elif final_level == "基础题":
        reasoning["why_not_lower"] = f"后处理最终判为基础题，原因：{reason_text}。题目需要概念辨析、基础化学用语、简单计算或基础实验操作，不能降为送分题。"
        reasoning["why_not_higher"] = "题目缺少中等题所需的多反应链、实验探究、图表归纳、成分推断证据链或守恒计算，因此不需要判为中等题。"
    elif final_level == "中等题":
        reasoning["why_not_lower"] = f"后处理最终判为中等题，原因：{reason_text}。题目存在一定综合性或标准化学分析任务，不能降为基础题。"
        reasoning["why_not_higher"] = "题目路径仍属于常规中考方法，缺少明显拔高卡点，如方案评价、证据冲突排除、复杂守恒、图像拐点反推或多反应多约束，因此不需要判为拔高题。"
    elif final_level == "拔高题":
        reasoning["why_not_lower"] = f"后处理最终判为拔高题，原因：{reason_text}。题目存在明显卡点，不能降为中等题。"
        reasoning["why_not_higher"] = "虽然题目有拔高因素，但尚未同时满足压轴题所需的复杂证据/计算/方案评价、多反应或多约束、递进多问等核心组合，因此不需要判为压轴题。"
    elif final_level == "压轴题":
        reasoning["why_not_lower"] = f"后处理最终判为压轴题，原因：{reason_text}。题目具备多项高阶特征和压轴核心组合，不能降为拔高题。"
        reasoning["why_not_higher"] = "压轴题已经是最高难度档，无更高档。"

def add_feature_audit_flags(
    rating_result: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    """记录V5可观测特征中的结构异常，只审计、不直接改档。"""
    flags: List[str] = []
    text = visible_text(data, include_analysis=True)
    model_features = rating_result.get("features") or {}
    if is_observable_feature_contract(model_features):
        observable_metrics = derive_observable_metrics(model_features)
        core_basis = str(
            rating_result.get("reasoning", {}).get("core_basis", "")
        )
        core_basis_without_negative_same_unit = re.sub(
            r"(?:不(?:属于)?|非)同(?:一)?单元", "", core_basis
        )
        if (
            observable_metrics["curriculum_span_type"] == "跨单元"
            and re.search(
                r"同(?:一)?单元(?:跨课题)?(?:并列|耦合|相邻课题)?",
                core_basis_without_negative_same_unit,
            )
        ):
            flags.append(
                "课程跨度自检：curriculum_topics含不同U前缀却写成同单元"
            )
        if (
            observable_metrics["curriculum_coupling_type"]
            in {"同单元跨课题并列", "跨单元并列"}
            and len(model_features["longest_solution_chain"]) >= 4
        ):
            flags.append(
                "纵向链自检：独立任务疑似按选项累计最长链，"
                "应只保留最高难单项自身的依赖链"
            )
        if (
            model_features.get("new_information_operation")
            == "依赖题干未给出的超纲化学知识"
        ):
            flags.append(
                "课程越界审计：题目依赖题干未给出的超纲化学知识；"
                "需人工复核，不能按陌生名称机械升档"
            )
        if (
            VISUAL_REFERENCE_RE.search(text)
            and model_features.get("graph_table_operation") == "无"
            and model_features.get("visual_task_structure")
            == "无必要视觉信息"
        ):
            flags.append(
                "题面明确引用图表/流程/装置，但视觉与图表字段均为无；"
                "需检查图片是否遗漏"
            )
    if rating_result.get("postprocess_trace"):
        flags.append("后处理已作一次结构校准，原始模型结果另行保留")
    rating_result["feature_audit_flags"] = list(dict.fromkeys(flags))


def postprocess_chemistry_difficulty(
    rating_result: Dict[str, Any],
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """V5正式后处理：仅保留当前Prompt实际使用的窄教师边界校准。

    生产路径固定为V5十七项可观测特征，只保留当前实际写回的教师边界规则。
    已停用的候选审计规则不再进入生产后处理链。
    """
    if not rating_result:
        return rating_result

    rating_result = validate_rating_contract(rating_result)
    normalize_reasoning_schema(rating_result)
    raw_level = rating_result["difficulty_level"]
    raw_coarse_difficulty = rating_result["coarse_difficulty"]
    rating_result["coarse_difficulty_raw"] = raw_coarse_difficulty
    rating_result["postprocess_original_level"] = raw_level
    rating_result["postprocess_trace"] = []
    rating_result["postprocess_actions"] = []
    rating_result["postprocess_profile"] = "chemistry_observable_v5_fxz_production"
    rating_result["postprocess_writeback_enabled"] = (
        CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
    )
    rating_result["teacher_distribution_guard_enabled"] = (
        CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS
    )
    rating_result["teacher_distribution_guard_writeback_enabled"] = (
        CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
    )

    feature_quality_flags = list(
        rating_result.get("feature_contract_quality_flags", [])
    )
    feature_quality_blocks_writeback = bool(feature_quality_flags)
    rating_result["feature_quality_blocks_writeback"] = (
        feature_quality_blocks_writeback
    )
    rating_result["writeback_eligible"] = not feature_quality_blocks_writeback
    rating_result["writeback_ineligible_reasons"] = (
        feature_quality_flags if feature_quality_blocks_writeback else []
    )

    model_features = rating_result["features"]
    if not is_observable_feature_contract(model_features):
        raise ChemistrySchemaError("FXZ生产脚本只接受V5十七项可观测特征")
    observable_metrics = derive_observable_metrics(model_features)
    observable_metrics.update(derive_question_structure_metrics(data or {}))
    rating_result["feature_schema_version"] = "chemistry_observable_v5"
    rating_result["observable_metrics"] = observable_metrics
    rating_result["schema_validation_passed"] = True

    # 教师分布校准只使用可复核的结构特征，并且每题最多提出一次调整。
    # 常规动作只移动一个相邻档；唯一的两档托底是“送分→中等”的多选项
    # 连续反应核验，它依赖题干中的多个反应箭头和条件核验，不依赖模型
    # 自报 depth。规则不按题库配额切档；生产默认写回，A/B 时可关闭。
    teacher_guard_active = bool(
        CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS
        or CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
    )
    teacher_candidate_result = copy.deepcopy(rating_result)
    parallel_floor = parallel_application_floor_signal(data)
    easy_many_fill_blank_subquestions_floor = (
        fill_blank_subquestion_count(data or {}) >= 4
    )
    measuring_cylinder_error_chain = (
        measuring_cylinder_error_chain_signal(data)
    )
    multi_rule_multitopic_medium = (
        observable_multi_rule_multitopic_medium_signal(model_features)
    )
    parallel_phenomena_multitopic_medium = (
        observable_parallel_phenomena_multitopic_medium_signal(
            model_features
        )
    )
    high_density_evidence_hard = (
        observable_high_density_evidence_hard_signal(model_features)
    )
    reaction_floor = reaction_validation_floor_signal(data)
    if teacher_guard_active:
        if (
            raw_level == "送分题"
            and reaction_floor
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "严重低估安全底线：多选项连续反应核验不是直接识记",
                rule="teacher_easy_to_medium_reaction_conversion_floor",
                evidence=[reaction_floor],
                max_level_distance=2,
            )
        elif (
            raw_level == "送分题"
            and parallel_floor
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "基础题",
                "严重低估安全底线：多个对象需要切换不同化学应用规则",
                rule="teacher_easy_to_basic_parallel_application_floor",
                evidence=[parallel_floor],
            )
        elif (
            raw_level == "送分题"
            and easy_many_fill_blank_subquestions_floor
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "基础题",
                "教师口径：四个及以上填空小问不按送分题处理",
                rule="teacher_easy_to_basic_four_fill_blank_subquestions",
                evidence=[
                    "填空小问数="
                    + str(fill_blank_subquestion_count(data or {})),
                ],
            )
        elif (
            raw_level == "基础题"
            and model_features.get("error_analysis_operation")
            in {
                "读数偏差到实际量判断",
                "操作偏差到最终结果方向",
            }
            and len(model_features["longest_solution_chain"]) >= 2
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "结构边界窄校准：误差题需要由偏差来源连续推到实际量或最终结果",
                rule="teacher_basic_to_medium_observable_error_chain",
                evidence=[
                    "误差操作="
                    + model_features["error_analysis_operation"],
                    "最长链="
                    + " → ".join(
                        model_features["longest_solution_chain"]
                    ),
                ],
            )
        elif (
            raw_level == "基础题"
            and measuring_cylinder_error_chain
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "结构边界窄校准：量筒俯仰视需完成示数—实际体积—误差方向连续推导",
                rule="teacher_basic_to_medium_measuring_cylinder_error_chain",
                evidence=[measuring_cylinder_error_chain],
            )
        elif (
            raw_level == "基础题"
            and multi_rule_multitopic_medium
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "结构边界窄校准：多项非重复任务跨课题切换多类具体回答规则",
                rule="teacher_basic_to_medium_multi_rule_multitopic",
                evidence=[multi_rule_multitopic_medium],
            )
        elif (
            raw_level == "基础题"
            and parallel_phenomena_multitopic_medium
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "结构边界窄校准：多课题反应现象需分别核对条件、产物状态与规范表述",
                rule=(
                    "teacher_basic_to_medium_"
                    "parallel_phenomena_multitopic"
                ),
                evidence=[parallel_phenomena_multitopic_medium],
            )
        elif (
            raw_level == "基础题"
            and reaction_floor
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "严重低估安全底线：连续转化或方程式—反应类型需要双重核验",
                rule="teacher_basic_to_medium_reaction_validation_floor",
                evidence=[reaction_floor],
            )
        elif (
            raw_level == "基础题"
            and controllable_gas_scheme_signal(
                model_features, observable_metrics, data
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "严重低估安全底线：可控气体发生装置需对多组方案进行双约束筛选",
                rule="teacher_basic_to_medium_controllable_gas_scheme_floor",
                evidence=[
                    "题面同时出现可随开随停装置和四组以上制气方案",
                    "条件操作数="
                    + str(observable_metrics["condition_operation_count"]),
                    "实验操作=" + model_features["experiment_operation"],
                ],
            )
        elif (
            raw_level == "基础题"
            and cross_module_knowledge_breadth_signal(
                model_features, observable_metrics, data
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "严重低估安全底线：跨多个生活板块的大量独立判断形成真实任务广度",
                rule="teacher_basic_to_medium_cross_module_breadth_floor",
                evidence=[
                    "题面包含三个以上知识板块和八项以上独立判断",
                    "课程单元数="
                    + str(observable_metrics["curriculum_unit_count"]),
                    "并行任务关系="
                    + model_features["parallel_task_relation"],
                ],
            )
        elif (
            raw_level == "基础题"
            and _has_nontrivial_chain(model_features, observable_metrics)
            and observable_metrics["condition_operation_count"] == 2
            and (
                observable_metrics["curriculum_topic_count"] >= 2
                or observable_metrics["rule_family_count"] >= 2
            )
            and observable_metrics["representation_operation_count"] >= 1
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "结构边界窄校准：关联约束、知识关系与表征转换共同形成完整常规模型",
                rule="teacher_basic_to_medium_linked_application",
                evidence=[
                    "最长链步骤="
                    + str(observable_metrics["longest_chain_steps"]),
                    "条件操作数="
                    + str(observable_metrics["condition_operation_count"]),
                    "课题数="
                    + str(observable_metrics["curriculum_topic_count"]),
                    "表征操作数="
                    + str(
                        observable_metrics["representation_operation_count"]
                    ),
                ],
            )
        elif (
            raw_level == "中等题"
            and high_density_evidence_hard
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "结构边界窄校准：高密度回答规则与多证据联合形成综合分析链",
                rule="teacher_medium_to_hard_high_density_evidence",
                evidence=[high_density_evidence_hard],
            )
        elif (
            raw_level == "中等题"
            and coordinated_multigraph_reaction_signal(
                model_features, observable_metrics, data
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "严重低估安全底线：三幅以上关联图像共同描述同一反应进程",
                rule="teacher_medium_to_hard_coordinated_multigraph_floor",
                evidence=[
                    "图表操作=" + model_features["graph_table_operation"],
                    "表征操作数="
                    + str(
                        observable_metrics["representation_operation_count"]
                    ),
                    "条件操作数="
                    + str(observable_metrics["condition_operation_count"]),
                    "证据操作="
                    + "、".join(model_features["evidence_operations"]),
                ],
            )
        elif (
            raw_level == "中等题"
            and observable_metrics["longest_chain_steps"] >= 2
            and dense_project_experiment_signal(
                model_features, observable_metrics, data
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "严重低估安全底线：四个以上实验形成变量、证据和方案的项目链",
                rule="teacher_medium_to_hard_dense_project_floor",
                evidence=[
                    "最长链步骤="
                    + str(observable_metrics["longest_chain_steps"]),
                    "实验操作=" + model_features["experiment_operation"],
                    "条件操作数="
                    + str(observable_metrics["condition_operation_count"]),
                    "证据操作="
                    + "、".join(model_features["evidence_operations"]),
                ],
            )
        elif (
            raw_level == "中等题"
            and multi_activity_project_signal(
                model_features, observable_metrics, data
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "严重低估安全底线：多活动、多实验共同形成变量与证据项目链",
                rule="teacher_medium_to_hard_multi_activity_project_floor",
                evidence=[
                    "题面包含至少两项活动和三项实验",
                    "实验操作=" + model_features["experiment_operation"],
                    "条件操作数="
                    + str(observable_metrics["condition_operation_count"]),
                ],
            )
        elif (
            raw_level == "中等题"
            and observable_metrics["longest_chain_steps"] >= 2
            and strong_segment_graph_chain_signal(
                model_features, observable_metrics
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "结构边界窄校准：分段图像与跨模块表征、约束和实验证据共同建模",
                rule="teacher_medium_to_hard_strong_graph_chain",
                evidence=[
                    "最长链步骤="
                    + str(observable_metrics["longest_chain_steps"]),
                    "图表操作=" + model_features["graph_table_operation"],
                    "课程单元数="
                    + str(observable_metrics["curriculum_unit_count"]),
                    "表征操作数="
                    + str(
                        observable_metrics["representation_operation_count"]
                    ),
                    "条件操作数="
                    + str(observable_metrics["condition_operation_count"]),
                    "实验操作=" + model_features["experiment_operation"],
                ],
            )
        elif (
            raw_level == "拔高题"
            and observable_dense_multiquestion_final_signal(
                model_features,
                data,
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "压轴题",
                "结构边界窄校准：多问共享的高密度高级定量链达到压轴边界",
                rule=(
                    "teacher_hard_to_final_"
                    "dense_multiquestion_quantitative_chain"
                ),
                evidence=[
                    "显式小问数="
                    + str(
                        observable_metrics[
                            "explicit_subquestion_count"
                        ]
                    ),
                    "有效任务数="
                    + str(observable_metrics["effective_task_count"]),
                    "最长链="
                    + " → ".join(
                        model_features["longest_solution_chain"]
                    ),
                    "高级计算="
                    + "、".join(
                        model_features["calculation_operations"]
                    ),
                ],
            )
        elif (
            raw_level == "拔高题"
            and observable_double_source_multireaction_final_signal(
                model_features
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "压轴题",
                "结构边界窄校准：双来源交叉验证与多反应定量关系共同定解",
                rule=(
                    "teacher_hard_to_final_"
                    "double_source_multireaction"
                ),
                evidence=[
                    "解题拓扑="
                    + model_features["solution_topology"],
                    "计算操作="
                    + "、".join(
                        model_features["calculation_operations"]
                    ),
                ],
            )
        elif (
            raw_level == "拔高题"
            and observable_multistage_multiquestion_multireaction_final_signal(
                model_features,
                data,
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "压轴题",
                "结构边界窄校准：多阶段或双来源结构与多反应定量关系共同贯穿多问",
                rule=(
                    "teacher_hard_to_final_"
                    "multistage_multiquestion_multireaction"
                ),
                evidence=[
                    "显式小问数="
                    + str(
                        observable_metrics[
                            "explicit_subquestion_count"
                        ]
                    ),
                    "解题拓扑="
                    + model_features["solution_topology"],
                    "反应结构="
                    + model_features["reaction_structure"],
                    "计算操作="
                    + "、".join(
                        model_features["calculation_operations"]
                    ),
                ],
            )
        elif (
            raw_level == "拔高题"
            and observable_strict_deep_quantitative_final_signal(
                model_features,
                data,
            )
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "压轴题",
                "结构边界严格校准：分支范围或组成不变量与高级定量操作形成交叉约束",
                rule=(
                    "teacher_hard_to_final_"
                    "strict_deep_quantitative_chain"
                ),
                evidence=[
                    "解题拓扑="
                    + model_features["solution_topology"],
                    "反应结构="
                    + model_features["reaction_structure"],
                    "条件操作="
                    + "、".join(
                        model_features["condition_operations"]
                    ),
                    "计算操作="
                    + "、".join(
                        model_features["calculation_operations"]
                    ),
                ],
            )
        # V5生产写回规则到此结束。

    teacher_candidate_actions = copy.deepcopy(
        teacher_candidate_result.get("postprocess_trace", [])
    )
    if len(teacher_candidate_actions) > 1:
        raise RuntimeError("教师分布窄校准违反每题单次调整约束")
    teacher_guard_action = (
        teacher_candidate_actions[0] if teacher_candidate_actions else None
    )
    teacher_guard_candidate_level = (
        teacher_candidate_result.get("difficulty_level", raw_level)
        if teacher_guard_action
        else raw_level
    )

    teacher_guard_writeback_applied = bool(
        CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
        and teacher_guard_action
        and not feature_quality_blocks_writeback
    )
    if teacher_guard_writeback_applied:
        rating_result = teacher_candidate_result
        sync_coarse_difficulty(rating_result)

    rating_result["coarse_difficulty_final"] = rating_result["coarse_difficulty"]
    rating_result["teacher_distribution_guard_candidate_level"] = (
        teacher_guard_candidate_level
    )
    rating_result["teacher_distribution_guard_candidate_action"] = (
        copy.deepcopy(teacher_guard_action) if teacher_guard_action else None
    )
    rating_result["teacher_distribution_guard_writeback_applied"] = (
        teacher_guard_writeback_applied
    )
    rating_result["teacher_distribution_guard_writeback_blocked_reason"] = (
        "特征存在兜底或证据不完整，禁止自动写回："
        + "、".join(feature_quality_flags)
        if teacher_guard_action and feature_quality_blocks_writeback
        else ""
    )

    sync_reasoning_after_postprocess(rating_result)
    rating_result["postprocess_actions"] = copy.deepcopy(
        rating_result.get("postprocess_trace", [])
    )
    if len(rating_result["postprocess_actions"]) > 1:
        raise RuntimeError("后处理违反每题单次调整约束")
    rating_result["automatic_level_change_applied"] = bool(
        rating_result["postprocess_actions"]
    )
    add_feature_audit_flags(rating_result, data)
    if teacher_guard_action and feature_quality_blocks_writeback:
        rating_result["feature_audit_flags"].append(
            "特征存在仅审计兜底或证据不完整，已阻止自动写回："
            + "、".join(feature_quality_flags)
        )
    elif (
        teacher_guard_action
        and not teacher_guard_writeback_applied
        and not CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
    ):
        rating_result["feature_audit_flags"].append(
            "存在结构边界窄校准候选，但专用写回关闭；仅记录候选动作"
        )
    rating_result["feature_audit_flags"] = list(
        dict.fromkeys(rating_result["feature_audit_flags"])
    )
    return rating_result


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


