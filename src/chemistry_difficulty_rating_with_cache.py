# -*- coding: utf-8 -*-
"""初中化学难度批量评级。

运行与审计方式对齐当前物理正式流程：OpenAI-compatible Responses API、
强制前缀缓存、并发、断点续跑、JSONL 输入输出，以及基于
“教材知识点覆盖 + 实际作答任务”的严格 schema。预测结果不保存教师
标签，也不执行基于分布或旧抽象特征的自动改档。
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
from typing import Dict, Any, Optional, List, Tuple, Sequence
from tqdm.asyncio import tqdm
from asyncio import Lock, Semaphore
from dotenv import load_dotenv
import junior_chemistry_schema as junior_schema

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

USE_CACHE = True
CHEMISTRY_IMAGE_MODE = os.getenv(
    "CHEMISTRY_IMAGE_MODE",
    "auto",
).strip().lower()
if CHEMISTRY_IMAGE_MODE not in {"off", "auto", "all"}:
    raise ValueError(
        f"不支持的 CHEMISTRY_IMAGE_MODE={CHEMISTRY_IMAGE_MODE!r}；"
        "可选值：off, auto, all"
    )

QUESTION_INPUT_FIELDS = (
    "parent_id", "question_id", "stem", "options", "analysis",
    "sub_questions", "stem_pic_url", "analysis_pic_url",
)
SUBQUESTION_INPUT_FIELDS = (
    "parent_id", "question_id", "stem", "options", "analysis",
    "stem_pic_url", "analysis_pic_url",
)
VISUAL_REFERENCE_RE = re.compile(
    r"(如图|图中|下图|图示|示意图|装置图|实验装置|流程图|"
    r"曲线|坐标图|关系图|图像|图象|表格|微观示意|粒子图|"
    r"看图|观察图|由图|据图|结合图)"
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
CACHE_FILE_PATH = os.getenv(
    "CHEMISTRY_CACHE_FILE_PATH", "chemistry_prompt_cache.json"
)
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
    """加载提示词并附加唯一的初中化学受控知识点目录。"""
    global DIFFICULTY_RATING_PROMPT_PREFIX, DIFFICULTY_RATING_PROMPT_SUFFIX

    if not os.path.exists(prompt_path):
        print(f"错误: 找不到提示词文件 {prompt_path}！")
        sys.exit(1)

    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    marker = "## 输入题目信息"
    if marker not in content:
        raise ValueError(f"提示词缺少分割标志: {marker}")
    prefix, _ = content.split(marker, 1)
    DIFFICULTY_RATING_PROMPT_PREFIX = (
        prefix.rstrip()
        + "\n\n## 受控教材知识点目录\n"
        + "只能选择实际作答需要的topic_id；不要输出别名、单元数或知识点数量。\n"
        + junior_schema.curriculum_catalog_text()
        + "\n\n"
        + marker
    )
    DIFFICULTY_RATING_PROMPT_SUFFIX = "\n\n请仅输出符合上述 JSON schema 的对象。"

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
        "signature_version": "chemistry-rating-run-v1",
        "input_sha256": compute_file_hash(input_path),
        "prompt_sha256": compute_file_hash(prompt_path),
        "script_sha256": compute_file_hash(Path(__file__).resolve()),
        "rating_profile": RATING_PROFILE,
        "model_name": MODEL_NAME,
        "temperature": TEMPERATURE,
        "image_mode": CHEMISTRY_IMAGE_MODE,
        "cache_enabled": USE_CACHE,
        "seed": seed,
        "num": num,
        "feature_schema_version": junior_schema.FEATURE_SCHEMA_VERSION,
        "structured_output_mode": "prompt_json_local_strict_schema",
        "postprocess_mode": "teacher_factor_boundary_review_writeback_v3",
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

async def create_prefix_cache(
    session: aiohttp.ClientSession,
    timeout_sec: int,
) -> Optional[str]:
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
    try:
        async with session.post(
            f"{BASE_URL}responses",
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=aiohttp.ClientTimeout(total=timeout_sec),
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                print(
                    f"创建前缀缓存失败且不重试 (状态码: {response.status}): "
                    f"{error_text[:200]}"
                )
                return None

            result = await response.json()
            response_id = result.get("id")
            if response_id:
                await set_cache(response_id, expire_at)
                print(
                    f"前缀缓存创建成功，耗时: {time.time() - t1:.2f}秒，"
                    f"缓存ID: {response_id}"
                )
                return response_id
    except Exception as e:
        print(f"创建前缀缓存异常且不重试: {e}")
    return None

async def get_or_create_cache(
    session: aiohttp.ClientSession,
    timeout_sec: int,
) -> Optional[str]:
    async with CACHE_GET_LOCK:
        cache_entry = await get_valid_cache()
        if cache_entry:
            return cache_entry["response_id"]
        print("未找到有效缓存，正在向服务器创建前缀缓存...")
        return await create_prefix_cache(session, timeout_sec)

# -------------------------- 3. 输入可见文本 --------------------------
def visible_text(data: Dict[str, Any], include_analysis: bool = False) -> str:
    """汇总题目可见文本，仅用于判断是否需要发送题图。"""
    fields = ["stem", "options", "sub_questions"]
    if include_analysis:
        fields.append("analysis")
    return json.dumps(
        {field: data.get(field) for field in fields if data.get(field)},
        ensure_ascii=False,
    )


# -------------------------- 4. 构建题目输入与模型调用 --------------------------
def make_output_base(data: Dict[str, Any]) -> Dict[str, Any]:
    """构造预测记录允许保存和发送的题目输入字段。"""
    output = {
        field: copy.deepcopy(data[field])
        for field in QUESTION_INPUT_FIELDS
        if field in data
    }
    sub_questions = data.get("sub_questions")
    if isinstance(sub_questions, list):
        output["sub_questions"] = [
            {
                field: copy.deepcopy(item[field])
                for field in SUBQUESTION_INPUT_FIELDS
                if field in item
            }
            for item in sub_questions
            if isinstance(item, dict)
        ]
    return output


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
        return list(available), ["all模式：发送全部可用图片"]

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
        urls = extract_image_urls(data.get(field))
        if urls:
            content.append({"type": "input_text", "text": labels[field]})
        for url in urls:
            if url in seen:
                continue
            content.append({"type": "input_image", "image_url": url})
            seen.add(url)
    return content


def extract_rating_from_response(
    result: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, str]:
    """读取完整JSON；只恢复完整对象中的确定性引号或前缀问题。"""
    output_text = ""
    for item in result.get("output", []):
        if (
            item.get("type") == "function_call"
            and item.get("name") == junior_schema.TOOL_NAME
        ):
            arguments = item.get("arguments", "")
            if isinstance(arguments, dict):
                return arguments, json.dumps(arguments, ensure_ascii=False), "strict"
            if isinstance(arguments, str):
                parsed, mode = junior_schema.parse_model_json_text(arguments)
                return parsed, arguments, mode
        if item.get("type") == "message":
            for content_item in item.get("content", []):
                if content_item.get("type") == "output_text":
                    output_text = content_item.get("text", "")
    parsed, mode = junior_schema.parse_model_json_text(output_text)
    return parsed, output_text, mode


async def call_model_with_cache(
    data: Dict[str, Any],
    session: aiohttp.ClientSession,
    timeout_sec: int,
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
        for url in extract_image_urls(data.get(field))
    ]
    image_status = {
        "question_input_mode": f"text_first_image_{CHEMISTRY_IMAGE_MODE}",
        "question_text_input_used": True,
        "structured_text_char_count": len(construct_question_content(data)),
        "image_mode": CHEMISTRY_IMAGE_MODE,
        "image_input_requested": bool(selected_fields),
        "image_input_used": False,
        "image_input_fields": selected_fields,
        "image_input_url_count": len(dict.fromkeys(selected_urls)),
        "image_selection_reasons": selection_reasons,
        "http_retry_count": 0,
        "model_request_count": 0,
        "response_status": "",
        "response_incomplete_reason": "",
        "response_output_item_statuses": [],
        "structured_output_json_complete": False,
        "json_parse_mode": "failed",
        "token_usage_consistent": True,
        "token_anomaly_flags": [],
    }

    response_id: Optional[str] = None
    if USE_CACHE:
        response_id = await get_or_create_cache(session, timeout_sec)
        if not response_id:
            print("警告: 无法获取有效缓存 ID，终止单题请求")
            return {}, "", 0.0, 0, 0, 0, image_status

    # 每道题最多发送一次模型请求；HTTP/API失败也不在题目内部重试。
    for _single_http_attempt in range(1):
        image_status["http_retry_count"] = 0
        user_content = build_user_content(data, selected_fields)
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
            image_status["model_request_count"] += 1
            async with session.post(
                f"{BASE_URL}responses",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    parsed_result, output_text, parse_mode = extract_rating_from_response(result)
                    usage = result.get("usage", {})
                    prompt_tokens = usage.get("input_tokens", 0)
                    completion_tokens = usage.get("output_tokens", 0)
                    total_tokens = usage.get("total_tokens", 0)
                    image_status["response_status"] = str(
                        result.get("status", "") or ""
                    )
                    incomplete_details = result.get("incomplete_details")
                    if isinstance(incomplete_details, dict):
                        image_status["response_incomplete_reason"] = str(
                            incomplete_details.get("reason", "") or ""
                        )
                    image_status["response_output_item_statuses"] = [
                        str(item.get("status", "") or "")
                        for item in result.get("output", [])
                        if isinstance(item, dict)
                    ]
                    output_is_complete = bool(parsed_result)
                    image_status["structured_output_json_complete"] = (
                        output_is_complete
                    )
                    image_status["json_parse_mode"] = parse_mode
                    image_status["token_usage_consistent"] = (
                        total_tokens == prompt_tokens + completion_tokens
                    )
                    anomaly_flags = []
                    if image_status["model_request_count"] != 1:
                        anomaly_flags.append("model_request_count_not_one")
                    if image_status["response_status"] == "incomplete":
                        anomaly_flags.append("response_incomplete")
                    if not output_is_complete:
                        anomaly_flags.append("structured_output_json_incomplete")
                    if not image_status["token_usage_consistent"]:
                        anomaly_flags.append("token_usage_sum_mismatch")
                    image_status["token_anomaly_flags"] = anomaly_flags
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
                    print("接口限流(429)，按单题不重试原则写入错误文件")
                    return {}, "", 0.0, 0, 0, 0, image_status

                error_text = await response.text()
                print(f"API请求失败 (状态码: {response.status}): {error_text[:200]}")
                if USE_CACHE and "InvalidParameter.PreviousResponseNotFound" in error_text:
                    print("检测到服务器缓存丢失；按单题不重试原则写入错误文件")
                    return {}, "", 0.0, 0, 0, 0, image_status
                if response.status >= 500:
                    print(f"服务器故障({response.status})，按单题不重试原则写入错误文件")
                    return {}, "", 0.0, 0, 0, 0, image_status
                if 400 <= response.status < 500:
                    return {}, "", 0.0, 0, 0, 0, image_status
        except aiohttp.ClientError as e:
            print(f"网络异常，按单题不重试原则写入错误文件: {e}")
            return {}, "", 0.0, 0, 0, 0, image_status
        except Exception as e:
            print(f"运行过程中请求异常: {e}")
            return {}, "", 0.0, 0, 0, 0, image_status

    return {}, "", 0.0, 0, 0, 0, image_status

# -------------------------- 5. 并发处理 --------------------------
async def process_single_question(
    data: Dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    output_path: str,
    error_path: str,
    timeout_sec: int,
) -> str:
    async with semaphore:
        question_id = data.get("question_id", "unknown")
        question_input = make_output_base(data)
        total_time = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        schema_errors: List[str] = []
        raw_result: Dict[str, Any] = {}
        raw_text = ""
        image_status: Dict[str, Any] = {}

        # 每道题只调用模型一次；schema失败直接写入错误文件，不自动重试。
        for _single_schema_attempt in range(1):
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
                    question_input,
                    session,
                    timeout_sec,
                )
                raw_result = copy.deepcopy(candidate)
                total_time += time_use
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += call_tokens

                try:
                    if not candidate:
                        raise junior_schema.ChemistrySchemaError(
                            "模型返回空对象或JSON解析失败"
                        )
                    rating_result = junior_schema.postprocess_chemistry_difficulty(
                        candidate,
                        question_input,
                    )
                except junior_schema.ChemistrySchemaError as exc:
                    schema_errors.append(str(exc))
                    raise RuntimeError(
                        f"schema校验失败（未自动重试）: {exc}"
                    ) from exc

                output_data = copy.deepcopy(question_input)
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
                output_data["feature_normalization_actions"] = copy.deepcopy(
                    rating_result.get("postprocess", {}).get(
                        "feature_normalization_actions",
                        [],
                    )
                )
                output_data["difficulty_rating"] = rating_result
                output_data["api_time_use"] = round(total_time, 2)
                output_data["api_prompt_tokens"] = total_prompt_tokens
                output_data["api_completion_tokens"] = total_completion_tokens
                output_data["api_total_tokens"] = total_tokens
                output_data.update(image_status)
                output_data["schema_validation_errors"] = schema_errors
                output_data["model_input_audit"] = {
                    "structured_text_primary": True,
                    "image_mode": CHEMISTRY_IMAGE_MODE,
                    "selected_image_fields": image_status.get(
                        "image_input_fields",
                        [],
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
                return "success"
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
                error_data["schema_validation_errors"] = schema_errors
                async with FILE_LOCK:
                    async with aiofiles.open(
                        error_path,
                        "a",
                        encoding="utf-8",
                    ) as f:
                        await f.write(
                            json.dumps(error_data, ensure_ascii=False) + "\n"
                        )
                if image_status.get("response_status"):
                    return "model_validation_error"
                return "request_error"


async def process_with_progress(
    data: Dict[str, Any],
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    pbar: tqdm,
    output_path: str,
    error_path: str,
    timeout_sec: int,
) -> str:
    status = await process_single_question(
        data,
        session,
        semaphore,
        output_path,
        error_path,
        timeout_sec,
    )
    pbar.update(1)
    return status


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

# -------------------------- 6. 主执行流 --------------------------
async def main_batch_run() -> None:
    global CURRENT_RUN_SIGNATURE, CURRENT_RUN_CONFIG
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
    parser.add_argument("-n", "--num", type=int, default=None, help="测试打标的限制数量（留空表示全部打标）")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样/打乱的种子，默认 42")
    args = parser.parse_args()

    random.seed(args.seed)
    load_prompt_config(args.prompt)
    print(f"评级配置: {RATING_PROFILE}")
    print(f"模型: {MODEL_NAME}")
    print(
        "temperature: "
        + ("服务端默认" if TEMPERATURE is None else str(TEMPERATURE))
    )
    print(f"图片模式: {CHEMISTRY_IMAGE_MODE}")
    print("前缀缓存: 强制启用（正式运行不允许逐题重复发送完整提示词）")

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
        response_id = await get_or_create_cache(session, args.timeout)
        if not response_id:
            raise RuntimeError("前缀缓存创建失败，批量任务未启动")

        print("正在用第一道真实题验证前缀缓存与JSON输出兼容性...")
        preflight_status = await process_single_question(
            to_process[0],
            session,
            semaphore,
            args.output,
            args.error,
            args.timeout,
        )
        pbar.update(1)
        if preflight_status == "request_error":
            raise RuntimeError(
                "启动验证失败，仅处理1题并已写入错误文件；"
                "为避免批量无效请求，本次运行已终止"
            )
        if preflight_status == "model_validation_error":
            print(
                "启动验证确认API与缓存兼容；第一题未通过本地Schema，"
                "已写入错误文件且不重试，继续处理剩余题目。"
            )
        else:
            print("启动验证通过，开始处理剩余题目。")
        tasks = [
            asyncio.create_task(
                process_with_progress(
                    q,
                    session,
                    semaphore,
                    pbar,
                    args.output,
                    args.error,
                    args.timeout,
                )
            )
            for q in to_process[1:]
        ]
        if tasks:
            await asyncio.gather(*tasks)

    pbar.close()
    print("\n✨ 化学多线程批量打标运行结束！")
    print(f"👉 成功保存打标结果至: {os.path.abspath(args.output)}")
    print(f"👉 单次请求失败日志在: {os.path.abspath(args.error)}")


if __name__ == "__main__":
    start_time = time.time()
    try:
        asyncio.run(main_batch_run())
    except KeyboardInterrupt:
        print("\n收到键盘中断信号，程序已安全退出。")
    except Exception as e:
        print(f"\n批量运行中遇到未捕获异常: {e}")
    print(f"本次打标运行耗时: {round((time.time() - start_time) / 60, 2)} 分钟。")
