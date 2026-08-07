# -*- coding: utf-8 -*-
"""初中化学难度批量评级。

运行与审计方式对齐当前物理正式流程：OpenAI-compatible Responses API、
可选前缀缓存、并发、重试、断点续跑、JSONL 输入输出、严格 Core-12
schema、原始/最终结果分离，以及可审计的窄后处理。常规校准每次最多
调整一个相邻档；只有基于题干客观结构的严重低估安全底线，才允许将
明显的连续反应核验从送分题直接托底到中等题。

化学使用历史效果更稳定的12个核心特征，不复用物理特征，也不把
Evidence-15的三个辅助观察量加入生产输出协议。
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
CHEMISTRY_ENABLE_LEVEL_WRITEBACK = os.getenv(
    "CHEMISTRY_ENABLE_LEVEL_WRITEBACK",
    "0",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD = os.getenv(
    "CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD",
    "0",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK = os.getenv(
    "CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK",
    "0",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
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
if (
    CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
    and (
        CHEMISTRY_ENABLE_LEVEL_WRITEBACK
        or CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK
    )
):
    raise ValueError(
        "教师分布窄校准写回不能与通用写回或压轴边界写回同时开启；"
        "请只选择一种写回策略"
    )
CHEMISTRY_IMAGE_MODE = os.getenv(
    "CHEMISTRY_IMAGE_MODE",
    "auto",
).strip().lower()
if CHEMISTRY_IMAGE_MODE not in {"off", "auto", "all"}:
    raise ValueError(
        f"不支持的 CHEMISTRY_IMAGE_MODE={CHEMISTRY_IMAGE_MODE!r}；"
        "可选值：off, auto, all"
    )
MAX_SCHEMA_RETRIES = int(os.getenv("CHEMISTRY_SCHEMA_RETRIES", "2"))

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
        "general_level_writeback_enabled": (
            CHEMISTRY_ENABLE_LEVEL_WRITEBACK
        ),
        "final_boundary_guard_enabled": (
            CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD
        ),
        "final_boundary_guard_writeback_enabled": (
            CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK
        ),
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
FEATURE_DEFAULTS = {
    "reasoning_depth": "0层",
    "reasoning_direction": "直接识记",
    "knowledge_relation": "单一知识点",
    "representation_conversion": "无",
    "reaction_relation": "无反应关系",
    "constraint_complexity": "无约束",
    "evidence_relation": "无证据任务",
    "experiment_requirement": "无",
    "graph_table_requirement": "无",
    "calculation_model": "无",
    "unfamiliar_information_transfer": "课内直接原型",
    "subquestion_dependency": "无多问",
}

ALLOWED_FEATURE_VALUES = {
    "reasoning_depth": {"0层", "1层", "2-3层", "4-5层", "6层及以上"},
    "reasoning_direction": {"直接识记", "正向推导", "逆向推导", "分类讨论或综合推导"},
    "knowledge_relation": {"单一知识点", "同模块简单关联", "同模块深度关联", "跨模块融合", "多模块深度融合"},
    "representation_conversion": {"无", "一次表征转换", "两类表征连续转换", "宏观-微观-符号-定量多重转换"},
    "reaction_relation": {
        "无反应关系",
        "单一直接反应",
        "2-3个并列或简单连续反应",
        "多反应连续转化",
        "先后、竞争或过量不足",
        "需要分情况判断的反应模型",
    },
    "constraint_complexity": {"无约束", "单一约束", "多个相互关联约束", "多层嵌套约束"},
    "evidence_relation": {
        "无证据任务",
        "单一证据直接对应",
        "多条清晰证据联合",
        "需要排除竞争解释",
        "证据冲突、筛选或多层排除",
    },
    "experiment_requirement": {
        "无",
        "基础操作或读数",
        "控制变量、现象解释或数据归纳",
        "方案设计、评价或补充实验",
        "多阶段探究与定量误差",
    },
    "graph_table_requirement": {"无", "直接读数", "多组比较归纳", "拐点、平台或分段反推", "多图表耦合建模"},
    "calculation_model": {"无", "口算或直接比例", "单一方程式或关系式", "单一守恒或多反应计算", "多重守恒、差量、联立或分类"},
    "unfamiliar_information_transfer": {"课内直接原型", "给定新信息直接应用", "迁移后建立关系", "完全陌生模型现场建立"},
    "subquestion_dependency": {"无多问", "多问相互独立", "多问共享模型但无答案依赖", "多问存在结果或任务链依赖"},
}

ENUM_NORMALIZE = {
    "representation_conversion": {
        "两类表征往返": "两类表征连续转换",
    },
    "reaction_relation": {
        "单一反应": "单一直接反应",
        "先后或竞争反应": "先后、竞争或过量不足",
    },
    "constraint_complexity": {
        "多约束": "多个相互关联约束",
    },
    "evidence_relation": {
        "无证据链": "无证据任务",
        "单一现象对应": "单一证据直接对应",
        "多现象证据链": "多条清晰证据联合",
        "证据冲突与排除": "证据冲突、筛选或多层排除",
    },
    "experiment_requirement": {
        "控制变量或现象分析": "控制变量、现象解释或数据归纳",
        "方案设计或误差评价": "方案设计、评价或补充实验",
    },
    "graph_table_requirement": {
        "图像反推或拐点分析": "拐点、平台或分段反推",
    },
    "calculation_model": {
        "多重守恒差量联立或分类": "多重守恒、差量、联立或分类",
    },
    "unfamiliar_information_transfer": {
        "无": "课内直接原型",
        "课内原型": "课内直接原型",
        "给定信息直接套用": "给定新信息直接应用",
        "迁移后推导": "迁移后建立关系",
    },
    "subquestion_dependency": {
        "多问但相互独立": "多问相互独立",
        "多问且层层递进": "多问存在结果或任务链依赖",
    },
}

def clean_enum_value(value: Any) -> str:
    if value is None:
        return ""
    v = str(value).strip()
    v = (
        v.replace("，", ",")
        .replace("、", ",")
        .replace("；", ";")
        .replace("：", ":")
        .replace("（", "(")
        .replace("）", ")")
        .replace(" ", "")
        .replace("\n", "")
        .replace("\t", "")
    )
    return v.strip('";,.:。')


def canonicalize_feature_value(field: str, value: Any) -> str:
    v = clean_enum_value(value)
    if not v:
        return FEATURE_DEFAULTS[field]

    if field == "step_count":
        if "12" in v or "十二" in v:
            return "12步以上"
        if any(k in v for k in ["9-12", "9到12", "九", "十", "11"]):
            return "9-12步"
        if any(k in v for k in ["6-8", "6到8", "六", "七", "八"]):
            return "6-8步"
        if any(k in v for k in ["3-5", "3到5", "三", "四", "五"]):
            return "3-5步"
        return "1-2步"

    if field == "equation_count":
        if "7" in v or "七" in v:
            return "7个以上"
        if any(k in v for k in ["4-6", "4到6", "四", "五", "六", "4个以上"]):
            return "4-6个"
        if any(k in v for k in ["2-3", "2到3", "2个", "3个", "两", "二", "三"]):
            return "2-3个"
        return "0-1个"

    if field == "calculation_complexity":
        if any(k in v for k in ["复杂", "守恒", "图像", "图象", "拐点", "混合物", "差量", "极值", "范围", "分类", "多变量"]):
            return "复杂守恒或图像计算"
        if any(k in v for k in ["方程式", "关系式", "质量守恒", "根据化学方程式"]):
            return "化学方程式计算或关系式计算"
        if any(k in v for k in ["简单", "笔算", "化合价", "相对分子", "质量分数", "代入"]):
            return "简单笔算"
        return "口算或直接判断"

    if field == "reasoning_chain":
        if any(k in v for k in ["逆向", "反推", "方案", "评价", "排除", "冲突", "干扰", "拐点", "先后"]):
            return "逆向推理或方案评价"
        if any(k in v for k in ["多层", "证据", "多步", "链条", "综合", "归纳"]):
            return "多层证据推理"
        if any(k in v for k in ["简单", "因果", "对应"]):
            return "简单因果推理"
        return "直接套用"

    if field == "problem_structure":
        if any(k in v for k in ["跨模块", "综合"]):
            # 若明确是实验/流程/计算综合，优先保留具体结构
            if any(k in v for k in ["流程", "工艺"]):
                return "工艺流程"
            if any(k in v for k in ["实验", "探究"]):
                return "实验探究"
            if any(k in v for k in ["计算", "守恒"]):
                return "计算综合"
            return "跨模块综合"
        if any(k in v for k in ["流程", "工艺", "制备"]):
            return "工艺流程"
        if any(k in v for k in ["图像", "图象", "表格", "曲线"]):
            return "图像表格分析"
        if any(k in v for k in ["推断", "鉴别", "除杂", "共存", "变质", "成分"]):
            return "物质推断"
        if any(k in v for k in ["计算", "守恒", "质量分数", "溶质质量分数", "关系式"]):
            return "计算综合"
        if any(k in v for k in ["实验探究", "猜想", "评价", "反思", "方案"]):
            return "实验探究"
        if any(k in v for k in ["实验", "操作", "仪器", "过滤", "蒸馏"]):
            return "实验基础操作"
        if any(k in v for k in ["方程式", "配平"]):
            return "方程式书写"
        if any(k in v for k in ["化学式", "化学用语", "分类", "化合价", "元素", "微粒", "离子"]):
            return "化学用语与分类"
        return "概念判断"

    if field == "additional_structure":
        if any(k in v for k in ["多模块", "跨模块"]):
            return "多模块综合"
        if any(k in v for k in ["流程", "工艺"]):
            return "流程图"
        if any(k in v for k in ["实验装置", "装置", "仪器"]):
            return "实验装置"
        if any(k in v for k in ["图像", "图象", "表格", "曲线"]):
            return "图像表格"
        if any(k in v for k in ["探究", "项目", "材料", "猜想"]):
            return "探究材料"
        if any(k in v for k in ["微观", "粒子", "结构示意图"]):
            return "微观示意图"
        return "无"

    if field == "information_carrier":
        has_flow = any(k in v for k in ["流程", "工艺"])
        has_exp = any(k in v for k in ["实验装置", "装置图", "实验图"])
        has_micro = any(k in v for k in ["微观", "粒子", "结构示意图"])
        has_graph = any(k in v for k in ["图像", "图象", "曲线", "图"])
        has_table = any(k in v for k in ["表格", "表"])
        if sum([has_flow, has_exp, has_micro, has_graph or has_table]) >= 2:
            return "多图表综合"
        if has_flow:
            return "流程图"
        if has_exp:
            return "实验装置图"
        if has_micro:
            return "微观示意图"
        if has_graph or has_table:
            return "图像或表格"
        if "单图" in v:
            return "单图识别"
        return "纯文字"

    if field == "reality_question":
        if v.lower() in ["true", "yes", "y", "1"] or "是" in v:
            return "是"
        return "否"

    if field == "subquestion_dependency":
        if any(k in v for k in ["层层", "递进", "依赖", "承接"]):
            return "多问且层层递进"
        if any(k in v for k in ["多问", "小题", "独立"]):
            return "多问但相互独立"
        return "无多问"

    if field == "knowledge_count":
        if any(k in v for k in ["4个及以上", "4个以上", "四个", "多个", "多知识点"]):
            return "4个及以上"
        if any(k in v for k in ["2-3", "2到3", "2个", "3个", "两", "二", "三"]):
            return "2-3个"
        if any(k in v for k in ["1个", "一个", "单一"]):
            return "1个"
        return "2-3个"

    if field == "knowledge_diff":
        if any(k in v for k in ["高", "难", "复杂"]):
            return "高"
        if any(k in v for k in ["中", "一般"]):
            return "中"
        return "低"

    if field == "cross_module":
        if "跨" in v or "综合" in v:
            return "跨模块综合"
        return "同一模块内部"

    if field == "chemistry_process_count":
        if any(k in v for k in ["多反应", "连续", "流程", "多阶段", "多步转化", "先后反应"]):
            return "多反应连续转化或流程"
        if any(k in v for k in ["2-3", "2到3", "两个", "三个", "若干", "多过程"]):
            return "2-3个反应或过程"
        if any(k in v for k in ["单一反应", "一个反应", "方程式"]):
            return "单一反应"
        return "单一事实"

    if field == "constraint_count":
        if any(k in v for k in ["多", "多个", "过量", "不足", "先后", "共同约束"]):
            return "多约束"
        if any(k in v for k in ["单", "一个", "有约束", "约束"]):
            return "单一约束"
        return "无约束"

    if field == "evidence_relation":
        if any(k in v for k in ["冲突", "排除", "干扰", "质疑", "反证"]):
            return "证据冲突与排除"
        if any(k in v for k in ["多现象", "多证据", "证据链", "多个现象", "综合现象"]):
            return "多现象证据链"
        if any(k in v for k in ["单一现象", "现象对应", "直接对应"]):
            return "单一现象对应"
        return "无证据链"

    if field == "experiment_requirement":
        if any(k in v for k in ["方案", "设计", "误差", "评价", "反思", "补充实验", "改进", "可靠性"]):
            return "方案设计或误差评价"
        if any(k in v for k in ["控制变量", "现象分析", "对照", "故障", "数据归纳", "探究", "分析"]):
            return "控制变量或现象分析"
        if any(k in v for k in ["读数", "操作", "仪器", "过滤", "蒸馏", "检验"]):
            return "基础操作或读数"
        return "无"

    if field == "graph_table_requirement":
        if any(k in v for k in ["反推", "拐点", "平台", "外推", "曲线关系", "图像分析", "图象分析"]):
            return "图像反推或拐点分析"
        if any(k in v for k in ["多组", "比较", "归纳", "趋势"]):
            return "多组比较归纳"
        if any(k in v for k in ["读数", "读取", "直接"]):
            return "直接读数"
        return "无"

    if field == "error_risk":
        if "高" in v:
            return "高易错点"
        if any(k in v for k in ["明显", "较大", "易错"]):
            return "明显易错点"
        if any(k in v for k in ["轻微", "较小"]):
            return "轻微易错点"
        return "无明显易错点"

    return FEATURE_DEFAULTS[field]


def normalize_feature_keys(features: Dict[str, Any]) -> Dict[str, Any]:
    """只接受 Core-12 正式字段名。

    旧版曾通过“字段名包含标准字段名”进行模糊归一化，这会把拼错字段
    静默伪装成合法字段，进而污染后处理。生产契约改为严格键名。
    """
    fixed: Dict[str, Any] = {}
    for k, v in (features or {}).items():
        clean_key = str(k).strip().strip('",， \n\t')
        fixed[clean_key] = v
    return fixed


def normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """兼容旧调用名，但执行与生产相同的严格 Core-12 校验。

    不再为缺失字段填默认值，也不依据关键词猜测枚举。
    """
    return validate_feature_contract(features)


class ChemistrySchemaError(ValueError):
    """模型输出不满足 Core-12 生产契约。"""


def validate_feature_contract(features: Any) -> Dict[str, str]:
    """严格校验 Core-12 字段，禁止缺字段后静默填默认值。

    只接受合法枚举和明确维护的历史别名；不使用模糊关键词把任意文本
    猜成某个枚举，因为这会让错误 feature 静默进入后处理。
    """
    if not isinstance(features, dict):
        raise ChemistrySchemaError("features必须是JSON对象")
    keyed = normalize_feature_keys(features)
    expected = set(FEATURE_DEFAULTS)
    actual = set(keyed)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ChemistrySchemaError(
            f"features字段不完整: missing={missing}, extra={extra}"
        )

    validated: Dict[str, str] = {}
    for field in FEATURE_DEFAULTS:
        raw_value = keyed[field]
        value = str(raw_value).strip()
        if value in ALLOWED_FEATURE_VALUES[field]:
            validated[field] = value
            continue
        alias = ENUM_NORMALIZE.get(field, {}).get(value)
        if alias in ALLOWED_FEATURE_VALUES[field]:
            validated[field] = alias
            continue
        clean_value = clean_enum_value(raw_value)
        alias = ENUM_NORMALIZE.get(field, {}).get(clean_value)
        if alias in ALLOWED_FEATURE_VALUES[field]:
            validated[field] = alias
            continue
        raise ChemistrySchemaError(
            f"features.{field}非法值{raw_value!r}；"
            f"允许值={sorted(ALLOWED_FEATURE_VALUES[field])}"
        )
    depth = validated["reasoning_depth"]
    direction = validated["reasoning_direction"]
    if depth == "0层" and direction != "直接识记":
        raise ChemistrySchemaError(
            "reasoning_depth=0层时reasoning_direction必须为直接识记"
        )
    if direction == "直接识记" and depth not in {"0层", "1层"}:
        raise ChemistrySchemaError(
            "直接识记不能对应2层以上连续推理"
        )
    if (
        validated["subquestion_dependency"]
        == "多问存在结果或任务链依赖"
        and depth in {"0层", "1层"}
    ):
        raise ChemistrySchemaError(
            "真实任务链依赖不能与0层/1层推理同时出现"
        )
    return validated


def validate_rating_contract(rating_result: Any) -> Dict[str, Any]:
    """校验固定顶层、Core-12特征、理由和相邻粗区间。"""
    if not isinstance(rating_result, dict):
        raise ChemistrySchemaError("模型输出必须是JSON对象")
    required = {
        "features",
        "coarse_difficulty",
        "reasoning",
        "difficulty_level",
    }
    missing = sorted(required - set(rating_result))
    if missing:
        raise ChemistrySchemaError(f"顶层字段缺失: {missing}")

    level = str(rating_result.get("difficulty_level", "")).strip()
    if level not in VALID_LEVELS:
        raise ChemistrySchemaError(f"difficulty_level非法: {level!r}")
    coarse = str(rating_result.get("coarse_difficulty", "")).strip()
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

    reasoning = rating_result.get("reasoning")
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

    prepared = copy.deepcopy(rating_result)
    prepared["features"] = validate_feature_contract(prepared["features"])
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
    rating_result["reasoning"] = normalized
    rating_result.pop("reason", None)


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


def coordinated_multigraph_reaction_signal(
    features: Dict[str, Any],
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
        and features["graph_table_requirement"]
        == "拐点、平台或分段反推"
        and features["knowledge_relation"]
        in {"跨模块融合", "多模块深度融合"}
        and features["representation_conversion"]
        in {
            "两类表征连续转换",
            "宏观-微观-符号-定量多重转换",
        }
        and features["constraint_complexity"]
        in {"多个相互关联约束", "多层嵌套约束"}
        and features["evidence_relation"]
        in {
            "多条清晰证据联合",
            "需要排除竞争解释",
            "证据冲突、筛选或多层排除",
        }
    )


def cross_module_knowledge_breadth_signal(
    features: Dict[str, Any],
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
        and features["reasoning_depth"] in {"0层", "1层"}
        and features["knowledge_relation"]
        in {"跨模块融合", "多模块深度融合"}
        and features["representation_conversion"] == "无"
        and features["experiment_requirement"] == "无"
        and features["graph_table_requirement"] == "无"
        and features["calculation_model"] == "无"
        and features["subquestion_dependency"] == "多问相互独立"
    )


def controllable_gas_scheme_signal(
    features: Dict[str, Any],
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
        and features["constraint_complexity"]
        in {"多个相互关联约束", "多层嵌套约束"}
        and features["experiment_requirement"] != "无"
    )


def multi_activity_project_signal(
    features: Dict[str, Any],
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
        and features["constraint_complexity"]
        in {"多个相互关联约束", "多层嵌套约束"}
        and features["experiment_requirement"]
        in {
            "控制变量、现象解释或数据归纳",
            "方案设计、评价或补充实验",
            "多阶段探究与定量误差",
        }
    )


def dense_project_experiment_signal(
    features: Dict[str, Any],
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
        and features["constraint_complexity"]
        in {"多个相互关联约束", "多层嵌套约束"}
        and features["evidence_relation"]
        in {
            "多条清晰证据联合",
            "需要排除竞争解释",
            "证据冲突、筛选或多层排除",
        }
        and features["experiment_requirement"]
        in {
            "控制变量、现象解释或数据归纳",
            "方案设计、评价或补充实验",
            "多阶段探究与定量误差",
        }
    )


def strong_segment_graph_chain_signal(
    features: Dict[str, Any],
) -> bool:
    return bool(
        features["graph_table_requirement"] == "拐点、平台或分段反推"
        and features["knowledge_relation"]
        in {"跨模块融合", "多模块深度融合"}
        and features["representation_conversion"]
        in {
            "两类表征连续转换",
            "宏观-微观-符号-定量多重转换",
        }
        and features["constraint_complexity"]
        in {"多个相互关联约束", "多层嵌套约束"}
        and features["evidence_relation"]
        in {
            "多条清晰证据联合",
            "需要排除竞争解释",
            "证据冲突、筛选或多层排除",
        }
        and features["experiment_requirement"]
        in {
            "控制变量、现象解释或数据归纳",
            "方案设计、评价或补充实验",
            "多阶段探究与定量误差",
        }
    )


def shared_new_information_signal(
    features: Dict[str, Any],
) -> bool:
    return bool(
        features["unfamiliar_information_transfer"]
        == "给定新信息直接应用"
        and features["subquestion_dependency"]
        == "多问共享模型但无答案依赖"
        and features["knowledge_relation"]
        in {
            "同模块深度关联",
            "跨模块融合",
            "多模块深度融合",
        }
    )


def final_promotion_ceiling_reason(
    features: Dict[str, Any],
    data: Dict[str, Any],
) -> str:
    """返回不应把拔高题再升为压轴题的客观低密度结构。"""
    text = visible_text(data, include_analysis=False)
    if (
        features["calculation_model"] in {"无", "口算或直接比例"}
        and features["knowledge_relation"] != "多模块深度融合"
        and features["representation_conversion"]
        != "宏观-微观-符号-定量多重转换"
        and features["graph_table_requirement"] != "多图表耦合建模"
    ):
        return (
            "实验或证据任务缺少压轴级定量、四重表征或多图表耦合，"
            "保持拔高题上限"
        )

    if (
        len(text) < 180
        and features["subquestion_dependency"] == "多问相互独立"
        and features["experiment_requirement"] == "无"
        and features["graph_table_requirement"] == "无"
    ):
        return "短题中的独立常规定量任务未形成压轴级整体耦合"

    explicit_questions = set(
        re.findall(
            r"问题\s*[一二三四五六123456]",
            text,
        )
    )
    if (
        len(explicit_questions) >= 4
        and features["subquestion_dependency"]
        != "多问存在结果或任务链依赖"
    ):
        return "四个以上并列研究问题没有形成结果或任务链依赖"

    return ""


def count_fill_blanks(text: str) -> int:
    return len(re.findall(r"_{2,}|（\s*）|\(\s*\)", text))


def count_subquestions(data: Dict[str, Any]) -> int:
    subqs = data.get("sub_questions", []) or []
    if isinstance(subqs, list) and subqs:
        return len(subqs)
    text = str(data.get("stem", "") or "")
    return max(len(re.findall(r"\([一二三四五六七八九十0-9]+\)|（[一二三四五六七八九十0-9]+）", text)), 0)


LONG_CONTEXT_KEYWORDS = [
    "项目式", "任务一", "任务二", "探究", "猜想", "评价与反思", "提出问题", "实验探究", "实验验证",
    "工艺流程", "流程", "制备", "滤渣", "滤液", "循环", "定量", "滴加", "图像", "图象", "曲线",
    "pH", "溶解度曲线", "离子数目", "变质", "除杂", "鉴别", "推断", "方案设计", "误差分析",
]


def is_long_context_or_new_situation(data: Dict[str, Any]) -> bool:
    stem = str(data.get("stem", "") or "")
    if len(stem) > 260:
        return True
    if contains_any(stem, LONG_CONTEXT_KEYWORDS):
        return True
    if count_subquestions(data) >= 4:
        return True
    return False


def is_trivial_concept_question(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    text = visible_text(data, include_analysis=False)
    return (
        len(text) < 120
        and features.get("step_count") == "1-2步"
        and features.get("equation_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_diff") == "低"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )



def is_pure_direct_recall_set(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """低阶直接识记集合题。

    用于纠正“空气成分用途/食品安全/身体健康常识”这类多选项、多填空被误升为基础题的情况。
    注意：只在无实验、无计算、无图表、无方程式推导时生效。
    """
    text = visible_text(data, include_analysis=True)
    simple_no_process = (
        features.get("equation_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
        and features.get("chemistry_process_count") in ["单一事实", "单一反应"]
        and features.get("evidence_relation") in ["无证据链", "单一现象对应"]
        and features.get("constraint_count") in ["无约束", "单一约束"]
        and features.get("information_carrier") in ["纯文字", "单图识别", "微观示意图"]
    )
    # 只覆盖“极低阶固定集合直接匹配”，避免把普通多选项概念辨析误降为送分题。
    direct_recall_patterns = [
        "①氧气", "②氮气", "③二氧化碳", "④稀有气体", "从①氧气", "选择适当的物质填空",
        "化学与我们的身体健康息息相关", "食品安全", "霉变大米", "公共场所禁止吸烟", "甲醛", "二氧化硫漂白",
        "化学发展史", "化学发展简史", "发展简史", "化学史", "化学家", "贡献",
        "侯德榜", "屠呦呦", "徐光宪", "张青莲", "门捷列夫", "拉瓦锡", "道尔顿",
    ]
    hard_exclusion_keywords = [
        "化学方程式", "配平", "计算", "质量分数", "溶质质量分数", "实验探究", "方案", "流程", "图像", "图象",
        "滤渣", "滤液", "变质", "推断", "鉴别", "除杂", "金属活动性", "置换"
    ]
    air_component_direct = (
        "①氧气" in text and "②氮气" in text and "③二氧化碳" in text and "④稀有气体" in text
    )
    health_direct = (
        "化学与我们的身体健康息息相关" in text
        or "食品安全" in text
        or ("霉变" in text and "甲醛" in text and "二氧化硫" in text)
    )
    history_direct = (
        "化学发展史" in text
        or "化学发展简史" in text
        or "发展简史" in text
        or "化学史" in text
        or ("化学家" in text and "贡献" in text)
        or contains_any(text, ["侯德榜", "屠呦呦", "徐光宪", "张青莲", "门捷列夫", "拉瓦锡", "道尔顿"])
    )
    if not simple_no_process:
        return False
    if history_direct:
        # 化学史题的解析可能出现“工艺流程/制碱工艺”等词，但它们只是人物贡献表述，不代表题目需要流程分析。
        return True
    return (air_component_direct or health_direct) and not contains_any(text, hard_exclusion_keywords)


def is_low_level_basic_application(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """基础题保护：多个独立基础空、一个方程式、化学式/质量守恒直接应用，不自动升中等。"""
    text = visible_text(data, include_analysis=True)
    if contains_any(text, [
        "滤渣", "滤液", "先后反应", "过量", "不足", "拐点", "平台", "离子数目", "压强变化图", "曲线",
        "方案评价", "质疑", "可靠性", "补充实验", "控制变量", "图像反推", "关系式法", "差量法", "元素守恒",
        "生成等量氢气", "相同质量", "不同金属", "金属用量", "制取氢气", "尾气处理", "节约能源", "炼铁"
    ]):
        return False

    return (
        features.get("step_count") in ["1-2步", "3-5步"]
        and features.get("equation_count") in ["0-1个", "2-3个"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("reasoning_chain") in ["直接套用", "简单因果推理"]
        and features.get("chemistry_process_count") in ["单一事实", "单一反应", "2-3个反应或过程"]
        and features.get("constraint_count") in ["无约束", "单一约束"]
        and features.get("evidence_relation") in ["无证据链", "单一现象对应"]
        and features.get("experiment_requirement") in ["无", "基础操作或读数"]
        and features.get("graph_table_requirement") in ["无", "直接读数"]
        and features.get("subquestion_dependency") != "多问且层层递进"
        and features.get("information_carrier") not in ["流程图", "多图表综合"]
    )


def is_standard_experiment_basic(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """标准实验基础题：气体制取、收集、验满、仪器、蜡烛/氧气性质等常规操作。"""
    text = visible_text(data, include_analysis=True)
    standard_keywords = [
        "气体制取", "制取氧气", "制取二氧化碳", "制取氢气", "发生装置", "收集装置", "验满", "检验",
        "试管", "长颈漏斗", "集气瓶", "排水法", "向上排空气", "向下排空气", "蜡烛燃烧", "氧气性质",
        "硫燃烧", "铁丝燃烧", "木炭燃烧", "过滤", "蒸馏", "玻璃棒", "水的净化"
    ]
    hard_exclusion_keywords = [
        "方案评价", "误差", "质疑", "补充实验", "可靠性", "控制变量", "图像", "图象", "曲线", "表格", "滤渣", "滤液",
        "变质", "混合物", "质量分数", "守恒", "关系式", "过量", "不足", "先后反应", "金属活动性", "尾气处理", "炼铁", "氧气含量", "气球", "压强", "制取氢气", "锌粒", "稀硫酸", "多孔隔板"
    ]
    return (
        contains_any(text, standard_keywords)
        and not contains_any(text, hard_exclusion_keywords)
        and features.get("step_count") in ["1-2步", "3-5步", "6-8步"]
        and features.get("equation_count") in ["0-1个", "2-3个"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("experiment_requirement") in ["基础操作或读数", "无"]
        and features.get("graph_table_requirement") in ["无", "直接读数"]
        and features.get("evidence_relation") in ["无证据链", "单一现象对应"]
        and features.get("subquestion_dependency") != "多问且层层递进"
    )


def is_standard_experiment_medium_combo(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """标准实验组合题：多个重要实验并列考查，有装置、方程式、现象/压强等综合，但无拔高卡点。"""
    text = visible_text(data, include_analysis=True)
    return (
        features.get("information_carrier") in ["实验装置图", "多图表综合"]
        and count_subquestions(data) >= 4
        and features.get("equation_count") in ["0-1个", "2-3个"]
        and features.get("experiment_requirement") in ["基础操作或读数", "控制变量或现象分析"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and contains_any(text, ["装置", "实验", "化学方程式", "气球", "压强", "尾气处理", "炼铁", "氧气含量"])
        and not contains_any(text, ["方案评价", "质疑", "补充实验", "复杂守恒", "图像反推", "拐点", "滤渣", "滤液"])
    )


def is_long_reading_direct_info(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """长阅读材料但只做信息定位/常识填空，不能因材料长自动判中等。"""
    text = visible_text(data, include_analysis=True)
    return (
        len(str(data.get("stem", "") or "")) > 220
        and features.get("step_count") in ["1-2步", "3-5步"]
        and features.get("equation_count") == "0-1个"
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("experiment_requirement") in ["无", "基础操作或读数"]
        and features.get("graph_table_requirement") in ["无", "直接读数"]
        and features.get("evidence_relation") in ["无证据链", "单一现象对应"]
        and features.get("chemistry_process_count") in ["单一事实", "单一反应", "2-3个反应或过程"]
        and not contains_any(text, ["方案评价", "补充实验", "质疑", "滤渣", "滤液", "拐点", "平台", "定量计算", "守恒", "混合物"])
    )


def is_single_path_standard_calculation(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """单线标准计算：沉淀/气体质量反推、纯度计算等，模型单一时最高多为中等。"""
    text = visible_text(data, include_analysis=True)
    return (
        features.get("problem_structure") == "计算综合"
        and features.get("equation_count") in ["0-1个", "2-3个"]
        and features.get("chemistry_process_count") in ["单一反应", "2-3个反应或过程"]
        and features.get("evidence_relation") in ["无证据链", "单一现象对应"]
        and features.get("experiment_requirement") in ["无", "基础操作或读数"]
        and features.get("graph_table_requirement") in ["无", "直接读数"]
        and not contains_any(text, ["拐点", "平台", "曲线", "图像", "图象", "多种", "不可能", "极值", "范围", "分类讨论"])
    )


def is_multi_standard_lab_independent_basic(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """多个标准实验操作并列题：加热、过滤、气体制取、验满等独立考查，通常为基础题。"""
    text = visible_text(data, include_analysis=True)
    standard_hits = 0
    for group in [
        ["加热液体", "给液体加热"],
        ["过滤", "滤纸", "漏斗", "玻璃棒"],
        ["制取氧气", "氧气的制取", "实验室制氧"],
        ["制取二氧化碳", "二氧化碳的制取", "实验室制取CO2", "实验室制取二氧化碳"],
        ["验满", "检验", "收集装置", "发生装置"],
    ]:
        if contains_any(text, group):
            standard_hits += 1
    hard_exclusion_keywords = [
        "控制变量", "对照实验", "方案", "方案评价", "误差", "质疑", "可靠性", "补充实验", "改进",
        "压强", "曲线", "图像", "图象", "表格", "质量分数", "纯度", "守恒", "关系式", "差量",
        "滤渣", "滤液", "金属活动性", "过量", "不足", "先后反应", "尾气处理", "炼铁", "产率"
    ]
    return (
        standard_hits >= 2
        and not contains_any(text, hard_exclusion_keywords)
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("experiment_requirement") in ["无", "基础操作或读数", "控制变量或现象分析"]
        and features.get("graph_table_requirement") in ["无", "直接读数"]
        and features.get("evidence_relation") in ["无证据链", "单一现象对应"]
    )


def is_air_oxygen_pressure_standard_medium(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """红磷测空气中氧气含量的压强曲线/气球变化：标准实验图像分析，通常为中等题而非拔高题。"""
    text = visible_text(data, include_analysis=True)
    return (
        contains_any(text, ["红磷", "测定空气中氧气含量", "空气中氧气含量", "氧气含量"])
        and contains_any(text, ["压强", "气压", "压力", "气球", "曲线", "图像", "图象", "图 2", "图2"])
        and not contains_any(text, ["方案评价", "质疑", "补充实验", "误差分析", "复杂守恒", "质量分数", "纯度", "滤渣", "滤液"])
    )


def is_bicarbonate_purity_hard(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """NaHCO3/小苏打性质表格 + 样品纯度/质量分数计算，通常有实验归纳和定量计算卡点，判拔高。"""
    text = visible_text(data, include_analysis=True)
    return (
        contains_any(text, ["NaHCO3", "NaHCO₃", "碳酸氢钠", "小苏打"])
        and contains_any(text, ["纯度", "样品中", "含量"])
        and contains_any(text, ["表格", "数据", "质量差", "反应前后", "反思", "测定"])
        and features.get("calculation_complexity") in ["化学方程式计算或关系式计算", "复杂守恒或图像计算"]
        and (
            features.get("information_carrier") == "多图表综合"
            or features.get("subquestion_dependency") == "多问且层层递进"
            or features.get("graph_table_requirement") == "图像反推或拐点分析"
        )
        and not ("配制一定质量分数" in text and features.get("subquestion_dependency") == "多问但相互独立")
    )


def is_common_substance_network_inference(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """A-E 常见物质转化推断：若需结合转化关系、颜色/黑色固体/CO-CuO等线索，通常至少中等题。"""
    text = visible_text(data, include_analysis=True)
    compact_text = re.sub(r"\s+", "", text)
    has_letters = (
        bool(re.search(r"A[~\-—至到、,，和]+[B-E]", compact_text))
        or contains_any(compact_text, ["A、B、C、D、E", "A～E", "A-E", "A~E", "ABCDE"])
    )
    return (
        has_letters
        and contains_any(text, ["常见物质", "物质转化", "转化关系", "推断", "反应关系", "框图"])
        and not contains_any(text, ["对于化学反应", "A}+\\mathrm{B}", "A+B", "置换反应", "复分解反应", "中和反应"])
    )




def is_solution_classification_basic(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """溶液/非溶液分类辨析：不是纯记忆，一般至少基础题。"""
    text = visible_text(data, include_analysis=True)
    return (
        contains_any(text, ["不属于溶液", "属于溶液", "溶液的是", "溶液的说法", "溶液中"])
        and not contains_any(text, ["溶质质量分数", "质量分数", "曲线", "图像", "图象", "配制", "计算"])
    )


def is_co_reduction_combustion_combo_medium(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """CO还原氧化铁 + 燃烧条件/尾气处理组合实验：超过基础操作，通常中等。"""
    text = visible_text(data, include_analysis=True)
    has_co_reduction = contains_any(text, ["CO", "一氧化碳"]) and contains_any(text, ["Fe2O3", "氧化铁", "还原氧化铁", "炼铁"])
    has_combustion = contains_any(text, ["燃烧条件", "燃烧的条件", "铁粉", "脱脂棉", "红磷", "白磷"])
    has_lab_combo = contains_any(text, ["实验 1", "实验1", "实验 2", "实验2", "尾气处理", "酒精灯", "装置"])
    hard_exclusion = contains_any(text, ["质量分数", "纯度", "守恒", "关系式", "图像反推", "拐点", "滤渣", "滤液", "方案评价"])
    return has_co_reduction and has_combustion and has_lab_combo and not hard_exclusion


def is_complex_equation_balancing_medium(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """陌生复杂方程式配平：需要元素守恒列系数关系，通常中等。"""
    text = visible_text(data, include_analysis=True)
    compact = re.sub(r"\s+", "", text)
    return (
        contains_any(text, ["配平", "化学计量数", "计量数"])
        and (
            contains_any(compact, ["S8", "Ca(OH)2", "CaS5", "CaS2O3"])
            or len(re.findall(r"[A-Z][a-z]?(?:_?\{?\d+\}?|\d*)", compact)) >= 6
        )
        and not contains_any(text, ["选择合适装置", "实验探究", "流程", "滤渣", "滤液"])
    )


def is_unfamiliar_material_transfer_medium(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """陌生材料迁移题：需要根据材料迁移相对分子质量、质量守恒、化合价/氧化还原方向，通常中等。"""
    text = visible_text(data, include_analysis=True)
    return (
        contains_any(text, ["阅读材料", "三氧化二碳", "C2O3", "C 2 O 3", "某星球", "化学性质与一氧化碳相似"])
        and contains_any(text, ["相对分子质量", "质量守恒", "化合价", "氧化", "还原", "酸性"])
        and not contains_any(text, ["图像", "图象", "曲线", "复杂守恒", "多变量", "方案评价"])
    )


def is_standard_precipitation_purity_table_medium(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """碳酸钠样品纯度 + 氯化钙沉淀表格：平台读数 + 单方程式计算，通常中等而非拔高。"""
    text = visible_text(data, include_analysis=True)
    return (
        contains_any(text, ["碳酸钠样品", "Na2CO3", "Na₂CO₃"])
        and contains_any(text, ["纯度", "质量分数", "含量"])
        and contains_any(text, ["氯化钙", "CaCl2", "CaCl₂", "沉淀", "平均分", "四份", "表"])
        and not contains_any(text, ["滤渣", "滤液", "过量不足", "先后反应", "拐点", "曲线", "图像", "图象", "方案评价", "干扰", "混合物中多种"])
    )


def is_single_reaction_decomposition_graph_medium(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """KClO3 单一分解反应质量变化图：常规图像辨析，通常中等而非拔高。"""
    text = visible_text(data, include_analysis=True)
    return (
        contains_any(text, ["KClO3", "KClO₃", "氯酸钾"])
        and contains_any(text, ["MnO2", "MnO₂", "二氧化锰"])
        and contains_any(text, ["分解", "加热", "质量", "图", "图像", "图象", "曲线"])
        and not contains_any(text, ["纯度", "质量分数", "过量", "不足", "滤渣", "滤液", "方案评价", "多反应", "多种金属", "混合物计算"])
    )

def should_downgrade_basic_to_easy(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    # 低阶直接识记集合题允许基础 -> 送分。
    if is_pure_direct_recall_set(features, data):
        return True

    if is_long_context_or_new_situation(data):
        return False

    text = visible_text(data, include_analysis=False)
    if ("下列" in text or "说法" in text or "正确的是" in text or "错误的是" in text) and len(text) > 90:
        return False

    # 多问/多空不再默认降为送分，避免宏观-微观-符号多小问被误降。
    if features.get("subquestion_dependency") != "无多问" or count_subquestions(data) > 0:
        return False

    simple_problem = features.get("problem_structure") in ["概念判断", "化学用语与分类"]
    simple_carrier = features.get("information_carrier") in ["纯文字", "单图识别", "微观示意图"]
    return (
        simple_problem
        and simple_carrier
        and features.get("step_count") == "1-2步"
        and features.get("equation_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_diff") == "低"
        and features.get("chemistry_process_count") in ["单一事实", "单一反应"]
        and features.get("constraint_count") == "无约束"
        and features.get("evidence_relation") in ["无证据链", "单一现象对应"]
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )


def should_upgrade_easy_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    text = visible_text(data, include_analysis=True)
    stem_options = visible_text(data, include_analysis=False)

    if is_pure_direct_recall_set(features, data):
        return []

    if is_solution_classification_basic(features, data):
        reasons.append("溶液/非溶液属于物质分类概念辨析，至少基础题")

    if features.get("step_count") != "1-2步":
        reasons.append(f'解题步骤数为"{features.get("step_count")}"')
    if features.get("knowledge_count") != "1个":
        reasons.append(f'知识点数量为"{features.get("knowledge_count")}"')
    if features.get("equation_count") != "0-1个":
        reasons.append("涉及多个化学方程式或反应关系")
    if features.get("calculation_complexity") in ["简单笔算", "化学方程式计算或关系式计算", "复杂守恒或图像计算"]:
        reasons.append(f'计算复杂度为"{features.get("calculation_complexity")}"')
    if features.get("experiment_requirement") != "无":
        reasons.append("含实验操作、现象分析或探究要求")
    if features.get("information_carrier") in ["实验装置图", "流程图", "图像或表格", "多图表综合"]:
        reasons.append(f'信息载体为"{features.get("information_carrier")}"，不属于单一概念直答')
    if features.get("graph_table_requirement") != "无":
        reasons.append("需要图像/表格处理")
    if features.get("chemistry_process_count") in ["2-3个反应或过程", "多反应连续转化或流程"]:
        reasons.append("涉及多个反应或过程")
    if features.get("evidence_relation") in ["多现象证据链", "证据冲突与排除"]:
        reasons.append("存在证据链分析")

    if features.get("subquestion_dependency") != "无多问":
        reasons.append("存在多个设问，不属于严格单点直答")
    if "宏观" in text and "微观" in text and "符号" in text:
        reasons.append("涉及宏观-微观-符号表征对应，至少基础题")
    if count_subquestions(data) >= 4:
        reasons.append("多小问数量较多")
    if count_fill_blanks(stem_options) >= 4 and features.get("knowledge_count") != "1个":
        reasons.append("多空填空且涉及不同知识点")

    force_basic_keywords = [
        "化合价", "相对分子质量", "质量分数", "溶质质量分数", "配平", "化学方程式", "符号表达式",
        "过滤", "蒸馏", "吸附", "电解水", "制取", "收集", "检验", "除杂", "鉴别",
        "单质", "化合物", "氧化物", "有机物", "酸碱盐", "金属活动性", "置换反应",
    ]
    if contains_any(text, force_basic_keywords) and not is_trivial_concept_question(features, data):
        reasons.append("命中化学基础应用关键词，至少基础题")

    return reasons


def should_upgrade_basic_to_medium(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    text = visible_text(data, include_analysis=True)

    # 特定常见物质转化推断：先于基础题保护，否则会被“步骤短/单线索”误降。
    if is_common_substance_network_inference(features, data):
        reasons.append("A-E常见物质转化推断需要结合物质特征、转化关系和方程式，达到中等题")
        return reasons
    if is_co_reduction_combustion_combo_medium(features, data):
        reasons.append("CO还原氧化铁与燃烧条件组合实验涉及尾气处理、操作顺序和条件对比，达到中等题")
        return reasons
    if is_complex_equation_balancing_medium(features, data):
        reasons.append("陌生复杂化学方程式配平需要元素守恒列系数关系，达到中等题")
        return reasons
    if is_unfamiliar_material_transfer_medium(features, data):
        reasons.append("陌生材料迁移题需要综合相对分子质量、质量守恒和化合价/氧化还原判断，达到中等题")
        return reasons

    # 先做基础题保护，避免 pH/变质/质量守恒等关键词把独立基础空误升中等。
    if is_low_level_basic_application(features, data) and not is_standard_experiment_medium_combo(features, data):
        return []

    if is_standard_experiment_medium_combo(features, data):
        reasons.append("多个重要实验装置/现象/方程式并列综合，达到中等题")

    if (
        (contains_any(text, ["生成等量氢气", "相同质量", "不同金属", "金属用量"]) and contains_any(text, ["氢气", "H_{2}", "H2"]))
        or ("制取氢气" in text and count_subquestions(data) >= 3)
    ):
        reasons.append("氢气制取中涉及装置/收集/金属与酸反应综合，达到中等题")

    if features.get("step_count") in ["6-8步", "9-12步", "12步以上"]:
        reasons.append(f'步骤数达"{features.get("step_count")}"')
    if features.get("chemistry_process_count") == "多反应连续转化或流程":
        reasons.append("存在多反应连续转化或流程")
    if features.get("calculation_complexity") in ["化学方程式计算或关系式计算", "复杂守恒或图像计算"]:
        reasons.append(f'计算需要"{features.get("calculation_complexity")}"')
    if features.get("experiment_requirement") in ["控制变量或现象分析", "方案设计或误差评价"]:
        reasons.append(f'实验要求为"{features.get("experiment_requirement")}"')
    if features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或拐点分析"]:
        reasons.append(f'图表处理要求为"{features.get("graph_table_requirement")}"')
    if features.get("evidence_relation") in ["多现象证据链", "证据冲突与排除"]:
        reasons.append(f'证据关系为"{features.get("evidence_relation")}"')
    if features.get("subquestion_dependency") == "多问且层层递进":
        reasons.append("多小问层层递进")
    if features.get("information_carrier") in ["流程图", "多图表综合"] and features.get("knowledge_count") != "1个":
        reasons.append("流程/多图表与多知识点结合")

    force_medium_keywords = [
        "控制变量", "对照实验", "催化剂", "探究", "项目式", "任务一", "任务二", "流程", "工艺", "滤渣", "滤液",
        "溶解度曲线", "图像", "图象", "曲线", "成分", "推断", "除杂", "鉴别", "金属活动性",
        "关系式", "混合物", "过量", "不足",
    ]
    if contains_any(text, force_medium_keywords) and (
        features.get("step_count") != "1-2步"
        or features.get("knowledge_count") != "1个"
        or features.get("experiment_requirement") != "无"
        or features.get("graph_table_requirement") != "无"
    ):
        reasons.append("命中实验/流程/图像/推断类中等综合关键词")

    # 基础升中等至少需要一个真实综合触发点；若只有关键词但特征仍是低阶，已被保护。
    return reasons


def should_downgrade_medium_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    text = visible_text(data, include_analysis=True)
    if "制取氢气" in text and count_subquestions(data) >= 3:
        return False
    if (
        is_common_substance_network_inference(features, data)
        or is_co_reduction_combustion_combo_medium(features, data)
        or is_complex_equation_balancing_medium(features, data)
        or is_unfamiliar_material_transfer_medium(features, data)
    ):
        return False
    if is_multi_standard_lab_independent_basic(features, data):
        return True
    if is_standard_experiment_basic(features, data):
        return True
    if is_long_reading_direct_info(features, data):
        return True
    if is_low_level_basic_application(features, data):
        return True

    if is_long_context_or_new_situation(data):
        return False
    return (
        features.get("step_count") in ["1-2步", "3-5步"]
        and features.get("equation_count") in ["0-1个", "2-3个"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("reasoning_chain") in ["直接套用", "简单因果推理"]
        and features.get("chemistry_process_count") in ["单一事实", "单一反应", "2-3个反应或过程"]
        and features.get("constraint_count") in ["无约束", "单一约束"]
        and features.get("evidence_relation") in ["无证据链", "单一现象对应"]
        and features.get("experiment_requirement") in ["无", "基础操作或读数"]
        and features.get("graph_table_requirement") in ["无", "直接读数"]
        and features.get("subquestion_dependency") != "多问且层层递进"
    )



def should_upgrade_medium_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    strong_reasons: List[str] = []
    support_reasons: List[str] = []
    text = visible_text(data, include_analysis=True)

    hard_keywords = [
        "部分变质", "氢氧化钠", "滴加盐酸", "离子数目", "拐点", "平台", "滤渣", "滤液", "循环物质",
        "方案评价", "可靠性", "质疑", "干扰", "过量", "不足", "先后反应", "差量法", "关系式法", "元素守恒",
        "合金", "混合物", "样品纯度", "纯度", "质量分数", "定量实验", "环保缺陷", "压强变化", "气球"
    ]

    # 单一标准图像/表格计算题停留中等，避免被“图像反推/多约束”误升拔高。
    if (
        is_air_oxygen_pressure_standard_medium(features, data)
        or is_standard_precipitation_purity_table_medium(features, data)
        or is_single_reaction_decomposition_graph_medium(features, data)
    ):
        return False, []

    # 强触发：存在明确卡点。
    if features.get("step_count") in ["9-12步", "12步以上"]:
        strong_reasons.append(f'步骤数达"{features.get("step_count")}"')
    if features.get("equation_count") in ["4-6个", "7个以上"] and features.get("information_carrier") == "多图表综合":
        strong_reasons.append(f'多个方程式与多图表信息结合，方程式数量为"{features.get("equation_count")}"')
    elif features.get("equation_count") in ["4-6个", "7个以上"] and contains_any(text, ["误差", "测定", "流程", "制备", "尾气处理"]):
        strong_reasons.append(f'多个方程式服务于实验/流程综合，方程式数量为"{features.get("equation_count")}"')
    if features.get("calculation_complexity") == "复杂守恒或图像计算":
        strong_reasons.append("需要复杂守恒或图像计算")
    if (
        features.get("graph_table_requirement") == "图像反推或拐点分析"
        and features.get("information_carrier") in ["图像或表格", "多图表综合"]
        and contains_any(text, ["图像", "图象", "曲线", "拐点", "平台", "pH", "压强", "沉淀", "气体质量", "离子数目"])
    ):
        strong_reasons.append("需要图像反推或拐点分析")
    if features.get("evidence_relation") == "证据冲突与排除":
        strong_reasons.append("存在证据冲突与干扰排除")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        strong_reasons.append("需要方案设计、可靠性评价或误差分析")
    if (
        features.get("problem_structure") in ["物质推断", "工艺流程", "计算综合"]
        and features.get("chemistry_process_count") == "多反应连续转化或流程"
        and features.get("constraint_count") == "多约束"
        and contains_any(text, hard_keywords)
    ):
        strong_reasons.append("物质推断/流程/计算中同时出现多反应、多约束和拔高关键词")
    if (
        features.get("information_carrier") == "多图表综合"
        and features.get("experiment_requirement") == "控制变量或现象分析"
        and features.get("calculation_complexity") == "化学方程式计算或关系式计算"
        and contains_any(text, ["样品纯度", "纯度", "质量分数", "测定", "压强变化", "气球"])
    ):
        strong_reasons.append("多图表实验分析与样品纯度/质量分数/压强变化计算结合")
    if is_bicarbonate_purity_hard(features, data):
        strong_reasons.append("NaHCO3/小苏打性质表格与样品纯度计算结合，存在实验归纳和定量计算卡点")
    if (
        contains_any(text, ["自动充气气球", "压强变化", "压强"] )
        and features.get("subquestion_dependency") == "多问且层层递进"
        and features.get("information_carrier") == "多图表综合"
        and features.get("constraint_count") == "多约束"
        and features.get("evidence_relation") == "多现象证据链"
    ):
        strong_reasons.append("项目式气球成分探究需要结合压强图像/数据与多现象证据链反推成分")

    # 支撑触发：单独不足以升拔高，但可与强触发组合。
    if features.get("chemistry_process_count") == "多反应连续转化或流程":
        support_reasons.append("存在多反应连续转化或流程")
    if features.get("constraint_count") == "多约束":
        support_reasons.append("存在过量/不足/先后反应等多约束")
    if features.get("evidence_relation") == "多现象证据链":
        support_reasons.append("存在多现象证据链")
    if features.get("information_carrier") == "多图表综合":
        support_reasons.append("需要整合多图表信息")
    if features.get("knowledge_count") == "4个及以上":
        support_reasons.append("知识点数量达到4个及以上")
    if features.get("subquestion_dependency") == "多问且层层递进":
        support_reasons.append("多小问层层递进")
    if contains_any(text, hard_keywords) and (
        features.get("calculation_complexity") != "口算或直接判断"
        or features.get("evidence_relation") != "无证据链"
        or features.get("experiment_requirement") != "无"
        or features.get("graph_table_requirement") != "无"
    ):
        support_reasons.append("命中变质/流程/图像/守恒/方案评价类拔高关键词")

    # 微观示意图/物质组成结构类选择题，即使模型把“读图”写成反推，也通常停留在中等题。
    if contains_any(text, ["了解物质的组成和结构", "微观示意图", "结构示意图"]) and not contains_any(text, ["滤渣", "滤液", "变质", "质量分数", "方案", "守恒", "压强", "曲线"]):
        return False, strong_reasons + support_reasons

    reasons = strong_reasons + support_reasons
    return len(strong_reasons) >= 1 and len(reasons) >= 2, reasons


def should_downgrade_standard_experiment(features: Dict[str, Any], data: Dict[str, Any]) -> Optional[str]:
    """拔高题降档：只降真正的标准实验/标准单线计算，避免把金属滤渣滤液、流程、图像探究误降。"""
    if is_air_oxygen_pressure_standard_medium(features, data):
        return "中等题"
    if is_standard_precipitation_purity_table_medium(features, data):
        return "中等题"
    if is_single_reaction_decomposition_graph_medium(features, data):
        return "中等题"
    if is_single_path_standard_calculation(features, data):
        return "中等题"

    # 有这些拔高核心结构时，不能按“标准实验”降档。
    if (
        features.get("chemistry_process_count") == "多反应连续转化或流程"
        or features.get("constraint_count") == "多约束"
        or features.get("evidence_relation") in ["多现象证据链", "证据冲突与排除"]
        or features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或拐点分析"]
        or features.get("information_carrier") == "多图表综合"
        or features.get("experiment_requirement") in ["控制变量或现象分析", "方案设计或误差评价"]
    ):
        return None

    if features.get("step_count") in ["1-2步", "3-5步"] and features.get("experiment_requirement") in ["无", "基础操作或读数"]:
        if features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]:
            return "基础题"
    if (
        features.get("step_count") in ["3-5步", "6-8步"]
        and features.get("calculation_complexity") != "复杂守恒或图像计算"
        and features.get("evidence_relation") != "证据冲突与排除"
        and features.get("experiment_requirement") != "方案设计或误差评价"
        and features.get("graph_table_requirement") != "图像反推或拐点分析"
    ):
        return "中等题"
    return None


def high_level_feature_count(features: Dict[str, Any], data: Dict[str, Any]) -> int:
    count = 0
    if features.get("step_count") == "12步以上":
        count += 1
    if features.get("equation_count") == "7个以上":
        count += 1
    if features.get("knowledge_count") == "4个及以上" and features.get("knowledge_diff") == "高":
        count += 1
    if features.get("chemistry_process_count") == "多反应连续转化或流程":
        count += 1
    if features.get("constraint_count") == "多约束":
        count += 1
    if features.get("evidence_relation") == "证据冲突与排除":
        count += 1
    if features.get("calculation_complexity") == "复杂守恒或图像计算":
        count += 1
    if features.get("experiment_requirement") == "方案设计或误差评价":
        count += 1
    if features.get("graph_table_requirement") == "图像反推或拐点分析":
        count += 1
    if features.get("information_carrier") == "多图表综合":
        count += 1
    if features.get("subquestion_dependency") == "多问且层层递进" and count_subquestions(data) >= 4:
        count += 1
    return count


def has_final_core_combo(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """压轴题核心组合约束。

    压轴不能只靠题干长、流程长、关键词多；必须同时具有：
    A. 复杂计算 / 证据冲突排除 / 方案评价之一；
    B. 图像拐点反推 / 多反应流程 / 多约束之一；
    C. 多问递进且至少 3 个小问。
    """
    text = visible_text(data, include_analysis=True)
    project_final_signal = (
        contains_any(text, ["蒸汽眼罩", "数字传感器", "探究一", "探究二"])
        and features.get("information_carrier") == "多图表综合"
        and features.get("experiment_requirement") == "控制变量或现象分析"
        and features.get("knowledge_count") == "4个及以上"
    )
    quantified_conflict = (
        features.get("evidence_relation") == "证据冲突与排除"
        and features.get("experiment_requirement") == "方案设计或误差评价"
        and contains_any(text, ["定量", "质量分数", "图像", "图象", "曲线", "气体质量", "二氧化碳质量", "氢气质量", "极值", "范围"])
    )
    core_a = (
        features.get("calculation_complexity") == "复杂守恒或图像计算"
        or features.get("graph_table_requirement") == "图像反推或拐点分析"
        or quantified_conflict
        or project_final_signal
    )
    core_b = (
        features.get("graph_table_requirement") == "图像反推或拐点分析"
        or features.get("chemistry_process_count") == "多反应连续转化或流程"
        or features.get("constraint_count") == "多约束"
    )
    core_c = (
        features.get("subquestion_dependency") == "多问且层层递进"
        and count_subquestions(data) >= 3
    )
    return core_a and core_b and core_c


def should_upgrade_hard_to_final(features: Dict[str, Any], data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    high_count = high_level_feature_count(features, data)
    final_core_combo = has_final_core_combo(features, data)

    if high_count >= 5:
        reasons.append(f"高阶化学特征达到 {high_count} 项，接近压轴题密度")
    elif high_count >= 4:
        reasons.append(f"高阶化学特征达到 {high_count} 项")

    if features.get("step_count") == "12步以上":
        reasons.append("解题链条超过12步")
    if count_subquestions(data) >= 5 and features.get("subquestion_dependency") == "多问且层层递进":
        reasons.append("多小问层层递进且数量较多")
    if final_core_combo:
        reasons.append("同时具备复杂证据/计算/方案评价、多反应或多约束、递进多问三类压轴核心结构")

    text = visible_text(data, include_analysis=True)
    final_keywords = [
        "综合", "工艺流程", "制备", "定量实验", "离子数目变化", "方案评价", "误差", "混合物", "合金", "质量分数",
        "变质", "滤渣", "滤液", "循环", "尾气处理", "环保", "关系式", "守恒", "极值", "不可能是",
    ]
    if contains_any(text, final_keywords) and high_count >= 4 and count_subquestions(data) >= 3 and final_core_combo:
        reasons.append("题目具备中考最后综合题属性")

    return len(reasons) >= 2 and high_count >= 4 and final_core_combo, reasons


def should_downgrade_final_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """只在题目明确低于压轴结构时降档。

    “没有满足主动升压轴条件”不等于“原判压轴必须降”。这是物理正式
    后处理中的双向独立原则，避免把模型已识别出的单问复杂压轴题机械降档。
    """
    high_count = high_level_feature_count(features, data)
    text = visible_text(data, include_analysis=True)
    strong_final_signal = (
        features.get("calculation_complexity")
        == "复杂守恒或图像计算"
        or features.get("evidence_relation") == "证据冲突与排除"
        or features.get("experiment_requirement")
        == "方案设计或误差评价"
        or (
            features.get("chemistry_process_count")
            == "多反应连续转化或流程"
            and features.get("constraint_count") == "多约束"
        )
        or contains_any(
            text,
            [
                "分类讨论",
                "多解",
                "有效解",
                "极值",
                "边界",
                "差量法",
                "多方程",
                "联立",
                "证据冲突",
            ],
        )
    )
    if features.get("step_count") in ["9-12步", "12步以上"]:
        return False
    if features.get("step_count") == "6-8步":
        return high_count < 2 and not strong_final_signal
    return high_count < 3 and not strong_final_signal


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


CORE12_DEPTH_ORDER = {
    "0层": 0,
    "1层": 1,
    "2-3层": 2,
    "4-5层": 3,
    "6层及以上": 4,
}


def core12_medium_evidence(features: Dict[str, Any]) -> List[str]:
    evidence: List[str] = []
    if features["reasoning_depth"] in {"2-3层", "4-5层", "6层及以上"}:
        evidence.append(f"推理深度={features['reasoning_depth']}")
    if features["reaction_relation"] in {
        "2-3个并列或简单连续反应",
        "多反应连续转化",
        "先后、竞争或过量不足",
        "需要分情况判断的反应模型",
    }:
        evidence.append(f"反应关系={features['reaction_relation']}")
    if features["evidence_relation"] in {
        "多条清晰证据联合",
        "需要排除竞争解释",
        "证据冲突、筛选或多层排除",
    }:
        evidence.append(f"证据关系={features['evidence_relation']}")
    if features["experiment_requirement"] in {
        "控制变量、现象解释或数据归纳",
        "方案设计、评价或补充实验",
        "多阶段探究与定量误差",
    }:
        evidence.append(f"实验要求={features['experiment_requirement']}")
    if features["graph_table_requirement"] in {
        "多组比较归纳",
        "拐点、平台或分段反推",
        "多图表耦合建模",
    }:
        evidence.append(f"图表要求={features['graph_table_requirement']}")
    if features["calculation_model"] in {
        "单一方程式或关系式",
        "单一守恒或多反应计算",
        "多重守恒、差量、联立或分类",
    }:
        evidence.append(f"计算模型={features['calculation_model']}")
    return evidence


def core12_basic_application_evidence(
    features: Dict[str, Any],
) -> List[str]:
    """送分题离开直接检索束的结构证据。"""
    evidence: List[str] = []
    if features["reasoning_depth"] == "1层":
        evidence.append("推理深度=1层")
    if features["reasoning_direction"] in {
        "正向推导",
        "逆向推导",
        "分类讨论或综合推导",
    }:
        evidence.append(f"推理方向={features['reasoning_direction']}")
    if features["knowledge_relation"] != "单一知识点":
        evidence.append(f"知识关系={features['knowledge_relation']}")
    if features["representation_conversion"] != "无":
        evidence.append(
            f"表征转换={features['representation_conversion']}"
        )
    if features["reaction_relation"] != "无反应关系":
        evidence.append(f"反应关系={features['reaction_relation']}")
    if features["constraint_complexity"] != "无约束":
        evidence.append(
            f"约束复杂度={features['constraint_complexity']}"
        )
    if features["evidence_relation"] != "无证据任务":
        evidence.append(f"证据关系={features['evidence_relation']}")
    if features["experiment_requirement"] != "无":
        evidence.append(f"实验要求={features['experiment_requirement']}")
    if features["calculation_model"] != "无":
        evidence.append(f"计算模型={features['calculation_model']}")
    if (
        features["unfamiliar_information_transfer"]
        != "课内直接原型"
    ):
        evidence.append(
            "陌生信息迁移="
            + features["unfamiliar_information_transfer"]
        )
    return evidence


def core12_complete_model_evidence(
    features: Dict[str, Any],
) -> List[str]:
    """基础题进入中等题比较所需的完整常规模型证据。"""
    evidence: List[str] = []
    if features["reaction_relation"] in {
        "2-3个并列或简单连续反应",
        "多反应连续转化",
        "先后、竞争或过量不足",
        "需要分情况判断的反应模型",
    }:
        evidence.append(f"反应关系={features['reaction_relation']}")
    if features["evidence_relation"] in {
        "多条清晰证据联合",
        "需要排除竞争解释",
        "证据冲突、筛选或多层排除",
    }:
        evidence.append(f"证据关系={features['evidence_relation']}")
    if features["experiment_requirement"] in {
        "控制变量、现象解释或数据归纳",
        "方案设计、评价或补充实验",
        "多阶段探究与定量误差",
    }:
        evidence.append(f"实验要求={features['experiment_requirement']}")
    if features["graph_table_requirement"] in {
        "多组比较归纳",
        "拐点、平台或分段反推",
        "多图表耦合建模",
    }:
        evidence.append(f"图表要求={features['graph_table_requirement']}")
    if features["calculation_model"] in {
        "单一方程式或关系式",
        "单一守恒或多反应计算",
        "多重守恒、差量、联立或分类",
    }:
        evidence.append(f"计算模型={features['calculation_model']}")
    if features["subquestion_dependency"] in {
        "多问共享模型但无答案依赖",
        "多问存在结果或任务链依赖",
    }:
        evidence.append(
            f"小问关系={features['subquestion_dependency']}"
        )
    return evidence


def core12_high_evidence(features: Dict[str, Any]) -> List[str]:
    evidence: List[str] = []
    if features["reasoning_depth"] in {"4-5层", "6层及以上"}:
        evidence.append(f"推理深度={features['reasoning_depth']}")
    if features["reasoning_direction"] in {"逆向推导", "分类讨论或综合推导"}:
        evidence.append(f"推理方向={features['reasoning_direction']}")
    if features["knowledge_relation"] in {"跨模块融合", "多模块深度融合"}:
        evidence.append(f"知识关系={features['knowledge_relation']}")
    if features["reaction_relation"] in {
        "多反应连续转化",
        "先后、竞争或过量不足",
        "需要分情况判断的反应模型",
    }:
        evidence.append(f"反应关系={features['reaction_relation']}")
    if features["constraint_complexity"] in {"多个相互关联约束", "多层嵌套约束"}:
        evidence.append(f"约束复杂度={features['constraint_complexity']}")
    if features["evidence_relation"] in {
        "需要排除竞争解释",
        "证据冲突、筛选或多层排除",
    }:
        evidence.append(f"证据关系={features['evidence_relation']}")
    if features["experiment_requirement"] in {
        "方案设计、评价或补充实验",
        "多阶段探究与定量误差",
    }:
        evidence.append(f"实验要求={features['experiment_requirement']}")
    if features["graph_table_requirement"] in {
        "拐点、平台或分段反推",
        "多图表耦合建模",
    }:
        evidence.append(f"图表要求={features['graph_table_requirement']}")
    if features["calculation_model"] in {
        "单一守恒或多反应计算",
        "多重守恒、差量、联立或分类",
    }:
        evidence.append(f"计算模型={features['calculation_model']}")
    if features["unfamiliar_information_transfer"] in {
        "迁移后建立关系",
        "完全陌生模型现场建立",
    }:
        evidence.append(
            "陌生信息迁移="
            + features["unfamiliar_information_transfer"]
        )
    if features["subquestion_dependency"] == "多问存在结果或任务链依赖":
        evidence.append("多问存在结果或任务链依赖")
    return evidence


def core12_decisive_evidence(features: Dict[str, Any]) -> List[str]:
    """中等题进入拔高题比较所需的决定性转换。"""
    evidence: List[str] = []
    if features["reasoning_direction"] in {
        "逆向推导",
        "分类讨论或综合推导",
    }:
        evidence.append(f"推理方向={features['reasoning_direction']}")
    if features["representation_conversion"] in {
        "两类表征连续转换",
        "宏观-微观-符号-定量多重转换",
    }:
        evidence.append(
            f"表征转换={features['representation_conversion']}"
        )
    if features["reaction_relation"] in {
        "先后、竞争或过量不足",
        "需要分情况判断的反应模型",
    }:
        evidence.append(f"反应关系={features['reaction_relation']}")
    if features["constraint_complexity"] in {
        "多个相互关联约束",
        "多层嵌套约束",
    }:
        evidence.append(
            f"约束复杂度={features['constraint_complexity']}"
        )
    if features["evidence_relation"] in {
        "需要排除竞争解释",
        "证据冲突、筛选或多层排除",
    }:
        evidence.append(f"证据关系={features['evidence_relation']}")
    if features["experiment_requirement"] in {
        "方案设计、评价或补充实验",
        "多阶段探究与定量误差",
    }:
        evidence.append(f"实验要求={features['experiment_requirement']}")
    if features["graph_table_requirement"] in {
        "拐点、平台或分段反推",
        "多图表耦合建模",
    }:
        evidence.append(f"图表要求={features['graph_table_requirement']}")
    if features["calculation_model"] in {
        "单一守恒或多反应计算",
        "多重守恒、差量、联立或分类",
    }:
        evidence.append(f"计算模型={features['calculation_model']}")
    if features["unfamiliar_information_transfer"] in {
        "迁移后建立关系",
        "完全陌生模型现场建立",
    }:
        evidence.append(
            "陌生信息迁移="
            + features["unfamiliar_information_transfer"]
        )
    return evidence


def core12_final_evidence(features: Dict[str, Any]) -> List[str]:
    """拔高题进入压轴题比较所需的强耦合证据。"""
    evidence: List[str] = []
    if features["reasoning_direction"] in {
        "逆向推导",
        "分类讨论或综合推导",
    }:
        evidence.append(f"推理方向={features['reasoning_direction']}")
    if features["knowledge_relation"] == "多模块深度融合":
        evidence.append("知识关系=多模块深度融合")
    if (
        features["representation_conversion"]
        == "宏观-微观-符号-定量多重转换"
    ):
        evidence.append(
            "表征转换=宏观-微观-符号-定量多重转换"
        )
    if features["reaction_relation"] in {
        "多反应连续转化",
        "先后、竞争或过量不足",
        "需要分情况判断的反应模型",
    }:
        evidence.append(f"反应关系={features['reaction_relation']}")
    if features["constraint_complexity"] == "多层嵌套约束":
        evidence.append("约束复杂度=多层嵌套约束")
    if features["evidence_relation"] == "证据冲突、筛选或多层排除":
        evidence.append("证据关系=证据冲突、筛选或多层排除")
    if features["experiment_requirement"] == "多阶段探究与定量误差":
        evidence.append("实验要求=多阶段探究与定量误差")
    if features["graph_table_requirement"] == "多图表耦合建模":
        evidence.append("图表要求=多图表耦合建模")
    if (
        features["calculation_model"]
        == "多重守恒、差量、联立或分类"
    ):
        evidence.append("计算模型=多重守恒、差量、联立或分类")
    if features["unfamiliar_information_transfer"] in {
        "迁移后建立关系",
        "完全陌生模型现场建立",
    }:
        evidence.append(
            "陌生信息迁移="
            + features["unfamiliar_information_transfer"]
        )
    if (
        features["subquestion_dependency"]
        == "多问存在结果或任务链依赖"
    ):
        evidence.append("多问存在结果或任务链依赖")
    return evidence


def core12_is_direct_retrieval(features: Dict[str, Any]) -> bool:
    return (
        features["reasoning_depth"] == "0层"
        and features["reasoning_direction"] == "直接识记"
        and features["knowledge_relation"] == "单一知识点"
        and features["representation_conversion"] == "无"
        and features["reaction_relation"] == "无反应关系"
        and features["constraint_complexity"] == "无约束"
        and features["evidence_relation"] == "无证据任务"
        and features["experiment_requirement"] == "无"
        and features["graph_table_requirement"] in {"无", "直接读数"}
        and features["calculation_model"] == "无"
        and features["unfamiliar_information_transfer"] == "课内直接原型"
        and features["subquestion_dependency"] in {"无多问", "多问相互独立"}
    )


def add_feature_audit_flags(
    rating_result: Dict[str, Any],
    data: Dict[str, Any],
) -> None:
    """记录 Core-12 的结构异常，只审计、不直接改档。"""
    features = rating_result.get("features") or {}
    level = rating_result.get("difficulty_level", "")
    flags: List[str] = []
    text = visible_text(data, include_analysis=True)
    if (
        VISUAL_REFERENCE_RE.search(text)
        and features.get("graph_table_requirement") == "无"
    ):
        flags.append(
            "题面明确引用图表/流程/装置，但graph_table_requirement=无，"
            "需结合image_input_used检查视觉信息是否只是呈现"
        )
    high_count = len(core12_high_evidence(features))
    final_count = len(core12_final_evidence(features))
    if level == "送分题" and not core12_is_direct_retrieval(features):
        flags.append(
            "送分题包含非全低Core-12：可能是教师口径下的熟悉模板透明映射，"
            "也可能是真实一步应用；仅审计，不自动升档"
        )
    if level == "基础题" and core12_is_direct_retrieval(features):
        flags.append(
            "基础题呈现全低Core-12：可能是送分边界，也可能漏识别复合任务；"
            "仅审计，不自动降档"
        )
    if level == "拔高题" and high_count < 2:
        flags.append(f"拔高题仅有{high_count}项高阶结构证据")
    if level == "压轴题" and (
        features["reasoning_depth"] != "6层及以上"
        and final_count < 2
    ):
        flags.append("压轴题缺少6层以上深链或特殊结构的复合证据")
    if rating_result.get("postprocess_trace"):
        flags.append("后处理已作一次结构校准，原始模型结果另行保留")
    rating_result["feature_audit_flags"] = list(dict.fromkeys(flags))


def postprocess_chemistry_difficulty(rating_result: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """化学后处理校准与审计主流程。

    chemistry_stable 默认启用教师分布结构校准并写回最终等级；做 A/B
    消融时可显式关闭 CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK。
    通用历史规则仍只在 CHEMISTRY_ENABLE_LEVEL_WRITEBACK=1 时写回。
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
    rating_result["postprocess_profile"] = (
        "chemistry_core12_teacher_distribution_v3_severe_zero_production"
    )
    rating_result["postprocess_writeback_enabled"] = (
        CHEMISTRY_ENABLE_LEVEL_WRITEBACK
        or CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK
        or CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
    )
    rating_result["general_level_writeback_enabled"] = (
        CHEMISTRY_ENABLE_LEVEL_WRITEBACK
    )
    rating_result["final_boundary_guard_enabled"] = (
        CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD
    )
    rating_result["final_boundary_guard_writeback_enabled"] = (
        CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK
    )
    rating_result["teacher_distribution_guard_enabled"] = (
        CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS
    )
    rating_result["teacher_distribution_guard_writeback_enabled"] = (
        CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
    )
    rating_result["feature_schema_version"] = "chemistry_core12_strict_v1"
    rating_result["schema_validation_passed"] = True
    features = rating_result["features"]
    medium_evidence = core12_medium_evidence(features)
    basic_evidence = core12_basic_application_evidence(features)
    complete_model_evidence = core12_complete_model_evidence(features)
    high_evidence = core12_high_evidence(features)
    decisive_evidence = core12_decisive_evidence(features)
    final_evidence = core12_final_evidence(features)
    depth = CORE12_DEPTH_ORDER[features["reasoning_depth"]]
    candidate_result = copy.deepcopy(rating_result)
    existing_final_guard_signal = bool(
        depth >= 3
        and len(final_evidence) >= 4
        and (
            features["subquestion_dependency"]
            == "多问存在结果或任务链依赖"
            or (
                features["reaction_relation"]
                in {
                    "多反应连续转化",
                    "先后、竞争或过量不足",
                    "需要分情况判断的反应模型",
                }
                and features["calculation_model"]
                in {
                    "单一守恒或多反应计算",
                    "多重守恒、差量、联立或分类",
                }
            )
        )
    )
    complex_final_guard_signal = bool(
        depth >= 3
        and len(final_evidence) >= 3
        and (
            features["calculation_model"]
            == "多重守恒、差量、联立或分类"
            or features["representation_conversion"]
            == "宏观-微观-符号-定量多重转换"
        )
    )
    final_ceiling_reason = (
        final_promotion_ceiling_reason(features, data)
        if (
            raw_level == "拔高题"
            and (
                existing_final_guard_signal
                or complex_final_guard_signal
            )
        )
        else ""
    )
    rating_result["final_promotion_ceiling_reason"] = (
        final_ceiling_reason
    )

    if raw_level == "送分题":
        # 教师591题实跑中，旧规则把“单一约束”和“单一证据直接对应”
        # 等弱描述叠加为升档证据，3次触发全部把正确送分题误升为基础题。
        # 熟悉类别的一条固定规则直接判断也可能合法表现为1层或单一证据，
        # 因而此边界只审计，不自动写回。
        pass
    elif raw_level == "基础题":
        # Core-12全低可能是真送分，也可能是模型漏识别复合任务。
        # 历史591题回放中该自动降档无净收益，因此只审计、不写回。
        if depth >= 2 and complete_model_evidence:
            set_level_with_reason(
                candidate_result,
                "中等题",
                "自动升档：形成2—3层以上完整常规化学模型",
                rule="core12_basic_to_medium_complete_model",
                evidence=[
                    f"推理深度={features['reasoning_depth']}",
                    *complete_model_evidence[:3],
                ],
            )
    elif raw_level == "中等题":
        if depth <= 1 and not complete_model_evidence:
            set_level_with_reason(
                candidate_result,
                "基础题",
                "自动降档：仅有0—1层显性应用且无完整常规模型",
                rule="core12_medium_to_basic_low_structure",
                evidence=[
                    f"推理深度={features['reasoning_depth']}",
                    "缺少联合证据、完整实验流程或完整计算模型",
                ],
            )
        elif (
            depth >= 3
            and len(decisive_evidence) >= 2
        ):
            set_level_with_reason(
                candidate_result,
                "拔高题",
                "自动升档：4—5层链中存在至少两项决定性转换证据",
                rule="core12_medium_to_hard_decisive_transform",
                evidence=[
                    f"推理深度={features['reasoning_depth']}",
                    *decisive_evidence[:4],
                ],
            )
    elif raw_level == "拔高题":
        if depth <= 2 and not decisive_evidence:
            set_level_with_reason(
                candidate_result,
                "中等题",
                "自动降档：常规2—3层正向模型且无决定性高阶卡点",
                rule="core12_hard_to_medium_routine_model",
                evidence=medium_evidence[:4],
            )
        elif (
            not final_ceiling_reason
            and (
                depth >= 4
                or (
                    CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD
                    and depth >= 3
                )
            )
            and existing_final_guard_signal
        ):
            set_level_with_reason(
                candidate_result,
                "压轴题",
                (
                    "实验保护升档：4—5层以上主链中至少四项压轴证据"
                    "通过任务依赖或反应—计算关系耦合"
                    if depth == 3
                    else
                    "自动升档：6层以上深链中多个高阶任务相互改变模型"
                ),
                rule=(
                    "core12_hard_to_final_4_5_coupled_guard"
                    if depth == 3
                    else
                    "core12_hard_to_final_coupled_chain"
                ),
                evidence=final_evidence[:6],
            )
    elif raw_level == "压轴题":
        if depth <= 3 and len(final_evidence) < 2:
            set_level_with_reason(
                candidate_result,
                "拔高题",
                "自动降档：题面明确只形成有限常规链，未达到压轴耦合",
                rule="core12_final_to_hard_clear_low_density",
                evidence=[
                    f"推理深度={features['reasoning_depth']}",
                    *high_evidence[:3],
                ],
            )

    candidate_actions = copy.deepcopy(
        candidate_result.get("postprocess_trace", [])
    )
    final_boundary_guard_action = next(
        (
            action
            for action in candidate_actions
            if action.get("rule")
            == "core12_hard_to_final_4_5_coupled_guard"
        ),
        None,
    )
    final_boundary_guard_candidate_level = (
        candidate_result.get("difficulty_level", raw_level)
        if final_boundary_guard_action
        else raw_level
    )

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
            and features["experiment_requirement"] == "基础操作或读数"
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "基础题",
                "结构边界窄校准：基础实验操作或读数属于规则应用，不是纯直接检索",
                rule="teacher_easy_to_basic_experiment_application",
                evidence=[
                    f"实验要求={features['experiment_requirement']}",
                ],
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
            and controllable_gas_scheme_signal(features, data)
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "严重低估安全底线：可控气体发生装置需对多组方案进行双约束筛选",
                rule="teacher_basic_to_medium_controllable_gas_scheme_floor",
                evidence=[
                    "题面同时出现可随开随停装置和四组以上制气方案",
                    f"约束复杂度={features['constraint_complexity']}",
                    f"实验要求={features['experiment_requirement']}",
                ],
            )
        elif (
            raw_level == "基础题"
            and cross_module_knowledge_breadth_signal(features, data)
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "严重低估安全底线：跨多个生活板块的大量独立判断形成真实任务广度",
                rule="teacher_basic_to_medium_cross_module_breadth_floor",
                evidence=[
                    "题面包含三个以上知识板块和八项以上独立判断",
                    f"知识关系={features['knowledge_relation']}",
                    f"小问关系={features['subquestion_dependency']}",
                ],
            )
        elif (
            raw_level == "基础题"
            and depth >= 1
            and features["constraint_complexity"]
            == "多个相互关联约束"
            and features["knowledge_relation"] != "单一知识点"
            and features["representation_conversion"] != "无"
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "中等题",
                "结构边界窄校准：关联约束、知识关系与表征转换共同形成完整常规模型",
                rule="teacher_basic_to_medium_linked_application",
                evidence=[
                    f"推理深度={features['reasoning_depth']}",
                    f"约束复杂度={features['constraint_complexity']}",
                    f"知识关系={features['knowledge_relation']}",
                    f"表征转换={features['representation_conversion']}",
                ],
            )
        elif (
            raw_level == "中等题"
            and coordinated_multigraph_reaction_signal(features, data)
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "严重低估安全底线：三幅以上关联图像共同描述同一反应进程",
                rule="teacher_medium_to_hard_coordinated_multigraph_floor",
                evidence=[
                    f"图表要求={features['graph_table_requirement']}",
                    f"表征转换={features['representation_conversion']}",
                    f"约束复杂度={features['constraint_complexity']}",
                    f"证据关系={features['evidence_relation']}",
                ],
            )
        elif (
            raw_level == "中等题"
            and depth >= 2
            and dense_project_experiment_signal(features, data)
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "严重低估安全底线：四个以上实验形成变量、证据和方案的项目链",
                rule="teacher_medium_to_hard_dense_project_floor",
                evidence=[
                    f"推理深度={features['reasoning_depth']}",
                    f"实验要求={features['experiment_requirement']}",
                    f"约束复杂度={features['constraint_complexity']}",
                    f"证据关系={features['evidence_relation']}",
                ],
            )
        elif (
            raw_level == "中等题"
            and multi_activity_project_signal(features, data)
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "严重低估安全底线：多活动、多实验共同形成变量与证据项目链",
                rule="teacher_medium_to_hard_multi_activity_project_floor",
                evidence=[
                    "题面包含至少两项活动和三项实验",
                    f"实验要求={features['experiment_requirement']}",
                    f"约束复杂度={features['constraint_complexity']}",
                ],
            )
        elif (
            raw_level == "中等题"
            and depth >= 2
            and strong_segment_graph_chain_signal(features)
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "结构边界窄校准：分段图像与跨模块表征、约束和实验证据共同建模",
                rule="teacher_medium_to_hard_strong_graph_chain",
                evidence=[
                    f"推理深度={features['reasoning_depth']}",
                    f"图表要求={features['graph_table_requirement']}",
                    f"知识关系={features['knowledge_relation']}",
                    f"表征转换={features['representation_conversion']}",
                    f"约束复杂度={features['constraint_complexity']}",
                    f"实验要求={features['experiment_requirement']}",
                ],
            )
        elif (
            raw_level == "中等题"
            and depth >= 2
            and shared_new_information_signal(features)
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "拔高题",
                "结构边界窄校准：给定新信息被多个小问在同一深层模型中共同使用",
                rule="teacher_medium_to_hard_shared_new_information",
                evidence=[
                    f"推理深度={features['reasoning_depth']}",
                    "陌生信息迁移="
                    + features["unfamiliar_information_transfer"],
                    f"小问关系={features['subquestion_dependency']}",
                    f"知识关系={features['knowledge_relation']}",
                ],
            )
        elif (
            raw_level == "拔高题"
            and not final_ceiling_reason
            and existing_final_guard_signal
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "压轴题",
                "结构边界窄校准：4—5层以上主链中至少四项压轴证据形成任务耦合",
                rule="core12_hard_to_final_4_5_coupled_guard",
                evidence=final_evidence[:6],
            )
        elif (
            raw_level == "拔高题"
            and not final_ceiling_reason
            and complex_final_guard_signal
        ):
            set_level_with_reason(
                teacher_candidate_result,
                "压轴题",
                "结构边界窄校准：4—5层复杂主模型由多重定量或四重表征转换闭合",
                rule="teacher_hard_to_final_complex_model",
                evidence=[
                    f"推理深度={features['reasoning_depth']}",
                    *final_evidence[:6],
                ],
            )

    teacher_candidate_actions = copy.deepcopy(
        teacher_candidate_result.get("postprocess_trace", [])
    )
    if len(teacher_candidate_actions) > 1:
        raise RuntimeError("教师分布窄校准违反每题单次调整约束")
    teacher_guard_action = (
        teacher_candidate_actions[0]
        if teacher_candidate_actions
        else None
    )
    teacher_guard_candidate_level = (
        teacher_candidate_result.get("difficulty_level", raw_level)
        if teacher_guard_action
        else raw_level
    )

    rating_result["postprocess_candidate_actions"] = candidate_actions
    rating_result["postprocess_candidate_level"] = candidate_result.get(
        "difficulty_level",
        raw_level,
    )
    general_writeback_applied = bool(
        CHEMISTRY_ENABLE_LEVEL_WRITEBACK and candidate_actions
    )
    final_guard_writeback_applied = bool(
        CHEMISTRY_ENABLE_FINAL_BOUNDARY_GUARD_WRITEBACK
        and final_boundary_guard_action
    )
    teacher_guard_writeback_applied = bool(
        CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
        and teacher_guard_action
    )
    writeback_applied = bool(
        general_writeback_applied
        or final_guard_writeback_applied
        or teacher_guard_writeback_applied
    )
    if writeback_applied:
        rating_result = (
            teacher_candidate_result
            if teacher_guard_writeback_applied
            else candidate_result
        )
        rating_result["postprocess_writeback_enabled"] = True
        rating_result["postprocess_candidate_actions"] = (
            candidate_actions
        )
        rating_result["postprocess_candidate_level"] = (
            candidate_result["difficulty_level"]
        )
        sync_coarse_difficulty(rating_result)

    rating_result["coarse_difficulty_final"] = rating_result[
        "coarse_difficulty"
    ]
    rating_result["final_boundary_guard_candidate_level"] = (
        final_boundary_guard_candidate_level
    )
    rating_result["final_boundary_guard_candidate_action"] = (
        copy.deepcopy(final_boundary_guard_action)
        if final_boundary_guard_action
        else None
    )
    rating_result["final_boundary_guard_writeback_applied"] = (
        final_guard_writeback_applied
    )
    rating_result["teacher_distribution_guard_candidate_level"] = (
        teacher_guard_candidate_level
    )
    rating_result["teacher_distribution_guard_candidate_action"] = (
        copy.deepcopy(teacher_guard_action)
        if teacher_guard_action
        else None
    )
    rating_result["teacher_distribution_guard_writeback_applied"] = (
        teacher_guard_writeback_applied
    )
    rating_result["final_promotion_ceiling_reason"] = (
        final_ceiling_reason
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
    rating_result["postprocess_evidence_counts"] = {
        "basic_application": len(basic_evidence),
        "complete_model": len(complete_model_evidence),
        "decisive_transform": len(decisive_evidence),
        "final_coupling": len(final_evidence),
    }
    add_feature_audit_flags(rating_result, data)
    if candidate_actions and not CHEMISTRY_ENABLE_LEVEL_WRITEBACK:
        rating_result["feature_audit_flags"].append(
            "存在通用候选校准，但自动写回默认关闭；"
            "仅记录postprocess_candidate_actions"
        )
    if teacher_guard_action and not teacher_guard_writeback_applied:
        rating_result["feature_audit_flags"].append(
            "存在结构边界窄校准候选，但专用写回默认关闭；"
            "仅记录teacher_distribution_guard_candidate_action"
        )
    if final_ceiling_reason:
        rating_result["feature_audit_flags"].append(
            "压轴升档被客观低密度上限阻止：" + final_ceiling_reason
        )
    return rating_result


def infer_level_from_features(features: Dict[str, Any], data: Dict[str, Any]) -> str:
    depth = CORE12_DEPTH_ORDER[features["reasoning_depth"]]
    final = len(core12_final_evidence(features))
    decisive = len(core12_decisive_evidence(features))
    complete = len(core12_complete_model_evidence(features))
    if depth >= 4 and final >= 4:
        return "压轴题"
    if depth >= 3 and decisive >= 2:
        return "拔高题"
    if depth >= 2 and complete:
        return "中等题"
    if core12_is_direct_retrieval(features):
        return "送分题"
    return "基础题"

# -------------------------- 5. 构建题目输入与模型调用 --------------------------
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
            repair_part = {
                "type": "input_text",
                "text": (
                    "【上次输出修复要求】\n"
                    + repair_feedback
                    + "\n保持实质难度判断不变，只修复缺失字段、非法枚举或JSON格式。"
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
        except aiohttp.ClientError as e:
            backoff = (2 ** retry) + random.uniform(0, 1)
            if retry == retries - 1:
                print(f"网络异常最终失败: {e}")
                return {}, "", 0.0, 0, 0, 0, image_status
            print(f"网络出现异常: {e}，将进行退避 {backoff:.2f} 秒后重试...")
            await asyncio.sleep(backoff)
        except Exception as e:
            print(f"运行过程中请求异常: {e}")
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
        question_input = make_output_base(data)
        total_time = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        schema_retry_count = 0
        schema_errors: List[str] = []
        repair_feedback = ""
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
                    question_input,
                    session,
                    retries,
                    timeout_sec,
                    repair_feedback=repair_feedback,
                )
                raw_result = copy.deepcopy(candidate)
                total_time += time_use
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += call_tokens

                try:
                    if not candidate:
                        raise ChemistrySchemaError("模型返回空对象或JSON解析失败")
                    rating_result = postprocess_chemistry_difficulty(
                        candidate,
                        question_input,
                    )
                except ChemistrySchemaError as exc:
                    schema_errors.append(str(exc))
                    if schema_retry_count >= MAX_SCHEMA_RETRIES:
                        raise RuntimeError(
                            f"schema校验重试耗尽({MAX_SCHEMA_RETRIES}): {exc}"
                        ) from exc
                    schema_retry_count += 1
                    repair_feedback = f"上次输出未通过Core-12 schema：{exc}"
                    continue

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
                output_data["difficulty_rating"] = rating_result
                output_data["api_time_use"] = round(total_time, 2)
                output_data["api_prompt_tokens"] = total_prompt_tokens
                output_data["api_completion_tokens"] = total_completion_tokens
                output_data["api_total_tokens"] = total_tokens
                output_data.update(image_status)
                output_data["schema_retry_count"] = schema_retry_count
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
