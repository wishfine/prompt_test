# -*- coding: utf-8 -*-
"""初中物理难度批量评级。

外围能力保持兼容：OpenAI-compatible Responses API、前缀缓存、并发、重试、
断点续跑、JSONL 输入输出和既有命令行参数均保留。评级规则集中在本文件的
“后处理规则”一节，避免历史 V5/V6/V7 同名函数互相覆盖。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import re
import sys
import time
from asyncio import Lock, Semaphore
from typing import Any, Dict, List, Optional, Tuple

import aiofiles
import aiohttp
import json_repair
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

load_dotenv()

API_KEY = os.getenv("API_KEY", "not-needed")
BASE_URL = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1")
if not BASE_URL.endswith("/"):
    BASE_URL += "/"
MODEL_NAME = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")
VALID_RATING_PROFILES = {"fused", "generalized", "v7_compat", "v7_stable"}
RATING_PROFILE = (os.getenv("RATING_PROFILE", "v7_stable").strip().lower() or "v7_stable")
if RATING_PROFILE not in VALID_RATING_PROFILES:
    raise ValueError(
        f"不支持的 RATING_PROFILE={RATING_PROFILE!r}；"
        f"可选值：{', '.join(sorted(VALID_RATING_PROFILES))}"
    )


def resolve_temperature(model_name: str, raw_value: str) -> Optional[float]:
    """Lite 服务端固定使用 temperature=1，其他模型保留环境变量配置。"""
    if "lite" in str(model_name).lower():
        return 1.0
    value = str(raw_value or "").strip()
    return float(value) if value else None


TEMPERATURE = resolve_temperature(MODEL_NAME, os.getenv("TEMPERATURE", ""))

FILE_LOCK = Lock()
CACHE_LOCK = Lock()
CACHE_GET_LOCK = Lock()
USE_CACHE = True
CACHE_FILE_PATH = "physics_prompt_cache.json"
CACHE_EXPIRE_SECONDS = 6 * 24 * 3600

DIFFICULTY_RATING_PROMPT_PREFIX = ""
DIFFICULTY_RATING_PROMPT_SUFFIX = ""
DEFAULT_PROMPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "prompts", "初中物理难度打标提示词.txt")
)

LEVEL_MAP = {"送分题": 1, "基础题": 2, "中等题": 3, "拔高题": 4, "压轴题": 5}
LEVEL_NAMES = set(LEVEL_MAP.values())

# 保持历史 18 维字段及 step_count 枚举不变。
FEATURE_DEFAULTS = {
    "step_count": "1-2步",
    "formula_count": "0-1个",
    "calculation_complexity": "口算或直接判断",
    "reasoning_chain": "直接套用",
    "problem_structure": "概念判断",
    "additional_structure": "无",
    "information_carrier": "纯文字",
    "reality_question": "否",
    "subquestion_dependency": "无多问",
    "knowledge_count": "1个",
    "knowledge_diff": "低",
    "cross_module": "同一模块内部",
    "state_count": "单状态",
    "constraint_count": "无约束",
    "variable_relation": "无变量关系",
    "experiment_requirement": "无",
    "graph_table_requirement": "无",
    "error_risk": "无明显易错点",
}

ALLOWED_FEATURE_VALUES = {
    "step_count": {"1-2步", "3-5步", "6-8步", "9-12步", "12步以上"},
    "formula_count": {"0-1个", "2-3个", "4-6个", "7个以上"},
    "calculation_complexity": {"口算或直接判断", "简单笔算", "多公式联立", "复杂方程或范围计算"},
    "reasoning_chain": {"直接套用", "简单因果推理", "多层因果推理", "逆向推理或临界分析"},
    "problem_structure": {"概念判断", "直接计算", "实验探究", "图像表格分析", "电路综合", "力学综合", "热学综合", "光学声学综合", "跨模块综合"},
    "additional_structure": {"无", "图像表格", "实验探究", "电路约束", "力学约束", "跨模块"},
    "information_carrier": {"纯文字", "单图识别", "电路图", "实验装置图", "图像或表格", "多图表综合"},
    "reality_question": {"是", "否"},
    "subquestion_dependency": {"无多问", "多问但相互独立", "多问且层层递进"},
    "knowledge_count": {"1个", "2-3个", "4个及以上"},
    "knowledge_diff": {"低", "中", "高"},
    "cross_module": {"同一模块内部", "跨模块综合"},
    "state_count": {"单状态", "双状态", "多状态", "连续变化或临界状态"},
    "constraint_count": {"无约束", "单一约束", "多约束"},
    "variable_relation": {"无变量关系", "简单正反比", "图像函数关系", "多变量耦合关系"},
    "experiment_requirement": {"无", "基础操作或读数", "控制变量或故障分析", "方案设计或误差评价"},
    "graph_table_requirement": {"无", "直接读数", "多组比较归纳", "图像反推或外推"},
    "error_risk": {"无明显易错点", "轻微易错点", "明显易错点", "高易错点"},
}


# -------------------------- Prompt / cache --------------------------
def load_prompt_config(prompt_path: str) -> None:
    global DIFFICULTY_RATING_PROMPT_PREFIX, DIFFICULTY_RATING_PROMPT_SUFFIX
    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"找不到提示词文件: {prompt_path}")
    content = open(prompt_path, "r", encoding="utf-8").read()
    namespace: Dict[str, Any] = {}
    try:
        exec(content, namespace)
    except Exception:
        namespace = {}
    prefix = namespace.get("DIFFICULTY_RATING_PROMPT_PREFIX")
    suffix = namespace.get("DIFFICULTY_RATING_PROMPT_SUFFIX")
    if prefix and suffix:
        DIFFICULTY_RATING_PROMPT_PREFIX = str(prefix)
        DIFFICULTY_RATING_PROMPT_SUFFIX = str(suffix)
        print("成功以 Python 变量结构解析提示词")
        return
    if "## 输入题目信息" in content:
        parts = content.split("## 输入题目信息", 1)
        DIFFICULTY_RATING_PROMPT_PREFIX = parts[0] + "## 输入题目信息"
        DIFFICULTY_RATING_PROMPT_SUFFIX = "\n\n请根据以上信息，对题目进行全面的难度分析和评级。"
        print("成功以纯文本标志位结构解析提示词")
        return
    raise ValueError("提示词必须包含 Python 变量或 '## 输入题目信息' 标志")


def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def load_cache() -> Dict[str, Any]:
    async with CACHE_LOCK:
        if not os.path.exists(CACHE_FILE_PATH):
            return {}
        try:
            async with aiofiles.open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
                text = await f.read()
            return json.loads(text) if text else {}
        except Exception as exc:
            print(f"加载缓存文件失败: {exc}")
            return {}


async def save_cache(data: Dict[str, Any]) -> None:
    async with CACHE_LOCK:
        async with aiofiles.open(CACHE_FILE_PATH, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))


def is_cache_valid(entry: Optional[Dict[str, Any]], now: int) -> bool:
    return bool(
        entry
        and now < int(entry.get("expire_at", 0))
        and entry.get("prefix_hash") == compute_text_hash(DIFFICULTY_RATING_PROMPT_PREFIX)
        and entry.get("model_name") == MODEL_NAME
    )


async def get_valid_cache() -> Optional[Dict[str, Any]]:
    data = await load_cache()
    entry = data.get("prompt_prefix_cache")
    return entry if is_cache_valid(entry, int(time.time())) else None


async def set_cache(response_id: str, expire_at: int) -> None:
    data = await load_cache()
    data["prompt_prefix_cache"] = {
        "response_id": response_id,
        "expire_at": expire_at,
        "prefix_hash": compute_text_hash(DIFFICULTY_RATING_PROMPT_PREFIX),
        "model_name": MODEL_NAME,
        "created_at": int(time.time()),
    }
    await save_cache(data)


async def create_prefix_cache(session: aiohttp.ClientSession, retries: int, timeout_sec: int) -> Optional[str]:
    expire_at = int(time.time()) + CACHE_EXPIRE_SECONDS
    payload = {
        "model": MODEL_NAME,
        "input": [{"role": "user", "content": DIFFICULTY_RATING_PROMPT_PREFIX}],
        "thinking": {"type": "disabled"},
        "expire_at": expire_at,
        "caching": {"type": "enabled", "prefix": True},
    }
    for attempt in range(retries):
        try:
            async with session.post(
                f"{BASE_URL}responses", json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    response_id = result.get("id")
                    if response_id:
                        await set_cache(response_id, expire_at)
                        print(f"前缀缓存创建成功，缓存ID: {response_id}")
                        return response_id
                else:
                    text = await response.text()
                    print(f"创建前缀缓存失败 (状态码: {response.status}): {text[:200]}")
                    if 400 <= response.status < 500:
                        return None
        except Exception as exc:
            if attempt == retries - 1:
                print(f"创建前缀缓存最终失败: {exc}")
                return None
            await asyncio.sleep(2 ** attempt + random.random())
    return None


async def get_or_create_cache(session: aiohttp.ClientSession, retries: int, timeout_sec: int) -> Optional[str]:
    async with CACHE_GET_LOCK:
        entry = await get_valid_cache()
        if entry:
            return entry["response_id"]
        print("未找到有效缓存，正在向服务器创建前缀缓存...")
        return await create_prefix_cache(session, retries, timeout_sec)


# -------------------------- Feature schema --------------------------
def _clean(value: Any) -> str:
    return str(value or "").strip().replace(" ", "").replace("\n", "")


def canonicalize_feature_value(field: str, value: Any) -> str:
    v = _clean(value)
    if not v:
        return FEATURE_DEFAULTS[field]
    if v in ALLOWED_FEATURE_VALUES[field]:
        return v
    if field == "step_count":
        if any(x in v for x in ["12步以上", "12步", "十二"]): return "12步以上"
        if any(x in v for x in ["9-12", "9到12", "九", "十", "11"]): return "9-12步"
        if any(x in v for x in ["6-8", "6到8", "六", "七", "八"]): return "6-8步"
        if any(x in v for x in ["3-5", "3到5", "三", "四", "五"]): return "3-5步"
        return "1-2步"
    if field == "formula_count":
        if "7" in v or "七" in v: return "7个以上"
        if any(x in v for x in ["4-6", "4到6", "四", "五", "六"]): return "4-6个"
        if any(x in v for x in ["2-3", "2到3", "2个", "3个", "二", "两", "三"]): return "2-3个"
        return "0-1个"
    if field == "calculation_complexity":
        if any(x in v for x in ["复杂", "范围", "极值", "不等式", "分类", "方程组"]): return "复杂方程或范围计算"
        if any(x in v for x in ["多公式", "联立", "差值法", "比例法"]): return "多公式联立"
        if any(x in v for x in ["简单", "笔算", "代入", "换算", "计算"]): return "简单笔算"
        return "口算或直接判断"
    if field == "reasoning_chain":
        if any(x in v for x in ["逆向", "临界", "反推", "分类", "极值", "范围"]): return "逆向推理或临界分析"
        if any(x in v for x in ["多层", "多步", "综合", "链条"]): return "多层因果推理"
        if any(x in v for x in ["简单", "因果"]): return "简单因果推理"
        return "直接套用"
    if field == "problem_structure":
        if "跨" in v or sum(any(x in v for x in group) for group in [["电", "电路", "电磁"], ["力", "运动", "浮力", "压强"], ["热", "温度", "内能"], ["光", "声", "透镜"]]) >= 2: return "跨模块综合"
        if any(x in v for x in ["实验", "探究", "测量"]): return "实验探究"
        if any(x in v for x in ["电", "电路", "电磁"]): return "电路综合"
        if any(x in v for x in ["力", "运动", "密度", "压强", "浮力", "杠杆", "滑轮", "机械"]): return "力学综合"
        if any(x in v for x in ["热", "温度", "内能", "物态"]): return "热学综合"
        if any(x in v for x in ["光", "声", "透镜", "平面镜"]): return "光学声学综合"
        if "图" in v or "表" in v: return "图像表格分析"
        if "计算" in v: return "直接计算"
        return "概念判断"
    if field == "additional_structure":
        if any(x in v for x in ["实验", "探究", "装置", "控制", "故障", "误差"]): return "实验探究"
        if any(x in v for x in ["电路", "电压", "电流", "电表", "量程"]): return "电路约束"
        if any(x in v for x in ["力", "压强", "浮力", "杠杆", "滑轮", "机械"]): return "力学约束"
        if any(x in v for x in ["图像", "表格", "图表"]): return "图像表格"
        if "跨" in v: return "跨模块"
        return "无"
    if field == "information_carrier":
        if any(x in v for x in ["多图", "多表", "综合"]): return "多图表综合"
        if "电路" in v: return "电路图"
        if "装置" in v: return "实验装置图"
        if any(x in v for x in ["图像", "表格", "曲线"]): return "图像或表格"
        if "图" in v: return "单图识别"
        return "纯文字"
    if field == "reality_question": return "是" if "是" in v or v.lower() in {"true", "yes", "1"} else "否"
    if field == "subquestion_dependency":
        if any(x in v for x in ["层层", "递进", "依赖", "承接"]): return "多问且层层递进"
        if any(x in v for x in ["多问", "小问", "独立"]): return "多问但相互独立"
        return "无多问"
    if field == "knowledge_count":
        if any(x in v for x in ["4个", "4个以上", "四个", "多个"]): return "4个及以上"
        if any(x in v for x in ["2", "3", "二", "三", "两"]): return "2-3个"
        return "1个"
    if field == "knowledge_diff": return "高" if any(x in v for x in ["高", "难", "复杂"]) else ("低" if any(x in v for x in ["低", "简单", "基础"]) else "中")
    if field == "cross_module": return "跨模块综合" if "跨" in v else "同一模块内部"
    if field == "state_count":
        if any(x in v for x in ["连续", "临界", "动态变化"]): return "连续变化或临界状态"
        if any(x in v for x in ["多状态", "三状态", "3状态", "三个状态", "3个状态"]): return "多状态"
        if any(x in v for x in ["双状态", "两状态", "两个"]): return "双状态"
        return "单状态"
    if field == "constraint_count": return "多约束" if "多" in v else ("单一约束" if "约束" in v else "无约束")
    if field == "variable_relation":
        if any(x in v for x in ["多变量", "耦合"]): return "多变量耦合关系"
        if any(x in v for x in ["函数", "图像", "曲线"]): return "图像函数关系"
        if any(x in v for x in ["正比", "反比", "比例"]): return "简单正反比"
        return "无变量关系"
    if field == "experiment_requirement":
        if any(x in v for x in ["方案", "设计", "误差", "评价", "可行", "改进"]): return "方案设计或误差评价"
        if any(x in v for x in ["控制变量", "故障", "归纳", "探究", "分析"]): return "控制变量或故障分析"
        if any(x in v for x in ["读数", "操作", "测量"]): return "基础操作或读数"
        return "无"
    if field == "graph_table_requirement":
        if any(x in v for x in ["反推", "外推", "函数"]): return "图像反推或外推"
        if any(x in v for x in ["多组", "比较", "归纳"]): return "多组比较归纳"
        if any(x in v for x in ["读数", "读取", "描点", "作图"]): return "直接读数"
        return "无"
    if field == "error_risk": return "高易错点" if "高" in v else ("明显易错点" if "明显" in v else ("轻微易错点" if "轻微" in v else "无明显易错点"))
    return FEATURE_DEFAULTS[field]


def normalize_features(features: Any) -> Dict[str, str]:
    source = features if isinstance(features, dict) else {}
    normalized: Dict[str, str] = {}
    for field in FEATURE_DEFAULTS:
        value = source.get(field, FEATURE_DEFAULTS[field])
        normalized[field] = canonicalize_feature_value(field, value)
        if normalized[field] not in ALLOWED_FEATURE_VALUES[field]:
            normalized[field] = FEATURE_DEFAULTS[field]
    return normalized


def normalize_reasoning_schema(result: Dict[str, Any]) -> None:
    reasoning = result.get("reasoning")
    if not isinstance(reasoning, dict):
        reasoning = {}
    result["reasoning"] = {k: str(reasoning.get(k, "")) for k in ["core_basis", "hard_point", "why_not_lower", "why_not_higher"]}


def extract_raw_difficulty_level(result: Any) -> Optional[str]:
    """读取模型原始等级，并兼容模型偶发地把等级放进 reasoning。

    顶层字段始终优先；函数只读，不修改原始模型结果，便于保留完整审计证据。
    """
    if not isinstance(result, dict):
        return None
    top_level = result.get("difficulty_level")
    if top_level in LEVEL_MAP:
        return str(top_level)
    reasoning = result.get("reasoning")
    if isinstance(reasoning, dict):
        nested = reasoning.get("difficulty_level")
        if nested in LEVEL_MAP:
            return str(nested)
    return None


def full_text_of(data: Dict[str, Any]) -> str:
    parts = [str(data.get("stem", "") or ""), str(data.get("options", "") or ""), str(data.get("analysis", "") or "")]
    for item in data.get("sub_questions", []) or []:
        if isinstance(item, dict):
            parts.extend(str(item.get(k, "") or "") for k in ["stem", "options", "analysis"])
        else:
            parts.append(str(item))
    return "\n".join(parts)


def visible_text_of(data: Dict[str, Any]) -> str:
    return "\n".join([str(data.get("stem", "") or ""), str(data.get("options", "") or "")])


def contains_any(text: str, words: List[str]) -> bool:
    return any(word in text for word in words)


def has_formula_intent(text: str) -> bool:
    return bool(re.search(r"(求|计算|多少|多大|等于|取值|质量|电流|电压|电阻|功率|压强|浮力|密度).{0,30}(\d|公式|多少|多大|为)", text)) or "=" in text


def count_subquestions(data: Dict[str, Any]) -> int:
    return len(data.get("sub_questions", []) or [])


def is_parallel_choice_or_independent_points(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    text = visible_text_of(data)
    independent = features.get("subquestion_dependency") == "多问但相互独立"
    choice = bool(data.get("options")) and features.get("reasoning_chain") in ["直接套用", "简单因果推理"] and features.get("state_count") in ["单状态", "双状态"]
    return independent or (choice and features.get("constraint_count") in ["无约束", "单一约束"] and features.get("variable_relation") in ["无变量关系", "简单正反比"] and len(text) < 500)


def is_standard_rule_diagram_task(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """教材规则完全显性的单一作图/标注任务，保护在基础档。

    不依赖具体章节或装置名，只判断任务结构是否为无计算、无实验数据、
    无状态/约束分析的顺向规则应用。
    """
    text = visible_text_of(data)
    has_diagram_action = contains_any(text, ["画出", "作出", "作图", "标出", "连接", "补全", "方向"])
    if not has_diagram_action or count_subquestions(data) > 1 or has_formula_intent(text):
        return False
    return (
        features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") in ["无", "直接读数"]
    )


def is_textbook_easy_diagram_or_direct_fill(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    text = visible_text_of(data)
    if has_formula_intent(text) or count_subquestions(data) > 1:
        return False
    if contains_any(text, ["完整", "画出", "作出", "连接", "光路", "平面镜成像", "螺线管", "磁感线", "受力示意图"]):
        # 仅保留单一方向/名称识别；完整规范作图至少基础。
        return contains_any(text, ["重力方向", "竖直向下", "特殊光线名称"])
    return features.get("knowledge_count") == "1个" and features.get("reasoning_chain") == "直接套用" and features.get("experiment_requirement") == "无" and features.get("constraint_count") == "无约束"


def should_downgrade_basic_to_easy(features: Dict[str, str], data: Dict[str, Any]) -> bool:
    """不以弱特征把模型的基础题自动降送分。

    这类降档在 133 题回归中把“地理方位判断”等空间图示题误降为送分。
    送分边界交给 Prompt 的真实语义判断；后处理只负责阻止明显错误升档。
    """
    return False


def should_upgrade_easy_to_basic(features: Dict[str, str], data: Dict[str, Any]) -> List[str]:
    reasons: List[str] = []
    text = visible_text_of(data)
    if has_formula_intent(text): reasons.append("需要公式或物理量计算")
    if features.get("subquestion_dependency") == "多问且层层递进": reasons.append("多问构成递进推理链")
    # 只识别明确的任务动作，不能因“透镜”“电路”等学科名词就否定送分。
    if contains_any(text, ["画出", "作出", "作图", "连接完整", "连接电路", "实验步骤", "探究过程"]):
        reasons.append("涉及规范作图、连接或实验过程")
    if features.get("experiment_requirement") != "无": reasons.append("需要实验操作或分析")
    if features.get("knowledge_count") != "1个" and features.get("reasoning_chain") != "直接套用":
        reasons.append("多个知识点构成应用推理")
    if features.get("reality_question") == "是" and features.get("reasoning_chain") != "直接套用":
        reasons.append("生活情境需要物理映射")
    return reasons


def core_high_signals(features: Dict[str, str], data: Dict[str, Any]) -> List[str]:
    signals: List[str] = []
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]: signals.append("多状态或连续临界状态")
    if features.get("constraint_count") == "多约束": signals.append("多约束")
    if features.get("cross_module") == "跨模块综合": signals.append("跨章节综合")
    if features.get("reasoning_chain") == "逆向推理或临界分析": signals.append("逆向推理或临界分析")
    if features.get("experiment_requirement") == "方案设计或误差评价": signals.append("方案设计或误差评价")
    if features.get("graph_table_requirement") == "图像反推或外推": signals.append("图像反推或外推")
    if features.get("calculation_complexity") == "多公式联立": signals.append("方程联立/比例法/差值法")
    if features.get("calculation_complexity") == "复杂方程或范围计算": signals.append("复杂方程或范围计算")
    if features.get("variable_relation") == "多变量耦合关系": signals.append("多变量耦合")
    if features.get("subquestion_dependency") == "多问且层层递进": signals.append("多问层层递进")
    return signals


def strong_migration_signals(features: Dict[str, str], data: Dict[str, Any]) -> List[str]:
    """中等升拔高所需的强迁移信号；不用题干长度或装置名凑信号。"""
    signals: List[str] = []
    if features.get("cross_module") == "跨模块综合": signals.append("真实跨模块迁移")
    if features.get("reasoning_chain") == "逆向推理或临界分析": signals.append("逆向或临界推理")
    if features.get("graph_table_requirement") == "图像反推或外推": signals.append("图像反推或外推")
    if features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"]: signals.append("多式联立或复杂计算")
    if features.get("variable_relation") == "多变量耦合关系": signals.append("多变量耦合")
    return signals


def strong_final_signals(data: Dict[str, Any], features: Dict[str, str]) -> List[str]:
    text = full_text_of(data)
    signals: List[str] = []
    if contains_any(text, ["分类讨论", "多解", "筛选有效解", "有效解"]): signals.append("分类讨论/多解筛选")
    if "不等式" in text: signals.append("不等式")
    if contains_any(text, ["边界覆盖", "边界条件", "可行性验证", "可行性"]): signals.append("边界覆盖或可行性验证")
    if "极值" in text or (features.get("reasoning_chain") == "逆向推理或临界分析" and features.get("constraint_count") == "多约束"):
        signals.append("临界极值或物理条件筛选")
    if features.get("experiment_requirement") == "方案设计或误差评价" and contains_any(text, ["方案比较", "标尺", "量程设计"]): signals.append("开放设计的方案比较/量程设计")
    if features.get("variable_relation") == "多变量耦合关系" and features.get("calculation_complexity") == "复杂方程或范围计算": signals.append("复杂多变量耦合")
    return signals


def is_project_or_control_case(data: Dict[str, Any]) -> bool:
    return contains_any(full_text_of(data), ["项目", "任务", "实践", "传感器", "热敏电阻", "压敏电阻", "电磁继电器", "自动控制", "自动控温"])


def has_strong_project_validation(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    text = full_text_of(data)
    allowed = ["可行性", "边界覆盖", "分类讨论", "多解", "不等式", "方案比较", "标尺", "量程设计", "筛选有效解", "有效解"]
    return contains_any(text, allowed)


def should_upgrade_basic_to_medium(features: Dict[str, str], data: Dict[str, Any]) -> List[str]:
    if is_standard_rule_diagram_task(data, features):
        return []
    # 单图方向标注、知识结构补空、基础连接题等，即使模型把步骤写成 3-5 步，
    # 也不能仅凭“知识点数 + 读图”升为中等。
    direct_visual = (
        features.get("information_carrier") == "单图识别"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") in ["直接套用", "简单因果推理"]
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") in ["无", "直接读数"]
    )
    if direct_visual:
        return []
    strong: List[str] = []
    weak: List[str] = []
    if features.get("experiment_requirement") == "控制变量或故障分析": strong.append("控制变量或故障分析")
    if features.get("graph_table_requirement") == "多组比较归纳": strong.append("多组数据归纳")
    if features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"]: strong.append("多公式联立")
    if features.get("graph_table_requirement") == "图像反推或外推": strong.append("图像反推或外推")
    if features.get("reasoning_chain") == "多层因果推理" and features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        strong.append("两个连续物理过程")
    if features.get("subquestion_dependency") == "多问且层层递进": strong.append("多问层层递进")
    cross_module_candidate = features.get("cross_module") == "跨模块综合"
    if features.get("step_count") == "3-5步": weak.append("3-5步")
    if features.get("knowledge_count") == "2-3个": weak.append("2-3个关联知识点")
    if features.get("state_count") == "双状态": weak.append("双状态")
    if features.get("variable_relation") == "简单正反比": weak.append("简单正反比")
    if features.get("experiment_requirement") == "基础操作或读数": weak.append("基础实验操作或读数")
    if features.get("graph_table_requirement") == "直接读数": weak.append("简单图表读数")
    # “跨模块”容易被模型用于并列基础概念。只有同时有多个过程支撑信号，
    # 或已经存在其他强语义证据时，才认定为“真实跨模块综合”。
    if cross_module_candidate and (strong or len(weak) >= 3):
        strong.append("真实跨模块综合")
    # 至少一个强信号，或两个弱信号且确有多层连续推理；弱特征本身不触发升档。
    # V7 顺序：实验归纳、控制变量等强语义保护先于“独立小问”拦截。
    if strong:
        return strong + weak
    if is_parallel_choice_or_independent_points(data, features) and features.get("subquestion_dependency") != "多问且层层递进":
        return []
    if len(weak) >= 2 and features.get("reasoning_chain") == "多层因果推理":
        return weak
    return []


def should_downgrade_medium_to_basic(features: Dict[str, str], data: Dict[str, Any]) -> bool:
    if is_parallel_choice_or_independent_points(data, features) and features.get("step_count") == "1-2步":
        return True
    return features.get("step_count") == "1-2步" and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"] and len(core_high_signals(features, data)) == 0 and features.get("knowledge_count") == "1个"


def should_upgrade_medium_to_hard(features: Dict[str, str], data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    signals = core_high_signals(features, data)
    strong = strong_migration_signals(features, data)
    step_count = features.get("step_count")
    if step_count == "3-5步":
        # 3-5 步的“多状态/多约束”常见于常规控制电路；必须另有真正迁移信号。
        ok = len(set(signals)) >= 2 and bool(strong)
    elif step_count == "6-8步":
        ok = len(set(signals)) >= 1
    else:
        ok = len(set(signals)) >= 2
    if is_parallel_choice_or_independent_points(data, features) and features.get("subquestion_dependency") != "多问且层层递进":
        ok = False
    return ok, signals + strong


def should_downgrade_hard_to_medium(features: Dict[str, str], data: Dict[str, Any]) -> bool:
    signals = core_high_signals(features, data)
    if features.get("step_count") == "1-2步":
        return len(signals) == 0
    # V7 原则：只有出现明确的常规中等结构才降档；
    # 不能因模型未抽取到高阶特征，就否定 3-5 步的低计算高建模原判。
    return False


def should_upgrade_hard_to_final(features: Dict[str, str], data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    signals = core_high_signals(features, data)
    strong = strong_final_signals(data, features)
    step_count = features.get("step_count")
    if step_count not in ["6-8步", "9-12步", "12步以上"] or len(set(signals)) < 3 or not strong:
        return False, signals + strong
    if is_project_or_control_case(data):
        high_model_count = len(set(signals))
        complex_or_critical = features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"] or features.get("reasoning_chain") == "逆向推理或临界分析" or step_count in ["6-8步", "9-12步", "12步以上"]
        if not (has_strong_project_validation(data, features) and high_model_count >= 2 and complex_or_critical):
            return False, signals + strong
    if is_parallel_choice_or_independent_points(data, features) and features.get("subquestion_dependency") != "多问且层层递进":
        return False, signals + strong
    return True, signals + strong


def should_downgrade_final_to_hard(features: Dict[str, str], data: Dict[str, Any]) -> bool:
    """“不够升压轴”不等于“原判压轴必降”，独立使用降档证据。"""
    signal_count = len(set(core_high_signals(features, data)))
    has_strong = bool(strong_final_signals(data, features))
    step_count = features.get("step_count")
    if step_count in ["1-2步", "3-5步"]:
        return signal_count < 3 and not has_strong
    if step_count == "6-8步":
        return signal_count < 2 and not has_strong
    # 9 步以上原则上尊重模型原判，除非上游特征明显自相矛盾；规范化后不猜测矛盾。
    return False


def sync_coarse_difficulty(result: Dict[str, Any]) -> None:
    level = result.get("difficulty_level")
    result["coarse_difficulty"] = {
        "送分题": "送分/基础区间（1-2档）",
        "基础题": "送分/基础区间（1-2档）",
        "中等题": "基础/中等区间（2-3档）",
        "拔高题": "中等/拔高区间（3-4档）",
        "压轴题": "拔高/压轴区间（4-5档）",
    }.get(level, "基础/中等区间（2-3档）")


def set_level_with_audit(result: Dict[str, Any], level: str, rule: str, evidence: List[str]) -> None:
    old = result.get("difficulty_level")
    if old == level:
        return
    actions = result.setdefault("postprocess_actions", [])
    actions.append({"rule": rule, "from": old, "to": level, "evidence": evidence[:8]})
    result["difficulty_level"] = level
    reasoning = result.setdefault("reasoning", {})
    prefix = f"自动调整：{rule}；证据：{'；'.join(evidence[:5])}。"
    reasoning["core_basis"] = prefix + str(reasoning.get("core_basis", ""))


def is_fused_standard_diagram_task(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """规范作图/接线只算一次显性规则应用，不受步骤数措辞漂移影响。"""
    text = visible_text_of(data)
    action = contains_any(text, ["画出", "作出", "作图", "标出", "连接完整", "补充完整", "补全"])
    analysis_request = contains_any(text, ["求", "计算", "说明理由", "分析变化", "设计方案", "评价"])
    return bool(
        action
        and not analysis_request
        and count_subquestions(data) == 0
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") in ["无约束", "单一约束"]
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )


def is_fused_textbook_easy(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """只允许结构稳定的教材原型从基础降到送分。"""
    text = visible_text_of(data)
    if is_fused_standard_diagram_task(data, features):
        # 仅保留两个完全显性的教材原型：单条特殊光线、水平面单物体二力图。
        return contains_any(text, ["经透镜射出的光线", "凸透镜特殊光线"]) or (
            contains_any(text, ["静止在水平地面", "静止在水平面"])
            and contains_any(text, ["受力示意图", "受力图"])
        )
    if features.get("cross_module") == "跨模块综合" or features.get("problem_structure") not in ["概念判断", "光学声学综合"]:
        return False
    direct_estimate = bool(
        contains_any(text, ["与实际相符", "估测", "估计值"])
        and not contains_any(text, ["求", "计算", "推导"])
        and features.get("information_carrier") == "纯文字"
    )
    if (has_formula_intent(text) and not direct_estimate) or count_subquestions(data) > 0:
        return False
    application_mapping = contains_any(text, ["制作", "制成", "改变", "变化", "每秒", "简谱", "音符", "主要利用"])
    multiple_processes = text.count("这是") >= 2 or text.count("因为") >= 2
    if application_mapping or multiple_processes:
        return False
    direct_semantic = bool(
        direct_estimate
        or contains_any(text, ["粒子与宇宙", "扩散"])
        or (contains_any(text, ["树荫", "水中的鱼", "水中游鱼"]) and contains_any(text, ["光", "折射", "直线传播"]))
        or (features.get("knowledge_count") == "1个" and contains_any(text, ["单位", "可再生", "不可再生", "熔化", "凝固", "汽化", "液化", "响度", "无线电波"]))
    )
    return bool(
        features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") in ["直接套用", "简单因果推理"]
        and features.get("knowledge_diff") == "低"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
        and direct_semantic
    )


def is_fused_cross_module_material_medium(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """三问以上材料题若包含模型转换和跨模块解释，不能按独立小问压成基础。"""
    count = count_subquestions(data)
    supports = 0
    supports += features.get("step_count") == "3-5步"
    supports += features.get("formula_count") in ["2-3个", "4-6个"]
    supports += features.get("information_carrier") in ["图像或表格", "多图表综合"]
    supports += features.get("variable_relation") in ["简单正反比", "图像函数关系"]
    text = full_text_of(data)
    has_continuous_model = bool(
        ("往返" in text and contains_any(text, ["液面", "传播", "反射", "路程"]))
        or (contains_any(text, ["卫星", "航天"]) and contains_any(text, ["周期", "推导", "轨道"]))
    )
    return bool(
        count >= 3
        and features.get("subquestion_dependency") == "多问但相互独立"
        and features.get("cross_module") == "跨模块综合"
        and has_continuous_model
        and supports >= 2
    )


def is_fused_multiple_experiment_medium(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """多个标准实验的读数、误差方向和现象解释形成整题中等负担。"""
    text = full_text_of(data)
    standard_reflection_chain = bool(
        "探究光的反射定律" in text
        and contains_any(text, ["多次改变", "表格数据", "纸板", "光路可逆"])
    )
    if standard_reflection_chain:
        return True
    return bool(
        count_subquestions(data) >= 3
        and features.get("subquestion_dependency") == "多问但相互独立"
        and features.get("problem_structure") == "实验探究"
        and features.get("information_carrier") in ["实验装置图", "多图表综合"]
        and contains_any(text, ["误差", "为什么", "原因", "调节", "不能", "不可以", "实验"])
    )


def is_fused_routine_heater_or_pressure_scale(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """模型明确的电热/压力秤常规综合保护在中等。"""
    text = full_text_of(data)
    heater = contains_any(text, ["加热挡", "保温挡", "高温挡", "低温挡", "双挡", "两个挡位"]) and contains_any(text, ["电热", "电热水壶", "电热器"])
    pressure_scale = contains_any(text, ["压力秤", "压敏电阻"]) and contains_any(text, ["欧姆", "量程", "R-F", "阻值"])
    strong_block = contains_any(text, [
        "黑箱", "非线性I-U", "分类讨论", "多解", "不等式", "边界覆盖", "可行性验证", "故障并存",
        "项目", "设计方案", "方案设计", "判断是否可行", "能否覆盖", "安全电流", "温度范围",
    ])
    common = not strong_block and features.get("reasoning_chain") != "逆向推理或临界分析" and features.get("variable_relation") != "多变量耦合关系"
    if pressure_scale:
        # 电表量程可能被抽成“多约束”，但直接 R-F 读图仍是常规中等结构。
        return common
    return bool(heater and common and features.get("constraint_count") in ["无约束", "单一约束"])


def is_fused_standard_experiment_medium(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    text = full_text_of(data)
    return bool(
        features.get("problem_structure") == "实验探究"
        and features.get("step_count") in ["1-2步", "3-5步"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("experiment_requirement") in ["基础操作或读数", "控制变量或故障分析"]
        and features.get("graph_table_requirement") in ["无", "直接读数", "多组比较归纳"]
        and not contains_any(text, ["缺表法", "等效替代", "自主设计", "新方案", "可行性", "边界验证", "异常点反推"])
    )


def is_fused_routine_control_medium(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """显性控制链即使被抽成多状态/多约束，也不因单个 feature 升拔高。"""
    text = full_text_of(data)
    explicit_control = contains_any(text, ["自动接通", "自动断开", "停止工作", "开始工作", "实现保温", "报警"])
    no_calculation = features.get("formula_count") == "0-1个" and features.get("calculation_complexity") == "口算或直接判断"
    return bool(
        explicit_control
        and no_calculation
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
        and not contains_any(text, ["参数范围", "方案设计", "可行性", "边界覆盖", "改装", "标尺"])
    )


def is_fused_hidden_model_hard(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """低计算但存在隐含过程、几何/力臂、路径或误差建模的拔高通道。"""
    text = full_text_of(data)
    stable_boundary_task = bool(
        ("力臂" in text and "支点" in text and contains_any(text, ["保持水平", "水平静止"]))
        or ("管口" in text and contains_any(text, ["操作顺序", "正确顺序", "快速上提", "快速下移"]))
        or (
            "空调" in text
            and contains_any(text, ["载流量", "导线承载"])
            and contains_any(text, ["线路设计", "设计线路", "线路的设计", "设计电路"])
        )
        or (
            contains_any(text, ["【项目要求】", "项目式"])
            and contains_any(text, ["自动控温", "控温电路"])
            and contains_any(text, ["安全工作电流", "温度范围"])
            and contains_any(text, ["是否可行", "可行性", "是否满足", "能否满足"])
        )
    )
    if stable_boundary_task:
        return True
    semantic_groups = [
        ["力臂", "支点"], ["路径", "反射"], ["误差", "方向"],
        ["顺序", "操作"], ["等效", "替代"], ["载流量", "选择"],
    ]
    semantic_count = sum(all(word in text for word in group) for group in semantic_groups)
    feature_support = sum([
        features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"],
        features.get("experiment_requirement") == "方案设计或误差评价",
        features.get("subquestion_dependency") == "多问且层层递进",
        features.get("state_count") in ["多状态", "连续变化或临界状态"],
    ])
    return semantic_count >= 1 and feature_support >= 2


def is_fused_validated_design_final(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """压轴项目必须同时有设计、范围/边界验证和模型支撑。"""
    text = full_text_of(data)
    design = contains_any(text, ["改装", "设计标尺", "方案设计", "设计方案", "自主设计", "设计控制", "设计电路", "项目式"])
    validation_groups = [
        contains_any(text, ["覆盖", "边界", "量程", "所有待测", "安全电流", "温度范围"]),
        contains_any(text, ["可行性", "是否可行", "能否实现", "能否满足"]),
        contains_any(text, ["筛选有效解", "方案比较", "筛选方案", "选择方案"]),
    ]
    validation_count = sum(validation_groups)
    model_support = sum([
        features.get("step_count") in ["6-8步", "9-12步", "12步以上"],
        features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"],
        features.get("knowledge_count") == "4个及以上",
        features.get("cross_module") == "跨模块综合",
        features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"],
        features.get("experiment_requirement") == "方案设计或误差评价",
    ])
    expression_design_chain = bool(
        count_subquestions(data) >= 3
        and contains_any(text, ["用物理量符号写出表达式", "写出表达式"])
        and contains_any(text, ["怎样调整", "做出怎样的调整", "如何调整"])
        and features.get("formula_count") in ["4-6个", "7个以上"]
        and features.get("cross_module") == "跨模块综合"
    )
    return expression_design_chain or (design and validation_count >= 2 and model_support >= 3)


def fused_basic_to_medium_evidence(features: Dict[str, str], data: Dict[str, Any]) -> List[str]:
    if is_fused_standard_diagram_task(data, features):
        return []
    if is_fused_multiple_experiment_medium(data, features):
        return ["多个标准实验", "包含读数/误差/现象解释", "整题综合负担达到中等"]
    if is_fused_cross_module_material_medium(data, features):
        return ["三问以上材料题", "跨模块模型转换", "包含计算与解释"]
    evidence: List[str] = []
    if features.get("experiment_requirement") == "控制变量或故障分析" and features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]:
        evidence.extend(["控制变量或故障分析", "多组数据归纳/图像反推"])
    if features.get("subquestion_dependency") == "多问且层层递进" and features.get("reasoning_chain") == "多层因果推理":
        evidence.extend(["多问层层递进", "连续因果链"])
    if features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"] and features.get("state_count") in ["双状态", "多状态", "连续变化或临界状态"]:
        evidence.extend(["多公式联动", "存在状态转换"])
    return evidence


def fused_medium_to_hard_evidence(features: Dict[str, str], data: Dict[str, Any]) -> List[str]:
    if is_fused_routine_control_medium(data, features) or is_fused_routine_heater_or_pressure_scale(data, features) or is_fused_standard_experiment_medium(data, features):
        return []
    if is_fused_hidden_model_hard(data, features):
        return ["隐含过程/几何/误差模型", "至少两个结构证据相互印证"]
    evidence = core_high_signals(features, data)
    stable_semantic = bool(
        features.get("experiment_requirement") == "方案设计或误差评价"
        or features.get("graph_table_requirement") == "图像反推或外推"
        or features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"]
        or features.get("reasoning_chain") == "逆向推理或临界分析"
    )
    required = 2 if features.get("step_count") == "6-8步" else 3
    return evidence if stable_semantic and len(set(evidence)) >= required else []


def fused_hard_to_final_evidence(features: Dict[str, str], data: Dict[str, Any]) -> List[str]:
    """非项目压轴必须由稳定的深耦合特征共同支持，不能复用“未降档”条件。"""
    evidence: List[str] = []
    if features.get("step_count") in ["9-12步", "12步以上"]: evidence.append("9步以上真实推理")
    if features.get("formula_count") in ["4-6个", "7个以上"]: evidence.append("长公式链")
    if features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"]: evidence.append("联立/范围计算")
    if features.get("reasoning_chain") == "逆向推理或临界分析": evidence.append("逆向或临界分析")
    if features.get("constraint_count") == "多约束": evidence.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系": evidence.append("多变量耦合")
    if features.get("graph_table_requirement") == "图像反推或外推": evidence.append("图像反推")
    if features.get("cross_module") == "跨模块综合": evidence.append("跨模块融合")
    if features.get("experiment_requirement") == "方案设计或误差评价": evidence.append("方案设计/评价")

    if features.get("step_count") in ["9-12步", "12步以上"]:
        return evidence if len(set(evidence)) >= 5 else []
    # 6-8步必须至少六项深耦合证据，防止普通实验评价或过程顺序题升压轴。
    return evidence if features.get("step_count") == "6-8步" and len(set(evidence)) >= 6 else []


def postprocess_fused(rating_result: Dict[str, Any], data: Dict[str, Any], raw_level: str) -> Dict[str, Any]:
    """融合最终版：单实现、结构语义优先、每次最多调整一档。"""
    features = rating_result["features"]

    if raw_level == "送分题":
        if not is_fused_textbook_easy(data, features):
            reasons = should_upgrade_easy_to_basic(features, data)
            if reasons:
                set_level_with_audit(rating_result, "基础题", "fused_easy_to_basic", reasons)
    elif raw_level == "基础题":
        if is_fused_textbook_easy(data, features):
            set_level_with_audit(rating_result, "送分题", "fused_basic_to_easy", ["教材原型直接识别", "无计算/实验/状态分析"])
        else:
            evidence = fused_basic_to_medium_evidence(features, data)
            if evidence:
                set_level_with_audit(rating_result, "中等题", "fused_basic_to_medium", evidence)
    elif raw_level == "中等题":
        if is_fused_standard_diagram_task(data, features):
            set_level_with_audit(rating_result, "基础题", "fused_medium_to_basic", ["单一规范作图/接线", "无计算、实验数据或状态分析"])
        else:
            evidence = fused_medium_to_hard_evidence(features, data)
            if evidence:
                set_level_with_audit(rating_result, "拔高题", "fused_medium_to_hard", evidence)
    elif raw_level == "拔高题":
        if is_fused_validated_design_final(data, features):
            set_level_with_audit(rating_result, "压轴题", "fused_hard_to_final", ["方案设计", "范围/边界覆盖", "可行性验证", "多模型支撑"])
        else:
            evidence = fused_hard_to_final_evidence(features, data)
            if evidence:
                set_level_with_audit(rating_result, "压轴题", "fused_hard_to_final", evidence)
    elif raw_level == "压轴题" and should_downgrade_final_to_hard(features, data):
        set_level_with_audit(rating_result, "拔高题", "fused_final_to_hard", core_high_signals(features, data))

    sync_coarse_difficulty(rating_result)
    return rating_result


def postprocess_v7_compat(
    rating_result: Dict[str, Any],
    data: Dict[str, Any],
    raw_level: str,
) -> Dict[str, Any]:
    """调用历史 120/133 版本的最终 V7 语义层，并补齐当前审计字段。

    旧模块仅作为冻结参考实现使用；API、缓存、并发和输出仍由当前脚本负责。
    """
    from legacy import physics_difficulty_rating_v7_reference as legacy_v7

    compat_result = legacy_v7.postprocess_physics_difficulty(copy.deepcopy(rating_result), data)
    if not isinstance(compat_result, dict):
        compat_result = rating_result
    final_level = compat_result.get("difficulty_level")
    if final_level not in LEVEL_MAP:
        final_level = raw_level
        compat_result["difficulty_level"] = final_level

    compat_result["difficulty_level_raw"] = raw_level
    compat_result["postprocess_actions"] = []
    if final_level != raw_level:
        core_basis = str(compat_result.get("reasoning", {}).get("core_basis", "")).strip()
        evidence = [core_basis.split("。", 1)[0]] if core_basis else ["历史 V7 边界语义规则命中"]
        compat_result["postprocess_actions"].append(
            {
                "rule": "v7_compat_semantic_layer",
                "from": raw_level,
                "to": final_level,
                "evidence": evidence,
            }
        )
    sync_coarse_difficulty(compat_result)
    return compat_result


def has_open_high_order_experiment(text: str) -> bool:
    """识别不能被“常规实验保护”覆盖的真正开放或反推结构。"""
    return contains_any(
        text,
        [
            "异常点", "偏离直线", "偏离图线", "评价猜想", "猜想是否合理", "提出新猜想", "新猜想",
            "缺表法", "等效替代", "自主设计", "设计实验", "设计方案", "改进方案", "推导表达式",
            "可行性验证", "边界覆盖", "量程设计", "标尺设计", "分类讨论", "多解筛选", "筛选有效解",
        ],
    )


def is_routine_measurement_experiment(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """常规测量、读表、计算及有限误差比较；不依赖某一道题的专属名词。"""
    text = full_text_of(data)
    visible = visible_text_of(data)
    if features.get("problem_structure") != "实验探究" or has_open_high_order_experiment(text):
        return False
    # 附着/带出液体后的误差可能发生抵消，需要等效关系推理，不属于普通误差来源判断。
    if contains_any(visible, ["带出", "附着", "沾水"]) and contains_any(visible, ["偏大", "偏小", "仍然准确", "是否准确"]):
        return False
    if features.get("step_count") not in ["3-5步", "6-8步"]:
        return False
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        return False
    electrical_measurement = (
        contains_any(text, ["测量", "伏安法", "探究电流与电压"])
        and contains_any(text, ["电压", "电压表"])
        and contains_any(text, ["电流", "电流表"])
        and contains_any(text, ["滑动变阻器", "变阻器"])
    )
    density_measurement = (
        "密度" in text
        and contains_any(text, ["测量", "天平", "质量"])
        and contains_any(text, ["体积", "排水", "浸没"])
    )
    return electrical_measurement or density_measurement


def is_routine_single_boundary_circuit(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """单一工作状态下按量程取一个端点的常规电路计算。"""
    text = full_text_of(data)
    if count_subquestions(data) != 0 or features.get("problem_structure") != "电路综合":
        return False
    if features.get("step_count") != "3-5步" or features.get("calculation_complexity") != "简单笔算":
        return False
    if features.get("formula_count") not in ["0-1个", "2-3个"]:
        return False
    if features.get("reasoning_chain") == "逆向推理或临界分析" or features.get("variable_relation") == "多变量耦合关系":
        return False
    if contains_any(text, ["多开关", "非线性", "I-U", "分类讨论", "多解", "黑箱", "可行性验证"]):
        return False
    return (
        contains_any(text, ["量程", "额定电流", "器材的安全", "元件安全"])
        and contains_any(text, ["滑动变阻器", "变阻器"])
        and contains_any(text, ["至少", "最小阻值", "最大阻值", "最大允许电流"])
    )


def is_closed_measurement_material(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """给定完整操作链、只要求读数/计算/表达式的封闭式材料实验。"""
    text = full_text_of(data)
    if count_subquestions(data) < 2 or features.get("step_count") != "3-5步":
        return False
    experiment_context = (
        features.get("problem_structure") == "实验探究"
        or features.get("additional_structure") == "实验探究"
        or contains_any(text, ["实验操作", "实验步骤", "测量过程", "开展了以下研究"])
    )
    if not experiment_context:
        return False
    if features.get("calculation_complexity") not in ["口算或直接判断", "简单笔算"]:
        return False
    if features.get("constraint_count") not in ["无约束", "单一约束"]:
        return False
    if features.get("variable_relation") not in ["无变量关系", "简单正反比"]:
        return False
    if features.get("graph_table_requirement") not in ["无", "直接读数"] or has_open_high_order_experiment(text):
        return False
    closed_actions = contains_any(text, ["实验操作", "实验步骤", "记录", "读数", "计算", "表达式", "测量"])
    measurement_relation = contains_any(text, ["质量", "体积", "密度", "电流", "电压", "压强"])
    return closed_actions and measurement_relation


def is_hidden_dynamic_lever(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """用题目结构识别低计算、高建模的动态杠杆，不依赖 state_count。"""
    text = full_text_of(data)
    return bool(
        features.get("problem_structure") == "力学综合"
        and features.get("step_count") in ["3-5步", "6-8步"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"]
        and contains_any(text, ["杠杆", "支点"])
        and "力臂" in text
        and contains_any(text, ["角度", "转动", "变化", "始终与", "垂直"])
    )


def is_simple_source_polarity_task(data: Dict[str, Any], features: Dict[str, str]) -> bool:
    """电源极性顺向判断及简单原因列举，避免被实验 feature 漂移抬档。"""
    text = full_text_of(data)
    return bool(
        features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("knowledge_diff") == "低"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("graph_table_requirement") == "无"
        and contains_any(text, ["发光二极管", "LED"])
        and contains_any(text, ["正极", "负极"])
        and not contains_any(text, ["测量数据", "数据表", "伏安特性", "设计实验", "控制变量法"])
    )


def v7_stable_override(
    raw_level: str,
    features: Dict[str, str],
    data: Dict[str, Any],
) -> Optional[Tuple[str, str, List[str]]]:
    """在冻结 V7 之后修正少量已复现边界；不替换 V7 主体规则。"""
    visible = visible_text_of(data)
    text = full_text_of(data)
    subquestion_count = count_subquestions(data)
    no_real_calculation = (
        features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
    )

    # 稳定规则只看可复核的题目结构，不让单个易漂移 feature 决定升降档。
    if raw_level == "中等题" and is_hidden_dynamic_lever(data, features):
        return "拔高题", "stable_hidden_dynamic_lever", ["动态力臂与角度关系", "低计算但需隐含几何建模"]

    if raw_level == "基础题" and is_simple_source_polarity_task(data, features):
        return "基础题", "stable_simple_source_polarity", ["顺向极性判断", "无数据归纳、计算或连续实验推理"]

    if raw_level == "拔高题" and is_routine_measurement_experiment(data, features):
        return "中等题", "stable_routine_measurement_experiment", ["标准测量流程", "无异常点反推、开放设计或边界验证"]

    if raw_level == "拔高题" and is_routine_single_boundary_circuit(data, features):
        return "中等题", "stable_routine_single_boundary_circuit", ["单状态量程端点计算", "无多状态、分类讨论或多解筛选"]

    if raw_level == "中等题" and is_closed_measurement_material(data, features):
        return "中等题", "stable_closed_measurement_material", ["操作链和所求关系均显性", "无开放设计、异常点反推或复杂边界"]

    standard_connection = bool(
        raw_level == "基础题"
        and subquestion_count == 0
        and no_real_calculation
        and contains_any(visible, ["实物连接完整", "实物图连接完整", "将图乙的实物连接完整", "补全实物图"])
        and contains_any(visible, ["电路图", "图甲", "图乙", "电压表", "电流表"])
    )
    standard_magnetic_diagram = bool(
        raw_level == "基础题"
        and subquestion_count == 0
        and no_real_calculation
        and contains_any(visible, ["小磁针", "磁感线"])
        and contains_any(visible, ["电源的正极", "电源正极", "正极"])
        and contains_any(visible, ["画出", "标出"])
    )
    if standard_connection or standard_magnetic_diagram:
        return "基础题", "v7_stable_standard_diagram_guard", ["单一规范作图/接线", "无计算和状态分析"]

    simple_buoyancy_application = bool(
        raw_level == "基础题"
        and subquestion_count == 0
        and no_real_calculation
        and features.get("knowledge_count") == "1个"
        and contains_any(visible, ["悬浮"])
        and contains_any(visible, ["吸进气室", "气室吸水", "排水"])
        and not contains_any(visible, ["求", "计算", "体积分数", "密度"])
    )
    if simple_buoyancy_application:
        return "基础题", "v7_stable_simple_state_guard", ["单知识点浮沉条件应用", "双状态表述不等于综合建模"]

    explicit_control_choice = bool(
        raw_level == "中等题"
        and subquestion_count == 0
        and no_real_calculation
        and features.get("problem_structure") == "电路综合"
        and contains_any(text, ["主加热盘", "副加热盘"])
        and contains_any(text, ["干烧", "温度过高"])
        and contains_any(visible, ["符合要求", "下列电路"])
    )
    if explicit_control_choice:
        return "中等题", "v7_stable_explicit_control_guard", ["控制逻辑完全显性", "无参数计算、范围筛选或开放设计"]

    parallel_cross_module_choice = bool(
        raw_level == "基础题"
        and subquestion_count == 0
        and no_real_calculation
        and features.get("cross_module") == "跨模块综合"
        and features.get("knowledge_count") in ["2-3个", "4个及以上"]
        and (bool(str(data.get("options", "") or "").strip()) or contains_any(visible, ["四个选项", "说法正确", "说法错误", "下列说法"]))
    )
    if parallel_cross_module_choice:
        return "基础题", "v7_stable_parallel_choice_guard", ["多个独立基础选项", "跨模块但没有连续推理"]

    two_mode_changeover_circuit = bool(
        raw_level == "基础题"
        and subquestion_count == 0
        and no_real_calculation
        and contains_any(text, ["单刀双掷开关"])
        and contains_any(text, ["太阳能板", "可充电电池"])
        and contains_any(text, ["电动机工作", "电动机不工作"])
        and "充电" in text
    )
    if two_mode_changeover_circuit:
        return "中等题", "v7_stable_two_mode_circuit", ["供电/充电双状态", "需逐项验证转换开关逻辑"]

    round_trip_material = bool(
        raw_level == "基础题"
        and subquestion_count >= 3
        and features.get("step_count") == "3-5步"
        and features.get("formula_count") == "2-3个"
        and features.get("cross_module") == "跨模块综合"
        and contains_any(text, ["往返时间", "往返总路程"])
        and contains_any(text, ["液面", "罐底"])
    )
    if round_trip_material:
        return "中等题", "v7_stable_round_trip_material", ["三问材料题", "往返模型、计算和压强解释形成完整负担"]

    multiple_standard_experiments = bool(
        raw_level == "基础题"
        and subquestion_count >= 4
        and features.get("problem_structure") == "实验探究"
        and features.get("information_carrier") == "多图表综合"
        and sum(term in text for term in ["杠杆", "托里拆利", "大气压", "扩散", "分子间", "误差"]) >= 3
    )
    if multiple_standard_experiments:
        return "中等题", "v7_stable_multi_experiment", ["四个标准实验", "包含读数、误差方向和现象解释"]

    independent_direct_calculations = bool(
        raw_level == "中等题"
        and 1 <= subquestion_count <= 2
        and features.get("subquestion_dependency") == "多问但相互独立"
        and features.get("step_count") in ["1-2步", "3-5步"]
        and features.get("formula_count") in ["0-1个", "2-3个"]
        and features.get("calculation_complexity") == "简单笔算"
        and features.get("reasoning_chain") in ["直接套用", "简单因果推理", "多层因果推理"]
        and features.get("state_count") == "单状态"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") in ["无", "直接读数"]
        and not contains_any(text, ["设计方案", "方案设计", "推导", "证明", "误差", "评价", "临界", "极值", "分类讨论", "多解"])
    )
    if independent_direct_calculations:
        return "基础题", "v7_stable_independent_calculation_guard", ["两个以内独立小问", "均为显性单公式直接应用"]

    routine_two_level_heater = bool(
        raw_level == "拔高题"
        and contains_any(text, ["电热水壶", "电热器"])
        and contains_any(text, ["加热挡", "加热和保温两个挡位"])
        and "保温挡" in text
        and features.get("state_count") == "双状态"
        and features.get("constraint_count") in ["无约束", "单一约束"]
        and features.get("variable_relation") in ["无变量关系", "简单正反比"]
        and features.get("reasoning_chain") != "逆向推理或临界分析"
        and not contains_any(text, ["安全范围", "安全电流", "故障", "临界", "极值", "不等式", "分类讨论", "多解筛选", "可行性验证"])
    )
    if routine_two_level_heater:
        return "中等题", "v7_stable_routine_heater_guard", ["常规双挡电热模型", "公式链显性且无临界、多解或安全范围"]

    return None


def postprocess_v7_stable(
    rating_result: Dict[str, Any],
    data: Dict[str, Any],
    raw_level: str,
) -> Dict[str, Any]:
    """正式教师标准版：只归一化和审计，不改写模型原始等级。

    1000 题教师标签回归表明，历史语义层和窄规则在中间档上的误改多于改对。
    正式 Prompt 已直接对齐教师五维标准，因此该 profile 保留归一化、
    difficulty_level_raw 和 postprocess_actions 审计字段，不再依赖易漂移的
    features 或题型关键词执行升降档。历史复现仍可显式使用 v7_compat。
    """
    stable_result = copy.deepcopy(rating_result)
    stable_result["difficulty_level"] = raw_level
    stable_result["difficulty_level_raw"] = raw_level
    stable_result["postprocess_actions"] = []
    sync_coarse_difficulty(stable_result)
    return stable_result


def postprocess_physics_difficulty(rating_result: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """统一后处理：最多调整一档，并记录每次调整的证据。"""
    if not isinstance(rating_result, dict) or not rating_result:
        return {}
    raw_level = extract_raw_difficulty_level(rating_result)
    rating_result["features"] = normalize_features(rating_result.get("features"))
    normalize_reasoning_schema(rating_result)
    if raw_level not in LEVEL_MAP:
        raw_level = "中等题"
    rating_result["difficulty_level"] = raw_level
    rating_result["difficulty_level_raw"] = raw_level
    rating_result["postprocess_actions"] = []

    if RATING_PROFILE == "fused":
        return postprocess_fused(rating_result, data, raw_level)
    if RATING_PROFILE == "v7_stable":
        return postprocess_v7_stable(rating_result, data, raw_level)
    if RATING_PROFILE == "v7_compat":
        return postprocess_v7_compat(rating_result, data, raw_level)

    f = rating_result["features"]
    current = raw_level

    if current == "基础题" and should_downgrade_basic_to_easy(f, data):
        set_level_with_audit(rating_result, "送分题", "basic_to_easy_strict", ["单一知识点", "直接识别", "无真实计算或模型分析"])
    elif current == "送分题":
        reasons = should_upgrade_easy_to_basic(f, data)
        if reasons:
            set_level_with_audit(rating_result, "基础题", "easy_to_basic_guard", reasons)
    elif current == "基础题":
        reasons = should_upgrade_basic_to_medium(f, data)
        if reasons:
            set_level_with_audit(rating_result, "中等题", "basic_to_medium", reasons)
    elif current == "中等题":
        if should_downgrade_medium_to_basic(f, data):
            set_level_with_audit(rating_result, "基础题", "medium_to_basic_guard", ["1-2步", "单一知识点", "无高阶结构"])
        else:
            ok, evidence = should_upgrade_medium_to_hard(f, data)
            if ok:
                set_level_with_audit(rating_result, "拔高题", "medium_to_hard", evidence)
    elif current == "拔高题":
        if should_downgrade_hard_to_medium(f, data):
            set_level_with_audit(rating_result, "中等题", "hard_to_medium_guard", core_high_signals(f, data))
        else:
            ok, evidence = should_upgrade_hard_to_final(f, data)
            if ok:
                set_level_with_audit(rating_result, "压轴题", "hard_to_final", evidence)
    elif current == "压轴题" and should_downgrade_final_to_hard(f, data):
        set_level_with_audit(rating_result, "拔高题", "final_to_hard_guard", core_high_signals(f, data))

    sync_coarse_difficulty(rating_result)
    return rating_result


# -------------------------- Input / API / output --------------------------
def sanitize_question_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """隔离历史 JSONL 的 difficulty，确保它不进入模型、规则或评估流程。"""
    return {key: value for key, value in data.items() if key != "difficulty"}


def make_output_base(data: Dict[str, Any]) -> Dict[str, Any]:
    """保留来源字段但显式标记为不可信，避免下游误作教师真值。"""
    output = dict(data)
    source_difficulty = output.pop("difficulty", None)
    if source_difficulty is not None:
        output["source_difficulty_untrusted"] = source_difficulty
    return output


def construct_question_content(data: Dict[str, Any]) -> str:
    parts: List[str] = []
    for label, key in [("题干", "stem"), ("选项", "options"), ("解析", "analysis")]:
        value = str(data.get(key, "") or "").strip()
        if value:
            parts.append(f"【{label}】\n{value}")
    sub_questions = data.get("sub_questions") or []
    if sub_questions:
        parts.append("【小题】")
        for i, item in enumerate(sub_questions, 1):
            if isinstance(item, dict):
                parts.append("\n".join([f"小题{i}", f"题干：{item.get('stem', '')}", f"选项：{item.get('options', '')}", f"解析：{item.get('analysis', '')}"]))
            else:
                parts.append(f"小题{i}：{item}")
    return "\n\n".join(parts)


def parse_model_response(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    candidates = [text]
    if "```" in text:
        candidates.append(text.split("```", 1)[1].replace("json", "", 1).split("```", 1)[0].strip())
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json_repair.loads(candidate)
            if isinstance(value, dict):
                return value
        except Exception:
            pass
    return {}


async def call_model_with_cache(question_content: str, session: aiohttp.ClientSession, retries: int, timeout_sec: int) -> Tuple[Dict[str, Any], float, int, int, int]:
    response_id = await get_or_create_cache(session, retries, timeout_sec) if USE_CACHE else None
    if USE_CACHE and not response_id:
        return {}, 0.0, 0, 0, 0
    dynamic_content = question_content + DIFFICULTY_RATING_PROMPT_SUFFIX
    for retry in range(retries):
        payload: Dict[str, Any] = {
            "model": MODEL_NAME,
            "input": [{"role": "user", "content": dynamic_content if response_id else DIFFICULTY_RATING_PROMPT_PREFIX + "\n\n" + dynamic_content}],
            "thinking": {"type": "disabled"},
        }
        if response_id:
            payload["previous_response_id"] = response_id
        if TEMPERATURE is not None:
            payload["temperature"] = TEMPERATURE
        started = time.time()
        try:
            async with session.post(f"{BASE_URL}responses", json=payload, headers={"Authorization": f"Bearer {API_KEY}"}, timeout=aiohttp.ClientTimeout(total=timeout_sec)) as response:
                if response.status == 200:
                    result = await response.json()
                    output_text = ""
                    for item in result.get("output", []):
                        for content in item.get("content", []) if item.get("type") == "message" else []:
                            if content.get("type") == "output_text":
                                output_text = content.get("text", "")
                    usage = result.get("usage", {})
                    return parse_model_response(output_text), time.time() - started, usage.get("input_tokens", 0), usage.get("output_tokens", 0), usage.get("total_tokens", 0)
                error_text = await response.text()
                print(f"API请求失败 (状态码: {response.status}): {error_text[:200]}")
                if USE_CACHE and "PreviousResponseNotFound" in error_text:
                    response_id = await create_prefix_cache(session, retries, timeout_sec)
                    continue
                if response.status == 429 or response.status >= 500:
                    await asyncio.sleep(2 ** retry + random.random())
                    continue
                return {}, 0.0, 0, 0, 0
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if retry == retries - 1:
                print(f"网络异常最终失败: {exc}")
                return {}, 0.0, 0, 0, 0
            await asyncio.sleep(2 ** retry + random.random())
        except Exception as exc:
            print(f"运行过程中请求异常: {exc}")
            if retry == retries - 1:
                return {}, 0.0, 0, 0, 0
            await asyncio.sleep(1)
    return {}, 0.0, 0, 0, 0


async def append_jsonl(path: str, data: Dict[str, Any]) -> None:
    async with FILE_LOCK:
        async with aiofiles.open(path, "a", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def process_single_question(data: Dict[str, Any], session: aiohttp.ClientSession, semaphore: Semaphore, output_path: str, error_path: str, retries: int, timeout_sec: int) -> None:
    async with semaphore:
        safe_data = sanitize_question_data(data)
        try:
            result, elapsed, prompt_tokens, completion_tokens, total_tokens = await call_model_with_cache(construct_question_content(safe_data), session, retries, timeout_sec)
            raw_result = copy.deepcopy(result)
            raw_level = extract_raw_difficulty_level(raw_result)
            postprocess_input = copy.deepcopy(result)
            if isinstance(postprocess_input, dict) and raw_level and postprocess_input.get("difficulty_level") not in LEVEL_MAP:
                postprocess_input["difficulty_level"] = raw_level
            result = postprocess_physics_difficulty(postprocess_input, safe_data)
            output = make_output_base(data)
            output.update({
                "rating_profile": RATING_PROFILE,
                "difficulty_rating_raw": raw_result,
                "difficulty_level_raw": raw_level,
                "postprocess_actions": result.get("postprocess_actions", []) if isinstance(result, dict) else [],
                "difficulty_rating": result,
                "api_time_use": round(elapsed, 2),
                "api_prompt_tokens": prompt_tokens,
                "api_completion_tokens": completion_tokens,
                "api_total_tokens": total_tokens,
            })
            if result.get("difficulty_level"):
                await append_jsonl(output_path, output)
            else:
                output["rating_error"] = "模型返回数据为空或格式错误"
                await append_jsonl(error_path, output)
        except Exception as exc:
            output = make_output_base(data)
            output["rating_error"] = str(exc)
            await append_jsonl(error_path, output)


def get_processed_question_ids(output_path: str) -> set:
    processed = set()
    if not os.path.exists(output_path):
        return processed
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                qid = json.loads(line).get("question_id")
                if qid is not None:
                    processed.add(str(qid))
            except Exception:
                continue
    return processed


async def main_batch_run() -> None:
    global USE_CACHE
    parser = argparse.ArgumentParser(description="初中物理难度评级批量打标")
    parser.add_argument("-p", "--prompt", default=DEFAULT_PROMPT_PATH)
    parser.add_argument("-i", "--input", default="../data/physics_sampled_5000_per_difficulty.jsonl")
    parser.add_argument("-o", "--output", default="physics_difficulty_rated_results.jsonl")
    parser.add_argument("-e", "--error", default="physics_difficulty_errors.jsonl")
    parser.add_argument("-c", "--concurrency", type=int, default=15)
    parser.add_argument("-t", "--timeout", type=int, default=180)
    parser.add_argument("-r", "--retries", type=int, default=3)
    parser.add_argument("-n", "--num", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    USE_CACHE = not args.no_cache
    print(f"评级配置: {RATING_PROFILE}")
    if args.seed is not None:
        random.seed(args.seed)
        print(f"固定随机种子: {args.seed}")
    if not USE_CACHE:
        print("已禁用前缀缓存：每次请求发送完整提示词")
    load_prompt_config(args.prompt)
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")
    questions = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    questions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    print(f"成功加载题目数据，共计 {len(questions)} 道题目。")
    if args.num is not None:
        questions = random.sample(questions, min(args.num, len(questions)))
        print(f"参数 -n 生效，随机抽样其中 {len(questions)} 道题进行测试。")
    elif args.seed is None:
        random.shuffle(questions)
    processed = get_processed_question_ids(args.output)
    pending = [q for q in questions if str(q.get("question_id")) not in processed]
    print(f"数据比对完成: 已完成数 {len(processed)}，待处理数 {len(pending)}")
    if not pending:
        print("所有题目都已完成打标！")
        return
    semaphore = Semaphore(args.concurrency)
    pbar = tqdm(total=len(pending), unit="item", desc="Batch Rating Progress")
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        if USE_CACHE:
            await get_or_create_cache(session, args.retries, args.timeout)
        tasks = [asyncio.create_task(process_single_question(q, session, semaphore, args.output, args.error, args.retries, args.timeout)) for q in pending]
        for task in asyncio.as_completed(tasks):
            await task
            pbar.update(1)
    pbar.close()
    print(f"\n✨ 批量打标结束，结果: {os.path.abspath(args.output)}")
    print(f"错误日志: {os.path.abspath(args.error)}")


if __name__ == "__main__":
    start = time.time()
    try:
        asyncio.run(main_batch_run())
    except KeyboardInterrupt:
        print("\n收到键盘中断信号，程序已安全退出。")
    except Exception as exc:
        print(f"\n批量运行失败: {exc}")
    print(f"本次打标运行耗时: {round((time.time() - start) / 60, 2)} 分钟。")
