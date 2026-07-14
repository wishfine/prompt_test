# -*- coding: utf-8 -*-
"""
@File    : chemistry_difficulty_rating_with_cache.py
@Description:
    基于前缀缓存（Prompt Cache）和高并发的初中化学题目难度批量评级脚本。
    v4：基于100题人工复核结果，收紧“标准实验题虚高”和“压轴题虚高”，增强金属滤渣滤液、流程/图表/守恒题的拔高识别。
    v5：基于第二轮100题复核结果，小修5类边界：化学史送分、标准实验多问基础、空气含量压强曲线中等、NaHCO3纯度拔高、常见物质转化推断中等。
    v6：基于300题复核结果，小修8类边界：化学发展简史送分、溶液分类基础、CO还原氧化铁+燃烧条件组合中等、陌生复杂方程式配平中等、陌生材料迁移中等、红磷气压曲线中等、标准碳酸钠沉淀纯度表格中等、KClO3单反应质量图中等。
    v6.1：冻结分类逻辑，仅增加后处理解释同步、postprocess_trace、feature_audit_flags，不改变最终 difficulty_level 的判定规则。
    设计理念对齐现有物理脚本：Prompt 前缀缓存 + JSON 容错解析 + 18维特征归一化 + 后处理双向纠偏。
    难度级别：送分题 / 基础题 / 中等题 / 拔高题 / 压轴题。
"""

import os
import sys
import json
import re
import random
import time
import hashlib
import asyncio
import aiofiles
import aiohttp
import argparse
try:
    import json_repair
except Exception:
    class _JsonRepairFallback:
        @staticmethod
        def loads(text):
            return json.loads(text)
    json_repair = _JsonRepairFallback()
from typing import Dict, Any, Optional, List, Tuple
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
_temperature_raw = os.getenv("TEMPERATURE", "").strip()
TEMPERATURE = float(_temperature_raw) if _temperature_raw else None

FILE_LOCK = Lock()
CACHE_LOCK = Lock()
CACHE_GET_LOCK = Lock()

CACHE_EXPIRE_DAYS = 6
CACHE_EXPIRE_SECONDS = CACHE_EXPIRE_DAYS * 24 * 3600
CACHE_FILE_PATH = "chemistry_prompt_cache.json"

DIFFICULTY_RATING_PROMPT_PREFIX = ""
DIFFICULTY_RATING_PROMPT_SUFFIX = ""

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
    "step_count": "1-2步",
    "equation_count": "0-1个",
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
    "chemistry_process_count": "单一事实",
    "constraint_count": "无约束",
    "evidence_relation": "无证据链",
    "experiment_requirement": "无",
    "graph_table_requirement": "无",
    "error_risk": "无明显易错点",
}

ALLOWED_FEATURE_VALUES = {
    "step_count": {"1-2步", "3-5步", "6-8步", "9-12步", "12步以上"},
    "equation_count": {"0-1个", "2-3个", "4-6个", "7个以上"},
    "calculation_complexity": {"口算或直接判断", "简单笔算", "化学方程式计算或关系式计算", "复杂守恒或图像计算"},
    "reasoning_chain": {"直接套用", "简单因果推理", "多层证据推理", "逆向推理或方案评价"},
    "problem_structure": {"概念判断", "化学用语与分类", "方程式书写", "实验基础操作", "实验探究", "工艺流程", "图像表格分析", "物质推断", "计算综合", "跨模块综合"},
    "additional_structure": {"无", "微观示意图", "实验装置", "流程图", "图像表格", "探究材料", "多模块综合"},
    "information_carrier": {"纯文字", "单图识别", "微观示意图", "实验装置图", "流程图", "图像或表格", "多图表综合"},
    "reality_question": {"是", "否"},
    "subquestion_dependency": {"无多问", "多问但相互独立", "多问且层层递进"},
    "knowledge_count": {"1个", "2-3个", "4个及以上"},
    "knowledge_diff": {"低", "中", "高"},
    "cross_module": {"同一模块内部", "跨模块综合"},
    "chemistry_process_count": {"单一事实", "单一反应", "2-3个反应或过程", "多反应连续转化或流程"},
    "constraint_count": {"无约束", "单一约束", "多约束"},
    "evidence_relation": {"无证据链", "单一现象对应", "多现象证据链", "证据冲突与排除"},
    "experiment_requirement": {"无", "基础操作或读数", "控制变量或现象分析", "方案设计或误差评价"},
    "graph_table_requirement": {"无", "直接读数", "多组比较归纳", "图像反推或拐点分析"},
    "error_risk": {"无明显易错点", "轻微易错点", "明显易错点", "高易错点"},
}

ENUM_NORMALIZE = {
    "equation_count": {
        "公式数量": "0-1个",
        "0个": "0-1个",
        "1个": "0-1个",
        "1-2个": "2-3个",
        "1-3个": "2-3个",
        "2个": "2-3个",
        "3个": "2-3个",
        "4个以上": "4-6个",
        "7个以上方程式": "7个以上",
    },
    "knowledge_count": {
        "1-2个": "2-3个",
        "2个": "2-3个",
        "3个": "2-3个",
        "2-4个": "2-3个",
        "4个以上": "4个及以上",
        "多个": "4个及以上",
    },
    "information_carrier": {
        "图像": "图像或表格",
        "图象": "图像或表格",
        "表格": "图像或表格",
        "实验图": "实验装置图",
        "装置图": "实验装置图",
        "流程": "流程图",
        "流程图+表格": "多图表综合",
        "实验装置图+表格": "多图表综合",
        "实验装置图和图像": "多图表综合",
    },
    "additional_structure": {
        "实验图": "实验装置",
        "实验装置图": "实验装置",
        "图像": "图像表格",
        "表格": "图像表格",
        "图表": "图像表格",
        "流程": "流程图",
        "流程图": "流程图",
        "项目式": "探究材料",
    },
    "experiment_requirement": {
        "方案设计": "方案设计或误差评价",
        "误差分析": "方案设计或误差评价",
        "误差评价": "方案设计或误差评价",
        "控制变量": "控制变量或现象分析",
        "现象分析": "控制变量或现象分析",
        "故障分析": "控制变量或现象分析",
        "数据归纳": "控制变量或现象分析",
    },
    "graph_table_requirement": {
        "图像反推": "图像反推或拐点分析",
        "图象反推": "图像反推或拐点分析",
        "拐点分析": "图像反推或拐点分析",
        "直接读取": "直接读数",
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
    fixed: Dict[str, Any] = {}
    key_aliases = {
        "formula_count": "equation_count",
        "chemical_equation_count": "equation_count",
        "equations_count": "equation_count",
        "reaction_count": "chemistry_process_count",
        "process_count": "chemistry_process_count",
        "state_count": "chemistry_process_count",
        "variable_relation": "evidence_relation",
    }
    for k, v in (features or {}).items():
        clean_key = str(k).strip().strip('",， \n\t')
        clean_key = key_aliases.get(clean_key, clean_key)
        for standard_key in FEATURE_DEFAULTS.keys():
            if standard_key in clean_key:
                clean_key = standard_key
                break
        fixed[clean_key] = v
    return fixed


def normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    features = normalize_feature_keys(features or {})
    normalized: Dict[str, str] = {}
    for field, default in FEATURE_DEFAULTS.items():
        value = features.get(field, default)
        if field in ENUM_NORMALIZE and value in ENUM_NORMALIZE[field]:
            value = ENUM_NORMALIZE[field][value]
        clean_value = clean_enum_value(value)
        if field in ENUM_NORMALIZE and clean_value in ENUM_NORMALIZE[field]:
            value = ENUM_NORMALIZE[field][clean_value]
        if value in ALLOWED_FEATURE_VALUES[field]:
            normalized[field] = value
            continue
        value = canonicalize_feature_value(field, value)
        if value not in ALLOWED_FEATURE_VALUES[field]:
            value = default
        normalized[field] = value

    # 结构联动修正：实验/流程/图像载体应反映到 additional_structure。
    if normalized["problem_structure"] == "工艺流程" and normalized["additional_structure"] == "无":
        normalized["additional_structure"] = "流程图"
    if normalized["problem_structure"] == "实验探究" and normalized["additional_structure"] == "无":
        normalized["additional_structure"] = "探究材料"
    if normalized["problem_structure"] == "图像表格分析" and normalized["additional_structure"] == "无":
        normalized["additional_structure"] = "图像表格"

    return normalized

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


def set_level_with_reason(rating_result: Dict[str, Any], level: str, core_basis_prefix: str) -> None:
    """设置后处理难度，并记录可审计的改档轨迹。

    v6.1 说明：
    - 不改变任何分类规则，只把每一次自动升/降档记录到 postprocess_trace；
    - 后续由 sync_reasoning_after_postprocess() 统一同步 why_not_lower / why_not_higher，
      避免最终档位与原始模型解释互相矛盾。
    """
    previous_level = rating_result.get("difficulty_level", "")
    rating_result.setdefault("postprocess_original_level", previous_level)
    rating_result.setdefault("postprocess_trace", [])
    if previous_level != level:
        rating_result["postprocess_trace"].append({
            "from": previous_level,
            "to": level,
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
    return high_level_feature_count(features, data) < 4 or not has_final_core_combo(features, data)


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


def add_feature_audit_flags(rating_result: Dict[str, Any], data: Dict[str, Any]) -> None:
    """增加 feature 审计标记，不参与最终难度决策。

    这些 flag 用于 HTML 人审或离线质量监控。它们只提示“features 可能需要人工关注”，
    不改变 difficulty_level、coarse_difficulty 或后处理分类结果。
    """
    features = rating_result.get("features") or {}
    level = rating_result.get("difficulty_level", "")
    flags: List[str] = []
    text_no_analysis = visible_text(data, include_analysis=False)
    text_all = visible_text(data, include_analysis=True)

    has_image_placeholder = "<image" in text_all.lower() or "[image" in text_all.lower() or "图片" in text_all
    has_image_url = bool(str(data.get("stem_pic_url", "") or "").strip() or str(data.get("analysis_pic_url", "") or "").strip())
    if (has_image_placeholder or has_image_url) and features.get("information_carrier") == "纯文字":
        flags.append("纯文本模式图像信息未进入模型：information_carrier=纯文字，图像类 feature 仅供参考")

    graph_markers = ["图", "图像", "图象", "曲线", "表", "数据", "压强", "气压", "质量变化", "坐标", "如下图", "如图"]
    if contains_any(text_all, graph_markers) and features.get("graph_table_requirement") == "无":
        flags.append("题干/解析存在图表或数据线索，但 graph_table_requirement=无，建议人审确认")

    flow_markers = ["流程", "工艺", "滤渣", "滤液", "转化关系", "框图", "A-G", "A～G", "A~G", "A-E", "A～E", "A~E"]
    if contains_any(text_all, flow_markers) and features.get("additional_structure") == "无":
        flags.append("题干/解析存在流程/框图/推断线索，但 additional_structure=无，建议人审确认")

    high_count = high_level_feature_count(features, data)
    if level == "拔高题" and high_count < 2:
        flags.append(f"拔高题但高阶特征计数偏低({high_count})，可能依赖题型规则或关键词升档")
    if level == "压轴题" and high_count < 4:
        flags.append(f"压轴题但高阶特征计数偏低({high_count})，建议人工复核压轴证据链")

    hard_markers = ["方案评价", "质疑", "可靠性", "补充实验", "干扰", "滤渣", "滤液", "先后反应", "过量", "不足", "拐点", "平台", "极值", "分类讨论", "复杂守恒", "元素守恒"]
    if level in ["送分题", "基础题"] and contains_any(text_all, hard_markers):
        flags.append("低档题中出现拔高关键词，若题目确有证据链/多约束/复杂计算，需人工确认是否低估")

    if rating_result.get("postprocess_trace"):
        flags.append("后处理已改档：reasoning 已按最终档位同步，原始模型解释仅作参考")

    # 去重并保持顺序。
    deduped: List[str] = []
    for flag in flags:
        if flag and flag not in deduped:
            deduped.append(flag)
    rating_result["feature_audit_flags"] = deduped


def postprocess_chemistry_difficulty(rating_result: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """化学后处理自动纠偏主流程。"""
    if not rating_result:
        return rating_result

    rating_result["features"] = normalize_features(rating_result.get("features", {}))
    normalize_reasoning_schema(rating_result)

    level = rating_result.get("difficulty_level", "")
    if level not in VALID_LEVELS:
        # 若模型输出非法等级，按特征粗略兜底。
        rating_result["difficulty_level"] = infer_level_from_features(rating_result["features"], data)
        set_level_with_reason(rating_result, rating_result["difficulty_level"], "自动修复：模型输出非法难度等级，已按特征兜底推断")

    if rating_result.get("difficulty_level") == "基础题" and should_downgrade_basic_to_easy(rating_result["features"], data):
        set_level_with_reason(rating_result, "送分题", "自动降档：无计算、无实验、无证据链的短纯概念直答题")

    if rating_result.get("difficulty_level") == "送分题":
        upgrade_reasons = should_upgrade_easy_to_basic(rating_result["features"], data)
        if upgrade_reasons:
            set_level_with_reason(rating_result, "基础题", "自动升档：" + "；".join(upgrade_reasons))

    if rating_result.get("difficulty_level") == "基础题":
        upgrade_reasons = should_upgrade_basic_to_medium(rating_result["features"], data)
        if upgrade_reasons:
            set_level_with_reason(rating_result, "中等题", "自动升档：" + "；".join(upgrade_reasons))

    if rating_result.get("difficulty_level") == "中等题" and should_downgrade_medium_to_basic(rating_result["features"], data):
        set_level_with_reason(rating_result, "基础题", "自动降档：短步骤、单反应/单概念题，未达到中等复杂度")

    if rating_result.get("difficulty_level") == "中等题":
        ok, reasons = should_upgrade_medium_to_hard(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "拔高题", "自动升档：" + "；".join(reasons))

    if rating_result.get("difficulty_level") == "拔高题":
        downgraded = should_downgrade_standard_experiment(rating_result["features"], data)
        if downgraded:
            set_level_with_reason(rating_result, downgraded, f"自动降档：标准实验/基础图表题，未达到拔高复杂度，降为{downgraded}")

    if rating_result.get("difficulty_level") == "拔高题":
        ok, reasons = should_upgrade_hard_to_final(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "压轴题", "自动升档：" + "；".join(reasons))

    if rating_result.get("difficulty_level") == "压轴题" and should_downgrade_final_to_hard(rating_result["features"], data):
        set_level_with_reason(rating_result, "拔高题", "自动降档：压轴核心组合不足或高阶特征不足")

    sync_coarse_difficulty(rating_result)
    sync_reasoning_after_postprocess(rating_result)
    add_feature_audit_flags(rating_result, data)
    return rating_result


def infer_level_from_features(features: Dict[str, Any], data: Dict[str, Any]) -> str:
    high = high_level_feature_count(features, data)
    if high >= 4:
        return "压轴题"
    if high >= 2 or features.get("step_count") == "9-12步":
        return "拔高题"
    if (
        features.get("step_count") == "6-8步"
        or features.get("experiment_requirement") in ["控制变量或现象分析", "方案设计或误差评价"]
        or features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或拐点分析"]
        or features.get("evidence_relation") in ["多现象证据链", "证据冲突与排除"]
    ):
        return "中等题"
    if should_downgrade_basic_to_easy(features, data):
        return "送分题"
    return "基础题"

# -------------------------- 5. 构建题目输入与模型调用 --------------------------
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

    sub_questions = data.get("sub_questions", []) or []
    if sub_questions:
        try:
            sub_questions.sort(key=lambda x: int(x.get("question_id", 0)) if isinstance(x, dict) else 0)
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
    question_content: str,
    session: aiohttp.ClientSession,
    retries: int,
    timeout_sec: int,
) -> Tuple[Dict[str, Any], float, int, int, int]:
    response_id = await get_or_create_cache(session, retries, timeout_sec)
    if not response_id:
        print("警告: 无法获取有效缓存 ID，终止单题请求")
        return {}, 0.0, 0, 0, 0

    dynamic_content = f"{question_content}{DIFFICULTY_RATING_PROMPT_SUFFIX}"

    for retry in range(retries):
        payload = {
            "model": MODEL_NAME,
            "previous_response_id": response_id,
            "input": [{"role": "user", "content": dynamic_content}],
            "thinking": {"type": "disabled"},
        }
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
                    return parsed_result, time.time() - t1, prompt_tokens, completion_tokens, total_tokens

                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    print(f"接口限流(429)，等待 {retry_after} 秒后进行第 {retry + 1} 次重试...")
                    await asyncio.sleep(retry_after)
                    continue

                error_text = await response.text()
                print(f"API请求失败 (状态码: {response.status}): {error_text[:200]}")
                if "InvalidParameter.PreviousResponseNotFound" in error_text:
                    print("检测到服务器缓存丢失，正在重建缓存...")
                    new_response_id = await create_prefix_cache(session, retries, timeout_sec)
                    if not new_response_id:
                        return {}, 0.0, 0, 0, 0
                    response_id = new_response_id
                    continue
                if response.status >= 500:
                    backoff = (2 ** retry) + random.uniform(0, 1)
                    print(f"服务器故障({response.status})，{backoff:.2f}秒后重试 (第{retry + 1}次)...")
                    await asyncio.sleep(backoff)
                    continue
                if 400 <= response.status < 500:
                    return {}, 0.0, 0, 0, 0
        except aiohttp.ClientError as e:
            backoff = (2 ** retry) + random.uniform(0, 1)
            if retry == retries - 1:
                print(f"网络异常最终失败: {e}")
                return {}, 0.0, 0, 0, 0
            print(f"网络出现异常: {e}，将进行退避 {backoff:.2f} 秒后重试...")
            await asyncio.sleep(backoff)
        except Exception as e:
            print(f"运行过程中请求异常: {e}")
            if retry == retries - 1:
                return {}, 0.0, 0, 0, 0
            new_response_id = await create_prefix_cache(session, retries, timeout_sec)
            if new_response_id:
                response_id = new_response_id
            await asyncio.sleep(1)

    return {}, 0.0, 0, 0, 0

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
        try:
            question_content = construct_question_content(data)
            rating_result, time_use, prompt_tokens, completion_tokens, total_tokens = await call_model_with_cache(
                question_content, session, retries, timeout_sec
            )

            rating_result = postprocess_chemistry_difficulty(rating_result, data)

            output_data = data.copy()
            output_data["difficulty_rating"] = rating_result
            output_data["api_time_use"] = round(time_use, 2)
            output_data["api_prompt_tokens"] = prompt_tokens
            output_data["api_completion_tokens"] = completion_tokens
            output_data["api_total_tokens"] = total_tokens

            if rating_result and rating_result.get("difficulty_level"):
                async with FILE_LOCK:
                    async with aiofiles.open(output_path, "a", encoding="utf-8") as f:
                        await f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            else:
                output_data["rating_error"] = "模型返回数据为空或格式错误"
                async with FILE_LOCK:
                    async with aiofiles.open(error_path, "a", encoding="utf-8") as f:
                        await f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        except Exception as e:
            error_data = data.copy()
            error_data["rating_error"] = f"question_id={question_id}; error={str(e)}"
            async with FILE_LOCK:
                async with aiofiles.open(error_path, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(error_data, ensure_ascii=False) + "\n")


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
                    if qid:
                        processed.add(qid)
                except Exception:
                    continue
    except Exception as e:
        print(f"扫描断点文件出错: {e}")
    return processed

# -------------------------- 7. 主执行流 --------------------------
async def main_batch_run() -> None:
    parser = argparse.ArgumentParser(description="初中化学难度评级多线程并发批量打标脚本 (带 Cache 优化)")
    parser.add_argument("-p", "--prompt", type=str, default="../prompts/初中化学难度打标提示词.txt", help="化学打标提示词文件路径")
    parser.add_argument("-i", "--input", type=str, default="../data/chemistry_sampled_5000_per_difficulty_v2.jsonl", help="输入待打标 JSONL 数据集路径")
    parser.add_argument("-o", "--output", type=str, default="chemistry_difficulty_rated_results.jsonl", help="输出保存打标结果的 JSONL 路径")
    parser.add_argument("-e", "--error", type=str, default="chemistry_difficulty_errors.jsonl", help="输出保存失败结果的 JSONL 路径")
    parser.add_argument("-c", "--concurrency", type=int, default=15, help="最大并发限制，默认 15")
    parser.add_argument("-t", "--timeout", type=int, default=180, help="单次 API 调用超时时间，默认 180 秒")
    parser.add_argument("-r", "--retries", type=int, default=3, help="失败最大重试次数，默认 3")
    parser.add_argument("-n", "--num", type=int, default=None, help="测试打标的限制数量（留空表示全部打标）")
    parser.add_argument("--seed", type=int, default=42, help="随机抽样/打乱的种子，默认 42")
    args = parser.parse_args()

    random.seed(args.seed)
    load_prompt_config(args.prompt)

    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在，终止运行！")
        sys.exit(1)

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
    to_process = [q for q in questions if q.get("question_id") not in processed_ids]
    print(f"数据比对完成: 已完成数 {len(processed_ids)}，待处理数 {len(to_process)}")

    if not to_process:
        print("所有题目都已完成打标！")
        return

    semaphore = Semaphore(args.concurrency)
    pbar = tqdm(total=len(to_process), unit="item", desc="Chemistry Rating Progress")

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
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
