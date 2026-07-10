# -*- coding: utf-8 -*-
"""
@File    : physics_difficulty_rating_with_cache.py
@Description:
    基于前缀缓存（Prompt Cache）和高并发的初中物理题目难度批量评级脚本。
    从 .env 读取 API 配置，自动根据物理特征字段进行后处理双向纠偏与校验。
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
import json_repair
from typing import Dict, Any, Optional, List
from tqdm.asyncio import tqdm
from asyncio import Lock, Semaphore
from dotenv import load_dotenv

# 加载 .env 配置
load_dotenv()

API_KEY = os.getenv("API_KEY", "not-needed")
BASE_URL = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1")
if not BASE_URL.endswith("/"):
    BASE_URL += "/"
MODEL_NAME = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")

# 全局锁
FILE_LOCK = Lock()
CACHE_LOCK = Lock()
CACHE_GET_LOCK = Lock()

# 缓存有效期配置
CACHE_EXPIRE_DAYS = 6
CACHE_EXPIRE_SECONDS = CACHE_EXPIRE_DAYS * 24 * 3600

# 全局提示词变量
DIFFICULTY_RATING_PROMPT_PREFIX = ""
DIFFICULTY_RATING_PROMPT_SUFFIX = ""
CACHE_FILE_PATH = "physics_prompt_cache.json"

# 难度数值映射
LEVEL_MAP = {
    "送分题": 1,
    "基础题": 2,
    "中等题": 3,
    "拔高题": 4,
    "压轴题": 5,
}

# -------------------------- 1. 提示词加载 --------------------------
def load_prompt_config(prompt_path: str):
    """动态解析提示词文件，支持 Python 格式与纯文本格式"""
    global DIFFICULTY_RATING_PROMPT_PREFIX, DIFFICULTY_RATING_PROMPT_SUFFIX
    
    if not os.path.exists(prompt_path):
        print(f"错误: 找不到提示词文件 {prompt_path}！")
        sys.exit(1)
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # 尝试 Python 变量 exec 执行
    try:
        namespace = {}
        exec(content, namespace)
        prefix = namespace.get("DIFFICULTY_RATING_PROMPT_PREFIX")
        suffix = namespace.get("DIFFICULTY_RATING_PROMPT_SUFFIX")
        if prefix and suffix:
            DIFFICULTY_RATING_PROMPT_PREFIX = prefix
            DIFFICULTY_RATING_PROMPT_SUFFIX = suffix
            print("成功以 Python 变量结构解析提示词")
            return
    except Exception:
        pass
        
    # 尝试纯文本切分
    if "## 输入题目信息" in content:
        parts = content.split("## 输入题目信息")
        DIFFICULTY_RATING_PROMPT_PREFIX = parts[0] + "## 输入题目信息"
        DIFFICULTY_RATING_PROMPT_SUFFIX = "\n\n请根据以上信息，对题目进行全面的难度分析和评级。"
        print("成功以 纯文本标志位 结构切分并解析提示词")
        return
        
    raise ValueError("提示词格式不正确，既不是有效的 Python 变量文件，也没有包含 '## 输入题目信息' 分割标志。")

# -------------------------- 2. 缓存管理模块 --------------------------
def compute_text_hash(text: str) -> str:
    """计算文本的 SHA256，用于检验前缀缓存的一致性"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

async def load_cache() -> Dict[str, Any]:
    """读取本地缓存记录"""
    async with CACHE_LOCK:
        if not os.path.exists(CACHE_FILE_PATH):
            return {}
        try:
            async with aiofiles.open(CACHE_FILE_PATH, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content else {}
        except Exception as e:
            print(f"加载缓存文件失败: {e}")
            return {}

async def save_cache(cache_data: Dict[str, Any]) -> None:
    """保存缓存记录到本地"""
    async with CACHE_LOCK:
        try:
            async with aiofiles.open(CACHE_FILE_PATH, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(cache_data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"保存缓存文件失败: {e}")

def is_cache_valid(cache_entry: Dict[str, Any], current_time: int) -> bool:
    """检查缓存是否有效"""
    if not cache_entry:
        return False
    expire_at = cache_entry.get('expire_at', 0)
    if current_time >= expire_at:
        return False
    expected_hash = compute_text_hash(DIFFICULTY_RATING_PROMPT_PREFIX)
    actual_hash = cache_entry.get('prefix_hash', '')
    return expected_hash == actual_hash

async def get_valid_cache() -> Optional[Dict[str, Any]]:
    """获取缓存"""
    cache_data = await load_cache()
    current_time = int(time.time())
    cache_entry = cache_data.get('prompt_prefix_cache')
    if is_cache_valid(cache_entry, current_time):
        return cache_entry
    return None

async def set_cache(response_id: str, expire_at: int) -> None:
    """写入缓存"""
    cache_data = await load_cache()
    cache_entry = {
        'response_id': response_id,
        'expire_at': expire_at,
        'prefix_hash': compute_text_hash(DIFFICULTY_RATING_PROMPT_PREFIX),
        'created_at': int(time.time())
    }
    cache_data['prompt_prefix_cache'] = cache_entry
    await save_cache(cache_data)

async def create_prefix_cache(session: aiohttp.ClientSession, retries: int, timeout_sec: int) -> Optional[str]:
    """发起请求创建前缀缓存"""
    current_time = int(time.time())
    expire_at = current_time + CACHE_EXPIRE_SECONDS
    
    payload = {
        "model": MODEL_NAME,
        "input": [{"role": "user", "content": DIFFICULTY_RATING_PROMPT_PREFIX}],
        "thinking": {"type": "disabled"},
        "expire_at": expire_at,
        "caching": {"type": "enabled", "prefix": True}
    }
    
    t1 = time.time()
    for attempt in range(retries):
        try:
            async with session.post(
                f"{BASE_URL}responses",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"创建前缀缓存失败 (状态码: {response.status}): {error_text[:200]}")
                    if 400 <= response.status < 500:
                        return None
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                result = await response.json()
                response_id = result.get('id')
                if response_id:
                    await set_cache(response_id, expire_at)
                    t2 = time.time()
                    print(f"前缀缓存创建成功，耗时: {t2 - t1:.2f}秒，缓存ID: {response_id}")
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
    """线程/协程安全地获取或创建缓存"""
    async with CACHE_GET_LOCK:
        cache_entry = await get_valid_cache()
        if cache_entry:
            return cache_entry['response_id']
        print("未找到有效缓存，正在向服务器创建前缀缓存...")
        return await create_prefix_cache(session, retries, timeout_sec)

# -------------------------- 3. 后处理纠偏升降档核心 --------------------------

# 枚举归一化词典，解决模型输出近义词、非标准枚举导致后处理失效的问题
ENUM_NORMALIZE = {
    "formula_count": {
        "1-3个": "2-3个",
        "1-2个": "2-3个",
        "2-4个": "2-3个",
        "4个以上": "4-6个",
        "7个以上公式": "7个以上",
    },
    "knowledge_count": {
        "1-2个": "2-3个",
        "2个": "2-3个",
        "3个": "2-3个",
        "2-4个": "2-3个",
        "4个以上": "4个及以上",
        "4个及以上知识点": "4个及以上",
        "多个独立识记点，合计4个及以上": "4个及以上",
    },
    "knowledge_diff": {
        "低到中": "中",
        "中到高": "高",
        "中高": "高",
        "较高": "高",
        "简单": "低",
        "较低": "低",
    },
    "information_carrier": {
        "图像": "图像或表格",
        "表格": "图像或表格",
        "图像和表格": "图像或表格",
        "图像加电路图": "多图表综合",
        "图像和电路图": "多图表综合",
        "电路图和图像": "多图表综合",
        "电路图+图像": "多图表综合",
        "图像+电路图": "多图表综合",
        "电路图+表格": "多图表综合",
        "表格+电路图": "多图表综合",
        "实验图": "实验装置图",
        "装置图": "实验装置图",
        "电路图+图像或表格": "多图表综合",
        "电路图+表格或图像": "多图表综合",
        "实验装置图和表格": "多图表综合",
        "实验装置图+表格": "多图表综合",
    },
    "graph_table_requirement": {
        "图像反推": "图像反推或外推",
        "图像外推": "图像反推或外推",
        "图像反推外推": "图像反推或外推",
        "直接读取": "直接读数",
    },
    "experiment_requirement": {
        "方案设计": "方案设计或误差评价",
        "误差分析": "方案设计或误差评价",
        "误差评价": "方案设计或误差评价",
        "控制变量": "控制变量或故障分析",
        "故障分析": "控制变量或故障分析",
        "方案设计或误差分析": "方案设计或误差评价",
        "数据归纳": "控制变量或故障分析",
    },
    "additional_structure": {
        "电路图": "电路约束",
        "实验装置": "实验探究",
        "图像": "图像表格",
        "表格": "图像表格",
        "控制变量或故障分析": "实验探究",
        "方案设计或误差评价": "实验探究",
        "实验要求": "实验探究",
        "实验分析": "实验探究",
        "多约束": "跨模块",
        "多图表": "图像表格",
        "表格数据": "图像表格",
        "多图表综合": "图像表格",
    },
    "problem_structure": {
        "声学计算": "光学声学综合",
        "声学综合": "光学声学综合",
        "光学综合": "光学声学综合",
        "电路分析": "电路综合",
        "力学分析": "力学综合",
        "电学综合": "电路综合",
        "概念判断+直接计算": "直接计算",
        "磁学作图": "电路综合",
        "作图题": "力学综合",
    },
    "state_count": {
        "单状态变化": "单状态",
        "连续变化": "连续变化或临界状态",
        "单状态/连续变化": "连续变化或临界状态",
        "单状态 / 连续变化": "连续变化或临界状态",
        "单状态 / 双状态": "双状态",
        "单状态/双状态": "双状态",
    },
    "calculation_complexity": {
        "直接套用": "口算或直接判断",
        "简单因果推理": "简单笔算",
        "多层因果推理": "简单笔算",
        "逆向推理或临界分析": "复杂方程或范围计算",
    }
}

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
    "error_risk": "无明显易错点"
}

# 合法枚举全集：用于 schema guard 兜底，保证后处理后的 features 一定合法
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

def clean_enum_value(value: Any) -> str:
    """将模型输出的 feature value 转为便于规则判断的短字符串。"""
    if value is None:
        return ""
    v = str(value).strip()
    # 统一常见分隔符和空白，保留中文语义关键词
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
    return v.strip('",;:。.')

def canonicalize_feature_value(field: str, value: Any) -> str:
    """按字段语义进行通用归一化。

    设计目标：
    1. 高频固定别名仍由 ENUM_NORMALIZE 精确处理；
    2. 长解释、混合枚举、带标点枚举由本函数按关键词归并；
    3. 最终 normalize_features 会再次用 ALLOWED_FEATURE_VALUES 做兜底校验。
    """
    v = clean_enum_value(value)
    if not v:
        return FEATURE_DEFAULTS.get(field, "")

    if field == "step_count":
        if "12" in v or "十二" in v:
            return "12步以上"
        if "9-12" in v or "9到12" in v or "九" in v or "十" in v or "11" in v:
            return "9-12步"
        if "6-8" in v or "6到8" in v or "六" in v or "七" in v or "八" in v:
            return "6-8步"
        if "3-5" in v or "3到5" in v or "三" in v or "四" in v or "五" in v:
            return "3-5步"
        return "1-2步"

    if field == "formula_count":
        if "7" in v or "七" in v:
            return "7个以上"
        if "4-6" in v or "4到6" in v or "4个以上" in v or "四" in v or "五" in v or "六" in v:
            return "4-6个"
        if "2-3" in v or "2到3" in v or "2个" in v or "3个" in v or "两" in v or "二" in v or "三" in v:
            return "2-3个"
        return "0-1个"

    if field == "calculation_complexity":
        if any(k in v for k in ["复杂", "范围", "极值", "不等式", "分类", "方程组"]):
            return "复杂方程或范围计算"
        if any(k in v for k in ["多公式", "联立", "多步计算", "串联计算"]):
            return "多公式联立"
        if any(k in v for k in ["简单", "笔算", "计算", "代入", "单位换算"]):
            return "简单笔算"
        return "口算或直接判断"

    if field == "reasoning_chain":
        if any(k in v for k in ["逆向", "临界", "反推", "分类", "极值", "范围"]):
            return "逆向推理或临界分析"
        if any(k in v for k in ["多层", "多步", "归纳", "综合推理", "链条"]):
            return "多层因果推理"
        if any(k in v for k in ["简单", "因果", "常规推理"]):
            return "简单因果推理"
        return "直接套用"

    if field == "problem_structure":
        # 先识别跨模块，避免“电热综合”被单独归为电路或热学
        module_hits = 0
        if any(k in v for k in ["电", "电路", "电学", "电磁"]):
            module_hits += 1
        if any(k in v for k in ["力", "运动", "密度", "压强", "浮力", "杠杆", "滑轮", "机械", "做功", "受力"]):
            module_hits += 1
        if any(k in v for k in ["热", "温度", "比热", "内能", "物态"]):
            module_hits += 1
        if any(k in v for k in ["光", "声", "凸透镜", "平面镜"]):
            module_hits += 1
        if "跨" in v or module_hits >= 2:
            return "跨模块综合"
        # “密度计算/运动学计算/运动计算”等无综合关系的结构，按直接计算处理
        if "计算" in v and "综合" not in v and not any(k in v for k in ["电路", "电学", "电磁"]):
            return "直接计算"
        if any(k in v for k in ["实验", "探究", "测量"]):
            return "实验探究"
        if any(k in v for k in ["电", "电路", "电学", "电磁"]):
            return "电路综合"
        if any(k in v for k in ["力", "运动", "密度", "压强", "浮力", "杠杆", "滑轮", "机械", "做功", "受力", "作图题"]):
            return "力学综合"
        if any(k in v for k in ["热", "温度", "比热", "内能", "物态"]):
            return "热学综合"
        if any(k in v for k in ["光", "声", "凸透镜", "平面镜"]):
            return "光学声学综合"
        if any(k in v for k in ["图像", "表格", "图表"]):
            return "图像表格分析"
        if "计算" in v:
            return "直接计算"
        return "概念判断"

    if field == "additional_structure":
        if any(k in v for k in ["实验", "探究", "装置", "方案", "控制变量", "故障", "误差"]):
            return "实验探究"
        if any(k in v for k in ["电路", "电压", "电流", "电表", "量程", "滑动变阻器"]):
            return "电路约束"
        if any(k in v for k in ["力", "压强", "浮力", "杠杆", "滑轮", "受力", "机械", "多约束"]):
            return "力学约束"
        if any(k in v for k in ["图像", "表格", "图表", "多图"]):
            return "图像表格"
        if "跨" in v:
            return "跨模块"
        return "无"

    if field == "information_carrier":
        has_circuit = "电路图" in v or "电路" in v
        has_exp = "实验装置" in v or "装置图" in v or "实验图" in v
        has_graph = "图像" in v or "图象" in v or "曲线" in v or "图" in v
        has_table = "表格" in v or "表" in v
        if has_circuit and (has_graph or has_table):
            return "多图表综合"
        if has_exp and (has_graph or has_table):
            return "多图表综合"
        if has_exp:
            return "实验装置图"
        if has_circuit:
            return "电路图"
        if "多图" in v or "多表" in v or ("图像" in v and "表格" in v) or "图表综合" in v:
            return "多图表综合"
        if has_graph or has_table or "图表" in v:
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
        if any(k in v for k in ["4个及以上", "4个以上", "四个", "多个", "多知识点", "合计4"]):
            return "4个及以上"
        if any(k in v for k in ["2-3", "2到3", "2个", "3个", "两", "二", "三"]):
            return "2-3个"
        if any(k in v for k in ["1个", "一个", "单一"]):
            return "1个"
        # 非空但无法判断时，按保守中间值处理，避免错误低估为 1 个
        return "2-3个"

    if field == "knowledge_diff":
        if "高" in v or "难" in v or "复杂" in v:
            return "高"
        if "中" in v or "一般" in v:
            return "中"
        if "低" in v or "简单" in v or "基础" in v:
            return "低"
        return "中"

    if field == "cross_module":
        if "跨" in v:
            return "跨模块综合"
        if any(k in v for k in ["内部", "同一", "力学", "热学", "电学", "光学", "声学"]):
            return "同一模块内部"
        return "同一模块内部"

    if field == "state_count":
        if any(k in v for k in ["连续", "临界", "范围变化", "动态变化"]):
            return "连续变化或临界状态"
        if "多状态" in v or "多个状态" in v:
            return "多状态"
        if any(k in v for k in ["双状态", "两状态", "两个状态", "2个状态"]):
            return "双状态"
        return "单状态"

    if field == "constraint_count":
        if "多" in v:
            return "多约束"
        if any(k in v for k in ["单", "一个", "1个", "有约束", "约束"]):
            return "单一约束"
        return "无约束"

    if field == "variable_relation":
        if any(k in v for k in ["多变量", "耦合", "多个变量"]):
            return "多变量耦合关系"
        if any(k in v for k in ["函数", "图像", "图象", "曲线", "二次"]):
            return "图像函数关系"
        if any(k in v for k in ["正比", "反比", "比例"]):
            return "简单正反比"
        return "无变量关系"

    if field == "experiment_requirement":
        if any(k in v for k in ["方案", "设计", "误差", "评价", "表达式", "缺表", "等效替代", "特殊方法"]):
            return "方案设计或误差评价"
        if any(k in v for k in ["控制变量", "故障", "数据归纳", "多组比较", "归纳", "探究", "分析"]):
            return "控制变量或故障分析"
        if any(k in v for k in ["读数", "操作", "测量", "基础"]):
            return "基础操作或读数"
        return "无"

    if field == "graph_table_requirement":
        # “直接读数比较/直接读数描点/直接作图”仍属于直接读数或直接处理，
        # 不因包含“比较”二字误升为多组归纳。
        if "直接" in v and any(k in v for k in ["读数", "读取", "描点", "作图"]):
            return "直接读数"
        if any(k in v for k in ["反推", "外推", "函数关系", "曲线关系"]):
            return "图像反推或外推"
        if any(k in v for k in ["多组", "比较", "归纳"]):
            return "多组比较归纳"
        if any(k in v for k in ["读数", "读取", "描点", "作图"]):
            return "直接读数"
        return "无"

    if field == "error_risk":
        if "高" in v:
            return "高易错点"
        if "明显" in v or "较大" in v:
            return "明显易错点"
        if "轻微" in v or "较小" in v:
            return "轻微易错点"
        return "无明显易错点"

    return FEATURE_DEFAULTS.get(field, "")

def normalize_feature_keys(features: Dict[str, Any]) -> Dict[str, Any]:
    """修复模型偶发输出的异常 feature key"""
    fixed = {}

    for k, v in features.items():
        clean_key = str(k).strip().strip('",， \n\t')

        if "step_count" in clean_key:
            clean_key = "step_count"
        elif "formula_count" in clean_key:
            clean_key = "formula_count"
        elif "calculation_complexity" in clean_key:
            clean_key = "calculation_complexity"
        elif "reasoning_chain" in clean_key:
            clean_key = "reasoning_chain"
        elif "problem_structure" in clean_key:
            clean_key = "problem_structure"
        elif "additional_structure" in clean_key:
            clean_key = "additional_structure"
        elif "information_carrier" in clean_key:
            clean_key = "information_carrier"
        elif "reality_question" in clean_key:
            clean_key = "reality_question"
        elif "subquestion_dependency" in clean_key:
            clean_key = "subquestion_dependency"
        elif "knowledge_count" in clean_key:
            clean_key = "knowledge_count"
        elif "knowledge_diff" in clean_key:
            clean_key = "knowledge_diff"
        elif "cross_module" in clean_key:
            clean_key = "cross_module"
        elif "state_count" in clean_key:
            clean_key = "state_count"
        elif "constraint_count" in clean_key:
            clean_key = "constraint_count"
        elif "variable_relation" in clean_key:
            clean_key = "variable_relation"
        elif "experiment_requirement" in clean_key:
            clean_key = "experiment_requirement"
        elif "graph_table_requirement" in clean_key:
            clean_key = "graph_table_requirement"
        elif "error_risk" in clean_key:
            clean_key = "error_risk"

        fixed[clean_key] = v

    return fixed

def fill_missing_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """补齐缺失字段，避免后处理取空字符串误触发"""
    for k, default in FEATURE_DEFAULTS.items():
        if not features.get(k):
            features[k] = default
    return features

def normalize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """规范模型输出中的非标准特征 key 和 value，并强制 schema 合法。

    处理顺序：
    1. 修复异常 key；
    2. 补齐缺失字段；
    3. 精确别名映射 ENUM_NORMALIZE；
    4. 合法枚举直接保留；
    5. 非法/长解释值按字段语义 canonicalize；
    6. 最终兜底，保证所有 features 字段都在 ALLOWED_FEATURE_VALUES 内。
    """
    if not features:
        features = {}

    features = normalize_feature_keys(features)
    features = fill_missing_features(features)

    normalized = {}

    for field, default in FEATURE_DEFAULTS.items():
        value = features.get(field, default)

        # 第1层：精确映射，处理高频固定别名
        if field in ENUM_NORMALIZE and value in ENUM_NORMALIZE[field]:
            value = ENUM_NORMALIZE[field][value]

        # 对去空白后的值再试一次精确映射，兼容“ 电路图、图像 ”这类输出
        clean_value = clean_enum_value(value)
        if field in ENUM_NORMALIZE and clean_value in ENUM_NORMALIZE[field]:
            value = ENUM_NORMALIZE[field][clean_value]

        # 第2层：已经是合法枚举，直接保留
        if value in ALLOWED_FEATURE_VALUES[field]:
            normalized[field] = value
            continue

        # 第3层：通用规则归一化
        value = canonicalize_feature_value(field, value)

        # 第4层：最终兜底，保证 schema 一定合法
        if value not in ALLOWED_FEATURE_VALUES[field]:
            value = default

        normalized[field] = value

    # 修正 additional_structure 中由“多约束/跨模块”等泛化值带来的结构类型
    # 注意：这里只修正附加结构，不改变 problem_structure 本身
    if normalized.get("additional_structure") == "跨模块":
        problem = normalized.get("problem_structure", "")
        if problem == "电路综合":
            normalized["additional_structure"] = "电路约束"
        elif problem == "力学综合":
            normalized["additional_structure"] = "力学约束"
        elif problem == "实验探究":
            normalized["additional_structure"] = "实验探究"

    return normalized

def normalize_reasoning_schema(rating_result: Dict[str, Any]) -> None:
    """修复 reasoning / reason 字段结构，将其规范合并为符合 prompt 规定的四段结构"""
    reasoning = rating_result.get("reasoning")
    reason = rating_result.get("reason")
    
    normalized = {
        "core_basis": "",
        "hard_point": "",
        "why_not_lower": "",
        "why_not_higher": ""
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
    """更改难度级别并注入原因理由"""
    rating_result["difficulty_level"] = level
    reasoning = rating_result.setdefault("reasoning", {
        "core_basis": "",
        "hard_point": "",
        "why_not_lower": "",
        "why_not_higher": ""
    })
    original_basis = reasoning.get("core_basis", "")
    reasoning["core_basis"] = f"【{core_basis_prefix}】。原始依据：{original_basis}"

def sync_coarse_difficulty(rating_result: Dict[str, Any]) -> None:
    """后处理评级修正后，自动同步粗定档 coarse_difficulty 字段以维护一致性"""
    level = rating_result.get("difficulty_level", "")
    if level in ["送分题", "基础题"]:
        rating_result["coarse_difficulty"] = "送分/基础区间（1-2档）"
    elif level == "中等题":
        rating_result["coarse_difficulty"] = "基础/中等区间（2-3档）"
    elif level == "拔高题":
        rating_result["coarse_difficulty"] = "中等/拔高区间（3-4档）"
    elif level == "压轴题":
        rating_result["coarse_difficulty"] = "拔高/压轴区间（4-5档）"

# 长文本新材料豁免词表：防止复杂背景被误降为送分题
# 注意：不要放“折射/滑轮/杠杆”这种泛词，否则简单概念题也会被误杀。
LONG_CONTEXT_KEYWORDS = [
    "阅读", "材料", "新定义", "自制", "传感器",
    "静电感应", "电磁继电器", "热敏", "光敏", "压敏",
    "控制电路", "自动控制", "报警器", "电子秤", "拉力计",
    "方案设计", "误差分析"
]

def is_long_context_or_new_situation(data: Dict[str, Any]) -> bool:
    """是否是偏长情境、需要建模分析的题目。主要看题干，避免被长解析误杀。"""
    if not data:
        return False

    stem = data.get("stem", "") or ""
    analysis = data.get("analysis", "") or ""

    # 主要看题干长度，不用 stem+analysis 的总长度
    if len(stem) > 240:
        return True

    # 新材料/新装置类关键词，出现在题干中才强保护
    if any(k in stem for k in LONG_CONTEXT_KEYWORDS):
        return True

    # 解析中出现复杂词，且题干也不短，才认为是复杂情境
    if len(stem) > 120 and any(k in analysis for k in LONG_CONTEXT_KEYWORDS):
        return True

    return False

def is_trivial_concept_question(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """是否为纯粹的、无计算的课本常识或生活常识概念直答题（针对现实生活情境豁免升档）"""
    text = data.get("stem", "") or ""
    return (
        len(text) < 70
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("graph_table_requirement") == "无"
        and features.get("knowledge_diff") == "低"
        and features.get("step_count") == "1-2步"
    )

def contains_any(text: str, keywords: List[str]) -> bool:
    """判断文本是否命中任一关键词"""
    return any(k in text for k in keywords)


def is_choice_like_question(text: str) -> bool:
    """是否为选择/判断辨析类题目"""
    return (
        "下列" in text
        or "说法" in text
        or "正确的是" in text
        or "不正确的是" in text
        or "错误的是" in text
        or "符合题意的是" in text
        or "不符合题意的是" in text
        or "选项" in text
    )


def count_fill_blanks(text: str) -> int:
    """粗略统计填空数量"""
    return len(re.findall(r"_{2,}|（\s*）|\(\s*\)", text))


def has_formula_calculation_intent(text: str) -> bool:
    """是否明显需要代入物理公式或进行物理量计算。
    教师口径下，凡是真正需要公式代入/简单应用的题，至少是基础题，不应降为送分题。
    """
    calc_cues = [
        "求", "计算", "多少", "多大", "为多少", "等于多少",
        "压强", "压力", "浮力", "密度", "速度", "路程", "时间",
        "电流", "电压", "电阻", "电功率", "功率", "电能", "热量",
        "机械效率", "有用功", "总功", "变阻器", "量程", "额定"
    ]
    return contains_any(text, calc_cues)

# --- 升降档条件判断 ---

def should_downgrade_basic_to_easy(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """规则：基础题 -> 送分题 降档
    教师口径对齐版：送分题只给纯识记/直接识别题。
    只要需要真正的物理公式代入、简单应用、受力/电路/实验/图表分析，就不降为送分。
    """
    if is_long_context_or_new_situation(data):
        return False

    stem = data.get("stem", "") or ""
    options = data.get("options", "") or ""
    analysis = data.get("analysis", "") or ""
    text = stem + "\n" + options + "\n" + analysis

    problem = features.get("problem_structure")

    # 教师口径：直接公式代入属于“较易/基础”，不是“容易/送分”。
    if problem == "直接计算" or has_formula_calculation_intent(text):
        return False

    # 只有纯概念/纯识记/直接现象识别才允许降为送分。
    simple_structure = problem in ["概念判断", "光学声学综合", "热学综合"]
    forbidden_structure = problem in [
        "实验探究",
        "电路综合",
        "力学综合",
        "跨模块综合",
        "图像表格分析",
    ]

    return (
        simple_structure
        and not forbidden_structure
        and features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_diff") == "低"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )

def should_upgrade_easy_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    """规则：送分题 -> 基础题 升档
    教师口径对齐版：送分题必须是一眼识别/直接记忆。
    凡是需要公式代入、简单应用、简单物理过程分析、读图、实验或电路/受力判断，至少基础题。
    """
    reasons = []

    stem = data.get("stem", "") or ""
    options = data.get("options", "") or ""
    analysis = data.get("analysis", "") or ""

    visible_text = stem + "\n" + options
    text = visible_text + "\n" + analysis

    if features.get("step_count") != "1-2步":
        reasons.append('解题步骤数为"{}"'.format(features.get("step_count")))

    if features.get("knowledge_count") != "1个":
        reasons.append('知识点数量为"{}"'.format(features.get("knowledge_count")))

    if features.get("state_count") != "单状态":
        reasons.append('物理状态数量为"{}"'.format(features.get("state_count")))

    if features.get("constraint_count") != "无约束":
        reasons.append("存在物理约束条件")

    if features.get("information_carrier") in ["电路图", "实验装置图", "多图表综合", "图像或表格"]:
        reasons.append('信息载体为"{}"，需要识图或读图'.format(features.get("information_carrier")))

    if features.get("experiment_requirement") != "无":
        reasons.append("含有实验操作探究")

    if features.get("problem_structure") == "直接计算" or has_formula_calculation_intent(text):
        reasons.append("需要公式代入或物理量计算，至少基础题")

    if features.get("reality_question") == "是" and not is_trivial_concept_question(features, data):
        reasons.append("涉及现实生活建模情境")

    choice_like = is_choice_like_question(visible_text)

    if choice_like and features.get("knowledge_count") != "1个":
        reasons.append("多概念选择题，不属于单一概念直答")

    if choice_like and len(visible_text) > 90:
        reasons.append("选择题存在选项辨析，不宜判为送分题")

    if count_fill_blanks(text) >= 3:
        reasons.append("多空填空题，不宜判为送分题")

    force_basic_keywords = [
        "二力平衡", "合力为零", "平衡力", "相互作用力",
        "电磁继电器", "螺线管", "电流方向", "磁场方向",
        "串联", "并联", "电压表", "电流表", "量程",
        "机翼", "升力", "流速", "压强",
        "飞轮", "冲程", "做功次数",
        "做功条件", "水平移动", "竖直举高",
        "同种电荷", "异种电荷", "电荷转移",
        "蒸发吸热", "潜热", "0℃", "冰水混合物",
    ]

    if contains_any(text, force_basic_keywords):
        reasons.append("含有基础推理、识图或易错辨析关键词，至少基础题")

    return reasons

def should_upgrade_basic_to_medium(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    """规则：基础题 -> 中等题 升档
    教师口径对齐版：3-5步 + 2-3个知识点/常规模型/实验图表处理，即可进入中等题。
    不再要求必须达到 6-8 步才升中等。
    """
    reasons = []

    stem = data.get("stem", "") or ""
    analysis = data.get("analysis", "") or ""
    text = stem + "\n" + analysis

    step = features.get("step_count")
    state = features.get("state_count")
    calc = features.get("calculation_complexity")
    cross = features.get("cross_module")
    problem = features.get("problem_structure")
    variable = features.get("variable_relation")
    experiment = features.get("experiment_requirement")
    graph = features.get("graph_table_requirement")
    formula = features.get("formula_count")
    knowledge = features.get("knowledge_count")
    reasoning = features.get("reasoning_chain")

    if step in ["6-8步", "9-12步", "12步以上"]:
        reasons.append(f'步骤数达"{step}"')

    # 教师标准：3-4步常规分析、2-3知识点综合即可算中等。
    if (
        step == "3-5步"
        and knowledge in ["2-3个", "4个及以上"]
        and (
            reasoning in ["多层因果推理", "逆向推理或临界分析"]
            or problem in ["实验探究", "图像表格分析", "电路综合", "力学综合", "跨模块综合"]
            or state in ["双状态", "多状态", "连续变化或临界状态"]
            or formula in ["2-3个", "4-6个", "7个以上"]
        )
    ):
        reasons.append("3-5步常规分析且涉及2-3个知识点或常规模型转化，达到中等题")

    if state in ["多状态", "连续变化或临界状态"]:
        reasons.append(f'物理状态为"{state}"')

    if state == "双状态" and (
        calc in ["多公式联立", "复杂方程或范围计算", "简单笔算"]
        or problem in ["电路综合", "力学综合", "跨模块综合"]
        or variable in ["图像函数关系", "多变量耦合关系"]
        or experiment in ["控制变量或故障分析", "方案设计或误差评价"]
    ):
        reasons.append("双状态且伴随真实物理建模")

    if calc in ["多公式联立", "复杂方程或范围计算"]:
        reasons.append(f'计算需要"{calc}"')

    if cross == "跨模块综合" and (
        step in ["3-5步", "6-8步", "9-12步", "12步以上"]
        or formula in ["2-3个", "4-6个", "7个以上"]
        or knowledge in ["2-3个", "4个及以上"]
        or variable in ["图像函数关系", "多变量耦合关系"]
    ):
        reasons.append("跨模块且存在实质综合")

    if graph in ["多组比较归纳", "图像反推或外推"]:
        reasons.append("需要图表信息处理")

    standard_experiment_medium_keywords = [
        "电磁铁磁性强弱", "液体压强", "压强计",
        "平均速度", "滑动摩擦力", "摩擦力大小",
        "杠杆平衡", "电流与电阻", "电流与电压",
        "重力与质量", "凸透镜成像", "熔化实验",
        "沸腾实验", "比热容", "焦耳定律"
    ]

    if (
        features.get("problem_structure") == "实验探究"
        and features.get("experiment_requirement") in ["控制变量或故障分析", "基础操作或读数"]
        and contains_any(text, standard_experiment_medium_keywords)
    ):
        reasons.append("标准实验探究题，涉及控制变量或实验归纳，至少中等题")

    if (
        features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]
        and features.get("information_carrier") in ["图像或表格", "多图表综合", "实验装置图"]
    ):
        reasons.append("需要图表信息处理，至少中等题")

    return reasons

def should_downgrade_medium_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """规则：中等题 -> 基础题 降档
    用于纠正单公式、短步骤、无实验图表反推的题被误判为中等。
    """
    stem = data.get("stem", "") or ""
    analysis = data.get("analysis", "") or ""
    text = stem + "\n" + analysis

    # 有这些结构时，不降
    if features.get("problem_structure") in ["实验探究", "电路综合", "力学综合", "跨模块综合"]:
        return False
    if features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]:
        return False
    if features.get("experiment_requirement") != "无":
        return False
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        return False
    if features.get("constraint_count") == "多约束":
        return False
    if features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]:
        return False

    simple_direct = (
        features.get("step_count") in ["1-2步", "3-5步"]
        and features.get("formula_count") in ["0-1个", "2-3个"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("reasoning_chain") in ["直接套用", "简单因果推理"]
    )

    simple_keywords = [
        "水平移动", "竖直举高", "做功", "动能", "机械能",
        "洒水车", "质量减小", "速度不变",
        "仪器使用", "估测", "生活常识"
    ]

    return simple_direct and contains_any(text, simple_keywords)

def should_upgrade_medium_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    """规则：中等题 -> 拔高题 升档
    教师口径对齐版：6-8步、多状态/多对象、隐含条件、2-3方程联立、跨章节迁移，已可判拔高题。
    """
    triggers = []
    force_hard = False

    stem = data.get("stem", "") or ""
    analysis = data.get("analysis", "") or ""
    text = stem + "\n" + analysis

    if features.get("step_count") in ["6-8步", "9-12步", "12步以上"]:
        triggers.append("步骤数达到较难题要求")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        triggers.append("公式链较长")
    if features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"]:
        triggers.append("需要方程联立或复杂计算")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        triggers.append("逆向推理或临界分析")
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        triggers.append("包含多状态或临界状态")
    if features.get("state_count") == "双状态" and features.get("problem_structure") in ["电路综合", "力学综合", "跨模块综合"]:
        triggers.append("双状态综合模型")
    if features.get("constraint_count") == "多约束":
        triggers.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        triggers.append("多变量耦合")
    if features.get("cross_module") == "跨模块综合":
        triggers.append("跨章节综合")
    if features.get("knowledge_count") == "4个及以上":
        triggers.append("知识点数量较多")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        triggers.append("方案设计或误差评价")
    if features.get("graph_table_requirement") == "图像反推或外推":
        triggers.append("图像反推或外推")

    hidden_or_math_keywords = [
        "隐含", "临界", "极值", "最大", "最小", "范围", "取值范围",
        "比例", "差值", "方程组", "几何关系", "分类讨论", "不等式"
    ]
    if contains_any(text, hidden_or_math_keywords):
        triggers.append("存在隐含条件、范围极值或方程组思想")

    multi_object_keywords = [
        "甲", "乙", "丙", "两个物体", "多个物体", "物体A", "物体B",
        "A、B", "A和B", "小车", "木块", "铁块", "容器", "滑轮组"
    ]
    if contains_any(text, multi_object_keywords) and features.get("problem_structure") in ["力学综合", "跨模块综合", "电路综合"]:
        triggers.append("多对象分析")

    if (
        features.get("problem_structure") == "实验探究"
        and features.get("experiment_requirement") == "控制变量或故障分析"
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("variable_relation") in ["无变量关系", "简单正反比"]
        and "方案设计或误差评价" not in triggers
        and "图像反推或外推" not in triggers
        and "存在隐含条件、范围极值或方程组思想" not in triggers
    ):
        return False, triggers

    if (
        len(stem) > 300
        and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"]
        and features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]
    ):
        triggers.append("长实验题且需要实验分析与图表处理")
        force_hard = True

    if (
        len(stem) > 300
        and ("阅读" in stem or "材料" in stem or "装置" in stem or "项目" in stem)
        and features.get("information_carrier") in ["图像或表格", "多图表综合", "实验装置图"]
        and features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]
    ):
        triggers.append("长阅读材料题且需要图表信息提取和变量关系分析")
        force_hard = True

    geometry_dynamic_keywords = [
        "反推", "旋转", "移动", "变化", "距离变化", "光斑",
        "反射方向", "成像区域", "物距", "像距", "几何关系"
    ]

    if (
        features.get("problem_structure") in ["光学声学综合", "图像表格分析", "跨模块综合"]
        and features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"]
        and any(k in stem for k in geometry_dynamic_keywords)
    ):
        triggers.append("图像/几何/光路动态反推")
        force_hard = True

    mechanical_composite_keywords = [
        "吊运", "支持力", "滑轮组", "机械效率",
        "绳端", "拉力", "功率", "有用功", "总功"
    ]

    hit_mechanical = [k for k in mechanical_composite_keywords if k in stem]

    if (
        features.get("problem_structure") in ["力学综合", "跨模块综合"]
        and len(hit_mechanical) >= 3
        and (
            features.get("formula_count") in ["2-3个", "4-6个", "7个以上"]
            or features.get("state_count") in ["双状态", "多状态", "连续变化或临界状态"]
            or features.get("constraint_count") in ["单一约束", "多约束"]
        )
    ):
        triggers.append("复杂机械/吊运/滑轮组综合")
        force_hard = True

    return force_hard or len(set(triggers)) >= 2, triggers

def should_downgrade_standard_experiment(features: Dict[str, Any], data: Dict[str, Any]) -> Optional[str]:
    """标准实验题降档规则。

    返回：
    - "基础题"：天平量筒、温度计、弹簧测力计、基础读数/直接公式型实验；
    - "中等题"：控制变量、表格归纳、常规实验结论型实验；
    - None：不降档。
    """
    stem = data.get("stem", "") or ""
    analysis = data.get("analysis", "") or ""
    text = stem + "\n" + analysis

    if features.get("problem_structure") != "实验探究":
        return None

    # 这些关键词出现时，通常不是标准基础/中等实验，不能轻易降档
    hard_experiment_keywords = [
        "缺少电压表", "缺少电流表", "只用电压表", "只用电流表",
        "等效替代", "表达式", "设计实验", "实验方案",
        "方案设计", "误差评价", "误差分析", "特殊方法",
        "多次开关", "滑动变阻器范围", "额定功率", "未知电阻",
        "黑箱", "传感器", "压敏", "热敏", "光敏"
    ]

    if any(k in text for k in hard_experiment_keywords):
        return None

    # 高阶结构保护：已经出现复杂模型时，不降档
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        return None
    if features.get("constraint_count") == "多约束":
        return None
    if features.get("variable_relation") == "多变量耦合关系":
        return None
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        return None
    if features.get("graph_table_requirement") == "图像反推或外推":
        return None

    # 1. 直接读数/基础测量型实验：可以降到基础题
    basic_experiment_keywords = [
        "天平", "量筒", "温度计", "弹簧测力计", "刻度尺",
        "秒表", "停表", "读数", "测密度", "测量密度"
    ]

    if (
        any(k in text for k in basic_experiment_keywords)
        and features.get("step_count") in ["1-2步", "3-5步", "6-8步"]
        and features.get("formula_count") in ["0-1个", "2-3个"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("experiment_requirement") in ["基础操作或读数", "控制变量或故障分析"]
    ):
        # 液体密度、多步骤测量、实验误差分析，通常按中等处理
        if (
            features.get("step_count") == "6-8步"
            or "葡萄酒" in text
            or "盐水" in text
            or "液体密度" in text
            or "测量液体密度" in text
            or "误差" in text
        ):
            return "中等题"

        return "基础题"

    # 2. 标准控制变量/表格归纳型实验：降到中等题
    standard_experiment_keywords = [
        "杠杆平衡", "液体压强", "浮力大小", "排开液体",
        "电阻大小", "电流与电阻", "电流与电压",
        "焦耳定律", "比热容", "熔化", "沸腾",
        "滑动摩擦力", "重力与质量", "凸透镜成像",
        "机械效率", "滑轮组机械效率", "测量机械效率",
        "平均速度", "测量平均速度",
        "电磁铁磁性强弱"
    ]

    if (
        any(k in text for k in standard_experiment_keywords)
        and features.get("step_count") in ["3-5步", "6-8步"]
        and features.get("formula_count") in ["0-1个", "2-3个"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("reasoning_chain") in ["简单因果推理", "多层因果推理"]
        and features.get("experiment_requirement") in ["基础操作或读数", "控制变量或故障分析"]
        and features.get("variable_relation") in ["无变量关系", "简单正反比", "图像函数关系"]
        and features.get("constraint_count") in ["无约束", "单一约束"]
    ):
        return "中等题"

    return None

def should_upgrade_continuous_buoyancy_to_final(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    """连续变化浮力/液面/弹簧/容器压强耦合题，满足核心条件时升压轴。"""
    stem = data.get("stem", "") or ""
    analysis = data.get("analysis", "") or ""
    text = stem + "\n" + analysis

    keywords = [
        "浮力", "漂浮", "浸没", "液面", "水面", "容器", "压强",
        "弹簧", "蜡烛", "燃烧", "铁块", "木块", "细线", "拉力"
    ]

    hit_keywords = [k for k in keywords if k in text]

    if len(hit_keywords) < 3:
        return False, []

    core_conditions = []

    if features.get("problem_structure") in ["力学综合", "跨模块综合"]:
        core_conditions.append("力学/跨模块综合")
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        core_conditions.append("多状态或连续临界状态")
    if features.get("constraint_count") == "多约束":
        core_conditions.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        core_conditions.append("多变量耦合")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        core_conditions.append("逆向推理或临界分析")
    if features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"]:
        core_conditions.append("多公式联立或复杂计算")
    if features.get("step_count") in ["9-12步", "12步以上"]:
        core_conditions.append("步骤数较多")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        core_conditions.append("公式链较长")

    # 至少 5 个核心条件，且必须包含连续/多状态 + 多变量耦合/多约束之一
    has_state = "多状态或连续临界状态" in core_conditions
    has_coupling = (
        "多变量耦合" in core_conditions
        or "多约束" in core_conditions
    )

    ok = len(core_conditions) >= 5 and has_state and has_coupling

    reasons = core_conditions + [f"命中关键词：{','.join(hit_keywords[:5])}"]
    return ok, reasons

def should_upgrade_hard_to_final(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    """规则：拔高题 -> 压轴题 升档
    教师口径对齐版：9步以上 + 多约束/极值范围/分类讨论/临界状态/陌生创新情境，可判压轴。
    不再机械要求必须 12步以上 或 7个以上公式。
    """
    ok_continuous, continuous_reasons = should_upgrade_continuous_buoyancy_to_final(features, data)
    if ok_continuous:
        return True, continuous_reasons

    stem = data.get("stem", "") or ""
    analysis = data.get("analysis", "") or ""
    text = stem + "\n" + analysis

    high_features = []

    if features.get("step_count") in ["9-12步", "12步以上"]:
        high_features.append("9步以上复杂推理")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        high_features.append("公式链较长")
    if features.get("knowledge_count") == "4个及以上":
        high_features.append("4个及以上知识点")
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        high_features.append("复杂方程或范围计算")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        high_features.append("逆向推理或临界分析")
    if features.get("state_count") == "连续变化或临界状态":
        high_features.append("连续变化或临界状态")
    elif features.get("state_count") == "多状态":
        high_features.append("多状态")
    if features.get("constraint_count") == "多约束":
        high_features.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        high_features.append("多变量耦合")
    if features.get("graph_table_requirement") == "图像反推或外推":
        high_features.append("图像反推或外推")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        high_features.append("方案设计或误差评价")
    if features.get("cross_module") == "跨模块综合":
        high_features.append("跨模块综合")

    final_keywords = [
        "极值", "最大", "最小", "范围", "取值范围", "不等式", "分类讨论",
        "多解", "筛选", "临界", "黑箱", "非线性", "I-U", "U-I",
        "压敏", "热敏", "光敏", "电磁继电器", "陌生", "创新", "自主设计"
    ]
    if contains_any(text, final_keywords):
        high_features.append("极值范围/分类讨论/陌生创新信号")

    new_context_keywords = ["阅读", "材料", "新定义", "自制", "传感器", "装置", "项目", "工程", "科技"]
    if contains_any(stem, new_context_keywords) and (
        features.get("experiment_requirement") == "方案设计或误差评价"
        or features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]
        or features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]
    ):
        high_features.append("陌生情境下的信息加工或创新设计")

    # 实验创新题：如果是陌生装置/创新方案 + 多变量或方程筛选，可以升压轴。
    if (
        features.get("experiment_requirement") == "方案设计或误差评价"
        and contains_any(text, ["创新", "自主设计", "设计实验", "实验方案", "表达式", "筛选", "未知", "传感器", "压敏", "热敏", "光敏"])
        and (
            features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]
            or features.get("calculation_complexity") == "复杂方程或范围计算"
            or features.get("constraint_count") == "多约束"
        )
    ):
        return True, high_features + ["创新实验设计达到困难题特征"]

    core_signals = {
        "9步以上复杂推理",
        "复杂方程或范围计算",
        "连续变化或临界状态",
        "多约束",
        "多变量耦合",
        "极值范围/分类讨论/陌生创新信号",
    }

    has_core_signal = any(x in high_features for x in core_signals)
    unique_count = len(set(high_features))

    should_upgrade = (
        unique_count >= 5
        or (unique_count >= 4 and has_core_signal)
    )

    return should_upgrade, high_features

def should_downgrade_final_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """规则：压轴题 -> 拔高题 降档保护
    教师口径对齐版：仍防止普通题误判压轴，但保留 9步以上 + 多约束/范围/极值/分类讨论 的压轴通道。
    """
    stem = data.get("stem", "") or ""
    analysis = data.get("analysis", "") or ""
    text = stem + "\n" + analysis

    final_keep_keywords = [
        "极值", "最大", "最小", "范围", "取值范围", "不等式", "分类讨论",
        "多解", "筛选", "临界", "黑箱", "非线性", "I-U", "U-I",
        "多开关", "满偏", "压敏", "热敏", "光敏", "电磁继电器",
        "陌生", "创新", "自主设计", "传感器"
    ]

    # 9-12步、4-6公式的普通电路安全范围题，若没有极值/分类讨论/多开关/非线性等压轴信号，仍降为拔高；
    # 若有这些信号，则保留压轴可能。
    if (
        features.get("problem_structure") == "电路综合"
        and features.get("step_count") == "9-12步"
        and features.get("formula_count") == "4-6个"
        and features.get("calculation_complexity") != "复杂方程或范围计算"
        and not contains_any(text, final_keep_keywords)
    ):
        return True

    if (
        "密度计" in text
        and not contains_any(text, ["弹簧", "杠杆", "液面变化", "连续变化", "多状态", "极值", "范围", "分类讨论"])
    ):
        return True

    high_features = 0
    if features.get("step_count") in ["9-12步", "12步以上"]:
        high_features += 1
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        high_features += 1
    if features.get("knowledge_count") == "4个及以上":
        high_features += 1
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        high_features += 1
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        high_features += 1
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        high_features += 1
    if features.get("constraint_count") == "多约束":
        high_features += 1
    if features.get("variable_relation") == "多变量耦合关系":
        high_features += 1
    if features.get("graph_table_requirement") == "图像反推或外推":
        high_features += 1
    if contains_any(text, final_keep_keywords):
        high_features += 1

    return high_features < 3

def postprocess_physics_difficulty(rating_result: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """物理后处理自动纠偏主流程"""
    if not rating_result:
        return rating_result

    # 1. 字段特征归一化
    rating_result["features"] = normalize_features(rating_result.get("features", {}))

    # 2. 标准化 reasoning 格式，防止字段污染
    normalize_reasoning_schema(rating_result)

    # 3. 基础题 -> 送分题 降档校验
    if rating_result.get("difficulty_level") == "基础题":
        if should_downgrade_basic_to_easy(rating_result["features"], data):
            set_level_with_reason(rating_result, "送分题", "自动降档：无计算、无实验的极短纯概念直答题")

    # 4. 送分题 -> 基础题 升档校验
    if rating_result.get("difficulty_level") == "送分题":
        upgrade_reasons = should_upgrade_easy_to_basic(rating_result["features"], data)
        if upgrade_reasons:
            set_level_with_reason(rating_result, "基础题", "自动升档：" + "；".join(upgrade_reasons))

    # 5. 基础题 -> 中等题 升档校验
    if rating_result.get("difficulty_level") == "基础题":
        upgrade_reasons = should_upgrade_basic_to_medium(rating_result["features"], data)
        if upgrade_reasons:
            set_level_with_reason(rating_result, "中等题", "自动升档：" + "；".join(upgrade_reasons))

    # 5.5 中等题 -> 基础题 降档校验
    if rating_result.get("difficulty_level") == "中等题":
        if should_downgrade_medium_to_basic(rating_result["features"], data):
            set_level_with_reason(
                rating_result,
                "基础题",
                "自动降档：单公式短步骤题，未达到中等复杂度"
            )

    # 6. 中等题 -> 拔高题 升档校验
    if rating_result.get("difficulty_level") == "中等题":
        ok, reasons = should_upgrade_medium_to_hard(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "拔高题", "自动升档：" + "；".join(reasons))

    # 6.5 拔高题 -> 基础/中等题 降档校验
    if rating_result.get("difficulty_level") == "拔高题":
        downgraded_level = should_downgrade_standard_experiment(rating_result["features"], data)
        if downgraded_level:
            set_level_with_reason(
                rating_result,
                downgraded_level,
                f"自动降档：标准实验题，未达到拔高复杂度，降为{downgraded_level}"
            )

    # 7. 拔高题 -> 压轴题 升档校验
    if rating_result.get("difficulty_level") == "拔高题":
        ok, reasons = should_upgrade_hard_to_final(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "压轴题", "自动升档：" + "；".join(reasons))

    # 8. 压轴题 -> 拔高题 降档校验
    if rating_result.get("difficulty_level") == "压轴题":
        if should_downgrade_final_to_hard(rating_result["features"], data):
            set_level_with_reason(rating_result, "拔高题", "自动降档：压轴高阶物理特征不足（少于3项）")

    # 9. 强行同步粗定档区间
    sync_coarse_difficulty(rating_result)

    return rating_result



# -------------------------- 3.9 V4 边界修正规则覆盖层 --------------------------
# 说明：以下函数会覆盖前面同名函数。目的是按 133 题对齐分析结果修边界，而不是整体升/降档。

def visible_text_of(data: Dict[str, Any]) -> str:
    """只取题干和选项，避免解析中的复杂公式/关键词误触发升档。"""
    return (data.get("stem", "") or "") + "\n" + (data.get("options", "") or "")


def full_text_of(data: Dict[str, Any]) -> str:
    return visible_text_of(data) + "\n" + (data.get("analysis", "") or "")


def has_formula_calculation_intent(text: str) -> bool:
    """是否明显需要代入公式或计算。
    不能只因为出现“压强/电流/热量”等物理量名就触发，必须同时有计算动作或数值/公式线索。
    """
    text = text or ""
    action_cues = ["求", "计算", "算出", "多少", "多大", "为多少", "等于多少", "取值范围", "最大", "最小"]
    quantity_cues = ["压强", "压力", "浮力", "密度", "速度", "路程", "时间", "电流", "电压", "电阻", "电功率", "功率", "电能", "热量", "机械效率", "有用功", "总功", "阻值", "质量", "体积"]
    formula_cues = ["=", "P=", "U=", "I=", "R=", "ρ=", "p=", "F=", "v=", "s=", "t=", "W=", "Q="]
    has_number = bool(re.search(r"\d", text))
    has_action = contains_any(text, action_cues)
    has_quantity = contains_any(text, quantity_cues)
    has_formula = contains_any(text, formula_cues)
    return (has_action and (has_quantity or has_number or has_formula)) or (has_formula and has_number)


def is_standard_low_level_diagram_question(data: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """极标准教材原型作图：如静止在水平面上的物体受力示意图，可按老师口径视作送分。"""
    stem = data.get("stem", "") or ""
    if not contains_any(stem, ["画出", "作出", "示意图"]):
        return False
    if contains_any(stem, ["电路", "滑轮", "杠杆", "力臂", "折射", "反射", "透镜", "连接完整"]):
        return False
    return (
        contains_any(stem, ["静止", "水平地面", "重力", "受力示意图"])
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
        and features.get("constraint_count") in ["无约束", "单一约束"]
    )


def is_parallel_choice_or_independent_points(data: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """并列选项/独立小问：知识点可能多，但不是“真正综合”。"""
    visible = visible_text_of(data)
    options = data.get("options", "") or ""
    subq = data.get("sub_questions", []) or []
    choice_parallel = (
        is_choice_like_question(visible)
        and len(options) > 20
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("state_count") in ["单状态", "双状态"]
        and features.get("constraint_count") in ["无约束", "单一约束"]
        and features.get("variable_relation") in ["无变量关系", "简单正反比"]
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") in ["无", "直接读数"]
    )
    independent_subq = (
        len(subq) >= 2
        and features.get("subquestion_dependency") == "多问但相互独立"
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("variable_relation") in ["无变量关系", "简单正反比"]
        and features.get("constraint_count") in ["无约束", "单一约束"]
        and features.get("graph_table_requirement") in ["无", "直接读数"]
    )
    return choice_parallel or independent_subq


def has_real_integration(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """是否存在真正连续推理/模型转化，而不是只把独立知识点数量相加。"""
    if is_parallel_choice_or_independent_points(data, features):
        return False
    return (
        features.get("state_count") in ["双状态", "多状态", "连续变化或临界状态"]
        or features.get("constraint_count") == "多约束"
        or features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]
        or features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"]
        or features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"]
        or features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"]
        or features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]
        or features.get("subquestion_dependency") == "多问且层层递进"
    )


def is_real_multi_object_modeling(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """真正多对象建模：需要对象间关系，而不是只出现“甲乙/木块/容器”等词。"""
    text = full_text_of(data)
    object_cues = ["甲", "乙", "丙", "两个物体", "多个物体", "小车", "木块", "铁块", "容器", "滑轮组", "物体A", "物体B"]
    relation_cues = ["分别", "受力", "相互作用", "连接", "拉力", "支持力", "摩擦力", "弹力", "浮力", "一起", "相对", "传递", "接触", "杠杆", "力臂", "绳端", "液面", "容器底", "列式", "方程"]
    return (
        contains_any(text, object_cues)
        and contains_any(text, relation_cues)
        and features.get("problem_structure") in ["力学综合", "跨模块综合", "电路综合", "实验探究"]
        and not is_parallel_choice_or_independent_points(data, features)
    )


def is_common_dynamic_circuit_not_final(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """常规动态电路/传感器电路：可以中等或拔高，但不要只靠量程、图像、最大值进压轴。"""
    text = full_text_of(data)
    if features.get("problem_structure") != "电路综合":
        return False
    final_cues = ["黑箱", "非线性", "I-U", "U-I", "多开关", "分类讨论", "不等式", "多解", "自主设计", "实验方案"]
    if contains_any(text, final_cues):
        return False
    return contains_any(text, ["量程", "滑动变阻器", "压敏", "热敏", "光敏", "最大", "范围", "图像", "图乙"])


def should_downgrade_basic_to_easy(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    if is_long_context_or_new_situation(data):
        return False
    visible = visible_text_of(data)
    text = full_text_of(data)
    problem = features.get("problem_structure")
    if is_standard_low_level_diagram_question(data, features):
        return True
    if problem == "直接计算" or has_formula_calculation_intent(visible):
        return False
    if contains_any(visible, ["如图", "根据图", "由图", "分析", "判断理由", "反应生成", "核反应", "守恒", "说明"]):
        return False
    return (
        problem in ["概念判断", "光学声学综合", "热学综合"]
        and features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_diff") == "低"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
        and count_fill_blanks(text) < 3
    )


def should_upgrade_easy_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons = []
    visible = visible_text_of(data)
    text = full_text_of(data)
    if is_standard_low_level_diagram_question(data, features):
        return reasons
    if features.get("step_count") != "1-2步":
        reasons.append(f'解题步骤数为"{features.get("step_count")}"')
    if features.get("knowledge_count") != "1个":
        reasons.append(f'知识点数量为"{features.get("knowledge_count")}"')
    if features.get("state_count") != "单状态":
        reasons.append(f'物理状态数量为"{features.get("state_count")}"')
    if features.get("constraint_count") != "无约束":
        reasons.append("存在物理约束条件")
    if features.get("information_carrier") in ["电路图", "实验装置图", "多图表综合", "图像或表格"]:
        if contains_any(visible, ["根据图", "由图", "如图", "读数", "图像", "表格", "电路"]):
            reasons.append(f'信息载体为"{features.get("information_carrier")}"，需要识图或读图')
    if features.get("experiment_requirement") != "无":
        reasons.append("含有实验操作探究")
    if features.get("problem_structure") == "直接计算" or has_formula_calculation_intent(visible):
        reasons.append("需要公式代入或物理量计算，至少基础题")
    if features.get("reality_question") == "是" and not is_trivial_concept_question(features, data):
        reasons.append("涉及现实生活建模情境")
    if is_choice_like_question(visible) and features.get("knowledge_count") != "1个" and not is_parallel_choice_or_independent_points(data, features):
        reasons.append("多概念选择题且存在综合辨析，不属于单一概念直答")
    if count_fill_blanks(text) >= 3 and not is_parallel_choice_or_independent_points(data, features):
        reasons.append("多空填空题且存在综合关系，不宜判为送分题")
    force_basic_keywords = ["电磁继电器", "螺线管", "电流方向", "磁场方向", "串联", "并联", "电压表", "电流表", "量程", "机翼", "升力", "流速", "做功条件", "电荷转移", "蒸发吸热", "潜热"]
    if contains_any(visible, force_basic_keywords):
        reasons.append("含有基础推理、识图或易错辨析关键词，至少基础题")
    return reasons


def should_upgrade_basic_to_medium(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons = []
    text = full_text_of(data)
    step = features.get("step_count")
    state = features.get("state_count")
    calc = features.get("calculation_complexity")
    cross = features.get("cross_module")
    problem = features.get("problem_structure")
    variable = features.get("variable_relation")
    experiment = features.get("experiment_requirement")
    graph = features.get("graph_table_requirement")
    formula = features.get("formula_count")
    knowledge = features.get("knowledge_count")
    reasoning = features.get("reasoning_chain")
    if is_parallel_choice_or_independent_points(data, features):
        return reasons
    if step in ["6-8步", "9-12步", "12步以上"] and has_real_integration(features, data):
        reasons.append(f'步骤数达"{step}"且存在真实综合')
    if step == "3-5步" and knowledge in ["2-3个", "4个及以上"] and has_real_integration(features, data) and (
        reasoning in ["多层因果推理", "逆向推理或临界分析"]
        or problem in ["实验探究", "图像表格分析", "电路综合", "力学综合", "跨模块综合"]
        or formula in ["2-3个", "4-6个", "7个以上"]
    ):
        reasons.append("3-5步连续推理且涉及常规模型转化，达到中等题")
    if state in ["多状态", "连续变化或临界状态"]:
        reasons.append(f'物理状态为"{state}"')
    if state == "双状态" and has_real_integration(features, data) and (
        calc in ["多公式联立", "复杂方程或范围计算", "简单笔算"]
        or problem in ["电路综合", "力学综合", "跨模块综合"]
        or variable in ["图像函数关系", "多变量耦合关系"]
        or experiment in ["控制变量或故障分析", "方案设计或误差评价"]
    ):
        reasons.append("双状态且伴随真实物理建模")
    if calc in ["多公式联立", "复杂方程或范围计算"]:
        reasons.append(f'计算需要"{calc}"')
    if cross == "跨模块综合" and has_real_integration(features, data):
        reasons.append("跨模块且存在实质综合")
    if graph in ["多组比较归纳", "图像反推或外推"]:
        reasons.append("需要图表信息处理")
    standard_experiment_medium_keywords = ["电磁铁磁性强弱", "液体压强", "压强计", "平均速度", "滑动摩擦力", "摩擦力大小", "杠杆平衡", "电流与电阻", "电流与电压", "重力与质量", "凸透镜成像", "熔化实验", "沸腾实验", "比热容", "焦耳定律"]
    if features.get("problem_structure") == "实验探究" and features.get("experiment_requirement") in ["控制变量或故障分析", "基础操作或读数"] and contains_any(text, standard_experiment_medium_keywords):
        reasons.append("标准实验探究题，涉及控制变量或实验归纳，至少中等题")
    return reasons


def should_upgrade_medium_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    stem = data.get("stem", "") or ""
    text = full_text_of(data)
    if is_parallel_choice_or_independent_points(data, features):
        return False, ["并列知识点/独立选项，不按真正综合升拔高"]
    core, support = [], []
    if features.get("step_count") in ["6-8步", "9-12步", "12步以上"]:
        support.append("步骤数达到较难题要求")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        support.append("公式链较长")
    if features.get("knowledge_count") == "4个及以上":
        support.append("知识点数量较多")
    if features.get("cross_module") == "跨模块综合":
        support.append("跨章节综合")
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        core.append("复杂方程或范围计算")
    elif features.get("calculation_complexity") == "多公式联立" and features.get("formula_count") in ["2-3个", "4-6个", "7个以上"]:
        core.append("需要方程联立或多公式联动")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        core.append("逆向推理或临界分析")
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        core.append("包含多状态或临界状态")
    if features.get("state_count") == "双状态" and features.get("problem_structure") in ["电路综合", "力学综合", "跨模块综合"] and has_real_integration(features, data):
        core.append("双状态综合模型")
    if features.get("constraint_count") == "多约束":
        core.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        core.append("多变量耦合")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        core.append("方案设计或误差评价")
    if features.get("graph_table_requirement") == "图像反推或外推":
        core.append("图像反推或外推")
    if contains_any(text, ["隐含", "临界", "极值", "取值范围", "比例", "差值", "方程组", "几何关系", "分类讨论", "不等式", "误差", "评价", "可行"]):
        core.append("存在隐含条件、范围极值、几何转化或评价要求")
    if is_real_multi_object_modeling(features, data):
        core.append("真正多对象建模")
    force_hard = False
    if len(stem) > 300 and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"] and features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"] and has_real_integration(features, data):
        core.append("长实验题且需要实验分析与图表处理")
        force_hard = True
    if len(stem) > 300 and contains_any(stem, ["阅读", "材料", "装置", "项目", "实践", "挑战"]) and features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]:
        core.append("长材料题且需要变量关系分析")
        force_hard = True
    if features.get("problem_structure") in ["光学声学综合", "图像表格分析", "跨模块综合"] and features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"] and contains_any(stem, ["反推", "旋转", "移动", "变化", "距离变化", "光斑", "反射方向", "成像区域", "物距", "像距", "几何关系"]):
        core.append("图像/几何/光路动态反推")
        force_hard = True
    triggers = core + support
    return force_hard or (len(set(core)) >= 1 and len(set(triggers)) >= 2), triggers


def should_upgrade_hard_to_final(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    ok_continuous, continuous_reasons = should_upgrade_continuous_buoyancy_to_final(features, data)
    if ok_continuous:
        return True, continuous_reasons
    stem = data.get("stem", "") or ""
    text = full_text_of(data)
    if is_common_dynamic_circuit_not_final(features, data):
        return False, ["常规动态电路/传感器电路，未达到压轴复杂度"]
    high, core = [], []
    if features.get("step_count") in ["9-12步", "12步以上"]:
        high.append("9步以上复杂推理")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        high.append("公式链较长")
    if features.get("knowledge_count") == "4个及以上":
        high.append("4个及以上知识点")
    if features.get("cross_module") == "跨模块综合":
        high.append("跨模块综合")
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        core.append("复杂方程或范围计算")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        core.append("逆向推理或临界分析")
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        core.append(features.get("state_count"))
    if features.get("constraint_count") == "多约束":
        core.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        core.append("多变量耦合")
    if features.get("graph_table_requirement") == "图像反推或外推":
        core.append("图像反推或外推")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        core.append("方案设计或误差评价")
    if contains_any(text, ["不等式", "分类讨论", "多解", "筛选", "黑箱", "非线性", "I-U", "U-I", "多开关", "满偏", "自主设计", "可行性", "挑战赛", "改装", "创新"]):
        core.append("分类讨论/创新设计/复杂筛选信号")
    if contains_any(stem, ["项目", "挑战", "改装", "自制", "设计", "装置", "实践"]) and features.get("experiment_requirement") == "方案设计或误差评价" and contains_any(text, ["可行", "是否", "判断", "范围", "最大", "最小", "标注", "改进", "评价", "挑战"]) and (features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"] or features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"] or features.get("constraint_count") == "多约束"):
        return True, high + core + ["创新实验/项目设计存在可行性或边界验证"]
    return len(set(core)) >= 3 and len(set(high + core)) >= 5, high + core


def should_downgrade_final_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    stem = data.get("stem", "") or ""
    text = full_text_of(data)
    innovation_keep = contains_any(stem, ["项目", "挑战", "改装", "自制", "设计", "装置", "实践"]) and features.get("experiment_requirement") == "方案设计或误差评价" and contains_any(text, ["可行", "是否", "判断", "范围", "最大", "最小", "标注", "改进", "评价", "挑战"])
    if innovation_keep:
        return False
    if is_common_dynamic_circuit_not_final(features, data):
        return True
    if "密度计" in text and not contains_any(text, ["弹簧", "杠杆", "液面变化", "连续变化", "多状态", "极值", "范围", "分类讨论", "可行"]):
        return True
    keep_keywords = ["不等式", "分类讨论", "多解", "筛选", "黑箱", "非线性", "I-U", "U-I", "多开关", "满偏", "自主设计", "可行性", "创新"]
    high_features = 0
    if features.get("step_count") in ["9-12步", "12步以上"]:
        high_features += 1
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        high_features += 1
    if features.get("knowledge_count") == "4个及以上":
        high_features += 1
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        high_features += 1
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        high_features += 1
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        high_features += 1
    if features.get("constraint_count") == "多约束":
        high_features += 1
    if features.get("variable_relation") == "多变量耦合关系":
        high_features += 1
    if features.get("graph_table_requirement") == "图像反推或外推":
        high_features += 1
    if contains_any(text, keep_keywords):
        high_features += 1
    return high_features < 3


def postprocess_physics_difficulty(rating_result: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """边界校准版：默认一次后处理最多自动调整一档，避免连续跨档。"""
    if not rating_result:
        return rating_result
    rating_result["features"] = normalize_features(rating_result.get("features", {}))
    normalize_reasoning_schema(rating_result)
    adjusted = False

    if rating_result.get("difficulty_level") == "基础题" and not adjusted:
        if should_downgrade_basic_to_easy(rating_result["features"], data):
            set_level_with_reason(rating_result, "送分题", "自动降档：教师口径下属于一眼识别/标准教材原型题")
            adjusted = True
    if rating_result.get("difficulty_level") == "送分题" and not adjusted:
        reasons = should_upgrade_easy_to_basic(rating_result["features"], data)
        if reasons:
            set_level_with_reason(rating_result, "基础题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "基础题" and not adjusted:
        reasons = should_upgrade_basic_to_medium(rating_result["features"], data)
        if reasons:
            set_level_with_reason(rating_result, "中等题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "中等题" and not adjusted:
        if should_downgrade_medium_to_basic(rating_result["features"], data):
            set_level_with_reason(rating_result, "基础题", "自动降档：单公式短步骤题，未达到中等复杂度")
            adjusted = True
    if rating_result.get("difficulty_level") == "中等题" and not adjusted:
        ok, reasons = should_upgrade_medium_to_hard(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "拔高题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "拔高题" and not adjusted:
        downgraded = should_downgrade_standard_experiment(rating_result["features"], data)
        if downgraded:
            set_level_with_reason(rating_result, downgraded, f"自动降档：标准实验题，未达到拔高复杂度，降为{downgraded}")
            adjusted = True
    if rating_result.get("difficulty_level") == "拔高题" and not adjusted:
        ok, reasons = should_upgrade_hard_to_final(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "压轴题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "压轴题" and not adjusted:
        if should_downgrade_final_to_hard(rating_result["features"], data):
            set_level_with_reason(rating_result, "拔高题", "自动降档：压轴高阶物理特征不足")
            adjusted = True
    sync_coarse_difficulty(rating_result)
    return rating_result


# -------------------------- 3.10 V5 中等保护与送分收紧覆盖层 --------------------------
# 说明：V5 在 V4 基础上继续小修边界：
# 1. 收紧“基础题 -> 送分题”，防止较易题被压成送分；
# 2. 给中等题增加保护，防止多知识点辨析/实验流程题被压成基础；
# 3. 给常规电学/常规实验题增加降档保护，防止中等题被抬成拔高；
# 4. 补强半陌生情境、隐含条件、过程顺序和几何转化的拔高识别。

def is_pure_recognition_easy_question(data: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """真正的送分题：单一概念/单位/物理学史/能源类别/现象直接识别。
    允许很短的“哪一个属于...”选择题，但不允许作图、图示理解、复杂选项辨析。
    """
    visible = visible_text_of(data)
    stem = data.get("stem", "") or ""
    options = data.get("options", "") or ""

    # 作图、图示理解、电磁/光学规范图，一律不降送分，至少基础。
    block_cues = [
        "画出", "作出", "作图", "示意图", "根据图", "由图", "如图", "图甲", "图乙",
        "平面镜", "成像", "光路", "反射光", "折射", "磁感线", "螺线管", "安培定则",
        "电路", "连接", "实验", "探究", "装置", "解释正确的是", "说法正确的是", "下列说法"
    ]
    if contains_any(visible, block_cues):
        return False

    if has_formula_calculation_intent(visible):
        return False

    # 长选项辨析不是送分。
    if is_choice_like_question(visible) and len(visible) > 100:
        return False

    easy_cues = [
        "单位", "物理学家", "首先", "命名", "属于", "哪种", "哪一个", "哪个", "特性", "方向是",
        "不可再生", "可再生", "扩散", "凝固", "熔化", "汽化", "液化", "升华", "凝华",
        "响度", "音调", "音色", "红绿蓝", "竖直向下"
    ]

    return (
        contains_any(visible, easy_cues)
        and features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_diff") == "低"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )


def is_mid_level_parallel_analysis(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """并列选项/独立小问中的中等题保护。
    并列知识点不应升拔高，但若包含多个易错概念辨析、真实运动过程、实验流程、图示空间判断，
    老师口径下常常是中等，而不是基础。
    """
    visible = visible_text_of(data)
    text = full_text_of(data)

    if not is_parallel_choice_or_independent_points(data, features):
        return False

    # 排除很短、很纯的单概念识记题。
    if is_pure_recognition_easy_question(data, features):
        return False

    medium_context_cues = [
        "每隔相等时间", "位置", "水平面", "斜坡", "滑行", "钢卷尺", "飞机", "运动状态",
        "惯性", "摩擦力", "重力", "受力", "机械能", "动能", "重力势能", "功率", "机械效率",
        "光的反射", "反射定律", "入射角", "反射角", "法线", "平面镜成像",
        "串联", "并联", "电压", "电流", "电阻", "电路", "电压表", "电流表",
        "实验", "探究", "装置", "表格", "数据", "记录", "结论", "地理方位", "磁铁", "大气压"
    ]

    multi_knowledge = features.get("knowledge_count") in ["2-3个", "4个及以上"]
    has_context = contains_any(text, medium_context_cues)
    has_structure = features.get("problem_structure") in [
        "实验探究", "图像表格分析", "电路综合", "力学综合", "光学声学综合", "跨模块综合"
    ]
    has_carrier = features.get("information_carrier") in ["单图识别", "电路图", "实验装置图", "图像或表格", "多图表综合"]
    not_too_easy = (
        features.get("reasoning_chain") in ["简单因果推理", "多层因果推理"]
        or features.get("error_risk") in ["明显易错点", "高易错点"]
        or len(visible) > 130
    )

    # 油条/能源等纯并列基础概念题不保护到中等。
    simple_parallel_cues = ["油条", "塑性形变", "热传递", "热量是", "不可再生能源", "可再生能源"]
    if contains_any(text, simple_parallel_cues) and not has_carrier and features.get("experiment_requirement") == "无":
        return False

    return multi_knowledge and (has_context or has_structure or has_carrier) and not_too_easy


def is_common_medium_circuit_or_experiment(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """常规中等电学/实验保护：有公式、有图表、有量程，不等于拔高。"""
    text = full_text_of(data)
    problem = features.get("problem_structure")

    hard_block_cues = [
        "黑箱", "非线性", "I-U", "U-I", "多开关", "分类讨论", "不等式", "多解", "自主设计",
        "缺表", "等效替代", "特殊方法", "表达式", "异常点", "偏离", "新猜想", "可行性", "挑战赛", "改装"
    ]
    if contains_any(text, hard_block_cues):
        return False

    # 常规动态电路/压敏图像/双挡电热器：模型清楚时按中等保护。
    circuit_medium = (
        problem in ["电路综合", "跨模块综合"]
        and contains_any(text, ["量程", "压敏", "热敏", "滑动变阻器", "电热水壶", "双挡", "低温挡", "高温挡", "图像", "图乙", "最大"])
        and features.get("calculation_complexity") in ["简单笔算", "多公式联立"]
        and features.get("reasoning_chain") in ["简单因果推理", "多层因果推理"]
        and features.get("state_count") in ["单状态", "双状态", "多状态"]
        and features.get("variable_relation") in ["无变量关系", "简单正反比", "图像函数关系"]
    )

    # 普通实验误差/数据处理：没有异常点反推、缺表法、新方案时，按中等保护。
    experiment_medium = (
        problem == "实验探究"
        and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"]
        and contains_any(text, ["测量", "密度", "小灯泡", "电功率", "表格", "数据", "误差", "实验步骤", "电压表", "电流表"])
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算", "多公式联立"]
        and features.get("state_count") in ["单状态", "双状态"]
        and features.get("constraint_count") in ["无约束", "单一约束"]
    )

    return circuit_medium or experiment_medium


def is_hidden_process_hard_case(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """较难题补强：半陌生情境 + 隐含条件/过程顺序/几何转化。"""
    text = full_text_of(data)
    stem = data.get("stem", "") or ""

    patterns = [
        # 动态杠杆/安全线路
        ["空调", "线路", "载流量"],
        ["教室", "加装", "空调"],
        ["杠杆", "力臂", "角度"],
        ["头颈", "杠杆"],
        # 大气压操作顺序
        ["汲水", "玻璃管", "最佳组合"],
        ["管口", "快速上提", "大气压"],
        # 虹/光路几何
        ["虹", "球形水珠", "法线"],
        ["色散", "反射", "折射", "α"],
        # 复杂实验评价/过程推理
        ["带出部分水", "仍然准确"],
        ["餐盘", "保持水平", "受力"],
        ["隐含", "顺序", "判断"],
    ]

    hit_pattern = any(all(k in text for k in p) for p in patterns)
    hard_reasoning = (
        features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"]
        or features.get("subquestion_dependency") == "多问且层层递进"
        or contains_any(text, ["最佳组合", "顺序", "几何", "力臂", "可行", "准确", "为什么", "原因"])
    )
    not_parallel = not is_parallel_choice_or_independent_points(data, features)
    return hit_pattern and hard_reasoning and not_parallel


def should_downgrade_basic_to_easy(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    if is_long_context_or_new_situation(data):
        return False
    return is_pure_recognition_easy_question(data, features)


def should_upgrade_easy_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons = []
    visible = visible_text_of(data)
    text = full_text_of(data)

    if is_pure_recognition_easy_question(data, features):
        return reasons

    # 作图、图示理解、四选项说法辨析，老师口径通常至少较易/基础。
    if contains_any(visible, ["画出", "作出", "作图", "示意图", "根据图", "由图", "如图", "图甲", "图乙"]):
        reasons.append("涉及作图或图示理解，至少基础题")
    if contains_any(visible, ["平面镜", "成像", "磁感线", "螺线管", "安培定则", "光路", "电路"]):
        reasons.append("涉及规范作图/电磁或电路规律应用，至少基础题")
    if is_choice_like_question(visible) and len(visible) > 90:
        reasons.append("选择题存在多选项辨析，至少基础题")
    if features.get("information_carrier") in ["电路图", "实验装置图", "多图表综合", "图像或表格"]:
        reasons.append(f'信息载体为"{features.get("information_carrier")}"，需要识图或读图')
    if features.get("experiment_requirement") != "无":
        reasons.append("含有实验操作探究")
    if features.get("problem_structure") == "直接计算" or has_formula_calculation_intent(visible):
        reasons.append("需要公式代入或物理量计算，至少基础题")
    if features.get("knowledge_count") != "1个" and not is_pure_recognition_easy_question(data, features):
        reasons.append(f'知识点数量为"{features.get("knowledge_count")}"')
    if features.get("state_count") != "单状态":
        reasons.append(f'物理状态数量为"{features.get("state_count")}"')
    if features.get("constraint_count") != "无约束":
        reasons.append("存在物理约束条件")
    if count_fill_blanks(text) >= 3 and not is_parallel_choice_or_independent_points(data, features):
        reasons.append("多空填空题且存在综合关系，不宜判为送分题")

    return reasons


def should_upgrade_basic_to_medium(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons = []
    text = full_text_of(data)
    step = features.get("step_count")
    state = features.get("state_count")
    calc = features.get("calculation_complexity")
    cross = features.get("cross_module")
    problem = features.get("problem_structure")
    variable = features.get("variable_relation")
    experiment = features.get("experiment_requirement")
    graph = features.get("graph_table_requirement")
    formula = features.get("formula_count")
    knowledge = features.get("knowledge_count")
    reasoning = features.get("reasoning_chain")

    # V5 新增：并列辨析题也可能是中等，但不继续升拔高。
    if is_mid_level_parallel_analysis(features, data):
        reasons.append("多知识点辨析/实验或真实过程分析达到中等题，但不按拔高处理")
        return reasons

    if is_parallel_choice_or_independent_points(data, features):
        return reasons

    if step in ["6-8步", "9-12步", "12步以上"] and has_real_integration(features, data):
        reasons.append(f'步骤数达"{step}"且存在真实综合')
    if step == "3-5步" and knowledge in ["2-3个", "4个及以上"] and has_real_integration(features, data) and (
        reasoning in ["多层因果推理", "逆向推理或临界分析"]
        or problem in ["实验探究", "图像表格分析", "电路综合", "力学综合", "光学声学综合", "跨模块综合"]
        or formula in ["2-3个", "4-6个", "7个以上"]
    ):
        reasons.append("3-5步连续推理且涉及常规模型转化，达到中等题")
    if state in ["多状态", "连续变化或临界状态"]:
        reasons.append(f'物理状态为"{state}"')
    if state == "双状态" and has_real_integration(features, data) and (
        calc in ["多公式联立", "复杂方程或范围计算", "简单笔算"]
        or problem in ["电路综合", "力学综合", "跨模块综合"]
        or variable in ["图像函数关系", "多变量耦合关系"]
        or experiment in ["控制变量或故障分析", "方案设计或误差评价"]
    ):
        reasons.append("双状态且伴随真实物理建模")
    if calc in ["多公式联立", "复杂方程或范围计算"]:
        reasons.append(f'计算需要"{calc}"')
    if cross == "跨模块综合" and has_real_integration(features, data):
        reasons.append("跨模块且存在实质综合")
    if graph in ["多组比较归纳", "图像反推或外推"]:
        reasons.append("需要图表信息处理")
    standard_experiment_medium_keywords = ["电磁铁磁性强弱", "液体压强", "压强计", "平均速度", "滑动摩擦力", "摩擦力大小", "杠杆平衡", "电流与电阻", "电流与电压", "重力与质量", "凸透镜成像", "熔化实验", "沸腾实验", "比热容", "焦耳定律", "光的反射定律", "串联电路电压"]
    if features.get("problem_structure") == "实验探究" and features.get("experiment_requirement") in ["控制变量或故障分析", "基础操作或读数"] and contains_any(text, standard_experiment_medium_keywords):
        reasons.append("标准实验探究题，涉及控制变量或实验归纳，至少中等题")
    return reasons


def should_downgrade_medium_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    # V5：中等保护，防止多知识点辨析/实验流程题被压成基础。
    if is_mid_level_parallel_analysis(features, data):
        return False
    if features.get("problem_structure") in ["实验探究", "电路综合", "力学综合", "跨模块综合"] and has_real_integration(features, data):
        return False
    if is_hidden_process_hard_case(features, data):
        return False

    text = full_text_of(data)
    if features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]:
        return False
    if features.get("experiment_requirement") != "无":
        return False
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        return False
    if features.get("constraint_count") == "多约束":
        return False
    if features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]:
        return False
    simple_direct = (
        features.get("step_count") in ["1-2步", "3-5步"]
        and features.get("formula_count") in ["0-1个", "2-3个"]
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"]
        and features.get("reasoning_chain") in ["直接套用", "简单因果推理"]
    )
    simple_keywords = ["水平移动", "竖直举高", "做功", "动能", "机械能", "洒水车", "质量减小", "速度不变", "仪器使用", "估测", "生活常识"]
    return simple_direct and contains_any(text, simple_keywords)


def should_upgrade_medium_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    stem = data.get("stem", "") or ""
    text = full_text_of(data)

    if is_parallel_choice_or_independent_points(data, features) and not is_hidden_process_hard_case(features, data):
        return False, ["并列知识点/独立选项，不按真正综合升拔高"]

    # V5：常规中等电学/实验题不升拔高。
    if is_common_medium_circuit_or_experiment(features, data):
        return False, ["常规电学/实验题，模型清楚，按中等保护"]

    core, support = [], []
    if features.get("step_count") in ["6-8步", "9-12步", "12步以上"]:
        support.append("步骤数达到较难题要求")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        support.append("公式链较长")
    if features.get("knowledge_count") == "4个及以上":
        support.append("知识点数量较多")
    if features.get("cross_module") == "跨模块综合":
        support.append("跨章节综合")
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        core.append("复杂方程或范围计算")
    elif features.get("calculation_complexity") == "多公式联立" and features.get("formula_count") in ["2-3个", "4-6个", "7个以上"]:
        core.append("需要方程联立或多公式联动")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        core.append("逆向推理或临界分析")
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        core.append("包含多状态或临界状态")
    if features.get("state_count") == "双状态" and features.get("problem_structure") in ["电路综合", "力学综合", "跨模块综合"] and has_real_integration(features, data):
        core.append("双状态综合模型")
    if features.get("constraint_count") == "多约束":
        core.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        core.append("多变量耦合")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        core.append("方案设计或误差评价")
    if features.get("graph_table_requirement") == "图像反推或外推":
        core.append("图像反推或外推")
    if contains_any(text, ["隐含", "临界", "极值", "取值范围", "比例", "差值", "方程组", "几何关系", "分类讨论", "不等式", "误差", "评价", "可行"]):
        core.append("存在隐含条件、范围极值、几何转化或评价要求")
    if is_real_multi_object_modeling(features, data):
        core.append("真正多对象建模")

    force_hard = False
    if is_hidden_process_hard_case(features, data):
        core.append("半陌生情境中的隐含条件/过程顺序/几何转化")
        force_hard = True
    if len(stem) > 300 and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"] and features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"] and has_real_integration(features, data):
        core.append("长实验题且需要实验分析与图表处理")
        force_hard = True
    if len(stem) > 300 and contains_any(stem, ["阅读", "材料", "装置", "项目", "实践", "挑战"]) and features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]:
        core.append("长材料题且需要变量关系分析")
        force_hard = True
    if features.get("problem_structure") in ["光学声学综合", "图像表格分析", "跨模块综合"] and features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"] and contains_any(stem, ["反推", "旋转", "移动", "变化", "距离变化", "光斑", "反射方向", "成像区域", "物距", "像距", "几何关系"]):
        core.append("图像/几何/光路动态反推")
        force_hard = True
    triggers = core + support
    return force_hard or (len(set(core)) >= 1 and len(set(triggers)) >= 2), triggers


def should_downgrade_hard_to_medium(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V5 新增：纠正常规中等题被抬成拔高。"""
    if is_hidden_process_hard_case(features, data):
        return False
    if is_real_multi_object_modeling(features, data) and features.get("reasoning_chain") == "逆向推理或临界分析":
        return False
    return is_common_medium_circuit_or_experiment(features, data)


def postprocess_physics_difficulty(rating_result: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """V5 边界校准版：默认一次后处理最多自动调整一档。"""
    if not rating_result:
        return rating_result
    rating_result["features"] = normalize_features(rating_result.get("features", {}))
    normalize_reasoning_schema(rating_result)
    adjusted = False

    if rating_result.get("difficulty_level") == "基础题" and not adjusted:
        if should_downgrade_basic_to_easy(rating_result["features"], data):
            set_level_with_reason(rating_result, "送分题", "自动降档：教师口径下属于单一概念直接识别题")
            adjusted = True
    if rating_result.get("difficulty_level") == "送分题" and not adjusted:
        reasons = should_upgrade_easy_to_basic(rating_result["features"], data)
        if reasons:
            set_level_with_reason(rating_result, "基础题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "基础题" and not adjusted:
        reasons = should_upgrade_basic_to_medium(rating_result["features"], data)
        if reasons:
            set_level_with_reason(rating_result, "中等题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "中等题" and not adjusted:
        if should_downgrade_medium_to_basic(rating_result["features"], data):
            set_level_with_reason(rating_result, "基础题", "自动降档：单公式短步骤题，未达到中等复杂度")
            adjusted = True
    if rating_result.get("difficulty_level") == "中等题" and not adjusted:
        ok, reasons = should_upgrade_medium_to_hard(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "拔高题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "拔高题" and not adjusted:
        if should_downgrade_hard_to_medium(rating_result["features"], data):
            set_level_with_reason(rating_result, "中等题", "自动降档：常规电学/实验题，模型清楚，未达到拔高复杂度")
            adjusted = True
    if rating_result.get("difficulty_level") == "拔高题" and not adjusted:
        downgraded = should_downgrade_standard_experiment(rating_result["features"], data)
        if downgraded:
            set_level_with_reason(rating_result, downgraded, f"自动降档：标准实验题，未达到拔高复杂度，降为{downgraded}")
            adjusted = True
    if rating_result.get("difficulty_level") == "拔高题" and not adjusted:
        ok, reasons = should_upgrade_hard_to_final(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "压轴题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "压轴题" and not adjusted:
        if should_downgrade_final_to_hard(rating_result["features"], data):
            set_level_with_reason(rating_result, "拔高题", "自动降档：压轴高阶物理特征不足")
            adjusted = True
    sync_coarse_difficulty(rating_result)
    return rating_result



# -------------------------- 3.11 V6 回收中等保护与恢复压轴保护覆盖层 --------------------------
# 说明：V6 针对 V5 的反弹问题继续小修：
# 1. 回收“并列辨析 -> 中等”的触发范围，避免较易题大量被抬到中等；
# 2. 适度放宽送分题：极简单教材原型图示/漫画/生活背景中的单概念识别可判送分；
# 3. 常规中等保护加入高阶黑名单，防止项目式/自动控制/复杂电路被误降；
# 4. 恢复项目式创新实验、自动控制、电磁继电器、可行性验证等压轴保护。


def has_high_level_project_or_control_signal(data: Dict[str, Any]) -> bool:
    """项目式/自动控制/复杂方案信号。命中后不得走常规中等保护。"""
    text = full_text_of(data)
    high_cues = [
        "项目", "项目式", "任务", "挑战", "挑战赛", "实践", "改装", "重新设计", "设计方案",
        "方案可行", "可行性", "是否能", "能否覆盖", "边界", "标注", "密度尺",
        "自动控制", "自动控温", "控温", "电磁继电器", "继电器", "控制电路",
        "NTC", "PTC", "热敏电阻", "压敏电阻", "光敏电阻", "传感器",
        "发热体", "电热丝", "安全电流", "安全电压", "额定电流", "温度范围",
        "多阶段", "多状态", "临界", "不等式", "分类讨论", "多解", "筛选", "黑箱", "非线性", "I-U", "U-I"
    ]
    return contains_any(text, high_cues)


def is_simple_picture_or_context_easy_question(data: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """V6 放宽送分：极简单教材原型图示/漫画/生活背景中的单概念识别。"""
    visible = visible_text_of(data)

    # 明确不是送分的内容：规范作图、实验、电路/电磁应用、公式计算、复杂图像表格。
    hard_block = [
        "画出", "作出", "作图", "连接", "改线", "实验", "探究", "装置", "表格", "数据",
        "电路", "电压表", "电流表", "滑动变阻器", "故障", "短路", "断路",
        "螺线管", "磁感线", "安培定则", "平面镜成像作图", "光路图", "反射光", "折射光",
        "求", "计算", "多少", "多大", "取值范围", "最大", "最小"
    ]
    if contains_any(visible, hard_block) or has_formula_calculation_intent(visible):
        return False

    # 多知识点长选项辨析仍至少基础，不放到送分。
    if is_choice_like_question(visible) and len(visible) > 140 and features.get("knowledge_count") != "1个":
        return False

    # 允许非常典型的一步直答，包括简单图片/漫画背景。
    easy_cues = [
        "主要描述", "描述的是", "主要利用", "主要是", "主要为了", "相当于", "经历的物态变化",
        "属于", "最小的是", "联系是通过", "传播是通过", "空间尺度", "单位", "命名",
        "响度", "音调", "音色", "声速", "熔化", "凝固", "汽化", "液化", "升华", "凝华",
        "惯性", "无线电波", "电磁波", "凸透镜", "硬度", "不可再生", "可再生", "扩散", "竖直向下"
    ]

    return (
        contains_any(visible, easy_cues)
        and features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") in ["直接套用", "简单因果推理"]
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_diff") == "低"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )


def is_pure_recognition_easy_question(data: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """V6：送分题既包括纯文字一眼识别，也包括极简单图示/生活背景单概念识别。"""
    visible = visible_text_of(data)

    # 先允许 V6 新增的简单图示/生活背景送分通道。
    if is_simple_picture_or_context_easy_question(data, features):
        return True

    # 其余仍沿用严格送分口径。
    if contains_any(visible, [
        "画出", "作出", "作图", "示意图", "根据图", "由图", "如图", "图甲", "图乙",
        "平面镜", "成像", "光路", "反射光", "折射", "磁感线", "螺线管", "安培定则",
        "电路", "连接", "实验", "探究", "装置", "说明", "分析"
    ]):
        return False

    if has_formula_calculation_intent(visible):
        return False

    if is_choice_like_question(visible) and len(visible) > 110:
        return False

    easy_cues = [
        "单位", "物理学家", "首先", "命名", "属于", "哪种", "哪一个", "哪个", "特性", "方向是",
        "不可再生", "可再生", "扩散", "凝固", "熔化", "汽化", "液化", "升华", "凝华",
        "响度", "音调", "音色", "红绿蓝", "竖直向下"
    ]

    return (
        contains_any(visible, easy_cues)
        and features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_diff") == "低"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )


def is_mid_level_parallel_analysis(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V6 回收中等保护：并列题只有强信号才保护到中等。
    避免“选项长 + 多知识点”自动把较易题抬到中等。
    """
    visible = visible_text_of(data)
    text = full_text_of(data)

    if not is_parallel_choice_or_independent_points(data, features):
        return False
    if is_pure_recognition_easy_question(data, features):
        return False

    # 直接计算小问互相独立，不因小问数量多升中等。
    if contains_any(visible, ["求", "计算"]) and features.get("calculation_complexity") in ["简单笔算", "口算或直接判断"]:
        if features.get("subquestion_dependency") in ["多问但相互独立", "无多问"]:
            return False

    # 明确基础并列概念题黑名单：这些最高通常基础，不走中等保护。
    simple_parallel_cues = [
        "油条", "塑性形变", "热传递", "热量是", "不可再生能源", "可再生能源",
        "大气压", "吸管喝水", "拔火罐", "水瓶琴", "声音的传播", "响度", "音调", "音色",
        "验电器", "同种电荷", "异种电荷", "电动机", "发电机", "电热毯", "电水壶",
        "功和功率", "起重机", "工人搬运", "水果电池", "发光二极管"
    ]
    if contains_any(text, simple_parallel_cues) and features.get("experiment_requirement") == "无":
        return False

    # 强实验流程/表格归纳：可保护中等。
    strong_experiment = (
        features.get("problem_structure") == "实验探究"
        and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"]
        and (
            features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]
            or contains_any(text, ["实验步骤", "记录数据", "表格", "多次实验", "控制变量", "实验结论", "反射定律", "串联电路电压"])
        )
    )

    # 真实连续运动/受力过程，且有明显易错辨析：可保护中等。
    mechanics_terms = ["受力", "惯性", "力可以改变", "运动状态", "摩擦力", "重力", "机械能", "动能", "重力势能"]
    strong_real_process = (
        contains_any(text, ["钢卷尺", "纸飞机", "飞离", "滑行", "每隔相等时间", "运动过程", "斜坡", "水平面"])
        and sum(1 for k in mechanics_terms if k in text) >= 3
        and features.get("reasoning_chain") in ["简单因果推理", "多层因果推理"]
    )

    # 图像/空间方位/光路实验判断：可保护中等，但必须不是普通单一概念图。
    strong_spatial_or_graph = (
        features.get("information_carrier") in ["实验装置图", "图像或表格", "多图表综合"]
        and contains_any(text, ["方位", "位置", "入射角", "反射角", "法线", "实验序号", "表格", "图像", "数据"])
        and features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"]
    )

    # 高易错多知识点辨析：允许中等，但要求选项较长且非普通概念匹配。
    strong_error_parallel = (
        features.get("knowledge_count") == "4个及以上"
        and features.get("error_risk") in ["明显易错点", "高易错点"]
        and features.get("reasoning_chain") == "多层因果推理"
        and len(visible) > 180
        and not contains_any(text, simple_parallel_cues)
    )

    return strong_experiment or strong_real_process or strong_spatial_or_graph or strong_error_parallel


def is_common_medium_circuit_or_experiment(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V6：常规中等保护只保护真正模型清楚的题；项目式/自动控制/复杂方案一律排除。"""
    text = full_text_of(data)
    problem = features.get("problem_structure")

    if has_high_level_project_or_control_signal(data):
        return False

    hard_block_cues = [
        "黑箱", "非线性", "I-U", "U-I", "多开关", "分类讨论", "不等式", "多解", "自主设计",
        "缺表", "等效替代", "特殊方法", "表达式", "异常点", "偏离", "新猜想", "挑战赛", "改装",
        "临界", "极值", "筛选", "可行", "方案", "项目", "自动控制", "继电器", "安全电流"
    ]
    if contains_any(text, hard_block_cues):
        return False

    circuit_medium = (
        problem in ["电路综合", "跨模块综合"]
        and contains_any(text, ["量程", "压敏", "热敏", "滑动变阻器", "电热水壶", "双挡", "低温挡", "高温挡", "图像", "图乙", "最大"])
        and features.get("calculation_complexity") in ["简单笔算", "多公式联立"]
        and features.get("reasoning_chain") in ["简单因果推理", "多层因果推理"]
        and features.get("state_count") in ["单状态", "双状态", "多状态"]
        and features.get("variable_relation") in ["无变量关系", "简单正反比", "图像函数关系"]
    )

    experiment_medium = (
        problem == "实验探究"
        and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"]
        and contains_any(text, ["测量", "密度", "小灯泡", "电功率", "表格", "数据", "误差", "实验步骤", "电压表", "电流表"])
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算", "多公式联立"]
        and features.get("state_count") in ["单状态", "双状态"]
        and features.get("constraint_count") in ["无约束", "单一约束"]
    )

    return circuit_medium or experiment_medium


def is_hidden_process_hard_case(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V6：较难题补强，但必须命中较明确的半陌生/隐含/几何/过程顺序模式。"""
    text = full_text_of(data)

    patterns = [
        ["空调", "线路", "载流量"],
        ["教室", "加装", "空调"],
        ["杠杆", "力臂", "角度"],
        ["头颈", "杠杆"],
        ["汲水", "玻璃管", "最佳组合"],
        ["管口", "快速上提", "大气压"],
        ["虹", "球形水珠", "法线"],
        ["色散", "反射", "折射", "α"],
        ["带出部分水", "仍然准确"],
        ["餐盘", "保持水平", "受力"],
        ["天坛", "回音壁", "第三个回声"],
        ["电磁继电器", "热敏电阻", "自动控温"],
        ["发热体", "安全电流", "温度范围"],
    ]

    hit_pattern = any(all(k in text for k in p) for p in patterns)
    hard_reasoning = (
        features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"]
        or features.get("subquestion_dependency") == "多问且层层递进"
        or contains_any(text, ["最佳组合", "顺序", "几何", "力臂", "可行", "准确", "为什么", "原因", "筛选", "安全"])
    )
    return hit_pattern and hard_reasoning and not is_parallel_choice_or_independent_points(data, features)


def is_project_or_innovation_final_case(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """项目式创新/自动控制/可行性验证压轴保护。"""
    text = full_text_of(data)
    stem = data.get("stem", "") or ""

    project_signal = contains_any(stem + text, [
        "项目", "挑战赛", "改装", "重新设计", "自制", "设计", "任务", "实践",
        "自动控温", "自动控制", "电磁继电器", "热敏电阻", "NTC", "PTC", "发热体", "安全电流"
    ])
    validation_signal = contains_any(text, ["可行", "是否", "能否", "判断", "范围", "最大", "最小", "标注", "筛选", "安全", "边界", "温度范围"])
    high_model_signal = (
        features.get("experiment_requirement") == "方案设计或误差评价"
        or features.get("constraint_count") == "多约束"
        or features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]
        or features.get("calculation_complexity") in ["多公式联立", "复杂方程或范围计算"]
        or features.get("reasoning_chain") == "逆向推理或临界分析"
        or features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]
    )
    return project_signal and validation_signal and high_model_signal


def should_downgrade_basic_to_easy(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    if is_long_context_or_new_situation(data):
        # 长情境原则上不送分，但若实际就是极短单概念题，前面函数会处理；这里保守不降。
        return False
    return is_pure_recognition_easy_question(data, features) or is_simple_picture_or_context_easy_question(data, features)


def should_upgrade_easy_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons = []
    visible = visible_text_of(data)
    text = full_text_of(data)

    if is_pure_recognition_easy_question(data, features) or is_simple_picture_or_context_easy_question(data, features):
        return reasons

    if contains_any(visible, ["画出", "作出", "作图", "示意图"]):
        reasons.append("涉及规范作图，至少基础题")
    if contains_any(visible, ["平面镜", "成像", "磁感线", "螺线管", "安培定则", "光路", "电路", "连接"]):
        reasons.append("涉及规范作图/电磁或电路规律应用，至少基础题")
    if is_choice_like_question(visible) and len(visible) > 150 and features.get("knowledge_count") != "1个":
        reasons.append("选择题存在多选项辨析，至少基础题")
    if features.get("information_carrier") in ["电路图", "实验装置图", "多图表综合", "图像或表格"] and contains_any(visible, ["根据图", "由图", "读数", "表格", "电路", "实验"]):
        reasons.append(f'信息载体为"{features.get("information_carrier")}"，需要识图或读图')
    if features.get("experiment_requirement") != "无":
        reasons.append("含有实验操作探究")
    if features.get("problem_structure") == "直接计算" or has_formula_calculation_intent(visible):
        reasons.append("需要公式代入或物理量计算，至少基础题")
    if features.get("knowledge_count") != "1个" and not is_parallel_choice_or_independent_points(data, features):
        reasons.append(f'知识点数量为"{features.get("knowledge_count")}"')
    if features.get("state_count") != "单状态":
        reasons.append(f'物理状态数量为"{features.get("state_count")}"')
    if features.get("constraint_count") != "无约束":
        reasons.append("存在物理约束条件")
    if count_fill_blanks(text) >= 3 and not is_parallel_choice_or_independent_points(data, features):
        reasons.append("多空填空题且存在综合关系，不宜判为送分题")

    return reasons


def should_upgrade_basic_to_medium(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons = []
    text = full_text_of(data)
    step = features.get("step_count")
    state = features.get("state_count")
    calc = features.get("calculation_complexity")
    cross = features.get("cross_module")
    problem = features.get("problem_structure")
    variable = features.get("variable_relation")
    experiment = features.get("experiment_requirement")
    graph = features.get("graph_table_requirement")
    formula = features.get("formula_count")
    knowledge = features.get("knowledge_count")
    reasoning = features.get("reasoning_chain")

    if is_mid_level_parallel_analysis(features, data):
        reasons.append("强易错/实验流程/真实连续过程辨析达到中等题，但不按拔高处理")
        return reasons

    if is_parallel_choice_or_independent_points(data, features):
        return reasons

    if step in ["6-8步", "9-12步", "12步以上"] and has_real_integration(features, data):
        reasons.append(f'步骤数达"{step}"且存在真实综合')
    if step == "3-5步" and knowledge in ["2-3个", "4个及以上"] and has_real_integration(features, data) and (
        reasoning in ["多层因果推理", "逆向推理或临界分析"]
        or problem in ["实验探究", "图像表格分析", "电路综合", "力学综合", "光学声学综合", "跨模块综合"]
        or formula in ["2-3个", "4-6个", "7个以上"]
    ):
        reasons.append("3-5步连续推理且涉及常规模型转化，达到中等题")
    if state in ["多状态", "连续变化或临界状态"]:
        reasons.append(f'物理状态为"{state}"')
    if state == "双状态" and has_real_integration(features, data) and (
        calc in ["多公式联立", "复杂方程或范围计算", "简单笔算"]
        or problem in ["电路综合", "力学综合", "跨模块综合"]
        or variable in ["图像函数关系", "多变量耦合关系"]
        or experiment in ["控制变量或故障分析", "方案设计或误差评价"]
    ):
        reasons.append("双状态且伴随真实物理建模")
    if calc in ["多公式联立", "复杂方程或范围计算"]:
        reasons.append(f'计算需要"{calc}"')
    if cross == "跨模块综合" and has_real_integration(features, data):
        reasons.append("跨模块且存在实质综合")
    if graph in ["多组比较归纳", "图像反推或外推"]:
        reasons.append("需要图表信息处理")

    standard_experiment_medium_keywords = [
        "电磁铁磁性强弱", "液体压强", "压强计", "平均速度", "滑动摩擦力", "摩擦力大小",
        "杠杆平衡", "电流与电阻", "电流与电压", "重力与质量", "凸透镜成像", "熔化实验",
        "沸腾实验", "比热容", "焦耳定律", "光的反射定律", "串联电路电压"
    ]
    if features.get("problem_structure") == "实验探究" and features.get("experiment_requirement") in ["控制变量或故障分析", "基础操作或读数"] and contains_any(text, standard_experiment_medium_keywords):
        reasons.append("标准实验探究题，涉及控制变量或实验归纳，至少中等题")
    return reasons


def should_upgrade_medium_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    stem = data.get("stem", "") or ""
    text = full_text_of(data)

    if is_project_or_innovation_final_case(features, data):
        return True, ["项目式/创新设计题，存在可行性、边界或安全约束分析"]

    if is_parallel_choice_or_independent_points(data, features) and not is_hidden_process_hard_case(features, data):
        return False, ["并列知识点/独立选项，不按真正综合升拔高"]

    if is_common_medium_circuit_or_experiment(features, data):
        return False, ["常规电学/实验题，模型清楚，按中等保护"]

    core, support = [], []
    if features.get("step_count") in ["6-8步", "9-12步", "12步以上"]:
        support.append("步骤数达到较难题要求")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        support.append("公式链较长")
    if features.get("knowledge_count") == "4个及以上":
        support.append("知识点数量较多")
    if features.get("cross_module") == "跨模块综合":
        support.append("跨章节综合")
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        core.append("复杂方程或范围计算")
    elif features.get("calculation_complexity") == "多公式联立" and features.get("formula_count") in ["2-3个", "4-6个", "7个以上"]:
        core.append("需要方程联立或多公式联动")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        core.append("逆向推理或临界分析")
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        core.append("包含多状态或临界状态")
    if features.get("state_count") == "双状态" and features.get("problem_structure") in ["电路综合", "力学综合", "跨模块综合"] and has_real_integration(features, data):
        core.append("双状态综合模型")
    if features.get("constraint_count") == "多约束":
        core.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        core.append("多变量耦合")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        core.append("方案设计或误差评价")
    if features.get("graph_table_requirement") == "图像反推或外推":
        core.append("图像反推或外推")
    if contains_any(text, ["隐含", "临界", "极值", "取值范围", "比例", "差值", "方程组", "几何关系", "分类讨论", "不等式", "误差", "评价", "可行", "安全筛选"]):
        core.append("存在隐含条件、范围极值、几何转化或评价要求")
    if is_real_multi_object_modeling(features, data):
        core.append("真正多对象建模")

    force_hard = False
    if is_hidden_process_hard_case(features, data):
        core.append("半陌生情境中的隐含条件/过程顺序/几何转化")
        force_hard = True
    if len(stem) > 300 and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"] and features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"] and has_real_integration(features, data):
        core.append("长实验题且需要实验分析与图表处理")
        force_hard = True
    if len(stem) > 300 and contains_any(stem, ["阅读", "材料", "装置", "项目", "实践", "挑战"]) and features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]:
        core.append("长材料题且需要变量关系分析")
        force_hard = True
    if features.get("problem_structure") in ["光学声学综合", "图像表格分析", "跨模块综合"] and features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"] and contains_any(stem, ["反推", "旋转", "移动", "变化", "距离变化", "光斑", "反射方向", "成像区域", "物距", "像距", "几何关系"]):
        core.append("图像/几何/光路动态反推")
        force_hard = True

    triggers = core + support
    return force_hard or (len(set(core)) >= 1 and len(set(triggers)) >= 2), triggers


def should_downgrade_hard_to_medium(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V6：拔高降中等必须非常谨慎，避免误伤较难/困难。"""
    if is_project_or_innovation_final_case(features, data):
        return False
    if has_high_level_project_or_control_signal(data):
        return False
    if is_hidden_process_hard_case(features, data):
        return False
    if is_real_multi_object_modeling(features, data) and features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"]:
        return False
    # 高阶特征出现两个及以上，不降。
    high_count = 0
    for cond in [
        features.get("constraint_count") == "多约束",
        features.get("state_count") in ["多状态", "连续变化或临界状态"],
        features.get("variable_relation") == "多变量耦合关系",
        features.get("calculation_complexity") == "复杂方程或范围计算",
        features.get("reasoning_chain") == "逆向推理或临界分析",
        features.get("graph_table_requirement") == "图像反推或外推",
    ]:
        if cond:
            high_count += 1
    if high_count >= 2:
        return False
    return is_common_medium_circuit_or_experiment(features, data)


def should_upgrade_hard_to_final(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    if is_project_or_innovation_final_case(features, data):
        return True, ["项目式创新/自动控制/方案可行性验证达到压轴特征"]
    # 其余沿用 V5 逻辑核心。
    ok_continuous, continuous_reasons = should_upgrade_continuous_buoyancy_to_final(features, data)
    if ok_continuous:
        return True, continuous_reasons
    stem = data.get("stem", "") or ""
    text = full_text_of(data)
    if is_common_dynamic_circuit_not_final(features, data) and not has_high_level_project_or_control_signal(data):
        return False, ["常规动态电路/传感器电路，未达到压轴复杂度"]
    high, core = [], []
    if features.get("step_count") in ["9-12步", "12步以上"]:
        high.append("9步以上复杂推理")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        high.append("公式链较长")
    if features.get("knowledge_count") == "4个及以上":
        high.append("4个及以上知识点")
    if features.get("cross_module") == "跨模块综合":
        high.append("跨模块综合")
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        core.append("复杂方程或范围计算")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        core.append("逆向推理或临界分析")
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        core.append(features.get("state_count"))
    if features.get("constraint_count") == "多约束":
        core.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        core.append("多变量耦合")
    if features.get("graph_table_requirement") == "图像反推或外推":
        core.append("图像反推或外推")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        core.append("方案设计或误差评价")
    if contains_any(text, ["不等式", "分类讨论", "多解", "筛选", "黑箱", "非线性", "I-U", "U-I", "多开关", "满偏", "自主设计", "可行性", "挑战赛", "改装", "创新"]):
        core.append("分类讨论/创新设计/复杂筛选信号")
    return len(set(core)) >= 3 and len(set(high + core)) >= 5, high + core


def should_downgrade_final_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    if is_project_or_innovation_final_case(features, data):
        return False
    if has_high_level_project_or_control_signal(data) and features.get("constraint_count") == "多约束":
        return False
    # 非项目式常规动态电路仍可降。
    if is_common_dynamic_circuit_not_final(features, data) and not has_high_level_project_or_control_signal(data):
        return True
    text = full_text_of(data)
    if "密度计" in text and not contains_any(text, ["弹簧", "杠杆", "液面变化", "连续变化", "多状态", "极值", "范围", "分类讨论", "可行"]):
        return True
    keep_keywords = ["不等式", "分类讨论", "多解", "筛选", "黑箱", "非线性", "I-U", "U-I", "多开关", "满偏", "自主设计", "可行性", "创新"]
    high_features = 0
    if features.get("step_count") in ["9-12步", "12步以上"]:
        high_features += 1
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        high_features += 1
    if features.get("knowledge_count") == "4个及以上":
        high_features += 1
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        high_features += 1
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        high_features += 1
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        high_features += 1
    if features.get("constraint_count") == "多约束":
        high_features += 1
    if features.get("variable_relation") == "多变量耦合关系":
        high_features += 1
    if features.get("graph_table_requirement") == "图像反推或外推":
        high_features += 1
    if contains_any(text, keep_keywords):
        high_features += 1
    return high_features < 3




# -------------------------- 3.12 V7 送分白名单 / 中等弱保护 / 杠杆补强覆盖层 --------------------------
# 说明：V7 在 V6 最优版本基础上只做小补丁：
# 1. 小幅放宽老师口径下的“容易题”：教材原型作图、简单结构图、直接光现象/估测可判送分；
# 2. 防止把“频率/压强/材料属性/惯性应用”等简单应用误降为送分；
# 3. 给阅读材料多问、常规实验现象解释、空间方位/磁场判断、机械效率计算加中等弱保护；
# 4. 补强餐盘/杠杆/力臂、等效密度测量、回音壁路径等低计算高建模较难题；
# 5. 继续把普通压力秤/常规压敏电阻图像题保护在中等，不误升拔高。


def is_plain_pressure_scale_medium_case(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """普通压力秤/压敏电阻图像题：量程 + 欧姆定律 + 图像读数，按中等保护。"""
    text = full_text_of(data)
    if features.get("problem_structure") != "电路综合":
        return False
    if not contains_any(text, ["压力秤", "压敏电阻", "压力", "R-F", "阻值变化与所受压力"]):
        return False
    hard_cues = ["黑箱", "非线性", "I-U", "U-I", "多开关", "分类讨论", "不等式", "多解", "筛选", "自主设计", "实验方案", "项目", "挑战"]
    if contains_any(text, hard_cues):
        return False
    return (
        contains_any(text, ["量程", "电流表", "电压表", "图像", "图乙", "最大", "最大测量"])
        and features.get("calculation_complexity") in ["简单笔算", "多公式联立"]
        and features.get("graph_table_requirement") in ["直接读数", "多组比较归纳"]
        and features.get("variable_relation") in ["简单正反比", "图像函数关系"]
    )


def is_textbook_easy_diagram_or_direct_fill(data: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """老师口径下可判送分的教材原型作图/结构图/直接填空。"""
    visible = visible_text_of(data)
    text = full_text_of(data)

    # 一票否决：这些虽然可能简单，但老师更常判较易/基础。
    block = [
        "平面镜成像", "磁感线", "螺线管", "安培定则", "电路", "连接", "实验", "探究", "装置", "表格", "数据",
        "菜刀", "刀刃", "受力面积", "压强", "频率", "每秒振动", "低于", "高于", "简谱", "音符", "音阶",
        "合金", "强度", "坚韧", "轻巧", "新材料", "材料", "主要利用惯性", "惯性的是", "杠杆", "滑轮"
    ]
    if contains_any(visible, block) or has_formula_calculation_intent(visible):
        return False

    common_feature_ok = (
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
    )
    if not common_feature_ok:
        return False

    # 1. 静止水平面单物体受力示意图，老师样本判容易。
    if contains_any(visible, ["静止在水平", "水平地面", "受力示意图"]) and contains_any(visible, ["篮球", "物体", "重力", "支持力"]):
        return True

    # 2. 凸透镜特殊光线教材原型作图。
    if contains_any(visible, ["画出经透镜射出的光线", "经透镜射出的光线", "透镜射出的光线", "特殊光线"]):
        return True

    # 3. 分子动理论/粒子宇宙/教材结构图纯识记。
    if contains_any(visible, ["分子动理论", "知识结构图", "粒子与宇宙", "原子核", "光年", "汤姆孙", "电子"]):
        return True

    # 4. 生活估测直接判断。
    if contains_any(visible, ["公交车", "一辆普通公交车", "市区正常行驶"]):
        return True

    # 5. 直接光现象/热值/扩散/能源优点填空，允许两个独立低阶空。
    if contains_any(visible, ["树荫", "水中游鱼", "光的", "热值", "满屋飘香", "花香", "扩散", "风力发电", "直线传播", "折射"]):
        return True

    return False


def is_simple_picture_or_context_easy_question(data: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """V7：只放宽老师明确按容易处理的教材原型；排除简单应用题。"""
    visible = visible_text_of(data)

    if is_textbook_easy_diagram_or_direct_fill(data, features):
        return True

    # 简单应用/生活应用题至少基础，不降送分。
    application_block = [
        "菜刀", "刀刃", "受力面积", "压强", "主要利用惯性", "惯性的是", "每秒振动", "频率", "简谱", "音符", "音阶",
        "合金", "强度", "密度小", "密度大", "坚韧", "轻巧", "新材料", "仿生机器人"
    ]
    if contains_any(visible, application_block):
        return False

    # 保留少量纯概念直答，不再使用“主要/描述的是”这种泛化触发词。
    easy_cues = [
        "单位", "物理学家", "首先", "命名", "属于哪种", "属于哪一", "不可再生", "可再生",
        "扩散现象", "凝固", "熔化", "汽化", "液化", "升华", "凝华", "响度", "音色", "红绿蓝", "竖直向下", "无线电波", "电磁波", "凸透镜"
    ]
    return (
        contains_any(visible, easy_cues)
        and features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") in ["直接套用", "简单因果推理"]
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_diff") == "低"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )


def is_pure_recognition_easy_question(data: Dict[str, Any], features: Dict[str, Any]) -> bool:
    """V7：送分题白名单 + 严格排除简单应用。"""
    visible = visible_text_of(data)
    if is_textbook_easy_diagram_or_direct_fill(data, features):
        return True
    if is_simple_picture_or_context_easy_question(data, features):
        return True
    if contains_any(visible, [
        "画出", "作出", "作图", "示意图", "根据图", "由图", "如图", "图甲", "图乙",
        "平面镜", "成像", "光路", "反射光", "折射", "磁感线", "螺线管", "安培定则",
        "电路", "连接", "实验", "探究", "装置", "说明", "分析",
        "菜刀", "刀刃", "受力面积", "压强", "频率", "每秒振动", "简谱", "音符", "音阶", "合金", "强度", "坚韧", "轻巧", "材料", "惯性的是"
    ]):
        return False
    if has_formula_calculation_intent(visible):
        return False
    if is_choice_like_question(visible) and len(visible) > 110:
        return False
    easy_cues = ["单位", "物理学家", "首先", "命名", "属于", "哪种", "哪一个", "哪个", "特性", "方向是", "不可再生", "可再生", "扩散", "凝固", "熔化", "汽化", "液化", "升华", "凝华", "响度", "音调", "音色", "红绿蓝", "竖直向下"]
    return (
        contains_any(visible, easy_cues)
        and features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_diff") == "低"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )


def is_weak_medium_protection_case(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V7：老师判中等但系统容易压基础的弱保护。只保护到中等，不升拔高。"""
    text = full_text_of(data)
    visible = visible_text_of(data)

    # 阅读材料多问 + 信息提取/公式推导。
    reading_medium = (
        (contains_any(text, ["阅读短文", "阅读材料"]) or (contains_any(text, ["卫星", "运载火箭", "捷龙"]) and contains_any(text, ["周期", "轨道", "圆周", "半径", "周长"])))
        and features.get("step_count") in ["3-5步", "6-8步"]
        and features.get("knowledge_count") in ["2-3个", "4个及以上"]
    )

    # 常规机械效率三步综合。
    mech_eff_medium = (
        contains_any(text, ["有用功", "总功", "机械效率"])
        and features.get("calculation_complexity") in ["简单笔算", "多公式联立"]
        and features.get("formula_count") in ["2-3个", "4-6个"]
    )

    # 多实验现象解释/实验小问综合。
    multi_exp_medium = (
        contains_any(text, ["有趣的实验", "爱探索", "以下有趣", "多个实验", "四个实验", "托里拆利", "分子间隙", "扩散实验"])
        and contains_any(text, ["声音的产生", "大气压", "浮力产生", "流体压强", "实验现象", "解释正确"])
        and features.get("knowledge_count") in ["2-3个", "4个及以上"]
    )

    # 玻璃镇纸/跨力光的浅综合。
    glass_paperweight_medium = contains_any(text, ["玻璃", "镇纸", "书法", "半球形"]) and contains_any(text, ["平衡力", "凸透镜", "实像", "虚像", "合力"])

    # 地磁方位/空间方向判断。
    spatial_magnet_medium = contains_any(text, ["条形磁铁", "自由旋转", "地理方位", "地磁", "白纸上标注"])

    # 熔化/反射/串联电压等标准实验流程归纳。
    standard_flow_medium = (
        features.get("problem_structure") == "实验探究"
        and contains_any(text, ["熔化", "光的反射", "串联电路电压", "实验步骤", "表格", "实验序号", "结论"])
    )

    return reading_medium or mech_eff_medium or multi_exp_medium or glass_paperweight_medium or spatial_magnet_medium or standard_flow_medium


def is_mid_level_parallel_analysis(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V7：在 V6 强保护基础上加入少量中等弱保护。"""
    if is_weak_medium_protection_case(features, data):
        return True

    visible = visible_text_of(data)
    text = full_text_of(data)
    if not is_parallel_choice_or_independent_points(data, features):
        return False
    if is_pure_recognition_easy_question(data, features):
        return False
    if contains_any(visible, ["求", "计算"]) and features.get("calculation_complexity") in ["简单笔算", "口算或直接判断"]:
        if features.get("subquestion_dependency") in ["多问但相互独立", "无多问"]:
            return False
    simple_parallel_cues = ["油条", "塑性形变", "热传递", "热量是", "不可再生能源", "可再生能源", "大气压", "吸管喝水", "拔火罐", "水瓶琴", "声音的传播", "响度", "音调", "音色", "验电器", "同种电荷", "异种电荷", "电动机", "发电机", "电热毯", "电水壶", "功和功率", "起重机", "工人搬运", "水果电池", "发光二极管"]
    if contains_any(text, simple_parallel_cues) and features.get("experiment_requirement") == "无":
        return False
    strong_experiment = (
        features.get("problem_structure") == "实验探究"
        and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"]
        and (features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"] or contains_any(text, ["实验步骤", "记录数据", "表格", "多次实验", "控制变量", "实验结论", "反射定律", "串联电路电压"]))
    )
    mechanics_terms = ["受力", "惯性", "力可以改变", "运动状态", "摩擦力", "重力", "机械能", "动能", "重力势能"]
    strong_real_process = contains_any(text, ["钢卷尺", "纸飞机", "飞离", "滑行", "每隔相等时间", "运动过程", "斜坡", "水平面"]) and sum(1 for k in mechanics_terms if k in text) >= 3 and features.get("reasoning_chain") in ["简单因果推理", "多层因果推理"]
    strong_spatial_or_graph = features.get("information_carrier") in ["实验装置图", "图像或表格", "多图表综合"] and contains_any(text, ["方位", "位置", "入射角", "反射角", "法线", "实验序号", "表格", "图像", "数据"]) and features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"]
    strong_error_parallel = features.get("knowledge_count") == "4个及以上" and features.get("error_risk") in ["明显易错点", "高易错点"] and features.get("reasoning_chain") == "多层因果推理" and len(visible) > 180 and not contains_any(text, simple_parallel_cues)
    return strong_experiment or strong_real_process or strong_spatial_or_graph or strong_error_parallel


def is_common_medium_circuit_or_experiment(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V7：压力秤、常规双挡电热器保护到中等；项目/自动控制仍排除。"""
    text = full_text_of(data)
    problem = features.get("problem_structure")
    if is_plain_pressure_scale_medium_case(features, data):
        return True
    if has_high_level_project_or_control_signal(data) and not is_plain_pressure_scale_medium_case(features, data):
        return False
    hard_block_cues = ["黑箱", "非线性", "I-U", "U-I", "多开关", "分类讨论", "不等式", "多解", "自主设计", "缺表", "等效替代", "特殊方法", "表达式", "异常点", "偏离", "新猜想", "挑战赛", "改装", "临界", "极值", "筛选", "可行", "方案", "项目", "自动控制", "继电器", "安全电流", "载流量", "导线选择"]
    if contains_any(text, hard_block_cues):
        return False
    circuit_medium = (
        problem in ["电路综合", "跨模块综合"]
        and contains_any(text, ["量程", "滑动变阻器", "电热水壶", "双挡", "低温挡", "高温挡", "图像", "图乙", "最大", "保温"])
        and features.get("calculation_complexity") in ["简单笔算", "多公式联立"]
        and features.get("reasoning_chain") in ["简单因果推理", "多层因果推理"]
        and features.get("state_count") in ["单状态", "双状态", "多状态"]
        and features.get("variable_relation") in ["无变量关系", "简单正反比", "图像函数关系"]
    )
    experiment_medium = (
        problem == "实验探究"
        and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"]
        and contains_any(text, ["测量", "密度", "小灯泡", "电功率", "表格", "数据", "误差", "实验步骤", "电压表", "电流表"])
        and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算", "多公式联立"]
        and features.get("state_count") in ["单状态", "双状态"]
        and features.get("constraint_count") in ["无约束", "单一约束"]
    )
    return circuit_medium or experiment_medium


def is_hidden_process_hard_case(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V7：补强餐盘杠杆、等效测量误差、回音壁路径、线路/导线筛选等低计算高建模题。"""
    text = full_text_of(data)

    force_patterns = [
        ["餐盘", "保持水平", "力臂"],
        ["餐盘", "支点", "杠杆"],
        ["杠杆", "力臂", "作图"],
        ["枇杷", "密度", "带出"],
        ["枇杷", "等效", "误差"],
        ["天坛", "回音壁", "第三个回声"],
        ["空调", "线路", "载流量"],
        ["三挡位", "电热器", "导线"],
        ["三挡位", "安全", "载流量"],
        ["干冰", "空气浮力", "密度"],
    ]
    if any(all(k in text for k in p) for p in force_patterns):
        return True

    patterns = [
        ["空调", "线路", "载流量"], ["教室", "加装", "空调"], ["杠杆", "力臂", "角度"], ["头颈", "杠杆"],
        ["汲水", "玻璃管", "最佳组合"], ["管口", "快速上提", "大气压"], ["虹", "球形水珠", "法线"], ["色散", "反射", "折射", "α"],
        ["带出部分水", "仍然准确"], ["天坛", "回音壁", "第三个回声"], ["电磁继电器", "热敏电阻", "自动控温"], ["发热体", "安全电流", "温度范围"]
    ]
    hit_pattern = any(all(k in text for k in p) for p in patterns)
    hard_reasoning = features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"] or features.get("subquestion_dependency") == "多问且层层递进" or contains_any(text, ["最佳组合", "顺序", "几何", "力臂", "可行", "准确", "为什么", "原因", "筛选", "安全"])
    return hit_pattern and hard_reasoning and not is_parallel_choice_or_independent_points(data, features)


def should_downgrade_basic_to_easy(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    # V7：长情境一般不降，但教材原型作图/结构图/直接填空可以降。
    if is_textbook_easy_diagram_or_direct_fill(data, features):
        return True
    if is_long_context_or_new_situation(data):
        return False
    return is_pure_recognition_easy_question(data, features) or is_simple_picture_or_context_easy_question(data, features)


def should_upgrade_easy_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons = []
    visible = visible_text_of(data)
    text = full_text_of(data)
    if is_pure_recognition_easy_question(data, features) or is_simple_picture_or_context_easy_question(data, features):
        return reasons
    # V7：这些不是送分，是简单应用/基础辨析。
    if contains_any(visible, ["菜刀", "刀刃", "受力面积", "压强"]):
        reasons.append("压强规律的生活应用，至少基础题")
    if contains_any(visible, ["每秒振动", "低于20", "人耳听不到"]):
        reasons.append("需要把振动次数转化为频率概念，至少基础题")
    if contains_any(visible, ["简谱", "音符", "音阶", "3、5、6"]):
        reasons.append("需要理解简谱音符与音调的对应，至少基础题")
    if contains_any(visible, ["合金", "硬度", "强度", "密度", "新材料", "坚韧", "轻巧", "仿生机器人"]):
        reasons.append("材料物理属性的生活应用，至少基础题")
    if contains_any(visible, ["主要利用惯性", "惯性的是", "摔打", "脱粒"]):
        reasons.append("惯性在生活场景中的应用辨析，至少基础题")
    if contains_any(visible, ["画出", "作出", "作图", "示意图"]):
        reasons.append("涉及规范作图，至少基础题")
    if contains_any(visible, ["平面镜", "成像", "磁感线", "螺线管", "安培定则", "光路", "电路", "连接"]):
        reasons.append("涉及规范作图/电磁或电路规律应用，至少基础题")
    if is_choice_like_question(visible) and len(visible) > 150 and features.get("knowledge_count") != "1个":
        reasons.append("选择题存在多选项辨析，至少基础题")
    if features.get("information_carrier") in ["电路图", "实验装置图", "多图表综合", "图像或表格"] and contains_any(visible, ["根据图", "由图", "读数", "表格", "电路", "实验"]):
        reasons.append(f'信息载体为"{features.get("information_carrier")}"，需要识图或读图')
    if features.get("experiment_requirement") != "无":
        reasons.append("含有实验操作探究")
    if features.get("problem_structure") == "直接计算" or has_formula_calculation_intent(visible):
        reasons.append("需要公式代入或物理量计算，至少基础题")
    if features.get("knowledge_count") != "1个" and not is_parallel_choice_or_independent_points(data, features):
        reasons.append(f'知识点数量为"{features.get("knowledge_count")}"')
    if features.get("state_count") != "单状态":
        reasons.append(f'物理状态数量为"{features.get("state_count")}"')
    if features.get("constraint_count") != "无约束":
        reasons.append("存在物理约束条件")
    if count_fill_blanks(text) >= 3 and not is_parallel_choice_or_independent_points(data, features):
        reasons.append("多空填空题且存在综合关系，不宜判为送分题")
    return reasons


def should_upgrade_basic_to_medium(features: Dict[str, Any], data: Dict[str, Any]) -> List[str]:
    reasons = []
    text = full_text_of(data)
    step = features.get("step_count")
    state = features.get("state_count")
    calc = features.get("calculation_complexity")
    cross = features.get("cross_module")
    problem = features.get("problem_structure")
    variable = features.get("variable_relation")
    experiment = features.get("experiment_requirement")
    graph = features.get("graph_table_requirement")
    formula = features.get("formula_count")
    knowledge = features.get("knowledge_count")
    reasoning = features.get("reasoning_chain")

    if is_hidden_process_hard_case(features, data):
        reasons.append("低计算但存在隐含过程/力臂/误差/路径建模，至少中等")
        return reasons
    if is_weak_medium_protection_case(features, data):
        reasons.append("阅读材料/实验现象/空间方位/机械效率等整题综合达到中等题")
        return reasons
    if is_mid_level_parallel_analysis(features, data):
        reasons.append("强易错/实验流程/真实连续过程辨析达到中等题，但不按拔高处理")
        return reasons
    if is_parallel_choice_or_independent_points(data, features):
        return reasons
    if step in ["6-8步", "9-12步", "12步以上"] and has_real_integration(features, data):
        reasons.append(f'步骤数达"{step}"且存在真实综合')
    if step == "3-5步" and knowledge in ["2-3个", "4个及以上"] and has_real_integration(features, data) and (reasoning in ["多层因果推理", "逆向推理或临界分析"] or problem in ["实验探究", "图像表格分析", "电路综合", "力学综合", "光学声学综合", "跨模块综合"] or formula in ["2-3个", "4-6个", "7个以上"]):
        reasons.append("3-5步连续推理且涉及常规模型转化，达到中等题")
    if state in ["多状态", "连续变化或临界状态"]:
        reasons.append(f'物理状态为"{state}"')
    if state == "双状态" and has_real_integration(features, data) and (calc in ["多公式联立", "复杂方程或范围计算", "简单笔算"] or problem in ["电路综合", "力学综合", "跨模块综合"] or variable in ["图像函数关系", "多变量耦合关系"] or experiment in ["控制变量或故障分析", "方案设计或误差评价"]):
        reasons.append("双状态且伴随真实物理建模")
    if calc in ["多公式联立", "复杂方程或范围计算"]:
        reasons.append(f'计算需要"{calc}"')
    if cross == "跨模块综合" and has_real_integration(features, data):
        reasons.append("跨模块且存在实质综合")
    if graph in ["多组比较归纳", "图像反推或外推"]:
        reasons.append("需要图表信息处理")
    standard_experiment_medium_keywords = ["电磁铁磁性强弱", "液体压强", "压强计", "平均速度", "滑动摩擦力", "摩擦力大小", "杠杆平衡", "电流与电阻", "电流与电压", "重力与质量", "凸透镜成像", "熔化实验", "沸腾实验", "比热容", "焦耳定律", "光的反射定律", "串联电路电压"]
    if features.get("problem_structure") == "实验探究" and features.get("experiment_requirement") in ["控制变量或故障分析", "基础操作或读数"] and contains_any(text, standard_experiment_medium_keywords):
        reasons.append("标准实验探究题，涉及控制变量或实验归纳，至少中等题")
    return reasons


def should_downgrade_medium_to_basic(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    if is_weak_medium_protection_case(features, data):
        return False
    if is_mid_level_parallel_analysis(features, data):
        return False
    if features.get("problem_structure") in ["实验探究", "电路综合", "力学综合", "跨模块综合"] and has_real_integration(features, data):
        return False
    if is_hidden_process_hard_case(features, data):
        return False
    text = full_text_of(data)
    if features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"]:
        return False
    if features.get("experiment_requirement") != "无":
        return False
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        return False
    if features.get("constraint_count") == "多约束":
        return False
    if features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]:
        return False
    simple_direct = features.get("step_count") in ["1-2步", "3-5步"] and features.get("formula_count") in ["0-1个", "2-3个"] and features.get("calculation_complexity") in ["口算或直接判断", "简单笔算"] and features.get("reasoning_chain") in ["直接套用", "简单因果推理"]
    simple_keywords = ["水平移动", "竖直举高", "做功", "动能", "机械能", "洒水车", "质量减小", "速度不变", "仪器使用", "估测", "生活常识"]
    return simple_direct and contains_any(text, simple_keywords)


def should_force_upgrade_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    """V7：少量较难题直接保护，解决基础->拔高严重低估。"""
    return is_hidden_process_hard_case(features, data) and contains_any(full_text_of(data), [
        "餐盘", "力臂", "杠杆", "枇杷", "带出", "等效", "误差", "天坛", "回音壁", "空调", "载流量", "三挡位", "导线", "干冰", "空气浮力"
    ])


def should_upgrade_medium_to_hard(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    text = full_text_of(data)
    stem = data.get("stem", "") or ""
    if is_plain_pressure_scale_medium_case(features, data):
        return False, ["普通压力秤/压敏电阻图像题，按中等保护"]
    if is_project_or_innovation_final_case(features, data):
        return True, ["项目式/创新设计题，存在可行性、边界或安全约束分析"]
    if is_parallel_choice_or_independent_points(data, features) and not is_hidden_process_hard_case(features, data):
        return False, ["并列知识点/独立选项，不按真正综合升拔高"]
    if is_common_medium_circuit_or_experiment(features, data):
        return False, ["常规电学/实验题，模型清楚，按中等保护"]
    core, support = [], []
    if features.get("step_count") in ["6-8步", "9-12步", "12步以上"]:
        support.append("步骤数达到较难题要求")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        support.append("公式链较长")
    if features.get("knowledge_count") == "4个及以上":
        support.append("知识点数量较多")
    if features.get("cross_module") == "跨模块综合":
        support.append("跨章节综合")
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        core.append("复杂方程或范围计算")
    elif features.get("calculation_complexity") == "多公式联立" and features.get("formula_count") in ["2-3个", "4-6个", "7个以上"]:
        core.append("需要方程联立或多公式联动")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        core.append("逆向推理或临界分析")
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        core.append("包含多状态或临界状态")
    if features.get("state_count") == "双状态" and features.get("problem_structure") in ["电路综合", "力学综合", "跨模块综合"] and has_real_integration(features, data):
        core.append("双状态综合模型")
    if features.get("constraint_count") == "多约束":
        core.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        core.append("多变量耦合")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        core.append("方案设计或误差评价")
    if features.get("graph_table_requirement") == "图像反推或外推":
        core.append("图像反推或外推")
    if contains_any(text, ["隐含", "临界", "极值", "取值范围", "比例", "差值", "方程组", "几何关系", "分类讨论", "不等式", "误差", "评价", "可行", "安全筛选", "力臂", "路径"]):
        core.append("存在隐含条件、范围极值、几何转化或评价要求")
    if is_real_multi_object_modeling(features, data):
        core.append("真正多对象建模")
    force_hard = False
    if is_hidden_process_hard_case(features, data):
        core.append("半陌生情境中的隐含条件/过程顺序/几何转化")
        force_hard = True
    if len(stem) > 300 and features.get("experiment_requirement") in ["控制变量或故障分析", "方案设计或误差评价"] and features.get("graph_table_requirement") in ["多组比较归纳", "图像反推或外推"] and has_real_integration(features, data):
        core.append("长实验题且需要实验分析与图表处理")
        force_hard = True
    if len(stem) > 300 and contains_any(stem, ["阅读", "材料", "装置", "项目", "实践", "挑战"]) and features.get("variable_relation") in ["图像函数关系", "多变量耦合关系"]:
        core.append("长材料题且需要变量关系分析")
        force_hard = True
    triggers = core + support
    return force_hard or (len(set(core)) >= 1 and len(set(triggers)) >= 2), triggers


def should_downgrade_hard_to_medium(features: Dict[str, Any], data: Dict[str, Any]) -> bool:
    if is_plain_pressure_scale_medium_case(features, data):
        return True
    if is_project_or_innovation_final_case(features, data):
        return False
    if has_high_level_project_or_control_signal(data) and not is_plain_pressure_scale_medium_case(features, data):
        return False
    if is_hidden_process_hard_case(features, data):
        return False
    if is_real_multi_object_modeling(features, data) and features.get("reasoning_chain") in ["多层因果推理", "逆向推理或临界分析"]:
        return False
    high_count = 0
    for cond in [features.get("constraint_count") == "多约束", features.get("state_count") in ["多状态", "连续变化或临界状态"], features.get("variable_relation") == "多变量耦合关系", features.get("calculation_complexity") == "复杂方程或范围计算", features.get("reasoning_chain") == "逆向推理或临界分析", features.get("graph_table_requirement") == "图像反推或外推"]:
        if cond:
            high_count += 1
    if high_count >= 2:
        return False
    return is_common_medium_circuit_or_experiment(features, data)



def should_upgrade_hard_to_final(features: Dict[str, Any], data: Dict[str, Any]) -> tuple:
    """V7：普通压力秤不升压轴；项目式/创新设计保留压轴。"""
    if is_plain_pressure_scale_medium_case(features, data):
        return False, ["普通压力秤/压敏电阻图像题，未达到压轴复杂度"]
    if is_project_or_innovation_final_case(features, data):
        return True, ["项目式创新/自动控制/方案可行性验证达到压轴特征"]
    ok_continuous, continuous_reasons = should_upgrade_continuous_buoyancy_to_final(features, data)
    if ok_continuous:
        return True, continuous_reasons
    stem = data.get("stem", "") or ""
    text = full_text_of(data)
    if is_common_dynamic_circuit_not_final(features, data) and not has_high_level_project_or_control_signal(data):
        return False, ["常规动态电路/传感器电路，未达到压轴复杂度"]
    high, core = [], []
    if features.get("step_count") in ["9-12步", "12步以上"]:
        high.append("9步以上复杂推理")
    if features.get("formula_count") in ["4-6个", "7个以上"]:
        high.append("公式链较长")
    if features.get("knowledge_count") == "4个及以上":
        high.append("4个及以上知识点")
    if features.get("cross_module") == "跨模块综合":
        high.append("跨模块综合")
    if features.get("calculation_complexity") == "复杂方程或范围计算":
        core.append("复杂方程或范围计算")
    if features.get("reasoning_chain") == "逆向推理或临界分析":
        core.append("逆向推理或临界分析")
    if features.get("state_count") in ["多状态", "连续变化或临界状态"]:
        core.append(features.get("state_count"))
    if features.get("constraint_count") == "多约束":
        core.append("多约束")
    if features.get("variable_relation") == "多变量耦合关系":
        core.append("多变量耦合")
    if features.get("graph_table_requirement") == "图像反推或外推":
        core.append("图像反推或外推")
    if features.get("experiment_requirement") == "方案设计或误差评价":
        core.append("方案设计或误差评价")
    if contains_any(text, ["不等式", "分类讨论", "多解", "筛选", "黑箱", "非线性", "I-U", "U-I", "多开关", "满偏", "自主设计", "可行性", "挑战赛", "改装", "创新"]):
        core.append("分类讨论/创新设计/复杂筛选信号")
    return len(set(core)) >= 3 and len(set(high + core)) >= 5, high + core

def postprocess_physics_difficulty(rating_result: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    """V7 边界小修版：默认一档调整，但对少量严重低估较难题允许直接保护到拔高。"""
    if not rating_result:
        return rating_result
    rating_result["features"] = normalize_features(rating_result.get("features", {}))
    normalize_reasoning_schema(rating_result)
    adjusted = False

    # 0. 严重低估保护：餐盘杠杆/等效测量误差/回音壁/线路筛选等低计算高建模题。
    if rating_result.get("difficulty_level") in ["基础题", "中等题"] and should_force_upgrade_to_hard(rating_result["features"], data):
        set_level_with_reason(rating_result, "拔高题", "自动升档：低计算但存在隐含建模、力臂/误差/路径/安全筛选等较难题特征")
        adjusted = True

    if rating_result.get("difficulty_level") == "基础题" and not adjusted:
        if should_downgrade_basic_to_easy(rating_result["features"], data):
            set_level_with_reason(rating_result, "送分题", "自动降档：教师口径下属于教材原型直接识别题")
            adjusted = True
    if rating_result.get("difficulty_level") == "送分题" and not adjusted:
        reasons = should_upgrade_easy_to_basic(rating_result["features"], data)
        if reasons:
            set_level_with_reason(rating_result, "基础题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "基础题" and not adjusted:
        reasons = should_upgrade_basic_to_medium(rating_result["features"], data)
        if reasons:
            set_level_with_reason(rating_result, "中等题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "中等题" and not adjusted:
        if should_downgrade_medium_to_basic(rating_result["features"], data):
            set_level_with_reason(rating_result, "基础题", "自动降档：单公式短步骤题，未达到中等复杂度")
            adjusted = True
    if rating_result.get("difficulty_level") == "中等题" and not adjusted:
        ok, reasons = should_upgrade_medium_to_hard(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "拔高题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "拔高题" and not adjusted:
        if should_downgrade_hard_to_medium(rating_result["features"], data):
            set_level_with_reason(rating_result, "中等题", "自动降档：常规电学/实验题或普通压力秤题，未达到拔高复杂度")
            adjusted = True
    if rating_result.get("difficulty_level") == "拔高题" and not adjusted:
        downgraded = should_downgrade_standard_experiment(rating_result["features"], data)
        if downgraded:
            set_level_with_reason(rating_result, downgraded, f"自动降档：标准实验题，未达到拔高复杂度，降为{downgraded}")
            adjusted = True
    if rating_result.get("difficulty_level") == "拔高题" and not adjusted:
        ok, reasons = should_upgrade_hard_to_final(rating_result["features"], data)
        if ok:
            set_level_with_reason(rating_result, "压轴题", "自动升档：" + "；".join(reasons))
            adjusted = True
    if rating_result.get("difficulty_level") == "压轴题" and not adjusted:
        if should_downgrade_final_to_hard(rating_result["features"], data):
            set_level_with_reason(rating_result, "拔高题", "自动降档：压轴高阶物理特征不足")
            adjusted = True
    sync_coarse_difficulty(rating_result)
    return rating_result

# -------------------------- 4. 构建网络调用 --------------------------
def construct_question_content(data: Dict[str, Any]) -> str:
    """将数据记录拼装成标准的打标输入文本"""
    parts = []
    stem = data.get('stem', '').strip()
    options = data.get('options', '').strip()
    analysis = data.get('analysis', '').strip()

    if stem:
        parts.append(f"【题干】\n{stem}")
    if options:
        parts.append(f"【选项】\n{options}")
    if analysis:
        parts.append(f"【解析】\n{analysis}")

    sub_questions = data.get('sub_questions', [])
    if sub_questions:
        try:
            # 依 ID 排序让子小题更规范
            sub_questions.sort(key=lambda x: int(x.get('question_id', 0)) if isinstance(x, dict) else 0)
        except Exception:
            pass

        parts.append("【小题】")
        for i, sq in enumerate(sub_questions, 1):
            parts.append(f"  小题{i}:")
            if isinstance(sq, dict):
                sq_stem = str(sq.get("stem", "")).strip()
                sq_options = str(sq.get("options", "")).strip()
                sq_analysis = str(sq.get("analysis", "")).strip()

                if sq_stem:
                    parts.append(f"    题干: {sq_stem}")
                if sq_options:
                    parts.append(f"    选项: {sq_options}")
                if sq_analysis:
                    parts.append(f"    解析: {sq_analysis}")
            else:
                parts.append(f"    题干: {sq}")

    return "\n\n".join(parts)

def parse_model_response(response_text: str) -> Dict:
    """容错并修复 JSON 输出"""
    if not response_text:
        return {}
    try:
        return json_repair.loads(response_text)
    except Exception:
        pass
    try:
        clean_text = response_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json")[1].split("```")[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```")[1].split("```")[0]
        return json_repair.loads(clean_text.strip())
    except Exception:
        pass
    try:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1:
            return json_repair.loads(response_text[start:end+1])
    except Exception:
        pass
    return {}

async def call_model_with_cache(
    question_content: str, 
    session: aiohttp.ClientSession, 
    retries: int, 
    timeout_sec: int
) -> tuple:
    """获取缓存并执行 API 接口调用"""
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
            "thinking": {"type": "disabled"}
        }

        t1 = time.time()
        try:
            async with session.post(
                f"{BASE_URL}responses",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=timeout_sec)
            ) as response:
                
                # 200 成功
                if response.status == 200:
                    result = await response.json()
                    output_text = ""
                    if 'output' in result:
                        for item in result['output']:
                            if item.get('type') == 'message' and 'content' in item:
                                for content_item in item['content']:
                                    if content_item.get('type') == 'output_text':
                                        output_text = content_item.get('text', '')

                    usage = result.get('usage', {})
                    prompt_tokens = usage.get('input_tokens', 0)
                    completion_tokens = usage.get('output_tokens', 0)
                    total_tokens = usage.get('total_tokens', 0)

                    parsed_result = parse_model_response(output_text)
                    t2 = time.time()
                    return parsed_result, t2 - t1, prompt_tokens, completion_tokens, total_tokens

                # 429 限流
                elif response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    print(f"接口限流(429)，等待 {retry_after} 秒后进行第 {retry+1} 次重试...")
                    await asyncio.sleep(retry_after)
                    continue

                else:
                    error_text = await response.text()
                    print(f"API请求失败 (状态码: {response.status}): {error_text[:200]}")

                    # 缓存丢失错误 -> 重建缓存
                    if "InvalidParameter.PreviousResponseNotFound" in error_text:
                        print("检测到服务器缓存丢失，正在重建缓存...")
                        new_response_id = await create_prefix_cache(session, retries, timeout_sec)
                        if not new_response_id:
                            return {}, 0.0, 0, 0, 0
                        response_id = new_response_id
                        continue

                    # 5xx 服务端错误 -> 阶梯退避
                    if response.status >= 500:
                        backoff = (2 ** retry) + random.uniform(0, 1)
                        print(f"服务器故障({response.status})，{backoff:.2f}秒后重试 (第{retry+1}次)...")
                        await asyncio.sleep(backoff)
                        continue

                    # 其他 4xx 客户端直接错误（无需重试）
                    if 400 <= response.status < 500:
                        return {}, 0.0, 0, 0, 0

        except aiohttp.ClientError as e:
            backoff = (2 ** retry) + random.uniform(0, 1)
            if retry == retries - 1:
                print(f"网络异常最终失败: {e}")
                return {}, 0.0, 0, 0, 0
            print(f"网络出现异常: {e}，将进行退避 {backoff:.2f} 秒后重试...")
            await asyncio.sleep(backoff)
            continue

        except Exception as e:
            print(f"运行过程中请求异常: {e}")
            if retry == retries - 1:
                return {}, 0.0, 0, 0, 0
            print("尝试重新建构缓存后再次请求...")
            new_response_id = await create_prefix_cache(session, retries, timeout_sec)
            if new_response_id:
                response_id = new_response_id
            await asyncio.sleep(1)
            continue

    return {}, 0.0, 0, 0, 0

# -------------------------- 5. 协程分发处理 --------------------------
async def process_single_question(
    data: Dict[str, Any], 
    session: aiohttp.ClientSession, 
    semaphore: Semaphore,
    output_path: str,
    error_path: str,
    retries: int,
    timeout_sec: int
) -> None:
    """并发调度单元"""
    async with semaphore:
        question_id = data.get('question_id', 'unknown')
        try:
            question_content = construct_question_content(data)
            rating_result, time_use, prompt_tokens, completion_tokens, total_tokens = await call_model_with_cache(
                question_content, session, retries, timeout_sec
            )

            # 执行物理升降档纠偏后处理
            rating_result = postprocess_physics_difficulty(rating_result, data)

            output_data = data.copy()
            output_data['difficulty_rating'] = rating_result
            output_data['api_time_use'] = round(time_use, 2)
            output_data['api_prompt_tokens'] = prompt_tokens
            output_data['api_completion_tokens'] = completion_tokens
            output_data['api_total_tokens'] = total_tokens

            if rating_result and rating_result.get('difficulty_level'):
                async with FILE_LOCK:
                    async with aiofiles.open(output_path, "a", encoding="utf-8") as f:
                        await f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            else:
                output_data['rating_error'] = "模型返回数据为空或格式错误"
                async with FILE_LOCK:
                    async with aiofiles.open(error_path, "a", encoding="utf-8") as f:
                        await f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
        except Exception as e:
            error_data = data.copy()
            error_data['rating_error'] = str(e)
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
    timeout_sec: int
) -> None:
    await process_single_question(data, session, semaphore, output_path, error_path, retries, timeout_sec)
    pbar.update(1)

def get_processed_question_ids(output_path: str) -> set:
    """实现断点续存，获取已处理的题目ID列表"""
    processed = set()
    if not os.path.exists(output_path):
        return processed
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
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

# -------------------------- 6. 主执行流 --------------------------
async def main_batch_run():
    parser = argparse.ArgumentParser(description="初中物理难度评级多线程并发批量打标脚本 (带 Cache 优化)")
    parser.add_argument("-p", "--prompt", type=str, default="../prompts/初中物理难度打标提示词.txt",
                        help="物理打标提示词文件路径")
    parser.add_argument("-i", "--input", type=str, default="../data/physics_sampled_5000_per_difficulty.jsonl",
                        help="输入待打标 JSONL 数据集路径")
    parser.add_argument("-o", "--output", type=str, default="physics_difficulty_rated_results.jsonl",
                        help="输出保存打标结果的 JSONL 路径")
    parser.add_argument("-e", "--error", type=str, default="physics_difficulty_errors.jsonl",
                        help="输出保存失败结果的 JSONL 路径")
    parser.add_argument("-c", "--concurrency", type=int, default=15,
                        help="最大并发限制，默认 15")
    parser.add_argument("-t", "--timeout", type=int, default=180,
                        help="单次 API 调用超时时间，默认 180 秒")
    parser.add_argument("-r", "--retries", type=int, default=3,
                        help="失败最大重试次数，默认 3")
    parser.add_argument("-n", "--num", type=int, default=None,
                        help="测试打标的限制数量（留空表示全部打标）")
    
    args = parser.parse_args()

    # 1. 动态加载并分配提示词缓存
    load_prompt_config(args.prompt)

    # 2. 读取题目列表
    if not os.path.exists(args.input):
        print(f"错误: 输入文件 {args.input} 不存在，终止运行！")
        sys.exit(1)

    print("正在加载待打标数据集...")
    questions = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    questions.append(json.loads(line))
                except Exception:
                    continue

    print(f"成功加载题目数据，共计 {len(questions)} 道题目。")

    # 是否限制打标数量（针对抽样大批量评测）
    if args.num is not None:
        questions = random.sample(questions, min(args.num, len(questions)))
        print(f"参数 -n 生效，随机抽样其中 {len(questions)} 道题进行测试。")
    else:
        # 全部跑测时，建议随机打乱顺序，使得线程处理更均匀
        random.shuffle(questions)
        print("全部打标启动：题目次序已随机打乱。")

    # 断点续传扫描
    processed_ids = get_processed_question_ids(args.output)
    to_process = [q for q in questions if q.get("question_id") not in processed_ids]
    print(f"数据比对完成: 已完成数 {len(processed_ids)}，待处理数 {len(to_process)}")

    if not to_process:
        print("所有题目都已完成打标！")
        return

    # 3. 启动并发
    semaphore = Semaphore(args.concurrency)
    pbar = tqdm(total=len(to_process), unit="item", desc="Batch Rating Progress")

    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        # 获取/新建前缀缓存
        await get_or_create_cache(session, args.retries, args.timeout)
        
        tasks = []
        for data in to_process:
            task = asyncio.create_task(
                process_with_progress(
                    data, 
                    session, 
                    semaphore, 
                    pbar, 
                    args.output, 
                    args.error, 
                    args.retries, 
                    args.timeout
                )
            )
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks)

    pbar.close()
    print("\n✨ 多线程批量打标运行结束！")
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
    print(f"本次打标运行耗时: {round((time.time() - start_time)/60, 2)} 分钟。")