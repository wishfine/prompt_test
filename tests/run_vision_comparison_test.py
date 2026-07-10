# -*- coding: utf-8 -*-
"""
@File    : run_vision_comparison_test_fixed.py
@Description:
    修复版：初中物理难度打标 V1 文本模式 vs V2 Vision-only 图片模式对比脚本。

主要修复点：
1. 按教师原始 difficulty 分层抽样，而不是按 V1 预测结果抽样；
2. 固定 random seed，并保存 sample_ids，保证实验可复现；
3. V2 使用 chat/completions 时兼容 prompt_tokens / completion_tokens；
4. 图片 URL 使用正则抽取，支持多 URL、中文逗号、空格、换行等情况；
5. V2 后处理直接复用 V1 主脚本中的 postprocess_physics_difficulty，避免两套规则漂移；
6. 可选择 V2 后处理是否允许读取原始文本：
   - image_only：默认，严格 Vision-only，不让后处理读取 stem/options/analysis；
   - text_available：复现实验旧逻辑，后处理可读取题库文本字段；
7. 增加分档准确率、≤1档比例、MAE、高难题召回、混淆矩阵、McNemar 对比等指标。
"""

import os
import re
import sys
import json
import time
import math
import random
import asyncio
import argparse
import importlib.util
from collections import defaultdict, Counter
from typing import Dict, Any, List, Optional, Tuple

import aiohttp
import aiofiles
import json_repair
from tqdm.asyncio import tqdm
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("API_KEY", "not-needed")
BASE_URL = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1")
if not BASE_URL.endswith("/"):
    BASE_URL += "/"
MODEL_NAME = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")


LEVEL_MAP = {
    "送分题": 1,
    "基础题": 2,
    "中等题": 3,
    "拔高题": 4,
    "压轴题": 5,
}

LEVEL_NAMES = {
    1: "送分题",
    2: "基础题",
    3: "中等题",
    4: "拔高题",
    5: "压轴题",
}

SAMPLE_PLAN = {
    "送分题": 10,
    "基础题": 12,
    "中等题": 12,
    "拔高题": 10,
    "压轴题": 6,
}


DIFFICULTY_RATING_PROMPT_PREFIX = ""
DIFFICULTY_RATING_PROMPT_SUFFIX = ""


# -------------------------- 1. 通用 IO 与配置 --------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                print(f"[WARN] JSONL 解析失败: {path}:{line_no}: {e}")
    return items


def load_prompt_config(prompt_path: str) -> None:
    """兼容纯文本提示词和 Python 变量提示词。"""
    global DIFFICULTY_RATING_PROMPT_PREFIX, DIFFICULTY_RATING_PROMPT_SUFFIX

    if not os.path.exists(prompt_path):
        raise FileNotFoundError(f"找不到提示词文件: {prompt_path}")

    with open(prompt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Python 变量结构
    try:
        namespace: Dict[str, Any] = {}
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

    # 纯文本结构
    if "## 输入题目信息" in content:
        parts = content.split("## 输入题目信息")
        DIFFICULTY_RATING_PROMPT_PREFIX = parts[0] + "## 输入题目信息"
        DIFFICULTY_RATING_PROMPT_SUFFIX = "\n\n请根据以上信息，对题目进行全面的难度分析和评级。"
        print("成功以纯文本标志位结构解析提示词")
        return

    raise ValueError("提示词格式不正确：既不是 Python 变量结构，也没有包含 '## 输入题目信息'。")


def load_v1_postprocess(v1_script_path: str):
    """动态导入 V1 主脚本中的 postprocess_physics_difficulty，避免复制后处理逻辑导致规则漂移。"""
    if not os.path.exists(v1_script_path):
        raise FileNotFoundError(f"找不到 V1 后处理脚本: {v1_script_path}")

    spec = importlib.util.spec_from_file_location("physics_v1_postprocess_module", v1_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法导入 V1 后处理脚本: {v1_script_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "postprocess_physics_difficulty"):
        raise AttributeError("V1 脚本中没有找到 postprocess_physics_difficulty 函数")

    return module.postprocess_physics_difficulty


def parse_model_response(response_text: str) -> Dict[str, Any]:
    """容错解析模型 JSON 输出。"""
    if not response_text:
        return {}

    try:
        obj = json_repair.loads(response_text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    try:
        clean_text = response_text
        if "```json" in clean_text:
            clean_text = clean_text.split("```json", 1)[1].split("```", 1)[0]
        elif "```" in clean_text:
            clean_text = clean_text.split("```", 1)[1].split("```", 1)[0]
        obj = json_repair.loads(clean_text.strip())
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    try:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            obj = json_repair.loads(response_text[start:end + 1])
            return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    return {}


def normalize_qid(qid: Any) -> str:
    return str(qid).strip()


def normalize_teacher_difficulty(value: Any) -> Optional[int]:
    """把教师 difficulty 统一成 1-5 的整数。"""
    if value is None:
        return None

    if isinstance(value, int):
        return value if value in LEVEL_NAMES else None

    s = str(value).strip()
    if s in LEVEL_MAP:
        return LEVEL_MAP[s]

    try:
        n = int(float(s))
        return n if n in LEVEL_NAMES else None
    except Exception:
        return None


def extract_http_image_urls(value: Any) -> List[str]:
    """从字段中抽取图片 URL，兼容字符串、list、多 URL 拼接、中文逗号、换行。"""
    if not value:
        return []

    if isinstance(value, list):
        urls: List[str] = []
        for x in value:
            urls.extend(extract_http_image_urls(x))
        return list(dict.fromkeys(urls))

    text = str(value).strip()
    # 先用正则取 http(s)，避免 "url1,url2" 被当成一个 URL。
    urls = re.findall(r"https?://[^\s,，]+", text)
    return list(dict.fromkeys([u.strip() for u in urls if u.strip()]))


def make_vision_postprocess_data(item: Dict[str, Any], mode: str) -> Dict[str, Any]:
    """构造 V2 后处理可见数据。

    image_only:
        严格 Vision-only，不让后处理读取原始 stem/options/analysis/sub_questions 文本；
        后处理只能基于模型输出的 features 做规则纠偏。
    text_available:
        与旧脚本一致，后处理可以读取题库中的文本字段；
        适合模拟生产中“图片+后台文本都可用”的情况，但不适合作为纯视觉能力评测。
    """
    if mode == "text_available":
        return item

    sanitized = item.copy()
    sanitized["stem"] = ""
    sanitized["options"] = ""
    sanitized["analysis"] = ""
    sanitized["sub_questions"] = []
    return sanitized


# -------------------------- 2. 抽样逻辑 --------------------------

def build_v1_index(v1_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}
    for item in v1_items:
        qid = normalize_qid(item.get("question_id", ""))
        if qid:
            index[qid] = item
    return index


def build_v2_index(v2_items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index = {}
    for item in v2_items:
        qid = normalize_qid(item.get("question_id", ""))
        if qid:
            index[qid] = item
    return index


def group_items_for_sampling(v1_items: List[Dict[str, Any]], sample_by: str) -> Dict[str, List[Dict[str, Any]]]:
    """按教师标签或 V1 预测标签分层。默认应使用 teacher。"""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for item in v1_items:
        if sample_by == "teacher":
            human_num = normalize_teacher_difficulty(item.get("difficulty"))
            if human_num is None:
                continue
            label = LEVEL_NAMES[human_num]
        else:
            rating = item.get("difficulty_rating", {})
            if not isinstance(rating, dict):
                continue
            label = rating.get("difficulty_level", "")

        if label in SAMPLE_PLAN:
            grouped[label].append(item)

    return grouped


async def save_sample_ids(path: str, sample_items: List[Dict[str, Any]], sample_by: str, seed: int) -> None:
    data = {
        "sample_by": sample_by,
        "seed": seed,
        "sample_plan": SAMPLE_PLAN,
        "question_ids": [normalize_qid(x.get("question_id")) for x in sample_items],
    }
    async with aiofiles.open(path, "w", encoding="utf-8") as f:
        await f.write(json.dumps(data, ensure_ascii=False, indent=2))


def load_sample_ids(path: str) -> Optional[List[str]]:
    if not path or not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ids = data.get("question_ids")
    if not isinstance(ids, list) or not ids:
        return None
    return [normalize_qid(x) for x in ids]


async def make_sample(
    v1_items: List[Dict[str, Any]],
    sample_ids_path: str,
    sample_by: str,
    seed: int,
    force_resample: bool,
) -> List[Dict[str, Any]]:
    v1_index = build_v1_index(v1_items)

    if sample_ids_path and os.path.exists(sample_ids_path) and not force_resample:
        qids = load_sample_ids(sample_ids_path)
        if qids:
            sampled = [v1_index[qid] for qid in qids if qid in v1_index]
            print(f"已从 sample_ids 文件复用样本: {len(sampled)} 道")
            return sampled

    random.seed(seed)
    grouped = group_items_for_sampling(v1_items, sample_by=sample_by)

    sampled: List[Dict[str, Any]] = []
    print(f"\n--- 抽样计划：sample_by={sample_by}, seed={seed} ---")
    for label, count in SAMPLE_PLAN.items():
        pool = grouped.get(label, [])
        if len(pool) < count:
            print(f"[WARN] {label} 池中数据不足 ({len(pool)} < {count})，已全部选取")
            picked = pool
        else:
            picked = random.sample(pool, count)
        sampled.extend(picked)
        print(f"  {label}: 计划 {count}, 实际 {len(picked)}")

    if sample_ids_path:
        await save_sample_ids(sample_ids_path, sampled, sample_by=sample_by, seed=seed)
        print(f"样本 question_id 已保存: {os.path.abspath(sample_ids_path)}")

    return sampled


# -------------------------- 3. V2 Vision-only 调用 --------------------------

async def rate_v2_only_images(
    data: Dict[str, Any],
    session: aiohttp.ClientSession,
    retries: int,
    timeout_sec: int,
) -> Tuple[Dict[str, Any], float, int, int, int, str]:
    """仅通过图片 URL + prompt 发送到 chat/completions，不传题干/解析文本。"""
    stem_urls = extract_http_image_urls(data.get("stem_pic_url", ""))
    analysis_urls = extract_http_image_urls(data.get("analysis_pic_url", ""))

    content_list: List[Dict[str, Any]] = [
        {"type": "text", "text": DIFFICULTY_RATING_PROMPT_PREFIX + "\n"},
        {"type": "text", "text": "【题干图示与解析图示如下，请仅根据图片中的内容进行难度评级。】\n"},
    ]

    for url in stem_urls:
        content_list.append({"type": "image_url", "image_url": {"url": url}})

    for url in analysis_urls:
        content_list.append({"type": "image_url", "image_url": {"url": url}})

    content_list.append({"type": "text", "text": DIFFICULTY_RATING_PROMPT_SUFFIX})

    payload = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content_list}],
        "thinking": {"type": "disabled"},
    }

    last_error = ""

    for retry in range(retries):
        t1 = time.time()
        try:
            async with session.post(
                f"{BASE_URL}chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    output_text = ""
                    try:
                        output_text = result["choices"][0]["message"]["content"]
                    except Exception:
                        output_text = ""

                    usage = result.get("usage", {}) or {}
                    # chat/completions 常见字段为 prompt_tokens / completion_tokens；
                    # responses 端点常见字段为 input_tokens / output_tokens。
                    prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
                    completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
                    total_tokens = usage.get("total_tokens", 0)

                    parsed_result = parse_model_response(output_text)
                    t2 = time.time()
                    return parsed_result, t2 - t1, prompt_tokens, completion_tokens, total_tokens, ""

                error_text = await response.text()
                last_error = f"HTTP {response.status}: {error_text[:500]}"

                if response.status == 429:
                    retry_after = int(response.headers.get("Retry-After", 5))
                    await asyncio.sleep(retry_after)
                    continue

                if response.status >= 500:
                    await asyncio.sleep((2 ** retry) + random.uniform(0, 1))
                    continue

                # 4xx 除 429 外通常不重试
                break

        except Exception as e:
            last_error = repr(e)
            await asyncio.sleep((2 ** retry) + random.uniform(0, 1))

    return {}, 0.0, 0, 0, 0, last_error or "unknown error"


# -------------------------- 4. 统计指标 --------------------------

def safe_mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def exact_accuracy(rows: List[Dict[str, Any]], key: str) -> float:
    valid = [r for r in rows if r.get(key) and r.get("human_num")]
    if not valid:
        return 0.0
    return sum(1 for r in valid if r[key] == r["human_num"]) / len(valid)


def within_one_rate(rows: List[Dict[str, Any]], key: str) -> float:
    valid = [r for r in rows if r.get(key) and r.get("human_num")]
    if not valid:
        return 0.0
    return sum(1 for r in valid if abs(r[key] - r["human_num"]) <= 1) / len(valid)


def mae(rows: List[Dict[str, Any]], key: str) -> float:
    valid = [r for r in rows if r.get(key) and r.get("human_num")]
    if not valid:
        return 0.0
    return sum(abs(r[key] - r["human_num"]) for r in valid) / len(valid)


def high_recall(rows: List[Dict[str, Any]], key: str) -> float:
    """教师为拔高/压轴时，模型是否也判为拔高/压轴。"""
    high = [r for r in rows if r.get("human_num", 0) >= 4 and r.get(key)]
    if not high:
        return 0.0
    return sum(1 for r in high if r[key] >= 4) / len(high)


def high_precision(rows: List[Dict[str, Any]], key: str) -> float:
    pred_high = [r for r in rows if r.get(key, 0) >= 4 and r.get("human_num")]
    if not pred_high:
        return 0.0
    return sum(1 for r in pred_high if r["human_num"] >= 4) / len(pred_high)


def mcnemar_exact_p(rows: List[Dict[str, Any]]) -> Tuple[int, int, float]:
    """返回 b, c, p。b=V1对V2错，c=V1错V2对。"""
    b = 0
    c = 0
    for r in rows:
        h = r.get("human_num")
        v1 = r.get("v1_num")
        v2 = r.get("v2_num")
        if not h or not v1 or not v2:
            continue
        v1_ok = (v1 == h)
        v2_ok = (v2 == h)
        if v1_ok and not v2_ok:
            b += 1
        elif (not v1_ok) and v2_ok:
            c += 1

    n = b + c
    if n == 0:
        return b, c, 1.0

    k = min(b, c)
    # 双侧 exact binomial p-value under p=0.5
    prob = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    p = min(1.0, 2 * prob)
    return b, c, p


def distribution(rows: List[Dict[str, Any]], key: str) -> Counter:
    c = Counter()
    for r in rows:
        num = r.get(key)
        if num in LEVEL_NAMES:
            c[LEVEL_NAMES[num]] += 1
    return c


def by_level_accuracy(rows: List[Dict[str, Any]], key: str) -> Dict[str, Tuple[int, int, float]]:
    result = {}
    for i in range(1, 6):
        subset = [r for r in rows if r.get("human_num") == i and r.get(key)]
        total = len(subset)
        correct = sum(1 for r in subset if r[key] == i)
        result[LEVEL_NAMES[i]] = (correct, total, correct / total if total else 0.0)
    return result


def confusion_matrix(rows: List[Dict[str, Any]], key: str) -> List[List[int]]:
    mat = [[0 for _ in range(5)] for _ in range(5)]
    for r in rows:
        h = r.get("human_num")
        p = r.get(key)
        if h in LEVEL_NAMES and p in LEVEL_NAMES:
            mat[h - 1][p - 1] += 1
    return mat


def md_distribution_table(title: str, dist: Counter) -> str:
    lines = [f"#### {title}", "", "| 难度 | 数量 |", "|---|---:|"]
    for i in range(1, 6):
        label = LEVEL_NAMES[i]
        lines.append(f"| {label} | {dist.get(label, 0)} |")
    return "\n".join(lines)


def md_accuracy_by_level(title: str, acc: Dict[str, Tuple[int, int, float]]) -> str:
    lines = [f"#### {title}", "", "| 教师难度 | 正确 / 总数 | 准确率 |", "|---|---:|---:|"]
    for i in range(1, 6):
        label = LEVEL_NAMES[i]
        correct, total, rate = acc[label]
        lines.append(f"| {label} | {correct} / {total} | {rate * 100:.1f}% |")
    return "\n".join(lines)


def md_confusion_matrix(title: str, mat: List[List[int]]) -> str:
    header = "| 教师 \\ 预测 | " + " | ".join(LEVEL_NAMES[i] for i in range(1, 6)) + " |"
    sep = "|---|" + "|".join(["---:"] * 5) + "|"
    lines = [f"#### {title}", "", header, sep]
    for i in range(5):
        lines.append("| " + LEVEL_NAMES[i + 1] + " | " + " | ".join(str(x) for x in mat[i]) + " |")
    return "\n".join(lines)


# -------------------------- 5. 主流程 --------------------------

async def async_run(args: argparse.Namespace) -> None:
    load_prompt_config(args.prompt)
    postprocess_physics_difficulty = load_v1_postprocess(args.v1_script)

    print("正在载入 V1 纯文本打标结果...")
    v1_items = read_jsonl(args.v1_rated)
    v1_index = build_v1_index(v1_items)
    print(f"V1 结果数: {len(v1_items)}, 唯一 question_id: {len(v1_index)}")

    print("正在载入 V2 带图片题库...")
    v2_items = read_jsonl(args.v2_source)
    v2_index = build_v2_index(v2_items)
    print(f"V2 题库数: {len(v2_items)}, 唯一 question_id: {len(v2_index)}")

    sampled_v1 = await make_sample(
        v1_items=v1_items,
        sample_ids_path=args.sample_ids,
        sample_by=args.sample_by,
        seed=args.seed,
        force_resample=args.force_resample,
    )

    test_queue: List[Dict[str, Any]] = []
    missing_in_v2 = []
    no_image = []

    for v1_item in sampled_v1:
        qid = normalize_qid(v1_item.get("question_id"))
        v2_item = v2_index.get(qid)
        if not v2_item:
            missing_in_v2.append(qid)
            continue

        stem_urls = extract_http_image_urls(v2_item.get("stem_pic_url", ""))
        analysis_urls = extract_http_image_urls(v2_item.get("analysis_pic_url", ""))
        if not stem_urls and not analysis_urls:
            no_image.append(qid)
            continue

        merged = v2_item.copy()
        merged["_v1_item"] = v1_item
        merged["_stem_image_count"] = len(stem_urls)
        merged["_analysis_image_count"] = len(analysis_urls)
        merged["_total_image_count"] = len(stem_urls) + len(analysis_urls)

        # V1 已经是主流程结果；默认不二次后处理，避免把旧 reasoning 反复污染。
        v1_rating = v1_item.get("difficulty_rating", {})
        if args.repostprocess_v1 and isinstance(v1_rating, dict):
            v1_rating = postprocess_physics_difficulty(json.loads(json.dumps(v1_rating, ensure_ascii=False)), v1_item)

        merged["_v1_rating"] = v1_rating
        merged["_v1_api_prompt_tokens"] = v1_item.get("api_prompt_tokens", 0)
        merged["_v1_api_completion_tokens"] = v1_item.get("api_completion_tokens", 0)
        merged["_v1_api_total_tokens"] = v1_item.get("api_total_tokens", 0)
        merged["_v1_api_time_use"] = v1_item.get("api_time_use", 0.0)

        test_queue.append(merged)

    if missing_in_v2:
        print(f"[WARN] {len(missing_in_v2)} 道样本在 V2 题库中缺失")
    if no_image:
        print(f"[WARN] {len(no_image)} 道样本没有任何图片 URL，已跳过")

    print(f"\n准备进行 V2 Vision-only 打标: {len(test_queue)} 道")
    if not test_queue:
        print("没有可测试样本，退出。")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)

    error_rows = []

    async with aiohttp.ClientSession(connector=connector) as session:
        async def worker(item: Dict[str, Any]) -> None:
            async with semaphore:
                res, time_use, in_tok, out_tok, total_tok, err = await rate_v2_only_images(
                    item, session, args.retries, args.timeout
                )

                if res:
                    postprocess_data = make_vision_postprocess_data(item, args.vision_postprocess_data_mode)
                    res = postprocess_physics_difficulty(res, postprocess_data)
                else:
                    error_rows.append({
                        "question_id": item.get("question_id"),
                        "error": err,
                        "stem_image_count": item.get("_stem_image_count", 0),
                        "analysis_image_count": item.get("_analysis_image_count", 0),
                    })

                item["_v2_rating"] = res
                item["_v2_api_prompt_tokens"] = in_tok
                item["_v2_api_completion_tokens"] = out_tok
                item["_v2_api_total_tokens"] = total_tok
                item["_v2_api_time_use"] = round(time_use, 2)
                item["_v2_error"] = err

        tasks = [asyncio.create_task(worker(item)) for item in test_queue]
        for task in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="V2 Vision Running"):
            await task

    if args.error_output and error_rows:
        async with aiofiles.open(args.error_output, "w", encoding="utf-8") as f:
            for row in error_rows:
                await f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"V2 错误记录已保存: {os.path.abspath(args.error_output)}")

    # -------------------------- 统计 --------------------------
    rows: List[Dict[str, Any]] = []
    for item in test_queue:
        human_num = normalize_teacher_difficulty(item.get("difficulty"))

        v1_rating = item.get("_v1_rating", {})
        v2_rating = item.get("_v2_rating", {})

        v1_str = v1_rating.get("difficulty_level", "") if isinstance(v1_rating, dict) else ""
        v2_str = v2_rating.get("difficulty_level", "") if isinstance(v2_rating, dict) else ""

        row = {
            "qid": normalize_qid(item.get("question_id")),
            "human_num": human_num,
            "human": LEVEL_NAMES.get(human_num, str(item.get("difficulty"))),
            "v1": v1_str,
            "v2": v2_str,
            "v1_num": LEVEL_MAP.get(v1_str),
            "v2_num": LEVEL_MAP.get(v2_str),
            "v1_prompt_tokens": int(item.get("_v1_api_prompt_tokens", 0) or 0),
            "v1_completion_tokens": int(item.get("_v1_api_completion_tokens", 0) or 0),
            "v1_total_tokens": int(item.get("_v1_api_total_tokens", 0) or 0),
            "v2_prompt_tokens": int(item.get("_v2_api_prompt_tokens", 0) or 0),
            "v2_completion_tokens": int(item.get("_v2_api_completion_tokens", 0) or 0),
            "v2_total_tokens": int(item.get("_v2_api_total_tokens", 0) or 0),
            "v1_time": float(item.get("_v1_api_time_use", 0.0) or 0.0),
            "v2_time": float(item.get("_v2_api_time_use", 0.0) or 0.0),
            "stem_image_count": item.get("_stem_image_count", 0),
            "analysis_image_count": item.get("_analysis_image_count", 0),
            "total_image_count": item.get("_total_image_count", 0),
        }
        rows.append(row)

    valid_rows = [r for r in rows if r.get("human_num") and r.get("v1_num") and r.get("v2_num")]
    valid_count = len(valid_rows)

    if valid_count == 0:
        print("没有 V1/V2/教师标签均有效的样本。")
        return

    v1_acc = exact_accuracy(valid_rows, "v1_num")
    v2_acc = exact_accuracy(valid_rows, "v2_num")
    v1_within1 = within_one_rate(valid_rows, "v1_num")
    v2_within1 = within_one_rate(valid_rows, "v2_num")
    v1_mae = mae(valid_rows, "v1_num")
    v2_mae = mae(valid_rows, "v2_num")
    v1_high_recall = high_recall(valid_rows, "v1_num")
    v2_high_recall = high_recall(valid_rows, "v2_num")
    v1_high_precision = high_precision(valid_rows, "v1_num")
    v2_high_precision = high_precision(valid_rows, "v2_num")

    agree = sum(1 for r in valid_rows if r["v1_num"] == r["v2_num"])
    agree_rate = agree / valid_count

    b, c, p = mcnemar_exact_p(valid_rows)

    v1_prompt_avg = safe_mean([r["v1_prompt_tokens"] for r in valid_rows])
    v1_completion_avg = safe_mean([r["v1_completion_tokens"] for r in valid_rows])
    v1_total_avg = safe_mean([r["v1_total_tokens"] for r in valid_rows])
    v2_prompt_avg = safe_mean([r["v2_prompt_tokens"] for r in valid_rows])
    v2_completion_avg = safe_mean([r["v2_completion_tokens"] for r in valid_rows])
    v2_total_avg = safe_mean([r["v2_total_tokens"] for r in valid_rows])
    v1_time_avg = safe_mean([r["v1_time"] for r in valid_rows])
    v2_time_avg = safe_mean([r["v2_time"] for r in valid_rows])
    image_avg = safe_mean([r["total_image_count"] for r in valid_rows])

    v1_logical_cost_ratio = (v2_total_avg - v1_total_avg) / v1_total_avg if v1_total_avg else 0.0

    # V1 使用 responses 前缀缓存时，api_total_tokens 是逻辑 token；
    # 可以用一个估算值展示实际物理计费差异，默认 800，可通过参数改。
    v1_effective_avg = args.v1_effective_token_estimate
    v2_vs_v1_effective_ratio = v2_total_avg / v1_effective_avg if v1_effective_avg else 0.0

    # -------------------------- 生成 Markdown 报告 --------------------------
    md = []
    md.append("# 初中物理难度打标 V1 文本 vs V2 Vision-only 对比测试报告")
    md.append("")
    md.append("## 1. 实验设置")
    md.append("")
    md.append(f"- 模型：`{MODEL_NAME}`")
    md.append(f"- 有效样本数：**{valid_count}**")
    md.append(f"- 抽样依据：`{args.sample_by}`")
    md.append(f"- 随机种子：`{args.seed}`")
    md.append(f"- V2 后处理可见数据模式：`{args.vision_postprocess_data_mode}`")
    md.append(f"- V2 平均图片数：**{image_avg:.2f} 张/题**")
    md.append("")
    md.append("> `image_only` 表示 V2 后处理阶段不读取题库原始 stem/options/analysis 文本，更符合纯图片能力评测；`text_available` 表示复现实验旧逻辑，后处理也可以读取题库文本字段。")
    md.append("")

    md.append("## 2. 核心结果摘要")
    md.append("")
    md.append("| 指标 | V1 文本 | V2 纯图片 | 变化 |")
    md.append("|---|---:|---:|---:|")
    md.append(f"| 教师 exact 对齐率 | {v1_acc * 100:.1f}% | {v2_acc * 100:.1f}% | {(v2_acc - v1_acc) * 100:+.1f} pct |")
    md.append(f"| 相差 ≤1 档比例 | {v1_within1 * 100:.1f}% | {v2_within1 * 100:.1f}% | {(v2_within1 - v1_within1) * 100:+.1f} pct |")
    md.append(f"| MAE | {v1_mae:.3f} | {v2_mae:.3f} | {v2_mae - v1_mae:+.3f} |")
    md.append(f"| 高难题召回率（教师≥拔高） | {v1_high_recall * 100:.1f}% | {v2_high_recall * 100:.1f}% | {(v2_high_recall - v1_high_recall) * 100:+.1f} pct |")
    md.append(f"| 高难题精确率（预测≥拔高） | {v1_high_precision * 100:.1f}% | {v2_high_precision * 100:.1f}% | {(v2_high_precision - v1_high_precision) * 100:+.1f} pct |")
    md.append(f"| V1/V2 打标一致率 | - | - | {agree_rate * 100:.1f}% |")
    md.append(f"| 平均输入 Token | {v1_prompt_avg:.1f} | {v2_prompt_avg:.1f} | {((v2_prompt_avg - v1_prompt_avg) / v1_prompt_avg * 100) if v1_prompt_avg else 0.0:+.1f}% |")
    md.append(f"| 平均输出 Token | {v1_completion_avg:.1f} | {v2_completion_avg:.1f} | {((v2_completion_avg - v1_completion_avg) / v1_completion_avg * 100) if v1_completion_avg else 0.0:+.1f}% |")
    md.append(f"| 平均总 Token（逻辑） | {v1_total_avg:.1f} | {v2_total_avg:.1f} | {v1_logical_cost_ratio * 100:+.1f}% |")
    md.append(f"| 平均耗时 | {v1_time_avg:.2f}s | {v2_time_avg:.2f}s | {((v2_time_avg - v1_time_avg) / v1_time_avg * 100) if v1_time_avg else 0.0:+.1f}% |")
    md.append("")
    md.append(f"- McNemar：V1 对 V2 错 = **{b}**，V1 错 V2 对 = **{c}**，exact p ≈ **{p:.4f}**。")
    md.append(f"- 若按 V1 前缀缓存后实际输入约 `{v1_effective_avg:.0f}` tokens/题估算，V2 / V1 实际 token 成本约为 **{v2_vs_v1_effective_ratio:.1f} 倍**。")
    md.append("")

    md.append("## 3. 分布对比")
    md.append("")
    md.append(md_distribution_table("教师标签分布", distribution(valid_rows, "human_num")))
    md.append("")
    md.append(md_distribution_table("V1 预测分布", distribution(valid_rows, "v1_num")))
    md.append("")
    md.append(md_distribution_table("V2 预测分布", distribution(valid_rows, "v2_num")))
    md.append("")

    md.append("## 4. 按教师难度分组准确率")
    md.append("")
    md.append(md_accuracy_by_level("V1 文本分组准确率", by_level_accuracy(valid_rows, "v1_num")))
    md.append("")
    md.append(md_accuracy_by_level("V2 纯图片分组准确率", by_level_accuracy(valid_rows, "v2_num")))
    md.append("")

    md.append("## 5. 混淆矩阵")
    md.append("")
    md.append(md_confusion_matrix("V1 文本混淆矩阵", confusion_matrix(valid_rows, "v1_num")))
    md.append("")
    md.append(md_confusion_matrix("V2 纯图片混淆矩阵", confusion_matrix(valid_rows, "v2_num")))
    md.append("")

    md.append("## 6. 逐题对比")
    md.append("")
    md.append("| 题目 ID | 教师 | V1 | V2 | V1是否对 | V2是否对 | V1/V2一致 | 图片数 | V1 Token | V2 Token | V1耗时 | V2耗时 |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in valid_rows:
        v1_ok = "✅" if r["v1_num"] == r["human_num"] else "❌"
        v2_ok = "✅" if r["v2_num"] == r["human_num"] else "❌"
        agree_icon = "✅" if r["v1_num"] == r["v2_num"] else "❌"
        md.append(
            f"| `{r['qid']}` | {r['human']} | {r['v1']} | {r['v2']} | "
            f"{v1_ok} | {v2_ok} | {agree_icon} | {r['total_image_count']} | "
            f"{r['v1_total_tokens']} | {r['v2_total_tokens']} | {r['v1_time']:.2f}s | {r['v2_time']:.2f}s |"
        )

    async with aiofiles.open(args.output, "w", encoding="utf-8") as f:
        await f.write("\n".join(md))

    # 保存机器可读结果
    if args.output_jsonl:
        async with aiofiles.open(args.output_jsonl, "w", encoding="utf-8") as f:
            for r in rows:
                await f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("初中物理 V1 文本 vs V2 Vision-only 对比测试摘要")
    print("=" * 70)
    print(f"有效样本数: {valid_count}")
    print(f"V1 教师 exact 对齐率: {v1_acc * 100:.1f}%")
    print(f"V2 教师 exact 对齐率: {v2_acc * 100:.1f}%")
    print(f"V1/V2 一致率: {agree_rate * 100:.1f}%")
    print(f"V1 MAE: {v1_mae:.3f} | V2 MAE: {v2_mae:.3f}")
    print(f"V1 高难题召回: {v1_high_recall * 100:.1f}% | V2 高难题召回: {v2_high_recall * 100:.1f}%")
    print(f"V1 总 Token 均值: {v1_total_avg:.1f} | V2 总 Token 均值: {v2_total_avg:.1f}")
    print(f"V2 / V1 实际成本估计: {v2_vs_v1_effective_ratio:.1f}x")
    print(f"报告已保存: {os.path.abspath(args.output)}")
    if args.output_jsonl:
        print(f"逐题 JSONL 已保存: {os.path.abspath(args.output_jsonl)}")
    print("=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="修复版：初中物理 V1 文本 vs V2 Vision-only 对比实验脚本")

    parser.add_argument("--prompt", type=str, default="../prompts/初中物理难度打标提示词.txt",
                        help="提示词文件路径")
    parser.add_argument("--v1-rated", type=str, default="physics_difficulty_rated_results.jsonl",
                        help="V1 纯文本打标结果 JSONL")
    parser.add_argument("--v2-source", type=str, default="../data/physics_sampled_5000_per_difficulty_v2.jsonl",
                        help="V2 带图片 URL 的源题库 JSONL")
    parser.add_argument("--v1-script", type=str, default="../src/physics_difficulty_rating_with_cache.py",
                        help="V1 主脚本路径，用于导入 postprocess_physics_difficulty")
    parser.add_argument("--output", type=str, default="physics_vision_comparison_report_fixed.md",
                        help="Markdown 报告输出路径")
    parser.add_argument("--output-jsonl", type=str, default="physics_vision_comparison_details_fixed.jsonl",
                        help="逐题对比 JSONL 输出路径")
    parser.add_argument("--error-output", type=str, default="physics_vision_comparison_errors_fixed.jsonl",
                        help="V2 调用失败记录 JSONL 输出路径")

    parser.add_argument("--sample-ids", type=str, default="physics_vision_comparison_sample_ids.json",
                        help="样本 ID 文件；存在时默认复用，保证复现")
    parser.add_argument("--force-resample", action="store_true",
                        help="忽略已有 sample_ids，重新按 seed 抽样")
    parser.add_argument("--sample-by", type=str, choices=["teacher", "v1_pred"], default="teacher",
                        help="分层抽样依据。严谨评测应使用 teacher；v1_pred 仅用于复现旧实验")
    parser.add_argument("--seed", type=int, default=20260707,
                        help="随机种子")

    parser.add_argument("--vision-postprocess-data-mode", type=str,
                        choices=["image_only", "text_available"],
                        default="image_only",
                        help="V2 后处理是否能读取原始文本。纯视觉评测用 image_only；复现旧逻辑用 text_available")
    parser.add_argument("--repostprocess-v1", action="store_true",
                        help="是否对 V1 已有 difficulty_rating 再用当前 postprocess 处理一次。默认不启用")

    parser.add_argument("--concurrency", type=int, default=5,
                        help="V2 Vision 并发数")
    parser.add_argument("--timeout", type=int, default=180,
                        help="单次 V2 API 调用超时秒数")
    parser.add_argument("--retries", type=int, default=3,
                        help="V2 API 调用重试次数")
    parser.add_argument("--v1-effective-token-estimate", type=float, default=800.0,
                        help="V1 前缀缓存后实际计费 token 的估算均值，用于成本倍率估算")

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    try:
        asyncio.run(async_run(args))
    except KeyboardInterrupt:
        print("\n收到键盘中断，已退出。")
    except Exception as e:
        print(f"\n运行失败: {repr(e)}")
        raise
