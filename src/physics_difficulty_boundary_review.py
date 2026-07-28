# -*- coding: utf-8 -*-
"""冻结首轮评级之后的相邻边界证据复核。

默认只审计、不写回。第二次 API 只能在一个指定的相邻边界中选择，并使用
可核验原文证据；图片采用窄范围条件路由，避免把“有图”本身当作难度信号。
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
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import aiofiles
import aiohttp
from dotenv import load_dotenv
from tqdm.asyncio import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import physics_difficulty_rating_with_cache as rating  # noqa: E402


load_dotenv()

PIPELINE_VERSION = "adjacent-boundary-review-v1"
DEFAULT_PROMPT = ROOT / "prompts" / "初中物理相邻边界复核提示词.txt"
LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
OUTPUT_LOCK = Lock()

QUESTION_FIELDS = ("stem", "options", "analysis", "sub_questions")
IMAGE_FIELD_NAMES = {
    "stem_pic_url",
    "analysis_pic_url",
    "stem_image_url",
    "analysis_image_url",
    "image_url",
    "image_urls",
    "pic_url",
    "pic_urls",
}
SOURCE_FIELDS = set(QUESTION_FIELDS)
BOUNDARY_STATUS = {"明确归档", "相邻边界均可"}
CONFIDENCES = {"高", "中", "低"}
POSTPROCESS_RULE_REVIEWS = {"not_applicable", "valid", "invalid"}

BOUNDARY_RUBRICS: Dict[Tuple[str, str], str] = {
    ("送分题", "基础题"): """
下档【送分题】：只有一个唯一、熟悉、完全显性的教材模板，不需要第二次物理决策；一步直接识别、直接读数、一步整数计算或单一标准动作仍可属于送分。
上档【基础题】：需要把题目条件映射到规律、完成一次公式应用、规范操作/作图，或在两个独立基础结论间作辨析；路径仍完全显性。
决定边界：是否出现第二次独立物理决策，或是否需要从情境映射到规律，而不只是识别唯一模板。
""".strip(),
    ("基础题", "中等题"): """
下档【基础题】：最高难任务通常只有1—2个有效决策；独立选项分别调用通用教材结论，或单模型只沿一条显性关系链得到一个结果。
上档【中等题】：单个任务形成约3—4个前后依赖的有效决策，或多个任务共享题目特有中间结论、同一连续过程、完整实验流程、充分必要条件/反例辨析。
决定边界：必须指出连续依赖或共享结构；“选项多、知识点多、逐项判断、工作量大”均不是中等依据。
""".strip(),
    ("中等题", "拔高题"): """
下档【中等题】：常规模型明确，约3—4个有效决策；图像、实验或公式只是常规读取和串联，没有决定性隐藏转换。
上档【拔高题】：存在决定性模型转换（隐含条件、等效替代、图线反推、几何转化、关键操作顺序、误差方向、临界筛选），或形成约5—6个有效决策的高密度完整链。
决定边界：不是高阶词语数量，而是学生是否必须完成不可省略的迁移/转换，或一条真实完整的5—6步综合链。
""".strip(),
    ("拔高题", "压轴题"): """
下档【拔高题】：有明显卡点、决定性转换或5—6步完整综合链，但仍可在单一主模型或有限状态中完成。
上档【压轴题】：通常实际7步以上；多对象、多状态、多约束形成依赖网络，并实际执行分类讨论、多解筛选、临界极值、边界覆盖、复杂多图反推、有效解筛选或开放方案可行性验证。
决定边界：必须存在全链耦合和至少一项强压轴操作；项目、传感器、量程、最大值或题目位置不能单独支持压轴。
""".strip(),
}


def extract_final_level(item: Dict[str, Any]) -> Optional[str]:
    difficulty_rating = item.get("difficulty_rating")
    if isinstance(difficulty_rating, dict):
        level = difficulty_rating.get("difficulty_level")
        if level in LEVEL_INDEX:
            return str(level)
    level = item.get("difficulty_level")
    if level in LEVEL_INDEX:
        return str(level)
    return None


def extract_rating(item: Dict[str, Any]) -> Dict[str, Any]:
    value = item.get("difficulty_rating")
    return value if isinstance(value, dict) else {}


def extract_features(item: Dict[str, Any]) -> Dict[str, str]:
    value = extract_rating(item).get("features")
    return rating.normalize_features(value)


def extract_actions(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions = item.get("postprocess_actions")
    if not isinstance(actions, list):
        actions = extract_rating(item).get("postprocess_actions")
    return [dict(value) for value in actions or [] if isinstance(value, dict)]


def adjacent_pair(first: str, second: str) -> Optional[Tuple[str, str]]:
    if first not in LEVEL_INDEX or second not in LEVEL_INDEX:
        return None
    values = sorted((first, second), key=LEVEL_INDEX.get)
    if LEVEL_INDEX[values[1]] - LEVEL_INDEX[values[0]] != 1:
        return None
    return values[0], values[1]


def _feature_is_low_structure(features: Dict[str, str]) -> bool:
    return bool(
        features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("reasoning_chain") in {"直接套用", "简单因果推理"}
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )


def _feature_supports_medium(features: Dict[str, str]) -> bool:
    strong = sum(
        [
            features.get("step_count") == "3-5步",
            features.get("reasoning_chain") == "多层因果推理",
            features.get("experiment_requirement") == "控制变量或故障分析",
            features.get("graph_table_requirement") == "多组比较归纳",
            features.get("subquestion_dependency") == "多问且层层递进",
            features.get("state_count") in {"双状态", "多状态"},
        ]
    )
    return strong >= 2


def _feature_supports_hard(features: Dict[str, str]) -> bool:
    decisive = sum(
        [
            features.get("step_count") in {"6-8步", "9-12步", "12步以上"},
            features.get("reasoning_chain") == "逆向推理或临界分析",
            features.get("calculation_complexity") in {"多公式联立", "复杂方程或范围计算"},
            features.get("experiment_requirement") == "方案设计或误差评价",
            features.get("graph_table_requirement") == "图像反推或外推",
            features.get("constraint_count") == "多约束",
            features.get("variable_relation") == "多变量耦合关系",
        ]
    )
    return decisive >= 2


def _feature_supports_final(features: Dict[str, str]) -> bool:
    high = sum(
        [
            features.get("step_count") in {"9-12步", "12步以上"},
            features.get("state_count") in {"多状态", "连续变化或临界状态"},
            features.get("constraint_count") == "多约束",
            features.get("variable_relation") == "多变量耦合关系",
            features.get("calculation_complexity") == "复杂方程或范围计算",
            features.get("information_carrier") == "多图表综合",
        ]
    )
    return high >= 4


def route_boundary_review(item: Dict[str, Any], scope: str = "risk") -> Dict[str, Any]:
    """确定唯一相邻边界；只依赖首轮输出，不读取任何真值标签。"""
    current = extract_final_level(item)
    if current not in LEVEL_INDEX:
        raise ValueError("首轮结果缺少合法最终等级")
    features = extract_features(item)
    actions = extract_actions(item)
    reasons: List[str] = []
    pair: Optional[Tuple[str, str]] = None

    for action in reversed(actions):
        candidate = adjacent_pair(str(action.get("from") or ""), str(action.get("to") or ""))
        if candidate and current in candidate:
            pair = candidate
            reasons.append(f"首轮后处理在{candidate[0]}/{candidate[1]}边界调整")
            break

    if pair is None:
        if current == "送分题":
            pair = ("送分题", "基础题")
            reasons.append("端点档送分题")
        elif current == "压轴题":
            pair = ("拔高题", "压轴题")
            reasons.append("端点档压轴题")
        elif current == "基础题":
            if _feature_is_low_structure(features):
                pair = ("送分题", "基础题")
                reasons.append("基础题呈现低结构唯一模板特征")
            elif _feature_supports_medium(features):
                pair = ("基础题", "中等题")
                reasons.append("基础题具有连续分析候选信号")
        elif current == "中等题":
            if _feature_is_low_structure(features):
                pair = ("基础题", "中等题")
                reasons.append("中等题与低结构特征冲突")
            elif _feature_supports_hard(features):
                pair = ("中等题", "拔高题")
                reasons.append("中等题具有决定性转换或高密度链候选信号")
        elif current == "拔高题":
            if _feature_supports_final(features):
                pair = ("拔高题", "压轴题")
                reasons.append("拔高题具有全链耦合候选信号")
            elif not _feature_supports_hard(features):
                pair = ("中等题", "拔高题")
                reasons.append("拔高题的决定性转换证据偏弱")

    if scope == "all" and pair is None:
        pair = {
            "基础题": ("基础题", "中等题"),
            "中等题": ("基础题", "中等题"),
            "拔高题": ("中等题", "拔高题"),
        }.get(current)
        if pair:
            reasons.append("全量相邻边界复核")

    selected = pair is not None and (scope == "all" or bool(reasons))
    return {
        "selected": selected,
        "current_level": current,
        "review_boundary": list(pair) if pair else [],
        "reasons": reasons,
        "scope": scope,
    }


def _walk_image_values(value: Any, key: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            yield from _walk_image_values(child_value, str(child_key))
    elif isinstance(value, list):
        for child in value:
            yield from _walk_image_values(child, key)
    elif key in IMAGE_FIELD_NAMES or any(token in key.lower() for token in ("image", "pic")):
        for url in re.findall(r"https?://[^\s,，;；\"']+", str(value or "")):
            yield url.rstrip(")]}。")


def collect_image_urls(item: Dict[str, Any]) -> List[str]:
    values: List[str] = []
    seen: set[str] = set()
    for url in _walk_image_values(item):
        if url not in seen:
            seen.add(url)
            values.append(url)
    return values


def _question_text(item: Dict[str, Any]) -> str:
    parts: List[str] = []
    for field in QUESTION_FIELDS:
        value = item.get(field)
        if isinstance(value, (dict, list)):
            parts.append(json.dumps(value, ensure_ascii=False))
        else:
            parts.append(str(value or ""))
    return "\n".join(parts)


def route_images(item: Dict[str, Any], image_mode: str) -> Dict[str, Any]:
    """窄路由：仅发送可能改变结构判断的图片，不因“如图”单独触发。"""
    urls = collect_image_urls(item)
    features = extract_features(item)
    text = _question_text(item)
    reasons: List[str] = []

    if image_mode == "all" and urls:
        reasons.append("image_mode=all")
    elif image_mode == "auto" and urls:
        # 题库常同时保存题干图和解析图，URL 数量本身不代表多图推理。
        multi_panel = bool(re.search(r"图[甲乙丙丁].*图[甲乙丙丁]|多图|两幅图|三幅图", text, re.S))
        graph_relation = bool(
            re.search(r"曲线|交点|斜率|非线性|I[-－—]?U|U[-－—]?I|R[-－—]?F|图像.*关系|图线", text, re.I)
            and features.get("information_carrier") in {"图像或表格", "多图表综合"}
        )
        circuit_markers = sum(
            marker in text
            for marker in ("开关", "滑动变阻器", "电流表", "电压表", "热敏", "压敏", "光敏", "继电器")
        )
        circuit_topology = (
            features.get("information_carrier") == "电路图" and circuit_markers >= 2
        )
        spatial_visual = bool(
            re.search(r"作图|画出|标出|作用点|力臂|光路|折射|反射|像的位置|空间位置", text)
            and features.get("information_carrier") in {"单图识别", "实验装置图", "多图表综合"}
        )
        unresolved_placeholder = bool(
            re.search(r"<image>|【图片】|见图|如下图", str(item.get("stem") or ""), re.I)
            and len(str(item.get("analysis") or "").strip()) < 80
        )
        feature_requires_graph = bool(
            features.get("graph_table_requirement") == "图像反推或外推"
            or features.get("information_carrier") == "多图表综合"
        )
        if multi_panel:
            reasons.append("多图或多面板关系")
        if graph_relation:
            reasons.append("曲线关系可能影响边界判断")
        if circuit_topology:
            reasons.append("复杂电路拓扑可能影响状态/约束判断")
        if spatial_visual:
            reasons.append("空间、光路、受力或规范作图依赖图形")
        if unresolved_placeholder:
            reasons.append("题干图像占位且解析未完整文字化")
        if feature_requires_graph:
            reasons.append("首轮特征声明需要图像反推或多图综合")

    included = bool(urls and reasons and image_mode != "off")
    return {
        "mode": image_mode,
        "image_available": bool(urls),
        "image_required": included,
        "image_included": included,
        "reasons": reasons,
        "selected_urls": urls[:4] if included else [],
        "available_url_count": len(urls),
    }


def build_review_content(item: Dict[str, Any], route: Dict[str, Any], image_route: Dict[str, Any]) -> str:
    boundary = tuple(route.get("review_boundary") or [])
    if boundary not in BOUNDARY_RUBRICS:
        raise ValueError("缺少合法相邻边界")
    safe_question = {
        field: item.get(field)
        for field in QUESTION_FIELDS
        if item.get(field) not in (None, "", [], {})
    }
    first_rating = extract_rating(item)
    payload = {
        "review_boundary": list(boundary),
        "boundary_rubric": BOUNDARY_RUBRICS[boundary],
        "question": safe_question,
        "first_stage": {
            "current_level": route["current_level"],
            "features": extract_features(item),
            "reasoning": first_rating.get("reasoning") or {},
            "postprocess_actions": extract_actions(item),
        },
        "risk_route": {"reasons": route.get("reasons") or []},
        "image_input": {
            "image_included": image_route["image_included"],
            "routing_reasons": image_route["reasons"],
        },
        "allowed_feature_values": {
            field: sorted(values)
            for field, values in rating.ALLOWED_FEATURE_VALUES.items()
        },
    }
    return (
        "【相邻边界复核输入】\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n只输出复核 JSON。"
    )


def canonical_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    return {"high": "高", "medium": "中", "mid": "中", "low": "低"}.get(text, str(value or "").strip())


def normalize_review(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    acceptable = source.get("acceptable_levels")
    evidence = source.get("decisive_evidence")
    corrections = source.get("feature_corrections")
    invalidated = source.get("invalidated_rules")
    image_evidence = source.get("new_image_evidence")
    return {
        "review_boundary": str(source.get("review_boundary") or "").strip(),
        "current_level": str(source.get("current_level") or "").strip(),
        "recommended_level": str(source.get("recommended_level") or "").strip(),
        "boundary_status": str(source.get("boundary_status") or "").strip(),
        "acceptable_levels": [str(v).strip() for v in acceptable or [] if str(v).strip()],
        "confidence": canonical_confidence(source.get("confidence")),
        "effective_decision_count": source.get("effective_decision_count"),
        "has_structural_revision": source.get("has_structural_revision"),
        "feature_corrections": dict(corrections) if isinstance(corrections, dict) else {},
        "decisive_evidence": [dict(v) for v in evidence or [] if isinstance(v, dict)],
        "postprocess_rule_review": str(source.get("postprocess_rule_review") or "").strip(),
        "invalidated_rules": [str(v).strip() for v in invalidated or [] if str(v).strip()],
        "image_reviewed": source.get("image_reviewed"),
        "image_adds_new_evidence": source.get("image_adds_new_evidence"),
        "new_image_evidence": [str(v).strip() for v in image_evidence or [] if str(v).strip()],
        "reason": str(source.get("reason") or "").strip(),
    }


def validate_review(review: Dict[str, Any], route: Dict[str, Any], image_route: Dict[str, Any]) -> Optional[str]:
    boundary = list(route.get("review_boundary") or [])
    if len(boundary) != 2:
        return "路由边界非法"
    if review.get("review_boundary") not in {"|".join(boundary), "/".join(boundary), "、".join(boundary)}:
        return "review_boundary 与指定边界不一致"
    if review.get("current_level") != route.get("current_level"):
        return "current_level 与首轮最终等级不一致"
    if review.get("recommended_level") not in boundary:
        return "recommended_level 超出指定相邻边界"
    if review.get("boundary_status") not in BOUNDARY_STATUS:
        return "boundary_status 非法"
    acceptable = review.get("acceptable_levels")
    if not isinstance(acceptable, list) or not acceptable or any(level not in boundary for level in acceptable):
        return "acceptable_levels 非法"
    if review["recommended_level"] not in acceptable:
        return "acceptable_levels 未包含 recommended_level"
    if review["boundary_status"] == "明确归档" and len(set(acceptable)) != 1:
        return "明确归档时只能有一个可接受等级"
    if review["boundary_status"] == "相邻边界均可" and set(acceptable) != set(boundary):
        return "相邻边界均可时必须列出两档"
    if review.get("confidence") not in CONFIDENCES:
        return "confidence 非法"
    count = review.get("effective_decision_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        return "effective_decision_count 必须是非负整数"
    if not isinstance(review.get("has_structural_revision"), bool):
        return "has_structural_revision 必须是布尔值"
    if not isinstance(review.get("image_reviewed"), bool) or not isinstance(review.get("image_adds_new_evidence"), bool):
        return "图片复核字段必须是布尔值"
    if review["image_reviewed"] and not image_route.get("image_included"):
        return "未发送图片却声明已复核图片"
    if review["image_adds_new_evidence"] and not review.get("new_image_evidence"):
        return "图片带来新证据时必须列出 new_image_evidence"
    if review.get("postprocess_rule_review") not in POSTPROCESS_RULE_REVIEWS:
        return "postprocess_rule_review 非法"
    for field, value in review.get("feature_corrections", {}).items():
        if field not in rating.ALLOWED_FEATURE_VALUES or value not in rating.ALLOWED_FEATURE_VALUES[field]:
            return f"非法 feature_correction: {field}={value}"
    if not review.get("reason"):
        return "缺少 reason"
    return None


def _normalized_text(value: Any) -> str:
    return "".join(str(value or "").split())


def _source_text(item: Dict[str, Any], field: str) -> str:
    value = item.get(field)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def verified_evidence(review: Dict[str, Any], item: Dict[str, Any]) -> List[Dict[str, Any]]:
    values: List[Dict[str, Any]] = []
    for evidence in review.get("decisive_evidence") or []:
        field = str(evidence.get("source_field") or "")
        excerpt = str(evidence.get("source_excerpt") or "").strip()
        finding = str(evidence.get("finding") or "").strip()
        if field not in SOURCE_FIELDS or not excerpt or not finding:
            continue
        if _normalized_text(excerpt) not in _normalized_text(_source_text(item, field)):
            continue
        values.append(evidence)
    return values


def should_apply_review(
    item: Dict[str, Any],
    route: Dict[str, Any],
    review: Dict[str, Any],
) -> Tuple[bool, str]:
    current = route["current_level"]
    target = review.get("recommended_level")
    if target == current:
        return False, "复核保留首轮等级"
    if abs(LEVEL_INDEX[target] - LEVEL_INDEX[current]) != 1:
        return False, "建议调整不是相邻一档"
    if review.get("boundary_status") != "明确归档":
        return False, "复核认为两档均可接受"
    if review.get("confidence") != "高":
        return False, "置信度未达到高"
    if review.get("has_structural_revision") is not True:
        return False, "没有指出首轮结构判断错误"
    evidence = verified_evidence(review, item)
    if len(evidence) < 2:
        return False, "不足两条可由题目原文核验的决定性证据"
    if not review.get("feature_corrections"):
        return False, "缺少影响边界的 feature 修正"

    actions = extract_actions(item)
    if actions:
        actual_rules = {str(action.get("rule") or "") for action in actions}
        invalidated = set(review.get("invalidated_rules") or [])
        if review.get("postprocess_rule_review") != "invalid" or not (actual_rules & invalidated):
            return False, "首轮发生后处理时，缺少对应规则前提失效证据"
    return True, "高置信相邻边界结论具备两条原文证据和结构修正"


def apply_review(item: Dict[str, Any], review: Dict[str, Any]) -> None:
    difficulty_rating = extract_rating(item)
    if not difficulty_rating:
        raise ValueError("缺少 difficulty_rating")
    item["difficulty_rating_before_boundary_review"] = copy.deepcopy(difficulty_rating)
    current = str(difficulty_rating.get("difficulty_level") or "")
    target = review["recommended_level"]
    corrected_features = rating.normalize_features(difficulty_rating.get("features"))
    corrected_features.update(review.get("feature_corrections") or {})
    difficulty_rating["features"] = corrected_features
    action = {
        "rule": "adjacent_boundary_review",
        "from": current,
        "to": target,
        "evidence": [str(value.get("finding") or "") for value in review.get("decisive_evidence") or []][:8],
    }
    difficulty_rating.setdefault("postprocess_actions", []).append(action)
    item.setdefault("postprocess_actions", []).append(copy.deepcopy(action))
    difficulty_rating["difficulty_level"] = target
    rating.finalize_postprocessed_result(difficulty_rating)


def _response_text(body: Dict[str, Any]) -> str:
    output_text = ""
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                output_text = str(content.get("text") or "")
    return output_text


async def call_review_model(
    prompt: str,
    content: str,
    image_urls: Sequence[str],
    route: Dict[str, Any],
    image_route: Dict[str, Any],
    session: aiohttp.ClientSession,
    model_name: str,
    temperature: Optional[float],
    retries: int,
    timeout_sec: int,
) -> Tuple[Dict[str, Any], float, int, int, int, str]:
    started = time.time()
    prompt_tokens = completion_tokens = total_tokens = 0
    last_error = ""
    mixed_content: List[Dict[str, Any]] = [
        {"type": "input_text", "text": prompt + "\n\n" + content}
    ]
    mixed_content.extend({"type": "input_image", "image_url": url} for url in image_urls)
    for attempt in range(retries):
        payload: Dict[str, Any] = {
            "model": model_name,
            "input": [{"role": "user", "content": mixed_content}],
            "thinking": {"type": "disabled"},
        }
        if temperature is not None:
            payload["temperature"] = temperature
        try:
            async with session.post(
                f"{rating.BASE_URL}responses",
                json=payload,
                headers={"Authorization": f"Bearer {rating.API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=timeout_sec),
            ) as response:
                if response.status == 200:
                    body = await response.json()
                    usage = body.get("usage") or {}
                    prompt_tokens += int(usage.get("input_tokens", 0) or 0)
                    completion_tokens += int(usage.get("output_tokens", 0) or 0)
                    total_tokens += int(usage.get("total_tokens", 0) or 0)
                    review = normalize_review(rating.parse_model_response(_response_text(body)))
                    error = validate_review(review, route, image_route)
                    if not error:
                        return review, time.time() - started, prompt_tokens, completion_tokens, total_tokens, ""
                    last_error = error
                else:
                    response_text = await response.text()
                    last_error = f"HTTP {response.status}: {response_text[:300]}"
                    if response.status < 500 and response.status != 429:
                        break
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2**attempt + random.random())
    return {}, time.time() - started, prompt_tokens, completion_tokens, total_tokens, last_error or "复核响应无效"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_signature(
    input_path: str,
    prompt: str,
    model_name: str,
    temperature: Optional[float],
    scope: str,
    image_mode: str,
    allow_writeback: bool,
    max_review_calls: Optional[int],
) -> str:
    value = {
        "pipeline_version": PIPELINE_VERSION,
        "input_sha256": sha256_file(input_path),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "model": model_name,
        "temperature": temperature,
        "scope": scope,
        "image_mode": image_mode,
        "allow_writeback": allow_writeback,
        "max_review_calls": max_review_calls,
    }
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path}:{line_number} 缺少 question_id")
            if question_id in seen:
                raise ValueError(f"{path} 存在重复 question_id={question_id}")
            if extract_final_level(item) not in LEVEL_INDEX:
                raise ValueError(f"{path}:{line_number} 缺少合法最终等级")
            seen.add(question_id)
            rows.append(item)
    return rows


def processed_ids(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    values: set[str] = set()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            try:
                values.add(str(json.loads(line)["question_id"]))
            except Exception:
                continue
    return values


def validate_resume_output(path: str, signature: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            agent = item.get("boundary_review_agent")
            actual = agent.get("run_signature") if isinstance(agent, dict) else None
            if actual != signature:
                raise ValueError(
                    f"{path}:{line_number} 的复核配置不一致；请更换输出文件或删除旧结果"
                )


async def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    async with OUTPUT_LOCK:
        async with aiofiles.open(path, "a", encoding="utf-8") as handle:
            await handle.write(json.dumps(item, ensure_ascii=False) + "\n")


async def process_item(
    item: Dict[str, Any],
    route: Dict[str, Any],
    prompt: str,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    output_path: str,
    error_path: str,
    model_name: str,
    temperature: Optional[float],
    image_mode: str,
    allow_writeback: bool,
    signature: str,
    retries: int,
    timeout_sec: int,
) -> None:
    output = copy.deepcopy(item)
    difficulty_rating = extract_rating(output)
    if difficulty_rating:
        rating.sync_final_adjacent_reasoning(difficulty_rating)
    image_route = route_images(output, image_mode)
    review: Dict[str, Any] = {}
    error = ""
    elapsed = 0.0
    prompt_tokens = completion_tokens = total_tokens = 0
    would_apply = applied = False
    decision_reason = "未进入相邻边界风险复核"

    if route.get("selected"):
        content = build_review_content(output, route, image_route)
        async with semaphore:
            review, elapsed, prompt_tokens, completion_tokens, total_tokens, error = await call_review_model(
                prompt,
                content,
                image_route["selected_urls"],
                route,
                image_route,
                session,
                model_name,
                temperature,
                retries,
                timeout_sec,
            )
        if not error:
            would_apply, decision_reason = should_apply_review(output, route, review)
            if would_apply and allow_writeback:
                apply_review(output, review)
                applied = True
            elif would_apply:
                decision_reason = "audit-only：满足高置信写回门槛，但保持冻结首轮等级"

    output["boundary_review_agent"] = {
        "pipeline_version": PIPELINE_VERSION,
        "run_signature": signature,
        "selected": bool(route.get("selected")),
        "route": route,
        "audit_only": not allow_writeback,
        "model": model_name if route.get("selected") else None,
        "temperature": temperature if route.get("selected") else None,
        "input_quality": image_route,
        "review": review,
        "would_apply": would_apply,
        "applied": applied,
        "decision_reason": decision_reason,
        "error": error,
        "elapsed_seconds": round(elapsed, 3),
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
    await append_jsonl(output_path, output)
    if error:
        await append_jsonl(
            error_path,
            {"question_id": output.get("question_id"), "route": route, "error": error},
        )


def summarize_output(path: str) -> Dict[str, Any]:
    stats: Counter[str] = Counter()
    tokens = 0
    boundaries: Counter[str] = Counter()
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            agent = item.get("boundary_review_agent") or {}
            stats["rows"] += 1
            if agent.get("selected"):
                stats["selected"] += 1
            if agent.get("would_apply"):
                stats["would_apply"] += 1
            if agent.get("applied"):
                stats["applied"] += 1
            if agent.get("error"):
                stats["errors"] += 1
            if (agent.get("input_quality") or {}).get("image_included"):
                stats["images_included"] += 1
            boundary = (agent.get("route") or {}).get("review_boundary") or []
            if boundary:
                boundaries["/".join(boundary)] += 1
            tokens += int((agent.get("usage") or {}).get("total_tokens", 0) or 0)
    return {
        **dict(stats),
        "total_tokens": tokens,
        "boundary_distribution": dict(boundaries),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="初中物理相邻边界二次复核（默认只审不改）")
    parser.add_argument("-i", "--input", required=True, help="冻结首轮打标结果 JSONL")
    parser.add_argument("-o", "--output", required=True, help="复核输出 JSONL")
    parser.add_argument("-e", "--error", required=True, help="错误日志 JSONL")
    parser.add_argument("-p", "--prompt", default=str(DEFAULT_PROMPT))
    parser.add_argument("-c", "--concurrency", type=int, default=15)
    parser.add_argument("-t", "--timeout", type=int, default=180)
    parser.add_argument("-r", "--retries", type=int, default=3)
    parser.add_argument("--model", default=os.getenv("BOUNDARY_REVIEW_MODEL", "doubao-seed-2.0-lite"))
    parser.add_argument("--temperature", default=os.getenv("BOUNDARY_REVIEW_TEMPERATURE", "1"))
    parser.add_argument("--review-scope", choices=("risk", "all"), default="risk")
    parser.add_argument("--image-mode", choices=("off", "auto", "all"), default="auto")
    parser.add_argument("--allow-writeback", action="store_true", help="显式允许高置信相邻一档写回；默认关闭")
    parser.add_argument("--max-review-calls", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


async def main() -> None:
    args = build_parser().parse_args()
    prompt = Path(args.prompt).read_text(encoding="utf-8")
    rows = load_jsonl(args.input)
    routes = [route_boundary_review(item, args.review_scope) for item in rows]
    route_stats = Counter("selected" if value["selected"] else "not_selected" for value in routes)
    print(f"首轮输入: {args.input}，题目数: {len(rows)}")
    print(f"相邻边界候选: {route_stats['selected']}，直接保留: {route_stats['not_selected']}")
    if args.dry_run:
        print(json.dumps({"route_stats": dict(route_stats)}, ensure_ascii=False, indent=2))
        return

    temperature = rating.resolve_temperature(args.model, args.temperature)
    signature = build_run_signature(
        args.input,
        prompt,
        args.model,
        temperature,
        args.review_scope,
        args.image_mode,
        args.allow_writeback,
        args.max_review_calls,
    )
    for path in (args.output, args.error):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    validate_resume_output(args.output, signature)
    done = processed_ids(args.output)
    pending: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    selected_seen = 0
    for item, route in zip(rows, routes):
        if str(item["question_id"]) in done:
            continue
        if route["selected"]:
            selected_seen += 1
            if args.max_review_calls is not None and selected_seen > args.max_review_calls:
                route = copy.deepcopy(route)
                route["selected"] = False
                route["reasons"].append("超过 max-review-calls，未调用复核模型")
        pending.append((item, route))
    print(
        f"已完成: {len(done)}，待写入: {len(pending)}，"
        f"模式: {'自动写回' if args.allow_writeback else 'audit-only'}，"
        f"图片: {args.image_mode}，模型: {args.model}，temperature={temperature}"
    )
    if not pending:
        print(json.dumps(summarize_output(args.output), ensure_ascii=False, indent=2))
        return

    semaphore = Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    progress = tqdm(total=len(pending), unit="item", desc="Boundary Review Progress")
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(
                process_item(
                    item,
                    route,
                    prompt,
                    session,
                    semaphore,
                    args.output,
                    args.error,
                    args.model,
                    temperature,
                    args.image_mode,
                    args.allow_writeback,
                    signature,
                    args.retries,
                    args.timeout,
                )
            )
            for item, route in pending
        ]
        for task in asyncio.as_completed(tasks):
            await task
            progress.update(1)
    progress.close()
    summary = summarize_output(args.output)
    summary_path = args.output + ".summary.json"
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("复核完成:", os.path.abspath(args.output))
    print("汇总:", json.dumps(summary, ensure_ascii=False, sort_keys=True))
    print("汇总文件:", os.path.abspath(summary_path))


if __name__ == "__main__":
    started = time.time()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("收到中断信号，已停止相邻边界复核。")
    finally:
        print(f"相邻边界复核耗时: {round((time.time() - started) / 60, 2)} 分钟")
