# -*- coding: utf-8 -*-
"""冻结首轮之后的选择性证据审计 Agent Pipeline。

默认流程：首轮结果 -> 确定性风险路由 -> Lite证据审计 -> 只记录不写回。
旧版独立盲审策略仅保留用于历史回放。
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
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

DEFAULT_PROMPT = ROOT / "prompts" / "初中物理难度盲审提示词.txt"
DEFAULT_EVIDENCE_PROMPT = ROOT / "prompts" / "初中物理难度证据审计提示词.txt"
LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
TASK_STRUCTURES = {"直接识别", "显性应用", "常规分析", "决定性转换", "高密度综合链", "全链耦合"}
OUTPUT_LOCK = Lock()
PIPELINE_VERSION = "evidence-audit-v2"

STRUCTURE_SCORE_MAPS: Dict[str, Dict[str, int]] = {
    "step_count": {"1-2步": 0, "3-5步": 2, "6-8步": 4, "9-12步": 6, "12步以上": 7},
    "formula_count": {"0-1个": 0, "2-3个": 1, "4-6个": 3, "7个以上": 4},
    "calculation_complexity": {"口算或直接判断": 0, "简单笔算": 1, "多公式联立": 3, "复杂方程或范围计算": 4},
    "reasoning_chain": {"直接套用": 0, "简单因果推理": 1, "多层因果推理": 3, "逆向推理或临界分析": 4},
    "state_count": {"单状态": 0, "双状态": 1, "多状态": 3, "连续变化或临界状态": 4},
    "constraint_count": {"无约束": 0, "单一约束": 1, "多约束": 3},
    "variable_relation": {"无变量关系": 0, "简单正反比": 1, "图像函数关系": 2, "多变量耦合关系": 4},
    "experiment_requirement": {"无": 0, "基础操作或读数": 1, "控制变量或故障分析": 2, "方案设计或误差评价": 4},
    "graph_table_requirement": {"无": 0, "直接读数": 1, "多组比较归纳": 2, "图像反推或外推": 4},
    "knowledge_count": {"1个": 0, "2-3个": 1, "4个及以上": 2},
    "subquestion_dependency": {"无多问": 0, "多问但相互独立": 1, "多问且层层递进": 3},
    "cross_module": {"同一模块内部": 0, "跨模块综合": 2},
}


def extract_final_level(item: Dict[str, Any]) -> Optional[str]:
    difficulty_rating = item.get("difficulty_rating")
    if isinstance(difficulty_rating, dict):
        level = difficulty_rating.get("difficulty_level")
        if level in LEVEL_INDEX:
            return str(level)
    raw = item.get("difficulty_level_raw")
    return str(raw) if raw in LEVEL_INDEX else None


def structural_score(features: Dict[str, Any]) -> int:
    return sum(mapping.get(str(features.get(field, "")), 0) for field, mapping in STRUCTURE_SCORE_MAPS.items())


def route_verification_risk(item: Dict[str, Any]) -> Dict[str, Any]:
    """只依据首轮输出内部一致性选择盲审候选，不读取任何标签。"""
    current = extract_final_level(item)
    if current not in LEVEL_INDEX:
        raise ValueError("首轮结果缺少合法最终等级")
    difficulty_rating = item.get("difficulty_rating") or {}
    features = difficulty_rating.get("features") if isinstance(difficulty_rating, dict) else {}
    features = features if isinstance(features, dict) else {}
    score = structural_score(features)
    reasons: List[str] = []
    directions: set[str] = set()

    actions = item.get("postprocess_actions") or difficulty_rating.get("postprocess_actions") or []
    raw = item.get("difficulty_level_raw")
    if actions or (raw in LEVEL_INDEX and raw != current):
        reasons.append("首轮发生后处理调整")
        if raw in LEVEL_INDEX and raw != current:
            directions.add("up" if LEVEL_INDEX[raw] > LEVEL_INDEX[current] else "down")

    # 两端档位代价最高且数量有限，全部进入盲审。
    if current == "送分题":
        reasons.append("两端档位送分题复核")
        directions.add("up")
    elif current == "压轴题":
        reasons.append("两端档位压轴题复核")
        directions.add("down")

    bounds = {
        "送分题": (None, 3),
        "基础题": (1, 8),
        "中等题": (5, 16),
        "拔高题": (11, 26),
        "压轴题": (21, None),
    }
    low, high = bounds[current]
    if low is not None and score < low:
        reasons.append(f"{current}结构负担明显靠近低一档")
        directions.add("down")
    if high is not None and score >= high:
        reasons.append(f"{current}结构负担明显靠近高一档")
        directions.add("up")

    low_structure_medium = bool(
        current == "中等题"
        and features.get("step_count") == "1-2步"
        and features.get("formula_count") == "0-1个"
        and features.get("calculation_complexity") == "口算或直接判断"
        and features.get("state_count") == "单状态"
        and features.get("constraint_count") == "无约束"
        and features.get("variable_relation") == "无变量关系"
        and features.get("experiment_requirement") == "无"
        and features.get("graph_table_requirement") == "无"
    )
    if low_structure_medium:
        reasons.append("中等题但呈现低结构直接任务")
        directions.add("down")

    dense_low_prediction = bool(
        current in {"基础题", "中等题"}
        and features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
        and features.get("calculation_complexity") in {"多公式联立", "复杂方程或范围计算"}
        and features.get("state_count") in {"多状态", "连续变化或临界状态"}
    )
    if dense_low_prediction:
        reasons.append("低预测档与长链多状态结构冲突")
        directions.add("up")

    return {
        "selected": bool(reasons),
        "current_level": current,
        "structural_score": score,
        "reasons": reasons,
        "allowed_directions": sorted(directions),
    }


def build_blind_review_content(item: Dict[str, Any]) -> str:
    """盲审只发送题目本身；首轮结论和所有标签均被隔离。"""
    safe = rating.sanitize_question_data(item)
    question_only = {
        key: safe.get(key)
        for key in ("stem", "options", "analysis", "sub_questions")
        if safe.get(key)
    }
    return "【待独立盲审题目】\n" + rating.construct_question_content(question_only) + "\n\n请严格输出盲审 JSON。"


def build_evidence_audit_content(item: Dict[str, Any], route: Dict[str, Any]) -> str:
    """提供题目与首轮主张用于反证审计，但隔离来源difficulty及评估标签。"""
    safe = rating.sanitize_question_data(item)
    question = {
        key: safe.get(key)
        for key in ("stem", "options", "analysis", "sub_questions")
        if safe.get(key)
    }
    difficulty_rating = item.get("difficulty_rating")
    difficulty_rating = difficulty_rating if isinstance(difficulty_rating, dict) else {}
    first_stage = {
        "raw_level": item.get("difficulty_level_raw"),
        "final_level": extract_final_level(item),
        "features": difficulty_rating.get("features") or {},
        "reasoning": difficulty_rating.get("reasoning") or {},
        "postprocess_actions": item.get("postprocess_actions")
        or difficulty_rating.get("postprocess_actions")
        or [],
    }
    risk = {
        "reasons": list(route.get("reasons") or []),
        "allowed_directions": list(route.get("allowed_directions") or []),
        "structural_score": route.get("structural_score"),
    }
    payload = {"question": question, "first_stage": first_stage, "risk_route": risk}
    return (
        "【待反证审计材料】\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\n请核验首轮主张，只输出证据审计 JSON。"
    )


EVIDENCE_AUDIT_ACTIONS = {"保留", "升一档", "降一档", "仅标记人工复核"}
EVIDENCE_CONTRADICTION_TYPES = {
    "feature_conflict",
    "reasoning_hallucination",
    "postprocess_precondition_failed",
    "missed_structure",
}
EVIDENCE_SOURCE_FIELDS = {"stem", "options", "analysis", "sub_questions"}


def _normalized_text(value: Any) -> str:
    return "".join(str(value or "").split())


def _source_field_text(item: Dict[str, Any], field: str) -> str:
    value = item.get(field)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def normalize_evidence_audit(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    verified = source.get("verified_current_evidence")
    contradictions = source.get("contradictions")
    required = source.get("candidate_required_conditions")
    return {
        "current_level_supported": source.get("current_level_supported"),
        "verified_current_evidence": [
            dict(item) for item in verified if isinstance(item, dict)
        ]
        if isinstance(verified, list)
        else [],
        "contradictions": [
            dict(item) for item in contradictions if isinstance(item, dict)
        ]
        if isinstance(contradictions, list)
        else [],
        "postprocess_rule_valid": source.get("postprocess_rule_valid"),
        "recommended_action": str(source.get("recommended_action") or "").strip(),
        "recommended_level": str(source.get("recommended_level") or "").strip(),
        "candidate_required_conditions": [
            str(item).strip() for item in required if str(item).strip()
        ]
        if isinstance(required, list)
        else [],
        "candidate_conditions_satisfied": source.get("candidate_conditions_satisfied"),
        "manual_review_required": source.get("manual_review_required"),
        "action_basis": str(source.get("action_basis") or "").strip(),
    }


def validate_evidence_audit(audit: Dict[str, Any]) -> Optional[str]:
    if not isinstance(audit.get("current_level_supported"), bool):
        return "current_level_supported 必须是布尔值"
    if audit.get("recommended_action") not in EVIDENCE_AUDIT_ACTIONS:
        return "recommended_action 非法"
    if audit.get("recommended_level") not in LEVEL_INDEX:
        return "recommended_level 非法"
    if not isinstance(audit.get("candidate_conditions_satisfied"), bool):
        return "candidate_conditions_satisfied 必须是布尔值"
    if not isinstance(audit.get("manual_review_required"), bool):
        return "manual_review_required 必须是布尔值"
    if not audit.get("action_basis"):
        return "缺少 action_basis"
    if not isinstance(audit.get("contradictions"), list):
        return "contradictions 必须是数组"
    if not isinstance(audit.get("verified_current_evidence"), list):
        return "verified_current_evidence 必须是数组"
    return None


def should_apply_evidence_audit(
    item: Dict[str, Any],
    route: Dict[str, Any],
    audit: Dict[str, Any],
) -> Tuple[bool, str]:
    """只有题目原文可核验的反证才能形成相邻档候选写回。"""
    current = extract_final_level(item)
    if current not in LEVEL_INDEX:
        return False, "当前等级非法"
    if not route.get("selected"):
        return False, "题目未被风险路由选中"
    action = audit.get("recommended_action")
    target = audit.get("recommended_level")
    if action not in EVIDENCE_AUDIT_ACTIONS or target not in LEVEL_INDEX:
        return False, "审计动作或建议等级非法"
    if action in {"保留", "仅标记人工复核"} or target == current:
        return False, "审计未提出可自动执行的等级变化"
    if audit.get("current_level_supported") is not False:
        return False, "未明确否定当前等级"
    if audit.get("manual_review_required") is not False:
        return False, "审计要求人工复核"
    if audit.get("candidate_conditions_satisfied") is not True:
        return False, "相邻候选档必要条件未全部满足"
    distance = LEVEL_INDEX[target] - LEVEL_INDEX[current]
    if abs(distance) != 1:
        return False, "建议调整不是相邻一档"
    expected_action = "升一档" if distance > 0 else "降一档"
    if action != expected_action:
        return False, "建议动作与建议等级不一致"
    direction = "up" if distance > 0 else "down"
    if direction not in set(route.get("allowed_directions") or []):
        return False, "建议方向没有得到确定性风险路由支持"

    contradictions = audit.get("contradictions")
    if not isinstance(contradictions, list) or not contradictions:
        return False, "缺少可核验反证"
    verified: List[Dict[str, Any]] = []
    for contradiction in contradictions:
        if not isinstance(contradiction, dict):
            continue
        contradiction_type = contradiction.get("contradiction_type")
        source_field = str(contradiction.get("source_field") or "")
        excerpt = str(contradiction.get("source_excerpt") or "").strip()
        if contradiction_type not in EVIDENCE_CONTRADICTION_TYPES:
            continue
        if source_field not in EVIDENCE_SOURCE_FIELDS or not excerpt:
            continue
        source_text = _normalized_text(_source_field_text(item, source_field))
        if _normalized_text(excerpt) not in source_text:
            continue
        if not contradiction.get("current_claim") or not contradiction.get("verified_fact"):
            continue
        verified.append(contradiction)
    if not verified:
        return False, "反证引用无法在题目或解析中核验"

    difficulty_rating = item.get("difficulty_rating")
    difficulty_rating = difficulty_rating if isinstance(difficulty_rating, dict) else {}
    actions = item.get("postprocess_actions") or difficulty_rating.get("postprocess_actions") or []
    rule_names = {str(value.get("rule") or "") for value in actions if isinstance(value, dict)}
    if rule_names:
        matching = [
            value
            for value in verified
            if value.get("contradiction_type") == "postprocess_precondition_failed"
            and str(value.get("affected_rule") or "") in rule_names
        ]
        if audit.get("postprocess_rule_valid") is not False or not matching:
            return False, "缺少与实际后处理规则匹配的前提失败反证"
    return True, "存在题目原文可核验反证，且相邻候选档条件满足"


def resolve_evidence_audit_decision(
    item: Dict[str, Any],
    route: Dict[str, Any],
    audit: Dict[str, Any],
    *,
    audit_only: bool,
) -> Tuple[bool, bool, str]:
    would_apply, reason = should_apply_evidence_audit(item, route, audit)
    if audit_only:
        if would_apply:
            return False, True, "audit-only：记录可核验候选调整，但保持冻结等级"
        return False, False, f"audit-only：{reason}"
    return would_apply, would_apply, reason


def resolve_audit_only_mode(
    strategy: str,
    *,
    requested_audit_only: bool,
    allow_writeback: bool,
) -> bool:
    if requested_audit_only and allow_writeback:
        raise ValueError("--audit-only 与 --allow-writeback 不能同时使用")
    if strategy == "evidence_audit":
        return not allow_writeback
    return requested_audit_only


def evaluate_review_response(
    item: Dict[str, Any],
    route: Dict[str, Any],
    raw_response: Any,
    *,
    strategy: str,
    accepted_confidences: Iterable[str],
    audit_only: bool,
) -> Dict[str, Any]:
    if strategy == "evidence_audit":
        audit = normalize_evidence_audit(raw_response)
        error = validate_evidence_audit(audit)
        if error:
            return {
                "blind_review": {},
                "evidence_audit": audit,
                "applied": False,
                "would_apply": False,
                "decision_reason": error,
                "error": error,
            }
        applied, would_apply, reason = resolve_evidence_audit_decision(
            item,
            route,
            audit,
            audit_only=audit_only,
        )
        return {
            "blind_review": {},
            "evidence_audit": audit,
            "applied": applied,
            "would_apply": would_apply,
            "decision_reason": reason,
            "error": "",
        }
    review = normalize_blind_review(raw_response)
    error = validate_blind_review(review)
    if error:
        return {
            "blind_review": review,
            "evidence_audit": {},
            "applied": False,
            "would_apply": False,
            "decision_reason": error,
            "error": error,
        }
    applied, would_apply, reason = resolve_review_decision(
        extract_final_level(item) or "",
        route,
        review,
        accepted_confidences,
        audit_only=audit_only,
    )
    return {
        "blind_review": review,
        "evidence_audit": {},
        "applied": applied,
        "would_apply": would_apply,
        "decision_reason": reason,
        "error": "",
    }


def canonical_confidence(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"高", "high"}:
        return "高"
    if text in {"中", "medium", "mid"}:
        return "中"
    if text in {"低", "low"}:
        return "低"
    return ""


def normalize_blind_review(value: Any) -> Dict[str, Any]:
    source = value if isinstance(value, dict) else {}
    level = str(source.get("review_level") or source.get("difficulty_level") or "").strip()
    acceptable = source.get("acceptable_levels")
    if not isinstance(acceptable, list):
        acceptable = [level] if level else []
    evidence = source.get("structural_evidence")
    if not isinstance(evidence, list):
        evidence = [str(evidence)] if evidence else []
    return {
        "review_level": level,
        "acceptable_levels": [str(item).strip() for item in acceptable if str(item).strip()],
        "boundary_status": str(source.get("boundary_status") or "").strip(),
        "confidence": canonical_confidence(source.get("confidence")),
        "effective_task_summary": str(source.get("effective_task_summary") or "").strip(),
        "effective_decision_count": source.get("effective_decision_count"),
        "task_structure": str(source.get("task_structure") or "").strip(),
        "structural_evidence": [str(item).strip() for item in evidence if str(item).strip()],
        "boundary_basis": str(source.get("boundary_basis") or "").strip(),
    }


def validate_blind_review(review: Dict[str, Any]) -> Optional[str]:
    level = review.get("review_level")
    if level not in LEVEL_INDEX:
        return "review_level 非法"
    acceptable = review.get("acceptable_levels")
    if not isinstance(acceptable, list) or not acceptable or len(acceptable) > 2:
        return "acceptable_levels 必须包含一个或两个等级"
    if any(item not in LEVEL_INDEX for item in acceptable) or level not in acceptable:
        return "acceptable_levels 含非法等级或未包含 review_level"
    if len(acceptable) == 2:
        indices = sorted(LEVEL_INDEX[item] for item in acceptable)
        if indices[1] - indices[0] != 1:
            return "acceptable_levels 必须相邻"
    boundary = review.get("boundary_status")
    if boundary not in {"明确归档", "相邻边界均可"}:
        return "boundary_status 非法"
    if boundary == "明确归档" and len(acceptable) != 1:
        return "明确归档只能包含一个可接受等级"
    if boundary == "相邻边界均可" and len(acceptable) != 2:
        return "相邻边界均可必须包含两个等级"
    if review.get("confidence") not in {"高", "中", "低"}:
        return "confidence 非法"
    decision_count = review.get("effective_decision_count")
    if isinstance(decision_count, bool) or not isinstance(decision_count, int) or decision_count < 0:
        return "effective_decision_count 必须是非负整数"
    if review.get("task_structure") not in TASK_STRUCTURES:
        return "task_structure 非法"
    for field in ("effective_task_summary", "boundary_basis"):
        if not review.get(field):
            return f"缺少 {field}"
    if not review.get("structural_evidence"):
        return "缺少 structural_evidence"
    return None


def should_apply_blind_review(
    current_level: str,
    route: Dict[str, Any],
    review: Dict[str, Any],
    accepted_confidences: Iterable[str],
) -> Tuple[bool, str]:
    error = validate_blind_review(review)
    if error:
        return False, error
    if not route.get("selected"):
        return False, "题目未被风险路由选中"
    acceptable = set(review.get("acceptable_levels") or [])
    if current_level in acceptable:
        return False, "盲审认为当前等级仍可接受"
    target = review["review_level"]
    if target == current_level:
        return False, "盲审保持当前等级"
    distance = LEVEL_INDEX[target] - LEVEL_INDEX[current_level]
    if abs(distance) != 1:
        return False, "盲审调整超过一档，仅记录不自动写回"
    if review.get("confidence") not in set(accepted_confidences):
        return False, "盲审置信度未达到自动写回阈值"
    direction = "up" if distance > 0 else "down"
    if direction not in set(route.get("allowed_directions") or []):
        return False, "盲审调整方向没有得到确定性风险路由支持"
    supported, support_reason = review_supports_target_level(target, review)
    if not supported:
        return False, f"盲审目标档结构证据不足：{support_reason}"
    return True, "高置信度盲审与风险方向一致，执行相邻档调整"


def resolve_review_decision(
    current_level: str,
    route: Dict[str, Any],
    review: Dict[str, Any],
    accepted_confidences: Iterable[str],
    *,
    audit_only: bool,
) -> Tuple[bool, bool, str]:
    """区分“按现有门控本可写回”和“本次是否真的写回”。"""
    would_apply, reason = should_apply_blind_review(
        current_level,
        route,
        review,
        accepted_confidences,
    )
    if audit_only:
        if would_apply:
            return False, True, "audit-only：记录潜在调整，但保持冻结等级"
        return False, False, f"audit-only：{reason}"
    return would_apply, would_apply, reason


def review_supports_target_level(target: str, review: Dict[str, Any]) -> Tuple[bool, str]:
    """避免仅凭 Mini 的高置信度写回与其任务结构自相矛盾的等级。"""
    count = review.get("effective_decision_count")
    if isinstance(count, bool) or not isinstance(count, int):
        return False, "有效决策数非法"
    evidence_text = "\n".join(
        [
            str(review.get("task_structure") or ""),
            *[str(value) for value in review.get("structural_evidence") or []],
        ]
    )

    if target == "送分题":
        return (count <= 1, "送分题应不超过一次有效物理决策")
    if target == "基础题":
        return (count <= 2, "基础题自动写回应保持在1—2个有效物理决策")
    if target == "中等题":
        return (3 <= count <= 4, "中等题自动写回应具有3—4个有效物理决策")
    if target == "拔高题":
        decisive_terms = [
            "决定性转换",
            "隐含条件",
            "图像反推",
            "等效替代",
            "关键操作顺序",
            "几何转化",
            "误差方向",
            "临界筛选",
        ]
        supported = count >= 5 or any(term in evidence_text for term in decisive_terms)
        return supported, "拔高题应达到5步左右或存在可核验的决定性转换"
    if target == "压轴题":
        signal_groups = [
            ["分类讨论", "多解筛选", "有效解筛选"],
            ["临界极值", "边界覆盖", "不等式"],
            ["复杂多变量耦合", "多图共同反推", "多对象、多过程"],
            ["开放方案设计", "可行性验证", "方案比较"],
        ]
        strong_count = sum(any(term in evidence_text for term in group) for group in signal_groups)
        supported = count >= 7 and strong_count >= 2
        return supported, "压轴题应达到7步以上并至少具备两类强压轴结构"
    return False, "目标等级非法"


def apply_verified_level(item: Dict[str, Any], target: str) -> None:
    """保存完整首轮快照后写回等级，保证 features/reasoning 可回放。"""
    if target not in LEVEL_INDEX:
        raise ValueError("待写回等级非法")
    difficulty_rating = item.get("difficulty_rating")
    if not isinstance(difficulty_rating, dict):
        raise ValueError("首轮 difficulty_rating 缺失")
    item["difficulty_rating_before_verification"] = copy.deepcopy(difficulty_rating)
    difficulty_rating["difficulty_level"] = target
    rating.sync_coarse_difficulty(difficulty_rating)


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


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_run_signature(
    input_path: str,
    prompt_text: str,
    model_name: str,
    temperature: Optional[float],
    accepted_confidences: Sequence[str],
    audit_only: bool = False,
    strategy: str = "blind_review",
) -> str:
    payload = {
        "pipeline_version": PIPELINE_VERSION,
        "input_sha256": sha256_file(input_path),
        "prompt_sha256": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
        "model": model_name,
        "temperature": temperature,
        "accepted_confidences": list(accepted_confidences),
        "audit_only": bool(audit_only),
        "strategy": strategy,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_resume_output(path: str, expected_signature: str) -> None:
    """禁止把不同输入、Prompt或模型配置续写到同一输出文件。"""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON，不能安全续跑") from exc
            agent = item.get("verification_agent")
            actual = agent.get("run_signature") if isinstance(agent, dict) else None
            if actual != expected_signature:
                raise ValueError(
                    f"{path}:{line_number} 的 Agent 配置不一致；请更换输出文件或删除旧实验结果"
                )


async def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    async with OUTPUT_LOCK:
        async with aiofiles.open(path, "a", encoding="utf-8") as handle:
            await handle.write(json.dumps(item, ensure_ascii=False) + "\n")


async def call_blind_model(
    prompt: str,
    content: str,
    session: aiohttp.ClientSession,
    model_name: str,
    temperature: Optional[float],
    retries: int,
    timeout_sec: int,
) -> Tuple[Dict[str, Any], float, int, int, int, str]:
    started = time.time()
    prompt_tokens = completion_tokens = total_tokens = 0
    last_error = ""
    for attempt in range(retries):
        payload: Dict[str, Any] = {
            "model": model_name,
            "input": [{"role": "user", "content": prompt + "\n\n" + content}],
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
                    usage = body.get("usage", {})
                    prompt_tokens += int(usage.get("input_tokens", 0) or 0)
                    completion_tokens += int(usage.get("output_tokens", 0) or 0)
                    total_tokens += int(usage.get("total_tokens", 0) or 0)
                    output_text = ""
                    for output_item in body.get("output", []):
                        if output_item.get("type") != "message":
                            continue
                        for output_content in output_item.get("content", []):
                            if output_content.get("type") == "output_text":
                                output_text = str(output_content.get("text") or "")
                    parsed = normalize_blind_review(rating.parse_model_response(output_text))
                    validation_error = validate_blind_review(parsed)
                    if not validation_error:
                        return parsed, time.time() - started, prompt_tokens, completion_tokens, total_tokens, ""
                    last_error = validation_error
                else:
                    text = await response.text()
                    last_error = f"HTTP {response.status}: {text[:300]}"
                    if response.status < 500 and response.status != 429:
                        break
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2**attempt + random.random())
    return {}, time.time() - started, prompt_tokens, completion_tokens, total_tokens, last_error or "盲审响应无效"


async def call_evidence_audit_model(
    prompt: str,
    content: str,
    session: aiohttp.ClientSession,
    model_name: str,
    temperature: Optional[float],
    retries: int,
    timeout_sec: int,
) -> Tuple[Dict[str, Any], float, int, int, int, str]:
    started = time.time()
    prompt_tokens = completion_tokens = total_tokens = 0
    last_error = ""
    for attempt in range(retries):
        payload: Dict[str, Any] = {
            "model": model_name,
            "input": [{"role": "user", "content": prompt + "\n\n" + content}],
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
                    usage = body.get("usage", {})
                    prompt_tokens += int(usage.get("input_tokens", 0) or 0)
                    completion_tokens += int(usage.get("output_tokens", 0) or 0)
                    total_tokens += int(usage.get("total_tokens", 0) or 0)
                    output_text = ""
                    for output_item in body.get("output", []):
                        if output_item.get("type") != "message":
                            continue
                        for output_content in output_item.get("content", []):
                            if output_content.get("type") == "output_text":
                                output_text = str(output_content.get("text") or "")
                    parsed = normalize_evidence_audit(rating.parse_model_response(output_text))
                    validation_error = validate_evidence_audit(parsed)
                    if not validation_error:
                        return parsed, time.time() - started, prompt_tokens, completion_tokens, total_tokens, ""
                    last_error = validation_error
                else:
                    text = await response.text()
                    last_error = f"HTTP {response.status}: {text[:300]}"
                    if response.status < 500 and response.status != 429:
                        break
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            last_error = str(exc)
        except Exception as exc:
            last_error = str(exc)
        if attempt < retries - 1:
            await asyncio.sleep(2**attempt + random.random())
    return (
        {},
        time.time() - started,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        last_error or "证据审计响应无效",
    )


async def process_item(
    source: Dict[str, Any],
    route: Dict[str, Any],
    prompt: str,
    session: aiohttp.ClientSession,
    semaphore: Semaphore,
    output_path: str,
    error_path: str,
    model_name: str,
    temperature: Optional[float],
    accepted_confidences: Sequence[str],
    run_signature: str,
    retries: int,
    timeout_sec: int,
    audit_only: bool,
    strategy: str,
) -> None:
    async with semaphore:
        item = copy.deepcopy(source)
        before = extract_final_level(item)
        review: Dict[str, Any] = {}
        evidence_audit: Dict[str, Any] = {}
        elapsed = prompt_tokens = completion_tokens = total_tokens = 0
        error = ""
        applied = False
        would_apply = False
        decision_reason = "未进入风险路由"
        if route["selected"]:
            if strategy == "evidence_audit":
                evidence_audit, elapsed, prompt_tokens, completion_tokens, total_tokens, error = (
                    await call_evidence_audit_model(
                        prompt,
                        build_evidence_audit_content(item, route),
                        session,
                        model_name,
                        temperature,
                        retries,
                        timeout_sec,
                    )
                )
                raw_response = evidence_audit
            else:
                review, elapsed, prompt_tokens, completion_tokens, total_tokens, error = await call_blind_model(
                    prompt,
                    build_blind_review_content(item),
                    session,
                    model_name,
                    temperature,
                    retries,
                    timeout_sec,
                )
                raw_response = review
            if raw_response:
                decision = evaluate_review_response(
                    item,
                    route,
                    raw_response,
                    strategy=strategy,
                    accepted_confidences=accepted_confidences,
                    audit_only=audit_only,
                )
                review = decision["blind_review"]
                evidence_audit = decision["evidence_audit"]
                applied = decision["applied"]
                would_apply = decision["would_apply"]
                decision_reason = decision["decision_reason"]
            else:
                decision_reason = "审计请求失败，保持首轮等级"

        after = before
        if applied:
            after = review["review_level"]
            apply_verified_level(item, after)

        source_tokens = int(item.get("api_total_tokens", 0) or 0)
        item["verification_agent"] = {
            "enabled": True,
            "strategy": strategy,
            "mode": "audit_only" if audit_only else "auto_apply",
            "pipeline_version": PIPELINE_VERSION,
            "run_signature": run_signature,
            "model": model_name,
            "temperature": temperature,
            "selected": route["selected"],
            "selection_reasons": route["reasons"],
            "allowed_directions": route["allowed_directions"],
            "structural_score": route["structural_score"],
            "blind_review": review,
            "evidence_audit": evidence_audit,
            "applied": applied,
            "would_apply": would_apply,
            "from": before,
            "to": after,
            "decision_reason": decision_reason,
            "error": error,
            "api_time_use": round(float(elapsed), 2),
            "api_prompt_tokens": int(prompt_tokens),
            "api_completion_tokens": int(completion_tokens),
            "api_total_tokens": int(total_tokens),
        }
        item["difficulty_level_before_verification"] = before
        item["difficulty_level_after_verification"] = after
        item["verification_applied"] = applied
        item["pipeline_api_total_tokens"] = source_tokens + int(total_tokens)
        await append_jsonl(output_path, item)
        if error:
            await append_jsonl(
                error_path,
                {"question_id": item.get("question_id"), "verification_error": error, "route": route},
            )


async def main() -> None:
    parser = argparse.ArgumentParser(description="冻结首轮后的证据审计 Agent Pipeline")
    parser.add_argument("-i", "--input", required=True, help="冻结版首轮完整结果 JSONL")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-e", "--error", required=True)
    parser.add_argument("-p", "--prompt", default="")
    parser.add_argument("-c", "--concurrency", type=int, default=20)
    parser.add_argument("-t", "--timeout", type=int, default=180)
    parser.add_argument("-r", "--retries", type=int, default=3)
    parser.add_argument("--model", default=os.getenv("PHYSICS_VERIFIER_MODEL", ""))
    parser.add_argument("--temperature", default=os.getenv("PHYSICS_VERIFIER_TEMPERATURE", ""))
    parser.add_argument(
        "--strategy",
        choices=("evidence_audit", "blind_review"),
        default="evidence_audit",
    )
    parser.add_argument("--accept-confidence", default="高")
    parser.add_argument("--max-review-calls", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="正常调用审计模型并保存意见，但禁止修改冻结等级",
    )
    parser.add_argument(
        "--allow-writeback",
        action="store_true",
        help="显式允许证据审计候选写回；首次实验不建议启用",
    )
    args = parser.parse_args()

    audit_only = resolve_audit_only_mode(
        args.strategy,
        requested_audit_only=args.audit_only,
        allow_writeback=args.allow_writeback,
    )
    default_prompt = DEFAULT_EVIDENCE_PROMPT if args.strategy == "evidence_audit" else DEFAULT_PROMPT
    prompt_path = Path(args.prompt) if args.prompt else default_prompt
    if not prompt_path.exists():
        raise FileNotFoundError(f"找不到审计提示词: {prompt_path}")
    prompt = prompt_path.read_text(encoding="utf-8")
    rows = load_jsonl(args.input)
    routes = [route_verification_risk(item) for item in rows]
    stats = Counter()
    for route in routes:
        stats["selected" if route["selected"] else "not_selected"] += 1
        for reason in route["reasons"]:
            stats[reason] += 1
    print(f"首轮输入: {args.input}，题目数: {len(rows)}")
    print(f"风险候选: {stats['selected']}，直接保留: {stats['not_selected']}")
    print("路由原因:", json.dumps(dict(stats), ensure_ascii=False, sort_keys=True))
    if args.dry_run:
        return

    accepted = [value.strip() for value in args.accept_confidence.split(",") if value.strip()]
    if any(value not in {"高", "中", "低"} for value in accepted):
        raise ValueError("--accept-confidence 只能由高、中、低组成")
    model_name = args.model or (
        "doubao-seed-2.0-lite" if args.strategy == "evidence_audit" else "doubao-seed-2.0-mini"
    )
    configured_temperature = args.temperature or ("1" if "lite" in model_name else "0")
    temperature = rating.resolve_temperature(model_name, configured_temperature)
    run_signature = build_run_signature(
        args.input,
        prompt,
        model_name,
        temperature,
        accepted,
        audit_only=audit_only,
        strategy=args.strategy,
    )
    for path in (args.output, args.error):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
    validate_resume_output(args.output, run_signature)
    done = processed_ids(args.output)
    selected_seen = 0
    pending: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for item, route in zip(rows, routes):
        if str(item["question_id"]) in done:
            continue
        if route["selected"]:
            selected_seen += 1
            if args.max_review_calls is not None and selected_seen > args.max_review_calls:
                route = copy.deepcopy(route)
                route["selected"] = False
                route["reasons"] = route["reasons"] + ["超过 max-review-calls，未调用盲审模型"]
        pending.append((item, route))
    print(f"已完成: {len(done)}，待写入: {len(pending)}")
    mode = "只审不改" if audit_only else "自动写回"
    print(
        f"审计策略: {args.strategy}，模型: {model_name}，temperature={temperature}，"
        f"模式={mode}，历史盲审置信度门槛={accepted}"
    )
    if not pending:
        return

    semaphore = Semaphore(args.concurrency)
    connector = aiohttp.TCPConnector(limit=args.concurrency * 2)
    progress = tqdm(total=len(pending), unit="item", desc="Verification Agent Progress")
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
                    model_name,
                    temperature,
                    accepted,
                    run_signature,
                    args.retries,
                    args.timeout,
                    audit_only,
                    args.strategy,
                )
            )
            for item, route in pending
        ]
        for task in asyncio.as_completed(tasks):
            await task
            progress.update(1)
    progress.close()
    print(f"Agent Pipeline 完成: {os.path.abspath(args.output)}")
    print(f"Agent 错误日志: {os.path.abspath(args.error)}")


if __name__ == "__main__":
    started = time.time()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("收到中断信号，已停止 Agent Pipeline。")
    finally:
        print(f"Agent Pipeline 耗时: {round((time.time() - started) / 60, 2)} 分钟")
