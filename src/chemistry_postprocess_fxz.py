"""初中化学难度 FXZ V5 后处理。

该模块只承载现行 V5 特征的校验、规则判定、候选写回与诊断；
规则条件与优先级从 FXZ 运行器原样迁出。
"""

from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Sequence

try:
    from chemistry_observable_features_fxz import (
        OBSERVABLE_FEATURE_FIELDS,
        derive_observable_metrics,
        normalize_observable_features,
        observable_feature_quality_flags,
        validate_observable_features,
    )
except ModuleNotFoundError:
    from src.chemistry_observable_features_fxz import (
        OBSERVABLE_FEATURE_FIELDS,
        derive_observable_metrics,
        normalize_observable_features,
        observable_feature_quality_flags,
        validate_observable_features,
    )


class ChemistrySchemaError(ValueError):
    """模型输出不满足当前 V5 十七项可观测特征契约。"""


LEVEL_MAP = {
    "送分题": 1,
    "基础题": 2,
    "中等题": 3,
    "拔高题": 4,
    "压轴题": 5,
}
VALID_LEVELS = set(LEVEL_MAP)
VISUAL_REFERENCE_RE = re.compile(
    r"(如图|图中|下图|图示|示意图|装置图|实验装置|流程图|"
    r"曲线|坐标图|关系图|图像|图象|表格|微观示意|粒子图|"
    r"看图|观察图|由图|据图|结合图)"
)

# 规则保留候选动作与证据，但不自动改写最终档位；待积累题目口径后再收紧。
# 新教师集回放表明当前“拔高→压轴”规则均为净误伤，因此整个边界只做
# 审计：保留命中规则和证据，最终等级保持模型原判。
TEACHER_GUARD_CANDIDATE_ONLY_RULES = {
    "teacher_basic_to_medium_parallel_phenomena_multitopic",
    "teacher_hard_to_final_dense_multiquestion_quantitative_chain",
    "teacher_hard_to_final_multistage_multiquestion_multireaction",
    "teacher_hard_to_final_strict_deep_quantitative_chain",
}


def is_observable_feature_contract(features: Any) -> bool:
    """当前生产后处理只接受 V5 十七项字段。"""
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
    """识别多阶段反应的多问多反应压轴窄通道。

    该信号不依赖模型对任务数和链长的精确分拆：程序确定
    存在至少四个显式小问，模型同时识别出多阶段反应网络及
    多反应定量关系。这一结构补充任务数或
    链长少拆时 dense 规则可能遗漏的压轴题。
    """
    if not is_observable_feature_contract(features):
        return False
    validated = validate_observable_features(features)
    question_metrics = derive_question_structure_metrics(data)
    return bool(
        question_metrics["explicit_subquestion_count"] >= 4
        and validated["solution_topology"]
        == "多阶段反应网络"
        and validated["reaction_structure"] != "无反应任务"
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
    """统计填空小问，避免把选择项计作小问。"""
    sub_questions = data.get("sub_questions", []) or []
    if isinstance(sub_questions, list) and sub_questions:
        return sum(
            1
            for sub_question in sub_questions
            if isinstance(sub_question, dict)
            and not str(sub_question.get("options", "") or "").strip()
        )

    stem = str(data.get("stem", "") or "")
    marker_matches = list(re.finditer(r"[（(]\s*\d+\s*[）)]", stem))
    if not marker_matches:
        return 0
    segments = []
    for index, match in enumerate(marker_matches):
        next_start = (
            marker_matches[index + 1].start()
            if index + 1 < len(marker_matches)
            else len(stem)
        )
        segments.append(stem[match.end():next_start])
    return sum(bool(re.search(r"[_＿]{2,}", segment)) for segment in segments)


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


def observable_multi_rule_multitopic_medium_signal(
    model_features: Dict[str, Any],
) -> Optional[str]:
    """V5下窄化的“基础→中等”多规则跨课题信号。

    只接受正式 V5 十七项契约，并同时要求至少四项任务、
    三类具体回答规则、两个课题和必要视觉表征。题长、小问数、
    单纯跨课题或不依赖图示的并列直接回答都不能单独触发。
    """
    if not is_observable_feature_contract(model_features):
        return None
    metrics = derive_observable_metrics(model_features)
    if not (
        metrics["effective_task_count"] >= 4
        and metrics["rule_family_count"] >= 3
        and metrics["curriculum_topic_count"] >= 2
        and model_features.get("visual_task_structure")
        != "无必要视觉信息"
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
    *,
    teacher_distribution_guards_enabled: bool,
    teacher_distribution_guards_writeback_enabled: bool,
) -> Dict[str, Any]:
    """V5正式后处理：执行当前Prompt使用的窄教师边界校准。

    生产路径固定为V5十七项可观测特征。低档窄规则可按开关写回；
    拔高到压轴的规则只保留候选证据，不再改写最终档位。
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
        teacher_distribution_guards_writeback_enabled
    )
    rating_result["teacher_distribution_guard_enabled"] = (
        teacher_distribution_guards_enabled
    )
    rating_result["teacher_distribution_guard_writeback_enabled"] = (
        teacher_distribution_guards_writeback_enabled
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
        teacher_distribution_guards_enabled
        or teacher_distribution_guards_writeback_enabled
    )
    teacher_candidate_result = copy.deepcopy(rating_result)
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
    teacher_guard_candidate_only = bool(
        teacher_guard_action
        and teacher_guard_action.get("rule")
        in TEACHER_GUARD_CANDIDATE_ONLY_RULES
    )
    hard_to_final_writeback_disabled = bool(
        teacher_guard_candidate_only
        and str(teacher_guard_action.get("rule", "")).startswith(
            "teacher_hard_to_final_"
        )
    )

    teacher_guard_writeback_applied = bool(
        teacher_distribution_guards_writeback_enabled
        and teacher_guard_action
        and not feature_quality_blocks_writeback
        and not teacher_guard_candidate_only
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
        else "拔高→压轴规则已关闭写回；仅保留候选动作与证据"
        if hard_to_final_writeback_disabled
        else "该规则当前仅记录候选动作，等待结合题目口径收紧后再开放写回"
        if teacher_guard_candidate_only
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
    elif teacher_guard_candidate_only:
        rating_result["feature_audit_flags"].append(
            (
                "拔高→压轴规则已关闭写回，仅审计候选："
                if hard_to_final_writeback_disabled
                else "规则当前仅审计，不自动写回最终档位："
            )
            + str(teacher_guard_action.get("rule", ""))
        )
    elif (
        teacher_guard_action
        and not teacher_guard_writeback_applied
        and not teacher_distribution_guards_writeback_enabled
    ):
        rating_result["feature_audit_flags"].append(
            "存在结构边界窄校准候选，但专用写回关闭；仅记录候选动作"
        )
    rating_result["feature_audit_flags"] = list(
        dict.fromkeys(rating_result["feature_audit_flags"])
    )
    return rating_result
