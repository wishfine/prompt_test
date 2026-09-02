# -*- coding: utf-8 -*-
"""高中化学两阶段难度 Pipeline 的纯函数核心。

本模块不依赖网络请求，集中处理：
1. 化学 feature schema 校验；
2. 高难特征严格触发与重复计数抑制；
3. 可选的 0.85 / 0.70 乘数效应和固定正确率分档；
4. 输入标签清洗、子题解析和图片充分性检查；
5. 第二阶段默认审计、启用时最多调整一档。
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


LEVEL_ORDER = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
LEVEL_INDEX = {value: index for index, value in enumerate(LEVEL_ORDER)}
CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER_ENABLED = True
PROGRAM_DERIVED_FEATURE_FIELDS = {
    "knowledge_L1",
    "knowledge_count",
    "knowledge_scope",
}

KNOWLEDGE_L1 = {
    "化学基本概念",
    "元素化学",
    "化学反应原理",
    "有机化学",
    "化学实验",
}

KNOWLEDGE_L2_TO_L1 = {
    "物质分类与化学用语": "化学基本概念",
    "物质的量与化学计量": "化学基本概念",
    "离子反应与氧化还原": "化学基本概念",
    "原子结构与元素周期律": "化学基本概念",
    "金属及其化合物": "元素化学",
    "非金属及其化合物": "元素化学",
    "化学反应与能量": "化学反应原理",
    "化学反应速率与平衡": "化学反应原理",
    "水溶液中的离子平衡": "化学反应原理",
    "电化学": "化学反应原理",
    "有机物结构与性质": "有机化学",
    "有机合成与推断": "有机化学",
    "基础实验操作": "化学实验",
    "物质检验、分离与提纯": "化学实验",
    "实验探究与方案设计": "化学实验",
}
KNOWLEDGE_L2 = set(KNOWLEDGE_L2_TO_L1)

CHEMISTRY_METHODS = {
    "守恒思想",
    "平衡思想",
    "结构决定性质",
    "定性与定量结合",
    "分类与讨论",
    "控制变量",
    "假设与验证",
    "等效与转化",
    "类比与迁移",
}

FEATURE_OPTIONS: dict[str, set[str]] = {
    "knowledge_count": {"1个", "2-3个", "4个及以上"},
    "knowledge_scope": {"单知识点", "同章节综合", "同模块跨章节", "跨模块综合"},
    "substance_count": {"1种", "2-3种", "4-6种", "7种及以上"},
    "substance_relation": {"单一物质", "相互独立", "同一反应体系", "前后转化依赖", "组成—性质—反应网络"},
    "reaction_count": {"0-1个", "2-3个", "4-6个", "7个及以上"},
    "reaction_relation": {"无反应链", "并列独立", "显性顺序衔接", "前后反应强依赖", "多路径反应网络"},
    "competing_reaction": {"无", "常规主反应判断", "竞争反应需要辨析", "副反应影响结论", "多反应竞争并需筛选"},
    "process_structure": {"单阶段", "两阶段显性流程", "多阶段显性流程", "多阶段强依赖", "循环或回流流程"},
    "primary_problem_structure": {"概念辨析", "直接计算", "无机推断", "有机推断", "反应原理综合", "实验探究", "工业流程", "有机合成", "复合题"},
    "step_count": {"1-2步", "3-5步", "6-8步", "9-12步", "12步以上"},
    "required_task_breadth": {"单一规则任务", "2-3个异质必要任务", "4个及以上异质必要任务", "多问递进任务链"},
    "subquestion_dependency": {"无多问", "相互独立", "后问依赖前问"},
    "model_explicitness": {"模型完全显性", "半隐含模型", "隐含模型", "需要自主建模"},
    "model_relation": {"单一模型", "同一模型多状态", "模型切换", "多模型耦合"},
    "reasoning_chain": {"直接套用", "简单因果", "多层因果", "逆向推理或临界分析"},
    "representation_conversion": {"无转换", "一次常规转换", "多次同类转换", "多表征连续转换", "逆向表征转换"},
    "evidence_relation": {"直接给定", "单证据对应", "多证据独立", "证据链相互支持", "证据冲突需排除"},
    "hidden_conditions": {"无", "单个隐含条件", "多个隐含条件"},
    "critical_condition": {"无临界", "显性给出临界", "需要推导过量不足边界", "隐含终点或有效区间"},
    "classification_discussion": {"无", "2类讨论", "3类讨论", "4类及以上"},
    "constraint_structure": {"无约束", "单一约束", "多约束相互独立", "多约束联合筛选"},
    "calculation_model": {"无定量计算", "常规化学计量", "多步化学计量", "浓度或气体综合", "平衡常数或Ka/Kb/Ksp", "多模型定量耦合"},
    "equation_structure": {"无方程", "单方程", "2-3个方程联立", "4个以上方程或不等式组"},
    "calculation_complexity": {"直接判断", "简单计算", "多步计算", "多方程联立", "参数或范围计算"},
    "parameter_operation": {"无参数", "单参数", "双参数", "多参数"},
    "information_carrier": {"纯文字", "单一图表", "实验装置", "工艺流程图", "光谱或图谱", "多载体综合"},
    "information_conversion": {"无信息转换", "直接读取", "单次关系转换", "多源信息联合转换", "流程或图谱反推"},
    "experiment_requirement": {"无", "基础操作或读数", "直接现象解释", "数据归纳", "控制变量或异常分析", "方案设计或误差反演"},
    "route_design_requirement": {"无", "已知路线补全", "路线比较选择", "合成路线设计", "分离提纯方案设计", "路线优化与可行性验证"},
    "context_type": {"纯化学", "工业生产", "实验探究", "有机合成", "生活环境与材料"},
    "context_load": {"纯包装", "简单规律映射", "需要信息转换", "需要自主情境建模"},
    "error_risk": {"无明显易错点", "轻微易错点", "明显易错点", "高易错点"},
}

REQUIRED_FEATURE_FIELDS = (
    "knowledge_L1",
    "knowledge_L2",
    "knowledge_points",
    *FEATURE_OPTIONS.keys(),
    "shared_model_across_subquestions",
    "chemistry_methods",
)

HIGH_DIFFICULTY_FEATURE_NAMES = (
    "多物质强耦合",
    "多反应或多阶段强耦合",
    "竞争反应与副反应判断",
    "多约束联合",
    "隐含临界或过量不足",
    "复杂分类讨论",
    "多模型或多平衡耦合",
    "复杂定量、参数或范围",
    "高层级信息转换",
    "跨模块深度综合",
    "高阶实验、合成或分离设计",
)


def build_stage1_output_schema() -> dict[str, Any]:
    """由本模块的规范枚举生成第一阶段严格输出 Schema。"""
    feature_properties: dict[str, Any] = {
        "knowledge_L1": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(KNOWLEDGE_L1)},
        },
        "knowledge_L2": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(KNOWLEDGE_L2)},
        },
        "knowledge_points": {"type": "array", "items": {"type": "string"}},
        "shared_model_across_subquestions": {"type": "boolean"},
        "chemistry_methods": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(CHEMISTRY_METHODS)},
        },
    }
    feature_properties.update({
        field: {"type": "string", "enum": sorted(options)}
        for field, options in FEATURE_OPTIONS.items()
    })
    return {
        "type": "object",
        "properties": {
            "features": {
                "type": "object",
                "properties": feature_properties,
                "required": list(REQUIRED_FEATURE_FIELDS),
                "additionalProperties": False,
            },
            "reason": {"type": "string"},
            "predicted_accuracy": {"type": "number"},
        },
        "required": ["features", "reason", "predicted_accuracy"],
        "additionalProperties": False,
    }


def build_stage2_output_schema() -> dict[str, Any]:
    """约束第二阶段固定枚举，避免复核结果引入非法特征名。"""
    string_or_boolean_or_string_array = {
        "anyOf": [
            {"type": "string"},
            {"type": "boolean"},
            {"type": "array", "items": {"type": "string"}},
        ]
    }
    return {
        "type": "object",
        "properties": {
            "difficulty_source": {"type": "string"},
            "feature_corrections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "field": {"type": "string"},
                        "from": string_or_boolean_or_string_array,
                        "to": string_or_boolean_or_string_array,
                        "evidence": {"type": "string"},
                    },
                    "required": ["field", "from", "to", "evidence"],
                    "additionalProperties": False,
                },
            },
            "missed_features": {"type": "array", "items": {"type": "string"}},
            "reviewed_high_difficulty_features": {
                "type": "array",
                "items": {"type": "string", "enum": list(HIGH_DIFFICULTY_FEATURE_NAMES)},
            },
            "high_feature_overlap_review": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "features": {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": list(HIGH_DIFFICULTY_FEATURE_NAMES),
                            },
                        },
                        "resolution": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["features", "resolution", "reason"],
                    "additionalProperties": False,
                },
            },
            "has_structural_revision": {"type": "boolean"},
            "adjacent_boundary_review": {
                "type": "object",
                "properties": {
                    "boundaries_checked": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["88边界", "85边界", "58边界", "38边界"],
                        },
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["维持", "应更简单一档", "应更难一档"],
                    },
                    "decisive_evidence": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": ["boundaries_checked", "verdict", "decisive_evidence"],
                "additionalProperties": False,
            },
            "reviewed_original_predicted_accuracy": {"type": "number"},
            "confidence": {"type": "string", "enum": ["高", "中", "低"]},
            "input_sufficiency_review": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["充分", "部分缺失", "信息不足"],
                    },
                    "missing_information": {
                        "type": "array", "items": {"type": "string"},
                    },
                },
                "required": ["status", "missing_information"],
                "additionalProperties": False,
            },
            "analysis": {"type": "string"},
        },
        "required": [
            "difficulty_source", "feature_corrections", "missed_features",
            "reviewed_high_difficulty_features", "high_feature_overlap_review",
            "has_structural_revision", "adjacent_boundary_review",
            "reviewed_original_predicted_accuracy", "confidence",
            "input_sufficiency_review", "analysis",
        ],
        "additionalProperties": False,
    }

MULTIPLIER_TRIGGER_COMBOS = (
    frozenset({"多反应或多阶段强耦合", "多约束联合", "高层级信息转换"}),
    frozenset({"多模型或多平衡耦合", "多约束联合", "高层级信息转换"}),
    frozenset({"竞争反应与副反应判断", "隐含临界或过量不足", "复杂分类讨论"}),
    frozenset({"高阶实验、合成或分离设计", "多约束联合", "多反应或多阶段强耦合"}),
)

QUESTION_MODEL_FIELDS = (
    "parent_id", "question_id", "stem", "options", "analysis", "structure_type",
    "sub_questions", "stem_image_url", "analysis_image_url", "stem_pic_url", "analysis_pic_url",
)
SUBQUESTION_MODEL_FIELDS = tuple(field for field in QUESTION_MODEL_FIELDS if field != "sub_questions")


@dataclass(frozen=True)
class HighDifficultyDetection:
    names: list[str]
    evidence: list[dict[str, Any]]
    possible_overlap_groups: list[list[str]]
    suppressed_overlaps: list[dict[str, Any]]


@dataclass(frozen=True)
class PreparedQuestion:
    question: dict[str, Any]
    source_difficulty_untrusted: Any
    input_quality: dict[str, Any]
    selected_image_urls: list[str]


@dataclass(frozen=True)
class FinalizationResult:
    final_level: str
    needs_manual_review: bool
    model_suggested_level: str
    adjustment_desc: str
    auto_adjustment_applied: bool


def map_accuracy_to_level(predicted_accuracy: Any) -> str:
    try:
        value = float(predicted_accuracy)
    except (TypeError, ValueError) as exc:
        raise ValueError("predicted_accuracy 必须为数值") from exc
    if not 0 <= value <= 100:
        raise ValueError("predicted_accuracy 必须在 0 到 100 之间")
    if value >= 88:
        return "难度1档"
    if value >= 85:
        return "难度2档"
    if value >= 58:
        return "难度3档"
    if value >= 38:
        return "难度4档"
    return "难度5档"


def multiplier_for_high_count(high_count: int) -> float:
    if high_count < 0:
        raise ValueError("high_count 不能为负数")
    if high_count >= 4:
        return 0.70
    if high_count >= 3:
        return 0.85
    return 1.0


def _matched_multiplier_trigger_combo(names: list[str]) -> list[str]:
    active = set(names)
    for combo in MULTIPLIER_TRIGGER_COMBOS:
        if combo.issubset(active):
            return [name for name in HIGH_DIFFICULTY_FEATURE_NAMES if name in combo]
    return []


def _is_full_dependent_organic_route(features: dict[str, Any]) -> bool:
    """识别“结构反推—中间体—路线设计”不可拆开的有机全链。

    路线图上的箭头显性，不等于各问的求解依赖弱。这里仅接受后问确实
    依赖前问、共享中间体、需要流程反推并完成新路线设计的严格组合；普通
    官能团识别或已知路线补空不会进入该通道。
    """
    return (
        features.get("primary_problem_structure") in {"有机合成", "有机推断"}
        and features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
        and features.get("subquestion_dependency") == "后问依赖前问"
        and features.get("shared_model_across_subquestions") is True
        and features.get("substance_relation") == "前后转化依赖"
        and features.get("reaction_count") in {"4-6个", "7个及以上"}
        and features.get("process_structure")
        in {"多阶段显性流程", "多阶段强依赖"}
        and features.get("information_conversion") == "流程或图谱反推"
        and features.get("evidence_relation") == "证据链相互支持"
        and features.get("route_design_requirement")
        in {"合成路线设计", "路线优化与可行性验证"}
    )


def _apply_chemistry_multiplier_policy(
    *,
    original_accuracy: float,
    high_names: list[str],
    multiplier_enabled: bool,
    full_dependent_organic_route: bool = False,
) -> dict[str, Any]:
    candidate = multiplier_for_high_count(len(high_names))
    matched_combo = (
        _matched_multiplier_trigger_combo(high_names)
        if multiplier_enabled
        else []
    )
    triggered = bool(matched_combo)
    applied = candidate if triggered else 1.0
    adjusted = round(original_accuracy * applied, 1)
    raw_level = map_accuracy_to_level(original_accuracy)
    adjusted_level = map_accuracy_to_level(adjusted)
    active = set(high_names)
    strong_final = (
        original_accuracy <= 52
        and (
            (
                "高阶实验、合成或分离设计" in active
                and "多约束联合" in active
            )
            or full_dependent_organic_route
        )
    )
    final_guard = (
        raw_level == "难度4档"
        and adjusted_level == "难度5档"
        and not strong_final
    )
    if final_guard:
        adjusted = 38.0
    return {
        "multiplier_candidate": candidate,
        "multiplier_triggered": triggered,
        "multiplier_trigger_combo": matched_combo,
        "multiplier_applied": applied,
        "multiplier_final_level_guard_applied": final_guard,
        "adjusted_accuracy": adjusted,
    }


def _chemistry_58_boundary_promotion_candidate(
    *,
    current_level: str,
    original_accuracy: float,
    features: dict[str, Any],
) -> bool:
    if current_level != "难度3档":
        return False
    near_58_boundary = 58 <= original_accuracy <= 62
    information_chain = near_58_boundary and (
        features.get("reasoning_chain") == "多层因果"
        and features.get("information_conversion")
        in {"多源信息联合转换", "流程或图谱反推"}
    )
    dependent_model_chain = near_58_boundary and (
        (
            features.get("subquestion_dependency") == "后问依赖前问"
            or features.get("shared_model_across_subquestions") is True
        )
        and features.get("model_relation") in {"模型切换", "多模型耦合"}
        and features.get("process_structure")
        in {"多阶段强依赖", "多阶段显性流程"}
    )
    representation_chain_evidence = (
        (
            features.get("substance_relation") == "同一反应体系"
            and features.get("reaction_count") in {"2-3个", "4-6个", "7个及以上"}
        )
        or features.get("calculation_model")
        in {"平衡常数或Ka/Kb/Ksp", "多模型定量耦合"}
        or features.get("information_conversion")
        in {"多源信息联合转换", "流程或图谱反推"}
    )
    representation_chain = (
        58 <= original_accuracy <= 68
        and features.get("reasoning_chain") == "多层因果"
        and features.get("representation_conversion") == "一次常规转换"
        and representation_chain_evidence
    )
    return information_chain or dependent_model_chain or representation_chain


def _structural_cluster_signals(features: dict[str, Any]) -> list[str]:
    """返回可审计的中高档结构类别；不使用题长、选项数或普通活跃特征。"""
    signals = [
        (
            "体系耦合",
            features.get("substance_relation")
            in {"前后转化依赖", "组成—性质—反应网络"},
        ),
        (
            "反应链",
            features.get("reaction_relation")
            in {"前后反应强依赖", "多路径反应网络"},
        ),
        (
            "多阶段流程",
            features.get("process_structure")
            in {"多阶段显性流程", "多阶段强依赖", "循环或回流流程"},
        ),
        (
            "模型迁移",
            features.get("model_relation") in {"模型切换", "多模型耦合"},
        ),
        (
            "高层表征转换",
            features.get("representation_conversion")
            in {"多次同类转换", "多表征连续转换", "逆向表征转换"},
        ),
        (
            "证据链",
            features.get("evidence_relation")
            in {"证据链相互支持", "证据冲突需排除"},
        ),
        (
            "隐含边界",
            features.get("hidden_conditions") == "多个隐含条件"
            or features.get("critical_condition")
            in {"需要推导过量不足边界", "隐含终点或有效区间"},
        ),
        (
            "联合约束",
            features.get("constraint_structure") == "多约束联合筛选",
        ),
        (
            "复杂定量",
            features.get("calculation_model")
            in {"平衡常数或Ka/Kb/Ksp", "多模型定量耦合"}
            or features.get("calculation_complexity")
            in {"多方程联立", "参数或范围计算"},
        ),
        (
            "高阶实验或路线",
            features.get("experiment_requirement")
            in {"控制变量或异常分析", "方案设计或误差反演"}
            or features.get("route_design_requirement")
            in {"合成路线设计", "分离提纯方案设计", "路线优化与可行性验证"},
        ),
    ]
    return [name for name, matched in signals if matched]


def _chemistry_structural_cluster_promotion_candidate(
    *,
    current_level: str,
    original_accuracy: float,
    features: dict[str, Any],
) -> tuple[bool, list[str]]:
    """用多个中等强度结构识别3→4边界，避免单字段宽泛升档。"""
    signals = _structural_cluster_signals(features)
    active = set(signals)
    if current_level != "难度3档" or not 58 <= original_accuracy <= 68:
        return False, signals

    model_with_multistage = {"模型迁移", "多阶段流程"}.issubset(active)
    complex_quantitative = "复杂定量" in active
    model_with_coupling = {"模型迁移", "体系耦合"}.issubset(active)
    dense_reaction_cluster = (
        {"体系耦合", "反应链"}.issubset(active) and len(active) >= 5
    )
    return (
        model_with_multistage
        or complex_quantitative
        or model_with_coupling
        or dense_reaction_cluster,
        signals,
    )


def _ensure_unique_strings(value: Any, field: str, *, nonempty: bool) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} 必须为列表")
    if nonempty and not value:
        raise ValueError(f"{field} 不得为空")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} 每项必须为非空字符串")
    if len(value) != len(set(value)):
        raise ValueError(f"{field} 不得包含重复值")
    return value


def validate_feature_schema(features: dict[str, Any]) -> None:
    if not isinstance(features, dict):
        raise ValueError("features 必须为对象")
    missing = [field for field in REQUIRED_FEATURE_FIELDS if field not in features]
    if missing:
        raise ValueError(f"features 缺少字段：{', '.join(missing)}")

    l1 = _ensure_unique_strings(features["knowledge_L1"], "knowledge_L1", nonempty=True)
    l2 = _ensure_unique_strings(features["knowledge_L2"], "knowledge_L2", nonempty=True)
    points = _ensure_unique_strings(features["knowledge_points"], "knowledge_points", nonempty=True)
    methods = _ensure_unique_strings(features["chemistry_methods"], "chemistry_methods", nonempty=False)
    invalid_l1 = [value for value in l1 if value not in KNOWLEDGE_L1]
    invalid_l2 = [value for value in l2 if value not in KNOWLEDGE_L2]
    invalid_methods = [value for value in methods if value not in CHEMISTRY_METHODS]
    if invalid_l1:
        raise ValueError(f"knowledge_L1 含非法值：{invalid_l1}")
    if invalid_l2:
        raise ValueError(f"knowledge_L2 含非法值：{invalid_l2}")
    if invalid_methods:
        raise ValueError(f"chemistry_methods 含非法值：{invalid_methods}")
    derived_l1 = {KNOWLEDGE_L2_TO_L1[value] for value in l2}
    if derived_l1 != set(l1):
        raise ValueError(
            "knowledge_L1 与 knowledge_L2 不一致；"
            f"L2 实际归属 {sorted(derived_l1)}，L1 为 {sorted(l1)}"
        )
    if not points:
        raise ValueError("knowledge_points 不得为空")
    if not isinstance(features["shared_model_across_subquestions"], bool):
        raise ValueError("shared_model_across_subquestions 必须为布尔值")
    for field, options in FEATURE_OPTIONS.items():
        if not isinstance(features[field], str) or features[field] not in options:
            raise ValueError(f"{field} 非法值 {features[field]!r}；允许值：{sorted(options)}")


def detect_stage1_score_feature_conflicts(
    stage1_rating: dict[str, Any],
) -> list[dict[str, Any]]:
    """检测明确的第一阶段证据—分数矛盾，不直接修改任一侧。"""
    features = stage1_rating.get("features")
    if not isinstance(features, dict):
        return []
    try:
        accuracy = float(stage1_rating["predicted_accuracy"])
    except (KeyError, TypeError, ValueError):
        return []

    high_count = len(detect_high_difficulty_features(features).names)
    common_direct = (
        features.get("step_count") == "1-2步"
        and features.get("model_explicitness") == "模型完全显性"
        and features.get("model_relation") == "单一模型"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("representation_conversion") == "无转换"
        and features.get("evidence_relation") in {"直接给定", "单证据对应"}
        and features.get("calculation_model") == "无定量计算"
        and features.get("calculation_complexity") == "直接判断"
        and features.get("information_conversion") in {"无信息转换", "直接读取"}
        and features.get("experiment_requirement") == "无"
        and features.get("route_design_requirement") == "无"
        and features.get("process_structure") == "单阶段"
        and features.get("subquestion_dependency") in {"无多问", "相互独立"}
        and features.get("shared_model_across_subquestions") is False
        and features.get("constraint_structure")
        in {"无约束", "单一约束", "多约束相互独立"}
        and high_count == 0
    )
    single_rule_direct = (
        common_direct
        and features.get("required_task_breadth") == "单一规则任务"
    )
    independent_basic = (
        common_direct
        and features.get("required_task_breadth")
        in {"2-3个异质必要任务", "4个及以上异质必要任务"}
    )

    issues: list[dict[str, Any]] = []
    if single_rule_direct and accuracy < 88:
        issues.append({
            "boundary": "88边界",
            "problem": "features 表示单一规则、直接显性的送分结构，但 predicted_accuracy 低于88",
            "evidence": [
                "step_count=1-2步",
                "required_task_breadth=单一规则任务",
                "model_explicitness=模型完全显性",
                "reasoning_chain=直接套用",
                "无表征转换、无定量计算、无高难特征",
            ],
            "repairable": False,
        })
    if independent_basic and accuracy < 85:
        issues.append({
            "boundary": "85边界",
            "problem": "features 表示可直接开始的独立基础应用，但 predicted_accuracy 低于85",
            "evidence": [
                f"step_count={features.get('step_count')}",
                f"required_task_breadth={features.get('required_task_breadth')}",
                f"model_explicitness={features.get('model_explicitness')}",
                f"reasoning_chain={features.get('reasoning_chain')}",
                "无串行依赖、无联合约束、无高难特征",
            ],
            "repairable": True,
        })

    reason = str(stage1_rating.get("reason") or "")
    compact_reason = re.sub(r"\s+", "", reason)

    def has_unnegated(pattern: str) -> bool:
        for match in re.finditer(pattern, compact_reason):
            prefix = compact_reason[max(0, match.start() - 8):match.start()]
            if not re.search(
                r"(?:未达到|达不到|不能达到|无法达到|不属于|未进入|不应进入)$",
                prefix,
            ):
                return True
        return False

    explicit_bands = []
    band_patterns = (
        ("88及以上", 88.0, 100.0, r"(?:88分?及以上|不低于88|正确率(?:达到|为)88%以上)"),
        ("85—88", 85.0, 88.0, r"(?:85(?:分)?[—–~～-]88|85至88)(?:分|%|区间)?"),
        ("58—85", 58.0, 85.0, r"(?:58(?:分)?[—–~～-]85|58至85)(?:分|%|区间)?"),
        ("38—58", 38.0, 58.0, r"(?:38(?:分)?[—–~～-]58|38至58)(?:分|%|区间)?"),
        ("低于38", 0.0, 38.0, r"(?:低于|小于)38(?:分|%|区间)?"),
    )
    for name, lower, upper, pattern in band_patterns:
        if has_unnegated(pattern):
            explicit_bands.append((name, lower, upper))
    # 只有 reason 明确落在唯一分数区间时才据此触发复核；若同时讨论
    # 多个相邻边界，不能把正常的边界比较误判成最终结论。
    if len(explicit_bands) == 1:
        name, lower, upper = explicit_bands[0]
        in_band = lower <= accuracy <= upper if name == "88及以上" else lower <= accuracy < upper
        if not in_band:
            issues.append({
                "boundary": "reason—分数一致性",
                "problem": (
                    f"reason 明确将题目置于{name}区间，但 predicted_accuracy={accuracy:g} 不在该区间"
                ),
                "evidence": [
                    f"reason_explicit_band={name}",
                    f"predicted_accuracy={accuracy:g}",
                ],
                "repairable": True,
            })

    directional_claims = (
        ("reason 明确认为不应低于58", accuracy < 58, r"(?:未进入|不属于)58分?以下|(?:高于|不低于)58分?"),
        ("reason 明确认为不应低于85", accuracy < 85, r"(?:未进入|不属于)85分?以下|(?:高于|不低于)85分?"),
    )
    for problem, contradicted, pattern in directional_claims:
        if contradicted and re.search(pattern, compact_reason):
            issues.append({
                "boundary": "reason—分数一致性",
                "problem": f"{problem}，但 predicted_accuracy={accuracy:g}",
                "evidence": [f"predicted_accuracy={accuracy:g}", "reason含明确的相邻边界排除结论"],
                "repairable": True,
            })

    deduplicated: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for issue in issues:
        key = (str(issue.get("boundary")), str(issue.get("problem")))
        if key not in seen:
            seen.add(key)
            deduplicated.append(issue)
    return deduplicated


def score_feature_repair_crosses_boundary(
    initial_issues: list[dict[str, Any]],
    repaired_accuracy: float,
) -> bool:
    """85边界修复不得越过88边界进入1档。"""
    return repaired_accuracy >= 88 and any(
        issue.get("boundary") == "85边界"
        and issue.get("repairable") is True
        for issue in initial_issues
    )


def validate_structural_revision_evidence(verification: dict[str, Any]) -> None:
    """结构修订必须对应可审计的 feature 变化，禁止只改分析措辞。"""
    if verification.get("has_structural_revision") is not True:
        return

    corrections = verification.get("feature_corrections")
    has_correction = isinstance(corrections, list) and bool(corrections)

    missed = verification.get("missed_features")
    has_real_omission = isinstance(missed, list) and any(
        isinstance(item, str)
        and item.strip()
        and item.strip().lower() not in {"无", "无遗漏", "none", "n/a"}
        for item in missed
    )
    if not has_correction and not has_real_omission:
        raise ValueError(
            "has_structural_revision=true 时必须提供具体 feature 结构修订："
            "feature_corrections 不得为空，或 missed_features 必须包含真实遗漏特征"
        )


def detect_active_features(features: dict[str, Any]) -> list[str]:
    """普通活跃特征只用于复核，不参与乘数选择。"""
    gates = [
        (features.get("knowledge_scope") != "单知识点", "知识综合"),
        (features.get("substance_count") != "1种", "多物质"),
        (features.get("reaction_count") != "0-1个", "多反应"),
        (features.get("process_structure") != "单阶段", "多阶段"),
        (features.get("step_count") != "1-2步", "多步骤"),
        (features.get("required_task_breadth") != "单一规则任务", "多个不可合并的必要化学判断"),
        (features.get("subquestion_dependency") != "无多问", "多小问"),
        (bool(features.get("shared_model_across_subquestions")), "小问共享模型"),
        (features.get("model_explicitness") != "模型完全显性", "模型隐含"),
        (features.get("model_relation") != "单一模型", "模型变换"),
        (features.get("reasoning_chain") not in {"直接套用", "简单因果"}, "长推理链"),
        (features.get("representation_conversion") != "无转换", "表征转换"),
        (features.get("evidence_relation") not in {"直接给定", "单证据对应"}, "多证据"),
        (features.get("hidden_conditions") != "无", "隐含条件"),
        (features.get("critical_condition") != "无临界", "临界条件"),
        (features.get("classification_discussion") != "无", "分类讨论"),
        (features.get("constraint_structure") != "无约束", "约束"),
        (bool(features.get("chemistry_methods")), "化学思想方法"),
        (features.get("calculation_model") != "无定量计算", "定量计算"),
        (features.get("information_conversion") not in {"无信息转换", "直接读取"}, "信息转换"),
        (features.get("experiment_requirement") != "无", "实验任务"),
        (features.get("route_design_requirement") != "无", "路线任务"),
        (features.get("context_load") not in {"纯包装", "简单规律映射"}, "情境转换"),
    ]
    return [name for active, name in gates if active]


def _evidence(name: str, fields: list[str], features: dict[str, Any], node: str) -> dict[str, Any]:
    return {
        "name": name,
        "fields": fields,
        "values": [features.get(field) for field in fields],
        "decision_node": node,
    }


def detect_high_difficulty_features(features: dict[str, Any]) -> HighDifficultyDetection:
    """仅计算结构性高难类别，并对同一决策节点的重叠信号做确定性抑制。"""
    evidence_by_name: dict[str, dict[str, Any]] = {}
    suppressed: list[dict[str, Any]] = []
    full_dependent_organic_route = _is_full_dependent_organic_route(features)

    multi_substance = (
        features.get("substance_count") in {"4-6种", "7种及以上"}
        and features.get("substance_relation") == "组成—性质—反应网络"
        and (
            features.get("model_relation") == "多模型耦合"
            or (
                features.get("reaction_relation") == "多路径反应网络"
                and features.get("evidence_relation") in {"证据链相互支持", "证据冲突需排除"}
            )
        )
    )
    if multi_substance:
        fields = ["substance_count", "substance_relation", "reaction_relation", "evidence_relation", "model_relation"]
        evidence_by_name["多物质强耦合"] = _evidence("多物质强耦合", fields, features, "substance_network")

    multi_reaction = (
        (
            features.get("reaction_relation") in {"前后反应强依赖", "多路径反应网络"}
            and features.get("process_structure") in {"多阶段强依赖", "循环或回流流程"}
            and (
                features.get("reaction_count") in {"4-6个", "7个及以上"}
                or features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
            )
        )
        or full_dependent_organic_route
    )
    if multi_reaction:
        fields = ["reaction_count", "reaction_relation", "process_structure"]
        node = "reaction_process_network"
        if full_dependent_organic_route:
            fields = [
                "subquestion_dependency",
                "shared_model_across_subquestions",
                "substance_relation",
                "reaction_count",
                "process_structure",
                "information_conversion",
                "evidence_relation",
                "route_design_requirement",
            ]
            node = "dependent_organic_route_chain"
        evidence_by_name["多反应或多阶段强耦合"] = _evidence("多反应或多阶段强耦合", fields, features, node)

    competition_high = (
        features.get("competing_reaction") in {"竞争反应需要辨析", "副反应影响结论", "多反应竞争并需筛选"}
        and (
            features.get("evidence_relation") == "证据冲突需排除"
            or features.get("classification_discussion") != "无"
            or features.get("hidden_conditions") != "无"
        )
    )
    if competition_high:
        fields = ["competing_reaction", "evidence_relation", "classification_discussion", "hidden_conditions"]
        evidence_by_name["竞争反应与副反应判断"] = _evidence("竞争反应与副反应判断", fields, features, "reaction_branch_selection")

    if features.get("constraint_structure") == "多约束联合筛选":
        evidence_by_name["多约束联合"] = _evidence("多约束联合", ["constraint_structure"], features, "joint_constraints")

    critical_high = (
        features.get("critical_condition") in {"需要推导过量不足边界", "隐含终点或有效区间"}
        and features.get("hidden_conditions") != "无"
        and (
            features.get("reasoning_chain") == "逆向推理或临界分析"
            or features.get("calculation_complexity") == "参数或范围计算"
            or features.get("classification_discussion") != "无"
        )
    )
    same_reaction_branch = competition_high and features.get("critical_condition") == "需要推导过量不足边界"
    if critical_high and same_reaction_branch:
        suppressed.append({"suppressed": "隐含临界或过量不足", "kept": "竞争反应与副反应判断", "reason": "同一反应分支筛选决策节点"})
    elif critical_high:
        fields = ["critical_condition", "hidden_conditions", "reasoning_chain", "calculation_complexity"]
        evidence_by_name["隐含临界或过量不足"] = _evidence("隐含临界或过量不足", fields, features, "hidden_boundary")

    classification_high = (
        features.get("classification_discussion") in {"3类讨论", "4类及以上"}
        or (
            features.get("classification_discussion") == "2类讨论"
            and features.get("model_relation") in {"模型切换", "多模型耦合"}
            and (
                features.get("competing_reaction") != "无"
                or features.get("critical_condition") != "无临界"
            )
        )
    )
    if classification_high and (
        competition_high or (critical_high and features.get("classification_discussion") == "2类讨论")
    ):
        kept = "竞争反应与副反应判断" if competition_high else "隐含临界或过量不足"
        suppressed.append({"suppressed": "复杂分类讨论", "kept": kept, "reason": "分类仅由同一竞争或临界节点产生"})
    elif classification_high:
        fields = ["classification_discussion", "model_relation", "competing_reaction", "critical_condition"]
        evidence_by_name["复杂分类讨论"] = _evidence("复杂分类讨论", fields, features, "complex_classification")

    cross_module = (
        features.get("knowledge_scope") == "跨模块综合"
        and features.get("model_relation") in {"模型切换", "多模型耦合"}
        and features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
    )
    model_high = (
        features.get("model_relation") in {"模型切换", "多模型耦合"}
        and (
            features.get("equation_structure") in {"2-3个方程联立", "4个以上方程或不等式组"}
            or features.get("process_structure") in {"多阶段强依赖", "循环或回流流程"}
            or features.get("reaction_relation") == "多路径反应网络"
            or (
                features.get("calculation_model") == "多模型定量耦合"
                and features.get("calculation_complexity")
                in {"多步计算", "多方程联立", "参数或范围计算"}
                and features.get("step_count")
                in {"6-8步", "9-12步", "12步以上"}
            )
        )
    )
    if cross_module:
        fields = ["knowledge_scope", "model_relation", "step_count"]
        evidence_by_name["跨模块深度综合"] = _evidence("跨模块深度综合", fields, features, "cross_module_bridge")
    if model_high and not (cross_module and features.get("model_relation") == "模型切换"):
        fields = ["model_relation", "equation_structure", "process_structure", "reaction_relation"]
        evidence_by_name["多模型或多平衡耦合"] = _evidence("多模型或多平衡耦合", fields, features, "model_coupling")
    elif model_high:
        suppressed.append({"suppressed": "多模型或多平衡耦合", "kept": "跨模块深度综合", "reason": "普通模型切换只用于跨模块桥接"})

    quantitative_high = (
        features.get("calculation_model") in {"平衡常数或Ka/Kb/Ksp", "多模型定量耦合"}
        and features.get("calculation_complexity") in {"多方程联立", "参数或范围计算"}
        and features.get("equation_structure") in {"2-3个方程联立", "4个以上方程或不等式组"}
        and (
            features.get("parameter_operation") in {"双参数", "多参数"}
            or features.get("critical_condition") != "无临界"
        )
    )
    if quantitative_high:
        fields = ["calculation_model", "calculation_complexity", "equation_structure", "parameter_operation", "critical_condition"]
        evidence_by_name["复杂定量、参数或范围"] = _evidence("复杂定量、参数或范围", fields, features, "quantitative_parameter")

    information_high = (
        features.get("information_conversion") in {"多源信息联合转换", "流程或图谱反推"}
        and features.get("evidence_relation") in {"证据链相互支持", "证据冲突需排除"}
        and features.get("reasoning_chain") in {"多层因果", "逆向推理或临界分析"}
    )
    if information_high:
        fields = ["information_carrier", "information_conversion", "evidence_relation", "reasoning_chain"]
        evidence_by_name["高层级信息转换"] = _evidence("高层级信息转换", fields, features, "information_inference")

    design_high = (
        features.get("experiment_requirement") in {"控制变量或异常分析", "方案设计或误差反演"}
        or features.get("route_design_requirement") in {"合成路线设计", "分离提纯方案设计", "路线优化与可行性验证"}
    ) and (
        features.get("reasoning_chain") in {"多层因果", "逆向推理或临界分析"}
        and (
            features.get("constraint_structure") == "多约束联合筛选"
            or features.get("evidence_relation") in {"证据链相互支持", "证据冲突需排除"}
        )
    )
    if design_high:
        fields = ["experiment_requirement", "route_design_requirement", "reasoning_chain", "constraint_structure", "evidence_relation"]
        evidence_by_name["高阶实验、合成或分离设计"] = _evidence("高阶实验、合成或分离设计", fields, features, "advanced_design")

    # 若“高层信息转换”仅是同一实验/路线设计中的证据处理，不重复计数。
    if (
        design_high
        and information_high
        and features.get("information_carrier") in {"实验装置", "工艺流程图"}
        and not full_dependent_organic_route
    ):
        evidence_by_name.pop("高层级信息转换", None)
        suppressed.append({"suppressed": "高层级信息转换", "kept": "高阶实验、合成或分离设计", "reason": "信息转换只是同一设计节点的内部证据处理"})

    names = [name for name in HIGH_DIFFICULTY_FEATURE_NAMES if name in evidence_by_name]
    overlaps: list[list[str]] = []
    for index, left in enumerate(names):
        left_fields = set(evidence_by_name[left]["fields"])
        for right in names[index + 1 :]:
            if left_fields.intersection(evidence_by_name[right]["fields"]):
                overlaps.append([left, right])
    return HighDifficultyDetection(
        names=names,
        evidence=[evidence_by_name[name] for name in names],
        possible_overlap_groups=overlaps,
        suppressed_overlaps=suppressed,
    )


def _apply_stage1_structural_level_guards(
    *,
    level: str,
    original_accuracy: float,
    features: dict[str, Any],
    high_names: list[str],
) -> tuple[str, list[dict[str, Any]]]:
    """回收已验证的相邻档位标尺偏差；每题至多调整一档。"""
    actions: list[dict[str, Any]] = []
    final_level = level

    if (
        level == "难度1档"
        and features.get("required_task_breadth")
        in {"2-3个异质必要任务", "4个及以上异质必要任务", "多问递进任务链"}
    ):
        final_level = "难度2档"
        actions.append({
            "rule": "multiple_required_tasks_level_one_floor",
            "from": level,
            "to": final_level,
            "evidence": [features.get("required_task_breadth")],
        })
    elif (
        level == "难度1档"
        and features.get("calculation_model") == "常规化学计量"
    ):
        final_level = "难度2档"
        actions.append({
            "rule": "standard_stoichiometry_level_one_floor",
            "from": level,
            "to": final_level,
            "evidence": ["常规化学计量"],
        })
    elif (
        level == "难度1档"
        and features.get("primary_problem_structure") == "概念辨析"
        and features.get("knowledge_count") == "4个及以上"
        and features.get("substance_relation") == "相互独立"
    ):
        final_level = "难度2档"
        actions.append({
            "rule": "independent_multi_concept_level_one_floor",
            "from": level,
            "to": final_level,
            "evidence": ["概念辨析", "4个及以上独立知识点", "相互独立"],
        })
    elif (
        level == "难度3档"
        and 80 <= original_accuracy <= 83
        and features.get("primary_problem_structure") == "概念辨析"
        and features.get("step_count") == "1-2步"
        and features.get("substance_relation") == "相互独立"
        and features.get("process_structure") == "单阶段"
        and features.get("representation_conversion") == "无转换"
        and not high_names
    ):
        final_level = "难度2档"
        actions.append({
            "rule": "low_structure_independent_concept_recovery",
            "from": level,
            "to": final_level,
            "evidence": ["1-2步", "相互独立", "无转换", "无高难结构"],
        })
    elif (
        level == "难度2档"
        and 85 <= original_accuracy < 88
        and features.get("step_count") == "3-5步"
        and features.get("substance_count") == "2-3种"
        and features.get("substance_relation") == "同一反应体系"
        and features.get("calculation_model") == "常规化学计量"
    ):
        final_level = "难度3档"
        actions.append({
            "rule": "standard_stoichiometry_chain_level_two_floor",
            "from": level,
            "to": final_level,
            "evidence": ["3-5步", "同一反应体系", "常规化学计量"],
        })
    return final_level, actions


def enrich_stage1_rating(
    stage1_rating: dict[str, Any],
    *,
    features_model_raw: dict[str, Any] | None = None,
    normalization_log: list[dict[str, Any]] | None = None,
    multiplier_enabled: bool | None = None,
) -> dict[str, Any]:
    rating = copy.deepcopy(stage1_rating)
    features = rating.get("features")
    validate_feature_schema(features)
    rating["features_model_raw"] = copy.deepcopy(
        features if features_model_raw is None else features_model_raw
    )
    rating["enum_normalization_log"] = copy.deepcopy(normalization_log or [])
    rating["enum_normalization_applied"] = bool(normalization_log)
    try:
        raw_accuracy = float(rating["predicted_accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("第一阶段 predicted_accuracy 缺失或不是数值") from exc
    if not 0 <= raw_accuracy <= 100:
        raise ValueError("第一阶段 predicted_accuracy 必须在 0 到 100 之间")

    distinct_points = list(dict.fromkeys(str(value).strip() for value in features["knowledge_points"]))
    features["knowledge_points"] = distinct_points
    features["knowledge_count"] = "1个" if len(distinct_points) == 1 else ("2-3个" if len(distinct_points) <= 3 else "4个及以上")
    # 实验是任务形式与方法维度，不能仅因“内容模块+实验”
    # 就把知识范围推导成跨模块综合。
    content_l1 = {
        value for value in features["knowledge_L1"] if value != "化学实验"
    }
    content_l2 = {
        value
        for value in features["knowledge_L2"]
        if KNOWLEDGE_L2_TO_L1[value] != "化学实验"
    }
    if len(content_l1) >= 2:
        features["knowledge_scope"] = "跨模块综合"
    elif len(content_l2) >= 2:
        features["knowledge_scope"] = "同模块跨章节"
    elif len(distinct_points) >= 2:
        features["knowledge_scope"] = "同章节综合"
    else:
        features["knowledge_scope"] = "单知识点"

    high = detect_high_difficulty_features(features)
    active = detect_active_features(features)
    high_count = len(high.names)
    enabled = (
        CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER_ENABLED
        if multiplier_enabled is None
        else bool(multiplier_enabled)
    )
    multiplier_policy = _apply_chemistry_multiplier_policy(
        original_accuracy=raw_accuracy,
        high_names=high.names,
        multiplier_enabled=enabled,
        full_dependent_organic_route=_is_full_dependent_organic_route(features),
    )
    multiplier_candidate = multiplier_policy["multiplier_candidate"]
    multiplier = multiplier_policy["multiplier_applied"]
    adjusted = multiplier_policy["adjusted_accuracy"]
    rating["original_predicted_accuracy"] = raw_accuracy
    rating["active_features"] = active
    rating["active_feature_count"] = len(active)
    rating["high_difficulty_features"] = high.names
    rating["high_difficulty_feature_evidence"] = high.evidence
    rating["possible_high_feature_overlaps"] = high.possible_overlap_groups
    rating["suppressed_high_feature_overlaps"] = high.suppressed_overlaps
    rating["high_difficulty_feature_count"] = high_count
    rating["high_difficulty_multiplier_enabled"] = enabled
    rating["multiplier_candidate"] = multiplier_candidate
    rating["multiplier_triggered"] = multiplier_policy["multiplier_triggered"]
    rating["multiplier_trigger_combo"] = multiplier_policy["multiplier_trigger_combo"]
    rating["multiplier_final_level_guard_applied"] = multiplier_policy[
        "multiplier_final_level_guard_applied"
    ]
    rating["multiplier_applied"] = multiplier
    rating["predicted_accuracy"] = adjusted
    raw_step1_level = map_accuracy_to_level(adjusted)
    guarded_level, guard_actions = _apply_stage1_structural_level_guards(
        level=raw_step1_level,
        original_accuracy=raw_accuracy,
        features=features,
        high_names=high.names,
    )
    rating["difficulty_level_step1_before_structural_guards"] = raw_step1_level
    rating["stage1_structural_guard_actions"] = guard_actions
    rating["difficulty_level_step1"] = guarded_level
    rating["full_dependent_organic_route_detected"] = _is_full_dependent_organic_route(features)
    return rating


def _unwrap_scalar_wrapper(value: Any) -> tuple[Any, bool]:
    """解开严格 JSON 输出偶发生成的单值对象包装，不猜测歧义对象。"""
    if not isinstance(value, dict):
        return value, False
    candidate = value.get("value")
    if isinstance(candidate, (str, bool, int, float)):
        return candidate, True
    if len(value) == 1:
        candidate = next(iter(value.values()))
        if isinstance(candidate, (str, bool, int, float)):
            return candidate, True
    return value, False


def build_neutral_stage2_verification(
    stage1: dict[str, Any],
    *,
    validation_error: str,
) -> dict[str, Any]:
    """Stage2 响应缺字段时生成可审计的维持结论，不采纳局部修正。"""
    level = str(stage1["difficulty_level_step1"])
    boundaries = {
        "难度1档": ["88边界"],
        "难度2档": ["88边界", "85边界"],
        "难度3档": ["85边界", "58边界"],
        "难度4档": ["58边界", "38边界"],
        "难度5档": ["38边界"],
    }.get(level, ["85边界"])
    return {
        "difficulty_source": "第二阶段输出不完整，未采纳局部复核内容，保持第一阶段结论",
        "feature_corrections": [],
        "missed_features": [],
        "reviewed_high_difficulty_features": copy.deepcopy(
            stage1.get("high_difficulty_features") or []
        ),
        "high_feature_overlap_review": [],
        "has_structural_revision": False,
        "adjacent_boundary_review": {
            "boundaries_checked": boundaries,
            "verdict": "维持",
            "decisive_evidence": ["第二阶段未返回完整必填字段，不能安全授权结构修正或改分"],
        },
        "reviewed_original_predicted_accuracy": float(
            stage1["original_predicted_accuracy"]
        ),
        "confidence": "低",
        "input_sufficiency_review": {
            "status": "充分",
            "missing_information": [],
        },
        "analysis": "第二阶段响应字段不完整。为避免使用局部JSON造成错误修正，本次忽略全部局部复核内容，并按第一阶段features、原始正确率和高难特征维持结果。",
        "stage2_output_recovered": True,
        "stage2_recovery_reason": validation_error,
    }


def normalize_stage1_rating(
    stage1_rating: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """对少量稳定、无歧义的格式偏差做兼容；不替模型改特征。"""
    if not isinstance(stage1_rating, dict):
        raise ValueError("第一阶段响应必须为对象")
    normalized = copy.deepcopy(stage1_rating)
    features = normalized.get("features")
    if not isinstance(features, dict):
        raise ValueError("第一阶段缺少 features 对象")
    log: list[dict[str, Any]] = []

    for field in FEATURE_OPTIONS:
        raw_value = features.get(field)
        unwrapped, changed = _unwrap_scalar_wrapper(raw_value)
        if changed:
            features[field] = unwrapped
            log.append({
                "field": field,
                "from": copy.deepcopy(raw_value),
                "to": copy.deepcopy(unwrapped),
                "reason": "单值对象包装确定性解包",
            })

    # 与原方案一致：知识点、二级模块和方法的完全重复不改变题目结构，
    # 在 schema 校验前确定性去重，避免模型重复列举导致整题失败。
    for field in ("knowledge_L2", "knowledge_points", "chemistry_methods"):
        values = features.get(field)
        if not isinstance(values, list):
            continue
        cleaned: list[str] = []
        for value in values:
            if not isinstance(value, str) or not value.strip():
                cleaned = []
                break
            normalized_value = value.strip()
            if normalized_value not in cleaned:
                cleaned.append(normalized_value)
        if cleaned and cleaned != values:
            features[field] = cleaned
            log.append(
                {
                    "field": field,
                    "from": copy.deepcopy(values),
                    "to": copy.deepcopy(cleaned),
                    "reason": "完全重复项确定性去重",
                }
            )

    raw_shared = features.get("shared_model_across_subquestions")
    shared, shared_unwrapped = _unwrap_scalar_wrapper(raw_shared)
    if shared_unwrapped:
        features["shared_model_across_subquestions"] = shared
        log.append({
            "field": "shared_model_across_subquestions",
            "from": copy.deepcopy(raw_shared),
            "to": copy.deepcopy(shared),
            "reason": "单值对象包装确定性解包",
        })
    if shared in ("是", "true", "True", 1):
        features["shared_model_across_subquestions"] = True
        log.append({"field": "shared_model_across_subquestions", "from": shared, "to": True})
    elif shared in ("否", "false", "False", 0):
        features["shared_model_across_subquestions"] = False
        log.append({"field": "shared_model_across_subquestions", "from": shared, "to": False})

    # L1 是 L2 的确定性父级，不应让模型重复填写造成整题失败。
    # 只有当 L2 本身完整合法时才自动派生；非法或缺失 L2 仍交给校验报错。
    knowledge_l2 = features.get("knowledge_L2")
    if (
        isinstance(knowledge_l2, list)
        and knowledge_l2
        and all(
            isinstance(value, str) and value in KNOWLEDGE_L2_TO_L1
            for value in knowledge_l2
        )
    ):
        derived_l1 = list(
            dict.fromkeys(KNOWLEDGE_L2_TO_L1[value] for value in knowledge_l2)
        )
        if features.get("knowledge_L1") != derived_l1:
            previous_l1 = copy.deepcopy(features.get("knowledge_L1"))
            features["knowledge_L1"] = derived_l1
            log.append(
                {
                    "field": "knowledge_L1",
                    "from": previous_l1,
                    "to": copy.deepcopy(derived_l1),
                    "reason": "由合法 knowledge_L2 的固定父级确定性派生",
                }
            )
    validate_feature_schema(features)
    return normalized, log


def recalculate_verification(
    *,
    current_level: str,
    original_high_count: int,
    original_high_features: list[str],
    original_accuracy: float,
    original_features: dict[str, Any],
    allow_auto_adjustment: bool,
    verification: dict[str, Any],
    multiplier_enabled: bool | None = None,
) -> dict[str, Any]:
    """应用二阶段的 feature 修正，再由程序重算高难特征、乘数和档位。"""
    reviewed = copy.deepcopy(verification)
    corrected_features = copy.deepcopy(original_features)
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for correction in reviewed.get("feature_corrections") or []:
        field = correction.get("field")
        source = correction.get("from")
        target = correction.get("to")
        if field not in REQUIRED_FEATURE_FIELDS:
            rejected.append({**copy.deepcopy(correction), "reason": "非 feature 字段"})
            continue
        if field in PROGRAM_DERIVED_FEATURE_FIELDS:
            rejected.append({
                **copy.deepcopy(correction),
                "reason": "程序派生字段不得授权结构改分",
            })
            continue
        if source != corrected_features.get(field):
            rejected.append({
                **copy.deepcopy(correction),
                "reason": "from 与当前已核验 feature 值不一致",
            })
            continue
        if target == source:
            rejected.append({
                **copy.deepcopy(correction),
                "reason": "to 与 from 相同，不构成结构修正",
            })
            continue
        candidate = copy.deepcopy(corrected_features)
        candidate[field] = target
        try:
            validate_feature_schema(candidate)
        except ValueError as exc:
            rejected.append({**copy.deepcopy(correction), "reason": str(exc)})
            continue
        corrected_features = candidate
        applied.append(copy.deepcopy(correction))

    try:
        model_reviewed_accuracy = float(
            reviewed["reviewed_original_predicted_accuracy"]
        )
    except (KeyError, TypeError, ValueError):
        model_reviewed_accuracy = float(original_accuracy)
    model_reviewed_accuracy = min(100.0, max(0.0, model_reviewed_accuracy))
    high = detect_high_difficulty_features(corrected_features)
    structural_revision_supported = bool(applied)
    reviewed_accuracy = (
        model_reviewed_accuracy
        if structural_revision_supported
        else float(original_accuracy)
    )
    reviewed_count = len(high.names)
    enabled = (
        CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER_ENABLED
        if multiplier_enabled is None
        else bool(multiplier_enabled)
    )
    multiplier_policy = _apply_chemistry_multiplier_policy(
        original_accuracy=reviewed_accuracy,
        high_names=high.names,
        multiplier_enabled=enabled,
        full_dependent_organic_route=_is_full_dependent_organic_route(
            corrected_features
        ),
    )
    multiplier_candidate = multiplier_policy["multiplier_candidate"]
    multiplier = multiplier_policy["multiplier_applied"]
    adjusted_accuracy = multiplier_policy["adjusted_accuracy"]
    reviewed_level = map_accuracy_to_level(adjusted_accuracy)
    boundary = reviewed.get("adjacent_boundary_review") or {}
    boundary_verdict = boundary.get("verdict", "维持")
    current_index = LEVEL_INDEX[current_level]
    reviewed_index = LEVEL_INDEX[reviewed_level]
    boundary_promotion_candidate = _chemistry_58_boundary_promotion_candidate(
        current_level=current_level,
        original_accuracy=reviewed_accuracy,
        features=corrected_features,
    )
    structural_cluster_candidate, structural_cluster_signals = (
        _chemistry_structural_cluster_promotion_candidate(
            current_level=current_level,
            original_accuracy=reviewed_accuracy,
            features=corrected_features,
        )
    )
    any_boundary_promotion_candidate = (
        boundary_promotion_candidate or structural_cluster_candidate
    )
    if any_boundary_promotion_candidate and reviewed_index == current_index:
        reviewed_direction = "应更难一档"
        proposed_reasonableness = "偏低"
        reviewed_target_level = "难度4档"
    elif reviewed_index == current_index:
        reviewed_direction = "维持"
        proposed_reasonableness = "合理"
        reviewed_target_level = reviewed_level
    elif reviewed_index < current_index:
        reviewed_direction = "应更简单一档"
        proposed_reasonableness = "偏高"
        reviewed_target_level = reviewed_level
    else:
        reviewed_direction = "应更难一档"
        proposed_reasonableness = "偏低"
        reviewed_target_level = reviewed_level
    action_map = {
        "维持": "维持",
        "应更简单一档": "建议降一档",
        "应更难一档": "建议升一档",
    }
    # 乘数由程序根据“修正后的合法 features”确定。若二阶段已提供并通过
    # 校验的结构修正，计数或组合变化正是应重新计算乘数的原因，不能反过来
    # 视为乘数不合理而阻断同一条合法改档链。
    multiplier_changed_after_supported_revision = (
        structural_revision_supported
        and (
            reviewed_count != original_high_count
            or set(high.names) != set(original_high_features)
        )
    )
    multiplier_reasonable = True
    input_review = reviewed.get("input_sufficiency_review") or {}
    unresolved_overlap = bool(reviewed.get("high_feature_overlap_review")) and any(
        str(item.get("resolution") or "") in {"无法确定", "需人工"}
        for item in reviewed.get("high_feature_overlap_review") or []
        if isinstance(item, dict)
    )
    boundary_verdict_consistent = (
        any_boundary_promotion_candidate
        or boundary_verdict == reviewed_direction
    )
    blocks_two_to_one = (
        current_level == "难度2档"
        and reviewed_target_level == "难度1档"
    )
    auto_adjustment_eligible = (
        allow_auto_adjustment
        and (structural_revision_supported or any_boundary_promotion_candidate)
        and reviewed.get("confidence") == "高"
        and reviewed_direction != "维持"
        and boundary_verdict_consistent
        and input_review.get("status") != "信息不足"
        and not unresolved_overlap
        and not blocks_two_to_one
    )
    reasonableness = (
        proposed_reasonableness if auto_adjustment_eligible else "合理"
    )
    adjusted_level = (
        reviewed_target_level if auto_adjustment_eligible else current_level
    )
    review_requires_manual = bool(
        reviewed_direction != "维持" and not auto_adjustment_eligible
    )
    reviewed.update(
        {
            "has_structural_revision_model_raw": reviewed.get(
                "has_structural_revision"
            ),
            "has_structural_revision": structural_revision_supported,
            "feature_corrections_applied": applied,
            "feature_corrections_rejected": rejected,
            "reviewed_features": corrected_features,
            "reviewed_high_difficulty_features_model": copy.deepcopy(
                reviewed.get("reviewed_high_difficulty_features") or []
            ),
            "reviewed_high_difficulty_features": high.names,
            "reviewed_high_difficulty_feature_evidence": high.evidence,
            "reviewed_suppressed_high_feature_overlaps": high.suppressed_overlaps,
            "reviewed_high_difficulty_feature_count": reviewed_count,
            "reviewed_full_dependent_organic_route_detected": (
                _is_full_dependent_organic_route(corrected_features)
            ),
            "reviewed_high_difficulty_multiplier_enabled": enabled,
            "reviewed_multiplier_candidate": multiplier_candidate,
            "reviewed_multiplier_triggered": multiplier_policy["multiplier_triggered"],
            "reviewed_multiplier_trigger_combo": multiplier_policy["multiplier_trigger_combo"],
            "reviewed_multiplier_final_level_guard_applied": multiplier_policy[
                "multiplier_final_level_guard_applied"
            ],
            "reviewed_multiplier_applied": multiplier,
            "reviewed_original_predicted_accuracy_model_raw": (
                model_reviewed_accuracy
            ),
            "reviewed_original_predicted_accuracy": reviewed_accuracy,
            "reviewed_predicted_accuracy": adjusted_accuracy,
            "reviewed_difficulty_level": reviewed_level,
            "reviewed_direction": reviewed_direction,
            "chemistry_58_boundary_promotion_candidate": (
                boundary_promotion_candidate
            ),
            "chemistry_structural_cluster_promotion_candidate": (
                structural_cluster_candidate
            ),
            "chemistry_structural_cluster_signals": structural_cluster_signals,
            "auto_downgrade_two_to_one_blocked": blocks_two_to_one,
            "boundary_verdict_consistent": boundary_verdict_consistent,
            "auto_adjustment_eligible": auto_adjustment_eligible,
            "review_action": (
                action_map[reviewed_direction]
                if auto_adjustment_eligible
                else "维持"
            ),
            "rating_reasonableness": reasonableness,
            "adjusted_difficulty_level": adjusted_level,
            "multiplier_reasonableness": "合理" if multiplier_reasonable else "不合理",
            "multiplier_changed_after_supported_revision": (
                multiplier_changed_after_supported_revision
            ),
            "stage2_auto_adjustment_enabled": bool(allow_auto_adjustment),
            "review_requires_manual": review_requires_manual,
        }
    )
    return reviewed


def _safe_question_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    value = str(item.get("question_id") or "")
    try:
        return (0, f"{int(value):030d}")
    except ValueError:
        return (1, value)


def _image_urls(question: dict[str, Any], keys: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for node in [question, *(question.get("sub_questions") or [])]:
        if not isinstance(node, dict):
            continue
        for key in keys:
            value = str(node.get(key) or "").strip()
            if value and value not in values:
                values.append(value)
    return values


def prepare_question(source_question: dict[str, Any], image_mode: str = "auto") -> PreparedQuestion:
    if image_mode not in {"off", "auto", "all"}:
        raise ValueError("image_mode 只能为 off、auto、all")
    source_difficulty = copy.deepcopy(source_question.get("difficulty"))
    question = {
        key: copy.deepcopy(source_question[key])
        for key in QUESTION_MODEL_FIELDS
        if key in source_question and key != "sub_questions"
    }
    subquestions = []
    for raw in source_question.get("sub_questions") or []:
        if not isinstance(raw, dict):
            continue
        subquestions.append({
            key: copy.deepcopy(raw[key])
            for key in SUBQUESTION_MODEL_FIELDS
            if key in raw
        })
    question["sub_questions"] = sorted(subquestions, key=_safe_question_sort_key)

    parent_analysis = str(question.get("analysis") or "").strip()
    child_analysis = any(str(item.get("analysis") or "").strip() for item in subquestions)
    has_analysis = bool(parent_analysis) or child_analysis
    text = " ".join([
        str(question.get("stem") or ""),
        str(question.get("options") or ""),
        *[str(item.get("stem") or "") + str(item.get("options") or "") for item in subquestions],
    ]).strip()
    image_required = not text or bool(re.search(r"如图|图中|下图|图像|图谱|光谱|流程图|装置图|<img", text, re.I))
    stem_urls = _image_urls(question, ("stem_image_url", "stem_pic_url"))
    analysis_urls = _image_urls(question, ("analysis_image_url", "analysis_pic_url"))
    all_urls = list(dict.fromkeys(stem_urls + analysis_urls))
    if image_mode == "all":
        selected = all_urls
    elif image_mode == "auto" and image_required:
        selected = all_urls
    else:
        selected = []

    if not text and not selected:
        sufficiency = "信息不足"
    elif image_required and not selected:
        sufficiency = "部分缺失" if has_analysis else "信息不足"
    elif not has_analysis:
        sufficiency = "部分缺失"
    else:
        sufficiency = "充分"
    return PreparedQuestion(
        question=question,
        source_difficulty_untrusted=source_difficulty,
        input_quality={
            "parent_analysis_available": bool(parent_analysis),
            "subquestion_analysis_available": child_analysis,
            "has_analysis": has_analysis,
            "image_required": image_required,
            "image_available": bool(all_urls),
            "image_included": bool(selected),
            "input_sufficiency": sufficiency,
        },
        selected_image_urls=selected,
    )


def normalize_level(value: Any) -> str:
    if value in LEVEL_INDEX:
        return str(value)
    match = re.search(r"难度[1-5]档", str(value or ""))
    return match.group(0) if match else ""


def finalize_level(
    *,
    current_level: str,
    review_action: str,
    model_suggested_level: Any,
    input_sufficiency: str,
    auto_adjustment_enabled: bool = False,
) -> FinalizationResult:
    if current_level not in LEVEL_INDEX:
        raise ValueError(f"无效 current_level：{current_level!r}")
    suggested = normalize_level(model_suggested_level) or current_level
    current_index = LEVEL_INDEX[current_level]
    suggested_index = LEVEL_INDEX[suggested]
    manual = input_sufficiency == "信息不足" or abs(suggested_index - current_index) >= 2

    if not auto_adjustment_enabled:
        return FinalizationResult(
            final_level=current_level,
            needs_manual_review=manual or suggested != current_level,
            model_suggested_level=suggested,
            adjustment_desc=f"审计模式·{review_action or '未知'}·维持{current_level}",
            auto_adjustment_applied=False,
        )

    if review_action == "建议降一档" and suggested_index < current_index:
        final_index = max(0, current_index - 1)
    elif review_action == "建议升一档" and suggested_index > current_index:
        final_index = min(4, current_index + 1)
    else:
        final_index = current_index
        if suggested != current_level:
            manual = True
    final_level = LEVEL_ORDER[final_index]
    return FinalizationResult(
        final_level=final_level,
        needs_manual_review=manual,
        model_suggested_level=suggested,
        adjustment_desc=(f"{review_action}·{current_level}→{final_level}" if final_level != current_level else f"{review_action}·维持{current_level}"),
        auto_adjustment_applied=final_level != current_level,
    )
