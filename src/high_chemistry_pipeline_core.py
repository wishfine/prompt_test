# -*- coding: utf-8 -*-
"""高中化学单阶段难度 Pipeline 的纯函数核心。

本模块与物理实现相互独立，只负责化学特征校验、程序化高难特征检测、
乘数处理、档位映射和输入清洗，不包含网络调用。
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


KNOWLEDGE_L1 = {
    "化学基本概念与定量关系",
    "元素化学与无机反应",
    "化学反应原理",
    "物质结构与性质",
    "有机化学",
    "化学实验与探究",
}

KNOWLEDGE_L2_TO_L1 = {
    "物质组成、分类与分散系": "化学基本概念与定量关系",
    "化学用语与物质组成": "化学基本概念与定量关系",
    "物质的量与化学计量": "化学基本概念与定量关系",
    "离子反应": "化学基本概念与定量关系",
    "氧化还原反应": "化学基本概念与定量关系",
    "金属及其化合物": "元素化学与无机反应",
    "非金属及其化合物": "元素化学与无机反应",
    "无机物转化与推断": "元素化学与无机反应",
    "化学反应与能量": "化学反应原理",
    "反应速率与化学平衡": "化学反应原理",
    "水溶液中的离子平衡": "化学反应原理",
    "电化学": "化学反应原理",
    "原子结构与元素周期律": "物质结构与性质",
    "化学键与分子结构": "物质结构与性质",
    "晶体结构与性质": "物质结构与性质",
    "有机物结构与命名": "有机化学",
    "烃及其衍生物": "有机化学",
    "有机反应与合成": "有机化学",
    "生物大分子与合成材料": "有机化学",
    "实验基础与安全": "化学实验与探究",
    "物质制备与性质实验": "化学实验与探究",
    "检验、鉴别与分离提纯": "化学实验与探究",
    "定量实验与数据处理": "化学实验与探究",
    "实验探究与方案评价": "化学实验与探究",
}
KNOWLEDGE_L2 = set(KNOWLEDGE_L2_TO_L1)

CHEMISTRY_METHODS = {
    "结构决定性质",
    "平衡思想",
    "守恒思想",
    "宏观微观符号结合",
    "证据推理",
    "控制变量",
    "转化与合成",
    "假设与验证",
}

FEATURE_OPTIONS: dict[str, set[str]] = {
    "knowledge_count": {"1个", "2-3个", "4个及以上"},
    "knowledge_scope": {"单知识点", "同章节综合", "同模块跨章节", "跨模块综合"},
    "knowledge_depth": {"基础概念", "标准模型", "深层课内模型", "陌生迁移"},
    "step_count": {"1-2步", "3-5步", "6-8步", "9-12步", "12步以上"},
    "substance_count": {"1种", "2-3种", "4-6种", "7种及以上"},
    "reaction_count": {"0个", "1个", "2-3个", "4个及以上"},
    "reaction_relation": {
        "无反应关系", "单一直接反应", "并列独立反应", "简单连续反应",
        "多阶段强依赖反应链", "竞争或副反应", "条件改变导致方向或产物变化",
    },
    "state_count": {"1个", "2个", "3个及以上"},
    "process_state_relation": {
        "单一关系", "状态相互独立", "显性顺序衔接", "前后状态强依赖",
        "连续变化伴随平衡或边界",
    },
    "constraint_structure": {"无约束", "单一约束", "多约束但相互独立", "多约束联合筛选"},
    "subquestion_dependency": {"无多问", "相互独立", "后问依赖前问"},
    "model_explicitness": {"模型完全显性", "半隐含模型", "隐含模型", "需要自主建模"},
    "model_relation": {"单一模型", "同一模型多状态", "模型切换", "多模型或多平衡耦合"},
    "reasoning_chain": {"直接判断", "简单因果", "多层因果", "逆向推理或条件筛选"},
    "hidden_conditions": {"无", "单个隐含条件", "多个隐含条件"},
    "critical_condition": {"无临界", "显性临界或过量条件", "需要推导临界", "隐含临界或过量条件"},
    "classification_discussion": {"无", "2类讨论", "3类讨论", "4类及以上"},
    "variable_relation": {
        "无变量关系", "简单正反比", "函数或图像关系", "分段或非线性关系", "多变量耦合",
    },
    "equation_structure": {"无方程", "单方程", "2-3个方程联立", "4个以上方程或不等式组"},
    "calculation_complexity": {"无需计算", "简单代数", "多方程联立", "参数或范围计算", "复杂近似计算"},
    "stoichiometric_calculation": {"无", "单一化学计量", "多步化学计量", "守恒差量或混合计算"},
    "equilibrium_calculation": {"无", "定性判断", "单一平衡定量", "多平衡耦合定量"},
    "information_carrier": {
        "纯文字", "单一示意图", "表格", "函数曲线", "实验装置", "工艺流程图", "光谱或图谱", "多载体综合",
    },
    "graph_structure": {
        "无图表", "直接读数", "单图关系转换", "单图反推隐藏量", "多图独立", "多图联合转换",
    },
    "experiment_requirement": {
        "无", "基础操作或现象判断", "标准数据处理", "控制变量或故障分析",
        "误差分析", "多步操作组合", "方案设计或可行性评价",
    },
    "synthesis_route": {"无", "补全单步反应", "常规多步路线", "自主设计或路线评价"},
    "separation_purification": {"无", "直接选择操作", "多步操作组合", "自主设计或方案评价"},
    "context_type": {"纯化学", "生活生产", "工业流程", "实验探究", "科技前沿"},
}

REQUIRED_FEATURE_FIELDS = (
    "knowledge_L1", "knowledge_L2", "knowledge_points",
    *FEATURE_OPTIONS.keys(),
    "shared_model_across_subquestions", "chemistry_methods",
)

HIGH_DIFFICULTY_FEATURE_NAMES = (
    "多物质多反应网络强耦合",
    "多阶段反应链强依赖",
    "多约束联合",
    "隐含反应或临界条件",
    "竞争副反应或复杂分类",
    "多模型或多平衡耦合",
    "复杂化学计量参数或范围",
    "高层级流程图谱信息转换",
    "跨模块深度综合",
    "高阶实验合成或分离方案设计",
)

UNTRUSTED_LABEL_FIELDS = {
    "difficulty", "percent_correct", "answered_count", "teacher_label",
    "teacher_difficulty", "label", "难度",
}
QUESTION_FIELDS = (
    "parent_id", "question_id", "stem", "options", "analysis", "structure_type",
    "sub_questions", "stem_image_url", "analysis_image_url", "stem_pic_url", "analysis_pic_url",
)
SUBQUESTION_FIELDS = (
    "parent_id", "question_id", "stem", "options", "analysis", "structure_type",
    "stem_image_url", "analysis_image_url", "stem_pic_url", "analysis_pic_url",
)
VISUAL_REFERENCE_RE = re.compile(
    r"(如图|图中|下图|图示|装置图|实验装置|流程图|曲线|图像|图象|图谱|光谱|"
    r"微观示意|粒子图|表格|据图|结合图|<img)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HighDifficultyDetection:
    names: list[str]
    evidence: list[dict[str, Any]]
    possible_overlap_groups: list[list[str]]


@dataclass(frozen=True)
class PreparedQuestion:
    question: dict[str, Any]
    source_difficulty_untrusted: Any
    input_quality: dict[str, Any]
    selected_image_urls: list[str]


def build_stage1_output_schema() -> dict[str, Any]:
    """Build the strict structured-output schema from the canonical enums."""
    feature_properties: dict[str, Any] = {
        "knowledge_L1": {
            "type": "array", "items": {"type": "string", "enum": sorted(KNOWLEDGE_L1)},
        },
        "knowledge_L2": {
            "type": "array", "items": {"type": "string", "enum": sorted(KNOWLEDGE_L2)},
        },
        "knowledge_points": {
            "type": "array", "items": {"type": "string"},
        },
    }
    feature_properties.update({
        field: {"type": "string", "enum": sorted(options)}
        for field, options in FEATURE_OPTIONS.items()
    })
    feature_properties["shared_model_across_subquestions"] = {"type": "boolean"}
    feature_properties["chemistry_methods"] = {
        "type": "array", "items": {"type": "string", "enum": sorted(CHEMISTRY_METHODS)},
    }
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
        "required": [
            "features", "reason", "predicted_accuracy",
        ],
        "additionalProperties": False,
    }


def map_accuracy_to_level(predicted_accuracy: Any) -> str:
    try:
        accuracy = float(predicted_accuracy)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"predicted_accuracy 必须为数值，实际为 {predicted_accuracy!r}") from exc
    if not 0 <= accuracy <= 100:
        raise ValueError("predicted_accuracy 必须位于 0 到 100")
    if accuracy >= 88:
        return "难度1档"
    if accuracy >= 85:
        return "难度2档"
    if accuracy >= 58:
        return "难度3档"
    if accuracy >= 38:
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


def _ensure_unique_strings(values: Any, field: str, *, nonempty: bool = True) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field} 必须为列表")
    cleaned = [str(value).strip() for value in values]
    if nonempty and not cleaned:
        raise ValueError(f"{field} 不得为空")
    if any(not value for value in cleaned):
        raise ValueError(f"{field} 不得包含空字符串")
    if len(cleaned) != len(set(cleaned)):
        raise ValueError(f"{field} 不得包含重复值")
    return cleaned


def _deduplicate_exact_strings(
    values: Any, field: str, *, nonempty: bool = True
) -> list[str]:
    if not isinstance(values, list):
        raise ValueError(f"{field} 必须为列表")
    cleaned: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} 不得包含空值或非字符串")
        normalized = value.strip()
        if normalized not in cleaned:
            cleaned.append(normalized)
    if nonempty and not cleaned:
        raise ValueError(f"{field} 不得为空")
    return cleaned


def validate_feature_schema(features: dict[str, Any]) -> None:
    if not isinstance(features, dict):
        raise ValueError("features 必须为对象")
    missing = [field for field in REQUIRED_FEATURE_FIELDS if field not in features]
    if missing:
        raise ValueError(f"features 缺少字段：{', '.join(missing)}")

    l1 = _ensure_unique_strings(features["knowledge_L1"], "knowledge_L1")
    l2 = _ensure_unique_strings(features["knowledge_L2"], "knowledge_L2")
    points = _ensure_unique_strings(features["knowledge_points"], "knowledge_points")
    invalid_l1 = [value for value in l1 if value not in KNOWLEDGE_L1]
    invalid_l2 = [value for value in l2 if value not in KNOWLEDGE_L2]
    if invalid_l1:
        raise ValueError(f"knowledge_L1 含非法值：{invalid_l1}")
    if invalid_l2:
        raise ValueError(f"knowledge_L2 含非法值：{invalid_l2}")
    derived_l1 = {KNOWLEDGE_L2_TO_L1[value] for value in l2}
    if derived_l1 != set(l1):
        raise ValueError(f"knowledge_L1 与 knowledge_L2 不一致：L2归属={sorted(derived_l1)}")
    if not points:
        raise ValueError("knowledge_points 不得为空")

    if not isinstance(features["shared_model_across_subquestions"], bool):
        raise ValueError("shared_model_across_subquestions 必须为布尔值")
    methods = _ensure_unique_strings(
        features["chemistry_methods"], "chemistry_methods", nonempty=False
    )
    invalid_methods = [value for value in methods if value not in CHEMISTRY_METHODS]
    if invalid_methods:
        raise ValueError(f"chemistry_methods 含非法值：{invalid_methods}")
    for field, options in FEATURE_OPTIONS.items():
        if features[field] not in options:
            raise ValueError(f"{field} 非法值 {features[field]!r}；允许值：{sorted(options)}")


def _derived_knowledge_count(points: list[str]) -> str:
    count = len(dict.fromkeys(value.strip() for value in points))
    return "1个" if count == 1 else ("2-3个" if count <= 3 else "4个及以上")


def _derived_knowledge_scope(l1: list[str], l2: list[str], points: list[str]) -> str:
    if len(set(l1)) >= 2:
        return "跨模块综合"
    if len(set(l2)) >= 2:
        return "同模块跨章节"
    if len(set(points)) >= 2:
        return "同章节综合"
    return "单知识点"


def prepare_stage1_rating(result: dict[str, Any]) -> dict[str, Any]:
    """校验严格结构化输出，并确定性重算可由其他字段派生的特征。"""
    if not isinstance(result, dict):
        raise ValueError("第一阶段结果必须为对象")
    prepared = copy.deepcopy(result)
    features = prepared.get("features")
    if not isinstance(features, dict):
        raise ValueError("第一阶段结果缺少 features 对象")
    points = _deduplicate_exact_strings(features.get("knowledge_points"), "knowledge_points")
    l2_values = _deduplicate_exact_strings(features.get("knowledge_L2"), "knowledge_L2")
    methods = _deduplicate_exact_strings(
        features.get("chemistry_methods"), "chemistry_methods", nonempty=False
    )
    features["knowledge_points"] = points
    features["knowledge_L2"] = l2_values
    features["chemistry_methods"] = methods
    invalid_l2 = [value for value in l2_values if value not in KNOWLEDGE_L2_TO_L1]
    if invalid_l2:
        raise ValueError(f"knowledge_L2 含非法值：{invalid_l2}")
    l1_order = [
        "化学基本概念与定量关系", "元素化学与无机反应", "化学反应原理",
        "物质结构与性质", "有机化学", "化学实验与探究",
    ]
    derived_l1 = [
        value for value in l1_order
        if value in {KNOWLEDGE_L2_TO_L1[item] for item in l2_values}
    ]
    features["knowledge_L1"] = derived_l1
    features["knowledge_count"] = _derived_knowledge_count(points)
    features["knowledge_scope"] = _derived_knowledge_scope(derived_l1, l2_values, points)
    validate_feature_schema(features)
    return prepared


def _high_evidence(name: str, features: dict[str, Any], fields: list[str], key: str) -> dict[str, Any]:
    return {
        "name": name,
        "fields": fields,
        "evidence": [f"{field}={features.get(field)}" for field in fields],
        "evidence_keys": [key],
    }


def detect_high_difficulty_features(features: dict[str, Any]) -> HighDifficultyDetection:
    """按联合条件检测十类化学高难特征，单个关键词不触发。"""
    evidence: dict[str, dict[str, Any]] = {}

    if (
        features.get("substance_count") in {"4-6种", "7种及以上"}
        and features.get("reaction_count") in {"2-3个", "4个及以上"}
        and (
            features.get("reaction_relation") in {"竞争或副反应", "条件改变导致方向或产物变化"}
            or (
                features.get("reaction_relation") == "多阶段强依赖反应链"
                and features.get("model_relation") in {"模型切换", "多模型或多平衡耦合"}
            )
        )
        and features.get("reasoning_chain") in {"多层因果", "逆向推理或条件筛选"}
    ):
        fields = ["substance_count", "reaction_count", "reaction_relation", "reasoning_chain"]
        evidence["多物质多反应网络强耦合"] = _high_evidence(
            "多物质多反应网络强耦合", features, fields, "reaction_network"
        )

    if (
        features.get("reaction_count") == "4个及以上"
        and features.get("reaction_relation") == "多阶段强依赖反应链"
        and features.get("process_state_relation") == "前后状态强依赖"
        and features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
    ):
        fields = ["reaction_count", "reaction_relation", "process_state_relation", "step_count"]
        evidence["多阶段反应链强依赖"] = _high_evidence(
            "多阶段反应链强依赖", features, fields, "dependent_reaction_chain"
        )

    if features.get("constraint_structure") == "多约束联合筛选":
        evidence["多约束联合"] = _high_evidence(
            "多约束联合", features, ["constraint_structure"], "joint_constraints"
        )

    hidden_high = (
        features.get("critical_condition") == "隐含临界或过量条件"
        and features.get("hidden_conditions") in {"单个隐含条件", "多个隐含条件"}
        and features.get("reasoning_chain") == "逆向推理或条件筛选"
    )
    if hidden_high:
        fields = ["critical_condition", "hidden_conditions", "reasoning_chain"]
        evidence["隐含反应或临界条件"] = _high_evidence(
            "隐含反应或临界条件", features, fields, "hidden_reaction_boundary"
        )

    competition_high = (
        features.get("reaction_relation") == "竞争或副反应"
        and features.get("reasoning_chain") in {"多层因果", "逆向推理或条件筛选"}
        and (
            features.get("classification_discussion") in {"3类讨论", "4类及以上"}
            or (
                features.get("classification_discussion") == "2类讨论"
                and features.get("model_relation") in {"模型切换", "多模型或多平衡耦合"}
                and features.get("constraint_structure") == "多约束联合筛选"
            )
        )
    )
    if competition_high:
        fields = ["reaction_relation", "classification_discussion", "reasoning_chain"]
        evidence["竞争副反应或复杂分类"] = _high_evidence(
            "竞争副反应或复杂分类", features, fields, "competition_classification"
        )

    model_high = (
        features.get("model_relation") == "多模型或多平衡耦合"
        and features.get("process_state_relation") in {"前后状态强依赖", "连续变化伴随平衡或边界"}
        and features.get("reasoning_chain") in {"多层因果", "逆向推理或条件筛选"}
    )
    if model_high:
        fields = ["model_relation", "process_state_relation", "reasoning_chain"]
        evidence["多模型或多平衡耦合"] = _high_evidence(
            "多模型或多平衡耦合", features, fields, "model_equilibrium_coupling"
        )

    calculation_high = (
        (
            features.get("stoichiometric_calculation") == "守恒差量或混合计算"
            or features.get("equilibrium_calculation") == "多平衡耦合定量"
        )
        and features.get("calculation_complexity") in {"多方程联立", "参数或范围计算", "复杂近似计算"}
        and features.get("equation_structure") in {"2-3个方程联立", "4个以上方程或不等式组"}
    )
    if calculation_high:
        fields = ["stoichiometric_calculation", "equilibrium_calculation", "calculation_complexity", "equation_structure"]
        evidence["复杂化学计量参数或范围"] = _high_evidence(
            "复杂化学计量参数或范围", features, fields, "complex_chemistry_calculation"
        )

    graph_high = (
        features.get("information_carrier") in {"工艺流程图", "光谱或图谱", "多载体综合"}
        and features.get("graph_structure") in {"单图反推隐藏量", "多图联合转换"}
        and features.get("reasoning_chain") in {"多层因果", "逆向推理或条件筛选"}
    )
    if graph_high:
        fields = ["information_carrier", "graph_structure", "reasoning_chain"]
        evidence["高层级流程图谱信息转换"] = _high_evidence(
            "高层级流程图谱信息转换", features, fields, "advanced_visual_conversion"
        )

    cross_module_high = (
        features.get("knowledge_scope") == "跨模块综合"
        and features.get("knowledge_depth") in {"深层课内模型", "陌生迁移"}
        and features.get("model_relation") in {"模型切换", "多模型或多平衡耦合"}
        and features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
    )
    if cross_module_high:
        fields = ["knowledge_scope", "knowledge_depth", "model_relation", "step_count"]
        evidence["跨模块深度综合"] = _high_evidence(
            "跨模块深度综合", features, fields, "cross_module_depth"
        )

    design_high = (
        (
            features.get("experiment_requirement") == "方案设计或可行性评价"
            or features.get("synthesis_route") == "自主设计或路线评价"
            or features.get("separation_purification") == "自主设计或方案评价"
        )
        and features.get("reasoning_chain") in {"多层因果", "逆向推理或条件筛选"}
        and features.get("constraint_structure") in {"单一约束", "多约束但相互独立", "多约束联合筛选"}
    )
    if design_high:
        fields = ["experiment_requirement", "synthesis_route", "separation_purification", "reasoning_chain", "constraint_structure"]
        evidence["高阶实验合成或分离方案设计"] = _high_evidence(
            "高阶实验合成或分离方案设计", features, fields, "advanced_design"
        )

    names = [name for name in HIGH_DIFFICULTY_FEATURE_NAMES if name in evidence]
    overlaps: list[list[str]] = []
    for index, left in enumerate(names):
        left_fields = set(evidence[left]["fields"])
        for right in names[index + 1:]:
            if left_fields & set(evidence[right]["fields"]):
                overlaps.append([left, right])
    return HighDifficultyDetection(
        names=names,
        evidence=[evidence[name] for name in names],
        possible_overlap_groups=overlaps,
    )


def group_overlapping_high_difficulty_features(
    names: list[str], overlap_pairs: list[list[str]]
) -> list[list[str]]:
    """将共享证据字段的高难特征按传递关系合并为独立乘数组。"""
    adjacency = {name: set() for name in names}
    for pair in overlap_pairs:
        if len(pair) != 2 or pair[0] not in adjacency or pair[1] not in adjacency:
            continue
        adjacency[pair[0]].add(pair[1])
        adjacency[pair[1]].add(pair[0])

    groups: list[list[str]] = []
    visited: set[str] = set()
    for name in names:
        if name in visited:
            continue
        pending = [name]
        members: set[str] = {name}
        visited.add(name)
        while pending:
            current = pending.pop()
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    members.add(neighbor)
                    pending.append(neighbor)
        groups.append([candidate for candidate in names if candidate in members])
    return groups


def detect_active_features(features: dict[str, Any]) -> list[str]:
    """记录普通活跃结构；每个认知类别最多计一次，不参与乘数。"""
    active: list[str] = []
    gates = [
        (features.get("knowledge_scope") != "单知识点", str(features.get("knowledge_scope"))),
        (features.get("knowledge_depth") in {"深层课内模型", "陌生迁移"}, str(features.get("knowledge_depth"))),
        (features.get("substance_count") in {"4-6种", "7种及以上"}, "多物质"),
        (features.get("reaction_count") in {"2-3个", "4个及以上"}, "多反应"),
        (features.get("reaction_relation") not in {"无反应关系", "单一直接反应"}, str(features.get("reaction_relation"))),
        (features.get("state_count") != "1个", "多状态"),
        (features.get("constraint_structure") != "无约束", "存在约束"),
        (
            features.get("subquestion_dependency") == "后问依赖前问"
            or features.get("shared_model_across_subquestions") is True,
            "多问依赖或共享模型",
        ),
        (features.get("model_explicitness") != "模型完全显性", str(features.get("model_explicitness"))),
        (features.get("model_relation") != "单一模型", str(features.get("model_relation"))),
        (features.get("reasoning_chain") != "直接判断", str(features.get("reasoning_chain"))),
        (features.get("hidden_conditions") != "无", "隐含条件"),
        (features.get("critical_condition") != "无临界", "临界或过量条件"),
        (features.get("classification_discussion") != "无", "分类讨论"),
        (features.get("variable_relation") not in {"无变量关系", "简单正反比"}, "复杂变量关系"),
        (bool(features.get("chemistry_methods")), "化学思想方法"),
        (features.get("equation_structure") in {"2-3个方程联立", "4个以上方程或不等式组"}, "方程联立"),
        (features.get("stoichiometric_calculation") in {"多步化学计量", "守恒差量或混合计算"}, "多步化学计量"),
        (features.get("equilibrium_calculation") in {"单一平衡定量", "多平衡耦合定量"}, "平衡定量"),
        (features.get("graph_structure") not in {"无图表", "直接读数"}, "图表信息转换"),
        (features.get("experiment_requirement") in {"控制变量或故障分析", "误差分析", "方案设计或可行性评价"}, "实验分析"),
        (features.get("synthesis_route") in {"常规多步路线", "自主设计或路线评价"}, "合成路线"),
        (features.get("separation_purification") in {"多步操作组合", "自主设计或方案评价"}, "分离提纯方案"),
    ]
    for enabled, name in gates:
        if enabled and name and name != "None" and name not in active:
            active.append(name)
    return active


def _validate_stage1_rating(rating: dict[str, Any]) -> None:
    if not str(rating.get("reason", "")).strip():
        raise ValueError("reason 不得为空")


def enrich_stage1_rating(
    stage1_rating: dict[str, Any],
    *,
    features_model_raw: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rating = copy.deepcopy(stage1_rating)
    features = rating.get("features")
    validate_feature_schema(features)

    rating["features_model_raw"] = copy.deepcopy(features if features_model_raw is None else features_model_raw)

    features["knowledge_points"] = list(dict.fromkeys(value.strip() for value in features["knowledge_points"]))
    rating["knowledge_count_model_raw"] = features["knowledge_count"]
    rating["knowledge_scope_model_raw"] = features["knowledge_scope"]
    features["knowledge_count"] = _derived_knowledge_count(features["knowledge_points"])
    features["knowledge_scope"] = _derived_knowledge_scope(
        features["knowledge_L1"], features["knowledge_L2"], features["knowledge_points"]
    )

    try:
        original_accuracy = float(rating["predicted_accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("predicted_accuracy 必须为数值") from exc
    if not 0 <= original_accuracy <= 100:
        raise ValueError("predicted_accuracy 必须位于 0 到 100")
    _validate_stage1_rating(rating)

    active_features = detect_active_features(features)
    detection = detect_high_difficulty_features(features)
    multiplier_groups = group_overlapping_high_difficulty_features(
        detection.names, detection.possible_overlap_groups
    )
    effective_high_count = len(multiplier_groups)
    multiplier = multiplier_for_high_count(effective_high_count)
    adjusted_accuracy = round(original_accuracy * multiplier, 1)
    rating["original_predicted_accuracy"] = round(original_accuracy, 1)
    rating["active_features"] = active_features
    rating["active_feature_count"] = len(active_features)
    rating["high_difficulty_features"] = detection.names
    rating["high_difficulty_feature_evidence"] = detection.evidence
    rating["possible_high_feature_overlaps"] = detection.possible_overlap_groups
    rating["high_difficulty_feature_count"] = len(detection.names)
    rating["high_difficulty_multiplier_groups"] = multiplier_groups
    rating["effective_high_difficulty_feature_count"] = effective_high_count
    rating["multiplier_applied"] = multiplier
    rating["predicted_accuracy"] = adjusted_accuracy
    rating["difficulty_level"] = map_accuracy_to_level(adjusted_accuracy)
    return rating


def _clean_subquestion(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {field: copy.deepcopy(value.get(field)) for field in SUBQUESTION_FIELDS if field in value}


def _safe_question_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    value = str(item.get("question_id") or "")
    try:
        return (0, f"{int(value):030d}")
    except ValueError:
        return (1, value)


def _collect_image_urls(question: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    values: list[str] = []
    for field in fields:
        value = str(question.get(field) or "").strip()
        if value and value not in values:
            values.append(value)
    for sub in question.get("sub_questions") or []:
        if not isinstance(sub, dict):
            continue
        for field in fields:
            value = str(sub.get(field) or "").strip()
            if value and value not in values:
                values.append(value)
    return values


def prepare_question(source_question: dict[str, Any], image_mode: str = "auto") -> PreparedQuestion:
    if image_mode not in {"off", "auto", "all"}:
        raise ValueError("image_mode 必须为 off/auto/all")
    if not isinstance(source_question, dict):
        raise ValueError("题目必须为对象")
    question = {field: copy.deepcopy(source_question.get(field)) for field in QUESTION_FIELDS if field in source_question}
    question["sub_questions"] = sorted([
        cleaned for value in (source_question.get("sub_questions") or [])
        if (cleaned := _clean_subquestion(value)) is not None
    ], key=_safe_question_sort_key)
    source_difficulty = copy.deepcopy(source_question.get("difficulty"))
    for field in UNTRUSTED_LABEL_FIELDS:
        question.pop(field, None)

    stem = str(question.get("stem") or "").strip()
    parent_analysis = str(question.get("analysis") or "").strip()
    options = str(question.get("options") or "").strip()
    structure_type = str(question.get("structure_type") or "").strip()
    children = question["sub_questions"]
    child_stems = [str(child.get("stem") or "").strip() for child in children]
    child_analyses = [str(child.get("analysis") or "").strip() for child in children]
    sub_analysis_available = any(child_analyses)
    has_analysis = bool(parent_analysis) or sub_analysis_available
    stem_images = _collect_image_urls(question, ("stem_image_url", "stem_pic_url"))
    analysis_images = _collect_image_urls(question, ("analysis_image_url", "analysis_pic_url"))
    images = list(dict.fromkeys(stem_images + analysis_images))
    has_image = bool(images)
    has_complete_children = bool(children) and all(child_stems)
    choice_requires_options = structure_type in {"danxuan", "duoxuan"}
    option_anomaly = choice_requires_options and not options

    if stem:
        content_mode = "text_complete"
    elif has_complete_children:
        content_mode = "subquestion_complete"
    elif has_image:
        content_mode = "image_dependent"
    else:
        content_mode = "insufficient"

    insufficient_reasons: list[str] = []
    if content_mode == "insufficient":
        insufficient_reasons.append("题干、题干图片和完整子题均缺失")
    if option_anomaly:
        insufficient_reasons.append("选择题缺少选项")
    if not has_analysis:
        insufficient_reasons.append("解析为空")

    question_text = "\n".join([stem, options, *child_stems]).strip()
    analysis_text = "\n".join([parent_analysis, *child_analyses]).strip()
    figure_reference = bool(VISUAL_REFERENCE_RE.search(question_text))
    image_required = not stem or not question_text or figure_reference
    if image_mode == "all":
        selected = images
    elif image_mode == "off":
        selected = []
    elif image_required:
        selected = list(stem_images)
        analysis_needs_image = not analysis_text or bool(VISUAL_REFERENCE_RE.search(analysis_text))
        if analysis_needs_image or not selected:
            selected.extend(url for url in analysis_images if url not in selected)
    else:
        selected = []

    if content_mode == "insufficient" and not selected:
        input_sufficiency = "不足"
    elif option_anomaly:
        input_sufficiency = "不足"
    elif image_required and not selected:
        input_sufficiency = "部分充分" if question_text else "不足"
    elif not has_analysis:
        input_sufficiency = "部分充分"
    else:
        input_sufficiency = "充分"

    quality = {
        "input_sufficiency": input_sufficiency,
        "content_mode": content_mode,
        "stem_text_present": bool(stem),
        "parent_analysis_available": bool(parent_analysis),
        "subquestion_analysis_available": sub_analysis_available,
        "has_analysis": has_analysis,
        "options_present": bool(options),
        "option_anomaly": option_anomaly,
        "subquestion_count": len(children),
        "all_subquestion_stems_present": has_complete_children,
        "image_required": image_required,
        "image_available": has_image,
        "image_included": bool(selected),
        "stem_image_included": any(url in selected for url in stem_images),
        "analysis_image_included": any(url in selected for url in analysis_images),
        "insufficient_reasons": insufficient_reasons,
    }
    return PreparedQuestion(question, source_difficulty, quality, selected)
