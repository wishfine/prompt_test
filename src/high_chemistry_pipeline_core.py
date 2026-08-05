# -*- coding: utf-8 -*-
"""高中化学两阶段难度 Pipeline 的纯函数核心。

本模块不依赖网络请求，集中处理：
1. 化学 feature schema 校验；
2. 高难特征严格触发与重复计数抑制；
3. 0.85 / 0.70 乘数效应和固定正确率分档；
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
        if features[field] not in options:
            raise ValueError(f"{field} 非法值 {features[field]!r}；允许值：{sorted(options)}")


def detect_active_features(features: dict[str, Any]) -> list[str]:
    """普通活跃特征只用于复核，不参与乘数选择。"""
    gates = [
        (features.get("knowledge_scope") != "单知识点", "知识综合"),
        (features.get("substance_count") != "1种", "多物质"),
        (features.get("reaction_count") != "0-1个", "多反应"),
        (features.get("process_structure") != "单阶段", "多阶段"),
        (features.get("step_count") != "1-2步", "多步骤"),
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
        features.get("reaction_count") in {"4-6个", "7个及以上"}
        and features.get("reaction_relation") in {"前后反应强依赖", "多路径反应网络"}
        and features.get("process_structure") in {"多阶段强依赖", "循环或回流流程"}
    )
    if multi_reaction:
        fields = ["reaction_count", "reaction_relation", "process_structure"]
        evidence_by_name["多反应或多阶段强耦合"] = _evidence("多反应或多阶段强耦合", fields, features, "reaction_process_network")

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
    if design_high and information_high and features.get("information_carrier") in {"实验装置", "工艺流程图"}:
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


def enrich_stage1_rating(
    stage1_rating: dict[str, Any],
    *,
    features_model_raw: dict[str, Any] | None = None,
    normalization_log: list[dict[str, Any]] | None = None,
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
    multiplier = multiplier_for_high_count(high_count)
    adjusted = round(raw_accuracy * multiplier, 1)
    rating["original_predicted_accuracy"] = raw_accuracy
    rating["active_features"] = active
    rating["active_feature_count"] = len(active)
    rating["high_difficulty_features"] = high.names
    rating["high_difficulty_feature_evidence"] = high.evidence
    rating["possible_high_feature_overlaps"] = high.possible_overlap_groups
    rating["suppressed_high_feature_overlaps"] = high.suppressed_overlaps
    rating["high_difficulty_feature_count"] = high_count
    rating["multiplier_applied"] = multiplier
    rating["predicted_accuracy"] = adjusted
    rating["difficulty_level_step1"] = map_accuracy_to_level(adjusted)
    return rating


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
    shared = features.get("shared_model_across_subquestions")
    if shared in {"是", "true", "True", 1}:
        features["shared_model_across_subquestions"] = True
        log.append({"field": "shared_model_across_subquestions", "from": shared, "to": True})
    elif shared in {"否", "false", "False", 0}:
        features["shared_model_across_subquestions"] = False
        log.append({"field": "shared_model_across_subquestions", "from": shared, "to": False})

    # L1 是 L2 的确定性父级，不应让模型重复填写造成整题失败。
    # 只有当 L2 本身完整合法时才自动派生；非法或缺失 L2 仍交给校验报错。
    knowledge_l2 = features.get("knowledge_L2")
    if (
        isinstance(knowledge_l2, list)
        and knowledge_l2
        and all(value in KNOWLEDGE_L2_TO_L1 for value in knowledge_l2)
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
) -> dict[str, Any]:
    """应用二阶段的 feature 修正，再由程序重算高难特征、乘数和档位。"""
    reviewed = copy.deepcopy(verification)
    corrected_features = copy.deepcopy(original_features)
    applied: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for correction in reviewed.get("feature_corrections") or []:
        field = correction.get("field")
        target = correction.get("to")
        if field not in REQUIRED_FEATURE_FIELDS:
            rejected.append({**copy.deepcopy(correction), "reason": "非 feature 字段"})
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
        reviewed_accuracy = float(reviewed["reviewed_original_predicted_accuracy"])
    except (KeyError, TypeError, ValueError):
        reviewed_accuracy = float(original_accuracy)
    reviewed_accuracy = min(100.0, max(0.0, reviewed_accuracy))
    high = detect_high_difficulty_features(corrected_features)
    reviewed_count = len(high.names)
    multiplier = multiplier_for_high_count(reviewed_count)
    adjusted_accuracy = round(reviewed_accuracy * multiplier, 1)
    reviewed_level = map_accuracy_to_level(adjusted_accuracy)
    boundary = reviewed.get("adjacent_boundary_review") or {}
    verdict = boundary.get("verdict", "维持")
    action_map = {
        "维持": "维持",
        "应更简单一档": "建议降一档",
        "应更难一档": "建议升一档",
    }
    reasonableness_map = {
        "维持": "合理",
        "应更简单一档": "偏高",
        "应更难一档": "偏低",
    }
    multiplier_reasonable = (
        reviewed_count == original_high_count
        and set(high.names) == set(original_high_features)
    )
    input_review = reviewed.get("input_sufficiency_review") or {}
    unresolved_overlap = bool(reviewed.get("high_feature_overlap_review")) and any(
        str(item.get("resolution") or "") in {"无法确定", "需人工"}
        for item in reviewed.get("high_feature_overlap_review") or []
        if isinstance(item, dict)
    )
    review_requires_manual = bool(
        reviewed.get("confidence") == "低"
        or input_review.get("status") == "信息不足"
        or unresolved_overlap
        or abs(LEVEL_INDEX[reviewed_level] - LEVEL_INDEX[current_level]) >= 2
    )
    reviewed.update(
        {
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
            "reviewed_multiplier_applied": multiplier,
            "reviewed_predicted_accuracy": adjusted_accuracy,
            "reviewed_difficulty_level": reviewed_level,
            "review_action": action_map.get(verdict, "维持"),
            "rating_reasonableness": reasonableness_map.get(verdict, "合理"),
            "adjusted_difficulty_level": reviewed_level,
            "multiplier_reasonableness": "合理" if multiplier_reasonable else "不合理",
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
