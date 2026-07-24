# -*- coding: utf-8 -*-
"""高中物理两阶段难度 Pipeline 的纯函数核心。

本模块不依赖网络和异步库，集中实现 feature 校验、高难特征检测、乘数效应、
输入清洗、正确率映射和最终一档调整，便于离线测试和审计。
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any


LEVEL_ORDER = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVEL_ORDER)}

KNOWLEDGE_L1 = {
    "力学",
    "电磁学",
    "热学",
    "光学",
    "近代物理",
    "物理实验",
}

KNOWLEDGE_L2 = {
    "运动学",
    "相互作用与牛顿运动定律",
    "曲线运动与万有引力",
    "机械能",
    "动量",
    "振动与波",
    "静电场",
    "恒定电流",
    "磁场",
    "电磁感应与交变电流",
    "热学",
    "光学",
    "原子与原子核",
    "物理实验",
}

KNOWLEDGE_L2_TO_L1 = {
    "运动学": "力学",
    "相互作用与牛顿运动定律": "力学",
    "曲线运动与万有引力": "力学",
    "机械能": "力学",
    "动量": "力学",
    "振动与波": "力学",
    "静电场": "电磁学",
    "恒定电流": "电磁学",
    "磁场": "电磁学",
    "电磁感应与交变电流": "电磁学",
    "热学": "热学",
    "光学": "光学",
    "原子与原子核": "近代物理",
    "物理实验": "物理实验",
}

PHYSICS_METHODS = {
    "守恒思想",
    "整体法与隔离法",
    "物理建模",
    "等效替代",
    "对称性",
    "图像法",
    "极限与临界",
    "微元或累积",
    "假设与验证",
}

FEATURE_OPTIONS: dict[str, set[str]] = {
    "knowledge_count": {"1个", "2-3个", "4个及以上"},
    "knowledge_scope": {"单知识点", "同章节综合", "同模块跨章节", "跨模块综合"},
    "knowledge_depth": {"基础概念", "标准模型", "深层模型", "陌生迁移"},
    "primary_problem_structure": {"概念辨析", "直接计算", "综合计算", "图像分析", "实验探究", "信息迁移", "复合题"},
    "step_count": {"1-2步", "3-5步", "6-8步", "9-12步", "12步以上"},
    "process_count": {"单过程", "两个过程", "三个及以上过程"},
    "object_count": {"单对象", "两个对象", "三个及以上对象"},
    "object_relation": {"无对象关系", "对象相互独立", "单向影响", "双向耦合", "共同受约束"},
    "state_count": {"1个", "2个", "3个及以上"},
    "state_transition": {"无状态转换", "离散状态转换", "连续演化"},
    "process_state_relation": {"单一关系", "状态相互独立", "显性顺序衔接", "前后状态强依赖", "连续变化伴随边界"},
    "constraint_structure": {"无约束", "单一约束", "多约束但相互独立", "多约束联合筛选"},
    "subquestion_dependency": {"无多问", "相互独立", "后问依赖前问"},
    "model_explicitness": {"模型完全显性", "半隐含模型", "隐含模型", "需要自主建模"},
    "model_relation": {"单一模型", "同一模型多状态", "模型切换", "多模型耦合"},
    "reasoning_chain": {"直接套用", "简单因果", "多层因果", "逆向推理或临界分析"},
    "hidden_conditions": {"无", "单个隐含条件", "多个隐含条件"},
    "critical_state": {"无临界", "显性临界", "需要推导临界", "隐含临界"},
    "classification_discussion": {"无", "2类讨论", "3类讨论", "4类及以上"},
    "variable_relation": {"无变量关系", "简单正反比", "函数或图像关系", "分段或非线性关系", "多变量耦合"},
    "formula_count": {"0-1个", "2-3个", "4-6个", "7个以上"},
    "equation_structure": {"无方程", "单方程", "2-3个方程联立", "4个以上方程或不等式组"},
    "calculation_complexity": {"直接判断", "简单代数", "多方程联立", "参数或范围计算", "复杂近似计算"},
    "parameter_operation": {"无参数", "单参数", "双参数", "多参数"},
    "numerical_complexity": {"无数值计算", "简单整数", "常规小数或科学记数", "复杂数值或近似"},
    "unit_conversion": {"无", "单次常规换算", "多次换算", "非国际单位制转换"},
    "information_carrier": {"纯文字", "单一示意图", "函数图像", "表格", "实验装置", "多载体综合"},
    "graph_structure": {"无图表", "直接读数", "单图关系转换", "单图反推隐藏量", "多图独立", "多图联合转换"},
    "drawing_requirement": {"无", "补充标注", "常规作图", "自主辅助图", "重构物理图景"},
    "experiment_requirement": {"无", "基础操作或读数", "标准数据处理", "控制变量或故障分析", "误差反演", "方案设计或可行性验证"},
    "context_type": {"纯物理", "生活应用", "实验探究", "工程技术", "科技前沿"},
    "context_load": {"纯包装", "简单规律映射", "需要信息转换", "需要自主情境建模"},
    "error_risk": {"无明显易错点", "轻微易错点", "明显易错点", "高易错点"},
}

REQUIRED_FEATURE_FIELDS = (
    "knowledge_L1",
    "knowledge_L2",
    "knowledge_points",
    *FEATURE_OPTIONS.keys(),
    "shared_model_across_subquestions",
    "physics_methods",
)

HIGH_DIFFICULTY_FEATURE_NAMES = (
    "多对象强耦合",
    "多过程或多状态强耦合",
    "多约束联合",
    "隐含临界状态",
    "复杂分类讨论",
    "多模型切换或耦合",
    "复杂参数、范围或极值",
    "高层级图像信息转换",
    "跨模块深度综合",
    "高阶实验设计或误差反演",
)

QUESTION_MODEL_FIELDS = (
    "parent_id",
    "question_id",
    "stem",
    "options",
    "analysis",
    "structure_type",
    "sub_questions",
    "stem_image_url",
    "analysis_image_url",
    "stem_pic_url",
    "analysis_pic_url",
)

SUBQUESTION_MODEL_FIELDS = (
    "parent_id",
    "question_id",
    "stem",
    "options",
    "analysis",
    "structure_type",
    "stem_image_url",
    "analysis_image_url",
    "stem_pic_url",
    "analysis_pic_url",
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


@dataclass(frozen=True)
class FinalizationResult:
    final_level: str
    needs_manual_review: bool
    model_suggested_level: str
    adjustment_desc: str


def map_accuracy_to_level(predicted_accuracy: Any) -> str:
    """按连续区间将乘数后的正确率映射为五档。"""
    try:
        accuracy = float(predicted_accuracy)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"predicted_accuracy 必须为数值，实际为 {predicted_accuracy!r}") from exc
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
    """按高难特征类别数选择乘数；必须优先判断 4 个及以上。"""
    if high_count < 0:
        raise ValueError("high_count 不能为负数")
    if high_count >= 4:
        return 0.70
    if high_count >= 3:
        return 0.85
    return 1.0


def validate_feature_schema(features: dict[str, Any]) -> None:
    """验证第一阶段 feature 的完整性和枚举合法性。"""
    if not isinstance(features, dict):
        raise ValueError("features 必须为对象")
    missing = [field for field in REQUIRED_FEATURE_FIELDS if field not in features]
    if missing:
        raise ValueError(f"features 缺少字段：{', '.join(missing)}")

    def ensure_unique_list(values: list[Any], field_name: str) -> None:
        if len(values) != len(set(values)):
            raise ValueError(f"{field_name} 不得包含重复值")

    knowledge_l1 = features["knowledge_L1"]
    if not isinstance(knowledge_l1, list) or not knowledge_l1:
        raise ValueError("knowledge_L1 必须为非空列表")
    if any(not isinstance(value, str) for value in knowledge_l1):
        raise ValueError("knowledge_L1 每项必须为字符串")
    invalid_l1 = [value for value in knowledge_l1 if value not in KNOWLEDGE_L1]
    if invalid_l1:
        raise ValueError(f"knowledge_L1 含非法值：{invalid_l1}")
    ensure_unique_list(knowledge_l1, "knowledge_L1")

    knowledge_l2 = features["knowledge_L2"]
    if not isinstance(knowledge_l2, list) or not knowledge_l2:
        raise ValueError("knowledge_L2 必须为非空列表")
    if any(not isinstance(value, str) for value in knowledge_l2):
        raise ValueError("knowledge_L2 每项必须为字符串")
    invalid_l2 = [value for value in knowledge_l2 if value not in KNOWLEDGE_L2]
    if invalid_l2:
        raise ValueError(f"knowledge_L2 含非法值：{invalid_l2}")
    ensure_unique_list(knowledge_l2, "knowledge_L2")
    derived_l1 = {KNOWLEDGE_L2_TO_L1[value] for value in knowledge_l2}
    if derived_l1 != set(knowledge_l1):
        raise ValueError(
            "knowledge_L1 与 knowledge_L2 不一致；"
            f"L2 实际归属 {sorted(derived_l1)}，L1 为 {sorted(knowledge_l1)}"
        )

    knowledge_points = features["knowledge_points"]
    if (
        not isinstance(knowledge_points, list)
        or not knowledge_points
        or any(
            not isinstance(value, str) or not value.strip()
            for value in knowledge_points
        )
    ):
        raise ValueError("knowledge_points 必须为非空字符串列表")

    shared_model = features["shared_model_across_subquestions"]
    if not isinstance(shared_model, bool):
        raise ValueError("shared_model_across_subquestions 必须为布尔值")

    methods = features["physics_methods"]
    if not isinstance(methods, list):
        raise ValueError("physics_methods 必须为列表")
    if any(not isinstance(value, str) for value in methods):
        raise ValueError("physics_methods 每项必须为字符串")
    invalid_methods = [value for value in methods if value not in PHYSICS_METHODS]
    if invalid_methods:
        raise ValueError(f"physics_methods 含非法值：{invalid_methods}")
    ensure_unique_list(methods, "physics_methods")

    for field, options in FEATURE_OPTIONS.items():
        value = features[field]
        if value not in options:
            raise ValueError(f"{field} 非法值 {value!r}；允许值：{sorted(options)}")


def detect_active_features(features: dict[str, Any]) -> list[str]:
    """检测普通活跃特征；每个认知类别最多计一次。"""
    active: list[str] = []
    gates = [
        (features.get("knowledge_scope") != "单知识点", str(features.get("knowledge_scope"))),
        (features.get("knowledge_depth") in {"深层模型", "陌生迁移"}, str(features.get("knowledge_depth"))),
        (features.get("process_count") != "单过程", "多过程"),
        (features.get("object_count") != "单对象", "多对象"),
        (
            features.get("state_count") != "1个"
            or features.get("state_transition") != "无状态转换",
            "多状态或状态转换",
        ),
        (features.get("constraint_structure") != "无约束", "存在约束"),
        (
            features.get("subquestion_dependency") == "后问依赖前问"
            or features.get("shared_model_across_subquestions") is True,
            "多问依赖或共享模型",
        ),
        (features.get("model_explicitness") != "模型完全显性", str(features.get("model_explicitness"))),
        (features.get("model_relation") != "单一模型", str(features.get("model_relation"))),
        (features.get("reasoning_chain") != "直接套用", str(features.get("reasoning_chain"))),
        (features.get("hidden_conditions") != "无", "隐含条件"),
        (features.get("critical_state") != "无临界", "临界状态"),
        (features.get("classification_discussion") != "无", "分类讨论"),
        (features.get("variable_relation") not in {"无变量关系", "简单正反比"}, "复杂变量关系"),
        (bool(features.get("physics_methods")), "物理思想方法"),
        (features.get("equation_structure") in {"2-3个方程联立", "4个以上方程或不等式组"}, "方程联立"),
        (features.get("parameter_operation") != "无参数", "参数运算"),
        (features.get("graph_structure") not in {"无图表", "直接读数"}, "图像信息转换"),
        (features.get("drawing_requirement") in {"自主辅助图", "重构物理图景"}, "自主作图"),
        (features.get("experiment_requirement") in {"控制变量或故障分析", "误差反演", "方案设计或可行性验证"}, "实验分析"),
        (features.get("context_load") in {"需要信息转换", "需要自主情境建模"}, "情境建模"),
    ]
    for enabled, name in gates:
        if enabled and name and name != "None" and name not in active:
            active.append(name)
    return active


def _high_evidence(name: str, fields: list[str], values: list[Any], key: str) -> dict[str, Any]:
    return {
        "name": name,
        "fields": fields,
        "evidence": [f"{field}={value}" for field, value in zip(fields, values)],
        "evidence_keys": [key],
    }


def detect_high_difficulty_features(features: dict[str, Any]) -> HighDifficultyDetection:
    """按严格联合条件检测并保守去重十类高中物理高难特征。"""
    evidence_by_name: dict[str, dict[str, Any]] = {}

    object_high = (
        features.get("object_count") in {"两个对象", "三个及以上对象"}
        and features.get("object_relation") in {"双向耦合", "共同受约束"}
        and (
            features.get("model_relation") in {"模型切换", "多模型耦合"}
            or features.get("equation_structure")
            in {"2-3个方程联立", "4个以上方程或不等式组"}
        )
        and features.get("reasoning_chain")
        in {"多层因果", "逆向推理或临界分析"}
    )
    if object_high:
        evidence_by_name["多对象强耦合"] = _high_evidence(
            "多对象强耦合",
            [
                "object_count",
                "object_relation",
                "model_relation",
                "equation_structure",
                "reasoning_chain",
            ],
            [
                features.get("object_count"),
                features.get("object_relation"),
                features.get("model_relation"),
                features.get("equation_structure"),
                features.get("reasoning_chain"),
            ],
            "object_coupling",
        )

    hidden_critical = (
        features.get("critical_state") == "隐含临界"
        and features.get("hidden_conditions") in {"单个隐含条件", "多个隐含条件"}
        and features.get("reasoning_chain") == "逆向推理或临界分析"
    )
    if hidden_critical:
        evidence_by_name["隐含临界状态"] = _high_evidence(
            "隐含临界状态",
            ["critical_state", "hidden_conditions", "reasoning_chain"],
            [features.get("critical_state"), features.get("hidden_conditions"), features.get("reasoning_chain")],
            "hidden_critical",
        )

    process_state_high = (
        features.get("process_count") in {"两个过程", "三个及以上过程"}
        and features.get("state_count") in {"2个", "3个及以上"}
        and features.get("state_transition")
        in {"离散状态转换", "连续演化"}
        and features.get("process_state_relation") in {"前后状态强依赖", "连续变化伴随边界"}
    )
    # 若唯一的复杂性就是同一个隐含临界边界，保留更具体的“隐含临界状态”。
    boundary_only_duplicate = (
        hidden_critical
        and features.get("process_state_relation") == "连续变化伴随边界"
    )
    if process_state_high and not boundary_only_duplicate:
        evidence_by_name["多过程或多状态强耦合"] = _high_evidence(
            "多过程或多状态强耦合",
            [
                "process_count",
                "state_count",
                "state_transition",
                "process_state_relation",
            ],
            [
                features.get("process_count"),
                features.get("state_count"),
                features.get("state_transition"),
                features.get("process_state_relation"),
            ],
            "process_state_coupling",
        )

    if features.get("constraint_structure") == "多约束联合筛选":
        evidence_by_name["多约束联合"] = _high_evidence(
            "多约束联合",
            ["constraint_structure"],
            [features.get("constraint_structure")],
            "joint_constraints",
        )

    classification_high = (
        features.get("classification_discussion") in {"3类讨论", "4类及以上"}
        or (
            features.get("classification_discussion") == "2类讨论"
            and features.get("model_relation") in {"模型切换", "多模型耦合"}
            and features.get("equation_structure") in {"2-3个方程联立", "4个以上方程或不等式组"}
        )
    )
    # 两类讨论若只是同一个隐含临界产生的两侧，不重复计数。
    if classification_high and not (
        hidden_critical and features.get("classification_discussion") == "2类讨论"
    ):
        classification_fields = ["classification_discussion"]
        if features.get("classification_discussion") == "2类讨论":
            classification_fields.extend(["model_relation", "equation_structure"])
        evidence_by_name["复杂分类讨论"] = _high_evidence(
            "复杂分类讨论",
            classification_fields,
            [features.get(field) for field in classification_fields],
            "complex_classification",
        )

    model_high = (
        features.get("model_relation") in {"模型切换", "多模型耦合"}
        and (
            features.get("equation_structure") in {"2-3个方程联立", "4个以上方程或不等式组"}
            or features.get("process_state_relation") == "前后状态强依赖"
        )
    )
    cross_module_high = (
        features.get("knowledge_scope") == "跨模块综合"
        and features.get("model_relation") in {"模型切换", "多模型耦合"}
        and features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
    )
    if cross_module_high:
        evidence_by_name["跨模块深度综合"] = _high_evidence(
            "跨模块深度综合",
            ["knowledge_scope", "model_relation", "step_count"],
            [features.get("knowledge_scope"), features.get("model_relation"), features.get("step_count")],
            "cross_module_bridge",
        )
    # 普通“模型切换”若只是在描述跨模块桥接，不与跨模块重复；真正多模型耦合可独立保留。
    if model_high and not (
        cross_module_high and features.get("model_relation") == "模型切换"
    ):
        evidence_by_name["多模型切换或耦合"] = _high_evidence(
            "多模型切换或耦合",
            ["model_relation", "equation_structure", "process_state_relation"],
            [features.get("model_relation"), features.get("equation_structure"), features.get("process_state_relation")],
            "model_switching",
        )

    parameter_high = (
        (
            features.get("parameter_operation") == "单参数"
            and features.get("calculation_complexity") == "参数或范围计算"
            and features.get("classification_discussion")
            in {"3类讨论", "4类及以上"}
            and features.get("variable_relation")
            in {"分段或非线性关系", "多变量耦合"}
        )
        or (
            features.get("parameter_operation") in {"双参数", "多参数"}
            and features.get("calculation_complexity") == "参数或范围计算"
            and features.get("equation_structure")
            in {"2-3个方程联立", "4个以上方程或不等式组"}
            and features.get("variable_relation")
            in {"分段或非线性关系", "多变量耦合"}
        )
    )
    if parameter_high:
        parameter_fields = [
            "parameter_operation",
            "calculation_complexity",
            (
                "classification_discussion"
                if features.get("parameter_operation") == "单参数"
                else "equation_structure"
            ),
            "variable_relation",
        ]
        evidence_by_name["复杂参数、范围或极值"] = _high_evidence(
            "复杂参数、范围或极值",
            parameter_fields,
            [features.get(field) for field in parameter_fields],
            "parameter_range_extreme",
        )

    graph_high = (
        features.get("graph_structure") in {"单图反推隐藏量", "多图联合转换"}
        and features.get("variable_relation") in {"函数或图像关系", "分段或非线性关系", "多变量耦合"}
        and features.get("reasoning_chain") in {"多层因果", "逆向推理或临界分析"}
    )
    if graph_high:
        evidence_by_name["高层级图像信息转换"] = _high_evidence(
            "高层级图像信息转换",
            ["graph_structure", "variable_relation", "reasoning_chain"],
            [features.get("graph_structure"), features.get("variable_relation"), features.get("reasoning_chain")],
            "high_graph_conversion",
        )

    experiment_high = (
        features.get("experiment_requirement") in {"误差反演", "方案设计或可行性验证"}
        and features.get("reasoning_chain") in {"多层因果", "逆向推理或临界分析"}
    )
    if experiment_high:
        evidence_by_name["高阶实验设计或误差反演"] = _high_evidence(
            "高阶实验设计或误差反演",
            ["experiment_requirement", "reasoning_chain"],
            [features.get("experiment_requirement"), features.get("reasoning_chain")],
            "advanced_experiment",
        )

    names = [name for name in HIGH_DIFFICULTY_FEATURE_NAMES if name in evidence_by_name]
    possible_overlap_groups: list[list[str]] = []
    for index, left_name in enumerate(names):
        left_fields = set(evidence_by_name[left_name]["fields"])
        for right_name in names[index + 1 :]:
            right_fields = set(evidence_by_name[right_name]["fields"])
            # 共享触发字段不直接删项，只作为第二阶段的重复计数审计线索。
            # 不同物理结构可能真实共存，机械合并会漏掉强耦合综合题。
            if left_fields & right_fields:
                possible_overlap_groups.append([left_name, right_name])
    return HighDifficultyDetection(
        names=names,
        evidence=[evidence_by_name[name] for name in names],
        possible_overlap_groups=possible_overlap_groups,
    )


def enrich_stage1_rating(stage1_rating: dict[str, Any]) -> dict[str, Any]:
    """保存原始正确率，应用高难特征乘数并映射第一步档位。"""
    rating = copy.deepcopy(stage1_rating)
    features = rating.get("features")
    validate_feature_schema(features)
    rating["features_model_raw"] = copy.deepcopy(features)
    distinct_points = list(
        dict.fromkeys(str(value).strip() for value in features["knowledge_points"])
    )
    derived_knowledge_count = (
        "1个"
        if len(distinct_points) == 1
        else ("2-3个" if len(distinct_points) <= 3 else "4个及以上")
    )
    rating["knowledge_count_model_raw"] = features.get("knowledge_count")
    rating["knowledge_scope_model_raw"] = features.get("knowledge_scope")
    features["knowledge_points"] = distinct_points
    features["knowledge_count"] = derived_knowledge_count
    content_l1 = {
        value for value in features["knowledge_L1"] if value != "物理实验"
    }
    content_l2 = {
        value for value in features["knowledge_L2"] if value != "物理实验"
    }
    if len(content_l1) >= 2:
        derived_scope = "跨模块综合"
    elif len(content_l2) >= 2:
        derived_scope = "同模块跨章节"
    elif len(distinct_points) >= 2:
        derived_scope = "同章节综合"
    else:
        derived_scope = "单知识点"
    features["knowledge_scope"] = derived_scope
    try:
        base_accuracy = float(rating["predicted_accuracy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("第一阶段 predicted_accuracy 缺失或不是数值") from exc
    if not 0.0 <= base_accuracy <= 100.0:
        raise ValueError("第一阶段 predicted_accuracy 必须在 0 到 100 之间")

    active_features = detect_active_features(features)
    high = detect_high_difficulty_features(features)
    high_count = len(high.names)
    multiplier = multiplier_for_high_count(high_count)

    adjusted_accuracy = round(base_accuracy * multiplier, 1)
    rating["original_predicted_accuracy"] = base_accuracy
    rating["active_features"] = active_features
    rating["active_feature_count"] = len(active_features)
    rating["high_difficulty_features"] = high.names
    rating["high_difficulty_feature_evidence"] = high.evidence
    rating["possible_high_feature_overlaps"] = high.possible_overlap_groups
    rating["high_difficulty_feature_count"] = high_count
    rating["multiplier_applied"] = multiplier
    rating["predicted_accuracy"] = adjusted_accuracy
    rating["difficulty_level_step1"] = map_accuracy_to_level(adjusted_accuracy)
    return rating


def _safe_question_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    value = str(item.get("question_id") or "")
    try:
        return (0, f"{int(value):030d}")
    except ValueError:
        return (1, value)


def _collect_image_urls(
    question: dict[str, Any],
    keys: tuple[str, ...],
) -> list[str]:
    urls: list[str] = []
    for key in keys:
        value = str(question.get(key) or "").strip()
        if value and value not in urls:
            urls.append(value)
    for subquestion in question.get("sub_questions") or []:
        for key in keys:
            value = str(subquestion.get(key) or "").strip()
            if value and value not in urls:
                urls.append(value)
    return urls


def prepare_question(source_question: dict[str, Any], image_mode: str = "auto") -> PreparedQuestion:
    """删除原标签、复制并排序小题，判断解析和图片信息是否充分。"""
    if image_mode not in {"off", "auto", "all"}:
        raise ValueError("image_mode 只能为 off、auto、all")
    source_difficulty = copy.deepcopy(source_question.get("difficulty"))
    question = {
        field: copy.deepcopy(source_question[field])
        for field in QUESTION_MODEL_FIELDS
        if field in source_question and field != "sub_questions"
    }
    sanitized_subquestions: list[dict[str, Any]] = []
    for raw_item in source_question.get("sub_questions") or []:
        if not isinstance(raw_item, dict):
            continue
        item = {
            field: copy.deepcopy(raw_item[field])
            for field in SUBQUESTION_MODEL_FIELDS
            if field in raw_item
        }
        sanitized_subquestions.append(item)
    question["sub_questions"] = sorted(
        sanitized_subquestions,
        key=_safe_question_sort_key,
    )

    parent_analysis = str(question.get("analysis") or "").strip()
    sub_analysis_available = any(
        str(item.get("analysis") or "").strip()
        for item in question["sub_questions"]
    )
    has_analysis = bool(parent_analysis) or sub_analysis_available
    text_parts = [
        str(question.get("stem") or ""),
        str(question.get("options") or ""),
        *[
            str(item.get("stem") or "") + str(item.get("options") or "")
            for item in question["sub_questions"]
        ],
    ]
    combined_text = " ".join(text_parts).strip()
    figure_reference = bool(
        re.search(
            r"如图|图中|图示|图甲|图乙|图像|图象|下图|左图|右图|见图|"
            r"示意图|图线|电路图|装置图|轨迹图|<img",
            combined_text,
            re.IGNORECASE,
        )
    )
    image_required = not combined_text or figure_reference
    stem_urls = _collect_image_urls(
        question,
        ("stem_image_url", "stem_pic_url"),
    )
    analysis_urls = _collect_image_urls(
        question,
        ("analysis_image_url", "analysis_pic_url"),
    )
    available_urls = list(dict.fromkeys(stem_urls + analysis_urls))
    image_available = bool(available_urls)
    if image_mode == "all":
        selected_urls = available_urls
    elif image_mode == "auto" and image_required:
        selected_urls = list(stem_urls)
        all_analysis_text = " ".join(
            [
                parent_analysis,
                *[
                    str(item.get("analysis") or "")
                    for item in question["sub_questions"]
                ],
            ]
        ).strip()
        analysis_needs_image = (
            not all_analysis_text
            or bool(
                re.search(
                    r"如图|图中|下图|见图|图甲|图乙|图像|图象|图线|<img",
                    all_analysis_text,
                    re.IGNORECASE,
                )
            )
        )
        if analysis_needs_image or not selected_urls:
            selected_urls.extend(
                url for url in analysis_urls if url not in selected_urls
            )
    else:
        selected_urls = []

    if not combined_text and not selected_urls:
        sufficiency = "信息不足"
    elif image_required and not selected_urls and not has_analysis:
        sufficiency = "信息不足"
    elif not has_analysis or (image_required and not selected_urls):
        sufficiency = "部分缺失"
    else:
        sufficiency = "充分"

    return PreparedQuestion(
        question=question,
        source_difficulty_untrusted=source_difficulty,
        input_quality={
            "parent_analysis_available": bool(parent_analysis),
            "subquestion_analysis_available": sub_analysis_available,
            "has_analysis": has_analysis,
            "image_required": image_required,
            "image_available": image_available,
            "image_included": bool(selected_urls),
            "stem_image_included": any(url in selected_urls for url in stem_urls),
            "analysis_image_included": any(
                url in selected_urls for url in analysis_urls
            ),
            "input_sufficiency": sufficiency,
        },
        selected_image_urls=selected_urls,
    )


def normalize_level(value: Any) -> str:
    if value in LEVEL_INDEX:
        return str(value)
    match = re.search(r"难度[1-5]档", str(value or ""))
    return match.group(0) if match else ""


def _multiplier_bucket(count: int | None) -> str:
    if count is None:
        return "unknown"
    if count >= 4:
        return "0.70"
    if count >= 3:
        return "0.85"
    return "1.00"


def finalize_level(
    *,
    current_level: str,
    reasonableness: str,
    model_suggested_level: Any,
    multiplier_reasonableness: str,
    input_sufficiency: str,
    original_high_count: int | None = None,
    reviewed_high_count: int | None = None,
) -> FinalizationResult:
    """根据二阶段复核最多调整一档，并标记跨档或审计冲突。"""
    if current_level not in LEVEL_INDEX:
        raise ValueError(f"无效 current_level：{current_level!r}")
    current_index = LEVEL_INDEX[current_level]
    normalized_suggestion = normalize_level(model_suggested_level)
    suggested_index = LEVEL_INDEX.get(normalized_suggestion, current_index)
    manual = False

    if reasonableness == "合理":
        final_index = current_index
        if normalized_suggestion and suggested_index != current_index:
            manual = True
    elif reasonableness == "偏高":
        final_index = max(0, current_index - 1)
        if normalized_suggestion and suggested_index >= current_index:
            manual = True
    elif reasonableness == "偏低":
        final_index = min(4, current_index + 1)
        if normalized_suggestion and suggested_index <= current_index:
            manual = True
    else:
        final_index = current_index
        manual = True

    if normalized_suggestion and abs(suggested_index - current_index) >= 2:
        manual = True
    if multiplier_reasonableness != "合理":
        manual = True
    if input_sufficiency == "信息不足":
        manual = True
    if (
        original_high_count is not None
        and reviewed_high_count is not None
        and _multiplier_bucket(original_high_count) != _multiplier_bucket(reviewed_high_count)
    ):
        manual = True

    final_level = LEVEL_ORDER[final_index]
    if final_level == current_level:
        adjustment = f"{reasonableness or '未知'}·维持{final_level}"
    else:
        adjustment = f"{reasonableness}·{current_level}→{final_level}"
    return FinalizationResult(
        final_level=final_level,
        needs_manual_review=manual,
        model_suggested_level=normalized_suggestion or current_level,
        adjustment_desc=adjustment,
    )
