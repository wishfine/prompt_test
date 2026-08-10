"""初中化学可观测特征 V4 正式契约。

该模块提供严格校验和派生量计算。正式特征不直接输出“推理深度”
“约束复杂度”等难度摘要，而是记录任务、规则、课程单元和具体
化学操作。V4 在 V3 基础上增加任务性质计数、解题拓扑和实验任务
结构；V2/V3 仍可严格读取，用于历史回放。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List


OBSERVABLE_V2_FEATURE_FIELDS = (
    "longest_solution_chain",
    "task_groups",
    "rule_families",
    "curriculum_units",
    "reaction_structure",
    "condition_operations",
    "representation_operations",
    "evidence_operations",
    "experiment_operation",
    "graph_table_operation",
    "calculation_operations",
    "new_information_operation",
)

OBSERVABLE_V3_FEATURE_FIELDS = (
    "longest_solution_chain",
    "task_groups",
    "rule_families",
    "curriculum_topics",
    "parallel_task_relation",
    "reaction_structure",
    "condition_operations",
    "representation_operations",
    "evidence_operations",
    "experiment_operation",
    "visual_task_structure",
    "graph_table_operation",
    "error_analysis_operation",
    "calculation_operations",
    "new_information_operation",
)

OBSERVABLE_FEATURE_FIELDS = (
    "longest_solution_chain",
    "task_groups",
    "direct_retrieval_task_count",
    "rule_application_task_count",
    "rule_families",
    "curriculum_topics",
    "parallel_task_relation",
    "solution_topology",
    "reaction_structure",
    "condition_operations",
    "representation_operations",
    "evidence_operations",
    "experiment_operation",
    "experiment_task_structure",
    "visual_task_structure",
    "graph_table_operation",
    "error_analysis_operation",
    "calculation_operations",
    "new_information_operation",
)

TASK_TYPES = {
    "直接事实与概念",
    "化学用语",
    "性质与反应判断",
    "实验操作与探究",
    "图表与数据",
    "证据推断",
    "定量计算",
    "方案设计与评价",
    "新信息应用",
}

RULE_FAMILIES = set(TASK_TYPES)
CURRICULUM_UNITS = {f"U{i}" for i in range(1, 12)}

CURRICULUM_TOPIC_NAMES = {
    "U1-1": "物质的变化和性质",
    "U1-2": "化学实验与科学探究",
    "U2-1": "空气",
    "U2-2": "氧气",
    "U2-3": "制取氧气",
    "U3-1": "分子和原子",
    "U3-2": "原子结构",
    "U3-3": "元素",
    "U4-1": "水资源及其利用",
    "U4-2": "水的组成",
    "U4-3": "物质组成的表示",
    "U5-1": "质量守恒定律",
    "U5-2": "化学方程式",
    "U6-1": "碳单质的多样性",
    "U6-2": "碳的氧化物",
    "U6-3": "二氧化碳的实验室制取",
    "U7-1": "燃料的燃烧",
    "U7-2": "化石能源的利用",
    "U8-1": "金属材料",
    "U8-2": "金属的化学性质",
    "U8-3": "金属资源的利用和保护",
    "U9-1": "溶液及其应用",
    "U9-2": "溶解度",
    "U9-3": "溶质的质量分数",
    "U10-1": "溶液的酸碱性",
    "U10-2": "常见的酸和碱",
    "U10-3": "常见的盐",
    "U11-1": "化学与人体健康",
    "U11-2": "化学与可持续发展",
}
CURRICULUM_TOPICS = set(CURRICULUM_TOPIC_NAMES)

PARALLEL_TASK_RELATIONS = {
    "单一答题目标",
    "同一规则下多个对象",
    "不同规则的独立任务",
    "共享同一化学模型的关联任务",
}

VISUAL_TASK_STRUCTURES = {
    "无必要视觉信息",
    "单图直接识别",
    "多图独立同规则识别",
    "多图独立不同规则判断",
    "共享装置流程或图表模型",
}

ERROR_ANALYSIS_OPERATIONS = {
    "无误差分析",
    "直接判断错误操作后果",
    "读数偏差到实际量判断",
    "操作偏差到最终结果方向",
    "多因素误差比较",
    "定量误差修正",
}

REACTION_STRUCTURES = {
    "无反应任务",
    "单一反应",
    "多个并列反应",
    "产物进入后一反应",
    "先后竞争或过量不足",
    "分情况反应模型",
}

CONDITION_OPERATIONS = {
    "条件直接读取",
    "条件切换",
    "反应先后",
    "过量不足",
    "范围或边界",
    "分类讨论",
    "干扰条件排除",
}

REPRESENTATION_OPERATIONS = {
    "宏观现象→微观粒子",
    "微观粒子→化学符号",
    "宏观现象→化学符号",
    "化学符号→宏观含义",
    "化学方程式→定量关系",
    "图表数据→化学关系",
    "文字新信息→化学关系",
}

EVIDENCE_OPERATIONS = {
    "单证据直接匹配",
    "多证据共同成立",
    "排除一个候选",
    "排除多个候选解释",
    "处理冲突证据",
    "补充实验获得唯一结论",
}

EXPERIMENT_OPERATIONS = {
    "无",
    "基础操作或读数",
    "变量控制",
    "现象解释",
    "数据归纳",
    "方案设计",
    "方案评价或补充实验",
    "多阶段定量探究",
}

GRAPH_TABLE_OPERATIONS = {
    "无",
    "直接读数",
    "多组比较",
    "趋势判断",
    "拐点平台或分段",
    "多图表联合",
}

CALCULATION_OPERATIONS = {
    "直接比例",
    "单一方程式",
    "单一守恒",
    "差量",
    "多反应定量关系",
    "联立",
    "范围或分类计算",
}

NEW_INFORMATION_OPERATIONS = {
    "无新信息",
    "直接查值",
    "给定关系直接代入",
    "根据新信息建立一个关系",
    "新关系被多个任务共同使用",
}

SOLUTION_TOPOLOGIES = {
    "单点直接回答",
    "单线性常规链",
    "条件分支或范围筛选",
    "未知组成或量反推",
    "双来源交叉验证",
    "多阶段反应网络",
}

EXPERIMENT_TASK_STRUCTURES = {
    "无实验判断",
    "名称或单点规范匹配",
    "多仪器或多条件比较",
    "操作偏差因果链",
    "控制变量或数据归纳",
    "方案设计或评价",
}


def _validate_unique_enum_list(
    features: Dict[str, Any],
    field: str,
    allowed: set[str],
    *,
    allow_empty: bool,
) -> List[str]:
    value = features[field]
    if not isinstance(value, list):
        raise ValueError(f"{field}必须是数组")
    if not allow_empty and not value:
        raise ValueError(f"{field}不能为空")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field}只能包含字符串")
    if len(value) != len(set(value)):
        raise ValueError(f"{field}存在重复值")
    invalid = [item for item in value if item not in allowed]
    if invalid:
        raise ValueError(f"{field}包含非法枚举: {invalid}")
    return value


def _validate_single_enum(
    features: Dict[str, Any],
    field: str,
    allowed: set[str],
) -> str:
    value = features[field]
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"{field}不在合法枚举中: {value!r}")
    return value


def validate_observable_features(features: Any) -> Dict[str, Any]:
    """严格校验可观测特征 V4，并兼容读取 V3/V2。

    校验器不静默补默认值：缺字段、多字段或枚举错误均直接
    拒绝，便于重试提示精确修正。
    """
    if not isinstance(features, dict):
        raise ValueError("features必须是JSON对象")
    actual = set(features)
    v4_expected = set(OBSERVABLE_FEATURE_FIELDS)
    v3_expected = set(OBSERVABLE_V3_FEATURE_FIELDS)
    v2_expected = set(OBSERVABLE_V2_FEATURE_FIELDS)
    is_v4 = actual == v4_expected
    is_v3 = actual == v3_expected
    is_v2 = actual == v2_expected
    if not (is_v4 or is_v3 or is_v2):
        missing = sorted(v4_expected - actual)
        extra = sorted(actual - v4_expected)
        raise ValueError(
            f"可观测特征字段集不匹配; missing={missing}; extra={extra}"
        )

    validated = copy.deepcopy(features)
    chain = validated["longest_solution_chain"]
    if not isinstance(chain, list) or not 1 <= len(chain) <= 12:
        raise ValueError(
            "longest_solution_chain必须包含1到12个必要化学决策步骤"
        )
    if any(
        not isinstance(step, str)
        or not step.strip()
        or len(step.strip()) > 80
        for step in chain
    ):
        raise ValueError(
            "longest_solution_chain的每一步必须是1到80字的具体操作"
        )
    chain = [step.strip() for step in chain]
    if len(chain) != len(set(chain)):
        raise ValueError("longest_solution_chain不得重复同一步骤")
    validated["longest_solution_chain"] = chain

    task_groups = validated["task_groups"]
    if not isinstance(task_groups, list) or not 1 <= len(task_groups) <= 12:
        raise ValueError("task_groups必须包含1到12组非重复任务")
    normalized_groups: List[Dict[str, Any]] = []
    seen_group_types: set[str] = set()
    for group in task_groups:
        if not isinstance(group, dict) or set(group) != {
            "task_type",
            "count",
        }:
            raise ValueError(
                "task_groups每项必须且只能包含task_type和count"
            )
        task_type = group["task_type"]
        count = group["count"]
        if task_type not in TASK_TYPES:
            raise ValueError(f"task_type非法: {task_type!r}")
        if task_type in seen_group_types:
            raise ValueError(f"task_groups存在重复任务类型: {task_type}")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 20:
            raise ValueError("task_groups.count必须是1到20的整数")
        seen_group_types.add(task_type)
        normalized_groups.append(
            {"task_type": task_type, "count": count}
        )
    validated["task_groups"] = normalized_groups

    if is_v4:
        direct_count = validated["direct_retrieval_task_count"]
        application_count = validated["rule_application_task_count"]
        for field, value in (
            ("direct_retrieval_task_count", direct_count),
            ("rule_application_task_count", application_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field}必须是非负整数")
        if direct_count + application_count != sum(
            group["count"] for group in normalized_groups
        ):
            raise ValueError(
                "任务性质计数必须恰好覆盖task_groups中的全部有效任务"
            )
        _validate_single_enum(
            validated,
            "solution_topology",
            SOLUTION_TOPOLOGIES,
        )
        _validate_single_enum(
            validated,
            "experiment_task_structure",
            EXPERIMENT_TASK_STRUCTURES,
        )

    _validate_unique_enum_list(
        validated,
        "rule_families",
        RULE_FAMILIES,
        allow_empty=False,
    )
    if is_v3 or is_v4:
        _validate_unique_enum_list(
            validated,
            "curriculum_topics",
            CURRICULUM_TOPICS,
            allow_empty=False,
        )
        _validate_single_enum(
            validated,
            "parallel_task_relation",
            PARALLEL_TASK_RELATIONS,
        )
    else:
        _validate_unique_enum_list(
            validated,
            "curriculum_units",
            CURRICULUM_UNITS,
            allow_empty=False,
        )
    _validate_single_enum(
        validated,
        "reaction_structure",
        REACTION_STRUCTURES,
    )
    _validate_unique_enum_list(
        validated,
        "condition_operations",
        CONDITION_OPERATIONS,
        allow_empty=True,
    )
    _validate_unique_enum_list(
        validated,
        "representation_operations",
        REPRESENTATION_OPERATIONS,
        allow_empty=True,
    )
    _validate_unique_enum_list(
        validated,
        "evidence_operations",
        EVIDENCE_OPERATIONS,
        allow_empty=True,
    )
    _validate_single_enum(
        validated,
        "experiment_operation",
        EXPERIMENT_OPERATIONS,
    )
    if is_v3 or is_v4:
        _validate_single_enum(
            validated,
            "visual_task_structure",
            VISUAL_TASK_STRUCTURES,
        )
    _validate_single_enum(
        validated,
        "graph_table_operation",
        GRAPH_TABLE_OPERATIONS,
    )
    _validate_unique_enum_list(
        validated,
        "calculation_operations",
        CALCULATION_OPERATIONS,
        allow_empty=True,
    )
    if is_v3 or is_v4:
        _validate_single_enum(
            validated,
            "error_analysis_operation",
            ERROR_ANALYSIS_OPERATIONS,
        )
    _validate_single_enum(
        validated,
        "new_information_operation",
        NEW_INFORMATION_OPERATIONS,
    )

    graph_conversions = [
        value
        for value in validated["representation_operations"]
        if value.startswith("图表数据→")
    ]
    if graph_conversions and validated["graph_table_operation"] == "无":
        raise ValueError("存在图表转换时graph_table_operation不能为无")
    if (
        (is_v3 or is_v4)
        and validated["graph_table_operation"] != "无"
        and validated["visual_task_structure"] == "无必要视觉信息"
    ):
        raise ValueError("存在图表任务时visual_task_structure不能为无必要视觉信息")
    if (
        any(
            group["task_type"] == "定量计算"
            for group in normalized_groups
        )
        and not validated["calculation_operations"]
    ):
        raise ValueError("定量计算任务必须记录calculation_operations")
    if (
        any(
            group["task_type"] == "实验操作与探究"
            for group in normalized_groups
        )
        and validated["experiment_operation"] == "无"
    ):
        raise ValueError("实验任务必须记录experiment_operation")
    if (
        (is_v3 or is_v4)
        and validated["error_analysis_operation"] != "无误差分析"
        and validated["experiment_operation"] == "无"
    ):
        raise ValueError("误差分析任务必须记录experiment_operation")
    if is_v4:
        experiment_structure = validated["experiment_task_structure"]
        if (
            validated["experiment_operation"] == "无"
            and experiment_structure != "无实验判断"
        ):
            raise ValueError("experiment_operation=无时实验任务结构必须为无实验判断")
        if (
            validated["experiment_operation"] != "无"
            and experiment_structure == "无实验判断"
        ):
            raise ValueError("存在实验操作时experiment_task_structure不能为无实验判断")
    return validated


def derive_observable_metrics(
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """从可审计数组派生 D/B/W/U 等数量，不接受模型自报计数。"""
    validated = validate_observable_features(features)
    is_v3_or_v4 = "curriculum_topics" in validated
    if is_v3_or_v4:
        curriculum_topics = validated["curriculum_topics"]
        curriculum_units = sorted(
            {topic.split("-", 1)[0] for topic in curriculum_topics}
        )
        topic_count = len(curriculum_topics)
        if len(curriculum_units) >= 2:
            curriculum_span_type = "跨单元"
        elif topic_count >= 2:
            curriculum_span_type = "同单元跨课题"
        else:
            curriculum_span_type = "单一课题"
    else:
        curriculum_units = validated["curriculum_units"]
        topic_count = len(curriculum_units)
        curriculum_span_type = (
            "跨单元" if len(curriculum_units) >= 2 else "单一课题"
        )

    if is_v3_or_v4:
        parallel_relation = validated["parallel_task_relation"]
        has_task_dependency = bool(
            len(validated["longest_solution_chain"]) >= 2
            and parallel_relation
            in {
                "单一答题目标",
                "共享同一化学模型的关联任务",
            }
        )
        if curriculum_span_type == "单一课题":
            curriculum_coupling_type = "单一课题"
        elif parallel_relation in {
            "同一规则下多个对象",
            "不同规则的独立任务",
        }:
            curriculum_coupling_type = (
                f"{curriculum_span_type}并列"
            )
        else:
            curriculum_coupling_type = (
                f"{curriculum_span_type}耦合"
            )
    else:
        # V2没有并列任务关系字段，只保留历史派生语义。
        curriculum_coupling_type = curriculum_span_type
        has_task_dependency = bool(
            len(validated["longest_solution_chain"]) >= 2
        )

    if is_v3_or_v4:
        curriculum_span_summary = (
            f"{curriculum_coupling_type}（"
            + "、".join(curriculum_topics)
            + "）"
        )
    else:
        curriculum_span_summary = (
            f"{curriculum_coupling_type}（"
            + "、".join(curriculum_units)
            + "）"
        )

    return {
        "longest_chain_steps": len(
            validated["longest_solution_chain"]
        ),
        "effective_task_count": sum(
            group["count"] for group in validated["task_groups"]
        ),
        "task_group_count": len(validated["task_groups"]),
        "rule_family_count": len(validated["rule_families"]),
        "curriculum_topic_count": topic_count,
        "curriculum_unit_count": len(curriculum_units),
        "curriculum_span_type": curriculum_span_type,
        "curriculum_coupling_type": curriculum_coupling_type,
        "curriculum_span_summary": curriculum_span_summary,
        "condition_operation_count": len(
            validated["condition_operations"]
        ),
        "representation_operation_count": len(
            validated["representation_operations"]
        ),
        "evidence_operation_count": len(
            validated["evidence_operations"]
        ),
        "calculation_operation_count": len(
            validated["calculation_operations"]
        ),
        "has_task_dependency": has_task_dependency,
        "direct_retrieval_task_count": validated.get(
            "direct_retrieval_task_count"
        ),
        "rule_application_task_count": validated.get(
            "rule_application_task_count"
        ),
        "solution_topology": validated.get("solution_topology"),
        "experiment_task_structure": validated.get(
            "experiment_task_structure"
        ),
    }
