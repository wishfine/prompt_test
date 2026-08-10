"""初中化学可观测特征 V2 正式契约。

该模块提供严格校验和派生量计算。正式特征不直接输出“推理深度”
“约束复杂度”等难度摘要，而是记录任务、规则、课程单元和具体
化学操作；历史 Core-12 兼容逻辑保留在运行脚本中。
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List


OBSERVABLE_FEATURE_FIELDS = (
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
    """严格校验可观测特征 V2。

    校验器不静默补默认值：缺字段、多字段或枚举错误均直接
    拒绝，便于重试提示精确修正。
    """
    if not isinstance(features, dict):
        raise ValueError("features必须是JSON对象")
    expected = set(OBSERVABLE_FEATURE_FIELDS)
    actual = set(features)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
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

    _validate_unique_enum_list(
        validated,
        "rule_families",
        RULE_FAMILIES,
        allow_empty=False,
    )
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
    return validated


def derive_observable_metrics(
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """从可审计数组派生 D/B/W/U 等数量，不接受模型自报计数。"""
    validated = validate_observable_features(features)
    return {
        "longest_chain_steps": len(
            validated["longest_solution_chain"]
        ),
        "effective_task_count": sum(
            group["count"] for group in validated["task_groups"]
        ),
        "task_group_count": len(validated["task_groups"]),
        "rule_family_count": len(validated["rule_families"]),
        "curriculum_unit_count": len(
            validated["curriculum_units"]
        ),
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
        "has_task_dependency": len(
            validated["longest_solution_chain"]
        ) >= 2,
    }
