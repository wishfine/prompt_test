"""初中化学可观测特征 V6 正式契约。

该模块提供严格校验和派生量计算。正式特征不直接输出“推理深度”
“约束复杂度”等难度摘要，而是记录任务、规则、课程单元和具体
化学操作。V6 在 V5 十七项基础上，只增加具体作答操作和真实跨学科
依赖；移除容易受最终等级反向影响的任务性质计数。V2/V3/V4/V5
仍可严格读取，用于历史回放。
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

OBSERVABLE_V4_FEATURE_FIELDS = (
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

OBSERVABLE_V5_FEATURE_FIELDS = (
    "longest_solution_chain",
    "task_groups",
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

OBSERVABLE_FEATURE_FIELDS = (
    "longest_solution_chain",
    "task_groups",
    "rule_families",
    "response_operations",
    "curriculum_topics",
    "cross_subject_operations",
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

RESPONSE_OPERATIONS = {
    "教材事实或名称直接匹配",
    "分类标准应用",
    "完整命题正误辨析",
    "化学用语书写",
    "化学用语含义解释",
    "性质用途或现象解释",
    "实验操作规范",
    "实验作用或目的解释",
    "异常或失败原因诊断",
    "图表读取或归纳",
    "证据推断或物质鉴别",
    "定量计算",
    "方案设计或评价",
    "规范原因表达",
    "开放举例或补写",
}

CROSS_SUBJECT_OPERATIONS = {
    "文言诗词或语义转译",
    "物理过程或物理量关系",
    "生物过程或健康机制",
    "数学函数、几何或统计模型",
}
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
    "微观粒子→宏观含义",
    "微观粒子→化学符号",
    "宏观对象→化学符号",
    "宏观现象→化学符号",
    "化学符号→宏观含义",
    "化学符号→定量关系",
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
    "组分消元或组成不变量",
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
    "依赖题干未给出的超纲化学知识",
}

SOLUTION_TOPOLOGIES = {
    "单点直接回答",
    "单线性常规链",
    "条件分支或范围筛选",
    "未知组成或量反推",
    "未知组分消元或组成不变量",
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


# 与物理生产脚本相同：先把模型偶发的近义输出收敛到
# 可审计枚举，再作严格校验。只收录语义唯一的别名；无法
# 唯一判断的值仍会被严格校验拒绝，不会默认填低档。
OBSERVABLE_FIELD_ALIASES = {
    "new_ininformation_operation": "new_information_operation",
}

TASK_TYPE_ALIASES = {
    "误差分析": "实验操作与探究",
    "微观粒子表征": "化学用语",
    "微观粒子与符号转换": "化学用语",
    "化学式推断": "化学用语",
    "反应条件与速率分析": "性质与反应判断",
    "方案评价": "方案设计与评价",
}

ENUM_VALUE_ALIASES = {
    "representation_operations": {
        "宏观含义→化学符号": "宏观对象→化学符号",
        "宏观物质→化学符号": "宏观对象→化学符号",
        "宏观名称→化学符号": "宏观对象→化学符号",
        "宏观元素→化学符号": "宏观对象→化学符号",
        "宏观要求→化学符号": "宏观对象→化学符号",
        "化学式→定量关系": "化学符号→定量关系",
        "元素质量→原子个数比": "化学符号→定量关系",
        "化学方程式→宏观含义": "化学符号→宏观含义",
        "实验现象→微观粒子": "宏观现象→微观粒子",
        "宏观特征→微观粒子": "宏观现象→微观粒子",
    },
    "evidence_operations": {
        "双来源交叉验证": "多证据共同成立",
    },
    "condition_operations": {
        "条件对比": "条件切换",
        "多条件比较": "条件切换",
    },
    "experiment_operation": {
        "方案设计与评价": "方案评价或补充实验",
        "方案评价": "方案评价或补充实验",
    },
    "experiment_task_structure": {
        "数据归纳": "控制变量或数据归纳",
    },
    "solution_topology": {
        "范围或边界筛选": "条件分支或范围筛选",
    },
}

OBSERVABLE_ENUM_VALUES_BY_FIELD = {
    "task_type": TASK_TYPES,
    "rule_families": RULE_FAMILIES,
    "response_operations": RESPONSE_OPERATIONS,
    "curriculum_topics": CURRICULUM_TOPICS,
    "cross_subject_operations": CROSS_SUBJECT_OPERATIONS,
    "parallel_task_relation": PARALLEL_TASK_RELATIONS,
    "solution_topology": SOLUTION_TOPOLOGIES,
    "reaction_structure": REACTION_STRUCTURES,
    "condition_operations": CONDITION_OPERATIONS,
    "representation_operations": REPRESENTATION_OPERATIONS,
    "evidence_operations": EVIDENCE_OPERATIONS,
    "experiment_operation": EXPERIMENT_OPERATIONS,
    "experiment_task_structure": EXPERIMENT_TASK_STRUCTURES,
    "visual_task_structure": VISUAL_TASK_STRUCTURES,
    "graph_table_operation": GRAPH_TABLE_OPERATIONS,
    "error_analysis_operation": ERROR_ANALYSIS_OPERATIONS,
    "calculation_operations": CALCULATION_OPERATIONS,
    "new_information_operation": NEW_INFORMATION_OPERATIONS,
}


def _clean_enum_text(value: Any) -> str:
    return "".join(str(value or "").strip().split())


def _canonical_task_type(value: Any) -> str:
    clean = _clean_enum_text(value)
    if clean in TASK_TYPES:
        return clean
    if clean in TASK_TYPE_ALIASES:
        return TASK_TYPE_ALIASES[clean]
    if "误差" in clean:
        return "实验操作与探究"
    if any(word in clean for word in ("微观", "化学式", "化学符号")):
        return "化学用语"
    if any(word in clean for word in ("方案设计", "方案评价")):
        return "方案设计与评价"
    if any(word in clean for word in ("反应条件", "反应速率")):
        return "性质与反应判断"
    return clean


def _canonical_enum_value(field: str, value: Any) -> str:
    clean = _clean_enum_text(value)
    allowed = OBSERVABLE_ENUM_VALUES_BY_FIELD[field]
    if clean in allowed:
        return clean
    return ENUM_VALUE_ALIASES.get(field, {}).get(clean, clean)


def normalize_observable_features(
    features: Any,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """按物理生产逻辑对常见近义枚举作确定性归一。

    未知值不填默认值，后续严格校验仍会拒绝，避免把
    语义不明的输出静默改成低档特征。
    """
    if not isinstance(features, dict):
        return features, []
    normalized: Dict[str, Any] = {}
    actions: List[Dict[str, Any]] = []

    def record(
        field: str,
        old: Any,
        new: Any,
        reason: str,
        *,
        force: bool = False,
    ) -> None:
        if force or old != new:
            actions.append(
                {"field": field, "from": old, "to": new, "reason": reason}
            )

    for raw_key, value in features.items():
        clean_key = _clean_enum_text(raw_key)
        key = OBSERVABLE_FIELD_ALIASES.get(clean_key, clean_key)
        record("features.key", clean_key, key, "字段名别名归一")
        if key in normalized and clean_key != key:
            continue
        normalized[key] = copy.deepcopy(value)

    groups = normalized.get("task_groups")
    if isinstance(groups, list):
        rebuilt: List[Any] = []
        positions: Dict[str, int] = {}
        for group in groups:
            if not isinstance(group, dict) or set(group) != {"task_type", "count"}:
                rebuilt.append(group)
                continue
            old_type = group.get("task_type")
            new_type = _canonical_task_type(old_type)
            record("task_groups.task_type", old_type, new_type, "任务类型近义归一")
            count = group.get("count")
            if (
                new_type in positions
                and isinstance(count, int)
                and not isinstance(count, bool)
                and isinstance(rebuilt[positions[new_type]].get("count"), int)
            ):
                rebuilt[positions[new_type]]["count"] += count
            else:
                positions[new_type] = len(rebuilt)
                rebuilt.append({"task_type": new_type, "count": count})
        record("task_groups", groups, rebuilt, "归一后合并重复任务类型")
        normalized["task_groups"] = rebuilt

    if isinstance(normalized.get("rule_families"), list):
        old_values = normalized["rule_families"]
        new_values: List[str] = []
        for value in old_values:
            canonical = _canonical_task_type(value)
            if canonical not in new_values:
                new_values.append(canonical)
        record("rule_families", old_values, new_values, "规则族近义归一与去重")
        normalized["rule_families"] = new_values

    for field in (
        "response_operations",
        "cross_subject_operations",
        "condition_operations",
        "representation_operations",
        "evidence_operations",
        "calculation_operations",
    ):
        values = normalized.get(field)
        if not isinstance(values, list):
            continue
        canonical_values: List[str] = []
        for value in values:
            canonical = _canonical_enum_value(field, value)
            if canonical not in canonical_values:
                canonical_values.append(canonical)
        record(field, values, canonical_values, "枚举近义归一与去重")
        normalized[field] = canonical_values

    # 表征转换和定量计算是两个不同侧面。模型偶尔会把合法的表征枚举
    # 写进 calculation_operations；这种串位可以无损修复，但不能凭空
    # 猜测缺失的计算方法。若移动后定量任务没有计算操作，后续一致性
    # 校验仍会要求模型重试。
    representation_values = normalized.get("representation_operations")
    calculation_values = normalized.get("calculation_operations")
    if isinstance(representation_values, list) and isinstance(
        calculation_values, list
    ):
        misplaced_values = [
            value
            for value in calculation_values
            if value in REPRESENTATION_OPERATIONS
        ]
        for misplaced in misplaced_values:
            calculation_values.remove(misplaced)
            if misplaced not in representation_values:
                representation_values.append(misplaced)
            record(
                "calculation_operations",
                misplaced,
                "representation_operations",
                "表征转换值从计算字段移回表征字段",
                force=True,
            )

    conditions = normalized.get("condition_operations")
    evidence = normalized.get("evidence_operations")
    if isinstance(conditions, list) and isinstance(evidence, list):
        for misplaced in ("多证据共同成立", "排除多个候选解释"):
            if misplaced in conditions:
                conditions.remove(misplaced)
                if misplaced not in evidence:
                    evidence.append(misplaced)
                record(
                    "condition_operations→evidence_operations",
                    misplaced,
                    misplaced,
                    "证据操作字段串位修复",
                    force=True,
                )
        if "分类讨论" in evidence:
            evidence.remove("分类讨论")
            if "分类讨论" not in conditions:
                conditions.append("分类讨论")
            record(
                "evidence_operations→condition_operations",
                "分类讨论",
                "分类讨论",
                "条件操作字段串位修复",
                force=True,
            )

    for field in (
        "parallel_task_relation",
        "solution_topology",
        "reaction_structure",
        "experiment_operation",
        "experiment_task_structure",
        "visual_task_structure",
        "graph_table_operation",
        "error_analysis_operation",
        "new_information_operation",
    ):
        if field not in normalized:
            continue
        old_value = normalized[field]
        new_value = _canonical_enum_value(field, old_value)
        record(field, old_value, new_value, "枚举近义归一")
        normalized[field] = new_value

    for field in ("curriculum_topics", "curriculum_units", "longest_solution_chain"):
        values = normalized.get(field)
        if not isinstance(values, list):
            continue
        deduped: List[Any] = []
        for value in values:
            clean_value = value.strip() if isinstance(value, str) else value
            if clean_value not in deduped:
                deduped.append(clean_value)
        record(field, values, deduped, "数组去重")
        normalized[field] = deduped

    graph_op = normalized.get("graph_table_operation")
    if (
        graph_op in GRAPH_TABLE_OPERATIONS - {"无"}
        and normalized.get("visual_task_structure") == "无必要视觉信息"
    ):
        visual = (
            "单图直接识别"
            if graph_op == "直接读数"
            else "共享装置流程或图表模型"
        )
        record(
            "visual_task_structure",
            normalized["visual_task_structure"],
            visual,
            "由已填图表操作修复视觉一致性",
        )
        normalized["visual_task_structure"] = visual

    experiment_op = normalized.get("experiment_operation")
    experiment_structure = normalized.get("experiment_task_structure")
    has_experiment_task = any(
        isinstance(group, dict)
        and group.get("task_type") == "实验操作与探究"
        for group in normalized.get("task_groups", [])
    )
    if experiment_op == "无" and (
        has_experiment_task
        or normalized.get(
            "error_analysis_operation",
            "无误差分析",
        ) != "无误差分析"
        or experiment_structure not in {None, "无实验判断"}
    ):
        operation_by_structure = {
            "控制变量或数据归纳": "数据归纳",
            "方案设计或评价": "方案评价或补充实验",
            "操作偏差因果链": "基础操作或读数",
            "多仪器或多条件比较": "基础操作或读数",
            "名称或单点规范匹配": "基础操作或读数",
        }
        inferred = operation_by_structure.get(
            experiment_structure,
            "基础操作或读数",
        )
        record(
            "experiment_operation",
            experiment_op,
            inferred,
            "由已填实验任务事实修复内部一致性",
        )
        normalized["experiment_operation"] = inferred
        experiment_op = inferred
    if experiment_op not in {None, "无"} and experiment_structure == "无实验判断":
        structure_by_operation = {
            "基础操作或读数": "名称或单点规范匹配",
            "变量控制": "控制变量或数据归纳",
            "现象解释": "控制变量或数据归纳",
            "数据归纳": "控制变量或数据归纳",
            "方案设计": "方案设计或评价",
            "方案评价或补充实验": "方案设计或评价",
            "多阶段定量探究": "方案设计或评价",
        }
        inferred = structure_by_operation[experiment_op]
        record(
            "experiment_task_structure",
            experiment_structure,
            inferred,
            "由已填实验操作修复内部一致性",
        )
        normalized["experiment_task_structure"] = inferred

    return normalized, actions


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
        cross_field_hint = ""
        if (
            field == "experiment_operation"
            and value in EXPERIMENT_TASK_STRUCTURES
        ):
            cross_field_hint = (
                "；该值属于experiment_task_structure，"
                "experiment_operation必须改填实际执行的实验操作"
            )
        elif (
            field == "experiment_task_structure"
            and value in EXPERIMENT_OPERATIONS
        ):
            cross_field_hint = (
                "；该值属于experiment_operation，"
                "experiment_task_structure必须改填实验任务的组织结构"
            )
        raise ValueError(
            f"{field}不在合法枚举中: {value!r}{cross_field_hint}；"
            f"允许值={sorted(allowed)}"
        )
    return value


def validate_observable_features(features: Any) -> Dict[str, Any]:
    """严格校验可观测特征 V6，并兼容读取 V5/V4/V3/V2。

    校验器不静默补默认值：缺字段、多字段或枚举错误均直接
    拒绝，便于重试提示精确修正。
    """
    if not isinstance(features, dict):
        raise ValueError("features必须是JSON对象")
    actual = set(features)
    v6_expected = set(OBSERVABLE_FEATURE_FIELDS)
    v5_expected = set(OBSERVABLE_V5_FEATURE_FIELDS)
    v4_expected = set(OBSERVABLE_V4_FEATURE_FIELDS)
    v3_expected = set(OBSERVABLE_V3_FEATURE_FIELDS)
    v2_expected = set(OBSERVABLE_V2_FEATURE_FIELDS)
    is_v6 = actual == v6_expected
    is_v5 = actual == v5_expected
    is_v4 = actual == v4_expected
    is_v3 = actual == v3_expected
    is_v2 = actual == v2_expected
    if not (is_v6 or is_v5 or is_v4 or is_v3 or is_v2):
        missing = sorted(v6_expected - actual)
        extra = sorted(actual - v6_expected)
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
    if is_v6 or is_v5 or is_v4:
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
    if is_v6:
        _validate_unique_enum_list(
            validated,
            "response_operations",
            RESPONSE_OPERATIONS,
            allow_empty=False,
        )
        _validate_unique_enum_list(
            validated,
            "cross_subject_operations",
            CROSS_SUBJECT_OPERATIONS,
            allow_empty=True,
        )
    if is_v3 or is_v4 or is_v5 or is_v6:
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
    if is_v3 or is_v4 or is_v5 or is_v6:
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
    if is_v3 or is_v4 or is_v5 or is_v6:
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
        (is_v3 or is_v4 or is_v5 or is_v6)
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
        (is_v3 or is_v4 or is_v5 or is_v6)
        and validated["error_analysis_operation"] != "无误差分析"
        and validated["experiment_operation"] == "无"
    ):
        raise ValueError("误差分析任务必须记录experiment_operation")
    if is_v6 or is_v5 or is_v4:
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
    if is_v6:
        invariant_topology = (
            validated["solution_topology"]
            == "未知组分消元或组成不变量"
        )
        invariant_operation = (
            "组分消元或组成不变量"
            in validated["calculation_operations"]
        )
        if invariant_topology and not invariant_operation:
            raise ValueError(
                "solution_topology为未知组分消元或组成不变量时，"
                "calculation_operations必须包含组分消元或组成不变量"
            )
        if invariant_operation and not invariant_topology:
            raise ValueError(
                "calculation_operations包含组分消元或组成不变量时，"
                "solution_topology必须为未知组分消元或组成不变量"
            )
    return validated


def derive_observable_metrics(
    features: Dict[str, Any],
) -> Dict[str, Any]:
    """从可审计数组派生 D/B/W/U 等数量，不接受模型自报计数。"""
    validated = validate_observable_features(features)
    has_topic_contract = "curriculum_topics" in validated
    if has_topic_contract:
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

    if has_topic_contract:
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

    if has_topic_contract:
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
        "response_operation_count": len(
            validated.get("response_operations", [])
        ),
        "cross_subject_operation_count": len(
            validated.get("cross_subject_operations", [])
        ),
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
