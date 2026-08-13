"""初中化学可观测特征 V5 正式契约。

该模块提供严格校验和派生量计算。正式特征不直接输出“推理深度”
“约束复杂度”等难度摘要，而是记录任务、规则、课程单元和具体
化学操作。正式模型输出恢复为稳定的 V5 十七项；V6 增加的具体
作答操作和跨学科依赖仅保留历史读取能力，不再要求模型填写。
V2/V3/V4/V6 仍可严格读取，用于历史回放。
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

OBSERVABLE_V6_FEATURE_FIELDS = (
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

# 正式模型输出使用稳定的 V5 十七项。V6 十九项只用于历史回放，
# 避免新增枚举继续增加 Schema 重试并干扰五档边界判断。
OBSERVABLE_FEATURE_FIELDS = OBSERVABLE_V5_FEATURE_FIELDS

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
    "其他未归类任务（仅审计）",
}

# rule_families不再复刻task_groups的粗任务类型，而是记录学生实际
# 切换的回答规则。保持单字段、有限枚举，避免重新引入V6的额外字段
# 和Schema负担；旧九类输出会在归一层确定性映射到下列新枚举。
RULE_FAMILIES = {
    "教材事实直接匹配",
    "分类或概念辨析",
    "化学用语书写",
    "化学用语含义辨析",
    "性质用途或现象判断",
    "反应关系或条件判断",
    "实验操作规范",
    "作用目的或原因解释",
    "异常失败或误差诊断",
    "图表读取或数据归纳",
    "证据推断或鉴别除杂",
    "定量关系与计算",
    "方案设计或评价",
    "新信息迁移",
    "跨学科语义或模型应用",
    "其他未归类规则（仅审计）",
}

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
    "其他未归类作答操作（仅审计）",
}

CROSS_SUBJECT_OPERATIONS = {
    "文言诗词或语义转译",
    "物理过程或物理量关系",
    "生物过程或健康机制",
    "数学函数、几何或统计模型",
    "其他未归类跨学科操作（仅审计）",
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
    "U_OTHER": "其他未归类课题（仅审计）",
}
CURRICULUM_TOPICS = set(CURRICULUM_TOPIC_NAMES)

PARALLEL_TASK_RELATIONS = {
    "单一答题目标",
    "同一规则下多个对象",
    "不同规则的独立任务",
    "共享同一化学模型的关联任务",
    "其他未归类并列关系（仅审计）",
}

VISUAL_TASK_STRUCTURES = {
    "无必要视觉信息",
    "单图直接识别",
    "多图独立同规则识别",
    "多图独立不同规则判断",
    "共享装置流程或图表模型",
    "其他未归类视觉结构（仅审计）",
}

ERROR_ANALYSIS_OPERATIONS = {
    "无误差分析",
    "直接判断错误操作后果",
    "读数偏差到实际量判断",
    "操作偏差到最终结果方向",
    "多因素误差比较",
    "定量误差修正",
    "其他未归类误差操作（仅审计）",
}

REACTION_STRUCTURES = {
    "无反应任务",
    "单一反应",
    "多个并列反应",
    "产物进入后一反应",
    "先后竞争或过量不足",
    "分情况反应模型",
    "其他未归类反应结构（仅审计）",
}

CONDITION_OPERATIONS = {
    "条件直接读取",
    "条件切换",
    "反应先后",
    "过量不足",
    "范围或边界",
    "分类讨论",
    "干扰条件排除",
    "其他未归类条件操作（仅审计）",
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
    "其他未归类表征操作（仅审计）",
}

EVIDENCE_OPERATIONS = {
    "单证据直接匹配",
    "多证据共同成立",
    "排除一个候选",
    "排除多个候选解释",
    "处理冲突证据",
    "补充实验获得唯一结论",
    "其他未归类证据操作（仅审计）",
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
    "其他未归类实验操作（仅审计）",
}

GRAPH_TABLE_OPERATIONS = {
    "无",
    "直接读数",
    "多组比较",
    "趋势判断",
    "拐点平台或分段",
    "多图表联合",
    "流程或关系图解析",
    "其他未归类图表操作（仅审计）",
}

CALCULATION_OPERATIONS = {
    "直接比例",
    "化学式组成计算",
    "单一方程式",
    "单一守恒",
    "组分消元或组成不变量",
    "差量",
    "多反应定量关系",
    "联立",
    "范围或分类计算",
    "其他未归类计算操作（仅审计）",
}

NEW_INFORMATION_OPERATIONS = {
    "无新信息",
    "直接查值",
    "给定关系直接代入",
    "根据新信息建立一个关系",
    "新关系被多个任务共同使用",
    "依赖题干未给出的超纲化学知识",
    "其他未归类新信息操作（仅审计）",
}

SOLUTION_TOPOLOGIES = {
    "单点直接回答",
    "单线性常规链",
    "条件分支或范围筛选",
    "未知组成或量反推",
    "未知组分消元或组成不变量",
    "双来源交叉验证",
    "多阶段反应网络",
    "其他未归类解题拓扑（仅审计）",
}

EXPERIMENT_TASK_STRUCTURES = {
    "无实验判断",
    "名称或单点规范匹配",
    "多仪器或多条件比较",
    "操作偏差因果链",
    "控制变量或数据归纳",
    "方案设计或评价",
    "其他未归类实验结构（仅审计）",
}


# 中文枚举是唯一传输契约。下列值只供程序内部兜底；模型看不到，
# 也不参与计数或触发自动升档。
OBSERVABLE_FALLBACK_LABEL_BY_FIELD = {
    "task_type": "其他未归类任务（仅审计）",
    "rule_families": "其他未归类规则（仅审计）",
    "response_operations": "其他未归类作答操作（仅审计）",
    "cross_subject_operations": "其他未归类跨学科操作（仅审计）",
    "parallel_task_relation": "其他未归类并列关系（仅审计）",
    "solution_topology": "其他未归类解题拓扑（仅审计）",
    "reaction_structure": "其他未归类反应结构（仅审计）",
    "condition_operations": "其他未归类条件操作（仅审计）",
    "representation_operations": "其他未归类表征操作（仅审计）",
    "evidence_operations": "其他未归类证据操作（仅审计）",
    "experiment_operation": "其他未归类实验操作（仅审计）",
    "experiment_task_structure": "其他未归类实验结构（仅审计）",
    "visual_task_structure": "其他未归类视觉结构（仅审计）",
    "graph_table_operation": "其他未归类图表操作（仅审计）",
    "error_analysis_operation": "其他未归类误差操作（仅审计）",
    "calculation_operations": "其他未归类计算操作（仅审计）",
    "new_information_operation": "其他未归类新信息操作（仅审计）",
}
OBSERVABLE_FALLBACK_LABELS = set(OBSERVABLE_FALLBACK_LABEL_BY_FIELD.values())


# 与物理生产脚本相同：先把模型偶发的近义输出收敛到
# 可审计枚举，再作严格校验。只收录语义唯一的别名；无法
# 唯一判断的值降级为仅审计兜底值，不会默认填成任何有效难度信号。
OBSERVABLE_FIELD_ALIASES = {
    "new_ininformation_operation": "new_information_operation",
}

TASK_TYPE_ALIASES = {
    "误差分析": "实验操作与探究",
    "概念辨析": "直接事实与概念",
    "能量转化判断": "性质与反应判断",
    "成分推断": "证据推断",
    "微观粒子表征": "化学用语",
    "微观粒子与符号转换": "化学用语",
    "化学式推断": "化学用语",
    "化学方程式": "化学用语",
    "条件筛选": "性质与反应判断",
    "性质与应用推断": "性质与反应判断",
    "反应条件与速率分析": "性质与反应判断",
    "方案评价": "方案设计与评价",
}

# 模型偶尔会把合法的细粒度rule_families值写进粗粒度task_type。
# 这些值与粗任务类型存在唯一语义归属，可以在本地无损归一，避免为
# 字段串位消耗一次Schema重试。这里只修复task_type，不反向补写
# rule_families，避免凭空增加规则族并影响后处理阈值。
RULE_FAMILY_TO_TASK_TYPE = {
    "教材事实直接匹配": "直接事实与概念",
    "分类或概念辨析": "直接事实与概念",
    "化学用语书写": "化学用语",
    "化学用语含义辨析": "化学用语",
    "性质用途或现象判断": "性质与反应判断",
    "反应关系或条件判断": "性质与反应判断",
    "实验操作规范": "实验操作与探究",
    "作用目的或原因解释": "实验操作与探究",
    "异常失败或误差诊断": "实验操作与探究",
    "图表读取或数据归纳": "图表与数据",
    "证据推断或鉴别除杂": "证据推断",
    "定量关系与计算": "定量计算",
    "方案设计或评价": "方案设计与评价",
    "新信息迁移": "新信息应用",
}

RULE_FAMILY_ALIASES = {
    # 历史V2-V6粗规则族：只用于无损回放，新Prompt不再输出这些值。
    "直接事实与概念": "教材事实直接匹配",
    "化学用语": "化学用语书写",
    "性质与反应判断": "性质用途或现象判断",
    "实验操作与探究": "实验操作规范",
    "图表与数据": "图表读取或数据归纳",
    "证据推断": "证据推断或鉴别除杂",
    "定量计算": "定量关系与计算",
    "方案设计与评价": "方案设计或评价",
    "新信息应用": "新信息迁移",
    # 模型常见近义表达。
    "教材事实或名称直接匹配": "教材事实直接匹配",
    "分类标准应用": "分类或概念辨析",
    "完整命题正误辨析": "分类或概念辨析",
    "概念辨析": "分类或概念辨析",
    "微观粒子表征分析": "分类或概念辨析",
    "化学用语含义解释": "化学用语含义辨析",
    "性质用途或现象解释": "性质用途或现象判断",
    "反应条件判断": "反应关系或条件判断",
    "实验作用或目的解释": "作用目的或原因解释",
    "规范原因表达": "作用目的或原因解释",
    "异常或失败原因诊断": "异常失败或误差诊断",
    "误差分析": "异常失败或误差诊断",
    "图表读取或归纳": "图表读取或数据归纳",
    "证据推断或物质鉴别": "证据推断或鉴别除杂",
    "定量计算": "定量关系与计算",
    "跨学科语义理解": "跨学科语义或模型应用",
    "跨学科模型应用": "跨学科语义或模型应用",
}

# 合法操作枚举偶尔被模型写入rule_families。下列值都能同时确定
# 对应规则族和原本所属操作字段，因此可无损搬回；语义不唯一的值
# 不在这里猜测，统一降级为内部审计值并禁止自动写回。
RULE_FAMILY_CROSS_FIELD_MOVES = {
    "微观粒子→化学符号": (
        "化学用语书写",
        "representation_operations",
        "微观粒子→化学符号",
    ),
    "宏观对象→化学符号": (
        "化学用语书写",
        "representation_operations",
        "宏观对象→化学符号",
    ),
    "化学符号→定量关系": (
        "定量关系与计算",
        "representation_operations",
        "化学符号→定量关系",
    ),
    "微观粒子→宏观含义": (
        "化学用语含义辨析",
        "representation_operations",
        "微观粒子→宏观含义",
    ),
    "宏观现象→微观粒子": (
        "分类或概念辨析",
        "representation_operations",
        "宏观现象→微观粒子",
    ),
    "范围或边界判断": (
        "反应关系或条件判断",
        "condition_operations",
        "范围或边界",
    ),
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
        "微观粒子→宏观现象": "微观粒子→宏观含义",
        "宏观对象→微观粒子": "宏观现象→微观粒子",
        "宏观现象→化学关系": "宏观现象→化学符号",
    },
    "evidence_operations": {
        "双来源交叉验证": "多证据共同成立",
        "排除候选解释": "排除一个候选",
        "排除完全变质": "排除一个候选",
        "排除未变质": "排除一个候选",
        "排除干扰物质": "排除一个候选",
        "排除干扰候选解释": "排除多个候选解释",
        "排除三个候选": "排除多个候选解释",
    },
    "condition_operations": {
        "条件对比": "条件切换",
        "多条件比较": "条件切换",
        "排除干扰条件排除": "干扰条件排除",
        "反应条件判断": "条件直接读取",
    },
    "experiment_operation": {
        "方案设计与评价": "方案评价或补充实验",
        "方案评价": "方案评价或补充实验",
        "方案设计或补充实验": "方案评价或补充实验",
    },
    "experiment_task_structure": {
        "数据归纳": "控制变量或数据归纳",
        "方案评价": "方案设计或评价",
        "方案评价或补充实验": "方案设计或评价",
    },
    "graph_table_operation": {
        "流程图解析": "流程或关系图解析",
        "关系图解析": "流程或关系图解析",
        "装置流程解析": "流程或关系图解析",
        "流程、装置或关系图解析": "流程或关系图解析",
    },
    "solution_topology": {
        "范围或边界筛选": "条件分支或范围筛选",
    },
    "calculation_operations": {
        "单一比例": "直接比例",
        "式量与组成计算": "化学式组成计算",
        "相对分子质量与组成计算": "化学式组成计算",
        "质量守恒": "单一守恒",
        "未知组分消元或组成不变量": "组分消元或组成不变量",
        "多个反应定量关系": "多反应定量关系",
    },
    "reaction_structure": {
        "单一分解反应": "单一反应",
    },
    "error_analysis_operation": {
        "读数偏差到最终结果方向": "操作偏差到最终结果方向",
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
OBSERVABLE_KNOWN_ENUM_LABELS = set().union(
    *OBSERVABLE_ENUM_VALUES_BY_FIELD.values()
)


def _clean_enum_text(value: Any) -> str:
    return "".join(str(value or "").strip().split())


def _normalization_reason(field: str, old: Any, new: Any) -> str:
    if new == OBSERVABLE_FALLBACK_LABEL_BY_FIELD.get(field):
        return "未知枚举降级为仅审计兜底值"
    return "枚举近义归一"


def _canonical_task_type(value: Any) -> str:
    clean = _clean_enum_text(value)
    if clean in TASK_TYPES:
        return clean
    if clean in TASK_TYPE_ALIASES:
        return TASK_TYPE_ALIASES[clean]
    if clean in RULE_FAMILY_TO_TASK_TYPE:
        return RULE_FAMILY_TO_TASK_TYPE[clean]
    if "误差" in clean:
        return "实验操作与探究"
    if any(word in clean for word in ("微观", "化学式", "化学符号")):
        return "化学用语"
    if any(word in clean for word in ("方案设计", "方案评价")):
        return "方案设计与评价"
    if any(word in clean for word in ("反应条件", "反应速率")):
        return "性质与反应判断"
    return OBSERVABLE_FALLBACK_LABEL_BY_FIELD["task_type"]


def _canonical_rule_family(value: Any) -> str:
    clean = _clean_enum_text(value)
    if clean in RULE_FAMILIES:
        return clean
    aliased = RULE_FAMILY_ALIASES.get(clean)
    return aliased or OBSERVABLE_FALLBACK_LABEL_BY_FIELD["rule_families"]


def _canonical_enum_value(field: str, value: Any) -> str:
    clean = _clean_enum_text(value)
    allowed = OBSERVABLE_ENUM_VALUES_BY_FIELD[field]
    if clean in allowed:
        return clean
    aliased = ENUM_VALUE_ALIASES.get(field, {}).get(clean)
    if aliased:
        return aliased
    # 先保留“合法值写错字段”的情况，后续确定性串位修复会把它移回
    # 正确字段。只有所有已知枚举都不匹配时才降级为内部兜底值。
    if clean in OBSERVABLE_KNOWN_ENUM_LABELS:
        return clean
    # 先保留未知值，给后续跨字段修复机会；所有修复结束后再统一降级。
    return clean


def _canonical_curriculum_topic(value: Any) -> Any:
    """仅剥离与教材映射逐字匹配的课题名称后缀。"""
    if not isinstance(value, str):
        return value
    clean = _clean_enum_text(value)
    if clean in CURRICULUM_TOPICS:
        return clean
    for code, name in CURRICULUM_TOPIC_NAMES.items():
        if clean in {
            f"{code}{name}",
            f"{code}({name})",
            f"{code}（{name}）",
        }:
            return code
    return "U_OTHER"


def normalize_observable_features(
    features: Any,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """按物理生产逻辑作确定性归一并保留可审计兜底值。

    语义唯一的近义值转成正式中文契约。未知枚举不会被猜成
    某个有效难度信号，而是转成“仅审计”兜底值；后处理据此禁用写回。
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

    def append_unique(field: str, value: str) -> None:
        values = normalized.get(field)
        if not isinstance(values, list):
            return
        if value not in values:
            values.append(value)

    known_feature_fields = set().union(
        OBSERVABLE_V2_FEATURE_FIELDS,
        OBSERVABLE_V3_FEATURE_FIELDS,
        OBSERVABLE_V4_FEATURE_FIELDS,
        OBSERVABLE_V5_FEATURE_FIELDS,
        OBSERVABLE_V6_FEATURE_FIELDS,
    )
    for raw_key, value in features.items():
        clean_key = _clean_enum_text(raw_key)
        key = OBSERVABLE_FIELD_ALIASES.get(clean_key, clean_key)
        record("features.key", clean_key, key, "字段名别名归一")
        if key not in known_feature_fields:
            record(
                "features.extra_field",
                clean_key,
                None,
                "未知额外字段删除，仅审计",
                force=True,
            )
            continue
        if key in normalized and clean_key != key:
            continue
        normalized[key] = copy.deepcopy(value)

    # solution_topology偶发收到证据操作。证据类型不能唯一决定解题拓扑：
    # “多证据共同成立”既可能是一条线性链，也可能是交叉验证；排除候选
    # 也不必然意味着条件分支。因此只无损搬回证据字段，拓扑降级为内部
    # 审计值，绝不为了通过Schema猜一个会参与后处理的有效枚举。
    topology_evidence_moves = {
        "排除多个候选解释",
        "多证据共同成立",
    }
    raw_topology = normalized.get("solution_topology")
    if raw_topology in topology_evidence_moves:
        normalized["solution_topology"] = (
            OBSERVABLE_FALLBACK_LABEL_BY_FIELD["solution_topology"]
        )
        append_unique("evidence_operations", raw_topology)
        record(
            "solution_topology→evidence_operations",
            raw_topology,
            normalized["solution_topology"],
            "语义不唯一，降级为仅审计兜底值",
            force=True,
        )

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
            record(
                "task_groups.task_type",
                old_type,
                new_type,
                _normalization_reason("task_type", old_type, new_type),
            )
            count = group.get("count")
            if isinstance(count, str) and count.strip().isdigit():
                converted_count = int(count.strip())
                record(
                    "task_groups.count",
                    count,
                    converted_count,
                    "数字字符串转整数",
                )
                count = converted_count
            if isinstance(count, int) and not isinstance(count, bool) and count == 0:
                record(
                    "task_groups",
                    {"task_type": old_type, "count": count},
                    None,
                    "零任务组删除，仅审计",
                    force=True,
                )
                continue
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
            clean_value = _clean_enum_text(value)
            cross_field_move = RULE_FAMILY_CROSS_FIELD_MOVES.get(
                clean_value
            )
            if cross_field_move:
                canonical, target_field, target_value = cross_field_move
                append_unique(target_field, target_value)
                record(
                    f"rule_families→{target_field}",
                    clean_value,
                    target_value,
                    "规则族中的操作枚举移回所属字段",
                    force=True,
                )
            else:
                canonical = _canonical_rule_family(clean_value)
            if canonical not in new_values:
                new_values.append(canonical)
        reason = (
            "未知枚举降级为仅审计兜底值"
            if OBSERVABLE_FALLBACK_LABEL_BY_FIELD["rule_families"]
            in new_values
            else "规则族近义归一与去重"
        )
        record("rule_families", old_values, new_values, reason)
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
        reason = (
            "未知枚举降级为仅审计兜底值"
            if OBSERVABLE_FALLBACK_LABEL_BY_FIELD.get(field)
            in canonical_values
            else "枚举近义归一与去重"
        )
        record(field, values, canonical_values, reason)
        normalized[field] = canonical_values

    # “宏观现象→宏观含义”没有跨表征转换，今天的重试结果也将其删除。
    # 本地丢弃这个伪操作比臆造一种化学转换更安全。
    representation_values = normalized.get("representation_operations")
    if isinstance(representation_values, list):
        for no_conversion in ("宏观现象→宏观含义",):
            if no_conversion not in representation_values:
                continue
            representation_values.remove(no_conversion)
            record(
                "representation_operations",
                no_conversion,
                None,
                "同一表征内部释义，不计作表征转换",
                force=True,
            )

    # 表征转换和定量计算是两个不同侧面。模型偶尔会把合法的表征枚举
    # 写进 calculation_operations；这种串位可以无损修复，但不能凭空
    # 猜测缺失的计算方法。若移动后定量任务没有计算操作，后续一致性
    # 质量审计会标记证据不完整并禁止依赖它自动写回。
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
            if (
                misplaced == "化学符号→定量关系"
                and "化学式组成计算" not in calculation_values
            ):
                calculation_values.append("化学式组成计算")
            record(
                "calculation_operations",
                misplaced,
                (
                    "representation_operations+化学式组成计算"
                    if misplaced == "化学符号→定量关系"
                    else "representation_operations"
                ),
                "表征转换值移回表征字段，并保留可核验的组成计算事实",
                force=True,
            )

        if "定量关系与计算" in calculation_values:
            calculation_values.remove("定量关系与计算")
            append_unique("rule_families", "定量关系与计算")
            record(
                "calculation_operations→rule_families",
                "定量关系与计算",
                "定量关系与计算",
                "规则族从计算操作字段移回规则族字段",
                force=True,
            )

    conditions = normalized.get("condition_operations")
    evidence = normalized.get("evidence_operations")
    calculation_values = normalized.get("calculation_operations")
    if (
        isinstance(conditions, list)
        and isinstance(evidence, list)
        and isinstance(calculation_values, list)
    ):
        # 图表分段描述、计算方法和证据操作不能留在条件数组中。
        graph_condition_moves = {
            "拐点分段",
            "分段或拐点确定",
            "分段条件",
            "拐点边界",
        }
        moved_graph_conditions = set()
        for misplaced in list(conditions):
            if misplaced not in graph_condition_moves:
                continue
            conditions.remove(misplaced)
            if normalized.get("graph_table_operation") in {None, "无"}:
                normalized["graph_table_operation"] = "拐点平台或分段"
            moved_graph_conditions.add(misplaced)
            record(
                "condition_operations→graph_table_operation",
                misplaced,
                "拐点平台或分段",
                "图表分段操作字段串位修复",
                force=True,
            )

        # 分段条件和拐点边界除图表操作外，还分别表达条件切换与范围。
        # 上面的移动已从conditions删除原值；补写其确定性条件含义。
        if "分段条件" in moved_graph_conditions:
            append_unique("condition_operations", "条件切换")
        if "拐点边界" in moved_graph_conditions:
            append_unique("condition_operations", "范围或边界")

        for misplaced in ("差量", "单一守恒"):
            if misplaced not in conditions:
                continue
            conditions.remove(misplaced)
            if misplaced not in calculation_values:
                calculation_values.append(misplaced)
            record(
                "condition_operations→calculation_operations",
                misplaced,
                misplaced,
                "计算方法字段串位修复",
                force=True,
            )

        evidence_to_condition = {
            "范围条件筛选": "范围或边界",
            "范围或边界筛选": "范围或边界",
            "排除干扰条件": "干扰条件排除",
            "排除干扰": "干扰条件排除",
        }
        for misplaced, target in evidence_to_condition.items():
            if misplaced not in evidence:
                continue
            evidence.remove(misplaced)
            if target not in conditions:
                conditions.append(target)
            record(
                "evidence_operations→condition_operations",
                misplaced,
                target,
                "范围筛选操作字段串位修复",
                force=True,
            )

        if "组分消元或组成不变量" in evidence:
            evidence.remove("组分消元或组成不变量")
            if "组分消元或组成不变量" not in calculation_values:
                calculation_values.append("组分消元或组成不变量")
            normalized["solution_topology"] = "未知组分消元或组成不变量"
            record(
                "evidence_operations→calculation_operations",
                "组分消元或组成不变量",
                "组分消元或组成不变量",
                "组分消元方法字段串位修复",
                force=True,
            )

        # “未知组成或量反推”是拓扑而不是计算方法。移回拓扑后不猜测
        # 具体计算操作；若本题确有定量任务且没有方法，质量审计会标记
        # 证据不完整，保留首轮等级但禁止依赖计算特征自动写回。
        if "未知组成或量反推" in calculation_values:
            calculation_values.remove("未知组成或量反推")
            normalized["solution_topology"] = "未知组成或量反推"
            record(
                "calculation_operations→solution_topology",
                "未知组成或量反推",
                "未知组成或量反推",
                "反推拓扑字段串位修复",
                force=True,
            )

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

        evidence_moves = {
            "排除一个候选": "排除一个候选",
            "排除干扰物质": "排除一个候选",
            "排除干扰候选解释": "排除多个候选解释",
        }
        for misplaced, target in evidence_moves.items():
            if misplaced not in conditions:
                continue
            conditions.remove(misplaced)
            if target not in evidence:
                evidence.append(target)
            record(
                "condition_operations→evidence_operations",
                misplaced,
                target,
                "候选排除操作字段串位修复",
                force=True,
            )

        if any(value in conditions for value in ("控制变量", "变量控制")):
            conditions[:] = [
                value
                for value in conditions
                if value not in {"控制变量", "变量控制"}
            ]
            if normalized.get("experiment_operation") in {None, "无"}:
                normalized["experiment_operation"] = "变量控制"
            record(
                "condition_operations→experiment_operation",
                "控制变量",
                "变量控制",
                "控制变量操作字段串位修复",
                force=True,
            )

        error_moves = {
            "操作偏差": "操作偏差到最终结果方向",
            "操作偏差到最终结果方向": "操作偏差到最终结果方向",
            "读数偏差到实际量判断": "读数偏差到实际量判断",
        }
        for misplaced, target in error_moves.items():
            if misplaced not in conditions:
                continue
            conditions.remove(misplaced)
            if normalized.get("error_analysis_operation") in {
                None,
                "无误差分析",
            }:
                normalized["error_analysis_operation"] = target
            record(
                "condition_operations→error_analysis_operation",
                misplaced,
                target,
                "误差操作字段串位修复",
                force=True,
            )

        for misplaced in ("分段", "拐点平台或分段"):
            if misplaced not in conditions:
                continue
            conditions.remove(misplaced)
            if normalized.get("graph_table_operation") in {None, "无"}:
                normalized["graph_table_operation"] = "拐点平台或分段"
            record(
                "condition_operations→graph_table_operation",
                misplaced,
                "拐点平台或分段",
                "图表分段操作字段串位修复",
                force=True,
            )

    # 出现图表数据转换只能确定graph不能为“无”，不能确定究竟是直接
    # 读数、多组比较、趋势、分段还是多图联合。使用内部兜底值，避免
    # 把语义冲突伪装成低档“直接读数”。
    if (
        isinstance(representation_values, list)
        and "图表数据→化学关系" in representation_values
        and normalized.get("graph_table_operation") in {None, "无"}
    ):
        normalized["graph_table_operation"] = (
            OBSERVABLE_FALLBACK_LABEL_BY_FIELD["graph_table_operation"]
        )
        record(
            "graph_table_operation",
            "无",
            normalized["graph_table_operation"],
            "语义不唯一，降级为仅审计兜底值",
            force=True,
        )

    raw_experiment_operation = normalized.get("experiment_operation")
    if raw_experiment_operation in EXPERIMENT_TASK_STRUCTURES - {
        "无实验判断",
        "其他未归类实验结构（仅审计）",
    }:
        normalized["experiment_task_structure"] = raw_experiment_operation
        normalized["experiment_operation"] = (
            OBSERVABLE_FALLBACK_LABEL_BY_FIELD["experiment_operation"]
        )
        record(
            "experiment_operation→experiment_task_structure",
            raw_experiment_operation,
            normalized["experiment_operation"],
            "语义不唯一，降级为仅审计兜底值",
            force=True,
        )
        raw_experiment_operation = normalized["experiment_operation"]
    if raw_experiment_operation in {
        "操作偏差因果链",
        "操作偏差到最终结果方向",
        "读数偏差到实际量判断",
    }:
        if raw_experiment_operation == "操作偏差因果链":
            normalized["experiment_task_structure"] = "操作偏差因果链"
        else:
            normalized["error_analysis_operation"] = raw_experiment_operation
            normalized["experiment_task_structure"] = "操作偏差因果链"
        normalized["experiment_operation"] = (
            OBSERVABLE_FALLBACK_LABEL_BY_FIELD["experiment_operation"]
        )
        record(
            "experiment_operation",
            raw_experiment_operation,
            normalized["experiment_operation"],
            "语义不唯一，降级为仅审计兜底值",
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
        record(
            field,
            old_value,
            new_value,
            _normalization_reason(field, old_value, new_value),
        )
        normalized[field] = new_value

    topics = normalized.get("curriculum_topics")
    if isinstance(topics, list):
        canonical_topics = [
            _canonical_curriculum_topic(value) for value in topics
        ]
        record(
            "curriculum_topics",
            topics,
            canonical_topics,
            (
                "未知枚举降级为仅审计兜底值"
                if "U_OTHER" in canonical_topics
                else "剥离与教材映射匹配的课题名称后缀"
            ),
        )
        normalized["curriculum_topics"] = canonical_topics

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
        inferred = OBSERVABLE_FALLBACK_LABEL_BY_FIELD["experiment_operation"]
        record(
            "experiment_operation",
            experiment_op,
            inferred,
            "语义不唯一，降级为仅审计兜底值",
        )
        normalized["experiment_operation"] = inferred
        experiment_op = inferred
    if experiment_op not in {None, "无"} and experiment_structure == "无实验判断":
        inferred = OBSERVABLE_FALLBACK_LABEL_BY_FIELD[
            "experiment_task_structure"
        ]
        record(
            "experiment_task_structure",
            experiment_structure,
            inferred,
            "语义不唯一，降级为仅审计兜底值",
        )
        normalized["experiment_task_structure"] = inferred

    # 跨字段搬运和一致性修复结束后，剩余非法枚举才进入内部兜底值。这样
    # “控制变量”写进条件字段等可修复串位不会过早丢失原语义。
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
        fallback = OBSERVABLE_FALLBACK_LABEL_BY_FIELD.get(field)
        if not fallback:
            continue
        repaired: List[str] = []
        for value in values:
            canonical = _canonical_enum_value(field, value)
            if canonical not in OBSERVABLE_ENUM_VALUES_BY_FIELD[field]:
                record(
                    field,
                    value,
                    fallback,
                    "未知枚举降级为仅审计兜底值",
                    force=True,
                )
                canonical = fallback
            if canonical not in repaired:
                repaired.append(canonical)
        normalized[field] = repaired

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
        value = normalized.get(field)
        if field not in normalized:
            continue
        if value not in OBSERVABLE_ENUM_VALUES_BY_FIELD[field]:
            fallback = OBSERVABLE_FALLBACK_LABEL_BY_FIELD[field]
            record(
                field,
                value,
                fallback,
                "未知枚举降级为仅审计兜底值",
                force=True,
            )
            normalized[field] = fallback

    return normalized, actions


def observable_feature_quality_flags(
    features: Dict[str, Any],
    normalization_actions: Iterable[Dict[str, Any]] = (),
) -> List[str]:
    """返回不阻断评级、但会阻止自动写回的特征质量标记。"""
    flags: List[str] = []
    for group in features.get("task_groups", []):
        if (
            isinstance(group, dict)
            and group.get("task_type")
            == OBSERVABLE_FALLBACK_LABEL_BY_FIELD["task_type"]
        ):
            flags.append("fallback_enum:task_groups.task_type")
    for field, fallback in OBSERVABLE_FALLBACK_LABEL_BY_FIELD.items():
        if field == "task_type":
            continue
        value = features.get(field)
        if value == fallback or (
            isinstance(value, list) and fallback in value
        ):
            flags.append(f"fallback_enum:{field}")
    if "U_OTHER" in features.get("curriculum_topics", []):
        flags.append("fallback_enum:curriculum_topics")
    has_quantitative_task = any(
        isinstance(group, dict)
        and group.get("task_type") == "定量计算"
        for group in features.get("task_groups", [])
    )
    valid_calculations = [
        value
        for value in features.get("calculation_operations", [])
        if value not in OBSERVABLE_FALLBACK_LABELS
    ]
    if has_quantitative_task and not valid_calculations:
        flags.append("incomplete_calculation_evidence")
    if any(
        action.get("reason")
        in {
            "未知枚举降级为仅审计兜底值",
            "语义不唯一，降级为仅审计兜底值",
        }
        for action in normalization_actions
    ):
        flags.append("ambiguous_enum_normalized_to_fallback")
    if any(
        action.get("reason")
        in {
            "未知额外字段删除，仅审计",
            "零任务组删除，仅审计",
        }
        for action in normalization_actions
    ):
        flags.append("structural_schema_repaired")
    return list(dict.fromkeys(flags))


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
    """严格校验正式 V5，并兼容读取历史 V6/V4/V3/V2。

    校验器本身不猜测语义默认值。调用方应先执行确定性归一：未知中文
    枚举会成为内部审计值；缺字段、多字段或结构错误仍拒绝，以便只修复
    features，并冻结首轮等级和理由。
    """
    if not isinstance(features, dict):
        raise ValueError("features必须是JSON对象")
    actual = set(features)
    v6_expected = set(OBSERVABLE_V6_FEATURE_FIELDS)
    v5_expected = set(OBSERVABLE_FEATURE_FIELDS)
    v4_expected = set(OBSERVABLE_V4_FEATURE_FIELDS)
    v3_expected = set(OBSERVABLE_V3_FEATURE_FIELDS)
    v2_expected = set(OBSERVABLE_V2_FEATURE_FIELDS)
    is_v6 = actual == v6_expected
    is_v5 = actual == v5_expected
    is_v4 = actual == v4_expected
    is_v3 = actual == v3_expected
    is_v2 = actual == v2_expected
    if not (is_v6 or is_v5 or is_v4 or is_v3 or is_v2):
        missing = sorted(v5_expected - actual)
        extra = sorted(actual - v5_expected)
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
    # 定量任务存在但计算操作缺失属于“证据不完整”，不再让模型重生成
    # 整份评级。observable_feature_quality_flags会记录该问题，并阻止依赖
    # 计算字段的自动写回。
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
        curriculum_topics = [
            topic
            for topic in validated["curriculum_topics"]
            if topic != "U_OTHER"
        ]
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

    effective_groups = [
        group
        for group in validated["task_groups"]
        if group["task_type"]
        != OBSERVABLE_FALLBACK_LABEL_BY_FIELD["task_type"]
    ]
    effective_rules = [
        value
        for value in validated["rule_families"]
        if value != OBSERVABLE_FALLBACK_LABEL_BY_FIELD["rule_families"]
    ]
    valid_response_operations = [
        value
        for value in validated.get("response_operations", [])
        if value not in OBSERVABLE_FALLBACK_LABELS
    ]
    valid_cross_subject_operations = [
        value
        for value in validated.get("cross_subject_operations", [])
        if value not in OBSERVABLE_FALLBACK_LABELS
    ]
    valid_condition_operations = [
        value
        for value in validated["condition_operations"]
        if value not in OBSERVABLE_FALLBACK_LABELS
    ]
    valid_representation_operations = [
        value
        for value in validated["representation_operations"]
        if value not in OBSERVABLE_FALLBACK_LABELS
    ]
    valid_evidence_operations = [
        value
        for value in validated["evidence_operations"]
        if value not in OBSERVABLE_FALLBACK_LABELS
    ]
    valid_calculation_operations = [
        value
        for value in validated["calculation_operations"]
        if value not in OBSERVABLE_FALLBACK_LABELS
    ]
    advanced_calculation_operations = [
        value
        for value in valid_calculation_operations
        if value
        in {
            "组分消元或组成不变量",
            "差量",
            "多反应定量关系",
            "联立",
            "范围或分类计算",
        }
    ]

    return {
        "longest_chain_steps": len(
            validated["longest_solution_chain"]
        ),
        "effective_task_count": sum(
            group["count"] for group in effective_groups
        ),
        "task_group_count": len(effective_groups),
        "rule_family_count": len(effective_rules),
        "response_operation_count": len(valid_response_operations),
        "cross_subject_operation_count": len(
            valid_cross_subject_operations
        ),
        "curriculum_topic_count": topic_count,
        "curriculum_unit_count": len(curriculum_units),
        "curriculum_span_type": curriculum_span_type,
        "curriculum_coupling_type": curriculum_coupling_type,
        "curriculum_span_summary": curriculum_span_summary,
        "condition_operation_count": len(valid_condition_operations),
        "representation_operation_count": len(
            valid_representation_operations
        ),
        "evidence_operation_count": len(valid_evidence_operations),
        "calculation_operation_count": len(valid_calculation_operations),
        "advanced_calculation_operations": advanced_calculation_operations,
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
