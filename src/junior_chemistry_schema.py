"""初中化学教师口径特征契约与审计型后处理。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

FEATURE_SCHEMA_VERSION = "junior_chemistry_teacher_factors_v3"
CURRICULUM_PATH = Path(__file__).resolve().parent.parent / "JUNIOR_CHEMISTRY_CURRICULUM.md"

LEVELS = {"送分题", "基础题", "中等题", "拔高题", "压轴题"}
STEPS = {"1步", "2-3步", "4步及以上"}
CALCULATION_STEPS = {"无", *STEPS}
RELATIONS = {"单项任务", "多项独立", "前后依赖"}
SCOPES = {"within_junior", "out_of_scope"}

ENUMS = {
    "task_types": {
        "直接识记", "概念辨析", "信息提取", "性质用途判断", "化学用语书写",
        "方程式书写配平", "实验操作判断", "装置分析", "现象判断与解释",
        "物质鉴别或推断", "除杂或分离", "计算", "方案评价或改进",
    },
    "information_processing": {
        "文字材料提取", "实验装置图分析", "流程图分析", "表格数据比较",
        "曲线读取", "图像分段分析", "微观示意图分析",
    },
    "reaction_processes": {
        "单一反应判断", "连续反应转化", "反应条件选择", "反应先后判断",
        "过量或不足判断", "物质鉴别或推断", "除杂或分离",
    },
    "experiment_tasks": {
        "基本操作判断", "装置选择", "装置作用分析", "装置连接顺序", "现象判断",
        "原因解释", "控制变量", "根据现象得出结论", "根据结论设计操作",
        "实验方案评价", "实验改进", "误差分析",
    },
    "calculation_types": {
        "化学方程式计算", "相对分子质量与元素质量计算", "溶质质量分数",
        "反应后溶液质量", "含杂质计算", "图像分段计算",
    },
    "special_methods": {
        "质量守恒", "元素守恒", "差量法", "极值法", "分情况计算", "多方程式联立",
    },
    "hidden_conditions": {
        "前后对象要求", "反应条件", "过量或不足", "反应先后", "纯净或干燥要求",
        "杂质或气体损失", "图像拐点或分段",
    },
    "interference_points": {
        "易混概念", "选项多规则切换", "规范表述易错", "特例或边界", "干扰数据",
    },
    "expression_requirements": {
        "化学符号自主书写", "化学方程式自主书写", "实验现象规范描述",
        "原因规范表达", "结论规范表达",
    },
    "unfamiliar_materials": {
        "课内拓展物质且给足信息", "陌生物质信息推断", "陌生装置或材料信息提取",
    },
    "interdisciplinary_context": {"物理-化学", "生物-化学", "社会生活背景"},
}


class ChemistrySchemaError(ValueError):
    """模型输出不满足初中化学严格契约。"""


def load_curriculum_topics() -> dict[str, dict[str, str]]:
    topics: dict[str, dict[str, str]] = {}
    for line in CURRICULUM_PATH.read_text(encoding="utf-8").splitlines():
        if not re.match(r"^\| U\d{2} \|", line):
            continue
        unit_id, unit_name, topic_id, topic_name, aliases = [
            item.strip() for item in line.split("|")[1:-1]
        ]
        if topic_id in topics:
            raise RuntimeError(f"重复topic_id: {topic_id}")
        topics[topic_id] = {
            "unit_id": unit_id,
            "unit_name": unit_name,
            "topic_name": topic_name,
            "aliases": aliases,
        }
    if not topics:
        raise RuntimeError(f"知识点目录为空: {CURRICULUM_PATH}")
    return topics


def curriculum_catalog_text() -> str:
    return "\n".join(
        f"- {topic_id}: {topic['unit_id']} {topic['topic_name']}（常见说法：{topic['aliases']}）"
        for topic_id, topic in load_curriculum_topics().items()
    )


def _exact_dict(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ChemistrySchemaError(
            f"{name}字段不匹配: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _string_list(value: Any, name: str, *, required: bool = False) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ChemistrySchemaError(f"{name}必须是字符串数组")
    result = list(dict.fromkeys(item.strip() for item in value if item.strip()))
    if required and not result:
        raise ChemistrySchemaError(f"{name}不得为空")
    return result


def _enum_list(value: Any, name: str, enum_name: str) -> list[str]:
    result = _string_list(value, name, required=True)
    if result == ["无"]:
        return result
    if "无" in result:
        raise ChemistrySchemaError(f"{name}的“无”不能与其他枚举并列")
    invalid = [item for item in result if item not in ENUMS[enum_name]]
    if invalid:
        raise ChemistrySchemaError(f"{name}存在非法枚举: {invalid}")
    return result


def _has_any(items: list[str]) -> bool:
    return items != ["无"]


def validate_rating_contract(value: Any) -> dict[str, Any]:
    value = _exact_dict(value, {"features", "reasoning", "difficulty_level"}, "顶层")
    if value["difficulty_level"] not in LEVELS:
        raise ChemistrySchemaError("difficulty_level非法")

    features = _exact_dict(
        value["features"],
        {
            "knowledge", "solution_process", "information_processing",
            "reaction_processes", "experiment_tasks", "calculation",
            "difficulty_conditions", "expression_requirements",
            "question_context", "curriculum_scope",
        },
        "features",
    )

    knowledge = _exact_dict(features["knowledge"], {"topic_ids"}, "knowledge")
    topic_ids = _string_list(knowledge["topic_ids"], "knowledge.topic_ids")
    topics = load_curriculum_topics()
    unknown = [topic_id for topic_id in topic_ids if topic_id not in topics]
    if unknown:
        raise ChemistrySchemaError(f"未收录topic_id: {unknown}")

    solution = _exact_dict(
        features["solution_process"],
        {"step_count", "task_types", "key_steps", "task_relation"},
        "solution_process",
    )
    if solution["step_count"] not in STEPS:
        raise ChemistrySchemaError("solution_process.step_count枚举非法")
    if solution["task_relation"] not in RELATIONS:
        raise ChemistrySchemaError("solution_process.task_relation枚举非法")
    task_types = _enum_list(solution["task_types"], "solution_process.task_types", "task_types")
    key_steps = _string_list(solution["key_steps"], "solution_process.key_steps", required=True)
    if task_types == ["无"]:
        raise ChemistrySchemaError("solution_process.task_types不能为['无']")
    if solution["task_relation"] == "单项任务" and len(task_types) != 1:
        raise ChemistrySchemaError("单项任务只能有一个task_type")

    information = _enum_list(
        features["information_processing"], "information_processing", "information_processing"
    )
    reactions = _exact_dict(
        features["reaction_processes"],
        {"processes", "requires_condition_selection"},
        "reaction_processes",
    )
    reaction_items = _enum_list(
        reactions["processes"], "reaction_processes.processes", "reaction_processes"
    )
    if not isinstance(reactions["requires_condition_selection"], bool):
        raise ChemistrySchemaError("requires_condition_selection必须为布尔值")
    if reactions["requires_condition_selection"] and "反应条件选择" not in reaction_items:
        raise ChemistrySchemaError("需要选择反应条件时processes必须包含“反应条件选择”")
    if not reactions["requires_condition_selection"] and "反应条件选择" in reaction_items:
        raise ChemistrySchemaError("processes包含“反应条件选择”时布尔标记必须为true")

    experiment_tasks = _enum_list(
        features["experiment_tasks"], "experiment_tasks", "experiment_tasks"
    )
    calculation = _exact_dict(
        features["calculation"],
        {"has_calculation", "calculation_steps", "types", "special_methods"},
        "calculation",
    )
    if not isinstance(calculation["has_calculation"], bool):
        raise ChemistrySchemaError("calculation.has_calculation必须为布尔值")
    if calculation["calculation_steps"] not in CALCULATION_STEPS:
        raise ChemistrySchemaError("calculation.calculation_steps枚举非法")
    calculation_types = _enum_list(calculation["types"], "calculation.types", "calculation_types")
    special_methods = _enum_list(
        calculation["special_methods"], "calculation.special_methods", "special_methods"
    )
    if calculation["has_calculation"]:
        if calculation["calculation_steps"] == "无" or calculation_types == ["无"]:
            raise ChemistrySchemaError("有计算时必须填写计算步数和计算类型")
    elif calculation["calculation_steps"] != "无" or calculation_types != ["无"] or special_methods != ["无"]:
        raise ChemistrySchemaError("无计算时calculation_steps必须为“无”，types和special_methods必须为['无']")
    if calculation["has_calculation"] != ("计算" in task_types):
        raise ChemistrySchemaError("has_calculation必须与task_types中的“计算”一致")

    conditions = _exact_dict(
        features["difficulty_conditions"],
        {"hidden_conditions", "interference_points"},
        "difficulty_conditions",
    )
    hidden_conditions = _enum_list(
        conditions["hidden_conditions"], "difficulty_conditions.hidden_conditions", "hidden_conditions"
    )
    interference_points = _enum_list(
        conditions["interference_points"], "difficulty_conditions.interference_points", "interference_points"
    )
    expression = _enum_list(
        features["expression_requirements"], "expression_requirements", "expression_requirements"
    )
    context = _exact_dict(
        features["question_context"],
        {"unfamiliar_materials", "interdisciplinary_context"},
        "question_context",
    )
    unfamiliar = _enum_list(
        context["unfamiliar_materials"], "question_context.unfamiliar_materials", "unfamiliar_materials"
    )
    interdisciplinary = _enum_list(
        context["interdisciplinary_context"],
        "question_context.interdisciplinary_context",
        "interdisciplinary_context",
    )

    scope = _exact_dict(features["curriculum_scope"], {"scope", "extra_points"}, "curriculum_scope")
    if scope["scope"] not in SCOPES:
        raise ChemistrySchemaError("curriculum_scope.scope非法")
    extra_points = _string_list(scope["extra_points"], "curriculum_scope.extra_points")
    if scope["scope"] == "within_junior" and (extra_points or not topic_ids):
        raise ChemistrySchemaError("within_junior必须有topic_ids且extra_points必须为空数组")
    if scope["scope"] == "out_of_scope" and not extra_points:
        raise ChemistrySchemaError("out_of_scope必须列出具体超纲内容")

    reasoning = _exact_dict(
        value["reasoning"],
        {"knowledge_points", "solution_process", "main_difficulty_factors", "level_basis"},
        "reasoning",
    )
    normalized_reasoning = {key: str(text).strip() for key, text in reasoning.items()}
    if any(not text for text in normalized_reasoning.values()):
        raise ChemistrySchemaError("reasoning字段不得为空")

    return {
        "features": {
            "knowledge": {"topic_ids": topic_ids},
            "solution_process": {
                "step_count": solution["step_count"],
                "task_types": task_types,
                "key_steps": key_steps,
                "task_relation": solution["task_relation"],
            },
            "information_processing": information,
            "reaction_processes": {
                "processes": reaction_items,
                "requires_condition_selection": reactions["requires_condition_selection"],
            },
            "experiment_tasks": experiment_tasks,
            "calculation": {
                "has_calculation": calculation["has_calculation"],
                "calculation_steps": calculation["calculation_steps"],
                "types": calculation_types,
                "special_methods": special_methods,
            },
            "difficulty_conditions": {
                "hidden_conditions": hidden_conditions,
                "interference_points": interference_points,
            },
            "expression_requirements": expression,
            "question_context": {
                "unfamiliar_materials": unfamiliar,
                "interdisciplinary_context": interdisciplinary,
            },
            "curriculum_scope": {"scope": scope["scope"], "extra_points": extra_points},
        },
        "reasoning": normalized_reasoning,
        "difficulty_level": value["difficulty_level"],
    }


def _candidate(rule: str, level: str, reason: str, evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule": rule,
        "candidate_level": level,
        "reason": reason,
        "evidence": evidence,
        "writeback_applied": False,
    }


def _build_audit_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    features = result["features"]
    knowledge = features["knowledge"]
    solution = features["solution_process"]
    calculation = features["calculation"]
    level = result["difficulty_level"]
    candidates: list[dict[str, Any]] = []

    if level == "送分题" and knowledge["unit_count"] >= 2:
        candidates.append(_candidate(
            "K1_easy_cross_unit", "基础题", "送分题涉及跨单元知识，需复核是否低估知识切换负担。",
            {"unit_count": knowledge["unit_count"], "topic_ids": knowledge["topic_ids"]},
        ))
    if level == "送分题" and knowledge["knowledge_point_count"] >= 2:
        candidates.append(_candidate(
            "K2_easy_multiple_topics", "基础题", "送分题涉及多个去重知识点，需复核是否仍属于直接识记。",
            {"knowledge_point_count": knowledge["knowledge_point_count"], "topic_ids": knowledge["topic_ids"]},
        ))
    if level in {"送分题", "基础题"} and knowledge["unit_count"] >= 3 and knowledge["knowledge_point_count"] >= 3:
        candidates.append(_candidate(
            "K3_multi_unit_composite", "中等题", "至少三个单元和三个知识点构成明显知识切换，需复核2/3档边界。",
            {"unit_count": knowledge["unit_count"], "knowledge_point_count": knowledge["knowledge_point_count"]},
        ))

    nontrivial_experiment_tasks = {
        "装置作用分析", "装置连接顺序", "原因解释", "控制变量",
        "根据现象得出结论", "根据结论设计操作", "实验方案评价", "实验改进", "误差分析",
    }
    nontrivial = {
        "experiment": bool(
            nontrivial_experiment_tasks.intersection(features["experiment_tasks"])
        ),
        "hidden_conditions": features["difficulty_conditions"]["hidden_conditions"] != ["无"],
        "interference": features["difficulty_conditions"]["interference_points"] != ["无"],
        "expression": features["expression_requirements"] != ["无"],
    }
    if level == "送分题" and any(nontrivial.values()):
        candidates.append(_candidate(
            "B1_easy_with_nontrivial_factor", "基础题", "送分题包含实验、隐藏条件、干扰或规范表达要求。",
            {key: value for key, value in nontrivial.items() if value},
        ))
    if level == "基础题" and solution["task_relation"] != "单项任务" and len(solution["key_steps"]) >= 3:
        candidates.append(_candidate(
            "B2_basic_multiple_tasks", "中等题", "基础题包含至少三个实质任务，需复核整体任务负担。",
            {"task_relation": solution["task_relation"], "key_steps": solution["key_steps"]},
        ))

    hard_experiment = {"根据结论设计操作", "实验方案评价", "实验改进", "误差分析"}
    hard_calc = calculation["calculation_steps"] == "4步及以上" and calculation["special_methods"] != ["无"]
    if level == "中等题" and (
        bool(hard_experiment.intersection(features["experiment_tasks"])) or hard_calc
    ):
        candidates.append(_candidate(
            "H1_medium_decisive_task", "拔高题", "存在高阶实验任务或带特殊方法的长计算，需复核局部高难下限。",
            {"experiment_tasks": features["experiment_tasks"], "calculation": calculation},
        ))
    if level == "压轴题" and solution["task_relation"] != "前后依赖":
        candidates.append(_candidate(
            "F1_final_without_dependency", "拔高题", "压轴题未呈现前后依赖任务，需复核是否仅为高难特征堆叠。",
            {"task_relation": solution["task_relation"], "step_count": solution["step_count"]},
        ))
    return candidates


def postprocess_chemistry_difficulty(value: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """严格校验、补充可审计统计；不读取题库难度，不修改模型档位。"""
    del data
    result = validate_rating_contract(value)
    topics = load_curriculum_topics()
    topic_ids = result["features"]["knowledge"]["topic_ids"]
    points = [
        {
            "topic_id": topic_id,
            "unit_id": topics[topic_id]["unit_id"],
            "unit_name": topics[topic_id]["unit_name"],
            "topic_name": topics[topic_id]["topic_name"],
        }
        for topic_id in topic_ids
    ]
    unit_ids = list(dict.fromkeys(point["unit_id"] for point in points))
    result["features"]["knowledge"] = {
        "topic_ids": topic_ids,
        "knowledge_points": points,
        "knowledge_point_count": len(topic_ids),
        "unit_count": len(unit_ids),
        "cross_unit": len(unit_ids) >= 2,
    }
    solution = result["features"]["solution_process"]
    solution["substantive_task_count"] = len(solution["key_steps"])

    original_level = result["difficulty_level"]
    result["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    result["postprocess"] = {
        "original_level": original_level,
        "final_level": original_level,
        "writeback_applied": False,
        "candidates": _build_audit_candidates(result),
    }
    result["postprocess_actions"] = []
    return result
