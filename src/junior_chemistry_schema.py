"""初中化学教师口径特征契约与审计型后处理。"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

FEATURE_SCHEMA_VERSION = "junior_chemistry_teacher_factors_v4"
CURRICULUM_PATH = Path(__file__).resolve().parent.parent / "JUNIOR_CHEMISTRY_CURRICULUM.md"

LEVELS = {"送分题", "基础题", "中等题", "拔高题", "压轴题"}
STEPS = {"1步", "2-3步", "4步及以上"}
CALCULATION_STEPS = {"无", *STEPS}
RELATIONS = {"单项任务", "多项独立", "前后依赖"}
SCOPES = {"within_junior", "out_of_scope"}

ENUMS = {
    "task_types": {
        "直接识记", "概念辨析与应用", "性质与用途判断", "信息读取与整理",
        "反应与转化分析", "实验操作与装置分析", "实验探究与方案评价",
        "物质鉴别推断与除杂", "化学用语与方程式书写", "定量计算",
        "解释与规范表达",
    },
    "information_processing": {
        "文字材料提取", "实验装置图分析", "流程图分析", "表格数据比较",
        "曲线读取", "图像分段分析", "微观示意图分析",
        "原子结构示意图分析", "元素周期表信息读取",
    },
    "reaction_processes": {
        "单一反应判断", "连续反应转化", "反应条件选择", "反应先后判断",
        "过量或不足判断", "物质鉴别或推断", "除杂或分离",
    },
    "experiment_tasks": {
        "基本操作判断", "装置选择", "装置作用分析", "装置连接顺序", "现象判断",
        "原因解释", "控制变量", "根据现象得出结论", "根据结论设计操作",
        "根据数据得出结论", "根据结论推断现象", "实验方案设计",
        "实验方案评价", "实验改进", "误差分析", "试剂选择", "物质检验与鉴别",
    },
    "calculation_types": {
        "化学方程式计算", "相对分子质量与元素质量计算", "溶质质量分数",
        "反应后溶液质量", "反应后沉淀或气体质量", "含杂质计算",
        "图像分段计算", "化合价计算", "溶解度相关计算", "其他课内定量计算",
    },
    "special_methods": {
        "质量守恒", "元素守恒", "差量法", "极值法", "分情况计算", "多方程式联立",
    },
    "hidden_conditions": {
        "前后对象要求", "反应条件", "过量或不足", "反应先后", "纯净或干燥要求",
        "杂质或气体损失", "图像拐点或分段", "溶液状态或溶剂变化",
    },
    "interference_points": {
        "易混概念", "选项多规则切换", "规范表述易错", "特例或边界",
        "干扰数据", "体系质量变化易错",
    },
    "expression_requirements": {
        "化学符号自主书写", "化学方程式自主书写", "实验现象规范描述",
        "原因规范表达", "结论规范表达", "仪器或操作名称自主书写",
        "试剂名称自主书写",
    },
    "unfamiliar_materials": {
        "课内拓展物质且给足信息", "陌生物质信息推断", "陌生装置或材料信息提取",
    },
    "interdisciplinary_context": {"物理-化学", "生物-化学", "社会生活背景"},
}

# 只收录语义确定、不会改变题目难度判断的同义表达。
# None 表示该值误放在当前字段，规范化时删除并留下审计记录。
ENUM_ALIASES: dict[str, dict[str, str | None]] = {
    "task_types": {
        "概念辨析": "概念辨析与应用",
        "信息提取": "信息读取与整理",
        "性质用途判断": "性质与用途判断",
        "化学用语书写": "化学用语与方程式书写",
        "方程式书写配平": "化学用语与方程式书写",
        "实验操作判断": "实验操作与装置分析",
        "装置分析": "实验操作与装置分析",
        "仪器识别": "实验操作与装置分析",
        "装置选择": "实验操作与装置分析",
        "装置作用分析": "实验操作与装置分析",
        "现象判断与解释": "解释与规范表达",
        "原因解释": "解释与规范表达",
        "原因规范表达": "解释与规范表达",
        "结论规范表达": "解释与规范表达",
        "根据现象得出结论": "实验探究与方案评价",
        "根据结论设计操作": "实验探究与方案评价",
        "实验方案设计": "实验探究与方案评价",
        "实验操作设计": "实验探究与方案评价",
        "实验方案评价": "实验探究与方案评价",
        "方案评价": "实验探究与方案评价",
        "方案评价或改进": "实验探究与方案评价",
        "控制变量": "实验探究与方案评价",
        "误差分析": "实验探究与方案评价",
        "实验误差分析": "实验探究与方案评价",
        "物质推断": "物质鉴别推断与除杂",
        "物质成分推断": "物质鉴别推断与除杂",
        "物质鉴别": "物质鉴别推断与除杂",
        "物质检验": "物质鉴别推断与除杂",
        "物质鉴别或推断": "物质鉴别推断与除杂",
        "除杂或分离": "物质鉴别推断与除杂",
        "试剂选择": "物质鉴别推断与除杂",
        "计算": "定量计算",
        "化合价计算": "定量计算",
        "相对分子质量与元素质量计算": "定量计算",
        "溶解度应用": "定量计算",
        "反应过程判断": "反应与转化分析",
        "反应过程分析": "反应与转化分析",
        "反应类型判断": "反应与转化分析",
        "反应先后判断": "反应与转化分析",
        "催化剂作用判断": "概念辨析与应用",
        "曲线读取": "信息读取与整理",
        "图像分段分析": "信息读取与整理",
    },
    "information_processing": {
        "元素周期表初步": "元素周期表信息读取",
    },
    "experiment_tasks": {
        "根据结论反推操作": "根据结论设计操作",
        "根据数据得出结论": "根据数据得出结论",
        "根据结论推断现象": "根据结论推断现象",
        "物质鉴别或推断": "物质检验与鉴别",
        "试剂选择": "试剂选择",
        "根据结论规范表达": None,
    },
    "calculation_types": {
        "反应后沉淀质量计算": "反应后沉淀或气体质量",
    },
    "hidden_conditions": {
        "饱和溶液要求": "溶液状态或溶剂变化",
        "饱和溶液不能再溶解原溶质": "溶液状态或溶剂变化",
        "生石灰消耗溶剂水": "溶液状态或溶剂变化",
        "元素守恒": None,
    },
    "interference_points": {
        "反应前后固体质量变化对溶液的影响": "体系质量变化易错",
    },
    "expression_requirements": {
        "仪器名称自主书写": "仪器或操作名称自主书写",
        "仪器名称书写": "仪器或操作名称自主书写",
        "操作名称规范填写": "仪器或操作名称自主书写",
        "试剂名称规范书写": "试剂名称自主书写",
        "化学用语自主书写": "化学符号自主书写",
        "化学式自主书写": "化学符号自主书写",
    },
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


def _record_action(
    actions: list[dict[str, Any]],
    *,
    path: str,
    original: Any,
    normalized: Any,
    reason: str,
) -> None:
    if original == normalized:
        return
    actions.append({
        "path": path,
        "original": copy.deepcopy(original),
        "normalized": copy.deepcopy(normalized),
        "reason": reason,
        "difficulty_level_changed": False,
    })


def _canonicalize_values(
    value: Any,
    *,
    path: str,
    enum_name: str,
    actions: list[dict[str, Any]],
) -> Any:
    if not isinstance(value, list):
        return value
    aliases = ENUM_ALIASES.get(enum_name, {})
    normalized: list[Any] = []
    for item in value:
        replacement = aliases.get(item, item)
        if replacement is not None and replacement not in normalized:
            normalized.append(replacement)
    if "无" in normalized and len(normalized) > 1:
        normalized = [item for item in normalized if item != "无"]
    if not normalized:
        normalized = ["无"]
    _record_action(
        actions,
        path=path,
        original=value,
        normalized=normalized,
        reason="受控枚举同义表达归一化",
    )
    return normalized


def normalize_rating_contract(value: Any) -> tuple[Any, list[dict[str, Any]]]:
    """只规范字段位置、枚举同义词和冗余标记，不修改难度等级。"""
    normalized = copy.deepcopy(value)
    actions: list[dict[str, Any]] = []
    if not isinstance(normalized, dict):
        return normalized, actions

    features = normalized.get("features")
    if isinstance(features, dict):
        for field in ("question_context", "curriculum_scope"):
            if field in normalized and field not in features:
                original = normalized.pop(field)
                features[field] = original
                _record_action(
                    actions,
                    path=field,
                    original="top_level",
                    normalized=f"features.{field}",
                    reason="固定字段位置归一化",
                )

        knowledge = features.get("knowledge")
        if isinstance(knowledge, dict) and isinstance(knowledge.get("topic_ids"), list):
            known_topics = load_curriculum_topics()
            original_ids = knowledge["topic_ids"]
            valid_ids = [topic_id for topic_id in original_ids if topic_id in known_topics]
            if valid_ids and valid_ids != original_ids:
                knowledge["topic_ids"] = list(dict.fromkeys(valid_ids))
                _record_action(
                    actions,
                    path="features.knowledge.topic_ids",
                    original=original_ids,
                    normalized=knowledge["topic_ids"],
                    reason="删除受控教材目录中不存在的topic_id；保留其余合法知识点",
                )

        solution = features.get("solution_process")
        if isinstance(solution, dict):
            solution["task_types"] = _canonicalize_values(
                solution.get("task_types"),
                path="features.solution_process.task_types",
                enum_name="task_types",
                actions=actions,
            )
        list_fields = (
            ("information_processing", "information_processing"),
            ("experiment_tasks", "experiment_tasks"),
            ("expression_requirements", "expression_requirements"),
        )
        for field, enum_name in list_fields:
            features[field] = _canonicalize_values(
                features.get(field),
                path=f"features.{field}",
                enum_name=enum_name,
                actions=actions,
            )

        reactions = features.get("reaction_processes")
        if isinstance(reactions, dict):
            reactions["processes"] = _canonicalize_values(
                reactions.get("processes"),
                path="features.reaction_processes.processes",
                enum_name="reaction_processes",
                actions=actions,
            )
            expected = (
                isinstance(reactions.get("processes"), list)
                and "反应条件选择" in reactions["processes"]
            )
            original = reactions.get("requires_condition_selection")
            if isinstance(original, bool) and original != expected:
                reactions["requires_condition_selection"] = expected
                _record_action(
                    actions,
                    path="features.reaction_processes.requires_condition_selection",
                    original=original,
                    normalized=expected,
                    reason="由反应条件选择枚举统一冗余布尔标记",
                )

        calculation = features.get("calculation")
        if isinstance(calculation, dict):
            calculation["types"] = _canonicalize_values(
                calculation.get("types"),
                path="features.calculation.types",
                enum_name="calculation_types",
                actions=actions,
            )
            calculation["special_methods"] = _canonicalize_values(
                calculation.get("special_methods"),
                path="features.calculation.special_methods",
                enum_name="special_methods",
                actions=actions,
            )
            task_types = solution.get("task_types") if isinstance(solution, dict) else None
            signals_calculation = any((
                calculation.get("has_calculation") is True,
                calculation.get("calculation_steps") in STEPS,
                _has_any(calculation.get("types")) if isinstance(calculation.get("types"), list) else False,
                _has_any(calculation.get("special_methods")) if isinstance(calculation.get("special_methods"), list) else False,
                isinstance(task_types, list) and "定量计算" in task_types,
            ))
            if isinstance(calculation.get("has_calculation"), bool) and calculation["has_calculation"] != signals_calculation:
                original = calculation["has_calculation"]
                calculation["has_calculation"] = signals_calculation
                _record_action(
                    actions,
                    path="features.calculation.has_calculation",
                    original=original,
                    normalized=signals_calculation,
                    reason="由计算任务、步数和类型统一冗余布尔标记",
                )
            if signals_calculation:
                if isinstance(task_types, list) and "定量计算" not in task_types:
                    original = list(task_types)
                    task_types.append("定量计算")
                    _record_action(
                        actions,
                        path="features.solution_process.task_types",
                        original=original,
                        normalized=task_types,
                        reason="计算字段与一级任务类别一致化",
                    )
                if calculation.get("calculation_steps") == "无":
                    calculation["calculation_steps"] = "1步"
                    _record_action(
                        actions,
                        path="features.calculation.calculation_steps",
                        original="无",
                        normalized="1步",
                        reason="存在明确计算任务时补足最低计算步数",
                    )
                if calculation.get("types") == ["无"]:
                    calculation["types"] = ["其他课内定量计算"]
                    _record_action(
                        actions,
                        path="features.calculation.types",
                        original=["无"],
                        normalized=["其他课内定量计算"],
                        reason="存在明确计算任务但模型未细分计算类型",
                    )

        conditions = features.get("difficulty_conditions")
        if isinstance(conditions, dict):
            conditions["hidden_conditions"] = _canonicalize_values(
                conditions.get("hidden_conditions"),
                path="features.difficulty_conditions.hidden_conditions",
                enum_name="hidden_conditions",
                actions=actions,
            )
            conditions["interference_points"] = _canonicalize_values(
                conditions.get("interference_points"),
                path="features.difficulty_conditions.interference_points",
                enum_name="interference_points",
                actions=actions,
            )

        context = features.get("question_context")
        if isinstance(context, dict):
            for field, enum_name in (
                ("unfamiliar_materials", "unfamiliar_materials"),
                ("interdisciplinary_context", "interdisciplinary_context"),
            ):
                context[field] = _canonicalize_values(
                    context.get(field),
                    path=f"features.question_context.{field}",
                    enum_name=enum_name,
                    actions=actions,
                )

    reasoning = normalized.get("reasoning")
    if isinstance(reasoning, dict):
        extras = []
        for field, label in (("why_not_lower", "为什么不更低"), ("why_not_higher", "为什么不更高")):
            if str(reasoning.get(field, "")).strip():
                extras.append(f"{label}：{str(reasoning.pop(field)).strip()}")
        if extras and "level_basis" in reasoning:
            original = reasoning["level_basis"]
            reasoning["level_basis"] = "；".join([str(original).strip(), *extras])
            _record_action(
                actions,
                path="reasoning.level_basis",
                original=original,
                normalized=reasoning["level_basis"],
                reason="相邻档理由合并到固定reasoning字段",
            )
    return normalized, actions


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
    if calculation["has_calculation"] != ("定量计算" in task_types):
        raise ChemistrySchemaError("has_calculation必须与task_types中的“定量计算”一致")

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
    normalized, normalization_actions = normalize_rating_contract(value)
    result = validate_rating_contract(normalized)
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
        "feature_normalization_actions": copy.deepcopy(normalization_actions),
    }
    result["postprocess_actions"] = normalization_actions
    return result
