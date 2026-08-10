"""初中化学教师口径特征契约、结构化输出 schema 与审计后处理。"""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any


FEATURE_SCHEMA_VERSION = "junior_chemistry_teacher_factors_v9"
CURRICULUM_PATH = Path(__file__).resolve().parent.parent / "JUNIOR_CHEMISTRY_CURRICULUM.md"
TOOL_NAME = "submit_junior_chemistry_rating"

LEVELS = ("送分题", "基础题", "中等题", "拔高题", "压轴题")
SCOPES = ("within_junior", "out_of_scope")
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}

# 30 个细粒度核心特征均为单选枚举。顺序同时用于 Prompt 文档测试和 JSON Schema。
# 具体知识点另由受控 topic_id 列表表达，避免把知识内容压成抽象标签。
FEATURE_OPTIONS: dict[str, tuple[str, ...]] = {
    "task_count": ("1项", "2-3项", "4项及以上"),
    "knowledge_distribution": (
        "单一知识点", "同知识点重复判断", "同单元不同知识点",
        "跨单元不同知识点", "多单元综合",
    ),
    "chemical_object_distribution": (
        "无明确化学对象", "单一化学对象", "同类多个化学对象",
        "不同类多个化学对象", "多类化学对象综合",
    ),
    "step_count": (
        "0步（直接识记）", "1步", "2-3步", "4-5步", "6步及以上",
    ),
    "task_relation": ("单项任务", "多项独立", "前后依赖", "多条任务链汇合"),
    "solution_method": (
        "直接识记或辨认", "一条规则直接应用", "多条规则分别判断",
        "连续推导", "定性与定量联合",
    ),
    "classification_discussion": ("无", "单一情况讨论", "多情况分类讨论"),
    "reverse_tracing": ("无", "有"),
    "visual_content": (
        "无图片信息", "仪器图", "基础装置图", "微观示意图", "数据表格",
        "反应流程图", "工业流程图", "曲线图", "多组对比实验表格",
        "多装置组合实验图", "多种图像综合",
    ),
    "visual_item_count": ("无图片", "1幅", "2-3幅", "4幅及以上"),
    "visual_complexity": (
        "无图片信息", "单一同类型图像", "多个同类型图像",
        "多个不同类型图像", "复杂高难图像",
    ),
    "information_operation": (
        "无需额外提取", "直接读取一个信息", "比较或整理多条信息",
        "由图表变化推断", "图像拐点或分段分析", "多来源信息筛选联合",
    ),
    "reaction_count": ("0个", "1个", "2-3个", "4个及以上"),
    "reaction_relation": (
        "无反应关系", "单一反应", "多个反应并列", "多个反应连续",
        "反应先后或过量不足", "分情况或竞争反应",
    ),
    "experiment_operation": (
        "无", "仪器识别或名称", "基本操作或读数判断", "装置选择或连接",
        "气体检验或验满", "试剂选择或物质检验", "多项实验操作联合",
    ),
    "experiment_analysis": (
        "无", "实验现象判断", "装置作用或实验目的", "实验原因解释",
        "控制变量", "根据现象或数据得出结论", "多个实验分析任务联合",
    ),
    "experiment_design": (
        "无", "补充实验步骤或操作", "根据结论设计操作", "实验方案设计",
        "实验方案评价", "实验改进", "多阶段探究设计",
    ),
    "error_analysis": (
        "无", "量筒读数误差", "天平称量误差", "实验操作导致误差",
        "装置或方案导致误差", "定量实验误差分析", "多种误差联合分析",
    ),
    "calculation_type": (
        "无", "化合价或化学式计算", "相对分子质量或元素质量计算",
        "化学方程式计算", "溶质质量分数或稀释计算", "溶解度计算",
        "反应后气体沉淀固体质量", "反应后溶液或溶质质量",
        "含杂质计算", "图像分段计算", "实验误差定量计算", "多类计算综合",
    ),
    "calculation_steps": ("无", "1步", "2-3步", "4步及以上"),
    "calculation_structure": (
        "无任何计算", "一步或口算", "单个化学方程式计算",
        "多个化学反应计算", "含杂质多步质量分数",
        "实验误差定量计算", "多模型综合计算",
    ),
    "special_method": (
        "无", "质量守恒", "元素守恒", "差量法", "极值法",
        "分情况计算", "多方程式联立", "循环反应计算", "多种特殊方法联合",
    ),
    "hidden_condition_count": ("0个", "1个", "2个", "3个及以上"),
    "hidden_condition_type": (
        "无", "前后对象要求", "反应条件", "过量或不足", "反应先后",
        "物质或溶液状态", "纯净干燥或杂质", "气体水分或质量损失",
        "图像拐点或分段", "剩余物或变质程度", "多类条件联合",
    ),
    "condition_relation": (
        "无条件限制", "单一条件", "多个独立条件",
        "多个关联条件", "多层嵌套条件",
    ),
    "interference_type": (
        "无", "易混概念", "多个选项规则切换", "规范表述易错",
        "干扰数据", "特例或边界", "体系质量关系易错",
        "多种剩余情况或竞争解释",
    ),
    "expression_type": (
        "无", "元素离子符号或化学式书写", "仪器操作或试剂名称书写",
        "化学方程式书写", "实验现象或操作规范描述",
        "原因或结论规范表达", "计算过程书写", "多类规范表达联合",
    ),
    "subjective_response": ("无", "有"),
    "given_information": (
        "题干未提供新增规则", "题干给出一条新事实或规则",
        "题干给出多条新事实或规则", "题干给出陌生物质或反应资料",
        "题干给出陌生装置流程或材料资料",
    ),
    "cross_subject": (
        "无", "古文或文言理解", "物理知识参与", "生物知识参与",
        "生产流程或工程信息参与", "多学科信息参与",
    ),
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


def _object_schema(properties: dict[str, Any], required: tuple[str, ...] | list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": False,
    }


def rating_json_schema() -> dict[str, Any]:
    """生成服务端约束和本地测试共用的唯一 JSON Schema。"""
    topic_ids = tuple(load_curriculum_topics())
    feature_properties: dict[str, Any] = {
        "knowledge": _object_schema({
            "topic_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(topic_ids)},
            },
        }, ("topic_ids",)),
    }
    for field, options in FEATURE_OPTIONS.items():
        feature_properties[field] = {"type": "string", "enum": list(options)}
    feature_properties["curriculum_scope"] = _object_schema({
        "scope": {"type": "string", "enum": list(SCOPES)},
        "extra_points": {
            "type": "array",
            "items": {"type": "string"},
        },
    }, ("scope", "extra_points"))

    reasoning_fields = (
        "knowledge_points", "solution_process", "main_difficulty_factors", "level_basis",
    )
    return _object_schema({
        "features": _object_schema(
            feature_properties,
            ("knowledge", *FEATURE_OPTIONS.keys(), "curriculum_scope"),
        ),
        "reasoning": _object_schema(
            {field: {"type": "string"} for field in reasoning_fields},
            reasoning_fields,
        ),
        "difficulty_level": {"type": "string", "enum": list(LEVELS)},
    }, ("features", "reasoning", "difficulty_level"))


def rating_tool_definition() -> dict[str, Any]:
    """Responses API 强制提交评级结果的函数工具定义。"""
    return {
        "type": "function",
        "name": TOOL_NAME,
        "description": "提交初中化学难度评级。所有特征必须直接选择受控枚举。",
        "parameters": rating_json_schema(),
        "strict": True,
    }


def _exact_dict(value: Any, expected: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ChemistrySchemaError(
            f"{name}字段不匹配: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ChemistrySchemaError(f"{name}必须是字符串数组")
    stripped = [item.strip() for item in value if item.strip()]
    if len(stripped) != len(set(stripped)):
        raise ChemistrySchemaError(f"{name}不得包含重复值")
    return stripped


def validate_rating_contract(value: Any) -> dict[str, Any]:
    """严格接受工具 schema 的原始输出；不猜测、不补词、不做近义映射。"""
    value = _exact_dict(value, {"features", "reasoning", "difficulty_level"}, "顶层")
    if value["difficulty_level"] not in LEVELS:
        raise ChemistrySchemaError("difficulty_level非法")

    expected_features = {"knowledge", *FEATURE_OPTIONS.keys(), "curriculum_scope"}
    features = _exact_dict(value["features"], expected_features, "features")
    knowledge = _exact_dict(features["knowledge"], {"topic_ids"}, "knowledge")
    topic_ids = _string_list(knowledge["topic_ids"], "knowledge.topic_ids")
    topics = load_curriculum_topics()
    unknown = [topic_id for topic_id in topic_ids if topic_id not in topics]
    if unknown:
        raise ChemistrySchemaError(f"未收录topic_id: {unknown}")

    validated_features: dict[str, Any] = {"knowledge": {"topic_ids": topic_ids}}
    for field, options in FEATURE_OPTIONS.items():
        raw = features[field]
        if not isinstance(raw, str) or raw not in options:
            raise ChemistrySchemaError(
                f"features.{field}非法值{raw!r}；允许值={list(options)}"
            )
        validated_features[field] = raw

    scope = _exact_dict(
        features["curriculum_scope"], {"scope", "extra_points"}, "curriculum_scope"
    )
    if scope["scope"] not in SCOPES:
        raise ChemistrySchemaError("curriculum_scope.scope非法")
    extra_points = _string_list(scope["extra_points"], "curriculum_scope.extra_points")
    if scope["scope"] == "within_junior" and (not topic_ids or extra_points):
        raise ChemistrySchemaError("within_junior必须选择topic_id且extra_points必须为空")
    if scope["scope"] == "out_of_scope" and not extra_points:
        raise ChemistrySchemaError("out_of_scope必须列出具体超纲内容")
    validated_features["curriculum_scope"] = {
        "scope": scope["scope"], "extra_points": extra_points,
    }

    reasoning_fields = {
        "knowledge_points", "solution_process", "main_difficulty_factors", "level_basis",
    }
    reasoning = _exact_dict(value["reasoning"], reasoning_fields, "reasoning")
    normalized_reasoning = {field: str(reasoning[field]).strip() for field in reasoning_fields}
    if any(not text for text in normalized_reasoning.values()):
        raise ChemistrySchemaError("reasoning字段不得为空")

    return {
        "features": validated_features,
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


def _normalize_feature_consistency(result: dict[str, Any]) -> list[dict[str, Any]]:
    """只修复能由其他受控字段或topic_id唯一确定的枚举组合。"""
    features = result["features"]
    actions: list[dict[str, Any]] = []

    def replace(field: str, final_value: str, reason: str) -> None:
        original_value = features[field]
        if original_value == final_value:
            return
        features[field] = final_value
        actions.append({
            "field": field,
            "original_value": original_value,
            "final_value": final_value,
            "reason": reason,
        })

    knowledge = features["knowledge"]
    if knowledge["knowledge_point_count"] <= 1:
        if features["knowledge_distribution"] not in {
            "单一知识点", "同知识点重复判断",
        }:
            replace("knowledge_distribution", "单一知识点", "按去重topic_id数量校正")
    elif knowledge["unit_count"] == 1:
        replace("knowledge_distribution", "同单元不同知识点", "按topic_id所属单元校正")
    elif knowledge["unit_count"] == 2:
        replace("knowledge_distribution", "跨单元不同知识点", "按topic_id所属单元校正")
    else:
        replace("knowledge_distribution", "多单元综合", "按topic_id所属单元校正")

    if features["task_count"] == "1项":
        replace("task_relation", "单项任务", "单项任务的关系唯一确定")
    elif features["task_relation"] == "单项任务":
        replace("task_relation", "多项独立", "多项任务不能标为单项任务")

    if features["reaction_count"] == "0个":
        replace("reaction_relation", "无反应关系", "零个反应时关系唯一确定")
    elif features["reaction_relation"] == "无反应关系":
        fallback = "单一反应" if features["reaction_count"] == "1个" else "多个反应并列"
        replace("reaction_relation", fallback, "存在反应时不能标为无反应关系")

    if features["visual_content"] == "无图片信息":
        replace("visual_item_count", "无图片", "无图片信息时数量唯一确定")
        replace("visual_complexity", "无图片信息", "无图片信息时复杂度唯一确定")
    else:
        if features["visual_item_count"] == "无图片":
            replace("visual_item_count", "1幅", "存在图片内容时至少有1幅")
        if features["visual_complexity"] == "无图片信息":
            replace("visual_complexity", "单一同类型图像", "存在图片内容时不能标为无图片信息")
        if (
            features["visual_item_count"] in {"2-3幅", "4幅及以上"}
            and features["visual_complexity"] == "单一同类型图像"
        ):
            replace("visual_complexity", "多个同类型图像", "多幅图片时不能标为单一图像")

    if features["calculation_type"] == "无":
        replace("calculation_steps", "无", "无计算时步骤唯一确定")
        replace("calculation_structure", "无任何计算", "无计算时结构唯一确定")
        replace("special_method", "无", "无计算时特殊方法唯一确定")
    else:
        if features["calculation_steps"] == "无":
            replace("calculation_steps", "1步", "存在计算时至少有1步")
        if features["calculation_structure"] == "无任何计算":
            structure_by_type = {
                "化学方程式计算": "单个化学方程式计算",
                "含杂质计算": "含杂质多步质量分数",
                "实验误差定量计算": "实验误差定量计算",
                "多类计算综合": "多模型综合计算",
            }
            replace(
                "calculation_structure",
                structure_by_type.get(features["calculation_type"], "一步或口算"),
                "存在计算时不能标为无任何计算",
            )

    if features["hidden_condition_count"] == "0个":
        replace("hidden_condition_type", "无", "零个隐藏条件时类型唯一确定")
        replace("condition_relation", "无条件限制", "零个隐藏条件时关系唯一确定")
    elif features["condition_relation"] == "无条件限制":
        fallback = (
            "单一条件" if features["hidden_condition_count"] == "1个" else "多个独立条件"
        )
        replace("condition_relation", fallback, "存在隐藏条件时不能标为无条件限制")

    replace(
        "subjective_response",
        "无" if features["expression_type"] == "无" else "有",
        "是否主观作答由具体表达要求唯一确定",
    )
    return actions


def _build_audit_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    features = result["features"]
    knowledge = features["knowledge"]
    level = result["difficulty_level"]
    candidates: list[dict[str, Any]] = []

    nontrivial = {
        "experiment": features["experiment_analysis"] not in {"无", "实验现象判断"}
        or features["experiment_design"] != "无",
        "hidden_condition": features["hidden_condition_count"] != "0个",
        "interference": features["interference_type"] in {
            "规范表述易错", "干扰数据", "特例或边界",
            "体系质量关系易错", "多种剩余情况或竞争解释",
        },
        "expression": features["expression_type"] != "无",
    }
    if level == "送分题" and any(nontrivial.values()):
        candidates.append(_candidate(
            "B1_easy_with_nontrivial_factor", "基础题",
            "送分题包含隐藏条件、干扰、规范表达或非基础实验任务。",
            {key: value for key, value in nontrivial.items() if value},
        ))
    hard_experiment = (
        features["experiment_design"] in {
            "实验方案评价", "实验改进", "多阶段探究设计",
        }
        or features["error_analysis"] != "无"
    )
    hard_calculation = (
        features["calculation_steps"] == "4步及以上"
        and features["special_method"] != "无"
    )
    if level == "中等题" and (hard_experiment or hard_calculation):
        candidates.append(_candidate(
            "H1_medium_decisive_task", "拔高题",
            "存在高阶实验或特殊计算模型，需复核局部高难下限。",
            {
                "experiment_design": features["experiment_design"],
                "calculation_type": features["calculation_type"],
                "special_method": features["special_method"],
            },
        ))
    if level == "压轴题" and features["task_relation"] not in {
        "前后依赖", "多条任务链汇合",
    }:
        candidates.append(_candidate(
            "F1_final_without_dependency", "拔高题",
            "压轴题未呈现前后依赖或多链汇合，需复核是否仅为高难特征堆叠。",
            {
                "task_relation": features["task_relation"],
                "step_count": features["step_count"],
            },
        ))
    return candidates


def _plain_char_count(value: Any) -> int:
    text = re.sub(r"<[^>]+>", "", str(value or ""))
    return len(re.sub(r"\s+", "", text))


def _count_options(value: Any) -> int:
    if isinstance(value, (list, tuple, dict)):
        return len(value)
    text = str(value or "")
    labels = re.findall(r"(?:^|\n|\s)([A-H])[\.．、:：)]\s*", text, flags=re.I)
    return len(dict.fromkeys(label.upper() for label in labels))


def _count_subquestions(data: dict[str, Any]) -> int:
    structured = data.get("sub_questions")
    structured_count = len(structured) if isinstance(structured, list) else 0
    text = "\n".join(str(data.get(field, "") or "") for field in ("stem", "options"))
    inline = re.findall(r"(?:^|\n|[；;。])\s*[（(](\d+)[）)]", text)
    inline_count = len(dict.fromkeys(inline))
    return max(structured_count, inline_count)


def _count_stem_images(data: dict[str, Any]) -> int:
    raw = data.get("stem_pic_url")
    if isinstance(raw, (list, tuple)):
        url_count = len([item for item in raw if str(item).strip()])
    else:
        url_count = len([
            item for item in re.split(r"[,，;；\s]+", str(raw or "")) if item.strip()
        ])
    placeholder_count = len(re.findall(r"<image\b|\[图片\]|【图片】", str(data.get("stem", "") or ""), re.I))
    return max(url_count, placeholder_count)


def compute_question_statistics(data: dict[str, Any]) -> dict[str, Any]:
    """计算无需模型判断的客观题面量，供审计和确定性边界规则使用。"""
    stem_char_count = _plain_char_count(data.get("stem"))
    if stem_char_count <= 60:
        stem_length_band = "60字以内"
    elif stem_char_count <= 100:
        stem_length_band = "61-100字"
    elif stem_char_count <= 300:
        stem_length_band = "101-300字"
    else:
        stem_length_band = "300字以上"
    option_count = _count_options(data.get("options"))
    subquestion_count = _count_subquestions(data)
    return {
        "stem_char_count": stem_char_count,
        "stem_length_band": stem_length_band,
        "option_count": option_count,
        "subquestion_count": subquestion_count,
        "stem_image_count": _count_stem_images(data),
        "question_item_count": max(option_count, subquestion_count, 1),
    }


def _writeback_floor(
    result: dict[str, Any],
    floor_level: str,
    rule: str,
    reason: str,
    evidence: dict[str, Any],
) -> None:
    current_level = result["difficulty_level"]
    if LEVEL_INDEX[current_level] >= LEVEL_INDEX[floor_level]:
        return
    result["difficulty_level"] = floor_level
    result["postprocess_actions"].append({
        "rule": rule,
        "original_level": current_level,
        "final_level": floor_level,
        "reason": reason,
        "evidence": evidence,
        "difficulty_level_changed": True,
    })


def postprocess_chemistry_difficulty(value: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """严格校验、计算客观题面量，并执行教师明确给出的最低档规则。"""
    result = validate_rating_contract(copy.deepcopy(value))
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
    normalization_actions = _normalize_feature_consistency(result)

    original_level = result["difficulty_level"]
    question_statistics = compute_question_statistics(data)
    candidates = _build_audit_candidates(result)
    result["postprocess_actions"] = []

    if result["features"]["error_analysis"] != "无":
        _writeback_floor(
            result, "中等题", "T1_error_analysis_floor",
            "教师规则：涉及实验误差分析时最低为中等题。",
            {"error_analysis": result["features"]["error_analysis"]},
        )
    final_level = result["difficulty_level"]
    result["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    result["postprocess"] = {
        "original_level": original_level,
        "final_level": final_level,
        "writeback_applied": final_level != original_level,
        "question_statistics": question_statistics,
        "candidates": candidates,
        "feature_normalization_actions": normalization_actions,
    }
    return result
