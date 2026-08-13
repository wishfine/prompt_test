"""初中化学教师口径特征契约、结构化输出 schema 与审计后处理。"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


FEATURE_SCHEMA_VERSION = "junior_chemistry_teacher_factors_v22"
CURRICULUM_PATH = Path(__file__).resolve().parent.parent / "JUNIOR_CHEMISTRY_CURRICULUM.md"
TOOL_NAME = "submit_junior_chemistry_rating"

LEVELS = ("送分题", "基础题", "中等题", "拔高题", "压轴题")
SCOPES = ("within_junior", "out_of_scope")
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}

# knowledge.topic_ids 是第 1 个核心特征；以下 18 个字段均为单选枚举。
# 共 19 个显式核心特征，不再用隐藏派生字段重复表达同一证据。
FEATURE_OPTIONS: dict[str, tuple[str, ...]] = {
    "knowledge_coverage": ("单一知识点", "同单元多知识点", "跨两个单元", "多单元综合"),
    "chemical_object_distribution": (
        "无明确化学对象", "单一化学对象", "同类多个化学对象",
        "不同类多个化学对象", "多类化学对象综合",
    ),
    "task_structure": (
        "单一任务", "2-3项独立任务", "4项及以上独立任务",
        "多项任务共享同一模型", "前后依赖任务", "多条任务链汇合",
    ),
    "step_count": ("0步（直接识记）", "1步", "2-3步", "4-5步", "6步及以上"),
    "solution_method": (
        "直接识记或辨认", "一条规则直接应用", "多条规则分别判断",
        "连续推导", "根据结果反推物质或组成", "定性与定量联合",
        "分类讨论", "实验探究或方案评价",
    ),
    "information_carrier": (
        "无额外信息", "普通文字材料", "古文或文言材料", "跨学科材料",
        "生活标识图", "仪器或基础装置图", "微观示意图", "数据表格",
        "曲线图", "反应流程图", "工业流程图", "多装置组合图",
        "多种信息载体联合",
    ),
    "information_operation": (
        "无需额外提取", "直接读取一个信息", "比较或整理多条信息",
        "由图表变化推断", "图像拐点或分段分析", "多来源信息筛选联合",
        "提取题给新规则并应用",
    ),
    "reaction_structure": (
        "无反应", "单一反应", "2-3个并列反应", "2-3个连续反应",
        "4个以上反应网络", "反应先后或过量不足", "分情况或竞争反应",
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
        "实验方案评价", "实验改进", "多阶段探究设计", "多类实验设计任务联合",
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
    "calculation_structure": (
        "无计算", "一步或口算", "2-3步常规计算", "4步及以上常规计算",
        "单个化学方程式完整计算", "多个反应连续计算",
        "含杂质或反应后体系计算", "图像分段计算",
        "实验误差定量计算", "多模型综合计算",
    ),
    "special_method": (
        "无", "质量守恒", "元素守恒", "差量法", "极值法",
        "分情况计算", "多方程式联立", "循环反应计算", "多种特殊方法联合",
    ),
    "condition_structure": (
        "无隐藏条件", "一个隐藏条件", "多个独立条件",
        "多个关联条件", "多层嵌套条件",
    ),
    "interference_type": (
        "无", "易混概念", "多个选项规则切换", "规范表述易错",
        "干扰数据", "特例或边界", "体系质量关系易错",
        "多种剩余情况或竞争解释", "多类干扰联合",
    ),
    "expression_type": (
        "无", "元素离子符号或化学式书写", "仪器操作或试剂名称书写",
        "化学方程式书写", "实验现象或操作规范描述",
        "原因或结论规范表达", "计算过程书写", "数值或简短答案填写",
        "物质名称性质或用途填写", "多类规范表达联合",
    ),
}

MODEL_FEATURE_OPTIONS = FEATURE_OPTIONS
FEATURE_RESIDUAL_OPTIONS: dict[str, str] = {}
FEATURE_MULTI_OPTIONS: dict[str, str] = {
    "information_carrier": "多种信息载体联合",
    "experiment_operation": "多项实验操作联合",
    "experiment_analysis": "多个实验分析任务联合",
    "experiment_design": "多类实验设计任务联合",
    "error_analysis": "多种误差联合分析",
    "calculation_type": "多类计算综合",
    "calculation_structure": "多模型综合计算",
    "special_method": "多种特殊方法联合",
    "interference_type": "多类干扰联合",
    "expression_type": "多类规范表达联合",
}
EXHAUSTIVE_FEATURE_FIELDS = frozenset(FEATURE_OPTIONS)


def validate_feature_registry() -> None:
    all_fields = set(FEATURE_OPTIONS)
    residual_fields = set(FEATURE_RESIDUAL_OPTIONS)
    if EXHAUSTIVE_FEATURE_FIELDS | residual_fields != all_fields:
        missing = all_fields - EXHAUSTIVE_FEATURE_FIELDS - residual_fields
        extra = (EXHAUSTIVE_FEATURE_FIELDS | residual_fields) - all_fields
        raise RuntimeError(f"特征覆盖分类不完整: missing={sorted(missing)}, extra={sorted(extra)}")
    if EXHAUSTIVE_FEATURE_FIELDS & residual_fields:
        raise RuntimeError("穷尽型特征与开放型特征不得重叠")
    for field, residual in FEATURE_RESIDUAL_OPTIONS.items():
        if residual not in FEATURE_OPTIONS[field] or residual == FEATURE_OPTIONS[field][0]:
            raise RuntimeError(f"{field}的保留证据兜底项配置错误")
    for registry in (FEATURE_MULTI_OPTIONS, FEATURE_ALIASES):
        for field, values in registry.items():
            targets = values.values() if isinstance(values, dict) else (values,)
            if any(target not in FEATURE_OPTIONS[field] for target in targets):
                raise RuntimeError(f"{field}存在指向非法枚举的映射")
    if set(MODEL_FEATURE_OPTIONS) != all_fields:
        raise RuntimeError("模型侧与本地侧特征字段必须完全一致")
    for field, options in MODEL_FEATURE_OPTIONS.items():
        if not options or any(option not in FEATURE_OPTIONS[field] for option in options):
            raise RuntimeError(f"{field}的模型侧精简枚举配置错误")


def validate_prompt_feature_catalog(prompt: str) -> None:
    """启动前验证Prompt的18项枚举与运行时Schema完全同源。"""
    documented = {
        field: tuple(re.findall(r"`([^`]+)`", values))
        for field, values in re.findall(
            r"#### \d+\. ([a-z_]+)\s*\n只能是：([^\n]+)", prompt
        )
    }
    if set(documented) != set(MODEL_FEATURE_OPTIONS):
        raise RuntimeError(
            "Prompt特征字段与Schema不一致: "
            f"missing={sorted(set(MODEL_FEATURE_OPTIONS) - set(documented))}, "
            f"extra={sorted(set(documented) - set(MODEL_FEATURE_OPTIONS))}"
        )
    mismatched = [
        field for field, options in MODEL_FEATURE_OPTIONS.items()
        if documented[field] != options
    ]
    if mismatched:
        raise RuntimeError(f"Prompt枚举顺序或内容与Schema不一致: {mismatched}")

# 模型侧只保留互不重复的字段。规范化必须保留模型已经明确表达的
# “存在反应/条件/图像分析/特殊方法”等实质证据，禁止因枚举串值而回落为无或0。
FEATURE_DEFAULTS: dict[str, str] = {
    field: options[0] for field, options in FEATURE_OPTIONS.items()
}
FEATURE_ALIASES: dict[str, dict[str, str]] = {
    "information_carrier": {
        "无图片信息": "无额外信息",
        "仪器图": "仪器或基础装置图",
        "基础装置图": "仪器或基础装置图",
        "多种图像综合": "多种信息载体联合",
    },
    "experiment_operation": {
        "仪器识别": "仪器识别或名称",
        "仪器名称": "仪器识别或名称",
        "基本操作": "基本操作或读数判断",
        "装置连接": "装置选择或连接",
    },
    "experiment_analysis": {
        "现象判断": "实验现象判断",
        "实验目的": "装置作用或实验目的",
        "实验目的分析": "装置作用或实验目的",
        "原因解释": "实验原因解释",
    },
    "calculation_type": {
        "单个化学方程式计算": "化学方程式计算",
        "多个化学反应计算": "化学方程式计算",
        "反应后气体质量": "反应后气体沉淀固体质量",
        "反应后沉淀质量": "反应后气体沉淀固体质量",
        "反应后固体质量": "反应后气体沉淀固体质量",
        "反应后溶液质量": "反应后溶液或溶质质量",
        "反应后溶质质量": "反应后溶液或溶质质量",
        "化学式计算": "化合价或化学式计算",
        "元素质量计算": "相对分子质量或元素质量计算",
        "元素质量分数计算": "相对分子质量或元素质量计算",
        "溶质质量分数计算": "溶质质量分数或稀释计算",
        "溶质质量分数稀释计算": "溶质质量分数或稀释计算",
    },
    "calculation_structure": {
        "无": "无计算",
        "无任何计算": "无计算",
        "口算": "一步或口算",
        "多步常规计算": "2-3步常规计算",
        "多个化学式计算": "2-3步常规计算",
        "多步化学式计算": "2-3步常规计算",
        "多项常规计算": "2-3步常规计算",
        "一个化学方程式计算": "单个化学方程式完整计算",
        "单个化学方程式计算": "单个化学方程式完整计算",
        "多反应计算": "多个反应连续计算",
        "多个反应计算": "多个反应连续计算",
        "多个化学反应计算": "多个反应连续计算",
        "含杂质计算": "含杂质或反应后体系计算",
        "含杂质多步质量分数": "含杂质或反应后体系计算",
        "多模型计算": "多模型综合计算",
    },
    "special_method": {
        "差量": "差量法",
        "极值": "极值法",
        "元素质量守恒": "元素守恒",
        "总质量守恒": "质量守恒",
    },
    "expression_type": {
        "计算结果填写": "数值或简短答案填写",
        "数值填写": "数值或简短答案填写",
        "简短答案填写": "数值或简短答案填写",
        "化学式或物质名称书写": "物质名称性质或用途填写",
        "物质名称书写": "物质名称性质或用途填写",
        "物质性质填写": "物质名称性质或用途填写",
        "物质用途填写": "物质名称性质或用途填写",
    },
}

validate_feature_registry()


class ChemistrySchemaError(ValueError):
    """模型输出不满足初中化学严格契约。"""


def _repair_missing_reasoning_quotes(text: str) -> str:
    fields = (
        "knowledge_points", "solution_process",
        "main_difficulty_factors", "level_basis",
    )
    next_fields = "|".join((*fields, "difficulty_level"))
    for field in fields:
        pattern = re.compile(
            rf'("{field}"\s*:\s*)([^"\s].*?)(?=,\s*"(?:{next_fields})"\s*:)',
            flags=re.S,
        )

        def add_quotes(match: re.Match[str]) -> str:
            value = match.group(2).strip()
            if value.endswith('"'):
                value = value[:-1].rstrip()
            return f'{match.group(1)}"{value}"'

        text = pattern.sub(add_quotes, text, count=1)
    return text


def _escape_unescaped_json_quotes(text: str) -> str:
    """仅转义JSON字符串内部的裸双引号，不改字段和值。"""
    output: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if not in_string:
            output.append(char)
            if char == '"':
                in_string = True
            continue
        if escaped:
            output.append(char)
            escaped = False
            continue
        if char == "\\":
            output.append(char)
            escaped = True
            continue
        if char != '"':
            output.append(char)
            continue
        following = text[index + 1:]
        next_nonspace = next((item for item in following if not item.isspace()), "")
        if next_nonspace in {"", ":", ",", "}", "]"}:
            output.append(char)
            in_string = False
        else:
            output.append('\\"')
    return "".join(output)


def parse_model_json_text(text: str) -> tuple[dict[str, Any], str]:
    """恢复完整对象中的引号错误或思考前缀；不补字段、不猜特征。"""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        parsed = None
    if isinstance(parsed, dict):
        return parsed, "strict"

    starts = [index for index, char in enumerate(str(text or "")) if char == "{"]
    decoder = json.JSONDecoder()
    for start in reversed(starts):
        candidate = str(text)[start:]
        candidate = _repair_missing_reasoning_quotes(candidate)
        candidate = _escape_unescaped_json_quotes(candidate)
        try:
            recovered, end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(recovered, dict):
            continue
        if not {"features", "reasoning", "difficulty_level"}.issubset(recovered):
            continue
        if candidate[end:].strip():
            continue
        return recovered, "local_recovered"
    return {}, "failed"


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
                "uniqueItems": True,
            },
        }, ("topic_ids",)),
    }
    for field, options in MODEL_FEATURE_OPTIONS.items():
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
        "reasoning": _object_schema({
            "knowledge_points": {"type": "string"},
            "solution_process": {
                "type": "array", "items": {"type": "string"}, "minItems": 1,
            },
            "main_difficulty_factors": {"type": "string"},
            "level_basis": {"type": "string"},
        }, reasoning_fields),
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


def _string_list(
    value: Any,
    name: str,
    *,
    deduplicate: bool = False,
) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ChemistrySchemaError(f"{name}必须是字符串数组")
    stripped = [item.strip() for item in value if item.strip()]
    if len(stripped) != len(set(stripped)) and not deduplicate:
        raise ChemistrySchemaError(f"{name}不得包含重复值")
    return list(dict.fromkeys(stripped)) if deduplicate else stripped


def _clean_enum_text(value: Any) -> str:
    return re.sub(r"[\s`'\"，,。；;：:]", "", str(value or ""))


def canonicalize_feature_value(
    field: str,
    value: Any,
    source: dict[str, Any],
) -> tuple[str, str]:
    """把单个模型特征确定性收敛到该字段的受控枚举。"""
    options = FEATURE_OPTIONS[field]
    if isinstance(value, str) and value in options:
        return value, "原值已是合法枚举"

    cleaned = _clean_enum_text(value)
    exact_by_cleaned = {
        _clean_enum_text(option): option for option in options
    }
    if cleaned in exact_by_cleaned:
        return exact_by_cleaned[cleaned], "清除空白或标点后匹配合法枚举"

    aliases = {
        _clean_enum_text(alias): final_value
        for alias, final_value in FEATURE_ALIASES.get(field, {}).items()
    }
    if cleaned in aliases:
        return aliases[cleaned], "按该字段的受控同义词映射"

    excluded = {
        FEATURE_DEFAULTS[field],
        FEATURE_RESIDUAL_OPTIONS.get(field, ""),
        FEATURE_MULTI_OPTIONS.get(field, ""),
    }
    contained = [
        option for option in options
        if option not in excluded
        if len(_clean_enum_text(option)) >= 4
        and _clean_enum_text(option) in cleaned
    ]
    contained.extend(
        target
        for alias, target in FEATURE_ALIASES.get(field, {}).items()
        if target not in excluded
        and len(_clean_enum_text(alias)) >= 2
        and _clean_enum_text(alias) in cleaned
    )
    contained = list(dict.fromkeys(contained))
    if len(contained) >= 2 and field in FEATURE_MULTI_OPTIONS:
        return FEATURE_MULTI_OPTIONS[field], "自由表述包含同字段多个合法类别，收敛到联合枚举"
    if len(contained) == 1:
        return contained[0], "自由表述中只包含一个合法枚举或受控同义词"

    if field == "calculation_structure":
        calculation_type = str(source.get("calculation_type", "") or "")
        structure_by_type = {
            "化学方程式计算": "单个化学方程式完整计算",
            "含杂质计算": "含杂质或反应后体系计算",
            "反应后气体沉淀固体质量": "含杂质或反应后体系计算",
            "反应后溶液或溶质质量": "含杂质或反应后体系计算",
            "图像分段计算": "图像分段计算",
            "实验误差定量计算": "实验误差定量计算",
            "多类计算综合": "多模型综合计算",
        }
        if calculation_type in structure_by_type:
            return structure_by_type[calculation_type], "依据计算类型确定计算结构"
        if calculation_type not in {"", "无"}:
            return "2-3步常规计算", "计算类型明确但结构串值，保留为常规计算"

    # 计算类型发生字段串值时，不能回落为“无”并清空其他计算证据。
    if field == "calculation_type":
        special_method = str(source.get("special_method", "") or "")
        calculation_structure = str(source.get("calculation_structure", "") or "")
        if calculation_structure not in {"", "无计算"} or special_method not in {"", "无"}:
            return "多类计算综合", "计算证据明确存在但计算对象无法唯一确定"

    residual = FEATURE_RESIDUAL_OPTIONS.get(field)
    if residual is not None and str(value or "").strip():
        return residual, "具体类别无法唯一映射，保留该类任务证据而不回落为无"

    return FEATURE_DEFAULTS[field], "无法唯一映射，使用该字段中性默认值"


def _canonicalize_topic_ids(
    raw_knowledge: Any,
    reasoning: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    """只在知识点文字能唯一支持时修复模型臆造或串值的topic_id。"""
    if not isinstance(raw_knowledge, dict):
        raise ChemistrySchemaError("knowledge必须是对象")
    raw_ids = raw_knowledge.get("topic_ids")
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, list):
        raise ChemistrySchemaError("knowledge.topic_ids必须是数组")

    topics = load_curriculum_topics()
    reasoning_text = " ".join(
        str(item or "")
        for item in (reasoning.values() if isinstance(reasoning, dict) else ())
    )
    final_ids = [
        str(raw_id).strip()
        for raw_id in raw_ids
        if str(raw_id).strip() in topics
    ]
    actions: list[dict[str, Any]] = []
    for raw_id in raw_ids:
        text = str(raw_id or "").strip()
        if text in topics:
            continue

        embedded = [topic_id for topic_id in topics if topic_id in text]
        if len(embedded) == 1:
            final_ids.append(embedded[0])
            actions.append({
                "field": "knowledge.topic_ids",
                "original_value": text,
                "final_value": embedded[0],
                "reason": "从串值中提取唯一合法topic_id",
            })
            continue

        unit_match = re.match(r"^(U\d{2})_T\d+", text)
        unit_hint = unit_match.group(1) if unit_match else ""
        context = f"{text} {reasoning_text}"
        scored: list[tuple[int, str]] = []
        for topic_id, topic in topics.items():
            if unit_hint and topic["unit_id"] != unit_hint:
                continue
            names = [topic["topic_name"], *re.split(r"[；;、]", topic["aliases"])]
            matches = [
                name.strip()
                for name in names
                if len(name.strip()) >= 3 and name.strip() in context
            ]
            if matches:
                scored.append((max(len(name) for name in matches), topic_id))
        scored.sort(reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            mapped = scored[0][1]
            final_ids.append(mapped)
            actions.append({
                "field": "knowledge.topic_ids",
                "original_value": text,
                "final_value": mapped,
                "reason": "依据知识点说明与受控目录唯一匹配",
            })
            continue

        if final_ids:
            actions.append({
                "field": "knowledge.topic_ids",
                "original_value": text,
                "final_value": None,
                "reason": "删除无法唯一映射的额外topic_id；保留其他合法知识点",
            })
            continue
        raise ChemistrySchemaError(f"未收录且无法唯一映射topic_id: {text!r}")

    return list(dict.fromkeys(final_ids)), actions


def normalize_rating_contract(value: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """规范模型枚举与字段集合；不修改模型给出的难度等级。"""
    if not isinstance(value, dict):
        raise ChemistrySchemaError("顶层必须是对象")
    if not isinstance(value.get("features"), dict):
        raise ChemistrySchemaError("features必须是对象")

    normalized = copy.deepcopy(value)
    source = normalized["features"]
    actions: list[dict[str, Any]] = []
    expected = {"knowledge", *FEATURE_OPTIONS.keys(), "curriculum_scope"}

    topic_ids, topic_actions = _canonicalize_topic_ids(
        source.get("knowledge"), normalized.get("reasoning")
    )
    source["knowledge"] = {"topic_ids": topic_ids}
    actions.extend(topic_actions)

    for field in sorted(set(source) - expected):
        actions.append({
            "field": field,
            "original_value": copy.deepcopy(source[field]),
            "final_value": None,
            "reason": "删除模型Schema中不存在的额外字段",
        })
        source.pop(field)

    for field in FEATURE_OPTIONS:
        original = source.get(field)
        final_value, reason = canonicalize_feature_value(field, original, source)
        source[field] = final_value
        if original != final_value:
            actions.append({
                "field": field,
                "original_value": copy.deepcopy(original),
                "final_value": final_value,
                "reason": reason,
            })

    return validate_rating_contract(normalized), actions


def validate_rating_contract(value: Any) -> dict[str, Any]:
    """严格接受工具 schema 的原始输出；不猜测、不补词、不做近义映射。"""
    value = _exact_dict(value, {"features", "reasoning", "difficulty_level"}, "顶层")
    if value["difficulty_level"] not in LEVELS:
        raise ChemistrySchemaError("difficulty_level非法")

    expected_features = {"knowledge", *FEATURE_OPTIONS.keys(), "curriculum_scope"}
    features = _exact_dict(value["features"], expected_features, "features")
    knowledge = _exact_dict(features["knowledge"], {"topic_ids"}, "knowledge")
    topic_ids = _string_list(
        knowledge["topic_ids"],
        "knowledge.topic_ids",
        deduplicate=True,
    )
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
    solution_process = _string_list(
        reasoning["solution_process"], "reasoning.solution_process", deduplicate=True,
    )
    normalized_reasoning = {
        "knowledge_points": str(reasoning["knowledge_points"]).strip(),
        "solution_process": solution_process,
        "main_difficulty_factors": str(reasoning["main_difficulty_factors"]).strip(),
        "level_basis": str(reasoning["level_basis"]).strip(),
    }
    if (
        not normalized_reasoning["knowledge_points"]
        or not solution_process
        or not normalized_reasoning["main_difficulty_factors"]
        or not normalized_reasoning["level_basis"]
    ):
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
    """只修复无歧义矛盾；不从一个特征推导另一项难度证据。"""
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

    if features["calculation_type"] == "无":
        replace("calculation_structure", "无计算", "无计算时结构唯一确定")
        replace("special_method", "无", "无计算时特殊方法唯一确定")
    elif features["calculation_structure"] == "无计算":
        replace("calculation_structure", "2-3步常规计算", "存在计算时不能标为无计算")
    return actions


def _apply_question_evidence_corrections(
    result: dict[str, Any], data: dict[str, Any],
) -> list[dict[str, Any]]:
    """仅用题面可直接确认的事实，阻止模型把实际题图清成“无图片”。"""
    features = result["features"]
    text = "\n".join(
        str(data.get(field, "") or "") for field in ("stem", "options")
    )
    has_image = bool(data.get("image_input_used")) or "<image>" in text or any(
        data.get(field) for field in ("stem_pic_url", "options_pic_url")
    )
    if not (
        has_image
        and features["information_carrier"] == "无额外信息"
        and any(keyword in text for keyword in ("标识", "标志", "图标"))
    ):
        return []
    original = features["information_carrier"]
    features["information_carrier"] = "生活标识图"
    return [{
        "field": "information_carrier",
        "original_value": original,
        "final_value": "生活标识图",
        "reason": "题面明确包含需要辨认的生活标识图，禁止把实际图片清成无图片信息",
    }]


def _build_audit_candidates(result: dict[str, Any]) -> list[dict[str, Any]]:
    """生成可解释的相邻档复核项；单一信号不直接改档。"""
    features = result["features"]
    level = result["difficulty_level"]
    candidates: list[dict[str, Any]] = []
    if level == "送分题":
        nontrivial = {
            "实验分析": features["experiment_analysis"] not in {"无", "实验现象判断"},
            "实验设计": features["experiment_design"] != "无",
            "隐藏条件": features["condition_structure"] != "无隐藏条件",
            "决定性干扰": features["interference_type"] not in {"无", "易混概念"},
            "规范表达": features["expression_type"] != "无",
        }
        if any(nontrivial.values()):
            candidates.append(_candidate(
                "B1_easy_with_nontrivial_factor", "基础题",
                "送分题包含实际参与作答的实验、条件、干扰或规范表达，需复核1/2档边界。",
                {name: active for name, active in nontrivial.items() if active},
            ))
    if level == "中等题":
        hard_signals = {
            "高阶实验": features["experiment_design"] in {
                "实验方案评价", "实验改进", "多阶段探究设计", "多类实验设计任务联合",
            } or features["error_analysis"] in {"定量实验误差分析", "多种误差联合分析"},
            "特殊方法": features["special_method"] in {
                "差量法", "极值法", "分情况计算", "多方程式联立",
                "循环反应计算", "多种特殊方法联合",
            },
            "复杂计算": features["calculation_structure"] in {
                "4步及以上常规计算", "多个反应连续计算",
                "含杂质或反应后体系计算", "图像分段计算",
                "实验误差定量计算", "多模型综合计算",
            },
            "复杂反应": features["reaction_structure"] in {
                "4个以上反应网络", "反应先后或过量不足", "分情况或竞争反应",
            },
            "关联条件": features["condition_structure"] in {"多个关联条件", "多层嵌套条件"},
            "复杂信息": features["information_operation"] in {
                "图像拐点或分段分析", "多来源信息筛选联合",
            },
        }
        matched = [name for name, active in hard_signals.items() if active]
        if matched:
            candidate = _candidate(
                "H2_medium_teacher_hard_signal_review", "拔高题",
                "命中教师关注的高难信号，进入3/4档复核；只有至少两类独立信号共同成立才允许升档。",
                {"matched_signals": matched, "required_signal_count": 2},
            )
            candidate["writeback_allowed"] = len(matched) >= 2
            candidates.append(candidate)
    return candidates


def _build_upper_level_review_candidate(
    result: dict[str, Any], data: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """用19个显式特征复核相邻边界，不再读取旧的重复字段。"""
    features = result["features"]
    data = data or {}
    text = "\n".join(str(data.get(field, "") or "") for field in ("stem", "options"))
    level = result["difficulty_level"]

    if level == "送分题":
        topic_ids = set(features["knowledge"]["topic_ids"])
        anchors = {
            "一步化合价或化学式计算": features["calculation_type"] == "化合价或化学式计算",
            "质量守恒后再计算": "U05_T01" in topic_ids and features["calculation_type"] != "无",
            "方程式意义易混": "U05_T02" in topic_ids and features["interference_type"] == "易混概念",
            "跨文字语境后判断": features["information_carrier"] in {"古文或文言材料", "跨学科材料"},
            "多条规则分别判断": features["solution_method"] == "多条规则分别判断",
            "需要规范书写或表达": features["expression_type"] != "无",
            "存在隐藏条件": features["condition_structure"] != "无隐藏条件",
        }
        matched = [name for name, active in anchors.items() if active]
        if matched:
            return _candidate(
                "M1_giveaway_to_foundation_teacher_anchor", "基础题",
                "命中教师已确认的1/2档锚点；一步不等于送分，仍需看知识规则、语境、条件和表达。",
                {"matched_anchors": matched},
            )

    if level == "基础题":
        medium_signals = {
            "知识覆盖": features["knowledge_coverage"] in {"跨两个单元", "多单元综合"},
            "多项任务": features["task_structure"] in {
                "4项及以上独立任务", "多项任务共享同一模型",
                "前后依赖任务", "多条任务链汇合",
            },
            "多步过程": features["step_count"] in {"4-5步", "6步及以上"},
            "信息处理": features["information_operation"] in {
                "比较或整理多条信息", "由图表变化推断",
                "图像拐点或分段分析", "多来源信息筛选联合",
            },
            "实验分析设计": features["experiment_analysis"] not in {"无", "实验现象判断"}
            or features["experiment_design"] != "无",
            "计算结构": features["calculation_structure"] not in {"无计算", "一步或口算"},
            "条件干扰": features["condition_structure"] in {"多个关联条件", "多层嵌套条件"}
            or features["interference_type"] in {
                "干扰数据", "体系质量关系易错", "多种剩余情况或竞争解释", "多类干扰联合",
            },
        }
        matched = [name for name, active in medium_signals.items() if active]
        if len(matched) >= 2:
            return _candidate(
                "M2_foundation_to_medium_multi_factor_review", "中等题",
                "至少两类独立的知识、任务、信息、实验、计算或条件证据共同成立。",
                {"matched_signals": matched, "required_signal_count": 2},
            )

    if level == "中等题":
        candidates = _build_audit_candidates(result)
        return candidates[-1] if candidates else None

    if level == "拔高题":
        pressure_groups = {
            "复杂任务结构": features["task_structure"] in {"前后依赖任务", "多条任务链汇合"}
            and features["step_count"] in {"4-5步", "6步及以上"},
            "复杂反应体系": features["reaction_structure"] in {
                "4个以上反应网络", "反应先后或过量不足", "分情况或竞争反应",
            },
            "复杂定量模型": features["calculation_structure"] in {
                "多个反应连续计算", "含杂质或反应后体系计算",
                "图像分段计算", "实验误差定量计算", "多模型综合计算",
            } and features["special_method"] != "无",
            "关联条件与排除": features["condition_structure"] in {"多个关联条件", "多层嵌套条件"}
            and features["interference_type"] in {
                "体系质量关系易错", "多种剩余情况或竞争解释", "多类干扰联合",
            },
            "多来源实验信息": features["information_operation"] == "多来源信息筛选联合"
            and (
                features["experiment_analysis"] == "多个实验分析任务联合"
                or features["experiment_design"] in {"多阶段探究设计", "多类实验设计任务联合"}
            ),
        }
        matched = [name for name, active in pressure_groups.items() if active]
        if len(matched) >= 3:
            return _candidate(
                "R2_hard_to_final_multi_feature_review", "压轴题",
                "至少三类独立高难环节在同一解题结构中共同成立，4/5档复核通过。",
                {"matched_signals": matched, "required_signal_count": 3},
            )
    return None


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
    """规范特征，并用多项独立证据复核中等题和拔高题的向上边界。"""
    result, model_normalization_actions = normalize_rating_contract(
        copy.deepcopy(value)
    )
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
    question_evidence_actions = _apply_question_evidence_corrections(result, data)
    normalization_actions = (
        model_normalization_actions
        + question_evidence_actions
        + _normalize_feature_consistency(result)
    )

    original_level = result["difficulty_level"]
    question_statistics = compute_question_statistics(data)
    candidates = _build_audit_candidates(result)
    review_candidate = _build_upper_level_review_candidate(result, data)
    if review_candidate is not None:
        candidates.append(review_candidate)
    result["postprocess_actions"] = []

    if result["features"]["error_analysis"] != "无":
        _writeback_floor(
            result, "中等题", "T1_error_analysis_floor",
            "教师规则：涉及实验误差分析时最低为中等题。",
            {"error_analysis": result["features"]["error_analysis"]},
        )
    if (
        review_candidate is not None
        and review_candidate.get("writeback_allowed", True)
    ):
        before_review = result["difficulty_level"]
        _writeback_floor(
            result,
            review_candidate["candidate_level"],
            review_candidate["rule"],
            review_candidate["reason"],
            review_candidate["evidence"],
        )
        review_candidate["writeback_applied"] = (
            result["difficulty_level"] != before_review
        )
    elif review_candidate is not None:
        review_candidate["writeback_applied"] = False
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
