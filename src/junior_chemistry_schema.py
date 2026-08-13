"""初中化学教师口径特征契约、结构化输出 schema 与审计后处理。"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any


FEATURE_SCHEMA_VERSION = "junior_chemistry_teacher_factors_v28"
CURRICULUM_PATH = Path(__file__).resolve().parent.parent / "JUNIOR_CHEMISTRY_CURRICULUM.md"
TOOL_NAME = "submit_junior_chemistry_rating"

LEVELS = ("送分题", "基础题", "中等题", "拔高题", "压轴题")
SCOPES = ("within_junior", "out_of_scope")
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
ADJACENT_LEVEL_PAIRS = tuple(
    f"{LEVELS[index]}/{LEVELS[index + 1]}" for index in range(len(LEVELS) - 1)
)
REVIEW_DIMENSIONS = (
    "无", "知识覆盖", "任务关系", "实质步骤", "信息处理", "反应结构",
    "实验任务", "误差分析", "计算结构或方法", "条件障碍", "表达要求",
)
REVIEW_RESOLUTIONS = ("维持较低档", "升至较高档")

# knowledge.topic_ids 是第 1 个核心特征；以下12个字段中，4个天然并列字段
# 使用受控枚举数组，其余字段为单选。知识覆盖由topic_id确定性计算，不重复要求模型判断。
FEATURE_OPTIONS: dict[str, tuple[str, ...]] = {
    "task_structure": (
        "单一任务", "多个独立任务",
        "多项任务共享同一模型", "前后依赖任务", "多条任务链汇合",
    ),
    "step_count": ("0步（直接识记）", "1步", "2-3步", "4-5步", "6步及以上"),
    "information_complexity": (
        "无额外信息处理", "直接读取单一信息", "比较整理多条信息",
        "图表装置或流程推断", "拐点分段或多来源联合",
        "题给新规则迁移应用",
    ),
    "reaction_structure": (
        "无反应", "单一反应", "2-3个并列反应", "2-3个连续反应",
        "4个以上反应网络", "反应先后或过量不足", "分情况或竞争反应",
    ),
    "experiment_operation": (
        "无", "仪器识别或名称", "基本操作或读数判断", "装置选择或连接",
        "气体检验或验满", "试剂选择或物质检验",
    ),
    "experiment_analysis": (
        "无", "实验现象判断", "装置作用或实验目的", "实验原因解释",
        "控制变量分析", "根据现象或数据得出结论",
    ),
    "experiment_design": (
        "无", "补充实验步骤或操作", "根据结论设计操作", "实验方案设计",
        "实验方案评价", "实验改进", "多阶段探究设计",
    ),
    "error_analysis": (
        "无", "量筒读数误差", "天平称量误差", "实验操作导致误差",
        "装置或方案导致误差", "定量实验误差分析", "多种误差联合分析",
    ),
    "calculation_structure": (
        "无计算", "直接数值或比例关系", "多项常规独立计算",
        "单个化学方程式完整计算", "多个反应连续计算",
        "含杂质或反应后体系计算", "图像分段计算",
        "实验误差定量计算", "多模型综合计算",
    ),
    "special_method": (
        "无", "质量守恒", "元素守恒", "差量法", "极值法",
        "分情况计算", "多方程式联立", "循环反应计算", "多种特殊方法联合",
    ),
    "difficulty_obstacle": (
        "无明显障碍", "一个轻微易混点或条件", "1-2个独立条件或标准陷阱",
        "多个关联条件", "多层嵌套条件或竞争解释", "大量干扰或矛盾信息",
    ),
    "expression_requirement": (
        "无", "元素离子符号或化学式书写", "仪器操作或试剂名称书写",
        "化学方程式书写", "实验现象或操作规范描述",
        "原因或结论规范表达", "计算过程书写",
    ),
}

MULTI_FEATURE_FIELDS = frozenset({
    "experiment_operation", "experiment_analysis", "experiment_design",
    "expression_requirement",
})

MODEL_FEATURE_OPTIONS = FEATURE_OPTIONS
FEATURE_RESIDUAL_OPTIONS: dict[str, str] = {}
FEATURE_MULTI_OPTIONS: dict[str, str] = {
    "error_analysis": "多种误差联合分析",
    "calculation_structure": "多模型综合计算",
    "special_method": "多种特殊方法联合",
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
    """启动前验证Prompt特征枚举与运行时Schema完全同源。"""
    documented = {
        field: tuple(re.findall(r"`([^`]+)`", values))
        for field, values in re.findall(
            r"#### \d+\. ([a-z_]+)(?:（受控多选）)?\s*\n只能是：([^\n]+)", prompt
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
    documented_multi = set(re.findall(r"#### \d+\. ([a-z_]+)（受控多选）", prompt))
    if documented_multi != set(MULTI_FEATURE_FIELDS):
        raise RuntimeError(
            "Prompt多选字段与Schema不一致: "
            f"missing={sorted(MULTI_FEATURE_FIELDS - documented_multi)}, "
            f"extra={sorted(documented_multi - MULTI_FEATURE_FIELDS)}"
        )

# 模型侧只保留互不重复的字段。规范化必须保留模型已经明确表达的
# “存在反应/条件/图像分析/特殊方法”等实质证据，禁止因枚举串值而回落为无或0。
FEATURE_DEFAULTS: dict[str, str] = {
    field: options[0] for field, options in FEATURE_OPTIONS.items()
}
FEATURE_ALIASES: dict[str, dict[str, str]] = {
    "task_structure": {
        "2-3项独立任务": "多个独立任务",
        "4项及以上独立任务": "多个独立任务",
    },
    "information_complexity": {
        "无图片信息": "无额外信息处理",
        "无需额外提取": "无额外信息处理",
        "直接读取一个信息": "直接读取单一信息",
        "比较或整理多条信息": "比较整理多条信息",
        "由图表变化推断": "图表装置或流程推断",
        "图像拐点或分段分析": "拐点分段或多来源联合",
        "多来源信息筛选联合": "拐点分段或多来源联合",
        "提取题给新规则并应用": "题给新规则迁移应用",
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
        "控制变量": "控制变量分析",
    },
    "calculation_structure": {
        "无": "无计算",
        "无任何计算": "无计算",
        "口算": "直接数值或比例关系",
        "一步或口算": "直接数值或比例关系",
        "2-3步常规计算": "多项常规独立计算",
        "4步及以上常规计算": "多项常规独立计算",
        "多步常规计算": "多项常规独立计算",
        "多个化学式计算": "多项常规独立计算",
        "多步化学式计算": "多项常规独立计算",
        "多项常规计算": "多项常规独立计算",
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
    "difficulty_obstacle": {
        "无": "无明显障碍",
        "无隐藏条件": "无明显障碍",
        "一个隐藏条件": "一个轻微易混点或条件",
        "易混概念": "一个轻微易混点或条件",
        "特例或边界": "一个轻微易混点或条件",
        "多个独立条件": "1-2个独立条件或标准陷阱",
        "干扰数据": "1-2个独立条件或标准陷阱",
        "多个关联条件": "多个关联条件",
        "体系质量关系易错": "多个关联条件",
        "多层嵌套条件": "多层嵌套条件或竞争解释",
        "多种剩余情况或竞争解释": "多层嵌套条件或竞争解释",
        "多类干扰联合": "大量干扰或矛盾信息",
    },
    "expression_requirement": {
        "数值或简短答案填写": "无",
        "计算结果填写": "无",
        "数值填写": "无",
        "简短答案填写": "无",
        "化学式或物质名称书写": "元素离子符号或化学式书写",
        "物质名称书写": "仪器操作或试剂名称书写",
        "物质性质填写": "原因或结论规范表达",
        "物质用途填写": "原因或结论规范表达",
    },
}

validate_feature_registry()


class ChemistrySchemaError(ValueError):
    """模型输出不满足初中化学严格契约。"""


def _repair_missing_reasoning_quotes(text: str) -> str:
    fields = (
        "level_basis",
    )
    next_fields = "|".join((*fields, "feature_review", "difficulty_level"))
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
        candidate = re.sub(
            r'("knowledge"\s*:\s*\{\s*"topic_ids"\s*:\s*\[[^\]]*\])\s*,\s*("[a-z_]+"\s*:)',
            r'\1}, \2',
            candidate,
            count=1,
        )
        candidate = _repair_missing_reasoning_quotes(candidate)
        candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
        candidate = _escape_unescaped_json_quotes(candidate)
        for suffix in ("", "}", "}}"):
            repaired = candidate + suffix
            try:
                recovered, end = decoder.raw_decode(repaired)
            except json.JSONDecodeError:
                continue
            if not isinstance(recovered, dict):
                continue
            if not {"features", "reasoning", "feature_review", "difficulty_level"}.issubset(recovered):
                continue
            if repaired[end:].strip():
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
        if field in MULTI_FEATURE_FIELDS:
            feature_properties[field] = {
                "type": "array",
                "items": {"type": "string", "enum": list(options)},
                "minItems": 1,
                "uniqueItems": True,
            }
        else:
            feature_properties[field] = {"type": "string", "enum": list(options)}
    feature_properties["step_count"]["description"] = (
        "按普通初中学生完成标准解答所需的最长连续显性步骤选择；"
        "不得按模型熟练解法合并识别规律、建立关系、求中间量和继续推导。"
    )
    feature_properties["curriculum_scope"] = _object_schema({
        "scope": {"type": "string", "enum": list(SCOPES)},
        "extra_points": {
            "type": "array",
            "items": {"type": "string"},
        },
    }, ("scope", "extra_points"))

    reasoning_fields = ("longest_substantive_chain", "level_basis")
    feature_review_fields = (
        "adjacent_pair", "supports_higher_level", "limits_higher_level",
        "resolution", "review_basis",
    )
    return _object_schema({
        "features": _object_schema(
            feature_properties,
            ("knowledge", *FEATURE_OPTIONS.keys(), "curriculum_scope"),
        ),
        "reasoning": _object_schema({
            "longest_substantive_chain": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": (
                    "只列最难任务的最长实质解题链；每项是一个会改变后续解法的学生化学决策。"
                    "不得合并规律识别、关系建立和必要中间量，也不得把抄写、重复代入、"
                    "纯算术整理或最终填答案机械拆成新步骤。"
                ),
            },
            "level_basis": {"type": "string"},
        }, reasoning_fields),
        "feature_review": _object_schema({
            "adjacent_pair": {"type": "string", "enum": list(ADJACENT_LEVEL_PAIRS)},
            "supports_higher_level": {
                "type": "array", "items": {"type": "string", "enum": list(REVIEW_DIMENSIONS)},
                "minItems": 1, "uniqueItems": True,
            },
            "limits_higher_level": {
                "type": "array", "items": {"type": "string", "enum": list(REVIEW_DIMENSIONS)},
                "minItems": 1, "uniqueItems": True,
            },
            "resolution": {"type": "string", "enum": list(REVIEW_RESOLUTIONS)},
            "review_basis": {"type": "string"},
        }, feature_review_fields),
        "difficulty_level": {"type": "string", "enum": list(LEVELS)},
    }, ("features", "reasoning", "feature_review", "difficulty_level"))


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

    residual = FEATURE_RESIDUAL_OPTIONS.get(field)
    if residual is not None and str(value or "").strip():
        return residual, "具体类别无法唯一映射，保留该类任务证据而不回落为无"

    return FEATURE_DEFAULTS[field], "无法唯一映射，使用该字段中性默认值"


def canonicalize_multi_feature_value(
    field: str,
    value: Any,
    source: dict[str, Any],
) -> tuple[list[str], str]:
    """把天然并列任务收敛为去重的受控枚举数组。"""
    raw_values = value if isinstance(value, list) else [value]
    final_values: list[str] = []
    reasons: list[str] = []
    for raw in raw_values:
        if isinstance(raw, str) and raw in FEATURE_OPTIONS[field]:
            final_values.append(raw)
            continue
        cleaned = _clean_enum_text(raw)
        aliases = {
            _clean_enum_text(alias): target
            for alias, target in FEATURE_ALIASES.get(field, {}).items()
        }
        if cleaned in aliases:
            final_values.append(aliases[cleaned])
            reasons.append("受控同义词映射")
            continue
        matches = [
            option for option in FEATURE_OPTIONS[field]
            if option != FEATURE_DEFAULTS[field]
            and len(_clean_enum_text(option)) >= 4
            and _clean_enum_text(option) in cleaned
        ]
        matches.extend(
            target
            for alias, target in FEATURE_ALIASES.get(field, {}).items()
            if len(_clean_enum_text(alias)) >= 2
            and _clean_enum_text(alias) in cleaned
        )
        if matches:
            final_values.extend(matches)
            reasons.append("从并列表述中提取合法枚举")
            continue
        final, reason = canonicalize_feature_value(field, raw, source)
        final_values.append(final)
        reasons.append(reason)

    final_values = list(dict.fromkeys(final_values))
    default = FEATURE_DEFAULTS[field]
    if len(final_values) > 1 and default in final_values:
        final_values.remove(default)
        reasons.append("存在实质任务时删除无值")
    if not final_values:
        final_values = [default]
    return final_values, "；".join(dict.fromkeys(reasons)) or "原值已是合法受控数组"


def _canonicalize_topic_ids(
    raw_knowledge: Any,
    reasoning: Any,
) -> tuple[list[str], list[dict[str, Any]]]:
    """只在知识点文字能唯一支持时修复模型臆造或串值的topic_id。"""
    if isinstance(raw_knowledge, list):
        raw_knowledge = {"topic_ids": raw_knowledge}
    elif isinstance(raw_knowledge, str):
        raw_knowledge = {"topic_ids": [raw_knowledge]}
    if not isinstance(raw_knowledge, dict):
        raise ChemistrySchemaError("knowledge必须是对象、topic_id数组或topic_id字符串")
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

        actions.append({
            "field": "knowledge.topic_ids",
            "original_value": text,
            "final_value": None,
            "reason": "删除无法唯一映射的臆造topic_id；不因单个错误知识点使整题失败",
        })

    if not final_ids:
        context = reasoning_text
        scored: list[tuple[int, str]] = []
        for topic_id, topic in topics.items():
            names = [topic["topic_name"], *re.split(r"[；;、]", topic["aliases"])]
            matches = [
                name.strip() for name in names
                if len(name.strip()) >= 3 and name.strip() in context
            ]
            if matches:
                scored.append((max(len(name) for name in matches), topic_id))
        scored.sort(reverse=True)
        if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            final_ids.append(scored[0][1])
            actions.append({
                "field": "knowledge.topic_ids",
                "original_value": copy.deepcopy(raw_ids),
                "final_value": scored[0][1],
                "reason": "全部原topic_id非法，依据知识点说明唯一恢复一个受控topic_id",
            })
    if not final_ids:
        raise ChemistrySchemaError("knowledge.topic_ids没有可验证的受控知识点")

    return list(dict.fromkeys(final_ids)), actions


def normalize_rating_contract(
    value: Any, *, allow_legacy_fields: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """规范模型枚举与字段集合；不修改模型给出的难度等级。"""
    if not isinstance(value, dict):
        raise ChemistrySchemaError("顶层必须是对象")
    if not isinstance(value.get("features"), dict):
        raise ChemistrySchemaError("features必须是对象")

    normalized = copy.deepcopy(value)
    source = normalized["features"]
    actions: list[dict[str, Any]] = []
    expected = {"knowledge", *FEATURE_OPTIONS.keys(), "curriculum_scope"}
    legacy_fields = {
        "experiment_task", "calculation_complexity", "calculation_type",
        "condition_structure", "interference_type", "expression_type",
        "information_operation",
    }
    present_legacy = sorted(set(source) & legacy_fields)
    if present_legacy and not allow_legacy_fields:
        raise ChemistrySchemaError(f"新运行不得输出旧版字段: {present_legacy}")

    legacy_experiment_task = source.pop("experiment_task", None)
    if legacy_experiment_task is not None:
        values = legacy_experiment_task if isinstance(legacy_experiment_task, list) else [legacy_experiment_task]
        operation_options = set(FEATURE_OPTIONS["experiment_operation"])
        analysis_options = set(FEATURE_OPTIONS["experiment_analysis"])
        source.setdefault(
            "experiment_operation",
            [item for item in values if item in operation_options and item != "无"] or ["无"],
        )
        source.setdefault(
            "experiment_analysis",
            [item for item in values if item in analysis_options and item != "无"] or ["无"],
        )
        actions.append({
            "field": "experiment_task",
            "original_value": copy.deepcopy(legacy_experiment_task),
            "final_value": {
                "experiment_operation": copy.deepcopy(source["experiment_operation"]),
                "experiment_analysis": copy.deepcopy(source["experiment_analysis"]),
            },
            "reason": "将旧统一实验任务按操作与分析职责拆分",
        })

    legacy_information = source.get("information_operation")
    if "information_complexity" not in source and legacy_information is not None:
        source["information_complexity"] = legacy_information

    if "calculation_structure" not in source and "calculation_complexity" in source:
        source["calculation_structure"] = source["calculation_complexity"]

    if "difficulty_obstacle" not in source:
        legacy_condition = str(source.get("condition_structure", "") or "")
        legacy_interference = str(source.get("interference_type", "") or "")
        condition_values = {
            "无隐藏条件": "无明显障碍",
            "一个隐藏条件": "一个轻微易混点或条件",
            "多个独立条件": "1-2个独立条件或标准陷阱",
            "多个关联条件": "多个关联条件",
            "多层嵌套条件": "多层嵌套条件或竞争解释",
        }
        interference_values = {
            "无": "无明显障碍",
            "易混概念": "一个轻微易混点或条件",
            "多个选项规则切换": "1-2个独立条件或标准陷阱",
            "规范表述易错": "一个轻微易混点或条件",
            "干扰数据": "1-2个独立条件或标准陷阱",
            "特例或边界": "一个轻微易混点或条件",
            "体系质量关系易错": "多个关联条件",
            "多种剩余情况或竞争解释": "多层嵌套条件或竞争解释",
            "多类干扰联合": "大量干扰或矛盾信息",
        }
        obstacle_levels = list(FEATURE_OPTIONS["difficulty_obstacle"])
        candidates = [
            condition_values.get(legacy_condition, "无明显障碍"),
            interference_values.get(legacy_interference, "无明显障碍"),
        ]
        source["difficulty_obstacle"] = max(
            candidates, key=obstacle_levels.index
        )

    if "expression_requirement" not in source and "expression_type" in source:
        source["expression_requirement"] = source["expression_type"]

    if "curriculum_scope" not in source:
        source["curriculum_scope"] = {"scope": "within_junior", "extra_points": []}
        actions.append({
            "field": "curriculum_scope",
            "original_value": None,
            "final_value": copy.deepcopy(source["curriculum_scope"]),
            "reason": "模型漏字段；已有受控初中topic_id时补为课内审计默认值",
        })

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
        if field in MULTI_FEATURE_FIELDS:
            final_value, reason = canonicalize_multi_feature_value(field, original, source)
        else:
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
    value = _exact_dict(
        value,
        {"features", "reasoning", "feature_review", "difficulty_level"},
        "顶层",
    )
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
        if field in MULTI_FEATURE_FIELDS:
            values = _string_list(raw, f"features.{field}", deduplicate=True)
            if not values or any(item not in options for item in values):
                raise ChemistrySchemaError(
                    f"features.{field}存在非法值{raw!r}；允许值={list(options)}"
                )
            if len(values) > 1 and options[0] in values:
                raise ChemistrySchemaError(f"features.{field}的无值不得与实质任务并存")
            validated_features[field] = values
        else:
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

    reasoning_fields = {"longest_substantive_chain", "level_basis"}
    if isinstance(value.get("reasoning"), dict):
        value["reasoning"] = {
            key: item for key, item in value["reasoning"].items()
            if key in reasoning_fields
        }
    reasoning = _exact_dict(value["reasoning"], reasoning_fields, "reasoning")
    substantive_chain = _string_list(
        reasoning["longest_substantive_chain"],
        "reasoning.longest_substantive_chain",
        deduplicate=True,
    )
    normalized_reasoning = {
        "longest_substantive_chain": substantive_chain,
        "level_basis": str(reasoning["level_basis"]).strip(),
    }
    if (
        not substantive_chain
        or not normalized_reasoning["level_basis"]
    ):
        raise ChemistrySchemaError("reasoning字段不得为空")

    review_fields = {
        "adjacent_pair", "supports_higher_level", "limits_higher_level",
        "resolution", "review_basis",
    }
    review = _exact_dict(value["feature_review"], review_fields, "feature_review")
    if review["adjacent_pair"] not in ADJACENT_LEVEL_PAIRS:
        raise ChemistrySchemaError("feature_review.adjacent_pair非法")
    supports = _string_list(
        review["supports_higher_level"],
        "feature_review.supports_higher_level",
        deduplicate=True,
    )
    limits = _string_list(
        review["limits_higher_level"],
        "feature_review.limits_higher_level",
        deduplicate=True,
    )
    if (
        not supports or not limits
        or any(item not in REVIEW_DIMENSIONS for item in (*supports, *limits))
        or (len(supports) > 1 and "无" in supports)
        or (len(limits) > 1 and "无" in limits)
    ):
        raise ChemistrySchemaError("feature_review证据维度非法")
    if review["resolution"] not in REVIEW_RESOLUTIONS:
        raise ChemistrySchemaError("feature_review.resolution非法")
    lower_level, higher_level = review["adjacent_pair"].split("/")
    expected_level = (
        higher_level if review["resolution"] == "升至较高档" else lower_level
    )
    if value["difficulty_level"] != expected_level:
        raise ChemistrySchemaError(
            "feature_review.resolution与difficulty_level不一致"
        )
    review_basis = str(review["review_basis"]).strip()
    if not review_basis:
        raise ChemistrySchemaError("feature_review.review_basis不得为空")
    normalized_review = {
        "adjacent_pair": review["adjacent_pair"],
        "supports_higher_level": supports,
        "limits_higher_level": limits,
        "resolution": review["resolution"],
        "review_basis": review_basis,
    }

    return {
        "features": validated_features,
        "reasoning": normalized_reasoning,
        "feature_review": normalized_review,
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

    def replace(field: str, final_value: Any, reason: str) -> None:
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

    if features["calculation_structure"] == "无计算":
        replace("special_method", "无", "无计算时特殊方法唯一确定")

    # longest_substantive_chain只列最长实质链，一项对应一个会改变后续解法的
    # 学生化学决策。因此数组长度是step_count的确定性下限；不从题目关键词
    # 猜测未列出的步骤，也不把机械书写或算术拆分成步骤，更不直接修改难度。
    process_length = len(result["reasoning"]["longest_substantive_chain"])
    if process_length >= 6:
        process_floor = "6步及以上"
    elif process_length >= 4:
        process_floor = "4-5步"
    elif process_length >= 2:
        process_floor = "2-3步"
    else:
        process_floor = None
    if process_floor is not None:
        step_options = list(FEATURE_OPTIONS["step_count"])
        if step_options.index(features["step_count"]) < step_options.index(process_floor):
            replace(
                "step_count",
                process_floor,
                "longest_substantive_chain列出的实质决策已超过模型选择的步骤档位",
            )
    return actions


def _apply_question_evidence_corrections(
    result: dict[str, Any], data: dict[str, Any],
) -> list[dict[str, Any]]:
    """精简后的信息特征只表示处理负担，不再从图片形式反推难度。"""
    return []


def _active_evidence_groups(result: dict[str, Any], target_level: str) -> dict[str, Any]:
    """按Prompt维度归并证据；同一维度的多个字段最多计一次。"""
    features = result["features"]
    knowledge = features["knowledge"]
    expressions = features["expression_requirement"]
    experiment_operation = features["experiment_operation"]
    experiment_analysis = features["experiment_analysis"]
    experiment_design = features["experiment_design"]
    calculation = features["calculation_structure"]
    method = features["special_method"]

    if target_level == "基础题":
        groups = {
            "知识": knowledge["knowledge_point_count"] >= 2,
            "任务过程": features["task_structure"] != "单一任务"
            or features["step_count"] in {"2-3步", "4-5步", "6步及以上"},
            "信息": features["information_complexity"] not in {"无额外信息处理", "直接读取单一信息"},
            "反应": features["reaction_structure"] not in {"无反应", "单一反应"},
            "实验": experiment_analysis != ["无"] or experiment_design != ["无"]
            or any(item not in {"无", "仪器识别或名称", "基本操作或读数判断"} for item in experiment_operation),
            "计算": calculation != "无计算" or method != "无",
            "条件": features["difficulty_obstacle"] != "无明显障碍",
            "表达": expressions != ["无"],
        }
    elif target_level == "中等题":
        groups = {
            "知识": knowledge["knowledge_point_count"] >= 3,
            "任务过程": features["task_structure"] in {"多项任务共享同一模型", "前后依赖任务", "多条任务链汇合"}
            or features["step_count"] in {"2-3步", "4-5步", "6步及以上"},
            "信息": features["information_complexity"] in {
                "比较整理多条信息", "图表装置或流程推断", "拐点分段或多来源联合", "题给新规则迁移应用",
            },
            "反应": features["reaction_structure"] not in {"无反应", "单一反应"},
            "实验": experiment_analysis != ["无"] or experiment_design != ["无"] or features["error_analysis"] != "无",
            "计算": calculation not in {"无计算", "直接数值或比例关系"},
            "条件": features["difficulty_obstacle"] not in {"无明显障碍", "一个轻微易混点或条件"},
            "表达": len(expressions) >= 2 or any(item in {
                "实验现象或操作规范描述", "原因或结论规范表达", "计算过程书写",
            } for item in expressions),
        }
    else:
        high = target_level == "拔高题"
        groups = {
            "知识": knowledge["knowledge_point_count"] >= (3 if high else 4)
            and knowledge["cross_unit"],
            "任务过程": features["task_structure"] in ({"前后依赖任务", "多条任务链汇合"} if high else {"多条任务链汇合"})
            or features["step_count"] in ({"4-5步", "6步及以上"} if high else {"6步及以上"}),
            "信息": features["information_complexity"] in (
                {"图表装置或流程推断", "拐点分段或多来源联合", "题给新规则迁移应用"}
                if high else {"拐点分段或多来源联合", "题给新规则迁移应用"}
            ),
            "反应": features["reaction_structure"] in (
                {"2-3个连续反应", "4个以上反应网络", "反应先后或过量不足", "分情况或竞争反应"}
                if high else {"4个以上反应网络", "反应先后或过量不足", "分情况或竞争反应"}
            ),
            "实验": any(item in {"实验方案评价", "实验改进", "多阶段探究设计"} for item in experiment_design)
            or features["error_analysis"] in {"定量实验误差分析", "多种误差联合分析"},
            "计算": calculation in (
                {"多个反应连续计算", "含杂质或反应后体系计算", "图像分段计算", "实验误差定量计算", "多模型综合计算"}
                if high else {"多模型综合计算"}
            ) or method in (
                {"差量法", "极值法", "分情况计算", "多方程式联立", "循环反应计算", "多种特殊方法联合"}
                if high else {"分情况计算", "多方程式联立", "循环反应计算", "多种特殊方法联合"}
            ),
            "条件": features["difficulty_obstacle"] in (
                {"多个关联条件", "多层嵌套条件或竞争解释", "大量干扰或矛盾信息"}
                if high else {"多层嵌套条件或竞争解释", "大量干扰或矛盾信息"}
            ),
        }
    active = {name: value for name, value in groups.items() if value}
    if "反应" in active and "计算" in active:
        active.pop("反应")
        active.pop("计算")
        active["反应与计算"] = True
    return active


def _build_boundary_review_candidate(result: dict[str, Any]) -> dict[str, Any] | None:
    """具体特征组合可写回；普通证据组只生成相邻档位复核候选。"""
    current = result["difficulty_level"]
    if current == "压轴题":
        return None
    target = LEVELS[LEVEL_INDEX[current] + 1]
    groups = _active_evidence_groups(result, target)
    if not groups:
        return None
    features = result["features"]
    knowledge_count = features["knowledge"]["knowledge_point_count"]
    expressions = set(features["expression_requirement"])
    experiment_operation = set(features["experiment_operation"])
    experiment_analysis = set(features["experiment_analysis"])
    matched_paths: list[str] = []

    if target == "中等题":
        if knowledge_count >= 4 and "化学方程式书写" in expressions:
            matched_paths.append("至少4个知识点且要求书写化学方程式")
        if knowledge_count >= 3 and "实验现象或操作规范描述" in expressions:
            matched_paths.append("至少3个知识点且要求规范描述实验现象或操作")
    elif target == "拔高题":
        if features["reaction_structure"] == "反应先后或过量不足":
            matched_paths.append("需要处理反应先后或过量不足")
        if (
            features["calculation_structure"] == "多个反应连续计算"
            and features["step_count"] == "2-3步"
            and features["special_method"] == "质量守恒"
        ):
            matched_paths.append("多个反应连续计算并使用质量守恒")
        if (
            "试剂选择" in experiment_operation
            and "实验原因解释" in experiment_analysis
        ):
            matched_paths.append("同时完成试剂选择和实验原因解释")
    elif target == "压轴题":
        # 暂无经教师样本验证的4→5窄路径；保留候选证据供复核，不自动写回。
        pass

    allowed = bool(matched_paths)
    candidate = _candidate(
        f"R_{LEVEL_INDEX[current] + 1}_{LEVEL_INDEX[target] + 1}_adjacent_review",
        target,
        "普通证据只触发复核；只有命中具体、可解释的共同特征路径才允许相邻升档。",
        {
            "evidence_groups": groups,
            "matched_review_paths": matched_paths,
        },
    )
    candidate["writeback_allowed"] = allowed
    return candidate


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


def postprocess_chemistry_difficulty(
    value: dict[str, Any], data: dict[str, Any], *, allow_legacy_fields: bool = True,
) -> dict[str, Any]:
    """按Prompt证据生成复核候选，仅对验证过的窄路径写回等级。"""
    result, model_normalization_actions = normalize_rating_contract(
        copy.deepcopy(value), allow_legacy_fields=allow_legacy_fields,
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
    if len(topic_ids) == 1:
        coverage_type = "单一知识点"
    elif len(unit_ids) == 1:
        coverage_type = "同单元多知识点"
    elif len(unit_ids) == 2:
        coverage_type = "跨两个单元"
    else:
        coverage_type = "多单元综合"
    result["features"]["knowledge"] = {
        "topic_ids": topic_ids,
        "knowledge_points": points,
        "knowledge_point_count": len(topic_ids),
        "unit_count": len(unit_ids),
        "cross_unit": len(unit_ids) >= 2,
        "coverage_type": coverage_type,
    }
    question_evidence_actions = _apply_question_evidence_corrections(result, data)
    normalization_actions = (
        model_normalization_actions
        + question_evidence_actions
        + _normalize_feature_consistency(result)
    )

    original_level = result["difficulty_level"]
    question_statistics = compute_question_statistics(data)
    candidates: list[dict[str, Any]] = []
    review_candidate = _build_boundary_review_candidate(result)
    if review_candidate is not None:
        candidates.append(review_candidate)
    result["postprocess_actions"] = []

    if result["features"]["error_analysis"] in {
        "量筒读数误差", "天平称量误差", "装置或方案导致误差",
        "定量实验误差分析", "多种误差联合分析",
    }:
        _writeback_floor(
            result, "中等题", "T1_error_analysis_floor",
            "教师规则：需要判断测量偏差、装置偏差或误差传递时最低为中等题。",
            {"error_analysis": result["features"]["error_analysis"]},
        )
    if review_candidate is not None and review_candidate["writeback_allowed"]:
        before_review = result["difficulty_level"]
        _writeback_floor(
            result,
            review_candidate["candidate_level"],
            review_candidate["rule"],
            review_candidate["reason"],
            review_candidate["evidence"],
        )
        review_candidate["writeback_applied"] = result["difficulty_level"] != before_review
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
