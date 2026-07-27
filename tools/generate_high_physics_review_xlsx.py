# -*- coding: utf-8 -*-
"""将高中物理两阶段评级 JSONL 生成便于筛选复核的 Excel 工作簿。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


GROUP_COLORS = {
    "标识": "475569",
    "第一阶段": "2563EB",
    "第二阶段": "7C3AED",
    "最终结果": "EA580C",
    "最终分数": "0F766E",
}

STAGE1_FIELD_NAMES = {
    "original_predicted_accuracy": "原始预测正确率",
    "predicted_accuracy": "乘数后预测正确率",
    "multiplier_applied": "应用乘数",
    "high_difficulty_feature_count": "高难特征数量",
    "active_feature_count": "活跃特征数量",
    "difficulty_level_step1": "映射档位",
}

STAGE2_FIELD_NAMES = {
    "reviewed_predicted_accuracy": "复核后预测正确率",
    "rating_reasonableness": "评级合理性",
    "multiplier_reasonableness": "乘数合理性",
    "adjusted_difficulty_level": "建议调整档位",
}

STAGE1_PRIORITY_FIELDS = [
    "original_predicted_accuracy",
    "multiplier_applied",
    "predicted_accuracy",
    "difficulty_level_step1",
    "high_difficulty_feature_count",
    "active_feature_count",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("question_id") is None:
                raise ValueError(f"{path} 第{line_number}行缺少 question_id")
            rows.append(row)
    return rows


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """递归展开对象；数组保留为JSON文本，避免列数无限扩张。"""
    if not isinstance(value, dict):
        return {prefix: value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        child_prefix = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            flattened.update(flatten(child, child_prefix))
        else:
            flattened[child_prefix] = child
    return flattened


def excel_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, str):
        # Excel 单个单元格最多32767字符。
        return value[:32760]
    return value


def ordered_union(
    flattened_rows: list[dict[str, Any]],
) -> list[str]:
    seen: set[str] = set()
    fields: list[str] = []
    for row in flattened_rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    return fields


def prioritize_fields(fields: list[str], priority: list[str]) -> list[str]:
    priority_fields = [field for field in priority if field in fields]
    return priority_fields + [
        field for field in fields if field not in priority_fields
    ]


def display_field_name(stage: str, field: str) -> str:
    translations = (
        STAGE1_FIELD_NAMES if stage == "第一阶段" else STAGE2_FIELD_NAMES
    )
    return f"{stage}.{translations.get(field, field)}"


def style_header(ws, groups: list[str]) -> None:
    for column, group in enumerate(groups, start=1):
        cell = ws.cell(row=1, column=column)
        cell.fill = PatternFill("solid", fgColor=GROUP_COLORS[group])
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(
            horizontal="center",
            vertical="center",
            wrap_text=True,
        )
    ws.row_dimensions[1].height = 42
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = ws.dimensions
    ws.sheet_view.showGridLines = False


def width_for_header(header: str) -> float:
    if header == "题目ID":
        return 22
    if any(token in header for token in (
        "reason",
        "evidence",
        "analysis",
        "difficulty_source",
        "stage2_error",
    )):
        return 45
    if any(token in header for token in (
        "features",
        "methods",
        "knowledge",
        "corrections",
    )):
        return 26
    return 18


def create_detail_sheet(
    wb: Workbook,
    results: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    ws = wb.active
    ws.title = "评级明细"

    stage1_rows = [
        flatten(row.get("difficulty_rating_stage1") or {})
        for row in results
    ]
    stage2_rows = [
        flatten(row.get("verification") or {})
        for row in results
    ]
    stage1_fields = prioritize_fields(
        ordered_union(stage1_rows),
        STAGE1_PRIORITY_FIELDS,
    )
    stage2_fields = ordered_union(stage2_rows)

    final_headers = [
        "模型建议档位",
        "最终档位",
        "最终调整说明",
        "是否需要人工复核",
        "第二阶段状态",
        "Pipeline版本",
    ]
    headers = (
        ["题目ID"]
        + [display_field_name("第一阶段", field) for field in stage1_fields]
        + [display_field_name("第二阶段", field) for field in stage2_fields]
        + final_headers
        + ["最终分数"]
    )
    groups = (
        ["标识"]
        + ["第一阶段"] * len(stage1_fields)
        + ["第二阶段"] * len(stage2_fields)
        + ["最终结果"] * len(final_headers)
        + ["最终分数"]
    )
    ws.append(headers)

    for result, stage1_flat, stage2_flat in zip(
        results,
        stage1_rows,
        stage2_rows,
    ):
        question_id = str(result["question_id"])
        final_level = result.get("final_difficulty_level", "")
        verification = result.get("verification") or {}
        final_score = verification.get("reviewed_predicted_accuracy")
        if final_score is None:
            final_score = (
                result.get("difficulty_rating_stage1") or {}
            ).get("predicted_accuracy")
        final_values = [
            result.get("model_suggested_level", ""),
            final_level,
            result.get("final_adjustment", ""),
            "是" if result.get("needs_manual_review") is True else "否",
            result.get("verification_status", "success"),
            result.get("pipeline_version", ""),
        ]
        ws.append(
            [question_id]
            + [excel_value(stage1_flat.get(field)) for field in stage1_fields]
            + [excel_value(stage2_flat.get(field)) for field in stage2_fields]
            + final_values
            + [final_score if final_score is not None else ""]
        )

    style_header(ws, groups)
    for column, header in enumerate(headers, start=1):
        ws.column_dimensions[get_column_letter(column)].width = (
            width_for_header(header)
        )
    for row in ws.iter_rows(min_row=2):
        row[0].number_format = "@"
        for cell in row:
            cell.alignment = Alignment(
                vertical="top",
                wrap_text=True,
            )
    score_column = len(headers)
    ws.conditional_formatting.add(
        f"{get_column_letter(score_column)}2:"
        f"{get_column_letter(score_column)}{ws.max_row}",
        ColorScaleRule(
            start_type="num",
            start_value=0,
            start_color="FCA5A5",
            mid_type="num",
            mid_value=58,
            mid_color="FDE68A",
            end_type="num",
            end_value=100,
            end_color="86EFAC",
        ),
    )
    return stage1_fields, stage2_fields


def create_question_sheet(
    wb: Workbook,
    results: list[dict[str, Any]],
) -> None:
    ws = wb.create_sheet("题目信息")
    headers = [
        "题目ID",
        "题干",
        "选项",
        "解析",
        "题干图片",
        "解析图片",
        "输入充分性",
    ]
    ws.append(headers)
    for result in results:
        question_id = str(result["question_id"])
        quality = result.get("input_quality") or {}
        ws.append([
            question_id,
            excel_value(result.get("stem")),
            excel_value(result.get("options")),
            excel_value(result.get("analysis")),
            excel_value(result.get("stem_image_url")),
            excel_value(result.get("analysis_image_url")),
            quality.get("input_sufficiency", ""),
        ])
    style_header(ws, ["标识"] + ["最终结果"] * (len(headers) - 1))
    widths = [22, 60, 45, 70, 40, 40, 16]
    for index, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(index)].width = width
    for row in ws.iter_rows(min_row=2):
        row[0].number_format = "@"
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

def create_field_guide(
    wb: Workbook,
    stage1_fields: list[str],
    stage2_fields: list[str],
) -> None:
    ws = wb.create_sheet("字段说明")
    ws.append(["字段组", "Excel列名", "说明"])
    for field in stage1_fields:
        ws.append([
            "第一阶段",
            display_field_name("第一阶段", field),
            (
                "模型原始预测正确率（乘数前）"
                if field == "original_predicted_accuracy"
                else "程序应用高难特征乘数后的第一阶段正确率"
                if field == "predicted_accuracy"
                else "第一阶段模型输出或程序派生字段"
            ),
        ])
    for field in stage2_fields:
        ws.append([
            "第二阶段",
            display_field_name("第二阶段", field),
            "第二阶段结构审计输出或程序复核字段",
        ])
    ws.append([
        "最终分数",
        "最终分数",
        (
            "第二阶段复核后的乘数后正确率；"
            "第二阶段失败时回退第一阶段乘数后正确率"
        ),
    ])
    style_header(ws, ["标识", "标识", "标识"])
    ws.column_dimensions["A"].width = 16
    ws.column_dimensions["B"].width = 55
    ws.column_dimensions["C"].width = 65
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = read_jsonl(Path(args.input))
    wb = Workbook()
    stage1_fields, stage2_fields = create_detail_sheet(
        wb,
        results,
    )
    create_question_sheet(wb, results)
    create_field_guide(wb, stage1_fields, stage2_fields)
    wb.calculation.fullCalcOnLoad = True
    wb.calculation.forceFullCalc = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    print(json.dumps({
        "input": args.input,
        "rows": len(results),
        "stage1_columns": len(stage1_fields),
        "stage2_columns": len(stage2_fields),
        "output": str(output),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
