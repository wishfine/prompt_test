#!/usr/bin/env python3
"""Build compact, auditable packets for the 1066-question label re-review."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADJ = ROOT / "tmp/adjudication_workbook.json"
AUDIT = ROOT / "tmp/final_candidate_audit.json"
OUT = ROOT / "tmp/gpt56_label_rereview_packets.jsonl"


def rows_from_sheet(payload: dict, sheet: str, header_row: int) -> list[dict]:
    values = payload[sheet]
    header = values[header_row]
    return [dict(zip(header, row)) for row in values[header_row + 1 :] if any(v is not None for v in row)]


def clean(value: object, limit: int = 1200) -> str:
    text = str(value or "").replace("\r", " ").replace("\n", " ").strip()
    return text[:limit]


def main() -> None:
    adjudication = json.loads(ADJ.read_text(encoding="utf-8"))
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    all_rows = rows_from_sheet(adjudication, "1066题裁定", 0)
    union_rows = rows_from_sheet(audit, "错题并集复核_424", 1)
    union = {str(row["question_id"]): row for row in union_rows}
    revision = {
        str(row["question_id"]): row
        for row in rows_from_sheet(audit, "GPT标签修订", 1)
    }
    boundary = {
        str(row["question_id"]): row
        for row in rows_from_sheet(audit, "相邻边界均可", 1)
    }

    packets: list[dict] = []
    for row in all_rows:
        qid = str(row["题目ID"])
        audit_row = union.get(qid, {})
        reasons: list[str] = []
        if qid in revision:
            reasons.append("已有人工复核建议改标")
        if qid in boundary:
            reasons.append("已有人工复核判为相邻双档边界")
        if row["教师标准档"] != row["最终裁定档"]:
            reasons.append("教师标签与GPT裁定不同")
        if row["裁定置信度"] != "高":
            reasons.append("GPT裁定置信度非高")
        if row["我的盲评"] != row["最终裁定档"]:
            reasons.append("初次独立盲评与GPT裁定不同")
        if row["教师分析疑点"]:
            reasons.append("教师分析存在疑点标记")
        models = [row["冻结版模型"], row["广度版模型"], row["共同机制版模型"]]
        if len(set(models)) == 1 and models[0] != row["最终裁定档"]:
            reasons.append("早期三模型一致反对GPT裁定")

        priority = "常规一致性复核"
        if qid in revision:
            priority = "P0_明确改标复核"
        elif qid in boundary:
            priority = "P1_边界复核"
        elif reasons:
            priority = "P1_冲突复核"

        packets.append(
            {
                "question_id": qid,
                "priority": priority,
                "risk_reasons": reasons,
                "gpt_label": row["最终裁定档"],
                "gpt_confidence": row["裁定置信度"],
                "teacher_label": row["教师标准档"],
                "blind_label": row["我的盲评"],
                "early_models": models,
                "existing_manual_label": audit_row.get("人工复核等级", ""),
                "existing_acceptable_levels": audit_row.get("可接受等级", ""),
                "existing_review_conclusion": audit_row.get("GPT复核结论", ""),
                "existing_manual_reason": audit_row.get("人工复核依据", ""),
                "stem": clean(row["题干"], 1600),
                "options": clean(row["选项"], 1200),
                "official_analysis": clean(row["官方解析"], 2600),
                "teacher_reason": clean(row["教师详细分析"], 1400),
                "gpt_reason": clean(row["具体裁定原因"], 1400),
                "blind_reason": clean(row["盲评依据"], 800),
                "structure": {
                    "step": row["结构_步骤"],
                    "type": row["结构_题型"],
                    "state": row["结构_状态"],
                    "constraint": row["结构_约束"],
                    "experiment": row["结构_实验"],
                    "graph": row["结构_图表"],
                    "reasoning": row["结构_推理"],
                },
            }
        )

    priority_order = {
        "P0_明确改标复核": 0,
        "P1_边界复核": 1,
        "P1_冲突复核": 2,
        "常规一致性复核": 3,
    }
    packets.sort(key=lambda item: (priority_order[item["priority"]], item["question_id"]))
    with OUT.open("w", encoding="utf-8") as handle:
        for item in packets:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUT), "count": len(packets)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
