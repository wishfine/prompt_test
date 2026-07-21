#!/usr/bin/env python3
"""生成老师 2026-07-21 复核的 60 题固定评测集。

输入题目只保留模型推理需要的题目信息，明确删除 JSONL 旧 ``difficulty``；
老师复核结果单独保存为 CSV，供 ``tests/teacher_label_regression.py`` 评测。
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW_PATH = ROOT / "data/labeled/复核标注不一致案例.md"
SOURCE_PATH = ROOT / "data/labeled/physics_difficulty_tiku_data_0714_1000.jsonl"
OUTPUT_JSONL = ROOT / "data/samples/physics_teacher_review_20260721_60.jsonl"
OUTPUT_CSV = ROOT / "data/labeled/physics_teacher_review_20260721_60.csv"
OUTPUT_MANIFEST = ROOT / "data/samples/physics_teacher_review_20260721_60.manifest.json"

# ``teacher_label_regression.py`` 的既有 CSV 映射口径。
SYSTEM_TO_TEACHER = {
    "送分题": "容易",
    "基础题": "较易",
    "中等题": "中等",
    "拔高题": "较难",
    "压轴题": "困难",
}
ITEM_PATTERN = re.compile(
    r"(?ms)^\s*(?P<case_no>\d+)\.\s*\n"
    r"题目ID:\s*(?P<question_id>\d+)\s*\n"
    r"教师复核:\s*(?P<level>送分题|基础题|中等题|拔高题|压轴题)\s*\n"
    r"原因:\s*(?P<reason>.*?)(?=^---\s*$|\Z)"
)


def load_review_items() -> list[dict[str, str]]:
    text = REVIEW_PATH.read_text(encoding="utf-8")
    items = [
        {
            "case_no": match.group("case_no"),
            "question_id": match.group("question_id"),
            "level": match.group("level"),
            "reason": match.group("reason").strip(),
        }
        for match in ITEM_PATTERN.finditer(text)
    ]
    if len(items) != 60:
        raise ValueError(f"老师复核文件应有 60 条，实际解析到 {len(items)} 条")
    ids = [item["question_id"] for item in items]
    if len(ids) != len(set(ids)):
        raise ValueError("老师复核文件中 question_id 重复")
    return items


def load_source_questions() -> dict[str, dict]:
    questions: dict[str, dict] = {}
    with SOURCE_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = str(item.get("question_id") or "").strip()
            if not question_id:
                raise ValueError(f"源数据第 {line_number} 行缺少 question_id")
            if question_id in questions:
                raise ValueError(f"源数据 question_id 重复：{question_id}")
            questions[question_id] = item
    return questions


def main() -> None:
    items = load_review_items()
    questions = load_source_questions()
    missing = [item["question_id"] for item in items if item["question_id"] not in questions]
    if missing:
        raise ValueError(f"以下复核 ID 未在源题库匹配：{missing}")

    OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSONL.open("w", encoding="utf-8") as handle:
        for review in items:
            # 评测输入不携带原 difficulty，也不携带人工复核等级或理由。
            clean = {
                key: value
                for key, value in questions[review["question_id"]].items()
                if key != "difficulty"
            }
            handle.write(json.dumps(clean, ensure_ascii=False) + "\n")

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["ID", "难度", "教师复核", "复核序号", "复核原因"],
            lineterminator="\n",
        )
        writer.writeheader()
        for review in items:
            writer.writerow(
                {
                    "ID": review["question_id"],
                    "难度": SYSTEM_TO_TEACHER[review["level"]],
                    "教师复核": review["level"],
                    "复核序号": review["case_no"],
                    "复核原因": review["reason"],
                }
            )

    distribution = Counter(item["level"] for item in items)
    manifest = {
        "name": "physics_teacher_review_20260721_60",
        "review_source": str(REVIEW_PATH.relative_to(ROOT)),
        "question_source": str(SOURCE_PATH.relative_to(ROOT)),
        "question_count": len(items),
        "question_id_type": "string",
        "input_removed_fields": ["difficulty"],
        "teacher_distribution": {
            level: distribution.get(level, 0)
            for level in ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
        },
        "evaluation_csv": str(OUTPUT_CSV.relative_to(ROOT)),
        "inference_jsonl": str(OUTPUT_JSONL.relative_to(ROOT)),
        "note": "这是老师挑选的复核争议题集合，不可用作全题库总体准确率估计。",
    }
    OUTPUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
