#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将高中化学500题复核文件拆成盲测输入与独立标签。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


QUESTION_FIELDS = (
    "parent_id",
    "question_id",
    "stem",
    "options",
    "analysis",
    "structure_type",
    "sub_questions",
    "stem_image_url",
    "analysis_image_url",
    "stem_pic_url",
    "analysis_pic_url",
)


def question_projection(row: dict[str, Any]) -> dict[str, Any]:
    """只保留允许发送给模型的题目字段。"""
    return {
        field: row[field]
        for field in QUESTION_FIELDS
        if field in row
    }


def label_projection(row: dict[str, Any]) -> dict[str, Any]:
    """保留标签与复核审计字段，不复制题目正文。"""
    question_fields = set(QUESTION_FIELDS) - {"question_id"}
    return {
        key: value
        for key, value in row.items()
        if key not in question_fields
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row.get("question_id") or "")
            if not question_id:
                raise ValueError(f"{path} 第{line_number}行缺少 question_id")
            if question_id in seen:
                raise ValueError(f"question_id 重复：{question_id}")
            if row.get("reviewed_difficulty_level") not in {
                "难度1档", "难度2档", "难度3档", "难度4档", "难度5档"
            }:
                raise ValueError(f"question_id={question_id} 缺少合法复核标签")
            seen.add(question_id)
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--blind-output", required=True)
    parser.add_argument("--labels-output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = read_jsonl(Path(args.source))
    blind = [question_projection(row) for row in rows]
    labels = [label_projection(row) for row in rows]
    write_jsonl(Path(args.blind_output), blind)
    write_jsonl(Path(args.labels_output), labels)
    distribution = Counter(row["reviewed_difficulty_level"] for row in labels)
    print(json.dumps({
        "questions": len(blind),
        "labels": len(labels),
        "label_distribution": dict(distribution),
        "blind_output": args.blind_output,
        "labels_output": args.labels_output,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
