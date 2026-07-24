# -*- coding: utf-8 -*-
"""把GPT-5.6复核文件拆成真正盲测输入和独立标签文件。"""

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
)

LABEL_FIELDS = (
    "question_id",
    "prior_test_sample_index",
    "sampling_seed",
    "source_difficulty_untrusted",
    "prior_label_stage1",
    "prior_label_stage1_value",
    "prior_label_stage1_method",
    "prior_label_stage1_confidence",
    "prior_label_stage1_reason",
    "review_method",
    "review_decision",
    "review_confidence",
    "review_reason",
    "reviewed_difficulty_level",
    "reviewed_difficulty_value",
    "source_label_gap",
    "needs_teacher_review",
    "teacher_review_priority",
    "data_quality_flags",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} 第 {line_number} 行不是对象")
            rows.append(value)
    return rows


def question_projection(row: dict[str, Any]) -> dict[str, Any]:
    return {
        field: row[field]
        for field in QUESTION_FIELDS
        if field in row
    }


def verify_matching_source(
    reviewed_rows: list[dict[str, Any]],
    blind_source_rows: list[dict[str, Any]],
) -> None:
    reviewed = {
        str(row.get("question_id")): question_projection(row)
        for row in reviewed_rows
    }
    blind = {
        str(row.get("question_id")): question_projection(row)
        for row in blind_source_rows
    }
    if reviewed.keys() != blind.keys():
        raise ValueError("两份源文件的 question_id 集合不一致")
    mismatches = [qid for qid in reviewed if reviewed[qid] != blind[qid]]
    if mismatches:
        raise ValueError(f"两份源文件有 {len(mismatches)} 道题内容不一致")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed", required=True)
    parser.add_argument("--blind-source")
    parser.add_argument("--output-blind", required=True)
    parser.add_argument("--output-labels", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    reviewed_rows = read_jsonl(Path(args.reviewed))
    if args.blind_source:
        verify_matching_source(
            reviewed_rows,
            read_jsonl(Path(args.blind_source)),
        )

    ids = [str(row.get("question_id")) for row in reviewed_rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("question_id 缺失或重复")
    if any(not row.get("reviewed_difficulty_level") for row in reviewed_rows):
        raise ValueError("存在缺失 reviewed_difficulty_level 的记录")

    blind_rows = [question_projection(row) for row in reviewed_rows]
    label_rows = [
        {
            field: row.get(field)
            for field in LABEL_FIELDS
            if field in row
        }
        for row in reviewed_rows
    ]
    write_jsonl(Path(args.output_blind), blind_rows)
    write_jsonl(Path(args.output_labels), label_rows)

    distribution = Counter(
        row["reviewed_difficulty_level"] for row in reviewed_rows
    )
    print(
        json.dumps(
            {
                "questions": len(blind_rows),
                "labels": len(label_rows),
                "reviewed_distribution": dict(distribution),
                "output_blind": str(Path(args.output_blind).resolve()),
                "output_labels": str(Path(args.output_labels).resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
