# -*- coding: utf-8 -*-
"""离线将高中物理V3结果应用V4乘数桶守卫。

该工具不调用模型，只重新计算 ``final_difficulty_level`` 等最终字段。
输出文件可直接作为V4批处理脚本的断点续跑起点，从而只补跑原先缺失题。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from high_physics_pipeline_core import finalize_level  # noqa: E402


def upgrade_record(row: dict[str, Any]) -> dict[str, Any]:
    upgraded = copy.deepcopy(row)
    stage1 = upgraded.get("difficulty_rating_stage1") or {}
    verification = upgraded.get("verification") or {}
    current_level = (
        upgraded.get("difficulty_level_step1")
        or stage1.get("difficulty_level_step1")
    )
    reviewed_features = verification.get(
        "reviewed_high_difficulty_features"
    )
    if not isinstance(reviewed_features, list):
        raise ValueError(
            f"question_id={upgraded.get('question_id')} 缺少复核高难特征"
        )
    original_high_count = stage1.get("high_difficulty_feature_count")
    if not isinstance(original_high_count, int):
        raise ValueError(
            f"question_id={upgraded.get('question_id')} 缺少第一阶段高难特征数"
        )
    final = finalize_level(
        current_level=current_level,
        reasonableness=verification.get("rating_reasonableness", ""),
        model_suggested_level=verification.get(
            "adjusted_difficulty_level"
        ),
        multiplier_reasonableness=verification.get(
            "multiplier_reasonableness",
            "不合理",
        ),
        input_sufficiency=(
            upgraded.get("input_quality") or {}
        ).get("input_sufficiency", "信息不足"),
        original_high_count=original_high_count,
        reviewed_high_count=len(reviewed_features),
    )
    upgraded["upgraded_from_pipeline_version"] = upgraded.get(
        "pipeline_version"
    )
    upgraded["pipeline_version"] = "high_physics_two_stage_v4"
    upgraded["model_suggested_level"] = final.model_suggested_level
    upgraded["final_difficulty_level"] = final.final_level
    upgraded["final_adjustment"] = final.adjustment_desc
    upgraded["needs_manual_review"] = final.needs_manual_review
    return upgraded


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path} 第{line_number}行不是对象")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="V3成功结果JSONL")
    parser.add_argument("--output", required=True, help="V4预填充结果JSONL")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input与output不能是同一个文件")
    rows = read_jsonl(input_path)
    upgraded = [upgrade_record(row) for row in rows]
    write_jsonl(output_path, upgraded)
    changed = sum(
        before.get("final_difficulty_level")
        != after.get("final_difficulty_level")
        for before, after in zip(rows, upgraded)
    )
    print(
        f"离线升级完成：{len(upgraded)}道，最终档位变化{changed}道，"
        f"输出={output_path.resolve()}"
    )


if __name__ == "__main__":
    main()
