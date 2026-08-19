#!/usr/bin/env python3
"""校验诗词基础下限，并按最终档位生成五份FXZ化学验收页。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chemistry_postprocess_fxz import poetry_chemistry_floor_signal
from src.sample_and_generate_chemistry_html_zyl import (
    LEVEL_MAP,
    generate_html_file,
)


LEVEL_NAMES = list(LEVEL_MAP)


class PoetryFloorViolation(ValueError):
    """仍有诗词/古文化学判断题被判为送分题。"""


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number} 不是合法JSON: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 必须是JSON对象")
            rows.append(row)
    return rows


def final_level(row: Dict[str, Any]) -> str:
    rating = row.get("difficulty_rating") or {}
    return str(rating.get("difficulty_level") or "").strip()


def poetry_floor_violation_ids(
    rows: Iterable[Dict[str, Any]],
) -> List[str]:
    return [
        str(row.get("question_id", ""))
        for row in rows
        if poetry_chemistry_floor_signal(row)
        and final_level(row) == "送分题"
    ]


def generate_split_review_files(
    rows: List[Dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    expected_count: int,
    release_label: str,
) -> Dict[int, int]:
    """门禁通过后，按五档写JSONL并调用ZYL逻辑生成HTML。"""
    if len(rows) != expected_count:
        raise ValueError(
            f"输入应为{expected_count}题，实际{len(rows)}题"
        )
    question_ids = [str(row.get("question_id", "")) for row in rows]
    if not all(question_ids) or len(set(question_ids)) != len(question_ids):
        raise ValueError("question_id存在空值或重复")

    violations = poetry_floor_violation_ids(rows)
    if violations:
        raise PoetryFloorViolation(
            "仍有诗词/古文化学判断题被判为送分题，拒绝生成可视化："
            + "、".join(violations)
        )

    grouped: Dict[str, List[Dict[str, Any]]] = {
        level_name: [] for level_name in LEVEL_NAMES
    }
    for row in rows:
        level_name = final_level(row)
        if level_name not in grouped:
            raise ValueError(
                f"ID={row.get('question_id')} 最终档位非法: {level_name!r}"
            )
        grouped[level_name].append(row)

    empty_levels = [name for name, values in grouped.items() if not values]
    if empty_levels:
        raise ValueError(
            "以下档位没有题目，无法生成五份非空页面："
            + "、".join(empty_levels)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[int, int] = {}
    for level_name, level_num in LEVEL_MAP.items():
        level_rows = grouped[level_name]
        counts[level_num] = len(level_rows)
        jsonl_path = output_dir / f"{prefix}_{level_num}.jsonl"
        html_path = output_dir / f"{prefix}_{level_num}.html"
        jsonl_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n"
                for row in level_rows
            ),
            encoding="utf-8",
        )
        generate_html_file(
            {level_num: level_rows},
            str(html_path),
            review_scope=str(len(level_rows)),
            release_label=release_label,
        )
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="诗词基础下限门禁通过后，按难度生成五份FXZ化学验收页"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output-dir", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--expected-count", type=int, default=1000)
    parser.add_argument("--release-label", default="fxz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        counts = generate_split_review_files(
            load_jsonl(Path(args.input)),
            output_dir=Path(args.output_dir),
            prefix=args.prefix,
            expected_count=args.expected_count,
            release_label=args.release_label,
        )
    except (ValueError, PoetryFloorViolation) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "门禁通过，已按档位生成："
        + " / ".join(f"{level}={counts[index]}" for index, level in enumerate(LEVEL_NAMES, 1))
    )


if __name__ == "__main__":
    main()
