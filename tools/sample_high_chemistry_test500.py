#!/usr/bin/env python3
"""Create a label-blind, five-stratum high-school chemistry test set."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
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
LEVELS = ("1", "2", "3", "4", "5")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def has_text_stem(row: dict[str, Any]) -> bool:
    return bool(str(row.get("stem") or "").strip())


def question_projection(row: dict[str, Any], index: int, seed: int) -> dict[str, Any]:
    projected = {field: row.get(field) for field in QUESTION_FIELDS}
    projected["test_sample_index"] = index
    projected["sampling_seed"] = seed
    return projected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--per-level", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()
    if args.per_level <= 0:
        raise ValueError("--per-level must be positive")

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_empty_stem: list[dict[str, str]] = []
    for row in read_jsonl(args.input):
        level = str(row.get("difficulty") or "")
        if level not in LEVELS:
            raise ValueError(f"Unexpected source difficulty: {level!r}")
        if not row.get("question_id"):
            raise ValueError("Missing question_id")
        if not has_text_stem(row):
            excluded_empty_stem.append(
                {"question_id": str(row["question_id"]), "source_difficulty": level}
            )
            continue
        candidates[level].append(row)

    selected: list[dict[str, Any]] = []
    selected_ids_by_source_level: dict[str, list[str]] = {}
    for level in LEVELS:
        population = sorted(candidates[level], key=lambda row: str(row["question_id"]))
        if len(population) < args.per_level:
            raise ValueError(f"Source difficulty {level} has only {len(population)} usable rows")
        rng = random.Random(f"{args.seed}:{level}")
        picked = rng.sample(population, args.per_level)
        selected.extend(picked)
        selected_ids_by_source_level[level] = [str(row["question_id"]) for row in picked]

    selected.sort(key=lambda row: str(row["question_id"]))
    if len({str(row["question_id"]) for row in selected}) != len(selected):
        raise ValueError("Duplicate question_id after sampling")

    projected = [
        question_projection(row, index, args.seed)
        for index, row in enumerate(selected, start=1)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in projected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    audit = {
        "input": str(args.input),
        "output": str(args.output),
        "sampling_seed": args.seed,
        "per_source_difficulty": args.per_level,
        "sample_count": len(projected),
        "source_level_counts_before_filter": {
            level: len(candidates[level]) + sum(
                item["source_difficulty"] == level for item in excluded_empty_stem
            )
            for level in LEVELS
        },
        "usable_source_level_counts": {level: len(candidates[level]) for level in LEVELS},
        "excluded_empty_stem_question_ids": excluded_empty_stem,
        "selected_question_ids_by_source_difficulty": selected_ids_by_source_level,
        "output_structure_type_distribution": dict(
            Counter(str(row.get("structure_type") or "") for row in projected)
        ),
        "output_does_not_include_source_difficulty": all(
            "difficulty" not in row and "percent_correct" not in row and "answered_count" not in row
            for row in projected
        ),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
