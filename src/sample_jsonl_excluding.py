#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 JSONL 题库中排除历史题目后进行可复现的分层抽样。"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} 必须是 JSON 对象")
            rows.append(row)
    return rows


def normalized_id(row: dict[str, Any], id_field: str) -> str:
    value = row.get(id_field)
    if value is None:
        return ""
    return str(value).strip()


def validate_unique_ids(
    rows: list[dict[str, Any]], path: Path, id_field: str
) -> list[str]:
    ids = [normalized_id(row, id_field) for row in rows]
    empty_count = sum(not value for value in ids)
    if empty_count:
        raise ValueError(f"{path} 有 {empty_count} 条记录缺少 {id_field}")
    duplicate_count = len(ids) - len(set(ids))
    if duplicate_count:
        raise ValueError(f"{path} 有 {duplicate_count} 个重复 {id_field}")
    return ids


def allocate_evenly(total: int, keys: list[str]) -> dict[str, int]:
    if not keys:
        return {}
    quotient, remainder = divmod(total, len(keys))
    return {
        key: quotient + (1 if index < remainder else 0)
        for index, key in enumerate(keys)
    }


def sample_rows(
    candidates: list[dict[str, Any]],
    sample_size: int,
    seed: int,
    stratify_field: str | None,
) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
    if sample_size > len(candidates):
        raise ValueError(
            f"排除后仅剩 {len(candidates)} 道题，无法抽取 {sample_size} 道"
        )

    rng = random.Random(seed)
    if not stratify_field:
        sampled = rng.sample(candidates, sample_size)
        return sampled, {"all": len(candidates)}, {"all": len(sampled)}

    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        key = str(row.get(stratify_field, "")).strip()
        if not key:
            raise ValueError(f"存在缺少分层字段 {stratify_field} 的题目")
        pools[key].append(row)

    keys = sorted(pools)
    target = allocate_evenly(sample_size, keys)
    sampled: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []

    for key in keys:
        pool = pools[key][:]
        rng.shuffle(pool)
        take = min(target[key], len(pool))
        sampled.extend(pool[:take])
        remaining.extend(pool[take:])

    missing = sample_size - len(sampled)
    if missing:
        sampled.extend(rng.sample(remaining, missing))

    rng.shuffle(sampled)
    source_distribution = {key: len(pools[key]) for key in keys}
    sampled_distribution = dict(
        sorted(
            Counter(str(row.get(stratify_field, "")).strip() for row in sampled).items()
        )
    )
    return sampled, source_distribution, sampled_distribution


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="排除历史题目后，从 JSONL 题库进行可复现抽样"
    )
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument(
        "-x",
        "--exclude",
        action="append",
        default=[],
        type=Path,
        help="排除清单 JSONL；可重复传入",
    )
    parser.add_argument("-n", "--sample-size", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--id-field", default="question_id")
    parser.add_argument(
        "--stratify-field",
        default="difficulty",
        help="分层字段；传入空字符串表示不分层",
    )
    parser.add_argument(
        "--require-excluded",
        type=int,
        default=None,
        help="要求排除清单中唯一 ID 数量等于该值",
    )
    args = parser.parse_args()

    if args.sample_size <= 0:
        raise ValueError("--sample-size 必须大于 0")

    source_rows = read_jsonl(args.input)
    source_ids = validate_unique_ids(source_rows, args.input, args.id_field)

    excluded_ids: set[str] = set()
    exclude_file_rows: dict[str, int] = {}
    for exclude_path in args.exclude:
        rows = read_jsonl(exclude_path)
        ids = validate_unique_ids(rows, exclude_path, args.id_field)
        excluded_ids.update(ids)
        exclude_file_rows[str(exclude_path)] = len(rows)

    if (
        args.require_excluded is not None
        and len(excluded_ids) != args.require_excluded
    ):
        raise ValueError(
            f"排除清单唯一 ID 数量为 {len(excluded_ids)}，"
            f"预期为 {args.require_excluded}"
        )

    source_id_set = set(source_ids)
    excluded_found = excluded_ids & source_id_set
    candidates = [
        row
        for row in source_rows
        if normalized_id(row, args.id_field) not in excluded_ids
    ]
    sampled, candidate_distribution, sampled_distribution = sample_rows(
        candidates,
        args.sample_size,
        args.seed,
        args.stratify_field or None,
    )

    sampled_ids = {normalized_id(row, args.id_field) for row in sampled}
    overlap = sampled_ids & excluded_ids
    if overlap:
        raise RuntimeError(f"抽样结果与排除清单仍有 {len(overlap)} 道重叠")
    if len(sampled_ids) != args.sample_size:
        raise RuntimeError("抽样结果数量或唯一 ID 数量不符合要求")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in sampled:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest_path = args.output.with_suffix(".manifest.json")
    manifest = {
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "input_rows": len(source_rows),
        "exclude_files": exclude_file_rows,
        "excluded_unique_ids": len(excluded_ids),
        "excluded_ids_found_in_input": len(excluded_found),
        "candidate_rows": len(candidates),
        "sample_size": len(sampled),
        "sample_unique_ids": len(sampled_ids),
        "sample_exclusion_overlap": len(overlap),
        "seed": args.seed,
        "id_field": args.id_field,
        "stratify_field": args.stratify_field or None,
        "candidate_distribution": candidate_distribution,
        "sample_distribution": sampled_distribution,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"抽样完成：{args.output}")
    print(f"审计清单：{manifest_path}")


if __name__ == "__main__":
    main()
