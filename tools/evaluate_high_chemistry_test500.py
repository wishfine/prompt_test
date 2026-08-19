#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评测高中化学500题两阶段结果，并拆分第一阶段与最终指标。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate_high_physics_test500 import (
    accuracy_scale_diagnostics,
    evaluate,
    read_by_id,
    review_diagnostics,
)


def build_report(
    labels: dict[str, dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "final": evaluate(labels, predictions, "final_difficulty_level"),
        "step1": evaluate(labels, predictions, "difficulty_level_step1"),
        "accuracy_scale_diagnostics": accuracy_scale_diagnostics(predictions),
        "review_diagnostics": review_diagnostics(predictions),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_report(
        read_by_id(Path(args.labels)),
        read_by_id(Path(args.predictions)),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
