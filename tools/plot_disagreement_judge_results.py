# -*- coding: utf-8 -*-
"""绘制匿名分歧裁判实验结果。

输入为 physics_difficulty_disagreement_judge.py 生成的 *.summary.json。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]


def load_report(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentage(value: Any) -> float:
    return 100 * float(value or 0)


def annotate_bars(axis: Any, bars: Any, suffix: str = "%") -> None:
    for bar in bars:
        value = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.5,
            f"{value:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_overall(reports: List[Dict[str, Any]], names: List[str], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.8))
    width = 0.24
    positions = list(range(len(reports)))
    majority = [percentage(report["majority_accuracy"]) for report in reports]
    final = [percentage(report["final_accuracy"]) for report in reports]
    oracle = [percentage(report["candidate_oracle_accuracy"]) for report in reports]

    bars1 = axis.bar([x - width for x in positions], majority, width, label="三次多数票")
    bars2 = axis.bar(positions, final, width, label="双裁判最终")
    bars3 = axis.bar([x + width for x in positions], oracle, width, label="候选理论上限")
    annotate_bars(axis, bars1)
    annotate_bars(axis, bars2)
    annotate_bars(axis, bars3)
    axis.axhline(90, color="#C44E52", linestyle="--", linewidth=1.2, label="90%目标")
    axis.set_xticks(positions, names)
    axis.set_ylim(75, 100)
    axis.set_ylabel("准确率（%）")
    axis.set_title("匿名双裁判：多数票、最终结果与候选上限")
    axis.legend(loc="lower right")
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_per_level(report: Dict[str, Any], name: str, output: Path) -> None:
    metrics = report.get("per_level_metrics") or {}
    precision = [percentage((metrics.get(level) or {}).get("precision")) for level in LEVELS]
    recall = [percentage((metrics.get(level) or {}).get("recall")) for level in LEVELS]
    f1 = [percentage((metrics.get(level) or {}).get("f1")) for level in LEVELS]

    figure, axis = plt.subplots(figsize=(11, 5.8))
    width = 0.24
    positions = list(range(len(LEVELS)))
    axis.bar([x - width for x in positions], precision, width, label="Precision")
    axis.bar(positions, recall, width, label="Recall")
    axis.bar([x + width for x in positions], f1, width, label="F1")
    axis.set_xticks(positions, LEVELS)
    axis.set_ylim(0, 105)
    axis.set_ylabel("百分比（%）")
    axis.set_title(f"{name}：各档 Precision / Recall / F1")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制匿名分歧裁判实验汇总图")
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--name", action="append", default=[])
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()

    reports = [load_report(path) for path in args.summary]
    names = list(args.name)
    if names and len(names) != len(reports):
        raise ValueError("--name 数量必须与 --summary 相同")
    if not names:
        names = [Path(path).stem.replace(".summary", "") for path in args.summary]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_overall(reports, names, output_dir / "Figure_1_overall_accuracy.png")
    for index, (report, name) in enumerate(zip(reports, names), 1):
        plot_per_level(
            report,
            name,
            output_dir / f"Figure_{index + 1}_per_level_metrics.png",
        )
    print(f"图表已输出到：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
