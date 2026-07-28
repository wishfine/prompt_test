# -*- coding: utf-8 -*-
"""绘制多模型匿名边界裁判基准图。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description="绘制多模型匿名裁判基准")
    parser.add_argument(
        "-i",
        "--input",
        default=(
            "outputs/model_benchmarks/anonymous_judge_497_20260728/"
            "benchmark_results.json"
        ),
    )
    parser.add_argument("-o", "--output-dir", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.input).read_text(encoding="utf-8"))
    successful = [
        value
        for value in report.get("results") or []
        if value.get("status") in {"success", "reused"}
        and isinstance(value.get("final_accuracy"), (int, float))
    ]
    if not successful:
        raise ValueError("没有可绘制的成功模型结果")
    successful.sort(key=lambda value: value["final_accuracy"], reverse=True)
    names = [value["model"] for value in successful]
    overall = [100 * value["final_accuracy"] for value in successful]
    disagreement = [100 * value["judge_accuracy"] for value in successful]
    tokens = [int(value.get("total_tokens") or 0) for value in successful]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(12, max(5.5, len(names) * 0.55)))
    positions = list(range(len(names)))
    bars = axis.barh(positions, overall, color="#4C72B0")
    axis.axvline(84.91, color="#777777", linestyle="--", label="Lite多数票 84.91%")
    axis.axvline(90, color="#C44E52", linestyle="--", label="目标 90%")
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlim(75, 95)
    axis.set_xlabel("497题最终准确率（%）")
    axis.set_title("Figure 1. 多模型匿名单裁判最终准确率")
    axis.legend()
    for bar, value in zip(bars, overall):
        axis.text(value + 0.15, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center")
    figure.tight_layout()
    figure.savefig(output_dir / "Figure_1_model_accuracy.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, max(5.5, len(names) * 0.55)))
    bars = axis.barh(positions, disagreement, color="#55A868")
    axis.axvline(64.17, color="#777777", linestyle="--", label="多数票分歧题 64.17%")
    axis.axvline(85.83, color="#C44E52", linestyle="--", label="达到总体90%所需 85.83%")
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlim(0, 100)
    axis.set_xlabel("120道分歧题裁判准确率（%）")
    axis.set_title("Figure 2. 分歧题准确率与 Token 消耗")
    axis.legend()
    for bar, value, token in zip(bars, disagreement, tokens):
        axis.text(
            value + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}% / {token:,} tokens",
            va="center",
            fontsize=9,
        )
    figure.tight_layout()
    figure.savefig(output_dir / "Figure_2_disagreement_accuracy.png", dpi=180)
    plt.close(figure)
    print(f"图表已输出：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
