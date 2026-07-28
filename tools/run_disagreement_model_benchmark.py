# -*- coding: utf-8 -*-
"""运行多模型匿名相邻边界单裁判基准。

每个模型仅判断三次 Lite 结果不一致的120道题；三次一致的377道题直接保留。
模型顺序执行以避免同时压垮网关，单个模型预检失败不会中断后续模型。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = [
    "doubao-seed-2.0-mini",
    "doubao-seed-2.0-lite",
    "doubao-seed-2.0-pro",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "glm-5.1",
    "qwen3-max",
    "glm-5.2",
    "doubao-seed-2.1-pro",
    "doubao-seed-2.1-turbo",
]
DEFAULT_RUNS = [
    "outputs/model_runs/lite_physics_teacherfeedback_497_run1.jsonl",
    "outputs/model_runs/lite_physics_teacherfeedback_497_run2.jsonl",
    "outputs/model_runs/lite_physics_teacherfeedback_497_run3.jsonl",
]
DEFAULT_LABELS = "data/labeled/physics_review_500_default.jsonl"
DEFAULT_REFERENCES = (
    "outputs/visualizations/lite_physics_v2_random1000_visualized500.jsonl"
)
DEFAULT_PROMPT = "prompts/初中物理匿名边界裁判提示词.txt"
DEFAULT_OUTPUT_DIR = "outputs/model_benchmarks/anonymous_judge_497_20260728"


def model_slug(model: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", model.strip()).strip("-")
    if not value:
        raise ValueError(f"无法生成模型目录名：{model!r}")
    return value


def configured_temperature(model: str) -> Optional[str]:
    # Mini 是当前候选中唯一明确需要温度0的模型；Lite 由评级脚本固定为1。
    if model == "doubao-seed-2.0-mini":
        return "0"
    return None


def build_command(
    args: argparse.Namespace,
    model: str,
    run_dir: Path,
) -> List[str]:
    command = [
        sys.executable,
        str(ROOT / "src" / "physics_difficulty_disagreement_judge.py"),
    ]
    for path in args.run:
        command.extend(["--run", str(ROOT / path)])
    command.extend(
        [
            "--labels",
            str(ROOT / args.labels),
            "--reference-questions",
            str(ROOT / args.reference_questions),
            "-o",
            str(run_dir / "results.jsonl"),
            "-e",
            str(run_dir / "errors.jsonl"),
            "-p",
            str(ROOT / args.prompt),
            "--strategy",
            "balanced",
            "--judge-model",
            model,
            "--second-judge-model",
            model,
            "--api-mode",
            "auto",
            "--disagreements-only",
            "--fewshot-per-level",
            str(args.fewshot_per_level),
            "-c",
            str(args.concurrency),
            "-t",
            str(args.timeout),
            "-r",
            str(args.retries),
        ]
    )
    temperature = configured_temperature(model)
    if temperature is not None:
        command.extend(["--temperature", temperature])
    return command


def tail_text(path: Path, limit: int = 2500) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def load_success(model: str, run_dir: Path, elapsed: float) -> Dict[str, Any]:
    summary_path = run_dir / "results.jsonl.summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    usage = summary.get("usage") or {}
    return {
        "model": model,
        "status": "success",
        "final_accuracy": summary.get("final_accuracy"),
        "final_correct": summary.get("final_correct"),
        "judge_accuracy": summary.get("judge_accuracy_on_valid_disagreements"),
        "judge_valid_count": summary.get("judge_valid_count"),
        "net_improvement": summary.get("net_improvement"),
        "improved": summary.get("improved_vs_majority"),
        "worsened": summary.get("worsened_vs_majority"),
        "errors": summary.get("errors"),
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
        "meets_90_percent": bool((summary.get("final_accuracy") or 0) >= 0.9),
        "summary_path": str(summary_path.relative_to(ROOT)),
        "elapsed_seconds": round(elapsed, 3),
    }


def load_existing(model: str, run_dir: Path) -> Optional[Dict[str, Any]]:
    summary_path = run_dir / "results.jsonl.summary.json"
    if not summary_path.exists():
        return None
    value = load_success(model, run_dir, 0.0)
    value["status"] = "reused"
    return value


def write_reports(output_dir: Path, results: Sequence[Dict[str, Any]]) -> None:
    ordered = list(results)
    payload = {
        "experiment": "anonymous-single-judge-model-benchmark",
        "baseline_majority_accuracy": 0.8491,
        "target_accuracy": 0.9,
        "required_disagreement_accuracy": 0.8583,
        "results": ordered,
    }
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    fields = [
        "model",
        "status",
        "final_accuracy",
        "judge_accuracy",
        "final_correct",
        "net_improvement",
        "improved",
        "worsened",
        "judge_valid_count",
        "errors",
        "total_tokens",
        "elapsed_seconds",
        "meets_90_percent",
        "summary_path",
        "error",
    ]
    with (output_dir / "benchmark_results.csv").open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)

    lines = [
        "# 多模型匿名边界单裁判基准",
        "",
        "- 三次 Lite 多数票基线：84.91%",
        "- 目标：90.00%",
        "- 分歧题裁判目标准确率：85.83%",
        "",
        "| 模型 | 状态 | 最终准确率 | 分歧题准确率 | 净增正确 | Token |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for result in ordered:
        final = result.get("final_accuracy")
        judge = result.get("judge_accuracy")
        lines.append(
            "| {model} | {status} | {final} | {judge} | {net} | {tokens} |".format(
                model=result["model"],
                status=result["status"],
                final=f"{100 * final:.2f}%" if isinstance(final, (int, float)) else "—",
                judge=f"{100 * judge:.2f}%" if isinstance(judge, (int, float)) else "—",
                net=result.get("net_improvement", "—"),
                tokens=result.get("total_tokens", "—"),
            )
        )
    (output_dir / "benchmark_results.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="运行匿名边界单裁判多模型基准")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--reference-questions", default=DEFAULT_REFERENCES)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("-o", "--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--fewshot-per-level", type=int, default=3)
    parser.add_argument("-c", "--concurrency", type=int, default=10)
    parser.add_argument("-t", "--timeout", type=int, default=180)
    parser.add_argument("-r", "--retries", type=int, default=3)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.model = [value.strip() for value in (args.model or DEFAULT_MODELS) if value.strip()]
    args.run = args.run or list(DEFAULT_RUNS)
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, Any]] = []

    print(
        "本实验用于检验：哪个可用模型能在同一批120道相邻分歧题上，"
        "显著超过三次Lite多数票并使497题总体准确率接近90%。"
    )
    print("模型顺序:", json.dumps(args.model, ensure_ascii=False))

    for index, model in enumerate(args.model, 1):
        run_dir = output_dir / model_slug(model)
        run_dir.mkdir(parents=True, exist_ok=True)
        existing = load_existing(model, run_dir)
        if existing:
            print(f"[{index}/{len(args.model)}] {model}: 已有完整结果，直接复用")
            results.append(existing)
            write_reports(output_dir, results)
            continue

        command = build_command(args, model, run_dir)
        command_text = " ".join(command)
        (run_dir / "command.txt").write_text(command_text + "\n", encoding="utf-8")
        print(f"[{index}/{len(args.model)}] {model}: 开始单题预检并运行")
        print(command_text)
        if args.dry_run:
            results.append({"model": model, "status": "dry_run"})
            continue

        started = time.time()
        log_path = run_dir / "run.log"
        env = os.environ.copy()
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n===== {model} START =====\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.write(f"===== {model} EXIT {completed.returncode} =====\n")
        elapsed = time.time() - started
        summary_path = run_dir / "results.jsonl.summary.json"
        if completed.returncode == 0 and summary_path.exists():
            result = load_success(model, run_dir, elapsed)
            print(
                f"{model}: 最终准确率={100 * result['final_accuracy']:.2f}%，"
                f"分歧题准确率={100 * result['judge_accuracy']:.2f}%，"
                f"净增={result['net_improvement']}"
            )
        else:
            result = {
                "model": model,
                "status": "failed",
                "returncode": completed.returncode,
                "elapsed_seconds": round(elapsed, 3),
                "error": tail_text(log_path),
            }
            print(f"{model}: 失败，已记录日志并继续下一个模型")
        results.append(result)
        write_reports(output_dir, results)
        if result["status"] == "failed" and args.stop_on_error:
            raise SystemExit(f"{model} 运行失败")

    if args.dry_run:
        write_reports(output_dir, results)
    print("汇总:", output_dir / "benchmark_results.md")


if __name__ == "__main__":
    main()
