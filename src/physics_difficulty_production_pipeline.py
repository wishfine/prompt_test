# -*- coding: utf-8 -*-
"""初中物理难度评级生产批处理入口。

固定使用冻结 Prompt、Doubao Lite、三次独立评级和结构化送分边界校准。
负责版本签名、断点续跑保护、输入输出完整性校验和运行监控摘要。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_SET = set(LEVELS)
REQUIRED_FEATURES = {
    "step_count",
    "formula_count",
    "calculation_complexity",
    "reasoning_chain",
    "problem_structure",
    "additional_structure",
    "information_carrier",
    "reality_question",
    "subquestion_dependency",
    "knowledge_count",
    "knowledge_diff",
    "cross_module",
    "state_count",
    "constraint_count",
    "variable_relation",
    "experiment_requirement",
    "graph_table_requirement",
    "error_risk",
}
REQUIRED_REASONING_FIELDS = {
    "core_basis",
    "hard_point",
    "why_not_lower",
    "why_not_higher",
}
PIPELINE_VERSION = "physics-production-lite-3x-structured-v1"
RUN_COUNT = 3
MODEL_NAME = "doubao-seed-2.0-lite"
RATING_PROFILE = "gpt56_hybrid"
PRODUCTION_ENV = {
    "RATING_PROFILE": RATING_PROFILE,
    "MODEL_NAME": MODEL_NAME,
    "TEMPERATURE": "1",
    "ENABLE_PROGRESSIVE_FINAL_CHAIN": "1",
    "ENABLE_LOW_STRUCTURE_CONCEPT_GUARD": "1",
    "ENABLE_HYBRID_SEVERE_DEVIATION_GUARDS": "1",
    "ENABLE_GPT56_INDEPENDENCE_GUARD": "1",
    "ENABLE_GPT56_STRUCTURAL_CALIBRATION": "1",
    "ENABLE_GPT56_SEVERE_DEVIATION_GUARDS": "1",
    "ENABLE_TEACHER_FEEDBACK_GUARDS": "1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_jsonl(path: Path) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    order: List[str] = []
    rows: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON：{exc}") from exc
            question_id = str(item.get("question_id") or "").strip()
            if not question_id:
                raise ValueError(f"{path}:{line_number} 缺少 question_id")
            if question_id in rows:
                raise ValueError(f"{path} question_id 重复：{question_id}")
            order.append(question_id)
            rows[question_id] = item
    if not order:
        raise ValueError(f"{path} 没有有效题目")
    return order, rows


def count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def extract_level(item: Dict[str, Any]) -> str:
    result = item.get("difficulty_rating")
    if isinstance(result, dict):
        level = str(result.get("difficulty_level") or "")
        if level in LEVEL_SET:
            return level
    return ""


def validate_rating_schema(item: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    result = item.get("difficulty_rating")
    if not isinstance(result, dict):
        return ["缺少 difficulty_rating"]
    level = str(result.get("difficulty_level") or "")
    if level not in LEVEL_SET:
        errors.append("difficulty_level 非法")
        return errors

    features = result.get("features")
    if not isinstance(features, dict):
        errors.append("缺少 features")
    else:
        missing_features = REQUIRED_FEATURES - set(features)
        if missing_features:
            errors.append(f"features 缺少 {len(missing_features)} 个必需字段")

    reasoning = result.get("reasoning")
    if not isinstance(reasoning, dict):
        errors.append("缺少 reasoning")
    else:
        missing_reasoning = REQUIRED_REASONING_FIELDS - set(reasoning)
        if missing_reasoning:
            errors.append(f"reasoning 缺少 {len(missing_reasoning)} 个必需字段")

    level_index = LEVELS.index(level)
    expected_lower = LEVELS[level_index - 1] if level_index > 0 else None
    expected_higher = LEVELS[level_index + 1] if level_index + 1 < len(LEVELS) else None
    if result.get("adjacent_lower_level") != expected_lower:
        errors.append("adjacent_lower_level 与最终等级不一致")
    if result.get("adjacent_higher_level") != expected_higher:
        errors.append("adjacent_higher_level 与最终等级不一致")
    if result.get("adjacent_reasoning_normalized") is not True:
        errors.append("相邻档解释未完成确定性同步")
    return errors


def validate_partial_run(
    path: Path,
    expected_ids: set[str],
) -> Dict[str, Any]:
    if not path.exists():
        return {"rows": 0, "complete": False}
    order, rows = load_jsonl(path)
    actual_ids = set(rows)
    extra = actual_ids - expected_ids
    if extra:
        sample = ", ".join(sorted(extra)[:3])
        raise ValueError(f"{path} 含当前输入之外的 question_id：{sample}")
    invalid = [
        question_id
        for question_id, item in rows.items()
        if validate_rating_schema(item)
    ]
    if invalid:
        raise ValueError(f"{path} 含 {len(invalid)} 条非法最终等级")
    return {
        "rows": len(order),
        "complete": actual_ids == expected_ids,
        "missing": len(expected_ids - actual_ids),
    }


def validate_complete_output(
    path: Path,
    expected_ids: set[str],
    *,
    expected_run_count: int | None = None,
) -> Dict[str, Any]:
    order, rows = load_jsonl(path)
    actual_ids = set(rows)
    missing = expected_ids - actual_ids
    extra = actual_ids - expected_ids
    invalid = [
        question_id
        for question_id, item in rows.items()
        if validate_rating_schema(item)
    ]
    if missing or extra or invalid:
        raise ValueError(
            f"{path} 完整性校验失败：缺少 {len(missing)}，多出 {len(extra)}，"
            f"非法等级 {len(invalid)}"
        )
    if expected_run_count is not None:
        wrong_run_count = []
        for question_id, item in rows.items():
            audit = item.get("lite_self_consistency")
            if not isinstance(audit, dict) or audit.get("run_count") != expected_run_count:
                wrong_run_count.append(question_id)
        if wrong_run_count:
            raise ValueError(
                f"{path} 有 {len(wrong_run_count)} 条集成运行次数不是 "
                f"{expected_run_count}"
            )
    return {
        "rows": len(order),
        "unique_ids": len(actual_ids),
        "sha256": sha256_file(path),
    }


def git_state(root: Path) -> Dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=root,
        text=True,
    ).strip()
    return {
        "commit": commit,
        "tracked_worktree_dirty": bool(tracked_status),
    }


def build_signature(
    *,
    root: Path,
    input_path: Path,
    prompt_path: Path,
    output_prefix: Path,
    concurrency: int,
    timeout: int,
    retries: int,
    no_cache: bool,
) -> Dict[str, Any]:
    rating_script = root / "src" / "physics_difficulty_rating_with_cache.py"
    ensemble_script = root / "src" / "physics_difficulty_lite_ensemble.py"
    production_script = root / "src" / "physics_difficulty_production_pipeline.py"
    production_wrapper = root / "scripts" / "run_physics_production.sh"
    git = git_state(root)
    return {
        "pipeline_version": PIPELINE_VERSION,
        "git_commit": git["commit"],
        "tracked_worktree_dirty": git["tracked_worktree_dirty"],
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "prompt": str(prompt_path),
        "prompt_sha256": sha256_file(prompt_path),
        "rating_script_sha256": sha256_file(rating_script),
        "ensemble_script_sha256": sha256_file(ensemble_script),
        "production_script_sha256": sha256_file(production_script),
        "production_wrapper_sha256": sha256_file(production_wrapper),
        "output_prefix": str(output_prefix),
        "base_url": os.getenv("BASE_URL", "http://172.22.0.35:4466/v1"),
        "model_name": MODEL_NAME,
        "rating_profile": RATING_PROFILE,
        "temperature": 1,
        "run_count": RUN_COUNT,
        "ensemble_mode": "structured_easy_guard",
        "environment": dict(PRODUCTION_ENV),
        "concurrency": concurrency,
        "timeout": timeout,
        "retries": retries,
        "no_cache": no_cache,
    }


def signature_payload(signature: Dict[str, Any]) -> str:
    comparable = {
        key: value
        for key, value in signature.items()
        if key not in {"tracked_worktree_dirty"}
    }
    return json.dumps(comparable, ensure_ascii=False, sort_keys=True)


def ensure_manifest_compatible(
    manifest_path: Path,
    signature: Dict[str, Any],
    run_paths: Sequence[Path],
) -> Dict[str, Any]:
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        previous_signature = existing.get("signature")
        if not isinstance(previous_signature, dict):
            raise ValueError(f"{manifest_path} 缺少合法 signature")
        if signature_payload(previous_signature) != signature_payload(signature):
            raise ValueError(
                "现有生产任务签名与本次配置不一致；请使用新的 output_prefix，"
                "不要把不同输入、Prompt或代码版本写入同一批结果"
            )
        return existing

    existing_outputs = [path for path in run_paths if path.exists()]
    if existing_outputs:
        names = ", ".join(path.name for path in existing_outputs[:3])
        raise ValueError(
            f"发现没有生产清单保护的历史输出：{names}；"
            "请换用新的 output_prefix，或恢复与这些输出对应的清单"
        )
    manifest = {
        "status": "running",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "signature": signature,
    }
    write_json_atomic(manifest_path, manifest)
    return manifest


def run_command(command: Sequence[str], *, cwd: Path, env: Dict[str, str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


def build_monitoring_summary(
    *,
    input_path: Path,
    run_paths: Sequence[Path],
    error_paths: Sequence[Path],
    final_path: Path,
    signature: Dict[str, Any],
) -> Dict[str, Any]:
    _input_order, input_rows = load_jsonl(input_path)
    _final_order, final_rows = load_jsonl(final_path)
    distribution: Counter[str] = Counter()
    decision_methods: Counter[str] = Counter()
    calibration_rules: Counter[str] = Counter()
    unanimous = 0
    disagreement = 0
    for item in final_rows.values():
        distribution[extract_level(item)] += 1
        audit = item.get("lite_self_consistency")
        if isinstance(audit, dict):
            unanimous += int(bool(audit.get("unanimous")))
            disagreement += int(not bool(audit.get("unanimous")))
            decision_methods[str(audit.get("decision_method") or "unknown")] += 1
            for action in audit.get("calibration_actions") or []:
                calibration_rules[str(action.get("rule") or "unknown")] += 1

    run_summaries = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    for run_path, error_path in zip(run_paths, error_paths):
        _order, rows = load_jsonl(run_path)
        run_prompt_tokens = sum(
            int(item.get("api_prompt_tokens") or 0) for item in rows.values()
        )
        run_completion_tokens = sum(
            int(item.get("api_completion_tokens") or 0) for item in rows.values()
        )
        run_total_tokens = sum(
            int(item.get("api_total_tokens") or 0) for item in rows.values()
        )
        total_prompt_tokens += run_prompt_tokens
        total_completion_tokens += run_completion_tokens
        total_tokens += run_total_tokens
        run_summaries.append(
            {
                "output": str(run_path),
                "output_sha256": sha256_file(run_path),
                "rows": len(rows),
                "prediction_distribution": dict(
                    Counter(extract_level(item) for item in rows.values())
                ),
                "error_log": str(error_path),
                "error_log_rows": count_jsonl_rows(error_path),
                "api_prompt_tokens": run_prompt_tokens,
                "api_completion_tokens": run_completion_tokens,
                "api_total_tokens": run_total_tokens,
                "average_api_time_seconds": round(
                    sum(float(item.get("api_time_use") or 0) for item in rows.values())
                    / len(rows),
                    3,
                ),
            }
        )

    total = len(final_rows)
    disagreement_rate = disagreement / total if total else 0.0
    calibration_count = sum(calibration_rules.values())
    calibration_rate = calibration_count / total if total else 0.0
    warnings: List[str] = []
    if disagreement_rate > 0.40:
        warnings.append(f"三次评级分歧率偏高：{disagreement_rate:.2%}")
    if calibration_rate > 0.10:
        warnings.append(f"结构化校准触发率偏高：{calibration_rate:.2%}")
    if any(summary["error_log_rows"] for summary in run_summaries):
        warnings.append("错误日志非空；最终输出已完整，但应检查历史失败原因")

    return {
        "pipeline_version": PIPELINE_VERSION,
        "generated_at": utc_now(),
        "git_commit": signature["git_commit"],
        "input": str(input_path),
        "input_sha256": signature["input_sha256"],
        "input_rows": len(input_rows),
        "final_output": str(final_path),
        "final_output_sha256": sha256_file(final_path),
        "final_rows": total,
        "prediction_distribution": {
            level: distribution.get(level, 0) for level in LEVELS
        },
        "unanimous_count": unanimous,
        "unanimous_rate": round(unanimous / total, 6) if total else 0.0,
        "disagreement_count": disagreement,
        "disagreement_rate": round(disagreement_rate, 6),
        "decision_methods": dict(decision_methods),
        "calibration_rules": dict(calibration_rules),
        "calibration_trigger_rate": round(calibration_rate, 6),
        "api_prompt_tokens": total_prompt_tokens,
        "api_completion_tokens": total_completion_tokens,
        "api_total_tokens": total_tokens,
        "runs": run_summaries,
        "warnings": warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="冻结版初中物理难度评级：三次 Lite + 结构化送分校准"
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument(
        "-o",
        "--output-prefix",
        required=True,
        help="输出前缀，例如 outputs/production/physics_20260728",
    )
    parser.add_argument("-c", "--concurrency", type=int, default=30)
    parser.add_argument("-t", "--timeout", type=int, default=180)
    parser.add_argument("-r", "--retries", type=int, default=3)
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="启用前缀缓存；生产冻结配置默认关闭",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="允许存在已跟踪但未提交的代码修改，不建议生产使用",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验配置并打印生产签名，不调用模型",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    input_path = Path(args.input).expanduser().resolve()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    prompt_path = (root / "prompts" / "初中物理难度打标提示词.txt").resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")
    if not prompt_path.exists():
        raise FileNotFoundError(f"冻结 Prompt 不存在：{prompt_path}")
    input_order, input_rows = load_jsonl(input_path)
    expected_ids = set(input_rows)

    signature = build_signature(
        root=root,
        input_path=input_path,
        prompt_path=prompt_path,
        output_prefix=output_prefix,
        concurrency=args.concurrency,
        timeout=args.timeout,
        retries=args.retries,
        no_cache=not args.use_cache,
    )
    if signature["tracked_worktree_dirty"] and not args.allow_dirty:
        raise ValueError(
            "检测到已跟踪文件存在未提交修改；生产运行要求干净版本。"
            "如仅做本地实验，可显式添加 --allow-dirty"
        )

    run_paths = [
        Path(f"{output_prefix}_run{index}.jsonl") for index in range(1, RUN_COUNT + 1)
    ]
    error_paths = [
        Path(f"{output_prefix}_run{index}_errors.jsonl")
        for index in range(1, RUN_COUNT + 1)
    ]
    final_path = Path(f"{output_prefix}_final.jsonl")
    manifest_path = Path(f"{output_prefix}_production_manifest.json")
    monitoring_path = Path(f"{output_prefix}_monitoring.json")

    print("生产配置签名：")
    print(json.dumps(signature, ensure_ascii=False, indent=2))
    print(f"输入题目：{len(input_order)}")
    if args.dry_run:
        return

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    manifest = ensure_manifest_compatible(manifest_path, signature, run_paths)
    manifest["status"] = "running"
    manifest["updated_at"] = utc_now()
    write_json_atomic(manifest_path, manifest)

    environment = os.environ.copy()
    environment.update(PRODUCTION_ENV)
    rating_script = root / "src" / "physics_difficulty_rating_with_cache.py"
    for index, (run_path, error_path) in enumerate(
        zip(run_paths, error_paths),
        1,
    ):
        state = validate_partial_run(run_path, expected_ids)
        if state["complete"]:
            print(f"Run{index} 已完整，跳过 API 调用：{run_path}")
            continue
        command = [
            sys.executable,
            "-u",
            str(rating_script),
            "-i",
            str(input_path),
            "-o",
            str(run_path),
            "-e",
            str(error_path),
            "-p",
            str(prompt_path),
            "-c",
            str(args.concurrency),
            "-t",
            str(args.timeout),
            "-r",
            str(args.retries),
        ]
        if not args.use_cache:
            command.append("--no-cache")
        run_command(command, cwd=root, env=environment)
        validate_complete_output(run_path, expected_ids)

    ensemble_script = root / "src" / "physics_difficulty_lite_ensemble.py"
    ensemble_command = [sys.executable, str(ensemble_script)]
    for run_path in run_paths:
        ensemble_command.extend(["--run", str(run_path)])
    ensemble_command.extend(
        [
            "--structured-easy-guard",
            "-o",
            str(final_path),
        ]
    )
    run_command(ensemble_command, cwd=root, env=environment)
    final_validation = validate_complete_output(
        final_path,
        expected_ids,
        expected_run_count=RUN_COUNT,
    )

    monitoring = build_monitoring_summary(
        input_path=input_path,
        run_paths=run_paths,
        error_paths=error_paths,
        final_path=final_path,
        signature=signature,
    )
    write_json_atomic(monitoring_path, monitoring)

    manifest.update(
        {
            "status": "complete",
            "updated_at": utc_now(),
            "final_output": str(final_path),
            "final_output_sha256": final_validation["sha256"],
            "monitoring": str(monitoring_path),
        }
    )
    write_json_atomic(manifest_path, manifest)
    print("生产评级完成：")
    print(json.dumps(monitoring, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
