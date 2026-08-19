# -*- coding: utf-8 -*-
"""高中化学两阶段难度评级运行入口。

本入口复用高中物理 Pipeline 稳定的 Responses API、Prompt Cache、
并发、重试、JSONL 断点续跑、图片输入和 token 统计外壳，但将
feature schema、高难特征检测、乘数复算和输入清洗替换为高中化学实现。

第二阶段默认开启物理同款证据守卫自动改档；可显式设置
``ENABLE_STAGE2_AUTO_ADJUST=0`` 关闭，且任何情况下最多调整一档。
化学高难乘数默认关闭；仅在消融实验中显式设置
``ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER=1`` 时启用。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import high_chemistry_pipeline_core as chemistry_core
import high_physics_difficulty_rating_and_verify as shared_runner


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "high_chemistry_two_stage_v3"
ENABLE_STAGE2_AUTO_ADJUST = (
    os.getenv("ENABLE_STAGE2_AUTO_ADJUST", "1").strip() == "1"
)
shared_runner.ENABLE_STAGE2_AUTO_ADJUST = ENABLE_STAGE2_AUTO_ADJUST
ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER = (
    os.getenv("ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER", "0").strip()
    == "1"
)
chemistry_core.CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER_ENABLED = (
    ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER
)
_shared_validate_verification = shared_runner.validate_verification
_shared_finalize_level = shared_runner.finalize_level

# 将共享运行器中的学科纯函数替换为化学实现。
shared_runner.HIGH_DIFFICULTY_FEATURE_NAMES = (
    chemistry_core.HIGH_DIFFICULTY_FEATURE_NAMES
)
shared_runner.REQUIRED_FEATURE_FIELDS = chemistry_core.REQUIRED_FEATURE_FIELDS
shared_runner.enrich_stage1_rating = chemistry_core.enrich_stage1_rating
shared_runner.normalize_stage1_rating = chemistry_core.normalize_stage1_rating
shared_runner.recalculate_verification = chemistry_core.recalculate_verification
shared_runner.prepare_question = chemistry_core.prepare_question

shared_runner.PIPELINE_VERSION = PIPELINE_VERSION
shared_runner.SUBJECT_DISPLAY_NAME = "高中化学"
shared_runner.PROGRESS_DESCRIPTION = "High Chemistry Pipeline"
shared_runner.DEFAULT_INPUT = ROOT / "data" / "high-chemistry-sample25k.jsonl"
shared_runner.DEFAULT_PROMPT = ROOT / "prompts" / "高中化学难度打标提示词.txt"
shared_runner.DEFAULT_OUTPUT = (
    ROOT / "outputs" / "model_runs" / "high_chemistry_two_stage.jsonl"
)
shared_runner.DEFAULT_ERRORS = (
    ROOT / "outputs" / "model_runs" / "high_chemistry_two_stage_errors.jsonl"
)
shared_runner.DEFAULT_CACHE = (
    ROOT / "outputs" / "cache" / "high_chemistry_stage1_prefix_cache.json"
)


def prepare_question(*args: Any, **kwargs: Any):
    """显式暴露化学输入清洗函数，便于线下审计和测试。"""
    return chemistry_core.prepare_question(*args, **kwargs)


def validate_verification(value: dict[str, Any]) -> dict[str, Any]:
    """在共享复核契约上增加化学的去重审计与信息充分性校验。"""
    normalized = _shared_validate_verification(value)
    chemistry_core.validate_structural_revision_evidence(normalized)
    overlap_review = normalized.get("high_feature_overlap_review")
    if not isinstance(overlap_review, list):
        raise ValueError("high_feature_overlap_review 必须为数组")
    for index, item in enumerate(overlap_review):
        if not isinstance(item, dict):
            raise ValueError(f"high_feature_overlap_review[{index}] 必须为对象")
        missing = {"features", "resolution", "reason"} - item.keys()
        if missing:
            raise ValueError(
                f"high_feature_overlap_review[{index}] 缺少字段：{sorted(missing)}"
            )
        if not isinstance(item["features"], list) or any(
            name not in chemistry_core.HIGH_DIFFICULTY_FEATURE_NAMES
            for name in item["features"]
        ):
            raise ValueError(f"high_feature_overlap_review[{index}].features 含非法值")
    input_review = normalized.get("input_sufficiency_review")
    if not isinstance(input_review, dict):
        raise ValueError("input_sufficiency_review 必须为对象")
    if input_review.get("status") not in {"充分", "部分缺失", "信息不足"}:
        raise ValueError("input_sufficiency_review.status 含非法值")
    missing_information = input_review.get("missing_information")
    if not isinstance(missing_information, list) or any(
        not isinstance(item, str) or not item.strip()
        for item in missing_information
    ):
        raise ValueError("input_sufficiency_review.missing_information 必须为字符串数组")
    return normalized


shared_runner.validate_verification = validate_verification


def _chemistry_finalize_adapter(
    *,
    current_level: str,
    reasonableness: str,
    model_suggested_level: Any,
    multiplier_reasonableness: str,
    input_sufficiency: str,
    original_high_count: int | None = None,
    reviewed_high_count: int | None = None,
):
    # 化学乘数关闭时，高难特征数量变化不再代表乘数桶变化；
    # 其余改档条件完全复用物理的最多一档与人工复核守卫。
    counts = (
        (original_high_count, reviewed_high_count)
        if ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER
        else (None, None)
    )
    return _shared_finalize_level(
        current_level=current_level,
        reasonableness=reasonableness,
        model_suggested_level=model_suggested_level,
        multiplier_reasonableness=multiplier_reasonableness,
        input_sufficiency=input_sufficiency,
        original_high_count=counts[0],
        reviewed_high_count=counts[1],
    )


shared_runner.finalize_level = _chemistry_finalize_adapter


def main() -> None:
    shared_runner.main()


if __name__ == "__main__":
    main()
