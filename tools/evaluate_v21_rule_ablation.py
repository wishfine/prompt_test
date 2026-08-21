# -*- coding: utf-8 -*-
"""高中化学 V21 / V21.1 规则消融与真实边际贡献评估器。

核心功能：
1. 完整基线重放校验（Replay Verification）：必须保证 replay_mismatch_count == 0；
2. 真实 Leave-One-Rule-Out (LORO) 消融：计算每条规则的 Trigger、Effective、Fixed、Harmed、Net Gain、Marginal Precision、ΔAccuracy、ΔQWK；
3. 改档方向归因（Directional Attribution）：统计 1→2, 2→1, 3→2, 3→4 等方向及 Fixed/Harmed 分布；
4. 规则组合消融（Group Ablation）：Group A (2档保护组) 与 Group B (floor 结构组)；
5. Stage 2 离线 Bypass 潜力分析；
6. 最终剩余错误结构特征分布报告。
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


LEVELS = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
INDEX_LEVEL = {index: level for index, level in enumerate(LEVELS)}

ALL_STRUCTURAL_RULE_IDS = [
    "direct_prototype_exact_1",
    "calculation_model_floor_2",
    "basic_explicit_application_ceiling_2",
    "parallel_basic_bundle_ceiling_2",
    "standard_chain_floor_3",
    "compressed_high_burden_floor_4",
    "hard_structural_cluster_floor_4",
    "regular_comprehensive_ceiling_3",
]


def min_level(left: str, right: str) -> str:
    return left if LEVEL_INDEX[left] <= LEVEL_INDEX[right] else right


def max_level(left: str, right: str) -> str:
    return left if LEVEL_INDEX[left] >= LEVEL_INDEX[right] else right


def quadratic_weighted_kappa(
    truth_values: list[str],
    prediction_values: list[str],
) -> float | None:
    if not truth_values or len(truth_values) != len(prediction_values):
        return None
    size = len(LEVELS)
    observed = [[0 for _ in range(size)] for _ in range(size)]
    truth_counts = [0 for _ in range(size)]
    prediction_counts = [0 for _ in range(size)]
    for truth, prediction in zip(truth_values, prediction_values):
        truth_index = LEVEL_INDEX[truth]
        prediction_index = LEVEL_INDEX[prediction]
        observed[truth_index][prediction_index] += 1
        truth_counts[truth_index] += 1
        prediction_counts[prediction_index] += 1

    observed_disagreement = 0.0
    expected_disagreement = 0.0
    denominator = float((size - 1) ** 2)
    sample_count = len(truth_values)
    for truth_index in range(size):
        for prediction_index in range(size):
            weight = ((truth_index - prediction_index) ** 2) / denominator
            observed_disagreement += weight * observed[truth_index][prediction_index]
            expected_disagreement += (
                weight
                * truth_counts[truth_index]
                * prediction_counts[prediction_index]
                / sample_count
            )
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else None
    return round(1.0 - observed_disagreement / expected_disagreement, 4)


def derive_structural_level_constraint_evaluator(
    features: dict[str, Any],
    high_names: list[str],
    disabled_rule_ids: set[str] | None = None,
) -> tuple[dict[str, Any], set[str]]:
    """Evaluator 专用结构约束派生函数，支持针对指定规则进行消融，同时返回 triggered_rule_ids。"""
    disabled = disabled_rule_ids or set()
    triggered_rules: set[str] = set()

    floor = "难度1档"
    ceiling = "难度5档"
    rule_ids: list[str] = []
    evidence: list[str] = []

    # 1. direct_prototype_exact_1
    direct_prototype = (
        features.get("primary_problem_structure") == "概念辨析"
        and features.get("knowledge_count") == "1个"
        and features.get("knowledge_scope") == "单知识点"
        and features.get("substance_count") == "1种"
        and features.get("substance_relation") == "单一物质"
        and features.get("reaction_count") == "0-1个"
        and features.get("reaction_relation") == "无反应链"
        and features.get("process_structure") == "单阶段"
        and features.get("step_count") == "1-2步"
        and features.get("required_task_breadth") == "单一规则任务"
        and features.get("model_explicitness") == "模型完全显性"
        and features.get("model_relation") == "单一模型"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("representation_conversion") == "无转换"
        and features.get("information_conversion") == "无信息转换"
        and features.get("experiment_requirement") == "无"
        and features.get("calculation_model") == "无定量计算"
        and not high_names
    )
    if direct_prototype:
        triggered_rules.add("direct_prototype_exact_1")
        if "direct_prototype_exact_1" not in disabled:
            return {
                "difficulty_floor": "难度1档",
                "difficulty_ceiling": "难度1档",
                "rule_ids": ["direct_prototype_exact_1"],
                "evidence": ["纯单知识点单步直接套用原型"],
                "confidence": "高",
                "constraint_conflict": False,
            }, triggered_rules

    # 2. calculation_model_floor_2
    if features.get("calculation_model") in {
        "常规化学计量",
        "多步化学计量",
        "平衡常数或Ka/Kb/Ksp",
        "多模型定量耦合",
    }:
        triggered_rules.add("calculation_model_floor_2")
        if "calculation_model_floor_2" not in disabled:
            floor = max_level(floor, "难度2档")
            rule_ids.append("calculation_model_floor_2")
            evidence.append(f"定量计算模型({features.get('calculation_model')})")

    # 3. basic_explicit_application_ceiling_2
    basic_explicit_app = (
        features.get("step_count") == "1-2步"
        and features.get("model_explicitness") == "模型完全显性"
        and features.get("reasoning_chain") in {"直接套用", "简单因果"}
        and features.get("subquestion_dependency") != "后问依赖前问"
        and not features.get("shared_model_across_subquestions", False)
        and features.get("model_relation") in {"单一模型", "同一模型多状态"}
        and features.get("information_conversion") in {"无信息转换", "直接读取", "单次关系转换"}
        and features.get("evidence_relation") in {"直接给定", "单证据对应", "多证据独立"}
        and features.get("critical_condition") in {"无临界", "显性给出临界"}
        and features.get("classification_discussion") == "无"
        and features.get("constraint_structure") in {"无约束", "单一约束", "多约束相互独立"}
        and features.get("calculation_model") in {"无定量计算", "常规化学计量"}
        and features.get("calculation_complexity") in {"直接判断", "简单计算"}
        and features.get("experiment_requirement") in {"无", "基础操作或读数", "直接现象解释"}
        and features.get("route_design_requirement") in {"无", "已知路线补全"}
        and features.get("required_task_breadth") != "4个及以上异质必要任务"
        and not high_names
    )
    if basic_explicit_app:
        triggered_rules.add("basic_explicit_application_ceiling_2")
        if "basic_explicit_application_ceiling_2" not in disabled:
            ceiling = min_level(ceiling, "难度2档")
            rule_ids.append("basic_explicit_application_ceiling_2")
            evidence.append("1-2步显性基础应用，无高难无强依赖")

    # 4. parallel_basic_bundle_ceiling_2
    parallel_basic_bundle = (
        features.get("required_task_breadth") in {"2-3个异质必要任务", "4个及以上异质必要任务"}
        and features.get("substance_relation") in {"单一物质", "相互独立"}
        and features.get("reaction_relation") in {"无反应链", "并列独立"}
        and features.get("process_structure") == "单阶段"
        and features.get("subquestion_dependency") != "后问依赖前问"
        and not features.get("shared_model_across_subquestions", False)
        and features.get("model_explicitness") == "模型完全显性"
        and features.get("model_relation") in {"单一模型", "同一模型多状态"}
        and features.get("reasoning_chain") in {"直接套用", "简单因果"}
        and features.get("information_conversion") in {"无信息转换", "直接读取"}
        and features.get("evidence_relation") in {"直接给定", "单证据对应", "多证据独立"}
        and features.get("hidden_conditions") == "无"
        and features.get("critical_condition") in {"无临界", "显性给出临界"}
        and features.get("constraint_structure") in {"无约束", "单一约束", "多约束相互独立"}
        and features.get("calculation_model") in {"无定量计算", "常规化学计量"}
        and features.get("calculation_complexity") in {"直接判断", "简单计算"}
        and features.get("experiment_requirement") in {"无", "基础操作或读数", "直接现象解释"}
        and features.get("route_design_requirement") in {"无", "已知路线补全"}
        and not high_names
    )
    if parallel_basic_bundle:
        triggered_rules.add("parallel_basic_bundle_ceiling_2")
        if "parallel_basic_bundle_ceiling_2" not in disabled:
            ceiling = min_level(ceiling, "难度2档")
            rule_ids.append("parallel_basic_bundle_ceiling_2")
            evidence.append("并列基础多任务(单阶段/无反应链或并列独立/模型显性/直接套用)")

    # 5. standard_chain_floor_3
    has_real_dependency = (
        features.get("reaction_relation") in {"显性顺序衔接", "前后反应强依赖", "多路径反应网络"}
        or features.get("subquestion_dependency") == "后问依赖前问"
        or (features.get("shared_model_across_subquestions") is True and features.get("process_structure") != "单阶段")
        or features.get("substance_relation") in {"前后转化依赖", "组成—性质—反应网络"}
        or features.get("process_structure") in {"多阶段显性流程", "多阶段强依赖", "循环或回流流程"}
    )
    standard_chain_tight = (
        features.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"}
        and has_real_dependency
        and (
            features.get("calculation_model") in {"多步化学计量", "平衡常数或Ka/Kb/Ksp", "多模型定量耦合"}
            or features.get("information_conversion") in {"多源信息联合转换", "流程或图谱反推", "单次关系转换"}
            or features.get("experiment_requirement") in {"控制变量或异常分析", "方案设计或误差反演", "数据归纳"}
            or features.get("reasoning_chain") in {"多层因果", "逆向推理或临界分析"}
        )
    )
    if standard_chain_tight:
        triggered_rules.add("standard_chain_floor_3")
        if "standard_chain_floor_3" not in disabled:
            floor = max_level(floor, "难度3档")
            rule_ids.append("standard_chain_floor_3")
            evidence.append("3-5步以上真实关联依赖链")

    # 6. Floor 4: complex_quantitative, model_migration, compressed_high
    complex_quantitative = "复杂定量、参数或范围" in high_names

    model_migration_multistage_strong = (
        features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
        and features.get("model_relation") in {"模型切换", "多模型耦合"}
        and features.get("process_structure") in {"多阶段强依赖", "循环或回流流程"}
    )

    has_additional_high_burden = (
        bool(set(high_names) - {"多模型或多平衡耦合"})
        or features.get("information_conversion") in {"多源信息联合转换", "流程或图谱反推"}
        or features.get("constraint_structure") == "多约束联合筛选"
        or features.get("experiment_requirement") in {"控制变量或异常分析", "方案设计或误差反演"}
        or features.get("route_design_requirement") in {
            "合成路线设计",
            "分离提纯方案设计",
            "路线优化与可行性验证",
        }
    )
    model_migration_system_coupling_strong = (
        features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
        and features.get("model_relation") in {"模型切换", "多模型耦合"}
        and features.get("substance_relation") in {"前后转化依赖", "组成—性质—反应网络"}
        and has_additional_high_burden
    )

    axis_model_ident = features.get("model_explicitness") in {"半隐含模型", "隐含模型", "需要自主建模"}
    axis_reasoning = features.get("reasoning_chain") in {"多层因果", "逆向推理或临界分析"}
    axis_model_relation = features.get("model_relation") in {"模型切换", "多模型耦合"}
    axis_quant = (
        features.get("calculation_model") in {"平衡常数或Ka/Kb/Ksp", "多模型定量耦合", "多步化学计量"}
        and features.get("calculation_complexity") in {"多方程联立", "参数或范围计算", "多步计算"}
    )
    axis_info = (
        features.get("information_conversion") in {"多源信息联合转换", "流程或图谱反推"}
        or (features.get("information_conversion") == "单次关系转换" and features.get("evidence_relation") in {"证据链相互支持", "证据冲突需排除"})
    )
    axis_exp = (
        features.get("experiment_requirement") in {"控制变量或异常分析", "方案设计或误差反演"}
        or features.get("route_design_requirement") in {"合成路线设计", "分离提纯方案设计", "路线优化与可行性验证"}
    )
    axis_constraint = (
        features.get("constraint_structure") == "多约束联合筛选"
        or features.get("critical_condition") in {"需要推导过量不足边界", "隐含终点或有效区间"}
    )
    axes = [axis_model_ident, axis_reasoning, axis_model_relation, axis_quant, axis_info, axis_exp, axis_constraint]
    is_compressed_high = (
        sum(axes) >= 2
        and features.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"}
        and (
            features.get("model_relation") in {"模型切换", "多模型耦合"}
            or features.get("information_conversion") not in {"无信息转换", "直接读取"}
            or features.get("calculation_model") not in {"无定量计算", "常规化学计量"}
            or features.get("experiment_requirement") not in {"无", "基础操作或读数", "直接现象解释"}
        )
    )

    is_hard_cluster = bool(complex_quantitative or model_migration_multistage_strong or model_migration_system_coupling_strong)
    if is_hard_cluster:
        triggered_rules.add("hard_structural_cluster_floor_4")
    if is_compressed_high:
        triggered_rules.add("compressed_high_burden_floor_4")

    # Determine whether floor 4 should be applied based on disabled rules
    apply_floor_4 = False
    if is_hard_cluster and "hard_structural_cluster_floor_4" not in disabled:
        apply_floor_4 = True
        rule_ids.append("hard_structural_cluster_floor_4")
    elif is_compressed_high and "compressed_high_burden_floor_4" not in disabled:
        apply_floor_4 = True
        rule_ids.append("compressed_high_burden_floor_4")

    if apply_floor_4:
        floor = max_level(floor, "难度4档")
        if complex_quantitative:
            evidence.append("复杂定量、参数或范围(高难特征)")
        if model_migration_multistage_strong:
            evidence.append("长步数(6-8步+)+模型迁移+多阶段强依赖")
        if model_migration_system_coupling_strong:
            evidence.append("长步数(6-8步+)+模型迁移+体系耦合+高层信息/约束/实验负担")
        if is_compressed_high:
            evidence.append(f"短链高密度综合(命中{sum(axes)}个独立强负担轴)")

    # 7. regular_comprehensive_ceiling_3
    regular_comprehensive_tight = (
        features.get("step_count") in {"1-2步", "3-5步"}
        and not high_names
        and features.get("model_explicitness") == "模型完全显性"
        and features.get("reasoning_chain") in {"直接套用", "简单因果"}
        and features.get("hidden_conditions") == "无"
        and features.get("information_conversion") in {"无信息转换", "直接读取"}
        and features.get("model_relation") in {"单一模型", "同一模型多状态"}
        and features.get("process_structure") not in {"多阶段强依赖", "循环或回流流程"}
        and features.get("calculation_model") not in {"多模型定量耦合"}
        and features.get("experiment_requirement") not in {"控制变量或异常分析", "方案设计或误差反演"}
        and features.get("route_design_requirement") not in {"合成路线设计", "分离提纯方案设计", "路线优化与可行性验证"}
        and features.get("constraint_structure") != "多约束联合筛选"
        and not (complex_quantitative or model_migration_multistage_strong or model_migration_system_coupling_strong or is_compressed_high)
    )
    if regular_comprehensive_tight:
        triggered_rules.add("regular_comprehensive_ceiling_3")
        if "regular_comprehensive_ceiling_3" not in disabled:
            ceiling = min_level(ceiling, "难度3档")
            rule_ids.append("regular_comprehensive_ceiling_3")
            evidence.append("普通常规综合(显性模型/简单因果/直接读取)")

    conflict = LEVEL_INDEX[floor] > LEVEL_INDEX[ceiling]
    return {
        "difficulty_floor": floor,
        "difficulty_ceiling": ceiling,
        "rule_ids": rule_ids,
        "evidence": evidence,
        "confidence": "高" if not conflict else "低",
        "constraint_conflict": conflict,
    }, triggered_rules


def apply_structural_level_constraint_evaluator(
    score_level: str,
    constraint: dict[str, Any],
) -> tuple[str, str, bool, bool]:
    if constraint.get("constraint_conflict"):
        return score_level, "conflict_maintained", True, False

    score_idx = LEVEL_INDEX[score_level]
    floor_idx = LEVEL_INDEX[constraint["difficulty_floor"]]
    ceiling_idx = LEVEL_INDEX[constraint["difficulty_ceiling"]]

    if floor_idx > ceiling_idx:
        return score_level, "conflict_maintained", True, False

    target_idx = min(ceiling_idx, max(floor_idx, score_idx))
    severe_disagreement = abs(target_idx - score_idx) >= 2

    if target_idx > score_idx:
        final_idx = min(target_idx, score_idx + 1)
    elif target_idx < score_idx:
        final_idx = max(target_idx, score_idx - 1)
    else:
        final_idx = score_idx

    final_level = INDEX_LEVEL[final_idx]

    if final_level == score_level:
        action = "maintained"
    elif final_idx > score_idx:
        action = f"raised_to_{final_level}"
    else:
        action = f"lowered_to_{final_level}"

    return final_level, action, False, severe_disagreement


def load_dataset(predictions_paths: list[str], labels_path: str) -> tuple[dict[str, str], list[dict[str, Any]]]:
    labels: dict[str, str] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row["question_id"])
            level = row.get("reviewed_difficulty_level") or row.get("manual_difficulty_level")
            if level in LEVEL_INDEX:
                labels[qid] = level

    rows: list[dict[str, Any]] = []
    for pattern in predictions_paths:
        for fp in sorted(glob.glob(pattern)):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    qid = str(row.get("question_id"))
                    if qid in labels:
                        row["teacher_level"] = labels[qid]
                        row["source_run_file"] = Path(fp).name
                        rows.append(row)
    return labels, rows


def run_ablation_analysis(rows: list[dict[str, Any]]) -> dict[str, Any]:
    # 1. Replay verification
    replay_mismatches = []
    for idx, row in enumerate(rows):
        s1 = row["difficulty_rating_stage1"]
        features = s1["features"]
        high_names = s1["high_difficulty_features"]
        score_level = s1["difficulty_level_from_score"]
        expected_s1_level = row["difficulty_level_step1"]

        constraint, _ = derive_structural_level_constraint_evaluator(features, high_names)
        replayed_level, _, _, _ = apply_structural_level_constraint_evaluator(score_level, constraint)
        if replayed_level != expected_s1_level:
            replay_mismatches.append({
                "row_index": idx,
                "question_id": row.get("question_id"),
                "expected": expected_s1_level,
                "replayed": replayed_level,
            })

    if replay_mismatches:
        raise RuntimeError(
            f"Replay verification FAILED with {len(replay_mismatches)} mismatches! "
            f"Evaluator logic must be 100% identical to production code."
        )

    # 2. Baseline performance
    teacher_levels = [r["teacher_level"] for r in rows]
    full_predictions = []
    triggered_rules_per_row = []
    score_derived_levels = []

    for r in rows:
        s1 = r["difficulty_rating_stage1"]
        score_level = s1["difficulty_level_from_score"]
        score_derived_levels.append(score_level)
        constraint, trig = derive_structural_level_constraint_evaluator(s1["features"], s1["high_difficulty_features"])
        pred_level, _, _, _ = apply_structural_level_constraint_evaluator(score_level, constraint)
        full_predictions.append(pred_level)
        triggered_rules_per_row.append(trig)

    total_count = len(rows)
    baseline_correct = sum(1 for p, t in zip(full_predictions, teacher_levels) if p == t)
    baseline_acc = round(baseline_correct / total_count, 4)
    baseline_qwk = quadratic_weighted_kappa(teacher_levels, full_predictions)

    score_correct = sum(1 for p, t in zip(score_derived_levels, teacher_levels) if p == t)
    score_acc = round(score_correct / total_count, 4)
    score_qwk = quadratic_weighted_kappa(teacher_levels, score_derived_levels)

    # 3. Leave-One-Rule-Out (LORO)
    single_rule_results = {}
    directional_stats = defaultdict(lambda: defaultdict(lambda: {"fixed": 0, "harmed": 0, "total": 0}))

    for rule_id in ALL_STRUCTURAL_RULE_IDS:
        trigger_count = 0
        effective_count = 0
        fixed = 0
        harmed = 0
        neutral = 0

        without_predictions = []
        for idx, r in enumerate(rows):
            s1 = r["difficulty_rating_stage1"]
            features = s1["features"]
            high_names = s1["high_difficulty_features"]
            score_level = s1["difficulty_level_from_score"]
            teacher = r["teacher_level"]
            full_pred = full_predictions[idx]

            trig = triggered_rules_per_row[idx]
            is_triggered = rule_id in trig
            if is_triggered:
                trigger_count += 1

            constraint_without, _ = derive_structural_level_constraint_evaluator(
                features, high_names, disabled_rule_ids={rule_id}
            )
            without_pred, _, _, _ = apply_structural_level_constraint_evaluator(score_level, constraint_without)
            without_predictions.append(without_pred)

            if without_pred != full_pred:
                effective_count += 1
                trans = f"{without_pred[2]}→{full_pred[2]}"
                directional_stats[rule_id][trans]["total"] += 1
                if full_pred == teacher and without_pred != teacher:
                    fixed += 1
                    directional_stats[rule_id][trans]["fixed"] += 1
                elif full_pred != teacher and without_pred == teacher:
                    harmed += 1
                    directional_stats[rule_id][trans]["harmed"] += 1
                else:
                    neutral += 1
            else:
                neutral += 1

        net_gain = fixed - harmed
        marginal_prec = round(fixed / (fixed + harmed), 4) if (fixed + harmed) > 0 else None
        without_correct = sum(1 for p, t in zip(without_predictions, teacher_levels) if p == t)
        without_acc = round(without_correct / total_count, 4)
        without_qwk = quadratic_weighted_kappa(teacher_levels, without_predictions)
        delta_acc = round(baseline_acc - without_acc, 4)
        delta_qwk = round((baseline_qwk or 0.0) - (without_qwk or 0.0), 4)

        single_rule_results[rule_id] = {
            "rule_id": rule_id,
            "trigger_count": trigger_count,
            "effective_count": effective_count,
            "fixed": fixed,
            "harmed": harmed,
            "neutral": neutral,
            "net_gain": net_gain,
            "marginal_precision": marginal_prec,
            "accuracy_full": baseline_acc,
            "accuracy_without_rule": without_acc,
            "delta_accuracy": delta_acc,
            "qwk_full": baseline_qwk,
            "qwk_without_rule": without_qwk,
            "delta_qwk": delta_qwk,
            "directional_transitions": dict(directional_stats[rule_id]),
        }

    # 4. Group Ablation
    groups = {
        "Group A: 2档保护组 - remove basic_explicit only": {"basic_explicit_application_ceiling_2"},
        "Group A: 2档保护组 - remove parallel_bundle only": {"parallel_basic_bundle_ceiling_2"},
        "Group A: 2档保护组 - remove regular_comprehensive only": {"regular_comprehensive_ceiling_3"},
        "Group A: 2档保护组 - remove basic + parallel": {
            "basic_explicit_application_ceiling_2",
            "parallel_basic_bundle_ceiling_2",
        },
        "Group A: 2档保护组 - remove parallel + regular": {
            "parallel_basic_bundle_ceiling_2",
            "regular_comprehensive_ceiling_3",
        },
        "Group A: 2档保护组 - remove all three (basic+parallel+regular)": {
            "basic_explicit_application_ceiling_2",
            "parallel_basic_bundle_ceiling_2",
            "regular_comprehensive_ceiling_3",
        },
        "Group B: floor 结构组 - remove calc_model only": {"calculation_model_floor_2"},
        "Group B: floor 结构组 - remove standard_chain only": {"standard_chain_floor_3"},
        "Group B: floor 结构组 - remove compressed_high only": {"compressed_high_burden_floor_4"},
        "Group B: floor 结构组 - remove hard_cluster only": {"hard_structural_cluster_floor_4"},
        "Group B: floor 结构组 - remove both floor 4 (compressed+hard_cluster)": {
            "compressed_high_burden_floor_4",
            "hard_structural_cluster_floor_4",
        },
        "Group B: floor 结构组 - remove all floor rules": {
            "calculation_model_floor_2",
            "standard_chain_floor_3",
            "compressed_high_burden_floor_4",
            "hard_structural_cluster_floor_4",
        },
        "All Rules Removed (pure continuous score level)": set(ALL_STRUCTURAL_RULE_IDS),
    }

    group_results = {}
    for group_name, disabled_set in groups.items():
        effective = 0
        fixed = 0
        harmed = 0
        grp_preds = []
        for idx, r in enumerate(rows):
            s1 = r["difficulty_rating_stage1"]
            features = s1["features"]
            high_names = s1["high_difficulty_features"]
            score_level = s1["difficulty_level_from_score"]
            teacher = r["teacher_level"]
            full_pred = full_predictions[idx]

            constraint_grp, _ = derive_structural_level_constraint_evaluator(
                features, high_names, disabled_rule_ids=disabled_set
            )
            grp_pred, _, _, _ = apply_structural_level_constraint_evaluator(score_level, constraint_grp)
            grp_preds.append(grp_pred)

            if grp_pred != full_pred:
                effective += 1
                if full_pred == teacher and grp_pred != teacher:
                    fixed += 1
                elif full_pred != teacher and grp_pred == teacher:
                    harmed += 1

        grp_correct = sum(1 for p, t in zip(grp_preds, teacher_levels) if p == t)
        grp_acc = round(grp_correct / total_count, 4)
        grp_qwk = quadratic_weighted_kappa(teacher_levels, grp_preds)
        group_results[group_name] = {
            "disabled_rules": sorted(disabled_set),
            "effective_count": effective,
            "fixed": fixed,
            "harmed": harmed,
            "net_gain": fixed - harmed,
            "marginal_precision": round(fixed / (fixed + harmed), 4) if (fixed + harmed) > 0 else None,
            "accuracy_without_group": grp_acc,
            "delta_accuracy": round(baseline_acc - grp_acc, 4),
            "qwk_without_group": grp_qwk,
            "delta_qwk": round((baseline_qwk or 0.0) - (grp_qwk or 0.0), 4),
        }

    # 5. Stage 2 Bypass Offline Analysis
    stage2_candidates = 0
    candidate_reasons = Counter()
    for r in rows:
        s1 = r["difficulty_rating_stage1"]
        score_level = s1.get("difficulty_level_from_score")
        step1_level = r.get("difficulty_level_step1")
        constraint = s1.get("structural_level_constraint") or {}
        input_quality = r.get("input_quality") or {}

        score_step1_mismatch = score_level != step1_level
        constraint_conflict = constraint.get("constraint_conflict") is True
        severe_disagreement = s1.get("structural_severe_disagreement") is True
        input_insufficiency = input_quality.get("input_sufficiency") != "充分"

        is_candidate = score_step1_mismatch or constraint_conflict or severe_disagreement or input_insufficiency
        if is_candidate:
            stage2_candidates += 1
            if score_step1_mismatch:
                candidate_reasons["score_structural_gap"] += 1
            if constraint_conflict:
                candidate_reasons["constraint_conflict"] += 1
            if severe_disagreement:
                candidate_reasons["severe_disagreement"] += 1
            if input_insufficiency:
                candidate_reasons["input_insufficiency"] += 1

    stage2_bypass_analysis = {
        "total_records": total_count,
        "stage2_candidate_count": stage2_candidates,
        "stage2_candidate_rate": round(stage2_candidates / total_count, 4),
        "potential_api_savings_rate": round(1.0 - (stage2_candidates / total_count), 4),
        "candidate_reasons_breakdown": dict(candidate_reasons),
    }

    # 6. Residual Error Structure Analysis
    error_patterns = [
        ("难度1档", "难度2档"),
        ("难度2档", "难度1档"),
        ("难度3档", "难度2档"),
        ("难度3档", "难度4档"),
        ("难度4档", "难度3档"),
        ("难度5档", "难度4档"),
    ]
    residual_error_reports = {}
    for t_lvl, p_lvl in error_patterns:
        key = f"Teacher{t_lvl[2]}→Pred{p_lvl[2]}"
        matched_rows = [r for idx, r in enumerate(rows) if r["teacher_level"] == t_lvl and full_predictions[idx] == p_lvl]
        
        feature_dist = {
            "count": len(matched_rows),
            "step_count": dict(Counter(r["difficulty_rating_stage1"]["features"].get("step_count") for r in matched_rows)),
            "required_task_breadth": dict(Counter(r["difficulty_rating_stage1"]["features"].get("required_task_breadth") for r in matched_rows)),
            "model_explicitness": dict(Counter(r["difficulty_rating_stage1"]["features"].get("model_explicitness") for r in matched_rows)),
            "model_relation": dict(Counter(r["difficulty_rating_stage1"]["features"].get("model_relation") for r in matched_rows)),
            "reasoning_chain": dict(Counter(r["difficulty_rating_stage1"]["features"].get("reasoning_chain") for r in matched_rows)),
            "information_conversion": dict(Counter(r["difficulty_rating_stage1"]["features"].get("information_conversion") for r in matched_rows)),
            "calculation_model": dict(Counter(r["difficulty_rating_stage1"]["features"].get("calculation_model") for r in matched_rows)),
            "calculation_complexity": dict(Counter(r["difficulty_rating_stage1"]["features"].get("calculation_complexity") for r in matched_rows)),
            "experiment_requirement": dict(Counter(r["difficulty_rating_stage1"]["features"].get("experiment_requirement") for r in matched_rows)),
            "triggered_rules": dict(Counter(rule for idx, r in enumerate(rows) if r in matched_rows for rule in triggered_rules_per_row[idx])),
            "raw_predicted_accuracy_mean": round(sum(r["difficulty_rating_stage1"]["original_predicted_accuracy"] for r in matched_rows) / len(matched_rows), 2) if matched_rows else None,
        }
        residual_error_reports[key] = feature_dist

    return {
        "summary": {
            "total_records_evaluated": total_count,
            "replay_verification": "PASS (0 mismatches)",
            "score_derived_baseline": {
                "accuracy": score_acc,
                "qwk": score_qwk,
            },
            "full_rules_baseline": {
                "accuracy": baseline_acc,
                "qwk": baseline_qwk,
            },
            "structural_net_gain_accuracy": round(baseline_acc - score_acc, 4),
            "structural_net_gain_qwk": round((baseline_qwk or 0.0) - (score_qwk or 0.0), 4),
        },
        "single_rule_ablation": single_rule_results,
        "group_ablation": group_results,
        "stage2_bypass_analysis": stage2_bypass_analysis,
        "residual_error_structure": residual_error_reports,
    }


def format_markdown_report(analysis: dict[str, Any]) -> str:
    lines = []
    lines.append("# 高中化学 V21.1 规则消融与真实边际贡献评估报告\n")
    lines.append("## 一、基准重放与结构约束总体增益\n")
    summary = analysis["summary"]
    lines.append(f"- **评测记录总数**: {summary['total_records_evaluated']}")
    lines.append(f"- **完整重放校验**: {summary['replay_verification']}")
    lines.append(f"- **连续打分基线 (Score-only Baseline)**: Accuracy = `{summary['score_derived_baseline']['accuracy'] * 100:.2f}%`, QWK = `{summary['score_derived_baseline']['qwk']}`")
    lines.append(f"- **完整规则基线 (Full-rules Step1)**: Accuracy = `{summary['full_rules_baseline']['accuracy'] * 100:.2f}%`, QWK = `{summary['full_rules_baseline']['qwk']}`")
    lines.append(f"- **结构约束净增益 (Structural Net Gain)**: **ΔAccuracy = `{summary['structural_net_gain_accuracy'] * 100:+.2f}%`**, **ΔQWK = `{summary['structural_net_gain_qwk']:+.4f}`**\n")

    lines.append("## 二、单规则消融表 (Leave-One-Rule-Out Ablation)\n")
    lines.append("| Rule ID | Trigger | Effective | Fixed | Harmed | Net Gain | Marginal Prec | Without Acc | ΔAccuracy | Without QWK | ΔQWK |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for rule_id, res in analysis["single_rule_ablation"].items():
        prec_str = f"{res['marginal_precision']*100:.1f}%" if res['marginal_precision'] is not None else "N/A"
        delta_acc_str = f"{res['delta_accuracy']*100:+.2f}%"
        delta_qwk_str = f"{res['delta_qwk']:+.4f}"
        lines.append(
            f"| `{rule_id}` | {res['trigger_count']} | {res['effective_count']} | {res['fixed']} | {res['harmed']} | {res['net_gain']} | {prec_str} | {res['accuracy_without_rule']*100:.2f}% | **{delta_acc_str}** | {res['qwk_without_rule']} | {delta_qwk_str} |"
        )
    lines.append("")

    lines.append("### 改档方向归因详情\n")
    for rule_id, res in analysis["single_rule_ablation"].items():
        transitions = res.get("directional_transitions") or {}
        if transitions:
            lines.append(f"#### `{rule_id}`")
            for trans, counts in sorted(transitions.items()):
                lines.append(f"- **{trans}** (Total: {counts['total']}): Fixed={counts['fixed']}, Harmed={counts['harmed']}")
            lines.append("")

    lines.append("## 三、规则组合消融表 (Group Ablation)\n")
    lines.append("| Group Name | Effective | Fixed | Harmed | Net Gain | Marginal Prec | Without Acc | ΔAccuracy |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for grp_name, res in analysis["group_ablation"].items():
        prec_str = f"{res['marginal_precision']*100:.1f}%" if res['marginal_precision'] is not None else "N/A"
        delta_acc_str = f"{res['delta_accuracy']*100:+.2f}%"
        lines.append(
            f"| {grp_name} | {res['effective_count']} | {res['fixed']} | {res['harmed']} | {res['net_gain']} | {prec_str} | {res['accuracy_without_group']*100:.2f}% | **{delta_acc_str}** |"
        )
    lines.append("")

    lines.append("## 四、Stage 2 离线 Bypass 潜力分析\n")
    bypass = analysis["stage2_bypass_analysis"]
    lines.append(f"- **总题目数**: {bypass['total_records']}")
    lines.append(f"- **Stage 2 必须复核题数 (Candidate Count)**: {bypass['stage2_candidate_count']} ({bypass['stage2_candidate_rate']*100:.2f}%)")
    lines.append(f"- **理论上可节省 Stage 2 请求比例 (Potential Savings)**: **`{bypass['potential_api_savings_rate']*100:.2f}%`**")
    lines.append(f"- **候选触发原因统计**: `{bypass['candidate_reasons_breakdown']}`\n")

    lines.append("## 五、最终剩余错误结构特征分布\n")
    for err_name, feat_dist in analysis["residual_error_structure"].items():
        lines.append(f"### {err_name} (共 {feat_dist['count']} 题)")
        if feat_dist['count'] > 0:
            lines.append(f"- **平均原始正确率**: `{feat_dist['raw_predicted_accuracy_mean']}`")
            lines.append(f"- **step_count**: `{feat_dist['step_count']}`")
            lines.append(f"- **required_task_breadth**: `{feat_dist['required_task_breadth']}`")
            lines.append(f"- **model_explicitness**: `{feat_dist['model_explicitness']}`")
            lines.append(f"- **model_relation**: `{feat_dist['model_relation']}`")
            lines.append(f"- **reasoning_chain**: `{feat_dist['reasoning_chain']}`")
            lines.append(f"- **information_conversion**: `{feat_dist['information_conversion']}`")
            lines.append(f"- **calculation_model**: `{feat_dist['calculation_model']}`")
            lines.append(f"- **calculation_complexity**: `{feat_dist['calculation_complexity']}`")
            lines.append(f"- **experiment_requirement**: `{feat_dist['experiment_requirement']}`")
            lines.append(f"- **命中规则分布**: `{feat_dist['triggered_rules']}`")
        lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=["outputs/model_runs/high_chemistry_reference500_v21_run*.jsonl"],
        help="预测结果 jsonl 文件路径或 glob 模式",
    )
    parser.add_argument(
        "--labels",
        default="data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_labels.jsonl",
        help="Ground Truth 标签文件路径",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/model_runs/v21_rule_ablation_metrics.json",
        help="消融分析 JSON 结果保存路径",
    )
    parser.add_argument(
        "--output-md",
        default="output/doc/high_chemistry_v21_rule_ablation_report.md",
        help="消融分析 Markdown 报告保存路径",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    labels, rows = load_dataset(args.predictions, args.labels)
    if not rows:
        raise ValueError(f"未能根据路径加载到有效预测数据：{args.predictions}")

    print(f"成功加载 {len(rows)} 条样本，开始执行完整规则消融与归因分析...")
    analysis = run_ablation_analysis(rows)

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 报告已保存至：{out_p}")

    if args.output_md:
        out_md = Path(args.output_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        md_text = format_markdown_report(analysis)
        out_md.write_text(md_text, encoding="utf-8")
        print(f"Markdown 报告已保存至：{out_md}")

    print("\n" + "=" * 80)
    print(format_markdown_report(analysis))
    print("=" * 80)


if __name__ == "__main__":
    main()
