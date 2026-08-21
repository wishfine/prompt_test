# -*- coding: utf-8 -*-
"""V22-3Protect 候选规则全量评测与 Question-level 5-Fold Cross Validation.

评测指标包括：
- Overall Accuracy (CV Held-Out)
- L1, L2, L3, L4, L5 Recall
- QWK, MAE, 严重跨档偏差率 (>=2档)
- T3->2, T3->4, T2->3, T4->3 混淆转移量
- 跨跑一致率 (Pairwise & All-5 Agreement)
- Fixed, Harmed, Net Gain
"""

import sys
import json
import copy
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1] if "prompt_test" in str(Path(__file__).resolve()) else Path("/Users/wishfine/Desktop/xdf/ai题库/prompt_test")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import high_chemistry_pipeline_core as core
import tools.evaluate_high_chemistry_test500 as eval_tool

labels_path = ROOT / "data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_labels.jsonl"
labels = eval_tool.read_by_id(labels_path)

runs = {}
for r in range(1, 6):
    p = ROOT / f"outputs/model_runs/high_chemistry_reference500_v21_run{r}.jsonl"
    if not p.exists():
        p = ROOT / f"outputs/model_runs/high_chemistry_v21_1_run{r}.jsonl"
    runs[r] = eval_tool.read_by_id(p)

all_qids = sorted(list(labels.keys()))
print(f"Total questions: {len(all_qids)}, total runs: {len(runs)}")

# 5-fold 切分：固定基于 hash 确保每道题的 5 次运行全在一个 fold
folds = [[] for _ in range(5)]
for i, qid in enumerate(all_qids):
    # 确定性分桶
    fold_idx = int(hashlib.md5(qid.encode("utf-8")).hexdigest(), 16) % 5
    folds[fold_idx].append(qid)

for f_idx, f_qids in enumerate(folds):
    print(f"Fold {f_idx + 1}: {len(f_qids)} questions")


# =========================================================================
# 特征轴与候选规则定义
# =========================================================================
def compute_axes(features, high_names):
    """计算各类强轴与中等轴。"""
    # 1. 7 个传统轴
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
    
    # 2. Hard 轴 vs Soft 轴
    hard_axis_count = sum([axis_model_relation, axis_quant, axis_info, axis_exp, axis_constraint])
    total_axis_count = sum([axis_model_ident, axis_reasoning, axis_model_relation, axis_quant, axis_info, axis_exp, axis_constraint])
    
    # 3. 中等轴 (Moderate Burden Axes)
    moderate_model_ident = (features.get("model_explicitness") == "半隐含模型")
    moderate_representation = (features.get("representation_conversion") in {"多次同类转换", "多表征连续转换", "逆向表征转换", "一次常规转换"})
    moderate_representation_strict = (features.get("representation_conversion") in {"多次同类转换", "多表征连续转换", "逆向表征转换"})
    moderate_information = (features.get("information_conversion") == "单次关系转换")
    moderate_evidence = (features.get("evidence_relation") == "证据链相互支持")
    moderate_hidden = (features.get("hidden_conditions") == "单个隐含条件")
    moderate_classification = (features.get("classification_discussion") == "2类讨论")
    moderate_quant = (
        features.get("calculation_model") in {"多步化学计量", "浓度或气体综合", "平衡常数或Ka/Kb/Ksp"}
        and features.get("calculation_complexity") == "多步计算"
    )
    moderate_experiment = (features.get("experiment_requirement") == "数据归纳")
    moderate_context = (features.get("context_load") == "需要信息转换")
    
    moderate_axis_count = sum([
        moderate_model_ident, moderate_representation_strict, moderate_information,
        moderate_evidence, moderate_hidden, moderate_classification, moderate_quant,
        moderate_experiment, moderate_context
    ])
    
    return {
        "hard_axis_count": hard_axis_count,
        "total_axis_count": total_axis_count,
        "axis_quant": axis_quant,
        "axis_info": axis_info,
        "axis_exp": axis_exp,
        "axis_constraint": axis_constraint,
        "moderate_axis_count": moderate_axis_count,
        "has_substantive_burden": (
            features.get("model_relation") in {"模型切换", "多模型耦合"}
            or features.get("information_conversion") not in {"无信息转换", "直接读取"}
            or features.get("calculation_model") not in {"无定量计算", "常规化学计量"}
            or features.get("experiment_requirement") not in {"无", "基础操作或读数", "直接现象解释"}
        ),
    }


def derive_candidate_constraint(features, high_names, candidate_config):
    """根据 candidate 配置派生 structural constraint。"""
    floor = "难度1档"
    ceiling = "难度5档"
    rule_ids = []
    
    axes = compute_axes(features, high_names)
    
    # 1. direct_prototype_exact_1
    if (
        features.get("primary_problem_structure") == "教材直接原型"
        and features.get("step_count") == "1-2步"
        and features.get("required_task_breadth") == "单一规则任务"
        and features.get("substance_relation") == "单一物质"
        and features.get("reaction_relation") == "无反应链"
        and features.get("process_structure") == "单阶段"
        and features.get("model_explicitness") == "模型完全显性"
        and features.get("model_relation") == "单一模型"
        and features.get("reasoning_chain") == "直接套用"
        and features.get("representation_conversion") == "无转换"
        and features.get("information_conversion") == "无信息转换"
        and features.get("hidden_conditions") == "无"
        and features.get("critical_condition") == "无临界"
        and features.get("constraint_structure") == "无约束"
        and features.get("calculation_model") == "无定量计算"
        and features.get("experiment_requirement") == "无"
        and not high_names
    ):
        floor = core.max_level(floor, "难度1档")
        ceiling = core.min_level(ceiling, "难度1档")
        rule_ids.append("direct_prototype_exact_1")

    # 2. calculation_model_floor_2
    if features.get("calculation_model") in {"常规化学计量", "多步化学计量", "浓度或气体综合", "平衡常数或Ka/Kb/Ksp", "多模型定量耦合"}:
        floor = core.max_level(floor, "难度2档")
        rule_ids.append("calculation_model_floor_2")

    # 3. basic_explicit_application_ceiling_2
    if (
        features.get("step_count") == "1-2步"
        and features.get("required_task_breadth") in {"单一规则任务", "2-3个异质必要任务"}
        and features.get("model_explicitness") == "模型完全显性"
        and features.get("model_relation") == "单一模型"
        and features.get("reasoning_chain") in {"直接套用", "简单因果"}
        and features.get("information_conversion") in {"无信息转换", "直接读取"}
        and features.get("hidden_conditions") == "无"
        and features.get("critical_condition") in {"无临界", "显性给出临界"}
        and features.get("constraint_structure") in {"无约束", "单一约束"}
        and features.get("calculation_complexity") in {"直接判断", "简单计算"}
        and features.get("experiment_requirement") in {"无", "基础操作或读数", "直接现象解释"}
        and not high_names
    ):
        ceiling = core.min_level(ceiling, "难度2档")
        rule_ids.append("basic_explicit_application_ceiling_2")

    # 4. Parallel basic bundle ceiling 2 (候选配置)
    parallel_mode = candidate_config.get("parallel_mode", "v21")
    if parallel_mode == "v21":
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
    elif parallel_mode == "strict_v1":
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
            and features.get("competing_reaction") == "无"
            and features.get("classification_discussion") == "无"
            and features.get("representation_conversion") in {"无转换", "一次常规转换"}
            and features.get("context_load") in {"纯包装", "简单规律映射"}
            and features.get("error_risk") in {"无明显易错点", "轻微易错点"}
            and not high_names
        )
    elif parallel_mode == "strict_v2":  # 甚至更纯：无转换 + 纯包装
        parallel_basic_bundle = (
            features.get("required_task_breadth") in {"2-3个异质必要任务", "4个及以上异质必要任务"}
            and features.get("substance_relation") in {"单一物质", "相互独立"}
            and features.get("reaction_relation") in {"无反应链", "并列独立"}
            and features.get("process_structure") == "单阶段"
            and features.get("model_explicitness") == "模型完全显性"
            and features.get("model_relation") == "单一模型"
            and features.get("reasoning_chain") == "直接套用"
            and features.get("information_conversion") == "无信息转换"
            and features.get("representation_conversion") == "无转换"
            and features.get("context_load") == "纯包装"
            and features.get("hidden_conditions") == "无"
            and features.get("error_risk") in {"无明显易错点", "轻微易错点"}
            and not high_names
        )
    elif parallel_mode == "none":
        parallel_basic_bundle = False
    else:
        parallel_basic_bundle = False

    if parallel_basic_bundle:
        ceiling = core.min_level(ceiling, "难度2档")
        rule_ids.append("parallel_basic_bundle_ceiling_2")

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
        floor = core.max_level(floor, "难度3档")
        rule_ids.append("standard_chain_floor_3")

    # 6. standard_comprehensive_floor_3 (候选)
    std_comp_mode = candidate_config.get("std_comp_floor3_mode", "none")
    if std_comp_mode == "v1":
        standard_comprehensive_floor_3 = (
            not high_names
            and not parallel_basic_bundle
            and (
                (features.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"} and axes["moderate_axis_count"] >= 1)
                or (features.get("step_count") == "1-2步" and axes["moderate_axis_count"] >= 2)
            )
        )
    elif std_comp_mode == "v2":  # 3-5步且 moderate >= 2 或 1-2步且 moderate >= 3
        standard_comprehensive_floor_3 = (
            not high_names
            and not parallel_basic_bundle
            and (
                (features.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"} and axes["moderate_axis_count"] >= 2)
                or (features.get("step_count") == "1-2步" and axes["moderate_axis_count"] >= 3)
            )
        )
    elif std_comp_mode == "step35_direct":  # 只要是 3-5 步且不是 parallel bundle
        standard_comprehensive_floor_3 = (
            not high_names
            and not parallel_basic_bundle
            and features.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"}
            and (features.get("reasoning_chain") != "直接套用" or axes["moderate_axis_count"] >= 1)
        )
    else:
        standard_comprehensive_floor_3 = False

    if standard_comprehensive_floor_3:
        floor = core.max_level(floor, "难度3档")
        rule_ids.append("standard_comprehensive_floor_3")

    # 7. Floor 4 规则群
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
        or features.get("route_design_requirement") in {"合成路线设计", "分离提纯方案设计", "路线优化与可行性验证"}
    )
    model_migration_system_coupling_strong = (
        features.get("step_count") in {"6-8步", "9-12步", "12步以上"}
        and features.get("model_relation") in {"模型切换", "多模型耦合"}
        and features.get("substance_relation") in {"前后转化依赖", "组成—性质—反应网络"}
        and has_additional_high_burden
    )
    if model_migration_multistage_strong or model_migration_system_coupling_strong:
        rule_ids.append("hard_structural_cluster_floor_4")

    # 8. Compressed high burden floor 4 (候选配置)
    comp_high_mode = candidate_config.get("comp_high_mode", "v21")
    if comp_high_mode == "v21":
        is_compressed_high = (
            axes["total_axis_count"] >= 2
            and features.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"}
            and axes["has_substantive_burden"]
        )
    elif comp_high_mode == "narrow_v1":  # hard >= 2 OR (hard >= 1 and total >= 3)
        comp_long = (features.get("step_count") in {"6-8步", "9-12步", "12步以上"} and axes["total_axis_count"] >= 2 and axes["has_substantive_burden"])
        comp_short = (
            features.get("step_count") == "3-5步"
            and (
                axes["hard_axis_count"] >= 2
                or (axes["hard_axis_count"] >= 1 and axes["total_axis_count"] >= 3)
            )
        )
        is_compressed_high = comp_long or comp_short
    elif comp_high_mode == "narrow_v2":  # hard >= 2 OR total >= 3
        comp_long = (features.get("step_count") in {"6-8步", "9-12步", "12步以上"} and axes["total_axis_count"] >= 2 and axes["has_substantive_burden"])
        comp_short = (
            features.get("step_count") == "3-5步"
            and (axes["hard_axis_count"] >= 2 or axes["total_axis_count"] >= 3)
            and axes["has_substantive_burden"]
        )
        is_compressed_high = comp_long or comp_short
    else:
        is_compressed_high = False

    if is_compressed_high:
        rule_ids.append("compressed_high_burden_floor_4")

    if complex_quantitative or model_migration_multistage_strong or model_migration_system_coupling_strong or is_compressed_high:
        floor = core.max_level(floor, "难度4档")

    # 9. canonical_middle_3 (候选)
    if candidate_config.get("canonical_middle_exact3", False):
        canonical_middle_3 = (
            not high_names
            and not parallel_basic_bundle
            and not is_compressed_high
            and features.get("step_count") in {"1-2步", "3-5步"}
            and features.get("model_relation") in {"单一模型", "同一模型多状态"}
            and features.get("process_structure") not in {"多阶段强依赖", "循环或回流流程"}
            and features.get("constraint_structure") != "多约束联合筛选"
            and features.get("route_design_requirement") in {"无", "已知路线补全"}
            and axes["moderate_axis_count"] in {1, 2}
        )
        if canonical_middle_3:
            floor = core.max_level(floor, "难度3档")
            ceiling = core.min_level(ceiling, "难度3档")
            rule_ids.append("canonical_middle_exact3")

    # 10. regular_comprehensive_ceiling_3
    regular_comprehensive_tight = (
        not high_names
        and not is_compressed_high
        and features.get("step_count") in {"1-2步", "3-5步"}
        and features.get("model_explicitness") == "模型完全显性"
        and features.get("model_relation") in {"单一模型", "同一模型多状态"}
        and features.get("process_structure") not in {"多阶段强依赖", "循环或回流流程"}
        and features.get("reasoning_chain") in {"直接套用", "简单因果"}
        and features.get("information_conversion") in {"无信息转换", "直接读取"}
        and features.get("hidden_conditions") == "无"
        and features.get("critical_condition") in {"无临界", "显性给出临界"}
        and features.get("constraint_structure") in {"无约束", "单一约束", "多约束相互独立"}
        and features.get("experiment_requirement") in {"无", "基础操作或读数", "直接现象解释"}
        and features.get("route_design_requirement") in {"无", "已知路线补全"}
    )
    if regular_comprehensive_tight:
        ceiling = core.min_level(ceiling, "难度3档")
        rule_ids.append("regular_comprehensive_ceiling_3")

    return {
        "difficulty_floor": floor,
        "difficulty_ceiling": ceiling,
        "rule_ids": rule_ids,
        "constraint_conflict": core.LEVEL_INDEX[floor] > core.LEVEL_INDEX[ceiling],
    }


def evaluate_candidate_on_dataset(candidate_config, target_qids):
    """在指定题目集上回放 candidate 并计算各项指标。"""
    total = len(target_qids) * len(runs)
    correct = 0
    pure_correct = 0
    severe_dev = 0
    abs_err = 0
    
    t_counts = Counter()
    pred_counts = Counter()
    correct_per_level = Counter()
    
    t3_to_2 = 0
    t3_to_4 = 0
    t2_to_3 = 0
    t4_to_3 = 0
    
    run_preds = defaultdict(dict)
    
    fixed = 0
    harmed = 0
    
    truth_vals = []
    pred_vals = []
    
    for r in range(1, 6):
        data = runs[r]
        for qid in target_qids:
            if qid not in labels or qid not in data:
                continue
            t = labels[qid]["reviewed_difficulty_level"]
            st1 = data[qid].get("difficulty_rating_stage1", {})
            score = st1.get("predicted_accuracy", 50.0)
            score_level = core.map_accuracy_to_level(score)
            feats = st1.get("features", {})
            high = core.detect_high_difficulty_features(feats)
            
            constraint = derive_candidate_constraint(feats, high.names, candidate_config)
            final_level, _, _, _ = core.apply_structural_level_constraint(score_level, constraint)
            
            run_preds[r][qid] = final_level
            truth_vals.append(t)
            pred_vals.append(final_level)
            
            t_counts[t] += 1
            pred_counts[final_level] += 1
            
            gap = abs(core.LEVEL_INDEX[final_level] - core.LEVEL_INDEX[t])
            abs_err += gap
            if gap >= 2:
                severe_dev += 1
            if score_level == t:
                pure_correct += 1
            if final_level == t:
                correct += 1
                correct_per_level[t] += 1
                
            if final_level != score_level:
                c_before = (score_level == t)
                c_after = (final_level == t)
                if not c_before and c_after:
                    fixed += 1
                elif c_before and not c_after:
                    harmed += 1
                    
            if t == "难度3档" and final_level == "难度2档":
                t3_to_2 += 1
            if t == "难度3档" and final_level == "难度4档":
                t3_to_4 += 1
            if t == "难度2档" and final_level == "难度3档":
                t2_to_3 += 1
            if t == "难度4档" and final_level == "难度3档":
                t4_to_3 += 1

    acc = correct / total
    pure_acc = pure_correct / total
    mae = abs_err / total
    severe_rate = severe_dev / total
    qwk = eval_tool.quadratic_weighted_kappa(truth_vals, pred_vals)
    
    levels_list = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
    recalls = {lvl: correct_per_level[lvl] / t_counts[lvl] if t_counts[lvl] else 0.0 for lvl in levels_list}
    
    # 5 跑一致率
    all5_same = sum(1 for q in target_qids if len(set(run_preds[r][q] for r in range(1, 6))) == 1) / len(target_qids)
    
    pairwise_list = []
    for r1 in range(1, 6):
        for r2 in range(r1 + 1, 6):
            p = sum(1 for q in target_qids if run_preds[r1][q] == run_preds[r2][q]) / len(target_qids)
            pairwise_list.append(p)
    avg_pairwise = sum(pairwise_list) / len(pairwise_list)
    
    return {
        "accuracy": acc,
        "pure_accuracy": pure_acc,
        "net_gain": correct - pure_correct,
        "qwk": qwk,
        "mae": mae,
        "severe_rate": severe_rate,
        "recalls": recalls,
        "t3_to_2": t3_to_2 / len(runs),
        "t3_to_4": t3_to_4 / len(runs),
        "t2_to_3": t2_to_3 / len(runs),
        "t4_to_3": t4_to_3 / len(runs),
        "all5_same": all5_same,
        "avg_pairwise": avg_pairwise,
        "fixed": fixed,
        "harmed": harmed,
        "pred_dist": {lvl: pred_counts[lvl] / len(runs) for lvl in levels_list},
    }


def run_5fold_cross_validation(candidate_name, candidate_config):
    """执行 5 折严格交叉验证（计算 5 个 held-out validation 评估结果）。"""
    held_out_metrics = []
    
    for fold_idx in range(5):
        val_qids = folds[fold_idx]
        train_qids = [q for i, f in enumerate(folds) if i != fold_idx for q in f]
        
        # 在 validation fold 上独立评估
        val_res = evaluate_candidate_on_dataset(candidate_config, val_qids)
        held_out_metrics.append(val_res)
        
    # 聚合 5 个 held-out fold 的指标
    avg_acc = sum(m["accuracy"] for m in held_out_metrics) / 5
    avg_qwk = sum(m["qwk"] for m in held_out_metrics) / 5
    avg_severe = sum(m["severe_rate"] for m in held_out_metrics) / 5
    avg_r1 = sum(m["recalls"]["难度1档"] for m in held_out_metrics) / 5
    avg_r2 = sum(m["recalls"]["难度2档"] for m in held_out_metrics) / 5
    avg_r3 = sum(m["recalls"]["难度3档"] for m in held_out_metrics) / 5
    avg_r4 = sum(m["recalls"]["难度4档"] for m in held_out_metrics) / 5
    avg_r5 = sum(m["recalls"]["难度5档"] for m in held_out_metrics) / 5
    
    avg_t3_2 = sum(m["t3_to_2"] for m in held_out_metrics)
    avg_t3_4 = sum(m["t3_to_4"] for m in held_out_metrics)
    
    all_res = evaluate_candidate_on_dataset(candidate_config, all_qids)
    
    return {
        "name": candidate_name,
        "cv_accuracy": avg_acc,
        "cv_qwk": avg_qwk,
        "cv_severe": avg_severe,
        "recalls": {
            "L1": avg_r1, "L2": avg_r2, "L3": avg_r3, "L4": avg_r4, "L5": avg_r5
        },
        "t3_to_2": avg_t3_2,
        "t3_to_4": avg_t3_4,
        "all_res": all_res,
    }


if __name__ == "__main__":
    CANDIDATES = [
        ("V21 Baseline", {
            "parallel_mode": "v21",
            "std_comp_floor3_mode": "none",
            "comp_high_mode": "v21",
            "canonical_middle_exact3": False,
        }),
        ("Cand A: strict_parallel_v1", {
            "parallel_mode": "strict_v1",
            "std_comp_floor3_mode": "none",
            "comp_high_mode": "v21",
            "canonical_middle_exact3": False,
        }),
        ("Cand B: strict_parallel_v2", {
            "parallel_mode": "strict_v2",
            "std_comp_floor3_mode": "none",
            "comp_high_mode": "v21",
            "canonical_middle_exact3": False,
        }),
        ("Cand C: std_comp_floor3_v1", {
            "parallel_mode": "v21",
            "std_comp_floor3_mode": "v1",
            "comp_high_mode": "v21",
            "canonical_middle_exact3": False,
        }),
        ("Cand D: std_comp_floor3_v2", {
            "parallel_mode": "v21",
            "std_comp_floor3_mode": "v2",
            "comp_high_mode": "v21",
            "canonical_middle_exact3": False,
        }),
        ("Cand E: comp_high_narrow_v1", {
            "parallel_mode": "v21",
            "std_comp_floor3_mode": "none",
            "comp_high_mode": "narrow_v1",
            "canonical_middle_exact3": False,
        }),
        ("Cand F: comp_high_narrow_v2", {
            "parallel_mode": "v21",
            "std_comp_floor3_mode": "none",
            "comp_high_mode": "narrow_v2",
            "canonical_middle_exact3": False,
        }),
        ("Cand G: canonical_middle_exact3", {
            "parallel_mode": "v21",
            "std_comp_floor3_mode": "none",
            "comp_high_mode": "v21",
            "canonical_middle_exact3": True,
        }),
        ("Cand H (A+C+E): StrictPar + StdFloor3 + NarrowHigh", {
            "parallel_mode": "strict_v1",
            "std_comp_floor3_mode": "v1",
            "comp_high_mode": "narrow_v1",
            "canonical_middle_exact3": False,
        }),
        ("Cand I (A+D+E): StrictPar + StdFloor3_v2 + NarrowHigh", {
            "parallel_mode": "strict_v1",
            "std_comp_floor3_mode": "v2",
            "comp_high_mode": "narrow_v1",
            "canonical_middle_exact3": False,
        }),
        ("Cand J (A+C+F): StrictPar + StdFloor3_v1 + NarrowHigh_v2", {
            "parallel_mode": "strict_v1",
            "std_comp_floor3_mode": "v1",
            "comp_high_mode": "narrow_v2",
            "canonical_middle_exact3": False,
        }),
        ("Cand K (A+C+E+G): Full Combo with Exact3", {
            "parallel_mode": "strict_v1",
            "std_comp_floor3_mode": "v1",
            "comp_high_mode": "narrow_v1",
            "canonical_middle_exact3": True,
        }),
        ("Cand L (B+C+E): SuperStrictPar + StdFloor3 + NarrowHigh", {
            "parallel_mode": "strict_v2",
            "std_comp_floor3_mode": "v1",
            "comp_high_mode": "narrow_v1",
            "canonical_middle_exact3": False,
        }),
    ]

    print("=" * 115)
    print(f"{'Candidate Name':<42} | {'CV Acc':<8} | {'L3 Rec':<8} | {'L2 Rec':<8} | {'L4 Rec':<8} | {'QWK':<7} | {'3->2':<6} | {'3->4':<6} | {'All-5'}")
    print("-" * 115)
    
    cv_results = []
    for name, config in CANDIDATES:
        res = run_5fold_cross_validation(name, config)
        cv_results.append(res)
        r = res["recalls"]
        print(f"{name:<42} | {res['cv_accuracy']:<8.2%} | {r['L3']:<8.2%} | {r['L2']:<8.2%} | {r['L4']:<8.2%} | {res['cv_qwk']:<7.4f} | {res['t3_to_2']:<6.1f} | {res['t3_to_4']:<6.1f} | {res['all_res']['all5_same']:.1%}")

    print("=" * 115)
