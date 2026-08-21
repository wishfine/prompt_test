# -*- coding: utf-8 -*-
"""高中化学 V22 Candidate Mining 与 Question-level 5-Fold 边界交叉验证工具.

离线评测以下候选：
  - Cand A (Baseline V21): 原始 V21 规则
  - Cand B (Strict Parallel Basic No-Error-Risk): 严格并列基础多任务 (删除 error_risk 门槛)
  - Cand C (B + Grouped Moderate Floor 3): 5组独立中等负担 + standard_comprehensive_floor_3
  - Cand D (C + Canonical Middle Exact 3): C + 典型中档 exact 3 规则 (仅 evaluator 离线)
  - Cand E (C + Floor 4 Rescue): C + 4档 rescue 规则 (3-5步+半隐含+多层因果+真实依赖)
  - Cand F (C + D + E): 协同全集候选
  - Cand G (C + E): 仅正向 Floor 协同候选
"""

import sys
import copy
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import tools.evaluate_high_chemistry_test500 as eval_tool
import high_chemistry_pipeline_core as core

LABELS_PATH = ROOT / "data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_labels.jsonl"
RUNS_DIR = ROOT / "outputs/model_runs"


def compute_qwk(actuals: list[str], preds: list[str]) -> float:
    """计算 Quadratic Weighted Kappa (QWK)."""
    n = len(actuals)
    if n == 0:
        return 0.0
    k = 5
    o_matrix = [[0] * k for _ in range(k)]
    for a, p in zip(actuals, preds):
        i = core.LEVEL_INDEX[a] - 1
        j = core.LEVEL_INDEX[p] - 1
        o_matrix[i][j] += 1
    
    act_counts = [0] * k
    pred_counts = [0] * k
    for i in range(k):
        for j in range(k):
            act_counts[i] += o_matrix[i][j]
            pred_counts[j] += o_matrix[i][j]
            
    e_matrix = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            e_matrix[i][j] = (act_counts[i] * pred_counts[j]) / n
            
    w_matrix = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(k):
            w_matrix[i][j] = ((i - j) ** 2) / ((k - 1) ** 2)
            
    numerator = sum(w_matrix[i][j] * o_matrix[i][j] for i in range(k) for j in range(k))
    denominator = sum(w_matrix[i][j] * e_matrix[i][j] for i in range(k) for j in range(k))
    if denominator == 0:
        return 1.0
    return 1.0 - (numerator / denominator)


def get_5fold_splits(qids: list[str]) -> list[tuple[list[str], list[str]]]:
    """根据 QID 的哈希值生成严格确定性 5-Fold 分割 (保证同一题目所有 run 在同一 fold)."""
    folds: list[list[str]] = [[] for _ in range(5)]
    for qid in sorted(qids):
        h = int(hashlib.md5(qid.encode("utf-8")).hexdigest(), 16)
        fold_idx = h % 5
        folds[fold_idx].append(qid)
    
    splits = []
    for f in range(5):
        val_qids = set(folds[f])
        train_qids = [q for q in qids if q not in val_qids]
        splits.append((train_qids, sorted(list(val_qids))))
    return splits


def build_candidate_constraint_fn(cand_name: str) -> Callable[[dict[str, Any], list[str], float], dict[str, Any]]:
    """根据候选名称返回结构约束派生函数."""

    def constraint_fn(features: dict[str, Any], high_names: list[str], raw_score: float) -> dict[str, Any]:
        floor = "难度1档"
        ceiling = "难度5档"
        rule_ids = []

        # 基础轴分析
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

        hard_axis_count = sum([axis_model_relation, axis_quant, axis_info, axis_exp, axis_constraint])
        total_axis_count = sum([axis_model_ident, axis_reasoning, axis_model_relation, axis_quant, axis_info, axis_exp, axis_constraint])

        has_substantive = (
            features.get("model_relation") in {"模型切换", "多模型耦合"}
            or features.get("information_conversion") not in {"无信息转换", "直接读取"}
            or features.get("calculation_model") not in {"无定量计算", "常规化学计量"}
            or features.get("experiment_requirement") not in {"无", "基础操作或读数", "直接现象解释"}
        )

        has_real_dependency = (
            features.get("reaction_relation") in {"显性顺序衔接", "前后反应强依赖", "多路径反应网络"}
            or features.get("subquestion_dependency") == "后问依赖前问"
            or (features.get("shared_model_across_subquestions") is True and features.get("process_structure") != "单阶段")
            or features.get("substance_relation") in {"前后转化依赖", "组成—性质—反应网络"}
            or features.get("process_structure") in {"多阶段显性流程", "多阶段强依赖", "循环或回流流程"}
        )

        # 5 大独立中等负担组
        moderate_model_condition = (
            features.get("model_explicitness") == "半隐含模型"
            or features.get("hidden_conditions") == "单个隐含条件"
        )
        moderate_information_group = (
            features.get("representation_conversion") in {
                "多次同类转换", "多表征连续转换", "逆向表征转换",
            }
            or features.get("information_conversion") == "单次关系转换"
            or features.get("evidence_relation") == "证据链相互支持"
            or features.get("context_load") == "需要信息转换"
        )
        moderate_classification = (
            features.get("classification_discussion") == "2类讨论"
        )
        moderate_quantitative = (
            features.get("calculation_model") in {
                "多步化学计量", "浓度或气体综合", "平衡常数或Ka/Kb/Ksp",
            }
            and features.get("calculation_complexity") == "多步计算"
        )
        moderate_experimental = (
            features.get("experiment_requirement") == "数据归纳"
        )
        moderate_group_count = sum([
            moderate_model_condition,
            moderate_information_group,
            moderate_classification,
            moderate_quantitative,
            moderate_experimental,
        ])

        # 1. direct_prototype_exact_1
        if (
            features.get("primary_problem_structure") == "教材直接原型"
            and features.get("step_count") == "1-2步"
            and features.get("required_task_breadth") == "单一规则任务"
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
        basic_explicit_app = (
            features.get("step_count") == "1-2步"
            and features.get("model_explicitness") == "模型完全显性"
            and features.get("reasoning_chain") in {"直接套用", "简单因果"}
            and features.get("information_conversion") in {"无信息转换", "直接读取"}
            and features.get("hidden_conditions") == "无"
            and features.get("critical_condition") in {"无临界", "显性给出临界"}
            and features.get("constraint_structure") in {"无约束", "单一约束"}
            and features.get("calculation_complexity") in {"直接判断", "简单计算"}
            and features.get("experiment_requirement") in {"无", "基础操作或读数", "直接现象解释"}
            and not high_names
        )
        if basic_explicit_app:
            ceiling = core.min_level(ceiling, "难度2档")
            rule_ids.append("basic_explicit_application_ceiling_2")

        # 4. parallel_basic_bundle ceiling 2
        if cand_name == "Cand A (V21 Baseline)":
            parallel_rule = (
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
            if parallel_rule:
                ceiling = core.min_level(ceiling, "难度2档")
                rule_ids.append("parallel_basic_bundle_ceiling_2")
        else:
            # Cand B ~ G: strict parallel basic (no error_risk)
            parallel_basic_bundle_strict = (
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
                and not high_names
            )
            if parallel_basic_bundle_strict:
                ceiling = core.min_level(ceiling, "难度2档")
                rule_ids.append("parallel_basic_bundle_strict_ceiling_2")

        # 5. standard_comprehensive_floor_3 (Cand C ~ G)
        if cand_name not in {"Cand A (V21 Baseline)", "Cand B (Strict Parallel Only)"}:
            standard_comprehensive_floor_3 = (
                not high_names
                and not basic_explicit_app
                and not parallel_basic_bundle_strict
                and (
                    (
                        features.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"}
                        and moderate_group_count >= 1
                    )
                    or (
                        features.get("step_count") == "1-2步"
                        and moderate_group_count >= 2
                    )
                )
            )
            if standard_comprehensive_floor_3:
                floor = core.max_level(floor, "难度3档")
                rule_ids.append("standard_comprehensive_floor_3")

        # 6. standard_chain_floor_3 (all cands)
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

        # 7. Canonical Middle Exact 3 (Cand D, F)
        if "Canonical Middle Exact 3" in cand_name or cand_name == "Cand F (C+D+E All)":
            canonical_middle_3 = (
                not high_names
                and not basic_explicit_app
                and not parallel_basic_bundle_strict
                and features.get("step_count") in {"1-2步", "3-5步"}
                and moderate_group_count in {1, 2}
                and features.get("model_relation") in {"单一模型", "同一模型多状态"}
                and features.get("model_explicitness") != "需要自主建模"
                and features.get("reasoning_chain") != "逆向推理或临界分析"
                and features.get("process_structure") not in {"多阶段强依赖", "循环或回流流程"}
                and features.get("constraint_structure") != "多约束联合筛选"
                and features.get("route_design_requirement") in {"无", "已知路线补全"}
            )
            if canonical_middle_3:
                floor = core.max_level(floor, "难度3档")
                ceiling = core.min_level(ceiling, "难度3档")
                rule_ids.append("canonical_middle_exact3")

        # 8. Floor 4 Rules
        # V21 baseline compressed high burden
        is_compressed_high = (
            total_axis_count >= 2
            and features.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"}
            and has_substantive
        )
        if is_compressed_high:
            floor = core.max_level(floor, "难度4档")
            rule_ids.append("compressed_high_burden_floor_4")

        # Cand E, F, G: model_reasoning_coupled_floor4 (Teacher 4 Rescue)
        if "Floor 4 Rescue" in cand_name or cand_name in {"Cand E (C + Floor 4 Rescue)", "Cand F (C+D+E All)", "Cand G (C + Floor 4 Rescue)"}:
            model_reasoning_coupled_floor4 = (
                features.get("step_count") in {"3-5步", "6-8步"}
                and features.get("model_explicitness") in {"半隐含模型", "隐含模型", "需要自主建模"}
                and features.get("reasoning_chain") in {"多层因果", "逆向推理或临界分析"}
                and has_real_dependency
            )
            if model_reasoning_coupled_floor4:
                floor = core.max_level(floor, "难度4档")
                rule_ids.append("model_reasoning_coupled_floor4")

        if "复杂定量、参数或范围" in high_names:
            floor = core.max_level(floor, "难度4档")
            rule_ids.append("complex_quantitative_floor_4")

        # 9. regular_comprehensive_ceiling_3
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

    return constraint_fn


def evaluate_dataset(
    dataset_name: str,
    run_records_dict: dict[int, dict[str, Any]],
    labels: dict[str, Any],
    candidate_names: list[str],
) -> None:
    """在指定数据集上对各候选执行全量 Replay 与 5-Fold 交叉验证."""
    print("\n" + "=" * 115)
    print(f"评测数据集: {dataset_name} ({len(run_records_dict)} 跑 x 500 题)")
    print("=" * 115)
    
    qids = sorted(list(labels.keys()))
    splits = get_5fold_splits(qids)
    
    table_rows = []

    for cand_name in candidate_names:
        c_fn = build_candidate_constraint_fn(cand_name)
        
        # 1. 全量评测
        all_actuals = []
        all_preds = []
        t3_to_2_cnt = 0
        t3_to_4_cnt = 0
        t4_to_3_cnt = 0
        
        for r, records in run_records_dict.items():
            for qid in qids:
                t = labels[qid]["reviewed_difficulty_level"]
                st1 = records[qid].get("difficulty_rating_stage1", {})
                score = st1.get("predicted_accuracy", 50.0)
                score_lvl = core.map_accuracy_to_level(score)
                feats = st1.get("features", {})
                high = core.detect_high_difficulty_features(feats)
                
                c = c_fn(feats, high.names, score)
                final_lvl, _, _, _ = core.apply_structural_level_constraint(score_lvl, c)
                
                all_actuals.append(t)
                all_preds.append(final_lvl)
                
                if t == "难度3档" and final_lvl == "难度2档":
                    t3_to_2_cnt += 1
                elif t == "难度3档" and final_lvl == "难度4档":
                    t3_to_4_cnt += 1
                elif t == "难度4档" and final_lvl == "难度3档":
                    t4_to_3_cnt += 1
                    
        acc = sum(1 for a, p in zip(all_actuals, all_preds) if a == p) / len(all_actuals)
        qwk = compute_qwk(all_actuals, all_preds)
        
        # 分档召回
        t_counts = Counter(all_actuals)
        c_counts = Counter(a for a, p in zip(all_actuals, all_preds) if a == p)
        rec1 = c_counts["难度1档"] / t_counts["难度1档"]
        rec2 = c_counts["难度2档"] / t_counts["难度2档"]
        rec3 = c_counts["难度3档"] / t_counts["难度3档"]
        rec4 = c_counts["难度4档"] / t_counts["难度4档"]
        rec5 = c_counts["难度5档"] / t_counts["难度5档"]
        
        # 2. 5-Fold Cross Validation (Held-out Mean)
        fold_accs = []
        for train_q, val_q in splits:
            val_actuals = []
            val_preds = []
            for r, records in run_records_dict.items():
                for qid in val_q:
                    t = labels[qid]["reviewed_difficulty_level"]
                    st1 = records[qid].get("difficulty_rating_stage1", {})
                    score = st1.get("predicted_accuracy", 50.0)
                    score_lvl = core.map_accuracy_to_level(score)
                    feats = st1.get("features", {})
                    high = core.detect_high_difficulty_features(feats)
                    c = c_fn(feats, high.names, score)
                    final_lvl, _, _, _ = core.apply_structural_level_constraint(score_lvl, c)
                    val_actuals.append(t)
                    val_preds.append(final_lvl)
            f_acc = sum(1 for a, p in zip(val_actuals, val_preds) if a == p) / len(val_actuals)
            fold_accs.append(f_acc)
            
        cv_mean = sum(fold_accs) / len(fold_accs)
        cv_min = min(fold_accs)
        cv_max = max(fold_accs)
        
        runs_count = len(run_records_dict)
        table_rows.append({
            "cand": cand_name,
            "acc": acc,
            "cv_mean": cv_mean,
            "cv_range": f"[{cv_min:.2%}, {cv_max:.2%}]",
            "rec1": rec1,
            "rec2": rec2,
            "rec3": rec3,
            "rec4": rec4,
            "rec5": rec5,
            "qwk": qwk,
            "t3_to_2": t3_to_2_cnt / runs_count,
            "t3_to_4": t3_to_4_cnt / runs_count,
            "t4_to_3": t4_to_3_cnt / runs_count,
        })

    # 输出表格
    header = f"{'候选名称':<36} | {'全量Acc':<8} | {'5-Fold CV':<10} | {'L2 Rec':<8} | {'L3 Rec':<8} | {'L4 Rec':<8} | {'QWK':<7} | {'3->2/跑':<8} | {'3->4/跑':<8} | {'4->3/跑':<8}"
    print(header)
    print("-" * len(header))
    for r in table_rows:
        print(
            f"{r['cand']:<36} | {r['acc']:<8.2%} | {r['cv_mean']:<10.2%} | {r['rec2']:<8.2%} | {r['rec3']:<8.2%} | {r['rec4']:<8.2%} | {r['qwk']:<7.4f} | {r['t3_to_2']:<8.1f} | {r['t3_to_4']:<8.1f} | {r['t4_to_3']:<8.1f}"
        )


def main():
    labels = eval_tool.read_by_id(LABELS_PATH)
    
    # 1. 读取 V21 5 跑
    v21_runs = {}
    for r in range(1, 6):
        p = RUNS_DIR / f"high_chemistry_reference500_v21_run{r}.jsonl"
        if not p.exists():
            p = RUNS_DIR / f"high_chemistry_v21_1_run{r}.jsonl"
        if p.exists():
            v21_runs[r] = eval_tool.read_by_id(p)
            
    # 2. 读取 V22 3 跑
    v22_runs = {}
    for r in range(1, 4):
        p = RUNS_DIR / f"high_chemistry_v22_run{r}.jsonl"
        if p.exists():
            v22_runs[r] = eval_tool.read_by_id(p)
            
    candidates = [
        "Cand A (V21 Baseline)",
        "Cand B (Strict Parallel Only)",
        "Cand C (B + Grouped Moderate Floor3)",
        "Cand D (C + Canonical Middle Exact3)",
        "Cand E (C + Floor 4 Rescue)",
        "Cand F (C+D+E All)",
    ]
    
    if v21_runs:
        evaluate_dataset("V21 历史数据 (5 跑)", v21_runs, labels, candidates)
        
    if v22_runs:
        evaluate_dataset("V22 生产数据 (3 跑)", v22_runs, labels, candidates)


if __name__ == "__main__":
    main()
