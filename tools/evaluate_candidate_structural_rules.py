# -*- coding: utf-8 -*-
"""高中化学结构规则离线真实 Replay 评测工具。

直接使用真实标注与历史运行记录进行确定性回放，统计每个规则与整套 Pipeline 的：
- 触发次数 (Trigger Count)
- 纠错次数 (Fixed)
- 误伤次数 (Harmed)
- 无效次数 (Neutral)
- 净收益 (Net Gain)
- 胜率/精确率 (Precision = Fixed / (Fixed + Harmed))
- 真实 Question ID 清单
"""

import copy
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parents[1] if "prompt_test" in str(Path(__file__).resolve()) else Path("/Users/wishfine/Desktop/xdf/ai题库/prompt_test")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import high_chemistry_pipeline_core as core
import tools.evaluate_high_chemistry_test500 as eval_tool

labels_path = ROOT / "data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_labels.jsonl"
blind_path = ROOT / "data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_blind.jsonl"

labels = eval_tool.read_by_id(labels_path)
blind = eval_tool.read_by_id(blind_path)

runs = {}
for r in [1, 2, 3]:
    p = ROOT / f"outputs/model_runs/high_chemistry_reference500_v20_run{r}.jsonl"
    runs[r] = eval_tool.read_by_id(p)


def evaluate_single_rule(rule_name: str, rule_type: str, target_level: str, condition_fn):
    """评估单个候选规则在历史数据上的独立表现。"""
    print(f"\n=======================================================")
    print(f"RULE REPLAY: {rule_name} ({rule_type} -> {target_level})")
    print(f"=======================================================")
    
    total_fixed = 0
    total_harmed = 0
    total_neutral = 0
    total_triggered = 0
    
    run_details = []
    
    for r in [1, 2, 3]:
        data = runs[r]
        r_fixed = []
        r_harmed = []
        r_neutral = []
        r_triggered = 0
        
        for qid, row in data.items():
            if qid not in labels:
                continue
            t = labels[qid]["reviewed_difficulty_level"]
            st1 = row.get("difficulty_rating_stage1", {})
            score = st1.get("predicted_accuracy", 50.0)
            score_level = core.map_accuracy_to_level(score)
            feats = st1.get("features", {})
            high = core.detect_high_difficulty_features(feats)
            
            if condition_fn(feats, high.names):
                # 检查该规则是否会改变 score_level
                adjusted_level = score_level
                if rule_type == "ceiling":
                    adjusted_level = core.min_level(score_level, target_level)
                elif rule_type == "floor":
                    adjusted_level = core.max_level(score_level, target_level)
                    
                if adjusted_level != score_level:
                    r_triggered += 1
                    c_before = (score_level == t)
                    c_after = (adjusted_level == t)
                    if not c_before and c_after:
                        r_fixed.append(qid)
                    elif c_before and not c_after:
                        r_harmed.append(qid)
                    else:
                        r_neutral.append(qid)
                        
        denom = len(r_fixed) + len(r_harmed)
        win_rate = f"{len(r_fixed)/denom:.1%}" if denom else "N/A"
        print(f"Run {r}: Triggered = {r_triggered:2d} | Fixed = {len(r_fixed):2d} | Harmed = {len(r_harmed):2d} | Neutral = {len(r_neutral):2d} | Net = {len(r_fixed)-len(r_harmed):+2d} | WinRate = {win_rate}")
        run_details.append({
            "run": r, "fixed": r_fixed, "harmed": r_harmed, "neutral": r_neutral
        })
        total_fixed += len(r_fixed)
        total_harmed += len(r_harmed)
        total_neutral += len(r_neutral)
        total_triggered += r_triggered
        
    denom_all = total_fixed + total_harmed
    win_rate_all = f"{total_fixed/denom_all:.1%}" if denom_all else "N/A"
    print(f"--> TOTAL (3 Runs): Triggered = {total_triggered:2d} | Fixed = {total_fixed:2d} | Harmed = {total_harmed:2d} | Neutral = {total_neutral:2d} | Net = {total_fixed-total_harmed:+2d} | WinRate = {win_rate_all}")
    
    # 打印部分具有代表性的真实 Question ID
    print("\nSample Real Question IDs (Run 1):")
    if run_details[0]["fixed"]:
        print(f"  Fixed sample ({len(run_details[0]['fixed'])} total): {run_details[0]['fixed'][:5]}")
    if run_details[0]["harmed"]:
        print(f"  Harmed sample ({len(run_details[0]['harmed'])} total): {run_details[0]['harmed'][:5]}")
    return {
        "rule_name": rule_name,
        "total_triggered": total_triggered,
        "total_fixed": total_fixed,
        "total_harmed": total_harmed,
        "total_neutral": total_neutral,
        "net": total_fixed - total_harmed,
        "win_rate": total_fixed / denom_all if denom_all else 0.0,
        "run_details": run_details,
    }


def evaluate_full_v21_pipeline():
    """评估完整 V21 核心逻辑在三跑数据上的总表现。"""
    print(f"\n=======================================================")
    print(f"FULL PIPELINE REPLAY: V21 CORE (All Rules Combined)")
    print(f"=======================================================")
    
    rule_counts = defaultdict(lambda: {"fixed": 0, "harmed": 0, "neutral": 0, "total": 0})
    accs = []
    pure_accs = []
    
    for r in [1, 2, 3]:
        data = runs[r]
        total = len(data)
        correct = 0
        pure_correct = 0
        dist = Counter()
        
        for qid, row in data.items():
            if qid not in labels:
                continue
            t = labels[qid]["reviewed_difficulty_level"]
            st1 = row.get("difficulty_rating_stage1", {})
            score = st1.get("predicted_accuracy", 50.0)
            score_level = core.map_accuracy_to_level(score)
            feats = st1.get("features", {})
            high = core.detect_high_difficulty_features(feats)
            
            # 使用生产 core 派生约束并应用
            constraint = core.derive_structural_level_constraint(feats, high.names)
            final_level, action, conflict, severe = core.apply_structural_level_constraint(score_level, constraint)
            
            dist[final_level] += 1
            if score_level == t:
                pure_correct += 1
            if final_level == t:
                correct += 1
                
            if final_level != score_level:
                c_before = (score_level == t)
                c_after = (final_level == t)
                outcome = "neutral"
                if not c_before and c_after:
                    outcome = "fixed"
                elif c_before and not c_after:
                    outcome = "harmed"
                for rule in constraint["rule_ids"]:
                    rule_counts[rule][outcome] += 1
                    rule_counts[rule]["total"] += 1
                    
        acc = correct / total
        pure_acc = pure_correct / total
        accs.append(acc)
        pure_accs.append(pure_acc)
        print(f"Run {r}: Accuracy = {acc:.2%} (Pure Score: {pure_acc:.2%}, Net: {correct-pure_correct:+d})")
        print(f"  Distribution: {dict(dist)}")
        
    avg_acc = sum(accs) / len(accs)
    avg_pure = sum(pure_accs) / len(pure_accs)
    print(f"\n--> V21 Full Pipeline Average Accuracy: {avg_acc:.2%} (Pure Score Avg: {avg_pure:.2%}, Avg Gain: {avg_acc-avg_pure:+.2%})")
    
    print("\nRule Performance Summary across 3 Runs in Full Pipeline:")
    for rule, st in sorted(rule_counts.items(), key=lambda x: x[1]["total"], reverse=True):
        denom = st["fixed"] + st["harmed"]
        win_rate = f"{st['fixed']/denom:.1%}" if denom else "N/A"
        print(f"  {rule:<40} | Fixed: {st['fixed']:<3} | Harmed: {st['harmed']:<3} | Net: {st['fixed']-st['harmed']:<+3} | Total: {st['total']:<3} | WinRate: {win_rate}")


if __name__ == "__main__":
    # 1. 独立评测 parallel_basic_bundle_ceiling_2 (含并列独立修复)
    parallel_bundle_cond = lambda f, h: (
        f.get("required_task_breadth") in {"2-3个异质必要任务", "4个及以上异质必要任务"}
        and f.get("substance_relation") in {"单一物质", "相互独立"}
        and f.get("reaction_relation") in {"无反应链", "并列独立"}
        and f.get("process_structure") == "单阶段"
        and f.get("subquestion_dependency") != "后问依赖前问"
        and not f.get("shared_model_across_subquestions", False)
        and f.get("model_explicitness") == "模型完全显性"
        and f.get("model_relation") in {"单一模型", "同一模型多状态"}
        and f.get("reasoning_chain") in {"直接套用", "简单因果"}
        and f.get("information_conversion") in {"无信息转换", "直接读取"}
        and f.get("evidence_relation") in {"直接给定", "单证据对应", "多证据独立"}
        and f.get("hidden_conditions") == "无"
        and f.get("critical_condition") in {"无临界", "显性给出临界"}
        and f.get("constraint_structure") in {"无约束", "单一约束", "多约束相互独立"}
        and f.get("calculation_model") in {"无定量计算", "常规化学计量"}
        and f.get("calculation_complexity") in {"直接判断", "简单计算"}
        and f.get("experiment_requirement") in {"无", "基础操作或读数", "直接现象解释"}
        and f.get("route_design_requirement") in {"无", "已知路线补全"}
        and not h
    )
    evaluate_single_rule("parallel_basic_bundle_ceiling_2", "ceiling", "难度2档", parallel_bundle_cond)

    # 2. 独立评测 compressed_high_burden_floor_4 (含自主建模修复)
    def compressed_high_cond(f, h):
        axis_model_ident = f.get("model_explicitness") in {"半隐含模型", "隐含模型", "需要自主建模"}
        axis_reasoning = f.get("reasoning_chain") in {"多层因果", "逆向推理或临界分析"}
        axis_model_relation = f.get("model_relation") in {"模型切换", "多模型耦合"}
        axis_quant = (
            f.get("calculation_model") in {"平衡常数或Ka/Kb/Ksp", "多模型定量耦合", "多步化学计量"}
            and f.get("calculation_complexity") in {"多方程联立", "参数或范围计算", "多步计算"}
        )
        axis_info = (
            f.get("information_conversion") in {"多源信息联合转换", "流程或图谱反推"}
            or (f.get("information_conversion") == "单次关系转换" and f.get("evidence_relation") in {"证据链相互支持", "证据冲突需排除"})
        )
        axis_exp = (
            f.get("experiment_requirement") in {"控制变量或异常分析", "方案设计或误差反演"}
            or f.get("route_design_requirement") in {"合成路线设计", "分离提纯方案设计", "路线优化与可行性验证"}
        )
        axis_constraint = (
            f.get("constraint_structure") == "多约束联合筛选"
            or f.get("critical_condition") in {"需要推导过量不足边界", "隐含终点或有效区间"}
        )
        axes = [axis_model_ident, axis_reasoning, axis_model_relation, axis_quant, axis_info, axis_exp, axis_constraint]
        return (
            sum(axes) >= 2
            and f.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"}
            and (
                f.get("model_relation") in {"模型切换", "多模型耦合"}
                or f.get("information_conversion") not in {"无信息转换", "直接读取"}
                or f.get("calculation_model") not in {"无定量计算", "常规化学计量"}
                or f.get("experiment_requirement") not in {"无", "基础操作或读数", "直接现象解释"}
            )
        )
    evaluate_single_rule("compressed_high_burden_floor_4", "floor", "难度4档", compressed_high_cond)

    # 3. 评测完整生产 V21 Pipeline
    evaluate_full_v21_pipeline()
