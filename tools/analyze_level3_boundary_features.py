# -*- coding: utf-8 -*-
"""高中化学 3 档边界特征模式挖掘工具 (Teacher 2 vs 3, Teacher 3 vs 4) - 纯 Python 实现.

通过聚合 5 跑数据的模式 (Mode) 特征，挖掘跨跑稳定且最能区分 2/3 和 3/4 档位的关键特征组合。
"""

import sys
import json
import math
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

print(f"Loaded {len(labels)} labels and {len(runs)} runs.")

# 1. 对每道题聚合 5 跑的 mode features
aggregated_questions = {}
for qid in labels:
    t = labels[qid]["reviewed_difficulty_level"]
    feats_list = [runs[r][qid].get("difficulty_rating_stage1", {}).get("features", {}) for r in runs if qid in runs[r]]
    scores_list = [runs[r][qid].get("difficulty_rating_stage1", {}).get("predicted_accuracy", 50.0) for r in runs if qid in runs[r]]
    
    mode_feats = {}
    for k in core.REQUIRED_FEATURE_FIELDS:
        vals = [f.get(k) for f in feats_list if f.get(k) is not None]
        if vals:
            if isinstance(vals[0], list):
                tuple_vals = [tuple(sorted(v)) for v in vals]
                mode_feats[k] = list(Counter(tuple_vals).most_common(1)[0][0])
            else:
                mode_feats[k] = Counter(vals).most_common(1)[0][0]
                
    aggregated_questions[qid] = {
        "qid": qid,
        "teacher_level": t,
        "mode_features": mode_feats,
        "avg_score": sum(scores_list) / len(scores_list) if scores_list else 50.0,
    }


def analyze_binary_boundary(t_class_a: str, t_class_b: str):
    """分析两档之间的边界特征。"""
    print(f"\n" + "=" * 75)
    print(f"BOUNDARY ANALYSIS: {t_class_a} vs {t_class_b}")
    print(f"=" * 75)
    
    subset = [q for q in aggregated_questions.values() if q["teacher_level"] in {t_class_a, t_class_b}]
    count_a = sum(1 for q in subset if q["teacher_level"] == t_class_a)
    count_b = sum(1 for q in subset if q["teacher_level"] == t_class_b)
    print(f"Total samples: {len(subset)} ({t_class_a}: {count_a}, {t_class_b}: {count_b})")
    
    categorical_keys = [
        "step_count", "required_task_breadth", "model_explicitness", "model_relation",
        "reasoning_chain", "representation_conversion", "information_conversion",
        "evidence_relation", "hidden_conditions", "critical_condition",
        "classification_discussion", "constraint_structure", "calculation_model",
        "calculation_complexity", "experiment_requirement", "route_design_requirement",
        "competing_reaction", "process_structure", "context_load", "error_risk"
    ]
    
    # 1. 单特征信号分析
    feature_signals = []
    for key in categorical_keys:
        val_counts_a = Counter(q["mode_features"].get(key) for q in subset if q["teacher_level"] == t_class_a)
        val_counts_b = Counter(q["mode_features"].get(key) for q in subset if q["teacher_level"] == t_class_b)
        all_vals = set(val_counts_a.keys()) | set(val_counts_b.keys())
        
        for v in all_vals:
            if v is None:
                continue
            ca = val_counts_a[v]
            cb = val_counts_b[v]
            tot = ca + cb
            if tot >= 12:  # 至少 12 个样本
                prob_b = cb / tot
                prob_a = ca / tot
                feature_signals.append({
                    "feature": key,
                    "value": v,
                    "count_total": tot,
                    f"count_{t_class_a}": ca,
                    f"count_{t_class_b}": cb,
                    f"p_{t_class_a}": prob_a,
                    f"p_{t_class_b}": prob_b,
                    "skew": abs(prob_b - prob_a),
                })
                
    feature_signals.sort(key=lambda x: x["skew"], reverse=True)
    print(f"\nTop 15 Most Discriminative Single Feature Values (Support >= 12):")
    print(f"{'Feature':<26} | {'Value':<22} | {'Total':<6} | {t_class_a:<8} | {t_class_b:<8} | {f'P({t_class_b})':<8}")
    print("-" * 90)
    for sig in feature_signals[:15]:
        print(f"{sig['feature']:<26} | {str(sig['value']):<22} | {sig['count_total']:<6} | {sig[f'count_{t_class_a}']:<8} | {sig[f'count_{t_class_b}']:<8} | {sig[f'p_{t_class_b}']:.1%}")

    # 2. 纯 Python 决策树桩 (Decision Stump) 组合挖掘
    print(f"\nTop 8 2-Feature Combinations for {t_class_a} vs {t_class_b}:")
    combos = []
    for i in range(len(categorical_keys)):
        for j in range(i + 1, len(categorical_keys)):
            k1, k2 = categorical_keys[i], categorical_keys[j]
            pair_counts_a = Counter((q["mode_features"].get(k1), q["mode_features"].get(k2)) for q in subset if q["teacher_level"] == t_class_a)
            pair_counts_b = Counter((q["mode_features"].get(k1), q["mode_features"].get(k2)) for q in subset if q["teacher_level"] == t_class_b)
            all_pairs = set(pair_counts_a.keys()) | set(pair_counts_b.keys())
            for p in all_pairs:
                ca = pair_counts_a[p]
                cb = pair_counts_b[p]
                tot = ca + cb
                if tot >= 15:
                    combos.append({
                        "k1": k1, "v1": p[0],
                        "k2": k2, "v2": p[1],
                        "total": tot,
                        f"count_{t_class_a}": ca,
                        f"count_{t_class_b}": cb,
                        f"p_{t_class_b}": cb / tot,
                        "skew": abs(cb/tot - ca/tot)
                    })
    combos.sort(key=lambda x: (x["skew"], x["total"]), reverse=True)
    print(f"{'Condition 1':<30} | {'Condition 2':<30} | {'Total':<6} | {t_class_a:<8} | {t_class_b:<8} | {f'P({t_class_b})':<8}")
    print("-" * 105)
    seen = set()
    count = 0
    for c in combos:
        desc = f"{c['k1']}={c['v1']} & {c['k2']}={c['v2']}"
        if desc in seen:
            continue
        seen.add(desc)
        print(f"{c['k1']}={str(c['v1']):<15} | {c['k2']}={str(c['v2']):<15} | {c['total']:<6} | {c[f'count_{t_class_a}']:<8} | {c[f'count_{t_class_b}']:<8} | {c[f'p_{t_class_b}']:.1%}")
        count += 1
        if count >= 8:
            break


if __name__ == "__main__":
    analyze_binary_boundary("难度2档", "难度3档")
    analyze_binary_boundary("难度3档", "难度4档")
