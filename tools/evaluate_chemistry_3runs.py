# -*- coding: utf-8 -*-
"""高中化学 3 跑与单跑全量评测工具（含详细 Mistake 错题分布、分档指标与 Stage1/Stage2 对比）."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import tools.evaluate_high_chemistry_test500 as eval_tool
import high_chemistry_pipeline_core as core

LEVELS = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]


def compute_metrics(labels: dict[str, Any], runs: dict[int, dict[str, Any]], field: str = "final_difficulty_level"):
    qids = sorted(list(labels.keys()))
    results = {}
    
    for r, records in runs.items():
        actuals = []
        preds = []
        confusion = {t: {p: 0 for p in LEVELS} for t in LEVELS}
        
        for qid in qids:
            if qid not in records:
                continue
            t = labels[qid]["reviewed_difficulty_level"]
            p = records[qid].get(field)
            if t in LEVELS and p in LEVELS:
                actuals.append(t)
                preds.append(p)
                confusion[t][p] += 1
                
        n = len(actuals)
        exact = sum(1 for a, p in zip(actuals, preds) if a == p)
        acc = exact / n if n else 0.0
        mistakes = n - exact
        qwk = eval_tool.quadratic_weighted_kappa(actuals, preds) or 0.0
        
        # 分档
        per_level = {}
        for lvl in LEVELS:
            t_cnt = sum(confusion[lvl].values())
            p_cnt = sum(confusion[other][lvl] for other in LEVELS)
            cor = confusion[lvl][lvl]
            rec = cor / t_cnt if t_cnt else 0.0
            prec = cor / p_cnt if p_cnt else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            per_level[lvl] = {"target": t_cnt, "pred": p_cnt, "correct": cor, "recall": rec, "precision": prec, "f1": f1}
            
        results[r] = {
            "n": n,
            "exact": exact,
            "mistakes": mistakes,
            "acc": acc,
            "qwk": qwk,
            "confusion": confusion,
            "per_level": per_level,
            "t3_to_2": confusion["难度3档"]["难度2档"],
            "t3_to_4": confusion["难度3档"]["难度4档"],
            "t4_to_3": confusion["难度4档"]["难度3档"],
            "t2_to_3": confusion["难度2档"]["难度3档"],
            "actuals": actuals,
            "preds": preds,
        }
    return results


def compute_majority_vote(labels: dict[str, Any], runs: dict[int, dict[str, Any]], field: str = "final_difficulty_level"):
    qids = sorted(list(labels.keys()))
    actuals = []
    preds = []
    confusion = {t: {p: 0 for p in LEVELS} for t in LEVELS}
    
    for qid in qids:
        t = labels[qid]["reviewed_difficulty_level"]
        p_votes = [runs[r][qid].get(field) for r in runs if qid in runs[r]]
        p_votes = [p for p in p_votes if p in LEVELS]
        if not p_votes:
            continue
        # 多数票投票
        c = Counter(p_votes)
        maj_pred = c.most_common(1)[0][0]
        actuals.append(t)
        preds.append(maj_pred)
        confusion[t][maj_pred] += 1
        
    n = len(actuals)
    exact = sum(1 for a, p in zip(actuals, preds) if a == p)
    acc = exact / n if n else 0.0
    mistakes = n - exact
    qwk = eval_tool.quadratic_weighted_kappa(actuals, preds) or 0.0
    
    per_level = {}
    for lvl in LEVELS:
        t_cnt = sum(confusion[lvl].values())
        p_cnt = sum(confusion[other][lvl] for other in LEVELS)
        cor = confusion[lvl][lvl]
        rec = cor / t_cnt if t_cnt else 0.0
        prec = cor / p_cnt if p_cnt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_level[lvl] = {"target": t_cnt, "pred": p_cnt, "correct": cor, "recall": rec, "precision": prec, "f1": f1}
        
    return {
        "n": n,
        "exact": exact,
        "mistakes": mistakes,
        "acc": acc,
        "qwk": qwk,
        "confusion": confusion,
        "per_level": per_level,
        "t3_to_2": confusion["难度3档"]["难度2档"],
        "t3_to_4": confusion["难度3档"]["难度4档"],
        "t4_to_3": confusion["难度4档"]["难度3档"],
        "t2_to_3": confusion["难度2档"]["难度3档"],
    }


def print_report(labels: dict[str, Any], runs: dict[int, dict[str, Any]], show_mistakes: bool = True):
    final_res = compute_metrics(labels, runs, "final_difficulty_level")
    step1_res = compute_metrics(labels, runs, "difficulty_level_step1")
    
    print("\n" + "=" * 105)
    print("                      高中化学评测大盘报告（单跑明细 + 3 跑汇总 + Mistake 错题分析）")
    print("=" * 105)
    
    # 1. 单跑总览
    header = f"{'Run':<6} | {'Final Acc':<11} | {'Final Mistakes':<14} | {'Step1 Acc':<11} | {'QWK':<7} | {'3->2 错题':<9} | {'3->4 错题':<9} | {'4->3 错题':<9} | {'2->3 错题':<9}"
    print(header)
    print("-" * len(header))
    for r in sorted(runs.keys()):
        f = final_res[r]
        s = step1_res[r]
        print(f"Run {r:<2} | {f['acc']:<11.2%} | {f['mistakes']:<4} ({f['mistakes']/f['n']:.1%})  | {s['acc']:<11.2%} | {f['qwk']:<7.4f} | {f['t3_to_2']:<9} | {f['t3_to_4']:<9} | {f['t4_to_3']:<9} | {f['t2_to_3']:<9}")
        
    # 平均
    num_runs = len(runs)
    avg_acc = sum(final_res[r]["acc"] for r in runs) / num_runs
    avg_mistakes = sum(final_res[r]["mistakes"] for r in runs) / num_runs
    avg_s1_acc = sum(step1_res[r]["acc"] for r in runs) / num_runs
    avg_qwk = sum(final_res[r]["qwk"] for r in runs) / num_runs
    avg_3_2 = sum(final_res[r]["t3_to_2"] for r in runs) / num_runs
    avg_3_4 = sum(final_res[r]["t3_to_4"] for r in runs) / num_runs
    avg_4_3 = sum(final_res[r]["t4_to_3"] for r in runs) / num_runs
    avg_2_3 = sum(final_res[r]["t2_to_3"] for r in runs) / num_runs
    
    print("-" * len(header))
    print(f"{'Mean':<6} | {avg_acc:<11.2%} | {avg_mistakes:<4.1f} ({avg_mistakes/500:.1%})  | {avg_s1_acc:<11.2%} | {avg_qwk:<7.4f} | {avg_3_2:<9.1f} | {avg_3_4:<9.1f} | {avg_4_3:<9.1f} | {avg_2_3:<9.1f}")
    
    # 多数票
    if num_runs >= 2:
        maj = compute_majority_vote(labels, runs, "final_difficulty_level")
        print(f"{'Vote':<6} | {maj['acc']:<11.2%} | {maj['mistakes']:<4} ({maj['mistakes']/maj['n']:.1%})  | {'--':<11} | {maj['qwk']:<7.4f} | {maj['t3_to_2']:<9} | {maj['t3_to_4']:<9} | {maj['t4_to_3']:<9} | {maj['t2_to_3']:<9}")
    print("=" * 105)
    
    # 2. 分档 Recall / Precision / F1 指标
    print("\n【分档表现指标（3 跑平均 vs 多数票）】")
    lvl_header = f"{'难度档位':<8} | {'Teacher题数':<10} | {'3跑平均Recall':<14} | {'3跑平均Prec':<13} | {'3跑平均F1':<11} | {'Vote Recall':<12} | {'Vote Prec':<11}"
    print(lvl_header)
    print("-" * len(lvl_header))
    for lvl in LEVELS:
        rec_avg = sum(final_res[r]["per_level"][lvl]["recall"] for r in runs) / num_runs
        prec_avg = sum(final_res[r]["per_level"][lvl]["precision"] for r in runs) / num_runs
        f1_avg = sum(final_res[r]["per_level"][lvl]["f1"] for r in runs) / num_runs
        t_cnt = final_res[1]["per_level"][lvl]["target"] if 1 in final_res else 0
        if num_runs >= 2:
            maj = compute_majority_vote(labels, runs, "final_difficulty_level")
            v_rec = maj["per_level"][lvl]["recall"]
            v_prec = maj["per_level"][lvl]["precision"]
            print(f"{lvl:<8} | {t_cnt:<10} | {rec_avg:<14.2%} | {prec_avg:<13.2%} | {f1_avg:<11.2%} | {v_rec:<12.2%} | {v_prec:<11.2%}")
        else:
            print(f"{lvl:<8} | {t_cnt:<10} | {rec_avg:<14.2%} | {prec_avg:<13.2%} | {f1_avg:<11.2%} | {'--':<12} | {'--':<11}")
    print("-" * len(lvl_header))

    # 3. 详细混淆矩阵 (3 跑累计)
    print("\n【全量混淆矩阵 (Confusion Matrix - 3跑累计)】")
    cum_conf = {t: {p: sum(final_res[r]["confusion"][t][p] for r in runs) for p in LEVELS} for t in LEVELS}
    c_header = f"{'真实\\预测':<10} | " + " | ".join(f"{lvl:<7}" for lvl in LEVELS) + " | 合计"
    print(c_header)
    print("-" * len(c_header))
    for t in LEVELS:
        row_str = f"{t:<10} | " + " | ".join(f"{cum_conf[t][p]:<7}" for p in LEVELS) + f" | {sum(cum_conf[t].values())}"
        print(row_str)
    print("-" * len(c_header))

    if show_mistakes:
        print("\n" + "=" * 105)
        print("【核心 Mistake 错题分类与题型归因统计 (基于 3 跑汇总)】")
        print("=" * 105)
        print(f"1. 3档 -> 2档 (高估正确率/基础化): 3 跑平均每跑 {avg_3_2:.1f} 题 (占总真3档 {avg_3_2/114:.1%})")
        print(f"2. 3档 -> 4档 (低估正确率/过度升档): 3 跑平均每跑 {avg_3_4:.1f} 题 (占总真3档 {avg_3_4/114:.1%})")
        print(f"3. 4档 -> 3档 (低估综合度/漏升4档): 3 跑平均每跑 {avg_4_3:.1f} 题 (占总真4档 {avg_4_3/172:.1%})")
        print(f"4. 2档 -> 3档 (高估综合度/误升3档): 3 跑平均每跑 {avg_2_3:.1f} 题 (占总真2档 {avg_2_3/154:.1%})")
        print("=" * 105)


def main():
    parser = argparse.ArgumentParser(description="高中化学评测与 Mistake 分析工具")
    parser.add_argument("-l", "--labels", default="data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_labels.jsonl")
    parser.add_argument("-r", "--runs", nargs="+", help="单跑或多跑输出 jsonl 文件路径")
    parser.add_argument("--no-mistakes", action="store_true", help="不显示详细错题分析")
    args = parser.parse_args()
    
    labels_path = Path(args.labels)
    if not labels_path.exists():
        labels_path = ROOT / args.labels
    labels = eval_tool.read_by_id(labels_path)
    
    runs = {}
    if args.runs:
        for idx, r_path_str in enumerate(args.runs, 1):
            p = Path(r_path_str)
            if not p.exists():
                p = ROOT / r_path_str
            runs[idx] = eval_tool.read_by_id(p)
    else:
        # 默认尝试读取 v22 run 1, 2, 3
        for idx in (1, 2, 3):
            p = ROOT / f"outputs/model_runs/high_chemistry_v22_run{idx}.jsonl"
            if p.exists():
                runs[idx] = eval_tool.read_by_id(p)
                
    if not runs:
        print("错误：未找到有效的 model runs jsonl 文件！请通过 -r 指定文件路径。")
        sys.exit(1)
        
    print_report(labels, runs, show_mistakes=not args.no_mistakes)


if __name__ == "__main__":
    main()
