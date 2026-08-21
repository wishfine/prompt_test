import copy
import json
import sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path("/Users/wishfine/Desktop/xdf/ai题库/prompt_test")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import high_chemistry_pipeline_core as core
import tools.evaluate_high_chemistry_test500 as eval_tool

labels_path = ROOT / "data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_labels.jsonl"
labels = eval_tool.read_by_id(labels_path)

runs = {}
for r in [1, 2, 3]:
    p = ROOT / f"outputs/model_runs/high_chemistry_reference500_v20_run{r}.jsonl"
    runs[r] = eval_tool.read_by_id(p)


def derive_candidate_v21_constraint(features: dict, high_names: list[str], options: dict) -> dict:
    """可配置的 V21 候选结构约束生成函数。"""
    floor = "难度1档"
    ceiling = "难度5档"
    rule_ids: list[str] = []
    evidence: list[str] = []

    # 1. direct_prototype (保留)
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
        return {
            "difficulty_floor": "难度1档",
            "difficulty_ceiling": "难度1档",
            "rule_ids": ["direct_prototype_exact_1"],
            "evidence": ["纯单知识点单步直接套用原型"],
            "confidence": "高",
            "constraint_conflict": False,
        }

    # 2. calculation_model_floor_2 (保留)
    if features.get("calculation_model") in {
        "常规化学计量",
        "多步化学计量",
        "平衡常数或Ka/Kb/Ksp",
        "多模型定量耦合",
    }:
        floor = core.max_level(floor, "难度2档")
        rule_ids.append("calculation_model_floor_2")
        evidence.append(f"定量计算模型({features.get('calculation_model')})")

    # 3. multiple_required_tasks_floor_2 (选项：保留 / 暂停)
    if options.get("enable_multiple_tasks_floor_2", False):
        if features.get("required_task_breadth") in {
            "2-3个异质必要任务",
            "4个及以上异质必要任务",
            "多问递进任务链",
        }:
            floor = core.max_level(floor, "难度2档")
            rule_ids.append("multiple_required_tasks_floor_2")

    # 4. basic_explicit_app ceiling 2 (保留)
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
        ceiling = core.min_level(ceiling, "难度2档")
        rule_ids.append("basic_explicit_application_ceiling_2")
        evidence.append("1-2步显性基础应用，无高难无强依赖")

    # 5. Candidate A: parallel_basic_bundle_ceiling_2
    if options.get("enable_parallel_basic_bundle_ceiling_2", False):
        parallel_basic_bundle = (
            features.get("required_task_breadth") in {"2-3个异质必要任务", "4个及以上异质必要任务"}
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
            ceiling = core.min_level(ceiling, "难度2档")
            rule_ids.append("parallel_basic_bundle_ceiling_2")
            evidence.append("并列基础多任务(模型显性/直接套用/无隐含条件)")

    # 6. standard_chain_floor_3 (选项：原版 / 收紧版)
    if options.get("standard_chain_version") == "v20":
        standard_chain = (
            features.get("step_count") in {"3-5步", "6-8步", "9-12步", "12步以上"}
            and features.get("substance_relation") in {"同一反应体系", "前后转化依赖", "组成—性质—反应网络"}
            and (
                features.get("reaction_count") in {"2-3个", "4-6个", "7个及以上"}
                or features.get("calculation_model") in {"常规化学计量", "多步化学计量", "平衡常数或Ka/Kb/Ksp", "多模型定量耦合"}
                or features.get("information_conversion") not in {"无信息转换", "直接读取"}
                or features.get("experiment_requirement") not in {"无", "基础操作或读数"}
                or features.get("reasoning_chain") in {"多层因果", "逆向推理或临界分析"}
            )
        )
        if standard_chain:
            floor = core.max_level(floor, "难度3档")
            rule_ids.append("standard_chain_floor_3")
    elif options.get("standard_chain_version") == "tightened":
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

    # 7. Candidate B: compressed_high_burden_floor_4
    if options.get("enable_compressed_high_burden_floor_4", False):
        # 强轴定义
        axis_model_ident = features.get("model_explicitness") in {"半隐含模型", "隐含模型"}
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

        strong_axes = [
            axis_model_ident,
            axis_reasoning,
            axis_model_relation,
            axis_quant,
            axis_info,
            axis_exp,
            axis_constraint,
        ]
        strong_axis_count = sum(1 for a in strong_axes if a)
        
        # 选项配置
        min_axes = options.get("compressed_min_axes", 2)
        if strong_axis_count >= min_axes and (features.get("model_explicitness") in {"半隐含模型", "隐含模型"} or strong_axis_count >= 3):
            floor = core.max_level(floor, "难度4档")
            rule_ids.append("compressed_high_burden_floor_4")
            evidence.append(f"短链高密度负担(命中{strong_axis_count}个强负担轴)")

    # 8. hard_structural_cluster_floor_4 (保留)
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
    if (
        complex_quantitative
        or model_migration_multistage_strong
        or model_migration_system_coupling_strong
    ):
        floor = core.max_level(floor, "难度4档")
        rule_ids.append("hard_structural_cluster_floor_4")

    # 9. regular_comprehensive_ceiling_3 (选项：原版 / 明显收紧版)
    if options.get("regular_comprehensive_version") == "v20":
        regular_comprehensive = (
            features.get("step_count") in {"1-2步", "3-5步"}
            and not high_names
            and features.get("model_relation") in {"单一模型", "同一模型多状态"}
            and features.get("process_structure") not in {"多阶段强依赖", "循环或回流流程"}
            and features.get("calculation_model") not in {"多模型定量耦合"}
            and features.get("information_conversion") not in {"多源信息联合转换", "流程或图谱反推"}
            and features.get("experiment_requirement") not in {"控制变量或异常分析", "方案设计或误差反演"}
            and features.get("route_design_requirement") not in {"合成路线设计", "分离提纯方案设计", "路线优化与可行性验证"}
            and features.get("constraint_structure") != "多约束联合筛选"
            and not (complex_quantitative or model_migration_multistage_strong or model_migration_system_coupling_strong)
        )
        if regular_comprehensive:
            ceiling = core.min_level(ceiling, "难度3档")
            rule_ids.append("regular_comprehensive_ceiling_3")
    elif options.get("regular_comprehensive_version") == "tightened":
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
            and not (complex_quantitative or model_migration_multistage_strong or model_migration_system_coupling_strong)
        )
        if regular_comprehensive_tight:
            ceiling = core.min_level(ceiling, "难度3档")
            rule_ids.append("regular_comprehensive_ceiling_3")

    conflict = core.LEVEL_INDEX[floor] > core.LEVEL_INDEX[ceiling]
    return {
        "difficulty_floor": floor,
        "difficulty_ceiling": ceiling,
        "rule_ids": rule_ids,
        "evidence": evidence,
        "confidence": "高" if not conflict else "低",
        "constraint_conflict": conflict,
    }


def replay_evaluation(name: str, options: dict):
    print(f"\n=======================================================")
    print(f"REPLAY CONFIG: {name}")
    print(f"=======================================================")
    
    rule_counts_all = defaultdict(lambda: {"fixed": 0, "harmed": 0, "neutral": 0, "total": 0})
    accs = []
    pure_accs = []
    
    for r in [1, 2, 3]:
        data = runs[r]
        total = len(data)
        correct_count = 0
        pure_correct_count = 0
        
        overrides = 0
        r_fixed = 0
        r_harmed = 0
        r_neutral = 0
        
        for qid, row in data.items():
            if qid not in labels:
                continue
            t = labels[qid]["reviewed_difficulty_level"]
            st1 = row.get("difficulty_rating_stage1", {})
            raw_acc = st1.get("predicted_accuracy", 50.0)
            score_level = core.map_accuracy_to_level(raw_acc)
            feats = st1.get("features", {})
            high = core.detect_high_difficulty_features(feats)
            
            # Replay candidate constraint
            constraint = derive_candidate_v21_constraint(feats, high.names, options)
            final_level, action, conflict, severe = core.apply_structural_level_constraint(score_level, constraint)
            
            if score_level == t:
                pure_correct_count += 1
            if final_level == t:
                correct_count += 1
                
            if final_level != score_level:
                overrides += 1
                c_before = (score_level == t)
                c_after = (final_level == t)
                if not c_before and c_after:
                    outcome = "fixed"
                    r_fixed += 1
                elif c_before and not c_after:
                    outcome = "harmed"
                    r_harmed += 1
                else:
                    outcome = "neutral"
                    r_neutral += 1
                for rule in constraint["rule_ids"]:
                    rule_counts_all[rule][outcome] += 1
                    rule_counts_all[rule]["total"] += 1
                    
        acc = correct_count / total
        pure_acc = pure_correct_count / total
        accs.append(acc)
        pure_accs.append(pure_acc)
        net = r_fixed - r_harmed
        win_rate = r_fixed / (r_fixed + r_harmed) if (r_fixed + r_harmed) else 0
        print(f"Run {r}: Accuracy = {acc:.2%} (Pure: {pure_acc:.2%}, Net Gain: {correct_count - pure_correct_count:+d}), Overrides = {overrides}, Fixed = {r_fixed}, Harmed = {r_harmed}, WinRate = {win_rate:.1%}")
        
    avg_acc = sum(accs) / len(accs)
    avg_pure = sum(pure_accs) / len(pure_accs)
    print(f"--> Average Accuracy: {avg_acc:.2%} (Pure Avg: {avg_pure:.2%}, Avg Gain: {avg_acc - avg_pure:+.2%})")
    
    print("\nRule Performance Summary across 3 Runs:")
    for rule, st in sorted(rule_counts_all.items(), key=lambda x: x[1]["total"], reverse=True):
        denom = st["fixed"] + st["harmed"]
        win_rate = f"{st['fixed']/denom:.1%}" if denom else "N/A"
        print(f"  {rule:<40} | Fixed: {st['fixed']:<3} | Harmed: {st['harmed']:<3} | Net: {st['fixed']-st['harmed']:<+3} | Total: {st['total']:<3} | WinRate: {win_rate}")


if __name__ == "__main__":
    # 1. Baseline V20 Configuration
    replay_evaluation("V20 Baseline", {
        "enable_multiple_tasks_floor_2": True,
        "standard_chain_version": "v20",
        "enable_parallel_basic_bundle_ceiling_2": False,
        "enable_compressed_high_burden_floor_4": False,
        "regular_comprehensive_version": "v20",
    })

    # 2. V21 Clean Core: Remove broad_task_burden + multiple_tasks_floor_2 + Add parallel_basic_bundle_ceiling_2 + Tighten standard_chain + Tighten regular_comprehensive
    replay_evaluation("V21 Candidate 1 (Clean Core + Parallel Basic Bundle)", {
        "enable_multiple_tasks_floor_2": False,
        "standard_chain_version": "tightened",
        "enable_parallel_basic_bundle_ceiling_2": True,
        "enable_compressed_high_burden_floor_4": False,
        "regular_comprehensive_version": "tightened",
    })

    # 3. V21 Candidate 2: Clean Core + Parallel Basic Bundle + Compressed High Burden Floor 4 (Min Axes = 2)
    replay_evaluation("V21 Candidate 2 (Candidate 1 + Compressed High Burden Floor 4, Min Axes=2)", {
        "enable_multiple_tasks_floor_2": False,
        "standard_chain_version": "tightened",
        "enable_parallel_basic_bundle_ceiling_2": True,
        "enable_compressed_high_burden_floor_4": True,
        "compressed_min_axes": 2,
        "regular_comprehensive_version": "tightened",
    })

    # 4. V21 Candidate 3: Clean Core + Parallel Basic Bundle + Compressed High Burden Floor 4 (Min Axes = 3)
    replay_evaluation("V21 Candidate 3 (Candidate 1 + Compressed High Burden Floor 4, Min Axes=3)", {
        "enable_multiple_tasks_floor_2": False,
        "standard_chain_version": "tightened",
        "enable_parallel_basic_bundle_ceiling_2": True,
        "enable_compressed_high_burden_floor_4": True,
        "compressed_min_axes": 3,
        "regular_comprehensive_version": "tightened",
    })
