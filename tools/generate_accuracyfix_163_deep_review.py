#!/usr/bin/env python3
"""生成 accuracyfix Run1 的 163 道残余错题深度诊断报告。"""

from __future__ import annotations

import copy
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import physics_difficulty_rating_with_cache as rating  # noqa: E402


LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}
AUDIT_TSV = ROOT / "tmp" / "accuracyfix_163_analysis.tsv"
LABEL_CSV = ROOT / "data" / "labeled" / "physics_adjudicated_labels_gpt56_1066.csv"
ACCURACYFIX_RUN = ROOT / "outputs" / "model_runs" / "lite_physics_gpt56_accuracyfix_1066_run1.jsonl"
FINAL_RUNS = [
    ROOT / "outputs" / "model_runs" / f"lite_physics_final_candidate_1066_run{i}.jsonl"
    for i in (1, 2, 3)
]
OUTPUT = ROOT / "output" / "doc" / "物理难度_163道残余错题逐题深度分析与修正建议.md"


def load_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                rows[str(item["question_id"])] = item
    return rows


def final_level(item: dict[str, Any]) -> str:
    return str(item["difficulty_rating"]["difficulty_level"])


def raw_level(item: dict[str, Any]) -> str:
    return str(item.get("difficulty_level_raw") or item["difficulty_rating_raw"]["difficulty_level"])


def distance(level: str, accepted: set[str]) -> int:
    return min(abs(LEVEL_INDEX[level] - LEVEL_INDEX[value]) for value in accepted)


def short(text: Any, limit: int = 260) -> str:
    value = " ".join(str(text or "").split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def issue_category(target: str, prediction: str, features: dict[str, str]) -> str:
    if target == "送分题" and prediction in {"基础题", "中等题"}:
        return "单一教材模板被过度解释为应用或过程分析"
    if target == "基础题" and prediction == "送分题":
        return "直接检索束扩张过度，漏掉规律应用或多回答规则"
    if target == "中等题" and prediction == "基础题":
        if features.get("experiment_requirement") != "无" or features.get("graph_table_requirement") != "无":
            return "完整实验/图表任务被拆成若干基础动作"
        if features.get("state_count") != "单状态" or features.get("constraint_count") != "无约束":
            return "连续状态或条件关系被压成一次显性应用"
        return "多项非平凡概念辨析被误当成低结构独立检索"
    if target == "拔高题" and prediction == "中等题":
        if features.get("experiment_requirement") in {"控制变量或故障分析", "方案设计或误差评价"}:
            return "实验拓展、异常反推或误差评价链未达到拔高"
        if features.get("graph_table_requirement") in {"多组比较归纳", "图像反推或外推"}:
            return "多图/图像反推和状态转换被按常规读图处理"
        if features.get("state_count") in {"双状态", "多状态", "连续变化或临界状态"}:
            return "多状态高密度链被压缩为3—4步常规分析"
        return "决定性转换或5—6步完整链没有被识别"
    if target == "压轴题" and prediction in {"中等题", "拔高题"}:
        return "复杂状态—参数—约束网络未被识别为全链耦合"
    if target == "基础题" and prediction == "中等题":
        return "知识覆盖、题量或图示背景被误当成连续分析"
    if target == "中等题" and prediction == "拔高题":
        return "常规模型被高阶 feature 或范围词过度升档"
    if target == "拔高题" and prediction == "压轴题":
        return "多状态/多约束存在，但缺少压轴所需的完整网络操作"
    if target == "送分题" and prediction == "中等题":
        return "低结构教材题被严重抬高"
    return "相邻档结构映射错误"


def owner(category: str, action_state: str) -> tuple[str, str]:
    if action_state == "后处理直接改错":
        return "后处理", "收紧或停用对应规则；无需改写题型专属关键词"
    if "全链耦合" in category:
        return "Prompt + 窄后处理", "Prompt 强制展开状态—约束网络；仅对6-8步+多状态+多约束做相邻档兜底"
    if "高密度链" in category or "图像反推" in category or "实验拓展" in category or "决定性转换" in category:
        return "Prompt 为主", "要求按解析列出独立规律、状态、方程与筛选环节；features 稳定时再考虑窄后处理"
    if "概念辨析" in category or "完整实验" in category or "连续状态" in category:
        return "Prompt 为主", "补充非平凡辨析/完整任务边界；禁止仅凭低结构 features 自动降档"
    if "教材模板" in category or "直接检索束" in category:
        return "Prompt 为主", "校准送分/基础的唯一模板与第二次物理决策边界，不宜用宽泛后处理"
    return "Prompt 为主", "修正相邻档结构定义，后处理只处理可审计的 feature 自相矛盾"


def action_effect(item: dict[str, Any], accepted: set[str]) -> str:
    actions = item.get("postprocess_actions") or []
    if not actions:
        return "未触发后处理"
    before = raw_level(item)
    after = final_level(item)
    if before in accepted and after not in accepted:
        return "后处理直接改错"
    if distance(after, accepted) < distance(before, accepted):
        return "后处理改善但仍未命中"
    if distance(after, accepted) > distance(before, accepted):
        return "后处理加重偏差"
    return "后处理未改变与目标的距离"


def load_audit_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with AUDIT_TSV.open(encoding="utf-8") as handle:
        for values in csv.reader(handle, delimiter="\t"):
            rows.append(
                {
                    "id": values[1],
                    "gpt": values[2].split("（", 1)[0],
                    "teacher": values[3],
                    "accepted_text": values[5],
                    "accepted": set(values[5].split(" / ")),
                    "audit_reason": values[17],
                }
            )
    return rows


def metric(rows: list[tuple[str, str]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "exact": sum(reference == prediction for reference, prediction in rows),
        "exact_rate": sum(reference == prediction for reference, prediction in rows) / n,
        "within_one": sum(abs(LEVEL_INDEX[reference] - LEVEL_INDEX[prediction]) <= 1 for reference, prediction in rows) / n,
        "mae": sum(abs(LEVEL_INDEX[reference] - LEVEL_INDEX[prediction]) for reference, prediction in rows) / n,
        "severe": sum(abs(LEVEL_INDEX[reference] - LEVEL_INDEX[prediction]) >= 2 for reference, prediction in rows),
        "distribution": Counter(prediction for _, prediction in rows),
    }


def main() -> None:
    audit = load_audit_rows()
    labels: dict[str, dict[str, str]] = {}
    with LABEL_CSV.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            labels[str(row["题目ID"])] = row

    accuracyfix = load_jsonl(ACCURACYFIX_RUN)
    final_runs = [load_jsonl(path) for path in FINAL_RUNS]

    # 用当前代码重放 final-candidate Run1 的原始结果，只测后处理净贡献。
    rating.RATING_PROFILE = "gpt56_hybrid"
    old_pairs: list[tuple[str, str]] = []
    replay_pairs: list[tuple[str, str]] = []
    replay_levels: dict[str, str] = {}
    for question_id, item in final_runs[0].items():
        reference = labels[question_id]["最终裁定档"]
        old_pairs.append((reference, final_level(item)))
        replayed = rating.postprocess_physics_difficulty(
            copy.deepcopy(item["difficulty_rating_raw"]), rating.sanitize_question_data(item)
        )
        replay_level = str(replayed["difficulty_level"])
        replay_levels[question_id] = replay_level
        replay_pairs.append((reference, replay_level))
    old_metric = metric(old_pairs)
    replay_metric = metric(replay_pairs)

    transition_counts: Counter[tuple[str, str]] = Counter()
    category_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    current_status_counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []

    for audit_row in audit:
        question_id = audit_row["id"]
        item = accuracyfix[question_id]
        features = item["difficulty_rating"]["features"]
        original_prediction = final_level(item)
        target = audit_row["gpt"]
        transition_counts[(audit_row["accepted_text"], original_prediction)] += 1
        action_state = action_effect(item, audit_row["accepted"])
        action_counts[action_state] += 1
        category = issue_category(target, original_prediction, features)
        category_counts[category] += 1
        current_predictions = [final_level(run[question_id]) for run in final_runs]
        accepted_flags = [value in audit_row["accepted"] for value in current_predictions]
        if all(accepted_flags):
            current_status = "三次均已修正"
        elif any(accepted_flags):
            current_status = "随机波动：至少一次修正"
        else:
            current_status = "三次均未修正"
        current_status_counts[current_status] += 1
        fix_owner, recommendation = owner(category, action_state)
        detail = {
            **audit_row,
            "item": item,
            "features": features,
            "original_prediction": original_prediction,
            "original_raw": raw_level(item),
            "action_state": action_state,
            "category": category,
            "owner": fix_owner,
            "recommendation": recommendation,
            "current_predictions": current_predictions,
            "current_status": current_status,
            "replay_level": replay_levels[question_id],
            "gpt_reason": labels[question_id]["具体裁定原因"],
        }
        details.append(detail)

    lines: list[str] = []
    lines.extend(
        [
            "# 物理难度 163 道残余错题逐题深度分析与修正建议",
            "",
            "> 口径：163 道来自 `accuracyfix Run1` 经人工边界审计后仍未进入可接受等级区间的题目。主参考为 GPT-5.6 裁定；教师标签仅作辅助。当前状态同时对照 `final-candidate` 三次运行。",
            "",
            "## 结论先行",
            "",
            "这 163 道的主因不是后处理，而是首轮模型把整题任务压缩成最高难局部的常规步骤。原 accuracyfix Run1 中有 **145/163** 未触发任何后处理；换到 final-candidate 后，仍有 **92/163** 三次都未进入人工可接受区间，且其中 90 道三次稳定输出同一错误档，属于系统性边界偏差。",
            "",
            "最主要的三个残余方向是：拔高被压成中等、中等被压成基础、压轴被压成拔高。可通过 Prompt 修正任务重建方式；只有 feature 与等级明显自相矛盾的少数情况适合后处理。",
            "",
            "## 163 道总体分解",
            "",
            "| 人工可接受档 -> accuracyfix 最终档 | 题数 |",
            "|---|---:|",
        ]
    )
    for (target, prediction), count in transition_counts.most_common():
        lines.append(f"| {target} -> {prediction} | {count} |")

    lines.extend(["", "### 到当前 final-candidate 的修正情况", "", "| 状态 | 题数 |", "|---|---:|"])
    for name in ["三次均已修正", "随机波动：至少一次修正", "三次均未修正"]:
        lines.append(f"| {name} | {current_status_counts[name]} |")

    lines.extend(["", "### 根因分类", "", "| 根因 | 题数 | 主要修复位置 |", "|---|---:|---|"])
    for category, count in category_counts.most_common():
        sample = next(detail for detail in details if detail["category"] == category)
        lines.append(f"| {category} | {count} | {sample['owner']} |")

    lines.extend(
        [
            "",
            "## Prompt 与后处理分别出了什么问题",
            "",
            "### Prompt 的主要问题",
            "",
            "1. **计步压缩仍偏强。** 模型正确写出多个公式、状态和图像环节，却把它们概括成“均为常规方法”，最终停在中等或拔高。算术中间量不计步是对的，但由一个物理量转入另一条独立规律、另一状态方程或约束筛选，必须算新的物理决策。",
            "2. **独立选项规则过度吸附基础档。** 独立选项不能累加成 `step_count`，但至少三个选项分别需要规律选择/过程还原，且最难项有两层条件判断时，整题辨析负担可以达到中等；它与“多个教材名词直接检索”不是同一结构。",
            "3. **标准实验被统一压低。** 仅连接、读数、控制变量和常规结论是中等；如果还包含异常/故障反推、参数试算、误差方向、表达式推导、等效替代或方案评价，并形成5—6步链，应进入拔高。",
            "4. **压轴过度依赖显式关键词。** 当真实结构已是6—8步、多状态、多约束，而且状态和约束共同参与同一求解网络时，不应因为没有出现“分类讨论”字样而停在拔高。",
            "",
            "### 后处理的确定性问题",
            "",
            "- 原先的宽泛 `medium_to_basic_low_structure_guard` 在 final-candidate Run1 全量回放中触发15次，改对5次、改错10次，净损失5题。当前代码已对 GPT-5.6 配置停用该宽规则，只保留更窄的显式独立判断。",
            "- 原先通用拔高升压轴规则会把常规实验误升压轴。当前 GPT-5.6 配置已排除 `problem_structure=实验探究` 的通用联合升级；真正项目式边界验证仍可由其他明确规则处理。",
            "- 新增两条不依赖题目关键词的相邻档一致性安全网：`基础题 + 多层因果推理 -> 中等题`，以及 `拔高题 + 6-8步 + 多状态 + 多约束 -> 压轴题`（实验探究除外）。",
            "",
            "## 当前后处理反事实回放",
            "",
            "下表是在 **final-candidate Run1 已经生成的 raw 输出** 上重放新后处理，不包含本次 Prompt 修改的效果，因此只是后处理净贡献的反事实回放，不是新版本线上成绩。",
            "",
            "| 指标 | 原 final-candidate Run1 | 新后处理重放 | 变化 |",
            "|---|---:|---:|---:|",
            f"| 严格正确数 | {old_metric['exact']} | {replay_metric['exact']} | {replay_metric['exact'] - old_metric['exact']:+d} |",
            f"| 严格准确率 | {old_metric['exact_rate']:.2%} | {replay_metric['exact_rate']:.2%} | {(replay_metric['exact_rate'] - old_metric['exact_rate']):+.2%} |",
            f"| 相差不超过一档 | {old_metric['within_one']:.2%} | {replay_metric['within_one']:.2%} | {(replay_metric['within_one'] - old_metric['within_one']):+.2%} |",
            f"| MAE | {old_metric['mae']:.4f} | {replay_metric['mae']:.4f} | {replay_metric['mae'] - old_metric['mae']:+.4f} |",
            f"| 严重偏差 | {old_metric['severe']} | {replay_metric['severe']} | {replay_metric['severe'] - old_metric['severe']:+d} |",
            "",
            "注意：同一后处理重放到更早 accuracyfix Prompt 的 raw 输出时收益很小，说明规则效果依赖 Prompt 生成的 features。必须重新调用模型跑全量，不能仅凭回放定版。",
            "",
            "## 建议的验收顺序",
            "",
            "1. 用当前正式 Prompt 和修改后的 `gpt56_hybrid` 后处理跑1次1066题，先看严格准确率、严重偏差和五档分布。",
            "2. 若准确率达到或超过 final-candidate 三次均值，再补跑2次检查波动；不做多数票，只用于稳定性诊断。",
            "3. 单独审计新规则的触发、改对和改错；任何规则全量净收益不为正就关闭。",
            "4. 对仍稳定错误的题优先使用第二阶段边界复核，不再继续堆题目关键词。",
            "",
            "---",
            "",
            "# 逐题复核（163 道）",
            "",
        ]
    )

    for index, detail in enumerate(details, 1):
        item = detail["item"]
        features = detail["features"]
        actions = item.get("postprocess_actions") or []
        action_text = "；".join(
            f"{action.get('rule')}：{action.get('from')}→{action.get('to')}" for action in actions
        ) or "无"
        raw_reasoning = item.get("difficulty_rating_raw", {}).get("reasoning", {})
        lines.extend(
            [
                f"## {index}. {detail['id']}｜{detail['accepted_text']} -> {detail['original_prediction']}",
                "",
                f"- **题目摘要**：{short(item.get('stem'), 420)}",
                f"- **GPT-5.6 / 教师原标**：{detail['gpt']} / {detail['teacher']}",
                f"- **人工可接受等级**：{detail['accepted_text']}",
                f"- **accuracyfix raw / 最终**：{detail['original_raw']} / {detail['original_prediction']}",
                f"- **final-candidate 三次**：{' / '.join(detail['current_predictions'])}（{detail['current_status']}）",
                f"- **当前后处理重放结果**：{detail['replay_level']}",
                f"- **原后处理动作**：{action_text}；{detail['action_state']}",
                f"- **关键 features**：`step={features.get('step_count')}`，`structure={features.get('problem_structure')}`，`state={features.get('state_count')}`，`constraint={features.get('constraint_count')}`，`reasoning={features.get('reasoning_chain')}`，`experiment={features.get('experiment_requirement')}`，`graph={features.get('graph_table_requirement')}`",
                f"- **逐题诊断**：{detail['category']}。模型把本题解释为“{short(raw_reasoning.get('core_basis'), 360)}”，这一解释遗漏或压缩了人工可接受档所要求的结构，因此不是单纯措辞差异。",
                f"- **裁定依据**：{short(detail['gpt_reason'], 420)}",
                f"- **修复归属**：{detail['owner']}。{detail['recommendation']}。",
                f"- **题目原图 URL**：[打开原图]({item.get('stem_pic_url', '')})",
                "",
                f"![题目 {detail['id']}]({item.get('stem_pic_url', '')})",
                "",
                f"- **解析图 URL**：[打开解析图]({item.get('analysis_pic_url', '')})",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "## 数据范围与限制",
            "",
            "- 本报告深入分析的是 accuracyfix Run1 的163道人工审计后不可接受结果，不是对1066道重新进行完整盲审。",
            "- final-candidate 三次结果用于区分系统性偏差与随机波动，不用于多数票改标签。",
            "- GPT-5.6 裁定本身仍可能存在边界噪声；本报告沿用前序人工审计给出的可接受等级集合。",
            "- Prompt 修改必须通过重新请求模型验证；后处理回放只能证明在既有 raw/features 上的反事实效果。",
        ]
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUTPUT)
    print(f"details={len(details)}")


if __name__ == "__main__":
    main()
