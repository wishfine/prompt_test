#!/usr/bin/env python3
"""Build a per-question audit for predictions outside every accepted label."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/model_runs/lite_physics_erroraudit_1066_run1.jsonl"
REREVIEW = ROOT / "data/labeled/physics_adjudicated_labels_gpt56_rereview_1066.csv"
ORIGINAL = ROOT / "data/labeled/physics_adjudicated_labels_gpt56_1066.csv"
TEACHER = ROOT / "data/labeled/physics_teacher_labels_0714.csv"
REPORT = ROOT / "output/doc/物理难度_erroraudit_198道真实错题逐题审计.md"

LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_NUM = {level: index for index, level in enumerate(LEVELS, 1)}
TEACHER_MAP = {
    "容易": "送分题",
    "较易": "基础题",
    "中等": "中等题",
    "较难": "拔高题",
    "困难": "压轴题",
}


def read_csv(path: Path, id_field: str) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            str(row.get(id_field, "")).strip(): row
            for row in csv.DictReader(handle)
            if str(row.get(id_field, "")).strip()
        }


def accepted_levels(row: dict[str, str]) -> set[str]:
    accepted = set(
        re.findall("送分题|基础题|中等题|拔高题|压轴题", row.get("可接受等级", ""))
    )
    accepted.add(row["修订后主标签"])
    return accepted


def compact(text: object, limit: int) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    return value if len(value) <= limit else value[: limit - 1] + "…"


def first_url(text: str) -> str:
    match = re.search(r"https?://[^\s,，;；]+", text or "")
    return match.group(0) if match else ""


def diagnose(target: str, prediction: str, row: dict[str, str], features: dict, basis: str) -> tuple[str, str]:
    pair = (target, prediction)
    # The rereview rationale intentionally names several possible high-level
    # structures, so subtype detection must use the actual question/solution
    # and model basis instead of those generic calibration phrases.
    joined = " ".join([row.get("题干", ""), row.get("官方解析", ""), basis])

    if pair == ("中等题", "基础题"):
        if (
            features.get("problem_structure") == "实验探究"
            or features.get("experiment_requirement") != "无"
            or re.search(r"实验|探究|测量", row.get("题干", ""))
        ):
            return (
                "完整实验或信息处理任务被拆成若干1—2步小项，整题任务负担B未被保留。",
                "P1：恢复受约束的整题B通道；完整实验的操作—数据—结论即使答案独立，也按整体常规分析。",
            )
        if (
            features.get("graph_table_requirement") != "无"
            or features.get("problem_structure") in {"电路综合", "图像表格分析"}
            or re.search(r"图像|图线|曲线|电路|光路|不同状态|运动过程", row.get("题干", ""))
        ):
            return (
                "多个判断共享题目特有图像、状态或装置，但模型只按最高难单项计步，忽略共享分析负担。",
                "P1：共享题目特有信息且至少两个判断需推导时进入中等比较，不恢复无条件的选项计数。",
            )
        return (
            "教师/GPT口径按整题多角度辨析定为中等，模型按单项显性应用压为基础。",
            "P1：增加同一概念条件集或同一现象多角度辨析的窄通道，并要求列出共享条件证据。",
        )
    if pair == ("基础题", "中等题"):
        return (
            "模型把普通教材概念、简单因果或标准定义包装成严格逻辑辨析，虽然特征仍是1—2步、单状态、无约束。",
            "P2：判中等时必须写出实际充分/必要条件或具体反例；只能复述教材结论则回到基础。",
        )
    if pair == ("拔高题", "中等题"):
        if re.search(r"误差|评价|猜想|方案|改进", row.get("题干", "") + " " + row.get("官方解析", "")):
            subtype = "误差方向、方案评价或异常解释被当成普通实验结论"
        elif re.search(r"空心|组成|密度之比|体积比|质量比|比例", joined):
            subtype = "比例、组成或隐含物理量反推被压成常规代入"
        elif re.search(r"几何|光路|成像|像距|力臂|路径|角度", joined):
            subtype = "动态几何、光路或隐含力臂关系未被识别为决定性转换"
        elif re.search(r"图像|图线|函数|曲线", joined):
            subtype = "图线身份、参数关系或状态信息的反推被当成直接读图"
        elif re.search(r"状态|熔化|浸没|漂浮|开关|滑片", joined):
            subtype = "前后状态转换及其联动关系被压缩成单一常规模型"
        else:
            subtype = "3—5步链中真正决定后续结果的隐藏关系未被识别"
        return (
            subtype + "。",
            "P3：中等结论前强制列出‘题面未直接给出且错误会使后续失效’的关系；命中即进入拔高比较。",
        )
    if pair == ("压轴题", "拔高题"):
        return (
            "模型只计算最高问表面步骤，未把前置参数、多个状态/图像和边界筛选累计为完整共享求解网络。",
            "P4：压轴审计必须显式列出全链节点；答案独立但共享复杂参数网络时仍按模型依赖累计。",
        )
    if pair == ("压轴题", "中等题"):
        return (
            "复杂多问被错误拆成独立常规小问，模型依赖和多约束网络整体丢失。",
            "P4：对三问以上且共享装置/参数/图像的题强制先重建全链，再允许判中等或拔高。",
        )
    if pair == ("中等题", "拔高题"):
        return (
            "常规双状态、直接图像读取或普通实验分析被误写成决定性反推。",
            "P3b/PP2：明确‘直接读图+常规公式’不是决定性转换；后处理升拔高需出现真实反推证据。",
        )
    if pair == ("拔高题", "压轴题"):
        return (
            "多状态、多约束和6—8步被机械聚合为压轴，但缺少分类、候选边界比较或有效解筛选。",
            "PP3：拔高升压轴除结构计数外，必须存在至少一个强压轴动作。",
        )
    if pair == ("送分题", "基础题"):
        return (
            "唯一熟悉教材模板被解释成一次生活映射或简单因果，送分边界偏严。",
            "P5：不宜宽泛降档；仅对唯一教材模板、无规律选择的直接识别做窄校准。",
        )
    if pair == ("基础题", "送分题"):
        return (
            "简单应用、规范判断或不同回答规则被压成直接检索。",
            "P5/PP1：送分必须确认没有第二次物理决策；收紧single-template后处理。",
        )
    if pair == ("送分题", "中等题"):
        return (
            "低结构概念题被误当成严格概念逻辑辨析，产生两档高估。",
            "P2：低结构特征组合不得仅凭‘概念边界’进入中等。",
        )
    if pair == ("中等题", "送分题"):
        return (
            "完整探究变量匹配被当成唯一教材模板，且后处理继续降档。",
            "PP1：禁用或收紧single-template规则；题干含探究变量/方案选择时禁止降送分。",
        )
    if pair == ("拔高题", "基础题"):
        return (
            "组成、守恒或比例反推链被严重压缩成一次显性代入。",
            "P3：比例题必须区分直接求值与反推未知组成/状态；后者进入决定性转换审计。",
        )
    return ("相邻边界结构识别与复核标签不一致。", "需要逐题复核，不建议增加宽泛关键词规则。")


def main() -> None:
    rereview = read_csv(REREVIEW, "题目ID")
    original = read_csv(ORIGINAL, "题目ID")
    teacher = read_csv(TEACHER, "ID")
    outputs: dict[str, dict] = {}
    with OUTPUT.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            outputs[str(item.get("question_id") or item.get("ID") or "")] = item

    errors = []
    for question_id, label_row in rereview.items():
        item = outputs[question_id]
        result = item.get("difficulty_rating") or {}
        raw = item.get("difficulty_rating_raw") or result
        prediction = result.get("difficulty_level")
        if prediction in accepted_levels(label_row):
            continue
        errors.append((question_id, label_row, prediction, raw.get("difficulty_level"), item))

    transition_counts = Counter((row[1]["修订后主标签"], row[2]) for row in errors)
    diagnosis_counts: Counter[str] = Counter()
    recommendation_counts: Counter[str] = Counter()
    audited = []
    for question_id, row, prediction, raw_prediction, item in errors:
        result = item.get("difficulty_rating") or {}
        features = result.get("features") or {}
        basis = str((result.get("reasoning") or {}).get("core_basis", ""))
        diagnosis, recommendation = diagnose(
            row["修订后主标签"], prediction, row, features, basis
        )
        diagnosis_counts[diagnosis] += 1
        recommendation_counts[recommendation.split("：", 1)[0]] += 1
        audited.append(
            (question_id, row, prediction, raw_prediction, item, features, basis, diagnosis, recommendation)
        )

    lines = [
        "# 物理难度 erroraudit：198 道非边界错题逐题审计",
        "",
        f"- 模型输出：`{OUTPUT.relative_to(ROOT)}`",
        f"- 主标签：`{REREVIEW.relative_to(ROOT)}`",
        f"- 审计题数：**{len(audited)}**（最终预测既不等于主标签，也不在 `可接受等级` 中）",
        "- 严格准确率：**72.61%**；计入可接受相邻档后的准确率：**81.43%**。",
        "",
        "## 三套标签结果",
        "",
        "| 标签口径 | 严格准确率 | 一档内 | MAE | 严重偏差 |",
        "|---|---:|---:|---:|---:|",
        "| 教师原标签 | 58.72% | 98.03% | 0.4325 | 21 |",
        "| 原始 GPT-5.6 裁定 | 71.11% | 99.34% | 0.2955 | 7 |",
        "| GPT-5.6 二次逐题复核主标签 | **72.61%** | **99.53%** | **0.2786** | **5** |",
        "",
        "## 总体归因",
        "",
        "- 189 道：Raw 已错，后处理未改变，属于 Prompt/模型任务结构识别问题。",
        "- 6 道：Raw 错，后处理向正确方向调整一档，但仍未进入可接受范围。",
        "- 3 道：Raw 本来严格正确，被后处理改错。",
        "- 182 道为高置信复核标签；165 道同时与教师标签、原始 GPT 裁定一致，不能主要归因于标签噪声。",
        "",
        "## 错误转移分布",
        "",
        "| 主标签 → 模型 | 数量 |",
        "|---|---:|",
    ]
    for (target, prediction), count in transition_counts.most_common():
        lines.append(f"| {target} → {prediction} | {count} |")

    lines += [
        "",
        "## 推荐修改优先级",
        "",
        "1. **P1：中等题整题任务负担窄通道。** 当前 Prompt 对独立小问压缩过强。仅当多个任务共享题目特有装置、图像、状态或实验流程时，允许整题 B 支持中等；不恢复无条件的‘选项多即中等’。",
        "2. **P3：拔高题决定性关系强制审计。** 58 道拔高→中等全部被模型写成3—5步常规分析。判中等前应强制输出题面未直接给出的关键关系；空心/组成反推、动态几何、误差方向、图线身份和状态联动属于重点。",
        "3. **P4：压轴全链节点清单。** 对答案依赖或模型依赖题，必须列出前置参数、状态方程、图像取值、每项约束和候选边界，避免只数最高问表面动作。",
        "4. **P2：基础→中等逻辑证据门槛。** 只有实际写出条件集合或反例才算严格逻辑辨析，不能凭‘概念边界’四个字升中等。",
        "5. **PP1：收紧 `gpt56_basic_to_easy_single_template_guard`。** 本轮唯一触发即恶化；探究变量、方案选择或实验问题不得降送分。",
        "6. **PP2/PP3：保留高档规则总体框架，但增加反证条件。** 直接读图不能触发中等→拔高；拔高→压轴必须出现分类、候选边界比较、有效解筛选或开放验证之一。",
        "",
        "## 逐题审计",
        "",
    ]

    for index, (question_id, row, prediction, raw_prediction, item, features, basis, diagnosis, recommendation) in enumerate(audited, 1):
        teacher_level = TEACHER_MAP.get(teacher.get(question_id, {}).get("难度", ""), "缺失")
        original_level = original.get(question_id, {}).get("最终裁定档", "缺失")
        actions = item.get("postprocess_actions") or []
        if raw_prediction == row["修订后主标签"] and prediction != raw_prediction:
            stage = "后处理把 Raw 正确结果改错"
        elif actions:
            stage = "Raw 错误，后处理调整后仍错"
        else:
            stage = "Prompt/模型 Raw 错误，后处理未触发"
        image_url = first_url(row.get("题目图片URL", ""))
        analysis_url = first_url(row.get("解析图片URL", ""))
        lines += [
            f"### {index}. `{question_id}`",
            "",
            f"- 标签：复核主标签 **{row['修订后主标签']}**；可接受 `{row.get('可接受等级') or '仅主标签'}`；原始GPT `{original_level}`；教师 `{teacher_level}`；置信度 `{row.get('修订后置信度')}`。",
            f"- 预测：Raw `{raw_prediction}` → 最终 `{prediction}`；阶段：**{stage}**。",
            f"- 结构：`{features.get('problem_structure')}` / `{features.get('step_count')}` / `{features.get('state_count')}` / `{features.get('constraint_count')}` / `{features.get('reasoning_chain')}`。",
            f"- 题目摘要：{compact(row.get('题干'), 240)}",
            f"- 模型依据：{compact(basis, 300)}",
            f"- 诊断：{diagnosis}",
            f"- 建议：{recommendation}",
        ]
        if actions:
            lines.append(f"- 后处理动作：`{compact(json.dumps(actions, ensure_ascii=False), 360)}`")
        if image_url:
            lines += [f"- [题目图片]({image_url})", f"  ![题目图片-{question_id}]({image_url})"]
        if analysis_url:
            lines += [f"- [解析图片]({analysis_url})", f"  ![解析图片-{question_id}]({analysis_url})"]
        lines.append("")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {REPORT} ({len(audited)} questions, {REPORT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
