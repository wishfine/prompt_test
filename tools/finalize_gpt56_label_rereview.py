#!/usr/bin/env python3
"""Finalize the conservative 1066-question GPT-5.6 label re-review.

The original adjudication CSV is preserved. This script creates a revised label
set with one primary label plus an auditable acceptable-level range.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKETS = ROOT / "tmp/gpt56_label_rereview_packets.jsonl"
OUT_CSV = ROOT / "data/labeled/physics_adjudicated_labels_gpt56_rereview_1066.csv"
OUT_JSON = ROOT / "tmp/physics_adjudicated_labels_gpt56_rereview_1066.json"
OUT_SUMMARY = ROOT / "tmp/physics_adjudicated_labels_gpt56_rereview_summary.json"
QUESTIONS = ROOT / "data/labeled/physics_difficulty_tiku_data_0714_1000.jsonl"

LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}

# The previous 424-question audit marked 84 GPT labels for revision. To avoid
# fitting the gold labels to model output, only the following 20 cases change the
# primary label after a second semantic pass. Other adjacent disagreements are
# retained as acceptable-level ranges while the original primary label remains.
CLEAR_PRIMARY_REVISIONS = {
    "2538082621992771584": "基础题",  # refraction mapping, not a 3-4 decision chain
    "2779382891008229376": "中等题",  # dry/wet road optical path comparison
    "2892517851348377600": "中等题",  # pressure difference plus weighing relation
    "3659088608210468864": "基础题",  # two linked energy-conversion judgments
    "3678659552448622592": "中等题",  # long project wrapper, but direct standard chain
    "2355737047499063296": "中等题",  # routine melting-curve interpretation
    "2359091761527738368": "中等题",  # direct power/efficiency/fuel chain
    "2747544369375850496": "中等题",  # standard boiling experiment
    "2842562290281402368": "中等题",  # standard average-speed experiment
    "2857543663990099968": "中等题",  # standard average-speed experiment
    "2861647610033246208": "中等题",  # standard pinhole-camera experiment
    "3673937014702317568": "中等题",  # direct pressure/buoyancy/equilibrium calculations
    "2426820619775299584": "拔高题",  # inverse pressure relation and parameter screening
    "2961803657007108096": "拔高题",  # nonlinear graph plus two circuit states
    "3650594325130444800": "拔高题",  # circuit design and two-scheme resistance derivation
    "3678675760415592448": "拔高题",  # hidden displacement inferred across buoyancy states
    "3028418693838782464": "拔高题",  # direct parallel-heater extrema, no final network
    "3044393486444158976": "拔高题",  # standard multi-state circuit and one range
    "3385313576457912320": "拔高题",  # duplicate standard multi-state circuit and range
    "3021025329385111552": "基础题",  # complete circuit connection is not pure retrieval
}


def ordered_levels(levels: set[str]) -> str:
    return "|".join(sorted(levels, key=LEVEL_INDEX.get))


def load_packets() -> list[dict]:
    return [json.loads(line) for line in PACKETS.read_text(encoding="utf-8").splitlines()]


def model_prediction(path: Path) -> dict[str, str]:
    output: dict[str, str] = {}
    if not path.exists():
        return output
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        qid = str(item.get("question_id") or item.get("parent_id") or "")
        rating = item.get("difficulty_rating") or {}
        level = rating.get("difficulty_level") or item.get("difficulty_level")
        if qid and level in LEVEL_INDEX:
            output[qid] = level
    return output


def evaluate(labels: dict[str, str], predictions: dict[str, str]) -> dict:
    ids = sorted(set(labels) & set(predictions))
    exact = sum(labels[qid] == predictions[qid] for qid in ids)
    distances = [abs(LEVEL_INDEX[labels[qid]] - LEVEL_INDEX[predictions[qid]]) for qid in ids]
    return {
        "evaluated": len(ids),
        "exact": exact,
        "exact_rate": round(exact / len(ids), 4) if ids else None,
        "within_one_rate": round(sum(d <= 1 for d in distances) / len(ids), 4) if ids else None,
        "mae": round(sum(distances) / len(ids), 4) if ids else None,
        "severe": sum(d >= 2 for d in distances),
    }


def main() -> None:
    packets = load_packets()
    questions = {
        str(item["question_id"]): item
        for item in (
            json.loads(line) for line in QUESTIONS.read_text(encoding="utf-8").splitlines()
        )
    }
    rows: list[dict] = []
    for packet in packets:
        qid = packet["question_id"]
        question = questions[qid]
        original = packet["gpt_label"]
        revised = original
        acceptable = {original}
        status = "保留原GPT裁定"
        source = "1066题全量一致性复核"
        rationale = packet["gpt_reason"]
        confidence = packet["gpt_confidence"]

        if packet["existing_review_conclusion"]:
            source = "三次运行错题并集逐题复核"
            rationale = packet["existing_manual_reason"] or rationale

        existing_levels = {
            level for level in str(packet["existing_acceptable_levels"] or "").split("|") if level in LEVEL_INDEX
        }
        if existing_levels:
            acceptable |= existing_levels

        if packet["priority"] == "P0_明确改标复核":
            proposed = packet["existing_manual_label"]
            if qid not in CLEAR_PRIMARY_REVISIONS:
                acceptable |= {original, proposed}
                status = "保留原主标签；相邻档有争议"
                confidence = "中"
                rationale = (
                    "第二轮复核认为现有证据只能证明该题处于相邻档边界，不能排除原GPT"
                    "主标签。为避免使用模型预测反向拟合真值，保留原主标签，并将另一档"
                    "记录为可接受等级。"
                )
            else:
                revised = CLEAR_PRIMARY_REVISIONS[qid]
                acceptable = {original, revised}
                status = "修订GPT主标签"
                confidence = "高"
                if qid == "2538082621992771584":
                    rationale = (
                        "需要将光的折射规律映射到水中刻度的视觉变化，超过送分题的直接"
                        "检索，但最长链仍只有1—2个常规决策，不足以达到中等题。"
                    )
        elif packet["priority"] == "P1_边界复核":
            status = "保留主标签；相邻双档均可"
            confidence = "中"
        elif packet["existing_review_conclusion"] == "GPT标签需修订":
            # Defensive branch for legacy extracts where priority metadata is absent.
            revised = packet["existing_manual_label"] or original
            acceptable |= {original, revised}
            status = "修订GPT主标签"
            confidence = "中"
        elif packet["existing_review_conclusion"]:
            status = "逐题复核后保留GPT裁定"

        if not packet["existing_review_conclusion"]:
            source = "未进入三次错题并集；逐题结构一致性复核"
            rationale = (
                f"题干、解析与结构证据支持{revised}："
                f"step={packet['structure']['step']}，题型={packet['structure']['type']}，"
                f"状态={packet['structure']['state']}，约束={packet['structure']['constraint']}，"
                f"推理={packet['structure']['reasoning']}。教师、盲评和多次模型证据中"
                "未形成足以推翻原裁定的同向结构证据。"
            )

        rows.append(
            {
                "题目ID": qid,
                "原GPT裁定档": original,
                "修订后主标签": revised,
                "可接受等级": ordered_levels(acceptable),
                "复核状态": status,
                "修订后置信度": confidence,
                "教师标准档": packet["teacher_label"],
                "初次盲评档": packet["blind_label"],
                "复核依据": rationale,
                "复核证据来源": source,
                "题干": packet["stem"],
                "选项": packet["options"],
                "官方解析": packet["official_analysis"],
                "题目图片URL": str(question.get("stem_pic_url") or ""),
                "解析图片URL": str(question.get("analysis_pic_url") or ""),
                "结构_步骤": packet["structure"]["step"],
                "结构_题型": packet["structure"]["type"],
                "结构_状态": packet["structure"]["state"],
                "结构_约束": packet["structure"]["constraint"],
                "结构_实验": packet["structure"]["experiment"],
                "结构_图表": packet["structure"]["graph"],
                "结构_推理": packet["structure"]["reasoning"],
            }
        )

    fieldnames = list(rows[0])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    original_labels = {row["题目ID"]: row["原GPT裁定档"] for row in rows}
    revised_labels = {row["题目ID"]: row["修订后主标签"] for row in rows}
    summary = {
        "questions": len(rows),
        "changed_primary_labels": sum(row["原GPT裁定档"] != row["修订后主标签"] for row in rows),
        "boundary_or_multi_acceptable": sum("|" in row["可接受等级"] for row in rows),
        "original_distribution": dict(Counter(original_labels.values())),
        "revised_distribution": dict(Counter(revised_labels.values())),
        "change_transitions": {
            f"{old}→{new}": count
            for (old, new), count in Counter(
                (row["原GPT裁定档"], row["修订后主标签"])
                for row in rows
                if row["原GPT裁定档"] != row["修订后主标签"]
            ).items()
        },
        "model_evaluations": {},
    }
    for name in [
        "lite_physics_gpt56_accuracyfix_1066_run1.jsonl",
        "lite_physics_gpt56_accuracyfix_1066_run2.jsonl",
        "lite_physics_gpt56_accuracyfix_1066_run3.jsonl",
        "lite_physics_final_candidate_1066_run1.jsonl",
        "lite_physics_final_candidate_1066_run2.jsonl",
        "lite_physics_final_candidate_1066_run3.jsonl",
    ]:
        path = ROOT / "outputs/model_runs" / name
        predictions = model_prediction(path)
        summary["model_evaluations"][name] = {
            "against_original": evaluate(original_labels, predictions),
            "against_revised": evaluate(revised_labels, predictions),
        }
    OUT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
