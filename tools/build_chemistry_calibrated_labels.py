#!/usr/bin/env python3
"""Build the independently reviewed 591-question chemistry label set.

The calibrated labels are deliberately stored separately from the original
teacher labels. Existing model predictions are used only to locate rows that
need closer inspection; they are never treated as ground truth or voted into
the calibrated label.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


LEVELS = ["送分题", "基础题", "中等题", "拔高题", "压轴题"]
LEVEL_INDEX = {level: index for index, level in enumerate(LEVELS)}


def _override(
    expected: str,
    calibrated: str,
    reason: str,
    confidence: str = "高",
) -> dict[str, str]:
    return {
        "expected": expected,
        "calibrated": calibrated,
        "reason": reason,
        "confidence": confidence,
    }


EASY_TO_BASIC_REASON = (
    "题目不是单一教材结论的直接检索：需要完成实质表征转换、"
    "书写规则转换，或在不同化学/实验规则之间切换，因此校准为基础题。"
)
BASIC_TO_EASY_REASON = (
    "最高难任务只需识别一个熟悉教材模板，同类选项不形成新的化学决策，"
    "没有实质转换、连续推理、实验分析或计算，校准为送分题。"
)
BASIC_TO_MEDIUM_REASON = (
    "整题包含多个非重复应用任务，或多问共享同一实验、装置或过程模型；"
    "需要在方程式、现象解释、装置判断、实验操作或计算规则之间切换，"
    "已超过一步基础应用，校准为中等题。"
)
MEDIUM_TO_BASIC_REASON = (
    "最高难任务仍是单一熟悉关系的一次应用或直接读图，多个选项没有形成"
    "连续依赖、共享复杂模型或完整实验推理，校准为基础题。"
)
MEDIUM_TO_HARD_REASON = (
    "题目包含决定性转换或高密度综合链，需要图像反推、竞争解释排除、"
    "方案评价、多步守恒/差量，或多个高阶任务共同作用，校准为拔高题。"
)
HARD_TO_MEDIUM_REASON = (
    "任务虽然有多个知识点或较长情境，但最高难链仍属于显性的常规"
    "2—3层分析；没有决定性隐含转换、复杂方案设计、多约束筛选或"
    "高密度综合链，校准为中等题。"
)
FINAL_TO_HARD_REASON = (
    "题目具有明显难点，但有效任务仍可压缩为有限的常规实验/计算主线，"
    "没有压轴所需的深层模型耦合、多阶段分类筛选或复杂边界验证，"
    "校准为拔高题。"
)
HARD_TO_FINAL_REASON = (
    "陌生工业流程、条件图表、实验操作、组成测定和连续定量反推形成"
    "不可拆分的任务链，后段结论依赖前段模型，达到压轴结构。"
)


# These are conservative, content-based corrections made after reading the
# stem, options and official analysis. Every correction is constrained to the
# adjudication rubric above; no prediction majority is used as an override.
OVERRIDES: dict[str, dict[str, str]] = {}


def _add(ids: set[str], expected: str, calibrated: str, reason: str) -> None:
    for question_id in ids:
        if question_id in OVERRIDES:
            raise ValueError(f"duplicate override: {question_id}")
        OVERRIDES[question_id] = _override(expected, calibrated, reason)


_add(
    {
        "3161742685600026624",
        "3476099216835158016",
        "3555973551201468416",
        "2639524193759182848",
        "3099633546970275840",
        "2606643775862042624",
        "3003746845333970944",
        "2141410963076493312",
    },
    "送分题",
    "基础题",
    EASY_TO_BASIC_REASON,
)

_add(
    {
        "3664463512542121984",
        "3552871706617802752",
        "3215104698926850048",
        "3636137661934694400",
        "3538290159134478336",
        "3662157434687926272",
        "3666693096570724352",
        "3654316756118904832",
        "3673957124007632896",
        "3677446659925356544",
        "3135993510787076096",
        "3621025384589434880",
        "3558738074168307712",
        "3083792523382542336",
        "3081750667475865600",
        "2965159942154932224",
        "2735666736520617984",
        "3556131672054366208",
    },
    "基础题",
    "送分题",
    BASIC_TO_EASY_REASON,
)

_add(
    {
        "2981768897492766720",
        "2770043486882512896",
        "2352881663246381056",
        "2842774740894482432",
        "3288987267384766464",
        "3118781479932170240",
        "3068617509970227200",
        "2853000570682052608",
        "2803794006878756864",
        "2779348700549464064",
        "2771891127206281216",
        "2770305054987542528",
        "2761608826538987520",
        "2756250657513058304",
        "2624079273197330432",
        "2141425977325858816",
    },
    "基础题",
    "中等题",
    BASIC_TO_MEDIUM_REASON,
)

_add(
    {
        "3654329907693068288",
        "3673960847068172288",
        "2752652145996636160",
        "3555776442573836288",
        "3066331252486434816",
        "3454263622748192768",
        "3421237609864204288",
        "3653138302101192704",
    },
    "中等题",
    "基础题",
    MEDIUM_TO_BASIC_REASON,
)

_add(
    {
        "3066423941995540480",
        "3025440272330825728",
        "3035385419707240448",
        "2774617420599955456",
        "2896153392938586112",
        "2770047955599212544",
        "2963159371790688256",
        "2770041819969994752",
        "2935369921261543424",
        "2781079964288024577",
    },
    "中等题",
    "拔高题",
    MEDIUM_TO_HARD_REASON,
)

_add(
    {
        "2942737603506753536",
        "2861738888741818368",
        "2811488622936129536",
        "3445568017076912128",
        "2779928946168139776",
        "3108328717947248640",
        "3446938339898404864",
        "3144513979363385344",
        "3663176643766169600",
        "3670136898046713856",
        "3676258298490802176",
        "3649838915192115200",
        "3489255760196538368",
        "3270077635874062336",
        "3079047761259257856",
        "2765119021919313920",
        "2753001482513895424",
        "2344167086254235648",
        "2747546704446791680",
        "3610800181249085440",
    },
    "拔高题",
    "中等题",
    HARD_TO_MEDIUM_REASON,
)

_add(
    {
        "3625328542991544320",
        "2758484049973329920",
    },
    "压轴题",
    "拔高题",
    FINAL_TO_HARD_REASON,
)

_add(
    {"2864654761832783872"},
    "拔高题",
    "压轴题",
    HARD_TO_FINAL_REASON,
)


def normalize_id(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def compact(text: Any, limit: int = 180) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = normalize_id(row.get("question_id"))
            if not question_id:
                raise ValueError(f"{path}:{line_number}: missing question_id")
            row["question_id"] = question_id
            rows.append(row)
    return rows


def load_labels(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        question_id = normalize_id(row.get("question_id"))
        if not question_id:
            continue
        if question_id in result:
            raise ValueError(f"duplicate teacher label: {question_id}")
        result[question_id] = row
    return result


def load_predictions(paths: list[Path]) -> dict[str, dict[str, str]]:
    predictions: dict[str, dict[str, str]] = {}
    for path in paths:
        run_name = path.stem
        for row in load_jsonl(path):
            question_id = row["question_id"]
            rating = row.get("difficulty_rating") or {}
            level = rating.get("difficulty_level") or row.get("difficulty_level")
            if level in LEVEL_INDEX:
                predictions.setdefault(question_id, {})[run_name] = level
    return predictions


def stable_review_reason(level: str, teacher_reason: str) -> str:
    prefix = {
        "送分题": "最高难任务属于单一熟悉模板的直接识别，没有第二次化学决策。",
        "基础题": "最高难任务需要一次实质应用、转换或基础实验判断，但模型完全显性。",
        "中等题": "最高难任务形成常规多步分析，或整题具有受控的非重复规则广度。",
        "拔高题": "题目存在决定性转换、方案评价、多约束或高密度综合链。",
        "压轴题": "题目包含复杂建模、多阶段证据和定量关系的深层耦合。",
    }[level]
    detail = compact(teacher_reason, 140)
    return prefix + (f" 原标注依据：{detail}" if detail else "")


def acceptable_levels(
    calibrated: str,
    predictions: dict[str, str],
    changed: bool,
) -> list[str]:
    accepted = {calibrated}
    if changed:
        return [calibrated]
    calibrated_index = LEVEL_INDEX[calibrated]
    for prediction in predictions.values():
        if abs(LEVEL_INDEX[prediction] - calibrated_index) == 1:
            accepted.add(prediction)
    return sorted(accepted, key=LEVEL_INDEX.__getitem__)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "question_id",
        "standard_stars",
        "standard_level",
        "standard_level_name",
        "calibrated_stars",
        "calibrated_level",
        "calibrated_level_name",
        "original_teacher_level",
        "label_changed",
        "review_status",
        "review_confidence",
        "acceptable_levels",
        "review_reason",
        "teacher_reason",
        "stem_summary",
        "stem_pic_url",
        "analysis_pic_url",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def accuracy(
    rows: list[dict[str, Any]],
    predictions: dict[str, dict[str, str]],
    label_field: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    run_names = sorted(
        {run_name for values in predictions.values() for run_name in values}
    )
    for run_name in run_names:
        evaluated = 0
        correct = 0
        for row in rows:
            prediction = predictions.get(row["question_id"], {}).get(run_name)
            if prediction not in LEVEL_INDEX:
                continue
            evaluated += 1
            correct += prediction == row[label_field]
        result[run_name] = {
            "evaluated": evaluated,
            "correct": correct,
            "accuracy": round(correct / evaluated, 6) if evaluated else None,
        }
    return result


def write_report(
    path: Path,
    rows: list[dict[str, Any]],
    source_path: Path,
    teacher_path: Path,
    prediction_paths: list[Path],
    predictions: dict[str, dict[str, str]],
) -> None:
    original_distribution = Counter(
        row["original_teacher_level"] for row in rows
    )
    calibrated_distribution = Counter(
        row["calibrated_level_name"] for row in rows
    )
    changed_rows = [row for row in rows if row["label_changed"] == "是"]
    direction = Counter(
        (row["original_teacher_level"], row["calibrated_level_name"])
        for row in changed_rows
    )
    review_status = Counter(row["review_status"] for row in rows)
    original_accuracy = accuracy(rows, predictions, "original_teacher_level")
    calibrated_accuracy = accuracy(rows, predictions, "calibrated_level_name")

    lines = [
        "# 初中化学591题逐题复核与校准标签报告",
        "",
        "## 结论",
        "",
        f"- 完整复核：{len(rows)} 题。",
        f"- 主标签修订：{len(changed_rows)} 题。",
        f"- 保留原标签：{len(rows) - len(changed_rows)} 题。",
        "- 校准标签独立保存，不覆盖老师原标签。",
        "- 模型结果只用于定位分歧题，不参与多数票定标。",
        "",
        "## 标签源文件",
        "",
        f"- 题目与官方解析：`{source_path}`",
        f"- 原老师标签：`{teacher_path}`",
    ]
    lines.extend(f"- 参考运行：`{item}`" for item in prediction_paths)
    lines.extend(
        [
            "",
            "## 校准口径",
            "",
            "1. 送分题：单一熟悉模板直接识别，无第二次化学决策。",
            "2. 基础题：一次实质应用、表征转换、规则切换或基础实验判断。",
            "3. 中等题：常规2—3层分析，或受控的非重复任务广度。",
            "4. 拔高题：决定性转换、方案评价、多约束或高密度综合链。",
            "5. 压轴题：复杂建模、多阶段证据与定量关系深层耦合。",
            "",
            "独立小问不累加为纵向步骤，但真实的规则切换和整题任务负担",
            "会进入相邻档校准。题干长度、项目名称、图片数量和题目位置",
            "均不单独决定难度。",
            "",
            "## 分布",
            "",
            "| 档位 | 原老师标签 | 校准标签 | 变化 |",
            "|---|---:|---:|---:|",
        ]
    )
    for level in LEVELS:
        before = original_distribution[level]
        after = calibrated_distribution[level]
        lines.append(f"| {level} | {before} | {after} | {after-before:+d} |")

    lines.extend(
        [
            "",
            "## 修订方向",
            "",
            "| 原标签 | 校准标签 | 数量 |",
            "|---|---|---:|",
        ]
    )
    for (before, after), count in sorted(
        direction.items(),
        key=lambda item: (
            LEVEL_INDEX[item[0][0]],
            LEVEL_INDEX[item[0][1]],
        ),
    ):
        lines.append(f"| {before} | {after} | {count} |")

    lines.extend(
        [
            "",
            "## 复核状态",
            "",
        ]
    )
    for status, count in review_status.most_common():
        lines.append(f"- {status}：{count}题")

    lines.extend(
        [
            "",
            "## 现有模型结果对不同标签的严格准确率",
            "",
            "这些结果仅用于观察标签修订影响，不用于决定校准标签。",
            "",
            "| 运行 | 覆盖题数 | 对原标签 | 对校准标签 |",
            "|---|---:|---:|---:|",
        ]
    )
    for run_name in sorted(set(original_accuracy) | set(calibrated_accuracy)):
        old = original_accuracy.get(run_name, {})
        new = calibrated_accuracy.get(run_name, {})
        old_acc = old.get("accuracy")
        new_acc = new.get("accuracy")
        lines.append(
            f"| {run_name} | {new.get('evaluated', old.get('evaluated', 0))} | "
            f"{old_acc:.2%} | {new_acc:.2%} |"
        )

    lines.extend(
        [
            "",
            "## 修改题目明细",
            "",
            "| question_id | 原标签 | 校准标签 | 题目摘要 | 复核理由 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in changed_rows:
        stem = str(row["stem_summary"]).replace("|", "｜")
        reason = str(row["review_reason"]).replace("|", "｜")
        lines.append(
            f"| {row['question_id']} | {row['original_teacher_level']} | "
            f"{row['calibrated_level_name']} | {stem} | {reason} |"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build(args: argparse.Namespace) -> None:
    source_path = Path(args.questions).expanduser().resolve()
    teacher_path = Path(args.teacher_labels).expanduser().resolve()
    prediction_paths = [
        Path(item).expanduser().resolve() for item in args.predictions
    ]
    questions = load_jsonl(source_path)
    teacher_labels = load_labels(teacher_path)
    predictions = load_predictions(prediction_paths)

    question_ids = [row["question_id"] for row in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("duplicate question_id in question source")
    if set(question_ids) != set(teacher_labels):
        missing_teacher = sorted(set(question_ids) - set(teacher_labels))
        missing_question = sorted(set(teacher_labels) - set(question_ids))
        raise ValueError(
            "ID mismatch: "
            f"missing_teacher={missing_teacher[:5]}, "
            f"missing_question={missing_question[:5]}"
        )

    unknown_overrides = sorted(set(OVERRIDES) - set(question_ids))
    if unknown_overrides:
        raise ValueError(f"override IDs not found: {unknown_overrides}")

    rows: list[dict[str, Any]] = []
    for question in questions:
        question_id = question["question_id"]
        teacher = teacher_labels[question_id]
        original = teacher["standard_level_name"]
        if original not in LEVEL_INDEX:
            raise ValueError(
                f"{question_id}: invalid teacher level {original!r}"
            )
        override = OVERRIDES.get(question_id)
        if override and override["expected"] != original:
            raise ValueError(
                f"{question_id}: expected {override['expected']}, got {original}"
            )
        calibrated = override["calibrated"] if override else original
        changed = calibrated != original
        run_predictions = predictions.get(question_id, {})
        prediction_levels = list(run_predictions.values())

        if changed:
            status = "明确改标"
            confidence = override["confidence"]
            review_reason = override["reason"]
        elif prediction_levels and all(
            level == calibrated for level in prediction_levels
        ):
            status = "一致保留"
            confidence = "高"
            review_reason = stable_review_reason(
                calibrated,
                teacher.get("reason", ""),
            )
        elif prediction_levels and any(
            abs(LEVEL_INDEX[level] - LEVEL_INDEX[calibrated]) == 1
            for level in prediction_levels
        ):
            status = "边界保留"
            confidence = "中"
            review_reason = (
                stable_review_reason(calibrated, teacher.get("reason", ""))
                + " 现有模型存在相邻档分歧；复核后主标签保留原档，"
                "同时记录相邻档为可接受边界。"
            )
        else:
            status = "复核保留"
            confidence = "中"
            review_reason = stable_review_reason(
                calibrated,
                teacher.get("reason", ""),
            )

        accepted = acceptable_levels(
            calibrated,
            run_predictions,
            changed,
        )
        rows.append(
            {
                "question_id": question_id,
                # Keep the calibrated file directly compatible with the existing
                # chemistry evaluator, which reads the three standard_* fields.
                "standard_stars": "★" * (LEVEL_INDEX[calibrated] + 1),
                "standard_level": str(LEVEL_INDEX[calibrated] + 1),
                "standard_level_name": calibrated,
                "calibrated_stars": "★" * (LEVEL_INDEX[calibrated] + 1),
                "calibrated_level": str(LEVEL_INDEX[calibrated] + 1),
                "calibrated_level_name": calibrated,
                "original_teacher_level": original,
                "label_changed": "是" if changed else "否",
                "review_status": status,
                "review_confidence": confidence,
                "acceptable_levels": " / ".join(accepted),
                "review_reason": review_reason,
                "teacher_reason": teacher.get("reason", ""),
                "stem_summary": compact(question.get("stem"), 220),
                "stem_pic_url": question.get("stem_pic_url", ""),
                "analysis_pic_url": question.get("analysis_pic_url", ""),
                "model_predictions_for_audit": run_predictions,
                "source_fields_reviewed": [
                    "stem",
                    "options",
                    "analysis",
                    "sub_questions",
                ],
                "calibration_version": "chemistry_manual_rereview_v1",
            }
        )

    if len(rows) != 591:
        raise ValueError(f"expected 591 reviewed rows, got {len(rows)}")

    write_csv(Path(args.output_csv).expanduser().resolve(), rows)
    write_jsonl(Path(args.output_jsonl).expanduser().resolve(), rows)
    write_report(
        Path(args.report).expanduser().resolve(),
        rows,
        source_path,
        teacher_path,
        prediction_paths,
        predictions,
    )
    print(
        json.dumps(
            {
                "reviewed": len(rows),
                "changed": sum(row["label_changed"] == "是" for row in rows),
                "original_distribution": Counter(
                    row["original_teacher_level"] for row in rows
                ),
                "calibrated_distribution": Counter(
                    row["calibrated_level_name"] for row in rows
                ),
                "status_distribution": Counter(
                    row["review_status"] for row in rows
                ),
                "output_csv": str(
                    Path(args.output_csv).expanduser().resolve()
                ),
                "output_jsonl": str(
                    Path(args.output_jsonl).expanduser().resolve()
                ),
                "report": str(Path(args.report).expanduser().resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--teacher-labels", required=True)
    parser.add_argument("--predictions", nargs="*", default=[])
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
