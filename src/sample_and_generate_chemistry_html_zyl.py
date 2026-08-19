# -*- coding: utf-8 -*-
"""初中化学冻结版难度验收可视化。

页面工作流与物理验收页保持一致：
- 未标注默认模型判定合理，教师只需标记异常题；
- 可选正确档位、填写理由并导出 JSONL/TXT；
- 题干或解析有多张图时只展示最后一张；
- 先展示与教师难度口径最相关的可观测证据，再展开全部17项特征。

该脚本只负责可视化，不修改 Prompt、模型原始等级或后处理结果。
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

try:
    from sample_and_generate_html import HTML_TEMPLATE as PHYSICS_HTML_TEMPLATE
except ImportError:  # 作为 src.* 模块导入时
    from src.sample_and_generate_html import (
        HTML_TEMPLATE as PHYSICS_HTML_TEMPLATE,
    )

try:
    from chemistry_observable_features_zyl import CURRICULUM_TOPIC_NAMES
except ImportError:
    from src.chemistry_observable_features_zyl import CURRICULUM_TOPIC_NAMES


CHEMISTRY_FROZEN_RELEASE = "d7fc644"

SAMPLE_PLAN = {
    "送分题": 100,
    "基础题": 120,
    "中等题": 120,
    "拔高题": 100,
    "压轴题": 60,
}

LEVEL_MAP = {
    "送分题": 1,
    "基础题": 2,
    "中等题": 3,
    "拔高题": 4,
    "压轴题": 5,
}

LEVEL_NAMES = {
    1: "难度1 — 送分题",
    2: "难度2 — 基础题",
    3: "难度3 — 中等题",
    4: "难度4 — 拔高题",
    5: "难度5 — 压轴题",
}

FULL_FEATURE_FIELDS: Tuple[Tuple[str, str], ...] = (
    ("longest_solution_chain", "最长解题链"),
    ("task_groups", "考查任务"),
    ("rule_families", "题目考查点"),
    ("curriculum_topics", "涉及教材课题"),
    ("parallel_task_relation", "选项/小问关联方式"),
    ("solution_topology", "解题任务结构"),
    ("reaction_structure", "反应之间的关系"),
    ("condition_operations", "审题条件与陷阱"),
    ("representation_operations", "化学信息转换"),
    ("evidence_operations", "证据推理方式"),
    ("experiment_operation", "实验考查要求"),
    ("experiment_task_structure", "实验任务结构"),
    ("visual_task_structure", "图像任务结构"),
    ("graph_table_operation", "图表信息处理"),
    ("error_analysis_operation", "误差分析"),
    ("calculation_operations", "计算方法"),
    ("new_information_operation", "新信息迁移"),
)


def escape(value: Any) -> str:
    return html.escape(str(value or ""))


def split_image_urls(value: Any) -> List[str]:
    """把图片字段归一化为去重且保持原顺序的 URL 列表。"""
    if isinstance(value, (list, tuple)):
        candidates = [str(item).strip() for item in value]
    else:
        candidates = [part.strip() for part in str(value or "").split(",")]
    return list(dict.fromkeys(url for url in candidates if url))


def render_chemistry_image_section(
    value: Any,
    *,
    title: str,
    kind: str,
) -> str:
    """渲染图片区；题干多图按教师要求只取最后一张。"""
    urls = split_image_urls(value)
    if not urls:
        return f"""
                <div class="media-section media-section-empty">
                    <div class="media-empty">【该题无{escape(title)}】</div>
                </div>
"""

    selected = urls[-1]
    count_hint = (
        f"共{len(urls)}张，题干多图时仅展示最后一张"
        if len(urls) > 1
        else "点击图片可放大查看"
    )
    return f"""
                <div class="media-section chemistry-media-section">
                    <div class="media-heading">
                        <span>{escape(title)}</span>
                        <span class="media-hint">{escape(count_hint)}</span>
                    </div>
                    <div class="image-container image-container-primary">
                        <figure class="image-frame primary-image">
                            <img src="{html.escape(selected)}"
                                 alt="{escape(title)}"
                                 data-image-role="{escape(kind)}-primary"
                                 onclick="openImagePreview(this)"
                                 onerror="markImageFailed(this)">
                            <figcaption>{escape(title)}·最后一张</figcaption>
                        </figure>
                    </div>
                </div>
"""


def _display_value(value: Any, *, field: str | None = None) -> str:
    if value is None or value == "" or value == []:
        return "无"
    if field == "curriculum_topics" and isinstance(value, list):
        return "、".join(
            f"{code} {CURRICULUM_TOPIC_NAMES.get(code, '')}".strip()
            for code in value
        )
    if field == "task_groups" and isinstance(value, list):
        return "；".join(
            f"{item.get('task_type', '未知任务')}×{item.get('count', '?')}"
            if isinstance(item, dict)
            else str(item)
            for item in value
        ) or "无"
    if field == "longest_solution_chain" and isinstance(value, list):
        return " → ".join(
            f"{index}.{step}" for index, step in enumerate(value, 1)
        ) or "无"
    if isinstance(value, list):
        return "、".join(str(item) for item in value) or "无"
    if isinstance(value, dict):
        return "；".join(f"{key}：{val}" for key, val in value.items())
    if isinstance(value, bool):
        return "是" if value else "否"
    return str(value)


CURRICULUM_TOPIC_PATTERN = re.compile(r"U(?:10|11|[1-9])-[1-3]")


def _display_curriculum_span(value: Any) -> str:
    """在课程跨度摘要中同时展示课题编码及教材课题名称。"""
    text = _display_value(value)

    def replace_topic(match: re.Match[str]) -> str:
        code = match.group(0)
        name = CURRICULUM_TOPIC_NAMES.get(code)
        return f"{code} {name}" if name else code

    return CURRICULUM_TOPIC_PATTERN.sub(replace_topic, text)


def build_priority_feature_items(
    item: Dict[str, Any],
    rating: Dict[str, Any],
) -> List[Tuple[str, str]]:
    """按教师评级口径的优先级组装首屏证据。"""
    features = rating.get("features") or {}
    metrics = rating.get("observable_metrics") or {}
    text_fields: List[Tuple[str, str]] = [
        (
            "最长解题链",
            _display_value(
                features.get("longest_solution_chain"),
                field="longest_solution_chain",
            ),
        ),
        (
            "考查任务",
            _display_value(features.get("task_groups"), field="task_groups"),
        ),
        ("题目考查点", _display_value(features.get("rule_families"))),
        ("计算方法", _display_value(features.get("calculation_operations"))),
        ("误差分析", _display_value(features.get("error_analysis_operation"))),
        ("知识点跨度", _display_curriculum_span(metrics.get("curriculum_span_summary"))),
        ("审题条件与陷阱", _display_value(features.get("condition_operations"))),
        ("选项/小问关联方式", _display_value(features.get("parallel_task_relation"))),
        ("解题任务结构", _display_value(features.get("solution_topology"))),
        ("实验任务结构", _display_value(features.get("experiment_task_structure"))),
        ("图像任务结构", _display_value(features.get("visual_task_structure"))),
        ("图表信息处理", _display_value(features.get("graph_table_operation"))),
    ]
    numeric_fields: List[Tuple[str, str]] = [
        ("题干字数", str(item.get("question_text_char_count", metrics.get("question_text_char_count", "无")))),
    ]
    return text_fields + numeric_fields


def build_full_feature_items(
    rating: Dict[str, Any],
) -> List[Tuple[str, str]]:
    features = rating.get("features") or {}
    return [
        (label, _display_value(features.get(key), field=key))
        for key, label in FULL_FEATURE_FIELDS
    ]


EXTRA_CSS = """
        .feature-block-title {
            margin: 16px 0 9px; color: #0b6e69; font-size: 15px;
            font-weight: 800; letter-spacing: .3px;
        }
        .priority-details { grid-template-columns: repeat(3, minmax(0, 1fr)); }
        .priority-details .rating-detail-item {
            border-color: #b9e6df; background: linear-gradient(180deg,#fff,#f4fffc);
            min-height: 72px;
        }
        .priority-details .feature-wide {
            grid-column: 1 / -1; min-height: 0;
        }
        .priority-details .task-structure-start { grid-column: 1; }
        .priority-details .compact-tail-start { grid-column: 1; }
        @media (max-width: 760px) {
            .priority-details { grid-template-columns: 1fr; }
            .priority-details .task-structure-start { grid-column: auto; }
            .priority-details .compact-tail-start { grid-column: auto; }
        }
        .priority-details .label { color: #0b766e; font-weight: 700; }
        .all-feature-details {
            margin-top: 15px; border-top: 1px dashed #b7dcd5; padding-top: 12px;
        }
        .all-feature-details summary {
            cursor: pointer; font-weight: 750; color: #315f5b; margin-bottom: 10px;
        }
        .all-feature-details[open] summary { color: #0b766e; }
        .feature-contract-note {
            margin-top: 10px; color: #64748b; font-size: 12px; line-height: 1.65;
        }
"""


HTML_TEMPLATE = (
    PHYSICS_HTML_TEMPLATE
    .replace("初中物理", "初中化学")
    .replace("physics_difficulty", "chemistry_difficulty")
    .replace("原始教师定位", "原始模型等级")
    .replace("#1e3c72", "#075e54")
    .replace("#2a5298", "#0f9d87")
    .replace("rgba(42,82,152", "rgba(15,157,135")
    .replace("</style>", EXTRA_CSS + "\n    </style>")
)


TASK_STRUCTURE_LABELS = {
    "解题任务结构",
    "实验任务结构",
    "图像任务结构",
}


def _render_feature_grid(
    items: Iterable[Tuple[str, str]],
    *,
    css_class: str = "",
) -> str:
    blocks = []
    for label, value in items:
        item_class = "rating-detail-item"
        if css_class == "priority-details" and label in {"最长解题链", "考查任务"}:
            item_class += " feature-wide"
        if css_class == "priority-details" and label in TASK_STRUCTURE_LABELS:
            item_class += " task-structure-card"
            if label == "解题任务结构":
                item_class += " task-structure-start"
        if css_class == "priority-details" and label == "图表信息处理":
            item_class += " compact-tail-start"
        blocks.append(
            f'<div class="{item_class}">'
            f'<div class="label">{escape(label)}</div>'
            f'<div class="value">{escape(value)}</div>'
            "</div>"
        )
    return f'<div class="rating-details {css_class}">' + "".join(blocks) + "</div>"


def generate_html_file(
    samples: Dict[int, List[Dict[str, Any]]],
    output_path: str,
    *,
    review_scope: str | None = None,
    release_label: str = CHEMISTRY_FROZEN_RELEASE,
) -> None:
    active_levels = [
        level for level in sorted(samples) if samples[level]
    ]
    nav_html = "".join(
        f'        <a href="#level-{level}" data-level="{level}">'
        f'{LEVEL_NAMES[level]} ({len(samples[level])})</a>\n'
        for level in active_levels
    )
    cards: List[str] = []
    all_questions: List[Dict[str, Any]] = []

    for level in active_levels:
        items = samples[level]
        cards.append(
            f'<div id="level-{level}" class="level-section">'
            f'<div class="level-header level-{level}">'
            f'<div class="level-title">{LEVEL_NAMES[level]}</div>'
            f'<div class="level-desc">本档验收 {len(items)} 题；'
            '题干多图时仅展示最后一张</div></div>'
        )
        for index, item in enumerate(items, 1):
            rating = item.get("difficulty_rating") or {}
            reasoning = rating.get("reasoning") or {}
            final_level = rating.get("difficulty_level", "")
            level_num = LEVEL_MAP.get(final_level, level)
            raw_level = item.get("difficulty_level_raw", final_level)
            question_id = str(item.get("question_id", "unknown"))
            postprocess_actions = item.get("postprocess_actions") or []
            action_names = [
                str(action.get("rule") or action.get("name") or action)
                if isinstance(action, dict)
                else str(action)
                for action in postprocess_actions
            ]

            all_questions.append(
                {
                    "question_id": question_id,
                    "parent_id": str(item.get("parent_id", question_id)),
                    "level_num": level_num,
                    "difficulty_level": final_level,
                    "raw_difficulty": raw_level,
                    "reasoning": reasoning,
                    "features": rating.get("features") or {},
                }
            )

            cards.append(
                f'<div class="question-card" data-qid="{escape(question_id)}">'
                '<div class="question-header">'
                f'<span class="question-id">#{index} | ID: {escape(question_id)}</span>'
                '<div class="question-tags">'
                f'<span class="tag tag-raw">原始模型: {escape(raw_level)}</span>'
                f'<span class="tag tag-time">后处理: {escape("、".join(action_names) or "无")}</span>'
                f'<span class="tag tag-tokens">{escape(item.get("api_total_tokens", 0))} tokens</span>'
                "</div></div><div class=\"question-body\">"
                f'<div class="difficulty-badge badge-{level_num}">{escape(final_level)}</div>'
            )
            cards.append(
                render_chemistry_image_section(
                    item.get("stem_pic_url"), title="题干图示", kind="stem"
                )
            )
            cards.append(
                render_chemistry_image_section(
                    item.get("analysis_pic_url"), title="解析图示", kind="analysis"
                )
            )

            cards.append(
                '<div class="rating-section">'
                '<div class="rating-title">化学可观测特征 & 判定理由</div>'
                '<div class="feature-block-title">关键可观测证据（按定档价值排序）</div>'
                + _render_feature_grid(
                    build_priority_feature_items(item, rating),
                    css_class="priority-details",
                )
            )
            if reasoning:
                if isinstance(reasoning, dict):
                    cards.append(
                        '<div class="rating-reasoning">'
                        f'<strong>1. 核心判定依据：</strong>{escape(reasoning.get("core_basis"))}<br/>'
                        f'<strong>2. 易错卡点：</strong>{escape(reasoning.get("hard_point"))}<br/>'
                        f'<strong>3. 为什么不低判一档：</strong>{escape(reasoning.get("why_not_lower"))}<br/>'
                        f'<strong>4. 为什么不高判一档：</strong>{escape(reasoning.get("why_not_higher"))}'
                        "</div>"
                    )
                else:
                    cards.append(
                        f'<div class="rating-reasoning">{escape(reasoning)}</div>'
                    )
            cards.append(
                '<details class="all-feature-details">'
                '<summary>展开全部17项特征</summary>'
                + _render_feature_grid(build_full_feature_items(rating))
                + '<div class="feature-contract-note">'
                '数量指标由程序从数组派生，不使用模型自报难度摘要。'
                "</div></details></div>"
            )

            cards.append(
                '<div class="annotation-section">'
                '<div class="annotation-title">'
                '人工评议验收（未标注即视为“模型判定合理”）</div>'
                '<div class="annotation-row"><label>验收意见：</label>'
                f'<button class="annotation-btn btn-wrong" data-qid="{escape(question_id)}" '
                'data-action="wrong" onclick="setAnnotation(this, \'wrong\')">'
                '✗ 模型判定不准</button>'
                f'<button class="annotation-btn btn-unmark" data-qid="{escape(question_id)}" '
                'data-action="unmark" onclick="setAnnotation(this, \'unmark\')">'
                '✓ 恢复默认合理</button></div>'
                '<div class="annotation-row"><label>建议正确档位：</label>'
                f'<select class="corrected-level-select" data-qid="{escape(question_id)}" '
                'onchange="saveCorrectedLevel(this)">'
                '<option value="">仅标错，暂不指定档位</option>'
                + "".join(
                    f'<option value="{name}">{name}</option>'
                    for name in LEVEL_MAP
                )
                + "</select></div>"
                '<div class="annotation-row"><label>'
                '修改意见与错误原因（输入后自动标记为判定不准）：</label></div>'
                f'<textarea class="annotation-textarea" data-qid="{escape(question_id)}" '
                'placeholder="请说明错误原因及推荐档位..." '
                'oninput="saveAnnotationText(this)"></textarea>'
                "</div></div></div>"
            )
        cards.append("</div>")

    content = HTML_TEMPLATE
    replacements = {
        "__NAV_ITEMS_PLACEHOLDER__": nav_html,
        "__QUESTION_CARDS_PLACEHOLDER__": "".join(cards),
        "__LEVEL_NAMES_PLACEHOLDER__": json.dumps(LEVEL_NAMES, ensure_ascii=False),
        "__LEVEL_MAP_PLACEHOLDER__": json.dumps(LEVEL_MAP, ensure_ascii=False),
        "__QUESTIONS_JSON_PLACEHOLDER__": json.dumps(all_questions, ensure_ascii=False),
        "__REVIEW_COUNT__": str(len(all_questions)),
        "__REVIEW_SCOPE__": review_scope or str(len(all_questions)),
        "__RELEASE_LABEL__": release_label,
    }
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    content = content.replace(
        "for (let lvl = 1; lvl <= 5; lvl++) {",
        "for (const lvl of "
        + json.dumps(active_levels, ensure_ascii=False)
        + ") {",
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    print(f"✨ 成功生成化学冻结版验收页: {target.resolve()}")


def build_sample_plan(sample_size: int) -> Dict[str, int]:
    if sample_size <= 0:
        raise ValueError("sample_size 必须大于0")
    total = sum(SAMPLE_PLAN.values())
    exact = {
        level: sample_size * weight / total
        for level, weight in SAMPLE_PLAN.items()
    }
    plan = {level: int(value) for level, value in exact.items()}
    remainder = sample_size - sum(plan.values())
    for level in sorted(
        SAMPLE_PLAN,
        key=lambda name: exact[name] - plan[name],
        reverse=True,
    )[:remainder]:
        plan[level] += 1
    return plan


def parse_level_plan(value: str) -> Dict[str, int]:
    """解析“送分,基础,中等,拔高,压轴”五档数量。"""
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != len(LEVEL_MAP):
        raise argparse.ArgumentTypeError(
            "--level-plan 必须提供5个逗号分隔整数："
            "送分,基础,中等,拔高,压轴"
        )
    try:
        counts = [int(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--level-plan 只能包含整数"
        ) from exc
    if any(count < 0 for count in counts) or not any(counts):
        raise argparse.ArgumentTypeError(
            "--level-plan 数量不能为负数，且总数必须大于0"
        )
    return dict(zip(LEVEL_MAP, counts))


def select_rows_by_level_plan(
    grouped_by_name: Dict[str, List[Dict[str, Any]]],
    plan: Dict[str, int],
    *,
    seed: int,
    allow_cross_level_fill: bool = False,
) -> Dict[str, List[Dict[str, Any]]]:
    """按最终模型档位抽样，可选从相邻档位补齐总数。

    跨档补数时，补入题仍归入其真实模型档位，不伪造原计划
    档位。例如送分题不足时优先多抽基础题。
    """
    rng = random.Random(seed)
    if allow_cross_level_fill:
        target_total = sum(plan.values())
        available_total = sum(
            len(grouped_by_name.get(level_name, []))
            for level_name in LEVEL_MAP
        )
        if available_total < target_total:
            raise ValueError(
                f"全部档位合计仅{available_total}题，"
                f"无法补齐计划总数{target_total}题"
            )

        selected = {level_name: [] for level_name in LEVEL_MAP}
        remaining: Dict[str, List[Dict[str, Any]]] = {}
        deficits: Dict[str, int] = {}
        for level_name in LEVEL_MAP:
            pool = list(grouped_by_name.get(level_name, []))
            rng.shuffle(pool)
            target_count = plan.get(level_name, 0)
            take_count = min(len(pool), target_count)
            selected[level_name] = pool[:take_count]
            remaining[level_name] = pool[take_count:]
            deficits[level_name] = target_count - take_count

        level_order = list(LEVEL_MAP)
        for shortage_level in level_order:
            shortage = deficits[shortage_level]
            if shortage <= 0:
                continue
            shortage_index = level_order.index(shortage_level)
            donor_levels = sorted(
                level_order,
                key=lambda name: (
                    abs(level_order.index(name) - shortage_index),
                    level_order.index(name),
                ),
            )
            for donor_level in donor_levels:
                if shortage <= 0:
                    break
                donor_pool = remaining[donor_level]
                take_count = min(shortage, len(donor_pool))
                if take_count <= 0:
                    continue
                selected[donor_level].extend(donor_pool[:take_count])
                remaining[donor_level] = donor_pool[take_count:]
                shortage -= take_count
            if shortage:
                raise ValueError(
                    f"{shortage_level}缺口仍有{shortage}题无法补齐"
                )
        return selected

    selected: Dict[str, List[Dict[str, Any]]] = {}
    for level_name, target_count in plan.items():
        pool = list(grouped_by_name.get(level_name, []))
        if len(pool) < target_count:
            raise ValueError(
                f"{level_name}仅{len(pool)}题，无法按计划抽取"
                f"{target_count}题；已禁止用其他档位补数"
            )
        selected[level_name] = rng.sample(pool, target_count)
    return selected


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} JSON无效: {exc}") from exc
    return rows


def _align_optional_image_source(
    rows: List[Dict[str, Any]],
    source_path: Path | None,
) -> None:
    if source_path is None:
        return
    image_index = {
        str(row.get("question_id")): row
        for row in _load_jsonl(source_path)
        if row.get("question_id") is not None
    }
    for row in rows:
        source = image_index.get(str(row.get("question_id")))
        if not source:
            continue
        for field in ("stem_pic_url", "analysis_pic_url"):
            if split_image_urls(source.get(field)):
                row[field] = source[field]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="初中化学冻结版难度评级 HTML 验收工具"
    )
    parser.add_argument("-i", "--input", required=True, help="冻结版评级 JSONL")
    parser.add_argument("-oh", "--output-html", required=True, help="输出 HTML")
    parser.add_argument("-oj", "--output-jsonl", help="可选：输出页面使用的 JSONL")
    parser.add_argument("-v2", "--v2-source", help="可选：题干/解析图片补充源")
    parser.add_argument("--all-results", action="store_true", help="渲染全部有效结果")
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--level-plan",
        type=parse_level_plan,
        help=(
            "按最终模型档位精确抽样，顺序为"
            "送分,基础,中等,拔高,压轴；"
            "例如 120,120,120,90,50"
        ),
    )
    parser.add_argument(
        "--allow-cross-level-fill",
        action="store_true",
        help=(
            "level-plan某档不足时，按相邻档位优先补齐总数；"
            "补入题仍按真实模型档位展示"
        ),
    )
    parser.add_argument("--release-label", default=CHEMISTRY_FROZEN_RELEASE)
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.input))
    _align_optional_image_source(
        rows,
        Path(args.v2_source) if args.v2_source else None,
    )
    grouped: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        level = (row.get("difficulty_rating") or {}).get("difficulty_level")
        if level in LEVEL_MAP:
            grouped[LEVEL_MAP[level]].append(row)

    selected: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    if args.all_results and args.level_plan:
        parser.error("--all-results 与 --level-plan 不能同时使用")
    if args.allow_cross_level_fill and not args.level_plan:
        parser.error("--allow-cross-level-fill 必须与 --level-plan 同时使用")
    if args.all_results:
        selected.update({level: list(grouped[level]) for level in range(1, 6)})
    elif args.level_plan:
        grouped_by_name = {
            level_name: grouped[level_num]
            for level_name, level_num in LEVEL_MAP.items()
        }
        selected_by_name = select_rows_by_level_plan(
            grouped_by_name,
            args.level_plan,
            seed=args.seed,
            allow_cross_level_fill=args.allow_cross_level_fill,
        )
        selected.update(
            {
                LEVEL_MAP[level_name]: rows_for_level
                for level_name, rows_for_level in selected_by_name.items()
            }
        )
    else:
        rng = random.Random(args.seed)
        plan = build_sample_plan(args.sample_size)
        for level_name, target_count in plan.items():
            level = LEVEL_MAP[level_name]
            pool = grouped[level]
            selected[level] = (
                rng.sample(pool, target_count)
                if len(pool) >= target_count
                else list(pool)
            )

    selected_rows = [row for level in range(1, 6) for row in selected[level]]
    if args.output_jsonl:
        output_jsonl = Path(args.output_jsonl)
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with output_jsonl.open("w", encoding="utf-8") as handle:
            for row in selected_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    generate_html_file(
        selected,
        args.output_html,
        review_scope=str(len(selected_rows)),
        release_label=args.release_label,
    )
    print(f"已渲染 {len(selected_rows)} 题：" + "、".join(
        f"{LEVEL_NAMES[level]} {len(selected[level])}题"
        for level in range(1, 6)
    ))


if __name__ == "__main__":
    main()
