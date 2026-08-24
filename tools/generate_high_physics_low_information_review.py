#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成高中物理“低信息负担组合”定向复核页面。"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


LOW_INFORMATION_SIGNATURE = {
    "information_carrier": "单一示意图",
    "graph_structure": "无图表",
    "drawing_requirement": "无",
    "experiment_requirement": "无",
    "context_type": "纯物理",
    "context_load": "纯包装",
}

SIGNATURE_LABELS = {
    "information_carrier": "信息载体",
    "graph_structure": "图像结构",
    "drawing_requirement": "作图要求",
    "experiment_requirement": "实验要求",
    "context_type": "情境类型",
    "context_load": "情境负担",
}

STRUCTURE_FIELDS = (
    ("step_count", "有效步骤"),
    ("process_count", "物理过程"),
    ("state_count", "状态数量"),
    ("model_relation", "模型关系"),
    ("constraint_structure", "约束结构"),
    ("reasoning_chain", "推理链"),
    ("hidden_conditions", "隐含条件"),
    ("critical_state", "临界状态"),
    ("classification_discussion", "分类讨论"),
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} 不是合法 JSON") from exc
            if row.get("question_id") is None:
                raise ValueError(f"{path}:{line_number} 缺少 question_id")
            rows.append(row)
    return rows


def load_reviewed_levels(path: Path) -> dict[str, str]:
    levels: dict[str, str] = {}
    for row in read_jsonl(path):
        level = (
            row.get("reviewed_difficulty_level")
            or row.get("difficulty_level")
            or row.get("label")
        )
        if isinstance(level, str) and level:
            levels[str(row["question_id"])] = level
    return levels


def select_level4_agreements(
    rows: list[dict[str, Any]],
    reviewed_levels: dict[str, str],
) -> list[dict[str, Any]]:
    """筛出六项组合命中、参考复核与模型均为难度4的题目。"""
    selected: list[dict[str, Any]] = []
    for row in rows:
        question_id = str(row.get("question_id"))
        stage1 = row.get("difficulty_rating_stage1") or {}
        features = stage1.get("features") or {}
        signature_matches = all(
            features.get(field) == value
            for field, value in LOW_INFORMATION_SIGNATURE.items()
        )
        if (
            signature_matches
            and reviewed_levels.get(question_id) == "难度4档"
            and row.get("final_difficulty_level") == "难度4档"
        ):
            selected.append(row)
    return selected


def _text(value: Any) -> str:
    return html.escape(str(value or ""))


def _image_block(url: Any, title: str) -> str:
    safe_url = str(url or "").strip()
    if not safe_url:
        return (
            '<section class="image-section">'
            f"<h3>{_text(title)}</h3><span>无可用 URL</span>"
            "</section>"
        )
    return (
        '<section class="image-section">'
        f"<h3>{_text(title)}</h3>"
        f'<img src="{html.escape(safe_url, quote=True)}" alt="{_text(title)}" '
        'loading="lazy" onclick="openImage(this.src)"></section>'
    )


def _chips(values: list[tuple[str, Any]], css_class: str) -> str:
    return "".join(
        f'<span class="chip {css_class}"><b>{_text(label)}</b>：{_text(value)}</span>'
        for label, value in values
    )


def _record_for_export(row: dict[str, Any]) -> dict[str, Any]:
    stage1 = row.get("difficulty_rating_stage1") or {}
    features = stage1.get("features") or {}
    return {
        "question_id": str(row.get("question_id")),
        "current_level": row.get("final_difficulty_level", ""),
        "original_predicted_accuracy": stage1.get(
            "original_predicted_accuracy"
        ),
        "adjusted_predicted_accuracy": stage1.get("predicted_accuracy"),
        "high_difficulty_feature_count": stage1.get(
            "high_difficulty_feature_count"
        ),
        "high_difficulty_features": stage1.get("high_difficulty_features") or [],
        "low_information_signature": {
            field: features.get(field)
            for field in LOW_INFORMATION_SIGNATURE
        },
        "structural_features": {
            field: features.get(field) for field, _ in STRUCTURE_FIELDS
        },
    }


def render_review_html(rows: list[dict[str, Any]]) -> str:
    records = [_record_for_export(row) for row in rows]
    cards: list[str] = []
    for index, (row, record) in enumerate(zip(rows, records), start=1):
        stage1 = row.get("difficulty_rating_stage1") or {}
        features = stage1.get("features") or {}
        signature_chips = _chips(
            [
                (SIGNATURE_LABELS[field], features.get(field, ""))
                for field in LOW_INFORMATION_SIGNATURE
            ],
            "chip-low",
        )
        structure_chips = _chips(
            [(label, features.get(field, "")) for field, label in STRUCTURE_FIELDS],
            "chip-structure",
        )
        high_names = stage1.get("high_difficulty_features") or []
        high_names_text = "、".join(str(name) for name in high_names) or "无"
        stem_image = row.get("stem_image_url") or row.get("stem_pic_url")
        analysis_image = row.get("analysis_image_url") or row.get("analysis_pic_url")
        cards.append(
            f'''
<article class="question-card" data-question-id="{_text(record["question_id"])}">
  <header class="card-header">
    <div><span class="ordinal">#{index:02d}</span><span class="question-id">题目ID：{_text(record["question_id"])}</span></div>
    <span class="level-badge">当前结论：难度4档</span>
  </header>
  <section class="score-panel">
    <div><span>原始预测正确率</span><strong>{_text(record["original_predicted_accuracy"])}</strong></div>
    <div><span>乘数后预测正确率</span><strong>{_text(record["adjusted_predicted_accuracy"])}</strong></div>
    <div><span>高难特征数量</span><strong>{_text(record["high_difficulty_feature_count"])}</strong></div>
    <div class="high-name"><span>已识别高难结构</span><strong>{_text(high_names_text)}</strong></div>
  </section>
  <section>
    <h3>老师反馈命中的低信息负担组合</h3>
    <div class="chips">{signature_chips}</div>
  </section>
  <section>
    <h3>题目实际结构</h3>
    <div class="chips">{structure_chips}</div>
  </section>
  {_image_block(stem_image, "题目图片（点击放大）")}
  {_image_block(analysis_image, "解析图片（点击放大）")}
  <section class="review-panel">
    <h3>老师复核意见</h3>
    <div class="decision-row">
      <label><input type="radio" name="decision-{index}" value="保持难度4档"> 保持难度4档</label>
      <label><input type="radio" name="decision-{index}" value="下调为难度3档"> 下调为难度3档</label>
      <label><input type="radio" name="decision-{index}" value="相邻档边界，可接受"> 相邻档边界，可接受</label>
      <label><input type="radio" name="decision-{index}" value="题干或解析信息不足"> 题干或解析信息不足</label>
    </div>
    <textarea placeholder="请说明是否应因低信息负担组合降档；若不应降档，请指出实际的核心难点。"></textarea>
  </section>
</article>'''
        )

    embedded_records = json.dumps(records, ensure_ascii=False).replace("</", "<\\/")
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>高中物理低信息负担组合复核（{len(rows)}题）</title>
<style>
  :root {{ color-scheme: light; --ink:#1e293b; --muted:#64748b; --line:#dbe4ee; --surface:#fff; --page:#f6f8fc; --blue:#1d4ed8; --blue-soft:#dbeafe; --orange:#c2410c; --orange-soft:#ffedd5; --green:#166534; --green-soft:#dcfce7; }}
  * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--page); color:var(--ink); font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif; }}
  .hero {{ background:linear-gradient(135deg,#0f3b8c,#2563eb); color:#fff; padding:28px max(20px,calc((100% - 1200px)/2)); }} .hero h1 {{ margin:0 0 8px; font-size:25px; }} .hero p {{ margin:0; opacity:.92; }}
  .summary {{ max-width:1200px; margin:18px auto; padding:0 20px; display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }} .summary div {{ background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:13px 16px; }} .summary strong {{ display:block; font-size:20px; color:var(--blue); }}
  .toolbar {{ position:sticky; top:0; z-index:5; background:rgba(246,248,252,.96); border-bottom:1px solid var(--line); padding:12px 20px; text-align:center; backdrop-filter:blur(8px); }} button {{ border:0; border-radius:8px; background:var(--blue); color:#fff; font-weight:700; padding:9px 16px; cursor:pointer; }}
  main {{ max-width:1200px; margin:0 auto 48px; padding:0 20px; }} .question-card {{ background:var(--surface); border:1px solid var(--line); border-radius:12px; margin:18px 0; overflow:hidden; box-shadow:0 2px 8px rgba(15,23,42,.04); }}
  .card-header {{ padding:14px 18px; background:#f8fafc; border-bottom:1px solid var(--line); display:flex; justify-content:space-between; gap:12px; align-items:center; }} .ordinal {{ color:var(--blue); font-weight:800; margin-right:12px; }} .question-id {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:13px; }} .level-badge {{ background:var(--orange-soft); color:var(--orange); border-radius:999px; padding:4px 10px; font-weight:700; white-space:nowrap; }}
  .score-panel {{ margin:16px 18px; display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); border:1px solid var(--line); border-radius:9px; overflow:hidden; }} .score-panel div {{ padding:10px 12px; border-right:1px solid var(--line); }} .score-panel div:last-child {{ border:0; }} .score-panel span {{ display:block; color:var(--muted); font-size:12px; }} .score-panel strong {{ font-size:16px; }} .score-panel .high-name {{ grid-column:span 1; }}
  section {{ margin:16px 18px; }} h3 {{ font-size:15px; margin:0 0 8px; }} .chips {{ display:flex; flex-wrap:wrap; gap:7px; }} .chip {{ border-radius:6px; padding:4px 8px; font-size:13px; }} .chip-low {{ background:var(--green-soft); color:var(--green); }} .chip-structure {{ background:var(--blue-soft); color:#1e40af; }}
  .image-section {{ border-left:3px solid #cbd5e1; padding-left:12px; }} .image-section span {{ color:var(--muted); }} .image-section img {{ max-width:min(100%,900px); max-height:680px; display:block; border:1px solid var(--line); border-radius:8px; cursor:zoom-in; background:#fff; }}
  .review-panel {{ padding:14px; border:1px solid #bfdbfe; background:#eff6ff; border-radius:9px; }} .decision-row {{ display:flex; flex-wrap:wrap; gap:12px 18px; }} .decision-row label {{ cursor:pointer; }} textarea {{ width:100%; min-height:86px; margin-top:12px; border:1px solid #93c5fd; border-radius:7px; padding:9px; font:inherit; resize:vertical; }}
  #lightbox {{ display:none; position:fixed; inset:0; z-index:10; background:rgba(15,23,42,.94); padding:35px; overflow:auto; }} #lightbox.open {{ display:flex; justify-content:center; align-items:flex-start; }} #lightbox img {{ max-width:none; max-height:none; border:0; cursor:zoom-out; }}
  @media (max-width:760px) {{ .summary {{ grid-template-columns:1fr; }} .score-panel {{ grid-template-columns:repeat(2,1fr); }} .score-panel .high-name {{ grid-column:span 2; }} .score-panel div:nth-child(2) {{ border-right:0; }} .card-header {{ align-items:flex-start; flex-direction:column; }} }}
</style>
</head>
<body>
<header class="hero"><h1>高中物理｜低信息负担组合复核</h1><p>仅包含六项低信息负担特征同时满足，且当前结论与参考复核均为难度4档的 {len(rows)} 道题。</p></header>
<section class="summary"><div><span>复核题数</span><strong>{len(rows)} 道</strong></div><div><span>待验证规则</span><strong>是否应至少降一档</strong></div><div><span>当前档位</span><strong>难度4档</strong></div></section>
<div class="toolbar"><button type="button" onclick="exportFeedback()">导出老师反馈</button></div>
<main>{''.join(cards)}</main>
<div id="lightbox" onclick="this.classList.remove('open')"><img alt="放大图片"></div>
<script>
const reviewRecords = {embedded_records};
function openImage(src) {{ const box=document.getElementById('lightbox'); box.querySelector('img').src=src; box.classList.add('open'); }}
function exportFeedback() {{
  const feedback = reviewRecords.map((record, index) => {{
    const card = document.querySelectorAll('.question-card')[index];
    const selected = card.querySelector('input[type="radio"]:checked');
    return {{...record, teacher_decision: selected ? selected.value : '', teacher_note: card.querySelector('textarea').value.trim()}};
  }});
  const blob = new Blob([JSON.stringify(feedback, null, 2)], {{type:'application/json;charset=utf-8'}});
  const link = document.createElement('a'); link.href=URL.createObjectURL(blob); link.download='高中物理低信息负担组合_17题老师反馈.json'; link.click(); URL.revokeObjectURL(link.href);
}}
</script>
</body>
</html>'''


def main() -> None:
    parser = argparse.ArgumentParser(description="生成高中物理低信息负担组合复核页")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = select_level4_agreements(
        read_jsonl(args.results),
        load_reviewed_levels(args.labels),
    )
    if not selected:
        raise ValueError("没有筛出符合条件的题目")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_review_html(selected), encoding="utf-8")
    print(f"已生成 {len(selected)} 道题复核页：{args.output.resolve()}")


if __name__ == "__main__":
    main()
