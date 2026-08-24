#!/usr/bin/env python3
"""Generate an interactive review page for English knowledge-point ordering."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _split_labels(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str):
        return [part.strip() for part in value.split(";") if part.strip()]
    return []


def _source_labels(row: dict[str, Any]) -> list[str]:
    return _split_labels(row.get("original_output", row.get("output")))


def _ordered_labels(row: dict[str, Any]) -> list[str]:
    ordered = row.get("ordered_output")
    if isinstance(ordered, list):
        return _split_labels(ordered)
    return _split_labels(row.get("sorted_output", row.get("output")))


def _status(row: dict[str, Any]) -> str:
    if row.get("sort_status") == "error":
        return "error"
    source = _source_labels(row)
    ordered = _ordered_labels(row)
    if row.get("sort_status") == "success" and source != ordered:
        return "changed"
    return "unchanged"


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(_status(row) for row in rows)
    primary = Counter(
        ordered[0]
        for row in rows
        if _status(row) != "error"
        for ordered in [_ordered_labels(row)]
        if ordered
    )
    label_counts = Counter(
        len(_ordered_labels(row))
        for row in rows
        if _status(row) != "error"
    )
    return {
        "total": len(rows),
        "success": statuses["changed"] + statuses["unchanged"],
        "errors": statuses["error"],
        "changed": statuses["changed"],
        "unchanged": statuses["unchanged"],
        "top_primary": [
            {"label": label, "count": count}
            for label, count in sorted(primary.items(), key=lambda item: (-item[1], item[0]))[:12]
        ],
        "label_count_distribution": [
            {"count": count, "questions": number}
            for count, number in sorted(label_counts.items())
        ],
    }


def _escape(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _labels_html(labels: list[str], *, ordered: bool) -> str:
    if not labels:
        return '<span class="empty-label">无可用标签</span>'
    pieces: list[str] = []
    for index, label in enumerate(labels, 1):
        primary = " primary" if ordered and index == 1 else ""
        pieces.append(
            f'<span class="label-pill{primary}"><span class="label-index">{index}</span>'
            f"{_escape(label)}</span>"
        )
    return "".join(pieces)


def _card(row: dict[str, Any], index: int) -> str:
    status = _status(row)
    question_id = row.get("question_id") or row.get("id") or f"line-{index + 1}"
    source = _source_labels(row)
    ordered = _ordered_labels(row)
    raw_input = row.get("input") or row.get("question") or ""
    error = row.get("sort_error") or ""
    error_html = (
        f'<div class="error-box">排序失败：{_escape(error)}</div>' if error else ""
    )
    return f"""
    <article class="question-card" data-status="{status}" data-search="{_escape(' '.join([str(question_id), str(raw_input), *source, *ordered]))}">
      <div class="card-header">
        <span class="qid">#{index + 1} · {_escape(question_id)}</span>
        <span class="status status-{status}">{'顺序有变化' if status == 'changed' else '排序失败' if status == 'error' else '顺序未变化'}</span>
      </div>
      <details class="question-details">
        <summary>查看题目与解析原文</summary>
        <pre>{_escape(raw_input)}</pre>
      </details>
      {error_html}
      <div class="order-grid">
        <section>
          <h3>原始知识点顺序 <span>{len(source)} 项</span></h3>
          <div class="labels">{_labels_html(source, ordered=False)}</div>
        </section>
        <div class="arrow">→</div>
        <section>
          <h3>排序后知识点顺序 <span>{len(ordered)} 项</span></h3>
          <div class="labels">{_labels_html(ordered, ordered=True)}</div>
        </section>
      </div>
      <div class="annotation-section">
        <div class="annotation-title">人工复核</div>
        <div class="annotation-row">
          <button class="annotation-btn" data-annotation-id="{_escape(question_id)}" data-action="accept">✓ 排序合理</button>
          <button class="annotation-btn" data-annotation-id="{_escape(question_id)}" data-action="review">⚠ 需要复核</button>
          <span class="annotation-state" data-state-id="{_escape(question_id)}"></span>
        </div>
        <textarea class="annotation-note" data-note-id="{_escape(question_id)}" placeholder="填写排序意见或建议的知识点顺序"></textarea>
      </div>
    </article>
    """


def generate_html(rows: list[dict[str, Any]], output_path: Path, title: str) -> None:
    summary = build_summary(rows)
    stat_items = "".join(
        f'<div class="stat"><span>{_escape(label)}</span><strong>{value}</strong></div>'
        for label, value in (
            ("题目总数", summary["total"]),
            ("排序成功", summary["success"]),
            ("顺序有变化", summary["changed"]),
            ("顺序未变化", summary["unchanged"]),
            ("调用失败", summary["errors"]),
        )
    )
    max_count = max((item["count"] for item in summary["top_primary"]), default=1)
    top_bars = "".join(
        f'<div class="bar-row"><span class="bar-label">{_escape(item["label"])}</span>'
        f'<span class="bar-track"><i style="width:{max(4, item["count"] / max_count * 100):.1f}%"></i></span>'
        f'<b>{item["count"]}</b></div>'
        for item in summary["top_primary"]
    ) or '<div class="muted">暂无成功结果</div>'
    cards = "\n".join(_card(row, index) for index, row in enumerate(rows))

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>
:root {{ --ink:#25313c; --muted:#6d7a86; --line:#dbe3e8; --blue:#2f6f9f; --blue-light:#eaf3f8; --gold:#c58a2b; --red:#b94b4b; --bg:#f5f7f8; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; }}
.shell {{ max-width:1420px; margin:0 auto; padding:28px 24px 64px; }}
h1 {{ margin:0 0 8px; font-size:28px; }} .subtitle {{ color:var(--muted); margin-bottom:20px; }}
.stats {{ display:flex; flex-wrap:wrap; gap:10px; margin:18px 0; }} .stat {{ background:#fff; border:1px solid var(--line); border-radius:10px; padding:11px 16px; min-width:130px; }} .stat span {{ display:block; color:var(--muted); font-size:13px; }} .stat strong {{ display:block; font-size:24px; margin-top:3px; }}
.toolbar {{ position:sticky; top:0; z-index:5; background:rgba(245,247,248,.95); padding:10px 0; display:flex; gap:10px; flex-wrap:wrap; }} .toolbar input {{ flex:1 1 300px; min-height:38px; border:1px solid var(--line); border-radius:8px; padding:0 12px; font-size:14px; }} button {{ border:1px solid var(--line); background:#fff; color:var(--ink); border-radius:8px; padding:9px 13px; cursor:pointer; }} button.active {{ background:var(--blue); border-color:var(--blue); color:#fff; }}
.overview {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(260px,360px); gap:16px; margin:10px 0 20px; }} .panel {{ background:#fff; border:1px solid var(--line); border-radius:12px; padding:16px; }} .panel h2 {{ font-size:16px; margin:0 0 12px; }} .bar-row {{ display:grid; grid-template-columns:minmax(120px,1fr) 2fr 35px; align-items:center; gap:9px; margin:8px 0; font-size:13px; }} .bar-label {{ overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }} .bar-track {{ height:10px; background:#edf1f3; border-radius:5px; overflow:hidden; }} .bar-track i {{ display:block; height:100%; background:var(--blue); border-radius:5px; }} .muted {{ color:var(--muted); }}
.question-card {{ background:#fff; border:1px solid var(--line); border-left:4px solid #b9c7d0; border-radius:12px; padding:16px; margin:12px 0; }} .question-card[data-status="changed"] {{ border-left-color:var(--gold); }} .question-card[data-status="error"] {{ border-left-color:var(--red); }} .card-header {{ display:flex; justify-content:space-between; gap:12px; align-items:center; }} .qid {{ font-weight:700; }} .status {{ font-size:12px; border-radius:999px; padding:4px 8px; background:#eef2f4; color:var(--muted); }} .status-changed {{ background:#fff3dc; color:#8b5a13; }} .status-error {{ background:#fdeaea; color:var(--red); }}
.question-details {{ margin:12px 0; }} summary {{ cursor:pointer; color:var(--blue); font-size:13px; }} pre {{ max-height:240px; overflow:auto; white-space:pre-wrap; background:#f7f9fa; padding:12px; border-radius:8px; font:12px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace; }}
.order-grid {{ display:grid; grid-template-columns:minmax(0,1fr) 30px minmax(0,1fr); gap:12px; align-items:center; }} .order-grid section {{ min-width:0; }} h3 {{ margin:0 0 8px; font-size:14px; }} h3 span {{ font-weight:400; color:var(--muted); font-size:12px; }} .labels {{ display:flex; flex-wrap:wrap; gap:7px; }} .label-pill {{ display:inline-flex; gap:6px; align-items:center; border:1px solid #d7e1e7; background:#f8fafb; border-radius:7px; padding:7px 9px; font-size:13px; line-height:1.35; word-break:break-all; }} .label-pill.primary {{ border-color:#9bc4dc; background:var(--blue-light); }} .label-index {{ color:var(--muted); font-size:11px; font-weight:700; }} .arrow {{ color:var(--gold); font-size:24px; text-align:center; }} .empty-label {{ color:var(--muted); font-size:13px; }} .error-box {{ background:#fff3f3; color:var(--red); border-radius:7px; padding:9px 11px; font-size:13px; margin:8px 0; }}
.annotation-section {{ border-top:1px solid var(--line); margin-top:15px; padding-top:12px; }} .annotation-title {{ font-weight:700; font-size:13px; margin-bottom:8px; }} .annotation-row {{ display:flex; align-items:center; flex-wrap:wrap; gap:8px; }} .annotation-btn {{ padding:7px 10px; font-size:12px; }} .annotation-btn.selected {{ background:#eaf3f8; border-color:#9bc4dc; color:var(--blue); }} .annotation-state {{ color:var(--muted); font-size:12px; }} .annotation-note {{ display:block; width:100%; min-height:58px; margin-top:9px; resize:vertical; border:1px solid var(--line); border-radius:7px; padding:8px; font:13px/1.5 inherit; }}
.hidden {{ display:none !important; }} @media (max-width:760px) {{ .shell {{ padding:18px 12px 40px; }} h1 {{ font-size:23px; }} .overview {{ grid-template-columns:1fr; }} .order-grid {{ grid-template-columns:1fr; }} .arrow {{ transform:rotate(90deg); }} .card-header {{ align-items:flex-start; flex-direction:column; gap:7px; }} }}
</style>
</head>
<body>
<main class="shell">
  <h1>{_escape(title)}</h1>
  <div class="subtitle">模型只负责重排已有知识点；程序已校验每条成功结果与原始标签多重集合一致。</div>
  <section class="stats">{stat_items}</section>
  <section class="overview">
    <div class="panel"><h2>排序后第一知识点 Top 12</h2>{top_bars}</div>
    <div class="panel"><h2>使用说明</h2><div class="muted">用下方筛选查看顺序发生变化的题目；点击“查看题目与解析原文”展开输入信息。蓝色标签为排序后的第 1 项。</div></div>
  </section>
  <div class="toolbar" aria-label="筛选">
    <input id="search" type="search" placeholder="搜索题目 ID、题干或知识点">
    <button class="active" data-filter="all">全部</button>
    <button data-filter="changed">顺序有变化</button>
    <button data-filter="unchanged">顺序未变化</button>
    <button data-filter="error">调用失败</button>
  </div>
  <section id="cards">{cards}</section>
</main>
<script>
let filter = 'all';
const cards = [...document.querySelectorAll('.question-card')];
const search = document.getElementById('search');
function applyFilter() {{
  const query = search.value.trim().toLowerCase();
  cards.forEach(card => {{
    const statusOk = filter === 'all' || card.dataset.status === filter;
    const textOk = !query || (card.dataset.search || '').toLowerCase().includes(query);
    card.classList.toggle('hidden', !(statusOk && textOk));
  }});
}}
document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {{
  filter = button.dataset.filter;
  document.querySelectorAll('[data-filter]').forEach(item => item.classList.toggle('active', item === button));
  applyFilter();
}}));
search.addEventListener('input', applyFilter);
const annotationKey = 'junior_english_knowledge_point_order_annotations_v1';
let annotations = {{}};
try {{ annotations = JSON.parse(localStorage.getItem(annotationKey) || '{{}}'); }} catch (error) {{ annotations = {{}}; }}
function saveAnnotations() {{ localStorage.setItem(annotationKey, JSON.stringify(annotations)); }}
function renderAnnotation(id) {{
  const value = annotations[id] || {{}};
  document.querySelectorAll('[data-annotation-id="' + CSS.escape(id) + '"]').forEach(button => button.classList.toggle('selected', button.dataset.action === value.action));
  const state = document.querySelector('[data-state-id="' + CSS.escape(id) + '"]');
  if (state) state.textContent = value.action === 'accept' ? '已标记为合理' : value.action === 'review' ? '已标记为需要复核' : '';
  const note = document.querySelector('[data-note-id="' + CSS.escape(id) + '"]');
  if (note && note.value !== (value.note || '')) note.value = value.note || '';
}}
document.querySelectorAll('[data-annotation-id]').forEach(button => button.addEventListener('click', () => {{
  const id = button.dataset.annotationId;
  annotations[id] = {{ ...(annotations[id] || {{}}), action: button.dataset.action }};
  saveAnnotations(); renderAnnotation(id);
}}));
document.querySelectorAll('[data-note-id]').forEach(note => note.addEventListener('input', () => {{
  const id = note.dataset.noteId;
  annotations[id] = {{ ...(annotations[id] || {{}}), note: note.value }};
  saveAnnotations();
}}));
document.querySelectorAll('[data-state-id]').forEach(state => renderAnnotation(state.dataset.stateId));
</script>
</body>
</html>"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"第 {line_number} 行不是 JSON 对象")
            rows.append(value)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="排序结果 JSONL")
    parser.add_argument("--output", required=True, help="输出 HTML")
    parser.add_argument("--title", default="初中英语知识点排序验收")
    args = parser.parse_args()
    rows = _read_jsonl(Path(args.input).expanduser().resolve())
    output_path = Path(args.output).expanduser().resolve()
    generate_html(rows, output_path, args.title)
    summary = build_summary(rows)
    print(
        f"HTML 已生成：{output_path}；题目 {summary['total']}，"
        f"成功 {summary['success']}，失败 {summary['errors']}，变化 {summary['changed']}"
    )


if __name__ == "__main__":
    main()
