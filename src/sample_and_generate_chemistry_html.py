# -*- coding: utf-8 -*-
"""
@File    : sample_and_generate_chemistry_html.py
@Description:
    从初中化学难度打标结果中按档抽样，或直接渲染全部结果，生成交互式
    评议验收网页。
    - 评级判定：读取 chemistry_stable 的最终 difficulty_rating。
    - 特征展示：对齐初中化学教师口径的受控特征契约。
    - 图片展示：优先使用原始 V2 数据中的题干和解析图片 URL。
"""

import json
import os
import html
import random
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

# 抽样配比计划 (对齐物理 500 题配比)
SAMPLE_PLAN = {
    "送分题": 100,
    "基础题": 120,
    "中等题": 120,
    "拔高题": 100,
    "压轴题": 60
}


def build_sample_plan(sample_size: int) -> Dict[str, int]:
    """按既有 500 题配比缩放抽样计划，并保证总数精确。"""
    if sample_size <= 0:
        raise ValueError("sample_size 必须大于 0")
    total_weight = sum(SAMPLE_PLAN.values())
    exact = {
        level: sample_size * weight / total_weight
        for level, weight in SAMPLE_PLAN.items()
    }
    plan = {level: int(value) for level, value in exact.items()}
    remainder = sample_size - sum(plan.values())
    ranked = sorted(
        SAMPLE_PLAN,
        key=lambda level: (
            exact[level] - plan[level],
            SAMPLE_PLAN[level],
        ),
        reverse=True,
    )
    for level in ranked[:remainder]:
        plan[level] += 1
    return plan


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

def escape(text: str) -> str:
    if not text:
        return ""
    return html.escape(str(text))


def split_image_urls(value: Any) -> List[str]:
    """把图片字段归一化为去重且保持原顺序的 URL 列表。"""
    if isinstance(value, (list, tuple)):
        candidates = [str(item).strip() for item in value]
    else:
        candidates = [
            part.strip()
            for part in str(value or "").replace("，", ",").split(",")
        ]
    return list(dict.fromkeys(url for url in candidates if url))


def contains_image_reference(*values: Any) -> bool:
    """判断题干、选项、解析或小题是否明确引用了图片。"""
    markers = (
        "<image",
        "<img",
        "如下图",
        "如图",
        "图中",
        "下图",
        "图示",
        "装置图",
        "流程图",
        "曲线",
        "坐标图",
        "关系图",
        "微观示意",
        "粒子图",
        "表格",
    )
    for value in values:
        if isinstance(value, (list, tuple)):
            if contains_image_reference(*value):
                return True
            continue
        if isinstance(value, dict):
            if contains_image_reference(*value.values()):
                return True
            continue
        text = str(value or "").lower()
        if any(marker in text for marker in markers):
            return True
    return False


def partition_visualization_images(
    value: Any,
    *,
    prefer_second_png_when_two: bool = False,
) -> Tuple[List[str], List[str]]:
    """把完整截图作为主图，把原始题图收进折叠备用区。

    数据中的 ``/image/externalized/`` 图片通常已经包含完整题干或解析文字，
    以及嵌入其中的题图。存在这种图片时，只默认展示完整截图；其他原始
    图片保留在“查看原始题图 / 备用图片”中，避免同一题图重复占据页面。
    如果没有完整截图，则保留物理可视化的两 PNG 顺序规则：题干第二张
    作为主图，第一张折叠备用；其他情况全部作为主图。
    """
    urls = split_image_urls(value)
    externalized = [
        url for url in urls if "/image/externalized/" in url.lower()
    ]
    if externalized:
        externalized_set = set(externalized)
        supporting = [
            url for url in urls if url not in externalized_set
        ]
        return externalized, supporting
    if (
        prefer_second_png_when_two
        and len(urls) == 2
        and all(
            url.lower().split("?", 1)[0].endswith(".png")
            for url in urls
        )
    ):
        return [urls[1]], [urls[0]]
    return urls, []


def render_image_section(
    value: Any,
    *,
    title: str,
    kind: str,
    content_requires_image: bool = False,
) -> str:
    """渲染主图和折叠备用图，所有图片均可点击放大。"""
    primary, supporting = partition_visualization_images(
        value,
        prefer_second_png_when_two=(kind == "stem"),
    )
    if not primary:
        if content_requires_image:
            empty_text = (
                f"【{escape(title)}资源缺失：题目正文包含图示标记，"
                "但数据未提供图片 URL】"
            )
            empty_class = "media-empty media-missing"
        else:
            empty_text = f"【该题无{escape(title)}】"
            empty_class = "media-empty"
        return f"""
                <div class="media-section media-section-empty">
                    <div class="{empty_class}">{empty_text}</div>
                </div>
"""

    safe_kind = escape(kind)
    pieces = [
        '                <div class="media-section">',
        '                    <div class="media-heading">',
        f'                        <span>{escape(title)}</span>',
        (
            '                        <span class="media-hint">'
            "点击图片可放大查看</span>"
        ),
        '                    </div>',
        (
            '                    <div class="image-container '
            'image-container-primary">'
        ),
    ]
    for index, url in enumerate(primary, 1):
        source_kind = (
            "externalized"
            if "/image/externalized/" in url.lower()
            else "original"
        )
        number_suffix = f" {index}" if len(primary) > 1 else ""
        pieces.append(
            '                        '
            '<figure class="image-frame primary-image">'
            f'<img src="{html.escape(url)}" '
            f'alt="{escape(title)} {index}" '
            f'data-image-role="{safe_kind}-primary" '
            f'data-image-source="{source_kind}" '
            'onclick="openImagePreview(this)" '
            'onerror="markImageFailed(this)">'
            f'<figcaption>完整{escape(title)}{number_suffix}'
            "</figcaption></figure>"
        )
    pieces.append("                    </div>")

    if supporting:
        pieces.extend(
            [
                '                    <details class="supporting-images">',
                (
                    "                        <summary>"
                    "查看原始题图 / 备用图片"
                    f"（{len(supporting)}张）</summary>"
                ),
                (
                    '                        <div class="image-container '
                    'image-container-supporting">'
                ),
            ]
        )
        for index, url in enumerate(supporting, 1):
            pieces.append(
                '                            '
                '<figure class="image-frame supporting-image">'
                f'<img src="{html.escape(url)}" '
                f'alt="{escape(title)}备用图 {index}" '
                f'data-image-role="{safe_kind}-supporting" '
                'onclick="openImagePreview(this)" '
                'onerror="markImageFailed(this)">'
                f"<figcaption>原始题图 {index}</figcaption></figure>"
            )
        pieces.extend(
            [
                "                        </div>",
                "                    </details>",
            ]
        )

    pieces.append("                </div>")
    return "\n".join(pieces) + "\n"


# HTML 网页基础骨架模板 (针对化学优化)
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>初中化学评级验收面板 (500题纯图片可视化优化版)</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f7fb;
            color: #333;
            line-height: 1.6;
            padding-top: 160px;
        }

        /* ===== Header ===== */
        .header {
            background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
            color: white;
            padding: 18px 20px;
            text-align: center;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 20px; margin-bottom: 0; font-weight: 700; letter-spacing: 1px; }

        /* ===== Stats Bar ===== */
        .stats-bar {
            background: white;
            padding: 10px 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 20px;
            flex-wrap: wrap;
            border-bottom: 1px solid #e0e0e0;
            position: fixed;
            top: 56px;
            left: 0;
            right: 0;
            z-index: 999;
            box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        }
        .stats-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            color: #555;
        }
        .stats-item .stats-value {
            font-weight: 700;
            color: #11998e;
        }
        .stats-level {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
            padding: 3px 10px;
            border-radius: 12px;
        }
        .stats-level-1 { background: #E8F5E9; color: #2E7D32; }
        .stats-level-2 { background: #F1F8E9; color: #558B2F; }
        .stats-level-3 { background: #FFF8E1; color: #F9A825; }
        .stats-level-4 { background: #FFF3E0; color: #EF6C00; }
        .stats-level-5 { background: #FFEBEE; color: #C62828; }

        /* ===== Nav ===== */
        .nav {
            background: white;
            padding: 10px 20px;
            display: flex;
            justify-content: center;
            gap: 8px;
            flex-wrap: wrap;
            border-bottom: 1px solid #e0e0e0;
            position: fixed;
            top: 100px;
            left: 0;
            right: 0;
            z-index: 998;
        }
        .nav a {
            padding: 8px 18px;
            border-radius: 8px;
            text-decoration: none;
            color: #666;
            background: #f5f5f5;
            transition: all 0.25s;
            font-size: 14px;
            font-weight: 500;
            border: 1px solid transparent;
        }
        .nav a:hover {
            background: #11998e;
            color: white;
            border-color: #11998e;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(17,153,142,0.3);
        }
        .nav a.nav-active {
            background: #11998e;
            color: white;
            border-color: #11998e;
        }

        /* ===== Export Bar ===== */
        .export-bar {
            position: fixed;
            top: 56px;
            right: 16px;
            z-index: 1001;
            display: flex;
            gap: 8px;
        }
        .export-btn {
            padding: 6px 14px;
            border-radius: 8px;
            border: 1px solid #ddd;
            background: white;
            color: #555;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }
        .export-btn:hover {
            background: #11998e;
            color: white;
            border-color: #11998e;
        }

        /* ===== Level Section ===== */
        .level-section {
            max-width: 1440px;
            margin: 0 auto;
            padding: 20px;
        }
        .level-header {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            border-left: 5px solid;
        }
        .level-1 { border-left-color: #4CAF50; }
        .level-2 { border-left-color: #8BC34A; }
        .level-3 { border-left-color: #FFC107; }
        .level-4 { border-left-color: #FF9800; }
        .level-5 { border-left-color: #F44336; }
        .level-title { font-size: 22px; font-weight: bold; margin-bottom: 4px; }
        .level-desc { color: #666; font-size: 14px; }

        /* ===== Question Card ===== */
        .question-card {
            background: white;
            margin-bottom: 25px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            overflow: hidden;
            border: 1px solid #eef2f6;
        }
        .question-header {
            padding: 12px 20px;
            background: #fafbfc;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }
        .question-id {
            font-family: monospace;
            font-size: 12px;
            color: #999;
        }
        .question-tags {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
        }
        .tag {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }
        .tag-time { background: #E3F2FD; color: #1565C0; }
        .tag-tokens { background: #F3E5F5; color: #6A1B9A; }

        /* ===== Big Difficulty Badge ===== */
        .difficulty-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 6px 20px;
            border-radius: 8px;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: 2px;
            margin: 12px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .badge-1 { background: #4CAF50; color: white; }
        .badge-2 { background: #8BC34A; color: white; }
        .badge-3 { background: #FFC107; color: #333; }
        .badge-4 { background: #FF9800; color: white; }
        .badge-5 { background: #F44336; color: white; }

        .question-body { padding: 20px; }

        /* ===== Smart Image Sections ===== */
        .media-section {
            margin: 18px 0 24px;
        }
        .media-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 16px;
            margin-bottom: 8px;
            color: #334155;
            font-size: 17px;
            font-weight: 700;
        }
        .media-hint {
            color: #64748b;
            font-size: 13px;
            font-weight: 500;
        }
        .media-empty {
            padding: 10px 0;
            color: #94a3b8;
            font-size: 14px;
            font-style: italic;
        }
        .media-missing {
            padding: 12px 14px;
            border: 1px dashed #f59e0b;
            border-radius: 8px;
            background: #fffbeb;
            color: #b45309;
            font-weight: 650;
        }
        .image-container {
            margin: 0;
            text-align: left;
            background: #fff;
            padding: 16px;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }
        .image-frame {
            margin: 0;
        }
        .image-frame img {
            display: block;
            height: auto;
            object-fit: contain;
            border-radius: 6px;
            border: 1px solid #e2e8f0;
            background: white;
            cursor: zoom-in;
        }
        .primary-image img {
            width: auto;
            max-width: min(100%, 980px);
            max-height: none;
            margin: 0 auto;
        }
        .primary-image img.image-layout-long-document {
            width: 100%;
            max-width: 1320px;
        }
        .primary-image img.image-layout-standard {
            width: auto;
            max-width: min(100%, 980px);
        }
        .image-frame figcaption {
            margin-top: 7px;
            color: #64748b;
            font-size: 13px;
            text-align: center;
        }
        .supporting-images {
            margin-top: 10px;
            border: 1px solid #dbe4ee;
            border-radius: 9px;
            background: #f8fafc;
        }
        .supporting-images summary {
            padding: 11px 14px;
            color: #475569;
            font-size: 14px;
            font-weight: 650;
            cursor: pointer;
            user-select: none;
        }
        .image-container-supporting {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            border: 0;
            border-top: 1px solid #dbe4ee;
            border-radius: 0 0 9px 9px;
            background: #f8fafc;
        }
        .supporting-image img {
            width: 100%;
            max-height: 440px;
            margin: 0 auto;
        }
        .image-error {
            padding: 18px;
            border: 1px dashed #f87171;
            border-radius: 6px;
            background: #fef2f2;
            color: #b91c1c;
            font-size: 14px;
        }

        /* ===== Image Lightbox ===== */
        .image-lightbox {
            position: fixed;
            inset: 0;
            z-index: 3000;
            display: none;
            overflow: auto;
            padding: 72px 3vw 40px;
            background: rgba(15, 23, 42, 0.94);
            cursor: zoom-out;
        }
        .image-lightbox.open {
            display: flex;
            align-items: flex-start;
            justify-content: center;
        }
        .image-lightbox img {
            display: block;
            width: auto;
            min-width: min(1100px, 94vw);
            max-width: none;
            height: auto;
            margin: 0 auto;
            border-radius: 8px;
            background: white;
            box-shadow: 0 18px 60px rgba(0,0,0,0.5);
            cursor: default;
        }
        .image-lightbox-close {
            position: fixed;
            top: 18px;
            right: 24px;
            width: 42px;
            height: 42px;
            border: 1px solid rgba(255,255,255,0.5);
            border-radius: 50%;
            background: rgba(15,23,42,0.75);
            color: white;
            font-size: 28px;
            line-height: 38px;
            cursor: pointer;
        }

        /* ===== Rating Section ===== */
        .rating-section {
            margin-top: 20px;
            padding: 20px;
            background: #f4faf4;
            border-radius: 8px;
            border: 1px solid #d4eed5;
        }
        .rating-title {
            font-weight: bold;
            color: #2E7D32;
            margin-bottom: 12px;
            font-size: 21px;
        }
        .rating-reasoning {
            font-size: 18px;
            color: #555;
            margin-bottom: 16px;
            line-height: 1.85;
            background: white;
            padding: 16px 18px;
            border-radius: 6px;
            border: 1px solid #e2f0d9;
        }
        .rating-reasoning strong {
            color: #1f4f23;
            font-size: 18px;
        }
        .rating-details {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
            gap: 10px;
            font-size: 16px;
        }
        .rating-detail-item {
            padding: 10px 12px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e2f0d9;
        }
        .rating-detail-item .label {
            color: #7f8c8d;
            font-size: 15px;
        }
        .rating-detail-item .value {
            color: #2c3e50;
            margin-top: 4px;
            font-size: 17px;
            font-weight: 600;
            line-height: 1.45;
        }

        /* ===== Annotation Section ===== */
        .annotation-section {
            margin-top: 20px;
            padding: 15px;
            background: #fff9e6;
            border-radius: 8px;
            border: 1px solid #ffe8a3;
        }
        .annotation-title {
            font-weight: bold;
            color: #b78a00;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .annotation-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }
        .annotation-row label {
            font-size: 13px;
            color: #555;
            font-weight: 500;
        }
        .annotation-btn {
            padding: 6px 18px;
            border-radius: 8px;
            border: 2px solid #ccc;
            background: white;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .annotation-btn:hover { transform: translateY(-1px); }
        .annotation-btn.btn-correct {
            border-color: #4CAF50;
            color: #4CAF50;
        }
        .annotation-btn.btn-correct.active {
            background: #4CAF50;
            color: white;
            box-shadow: 0 2px 8px rgba(76,175,80,0.3);
        }
        .annotation-btn.btn-wrong {
            border-color: #F44336;
            color: #F44336;
        }
        .annotation-btn.btn-wrong.active {
            background: #F44336;
            color: white;
            box-shadow: 0 2px 8px rgba(244,67,54,0.3);
        }
        .annotation-btn.btn-unmark {
            border-color: #999;
            color: #999;
        }
        .annotation-btn.btn-unmark.active {
            background: #999;
            color: white;
        }
        .annotation-textarea {
            width: 100%;
            min-height: 60px;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 13px;
            resize: vertical;
            font-family: inherit;
            line-height: 1.5;
        }
        .annotation-textarea:focus {
            outline: none;
            border-color: #FF9800;
            box-shadow: 0 0 0 3px rgba(255,152,0,0.1);
        }
        .corrected-level-select {
            min-width: 180px;
            padding: 7px 10px;
            border: 1px solid #ddd;
            border-radius: 8px;
            background: white;
            font-size: 13px;
        }

        /* ===== Back to Top ===== */
        .back-to-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            background: #11998e;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            box-shadow: 0 2px 10px rgba(0,0,0,0.2);
            text-decoration: none;
            font-size: 20px;
            transition: all 0.3s;
            z-index: 100;
        }
        .back-to-top:hover {
            background: #38ef7d;
            transform: translateY(-3px);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>初中化学评级验收面板 (500题纯图片可视化优化版)</h1>
    </div>

    <div class="stats-bar" id="statsBar">
        <!-- JS 动态计算填充 -->
    </div>

    <div class="nav" id="navBar">
__NAV_ITEMS_PLACEHOLDER__
    </div>

    <div class="export-bar">
        <button class="export-btn" onclick="exportJSONL()">导出 JSONL 标注修正包</button>
        <button class="export-btn" onclick="exportTXT()">导出 TXT 摘要报表</button>
    </div>

    <div class="image-lightbox" id="imageLightbox" onclick="closeImagePreview(event)">
        <button class="image-lightbox-close" type="button"
                aria-label="关闭图片预览"
                onclick="closeImagePreview(event, true)">×</button>
        <img id="imageLightboxTarget" alt="放大图片">
    </div>

__QUESTION_CARDS_PLACEHOLDER__

    <a href="#" class="back-to-top">↑</a>

    <script>
    // ===== 标注数据管理 (独立缓存 key: chemistry_difficulty_annotations_500) =====
    const LEVEL_NAMES = __LEVEL_NAMES_PLACEHOLDER__;
    const LEVEL_MAP = __LEVEL_MAP_PLACEHOLDER__;
    const allQuestions = __QUESTIONS_JSON_PLACEHOLDER__;

    function openImagePreview(image) {
        const lightbox = document.getElementById('imageLightbox');
        const target = document.getElementById('imageLightboxTarget');
        target.src = image.currentSrc || image.src;
        target.alt = image.alt || '放大图片';
        lightbox.classList.add('open');
        document.body.style.overflow = 'hidden';
    }

    function closeImagePreview(event, forceClose = false) {
        const lightbox = document.getElementById('imageLightbox');
        if (!forceClose && event && event.target !== lightbox) return;
        lightbox.classList.remove('open');
        document.getElementById('imageLightboxTarget').removeAttribute('src');
        document.body.style.overflow = '';
    }

    function markImageFailed(image) {
        const message = document.createElement('div');
        message.className = 'image-error';
        message.textContent = '图示加载失败，可能是图片地址失效或当前网络无法访问。';
        const frame = image.closest('.image-frame');
        if (frame) {
            image.replaceWith(message);
        }
    }

    function applyAdaptiveImageSizing(image) {
        const sourceKind = image.dataset.imageSource || 'original';
        const width = image.naturalWidth || 0;
        const height = image.naturalHeight || 0;
        const aspect = width > 0 ? height / width : 0;
        const isLongDocument = sourceKind === 'externalized'
            && (aspect >= 1.12 || (height >= 1600 && aspect >= 0.82));

        image.classList.toggle('image-layout-long-document', isLongDocument);
        image.classList.toggle('image-layout-standard', !isLongDocument);
        image.dataset.naturalWidth = String(width);
        image.dataset.naturalHeight = String(height);
        image.dataset.layout = isLongDocument ? 'long-document' : 'standard';
    }

    function initializeAdaptiveImageSizing() {
        document.querySelectorAll('.primary-image img').forEach(image => {
            if (image.complete && image.naturalWidth > 0) {
                applyAdaptiveImageSizing(image);
            } else {
                image.addEventListener(
                    'load',
                    () => applyAdaptiveImageSizing(image),
                    { once: true },
                );
            }
        });
    }

    document.addEventListener('keydown', event => {
        if (event.key === 'Escape') closeImagePreview(null, true);
    });

    function loadAnnotations() {
        try {
            return JSON.parse(localStorage.getItem('chemistry_difficulty_annotations_500') || '{}');
        } catch { return {}; }
    }

    function saveAnnotations(annotations) {
        localStorage.setItem('chemistry_difficulty_annotations_500', JSON.stringify(annotations));
        updateStats();
    }

    function setAnnotation(btn, action) {
        const qid = btn.getAttribute('data-qid');
        const annotations = loadAnnotations();
        const card = btn.closest('.question-card');
        const textarea = card.querySelector('.annotation-textarea');

        if (action === 'unmark' || action === 'correct') {
            delete annotations[qid];
            textarea.value = '';
            const levelSelect = card.querySelector('.corrected-level-select');
            if (levelSelect) levelSelect.value = '';
        } else {
            if (!annotations[qid]) annotations[qid] = {};
            annotations[qid].verdict = action;
        }

        const buttons = card.querySelectorAll('.annotation-btn');
        buttons.forEach(b => b.classList.remove('active'));
        if (action === 'wrong') {
            btn.classList.add('active');
        }

        saveAnnotations(annotations);
    }

    function saveCorrectedLevel(select) {
        const qid = select.getAttribute('data-qid');
        const annotations = loadAnnotations();
        const card = select.closest('.question-card');
        const wrongBtn = card.querySelector('.annotation-btn[data-action="wrong"]');
        if (select.value) {
            if (!annotations[qid]) annotations[qid] = {};
            annotations[qid].verdict = 'wrong';
            annotations[qid].corrected_level = select.value;
            if (wrongBtn) wrongBtn.classList.add('active');
        } else if (annotations[qid]) {
            delete annotations[qid].corrected_level;
        }
        saveAnnotations(annotations);
    }

    // 输入修改意见后自动标记为异常，避免遗漏验收状态。
    function saveAnnotationText(textarea) {
        const qid = textarea.getAttribute('data-qid');
        const annotations = loadAnnotations();
        const reason = textarea.value.trim();
        const card = textarea.closest('.question-card');
        const wrongBtn = card.querySelector('.annotation-btn[data-action="wrong"]');
        if (reason) {
            if (!annotations[qid]) annotations[qid] = {};
            annotations[qid].verdict = 'wrong';
            annotations[qid].reason = textarea.value;
            if (wrongBtn) wrongBtn.classList.add('active');
        } else if (annotations[qid]) {
            delete annotations[qid].reason;
            if (!annotations[qid].verdict || annotations[qid].verdict === 'correct') {
                delete annotations[qid];
            }
        }
        saveAnnotations(annotations);
    }

    // ===== 仪表盘汇总 =====
    function updateStats() {
        const annotations = loadAnnotations();
        const total = allQuestions.length;
        const wrong = allQuestions.filter(q => {
            const ann = annotations[q.question_id];
            return ann && ann.verdict === 'wrong';
        }).length;
        const correct = total - wrong;

        const levelStats = {};
        for (let lvl = 1; lvl <= 5; lvl++) {
            levelStats[lvl] = { total: 0, wrong: 0, correct: 0 };
        }
        allQuestions.forEach(q => {
            const lvl = q.level_num;
            if (lvl >= 1 && lvl <= 5) {
                levelStats[lvl].total++;
                const ann = annotations[q.question_id];
                if (ann && ann.verdict === 'wrong') levelStats[lvl].wrong++;
                else levelStats[lvl].correct++;
            }
        });

        let html = '<div class="stats-item">评估总数 <span class="stats-value">' + total + '</span> 题</div>';
        html += '<div class="stats-item">已标异常 <span class="stats-value">' + wrong + '</span> 题</div>';
        html += '<div class="stats-item">模型难度合理率 <span class="stats-value">' + (total > 0 ? (correct / total * 100).toFixed(1) + '%' : '—') + '</span></div>';
        html += '<span style="color:#ddd;">|</span>';

        for (let lvl = 1; lvl <= 5; lvl++) {
            const s = levelStats[lvl];
            const acc = s.total > 0 ? (s.correct / s.total * 100).toFixed(1) + '%' : '—';
            html += '<span class="stats-level stats-level-' + lvl + '">' +
                LEVEL_NAMES[lvl].replace('难度' + lvl + ' — ', '') +
                ': 抽样' + s.total + ' / 异常' + s.wrong + ' / 合理率' + acc + '</span>';
        }

        document.getElementById('statsBar').innerHTML = html;
    }

    function restoreAnnotations() {
        const annotations = loadAnnotations();
        document.querySelectorAll('.question-card').forEach(card => {
            const qid = card.getAttribute('data-qid');
            const ann = annotations[qid];
            if (ann) {
                if (ann.verdict === 'wrong') {
                    const btn = card.querySelector('.annotation-btn[data-action="' + ann.verdict + '"]');
                    if (btn) btn.classList.add('active');
                }
                if (ann.reason) {
                    const textarea = card.querySelector('.annotation-textarea');
                    if (textarea) textarea.value = ann.reason;
                }
                if (ann.corrected_level) {
                    const select = card.querySelector('.corrected-level-select');
                    if (select) select.value = ann.corrected_level;
                }
            }
        });
        updateStats();
    }

    function exportJSONL() {
        const annotations = loadAnnotations();
        let exportLines = [];
        allQuestions.forEach(q => {
            const ann = annotations[q.question_id];
            const manuallyRejected = Boolean(ann && ann.verdict === 'wrong');
            exportLines.push(JSON.stringify({
                question_id: q.question_id,
                model_difficulty_level: q.difficulty_level,
                verdict: manuallyRejected ? 'wrong' : 'correct',
                review_source: manuallyRejected ? 'manual_exception' : 'default_model_accepted',
                human_reviewed: manuallyRejected,
                human_difficulty_level: manuallyRejected
                    ? (ann.corrected_level || "")
                    : q.difficulty_level,
                human_notes: manuallyRejected ? (ann.reason || "") : ""
            }, null, 0));
        });

        const blob = new Blob([exportLines.join('\\n')], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chemistry_difficulty_human_annotations_500.jsonl';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function exportTXT() {
        const annotations = loadAnnotations();
        let text = "==================================================\\n";
        text += "        初中化学验收 500 题人工评议摘要报表\\n";
        text += "==================================================\\n\\n";
        
        let wrongCount = 0;
        let details = "";

        allQuestions.forEach(q => {
            const ann = annotations[q.question_id];
            if (ann && ann.verdict === 'wrong') {
                wrongCount++;
                details += `题目ID: ${q.question_id}\\n`;
                details += `模型定位: ${q.difficulty_level}\\n`;
                details += `评议结论: 【判定有误】\\n`;
                if (ann.corrected_level) details += `建议正确档位: ${ann.corrected_level}\\n`;
                if (ann.reason) details += `评审备注: ${ann.reason}\\n`;
                details += "--------------------------------------------------\\n";
            }
        });

        const correctCount = allQuestions.length - wrongCount;
        text += `默认验收总数: ${allQuestions.length} 道\\n`;
        text += `判定合理（未标异常）: ${correctCount} 道\\n`;
        text += `人工标记不准: ${wrongCount} 道\\n`;
        text += `合理率: ${allQuestions.length > 0 ? (correctCount / allQuestions.length * 100).toFixed(1) + '%' : 'N/A'}\\n\\n`;
        text += "================== 异常题详细列表 ==================\\n\\n";
        text += details;

        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'chemistry_difficulty_review_report_500.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // 导航栏平滑滑动
    document.querySelectorAll('.nav a').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            document.querySelector(targetId).scrollIntoView({
                behavior: 'smooth'
            });
            document.querySelectorAll('.nav a').forEach(a => a.classList.remove('nav-active'));
            this.classList.add('nav-active');
        });
    });

    window.onload = () => {
        restoreAnnotations();
        initializeAdaptiveImageSizing();
    };
    </script>
</body>
</html>
"""

def generate_html_file(samples: Dict[int, List[Dict[str, Any]]], output_path: str):
    # 1. 构造导航栏
    nav_html = ""
    for level in sorted(samples.keys()):
        count = len(samples[level])
        nav_html += f'        <a href="#level-{level}" data-level="{level}">{LEVEL_NAMES[level]} ({count})</a>\n'

    # 2. 构造题目 Cards 列表
    cards_html = ""
    all_questions_list = []
    
    for level in sorted(samples.keys()):
        items = samples[level]
        cards_html += f"""
    <div id="level-{level}" class="level-section">
        <div class="level-header level-{level}">
            <div class="level-title">{LEVEL_NAMES[level]}</div>
            <div class="level-desc">本档抽样验证共 {len(items)} 道题目（完整截图优先，原始题图可展开）</div>
        </div>
"""
        for idx, item in enumerate(items, 1):
            rating = item.get('difficulty_rating', {})
            features_obj = rating.get('features', {})
            reasoning = rating.get('reasoning', {})
            difficulty_level = rating.get('difficulty_level', '')
            level_num = LEVEL_MAP.get(difficulty_level, 0)

            question_id = item.get('question_id', 'unknown')
            parent_id = item.get('parent_id', question_id)
            api_time = item.get('api_time_use', 0)
            api_tokens = item.get('api_total_tokens', 0)
            stem_url = item.get('stem_pic_url', '')
            analysis_url = item.get('analysis_pic_url', '')

            # 存入 JS 变量的数据（省略繁杂文本）
            all_questions_list.append({
                'question_id': question_id,
                'parent_id': parent_id,
                'level_num': level_num,
                'difficulty_level': difficulty_level,
                'reasoning': reasoning,
                'features': features_obj,
                'stem_url': stem_url,
                'analysis_url': analysis_url,
                'api_time': api_time,
                'api_tokens': api_tokens
            })

            cards_html += f"""
        <div class="question-card" data-qid="{escape(question_id)}">
            <div class="question-header">
                <span class="question-id">#{idx} | ID: {question_id}</span>
                <div class="question-tags">
                    <span class="tag tag-time">消耗: {api_time}s</span>
                    <span class="tag tag-tokens">{api_tokens} tokens</span>
                </div>
            </div>
            <div class="question-body">
                <div class="difficulty-badge badge-{level_num}">{escape(difficulty_level)}</div>
"""
            # externalized 完整截图优先显示；原始题图折叠保留，避免重复。
            cards_html += render_image_section(
                stem_url,
                title="题干图示",
                kind="stem",
                content_requires_image=contains_image_reference(
                    item.get("stem"),
                    item.get("options"),
                    item.get("sub_questions"),
                ),
            )
            cards_html += render_image_section(
                analysis_url,
                title="解析图示",
                kind="analysis",
                content_requires_image=contains_image_reference(
                    item.get("analysis"),
                    item.get("sub_questions"),
                ),
            )

            # 理由与特征 (针对化学维度修改说明)
            if features_obj or reasoning:
                cards_html += """
                <div class="rating-section">
                    <div class="rating-title">化学特征维度 & 判定理由</div>
"""
                if reasoning:
                    if isinstance(reasoning, dict):
                        basis_txt = reasoning.get('knowledge_points', '')
                        hard_txt = reasoning.get('solution_process', '')
                        why_l = reasoning.get('main_difficulty_factors', '')
                        why_h = reasoning.get('level_basis', '')
                        cards_html += f"""
                        <div class="rating-reasoning">
                            <strong>1. 涉及知识点：</strong>{escape(basis_txt)}<br/>
                            <strong>2. 解题过程：</strong>{escape(hard_txt)}<br/>
                            <strong>3. 主要难度因素：</strong>{escape(why_l)}<br/>
                            <strong>4. 定档依据：</strong>{escape(why_h)}
                        </div>
                        """
                    else:
                        cards_html += f'                    <div class="rating-reasoning"><strong>判定依据与理由：</strong>{escape(str(reasoning))}</div>\n'

                if features_obj:
                    cards_html += '                    <div class="rating-details">\n'
                    feature_fields = [
                        ('knowledge', '知识覆盖'),
                        ('solution_process', '解题任务与步骤'),
                        ('information_processing', '信息处理'),
                        ('reaction_processes', '反应与过程'),
                        ('experiment_tasks', '实验任务'),
                        ('calculation', '计算与特殊方法'),
                        ('difficulty_conditions', '隐藏条件与干扰'),
                        ('expression_requirements', '表达要求'),
                        ('question_context', '题目情境'),
                        ('curriculum_scope', '课内范围'),
                    ]
                    for key, label in feature_fields:
                        value = features_obj.get(key, '')
                        if isinstance(value, dict):
                            value = '；'.join(f'{k}={v}' for k, v in value.items())
                        if isinstance(value, list):
                            value = '、'.join(str(v) for v in value)
                        if value:
                            display_value = str(value)[:150]
                            cards_html += f"""                        <div class="rating-detail-item">
                            <div class="label">{label}</div>
                            <div class="value">{escape(display_value)}</div>
                        </div>
"""
                    cards_html += '                    </div>\n'
                cards_html += '                </div>\n'

            # 验收意见栏
            cards_html += f"""
                <div class="annotation-section">
                    <div class="annotation-title">人工评议验收（默认模型判定合理）</div>
                    <div class="annotation-row">
                        <label>验收意见：</label>
                        <button class="annotation-btn btn-wrong" data-qid="{escape(question_id)}" data-action="wrong" onclick="setAnnotation(this, 'wrong')">✗ 模型判定不准</button>
                        <button class="annotation-btn btn-unmark" data-qid="{escape(question_id)}" data-action="unmark" onclick="setAnnotation(this, 'unmark')">✓ 恢复默认合理</button>
                    </div>
                    <div class="annotation-row">
                        <label>建议正确档位：</label>
                        <select class="corrected-level-select" data-qid="{escape(question_id)}" onchange="saveCorrectedLevel(this)">
                            <option value="">仅标错，暂不指定档位</option>
                            <option value="送分题">送分题</option>
                            <option value="基础题">基础题</option>
                            <option value="中等题">中等题</option>
                            <option value="拔高题">拔高题</option>
                            <option value="压轴题">压轴题</option>
                        </select>
                    </div>
                    <div class="annotation-row">
                        <label>修改意见与错误原因（输入内容后会自动标记为“判定不准”）：</label>
                    </div>
                    <textarea class="annotation-textarea" data-qid="{escape(question_id)}" placeholder="请说明缺陷具体原因及您的推荐档级..." oninput="saveAnnotationText(this)"></textarea>
                </div>
            </div>
        </div>
"""
        cards_html += "    </div>\n"

    # 3. 组合并执行占位替换
    questions_json = json.dumps(all_questions_list, ensure_ascii=False)
    
    html_content = HTML_TEMPLATE
    html_content = html_content.replace("__NAV_ITEMS_PLACEHOLDER__", nav_html)
    html_content = html_content.replace("__QUESTION_CARDS_PLACEHOLDER__", cards_html)
    html_content = html_content.replace("__LEVEL_NAMES_PLACEHOLDER__", json.dumps(LEVEL_NAMES, ensure_ascii=False))
    html_content = html_content.replace("__LEVEL_MAP_PLACEHOLDER__", json.dumps(LEVEL_MAP, ensure_ascii=False))
    html_content = html_content.replace("__QUESTIONS_JSON_PLACEHOLDER__", questions_json)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"成功生成化学交互可视化网页: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="化学难度评级结果交互式 HTML 验收工具"
    )
    parser.add_argument("-i", "--input", type=str, default="chemistry_difficulty_rated_results.jsonl",
                        help="输入的化学已打标结果 JSONL 路径")
    parser.add_argument("-v2", "--v2-source", type=str, default="data/chemistry_sampled_5000_per_difficulty_v2.jsonl",
                        help="含有化学全量图片的 V2 数据集路径")
    parser.add_argument("-oj", "--output-jsonl", type=str, default="chemistry_sampled_500_results.jsonl",
                        help="输出抽样后的 500 题 JSONL 数据集路径")
    parser.add_argument("-oh", "--output-html", type=str, default="chemistry_difficulty_rated_validation_500.html",
                        help="生成的化学可视化 HTML 验收网页保存路径")
    parser.add_argument(
        "--all-results",
        action="store_true",
        help="不再二次抽样，按模型最终等级渲染输入中的全部有效结果",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260730,
        help="按档抽样时使用的固定随机种子",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="按模型档位抽样的总题数；--all-results 时忽略",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 找不到打标输入文件 {args.input}！")
        return
    if not os.path.exists(args.v2_source):
        print(f"错误: 找不到 V2 图片资源文件 {args.v2_source}！")
        return

    # 1. 载入所有打标结果。
    print(f"正在读取化学打标数据: {args.input} ...")
    raw_data = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw_data.append(json.loads(line))
                except Exception:
                    continue

    print(f"成功载入 {len(raw_data)} 条打标记录。正在建立 V2 全量图片索引...")

    # 2. 建立 V2 图片库索引 (以 question_id 为 Key)
    v2_image_index = {}
    with open(args.v2_source, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    qid = str(item.get("question_id") or "").strip()
                    if qid:
                        v2_image_index[qid] = {
                            "stem_pic_url": item.get("stem_pic_url", ""),
                            "analysis_pic_url": item.get("analysis_pic_url", "")
                        }
                except Exception:
                    continue
    print(f"图片索引建立完成，共索引 {len(v2_image_index)} 道题的图片。")

    # 3. 将打标数据映射并对齐图片资源。
    aligned_data = []
    missing_pics = 0
    for item in raw_data:
        qid = str(item.get("question_id") or "").strip()
        if qid in v2_image_index:
            indexed = v2_image_index[qid]
            if indexed["stem_pic_url"]:
                item["stem_pic_url"] = indexed["stem_pic_url"]
            if indexed["analysis_pic_url"]:
                item["analysis_pic_url"] = indexed["analysis_pic_url"]
        else:
            missing_pics += 1
        aligned_data.append(item)
    
    if missing_pics > 0:
        print(
            f"提示: 有 {missing_pics} 道化学题目"
            "在 V2 数据集中没有对齐到图片 URL。"
        )

    # 4. 按大模型打标难度分组
    grouped_data = defaultdict(list)
    for item in aligned_data:
        rating = item.get('difficulty_rating', {})
        if not rating or not isinstance(rating, dict):
            continue
        level = rating.get('difficulty_level', '')
        if level in LEVEL_MAP:
            grouped_data[level].append(item)

    # 5. 精准抽样，或直接保留全部有效结果。
    sampled_data = []
    sampled_for_html = defaultdict(list)

    if args.all_results:
        print("\n================ 全量结果渲染 ================")
        for level in SAMPLE_PLAN:
            items = grouped_data[level]
            sampled_data.extend(items)
            sampled_for_html[LEVEL_MAP[level]] = items
            print(f"  {level}: {len(items)} 道")
    else:
        if len(aligned_data) < args.sample_size:
            raise ValueError(
                f"打标结果仅 {len(aligned_data)} 道，"
                f"无法抽取 {args.sample_size} 道"
            )
        rng = random.Random(args.seed)
        sample_plan = build_sample_plan(args.sample_size)
        remaining_by_level: Dict[str, List[Dict[str, Any]]] = {}
        print(
            "\n================ "
            f"抽样计划执行（seed={args.seed}） ================"
        )
        for level, target_count in sample_plan.items():
            pool = grouped_data[level]
            pool_size = len(pool)
            if pool_size >= target_count:
                sampled_items = rng.sample(pool, target_count)
                print(
                    f"  {level}: 池内共有 {pool_size} 道，"
                    f"精准抽样 {target_count} 道"
                )
            else:
                sampled_items = list(pool)
                print(
                    f"  {level}: 池内仅有 {pool_size} 道，"
                    f"全部保留（计划 {target_count} 道）"
                )

            sampled_data.extend(sampled_items)
            sampled_for_html[LEVEL_MAP[level]] = sampled_items
            sampled_ids = {id(item) for item in sampled_items}
            remaining_by_level[level] = [
                item for item in pool if id(item) not in sampled_ids
            ]

        missing = args.sample_size - len(sampled_data)
        if missing:
            refill_pool = [
                item
                for level in SAMPLE_PLAN
                for item in remaining_by_level[level]
            ]
            if len(refill_pool) < missing:
                raise ValueError(
                    f"合法五档结果不足，尚缺 {missing} 道，无法补足抽样"
                )
            refill = rng.sample(refill_pool, missing)
            sampled_data.extend(refill)
            for item in refill:
                level = item["difficulty_rating"]["difficulty_level"]
                sampled_for_html[LEVEL_MAP[level]].append(item)
            print(f"  从其他档剩余题目补足 {missing} 道")

    # 6. 导出抽样 JSONL
    print(f"\n正在导出抽样后的 JSONL 副本至: {args.output_jsonl} ...")
    Path(args.output_jsonl).expanduser().resolve().parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    Path(args.output_html).expanduser().resolve().parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"成功写入 {len(sampled_data)} 条抽样数据。")

    # 7. 渲染 HTML。
    print(f"正在渲染化学可视化验收网页至: {args.output_html} ...")
    generate_html_file(sampled_for_html, args.output_html)
    print("化学可视化网页生成完成。")

if __name__ == "__main__":
    main()
