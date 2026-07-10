# -*- coding: utf-8 -*-
"""
@File    : sample_and_generate_chemistry_html.py
@Description:
    从已打标的 3000 道初中化学题中，精准抽样 500 道生成交互式评议验收网页。
    - 评级判定：依据 V1 纯文本打标与后处理纠偏结果。
    - 可视化优化：题干和解析完全采用 V2 对应的图片 URL 进行渲染展示，不再渲染任何纯文本，规避 LaTeX 乱码。
"""

import json
import os
import html
import random
import argparse
from collections import defaultdict
from typing import Dict, Any, List

# 抽样配比计划 (对齐物理 500 题配比)
SAMPLE_PLAN = {
    "送分题": 100,
    "基础题": 120,
    "中等题": 120,
    "拔高题": 100,
    "压轴题": 60
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

def escape(text: str) -> str:
    if not text:
        return ""
    return html.escape(text)

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
            max-width: 1200px;
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
        .tag-raw { background: #eceff1; color: #37474f; }
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

        /* ===== Images Container ===== */
        .image-container {
            margin: 10px 0 20px 0;
            text-align: left;
            background: #fff;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .image-container img {
            max-width: 100%;
            max-height: 500px;
            object-fit: contain;
            border-radius: 4px;
            border: 1px solid #f1f5f9;
        }

        /* ===== Rating Section ===== */
        .rating-section {
            margin-top: 20px;
            padding: 15px;
            background: #f4faf4;
            border-radius: 8px;
            border: 1px solid #d4eed5;
        }
        .rating-title {
            font-weight: bold;
            color: #2E7D32;
            margin-bottom: 10px;
            font-size: 14px;
        }
        .rating-reasoning {
            font-size: 13px;
            color: #555;
            margin-bottom: 12px;
            line-height: 1.5;
            background: white;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #e2f0d9;
        }
        .rating-details {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
            font-size: 12px;
        }
        .rating-detail-item {
            padding: 6px 10px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e2f0d9;
        }
        .rating-detail-item .label {
            color: #7f8c8d;
            font-size: 11px;
        }
        .rating-detail-item .value {
            color: #2c3e50;
            margin-top: 2px;
            font-weight: 500;
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

__QUESTION_CARDS_PLACEHOLDER__

    <a href="#" class="back-to-top">↑</a>

    <script>
    // ===== 标注数据管理 (独立缓存 key: chemistry_difficulty_annotations_500) =====
    const LEVEL_NAMES = __LEVEL_NAMES_PLACEHOLDER__;
    const LEVEL_MAP = __LEVEL_MAP_PLACEHOLDER__;
    const allQuestions = __QUESTIONS_JSON_PLACEHOLDER__;

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

        if (action === 'unmark') {
            delete annotations[qid];
            textarea.value = '';
        } else {
            if (!annotations[qid]) annotations[qid] = {};
            annotations[qid].verdict = action;
        }

        const buttons = card.querySelectorAll('.annotation-btn');
        buttons.forEach(b => b.classList.remove('active'));
        if (action !== 'unmark') {
            btn.classList.add('active');
        }

        saveAnnotations(annotations);
    }

    // 保存键盘输入的文字修改意见
    function saveAnnotationText(textarea) {
        const qid = textarea.getAttribute('data-qid');
        const annotations = loadAnnotations();
        if (!annotations[qid]) annotations[qid] = {};
        annotations[qid].reason = textarea.value;
        saveAnnotations(annotations);
    }

    // ===== 仪表盘汇总 =====
    function updateStats() {
        const annotations = loadAnnotations();
        const total = allQuestions.length;
        const annotated = Object.keys(annotations).filter(k => annotations[k].verdict).length;
        const correct = Object.keys(annotations).filter(k => annotations[k].verdict === 'correct').length;

        const levelStats = {};
        for (let lvl = 1; lvl <= 5; lvl++) {
            levelStats[lvl] = { total: 0, annotated: 0, correct: 0 };
        }
        allQuestions.forEach(q => {
            const lvl = q.level_num;
            if (lvl >= 1 && lvl <= 5) {
                levelStats[lvl].total++;
                const ann = annotations[q.question_id];
                if (ann && ann.verdict) {
                    levelStats[lvl].annotated++;
                    if (ann.verdict === 'correct') levelStats[lvl].correct++;
                }
            }
        });

        let html = '<div class="stats-item">评估总数 <span class="stats-value">' + total + '</span> 题</div>';
        html += '<div class="stats-item">已评审数 <span class="stats-value">' + annotated + '</span> 题</div>';
        html += '<div class="stats-item">模型难度合理率 <span class="stats-value">' + (annotated > 0 ? (correct / annotated * 100).toFixed(1) + '%' : '—') + '</span></div>';
        html += '<span style="color:#ddd;">|</span>';

        for (let lvl = 1; lvl <= 5; lvl++) {
            const s = levelStats[lvl];
            const acc = s.annotated > 0 ? (s.correct / s.annotated * 100).toFixed(1) + '%' : '—';
            html += '<span class="stats-level stats-level-' + lvl + '">' +
                LEVEL_NAMES[lvl].replace('难度' + lvl + ' — ', '') +
                ': 抽样' + s.total + ' / 评审' + s.annotated + ' / 合理率' + acc + '</span>';
        }

        document.getElementById('statsBar').innerHTML = html;
    }

    function restoreAnnotations() {
        const annotations = loadAnnotations();
        document.querySelectorAll('.question-card').forEach(card => {
            const qid = card.getAttribute('data-qid');
            const ann = annotations[qid];
            if (ann) {
                if (ann.verdict) {
                    const btn = card.querySelector('.annotation-btn[data-action="' + ann.verdict + '"]');
                    if (btn) btn.classList.add('active');
                }
                if (ann.reason) {
                    const textarea = card.querySelector('.annotation-textarea');
                    if (textarea) textarea.value = ann.reason;
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
            if (ann && ann.verdict) {
                exportLines.push(JSON.stringify({
                    question_id: q.question_id,
                    model_difficulty_level: q.difficulty_level,
                    verdict: ann.verdict,
                    human_notes: ann.reason || ""
                }, null, 0));
            }
        });

        if (exportLines.length === 0) {
            alert("目前没有任何标注修改意见，请先点击 ✓ 或 ✗ 选项！");
            return;
        }

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
        
        let correctCount = 0;
        let wrongCount = 0;
        let details = "";

        allQuestions.forEach(q => {
            const ann = annotations[q.question_id];
            if (ann && ann.verdict) {
                const statusStr = ann.verdict === 'correct' ? "【判定合理】" : "【判定有误】";
                if (ann.verdict === 'correct') correctCount++; else wrongCount++;
                
                details += `题目ID: ${q.question_id}\\n`;
                details += `模型定位: ${q.difficulty_level} (原始教师定位: ${q.raw_difficulty}档)\\n`;
                details += `评议结论: ${statusStr}\\n`;
                if (ann.reason) details += `评审备注: ${ann.reason}\\n`;
                details += "--------------------------------------------------\\n";
            }
        });

        text += `评审总数: ${correctCount + wrongCount} 道\\n`;
        text += `判定合理: ${correctCount} 道\\n`;
        text += `判定不准: ${wrongCount} 道\\n`;
        text += `合理率: ${correctCount + wrongCount > 0 ? (correctCount / (correctCount + wrongCount) * 100).toFixed(1) + '%' : 'N/A'}\\n\\n`;
        text += "================== 详细评议列表 ==================\\n\\n";
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

    window.onload = restoreAnnotations;
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
            <div class="level-desc">本档抽样验证共 {len(items)} 道题目 (已全面采用图片 URL 渲染)</div>
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
            raw_diff = item.get('difficulty', '无')

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
                'api_tokens': api_tokens,
                'raw_difficulty': raw_diff
            })

            cards_html += f"""
        <div class="question-card" data-qid="{escape(question_id)}">
            <div class="question-header">
                <span class="question-id">#{idx} | ID: {question_id}</span>
                <div class="question-tags">
                    <span class="tag tag-raw">原始教师难度: {raw_diff}档</span>
                    <span class="tag tag-time">消耗: {api_time}s</span>
                    <span class="tag tag-tokens">{api_tokens} tokens</span>
                </div>
            </div>
            <div class="question-body">
                <div class="difficulty-badge badge-{level_num}">{escape(difficulty_level)}</div>
"""
            # 渲染题干图片 (多图支持)
            if stem_url:
                cards_html += """
                <div style="margin-bottom: 20px;">
                    <div style="font-weight: bold; color: #555; margin-bottom: 6px;">题干图示：</div>
                    <div class="image-container">
"""
                for u in stem_url.split(','):
                    if u.strip():
                        cards_html += f'                        <img src="{html.escape(u.strip())}" alt="题干图示" onerror="this.outerHTML=\'<div style=\\\'color:#e53e3e;font-style:italic\\\'>(图示加载失败，可能为局域网内网地址)</div>\'">\n'
                cards_html += """                    </div>
                </div>
"""
            else:
                cards_html += """
                <div style="margin-bottom: 20px;">
                    <div style="font-weight: bold; color: #999; font-style: italic; margin-bottom: 6px;">【该题无题干图示】</div>
                </div>
"""

            # 渲染解析图片 (多图支持)
            if analysis_url:
                cards_html += """
                <div style="margin-bottom: 20px;">
                    <div style="font-weight: bold; color: #11998e; margin-bottom: 6px;">解析图示：</div>
                    <div class="image-container">
"""
                for u in analysis_url.split(','):
                    if u.strip():
                        cards_html += f'                        <img src="{html.escape(u.strip())}" alt="解析图示" onerror="this.outerHTML=\'<div style=\\\'color:#e53e3e;font-style:italic\\\'>(图示加载失败)</div>\'">\n'
                cards_html += """                    </div>
                </div>
"""
            else:
                cards_html += """
                <div style="margin-bottom: 20px;">
                    <div style="font-weight: bold; color: #999; font-style: italic; margin-bottom: 6px;">【该题无解析图示】</div>
                </div>
"""

            # 理由与特征 (针对化学维度修改说明)
            if features_obj or reasoning:
                cards_html += """
                <div class="rating-section">
                    <div class="rating-title">化学特征维度 & 判定理由</div>
"""
                if reasoning:
                    if isinstance(reasoning, dict):
                        basis_txt = reasoning.get('core_basis', '')
                        hard_txt = reasoning.get('hard_point', '')
                        why_l = reasoning.get('why_not_lower', '')
                        why_h = reasoning.get('why_not_higher', '')
                        cards_html += f"""
                        <div class="rating-reasoning">
                            <strong>1. 核心判定依据：</strong>{escape(basis_txt)}<br/>
                            <strong>2. 易错卡点：</strong>{escape(hard_txt)}<br/>
                            <strong>3. 为什么不低判定一档：</strong>{escape(why_l)}<br/>
                            <strong>4. 为什么不高判定一档：</strong>{escape(why_h)}
                        </div>
                        """
                    else:
                        cards_html += f'                    <div class="rating-reasoning"><strong>判定依据与理由：</strong>{escape(str(reasoning))}</div>\n'

                if features_obj:
                    cards_html += '                    <div class="rating-details">\n'
                    # 对齐化学打标中的 18 维归一化特征
                    feature_fields = [
                        ('step_count', '解析步骤数'),
                        ('equation_count', '化学方程式数量'),
                        ('calculation_complexity', '计算复杂度'),
                        ('reasoning_chain', '推理链条'),
                        ('problem_structure', '题型结构'),
                        ('additional_structure', '附加结构'),
                        ('information_carrier', '信息载体'),
                        ('reality_question', '现实生活情境'),
                        ('subquestion_dependency', '子问依赖性'),
                        ('knowledge_count', '知识点个数'),
                        ('knowledge_diff', '知识点难度'),
                        ('cross_module', '跨模块综合'),
                        ('chemistry_process_count', '化学反应/转化过程数'),
                        ('constraint_count', '反应约束条件'),
                        ('evidence_relation', '证据推理关系'),
                        ('experiment_requirement', '实验探究要求'),
                        ('graph_table_requirement', '图像分析要求'),
                        ('error_risk', '易错风险'),
                    ]
                    for key, label in feature_fields:
                        value = features_obj.get(key, '')
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
                    <div class="annotation-title">人工评议验收</div>
                    <div class="annotation-row">
                        <label>验收意见：</label>
                        <button class="annotation-btn btn-correct" data-qid="{escape(question_id)}" data-action="correct" onclick="setAnnotation(this, 'correct')">✓ 模型判定合理</button>
                        <button class="annotation-btn btn-wrong" data-qid="{escape(question_id)}" data-action="wrong" onclick="setAnnotation(this, 'wrong')">✗ 模型判定不准</button>
                        <button class="annotation-btn btn-unmark" data-qid="{escape(question_id)}" data-action="unmark" onclick="setAnnotation(this, 'unmark')">— 清除状态</button>
                    </div>
                    <div class="annotation-row">
                        <label>修改意见与错误原因归类 (如有错误，请指出正确档位与缺陷，如模型把送分判为基础)：</label>
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
    print(f"✨ 成功渲染生成纯图片交互可视化网页: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="基于 V1 评级、V2 纯图片可视化的 500 道初中化学题抽样生成网页工具")
    parser.add_argument("-i", "--input", type=str, default="chemistry_difficulty_rated_results.jsonl",
                        help="输入的化学已打标 V1 结果 JSONL 路径")
    parser.add_argument("-v2", "--v2-source", type=str, default="../data/chemistry_sampled_5000_per_difficulty_v2.jsonl",
                        help="含有化学全量图片的 V2 数据集路径")
    parser.add_argument("-oj", "--output-jsonl", type=str, default="chemistry_sampled_500_results.jsonl",
                        help="输出抽样后的 500 题 JSONL 数据集路径")
    parser.add_argument("-oh", "--output-html", type=str, default="chemistry_difficulty_rated_validation_500.html",
                        help="生成的化学可视化 HTML 验收网页保存路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子数")

    args = parser.parse_args()

    # 设置随机数种子，实现结果可复现
    random.seed(args.seed)

    if not os.path.exists(args.input):
        print(f"错误: 找不到打标输入文件 {args.input}！")
        return
    if not os.path.exists(args.v2_source):
        print(f"错误: 找不到 V2 图片资源文件 {args.v2_source}！")
        return

    # 1. 载入所有打标数据 (V1 结果)
    print(f"正在读取 V1 打标数据: {args.input} ...")
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
                    qid = item.get("question_id")
                    if qid:
                        v2_image_index[qid] = {
                            "stem_pic_url": item.get("stem_pic_url", ""),
                            "analysis_pic_url": item.get("analysis_pic_url", "")
                        }
                except Exception:
                    continue
    print(f"图片索引建立完成，共索引 {len(v2_image_index)} 道题的图片。")

    # 3. 将 V1 数据映射并对齐图片资源
    aligned_data = []
    missing_pics = 0
    for item in raw_data:
        qid = item.get("question_id")
        if qid in v2_image_index:
            item["stem_pic_url"] = v2_image_index[qid]["stem_pic_url"]
            item["analysis_pic_url"] = v2_image_index[qid]["analysis_pic_url"]
        else:
            missing_pics += 1
        aligned_data.append(item)
    
    if missing_pics > 0:
        print(f"⚠️ 提示: 有 {missing_pics} 道化学题目在 V2 数据集中没有对齐到图片 URL。")

    # 4. 按大模型打标难度分组
    grouped_data = defaultdict(list)
    for item in aligned_data:
        rating = item.get('difficulty_rating', {})
        if not rating or not isinstance(rating, dict):
            continue
        level = rating.get('difficulty_level', '')
        if level in LEVEL_MAP:
            grouped_data[level].append(item)

    # 5. 精准抽样 (500 题)
    sampled_data = []
    sampled_for_html = defaultdict(list)

    print("\n================ 抽样计划执行 ================")
    for level, target_count in SAMPLE_PLAN.items():
        pool = grouped_data[level]
        pool_size = len(pool)
        
        if pool_size >= target_count:
            sampled_items = random.sample(pool, target_count)
            print(f"  🎯 {level}: 池内共有 {pool_size} 道，精准抽样 {target_count} 道")
        else:
            sampled_items = pool
            print(f"  ⚠️ {level}: 不足！池内仅有 {pool_size} 道，全部保留 (计划抽 {target_count} 道)")
            
        sampled_data.extend(sampled_items)
        level_num = LEVEL_MAP[level]
        sampled_for_html[level_num] = sampled_items

    # 6. 导出抽样 JSONL
    print(f"\n正在导出抽样后的 JSONL 副本至: {args.output_jsonl} ...")
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for item in sampled_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"👉 成功写入 {len(sampled_data)} 条抽样数据。")

    # 7. 渲染纯图片 HTML
    print(f"正在渲染纯图片可视化验收网页至: {args.output_html} ...")
    generate_html_file(sampled_for_html, args.output_html)
    print("✨ 纯图片可视化优化网页生成已顺利完成！")

if __name__ == "__main__":
    main()
