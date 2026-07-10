# -*- coding: utf-8 -*-
"""
@File    : generate-physics-difficulty-html.py
@Description:
    从物理难度评级结果 JSONL 中按 5 档难度进行采样，生成便于人工观察与标注验证的 HTML 可视化网页。
    支持标注对错、标注原因、导出 jsonl/txt、实时统计准确率。
"""

import json
import os
import html
import random
import argparse
from collections import defaultdict
from typing import Dict, Any, List

# 难度分档定义
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

LEVEL_COLORS = {
    1: "#4CAF50",
    2: "#8BC34A",
    3: "#FFC107",
    4: "#FF9800",
    5: "#F44336",
}

def load_data(filepath: str) -> List[Dict[str, Any]]:
    data = []
    if not os.path.exists(filepath):
        print(f"错误: 找不到输入文件 {filepath}")
        return data
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except Exception:
                continue
    return data

def sample_by_level(data: List[Dict[str, Any]], sample_size: int) -> Dict[int, List[Dict[str, Any]]]:
    level_groups = defaultdict(list)
    for item in data:
        rating = item.get('difficulty_rating', {})
        if not rating or not isinstance(rating, dict):
            continue
        level_str = rating.get('difficulty_level', '')
        level_num = LEVEL_MAP.get(level_str, 0)
        if level_num:
            level_groups[level_num].append(item)

    sampled = {}
    for level in sorted(level_groups.keys()):
        items = level_groups[level]
        if len(items) > sample_size:
            sampled[level] = random.sample(items, sample_size)
        else:
            sampled[level] = items
        print(f"  {LEVEL_NAMES[level]}: 结果中共有 {len(items)} 道，本次采样 {len(sampled[level])} 道进行可视化评估")
    return sampled

def escape(text: str) -> str:
    if not text:
        return ""
    return html.escape(text)

# HTML 网页基础骨架模板
HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>初中物理题目难度评级验证样本</title>
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
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
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
            color: #1e3c72;
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
            background: #1e3c72;
            color: white;
            border-color: #1e3c72;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(30,60,114,0.3);
        }
        .nav a.nav-active {
            background: #1e3c72;
            color: white;
            border-color: #1e3c72;
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
            background: #1e3c72;
            color: white;
            border-color: #1e3c72;
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
            margin-bottom: 15px;
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
        
        .raw-text-container {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 6px;
            padding: 15px;
            font-size: 14.5px;
            margin-bottom: 15px;
            white-space: pre-wrap;
            word-break: break-all;
        }

        /* ===== Images ===== */
        .image-container {
            margin: 10px 0;
            text-align: center;
            background: #fafafa;
            padding: 10px;
            border-radius: 8px;
            border: 1px dashed #ddd;
        }
        .image-container img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        }

        /* ===== Rating Section ===== */
        .rating-section {
            margin-top: 15px;
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
            margin-top: 15px;
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
            background: #1e3c72;
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
            background: #2a5298;
            transform: translateY(-3px);
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>初中物理题目难度评级验证样本</h1>
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
    // ===== 标注数据管理 =====
    const LEVEL_NAMES = __LEVEL_NAMES_PLACEHOLDER__;
    const LEVEL_MAP = __LEVEL_MAP_PLACEHOLDER__;
    const allQuestions = __QUESTIONS_JSON_PLACEHOLDER__;

    function loadAnnotations() {
        try {
            return JSON.parse(localStorage.getItem('physics_difficulty_annotations') || '{}');
        } catch { return {}; }
    }

    function saveAnnotations(annotations) {
        localStorage.setItem('physics_difficulty_annotations', JSON.stringify(annotations));
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
        a.download = 'physics_difficulty_human_annotations.jsonl';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    function exportTXT() {
        const annotations = loadAnnotations();
        let text = "==================================================\\n";
        text += "        初中物理题目难度评级人工评议摘要报表\\n";
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
                details += `模型定位: ${q.difficulty_level} (原始原始教师定位: ${q.raw_difficulty}档)\\n`;
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
        a.download = 'physics_difficulty_review_report.txt';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

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
            <div class="level-desc">本档抽样验证共 {len(items)} 道题目</div>
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

            # 存入 JS 使用的数据
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
                'stem': item.get('stem', ''),
                'options': item.get('options', ''),
                'analysis': item.get('analysis', ''),
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
            # 题干文本
            if item.get('stem'):
                cards_html += f"""
                <div style="font-weight: bold; color: #333; margin-bottom: 6px;">题干文本:</div>
                <div class="raw-text-container">{escape(item.get('stem'))}</div>
"""
            # 题干图示
            if stem_url:
                cards_html += f"""
                <div style="margin-bottom: 15px;">
                    <div style="font-weight: bold; color: #555; margin-bottom: 6px;">题干图示：</div>
                    <div class="image-container">
                        <img src="{html.escape(stem_url)}" alt="题干图示" onerror="this.outerHTML='<div style=\\'color:#999;font-style:italic\\'>题干图示加载失败，可能为局域网内网地址</div>'">
                    </div>
                </div>
"""
            # 解析文本
            if item.get('analysis'):
                cards_html += f"""
                <div style="font-weight: bold; color: #1e3c72; margin-bottom: 6px;">解析文本:</div>
                <div class="raw-text-container" style="background:#f4f7fb; border-color:#d9e2ec;">{escape(item.get('analysis'))}</div>
"""
            # 解析图示
            if analysis_url:
                cards_html += f"""
                <div style="margin-bottom: 15px;">
                    <div style="font-weight: bold; color: #1e3c72; margin-bottom: 6px;">解析图示：</div>
                    <div class="image-container">
                        <img src="{html.escape(analysis_url)}" alt="解析图示" onerror="this.outerHTML='<div style=\\'color:#999;font-style:italic\\'>解析图示加载失败</div>'">
                    </div>
                </div>
"""

            # 理由与特征
            if features_obj or reasoning:
                cards_html += """
                <div class="rating-section">
                    <div class="rating-title">物理特征维度 & 判定理由</div>
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
                    feature_fields = [
                        ('step_count', '解析步骤数'),
                        ('formula_count', '公式数量'),
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
                        ('state_count', '物理状态数'),
                        ('constraint_count', '约束条件'),
                        ('variable_relation', '变量关系'),
                        ('experiment_requirement', '实验要求'),
                        ('graph_table_requirement', '图表处理要求'),
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
    print(f"✨ 成功渲染生成交互可视化网页: {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(description="初中物理题目难度评级数据可视化 HTML 网页生成工具")
    parser.add_argument("-i", "--input", type=str, default="physics_difficulty_rated_results.jsonl",
                        help="物理评级跑出的输出 JSONL 路径")
    parser.add_argument("-o", "--output", type=str, default="physics_difficulty_rated_validation.html",
                        help="生成的可视化网页保存路径")
    parser.add_argument("-s", "--sample-size", type=int, default=400,
                        help="每档难度采样的最大题数，默认 400")

    args = parser.parse_args()

    print(f"正在加载打标数据: {args.input} ...")
    data = load_data(args.input)
    if not data:
        print("错误: 载入的数据量为0，请检查评级脚本是否正常跑出数据。")
        return

    print(f"数据载入成功，共 {len(data)} 条。正在按 5 档进行分档抽样...")
    samples = sample_by_level(data, args.sample_size)

    print("正在渲染并生成验收 HTML 页面...")
    generate_html_file(samples, args.output)
    print("👉 请直接双击或使用浏览器打开生成的 HTML 网页进行验收。")

if __name__ == "__main__":
    main()
