# -*- coding: utf-8 -*-
"""
@File    : generate-difficulty-html.py
@Description:
    从难度评级结果中按难度级别分组展示，生成HTML文件便于人工观察与标注验证。
    支持标注对错、标注原因、导出jsonl/txt、实时统计准确率。
    输入：difficulty-rating-5-level-prompt-*.py的输出JSONL文件
"""

import json
import os
import html
import random
from collections import defaultdict
from typing import Dict, Any, List

# -------------------------- 1. 配置 --------------------------
INPUT_PATH = "/home/share_ssd_data/nfs-data1/wangmeng148/coding/vllm-main/scripts/tiku_difficulty_cls/result/difficulty_rated_06111810_shuffle_new_features_with_cache.jsonl"
# INPUT_PATH = "/home/share_ssd_data/nfs-data1/wangmeng148/coding/vllm-main/scripts/tiku_difficulty_cls/result/difficulty_rated_0528_fix_level1_vllm.jsonl"
IMAGE_URLS_PATH = "/home/share_ssd_data/nfs-data1/wangmeng148/coding/vllm-main/scripts/tiku_difficulty_cls/result/html_samples/pids_image_urls_merged.jsonl"
OUTPUT_DIR = "/home/share_ssd_data/nfs-data1/wangmeng148/coding/vllm-main/scripts/tiku_difficulty_cls/result/difficulty_result/html_samples_06121520"
# OUTPUT_DIR = "/home/share_ssd_data/nfs-data1/wangmeng148/coding/vllm-main/scripts/tiku_difficulty_cls/result/difficulty_result/html_samples_0528_fix_level1_vllm"
SAMPLE_SIZE = 400

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except:
                continue
    return data


def load_image_urls(filepath: str) -> Dict[str, Dict[str, str]]:
    """加载parent_id到图片URL的映射"""
    image_urls = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                parent_id = item.get('parent_id')
                urls = item.get('urls', {})
                if parent_id:
                    image_urls[parent_id] = urls
            except:
                continue
    return image_urls


def sample_by_level(data: List[Dict[str, Any]], sample_size: int) -> Dict[int, List[Dict[str, Any]]]:
    level_groups = defaultdict(list)

    for item in data:
        rating = item.get('difficulty_rating', {})
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
        print(f"  {LEVEL_NAMES[level]}: 共{len(items)}道，采样{len(sampled[level])}道")

    return sampled


def escape(text: str) -> str:
    if not text:
        return ""
    return html.escape(text)


def generate_html(samples: Dict[int, List[Dict[str, Any]]], image_urls: Dict[str, Dict[str, str]], output_path: str):
    # Collect all question data for JS
    all_questions = []
    for level in sorted(samples.keys()):
        for item in samples[level]:
            rating = item.get('difficulty_rating', {})
            features_obj = rating.get('features', {})
            reasoning = rating.get('reason', '')
            difficulty_level = rating.get('difficulty_level', '')
            level_num = LEVEL_MAP.get(difficulty_level, 0)
            question_id = item.get('question_id', 'unknown')
            parent_id = item.get('parent_id', question_id)
            urls = image_urls.get(parent_id, {})
            stem_url = urls.get('stem', '')
            analysis_url = urls.get('analysis', '')

            all_questions.append({
                'question_id': question_id,
                'parent_id': parent_id,
                'level_num': level_num,
                'difficulty_level': difficulty_level,
                'reasoning': reasoning,
                'features': features_obj,
                'stem_url': stem_url,
                'analysis_url': analysis_url,
                'api_time': item.get('api_time_use', 0),
                'api_tokens': item.get('api_total_tokens', 0),
                'stem': item.get('stem', ''),
                'options': item.get('options', ''),
                'analysis': item.get('analysis', ''),
            })

    questions_json = json.dumps(all_questions, ensure_ascii=False)

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>初中数学题目难度评级验证样本</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
            padding-top: 160px;
        }}

        /* ===== Header ===== */
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 18px 20px;
            text-align: center;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            z-index: 1000;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header h1 {{ font-size: 20px; margin-bottom: 0; }}

        /* ===== Stats Bar ===== */
        .stats-bar {{
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
        }}
        .stats-item {{
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
            color: #555;
        }}
        .stats-item .stats-value {{
            font-weight: 700;
            color: #333;
        }}
        .stats-level {{
            display: inline-flex;
            align-items: center;
            gap: 4px;
            font-size: 12px;
            padding: 3px 10px;
            border-radius: 12px;
        }}
        .stats-level-1 {{ background: #E8F5E9; color: #2E7D32; }}
        .stats-level-2 {{ background: #F1F8E9; color: #558B2F; }}
        .stats-level-3 {{ background: #FFF8E1; color: #F9A825; }}
        .stats-level-4 {{ background: #FFF3E0; color: #EF6C00; }}
        .stats-level-5 {{ background: #FFEBEE; color: #C62828; }}

        /* ===== Nav ===== */
        .nav {{
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
        }}
        .nav a {{
            padding: 8px 18px;
            border-radius: 8px;
            text-decoration: none;
            color: #666;
            background: #f5f5f5;
            transition: all 0.25s;
            font-size: 14px;
            font-weight: 500;
            border: 1px solid transparent;
        }}
        .nav a:hover {{
            background: #667eea;
            color: white;
            border-color: #667eea;
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(102,126,234,0.3);
        }}
        .nav a.nav-active {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        /* ===== Export Bar ===== */
        .export-bar {{
            position: fixed;
            top: 56px;
            right: 16px;
            z-index: 1001;
            display: flex;
            gap: 8px;
        }}
        .export-btn {{
            padding: 6px 14px;
            border-radius: 8px;
            border: 1px solid #ddd;
            background: white;
            color: #555;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }}
        .export-btn:hover {{
            background: #667eea;
            color: white;
            border-color: #667eea;
        }}

        /* ===== Level Section ===== */
        .level-section {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        .level-header {{
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            border-left: 5px solid;
        }}
        .level-1 {{ border-left-color: #4CAF50; }}
        .level-2 {{ border-left-color: #8BC34A; }}
        .level-3 {{ border-left-color: #FFC107; }}
        .level-4 {{ border-left-color: #FF9800; }}
        .level-5 {{ border-left-color: #F44336; }}
        .level-title {{ font-size: 22px; font-weight: bold; margin-bottom: 4px; }}
        .level-desc {{ color: #666; font-size: 14px; }}

        /* ===== Question Card ===== */
        .question-card {{
            background: white;
            margin-bottom: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
            overflow: hidden;
        }}
        .question-header {{
            padding: 12px 20px;
            background: #fafafa;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 8px;
        }}
        .question-id {{
            font-family: monospace;
            font-size: 12px;
            color: #999;
        }}
        .question-tags {{
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            align-items: center;
        }}
        .tag {{
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }}
        .tag-level-1 {{ background: #E8F5E9; color: #2E7D32; }}
        .tag-level-2 {{ background: #F1F8E9; color: #558B2F; }}
        .tag-level-3 {{ background: #FFF8E1; color: #F9A825; }}
        .tag-level-4 {{ background: #FFF3E0; color: #EF6C00; }}
        .tag-level-5 {{ background: #FFEBEE; color: #C62828; }}
        .tag-time {{ background: #E3F2FD; color: #1565C0; }}
        .tag-tokens {{ background: #F3E5F5; color: #6A1B9A; }}

        /* ===== Big Difficulty Badge ===== */
        .difficulty-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 6px 20px;
            border-radius: 8px;
            font-size: 22px;
            font-weight: 800;
            letter-spacing: 2px;
            margin: 12px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.12);
        }}
        .badge-1 {{ background: #4CAF50; color: white; }}
        .badge-2 {{ background: #8BC34A; color: white; }}
        .badge-3 {{ background: #FFC107; color: #333; }}
        .badge-4 {{ background: #FF9800; color: white; }}
        .badge-5 {{ background: #F44336; color: white; }}

        .question-body {{ padding: 20px; }}

        /* ===== Images ===== */
        .image-container {{
            margin: 10px 0;
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        }}

        /* ===== Rating Section ===== */
        .rating-section {{
            margin-top: 15px;
            padding: 15px;
            background: #f0faf0;
            border-radius: 8px;
            border: 1px solid #c8e6c9;
        }}
        .rating-title {{
            font-weight: bold;
            color: #2E7D32;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .rating-reasoning {{
            font-size: 13px;
            color: #555;
            margin-bottom: 10px;
            line-height: 1.5;
        }}
        .rating-details {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 8px;
            font-size: 12px;
        }}
        .rating-detail-item {{
            padding: 6px 10px;
            background: white;
            border-radius: 6px;
            border: 1px solid #e8f5e9;
        }}
        .rating-detail-item .label {{
            color: #999;
            font-size: 11px;
        }}
        .rating-detail-item .value {{
            color: #333;
            margin-top: 2px;
        }}

        /* ===== Annotation Section ===== */
        .annotation-section {{
            margin-top: 15px;
            padding: 15px;
            background: #fff8e1;
            border-radius: 8px;
            border: 1px solid #ffe082;
        }}
        .annotation-title {{
            font-weight: bold;
            color: #F9A825;
            margin-bottom: 10px;
            font-size: 14px;
        }}
        .annotation-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }}
        .annotation-row label {{
            font-size: 13px;
            color: #555;
            font-weight: 500;
        }}
        .annotation-btn {{
            padding: 6px 18px;
            border-radius: 8px;
            border: 2px solid #ccc;
            background: white;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .annotation-btn:hover {{ transform: translateY(-1px); }}
        .annotation-btn.btn-correct {{
            border-color: #4CAF50;
            color: #4CAF50;
        }}
        .annotation-btn.btn-correct.active {{
            background: #4CAF50;
            color: white;
            box-shadow: 0 2px 8px rgba(76,175,80,0.3);
        }}
        .annotation-btn.btn-wrong {{
            border-color: #F44336;
            color: #F44336;
        }}
        .annotation-btn.btn-wrong.active {{
            background: #F44336;
            color: white;
            box-shadow: 0 2px 8px rgba(244,67,54,0.3);
        }}
        .annotation-btn.btn-unmark {{
            border-color: #999;
            color: #999;
        }}
        .annotation-btn.btn-unmark.active {{
            background: #999;
            color: white;
        }}
        .annotation-textarea {{
            width: 100%;
            min-height: 60px;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 13px;
            resize: vertical;
            font-family: inherit;
            line-height: 1.5;
        }}
        .annotation-textarea:focus {{
            outline: none;
            border-color: #FF9800;
            box-shadow: 0 0 0 3px rgba(255,152,0,0.1);
        }}

        /* ===== Back to Top ===== */
        .back-to-top {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            background: #667eea;
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
        }}
        .back-to-top:hover {{
            background: #5a6fd6;
            transform: translateY(-3px);
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>初中数学题目难度评级验证样本</h1>
    </div>

    <div class="stats-bar" id="statsBar">
        <!-- filled by JS -->
    </div>

    <div class="nav" id="navBar">
"""

    for level in sorted(samples.keys()):
        count = len(samples[level])
        html_content += f'        <a href="#level-{level}" data-level="{level}">{LEVEL_NAMES[level]} ({count})</a>\n'

    html_content += f'''    </div>

    <div class="export-bar">
        <button class="export-btn" onclick="exportJSONL()">导出 JSONL</button>
        <button class="export-btn" onclick="exportTXT()">导出 TXT</button>
    </div>

'''

    for level in sorted(samples.keys()):
        items = samples[level]
        html_content += f'''
    <div id="level-{level}" class="level-section">
        <div class="level-header level-{level}">
            <div class="level-title">{LEVEL_NAMES[level]}</div>
            <div class="level-desc">共 {len(items)} 道题目</div>
        </div>
'''

        for idx, item in enumerate(items, 1):
            rating = item.get('difficulty_rating', {})
            features_obj = rating.get('features', {})
            reasoning = rating.get('reason', '')
            difficulty_level = rating.get('difficulty_level', '')
            level_num = LEVEL_MAP.get(difficulty_level, 0)

            question_id = item.get('question_id', 'unknown')
            parent_id = item.get('parent_id', question_id)
            api_time = item.get('api_time_use', 0)
            api_tokens = item.get('api_total_tokens', 0)

            # 获取图片URL
            urls = image_urls.get(parent_id, {})
            stem_url = urls.get('stem', '')
            analysis_url = urls.get('analysis', '')

            html_content += f'''
        <div class="question-card" data-qid="{escape(question_id)}">
            <div class="question-header">
                <span class="question-id">#{idx} | ID: {question_id}</span>
                <div class="question-tags">
                    <span class="tag tag-time">{api_time}s</span>
                    <span class="tag tag-tokens">{api_tokens} tokens</span>
                </div>
            </div>
            <div class="question-body">
                <div class="difficulty-badge badge-{level_num}">{escape(difficulty_level)}</div>
'''

            # 题干图片
            if stem_url:
                html_content += f'''
                <div style="margin-bottom: 15px;">
                    <div style="font-weight: bold; color: #333; margin-bottom: 8px;">题干</div>
                    <div class="image-container">
                        <img src="{html.escape(stem_url)}" alt="题干图片" onerror="this.outerHTML='<div style=\\'color:#999;font-style:italic\\'>题干图片加载失败</div>'">
                    </div>
                </div>
'''
            else:
                html_content += '''
                <div style="color:#999;font-style:italic;margin-bottom:15px;">暂无题干图片</div>
'''

            # 解析图片
            if analysis_url:
                html_content += f'''
                <div style="margin-bottom: 15px;">
                    <div style="font-weight: bold; color: #667eea; margin-bottom: 8px;">解析</div>
                    <div class="image-container">
                        <img src="{html.escape(analysis_url)}" alt="解析图片" onerror="this.outerHTML='<div style=\\'color:#999;font-style:italic\\'>解析图片加载失败</div>'">
                    </div>
                </div>
'''
            else:
                html_content += '''
                <div style="color:#999;font-style:italic;margin-bottom:15px;">暂无解析图片</div>
'''

            # Rating details
            if features_obj or reasoning:
                html_content += '''
                <div class="rating-section">
                    <div class="rating-title">模型评级分析</div>
'''
                if reasoning:
                    html_content += f'                    <div class="rating-reasoning"><strong>判定理由：</strong>{escape(reasoning)}</div>\n'

                if features_obj:
                    html_content += '                    <div class="rating-details">\n'
                    feature_fields = [
                        ('step_count', '解析步骤数'),
                        ('auxiliary_line', '辅助线'),
                        ('reality_question', '现实生活问题'),
                        ('classification_discussion', '分类讨论'),
                        ('problem_structure', '题型结构'),
                        ('dynamic_geometry', '动态几何'),
                        ('new_definition', '题干属性'),
                        ('knowledge_count', '知识点个数'),
                        ('knowledge_diff', '知识点难度'),
                        ('cross_module', '知识点横跨'),
                        ('thinking_direction', '思维方向'),
                        ('cognitive_level', '认知水平'),
                    ]
                    for key, label in feature_fields:
                        value = features_obj.get(key, '')
                        if isinstance(value, list):
                            value = '、'.join(str(v) for v in value)
                        if value:
                            display_value = str(value)[:150] + ('...' if len(str(value)) > 150 else '')
                            html_content += f'''                        <div class="rating-detail-item">
                            <div class="label">{label}</div>
                            <div class="value">{escape(display_value)}</div>
                        </div>
'''
                    html_content += '                    </div>\n'
                html_content += '                </div>\n'

            # Annotation section
            html_content += f'''
                <div class="annotation-section">
                    <div class="annotation-title">人工标注</div>
                    <div class="annotation-row">
                        <label>评级结果：</label>
                        <button class="annotation-btn btn-correct" data-qid="{escape(question_id)}" data-action="correct" onclick="setAnnotation(this, 'correct')">✓ 正确</button>
                        <button class="annotation-btn btn-wrong" data-qid="{escape(question_id)}" data-action="wrong" onclick="setAnnotation(this, 'wrong')">✗ 错误</button>
                        <button class="annotation-btn btn-unmark" data-qid="{escape(question_id)}" data-action="unmark" onclick="setAnnotation(this, 'unmark')">— 未标注</button>
                    </div>
                    <div class="annotation-row">
                        <label>错误原因：</label>
                    </div>
                    <textarea class="annotation-textarea" data-qid="{escape(question_id)}" placeholder="如评级错误，请描述正确难度及原因..." oninput="saveAnnotationText(this)"></textarea>
                </div>
'''

            html_content += '''
            </div>
        </div>
'''

        html_content += '''
    </div>
'''

    html_content += f'''
    <a href="#" class="back-to-top">↑</a>

    <script>
    // ===== Annotation Data =====
    const LEVEL_NAMES = {json.dumps(LEVEL_NAMES, ensure_ascii=False)};
    const LEVEL_MAP = {json.dumps(LEVEL_MAP, ensure_ascii=False)};
    const allQuestions = {questions_json};

    // Load annotations from localStorage
    function loadAnnotations() {{
        try {{
            return JSON.parse(localStorage.getItem('difficulty_annotations') || '{{}}');
        }} catch {{ return {{}}; }}
    }}

    function saveAnnotations(annotations) {{
        localStorage.setItem('difficulty_annotations', JSON.stringify(annotations));
        updateStats();
    }}

    // Set annotation verdict (correct/wrong/unmark)
    function setAnnotation(btn, action) {{
        const qid = btn.getAttribute('data-qid');
        const annotations = loadAnnotations();
        const card = btn.closest('.question-card');
        const textarea = card.querySelector('.annotation-textarea');

        if (action === 'unmark') {{
            delete annotations[qid];
            textarea.value = '';
        }} else {{
            if (!annotations[qid]) annotations[qid] = {{}};
            annotations[qid].verdict = action;
        }}

        // Update button states
        const buttons = card.querySelectorAll('.annotation-btn');
        buttons.forEach(b => b.classList.remove('active'));
        if (action !== 'unmark') {{
            btn.classList.add('active');
        }}

        saveAnnotations(annotations);
    }}

    // Save annotation text
    function saveAnnotationText(textarea) {{
        const qid = textarea.getAttribute('data-qid');
        const annotations = loadAnnotations();
        if (!annotations[qid]) annotations[qid] = {{}};
        annotations[qid].reason = textarea.value;
        saveAnnotations(annotations);
    }}

    // ===== Stats =====
    function updateStats() {{
        const annotations = loadAnnotations();
        const total = allQuestions.length;
        const annotated = Object.keys(annotations).filter(k => annotations[k].verdict).length;
        const correct = Object.keys(annotations).filter(k => annotations[k].verdict === 'correct').length;

        // Per-level stats
        const levelStats = {{}};
        for (let lvl = 1; lvl <= 5; lvl++) {{
            levelStats[lvl] = {{ total: 0, annotated: 0, correct: 0 }};
        }}
        allQuestions.forEach(q => {{
            const lvl = q.level_num;
            if (lvl >= 1 && lvl <= 5) {{
                levelStats[lvl].total++;
                const ann = annotations[q.question_id];
                if (ann && ann.verdict) {{
                    levelStats[lvl].annotated++;
                    if (ann.verdict === 'correct') levelStats[lvl].correct++;
                }}
            }}
        }});

        // Build stats HTML
        let html = '<div class="stats-item">共 <span class="stats-value">' + total + '</span> 题</div>';
        html += '<div class="stats-item">已标注 <span class="stats-value">' + annotated + '</span> 题</div>';
        html += '<div class="stats-item">整体准确率 <span class="stats-value">' + (annotated > 0 ? (correct / annotated * 100).toFixed(1) + '%' : '—') + '</span></div>';
        html += '<span style="color:#ddd;">|</span>';

        for (let lvl = 1; lvl <= 5; lvl++) {{
            const s = levelStats[lvl];
            const acc = s.annotated > 0 ? (s.correct / s.annotated * 100).toFixed(1) + '%' : '—';
            html += '<span class="stats-level stats-level-' + lvl + '">' +
                LEVEL_NAMES[lvl].replace('难度' + lvl + ' — ', '') +
                ': ' + s.total + '题 / 标' + s.annotated + ' / 准' + acc + '</span>';
        }}

        document.getElementById('statsBar').innerHTML = html;
    }}

    // ===== Restore annotations on page load =====
    function restoreAnnotations() {{
        const annotations = loadAnnotations();
        document.querySelectorAll('.question-card').forEach(card => {{
            const qid = card.getAttribute('data-qid');
            const ann = annotations[qid];
            if (ann) {{
                if (ann.verdict) {{
                    const btn = card.querySelector('.annotation-btn[data-action="' + ann.verdict + '"]');
                    if (btn) btn.classList.add('active');
                }}
                if (ann.reason) {{
                    const textarea = card.querySelector('.annotation-textarea');
                    if (textarea) textarea.value = ann.reason;
                }}
            }}
        }});
    }}

    // ===== Export JSONL =====
    function exportJSONL() {{
        const annotations = loadAnnotations();
        const lines = allQuestions.map(q => {{
            const ann = annotations[q.question_id] || {{}};
            return JSON.stringify({{
                question_id: q.question_id,
                difficulty_level: q.difficulty_level,
                level_num: q.level_num,
                annotation_verdict: ann.verdict || '',
                annotation_reason: ann.reason || '',
                features: q.features,
                reasoning: q.reasoning
            }});
        }});
        downloadFile('difficulty_annotations.jsonl', lines.join('\\n'));
    }}

    // ===== Export TXT =====
    function exportTXT() {{
        const annotations = loadAnnotations();
        const lines = [];
        lines.push('初中数学题目难度评级标注结果');
        lines.push('='.repeat(60));
        lines.push('');

        for (let lvl = 1; lvl <= 5; lvl++) {{
            const qs = allQuestions.filter(q => q.level_num === lvl);
            if (qs.length === 0) continue;
            lines.push(LEVEL_NAMES[lvl] + ' (' + qs.length + '题)');
            lines.push('-'.repeat(40));
            qs.forEach((q, i) => {{
                const ann = annotations[q.question_id] || {{}};
                const verdict = ann.verdict === 'correct' ? '✓正确' : (ann.verdict === 'wrong' ? '✗错误' : '未标注');
                lines.push((i+1) + '. [ID:' + q.question_id + '] ' + q.difficulty_level + ' | 标注: ' + verdict);
                if (ann.reason) lines.push('   原因: ' + ann.reason);
            }});
            lines.push('');
        }}

        // Summary
        const annotated = Object.keys(annotations).filter(k => annotations[k].verdict).length;
        const correct = Object.keys(annotations).filter(k => annotations[k].verdict === 'correct').length;
        lines.push('统计汇总');
        lines.push('-'.repeat(40));
        lines.push('总题数: ' + allQuestions.length);
        lines.push('已标注: ' + annotated);
        lines.push('正确数: ' + correct);
        lines.push('准确率: ' + (annotated > 0 ? (correct / annotated * 100).toFixed(1) + '%' : '—'));

        downloadFile('difficulty_annotations.txt', lines.join('\\n'));
    }}

    function downloadFile(filename, content) {{
        const blob = new Blob([content], {{ type: 'text/plain;charset=utf-8' }});
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
    }}

    // ===== Smooth scroll for nav =====
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
        anchor.addEventListener('click', function(e) {{
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {{
                const navHeight = 160;
                const targetPosition = target.getBoundingClientRect().top + window.pageYOffset - navHeight;
                window.scrollTo({{top: targetPosition, behavior: 'smooth'}});
            }}
        }});
    }});

    // ===== Init =====
    restoreAnnotations();
    updateStats();
    </script>
</body>
</html>
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    print(f"HTML文件已生成: {output_path}")


def main():
    print("开始加载数据...")
    data = load_data(INPUT_PATH)
    print(f"加载完成，共 {len(data)} 条记录")

    print("\n加载图片URL映射...")
    image_urls = load_image_urls(IMAGE_URLS_PATH)
    print(f"图片URL映射加载完成，共 {len(image_urls)} 条记录")

    print("\n按难度级别分组...")
    samples = sample_by_level(data, SAMPLE_SIZE)

    if not samples:
        print("警告：没有找到有效的评级数据！")
        return

    print("\n生成HTML文件...")
    output_path = os.path.join(OUTPUT_DIR, "difficulty_v5_samples.html")
    generate_html(samples, image_urls, output_path)

    print("\n完成！")
    print(f"HTML文件路径: {output_path}")


if __name__ == "__main__":
    main()