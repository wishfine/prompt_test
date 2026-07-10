# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY", "not-needed")
BASE_URL = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1")
if not BASE_URL.endswith("/"):
    BASE_URL += "/"
MODEL_NAME = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")

PROMPT_FILE = "../prompts/初中物理难度打标提示词.txt"
DATASET_FILE = "../data/physics_sampled_5000_per_difficulty_v2.jsonl"

def load_prompt():
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    if "## 输入题目信息" in content:
        parts = content.split("## 输入题目信息")
        prefix = parts[0] + "## 输入题目信息"
        suffix = "\n\n请根据以上信息，对题目进行全面的难度分析 and 评级。"
        return prefix, suffix
    return content, ""

def test_single_multimodal():
    prefix, suffix = load_prompt()
    
    # 找出一道有图的题
    test_item = None
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            if item.get("stem_pic_url") and item.get("analysis_pic_url"):
                test_item = item
                break
                
    if not test_item:
        print("未找到同时有题干和解析图片的题目！")
        return
        
    print(f"找到测试题 ID: {test_item['question_id']}")
    print(f"题干文本: {test_item['stem']}")
    print(f"题干图片: {test_item['stem_pic_url']}")
    print(f"解析图片: {test_item['analysis_pic_url']}")

    # 构造多模态 Payload (只传入图片 URL，不传入文字)
    print("\n--- 1. 构造多模态只读图 Payload (直连 chat/completions) ---")
    content_list = []
    # 1. 拼入前缀提示词
    content_list.append({"type": "text", "text": prefix + "\n"})
    # 2. 拼入引导语与图片
    content_list.append({"type": "text", "text": "【题干图示与解析图示如下，请仅根据图片中的内容进行难度评级。】\n"})
    
    for url in test_item["stem_pic_url"].split(","):
        if url.strip():
            content_list.append({
                "type": "image_url",
                "image_url": {"url": url.strip()}
            })
            
    for url in test_item["analysis_pic_url"].split(","):
        if url.strip():
            content_list.append({
                "type": "image_url",
                "image_url": {"url": url.strip()}
            })
            
    # 3. 拼入后缀提示词
    content_list.append({"type": "text", "text": suffix})

    # 发送请求
    payload_chat = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": content_list}],
        "thinking": {"type": "disabled"}
    }
    
    url_endpoint = f"{BASE_URL}chat/completions"

    print(f"请求地址: {url_endpoint}")
    try:
        t1 = time.time()
        res = requests.post(
            url_endpoint,
            json=payload_chat,
            headers={"Authorization": f"Bearer {API_KEY}"},
            timeout=120
        )
        t2 = time.time()
        print(f"打分响应状态码: {res.status_code}，耗时: {t2-t1:.2f}秒")
        if res.status_code == 200:
            result = res.json()
            output_text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            print("\n[模型返回结果]:")
            print(output_text[:300] + "...")
            print("\n[Token 消耗]:")
            print(json.dumps(usage, indent=2))
        else:
            print(f"打标失败报错: {res.text}")
    except Exception as e:
        print(f"请求打标失败: {e}")

if __name__ == "__main__":
    test_single_multimodal()
