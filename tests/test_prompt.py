import os
import sys
import json
import random
import argparse
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# 加载当前目录下的 .env 配置文件
load_dotenv()

# 常量定义
PROMPT_FILE = "../prompts/初中物理难度打标提示词.txt"
DATASET_FILE = "../data/physics_sampled_5000_per_difficulty.jsonl"

def load_system_prompt():
    """读取提示词模板文件"""
    if not os.path.exists(PROMPT_FILE):
        print(f"错误: 找不到提示词文件 {PROMPT_FILE}，请确保该文件在父目录下存在！")
        sys.exit(1)
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()

def load_samples(difficulty=None, num_samples=3, use_random=False):
    """从 jsonl 数据集中提取样本数据"""
    if not os.path.exists(DATASET_FILE):
        print(f"错误: 找不到数据集文件 {DATASET_FILE}，请确认数据集路径！")
        sys.exit(1)
    
    samples = []
    with open(DATASET_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                if difficulty is None or int(data.get("difficulty")) == int(difficulty):
                    samples.append(data)
            except Exception:
                continue

    if not samples:
        print(f"未找到匹配的样本 (难度过滤: {difficulty})")
        return []

    if use_random:
        return random.sample(samples, min(num_samples, len(samples)))
    else:
        return samples[:num_samples]

def format_user_content(sample):
    """将样本数据格式化为用户输入"""
    content = []
    content.append(f"【题干】\n{sample.get('stem', '').strip()}\n")
    
    options = sample.get('options')
    if options:
        content.append(f"【选项】\n{options}\n")
        
    analysis = sample.get('analysis')
    if analysis:
        content.append(f"【解析】\n{analysis}\n")
        
    sub_qs = sample.get('sub_questions', [])
    if sub_qs:
        content.append("【子问列表】")
        for idx, sq in enumerate(sub_qs):
            content.append(f"小问 {idx+1}: {sq}")
            
    return "\n".join(content)

def request_llm_worker(client, model_name, system_prompt, sample, idx, total, print_lock):
    """单道题目的打标工作线程"""
    user_content = format_user_content(sample)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            }
        )
        
        result_text = response.choices[0].message.content
        result_json = json.loads(result_text)
        
        pred_level = result_json.get("difficulty_level")
        
        # 使用线程锁安全打印模型独立的打标结论
        with print_lock:
            print(f"[{idx}/{total}] ID: {sample.get('id')} | 预测难度: {pred_level}")
                
        return {
            "index": idx,
            "question_id": sample.get("id"),
            "stem": sample.get("stem"),
            "predicted_difficulty": pred_level,
            "model_output": result_json
        }
    except Exception as e:
        with print_lock:
            print(f"[{idx}/{total}] ID: {sample.get('id')} | ❌ 避开异常，标注失败: {e}")
        return {
            "index": idx,
            "question_id": sample.get("id"),
            "stem": sample.get("stem"),
            "predicted_difficulty": "ERROR",
            "error_message": str(e),
            "model_output": {}
        }

def run_prompt_test(prompt_path, difficulty, num_samples, use_random, concurrency):
    # 检查 API Key
    api_key = os.getenv("API_KEY", "not-needed")
    base_url = os.getenv("BASE_URL", "http://172.22.0.35:4466/v1")
    model_name = os.getenv("MODEL_NAME", "doubao-seed-2.0-lite")

    print(f"正在加载提示词: {prompt_path}...")
    if not os.path.exists(prompt_path):
        print(f"错误: 找不到提示词文件 {prompt_path}！")
        sys.exit(1)
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print("正在加载样本数据...")
    samples = load_samples(difficulty, num_samples, use_random)
    
    if not samples:
        return

    # 初始化 OpenAI 兼容客户端
    client = OpenAI(api_key=api_key, base_url=base_url)

    print(f"\n======== 开始并发打标 (共 {len(samples)} 道题, 线程数: {concurrency}) ========")
    test_records = [None] * len(samples)
    print_lock = Lock()

    # 使用线程池并发发起 API 请求
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                request_llm_worker, 
                client, 
                model_name, 
                system_prompt, 
                sample, 
                idx + 1, 
                len(samples), 
                print_lock
            ): idx
            for idx, sample in enumerate(samples)
        }
        
        for future in as_completed(futures):
            idx = futures[future]
            try:
                record = future.result()
                test_records[idx] = record
            except Exception as e:
                print(f"线程执行出现未知异常: {e}")

    # 运行结束，写入文件
    jsonl_file = "test_results.jsonl"
    summary_file = "test_results_summary.txt"

    # 1. 写入 jsonl 详细数据
    with open(jsonl_file, "w", encoding="utf-8") as f_jsonl:
        for rec in test_records:
            if rec is not None:
                f_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 2. 写入 总结报表（清晰列出每道题的预测难度和分析，方便教研组抽审）
    with open(summary_file, "w", encoding="utf-8") as f_sum:
        f_sum.write("==================================================\n")
        f_sum.write("            初中物理难度自主打标测试报告\n")
        f_sum.write("==================================================\n")
        f_sum.write(f"提示词路径: {prompt_path}\n")
        f_sum.write(f"测试模型: {model_name}\n")
        f_sum.write(f"总打标题目数: {len(test_records)} 道\n")
        f_sum.write("==================================================\n\n")
        
        f_sum.write("📋 【打标结果明细列表】:\n")
        f_sum.write("-" * 50 + "\n")
        for rec in test_records:
            if rec is not None:
                f_sum.write(f"题号: {rec['index']} | 题目ID: {rec['question_id']} | 【预测难度】: {rec['predicted_difficulty']}\n")
                f_sum.write(f"题干: {rec['stem'][:150].strip()}...\n")
                reasoning = rec["model_output"].get("reasoning", {})
                f_sum.write(f"定位依据: {reasoning.get('core_basis', '')}\n")
                f_sum.write(f"难点分析: {reasoning.get('hard_point', '')}\n")
                f_sum.write("-" * 50 + "\n")

    print(f"\n✨ 打标完成！结果已自动保存至本地:")
    print(f"👉 结构化记录: {os.path.abspath(jsonl_file)}")
    print(f"👉 易读版打标明细报告: {os.path.abspath(summary_file)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="初中物理难度打标 Prompt 效果测试脚本")
    parser.add_argument("-p", "--prompt", type=str, default="../prompts/初中物理难度打标提示词.txt",
                        help="指定提示词文件的路径，默认为 ../prompts/初中物理难度打标提示词.txt")
    parser.add_argument("-d", "--difficulty", type=int, choices=[1, 2, 3, 4, 5], default=None,
                        help="过滤测试样本的难度档位 (1-5)，不指定则从所有难度中随机或顺序抽取")
    parser.add_argument("-n", "--num", type=int, default=3,
                        help="测试题目数量 (默认 3)")
    parser.add_argument("-r", "--random", action="store_true",
                        help="是否随机抽取样本，默认顺序抽取")
    parser.add_argument("-c", "--concurrency", type=int, default=10,
                        help="多线程打标线程并发数，默认 10")
    
    args = parser.parse_args()
    run_prompt_test(args.prompt, args.difficulty, args.num, args.random, args.concurrency)
