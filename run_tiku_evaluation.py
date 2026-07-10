#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : run_tiku_evaluation.py
@Description:
    一键评估 physics_difficulty_tiku_data_v2.jsonl 难度的启动脚本。
    自动检测虚拟环境，并对齐 prompt_test 工作目录以正确加载 .env 接口配置。
"""

import os
import sys
import subprocess

def main():
    # 确定根目录和各文件绝对路径
    root_dir = os.path.dirname(os.path.abspath(__file__))
    
    script_path = os.path.join(root_dir, "src", "physics_difficulty_rating_with_cache.py")
    input_path = os.path.join(root_dir, "data", "physics_difficulty_tiku_data_v2.jsonl")
    output_path = os.path.join(root_dir, "data", "physics_difficulty_tiku_rated_v2_results.jsonl")
    error_path = os.path.join(root_dir, "data", "physics_difficulty_tiku_rated_v2_errors.jsonl")
    prompt_path = os.path.join(root_dir, "prompts", "初中物理难度打标提示词.txt")
    
    # 1. 寻找可用的 Python 解释器
    venv_python = os.path.join(root_dir, "venv", "bin", "python")
    if os.path.exists(venv_python):
        python_bin = venv_python
    else:
        python_bin = sys.executable or "python3"

    print("=================== 物理题库难度评估启动 ===================")
    print(f"解释器  : {python_bin}")
    print(f"主脚本  : {script_path}")
    print(f"输入文件: {input_path}")
    print(f"输出文件: {output_path}")
    print(f"提示词  : {prompt_path}")
    print("==========================================================")
    
    # 2. 构造命令
    cmd = [
        python_bin, script_path,
        "-i", input_path,
        "-o", output_path,
        "-e", error_path,
        "-p", prompt_path,
        "-c", "15"  # 默认并发15，可以根据需要调节
    ]
    
    # 3. 必须在 prompt_test 目录下执行，以正确载入 .env 文件中的 API_KEY/BASE_URL
    cwd = root_dir
    
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
        print("\n✨ 物理题库难度打标评估运行结束！")
        print(f"👉 成功结果已保存至: {output_path}")
        print(f"👉 错误日志已保存至: {error_path}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 打标脚本执行失败，退出码: {e.returncode}")
    except Exception as e:
        print(f"\n❌ 启动执行时发生未知异常: {e}")

if __name__ == "__main__":
    main()
