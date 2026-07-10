# 初中物理难度打标 Prompt 自动化测试工具

该工具用于从 `physics_sampled_5000_per_difficulty.jsonl` 中自动抽取物理题目，调用大模型（使用 `初中物理难度打标提示词.txt` 作为系统提示词），从而快速直观地测试、验证 Prompt 的准确率和输出效果。

## ⚙️ 快速使用步骤

### 1. 安装依赖
请确保系统已安装 Python 3，然后安装依赖库：
```bash
pip install -r requirements.txt
```

### 2. 配置 API Key
将当前文件夹下的 `.env.example` 复制并重命名为 `.env`：
```bash
cp .env.example .env
```
然后编辑 `.env` 文件，填写您的模型提供商参数（如 API_KEY, BASE_URL 等）：
```ini
API_KEY=your_real_api_key
BASE_URL=https://api.deepseek.com/v1 # 或您的本地/其他服务商地址
MODEL_NAME=deepseek-chat
```

### 3. 运行测试

*   **测试任意难度的前 3 道题**：
    ```bash
    python test_prompt.py
    ```

*   **指定测试 4 档（拔高题）的 5 道题目**：
    ```bash
    python test_prompt.py --difficulty 4 --num 5
    ```

*   **随机抽取 2 道 5 档（压轴题）的题目进行测试**：
    ```bash
    python test_prompt.py --difficulty 5 --num 2 --random
    ```

## 📋 测试结果解读
脚本运行后，控制台会输出：
1.  题目的题干预览及大模型打标过程中返回的 JSON。
2.  自动提取 JSON 中的 `difficulty_level` 并与原库中的 `difficulty` 真实标注进行横向对比，若一致则显示 `匹配成功 (PASS)`，若不一致则显示 `存在偏离 (FAIL)`，帮助您迅速排查打标偏差。
