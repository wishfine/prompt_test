# 高中化学两阶段测试集

## 2026-08-07 ChatGPT 全量复核500题

- `chatgpt_reference_20260807_test500_blind.jsonl`：模型输入，只包含题目、解析和图片字段。
- `chatgpt_reference_20260807_test500_labels.jsonl`：独立参考标签和复核审计字段，只用于评测，禁止发送给模型。

参考标签分布：

| 档位 | 数量 |
|---|---:|
| 难度1档 | 49 |
| 难度2档 | 154 |
| 难度3档 | 114 |
| 难度4档 | 172 |
| 难度5档 | 11 |

原始文件中的 `label_status` 明确说明这是一套多轮 AI 复核参考标签，并非教师金标准。

## 运行

```bash
MODEL_NAME=doubao-seed-2.0-lite \
TEMPERATURE=1 \
ENABLE_STAGE2_AUTO_ADJUST=1 \
python src/high_chemistry_difficulty_rating_and_verify.py \
  -i data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_blind.jsonl \
  -o outputs/model_runs/high_chemistry_reference500_twostage_run1.jsonl \
  -e outputs/model_runs/high_chemistry_reference500_twostage_run1_errors.jsonl \
  -p prompts/高中化学难度打标提示词.txt \
  -c 20 --image-mode auto --no-cache
```

## 评测

```bash
python tools/evaluate_high_chemistry_test500.py \
  --labels data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_labels.jsonl \
  --predictions outputs/model_runs/high_chemistry_reference500_twostage_run1.jsonl \
  --output outputs/model_runs/high_chemistry_reference500_twostage_run1_metrics.json
```

评测报告同时输出第一阶段、最终结果、原始正确率集中度和第二阶段触发诊断。
