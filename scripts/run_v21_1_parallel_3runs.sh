#!/usr/bin/env bash
set -e

# ==============================================================================
# 高中化学 V21.1 3跑并行测试与自动评测脚本
# ==============================================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

# 1. 基础配置（可替换为全新 holdout 路径）
INPUT_FILE="${INPUT_FILE:-data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_blind.jsonl}"
LABELS_FILE="${LABELS_FILE:-data/high_chemistry/test_sets/chatgpt_reference_20260807_test500_labels.jsonl}"
PROMPT_FILE="${PROMPT_FILE:-prompts/高中化学难度打标提示词.txt}"
MODEL_NAME="${MODEL_NAME:-doubao-seed-2.0-lite}"
CONCURRENCY="${CONCURRENCY:-20}"

mkdir -p outputs/model_runs outputs/logs output/doc

echo "================================================================================"
echo "启动 V21.1 高中化学 3 跑并行运行"
echo "输入数据: ${INPUT_FILE}"
echo "标签数据: ${LABELS_FILE}"
echo "模型名称: ${MODEL_NAME}"
echo "并发限制: ${CONCURRENCY} (单跑)"
echo "================================================================================"

# 2. 并行启动 3 个 Run
PIDS=()
for i in 1 2 3; do
  OUT_JSONL="outputs/model_runs/high_chemistry_v21_1_run${i}.jsonl"
  ERR_JSONL="outputs/model_runs/high_chemistry_v21_1_run${i}_errors.jsonl"
  LOG_FILE="outputs/logs/high_chemistry_v21_1_run${i}.log"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] 启动 Run ${i} -> ${OUT_JSONL} (日志: ${LOG_FILE})"

  MODEL_NAME="${MODEL_NAME}" \
  TEMPERATURE=1 \
  ENABLE_STAGE2_AUTO_ADJUST=1 \
  ./venv/bin/python src/high_chemistry_difficulty_rating_and_verify.py \
    -i "${INPUT_FILE}" \
    -o "${OUT_JSONL}" \
    -e "${ERR_JSONL}" \
    -p "${PROMPT_FILE}" \
    -c "${CONCURRENCY}" \
    --image-mode auto \
    --no-cache > "${LOG_FILE}" 2>&1 &

  PIDS+=($!)
done

echo "3 跑已全部在后台并发启动，PIDs: ${PIDS[*]}"
echo "正在等待所有任务执行完毕..."

# 3. 等待所有并发子进程结束
for pid in "${PIDS[@]}"; do
  wait "${pid}"
done

echo "================================================================================"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 3 跑模型调用已全部执行完成，开始自动评估..."
echo "================================================================================"

# 4. 执行单跑评测
for i in 1 2 3; do
  OUT_JSONL="outputs/model_runs/high_chemistry_v21_1_run${i}.jsonl"
  METRICS_JSON="outputs/model_runs/high_chemistry_v21_1_run${i}_metrics.json"
  echo "正在评估 Run ${i}..."
  ./venv/bin/python tools/evaluate_high_chemistry_test500.py \
    --labels "${LABELS_FILE}" \
    --predictions "${OUT_JSONL}" \
    --output "${METRICS_JSON}"
done

# 5. 执行 3 跑一致性与稳定性评测
echo "正在计算 3 跑稳定性与一致性 (Pairwise & All-Three Agreement)..."
./venv/bin/python tools/evaluate_high_chemistry_stability.py \
  --labels "${LABELS_FILE}" \
  --runs outputs/model_runs/high_chemistry_v21_1_run1.jsonl \
         outputs/model_runs/high_chemistry_v21_1_run2.jsonl \
         outputs/model_runs/high_chemistry_v21_1_run3.jsonl \
  --output outputs/model_runs/high_chemistry_v21_1_stability_3runs.json

# 6. 执行 3 跑汇总规则真实消融分析
echo "正在执行 3 跑 LORO 规则边际贡献消融分析..."
PYTHONPATH=src ./venv/bin/python tools/evaluate_v21_rule_ablation.py \
  --labels "${LABELS_FILE}" \
  --predictions "outputs/model_runs/high_chemistry_v21_1_run1.jsonl" \
                "outputs/model_runs/high_chemistry_v21_1_run2.jsonl" \
                "outputs/model_runs/high_chemistry_v21_1_run3.jsonl" \
  --output-json outputs/model_runs/high_chemistry_v21_1_rule_ablation_3runs.json \
  --output-md output/doc/high_chemistry_v21_1_rule_ablation_3runs_report.md

echo "================================================================================"
echo "全部并行运行与评估分析已完成！"
echo "- 单跑评测指标: outputs/model_runs/high_chemistry_v21_1_run*_metrics.json"
echo "- 3 跑稳定性报告: outputs/model_runs/high_chemistry_v21_1_stability_3runs.json"
echo "- 规则消融分析报告: output/doc/high_chemistry_v21_1_rule_ablation_3runs_report.md"
echo "================================================================================"
