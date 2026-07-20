# 初中物理难度打标评测工具

本项目用于：读取题目 JSONL，调用服务器上的 OpenAI-compatible Responses API，输出模型难度评级，并与老师人工标签进行对比。

## 数据分层

```text
data/
├── labeled/                         老师人工标注数据（评测基准）
│   ├── physics_difficulty_tiku_data_0714_1000.jsonl
│   └── physics_teacher_labels_0714.csv
├── physics_difficulty_tiku_data_v2.jsonl       历史评测输入
├── physics_difficulty_tiku_rated_v2_results.jsonl  历史模型结果
└── physics_sampled_5000_per_difficulty*.jsonl  大规模抽样题库

outputs/
├── model_runs/                      当前冻结结果，不作为输入数据
│   └── history/                     历史模型实验版本
└── logs/
    └── history/                     历史运行日志

prompts/                             正式生产 Prompt
└── archive/                         冻结历史 Prompt（仅兼容回放）
src/                                 正式评级和渲染脚本
└── legacy/                          冻结的历史后处理参考实现
tests/                               单题、视觉和对比实验脚本
archive/                             历史脚本，仅供追溯
docs/                                分档标准文档
```

重要字段区分：

- 题目 JSONL 中的 `stem`、`options`、`analysis`、`sub_questions` 和图片 URL 是题目内容。
- 老师真实标签来自 `data/labeled/physics_teacher_labels_0714.csv`：`ID` 对应 `question_id`，`难度` 是人工标签。
- 模型结果中的 `difficulty_rating_raw` 是模型原始 JSON；`difficulty_rating.difficulty_level` 是后处理后的评级。
- 输出顶层的 `difficulty_level_raw`、`postprocess_actions` 用于审计；输入原 `difficulty` 会被改名为 `source_difficulty_untrusted`。
- 不要使用题目 JSONL 中的 `difficulty` 字段作为最新老师标签；最新评测应以 CSV 为准。

教师标签映射为：容易=送分题，较易=基础题，中等=中等题，较难=拔高题，困难=压轴题。

## 配置

```bash
cp .env.example .env
```

`.env` 示例：

```ini
API_KEY=not-needed
BASE_URL=http://172.22.0.35:4466/v1
MODEL_NAME=doubao-seed-2.0-lite
# Lite 服务端固定为 1，脚本会忽略其他传入值并发送 1。
TEMPERATURE=1
# 默认使用正式 Prompt + 冻结兼容语义层 + 稳定结构规则。
RATING_PROFILE=v7_stable
```

Mini 等支持调温度的模型仍会读取 `TEMPERATURE`；未配置时不发送该字段。

真实模型调用需要在服务器的 venv 中执行，本机只适合做静态检查。

## 评级配置

- `v7_stable`：默认值。使用正式 Prompt、冻结兼容后处理，再执行少量可泛化的结构稳定规则。
- `v7_compat`：旧 V7 原样对照，不执行稳定补丁。
- `fused`、`generalized`：保留用于历史实验对照，不再作为生产默认路径。

正式 Prompt 当前冻结为 `accuracyfix` 生产版，运行配置名仍为 `v7_stable`。文件内容精确对应提交 `4819dca` 中的 `prompts/初中物理难度打标提示词.txt`；实验昵称不是新的 `RATING_PROFILE`，运行时不要填写 `accuracyfix`。

该版本使用 Lite 在同一批 1066 题上独立运行三次，以 GPT-5.6 裁定标签严格评估，最终完全一致率分别为 72.98%、71.95%、72.23%，平均 72.39%；相差不超过一档比例平均 99.65%，MAE 平均 0.2795。后续 final-candidate 扩充规则后三次平均准确率降至 70.51%，因此正式 Prompt 已恢复并冻结到 `accuracyfix`。第二阶段边界复核工具继续保留，但不参与首轮 Prompt，也不进行三次投票。

## 评测命令

运行 V7 稳定版（Lite 温度固定为 1）：

```bash
RATING_PROFILE=v7_stable MODEL_NAME=doubao-seed-2.0-lite TEMPERATURE=1 \
python src/physics_difficulty_rating_with_cache.py \
  -i data/labeled/physics_difficulty_tiku_data_v2.jsonl \
  -o outputs/model_runs/lite_physics_v2_v7_stable_run1.jsonl \
  -e outputs/model_runs/lite_physics_v2_v7_stable_run1_errors.jsonl \
  -p prompts/初中物理难度打标提示词.txt \
  -c 30 --no-cache
```

该输入只有 133 题，因此不要加 `-n`。Lite 的 `temperature` 服务端固定为 1；稳定性应通过同一输入连续跑三次比较，而不是设置 0。

从最新老师标注题目抽取固定样本、禁用缓存并运行：

```bash
source venv/bin/activate
python src/physics_difficulty_rating_with_cache.py \
  -i data/labeled/physics_difficulty_tiku_data_0714_1000.jsonl \
  -o outputs/model_runs/lite_default_100.jsonl \
  -e outputs/model_runs/lite_default_100_errors.jsonl \
  -p prompts/初中物理难度打标提示词.txt \
  -c 20 -n 100 --seed 20260714 --no-cache
```

参数说明：

- `--seed` 固定抽样结果，便于重复实验；
- `--no-cache` 不使用前缀缓存，适合稳定性对照；
- `-c` 控制并发数，服务器出现 429 时应调低；
- 输出顶层 `rating_profile` 记录本次规则配置；`difficulty_level_raw` 是模型原始评级，`difficulty_rating.difficulty_level` 是后处理后的评级，`postprocess_actions` 记录后处理动作。

完整的数据口径、few-shot 表、后处理规则和 200 题分层回归命令见 [PHYSICS_RATING_REVISION.md](PHYSICS_RATING_REVISION.md)。

## 第二阶段边界复核

`src/physics_boundary_second_review.py` 只复核单次首轮结果，不对三次运行投票或合并。三次实验应分别评估稳定性；需要复核错题时，先从指定的一次运行导出全部错题，再逐题复核。复核器默认只处理以下候选：

- 任一次首轮发生过后处理调整；
- 最终等级与 18 维结构特征明显靠近相邻档边界。

复核结果只有在等级合法、相对首轮最多移动一档、证据字段完整且置信度为“高”时才自动生效。首轮结果、复核原始 JSON、调整原因、耗时和 token 均保留在输出中。

先导出 Run1 全部错题：

```bash
python tests/adjudication_label_regression.py \
  --csv data/labeled/physics_adjudicated_labels_gpt56_1066.csv \
  --jsonl data/labeled/physics_difficulty_tiku_data_0714_1000.jsonl \
  --evaluate outputs/model_runs/history/lite_physics_final_candidate_1066_run1.jsonl \
  --export-mismatches outputs/model_runs/history/lite_physics_final_candidate_1066_run1_mismatches.jsonl
```

再查看复核候选数量，不发模型请求：

```bash
python src/physics_boundary_second_review.py \
  -i outputs/model_runs/history/lite_physics_final_candidate_1066_run1_mismatches.jsonl \
  -o outputs/model_runs/lite_physics_final_candidate_1066_run1_reviewed.jsonl \
  -e outputs/model_runs/lite_physics_final_candidate_1066_run1_review_errors.jsonl \
  --review-mode all --dry-run
```

正式复核时删除 `--dry-run`。错题包应使用 `all`，保证每道错题都复核；完整1066题输入可使用 `selective` 或 `broad`。复核输出会把分歧归为“模型确实误判”“参考标签需修订”“相邻边界均可”或“双方均需修订”。可用 `--model doubao-seed-2.0-pro` 单独指定复核模型。

## 冻结首轮后的盲审 Agent Pipeline

`src/physics_difficulty_agent_pipeline.py` 不修改首轮 Prompt 或后处理。它先用确定性结构规则选择高风险题，再把题干、选项和解析发送给 `doubao-seed-2.0-mini` 做 `temperature=0` 独立盲审。盲审请求不会包含首轮等级、features、reasoning、后处理动作、来源 difficulty 或评估标签。

只有同时满足以下条件才自动写回：盲审结论为高置信度、相对首轮只差一档、盲审不再接受当前等级、证据字段完整、有效决策数与目标档一致，并且调整方向与风险路由一致。压轴写回还要求至少两类强压轴结构。两档以上分歧只记录，不自动跨档。

Pipeline 会在调整前保存完整 `difficulty_rating_before_verification` 快照。每行还记录由输入文件、盲审 Prompt、模型、温度和置信度门槛共同生成的 `run_signature`；断点续跑检测到签名不一致时会拒绝混写，要求更换输出文件。

先离线查看候选数量，不调用模型：

```bash
python src/physics_difficulty_agent_pipeline.py \
  -i outputs/model_runs/lite_physics_erroraudit_guard_1066_run1.jsonl \
  -o outputs/model_runs/lite_physics_erroraudit_guard_agent_run1.jsonl \
  -e outputs/model_runs/lite_physics_erroraudit_guard_agent_run1_errors.jsonl \
  --dry-run
```

正式运行时删除 `--dry-run`，也可显式指定 `--model doubao-seed-2.0-mini --temperature 0`。运行后继续用 `tests/adjudication_label_regression.py` 评估；报告中的 `before_verification_evaluation` 与 `verification_agent` 会分别给出首轮基线和 Agent 的改对、改错、置信度及档位转移。

在授予盲审模型自动写回权限前，推荐先使用 Lite 做只审不改实验：

```bash
python src/physics_difficulty_agent_pipeline.py \
  -i outputs/model_runs/lite_physics_erroraudit_guard_1066_run1.jsonl \
  -o outputs/model_runs/lite_physics_agent_lite_audit_run1.jsonl \
  -e outputs/model_runs/lite_physics_agent_lite_audit_run1_errors.jsonl \
  -p prompts/初中物理难度盲审提示词.txt \
  --model doubao-seed-2.0-lite \
  --temperature 1 \
  --audit-only \
  -c 30
```

`--audit-only` 会正常调用模型并保存盲审等级、置信度及“按现有门控本可写回”的 `would_apply`，但 `verification_applied` 始终为 `false`，最终等级与冻结输入完全一致。运行签名包含该模式，不能与自动写回结果混用同一输出文件。评测报告中的 `verification_agent.audit_comparison` 会直接比较风险题上冻结判断与盲审判断的准确率及分歧胜负。

## 代码约束

`src/physics_difficulty_rating_with_cache.py` 保持以下兼容性：

- Responses API、缓存、并发、重试、断点续跑和 JSONL 输入输出；
- 五档 `difficulty_level` 字符串；
- 18 个 `features` 字段及其合法枚举；
- `coarse_difficulty` 和四个 `reasoning` 字段。

历史完整实现冻结在 `src/legacy/`；默认 `v7_stable` 先调用该实现，再应用主脚本中的少量结构稳定规则。正式入口是 `prompts/初中物理难度打标提示词.txt`，归档 Prompt 保持原样，仅用于对照。
