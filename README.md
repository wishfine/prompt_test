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

## 初中英语知识点排序

排序 Prompt 位于 `prompts/初中英语知识点排序提示词.txt`。程序只允许模型重排每条记录原有 `output` 中的分号分隔项，并会用多重集合校验阻止标签增删改写。成功结果的 `output` 会改为排序后字符串，原始顺序保存在 `original_output`，同时保留 `ordered_output` 数组和 `sorted_output` 字段。项目中确认可用的模型名是 `doubao-seed-2.0-lite` 和 `deepseek-v4-flash`。

在服务器项目目录执行（默认使用豆包 Lite）：

```bash
source venv/bin/activate
python scripts/sort_junior_english_knowledge_points.py \
  --input /home/share_ssd_data/nfs-data1/wangmeng148/coding/vllm-main/scripts/tiku_multiq_func_label_cls/data_process/output/main_questions_2to3_kp_labels_1000.jsonl \
  --output outputs/english_kp_ordering/doubao_seed_2.0_lite.jsonl \
  --prompt prompts/初中英语知识点排序提示词.txt \
  --model doubao-seed-2.0-lite \
  --max-workers 20 \
  --retries 3 \
  --resume
```

如需改用 DeepSeek V4 Flash，只替换模型名和输出文件：

```bash
python scripts/sort_junior_english_knowledge_points.py \
  --input /home/share_ssd_data/nfs-data1/wangmeng148/coding/vllm-main/scripts/tiku_multiq_func_label_cls/data_process/output/main_questions_2to3_kp_labels_1000.jsonl \
  --output outputs/english_kp_ordering/deepseek_v4_flash.jsonl \
  --prompt prompts/初中英语知识点排序提示词.txt \
  --model deepseek-v4-flash \
  --max-workers 20 \
  --retries 3 \
  --resume
```

排序完成后生成接近初中化学验收页逻辑的交互式 HTML（统计栏、Top 知识点、搜索、状态筛选、逐题原始/排序后对照）：

```bash
python scripts/generate_junior_english_knowledge_points_html.py \
  --input outputs/english_kp_ordering/doubao_seed_2.0_lite.jsonl \
  --output outputs/english_kp_ordering/doubao_seed_2.0_lite.html
```

HTML 页面可直接在服务器下载后用浏览器打开；也可以把 `--input` 换成 DeepSeek 的结果文件。

## 评级配置

- 正式生产入口固定使用 `gpt56_hybrid`、`doubao-seed-2.0-lite`、三次独立评级和结构化送分边界校准；不要再手工拼接生产环境变量。
- `v7_stable`：直接调用底层评级脚本时的兼容默认值，保留用于历史回放，不是当前生产编排入口。
- `v7_compat`：旧 V7 原样对照，不执行稳定补丁。
- `fused`、`generalized`：保留用于历史实验对照，不再作为生产默认路径。

正式生产固定读取 `prompts/初中物理难度打标提示词.txt`，并由生产入口设置
`RATING_PROFILE=gpt56_hybrid`。`accuracyfix` 等名称只是历史实验昵称，不是可用的
`RATING_PROFILE`；`v7_stable` 仅用于历史单次回放。

历史 `accuracyfix` 实验使用 Lite 在同一批 1066 题上独立运行三次，以 GPT-5.6
裁定标签严格评估，最终完全一致率分别为 72.98%、71.95%、72.23%，平均
72.39%；相差不超过一档比例平均 99.65%，MAE 平均 0.2795。该结果仅作历史
基线，不代表当前三次集成生产指标。第二阶段边界复核工具继续保留，但不进入
正式生产流程。

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

## 正式生产运行

正式生产只使用以下冻结流程：

```text
同一输入独立调用 Lite 三次
→ 每次执行正式 Prompt 与 gpt56_hybrid 后处理
→ 确定性多数票
→ 仅在送分/基础分歧有显性应用证据时执行结构化校准
→ 输出完整性、版本签名和分布监控
```

运行：

```bash
cd ~/prompt_test
git pull --ff-only origin main
mkdir -p outputs/production

nohup scripts/run_physics_production.sh \
  -i data/samples/physics_batch.jsonl \
  -o outputs/production/physics_batch_20260728 \
  -c 30 \
  > outputs/production/physics_batch_20260728.log 2>&1 &
```

输出包括：

- `*_run1.jsonl`、`*_run2.jsonl`、`*_run3.jsonl`：三次独立结果；
- `*_final.jsonl`：三次集成后的正式结果；
- `*_production_manifest.json`：输入、Prompt、代码、Git 提交和参数签名；
- `*_monitoring.json`：分布、一致率、分歧率、校准触发率、Token、耗时和错误日志摘要。

生产入口默认禁用缓存，并固定模型、温度、Prompt、评级配置、后处理开关和
三次调用数。若任务中断，使用完全相同的命令可断点续跑；如果输入、Prompt、
代码版本或参数发生变化，签名校验会拒绝混写，必须使用新的输出前缀。

仅检查配置、不调用模型：

```bash
scripts/run_physics_production.sh \
  -i data/samples/physics_batch.jsonl \
  -o outputs/production/physics_batch_20260728 \
  --dry-run
```

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

## 冻结首轮后的证据审计 Agent Pipeline

`src/physics_difficulty_agent_pipeline.py` 不修改首轮 Prompt 或后处理。V2先用确定性结构规则选择高风险题，再让 Lite 核验首轮features、reasoning和后处理动作是否得到题干与官方解析支持。它不是第二个自由分类器：没有题目原文可逐字核验的反证时必须保留冻结等级。

证据审计请求会包含题目、冻结等级、首轮结构主张、后处理动作和风险原因，但继续隔离来源 difficulty 和所有评估标签。建议调整必须满足：只移动相邻一档、方向得到风险路由支持、当前等级被明确否定、候选档必要条件全部满足、至少一条反证的原文摘录能在指定题目字段中逐字核验。若试图逆转后处理，还必须指出输入中真实存在的具体rule及其失败前提。模型自报置信度和自报步骤数不再参与写回门槛。

V2默认使用 `evidence_audit` 策略、`doubao-seed-2.0-lite` 和只审不改模式。只有显式添加 `--allow-writeback` 才可能写回；在审计收益经过验证前不要启用。旧 `blind_review` 策略只用于历史实验回放。

Pipeline 会在调整前保存完整 `difficulty_rating_before_verification` 快照。每行还记录由输入文件、审计 Prompt、策略、模型、温度和安全模式共同生成的 `run_signature`；断点续跑检测到签名不一致时会拒绝混写，要求更换输出文件。

先离线查看候选数量，不调用模型：

```bash
python src/physics_difficulty_agent_pipeline.py \
  -i outputs/model_runs/lite_physics_erroraudit_guard_1066_run1.jsonl \
  -o outputs/model_runs/lite_physics_evidence_audit_run1.jsonl \
  -e outputs/model_runs/lite_physics_evidence_audit_run1_errors.jsonl \
  --dry-run
```

正式证据审计默认只审不改：

```bash
python src/physics_difficulty_agent_pipeline.py \
  -i outputs/model_runs/lite_physics_erroraudit_guard_1066_run1.jsonl \
  -o outputs/model_runs/lite_physics_evidence_audit_run1.jsonl \
  -e outputs/model_runs/lite_physics_evidence_audit_run1_errors.jsonl \
  -p prompts/初中物理难度证据审计提示词.txt \
  --strategy evidence_audit \
  --model doubao-seed-2.0-lite \
  --temperature 1 \
  -c 30
```

输出会保存 `evidence_audit`、可核验反证、`would_apply` 和门控原因，但 `verification_applied` 始终为 `false`，最终等级与冻结输入完全一致。评测报告中的 `verification_agent.audit_comparison` 会比较风险题上的冻结等级与审计建议等级；重点查看 `recommended_level_evaluation` 以及 `audit_better`、`before_better`。只有审计建议在分歧题上稳定优于冻结结果，才考虑单独测试显式写回模式。

## 代码约束

`src/physics_difficulty_rating_with_cache.py` 保持以下兼容性：

- Responses API、缓存、并发、重试、断点续跑和 JSONL 输入输出；
- 五档 `difficulty_level` 字符串；
- 18 个 `features` 字段及其合法枚举；
- `coarse_difficulty` 和四个 `reasoning` 字段。

历史完整实现冻结在 `src/legacy/`；默认 `v7_stable` 先调用该实现，再应用主脚本中的少量结构稳定规则。正式入口是 `prompts/初中物理难度打标提示词.txt`，归档 Prompt 保持原样，仅用于对照。
