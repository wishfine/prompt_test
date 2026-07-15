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
└── model_runs/                      模型实验结果，不作为输入数据

prompts/                             当前模型提示词
└── archive/                         已验证历史 Prompt（兼容回放）
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
# 默认使用去冗余融合版；历史基线可显式切换。
RATING_PROFILE=fused
```

Mini 等支持调温度的模型仍会读取 `TEMPERATURE`；未配置时不发送该字段。

真实模型调用需要在服务器的 venv 中执行，本机只适合做静态检查。

## 评级配置

- `fused`：最终融合版，默认值。使用压缩后的 9 个边界示例 Prompt 和单一、可审计的稳定后处理。
- `generalized`：上一版通用后处理，仅保留用于对照。
- `v7_compat`：调用冻结的旧 V7 边界语义层；配套使用归档 Prompt，可复现历史高准确率基线。API、缓存、并发、重试及审计输出仍全部走当前主脚本。

融合后处理在三次 V7 Prompt 在线原始结果上离线回放，完全一致率分别达到 `122/133`、`128/133`、`128/133`，平均 `94.74%`；三次多数票为 `127/133（95.49%）`。原始等级波动题由 18 道降为最终等级波动 15 道，并且“原始等级稳定、后处理造成波动”从旧规则的 5 道降为 0 道。这里是后处理离线回放结果；最终融合 Prompt 仍应在服务器连续在线跑三次确认。

## 评测命令

运行最终融合版（Lite 温度固定为 1）：

```bash
RATING_PROFILE=fused MODEL_NAME=doubao-seed-2.0-lite TEMPERATURE=1 \
python src/physics_difficulty_rating_with_cache.py \
  -i data/labeled/physics_difficulty_tiku_data_v2.jsonl \
  -o outputs/model_runs/lite_physics_v2_fused_run1.jsonl \
  -e outputs/model_runs/lite_physics_v2_fused_run1_errors.jsonl \
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

## 代码约束

`src/physics_difficulty_rating_with_cache.py` 保持以下兼容性：

- Responses API、缓存、并发、重试、断点续跑和 JSONL 输入输出；
- 五档 `difficulty_level` 字符串；
- 18 个 `features` 字段及其合法枚举；
- `coarse_difficulty` 和四个 `reasoning` 字段。

最终规则统一维护在物理 Prompt 和主脚本的 `fused` 路径中；旧 V7 完整实现只冻结在 `src/legacy/`，由 `v7_compat` 显式调用，不参与默认路径。
