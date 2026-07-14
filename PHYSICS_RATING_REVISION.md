# 物理难度评级：教师标签基准修订说明

## 数据口径与匹配报告

教师 CSV `data/labeled/physics_teacher_labels_0714.csv` 是唯一真值；映射只用于离线评测：`容易/较易/中等/较难/困难` 对应 `送分题/基础题/中等题/拔高题/压轴题`。JSONL 中原有 `difficulty` 不参与模型输入、few-shot 选择、规则或评估。

| 项目 | 结果 |
| --- | ---: |
| CSV 有效标签 | 1091 |
| JSONL 题目 | 1066 |
| 成功匹配 | 1066 |
| CSV 未匹配 | 25 |
| JSONL 未匹配 | 0 |
| 容易 / 较易 / 中等 / 较难 / 困难 | 147 / 345 / 386 / 135 / 53 |

CSV 的 `ID` 和 JSONL 的 `question_id` 均在脚本中按字符串处理。原始数据不被重写；批量输出会将输入 `difficulty` 改名为 `source_difficulty_untrusted`。

## Prompt 修订

- 明确要求完整合法 JSON，而非仅输出等级名称。
- 模板中每个 feature 仅使用一个合法示例值；完整示例可由 `json.loads()` 解析。
- 保留 18 个 feature 和原有 `step_count` 五个枚举。
- 增加相邻档复核、独立/递进多问、作图、多空、项目/自动控制的边界说明。
- 采用 9 个 CSV—JSONL 匹配的真实题目 few-shot，不读取 JSONL 原 `difficulty`。

### few-shot 选择表

| question_id | 教师标签 | 题目摘要 | 选择原因 / 校准边界 |
| --- | --- | --- | --- |
| 2141171782236581888 | 容易 | 摩擦力概念辨析 | 单知识点直接识别的送分下界 |
| 3659087202128773120 | 较易 | 刀刃做薄增大压强 | 生活情境映射仍是基础 |
| 2896407737920417792 | 较易 | 温度自动报警器 | 继电器关键词不自动升档 |
| 3650594805423308800 | 中等 | 压敏电阻压力秤 | 常规电路、图像、量程的中等上界 |
| 3659088567173398528 | 中等 | 电热水壶控制电路 | 显性多开关逻辑为中等 |
| 3659088585494118400 | 较难 | 玻璃管汲水顺序 | 低计算但高过程推理的拔高 |
| 3660872748428980224 | 较难 | 电阻关系探究与缺表法 | 递进实验设计 / 等效替代的拔高 |
| 3135959210371682304 | 困难 | 登月服气密性检测 | 非项目式多对象、多过程压轴 |
| 3659087386644594688 | 困难 | 自动饮水机 | 项目情境需真实多约束耦合才压轴 |

## 后处理规则

1. 所有动作最多调整一档，并写入 `postprocess_actions`。
2. 送分降档必须同时满足严格低复杂度特征；送分升基础保留公式、规范作图/实验、多个知识点、递进推理等守卫。
3. 基础升中等：至少一个强信号（控制变量/故障、多组归纳、多公式、图像反推、连续过程、递进多问、真实跨模块），或至少两个弱信号；独立题和基础读数排除。
4. 中等升拔高：3-5 步需两个核心高阶信号且至少一个强迁移信号；6-8 步需一个核心高阶信号且不是独立并列题。
5. 拔高升压轴：至少三个核心高阶信号和一个强压轴信号。项目/控制题还必须有可行性/边界/分类/方案比较等强验证、至少两个高阶模型信号，以及复杂计算、临界推理或 6 步以上证据。
6. 压轴降拔高独立判断：1-5 步时核心信号不足 3 且无强压轴信号；6-8 步时核心信号不足 2 且无强信号；9 步以上原则保留。缺少“主动升压轴”的证据不会自动降档。

每条成功输出同时包含：

```json
{
  "source_difficulty_untrusted": 3,
  "difficulty_rating_raw": {},
  "difficulty_level_raw": "中等题",
  "postprocess_actions": [],
  "difficulty_rating": {}
}
```

`difficulty_rating_raw` 是模型解析结果的深拷贝，未被归一化覆盖。

## 验证与回归

```bash
source venv/bin/activate
python -m py_compile src/physics_difficulty_rating_with_cache.py \
  tests/test_physics_postprocess.py tests/teacher_label_regression.py
python -m unittest tests/test_physics_postprocess.py
python tests/teacher_label_regression.py
```

固定生成五档各 40 题的 200 题样本（样本中没有教师标签和旧 `difficulty`）：

```bash
python tests/teacher_label_regression.py \
  --write-stratified outputs/model_runs/teacher_0714_stratified_200.jsonl \
  --per-label 40 --seed 20260714
```

服务器上先跑这 200 题，再按 CSV 评估；不应直接跑完整集：

```bash
MODEL_NAME=doubao-seed-2.0-lite \
python src/physics_difficulty_rating_with_cache.py \
  -i outputs/model_runs/teacher_0714_stratified_200.jsonl \
  -o outputs/model_runs/lite_teacher_0714_200.jsonl \
  -e outputs/model_runs/lite_teacher_0714_200_errors.jsonl \
  -p prompts/初中物理难度打标提示词.txt \
  -c 20 --no-cache

python tests/teacher_label_regression.py \
  --evaluate outputs/model_runs/lite_teacher_0714_200.jsonl
```

评估会输出完全一致率、±1 档比例、MAE、严重偏差、混淆矩阵、每档准确率、系统判高/判低数，以及每条后处理规则的触发、改进、变差、无变化次数；所有评估只读取 CSV 教师标签。
