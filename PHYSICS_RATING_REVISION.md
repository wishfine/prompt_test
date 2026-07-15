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
- 将旧 V7 的 26 个重复边界示例整理为 9 个互补校准示例；第一版压到约 15 KB 后出现明显低档偏置，因此修订版恢复到约 35 KB，保留完整教师口径、五档检核、特征判定和低计算高建模锚点，同时仍删除版本补丁式重复和题目专属关键词。
- 采用 9 个 CSV—JSONL 匹配的真实题目 few-shot，不读取 JSONL 原 `difficulty`。

### few-shot 选择表

| question_id | 教师标签 | 题目摘要 | 选择原因 / 校准边界 |
| --- | --- | --- | --- |
| 3659087291802992640 | 容易 | 凸透镜教材特殊光线 | 教材原型作图仍可送分 |
| 3650594574098083840 | 较易 | 按电路图补全实物连接 | 规范作图/接线不因步骤措辞漂移升中等 |
| 3665219116960403456 | 较易 | 太阳能热水器两个独立直接计算问 | 独立小问不机械累计升中等 |
| 3665219082024722432 | 中等 | 雷达料位器材料题 | 往返模型、液面高度和压强解释形成真实连续分析 |
| 3673096986324992000 | 中等 | 四个标准实验分析 | 多实验综合负担不因“相互独立”压成基础 |
| 3650594805423308800 | 中等 | 压敏电阻压力秤 | 常规电路、图像和量程不因单个高阶词升拔高 |
| 3659088702582308864 | 较难 | 天坛回音壁路径 | 低计算但空间路径高建模的拔高 |
| 3135959210371682304 | 困难 | 登月服气密性检测 | 非项目式多对象、多过程压轴 |
| 3659089430361161728 | 困难 | 天平改装液体密度测量仪 | 设计、量程边界覆盖和可行性验证共同触发压轴 |

## 后处理规则

1. 所有动作最多调整一档，并写入 `postprocess_actions`。
2. 送分边界使用“题型语义 + 结构约束”：纯文字常识量估测可保持送分，借照片比例估测至少基础；生活规律映射和规范电磁/电路作图至少基础。
3. 基础升中等仅保留三类稳定结构：多组实验归纳/故障分析、真实递进推理、包含往返/轨道等连续模型的跨模块材料题。普通跨学科装置的独立判断和直接计算不升档。
4. 中等降基础仅处理两个以内独立直接小问及单一规范作图/接线；中等升拔高必须有稳定高阶语义和至少 2—3 个相互印证的核心信号。压力秤、双挡电热器、显性控制链和标准实验有中等保护。
5. 拔高升压轴分成独立规则：项目题必须同时满足设计、至少两个范围/边界/可行性证据和至少三个模型支撑；非项目题 9 步以上仍至少需要五项深耦合证据，6—8 步至少六项。普通误差评价、玻璃管过程顺序不升压轴。
6. 压轴降拔高与拔高升压轴互不取反；原判压轴不会只因“不满足主动升级条件”自动降档。
7. Lite 模型的 `temperature` 固定为 `1`；脚本对 Lite 忽略其他传入值。Mini 等其他模型仍保留 `TEMPERATURE` 配置。

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

### 最终融合版与历史 V7 基线

默认配置为 `fused`。`generalized` 和冻结的 `v7_compat` 仅用于历史对照；兼容模式仍使用归档 Prompt 与旧语义层，外围 API、缓存、并发、重试、Lite 温度固定为 1、`difficulty` 隔离和审计字段均由当前主脚本负责。

133 题结果：

| 路径 | 完全一致 | MAE | 严重偏差 |
| --- | ---: | ---: | ---: |
| V7 Prompt 三次在线结果 + 旧兼容后处理 | 114 / 123 / 122（平均 89.97%） | — | — |
| 过度压缩融合 Prompt 三次在线原始结果 | 89 / 89 / 87（平均 66.42%） | — | 1 / 2 / 2 |
| 过度压缩融合 Prompt + 当时后处理 | 102 / 101 / 101（平均 76.19%） | — | 2 / 2 / 2 |
| 同三组 `difficulty_rating_raw` + 修订后处理 | 106 / 106 / 104（平均 79.20%） | — | 2 / 1 / 2 |

必须使用 `difficulty_rating_raw` 回放后处理，不能把已经过旧规则处理的 `difficulty_rating` 再次作为原始结果。此前 `122/128/128` 的数字属于错误的叠加回放，现已废弃并更正。修订后处理在三组真正原始结果上分别改对 19、18、18 次，改错 0 次；原始等级波动 20 道降为最终波动 14 道，“原始等级不变但后处理造成波动”为 0。新约 35 KB Prompt 尚未在线运行，不能用后处理回放数字冒充新 Prompt 的最终准确率。

复跑命令：

```bash
RATING_PROFILE=fused MODEL_NAME=doubao-seed-2.0-lite TEMPERATURE=1 \
python src/physics_difficulty_rating_with_cache.py \
  -i data/labeled/physics_difficulty_tiku_data_v2.jsonl \
  -o outputs/model_runs/lite_physics_v2_fused_balanced_run1.jsonl \
  -e outputs/model_runs/lite_physics_v2_fused_balanced_run1_errors.jsonl \
  -p prompts/初中物理难度打标提示词.txt \
  -c 30 --no-cache
```

```bash
source venv/bin/activate
python -m py_compile src/physics_difficulty_rating_with_cache.py \
  tests/test_physics_postprocess.py tests/teacher_label_regression.py
python -m unittest discover -s tests -v
python tests/teacher_label_regression.py
```

融合版专项测试覆盖教材原型、生活应用、照片估测、标准作图/接线、独立计算、多实验综合、雷达材料、压力秤、显性控制链、餐盘隐含杠杆、玻璃管操作顺序、空调安全设计、控温项目、表达式与结构调整、项目边界验证及非项目压轴；全套离线测试还覆盖通用与 V7 兼容路径。

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
