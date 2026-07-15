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

当前生产默认使用 `prompts/初中物理难度打标提示词.txt`。它直接以已验证效果最好的历史 Prompt 骨架为基线，归档文件保持不变，只用于兼容对照。

- 明确要求完整合法 JSON，而非仅输出等级名称。
- 模板中每个 feature 仅使用一个合法示例值；完整示例可由 `json.loads()` 解析。
- 保留 18 个 feature 和原有 `step_count` 五个枚举。
- 增加相邻档复核、独立/递进多问、作图、多空、项目/自动控制的边界说明。
- 删除正式 Prompt 中的历史版本名，把重复边界条款合并为 6 组版本无关的最终规则。
- 保留两层样例：五档描述下的典型代表题负责建立档位原型；输入前的 10 个相邻档 few-shot 专门处理边界。
- 将原有 26 个重复边界样例整理为 10 个互补校准示例，并增加 feature 稳定性自检；不读取 JSONL 原 `difficulty`。
- 明确完整受力图、凸透镜完整光线、平面镜成像、螺线管和电路连接通常至少基础；跨不同知识点的多个空通常至少基础。

### few-shot 选择表

| question_id | 教师标签 | 题目摘要 | 选择原因 / 校准边界 |
| --- | --- | --- | --- |
| 3659088519895203840 | 容易 | 不可再生能源直接识别 | 现实背景只是包装，单概念保持送分 |
| 3659087202128773120 | 较易 | 菜刀刀刃变薄增大压强 | 简单生活映射至少基础；与完整规范作图共校准送分/基础边界 |
| 3650594178713079808 | 较易 | 油条制作的四个独立概念选项 | 独立知识点不因数量或表面跨模块升中等 |
| 3659089298106368000 | 中等 | 纸飞机连续运动过程辨析 | 同一过程的受力、惯性和运动状态联合判断达到中等 |
| 3659089396571848704 | 中等 | 光的反射标准实验 | 标准流程与多组归纳为中等，无开放反推不升拔高 |
| 3650594805423308800 | 中等 | 压敏电阻压力秤 | 常规电路、图像和量程不因关键词升拔高 |
| 3659088585494118400 | 较难 | 玻璃管汲水顺序 | 公式少但隐含过程顺序与压强建模达到拔高 |
| 3659087340205260800 | 较难 | 螺栓异常点与新猜想 | 异常点反推、误差方向和新解释区别于标准测量 |
| 3659089430361161728 | 困难 | 天平改装液体密度测量仪 | 方案、函数、边界覆盖和可行性验证共同触发压轴 |
| 3135959210371682304 | 困难 | 登月服气密性检测 | 非项目式多对象、多过程、多图像耦合压轴 |

## 后处理规则

1. 所有动作最多调整一档，并写入 `postprocess_actions`。
2. 送分边界使用“题型语义 + 结构约束”：纯文字常识量估测可保持送分，借照片比例估测至少基础；生活规律映射和规范电磁/电路作图至少基础。
3. 基础升中等仅保留三类稳定结构：多组实验归纳/故障分析、真实递进推理、包含往返/轨道等连续模型的跨模块材料题。普通跨学科装置的独立判断和直接计算不升档。
4. 中等降基础仅处理两个以内独立直接小问及单一规范作图/接线；中等升拔高必须有稳定高阶语义和至少 2—3 个相互印证的核心信号。压力秤、双挡电热器、显性控制链和标准实验有中等保护。
5. 拔高升压轴分成独立规则：项目题必须同时满足设计、至少两个范围/边界/可行性证据和至少三个模型支撑；非项目题 9 步以上仍至少需要五项深耦合证据，6—8 步至少六项。普通误差评价、玻璃管过程顺序不升压轴。
6. 压轴降拔高与拔高升压轴互不取反；原判压轴不会只因“不满足主动升级条件”自动降档。
7. Lite 模型的 `temperature` 固定为 `1`；脚本对 Lite 忽略其他传入值。Mini 等其他模型仍保留 `TEMPERATURE` 配置。
8. 若模型偶发把 `difficulty_level` 放进 `reasoning`，读取时恢复为原始等级参与后处理和顶层审计，但 `difficulty_rating_raw` 仍保留模型原始 JSON，不被改写。
9. 新增结构稳定规则：动态杠杆不依赖 state_count 升拔高；简单电源极性判断不因实验 feature 漂移升中等；单状态量程端点计算保持中等；给定完整步骤的封闭测量材料题不因一个“方案设计或误差评价”值升拔高。
10. 标准测量实验保护设有高阶反例出口：异常点、新猜想、缺表法、等效替代、开放设计、边界验证，以及附着/带出液体后的误差抵消推理不会被压成中等。

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

### 稳定版与历史实验对照

默认配置为 `v7_stable`：使用正式 Prompt，先调用冻结兼容语义层，再执行少量结构稳定规则。`v7_compat` 是未加稳定规则的历史原样对照；`fused` 和 `generalized` 仅保留用于历史实验。外围 API、缓存、并发、重试、Lite 温度固定为 1、`difficulty` 隔离和审计字段均由当前主脚本负责。

133 题结果：

| 路径 | 完全一致 | MAE | 严重偏差 |
| --- | ---: | ---: | ---: |
| 现有三次在线保存结果 | 124 / 125 / 122（平均 92.98%） | — | 0 / 0 / 0 |
| 同三组 `difficulty_rating_raw` + 当前结构稳定规则回放 | 129 / 127 / 125（平均 95.49%） | — | 0 / 0 / 0 |
| 当前回放多数投票 | 130 / 133（97.74%） | — | 0 |

必须使用 `difficulty_rating_raw` 回放后处理，不能把已经处理过的 `difficulty_rating` 再次作为原始结果。当前规则相对三份已保存结果分别净改对 5、2、3 题，改错 0 题；四类“原始等级一致但后处理随 feature 漂移”的题已固定为一致结果。正式 Prompt 尚未在线运行，不能用上述后处理回放数字冒充新 Prompt 的最终准确率。

复跑命令：

```bash
RATING_PROFILE=v7_stable MODEL_NAME=doubao-seed-2.0-lite TEMPERATURE=1 \
python src/physics_difficulty_rating_with_cache.py \
  -i data/labeled/physics_difficulty_tiku_data_v2.jsonl \
  -o outputs/model_runs/lite_physics_v2_v7_stable_run1.jsonl \
  -e outputs/model_runs/lite_physics_v2_v7_stable_run1_errors.jsonl \
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

V7 稳定版专项测试覆盖常规双挡电热器、规范接线、雷达往返材料题、独立直接计算、多个标准实验、简单浮沉状态变化、显性多状态控制、电磁作图、跨模块并列选项和双状态转换开关；全套离线测试仍覆盖融合、通用与 V7 原样兼容路径。

固定生成五档各 40 题的 200 题样本（样本中没有教师标签和旧 `difficulty`）：

```bash
python tests/teacher_label_regression.py \
  --write-stratified outputs/model_runs/teacher_0714_stratified_200.jsonl \
  --per-label 40 --seed 20260714
```

服务器上先用 133 题连续三跑确认格式与波动，再用这 200 题检查五档分层。通过后可对 1066 题完整集运行，或使用同一 seed 固定抽取 1000 题连续三跑：

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
