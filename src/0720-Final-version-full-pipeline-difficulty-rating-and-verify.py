# -*- coding: utf-8 -*-
"""
@File    : full-pipeline-difficulty-rating-and-verify.py
@Description:
    融合【难度粗标+特征提取】与【大模型验证】为单一完整打标脚本。

    流程（每道题顺序执行两步）：
      第一步：调用大模型提取结构化特征 + 预估准确率（前缀缓存），并施加
              「乘数效应」后处理；再用新分档策略(≥88/85~87.5/58~84.5/38~57.5/<38)
              由(乘数后)predicted_accuracy 映射 difficulty_level_step1。
      第二步：将第一步结果反馈给大模型做评分核对(难度来源/特征遗漏/评分合理性/
              建议档位)，并按后处理规则得到最终档位：
                - adjusted_difficulty_level 始终只写难度档位(难度1档~难度5档)，
                  不写"无需调整"，保证最终档位只走一套逻辑；
                - 若 rating_reasonableness 为「偏高」或「偏低」，最多调整 1 档
                  (偏高=降1档/偏低=升1档)；模型给出的档位若超过1档则后处理夹紧。
      最终：写出包含所有模型返回字段 + 后处理字段的 jsonl(无 Excel)。

    输入：直接从全量高中数学题目 jsonl 读取(不依赖 Excel 过滤)。
    支持断点续跑：输出 jsonl 中已写入 final_difficulty_level 的 question_id 自动跳过。
"""

import json
import json_repair
import os
import sys
import asyncio
import aiofiles
import aiohttp
import random
import time
import hashlib
import re
from collections import Counter
from typing import Dict, Any, Optional, Tuple
from tqdm.asyncio import tqdm
from asyncio import Lock, Semaphore

# -------------------------- 1. 基础配置 --------------------------
API_KEY = "81dea0da1e2f4bfb9177029a5676e998"
BASE_URL = "https://menshen.test.xdf.cn/v1/"
MODEL_NAME = "doubao-seed-2.0-lite"

JSONL_INPUT_PATH = "/home/share_ssd_data/nfs-data1/wangmeng148/coding/vllm-main/scripts/tiku_difficulty_cls/agent/data-process/output/high-math-all-question-data-format-merge-0616.jsonl"

# 可选：命令行第一个参数为本轮最多处理的题目数(用于分批跑)；不传或非数字=不限量
BATCH_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0

OUTPUT_DIR = "/home/share_ssd_data/nfs-data1/wangmeng148/coding/vllm-main/scripts/tiku_difficulty_cls/high-math-agent/result"
JSONL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "high_math_full_pipeline_rated_verified.jsonl")
ERROR_FILE_PATH = os.path.join(OUTPUT_DIR, "high_math_full_pipeline_errors.jsonl")
CACHE_FILE_PATH = os.path.join(OUTPUT_DIR, "high_math_full_pipeline_prompt_cache.json")

MAX_CONCURRENCY = 200
TIMEOUT = 600
MAX_RETRIES = 4                       # 含「格式解析失败」重试
STEP1_MAX_OUTPUT_TOKENS = 3000        # 第一步：特征JSON+reason(200字)+准确率，原1500易截断，提至3000
STEP2_MAX_OUTPUT_TOKENS = 1200        # 第二步：4字段+分析(500字)，实测平均~219
PARSE_RETRY_BACKOFF = 1.0             # 格式解析失败时的基础退避秒数
FILE_LOCK = Lock()
CACHE_LOCK = Lock()
CACHE_GET_LOCK = Lock()

CACHE_EXPIRE_DAYS = 5
CACHE_EXPIRE_SECONDS = CACHE_EXPIRE_DAYS * 24 * 3600

LEVEL_ORDER = ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"]
LEVEL_INDEX = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}

STRUCTURE_TYPE_MAP = {
    'danxuan': '单选', 'duoxuan': '多选', 'fillblank': '填空', 'zhuguan': '主观',
    'fuhe': '复合', 'panduan': '判断', 'fuhe_zhuguan': '复合主观',
    'fuhe_fillblank': '复合填空', 'fuhe_danxuan': '复合单选', 'completion': '补全',
}

# -------------------------- 2. 第一步特征提取+评级提示词 --------------------------
FEATURE_EXTRACTION_PROMPT_PREFIX = r"""你是一位资深高中数学教研专家，擅长从题目文本中提取结构化的关键特征信息，并且可以基于这些题目特征信息，给出当本题在高考中出现时的预期准确率。请根据给定的题目（题干+解析），严格按照下方定义的特征体系和可选项，以JSON格式提取题目的关键信息，并且给出本题的预期正确率。

================================================================================
## 特征体系与可选项定义

### 一、学科领域

- `domains`（list[string]）：涉及的学科领域列表，每项可选项：`代数`、`几何`、`三角`、`向量`、`概率统计`

判断标准：
- 代数：涉及方程、不等式、函数、代数式、指数、对数、幂、数列、集合、逻辑、复数、排列组合等
- 几何：涉及三角形、四边形、圆、椭圆、双曲线、抛物线、圆锥曲线、平行/垂直/对称、面积/体积、三视图、线面关系等
- 三角：涉及sin/cos/tan、正弦/余弦/正切、诱导公式、和差角/倍角公式、正弦/余弦定理、解三角形等
- 向量：涉及向量运算、数量积、点乘、基底等
- 概率统计：涉及概率、统计、频率、期望、方差、分布、随机变量等

### 二、题型分类

- `question_types`（list[object]）：题型分类列表，每项包含`main_type`和`sub_types`

6大题型及子类（一道题可同时属于多个题型）：

1. **计算求解类** → 直接计算、解方程、解不等式、求导运算
2. **判断选择类** → 真假判断、充要条件判断、大小比较
3. **证明推理类** → 直接证明、存在性证明、反证法证明
4. **最值优化类** → 求最值、求取值范围、恒成立问题
5. **应用建模类** → 实际应用、利润成本、物理模型
6. **图形分析类** → 图像识别、几何作图、三视图

### 三、知识点体系（3级）

- `knowledge_L1`（list[string]）：涉及的一级大模块
- `knowledge_L2`（list[string]）：涉及的二级子模块（格式："大模块>子模块"）
- `knowledge_L3`（list[string]）：涉及的三级细分知识点（格式："大模块>子模块>细分知识点"）

一级大模块（13个）：集合与常用逻辑用语、函数概念与基本初等函数、导数及其应用、三角函数、数列、不等式、立体几何、平面解析几何、平面向量、概率与统计、计数原理、复数、算法初步与框图

二级子模块与三级细分知识点完整列表：

1. 集合与常用逻辑用语
   - 集合的概念与运算 → 集合的表示、交集运算、并集运算、补集运算、子集与包含、集合的综合运算
   - 命题与逻辑 → 充分条件与必要条件、命题否定、全称与存在量词、逆否命题

2. 函数概念与基本初等函数
   - 函数的概念 → 定义域、值域、函数解析式、分段函数
   - 函数的性质 → 单调性、奇偶性、周期性、对称性
   - 基本初等函数 → 一次函数、二次函数、指数函数、对数函数、幂函数
   - 函数零点与方程根 → 零点存在性、二分法、函数零点分布

3. 导数及其应用
   - 导数的概念与运算 → 导数定义、求导公式、复合函数求导
   - 导数的应用 → 单调性分析、极值求解、最值求解、切线方程、恒成立问题、不等式证明

4. 三角函数
   - 三角函数的定义 → 任意角三角函数、同角三角函数关系、诱导公式
   - 三角恒等变换 → 和差角公式、倍角公式、半角公式、辅助角公式
   - 三角函数的图象与性质 → 图象变换、周期与振幅
   - 解三角形 → 正弦定理、余弦定理、面积公式

5. 数列
   - 等差数列 → 通项公式、前n项和、性质
   - 等比数列 → 通项公式、前n项和、性质
   - 数列综合 → 递推关系、求和方法、数列与不等式、数列与函数

6. 不等式
   - 一元二次不等式 → 解一元二次不等式、三个二次关系
   - 基本不等式 → 均值不等式、不等式链
   - 绝对值不等式 → 绝对值三角不等式、解绝对值不等式
   - 线性规划 → 简单线性规划

7. 立体几何
   - 空间几何体 → 棱柱、棱锥、球、圆柱与圆锥、正方体与长方体、三视图与直观图
   - 点线面位置关系 → 线线关系、线面关系、面面关系、二面角
   - 空间向量与立体几何 → 空间向量运算、空间向量证平行、空间向量证垂直、空间角与距离

8. 平面解析几何
   - 直线与方程 → 直线方程、斜率与倾斜角、两条直线位置关系、距离公式
   - 圆与方程 → 圆的方程、直线与圆的位置关系、圆与圆的位置关系
   - 圆锥曲线 → 椭圆、双曲线、抛物线、离心率、焦点与准线、渐近线、直线与圆锥曲线、圆锥曲线综合

9. 平面向量
   - 向量的线性运算 → 向量加减、数乘运算、共线向量
   - 向量的数量积 → 数量积运算、向量垂直、夹角问题
   - 向量的应用 → 向量在几何中、向量在代数中

10. 概率与统计
    - 概率 → 古典概型、条件概率、独立事件、n次独立重复试验
    - 统计 → 抽样方法、频率分布、样本估计总体
    - 随机变量及其分布 → 离散型随机变量、二项分布、正态分布、超几何分布
    - 统计案例 → 回归分析、独立性检验

11. 计数原理
    - 排列与组合 → 排列、组合、排列组合综合
    - 二项式定理 → 二项展开式、通项公式

12. 复数
    - 复数的运算 → 复数代数形式、复数乘除、复数模运算
    - 复数的几何意义 → 复平面

13. 算法初步与框图
    - 算法与程序框图 → 程序框图、基本算法语句

### 四、认知特征体系

5.1 分类讨论
- `classification_discussion.needed`（boolean）：是否需要分类讨论
- `classification_discussion.scale`（string）：规模，可选项：`2类讨论`、`3类讨论`、`4类及以上`、`不适用`
- `classification_discussion.trigger`（string）：触发原因，可选项：`绝对值触发`、`参数范围触发`、`图形位置触发`、`零点存在性触发`、`其他`、`不适用`

5.2 参数讨论
- `parameter_discussion.has_parameter`（boolean）：是否含参数
- `parameter_discussion.count`（string）：参数个数，可选项：`单参数`、`双参数`、`多参数(3+)`、`不适用`
- `parameter_discussion.type`（string）：讨论类型，可选项：`恒成立问题`、`能成立问题`、`求参数范围`、`参数与零点`、`不适用`

5.3 辅助线
- `auxiliary_line.needed`（boolean）：是否需要辅助线
- `auxiliary_line.type`（string）：辅助线类型，可选项：`连接线段`、`作垂线`、`作平行线`、`延长线`、`取中点/特殊点`、`构造辅助图形`、`不适用`

5.4 转化与化归
- `transformation.needed`（boolean）：是否需要转化与化归
- `transformation.type`（string）：转化类型，可选项：`换元法`、`数形结合`、`整体代换`、`分离参数`、`等价转化`、`逆向转化`、`不适用`

5.5 构造法
- `construction.needed`（boolean）：是否需要构造法
- `construction.type`（string）：构造类型，可选项：`构造函数`、`构造方程`、`构造不等式`、`构造数列`、`构造几何图形`、`构造向量`、`不适用`

5.6 逆向推理
- `reverse_reasoning.needed`（boolean）：是否需要逆向推理
- `reverse_reasoning.type`（string）：逆向推理类型，可选项：`反证法`、`分析法(执果索因)`、`逆推法`、`不适用`

5.7 多解问题
- `multiple_solutions.has_multiple`（boolean）：是否有多解
- `multiple_solutions.source`（string）：多解来源，可选项：`绝对值多解`、`开方多解`、`图形多解`、`参数多解`、`不适用`

5.8 数形结合
- `number_shape.needed`（boolean）：是否需要数形结合
- `number_shape.type`（string）：数形结合类型，可选项：`以形助数`、`以数解形`、`函数图像分析`、`不适用`
- `number_shape.direction`（string）：数形转换方向，可选项：`数→形单向`、`数⇆形双向`、`不适用`
  - 数→形单向：仅将代数问题转化为图形辅助理解，单向"读懂"即可
  - 数⇆形双向：需在数与形之间反复"翻译"，既要从图形提取代数信息，又要将代数结果反馈到图形
- `number_shape.graph_type`（string）：涉及的图形类型，可选项：`函数图象`、`几何图形`、`动态图形`、`不适用`
  - 函数图象：函数的图象（如二次函数、指数函数图象等）
  - 几何图形：静态的几何图形（如三角形、圆、圆锥曲线等）
  - 动态图形：含动点、变换的动态图形（需结合动态过程分析）
- `number_shape.abstraction_level`（string）：形→数的抽象层级（从图形中提取信息后需要的数学化步骤数），可选项：`1步直接代入`、`2-3步推理`、`多步推理`、`不适用`
  - 1步直接代入：从图形可直接读出坐标、面积等代入计算
  - 2-3步推理：需对图形信息做少量转化后才能用于计算
  - 多步推理：需对图形信息做多步数学化推理（如建立方程组、不等式等）才能求解

5.9 隐含条件
- `hidden_conditions.has_hidden`（boolean）：题干是否含需要学生自行发现的隐含条件
  - 判断标准：隐含条件指题目未直接给出、但解题时必须主动发掘的限制条件或性质，这类条件是导致"看似简单实则难"的重要原因
- `hidden_conditions.types`（list[string]）：隐含条件类型列表，可选项：`定义域/值域限制`、`隐含几何关系`、`隐含对称性`、`参数隐含范围`、`隐含单调性/奇偶性`、`其他隐含条件`、`不适用`
  - 定义域/值域限制：如对数函数真数>0、偶次根式被开方数≥0等需主动注意的定义域
  - 隐含几何关系：如图形中隐含的平行、垂直、共线、切点等关系
  - 隐含对称性：如函数隐含的轴对称/中心对称性质
  - 参数隐含范围：如由实际意义决定的参数取值范围（人数为正整数等）
  - 隐含单调性/奇偶性：如由函数结构可推断但未明说的单调性或奇偶性
  - 当`has_hidden`为false时，填`["不适用"]`

5.10 数学思想方法
- `math_thought.function_equation`（boolean）：是否体现函数与方程思想
- `math_thought.classification`（boolean）：是否体现分类讨论思想
- `math_thought.number_shape`（boolean）：是否体现数形结合思想
- `math_thought.transformation`（boolean）：是否体现转化与化归思想

### 五、解析与题干复杂度特征

6.1 解析独立步骤数
- `step_count`（string）：基于解析内容判断的独立步骤数，若题目包含多问，则取步骤数最多的小问，可选项：`1-2步`、`3-5步`、`6-10步`、`11-15步`、`15步以上`

6.2 动态几何元素
- `dynamic_geometry`（string）：题干中涉及的动态几何元素，可选项：`无动态`、`单一动点`、`多动点+变换`（多动点+折叠/旋转/平移等变换）

6.3 题干属性
- `new_definition`（string）：题干是否含新定义或抽象探究背景，可选项：`无新定义`、`含新定义`、`抽象文字探究背景`
  - 无新定义：常规题干，使用标准数学概念和表述
  - 含新定义：题目中给出新的定义、符号或运算规则，需要先理解新定义再解题
  - 抽象文字探究背景：题目以抽象的文字描述或探究性背景呈现，理解题干本身就有较大难度

6.4 现实生活问题
- `reality_question`（boolean）：是否为现实生活问题
  - 判断标准：题目中涉及的实际情况会影响学生对问题理解的难度。若题干中的现实背景不影响问题理解难度（如"2025年某地访问人数共计867500000人，用科学记数法表示"），则不是现实生活问题

### 六、解题方法

- `solving_methods`（list[string]）：使用的解题方法列表，可选项：`配方法`、`换元法`、`待定系数法`、`定义法`、`数学归纳法`、`反证法`、`整体代换`、`分离参数`

### 七、准确率区间含义

根据大量真实考试数据统计，各难度等级对应的典型准确率范围如下：

| 典型准确率范围 | 说明 |
|--------------|------|
| 80%-99% | 属于高考中最为简单的题目，直接套用公式或概念即可，绝大多数学生能做对 |
| 60%-80% | 高考中负责考查基础知识了解的题目，需要1-2步推理或简单综合，大部分学生能做对 |
| 40%-60% | 高考中的中等难度题目，普遍需要多步推理或分类讨论，约一半学生能做对 |
| 20%-40% | 高考中的倒数第二道大题或倒数第二道选择题这个难度，仅少部分学生可以完整打出，需要构造法或考查知识点的深度综合 |
| 1%-20% | 高考中的最后一道大题级别，需要非常规方法突破，极少数学生能做对 |

## 输出要求

请严格按照以下JSON格式输出，不要添加任何额外内容：

```json
{
  "features": {
    "domains": ["代数", "几何"],
    "question_types": [
      {"main_type": "最值优化类", "sub_types": ["求最值"]}
    ],
    "knowledge_L1": ["导数及其应用"],
    "knowledge_L2": ["导数及其应用>导数的应用"],
    "knowledge_L3": ["导数及其应用>导数的应用>最值求解"],
    "classification_discussion": {
      "needed": true,
      "scale": "2类讨论",
      "trigger": "参数范围触发"
    },
    "parameter_discussion": {
      "has_parameter": true,
      "count": "单参数",
      "type": "求参数范围"
    },
    "auxiliary_line": {
      "needed": false,
      "type": "不适用"
    },
    "transformation": {
      "needed": true,
      "type": "分离参数"
    },
    "construction": {
      "needed": false,
      "type": "不适用"
    },
    "reverse_reasoning": {
      "needed": false,
      "type": "不适用"
    },
    "multiple_solutions": {
      "has_multiple": false,
      "source": "不适用"
    },
    "number_shape": {
      "needed": true,
      "type": "函数图像分析",
      "direction": "数⇆形双向",
      "graph_type": "函数图象",
      "abstraction_level": "2-3步推理"
    },
    "hidden_conditions": {
      "has_hidden": true,
      "types": ["参数隐含范围"]
    },
    "math_thought": {
      "function_equation": true,
      "classification": true,
      "number_shape": true,
      "transformation": true
    },
    "step_count": "3-5步",
    "dynamic_geometry": "无动态",
    "new_definition": "无新定义",
    "reality_question": false,
    "solving_methods": ["分离参数"]
  },
  "reason": "先基于所抽取的上述特征判断题目正确率应该处于哪个区间，再进一步定位区间内的具体准确率（限定在200字以内）",
  "predicted_accuracy": 70.0
}
```

注意：
- `features`中各字段必须严格取对应可选项之一
- 认知特征中，当主判断为false时，子选项填"不适用"
- `number_shape`中，当`needed`为false时，`direction`、`graph_type`、`abstraction_level`均填"不适用"
- `hidden_conditions`中，当`has_hidden`为false时，`types`填`["不适用"]`
- 知识点必须从上方定义的三级知识点列表中选取，不可自行编造
- knowledge_L2中的子模块必须属于knowledge_L1中的某个大模块；knowledge_L3中的细分知识点必须属于knowledge_L2中的某个子模块
- `predicted_accuracy`字段为浮点数，表示预测的本题出现在高考时的准确率百分比（0-100），保留一位小数
- 预测准确率时需综合考虑上述细粒度特征以及题目可能包含的其它特殊情况，例如涉及到超纲知识点、特殊的计算方法等；不同特征对准确率的影响权重不同，需根据实际情况进行调整
- 特别注意：隐含条件（hidden_conditions）和数形结合的抽象层级（number_shape.abstraction_level）对题目难度影响显著，含隐含条件或需多步推理的数形转换题目，学生实际作答正确率往往低于表面判断，预估准确率时应适当下调

================================================================================
## 输入题目信息"""

FEATURE_EXTRACTION_PROMPT_SUFFIX = """\n\n请根据以上信息，提取题目的结构化特征信息，并给出预估准确率。"""


# -------------------------- 3. 缓存管理(第一步前缀缓存) --------------------------
def compute_text_hash(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


async def load_cache() -> Dict[str, Any]:
    async with CACHE_LOCK:
        if not os.path.exists(CACHE_FILE_PATH):
            return {}
        try:
            async with aiofiles.open(CACHE_FILE_PATH, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content) if content else {}
        except Exception as e:
            print(f"加载缓存文件失败: {e}")
            return {}


async def save_cache(cache_data: Dict[str, Any]) -> None:
    async with CACHE_LOCK:
        try:
            cache_dir = os.path.dirname(CACHE_FILE_PATH)
            if cache_dir and not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)
            async with aiofiles.open(CACHE_FILE_PATH, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(cache_data, ensure_ascii=False, indent=2))
        except Exception as e:
            print(f"保存缓存文件失败: {e}")


def is_cache_valid(cache_entry: Dict[str, Any], current_time: int) -> bool:
    if not cache_entry:
        return False
    expire_at = cache_entry.get('expire_at', 0)
    if current_time >= expire_at:
        return False
    expected_hash = compute_text_hash(FEATURE_EXTRACTION_PROMPT_PREFIX)
    actual_hash = cache_entry.get('prefix_hash', '')
    return expected_hash == actual_hash


async def get_valid_cache() -> Optional[Dict[str, Any]]:
    cache_data = await load_cache()
    current_time = int(time.time())
    cache_entry = cache_data.get('prompt_prefix_cache')
    if is_cache_valid(cache_entry, current_time):
        return cache_entry
    return None


async def set_cache(response_id: str, expire_at: int) -> None:
    cache_data = await load_cache()
    cache_entry = {
        'response_id': response_id,
        'expire_at': expire_at,
        'prefix_hash': compute_text_hash(FEATURE_EXTRACTION_PROMPT_PREFIX),
        'created_at': int(time.time())
    }
    cache_data['prompt_prefix_cache'] = cache_entry
    await save_cache(cache_data)


async def get_or_create_cache(session: aiohttp.ClientSession) -> Optional[str]:
    async with CACHE_GET_LOCK:
        cache_entry = await get_valid_cache()
        if cache_entry:
            return cache_entry['response_id']
        print("未找到有效缓存，创建新的前缀缓存...")
        response_id = await create_prefix_cache(session)
        return response_id


# -------------------------- 4. 通用工具函数 --------------------------
def parse_model_response(response_text: str) -> Dict:
    if not response_text:
        return {}
    try:
        return json_repair.loads(response_text)
    except Exception:
        pass
    try:
        clean_text = response_text
        if "```json" in clean_text:
            # 处理被截断的代码块：只有开头```json无结尾```时，split只得到1段
            parts = clean_text.split("```json")
            if len(parts) >= 2:
                tail = parts[1]
                if "```" in tail:
                    clean_text = tail.split("```")[0]
                else:
                    clean_text = tail          # 输出被max_output_tokens截断，无结尾标记
        elif "```" in clean_text:
            parts = clean_text.split("```")
            if len(parts) >= 3:
                clean_text = parts[1]
            elif len(parts) == 2:
                clean_text = parts[1]            # 同上：截断导致只有开头标记
        return json_repair.loads(clean_text.strip())
    except Exception:
        pass
    try:
        s = response_text.find("{")
        e = response_text.rfind("}")
        if s != -1 and e != -1:
            return json_repair.loads(response_text[s:e + 1])
    except Exception:
        pass
    # 最后一道兜底：输出被截断导致没有闭合`}`时，取首个`{`到末尾交给json_repair修复
    try:
        s = response_text.find("{")
        if s != -1:
            return json_repair.loads(response_text[s:])
    except Exception:
        pass
    return {}


def _step1_valid(parsed: Dict) -> bool:
    """第一步结果是否有效：必须为dict，且能取到数值型 predicted_accuracy。"""
    if not isinstance(parsed, dict) or not parsed:
        return False
    # features 必须为 dict，否则下游 rating.get('features') 会抛 'str' object has no attribute 'get'
    feats = parsed.get('features')
    if feats is not None and not isinstance(feats, dict):
        return False
    pa = parsed.get('predicted_accuracy')
    if pa is None:
        return False
    try:
        float(pa)
        return True
    except (ValueError, TypeError):
        return False


def _step2_valid(parsed: Dict) -> bool:
    """第二步结果是否有效：必须为dict，且含非空字符串 analysis 字段。"""
    if not isinstance(parsed, dict) or not parsed:
        return False
    analysis = parsed.get('analysis')
    return isinstance(analysis, str) and bool(analysis.strip())


def construct_question_content(data: Dict[str, Any]) -> str:
    """组装题目文本(题型结构/题干/选项/解析/小题)。无解析时显式标注。"""
    parts = []
    stem = (data.get('stem') or '').strip()
    options = (data.get('options') or '').strip()
    analysis = (data.get('analysis') or '').strip()
    structure_type = data.get('structure_type', '')
    st_label = STRUCTURE_TYPE_MAP.get(structure_type, structure_type)
    if st_label:
        parts.append(f"【题型结构】{st_label}")
    if stem:
        parts.append(f"【题干】\n{stem}")
    if options:
        parts.append(f"【选项】\n{options}")
    if analysis:
        parts.append(f"【解析】\n{analysis}")
    else:
        parts.append("【解析】(本题无解析)")
    sub_questions = data.get('sub_questions', [])
    if sub_questions:
        try:
            sub_questions.sort(key=lambda x: int(x.get('question_id', 0)))
        except (ValueError, TypeError):
            pass
        parts.append("【小题】")
        for i, sq in enumerate(sub_questions, 1):
            sq_stem = (sq.get('stem') or '').strip()
            sq_options = (sq.get('options') or '').strip()
            sq_analysis = (sq.get('analysis') or '').strip()
            parts.append(f"  小题{i}:")
            if sq_stem:
                parts.append(f"    题干: {sq_stem}")
            if sq_options:
                parts.append(f"    选项: {sq_options}")
            if sq_analysis:
                parts.append(f"    解析: {sq_analysis}")
    return "\n\n".join(parts)


# -------------------------- 5. 第一步：特征提取+评级(前缀缓存) --------------------------
async def create_prefix_cache(session: aiohttp.ClientSession) -> Optional[str]:
    current_time = int(time.time())
    expire_at = current_time + CACHE_EXPIRE_SECONDS
    payload = {
        "model": MODEL_NAME,
        "input": [{"role": "user", "content": FEATURE_EXTRACTION_PROMPT_PREFIX}],
        "thinking": {"type": "disabled"},
        "expire_at": expire_at,
        "caching": {"type": "enabled", "prefix": True}
    }
    t1 = time.time()
    for attempt in range(MAX_RETRIES):
        try:
            async with session.post(
                f"{BASE_URL}responses", json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=TIMEOUT)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"创建前缀缓存失败 (状态码: {response.status}): {error_text[:200]}")
                    if 400 <= response.status < 500:
                        return None
                    await asyncio.sleep(2 ** attempt)
                    continue
                result = await response.json()
                response_id = result.get('id')
                if response_id:
                    await set_cache(response_id, expire_at)
                    print(f"前缀缓存创建成功，耗时: {time.time() - t1:.2f}秒，缓存ID: {response_id}")
                    return response_id
        except Exception as e:
            backoff = (2 ** attempt) + random.uniform(0, 1)
            if attempt == MAX_RETRIES - 1:
                print(f"创建前缀缓存最终失败: {e}")
                return None
            print(f"创建前缀缓存失败，{backoff:.2f}秒后重试: {e}")
            await asyncio.sleep(backoff)
    return None


async def call_model_with_cache(question_content: str, session: aiohttp.ClientSession) -> Tuple[Dict, float, int, int, int]:
    """第一步：特征提取+准确率预估(复用前缀缓存)。返回 (parsed, time_use, pt, ct, tt)。"""
    response_id = await get_or_create_cache(session)
    if not response_id:
        print("获取/创建缓存失败，无法继续第一步")
        return {}, 0.0, 0, 0, 0
    dynamic_content = f"{question_content}{FEATURE_EXTRACTION_PROMPT_SUFFIX}"
    for retry in range(MAX_RETRIES):
        payload = {
            "model": MODEL_NAME,
            "previous_response_id": response_id,
            "input": [{"role": "user", "content": dynamic_content}],
            "thinking": {"type": "disabled"},
            "max_output_tokens": STEP1_MAX_OUTPUT_TOKENS,
        }
        t1 = time.time()
        try:
            async with session.post(
                f"{BASE_URL}responses", json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=TIMEOUT)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    output_text = ""
                    if 'output' in result:
                        for item in result['output']:
                            if item.get('type') == 'message' and 'content' in item:
                                for content_item in item['content']:
                                    if content_item.get('type') == 'output_text':
                                        output_text = content_item.get('text', '')
                    usage = result.get('usage', {}) or {}
                    pt = usage.get('input_tokens', 0) or 0
                    ct = usage.get('output_tokens', 0) or 0
                    tt = usage.get('total_tokens', 0) or 0
                    cached_tokens = (usage.get('input_tokens_details', {}) or {}).get('cached_tokens', 0) or 0
                    if cached_tokens > 0:
                        print(f"  缓存命中，节省token: {cached_tokens}")
                    parsed_result = parse_model_response(output_text)
                    if _step1_valid(parsed_result):
                        return parsed_result, time.time() - t1, pt, ct, tt
                    # 格式解析失败 → 重试
                    print(f"  第一步返回格式无效(第{retry+1}/{MAX_RETRIES})，重试。原始输出前150字: {output_text[:150]!r}")
                    await asyncio.sleep(PARSE_RETRY_BACKOFF + random.uniform(0, 1))
                    continue
                elif response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    print(f"  触发限流(429)，{retry_after}秒后重试 (第{retry+1}/{MAX_RETRIES}次)")
                    await asyncio.sleep(retry_after)
                    continue
                else:
                    error_text = await response.text()
                    print(f"  第一步API失败 (状态码 {response.status}): {error_text[:200]}")
                    if "InvalidParameter.PreviousResponseNotFound" in error_text:
                        print("  缓存失效，重新创建缓存...")
                        new_response_id = await get_or_create_cache(session)
                        if not new_response_id:
                            return {}, 0.0, 0, 0, 0
                        response_id = new_response_id
                        continue
                    if response.status >= 500:
                        await asyncio.sleep((2 ** retry) + random.uniform(0, 1))
                        continue
                    if 400 <= response.status < 500:
                        return {}, 0.0, 0, 0, 0
        except aiohttp.ClientError as e:
            if retry == MAX_RETRIES - 1:
                print(f"  第一步网络异常最终失败: {e}")
                return {}, 0.0, 0, 0, 0
            await asyncio.sleep((2 ** retry) + random.uniform(0, 1))
        except Exception as e:
            if retry == MAX_RETRIES - 1:
                print(f"  第一步请求最终失败: {e}")
                return {}, 0.0, 0, 0, 0
            print("  尝试重新创建缓存...")
            new_response_id = await get_or_create_cache(session)
            if not new_response_id:
                return {}, 0.0, 0, 0, 0
            response_id = new_response_id
    print(f"  第一步达到最大重试次数({MAX_RETRIES})，放弃本次请求")
    return {}, 0.0, 0, 0, 0


# -------------------------- 6. 第一步后处理：乘数效应 --------------------------
def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ('true', '是', 'yes', '1')
    return bool(value)


def _detect_high_difficulty_features(features: Dict[str, Any]) -> Dict[str, bool]:
    """检测7项高难度特征是否出现。"""
    classification = _to_bool((features.get('classification_discussion') or {}).get('needed', False))
    construction = _to_bool((features.get('construction') or {}).get('needed', False))
    transformation = _to_bool((features.get('transformation') or {}).get('needed', False))
    ns = features.get('number_shape') or {}
    ns_needed = _to_bool(ns.get('needed', False))
    number_shape_high = ns_needed and (ns.get('direction', '') == '数⇆形双向' or ns.get('abstraction_level', '') == '多步推理')
    reverse_reasoning = _to_bool((features.get('reverse_reasoning') or {}).get('needed', False))
    aux = features.get('auxiliary_line') or {}
    auxiliary_high = _to_bool(aux.get('needed', False)) and aux.get('type', '') == '构造辅助图形'
    hidden_conditions = _to_bool((features.get('hidden_conditions') or {}).get('has_hidden', False))
    return {
        'classification': classification,
        'construction': construction,
        'transformation': transformation,
        'number_shape_high': number_shape_high,
        'reverse_reasoning': reverse_reasoning,
        'auxiliary_high': auxiliary_high,
        'hidden_conditions': hidden_conditions,
    }


def apply_multiplier_effect(rating_result: Dict[str, Any]) -> Dict[str, Any]:
    """对模型预测准确率施加乘数效应后处理。"""
    if not rating_result:
        return rating_result
    features = rating_result.get('features', {})
    if not features:
        return rating_result
    feat = _detect_high_difficulty_features(features)
    trigger_combos = [
        ['classification', 'construction', 'transformation'],
        ['classification', 'number_shape_high', 'reverse_reasoning'],
        ['auxiliary_high', 'transformation', 'hidden_conditions'],
        ['construction', 'reverse_reasoning', 'transformation'],
    ]
    triggered = any(all(feat[k] for k in combo) for combo in trigger_combos)
    if not triggered:
        return rating_result
    feature_count = sum(1 for v in feat.values() if v)
    original_acc = rating_result.get('predicted_accuracy')
    if original_acc is None:
        return rating_result
    try:
        base_acc = float(original_acc)
    except (ValueError, TypeError):
        return rating_result
    if feature_count >= 4:
        multiplier = 0.7
    elif feature_count >= 3:
        multiplier = 0.85
    else:
        return rating_result
    adjusted_acc = round(base_acc * multiplier, 1)
    adjusted_acc = max(0.0, min(100.0, adjusted_acc))
    rating_result['original_predicted_accuracy'] = base_acc
    rating_result['predicted_accuracy'] = adjusted_acc
    rating_result['multiplier_applied'] = multiplier
    rating_result['multiplier_feature_count'] = feature_count
    return rating_result


# -------------------------- 7. 活跃特征计数 + 新档位映射 --------------------------
GATES = [
    ('classification_discussion', 'needed'),
    ('parameter_discussion', 'has_parameter'),
    ('auxiliary_line', 'needed'),
    ('transformation', 'needed'),
    ('construction', 'needed'),
    ('reverse_reasoning', 'needed'),
    ('multiple_solutions', 'has_multiple'),
    ('number_shape', 'needed'),
    ('hidden_conditions', 'has_hidden'),
]
MATH_THOUGHT_KEYS = ['function_equation', 'classification', 'number_shape', 'transformation']


def count_active_features(features: Dict[str, Any]) -> int:
    """活跃特征数量：9 门控认知特征 + 4 数学思想 + 新定义 + 现实问题 + 动态几何。"""
    if not features:
        return 0
    n = 0
    for g, f in GATES:
        if _to_bool((features.get(g) or {}).get(f)):
            n += 1
    mt = features.get('math_thought') or {}
    for k in MATH_THOUGHT_KEYS:
        if _to_bool(mt.get(k)):
            n += 1
    if features.get('new_definition') in ('含新定义', '抽象文字探究背景'):
        n += 1
    if _to_bool(features.get('reality_question')):
        n += 1
    dg = features.get('dynamic_geometry', '')
    if dg and dg != '无动态':
        n += 1
    return n


def map_difficulty_level(acc) -> str:
    """新分档策略：≥88→1档；85~87.5→2档；58~84.5→3档；38~57.5→4档；<38→5档。"""
    try:
        acc = float(acc)
    except (ValueError, TypeError):
        return ""
    if acc >= 88:
        return "难度1档"
    elif acc >= 85:
        return "难度2档"
    elif acc >= 58:
        return "难度3档"
    elif acc >= 38:
        return "难度4档"
    else:
        return "难度5档"


# -------------------------- 8. 第二步：验证提示词 --------------------------
def build_verification_prompt(data: Dict[str, Any], rating: Dict[str, Any],
                             feature_count: int, level: str) -> str:
    question_content = construct_question_content(data)
    pred_acc = rating.get('predicted_accuracy')
    features_json = json.dumps(rating.get('features', {}), ensure_ascii=False, indent=2)
    has_analysis = bool((data.get('analysis') or '').strip())
    prompt = f"""你是资深高中数学教研专家。下面是一道题目，以及难度评估系统已为其提取的结构化特征与难度评估结果。

【关键背景】
该题被判定为【{level}】(预测准确率={pred_acc}%)，活跃特征数量={feature_count} 个。
{'⚠ 本题在题库中【无解析】，提取出的特征与评分是模型仅凭题干推断的，分析时请特别关注这一点。' if not has_analysis else ''}

请基于题目全部信息(题干/选项/解析/小题)以及已提取的特征，深入分析本题的难度评估。

请从以下角度逐项分析：
1. 难度来源判定：本题的难度主要来自什么(计算量/抽象理解/隐含条件/知识点深度综合/超纲内容/新定义背景/复杂分类/认知复杂度特征组合/其他)？这些难度来源是否被当前特征体系充分覆盖？若未被覆盖请明确指出。
2. 特征提取是否合理：已提取的特征是否准确？是否存在本应提取却漏掉的特征(请给出具体特征名，如"应为隐含条件但未提取")？特征数量(={feature_count})与本题难度是否匹配？
3. 评分合理性核对：当前预测准确率={pred_acc}% → 档位【{level}】。结合本题实际难度判断该评分是否合理。注意：偏高指当前档位难度被高估(实际应更简单/档位应更低)，偏低指难度被低估(实际应更难/档位应更高)。

请严格按照以下JSON格式输出，不要添加任何额外内容：
{{
  "difficulty_source": "本题难度的主要来源(200字内)",
  "missed_features": ["本应提取却漏掉的特征名列表，无则填[\\"无\\"]"],
  "rating_reasonableness": "合理" 或 "偏高" 或 "偏低",
  "adjusted_difficulty_level": "始终填写难度档位(难度1档~难度5档之一)：评分【合理】时填写当前档位【{level}】；偏高时填写降档后的建议档位；偏低时填写升档后的建议档位。不得填写\\"无需调整\\"等非档位文本",
  "analysis": "详细分析(500字内)"
}}

================================================================================
## 题目信息

{question_content}

================================================================================
## 已提取的结构化特征(JSON)

{features_json}

================================================================================
## 难度评估信息

predicted_accuracy: {pred_acc}%
difficulty_level: {level}
活跃特征数量: {feature_count}
"""
    return prompt


# -------------------------- 9. 第二步：模型调用(无缓存) --------------------------
async def call_model_plain(prompt: str, session: aiohttp.ClientSession) -> Tuple[Dict, float, int, int, int]:
    """第二步：验证调用。返回 (parsed, time_use, pt, ct, tt)。"""
    payload = {
        "model": MODEL_NAME,
        "input": [{"role": "user", "content": prompt}],
        "thinking": {"type": "disabled"},
        "max_output_tokens": STEP2_MAX_OUTPUT_TOKENS,
    }
    t1 = time.time()
    for retry in range(MAX_RETRIES):
        try:
            async with session.post(
                f"{BASE_URL}responses", json=payload,
                headers={"Authorization": f"Bearer {API_KEY}"},
                timeout=aiohttp.ClientTimeout(total=TIMEOUT),
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    output_text = ""
                    for item in result.get('output', []):
                        if item.get('type') == 'message' and 'content' in item:
                            for ci in item['content']:
                                if ci.get('type') == 'output_text':
                                    output_text = ci.get('text', '')
                    usage = result.get('usage', {}) or {}
                    pt = usage.get('input_tokens', 0) or 0
                    ct = usage.get('output_tokens', 0) or 0
                    tt = usage.get('total_tokens', 0) or 0
                    parsed = parse_model_response(output_text)
                    if _step2_valid(parsed):
                        return parsed, time.time() - t1, pt, ct, tt
                    # 格式解析失败 → 重试
                    print(f"  第二步返回格式无效(第{retry+1}/{MAX_RETRIES})，重试。原始输出前150字: {output_text[:150]!r}")
                    await asyncio.sleep(PARSE_RETRY_BACKOFF + random.uniform(0, 1))
                    continue
                elif response.status == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    print(f"  第二步限流(429)，{retry_after}秒后重试 (第{retry+1}/{MAX_RETRIES}次)")
                    await asyncio.sleep(retry_after)
                    continue
                else:
                    error_text = await response.text()
                    print(f"  第二步API失败 (状态码 {response.status}): {error_text[:200]}")
                    if response.status >= 500:
                        await asyncio.sleep((2 ** retry) + random.uniform(0, 1))
                        continue
                    return {}, 0.0, 0, 0, 0
        except aiohttp.ClientError as e:
            if retry == MAX_RETRIES - 1:
                print(f"  第二步网络异常最终失败: {e}")
                return {}, 0.0, 0, 0, 0
            await asyncio.sleep((2 ** retry) + random.uniform(0, 1))
        except Exception as e:
            if retry == MAX_RETRIES - 1:
                print(f"  第二步请求最终失败: {e}")
                return {}, 0.0, 0, 0, 0
            await asyncio.sleep((2 ** retry) + random.uniform(0, 1))
    return {}, 0.0, 0, 0, 0


# -------------------------- 10. 第二步后处理：最多调整1档 --------------------------
def normalize_model_level(raw) -> str:
    """从模型输出中规整出 难度1档~难度5档 之一，无法识别返回 ''。"""
    if not raw:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)
    if raw in LEVEL_INDEX:
        return raw
    m = re.search(r"难度[1-5]档", raw)
    return m.group(0) if m else ""


def clamp_adjusted_level(current_level: str, reasonableness: str, model_adj_raw) -> Tuple[str, str]:
    """第二步后处理：最多调整1档。

    - 合理 → 不调整，取当前档位。
    - 偏高(难度高估，应更简单) → 向下降1档(不低于难度1档)。
    - 偏低(难度低估，应更难) → 向上升1档(不高于难度5档)。
    - 参考模型给出的建议档位：若其方向与 reasonableness 一致，则在 ±1 档内取模型值；
      若方向矛盾或超过1档，则按 ±1 档夹紧。
    返回 (final_level, adjustment_desc)。
    """
    cur = LEVEL_INDEX.get(current_level)
    if cur is None:
        return current_level, "当前档位无效·未调整"
    rs = (reasonableness or "").strip()
    if rs == "合理":
        return current_level, "合理·未调整"
    if rs not in ("偏高", "偏低"):
        return current_level, f"合理性未知({rs})·未调整"

    if rs == "偏高":          # 应更简单 → 降档
        direction, dir_text = -1, "降1档"
    else:                    # 偏低 → 升档
        direction, dir_text = +1, "升1档"

    target = max(0, min(4, cur + direction))
    model_level = normalize_model_level(model_adj_raw)
    if model_level and model_level != current_level:
        m = LEVEL_INDEX[model_level]
        if direction > 0 and m > cur:        # 模型也建议升，取较小者(≤1档)
            target = min(m, cur + 1)
        elif direction < 0 and m < cur:      # 模型也建议降，取较大者(≥-1档)
            target = max(m, cur - 1)
        # 模型方向矛盾 → 忽略模型，沿用 ±1
    target = max(0, min(4, target))
    final = LEVEL_ORDER[target]
    if final == current_level:
        return final, f"{rs}·已到边界·维持{final}"
    return final, f"{rs}·{current_level}→{final}({dir_text})"


# -------------------------- 11. 单题处理(两步合一) --------------------------
async def process_one_question(data: Dict[str, Any], session: aiohttp.ClientSession,
                               semaphore: Semaphore, pbar: tqdm) -> None:
    async with semaphore:
        question_id = data.get('question_id', 'unknown')
        try:
            question_content = construct_question_content(data)

            # ===== 第一步：特征提取 + 准确率预估 =====
            s1_parsed, s1_time, s1_pt, s1_ct, s1_tt = await call_model_with_cache(question_content, session)
            if not s1_parsed or s1_parsed.get('predicted_accuracy') is None:
                err = data.copy()
                err['question_id'] = question_id
                err['pipeline_error'] = "第一步失败：API返回为空或predicted_accuracy缺失"
                async with FILE_LOCK:
                    async with aiofiles.open(ERROR_FILE_PATH, "a", encoding="utf-8") as f:
                        await f.write(json.dumps(err, ensure_ascii=False) + "\n")
                print(f"【{question_id}】第一步失败，已记录错误")
                pbar.update(1)
                return

            # 第一步后处理：乘数效应
            rating = s1_parsed
            rating = apply_multiplier_effect(rating)
            pred_acc = rating.get('predicted_accuracy')
            level_step1 = map_difficulty_level(pred_acc)
            features = rating.get('features') or {}
            feature_count = count_active_features(features)
            has_analysis = bool((data.get('analysis') or '').strip())

            # ===== 第二步：验证核对 =====
            v_prompt = build_verification_prompt(data, rating, feature_count, level_step1)
            s2_parsed, s2_time, s2_pt, s2_ct, s2_tt = await call_model_plain(v_prompt, session)
            if not s2_parsed or not s2_parsed.get('analysis'):
                # 第二步失败：保留第一步结果，但记录错误以便续跑重试第二步
                err = data.copy()
                err['question_id'] = question_id
                err['step1_rating'] = rating
                err['step1_difficulty_level'] = level_step1
                err['pipeline_error'] = "第二步失败：分析结果为空或analysis字段缺失"
                async with FILE_LOCK:
                    async with aiofiles.open(ERROR_FILE_PATH, "a", encoding="utf-8") as f:
                        await f.write(json.dumps(err, ensure_ascii=False) + "\n")
                print(f"【{question_id}】第二步失败(第一步 acc={pred_acc}% {level_step1})，已记录错误待重试")
                pbar.update(1)
                return

            # 第二步后处理：最多调整1档 → 最终档位
            reasonableness = s2_parsed.get('rating_reasonableness', '')
            model_adj = s2_parsed.get('adjusted_difficulty_level', '')
            final_level, adj_desc = clamp_adjusted_level(level_step1, reasonableness, model_adj)

            output_data = data.copy()
            output_data['difficulty_rating'] = rating          # 含 features/reason/predicted_accuracy(+乘数信息)
            output_data['difficulty_level_step1'] = level_step1
            output_data['feature_count'] = feature_count
            output_data['has_analysis'] = has_analysis
            output_data['verification'] = {
                'difficulty_source': s2_parsed.get('difficulty_source', ''),
                'missed_features': s2_parsed.get('missed_features', []),
                'rating_reasonableness': reasonableness,
                'adjusted_difficulty_level': model_adj,         # 模型原始建议(规整前)
                'adjusted_difficulty_level_normalized': normalize_model_level(model_adj),
                'analysis': s2_parsed.get('analysis', ''),
            }
            output_data['final_difficulty_level'] = final_level
            output_data['adjustment_desc'] = adj_desc
            output_data['step1_api_time_use'] = round(s1_time, 2)
            output_data['step1_api_prompt_tokens'] = s1_pt
            output_data['step1_api_completion_tokens'] = s1_ct
            output_data['step1_api_total_tokens'] = s1_tt
            output_data['step2_api_time_use'] = round(s2_time, 2)
            output_data['step2_api_prompt_tokens'] = s2_pt
            output_data['step2_api_completion_tokens'] = s2_ct
            output_data['step2_api_total_tokens'] = s2_tt

            async with FILE_LOCK:
                async with aiofiles.open(JSONL_OUTPUT_PATH, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(output_data, ensure_ascii=False) + "\n")
            mult = rating.get('multiplier_applied')
            mult_str = f" ×{mult}" if mult else ""
            print(f"【{question_id}】acc={pred_acc}%{mult_str} | step1={level_step1} | 合理性={reasonableness} | 最终={final_level} ({adj_desc})")
        except Exception as e:
            print(f"【{question_id}】处理异常: {e}")
            import traceback
            traceback.print_exc()
            err = data.copy()
            err['question_id'] = question_id
            err['pipeline_error'] = str(e)
            async with FILE_LOCK:
                async with aiofiles.open(ERROR_FILE_PATH, "a", encoding="utf-8") as f:
                    await f.write(json.dumps(err, ensure_ascii=False) + "\n")
        pbar.update(1)


# -------------------------- 12. 断点续跑 + 主流程 --------------------------
def get_processed_ids(output_path: str) -> set:
    ids = set()
    if not os.path.exists(output_path):
        return ids
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if 'question_id' in d and 'final_difficulty_level' in d:
                    ids.add(d['question_id'])
            except Exception:
                continue
    return ids


async def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("融合打标脚本：第一步 难度粗标+特征提取 → 第二步 大模型验证")
    print("=" * 60)
    print(f"输入: {JSONL_INPUT_PATH}")
    print(f"输出JSONL: {JSONL_OUTPUT_PATH}")
    print(f"错误文件: {ERROR_FILE_PATH}")
    print(f"并发: {MAX_CONCURRENCY}\n")

    if not os.path.exists(JSONL_INPUT_PATH):
        print(f"错误: 输入文件不存在 {JSONL_INPUT_PATH}")
        return

    # 1. 读取全部题目
    all_data = []
    seen = set()
    with open(JSONL_INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            qid = record.get('question_id')
            if qid is None or qid in seen:
                continue
            seen.add(qid)
            all_data.append(record)
    print(f"输入文件去重后共 {len(all_data)} 条题目记录")

    # 2. 断点续跑
    processed = get_processed_ids(JSONL_OUTPUT_PATH)
    todo = [d for d in all_data if d.get('question_id') not in processed]
    print(f"已处理(两步完成) {len(processed)} 题，本次待处理 {len(todo)} 题")

    if not todo:
        print("全部已处理完成。")
        return

    random.shuffle(todo)
    print("待处理数据已随机打乱\n")

    if BATCH_LIMIT > 0 and len(todo) > BATCH_LIMIT:
        print(f"本轮限量处理：从 {len(todo)} 题中取前 {BATCH_LIMIT} 题\n")
        todo = todo[:BATCH_LIMIT]

    semaphore = Semaphore(MAX_CONCURRENCY)
    pbar = tqdm(total=len(todo), unit="item", desc="融合打标进度")
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENCY * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        print("检查并创建第一步前缀缓存...")
        await get_or_create_cache(session)
        tasks = [process_one_question(d, session, semaphore, pbar) for d in todo]
        await asyncio.gather(*tasks)
    pbar.close()
    print(f"\n完成! 结果: {JSONL_OUTPUT_PATH}")


if __name__ == "__main__":
    start = time.time()
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    except Exception as e:
        print(f"\n程序发生未捕获异常: {e}")
    print(f"本次运行耗时: {round((time.time() - start) / 60, 2)} 分钟")
