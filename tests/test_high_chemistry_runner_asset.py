# -*- coding: utf-8 -*-
"""高中化学两阶段 Prompt 与运行脚本静态契约。"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "src" / "high_chemistry_difficulty_rating_and_verify.py"
PROMPT = ROOT / "prompts" / "高中化学难度打标提示词.txt"
sys.path.insert(0, str(ROOT / "src"))

import high_chemistry_pipeline_core as chemistry_core  # noqa: E402


class HighChemistryAssetTests(unittest.TestCase):
    def test_runner_exists_and_compiles(self) -> None:
        self.assertTrue(RUNNER.exists())
        source = RUNNER.read_text(encoding="utf-8")
        compile(source, str(RUNNER), "exec")
        self.assertIn("high_chemistry_two_stage_v5", source)
        self.assertIn("ENABLE_STAGE2_AUTO_ADJUST", source)
        self.assertIn("ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER", source)
        self.assertIn(
            'os.getenv("ENABLE_CHEMISTRY_HIGH_DIFFICULTY_MULTIPLIER", "1")',
            source,
        )
        self.assertIn('os.getenv("ENABLE_STAGE2_AUTO_ADJUST", "1")', source)
        self.assertIn("shared_runner.ENABLE_STAGE2_AUTO_ADJUST =", source)
        self.assertIn("_shared_finalize_level", source)

    def test_prompt_defines_both_stages_and_complete_schema(self) -> None:
        self.assertTrue(PROMPT.exists())
        namespace = {}
        source = PROMPT.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT), "exec"), namespace)
        for name in (
            "FEATURE_EXTRACTION_PROMPT_PREFIX",
            "FEATURE_EXTRACTION_PROMPT_SUFFIX",
            "VERIFICATION_PROMPT_PREFIX",
            "VERIFICATION_PROMPT_SUFFIX",
        ):
            self.assertTrue(namespace.get(name), name)
        prefix = namespace["FEATURE_EXTRACTION_PROMPT_PREFIX"]
        for field in (
            "substance_relation",
            "reaction_relation",
            "competing_reaction",
            "evidence_relation",
            "route_design_requirement",
            "predicted_accuracy",
        ):
            self.assertIn(field, prefix)
        verification = namespace["VERIFICATION_PROMPT_PREFIX"]
        self.assertIn("原始预测正确率", verification)
        self.assertIn("重复计数", verification)
        self.assertIn("相邻边界", verification)

    def test_stage1_does_not_disclose_postprocess_mechanics(self) -> None:
        namespace = {}
        source = PROMPT.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT), "exec"), namespace)
        stage1 = namespace["FEATURE_EXTRACTION_PROMPT_PREFIX"]
        stage2 = namespace["VERIFICATION_PROMPT_PREFIX"]
        for forbidden in (
            "0.85",
            "0.70",
            "乘数",
            "程序会按同一决策节点去重",
            "固定档位边界将由程序",
        ):
            self.assertNotIn(forbidden, stage1)
        self.assertIn("原始预测正确率", stage1)
        self.assertIn("0.85", stage2)
        self.assertIn("0.70", stage2)

    def test_stage1_defines_chemistry_counting_and_conversion_boundaries(self) -> None:
        namespace = {}
        source = PROMPT.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT), "exec"), namespace)
        stage1 = namespace["FEATURE_EXTRACTION_PROMPT_PREFIX"]
        self.assertGreaterEqual(len(stage1), 8000)
        for required in (
            "目标考生定义",
            "知识点计数",
            "物质计数",
            "反应节点计数",
            "宏观现象、微观粒子、化学符号",
            "字段间一致性自检",
        ):
            self.assertIn(required, stage1)

    def test_stage1_defines_taxonomy_mapping_and_experiment_task_boundary(self) -> None:
        namespace = {}
        source = PROMPT.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT), "exec"), namespace)
        stage1 = namespace["FEATURE_EXTRACTION_PROMPT_PREFIX"]
        for required in (
            "knowledge_L2 到 knowledge_L1 的固定映射",
            "原子结构与元素周期律 → 化学基本概念",
            "实验探究与方案设计 → 化学实验",
            "experiment_requirement 描述实验任务深度",
            "不得填写 knowledge_L1 或 knowledge_L2 的模块名称",
        ):
            self.assertIn(required, stage1)

    def test_v2_accuracy_scale_separates_easy_middle_and_hard_structures(self) -> None:
        namespace = {}
        source = PROMPT.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT), "exec"), namespace)
        stage1 = namespace["FEATURE_EXTRACTION_PROMPT_PREFIX"]
        for required in (
            "普通高考考生总体中预计能够完整答对本题的比例",
            "同一回答规则下的直接选项辨析",
            "彼此独立的基础判断",
            "6—8个有效化学决策",
            "进入38—58比较",
            "两个以上高阶阶段",
            "共享同一物质流",
            "答案依赖",
            "模型依赖",
            "轻微易错点不能单独把正确率压到88以下",
            "唯一回答规则",
            "完整作答稳定性",
            "完整完成本题全部必要任务的概率",
        ):
            self.assertIn(required, stage1)
        self.assertIn("与上下相邻边界的距离", stage1)
        self.assertIn("不得把边界值、整数中点或示例值当作模板分数", stage1)
        self.assertNotIn("89、86、68、52、42", stage1)
        self.assertNotIn("80—85", stage1)
        self.assertNotIn("70—80", stage1)
        self.assertNotRegex(stage1, r'"predicted_accuracy"\s*:\s*\d')

    def test_stage1_uses_physics_boundaries_with_sharper_85_and_58_semantics(self) -> None:
        namespace = {}
        source = PROMPT.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT), "exec"), namespace)
        stage1 = namespace["FEATURE_EXTRACTION_PROMPT_PREFIX"]
        for boundary in ("88 分边界", "85 分边界", "58 分边界", "38 分边界"):
            self.assertIn(boundary, stage1)
        for obsolete in ("75 分边界", "55 分边界", "35 分边界"):
            self.assertNotIn(obsolete, stage1)
        for required in (
            "三个及以上异质必要决策",
            "多个必须完整作答的异质输出",
            "不能因各任务相互独立就默认达到85",
            "并同时具有下列至少两类真实结构",
        ):
            self.assertIn(required, stage1)

    def test_stage2_does_not_treat_program_derived_fields_as_structural_revision(self) -> None:
        namespace = {}
        source = PROMPT.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT), "exec"), namespace)
        stage2 = namespace["VERIFICATION_PROMPT_PREFIX"]
        self.assertIn(
            "knowledge_L1、knowledge_count、knowledge_scope 由程序确定性派生",
            stage2,
        )
        self.assertIn("不得写入 feature_corrections", stage2)
        for required in (
            "程序接受的非派生 feature 修正",
            "程序确认的58边界结构候选",
            "confidence=高",
            "复核后正确率确实跨越相邻边界",
            "verdict 与程序计算方向一致",
            "最多调整一档",
            "58—62",
            "多层因果+高层信息转换",
            "58—68",
            "多层因果+一次常规表征转换",
            "不得自动从难度2档降入难度1档",
            "先命中特征组合，再按高难特征数量选择乘数",
            "普通4档不得仅因乘数跌入5档",
        ):
            self.assertIn(required, stage2)

    def test_v2_stage2_audits_both_sides_of_middle_band(self) -> None:
        namespace = {}
        source = PROMPT.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT), "exec"), namespace)
        stage2 = namespace["VERIFICATION_PROMPT_PREFIX"]
        for required in (
            "第一步档位为难度3档",
            "同时检查85边界和58边界",
            "为什么不能达到85及以上",
            "为什么不应低于58",
            "不得只修改文字表述",
            "没有得到程序支持的真实结构修正",
            "必须原样保持第一阶段 original_predicted_accuracy",
        ):
            self.assertIn(required, stage2)

    def test_stage1_output_contract_relies_on_complete_feature_enums(self) -> None:
        namespace = {}
        source = PROMPT.read_text(encoding="utf-8")
        exec(compile(source, str(PROMPT), "exec"), namespace)
        stage1_text = namespace["FEATURE_EXTRACTION_PROMPT_PREFIX"]
        self.assertIn("顶层字段必须且只能包括", stage1_text)
        self.assertIn("features、reason 和 predicted_accuracy", stage1_text)
        for field in chemistry_core.REQUIRED_FEATURE_FIELDS:
            self.assertIn(field, stage1_text)

    def test_runner_reuses_operational_capabilities_without_sending_label(self) -> None:
        source = RUNNER.read_text(encoding="utf-8")
        self.assertIn("high_physics_difficulty_rating_and_verify", source)
        self.assertIn("prepare_question", source)
        self.assertNotIn("source_difficulty_untrusted\"]", source)


if __name__ == "__main__":
    unittest.main()
