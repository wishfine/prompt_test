from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "src" / "chemistry_difficulty_rating_with_cache.py"
PROMPT_PATH = ROOT / "prompts" / "初中化学难度打标提示词.txt"


def load_runtime():
    spec = importlib.util.spec_from_file_location(
        "chemistry_observable_runtime_integration",
        SCRIPT_PATH,
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def observable_features() -> dict:
    return {
        "longest_solution_chain": [
            "从图像平台读取生成气体质量",
            "根据化学方程式建立反应物质量关系",
            "利用质量守恒求反应后溶液质量",
            "计算反应后溶质质量分数",
        ],
        "task_groups": [
            {"task_type": "图表与数据", "count": 1},
            {"task_type": "定量计算", "count": 2},
        ],
        "rule_families": [
            "图表读取或数据归纳",
            "反应关系或条件判断",
            "定量关系与计算",
        ],
        "curriculum_units": ["U5", "U9"],
        "reaction_structure": "单一反应",
        "condition_operations": [],
        "representation_operations": [
            "图表数据→化学关系",
            "化学方程式→定量关系",
        ],
        "evidence_operations": ["多证据共同成立"],
        "experiment_operation": "无",
        "graph_table_operation": "拐点平台或分段",
        "calculation_operations": ["单一方程式", "单一守恒"],
        "new_information_operation": "无新信息",
    }


def observable_v3_features() -> dict:
    features = observable_features()
    features.pop("curriculum_units")
    features.update(
        {
            "curriculum_topics": ["U5-1", "U9-3"],
            "parallel_task_relation": "共享同一化学模型的关联任务",
            "visual_task_structure": "共享装置流程或图表模型",
            "error_analysis_operation": "无误差分析",
        }
    )
    return features


def observable_v4_features() -> dict:
    features = observable_v3_features()
    features.update(
        {
            "direct_retrieval_task_count": 1,
            "rule_application_task_count": 2,
            "solution_topology": "单线性常规链",
            "experiment_task_structure": "无实验判断",
        }
    )
    return features


def rating(level: str = "中等题") -> dict:
    coarse = {
        "送分题": "送分/基础区间（1-2档）",
        "基础题": "基础/中等区间（2-3档）",
        "中等题": "基础/中等区间（2-3档）",
        "拔高题": "中等/拔高区间（3-4档）",
        "压轴题": "拔高/压轴区间（4-5档）",
    }[level]
    return {
        "features": observable_features(),
        "coarse_difficulty": coarse,
        "reasoning": {
            "core_basis": "最长必要链为四个化学决策。",
            "hard_point": "图像、方程式与守恒连续衔接。",
            "why_not_lower": "不是单点直接应用。",
            "why_not_higher": "没有多反应分类与多阶段耦合。",
        },
        "difficulty_level": level,
    }


class ChemistryObservableRuntimeIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = load_runtime()

    def test_formal_prompt_uses_observable_v2_output_contract(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        for field in self.runtime.OBSERVABLE_FEATURE_FIELDS:
            self.assertIn(f'"{field}"', prompt)
        for legacy_field in (
            "reasoning_depth",
            "reasoning_direction",
            "constraint_complexity",
            "evidence_relation",
            "unfamiliar_information_transfer",
            "subquestion_dependency",
        ):
            self.assertNotIn(f'"{legacy_field}"', prompt)

    def test_rating_contract_accepts_observable_features(self) -> None:
        validated = self.runtime.validate_rating_contract(rating())
        self.assertEqual(
            set(validated["features"]),
            set(self.runtime.OBSERVABLE_V2_FEATURE_FIELDS),
        )

    def test_rating_contract_accepts_observable_v3_features(self) -> None:
        item = rating()
        item["features"] = observable_v3_features()

        validated = self.runtime.validate_rating_contract(item)

        self.assertEqual(
            set(validated["features"]),
            set(self.runtime.OBSERVABLE_V3_FEATURE_FIELDS),
        )

    def test_rating_contract_accepts_observable_v4_features(self) -> None:
        item = rating()
        item["features"] = observable_v4_features()

        validated = self.runtime.validate_rating_contract(item)

        self.assertEqual(
            set(validated["features"]),
            set(self.runtime.OBSERVABLE_V4_FEATURE_FIELDS),
        )

    def test_v4_does_not_emit_generic_core12_candidate(self) -> None:
        item = rating("基础题")
        item["features"] = observable_v4_features()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "根据反应现象确定反应物",
                    "书写化学方程式建立质量关系",
                    "利用方程式计算生成物质量",
                ],
                "direct_retrieval_task_count": 0,
                "rule_application_task_count": 3,
                "solution_topology": "单线性常规链",
                "calculation_operations": ["单一方程式"],
                "representation_operations": ["化学方程式→定量关系"],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "根据反应现象和方程式完成计算。", "options": ""},
        )

        self.assertEqual(result["feature_schema_version"], "chemistry_observable_v4")
        self.assertEqual(result["postprocess_candidate_actions"], [])

    def test_v4_direct_experiment_match_is_not_breadth_candidate(self) -> None:
        item = rating("送分题")
        item["features"] = observable_v4_features()
        item["features"].update(
            {
                "longest_solution_chain": ["识别量筒名称"],
                "task_groups": [
                    {"task_type": "实验操作与探究", "count": 1},
                ],
                "direct_retrieval_task_count": 1,
                "rule_application_task_count": 0,
                "rule_families": ["实验操作与探究"],
                "representation_operations": [],
                "evidence_operations": [],
                "experiment_operation": "基础操作或读数",
                "experiment_task_structure": "名称或单点规范匹配",
                "visual_task_structure": "单图直接识别",
                "graph_table_operation": "无",
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "下列仪器中，量筒是（ ）。", "options": ""},
        )

        self.assertIsNone(
            result["teacher_distribution_guard_candidate_action"]
        )

    def test_v4_multi_condition_experiment_is_audit_candidate(self) -> None:
        item = rating("送分题")
        item["features"] = observable_v4_features()
        item["features"].update(
            {
                "longest_solution_chain": ["比较三种仪器是否可直接加热"],
                "task_groups": [
                    {"task_type": "实验操作与探究", "count": 3},
                ],
                "direct_retrieval_task_count": 0,
                "rule_application_task_count": 3,
                "rule_families": ["实验操作与探究"],
                "representation_operations": [],
                "evidence_operations": [],
                "experiment_operation": "基础操作或读数",
                "experiment_task_structure": "多仪器或多条件比较",
                "visual_task_structure": "多图独立同规则识别",
                "graph_table_operation": "无",
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "比较三种仪器能否直接加热。", "options": ""},
        )

        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_easy_to_basic_experiment_application",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_v4_shared_new_relation_is_audit_candidate(self) -> None:
        item = rating("中等题")
        item["features"] = observable_v4_features()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "根据题给新关系建立物质组成判断",
                    "将该关系用于后续定量核验",
                ],
                "direct_retrieval_task_count": 0,
                "rule_application_task_count": 3,
                "solution_topology": "未知组成或量反推",
                "parallel_task_relation": "共享同一化学模型的关联任务",
                "new_information_operation": "新关系被多个任务共同使用",
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "根据题给新关系完成组成反推和后续定量核验。"},
        )

        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_medium_to_hard_shared_new_information",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_postprocess_derives_metrics_without_replacing_features(self) -> None:
        original = rating()
        result = self.runtime.postprocess_chemistry_difficulty(
            copy.deepcopy(original),
            {"stem": "根据图像和化学方程式完成计算。", "options": ""},
        )

        self.assertEqual(
            result["feature_schema_version"],
            "chemistry_observable_v2",
        )
        self.assertEqual(
            result["postprocess_profile"],
            "chemistry_observable_v2_teacher_distribution_v2_safe",
        )
        self.assertEqual(result["observable_metrics"]["longest_chain_steps"], 4)
        self.assertEqual(result["observable_metrics"]["effective_task_count"], 3)
        self.assertNotIn("derived_core12_projection", result)
        self.assertEqual(result["features"], original["features"])

    def test_schema_retry_message_does_not_request_legacy_core12(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("上次输出未通过Core-12 schema", source)
        self.assertIn("上次输出未通过化学特征schema", source)

    def test_observable_projection_is_fact_derived(self) -> None:
        projection = self.runtime.project_observable_to_core12(
            observable_features()
        )
        self.assertEqual(projection["reasoning_depth"], "4-5层")
        self.assertEqual(
            projection["graph_table_requirement"],
            "拐点、平台或分段反推",
        )
        self.assertEqual(
            projection["calculation_model"],
            "单一守恒或多反应计算",
        )
        self.assertEqual(
            projection["knowledge_relation"],
            "跨模块融合",
        )

    def test_parallel_cross_unit_coverage_is_not_projected_as_fusion(
        self,
    ) -> None:
        features = observable_v3_features()
        features["curriculum_topics"] = ["U2-2", "U7-1"]
        features["parallel_task_relation"] = "同一规则下多个对象"

        projection = self.runtime.project_observable_to_core12(features)

        self.assertEqual(
            projection["knowledge_relation"],
            "同模块简单关联",
        )
        self.assertEqual(
            projection["subquestion_dependency"],
            "多问相互独立",
        )

    def test_cross_unit_reasoning_and_parallel_chain_are_audited(
        self,
    ) -> None:
        item = rating("基础题")
        item["features"] = observable_v3_features()
        item["features"].update(
            {
                "curriculum_topics": ["U2-2", "U7-1"],
                "parallel_task_relation": "同一规则下多个对象",
                "longest_solution_chain": [
                    "判断选项A",
                    "判断选项B",
                    "判断选项C",
                    "判断选项D",
                ],
            }
        )
        item["reasoning"]["core_basis"] = (
            "覆盖U2-2和U7-1两个同单元相邻课题。"
        )

        result = self.runtime.postprocess_chemistry_difficulty(item, {})

        self.assertTrue(
            any(
                "不同U前缀却写成同单元" in flag
                for flag in result["feature_audit_flags"]
            )
        )
        self.assertTrue(
            any(
                "独立任务疑似按选项累计最长链" in flag
                for flag in result["feature_audit_flags"]
            )
        )

    def test_cross_unit_audit_ignores_negated_same_unit_phrase(
        self,
    ) -> None:
        item = rating("基础题")
        item["features"] = observable_v3_features()
        item["features"].update(
            {
                "curriculum_topics": ["U2-2", "U7-1"],
                "parallel_task_relation": "同一规则下多个对象",
            }
        )
        item["reasoning"]["core_basis"] = (
            "覆盖U2-2和U7-1两个不同单元的并列课题，"
            "不构成共享模型。"
        )

        result = self.runtime.postprocess_chemistry_difficulty(item, {})

        self.assertEqual(
            result["observable_metrics"]["curriculum_span_summary"],
            "跨单元并列（U2-2、U7-1）",
        )
        self.assertFalse(
            any(
                "不同U前缀却写成同单元" in flag
                for flag in result["feature_audit_flags"]
            )
        )

    def test_multi_rule_breadth_is_candidate_only_with_writeback_on(
        self,
    ) -> None:
        item = rating("基础题")
        item["features"] = observable_v3_features()
        item["features"].update(
            {
                "longest_solution_chain": ["分别完成各类规则应用"],
                "task_groups": [
                    {"task_type": "直接事实与概念", "count": 2},
                    {"task_type": "化学用语", "count": 2},
                ],
                "rule_families": ["直接事实与概念", "化学用语"],
                "curriculum_topics": ["U3-2", "U4-3"],
                "parallel_task_relation": "不同规则的独立任务",
                "reaction_structure": "无反应任务",
                "representation_operations": [],
                "evidence_operations": ["单证据直接匹配"],
                "calculation_operations": [],
            }
        )
        old_enabled = self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS
        old_writeback = (
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
        )
        try:
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = True
            result = self.runtime.postprocess_chemistry_difficulty(item, {})
        finally:
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = old_enabled
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = (
                old_writeback
            )

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(
            result["teacher_distribution_guard_candidate_level"],
            "中等题",
        )
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_basic_to_medium_multi_rule_breadth_candidate",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_observable_chain_drives_basic_to_medium_candidate(self) -> None:
        item = rating("基础题")
        item["features"]["longest_solution_chain"] = [
            "根据现象确定参加反应的物质",
            "书写化学方程式建立质量关系",
            "根据方程式计算生成物质量",
        ]
        item["features"]["graph_table_operation"] = "直接读数"
        item["features"]["representation_operations"] = [
            "化学方程式→定量关系"
        ]
        item["features"]["calculation_operations"] = ["单一方程式"]

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "根据反应现象和方程式完成计算。", "options": ""},
        )

        self.assertEqual(
            result["postprocess_candidate_level"],
            "中等题",
        )
        self.assertEqual(
            result["postprocess_candidate_actions"][0]["rule"],
            "core12_basic_to_medium_complete_model",
        )

    def test_observable_projection_does_not_reuse_legacy_final_guard(
        self,
    ) -> None:
        item = rating("拔高题")
        item["features"].update(
            {
                "longest_solution_chain": [
                    "根据装置限制确定反应条件",
                    "由反应现象确定中间产物",
                    "把中间产物用于后一反应",
                    "联合图表数据建立计算关系",
                    "根据计算结果评价实验方案",
                ],
                "task_groups": [
                    {"task_type": "实验操作与探究", "count": 2},
                    {"task_type": "图表与数据", "count": 1},
                    {"task_type": "定量计算", "count": 1},
                ],
                "rule_families": [
                    "实验操作与探究",
                    "性质与反应判断",
                    "图表与数据",
                    "定量计算",
                ],
                "curriculum_units": ["U5", "U9", "U10"],
                "reaction_structure": "产物进入后一反应",
                "condition_operations": ["条件切换", "干扰条件排除"],
                "representation_operations": [
                    "宏观现象→化学符号",
                    "化学方程式→定量关系",
                    "图表数据→化学关系",
                ],
                "evidence_operations": ["多证据共同成立"],
                "experiment_operation": "多阶段定量探究",
                "graph_table_operation": "多图表联合",
                "calculation_operations": ["单一守恒", "多反应定量关系"],
                "new_information_operation": "新关系被多个任务共同使用",
            }
        )
        old_enabled = self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS
        old_writeback = (
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK
        )
        try:
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = True
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = True
            result = self.runtime.postprocess_chemistry_difficulty(
                item,
                {"stem": "根据装置、图表和实验数据完成探究。"},
            )
        finally:
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS = old_enabled
            self.runtime.CHEMISTRY_ENABLE_TEACHER_DISTRIBUTION_GUARDS_WRITEBACK = (
                old_writeback
            )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertIsNone(
            result["teacher_distribution_guard_candidate_action"]
        )

    def test_prompt_restores_detailed_boundary_calibration(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        required_anchors = (
            "不同化学命题或不同作答目标",
            "普通方案正误判断不等于方案评价",
            "纯算术、机械配平、重复代入",
            "只有信息、反应、实验、图表、条件或计算共同参与关键链才成立",
            "工业流程、未知组成与守恒联立",
            "同一透明规则下的机械筛选应合并",
            "多规则综合填空的受控广度",
            "同深度不同耦合",
            "固定基团转换关系",
            "枚举防错表",
        )
        for anchor in required_anchors:
            self.assertIn(anchor, prompt)
        self.assertGreaterEqual(prompt.count("【Case"), 29)
        for obsolete_field in (
            "knowledge_distribution",
            "chemical_object_distribution",
            "step_count",
            "task_count",
        ):
            self.assertNotIn(obsolete_field, prompt)

    def test_prompt_keeps_topic_coverage_separate_from_task_coupling(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        for anchor in (
            "只记录实际参与生成答案的最小课题集合",
            "`curriculum_topics`只描述知识覆盖",
            "任务之间是否共享中间结论或模型由`parallel_task_relation`记录",
            "不得根据课题数量或跨单元本身升降难度",
        ):
            self.assertIn(anchor, prompt)
        self.assertNotIn("历史V2", prompt)

    def test_prompt_records_teacher_observables_and_topic_span(self) -> None:
        prompt = PROMPT_PATH.read_text(encoding="utf-8")
        for anchor in (
            '"curriculum_topics"',
            '"parallel_task_relation"',
            '"visual_task_structure"',
            '"error_analysis_operation"',
            "跨学科语义或模型应用",
            "读数偏差到实际量判断",
            "只描述知识覆盖",
            "不得根据课题数量或跨单元本身升降难度",
        ):
            self.assertIn(anchor, prompt)

    def test_observable_error_chain_can_floor_basic_to_medium(self) -> None:
        item = rating("基础题")
        item["features"] = observable_v3_features()
        item["features"]["longest_solution_chain"] = [
            "判断仰视使量筒示数偏小",
            "由示数与真实体积关系得到实际取液体积",
            "判断配制溶液的质量分数偏差方向",
        ]
        item["features"]["experiment_operation"] = "基础操作或读数"
        item["features"]["error_analysis_operation"] = (
            "操作偏差到最终结果方向"
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "量筒仰视取液后判断配制结果偏差。"},
        )

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(
            result["postprocess_actions"][0]["rule"],
            "teacher_basic_to_medium_observable_error_chain",
        )

    def test_direct_error_consequence_does_not_force_medium(self) -> None:
        item = rating("基础题")
        item["features"] = observable_v3_features()
        item["features"]["longest_solution_chain"] = [
            "直接判断错误操作可能造成的后果"
        ]
        item["features"]["error_analysis_operation"] = (
            "直接判断错误操作后果"
        )
        item["features"]["experiment_operation"] = "基础操作或读数"

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "判断错误实验操作可能导致的后果。"},
        )

        self.assertEqual(result["difficulty_level"], "基础题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_v3_deep_quantitative_reaction_chain_remains_audit_only(self) -> None:
        item = rating("拔高题")
        item["features"] = observable_v3_features()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "根据前段反应确定中间产物",
                    "将中间产物代入后续反应",
                    "根据前后质量变化建立差量关系",
                    "联合守恒求出未知组成",
                    "使用组成完成后续定量验证",
                ],
                "reaction_structure": "产物进入后一反应",
                "calculation_operations": [
                    "单一守恒",
                    "差量",
                    "多反应定量关系",
                ],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "前段产物继续反应，根据差量与守恒求未知组成。"},
        )

        self.assertEqual(result["difficulty_level"], "拔高题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_hard_to_final_deep_quantitative_chain",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )
        self.assertIn(
            "历史V3",
            result["teacher_distribution_guard_writeback_blocked_reason"],
        )
        self.assertTrue(
            any(
                "历史V3" in flag
                for flag in result["feature_audit_flags"]
            )
        )

    def test_explicit_difference_method_is_audit_only(self) -> None:
        item = rating("中等题")
        item["features"] = observable_v4_features()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "建立反应前后质量差",
                    "由质量差求反应量",
                    "据方程式求未知量",
                    "核验最终结果",
                ],
                "reaction_structure": "单一反应",
                "calculation_operations": ["单一方程式", "差量"],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "根据反应前后质量差求未知物质质量。"},
        )

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_medium_to_hard_explicit_difference_method",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_v3_difference_method_remains_unchanged(self) -> None:
        item = rating("中等题")
        item["features"] = observable_v3_features()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "建立反应前后质量差",
                    "由质量差求反应量",
                    "据方程式求未知量",
                    "核验最终结果",
                ],
                "reaction_structure": "单一反应",
                "calculation_operations": ["单一方程式", "差量"],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "历史V3输出仅供回放，不启用新写回。"},
        )

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_single_equation_without_difference_stays_medium(self) -> None:
        item = rating("中等题")
        item["features"] = observable_v3_features()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "写出熟悉反应方程式",
                    "代入已知质量",
                    "求目标物质质量",
                ],
                "reaction_structure": "单一反应",
                "calculation_operations": ["单一方程式"],
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "根据一个熟悉方程式完成常规计算。"},
        )

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(result["postprocess_actions"], [])

    def test_shared_new_information_promotion_is_audit_only(self) -> None:
        item = rating("中等题")
        item["features"] = observable_v3_features()
        item["features"].update(
            {
                "longest_solution_chain": [
                    "读取题给的新关系",
                    "把同一查值结果用于两个相关任务",
                ],
                "new_information_operation": "直接查值",
                "parallel_task_relation": "共享同一化学模型的关联任务",
            }
        )

        result = self.runtime.postprocess_chemistry_difficulty(
            item,
            {"stem": "根据题给新关系完成两个相关任务。"},
        )

        self.assertEqual(result["difficulty_level"], "中等题")
        self.assertEqual(result["postprocess_actions"], [])
        self.assertEqual(
            result["teacher_distribution_guard_candidate_action"]["rule"],
            "teacher_medium_to_hard_shared_new_information",
        )
        self.assertFalse(
            result["teacher_distribution_guard_writeback_applied"]
        )

    def test_schema_retry_budget_allows_three_repairs(self) -> None:
        self.assertGreaterEqual(self.runtime.MAX_SCHEMA_RETRIES, 3)

    def test_legacy_core12_remains_readable_for_historical_results(self) -> None:
        legacy = copy.deepcopy(self.runtime.FEATURE_DEFAULTS)
        validated = self.runtime.validate_feature_contract(legacy)
        self.assertEqual(validated, legacy)


if __name__ == "__main__":
    unittest.main()
