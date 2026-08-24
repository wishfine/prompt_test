#!/usr/bin/env node

import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const FEATURE_FIELD_NAMES = {
  knowledge_L1: "知识模块",
  knowledge_L2: "知识专题",
  knowledge_points: "知识点",
  knowledge_count: "知识点数量",
  knowledge_scope: "知识范围",
  knowledge_depth: "知识深度",
  primary_problem_structure: "题型结构",
  step_count: "有效步骤",
  process_count: "物理过程数",
  object_count: "研究对象数",
  object_relation: "对象关系",
  state_count: "状态数",
  state_transition: "状态变化",
  process_state_relation: "过程与状态关系",
  constraint_structure: "约束结构",
  subquestion_dependency: "小问依赖",
  shared_model_across_subquestions: "是否共享复杂模型",
  model_explicitness: "模型显隐",
  model_relation: "模型关系",
  reasoning_chain: "推理链",
  hidden_conditions: "隐含条件",
  critical_state: "临界状态",
  classification_discussion: "分类讨论",
  variable_relation: "变量关系",
  physics_methods: "物理方法",
  formula_count: "公式数量",
  equation_structure: "方程结构",
  calculation_complexity: "计算复杂度",
  parameter_operation: "参数处理",
  numerical_complexity: "数值复杂度",
  unit_conversion: "单位换算",
  information_carrier: "信息载体",
  graph_structure: "图像结构",
  drawing_requirement: "作图要求",
  experiment_requirement: "实验要求",
  context_type: "情境类型",
  context_load: "情境负担",
  error_risk: "易错风险",
};


function parseArgs(argv) {
  const args = {};
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith("--")) continue;
    args[token.slice(2)] = argv[index + 1];
    index += 1;
  }
  if (!args.input || !args.output) {
    throw new Error(
      "用法：node generate_high_physics_teacher_review_xlsx.mjs " +
      "--input <结果.jsonl> --output <复核表.xlsx>",
    );
  }
  return args;
}


async function readJsonl(filePath) {
  const text = await fs.readFile(filePath, "utf8");
  return text
    .split(/\r?\n/)
    .filter((line) => line.trim())
    .map((line, index) => {
      const row = JSON.parse(line);
      if (row.question_id === undefined || row.question_id === null) {
        throw new Error(`${filePath} 第${index + 1}行缺少 question_id`);
      }
      return row;
    });
}


function plainText(value) {
  if (value === null || value === undefined || value === "") return "无";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (Array.isArray(value)) {
    const cleaned = value
      .map((item) => plainText(item))
      .filter((item) => item !== "无");
    return cleaned.length ? cleaned.join("、") : "无";
  }
  if (typeof value === "object") {
    return Object.entries(value)
      .map(([key, child]) => `${key}：${plainText(child)}`)
      .join("；") || "无";
  }
  const text = String(value).trim();
  if (!text || text === "[]") return "无";
  if (/^false$/i.test(text)) return "否";
  if (/^true$/i.test(text)) return "是";
  const fieldNames = {
    reviewed_high_difficulty_features: "复核后的高难特征",
    original_predicted_accuracy: "原始预测正确率",
    shared_model_across_subquestions: "小问是否共享复杂模型",
    task_completion_structure: "任务完成结构",
    whole_question_burden: "整题负担",
    local_model_familiarity: "局部模型熟悉度",
    adjacent_boundary_review: "相邻边界复核",
    threshold_review: "分数边界复核",
  };
  let humanized = text;
  for (const [rawName, chineseName] of Object.entries(fieldNames)) {
    humanized = humanized.replaceAll(rawName, chineseName);
  }
  return humanized.replaceAll("features", "特征").replaceAll("feature", "特征");
}


function compactQuestion(row) {
  const stem = plainText(row.stem).replace(/\s+/g, " ");
  const options = row.options;
  let optionText = "";
  if (Array.isArray(options) && options.length) {
    optionText = options.map((item) => plainText(item)).join("；");
  } else if (options && options !== "无") {
    optionText = plainText(options);
  }
  const full = optionText === "无" || !optionText
    ? stem
    : `${stem}【选项】${optionText}`;
  return full.length > 260 ? `${full.slice(0, 257)}…` : full;
}


function scoreValue(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}


function multiplierText(value) {
  const number = Number(value);
  if (Math.abs(number - 1) < 1e-9) return "不调整（×1.00）";
  if (Math.abs(number - 0.85) < 1e-9) {
    return "正确率下调15%（×0.85）";
  }
  if (Math.abs(number - 0.7) < 1e-9) {
    return "正确率下调30%（×0.70）";
  }
  return Number.isFinite(number) ? `正确率乘以${number}` : "无";
}


function highFeatureText(stage1) {
  const features = stage1.high_difficulty_features;
  if (!Array.isArray(features) || features.length === 0) return "无";
  return `${features.length}项：${features.join("、")}`;
}


function highFeatureEvidenceText(stage1) {
  const entries = Array.isArray(stage1.high_difficulty_feature_evidence)
    ? stage1.high_difficulty_feature_evidence
    : [];
  if (!entries.length) return "无";
  return entries.map((entry) => {
    const name = plainText(entry?.name);
    const evidence = Array.isArray(entry?.evidence)
      ? entry.evidence.map(plainText).join("、")
      : plainText(entry?.evidence);
    return `${name}：${evidence}`;
  }).join("\n");
}


function reviewedHighFeatureText(verification) {
  const features = Array.isArray(verification.reviewed_high_difficulty_features)
    ? verification.reviewed_high_difficulty_features
    : [];
  return features.length
    ? `${features.length}项：${features.map(plainText).join("、")}`
    : "无";
}


function coreFeatureColumns(features) {
  return [
    combinePairs([
      ["知识范围", features.knowledge_scope],
      ["知识深度", features.knowledge_depth],
      ["题型", features.primary_problem_structure],
    ]),
    combinePairs([
      ["有效步骤", features.step_count],
      ["任务结构", features.subquestion_dependency],
      ["整题关系", features.process_state_relation],
    ]),
    combinePairs([
      ["对象", features.object_count],
      ["对象关系", features.object_relation],
      ["过程", features.process_count],
      ["状态", features.state_count],
      ["状态变化", features.state_transition],
    ]),
    combinePairs([
      ["模型显隐", features.model_explicitness],
      ["模型关系", features.model_relation],
      ["推理链", features.reasoning_chain],
      ["共享模型", features.shared_model_across_subquestions],
    ]),
    combinePairs([
      ["约束", features.constraint_structure],
      ["隐含条件", features.hidden_conditions],
      ["临界", features.critical_state],
      ["分类讨论", features.classification_discussion],
    ]),
    combinePairs([
      ["方程", features.equation_structure],
      ["计算", features.calculation_complexity],
      ["图像", features.graph_structure],
      ["实验", features.experiment_requirement],
      ["信息载体", features.information_carrier],
    ]),
  ];
}


function featureCorrectionsText(verification) {
  if (!verification || typeof verification !== "object") return "未完成复核";
  const parts = [];
  const corrections = Array.isArray(verification.supported_feature_corrections)
    ? verification.supported_feature_corrections
    : [];
  for (const correction of corrections) {
    if (!correction || typeof correction !== "object") continue;
    const field = FEATURE_FIELD_NAMES[correction.field] || correction.field || "字段";
    const evidence = correction.evidence ? `；依据：${correction.evidence}` : "";
    parts.push(
      `${field}：${plainText(correction.from)} → ${plainText(correction.to)}${evidence}`,
    );
  }
  const missed = Array.isArray(verification.missed_features)
    ? verification.missed_features.filter((item) => plainText(item) !== "无")
    : [];
  if (missed.length) parts.push(`遗漏特征：${missed.map(plainText).join("、")}`);
  if (verification.high_difficulty_features_changed === true) {
    parts.push("高难特征集合发生变化");
  }
  return parts.length ? parts.join("；") : "无";
}


function thresholdBoundaryText(stage1) {
  const level = stage1.difficulty_level_step1;
  const review = stage1.threshold_review || {};
  const evidence = stage1.threshold_evidence || {};
  const relevant = {
    "难度1档": [["88", "can_reach_88", "boundary_88"]],
    "难度2档": [
      ["88", "can_reach_88", "boundary_88"],
      ["85", "can_reach_85", "boundary_85"],
    ],
    "难度3档": [
      ["85", "can_reach_85", "boundary_85"],
      ["58", "can_reach_58", "boundary_58"],
    ],
    "难度4档": [
      ["58", "can_reach_58", "boundary_58"],
      ["38", "can_reach_38", "boundary_38"],
    ],
    "难度5档": [["38", "can_reach_38", "boundary_38"]],
  }[level] || [];
  if (!relevant.length) return "无";
  return relevant.map(([score, flag, evidenceKey]) => {
    const verdict = review[flag] === true ? "通过" : "未通过";
    return `${score}分边界：${verdict}。${plainText(evidence[evidenceKey])}`;
  }).join("\n");
}


function combinePairs(pairs) {
  return pairs
    .map(([label, value]) => `${label}：${plainText(value)}`)
    .join("；");
}


function detailImageText(value) {
  if (Array.isArray(value)) return plainText(value);
  return plainText(value);
}


function buildMainRows(results) {
  const featureKeys = Object.keys(FEATURE_FIELD_NAMES);
  return results.map((row) => {
    const stage1 = row.difficulty_rating_stage1 || {};
    const features = stage1.features || {};
    const verification = row.verification || {};
    const finalScore = scoreValue(verification.reviewed_predicted_accuracy)
      ?? scoreValue(stage1.predicted_accuracy);
    const stage1Level = plainText(stage1.difficulty_level_step1);
    const finalLevel = plainText(row.final_difficulty_level);
    const finalAdjustment = stage1Level === finalLevel
      ? "未调整"
      : `${stage1Level} → ${finalLevel}`;
    const correctionText = featureCorrectionsText(verification);
    const reviewCorrections = verification.has_structural_revision === true
      ? `存在结构修正；${correctionText}`
      : correctionText;
    return [
      String(row.question_id),
      finalLevel,
      finalScore,
      finalAdjustment,
      "未标注（默认认可模型）",
      "",
      "",
      ...featureKeys.map((key) => plainText(features[key])),
      scoreValue(stage1.original_predicted_accuracy),
      highFeatureText(stage1),
      highFeatureEvidenceText(stage1),
      multiplierText(stage1.multiplier_applied),
      scoreValue(stage1.predicted_accuracy),
      stage1Level,
      plainText(stage1.reason),
      thresholdBoundaryText(stage1),
      plainText(verification.adjacent_boundary_review?.decisive_evidence),
      scoreValue(verification.reviewed_original_predicted_accuracy),
      reviewedHighFeatureText(verification),
      multiplierText(verification.reviewed_multiplier),
      scoreValue(verification.reviewed_predicted_accuracy),
      plainText(verification.reviewed_difficulty_level),
      plainText(verification.confidence),
      reviewCorrections,
      combinePairs([
        ["难度来源", verification.difficulty_source],
        ["复核说明", verification.analysis],
      ]),
    ];
  });
}


function buildQuestionRows(results) {
  return results.map((row) => [
    String(row.question_id),
    plainText(row.final_difficulty_level),
    plainText(row.stem),
    plainText(row.options),
    plainText(row.analysis),
    detailImageText(row.stem_image_url),
    detailImageText(row.analysis_image_url),
    plainText(row.input_quality?.input_sufficiency),
  ]);
}


function styleTitleRows(sheet, endColumn) {
  sheet.getRange(`A1:${endColumn}1`).merge();
  sheet.getRange(`A2:${endColumn}2`).merge();
  sheet.getRange(`A3:${endColumn}3`).merge();
  sheet.getRange(`A1:${endColumn}1`).format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF", size: 16 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${endColumn}2`).format = {
    fill: "#DCE6F1",
    font: { color: "#1F2937", size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A3:${endColumn}3`).format = {
    fill: "#FFF2CC",
    font: { color: "#7C2D12", size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  sheet.getRange(`A1:${endColumn}1`).format.rowHeight = 30;
  sheet.getRange(`A2:${endColumn}2`).format.rowHeight = 42;
  sheet.getRange(`A3:${endColumn}3`).format.rowHeight = 36;
}


function applyDifficultyFormatting(range) {
  const colors = {
    "难度1档": "#DCFCE7",
    "难度2档": "#ECFCCB",
    "难度3档": "#FEF3C7",
    "难度4档": "#FED7AA",
    "难度5档": "#FECACA",
  };
  for (const [text, fill] of Object.entries(colors)) {
    range.conditionalFormats.add("containsText", {
      text,
      format: { fill, font: { bold: true, color: "#1F2937" } },
    });
  }
}


async function buildWorkbook(results, outputPath) {
  const workbook = Workbook.create();
  const main = workbook.worksheets.add("老师复核表");
  const details = workbook.worksheets.add("题目详情");

  const featureKeys = Object.keys(FEATURE_FIELD_NAMES);
  const headers = [
    "题目ID", "最终难度", "最终正确率", "最终调整说明",
    "老师复核结论", "老师建议档位", "老师备注",
    ...featureKeys.map((key) => FEATURE_FIELD_NAMES[key]),
    "原始预测正确率", "高难特征", "高难特征证据", "乘数处理",
    "乘数后正确率", "第一阶段档位", "初判理由", "关键边界依据",
    "复核依据", "复核后原始正确率", "复核后高难特征", "复核后乘数",
    "复核后正确率", "复核后档位", "复核置信度", "结构与特征修正",
    "复核理由",
  ];
  const mainRows = buildMainRows(results);
  main.getRange("A1").values = [["高中物理难度评级——教师复核表"]];
  main.getRange("A2").values = [[
    "阅读顺序：先看左侧最终结果，再看第一阶段初判与程序乘数处理，最后看第二阶段复核。第一阶段先估计原始正确率，程序根据高难特征应用乘数并映射档位；第二阶段只复核结构与相邻边界，默认维持第一阶段。老师不填写时，默认认可模型结果。",
  ]];
  main.getRange("A3").values = [[
    "显示规则：‘无’表示未识别或不适用；内部逻辑值统一显示为‘是/否’；空列表统一显示为‘无’；乘数×1.00表示不调整，×0.85/×0.70表示正确率下调15%/30%；内部规范化日志、提示词原始字段和接口调用字段不展示。",
  ]];
  // 宽表只在左侧结果区展示说明，避免标题居中到几十列之外后首屏不可见。
  styleTitleRows(main, "G");

  main.getRange("A4:D4").merge();
  main.getRange("E4:G4").merge();
  main.getRange("H4:BA4").merge();
  main.getRange("BB4:BJ4").merge();
  main.getRange("A4").values = [["题目与最终结果"]];
  main.getRange("E4").values = [["老师填写（不填默认认可模型）"]];
  main.getRange("H4").values = [["第一阶段：38项特征 + 正确率与程序后处理"]];
  main.getRange("BB4").values = [["第二阶段：结构与结果复核"]];
  const groupStyles = [
    ["A4:D4", "#1F4E78"],
    ["E4:G4", "#BF9000"],
    ["H4:BA4", "#2563EB"],
    ["BB4:BJ4", "#7C3AED"],
  ];
  for (const [rangeAddress, fill] of groupStyles) {
    main.getRange(rangeAddress).format = {
      fill,
      font: { bold: true, color: "#FFFFFF", size: 11 },
      horizontalAlignment: "center",
      verticalAlignment: "center",
    };
  }
  main.getRange("A4:BJ4").format.rowHeight = 24;
  main.getRange("A5:BJ5").values = [headers];
  main.getRange("A5:D5").format.fill = "#D9EAF7";
  main.getRange("E5:G5").format.fill = "#FFF2CC";
  main.getRange("H5:BA5").format.fill = "#DCE6FF";
  main.getRange("BB5:BJ5").format.fill = "#E9D5FF";
  main.getRange("A5:BJ5").format.font = { bold: true, color: "#1F2937" };
  main.getRange("A5:BJ5").format.wrapText = true;
  main.getRange("A5:BJ5").format.horizontalAlignment = "center";
  main.getRange("A5:BJ5").format.verticalAlignment = "center";
  main.getRange("A5:BJ5").format.rowHeight = 38;

  const lastRow = 5 + mainRows.length;
  main.getRange(`A6:BJ${lastRow}`).values = mainRows;
  main.getRange(`A6:BJ${lastRow}`).format = {
    verticalAlignment: "top",
    wrapText: true,
    font: { size: 9, color: "#1F2937" },
    borders: {
      insideHorizontal: { style: "thin", color: "#E5E7EB" },
    },
  };
  main.getRange(`A6:A${lastRow}`).format.numberFormat = "@";
  main.getRange(`C6:C${lastRow}`).format.numberFormat = "0.0";
  main.getRange(`AT6:AT${lastRow}`).format.numberFormat = "0.0";
  main.getRange(`AX6:AX${lastRow}`).format.numberFormat = "0.0";
  main.getRange(`BC6:BC${lastRow}`).format.numberFormat = "0.0";
  main.getRange(`BF6:BF${lastRow}`).format.numberFormat = "0.0";
  main.getRange(`A6:A${lastRow}`).format.horizontalAlignment = "center";
  main.getRange(`B6:BJ${lastRow}`).format.verticalAlignment = "top";
  main.getRange(`E6:G${lastRow}`).format.fill = "#FFFBEB";
  main.getRange(`A6:BJ${lastRow}`).format.rowHeight = 72;

  const featureWidths = featureKeys.map((key) => {
    if (["knowledge_points", "physics_methods"].includes(key)) return 36;
    if (["knowledge_L1", "knowledge_L2"].includes(key)) return 28;
    if (["process_state_relation", "subquestion_dependency"].includes(key)) return 26;
    return 20;
  });
  const widths = [
    22, 12, 13, 22, 25, 14, 38,
    ...featureWidths,
    15, 34, 48, 25, 15, 13, 52, 58,
    52, 18, 38, 25, 18, 13, 12, 48, 62,
  ];
  widths.forEach((width, index) => {
    main.getRangeByIndexes(0, index, lastRow, 1).format.columnWidth = width;
  });
  main.freezePanes.freezeRows(5);
  main.freezePanes.freezeColumns(3);
  main.showGridLines = false;

  const table = main.tables.add(`A5:BJ${lastRow}`, true, "TeacherReviewTable");
  table.style = "TableStyleMedium2";
  table.showBandedColumns = false;
  table.showFilterButton = true;

  main.getRange(`E6:E${lastRow}`).dataValidation = {
    rule: {
      type: "list",
      values: [
        "未标注（默认认可模型）",
        "模型合理",
        "模型偏易",
        "模型偏难",
        "信息不足",
      ],
    },
  };
  main.getRange(`F6:F${lastRow}`).dataValidation = {
    rule: {
      type: "list",
      values: ["难度1档", "难度2档", "难度3档", "难度4档", "难度5档"],
    },
  };
  applyDifficultyFormatting(main.getRange(`B6:B${lastRow}`));
  applyDifficultyFormatting(main.getRange(`AY6:AY${lastRow}`));
  applyDifficultyFormatting(main.getRange(`BG6:BG${lastRow}`));
  main.getRange(`D6:D${lastRow}`).conditionalFormats.add("notContainsText", {
    text: "未调整",
    format: { fill: "#FEF3C7", font: { bold: true, color: "#92400E" } },
  });

  const detailHeaders = [
    "题目ID", "最终难度", "完整题干", "选项", "官方解析",
    "题干图片", "解析图片", "输入信息充分性",
  ];
  const detailRows = buildQuestionRows(results);
  details.getRange("A1:H1").merge();
  details.getRange("A1").values = [["题目详情（与老师复核表按题目ID对应）"]];
  details.getRange("A2:H2").merge();
  details.getRange("A2").values = [[
    "本Sheet仅用于查看完整题干、选项、解析和图片地址；评级与老师复核请在“老师复核表”中完成。",
  ]];
  details.getRange("A3:H3").values = [detailHeaders];
  details.getRange(`A4:H${3 + detailRows.length}`).values = detailRows;
  details.getRange("A1:H1").format = {
    fill: "#17365D",
    font: { bold: true, color: "#FFFFFF", size: 15 },
    horizontalAlignment: "center",
    verticalAlignment: "center",
  };
  details.getRange("A2:H2").format = {
    fill: "#DCE6F1",
    font: { color: "#1F2937", size: 10 },
    wrapText: true,
    verticalAlignment: "center",
  };
  details.getRange("A3:H3").format = {
    fill: "#1F4E78",
    font: { bold: true, color: "#FFFFFF" },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  details.getRange(`A4:H${3 + detailRows.length}`).format = {
    verticalAlignment: "top",
    wrapText: true,
    font: { size: 9 },
    borders: {
      insideHorizontal: { style: "thin", color: "#E5E7EB" },
    },
  };
  details.getRange(`A4:A${3 + detailRows.length}`).format.numberFormat = "@";
  [22, 12, 64, 45, 72, 38, 38, 18].forEach((width, index) => {
    details.getRangeByIndexes(0, index, 3 + detailRows.length, 1).format.columnWidth = width;
  });
  details.getRange(`A4:H${3 + detailRows.length}`).format.rowHeight = 72;
  details.getRange("A1:H1").format.rowHeight = 30;
  details.getRange("A2:H2").format.rowHeight = 30;
  details.getRange("A3:H3").format.rowHeight = 34;
  details.freezePanes.freezeRows(3);
  details.freezePanes.freezeColumns(2);
  details.showGridLines = false;
  const detailTable = details.tables.add(
    `A3:H${3 + detailRows.length}`,
    true,
    "QuestionDetailsTable",
  );
  detailTable.style = "TableStyleMedium2";
  detailTable.showFilterButton = true;
  applyDifficultyFormatting(details.getRange(`B4:B${3 + detailRows.length}`));

  const outputDir = path.dirname(outputPath);
  await fs.mkdir(outputDir, { recursive: true });

  const mainInspect = await workbook.inspect({
    kind: "table",
    range: "老师复核表!A1:G8",
    include: "values,formulas",
    tableMaxRows: 10,
    tableMaxCols: 7,
    maxChars: 12000,
  });
  console.log(mainInspect.ndjson);
  const featureInspect = await workbook.inspect({
    kind: "table",
    range: "老师复核表!H5:N8",
    include: "values,formulas",
    tableMaxRows: 4,
    tableMaxCols: 7,
    maxChars: 6000,
  });
  console.log(featureInspect.ndjson);
  const pipelineInspect = await workbook.inspect({
    kind: "table",
    range: "老师复核表!AT5:BJ8",
    include: "values,formulas",
    tableMaxRows: 4,
    tableMaxCols: 17,
    maxChars: 10000,
  });
  console.log(pipelineInspect.ndjson);
  const errorScan = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "公式错误扫描",
  });
  console.log(errorScan.ndjson);

  const mainPreview = await workbook.render({
    sheetName: "老师复核表",
    range: "A1:G11",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    "/tmp/high_physics_teacher_review_main.png",
    new Uint8Array(await mainPreview.arrayBuffer()),
  );
  const stage1Preview = await workbook.render({
    sheetName: "老师复核表",
    range: "H1:Z11",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    "/tmp/high_physics_teacher_review_stage1.png",
    new Uint8Array(await stage1Preview.arrayBuffer()),
  );
  const stage1MorePreview = await workbook.render({
    sheetName: "老师复核表",
    range: "AA1:BA11",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    "/tmp/high_physics_teacher_review_stage1_more.png",
    new Uint8Array(await stage1MorePreview.arrayBuffer()),
  );
  const stage2Preview = await workbook.render({
    sheetName: "老师复核表",
    range: "BB1:BJ11",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    "/tmp/high_physics_teacher_review_stage2.png",
    new Uint8Array(await stage2Preview.arrayBuffer()),
  );
  const detailPreview = await workbook.render({
    sheetName: "题目详情",
    range: "A1:H7",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    "/tmp/high_physics_teacher_review_details.png",
    new Uint8Array(await detailPreview.arrayBuffer()),
  );

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  console.log(JSON.stringify({
    inputRows: results.length,
    visibleMainColumns: headers.length,
    sheets: ["老师复核表", "题目详情"],
    output: outputPath,
  }, null, 2));
}


const args = parseArgs(process.argv.slice(2));
const results = await readJsonl(args.input);
await buildWorkbook(results, args.output);
