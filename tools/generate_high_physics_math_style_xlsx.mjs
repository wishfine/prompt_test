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
  const ratingOnly = args["rating-only"] === "1";
  if (
    !args.input
    || !args["rating-output"]
    || (!ratingOnly && !args["feature-output"])
  ) {
    throw new Error(
      "用法：node generate_high_physics_math_style_xlsx.mjs " +
      "--input <结果.jsonl> [--feature-output <带特征.xlsx>] " +
      "--rating-output <评级结果.xlsx> [--rating-only 1]",
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


function cellText(value) {
  if (value === null || value === undefined || value === "") return "无";
  if (typeof value === "boolean") return value ? "是" : "否";
  if (Array.isArray(value)) {
    const values = value.map(cellText).filter((item) => item !== "无");
    return values.length ? values.join(", ") : "无";
  }
  if (typeof value === "object") {
    const values = Object.entries(value).map(
      ([key, child]) => `${key}: ${cellText(child)}`,
    );
    return values.length ? values.join("; ") : "无";
  }
  const text = String(value).trim();
  if (!text || text === "[]") return "无";
  if (/^false$/i.test(text)) return "否";
  if (/^true$/i.test(text)) return "是";
  return text;
}


function numberOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}


function finalAccuracy(row) {
  return numberOrNull(row.verification?.reviewed_predicted_accuracy)
    ?? numberOrNull(row.difficulty_rating_stage1?.predicted_accuracy);
}


function finalLevel(row) {
  return cellText(
    row.final_difficulty_level
      ?? row.verification?.reviewed_difficulty_level
      ?? row.difficulty_rating_stage1?.difficulty_level_step1,
  );
}


function finalAdjustmentForSpreadsheet(row) {
  const adjustment = cellText(row.final_adjustment);
  // “转人工复核”是内部审计状态，不应放入给老师查看的工作表。
  // 保留它未被自动采信这一事实，以及实际维持的档位即可。
  if (adjustment.includes("转人工复核")) {
    return `复核意见未采纳·维持${finalLevel(row)}`;
  }
  return adjustment;
}


function correctionText(value) {
  if (!Array.isArray(value) || value.length === 0) return "无";
  const parts = value.map((item) => {
    if (!item || typeof item !== "object") return cellText(item);
    const fieldName = FEATURE_FIELD_NAMES[item.field] ?? cellText(item.field);
    return [
      `字段：${fieldName}`,
      `原值：${cellText(item.from)}`,
      `修正为：${cellText(item.to)}`,
      `依据：${cellText(item.evidence)}`,
    ].join("；");
  }).filter((item) => item && item !== "无");
  return parts.length ? parts.join("｜") : "无";
}


function featureRows(results) {
  const keys = Object.keys(FEATURE_FIELD_NAMES);
  return results.map((row) => {
    const features = row.difficulty_rating_stage1?.features || {};
    return [
      String(row.question_id),
      ...keys.map((key) => cellText(features[key])),
      finalAccuracy(row),
      finalLevel(row),
    ];
  });
}


function ratingRows(results) {
  return results.map((row) => {
    const verification = row.verification || {};
    const boundaryReview = verification.adjacent_boundary_review || {};
    return [
      String(row.question_id),
      cellText(verification.difficulty_source),
      cellText(verification.missed_features),
      cellText(verification.has_structural_revision),
      cellText(boundaryReview.boundaries_checked),
      cellText(boundaryReview.decisive_evidence),
      numberOrNull(verification.reviewed_original_predicted_accuracy),
      cellText(verification.reviewed_high_difficulty_features),
      cellText(verification.analysis),
      numberOrNull(verification.reviewed_high_difficulty_feature_count)
        ?? numberOrNull(row.reviewed_high_difficulty_feature_count)
        ?? 0,
      numberOrNull(verification.reviewed_multiplier),
      numberOrNull(verification.reviewed_predicted_accuracy),
      finalAdjustmentForSpreadsheet(row),
    ];
  });
}


function applyFlatSheetFormatting(sheet, rowCount, columnCount, widths) {
  const used = sheet.getRangeByIndexes(0, 0, rowCount, columnCount);
  used.format = {
    font: { size: 11, color: "#000000" },
    verticalAlignment: "center",
  };
  sheet.getRangeByIndexes(0, 0, 1, columnCount).format.font = {
    bold: false,
    size: 11,
    color: "#000000",
  };
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, rowCount, 1).format.columnWidth = width;
  });
  sheet.getRangeByIndexes(0, 0, rowCount, 1).format.numberFormat = "@";
  sheet.getRangeByIndexes(0, 0, rowCount, 1).format.horizontalAlignment = "left";
  used.format.rowHeight = 20;
  sheet.freezePanes.freezeRows(1);
  sheet.showGridLines = true;
}


async function buildFeatureWorkbook(results, outputPath) {
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("特征明细");
  const keys = Object.keys(FEATURE_FIELD_NAMES);
  const headers = [
    "question_id",
    ...keys.map((key) => FEATURE_FIELD_NAMES[key]),
    "predicted_accuracy",
    "difficulty_level",
  ];
  const rows = featureRows(results);
  sheet.getRangeByIndexes(0, 0, rows.length + 1, headers.length).values = [
    headers,
    ...rows,
  ];
  const featureWidths = keys.map((key) => {
    if (key === "knowledge_points") return 42;
    if (["knowledge_L1", "knowledge_L2", "physics_methods"].includes(key)) {
      return 28;
    }
    return 20;
  });
  applyFlatSheetFormatting(
    sheet,
    rows.length + 1,
    headers.length,
    [23, ...featureWidths, 20, 16],
  );
  sheet.getRangeByIndexes(1, headers.length - 2, rows.length, 1)
    .format.numberFormat = "0.0";

  const check = await workbook.inspect({
    kind: "table",
    range: "特征明细!A1:AO8",
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: 41,
    maxChars: 16000,
  });
  console.log(check.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "带特征结果公式错误扫描",
  });
  console.log(errors.ndjson);
  const preview = await workbook.render({
    sheetName: "特征明细",
    range: "A1:AO12",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    "/tmp/high_physics_math_style_features.png",
    new Uint8Array(await preview.arrayBuffer()),
  );
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
}


async function buildRatingWorkbook(results, outputPath) {
  const workbook = Workbook.create();
  const sheet = workbook.worksheets.add("评级结果");
  const headers = [
    "题目ID",
    "难度来源",
    "遗漏特征",
    "是否存在结构性修正",
    "已复核的相邻边界",
    "边界决定性证据",
    "复核后原始预测正确率",
    "复核后高难特征",
    "复核详细分析",
    "复核后高难特征数量",
    "复核后正确率乘数",
    "复核后预测正确率",
    "最终调整动作",
  ];
  const rows = ratingRows(results);
  sheet.getRangeByIndexes(0, 0, rows.length + 1, headers.length).values = [
    headers,
    ...rows,
  ];
  applyFlatSheetFormatting(
    sheet,
    rows.length + 1,
    headers.length,
    [
      22, 40, 30, 18, 20, 46, 18, 36, 56, 18, 16, 18, 16,
    ],
  );
  sheet.getRange(`G2:G${rows.length + 1}`).format.numberFormat = "0.0";
  sheet.getRange(`J2:J${rows.length + 1}`).format.numberFormat = "0";
  sheet.getRange(`K2:K${rows.length + 1}`).format.numberFormat = "0.00";
  sheet.getRange(`L2:L${rows.length + 1}`).format.numberFormat = "0.0";
  sheet.getRange(`B1:M${rows.length + 1}`).format.wrapText = true;
  sheet.getRange(`A2:M${rows.length + 1}`).format.verticalAlignment = "top";
  sheet.getRange(`A2:M${rows.length + 1}`).format.rowHeight = 72;
  sheet.getRange("A1:M1").format.rowHeight = 36;
  sheet.freezePanes.freezeColumns(1);

  const check = await workbook.inspect({
    kind: "table",
    range: "评级结果!A1:M5",
    include: "values,formulas",
    tableMaxRows: 8,
    tableMaxCols: 13,
    maxChars: 24000,
  });
  console.log(check.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "评级结果公式错误扫描",
  });
  console.log(errors.ndjson);
  const preview = await workbook.render({
    sheetName: "评级结果",
    range: "A1:G6",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    "/tmp/high_physics_math_style_rating.png",
    new Uint8Array(await preview.arrayBuffer()),
  );
  const previewReview = await workbook.render({
    sheetName: "评级结果",
    range: "H1:M6",
    scale: 1,
    format: "png",
  });
  await fs.writeFile(
    "/tmp/high_physics_math_style_rating_review.png",
    new Uint8Array(await previewReview.arrayBuffer()),
  );
  await fs.mkdir(path.dirname(outputPath), { recursive: true });
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
}


const args = parseArgs(process.argv.slice(2));
const results = await readJsonl(args.input);
if (args["rating-only"] !== "1") {
  await buildFeatureWorkbook(results, args["feature-output"]);
}
await buildRatingWorkbook(results, args["rating-output"]);
console.log(JSON.stringify({
  inputRows: results.length,
  ratingColumns: 13,
  ratingOnly: args["rating-only"] === "1",
  featureColumns: args["rating-only"] === "1"
    ? null
    : Object.keys(FEATURE_FIELD_NAMES).length + 3,
  featureOutput: args["feature-output"] ?? null,
  ratingOutput: args["rating-output"],
}, null, 2));
