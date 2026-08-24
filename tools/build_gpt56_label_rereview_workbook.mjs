import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const root = `${path.dirname(path.dirname(fileURLToPath(import.meta.url)))}/`;
const rows = JSON.parse(await fs.readFile(`${root}tmp/physics_adjudicated_labels_gpt56_rereview_1066.json`, "utf8"));
const summary = JSON.parse(await fs.readFile(`${root}tmp/physics_adjudicated_labels_gpt56_rereview_summary.json`, "utf8"));
const outputPath = `${root}output/spreadsheet/physics_gpt56_labels_rereview_1066.xlsx`;

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("复核汇总");
const detailSheet = workbook.worksheets.add("1066题逐题复核");
const changedSheet = workbook.worksheets.add("主标签修改20");
const boundarySheet = workbook.worksheets.add("相邻边界157");
const methodSheet = workbook.worksheets.add("复核口径");

const navy = "#16324F";
const blue = "#2F6B9A";
const paleBlue = "#EAF2F8";
const paleGold = "#FFF4D6";
const paleRed = "#FDECEC";
const paleGreen = "#EAF6EE";
const gray = "#5F6B76";
const lightBorder = "#D7DEE5";

function styleTitle(sheet, range, title) {
  sheet.mergeCells(range);
  const cell = sheet.getRange(range);
  cell.values = [[title]];
  cell.format = {
    fill: navy,
    font: { bold: true, color: "#FFFFFF", size: 16 },
    verticalAlignment: "center",
  };
  cell.format.rowHeight = 34;
}

function styleHeader(range) {
  range.format = {
    fill: blue,
    font: { bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: lightBorder },
  };
  range.format.rowHeight = 28;
}

function writeDetailSheet(sheet, title, selectedRows, tableName) {
  const headers = Object.keys(rows[0]);
  styleTitle(sheet, `A1:${columnName(headers.length)}1`, title);
  sheet.getRangeByIndexes(1, 0, 1, headers.length).values = [headers];
  styleHeader(sheet.getRangeByIndexes(1, 0, 1, headers.length));
  const lastRow = selectedRows.length + 2;
  const lastCol = columnName(headers.length);
  if (selectedRows.length) {
    sheet.getRange(`A3:A${lastRow}`).format.numberFormat = "@";
  }
  if (selectedRows.length) {
    const matrix = selectedRows.map((row) => headers.map((header) => {
      const value = row[header] ?? "";
      // Prefix a zero-width text marker so spreadsheet engines never coerce
      // 19-digit identifiers to IEEE-754 numbers. The canonical CSV remains
      // the machine-readable source without this display-only marker.
      return header === "题目ID" ? `\u200B${String(value)}` : value;
    }));
    sheet.getRangeByIndexes(2, 0, matrix.length, headers.length).values = matrix;
  }
  const table = sheet.tables.add(`A2:${lastCol}${lastRow}`, true, tableName);
  table.style = "TableStyleMedium2";
  sheet.freezePanes.freezeRows(2);
  sheet.freezePanes.freezeColumns(3);
  sheet.showGridLines = false;
  sheet.getRange(`A2:${lastCol}${lastRow}`).format.verticalAlignment = "top";
  sheet.getRange(`A3:J${lastRow}`).format.wrapText = true;
  const widths = [22, 12, 12, 19, 22, 12, 12, 12, 55, 30, 70, 42, 72, 36, 36, 12, 14, 12, 12, 14, 14, 16, 14];
  widths.forEach((width, index) => {
    sheet.getRangeByIndexes(0, index, lastRow, 1).format.columnWidth = width;
  });
  if (lastRow >= 3) {
    sheet.getRange(`I3:I${lastRow}`).conditionalFormats.add("containsText", {
      text: "修订GPT主标签",
      format: { fill: paleRed, font: { bold: true, color: "#A61B1B" } },
    });
    sheet.getRange(`I3:I${lastRow}`).conditionalFormats.add("containsText", {
      text: "边界",
      format: { fill: paleGold, font: { color: "#7A5200" } },
    });
  }
}

function columnName(count) {
  let n = count;
  let result = "";
  while (n > 0) {
    n -= 1;
    result = String.fromCharCode(65 + (n % 26)) + result;
    n = Math.floor(n / 26);
  }
  return result;
}

// Summary sheet.
styleTitle(summarySheet, "A1:O1", "初中物理1066题 GPT-5.6标签重新复核");
summarySheet.getRange("A3:B3").values = [["核心结论", "数量"]];
styleHeader(summarySheet.getRange("A3:B3"));
summarySheet.getRange("A4:A7").values = [["逐题覆盖"], ["主标签修改"], ["相邻双档边界"], ["保留原主标签"]];
summarySheet.getRange("B4").formulas = [["=COUNTA('1066题逐题复核'!A3:A1068)"]];
summarySheet.getRange("B5").formulas = [["=COUNTIF('1066题逐题复核'!E3:E1068,\"修订GPT主标签\")"]];
summarySheet.getRange("B6").formulas = [["=COUNTA('相邻边界157'!A3:A159)"]];
summarySheet.getRange("B7").formulas = [["=B4-B5"]];
summarySheet.getRange("A4:B7").format.borders = { preset: "inside", style: "thin", color: lightBorder };
summarySheet.getRange("B4:B7").format.numberFormat = "#,##0";

summarySheet.getRange("D3:F3").values = [["档位", "原GPT分布", "复核后分布"]];
styleHeader(summarySheet.getRange("D3:F3"));
const levels = ["送分题", "基础题", "中等题", "拔高题", "压轴题"];
summarySheet.getRange("D4:D8").values = levels.map((level) => [level]);
summarySheet.getRange("E4").formulas = [["=COUNTIF('1066题逐题复核'!B3:B1068,D4)"]];
summarySheet.getRange("E4:E8").fillDown();
summarySheet.getRange("F4").formulas = [["=COUNTIF('1066题逐题复核'!C3:C1068,D4)"]];
summarySheet.getRange("F4:F8").fillDown();
summarySheet.getRange("D4:F8").format.borders = { preset: "inside", style: "thin", color: lightBorder };

summarySheet.getRange("A10:C10").values = [["主标签调整方向", "数量", "判断"]];
styleHeader(summarySheet.getRange("A10:C10"));
const transitions = Object.entries(summary.change_transitions);
summarySheet.getRangeByIndexes(10, 0, transitions.length, 3).values = transitions.map(([name, count]) => [name, count, "结构证据明确，排除原档"]);
summarySheet.getRange(`A11:C${10 + transitions.length}`).format.borders = { preset: "inside", style: "thin", color: lightBorder };

summarySheet.getRange("A19:G19").values = [["历史模型运行", "原标签准确率", "复核标签准确率", "变化", "复核后一档内", "复核后MAE", "复核后严重偏差"]];
styleHeader(summarySheet.getRange("A19:G19"));
const modelRows = Object.entries(summary.model_evaluations).map(([name, value]) => [
  name.replace("lite_physics_", "").replace("_1066", "").replace(".jsonl", ""),
  value.against_original.exact_rate,
  value.against_revised.exact_rate,
  value.against_revised.exact_rate - value.against_original.exact_rate,
  value.against_revised.within_one_rate,
  value.against_revised.mae,
  value.against_revised.severe,
]);
summarySheet.getRangeByIndexes(19, 0, modelRows.length, 7).values = modelRows;
summarySheet.getRange(`B20:E${19 + modelRows.length}`).format.numberFormat = "0.00%";
summarySheet.getRange(`F20:F${19 + modelRows.length}`).format.numberFormat = "0.0000";
summarySheet.getRange(`A20:G${19 + modelRows.length}`).format.borders = { preset: "inside", style: "thin", color: lightBorder };

summarySheet.getRange("A27:O30").merge();
summarySheet.getRange("A27").values = [[
  "解释：20道主标签修改只包含能够从题目任务结构中排除原档的明确错误；157道相邻边界题保留单一主标签，同时记录可接受等级。历史模型准确率的变化仅用于影响回放，不作为改标证据，因为这些题曾参与误差分析。",
]];
summarySheet.getRange("A27:O30").format = { fill: paleGold, font: { color: "#5D4500" }, wrapText: true, verticalAlignment: "center" };

const chart = summarySheet.charts.add("bar", summarySheet.getRange("D3:F8"));
chart.title = "原GPT标签与复核后标签分布";
chart.hasLegend = true;
chart.xAxis = { axisType: "textAxis", textStyle: { fontSize: 10 } };
chart.yAxis = { numberFormatCode: "0" };
chart.setPosition("H3", "O18");

summarySheet.showGridLines = false;
summarySheet.freezePanes.freezeRows(1);
summarySheet.getRange("A1:O30").format.verticalAlignment = "center";
summarySheet.getRange("A:A").format.columnWidth = 32;
summarySheet.getRange("B:B").format.columnWidth = 15;
summarySheet.getRange("C:C").format.columnWidth = 30;
summarySheet.getRange("D:F").format.columnWidth = 16;
summarySheet.getRange("G:G").format.columnWidth = 16;

writeDetailSheet(detailSheet, "1066题逐题标签复核底表", rows, "AllReviewTable");
writeDetailSheet(changedSheet, "20道明确主标签修改", rows.filter((row) => row["原GPT裁定档"] !== row["修订后主标签"]), "ChangedLabelsTable");
writeDetailSheet(boundarySheet, "157道相邻档边界题", rows.filter((row) => row["可接受等级"].includes("|")), "BoundaryLabelsTable");

styleTitle(methodSheet, "A1:C1", "标签复核口径与使用说明");
methodSheet.getRange("A3:C3").values = [["项目", "规则", "用途"]];
styleHeader(methodSheet.getRange("A3:C3"));
methodSheet.getRange("A4:C10").values = [
  ["主标签", "每题必须保留一个可用于严格评估的五档标签", "生产评估和混淆矩阵"],
  ["可接受等级", "仅在相邻两档均有充分结构证据时记录双档", "边界容忍评估与人工仲裁"],
  ["明确改标", "必须能依据题干与解析排除原档；模型多数票不能单独改标", "本次共20题"],
  ["送分/基础", "看是否只有一个唯一熟悉模板和一次透明映射", "不按选项数量机械定档"],
  ["基础/中等", "看是否形成3—4个连续决策、完整标准实验或共享过程", "独立基础判断不累加步骤"],
  ["中等/拔高", "看决定性反推、隐含转换、误差评价或5—6步迁移链", "题长和项目背景不是证据"],
  ["拔高/压轴", "看多对象、多状态、多约束是否在同一网络中耦合，并完成分类/边界筛选", "仅有范围或最大值不构成压轴"],
];
methodSheet.getRange("A4:C10").format = { wrapText: true, verticalAlignment: "top", borders: { preset: "inside", style: "thin", color: lightBorder } };
methodSheet.getRange("A:A").format.columnWidth = 20;
methodSheet.getRange("B:B").format.columnWidth = 72;
methodSheet.getRange("C:C").format.columnWidth = 32;
methodSheet.showGridLines = false;

await fs.mkdir(`${root}output/spreadsheet`, { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);

const inspect = await workbook.inspect({
  kind: "sheet,table,drawing",
  maxChars: 10000,
  tableMaxRows: 6,
  tableMaxCols: 10,
});
process.stdout.write(`${inspect.ndjson}\n`);

const preview = await workbook.render({ sheetName: "复核汇总", range: "A1:O30", scale: 1.2, format: "png" });
await fs.writeFile(`${root}tmp/physics_gpt56_labels_rereview_summary.png`, new Uint8Array(await preview.arrayBuffer()));
for (const [sheetName, range, suffix] of [
  ["1066题逐题复核", "A1:J10", "all"],
  ["主标签修改20", "A1:J12", "changed"],
  ["相邻边界157", "A1:J10", "boundary"],
  ["复核口径", "A1:C10", "method"],
]) {
  const sheetPreview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  await fs.writeFile(`${root}tmp/physics_gpt56_labels_rereview_${suffix}.png`, new Uint8Array(await sheetPreview.arrayBuffer()));
}
process.stdout.write(`${outputPath}\n`);
