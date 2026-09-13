import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("live-contract.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const exports = {};
vm.runInNewContext(outputText, { exports });
const { liveContractLabels } = exports;
const drawing = { drawing: { contract: "microduck-drawing-v1", assistance: 0, points: [] } };
const brush = {
  drawing: {
    contract: "microduck-brush-v2",
    assistance: 0,
    points: [],
    colors: [],
    palette: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
    brush_radius: 0.001,
  },
};

for (const [name, roster, observations, actions] of [
  ["empty", [], "—", "—"],
  ["original seven", Array.from({ length: 7 }, () => ({})), "61", "14"],
  ["drawing only", [drawing], "83", "15"],
  ["mixed", [{}, drawing], "61 / 83", "14 / 15"],
  ["brush only", [brush], "93", "15"],
  ["pencil and brush", [drawing, brush], "83 / 93", "15"],
  ["all live contracts", [{}, drawing, brush], "61 / 83 / 93", "14 / 15"],
]) {
  test(`live contract labels follow ${name} roster, not selected recipe`, () => {
    const labels = liveContractLabels(roster);
    assert.equal(labels.observations, observations);
    assert.equal(labels.actions, actions);
  });
}
