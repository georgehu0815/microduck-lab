import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("scene-layout.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const exports = {};
vm.runInNewContext(outputText, { exports });
const { gridOffsets } = exports;

test("custom-scene rosters use one wide spacing contract", () => {
  const standard = gridOffsets([{ sceneKey: undefined }, { sceneKey: undefined }]);
  const custom = gridOffsets([{ sceneKey: "basketball" }, { sceneKey: "bridge" }]);
  assert.equal(standard[1][0] - standard[0][0], 0.65);
  assert.equal(custom[1][0] - custom[0][0], 2.6);
});

test("numeric callers retain standard spacing", () => {
  const offsets = gridOffsets(2);
  assert.equal(offsets[1][0] - offsets[0][0], 0.65);
});
