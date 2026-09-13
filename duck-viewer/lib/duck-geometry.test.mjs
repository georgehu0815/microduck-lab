import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const require = createRequire(import.meta.url);
const source = readFileSync(new URL("../components/Duck.tsx", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    esModuleInterop: true,
    jsx: ts.JsxEmit.ReactJSX,
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
  },
});

function loadDuckModule() {
  const exports = {};
  const dependencies = {
    "@/lib/assign": { assignDrag: {} },
    "@/lib/record": { captureWantsCleanFrame: () => false },
    "@/lib/select": { getSelectedDuck: () => null },
    "@/lib/ui": { getDuckLabels: () => true },
    "@react-three/drei": { Html: () => null },
    "@react-three/fiber": { useFrame: () => undefined },
  };
  vm.runInNewContext(outputText, {
    exports,
    require: (id) => dependencies[id] ?? require(id),
  });
  return exports;
}

test("body geometry merges streamed meshes with primitives and recomputes normals", () => {
  const { buildBodyGeometries } = loadDuckModule();
  const [body] = buildBodyGeometries({
    bodies: ["mixed"],
    meshes: [{ v: [0, 0, 0, 1, 0, 0, 0, 1, 0], f: [0, 1, 2] }],
    geoms: [{
      mesh: 0,
      body: 0,
      pos: [0, 0, 0],
      quat: [1, 0, 0, 0],
      rgba: [1, 1, 1, 1],
    }],
    primitives: [{
      type: "box",
      body: 0,
      pos: [0, 0, 0],
      quat: [1, 0, 0, 0],
      size: [0.1, 0.1, 0.1],
      rgba: [1, 0, 0, 1],
      mat: "",
    }],
  });
  assert.ok(body.geometry);
  assert.ok(body.geometry.getAttribute("position").count > 3);
  assert.equal(
    body.geometry.getAttribute("normal").count,
    body.geometry.getAttribute("position").count
  );
  assert.equal(
    body.geometry.getAttribute("color").count,
    body.geometry.getAttribute("position").count
  );
  assert.equal(body.geometry.getAttribute("uv"), undefined);
  body.geometry.dispose();
});

test("basketball material falls back to orange when its texture is unavailable", () => {
  const { buildBodyGeometries } = loadDuckModule();
  const [body] = buildBodyGeometries({
    bodies: ["basketball"],
    meshes: [],
    geoms: [],
    primitives: [{
      type: "sphere",
      body: 0,
      pos: [0, 0, 0],
      quat: [1, 0, 0, 0],
      size: [0.12],
      rgba: [1, 1, 1, 1],
      mat: "basketball_mat",
    }],
  });
  const colors = body.geometry.getAttribute("color").array;
  const { Color } = require("three");
  const expected = new Color("#d87519");
  assert.ok(Math.abs(colors[0] - expected.r) < 1e-6);
  assert.ok(Math.abs(colors[1] - expected.g) < 1e-6);
  assert.ok(Math.abs(colors[2] - expected.b) < 1e-6);
  body.geometry.dispose();
});
