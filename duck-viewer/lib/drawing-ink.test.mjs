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

test("drawing ink uses actual xyz points and breaks between stroke ids", () => {
  const { writeDrawingSegments } = loadDuckModule();
  const target = new Float32Array(36);
  const count = writeDrawingSegments(
    {
      contract: "microduck-drawing-v1",
      assistance: 0,
      points: [
        [0.1597, -0.02, 0.21, 1],
        [0.1597, -0.01, 0.22, 1],
        [0.1597, 0.01, 0.23, 2],
        [0.1597, 0.02, 0.24, 2],
      ],
    },
    target
  );

  assert.equal(count, 4);
  assert.deepEqual(
    Array.from(target.slice(0, 12)),
    [
      0.15970000624656677, -0.019999999552965164, 0.20999999344348907,
      0.15970000624656677, -0.009999999776482582, 0.2199999988079071,
      0.15970000624656677, 0.009999999776482582, 0.23000000417232513,
      0.15970000624656677, 0.019999999552965164, 0.23999999463558197,
    ]
  );
});

test("drawing ink rejects assisted or unknown contracts", () => {
  const { writeDrawingSegments } = loadDuckModule();
  const target = new Float32Array(12);
  const points = [[0.1597, 0, 0.2, 1], [0.1597, 0.1, 0.2, 1]];
  assert.equal(
    writeDrawingSegments(
      { contract: "microduck-drawing-v1", assistance: 1, points },
      target
    ),
    0
  );
  assert.equal(
    writeDrawingSegments(
      { contract: "microduck-drawing-v0", assistance: 0, points },
      target
    ),
    0
  );
});

test("brush ink uses palette colors and breaks on stroke or color changes", () => {
  const { writeDrawingSegments } = loadDuckModule();
  const positions = new Float32Array(48);
  const colors = new Float32Array(48);
  const count = writeDrawingSegments(
    {
      contract: "microduck-brush-v2",
      assistance: 0,
      brush_radius: 0.001,
      palette: [
        [0.95, 0.69, 0.08],
        [0.96, 0.32, 0.05],
        [0.13, 0.12, 0.1],
        [0.1, 0.49, 0.84],
      ],
      points: [
        [0.1597, -0.03, 0.21, 1],
        [0.1597, -0.02, 0.22, 1],
        [0.1597, -0.01, 0.23, 1],
        [0.1597, 0.0, 0.24, 2],
        [0.1597, 0.01, 0.25, 2],
      ],
      colors: [0, 0, 1, 1, 1],
    },
    positions,
    colors
  );

  assert.equal(count, 4);
  const linear = (channel) =>
    channel <= 0.04045
      ? channel / 12.92
      : ((channel + 0.055) / 1.055) ** 2.4;
  const expected = [
    ...[0.95, 0.69, 0.08].map(linear),
    ...[0.95, 0.69, 0.08].map(linear),
    ...[0.96, 0.32, 0.05].map(linear),
    ...[0.96, 0.32, 0.05].map(linear),
  ];
  Array.from(colors.slice(0, 12)).forEach((value, index) => {
    assert.ok(Math.abs(value - expected[index]) < 1e-6);
  });
});

test("drawing ink keeps a full brush trace within the 6000 point bound", () => {
  const { writeDrawingSegments } = loadDuckModule();
  const points = Array.from(
    { length: 3500 },
    (_, index) => [0.1597, index / 100000, 0.21, 1]
  );
  const count = writeDrawingSegments(
    {
      contract: "microduck-brush-v2",
      assistance: 0,
      brush_radius: 0.001,
      palette: [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
      points,
      colors: Array(points.length).fill(0),
    },
    new Float32Array((6000 - 1) * 6),
    new Float32Array((6000 - 1) * 6)
  );
  assert.equal(count, (points.length - 1) * 2);
});

test("drawing ink material consumes generated vertex colors", () => {
  assert.match(source, /<lineBasicMaterial\s+vertexColors/);
});

test("incomplete brush payload from an older Lab fails closed without crashing", () => {
  const { writeDrawingSegments } = loadDuckModule();
  const payload = { contract: "microduck-brush-v2", assistance: 0,
    points: [[.1597, 0, .22, 1], [.1597, .001, .22, 1]] };
  assert.equal(writeDrawingSegments(payload, new Float32Array(12)), 0);
});
