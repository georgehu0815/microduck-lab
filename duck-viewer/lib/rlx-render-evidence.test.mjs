import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import vm from "node:vm";
import { createRequire } from "node:module";
import { test } from "node:test";
import ts from "typescript";

const loadDependency = createRequire(import.meta.url);

function loadEvidence() {
  const source = fs.readFileSync(
    new URL("rlx-render-evidence.ts", import.meta.url),
    "utf8"
  );
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      esModuleInterop: true,
    },
  });
  const exports = {};
  vm.runInNewContext(
    outputText,
    { exports, process, require: loadDependency },
    { filename: "rlx-render-evidence.ts" }
  );
  return exports;
}

const {
  prepareRenderEvidence,
  finalizeRenderEvidence,
  readRenderEvidence,
} = loadEvidence();

function fixture(t, checkpoint = false) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "render-evidence-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const source = path.join(
    root,
    checkpoint ? "dance.safetensors" : "dance.onnx"
  );
  const sidecar = `${source}.json`;
  const clipPath = path.join(root, "dance.json");
  const externalPolicy = path.join(root, "alpha_stand.onnx");
  const video = path.join(root, "render", "ep0.mp4");
  const sheet = path.join(root, "render", "ep0_sheet.png");
  const receipt = path.join(root, "render", "evidence.json");
  fs.mkdirSync(path.dirname(video), { recursive: true });
  fs.writeFileSync(source, "source bytes");
  if (checkpoint) fs.writeFileSync(sidecar, "sidecar bytes");
  fs.writeFileSync(clipPath, "clip bytes");
  fs.writeFileSync(externalPolicy, "stand policy bytes");
  fs.writeFileSync(video, "video bytes");
  fs.writeFileSync(sheet, "sheet bytes");
  const sourceFiles = checkpoint ? [source, sidecar] : [source];
  const preparation = {
    sourceFiles,
    source,
    recipeKey: "dance/full/seed=42",
    clipPath,
    externalFiles: [],
  };
  const output = { video, sheet, receipt };
  const validation = { ...preparation, video, sheet };
  return { ...preparation, ...output, validation, sourceFiles, sidecar, externalPolicy };
}

function prepareAndFinalize(f) {
  const context = prepareRenderEvidence(f);
  const receipt = finalizeRenderEvidence(context, f);
  assert.ok(receipt);
  return receipt;
}

test("binds render media to ONNX, clip, and recipe bytes", (t) => {
  const f = fixture(t);
  const receipt = prepareAndFinalize(f);

  assert.equal(receipt.version, 1);
  assert.equal(receipt.source, f.source);
  assert.equal(receipt.sourceFiles.length, 1);
  assert.equal(receipt.video.path, f.video);
  assert.equal(receipt.sheet.path, f.sheet);
  assert.ok(fs.existsSync(f.receipt));
  assert.ok(readRenderEvidence(f.receipt, f.validation));
  assert.equal(
    fs.readdirSync(path.dirname(f.receipt)).some((name) => name.endsWith(".tmp")),
    false
  );
});

test("checkpoint evidence includes and validates its metadata sidecar", (t) => {
  const f = fixture(t, true);
  f.sourceFiles.reverse();
  const receipt = prepareAndFinalize(f);

  assert.equal(receipt.sourceFiles.length, 2);
  assert.ok(receipt.sourceFilesSha256[f.sidecar]);
  assert.ok(
    readRenderEvidence(f.receipt, {
      ...f.validation,
      sourceFiles: [...f.sourceFiles].reverse(),
    })
  );
  fs.writeFileSync(f.sidecar, "changed sidecar bytes");
  assert.equal(readRenderEvidence(f.receipt, f.validation), null);
});

test("supports renders without a choreography clip", (t) => {
  const f = fixture(t);
  f.clipPath = null;
  f.validation.clipPath = null;
  const receipt = prepareAndFinalize(f);

  assert.equal(receipt.clipPath, null);
  assert.equal(receipt.clipSha256, null);
  assert.ok(readRenderEvidence(f.receipt, f.validation));
});

test("binds Backflip render evidence to the external pretrained stand policy", (t) => {
  const f = fixture(t);
  f.externalFiles = [f.externalPolicy];
  f.validation.externalFiles = [f.externalPolicy];
  const recipeOptions = {
    backflip_protocol_version: "spotter-launch-landing-stand-v2",
    stand_policy_sha256: "stand-hash",
  };
  f.validation.recipeOptions = recipeOptions;
  const context = prepareRenderEvidence(f);
  const receipt = finalizeRenderEvidence(context, f, recipeOptions);
  assert.ok(receipt);

  assert.equal(receipt.externalFiles.length, 1);
  assert.ok(receipt.externalFilesSha256[f.externalPolicy]);
  assert.deepEqual(
    JSON.parse(JSON.stringify(receipt.recipeOptions)),
    recipeOptions
  );
  assert.ok(readRenderEvidence(f.receipt, f.validation));
  assert.equal(
    readRenderEvidence(f.receipt, {
      ...f.validation,
      recipeOptions: {
        ...recipeOptions,
        backflip_protocol_version: "spotter-launch-landing-stand-v1",
      },
    }),
    null
  );
  fs.writeFileSync(f.externalPolicy, "updated stand policy bytes");
  assert.equal(readRenderEvidence(f.receipt, f.validation), null);
});

for (const [label, mutate] of [
  ["source", (f) => fs.writeFileSync(f.source, "changed source bytes")],
  ["video", (f) => fs.writeFileSync(f.video, "changed video bytes")],
  ["sheet", (f) => fs.writeFileSync(f.sheet, "changed sheet bytes")],
  ["clip", (f) => fs.writeFileSync(f.clipPath, "changed clip bytes")],
]) {
  test(`invalidates evidence when ${label} bytes change`, (t) => {
    const f = fixture(t);
    prepareAndFinalize(f);
    mutate(f);
    assert.equal(readRenderEvidence(f.receipt, f.validation), null);
  });
}

test("invalidates evidence when settings or owned paths change", (t) => {
  const f = fixture(t);
  prepareAndFinalize(f);

  assert.equal(
    readRenderEvidence(f.receipt, {
      ...f.validation,
      recipeKey: "dance/smoke/seed=42",
    }),
    null
  );
  assert.equal(
    readRenderEvidence(f.receipt, {
      ...f.validation,
      clipPath: null,
    }),
    null
  );
  assert.equal(
    readRenderEvidence(f.receipt, {
      ...f.validation,
      sourceFiles: [f.source, f.sidecar],
    }),
    null
  );
});

test("missing files and legacy or malformed receipts are unbound", (t) => {
  const f = fixture(t);
  assert.equal(readRenderEvidence(f.receipt, f.validation), null);
  fs.writeFileSync(f.receipt, JSON.stringify({ rendered: true }));
  assert.equal(readRenderEvidence(f.receipt, f.validation), null);
  fs.writeFileSync(f.receipt, "false");
  assert.equal(readRenderEvidence(f.receipt, f.validation), null);
  fs.writeFileSync(f.receipt, "null");
  assert.equal(readRenderEvidence(f.receipt, f.validation), null);

  prepareAndFinalize(f);
  fs.rmSync(f.sheet);
  assert.equal(readRenderEvidence(f.receipt, f.validation), null);
});

test("finalize refuses to claim media after source or clip changes", (t) => {
  for (const changed of ["source", "clip"]) {
    const f = fixture(t);
    const context = prepareRenderEvidence(f);
    fs.writeFileSync(
      changed === "source" ? f.source : f.clipPath,
      `changed ${changed} bytes`
    );
    assert.equal(finalizeRenderEvidence(context, f), null);
    assert.equal(fs.existsSync(f.receipt), false);
  }
});

test("finalize refuses missing media and prepare requires owned source files", (t) => {
  const f = fixture(t);
  const context = prepareRenderEvidence(f);
  fs.rmSync(f.video);
  assert.equal(finalizeRenderEvidence(context, f), null);
  assert.equal(fs.existsSync(f.receipt), false);
  assert.throws(
    () =>
      prepareRenderEvidence({
        ...f,
        sourceFiles: [],
      }),
    /included in sourceFiles/
  );
});
