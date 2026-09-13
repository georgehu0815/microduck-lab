import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("../app/api/rlx/runs/route.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
});

test("Studio catalog rescans checkpoint and ONNX-only runs across scenarios without including empty or escaped directories", async () => {
  const files = new Map([
    ["/studio/dance/first/checkpoint", 1000],
    ["/studio/running/export-only/onnx", 3000],
  ]);
  const directories = { dance: ["first", "empty", "linked"], running: ["export-only"] };
  const artifactPath = (scenario, runName, kind) => `/studio/${scenario}/${runName}/${kind}`;
  const dependencies = {
    "node:fs/promises": {
      readdir: async (root) => directories[path.basename(root)].map((name) => ({ name, isDirectory: () => true })),
      realpath: async (directory) => directory.endsWith("/linked") ? "/outside" : directory,
      stat: async (filename) => {
        if (!files.has(filename)) throw new Error("ENOENT");
        return { mtime: new Date(files.get(filename)) };
      },
    },
    "node:path": path,
    "next/server": { NextResponse: { json: (data, options) => Response.json(data, options) } },
    "@/lib/experiments": { EXPERIMENTS: [{ id: "dance" }, { id: "running" }] },
    "@/lib/evaluation": { evaluationVerdict: () => ({ taskPassed: false }) },
    "@/lib/rlx-job": {
      artifactPath,
      sanitizeRunName: (name) => name,
      snapshot: async (scenario, runName) => ({
        artifacts: {
          onnx: files.has(artifactPath(scenario, runName, "onnx")),
          checkpoint: files.has(artifactPath(scenario, runName, "checkpoint")),
          renderVideo: false,
        },
      }),
    },
  };
  const exports = {};
  vm.runInNewContext(outputText, { exports, require: (id) => dependencies[id] });
  const first = await (await exports.GET()).json();
  assert.deepEqual(first.runs.map((run) => run.runName), ["export-only", "first"]);
  assert.equal(first.runs[0].onnx, true);
  assert.equal(first.runs[0].checkpoint, false);
  assert.equal(first.runs[0].policyModifiedAt, new Date(3000).toISOString());
  files.set("/studio/dance/new-training/checkpoint", 5000);
  directories.dance.push("new-training");
  const refreshed = await (await exports.GET()).json();
  assert.deepEqual(refreshed.runs.map((run) => run.runName), ["new-training", "export-only", "first"]);
  assert.equal(refreshed.runs[0].trainedAt, new Date(5000).toISOString());
});
