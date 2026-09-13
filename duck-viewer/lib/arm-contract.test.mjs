import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("arm-contract.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const exports = {};
vm.runInNewContext(outputText, { exports, Set, Object, Array, Number });
const {
  isSafeArtifactPath,
  parseArmRuns,
  parseArmState,
  validateResetBody,
  validateStepBody,
  validateTeacherBody,
} = exports;
const plain = (value) => JSON.parse(JSON.stringify(value));

const state = {
  case_id: "arm-reach-v1",
  seed: 7,
  step: 2,
  time: 0.04,
  contract: "md-arm-table-v1",
  observation_dim: 66,
  action_dim: 6,
  controller: "teacher",
  stage: "reach",
  terminated: false,
  truncated: false,
  hardware_enabled: false,
  joints: [0, 0, 0, 0, 0, 0],
  geoms: [{
    name: "base",
    type: "box",
    size: [0.1, 0.1, 0.1],
    pos: [0, 0, 0.1],
    quat: [1, 0, 0, 0],
    rgba: [0.2, 0.3, 0.4, 1],
  }],
  metrics: { error_m: 0.02 },
};

test("accepts exact reset, step, and teacher request shapes", () => {
  assert.deepEqual(plain(validateResetBody({ case_id: "arm-reach-v1", seed: 3 })), {
    case_id: "arm-reach-v1",
    seed: 3,
  });
  assert.deepEqual(plain(validateStepBody({ action: [0, 0, 0, 0, 0, 0] })), {
    action: [0, 0, 0, 0, 0, 0],
  });
  assert.deepEqual(plain(validateTeacherBody({ ticks: 50 })), { ticks: 50 });
});

test("rejects malformed or non-finite mutations", () => {
  assert.equal(validateResetBody({ case_id: "unknown", seed: 0 }), null);
  assert.equal(validateStepBody({ action: [0, 0, 0, 0, 0, Number.NaN] }), null);
  assert.equal(validateStepBody({ action: new Array(7).fill(0) }), null);
  assert.equal(validateTeacherBody({ ticks: 51 }), null);
});

test("validates state dimensions against the selected case", () => {
  assert.ok(parseArmState(state));
  assert.equal(parseArmState({ ...state, observation_dim: 116 }), null);
  assert.equal(parseArmState({ ...state, hardware_enabled: true }), null);
});

test("validates run evidence without inventing verdicts", () => {
  assert.ok(parseArmRuns({
    hardware_enabled: false,
    runs: [{
      case_id: "arm-reach-v1",
      run_id: "teacher-7",
      path: "runs/teacher-7",
      controller: "teacher",
      passed: null,
      metrics: {},
      video_path: "runs/teacher-7/rollout.mp4",
    }],
  }));
  assert.equal(parseArmRuns({
    hardware_enabled: false,
    runs: [{ case_id: "arm-reach-v1", passed: "yes" }],
  }), null);
});

test("artifact paths are relative and traversal-free", () => {
  assert.equal(isSafeArtifactPath("runs/a/rollout.mp4"), true);
  assert.equal(isSafeArtifactPath("../secret"), false);
  assert.equal(isSafeArtifactPath("/tmp/video.mp4"), false);
  assert.equal(isSafeArtifactPath("http://host/video.mp4"), false);
  assert.equal(isSafeArtifactPath("runs\\video.mp4"), false);
});
