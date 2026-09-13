import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("./wingpod-live.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const exports = {};
vm.runInNewContext(outputText, { exports });
const { parseWingPodLiveState, validateWingPodMutation } = exports;

function state() {
  return {
    case_id: "wingpod-tennis-v1",
    seed: 0,
    step: 2,
    time: 0.04,
    contract: "wingpod-tennis-live-v1",
    observation_dim: 66,
    action_dim: 15,
    controller: "authored_ik_fsm_teacher_not_ppo",
    stage: "approach",
    terminated: false,
    truncated: false,
    hardware_enabled: false,
    joints: Array(15).fill(0),
    geoms: [{ name: "camera", type: "ellipsoid", size: [0.1, 0.2, 0.3], pos: [0, 0, 0], quat: [1, 0, 0, 0], rgba: [1, 1, 1, 1] }],
    metrics: {
      simulation_only: true,
      trained_policy: false,
      vision_control: false,
      task_success: false,
      failure_reason: null,
      controller_stage: "settle",
      ball_position_m: [0.1, 0.2, 0.03],
      ball_inside_bin: false,
      tcp_ball_distance_m: 0.2,
      carry_distance_m: 0,
    },
  };
}

test("accepts the fixed WingPod live MuJoCo contract", () => {
  assert.ok(parseWingPodLiveState(state()));
});

test("rejects policy and hardware overclaims", () => {
  for (const key of ["trained_policy", "vision_control"]) {
    const candidate = state();
    candidate.metrics[key] = true;
    assert.equal(parseWingPodLiveState(candidate), null);
  }
  const hardware = state();
  hardware.hardware_enabled = true;
  assert.equal(parseWingPodLiveState(hardware), null);
});

test("bounds WingPod reset and real-time stepping requests", () => {
  assert.deepEqual({ ...validateWingPodMutation("wingpod-reset", { seed: 7 }) }, { seed: 7 });
  assert.deepEqual({ ...validateWingPodMutation("wingpod-teacher", { ticks: 2 }) }, { ticks: 2 });
  assert.equal(validateWingPodMutation("wingpod-teacher", { ticks: 11 }), null);
  assert.equal(validateWingPodMutation("wingpod-reset", { seed: -1 }), null);
});