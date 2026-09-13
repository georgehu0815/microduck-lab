import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

function load(name) {
  const source = readFileSync(new URL(name, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, { compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 } });
  const exports = {};
  vm.runInNewContext(outputText, { exports });
  return exports;
}

const { skillEvidence, rolloutMetricRanges, evidenceLabel } = load("studio-evidence.ts");
const { evaluationVerdict } = load("evaluation.ts");

test("evidence shows worst/best episodes without substituting average reward for acceptance", () => {
  const evidence = skillEvidence({ mean_return: 9999, dance_assessment: { episodes: [
    { passed: true, pose_rmse_rad: .08, dynamic_gain: .4 },
    { passed: false, pose_rmse_rad: .3, dynamic_gain: -.1 },
    { pose_rmse_rad: null },
  ], criteria: { max_pose_rmse_rad: .15 } } }, "dance");
  assert.equal(evidence.passed, 1);
  assert.equal(evidence.episodes.length, 3);
  assert.equal(evidence.ranges[0].maximum, .3);
  assert.equal(evidence.ranges[0].measured, 2);
  assert.equal(evidence.criteria.max_pose_rmse_rad, .15);
  assert.equal(evidenceLabel("pose_rmse_rad"), "pose rmse rad");
});

test("missing, malformed and nonfinite measurements stay unmeasured", () => {
  assert.equal(skillEvidence(null, "swing").episodes.length, 0);
  assert.equal(skillEvidence({ swing_assessment: { episodes: [null, 4, { max_alignment: Infinity }] } }, "swing").ranges.length, 0);
  assert.equal(skillEvidence({ dance_assessment: { episodes: [] } }, "running").episodes.length, 0);
});

test("Backflip evidence uses the authoritative assessment and landing handoff metrics", () => {
  const evidence = skillEvidence({ backflip_assessment: {
    criteria: { min_stand_hold_seconds: 2, min_landing_policy_steps: 10 },
    episodes: [
      {
        passed: true,
        rotation_rad: 6.4,
        landing_policy_steps: 18,
        stand_hold_seconds: 2.4,
        min_stand_height_m: 0.11,
        stand_upright_fraction: 0.98,
        assist_after_release_steps: 0,
        assist_force_n: 12.5,
        assist_torque_nm: 0.42,
      },
      {
        passed: false,
        rotation_rad: 5.9,
        landing_policy_steps: 9,
        stand_hold_seconds: 1.2,
        min_stand_height_m: 0.08,
        stand_upright_fraction: 0.75,
        assist_after_release_steps: 1,
        assist_force_n: 9.5,
        assist_torque_nm: 0.31,
      },
    ],
  } }, "backflip");
  assert.equal(evidence.passed, 1);
  assert.equal(evidence.episodes.length, 2);
  assert.equal(evidence.criteria.min_stand_hold_seconds, 2);
  assert.deepEqual(
    JSON.parse(JSON.stringify(evidence.ranges.map(({ metric }) => metric))),
    [
      "rotation_rad",
      "landing_policy_steps",
      "stand_hold_seconds",
      "min_stand_height_m",
      "stand_upright_fraction",
      "assist_after_release_steps",
      "assist_force_n",
      "assist_torque_nm",
    ]
  );
});

test("Backflip actual-unit assistance channels use rollout metric ranges", () => {
  const ranges = rolloutMetricRanges({
    recipe_metrics: {
      assist_force_n: { min: 0, mean: 0.7, max: 10.5 },
      assist_torque_nm: { min: 0, mean: 0.02, max: 1.1 },
      malformed: { min: 0, max: Infinity },
    },
  }, ["assist_force_n", "assist_torque_nm", "malformed"]);
  assert.deepEqual(JSON.parse(JSON.stringify(ranges)), [
    { metric: "assist_force_n", minimum: 0, maximum: 10.5 },
    { metric: "assist_torque_nm", minimum: 0, maximum: 1.1 },
  ]);
});

test("acceptance requires skill mode, pipeline validity, matching scenario and settings", () => {
  const passing = { recipe: "running", evaluation_mode: "skill", skill_status: "passed", pipeline_passed: true, passed: true };
  assert.equal(evaluationVerdict(passing, "running").taskPassed, true);
  for (const override of [{ pipeline_passed: false }, { recipe: "dance" }, { recipe: undefined }, { evaluation_mode: "pipeline" }, { evaluation_settings_match: false }, { passed: false }]) {
    assert.equal(evaluationVerdict({ ...passing, ...override }, "running").taskPassed, false);
  }
  assert.equal(evaluationVerdict({ ...passing, recipe: "backflip" }, "backflip").taskPassed, true);
});

test("Basketball exposes verified balance-only evidence without passing steering", () => {
  const report = {
    recipe: "basketball",
    evaluation_mode: "skill",
    skill_status: "failed",
    pipeline_passed: true,
    passed: false,
    report: { protocol: { seconds: 60 } },
    basketball_assessment: {
      cases: [
        {
          classification: "zero_command_balance",
          passed: true,
          trial_count: 3,
          sustained_balance_count: 3,
        },
        {
          classification: "commanded_rolling",
          passed: false,
        },
      ],
    },
  };
  const verdict = evaluationVerdict(report, "basketball");
  assert.equal(verdict.balanceOnly, true);
  assert.equal(verdict.basketballBalancePassed, true);
  assert.equal(verdict.basketballSteeringPassed, false);
  assert.equal(verdict.taskPassed, false);

  assert.equal(evaluationVerdict({
    ...report,
    passed: true,
    skill_status: "passed",
    basketball_assessment: {
      cases: [
        report.basketball_assessment.cases[0],
        ...["forward", "lateral", "yaw"].map(() => ({
          classification: "commanded_rolling",
          passed: true,
        })),
      ],
    },
  }, "basketball").taskPassed, false);
  assert.equal(evaluationVerdict({
    ...report,
    report: { protocol: { seconds: 59.99 } },
  }, "basketball").balanceOnly, false);
});

test("Bridge requires an explicit unassisted bridge assessment", () => {
  const report = {
    recipe: "bridge",
    evaluation_mode: "skill",
    skill_status: "passed",
    pipeline_passed: true,
    passed: true,
  };
  assert.equal(evaluationVerdict(report, "bridge").taskPassed, false);
  assert.equal(evaluationVerdict({
    ...report,
    bridge_assessment: {
      passed: true,
      criteria: { requires_unassisted: true },
      episodes: [{ assisted_steps: 1 }],
    },
  }, "bridge").taskPassed, false);
  assert.equal(evaluationVerdict({
    ...report,
    bridge_assessment: {
      passed: true,
      criteria: { requires_unassisted: true },
      episodes: [{ assisted_steps: 0 }],
    },
  }, "bridge").taskPassed, true);
});

test("Drawing requires explicit passed and unassisted assessment fields", () => {
  const report = {
    recipe: "drawing",
    evaluation_mode: "skill",
    skill_status: "passed",
    pipeline_passed: true,
    passed: true,
    drawing_assessment: {
      passed: true,
      unassisted: true,
      episodes: [{
        passed: true,
        coverage: 0.91,
        precision: 0.88,
        symmetric_chamfer_m: 0.0007,
        length_ratio: 1.03,
        ink_contact_fraction: 0.62,
      }],
    },
  };
  const verdict = evaluationVerdict(report, "drawing");
  assert.equal(verdict.drawingUnassisted, true);
  assert.equal(verdict.taskPassed, true);
  assert.equal(evaluationVerdict({
    ...report,
    drawing_assessment: { ...report.drawing_assessment, unassisted: false },
  }, "drawing").taskPassed, false);
  assert.equal(evaluationVerdict({
    ...report,
    drawing_assessment: { ...report.drawing_assessment, passed: false },
  }, "drawing").taskPassed, false);

  const evidence = skillEvidence(report, "drawing");
  assert.equal(evidence.passed, 1);
  assert.deepEqual(
    JSON.parse(JSON.stringify(evidence.ranges.map(({ metric }) => metric))),
    ["coverage", "precision", "symmetric_chamfer_m", "length_ratio", "ink_contact_fraction"]
  );
});
