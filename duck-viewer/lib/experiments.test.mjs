import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("experiments.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const exports = {};
vm.runInNewContext(outputText, { exports, Math });

const { defaultRewardWeights, EXPERIMENTS, getDrawingTool, getExperiment, isExperimentId } = exports;

test("Studio exposes exactly eight scenarios with honest diagnostic readiness", () => {
  const ids = ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge", "drawing"];
  assert.deepEqual(
    ids.filter((id) => isExperimentId(id)),
    ids
  );
  assert.equal(isExperimentId("jumping"), false);
  assert.equal(EXPERIMENTS.length, 8);

  const basketball = getExperiment("basketball");
  assert.equal(basketball.commandAdapter, "basketball");
  assert.equal(basketball.defaultRunName, "basketball-balance-01");
  assert.deepEqual(JSON.parse(JSON.stringify(basketball.artifactFiles)), {
    checkpoint: "checkpoint.pt",
    metadata: "checkpoint.pt.json",
    onnx: "policy.onnx",
    evaluation: "eval.json",
    renderSheet: "contact-sheet.png",
    renderVideo: "rollout.mp4",
  });
  assert.match(basketball.readiness, /BALANCE ONLY/);
  assert.match(basketball.readiness, /steering not passed/i);
  assert.match(basketball.policyContract, /Recurrent ONNX/);
  assert.deepEqual(JSON.parse(JSON.stringify(defaultRewardWeights("basketball"))), {});

  const bridge = getExperiment("bridge");
  assert.equal(bridge.defaultRunName, "bridge-studio-01");
  assert.equal(bridge.fullTimesteps, 32_768);
  assert.equal(bridge.fullEnvs, 4);
  assert.equal(bridge.fullNumSteps, 128);
  assert.equal(bridge.fullNumMinibatches, 4);
  assert.equal(bridge.maxEpisodeSeconds, 20);
  assert.equal(bridge.seed, 7);
  assert.equal(bridge.initialStd, 0.03);
  assert.deepEqual(JSON.parse(JSON.stringify(bridge.fullEnvironment)), {
    domainRand: false,
    obsNoise: false,
    actionDelay: false,
    randomYaw: false,
  });
  assert.deepEqual(JSON.parse(JSON.stringify(bridge.ppo)), {
    learningRate: 0.00001,
    gamma: 0.99,
    clipCoefficient: 0.1,
    updateEpochs: 2,
    entropyCoefficient: 0,
    maxGradNorm: 0.5,
  });
  assert.equal(bridge.metricKey, "bridge_progress_m");
  assert.match(bridge.readiness, /TRAINING PILOT/);
  assert.match(bridge.readiness, /crossing not yet verified/i);
  assert.match(bridge.guidance.steps.join(" "), /1,000 control steps/);
  assert.match(bridge.guidance.distinctions.join(" "), /actor warm-starts with a fresh critic and optimizer/i);
  assert.deepEqual(JSON.parse(JSON.stringify(defaultRewardWeights("bridge"))), {});

  const drawing = getExperiment("drawing");
  assert.equal(drawing.commandAdapter, "drawing");
  assert.equal(drawing.defaultRunName, "drawing-pilot-01");
  assert.equal(drawing.fullTimesteps, 49_152);
  assert.equal(drawing.fullNumSteps, 6_144);
  assert.equal(drawing.fullNumMinibatches, 12);
  assert.equal(drawing.initialStd, 0.0003);
  assert.deepEqual(JSON.parse(JSON.stringify(drawing.ppo)), {
    learningRate: 0.0000001,
    gamma: 0.995,
    clipCoefficient: 0.05,
    updateEpochs: 2,
    entropyCoefficient: 0,
    maxGradNorm: 0.3,
  });
  assert.equal(drawing.policyContract.includes("microduck-brush-v2"), true);
  assert.equal(drawing.policyContract.includes("93"), true);
  assert.equal(drawing.policyContract.includes("15"), true);
  assert.deepEqual(JSON.parse(JSON.stringify(drawing.artifactFiles)), {
    checkpoint: "policy.zip",
    metadata: "policy.zip.json",
    onnx: "policy.onnx",
    evaluation: "eval.json",
    renderSheet: "frame_sheet.png",
    renderVideo: "rollout.mp4",
  });
  assert.equal(drawing.preview, "/experiments/drawing/preview.svg");
  assert.equal(drawing.poster, "/experiments/drawing/poster.jpg");
  assert.equal(drawing.video, "/experiments/drawing/rollout.mp4");
  assert.match(drawing.readiness, /RUN-SPECIFIC/);
  assert.match(drawing.readiness, /saved evaluation/i);
  assert.doesNotMatch(drawing.readiness, /no accepted learned drawing result/i);
  assert.match(drawing.result, /drawing_assessment\.passed/);
  assert.match(drawing.description, /microduck-drawing-v1/);
  assert.match(drawing.description, /microduck-brush-v2/);
  assert.deepEqual(JSON.parse(JSON.stringify(defaultRewardWeights("drawing"))), {});

  const pencil = getDrawingTool("pencil");
  assert.deepEqual(JSON.parse(JSON.stringify({
    contractVersion: pencil.contractVersion,
    observations: pencil.observations,
    actions: pencil.actions,
    fullTimesteps: pencil.fullTimesteps,
    smokeTimesteps: pencil.smokeTimesteps,
    fullNumSteps: pencil.fullNumSteps,
    evalEpisodes: pencil.evalEpisodes,
    maxEpisodeSeconds: pencil.maxEpisodeSeconds,
    learningRate: pencil.ppo.learningRate,
  })), {
    contractVersion: "microduck-drawing-v1",
    observations: 83,
    actions: 15,
    fullTimesteps: 32_768,
    smokeTimesteps: 768,
    fullNumSteps: 256,
    evalEpisodes: 8,
    maxEpisodeSeconds: 32,
    learningRate: 0.00003,
  });

  const brush = getDrawingTool();
  assert.deepEqual(JSON.parse(JSON.stringify({
    id: brush.id,
    contractVersion: brush.contractVersion,
    observations: brush.observations,
    actions: brush.actions,
    fullTimesteps: brush.fullTimesteps,
    fullNumSteps: brush.fullNumSteps,
    fullNumMinibatches: brush.fullNumMinibatches,
    batchSize: brush.batchSize,
    evalEpisodes: brush.evalEpisodes,
    evaluationConfigs: brush.evaluationConfigs,
    maxEpisodeSeconds: brush.maxEpisodeSeconds,
    initialStd: brush.initialStd,
    explorationLabel: brush.explorationLabel,
    trainerLabel: brush.trainerLabel,
    actorLabel: brush.actorLabel,
  })), {
    id: "brush",
    contractVersion: "microduck-brush-v2",
    observations: 93,
    actions: 15,
    fullTimesteps: 49_152,
    fullNumSteps: 6_144,
    fullNumMinibatches: 12,
    batchSize: 512,
    evalEpisodes: 4,
    evaluationConfigs: 10,
    maxEpisodeSeconds: 120,
    initialStd: 0.0003,
    explorationLabel: "Head std 0.0003 · legs 0.0001 · jaw 0.0002",
    trainerLabel: "SB3 PPO · PyTorch CPU",
    actorLabel: "Learned-feedback actor · per-joint exploration",
  });
});

test("Backflip frontend metadata exposes exact teacher transfer, the v2 gate, and controller boundaries", () => {
  const experiment = getExperiment("backflip");
  assert.equal(isExperimentId("backflip"), true);
  assert.equal(experiment.title, "Backflip showcase");
  assert.equal(experiment.artifactStem, "backflip");
  assert.equal(experiment.fullTimesteps, 400_000);
  assert.equal(experiment.fullEnvs, 16);
  assert.equal(experiment.fullNumSteps, 128);
  assert.equal(experiment.fullNumMinibatches, 4);
  assert.equal(experiment.maxEpisodeSeconds, 12);
  assert.equal(experiment.seed, 7);
  assert.equal(experiment.initialStd, 0.03);
  assert.equal(experiment.normalizeRewards, true);
  assert.deepEqual(
    JSON.parse(JSON.stringify(experiment.fullEnvironment)),
    { domainRand: false, obsNoise: false, actionDelay: false, randomYaw: false }
  );
  assert.equal(experiment.ppo.learningRate, 0.000003);
  assert.equal(experiment.ppo.updateEpochs, 2);
  assert.equal(experiment.ppo.entropyCoefficient, 0);
  assert.match(experiment.description, /spotter-assisted launch → PPO landing → pretrained stand-policy handoff/);
  assert.match(experiment.guidance.requiredInput, /spotter-launch-landing-stand-v2/);
  assert.match(experiment.result, /5\.8 rad of credited rotation before handoff/);
  assert.match(experiment.result, /10 consecutive stable PPO landing steps/);
  assert.match(experiment.result, /no override or body support/);
  assert.ok(experiment.guidance.steps.every((step) => !/\bBC\b|behavior cloning/i.test(step)));
  assert.match(experiment.guidance.steps.join(" "), /exactly transferring the pretrained stand actor and observation normalizer/);
  assert.match(experiment.guidance.steps.join(" "), /16 pretrained-teacher landing episodes of 600 steps \(9,600 calibration steps\)/);
  assert.match(experiment.guidance.steps.join(" "), /fixes the discounted-return RMS/);
  assert.match(experiment.guidance.steps.join(" "), /10 critic-only calibration epochs/);
  assert.match(experiment.guidance.steps.join(" "), /actor remains unchanged/);
  assert.match(experiment.guidance.steps.join(" "), /calibrated reward normalization is frozen/);
  assert.match(experiment.guidance.steps.join(" "), /fresh optimizer/);
  assert.match(experiment.guidance.distinctions.join(" "), /does not by itself establish positive PPO improvement/);
  assert.match(experiment.description, /Showcase acceptance does not establish PPO improvement/);
  assert.match(experiment.guidance.distinctions.join(" "), /return regresses 12\.63% versus initialization/);
  assert.match(experiment.guidance.distinctions.join(" "), /extended learning audit FAILS/);
  assert.match(experiment.guidance.steps.join(" "), /5\.3 rad threshold/);
  assert.match(experiment.guidance.steps.join(" "), /3\.5 and 6\.5 rad\/s/);
  assert.match(experiment.guidance.distinctions.join(" "), /stand-policy phase cannot finish credited rotation/i);
  assert.match(experiment.guidance.output, /landing ONNX alone is not the complete backflip controller/i);
  assert.deepEqual(
    JSON.parse(JSON.stringify(defaultRewardWeights("backflip"))),
    {
      landing_upright: 3,
      landing_pose: 2,
      landing_settle: 1,
      landing_joint_speed_penalty: 0.05,
      landing_action_rate_penalty: 0.02,
      landing_action_size_penalty: 0.01,
    }
  );
  assert.match(experiment.reward.summary, /PPO landing only/);
  assert.match(experiment.reward.summary, /zero during assisted launch/);
  assert.deepEqual(
    JSON.parse(JSON.stringify(experiment.reward.terms.map((term) => term.penalty === true))),
    [false, false, false, true, true, true]
  );
});
