import { spawn, type ChildProcess } from "node:child_process";
import { existsSync, readFileSync, realpathSync, statSync } from "node:fs";
import { createHash } from "node:crypto";
import { readFile, stat, writeFile } from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import {
  prepareRenderEvidence,
  finalizeRenderEvidence,
  readRenderEvidence,
  type RenderRecipeOptions,
} from "@/lib/rlx-render-evidence";

import {
  defaultRewardWeights,
  drawingToolFromContract,
  getExperiment,
  getDrawingTool,
  isExperimentId,
  type DrawingTool,
  type ExperimentId,
} from "@/lib/experiments";
import {
  collectTrainingHistory,
  prepareTrainingHistory,
  restoreTrainingInvocation,
  type RlxActiveTrainingHistory,
  type RlxTrainingHistory,
} from "@/lib/rlx-history";

export type RlxOperation = "train" | "eval" | "render" | "export";
const BACKFLIP_PROTOCOL_VERSION = "spotter-launch-landing-stand-v2";
export type RlxJobPhase =
  | "idle"
  | "running"
  | "succeeded"
  | "failed"
  | "cancelled";

export interface RlxRecipe {
  experimentId: ExperimentId;
  drawingTool?: DrawingTool;
  runName: string;
  profile: "smoke" | "full";
  totalTimesteps: number;
  numEnvs: number;
  numSteps: number;
  numMinibatches: number;
  maxEpisodeS: number;
  evalSteps: number;
  evalEpisodes: number;
  renderSeconds: number;
  danceClip: string | null;
  dancePoseSigma: number | null;
  locomotionForwardCommand: number | null;
  initialStd: number;
  normalizeRewards: boolean;
  freezeObservationNormalization: boolean;
  checkpointInterval: number;
  seed: number;
  learningRate: number;
  gamma: number;
  clipCoefficient: number;
  updateEpochs: number;
  entropyCoefficient: number;
  maxGradNorm: number;
  domainRand: boolean;
  obsNoise: boolean;
  actionDelay: boolean;
  randomYaw: boolean;
  bridgeCurriculum: boolean;
  stiltHeightCm: number;
  stiltBlend: number;
  stiltMassKg: number;
  swingInitialAngleDeg: number;
  swingInitialRateRadS: number;
  swingPlanarActions: boolean;
  swingMinSpanDeg: number;
  resumeFromCheckpoint: boolean;
  rewardWeights: Record<string, number>;
}

export interface RlxArtifacts {
  checkpoint: boolean;
  metadata: boolean;
  onnx: boolean;
  renderSheet: boolean;
  renderVideo: boolean;
  evaluation: boolean;
  checkpointPath: string;
  onnxPath: string;
  renderDirectory: string;
}

export interface RlxRewardPoint {
  step: number;
  reward: number;
}

export interface RlxJobSnapshot {
  renderVerified: boolean;
  renderEvidenceId: string | null;
  phase: RlxJobPhase;
  operation: RlxOperation | null;
  activeJob: {
    operation: RlxOperation;
    experimentId: ExperimentId;
    runName: string;
    startedAt: string | null;
  } | null;
  experimentId: ExperimentId;
  drawingTool?: DrawingTool;
  runName: string;
  startedAt: string | null;
  finishedAt: string | null;
  exitCode: number | null;
  logs: string[];
  result: Record<string, unknown> | null;
  evaluation: Record<string, unknown> | null;
  savedRecipe: RlxRecipe | null;
  rewardHistory: RlxRewardPoint[];
  trainingHistory: RlxTrainingHistory;
  normalizeRewards: boolean;
  trainingSteps: number;
  trainingTotal: number;
  artifacts: RlxArtifacts;
}

interface MutableRlxJob {
  environmentKeys: Record<string, string>;
  generation: number;
  phase: RlxJobPhase;
  operation: RlxOperation | null;
  experimentId: ExperimentId;
  runName: string;
  startedAt: string | null;
  finishedAt: string | null;
  exitCode: number | null;
  logs: string[];
  result: Record<string, unknown> | null;
  evaluation: Record<string, unknown> | null;
  rewardHistory: RlxRewardPoint[];
  normalizeRewards: boolean;
  trainingSteps: number;
  trainingTotal: number;
  trainingHistoryContext: RlxActiveTrainingHistory | null;
  stdoutBuffer: string;
  stderrBuffer: string;
  child: ChildProcess | null;
}

declare global {
  var __microduckRlxJob: MutableRlxJob | undefined;
}

const INITIAL_JOB: MutableRlxJob = {
  environmentKeys: {},
  generation: 0,
  phase: "idle",
  operation: null,
  experimentId: "dance",
  runName: "dance-studio",
  startedAt: null,
  finishedAt: null,
  exitCode: null,
  logs: [],
  result: null,
  evaluation: null,
  rewardHistory: [],
  normalizeRewards: false,
  trainingSteps: 0,
  trainingTotal: 0,
  trainingHistoryContext: null,
  stdoutBuffer: "",
  stderrBuffer: "",
  child: null,
};

const job = (globalThis.__microduckRlxJob ??= { ...INITIAL_JOB });
job.environmentKeys ??= {};
job.generation ??= 0;
job.experimentId ??= "dance";
job.rewardHistory ??= [];
job.normalizeRewards ??=
  job.child?.spawnargs?.includes("--normalize-rewards") ?? false;
job.trainingSteps ??= 0;
job.trainingTotal ??= 0;
job.trainingHistoryContext ??= null;
job.stdoutBuffer ??= "";
job.stderrBuffer ??= "";

export function sanitizeRunName(value: unknown): string {
  const normalized = String(value ?? "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 48);
  if (!normalized) throw new Error("Run name must contain a letter or number.");
  return normalized;
}

function rlxRoot(): string {
  const candidates = [
    path.resolve(process.cwd(), "../rlx"),
    path.resolve(process.cwd(), "rlx"),
  ];
  const root = candidates.find((candidate) =>
    existsSync(path.join(candidate, "examples/ppo_microduck_dance.py"))
  );
  if (!root) throw new Error("Cannot locate the sibling RLX checkout.");
  return root;
}

function pathsFor(experimentId: ExperimentId, runName: string) {
  const root = rlxRoot();
  const experiment = getExperiment(experimentId);
  const runDirectory = path.join(root, "runs", "studio", experimentId, runName);
  const files = experiment.artifactFiles ?? {
    checkpoint: `${experiment.artifactStem}.safetensors`,
    metadata: `${experiment.artifactStem}.safetensors.json`,
    onnx: `${experiment.artifactStem}.onnx`,
    evaluation: "evaluation.json",
  };
  return {
    root,
    runDirectory,
    checkpoint: path.join(runDirectory, files.checkpoint),
    metadata: path.join(runDirectory, files.metadata),
    onnx: path.join(runDirectory, files.onnx),
    renderDirectory: path.join(runDirectory, "render"),
    renderSheet: path.join(
      runDirectory,
      "render",
      files.renderSheet ?? "ep0_sheet.png"
    ),
    renderVideo: path.join(
      runDirectory,
      "render",
      files.renderVideo ?? "ep0.mp4"
    ),
    evaluation: path.join(runDirectory, files.evaluation),
    evaluationFallback: path.join(runDirectory, "evaluation.json"),
    trainingMetrics: path.join(runDirectory, "training-metrics.jsonl"),
  };
}

export function artifactPath(
  experimentIdValue: unknown,
  runNameValue: unknown,
  kind: "checkpoint" | "metadata" | "onnx" | "sheet" | "video" | "evaluation"
): string {
  const experimentId = normalizeExperimentId(experimentIdValue);
  const runName = sanitizeRunName(runNameValue);
  const paths = pathsFor(experimentId, runName);
  const selected = {
    checkpoint: paths.checkpoint,
    metadata: paths.metadata,
    onnx: paths.onnx,
    sheet: paths.renderSheet,
    video: paths.renderVideo,
    evaluation: paths.evaluation,
  }[kind];
  if (
    kind === "evaluation" &&
    !existsSync(selected) &&
    existsSync(paths.evaluationFallback)
  ) {
    return paths.evaluationFallback;
  }
  return selected;
}

async function artifactState(
  experimentId: ExperimentId,
  runName: string
): Promise<RlxArtifacts> {
  const paths = pathsFor(experimentId, runName);
  return {
    checkpoint: existsSync(paths.checkpoint),
    metadata: existsSync(paths.metadata),
    onnx: existsSync(paths.onnx),
    renderSheet: existsSync(paths.renderSheet),
    renderVideo: existsSync(paths.renderVideo),
    evaluation:
      existsSync(paths.evaluation) || existsSync(paths.evaluationFallback),
    checkpointPath: path.relative(paths.root, paths.checkpoint),
    onnxPath: path.relative(paths.root, paths.onnx),
    renderDirectory: path.relative(paths.root, paths.renderDirectory),
  };
}

function appendLine(line: string) {
  const trimmed = line.trimEnd();
  if (!trimmed) return;
  if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
    try {
      const payload = JSON.parse(trimmed) as Record<string, unknown>;
      if (payload.event === "training_progress") {
        const steps =
          typeof payload.steps === "number" ? payload.steps : Number.NaN;
        const total =
          typeof payload.total === "number" ? payload.total : Number.NaN;
        const reward =
          typeof payload.mean_reward === "number"
            ? payload.mean_reward
            : Number.NaN;
        if (Number.isFinite(steps)) job.trainingSteps = Math.max(0, steps);
        if (Number.isFinite(total)) job.trainingTotal = Math.max(0, total);
        if (Number.isFinite(steps) && Number.isFinite(reward)) {
          const point = { step: Math.max(0, steps), reward };
          const previous = job.rewardHistory.at(-1);
          if (previous?.step === point.step) {
            job.rewardHistory[job.rewardHistory.length - 1] = point;
          } else {
            job.rewardHistory.push(point);
          }
        }
        return;
      }
    } catch {
      // Non-JSON process output remains visible in the log.
    }
  }
  job.logs.push(trimmed);
  if (job.logs.length > 240) job.logs.splice(0, job.logs.length - 240);
}

function appendStream(stream: "stdout" | "stderr", chunk: Buffer | string) {
  const bufferKey = stream === "stdout" ? "stdoutBuffer" : "stderrBuffer";
  const combined = job[bufferKey] + String(chunk);
  const lines = combined.split(/\r?\n/);
  job[bufferKey] = lines.pop() ?? "";
  lines.forEach(appendLine);
}

function flushStreams() {
  if (job.stdoutBuffer) appendLine(job.stdoutBuffer);
  if (job.stderrBuffer) appendLine(job.stderrBuffer);
  job.stdoutBuffer = "";
  job.stderrBuffer = "";
}

function positiveInt(value: unknown, fallback: number, max: number): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(1, Math.min(max, Math.round(parsed)));
}

function boundedFloat(
  value: unknown,
  fallback: number,
  min: number,
  max: number
): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(min, Math.min(max, parsed));
}

function explicitPositiveInt(
  value: unknown,
  fallback: number,
  max: number,
  label: string
): number {
  if (value == null) return fallback;
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 1 ||
    value > max
  ) {
    throw new Error(`${label} must be an integer between 1 and ${max}.`);
  }
  return value;
}

function explicitPositiveFloat(
  value: unknown,
  fallback: number,
  max: number,
  label: string
): number {
  if (value == null) return fallback;
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value <= 0 ||
    value > max
  ) {
    throw new Error(`${label} must be a number greater than 0 and at most ${max}.`);
  }
  return value;
}

function explicitNonNegativeInt(
  value: unknown,
  fallback: number,
  max: number,
  label: string
): number {
  if (value == null) return fallback;
  if (
    typeof value !== "number" ||
    !Number.isSafeInteger(value) ||
    value < 0 ||
    value > max
  ) {
    throw new Error(`${label} must be an integer between 0 and ${max}.`);
  }
  return value;
}

function explicitBoolean(
  value: unknown,
  fallback: boolean,
  label: string
): boolean {
  if (value == null) return fallback;
  if (typeof value !== "boolean") {
    throw new Error(`${label} must be a boolean.`);
  }
  return value;
}

function isWithin(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function normalizeDanceClip(
  experimentId: ExperimentId,
  value: unknown
): string | null {
  if (value == null) return null;
  if (typeof value !== "string") {
    throw new Error("Dance clip must be a file path string.");
  }
  const input = value.trim();
  if (!input) throw new Error("Dance clip path cannot be empty.");
  if (experimentId !== "dance") {
    throw new Error("Dance clips can only be used with the dance experiment.");
  }

  const root = rlxRoot();
  const workspaceRoot = path.dirname(root);
  const allowedRoots = [
    path.join(workspaceRoot, "dance-clip"),
    path.join(root, "artifacts"),
  ];
  const candidates = path.isAbsolute(input)
    ? [path.resolve(input)]
    : [path.resolve(workspaceRoot, input), path.resolve(root, input)];
  const candidate = candidates.find((item) =>
    allowedRoots.some((allowedRoot) => isWithin(allowedRoot, item))
  );
  if (!candidate) {
    throw new Error(
      "Dance clip must be under workspace dance-clip/ or rlx/artifacts/."
    );
  }
  if (!existsSync(candidate)) {
    throw new Error(`Dance clip does not exist: ${candidate}`);
  }

  const resolved = realpathSync(candidate);
  const resolvedRoots = allowedRoots.map((allowedRoot) =>
    existsSync(/* turbopackIgnore: true */ allowedRoot)
      ? realpathSync(/* turbopackIgnore: true */ allowedRoot)
      : allowedRoot
  );
  if (!resolvedRoots.some((allowedRoot) => isWithin(allowedRoot, resolved))) {
    throw new Error(
      "Dance clip must resolve under workspace dance-clip/ or rlx/artifacts/."
    );
  }
  if (!statSync(resolved).isFile()) {
    throw new Error(`Dance clip is not a regular file: ${resolved}`);
  }
  return resolved;
}

function normalizeDancePoseSigma(
  experimentId: ExperimentId,
  value: unknown
): number | null {
  if (value == null) return null;
  if (experimentId !== "dance") {
    throw new Error("Dance pose sigma can only be used with the dance experiment.");
  }
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error("Dance pose sigma must be a finite number greater than 0.");
  }
  return value;
}

function normalizeLocomotionForwardCommand(
  experimentId: ExperimentId,
  value: unknown
): number | null {
  if (value == null) return null;
  if (experimentId !== "running" && experimentId !== "stilts") {
    throw new Error("Locomotion forward command can only be used with running or stilts.");
  }
  if (
    typeof value !== "number" ||
    !Number.isFinite(value) ||
    value <= 0 ||
    value > 1.5
  ) {
    throw new Error("Locomotion forward command must be a finite number greater than 0 and at most 1.5.");
  }
  return value;
}

function normalizeRewardWeights(
  experimentId: ExperimentId,
  value: unknown
): Record<string, number> {
  const defaults = defaultRewardWeights(experimentId);
  if (value == null) return defaults;
  if (typeof value !== "object" || Array.isArray(value)) {
    throw new Error("Reward weights must be an object.");
  }
  const input = value as Record<string, unknown>;
  for (const key of Object.keys(input)) {
    if (!(key in defaults)) {
      throw new Error(`Unknown reward weight for ${experimentId}: ${key}`);
    }
  }
  return Object.fromEntries(
    Object.entries(defaults).map(([key, fallback]) => {
      const parsed = Number(input[key] ?? fallback);
      if (!Number.isFinite(parsed) || parsed < 0) {
        throw new Error(`Reward weight ${key} must be a non-negative number.`);
      }
      return [key, Math.min(parsed, 10_000)];
    })
  );
}

function normalizeDrawingTool(
  experimentId: ExperimentId,
  value: unknown
): DrawingTool | undefined {
  if (experimentId !== "drawing") return undefined;
  if (value === undefined || value === null || value === "") return "brush";
  if (value === "pencil" || value === "brush") return value;
  throw new Error(`Unknown drawing tool: ${String(value)}`);
}

export function normalizeRecipe(input: Partial<RlxRecipe>): RlxRecipe {
  const experimentId = normalizeExperimentId(input.experimentId);
  const experiment = getExperiment(experimentId);
  const drawingTool = normalizeDrawingTool(experimentId, input.drawingTool);
  const drawingDefinition = drawingTool ? getDrawingTool(drawingTool) : null;
  const profile = input.profile === "full" ? "full" : "smoke";
  const smoke = profile === "smoke";
  const drawing = experimentId === "drawing";
  const requestedFreezeObservationNormalization = explicitBoolean(
    input.freezeObservationNormalization,
    (experimentId === "backflip" || experimentId === "bridge") && !smoke,
    "Freeze observation normalization"
  );
  const freezeObservationNormalization = experimentId === "basketball"
    ? true
    : smoke
      ? false
    : requestedFreezeObservationNormalization;
  if (
    freezeObservationNormalization &&
    !drawing &&
    input.resumeFromCheckpoint !== true &&
    !(experimentId === "backflip" && profile === "full") &&
    !(experimentId === "bridge" && profile === "full") &&
    experimentId !== "basketball"
  ) {
    throw new Error("Freeze observation normalization requires resumeFromCheckpoint to be true.");
  }
  const numEnvs = drawing ? experiment.fullEnvs : positiveInt(
    input.numEnvs, smoke ? 2 : experiment.fullEnvs, 64
  );
  const numSteps = drawing ? drawingDefinition!.fullNumSteps : explicitPositiveInt(
    input.numSteps, smoke ? 2 : experiment.fullNumSteps, 100_000, "Rollout steps"
  );
  const numMinibatches = drawing ? drawingDefinition!.fullNumMinibatches : explicitPositiveInt(
    input.numMinibatches, smoke ? 1 : experiment.fullNumMinibatches, 100_000, "Minibatches"
  );
  const batch = numEnvs * numSteps;
  if (numMinibatches > batch || batch % numMinibatches !== 0) {
    throw new Error(
      "Minibatches must divide numEnvs * numSteps and cannot exceed that batch."
    );
  }
  const maxEpisodeS = drawing
    ? drawingDefinition!.maxEpisodeSeconds
    : explicitPositiveFloat(
        input.maxEpisodeS,
        smoke ? 1 : experiment.maxEpisodeSeconds,
        3_600,
        "Maximum episode seconds"
      );
  const evalSteps = explicitPositiveInt(
    input.evalSteps,
    smoke ? 4 : Math.max(
      experimentId === "running" ? 600 : experimentId === "bridge" ? 1_000 : 500,
      Math.ceil(maxEpisodeS * 50)
    ),
    10_000_000,
    "Evaluation steps"
  );
  const evalEpisodes = explicitPositiveInt(
    input.evalEpisodes,
    smoke
      ? drawingTool === "brush" ? drawingDefinition!.evalEpisodes : 1
      : drawing ? drawingDefinition!.evalEpisodes : 1,
    10_000,
    "Evaluation episodes"
  );
  if (!smoke && experimentId === "running" && (maxEpisodeS < 12 || evalSteps < 600)) {
    throw new Error("Full Running evaluation requires at least 12 episode seconds and 600 evaluation steps.");
  }
  if (!smoke && experimentId === "bridge" && (maxEpisodeS < 20 || evalSteps < 1_000)) {
    throw new Error("Full Bridge evaluation requires 20 episode seconds and at least 1,000 evaluation steps.");
  }
  if (
    !smoke &&
    experimentId === "bridge" &&
    evalSteps % Math.round(maxEpisodeS * 50) !== 0
  ) {
    throw new Error("Full Bridge evaluation steps must cover a whole number of complete episode horizons.");
  }
  const renderSeconds = explicitPositiveFloat(
    input.renderSeconds,
    experimentId === "dance" ? 120 : maxEpisodeS,
    3_600,
    "Render seconds"
  );
  let totalTimesteps = positiveInt(
    input.totalTimesteps,
    smoke ? drawing ? drawingDefinition!.smokeTimesteps : 4 : drawing ? drawingDefinition!.fullTimesteps : experiment.fullTimesteps,
    40_000_000
  );
  if (drawing) totalTimesteps = Math.max(totalTimesteps, drawingDefinition!.smokeTimesteps);
  totalTimesteps = Math.max(batch, Math.ceil(totalTimesteps / batch) * batch);
  return {
    experimentId,
    drawingTool,
    runName: sanitizeRunName(input.runName ?? experiment.defaultRunName),
    profile,
    totalTimesteps,
    numEnvs,
    numSteps,
    numMinibatches,
    maxEpisodeS,
    evalSteps,
    evalEpisodes,
    renderSeconds,
    danceClip: normalizeDanceClip(experimentId, input.danceClip),
    dancePoseSigma: normalizeDancePoseSigma(
      experimentId,
      input.dancePoseSigma
    ),
    locomotionForwardCommand: normalizeLocomotionForwardCommand(
      experimentId,
      input.locomotionForwardCommand
    ),
    initialStd: drawing
      ? drawingDefinition!.initialStd
      : explicitPositiveFloat(
          input.initialStd,
          experiment.initialStd,
          10,
          "Initial policy standard deviation"
        ),
    normalizeRewards: drawing ? experiment.normalizeRewards : explicitBoolean(
      input.normalizeRewards, experiment.normalizeRewards, "Reward normalization"
    ),
    freezeObservationNormalization: drawing ? false : freezeObservationNormalization,
    checkpointInterval: drawing ? 0 : explicitNonNegativeInt(
      input.checkpointInterval, 100_000, 40_000_000, "Checkpoint interval"
    ),
    seed: positiveInt(input.seed, experiment.seed, 2_147_483_647),
    learningRate: drawing ? drawingDefinition!.ppo.learningRate : boundedFloat(
      input.learningRate, experiment.ppo.learningRate, 0.000001, 0.1
    ),
    gamma: drawing ? drawingDefinition!.ppo.gamma : boundedFloat(input.gamma, experiment.ppo.gamma, 0.8, 1),
    clipCoefficient: drawing ? drawingDefinition!.ppo.clipCoefficient : boundedFloat(
      input.clipCoefficient, experiment.ppo.clipCoefficient, 0.01, 1
    ),
    updateEpochs: drawing ? drawingDefinition!.ppo.updateEpochs : positiveInt(
      input.updateEpochs, experiment.ppo.updateEpochs, 20
    ),
    entropyCoefficient: drawing ? drawingDefinition!.ppo.entropyCoefficient : boundedFloat(
      input.entropyCoefficient, experiment.ppo.entropyCoefficient, 0, 1
    ),
    maxGradNorm: drawing ? drawingDefinition!.ppo.maxGradNorm : boundedFloat(
      input.maxGradNorm, experiment.ppo.maxGradNorm, 0.01, 10
    ),
    domainRand: drawing ? false : smoke ? false : input.domainRand === undefined
      ? experiment.fullEnvironment.domainRand
      : input.domainRand !== false,
    obsNoise: drawing ? false : smoke ? false : input.obsNoise === undefined
      ? experiment.fullEnvironment.obsNoise
      : input.obsNoise !== false,
    actionDelay: drawing ? false : smoke ? false : input.actionDelay === undefined
      ? experiment.fullEnvironment.actionDelay
      : input.actionDelay !== false,
    randomYaw: drawing ? false : smoke ? false : input.randomYaw === undefined
      ? experiment.fullEnvironment.randomYaw
      : input.randomYaw !== false,
    bridgeCurriculum: experimentId === "bridge" && profile === "full",
    stiltHeightCm: boundedFloat(
      input.stiltHeightCm,
      experiment.controls?.stiltHeightCm ?? 10,
      0.8,
      300
    ),
    stiltBlend: boundedFloat(
      input.stiltBlend,
      experiment.controls?.stiltBlend ?? 0.5,
      0,
      1
    ),
    stiltMassKg: boundedFloat(
      input.stiltMassKg,
      experiment.controls?.stiltMassKg ?? 0.029,
      0.001,
      2
    ),
    swingInitialAngleDeg: boundedFloat(
      input.swingInitialAngleDeg,
      0,
      0,
      30
    ),
    swingInitialRateRadS: boundedFloat(
      input.swingInitialRateRadS,
      0,
      0,
      1
    ),
    swingPlanarActions:
      experimentId === "swing" && input.swingPlanarActions !== false,
    swingMinSpanDeg: boundedFloat(input.swingMinSpanDeg ?? 150, 150, 1, 180),
    resumeFromCheckpoint: drawing ? false : input.resumeFromCheckpoint === true,
    rewardWeights: normalizeRewardWeights(experimentId, input.rewardWeights),
  };
}

export function normalizeExperimentId(value: unknown): ExperimentId {
  if (!isExperimentId(value)) {
    if (value == null || value === "") return "dance";
    throw new Error(`Unknown Microduck experiment: ${String(value)}`);
  }
  return value;
}

function pythonArgs(): string[] {
  if (process.env.MICRODUCK_STUDIO_PYTHON_DIRECT) {
    return [process.env.MICRODUCK_STUDIO_PYTHON_DIRECT];
  }
  const configured = process.env.MICRODUCK_STUDIO_PYTHON;
  const preferred = configured || "/usr/local/bin/python3.12";
  return [
    "run",
    "--isolated",
    "--no-project",
    "--python",
    preferred,
    "--with-editable",
    ".",
    "--with-editable",
    "../microduck_local",
  ];
}

function commonArgs(recipe: RlxRecipe): string[] {
  const args = [
    "--recipe",
    recipe.experimentId,
    "--num-envs",
    String(recipe.numEnvs),
    "--seed",
    String(recipe.seed),
    "--max-episode-s",
    String(recipe.maxEpisodeS),
    recipe.domainRand ? "--domain-rand" : "--no-domain-rand",
    recipe.obsNoise ? "--obs-noise" : "--no-obs-noise",
    recipe.actionDelay ? "--action-delay" : "--no-action-delay",
    recipe.randomYaw ? "--random-yaw" : "--no-random-yaw",
    "--weight-overrides",
    JSON.stringify(recipe.rewardWeights),
  ];
  if (recipe.danceClip) {
    args.push("--dance-clip", recipe.danceClip);
  }
  if (recipe.dancePoseSigma !== null) {
    args.push("--dance-pose-sigma", String(recipe.dancePoseSigma));
  }
  if (recipe.locomotionForwardCommand !== null) {
    args.push("--locomotion-forward-command", String(recipe.locomotionForwardCommand));
  }
  if (recipe.experimentId === "stilts") {
    args.push(
      "--stilt-height-cm",
      String(recipe.stiltHeightCm),
      "--stilt-blend",
      String(recipe.stiltBlend),
      "--stilt-mass-kg",
      String(recipe.stiltMassKg)
    );
  }
  if (recipe.experimentId === "swing") {
    args.push(
      recipe.swingPlanarActions
        ? "--swing-planar-actions"
        : "--no-swing-planar-actions"
    );
  }
  return args;
}

function evaluationEnvironmentKey(recipe: RlxRecipe, operation: RlxOperation = "eval"): string {
  if (recipe.experimentId === "drawing") {
    const drawing = getDrawingTool(recipe.drawingTool ?? "brush");
    return JSON.stringify({
      contract: drawing.contractVersion,
      operation,
      seed: recipe.seed,
      maxEpisodeS: operation === "render" ? recipe.renderSeconds : recipe.maxEpisodeS,
      evalEpisodes: operation === "eval" ? recipe.evalEpisodes : undefined,
    });
  }
  return JSON.stringify(commonArgs({
    ...recipe,
    ...(operation === "render" ? {
      domainRand: false,
      obsNoise: false,
      actionDelay: false,
      randomYaw: false,
      maxEpisodeS: recipe.renderSeconds,
    } : {}),
    rewardWeights: Object.fromEntries(Object.entries(recipe.rewardWeights).sort(([left], [right]) => left.localeCompare(right))),
  }));
}

function evaluationSettings(recipe: RlxRecipe) {
  return {
    evaluation_mode: recipe.profile === "smoke" ? "pipeline" : "skill",
    eval_steps: recipe.evalSteps,
    ...(recipe.experimentId === "drawing"
      ? { eval_episodes: recipe.evalEpisodes }
      : {}),
    ...(recipe.locomotionForwardCommand !== null
      ? { locomotion_forward_command: recipe.locomotionForwardCommand }
      : {}),
    ...(recipe.experimentId === "swing"
      ? { swing_min_span_deg: recipe.swingMinSpanDeg }
      : {}),
    recipe,
  };
}

function backflipRecipeOptions(
  result: Record<string, unknown> | null
): RenderRecipeOptions | null {
  const evaluatedOptions = (
    result?.evaluation as {
      environment?: { recipe_options?: Record<string, unknown> };
    } | undefined
  )?.environment?.recipe_options;
  const renderedOptions = (
    result?.environment as {
      recipe_options?: Record<string, unknown>;
    } | undefined
  )?.recipe_options;
  const options = evaluatedOptions ?? renderedOptions;
  if (
    options?.backflip_protocol_version !== BACKFLIP_PROTOCOL_VERSION ||
    typeof options.stand_policy_sha256 !== "string"
  ) {
    return null;
  }
  return {
    backflip_protocol_version: options.backflip_protocol_version,
    stand_policy_sha256: options.stand_policy_sha256,
  };
}

function renderEvidenceInputs(recipe: RlxRecipe, source: string, sourceType: string) {
  const paths = pathsFor(recipe.experimentId, recipe.runName);
  const standPolicy = path.resolve(paths.root, "../microduck/policies/alpha_stand.onnx");
  const drawing = recipe.experimentId === "drawing";
  return {
    source,
    sourceFiles: drawing
      ? [paths.checkpoint, paths.metadata, paths.onnx]
      : sourceType === "checkpoint" ? [source, paths.metadata] : [source],
    recipeKey: evaluationEnvironmentKey(recipe),
    clipPath: recipe.experimentId === "dance" ? recipe.danceClip ?? path.join(paths.root, "assets/clips/dance-120bpm.json") : null,
    externalFiles: recipe.experimentId === "backflip" ? [standPolicy] : [],
  };
}

function commandFor(operation: RlxOperation, recipe: RlxRecipe): string[] {
  const paths = pathsFor(recipe.experimentId, recipe.runName);
  const adapter = getExperiment(recipe.experimentId).commandAdapter;
  const drawingTool = recipe.experimentId === "drawing"
    ? recipe.drawingTool ?? "brush"
    : null;
  const script = adapter === "basketball"
    ? "examples/ppo_microduck_balance.py"
    : adapter === "drawing"
      ? drawingTool === "brush"
        ? "examples/ppo_microduck_brush.py"
        : "examples/ppo_microduck_drawing.py"
      : "examples/ppo_microduck_studio.py";
  const base = [
    ...pythonArgs(),
    script,
    operation,
  ];
  if (adapter === "drawing") {
    if (drawingTool === "brush") {
      if (operation === "train") {
        return [
          ...base,
          "--out",
          paths.runDirectory,
          "--seed",
          String(recipe.seed),
          "--steps",
          String(recipe.totalTimesteps),
          "--dagger",
          "2",
          "--learning-rate",
          "1e-7",
          "--anchor-limit",
          "0.00025",
        ];
      }
      if (operation === "eval") {
        return [
          ...base,
          "--onnx",
          paths.onnx,
          "--out",
          paths.evaluation,
          "--seed",
          String(recipe.seed),
          "--episodes",
          String(recipe.evalEpisodes),
        ];
      }
      if (operation === "render") {
        return [
          ...base,
          "--onnx",
          paths.onnx,
          "--out",
          paths.renderDirectory,
          "--seed",
          String(recipe.seed),
          "--seconds",
          String(recipe.renderSeconds),
        ];
      }
      return [
        ...base,
        "--checkpoint",
        paths.checkpoint,
        "--onnx-output",
        paths.onnx,
      ];
    }
    if (operation === "train") {
      return [
        ...base,
        "--output",
        paths.runDirectory,
        "--total-timesteps",
        String(recipe.totalTimesteps),
        "--seed",
        String(recipe.seed),
        "--max-episode-s",
        String(recipe.maxEpisodeS),
      ];
    }
    if (operation === "eval") {
      return [
        ...base,
        "--checkpoint",
        paths.checkpoint,
        "--eval-output",
        paths.evaluation,
        "--eval-episodes",
        String(recipe.evalEpisodes),
        "--seed",
        String(recipe.seed),
        "--max-episode-s",
        String(recipe.maxEpisodeS),
      ];
    }
    if (operation === "render") {
      return [
        ...base,
        "--checkpoint",
        paths.checkpoint,
        "--render-output",
        paths.renderDirectory,
        "--render-seconds",
        String(recipe.renderSeconds),
        "--seed",
        String(recipe.seed),
        "--max-episode-s",
        String(recipe.maxEpisodeS),
      ];
    }
    return [
      ...base,
      "--checkpoint",
      paths.checkpoint,
      "--onnx-output",
      paths.onnx,
    ];
  }
  if (operation === "train") {
    const basketball = recipe.experimentId === "basketball";
    return [
      ...base,
      ...(basketball ? ["--output-dir", paths.runDirectory] : []),
      "--checkpoint",
      paths.checkpoint,
      ...(recipe.resumeFromCheckpoint && existsSync(paths.checkpoint)
        ? ["--init-from", paths.checkpoint]
        : []),
      ...(recipe.freezeObservationNormalization
        ? ["--freeze-observation-normalization"]
        : []),
      "--onnx-output",
      paths.onnx,
      ...(recipe.experimentId === "backflip"
        ? ["--backend", "dummy"]
        : []),
      ...(recipe.bridgeCurriculum ? ["--bridge-curriculum"] : []),
      "--total-timesteps",
      String(recipe.totalTimesteps),
      "--num-steps",
      String(recipe.numSteps),
      "--num-minibatches",
      String(basketball ? 1 : recipe.numMinibatches),
      "--checkpoint-interval",
      String(recipe.checkpointInterval),
      "--initial-std",
      String(recipe.initialStd),
      basketball
        ? "--no-normalize-rewards"
        : recipe.normalizeRewards
        ? "--normalize-rewards"
        : "--no-normalize-rewards",
      "--update-epochs",
      basketball ? "5" : recipe.profile === "smoke" ? "1" : String(recipe.updateEpochs),
      "--learning-rate",
      String(recipe.learningRate),
      "--gamma",
      String(recipe.gamma),
      "--gae-lambda",
      recipe.experimentId === "swing" ? "0.98" : "0.95",
      "--clip-coefficient",
      String(recipe.clipCoefficient),
      "--entropy-coefficient",
      String(basketball ? 0.01 : recipe.entropyCoefficient),
      "--max-grad-norm",
      String(basketball ? 1 : recipe.maxGradNorm),
      ...(recipe.experimentId === "swing"
        ? [
            "--swing-initial-angle-deg",
            String(recipe.swingInitialAngleDeg),
            "--swing-initial-rate-rad-s",
            String(recipe.swingInitialRateRadS),
          ]
        : []),
      ...commonArgs(recipe),
    ];
  }
  if (operation === "eval") {
    const settings = evaluationSettings(recipe);
    return [
      ...base,
      ...(recipe.experimentId === "basketball"
        ? ["--output-dir", paths.runDirectory]
        : []),
      ...(existsSync(paths.onnx)
        ? ["--policy", paths.onnx]
        : ["--checkpoint", paths.checkpoint]),
      "--backend",
      recipe.experimentId === "basketball" ? "standalone" : "dummy",
      "--evaluation-mode",
      settings.evaluation_mode,
      "--eval-steps",
      String(settings.eval_steps),
      ...(recipe.experimentId === "swing"
        ? ["--swing-min-span-deg", String(recipe.swingMinSpanDeg)]
        : []),
      ...commonArgs(recipe),
    ];
  }
  if (operation === "export") {
    if (recipe.experimentId === "basketball") {
      throw new Error("Basketball training emits its recurrent policy.onnx directly; standalone export is unavailable.");
    }
    return [
      ...base,
      "--recipe",
      recipe.experimentId,
      "--checkpoint",
      paths.checkpoint,
      "--output",
      paths.onnx,
    ];
  }
  return [
    ...base,
    ...(recipe.experimentId === "basketball"
      ? ["--output-dir", paths.runDirectory]
      : []),
    ...(existsSync(paths.onnx)
      ? ["--policy", paths.onnx]
      : ["--checkpoint", paths.checkpoint]),
    "--output",
    paths.renderDirectory,
    "--episodes",
    "1",
    "--width",
    "640",
    "--height",
    "360",
    "--camera",
    "three-quarter",
    "--render-seconds",
    String(recipe.renderSeconds),
    ...commonArgs(recipe),
  ];
}

function metadataContractVersion(metadata: Record<string, unknown> | null): unknown {
  const nested = metadata?.metadata;
  return metadata?.contract_version ?? (
    typeof nested === "object" && nested !== null && !Array.isArray(nested)
      ? (nested as Record<string, unknown>).contract_version
      : undefined
  );
}

function drawingEvidenceTool(
  report: Record<string, unknown>,
  metadata: Record<string, unknown> | null
): DrawingTool | null {
  const reportTool = drawingToolFromContract(report.contract_version);
  const metadataTool = drawingToolFromContract(metadataContractVersion(metadata));
  if (!reportTool || !metadataTool || reportTool !== metadataTool) return null;
  const request = report.evaluation_request;
  const recipe = typeof request === "object" && request !== null && !Array.isArray(request)
    ? (request as { recipe?: Partial<RlxRecipe> }).recipe
    : null;
  if (
    recipe?.drawingTool !== undefined &&
    recipe.drawingTool !== reportTool
  ) {
    return null;
  }
  return reportTool;
}

const BRUSH_SOURCE_HASH_KEYS = [
  "pipeline_sha256",
  "environment_sha256",
  "reference_sha256",
  "actor_sha256",
  "base_environment_sha256",
  "base_reference_sha256",
] as const;

function brushSourceHashes(
  value: unknown
): Record<(typeof BRUSH_SOURCE_HASH_KEYS)[number], string> | null {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    return null;
  }
  const hashes = value as Record<string, unknown>;
  if (
    Object.keys(hashes).length !== BRUSH_SOURCE_HASH_KEYS.length ||
    !BRUSH_SOURCE_HASH_KEYS.every(
      (key) => typeof hashes[key] === "string" && /^[a-f0-9]{64}$/.test(hashes[key])
    )
  ) {
    return null;
  }
  return hashes as Record<(typeof BRUSH_SOURCE_HASH_KEYS)[number], string>;
}

function validDrawingReportShape(
  report: Record<string, unknown>,
  drawingTool: DrawingTool
): boolean {
  const drawing = getDrawingTool(drawingTool);
  const evaluation = report.evaluation;
  const environment = typeof evaluation === "object" && evaluation !== null && !Array.isArray(evaluation)
    ? (evaluation as Record<string, unknown>).environment
    : null;
  const settings = typeof evaluation === "object" && evaluation !== null && !Array.isArray(evaluation)
    ? evaluation as Record<string, unknown>
    : null;
  const environmentRecord = typeof environment === "object" && environment !== null && !Array.isArray(environment)
    ? environment as Record<string, unknown>
    : null;
  const recipeOptions = environmentRecord?.recipe_options;
  const rewardWeights = environmentRecord?.reward_weights;
  const assessment = report.drawing_assessment;
  const assessmentRecord = typeof assessment === "object" && assessment !== null && !Array.isArray(assessment)
    ? assessment as Record<string, unknown>
    : null;
  const request = report.evaluation_request;
  const requestedRecipe = typeof request === "object" && request !== null && !Array.isArray(request)
    ? (request as { recipe?: Partial<RlxRecipe> }).recipe
    : null;
  let expectedRecipe: RlxRecipe | null = null;
  if (requestedRecipe) {
    try {
      expectedRecipe = normalizeRecipe({
        ...requestedRecipe,
        drawingTool,
      });
    } catch {
      return false;
    }
  }
  if (
    report.recipe !== "drawing" ||
    (report.evaluation_mode !== "skill" && report.evaluation_mode !== "pipeline") ||
    typeof report.finite !== "boolean" ||
    typeof report.pipeline_passed !== "boolean" ||
    typeof report.passed !== "boolean" ||
    (report.skill_status !== "passed" && report.skill_status !== "failed") ||
    settings?.mode !== report.evaluation_mode ||
    !Number.isInteger(settings.eval_episodes) ||
    Number(settings.eval_episodes) < 1 ||
    (expectedRecipe !== null && settings.eval_episodes !== expectedRecipe.evalEpisodes) ||
    (expectedRecipe !== null && settings.seed !== expectedRecipe.seed) ||
    environmentRecord?.recipe !== "drawing" ||
    environmentRecord?.actuator !== "xml" ||
    environmentRecord?.max_episode_s !== drawing.maxEpisodeSeconds ||
    (expectedRecipe !== null && environmentRecord.max_episode_s !== expectedRecipe.maxEpisodeS) ||
    environmentRecord?.assistance !== 0 ||
    typeof recipeOptions !== "object" ||
    recipeOptions === null ||
    Array.isArray(recipeOptions) ||
    Object.keys(recipeOptions as Record<string, unknown>).length !== 0 ||
    typeof rewardWeights !== "object" ||
    rewardWeights === null ||
    Array.isArray(rewardWeights) ||
    !assessmentRecord ||
    typeof assessmentRecord.passed !== "boolean" ||
    typeof assessmentRecord.unassisted !== "boolean" ||
    !Array.isArray(assessmentRecord.episodes) ||
    !assessmentRecord.episodes.every(
      (episode) => typeof episode === "object" && episode !== null && !Array.isArray(episode)
    )
  ) {
    return false;
  }
  if (drawingTool === "brush") {
    const episodes = assessmentRecord.episodes as Record<string, unknown>[];
    const conditionCounts = new Map<string, number>();
    for (const episode of episodes) {
      if (typeof episode.case !== "string") return false;
      conditionCounts.set(
        episode.case,
        (conditionCounts.get(episode.case) ?? 0) + 1
      );
    }
    return (
      report.unique_conditions === drawing.evaluationConfigs &&
      report.minimum_seeds_per_condition === drawing.evalEpisodes &&
      report.environment_source_match === true &&
      Array.isArray(report.provenance_errors) &&
      brushSourceHashes(report.source_hashes) !== null &&
      episodes.length === drawing.evaluationConfigs * drawing.evalEpisodes &&
      conditionCounts.size === drawing.evaluationConfigs &&
      [...conditionCounts.values()].every(
        (count) => count === drawing.evalEpisodes
      ) &&
      typeof assessmentRecord.mean_coverage === "number" &&
      Number.isFinite(assessmentRecord.mean_coverage) &&
      typeof assessmentRecord.mean_precision === "number" &&
      Number.isFinite(assessmentRecord.mean_precision) &&
      Number.isInteger(assessmentRecord.accepted_episodes) &&
      Number(assessmentRecord.accepted_episodes) >= 0 &&
      Number(assessmentRecord.accepted_episodes) <= episodes.length &&
      assessmentRecord.required_episodes === episodes.length &&
      (assessmentRecord.passed !== true ||
        assessmentRecord.accepted_episodes === episodes.length)
    );
  }
  return true;
}

async function readDrawingMetadata(
  experimentId: ExperimentId,
  runName: string
): Promise<Record<string, unknown> | null> {
  if (experimentId !== "drawing") return null;
  try {
    const metadata = JSON.parse(
      await readFile(pathsFor(experimentId, runName).metadata, "utf8")
    );
    return typeof metadata === "object" && metadata !== null && !Array.isArray(metadata)
      ? metadata as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

/** Only current owned artifact bytes can supply an authoritative evaluation verdict. */
async function boundEvaluation(
  report: Record<string, unknown> | null,
  experimentId: ExperimentId,
  runName: string,
  environmentKey: string | undefined,
  drawingMetadata: Record<string, unknown> | null = null
): Promise<Record<string, unknown> | null> {
  if (!report) return null;
  if (report.passed === undefined && typeof report.success === "boolean") {
    report = { ...report, passed: report.success };
  }
  if (report.source_sha256 == null) {
    return { ...report, passed: false, skill_status: "not_assessed" };
  }
  const paths = pathsFor(experimentId, runName);
  const source = report.source_type === "policy"
    ? paths.onnx
    : report.source_type === "checkpoint" ? paths.checkpoint : null;
  if (!source || typeof report.source_sha256 !== "string") return null;
  const files = report.source_files_sha256;
  if (typeof files !== "object" || files === null || Array.isArray(files)) return null;
  const hashes = files as Record<string, unknown>;
  const ownedSources = experimentId === "drawing" && report.source_type === "policy"
    ? [source, paths.metadata]
    : report.source_type === "checkpoint" ? [source, paths.metadata] : [source];
  for (const file of ownedSources) {
    if (typeof hashes[file] !== "string") return null;
    let bytes: Buffer;
    try {
      bytes = await readFile(file);
    } catch {
      // Missing or unreadable owned artifacts cannot retain a measured verdict.
      return null;
    }
    const hash = createHash("sha256").update(bytes).digest("hex");
    if (hash !== hashes[file] || (file === source && hash !== report.source_sha256)) return null;
  }
  if (experimentId === "drawing") {
    try {
      const metadata = drawingMetadata ?? JSON.parse(
        await readFile(paths.metadata, "utf8")
      ) as Record<string, unknown>;
      const drawingTool = drawingEvidenceTool(report, metadata);
      if (!drawingTool || !validDrawingReportShape(report, drawingTool)) {
        return null;
      }
      if (drawingTool === "brush") {
        const reportHashes = brushSourceHashes(report.source_hashes);
        const metadataHashes = brushSourceHashes(metadata.source_hashes);
        if (
          !reportHashes ||
          !metadataHashes ||
          BRUSH_SOURCE_HASH_KEYS.some(
            (key) => reportHashes[key] !== metadataHashes[key]
          )
        ) {
          return null;
        }
      }
      const checkpointHash = createHash("sha256")
        .update(await readFile(paths.checkpoint))
        .digest("hex");
      const onnx = metadata.onnx as Record<string, unknown> | undefined;
      if (
        metadata.checkpoint_sha256 !== checkpointHash ||
        onnx?.sha256 !== report.source_sha256
      ) {
        return null;
      }
    } catch {
      return null;
    }
  }
  if (experimentId === "dance") {
    const options = (report.evaluation as { environment?: { recipe_options?: Record<string, unknown> } } | undefined)?.environment?.recipe_options;
    if (typeof options?.dance_clip_sha256 === "string") {
      try {
        const clip = normalizeDanceClip("dance", options.dance_clip);
        if (!clip || createHash("sha256").update(await readFile(pathToFileURL(clip))).digest("hex") !== options.dance_clip_sha256) {
          return { ...report, passed: false, skill_status: "not_assessed", evaluation_settings_match: false };
        }
      } catch {
        return { ...report, passed: false, skill_status: "not_assessed", evaluation_settings_match: false };
      }
    }
  }
  if (experimentId === "backflip") {
    const options = (report.evaluation as { environment?: { recipe_options?: Record<string, unknown> } } | undefined)?.environment?.recipe_options;
    const protocolVersion = options?.backflip_protocol_version;
    const expectedStandHash = options?.stand_policy_sha256;
    const standPolicy = path.resolve(paths.root, "../microduck/policies/alpha_stand.onnx");
    if (
      protocolVersion !== BACKFLIP_PROTOCOL_VERSION ||
      typeof expectedStandHash !== "string"
    ) {
      return { ...report, passed: false, skill_status: "not_assessed", evaluation_settings_match: false };
    }
    try {
      const currentStandHash = createHash("sha256")
        .update(await readFile(standPolicy))
        .digest("hex");
      if (currentStandHash !== expectedStandHash) {
        return { ...report, passed: false, skill_status: "not_assessed", evaluation_settings_match: false };
      }
    } catch {
      return { ...report, passed: false, skill_status: "not_assessed", evaluation_settings_match: false };
    }
  }
  if (environmentKey !== undefined) {
    const request = report.evaluation_request as { recipe?: Partial<RlxRecipe> } | undefined;
    let matches = false;
    try {
      const inferredDrawingTool = experimentId === "drawing"
        ? drawingEvidenceTool(report, drawingMetadata)
        : null;
      matches = request?.recipe != null &&
        evaluationEnvironmentKey(normalizeRecipe({
          ...request.recipe,
          ...(inferredDrawingTool && request.recipe.drawingTool === undefined
            ? { drawingTool: inferredDrawingTool }
            : {}),
        })) === environmentKey;
    } catch {
      matches = false;
    }
    if (!matches) {
      return { ...report, passed: false, skill_status: "not_assessed", evaluation_settings_match: false };
    }
  }
  return report;
}

export async function snapshot(
  experimentIdValue?: unknown,
  runNameValue?: unknown
): Promise<RlxJobSnapshot> {
  const state = { ...job, logs: [...job.logs], rewardHistory: [...job.rewardHistory] };
  const experimentId = experimentIdValue
    ? normalizeExperimentId(experimentIdValue)
    : state.experimentId;
  const runName = runNameValue
    ? sanitizeRunName(runNameValue)
    : state.runName;
  const sameRun =
    runName === state.runName && experimentId === state.experimentId;
  const environmentKey = state.environmentKeys[`${experimentId}/${runName}`];
  let savedEvaluation: Record<string, unknown> | null = null;
  const evaluationPaths = pathsFor(experimentId, runName);
  for (const candidate of [
    evaluationPaths.evaluation,
    evaluationPaths.evaluationFallback,
  ]) {
    try {
      savedEvaluation = JSON.parse(
        await readFile(candidate, "utf8")
      ) as Record<string, unknown>;
      break;
    } catch {
      savedEvaluation = null;
    }
  }
  const drawingMetadata = await readDrawingMetadata(experimentId, runName);
  const evaluation = await boundEvaluation(
    sameRun && state.operation === "train"
      ? state.evaluation
      : (sameRun ? state.evaluation : null) ?? savedEvaluation,
    experimentId,
    runName,
    environmentKey,
    drawingMetadata
  );
  const restoredDrawingTool = experimentId === "drawing"
    ? evaluation
      ? drawingEvidenceTool(evaluation, drawingMetadata)
      : drawingToolFromContract(metadataContractVersion(drawingMetadata))
    : null;
  let savedRecipe: RlxRecipe | null = null;
  if (evaluation?.evaluation_settings_match !== false) {
    const request = evaluation?.evaluation_request;
    const recipe =
      typeof request === "object" && request !== null && !Array.isArray(request)
        ? (request as { recipe?: Partial<RlxRecipe> }).recipe
        : null;
    if (recipe) {
      try {
        savedRecipe = normalizeRecipe({
          ...recipe,
          ...(restoredDrawingTool && recipe.drawingTool === undefined
            ? { drawingTool: restoredDrawingTool }
            : {}),
        });
        if (savedRecipe.experimentId !== experimentId || savedRecipe.runName !== runName) savedRecipe = null;
      } catch {
        savedRecipe = null;
      }
    }
  }
  const paths = pathsFor(experimentId, runName);
  const evaluatedBackflipRecipeOptions = experimentId === "backflip"
    ? backflipRecipeOptions(evaluation)
    : null;
  const renderSource = experimentId === "drawing"
    ? paths.onnx
    : evaluation?.source_type === "policy" ? paths.onnx : paths.checkpoint;
  const renderEvidence = savedRecipe && evaluation?.source_sha256
    ? readRenderEvidence(path.join(paths.renderDirectory, "evidence.json"), {
        ...renderEvidenceInputs(savedRecipe, renderSource, String(evaluation.source_type)),
        recipeKey: evaluationEnvironmentKey(savedRecipe, "render"),
        ...(evaluatedBackflipRecipeOptions
          ? { recipeOptions: evaluatedBackflipRecipeOptions }
          : {}),
        video: paths.renderVideo,
        sheet: paths.renderSheet,
      }) : null;
  const evaluatedClipHash = (evaluation?.evaluation as { environment?: { recipe_options?: Record<string, unknown> } } | undefined)?.environment?.recipe_options?.dance_clip_sha256;
  const standPolicy = path.resolve(paths.root, "../microduck/policies/alpha_stand.onnx");
  const renderVerified = renderEvidence !== null && (experimentId !== "dance" ||
    (typeof evaluatedClipHash === "string" && renderEvidence.clipSha256 === evaluatedClipHash)) &&
    (experimentId !== "backflip" ||
      (evaluatedBackflipRecipeOptions !== null &&
        renderEvidence.recipeOptions?.backflip_protocol_version === evaluatedBackflipRecipeOptions.backflip_protocol_version &&
        renderEvidence.recipeOptions?.stand_policy_sha256 === evaluatedBackflipRecipeOptions.stand_policy_sha256 &&
        renderEvidence.externalFilesSha256?.[standPolicy] === evaluatedBackflipRecipeOptions.stand_policy_sha256));
  const trainingHistory = collectTrainingHistory(
    paths.root,
    paths.runDirectory,
    paths.metadata,
    paths.trainingMetrics,
    sameRun ? state.trainingHistoryContext : null
  );
  const restoredTraining = restoreTrainingInvocation(
    paths.runDirectory,
    paths.metadata,
    trainingHistory
  );
  const hasMemoryTraining =
    sameRun &&
    (state.trainingHistoryContext !== null ||
      state.rewardHistory.length > 0 ||
      state.trainingSteps > 0 ||
      state.trainingTotal > 0);
  return {
    renderVerified,
    renderEvidenceId: renderVerified ? renderEvidence?.video.sha256 ?? null : null,
    phase: sameRun ? state.phase : "idle",
    operation: sameRun ? state.operation : null,
    activeJob:
      state.phase === "running" && state.operation
        ? {
            operation: state.operation,
            experimentId: state.experimentId,
            runName: state.runName,
            startedAt: state.startedAt,
          }
        : null,
    experimentId,
    drawingTool: restoredDrawingTool ?? undefined,
    runName,
    startedAt: sameRun ? state.startedAt : null,
    finishedAt: sameRun ? state.finishedAt : null,
    exitCode: sameRun ? state.exitCode : null,
    logs: sameRun ? state.logs : [],
    result: sameRun ? state.result : null,
    evaluation,
    savedRecipe,
    rewardHistory:
      sameRun && state.rewardHistory.length
        ? state.rewardHistory
        : restoredTraining.rewardHistory,
    trainingHistory,
    normalizeRewards: hasMemoryTraining
      ? state.normalizeRewards
      : restoredTraining.normalizeRewards,
    trainingSteps: hasMemoryTraining
      ? state.trainingSteps
      : restoredTraining.trainingSteps,
    trainingTotal: hasMemoryTraining
      ? state.trainingTotal
      : restoredTraining.trainingTotal,
    artifacts: await artifactState(experimentId, runName),
  };
}

export function startJob(
  operation: RlxOperation,
  recipeInput: Partial<RlxRecipe>
): RlxRecipe {
  if (existsSync(path.resolve(rlxRoot(), "../.restart-lab/lock"))) {
    throw new Error("Lab restart in progress. Wait for restart verification to finish before launching RLX jobs.");
  }
  if (job.child && job.phase === "running") {
    throw new Error(`${job.operation ?? "RLX"} is already running.`);
  }
  const recipe = normalizeRecipe(recipeInput);
  const paths = pathsFor(recipe.experimentId, recipe.runName);
  if (recipe.experimentId === "basketball" && operation !== "export") {
    const workspace = path.resolve(paths.root, "..");
    const referenceDirectory = path.join(
      workspace,
      "microduck-playground/src/mjlab_microduck/robot/assets/basketball"
    );
    const required = [
      path.join(referenceDirectory, "basketball.obj"),
      path.join(referenceDirectory, "basketball.png"),
    ];
    if (operation === "train") {
      const sourceDirectory = recipe.resumeFromCheckpoint
        ? paths.runDirectory
        : path.join(workspace, "microduck-playground/artifacts/basketball");
      required.push(
        path.join(sourceDirectory, "checkpoint.pt"),
        path.join(sourceDirectory, "policy.onnx")
      );
    } else {
      required.push(paths.onnx);
    }
    const missing = required.filter((file) => !existsSync(file));
    if (missing.length) {
      throw new Error(
        `Basketball local reference setup is incomplete. Missing: ${missing.join(", ")}. ` +
        "Install the supplied checkpoint/policy and basketball mesh/texture at these exact paths; Studio does not download reference assets."
      );
    }
  }
  if (operation === "export" && !existsSync(paths.checkpoint)) {
    throw new Error("Export requires a checkpoint; an ONNX policy cannot be re-exported.");
  }
  if (
    recipe.experimentId === "drawing" &&
    operation !== "train" &&
    (
      !existsSync(paths.checkpoint) ||
      !existsSync(paths.metadata) ||
      (operation !== "export" && !existsSync(paths.onnx))
    )
  ) {
    throw new Error(
      operation === "export"
        ? "Drawing export requires policy.zip and policy.zip.json."
        : "Drawing evaluation and rendering require policy.zip, policy.zip.json, and policy.onnx from the same run."
    );
  }
  if (
    operation !== "train" &&
    !existsSync(paths.checkpoint) &&
    !existsSync(paths.onnx)
  ) {
    throw new Error("Train or select a run with a checkpoint before continuing.");
  }
  if (
    operation === "train" &&
    recipe.resumeFromCheckpoint &&
    !existsSync(paths.checkpoint)
  ) {
    throw new Error(
      "This run has no checkpoint to continue. Run the Discovery stage first."
    );
  }
  if (
    operation === "train" &&
    !recipe.resumeFromCheckpoint &&
    existsSync(paths.checkpoint)
  ) {
    throw new Error(
      "This run already has a checkpoint. Choose a new run name or enable resume to preserve the saved artifacts."
    );
  }
  const trainingStartStep =
    operation === "train"
      ? prepareTrainingHistory(
          paths.root,
          paths.runDirectory,
          paths.metadata,
          paths.trainingMetrics
        )
      : 0;

  const renderSource = recipe.experimentId === "drawing"
    ? paths.onnx
    : existsSync(paths.onnx) ? paths.onnx : paths.checkpoint;
  const renderContext = operation === "render" ? prepareRenderEvidence({
    ...renderEvidenceInputs(recipe, renderSource, existsSync(paths.onnx) ? "policy" : "checkpoint"),
    recipeKey: evaluationEnvironmentKey(recipe, "render"),
  }) : null;
  const child = spawn(process.env.MICRODUCK_STUDIO_PYTHON_DIRECT ? "/usr/bin/env" : "uv", commandFor(operation, recipe), {
    cwd: paths.root,
    env: {
      ...process.env,
      VIRTUAL_ENV: undefined,
      UV_PYTHON_PREFERENCE: "only-system",
      PYTHONUNBUFFERED: "1",
    },
    stdio: ["ignore", "pipe", "pipe"],
  });
  const changedRun =
    job.experimentId !== recipe.experimentId || job.runName !== recipe.runName;
  if (operation !== "export") {
    job.environmentKeys[`${recipe.experimentId}/${recipe.runName}`] = evaluationEnvironmentKey(recipe);
  }
  job.phase = "running";
  job.operation = operation;
  job.experimentId = recipe.experimentId;
  job.runName = recipe.runName;
  job.startedAt = new Date().toISOString();
  job.finishedAt = null;
  job.exitCode = null;
  job.logs = [
    `[studio] ${operation} started`,
    `[studio] experiment: ${recipe.experimentId}`,
    `[studio] run: ${recipe.runName}`,
  ];
  job.result = null;
  if (operation === "train") {
    job.evaluation = null;
    job.rewardHistory = [];
    job.normalizeRewards = recipe.normalizeRewards;
    job.trainingSteps = 0;
    job.trainingTotal = recipe.totalTimesteps;
    job.trainingHistoryContext = {
      id: `active-${job.startedAt}`,
      startStep: recipe.resumeFromCheckpoint ? trainingStartStep : 0,
      normalizeRewards: recipe.normalizeRewards,
      totalTimesteps: recipe.totalTimesteps,
      status: "active",
    };
  } else if (changedRun) {
    job.evaluation = null;
    job.rewardHistory = [];
    job.trainingSteps = 0;
    job.trainingTotal = 0;
    job.trainingHistoryContext = null;
  }
  job.stdoutBuffer = "";
  job.stderrBuffer = "";
  job.child = child;

  const generation = ++job.generation;
  const ownsJob = () => job.generation === generation;
  child.stdout?.on("data", (chunk) => {
    if (ownsJob() && job.phase === "running") appendStream("stdout", chunk);
  });
  child.stderr?.on("data", (chunk) => {
    if (ownsJob() && job.phase === "running") appendStream("stderr", chunk);
  });
  child.on("error", (error) => {
    if (ownsJob() && job.phase === "running") appendLine(`[studio] ${error.message}`);
  });
  child.on("close", (code, signal) => {
    if (!ownsJob()) return;
    if (job.phase === "cancelled") {
      job.child = null;
      return;
    }
    flushStreams();
    job.exitCode = code;
    job.finishedAt = new Date().toISOString();
    job.child = null;
    job.phase = code === 0 ? "succeeded" : "failed";
    if (operation === "train" && job.trainingHistoryContext) {
      job.trainingHistoryContext = {
        ...job.trainingHistoryContext,
        status: "complete",
      };
    }
    if (signal) appendLine(`[studio] process ended by ${signal}`);
    const jsonLine = [...job.logs]
      .reverse()
      .find((line) => line.startsWith("{") && line.endsWith("}"));
    const resultText = jsonLine ?? (
      operation === "eval" &&
      recipe.experimentId === "drawing" &&
      existsSync(paths.evaluation)
        ? readFileSync(paths.evaluation, "utf8")
        : null
    );
    if (resultText) {
      try {
        job.result = JSON.parse(resultText) as Record<string, unknown>;
        if (operation === "eval") {
          job.result = {
            ...job.result,
            evaluation_request: evaluationSettings(recipe),
          };
          job.evaluation = job.result;
          void writeFile(
            paths.evaluation,
            `${JSON.stringify(job.result, null, 2)}\n`
          ).catch((error: unknown) => {
            if (!ownsJob()) return;
            appendLine(
              `[studio] could not persist evaluation: ${
                error instanceof Error ? error.message : String(error)
              }`
            );
          });
        }
      } catch {
        job.result = null;
      }
    }
    if (code === 0 && renderContext) {
      const recipeOptions = recipe.experimentId === "backflip"
        ? backflipRecipeOptions(job.result)
        : undefined;
      const receipt = recipe.experimentId === "backflip" && !recipeOptions
        ? null
        : finalizeRenderEvidence(renderContext, {
            video: paths.renderVideo,
            sheet: paths.renderSheet,
            receipt: path.join(paths.renderDirectory, "evidence.json"),
          }, recipeOptions ?? undefined);
      if (!receipt) appendLine("[studio] Render completed but source/media provenance could not be verified; visual review remains gated.");
    }
  });
  return recipe;
}

export function cancelJob(): boolean {
  if (!job.child || job.phase !== "running") return false;
  job.phase = "cancelled";
  job.finishedAt = new Date().toISOString();
  job.child.kill("SIGTERM");
  appendLine("[studio] cancellation requested");
  return true;
}

export async function readArtifact(
  experimentIdValue: unknown,
  runNameValue: unknown,
  kind: "checkpoint" | "metadata" | "onnx" | "sheet" | "video"
) {
  const filePath = artifactPath(experimentIdValue, runNameValue, kind);
  const [data, fileStat] = await Promise.all([readFile(filePath), stat(filePath)]);
  return { data, filePath, size: fileStat.size };
}

export async function readDanceChoreography(): Promise<string> {
  return readFile(
    path.join(rlxRoot(), "assets", "clips", "dance-120bpm.json"),
    "utf8"
  );
}
