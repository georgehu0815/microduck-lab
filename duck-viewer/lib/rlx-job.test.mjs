import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";
import { test } from "node:test";
import { createRequire } from "node:module";
import { createHash } from "node:crypto";
import ts from "typescript";

const loadDependency = createRequire(import.meta.url);

// Compile with the project's installed TypeScript; all subprocesses and artifact I/O are fakes.
function load(filename, dependencies = {}) {
  const source = fs.readFileSync(new URL(filename, import.meta.url), "utf8");
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  });
  const exports = {};
  vm.runInNewContext(outputText, {
    exports, process, Buffer,
    require: (id) => id in dependencies ? dependencies[id] : loadDependency(id),
  }, { filename });
  return exports;
}
const experiments = load("experiments.ts");
const { evaluationVerdict } = load("evaluation.ts");
const plain = (value) => JSON.parse(JSON.stringify(value));

function fixture() {
  const files = new Map();
  const children = [];
  const reads = [];
  const renderFinalizations = [];
  const renderEvidenceReads = [];
  let clock = 0;
  let writeImplementation;
  let readRenderEvidenceImplementation = () => null;
  const fsDependency = {
    existsSync: (file) => file.endsWith("examples/ppo_microduck_dance.py") || files.has(file),
    realpathSync: (file) => file,
    statSync: (file) => {
      if (!files.has(file)) throw new Error("ENOENT");
      const value = files.get(file);
      return {
        isFile: () => true,
        size: Buffer.byteLength(value),
        mtimeMs: clock,
      };
    },
    readFileSync: (file, encoding) => {
      reads.push(file);
      if (!files.has(file)) throw new Error("ENOENT");
      const value = files.get(file);
      return encoding ? String(value) : Buffer.from(value);
    },
    writeFileSync: (file, data) => {
      clock += 1;
      files.set(file, data);
    },
    renameSync: (source, destination) => {
      clock += 1;
      files.set(destination, files.get(source));
      files.delete(source);
    },
  };
  const history = load("rlx-history.ts", {
    "node:fs": fsDependency,
  });
  const api = load("rlx-job.ts", {
    "@/lib/experiments": experiments,
    "@/lib/rlx-history": history,
    "@/lib/rlx-render-evidence": {
      prepareRenderEvidence: (input) => input,
      finalizeRenderEvidence: (context, output, recipeOptions) => {
        renderFinalizations.push({ context, output, recipeOptions });
        return {};
      },
      readRenderEvidence: (receiptPath, expected) => {
        renderEvidenceReads.push({ receiptPath, expected });
        return readRenderEvidenceImplementation(receiptPath, expected);
      },
    },
    "node:fs": fsDependency,
    "node:fs/promises": {
      readFile: async (file) => {
        if (typeof file !== "string") file = fileURLToPath(file);
        reads.push(file);
        if (!files.has(file)) throw new Error("ENOENT");
        return files.get(file);
      },
      writeFile: async (file, data) => {
        if (writeImplementation) return writeImplementation(file, data);
        files.set(file, data);
      },
    },
    "node:child_process": {
      spawn: (command, args, options) => {
        const child = new EventEmitter();
        Object.assign(child, {
          command, args, options,
          stdout: new EventEmitter(), stderr: new EventEmitter(),
          kill: (signal) => { child.killed = signal; return true; },
        });
        children.push(child);
        return child;
      },
    },
  });
  const artifact = (runName, kind = "checkpoint", experimentId = "swing") =>
    api.artifactPath(experimentId, runName, kind);
  const clip = (name = "custom.json", location = "dance-clip") =>
    location === "rlx-artifacts"
      ? path.resolve(process.cwd(), "../rlx/artifacts", name)
      : path.resolve(process.cwd(), "../dance-clip", name);
  const standPolicy = path.resolve(process.cwd(), "../microduck/policies/alpha_stand.onnx");
  const basketballBootstrap = path.resolve(process.cwd(), "../microduck-playground/artifacts/basketball");
  const basketballAssets = path.resolve(process.cwd(), "../microduck-playground/src/mjlab_microduck/robot/assets/basketball");
  return {
    api, files, children, artifact, clip, standPolicy, reads,
    renderFinalizations, renderEvidenceReads,
    add: (runName, kind = "checkpoint", experimentId = "swing") => files.set(artifact(runName, kind, experimentId), "artifact"),
    addClip: (name, location) => {
      const file = clip(name, location);
      files.set(file, "{}");
      return file;
    },
    addStandPolicy: (bytes = "stand policy") => files.set(standPolicy, bytes),
    addBasketballReferences: () => {
      for (const file of [
        path.join(basketballBootstrap, "checkpoint.pt"),
        path.join(basketballBootstrap, "policy.onnx"),
        path.join(basketballAssets, "basketball.obj"),
        path.join(basketballAssets, "basketball.png"),
      ]) files.set(file, "basketball reference");
    },
    deferWrites: (implementation) => { writeImplementation = implementation; },
    setReadRenderEvidence: (implementation) => {
      readRenderEvidenceImplementation = implementation;
    },
  };
}
const input = (runName = "test-a", extra = {}) => ({ experimentId: "swing", runName, ...extra });
const arg = (child, flag) => child.args[child.args.indexOf(flag) + 1];
function sourceReport(f, runName = "test-a", sourceType = "policy", experimentId = "swing") {
  const source = f.artifact(runName, sourceType === "policy" ? "onnx" : "checkpoint", experimentId);
  const files = sourceType === "policy"
    ? experimentId === "drawing"
      ? [source, f.artifact(runName, "metadata", experimentId)]
      : [source]
    : [source, f.artifact(runName, "metadata", experimentId)];
  const hashes = Object.fromEntries(files.map((file) => [file, createHash("sha256").update(f.files.get(file)).digest("hex")]));
  return {
    source_type: sourceType, source_sha256: hashes[source], source_files_sha256: hashes,
    evaluation_mode: "skill", skill_status: "passed", passed: true, pipeline_passed: true, finite: true,
  };
}
function finish(child, report, code = 0) {
  if (report) child.stdout.emit("data", `${JSON.stringify(report)}\n`);
  child.emit("close", code, null);
}
function progress(child) {
  child.stdout.emit("data", '{"event":"training_progress","steps":4,"total":4,"mean_reward":2.5}\n');
}

test("restart drain lock rejects every launch without spawning or changing artifacts", () => {
  const f = fixture();
  const lock = path.resolve(process.cwd(), "../.restart-lab/lock");
  f.files.set(lock, "locked");
  for (const operation of ["train", "eval", "render", "export"]) {
    assert.throws(() => f.api.startJob(operation, input()), /Lab restart in progress/);
  }
  assert.equal(f.children.length, 0);
  assert.equal(f.files.size, 1);
  f.files.delete(lock);
  f.api.startJob("train", input());
  assert.equal(f.children.length, 1);
});

test("a preinstalled Python runtime bypasses uv for all four operation types", () => {
  const previous = process.env.MICRODUCK_STUDIO_PYTHON_DIRECT;
  process.env.MICRODUCK_STUDIO_PYTHON_DIRECT = "/installed/env/bin/python";
  try {
    for (const operation of ["train", "eval", "render", "export"]) {
      const f = fixture();
      if (operation !== "train") f.add("test-a");
      f.api.startJob(operation, input());
      const child = f.children[0];
      assert.equal(child.command, "/usr/bin/env");
      assert.equal(child.args[0], "/installed/env/bin/python");
      assert.equal(child.args[1], "examples/ppo_microduck_studio.py");
      assert.equal(child.args[2], operation);
      assert.equal(child.args.includes("--with-editable"), false);
    }
  } finally {
    if (previous === undefined) delete process.env.MICRODUCK_STUDIO_PYTHON_DIRECT;
    else process.env.MICRODUCK_STUDIO_PYTHON_DIRECT = previous;
  }
});

test("export includes the CLI-required recipe and checkpoint/output without environment arguments", () => {
  const f = fixture();
  f.add("test-a", "onnx");
  assert.throws(() => f.api.startJob("export", input()), /Export requires a checkpoint/);
  assert.equal(f.children.length, 0);
  f.add("test-a");
  f.api.startJob("export", input());
  const child = f.children[0];
  assert.equal(child.command, "uv");
  assert.equal(arg(child, "--recipe"), "swing");
  assert.equal(child.args.includes("--num-envs"), false);
  assert.equal(child.args.includes("--weight-overrides"), false);
  assert.deepEqual(Array.from(child.args.slice(child.args.indexOf("export") + 1)), [
    "--recipe", "swing", "--checkpoint", f.artifact("test-a"), "--output", f.artifact("test-a", "onnx"),
  ]);
});

test("Basketball uses recurrent artifact names and the standalone shared-shaped adapter", () => {
  const train = fixture();
  train.addBasketballReferences();
  const recipe = train.api.startJob("train", {
    experimentId: "basketball",
    runName: "basketball-balance-01",
    profile: "full",
  });
  const child = train.children[0];
  assert.equal(child.args[child.args.indexOf("examples/ppo_microduck_balance.py") + 1], "train");
  assert.equal(arg(child, "--recipe"), "basketball");
  assert.equal(arg(child, "--checkpoint"), train.artifact("basketball-balance-01", "checkpoint", "basketball"));
  assert.equal(arg(child, "--onnx-output"), train.artifact("basketball-balance-01", "onnx", "basketball"));
  assert.match(train.artifact("basketball-balance-01", "checkpoint", "basketball"), /checkpoint\.pt$/);
  assert.match(train.artifact("basketball-balance-01", "metadata", "basketball"), /checkpoint\.pt\.json$/);
  assert.match(train.artifact("basketball-balance-01", "onnx", "basketball"), /policy\.onnx$/);
  assert.equal(recipe.maxEpisodeS, 60);
  assert.equal(recipe.evalSteps, 3000);
  assert.equal(recipe.bridgeCurriculum, false);
  assert.deepEqual(JSON.parse(arg(child, "--weight-overrides")), {});
  assert.equal(arg(child, "--output-dir"), path.dirname(train.artifact("basketball-balance-01", "checkpoint", "basketball")));
  assert.equal(arg(child, "--num-minibatches"), "1");
  assert.equal(arg(child, "--update-epochs"), "5");
  assert.equal(arg(child, "--entropy-coefficient"), "0.01");
  assert.equal(arg(child, "--max-grad-norm"), "1");
  assert.equal(child.args.includes("--no-normalize-rewards"), true);

  const evaluate = fixture();
  evaluate.addBasketballReferences();
  evaluate.add("basketball-balance-01", "onnx", "basketball");
  evaluate.api.startJob("eval", {
    experimentId: "basketball",
    runName: "basketball-balance-01",
    profile: "full",
  });
  assert.equal(evaluate.children[0].args.includes("examples/ppo_microduck_balance.py"), true);
  assert.equal(arg(evaluate.children[0], "--policy"), evaluate.artifact("basketball-balance-01", "onnx", "basketball"));
  assert.equal(arg(evaluate.children[0], "--backend"), "standalone");

  const exportAttempt = fixture();
  exportAttempt.add("basketball-balance-01", "checkpoint", "basketball");
  assert.throws(() => exportAttempt.api.startJob("export", {
    experimentId: "basketball",
    runName: "basketball-balance-01",
  }), /emits its recurrent policy\.onnx directly/);
  assert.equal(exportAttempt.children.length, 0);
});

test("Drawing uses only the agreed independent CLI adapter flags and normal artifacts", () => {
  const train = fixture();
  const recipe = train.api.startJob("train", {
    experimentId: "drawing",
    drawingTool: "pencil",
    runName: "drawing-pilot-01",
    profile: "full",
  });
  const trainChild = train.children[0];
  assert.equal(trainChild.args.includes("examples/ppo_microduck_drawing.py"), true);
  assert.deepEqual(Array.from(trainChild.args.slice(trainChild.args.indexOf("train") + 1)), [
    "--output", path.dirname(train.artifact("drawing-pilot-01", "checkpoint", "drawing")),
    "--total-timesteps", "32768",
    "--seed", "7",
    "--max-episode-s", "32",
  ]);
  assert.equal(recipe.initialStd, 0.02);
  const normalized = train.api.normalizeRecipe({
    experimentId: "drawing",
    drawingTool: "pencil",
    profile: "smoke",
    totalTimesteps: 4,
    numEnvs: 64,
    numSteps: 2,
    numMinibatches: 8,
    maxEpisodeS: 1,
    initialStd: 0.9,
    normalizeRewards: true,
    freezeObservationNormalization: true,
    checkpointInterval: 10,
    learningRate: 0.2,
    gamma: 0.8,
    clipCoefficient: 0.9,
    updateEpochs: 20,
    entropyCoefficient: 1,
    maxGradNorm: 10,
    domainRand: true,
    obsNoise: true,
    actionDelay: true,
    randomYaw: true,
    resumeFromCheckpoint: true,
  });
  assert.deepEqual(plain({
    totalTimesteps: normalized.totalTimesteps,
    numEnvs: normalized.numEnvs,
    numSteps: normalized.numSteps,
    numMinibatches: normalized.numMinibatches,
    maxEpisodeS: normalized.maxEpisodeS,
    initialStd: normalized.initialStd,
    normalizeRewards: normalized.normalizeRewards,
    freezeObservationNormalization: normalized.freezeObservationNormalization,
    checkpointInterval: normalized.checkpointInterval,
    learningRate: normalized.learningRate,
    gamma: normalized.gamma,
    clipCoefficient: normalized.clipCoefficient,
    updateEpochs: normalized.updateEpochs,
    entropyCoefficient: normalized.entropyCoefficient,
    maxGradNorm: normalized.maxGradNorm,
    domainRand: normalized.domainRand,
    obsNoise: normalized.obsNoise,
    actionDelay: normalized.actionDelay,
    randomYaw: normalized.randomYaw,
    resumeFromCheckpoint: normalized.resumeFromCheckpoint,
  }), {
    totalTimesteps: 768,
    numEnvs: 1,
    numSteps: 256,
    numMinibatches: 1,
    maxEpisodeS: 32,
    initialStd: 0.02,
    normalizeRewards: false,
    freezeObservationNormalization: false,
    checkpointInterval: 0,
    learningRate: 0.00003,
    gamma: 0.995,
    clipCoefficient: 0.1,
    updateEpochs: 5,
    entropyCoefficient: 0,
    maxGradNorm: 0.5,
    domainRand: false,
    obsNoise: false,
    actionDelay: false,
    randomYaw: false,
    resumeFromCheckpoint: false,
  });
  assert.equal(recipe.evalEpisodes, 8);
  assert.deepEqual(plain(recipe.rewardWeights), {});
  assert.match(train.artifact("drawing-pilot-01", "checkpoint", "drawing"), /policy\.zip$/);
  assert.match(train.artifact("drawing-pilot-01", "metadata", "drawing"), /policy\.zip\.json$/);
  assert.match(train.artifact("drawing-pilot-01", "onnx", "drawing"), /policy\.onnx$/);
  assert.match(train.artifact("drawing-pilot-01", "evaluation", "drawing"), /eval\.json$/);
  assert.match(train.artifact("drawing-pilot-01", "sheet", "drawing"), /render[/\\]frame_sheet\.png$/);
  assert.match(train.artifact("drawing-pilot-01", "video", "drawing"), /render[/\\]rollout\.mp4$/);

  for (const operation of ["eval", "render", "export"]) {
    const f = fixture();
    f.add("drawing-pilot-01", "checkpoint", "drawing");
    f.add("drawing-pilot-01", "metadata", "drawing");
    f.add("drawing-pilot-01", "onnx", "drawing");
    f.api.startJob(operation, {
      experimentId: "drawing",
      drawingTool: "pencil",
      runName: "drawing-pilot-01",
      profile: "full",
    });
    const child = f.children[0];
    const args = Array.from(child.args.slice(child.args.indexOf(operation) + 1));
    assert.equal(args.includes("--recipe"), false);
    assert.equal(args.includes("--backend"), false);
    assert.equal(args.includes("--num-envs"), false);
    assert.equal(args.includes("--initial-std"), false);
    assert.equal(arg(child, "--checkpoint"), f.artifact("drawing-pilot-01", "checkpoint", "drawing"));
    if (operation === "eval") {
      assert.deepEqual(args, [
        "--checkpoint", f.artifact("drawing-pilot-01", "checkpoint", "drawing"),
        "--eval-output", f.artifact("drawing-pilot-01", "evaluation", "drawing"),
        "--eval-episodes", "8",
        "--seed", "7",
        "--max-episode-s", "32",
      ]);
    } else if (operation === "render") {
      assert.deepEqual(args, [
        "--checkpoint", f.artifact("drawing-pilot-01", "checkpoint", "drawing"),
        "--render-output", path.dirname(f.artifact("drawing-pilot-01", "sheet", "drawing")),
        "--render-seconds", "32",
        "--seed", "7",
        "--max-episode-s", "32",
      ]);
    } else {
      assert.deepEqual(args, [
        "--checkpoint", f.artifact("drawing-pilot-01", "checkpoint", "drawing"),
        "--onnx-output", f.artifact("drawing-pilot-01", "onnx", "drawing"),
      ]);
    }
  }
});

test("Brush defaults and all commands use the bounded brush CLI contract", () => {
  const train = fixture();
  const recipe = train.api.startJob("train", {
    experimentId: "drawing",
    drawingTool: "brush",
    runName: "drawing-pilot-01",
    profile: "full",
  });
  const child = train.children[0];
  assert.equal(child.args.includes("examples/ppo_microduck_brush.py"), true);
  assert.deepEqual(Array.from(child.args.slice(child.args.indexOf("train") + 1)), [
    "--out", path.dirname(train.artifact("drawing-pilot-01", "checkpoint", "drawing")),
    "--seed", "7",
    "--steps", "49152",
    "--dagger", "2",
    "--learning-rate", "1e-7",
    "--anchor-limit", "0.00025",
  ]);
  assert.deepEqual(plain({
    drawingTool: recipe.drawingTool,
    totalTimesteps: recipe.totalTimesteps,
    numEnvs: recipe.numEnvs,
    numSteps: recipe.numSteps,
    numMinibatches: recipe.numMinibatches,
    maxEpisodeS: recipe.maxEpisodeS,
    evalEpisodes: recipe.evalEpisodes,
    renderSeconds: recipe.renderSeconds,
    initialStd: recipe.initialStd,
    learningRate: recipe.learningRate,
    clipCoefficient: recipe.clipCoefficient,
    updateEpochs: recipe.updateEpochs,
    maxGradNorm: recipe.maxGradNorm,
  }), {
    drawingTool: "brush",
    totalTimesteps: 49_152,
    numEnvs: 1,
    numSteps: 6_144,
    numMinibatches: 12,
    maxEpisodeS: 120,
    evalEpisodes: 4,
    renderSeconds: 120,
    initialStd: 0.0003,
    learningRate: 1e-7,
    clipCoefficient: 0.05,
    updateEpochs: 2,
    maxGradNorm: 0.3,
  });

  for (const operation of ["eval", "render", "export"]) {
    const f = fixture();
    f.add("drawing-pilot-01", "checkpoint", "drawing");
    f.add("drawing-pilot-01", "metadata", "drawing");
    f.add("drawing-pilot-01", "onnx", "drawing");
    f.api.startJob(operation, {
      experimentId: "drawing",
      drawingTool: "brush",
      runName: "drawing-pilot-01",
      profile: "full",
    });
    const operationChild = f.children[0];
    assert.equal(operationChild.args.includes("examples/ppo_microduck_brush.py"), true);
    const args = Array.from(operationChild.args.slice(operationChild.args.indexOf(operation) + 1));
    if (operation === "eval") {
      assert.deepEqual(args, [
        "--onnx", f.artifact("drawing-pilot-01", "onnx", "drawing"),
        "--out", f.artifact("drawing-pilot-01", "evaluation", "drawing"),
        "--seed", "7",
        "--episodes", "4",
      ]);
    } else if (operation === "render") {
      assert.deepEqual(args, [
        "--onnx", f.artifact("drawing-pilot-01", "onnx", "drawing"),
        "--out", path.dirname(f.artifact("drawing-pilot-01", "sheet", "drawing")),
        "--seed", "7",
        "--seconds", "120",
      ]);
    } else {
      assert.deepEqual(args, [
        "--checkpoint", f.artifact("drawing-pilot-01", "checkpoint", "drawing"),
        "--onnx-output", f.artifact("drawing-pilot-01", "onnx", "drawing"),
      ]);
    }
  }
});

test("Drawing eval reads the requested eval.json when the CLI does not print JSON", async () => {
  const f = fixture();
  f.add("drawing-pilot-01", "checkpoint", "drawing");
  f.add("drawing-pilot-01", "onnx", "drawing");
  const checkpoint = f.artifact("drawing-pilot-01", "checkpoint", "drawing");
  const onnx = f.artifact("drawing-pilot-01", "onnx", "drawing");
  f.files.set(f.artifact("drawing-pilot-01", "metadata", "drawing"), JSON.stringify({
    contract_version: "microduck-drawing-v1",
    checkpoint_sha256: createHash("sha256").update(f.files.get(checkpoint)).digest("hex"),
    onnx: {
      sha256: createHash("sha256").update(f.files.get(onnx)).digest("hex"),
    },
  }));
  const report = {
    ...sourceReport(f, "drawing-pilot-01", "policy", "drawing"),
    recipe: "drawing",
    contract_version: "microduck-drawing-v1",
    evaluation: {
      mode: "skill",
      seed: 7,
      eval_episodes: 8,
      environment: {
        recipe: "drawing",
        actuator: "xml",
        max_episode_s: 32,
        assistance: 0,
        recipe_options: {},
        reward_weights: {},
      },
    },
    drawing_assessment: { passed: true, unassisted: true, episodes: [] },
  };
  f.api.startJob("eval", {
    experimentId: "drawing",
    drawingTool: "pencil",
    runName: "drawing-pilot-01",
    profile: "full",
  });
  f.files.set(f.artifact("drawing-pilot-01", "evaluation", "drawing"), JSON.stringify(report));
  finish(f.children[0], null);
  await new Promise((resolve) => setImmediate(resolve));
  const state = await f.api.snapshot("drawing", "drawing-pilot-01");
  assert.equal(state.evaluation.drawing_assessment.passed, true);
  assert.equal(state.savedRecipe.evalEpisodes, 8);
  assert.equal(evaluationVerdict(state.evaluation, "drawing").taskPassed, true);
});

test("saved Drawing runs restore brush only from recipe or exact contract metadata", async () => {
  function savedBrushFixture(runName, {
    metadataContract = "microduck-brush-v2",
    reportContract = "microduck-brush-v2",
    completeAssessment = true,
    conditionCount = 10,
    sourcePackMismatch = false,
  } = {}) {
    const f = fixture();
    f.add(runName, "checkpoint", "drawing");
    f.add(runName, "onnx", "drawing");
    const checkpoint = f.artifact(runName, "checkpoint", "drawing");
    const onnx = f.artifact(runName, "onnx", "drawing");
    const sourceHashes = {
      pipeline_sha256: "1".repeat(64),
      environment_sha256: "2".repeat(64),
      reference_sha256: "3".repeat(64),
      actor_sha256: "4".repeat(64),
      base_environment_sha256: "5".repeat(64),
      base_reference_sha256: "6".repeat(64),
    };
    f.files.set(f.artifact(runName, "metadata", "drawing"), JSON.stringify({
      contract_version: metadataContract,
      source_hashes: sourcePackMismatch
        ? { ...sourceHashes, actor_sha256: "7".repeat(64) }
        : sourceHashes,
      checkpoint_sha256: createHash("sha256").update(f.files.get(checkpoint)).digest("hex"),
      onnx: {
        sha256: createHash("sha256").update(f.files.get(onnx)).digest("hex"),
      },
    }));
    const report = {
      ...sourceReport(f, runName, "policy", "drawing"),
      recipe: "drawing",
      contract_version: reportContract,
      success: true,
      unique_conditions: conditionCount,
      minimum_seeds_per_condition: 4,
      environment_source_match: true,
      provenance_errors: [],
      source_hashes: sourceHashes,
      evaluation: {
        mode: "skill",
        seed: 7,
        eval_episodes: 4,
        environment: {
          recipe: "drawing",
          actuator: "xml",
          max_episode_s: 120,
          assistance: 0,
          recipe_options: {},
          reward_weights: { coverage: 1 },
        },
      },
      drawing_assessment: {
        passed: true,
        unassisted: true,
        episodes: Array.from({ length: conditionCount }, (_, condition) =>
          Array.from({ length: 4 }, (_, seed) => ({
            case: `condition-${condition}`,
            seed,
            passed: true,
          }))
        ).flat(),
        mean_coverage: 1,
        ...(completeAssessment ? { mean_precision: 1 } : {}),
        accepted_episodes: conditionCount * 4,
        required_episodes: conditionCount * 4,
      },
      evaluation_request: {
        recipe: {
          experimentId: "drawing",
          runName,
          profile: "full",
        },
      },
    };
    delete report.passed;
    f.files.set(f.artifact(runName, "evaluation", "drawing"), JSON.stringify(report));
    return f;
  }

  const brush = savedBrushFixture("pencil-looking-name");
  const restored = await brush.api.snapshot("drawing", "pencil-looking-name");
  assert.equal(restored.drawingTool, "brush");
  assert.equal(restored.savedRecipe.drawingTool, "brush");
  assert.equal(restored.savedRecipe.evalEpisodes, 4);
  assert.equal(restored.evaluation.passed, true);
  assert.equal(evaluationVerdict(restored.evaluation, "drawing").taskPassed, true);

  brush.files.delete(brush.artifact("pencil-looking-name", "evaluation", "drawing"));
  const metadataOnly = await brush.api.snapshot("drawing", "pencil-looking-name");
  assert.equal(metadataOnly.evaluation, null);
  assert.equal(metadataOnly.savedRecipe, null);
  assert.equal(metadataOnly.drawingTool, "brush");

  const mismatched = savedBrushFixture("contract-mismatch", {
    metadataContract: "microduck-drawing-v1",
  });
  const rejectedMismatch = await mismatched.api.snapshot("drawing", "contract-mismatch");
  assert.equal(rejectedMismatch.evaluation, null);
  assert.equal(rejectedMismatch.savedRecipe, null);
  assert.equal(rejectedMismatch.drawingTool, "pencil");

  const wrongShape = savedBrushFixture("wrong-shape", {
    completeAssessment: false,
  });
  const rejectedShape = await wrongShape.api.snapshot("drawing", "wrong-shape");
  assert.equal(rejectedShape.evaluation, null);
  assert.equal(rejectedShape.savedRecipe, null);
  assert.equal(rejectedShape.drawingTool, "brush");

  const preliminary = savedBrushFixture("brush-color-v2-01", {
    conditionCount: 5,
  });
  const rejectedPreliminary = await preliminary.api.snapshot("drawing", "brush-color-v2-01");
  assert.equal(rejectedPreliminary.evaluation, null);
  assert.equal(rejectedPreliminary.savedRecipe, null);
  assert.equal(rejectedPreliminary.drawingTool, "brush");

  const changedSourcePack = savedBrushFixture("source-pack-mismatch", {
    sourcePackMismatch: true,
  });
  const rejectedSourcePack = await changedSourcePack.api.snapshot("drawing", "source-pack-mismatch");
  assert.equal(rejectedSourcePack.evaluation, null);
  assert.equal(rejectedSourcePack.savedRecipe, null);
});

test("Basketball fails before spawn when local reference inputs are incomplete", () => {
  const fresh = fixture();
  assert.throws(
    () => fresh.api.startJob("train", {
      experimentId: "basketball",
      runName: "basketball-fresh",
    }),
    /Basketball local reference setup is incomplete.*checkpoint\.pt.*policy\.onnx.*does not download reference assets/
  );
  assert.equal(fresh.children.length, 0);

  const resumed = fixture();
  resumed.addBasketballReferences();
  resumed.add("basketball-resume", "checkpoint", "basketball");
  assert.throws(
    () => resumed.api.startJob("train", {
      experimentId: "basketball",
      runName: "basketball-resume",
      resumeFromCheckpoint: true,
    }),
    /basketball-resume[/\\]policy\.onnx/
  );
  assert.equal(resumed.children.length, 0);

  const evaluate = fixture();
  evaluate.addBasketballReferences();
  evaluate.add("basketball-eval", "checkpoint", "basketball");
  assert.throws(
    () => evaluate.api.startJob("eval", {
      experimentId: "basketball",
      runName: "basketball-eval",
    }),
    /basketball-eval[/\\]policy\.onnx/
  );
  assert.equal(evaluate.children.length, 0);
});

test("Bridge Full enables curriculum only for training and enforces unassisted evaluation horizon", () => {
  const train = fixture();
  const recipe = train.api.startJob("train", {
    experimentId: "bridge",
    runName: "bridge-studio-01",
    profile: "full",
  });
  assert.equal(recipe.bridgeCurriculum, true);
  assert.equal(recipe.totalTimesteps, 32768);
  assert.equal(recipe.numEnvs, 4);
  assert.equal(recipe.numSteps, 128);
  assert.equal(recipe.numMinibatches, 4);
  assert.equal(recipe.maxEpisodeS, 20);
  assert.equal(recipe.evalSteps, 1000);
  assert.equal(recipe.seed, 7);
  assert.equal(recipe.initialStd, 0.03);
  assert.equal(recipe.learningRate, 0.00001);
  assert.equal(recipe.gamma, 0.99);
  assert.equal(recipe.clipCoefficient, 0.1);
  assert.equal(recipe.updateEpochs, 2);
  assert.equal(recipe.entropyCoefficient, 0);
  assert.equal(recipe.maxGradNorm, 0.5);
  assert.equal(recipe.freezeObservationNormalization, true);
  assert.equal(recipe.domainRand, false);
  assert.equal(recipe.obsNoise, false);
  assert.equal(recipe.actionDelay, false);
  assert.equal(recipe.randomYaw, false);
  assert.equal(train.children[0].args.includes("--bridge-curriculum"), true);
  assert.equal(train.children[0].args.includes("--freeze-observation-normalization"), true);
  assert.equal(train.children[0].args.includes("--init-from"), false);
  assert.deepEqual(JSON.parse(arg(train.children[0], "--weight-overrides")), {});

  for (const operation of ["eval", "render"]) {
    const f = fixture();
    f.add("bridge-studio-01", "onnx", "bridge");
    f.api.startJob(operation, {
      experimentId: "bridge",
      runName: "bridge-studio-01",
      profile: "full",
    });
    const child = f.children[0];
    assert.equal(child.args.includes("examples/ppo_microduck_studio.py"), true);
    assert.equal(arg(child, "--recipe"), "bridge");
    assert.equal(child.args.includes("--bridge-curriculum"), false);
    assert.equal(arg(child, "--max-episode-s"), "20");
    if (operation === "eval") assert.equal(arg(child, "--eval-steps"), "1000");
  }

  assert.throws(() => fixture().api.normalizeRecipe({
    experimentId: "bridge",
    profile: "full",
    maxEpisodeS: 19,
    evalSteps: 1000,
  }), /requires 20 episode seconds/);
  assert.throws(() => fixture().api.normalizeRecipe({
    experimentId: "bridge",
    profile: "full",
    maxEpisodeS: 20,
    evalSteps: 1500,
  }), /whole number of complete episode horizons/);
  assert.equal(fixture().api.normalizeRecipe({
    experimentId: "bridge",
    profile: "full",
    maxEpisodeS: 20,
    evalSteps: 2000,
  }).evalSteps, 2000);
  assert.throws(() => fixture().api.normalizeRecipe({
    experimentId: "bridge",
    profile: "full",
    maxEpisodeS: 20,
    evalSteps: 999,
  }), /at least 1,000 evaluation steps/);
});

for (const profile of ["smoke", "full"]) {
  test(`${profile} Swing evaluation declares its scope, horizon, and explicit default target`, () => {
    const f = fixture();
    f.add("test-a");
    const recipe = f.api.startJob("eval", input("test-a", { profile }));
    const child = f.children[0];
    assert.equal(recipe.swingMinSpanDeg, 150);
    assert.equal(arg(child, "--evaluation-mode"), profile === "smoke" ? "pipeline" : "skill");
    assert.equal(arg(child, "--eval-steps"), profile === "smoke" ? "4" : "1200");
    assert.equal(arg(child, "--max-episode-s"), profile === "smoke" ? "1" : "24");
    assert.equal(arg(child, "--swing-min-span-deg"), "150");
    assert.equal(arg(child, "--checkpoint"), f.artifact("test-a"));
  });
}

test("evaluation selects ONNX when present and forwards an edited target", () => {
  const f = fixture();
  f.add("test-a", "onnx");
  f.api.startJob("eval", input("test-a", { swingMinSpanDeg: 160 }));
  assert.equal(arg(f.children[0], "--policy"), f.artifact("test-a", "onnx"));
  assert.equal(arg(f.children[0], "--swing-min-span-deg"), "160");
  assert.equal(f.api.normalizeRecipe(input("a", { swingMinSpanDeg: Infinity })).swingMinSpanDeg, 150);
});

test("non-Swing evaluation declares scope without Swing-specific criteria", () => {
  const f = fixture();
  f.add("test-a", "checkpoint", "dance");
  f.api.startJob("eval", input("test-a", { experimentId: "dance", profile: "full" }));
  assert.equal(arg(f.children[0], "--evaluation-mode"), "skill");
  assert.equal(f.children[0].args.includes("--swing-min-span-deg"), false);
});

test("Dance API flow preserves its PPO, evaluation, render, and export contract", async () => {
  const f = fixture();
  const danceClip = f.addClip("derived/dance.json");
  const recipeInput = input("dance-e2e", {
    experimentId: "dance",
    profile: "full",
    totalTimesteps: 1_000,
    numEnvs: 8,
    numSteps: 32,
    numMinibatches: 8,
    maxEpisodeS: 12,
    evalSteps: 900,
    renderSeconds: 150,
    initialStd: 0.3,
    normalizeRewards: true,
    checkpointInterval: 512,
    danceClip,
    dancePoseSigma: 0.2,
  });

  const recipe = f.api.startJob("train", recipeInput);
  const train = f.children[0];
  assert.equal(recipe.totalTimesteps, 1_024);
  assert.equal(recipe.danceClip, danceClip);
  assert.equal(arg(train, "--recipe"), "dance");
  assert.equal(arg(train, "--num-steps"), "32");
  assert.equal(arg(train, "--num-minibatches"), "8");
  assert.equal(arg(train, "--num-envs"), "8");
  assert.equal(arg(train, "--max-episode-s"), "12");
  assert.equal(arg(train, "--dance-clip"), danceClip);
  assert.equal(arg(train, "--dance-pose-sigma"), "0.2");
  assert.equal(arg(train, "--initial-std"), "0.3");
  assert.equal(arg(train, "--checkpoint-interval"), "512");
  assert.equal(train.args.includes("--normalize-rewards"), true);
  assert.equal(train.args.includes("--no-normalize-rewards"), false);
  assert.equal((await f.api.snapshot()).normalizeRewards, true);
  assert.equal(arg(train, "--checkpoint"), f.artifact("dance-e2e", "checkpoint", "dance"));
  assert.equal(arg(train, "--onnx-output"), f.artifact("dance-e2e", "onnx", "dance"));
  assert.equal(train.args.includes("--swing-initial-angle-deg"), false);
  finish(train);

  f.add("dance-e2e", "checkpoint", "dance");
  f.add("dance-e2e", "metadata", "dance");
  f.api.startJob("eval", recipeInput);
  const evaluate = f.children[1];
  assert.equal(arg(evaluate, "--evaluation-mode"), "skill");
  assert.equal(arg(evaluate, "--eval-steps"), "900");
  assert.equal(arg(evaluate, "--max-episode-s"), "12");
  assert.equal(arg(evaluate, "--dance-clip"), danceClip);
  assert.equal(arg(evaluate, "--dance-pose-sigma"), "0.2");
  assert.equal(evaluate.args.includes("--initial-std"), false);
  assert.equal(evaluate.args.includes("--normalize-rewards"), false);
  assert.equal(evaluate.args.includes("--checkpoint-interval"), false);
  assert.equal(arg(evaluate, "--checkpoint"), f.artifact("dance-e2e", "checkpoint", "dance"));
  assert.equal(evaluate.args.includes("--swing-min-span-deg"), false);
  finish(evaluate, {
    ...sourceReport(f, "dance-e2e", "checkpoint", "dance"),
    skill_status: "not_assessed",
  });
  const evaluated = await f.api.snapshot();
  assert.equal(evaluated.evaluation.evaluation_request.evaluation_mode, "skill");
  assert.equal(evaluated.evaluation.evaluation_request.eval_steps, 900);
  assert.equal(evaluated.evaluation.evaluation_request.recipe.danceClip, danceClip);
  assert.equal(evaluated.savedRecipe.danceClip, danceClip);
  assert.equal(evaluated.savedRecipe.totalTimesteps, 1_024);

  f.api.startJob("render", recipeInput);
  const render = f.children[2];
  assert.equal(arg(render, "--render-seconds"), "150");
  assert.equal(arg(render, "--max-episode-s"), "12");
  assert.equal(arg(render, "--dance-clip"), danceClip);
  assert.equal(arg(render, "--dance-pose-sigma"), "0.2");
  assert.equal(render.args.includes("--initial-std"), false);
  assert.equal(render.args.includes("--normalize-rewards"), false);
  assert.equal(render.args.includes("--checkpoint-interval"), false);
  assert.equal(arg(render, "--episodes"), "1");
  assert.equal(arg(render, "--output"), path.dirname(f.artifact("dance-e2e", "sheet", "dance")));
  finish(render, { rendered: true });

  f.api.startJob("export", recipeInput);
  const exportJob = f.children[3];
  assert.deepEqual(Array.from(exportJob.args.slice(exportJob.args.indexOf("export") + 1)), [
    "--recipe", "dance",
    "--checkpoint", f.artifact("dance-e2e", "checkpoint", "dance"),
    "--output", f.artifact("dance-e2e", "onnx", "dance"),
  ]);
  assert.equal(exportJob.args.includes("--dance-clip"), false);
  assert.equal(exportJob.args.includes("--dance-pose-sigma"), false);
  assert.equal(exportJob.args.includes("--initial-std"), false);
  assert.equal(exportJob.args.includes("--normalize-rewards"), false);
  assert.equal(exportJob.args.includes("--checkpoint-interval"), false);
});

test("policy exploration and reward normalization defaults remain recipe-specific", () => {
  const f = fixture();
  const dance = f.api.normalizeRecipe({ experimentId: "dance" });
  const swing = f.api.normalizeRecipe({ experimentId: "swing" });
  const running = f.api.normalizeRecipe({ experimentId: "running" });
  const backflip = f.api.normalizeRecipe({ experimentId: "backflip", profile: "full" });
  assert.equal(dance.initialStd, Math.exp(-0.5));
  assert.equal(dance.normalizeRewards, false);
  assert.equal(swing.initialStd, 0.1);
  assert.equal(swing.normalizeRewards, true);
  assert.equal(running.initialStd, Math.exp(-0.5));
  assert.equal(running.normalizeRewards, false);
  assert.equal(backflip.initialStd, 0.03);
  assert.equal(backflip.normalizeRewards, true);
  assert.equal(backflip.freezeObservationNormalization, true);
  assert.equal(backflip.seed, 7);
  assert.equal(backflip.totalTimesteps, 401_408);
  assert.equal(backflip.numEnvs, 16);
  assert.equal(backflip.numSteps, 128);
  assert.equal(backflip.numMinibatches, 4);
  assert.equal(backflip.updateEpochs, 2);
  assert.equal(backflip.learningRate, 0.000003);
  assert.equal(backflip.entropyCoefficient, 0);
  assert.equal(backflip.maxEpisodeS, 12);
  assert.equal(backflip.evalSteps, 600);
  assert.equal(backflip.renderSeconds, 12);
  assert.equal(backflip.domainRand, false);
  assert.equal(backflip.obsNoise, false);
  assert.equal(backflip.actionDelay, false);
  assert.equal(backflip.randomYaw, false);
  assert.deepEqual(plain(backflip.rewardWeights), {
    landing_upright: 3,
    landing_pose: 2,
    landing_settle: 1,
    landing_joint_speed_penalty: 0.05,
    landing_action_rate_penalty: 0.02,
    landing_action_size_penalty: 0.01,
  });
  assert.equal(dance.checkpointInterval, 100_000);
  assert.equal(dance.dancePoseSigma, null);
});

test("Backflip smoke stays four steps while Full emits the fixed PPO preset", () => {
  const smoke = fixture();
  const smokeRecipe = smoke.api.startJob("train", {
    experimentId: "backflip",
    runName: "backflip-smoke",
  });
  assert.equal(smokeRecipe.totalTimesteps, 4);
  assert.equal(smokeRecipe.freezeObservationNormalization, false);
  assert.equal(arg(smoke.children[0], "--total-timesteps"), "4");
  assert.equal(arg(smoke.children[0], "--backend"), "dummy");
  assert.equal(arg(smoke.children[0], "--update-epochs"), "1");
  assert.equal(smoke.children[0].args.includes("--init-from"), false);

  const full = fixture();
  const fullRecipe = full.api.startJob("train", {
    experimentId: "backflip",
    runName: "backflip-full",
    profile: "full",
  });
  const child = full.children[0];
  assert.equal(fullRecipe.totalTimesteps, 401_408);
  assert.equal(arg(child, "--recipe"), "backflip");
  assert.equal(arg(child, "--backend"), "dummy");
  assert.equal(arg(child, "--total-timesteps"), "401408");
  assert.equal(arg(child, "--num-envs"), "16");
  assert.equal(arg(child, "--num-steps"), "128");
  assert.equal(arg(child, "--num-minibatches"), "4");
  assert.equal(arg(child, "--update-epochs"), "2");
  assert.equal(arg(child, "--learning-rate"), "0.000003");
  assert.equal(arg(child, "--initial-std"), "0.03");
  assert.equal(child.args.includes("--freeze-observation-normalization"), true);
  assert.equal(arg(child, "--entropy-coefficient"), "0");
  assert.equal(arg(child, "--seed"), "7");
  assert.equal(child.args.includes("--normalize-rewards"), true);
  assert.equal(child.args.includes("--no-domain-rand"), true);
  assert.equal(child.args.includes("--no-obs-noise"), true);
  assert.equal(child.args.includes("--no-action-delay"), true);
  assert.equal(child.args.includes("--no-random-yaw"), true);
  assert.deepEqual(JSON.parse(arg(child, "--weight-overrides")), {
    landing_upright: 3,
    landing_pose: 2,
    landing_settle: 1,
    landing_joint_speed_penalty: 0.05,
    landing_action_rate_penalty: 0.02,
    landing_action_size_penalty: 0.01,
  });

  const dance = fixture();
  dance.api.startJob("train", {
    experimentId: "dance",
    runName: "dance-fork-default",
  });
  assert.equal(dance.children[0].args.includes("--backend"), false);
});

test("Backflip evaluation requires current stand-policy provenance and explicit skill scope", async () => {
  const f = fixture();
  f.add("backflip-eval", "checkpoint", "backflip");
  f.add("backflip-eval", "metadata", "backflip");
  f.addStandPolicy();
  const recipe = {
    experimentId: "backflip",
    runName: "backflip-eval",
    profile: "full",
  };
  f.api.startJob("eval", recipe);
  const standHash = createHash("sha256").update(f.files.get(f.standPolicy)).digest("hex");
  finish(f.children[0], {
    ...sourceReport(f, "backflip-eval", "checkpoint", "backflip"),
    recipe: "backflip",
    evaluation: {
      environment: {
        recipe_options: {
          backflip_protocol_version: "spotter-launch-landing-stand-v2",
          stand_policy_sha256: standHash,
        },
      },
    },
    backflip_assessment: {
      passed: true,
      episodes: [{ passed: true, rotation_rad: 6.4, stand_hold_seconds: 2.1 }],
    },
  });
  const accepted = await f.api.snapshot();
  assert.equal(evaluationVerdict(accepted.evaluation, "backflip").taskPassed, true);
  assert.equal(accepted.savedRecipe.experimentId, "backflip");

  f.api.startJob("eval", recipe);
  finish(f.children[1], {
    ...sourceReport(f, "backflip-eval", "checkpoint", "backflip"),
    recipe: "backflip",
    evaluation: {
      environment: {
        recipe_options: {
          backflip_protocol_version: "spotter-launch-landing-stand-v1",
          stand_policy_sha256: standHash,
        },
      },
    },
    backflip_assessment: {
      passed: true,
      episodes: [{ passed: true, rotation_rad: 6.4, stand_hold_seconds: 2.1 }],
    },
  });
  const obsolete = await f.api.snapshot();
  assert.equal(obsolete.evaluation.skill_status, "not_assessed");
  assert.equal(obsolete.evaluation.evaluation_settings_match, false);

  f.addStandPolicy("new stand policy bytes");
  const invalidated = await f.api.snapshot();
  assert.equal(invalidated.evaluation.skill_status, "not_assessed");
  assert.equal(invalidated.evaluation.evaluation_settings_match, false);
  assert.equal(evaluationVerdict(invalidated.evaluation, "backflip").taskPassed, false);
});

test("Backflip render receipts require and record the exact Python protocol provenance", () => {
  const accepted = fixture();
  accepted.add("backflip-render", "onnx", "backflip");
  accepted.addStandPolicy();
  accepted.api.startJob("render", {
    experimentId: "backflip",
    runName: "backflip-render",
    profile: "full",
  });
  finish(accepted.children[0], {
    rendered: true,
    evaluation: {
      environment: {
        recipe_options: {
          backflip_protocol_version: "spotter-launch-landing-stand-v2",
          stand_policy_sha256: "current-stand-hash",
        },
      },
    },
  });
  assert.equal(accepted.renderFinalizations.length, 1);
  assert.deepEqual(plain(accepted.renderFinalizations[0].recipeOptions), {
    backflip_protocol_version: "spotter-launch-landing-stand-v2",
    stand_policy_sha256: "current-stand-hash",
  });

  const obsolete = fixture();
  obsolete.add("backflip-render-v0", "onnx", "backflip");
  obsolete.addStandPolicy();
  obsolete.api.startJob("render", {
    experimentId: "backflip",
    runName: "backflip-render-v0",
    profile: "full",
  });
  finish(obsolete.children[0], {
    rendered: true,
    evaluation: {
      environment: {
        recipe_options: {
          backflip_protocol_version: "spotter-launch-landing-stand-v1",
          stand_policy_sha256: "current-stand-hash",
        },
      },
    },
  });
  assert.equal(obsolete.renderFinalizations.length, 0);
});

test("Backflip render completion reads protocol provenance from renderer output", () => {
  const f = fixture();
  f.add("backflip-render-output", "onnx", "backflip");
  f.api.startJob("render", {
    experimentId: "backflip",
    runName: "backflip-render-output",
    profile: "full",
  });
  finish(f.children[0], {
    rendered: true,
    environment: {
      recipe_options: {
        backflip_protocol_version: "spotter-launch-landing-stand-v2",
        stand_policy_sha256: "stand-hash",
      },
    },
  });
  assert.equal(f.renderFinalizations.length, 1);
  assert.deepEqual(plain(f.renderFinalizations[0].recipeOptions), {
    backflip_protocol_version: "spotter-launch-landing-stand-v2",
    stand_policy_sha256: "stand-hash",
  });
});

test("Backflip render verification matches the evaluated protocol and stand-policy hash", async () => {
  const f = fixture();
  f.add("backflip-bound-render", "onnx", "backflip");
  f.addStandPolicy();
  const standHash = createHash("sha256")
    .update(f.files.get(f.standPolicy))
    .digest("hex");
  const recipe = {
    experimentId: "backflip",
    runName: "backflip-bound-render",
    profile: "full",
  };

  f.api.startJob("eval", recipe);
  finish(f.children[0], {
    ...sourceReport(f, recipe.runName, "policy", "backflip"),
    recipe: "backflip",
    evaluation: {
      environment: {
        recipe_options: {
          backflip_protocol_version: "spotter-launch-landing-stand-v2",
          stand_policy_sha256: standHash,
        },
      },
    },
    backflip_assessment: {
      passed: true,
      episodes: [{ passed: true, rotation_rad: 6.4, stand_hold_seconds: 2.1 }],
    },
  });

  f.api.startJob("render", recipe);
  finish(f.children[1], {
    rendered: true,
    evaluation: {
      environment: {
        recipe_options: {
          backflip_protocol_version: "spotter-launch-landing-stand-v2",
          stand_policy_sha256: standHash,
        },
      },
    },
  });
  f.setReadRenderEvidence((_receiptPath, expected) => ({
    ...expected,
    clipSha256: null,
    externalFilesSha256: { [f.standPolicy]: standHash },
    video: { sha256: "bound-video" },
  }));

  const snapshot = await f.api.snapshot();
  assert.equal(snapshot.renderVerified, true);
  assert.deepEqual(plain(f.renderEvidenceReads.at(-1).expected.recipeOptions), {
    backflip_protocol_version: "spotter-launch-landing-stand-v2",
    stand_policy_sha256: standHash,
  });
});

test("Dance clip paths are explicit, existing, and confined to owned roots", () => {
  const f = fixture();
  const workspaceClip = f.addClip("authored.json");
  const artifactClip = f.addClip("derived.json", "rlx-artifacts");
  assert.equal(
    f.api.normalizeRecipe({ experimentId: "dance", danceClip: workspaceClip }).danceClip,
    workspaceClip
  );
  assert.equal(
    f.api.normalizeRecipe({ experimentId: "dance", danceClip: "artifacts/derived.json" }).danceClip,
    artifactClip
  );
  assert.throws(
    () => f.api.normalizeRecipe({ experimentId: "dance", danceClip: 42 }),
    /file path string/
  );
  assert.throws(
    () => f.api.normalizeRecipe({ experimentId: "dance", danceClip: "" }),
    /cannot be empty/
  );
  assert.throws(
    () => f.api.normalizeRecipe({ experimentId: "dance", danceClip: "/tmp/outside.json" }),
    /must be under/
  );
  assert.throws(
    () => f.api.normalizeRecipe({ experimentId: "dance", danceClip: "dance-clip/missing.json" }),
    /does not exist/
  );
  assert.throws(
    () => f.api.normalizeRecipe({ experimentId: "running", danceClip: workspaceClip }),
    /only be used with the dance experiment/
  );
});

test("Dance pose sigma is optional, positive, finite, and Dance-only", () => {
  const f = fixture();
  assert.equal(
    f.api.normalizeRecipe({
      experimentId: "dance",
      dancePoseSigma: 0.2,
    }).dancePoseSigma,
    0.2
  );
  for (const dancePoseSigma of [0, -0.1, Number.NaN, Number.POSITIVE_INFINITY, "0.2"]) {
    assert.throws(
      () => f.api.normalizeRecipe({ experimentId: "dance", dancePoseSigma }),
      /finite number greater than 0/
    );
  }
  assert.throws(
    () => f.api.normalizeRecipe({ experimentId: "swing", dancePoseSigma: 0.2 }),
    /only be used with the dance experiment/
  );
});

test("new rollout overrides reject invalid or incompatible values", () => {
  const f = fixture();
  assert.throws(
    () => f.api.normalizeRecipe(input("a", { numSteps: "32" })),
    /Rollout steps must be an integer/
  );
  assert.throws(
    () => f.api.normalizeRecipe(input("a", { numEnvs: 3, numSteps: 5, numMinibatches: 4 })),
    /Minibatches must divide/
  );
  assert.throws(
    () => f.api.normalizeRecipe(input("a", { maxEpisodeS: Number.NaN })),
    /Maximum episode seconds/
  );
  assert.throws(
    () => f.api.normalizeRecipe(input("a", { evalSteps: 1.5 })),
    /Evaluation steps must be an integer/
  );
  assert.throws(
    () => f.api.normalizeRecipe(input("a", { renderSeconds: 0 })),
    /Render seconds/
  );
  assert.throws(
    () => f.api.normalizeRecipe(input("a", { initialStd: 0 })),
    /Initial policy standard deviation/
  );
  assert.throws(
    () => f.api.normalizeRecipe(input("a", { normalizeRewards: "yes" })),
    /Reward normalization must be a boolean/
  );
  assert.throws(
    () => f.api.normalizeRecipe(input("a", { checkpointInterval: -1 })),
    /Checkpoint interval/
  );
});

test("render accepts ONNX-only runs while artifact-free runs cannot evaluate or render", () => {
  const f = fixture();
  assert.throws(() => f.api.startJob("eval", input()), /checkpoint before continuing/);
  assert.throws(() => f.api.startJob("render", input()), /checkpoint before continuing/);
  assert.equal(f.children.length, 0);
  f.add("test-a", "onnx");
  f.api.startJob("render", input());
  assert.equal(arg(f.children[0], "--policy"), f.artifact("test-a", "onnx"));
  assert.equal(f.children[0].args.includes("--evaluation-mode"), false);
});

test("selecting another run loads only that run's persisted evaluation", async () => {
  const f = fixture();
  f.add("test-a");
  f.add("test-b");
  const savedPath = path.join(path.dirname(f.artifact("test-b")), "evaluation.json");
  f.files.set(savedPath, JSON.stringify({ marker: "selected-run", passed: false }));
  f.api.startJob("eval", input());
  finish(f.children[0], { marker: "previous-run", passed: true });
  f.api.startJob("render", input("test-b"));
  assert.equal((await f.api.snapshot()).evaluation.marker, "selected-run");
});

test("training keeps its PPO arguments and never receives evaluation-only flags", () => {
  const f = fixture();
  f.api.startJob("train", input());
  const child = f.children[0];
  assert.equal(arg(child, "--num-steps"), "2");
  assert.equal(arg(child, "--onnx-output"), f.artifact("test-a", "onnx"));
  assert.equal(arg(child, "--initial-std"), "0.1");
  assert.equal(arg(child, "--checkpoint-interval"), "100000");
  assert.equal(child.args.includes("--normalize-rewards"), true);
  assert.equal(child.args.includes("--evaluation-mode"), false);
  assert.equal(child.args.includes("--swing-min-span-deg"), false);
  assert.throws(() => f.api.startJob("train", input("test-b")), /already running/);
});

test("fresh training refuses to overwrite an existing checkpoint before touching history", () => {
  const f = fixture();
  f.add("saved-run");
  const metrics = path.join(
    path.dirname(f.artifact("saved-run")),
    "training-metrics.jsonl"
  );
  f.files.set(metrics, "existing durable telemetry\n");
  assert.throws(
    () => f.api.startJob("train", input("saved-run")),
    /Choose a new run name or enable resume/
  );
  assert.equal(f.children.length, 0);
  assert.equal(f.files.get(metrics), "existing durable telemetry\n");
});

test("live reward history retains the full selected invocation beyond 240 samples", async () => {
  const f = fixture();
  f.api.startJob("train", input("long-run", { totalTimesteps: 600 }));
  const child = f.children[0];
  for (let step = 1; step <= 300; step += 1) {
    child.stdout.emit(
      "data",
      `${JSON.stringify({
        event: "training_progress",
        steps: step,
        total: 600,
        mean_reward: step / 10,
      })}\n`
    );
  }
  const snapshot = await f.api.snapshot();
  assert.equal(snapshot.rewardHistory.length, 300);
  assert.deepEqual(plain(snapshot.rewardHistory.at(-1)), { step: 300, reward: 30 });
});

test("cold snapshots restore the latest invocation graph and counters from disk", async () => {
  const f = fixture();
  const metadata = f.artifact("saved-run", "metadata");
  const runDirectory = path.dirname(metadata);
  f.files.set(metadata, JSON.stringify({
    metadata: {
      steps: 40,
      ppo: {
        total_timesteps: 40,
        normalize_rewards: true,
      },
    },
  }));
  f.files.set(path.join(runDirectory, "training-metrics.jsonl"), [
    JSON.stringify({
      phase: "collection",
      steps: 20,
      env_steps: 20,
      seconds: 0.1,
      mean_reward: 1.25,
    }),
    JSON.stringify({
      phase: "collection",
      steps: 20,
      env_steps: 40,
      seconds: 0.1,
      mean_reward: 2.5,
    }),
  ].join("\n"));
  const snapshot = await f.api.snapshot("swing", "saved-run");
  assert.deepEqual(plain(snapshot.rewardHistory), [
    { step: 20, reward: 1.25 },
    { step: 40, reward: 2.5 },
  ]);
  assert.equal(snapshot.trainingSteps, 40);
  assert.equal(snapshot.trainingTotal, 40);
  assert.equal(snapshot.normalizeRewards, true);
});

for (const operation of ["eval", "render"]) {
  for (const otherExperiment of [false, true]) {
    test(`${operation} clears old-run telemetry and evaluation on ${otherExperiment ? "experiment" : "name"} change`, async () => {
      const f = fixture();
      f.api.startJob("train", input());
      progress(f.children[0]);
      finish(f.children[0]);
      f.add("test-a");
      f.api.startJob("eval", input());
      finish(f.children[1], { passed: true, source_sha256: "old-source" });
      const next = otherExperiment ? input("test-a", { experimentId: "dance" }) : input("test-b");
      f.add(next.runName, "checkpoint", next.experimentId);
      f.api.startJob(operation, next);
      const snapshot = await f.api.snapshot();
      assert.equal(snapshot.trainingSteps, 0);
      assert.equal(snapshot.trainingTotal, 0);
      assert.equal(snapshot.rewardHistory.length, 0);
      assert.equal(snapshot.evaluation, null);
      assert.equal(snapshot.result, null);
    });
  }
}

test("same-run eval/render preserve training stats and persist settings with authoritative outcomes", async () => {
  const f = fixture();
  f.api.startJob("train", input());
  progress(f.children[0]);
  finish(f.children[0]);
  f.add("test-a");
  f.add("test-a", "onnx");
  const report = {
    ...sourceReport(f),
    evaluation_mode: "skill", passed: false, pipeline_passed: true, skill_status: "failed",
    finite: true,
    evaluation: { mode: "skill", steps_per_env: 1200, swing_criteria: { min_bidirectional_span_deg: 160 } },
    swing_assessment: { passed: false, failures: ["incomplete"] },
  };
  f.api.startJob("eval", input("test-a", { profile: "full", swingMinSpanDeg: 160 }));
  finish(f.children[1], report);
  const evaluated = await f.api.snapshot();
  assert.equal(evaluated.evaluation.passed, false);
  assert.equal(evaluated.evaluation.skill_status, "failed");
  assert.equal(evaluated.evaluation.source_sha256, report.source_sha256);
  assert.equal(evaluated.evaluation.evaluation.steps_per_env, 1200);
  assert.equal(evaluated.evaluation.evaluation_request.swing_min_span_deg, 160);
  assert.equal(evaluated.evaluation.evaluation_request.evaluation_mode, "skill");
  assert.equal(evaluated.evaluation.evaluation_request.recipe.profile, "full");
  const saved = JSON.parse(f.files.get(path.join(path.dirname(f.artifact("test-a")), "evaluation.json")));
  assert.equal(saved.source_sha256, report.source_sha256);
  assert.equal(saved.evaluation_request.eval_steps, 1200);
  f.api.startJob("render", input());
  finish(f.children[2], { rendered: true });
  const rendered = await f.api.snapshot();
  assert.equal(rendered.trainingSteps, 4);
  assert.equal(rendered.trainingTotal, 4);
  assert.equal(rendered.rewardHistory[0].reward, 2.5);
  assert.equal(rendered.evaluation.source_sha256, report.source_sha256);
  f.api.startJob("train", input("test-a", { resumeFromCheckpoint: true }));
  const restarted = await f.api.snapshot();
  assert.equal(restarted.trainingSteps, 0);
  assert.equal(restarted.rewardHistory.length, 0);
  assert.equal(restarted.evaluation, null);
});

for (const newRunName of ["test-a", "test-b"]) {
  test(`cancelled callbacks cannot mutate a replacement job (${newRunName})`, async () => {
    const f = fixture();
    f.api.startJob("train", input());
    const old = f.children[0];
    old.stdout.emit("data", "old partial");
    assert.equal(f.api.cancelJob(), true);
    assert.equal(old.killed, "SIGTERM");
    f.api.startJob("train", input(newRunName));
    const current = f.children[1];
    const before = JSON.stringify(await f.api.snapshot());
    progress(old);
    old.stderr.emit("data", "late stderr\n");
    old.emit("error", new Error("late error"));
    finish(old, { passed: true }, 0);
    assert.equal(JSON.stringify(await f.api.snapshot()), before);
    progress(current);
    assert.equal((await f.api.snapshot()).trainingSteps, 4);
    assert.equal(f.api.cancelJob(), true);
    assert.equal(current.killed, "SIGTERM");
  });
}

test("cancelled evaluation cannot publish its buffered report", async () => {
  const f = fixture();
  f.add("test-a");
  f.api.startJob("eval", input());
  f.children[0].stdout.emit("data", '{"passed":true}');
  f.api.cancelJob();
  finish(f.children[0]);
  const snapshot = await f.api.snapshot();
  assert.equal(snapshot.phase, "cancelled");
  assert.equal(snapshot.evaluation, null);
  assert.equal(snapshot.artifacts.evaluation, false);
});

test("a previous evaluation persistence error cannot append to a new job", async () => {
  const f = fixture();
  let rejectWrite;
  f.deferWrites(() => new Promise((_, reject) => { rejectWrite = reject; }));
  f.add("test-a");
  f.api.startJob("eval", input());
  finish(f.children[0], { passed: false });
  f.api.startJob("train", input("test-b"));
  rejectWrite(new Error("old write failed"));
  await new Promise(setImmediate);
  assert.equal((await f.api.snapshot()).logs.some((line) => line.includes("old write failed")), false);
});

for (const sourceType of ["policy", "checkpoint"]) {
  test(`${sourceType} evaluation remains bound only to the evaluated current bytes`, async () => {
    const f = fixture();
    f.add("test-a");
    f.add("test-a", "metadata");
    if (sourceType === "policy") f.add("test-a", "onnx");
    const report = sourceReport(f, "test-a", sourceType);
    f.api.startJob("eval", input("test-a", { profile: "full" }));
    finish(f.children[0], report);
    assert.equal((await f.api.snapshot()).evaluation.passed, true);
    const source = f.artifact("test-a", sourceType === "policy" ? "onnx" : "checkpoint");
    f.files.set(source, "changed policy bytes");
    assert.equal((await f.api.snapshot()).evaluation, null);
    f.files.set(source, "artifact");
    assert.equal((await f.api.snapshot()).evaluation.passed, true);
    f.files.delete(source);
    assert.equal((await f.api.snapshot()).evaluation, null);
  });
}

for (const changedKind of ["checkpoint", "metadata"]) {
  test(`same-run retrain/render cannot revive saved evidence after ${changedKind} changes`, async () => {
    const f = fixture();
    f.add("test-a");
    f.add("test-a", "metadata");
    f.api.startJob("eval", input("test-a", { profile: "full" }));
    finish(f.children[0], sourceReport(f, "test-a", "checkpoint"));
    assert.equal((await f.api.snapshot()).evaluation.passed, true);
    f.api.startJob("train", input("test-a", { resumeFromCheckpoint: true }));
    f.files.set(f.artifact("test-a", changedKind), "retrained bytes");
    finish(f.children[1]);
    f.api.startJob("render", input());
    assert.equal((await f.api.snapshot()).evaluation, null);
    assert.equal(f.files.has(path.join(path.dirname(f.artifact("test-a")), "evaluation.json")), true);
  });
}

test("missing checkpoint sidecars and missing sidecar hashes invalidate saved evaluations", async () => {
  const f = fixture();
  f.add("test-a");
  f.add("test-a", "metadata");
  const report = sourceReport(f, "test-a", "checkpoint");
  const savedPath = path.join(path.dirname(f.artifact("test-a")), "evaluation.json");
  f.files.set(savedPath, JSON.stringify(report));
  f.files.delete(f.artifact("test-a", "metadata"));
  assert.equal((await f.api.snapshot("swing", "test-a")).evaluation, null);
  f.add("test-a", "metadata");
  delete report.source_files_sha256[f.artifact("test-a", "metadata")];
  f.files.set(savedPath, JSON.stringify(report));
  assert.equal((await f.api.snapshot("swing", "test-a")).evaluation, null);
});

test("changing reference clip bytes invalidates a saved dance verdict and recipe", async () => {
  const f = fixture();
  f.add("dance-input-test", "onnx", "dance");
  const clip = f.addClip("reference.json");
  const recipe = { experimentId: "dance", runName: "dance-input-test", profile: "full", danceClip: clip };
  const report = {
    ...sourceReport(f, "dance-input-test", "policy", "dance"),
    recipe: "dance",
    evaluation_request: { recipe },
    evaluation: { environment: { recipe_options: { dance_clip: clip, dance_clip_sha256: createHash("sha256").update("{}").digest("hex") } } },
  };
  f.files.set(path.join(path.dirname(f.artifact("dance-input-test", "onnx", "dance")), "evaluation.json"), JSON.stringify(report));
  assert.equal((await f.api.snapshot("dance", "dance-input-test")).evaluation.passed, true);
  f.files.set(clip, "changed clip bytes");
  const invalidated = await f.api.snapshot("dance", "dance-input-test");
  assert.equal(invalidated.evaluation.skill_status, "not_assessed");
  assert.equal(invalidated.savedRecipe, null);
  assert.equal(invalidated.renderVerified, false);
});

test("nominal rendering retains the evaluation of its unchanged randomized recipe", async () => {
  const f = fixture();
  f.add("randomized");
  f.add("randomized", "onnx");
  const recipe = input("randomized", { profile: "full", domainRand: true, obsNoise: true, actionDelay: true, maxEpisodeS: 24, renderSeconds: 24 });
  f.api.startJob("eval", recipe);
  finish(f.children[0], { ...sourceReport(f, "randomized"), recipe: "swing" });
  assert.equal((await f.api.snapshot()).evaluation.passed, true);
  f.api.startJob("render", recipe);
  finish(f.children[1]);
  assert.equal((await f.api.snapshot()).evaluation.passed, true);
  f.api.startJob("render", { ...recipe, swingPlanarActions: false });
  assert.equal((await f.api.snapshot()).evaluation.skill_status, "not_assessed");
});

test("source hashes never authorize reads of report-supplied paths", async () => {
  const f = fixture();
  f.add("test-a", "onnx");
  const report = sourceReport(f);
  report.source = "/unowned/secret";
  report.source_files_sha256["/unowned/secret"] = "ignored";
  const savedPath = path.join(path.dirname(f.artifact("test-a")), "evaluation.json");
  f.files.set(savedPath, JSON.stringify(report));
  assert.equal((await f.api.snapshot("swing", "test-a")).evaluation.passed, true);
  assert.equal(f.reads.includes("/unowned/secret"), false);
  report.source_type = "unsupported";
  f.files.set(savedPath, JSON.stringify(report));
  assert.equal((await f.api.snapshot("swing", "test-a")).evaluation, null);
  assert.equal(f.reads.includes("/unowned/secret"), false);
});

test("saved recipes are exposed only from currently bound evaluation evidence", async () => {
  const f = fixture();
  f.add("test-a", "onnx");
  f.api.startJob("eval", input("test-a", { profile: "full", totalTimesteps: 128 }));
  finish(f.children[0], sourceReport(f));
  const bound = await f.api.snapshot();
  assert.equal(bound.savedRecipe.runName, "test-a");
  assert.equal(bound.savedRecipe.profile, "full");
  f.files.set(f.artifact("test-a", "onnx"), "changed");
  const invalidated = await f.api.snapshot();
  assert.equal(invalidated.evaluation, null);
  assert.equal(invalidated.savedRecipe, null);
  f.files.set(f.artifact("test-a", "onnx"), "artifact");
  f.api.startJob("render", input("test-a", { profile: "full", seed: 42 }));
  const mismatched = await f.api.snapshot();
  assert.equal(mismatched.evaluation.evaluation_settings_match, false);
  assert.equal(mismatched.savedRecipe, null);
});

test("reports without source hashes remain visible only as unassessed", async () => {
  const f = fixture();
  const savedPath = path.join(path.dirname(f.artifact("test-a")), "evaluation.json");
  f.files.set(savedPath, JSON.stringify({ passed: true, skill_status: "passed", evaluation_mode: "skill", finite: true }));
  const report = (await f.api.snapshot("swing", "test-a")).evaluation;
  assert.equal(report.finite, true);
  assert.equal(report.skill_status, "not_assessed");
  assert.equal(evaluationVerdict(report, "swing").taskPassed, false);
  assert.equal(evaluationVerdict(report, "dance").taskPassed, false);
});

test("snapshot captures run ownership before asynchronous evidence reads", async () => {
  const f = fixture();
  f.api.startJob("train", input());
  progress(f.children[0]);
  finish(f.children[0]);
  const pending = f.api.snapshot();
  f.api.startJob("train", input("test-b"));
  const previous = await pending;
  assert.equal(previous.runName, "test-a");
  assert.equal(previous.trainingSteps, 4);
  assert.equal(previous.rewardHistory[0].reward, 2.5);
  assert.equal((await f.api.snapshot()).trainingSteps, 0);
});

test("UI requires authoritative scoped skill verdict, not finite output or large spans", () => {
  const pipeline = { evaluation_mode: "pipeline", passed: true, finite: true, pipeline_passed: true, skill_status: "not_assessed" };
  assert.equal(evaluationVerdict(pipeline, "swing").taskPassed, false);
  assert.equal(evaluationVerdict(pipeline, "swing").skillAssessed, false);
  assert.equal(evaluationVerdict(pipeline, "dance").taskPassed, false);
  assert.equal(evaluationVerdict({ passed: true, finite: true }, "swing").taskPassed, false);
  const skill = { ...pipeline, recipe: "swing", evaluation_mode: "skill", skill_status: "failed", swing_span_deg: 175 };
  assert.equal(evaluationVerdict(skill, "swing").taskPassed, false);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "passed" }, "swing").taskPassed, true);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "passed", passed: false }, "swing").taskPassed, false);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "not_assessed" }, "dance").taskPassed, false);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "failed" }, "dance").taskPassed, false);
  assert.equal(evaluationVerdict({ ...skill, recipe: "dance", skill_status: "passed" }, "dance").taskPassed, true);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "not_assessed" }, "running").taskPassed, false);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "failed" }, "running").taskPassed, false);
  assert.equal(evaluationVerdict({ ...skill, recipe: "running", skill_status: "passed" }, "running").taskPassed, true);
  assert.equal(evaluationVerdict({ ...skill, skill_status: "not_assessed" }, "stilts").taskPassed, false);
  assert.equal(evaluationVerdict({ ...skill, recipe: "stilts", skill_status: "passed" }, "stilts").taskPassed, true);
  assert.equal(evaluationVerdict({ ...skill, recipe: "backflip", skill_status: "passed" }, "backflip").taskPassed, true);
  assert.equal(evaluationVerdict({ ...skill, evaluation: { swing_criteria: { min_bidirectional_span_deg: 150 } } }, "swing").swingMinSpanDeg, 150);
});
