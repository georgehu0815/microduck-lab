import assert from "node:assert/strict";
import { EventEmitter } from "node:events";
import { createHash } from "node:crypto";
import fs from "node:fs";
import { createRequire } from "node:module";
import vm from "node:vm";
import { test } from "node:test";
import ts from "typescript";

const loadDependency = createRequire(import.meta.url);

function load(filename, dependencies = {}) {
  const source = fs.readFileSync(new URL(filename, import.meta.url), "utf8");
  const { outputText, diagnostics } = ts.transpileModule(source, {
    reportDiagnostics: true,
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      target: ts.ScriptTarget.ES2022,
      esModuleInterop: true,
    },
  });
  assert.equal(diagnostics.length, 0);
  const exports = {};
  vm.runInNewContext(outputText, {
    exports, process, Buffer,
    require: (id) => id in dependencies ? dependencies[id] : loadDependency(id),
  }, { filename });
  return exports;
}

const experiments = load("experiments.ts");
const history = {
  collectTrainingHistory: () => ({ segments: [] }),
  prepareTrainingHistory: () => 0,
  restoreTrainingInvocation: () => ({
    rewardHistory: [],
    trainingSteps: 0,
    trainingTotal: 0,
    normalizeRewards: false,
  }),
};

for (const experimentId of ["running", "stilts"]) {
  test(`${experimentId} yaw-only shaping is optional and reaches the CLI`, () => {
    const { api, children } = fixture();
    assert.equal(api.normalizeRecipe({ experimentId }).rewardWeights.yaw_tracking, 0);
    const recipe = api.startJob("train", {
      experimentId,
      rewardWeights: { yaw_tracking: 8 },
      resumeFromCheckpoint: true,
    });
    assert.equal(recipe.rewardWeights.yaw_tracking, 8);
    const args = children[0].args;
    assert.equal(JSON.parse(args[args.indexOf("--weight-overrides") + 1]).yaw_tracking, 8);
  });
}

function fixture() {
  const children = [];
  const writes = new Map();
  const files = new Map();
  const api = load("rlx-job.ts", {
    "@/lib/experiments": experiments,
    "@/lib/rlx-history": history,
    "@/lib/rlx-render-evidence": {
      prepareRenderEvidence: (input) => input,
      finalizeRenderEvidence: () => null,
      readRenderEvidence: () => null,
    },
    "node:fs": { existsSync: (file) => !file.endsWith("/.restart-lab/lock") },
    "node:fs/promises": {
      writeFile: async (file, data) => { writes.set(file, JSON.parse(data)); },
      readFile: async (file) => {
        if (files.has(file)) return files.get(file);
        if (writes.has(file)) return JSON.stringify(writes.get(file));
        throw new Error("ENOENT");
      },
    },
    "node:child_process": {
      spawn: (_command, args) => {
        const child = new EventEmitter();
        Object.assign(child, {
          args, stdout: new EventEmitter(), stderr: new EventEmitter(),
        });
        children.push(child);
        return child;
      },
    },
  });
  return { api, children, writes, files };
}

test("Full Running defaults satisfy the 600-step horizon without changing smoke", () => {
  const { api, children } = fixture();
  const recipe = api.startJob("eval", { experimentId: "running", profile: "full" });
  assert.equal(recipe.maxEpisodeS, 12);
  assert.equal(recipe.evalSteps, 600);
  const args = children[0].args;
  assert.equal(args[args.indexOf("--eval-steps") + 1], "600");
  assert.equal(args[args.indexOf("--max-episode-s") + 1], "12");
  for (const overrides of [{ maxEpisodeS: 10 }, { evalSteps: 599 }]) {
    assert.throws(
      () => api.normalizeRecipe({ experimentId: "running", profile: "full", ...overrides }),
      /at least 12 episode seconds and 600 evaluation steps/
    );
  }
  const longer = api.normalizeRecipe({ experimentId: "running", profile: "full", maxEpisodeS: 15 });
  assert.equal(longer.evalSteps, 750);
  const smoke = api.normalizeRecipe({ experimentId: "running", profile: "smoke" });
  assert.equal(smoke.numSteps, 2);
  assert.equal(smoke.numEnvs, 2);
  assert.equal(smoke.maxEpisodeS, 1);
  assert.equal(smoke.evalSteps, 4);
});

function finish(child, report = { rendered: true }) {
  child.stdout.emit("data", `${JSON.stringify(report)}\n`);
  child.emit("close", 0, null);
}

function evaluatedFixture(experimentId, overrides = {}) {
  const current = fixture();
  const recipe = current.api.startJob("eval", {
    experimentId, profile: "full", runName: "bound-environment",
    domainRand: false, obsNoise: false, actionDelay: false, randomYaw: false,
    maxEpisodeS: 12, renderSeconds: 12,
    ...(experimentId === "running" ? { locomotionForwardCommand: 0.6 } : {}),
    ...(experimentId === "stilts" ? { locomotionForwardCommand: 0.15 } : {}),
    ...overrides,
  });
  const source = current.api.artifactPath(experimentId, recipe.runName, "onnx");
  current.files.set(source, Buffer.from("unchanged policy"));
  const hash = createHash("sha256").update(current.files.get(source)).digest("hex");
  finish(current.children.at(-1), {
    passed: true, skill_status: "passed", evaluation_mode: "skill",
    source_type: "policy", source_sha256: hash, source_files_sha256: { [source]: hash },
  });
  return { ...current, recipe };
}

for (const experimentId of ["running", "stilts", "dance"]) {
  test(`${experimentId} retains matching evaluation across render and export`, async () => {
    const { api, children, recipe } = evaluatedFixture(experimentId);
    assert.equal((await api.snapshot()).evaluation.passed, true);
    api.startJob("render", { ...recipe, learningRate: 0.0001 });
    assert.equal((await api.snapshot()).evaluation.passed, true);
    finish(children.at(-1));
    assert.equal((await api.snapshot()).evaluation.skill_status, "passed");
    api.startJob("export", recipe);
    finish(children.at(-1));
    assert.equal((await api.snapshot()).evaluation.passed, true);
  });
}

for (const [experimentId, changes] of [
  ["running", { locomotionForwardCommand: 0.7 }],
  ["running", { locomotionForwardCommand: null }],
  ["running", { seed: 42 }],
  ["stilts", { stiltHeightCm: 15 }],
  ["stilts", { stiltBlend: 0.8 }],
  ["stilts", { stiltMassKg: 0.05 }],
  ["dance", { dancePoseSigma: 0.2 }],
]) {
  test(`${experimentId} render change ${JSON.stringify(changes)} invalidates prior skill evidence`, async () => {
    const { api, children, recipe } = evaluatedFixture(experimentId);
    assert.equal((await api.snapshot()).evaluation.passed, true);
    api.startJob("render", { ...recipe, ...changes });
    const running = await api.snapshot();
    assert.equal(running.evaluation.passed, false);
    assert.equal(running.evaluation.skill_status, "not_assessed");
    assert.equal(running.evaluation.evaluation_settings_match, false);
    finish(children.at(-1));
    assert.equal((await api.snapshot()).evaluation.passed, false);
    api.startJob("export", recipe);
    finish(children.at(-1));
    assert.equal((await api.snapshot()).evaluation.passed, false);
    api.startJob("render", { ...recipe, runName: "another-run" });
    finish(children.at(-1));
    const saved = await api.snapshot(experimentId, recipe.runName);
    assert.equal(saved.evaluation.passed, false);
    assert.equal(saved.evaluation.skill_status, "not_assessed");
  });
}

test("changing video duration preserves the saved evaluation but does not grant visual provenance", async () => {
  const { api, children, recipe } = evaluatedFixture("running");
  api.startJob("render", { ...recipe, renderSeconds: 15 });
  finish(children.at(-1));
  const rendered = await api.snapshot();
  assert.equal(rendered.evaluation.passed, true);
  assert.equal(rendered.evaluation.evaluation_request.recipe.renderSeconds, recipe.renderSeconds);
  assert.equal(rendered.renderVerified, false);
});

for (const flag of ["domainRand", "obsNoise", "actionDelay", "randomYaw"]) {
  test(`nominal video projection does not relabel the evaluation's ${flag}=true scope`, async () => {
    const { api, children, recipe } = evaluatedFixture("running", { [flag]: true });
    assert.equal((await api.snapshot()).evaluation.passed, true);
    api.startJob("render", recipe);
    assert.equal((await api.snapshot()).evaluation.skill_status, "passed");
    finish(children.at(-1));
    const rendered = await api.snapshot();
    assert.equal(rendered.evaluation.passed, true);
    assert.equal(rendered.evaluation.evaluation_request.recipe[flag], true);
    assert.equal(rendered.renderVerified, false);
  });
}

test("changing the requested evaluation recipe is not hidden by renderer normalization", async () => {
  const { api, children, recipe } = evaluatedFixture("running");
  api.startJob("render", {
    ...recipe,
    domainRand: true, obsNoise: true, actionDelay: true, randomYaw: true,
    maxEpisodeS: 20,
  });
  finish(children.at(-1));
  assert.equal((await api.snapshot()).evaluation.passed, false);
});

test("forward command preserves absent/null defaults for every recipe", () => {
  const { api } = fixture();
  for (const experimentId of ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge"]) {
    for (const locomotionForwardCommand of [undefined, null]) {
      assert.equal(api.normalizeRecipe({ experimentId, locomotionForwardCommand }).locomotionForwardCommand, null);
    }
  }
});

test("observation normalization freezing defaults for Full Backflip and Bridge plus all recurrent Basketball runs", () => {
  const { api, children } = fixture();
  for (const experimentId of ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge"]) {
    const basketball = experimentId === "basketball";
    assert.equal(api.normalizeRecipe({ experimentId }).freezeObservationNormalization, basketball);
    for (const freezeObservationNormalization of [undefined, null, false]) {
      assert.equal(
        api.normalizeRecipe({ experimentId, freezeObservationNormalization }).freezeObservationNormalization,
        basketball
      );
    }
    assert.equal(api.normalizeRecipe({
      experimentId, profile: "full", freezeObservationNormalization: true, resumeFromCheckpoint: true,
    }).freezeObservationNormalization, true);
  }
  assert.equal(api.normalizeRecipe({
    experimentId: "backflip", profile: "full",
  }).freezeObservationNormalization, true);
  assert.equal(api.normalizeRecipe({
    experimentId: "backflip", profile: "full", freezeObservationNormalization: false,
  }).freezeObservationNormalization, false);
  assert.equal(api.normalizeRecipe({
    experimentId: "backflip", profile: "smoke", freezeObservationNormalization: true,
  }).freezeObservationNormalization, false);
  assert.equal(api.normalizeRecipe({
    experimentId: "bridge", profile: "full",
  }).freezeObservationNormalization, true);
  assert.equal(api.normalizeRecipe({
    experimentId: "bridge", profile: "full", freezeObservationNormalization: false,
  }).freezeObservationNormalization, false);
  assert.equal(api.normalizeRecipe({
    experimentId: "bridge", profile: "smoke", freezeObservationNormalization: true,
  }).freezeObservationNormalization, false);
  for (const resumeFromCheckpoint of [undefined, null, false, "true", 1]) {
    assert.throws(
      () => api.normalizeRecipe({
        experimentId: "swing",
        profile: "full",
        freezeObservationNormalization: true,
        resumeFromCheckpoint,
      }),
      /requires resumeFromCheckpoint to be true/
    );
  }
  for (const freezeObservationNormalization of ["true", "false", 0, 1, {}, []]) {
    assert.throws(
      () => api.normalizeRecipe({ freezeObservationNormalization, resumeFromCheckpoint: true }),
      /must be a boolean/
    );
  }
  assert.equal(children.length, 0);
});

test("only resumed training receives the observation normalization freeze flag", () => {
  const { api, children, writes } = fixture();
  for (const freezeObservationNormalization of [undefined, false, true]) {
    for (const operation of ["train", "eval", "render", "export"]) {
      const recipe = api.startJob(operation, {
        experimentId: "swing", profile: "full", resumeFromCheckpoint: true, freezeObservationNormalization,
      });
      const child = children.at(-1);
      const expected = freezeObservationNormalization === true;
      assert.equal(recipe.freezeObservationNormalization, expected);
      assert.equal(
        child.args.filter((arg) => arg === "--freeze-observation-normalization").length,
        operation === "train" && expected ? 1 : 0
      );
      assert.equal(child.args.includes("--no-freeze-observation-normalization"), false);
      if (operation === "train") {
        const initIndex = child.args.indexOf("--init-from");
        assert.notEqual(initIndex, -1);
        assert.equal(child.args[initIndex + 1], api.artifactPath("swing", recipe.runName, "checkpoint"));
      }
      finish(child, { passed: true });
      if (operation === "eval") {
        const storedRecipe = [...writes.values()].at(-1).evaluation_request.recipe;
        assert.equal(storedRecipe.freezeObservationNormalization, expected);
        assert.equal(storedRecipe.resumeFromCheckpoint, true);
      }
    }
  }
});

for (const freezeObservationNormalization of [false, true]) {
  test(`changing freezeObservationNormalization from ${freezeObservationNormalization} does not change the environment key`, async () => {
    const { api, children, recipe } = evaluatedFixture("swing", {
      resumeFromCheckpoint: true, freezeObservationNormalization,
    });
    assert.equal((await api.snapshot()).evaluation.passed, true);
    api.startJob("render", {
      ...recipe, freezeObservationNormalization: !freezeObservationNormalization,
    });
    finish(children.at(-1));
    const rendered = await api.snapshot();
    assert.equal(rendered.evaluation.passed, true);
    assert.equal(rendered.evaluation.evaluation_request.recipe.freezeObservationNormalization, freezeObservationNormalization);
  });
}

test("forward command accepts only finite positive numbers up to 1.5 for locomotion", () => {
  const { api } = fixture();
  for (const experimentId of ["running", "stilts"]) {
    for (const value of [0.15, 0.6, 1.5]) {
      assert.equal(api.normalizeRecipe({ experimentId, locomotionForwardCommand: value }).locomotionForwardCommand, value);
    }
    for (const value of [0, -0.1, 1.5001, NaN, Infinity, -Infinity, "0.6", true, {}, []]) {
      assert.throws(
        () => api.normalizeRecipe({ experimentId, locomotionForwardCommand: value }),
        /finite number greater than 0 and at most 1.5/
      );
    }
  }
  for (const experimentId of ["dance", "swing", "backflip"]) {
    assert.throws(
      () => api.normalizeRecipe({ experimentId, locomotionForwardCommand: 0.6 }),
      /only be used with running or stilts/
    );
  }
});

for (const [experimentId, command] of [["running", 0.6], ["stilts", 0.15]]) {
  test(`${experimentId} forwards the same command for train/eval/render, not export`, () => {
    const { api, children, writes } = fixture();
    for (const profile of ["smoke", "full"]) {
      for (const operation of ["train", "eval", "render", "export"]) {
        const recipe = api.startJob(operation, {
          experimentId, profile, locomotionForwardCommand: command,
          ...(operation === "train" ? { resumeFromCheckpoint: true } : {}),
        });
        const child = children.at(-1);
        const flagIndex = child.args.indexOf("--locomotion-forward-command");
        if (operation === "export") {
          assert.equal(flagIndex, -1);
        } else {
          assert.notEqual(flagIndex, -1);
          assert.equal(child.args[flagIndex + 1], String(command));
          assert.equal(child.args.filter((arg) => arg === "--locomotion-forward-command").length, 1);
        }
        child.stdout.emit("data", '{"passed":true}\n');
        child.emit("close", 0, null);
        if (operation === "eval") {
          const report = [...writes.values()].at(-1);
          assert.equal(report.evaluation_request.locomotion_forward_command, command);
          assert.deepEqual(report.evaluation_request.recipe, JSON.parse(JSON.stringify(recipe)));
        }
      }
    }
  });
}

test("unset forward commands emit no CLI override or evaluation setting", () => {
  const { api, children, writes } = fixture();
  for (const experimentId of ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge"]) {
    for (const locomotionForwardCommand of [undefined, null]) {
      for (const operation of ["train", "eval", "render"]) {
        api.startJob(operation, {
          experimentId,
          locomotionForwardCommand,
          ...(operation === "train" ? { resumeFromCheckpoint: true } : {}),
        });
        const child = children.at(-1);
        assert.equal(child.args.includes("--locomotion-forward-command"), false);
        child.stdout.emit("data", '{"passed":true}\n');
        child.emit("close", 0, null);
        if (operation === "eval") {
          const report = [...writes.values()].at(-1);
          assert.equal(Object.hasOwn(report.evaluation_request, "locomotion_forward_command"), false);
          assert.equal(report.evaluation_request.recipe.locomotionForwardCommand, null);
        }
      }
    }
  }
});
