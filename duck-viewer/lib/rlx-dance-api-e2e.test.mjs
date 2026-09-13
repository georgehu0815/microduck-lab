import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import {
  main,
  mergeRecipe,
  operationEvidenceError,
  parseArgs,
} from "../scripts/rlx-dance-api-e2e.mjs";

test("HTTP driver creates report parents even for an early recipe failure", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "rlx-report-parent-"));
  const report = path.join(directory, "new", "nested", "report.json");
  try {
    await assert.rejects(main(["--execute", "--recipe-json", path.join(directory, "missing.json"), "--report", report]), /ENOENT/);
    const saved = JSON.parse(await readFile(report, "utf8"));
    assert.match(saved.failure.message, /ENOENT/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("Dance HTTP driver merges recipe JSON with explicit CLI precedence", () => {
  const options = parseArgs([
    "--execute",
    "--recipe-json",
    "/tmp/dance.json",
    "--run",
    "cli-run",
    "--profile",
    "smoke",
  ]);
  assert.deepEqual(
    mergeRecipe(
      {
        experimentId: "dance",
        runName: "file-run",
        profile: "full",
        danceClip: "dance-clip/derived/dance.json",
        dancePoseSigma: 0.2,
        numSteps: 64,
        evalSteps: 1_200,
        initialStd: 0.3,
        normalizeRewards: true,
        checkpointInterval: 1_024,
      },
      options
    ),
    {
      experimentId: "dance",
      runName: "cli-run",
      profile: "smoke",
      danceClip: "dance-clip/derived/dance.json",
      dancePoseSigma: 0.2,
      numSteps: 64,
      evalSteps: 1_200,
      initialStd: 0.3,
      normalizeRewards: true,
      checkpointInterval: 1_024,
    }
  );
});

test("HTTP driver accepts all scenarios and defaults to Dance", () => {
  const options = parseArgs(["--execute"]);
  assert.equal(mergeRecipe({}, options).experimentId, "dance");
  assert.equal(mergeRecipe({}, options).runName, "dance-api-e2e");
  for (const experimentId of ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge"]) {
    assert.deepEqual(
      mergeRecipe({ experimentId }, options),
      {
        experimentId,
        runName: `${experimentId}-api-e2e`,
        profile: "full",
      }
    );
  }
});

test("HTTP driver gives explicit --experiment precedence over recipe JSON", () => {
  const options = parseArgs([
    "--execute",
    "--experiment",
    "running",
  ]);
  assert.deepEqual(
    mergeRecipe(
      {
        experimentId: "swing",
        runName: "file-run",
      },
      options
    ),
    {
      experimentId: "running",
      runName: "file-run",
      profile: "full",
    }
  );
});

test("HTTP driver rejects unknown scenarios and non-object recipe JSON", () => {
  const options = parseArgs(["--execute"]);
  assert.throws(
    () => parseArgs(["--execute", "--experiment", "jumping"]),
    /must be one of: dance, swing, running, stilts, backflip, basketball, bridge/
  );
  assert.throws(
    () => mergeRecipe({ experimentId: "jumping" }, options),
    /must be one of: dance, swing, running, stilts, backflip/
  );
  assert.throws(
    () => mergeRecipe([], options),
    /must contain a JSON object/
  );
});

test("Dance HTTP driver applies and validates a bounded operation timeout", () => {
  assert.equal(parseArgs(["--execute"]).timeoutSeconds, 7_200);
  assert.equal(
    parseArgs(["--execute", "--timeout-seconds", "12.5"]).timeoutSeconds,
    12.5
  );
  assert.throws(
    () => parseArgs(["--execute", "--timeout-seconds", "0"]),
    /must be a positive number/
  );
});

function state(overrides = {}) {
  return {
    phase: "succeeded",
    operation: "train",
    logs: [],
    evaluation: null,
    artifacts: {
      checkpoint: true,
      onnx: true,
      evaluation: false,
      renderVideo: false,
      renderSheet: false,
    },
    ...overrides,
  };
}

test("Dance HTTP driver requires operation artifacts after successful jobs", () => {
  assert.equal(operationEvidenceError("train", state(), "full"), null);
  assert.match(
    operationEvidenceError(
      "train",
      state({ artifacts: { checkpoint: true, onnx: false } }),
      "full"
    ),
    /without an ONNX/
  );
  assert.match(
    operationEvidenceError(
      "render",
      state({
        operation: "render",
        artifacts: { renderVideo: true, renderSheet: false },
      }),
      "full"
    ),
    /contact sheet/
  );
  assert.equal(
    operationEvidenceError(
      "render",
      state({
        operation: "render",
        artifacts: { renderVideo: true, renderSheet: true },
      }),
      "full"
    ),
    null
  );
  assert.equal(
    operationEvidenceError(
      "export",
      state({ operation: "export", artifacts: { onnx: true } }),
      "full"
    ),
    null
  );
});

test("Full Dance eval accepts explicit pass or failure evidence only", () => {
  const evaluationState = (skillStatus, phase = "succeeded") =>
    state({
      phase,
      operation: "eval",
      evaluation: {
        evaluation_mode: "skill",
        skill_status: skillStatus,
      },
      artifacts: { evaluation: true },
    });

  assert.equal(
    operationEvidenceError("eval", evaluationState("passed"), "full"),
    null
  );
  assert.equal(
    operationEvidenceError("eval", evaluationState("failed", "failed"), "full"),
    null
  );
  assert.match(
    operationEvidenceError("eval", evaluationState("not_assessed"), "full"),
    /explicit skill_status/
  );
  assert.equal(
    operationEvidenceError("eval", evaluationState("not_assessed"), "smoke"),
    null
  );
  assert.match(
    operationEvidenceError(
      "eval",
      state({
        operation: "eval",
        evaluation: { skill_status: "passed" },
        artifacts: { evaluation: false },
      }),
      "full"
    ),
    /persisted evaluation artifact/
  );
});

for (const experimentId of ["running", "stilts", "backflip"]) {
test(`${experimentId} full eval rejects a non-assessed skill status`, () => {
  assert.match(
    operationEvidenceError(
      "eval",
      state({
        operation: "eval",
        evaluation: {
          evaluation_mode: "skill",
          skill_status: "not_assessed",
          passed: true,
        },
        artifacts: { evaluation: true },
      }),
      "full",
      experimentId
    ),
    /explicit skill_status/
  );
});
}

test("Full Dance skill failure still renders, exports, and persists its report", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "dance-api-e2e-"));
  const reportPath = path.join(directory, "report.json");
  const originalFetch = globalThis.fetch;
  const launched = [];
  const recipe = {
    experimentId: "dance",
    runName: "failed-skill",
    profile: "full",
  };
  const snapshotState = (operation, overrides = {}) => ({
    phase: operation ? "succeeded" : "idle",
    operation,
    activeJob: null,
    experimentId: "dance",
    runName: recipe.runName,
    logs: [],
    evaluation: null,
    trainingSteps: 0,
    trainingTotal: 0,
    artifacts: {
      checkpoint: true,
      onnx: true,
      evaluation: false,
      renderVideo: false,
      renderSheet: false,
    },
    ...overrides,
  });
  const snapshots = [
    snapshotState(null),
    snapshotState("eval", {
      phase: "failed",
      evaluation: {
        evaluation_mode: "skill",
        skill_status: "failed",
        passed: false,
      },
      artifacts: {
        checkpoint: true,
        onnx: true,
        evaluation: true,
        renderVideo: false,
        renderSheet: false,
      },
    }),
    snapshotState("render", {
      artifacts: {
        checkpoint: true,
        onnx: true,
        evaluation: true,
        renderVideo: true,
        renderSheet: true,
      },
    }),
    snapshotState("export", {
      artifacts: {
        checkpoint: true,
        onnx: true,
        evaluation: true,
        renderVideo: true,
        renderSheet: true,
      },
    }),
    snapshotState("export", {
      artifacts: {
        checkpoint: true,
        onnx: true,
        evaluation: true,
        renderVideo: true,
        renderSheet: true,
      },
    }),
  ];

  globalThis.fetch = async (_url, init = {}) => {
    if (init.method === "POST") {
      const body = JSON.parse(init.body);
      launched.push(body.action);
      return Response.json({
        accepted: true,
        action: body.action,
        recipe,
      }, { status: 202 });
    }
    const state = snapshots.shift();
    assert.ok(state, "unexpected extra status request");
    return Response.json(state);
  };

  try {
    await assert.rejects(
      main([
        "--execute",
        "--skip-train",
        "--run",
        recipe.runName,
        "--profile",
        "full",
        "--report",
        reportPath,
      ]),
      /skill evaluation failed/
    );
    assert.deepEqual(launched, ["eval", "render", "export"]);
    const report = JSON.parse(await readFile(reportPath, "utf8"));
    assert.deepEqual(
      report.operations.map((operation) => operation.action),
      ["eval", "render", "export"]
    );
    assert.equal(
      report.operations[0].state.evaluation.skill_status,
      "failed"
    );
    assert.equal(report.operations[1].state.artifacts.renderVideo, true);
    assert.equal(report.operations[2].state.artifacts.onnx, true);
    assert.match(report.failure.message, /evidence were collected/);
  } finally {
    globalThis.fetch = originalFetch;
    await rm(directory, { recursive: true, force: true });
  }
});

test("Swing selection is used for status, launches, and failure evidence", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "swing-api-e2e-"));
  const reportPath = path.join(directory, "report.json");
  const originalFetch = globalThis.fetch;
  const launched = [];
  const recipe = {
    experimentId: "swing",
    runName: "failed-swing",
    profile: "full",
  };
  const snapshotState = (operation, overrides = {}) => ({
    phase: operation ? "succeeded" : "idle",
    operation,
    activeJob: null,
    experimentId: recipe.experimentId,
    runName: recipe.runName,
    logs: [],
    evaluation: null,
    trainingSteps: 0,
    trainingTotal: 0,
    artifacts: {
      checkpoint: true,
      onnx: true,
      evaluation: false,
      renderVideo: false,
      renderSheet: false,
    },
    ...overrides,
  });
  const snapshots = [
    snapshotState(null),
    snapshotState("eval", {
      phase: "failed",
      evaluation: {
        evaluation_mode: "skill",
        skill_status: "failed",
        passed: false,
      },
      artifacts: {
        checkpoint: true,
        onnx: true,
        evaluation: true,
        renderVideo: false,
        renderSheet: false,
      },
    }),
    snapshotState("render", {
      artifacts: {
        checkpoint: true,
        onnx: true,
        evaluation: true,
        renderVideo: true,
        renderSheet: true,
      },
    }),
    snapshotState("export", {
      artifacts: {
        checkpoint: true,
        onnx: true,
        evaluation: true,
        renderVideo: true,
        renderSheet: true,
      },
    }),
    snapshotState("export", {
      artifacts: {
        checkpoint: true,
        onnx: true,
        evaluation: true,
        renderVideo: true,
        renderSheet: true,
      },
    }),
  ];

  globalThis.fetch = async (url, init = {}) => {
    if (init.method === "POST") {
      const body = JSON.parse(init.body);
      assert.equal(body.recipe.experimentId, recipe.experimentId);
      launched.push(body.action);
      return Response.json({
        accepted: true,
        action: body.action,
        recipe,
      }, { status: 202 });
    }
    const requestUrl = new URL(url);
    assert.equal(requestUrl.searchParams.get("experiment"), recipe.experimentId);
    assert.equal(requestUrl.searchParams.get("run"), recipe.runName);
    const nextState = snapshots.shift();
    assert.ok(nextState, "unexpected extra status request");
    return Response.json(nextState);
  };

  try {
    await assert.rejects(
      main([
        "--execute",
        "--experiment",
        recipe.experimentId,
        "--skip-train",
        "--run",
        recipe.runName,
        "--profile",
        "full",
        "--report",
        reportPath,
      ]),
      /Swing skill evaluation failed/
    );
    assert.deepEqual(launched, ["eval", "render", "export"]);
    const report = JSON.parse(await readFile(reportPath, "utf8"));
    assert.equal(report.requestedRecipe.experimentId, recipe.experimentId);
    assert.equal(report.normalizedRecipe.experimentId, recipe.experimentId);
    assert.equal(report.final.experimentId, recipe.experimentId);
    assert.match(report.failure.message, /evidence were collected/);
  } finally {
    globalThis.fetch = originalFetch;
    await rm(directory, { recursive: true, force: true });
  }
});
