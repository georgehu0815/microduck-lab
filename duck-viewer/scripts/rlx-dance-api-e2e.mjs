#!/usr/bin/env node

import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";

const DEFAULT_BASE_URL = "http://127.0.0.1:63317";
const DEFAULT_TIMEOUT_SECONDS = 7_200;
const POLL_INTERVAL_MS = 2_000;
const ARTIFACT_GRACE_MS = 30_000;
const HTTP_REQUEST_TIMEOUT_MS = 30_000;
const EXPERIMENT_IDS = ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge"];
const DEFAULT_EXPERIMENT_ID = "dance";

function usage() {
  console.log(`Usage:
  npm run e2e:rlx:dance -- --execute [options]

Options:
  --base-url URL     Studio URL (default: ${DEFAULT_BASE_URL})
  --experiment ID    dance, swing, running, stilts, backflip, basketball, or bridge (default: dance)
  --run NAME         Run name (default: <experiment>-api-e2e)
  --profile PROFILE  smoke or full (default: full)
  --recipe-json PATH Merge a JSON recipe object before CLI overrides
  --timeout-seconds N Per-operation timeout (default: ${DEFAULT_TIMEOUT_SECONDS})
  --skip-train       Start from an existing checkpoint
  --report PATH      Write the collected JSON report

The script launches real Studio API jobs. It refuses to run without --execute.`);
}

export function parseArgs(argv) {
  const options = {
    baseUrl: DEFAULT_BASE_URL,
    experimentId: null,
    runName: null,
    profile: null,
    recipePath: null,
    skipTrain: false,
    execute: false,
    reportPath: null,
    timeoutSeconds: DEFAULT_TIMEOUT_SECONDS,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--execute") options.execute = true;
    else if (value === "--skip-train") options.skipTrain = true;
    else if (value === "--help" || value === "-h") {
      usage();
      process.exit(0);
    } else if (value === "--base-url") options.baseUrl = requiredValue(argv, ++index, value);
    else if (value === "--experiment") {
      options.experimentId = validatedExperimentId(
        requiredValue(argv, ++index, value),
        value
      );
    } else if (value === "--run") options.runName = requiredValue(argv, ++index, value);
    else if (value === "--profile") options.profile = requiredValue(argv, ++index, value);
    else if (value === "--recipe-json") options.recipePath = requiredValue(argv, ++index, value);
    else if (value === "--timeout-seconds") {
      options.timeoutSeconds = positiveNumber(
        requiredValue(argv, ++index, value),
        value
      );
    } else if (value === "--report") options.reportPath = requiredValue(argv, ++index, value);
    else throw new Error(`Unknown argument: ${value}`);
  }
  if (!options.execute) {
    throw new Error("Refusing to launch RLX jobs without --execute.");
  }
  if (options.profile && !["smoke", "full"].includes(options.profile)) {
    throw new Error("--profile must be smoke or full.");
  }
  options.baseUrl = new URL(options.baseUrl).origin;
  return options;
}

function positiveNumber(value, option) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${option} must be a positive number.`);
  }
  return parsed;
}

function requiredValue(argv, index, option) {
  const value = argv[index];
  if (!value || value.startsWith("--")) {
    throw new Error(`${option} requires a value.`);
  }
  return value;
}

function validatedExperimentId(value, option) {
  if (!EXPERIMENT_IDS.includes(value)) {
    throw new Error(
      `${option} must be one of: ${EXPERIMENT_IDS.join(", ")}.`
    );
  }
  return value;
}

export function mergeRecipe(recipe, options) {
  if (typeof recipe !== "object" || recipe === null || Array.isArray(recipe)) {
    throw new Error("--recipe-json must contain a JSON object.");
  }
  const recipeExperimentId = recipe.experimentId == null
    ? null
    : validatedExperimentId(recipe.experimentId, "--recipe-json experimentId");
  const selectedExperimentId =
    options.experimentId ?? recipeExperimentId ?? DEFAULT_EXPERIMENT_ID;
  return {
    ...recipe,
    experimentId: selectedExperimentId,
    runName:
      options.runName ??
      recipe.runName ??
      `${selectedExperimentId}-api-e2e`,
    profile: options.profile ?? recipe.profile ?? "full",
  };
}

async function loadRecipe(options) {
  let recipe = {};
  if (options.recipePath) {
    recipe = JSON.parse(await readFile(options.recipePath, "utf8"));
  }
  return mergeRecipe(recipe, options);
}

async function jsonRequest(url, init, timeoutMs = HTTP_REQUEST_TIMEOUT_MS) {
  const response = await fetch(url, {
    ...init,
    signal: AbortSignal.timeout(Math.max(1, timeoutMs)),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${body.error ?? JSON.stringify(body)}`);
  }
  return body;
}

function statusUrl(options, recipe) {
  const url = new URL("/api/rlx", options.baseUrl);
  url.searchParams.set("experiment", recipe.experimentId);
  url.searchParams.set("run", recipe.runName);
  return url;
}

async function snapshot(options, recipe, timeoutMs) {
  return jsonRequest(
    statusUrl(options, recipe),
    { cache: "no-store" },
    timeoutMs
  );
}

async function launch(options, action, recipe) {
  const response = await jsonRequest(new URL("/api/rlx", options.baseUrl), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      action,
      recipe,
    }),
  });
  console.log(`${action}: accepted`);
  return response;
}

export function operationEvidenceError(
  action,
  state,
  profile,
  selectedExperimentId = DEFAULT_EXPERIMENT_ID
) {
  const evaluation = state?.evaluation;
  const explicitSkillStatus =
    evaluation?.skill_status === "passed" ||
    evaluation?.skill_status === "failed";
  const requiresExplicitSkillStatus =
    profile === "full" &&
    EXPERIMENT_IDS.includes(selectedExperimentId);
  const acceptedFailedEvaluation =
    action === "eval" &&
    profile === "full" &&
    state?.phase === "failed" &&
    evaluation?.skill_status === "failed";

  if (
    state?.operation !== action ||
    (state?.phase !== "succeeded" && !acceptedFailedEvaluation)
  ) {
    return (
      `${action} ended as phase=${state?.phase ?? "unknown"} ` +
      `operation=${state?.operation ?? "none"}; ` +
      `last log=${state?.logs?.at?.(-1) ?? "none"}`
    );
  }
  if (action === "train" && !state.artifacts?.checkpoint) {
    return "train succeeded without a checkpoint artifact.";
  }
  if ((action === "train" || action === "export") && !state.artifacts?.onnx) {
    return `${action} succeeded without an ONNX artifact.`;
  }
  if (action === "eval") {
    if (!evaluation || typeof evaluation !== "object") {
      return "eval completed without an evaluation report.";
    }
    if (!state.artifacts?.evaluation) {
      return "eval completed without a persisted evaluation artifact.";
    }
    if (requiresExplicitSkillStatus && !explicitSkillStatus) {
      const experimentName =
        selectedExperimentId[0].toUpperCase() + selectedExperimentId.slice(1);
      return (
        `Full ${experimentName} eval requires explicit skill_status passed or failed; ` +
        `received ${String(evaluation.skill_status ?? "missing")}.`
      );
    }
  }
  if (action === "render" && !state.artifacts?.renderVideo) {
    return "render succeeded without an MP4 artifact.";
  }
  if (action === "render" && !state.artifacts?.renderSheet) {
    return "render succeeded without a contact sheet artifact.";
  }
  return null;
}

function timeoutError(action, state, timeoutSeconds) {
  const error = new Error(
    `${action} timed out after ${timeoutSeconds} seconds; ` +
    `last phase=${state?.phase ?? "unknown"} operation=${state?.operation ?? "none"}.`
  );
  error.state = state;
  return error;
}

function evidenceError(message, state) {
  const error = new Error(message);
  error.state = state;
  return error;
}

async function waitForCompletion(options, action, recipe) {
  const deadline = Date.now() + options.timeoutSeconds * 1_000;
  let terminalAt = null;
  let lastState = null;
  for (;;) {
    const remainingMs = deadline - Date.now();
    if (remainingMs <= 0) {
      throw timeoutError(action, lastState, options.timeoutSeconds);
    }
    let state;
    try {
      state = await snapshot(
        options,
        recipe,
        Math.min(remainingMs, HTTP_REQUEST_TIMEOUT_MS)
      );
    } catch (error) {
      if (Date.now() >= deadline) {
        throw timeoutError(action, lastState, options.timeoutSeconds);
      }
      throw error;
    }
    lastState = state;
    const active = state.activeJob;
    const ownsActiveJob =
      active?.experimentId === recipe.experimentId &&
      active?.runName === recipe.runName;
    if (state.phase === "running" || ownsActiveJob) {
      const progress = state.trainingTotal > 0
        ? ` ${state.trainingSteps}/${state.trainingTotal}`
        : "";
      console.log(`${action}: running${progress}`);
      await pollDelay(deadline);
      continue;
    }
    terminalAt ??= Date.now();
    const error = operationEvidenceError(
      action,
      state,
      recipe.profile,
      recipe.experimentId
    );
    if (!error) {
      const outcome =
        action === "eval" && state.evaluation?.skill_status === "failed"
          ? "skill failed"
          : "succeeded";
      console.log(`${action}: ${outcome}`);
      return state;
    }
    if (
      state.operation !== action ||
      Date.now() - terminalAt >= ARTIFACT_GRACE_MS
    ) {
      throw evidenceError(error, state);
    }
    await pollDelay(deadline);
  }
}

async function pollDelay(deadline) {
  const delayMs = Math.min(POLL_INTERVAL_MS, Math.max(1, deadline - Date.now()));
  await new Promise((resolve) => setTimeout(resolve, delayMs));
}

function serializedError(error) {
  return {
    message: error instanceof Error ? error.message : String(error),
  };
}

export async function main(argv = process.argv.slice(2)) {
  const options = parseArgs(argv);
  const report = {
    generatedAt: new Date().toISOString(),
    baseUrl: options.baseUrl,
    timeoutSeconds: options.timeoutSeconds,
    requestedRecipe: null,
    normalizedRecipe: null,
    operations: [],
    final: null,
    failure: null,
  };
  let recipe = null;
  let failure = null;
  let skillFailed = false;

  try {
    const requestedRecipe = await loadRecipe(options);
    report.requestedRecipe = requestedRecipe;
    recipe = requestedRecipe;
    const operations = options.skipTrain
      ? ["eval", "render", "export"]
      : ["train", "eval", "render", "export"];
    const initial = await snapshot(options, requestedRecipe);
    if (initial.activeJob) {
      throw new Error(
        `RLX is already running ${initial.activeJob.operation} for ` +
        `${initial.activeJob.experimentId}/${initial.activeJob.runName}.`
      );
    }
    if (options.skipTrain && !initial.artifacts.checkpoint) {
      throw new Error(
        `--skip-train requires an existing ${requestedRecipe.experimentId} checkpoint.`
      );
    }

    for (const action of operations) {
      const operation = {
        action,
        accepted: null,
        state: null,
        error: null,
      };
      report.operations.push(operation);
      try {
        const accepted = await launch(options, action, recipe);
        operation.accepted = accepted;
        recipe = accepted.recipe;
        report.normalizedRecipe = recipe;
        operation.state = await waitForCompletion(options, action, recipe);
        if (
          action === "eval" &&
          recipe.profile === "full" &&
          operation.state.evaluation?.skill_status === "failed"
        ) {
          skillFailed = true;
        }
      } catch (error) {
        operation.error = serializedError(error);
        if (error?.state) operation.state = error.state;
        throw error;
      }
    }
    if (skillFailed) {
      const experimentName =
        recipe.experimentId[0].toUpperCase() + recipe.experimentId.slice(1);
      throw new Error(
        `Full ${experimentName} skill evaluation failed; ` +
        "render and export evidence were collected."
      );
    }
  } catch (error) {
    failure = error;
    report.failure = serializedError(error);
  } finally {
    if (recipe) {
      try {
        report.final = await snapshot(options, recipe);
      } catch (error) {
        report.finalSnapshotError = serializedError(error);
      }
    }
    if (options.reportPath) {
      try {
        await mkdir(dirname(options.reportPath), { recursive: true });
        await writeFile(
          options.reportPath,
          `${JSON.stringify(report, null, 2)}\n`
        );
        console.log(`report: ${options.reportPath}`);
      } catch (error) {
        if (!failure) failure = error;
        else console.error(`report write failed: ${serializedError(error).message}`);
      }
    } else if (report.final) {
      console.log(JSON.stringify(report.final, null, 2));
    }
  }
  if (failure) throw failure;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
