#!/usr/bin/env node

import { mkdir, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { pathToFileURL } from "node:url";

export const DEFAULT_BASE_URL = "http://127.0.0.1:63317";
export const SCENARIOS = ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge"];

function usage() {
  console.log(`Usage:
  node scripts/verify-lab-ready.mjs [options]

Options:
  --base-url URL       Studio URL (default: ${DEFAULT_BASE_URL})
  --report PATH        Write a JSON readiness report
  --readiness-only     Check routes and scenario recipe availability only
  --help, -h           Show this help

Default mode is a strict, read-only cold-restart check of saved accepted runs.
It does not train a policy or prove physical skill.`);
}

function requiredValue(argv, index, option) {
  const value = argv[index];
  if (!value || value.startsWith("--")) {
    throw new Error(`${option} requires a value.`);
  }
  return value;
}

export function normalizeBaseUrl(value) {
  const input = String(value ?? "").trim();
  const url = new URL(/^[a-z][a-z\d+.-]*:\/\//i.test(input) ? input : `http://${input}`);
  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error("--base-url must use http or https.");
  }
  return url.origin;
}

export function parseArgs(argv = []) {
  const options = {
    baseUrl: DEFAULT_BASE_URL,
    reportPath: null,
    readinessOnly: false,
    help: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--base-url") {
      options.baseUrl = requiredValue(argv, ++index, value);
    } else if (value === "--report") {
      options.reportPath = requiredValue(argv, ++index, value);
    } else if (value === "--readiness-only") {
      options.readinessOnly = true;
    } else if (value === "--help" || value === "-h") {
      options.help = true;
    } else {
      throw new Error(`Unknown argument: ${value}`);
    }
  }
  options.baseUrl = normalizeBaseUrl(options.baseUrl);
  return options;
}

function requestUrl(baseUrl, pathname, searchParams) {
  const url = new URL(pathname, `${baseUrl}/`);
  if (searchParams) url.search = new URLSearchParams(searchParams).toString();
  return url;
}

async function request(fetchImpl, url, init = {}) {
  const response = await fetchImpl(url, {
    ...init,
    signal: init.signal ?? AbortSignal.timeout(15_000),
  });
  if (!response.ok) {
    let detail = "";
    try {
      detail = await response.text();
    } catch {
      detail = response.statusText;
    }
    throw new Error(`${response.status} ${response.statusText}${detail ? `: ${detail}` : ""}`);
  }
  return response;
}

async function requestJson(fetchImpl, url) {
  const response = await request(fetchImpl, url);
  try {
    return await response.json();
  } catch {
    throw new Error(`${url.pathname} did not return JSON.`);
  }
}

function finiteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

export function validateTrainingHistory(snapshot) {
  const issues = [];
  const segments = Array.isArray(snapshot?.trainingHistory?.segments)
    ? snapshot.trainingHistory.segments
    : [];
  const ppoSegments = segments.filter((segment) => segment?.kind === "ppo");
  if (ppoSegments.length === 0) {
    issues.push("persisted training history has no PPO segments");
    return issues;
  }
  for (const segment of ppoSegments) {
    const label = typeof segment.id === "string" ? segment.id : "unnamed PPO segment";
    const collections = Array.isArray(segment.collections) ? segment.collections : [];
    const updates = Array.isArray(segment.updates) ? segment.updates : [];
    if (!collections.some((sample) => finiteNumber(sample?.step) && finiteNumber(sample?.meanReward))) {
      issues.push(`${label} has no persisted reward samples`);
    }
    if (!updates.some((sample) =>
      finiteNumber(sample?.step) &&
      [sample?.meanLoss, sample?.policyLoss, sample?.valueLoss].some(finiteNumber)
    )) {
      issues.push(`${label} has no persisted loss samples`);
    }
  }
  return issues;
}

export function validateScenarioSnapshot(snapshot, scenario, runName) {
  const issues = [];
  if (snapshot?.experimentId !== scenario) issues.push(`snapshot experiment is not ${scenario}`);
  if (snapshot?.runName !== runName) issues.push(`snapshot run is not ${runName}`);
  if (snapshot?.evaluation?.passed !== true) issues.push("evaluation.passed is not true");
  if (snapshot?.renderVerified !== true) issues.push("renderVerified is not true");
  if (!snapshot?.savedRecipe || snapshot.savedRecipe.experimentId !== scenario ||
      snapshot.savedRecipe.runName !== runName) {
    issues.push("saved evaluation recipe is missing or mismatched");
  }
  if (!Array.isArray(snapshot?.rewardHistory) || snapshot.rewardHistory.length === 0) {
    issues.push("restored reward history is empty");
  }
  if (snapshot?.artifacts?.evaluation !== true) issues.push("persisted evaluation artifact is missing");
  if (snapshot?.artifacts?.metadata !== true) issues.push("persisted metadata artifact is missing");
  if (snapshot?.artifacts?.renderVideo !== true) issues.push("saved MP4 artifact is missing");
  issues.push(...validateTrainingHistory(snapshot));
  return issues;
}

export function inspectMp4Bytes(bytes) {
  const data = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes);
  if (data.length < 12) return { passed: false, error: "MP4 response is shorter than 12 bytes." };
  const boxSize = data.readUInt32BE(0);
  const boxType = data.subarray(4, 8).toString("ascii");
  if (boxType !== "ftyp") return { passed: false, error: "MP4 response does not start with an ftyp box." };
  if (boxSize < 8) return { passed: false, error: "MP4 ftyp box has an invalid size." };
  return {
    passed: true,
    boxType,
    boxSize,
    majorBrand: data.subarray(8, 12).toString("ascii"),
    sampledBytes: data.length,
  };
}

function acceptedCandidates(runs, scenario) {
  return runs
    .filter((run) =>
      run?.experimentId === scenario &&
      run?.taskPassed === true &&
      typeof run?.runName === "string" &&
      run.runName.length > 0
    )
    .sort((left, right) =>
      String(right.modifiedAt ?? "").localeCompare(String(left.modifiedAt ?? ""))
    );
}

async function verifyUi(fetchImpl, baseUrl) {
  const response = await request(fetchImpl, requestUrl(baseUrl, "/"));
  const contentType = response.headers.get("content-type") ?? "";
  const body = await response.text();
  if (!contentType.toLowerCase().includes("text/html")) {
    throw new Error(`UI returned ${contentType || "no content type"}, expected text/html.`);
  }
  if (!body.trim()) throw new Error("UI returned an empty HTML document.");
  return { passed: true, status: response.status, contentType, bytes: Buffer.byteLength(body) };
}

async function verifyClips(fetchImpl, baseUrl) {
  const data = await requestJson(fetchImpl, requestUrl(baseUrl, "/api/rlx/clips"));
  if (!Array.isArray(data?.clips) || data.clips.length === 0) {
    throw new Error("/api/rlx/clips returned no clips.");
  }
  return { passed: true, count: data.clips.length };
}

async function verifyRecipeRoute(fetchImpl, baseUrl, scenario) {
  const runName = `${scenario}-readiness-probe`;
  const snapshot = await requestJson(
    fetchImpl,
    requestUrl(baseUrl, "/api/rlx", { experiment: scenario, run: runName })
  );
  if (snapshot?.experimentId !== scenario || snapshot?.runName !== runName ||
      typeof snapshot?.artifacts !== "object" || snapshot.artifacts === null) {
    throw new Error(`${scenario} snapshot/recipe route returned an invalid shape.`);
  }
  return { passed: true, scenario, runName };
}

async function verifyVideo(fetchImpl, baseUrl, scenario, runName) {
  const url = requestUrl(baseUrl, "/api/rlx/artifact", {
    experiment: scenario,
    run: runName,
    kind: "video",
    inline: "1",
  });
  const response = await request(fetchImpl, url, {
    headers: { Range: "bytes=0-63" },
  });
  const contentType = response.headers.get("content-type") ?? "";
  if (!contentType.toLowerCase().startsWith("video/mp4")) {
    throw new Error(`video returned ${contentType || "no content type"}, expected video/mp4.`);
  }
  const inspection = inspectMp4Bytes(await response.arrayBuffer());
  if (!inspection.passed) throw new Error(inspection.error);
  return { passed: true, status: response.status, contentType, ...inspection };
}

async function verifySavedScenario(fetchImpl, baseUrl, scenario, runs) {
  const candidates = acceptedCandidates(runs, scenario);
  if (candidates.length === 0) {
    throw new Error(`No saved accepted ${scenario} run was discovered.`);
  }
  const attempts = [];
  for (const candidate of candidates) {
    const attempt = { runName: candidate.runName, passed: false, issues: [] };
    attempts.push(attempt);
    try {
      const snapshot = await requestJson(
        fetchImpl,
        requestUrl(baseUrl, "/api/rlx", {
          experiment: scenario,
          run: candidate.runName,
        })
      );
      attempt.issues = validateScenarioSnapshot(snapshot, scenario, candidate.runName);
      if (attempt.issues.length > 0) continue;
      const video = await verifyVideo(fetchImpl, baseUrl, scenario, candidate.runName);
      attempt.passed = true;
      return {
        passed: true,
        scenario,
        runName: candidate.runName,
        modifiedAt: candidate.modifiedAt ?? null,
        rewardSamples: snapshot.rewardHistory.length,
        ppoSegments: snapshot.trainingHistory.segments.filter((segment) => segment?.kind === "ppo").length,
        video,
        attempts,
      };
    } catch (error) {
      attempt.issues.push(error instanceof Error ? error.message : String(error));
    }
  }
  throw new Error(
    `No saved accepted ${scenario} run had complete restart evidence: ` +
    attempts.map((attempt) => `${attempt.runName}: ${attempt.issues.join("; ")}`).join(" | ")
  );
}

async function captureCheck(report, name, operation) {
  try {
    const result = await operation();
    report.checks[name] = result;
    return result;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    report.checks[name] = { passed: false, error: message };
    report.errors.push(`${name}: ${message}`);
    return null;
  }
}

export async function verifyLabReady(options = {}, dependencies = {}) {
  const baseUrl = normalizeBaseUrl(options.baseUrl ?? DEFAULT_BASE_URL);
  const readinessOnly = options.readinessOnly === true;
  const fetchImpl = dependencies.fetchImpl ?? globalThis.fetch;
  if (typeof fetchImpl !== "function") throw new Error("A Fetch API implementation is required.");
  const report = {
    passed: false,
    mode: readinessOnly ? "readiness-only" : "strict",
    baseUrl,
    checkedAt: new Date().toISOString(),
    scope: "Read-only cold restart verification; not fresh training or physical skill proof.",
    checks: {},
    scenarios: {},
    errors: [],
  };

  await captureCheck(report, "ui", () => verifyUi(fetchImpl, baseUrl));
  await captureCheck(report, "clips", () => verifyClips(fetchImpl, baseUrl));
  const runsData = await captureCheck(report, "runsApi", async () => {
    const data = await requestJson(fetchImpl, requestUrl(baseUrl, "/api/rlx/runs"));
    if (!Array.isArray(data?.runs)) throw new Error("/api/rlx/runs did not return a runs array.");
    return { passed: true, count: data.runs.length, runs: data.runs };
  });

  for (const scenario of SCENARIOS) {
    const result = await captureCheck(report, `scenario:${scenario}`, () =>
      readinessOnly
        ? verifyRecipeRoute(fetchImpl, baseUrl, scenario)
        : verifySavedScenario(fetchImpl, baseUrl, scenario, runsData?.runs ?? [])
    );
    report.scenarios[scenario] = result ?? report.checks[`scenario:${scenario}`];
  }
  if (runsData) delete report.checks.runsApi.runs;
  report.passed = report.errors.length === 0;
  return report;
}

export async function main(argv = process.argv.slice(2), dependencies = {}) {
  const options = parseArgs(argv);
  if (options.help) {
    usage();
    return { passed: true, help: true };
  }
  let report;
  try {
    report = await verifyLabReady(options, dependencies);
  } catch (error) {
    report = {
      passed: false,
      mode: options.readinessOnly ? "readiness-only" : "strict",
      baseUrl: options.baseUrl,
      checkedAt: new Date().toISOString(),
      errors: [error instanceof Error ? error.message : String(error)],
    };
  }
  if (options.reportPath) {
    await mkdir(dirname(options.reportPath), { recursive: true });
    await writeFile(options.reportPath, `${JSON.stringify(report, null, 2)}\n`);
  }
  console.log(JSON.stringify(report, null, 2));
  if (!report.passed) {
    const error = new Error(`Lab restart readiness failed with ${report.errors.length} error(s).`);
    error.report = report;
    throw error;
  }
  return report;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error instanceof Error ? error.message : String(error));
    process.exitCode = 1;
  });
}
