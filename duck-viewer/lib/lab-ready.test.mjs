import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import {
  inspectMp4Bytes,
  main,
  normalizeBaseUrl,
  parseArgs,
  validateScenarioSnapshot,
  verifyLabReady,
} from "../scripts/verify-lab-ready.mjs";

const scenarios = ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge"];

function mp4Bytes() {
  return Buffer.from([
    0, 0, 0, 24, 0x66, 0x74, 0x79, 0x70,
    0x69, 0x73, 0x6f, 0x6d, 0, 0, 0, 0,
    0x69, 0x73, 0x6f, 0x6d, 0x6d, 0x70, 0x34, 0x32,
  ]);
}

function snapshot(scenario, runName, overrides = {}) {
  return {
    experimentId: scenario,
    runName,
    evaluation: { passed: true },
    renderVerified: true,
    savedRecipe: { experimentId: scenario, runName },
    rewardHistory: [{ step: 10, reward: 1.25 }],
    trainingHistory: {
      segments: [{
        id: `${runName}-ppo`,
        kind: "ppo",
        collections: [{ step: 10, meanReward: 1.25 }],
        updates: [{ step: 10, meanLoss: 0.4, policyLoss: 0.2, valueLoss: 0.1 }],
      }],
    },
    artifacts: {
      evaluation: true,
      metadata: true,
      renderVideo: true,
    },
    ...overrides,
  };
}

function json(data, init = {}) {
  return new Response(JSON.stringify(data), {
    status: 200,
    headers: { "content-type": "application/json" },
    ...init,
  });
}

function fixtureFetch({ runs = null, snapshots = {}, invalidNewestDance = false } = {}) {
  const savedRuns = runs ?? scenarios.flatMap((scenario) => {
    const entries = [{
      experimentId: scenario,
      runName: `${scenario}-accepted`,
      modifiedAt: "2026-09-08T12:00:00.000Z",
      taskPassed: true,
      video: true,
    }];
    if (scenario === "dance" && invalidNewestDance) {
      entries.unshift({
        experimentId: scenario,
        runName: "dance-newest-incomplete",
        modifiedAt: "2026-09-08T13:00:00.000Z",
        taskPassed: true,
        video: true,
      });
    }
    return entries;
  });
  return async (input) => {
    const url = new URL(input);
    if (url.pathname === "/") {
      return new Response("<!doctype html><title>Microduck Studio</title>", {
        headers: { "content-type": "text/html; charset=utf-8" },
      });
    }
    if (url.pathname === "/api/rlx/clips") {
      return json({ clips: [{ path: "dance-clip/test.clip.json" }] });
    }
    if (url.pathname === "/api/rlx/runs") return json({ runs: savedRuns });
    if (url.pathname === "/api/rlx") {
      const scenario = url.searchParams.get("experiment");
      const runName = url.searchParams.get("run");
      const key = `${scenario}/${runName}`;
      if (snapshots[key]) return json(snapshots[key]);
      if (runName.endsWith("-readiness-probe")) {
        return json({ experimentId: scenario, runName, artifacts: {} });
      }
      if (runName === "dance-newest-incomplete") {
        return json(snapshot(scenario, runName, { renderVerified: false }));
      }
      return json(snapshot(scenario, runName));
    }
    if (url.pathname === "/api/rlx/artifact") {
      return new Response(mp4Bytes(), {
        status: 206,
        headers: {
          "content-type": "video/mp4",
          "content-range": "bytes 0-23/24",
        },
      });
    }
    return new Response("not found", { status: 404 });
  };
}

test("CLI parses the default strict mode and portable readiness-only mode", () => {
  assert.deepEqual(parseArgs([]), {
    baseUrl: "http://127.0.0.1:63317",
    reportPath: null,
    readinessOnly: false,
    help: false,
  });
  assert.equal(normalizeBaseUrl("localhost:7000/path"), "http://localhost:7000");
  assert.equal(parseArgs(["--base-url", "https://studio.test/x", "--readiness-only"]).readinessOnly, true);
  assert.throws(() => parseArgs(["--allow-missing-evidence"]), /Unknown argument/);
});

test("strict verification discovers accepted runs and falls back from incomplete newer evidence", async () => {
  const report = await verifyLabReady(
    { baseUrl: "http://studio.test" },
    { fetchImpl: fixtureFetch({ invalidNewestDance: true }) }
  );
  assert.equal(report.passed, true);
  assert.equal(report.mode, "strict");
  assert.equal(report.scenarios.dance.runName, "dance-accepted");
  assert.deepEqual(
    report.scenarios.dance.attempts.map(({ runName, passed }) => ({ runName, passed })),
    [
      { runName: "dance-newest-incomplete", passed: false },
      { runName: "dance-accepted", passed: true },
    ]
  );
  for (const scenario of scenarios) {
    assert.equal(report.scenarios[scenario].video.boxType, "ftyp");
  }
});

test("strict snapshot validation requires accepted evaluation, render provenance, reward and loss history", () => {
  const broken = snapshot("running", "run", {
    evaluation: { passed: false },
    renderVerified: false,
    rewardHistory: [],
    trainingHistory: {
      segments: [{
        id: "ppo",
        kind: "ppo",
        collections: [],
        updates: [{ step: 1, meanLoss: null, policyLoss: null, valueLoss: null }],
      }],
    },
  });
  assert.deepEqual(validateScenarioSnapshot(broken, "running", "run"), [
    "evaluation.passed is not true",
    "renderVerified is not true",
    "restored reward history is empty",
    "ppo has no persisted reward samples",
    "ppo has no persisted loss samples",
  ]);
});

test("readiness-only verifies all routes without requiring saved runs", async () => {
  const report = await verifyLabReady(
    { baseUrl: "http://studio.test", readinessOnly: true },
    { fetchImpl: fixtureFetch({ runs: [] }) }
  );
  assert.equal(report.passed, true);
  assert.equal(report.checks.runsApi.count, 0);
  assert.deepEqual(Object.keys(report.scenarios), scenarios);
});

test("missing Backflip evidence keeps strict restart readiness failed until trained", async () => {
  const directory = await mkdtemp(path.join(os.tmpdir(), "lab-ready-"));
  const reportPath = path.join(directory, "nested", "report.json");
  const runs = scenarios
    .filter((scenario) => scenario !== "backflip")
    .map((scenario) => ({
      experimentId: scenario,
      runName: `${scenario}-accepted`,
      modifiedAt: "2026-09-08T12:00:00.000Z",
      taskPassed: true,
    }));
  try {
    await assert.rejects(
      main(
        ["--base-url", "http://studio.test", "--report", reportPath],
        { fetchImpl: fixtureFetch({ runs }) }
      ),
      /readiness failed/
    );
    const report = JSON.parse(await readFile(reportPath, "utf8"));
    assert.equal(report.passed, false);
    assert.match(report.scenarios.backflip.error, /No saved accepted backflip run/);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("basic MP4 inspection rejects non-ISO media bytes", () => {
  assert.equal(inspectMp4Bytes(mp4Bytes()).passed, true);
  assert.match(inspectMp4Bytes(Buffer.from("not an mp4 file")).error, /ftyp/);
});
