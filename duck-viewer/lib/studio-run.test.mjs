import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("studio-run.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
});
const requireLocal = (specifier) => {
  assert.equal(specifier, "./static-assets");
  return { IS_STATIC_EXPORT: false, publicAssetUrl: (path) => path };
};
const exports = {};
vm.runInNewContext(outputText, { exports, require: requireLocal });
const { availableProfileRunName } = exports;
const { latestDiagnosticRun, latestReviewRun, latestVerifiedRun } = exports;

test("policy refresh scans without cache and sorts training/export timestamps rather than evaluation dates", async () => {
  const controller = new AbortController();
  const catalog = [
    { runName: "old-training", trainedAt: "2026-09-01T00:00:00Z", modifiedAt: "2026-09-08T00:00:00Z" },
    { runName: "latest-training", trainedAt: "2026-09-07T00:00:00Z" },
    { runName: "latest-export", policyModifiedAt: "2026-09-08T00:00:00Z", onnx: true, drawingTool: "brush" },
  ];
  const client = {};
  vm.runInNewContext(outputText, {
    exports: client,
    require: requireLocal,
    fetch: async (url, options) => {
      assert.equal(url, "/api/rlx/runs");
      assert.equal(options.cache, "no-store");
      assert.equal(options.signal, controller.signal);
      return Response.json({ runs: catalog });
    },
  });
  const runs = await client.fetchSavedRuns(controller.signal);
  assert.deepEqual(Array.from(runs, (run) => run.runName), ["latest-export", "latest-training", "old-training"]);
  assert.equal(runs[0].drawingTool, "brush");
});

test("Studio policy refresh exposes failures for retry", async () => {
  const client = {};
  vm.runInNewContext(outputText, {
    exports: client,
    require: requireLocal,
    fetch: async () => new Response(null, { status: 500 }),
  });
  await assert.rejects(client.fetchSavedRuns(), /Saved run catalog unavailable/);
});

function previewRun(runName, trainedAt, overrides = {}) {
  return { experimentId: "dance", runName, trainedAt, modifiedAt: trainedAt,
    checkpoint: true, taskPassed: true, skillAssessed: true, video: true,
    renderVerified: true, renderEvidenceId: "matched-video-hash", ...overrides };
}

test("previews choose the newest successful training, not the newest evaluation", () => {
  const earlier = previewRun("earlier", "2026-09-07T00:00:00Z", { modifiedAt: "2026-09-09T00:00:00Z" });
  const later = previewRun("later", "2026-09-08T00:00:00Z");
  const runs = [earlier, later];
  assert.equal(latestVerifiedRun(runs, "dance").runName, "later");
  assert.equal(runs[0], earlier);
});

test("previews exclude failed, stale, missing and unmatched artifacts", () => {
  const accepted = previewRun("accepted", "2026-09-07T00:00:00Z");
  for (const invalid of [{ taskPassed: false }, { checkpoint: false }, { video: false },
    { renderVerified: false }, { renderEvidenceId: null }, { trainedAt: "bad date" }, { experimentId: "swing" }]) {
    assert.equal(latestVerifiedRun([previewRun("rejected", "2026-09-08T00:00:00Z", invalid), accepted], "dance").runName, "accepted");
  }
});

test("each scenario selects its own verified rollout; missing evidence has no fallback", () => {
  const scenarios = ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge", "drawing"];
  const runs = scenarios.map((experimentId) => previewRun(`${experimentId}-trained`, "2026-09-08T00:00:00Z", { experimentId }));
  for (const experimentId of scenarios) assert.equal(latestVerifiedRun(runs, experimentId).runName, `${experimentId}-trained`);
  assert.equal(latestVerifiedRun([], "dance"), undefined);
  assert.equal(latestVerifiedRun([previewRun("unverified", "2026-09-08T00:00:00Z", { renderVerified: false })], "dance"), undefined);
});

test("completed smoke gets a separate Full run for every scenario", () => {
  for (const scenario of ["dance", "swing", "running", "stilts", "backflip", "basketball", "bridge", "drawing"]) {
    const run = `${scenario}-studio`;
    assert.equal(availableProfileRunName(run, "full", [run]), `${run}-full`);
  }
});

test("Basketball balance-only evidence is previewable but never generalizes to other tasks", () => {
  const basketball = previewRun("basketball-balance-01", null, {
    experimentId: "basketball",
    taskPassed: false,
    balanceOnly: true,
    checkpoint: false,
    onnx: true,
    policyModifiedAt: "2026-09-10T20:00:00Z",
  });
  assert.equal(latestVerifiedRun([basketball], "basketball").runName, "basketball-balance-01");
  assert.equal(latestVerifiedRun([basketball], "bridge"), undefined);
  assert.equal(latestVerifiedRun([{ ...basketball, balanceOnly: false }], "basketball"), undefined);
});

test("source-bound failed Bridge pilots are diagnostic previews but never verified previews", () => {
  const pilot = previewRun("bridge-studio-02", "2026-09-10T21:00:00Z", {
    experimentId: "bridge",
    taskPassed: false,
    skillAssessed: true,
    video: true,
    renderVerified: true,
    renderEvidenceId: "bridge-failed-render",
  });
  assert.equal(latestReviewRun([pilot], "bridge").runName, "bridge-studio-02");
  assert.equal(latestDiagnosticRun([pilot], "bridge").runName, "bridge-studio-02");
  assert.equal(latestVerifiedRun([pilot], "bridge"), undefined);
  assert.equal(latestDiagnosticRun([{ ...pilot, renderVerified: false }], "bridge"), undefined);
  assert.equal(latestDiagnosticRun([{ ...pilot, renderEvidenceId: null }], "bridge"), undefined);
  assert.equal(latestDiagnosticRun([pilot], "basketball"), undefined);
});

test("source-bound failed Drawing pilots remain diagnostic until accepted", () => {
  const pilot = previewRun("drawing-pilot-01", "2026-09-11T12:00:00Z", {
    experimentId: "drawing",
    taskPassed: false,
    skillAssessed: true,
    renderEvidenceId: "drawing-failed-render",
  });
  assert.equal(latestReviewRun([pilot], "drawing").runName, "drawing-pilot-01");
  assert.equal(latestDiagnosticRun([pilot], "drawing").runName, "drawing-pilot-01");
  assert.equal(latestVerifiedRun([pilot], "drawing"), undefined);
});

test("Drawing preview selection preserves the saved tool contract", () => {
  const pencil = previewRun("legacy-pencil", "2026-09-10T12:00:00Z", {
    experimentId: "drawing",
    drawingTool: "pencil",
  });
  const brush = previewRun("current-brush", "2026-09-11T12:00:00Z", {
    experimentId: "drawing",
    drawingTool: "brush",
  });
  assert.equal(latestVerifiedRun([pencil, brush], "drawing").drawingTool, "brush");
  assert.equal(latestReviewRun([pencil], "drawing").drawingTool, "pencil");
});

test("unoccupied custom names are preserved", () => {
  assert.equal(availableProfileRunName("my-custom-run", "full", ["dance-studio"]), "my-custom-run");
});

test("repeated presets avoid existing runs without chaining suffixes", () => {
  const reserved = ["dance-studio", "dance-studio-full", "dance-studio-full-2"];
  assert.equal(availableProfileRunName("dance-studio-full", "full", reserved), "dance-studio-full-3");
  assert.equal(availableProfileRunName("dance-studio-full", "smoke", reserved), "dance-studio-smoke");
});

test("collision checks match API canonicalization and the 48-character limit", () => {
  assert.equal(availableProfileRunName("Dance Studio", "full", ["dance-studio"]), "dance-studio-full");
  const long = "a".repeat(48);
  const full = `${"a".repeat(43)}-full`;
  const next = availableProfileRunName(long, "full", [long, full]);
  assert.equal(next.length, 48);
  assert.match(next, /-full-2$/);
});
