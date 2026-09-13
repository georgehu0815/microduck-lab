import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import vm from "node:vm";
import ts from "typescript";

const source = readFileSync(new URL("./wingpod.ts", import.meta.url), "utf8");
const { outputText } = ts.transpileModule(source, {
  compilerOptions: {
    module: ts.ModuleKind.CommonJS,
    target: ts.ScriptTarget.ES2022,
  },
});
const exports = {};
vm.runInNewContext(outputText, {
  exports,
  require(specifier) {
    assert.equal(specifier, "./static-assets");
    return { publicAssetUrl: (path) => path };
  },
});

const {
  WINGPOD_DOWNLOADS,
  WINGPOD_IMAGES,
  WINGPOD_SCOPE_BADGES,
  WINGPOD_VIDEOS,
  createWingPodMediaModel,
  filterWingPodCases,
  parseWingPodCasesManifest,
  wingPodAssetUrl,
  wingPodCaseAssetUrl,
} = exports;

test("WingPod video chooser exposes two distinct, downloadable MP4 evidence records", () => {
  assert.deepEqual(
    Array.from(WINGPOD_VIDEOS, ({ id, durationSeconds }) => [id, durationSeconds]),
    [["appearance", 20], ["tennis", 50.84]],
  );
  assert.equal(new Set(Array.from(WINGPOD_VIDEOS, ({ file }) => file)).size, 2);
  for (const video of WINGPOD_VIDEOS) {
    assert.match(video.file, /\.mp4$/);
    assert.ok(video.durationSeconds > 0);
    assert.ok(video.title.en && video.title.zh);
    assert.ok(video.summary.en && video.summary.zh);
  }
});

test("media model applies the static base path to video and poster and falls back safely", () => {
  const withPagesPrefix = (path) => `/microduck-lab${path}`;
  const tennis = createWingPodMediaModel("tennis", withPagesPrefix);
  assert.equal(tennis.videoSrc, "/microduck-lab/static/wingpod/tennis-return.mp4");
  assert.equal(tennis.posterSrc, "/microduck-lab/static/wingpod/sequence-contact-sheet-soft-956664897334.jpg");
  assert.equal(tennis.downloadName, "tennis-return.mp4");

  const fallback = createWingPodMediaModel("missing", withPagesPrefix);
  assert.equal(fallback.selected.id, "appearance");
  assert.equal(fallback.videoSrc, "/microduck-lab/static/wingpod/wingpod-camera-eyes.mp4");
  assert.equal(wingPodAssetUrl("BOM.csv", withPagesPrefix), "/microduck-lab/static/wingpod/BOM.csv");
});

test("hero, detail and poster use source-matched image URLs instead of cached legacy URLs", () => {
  const manifest = JSON.parse(readFileSync(new URL("../public/static/wingpod/manifest.json", import.meta.url)));
  const standalone = readFileSync(new URL("../public/wingpod-v2/index.html", import.meta.url), "utf8");
  for (const [original, versioned] of Object.entries(WINGPOD_IMAGES)) {
    const entry = manifest.files[versioned];
    assert.ok(entry, `${versioned} must be published`);
    assert.match(entry.source, /wingpod-camera-v2-soft\//);
    const bytes = readFileSync(new URL(`../public/static/wingpod/${versioned}`, import.meta.url));
    const hash = createHash("sha256").update(bytes).digest("hex");
    assert.equal(hash, entry.sha256);
    assert.ok(versioned.includes(hash.slice(0, 12)));
    assert.equal(wingPodAssetUrl(original, (path) => `/microduck-lab${path}`), `/microduck-lab/static/wingpod/${versioned}`);
    assert.ok(standalone.includes(`"${original}": "${versioned}"`));
  }
});

test("scope and BOM contracts preserve evidence boundaries without inventing inventory counts", () => {
  assert.deepEqual(
    Array.from(WINGPOD_SCOPE_BADGES, ({ id }) => id),
    ["simulation", "no-vision", "not-release"],
  );
  assert.deepEqual(
    Array.from(WINGPOD_DOWNLOADS, ({ format }) => format),
    ["CSV", "JSON", "Markdown"],
  );
  assert.equal(
    WINGPOD_DOWNLOADS.some((entry) => /\d+\s*(?:rows?|items?|parts?)/i.test(entry.label.en)),
    false,
  );
});

test("case manifest parsing validates evidence fields and filters positive cases separately from controls", () => {
  const manifest = parseWingPodCasesManifest({
    cases: [
      {
        id: "nominal-0",
        variant: "nominal",
        seed: 0,
        kind: "positive",
        taskSuccess: true,
        expectedOutcomeMatched: true,
        video: "cases/nominal-0.mp4",
        receipt: "cases/nominal-0.json",
        durationSeconds: 7.2,
        frames: 360,
        zeroError: true,
      },
      {
        id: "control-no-contact",
        variant: "control",
        seed: 31,
        kind: "negative_control",
        taskSuccess: false,
        expectedOutcomeMatched: true,
        video: "cases/control-no-contact.mp4",
        receipt: "cases/control-no-contact.json",
        durationSeconds: 6,
        frames: 300,
        zeroError: true,
      },
    ],
    summary: {
      positivePassed: 1,
      positiveTotal: 1,
      controlsMatched: 1,
      controlsTotal: 1,
    },
  });
  assert.equal(filterWingPodCases(manifest.cases, "nominal", "positive").length, 1);
  assert.equal(filterWingPodCases(manifest.cases, "all", "negative_control")[0].taskSuccess, false);
  assert.equal(manifest.summary.controlsMatched, 1);
});

test("case asset paths remain under the WingPod static directory", () => {
  const withPagesPrefix = (path) => `/microduck-lab${path}`;
  assert.equal(
    wingPodCaseAssetUrl("cases/nominal-0.mp4", withPagesPrefix),
    "/microduck-lab/static/wingpod/cases/nominal-0.mp4",
  );
  for (const unsafe of ["../secret.mp4", "/cases/absolute.mp4", "https://example.com/video.mp4", "cases/%2e%2e/secret.mp4", "cases\\secret.mp4", "cases//test.mp4"]) {
    assert.throws(() => wingPodCaseAssetUrl(unsafe, withPagesPrefix), /Unsafe/);
  }
  assert.throws(
    () => parseWingPodCasesManifest({ cases: [], summary: { positivePassed: 2, positiveTotal: 1, controlsMatched: 0, controlsTotal: 0 } }),
    /summary/,
  );
});

test("case summaries cannot upgrade missing or mismatched evidence to all-passed", () => {
  const entry = { id: "nominal-0", variant: "nominal", seed: 0, kind: "positive",
    taskSuccess: true, expectedOutcomeMatched: true, video: "cases/nominal-0.mp4",
    receipt: "cases/nominal-0.json", durationSeconds: 1, frames: 25, zeroError: true };
  const summary = { positivePassed: 1, positiveTotal: 1, controlsMatched: 0, controlsTotal: 0 };
  assert.throws(() => parseWingPodCasesManifest({ cases: [entry], summary: { ...summary, positiveTotal: 30, positivePassed: 30 } }), /does not match/);
  assert.throws(() => parseWingPodCasesManifest({ cases: [entry, entry], summary }), /does not match/);
  assert.throws(() => parseWingPodCasesManifest({ cases: [{ ...entry, zeroError: false }], summary }), /does not match/);
  assert.throws(() => parseWingPodCasesManifest({ cases: [{ ...entry, expectedOutcomeMatched: false }], summary }), /does not match/);
});
