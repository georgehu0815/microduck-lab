import assert from "node:assert/strict";
import { writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ARM_BACKEND = "http://127.0.0.1:8812";
const UI_BASE_URL = process.env.ARM_UI_BASE_URL ?? "http://127.0.0.1:63318";
const PLAYWRIGHT_MODULE = process.env.ARM_PLAYWRIGHT_MODULE ?? "playwright";
const here = path.dirname(fileURLToPath(import.meta.url));
const artifactPath = process.env.ARM_E2E_ARTIFACT ??
  path.join(here, "live-e2e-latest.json");
const desktopScreenshotPath = process.env.ARM_E2E_DESKTOP_SCREENSHOT ??
  path.join(here, "live-e2e-desktop.png");
const mobileScreenshotPath = process.env.ARM_E2E_MOBILE_SCREENSHOT ??
  path.join(here, "live-e2e-mobile.png");
const evidenceScreenshotPath = process.env.ARM_E2E_EVIDENCE_SCREENSHOT ??
  path.join(here, "live-e2e-evidence.png");
const CASES = [
  { name: /末端触达/, id: "arm-reach-v1", actionDim: 6 },
  { name: /抓取与放置/, id: "arm-pick-place-v1", actionDim: 6 },
  { name: /绕障移物/, id: "arm-relocate-v1", actionDim: 6 },
  { name: /定点搬运/, id: "arm-carry-v1", actionDim: 6 },
  { name: /双臂交接/, id: "arms-handover-v1", actionDim: 12 },
  { name: /协作搬托盘/, id: "arms-co-carry-v1", actionDim: 12 },
];

async function loadPlaywright() {
  try {
    const specifier = PLAYWRIGHT_MODULE.startsWith("/")
      ? pathToFileURL(PLAYWRIGHT_MODULE).href
      : PLAYWRIGHT_MODULE;
    return await import(specifier);
  } catch (error) {
    throw new Error(
      "Playwright is required for the live Arm UI audit. Set " +
      "ARM_PLAYWRIGHT_MODULE to an installed playwright/index.mjs path.",
      { cause: error }
    );
  }
}

async function readJson(response, label) {
  const text = await response.text();
  const ok = typeof response.ok === "function" ? response.ok() : response.ok;
  const status = typeof response.status === "function"
    ? response.status()
    : response.status;
  assert.ok(ok, `${label} returned ${status}: ${text}`);
  return JSON.parse(text);
}

async function readStep(page) {
  const value = await page
    .getByText("step", { exact: true })
    .locator("..")
    .locator("strong")
    .textContent();
  return Number(value);
}

const health = await readJson(await fetch(`${ARM_BACKEND}/health`), "Arm /health");
const runCatalog = await readJson(
  await fetch(`${ARM_BACKEND}/runs`),
  "Arm /runs"
);
const strictPpoRuns = runCatalog.runs.filter((run) =>
  run.controller === "ppo_residual" &&
  run.metrics?.episodes === 100 &&
  run.metrics?.sources_match === true &&
  run.metrics?.model_match === true &&
  run.metrics?.checkpoint_match === true &&
  run.metrics?.context_match === true
);
const catalogVideos = runCatalog.runs.filter((run) => run.video_path);
assert.equal(strictPpoRuns.length, 18, "Expected 18 strict 100-episode PPO evaluations.");
assert.equal(catalogVideos.length, 6, "Expected one real MP4 for each Arm case.");
assert.deepEqual(
  new Set(catalogVideos.map((run) => run.case_id)),
  new Set(CASES.map((entry) => entry.id))
);
const playwright = await loadPlaywright();
const browser = await playwright.chromium.launch({ headless: true });
const context = await browser.newContext();
const page = await context.newPage();
await page.setViewportSize({ width: 1440, height: 1000 });
const browserErrors = [];
const browserWarnings = [];
const offlineBrowserErrors = [];
let testingOffline = false;

page.on("console", (message) => {
  if (
    message.type() === "warning" &&
    /unknown color|color.*parse|parse.*color/i.test(message.text())
  ) {
    browserWarnings.push(message.text());
  }
  if (message.type() !== "error") return;
  if (testingOffline) offlineBrowserErrors.push(message.text());
  else browserErrors.push(message.text());
});
page.on("pageerror", (error) => {
  if (testingOffline) offlineBrowserErrors.push(error.message);
  else browserErrors.push(error.message);
});

const cases = [];
let initial = null;
let mobileSummary = null;
let offline = null;
let videos = [];
let failure = null;

try {
  await page.goto(`${UI_BASE_URL}/arm`, { waitUntil: "networkidle" });
  await page.getByText("仿真在线", { exact: true }).waitFor();

  initial = await page.evaluate(() => ({
    title: document.querySelector("h1")?.textContent?.trim(),
    caseCount: document.querySelectorAll("[aria-pressed]").length,
    sliderCount: document.querySelectorAll('input[type="range"]').length,
    hardwareLocked: document.body.textContent?.includes("HARDWARE · LOCKED"),
    unixTransportNotIntegrated: document.body.textContent?.includes(
      "Unix transport not integrated"
    ),
    teacherNotPpo: document.body.textContent?.includes("NOT PPO"),
    unavailable: document.body.textContent?.includes("Arm backend unavailable"),
    horizontalOverflow:
      document.documentElement.scrollWidth > document.documentElement.clientWidth,
  }));

  assert.equal(initial.caseCount, 6);
  assert.equal(initial.sliderCount, 6);
  assert.equal(initial.hardwareLocked, true);
  assert.equal(initial.unixTransportNotIntegrated, true);
  assert.equal(initial.teacherNotPpo, true);
  assert.equal(initial.unavailable, false);
  assert.equal(initial.horizontalOverflow, false);
  assert.equal(
    await page.locator("tbody tr").count(),
    4,
    "Default evidence view should show four current matching runs."
  );
  assert.equal(
    await page.getByRole("checkbox", { name: "显示历史/其他案例" }).isChecked(),
    false
  );
  assert.equal(await page.locator("details[open]").count(), 0);
  await page.getByRole("checkbox", { name: "显示历史/其他案例" }).check();
  assert.equal(
    await page.locator("tbody tr").count(),
    runCatalog.runs.length,
    "History checkbox should reveal every backend run."
  );
  const unattestedBcReach = page.locator("tbody tr")
    .filter({ hasText: "arm-reach-v1" })
    .filter({ hasText: "BC · bc" });
  assert.equal(await unattestedBcReach.count(), 1);
  assert.ok((await unattestedBcReach.textContent())?.includes("100/100"));
  assert.ok((await unattestedBcReach.textContent())?.includes("证据未核实"));
  assert.ok(!(await unattestedBcReach.textContent())?.includes("未通过"));
  assert.ok((await unattestedBcReach.textContent())?.includes("PROVENANCE UNMATCHED"));
  await page.getByRole("checkbox", { name: "显示历史/其他案例" }).uncheck();
  assert.equal(await page.locator("tbody tr").count(), 4);

  const firstRunDetails = page.locator("tbody details").first();
  await firstRunDetails.locator("summary").click();
  assert.equal(await firstRunDetails.locator("small").count(), 1);
  assert.ok((await firstRunDetails.textContent())?.includes("episodes"));
  await firstRunDetails.locator("summary").click();

  for (const testCase of CASES) {
    await page.getByRole("button", { name: testCase.name }).click();
    assert.equal(
      await page.locator('input[type="range"]').count(),
      testCase.actionDim
    );
    const resetResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/arm/reset") &&
        response.request().method() === "POST"
    );
    await page.getByRole("button", { name: /重置所选案例/ }).click();
    const resetState = await readJson(await resetResponse, `UI reset ${testCase.id}`);
    assert.equal(resetState.case_id, testCase.id);
    assert.equal(resetState.action_dim, testCase.actionDim);
    const resetStep = await readStep(page);

    const stepResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/arm/step") &&
        response.request().method() === "POST"
    );
    await page.getByRole("button", { name: /单步执行/ }).click();
    const manualState = await readJson(
      await stepResponse,
      `UI manual step ${testCase.id}`
    );
    const manualStep = await readStep(page);
    assert.equal(manualState.case_id, testCase.id);
    assert.ok(manualStep > resetStep, `${testCase.id} manual step did not advance.`);

    const ticksInput = page.locator('input[type="number"]').nth(1);
    await ticksInput.fill("1");
    const teacherResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/arm/teacher") &&
        response.request().method() === "POST"
    );
    await page.getByRole("button", { name: /推进 Teacher/ }).click();
    const teacherState = await readJson(
      await teacherResponse,
      `UI Teacher step ${testCase.id}`
    );
    const teacherStep = await readStep(page);
    assert.equal(teacherState.case_id, testCase.id);
    assert.ok(teacherStep > manualStep, `${testCase.id} Teacher step did not advance.`);

    const autoplayResponse = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/arm/teacher") &&
        response.request().method() === "POST"
    );
    await page.getByRole("button", { name: /连续播放/ }).click();
    const autoplayState = await readJson(
      await autoplayResponse,
      `UI Teacher autoplay ${testCase.id}`
    );
    await page.getByRole("button", { name: "暂停", exact: true }).click();
    const autoplayStep = await readStep(page);
    assert.equal(autoplayState.case_id, testCase.id);
    assert.ok(
      autoplayStep > teacherStep,
      `${testCase.id} Teacher autoplay did not advance.`
    );

    cases.push({
      case_id: resetState.case_id,
      observation_dim: resetState.observation_dim,
      action_dim: resetState.action_dim,
      geom_count: resetState.geoms.length,
      reset: {
        step: resetStep,
        controller: resetState.controller,
        stage: resetState.stage,
      },
      manual: {
        step: manualStep,
        controller: manualState.controller,
      },
      teacher: {
        step: teacherStep,
        controller: teacherState.controller,
      },
      teacher_autoplay: {
        step: autoplayStep,
        controller: autoplayState.controller,
      },
    });
  }

  await page.getByRole("button", { name: /末端触达/ }).click();
  const reachReset = page.waitForResponse(
    (response) =>
      response.url().endsWith("/api/arm/reset") &&
      response.request().method() === "POST"
  );
  await page.getByRole("button", { name: /重置所选案例/ }).click();
  await readJson(await reachReset, "UI reset arm-reach-v1");

  for (const run of catalogVideos) {
    const testCase = CASES.find((entry) => entry.id === run.case_id);
    assert.ok(testCase, `Unknown video case ${run.case_id}`);
    await page.getByRole("button", { name: testCase.name }).click();
    const artifactUrl =
      `${UI_BASE_URL}/api/arm/artifact?path=${encodeURIComponent(run.video_path)}`;
    const rangeResponse = await context.request.get(artifactUrl, {
      headers: { Range: "bytes=0-1023" },
    });
    const rangeBody = await rangeResponse.body();
    assert.equal(rangeResponse.status(), 206, `${run.case_id} range status`);
    assert.match(
      rangeResponse.headers()["content-type"] ?? "",
      /^video\/mp4\b/,
      `${run.case_id} content type`
    );
    assert.match(
      rangeResponse.headers()["content-range"] ?? "",
      /^bytes 0-1023\/\d+$/,
      `${run.case_id} content range`
    );
    assert.equal(rangeBody.byteLength, 1024, `${run.case_id} range bytes`);

    const row = page.locator("tbody tr")
      .filter({ hasText: run.case_id })
      .filter({ hasText: run.run_id })
      .filter({ has: page.getByRole("button", { name: /查看/ }) });
    assert.equal(await row.count(), 1, `${run.case_id} video row count`);
    await row.getByRole("button", { name: /查看/ }).click();
    const video = page.locator("video");
    await video.waitFor();
    await video.evaluate((element) => {
      if (element.readyState >= HTMLMediaElement.HAVE_METADATA) return;
      return new Promise((resolve, reject) => {
        const timeout = window.setTimeout(
          () => reject(new Error("Timed out loading MP4 metadata.")),
          10_000
        );
        element.addEventListener("loadedmetadata", () => {
          window.clearTimeout(timeout);
          resolve(undefined);
        }, { once: true });
        element.addEventListener("error", () => {
          window.clearTimeout(timeout);
          reject(new Error(element.error?.message || "MP4 metadata failed."));
        }, { once: true });
        element.load();
      });
    });
    const metadata = await video.evaluate(async (element) => {
      await element.play();
      await new Promise((resolve) => window.setTimeout(resolve, 250));
      element.pause();
      return {
        currentTime: element.currentTime,
        duration: element.duration,
        readyState: element.readyState,
        videoWidth: element.videoWidth,
        videoHeight: element.videoHeight,
        src: element.currentSrc,
      };
    });
    assert.ok(Number.isFinite(metadata.duration) && metadata.duration > 0);
    assert.ok(metadata.readyState >= 1);
    assert.ok(metadata.videoWidth > 0 && metadata.videoHeight > 0);
    assert.ok(metadata.currentTime > 0, `${run.case_id} MP4 did not play.`);
    videos.push({
      case_id: run.case_id,
      run_id: run.run_id,
      video_path: run.video_path,
      range: {
        status: rangeResponse.status(),
        content_type: rangeResponse.headers()["content-type"],
        content_range: rangeResponse.headers()["content-range"],
        bytes: rangeBody.byteLength,
      },
      metadata,
    });
    await page.getByRole("button", { name: "关闭视频" }).click();
  }

  assert.equal(videos.length, 6);
  await page.getByRole("button", { name: /末端触达/ }).click();
  await page.locator("#evidence").scrollIntoViewIfNeeded();
  await page.screenshot({ path: evidenceScreenshotPath });
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: desktopScreenshotPath });

  const mobile = await context.newPage();
  await mobile.setViewportSize({ width: 390, height: 900 });
  await mobile.goto(`${UI_BASE_URL}/arm`, { waitUntil: "networkidle" });
  await mobile.getByText("仿真在线", { exact: true }).waitFor();
  mobileSummary = await mobile.evaluate(() => ({
    caseCount: document.querySelectorAll("[aria-pressed]").length,
    sliderCount: document.querySelectorAll('input[type="range"]').length,
    hardwareLocked: document.body.textContent?.includes("HARDWARE · LOCKED"),
    horizontalOverflow:
      document.documentElement.scrollWidth > document.documentElement.clientWidth,
  }));
  assert.equal(mobileSummary.caseCount, 6);
  assert.equal(mobileSummary.hardwareLocked, true);
  assert.equal(mobileSummary.horizontalOverflow, false);
  await mobile.evaluate(() => window.scrollTo(0, 0));
  await mobile.screenshot({ path: mobileScreenshotPath });
  await mobile.close();

  testingOffline = true;
  await context.setOffline(true);
  await page.getByRole("button", { name: "重新连接" }).click();
  await page.getByText("Arm backend unavailable", { exact: true }).waitFor();
  offline = await page.evaluate(() => ({
    unavailable: document.body.textContent?.includes("Arm backend unavailable"),
    noTelemetry: document.body.textContent?.includes("没有实时几何"),
    noInventedRuns: document.body.textContent?.includes(
      "没有使用示例数据或虚构成功历史"
    ),
  }));
  assert.deepEqual(offline, {
    unavailable: true,
    noTelemetry: true,
    noInventedRuns: true,
  });
  await context.setOffline(false);
  await page.getByRole("button", { name: "重新连接" }).click();
  await page.getByText("仿真在线", { exact: true }).waitFor();
  testingOffline = false;

  const result = {
    passed: true,
    checked_at: new Date().toISOString(),
    backend: ARM_BACKEND,
    ui: `${UI_BASE_URL}/arm`,
    health,
    initial,
    cases,
    run_catalog: {
      total: runCatalog.runs.length,
      strict_ppo_100_episode_count: strictPpoRuns.length,
      strict_ppo_by_case: Object.fromEntries(CASES.map(({ id }) => [
        id,
        strictPpoRuns
          .filter((run) => run.case_id === id)
          .map((run) => ({
            run_id: run.run_id,
            passed: run.passed,
            successes: run.metrics.successes,
            episodes: run.metrics.episodes,
            success_rate: run.metrics.success_rate,
          })),
      ])),
      diagnostic_or_failed_count: runCatalog.runs.filter(
        (run) => run.passed !== true
      ).length,
      videos: videos.length,
    },
    videos,
    mobile: mobileSummary,
    offline,
    offline_browser_errors: offlineBrowserErrors,
    browser_errors: browserErrors,
    browser_warnings: browserWarnings,
    screenshots: {
      desktop: desktopScreenshotPath,
      mobile: mobileScreenshotPath,
      evidence: evidenceScreenshotPath,
    },
    mocks: false,
  };
  assert.deepEqual(browserErrors, []);
  assert.deepEqual(browserWarnings, []);
  await writeFile(artifactPath, `${JSON.stringify(result, null, 2)}\n`);
  console.log(JSON.stringify(result, null, 2));
} catch (error) {
  failure = error;
  testingOffline = false;
  const result = {
    passed: false,
    checked_at: new Date().toISOString(),
    backend: ARM_BACKEND,
    ui: `${UI_BASE_URL}/arm`,
    health,
    initial,
    cases,
    run_catalog: {
      total: runCatalog.runs.length,
      strict_ppo_100_episode_count: strictPpoRuns.length,
      videos: catalogVideos.length,
    },
    videos,
    mobile: mobileSummary,
    offline,
    offline_browser_errors: offlineBrowserErrors,
    browser_errors: browserErrors,
    browser_warnings: browserWarnings,
    screenshots: {
      desktop: desktopScreenshotPath,
      mobile: mobileScreenshotPath,
      evidence: evidenceScreenshotPath,
    },
    mocks: false,
    error: error instanceof Error ? error.message : String(error),
  };
  await writeFile(artifactPath, `${JSON.stringify(result, null, 2)}\n`);
  console.error(JSON.stringify(result, null, 2));
} finally {
  await browser.close();
}

if (failure) throw failure;
