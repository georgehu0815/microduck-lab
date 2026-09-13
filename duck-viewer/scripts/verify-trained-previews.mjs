import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium } = createRequire(
  process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json"
)("playwright");

const baseUrl = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const output = path.resolve(
  process.env.STUDIO_EVIDENCE_DIR ||
    "../rlx/artifacts/trained-previews-20260908/browser"
);
const scenarios = [
  { id: "dance", title: "Dance imitation", shortTitle: "Dance", preview: "task" },
  { id: "swing", title: "Self-pumped swing", shortTitle: "Swing", preview: "task" },
  { id: "running", title: "Fast running", shortTitle: "Running", preview: "task" },
  { id: "stilts", title: "Stilt walking", shortTitle: "Stilts", preview: "task" },
  { id: "backflip", title: "Backflip showcase", shortTitle: "Backflip", preview: "task" },
  { id: "basketball", title: "Basketball balancing", shortTitle: "Basketball", preview: "balance" },
  { id: "bridge", title: "Suspended bridge", shortTitle: "Bridge", preview: "diagnostic" },
];

await mkdir(output, { recursive: true });

function isEligible(run, scenario) {
  const accepted = run?.taskPassed === true ||
    (scenario?.preview === "balance" && run?.balanceOnly === true) ||
    (scenario?.preview === "diagnostic" && run?.taskPassed === false && run?.skillAssessed === true);
  return accepted &&
    run.video === true &&
    (run.checkpoint === true || (scenario.id === "basketball" && run.onnx === true)) &&
    run.renderVerified === true &&
    typeof run.renderEvidenceId === "string" &&
    run.renderEvidenceId.length > 0;
}

function trainedTime(run) {
  return Date.parse(run.trainedAt || run.modifiedAt || "");
}

function newestEligible(runs, experimentId) {
  const scenario = scenarios.find((candidate) => candidate.id === experimentId);
  return runs
    .filter((run) => run.experimentId === experimentId && isEligible(run, scenario))
    .sort((left, right) => trainedTime(right) - trainedTime(left))[0] ?? null;
}

async function fetchCatalog() {
  const response = await fetch(new URL("/api/rlx/runs", baseUrl), {
    cache: "no-store",
    signal: AbortSignal.timeout(30_000),
  });
  assert.equal(response.status, 200, "/api/rlx/runs must be available");
  const payload = await response.json();
  assert.ok(Array.isArray(payload.runs), "/api/rlx/runs must return a runs array");
  return payload.runs;
}

async function installRoutes(page, runs = null) {
  await page.route("**/api/rlx**", async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    if (request.method() === "POST") {
      await route.fulfill({
        status: 409,
        contentType: "application/json",
        body: JSON.stringify({
          error: "Browser regression intercepted POST; no RLX job was launched.",
        }),
      });
      return;
    }
    if (runs && url.pathname === "/api/rlx/runs") {
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ runs }),
      });
      return;
    }
    await route.continue();
  });
}

async function experimentRow(page, title) {
  const row = page.getByRole("radio").filter({ hasText: title });
  await row.waitFor();
  return row;
}

function assertRunArtifactUrl(href, experimentId, runName, kind) {
  const url = new URL(href, baseUrl);
  assert.equal(url.pathname, "/api/rlx/artifact");
  assert.equal(url.searchParams.get("experiment"), experimentId);
  assert.equal(url.searchParams.get("run"), runName);
  assert.equal(url.searchParams.get("kind"), kind);
  if (kind === "video") assert.equal(url.searchParams.get("inline"), "1");
}

function assertEvaluationUrl(href, experimentId, runName) {
  const url = new URL(href, baseUrl);
  assert.equal(url.pathname, "/api/rlx");
  assert.equal(url.searchParams.get("experiment"), experimentId);
  assert.equal(url.searchParams.get("run"), runName);
}

async function verifyPlayback(page, video, scenario, expectedRun) {
  await video.scrollIntoViewIfNeeded();
  await video.waitFor({ state: "visible" });
  await video.evaluate((element) => new Promise((resolve, reject) => {
    if (element.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      resolve();
      return;
    }
    const timeout = window.setTimeout(
      () => reject(new Error(`Timed out loading ${element.currentSrc || element.src}`)),
      30_000
    );
    element.addEventListener("loadeddata", () => {
      window.clearTimeout(timeout);
      resolve();
    }, { once: true });
    element.addEventListener("error", () => {
      window.clearTimeout(timeout);
      reject(new Error(element.error?.message || "Video failed to decode"));
    }, { once: true });
  }));
  const before = await video.evaluate(async (element) => {
    element.currentTime = 0;
    await element.play();
    return element.currentTime;
  });
  await page.waitForFunction(
    ({ title, start }) => {
      const row = [...document.querySelectorAll('[role="radio"]')]
        .find((element) => element.textContent?.includes(title));
      const element = row?.querySelector("video");
      return Boolean(element && element.currentTime > start + 0.2);
    },
    { title: scenario.title, start: before },
    { timeout: 30_000 }
  );
  const playback = await video.evaluate((element) => {
    element.pause();
    return {
      currentTime: element.currentTime,
      duration: element.duration,
      readyState: element.readyState,
      videoWidth: element.videoWidth,
      videoHeight: element.videoHeight,
      src: element.currentSrc || element.src,
      error: element.error?.message ?? null,
    };
  });
  assert.equal(playback.error, null);
  assert.ok(playback.currentTime > before + 0.2, `${scenario.id} video did not advance`);
  assert.ok(Number.isFinite(playback.duration) && playback.duration > 0);
  assert.ok(playback.videoWidth > 0 && playback.videoHeight > 0);
  assertRunArtifactUrl(playback.src, scenario.id, expectedRun, "video");
  return playback;
}

async function inspectExperimentRows(page, expectedByScenario, { playVideos }) {
  const results = [];
  for (const scenario of scenarios) {
    const expected = expectedByScenario.get(scenario.id);
    if (scenario.preview === "none") {
      assert.equal(expected, null, `${scenario.id} must not claim a verified preview`);
      const row = await experimentRow(page, scenario.title);
      await row.getByText("No verified rollout yet", { exact: true }).waitFor();
      assert.equal(await row.locator("video, img").count(), 0);
      results.push({
        experimentId: scenario.id,
        runName: null,
        placeholder: "No verified rollout yet",
      });
      continue;
    }
    assert.ok(expected, `Missing expected eligible ${scenario.id} run`);
    const row = await experimentRow(page, scenario.title);
    if (scenario.preview === "diagnostic") {
      assert.equal(expected.taskPassed, false);
      await row.getByText("TRAINING PILOT · CROSSING FAILED", { exact: true }).waitFor();
    }
    if (scenario.preview === "balance") {
      assert.equal(expected.taskPassed, false);
      await row.getByText("BALANCE ONLY · STEERING NOT PASSED", { exact: true }).waitFor();
    }
    await row.getByText(expected.runName, { exact: true }).waitFor();
    assert.equal(
      await row.getByText(expected.runName, { exact: true }).count(),
      1,
      `${scenario.id} must show one run-name label`
    );

    const video = row.locator("video");
    assert.equal(await video.count(), 1, `${scenario.id} must show one trained video`);
    const source = await video.getAttribute("src");
    assert.ok(source, `${scenario.id} video must have a src`);
    assertRunArtifactUrl(source, scenario.id, expected.runName, "video");

    const evaluation = row.getByRole("link", { name: /Evaluation/i });
    assert.equal(await evaluation.count(), 1, `${scenario.id} must show one Evaluation link`);
    const evaluationHref = await evaluation.getAttribute("href");
    assert.ok(evaluationHref, `${scenario.id} Evaluation link must have an href`);
    assertEvaluationUrl(evaluationHref, scenario.id, expected.runName);

    const playback = playVideos
      ? await verifyPlayback(page, video, scenario, expected.runName)
      : null;
    results.push({
      experimentId: scenario.id,
      runName: expected.runName,
      trainedAt: expected.trainedAt,
      renderEvidenceId: expected.renderEvidenceId,
      video: source,
      evaluation: evaluationHref,
      playback,
    });
  }
  return results;
}

async function layoutEvidence(page) {
  const section = page.locator("#experiments");
  await section.scrollIntoViewIfNeeded();
  const layout = await page.evaluate(() => {
    const section = document.querySelector("#experiments");
    const rows = [...document.querySelectorAll('#experiments [role="radio"]')];
    return {
      viewport: { width: window.innerWidth, height: window.innerHeight },
      document: {
        clientWidth: document.documentElement.clientWidth,
        scrollWidth: document.documentElement.scrollWidth,
      },
      section: section
        ? { clientWidth: section.clientWidth, scrollWidth: section.scrollWidth }
        : null,
      rows: rows.map((row) => ({
        text: row.querySelector("strong")?.textContent ?? "",
        clientWidth: row.clientWidth,
        scrollWidth: row.scrollWidth,
      })),
    };
  });
  assert.ok(layout.section, "#experiments must exist");
  assert.ok(
    layout.document.scrollWidth <= layout.document.clientWidth + 1,
    `Document overflow: ${JSON.stringify(layout.document)}`
  );
  assert.ok(
    layout.section.scrollWidth <= layout.section.clientWidth + 1,
    `#experiments overflow: ${JSON.stringify(layout.section)}`
  );
  for (const row of layout.rows) {
    assert.ok(
      row.scrollWidth <= row.clientWidth + 1,
      `Experiment row overflow: ${JSON.stringify(row)}`
    );
  }
  return layout;
}

function newerIneligibleRuns(expectedByScenario) {
  const additions = [];
  for (const scenario of scenarios) {
    if (scenario.preview === "none") continue;
    const selected = expectedByScenario.get(scenario.id);
    const baseTime = Math.max(
      trainedTime(selected),
      Date.parse("2026-09-08T12:00:00.000Z")
    );
    const failedAt = new Date(baseTime + 60_000).toISOString();
    const unverifiedAt = new Date(baseTime + 120_000).toISOString();
    additions.push(
      {
        ...selected,
        runName: `${scenario.id}-newer-failed`,
        trainedAt: failedAt,
        modifiedAt: failedAt,
        taskPassed: false,
        balanceOnly: false,
        skillAssessed: false,
      },
      {
        ...selected,
        runName: `${scenario.id}-newer-unverified`,
        trainedAt: unverifiedAt,
        modifiedAt: unverifiedAt,
        renderVerified: false,
        renderEvidenceId: null,
      }
    );
  }
  return additions;
}

async function newPage(browser, runs = null, viewport = { width: 1600, height: 1100 }) {
  const context = await browser.newContext({
    viewport,
    deviceScaleFactor: 1,
  });
  const page = await context.newPage();
  page.setDefaultTimeout(15_000);
  await installRoutes(page, runs);
  return { context, page };
}

async function verifyReducedMotion(browser, expectedByScenario) {
  const context = await browser.newContext({
    viewport: { width: 1600, height: 1100 },
    deviceScaleFactor: 1,
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  page.setDefaultTimeout(15_000);
  await installRoutes(page);
  try {
    await page.goto(`${baseUrl}/#experiments`, { waitUntil: "domcontentloaded" });
    const results = [];
    for (const scenario of scenarios) {
      const expected = expectedByScenario.get(scenario.id);
      const row = await experimentRow(page, scenario.title);
      if (scenario.preview === "none") {
        await row.getByText("No verified rollout yet", { exact: true }).waitFor();
        assert.equal(await row.locator("video, img").count(), 0);
        results.push({
          experimentId: scenario.id,
          runName: null,
          placeholder: "No verified rollout yet",
        });
        continue;
      }
      const video = row.locator("video");
      const sheet = row.getByRole("img", {
        name: new RegExp("trained rollout contact sheet$", "i"),
      });
      await video.waitFor({ state: "attached" });
      await sheet.waitFor({ state: "attached" });
      assert.equal(await video.count(), 1);
      assert.equal(
        await video.getAttribute("aria-label"),
        `${scenario.shortTitle} trained rollout: ${expected.runName}`
      );
      assert.equal(await video.isHidden(), true, `${scenario.id} video must be hidden`);
      assert.equal(await sheet.count(), 1);
      assert.equal(await sheet.isVisible(), true, `${scenario.id} contact sheet must be visible`);
      const source = await sheet.getAttribute("src");
      assert.ok(source);
      const url = new URL(source, baseUrl);
      assert.equal(url.pathname, "/api/rlx/artifact");
      assert.equal(url.searchParams.get("experiment"), scenario.id);
      assert.equal(url.searchParams.get("run"), expected.runName);
      assert.equal(url.searchParams.get("kind"), "sheet");
      assert.equal(url.searchParams.get("v"), expected.renderEvidenceId);
      results.push({ experimentId: scenario.id, runName: expected.runName, sheet: source });
    }
    return results;
  } finally {
    await context.close();
  }
}

const liveRuns = await fetchCatalog();
const expectedByScenario = new Map(
  scenarios.map((scenario) => [
    scenario.id,
    newestEligible(liveRuns, scenario.id),
  ])
);
for (const scenario of scenarios) {
  const selected = expectedByScenario.get(scenario.id);
  if (scenario.preview === "none") {
    assert.equal(selected, null, `${scenario.id} must remain unverified`);
    continue;
  }
  assert.ok(selected, `Live catalog has no eligible ${scenario.id} run`);
  assert.ok(
    Number.isFinite(Date.parse(selected.trainedAt)),
    `${scenario.id} selected run must have an ISO trainedAt timestamp`
  );
}

const browser = await chromium.launch({
  headless: true,
  args: ["--autoplay-policy=no-user-gesture-required"],
});
const receipt = {
  passed: false,
  checkedAt: new Date().toISOString(),
  baseUrl,
  liveCatalogSize: liveRuns.length,
  cases: {},
  pageErrors: [],
};

try {
  {
    const { context, page } = await newPage(browser);
    page.on("pageerror", (error) => receipt.pageErrors.push(error.message));
    try {
      await page.goto(`${baseUrl}/#experiments`, { waitUntil: "domcontentloaded" });
      receipt.cases.live = {
        rows: await inspectExperimentRows(page, expectedByScenario, {
          playVideos: true,
        }),
        desktopLayout: await layoutEvidence(page),
      };
      await page.locator("#experiments").screenshot({
        path: path.join(output, "trained-previews-desktop.png"),
      });
      await page.setViewportSize({ width: 390, height: 844 });
      receipt.cases.live.mobileLayout = await layoutEvidence(page);
      await page.locator("#experiments").screenshot({
        path: path.join(output, "trained-previews-mobile.png"),
      });
    } finally {
      await context.close();
    }
  }

  {
    const mockedRuns = [...newerIneligibleRuns(expectedByScenario), ...liveRuns];
    const { context, page } = await newPage(browser, mockedRuns);
    page.on("pageerror", (error) => receipt.pageErrors.push(error.message));
    try {
      await page.goto(`${baseUrl}/#experiments`, { waitUntil: "domcontentloaded" });
      const rows = await inspectExperimentRows(page, expectedByScenario, {
        playVideos: false,
      });
      for (const scenario of scenarios) {
        const row = await experimentRow(page, scenario.title);
        assert.equal(
          await row.getByText(`${scenario.id}-newer-failed`, { exact: true }).count(),
          0
        );
        assert.equal(
          await row.getByText(`${scenario.id}-newer-unverified`, { exact: true }).count(),
          0
        );
      }
      receipt.cases.newerIneligibleIgnored = { rows };
    } finally {
      await context.close();
    }
  }

  receipt.cases.reducedMotion = {
    rows: await verifyReducedMotion(browser, expectedByScenario),
  };

  {
    const missingScenario = scenarios.find((scenario) => scenario.preview === "task");
    const mockedRuns = [
      ...newerIneligibleRuns(expectedByScenario),
      ...liveRuns.filter((run) =>
        run.experimentId !== missingScenario.id || !isEligible(run, missingScenario)
      ),
    ];
    const { context, page } = await newPage(browser, mockedRuns);
    page.on("pageerror", (error) => receipt.pageErrors.push(error.message));
    try {
      await page.goto(`${baseUrl}/#experiments`, { waitUntil: "domcontentloaded" });
      const row = await experimentRow(page, missingScenario.title);
      await row.getByText("No verified rollout yet", { exact: true }).waitFor();
      assert.equal(
        await row.locator("video, img").count(),
        0,
        "Missing eligible candidate must not fall back to legacy GIF/video media"
      );
      assert.equal(
        await row.getByRole("link", { name: /Evaluation/i }).count(),
        0,
        "Missing eligible candidate must not expose a legacy Evaluation link"
      );
      receipt.cases.missingEligible = {
        experimentId: missingScenario.id,
        placeholder: "No verified rollout yet",
        mediaCount: await row.locator("video, img").count(),
      };
    } finally {
      await context.close();
    }
  }

  assert.deepEqual(receipt.pageErrors, []);
  receipt.passed = true;
} catch (error) {
  receipt.error = error instanceof Error
    ? `${error.name}: ${error.message}\n${error.stack ?? ""}`
    : String(error);
  throw error;
} finally {
  await writeFile(
    path.join(output, "verification.json"),
    `${JSON.stringify(receipt, null, 2)}\n`
  );
  await browser.close();
}

console.log(JSON.stringify({
  passed: receipt.passed,
  videosPlayed: receipt.cases.live?.rows.length ?? 0,
  output,
}));
