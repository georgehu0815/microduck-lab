import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json")("playwright");
const baseUrl = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const output = path.resolve(process.env.STUDIO_EVIDENCE_DIR || "../rlx/artifacts/studio-ui-20260908/browser");
await mkdir(output, { recursive: true });
const cases = [
  { scenario: "dance", title: "Dance imitation", run: "dance-e2e-20260907-low-noise", seconds: 8, stages: 1 },
  { scenario: "swing", title: "Self-pumped swing", run: "swing-e2e-20260907-v3", seconds: 24, stages: 2 },
  { scenario: "running", title: "Fast running", run: "running-e2e-20260907-v4", seconds: 12, stages: 2 },
  { scenario: "stilts", title: "Stilt walking", run: "stilts-e2e-20260907-v3", seconds: 10, stages: 2 },
];
const browser = await chromium.launch({ headless: true });
const receipt = { passed: false, checkedAt: new Date().toISOString(), cases: [], pageErrors: [], requestCaptureOnly: true };
let page;
try {
  page = await browser.newPage({ viewport: { width: 1600, height: 1100 }, deviceScaleFactor: 1.5 });
  page.on("pageerror", (error) => receipt.pageErrors.push(error.message));
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  const backflipRow = page.getByRole("radio").filter({ hasText: "Backflip showcase" });
  await backflipRow.waitFor();
  await backflipRow.getByText(/spotter-assisted launch → PPO landing → pretrained stand-policy handoff/).first().waitFor();
  const backflipRun = "backflip-e2e-20260908-v5";
  await backflipRow.getByText(backflipRun, { exact: true }).waitFor();
  await backflipRow.getByText("TRAINED · PASSED", { exact: true }).waitFor();
  const backflipVideo = backflipRow.locator("video");
  await backflipVideo.waitFor({ state: "visible" });
  await backflipVideo.evaluate(async (element) => { await element.play(); });
  await page.waitForFunction(() => {
    const row = [...document.querySelectorAll('[role="radio"]')]
      .find((element) => element.textContent?.includes("Backflip showcase"));
    const video = row?.querySelector("video");
    return Boolean(video && video.currentTime > .2);
  });
  const backflipPlayback = await backflipVideo.evaluate((element) => {
    element.pause();
    return {
      duration: element.duration,
      currentTime: element.currentTime,
      src: element.currentSrc,
      error: element.error?.message ?? null,
    };
  });
  assert.equal(backflipPlayback.error, null);
  assert.ok(backflipPlayback.currentTime > .2);
  assert.ok(Math.abs(backflipPlayback.duration - 12) < .03);
  const backflipVideoUrl = new URL(backflipPlayback.src);
  assert.equal(backflipVideoUrl.pathname, "/api/rlx/artifact");
  assert.equal(backflipVideoUrl.searchParams.get("experiment"), "backflip");
  assert.equal(backflipVideoUrl.searchParams.get("run"), backflipRun);
  assert.equal(backflipVideoUrl.searchParams.get("kind"), "video");
  assert.equal(backflipVideoUrl.searchParams.get("inline"), "1");
  const backflipFullVideoUrl = new URL(await backflipRow.getByRole("link", { name: "Full video ↗" }).getAttribute("href"), baseUrl);
  assert.equal(backflipFullVideoUrl.searchParams.get("run"), backflipRun);
  await backflipRow.getByRole("button", { name: "Show Backflip showcase guidance" }).click();
  const backflipGuide = page.getByRole("dialog");
  await backflipGuide.getByText(/spotter-launch-landing-stand-v2/).waitFor();
  await backflipGuide.getByText(/exactly transferring the pretrained stand actor and observation normalizer/).waitFor();
  await backflipGuide.getByText(/16 pretrained-teacher landing episodes of 600 steps \(9,600 calibration steps\)/).waitFor();
  await backflipGuide.getByText(/fixes the discounted-return RMS/).waitFor();
  await backflipGuide.getByText(/10 critic-only calibration epochs/).waitFor();
  await backflipGuide.getByText(/actor remains unchanged/).waitFor();
  await backflipGuide.getByText(/calibrated reward normalization is frozen/).waitFor();
  await backflipGuide.getByText(/fresh optimizer/).waitFor();
  await backflipGuide.getByText(/does not by itself establish positive PPO improvement/).waitFor();
  await backflipGuide.getByText(/5\.3 rad threshold/).waitFor();
  await backflipGuide.getByText(/3\.5 and 6\.5 rad\/s/).waitFor();
  await backflipGuide.getByText(/at least 5\.8 rad of credited rotation before handoff/).waitFor();
  await backflipGuide.getByText(/10 consecutive stable PPO landing steps/).waitFor();
  await backflipGuide.getByText(/no spotter, stand override, or body support/).waitFor();
  await backflipGuide.getByText(/stand-policy phase cannot finish credited rotation/).waitFor();
  await backflipGuide.getByRole("button", { name: "Close recipe guidance" }).click();
  receipt.backflipGuide = {
    catalogAvailable: true,
    verifiedPreviewPresent: true,
    runName: backflipRun,
    sourceBoundVideo: true,
    playback: backflipPlayback,
    protocol: "spotter-launch-landing-stand-v2",
    exactActorNormalizerTransfer: true,
    calibrationSteps: 9600,
    criticOnlyCalibrationEpochs: 10,
    actorUnchangedDuringCalibration: true,
    rewardNormalizationFrozenAfterCalibration: true,
    freshPpoOptimizer: true,
    positivePpoImprovementClaimed: false,
    launchReleaseThresholdRad: 5.3,
    launchReleaseRateRadS: [3.5, 6.5],
    minRotationBeforeHandoffRad: 5.8,
    consecutiveStableLandingSteps: 10,
    bodySupportForbidden: true,
    standCannotFinishRotation: true,
  };
  await page.getByRole("region", { name: "Saved scenario evidence" }).getByRole("button", { name: "Review Dance", exact: true }).waitFor();
  for (const scenario of cases) {
    await page.getByRole("radio").filter({ hasText: scenario.title }).click();
    await page.getByLabel("Saved runs", { exact: true }).selectOption(scenario.run);
    const panel = page.locator("#evaluation");
    await panel.getByText("Accepted", { exact: true }).waitFor({ timeout: 60000 });
    assert.equal(await page.getByLabel("Run name", { exact: true }).inputValue(), scenario.run);
    const response = await page.request.get(`${baseUrl}/api/rlx?experiment=${scenario.scenario}&run=${scenario.run}`);
    assert.equal(response.status(), 200);
    const snapshot = await response.json();
    assert.equal(snapshot.evaluation.passed, true);
    assert.equal(snapshot.renderVerified, true);
    assert.equal(snapshot.savedRecipe.runName, scenario.run);
    assert.ok(snapshot.rewardHistory.length > 240 || scenario.scenario === "swing");
    assert.ok(snapshot.trainingHistory.segments.length >= scenario.stages);
    assert.ok(snapshot.trainingHistory.segments.filter((segment) => segment.kind === "ppo").every((segment) => segment.collections.length > 0), "Recorded ancestor reward samples must be recovered, not replaced by empty metadata stages");
    assert.equal(snapshot.rewardHistory.at(-1).step, snapshot.trainingSteps);
    const lifecycle = page.getByRole("region", { name: "Full training lifecycle" });
    await lifecycle.waitFor();
    assert.ok(await lifecycle.getByRole("img", { name: /Policy loss/ }).count() > 0);
    assert.ok(await lifecycle.getByRole("img", { name: /Value loss/ }).count() > 0);
    await lifecycle.screenshot({ path: path.join(output, `${scenario.scenario}-lifecycle.png`) });
    await panel.getByText("16 / 16 episodes passed", { exact: true }).waitFor();
    const video = panel.locator("video");
    await video.scrollIntoViewIfNeeded();
    await video.evaluate(async (element) => { await element.play(); });
    await page.waitForFunction((seconds) => {
      const element = document.querySelector("#evaluation video");
      return element && element.currentTime > .5 && Math.abs(element.duration - seconds) < .03;
    }, scenario.seconds);
    await video.evaluate((element) => element.pause());
    const playback = await video.evaluate((element) => ({ duration: element.duration, currentTime: element.currentTime, src: element.currentSrc, error: element.error?.message ?? null }));
    assert.equal(playback.error, null);
    assert.ok(playback.src.includes(scenario.run));
    await panel.screenshot({ path: path.join(output, `${scenario.scenario}-evaluation.png`) });
    const onnxArtifact = page.locator("#artifacts a").filter({ hasText: "ONNX policy" });
    assert.equal(await onnxArtifact.getAttribute("aria-disabled"), "true");
    await panel.getByLabel("I reviewed the rendered motion", { exact: false }).check();
    assert.equal(await onnxArtifact.getAttribute("aria-disabled"), "false");
    await page.getByLabel("Run name", { exact: true }).fill("missing-evidence-regression");
    assert.equal(await panel.getByText("Accepted", { exact: true }).count(), 0, "Previous run verdict must disappear immediately");
    assert.equal(await panel.locator("video").count(), 0, "Previous run video must disappear immediately");
    assert.equal(await onnxArtifact.getAttribute("aria-disabled"), "true");
    await page.getByLabel("Saved runs", { exact: true }).selectOption(scenario.run);
    await panel.getByText("Accepted", { exact: true }).waitFor();
    assert.equal(await panel.getByLabel("I reviewed the rendered motion", { exact: false }).isChecked(), false);
    await page.setViewportSize({ width: 390, height: 844 });
    const layout = await panel.evaluate((element) => ({ width: element.clientWidth, scrollWidth: element.scrollWidth }));
    assert.ok(layout.scrollWidth <= layout.width + 1, `${scenario.scenario} mobile evaluation overflow: ${JSON.stringify(layout)}`);
    await panel.screenshot({ path: path.join(output, `${scenario.scenario}-mobile.png`) });
    await page.setViewportSize({ width: 1600, height: 1100 });
    receipt.cases.push({ ...scenario, playback, layout, samples: snapshot.rewardHistory.length, transitions: snapshot.trainingSteps, stages: snapshot.trainingHistory.segments.map(({ id, kind, startStep, endStep }) => ({ id, kind, startStep, endStep })) });
  }
  await page.getByRole("radio").filter({ hasText: "Dance imitation" }).click();
  await page.getByRole("button", { name: /Default full/ }).click();
  const clipSelect = page.getByLabel("Dance reference clip", { exact: true });
  const clipPath = "dance-clip/cumbia_microduck_v2.clip.json";
  await clipSelect.selectOption(clipPath);
  await page.getByLabel("Run name", { exact: true }).fill("dance-clip-payload-regression");
  let requestPayload;
  await page.route("**/api/rlx", async (route) => {
    if (route.request().method() !== "POST") return route.continue();
    requestPayload = route.request().postDataJSON();
    await route.fulfill({ status: 409, contentType: "application/json", body: JSON.stringify({ error: "Regression test captured request without launching training." }) });
  });
  await page.getByRole("button", { name: "Start RLX", exact: true }).click();
  await page.getByRole("status").filter({ hasText: "Regression test captured request" }).first().waitFor();
  assert.equal(requestPayload.recipe.danceClip, clipPath);
  assert.equal(requestPayload.recipe.maxEpisodeS, 60);
  assert.equal(requestPayload.recipe.evalSteps, 3000);
  assert.equal(requestPayload.recipe.renderSeconds, 60);
  assert.equal(requestPayload.action, "train");
  receipt.clipRequest = requestPayload;
  await page.unroute("**/api/rlx");
  await page.route("**/api/rlx", async (route) => {
    if (route.request().method() !== "POST") return route.continue();
    const payload = route.request().postDataJSON();
    await route.fulfill({ status: 202, contentType: "application/json", body: JSON.stringify({ accepted: true, recipe: { ...payload.recipe, runName: "canonical-run-regression" } }) });
  });
  await page.getByLabel("Run name", { exact: true }).fill("Mixed Name");
  await page.getByRole("button", { name: "Start RLX", exact: true }).click();
  await page.waitForFunction(() => document.querySelector('#recipe input[pattern]')?.value === "canonical-run-regression");
  receipt.canonicalRecipeApplied = true;
  assert.equal(receipt.pageErrors.length, 0);
  receipt.passed = true;
} catch (error) {
  receipt.error = String(error);
  if (page) {
    await page.screenshot({ path: path.join(output, "failure.png"), fullPage: true }).catch(() => {});
    await writeFile(path.join(output, "failure.html"), await page.content());
  }
  throw error;
} finally {
  await writeFile(path.join(output, "verification.json"), JSON.stringify(receipt, null, 2) + "\n");
  await browser.close();
}
console.log(JSON.stringify({ passed: receipt.passed, cases: receipt.cases.length, output }));
