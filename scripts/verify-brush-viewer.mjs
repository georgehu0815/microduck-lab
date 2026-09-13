import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { mkdir, readFile, writeFile } from "node:fs/promises";
const { chromium } = createRequire(process.env.PLAYWRIGHT_REQUIRE_ROOT || "/Users/ghu/community/package.json")("playwright");
const studio = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const lab = process.env.BRUSH_LAB || "127.0.0.1:8792";
const run = process.env.BRUSH_RUN || "brush-color-v2-03";
const output = "docs/drawing-case/brush/evidence";
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const deadline = setTimeout(() => void browser.close(), 300_000);
const page = await browser.newPage({ viewport: { width: 1536, height: 1100 } });
const errors = [];
page.on("pageerror", error => errors.push(error.message));
page.setDefaultTimeout(90_000);
const report = { checkedAt: new Date().toISOString(), run, passed: false,
  scope: "Bounded eight-case live UI/contact-color stream smoke; the complete four-color painting is verified separately from the source-bound full render and 40-episode ONNX gate." };
try {
  await page.addInitScript(() => {
    localStorage.setItem("ducklab.hudOpen", "false");
    localStorage.setItem("ducklab.camera", JSON.stringify({ p: [-.28, .38, -5.65], t: [.1, .18, -5.2] }));
  });
  await page.goto(`${studio}/?lab=${lab}`, { waitUntil: "domcontentloaded" });
  const cards = page.locator('#experiments [role="radio"]');
  await cards.nth(7).waitFor();
  report.catalog = {
    cards: await cards.count(),
  };
  assert.equal(report.catalog.cards, 8);
  console.log("Eight scenario cards ready");
  await page.locator("#experiments").scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/viewer-cards.png` });
  await cards.filter({ hasText: "Swimming-duck drawing" }).click();
  await page.getByLabel("Saved runs", { exact: true }).selectOption(run);
  await page.locator('[data-duck-id="d7"]').waitFor({ state: "attached" });
  await page.locator("[data-duck-id]").nth(7).waitFor({ state: "attached" });
  assert.equal(await page.locator("[data-duck-id]").count(), 8);
  console.log("Eight live ducks mounted");
  await page.getByText("61 / 93 observations", { exact: true }).waitFor();
  report.live = await page.evaluate(address => new Promise((resolve, reject) => {
    const socket = new WebSocket(`ws://${address}/ws`);
    const timer = setTimeout(() => { socket.close(); reject(new Error("live brush paint stream did not advance")); }, 180_000);
    socket.onopen = () => socket.send(JSON.stringify({ reset: true }));
    socket.onerror = () => { clearTimeout(timer); reject(new Error("live socket error")); };
    socket.onmessage = event => {
      const frame = JSON.parse(event.data);
      const duck = frame.ducks?.find(item => item.id === "d7");
      if (!duck?.drawing || duck.step < 600 || duck.drawing.points.length < 200) return;
      clearTimeout(timer);
      socket.close();
      resolve({ ducks: frame.ducks.length, step: duck.step, falls: duck.falls,
        contract: duck.drawing.contract, points: duck.drawing.points.length,
        colors: [...new Set(duck.drawing.colors)], palette: duck.drawing.palette,
        brushRadius: duck.drawing.brush_radius, assistance: duck.drawing.assistance });
    };
  }), lab);
  assert.equal(report.live.ducks, 8);
  assert.equal(report.live.contract, "microduck-brush-v2");
  assert.equal(report.live.assistance, 0);
  assert.ok(report.live.points >= 200);
  assert.equal(report.live.palette.length, 4);
  assert.ok(report.live.palette.every(color =>
    color.length === 3 && color.every(channel => Number.isFinite(channel) && channel >= 0 && channel <= 1)));
  assert.ok(Number.isFinite(report.live.brushRadius) && report.live.brushRadius > 0);
  assert.ok(report.live.colors.includes(0));
  assert.ok(report.live.colors.every(color => Number.isInteger(color) && color >= 0 && color < 4));
  console.log("Physical brush color stream verified");
  const raw = JSON.parse(await readFile(`rlx/runs/studio/drawing/${run}/render/raw_trace.json`, "utf8"));
  const full = JSON.parse(await readFile(`rlx/runs/studio/drawing/${run}/render/render.json`, "utf8"));
  const evaluation = JSON.parse(await readFile(`rlx/runs/studio/drawing/${run}/eval.json`, "utf8"));
  report.fullRender = { colors: [...new Set(raw.rows.map(row => row[7]))].sort(), seconds: full.elapsed_seconds,
    passed: full.assessment.passed, unassisted: full.assessment.unassisted,
    paintSource: full.paint_source, sourceSha256: full.source_sha256,
    evaluationSourceSha256: evaluation.source_sha256 };
  assert.deepEqual(report.fullRender.colors, [0, 1, 2, 3]);
  assert.equal(report.fullRender.passed, true);
  assert.equal(report.fullRender.unassisted, true);
  assert.equal(report.fullRender.paintSource, "actual MuJoCo brush-tip contact trace only");
  assert.equal(report.fullRender.sourceSha256, report.fullRender.evaluationSourceSha256);
  const canvas = page.locator("canvas").first();
  await canvas.scrollIntoViewIfNeeded();
  await page.screenshot({ path: `${output}/viewer-live.png` });
  const apiResponse = await fetch(`${studio}/api/rlx?experiment=drawing&run=${run}`, {
    signal: AbortSignal.timeout(30_000),
  });
  assert.equal(apiResponse.status, 200);
  const snapshot = await apiResponse.json();
  report.apiSnapshot = {
      runName: snapshot.runName,
      drawingTool: snapshot.drawingTool,
      renderVerified: snapshot.renderVerified,
      acceptedEpisodes: snapshot.evaluation?.drawing_assessment?.accepted_episodes,
      requiredEpisodes: snapshot.evaluation?.drawing_assessment?.required_episodes,
      provenanceErrors: snapshot.evaluation?.provenance_errors,
      sourceSha256: snapshot.evaluation?.source_sha256,
  };
  assert.deepEqual(report.apiSnapshot, {
    runName: run,
    drawingTool: "brush",
    renderVerified: true,
    acceptedEpisodes: 40,
    requiredEpisodes: 40,
    provenanceErrors: [],
    sourceSha256: report.fullRender.sourceSha256,
  });
  console.log("Source-bound Studio API verified");
  await page.reload({ waitUntil: "domcontentloaded" });
  await cards.filter({ hasText: "Swimming-duck drawing" }).click();
  await page.getByLabel("Saved runs", { exact: true }).selectOption(run);
  const reviewAction = page.getByRole("link", { name: "Review result", exact: true });
  await reviewAction.waitFor();
  report.reviewAction = await reviewAction.evaluate(element => ({
    tagName: element.tagName,
    href: element.getAttribute("href"),
  }));
  assert.deepEqual(report.reviewAction, { tagName: "A", href: "#evaluation" });
  await reviewAction.click();
  const evaluationGate = page.locator("#evaluation");
  await evaluationGate.getByText(`40 / 40 episodes passed`, { exact: false }).waitFor();
  const rolloutPlayer = evaluationGate.locator('section[aria-labelledby="rollout-player-title"]');
  await rolloutPlayer.getByRole("heading", { name: `${run} rollout`, exact: true }).waitFor();
  const video = rolloutPlayer.locator("video");
  await video.waitFor();
  await video.evaluate(element => new Promise((resolve, reject) => {
    if (element.readyState >= HTMLMediaElement.HAVE_METADATA) {
      resolve();
      return;
    }
    element.addEventListener("loadedmetadata", resolve, { once: true });
    element.addEventListener("error", () => reject(new Error("rollout video metadata failed to load")), { once: true });
  }));
  await video.evaluate(element => { element.muted = true; return element.play(); });
  const before = await video.evaluate(element => element.currentTime);
  await page.waitForTimeout(1200);
  const after = await video.evaluate(element => element.currentTime);
  assert.ok(after > before);
  report.video = {
    before,
    after,
    duration: await video.evaluate(element => element.duration),
    paused: await video.evaluate(element => element.paused),
    src: await video.getAttribute("src"),
  };
  assert.ok(report.video.src.includes(`source=${encodeURIComponent(report.fullRender.sourceSha256)}`));
  console.log("Selected brush rollout playback verified");
  await page.screenshot({ path: `${output}/viewer-video.png` });
  await page.keyboard.press("Escape");
  await page.setViewportSize({ width: 390, height: 900 });
  await page.evaluate(() => scrollTo(0, 0));
  await page.waitForTimeout(1000);
  report.mobile = await page.evaluate(() => ({
    width: innerWidth,
    scrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
  }));
  assert.ok(report.mobile.scrollWidth <= report.mobile.width);
  assert.ok(report.mobile.bodyScrollWidth <= report.mobile.width);
  await page.screenshot({ path: `${output}/viewer-mobile.png` });
  assert.deepEqual(errors, []);
  report.passed = true;
} catch (error) {
  report.failure = error instanceof Error ? `${error.name}: ${error.message}` : String(error);
  throw error;
} finally {
  clearTimeout(deadline);
  report.errors = errors;
  await writeFile(`${output}/viewer.json`, `${JSON.stringify(report, null, 2)}\n`);
  await browser.close();
}
console.log(JSON.stringify(report, null, 2));
