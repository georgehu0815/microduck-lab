import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json")("playwright");
const studio = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const lab = process.env.LAB_ADDRESS || "127.0.0.1:8790";
const output = path.resolve(process.env.EVIDENCE_DIR || "docs/bridge-showcase/evidence/studio");
const cases = [
  { id: "dance", title: "Dance imitation", run: "dance-e2e-20260907-low-noise" },
  { id: "swing", title: "Self-pumped swing", run: "swing-e2e-20260907-v3" },
  { id: "running", title: "Fast running", run: "running-e2e-20260907-v4" },
  { id: "stilts", title: "Stilt walking", run: "stilts-e2e-20260907-v3" },
  { id: "backflip", title: "Backflip showcase", run: "backflip-e2e-20260908-v5" },
  { id: "basketball", title: "Basketball balancing", run: "basketball-balance-01" },
  { id: "bridge", title: "Suspended bridge", run: "bridge-studio-02" },
];
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1536, height: 1000 } });
page.setDefaultTimeout(30000);
await page.addInitScript(() => {
  localStorage.setItem("ducklab.camera", JSON.stringify({ p: [0, 5, 1.8], t: [0, 0.3, -2.6] }));
});
const errors = [];
const contextMessages = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (/WebGL.*Context Lost/i.test(message.text())) contextMessages.push(message.text());
});
const liveOnly = process.argv.includes("--live-only");
const report = { checkedAt: new Date().toISOString(), liveOnly, passed: false, errors, contextMessages, checks: {} };
try {
  await page.goto(studio, { waitUntil: "domcontentloaded" });
  await page.addStyleTag({ content: 'html, body, * { scroll-behavior: auto !important; }' });
  const rows = page.locator('#experiments [role="radio"]');
  await rows.nth(6).waitFor();
  assert.ok(await rows.count() >= cases.length, "catalog must retain all seven baseline cases");
  report.checks.cards = await rows.allTextContents();
  await page.locator("#experiments").screenshot({ path: path.join(output, "seven-cases.png") });
  report.checks.scenarios = [];
  for (const scenario of liveOnly ? [] : cases) {
    console.log(`${scenario.id}: checking controls - verify-seven-cases.mjs:44`);
    const row = rows.filter({ hasText: scenario.title }).first();
    await row.click();
    assert.equal(await row.getAttribute("aria-checked"), "true");
    await row.getByRole("button", { name: `Show ${scenario.title} guidance` }).click();
    await page.getByRole("dialog").waitFor();
    await page.getByRole("button", { name: "Close recipe guidance" }).click();
    await page.getByLabel("Saved runs", { exact: true }).selectOption(scenario.run);
    await page.waitForFunction((runName) => document.querySelector('input[aria-label="Run name"]')?.value === runName ||
      [...document.querySelectorAll("label")].some((label) => label.textContent?.trim() === "Run name" &&
        (label.control?.value === runName || label.querySelector("input")?.value === runName)), scenario.run);
    assert.equal(await page.getByLabel("Run name", { exact: true }).inputValue(), scenario.run);
    const response = await page.request.get(`${studio}/api/rlx?experiment=${scenario.id}&run=${scenario.run}`);
    assert.equal(response.status(), 200);
    const snapshot = await response.json();
    assert.equal(snapshot.artifacts.onnx, true, `${scenario.id}: ONNX missing`);
    assert.equal(snapshot.renderVerified, true, `${scenario.id}: stale render`);
    assert.equal(snapshot.evaluation.passed, !["basketball", "bridge"].includes(scenario.id));
    report.checks.scenarios.push({ ...scenario, artifacts: snapshot.artifacts,
      taskPassed: snapshot.evaluation.passed, renderVerified: snapshot.renderVerified });
    console.log(`${scenario.id}: selection, guidance, saved run, evaluation and artifacts passed - verify-seven-cases.mjs:64`);
  }
  await page.goto(`${studio}/?lab=${lab}`, { waitUntil: "domcontentloaded" });
  const live = await page.evaluate(async (address) => {
    const frames = await new Promise((resolve, reject) => {
      const socket = new WebSocket(`ws://${address}/ws`);
      const received = [];
      const timeout = setTimeout(() => { socket.close(); reject(new Error("no live frames")); }, 15000);
      socket.onerror = () => { clearTimeout(timeout); reject(new Error("lab websocket failed")); };
      socket.onmessage = (event) => {
        const frame = JSON.parse(event.data);
        if (!frame.ducks) return;
        received.push(frame.ducks);
        if (received.length >= 30) {
          clearTimeout(timeout);
          socket.close();
          resolve(received);
        }
      };
    });
    const first = frames[0];
    const last = frames.at(-1);
    const scenes = await Promise.all(last.map(async (duck) => {
      const response = await fetch(`http://${address}/scene?duck=${encodeURIComponent(duck.id)}&version=${encodeURIComponent(duck.sceneKey)}`);
      if (!response.ok) throw new Error(`scene status ${response.status}`);
      const scene = await response.json();
      return { duck: duck.id, name: duck.name, policy: duck.policy, bodies: scene.bodies, tendons: scene.tendons?.length || 0, primitives: scene.primitives?.length || 0,
        ballVisible: scene.geoms.some((geom) => scene.bodies[geom.body] === "basketball"),
        advanced: first.find((candidate) => candidate.id === duck.id)?.step !== duck.step };
    }));
    return { first, last, scenes };
  }, lab);
  report.checks.live = live;
  for (const scenario of cases) {
    assert.ok(live.scenes.some((scene) => scene.name === scenario.id || scene.name.includes(scenario.run) || scene.policy?.includes(scenario.run)), `${scenario.id}: live policy missing`);
  }
  assert.ok(live.scenes.some((scene) => scene.bodies.includes("basketball") && scene.ballVisible));
  assert.ok(live.scenes.some((scene) => scene.bodies.includes("bridge_plank") && scene.tendons === 4 && scene.primitives >= 9));
  assert.ok(live.scenes.every((scene) => scene.advanced));
  report.checks.live = live;
  const canvas = page.locator("canvas").first();
  await canvas.scrollIntoViewIfNeeded();
  await page.waitForFunction(() => document.querySelectorAll('[data-duck-id]').length === 7, null, { timeout: 90000 });
  report.checks.renderedDucks = await page.locator('[data-duck-id]').evaluateAll((labels) => labels.map((label) => ({ id: label.dataset.duckId, name: label.textContent })));
  await page.waitForTimeout(1000);
  await page.evaluate((address) => new Promise((resolve, reject) => {
    const socket = new WebSocket(`ws://${address}/ws`);
    const timeout = setTimeout(() => { socket.close(); reject(new Error("screenshot reset was not observed")); }, 30000);
    socket.onopen = () => socket.send(JSON.stringify({ reset: true }));
    socket.onmessage = (event) => {
      const frame = JSON.parse(event.data);
      if (frame.ducks?.length === 7 && frame.ducks.every((duck) => duck.step < 100 && duck.falls === 0)) {
        clearTimeout(timeout);
        socket.close();
        resolve(true);
      }
    };
  }), lab);
  report.checks.screenshotReset = true;
  await page.waitForTimeout(100);
  await canvas.screenshot({ path: path.join(output, "live-scenes-desktop.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(300);
  const dimensions = await page.evaluate(() => ({ width: innerWidth, scroll: document.documentElement.scrollWidth }));
  assert.ok(dimensions.scroll <= dimensions.width + 1, JSON.stringify(dimensions));
  report.checks.mobile = dimensions;
  await page.locator("#experiments").screenshot({ path: path.join(output, "seven-cases-mobile.png") });
  report.checks.webgl = await page.evaluate(() => Array.from(document.querySelectorAll("canvas"), (canvas) => {
    const context = canvas.getContext("webgl2") || canvas.getContext("webgl");
    return context ? { lost: context.isContextLost(), width: canvas.width, height: canvas.height } : null;
  }).filter(Boolean));
  assert.ok(report.checks.webgl.length > 0);
  assert.ok(report.checks.webgl.every((context) => !context.lost && context.width > 0 && context.height > 0));
  assert.deepEqual(errors, []);
  assert.deepEqual(contextMessages, []);
  report.passed = true;
} finally {
  await writeFile(path.join(output, "verification.json"), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify({ passed: true, output }));
