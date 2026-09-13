import assert from "node:assert/strict";
import { mkdir, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json")("playwright");
const studio = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const lab = process.env.LAB_ADDRESS || "127.0.0.1:8790";
const output = path.resolve(process.env.EVIDENCE_DIR || "docs/bridge-showcase/evidence/studio");
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1536, height: 1000 } });
await page.addInitScript(() => {
  localStorage.setItem("ducklab.camera", JSON.stringify({ p: [0.45, 1.8, 4.8], t: [0.45, 0.3, 0] }));
});
const errors = [];
const contextMessages = [];
page.on("pageerror", (error) => errors.push(error.message));
page.on("console", (message) => {
  if (/WebGL.*Context Lost/i.test(message.text())) contextMessages.push(message.text());
});
const report = { checkedAt: new Date().toISOString(), passed: false, errors, contextMessages, checks: {} };
try {
  await page.goto(`${studio}/?lab=${lab}`, { waitUntil: "networkidle" });
  const rows = page.locator('#experiments [role="radio"]');
  await rows.nth(6).waitFor();
  assert.equal(await rows.count(), 7);
  report.checks.cards = await rows.allTextContents();
  await page.locator("#experiments").screenshot({ path: path.join(output, "seven-cases.png") });
  for (const scenario of ["Basketball", "Suspended bridge"]) {
    const row = rows.filter({ hasText: scenario }).first();
    await row.click();
    await page.waitForTimeout(300);
    assert.equal(await row.getAttribute("aria-checked"), "true");
    await page.screenshot({ path: path.join(output, `${scenario.toLowerCase().replaceAll(" ", "-")}-desktop.png`), fullPage: false });
  }
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
      return { duck: duck.id, bodies: scene.bodies, tendons: scene.tendons?.length || 0, primitives: scene.primitives?.length || 0,
        ballVisible: scene.geoms.some((geom) => scene.bodies[geom.body] === "basketball"),
        advanced: first.find((candidate) => candidate.id === duck.id)?.step !== duck.step };
    }));
    return { first, last, scenes };
  }, lab);
  assert.ok(live.scenes.some((scene) => scene.bodies.includes("basketball") && scene.ballVisible));
  assert.ok(live.scenes.some((scene) => scene.bodies.includes("bridge_plank") && scene.tendons === 4 && scene.primitives >= 9));
  assert.ok(live.scenes.every((scene) => scene.advanced));
  report.checks.live = live;
  const canvas = page.locator("canvas").first();
  await canvas.scrollIntoViewIfNeeded();
  await page.evaluate((address) => new Promise((resolve, reject) => {
    const socket = new WebSocket(`ws://${address}/ws`);
    const timeout = setTimeout(() => { socket.close(); reject(new Error("screenshot reset was not observed")); }, 10000);
    socket.onopen = () => socket.send(JSON.stringify({ reset: true }));
    socket.onmessage = (event) => {
      const frame = JSON.parse(event.data);
      if (frame.ducks?.length && frame.ducks.every((duck) => duck.step < 10)) {
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
  report.passed = true;
} finally {
  await writeFile(path.join(output, "verification.json"), JSON.stringify(report, null, 2));
  await browser.close();
}
console.log(JSON.stringify({ passed: true, output }));
