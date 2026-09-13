import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const { chromium } = createRequire(process.env.PLAYWRIGHT_PACKAGE || "/Users/ghu/community/package.json")("playwright");
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const baseUrl = process.env.STUDIO_URL || "http://127.0.0.1:63317";
const output = path.resolve(process.env.ARM_VIEWER_EVIDENCE_DIR || path.join(root, "rlx/artifacts/arm-viewer-20260912"));
await mkdir(output, { recursive: true });
const receipt = { passed: false, checkedAt: new Date().toISOString(), baseUrl, videos: [], liveCases: [], pageErrors: [] };

async function inventory(directory) {
  const files = [];
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const filename = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...await inventory(filename));
    else if (entry.isFile() && entry.name.endsWith(".mp4")) files.push(filename);
  }
  return files;
}

const browser = await chromium.launch({ headless: true });
try {
  const page = await browser.newPage({ viewport: { width: 1536, height: 1100 }, deviceScaleFactor: 1 });
  page.on("pageerror", (error) => receipt.pageErrors.push(error.message));
  const response = await page.request.get(`${baseUrl}/api/arm/videos`);
  assert.equal(response.status(), 200);
  const library = await response.json();
  const files = await inventory(path.join(root, "rlx/runs/arm"));
  const hashes = new Set(await Promise.all(files.map(async (filename) => createHash("sha256").update(await readFile(filename)).digest("hex"))));
  assert.equal(library.totalFiles, files.length);
  assert.equal(library.uniqueVideos, hashes.size);
  assert.equal(library.videos.length, hashes.size);
  assert.equal(library.hardwareEnabled, false);
  assert.equal(new Set(library.videos.flatMap((video) => video.caseIds)).size, 6);
  receipt.totalFiles = files.length;
  receipt.uniqueVideos = hashes.size;

  await page.route("**/api/arm/health", (route) => route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ error: "Simulated offline service for independent-library test" }) }));
  await page.goto(`${baseUrl}/arm#videos`, { waitUntil: "domcontentloaded" });
  const libraryPanel = page.getByTestId("arm-video-library");
  await libraryPanel.waitFor();
  await page.waitForFunction((count) => document.querySelectorAll("[data-video-id]").length === count, library.uniqueVideos);
  assert.equal(await page.locator("canvas").count(), 0);
  receipt.offlineLibraryAvailable = true;
  async function showVideoFrame() {
    await page.waitForFunction(() => {
      const player = document.querySelector('[data-testid="arm-video-library"] video');
      return player && player.readyState >= 2 && player.videoWidth > 0;
    });
    await libraryPanel.locator("video").evaluate(async (element) => { element.muted = true; element.currentTime = 0; await element.play(); });
    await page.waitForFunction(() => document.querySelector('[data-testid="arm-video-library"] video')?.currentTime > 0.4);
    await libraryPanel.locator("video").evaluate((element) => element.pause());
  }
  await showVideoFrame();
  await page.screenshot({ path: path.join(output, "desktop-library.png"), fullPage: true });

  for (const video of library.videos) {
    await page.locator(`[data-video-id="${video.id}"]`).click();
    const player = libraryPanel.locator("video");
    await page.waitForFunction((url) => {
      const element = document.querySelector('[data-testid="arm-video-library"] video');
      return element && element.getAttribute("src") === url && element.readyState >= 2;
    }, video.videoUrl, { timeout: 30000 });
    await player.evaluate(async (element) => { element.muted = true; element.currentTime = 0; await element.play(); });
    await page.waitForFunction(() => document.querySelector('[data-testid="arm-video-library"] video')?.currentTime > 0.12);
    const media = await player.evaluate(async (element) => {
      element.pause();
      const target = Math.max(0, element.duration - 0.1);
      if (Math.abs(element.currentTime - target) > 0.02) {
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error("Seek timeout")), 10000);
          element.addEventListener("seeked", () => { clearTimeout(timeout); resolve(); }, { once: true });
          element.currentTime = target;
        });
      }
      return { duration: element.duration, width: element.videoWidth, height: element.videoHeight, mediaError: element.error?.message ?? null };
    });
    assert(media.width > 0 && media.height > 0 && media.duration > 0 && !media.mediaError);
    receipt.videos.push({ id: video.id, outcome: video.outcome, provenance: video.provenance, playedAndSeeked: true, ...media });
  }

  await page.getByLabel("Outcome", { exact: true }).selectOption("failed");
  const failedCount = library.videos.filter((video) => video.outcome === "failed").length;
  assert(failedCount > 0);
  assert.equal(await page.locator("[data-video-id]").count(), failedCount);
  await page.getByLabel("Search videos", { exact: true }).fill("no-such-arm-recording");
  await page.getByText("No matching videos", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Show all", exact: true }).click();
  assert.equal(await page.locator("[data-video-id]").count(), library.uniqueVideos);
  await page.getByLabel("Type", { exact: true }).selectOption("compilation");
  assert.equal(await page.locator("[data-video-id]").count(), library.videos.filter((video) => video.kind === "compilation").length);
  await page.getByRole("button", { name: "Show all", exact: true }).click();
  await page.getByLabel("Provenance", { exact: true }).selectOption("historical");
  assert.equal(await page.locator("[data-video-id]").count(), library.videos.filter((video) => video.provenance === "historical").length);
  await page.getByRole("button", { name: "Show all", exact: true }).click();
  const caseIds = [...new Set(library.videos.flatMap((video) => video.caseIds))];
  for (const caseId of caseIds) {
    await page.getByLabel("Case", { exact: true }).selectOption(caseId);
    assert.equal(await page.locator("[data-video-id]").count(), library.videos.filter((video) => video.caseIds.includes(caseId)).length);
  }
  await page.getByRole("button", { name: "Show all", exact: true }).click();
  const mobileVideo = library.videos.find((video) => video.provenance === "current" && video.caseIds.includes("arms-co-carry-v1") && video.selectionReasons.includes("first-new-seed"));
  assert(mobileVideo);
  await page.locator(`[data-video-id="${mobileVideo.id}"]`).click();
  await page.setViewportSize({ width: 390, height: 900 });
  await showVideoFrame();
  assert(await page.getByRole("link", { name: "Back to Microduck Studio" }).isVisible());
  await page.screenshot({ path: path.join(output, "mobile-library.png"), fullPage: true });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), "Mobile overflow");
  receipt.mobileOverflow = false;

  await page.unroute("**/api/arm/health");
  await page.setViewportSize({ width: 1536, height: 1100 });
  await page.getByRole("button", { name: "Live simulator", exact: true }).click();
  await page.getByRole("button", { name: "Reconnect", exact: true }).click();
  await page.getByText("Simulation online", { exact: true }).waitFor();
  await page.getByRole("button", { name: "Reset selected case", exact: true }).waitFor();
  for (const caseId of caseIds) {
    await page.getByRole("region", { name: "Arm experiment cases" }).getByRole("button").filter({ hasText: caseId }).click();
    const resetResponse = page.waitForResponse((reply) => reply.url().endsWith("/api/arm/reset") && reply.request().method() === "POST");
    await page.getByRole("button", { name: "Reset selected case", exact: true }).click();
    const state = await (await resetResponse).json();
    assert.equal(state.case_id, caseId);
    assert.equal(state.hardware_enabled, false);
    const teacherResponse = page.waitForResponse((reply) => reply.url().endsWith("/api/arm/teacher") && reply.request().method() === "POST");
    await page.getByRole("button", { name: "Advance Teacher", exact: true }).click();
    const stepped = await (await teacherResponse).json();
    assert.equal(stepped.case_id, caseId);
    assert(stepped.step > state.step);
    receipt.liveCases.push({ caseId, resetStep: state.step, teacherStep: stepped.step, hardwareEnabled: stepped.hardware_enabled });
  }
  await page.screenshot({ path: path.join(output, "desktop-live.png"), fullPage: true });
  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await page.getByRole("link", { name: "Arm videos", exact: true }).click();
  await page.getByTestId("arm-video-library").waitFor();
  receipt.studioNavigationPassed = true;
  assert.deepEqual(receipt.pageErrors, []);
  receipt.passed = true;
  console.log(JSON.stringify({ passed: true, allVideos: receipt.videos.length, liveCases: receipt.liveCases.length, totalFiles: receipt.totalFiles, output }, null, 2));
} catch (error) {
  receipt.error = error.stack || String(error);
  throw error;
} finally {
  await writeFile(path.join(output, "verification.json"), JSON.stringify(receipt, null, 2) + "\n");
  await browser.close();
}
